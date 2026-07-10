from __future__ import annotations

import torch
from torch import nn

from model.MSHNet import MSHNet
from model.dea_scale_interaction_exchange import (
    ScaleInteractionExchangeMSHNet,
    decoder_mobius_decomposition,
)


def _forward_with_decoder_features(model, x):
    """Capture native decoder outputs without importing analysis utilities."""

    captured = {}
    handles = []
    for stage in (0, 1, 2, 3):
        def capture(_module, _inputs, output, *, key=stage):
            captured[key] = output

        handles.append(
            getattr(model, "decoder_%d" % stage).register_forward_hook(capture)
        )
    try:
        masks, pred = model(x, True)
    finally:
        for handle in handles:
            handle.remove()

    scale_logits = torch.cat(
        [
            masks[0],
            model.up(masks[1]),
            model.up_4(masks[2]),
            model.up_8(masks[3]),
        ],
        dim=1,
    )
    return {
        "masks": tuple(masks),
        "pred": pred,
        "scale_logits": scale_logits,
        "decoder_features": tuple(captured[stage] for stage in (0, 1, 2, 3)),
    }


def test_sied_decomposition_identity() -> None:
    torch.manual_seed(211)
    decoder = nn.Sequential(
        nn.Conv2d(7, 6, kernel_size=3, padding=1),
        nn.Tanh(),
        nn.Conv2d(6, 5, kernel_size=1),
    ).eval()
    current = torch.randn(2, 3, 9, 11)
    inherited = torch.randn(2, 4, 9, 11)

    for anchor_mode in ("zero", "mean"):
        terms = decoder_mobius_decomposition(
            decoder,
            current,
            inherited,
            anchor_mode=anchor_mode,
        )
        reconstructed = (
            terms["q00"]
            + terms["current_main"]
            + terms["inherited_main"]
            + terms["interaction"]
        )

        assert torch.allclose(
            reconstructed, terms["q11"], atol=2e-6, rtol=2e-6
        )
        assert terms["interaction"].abs().sum().item() > 0.0


def test_sied_alpha_zero_is_bitwise_mshnet() -> None:
    torch.manual_seed(223)
    baseline = MSHNet(3).eval()
    sied = ScaleInteractionExchangeMSHNet(
        3,
        alpha=0.0,
        active_stages=(0, 1, 2, 3),
        anchor_mode="mean",
    ).eval()
    sied.load_state_dict(baseline.state_dict(), strict=True)
    x = torch.randn(1, 3, 32, 48)

    counts = {stage: 0 for stage in (0, 1, 2, 3)}
    handles = []
    for stage in counts:
        def count_call(_module, _inputs, *, key=stage):
            counts[key] += 1

        handles.append(
            getattr(sied, "decoder_%d" % stage).register_forward_pre_hook(
                count_call
            )
        )

    try:
        with torch.no_grad():
            baseline_output = _forward_with_decoder_features(baseline, x)
            sied_output = _forward_with_decoder_features(sied, x)
    finally:
        for handle in handles:
            handle.remove()

    assert counts == {0: 1, 1: 1, 2: 1, 3: 1}
    assert torch.equal(sied_output["pred"], baseline_output["pred"])
    assert torch.equal(
        sied_output["scale_logits"], baseline_output["scale_logits"]
    )
    for actual, expected in zip(
        sied_output["masks"], baseline_output["masks"]
    ):
        assert torch.equal(actual, expected)
    for actual, expected in zip(
        sied_output["decoder_features"],
        baseline_output["decoder_features"],
    ):
        assert torch.equal(actual, expected)

    # ``return_dict=True`` uses the diagnostic implementation rather than
    # delegating to ``MSHNet.forward``.  It must still be an exact one-call
    # baseline path, not merely numerically close.
    dict_counts = {stage: 0 for stage in (0, 1, 2, 3)}
    handles = []
    for stage in dict_counts:
        def count_dict_call(_module, _inputs, *, key=stage):
            dict_counts[key] += 1

        handles.append(
            getattr(sied, "decoder_%d" % stage).register_forward_pre_hook(
                count_dict_call
            )
        )
    try:
        with torch.no_grad():
            sied_dict = sied(x, True, return_dict=True)
    finally:
        for handle in handles:
            handle.remove()

    assert dict_counts == {0: 1, 1: 1, 2: 1, 3: 1}
    assert sied_dict["sied"]["hard_baseline"] is True
    assert torch.equal(sied_dict["pred"], baseline_output["pred"])
    assert torch.equal(
        sied_dict["scale_logits_full"], baseline_output["scale_logits"]
    )
    for actual, expected in zip(sied_dict["masks"], baseline_output["masks"]):
        assert torch.equal(actual, expected)
    for actual, expected in zip(
        sied_dict["decoder_features"], baseline_output["decoder_features"]
    ):
        assert torch.equal(actual, expected)

    assert not any("alpha" in name for name, _ in sied.named_parameters())


def test_sied_affine_decoder_has_zero_interaction() -> None:
    torch.manual_seed(227)
    decoder = nn.Conv2d(7, 5, kernel_size=1, bias=True).double().eval()
    current = torch.randn(2, 3, 7, 9, dtype=torch.float64)
    inherited = torch.randn(2, 4, 7, 9, dtype=torch.float64)

    for anchor_mode in ("zero", "mean"):
        terms = decoder_mobius_decomposition(
            decoder,
            current,
            inherited,
            anchor_mode=anchor_mode,
        )
        assert torch.allclose(
            terms["interaction"],
            torch.zeros_like(terms["interaction"]),
            atol=1e-12,
            rtol=1e-12,
        )


def test_sied_counterfactual_branches_do_not_update_bn() -> None:
    torch.manual_seed(229)
    model = ScaleInteractionExchangeMSHNet(
        3,
        alpha=0.1,
        active_stages=(0,),
        freeze_bn_statistics=True,
    ).train()
    batch_norms = {
        name: module
        for name, module in model.named_modules()
        if isinstance(module, nn.modules.batchnorm._BatchNorm)
    }
    assert batch_norms
    before = {
        name: (
            module.running_mean.detach().clone(),
            module.running_var.detach().clone(),
            module.num_batches_tracked.detach().clone(),
        )
        for name, module in batch_norms.items()
    }

    output = model(
        torch.randn(2, 3, 32, 32),
        True,
        return_dict=True,
    )

    assert torch.isfinite(output["pred"]).all()
    assert all(not module.training for module in batch_norms.values())
    for name, module in batch_norms.items():
        actual = (
            module.running_mean,
            module.running_var,
            module.num_batches_tracked,
        )
        for actual_buffer, expected_buffer in zip(actual, before[name]):
            assert torch.equal(actual_buffer, expected_buffer), name


def test_sied_active_path_has_decoder_gradients() -> None:
    torch.manual_seed(233)
    model = ScaleInteractionExchangeMSHNet(
        3,
        alpha=0.1,
        active_stages=(0,),
        anchor_mode="mean",
        freeze_bn_statistics=True,
    ).train()
    x = torch.randn(2, 3, 32, 32, requires_grad=True)

    output = model(x, True, return_dict=True)
    terms = output["sied"]["stage_terms"][0]
    loss = output["pred"].square().mean()
    loss.backward()

    assert terms["interaction"].grad_fn is not None
    assert terms["exchange"].grad_fn is not None
    assert x.grad is not None and torch.isfinite(x.grad).all()
    decoder_grads = [
        parameter.grad
        for name, parameter in model.named_parameters()
        if name.startswith("decoder_0.")
    ]
    assert decoder_grads
    assert all(gradient is not None for gradient in decoder_grads)
    assert all(torch.isfinite(gradient).all() for gradient in decoder_grads)
    assert sum(gradient.abs().sum().item() for gradient in decoder_grads) > 0.0


def test_sied_only_evaluates_coalitions_at_active_stage() -> None:
    torch.manual_seed(239)
    model = ScaleInteractionExchangeMSHNet(
        3,
        alpha=0.05,
        active_stages=(0,),
    ).eval()
    counts = {stage: 0 for stage in (0, 1, 2, 3)}
    handles = []
    for stage in counts:
        def count_call(_module, _inputs, *, key=stage):
            counts[key] += 1

        handles.append(
            getattr(model, "decoder_%d" % stage).register_forward_pre_hook(
                count_call
            )
        )

    try:
        with torch.no_grad():
            output = model(
                torch.randn(1, 3, 32, 32),
                True,
                return_dict=True,
            )
    finally:
        for handle in handles:
            handle.remove()

    assert counts == {0: 4, 1: 1, 2: 1, 3: 1}
    assert output["sied"]["active_stages"] == (0,)
    assert set(output["sied"]["stage_terms"][0]) >= {
        "q11",
        "q10",
        "q01",
        "q00",
        "current_main",
        "inherited_main",
        "interaction",
        "exchange",
        "state",
    }
