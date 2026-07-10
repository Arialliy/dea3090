from __future__ import annotations

import torch
from torch import nn

from model.MSHNet import MSHNet
from model.dea_shared_discrepancy_stencil import (
    STENCIL_DELTAS,
    SharedDiscrepancyStencil,
    SharedDiscrepancyStencilMSHNet,
)


def _load_baseline(baseline: MSHNet, model: SharedDiscrepancyStencilMSHNet):
    incompatible = model.load_state_dict(baseline.state_dict(), strict=False)
    assert incompatible.missing_keys == ["stencil.theta"]
    assert incompatible.unexpected_keys == []


def _capture_native(model, x):
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
    return tuple(masks), pred, tuple(captured[stage] for stage in (0, 1, 2, 3))


def _set_bn_eval(module):
    for child in module.modules():
        if isinstance(child, nn.modules.batchnorm._BatchNorm):
            child.eval()


def test_shared_stencil_is_dc_null_and_uses_replicate_boundary() -> None:
    stencil = SharedDiscrepancyStencil()
    with torch.no_grad():
        stencil.theta[STENCIL_DELTAS.index((0, -1))] = 1.0

    constant = torch.full((2, 3, 5, 7), 4.25)
    assert torch.count_nonzero(stencil(constant)) == 0

    impulse = torch.zeros(1, 1, 4, 5)
    impulse[0, 0, 0, 0] = 1.0
    correction = stencil(impulse)
    assert correction[0, 0, 0, 0].item() == 0.0
    assert correction[0, 0, 0, 1].item() == 1.0
    assert torch.count_nonzero(correction[..., -1]) == 0
    assert torch.count_nonzero(correction) == 1


def test_shared_stencil_has_exactly_eight_new_parameters_and_no_ab_maps() -> None:
    baseline = MSHNet(3)
    model = SharedDiscrepancyStencilMSHNet(3)
    baseline_names = set(dict(baseline.named_parameters()))
    model_parameters = dict(model.named_parameters())
    new_names = set(model_parameters) - baseline_names

    assert new_names == {"stencil.theta"}
    assert model_parameters["stencil.theta"].numel() == 8
    assert model.stencil.max_l1 is None
    assert torch.equal(
        model.stencil.effective_weights(), model.stencil.theta
    )
    assert not any(name.startswith(("A.", "B.", "a_map.", "b_map.")) for name in new_names)

    bounded = SharedDiscrepancyStencil((1.0,) * 8, max_l1=2.0)
    assert torch.allclose(bounded.effective_l1(), torch.tensor(2.0))


def test_zero_stencil_matches_mshnet_in_eval_and_one_training_step() -> None:
    torch.manual_seed(401)
    baseline = MSHNet(3)
    model = SharedDiscrepancyStencilMSHNet(3, freeze_bn_statistics=True)
    _load_baseline(baseline, model)
    x_eval = torch.randn(1, 3, 32, 32)
    baseline.eval()
    model.eval()

    with torch.no_grad():
        baseline_masks, baseline_pred, baseline_features = _capture_native(
            baseline, x_eval
        )
        model_masks, model_pred, model_features = _capture_native(model, x_eval)
    assert torch.equal(model_pred, baseline_pred)
    for actual, expected in zip(model_masks, baseline_masks):
        assert torch.equal(actual, expected)
    # Hooks see the final call at each stage in the stencil model, so its
    # captured tensors are alternate states rather than factual states.  The
    # returned decoder features are the factual zero-stencil trajectory.
    with torch.no_grad():
        model_dict = model(x_eval, True, return_dict=True)
    for actual, expected in zip(model_dict["decoder_features"], baseline_features):
        assert torch.equal(actual, expected)

    baseline.train()
    model.train()
    _set_bn_eval(baseline)
    x = torch.randn(2, 3, 16, 16)
    target = torch.randn(2, 1, 16, 16)
    baseline_optimizer = torch.optim.SGD(baseline.parameters(), lr=1e-3)
    model_optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

    baseline_masks, baseline_pred = baseline(x, True)
    model_masks, model_pred = model(x, True)
    baseline_loss = (baseline_pred - target).square().mean() + sum(
        mask.square().mean() for mask in baseline_masks
    )
    model_loss = (model_pred - target).square().mean() + sum(
        mask.square().mean() for mask in model_masks
    )
    assert torch.equal(model_pred, baseline_pred)
    assert torch.equal(model_loss, baseline_loss)
    baseline_loss.backward()
    model_loss.backward()

    baseline_parameters = dict(baseline.named_parameters())
    model_parameters = dict(model.named_parameters())
    for name, baseline_parameter in baseline_parameters.items():
        model_gradient = model_parameters[name].grad
        baseline_gradient = baseline_parameter.grad
        assert (model_gradient is None) == (baseline_gradient is None), name
        if baseline_gradient is not None:
            assert torch.equal(model_gradient, baseline_gradient), name

    baseline_optimizer.step()
    model_optimizer.step()
    for name, baseline_parameter in baseline_parameters.items():
        assert torch.equal(model_parameters[name], baseline_parameter), name

    baseline_modules = dict(baseline.named_modules())
    model_modules = dict(model.named_modules())
    for name, baseline_module in baseline_modules.items():
        if not isinstance(baseline_module, nn.modules.batchnorm._BatchNorm):
            continue
        model_module = model_modules[name]
        for actual, expected in zip(
            (
                model_module.running_mean,
                model_module.running_var,
                model_module.num_batches_tracked,
            ),
            (
                baseline_module.running_mean,
                baseline_module.running_var,
                baseline_module.num_batches_tracked,
            ),
        ):
            assert torch.equal(actual, expected), name


def test_zero_stencil_still_has_nonzero_kernel_gradient() -> None:
    torch.manual_seed(409)
    model = SharedDiscrepancyStencilMSHNet(3).train()
    output = model(
        torch.randn(2, 3, 32, 32),
        True,
        return_dict=True,
    )
    output["pred"].square().mean().backward()

    gradient = model.stencil.theta.grad
    assert gradient is not None
    assert torch.isfinite(gradient).all()
    assert gradient.abs().sum().item() > 0.0


def test_decoder_call_counts_are_two_then_three_three_three() -> None:
    torch.manual_seed(419)
    model = SharedDiscrepancyStencilMSHNet(3).eval()
    counts = {stage: 0 for stage in (0, 1, 2, 3)}
    handles = []
    for stage in counts:
        def count(_module, _inputs, *, key=stage):
            counts[key] += 1

        handles.append(
            getattr(model, "decoder_%d" % stage).register_forward_pre_hook(count)
        )
    try:
        with torch.no_grad():
            model(torch.randn(1, 3, 32, 32), True)
    finally:
        for handle in handles:
            handle.remove()

    assert counts == {0: 3, 1: 3, 2: 3, 3: 2}


def test_nonzero_stencil_correction_propagates_to_finer_states() -> None:
    torch.manual_seed(421)
    zero = SharedDiscrepancyStencilMSHNet(3).eval()
    nonzero = SharedDiscrepancyStencilMSHNet(3).eval()
    nonzero.load_state_dict(zero.state_dict(), strict=True)
    with torch.no_grad():
        nonzero.stencil.theta[STENCIL_DELTAS.index((0, 1))] = 0.05
    x = torch.randn(1, 3, 32, 32)

    with torch.no_grad():
        zero_output = zero(x, True, return_dict=True)
        nonzero_output = nonzero(x, True, return_dict=True)

    zero_terms = zero_output["sds"]["stage_terms"]
    nonzero_terms = nonzero_output["sds"]["stage_terms"]
    assert torch.equal(nonzero_terms[3]["state"], zero_terms[3]["state"])
    assert not torch.equal(nonzero_terms[2]["state"], zero_terms[2]["state"])
    assert not torch.equal(
        nonzero_terms[1]["inherited"], zero_terms[1]["inherited"]
    )
    assert not torch.equal(nonzero_terms[0]["state"], zero_terms[0]["state"])
    assert not torch.equal(nonzero_output["pred"], zero_output["pred"])
