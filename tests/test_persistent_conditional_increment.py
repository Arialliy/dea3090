from __future__ import annotations

import torch
from torch import nn

from model.MSHNet import MSHNet
from model.dea_persistent_conditional_increment import (
    PersistentConditionalIncrementMSHNet,
    persistent_conditional_increment_step,
)


def _forward_with_decoder_features(model, x):
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


def _decoder_call_counts(model, x, alpha):
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
            model(x, True, alpha=alpha)
    finally:
        for handle in handles:
            handle.remove()
    return counts


def test_pci_alpha_zero_is_bitwise_mshnet() -> None:
    torch.manual_seed(307)
    baseline = MSHNet(3).eval()
    model = PersistentConditionalIncrementMSHNet(
        3,
        alpha=0.0,
        anchor_mode="mean",
    ).eval()
    model.load_state_dict(baseline.state_dict(), strict=True)
    x = torch.randn(1, 3, 32, 48)

    with torch.no_grad():
        baseline_masks, baseline_pred, baseline_features = (
            _forward_with_decoder_features(baseline, x)
        )
        pci_masks, pci_pred, pci_features = _forward_with_decoder_features(model, x)
        pci_dict = model(x, True, return_dict=True)

    assert torch.equal(pci_pred, baseline_pred)
    for actual, expected in zip(pci_masks, baseline_masks):
        assert torch.equal(actual, expected)
    for actual, expected in zip(pci_features, baseline_features):
        assert torch.equal(actual, expected)

    assert pci_dict["pci"]["hard_baseline"] is True
    assert torch.equal(pci_dict["pred"], baseline_pred)
    for actual, expected in zip(pci_dict["masks"], baseline_masks):
        assert torch.equal(actual, expected)
    for actual, expected in zip(pci_dict["decoder_features"], baseline_features):
        assert torch.equal(actual, expected)
    assert not any("alpha" in name for name, _ in model.named_parameters())


def test_pci_alpha_zero_matches_one_native_training_step() -> None:
    torch.manual_seed(309)
    baseline = MSHNet(3)
    model = PersistentConditionalIncrementMSHNet(
        3,
        alpha=0.0,
        freeze_bn_statistics=False,
    )
    model.load_state_dict(baseline.state_dict(), strict=True)
    baseline.train()
    model.train()
    baseline_optimizer = torch.optim.SGD(baseline.parameters(), lr=1e-3)
    model_optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
    x = torch.randn(2, 3, 16, 16)
    target = torch.randn(2, 1, 16, 16)

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
    assert baseline_parameters.keys() == model_parameters.keys()
    for name in baseline_parameters:
        baseline_gradient = baseline_parameters[name].grad
        model_gradient = model_parameters[name].grad
        assert (model_gradient is None) == (baseline_gradient is None), name
        if baseline_gradient is not None:
            assert torch.equal(model_gradient, baseline_gradient), name

    baseline_optimizer.step()
    model_optimizer.step()
    for name in baseline_parameters:
        assert torch.equal(
            model_parameters[name], baseline_parameters[name]
        ), name

    baseline_modules = dict(baseline.named_modules())
    model_modules = dict(model.named_modules())
    for name, baseline_module in baseline_modules.items():
        if not isinstance(baseline_module, nn.modules.batchnorm._BatchNorm):
            continue
        model_module = model_modules[name]
        for model_buffer, baseline_buffer in zip(
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
            assert torch.equal(model_buffer, baseline_buffer), name


def test_pci_homotopy_formula_and_alpha_one_equals_alternate() -> None:
    torch.manual_seed(311)
    model = PersistentConditionalIncrementMSHNet(3, alpha=0.35).eval()
    x = torch.randn(1, 3, 32, 32)

    with torch.no_grad():
        mixed = model(x, True, return_dict=True)
        final = model(x, True, return_dict=True, alpha=1.0)

    for stage in (2, 1, 0):
        terms = mixed["pci"]["stage_terms"][stage]
        expected = 0.65 * terms["factual"] + 0.35 * terms["alternate"]
        assert torch.allclose(terms["state"], expected, atol=1e-7, rtol=1e-6)
        assert torch.allclose(
            terms["increment"],
            terms["state"] - terms["local"],
            atol=1e-7,
            rtol=1e-6,
        )

        final_terms = final["pci"]["stage_terms"][stage]
        assert final_terms["factual"] is None
        assert torch.equal(final_terms["state"], final_terms["alternate"])


def test_pci_decoder_call_counts_match_homotopy_contract() -> None:
    torch.manual_seed(313)
    model = PersistentConditionalIncrementMSHNet(3).eval()
    x = torch.randn(1, 3, 32, 32)

    assert _decoder_call_counts(model, x, 0.0) == {
        0: 1,
        1: 1,
        2: 1,
        3: 1,
    }
    assert _decoder_call_counts(model, x, 1.0) == {
        0: 2,
        1: 2,
        2: 2,
        3: 2,
    }
    assert _decoder_call_counts(model, x, 0.5) == {
        0: 3,
        1: 3,
        2: 3,
        3: 2,
    }


def test_pci_counterfactual_calls_do_not_update_bn_buffers() -> None:
    torch.manual_seed(317)
    model = PersistentConditionalIncrementMSHNet(
        3,
        alpha=0.5,
        freeze_bn_statistics=True,
    ).train()
    batch_norms = {
        name: module
        for name, module in model.named_modules()
        if isinstance(module, nn.modules.batchnorm._BatchNorm)
    }
    before = {
        name: (
            module.running_mean.detach().clone(),
            module.running_var.detach().clone(),
            module.num_batches_tracked.detach().clone(),
        )
        for name, module in batch_norms.items()
    }

    output = model(torch.randn(2, 3, 32, 32), True, return_dict=True)

    assert torch.isfinite(output["pred"]).all()
    assert batch_norms and all(not module.training for module in batch_norms.values())
    for name, module in batch_norms.items():
        for actual, expected in zip(
            (module.running_mean, module.running_var, module.num_batches_tracked),
            before[name],
        ):
            assert torch.equal(actual, expected), name


def test_pci_alpha_one_path_has_finite_decoder_gradients() -> None:
    torch.manual_seed(331)
    model = PersistentConditionalIncrementMSHNet(
        3,
        alpha=1.0,
        freeze_bn_statistics=True,
    ).train()
    x = torch.randn(2, 3, 32, 32, requires_grad=True)

    output = model(x, True, return_dict=True)
    loss = output["pred"].square().mean()
    loss.backward()

    assert x.grad is not None and torch.isfinite(x.grad).all()
    for stage in (0, 1, 2, 3):
        gradients = [
            parameter.grad
            for name, parameter in model.named_parameters()
            if name.startswith("decoder_%d." % stage)
        ]
        assert gradients
        assert all(gradient is not None for gradient in gradients)
        assert all(torch.isfinite(gradient).all() for gradient in gradients)
        assert sum(gradient.abs().sum().item() for gradient in gradients) > 0.0


def test_pci_state_changes_when_previous_increment_is_perturbed() -> None:
    torch.manual_seed(337)
    decoder = nn.Sequential(
        nn.Conv2d(7, 6, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.Conv2d(6, 5, kernel_size=1),
    ).eval()
    current = torch.randn(1, 3, 9, 11)
    inherited = torch.randn(1, 4, 9, 11)
    increment = torch.randn_like(inherited) * 0.1
    perturbation = torch.randn_like(inherited) * 0.2

    reference = persistent_conditional_increment_step(
        decoder,
        current,
        inherited,
        increment,
        alpha=1.0,
        anchor_mode="mean",
    )
    perturbed = persistent_conditional_increment_step(
        decoder,
        current,
        inherited,
        increment + perturbation,
        alpha=1.0,
        anchor_mode="mean",
    )

    assert torch.allclose(
        perturbed["persistent_input"] - reference["persistent_input"],
        perturbation,
    )
    assert not torch.equal(perturbed["alternate"], reference["alternate"])
    assert not torch.equal(perturbed["state"], reference["state"])
