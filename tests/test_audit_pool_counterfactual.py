import torch

from model.baselines.mshnet_deterministic import MSHNet
from tools.audit_pool_counterfactual import forward_with_pool_counterfactual


def test_alpha_zero_preserves_full_mshnet_forward_at_every_boundary() -> None:
    torch.manual_seed(3)
    model = MSHNet(3).eval()
    image = torch.randn(1, 3, 32, 32)
    with torch.no_grad():
        _, canonical = model(image, True)
        for stage in range(4):
            audited = forward_with_pool_counterfactual(
                model, image, stage=stage, alpha=0.0
            )
            torch.testing.assert_close(audited, canonical, rtol=0, atol=0)
