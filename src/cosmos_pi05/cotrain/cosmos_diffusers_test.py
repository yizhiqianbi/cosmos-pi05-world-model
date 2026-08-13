import torch

from cosmos_pi05.cotrain.cosmos_diffusers import rectified_flow_interpolate
from cosmos_pi05.cotrain.cosmos_diffusers import sample_waver_sigmas


def test_waver_sigmas_are_bounded():
    sigma = sample_waver_sigmas(4096, torch.device("cpu"))
    assert sigma.shape == (4096,)
    assert torch.all(sigma >= 0.001)
    assert torch.all(sigma <= 0.999)


def test_first_latent_frame_remains_clean():
    clean = torch.randn(2, 4, 5, 3, 3)
    noise = torch.randn_like(clean)
    sigma = torch.tensor([0.2, 0.8])
    condition = torch.zeros(5, 1, 1)
    condition[0] = 1
    xt, velocity, sigma_eff = rectified_flow_interpolate(clean, noise, sigma, condition)
    torch.testing.assert_close(xt[:, :, 0], clean[:, :, 0])
    torch.testing.assert_close(velocity, noise - clean)
    assert torch.all(sigma_eff[:, :, 0] == 0)
