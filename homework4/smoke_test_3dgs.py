import torch

from gaussian_model import GaussianModel
from gaussian_renderer import GaussianRenderer


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    points = torch.tensor(
        [
            [-0.2, -0.2, 3.0],
            [0.2, -0.2, 3.2],
            [0.0, 0.2, 3.4],
            [0.3, 0.3, 3.6],
        ],
        dtype=torch.float32,
    )
    colors = torch.tensor(
        [
            [255, 0, 0],
            [0, 255, 0],
            [0, 0, 255],
            [255, 255, 0],
        ],
        dtype=torch.float32,
    )

    model = GaussianModel(points, colors).to(device)
    params = model()
    renderer = GaussianRenderer(image_height=16, image_width=16).to(device)
    K = torch.tensor(
        [[40.0, 0.0, 8.0], [0.0, 40.0, 8.0], [0.0, 0.0, 1.0]],
        dtype=torch.float32,
        device=device,
    )
    R = torch.eye(3, dtype=torch.float32, device=device)
    t = torch.zeros(3, 1, dtype=torch.float32, device=device)

    image = renderer(
        params["positions"],
        params["covariance"],
        params["colors"],
        params["opacities"],
        K,
        R,
        t,
    )

    assert params["covariance"].shape == (4, 3, 3)
    assert image.shape == (16, 16, 3)
    assert torch.isfinite(params["covariance"]).all()
    assert torch.isfinite(image).all()
    assert float(image.min()) >= 0.0
    assert float(image.max()) <= 1.0
    print("SMOKE_OK", device, image.shape, float(image.sum()))


if __name__ == "__main__":
    main()
