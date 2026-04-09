import torch
from src.model import SkinCancerModel


def test_model():
    model = SkinCancerModel()
    model.eval()

    images = torch.randn(2, 3, 224, 224)
    ages = torch.tensor([0.45, 0.62]).unsqueeze(1)
    genders = torch.tensor([0.0, 1.0]).unsqueeze(1)

    with torch.no_grad():
        outputs = model(images, ages, genders)

    print("Output shape:", outputs.shape)


if __name__ == "__main__":
    test_model()
