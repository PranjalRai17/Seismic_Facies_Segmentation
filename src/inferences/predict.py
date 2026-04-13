import torch
import numpy as np
import cv2
import tifffile as tiff
import matplotlib.pyplot as plt
import argparse

from src.models.unet import UNet


# ------------------ LOAD IMAGE ------------------
def load_image(path, img_size=256):
    img = tiff.imread(path)

    img = cv2.resize(img, (img_size, img_size))
    img = (img - np.mean(img)) / (np.std(img) + 1e-6)

    img = torch.tensor(img, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    return img


# ------------------ PREDICTION ------------------
def predict(model, image, device):
    model.eval()

    with torch.no_grad():
        output = model(image.to(device))
        pred = torch.argmax(output, dim=1).squeeze().cpu().numpy()

    return pred


# ------------------ VISUALIZATION ------------------
def visualize(image, pred):
    plt.figure(figsize=(12,5))

    plt.subplot(1,2,1)
    plt.imshow(image.squeeze(), cmap='gray')
    plt.title("Seismic")
    plt.axis('off')

    plt.subplot(1,2,2)
    plt.imshow(pred, cmap='jet')
    plt.title("Prediction")
    plt.axis('off')

    plt.show()


# ------------------ MAIN ------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Seismic Facies Segmentation")
    parser.add_argument("--image", type=str, required=True, help="Path to input .tiff file")
    parser.add_argument("--model", type=str, default="models/best_model.pth", help="Path to model weights")
    parser.add_argument("--img_size", type=int, default=256)

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # load model
    model = UNet(num_classes=10).to(device)
    model.load_state_dict(torch.load(args.model, map_location=device))

    # load image
    image = load_image(args.image, img_size=args.img_size)

    # predict
    pred = predict(model, image, device)

    # visualize
    visualize(image, pred)