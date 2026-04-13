import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Subset

from src.models.unet import UNet
from src.datasets.seismic_dataset import SeismicDataset
from src.utils.losses import CombinedLoss
from src.utils.metrics import compute_iou, pixel_accuracy


def train(
    seismic_dir,
    mask_dir,
    epochs=25,
    batch_size=1,
    lr=1e-4,
    img_size=256,
    save_path="models/best_model.pth"
):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # -------- DATASET --------
    dataset = SeismicDataset(seismic_dir, mask_dir, img_size=img_size, augment=True)

    # spatial split (IMPORTANT)
    dataset_size = len(dataset)
    split = int(0.8 * dataset_size)

    train_dataset = Subset(dataset, list(range(0, split)))
    val_dataset = Subset(dataset, list(range(split, dataset_size)))

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # -------- MODEL --------
    model = UNet(num_classes=10).to(device)

    # -------- LOSS --------
    criterion = CombinedLoss()

    # -------- OPTIMIZER --------
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

    # -------- SCHEDULER --------
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=2
    )

    # -------- TRAINING --------
    best_val = float('inf')
    patience = 5
    counter = 0

    for epoch in range(epochs):

        # ---- TRAIN ----
        model.train()
        train_loss = 0

        for seismic, mask in train_loader:
            seismic, mask = seismic.to(device), mask.to(device)

            outputs = model(seismic)
            loss = criterion(outputs, mask)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)

        # ---- VALIDATION ----
        model.eval()
        val_loss = 0
        iou_total = 0
        acc_total = 0

        with torch.no_grad():
            for seismic, mask in val_loader:
                seismic, mask = seismic.to(device), mask.to(device)

                outputs = model(seismic)
                loss = criterion(outputs, mask)

                val_loss += loss.item()
                iou_total += compute_iou(outputs, mask)
                acc_total += pixel_accuracy(outputs, mask)

        val_loss /= len(val_loader)
        iou_score = iou_total / len(val_loader)
        acc_score = acc_total / len(val_loader)

        # ---- SCHEDULER ----
        scheduler.step(val_loss)

        # ---- LOGGING ----
        print(f"\nEpoch {epoch+1}/{epochs}")
        print(f"Train Loss: {train_loss:.4f}")
        print(f"Val Loss:   {val_loss:.4f}")
        print(f"IoU:        {iou_score:.4f}")
        print(f"Accuracy:   {acc_score:.4f}")

        # ---- SAVE BEST ----
        if val_loss < best_val:
            best_val = val_loss
            counter = 0
            torch.save(model.state_dict(), save_path)
            print("✅ Saved best model")

        else:
            counter += 1

        # ---- EARLY STOP ----
        if counter >= patience:
            print("⛔ Early stopping triggered")
            break


if __name__ == "__main__":
    train(
        seismic_dir="data/inlines",
        mask_dir="data/masks"
    )