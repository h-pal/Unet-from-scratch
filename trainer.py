import torch
from torch import nn
from torch.utils.data import DataLoader
from datasets import load_dataset
from torch.utils.data import Dataset as PDataset
from torchvision.transforms import v2
from unet_model import Unet
import torchvision.transforms.functional as F
import wandb

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize WandB
wandb.init(project="Practise", name="UNET-Trainer")

# Load dataset
ds_train = load_dataset("farmaieu/plantorgans", cache_dir="/uufs/chpc.utah.edu/common/home/tasdizen-group1/tutorial/data/", split='train')
ds_val = load_dataset("farmaieu/plantorgans", cache_dir="/uufs/chpc.utah.edu/common/home/tasdizen-group1/tutorial/data/", split='validation')

# Data transforms with augmentation
image_transforms = v2.Compose([
    v2.ToImage(),
    v2.Resize((572, 572)),
    v2.RandomHorizontalFlip(p=0.5),
    v2.RandomVerticalFlip(p=0.5),
    v2.RandomRotation(degrees=30),
    v2.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    v2.ToTensor(),
    v2.ConvertImageDtype(torch.float32),
])

label_transforms = v2.Compose([
    v2.ToImage(),
    v2.Resize((388, 388), interpolation=F.InterpolationMode.NEAREST),
    v2.ToTensor(),
])

# Dataset class
class MyDataIterator(PDataset):
    def __init__(self, hf_ds, img_transforms, lab_transforms):
        self.ds = hf_ds
        self.im_transforms = img_transforms
        self.lab_transforms = lab_transforms

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        image = self.ds[idx]['image']
        seg_map = self.ds[idx]['label']

        image = self.im_transforms(image)
        seg_map = self.lab_transforms(seg_map).to(torch.int64).squeeze(0)

        num_classes = 5
        one_hot_mask = torch.zeros((num_classes, seg_map.shape[0], seg_map.shape[1]), dtype=torch.float32)
        one_hot_mask = one_hot_mask.scatter_(0, seg_map.unsqueeze(0), 1.0)

        return {"image": image, "label": one_hot_mask}

# Dice Loss
class DiceLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, preds, gt, return_binary_score=True):
        """
        Args:
            preds: Tensor of shape (batch_size, num_classes, height, width)
            gt: Tensor of shape (batch_size, num_classes, height, width)
            return_binary_score: If True, also return Dice score with binarized predictions
        Returns:
            Dice loss and metric
        """
        assert preds.shape == gt.shape, f"Shape mismatch: preds {preds.shape}, gt {gt.shape}"

        # Apply softmax for loss calculation
        preds_soft = torch.softmax(preds, dim=1)

        # Calculate loss using soft predictions
        preds_flat = preds_soft.view(preds_soft.size(0), preds_soft.size(1), -1)
        gt_flat = gt.view(gt.size(0), gt.size(1), -1)

        intersection = (preds_flat * gt_flat).sum(dim=2)
        sum_pred = preds_flat.sum(dim=2)
        sum_gt = gt_flat.sum(dim=2)

        dice_per_channel = (2.0 * intersection + 1e-6) / (sum_pred + sum_gt + 1e-6)
        dice_per_case = dice_per_channel.mean(dim=1)
        dice_loss = 1 - dice_per_case.mean()

        if return_binary_score:
            # Calculate Dice score using binarized predictions
            preds_binary = torch.argmax(preds_soft, dim=1)
            preds_one_hot = torch.zeros_like(preds_soft)
            preds_one_hot.scatter_(1, preds_binary.unsqueeze(1), 1)

            # Flatten binary predictions
            preds_binary_flat = preds_one_hot.view(preds_one_hot.size(0), preds_one_hot.size(1), -1)

            # Calculate binary Dice score
            intersection_binary = (preds_binary_flat * gt_flat).sum(dim=2)
            sum_pred_binary = preds_binary_flat.sum(dim=2)

            dice_binary_per_channel = (2.0 * intersection_binary + 1e-6) / (sum_pred_binary + sum_gt + 1e-6)
            dice_binary_score = dice_binary_per_channel.mean()

            return dice_loss, dice_binary_score

        return dice_loss

# Combined Loss (Dice + CrossEntropy)
class CombinedLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.dice_loss = DiceLoss()
        self.ce_loss = nn.CrossEntropyLoss()

    def forward(self, preds, gt):
        dice_loss, dice_metric = self.dice_loss(preds, gt)
        ce_loss = self.ce_loss(preds, torch.argmax(gt, dim=1))
        return dice_loss + ce_loss, dice_metric

# Training and validation functions
def train(data_loader, model, loss_fn, opt):
    model.train()
    epoch_loss, epoch_metric, step = 0, 0, 0
    for batch in data_loader:
        step += 1
        x = batch["image"].to(device)
        mask = batch["label"].to(device)

        preds = model(x)
        loss, dice_metric = loss_fn(preds, mask)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Gradient clipping
        opt.step()
        opt.zero_grad()

        epoch_metric += dice_metric.item()
        epoch_loss += loss.item()
    return epoch_loss / step, epoch_metric / step

def validation(data_loader, model, loss_fn):
    model.eval()
    epoch_loss, epoch_metric, step = 0, 0, 0
    for batch in data_loader:
        step += 1
        x = batch["image"].to(device)
        mask = batch["label"].to(device)

        with torch.no_grad():
            preds = model(x)
            loss, dice_metric = loss_fn(preds, mask)

        epoch_metric += dice_metric.item()
        epoch_loss += loss.item()
    return epoch_loss / step, epoch_metric / step

# Data loaders
train_images = MyDataIterator(ds_train, image_transforms, label_transforms)
val_images = MyDataIterator(ds_val, image_transforms, label_transforms)

batch_size = 8
train_loader = DataLoader(train_images, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_images, batch_size=batch_size)

# Model, loss, and optimizer
input_channels = 3
output_classes = 5
model = Unet(input_channels, output_classes).to(device)

loss = CombinedLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=True)

# Training loop
epochs = 10000
val_every = 5

for epoch in range(epochs):
    train_loss, train_metric = train(train_loader, model, loss, optimizer)
    wandb.log({
        'epoch': epoch,
        'train_loss': train_loss,
        'train_dice_score': train_metric
    })

    if epoch % val_every == 0:
        val_loss, val_metric = validation(val_loader, model, loss)
        scheduler.step(val_loss)  # Update learning rate
        wandb.log({'val_loss': val_loss, 'val_dice_score': val_metric})

    if epoch % 10 == 0:  # Print every 10 epochs
        print(f"Epoch {epoch}")
        print(f"Train Loss: {train_loss:.4f}, Train Dice: {train_metric:.4f}")
        if epoch % val_every == 0:
            print(f"Val Loss: {val_loss:.4f}, Val Dice: {val_metric:.4f}")