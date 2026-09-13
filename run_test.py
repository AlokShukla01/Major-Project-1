import os
import cv2
import glob
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader, Subset
import torch.nn as nn
import torch.optim as optim

if torch.backends.mps.is_available():
    device = torch.device('mps')
elif torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

class SUNPolypDataset(Dataset):
    def __init__(self, root_dir, image_size=(128, 128)):
        self.root_dir = root_dir
        self.image_size = image_size
        self.samples = []
        anno_dir = os.path.join(root_dir, 'annotation_txt')
        txt_files = glob.glob(os.path.join(anno_dir, '*.txt'))
        for txt_file in txt_files:
            case_name = os.path.basename(txt_file).replace('.txt', '')
            case_dir = os.path.join(root_dir, case_name)
            if not os.path.exists(case_dir): continue
            with open(txt_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if not line: continue
                    parts = line.split(' ')
                    if len(parts) >= 2:
                        filename = parts[0]
                        coords = parts[1].split(',')
                        if len(coords) >= 4:
                            box = [int(c) for c in coords[:4]]
                            img_path = os.path.join(case_dir, filename)
                            if os.path.exists(img_path):
                                self.samples.append((img_path, box))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, box = self.samples[idx]
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w = img.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        y1, x1, y2, x2 = box[0], box[1], box[2], box[3]
        col_min, col_max = min(x1, x2), max(x1, x2)
        row_min, row_max = min(y1, y2), max(y1, y2)
        mask[row_min:row_max, col_min:col_max] = 1
        img = cv2.resize(img, self.image_size)
        mask = cv2.resize(mask, self.image_size, interpolation=cv2.INTER_NEAREST)
        img = img.astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))
        return torch.tensor(img, dtype=torch.float32), torch.tensor(mask, dtype=torch.float32).unsqueeze(0)

def calculate_metrics(pred, target, threshold=0.5):
    pred = (torch.sigmoid(pred) > threshold).float()
    target = target.float()
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum()
    dice = (2. * intersection + 1e-6) / (union + 1e-6)
    iou = (intersection + 1e-6) / (union - intersection + 1e-6)
    correct = (pred == target).sum()
    accuracy = correct / torch.numel(pred)
    return accuracy.item(), dice.item(), iou.item()

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.conv(x)

class BasicUNet(nn.Module):
    def __init__(self, n_channels=3, n_classes=1):
        super().__init__()
        self.inc = DoubleConv(n_channels, 64)
        self.down1 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(64, 128))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(128, 256))
        self.up1 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.conv_up1 = DoubleConv(256, 128)
        self.up2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv_up2 = DoubleConv(128, 64)
        self.outc = nn.Conv2d(64, n_classes, kernel_size=1)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x = self.up1(x3)
        x = torch.cat([x2, x], dim=1)
        x = self.conv_up1(x)
        x = self.up2(x)
        x = torch.cat([x1, x], dim=1)
        x = self.conv_up2(x)
        return self.outc(x)

if __name__ == '__main__':
    dataset_path = '/Users/alokkumarshukla/Desktop/Major Project 1/Sun/sundatabase_positive_part1'
    dataset = SUNPolypDataset(dataset_path)
    model = BasicUNet().to(device)
    subset_indices = list(range(500))
    subset_dataset = Subset(dataset, subset_indices)
    train_size = int(0.8 * len(subset_dataset))
    val_size = len(subset_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(subset_dataset, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    epochs = 3
    print("Starting training...")
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for imgs, masks in train_loader:
            imgs, masks = imgs.to(device), masks.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
        model.eval()
        val_acc, val_dice, val_iou = 0, 0, 0
        with torch.no_grad():
            for imgs, masks in val_loader:
                imgs, masks = imgs.to(device), masks.to(device)
                outputs = model(imgs)
                acc, dice, iou = calculate_metrics(outputs, masks)
                val_acc += acc
                val_dice += dice
                val_iou += iou

        avg_train_loss = train_loss / len(train_loader)
        avg_val_acc = val_acc / len(val_loader)
        avg_val_dice = val_dice / len(val_loader)
        avg_val_iou = val_iou / len(val_loader)
        print(f'Epoch {epoch+1}/{epochs} | Train Loss: {avg_train_loss:.4f} | Val Acc: {avg_val_acc:.4f} | Val Dice: {avg_val_dice:.4f} | Val IoU: {avg_val_iou:.4f}')
