import torch
from torch.utils.data import DataLoader
import numpy as np
import argparse
import os
import time
import pickle
from models import utils, caption
from datasets import xray
from utils.engine import train_one_epoch, evaluate
from models.model import swin_tiny_patch4_window7_224 as create_model
from utils.stloss import SoftTarget
from torch.nn import BCEWithLogitsLoss
import albumentations as A
from albumentations.pytorch import ToTensorV2

# Define Focal Loss
class FocalLoss(BCEWithLogitsLoss):
    def __init__(self, gamma=2, alpha=0.25, reduction='mean'):
        super().__init__(reduction=reduction)
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, inputs, targets):
        BCE_loss = super().forward(inputs, targets)
        pt = torch.exp(-BCE_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss
        return focal_loss

# Define data augmentation
def get_augmentation():
    return A.Compose([
        A.Resize(300, 300),
        A.RandomCrop(256, 256),
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.2),
        A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=15, p=0.5),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])

# Build model
def build_diagnosisbot(num_classes, detector_weight_path):
    model = create_model(num_classes=num_classes)
    assert os.path.exists(detector_weight_path), f"File: '{detector_weight_path}' does not exist."
    model.load_state_dict(torch.load(detector_weight_path, map_location=torch.device('cpu')), strict=True)
    for _, v in model.named_parameters():
        v.requires_grad = False
    return model

def build_tmodel(config, device):
    tmodel, _ = caption.build_model(config)
    tcheckpoint = torch.load(config.t_model_weight_path, map_location='cpu')
    tmodel.load_state_dict(tcheckpoint['model'])
    tmodel.to(device)
    return tmodel

# Main function
def main(config):
    device = torch.device(config.device)
    print(f"Device: {device}")

    if os.path.exists(config.thresholds_path):
        with open(config.thresholds_path, "rb") as f:
            thresholds = pickle.load(f)

    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    detector = build_diagnosisbot(config.num_classes, config.detector_weight_path).to(device)

    model, criterion = caption.build_model(config)
    criterionKD = SoftTarget(4.0)
    model.to(device)

    param_dicts = [
        {"params": [p for n, p in model.named_parameters() if "backbone" not in n and p.requires_grad]},
        {"params": [p for n, p in model.named_parameters() if "backbone" in n and p.requires_grad], "lr": config.lr_backbone},
    ]
    optimizer = torch.optim.AdamW(param_dicts, lr=config.lr, weight_decay=config.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs)

    transform = get_augmentation()
    dataset_train = xray.build_dataset(config, mode='training', anno_path=config.anno_path, data_dir=config.data_dir,
                                       dataset_name=config.dataset_name, image_size=config.image_size, transform=transform)
    dataset_val = xray.build_dataset(config, mode='validation', anno_path=config.anno_path, data_dir=config.data_dir,
                                     dataset_name=config.dataset_name, image_size=config.image_size, transform=transform)
    dataset_test = xray.build_dataset(config, mode='test', anno_path=config.anno_path, data_dir=config.data_dir,
                                      dataset_name=config.dataset_name, image_size=config.image_size, transform=transform)

    data_loader_train = DataLoader(dataset_train, batch_size=config.batch_size, shuffle=True, num_workers=config.num_workers)
    data_loader_val = DataLoader(dataset_val, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers)
    data_loader_test = DataLoader(dataset_test, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers)

    if config.mode == "train":
        tmodel = build_tmodel(config, device)
        for epoch in range(config.start_epoch, config.epochs):
            print(f"Epoch {epoch}/{config.epochs}")
            train_loss = train_one_epoch(model, tmodel, detector, criterion, criterionKD, data_loader_train, optimizer,
                                         device, config.clip_max_norm, thresholds, config=config)
            lr_scheduler.step()
            torch.save({'model': model.state_dict()}, f"{config.dataset_name}_epoch_{epoch}.pth")
            val_result = evaluate(model, detector, criterion, data_loader_val, device, config, thresholds)
            print(f"Validation Results: {val_result}")
    elif config.mode == "test":
        weights_dict = torch.load(config.test_path, map_location='cpu')['model']
        model.load_state_dict(weights_dict, strict=False)
        test_result = evaluate(model, detector, criterion, data_loader_test, device, config, thresholds)
        print(f"Test Results: {test_result}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--lr_backbone', type=float, default=1e-5)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--dataset_name', type=str, default='mimic_cxr')
    parser.add_argument('--anno_path', type=str, default='../dataset/mimic_cxr/annotation.json')
    parser.add_argument('--data_dir', type=str, default='../dataset/mimic_cxr/images300')
    parser.add_argument('--num_classes', type=int, default=14)
    parser.add_argument('--mode', type=str, default="train")
    parser.add_argument('--thresholds_path', type=str, default="./datasets/thresholds.pkl")
    parser.add_argument('--detector_weight_path', type=str, default="./weight_path/diagnosisbot.pth")
    parser.add_argument('--t_model_weight_path', type=str, default="./weight_path/mimic_t_model.pth")
    parser.add_argument('--test_path', type=str, default="")
    parser.add_argument('--clip_max_norm', type=float, default=0.1)
    parser.add_argument('--start_epoch', type=int, default=0)
    config = parser.parse_args()
    main(config)
