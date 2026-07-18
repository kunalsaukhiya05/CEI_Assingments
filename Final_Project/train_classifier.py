import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.metrics import f1_score, confusion_matrix, classification_report
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from models import BaselineCNN, get_resnet18
from PIL import Image

DATA_DIR = './data'
BATCH_SIZE = 64
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def get_dataloaders():
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Custom Dataset to read from split files
    class EuroSATSplit(torch.utils.data.Dataset):
        def __init__(self, split_file, transform=None):
            with open(split_file, 'r') as f:
                self.files = [line.strip() for line in f.readlines()]
            self.transform = transform
            self.classes = sorted(list(set([os.path.basename(os.path.dirname(f)) for f in self.files])))
            self.class_to_idx = {c: i for i, c in enumerate(self.classes)}
            
        def __len__(self):
            return len(self.files)
            
        def __getitem__(self, idx):
            filepath = self.files[idx]
            img = Image.open(filepath).convert('RGB')
            if self.transform:
                img = self.transform(img)
            label = self.class_to_idx[os.path.basename(os.path.dirname(filepath))]
            return img, label

    train_split_path = os.path.join(DATA_DIR, 'splits', 'train.txt')
    val_split_path = os.path.join(DATA_DIR, 'splits', 'val.txt')
    
    if os.path.exists(train_split_path) and os.path.exists(val_split_path):
        train_dataset = EuroSATSplit(train_split_path, transform=transform)
        val_dataset = EuroSATSplit(val_split_path, transform=transform)
    else:
        # Fallback to ImageFolder if splits don't exist
        print("Splits not found. Falling back to ImageFolder.")
        full_dataset = datasets.ImageFolder(os.path.join(DATA_DIR, '2750'), transform=transform)
        train_size = int(0.8 * len(full_dataset))
        val_size = len(full_dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])
        
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # UC Merced Dataloader
    ucmerced_dir = os.path.join(DATA_DIR, 'UCMerced_LandUse', 'Images')
    if os.path.exists(ucmerced_dir):
        ucmerced_dataset = datasets.ImageFolder(ucmerced_dir, transform=transform)
        ucmerced_loader = DataLoader(ucmerced_dataset, batch_size=BATCH_SIZE, shuffle=False)
    else:
        ucmerced_loader = None
        
    return train_loader, val_loader, ucmerced_loader

def train_epoch(model, dataloader, criterion, optimizer):
    model.train()
    running_loss = 0.0
    for i, (inputs, labels) in enumerate(dataloader):
        inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * inputs.size(0)
        
        if (i + 1) % 50 == 0:
            print(f"  Batch {i+1}/{len(dataloader)} - Loss: {loss.item():.4f}", flush=True)
            
    return running_loss / len(dataloader.dataset)

def evaluate(model, dataloader, criterion):
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
    loss = running_loss / len(dataloader.dataset)
    f1 = f1_score(all_labels, all_preds, average='macro')
    return loss, f1, all_labels, all_preds

def train_baseline(train_loader, val_loader):
    print("--- Training Baseline CNN ---")
    model = BaselineCNN(num_classes=10).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    epochs = 10
    for epoch in range(epochs):
        train_loss = train_epoch(model, train_loader, criterion, optimizer)
        val_loss, val_f1, _, _ = evaluate(model, val_loader, criterion)
        print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f} - Val F1: {val_f1:.4f}", flush=True)
    return model

def train_resnet(train_loader, val_loader):
    print("--- Training ResNet-18 (Two-Phase Fine-Tuning) ---")
    model = get_resnet18(num_classes=10).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    
    # Phase 1: Freeze backbone, train head
    for param in model.parameters():
        param.requires_grad = False
    for param in model.fc.parameters():
        param.requires_grad = True
        
    optimizer = optim.Adam(model.fc.parameters(), lr=0.001)
    
    print("Phase 1: Frozen Backbone (3 epochs)")
    frozen_val_f1 = 0
    for epoch in range(3):
        train_loss = train_epoch(model, train_loader, criterion, optimizer)
        val_loss, val_f1, _, _ = evaluate(model, val_loader, criterion)
        frozen_val_f1 = val_f1
        print(f"Epoch {epoch+1}/3 - Train Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f} - Val F1: {val_f1:.4f}", flush=True)
        
    # Phase 2: Unfreeze last 2 blocks (layer3 and layer4)
    for param in model.layer3.parameters():
        param.requires_grad = True
    for param in model.layer4.parameters():
        param.requires_grad = True
        
    optimizer = optim.Adam([
        {'params': model.layer3.parameters(), 'lr': 0.0001},
        {'params': model.layer4.parameters(), 'lr': 0.0001},
        {'params': model.fc.parameters(), 'lr': 0.001}
    ])
    
    print("Phase 2: Unfrozen Blocks (5 epochs)")
    unfrozen_val_f1 = 0
    best_model_state = None
    for epoch in range(5):
        train_loss = train_epoch(model, train_loader, criterion, optimizer)
        val_loss, val_f1, _, _ = evaluate(model, val_loader, criterion)
        if val_f1 > unfrozen_val_f1:
            best_model_state = model.state_dict()
            unfrozen_val_f1 = val_f1
        print(f"Epoch {epoch+1}/5 - Train Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f} - Val F1: {val_f1:.4f}", flush=True)
    
    # Load best state
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    # Save ablation table
    ablation_data = {
        'Phase': ['Phase 1 (Frozen)', 'Phase 2 (Unfrozen)'],
        'Val Macro F1': [frozen_val_f1, unfrozen_val_f1]
    }
    pd.DataFrame(ablation_data).to_csv('ablation_table.csv', index=False)
    print("Saved ablation_table.csv")
    
    # Save checkpoint
    torch.save(model.state_dict(), 'resnet18_finetuned.pt')
    print("Saved checkpoint resnet18_finetuned.pt")
    
    return model

def plot_confusion_matrix(labels, preds, classes, filename):
    cm = confusion_matrix(labels, preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.title('Confusion Matrix')
    plt.ylabel('True')
    plt.xlabel('Predicted')
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

def main():
    train_loader, val_loader, ucmerced_loader = get_dataloaders()
    
    # Train Baseline
    baseline_model = train_baseline(train_loader, val_loader)
    
    # Train ResNet-18
    resnet_model = train_resnet(train_loader, val_loader)
    
    # Evaluate ResNet on EuroSAT Val
    criterion = nn.CrossEntropyLoss()
    _, val_f1, val_labels, val_preds = evaluate(resnet_model, val_loader, criterion)
    
    if hasattr(val_loader.dataset, 'classes'):
        classes = val_loader.dataset.classes
    else:
        classes = [str(i) for i in range(10)]
    
    print("\nEuroSAT Validation Results (ResNet-18):")
    print(classification_report(val_labels, val_preds, target_names=classes))
    plot_confusion_matrix(val_labels, val_preds, classes, 'cm_eurosat.png')
    
    # Evaluate on UC Merced (Feature Extraction Linear Probe or Zero-shot)
    # Since UC Merced has 21 classes, we will train a quick linear classifier on top of the ResNet features
    # to get the per-class F1 for UC Merced as requested.
    if ucmerced_loader:
        print("\n--- Evaluating on UC Merced (Linear Probe) ---")
        from models import get_embedding_extractor
        extractor = get_embedding_extractor(resnet_model).to(DEVICE)
        extractor.eval()
        
        features = []
        labels_list = []
        with torch.no_grad():
            for inputs, labels in ucmerced_loader:
                inputs = inputs.to(DEVICE)
                f = extractor(inputs)
                f = nn.AdaptiveAvgPool2d((1, 1))(f)
                f = torch.flatten(f, 1)
                features.append(f.cpu())
                labels_list.append(labels)
                
        features = torch.cat(features)
        labels_tensor = torch.cat(labels_list)
        
        # Train linear probe
        ucm_classes = ucmerced_loader.dataset.classes
        probe = nn.Linear(512, len(ucm_classes)).to(DEVICE)
        probe_optimizer = optim.Adam(probe.parameters(), lr=0.01)
        probe_criterion = nn.CrossEntropyLoss()
        
        features = features.to(DEVICE)
        labels_tensor = labels_tensor.to(DEVICE)
        
        for _ in range(10): # quick training
            probe_optimizer.zero_grad()
            outputs = probe(features)
            loss = probe_criterion(outputs, labels_tensor)
            loss.backward()
            probe_optimizer.step()
            
        _, ucm_preds = torch.max(probe(features), 1)
        ucm_labels = labels_tensor.cpu().numpy()
        ucm_preds = ucm_preds.cpu().numpy()
        
        print(classification_report(ucm_labels, ucm_preds, target_names=ucm_classes))
        plot_confusion_matrix(ucm_labels, ucm_preds, ucm_classes, 'cm_ucmerced.png')

if __name__ == '__main__':
    main()
