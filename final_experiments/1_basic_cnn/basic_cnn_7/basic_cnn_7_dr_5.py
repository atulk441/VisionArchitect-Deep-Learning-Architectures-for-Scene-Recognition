import os
import time
import csv
from itertools import product
from PIL import Image
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, Subset
from torchvision import datasets, transforms
from sklearn.metrics import precision_score, recall_score, f1_score


class SimpleCNN(nn.Module):
    def __init__(self, num_classes, filters_list, dropout=0.3, use_batchnorm=False, pooling='max'):
        super().__init__()
        layers = []
        in_channels = 3

        for i, out_channels in enumerate(filters_list):
            layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))

            if use_batchnorm:
                layers.append(nn.BatchNorm2d(out_channels))

            layers.append(nn.ReLU())

            if pooling and i % 2 == 0:
                if pooling == 'max':
                    layers.append(nn.MaxPool2d(2))
                elif pooling == 'avg':
                    layers.append(nn.AvgPool2d(2))

            if dropout > 0:
                layers.append(nn.Dropout(dropout))

            in_channels = out_channels

        layers.append(nn.AdaptiveAvgPool2d((1, 1)))
        self.features = nn.Sequential(*layers)
        self.classifier = nn.Linear(filters_list[-1], num_classes)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss, correct, total = 0, 0, 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        out = model(imgs)
        loss = criterion(out, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * imgs.size(0)
        _, preds = out.max(1)
        correct += preds.eq(labels).sum().item()
        total += imgs.size(0)
    return total_loss / total, correct / total

def eval_one_epoch(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0, 0, 0
    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            out = model(imgs)
            loss = criterion(out, labels)
            total_loss += loss.item() * imgs.size(0)
            _, preds = out.max(1)
            correct += preds.eq(labels).sum().item()
            total += imgs.size(0)
    return total_loss / total, correct / total

def get_optimizer(name, params, lr, weight_decay):
    if name == 'SGD':
        return optim.SGD(params, lr=lr, weight_decay=weight_decay, momentum=0.9)
    if name == 'Adam':
        return optim.Adam(params, lr=lr, weight_decay=weight_decay)
    if name == 'AdamW':
        return optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    raise ValueError(f"Unknown optimizer: {name}")

def get_scheduler(name, optimizer):
    if name == 'StepLR':
        return optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    if name == 'CosineAnnealing':
        return optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)
    if name == 'ReduceLROnPlateau':
        return optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)
    return None

def compute_test_metrics(model, loader, device):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            _, preds = outputs.max(1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    precision = precision_score(all_labels, all_preds, average='macro', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='macro', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    return precision, recall, f1


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    train_root = '/home/atulk441/scratch/seg_train/seg_train'
    test_root  = '/home/atulk441/scratch/seg_test/seg_test'
    transform = transforms.Compose([transforms.Resize((150, 150)), transforms.ToTensor()])

    full_train = datasets.ImageFolder(root=train_root, transform=transform)
    test_ds    = datasets.ImageFolder(root=test_root, transform=transform)

    total = len(full_train)
    val_size = int(0.2 * total)
    train_size = total - val_size
    train_ds, val_ds = random_split(full_train, [train_size, val_size], generator=torch.Generator().manual_seed(42))

    num_classes = len(full_train.classes)

    param_grid = {
        'lr': [0.001],
        'optimizer': ['SGD', 'Adam', 'AdamW'],
        'num_filters': [[32, 64, 128, 256, 128, 64, 32]],
        'dropout': [0.5],
        'batch_norm': [True, False],
        'pooling': ['none', 'max', 'avg'],
    }


    batch_size = 128
    weight_decay = 1e-4
    scheduler_name = 'ReduceLROnPlateau'
    number_blocks = len(param_grid['num_filters'][0])
    dir_name = f"results_{number_blocks}_dr_{param_grid['dropout'][0]}"
    os.makedirs(dir_name, exist_ok=True)

    global_best_val = 0
    global_best_run = None

    for setting in product(*param_grid.values()):
        config = dict(zip(param_grid.keys(), setting))

        run_id = (
            f"nb-{number_blocks}_bn-{config['batch_norm']}_"
            f"opt-{config['optimizer']}_"
            f"pool-{config['pooling']}_dr-{config['dropout']}"
        )
        csv_file = f"{dir_name}/run_{run_id}.csv"

        with open(csv_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['epoch', 'train_loss', 'train_acc', 'val_loss', 'val_acc'])


        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False)
        test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False)


        model = SimpleCNN(num_classes,
                          filters_list=config['num_filters'],
                          dropout=config['dropout'],
                          use_batchnorm=config['batch_norm'],
                          pooling=config['pooling']).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = get_optimizer(config['optimizer'], model.parameters(), config['lr'], weight_decay)
        scheduler = get_scheduler(scheduler_name, optimizer)

        best_val = 0
        best_val_epoch = 0
        best_train_acc_at_val = 0
        patience, wait = 5, 0

        for epoch in range(1, 51):
            train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
            val_loss, val_acc     = eval_one_epoch(model, val_loader, criterion, device)

            with open(csv_file, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([epoch, train_loss, train_acc, val_loss, val_acc])

            print(
                f"Run {run_id} | Epoch {epoch:02d} | "
                f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | "
                f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}"
            )

            if val_acc > best_val:
                best_val = val_acc
                best_val_epoch = epoch
                best_train_acc_at_val = train_acc
                wait = 0
            else:
                wait += 1
                if wait >= patience:
                    print(f"Run {run_id}: Early stopping at epoch {epoch}")
                    break

            if scheduler:
                scheduler.step(val_acc)

        if best_val > global_best_val:
            global_best_val = best_val
            global_best_run = run_id
            print(f"New global best model saved from run {run_id} with val_acc={best_val:.4f}")

        _, test_acc = eval_one_epoch(model, test_loader, criterion, device)
        precision, recall, f1 = compute_test_metrics(model, test_loader, device)
        print(f"Run {run_id} Test Acc: {test_acc:.4f} | Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")

        global_csv = f'{dir_name}/all_best_results.csv'
        write_header = not os.path.exists(global_csv) or os.stat(global_csv).st_size == 0
        with open(global_csv, 'a', newline='') as f:
            writer = csv.writer(f)
            if write_header:
                writer.writerow([
                    'run_id', *config.keys(),
                    'best_val_acc', 'val_epoch', 'train_acc_at_best_val', 'test_acc',
                    'test_precision', 'test_recall', 'test_f1'
                ])

            writer.writerow([
                run_id, *config.values(),
                best_val, best_val_epoch, best_train_acc_at_val, test_acc,
                precision, recall, f1
            ])

        test_results_csv = f'{dir_name}/test_results.csv'
        test_write_header = not os.path.exists(test_results_csv) or os.stat(test_results_csv).st_size == 0
        with open(test_results_csv, 'a', newline='') as f:
            writer = csv.writer(f)
            if test_write_header:
                writer.writerow(['run_id', 'test_accuracy', 'precision', 'recall', 'f1_score'])
            writer.writerow([run_id, test_acc, precision, recall, f1])

    print(f"Global best run: {global_best_run} with val accuracy: {global_best_val:.4f}")

    import glob
    import pandas as pd
    import matplotlib.pyplot as plt
    all_files = glob.glob(f'{dir_name}/run_*.csv')
    plt.figure(figsize=(12, 8))
    for file in all_files:
        df = pd.read_csv(file)
        run_name = os.path.basename(file).replace('run_', '').replace('.csv', '')
        plt.plot(df['epoch'], df['val_acc'], label=run_name)
    plt.xlabel('Epoch')
    plt.ylabel('Validation Accuracy')
    plt.title('Validation Accuracy per Run (Hyperparameter Combinations)')
    plt.legend(fontsize='small', ncol=2)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'{dir_name}/val_acc_comparison.png')
    plt.show()
