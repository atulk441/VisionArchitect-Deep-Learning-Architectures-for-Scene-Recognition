import os
import csv
from itertools import product
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets
from sklearn.metrics import precision_score, recall_score, f1_score
import pandas as pd
import matplotlib.pyplot as plt
import glob
import clip

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss, correct, total = 0, 0, 0
    for imgs, labels in loader:
        imgs = imgs.to(device).float()
        labels = labels.to(device)
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
            imgs = imgs.to(device).float()
            labels = labels.to(device)
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
    if name == 'ReduceLROnPlateau':
        return optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=3)
    if name == 'StepLR':
        return optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    if name == 'CosineAnnealing':
        return optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)
    return None

def compute_test_metrics(model, loader, device):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(device).float()
            labels = labels.to(device)
            outputs = model(imgs)
            _, preds = outputs.max(1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    precision = precision_score(all_labels, all_preds, average='macro', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='macro', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    return precision, recall, f1

class CLIPFineTuner(nn.Module):
    def __init__(self, clip_model, num_classes):
        super().__init__()
        self.visual = clip_model.visual
        for param in self.visual.parameters():
            param.requires_grad = False  
        self.classifier = nn.Linear(self.visual.output_dim, num_classes)

    def forward(self, x):
        with torch.no_grad():
            x = self.visual(x)
        return self.classifier(x)

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    train_root = '/home/atulk441/scratch/seg_train/seg_train'
    test_root  = '/home/atulk441/scratch/seg_test/seg_test'

    clip_model, preprocess = clip.load("ViT-L/14@336px", device=device, download_root='/home/atulk441/scratch/my_clip_models')
    clip_model = clip_model.float()

    full_train = datasets.ImageFolder(root=train_root, transform=preprocess)
    test_ds = datasets.ImageFolder(root=test_root, transform=preprocess)

    total = len(full_train)
    val_size = int(0.2 * total)
    train_size = total - val_size
    train_ds, val_ds = random_split(full_train, [train_size, val_size], generator=torch.Generator().manual_seed(42))

    num_classes = len(full_train.classes)

    param_grid = {
        'lr': [0.001],
        'optimizer': ['Adam'],
        'weight_decay': [1e-4],
    }

    batch_size = 128
    scheduler_name = 'ReduceLROnPlateau'
    dir_name = f"clip_freeze_l14"
    os.makedirs(dir_name, exist_ok=True)

    global_best_val = 0
    global_best_run = None

    for setting in product(*param_grid.values()):
        config = dict(zip(param_grid.keys(), setting))
        
        run_id = (
            f"lr-{config['lr']}_"
            f"opt-{config['optimizer']}_"
            f"wd-{config['weight_decay']}"
        )
        csv_file = f"{dir_name}/run_{run_id}.csv"

        with open(csv_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['epoch', 'train_loss', 'train_acc', 'val_loss', 'val_acc'])

        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

        model = CLIPFineTuner(clip_model, num_classes).to(device)

        criterion = nn.CrossEntropyLoss()
        optimizer = get_optimizer(config['optimizer'], model.parameters(), config['lr'], config['weight_decay'])
        scheduler = get_scheduler(scheduler_name, optimizer)

        best_val = 0
        best_val_epoch = 0
        best_train_acc_at_val = 0
        patience, wait = 5, 0

        for epoch in range(1, 51):
            train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
            val_loss, val_acc = eval_one_epoch(model, val_loader, criterion, device)

            with open(csv_file, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([epoch, train_loss, train_acc, val_loss, val_acc])

            print(
                f"Run {run_id} | Epoch {epoch:03d} | "
                f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | "
                f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}"
            )

            if scheduler:
                scheduler.step(val_acc)

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

        if best_val > global_best_val:
            global_best_val = best_val
            global_best_run = run_id

            print(f"New global best model from run {run_id} with val_acc={best_val:.4f}")

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

    all_files = glob.glob(f'{dir_name}/run_*.csv')
    plt.figure(figsize=(12, 8))
    for file in all_files:
        df = pd.read_csv(file)
        run_name = os.path.basename(file).replace('run_', '').replace('.csv', '')
        plt.plot(df['epoch'], df['val_acc'], label=run_name)
    plt.xlabel('Epoch')
    plt.ylabel('Validation Accuracy')
    plt.title('Validation Accuracy per Run (Hyperparameter Combinations)')
    plt.legend(fontsize='small', ncol=1)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'{dir_name}/val_acc_comparison.png')
    plt.show()
