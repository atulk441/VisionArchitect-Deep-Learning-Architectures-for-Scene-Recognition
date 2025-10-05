import os
import csv
from itertools import product
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from sklearn.metrics import precision_score, recall_score, f1_score

class Encoder(nn.Module):
    def __init__(self, use_batchnorm=True, dropout=0.3, pooling='max', lateral_channels=256):
        super().__init__()
        channels = [3, 32, 32, 64, 64, 128, 128, 256]
        dropout_idxs = {1, 4, 6}

        self.blocks = nn.ModuleList()
        self.out_chs = []
        in_ch = channels[0]
        for i, out_ch in enumerate(channels[1:]):
            layers = [nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)]
            if use_batchnorm:
                layers.append(nn.BatchNorm2d(out_ch))
            layers.append(nn.ReLU(inplace=True))
            if pooling != 'none' and i % 2 == 0:
                layers.append(nn.MaxPool2d(2) if pooling == 'max' else nn.AvgPool2d(2))
            if dropout > 0 and i in dropout_idxs:
                layers.append(nn.Dropout(dropout))
            block = nn.Sequential(*layers)
            self.blocks.append(block)
            self.out_chs.append(out_ch)
            in_ch = out_ch

        self.adaptive_pool = nn.AdaptiveAvgPool2d((8, 8))

        self.lateral_convs = nn.ModuleList([
            nn.Conv2d(c, lateral_channels, kernel_size=1) for c in self.out_chs
        ])
        self.smooth_convs = nn.ModuleList([
            nn.Conv2d(lateral_channels, lateral_channels, kernel_size=3, padding=1)
            for _ in self.out_chs
        ])
        self.lateral_channels = lateral_channels

    def forward(self, x):
        feats = []
        for block in self.blocks:
            x = block(x)
            feats.append(x)
        x_enc = self.adaptive_pool(x)

        lateral_feats = [l(feat) for l, feat in zip(self.lateral_convs, feats)]
        for i in range(len(lateral_feats) - 1, 0, -1):
            up = F.interpolate(
                lateral_feats[i],
                size=lateral_feats[i - 1].shape[-2:],
                mode='nearest'
            )
            lateral_feats[i - 1] += up
        smoothed = [s(f) for s, f in zip(self.smooth_convs, lateral_feats)]

        return x_enc, smoothed

class Decoder(nn.Module):
    def __init__(self, use_batchnorm=True, dropout=0.3, pooling='max'):
        super().__init__()
        channels = [256, 128, 128, 64, 64, 32, 32, 3]
        dropout_idxs = {1, 4, 6}
        layers = []
        for i in range(len(channels) - 1):
            layers.append(nn.ConvTranspose2d(channels[i], channels[i + 1], kernel_size=3, padding=1))
            if use_batchnorm and channels[i + 1] != 3:
                layers.append(nn.BatchNorm2d(channels[i + 1]))
            layers.append(nn.ReLU(inplace=True) if i < len(channels) - 2 else nn.Sigmoid())
            if pooling != 'none' and i % 2 == 0:
                if pooling == 'max':
                    layers.append(nn.Upsample(scale_factor=2, mode='nearest'))
                else:
                    layers.append(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False))
            if dropout > 0 and channels[i + 1] != 3 and i in dropout_idxs:
                layers.append(nn.Dropout(dropout))
        self.decoder = nn.Sequential(*layers)
        self.final_upsample = nn.Upsample(size=(150, 150), mode='bilinear', align_corners=False)

    def forward(self, x):
        x = self.decoder(x)
        return self.final_upsample(x)

class ClassifierHead(nn.Module):
    def __init__(self, in_channels, num_levels, num_classes, lateral_channels=256):
        super().__init__()
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        total_dim = lateral_channels * num_levels
        self.classifier = nn.Sequential(
            nn.Linear(total_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, fpn_feats):
        pooled = [self.global_pool(f).flatten(1) for f in fpn_feats]
        concat = torch.cat(pooled, dim=1)
        return self.classifier(concat)

def train_autoencoder_epoch(encoder, decoder, loader, criterion, optimizer, device):
    encoder.train(); decoder.train()
    total_loss = 0
    for imgs, _ in loader:
        imgs = imgs.to(device)
        optimizer.zero_grad()
        encoded, _ = encoder(imgs)
        decoded = decoder(encoded)
        loss = criterion(decoded, imgs)
        loss.backward(); optimizer.step()
        total_loss += loss.item() * imgs.size(0)
    return total_loss / len(loader.dataset)

def eval_autoencoder_epoch(encoder, decoder, loader, criterion, device):
    encoder.eval(); decoder.eval()
    total_loss = 0
    with torch.no_grad():
        for imgs, _ in loader:
            imgs = imgs.to(device)
            encoded, _ = encoder(imgs)
            decoded = decoder(encoded)
            loss = criterion(decoded, imgs)
            total_loss += loss.item() * imgs.size(0)
    return total_loss / len(loader.dataset)

def train_classifier_epoch(encoder, classifier, loader, criterion, optimizer, device):
    encoder.eval(); classifier.train()
    total_loss, correct = 0, 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        with torch.no_grad():
            _, fpn_feats = encoder(imgs)
        optimizer.zero_grad()
        outputs = classifier(fpn_feats)
        loss = criterion(outputs, labels)
        loss.backward(); optimizer.step()
        total_loss += loss.item() * imgs.size(0)
        preds = outputs.argmax(1)
        correct += preds.eq(labels).sum().item()
    return total_loss / len(loader.dataset), correct / len(loader.dataset)

def eval_classifier_epoch(encoder, classifier, loader, criterion, device):
    encoder.eval(); classifier.eval()
    total_loss, correct = 0, 0
    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            _, fpn_feats = encoder(imgs)
            outputs = classifier(fpn_feats)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * imgs.size(0)
            preds = outputs.argmax(1)
            correct += preds.eq(labels).sum().item()
    return total_loss / len(loader.dataset), correct / len(loader.dataset)

def compute_test_metrics(encoder, classifier, loader, device):
    encoder.eval(); classifier.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            _, fpn_feats = encoder(imgs)
            outputs = classifier(fpn_feats)
            preds = outputs.argmax(1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    from sklearn.metrics import precision_score, recall_score, f1_score
    precision = precision_score(all_labels, all_preds, average='macro', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='macro', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    return precision, recall, f1


def get_optimizer(name, params, lr, weight_decay):
    if name == 'SGD':
        return optim.SGD(params, lr=lr, weight_decay=weight_decay, momentum=0.9)
    elif name == 'Adam':
        return optim.Adam(params, lr=lr, weight_decay=weight_decay)
    elif name == 'AdamW':
        return optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    else:
        raise ValueError(f"Unknown optimizer: {name}")

def get_scheduler(name, optimizer, mode='max'):
    if name == 'StepLR':
        return optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    if name == 'CosineAnnealing':
        return optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)
    if name == 'ReduceLROnPlateau':
        return optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode=mode, factor=0.5, patience=3)
    return None


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

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
    batch_size = 128
    weight_decay = 1e-4
    num_epochs_ae = 50
    num_epochs_cls = 50
    scheduler_name = 'ReduceLROnPlateau'

    param_grid = {
        'lr': [0.001],
        'optimizer': ['Adam'],
        'dropout': [0.1],
        'batch_norm': [True],
        'pooling': ['max'],
    }

    dir_name = f"ae_cnn_results"
    os.makedirs(dir_name, exist_ok=True)
    global_best_val = 0
    global_best_run = None

    for setting in product(*param_grid.values()):
        config = dict(zip(param_grid.keys(), setting))
        run_id = f"opt-{config['optimizer']}_bn-{config['batch_norm']}_pool-{config['pooling']}_dr-{config['dropout']}"

        encoder = Encoder(
            use_batchnorm=config['batch_norm'],
            pooling=config['pooling'],
            dropout=config['dropout']
        ).to(device)

        decoder = Decoder(
            use_batchnorm=config['batch_norm'],
            pooling=config['pooling'],
            dropout=config['dropout']
        ).to(device)

        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        val_loader   = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
        test_loader  = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

        sample_batch, _ = next(iter(train_loader))
        sample_batch = sample_batch.to(device)
        with torch.no_grad():
            _, fpn_feats = encoder(sample_batch)
        num_levels = len(fpn_feats)
        classifier = ClassifierHead(in_channels=encoder.lateral_channels,
                                 num_levels=num_levels,
                                 num_classes=num_classes).to(device)

        ae_params = list(encoder.parameters()) + list(decoder.parameters())
        optimizer_ae = get_optimizer(config['optimizer'], ae_params, config['lr'], weight_decay)
        scheduler_ae = get_scheduler(scheduler_name, optimizer_ae, mode='min')
        criterion_ae = nn.MSELoss()
        
        ae_csv_file = f"{dir_name}/ae_{run_id}.csv"
        with open(ae_csv_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['epoch', 'train_loss', 'val_loss'])

            ae_best_val_loss = float('inf')
            ae_wait = 0
            ae_patience = 5

            for epoch in range(num_epochs_ae):
                loss_ae = train_autoencoder_epoch(encoder, decoder, train_loader, criterion_ae, optimizer_ae, device)
                val_loss_ae = eval_autoencoder_epoch(encoder, decoder, val_loader, criterion_ae, device)
                writer.writerow([epoch + 1, loss_ae, val_loss_ae])
                print(f"AE Epoch {epoch+1}: Train Loss={loss_ae:.4f} | Val Loss={val_loss_ae:.4f}")
                if val_loss_ae < ae_best_val_loss:
                    ae_best_val_loss = val_loss_ae
                    ae_wait = 0
                else:
                    ae_wait += 1
                    if ae_wait >= ae_patience:
                        print("Early stopping Triggered for AE")
                        break
                if scheduler_ae:
                    scheduler_ae.step(val_loss_ae)

        for param in encoder.parameters():
            param.requires_grad = False

        optimizer_cls = get_optimizer(config['optimizer'], classifier.parameters(), config['lr'], weight_decay)
        scheduler_cls = get_scheduler(scheduler_name, optimizer_cls, mode='max')
        criterion_cls = nn.CrossEntropyLoss()

        csv_file = f"{dir_name}/run_{run_id}.csv"
        with open(csv_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['epoch', 'train_loss', 'train_acc', 'val_loss', 'val_acc'])

        best_val = 0
        best_val_epoch = 0
        best_train_acc_at_val = 0
        patience, wait = 5, 0

        for epoch in range(1, num_epochs_cls + 1):
            train_loss, train_acc = train_classifier_epoch(encoder, classifier, train_loader, criterion_cls, optimizer_cls, device)
            val_loss, val_acc = eval_classifier_epoch(encoder, classifier, val_loader, criterion_cls, device)

            with open(csv_file, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([epoch, train_loss, train_acc, val_loss, val_acc])

            print(f"Run {run_id} | Epoch {epoch:02d} | Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

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
            if scheduler_cls:
                scheduler_cls.step(val_acc)

        if best_val > global_best_val:
            global_best_val = best_val
            global_best_run = run_id

            print(f"{run_id} with val_acc={best_val:.4f}")

        _, test_acc = eval_classifier_epoch(encoder, classifier, test_loader, criterion_cls, device)
        precision, recall, f1 = compute_test_metrics(encoder, classifier, test_loader, device)
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
    plt.title('Classifier Validation Accuracy per Run (Hyperparameter Combinations)')
    plt.legend(fontsize='small', ncol=2)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'{dir_name}/clf_val_acc_comparison.png')
    plt.show()


    all_files = glob.glob(f'{dir_name}/ae_*.csv')
    plt.figure(figsize=(12, 8))
    for file in all_files:
        df = pd.read_csv(file)
        run_name = os.path.basename(file).replace('ae_', '').replace('.csv', '')
        plt.plot(df['epoch'], df['val_loss'], label=run_name)
    plt.xlabel('Epoch')
    plt.ylabel('Validation Loss')
    plt.title('AutoEncoder Validation Loss per Run (Hyperparameter Combinations)')
    plt.legend(fontsize='small', ncol=2)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'{dir_name}/ae_val_loss_comparison.png')
    plt.show()