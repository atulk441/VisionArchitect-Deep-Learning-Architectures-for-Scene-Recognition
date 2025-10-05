import os
import csv
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from sklearn.metrics import precision_score, recall_score, f1_score
from itertools import product


class SupConLoss(nn.Module):
    def __init__(self, temperature=0.07, contrast_mode='all', base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels):
        device = features.device
        batch_size = features.shape[0]
        labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels, labels.T).float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)

        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)

        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        mask = mask.repeat(anchor_count, contrast_count)
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        mask_pos_pairs = mask.sum(1)
        mask_pos_pairs = torch.where(mask_pos_pairs < 1e-6, 1, mask_pos_pairs)
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask_pos_pairs

        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()
        return loss

class Encoder(nn.Module):
    def __init__(self, use_batchnorm=True, dropout=0.3, pooling='max'):
        super().__init__()
        channels = [3, 32, 32, 64, 64, 128]
        layers = []
        dropout_layers = {1, 4}

        for i in range(len(channels) - 1):
            layers.append(nn.Conv2d(channels[i], channels[i + 1], kernel_size=3, padding=1))
            if use_batchnorm:
                layers.append(nn.BatchNorm2d(channels[i + 1]))
            layers.append(nn.ReLU(inplace=True))
            if pooling != 'none' and i % 2 == 0:
                if pooling == 'max':
                    layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
                elif pooling == 'avg':
                    layers.append(nn.AvgPool2d(kernel_size=2, stride=2))
            if dropout > 0 and i in dropout_layers:
                layers.append(nn.Dropout(dropout))
        self.encoder = nn.Sequential(*layers)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((8, 8))

    def forward(self, x):
        x = self.encoder(x)
        x = self.adaptive_pool(x)
        return x

class ProjectionHead(nn.Module):
    def __init__(self, in_dim, proj_dim=128):
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.ReLU(),
            nn.Linear(256, proj_dim)
        )

    def forward(self, x):
        x = x.view(x.size(0), -1)
        return F.normalize(self.projection(x), dim=1)

class ClassifierHead(nn.Module):
    def __init__(self, encoded_channels, encoded_spatial_dim, num_classes):
        super().__init__()
        self.flatten_dim = encoded_channels * encoded_spatial_dim * encoded_spatial_dim
        self.classifier = nn.Sequential(
            nn.Linear(self.flatten_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.classifier(x)
from torchvision import transforms

class MultiViewDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, n_views=2, transform=None):
        self.dataset = dataset
        self.n_views = n_views
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, label = self.dataset[idx]
        views = []

        for _ in range(self.n_views):
            if self.transform:
                views.append(self.transform(img))
            else:
                views.append(img)

        views = torch.stack(views, dim=0)  
        return views, label

def train_supcon_epoch(encoder, projector, dataloader, criterion, optimizer, device):
    encoder.train()
    projector.train()
    total_loss = 0.0
    for images, labels in train_loader:
        images = images.to(device) 
        labels = labels.to(device)
        
        batch_size, n_views, C, H, W = images.shape        
        images = images.view(batch_size * n_views, C, H, W)
        features = encoder(images)  
        features = features.view(batch_size, n_views, -1)
        projections = projector(features.view(batch_size * n_views, -1))
        projections = projections.view(batch_size, n_views, -1)
        loss = criterion(projections, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)

    return total_loss / len(dataloader.dataset)


def eval_supcon_epoch(encoder, projector, dataloader, criterion, device):
    encoder.eval()
    projector.eval()
    total_loss = 0.0

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)  
            labels = labels.to(device)

            batch_size, n_views, C, H, W = images.shape
            images = images.view(batch_size * n_views, C, H, W)
            features = encoder(images)
            features = features.view(batch_size, n_views, -1)
            projections = projector(features.view(batch_size * n_views, -1))
            projections = projections.view(batch_size, n_views, -1)
            loss = criterion(projections, labels)
            total_loss += loss.item() * batch_size  

    return total_loss / len(dataloader.dataset)

def train_classifier_epoch(encoder, classifier, loader, criterion, optimizer, device):
    encoder.eval()
    classifier.train()
    total_loss, correct = 0, 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        with torch.no_grad():
            encoded = encoder(imgs)
        optimizer.zero_grad()
        outputs = classifier(encoded)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * imgs.size(0)
        preds = outputs.argmax(1)
        correct += preds.eq(labels).sum().item()
    return total_loss / len(loader.dataset), correct / len(loader.dataset)

def eval_classifier_epoch(encoder, classifier, loader, criterion, device):
    encoder.eval()
    classifier.eval()
    total_loss, correct = 0, 0
    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            encoded = encoder(imgs)
            outputs = classifier(encoded)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * imgs.size(0)
            preds = outputs.argmax(1)
            correct += preds.eq(labels).sum().item()
    return total_loss / len(loader.dataset), correct / len(loader.dataset)

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


def compute_test_metrics(encoder, classifier, loader, device):
    encoder.eval()
    classifier.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            encoded = encoder(imgs)
            outputs = classifier(encoded)
            preds = outputs.argmax(1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    precision = precision_score(all_labels, all_preds, average='macro', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='macro', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    return precision, recall, f1

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    train_root = '/home/atulk441/scratch/seg_train/seg_train'
    test_root  = '/home/atulk441/scratch/seg_test/seg_test'

    transform = transforms.Compose([transforms.Resize((150, 150)), transforms.ToTensor()])

    transform_contrastive = transforms.Compose([
    transforms.RandomResizedCrop(150),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
    transforms.ToTensor(),
    ])

    base_dataset = datasets.ImageFolder(root=train_root, transform=None)
    multi_view_train = MultiViewDataset(base_dataset, n_views=2, transform=transform_contrastive)
    test_ds    = datasets.ImageFolder(root=test_root, transform=transform)

    total = len(multi_view_train)
    val_size = int(0.2 * total)
    train_size = total - val_size
    train_ds, val_ds = random_split(multi_view_train, [train_size, val_size], generator=torch.Generator().manual_seed(42))

    num_classes = len(base_dataset.classes)
    batch_size = 128
    weight_decay = 1e-4
    num_epochs_contrastive = 50
    num_epochs_cls = 50
    scheduler_name = 'ReduceLROnPlateau'

    param_grid = {
        'lr': [0.001],
        'optimizer': ['Adam', 'AdamW'],
        'dropout': [0.0, 0.2],
        'num_filters': [5],
        'batch_norm': [True],
        'pooling': ['max', 'avg'],
    }
    number_blocks = param_grid['num_filters'][0]
    dir_name = f"results_{number_blocks}"
    os.makedirs(dir_name, exist_ok=True)
    global_best_val = 0
    global_best_run = None

    for setting in product(*param_grid.values()):
        config = dict(zip(param_grid.keys(), setting))
        run_id = f"opt-{config['optimizer']}_nb-{number_blocks}_bn-{config['batch_norm']}_pool-{config['pooling']}_dr-{config['dropout']}"

        encoder = Encoder(
            use_batchnorm=config['batch_norm'],
            pooling=config['pooling'],
            dropout=config['dropout']
        ).to(device)

        with torch.no_grad():
            encoded_sample = encoder(torch.randn(1, 3, 150, 150).to(device))
        encoded_channels, encoded_spatial_dim = encoded_sample.shape[1], encoded_sample.shape[2]

        projector = ProjectionHead(encoded_channels * encoded_spatial_dim * encoded_spatial_dim).to(device)

        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        val_loader   = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
        test_loader  = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

        supcon_criterion = SupConLoss()
        contrastive_params = list(encoder.parameters()) + list(projector.parameters())
        optimizer_contrastive = get_optimizer(config['optimizer'], contrastive_params, config['lr'], weight_decay)
        scheduler_contrastive = get_scheduler(scheduler_name, optimizer_contrastive, mode='min')

        contrastive_csv_file = f"{dir_name}/contrastive_{run_id}.csv"
        with open(contrastive_csv_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['epoch', 'train_loss', 'val_loss'])

            best_val_loss = float('inf')
            wait = 0
            patience = 5

            for epoch in range(1, num_epochs_contrastive + 1):
                train_loss = train_supcon_epoch(encoder, projector, train_loader, supcon_criterion, optimizer_contrastive, device)
                val_loss = eval_supcon_epoch(encoder, projector, val_loader, supcon_criterion, device)

                writer.writerow([epoch, train_loss, val_loss])
                print(f"Contrastive Epoch {epoch}: Train Loss={train_loss:.4f} | Val Loss={val_loss:.4f}")

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    wait = 0
                else:
                    wait += 1
                    if wait >= patience:
                        print("Early stopping triggered for contrastive training")
                        break

                if scheduler_contrastive:
                    scheduler_contrastive.step(val_loss)


        transform_cls = transforms.Compose([
            transforms.Resize((150, 150)),
            transforms.ToTensor()
        ])
        full_train_cls = datasets.ImageFolder(root=train_root, transform=transform_cls)
        test_ds_cls    = datasets.ImageFolder(root=test_root, transform=transform_cls)
        train_ds_cls, val_ds_cls = random_split(full_train_cls, [train_size, val_size], generator=torch.Generator().manual_seed(42))

        train_loader = DataLoader(train_ds_cls, batch_size=batch_size, shuffle=True)
        val_loader   = DataLoader(val_ds_cls, batch_size=batch_size, shuffle=False)
        test_loader  = DataLoader(test_ds_cls, batch_size=batch_size, shuffle=False)

        for param in encoder.parameters():
            param.requires_grad = False

        classifier = ClassifierHead(encoded_channels, encoded_spatial_dim, num_classes).to(device)

        criterion_cls = nn.CrossEntropyLoss()
        optimizer_cls = get_optimizer(config['optimizer'], classifier.parameters(), config['lr'], weight_decay)
        scheduler_cls = get_scheduler(scheduler_name, optimizer_cls, mode='max')

        cls_csv_file = f"{dir_name}/run_{run_id}.csv"
        with open(cls_csv_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['epoch', 'train_loss', 'train_acc', 'val_loss', 'val_acc'])

        best_val = 0
        best_val_epoch = 0
        best_train_acc_at_val = 0
        patience, wait = 5, 0

        for epoch in range(1, num_epochs_cls + 1):
            train_loss, train_acc = train_classifier_epoch(encoder, classifier, train_loader, criterion_cls, optimizer_cls, device)
            val_loss, val_acc = eval_classifier_epoch(encoder, classifier, val_loader, criterion_cls, device)

            with open(cls_csv_file, 'a', newline='') as f:
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

    all_files = glob.glob(f'{dir_name}/contrastive_*.csv')
    plt.figure(figsize=(12, 8))
    for file in all_files:
        df = pd.read_csv(file)
        run_name = os.path.basename(file).replace('contrastive_', '').replace('.csv', '')
        plt.plot(df['epoch'], df['val_loss'], label=run_name)
    plt.xlabel('Epoch')
    plt.ylabel('Validation Loss')
    plt.title('Contrastive Validation Loss per Run (Hyperparameter Combinations)')
    plt.legend(fontsize='small', ncol=2)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'{dir_name}/contrastive_val_loss_comparison.png')
    plt.show()
