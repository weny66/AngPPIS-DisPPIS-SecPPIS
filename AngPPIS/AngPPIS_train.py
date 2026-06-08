import os
import re
import csv
import random
import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader, Subset, WeightedRandomSampler
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    matthews_corrcoef,
    balanced_accuracy_score,
    confusion_matrix,
)

from tqdm import tqdm
MATRIX_DIR = "input"
LABEL_DIR = "lab_raw"

BASE_OUTPUT_DIR = "output_smallcnn_npy_stats_5runs"

SEEDS = [42, 2024, 2025, 3407, 10086]

BATCH_SIZE = 32
NUM_EPOCHS = 15

LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-3

TRAIN_RATIO = 0.7
VAL_RATIO = 0.2
TEST_RATIO = 0.1

TARGET_SIZE = 224

SAMPLER_POWER = 0.25
STAT_DIM = 7

os.makedirs(BASE_OUTPUT_DIR, exist_ok=True)

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def extract_protein_id(filename):
    """
    例如：
        1AY7A_0.npy -> 1AY7A
        protein_x_y_123.npy -> protein_x_y
    """
    base = os.path.splitext(os.path.basename(filename))[0]
    protein_id = re.sub(r"_\d+$", "", base)
    return protein_id

def build_metadata(matrix_dir, label_dir):
    metadata = []

    matrix_files = sorted([f for f in os.listdir(matrix_dir) if f.endswith(".npy")])

    skipped_no_label = 0

    for matrix_file in matrix_files:
        sample_id = os.path.splitext(matrix_file)[0]
        label_file = sample_id + ".npy"

        matrix_path = os.path.join(matrix_dir, matrix_file)
        label_path = os.path.join(label_dir, label_file)

        if not os.path.exists(label_path):
            skipped_no_label += 1
            continue

        label = int(np.squeeze(np.load(label_path)))

        if label not in [0, 1]:
            raise ValueError(f"Label must be 0 or 1, got {label} in {label_path}")

        metadata.append({
            "sample_id": sample_id,
            "protein_id": extract_protein_id(matrix_file),
            "matrix_path": matrix_path,
            "label_path": label_path,
            "label": label
        })

    if len(metadata) == 0:
        raise ValueError("No matched matrix-label pairs found.")

    print(f"Skipped matrix files without labels: {skipped_no_label}")
    print(f"Total samples: {len(metadata)}")
    print(f"Total proteins: {len(set([m['protein_id'] for m in metadata]))}")

    return metadata

def split_by_protein(metadata, train_ratio=0.7, val_ratio=0.2, test_ratio=0.1, seed=42):
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6

    protein_ids = sorted(list(set([m["protein_id"] for m in metadata])))

    random.seed(seed)
    random.shuffle(protein_ids)

    n_total = len(protein_ids)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)

    train_proteins = set(protein_ids[:n_train])
    val_proteins = set(protein_ids[n_train:n_train + n_val])
    test_proteins = set(protein_ids[n_train + n_val:])

    train_indices = []
    val_indices = []
    test_indices = []

    for idx, m in enumerate(metadata):
        if m["protein_id"] in train_proteins:
            train_indices.append(idx)
        elif m["protein_id"] in val_proteins:
            val_indices.append(idx)
        elif m["protein_id"] in test_proteins:
            test_indices.append(idx)
        else:
            raise RuntimeError("Protein ID not assigned.")

    return train_indices, val_indices, test_indices, train_proteins, val_proteins, test_proteins

def save_split_metadata(metadata, train_indices, val_indices, test_indices, output_dir):
    split_map = {}

    for idx in train_indices:
        split_map[idx] = "train"
    for idx in val_indices:
        split_map[idx] = "validation"
    for idx in test_indices:
        split_map[idx] = "test"

    split_file = os.path.join(output_dir, "protein_level_split_metadata.csv")

    with open(split_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "sample_id",
            "protein_id",
            "label",
            "split",
            "matrix_path",
            "label_path"
        ])

        for idx, m in enumerate(metadata):
            writer.writerow([
                m["sample_id"],
                m["protein_id"],
                m["label"],
                split_map[idx],
                m["matrix_path"],
                m["label_path"]
            ])

    print(f"Saved split metadata to: {split_file}")

def summarize_split(metadata, indices, split_name):
    labels = [metadata[i]["label"] for i in indices]
    proteins = set([metadata[i]["protein_id"] for i in indices])

    count_0 = labels.count(0)
    count_1 = labels.count(1)
    total = count_0 + count_1

    print(f"\n[{split_name}]")
    print(f"Proteins: {len(proteins)}")
    print(f"Samples: {len(indices)}")
    print(f"Label 0: {count_0}")
    print(f"Label 1: {count_1}")
    print(f"Positive ratio: {count_1 / total:.4f}")


def compute_angle_statistics(matrix):
    matrix = np.asarray(matrix, dtype=np.float32)
    matrix = np.clip(matrix, 0.0, 180.0)

    values = matrix[matrix > 0]

    if values.size == 0:
        values = matrix.flatten()

    stats = np.array([
        np.mean(values),
        np.std(values),
        np.min(values),
        np.max(values),
        np.median(values),
        np.percentile(values, 25),
        np.percentile(values, 75)
    ], dtype=np.float32)

    stats = stats / 180.0

    return stats

class ProteinMatrixStatsDataset(Dataset):
    def __init__(self, metadata, target_size=224):
        self.metadata = metadata
        self.target_size = target_size

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        item = self.metadata[idx]
        matrix = np.load(item["matrix_path"]).astype(np.float32)
        stats = compute_angle_statistics(matrix)
        stats = torch.tensor(stats, dtype=torch.float32)
        matrix = np.clip(matrix, 0.0, 180.0)
        matrix = matrix / 180.0
        matrix = torch.tensor(matrix, dtype=torch.float32).unsqueeze(0)
        matrix = F.interpolate(
            matrix.unsqueeze(0),
            size=(self.target_size, self.target_size),
            mode="bilinear",
            align_corners=False
        ).squeeze(0)
        label = torch.tensor(int(item["label"]), dtype=torch.long)

        return matrix, stats, label

def create_weighted_sampler(metadata, indices, power=0.25, seed=42):
    labels = np.array([metadata[i]["label"] for i in indices])
    class_counts = np.bincount(labels, minlength=2)

    print("\nTraining class counts for WeightedRandomSampler:", class_counts)

    if class_counts[0] == 0 or class_counts[1] == 0:
        raise ValueError("Both classes must exist in training set.")

    class_sampling_weights = (1.0 / class_counts) ** power
    sample_weights = np.array([class_sampling_weights[label] for label in labels])
    sample_weights = torch.DoubleTensor(sample_weights)

    generator = torch.Generator()
    generator.manual_seed(seed)

    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True,
        generator=generator
    )

    return sampler


def check_sampler_distribution(train_sampler, metadata, train_indices, max_samples=5000):
    sampled_indices = list(iter(train_sampler))[:max_samples]
    sampled_labels = [metadata[train_indices[i]]["label"] for i in sampled_indices]

    count_0 = sampled_labels.count(0)
    count_1 = sampled_labels.count(1)
    total = count_0 + count_1

    print("\nApproximate sampled training distribution:")
    print(f"Sampled Label 0: {count_0}")
    print(f"Sampled Label 1: {count_1}")
    print(f"Sampled positive ratio: {count_1 / total:.4f}")

class SmallCNNMatrixStats(nn.Module):
    def __init__(self, stat_dim=7):
        super(SmallCNNMatrixStats, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.AdaptiveAvgPool2d((1, 1))
        )

        self.stat_mlp = nn.Sequential(
            nn.Linear(stat_dim, 16),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2)
        )

        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(128 + 16, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(64, 2)
        )

    def forward(self, matrix, stats):
        x = self.features(matrix)
        x = torch.flatten(x, 1)

        s = self.stat_mlp(stats)

        combined = torch.cat([x, s], dim=1)

        out = self.classifier(combined)

        return out

def evaluate(model, dataloader, device, criterion, threshold=0.5, return_probs=False):
    model.eval()

    y_true = []
    y_prob = []
    total_loss = 0.0

    with torch.no_grad():
        for matrices, stats, labels in tqdm(dataloader, desc="Evaluating", leave=False):
            matrices = matrices.to(device)
            stats = stats.to(device)
            labels = labels.to(device)

            outputs = model(matrices, stats)
            loss = criterion(outputs, labels)
            total_loss += loss.item()

            probs = torch.softmax(outputs, dim=1)[:, 1]

            y_true.extend(labels.cpu().numpy())
            y_prob.extend(probs.cpu().numpy())

    y_true = np.array(y_true)
    y_prob = np.array(y_prob)

    y_pred = (y_prob >= threshold).astype(int)

    avg_loss = total_loss / len(dataloader)

    accuracy = accuracy_score(y_true, y_pred)
    precision_pos = precision_score(y_true, y_pred, pos_label=1, zero_division=0)
    recall_pos = recall_score(y_true, y_pred, pos_label=1, zero_division=0)
    f1_pos = f1_score(y_true, y_pred, pos_label=1, zero_division=0)
    balanced_acc = balanced_accuracy_score(y_true, y_pred)
    mcc = matthews_corrcoef(y_true, y_pred)

    if len(np.unique(y_true)) == 2:
        auroc = roc_auc_score(y_true, y_prob)
        auprc = average_precision_score(y_true, y_prob)
    else:
        auroc = np.nan
        auprc = np.nan

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

    metrics = {
        "loss": avg_loss,
        "accuracy": accuracy,
        "precision_pos": precision_pos,
        "recall_pos": recall_pos,
        "f1_pos": f1_pos,
        "balanced_accuracy": balanced_acc,
        "auroc": auroc,
        "auprc": auprc,
        "mcc": mcc,
        "confusion_matrix": cm,
        "threshold": threshold
    }

    if return_probs:
        metrics["y_true"] = y_true
        metrics["y_prob"] = y_prob

    return metrics

def find_best_threshold(y_true, y_prob, metric="mcc"):
    thresholds = np.arange(0.05, 0.96, 0.01)

    best_threshold = 0.5
    best_score = -999

    for threshold in thresholds:
        y_pred = (y_prob >= threshold).astype(int)

        if metric == "mcc":
            score = matthews_corrcoef(y_true, y_pred)
        elif metric == "f1":
            score = f1_score(y_true, y_pred, pos_label=1, zero_division=0)
        elif metric == "balanced_accuracy":
            score = balanced_accuracy_score(y_true, y_pred)
        else:
            raise ValueError("metric must be 'mcc', 'f1', or 'balanced_accuracy'")

        if score > best_score:
            best_score = score
            best_threshold = threshold

    return best_threshold, best_score

def train_one_epoch(model, dataloader, optimizer, criterion, device):
    model.train()

    running_loss = 0.0

    for matrices, stats, labels in tqdm(dataloader, desc="Training", leave=False):
        matrices = matrices.to(device)
        stats = stats.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(matrices, stats)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    avg_loss = running_loss / len(dataloader)
    return avg_loss

def run_one_seed(metadata, seed):
    print("\n" + "=" * 90)
    print(f"Running seed: {seed}")
    print("=" * 90)

    set_seed(seed)

    output_dir = os.path.join(BASE_OUTPUT_DIR, f"seed_{seed}")
    os.makedirs(output_dir, exist_ok=True)

    # protein-level split
    train_indices, val_indices, test_indices, train_proteins, val_proteins, test_proteins = split_by_protein(
        metadata,
        train_ratio=TRAIN_RATIO,
        val_ratio=VAL_RATIO,
        test_ratio=TEST_RATIO,
        seed=seed
    )

    save_split_metadata(metadata, train_indices, val_indices, test_indices, output_dir)

    summarize_split(metadata, train_indices, "Train")
    summarize_split(metadata, val_indices, "Validation")
    summarize_split(metadata, test_indices, "Independent Test")

    assert train_proteins.isdisjoint(val_proteins)
    assert train_proteins.isdisjoint(test_proteins)
    assert val_proteins.isdisjoint(test_proteins)

    print("\nProtein-level split check passed: no protein overlap among train, validation, and test sets.")

    # Dataset
    full_dataset = ProteinMatrixStatsDataset(metadata, target_size=TARGET_SIZE)

    train_dataset = Subset(full_dataset, train_indices)
    val_dataset = Subset(full_dataset, val_indices)
    test_dataset = Subset(full_dataset, test_indices)

    # Sampler
    train_sampler = create_weighted_sampler(
        metadata,
        train_indices,
        power=SAMPLER_POWER,
        seed=seed
    )

    check_sampler_distribution(train_sampler, metadata, train_indices)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        sampler=train_sampler,
        shuffle=False,
        num_workers=0
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )

    # Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("\nUsing device:", device)

    model = SmallCNNMatrixStats(stat_dim=STAT_DIM).to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=2,
        min_lr=1e-6
    )

    # Log
    log_file = os.path.join(output_dir, "training_log.csv")

    with open(log_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "epoch",
            "learning_rate",
            "train_loss",
            "val_loss",
            "val_accuracy",
            "val_precision_pos",
            "val_recall_pos",
            "val_f1_pos",
            "val_balanced_accuracy",
            "val_auroc",
            "val_auprc",
            "val_mcc",
            "val_threshold"
        ])

    best_val_auprc = -1
    best_model_path = os.path.join(output_dir, "best_model_by_val_auprc.pth")

    for epoch in range(1, NUM_EPOCHS + 1):
        print(f"\nSeed {seed} | Epoch {epoch}/{NUM_EPOCHS}")

        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            device
        )

        val_metrics = evaluate(
            model,
            val_loader,
            device,
            criterion,
            threshold=0.5,
            return_probs=False
        )

        scheduler.step(val_metrics["auprc"])
        current_lr = optimizer.param_groups[0]["lr"]

        print(f"Train Loss: {train_loss:.4f}")
        print(f"Current learning rate: {current_lr:.8f}")
        print(
            f"Validation | "
            f"Loss: {val_metrics['loss']:.4f} | "
            f"Acc: {val_metrics['accuracy']:.4f} | "
            f"Precision+: {val_metrics['precision_pos']:.4f} | "
            f"Recall+: {val_metrics['recall_pos']:.4f} | "
            f"F1+: {val_metrics['f1_pos']:.4f} | "
            f"AUPRC: {val_metrics['auprc']:.4f} | "
            f"AUROC: {val_metrics['auroc']:.4f} | "
            f"MCC: {val_metrics['mcc']:.4f}"
        )

        with open(log_file, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                epoch,
                f"{current_lr:.8f}",
                f"{train_loss:.6f}",
                f"{val_metrics['loss']:.6f}",
                f"{val_metrics['accuracy']:.6f}",
                f"{val_metrics['precision_pos']:.6f}",
                f"{val_metrics['recall_pos']:.6f}",
                f"{val_metrics['f1_pos']:.6f}",
                f"{val_metrics['balanced_accuracy']:.6f}",
                f"{val_metrics['auroc']:.6f}",
                f"{val_metrics['auprc']:.6f}",
                f"{val_metrics['mcc']:.6f}",
                f"{val_metrics['threshold']:.2f}"
            ])

        if not np.isnan(val_metrics["auprc"]) and val_metrics["auprc"] > best_val_auprc:
            best_val_auprc = val_metrics["auprc"]
            torch.save(model.state_dict(), best_model_path)
            print(f"Best model saved based on validation AUPRC: {best_val_auprc:.4f}")

    # Final test
    print("\nLoading best model for final independent test evaluation...")
    model.load_state_dict(torch.load(best_model_path, map_location=device))

    val_metrics_for_threshold = evaluate(
        model,
        val_loader,
        device,
        criterion,
        threshold=0.5,
        return_probs=True
    )

    best_threshold, best_threshold_score = find_best_threshold(
        val_metrics_for_threshold["y_true"],
        val_metrics_for_threshold["y_prob"],
        metric="mcc"
    )

    print(f"\nBest threshold selected on validation set: {best_threshold:.2f}")
    print(f"Best validation MCC at selected threshold: {best_threshold_score:.4f}")

    test_metrics = evaluate(
        model,
        test_loader,
        device,
        criterion,
        threshold=best_threshold,
        return_probs=False
    )

    print("\nFinal Independent Test Results")
    print("===================================")
    print(f"Seed: {seed}")
    print(f"Selected threshold from validation set: {best_threshold:.2f}")
    print(f"Validation MCC at selected threshold: {best_threshold_score:.4f}")
    print(f"Test Loss: {test_metrics['loss']:.4f}")
    print(f"Accuracy: {test_metrics['accuracy']:.4f}")
    print(f"Positive Precision: {test_metrics['precision_pos']:.4f}")
    print(f"Positive Recall: {test_metrics['recall_pos']:.4f}")
    print(f"Positive F1-score: {test_metrics['f1_pos']:.4f}")
    print(f"Balanced Accuracy: {test_metrics['balanced_accuracy']:.4f}")
    print(f"AUROC: {test_metrics['auroc']:.4f}")
    print(f"AUPRC: {test_metrics['auprc']:.4f}")
    print(f"MCC: {test_metrics['mcc']:.4f}")
    print("Confusion Matrix [[TN, FP], [FN, TP]]:")
    print(test_metrics["confusion_matrix"])

    test_result_file = os.path.join(output_dir, "final_independent_test_results.txt")

    with open(test_result_file, "w", encoding="utf-8") as f:
        f.write("Final Independent Test Results\n")
        f.write("===================================\n")
        f.write(f"Seed: {seed}\n")
        f.write("Model: SmallCNNMatrixStats trained from normalized float .npy angular matrices\n")
        f.write("Statistics: mean, std, min, max, median, q25, q75 from non-zero matrix elements\n")
        f.write(f"Input: {MATRIX_DIR}/*.npy\n")
        f.write(f"Sampler power: {SAMPLER_POWER}\n")
        f.write(f"Best validation AUPRC: {best_val_auprc:.6f}\n")
        f.write(f"Selected threshold from validation set: {best_threshold:.2f}\n")
        f.write(f"Validation MCC at selected threshold: {best_threshold_score:.6f}\n")
        f.write(f"Test Loss: {test_metrics['loss']:.6f}\n")
        f.write(f"Accuracy: {test_metrics['accuracy']:.6f}\n")
        f.write(f"Positive Precision: {test_metrics['precision_pos']:.6f}\n")
        f.write(f"Positive Recall: {test_metrics['recall_pos']:.6f}\n")
        f.write(f"Positive F1-score: {test_metrics['f1_pos']:.6f}\n")
        f.write(f"Balanced Accuracy: {test_metrics['balanced_accuracy']:.6f}\n")
        f.write(f"AUROC: {test_metrics['auroc']:.6f}\n")
        f.write(f"AUPRC: {test_metrics['auprc']:.6f}\n")
        f.write(f"MCC: {test_metrics['mcc']:.6f}\n")
        f.write("Confusion Matrix [[TN, FP], [FN, TP]]:\n")
        f.write(str(test_metrics["confusion_matrix"]))
        f.write("\n")

    result = {
        "seed": seed,
        "best_val_auprc": best_val_auprc,
        "selected_threshold": best_threshold,
        "val_mcc_at_threshold": best_threshold_score,
        "test_loss": test_metrics["loss"],
        "test_accuracy": test_metrics["accuracy"],
        "test_precision_pos": test_metrics["precision_pos"],
        "test_recall_pos": test_metrics["recall_pos"],
        "test_f1_pos": test_metrics["f1_pos"],
        "test_balanced_accuracy": test_metrics["balanced_accuracy"],
        "test_auroc": test_metrics["auroc"],
        "test_auprc": test_metrics["auprc"],
        "test_mcc": test_metrics["mcc"],
        "tn": int(test_metrics["confusion_matrix"][0, 0]),
        "fp": int(test_metrics["confusion_matrix"][0, 1]),
        "fn": int(test_metrics["confusion_matrix"][1, 0]),
        "tp": int(test_metrics["confusion_matrix"][1, 1]),
    }

    return result

def save_5run_summary(results, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    all_runs_file = os.path.join(output_dir, "all_runs_results.csv")

    fieldnames = [
        "seed",
        "best_val_auprc",
        "selected_threshold",
        "val_mcc_at_threshold",
        "test_loss",
        "test_accuracy",
        "test_precision_pos",
        "test_recall_pos",
        "test_f1_pos",
        "test_balanced_accuracy",
        "test_auroc",
        "test_auprc",
        "test_mcc",
        "tn",
        "fp",
        "fn",
        "tp"
    ]

    with open(all_runs_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for r in results:
            writer.writerow(r)

    metric_cols = [
        "best_val_auprc",
        "val_mcc_at_threshold",
        "test_loss",
        "test_accuracy",
        "test_precision_pos",
        "test_recall_pos",
        "test_f1_pos",
        "test_balanced_accuracy",
        "test_auroc",
        "test_auprc",
        "test_mcc",
    ]

    summary_file = os.path.join(output_dir, "summary_mean_sd.csv")

    with open(summary_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "mean", "sd", "mean ± sd"])

        print("\n" + "=" * 90)
        print("5-run summary: mean ± SD")
        print("=" * 90)

        for col in metric_cols:
            values = np.array([r[col] for r in results], dtype=float)
            mean_val = np.mean(values)
            sd_val = np.std(values, ddof=1) if len(values) > 1 else 0.0

            mean_sd_text = f"{mean_val:.4f} ± {sd_val:.4f}"

            writer.writerow([
                col,
                f"{mean_val:.6f}",
                f"{sd_val:.6f}",
                mean_sd_text
            ])

            print(f"{col}: {mean_sd_text}")

    print(f"\nAll run results saved to: {all_runs_file}")
    print(f"Summary mean ± SD saved to: {summary_file}")

if __name__ == "__main__":
    print("=" * 90)
    print("SmallCNNMatrixStats 5-run protein-level evaluation")
    print("=" * 90)

    metadata = build_metadata(MATRIX_DIR, LABEL_DIR)

    all_results = []

    for seed in SEEDS:
        result = run_one_seed(metadata, seed)
        all_results.append(result)

    save_5run_summary(all_results, BASE_OUTPUT_DIR)

    print("\nAll 5 runs completed.")