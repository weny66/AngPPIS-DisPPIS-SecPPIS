import os
import csv
import random
import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader, Subset, WeightedRandomSampler
import torch.nn as nn
import torch.optim as optim

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

DISTANCE_DIR = "distances"
LABEL_DIR = "labels"

SEEDS = [42, 2024, 2025, 3407, 10086]
BASE_OUTPUT_DIR = "output_disppis_distance_smallmodel_groupnorm_5runs"
BATCH_SIZE = 32
NUM_EPOCHS = 15
LEARNING_RATE = 3e-5
WEIGHT_DECAY = 1e-3
TRAIN_RATIO = 0.7
VAL_RATIO = 0.2
TEST_RATIO = 0.1
DISTANCE_CLIP = 50.0
MAX_DISTANCE_VECTOR_LENGTH = None
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
    base = os.path.splitext(os.path.basename(filename))[0]
    return base


def build_metadata(distance_dir, label_dir):
    metadata = []
    protein_info = {}

    distance_files = sorted([f for f in os.listdir(distance_dir) if f.endswith(".npy")])

    skipped_no_label = 0
    skipped_shape_mismatch = 0

    max_len = 0

    for distance_file in distance_files:
        protein_id = extract_protein_id(distance_file)

        distance_path = os.path.join(distance_dir, distance_file)
        label_path = os.path.join(label_dir, distance_file)

        if not os.path.exists(label_path):
            skipped_no_label += 1
            continue

        labels = np.load(label_path)
        labels = np.asarray(labels).astype(int).reshape(-1)

        distance_matrix = np.load(distance_path, mmap_mode="r")

        if distance_matrix.ndim != 2 or distance_matrix.shape[0] != distance_matrix.shape[1]:
            print(f"Skipping non-square matrix: {distance_file}, shape={distance_matrix.shape}")
            skipped_shape_mismatch += 1
            continue

        n_atoms = distance_matrix.shape[0]

        if len(labels) != n_atoms:
            print(f"Length mismatch: {protein_id}, distance atoms={n_atoms}, labels={len(labels)}")
            n = min(n_atoms, len(labels))
            skipped_shape_mismatch += 1
        else:
            n = n_atoms

        max_len = max(max_len, n_atoms)

        count_0 = int(np.sum(labels[:n] == 0))
        count_1 = int(np.sum(labels[:n] == 1))

        protein_info[protein_id] = {
            "distance_path": distance_path,
            "label_path": label_path,
            "n_atoms": n_atoms,
            "n_samples": n,
            "label_0": count_0,
            "label_1": count_1
        }

        for atom_idx in range(n):
            label = int(labels[atom_idx])

            if label not in [0, 1]:
                raise ValueError(
                    f"Label must be 0 or 1, got {label} in {label_path}, index={atom_idx}"
                )

            metadata.append({
                "sample_id": f"{protein_id}_{atom_idx}",
                "protein_id": protein_id,
                "atom_idx": atom_idx,
                "distance_path": distance_path,
                "label_path": label_path,
                "label": label
            })

    print(f"Skipped distance files without labels: {skipped_no_label}")
    print(f"Skipped or adjusted shape mismatch files: {skipped_shape_mismatch}")
    print(f"Total samples: {len(metadata)}")
    print(f"Total proteins: {len(protein_info)}")
    print(f"Maximum atom length: {max_len}")

    if len(metadata) == 0:
        raise ValueError("No valid distance-label samples found.")

    if MAX_DISTANCE_VECTOR_LENGTH is None:
        target_len = max_len
    else:
        target_len = min(max_len, MAX_DISTANCE_VECTOR_LENGTH)

    print(f"Target distance vector length: {target_len}")

    return metadata, protein_info, target_len

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
            "atom_idx",
            "label",
            "split",
            "distance_path",
            "label_path"
        ])

        for idx, m in enumerate(metadata):
            writer.writerow([
                m["sample_id"],
                m["protein_id"],
                m["atom_idx"],
                m["label"],
                split_map[idx],
                m["distance_path"],
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


def compute_distance_statistics(distance_vector):

    values = np.asarray(distance_vector, dtype=np.float32)
    values = np.clip(values, 0.0, DISTANCE_CLIP)

    nonzero = values[values > 0]

    if nonzero.size == 0:
        nonzero = values

    stats = np.array([
        np.mean(nonzero),
        np.std(nonzero),
        np.min(nonzero),
        np.max(nonzero),
        np.median(nonzero),
        np.percentile(nonzero, 25),
        np.percentile(nonzero, 75),
    ], dtype=np.float32)

    stats = stats / DISTANCE_CLIP

    return stats


class DistanceVectorStatsDataset(Dataset):
    def __init__(self, metadata, target_len):
        self.metadata = metadata
        self.target_len = target_len
        self.matrix_cache = {}

    def __len__(self):
        return len(self.metadata)

    def _load_matrix(self, distance_path):
        if distance_path not in self.matrix_cache:
            self.matrix_cache[distance_path] = np.load(distance_path, mmap_mode="r")
        return self.matrix_cache[distance_path]

    def __getitem__(self, idx):
        item = self.metadata[idx]

        matrix = self._load_matrix(item["distance_path"])
        atom_idx = item["atom_idx"]

        distance_vector = np.asarray(matrix[:, atom_idx], dtype=np.float32)

        stats = compute_distance_statistics(distance_vector)
        stats = torch.tensor(stats, dtype=torch.float32)

        distance_vector = np.clip(distance_vector, 0.0, DISTANCE_CLIP)
        distance_vector = distance_vector / DISTANCE_CLIP

        current_len = len(distance_vector)

        if current_len >= self.target_len:
            distance_vector = distance_vector[:self.target_len]
        else:
            padded = np.ones(self.target_len, dtype=np.float32)
            padded[:current_len] = distance_vector
            distance_vector = padded

        distance_tensor = torch.tensor(distance_vector, dtype=torch.float32).unsqueeze(0)

        label = torch.tensor(int(item["label"]), dtype=torch.long)

        return distance_tensor, stats, label


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


class DistanceCNNStats(nn.Module):
    def __init__(self, stat_dim=7):
        super(DistanceCNNStats, self).__init__()

        self.features = nn.Sequential(
            nn.Conv1d(1, 8, kernel_size=7, padding=3),
            nn.GroupNorm(4, 8),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(4),

            nn.Conv1d(8, 16, kernel_size=5, padding=2),
            nn.GroupNorm(4, 16),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(4),

            nn.Conv1d(16, 32, kernel_size=5, padding=2),
            nn.GroupNorm(8, 32),
            nn.ReLU(inplace=True),

            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.GroupNorm(8, 64),
            nn.ReLU(inplace=True),

            nn.AdaptiveAvgPool1d(1)
        )

        self.stat_mlp = nn.Sequential(
            nn.Linear(stat_dim, 16),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2)
        )

        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(64 + 16, 32),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(32, 2)
        )

    def forward(self, distance_vector, stats):
        x = self.features(distance_vector)
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
        for distance_vectors, stats, labels in tqdm(dataloader, desc="Evaluating", leave=False):
            distance_vectors = distance_vectors.to(device)
            stats = stats.to(device)
            labels = labels.to(device)

            outputs = model(distance_vectors, stats)
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

    for distance_vectors, stats, labels in tqdm(dataloader, desc="Training", leave=False):
        distance_vectors = distance_vectors.to(device)
        stats = stats.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(distance_vectors, stats)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    avg_loss = running_loss / len(dataloader)
    return avg_loss


def run_one_seed(metadata, target_len, seed):
    print("\n" + "=" * 90)
    print(f"Running seed: {seed}")
    print("=" * 90)

    set_seed(seed)

    output_dir = os.path.join(BASE_OUTPUT_DIR, f"seed_{seed}")
    os.makedirs(output_dir, exist_ok=True)

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

    full_dataset = DistanceVectorStatsDataset(metadata, target_len=target_len)

    train_dataset = Subset(full_dataset, train_indices)
    val_dataset = Subset(full_dataset, val_indices)
    test_dataset = Subset(full_dataset, test_indices)

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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("\nUsing device:", device)

    model = DistanceCNNStats(stat_dim=STAT_DIM).to(device)

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
        patience=4,
        min_lr=1e-6
    )

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

    test_result_file = os.path.join(output_dir, "final_independent_test_results.txt")

    with open(test_result_file, "w", encoding="utf-8") as f:
        f.write("Final Independent Test Results\n")
        f.write("===================================\n")
        f.write(f"Seed: {seed}\n")
        f.write("Model: Small DistanceCNNStats trained from continuous atom-distance vectors\n")
        f.write("Stability strategy: small model, lr=3e-5, scheduler patience=4\n")
        f.write("Statistics: mean, std, min, max, median, q25, q75 from non-zero distance values\n")
        f.write(f"Distance clip: {DISTANCE_CLIP}\n")
        f.write(f"Target vector length: {target_len}\n")
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

    return result

def save_run_summary(results, output_dir):
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
        print("Run summary: mean ± SD")
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
    print("DisPPIS distance-vector small-model stability test")
    print("Strategy: seed=42, small model, lr=3e-5, sampler=0.25, scheduler patience=4")
    print("=" * 90)

    metadata, protein_info, target_len = build_metadata(DISTANCE_DIR, LABEL_DIR)

    all_results = []

    for seed in SEEDS:
        result = run_one_seed(metadata, target_len, seed)
        all_results.append(result)

    save_run_summary(all_results, BASE_OUTPUT_DIR)

    print("\nAll runs completed.")