import torch
from torch import nn, optim
import argparse
from tqdm import tqdm

from dataloader import create_fake_dataloader, get_dataloader
from torch_geometric.data import DataLoader
from sklearn.model_selection import train_test_split
from torch.utils.tensorboard import SummaryWriter
import datetime
import os

from models import FuncGNN

torch.manual_seed(42)

parser = argparse.ArgumentParser(
    description="Protein Function Prediction with E(3)-Equivariant GNNs and Multi-task Learning"
)

# NUM_TASKS = 4598


def str_to_bool(value):
    if value.lower() in {"false", "f", "0", "no", "n"}:
        return False
    elif value.lower() in {"true", "t", "1", "yes", "y"}:
        return True
    raise ValueError(f"{value} is not a valid boolean value")


parser.add_argument(
    "--epochs",
    type=int,
    default=100,
    help="number of epochs to train (default: 10)",
)

parser.add_argument(
    "--batch_size",
    type=int,
    default=16,
    help="batch size",
)

parser.add_argument(
    "--num_layers",
    type=int,
    default=16,
    help="number of layers in spatial model",
)

parser.add_argument(
    "--feature_dim",
    type=int,
    default=11,
    help="feature dimension",
)

parser.add_argument(
    "--edge_dim",
    type=int,
    default=0,
    help="edge feature dimension",
)

parser.add_argument(
    "--hidden_dim",
    type=int,
    default=256,
    help="hidden dimension",
)

parser.add_argument(
    "--position_dim",
    type=int,
    default=3,
    help="dimension of positions of atoms",
)

parser.add_argument(
    "--task_embed_dim",
    type=int,
    default=128,
    help="task embedding latent dimension",
)

parser.add_argument(
    "--num_tasks",
    type=int,
    default=4598,
    help="number of tasks",
)

parser.add_argument(
    "--num_classes",
    type=int,
    default=3,
    help="number of classes",
)

parser.add_argument(
    "--model_type",
    type=str,
    default="egnn_t0",
    help="Model type, either EGNN or GAT (default: egnn)",
)

parser.add_argument("--weight_decay", type=float, default=1e-6, help="weight decay")

parser.add_argument("--lr", type=float, default=1e-4, help="learning rate")

parser.add_argument(
    "--tensorboard", type=str_to_bool, default=False, help="Uses tensorboard"
)

args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
loss_mse = nn.MSELoss()

DATASET_DIR = "data/processed_data/protein_inputs"
# DATASET_DIR = "temp_proteins"


def create_summary_writer(lr, weight_decay, hidden_size, num_equivariant_layers):
    dt = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = f"./runs/{dt}_funcgnn_lr_{lr}_wd_{weight_decay}_hid_size_{hidden_size}_num_equi_layers_{num_equivariant_layers}/"

    writer = SummaryWriter(log_dir)
    return writer


class EarlyStopper:
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0

        self.min_validation_loss = float("inf")
        self.last_validation_loss = float("inf")

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            print(f"Counter: {self.counter}")
            if self.counter >= self.patience:
                return True

        return False


def main():
    batch_size = args.batch_size
    num_layers = args.num_layers
    feature_dim = args.feature_dim
    edge_dim = args.edge_dim
    hidden_dim = args.hidden_dim
    task_embed_dim = args.task_embed_dim
    num_tasks = args.num_tasks
    num_classes = args.num_classes
    position_dim = args.position_dim
    model_type = args.model_type

    model = FuncGNN(
        num_layers,
        feature_dim,
        edge_dim,
        hidden_dim,
        task_embed_dim,
        num_tasks,
        position_dim,
        num_classes,
        model_type=model_type,
    ).to(device)

    protein_data, dl = get_dataloader(DATASET_DIR, batch_size=batch_size)
    # protein_data, dl = create_fake_dataloader(num_proteins=1000, num_tasks=4598)

    train_data, temp_data = train_test_split(
        protein_data, test_size=0.2, random_state=42
    )
    val_data, test_data = train_test_split(temp_data, test_size=0.5, random_state=42)

    # Step 2: Create DataLoader objects for each subset
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"total params: {total_params}")

    weight_decay = args.weight_decay
    lr = args.lr

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    if args.tensorboard:
        writer = create_summary_writer(lr, weight_decay, hidden_dim, num_layers)

    # # early stopping
    early_stopper = EarlyStopper(patience=10, min_delta=0.005)
    results = {"epochs": [], "losess": []}
    best_val_loss = float("inf")
    best_test_loss = float("inf")
    best_train_loss = float("inf")
    best_epoch = 0

    for epoch in range(0, args.epochs):
        train_loss = train(model, optimizer, epoch, train_loader)
        if args.tensorboard:
            writer.add_scalar("Loss/train", train_loss, epoch)

        val_loss = val(model, epoch, val_loader, "val")
        test_loss = val(model, epoch, test_loader, "test")

        if args.tensorboard:
            writer.add_scalar("Loss/val", val_loss, epoch)
            writer.add_scalar("Loss/test", test_loss, epoch)

        results["epochs"].append(epoch)
        results["losess"].append(test_loss)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_test_loss = test_loss
            best_train_loss = train_loss
            best_epoch = epoch
            # save model
            os.makedirs("best_models", exist_ok=True)

            torch.save(
                model.state_dict(),
                f"best_models/funcgnn.pt",
            )

        print(
            "*** Best Train Loss: %.5f \t Best Val Loss: %.5f \t Best Test Loss: %.5f \t Best epoch %d"
            % (best_train_loss, best_val_loss, best_test_loss, best_epoch)
        )

        if early_stopper.early_stop(val_loss):
            print(f"EARLY STOPPED")
            break

    return best_train_loss, best_val_loss, best_test_loss, best_epoch, total_params


def add_negative_samples(task_indices, labels, num_tasks=4598):
    new_task_indices = task_indices.clone()
    new_labels = labels.clone()
    new_labels[:, 1] = 2
    new_task_indices[:, 1] = torch.randint(
        low=0, high=num_tasks, size=(new_task_indices.shape[0],), device=device
    )

    task_indices = torch.cat((task_indices, new_task_indices), dim=0)
    labels = torch.cat((labels, new_labels), dim=0)

    return task_indices, labels


def train(model, optimizer, epoch, loader):
    model.train()

    ce_loss = torch.nn.CrossEntropyLoss()
    res = {"epoch": epoch, "loss": 0, "counter": 0}

    TP = 0
    TN = 0
    FP = 0
    FN = 0

    for data in tqdm(loader):
        # features h = (atom_types, structure_features)
        h = torch.cat(
            (data.atom_types.view(-1, 1), data.structure_features), dim=-1
        ).to(device)
        x = data.pos.to(device)
        edge_index = data.edge_index.to(device)
        tasks_indices = data.task_indices.to(device)
        labels = data.labels.to(device)
        edge_attr = None
        batch = data.batch.to(device)
        # batch_size = number of graphs (each graph represents a protein)
        batch_size = data.ptr.size(0) - 1

        tasks_indices, labels = add_negative_samples(tasks_indices, labels)

        # dictionary mapping b (protein idx) -> (num_tasks_for_protein_b, classes)
        y_pred_dict = model(
            h, x, edge_index, edge_attr, batch, tasks_indices, batch_size
        )

        loss = 0
        protein_idxs = tasks_indices[:, 0]
        unique_protein_idxs = torch.unique(protein_idxs)
        valid_loss = False
        for b in range(batch_size):
            protein_idx = unique_protein_idxs[b]
            mask = protein_idxs == protein_idx
            y = labels[:, 1][mask]
            y_pred = y_pred_dict[b]

            preds = torch.argmax(y_pred, dim=-1)
            TN += torch.logical_and(preds == y, y == 2).sum()
            TP += torch.logical_and(preds == y, y != 2).sum()
            FP += torch.logical_and(preds != y, y == 2).sum()
            FN += torch.logical_and(preds != y, y != 2).sum()

            # print(y_pred)
            protein_loss = ce_loss(y_pred, y)

            num_protein_tasks = y_pred.size(0)
            new_loss = protein_loss  # / num_protein_tasks
            # print(new_loss)
            # HACK ensure loss is real num
            if not (torch.isnan(new_loss).any() or torch.isinf(new_loss).any()):
                valid_loss = True
                loss += new_loss
            # else:
            # loss += torch.tensor(0., device=device)

        if valid_loss:
            optimizer.zero_grad()

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()
            res["loss"] += loss.item()
            res["counter"] += batch_size

    F1 = TP / (TP + 0.5 * (FP + FN))
    acc = (TP + TN) / (TP + TN + FP + FN)
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    print(
        "%s epoch %d avg loss: %.5f, f1 score: %.5f, acc: %.5f, precision: %.5f, recall: %.5f"
        % ("train", epoch, res["loss"] / res["counter"], F1, acc, precision, recall)
    )

    return res["loss"] / res["counter"]


def val(model, epoch, loader, partition):
    model.eval()

    ce_loss = torch.nn.CrossEntropyLoss()
    res = {"epoch": epoch, "loss": 0, "counter": 0}

    TP = 0
    TN = 0
    FP = 0
    FN = 0

    with torch.no_grad():
        for data in tqdm(loader):
            data = data.to(device)
            # features h = (atom_types, structure_features)
            h = torch.cat(
                (data.atom_types.view(-1, 1), data.structure_features), dim=-1
            )
            x = data.pos
            edge_index = data.edge_index
            tasks_indices = data.task_indices
            labels = data.labels
            edge_attr = None
            batch = data.batch
            # batch_size = number of graphs (each graph represents a protein)
            batch_size = data.ptr.size(0) - 1

            tasks_indices, labels = add_negative_samples(tasks_indices, labels)

            # dictionary mapping b (protein idx) -> (num_tasks_for_protein_b, classes)
            y_pred_dict = model(
                h, x, edge_index, edge_attr, batch, tasks_indices, batch_size
            )

            loss = 0
            protein_idxs = tasks_indices[:, 0]
            unique_protein_idxs = torch.unique(protein_idxs)
            for b in range(batch_size):
                protein_idx = unique_protein_idxs[b]
                mask = protein_idxs == protein_idx
                y = labels[:, 1][mask]
                y_pred = y_pred_dict[b]

                preds = torch.argmax(y_pred, dim=-1)
                TN += torch.logical_and(preds == y, y == 2).sum()
                TP += torch.logical_and(preds == y, y != 2).sum()
                FP += torch.logical_and(preds != y, y == 2).sum()
                FN += torch.logical_and(preds != y, y != 2).sum()

                protein_loss = ce_loss(y_pred, y)
                # print(f"Protein loss: {protein_loss / y_pred.size(0)} for protein: {b}")
                num_protein_tasks = y_pred.size(0)
                loss += protein_loss  # / num_protein_tasks

            res["loss"] += loss.item()
            res["counter"] += batch_size

    F1 = TP / (TP + 0.5 * (FP + FN))
    acc = (TP + TN) / (TP + TN + FP + FN)
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    print(
        "%s epoch %d avg loss: %.5f, f1 score: %.5f, acc: %.5f, precision: %.5f, recall: %.5f"
        % (partition, epoch, res["loss"] / res["counter"], F1, acc, precision, recall)
    )

    return res["loss"] / res["counter"]


if __name__ == "__main__":
    main()
