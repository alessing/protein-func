import torch
from torch import nn, optim
import argparse
from tqdm import tqdm

from dataloader import create_fake_dataloader
from torch_geometric.data import DataLoader
from sklearn.model_selection import train_test_split
from torch.utils.tensorboard import SummaryWriter


from model.models import FuncGNN

torch.manual_seed(42)

parser = argparse.ArgumentParser(
    description="Protein Function Prediction with E(3)-Equivariant GNNs and Multi-task Learning"
)


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

parser.add_argument("--seed", type=int, default=1, help="random seed (default: 1)")

parser.add_argument("--weight_decay", type=float, default=1e-6, help="weight decay")

parser.add_argument("--lr", type=float, default=1e-4, help="learning rate")

parser.add_argument(
    "--tensorboard", type=str_to_bool, default=False, help="Uses tensorboard"
)

time_exp_dic = {"time": 0, "counter": 0}


args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
loss_mse = nn.MSELoss()


def create_summary_writer(
    lr,
    weight_decay,
):
    dt = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = f"./runs/{dt}_funcgnn_lr_{lr}_wd_{weight_decay}/"

    writer = SummaryWriter(log_dir)
    return writer


class EarlyStopper:
    def __init__(
        self, patience=1, min_delta=0, plateau_patience=5, plateau_threshold=1e-4
    ):
        self.patience = patience
        self.plateau_patience = plateau_patience

        self.min_delta = min_delta
        self.plateau_threshold = plateau_threshold

        self.counter = 0
        self.plateau_counter = 0

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
    num_tasks = 1000
    model = FuncGNN(10, 11, 1, 32, 32, 32, 3)

    protein_data, dl = create_fake_dataloader(num_tasks)

    train_data, temp_data = train_test_split(
        protein_data, test_size=0.2, random_state=42
    )
    val_data, test_data = train_test_split(temp_data, test_size=0.5, random_state=42)

    # Step 2: Create DataLoader objects for each subset
    train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=32, shuffle=False)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"total params: {total_params}")

    if torch.cuda.is_available():
        model = model.cuda()

    weight_decay = args.weight_decay
    lr = args.lr

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    if args.tensorboard:
        writer = create_summary_writer(
            lr,
            weight_decay,
        )

    # # early stopping
    early_stopper = EarlyStopper(patience=10, min_delta=0.005)
    results = {"epochs": [], "losess": []}
    best_val_loss = float("inf")
    best_test_loss = float("inf")
    best_train_loss = float("inf")
    best_epoch = 0
    for epoch in range(0, args.epochs):
        train_loss = train(model, optimizer, epoch, train_loader)
        breakpoint()
        if args.tensorboard:
            writer.add_scalar("Loss/train", train_loss, epoch)

        # if epoch % args.test_interval == 0:
        val_loss = val(
            model,
            epoch,
            val_loader,
        )
        test_loss = val(
            model,
            epoch,
            test_loader,
        )

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
            torch.save(
                model.state_dict(),
                f"best_models/funcgnn.pt",
            )

        print(
            "*** Best Train Loss: %.5f \t Best Val Loss: %.5f \t Best Test Loss: %.5f \t Best epoch %d"
            % (best_train_loss, best_val_loss, best_test_loss, best_epoch)
        )

        # json_object = json.dumps(results, indent=4)
        # with open(args.outf + "/" + args.exp_name + "/losess.json", "w") as outfile:
        #     outfile.write(json_object)
        if early_stopper.early_stop(val_loss):
            print(f"EARLY STOPPED")
            break

    return best_train_loss, best_val_loss, best_test_loss, best_epoch, total_params


def train(model, optimizer, epoch, loader):
    model.train()

    res = {"epoch": epoch, "loss": 0, "coord_reg": 0, "counter": 0}

    for data in tqdm(loader):
        # B = batch_size, N = n_nodes, L = seq_len, n = 3
        print(data)
        # data.edge_index, data.pos, data.atom_types, data.structure_features

        # print(data.atom_types.shape)
        # breakpoint()
        h = torch.cat((data.atom_types.view(-1, 1), data.structure_features), dim=-1)
        x = data.pos
        edge_index = data.edge_index
        tasks_indices = data.task_indices

        model(h, x, edge_index, edge_attr, batch, task_idx)

        # protein label is associated with (index), 2nd is actual label

        breakpoint()
        optimizer.zero_grad()

        # BCE LOS
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        optimizer.step()
        res["loss"] += loss.item() * B
        res["counter"] += B

    prefix = ""
    print(
        "%s epoch %d avg loss: %.5f"
        % (prefix + loader.dataset.partition, epoch, res["loss"] / res["counter"])
    )

    return res["loss"] / res["counter"]


if __name__ == "__main__":
    main()
