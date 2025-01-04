import torch
from torch import nn, optim
import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt

from torch_geometric.data import DataLoader
from sklearn.model_selection import train_test_split
from torch.utils.tensorboard import SummaryWriter
import datetime
import os
from torch_geometric.datasets import TUDataset
from torch_geometric.nn import global_mean_pool, GAT
from models.rgat import RGAT
import json


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class GNN(nn.Module):

    def __init__(self, hidden_dim=64, num_layers=4, dropout=0.1, num_relations=4, lora_dim=8, num_blocks=None, heads=1, num_bases=None, gnn_type='rgat', node_feature_dim=7):
        super().__init__()

        self.gnn_type = gnn_type

        if self.gnn_type == 'rgat':
            self.linear_proj = nn.Linear(node_feature_dim, hidden_dim)
            self.gnn = RGAT(in_channels=hidden_dim,
                        hidden_channels=hidden_dim,
                        num_layers=num_layers,
                        out_channels=hidden_dim,
                        dropout=dropout,
                        num_relations=num_relations,
                        lora_dim=lora_dim,
                        num_blocks=num_blocks,
                        heads=heads,
                        num_bases=num_bases)
        elif self.gnn_type == 'gat':
            self.gnn = GAT(in_channels=node_feature_dim,
                        hidden_channels=hidden_dim,
                        num_layers=num_layers,
                        out_channels=hidden_dim,
                        dropout=dropout,)
        else:
            raise ValueError("Unreckognized GNN Type", gnn_type)
        
        self.mlp = nn.Sequential(nn.Linear(hidden_dim, hidden_dim),
                                 nn.ReLU(),
                                 #nn.Linear(hidden_dim, hidden_dim),
                                 #nn.ReLU(),
                                 #nn.Linear(hidden_dim, hidden_dim),
                                 #nn.ReLU(),
                                 nn.Linear(hidden_dim, 1),
                                )
        
    def forward(self, x, edge_index, edge_type, batch):
        if self.gnn_type == 'rgat':
            x = self.linear_proj(x)
            x = self.gnn(x=x, edge_index=edge_index, batch=batch, edge_type=edge_type)
        elif self.gnn_type == 'gat':
            x = self.gnn(x=x, edge_index=edge_index, batch=batch)

        x = global_mean_pool(x, batch)

        x = self.mlp(x)

        return x


# Function to create a summary writer for TensorBoard
def create_summary_writer(lr, hidden_dim, num_layers, lora_dim, gnn_type, dataset, num_blocks, num_bases):
    """
    Create a TensorBoard summary writer.

    Args:
        lr (float): Learning rate.
        weight_decay (float): Weight decay.
        hidden_dim (int): Hidden dimension size.
        num_layers (int): Number of layers.
        use_conf (bool): Whether to use confidence score.
        num_blocks (int): Number of blocks.
        lora_dim (int): Lora dimension.
        feature_dim (int): Feature dimension.

    Returns:
        SummaryWriter: The TensorBoard summary writer.
    """
    os.makedirs("runs", exist_ok=True)
    dt = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = f"./runs/{dt}_{dataset}_{gnn_type}_lr_{lr}_hid_size_{hidden_dim}_num_layers_{num_layers}_num_blocks_{num_blocks}_lora_{lora_dim}_num_bases_{num_bases}/"

    writer = SummaryWriter(log_dir)
    return writer

def calculate_epoch(model, epoch, loader, opt=None):

    train = False if opt is None else True

    if train:
        model.train()
    else:
        model.eval()

    # Define the loss function
    loss_fn = nn.BCEWithLogitsLoss()

    total_loss = 0
    num_graphs = 0
    num_accurate = 0

    for data in loader:
        x = data.x.to(device)
        edge_index = data.edge_index.to(device)
        y = data.y.to(device)
        batch = data.batch.to(device)

        #TODO: this could be made more efficient by dataset preprocessing
        edge_type = data.edge_attr.to(device)
        edge_type = torch.argmax(edge_type, dim=1)

        if train:
            preds = model(x=x, edge_index=edge_index, batch=batch, edge_type=edge_type)
        else:
            with torch.inference_mode():
                preds = model(x=x, edge_index=edge_index, batch=batch, edge_type=edge_type)
        preds = preds.squeeze(1)

        loss = loss_fn(preds, y.float())

        total_loss += loss.item()*y.shape[0]
        num_graphs += y.shape[0]

        # preds > 0. means pos prediction since preds unnormalized logit
        num_accurate += int(((preds > 0.) == (y > 0.5)).sum().item())

        if train:
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            opt.step()

    res = {'MeanLoss': total_loss/num_graphs,
           "Accuracy": float(num_accurate)/num_graphs}


    return res

def main(batch_size=64, lr=5e-4, dropout=0.1, weight_decay=1e-5, epochs=1000, num_layers=4, hidden_dim=64, lora_dim=8, num_blocks=None, heads=1, num_bases=None, gnn_type='rgat', dataset='MUTAG', sweep=False):
    
    dataset = TUDataset(root='data/TUDataset', name=dataset, use_node_attr=True)
    node_feature_dim = dataset[0].x.shape[1]
    num_relations = dataset.num_edge_labels
    
    # Split the data into training, validation, and test sets
    train_data, temp_data = train_test_split(
        dataset, test_size=0.2, random_state=42
    )
    val_data, test_data = train_test_split(temp_data, test_size=0.5, random_state=42)

    # Create DataLoader objects for each subset
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)


    model = GNN(
        num_layers=num_layers,
        hidden_dim=hidden_dim,
        dropout=dropout,
        num_relations=num_relations,
        lora_dim=lora_dim,
        num_blocks=num_blocks,
        heads=1,
        num_bases=num_bases,
        gnn_type=gnn_type,
        node_feature_dim=node_feature_dim
    ).to(device)
    num_params = sum(p.numel() for p in model.parameters())

    # Calculate the total number of parameters in the model
    total_params = sum(p.numel() for p in model.parameters())
    print(f"total params: {total_params}")

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    writer = create_summary_writer(lr=lr, hidden_dim=hidden_dim, num_layers=num_layers, lora_dim=lora_dim, gnn_type=gnn_type, dataset=dataset, num_blocks=num_blocks, num_bases=num_bases)


    res = {'best_epoch': 0, 'best_val_acc': 0, 'best_test_acc': 0, 'num_params': num_params, 'train_losses': []}
    for epoch in tqdm(range(epochs)):

        train_res = calculate_epoch(model, epoch, train_loader, optimizer)
        for k, v in train_res.items():
            print("Train", str(k), v)
            writer.add_scalar(f"Train/{str(k)}", v, epoch)
            if str(k) == "MeanLoss":
                res['train_losses'].append(v)

        val_res = calculate_epoch(model, epoch, val_loader)
        for k, v in val_res.items():
            print("Val", str(k), v)
            writer.add_scalar(f"Val/{str(k)}", v, epoch)
            if str(k) == "Accuracy" and res['best_val_acc'] < v:
                res['best_val_acc'] = v 
                res['best_epoch'] = epoch

        test_res = calculate_epoch(model, epoch, test_loader)
        for k, v in test_res.items():
            print("Test", str(k), v)
            writer.add_scalar(f"Test/{str(k)}", v, epoch)
            if res['best_epoch'] == epoch and str(k) == "Accuracy":
                res['best_test_acc'] = v

    return res

def run_parameter_sweep(args, param_name, param_values, seeds=[0, 10]):
    """
    Run a parameter sweep across specified values and seeds.
    
    Args:
        args: ArgumentParser args
        param_name: str, one of 'lora_dim', 'num_blocks', or 'num_bases'
        param_values: list of values to sweep over
        seeds: list of random seeds to use
    """
    best_res = {"best_val_acc": -1}
    
    for param_value in param_values:
        for seed in seeds:
            torch.manual_seed(seed)
            
            # Create copy of args dict and update the swept parameter
            sweep_args = vars(args).copy()
            sweep_args[param_name] = param_value
            
            res = main(**sweep_args)
            if res['best_val_acc'] > best_res['best_val_acc']:
                best_res = res
    
    return best_res

def plot_sweeps_test_accs(default_res, sweep_res):
    models = ["full_rank", "blocks", "bases", "lora"]
    test_acc = [default_res['best_test_acc'], sweep_res['num_blocks']['best_test_acc'], sweep_res['num_bases']['best_test_acc'], sweep_res['lora_dim']['best_test_acc']]
    num_params = [default_res['num_params'], sweep_res['num_blocks']['num_params'], sweep_res['num_bases']['num_params'], sweep_res['lora_dim']['num_params']]

    # Create scatter plot
    plt.figure(figsize=(8, 6))
    plt.scatter(num_params, test_acc, color='blue')

    # Annotate each point
    for i, model in enumerate(models):
        plt.annotate(model, (num_params[i], test_acc[i]), textcoords="offset points", xytext=(0, 5), ha='center')

    # Set labels and title
    plt.xlabel("Number of Parameters")
    plt.ylabel("Test Accuracy")
    plt.title("Test Accuracy vs Number of Parameters")

    # Show plot
    plt.grid(True)
    plt.savefig(f"data/plots/sweep_test_acc_vs_params_{args.dataset}.png")

def plot_performance_effciencies(default_res, sweep_res):
    models = ["full_rank", "blocks", "bases", "lora"]
    performance_efficiencies = [
        default_res['best_test_acc'] / default_res['num_params'],
        sweep_res['num_blocks']['best_test_acc'] / sweep_res['num_blocks']['num_params'],
        sweep_res['num_bases']['best_test_acc'] / sweep_res['num_bases']['num_params'],
        sweep_res['lora_dim']['best_test_acc'] / sweep_res['lora_dim']['num_params']
    ]

    # Define light colors for the bar plot
    bar_colors = ['lightblue', 'lightcoral', 'lightgreen', 'lightskyblue']

    # Bar Plot for Performance Efficiency with different colors for each model
    plt.figure(figsize=(8, 6))
    plt.bar(models, performance_efficiencies, color=bar_colors)

    # Set labels and title
    plt.xlabel("Models")
    plt.ylabel("Performance Efficiency (Test Accuracy / Number of Parameters)")
    plt.title("Performance Efficiency of Different Models")

    # Show plot
    plt.grid(True)
    plt.savefig(f"data/plots/sweep_performance_efficiencies_{args.dataset}.png")

def plot_train_losses(default_res, sweep_res):
    plt.figure(figsize=(8, 6))
    plt.plot(default_res['train_losses'], label='full_rank')
    for param_name in sweep_res.keys():
        plt.plot(sweep_res[param_name]['train_losses'], label=param_name)
    plt.xlabel("Epoch")
    plt.ylabel("Train Loss")
    plt.title("Train Loss of Different Models")
    plt.legend()
    plt.savefig(f"data/plots/sweep_train_losses_{args.dataset}.png")

if __name__ == '__main__':
    os.makedirs("results", exist_ok=True)
    
    parser = argparse.ArgumentParser()

    parser.add_argument('--gnn_type', default='rgat')
    parser.add_argument('--dataset', default='MUTAG')
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--epochs', type=int, default=100)

    parser.add_argument('--num_layers', type=int, default=4)
    parser.add_argument('--hidden_dim', type=int, default=128)
    parser.add_argument('--dropout', type=int, default=0.1)

    parser.add_argument('--lora_dim', type=int, default=0)
    parser.add_argument('--num_blocks', type=int, default=None)
    parser.add_argument('--heads', type=int, default=1)
    parser.add_argument('--num_bases', type=int, default=None)
    parser.add_argument('--sweep', action='store_true')

    args = parser.parse_args()

    if args.sweep:
        default_res = main(**vars(args))

        # Define sweep parameters
        sweep_configs = {
            'lora_dim': [2, 4, 8],
            'num_blocks': [2, 4, 8],
            'num_bases': [1, 2, 4]
        }
        sweep_res = {}
        for param_name, param_values in sweep_configs.items():
            best_res = run_parameter_sweep(args, param_name, param_values)
            sweep_res[param_name] = best_res
        
        print(default_res)
        print(sweep_res)

        plot_sweeps_test_accs(default_res, sweep_res)
        plot_performance_effciencies(default_res, sweep_res)
        plot_train_losses(default_res, sweep_res)


    else:
        res = main(**vars(args))
        with open(f'results/run_{args.dataset}_ldim_{args.lora_dim}_blks_{args.num_blocks}_heads_{args.heads}_bases_{args.num_bases}_nparams_{res["num_params"]}.json', 'w') as fp:
            json.dump(res, fp)