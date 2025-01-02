import torch
from torch import nn, optim
import argparse
from tqdm import tqdm

from dataloader import create_fake_dataloader, get_dataloader
from torch_geometric.data import DataLoader, Data
from sklearn.model_selection import train_test_split
from torch.utils.tensorboard import SummaryWriter
import datetime
import os
from torch_geometric.datasets import TUDataset
from torch_geometric.nn import global_add_pool, global_mean_pool, GAT

from models.rgat import RGAT

# Set a manual seed for reproducibility
#torch.manual_seed(42)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class GNN(nn.Module):

    def __init__(self, hidden_dim=64, num_layers=4, num_relations=4, lora_dim=4, gnn_type='rgat'):
        super().__init__()

        self.gnn_type = gnn_type

        if self.gnn_type == 'rgat':
            self.gnn = RGAT(in_channels=7,
                        hidden_channels=hidden_dim,
                        num_layers=num_layers,
                        out_channels=hidden_dim,
                        dropout=0.1,
                        num_relations=num_relations,
                        lora_dim=lora_dim)
        elif self.gnn_type == 'gat':
            self.gnn = GAT(in_channels=7,
                        hidden_channels=hidden_dim,
                        num_layers=num_layers,
                        out_channels=hidden_dim,
                        dropout=0.1,)
        else:
            raise ValueError("Unreckognized GNN Type")
        
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
            x = self.gnn(x=x, edge_index=edge_index, batch=batch, edge_type=edge_type)
        elif self.gnn_type == 'gat':
            x = self.gnn(x=x, edge_index=edge_index, batch=batch)

        x = global_mean_pool(x, batch)

        x = self.mlp(x)

        return x


# Function to create a summary writer for TensorBoard
def create_summary_writer(lr, weight_decay, hidden_dim, num_layers, use_conf, num_blocks, lora_dim, feature_dim):
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
    log_dir = f"./runs/{dt}_mutag_rgat_lr_{lr}_wd_{weight_decay}_hid_size_{hidden_dim}_num_layers_{num_layers}_conf_{use_conf}_nb_{num_blocks}_lora_{lora_dim}_fdim_{feature_dim}/"

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

    for data in tqdm(loader):
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

def main(batch_size=64, lr=5e-4, weight_decay=1e-5, epochs=1000):
    
    dataset = TUDataset(root='data/TUDataset', name='MUTAG')
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


    model = GNN()

    # Calculate the total number of parameters in the model
    total_params = sum(p.numel() for p in model.parameters())
    print(f"total params: {total_params}")

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)


    for epoch in range(epochs):

        train_res = calculate_epoch(model, epoch, train_loader, optimizer)
        for k, v in train_res.items():
            print("Train", str(k), v)

        val_res = calculate_epoch(model, epoch, val_loader)
        for k, v in val_res.items():
            print("Val", str(k), v)

        test_res = calculate_epoch(model, epoch, test_loader)
        for k, v in test_res.items():
            print("Test", str(k), v)



if __name__ == '__main__':
    main()

    
