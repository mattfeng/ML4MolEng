#!/usr/bin/env python
# coding: utf-8

import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import Descriptors,Crippen
from rdkit.Chem import Draw
from rdkit.Chem.Draw import IPythonConsole
import pandas as pd
import sys
from tqdm import tqdm

import itertools
from itertools import repeat

from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

import torch
from torch import optim
from torch.utils.data import DataLoader
from torch import nn
from torch.nn import ModuleDict

from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*') # turn off RDKit warning message 

def smiles2graph(smiles):
    """
    Transfrom smiles into a list nodes (atomic number)
    
    Args:
        smiles (str): SMILES strings
    
    Return:
        z(np.array), A (np.array): list of atomic numbers, adjacency matrix 
    """
    
    mol = Chem.MolFromSmiles(smiles) # no hydrogen 
    z = np.array([atom.GetAtomicNum() for atom in mol.GetAtoms()])
    A = np.stack(Chem.GetAdjacencyMatrix(mol))
    
    return z, A

class GraphDataset(torch.utils.data.Dataset):
    def __init__(self,
                 AtomicNum_list,
                 Edge_list,
                 Natom_list,
                 y_list):
        """
        Represents a dataset of molecular graphs.
        
        Args: 
            z_list (list of torch.LongTensor)
            a_list (list of torch.LongTensor)
            N_list (list of int)
            y_list (list of torch.FloatTensor)
        """
        self.AtomicNum_list = AtomicNum_list # atomic number
        self.Edge_list = Edge_list           # edge list 
        self.Natom_list = Natom_list         # Number of atoms 
        self.y_list = y_list                 # properties to predict 

    def __len__(self):
        return len(self.Natom_list)

    def __getitem__(self, idx):
        AtomicNum = torch.LongTensor(self.AtomicNum_list[idx])
        Edge = torch.LongTensor(self.Edge_list[idx])
        Natom = self.Natom_list[idx]
        y = torch.Tensor(self.y_list[idx])
        
        return AtomicNum, Edge, Natom, y

def collate_graphs(batch):
    """
    Batch multiple graphs into one batched graph.
    
    Args:
        batch (tuple): tuples of AtomicNum, Edge, Natom and y obtained from
            GraphDataset.__getitem__() 
        
    Return 
        (tuple): Batched AtomicNum, Edge, Natom, y
    """
    
    AtomicNum_batch = []
    Edge_batch = []
    Natom_batch = []
    y_batch = []

    cumulative_atoms = np.cumsum([0] + [b[2] for b in batch])[:-1]
    
    for i in range(len(batch)):
        z, a, N, y = batch[i]
        index_shift = cumulative_atoms[i]
        a = a + index_shift
        AtomicNum_batch.append(z) 
        Edge_batch.append(a)
        Natom_batch.append(N)
        y_batch.append(y)
        
    AtomicNum_batch = torch.cat(AtomicNum_batch)
    Edge_batch = torch.cat(Edge_batch, dim=1)
    Natom_batch = Natom_batch
    y_batch = torch.cat(y_batch)
    
    return AtomicNum_batch, Edge_batch, Natom_batch, y_batch 


def scatter_add(src, index, dim_size, dim=-1, fill_value=0):
    """
    Sums all values from the src tensor into out at the indices specified in the index 
    tensor along a given axis dim. 
    """
    
    # make index the same shape as src
    # this will make `scatter_add_` add vectors from `src` to `out`
    index_size = list(repeat(1, src.dim()))
    index_size[dim] = src.size(dim)
    index = index.view(index_size).expand_as(src)
    
    # create the shape of the out vector
    # out will have shape src.size() but with `dim` changed to dim_size
    # e.g.
    #    - src contains 1 row vector for each edge,
    #    - out's rows should have the same dim as those vectors,
    #      but the number of rows should be the number of nodes,
    #      not the number of edges
    dim = range(src.dim())[dim] # convert -1 to actual dim number
    out_size = list(src.size())
    out_size[dim] = dim_size

    out = src.new_full(out_size, fill_value)

    return out.scatter_add_(dim, index, src)

class GNN(nn.Module):
    """
    A GNN model.
    """
    def __init__(self, n_convs=3, n_embed=64):
        super(GNN, self).__init__()

        self.atom_embed = nn.Embedding(100, n_embed)
        # Declare MLPs in a ModuleList
        self.convolutions = nn.ModuleList([ 
                ModuleDict({
                    "update_mlp": nn.Sequential(nn.Linear(n_embed, n_embed), 
                                                nn.ReLU(), 
                                                nn.Linear(n_embed, n_embed)),
                    "message_mlp": nn.Sequential(nn.Linear(n_embed, n_embed), 
                                                 nn.ReLU(), 
                                                 nn.Linear(n_embed, n_embed)) 
                })
                for _ in range(n_convs)
            ])
        # Declare readout layers
        self.readout = nn.Sequential(
            nn.Linear(n_embed, n_embed),
            nn.ReLU(),
            nn.Linear(n_embed, 1)
            )

    def forward(self, AtomicNum, Edge, Natom):
        # Parameterize embedding 
        h = self.atom_embed(AtomicNum) # shape=(Natom, n_embed)

        for conv in self.convolutions:
            prod = h[Edge[0]] * h[Edge[1]]  # shape=(Nedge, n_embed)
            msgs = conv["message_mlp"](prod) # shape=(Nedge, n_embed)
            # send the messages to nodes, undirected graph
            # sum(Natom) is needed because we collated the graph
            agg_msg = scatter_add(src=msgs, index=Edge[1], dim=0, dim_size=sum(Natom)) + \
                      scatter_add(src=msgs, index=Edge[0], dim=0, dim_size=sum(Natom))
            # transform the message using UpdateMLP, and add as residual connection
            h += conv["update_mlp"](agg_msg)

        readout = self.readout(h)
        # readout for each individual graph in the batch
        readout_split = torch.split(readout, Natom)
        output = torch.stack(
            [split.sum(0) for split in readout_split],
            dim=0
            ).squeeze()

        return output


def permute_graph(z, a, perm):
    """
    Permutes the order of nodes in a molecular graph.

    Args: 
        z(np.array): atomic number array
        a(np.array): edge index pairs 

    Return: 
        (np.array, np.array): permuted atomic number, and edge list 
    """
    
    z = np.array(z)
    perm = np.array(perm)
    assert len(perm) == len(z)
    
    z_perm = z[perm]
    a_perm = np.zeros(a.shape).astype(int)
    
    for i, edge in enumerate(a):
        for j in range(len(edge)):
            a_perm[i, j] = np.where(perm==edge[j])[0]
    return z_perm, a_perm


def loop(model, optimizer, loader, epoch, evaluation=False, device="cpu"):
    if evaluation:
        model.eval()
        mode = "eval"
    else:
        model.train()
        mode = "train"
    batch_losses = []

    # Define tqdm progress bar 
    tqdm_data = tqdm(loader, position=0, leave=True, desc="{} (epoch #{})".format(mode, epoch))

    for data in tqdm_data:
        AtomicNumber, Edge, Natom, y = data 
        AtomicNumber = AtomicNumber.to(device)
        Edge = Edge.to(device)
        y = y.to(device)

        # make predictions 
        pred = model(AtomicNumber, Edge, Natom)

        # define loss 
        loss = (pred - y).pow(2).mean()

        if not evaluation:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        batch_losses.append(loss.item())

        postfix = ["batch loss={:.3f}".format(loss.item()) , 
                   "avg. loss={:.3f}".format(np.array(batch_losses).mean())]

        tqdm_data.set_postfix_str(" ".join(postfix))

    return np.array(batch_losses).mean()


if __name__ == "__main__":
    import os
    import wandb

    params = {
        "batch_size": 256,
        "n_convs": 8,
        "n_embed": 256,
        "lr": 1e-3,
        }

    DEBUG = int(os.environ.get("DEBUG", 1))

    if not DEBUG:
        wandb.init(project="ml4moleng_ps4", entity="mattfeng", config=params)

    # load data
    df = pd.read_csv("./data/qm9.csv", index_col=0)
    df = shuffle(df).reset_index()

    atomic_num_list = []
    edge_list = []
    y_list = []
    natom_list = []

    # read and format data
    print("[i] Read data")

    for row in df.itertuples():
        smiles = row.smiles
        atoms, adj = smiles2graph(smiles)
        edges = torch.LongTensor(np.array(np.nonzero(adj)))
        natom = len(atoms)
        alpha = row.alpha

        atomic_num_list.append(atoms)
        edge_list.append(edges)
        # note we're adding a list; this is needed to convert it to a Tensor.
        # see the example above for collate_graph
        y_list.append([alpha])
        natom_list.append(natom)

    data = list(zip(atomic_num_list, edge_list, natom_list, y_list))

    # create data loaders
    print("[i] Create dataloaders")
    data_train_val, data_test = train_test_split(
        data,
        test_size=0.2,
        shuffle=True,
        random_state=54321
        )
    data_train, data_val = train_test_split(
        data_train_val,
        train_size=7/8,
        shuffle=False
        )

    graphs_train = GraphDataset(*list(zip(*data_train)))
    graphs_val = GraphDataset(*list(zip(*data_val)))
    graphs_test = GraphDataset(*list(zip(*data_test)))

    train_loader = DataLoader(graphs_train, batch_size=params["batch_size"], shuffle=True, collate_fn=collate_graphs)
    val_loader = DataLoader(graphs_val, batch_size=params["batch_size"], shuffle=False, collate_fn=collate_graphs)
    test_loader = DataLoader(graphs_test, batch_size=params["batch_size"], shuffle=False, collate_fn=collate_graphs)

    for row in df.itertuples():
        smiles = row.smiles
        atoms, adj = smiles2graph(smiles)
        edges = torch.LongTensor(np.array(np.nonzero(adj)))
        natom = len(atoms)
        alpha = row.alpha

        atomic_num_list.append(atoms)
        edge_list.append(edges)
        # note we're adding a list; this is needed to convert it to a Tensor.
        # see the example above for collate_graph
        y_list.append([alpha])
        natom_list.append(natom)

    # train
    print("[i] Begin training")

    device = 0
    model = GNN(n_convs=params["n_convs"], n_embed=params["n_embed"]).to(device)

    optimizer = optim.Adam(model.parameters(), lr=params["lr"])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=50, verbose=True)

    for epoch in range(500):
        train_loss = loop(model, optimizer, train_loader, epoch, device=device)
        val_loss = loop(model, optimizer, val_loader, epoch, evaluation=True, device=device)

        if not DEBUG:
            wandb.log({
                "train_loss": train_loss,
                "val_loss": val_loss
                })

        # save model
        if epoch % 20 == 0:
            torch.save(model.state_dict(), "saved_models_part1/gcn_model_{}.pt".format(epoch))

