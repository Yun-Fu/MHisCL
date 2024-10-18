

import pandas as pd
import numpy as np
import torch

from torch_geometric.datasets import JODIEDataset
from typing import Optional
from torch_geometric.transforms import BaseTransform

from .temporal_data import TemporalData

class TemporalSplit(BaseTransform):
    def __init__(
        self,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        key: Optional[str] = "t",
    ):
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.key = key

    def forward(self, data):
        key = self.key
        t = data[key].sort().values
        val_ratio = self.val_ratio
        test_ratio = self.test_ratio
        val_time, test_time = np.quantile(
            t.cpu().numpy(), [1. - val_ratio - test_ratio, 1. - test_ratio])
        data.train_mask = data[key] < val_time
        data.val_mask = torch.logical_and(data[key] >= val_time, data[key]
                                          < test_time)
        data.test_mask = data[key] >= test_time
        return data

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.val_ratio}, '
                f'{self.test_ratio})')


def load_dataset(dataset, val_ratio= 0.15, test_ratio= 0.15, zero_edge=False):
    if dataset in ["otc", "alpha"]:
        graph_df = pd.read_csv(f"bitcoin{dataset}.csv")
        row = torch.from_numpy(graph_df.u.values).to(torch.long)
        col = torch.from_numpy(graph_df.i.values).to(torch.long)
        edge_index = torch.stack([row, col], dim=0)
        edge_index = edge_index - 1
        labels = torch.from_numpy(graph_df.label.values).to(torch.float)
        stamps = graph_df.ts.values
        num_nodes = edge_index.max().item() + 1
        t = torch.from_numpy(stamps).long()
        data = TemporalData(
            src=edge_index[0], dst=edge_index[1], t=t, y=labels, num_nodes=num_nodes
        )
        data = TemporalSplit(val_ratio=val_ratio, test_ratio=test_ratio)(data)
        data.x = torch.zeros(data.num_nodes, 1)  # assign zero node features
        data.msg = torch.zeros(data.num_events, 1)  # assign zero edge features
    else:
        data = JODIEDataset(root="../data/", 
                            name=dataset, 
                            transform=TemporalSplit(val_ratio=val_ratio, test_ratio=test_ratio))[0]
        data = TemporalData(**data.to_dict())

        data.x = torch.zeros(data.num_nodes, 1)  # assign zero node features
        if zero_edge:
            data.msg = torch.zeros(data.num_events, 1)  # assign zero edge features

    print("=" * 20, "Data statistics", "=" * 20)
    print(f"Name: {dataset}")
    print(f"Number of nodes: {data.num_nodes}")
    print(f"Number of edges: {data.num_events}")
    print(f"Number of node features: {data.x.size(1)}")
    print(f"Number of edge features: {data.msg.size(1)}")
    print(f"Number of anomalies: {data.y.float().mean().item():.3%}")
    return data
