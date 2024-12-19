import argparse

import torch
from torch_geometric import seed_everything
from tqdm import tqdm

from hiscl.dataset import load_dataset
from hiscl.history import History
from hiscl.loader import EventLoader
from hiscl.measure import Measure
from hiscl.model import GatedTGAT,TGAT,GatedTGNN
from hiscl.loss import contrastive_loss, cosine_similarity

import random
import os
def set_seed(seed: int):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) 
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = False

def train(loader, model, history, optimizer, device, data, args):
    model.train()
    history.train()
    pbar = tqdm(loader)
    total_loss = 0
    for batch in pbar:
        batch = batch.to(device)
        optimizer.zero_grad()
        out = model.encode(batch.x, batch.edge_index, batch.t, batch.msg)
        src = batch.src[: batch.batch_size]
        out = out[src]
        idx = batch.input_id.cpu()
        num_neg = batch.batch_size
        neg_mem = history.get_history(torch.randint(
            0, history.num_nodes, size=(num_neg,)))
        raw_msg = torch.zeros(idx.size(0),1).to(device) 
        cur_mem, prev_mem = history(raw_msg, data.src[idx]) 

        if args.con=='his':
            loss = contrastive_loss(prev_mem, cur_mem, args.tau) 
            loss += torch.exp(cosine_similarity(cur_mem, neg_mem) 
                          ).sum(dim=1).log().mean() 
        elif args.con=='stru':
            loss = contrastive_loss(out, cur_mem, args.tau) 
            loss += torch.exp(cosine_similarity(out, neg_mem)
                            ).sum(dim=1).log().mean()
        elif args.con=='all':
            loss = args.alpha*contrastive_loss(out, cur_mem, args.tau) + \
                contrastive_loss(prev_mem, cur_mem, args.tau)
            loss += torch.exp(cosine_similarity(cur_mem, neg_mem) 
                            ).sum(dim=1).log().mean()
            loss += args.alpha*torch.exp(cosine_similarity(out, neg_mem) 
                            ).sum(dim=1).log().mean()


        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        pbar.set_description(f"Train Loss = {loss.item():.4f}")

    return total_loss / len(loader)


@torch.no_grad()
def test(loader, model, history, measure, device, data):
    preds = []
    labels = []
    model.eval()
    history.eval()
    for batch in tqdm(loader):
        torch.cuda.empty_cache()
        batch = batch.to(device)
        out = model.encode(batch.x, batch.edge_index, batch.t, batch.msg)
        src = batch.src[: batch.batch_size]
        out = out[src]
        label = batch.y[: batch.batch_size]
        idx = batch.input_id.cpu()
        raw_msg = torch.zeros(idx.size(0),1).to(device) 
        cur_mem, prev_mem = history(raw_msg, data.src[idx], update=True)
        if args.con=='his':
            pred = 1-torch.diag(cosine_similarity(cur_mem, prev_mem)).view(-1)
        elif args.con=='stru': 
            pred = 1-torch.diag(cosine_similarity(out, cur_mem)).view(-1)
        elif args.con=='all':
            p1 = torch.diag(cosine_similarity(out, cur_mem)).view(-1)
            p2 = torch.diag(cosine_similarity(cur_mem, prev_mem)).view(-1)
            pred = (1 - p1) + (1 - p2) 

        pred = pred.sigmoid() 
        torch.cuda.empty_cache()
        preds.append(pred.detach().cpu())
        labels.append(label.detach().cpu())
    preds = torch.cat(preds)
    preds[torch.isnan(preds)] = 0.0 
    labels = torch.cat(labels)
    return measure(labels[labels!=2], preds[labels!=2])


def read_parser():
    # Create the parser
    parser = argparse.ArgumentParser(description="Process some parameters.")

    # Add the arguments
    parser.add_argument("--dataset", type=str,
                        default='wikipedia', help="The dataset to use")
    parser.add_argument("--alpha", type=float,
                        default=1, help="The weight for structure contrastive loss")
    parser.add_argument("--con", type=str,
                        default='all', help="The contrastive type") # all, his, stru 
    parser.add_argument("--time_enc", type=str,
                        default='sste', help="Type of time encoding")
    parser.add_argument("--aggr", type=str,
                        default='gat', help="Type of aggregation")
    parser.add_argument("--num_neighbors", nargs="+",
                        type=int, help="The number of neighbors",)
    parser.add_argument("--num_workers", type=int,
                        default=8, help="The number of workers")
    parser.add_argument("--hidden_channels", type=int,
                        default=256, help="The number of hidden channels",)
    parser.add_argument("--num_timeslots", type=int,
                        default=1, help="The number of timeslots")
    parser.add_argument("--heads", type=int, default=1,
                        help="The number of heads")
    parser.add_argument("--nums_layers", type=int,
                        default=2, help="The number of layers")
    parser.add_argument("--dropout", type=float,
                        default=0.5, help="The dropout rate")
    parser.add_argument("--lr", type=float, default=0.0001,
                        help="The learning rate")
    parser.add_argument("--epochs", type=int, default=20,
                        help="The number of epochs")
    parser.add_argument("--history_retrieve", type=str,
                        default='mean', help="The history retrieval method")
    parser.add_argument("--recurrent", type=str,
                        default='gru', help="recurrent network")
    parser.add_argument("--tau", type=float, default=0.02,
                        help="Tau in loss function")
    parser.add_argument("--batch_size", type=int,
                        default=256, help="The batch size")
    parser.add_argument("--zero_edge", action="store_true",
                        default=False, help="Using zero egdes feat")
    parser.add_argument("--gpu", type=int, default=1)

    
    # Parse the arguments
    args = parser.parse_args()
    print(args)
    return args


import numpy as np
import os.path as osp
import logging

args = read_parser()
if not os.path.exists('log'):
    os.makedirs('log')
logging.basicConfig(format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO, filename=f'log/{osp.basename(__file__)[:-3]}_{args.dataset}.txt')
log = logging.getLogger() 
log.info('########################## START #########################')
log.info(f'\n{args.dataset}-param: {args.__dict__}')

data = load_dataset(args.dataset,zero_edge=args.zero_edge)
device = torch.device(f"cuda:{args.gpu}")
print(f"Using {device}")

set_seed(42)
sample_args = {"num_neighbors": args.num_neighbors,
            "num_workers": args.num_workers}
train_loader = EventLoader(
    data,
    input_events=data.train_mask,
    shuffle=True,
    batch_size=args.batch_size,
    **sample_args,
)
valid_loader = EventLoader(
    data,
    input_events=data.val_mask, 
    shuffle=False,
    batch_size=args.batch_size,
    **sample_args,
)
test_loader = EventLoader(
    data,
    input_events=data.test_mask,
    shuffle=False,
    batch_size=args.batch_size,
    **sample_args,
)
history = History(
    data.num_nodes,
    args.num_timeslots,
    args.hidden_channels,
    device=device,
    history_retrieve=args.history_retrieve,
    recurrent=args.recurrent,
).to(device)
if args.aggr=='gat':
    model_class = GatedTGAT if args.time_enc=='sste' else TGAT
    model = model_class(
            in_channels=data.x.size(1),
            out_channels=1,
            hidden_channels=args.hidden_channels,
            edge_dim=data.msg.size(1),
            heads=args.heads,
            num_layers=args.nums_layers,
            dropout=args.dropout,
        ).to(device)
else:
    model = GatedTGNN(
            in_channels=data.x.size(1),
            out_channels=1,
            hidden_channels=args.hidden_channels,
            edge_dim=data.msg.size(1),
            heads=args.heads,
            num_layers=args.nums_layers,
            dropout=args.dropout,
            aggr=args.aggr  
        ).to(device)

print(model)
print(history)

optimizer = torch.optim.Adam(
    list(model.parameters()) + list(history.parameters()), lr=args.lr
)
measure = Measure("auc")
best = 0
best_val = 0
for epoch in range(1, 1 + args.epochs):
    loss = train(train_loader, model, history, optimizer, device, data, args)
    val_auc = test(valid_loader, model, history, measure, device, data)
    test_auc = test(test_loader, model, history, measure, device, data)

    if val_auc > best_val:
        best_val = val_auc 
        best = test_auc

    print(
        f"Epoch: {epoch:03d}, Loss: {loss:.4f}, "
        f"Val AUC: {val_auc:.2%}, Test AUC: {test_auc:.2%}, Best AUC: {best:.2%}"
    )
    log.info(
        f"Epoch: {epoch:03d}, Loss: {loss:.4f}, "
        f"Val AUC: {val_auc:.2%}, Test AUC: {test_auc:.2%}, Best AUC: {best:.2%}"
    )
