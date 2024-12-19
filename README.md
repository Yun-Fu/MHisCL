<<<<<<< HEAD
# MHisCL
=======
# Code for MHisCL

# Dataset
- Download wikipedia, reddit and mooc datasets from [here](http://snap.stanford.edu/jodie/) and store their csv files in a folder named
```data/```.
- Download the sample datasets (eg. wikipedia and reddit) from [here](http://snap.stanford.edu/jodie/) and store csv file in a folder named
```data/```.

# Reproduction

```bash
# Wikipedia
python main.py --dataset wikipedia --num_neighbors 15 10 --hidden_channels 64 --num_timeslots 14 --heads 8 --nums_layers 2 --dropout 0.3 --lr 5e-4 --epochs 30  

# Reddit
python main.py --dataset reddit --num_neighbors 30 15 --hidden_channels 128 --num_timeslots 5 --heads 8 --nums_layers 2 --dropout 0.7 --lr 1e-4 --epochs 20 --zero_edge

# Mooc
python main.py --dataset mooc --num_neighbors 15 10 --hidden_channels 64 --num_timeslots 3 --heads 8 --dropout 0.7 --lr 1e-4 --epochs 35 

# Bitcoin-alpha
python main.py --dataset alpha --num_neighbors 15 10  --hidden_channels 128 --num_timeslots 20 --heads 4 --dropout 0.5 --lr 5e-4 --epochs 20 

# Bitcoin-otc
python main.py --dataset otc --num_neighbors 15 10  --hidden_channels 128 --num_timeslots 10 --heads 8 --dropout 0.7 --lr 5e-4 --epochs 5 

# Amazon
python main.py --dataset amazon --num_neighbors 15 10 --hidden_channels 64 --num_timeslots 20 --heads 4 --dropout 0.3 --lr 5e-4 --epochs 10 

```
>>>>>>> 1a760e8 (v0)
