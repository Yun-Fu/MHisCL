
python main.py --n_run 10 --dataset wikipedia --num_neighbors 15 10 --hidden_channels 64 --num_timeslots 14 --heads 8 --nums_layers 2 --dropout 0.3 --lr 5e-4 --epochs 30 & 
python main.py --n_run 10 --dataset reddit --num_neighbors 30 15 --hidden_channels 128 --num_timeslots 5 --heads 8 --nums_layers 2 --dropout 0.7 --lr 1e-4 --epochs 20 & 
python main.py --n_run 10 --dataset mooc --num_neighbors 15 10 --hidden_channels 64 --num_timeslots 3 --heads 8 --dropout 0.7 --lr 1e-4 --epochs 35 & 
python main.py --n_run 10 --dataset otc --num_neighbors 15 10  --hidden_channels 128 --num_timeslots 10 --heads 8 --dropout 0.7 --lr 5e-4 --epochs 5 & 
python main.py --n_run 10 --dataset alpha --num_neighbors 15 10  --hidden_channels 128 --num_timeslots 20 --heads 4 --dropout 0.5 --lr 5e-4 --epochs 20 & 