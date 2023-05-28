# Extra datasets
# PEMS
python -u main.py --model MFND3R --mode direct --data PEMS08 --root_path ./data/PEMS/ --features M --input_len 720  --pred_len 96 --ODA_layers 3 --VRCA_layers 1 --learning_rate 0.0001 --dropout 0.05 --d_model 32 --batch_size 4 --train_epochs 10 --use_RevIN --itr 5 --train --patience 1 --loss mae

python -u main.py --model MFND3R --mode direct --data PEMS08 --root_path ./data/PEMS/ --features M --input_len 720  --pred_len 192 --ODA_layers 3 --VRCA_layers 1 --learning_rate 0.0001 --dropout 0.05 --d_model 32 --batch_size 4 --train_epochs 10 --use_RevIN --itr 5 --train --patience 1 --loss mae

python -u main.py --model MFND3R --mode direct --data PEMS08 --root_path ./data/PEMS/ --features M --input_len 720  --pred_len 336 --ODA_layers 3 --VRCA_layers 1 --learning_rate 0.0001 --dropout 0.05 --d_model 32 --batch_size 4 --train_epochs 10 --use_RevIN --itr 5 --train --patience 1 --loss mae

python -u main.py --model MFND3R --mode direct --data PEMS08 --root_path ./data/PEMS/ --features M --input_len 720  --pred_len 720 --ODA_layers 3 --VRCA_layers 1 --learning_rate 0.0001 --dropout 0.05 --d_model 32 --batch_size 4 --train_epochs 10 --use_RevIN --itr 5 --train --patience 1 --loss mae

# ETTh2
python -u main.py --model MFND3R --mode direct --data ETTh2 --root_path ./data/ETT/ --features M --input_len 720  --pred_len 96 --ODA_layers 3 --VRCA_layers 1 --learning_rate 0.0001 --dropout 0.05 --d_model 32 --batch_size 32 --train_epochs 10 --use_RevIN --itr 5 --train --patience 1

python -u main.py --model MFND3R --mode direct --data ETTh2 --root_path ./data/ETT/ --features M --input_len 720  --pred_len 192 --ODA_layers 3 --VRCA_layers 1 --learning_rate 0.0001 --dropout 0.05 --d_model 32 --batch_size 32 --train_epochs 10 --use_RevIN --itr 5 --train --patience 1

python -u main.py --model MFND3R --mode direct --data ETTh2 --root_path ./data/ETT/ --features M --input_len 720  --pred_len 336 --ODA_layers 3 --VRCA_layers 1 --learning_rate 0.0001 --dropout 0.05 --d_model 32 --batch_size 32 --train_epochs 10 --use_RevIN --itr 5 --train --patience 1

python -u main.py --model MFND3R --mode direct --data ETTh2 --root_path ./data/ETT/ --features M --input_len 720  --pred_len 720 --ODA_layers 3 --VRCA_layers 1 --learning_rate 0.0001 --dropout 0.05 --d_model 32 --batch_size 32 --train_epochs 10 --use_RevIN --itr 5 --train --patience 1

# Ship2
python -u main.py --model MFND3R --mode direct --data Ship2 --root_path ./data/Ship/ --features M --input_len 720  --pred_len 96 --ODA_layers 3 --VRCA_layers 1 --learning_rate 0.0001 --dropout 0.05 --d_model 32 --batch_size 8 --train_epochs 10 --use_RevIN --itr 5 --train --patience 1 --target LAT_0

python -u main.py --model MFND3R --mode direct --data Ship2 --root_path ./data/Ship/ --features M --input_len 720 --pred_len 192 --ODA_layers 3 --VRCA_layers 1 --learning_rate 0.0001 --dropout 0.05 --d_model 32 --batch_size 8 --train_epochs 10 --use_RevIN --itr 5 --train --patience 1 --target LAT_0

python -u main.py --model MFND3R --mode direct --data Ship2 --root_path ./data/Ship/ --features M --input_len 720 --pred_len 336 --ODA_layers 3 --VRCA_layers 1 --learning_rate 0.0001 --dropout 0.05 --d_model 32 --batch_size 8 --train_epochs 10 --use_RevIN --itr 5 --train --patience 1 --target LAT_0

python -u main.py --model MFND3R --mode direct --data Ship2 --root_path ./data/Ship/ --features M --input_len 720 --pred_len 720 --ODA_layers 3 --VRCA_layers 1 --learning_rate 0.0001 --dropout 0.05 --d_model 32 --batch_size 8 --train_epochs 10 --use_RevIN --itr 5 --train --patience 1 --target LAT_0
