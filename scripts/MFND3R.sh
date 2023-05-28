# ETT
python -u main.py --model MFND3R --mode direct --data ETTh1 --root_path ./data/ETT/ --features M --input_len 96  --pred_len 96 --ODA_layers 3 --VRCA_layers 1 --learning_rate 0.0001 --dropout 0.05 --d_model 32 --batch_size 32 --train_epochs 10 --use_RevIN --itr 5 --train --patience 1

python -u main.py --model MFND3R --mode direct --data ETTh1 --root_path ./data/ETT/ --features M --input_len 96  --pred_len 192 --ODA_layers 3 --VRCA_layers 1 --learning_rate 0.0001 --dropout 0.05 --d_model 32 --batch_size 32 --train_epochs 10 --use_RevIN --itr 5 --train --patience 1

python -u main.py --model MFND3R --mode direct --data ETTh1 --root_path ./data/ETT/ --features M --input_len 96  --pred_len 336 --ODA_layers 3 --VRCA_layers 1 --learning_rate 0.0001 --dropout 0.05 --d_model 32 --batch_size 32 --train_epochs 10 --use_RevIN --itr 5 --train --patience 1

python -u main.py --model MFND3R --mode direct --data ETTh1 --root_path ./data/ETT/ --features M --input_len 96  --pred_len 720 --ODA_layers 3 --VRCA_layers 1 --learning_rate 0.0001 --dropout 0.05 --d_model 32 --batch_size 16 --train_epochs 10 --use_RevIN --itr 5 --train --patience 1

python -u main.py --model MFND3R --mode direct --data ETTm2 --root_path ./data/ETT/ --features M --input_len 96  --pred_len 96 --ODA_layers 3 --VRCA_layers 1 --learning_rate 0.0001 --dropout 0.05 --d_model 32 --batch_size 32 --train_epochs 10 --use_RevIN --itr 5 --train --patience 1

python -u main.py --model MFND3R --mode direct --data ETTm2 --root_path ./data/ETT/ --features M --input_len 96  --pred_len 192 --ODA_layers 3 --VRCA_layers 1 --learning_rate 0.0001 --dropout 0.05 --d_model 32 --batch_size 32 --train_epochs 10 --use_RevIN --itr 5 --train --patience 1

python -u main.py --model MFND3R --mode direct --data ETTm2 --root_path ./data/ETT/ --features M --input_len 96  --pred_len 336 --ODA_layers 3 --VRCA_layers 1 --learning_rate 0.0001 --dropout 0.05 --d_model 32 --batch_size 32 --train_epochs 10 --use_RevIN --itr 5 --train --patience 1

python -u main.py --model MFND3R --mode direct --data ETTm2 --root_path ./data/ETT/ --features M --input_len 96  --pred_len 720 --ODA_layers 3 --VRCA_layers 1 --learning_rate 0.0001 --dropout 0.05 --d_model 32 --batch_size 16 --train_epochs 10 --use_RevIN --itr 5 --train --patience 1

# ECL
python -u main.py --model MFND3R --mode direct --data ECL --root_path ./data/ECL/ --features M --target MT_321 --input_len 96  --pred_len 96 --ODA_layers 3 --VRCA_layers 1 --learning_rate 0.0001 --dropout 0.05 --d_model 32 --batch_size 32 --train_epochs 10 --use_RevIN --itr 5 --train --patience 1

python -u main.py --model MFND3R --mode direct --data ECL --root_path ./data/ECL/ --features M --target MT_321 --input_len 96  --pred_len 192 --ODA_layers 3 --VRCA_layers 1 --learning_rate 0.0001 --dropout 0.05 --d_model 32 --batch_size 32 --train_epochs 10 --use_RevIN --itr 5 --train --patience 1

python -u main.py --model MFND3R --mode direct --data ECL --root_path ./data/ECL/ --features M --target MT_321 --input_len 96  --pred_len 336 --ODA_layers 3 --VRCA_layers 1 --learning_rate 0.0001 --dropout 0.05 --d_model 32 --batch_size 16 --train_epochs 10 --use_RevIN --itr 5 --train --patience 1

python -u main.py --model MFND3R --mode direct --data ECL --root_path ./data/ECL/ --features M --target MT_321 --input_len 96  --pred_len 720 --ODA_layers 3 --VRCA_layers 1 --learning_rate 0.0001 --dropout 0.05 --d_model 32 --batch_size 8 --train_epochs 10 --use_RevIN --itr 5 --train --patience 1

#weather
python -u main.py --model MFND3R --mode direct --data weather --root_path ./data/weather/ --features M --input_len 96  --pred_len 96 --ODA_layers 3 --VRCA_layers 1 --learning_rate 0.0001 --dropout 0.05 --d_model 32 --batch_size 32 --train_epochs 10 --use_RevIN --itr 5 --train --patience 1

python -u main.py --model MFND3R --mode direct --data weather --root_path ./data/weather/ --features M --input_len 96  --pred_len 192 --ODA_layers 3 --VRCA_layers 1 --learning_rate 0.0001 --dropout 0.05 --d_model 32 --batch_size 32 --train_epochs 10 --use_RevIN --itr 5 --train --patience 1

python -u main.py --model MFND3R --mode direct --data weather --root_path ./data/weather/ --features M --input_len 96  --pred_len 336 --ODA_layers 3 --VRCA_layers 1 --learning_rate 0.0001 --dropout 0.05 --d_model 32 --batch_size 32 --train_epochs 10 --use_RevIN --itr 5 --train --patience 1

python -u main.py --model MFND3R --mode direct --data weather --root_path ./data/weather/ --features M --input_len 96  --pred_len 720 --ODA_layers 3 --VRCA_layers 1 --learning_rate 0.0001 --dropout 0.05 --d_model 32 --batch_size 16 --train_epochs 10 --use_RevIN --itr 5 --train --patience 1

#Solar
python -u main.py --model MFND3R --mode direct --data Solar --root_path ./data/Solar/ --features M --input_len 96  --pred_len 96 --ODA_layers 3 --VRCA_layers 1 --learning_rate 0.0001 --dropout 0.05 --d_model 32 --batch_size 32 --train_epochs 10 --use_RevIN --itr 5 --train --patience 1

python -u main.py --model MFND3R --mode direct --data Solar --root_path ./data/Solar/ --features M --input_len 96  --pred_len 192 --ODA_layers 3 --VRCA_layers 1 --learning_rate 0.0001 --dropout 0.05 --d_model 32 --batch_size 32 --train_epochs 10 --use_RevIN --itr 5 --train --patience 1

python -u main.py --model MFND3R --mode direct --data Solar --root_path ./data/Solar/ --features M --input_len 96  --pred_len 336 --ODA_layers 3 --VRCA_layers 1 --learning_rate 0.0001 --dropout 0.05 --d_model 32 --batch_size 16 --train_epochs 10 --use_RevIN --itr 5 --train --patience 1

python -u main.py --model MFND3R --mode direct --data Solar --root_path ./data/Solar/ --features M --input_len 96  --pred_len 720 --ODA_layers 3 --VRCA_layers 1 --learning_rate 0.0001 --dropout 0.05 --d_model 32 --batch_size 8 --train_epochs 10 --use_RevIN --itr 5 --train --patience 1

#Ship
python -u main.py --model MFND3R --mode direct --data Ship1 --root_path ./data/Ship/ --features M --input_len 96  --pred_len 96 --ODA_layers 3 --VRCA_layers 1 --learning_rate 0.0001 --dropout 0.05 --d_model 32 --batch_size 32 --train_epochs 10 --use_RevIN --itr 5 --train --patience 1 --target LAT_0

python -u main.py --model MFND3R --mode direct --data Ship1 --root_path ./data/Ship/ --features M --input_len 96  --pred_len 192 --ODA_layers 3 --VRCA_layers 1 --learning_rate 0.0001 --dropout 0.05 --d_model 32 --batch_size 32 --train_epochs 10 --use_RevIN --itr 5 --train --patience 1 --target LAT_0

python -u main.py --model MFND3R --mode direct --data Ship1 --root_path ./data/Ship/ --features M --input_len 96  --pred_len 336 --ODA_layers 3 --VRCA_layers 1 --learning_rate 0.0001 --dropout 0.05 --d_model 32 --batch_size 16 --train_epochs 10 --use_RevIN --itr 5 --train --patience 1 --target LAT_0

python -u main.py --model MFND3R --mode direct --data Ship1 --root_path ./data/Ship/ --features M --input_len 96  --pred_len 720 --ODA_layers 3 --VRCA_layers 1 --learning_rate 0.0001 --dropout 0.05 --d_model 32 --batch_size 8 --train_epochs 10 --use_RevIN --itr 5 --train --patience 1 --target LAT_0

# M4
python -u main.py --model MFND3R --mode direct --data M4_yearly --root_path ./data/M4/ --features S --freq Yearly --input_len 12  --pred_len 6 --ODA_layers 3 --VRCA_layers 0 --learning_rate 0.001 --dropout 0.05 --d_model 32 --batch_size 32 --train_epochs 10 --use_RevIN --itr 5 --train --patience 1 --loss smape

python -u main.py --model MFND3R --mode direct --data M4_quarterly --root_path ./data/M4/ --features S --freq Quarterly --input_len 24  --pred_len 8 --ODA_layers 3 --VRCA_layers 0 --learning_rate 0.001 --dropout 0.05 --d_model 32 --batch_size 32 --train_epochs 10 --use_RevIN --itr 5 --train --patience 1 --loss smape

python -u main.py --model MFND3R --mode direct --data M4_monthly --root_path ./data/M4/ --features S --freq Monthly --input_len 72  --pred_len 18 --ODA_layers 3 --VRCA_layers 0 --learning_rate 0.001 --dropout 0.05 --d_model 32 --batch_size 32 --train_epochs 10 --use_RevIN --itr 5 --train --patience 1 --loss smape

python -u main.py --model MFND3R --mode direct --data M4_weekly --root_path ./data/M4/ --features S --freq Weekly --input_len 65  --pred_len 13 --ODA_layers 3 --VRCA_layers 0 --learning_rate 0.001 --dropout 0.05 --d_model 32 --batch_size 4 --train_epochs 10 --use_RevIN --itr 5 --train --patience 1 --loss smape

python -u main.py --model MFND3R --mode direct --data M4_daily --root_path ./data/M4/ --features S --freq Daily --input_len 84  --pred_len 14 --ODA_layers 3 --VRCA_layers 0 --learning_rate 0.001 --dropout 0.05 --d_model 32 --batch_size 8 --train_epochs 10 --use_RevIN --itr 5 --train --patience 1 --loss smape

python -u main.py --model MFND3R --mode direct --data M4_hourly --root_path ./data/M4/ --features S --freq Hourly --input_len 336  --pred_len 48 --ODA_layers 3 --VRCA_layers 0 --learning_rate 0.001 --dropout 0.05 --d_model 32 --batch_size 8 --train_epochs 10 --use_RevIN --itr 5 --train --patience 1 --loss smape


python -u main.py --model MFND3R --mode iterative --data M4_yearly --root_path ./data/M4/ --features S --freq Yearly --input_len 12  --pred_len 6 --ODA_layers 3 --VRCA_layers 0 --learning_rate 0.001 --dropout 0.05 --d_model 32 --batch_size 32 --train_epochs 10 --use_RevIN --itr 5 --train --patience 1 --loss smape

python -u main.py --model MFND3R --mode iterative --data M4_quarterly --root_path ./data/M4/ --features S --freq Quarterly --input_len 24  --pred_len 8 --ODA_layers 3 --VRCA_layers 0 --learning_rate 0.001 --dropout 0.05 --d_model 32 --batch_size 32 --train_epochs 10 --use_RevIN --itr 5 --train --patience 1 --loss smape

python -u main.py --model MFND3R --mode iterative --data M4_monthly --root_path ./data/M4/ --features S --freq Monthly --input_len 72  --pred_len 18 --ODA_layers 3 --VRCA_layers 0 --learning_rate 0.001 --dropout 0.05 --d_model 32 --batch_size 32 --train_epochs 10 --use_RevIN --itr 5 --train --patience 1 --loss smape

python -u main.py --model MFND3R --mode iterative --data M4_weekly --root_path ./data/M4/ --features S --freq Weekly --input_len 65  --pred_len 13 --ODA_layers 3 --VRCA_layers 0 --learning_rate 0.001 --dropout 0.05 --d_model 32 --batch_size 4 --train_epochs 10 --use_RevIN --itr 5 --train --patience 1 --loss smape

python -u main.py --model MFND3R --mode iterative --data M4_daily --root_path ./data/M4/ --features S --freq Daily --input_len 84  --pred_len 14 --ODA_layers 3 --VRCA_layers 0 --learning_rate 0.001 --dropout 0.05 --d_model 32 --batch_size 8 --train_epochs 10 --use_RevIN --itr 5 --train --patience 1 --loss smape

python -u main.py --model MFND3R --mode iterative --data M4_hourly --root_path ./data/M4/ --features S --freq Hourly --input_len 336  --pred_len 48 --ODA_layers 3 --VRCA_layers 0 --learning_rate 0.001 --dropout 0.05 --d_model 32 --batch_size 8 --train_epochs 10 --use_RevIN --itr 5 --train --patience 1 --loss smape
