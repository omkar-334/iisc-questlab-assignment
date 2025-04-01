if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/LongForecasting" ]; then
    mkdir ./logs/LongForecasting
fi

seq_len=336
model_name=DLinear

python -um DLinear.run_longexp \
  --is_training 1 \
  --root_path ./ \
  --data_path exchange_rate.csv \
  --model_id Exchange_$seq_len'_'96 \
  --model $model_name \
  --data custom \
  --features MS \
  --seq_len $seq_len \
  --pred_len 14 \
  --enc_in 8 \
  --des 'Exp' \
  --itr 1 --batch_size 16 --learning_rate 0.001 >logs/LongForecasting/$model_name'_'Exchange_$seq_len'_'96.log 