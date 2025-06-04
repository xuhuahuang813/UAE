#!/bin/bash
# conda activate sam
# 训练文件和测试文件需要在train和eval的代码文件中指定。路径分别是UAE_single_table/training_queries和UAE_single_table/test_queries

# UAE
# 可能需要修改的参数 dataset workload-size
# 训练（默认）
CUDA_VISIBLE_DEVICES=0 python train_uae.py \
        --num-gpus=0 \
        --dataset=dmv \
        --workload-size=50000 \
        --train-query-path=/data/homedata/hxh/data/MirrorTestData25/250407_TrainData/dmv_10w_0604_for_hist_constraint.json \
        --column-masking \
        --residual \
        --bs=1024 
# 训练
CUDA_VISIBLE_DEVICES=0 python train_uae.py \
        --num-gpus=0 \
        --dataset=dmv \
        --workload-size=50000 \
        --train-query-path=training_queries/dmv_10w.json \
        --column-masking \
        --residual \
        --bs=4096 \
        --constant-lr=5e-4 \
        --layers=2 \
        --fc-hiddens=128 \
        --direct-io
        
# 可能需要修改的参数 dataset glob err-csv psample
# 预测（默认）
CUDA_VISIBLE_DEVICES=0 python eval_model.py \
        --dataset=dmv \
        --glob='uae-dmv-bs-1024-3epochs-psample-200-seed-0-tau-1.0-q-weight-0.0001-layers-4.pt'  \
        --test-query-path=/data/homedata/hxh/data/MirrorTestData25/250407_TrainData/dmv_10w_0604_test_selected_filtered.json \
        --test-query-num=2000 \
        --err-csv='./results/250604_dmv_all.csv' \
        --column-masking \
        --residual

CUDA_VISIBLE_DEVICES=0 python eval_model.py \
        --dataset=dmv \
        --glob='uae-dmv-bs-4096-9epochs-psample-200-seed-0-tau-1.0-q-weight-0.0001-layers-2.pt'  \
        --test-query-path=test_queries/dmv_10w_test.json \
        --test-query-num=2000 \
        --err-csv='./results/dmv_10w_test.csv' \
        --column-masking \
        --residual \
        --layers=2 \
        --fc-hiddens=128 \
        --direct-io

# UAE-Q
# 可能需要修改的参数 dataset workload-size
CUDA_VISIBLE_DEVICES=0 python train_uae.py \
        --num-gpus=1 \
        --dataset=dmv \
        --workload-size=50000 \
        --train-query-path=training_queries/dmv_5w_train.json \
        --column-masking \
        --residual \
        --q-bs=100

CUDA_VISIBLE_DEVICES=0 python train_uae.py \
        --cuda-num=1 \
        --num-gpus=1 \
        --dataset=dmv \
        --workload-size=50000 \
        --train-query-path=training_queries/dmv_5w_train.json \
        --epochs=20 \
        --constant-lr=5e-4 \
        --q-bs=100 \
        --run-uaeq  \
        --residual \
        --layers=2 \
        --fc-hiddens=128 \
        --direct-io \
        --column-masking > 250213_uae_q_resmade_dmv.txt 2>&1 &

# 可能需要修改的参数 dataset glob err-csv psample
# python eval_model.py \
#         --dataset=dmv \
#         --glob='uaeq-dmv-q_bs-32-9epochs-psample-200-seed-0-tau-1.0-layers-2.pt'  \
#         --err-csv='./results/dmv_q_psample_1000.csv' \
#         --psample=100 \
#         --residual \
#         --direct-io \
#         --column-masking 

# nohup python train_uae.py --cuda-num=1 --num-gpus=1 --dataset=dmv --workload-size=19000 --epochs=10 --constant-lr=5e-4 --q-bs=2048 --run-uaeq --residual --layers=2 --fc-hiddens=128 --direct-io --column-masking > output_dmv_uae_q.log 2>&1 &