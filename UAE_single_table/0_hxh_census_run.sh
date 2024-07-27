#!/bin/bash
# 训练文件和测试文件需要在train和eval的代码文件中指定。路径分别是UAE_single_table/training_queries和UAE_single_table/test_queries

# UAE
# 可能需要修改的参数 dataset workload-size
python train_uae.py \
        --num-gpus=1 \
        --dataset=census \
        --workload-size=19000 \
        --epochs=5 \
        --constant-lr=5e-4 \
        --bs=100  \
        --residual \
        --layers=2 \
        --fc-hiddens=128 \
        --direct-io \
        --column-masking

# 可能需要修改的参数 dataset glob err-csv psample
python eval_model.py \
        --dataset=census \
        --glob='uae-census-bs-100-4epochs-psample-200-seed-0-tau-1.0-q-weight-0.0001-layers-2.pt'  \
        --err-csv='./results/census.csv' \
        --psample=4000 \
        --residual \
        --direct-io \
        --column-masking 

# UAE-Q
# 可能需要修改的参数 dataset workload-size
python train_uae.py \
        --num-gpus=1 \
        --dataset=census \
        --workload-size=19000 \
        --epochs=5 \
        --constant-lr=5e-4 \
        --q-bs=100 \
        --run-uaeq  \
        --residual \
        --layers=2 \
        --fc-hiddens=128 \
        --direct-io \
        --column-masking

# 可能需要修改的参数 dataset glob err-csv psample
python eval_model.py \
        --dataset=census \
        --glob='uaeq-census-q_bs-100-4epochs-psample-200-seed-0-tau-1.0-layers-2.pt'  \
        --err-csv='./results/census_q_psample_1000.csv' \
        --psample=1000 \
        --residual \
        --direct-io \
        --column-masking 