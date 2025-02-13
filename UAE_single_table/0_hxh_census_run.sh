#!/bin/bash
# conda activate sam
# 训练文件和测试文件需要在train和eval的代码文件中指定。路径分别是UAE_single_table/training_queries和UAE_single_table/test_queries

# UAE
# 可能需要修改的参数 dataset workload-size
# 需要保留--column-masking，否则会报错
# 训练Made时，去掉--residual；训练ResMade时，保留--residual，并将BS设置为512避免cuda内存不够报错。
# 训练（默认）
python train_uae.py \
        --num-gpus=1 \
        --dataset=census \
        --workload-size=19000 \
        --train-query-path=training_queries/census_train_3-train-mirror.txt \
        --column-masking \
        --residual \
        --bs=512
# 训练（自定义）
python train_uae.py \
        --num-gpus=1 \
        --dataset=census \
        --workload-size=19000 \
        --train-query-path=training_queries/census_train_3-train-mirror.txt \
        --column-masking \
        --residual \
        --epochs=2 \
        --constant-lr=5e-4 \
        --bs=100  \
        --layers=2 \
        --fc-hiddens=128 \
        --direct-io 


# 可能需要修改的参数 dataset glob err-csv psample
# 需要保留--column-masking，否则会报错
# 使用Made预测时，去掉--residual；使用ResMade预测时，保留--residual。
# 预测（默认）
python eval_model.py \
        --dataset=census \
        --glob='uae-census-bs-512-19epochs-psample-200-seed-0-tau-1.0-q-weight-0.0001-layers-4.pt'  \
        --test-query-path=test_queries/census_train_3-test-mirror.txt \
        --err-csv='./results/census.csv' \
        --column-masking \
        --residual
# 预测（自定义）
python eval_model.py \
        --dataset=census \
        --glob='uae-census-bs-1024-19epochs-psample-200-seed-0-tau-1.0-q-weight-0.0001-layers-4.pt'  \
        --test-query-path=test_queries/census_train_3-test-mirror.txt \
        --err-csv='./results/census.csv' \
        --column-masking \
        --psample=4000 \
        --residual \
        --direct-io \
         

# UAE-Q
# 可能需要修改的参数 dataset workload-size
# 训练（默认）
python train_uae.py \
        --num-gpus=1 \
        --dataset=census \
        --workload-size=19000 \
        --train-query-path=training_queries/census_train_3-train-mirror.txt \
        --column-masking \
        --run-uaeq \
        --residual
# 训练（自定义）
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
# 预测（默认）
python eval_model.py \
        --dataset=census \
        --glob='uaeq-census-q_bs-100-19epochs-psample-200-seed-0-tau-1.0-layers-4.pt'  \
        --test-query-path=test_queries/census_train_3-test-mirror.txt \
        --err-csv='./results/census_q.csv' \
        --column-masking \
        --residual
# 预测（自定义）
python eval_model.py \
        --dataset=census \
        --glob='uaeq-census-q_bs-100-4epochs-psample-200-seed-0-tau-1.0-layers-2.pt'  \
        --err-csv='./results/census_q_psample_1000.csv' \
        --psample=1000 \
        --residual \
        --direct-io \
        --column-masking 