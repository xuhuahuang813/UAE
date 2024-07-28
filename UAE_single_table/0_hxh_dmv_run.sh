#!/bin/bash
# 训练文件和测试文件需要在train和eval的代码文件中指定。路径分别是UAE_single_table/training_queries和UAE_single_table/test_queries

# UAE
# 可能需要修改的参数 dataset workload-size
# python train_uae.py \
        # --num-gpus=1 \
        # --dataset=dmv \
        # --workload-size=19000 \
        # --epochs=10 \
        # --constant-lr=5e-4 \
        # --bs=4096  \
        # --residual \
        # --layers=2 \
        # --fc-hiddens=128 \
        # --direct-io \
        # --column-masking

# 可能需要修改的参数 dataset glob err-csv psample
# python eval_model.py \
#         --dataset=dmv \
#         --glob='uae-dmv-bs-4096-9epochs-psample-200-seed-0-tau-1.0-q-weight-0.0001-layers-2.pt'  \
#         --err-csv='./results/dmv.csv' \
#         --psample=100 \
#         --residual \
#         --direct-io \
#         --column-masking 

# UAE-Q
# 可能需要修改的参数 dataset workload-size
# python train_uae.py \
#         --cuda-num=1 \
#         --num-gpus=1 \
#         --dataset=dmv \
#         --workload-size=19000 \
#         --epochs=10 \
#         --constant-lr=5e-4 \
#         --q-bs=32 \
#         --run-uaeq  \
#         --residual \
#         --layers=2 \
#         --fc-hiddens=128 \
#         --direct-io \
#         --column-masking

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