data_folder="/home/user/sparc/data"
log_dir="/home/user/sparc/model"
classes="0 1 2 3 4 6 7 8 10"
in_ch="curv iH sulc"

export CUDA_VISIBLE_DEVICES=0

python train.py \
--batch-size 4 \
--test-batch-size 10 \
--epochs 30 \
--data_folder ${data_folder} \
--max_level 5 \
--min_level 0 \
--feat 32 \
--log_dir ${log_dir} \
--log-interval 70 \
--fold 1 \
--lr 0.01 \
--classes ${classes} \
--in_ch ${in_ch} \
--deg 15 \
--seed 1 \
--hemi lh \
--drop 0