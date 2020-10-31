ckpt=$1
root=$2
out_prefix="prob"
in_ch="curv iH sulc"

env -u MKL_NUM_THREADS \
python test.py \
--ckpt ${ckpt} \
--export_file ${out_prefix} \
--test-batch-size 1 \
--data_folder ${root} \
--max_level 5 \
--min_level 0 \
--feat 32 \
--in_ch ${in_ch} \
--fmt txt