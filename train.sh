PYTHON='/home/duruoyi/anaconda3/envs/py38/bin/python3.8'

hostname
nvidia-smi

export CUDA_VISIBLE_DEVICES=0

# Get unique log file,
SAVE_DIR=/home/duruoyi/OCD/outputs/cub/

EXP_NUM=$(ls ${SAVE_DIR} | wc -l)
EXP_NUM=$((${EXP_NUM}+1))
echo $EXP_NUM

${PYTHON} -m train_smile \
            --dataset_name 'cub' \
            --batch_size 128 \
            --grad_from_block 11 \
            --epochs 100 \
            --base_model vit_dino \
            --num_workers 16 \
            --use_ssb_splits 'True' \
            --sup_con_weight 0.5 \
            --weight_decay 5e-5 \
            --contrast_unlabel_only 'False' \
            --transform 'imagenet' \
            --lr 0.01 \
            --eval_funcs 'v1' 'v2' \
> ${SAVE_DIR}logfile_${EXP_NUM}.out