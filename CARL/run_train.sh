DATA_DIR=data/
OUTPUT_DIR=saved/

LR=2e-7

MODEL_DIR=/path/to/Meta-Llama-3.1-8B-Instruct
CUDA_VISIBLE_DEVICES=3 python train_policy.py \
    --base_model ${MODEL_DIR} \
    --output_dir ${OUTPUT_DIR} \
    --batch_size 2 \
    --micro_batch_size 2 \
    --num_epochs 5 \
    --learning_rate ${LR} \
    --cutoff_len 1000 \
    --eval_steps 50 \
    --save_steps 50 \
    --val_set_size 6000 \
    --lora_r 64 \
    --lora_alpha 16 \
    --lora_dropout 0.05 \
    --lora_target_modules '[q_proj,v_proj,k_proj,o_proj]' \
    --train_on_inputs false \
    --group_by_length \
    --save_total_limit 10 \
    --seed 201 \