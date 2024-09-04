WORK_DIR=$(cd "$(dirname "$0")";pwd)
export PYTHONPATH=${WORK_DIR}
OUTPUT_DIR_FT=results/mobilevlm_v2-2.finetune-lora
python mobilevlm/train/train_mem.py \
        --lora_enable True \
        --lora_r 8 \
        --lora_alpha 16 \
        --learning_rate 3e-4 \
        --weight_decay 0. \
        --model_name_or_path ../MobileVLM_V2-3B \
        --version v1 \
        --data_path /media/armine/6E94666294662CB1/A_Content/Datasets/ADEChallengeData2016 \
        --image_folder data/finetune_data \
        --vision_tower openai/clip-vit-large-patch14-336 \
        --vision_tower_type clip \
        --mm_projector_type ldpnetv2 \
        --mm_vision_select_layer -2 \
        --mm_use_im_start_end False \
        --mm_use_im_patch_token False \
        --mm_use_seg True \
        --image_aspect_ratio pad \
        --group_by_modality_length True \
        --bf16 True \
        --output_dir ${OUTPUT_DIR_FT} \
        --num_train_epochs 1 \
        --per_device_train_batch_size 16 \
        --per_device_eval_batch_size 1 \
        --gradient_accumulation_steps 1 \
        --evaluation_strategy "no" \
        --save_strategy "steps" \
        --save_steps 50000 \
        --save_total_limit 1 \
        --warmup_ratio 0.03 \
        --lr_scheduler_type "cosine" \
        --logging_steps 1 \
        --log_level "info" \
        --tf32 True \
        --model_max_length 2048 \
        --gradient_checkpointing True \
        --dataloader_num_workers 8 \
        --lazy_preprocess True \
        --report_to none \
        --segment_label True \
        --segment_encoder_path "../sam_vit_h_4b8939.pth" \
        --train_mask_decoder true \
        --wandb_enable true\
        2>&1 | tee -a ${OUTPUT_DIR_FT}/log.txt