deepspeed mobilevlm/train/train_mem.py \
            --deepspeed scripts/deepspeed/zero2.json \
            --lora_enable True \
            --lora_r 8 \
            --lora_alpha 16 \
            --learning_rate 2e-4 \
            --model_name_or_path ../MobileVLM_V2-3B \
            --version v1 \
            --data_path data/finetune_data/data.json \
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
            --output_dir results/mobilevlm_v2-2.finetune-lora \
            --num_train_epochs 1 \
            --per_device_train_batch_size 8 \
            --per_device_eval_batch_size 1 \
            --gradient_accumulation_steps 1 \
            --evaluation_strategy "no" \
            --save_strategy "steps" \
            --save_steps 50000 \
            --save_total_limit 1 \
            --weight_decay 0. \
            --warmup_ratio 0.03 \
            --lr_scheduler_type "cosine" \
            --logging_steps 1 \
            --tf32 True \
            --model_max_length 2048 \
            --gradient_checkpointing True \
            --dataloader_num_workers 8 \
            --lazy_preprocess True \
            --report_to none