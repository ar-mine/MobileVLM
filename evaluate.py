import argparse
import os
import sys

import cv2
import numpy as np
import torch
import json
from transformers import AutoTokenizer, BitsAndBytesConfig, CLIPImageProcessor

from mobilevlm.train.preprocess import preprocess_sam, preprocess_ade
from mobilevlm.model.mobilelisa import MobileLisaForCasualLM, sigmoid_ce_loss
from mobilevlm import conversation as conversation_lib
from mobilevlm.utils import tokenizer_image_token
from mobilevlm.model.segment_anything.utils.transforms import ResizeLongestSide
from mobilevlm.constants import (DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN,
                                 DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX, ADE20K_CLASSES)

DEBUG = True


def parse_args(args):
    parser = argparse.ArgumentParser(description="LISA chat")
    parser.add_argument("--model_path", default="xinlai/LISA-13B-llama2-v1")
    parser.add_argument("--vis_save_path", default="./vis_output", type=str)
    parser.add_argument(
        "--precision",
        default="bf16",
        type=str,
        choices=["fp32", "bf16", "fp16"],
        help="precision for inference",
    )
    parser.add_argument("--image_size", default=1024, type=int, help="image size")
    parser.add_argument("--model_max_length", default=512, type=int)
    parser.add_argument("--lora_r", default=8, type=int)
    parser.add_argument(
        "--vision-tower", default="openai/clip-vit-large-patch14", type=str
    )
    parser.add_argument("--local-rank", default=0, type=int, help="node rank")
    parser.add_argument("--load_in_8bit", action="store_true", default=False)
    parser.add_argument("--load_in_4bit", action="store_true", default=False)
    parser.add_argument("--use_mm_start_end", action="store_true", default=False)
    parser.add_argument(
        "--conv_type",
        default="v1",
        type=str,
        choices=["v1", "llava_llama_2"],
    )
    parser.add_argument("--image_path", type=str)
    parser.add_argument("--gt_path", type=str)
    return parser.parse_args(args)


def main(args):
    # Get parameters and prepare output folder
    # TODO: Keep training parameters same as inference
    args = parse_args(args)

    # Create tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        cache_dir=None,
        model_max_length=args.model_max_length,
        padding_side="right",
        use_fast=False,
    )
    tokenizer.pad_token = tokenizer.unk_token
    args.seg_token_idx = tokenizer("[SEG]", add_special_tokens=False).input_ids[0]

    # Load parameters for different precision
    torch_dtype = torch.float32
    if args.precision == "bf16":
        torch_dtype = torch.bfloat16
    elif args.precision == "fp16":
        torch_dtype = torch.half
    kwargs = {"torch_dtype": torch_dtype}
    if args.load_in_4bit:
        kwargs.update(
            {
                "torch_dtype": torch.half,
                "load_in_4bit": True,
                "quantization_config": BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                    llm_int8_skip_modules=["visual_model"],
                ),
            }
        )
    elif args.load_in_8bit:
        kwargs.update(
            {
                "torch_dtype": torch.half,
                "quantization_config": BitsAndBytesConfig(
                    llm_int8_skip_modules=["visual_model"],
                    load_in_8bit=True,
                ),
            }
        )

    # Load the main model
    model = MobileLisaForCasualLM.from_pretrained(
        args.model_path,
        low_cpu_mem_usage=True,
        vision_tower=args.vision_tower,
        seg_token_idx=args.seg_token_idx,
        train_mask_decoder = False,
        inference=True,
        **kwargs
    )

    model.config.eos_token_id = tokenizer.eos_token_id
    model.config.bos_token_id = tokenizer.bos_token_id
    model.config.pad_token_id = tokenizer.pad_token_id
    # TODO: Make sure the CLIP module match
    model.get_model().initialize_vision_modules(model.get_model().config)
    # TODO: remove unused segment
    vision_tower = model.get_model().get_vision_tower()
    vision_tower.to(dtype=torch_dtype)

    # Transfer model to target precision
    if args.precision == "bf16":
        model = model.bfloat16().cuda()
    elif (
        args.precision == "fp16" and (not args.load_in_4bit) and (not args.load_in_8bit)
    ):
        vision_tower = model.get_model().get_vision_tower()
        model.model.vision_tower = None
        import deepspeed

        model_engine = deepspeed.init_inference(
            model=model,
            dtype=torch.half,
            replace_with_kernel_inject=True,
            replace_method="auto",
        )
        model = model_engine.module
        model.model.vision_tower = vision_tower.half().cuda()
    elif args.precision == "fp32":
        model = model.float().cuda()

    vision_tower = model.get_model().get_vision_tower()
    vision_tower.to(device=args.local_rank)

    clip_image_processor = CLIPImageProcessor.from_pretrained(model.config.vision_tower)
    transform = ResizeLongestSide(args.image_size)
    state_dict = torch.load(os.path.join(args.model_path, "pytorch_model.bin"), map_location="cpu")
    model.load_state_dict(state_dict, strict=True)

    model.eval()

    with open('/media/armine/6E94666294662CB1/A_Content/Datasets/ADEChallengeData2016/sample.json', 'r') as f:
        ori_data = json.load(f)

    for data in ori_data:
        image_path = data["image"]
        image_np = cv2.imread(image_path)
        image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
        gt_path = data["annotation"]
        gt_np = preprocess_ade(cv2.imread(gt_path, 0))

        prompt = data["conversations"][0]["value"]
        gt_idx = data["sampled_indices"][0]
        class_name = ADE20K_CLASSES[gt_idx]
        conv = conversation_lib.conv_templates[args.conv_type].copy()
        conv.messages = []

        #prompt = DEFAULT_IMAGE_TOKEN + "\n" + f"Can you segment the {class_name} in this image?"
        if args.use_mm_start_end:
            replace_token = (
                DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
            )
            prompt = prompt.replace(DEFAULT_IMAGE_TOKEN, replace_token)

        conv.append_message(conv.roles[0], prompt)
        conv.append_message(conv.roles[1], "")
        prompt = conv.get_prompt()

        original_size_list = [image_np.shape[:2]]

        image_clip = (
            clip_image_processor.preprocess(image_np, return_tensors="pt")[
                "pixel_values"
            ][0]
            .unsqueeze(0)
            .cuda()
        )
        if args.precision == "bf16":
            image_clip = image_clip.bfloat16()
        elif args.precision == "fp16":
            image_clip = image_clip.half()
        else:
            image_clip = image_clip.float()

        image = transform.apply_image(image_np)
        resize_list = [image.shape[:2]]

        image = (
            preprocess_sam(torch.from_numpy(image).permute(2, 0, 1).contiguous())
            .unsqueeze(0)
            .cuda()
        )
        if args.precision == "bf16":
            image = image.bfloat16()
        elif args.precision == "fp16":
            image = image.half()
        else:
            image = image.float()

        input_ids = tokenizer_image_token(prompt, tokenizer, return_tensors="pt")
        input_ids = input_ids.unsqueeze(0).cuda()

        output_ids, pred_masks = model.evaluate(
            image_clip,
            image,
            input_ids,
            resize_list,
            original_size_list,
            max_new_tokens=512,
            tokenizer=tokenizer,
        )
        output_ids = output_ids[0][output_ids[0] != IMAGE_TOKEN_INDEX]


        assert len(pred_masks) == 1

        pred_mask = pred_masks[0].detach().cpu().numpy()[0]
        #pred_mask = pred_mask > 0

        gt_mask = (gt_np == gt_idx)
        loss = sigmoid_ce_loss(torch.Tensor([pred_mask]), torch.Tensor([gt_mask]), num_masks=1)

        print(f"Testing {class_name}: {loss.item()}")

if __name__ == "__main__":
    main(sys.argv[1:])
