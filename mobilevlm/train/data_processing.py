import glob
import json
import os
import random
from typing import Dict, Union

import cv2
from PIL import Image
import numpy as np

from mobilevlm.constants import SHORT_QUESTION_LIST, LONG_QUESTION_LIST, ANSWER_LIST

def get_mask_from_json(json_path: str,
                       img: Union[Image.Image, np.ndarray],
                       ) -> np.ndarray:
    if isinstance(img, Image.Image):
        img = np.array(img)
    try:
        with open(json_path, "r") as r:
            anno = json.loads(r.read())
    except:
        with open(json_path, "r", encoding="cp1252") as r:
            anno = json.loads(r.read())

    inform = anno["shapes"]

    height, width = img.shape[:2]

    ### sort polies by area
    area_list = []
    valid_poly_list = []
    for i in inform:
        label_id = i["label"]
        points = i["points"]
        if "flag" == label_id.lower():  ## meaningless deprecated annotations
            continue

        tmp_mask = np.zeros((height, width), dtype=np.uint8)
        cv2.polylines(tmp_mask, np.array([points], dtype=np.int32), True, 1, 1)
        cv2.fillPoly(tmp_mask, np.array([points], dtype=np.int32), 1)
        tmp_area = tmp_mask.sum()

        area_list.append(tmp_area)
        valid_poly_list.append(i)

    ### ground-truth mask
    sort_index = np.argsort(area_list)[::-1].astype(np.int32)
    sort_index = list(sort_index)
    sort_inform = []
    for s_idx in sort_index:
        sort_inform.append(valid_poly_list[s_idx])

    mask = np.zeros((height, width), dtype=np.uint8)
    for i in sort_inform:
        label_id = i["label"]
        points = i["points"]

        if "ignore" in label_id.lower():
            label_value = 255  # ignored during evaluation
        else:
            label_value = 1  # target

        cv2.polylines(mask, np.array([points], dtype=np.int32), True, label_value, 1)
        cv2.fillPoly(mask, np.array([points], dtype=np.int32), label_value)

    return mask


def meta_info_retrieve(sample: Dict,
                       json_path: str,
                       num_classes_per_sample: int):
    try:
        with open(json_path, "r") as r:
            anno = json.loads(r.read())
    except:
        with open(json_path, "r", encoding="cp1252") as r:
            anno = json.loads(r.read())
    sents, is_sentence = anno["text"], anno["is_sentence"]
    if len(sents) >= num_classes_per_sample:
        sampled_indices = np.random.choice(
            list(range(len(sents))), size=num_classes_per_sample, replace=False
        )
    else:
        sampled_indices = list(range(len(sents)))
    sampled_sents = np.vectorize(sents.__getitem__)(sampled_indices).tolist()
    questions = []
    answers = []
    for text in sampled_sents:
        if is_sentence:
            question_template = random.choice(LONG_QUESTION_LIST)
            q = {'from': 'human', 'value': question_template.format(sent=text)}
        else:
            question_template = random.choice(SHORT_QUESTION_LIST)
            q = {'from': 'human', 'value': question_template.format(class_name=text.lower())}
        questions.append(q)
        a = {'from': 'gpt', 'value': random.choice(ANSWER_LIST)}
        answers.append(a)
    conversations = []
    for question, answer in zip(questions, answers):
        conversations.extend([question, answer])
    sample['conversations'] = conversations
    sample['sampled_indices'] = sampled_indices


if __name__ == "__main__":
    data_dir = "./train"
    vis_dir = "./vis"

    if not os.path.exists(vis_dir):
        os.makedirs(vis_dir)

    json_path_list = sorted(glob.glob(data_dir + "/*.json"))
    for json_path in json_path_list:
        img_path = json_path.replace(".json", ".jpg")
        img = cv2.imread(img_path)[:, :, ::-1]

        # In generated mask, value 1 denotes valid target region, and value 255 stands for region ignored during evaluaiton.
        mask, comments, is_sentence = get_mask_from_json(json_path, img)

        ## visualization. Green for target, and red for ignore.
        valid_mask = (mask == 1).astype(np.float32)[:, :, None]
        ignore_mask = (mask == 255).astype(np.float32)[:, :, None]
        vis_img = img * (1 - valid_mask) * (1 - ignore_mask) + (
            (np.array([0, 255, 0]) * 0.6 + img * 0.4) * valid_mask
            + (np.array([255, 0, 0]) * 0.6 + img * 0.4) * ignore_mask
        )
        vis_img = np.concatenate([img, vis_img], 1)
        vis_path = os.path.join(
            vis_dir, json_path.split("/")[-1].replace(".json", ".jpg")
        )
        cv2.imwrite(vis_path, vis_img[:, :, ::-1])
        print("Visualization has been saved to: ", vis_path)
