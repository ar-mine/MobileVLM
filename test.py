import json
import cv2
import tqdm
import numpy as np
from mobilevlm.constants import SHORT_QUESTION_LIST, LONG_QUESTION_LIST, ANSWER_LIST
import random


ade20k_classes = [
    "wall", "building", "sky", "floor", "tree", "ceiling", "road",
    "bed", "windowpane", "grass", "cabinet", "sidewalk",
    "person", "earth", "door", "table", "mountain", "plant",
    "curtain", "chair", "car", "water", "painting", "sofa",
    "shelf", "house", "sea", "mirror", "rug", "field", "armchair",
    "seat", "fence", "desk", "rock", "wardrobe", "lamp",
    "bathtub", "railing", "cushion", "base", "box", "column",
    "signboard", "chest of drawers", "counter", "sand", "sink",
    "skyscraper", "fireplace", "refrigerator", "grandstand",
    "path", "stairs", "runway", "case", "pool table", "pillow",
    "screen door", "stairway", "river", "bridge", "bookcase",
    "blind", "coffee table", "toilet", "flower", "book", "hill",
    "bench", "countertop", "stove", "palm", "kitchen island",
    "computer", "swivel chair", "boat", "bar", "arcade machine",
    "hovel", "bus", "towel", "light", "truck", "tower",
    "chandelier", "awning", "streetlight", "booth",
    "television receiver", "airplane", "dirt track", "apparel",
    "pole", "land", "bannister", "escalator", "ottoman", "bottle",
    "buffet", "poster", "stage", "van", "ship", "fountain",
    "conveyer belt", "canopy", "washer", "plaything",
    "swimming pool", "stool", "barrel", "basket", "waterfall",
    "tent", "bag", "minibike", "cradle", "oven", "ball", "food",
    "step", "tank", "trade name", "microwave", "pot", "animal",
    "bicycle", "lake", "dishwasher", "screen", "blanket",
    "sculpture", "hood", "sconce", "vase", "traffic light",
    "tray", "ashcan", "fan", "pier", "crt screen", "plate",
    "monitor", "bulletin board", "shower", "radiator", "glass",
    "clock", "flag"
]


# Step 1: 读取 JSON 文件
with open('/media/armine/6E94666294662CB1/A_Content/Datasets/ADEChallengeData2016/training.json', 'r') as f:
    data = json.load(f)

new_data = []
for sample in tqdm.tqdm(data):
    image_path, label_path = sample['image'], sample['annotation']
    label = cv2.imread(label_path, 0)
    # Process annotation
    label[label == 0] = 255
    label -= 1
    label[label == 254] = 255
    unique_label = np.unique(label).tolist()
    if len(unique_label) > 1:
        if 255 in unique_label:
            unique_label.remove(255)

        questions = []
        answers = []
        # Generate grounding sentence
        for label_idx in unique_label:
            class_name = ade20k_classes[label_idx]

            question_template = random.choice(SHORT_QUESTION_LIST)
            q = {'from': 'human', 'value': question_template.format(class_name=class_name.lower())}
            a = {'from': 'gpt', 'value': random.choice(ANSWER_LIST)}

            new_data.append({
                'image': image_path,
                'annotation': label_path,
                'conversations': [q, a],
                'sampled_indices': [label_idx]
            })

# Step 3: 将前 100 个元素保存为新的 JSON 文件
with open('/media/armine/6E94666294662CB1/A_Content/Datasets/ADEChallengeData2016/training2.json', 'w') as f:
    json.dump(new_data, f, indent=4)

print("已成功保存")
