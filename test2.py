import json
import cv2
import tqdm
import numpy as np
from mobilevlm.constants import SHORT_QUESTION_LIST, LONG_QUESTION_LIST, ANSWER_LIST
import random


# Step 1: 读取 JSON 文件
with open('/media/armine/6E94666294662CB1/A_Content/Datasets/ADEChallengeData2016/training2.json', 'r') as f:
    data = json.load(f)

# Step 3: 将前 100 个元素保存为新的 JSON 文件
with open('/media/armine/6E94666294662CB1/A_Content/Datasets/ADEChallengeData2016/sample.json', 'w') as f:
    json.dump(data[:100], f, indent=4)

print("已成功保存")
