import json


with open('data/finetune_data/data.json', 'r') as f:
    data_ori = json.load(f)


with open('data/finetune_data/index.json', 'r') as f:
    data_add = json.load(f)


with open('data/finetune_data/index.json', 'w') as f:
    json.dump(data_add+data_ori, f, indent=4)