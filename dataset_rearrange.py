import json
from tqdm import tqdm

new_data = []

with open('data/finetune_data/MobileVLM_V2_FT_Mix2M.json', 'r') as f:
# with open('data/finetune_data/data.json', 'r') as f:
    data = json.load(f)

for d in tqdm(data):
    try:
        if 'image' not in d:
            new_data.append(d)
            continue
        if d['image'].startswith('textvqa'):
            new_data.append(d)
    except:
        print(d)

with open('data/finetune_data/data.json', 'w') as file:
    json.dump(new_data, file, indent=4)
