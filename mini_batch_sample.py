import json

SAMPLE_MAX = 10
CLASSES = 10

counter = [SAMPLE_MAX for _ in range(CLASSES)]
new_data = []

with open('/media/armine/6E94666294662CB1/A_Content/Datasets/ADEChallengeData2016/training2.json', 'r') as f:
    ori_data = json.load(f)

for data in ori_data:
    idx = data["sampled_indices"][0]
    if idx > SAMPLE_MAX-1:
        continue
    if counter[idx] > 0:
        counter[idx] -= 1
        new_data.append(data)
    if sum(counter) == 0:
        break

with open('/media/armine/6E94666294662CB1/A_Content/Datasets/ADEChallengeData2016/sample.json', 'w') as f:
    json.dump(new_data, f, indent=4)