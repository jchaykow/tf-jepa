import torch, collections, json
labels = torch.load("./datasets/Gesture/train.pt")["labels"].tolist()
freq   = collections.Counter(labels)
print(json.dumps(freq, indent=2))