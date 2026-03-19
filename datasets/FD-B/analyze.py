import torch, collections, json
labels = torch.load("./datasets/FD-B/train.pt")["labels"].tolist()
freq   = collections.Counter(labels)
print(json.dumps(freq, indent=2))