import json
import os
import numpy as np
import torch


def get_data(fold, split, representation_level, coords=0):
    # if split == "validation":
    #     split_append = "val"
    # elif split == "testing":
    #     split_append = "test"
    # else:
    #     split_append = "train"

    # Path to the data
    # /home/schobs/Documents/PhD/WeiGP/Mini-GP/data/fold0.json
    data_path = f"data/fold{fold}/{split}/{representation_level}/"
    # Load the JSON file
    json_path = f"data/fold{fold}.json"
    with open(json_path) as f:
        data_json = json.load(f)

    # Extracting the samples for the given split
    data_split = data_json[split]

    # Dictionaries to store tensors and labels
    tensors = []
    labels = []

    # Read the tensor files and labels
    counter = 0
    for sample in data_split:
        uid = sample["id"]
        tensor_path = os.path.join(data_path, f"{uid}.pth")
        tensor = torch.tensor(torch.load(tensor_path).flatten(), dtype=torch.float32)
        tensors.append(tensor)

        if coords == "all":
            label = torch.tensor(sample["coordinates"], dtype=torch.float32)
        else:
            label = torch.tensor(sample["coordinates"][coords], dtype=torch.float32)

        labels.append(label)
        counter += 1
        # if counter > 10:
        #     break

    tensors = torch.stack(tensors)
    labels = torch.stack(labels)
    return tensors, labels
