import torch

class WelSSiSKo_QADataset(torch.utils.data.Dataset):
    def __init__(self, encodings):
        self.encodings = encodings

    def __getitem__(self, idx):
        data = {}
        for key, val in self.encodings.items():
            data[key] = torch.tensor(val[idx], dtype=torch.long)
        # return {key: torch.tensor(val[idx], dtype=torch.long) for key, val in self.encodings.items()}
        return data

    def __len__(self):
        return len(self.encodings.input_ids)