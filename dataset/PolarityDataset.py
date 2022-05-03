import numpy as np
from torch.utils.data import Dataset


class PolarityDataset(Dataset):
    def __init__(self, embeds, polarities):
        self.embeds = embeds
        self.polarities = polarities
        # self.input_val = list(input.values())

    def __len__(self):
        return len(self.embeds)
        # return len(self.input_val)

    def __getitem__(self, idx):
        embed = self.embeds[idx]
        polarity = self.polarities[idx]

        return embed.astype(np.float32), polarity.astype(np.float32)
