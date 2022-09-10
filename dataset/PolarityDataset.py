import numpy as np
from torch.utils.data import Dataset


class PolarityDataset(Dataset):
    """
    Polarity dataset class used during training
    """
    def __init__(self, embeds, polarities):
        """
            :param embeds: Embeddings vector
            :param polarities: Polarities vector
        """
        self.embeds = embeds
        self.polarities = polarities

    def __len__(self):
        return len(self.embeds)

    def __getitem__(self, idx):
        embed = self.embeds[idx]
        polarity = self.polarities[idx]

        return embed.astype(np.float32), polarity.astype(np.float32)
