import torch.utils.data as data


class COCO(data.Dataset):
    def __init__(self, root):
        raise NotImplementedError("TODO")

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: Tuple (image, target). target is a list of captions for the
                   image.
        """
        raise NotImplementedError("TODO")

    def __len__(self):
        raise NotImplementedError("TODO")