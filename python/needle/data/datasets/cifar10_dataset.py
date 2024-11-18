import os
import pickle
from typing import Iterator, Optional, List, Sized, Union, Iterable, Any
import numpy as np
from ..data_basic import Dataset

class CIFAR10Dataset(Dataset):
    def __init__(
        self,
        base_folder: str,
        train: bool,
        p: Optional[int] = 0.5,
        transforms: Optional[List] = None
    ):
        """
        Parameters:
        base_folder - cifar-10-batches-py folder filepath
        train - bool, if True load training dataset, else load test dataset
        Divide pixel values by 255. so that images are in 0-1 range.
        Attributes:
        X - numpy array of images
        y - numpy array of labels
        """
        ### BEGIN YOUR SOLUTION
        self.transforms = [] if transforms is None else transforms
        X = []
        Y = []

        if train:
          for i in range(1,6):
            with open(os.path.join(base_folder,f"data_batch_{i}"),"rb") as fo:
              data = pickle.load(fo,encoding="bytes")
              X.append(data[b"data"])
              Y.append(data[b"labels"])

        else:
          with open(os.path.join(base_folder,"test_batch"),"rb") as fo:
            data = pickle.load(fo,encoding="bytes")
            X.append(data[b"data"])
            Y.append(data[b"labels"])
        
        X = np.concatenate(X,axis = 0)
        Y = np.concatenate(Y,axis = None)
        self.X = (X.astype(np.float32) / 255.0).reshape((-1,3,32,32))
        self.Y = Y
        self.p = p




        ### END YOUR SOLUTION

    def __getitem__(self, index) -> object:
        """
        Returns the image, label at given index
        Image should be of shape (3, 32, 32)
        """
        ### BEGIN YOUR SOLUTION
        if self.transforms:
          image = np.array([self.apply_transforms(current_img) for current_img in self.X[index]])

        else:
          image = self.X[index]

        label = self.Y[index]
        return image,label


        ### END YOUR SOLUTION

    def __len__(self) -> int:
        """
        Returns the total number of examples in the dataset
        """
        ### BEGIN YOUR SOLUTION
        return len(self.Y)
        ### END YOUR SOLUTION
