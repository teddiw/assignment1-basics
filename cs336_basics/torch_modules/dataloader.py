import torch
import torch.nn as nn
import numpy as np
from einops import rearrange
from jaxtyping import Float, Int
from torch import Tensor, LongTensor
from typing import Tuple 
import numpy.typing as npt

class DataLoader(nn.Module):
    def __init__(self, 
                 batch_size: int,
                 context_length: int,
                 device: str = 'cpu',
                 data_file_path: str | None = None,
                 dataset: npt.NDArray | None = None, # for testing
                 dtype: type = 'uint16', # TODO check dtype
                 ):
        super(DataLoader, self).__init__()
        
        self.batch_size = batch_size
        self.context_length = context_length
        self.device = device
        self.dtype = dtype

        assert data_file_path or (dataset is not None)
        if (data_file_path):
            self.data_file_path = data_file_path
            self.data = np.load(data_file_path, mmap_mode='r')
            self.data = torch.from_numpy(self.data.copy())
            
        else:
            self.data = dataset

        self.data = self.data.to(dtype)
        self.curr_index = 0

        # prepare indices to extract batches from sequences of tokens
        indices = np.arange(self.context_length)
        indices = np.tile(indices, [self.batch_size, 1])
        indices_sum = rearrange(np.arange(self.batch_size), 'n -> n 1')
        self.indices = indices + indices_sum

        self.sequential_indices = np.arange(self.context_length)
        self.sequential_indices = np.tile(self.sequential_indices, [self.batch_size, 1])

    def get_random_batch(self, 
                  ) -> Tuple[Float[Tensor, "batch_size context_length"], Float[Tensor, "batch_size context_length"]]:
        # sample a valid starting index
        starting_indices = np.random.randint(0, len(self.data) - self.context_length, size=[self.batch_size, 1])
        sampled_indices = self.sequential_indices + starting_indices

        # Get the actual end index of the slice into self.data for train batches
        train_batches = self.data[sampled_indices].to(torch.int32)
        train_batches = train_batches.to(self.device)

        target_batches = self.data[sampled_indices+1].to(torch.int32)
        target_batches = target_batches.to(self.device)

        return (train_batches, target_batches)
        
        # # Uses an incremented starting index from the previous batch and deterministically chooses the next batch
        # actual_num_in_batch = self.context_length + self.batch_size
        # while (self.curr_index+actual_num_in_batch + 1 < len(self.data)):
        #     # Get the actual end index of the slice into self.data for train batches
        #     train_elements = self.data[self.curr_index:self.curr_index+actual_num_in_batch]
        #     train_elements = torch.from_numpy(train_elements)
        #     train_batches = train_elements[self.indices].to(self.device)

        #     target_elements = self.data[self.curr_index:self.curr_index+actual_num_in_batch]
        #     target_elements = torch.from_numpy(target_elements)
        #     target_batches = target_elements[self.indices+1].to(self.device)
        #     self.curr_index += actual_num_in_batch
        #     yield (train_batches, self.batch_size)

if __name__ == "__main__":
    
    data_loader = DataLoader(10, 4, dataset=np.arange(50))
    for batch in data_loader:
        print(batch[0])
        print(batch[1])
        breakpoint()
