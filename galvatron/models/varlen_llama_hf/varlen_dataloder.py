import os
import torch
import torch.distributed
from torch.utils.data import Dataset
import numpy as np
from galvatron.core import get_args
from tqdm import tqdm

class DataLoaderForVarlenLlama(Dataset):
    def __init__(self, args, device):
        self.vocab_size = args.vocab_size
        self.sentence_length = args.seq_length#sentence length用的sequeceng elngth做的替代
        self.dataset_size = 4000
        self.device = device
        world_size = torch.distributed.get_world_size()#他这里就是就求了觉得很奇怪
        self.input_ids = []
        if args.dataset == "fix_length":
            self.data_length = np.full((self.dataset_size,), self.sentence_length)
        elif args.dataset == "random":
            raw_lengths = np.random.randint(2,self.sentence_length,(self.dataset_size,))
            # Pad lengths to be multiples of 2*world_size for CP/SP compatibility
            align = 2 * world_size
            self.data_length = ((raw_lengths - 1) // align + 1) * align
            self.data_length = np.minimum(self.data_length, self.sentence_length)
        else:
            text_length = []
            tmp = 0
            # Search for dataset in multiple locations
            _script_dir = os.path.dirname(os.path.abspath(__file__))
            _dataset_search_paths = [
                getattr(args, 'dataset_dir', ''),
                os.path.join(_script_dir, '..', '..', '..', 'varlen_datasets'),
                os.path.join(_script_dir, '..', '..', '..', '..', 'varlen_datasets'),
                "/home/pkuhetu/lqs/flexsp/Hetu-Galvatron/galvatron/datasets",
            ]
            _dataset_file = None
            for _dp in _dataset_search_paths:
                _candidate = os.path.join(_dp, f"{args.dataset}.txt")
                if os.path.exists(_candidate):
                    _dataset_file = _candidate
                    break
            if _dataset_file is None:
                raise FileNotFoundError(
                    f"Dataset '{args.dataset}.txt' not found in: {_dataset_search_paths}")
            with open(_dataset_file, "r") as f:
                for i, line in enumerate(tqdm(f, total=self.dataset_size, desc="Loading text lengths")):
                    if i >= self.dataset_size + tmp:
                        break
                    sentence_len = int(line.strip())
                    pad_len = ((sentence_len - 1) // (2 * world_size) + 1) * (2 * world_size) #force the length to be multiple of 2 * world_size
                    if pad_len > (args.seq_length - 2 * torch.distributed.get_world_size()):
                        tmp += 1
                        continue
                    text_length.append(min(pad_len, args.seq_length - 2 * torch.distributed.get_world_size()))
            self.data_length = np.array(text_length)#data length是重要的
        
        for i in range(self.dataset_size):
            sentence = np.zeros(self.data_length[i])
            self.input_ids.append(sentence)#然后是数据  #为什么这个数据是假的
        
        # self.input_ids = np.array(self.input_ids)

    def __len__(self):
        return self.dataset_size

    def __getitem__(self, idx):
        if idx >= self.dataset_size:
            raise IndexError
        input_ids = torch.LongTensor(self.input_ids[idx]).to(self.device)
        return input_ids