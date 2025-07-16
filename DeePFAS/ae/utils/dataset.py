import pickle
from typing import Callable

from pyteomics import mgf
from torch.utils.data import Dataset

from ..config.models import Ms2VecConfig


class ValDataset(Dataset):
    """Dataset used evaluate performance."""

    def __init__(
        self,
        data_file: str,
        use_record_idx: bool,
        train_set_ratio: float,
        dataset_idx_path: str,
        config: Ms2VecConfig,
        transform: Callable = lambda s: s  # noqa:WPS404
    ):
        """
            Initialize valid dataset.

        Args:
            data_file:
                dataset file path
            use_record_idx:
                use existing idx file
            train_set_ratio:
                dataset ratio of training set
            transform:
                transform func of return data
            dataset_idx_path:
                idx file path
            config:
                config of training set
        """
        self.transform = transform
        if use_record_idx:
            with open(dataset_idx_path, 'rb') as f:
                self.offsets = pickle.load(f)
        else:
            self.offsets = [0]
            with open(data_file, 'r', encoding='utf-8') as fp:
                while fp.readline() != '':
                    self.offsets.append(fp.tell())
            self.offsets.pop()
        sample_size = config.hparams.sample_size
        self.offsets = self.offsets[int(sample_size * train_set_ratio):sample_size]
        self.fp = open(data_file, 'rb')

    def __len__(self):
        return len(self.offsets)

    def __getitem__(self, idx):
        self.fp.seek(self.offsets[idx], 0)
        return self.transform(self.fp.readline())

class SpectrumDataset(Dataset):
    def __init__(self,
                 data_file: str,
                 transform: Callable):
        self.transform = transform
        self.data = mgf.read(data_file)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.transform(self.data[idx])

    def close_file(self):
        self.data.close()

class InferenceSpectrumDataset(Dataset):
    def __init__(self,
                 data_file: str,
                 transform: Callable,
                 device,
                 loss_mz_from,
                 loss_mz_to,
                 ce_range,
                 mass_range,
                 max_num_peaks,
                 min_num_peaks,
                 resolution,
                 ignore_MzRange,
                 ignore_CE):
        self.transform = transform
        self.data = mgf.read(data_file)
        self.device = device
        self.loss_mz_from = loss_mz_from
        self.loss_mz_to = loss_mz_to
        self.max_num_peaks = max_num_peaks
        self.min_num_peaks = min_num_peaks
        self.resolution = resolution
        self.ce_range = ce_range
        self.mass_range = mass_range
        self.ignore_MzRange = ignore_MzRange
        self.ignore_CE = ignore_CE
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        return self.transform(self.data[idx],
                              self.device, 
                              self.loss_mz_from, 
                              self.loss_mz_to, 
                              self.max_num_peaks, 
                              self.min_num_peaks,
                              self.ce_range,
                              self.mass_range,
                              self.resolution,
                              self.ignore_MzRange,
                              self.ignore_CE)
    def close_file(self):
        self.data.close()

class Spectrum4ChannelsDataset(Dataset):
    def __init__(self,
                 data_file: str,
                 transform: Callable):
        self.transform = transform
        self.data = mgf.read(data_file)
        self.seek = -1
        self.num_data = len(self.data)
    def __len__(self):
        return self.num_data
    def __getitem__(self, idx):

        self.seek = (self.seek + 1) % self.num_data
        first_data = self.data[self.seek]
        left, right = min(self.seek + 1, self.num_data), min(self.seek + 4, self.num_data)
        spec_list = []
        spec_list.append(first_data)
        for i in range(left, right):
            if first_data['params']['canonicalsmiles'] == self.data[i]['params']['canonicalsmiles']:
                self.seek += 1
                spec_list.append(self.data[i])
            else:
                break
        return self.transform(spec_list)

    def close_file(self):
        self.data.close()

class SmilesDataset(Dataset):
    def __init__(self,
                 data_file: str,
                 data_idx_file: str,
                 test_ratio: float):
        with open(data_idx_file, 'rb') as idx_f:
            idx_list = pickle.load(idx_f)
            idx_list = idx_list[:int(len(idx_list) * test_ratio)]
        self.data = []
        with open(data_file, 'r') as data:
            for _ in range(len(idx_list)):
                try:
                    line = data.readline().replace('\n', '').strip()
                    self.data.append(line)
                except:
                    raise ValueError('test set readline error')

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]
