import math
import pickle
import time
from multiprocessing import Lock, Manager, Process
from typing import Callable, List

import numpy as np
import torch
from pyteomics import mgf
from torch.utils.data import DataLoader, Dataset

from ..config.models import Hparams, Ms2VecConfig
from .batch import batch
from .dataset import Spectrum4ChannelsDataset, SpectrumDataset, ValDataset
from .sampler import SequentialSampler


class WrappedDataLoader(object):
    def __init__(
        self, 
        dataloader: DataLoader, 
        func: Callable
    ):
        self.dataloader = dataloader
        self.func = func 

    def __len__(self):
        return len(self.dataloader)

    def __iter__(self):
        iter_dataloader = iter(self.dataloader)
        yield from (
            self.func(*batch) for batch in iter_dataloader
        )

class BufCtrl(object):

    def __init__(
        self,
        read_count_global,
        dataset_idx_path,
        batch_size,
        dataset_path,
        buffer_single,
        buffer_batch,
        buffer_lock,
        batch_buffer_lock,
        instances_buffer_size,
        buffer_max_batch,
        batch_collate_fn,
        ins_collate_fn,
        boundary,
        use_properties
    ):
        """
            Initialize buffer controller.

        Args:   
            read_count_global:
                global count 
            dataset_idx_path:
                idx file path
            batch_size:
                batch size
            dataset_path:
                dataset file path
            buffer_single:
                buffer used to store samples
            buffer_batch:
                buffer used to store batch samples
            buffer_lock:
                instances buffer's lock
            batch_buffer_lock:
                batch instances buffer's lock
            instances_buffer_size:
                capacity of buffer
            buffer_max_batch:
                max capacity of batch buffer
            collate_fn:
                func to transform return data
            boundary:
                num of train dataset
        """
        self.read_count_global = read_count_global
        self.dataset_idx_path = dataset_idx_path
        self.batch_size = batch_size
        self.dataset_path = dataset_path
        self.buffer_single = buffer_single
        self.buffer_batch = buffer_batch
        self.buffer_lock = buffer_lock
        self.batch_buffer_lock = batch_buffer_lock
        self.instances_buffer_size = instances_buffer_size
        self.batch_collate_fn = batch_collate_fn
        self.ins_collate_fn = ins_collate_fn
        self.boundary = boundary
        self.buffer_max_batch = buffer_max_batch
        self.use_properties = use_properties
    def buf_thread(self, process, num_process):
        print('=========start buf thread=========')
        read_count = 0
        while True:
            with open(self.dataset_idx_path, 'rb') as f:
                dataset_idx_list = pickle.load(f)
            dataset_idx_list = dataset_idx_list[:self.boundary]
            f_read = open(self.dataset_path, 'rb')
            num_data = len(dataset_idx_list)
            try:
                while True:

                    if len(self.buffer_single) >= self.instances_buffer_size * 0.2:
                        self._fill_batch()

                    if read_count >= num_data:
                        break
                    start_idx = dataset_idx_list[read_count]
                    if len(self.buffer_single) >= self.instances_buffer_size:
                        continue
                    read_count += 1

                    if read_count % num_process != process:  # skip
                        continue

                    self.read_count_global.value = read_count
                    f_read.seek(start_idx, 0)
                    self.buffer_single.append(self.ins_collate_fn(f_read.readline(), self.use_properties))
                f_read.close()
                read_count = 0
            except ValueError as e:
                f_read.close()
                print('DataLoader buffer thread Error !!')
                print('Error Msg', e)

    def _fill_batch(self):

        if len(self.buffer_batch) < self.buffer_max_batch and len(self.buffer_single) > self.instances_buffer_size * 0.5:

            if not self.buffer_lock.acquire(False):
                time.sleep(0.1)
                return
            if len(self.buffer_batch) < self.buffer_max_batch:
                num_data = len(self.buffer_single)
                batch_idx = np.random.choice(num_data, self.batch_size, replace=num_data < self.batch_size)
                batch_idx.sort()
                instances = [self.buffer_single.pop(i) for i in batch_idx[::-1]]
                self.buffer_batch.append(self.batch_collate_fn(instances, self.use_properties))
            self.buffer_lock.release()

class MultiprocessDataLoader(object):
    """Multiprocessing (load data)."""

    def __init__(self, 
                 hparams: Hparams, 
                 ins_collate_fn: Callable, 
                 batch_collate_fn: Callable,
                 to_device_collate_fn: Callable,
                 dataset: ValDataset = None,
                 dataset_path: str = None, 
                 dataset_idx_path: str = None,  
                 is_train: bool = True):

        self.ins_collate_fn = ins_collate_fn
        self.batch_collate_fn = batch_collate_fn
        self.to_device_collate_fn = to_device_collate_fn
        self.dataset_path = dataset_path
        self.dataset_idx_path = dataset_idx_path
        self.is_train = is_train
        self.batch_size = hparams.batch_size
        self.use_properties = hparams.use_properties
        if is_train:
            self.instances_buffer_size = hparams.instances_buffer_size
            self.buffer_max_batch = hparams.buffer_max_batch
            self.num_process = hparams.train_num_workers
            manager = Manager()
            self.buffer_single = manager.list()
            self.buffer_batch = manager.list()
            self.buffer_lock = Lock()
            self.batch_buffer_lock = Lock()
            self.buffer_thread = None
            self.read_count_global = manager.Value('i', 0)
            self.sample_size = hparams.sample_size
            self.boundary = int(self.sample_size * hparams.train_ratio)
            self.num_batches = math.ceil(self.boundary / self.batch_size)
            self.iscomplete = False
        else:
            self.dataset = dataset
            self.num_batches = math.ceil(len(dataset) / hparams.batch_size)

    def get_idx(self, sampler):
        yield from (
            idx for idx in sampler
        )

    def __len__(self):
        return self.num_batches

    def kill_process(self):
        if self.buffer_thread is not None:
            for thread in self.buffer_thread:
                print('Terminating process {0}'.format(thread.pid))
                thread.terminate()
            self.iscomplete = True

    def set_start_idx(self, idx):
        self.read_count_global.value = idx

    def get_current_read_count(self):
        return self.read_count_global.value

    def _fill_buf(self):
        if self.buffer_thread is None:
            self.buffer_thread = []
            buf_ctrl = BufCtrl(
                read_count_global=self.read_count_global,
                dataset_idx_path=self.dataset_idx_path,
                batch_size=self.batch_size,
                dataset_path=self.dataset_path,
                buffer_single=self.buffer_single,
                buffer_batch=self.buffer_batch,
                buffer_lock=self.buffer_lock,
                batch_buffer_lock=self.batch_buffer_lock,
                instances_buffer_size=self.instances_buffer_size,
                buffer_max_batch=self.buffer_max_batch,
                batch_collate_fn=self.batch_collate_fn,
                ins_collate_fn=self.ins_collate_fn,
                boundary=self.boundary,
                use_properties=self.use_properties
            )
            for process in range(self.num_process):
                buffer_thread = Process(target=buf_ctrl.buf_thread, args=(process, self.num_process))
                buffer_thread.start()
                self.buffer_thread.append(buffer_thread)

    def __iter__(self):
        try:
            if self.is_train:
                if self.buffer_thread is None:
                    self._fill_buf()
                while not self.iscomplete:
                    if self.buffer_batch:
                        yield self.to_device_collate_fn(self.buffer_batch.pop(0), self.use_properties)
            else:
                sampler = SequentialSampler(self.dataset)
                reader = batch(
                    self.get_idx(sampler), 
                    batch_size=self.batch_size
                )
                for batch_indices in reader():
                    samples = self.batch_collate_fn([self.ins_collate_fn(self.dataset[idx], self.use_properties) for idx in batch_indices], self.use_properties)
                    yield self.to_device_collate_fn(samples, self.use_properties)
        except KeyboardInterrupt:
            self.kill_process()

class SpectrumBufCtrl(object):
    def __init__(self,
                 batch_size,
                 dataset_path,
                 buffer_single,
                 buffer_batch,
                 buffer_lock,
                 instances_buffer_size,
                 buffer_max_batch,
                 collate_fn):
        self.dataset_path = dataset_path
        self.batch_size = batch_size
        self.buffer_single = buffer_single
        self.buffer_batch = buffer_batch
        self.buffer_lock = buffer_lock
        self.instances_buffer_size = instances_buffer_size
        self.collate_fn = collate_fn
        self.buffer_max_batch = buffer_max_batch
    
    def buf_thread(self,
                   process,
                   num_process):
        print('=========start buf thread=========')
        read_count = 0
        reader = mgf.read(self.dataset_path)
        num_data = len(reader)
        while True:
            try:
                while True:
                    if len(self.buffer_single) >= self.instances_buffer_size * 0.2:
                        self._fill_batch()    
                    if read_count >= num_data:
                        break
                    if len(self.buffer_single) >= self.instances_buffer_size:
                        continue
                    read_count += 1
                    if read_count % num_process != process:
                        continue

                    self.buffer_single.append(self.collate_fn(reader[read_count - 1]))
                read_count = 0
            except ValueError as e:
                reader.close()
                print('DataLoader buffer thread Error !!')
                print('Error Msg', e)
        reader.close()

    def _fill_batch(self):

        if len(self.buffer_batch) < self.buffer_max_batch and len(self.buffer_single) > self.instances_buffer_size * 0.5:

            if not self.buffer_lock.acquire(False):
                time.sleep(0.1)
                return
            if len(self.buffer_batch) < self.buffer_max_batch:
                num_data = len(self.buffer_single)
                batch_idx = np.random.choice(num_data, self.batch_size, replace=num_data < self.batch_size)
                batch_idx.sort()
                instances = [self.buffer_single.pop(i) for i in batch_idx[::-1]]
                self.buffer_batch.append(instances)
            self.buffer_lock.release()

class SpectrumDataLoader(object):
    def __init__(self, 
                 hparams: Hparams, 
                 ins_collate_fn: Callable, 
                 batch_collate_fn: Callable,
                 dataset: SpectrumDataset = None,
                 dataset_path: str = None, 
                 is_train: bool = True):

        self.batch_collate_fn = batch_collate_fn
        self.ins_collate_fn = ins_collate_fn
        self.dataset_path = dataset_path
        self.is_train = is_train
        self.batch_size = hparams.batch_size
        self.instances_buffer_size = hparams.instances_buffer_size
        self.buffer_max_batch = hparams.buffer_max_batch

        if is_train:
            manager = Manager()
            self.buffer_single = manager.list()
            self.buffer_batch = manager.list()
            self.buffer_lock = Lock()
            self.num_process = hparams.train_num_workers
            self.buffer_thread = None
            self.num_batches = hparams.train_steps
            self.iscomplete = False
        else:
            self.dataset = dataset
            self.num_batches = math.ceil(len(dataset) / hparams.batch_size)

    def get_idx(self, sampler):
        yield from (
            idx for idx in sampler
        )

    def __len__(self):
        return self.num_batches

    def close_testset(self):
        self.dataset.close_file()

    def kill_process(self):
        if self.buffer_thread is not None:
            for thread in self.buffer_thread:
                print('Terminating process {0}'.format(thread.pid))
                thread.terminate()
            self.iscomplete = True

    def _fill_buf(self):
        if self.buffer_thread is None:
            self.buffer_thread = []
            buf_ctrl = SpectrumBufCtrl(
                batch_size=self.batch_size,
                dataset_path=self.dataset_path,
                buffer_single=self.buffer_single,
                buffer_batch=self.buffer_batch,
                buffer_lock=self.buffer_lock,
                instances_buffer_size=self.instances_buffer_size,
                buffer_max_batch=self.buffer_max_batch,
                collate_fn=self.ins_collate_fn
            )
            for process in range(self.num_process):
                buffer_thread = Process(target=buf_ctrl.buf_thread, args=(process, self.num_process))
                buffer_thread.start()
                self.buffer_thread.append(buffer_thread)

    def __iter__(self):
        try:
            if self.is_train:
                if self.buffer_thread is None:
                    self._fill_buf()
                while not self.iscomplete:
                    if self.buffer_batch:
                        yield self.batch_collate_fn(self.buffer_batch.pop(0))
            else:
                sampler = SequentialSampler(self.dataset)
                reader = batch(
                    self.get_idx(sampler), 
                    batch_size=self.batch_size
                )
                for batch_indices in reader():
                    samples = [self.dataset[idx] for idx in batch_indices]
                    yield self.batch_collate_fn(samples)
        except KeyboardInterrupt:
            self.kill_process()
class Spectrum4ChannelsBufCtrl(object):
    def __init__(self,
                 batch_size,
                 dataset_path,
                 buffer_single,
                 buffer_batch,
                 buffer_lock,
                 instances_buffer_size,
                 buffer_max_batch,
                 collate_fn):
        self.dataset_path = dataset_path
        self.batch_size = batch_size
        self.buffer_single = buffer_single
        self.buffer_batch = buffer_batch
        self.buffer_lock = buffer_lock
        self.instances_buffer_size = instances_buffer_size
        self.collate_fn = collate_fn
        self.buffer_max_batch = buffer_max_batch

    def buf_thread(self,
                   process,
                   num_process):
        print('=========start buf thread=========')
        read_count = 0
        reader = mgf.read(self.dataset_path)
        seek = -1
        num_data = len(reader)
        while True:
            try:
                while True:
                    if len(self.buffer_single) >= self.instances_buffer_size * 0.2:
                        self._fill_batch()    
                    if read_count >= num_data:
                        break
                    if len(self.buffer_single) >= self.instances_buffer_size:
                        continue
                    read_count += 1
                    seek = (seek + 1) % num_data
                    first_data = reader[seek]
                    left, right = min(seek + 1, num_data), min(seek + 4, num_data)
                    spec_list = []
                    spec_list.append(first_data)
                    for i in range(left, right):
                        if first_data['params']['canonicalsmiles'] == reader[i]['params']['canonicalsmiles']:
                            seek += 1
                            spec_list.append(reader[i])
                        else:
                            break
                    if read_count % num_process != process:
                        continue
                    self.buffer_single.append(self.collate_fn(spec_list))
                read_count = 0
            except ValueError as e:
                reader.close()
                print('DataLoader buffer thread Error !!')
                print('Error Msg', e)
        reader.close()

    def _fill_batch(self):

        if len(self.buffer_batch) < self.buffer_max_batch and len(self.buffer_single) > self.instances_buffer_size * 0.5:

            if not self.buffer_lock.acquire(False):
                time.sleep(0.1)
                return
            if len(self.buffer_batch) < self.buffer_max_batch:
                num_data = len(self.buffer_single)
                batch_idx = np.random.choice(num_data, self.batch_size, replace=num_data < self.batch_size)
                batch_idx.sort()
                instances = [self.buffer_single.pop(i) for i in batch_idx[::-1]]
                self.buffer_batch.append(instances)
            self.buffer_lock.release()

class Spectrum4ChannelsDataLoader(object):
    def __init__(self, 
                 config: Ms2VecConfig, 
                 ins_collate_fn: Callable, 
                 batch_collate_fn: Callable,
                 dataset: SpectrumDataset = None,
                 dataset_path: str = None, 
                 is_train: bool = True):
        self.config = config
        self.batch_collate_fn = batch_collate_fn
        self.ins_collate_fn = ins_collate_fn
        self.dataset_path = dataset_path
        self.is_train = is_train
        self.batch_size = config.hparams.batch_size
        self.instances_buffer_size = config.hparams.instances_buffer_size
        self.buffer_max_batch = config.hparams.buffer_max_batch
        if is_train:
            manager = Manager()
            self.buffer_single = manager.list()
            self.buffer_batch = manager.list()
            self.buffer_lock = Lock()
            self.num_process = config.hparams.train_num_workers
            self.buffer_thread = None
            self.num_batches = config.hparams.train_steps
            self.iscomplete = False
        else:
            self.dataset = dataset
            self.num_batches = math.ceil(len(dataset) / config.hparams.batch_size)

    def get_idx(self, sampler):
        yield from (
            idx for idx in sampler
        )

    def __len__(self):
        return self.num_batches

    def close_testset(self):
        self.dataset.close_file()

    def kill_process(self):
        if self.buffer_thread is not None:
            for thread in self.buffer_thread:
                print('Terminating process {0}'.format(thread.pid))
                thread.terminate()
            self.iscomplete = True

    def _fill_buf(self):
        if self.buffer_thread is None:
            self.buffer_thread = []
            buf_ctrl = Spectrum4ChannelsBufCtrl(
                batch_size=self.batch_size,
                dataset_path=self.dataset_path,
                buffer_single=self.buffer_single,
                buffer_batch=self.buffer_batch,
                buffer_lock=self.buffer_lock,
                instances_buffer_size=self.instances_buffer_size,
                buffer_max_batch=self.buffer_max_batch,
                collate_fn=self.ins_collate_fn
            )
            for process in range(self.num_process):
                buffer_thread = Process(target=buf_ctrl.buf_thread, args=(process, self.num_process))
                buffer_thread.start()
                self.buffer_thread.append(buffer_thread)

    def __iter__(self):
        if self.is_train:
            if self.buffer_thread is None:
                self._fill_buf()
            while not self.iscomplete:
                if self.buffer_batch:
                    yield self.batch_collate_fn(
                        self.buffer_batch.pop(0), 
                        self.config.spectrum.max_mz, 
                        self.config.spectrum.min_mz,
                        self.config.spectrum.resolution,
                        self.config.spectrum.neg,
                        self.config.path.smiles_embedding_path
                    )
        else:
            sampler = SequentialSampler(self.dataset)
            reader = batch(
                self.get_idx(sampler), 
                batch_size=self.batch_size
            )
            for batch_indices in reader():
                samples = [self.dataset[idx] for idx in batch_indices]
                yield self.batch_collate_fn(
                    samples, 
                    self.config.spectrum.max_mz, 
                    self.config.spectrum.min_mz,
                    self.config.spectrum.resolution,
                    self.config.spectrum.neg,
                    self.config.path.smiles_embedding_path
                )
