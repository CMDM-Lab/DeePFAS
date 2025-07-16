import math
import pickle
import time
from multiprocessing import Lock, Manager, Process
from typing import Callable, List

import numpy as np
from pyteomics import mgf
from torch.utils.data import DataLoader, Dataset
from ae.config.models import Hparams

class BufCtrl(object):

    def __init__(
        self,
        read_count_global,
        batch_size,
        buffer_single,
        buffer_batch,
        buffer_lock,
        instances_buffer_size,
        buffer_max_batch,
        batch_collate_fn,
        dataset,
        process_status_map,
    ):

        self.read_count_global = read_count_global
        self.batch_size = batch_size
        self.buffer_single = buffer_single
        self.buffer_batch = buffer_batch
        self.buffer_lock = buffer_lock
        self.instances_buffer_size = instances_buffer_size
        self.batch_collate_fn = batch_collate_fn
        self.buffer_max_batch = buffer_max_batch
        self.dataset = dataset
        self.process_status_map = process_status_map

    def buf_thread(self, process, num_process):
        print('=========start buf thread=========')
        read_count = 0
        self.process_status_map[process] = True
        while True:

            num_data = len(self.dataset)
            try:
                while True:

                    if read_count >= num_data:
                        break

                    if len(self.buffer_single) >= self.instances_buffer_size:
                        continue

                    read_count += 1

                    if read_count % num_process != process:  # skip
                        continue

                    self.read_count_global.value = read_count
                    self.buffer_single.append(self.dataset[read_count - 1])
                read_count = 0
            except ValueError as e:
                print('DataLoader buffer thread Error !!')
                print('Error Msg', e)
            break
        self.process_status_map[process] = False
        print(f'process {process} is completed !!')

class MultiprocessDataLoader(object):
    """Multiprocessing (load data)."""

    def __init__(self,
                 hparams: Hparams,
                 batch_collate_fn: Callable,
                 to_device_collate_fn: Callable,
                 dataset,
                 is_train,
                 shuffle):

        self.batch_collate_fn = batch_collate_fn
        self.to_device_collate_fn = to_device_collate_fn
        self.dataset = dataset
        self.batch_size = hparams.batch_size
        self.instances_buffer_size = hparams.instances_buffer_size
        self.buffer_max_batch = hparams.buffer_max_batch
        self.num_process = hparams.train_num_workers if is_train else hparams.val_num_workers
        manager = Manager()
        self.buffer_single = manager.list()
        self.buffer_batch = manager.list()
        self.process_status_map = manager.dict()
        self.buffer_lock = Lock()
        self.buffer_thread = None
        self.read_count_global = manager.Value('i', 0)
        self.sample_size = len(self.dataset)
        self.num_batches = math.ceil(self.sample_size / self.batch_size)
        self.iscomplete = False
        self.batch_cnt = 0
        self.shuffle = shuffle

    def __len__(self):
        return self.num_batches

    def kill_process(self):
        if self.buffer_thread is not None:
            for thread in self.buffer_thread:
                print('Terminating process {0}'.format(thread.pid))
                thread.terminate()
            del self.buffer_thread[:]
            self.iscomplete = True
            self.buffer_thread = None

    def set_start_idx(self, idx):
        self.read_count_global.value = idx

    def get_current_read_count(self):
        return self.read_count_global.value

    def _fill_buf(self):
        if self.buffer_thread is None:
            self.buffer_thread = []
            buf_ctrl = BufCtrl(
                read_count_global=self.read_count_global,
                batch_size=self.batch_size,
                buffer_single=self.buffer_single,
                buffer_batch=self.buffer_batch,
                buffer_lock=self.buffer_lock,
                instances_buffer_size=self.instances_buffer_size,
                buffer_max_batch=self.buffer_max_batch,
                batch_collate_fn=self.batch_collate_fn,
                dataset=self.dataset,
                process_status_map=self.process_status_map,
            )

            # process single instance data
            for process in range(self.num_process):
                buffer_thread = Process(target=buf_ctrl.buf_thread, args=(process, self.num_process))
                buffer_thread.start()
                self.buffer_thread.append(buffer_thread)

            # packet batch
            batch_collector = Process(target=self._fill_batch)
            batch_collector.start()
            # self.buffer_thread.append(batch_collector)

    def all_end(self):
        for k, v in self.process_status_map.items():
            if v:
                return False
        return True

    def _fill_batch(self):
        while True:
            if len(self.buffer_batch) < self.buffer_max_batch:
                all_end = self.all_end()
                if len(self.buffer_single) == 0 and all_end:
                    break
                if not all_end and len(self.buffer_single) < self.batch_size:
                    continue

                # self.buffer_lock.acquire(True)
                num_data = len(self.buffer_single)
                batch_idx = np.random.choice(num_data, min(self.batch_size, num_data), replace=False)
                batch_idx.sort()
                instances = [self.buffer_single.pop(i) for i in batch_idx[::-1]]
                # self.buffer_lock.release()
                self.buffer_batch.append(self.batch_collate_fn(instances))
        self.kill_process()

    def check_process_alive(self):
        if self.buffer_thread is None:
            return
        for thread in self.buffer_thread:
            if thread.is_alive():
                return
        self.kill_process()
        
    def __iter__(self):
        try:
            self.check_process_alive()
            if self.buffer_thread is None:
                if self.shuffle:
                    np.random.shuffle(self.dataset)
                self._fill_buf()
                self.iscomplete = False

            while not self.iscomplete:
                if self.buffer_batch:
                    yield self.to_device_collate_fn(self.buffer_batch.pop(0))
        except KeyboardInterrupt:
            self.kill_process()
