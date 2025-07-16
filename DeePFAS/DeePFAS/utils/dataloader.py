import math
from multiprocessing import Lock, Manager, Process
from typing import Callable

import numpy as np
from DeePFAS.config.models import Hparams
import random
from DeePFAS.models.metrics.eval import epoch_time
import time
from collections import deque
from pyteomics import mgf
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
        dataset_pth,
        spec_processor,
        process_status_map,
        shuffle
    ):

        self.read_count_global = read_count_global
        self.batch_size = batch_size
        self.buffer_single = buffer_single
        self.buffer_batch = buffer_batch
        self.buffer_lock = buffer_lock
        self.instances_buffer_size = instances_buffer_size
        self.batch_collate_fn = batch_collate_fn
        self.buffer_max_batch = buffer_max_batch
        self.dataset_pth = dataset_pth
        self.process_status_map = process_status_map
        self.shuffle = shuffle
        self.spec_processor = spec_processor

    def buf_thread(self, process, num_process):
        print('=========start buf thread=========')
        read_count = 0
        self.process_status_map[process] = True
        with mgf.read(self.dataset_pth) as f:
            num_data = len(f)
        idx_map = np.arange(num_data)
        if self.shuffle:
            np.random.shuffle(idx_map)
        while True:

            #num_data = len(self.dataset)
            f = mgf.read(self.dataset_pth)
            try:
                while True:

                    if read_count >= num_data:
                        break

                    read_count += 1
                    next_idx = int(idx_map[read_count - 1])
                    if (next_idx + 1) % num_process != process:
                        continue
                    if len(self.buffer_single) >= self.instances_buffer_size:
                        read_count -= 1
                        continue

                    #read_count += 1

                    #if read_count % num_process != process:  # skip
                    #    continue

                    #self.read_count_global.value = read_count
                    #start_time = time.time()
                    #self._fill_batch()
                    #self.buffer_single.append(self.dataset[read_count - 1])
                    #end_time = time.time()
                    #mins, secs = epoch_time(start_time, end_time)
                    #print(f'Inference Time: {mins}m {secs}s')

                    self.buffer_single.append(self.spec_processor(f[next_idx]))

                read_count = 0
            except ValueError as e:
                print('DataLoader buffer thread Error !!')
                print('Error Msg', e)
            f.close()
            break
        self.process_status_map[process] = False
        # print(f'process {process} is completed !!')

class MultiprocessDataLoader(object):
    """Multiprocessing (load data)."""

    def __init__(self,
                 hparams: Hparams,
                 batch_collate_fn: Callable,
                 to_device_collate_fn: Callable,
                 spec_processor,
                 dataset_pth,
                 is_train,
                 shuffle):

        self.batch_collate_fn = batch_collate_fn
        self.to_device_collate_fn = to_device_collate_fn
        self.spec_processor = spec_processor
        self.dataset_pth = dataset_pth
        self.batch_size = hparams.batch_size
        self.instances_buffer_size = hparams.instances_buffer_size
        self.buffer_max_batch = hparams.buffer_max_batch
        self.num_process = hparams.train_num_workers if is_train else hparams.val_num_workers
        manager = Manager()
        self.buffer_single = manager.list()
        self.buffer_batch = manager.list()
        #self.buffer_batch = deque()
        self.process_status_map = manager.dict()
        self.buffer_lock = Lock()
        self.buffer_thread = None
        self.read_count_global = manager.Value('i', 0)
        with mgf.read(self.dataset_pth) as f:
            self.sample_size = len(f)
        self.num_batches = math.ceil(self.sample_size / self.batch_size)
        self.iscomplete = False
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
                dataset_pth=self.dataset_pth,
                spec_processor=self.spec_processor,
                process_status_map=self.process_status_map,
                shuffle=self.shuffle
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
                if len(self.buffer_single) < self.batch_size:
                    all_end = self.all_end()
                    if not all_end:
                        continue
                    if len(self.buffer_single) == 0 and all_end:
                        break

                # self.buffer_lock.acquire(True)
                num_data = len(self.buffer_single)
                batch_idx = np.random.choice(num_data, min(self.batch_size, num_data), replace=False)
                batch_idx.sort()
                instances = [self.buffer_single.pop(i) for i in batch_idx[::-1]]
                # self.buffer_lock.release()
                #start_time = time.time()
                data = self.batch_collate_fn(instances)
                #end_time = time.time()
                #mins, secs = epoch_time(start_time, end_time)
                #print(f'Inference Time: {mins}m {secs}s')
                #start_time = time.time()
                self.buffer_batch.append(data)
                #end_time = time.time()
                #mins, secs = epoch_time(start_time, end_time)
                #print(f'Inference Time: {mins}m {secs}s')
        self.kill_process()

    def check_process_alive(self):
        if self.buffer_thread is None:
            return
        for thread in self.buffer_thread:
            if thread.is_alive():
                return
        self.kill_process()

    # def get_batch(self):
    #     while True:
    #         all_end = self.all_end()
    #         if len(self.buffer_single) == 0 and all_end:
    #             break
    #         if not all_end and len(self.buffer_single) < self.batch_size:
    #             continue

    #         # self.buffer_lock.acquire(True)
    #         num_data = len(self.buffer_single)
    #         batch_idx = np.random.choice(num_data, min(self.batch_size, num_data), replace=False)
    #         batch_idx.sort()
    #         instances = [self.buffer_single.pop(i) for i in batch_idx[::-1]]
    #         yield self.batch_collate_fn(instances)
    #     self.kill_process()
    #     yield None

    def __iter__(self):
        try:
            self.check_process_alive()
            if self.buffer_thread is None:
                self._fill_buf()
                self.iscomplete = False
                # self.batch_iter = self.get_batch()
            while not self.iscomplete:
                if len(self.buffer_batch) > 0:

                    yield self.to_device_collate_fn(self.buffer_batch.pop(0))
                # data = next(self.batch_iter)
                # if data is not None:
                #     yield self.to_device_collate_fn(data)
        except Exception as e:
            print(e)
            self.kill_process()
