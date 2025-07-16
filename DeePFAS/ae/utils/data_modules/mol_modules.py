
import os
import sys
from pathlib import Path
from typing import Union
import pickle
import h5py
from rdkit import Chem
import numpy as np
import time
from rdkit.Chem import AllChem, Descriptors, RDConfig, rdMolDescriptors
from torch.utils.data import Dataset, DataLoader
from DeePFAS.models.metrics.eval import epoch_time
from multiprocessing import Lock, Manager, Process
import pytorch_lightning as pl
import math
from ae.utils.smiles_process import (ELEMENTS, VOC_MAP, get_elements_chempy,
                                     get_elements_chempy_map,
                                     get_elements_chempy_umap,
                                     remove_salt_stereo, tokens_add_sos_eos,
                                     tokens_from_smiles, randomize_smiles)
from ae.utils.dataloader import MultiprocessDataLoader
from ae.config.models import Hparams
import torch
sys.path.append(os.path.join(RDConfig.RDContribDir, 'SA_Score'))
import sascorer
import random

class MolProcess:
    def __init__(
            self, max_len: int,
            isomeric: bool = False,
            canonical: bool = True,
            use_properties: bool = True,
            contrastive_learning_decoys = None,
            num_decoys: int = 0,
            randomized: bool = True,
        ):
        self.max_len = max_len
        self.isomeric = isomeric
        self.canonical = canonical
        self.use_properties = use_properties
        self.num_decoys = num_decoys
        self.randomized = randomized

    def __call__(self, smiles):
        smiles = smiles.decode().replace('\n', '').strip()
        if self.randomized:
            rand_smiles = randomize_smiles(smiles, canonical=False, isomericSmiles=False)
        randomized_smiles_ids = [VOC_MAP[token] for token in tokens_from_smiles([rand_smiles if self.randomized else smiles])]
        smiles = remove_salt_stereo(smiles, isomeric=self.isomeric, canonical=self.canonical)
        canonical_smiles_ids = tokens_add_sos_eos([VOC_MAP[token] for token in tokens_from_smiles([smiles])])
        if self.use_properties:
            mol = Chem.MolFromSmiles(smiles)
            elements_chempy = get_elements_chempy(ELEMENTS.keys())
            elements_chempy_map = get_elements_chempy_map(elements_chempy)
            fo = rdMolDescriptors.CalcMolFormula(mol)
            fo_map = get_elements_chempy_umap(fo)
            fo_slots = [0] * len(ELEMENTS)
            for chempy_idx, count in fo_map.items():
                if chempy_idx:  # 0 -> charge ['+', '-']
                    fo_slots[elements_chempy_map[chempy_idx]] = count
            form_list = fo_slots
            form_list.extend([
                Descriptors.MolLogP(mol),
                Descriptors.MolMR(mol),
                Descriptors.NumValenceElectrons(mol),
                Descriptors.NumHDonors(mol),
                Descriptors.NumHAcceptors(mol),
                Descriptors.BalabanJ(mol),
                Descriptors.TPSA(mol),
                Descriptors.qed(mol),
                sascorer.calculateScore(mol),
            ])

        return {
            'rand_smiles_ids': randomized_smiles_ids,
            'caonical_smiles_ids': canonical_smiles_ids,
            'formula': form_list if self.use_properties else None,
        }

def load_data(f_pth, chunk_size, dataset_len, chunk_buffer, start, end):
    with h5py.File(f_pth, 'r') as f:
        for idx in range(start, end):
            if idx * chunk_size >= dataset_len:
                break
            chunk_buffer.append((
                f['CANONICALSMILES'][idx * chunk_size: min(chunk_size * (idx + 1), dataset_len)],
                f['chemical_emb'][idx * chunk_size: min(chunk_size * (idx + 1), dataset_len)],
            ))

class SearchTopkMolProcess:
    def __init__(self):
        pass

    def __call__(self, data):
        smiles, emb = data
        smiles = smiles.decode()

        return {
            'smiles': smiles,
            'emb': emb
        }

class SearchTopkMolData(Dataset):
    def __init__(self, hdf5_pth, num_cpus, data_processor: SearchTopkMolProcess):
        super().__init__()
    
        with h5py.File(hdf5_pth, 'r') as f:
            chunks_smiles = f['CANONICALSMILES'].chunks[0]
            chunks_chemical_emb = f['chemical_emb'].chunks[0]
            if chunks_smiles != chunks_chemical_emb:
                print(f'WARNING: chunks of smiles and chemical embedding are different!!\n')
        self.f = h5py.File(hdf5_pth, 'r')
        self.dataset_len = len(self.f['CANONICALSMILES'])
        self.dataset_pth = hdf5_pth
        self.data_processor = data_processor
        self.chunk_size = self.f['CANONICALSMILES'].chunks[0]
        if self.dataset_len <= 800000:
            print(f'WARNING: dataset size is small, minimum chunk size is 800000')
            self.chunk_size = self.dataset_len
        self.chunks = math.ceil(self.dataset_len / self.chunk_size)
        self.manager = Manager()
        self.chunks_queue = self.manager.list()
        self.buffer_thread = []
        self.num_cpus = num_cpus
        self.num_chunks_per_process = self.dataset_len // self.chunk_size

    def fill_chunks(self):
        for i in range(self.num_cpus):
            process = Process(
                target=load_data,
                args=(
                    self.dataset_pth,
                    self.chunk_size,
                    self.dataset_len,
                    self.chunks_queue,
                    i * self.num_chunks_per_process,
                    (i + 1) * self.num_chunks_per_process,
                )
            )
            process.start()
            print('=========start chunks thread=========')
            self.buffer_thread.append(process)

    def load_new_data(self, idx):
        if hasattr(self, 'smiles'):
            del self.smiles
            del self.embs

        # start_time = time.time()
        self.smiles = self.f['CANONICALSMILES'][idx * self.chunk_size: min(self.chunk_size * (idx + 1), self.dataset_len)]
        self.embs = self.f['chemical_emb'][idx * self.chunk_size: min(self.chunk_size * (idx + 1), self.dataset_len)]
        
        # end_time = time.time()
        # mins, secs = epoch_time(start_time, end_time)
        # print(f'Load hdf5 Time: {mins}m {secs}s')

    def __len__(self):
        return self.dataset_len

    def __getitem__(self, idx):

        # single process
        new_idx = idx % self.chunk_size
        if new_idx % self.chunk_size == 0:
            self.load_new_data(idx // self.chunk_size)
        data = self.smiles[new_idx], self.embs[new_idx]
        
        # multiprocess

        # if idx == 0:
        #     self.kill_process()
        #     self.fill_chunks()

        # if idx % self.chunk_size == 0:
        #     if hasattr(self, "smiles"):
        #         del self.smiles
        #     if hasattr(self, "embs"):
        #         del self.embs
        #     while len(self.chunks_queue) < 1:
        #         pass
        #     start_time = time.time()
        #     data = self.chunks_queue.pop(0)
        #     self.smiles, self.embs = data
        #     end_time = time.time()
        #     mins, secs = epoch_time(start_time, end_time)
        #     print(f'Load hdf5 Time: {mins}m {secs}s')
        # new_idx = idx % self.chunk_size
        # data = self.smiles[new_idx], self.embs[new_idx]

        return self.data_processor(data)

    def kill_process(self):
        if self.buffer_thread is not None:
            for thread in self.buffer_thread:
                print('Terminating process {0}'.format(thread.pid))
                thread.terminate()
            del self.buffer_thread[:]

    def close_file(self):
        self.f.close()

class MolData(Dataset):
    def __init__(self, dataset, data_processor: MolProcess):
        super().__init__()
        # self.hdf5_pth = Path(hdf5_pth)
        # self.f = h5py.File(hdf5_pth, mode='r')
        # self.sample_size = len(self.f['CANONICALSMILES'])
        self.dataset = dataset
        self.data_processor = data_processor

    def __getitem__(self, idx):
        return self.data_processor(self.dataset[idx])

    def __len__(self):
        return len(self.dataset)

class RandomSplitModule(pl.LightningDataModule):
    def __init__(self, dataset, batch_size, val_frac, train_num_workers, val_num_workers, collate_fn):
        super().__init__()
        self.val_frac = val_frac
        self.train_num_workers = train_num_workers
        self.val_num_workers = val_num_workers
        self.batch_size = batch_size
        self.collate_fn = collate_fn
        val_size = round(val_frac * len(dataset))
        train_size = len(dataset) - val_size
        self.train_subset, self.val_subset = torch.utils.data.random_split(dataset, [train_size, val_size])

    def train_dataloader(self):
        return DataLoader(
            dataset=self.train_subset,
            batch_size=self.batch_size,
            num_workers=self.train_num_workers,
            collate_fn=self.collate_fn,
            shuffle=True,
            drop_last=False,
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.val_subset,
            batch_size=self.batch_size,
            num_workers=self.val_num_workers,
            collate_fn=self.collate_fn,
            shuffle=False,
            drop_last=False,
        )

class SingleProcessModule(pl.LightningDataModule):
    def __init__(self, hparams: Hparams, dataset_pth, cpu_count, mol_processor: SearchTopkMolProcess, batch_collate_fn, to_device_collate):
        super().__init__()
        self.hyper_hparams = hparams
        self.dataset = SearchTopkMolData(dataset_pth, cpu_count, mol_processor)
        self.collate_fn = lambda x: to_device_collate(batch_collate_fn(x))

    def close_dataset(self):
        self.dataset.close_file()

    def Getdataloader(self):
        return DataLoader(
            self.dataset,
            self.hyper_hparams.batch_size,
            shuffle=False,
            collate_fn=self.collate_fn
        )

class MultiprocessModule(pl.LightningDataModule):
    def __init__(self, hparams: Hparams, dataset, mol_processor, batch_collate_fn, to_device_collate, retrieval=False):
        super().__init__()
        self.hyper_hparams = hparams
        self.batch_collate_fn = batch_collate_fn
        self.to_device_collate = to_device_collate
        self.mol_processor = mol_processor
        if retrieval:
            self.retrieval_dataset = MolData(dataset, self.mol_processor)
        else:
            self.val_size = round(self.hyper_hparams.val_frac * len(dataset))
            self.train_size = len(dataset) - self.val_size
            np.random.shuffle(dataset)
            self.train_subset, self.val_subset = (
                MolData(dataset[:self.train_size], self.mol_processor),
                MolData(dataset[self.train_size:], self.mol_processor),
            )

    def train_dataloader(self):
        return MultiprocessDataLoader(
            hparams=self.hyper_hparams,
            batch_collate_fn=self.batch_collate_fn,
            to_device_collate_fn=self.to_device_collate,
            dataset=self.train_subset,
            is_train=True,
            shuffle=True
        )

    def val_dataloader(self):
        return MultiprocessDataLoader(
            hparams=self.hyper_hparams,
            batch_collate_fn=self.batch_collate_fn,
            to_device_collate_fn=self.to_device_collate,
            dataset=self.val_subset,
            is_train=False,
            shuffle=False
        )

    def test_dataloader(self):
        return MultiprocessDataLoader(
            hparams=self.hyper_hparams,
            batch_collate_fn=self.batch_collate_fn,
            to_device_collate_fn=self.to_device_collate,
            dataset=self.retrieval_dataset,
            is_train=False,
            shuffle=False
        )
