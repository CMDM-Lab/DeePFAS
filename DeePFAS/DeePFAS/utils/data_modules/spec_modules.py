

import torch
from rdkit import Chem
from torch.utils.data import Dataset

from DeePFAS.utils.smiles_process.functions import (VOC_MAP,
                                                    tokens_add_sos_eos,
                                                    tokens_from_smiles)
from DeePFAS.utils.spectra_process.functions import (cal_losses,
                                                     normalize_peaks,
                                                     tokenzie_mz)

from DeePFAS.utils.dataloader import MultiprocessDataLoader
import numpy as np
from colorama import Fore
from colorama import Style

from DeePFAS.config.models import Hparams
import pytorch_lightning as pl

class SpecProcess:
    def __init__(self,
                 tokenized=True,
                 resolution=2,
                 isomericSmiles=False,
                 canonical=True,
                 use_neutral_loss=True,
                 loss_mz_from=1.0,
                 loss_mz_to=1000.0,
                 mode='eval'):

        self.isomericSmiles = isomericSmiles
        self.canonical = canonical
        self.resolution = resolution
        self.tokenized = tokenized
        self.loss_mz_from = loss_mz_from
        self.loss_mz_to = loss_mz_to
        self.use_neutral_loss = use_neutral_loss
        self.mode = mode

    def __call__(self, data):

        m_z, intensity = data['m/z array'], data['intensity array']
        #print(f"\n{Fore.RED}Title {data['params']['title']}{Style.RESET_ALL}")
        if self.use_neutral_loss:
            losses_mz, losses_intensity = cal_losses(
                data['params']['pepmass'][0],
                m_z,
                intensity,
                self.loss_mz_from,
                self.loss_mz_to,
            )

            m_z, intensity = list(m_z) + list(losses_mz), list(intensity) + list(losses_intensity)
            # m_z, intensity = torch.tensor(m_z), torch.tensor(intensity)
        if not isinstance(m_z, list):
            m_z, intensity = list(m_z), list(intensity)

        peaks = [[m, it] for m, it in zip(m_z, intensity)]
        peaks.sort(key=lambda x: x[0], reverse=False)
        peaks = torch.tensor(peaks)

        if self.tokenized:
            peaks = tokenzie_mz(peaks, resolution=self.resolution)
        else:
            peaks = normalize_peaks(peaks, resolution=self.resolution)

        m_z, intensity = peaks[:, 0].long().tolist(), peaks[:, 1].tolist()

        if self.mode == 'eval':
            smiles = Chem.MolToSmiles(Chem.MolFromSmiles(data['params']['canonicalsmiles']),
                                    isomericSmiles=self.isomericSmiles,
                                    canonical=self.canonical,)

            canonicalsmiles_ids = tokens_add_sos_eos([VOC_MAP[token] for token in tokens_from_smiles([smiles])])
            randomizedsmiles_ids = [VOC_MAP[token] for token in tokens_from_smiles([data['params']['randomizedsmiles']])]

        return {
            'canonicalsmiles': canonicalsmiles_ids if self.mode == 'eval' else None,
            'randomizedsmiles': randomizedsmiles_ids if self.mode == 'eval' else None,
            'pepmass': data['params']['pepmass'][0],
            'm_z': m_z,
            'intensity': intensity,
            'precursor_type': data['params']['precursor_type'],
            'title': data['params']['title'],
        }

class SpecData(Dataset):
    def __init__(self, dataset, data_processor: SpecProcess):
        super().__init__()
        self.dataset = dataset
        self.data_processor = data_processor
        self.index_map = np.array((range(len(dataset))))

    def shuffle(self):
        np.random.shuffle(self.index_map)

    def __getitem__(self, idx):
        return self.data_processor(self.dataset[self.index_map[idx]])

    def __len__(self):
        return len(self.dataset)

class MultiprocessModule(pl.LightningDataModule):
    def __init__(self, hparams: Hparams, train_dataset, val_dataset, test_dataset, spec_processor, batch_collate_fn, to_device_collate):
        super().__init__()
        self.hyper_hparams = hparams
        self.batch_collate_fn = batch_collate_fn
        self.to_device_collate = to_device_collate
        self.spec_processor = spec_processor
        self.train_dataset_pth = train_dataset
        self.val_dataset_pth = val_dataset
        self.test_dataset_pth = test_dataset
        #if test_dataset is None:
            #self.train_size = len(train_dataset)
            #self.val_size = len(val_dataset)
            #np.random.shuffle(train_dataset)
            #self.train_subset, self.val_subset = (
            #    SpecData(train_dataset, self.spec_processor),
            #    SpecData(val_dataset, self.spec_processor),
            #)
        #else:
        #    self.test_subset = SpecData(test_dataset, self.spec_processor)

    def train_dataloader(self):
        return MultiprocessDataLoader(
            hparams=self.hyper_hparams,
            batch_collate_fn=self.batch_collate_fn,
            to_device_collate_fn=self.to_device_collate,
            spec_processor=self.spec_processor,
            dataset_pth=self.train_dataset_pth,
            is_train=True,
            shuffle=True,
        )
        #return DataLoader(
        #    dataset=self.train_subset,
        #    batch_size=self.hyper_hparams.batch_size,
        #    num_workers=self.hyper_hparams.train_num_workers,
        #    collate_fn=self.batch_collate_fn,
        #    shuffle=True,
        #    drop_last=False,
        #)

    def val_dataloader(self):
        return MultiprocessDataLoader(
            hparams=self.hyper_hparams,
            batch_collate_fn=self.batch_collate_fn,
            to_device_collate_fn=self.to_device_collate,
            spec_processor=self.spec_processor,
            dataset_pth=self.val_dataset_pth,
            is_train=False,
            shuffle=False,
        )
        #return DataLoader(
        #    dataset=self.val_subset,
        #    batch_size=self.hyper_hparams.batch_size,
        #    num_workers=self.hyper_hparams.val_num_workers,
        #    collate_fn=self.batch_collate_fn,
        #    shuffle=False,
        #    drop_last=False,
        #)


    def test_dataloader(self):
        return MultiprocessDataLoader(
            hparams=self.hyper_hparams,
            batch_collate_fn=self.batch_collate_fn,
            to_device_collate_fn=self.to_device_collate,
            spec_processor=self.spec_processor,
            dataset_pth=self.test_dataset_pth,
            is_train=False,
            shuffle=False,
        )
