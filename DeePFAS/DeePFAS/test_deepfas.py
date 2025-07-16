import argparse
import math
import multiprocessing
import pickle
from functools import partial
from pathlib import Path
from colorama import init as colorama_init
import faulthandler
import signal

import h5py
import pytorch_lightning as pl
import torch
import wandb
from pyteomics import mgf
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger

from ae.config.models import MolExtractorConfig
from ae.models.gru_autoencoder import TranslationModel
from ae.models.mol_extracter import MolExtracter
from ae.utils.data_modules import mol_modules
from ae.utils.smiles_process import (batch_search_buffer_collate_fn, batch_to_device_search_buffer_collate_fn)
from DeePFAS.config.models import DeePFASConfig
from DeePFAS.models.model.deepfas import InfernceDeePFAS
from DeePFAS.utils.data_modules.spec_modules import (MultiprocessModule,
                                                     SpecProcess)
from multiprocessing import cpu_count
from DeePFAS.utils.spectra_process.functions import (
    test_batch_spec_collate_fn, test_batch_spec_collate_to_device)

import numpy as np

torch.set_printoptions(profile='full')
torch.set_float32_matmul_precision('high')

def arg_parser():
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument('--ae_config_pth', type=str, required=True)
    parser.add_argument('--deepfas_config_pth', type=str, required=True)
    parser.add_argument('--test_data_pth', type=str, required=True)
    parser.add_argument('--retrieval_data_pth', type=str, required=True)
    parser.add_argument('--results_dir', type=str, required=True)
    parser.add_argument('--topk', type=int, required=True)
    parser.add_argument('--mode', type=str, choices=['eval', 'inference'] ,required=True)
    return parser.parse_args()

def test_deepfas(args: str = None):

    faulthandler.register(signal.SIGUSR1)

    multiprocessing.set_start_method('fork', force=True)
    colorama_init()

    if args is None:
        args = arg_parser()
    device_name = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device_name)

    nums_cpus = 1
    assert nums_cpus > 0

    # nums_cpus = 1
    ae_args = MolExtractorConfig.from_json_path(args.ae_config_pth)

    deepfas_args = DeePFASConfig.from_json_path(
        args.deepfas_config_pth,
    )

    seed_everything(deepfas_args.hparams.seed)

    # run_dir = Path(deepfas_args.logger.project_name) / deepfas_args.logger.job_key
    # run_dir.mkdir(parents=True, exist_ok=True)

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    mode = args.mode
    spec_processor = SpecProcess(
        tokenized=deepfas_args.spectrum.tokenized,
        resolution=deepfas_args.spectrum.resolution,
        loss_mz_from=deepfas_args.spectrum.loss_mz_from,
        loss_mz_to=deepfas_args.spectrum.loss_mz_to,
        use_neutral_loss=deepfas_args.spectrum.use_neutral_loss,
        mode=mode
    )

    # with mgf.read(args.test_data_pth) as f:
    #     test_dataset = list(f[:])

    data_modules = MultiprocessModule(
        hparams=deepfas_args.hparams,
        train_dataset=None,
        val_dataset=None,
        test_dataset=args.test_data_pth,
        spec_processor=spec_processor,
        batch_collate_fn=partial(test_batch_spec_collate_fn, mode=mode),
        to_device_collate=partial(test_batch_spec_collate_to_device, mode=mode, device=device),
    )

    mol_processor = mol_modules.SearchTopkMolProcess()

    retrieval_modules = mol_modules.SingleProcessModule(
        hparams=ae_args.hparams,
        dataset_pth=args.retrieval_data_pth,
        cpu_count=nums_cpus,
        mol_processor=mol_processor,
        batch_collate_fn=batch_search_buffer_collate_fn,
        to_device_collate=partial(batch_to_device_search_buffer_collate_fn, device=device),
    )

    retrieval_dataloader = retrieval_modules.Getdataloader()
    strategy = "auto"

    model = InfernceDeePFAS(
        ae_config=ae_args,
        deepfas_config=deepfas_args,
        deepfas_model_pth=deepfas_args.path.save_model_path,
        retrieval_results_pth=results_dir,
        retrieval_dataloader=retrieval_dataloader,
        mode=mode,
        topk=args.topk,
        device=device,
    )

    trainer = pl.Trainer(
        strategy=strategy, max_epochs=deepfas_args.hparams.max_epoch,
        accelerator=device_name, devices=deepfas_args.hparams.num_devices, log_every_n_steps=deepfas_args.hparams.log_every_n_steps,
        precision=deepfas_args.hparams.precision, gradient_clip_val=deepfas_args.hparams.clip,
        num_sanity_val_steps=deepfas_args.hparams.num_sanity_val_steps,
        use_distributed_sampler=deepfas_args.hparams.num_devices > 1, val_check_interval=deepfas_args.hparams.val_check_interval,
        accumulate_grad_batches=deepfas_args.hparams.accumulate_grad_batches,
    )

    test_dataloaders = [l for l in [data_modules.test_dataloader()] if l is not None]
    trainer.test(
        model,
        dataloaders=test_dataloaders,
    )
    retrieval_modules.close_dataset()

if __name__ == '__main__':
    test_deepfas()
