from ae.models.mol_extracter import MolExtracter
from ae.utils.data_modules.mol_modules import MolProcess, MultiprocessModule
import pytorch_lightning as pl
from ae.config.models import MolExtractorConfig
from ae.utils.smiles_process import batch_buffer_collate_fn, batch_to_device_collate_fn
from functools import partial
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger
from pathlib import Path
import torch
import wandb
import argparse
import h5py
import pickle
import multiprocessing
import numpy as np
torch.set_printoptions(profile='full')
torch.set_float32_matmul_precision('high')

def arg_parser():
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument('--config_pth', type=str, required=True)
    return parser.parse_args()

def train_ae(args_pth: str = None):

    multiprocessing.set_start_method('fork', force=True)
    if args_pth is None:
        args = arg_parser()
    device_name = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device_name)
    args = MolExtractorConfig.from_json_path(args.config_pth)
    
    seed_everything(args.hparams.seed)

    run_dir = Path(args.logger.project_name) / args.logger.job_key
    run_dir.mkdir(parents=True, exist_ok=True)

    use_contrastive_learning = args.hparams.use_contrastive_learning
    if use_contrastive_learning:
        with open(args.path.decoys_dataset_pth, 'rb') as f:
            contrastive_learning_decoys = pickle.load(f)

    mol_processor = MolProcess(
        max_len=args.hparams.max_len,
        use_properties=args.hparams.use_properties,
        contrastive_learning_decoys=contrastive_learning_decoys if use_contrastive_learning else None,
        num_decoys=args.hparams.num_decoys,
        randomized=args.hparams.randomized,
    )

    with h5py.File(args.path.dataset_path, 'r') as f:
        if args.hparams.debug:
            dataset = f['CANONICALSMILES'][:1000]
        else:
            dataset = f['CANONICALSMILES'][:]
        dataset = np.array(dataset)

    data_modules = MultiprocessModule(
        hparams=args.hparams,
        dataset=dataset,
        mol_processor=mol_processor,
        batch_collate_fn=partial(
            batch_buffer_collate_fn,
            use_properties=args.hparams.use_properties,
        ),
        to_device_collate=partial(batch_to_device_collate_fn, device=device),
    )

    wandb.init(
        project=args.logger.project_name,
        dir=run_dir,
        name=args.logger.run_name,
        entity=args.logger.wandb_entity_name,
    )
    wandb_logger = WandbLogger()

    # strategy = pl.strategies.DDPStrategy(find_unused_parameters=False)
    strategy = "auto"

    model = MolExtracter(args)

    callbacks = [
        LearningRateMonitor(logging_interval='step'),
        pl.callbacks.ModelCheckpoint(
            monitor='Val loss', save_top_k=args.hparams.save_top_k, mode='min',
            dirpath=run_dir, save_last=args.hparams.save_last,
            every_n_train_steps=args.hparams.save_every_n_train_steps,
            every_n_epochs=args.hparams.save_every_n_train_epochs,
            verbose=args.hparams.verbose,
        )
    ]
    trainer = pl.Trainer(
        strategy=strategy, max_epochs=args.hparams.max_epoch, logger=wandb_logger,
        accelerator=device_name, devices=args.hparams.num_devices, log_every_n_steps=args.hparams.log_every_n_steps,
        precision=args.hparams.precision, callbacks=callbacks, gradient_clip_val=args.hparams.clip,
        num_sanity_val_steps=args.hparams.num_sanity_val_steps,
        use_distributed_sampler=args.hparams.num_devices > 1, val_check_interval=args.hparams.val_check_interval,
        accumulate_grad_batches=args.hparams.accumulate_grad_batches,
    )

    # wandb_logger.watch(model, log_graph=False)
    # wandb_logger.experiment.config.update({
    #     'num_params': sum(p.numel() for p in model.parameters() if p.requires_grad)
    # })

    train_dataloaders = data_modules.train_dataloader()
    val_dataloaders = [l for l in [data_modules.val_dataloader()] if l is not None]
    # trainer.validate(
    #     model,
    #     dataloaders=[l for l in [data_modules.val_dataloader()] if l is not None]
    # )
    trainer.fit(
        model,
        train_dataloaders=train_dataloaders,
        val_dataloaders=val_dataloaders,
    )
    train_dataloaders.kill_process()

if __name__ == '__main__':
    # path = './config/ae_config.json'
    train_ae()
