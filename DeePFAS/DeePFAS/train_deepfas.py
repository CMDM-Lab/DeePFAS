from ae.models.gru_autoencoder import TranslationModel
from ae.models.mol_extracter import MolExtracter
from DeePFAS.models.model.deepfas import DeePFAS
from DeePFAS.utils.data_modules.spec_modules import SpecProcess, MultiprocessModule
import pytorch_lightning as pl
from ae.config.models import MolExtractorConfig
from DeePFAS.config.models import DeePFASConfig
from DeePFAS.utils.spectra_process.functions import batch_spec_collate_fn, batch_spec_collate_to_device
from ae.utils.smiles_process.functions import idx_to_smiles
from functools import partial
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger
from DeePFAS.models.metrics.eval import epoch_time
from DeePFAS.utils.dataloader import MultiprocessDataLoader
from pathlib import Path
import multiprocessing
import torch
import wandb
import argparse
import h5py
import pickle
import math
from pyteomics import mgf
from tqdm import tqdm
import time
import os
import torch.nn.functional as F
import pandas as pd
torch.set_printoptions(profile='full')
torch.set_float32_matmul_precision('high')

def arg_parser():
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument('--ae_config_pth', type=str, required=True)
    parser.add_argument('--deepfas_config_pth', type=str, required=True)
    return parser.parse_args()

def initialize_weights(m):
    if hasattr(m, 'weight') and m.weight.dim() > 1:
        torch.nn.init.kaiming_uniform_(m.weight)

def train(model: DeePFAS, config: DeePFASConfig, dataloader: MultiprocessDataLoader, optimizer, record_process_pth, record_process):
    model.train()
    epoch_loss = 0
    optimizer.zero_grad()
    train_iter = iter(dataloader)
    for step in tqdm(range(dataloader.num_batches)):
        batch = next(train_iter)
        randomized_smiles, canonical_smiles, m_z, intensity = batch
        _, _, z, _ = model.ae.forward_encoder(randomized_smiles)
        z_p = model(m_z, intensity)
        loss = F.mse_loss(z_p, z)
        if step % model.train_step_check_result == 0:
            pred = model.ae.sample(
                len(z_p),
                max_len=model.ae_config.hparams.max_len,
                z=z_p,
                device=model.device,
            )
            pred_2 = model.ae.sample(
                len(z),
                max_len=model.ae_config.hparams.max_len,
                z=z,
                device=model.device,
            )
            batch_acc = sum([r == idx_to_smiles(p) for r, p in zip(pred, canonical_smiles)])
            print(f'Real canonical smiles: {idx_to_smiles(canonical_smiles[0])}\n')
            print(f'Pred smiles: {pred[0]}\n')
            print(f'Batch Acc: {batch_acc / model.ae_config.hparams.batch_size}\n')
            print(f'AE smiles: {pred_2[0]}\n')
            print(f'Batch Loss: {float(loss)}\n')
            with open(record_process_pth, 'wb') as f:
                record_process['step'] = step + 1
                pickle.dump(record_process, f)
        epoch_loss += float(loss)
        loss.backward()
        if (step + 1) % config.hparams.accumulate_grad_batches == 0:
            optimizer.step()
            optimizer.zero_grad()

    if dataloader.num_batches % config.hparams.accumulate_grad_batches > 0:
        optimizer.step()
        optimizer.zero_grad()

    return epoch_loss / dataloader.num_batches

def eval(model: DeePFAS, config: DeePFASConfig, dataloader: MultiprocessDataLoader):
    model.eval()
    val_iter = iter(dataloader)
    val_results = {
        "REAL_CAN": [],
        "GENERATED": [],
        "RANDOMIZED_CONVERTED": [],
    }
    with torch.no_grad():
        epoch_loss = 0
        for step in tqdm(range(dataloader.num_batches)):
            batch = next(val_iter)
            randomized_smiles, canonical_smiles, m_z, intensity = batch
            z_p = model(m_z, intensity)
            _, _, z, _ = model.ae.forward_encoder(randomized_smiles)
            loss = F.mse_loss(z_p, z)
            current_samples = model.ae.sample(
                len(z_p),
                max_len=model.ae_config.hparams.max_len,
                z=z_p,
                device=model.device,
            )
            epoch_loss += float(loss)
            if step % model.val_step_check_result == 0:
                print('step :', round((step / dataloader.num_batches) * 100, 2), '%')
                print(f'Batch Loss: {float(loss)}\n')
            val_results['RANDOMIZED_CONVERTED'].extend(randomized_smiles)
            val_results['GENERATED'].extend(current_samples)
            val_results['REAL_CAN'].extend(canonical_smiles)
        val_results['RANDOMIZED_CONVERTED'] = [idx_to_smiles(x) for x in val_results['RANDOMIZED_CONVERTED']]
        val_results['REAL_CAN'] = [idx_to_smiles(x) for x in val_results['REAL_CAN']]
        samples = pd.DataFrame(val_results)
        samples['MATCH'] = samples['REAL_CAN'] == samples['GENERATED']
        total = len(samples)
        match = samples['MATCH'].sum()
        pct_match = samples['MATCH'].mean()
        print(samples.head())
        print(f'Total: {total} Matched: {match} Percent Matched {pct_match}\n')
        return pct_match, epoch_loss / dataloader.num_batches

def trainer(config: DeePFASConfig, model: DeePFAS, data_modules: MultiprocessModule):
    train_loader, val_loader = data_modules.train_dataloader(), data_modules.val_dataloader()
    if os.path.isfile(config.path.save_model_path):
        model.load_state_dict(torch.load(config.path.save_model_path))
        print('successfully load previous best model parameters')
    else:
        model.apply(initialize_weights)
    optis = model.configure_optimizers()
    optim, scheduler = optis['optimizer'], optis['lr_scheduler']['scheduler']
    best_pct = {'val_acc': float('-inf'), 'val_loss': 1e6}
    os.makedirs(config.path.model_dir, exist_ok=True)

    best_pct_pth = os.path.join(config.path.model_dir, 'ms2vec_best_pct.pkl')
    if os.path.isfile(best_pct_pth):
        with open(best_pct_pth, 'rb') as f:
            best_pct = pickle.load(f)

    record_process = {'epoch': 1, 'step': 0}
    record_process_pth = os.path.join(config.path.model_dir, 'record_process.pkl')
    for epoch in tqdm(range(config.hparams.max_epoch)):
        start_time = time.time()
        with open(record_process_pth, 'wb') as f:
            record_process['epoch'] = epoch + 1
            pickle.dump(record_process, f)
        train_loss = train(
            model=model,
            config=config,
            dataloader=train_loader,
            optimizer=optim,
            record_process_pth=record_process_pth,
            record_process=record_process
        )
        pct_match, val_loss = eval(
            model=model,
            config=config,
            dataloader=val_loader
        )
        end_time = time.time()
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)
        scheduler.step(val_loss)
        if val_loss < best_pct['val_loss']:
            best_pct['val_acc'] = pct_match
            best_pct['val_loss'] = val_loss
            print('<---------- update model ---------->')
            torch.save(model.model.state_dict(), config.path.save_model_path)
            with open(best_pct_pth, 'wb') as f:
                pickle.dump(best_pct, f)
        print(f'Epoch: {epoch + 1} | Time: {epoch_mins}m {epoch_secs}s\n')
        print(f'Epoch Train Loss: {train_loss:.6f}\n')
        print(f'Epoch Val Loss: {val_loss:.6f}\n')

def train_deepfas(args: str = None):

    multiprocessing.set_start_method('fork', force=True)
    if args is None:
        args = arg_parser()
    device_name = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device_name)

    ae_args = MolExtractorConfig.from_json_path(args.ae_config_pth)

    deepfas_args = DeePFASConfig.from_json_path(
        args.deepfas_config_pth,
    )

    seed_everything(deepfas_args.hparams.seed)

    spec_processor = SpecProcess(
        tokenized=deepfas_args.spectrum.tokenized,
        resolution=deepfas_args.spectrum.resolution,
        loss_mz_from=deepfas_args.spectrum.loss_mz_from,
        loss_mz_to=deepfas_args.spectrum.loss_mz_to,
        use_neutral_loss=deepfas_args.spectrum.use_neutral_loss,
    )

    with mgf.read(deepfas_args.path.train_data_path) as f:
        train_dataset = list(f[:])


    with mgf.read(deepfas_args.path.val_data_path) as f:
        val_dataset = list(f[:])

    train_step_check_result = math.ceil(len(train_dataset) * deepfas_args.hparams.train_check_result_ratio)
    val_step_check_result = math.ceil(len(val_dataset) * deepfas_args.hparams.val_check_result_ratio)
    print(f'Step check results (Train): {train_step_check_result}\n')
    print(f'Step check results (Val): {val_step_check_result}\n')
    data_modules = MultiprocessModule(
        hparams=deepfas_args.hparams,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        test_dataset=None,
        spec_processor=spec_processor,
        batch_collate_fn=batch_spec_collate_fn,
        to_device_collate=partial(batch_spec_collate_to_device, device=device),
    )

    src_pad_idx = 0

    model = DeePFAS(
        deepfas_args=deepfas_args,
        ae_args=ae_args,
        device=device,
        src_pad_idx=src_pad_idx,
        train_step_check_result=train_step_check_result,
        val_step_check_result=val_step_check_result,
    ).to(device)

    trainer(deepfas_args, model, data_modules)


    # pytorch-lightning

    # run_dir = Path(deepfas_args.logger.project_name) / deepfas_args.logger.job_key
    # run_dir.mkdir(parents=True, exist_ok=True)

    # wandb.init(
    #     project=deepfas_args.logger.project_name,
    #     dir=run_dir,
    #     name=deepfas_args.logger.run_name,
    #     entity=deepfas_args.logger.wandb_entity_name,
    # )
    # wandb_logger = WandbLogger()

    # # strategy = pl.strategies.DDPStrategy(find_unused_parameters=False)
    # strategy = 'auto'
    # callbacks = [
    #     LearningRateMonitor(logging_interval='step'),
    #     pl.callbacks.ModelCheckpoint(
    #         monitor='Val loss', save_top_k=deepfas_args.hparams.save_top_k, mode='min',
    #         dirpath=deepfas_args.path.model_dir, save_last=deepfas_args.hparams.save_last,
    #         every_n_train_steps=deepfas_args.hparams.save_every_n_train_steps,
    #         every_n_epochs=deepfas_args.hparams.save_every_n_train_epochs,
    #         verbose=deepfas_args.hparams.verbose,
    #     ),
    # ]
    # trainer = pl.Trainer(
    #     strategy=strategy, max_epochs=deepfas_args.hparams.max_epoch, logger=wandb_logger,
    #     accelerator=device_name, devices=deepfas_args.hparams.num_devices, log_every_n_steps=deepfas_args.hparams.log_every_n_steps,
    #     precision=deepfas_args.hparams.precision, callbacks=callbacks, gradient_clip_val=deepfas_args.hparams.clip,
    #     num_sanity_val_steps=deepfas_args.hparams.num_sanity_val_steps,
    #     use_distributed_sampler=deepfas_args.hparams.num_devices > 1, val_check_interval=deepfas_args.hparams.val_check_interval,
    #     accumulate_grad_batches=deepfas_args.hparams.accumulate_grad_batches
    # )


    # train_dataloaders = data_modules.train_dataloader()
    # val_dataloaders = [l for l in [data_modules.val_dataloader()] if l is not None]

    # trainer.fit(
    #     model,
    #     train_dataloaders=train_dataloaders,
    #     val_dataloaders=val_dataloaders,
    # )
    # train_dataloaders.kill_process()

if __name__ == '__main__':
    # path = './config/ae_config.json'
    train_deepfas()
