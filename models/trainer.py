import math
import os
import pickle
import queue
import sys
import time
from typing import Callable, Iterable, List, Union

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from torch.optim import Adam
from tqdm import tqdm

from .config.models import GruAutoEncoderConfig, GruOptimConfig, Ms2VecConfig
from .model.gru_autoencoder import TranslationModel
from .model.ms2embedding import AttentionConv, AttentionConv2D, Conv1D, Net1D
from .model.properties_model import PropertiesModel
from .utils.dataloader import (MultiprocessDataLoader,
                               Spectrum4ChannelsDataLoader, SpectrumDataLoader)
from .utils.dataset import (Spectrum4ChannelsDataset, SpectrumDataset,
                            ValDataset)
from .utils.epoch_timer import epoch_time
from .utils.misc import KLAnnealer
from .utils.smiles_process import (VOC_MAP, batch_add_eos, batch_add_pad,
                                   batch_add_sos, batch_buffer_collate_fn,
                                   idx_to_smiles, single_buffer_collate_fn,
                                   smiles_batch_indices, to_device_collate_fn)
from .utils.spectra_process.functions import (batch_spec_collate_fn,
                                              batch_spec_conv_collate_fn,
                                              spec_collate_fn,
                                              spec_conv_collate_fn)
from .utils.visualize import loss_acc_curve


def count_parameters(model):

    return sum(p.numel() for p in model.parameters() if p.requires_grad) if model else 0

def initialize_weights(m):
    if hasattr(m, 'weight') and m.weight.dim() > 1:
        torch.nn.init.kaiming_uniform_(m.weight)

def get_device():
    if torch.cuda.is_available():
        dev = 'cuda:0'
    # elif torch.backends.mps.is_available():
    #     dev = 'mps'
    else:
        dev = 'cpu'
    return torch.device(dev)

def _n_epoch(config: GruOptimConfig):
    return sum(
        config.lr_n_period * (config.lr_n_mult ** i)
        for i in range(config.lr_n_restarts)
    )

def smiles_acc(pred, trg):
    cnt = 0
    pred = torch.argmax(pred, dim=2)
    for i in range(len(pred)):
        cnt += 1 if not torch.sum(pred[i, :len(trg[i]) - 1] ^ trg[i][1:len(trg[i])]).item() else 0
    return cnt

def spec_smiles_acc(pred, trg, mp, not_correct=None):
    cnt = 0
    pred = torch.argmax(pred, dim=2)
    for i in range(len(pred)):
        trg_smiles = idx_to_smiles(trg[i][1:len(trg[i])])
        if mp.get(trg_smiles):
            continue
        mp[trg_smiles] = 1
        if torch.sum(pred[i, :len(trg[i]) - 1] ^ trg[i][1:len(trg[i])]).item():
            if not_correct:
                not_correct.append(trg_smiles)
        else:
            cnt += 1
    return cnt

def load_dataset(
    config: GruAutoEncoderConfig,
    ins_collate_fn: Callable,
    batch_collate_fn: Callable,
    to_device_collate_fn: Callable
):

    val_dataset = ValDataset(data_file=config.path.data_path,
                             use_record_idx=config.hparams.use_record_idx,
                             train_set_ratio=config.hparams.train_ratio,
                             dataset_idx_path=config.path.data_id_path,
                             config=config)

    return (
        MultiprocessDataLoader(
            hparams=config.hparams,
            dataset_path=config.path.data_path,
            dataset_idx_path=config.path.data_id_path,
            ins_collate_fn=ins_collate_fn,
            batch_collate_fn=batch_collate_fn,
            to_device_collate_fn=to_device_collate_fn,
            is_train=True
        ),
        MultiprocessDataLoader(
            hparams=config.hparams,
            dataset_path=config.path.data_path,
            dataset_idx_path=config.path.data_id_path,
            ins_collate_fn=ins_collate_fn,
            batch_collate_fn=batch_collate_fn,
            to_device_collate_fn=to_device_collate_fn,
            is_train=False,
            dataset=val_dataset
        )   
    )

def get_trg(
    batch: List[str],
    length: int,
    des: str
):
    if des == 'in':
        trg = batch_add_pad(batch_add_sos(batch), length, VOC_MAP['<PAD>'])
    elif des == 'out':
        trg = batch_add_pad(batch_add_eos(batch), length, VOC_MAP['<PAD>'])
    else:
        return None
    return torch.tensor(trg)

def get_src(
    batch: List[str],
    length: int
):
    return torch.tensor(batch_add_pad(smiles_batch_indices(batch), length, VOC_MAP['<PAD>']))

def load_spectrum_dataset(
    config: Ms2VecConfig,
    ins_collate_fn: Callable,
    batch_collate_fn: Callable,
):
    val_dataset = SpectrumDataset(data_file=config.path.test_data_path,
                                  transform=ins_collate_fn)
    return (
        SpectrumDataLoader(hparams=config.hparams,
                           dataset_path=config.path.train_data_path,
                           ins_collate_fn=ins_collate_fn,
                           batch_collate_fn=batch_collate_fn,
                           is_train=True),
        SpectrumDataLoader(hparams=config.hparams,
                           dataset_path=config.path.train_data_path,
                           ins_collate_fn=ins_collate_fn,
                           batch_collate_fn=batch_collate_fn,
                           is_train=False,
                           dataset=val_dataset)
    )

def load_specrum_4_dataset(
    config: Ms2VecConfig,
    ins_collate_fn: Callable,
    batch_collate_fn: Callable,
):
    val_dataset = Spectrum4ChannelsDataset(data_file=config.path.test_data_path,
                                           transform=ins_collate_fn)
    return (
        Spectrum4ChannelsDataLoader(config=config,
                                    dataset_path=config.path.train_data_path,
                                    ins_collate_fn=ins_collate_fn,
                                    batch_collate_fn=batch_collate_fn,
                                    is_train=True),
        Spectrum4ChannelsDataLoader(config=config,
                                    dataset_path=config.path.train_data_path,
                                    ins_collate_fn=ins_collate_fn,
                                    batch_collate_fn=batch_collate_fn,
                                    is_train=False,
                                    dataset=val_dataset)
    )

def evaluate_ms2vec(auto_encoder: TranslationModel,
                    val_loader: SpectrumDataLoader,
                    model: AttentionConv,
                    device: torch.device,
                    config: Ms2VecConfig):
    model.eval()
    auto_encoder.eval()
    val_iter = iter(val_loader)
    with torch.no_grad():
        samples, real_smiles, rand_smiles_converted = [], [], []
        losses = 0
        for i in tqdm(range(config.hparams.val_steps)):
            batch = next(val_iter)
            if isinstance(model, Net1D):
                spectrum, randomized_smiles, canonical_smiles = batch
                spectrum = spectrum.to(device)
                z_p = model(spectrum)
            elif isinstance(model, Conv1D):
                spectrum, randomized_smiles, canonical_smiles = batch
                spectrum = spectrum.to(device)
                z_p = model(spectrum)
            else:
                randomized_smiles, canonical_smiles, m_z, intensity = batch
                z_p = model(m_z, intensity)
            randomized_smiles, canonical_smiles = [item.to(device) for item in randomized_smiles], [item.to(device) for item in canonical_smiles]
            _, _, z, _ = auto_encoder.forward_encoder(randomized_smiles)
            loss = F.mse_loss(z_p, z)
            current_samples = auto_encoder.sample(len(z_p), max_len=config.hparams.max_len, z=z_p, device=z_p.device)
            samples.extend(current_samples)
            real_smiles.extend([idx_to_smiles(ids) for ids in canonical_smiles])
            # rand_smiles_converted.extend([idx_to_smiles(ids) for ids in randomized_smiles])
            losses += float(loss)
            if i % max(config.hparams.val_step_check_result, 1) == 0:
                print('step :', round((i / config.hparams.val_steps) * 100, 2), '%')
                print('loss :', float(loss))
        losses = losses / config.hparams.val_steps
        print(f'val epoch loss: {losses}')
        # samples = pd.DataFrame({'REAL_CAN': real_smiles, 'GENERATED': samples, 'RANDOMIZED_CONVERTED': rand_smiles_converted})
        samples = pd.DataFrame({'REAL_CAN': real_smiles, 'GENERATED': samples})
        samples['MATCH'] = samples['REAL_CAN'] == samples['GENERATED']
        total = len(samples)
        match = samples['MATCH'].sum()
        pct_match = samples['MATCH'].mean()
        print(samples.head())
        print(f'Total: {total} Matched: {match} Percent Matched {pct_match}')
    return pct_match, losses

def train_ms2vec(auto_encoder: TranslationModel,
                 model: Union[AttentionConv, AttentionConv2D, Net1D],
                 dataloader: Iterable,
                 optimizer: torch.optim.Adam,
                 device: torch.device,
                 config: Ms2VecConfig,
                 record_process: List[float]):

    model.train()
    epoch_loss, epoch_acc = 0, 0
    odd_or_even = config.hparams.train_steps & 1
    optimizer.zero_grad()

    for i in tqdm(range(config.hparams.train_steps)):
        batch = next(dataloader)
        if isinstance(model, Net1D):
            spectrum, randomized_smiles, canonical_smiles = batch
            spectrum = spectrum.to(device)
            z_p = model(spectrum)
        elif isinstance(model, Conv1D):
            spectrum, randomized_smiles, canonical_smiles = batch
            spectrum = spectrum.to(device)
            z_p = model(spectrum)
        else:
            randomized_smiles, canonical_smiles, m_z, intensity = batch
            z_p = model(m_z, intensity)
        randomized_smiles, canonical_smiles = [item.to(device) for item in randomized_smiles], [item.to(device) for item in canonical_smiles]
        _, _, z, _ = auto_encoder.forward_encoder(randomized_smiles)
        loss = F.mse_loss(z_p, z)
        loss.backward()
        # padded_randomized_smiles = nn.utils.rnn.pad_sequence(
        #     sequences=randomized_smiles,
        #     batch_first=True,
        #     padding_value=VOC_MAP[PAD_CHAR]
        # )
        # padded_canonical_smiles = nn.utils.rnn.pad_sequence(
        #     sequences=canonical_smiles,
        #     batch_first=True,
        #     padding_value=VOC_MAP[PAD_CHAR]
        # )
        _, pred = auto_encoder.forward_decoder(canonical_smiles, z_p)
        _, pred_2 = auto_encoder.forward_decoder(canonical_smiles, z)
        batch_acc = smiles_acc(pred[:, :-1], canonical_smiles)
        epoch_acc += batch_acc
        if i % config.hparams.train_step_check_result == 0:
            print('step :', round((i / config.hparams.train_steps) * 100, 2), '%')
            # print(f'Real random smiles: {idx_to_smiles(padded_randomized_smiles[0, 1:])}\n')
            print(f'Real canonical smiles: {idx_to_smiles(canonical_smiles[0][1:])}\n')
            print(f'Pred smiles: {idx_to_smiles(pred.argmax(2)[0, :-1])}\n')
            print('Batch Acc: ', batch_acc / config.hparams.batch_size)
            print(f'AE smiles: {idx_to_smiles(pred_2.argmax(2)[0, :-1])}\n')
            print('lr :', optimizer.param_groups[0]['lr'])
            print('msms loss :', float(loss))
            with open('./saved/record_process.pkl', 'wb') as f:
                record_process[1] = i + 1
                pickle.dump(record_process, f)
        # torch.nn.utils.clip_grad_norm_(model.parameters(), config.hparams.clip)
        if i & 1:
            optimizer.step()
            optimizer.zero_grad()
        epoch_loss += float(loss)

    if odd_or_even & 1:
        optimizer.step()
        optimizer.zero_grad()

    epoch_acc_ratio = epoch_acc / config.hparams.train_steps
    return epoch_loss / config.hparams.train_steps, epoch_acc_ratio

def ms2vec_trainer(
    config: Ms2VecConfig,
    autoencoder: TranslationModel,
    model: Union[AttentionConv, AttentionConv2D, Net1D]
):

    device = get_device()
    if isinstance(model, Net1D):
        train_loader, val_loader = load_specrum_4_dataset(config, spec4_collate_fn, batch_spec4_collate_fn)
    elif isinstance(model, Conv1D):
        train_loader, val_loader = load_spectrum_dataset(config, spec_conv_collate_fn, batch_spec_conv_collate_fn)
    else:
        train_loader, val_loader = load_spectrum_dataset(config, spec_collate_fn, batch_spec_collate_fn)
    train_iter = iter(train_loader)

    print(f'The model has {count_parameters(model):,} trainable parameters')

    if os.path.isfile(config.path.save_model_path):
        model.load_state_dict(torch.load(config.path.save_model_path))
        print('successfully load previous best model parameters')
    else:
        model.apply(initialize_weights)
    optimizer = Adam(
        params=model.parameters(),
        lr=config.hparams.lr,
        weight_decay=config.hparams.weight_decay,
        eps=config.hparams.adam_eps
    )
    if isinstance(model, AttentionConv2D):
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer=optimizer,
            verbose=True,
            factor=config.hparams.factor,
            patience=config.hparams.patience
        )
    else:
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.98)
    train_losses = []
    train_epoch_acc = []
    os.makedirs(config.path.model_dir, exist_ok=True)
    best_pct = [float('-inf'), 1e6]
    if os.path.isfile(config.path.save_best_pct):
        with open(config.path.save_best_pct, 'rb') as f:
            best_pct = pickle.load(f)
    record_process = [1, 0]
    valid_queue = queue.Queue(5)
    for step in tqdm(range(config.hparams.epoch)):
        start_time = time.time()
        with open('./saved/record_process.pkl', 'wb') as f:
            record_process[0] = step + 1
            pickle.dump(record_process, f)
        train_loss, train_acc = train_ms2vec(
            auto_encoder=autoencoder,
            model=model,
            dataloader=train_iter,
            optimizer=optimizer,
            device=device,
            config=config,
            record_process=record_process
        )
        pct_match, valid_loss = evaluate_ms2vec(
            auto_encoder=autoencoder,
            model=model,
            val_loader=val_loader,
            device=device,
            config=config
        )
        end_time = time.time()

        train_losses.append(train_loss)
        train_epoch_acc.append(train_acc)
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)
        scheduler.step(valid_loss)
        if pct_match > best_pct[0] or valid_loss < best_pct[1]:
            best_pct[0] = pct_match
            best_pct[1] = valid_loss
            print('<---------- update model ---------->')
            torch.save(model.state_dict(), config.path.save_model_path)
            with open(config.path.save_best_pct, 'wb') as f:
                pickle.dump(best_pct, f)
        if step > 4:
            rem = valid_queue.get()
        valid_queue.put(valid_loss)
        if step > 4:
            early_stopping(list(valid_queue.queue))
        os.makedirs(config.path.loss_dir, exist_ok=True)
        with open(config.path.save_loss_path, 'a') as f:
            f.write(str(train_losses))

        print(f'Epoch: {step + 1} | Time: {epoch_mins}m {epoch_secs}s')
        print(f'\tTrain Acc: {train_acc:.3f}')
    train_loader.kill_process()

def early_stopping(losses):
    print(np.std(losses))
    if np.std(losses)<0.0000001:
        sys.exit('Valid loss converging!')
    counter = 0
    if losses[4]>losses[0]:
        counter = counter + 1
    if losses[3]>losses[0]:
        counter = counter + 1
    if losses[2]>losses[0]:
        counter = counter + 1
    if losses[1]>losses[0]:
        counter = counter + 1
    if counter > 4:
        sys.exit('Loss increasing!')
    return

def train_gru(model: TranslationModel,
              properties_model: PropertiesModel,
              train_loader: Iterable,
              val_loader: MultiprocessDataLoader,
              optimizer: torch.optim.Adam,
              config: GruAutoEncoderConfig,
              kl_weight: float,
              scheduler: torch.optim.lr_scheduler.ReduceLROnPlateau,
              best_pct: List[float],
              record_process: List[float]):

    model.train()
    epoch_loss, epoch_acc = 0, 0
    odd_or_even = config.hparams.train_steps & 1

    for i in tqdm(range(config.hparams.train_steps)):
        batch = next(train_loader)
        randomized_smiles, canonical_smiles, properties = batch

        mu, logvar, z, kl_loss, recon_loss, pred = model(
            randomized_smiles, canonical_smiles
        )
        p_loss = properties_model(z, properties) if config.hparams.use_properties else 0

        batch_acc = smiles_acc(pred[:, :-1], canonical_smiles)

        epoch_acc += batch_acc

        loss = (
            kl_weight * kl_loss + 
            recon_loss +
            p_loss * config.hparams.regression_loss_weight
        )
        epoch_loss += float(loss)
        loss.backward()
        if not (i & 1):
            # TODO: estimate H count & formula & charset
            clip_grad_norm_(model.parameters(), config.hparams.clip)
            optimizer.step()
            optimizer.zero_grad()

        if i % config.hparams.train_step_check_result == 0:
            print('step :', round((i / config.hparams.train_steps) * 100, 2), '%')
            print(f'Real random smiles: {idx_to_smiles(randomized_smiles[0])}\n')
            print(f'Real canonical smiles: {idx_to_smiles(canonical_smiles[0][1:])}\n')
            print(f'Pred smiles: {idx_to_smiles(pred.argmax(2)[0, :-1])}\n')
            print('Batch Acc: ', batch_acc / config.hparams.batch_size)
            print('recon loss :', float(recon_loss))
            print('lr :', optimizer.param_groups[0]['lr'])
            print('properties loss :', float(p_loss))
            print('kl weight :', float(kl_weight))
            print('kl loss :', float(kl_loss))
            print('all loss :', float(loss))
            with open('./gru_saved/record_process.pkl', 'wb') as f:
                record_process[1] = i + 1
                pickle.dump(record_process, f)

        if i and i % config.hparams.step_evaluate == 0:
            pct_match, val_loss = evaluate_gru(model=model, 
                                               properties_model=properties_model,
                                               val_loader=val_loader, 
                                               config=config)

            scheduler.step(pct_match)
            if pct_match > best_pct[0]:
                best_pct[0] = pct_match
                print('<---------- update model ---------->')
                os.makedirs(config.path.model_dir, exist_ok=True)
                torch.save(model.state_dict(), config.path.save_model_path)
                if config.hparams.use_properties:
                    torch.save(properties_model.state_dict(), config.path.properties_model_path)
                with open(config.path.save_best_pct, 'wb') as f:
                    pickle.dump(best_pct, f)

    if odd_or_even & 1:
        optimizer.zero_grad()
        clip_grad_norm_(model.parameters(), config.hparams.clip)
        optimizer.step()

    epoch_acc_ratio = epoch_acc / (config.hparams.sample_size)
    return epoch_loss / config.hparams.train_steps, epoch_acc_ratio

def evaluate_gru(
    model: TranslationModel,
    properties_model: PropertiesModel,
    val_loader: MultiprocessDataLoader,
    config: GruAutoEncoderConfig
):
    val_iter = iter(val_loader)
    with torch.no_grad():
        samples, real_smiles, rand_smiles_converted = [], [], []
        losses = 0
        for i in tqdm(range(config.hparams.val_steps)):
            batch = next(val_iter)
            randomized_smiles, canonical_smiles, properties = batch
            mu, logvar, z, kl_loss, recon_loss, _ = model(
                randomized_smiles, canonical_smiles
            )
            p_loss = properties_model(z, properties) if config.hparams.use_properties else 0
            current_samples = model.sample(len(mu), max_len=config.hparams.max_len, z=mu, device=mu.device)
            samples.extend(current_samples)
            real_smiles.extend([idx_to_smiles(ids) for ids in canonical_smiles])
            rand_smiles_converted.extend([idx_to_smiles(ids) for ids in randomized_smiles])
            loss = recon_loss + kl_loss + p_loss
            losses += loss.item()
            if i % config.hparams.val_step_check_result == 0:
                print('step :', round((i / config.hparams.val_steps) * 100, 2), '%')
                print('recon loss :', float(recon_loss))
                print('kl loss :', float(kl_loss))
                print('all loss :', float(loss))
                print('properties loss :', float(p_loss))
        losses = losses / config.hparams.val_steps
        print(f'val epoch loss: {losses}')
        samples = pd.DataFrame({'REAL_CAN': real_smiles, 'GENERATED': samples, 'RANDOMIZED_CONVERTED': rand_smiles_converted})
        samples['MATCH'] = samples['REAL_CAN'] == samples['GENERATED']
        total = len(samples)
        match = samples['MATCH'].sum()
        pct_match = samples['MATCH'].mean()
        print(samples.head())
        print(f'Total: {total} Matched: {match} Percent Matched {pct_match}')
    return pct_match, losses

def get_optim_params(*models):
    params = []
    for model in models:
        if model:
            params.extend(model.parameters())
    return params

def gru_trainer(config: GruAutoEncoderConfig):
    device = get_device()
    train_loader, val_loader = load_dataset(config=config, 
                                            ins_collate_fn=single_buffer_collate_fn, 
                                            batch_collate_fn=batch_buffer_collate_fn,
                                            to_device_collate_fn=to_device_collate_fn)
    train_iter = iter(train_loader)

    model = TranslationModel(config).to(device)
    properties_model = PropertiesModel(config.embedding.z_dim, 20).to(device) if config.hparams.use_properties else None

    auto_params = count_parameters(model)
    properties_params = count_parameters(properties_model)
    total_params = auto_params + properties_params
    print(f'The model has total {total_params:,} trainable parameters')
    print(f'GRU AutoEncoder has {auto_params:,} trainable parameters')
    print(f'Properties Model has {properties_params:,} trainable parameters')

    if os.path.isfile(config.path.save_model_path):
        model.load_state_dict(torch.load(config.path.save_model_path, map_location=device))
        print('successfully load previous best AE parameters')
    else:
        model.apply(initialize_weights)
    if os.path.isfile(config.path.properties_model_path) and properties_model:
        properties_model.load_state_dict(torch.load(config.path.properties_model_path, map_location=device))
        print('successfully load previous best Properties model parameters')
    else:
        model.apply(initialize_weights)

    optimizer = Adam(
        params=get_optim_params(model, properties_model),
        lr=config.optim.lr_start
    )
    n_epoch = int(_n_epoch(config.optim))
    kl_annealer = KLAnnealer(n_epoch, config.optim)
    # lr_annealer = CosineAnnealingLRWithRestart(optimizer, config.optim)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer=optimizer,
        factor=config.hparams.factor,
        patience=config.hparams.patience,
        verbose=True,
        min_lr=config.hparams.min_lr
    )
    model.zero_grad()
    train_epoch_losses = []
    train_epoch_acc = []
    os.makedirs(config.path.model_dir, exist_ok=True)
    best_pct = [float('-inf'), 0]
    if os.path.isfile(config.path.save_best_pct):
        with open(config.path.save_best_pct, 'rb') as f:
            best_pct = pickle.load(f)
    print(f'current best val / train pct: {best_pct[0]}, {best_pct[1]}')
    record_process = [1, 0]
    for step in tqdm(range(n_epoch)):

        kl_weight = kl_annealer(step)
        start_time = time.time()
        with open('./gru_saved/record_process.pkl', 'wb') as f:
            record_process[0] = step + 1
            pickle.dump(record_process, f)
        train_loss, train_acc = train_gru(
            model=model,
            properties_model=properties_model,
            train_loader=train_iter,
            val_loader=val_loader,
            optimizer=optimizer,
            config=config,
            kl_weight=kl_weight,
            scheduler=scheduler,
            best_pct=best_pct,
            record_process=record_process
        )
        pct_match, val_loss = evaluate_gru(
            model=model, 
            properties_model=properties_model,
            val_loader=val_loader, 
            config=config
        )
        end_time = time.time()
        # lr_annealer.step()
        train_epoch_losses.append(train_loss)
        train_epoch_acc.append(train_acc)
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        os.makedirs(config.path.loss_dir, exist_ok=True)
        with open(config.path.save_loss_path, 'a') as f:
            f.write(str(train_loss))
        print(f'Epoch: {step + 1} | Time: {epoch_mins}m {epoch_secs}s')
        print(f'\tTrain Acc: {train_acc:.3f} | Train Loss: {train_loss:.3f}')
        if pct_match > best_pct[0] or pct_match + train_acc > sum(best_pct):
            best_pct = [pct_match, train_acc]
            print('<---------- update model ---------->')
            torch.save(model.state_dict(), config.path.save_model_path)
            if config.hparams.use_properties:
                torch.save(properties_model.state_dict(), config.path.properties_model_path)
            with open(config.path.save_best_pct, 'wb') as f:
                pickle.dump(best_pct, f)

    loss_acc_curve(
        train_losses=train_epoch_losses,
        train_acc=train_epoch_acc,
        name='p1',
        config=config.path
    )

    train_loader.kill_process()
