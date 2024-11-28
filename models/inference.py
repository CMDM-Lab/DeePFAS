import json
import math
import os
import pickle
import sys
import time
from queue import PriorityQueue
from typing import Iterable, List, Union

import numpy as np
import torch
import torch.nn.functional as F
from pyteomics import mgf
from rdkit import Chem
from rdkit.Chem import Descriptors, RDConfig, rdMolDescriptors
from torch.utils.data import DataLoader
from tqdm import tqdm

from .config.models import GruAutoEncoderConfig, Ms2VecConfig
from .eval import (get_formula_diff, get_mcs, get_RDFsim, get_sim,
                   get_weight_diff)
from .model.gru_autoencoder import TranslationModel
from .model.ms2embedding import AttentionConv, AttentionConv2D, Conv1D, Net1D
from .model.properties_model import PropertiesModel
from .trainer import (batch_buffer_collate_fn, load_dataset,
                      single_buffer_collate_fn, to_device_collate_fn)
from .utils.dataloader import MultiprocessDataLoader
from .utils.dataset import InferenceSpectrumDataset
from .utils.epoch_timer import epoch_time
from .utils.smiles_process import (ELEMENTS, batch_add_eos_sos,
                                   get_elements_chempy,
                                   get_elements_chempy_map,
                                   get_elements_chempy_umap, idx_to_smiles,
                                   remove_salt_stereo, smiles_batch_indices)
from .utils.SmilesEnumerator import SmilesEnumerator
from .utils.spectra_process.functions import (cal_mw,
                                              inference_batch_spec_collate_fn,
                                              inference_spec_collate_fn)

sys.path.append(os.path.join(RDConfig.RDContribDir, 'SA_Score'))
import sascorer


def get_absolute_path(relative_path):
    current_file_dir = os.path.dirname(os.path.abspath(__file__))
    absolute_path = os.path.join(current_file_dir, relative_path)
    normalized_path = os.path.normpath(absolute_path)
    return normalized_path

def inference_topk(
    dataloader: Iterable,
    model: TranslationModel,
    pred_z: torch.Tensor,
    mw: float,
    topk: int,
    config: GruAutoEncoderConfig,
):
    pq = PriorityQueue(topk)
    model.eval()
    mol_map = {}
    min_loss = 1e5
    with torch.no_grad():
        start_time = time.time()
        for _ in tqdm(range(config.hparams.train_steps)):
            batch = next(dataloader)
            randomized_smiles, canonical_smiles, properties = batch
            mu, logvar, z, kl_loss = model.forward_encoder(randomized_smiles)

            batch_loss = F.mse_loss(pred_z, z, reduction='none')
            batch_loss = torch.sum(batch_loss, dim=1).unsqueeze(1)
            for i, loss in enumerate(batch_loss):
                min_loss = min(loss.item(), min_loss)
                smiles_str = idx_to_smiles(canonical_smiles[i][1:-1])
                if mol_map.get(smiles_str) == 1:
                    continue
                if pq.qsize() < topk:
                    mol_map[smiles_str] = 1
                    pq.put((-loss.item(), smiles_str))
                else:
                    try:
                        tail = pq.get()
                        mol_map[tail[1]] = 0
                    except:
                        print(tail)
                        raise ValueError('Error')
                    if loss.item() < -tail[0]:
                        pq.put((-loss.item(), smiles_str))
                        mol_map[smiles_str] = 1
                    else:
                        pq.put(tail)
                        mol_map[smiles_str] = 1
        end_time = time.time()
        mins, secs = epoch_time(start_time, end_time)
        print(f'Inference Time: {mins}m {secs}s')
    print(f'min loss{min_loss}')
    candidates = []
    while not pq.empty():
        try:
            loss, smiles = pq.get()
        except:
            print(loss)
            print(smiles)
            raise ValueError('Error')
        mol = Chem.MolFromSmiles(smiles)
        candidate_mw = Descriptors.ExactMolWt(mol)
        candidates.append({'loss': abs(loss), 'smiles': smiles, 'mw': candidate_mw, 'mw_diff': abs(mw - candidate_mw)})

    candidates.sort(key=lambda x: (x['mw_diff'], x['loss']))
    return candidates

def inference_eval(
    compound_loader: MultiprocessDataLoader, 
    spectrum_loader: DataLoader,
    autoencoder: TranslationModel,
    model: Union[Conv1D, AttentionConv2D, Net1D],
    topk: int,
    config: GruAutoEncoderConfig,
    device: torch.device,
    data_file: str,
    out_dir: str,
    mode: str
):
    model.eval()
    autoencoder.eval()
    results = {}
    statistic_map = {}
    compound_iter = iter(compound_loader)
    os.makedirs(out_dir, exist_ok=True)
    with mgf.read(data_file) as mgf_f:
        if mode == 'eval':
            real_smiles = [remove_salt_stereo(mgf_f[i]['params']['canonicalsmiles']) for i in tqdm(range(len(mgf_f)))]
        title_list = [int(mgf_f[i]['params']['title']) for i in tqdm(range(len(mgf_f)))]
        mw_list = [
            cal_mw(
                float(mgf_f[i]['params']['pepmass'][0]), 
                mgf_f[i]['params']['precursor_type']
            ) for i in tqdm(range(len(mgf_f)))
        ]
    with torch.no_grad():
        for i, batch in tqdm(enumerate(spectrum_loader)):
            if isinstance(model, AttentionConv2D):
                m_z, intensity = batch
                z = model(m_z, intensity)
            elif isinstance(model, Conv1D):
                spectrum = batch.to(device)
                z = model(spectrum)      
            elif isinstance(model, Net1D):
                spectrum = batch.to(device)
                z = model(spectrum)
            else:
                raise ValueError('Unknown model!!')
            candidates = inference_topk(
                dataloader=compound_iter,
                model=autoencoder,
                config=config,
                mw=mw_list[i],
                pred_z=z,
                topk=topk
            )
            if mode == 'eval':
                out_file = os.path.join(out_dir, f'{i+1}_pred.txt')
                hit = real_smiles[i] in [candidate['smiles'] for candidate in candidates]
                if hit:
                    print('target compound in candidates')
                else:
                    print('target compound not in candidates')
                print(f'Target Compound: {real_smiles[i]}\n')
                print(candidates)
                with open(out_file, 'a') as f:
                    if real_smiles:
                        rank = []
                        f.write(f'Target Compound: {real_smiles[i]}\n')
                        if hit:
                            f.write(f'Target compound in candidates\n')
                            for j, candidate in enumerate(candidates):
                                if real_smiles[i] == candidate['smiles']:
                                    rank.append(j+1)
                        else:
                            f.write(f'Target compound not in candidates\n')
                    for j, candidate in enumerate(candidates):
                        f.write(f"candidate {j+1}\nloss:{candidate['loss']} {candidate['smiles']}\n")
                if statistic_map.get(real_smiles[i]) is None:
                    statistic_map[real_smiles[i]] = {
                        'hit top20': int(hit),
                        'title': [title_list[i]],
                        'rank': rank if hit else ['()'],
                        'not hit top20': int(not hit),
                        'candidates': [candidates],
                        'similarity': [get_RDFsim(candidates, real_smiles[i])],
                        'mcs': [get_mcs(candidates, real_smiles[i])],
                        # 'dmf': [get_formula_diff(candidates, real_smiles[i])],
                        # 'dmw': [get_weight_diff(candidates, real_smiles[i])]
                    }
                else:
                    statistic_map[real_smiles[i]]['candidates'].append(candidates)
                    statistic_map[real_smiles[i]]['hit top20'] += int(hit)
                    statistic_map[real_smiles[i]]['not hit top20'] += int(not hit)
                    statistic_map[real_smiles[i]]['rank'].append(f"({', '.join([str(r) for r in rank])})")
                    statistic_map[real_smiles[i]]['similarity'].append(get_RDFsim(candidates, real_smiles[i]))
                    statistic_map[real_smiles[i]]['mcs'].append(get_mcs(candidates, real_smiles[i]))
                    # statistic_map[real_smiles[i]]['dmf'].append(get_formula_diff(candidates, real_smiles[i]))
                    # statistic_map[real_smiles[i]]['dmw'].append(get_weight_diff(candidates, real_smiles[i]))
                    statistic_map[real_smiles[i]]['title'].append(title_list[i])
            else:
                results[int(title_list[i])] = candidates
    if mode == 'eval':
        # with open(os.path.join(out_dir, 'statistic.txt'), 'a') as f:
        #     for k, v in statistic_map.items():
        #         mcs_sim_list = []
        #         for title, sim, mcs, dmf, dmw in zip(v['title'], v['similarity'], v['mcs'], v['dmf'], v['dmw']):
        #             data = (
        #                 f"Maximum common substructure (ratio, tanimoto, coef): \n"
        #                 f"title(id): {title}\n"
        #                 f"cloest SMILES: {mcs['closest_smiles']}\n"
        #                 f"farest SMILES: {mcs['farest_smiles']}\n"
        #                 f"max ratio: {mcs['max_mcs_ratio']} min_ratio: {mcs['min_mcs_ratio']} avg_ratio: {mcs['avg_mcs_ratio']}\n"
        #                 f"max_tan: {mcs['max_mcs_tan']} min_tan: {mcs['min_mcs_tan']} avg_tan: {mcs['avg_mcs_tan']}\n"
        #                 f"max_coef: {mcs['max_mcs_coef']} min_tan: {mcs['min_mcs_coef']} avg_tan: {mcs['avg_mcs_coef']}\n"
        #                 f"Fingerprint Similarity: \n"
        #                 f"cloest SMILES: {sim['closest_smiles']}\n"
        #                 f"farest SMILES: {sim['farest_smiles']}\n"
        #                 f"max_sim: {sim['max_sim']} min_sim: {sim['min_sim']} avg_sim: {sim['avg_sim']}\n"
        #                 f"DMF: \n"
        #                 f"cloest SMILES: {dmf['closest_smiles']}\n"
        #                 f"cloest Molecular Formula: {dmf['closest_formula']}\n"
        #                 f"min_dmf: {dmf['min_dmf']} avg_dmf: {dmf['avg_dmf']}\n"            
        #                 f"DMW: \n"
        #                 f"cloest SMILES: {dmw['closest_smiles']}\n"
        #                 f"cloest Molecular Weight: {dmw['closest_mw']}\n"
        #                 f"min_dmw: {dmw['min_dmw']} avg_dmf: {dmw['avg_dmw']}\n\n"       
        #             )
        #             mcs_sim_list.append(data)
        #         data = (
        #             f"{k}\nhit top20: {v['hit top20']} / {v['hit top20'] + v['not hit top20']}\n"
        #             f"rank: {', '.join([str(r) for r in v['rank']])}\n"
        #             f"title(id): {', '.join([str(r) for r in v['title']])}\n"
        #             f"{''.join([o for o in mcs_sim_list])}"
        #         )
        #         f.write(data)
        with open(os.path.join(out_dir, 'statistic.pkl'), 'wb') as f:
            pickle.dump(statistic_map, f)
    with open(os.path.join(out_dir, 'statistic.json'), 'w') as f:
        json.dump(statistic_map if mode == 'eval' else results, f, indent=4)
    compound_loader.kill_process()
    return statistic_map if mode == 'eval' else results

def library_matching(
    dataloader: Iterable, 
    intput_smi: str,
    topk: int,
    config: GruAutoEncoderConfig,
    out_file: str = None,
    real_smiles: str = None
):
    pq = PriorityQueue(topk)
    mol_map = {}
    start_time = time.time()
    for _ in tqdm(range(config.hparams.train_steps)):
        batch = next(dataloader)
        randomized_smiles, canonical_smiles, properties = batch
        for i, smi in enumerate(canonical_smiles):
            smiles = idx_to_smiles(smi[1:-1])
            sim = get_sim(intput_smi, smiles)
            if mol_map.get(smiles) == 1:
                continue
            if pq.qsize() < topk:
                mol_map[smiles] = 1
                pq.put((sim, smiles))
            else:
                try:
                    tail = pq.get()
                    mol_map[tail[1]] = 0
                except:
                    print(tail)
                    raise ValueError('Error')
                if sim > tail[0]:
                    pq.put((sim, smiles))
                    mol_map[smiles] = 1
                else:
                    pq.put(tail)
                    mol_map[smiles] = 1
    end_time = time.time()
    mins, secs = epoch_time(start_time, end_time)
    print(f'Inference Time: {mins}m {secs}s')
    candidates = []
    while not pq.empty():
        try:
            sim, smiles = pq.get()
        except:
            print(sim)
            print(smiles)
            raise ValueError('Error')
        candidates.append({'sim': sim, 'smiles': smiles})
    candidates.reverse()
    hit, rank = None, None
    print(f'Pred Compound: {intput_smi}')
    if real_smiles:
        hit = real_smiles in [candidate['smiles'] for candidate in candidates]
        if hit:
            print('target compound in candidates')
        else:
            print('target compound not in candidates')
        print(f'Target Compound: {real_smiles}\n')
        print(candidates)
    if out_file:
        with open(out_file, 'a') as f:
            if real_smiles:
                rank = []
                f.write(f'Target Compound: {real_smiles}\n')
                if hit:
                    f.write(f'Target compound in candidates\n')
                    for i, candidate in enumerate(candidates):
                        if real_smiles == candidate['smiles']:
                            rank.append(i+1)
                else:
                    f.write(f'Target compound not in candidates\n')
            for i, candidate in enumerate(candidates):
                f.write(f"candidate {i+1}\nsimilarity:{candidate['sim']} {candidate['smiles']}\n")
    return candidates, hit, rank

def direct_predict(
    model: TranslationModel,
    properties_model: PropertiesModel,
    device: torch.device,
    smiles_list,
    randomizedsmiles_list=None
):
    model.eval()
    sme = SmilesEnumerator()
    randomized_smiles_ids = smiles_batch_indices(randomizedsmiles_list if randomizedsmiles_list else [sme.randomize_smiles(smiles) for smiles in smiles_list])
    # randomized_smiles_ids = smiles_batch_indices(smiles_list)
    canonical_smiles_ids = batch_add_eos_sos([remove_salt_stereo(smiles) for smiles in smiles_list])
    tensors = [[r_idx, c_idx] for r_idx, c_idx in zip(randomized_smiles_ids, canonical_smiles_ids)]
    tensors.sort(key=lambda x: len(x[0]), reverse=True)
    randomized_smiles = [torch.tensor(r_idx, device=device) for r_idx, _ in tensors]
    canonical_smiles = [torch.tensor(c_idx, device=device) for _, c_idx in tensors]
    del tensors
    mols = [Chem.MolFromSmiles(smiles) for smiles in smiles_list]
    h_count_list = [[0]] * len(mols)
    form_list = [0] * len(mols)
    elements_chempy = get_elements_chempy(ELEMENTS.keys())
    elements_chempy_map = get_elements_chempy_map(elements_chempy)
    for idx, mol in enumerate(mols):
        fo = rdMolDescriptors.CalcMolFormula(mol)
        fo_map = get_elements_chempy_umap(fo)
        fo = [0] * len(ELEMENTS)
        for chempy_idx, count in fo_map.items():
            if chempy_idx:  # 0 -> charge ['+', '-']
                fo[elements_chempy_map[chempy_idx]] = count
            if chempy_idx == 1:  # H -> 1
                h_count_list[idx] = [count]
        form_list[idx] = fo
    properties = [
        form_list,  # formula
        [[Descriptors.MolLogP(m)] for m in mols],  # logP
        [[Descriptors.MolMR(m)] for m in mols],  # Molar refractivity
        [[Descriptors.NumValenceElectrons(m)] for m in mols],  # Number of valence electrons
        [[Descriptors.NumHDonors(m)] for m in mols],  # Number of hydrogen bond donors
        [[Descriptors.NumHAcceptors(m)] for m in mols],  # Number of hydrogen bond acceptors
        [[Descriptors.BalabanJ(m)] for m in mols],  # Balaban’s J value
        [[Descriptors.TPSA(m)] for m in mols],  # Topological polar surface area
        [[Descriptors.qed(m)] for m in mols],  # Drug likeliness (QED)
        [[sascorer.calculateScore(m)] for m in mols]  # Synthetic accessibility (SA)
    ]
    properties = torch.concat([torch.tensor(ins, dtype=torch.float32) for ins in properties], dim=1).to(device)
    mu, logvar, z, kl_loss, recon_loss, preds = model(
        randomized_smiles, canonical_smiles
    )
    # l, y = model.forward_decoder([canonical_smiles[2]], z[2].unsqueeze(0))
    # print(f'Pred smiles: {idx_to_smiles(y.argmax(2)[0, :-1])}\n')
    # print(z[0])
    for i, pred in enumerate(preds):
        print(f'Real randomized smiles: {idx_to_smiles(randomized_smiles[i])}\n')
        print(f'Real canonical smiles: {idx_to_smiles(canonical_smiles[i][1:])}\n')
        print(f'Pred smiles: {idx_to_smiles(pred.argmax(1)[:-1])}\n')
        p_loss, y = properties_model(z[i], properties[i])
        # print(y)
        # print(p_loss)
        # print(properties[i])
    return z

def direct_inference(
    device: torch.device,
    auto_encoder: TranslationModel,
    model: Union[Net1D, AttentionConv, AttentionConv2D, Conv1D],
    dataloader: DataLoader,
    topk: int,
    beam_width: int,
    max_len: int,
    target_compound=None
):
    results = []
    model.eval()
    auto_encoder.eval()
    for batch in tqdm(dataloader):
        if isinstance(model, AttentionConv2D):
            m_z, intensity = batch
            z = model(m_z, intensity)
        elif isinstance(model, Conv1D):
            spectrum = batch.to(device)
            z = model(spectrum)      
        elif isinstance(model, Net1D):
            spectrum = batch.to(device)
            z = model(spectrum)
        # candidates = auto_encoder.beam_search(batch_size=batch_size,
        #                                       max_len=max_len,
        #                                       z=z,
        #                                       beam_width=beam_width,
        #                                       topk=topk,
        #                                       device=device)
        # candidates = auto_encoder.decode_candidates(candidates, topk)
        # results.extend(candidates)
        candidates_list = ['' for _ in range(len(z))]
        for _ in range(topk):
            candidates = auto_encoder.sample(n_batch=len(z), max_len=max_len, z=z, topk=topk, device=device)
            for i, item in enumerate(candidates_list):

                candidates_list[i] = f'{candidates_list[i]}{candidates[i][5:-5]},'
        results.extend(candidates_list)

    with open('./inference_test_results.txt', 'a') as f:
        for i, item in enumerate(results):
            f.write(f"{i+1}: {item[:-1]}\n")
    return results

def inference(dataset_path=None,
              data_id_path=None,
              topk=20,
              data_file=None,
              out_dir=None,
              mode='inference',
              ignore_MzRange='not ignore',
              ignore_CE='not ignore'):
    torch.manual_seed(57)
    np.random.seed(57)
    autoencoder_config_path = get_absolute_path('./config/gru_config.json')
    model_config_path = get_absolute_path('./config/ms2conv.json')
    dataset_path = get_absolute_path('../../DATASET/randomizedsmiles.tsv') if dataset_path is None else dataset_path
    data_id_path = get_absolute_path('../../DATASET/randomizedsmiles.id') if data_id_path is None else data_id_path
    data_file = get_absolute_path('../../DATASET/testset150.mgf') if data_file is None else data_file
    out_dir = get_absolute_path('./results') if out_dir is None else out_dir
    batch_size = 1
    max_num_peaks = 100
    min_num_peaks = 5
    resolution = 1
    loss_mz_from = 50.0
    loss_mz_to = 1000.0
    ce_range = [10.0, 46.0]
    mass_range = [50.0, 1000.0]
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    autoencoder_config = GruAutoEncoderConfig.load_config(
        config_path=autoencoder_config_path,
        data_id_path=data_id_path,
        data_path=dataset_path
    )
    autoencoder = TranslationModel(autoencoder_config).to(device)
    for param in autoencoder.parameters():
        param.requires_grad = False
    if os.path.isfile(get_absolute_path(autoencoder_config.path.save_model_path)):
        autoencoder.load_state_dict(torch.load(get_absolute_path(autoencoder_config.path.save_model_path), map_location=device))
        print('successfully load previous best autoencoder parameters')
    else:
        raise ValueError('autoencoder not exist')

    model_config = Ms2VecConfig.load_config(model_config_path,
                                            train_data_path=data_file,
                                            test_data_path=data_file)
    src_pad_idx = 0
    model = AttentionConv2D(device=device, config=model_config, src_pad_idx=src_pad_idx).to(device)

    if os.path.isfile(get_absolute_path(model_config.path.save_model_path)):
        model.load_state_dict(torch.load(get_absolute_path(model_config.path.save_model_path), map_location=device))
        print('successfully load previous best model parameters')
    else:
        raise ValueError('ms2vec not exist')

    train_loader, val_loader = load_dataset(config=autoencoder_config, 
                                            ins_collate_fn=single_buffer_collate_fn, 
                                            batch_collate_fn=batch_buffer_collate_fn,
                                            to_device_collate_fn=to_device_collate_fn)
    if isinstance(model, AttentionConv2D):
        dataset = InferenceSpectrumDataset(data_file=data_file,
                                           transform=inference_spec_collate_fn,
                                           device=device,
                                           loss_mz_from=loss_mz_from,
                                           loss_mz_to=loss_mz_to,
                                           ce_range=ce_range,
                                           mass_range=mass_range,
                                           max_num_peaks=max_num_peaks,
                                           min_num_peaks=min_num_peaks,
                                           resolution=resolution,
                                           ignore_MzRange=ignore_MzRange,
                                           ignore_CE=ignore_CE)

        dataloader = DataLoader(dataset=dataset,
                                collate_fn=inference_batch_spec_collate_fn,
                                batch_size=batch_size,
                                drop_last=False)
    else:
        raise ValueError('Unknown model')
    results = inference_eval(compound_loader=train_loader,
                             spectrum_loader=dataloader,
                             autoencoder=autoencoder,
                             model=model,
                             topk=topk,
                             config=autoencoder_config,
                             device=device,
                             data_file=data_file,
                             out_dir=out_dir,
                             mode=mode)
    return results
