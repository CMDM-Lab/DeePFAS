
import copy
import json
import random
import re
import sys
from typing import List, Optional, Tuple

import numpy as np
import torch
from molmass import Formula
from pyteomics import mgf
from rdkit import Chem
from rdkit.Chem import Descriptors
from tqdm import tqdm
from DeePFAS.utils.smiles_process.functions import (VOC_MAP,
                                                    tokens_add_sos_eos,
                                                    tokens_from_smiles)
from ..smiles_process import (ELEMENTS, VOC_MAP, batch_add_eos_sos,
                              elements_filter, randomize_smiles,
                              remove_salt_stereo, smiles_batch_indices)
from ..SmilesEnumerator import SmilesEnumerator
from .definitions import CHARGE_FACTOR_MAP, none_or_nan


def spec2vec(spectrum: List[float],
             max_mz: float,
             min_mz: float,
             resolution: int):
    # spectrum: [(m_z, int), ...]
    mult = pow(10, resolution)
    length = int((max_mz - min_mz) * mult)
    vec = np.zeros(length)
    for m_z, it in spectrum:
        m_z = round(float(m_z), resolution) * mult
        try:
            vec[int(m_z) - int(min_mz * mult)] = vec[int(m_z) - int(min_mz * mult)] + it
        except:
            print(m_z)
    vec = vec / np.max(vec)
    return torch.tensor(vec.tolist(), dtype=torch.float32)

def get_dim(max_mz, min_mz, resolution):
    mult = pow(10, resolution)
    return int((max_mz - min_mz) * mult)

def tokenzie_mz(spectrum: torch.Tensor,
                resolution: int):
    mult = pow(10, resolution)
    spectrum[:, 1] = spectrum[:, 1] / torch.max(spectrum[:, 1])
    spectrum[:, 0] = (spectrum[:, 0] * mult).int()
    return spectrum

def batch_cal_losses(batch,
                     loss_mz_from=0,
                     loss_mz_to=1000.0):
    return [
        cal_losses(pepmass, m_z, intensity, loss_mz_from, loss_mz_to) 
        for pepmass, m_z, intensity in batch
    ]

def cal_losses(precursor_mz,
               m_z,
               intensity,
               loss_mz_from=0, 
               loss_mz_to=1000.0):

    losses_mz = (precursor_mz - m_z)[::-1]
    losses_intensity = intensity[::-1]
    mask = np.where((losses_mz >= loss_mz_from)
                    & (losses_mz <= loss_mz_to))

    return losses_mz[mask], losses_intensity[mask]

def read_mgf(mgf_path: str):
    # with mgf.MGF(mgf_path) as mgf_file:
    with mgf.read(mgf_path) as mgf_file:
        data = mgf_file[0]
    return data

def randomized_spectrum(m_z, intensity):
    randomized_indices = torch.randperm(m_z.shape[0])
    return (
        m_z[randomized_indices],
        intensity[randomized_indices]
    )

def spec_collate_fn(data):
    m_z, intensity = torch.from_numpy(data['m/z array']).float(), torch.from_numpy(data['intensity array']).float()
    m_z, intensity = randomized_spectrum(m_z, intensity)
    peaks = torch.stack([m_z, intensity], dim=0).T  # [[mz, int], ...]
    smiles = Chem.MolToSmiles(Chem.MolFromSmiles(data['params']['canonicalsmiles']), 
                              isomericSmiles=False, 
                              canonical=True)
    return (
        smiles,
        data['params']['randomizedsmiles'],
        data['params']['pepmass'],
        peaks
    )

def inference_spec_collate_fn(spec, 
                              device,
                              loss_mz_from, 
                              loss_mz_to,
                              max_num_peaks,
                              min_num_peaks,
                              ce_range,
                              mass_range,
                              resolution,
                              ignore_MzRange='not ignore',
                              ignore_CE='not ignore'):

    if spec.get('m/z array') is None:
        raise ValueError('m/z array is empty')
    # elif len(spec['m/z array']) < min_num_peaks:
    #     print(spec)
    #     raise ValueError(f'num of peaks less than {min_num_peaks}')
    elif spec['m/z array'][-1] > mass_range[1]:
        raise ValueError(f'm/z peak is bigger than {mass_range[1]}')
    elif spec['m/z array'][0] < mass_range[0] and ignore_MzRange == 'not ignore':
        raise ValueError(f'm/z peak is smaller than {mass_range[0]}')
    elif int(spec['params']['mslevel']) != 2:
        raise ValueError('input spectra is not MS2 spectra')
    elif spec['params']['precursor_type'] not in ['[M-H]-', '[M-H]1-']:
        raise ValueError('invalid precursor type, make sure precursor type belongs to [M-H]-')
    ace = parse_ace_str(spec['params']['collision_energy'])
    if none_or_nan(ace):
        nce = parse_nce_str(spec['params']['collision_energy'])
        row = {
            'nce': nce,
            'prec_type': spec['params']['precursor_type'],
            'prec_mz': spec['params']['pepmass'][0]
        }
        ace = nce_to_ace(row)
    if (ace < ce_range[0] or ace > ce_range[1]) and ignore_CE == 'not ignore':
        raise ValueError(f'collision energy is not in range [{ce_range[0]}, {ce_range[1]}]')

    losses_mz, losses_intensity = cal_losses(spec['params']['pepmass'][0], 
                                             spec['m/z array'], 
                                             spec['intensity array'],
                                             loss_mz_from,
                                             loss_mz_to)
    losses_mz, losses_intensity = list(losses_mz), list(losses_intensity)
    m_z, intensity = list(spec['m/z array']) + losses_mz, list(spec['intensity array']) + losses_intensity
    spec_map = {}
    # merge peak's intensity (average)
    for mz, it in zip(m_z, intensity):
        mz = round(mz, resolution)
        if spec_map.get(mz):
            spec_map[mz][1] += 1
            spec_map[mz][0] += it
        else:
            spec_map[mz] = [it, 1]
    spec_map = {k: v[0] / v[1] for k, v in spec_map.items()}
    # removed peaks whose intensity values were smaller than 1%
    m_z, intensity = [k for k, _ in spec_map.items()], [v for _, v in spec_map.items()]
    max_it = max(intensity)
    spec_list = [[k, v] for k, v in zip(m_z, intensity) if v / max_it >= 0.01]
    spec_list.sort(key=lambda x: x[1], reverse=True)
    if len(spec_list) > max_num_peaks:
        spec_list = spec_list[:max_num_peaks]
    spec_list = np.array(spec_list)
    spec_list[:, 1] /= max(spec_list[:, 1])
    # elif len(spec_list) < min_num_peaks:
    #     print(spec)
    #     print(spec_list)
    #     raise ValueError(f'num of peaks < {min_num_peaks}')
    m_z, intensity = [k for k, v in spec_list], [v for k, v in spec_list]
    m_z, intensity = torch.tensor(m_z, device=device), torch.tensor(intensity, device=device)
    # m_z, intensity = randomized_spectrum(m_z, intensity)
    # smiles = spec['params']['canonicalsmiles']
    # mw = Descriptors.ExactMolWt(Chem.MolFromSmiles(smiles))
    # diff = cal_diff(mw, spec['params']['pepmass'][0], spec['params']['precursor_type'])

    return torch.tensor([[k, v] for k, v in zip(m_z.tolist(), intensity.tolist())], device=device)

def normalize_peaks(spec, resolution):
    spec[:, 1] = spec[:, 1] / torch.max(spec[:, 1])
    spec[:, 0] = spec[:, 0].round(decimals=resolution)
    return spec

def inference_batch_spec_collate_fn(batch, resolution=1):
    peaks = [tokenzie_mz(data, resolution) for data in batch]
    intensity = [peak[:, 1] for peak in peaks]
    m_z = [peak[:, 0].long() for peak in peaks]
    return (
        [peak.tolist() for peak in m_z],
        [peak.tolist() for peak in intensity]
    )

def smiles_collate_fn(data):
    # smiles = Chem.MolToSmiles(
    #     Chem.MolFromSmiles(data['params']['canonicalsmiles']), 
    #     isomericSmiles=False, 
    #     canonical=True
    # )

    return data['params']['canonicalsmiles']

def batch_smiles_collate_fn(batch):
    sme = SmilesEnumerator()
    smiles_list = batch
    randomized_smiles_ids = smiles_batch_indices([sme.randomize_smiles(smiles) for smiles in smiles_list])
    canonical_smiles_ids = batch_add_eos_sos([remove_salt_stereo(smiles) for smiles in smiles_list])
    tensors = [[r_idx, c_idx] for r_idx, c_idx in zip(randomized_smiles_ids, canonical_smiles_ids)]
    tensors.sort(key=lambda x: len(x[0]), reverse=True)
    return (
        [torch.tensor(r_idx) for r_idx, _ in tensors],
        [torch.tensor(c_idx) for _, c_idx in tensors],
    )

def batch_spec_collate_fn(batch):

    randomized_smiles_ids = [data['randomizedsmiles'] for data in batch]
    canonical_smiles_ids = [data['canonicalsmiles'] for data in batch]
    # canonical_smiles_ids = [
    #     tokens_add_sos_eos(
    #         [VOC_MAP[token] for token in tokens_from_smiles([data['canonicalsmiles']])]
    #     ) for data in batch
    # ]
    # randomized_smiles_ids = [
    #     [VOC_MAP[token] for token in tokens_from_smiles([data['randomizedsmiles']])]
    #     for data in batch
    # ]
    m_z = [data['m_z'] for data in batch]
    intensity = [data['intensity']for data in batch]

    tensors = [[r_idx, c_idx, m, i] for r_idx, c_idx, m, i in zip(randomized_smiles_ids, canonical_smiles_ids, m_z, intensity)]
    tensors.sort(key=lambda x: len(x[0]), reverse=True)

    return (
        [torch.tensor(r_idx).long() for r_idx, _, _, _ in tensors],
        [torch.tensor(c_idx).long() for _, c_idx, _, _ in tensors],
        [m for _, _, m, _ in tensors],
        [i for _, _, _, i in tensors],
    )

def batch_spec_collate_to_device(batch, device):
    return (
        [data.to(device) for data in batch[0]],
        [data.to(device) for data in batch[1]],
        batch[2],
        batch[3],
    )

def test_batch_spec_collate_fn(batch, mode='eval'):

    randomized_smiles_ids = [data['randomizedsmiles'] for data in batch] if mode == 'eval' else None
    canonical_smiles_ids = [data['canonicalsmiles'] for data in batch] if mode == 'eval' else None
    m_z = [data['m_z'] for data in batch]
    intensity = [data['intensity']for data in batch]
    pepmass = [data['pepmass']for data in batch]
    titles = [data['title'] for data in batch]
    precursor_type = [data['precursor_type']for data in batch]
    if mode == 'eval':
        tensors = [[r_idx, c_idx, m, i, pe, pr, ti] for r_idx, c_idx, m, i, pe, pr, ti in zip(
            randomized_smiles_ids, canonical_smiles_ids, m_z, intensity, pepmass, precursor_type, titles,
        )]
        tensors.sort(key=lambda x: len(x[0]), reverse=True)

    return (
        [torch.tensor(r_idx).long() for r_idx, _, _, _, _, _, _ in tensors],
        [torch.tensor(c_idx).long() for _, c_idx, _, _, _, _, _ in tensors],
        [m for _, _, m, _, _, _, _ in tensors],
        [i for _, _, _, i, _, _, _ in tensors],
        [pe for _, _, _, _, pe, _, _ in tensors],
        [pr for _, _, _, _, _, pr, _ in tensors],
        [ti for _, _, _, _, _, _, ti in tensors],
    )  if mode == 'eval' else (None, None, m_z, intensity, pepmass, precursor_type, titles)

def test_batch_spec_collate_to_device(batch, mode, device):
    return (
        [data.to(device) for data in batch[0]] if mode == 'eval' else None,
        [data.to(device) for data in batch[1]] if mode == 'eval' else None,
        batch[2],
        batch[3],
        batch[4],
        batch[5],
        batch[6],
    )

def spec_conv_collate_fn(spectrum):
    peaks = [[mz, it] for mz, it in zip(spectrum['m/z array'], spectrum['intensity array'])]
    return {
        'smiles': spectrum['params']['canonicalsmiles'],
        'randomized_smiles': spectrum['params']['randomizedsmiles'],
        'peaks': peaks
    }

def batch_spec_conv_collate_fn(batch, max_mz=1000.0, min_mz=50.0, resolution=1):

    spec_list = [spec2vec(data['peaks'], max_mz, min_mz, resolution) for data in batch]
    randomized_smiles_ids = smiles_batch_indices([data['randomized_smiles'] for data in batch])
    canonical_smiles_ids = batch_add_eos_sos([data['smiles'] for data in batch])
    tensors = [[r_idx, c_idx, spec] for r_idx, c_idx, spec in zip(randomized_smiles_ids, canonical_smiles_ids, spec_list)]
    tensors.sort(key=lambda x: len(x[0]), reverse=True)
    return (
        torch.stack([spec for _, _, spec in tensors], dim=0).unsqueeze(1),
        [torch.tensor(r_idx) for r_idx, _, _ in tensors],
        [torch.tensor(c_idx) for _, c_idx, _ in tensors]
    )

def inference_spec_conv_collate_fn(spec, 
                                   loss_mz_from, 
                                   loss_mz_to,
                                   max_num_peaks,
                                   min_num_peaks,
                                   ce_range,
                                   mass_range,
                                   resolution):
    if spec.get('m/z array') is None:
        raise ValueError('m/z array is empty')
    # elif len(spec['m/z array']) < min_num_peaks:
    #     print(spec)
    #     raise ValueError(f'num of peaks less than {min_num_peaks}')
    elif spec['m/z array'][-1] > mass_range[1]:
        raise ValueError(f'm/z peak is bigger than {mass_range[1]}')
    elif spec['m/z array'][0] < mass_range[0]:
        raise ValueError(f'm/z peak is smaller than {mass_range[0]}')
    elif int(spec['params']['mslevel']) != 2:
        raise ValueError('input spectra is not MS2 spectra')
    elif spec['params']['precursor_type'] not in ['[M-H]-', '[M-H]1-']:
        raise ValueError('invalid precursor type, make sure precursor type belongs to [M-H]-')
    ace = parse_ace_str(spec['params']['collision_energy'])
    if none_or_nan(ace):
        nce = parse_nce_str(spec['params']['collision_energy'])
        row = {
            'nce': nce,
            'prec_type': spec['params']['precursor_type'],
            'prec_mz': spec['params']['pepmass'][0]
        }
        ace = nce_to_ace(row)
    if ace < ce_range[0] or ace > ce_range[1]:
        raise ValueError(f'collision energy is not in range [{ce_range[0]}, {ce_range[1]}]')

    losses_mz, losses_intensity = cal_losses(spec['params']['pepmass'][0], 
                                             spec['m/z array'], 
                                             spec['intensity array'],
                                             loss_mz_from,
                                             loss_mz_to)
    losses_mz, losses_intensity = list(losses_mz), list(losses_intensity)
    m_z, intensity = list(spec['m/z array']) + losses_mz, list(spec['intensity array']) + losses_intensity
    spec_map = {}
    # merge peak's intensity (average)
    for mz, it in zip(m_z, intensity):
        mz = round(mz, resolution)
        if spec_map.get(mz):
            spec_map[mz][1] += 1
            spec_map[mz][0] += it
        else:
            spec_map[mz] = [it, 1]
    spec_map = {k: v[0] / v[1] for k, v in spec_map.items()}
    # removed peaks whose intensity values were smaller than 10%
    m_z, intensity = [k for k, _ in spec_map.items()], [v for _, v in spec_map.items()]
    max_it = max(intensity)
    spec_list = [[k, v] for k, v in zip(m_z, intensity) if v / max_it > 0.1]
    spec_list.sort(key=lambda x: x[1], reverse=True)
    if len(spec_list) > max_num_peaks:
        spec_list = spec_list[:max_num_peaks]
    if len(spec_list) < 1:
        raise ValueError(f'num of peaks < {min_num_peaks}')

    return spec_list

def inference_batch_spec_conv_collate_fn(batch, device, max_mz=1000.0, min_mz=50.0, resolution=0):
    spec_list = [spec2vec(data, max_mz, min_mz, resolution) for data in batch]
    return torch.stack(spec_list, dim=0).unsqueeze(1).to(device)

# ------ reference MassFormer ------ #

def get_charge(prec_type_str):

    end_brac_idx = prec_type_str.index(']')
    charge_str = prec_type_str[end_brac_idx + 1:]
    if charge_str == '-':
        charge_str = '1-'
    elif charge_str == '+':
        charge_str = '1+'
    assert len(charge_str) >= 2
    sign = charge_str[-1]
    if sign not in ['+', '-']:
        print(prec_type_str)
    assert sign in ['+', '-']
    magnitude = int(charge_str[:-1])
    if sign == '+':
        charge = magnitude
    else:
        charge = -magnitude
    return charge

def nce_to_ace_helper(nce, charge, prec_mz):

    if charge in CHARGE_FACTOR_MAP:
        charge_factor = CHARGE_FACTOR_MAP[charge]
    else:
        charge_factor = CHARGE_FACTOR_MAP['large']
    ace = (nce * prec_mz * charge_factor) / 500.0
    return ace


def ace_to_nce_helper(ace, charge, prec_mz):

    if CHARGE_FACTOR_MAP.get(charge):
        charge_factor = CHARGE_FACTOR_MAP[charge]
    else:
        charge_factor = CHARGE_FACTOR_MAP['large']
    nce = (ace * 500.0) / (prec_mz * charge_factor)
    return nce

def nce_to_ace(row):

    prec_mz = row['prec_mz']
    nce = row['nce']
    prec_type = row['prec_type']
    # charge = np.abs(get_charge(prec_type))
    charge = 1.0
    ace = nce_to_ace_helper(nce, charge, prec_mz)
    return ace


def ace_to_nce(row):

    prec_mz = row['prec_mz']
    ace = row['ace']
    prec_type = row['prec_type']
    # charge = np.abs(get_charge(prec_type))
    charge = 1.0
    nce = ace_to_nce_helper(ace, charge, prec_mz)
    return nce

def parse_nce_str(ce_str):

    if none_or_nan(ce_str):
        return np.nan
    matches = {
        # nist ones
        r'^NCE=[\d]+[.]?[\d]*% [\d]+[.]?[\d]*eV$': lambda x: float(x.split()[0].lstrip('NCE=').rstrip('%')),
        r'^NCE=[\d]+[.]?[\d]*%$': lambda x: float(x.lstrip('NCE=').rstrip('%')),
        # other ones
        # this case is ambiguous
        r'^[\d]+[.]?[\d]*$': lambda x: 100. * float(x) if float(x) < 2. else np.nan,
        r'^[\d]+[.]?[\d]*[ ]?[%]? \(nominal\)$': lambda x: float(x.rstrip(' %(nominal)')),
        r'^HCD [\d]+[.]?[\d]*%$': lambda x: float(x.lstrip('HCD ').rstrip('%')),
        r'^[\d]+[.]?[\d]* NCE$': lambda x: float(x.rstrip('NCE')),
        r'^[\d]+[.]?[\d]*\(NCE\)$': lambda x: float(x.rstrip('(NCE)')),
        r'^[\d]+[.]?[\d]*[ ]?%$': lambda x: float(x.rstrip(' %')),
        r'^HCD \(NCE [\d]+[.]?[\d]*%\)$': lambda x: float(x.lstrip('HCD (NCE').rstrip('%)')),
    }
    for k, v in matches.items():
        if re.match(k, ce_str):
            return v(ce_str)
    return np.nan

def parse_ace_str(ce_str):

    if none_or_nan(ce_str):
        return np.nan
    matches = {
        # nist ones
        # this case is ambiguous (float(x) >= 2. or float(x) == 0.)
        r'^[\d]+[.]?[\d]*$': lambda x: float(x),
        r'^[\d]+[.]?[\d]*[ ]?eV$': lambda x: float(x.rstrip(' eV')),
        r'^NCE=[\d]+[.]?[\d]*% [\d]+[.]?[\d]*eV$': lambda x: float(x.split()[1].rstrip('eV')),
        # other ones
        r'^[\d]+[.]?[\d]*HCD$': lambda x: float(x.rstrip('HCD')),
        r'^CE [\d]+[.]?[\d]*$': lambda x: float(x.lstrip('CE ')),
    }
    for k, v in matches.items():
        if re.match(k, ce_str):
            return v(ce_str)
    return np.nan

def merge_diff_ce(files, 
                  max_len, 
                  max_mw, 
                  min_num_peaks,
                  out_path, 
                  ce_range=[10.0, 45.0],
                  mass_range=[50.0, 500.0],
                  valid_mol=None,
                  tolerence=1,
                  precursor_type=None,
                  mode=None,
                  loss_mz_from=0,
                  loss_mz_to=1000.0):
    mol_map = {}
    for file in files:
        with mgf.read(file) as mgf_file:
            for i in tqdm(range(len(mgf_file))):
                canoical_smiles = remove_salt_stereo(mgf_file[i]['params']['canonicalsmiles'])
                if canoical_smiles is None:
                    continue
                mol = Chem.MolFromSmiles(canoical_smiles)
                mw = Descriptors.ExactMolWt(mol)
                if not elements_filter(mol, ELEMENTS):
                    continue
                ace = parse_ace_str(mgf_file[i]['params']['collision_energy'])
                if none_or_nan(ace):
                    nce = parse_nce_str(mgf_file[i]['params']['collision_energy'])
                    spec = {
                        'nce': nce,
                        'prec_type': mgf_file[i]['params']['precursor_type'],
                        'prec_mz': mgf_file[i]['params']['pepmass'][0]
                    }
                    ace = nce_to_ace(spec)
                if any([
                    int(mgf_file[i]['params']['mslevel']) != 2,
                    mgf_file[i]['params']['precursor_type'] not in precursor_type if precursor_type else None,
                    mgf_file[i]['params']['source_instrument'] == 'nan' if mgf_file[i]['params'].get('source_instrument') else False,
                    len(mgf_file[i]['params']['canonicalsmiles']) > max_len,
                    mw > max_mw,
                    ace < ce_range[0] or ace > ce_range[1],
                    # not elements_filter(mol, ELEMENTS),
                    mgf_file[i].get('m/z array') is None or len(mgf_file[i]['m/z array']) < min_num_peaks or
                    mgf_file[i]['m/z array'][-1] > mass_range[1] or mgf_file[i]['m/z array'][0] < mass_range[0],
                    (valid_mol.get(canoical_smiles) is None) if valid_mol else False
                ]):
                    continue
                if mode == 'neg':
                    if mgf_file[i]['params']['precursor_type'][-1] != '-':
                        continue
                elif mode == 'pos':
                    if mgf_file[i]['params']['precursor_type'][-1] != '+':
                        continue

                spectrum = copy.deepcopy(mgf_file[i])
                spectrum['params']['randomizedsmiles'] = valid_mol[canoical_smiles]
                spectrum['params']['canonicalsmiles'] = canoical_smiles
                if mol_map.get(canoical_smiles) is None:
                    mol_map[canoical_smiles] = []
                mol_map[canoical_smiles].append(spectrum)
    print(f'length of mol map: {len(mol_map)}')
    filte_spectrum4(out_path, mol_map, min_num_peaks, tolerence, loss_mz_from, loss_mz_to)

def cal_nce_num(files, 
                max_len, 
                max_mw,
                min_num_peaks,
                nce_range,
                valid_mol=None,
                precursor_type=None):
    mol_map = {}
    for file in files:
        with mgf.read(file) as mgf_file:
            for i in tqdm(range(len(mgf_file))):
                canoical_smiles = remove_salt_stereo(mgf_file[i]['params']['canonicalsmiles'])
                ce = mgf_file[i]['params']['collision_energy']
                nce = parse_nce_str(ce)
                if canoical_smiles is None or none_or_nan(nce):
                    continue
                mol = Chem.MolFromSmiles(canoical_smiles)
                mw = Descriptors.ExactMolWt(mol)
                if any([
                    mgf_file[i]['params']['precursor_type'] not in precursor_type,
                    mgf_file[i]['params']['source_instrument'] != 'HCD',
                    len(mgf_file[i]['params']['canonicalsmiles']) > max_len,
                    mw > max_mw,
                    not elements_filter(mol, ELEMENTS),
                    mgf_file[i].get('m/z array') is None or len(mgf_file[i]['m/z array']) < min_num_peaks or
                    mgf_file[i]['m/z array'][-1] > 500.0 or mgf_file[i]['m/z array'][0] < 50.0,
                    (valid_mol.get(canoical_smiles) is None) if valid_mol else False,
                    nce < nce_range[0] or nce > nce_range[1]
                ]):
                    continue
                if mol_map.get(canoical_smiles) is None:
                    mol_map[canoical_smiles] = []
                mol_map[canoical_smiles].append(mgf_file[i])
    print(len(mol_map))
    print(sum([len(v) for k, v in mol_map.items()]))

def cal_ace_num(files, 
                max_len, 
                max_mw,
                min_num_peaks,
                ace_range,
                valid_mol=None,
                precursor_type=None):
    mol_map = {}
    for file in files:
        with mgf.read(file) as mgf_file:
            for i in tqdm(range(len(mgf_file))):
                ace = parse_ace_str(mgf_file[i]['params']['collision_energy'])
                spec = {
                    'prec_type': mgf_file[i]['params']['precursor_type'],
                    'prec_mz': mgf_file[i]['params']['pepmass'][0]
                }
                if none_or_nan(ace):
                    spec['nce'] = parse_nce_str(mgf_file[i]['params']['collision_energy'])
                    # ace = round(nce_to_ace(spec))
                    ace = nce_to_ace(spec)
                canoical_smiles = remove_salt_stereo(mgf_file[i]['params']['canonicalsmiles'])
                mol = Chem.MolFromSmiles(canoical_smiles)
                mw = Descriptors.ExactMolWt(mol)
                if any([
                    mgf_file[i]['params']['precursor_type'] not in precursor_type,
                    mgf_file[i]['params']['source_instrument'] != 'HCD',
                    len(mgf_file[i]['params']['canonicalsmiles']) > max_len,
                    mw > max_mw,
                    not elements_filter(mol, ELEMENTS),
                    mgf_file[i].get('m/z array') is None or len(mgf_file[i]['m/z array']) < min_num_peaks or
                    mgf_file[i]['m/z array'][-1] > 500.0 or mgf_file[i]['m/z array'][0] < 50.0,
                    (valid_mol.get(canoical_smiles) is None) if valid_mol else False,
                    ace < ace_range[0] or ace > ace_range[1]
                ]):
                    continue
                if mol_map.get(canoical_smiles) is None:
                    mol_map[canoical_smiles] = []
                mol_map[canoical_smiles].append(mgf_file[i])
    print(len(mol_map))
    print(sum([len(v) for k, v in mol_map.items()]))

def cal_energy_distribution(path, mode='nce', max_len=110, max_mw=1000):
    assert mode in ['nce', 'ace']
    with mgf.read(path) as mgf_file:
        energy_map = {
            'pos': {},
            'neg': {}
        }
        mol_map = {}
        for i in tqdm(range(len(mgf_file))):
            mol = Chem.MolFromSmiles(mgf_file[i]['params']['canonicalsmiles'])
            mw = Descriptors.ExactMolWt(mol)
            if any([
                mgf_file[i]['params']['precursor_type'] not in ['[M-H]-', '[M+H]+'],
                mgf_file[i]['params']['source_instrument'] != 'HCD',
                len(mgf_file[i]['params']['canonicalsmiles']) > max_len,
                mw > max_mw,
                not elements_filter(mol, ELEMENTS)
            ]):
                continue
            energy = (
                parse_nce_str(mgf_file[i]['params']['collision_energy'])
                if mode == 'nce' else parse_ace_str(mgf_file[i]['params']['collision_energy'])
            )
            spec = {
                'prec_type': mgf_file[i]['params']['precursor_type'],
                'prec_mz': mgf_file[i]['params']['pepmass'][0]
            }
            if none_or_nan(energy):
                after_mode = 'nce' if mode == 'ace' else 'ace'
                spec[after_mode] = (
                    parse_ace_str(mgf_file[i]['params']['collision_energy']) 
                    if mode == 'nce' else parse_nce_str(mgf_file[i]['params']['collision_energy'])
                )
                energy = round(ace_to_nce(spec)) if mode == 'nce' else round(nce_to_ace(spec))

            ionization_mode = 'neg' if mgf_file[i]['params']['ionmode'] == 'Negative' else 'pos'
            if not energy_map[ionization_mode].get(energy):
                energy_map[ionization_mode][energy] = 1
            else:
                energy_map[ionization_mode][energy] += 1
        data = {
            'pos': {
                'Collision Energy': None,
                'Number': None,
                'Percentage': None
            },
            'neg': {
                'Collision Energy': None,
                'Number': None,
                'Percentage': None
            },
            'mode': mode
        }
        for ionization_mode in energy_map.keys():
            print(f'ionization mode: {ionization_mode}')
            num_of_energy = list(energy_map[ionization_mode].items())
            num_of_energy.sort(key=lambda x: x[1], reverse=True)
            for e, n in num_of_energy:
                print(f'{e}: {n}')
            print('\n')
            data[ionization_mode]['Collision Energy'] = [int(k) for k, _ in num_of_energy]
            data[ionization_mode]['Number'] = [v for _, v in num_of_energy]
            total = sum([n for e, n in num_of_energy])
            num_of_energy = [[e, n / total * 100.0] for e, n in num_of_energy]
            for e, n in num_of_energy:
                print(f'{e}: {n} %')
            print('\n')
            data[ionization_mode]['Percentage'] = [round(v, 1) for _, v in num_of_energy]
        return data

def cal_diff(mass, pepmass, adduct):
    mult = 1
    pattern = r"([+-][A-Za-z0-9]+|[+-]\di)"
    for i, a in enumerate(adduct):
        if a in ['-', '+']:
            try:
                if len(adduct[1:i]) > 1 and adduct[2] == 'M':
                    mult = int(adduct[1])
                break
            except:
                print(adduct)
                raise ValueError('adduct error')

    components = re.findall(pattern, adduct)
    tmp = 0
    for c in components:
        sign = 1 if c[0] == '+' else -1
        res = c[1:]
        if 'i' in c:
            tmp += sign * int(res[0]) * 1.00784
        else:
            try:
                formula = Formula(res)
                t_mass = formula.mass
            except:
                pattern = r"(\d+)([A-Z][a-z]?)"
                formula = re.findall(pattern, res)
                formula = [f'{f[1]}{f[0]}' for f in formula]
                formula = ''.join(formula)
                formula = Formula(formula)
                t_mass = formula.mass
            tmp += sign * t_mass
    # diff
    return pepmass - (mult * mass - tmp)

def cal_mw(pepmass, adduct):
    pattern = r"([+-][A-Za-z0-9]+|[+-]\di)"
    mult = 1
    for i, a in enumerate(adduct):
        if a in ['-', '+']:
            try:
                if len(adduct[1:i]) > 1 and adduct[2] == 'M':
                    mult = int(adduct[1])
                break
            except:
                print(adduct)
                raise ValueError('adduct error')
    components = re.findall(pattern, adduct)
    tmp = 0
    for c in components:
        sign = 1 if c[0] == '+' else -1
        res = c[1:]
        # if 'i' in c:
        #     tmp += sign * int(res[0]) * 1.00784
        # else:
        try:
            formula = Formula(res)
            t_mass = formula.mass
        except:
            pattern = r"(\d+)([A-Z][a-z]?)"
            formula = re.findall(pattern, res)
            formula = [f'{f[1]}{f[0]}' for f in formula]
            formula = ''.join(formula)
            formula = Formula(formula)
            t_mass = formula.mass
        tmp += sign * t_mass
    return (pepmass - tmp) / mult

def gen_mol_mgf(files, out_f):
    mol_map = {}
    data = []
    for file in files:
        with mgf.read(file) as f:
            for o in tqdm(f):
                smiles = o['params']['canonicalsmiles']
                mol = Chem.MolFromSmiles(smiles)
                if not elements_filter(mol, ELEMENTS):
                    continue
                if mol_map.get(smiles) is None:
                    mol_map[smiles] = o
    print(len(mol_map))
    for i, (k, v) in tqdm(enumerate(mol_map.items())):
        c = copy.deepcopy(v)
        c['params']['title'] = i
        data.append(c)
    mgf.write(data, out_f, file_mode='w', write_charges=False)

def merge_multiple_mgf(files, out_f, cond=None, shuffle=True):
    data = []
    for file in files:
        with mgf.read(file) as f:
            for o in tqdm(f):
                if cond:
                    if cond(o):
                        data.append(o)
                else:
                    data.append(o)
    if shuffle:
        random.shuffle(data)
    for i, o in tqdm(enumerate(data)):
        data[i]['params']['title'] = i
    mgf.write(data, out_f, file_mode='w', write_charges=False)

def split_dataset(path,
                  test_num_mol,
                  test_out,
                  train_out,
                  shuffle=True):
    with mgf.read(path) as f:
        train_data, test_data = [], []
        mol_map = {}
        for o in tqdm(f):
            smiles = o['params']['canonicalsmiles']
            smiles = remove_salt_stereo(smiles)
            if mol_map.get(smiles) is None:
                mol_map[smiles] = []
            mol_map[smiles].append(o)
        mol_map = [(k, v) for k, v in mol_map.items()]
        random.shuffle(mol_map)
        for k, v in mol_map[-test_num_mol:]:
            test_data.extend(v)
        for k, v in mol_map[:-test_num_mol]:
            train_data.extend(v)
        if shuffle:
            random.shuffle(test_data)
            random.shuffle(train_data)
        for i, o in tqdm(enumerate(test_data)):
            test_data[i]['params']['title'] = i
        for i, o in tqdm(enumerate(train_data)):
            train_data[i]['params']['title'] = i
        mgf.write(test_data, test_out, file_mode='w', write_charges=False)
        mgf.write(train_data, train_out, file_mode='w', write_charges=False)

def oversampling(file, out_file):
    mol_map = {}
    data = []
    with mgf.read(file) as f:
        for o in tqdm(f):
            smiles = o['params']['canonicalsmiles']
            if mol_map.get(smiles) is None:
                mol_map[smiles] = []
            mol_map[smiles].append(o)
    max_num = max([len(v) for k, v in mol_map.items()])
    for k, v in mol_map.items():
        data.extend(v)
        data.extend([random.choice(v) for _ in range(max_num - len(v))])
    random.shuffle(data)
    for i, o in enumerate(data):
        data[i]['params']['title'] = i
    print(f'Total num of data: {len(data)}')
    mgf.write(data, out_file, file_mode='w', write_charges=False)
