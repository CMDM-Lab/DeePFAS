"""
Define some functions.

Author: Heng Wang
Date: 1/23/2024
"""
import os
import re
import sys
from typing import List

import chempy
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F  # noqa: N812
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors, RDConfig, rdMolDescriptors
from rdkit.Chem.SaltRemover import SaltRemover

from ..SmilesEnumerator import SmilesEnumerator
from .definitions import (ELEMENTS, FINAL_CHAR, INITIAL_CHAR, PAD_CHAR, VOC,
                          VOC_MAP)

sys.path.append(os.path.join(RDConfig.RDContribDir, 'SA_Score'))
import sascorer


def tokens_add_pad(
    tokens: List[int],
    length: int,
    pad_char=VOC_MAP[PAD_CHAR],
    tail=False
):
    if tail:
        tokens = [pad_char] * (length - len(tokens)) + tokens
    else:
        tokens.extend([pad_char] * (length - len(tokens)))
    return tokens[:length]

def tokens_add_sos_eos(
    tokens: List[int],
    initial_char=INITIAL_CHAR,
    final_char=FINAL_CHAR,
):
    tokens.insert(0, VOC_MAP[initial_char])
    tokens.append(VOC_MAP[final_char])
    return tokens

def tokens_add_sos(
    tokens: List[int],
    initial_char=INITIAL_CHAR
):
    tokens.insert(0, VOC_MAP[initial_char])
    return tokens

def tokens_add_eos(
    tokens: List[int],
    final_char=FINAL_CHAR
):
    tokens.append(VOC_MAP[final_char])
    return tokens
def tokens_from_smiles(smiles: List[str]):
    return [
        # this make C102 -> 'C', '1', '0', '2'
        # re.findall(r'[^[]|\[.*?\]', s.replace('Cl', 'L').replace('Br', 'R'))
        re.findall(r'Cl|Br|[#%\)\(\+\-1032547698:=@CBFIHONPS\[\]cionps]', s)
        # this make C102 -> 'C', '102'
        # re.findall(r'\d+|[^[]|\[.*?\]', s.replace('Cl', 'L').replace('Br', 'R'))
        for s in smiles
    ][0]

def smiles_batch_indices(
    batch_smiles: List[str]
):

    return [
        [VOC_MAP[token] for token in tokens_from_smiles([smiles])]
        for smiles in batch_smiles
    ]

def batch_add_sos(
    batch_indices: List[str]
):

    return [
        tokens_add_sos(
            [VOC_MAP[token] for token in tokens_from_smiles([smiles])],
        )
        for smiles in batch_indices
    ]

def batch_add_eos(
    batch_indices: List[str]
):
    return [
        tokens_add_eos(
            [VOC_MAP[token] for token in tokens_from_smiles([smiles])],
        )
        for smiles in batch_indices
    ]

def batch_add_eos_sos(
    batch_indices: List[str]
):
    return [
        tokens_add_sos_eos(
            [VOC_MAP[token] for token in tokens_from_smiles([smiles])],
        )
        for smiles in batch_indices
    ]
def batch_add_pad(
    batch_indices: List[List[int]],
    length: int,
    pad_char: int,
    tail: bool = False
):

    return [tokens_add_pad(indices, length, pad_char, tail=tail) for indices in batch_indices]


def get_voc():
    """Build `VOC` mapping table, check if token in `VOC` or not.

    Returns:
        `VOC` mapping table
    """
    return {token: 1 for token in VOC}

def get_elements():
    """Get `ELEMENTS` list to cal formula (including H).

    However, `ELEMENTS` in config.py exclude H.

    Returns:
        lists of elements
    """
    return ['C', 'F', 'H', 'I', 'Cl', 'N', 'O', 'P', 'Br', 'S']

def get_elements_chempy(elements):
    """Map each element to an integer (`elements`).

    Args:
        elements: get_elements()

        ex:
            'H' -> 1
            'C' -> 0

    Returns:
        list including idx of atom in elements
    """
    return [
        chempy.util.periodic.atomic_number(ele)
        for ele in elements
    ]

def get_elements_chempy_map(elements_chempy):
    """Get map of elements.

        key -> idx

    Args:
        elements_chempy: get_elements_chempy()

    Returns:
        map of elements
    """
    return {key: idx for idx, key in enumerate(elements_chempy)}

def get_elements_chempy_umap(fo):
    """Get map of elements.

        key -> idx

    Args:
        fo: formula

    Returns:
        umap of elements
    """
    return {key: idx for key, idx in chempy.Substance.from_formula(fo).composition.items()}  # noqa: C416

def tokens_encode_one(tokens: str):
    return torch.tensor([VOC_MAP.get(c, 0) for c in tokens])

def tokens_onehot(tokens_mat):
    return F.one_hot(tokens_mat, len(VOC))

def get_hcount_mask():
    """mask mask pad and termination character outputs.

    Returns:
        hcount_mask:
            estimate H' count mask
    """
    pad_mask = tokens_onehot(tokens_encode_one(PAD_CHAR))
    term_mask = tokens_onehot(tokens_encode_one(FINAL_CHAR))
    return torch.ones_like(pad_mask) - pad_mask - term_mask

def open_logger() -> None:
    RDLogger.EnableLog('rdApp.*')

def close_logger() -> None:
    RDLogger.DisableLog('rdApp.*')

def remove_eol(data: List[str]):

    return [item.decode().replace('\n', '').strip() for item in data]

def single_buffer_collate_fn(line: str, use_properties=False):
    smiles = line.decode().replace('\n', '').strip()
    randomized_smiles_ids = [VOC_MAP[token] for token in tokens_from_smiles([smiles])]
    smiles = remove_salt_stereo(smiles)
    canonical_smiles_ids = tokens_add_sos_eos([VOC_MAP[token] for token in tokens_from_smiles([smiles])])
    if use_properties:
        mol = Chem.MolFromSmiles(smiles)
        elements_chempy = get_elements_chempy(ELEMENTS.keys())
        elements_chempy_map = get_elements_chempy_map(elements_chempy)
        fo = rdMolDescriptors.CalcMolFormula(mol)
        fo_map = get_elements_chempy_umap(fo)
        fo = [0] * len(ELEMENTS)
        for chempy_idx, count in fo_map.items():
            if chempy_idx:  # 0 -> charge ['+', '-']
                fo[elements_chempy_map[chempy_idx]] = count
        form_list = fo
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

    return (
        [randomized_smiles_ids, canonical_smiles_ids],
        form_list if use_properties else None
    )

def single_collate_fn(data: List[str], use_properties=False):
    data = remove_eol(data)
    sme = SmilesEnumerator()
    randomized_smiles_ids = smiles_batch_indices([sme.randomize_smiles(smiles) for smiles in data])
    canonical_smiles_ids = batch_add_eos_sos(data)
    tensors = [[r_idx, c_idx] for r_idx, c_idx in zip(randomized_smiles_ids, canonical_smiles_ids)]
    tensors.sort(key=lambda x: len(x[0]), reverse=True)

    if use_properties:
        # generate formula & H count
        mols = [Chem.MolFromSmiles(smiles) for smiles in data]
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

    return (
        tensors,
        properties if use_properties else None
    )

# def batch_buffer_collate_fn(data, use_properties=False):
#     tensors = [(ins[0][0], ins[0][1], ins[1]) for ins in data]
#     tensors.sort(key=lambda x: len(x[0]), reverse=True)
#     properties = [ins[2] for ins in tensors] if use_properties else None
#     tensors = [[ins[0], ins[1]] for ins in tensors]
#     return (
#         tensors,
#         properties
#     )

def batch_buffer_collate_fn(data, use_properties, use_contrastive_learning):
    # Please use partial to the function
    tensors = [
        (
            ins['rand_smiles_ids'],
            ins['caonical_smiles_ids'],
            ins['formula'],
            ins['neg_samples'],
        ) for ins in data
    ]
    tensors.sort(key=lambda x: len(x[0]), reverse=True)
    properties = [ins[2] for ins in tensors] if use_properties else None
    neg_samples = [ins[3] for ins in tensors] if use_contrastive_learning else None 
    aux_masks = ([True] + [False] * len(neg_samples[0])) if use_contrastive_learning else None
    # return metadata
    # randomized smiles ids: (B, seq_l)
    # caonical smiles ids: (B, seq_l)
    # properties: (B, properties_dim)
    # neg_samples: (B, num of neg samples, 4096)
    # aux_masks: (1 + num of neg samples)
    return (
        [r_idx for r_idx, _, _, _ in tensors],
        [c_idx for _, c_idx, _, _ in tensors],
        properties,
        neg_samples,
        aux_masks,
    )

def batch_to_device_collate_fn(data, device):

    r_ids, c_ids, properties, neg_samples, aux_masks = data
    def to_tensor(d, dtype=torch.float32):
        return None if d is None else torch.tensor(d, device=device, dtype=dtype)
    return (
        [to_tensor(r_idx).long() for r_idx in r_ids],
        [to_tensor(c_idx).long() for c_idx in c_ids],
        to_tensor(properties),
        to_tensor(neg_samples),
        to_tensor(aux_masks, dtype=torch.bool),
    )

def batch_collate_fn(data: List[str]):

    def get_device():
        if torch.cuda.is_available():
            dev = 'cuda:0'
        # elif torch.backends.mps.is_available():
        #     dev = 'mps'
        else:
            dev = 'cpu'
        return torch.device(dev)
    device = get_device()
    canonical_smiles_ids = batch_add_eos_sos(data)
    tensors = canonical_smiles_ids
    tensors.sort(key=lambda x: len(x), reverse=True)
    return (
        [torch.tensor(c_idx, device=device) for c_idx in tensors],
        [torch.tensor(c_idx, device=device) for c_idx in tensors]
    )

def to_device_collate_fn(data, use_properties=False):

    def get_device():
        if torch.cuda.is_available():
            dev = 'cuda:0'
        # elif torch.backends.mps.is_available():
        #     dev = 'mps'
        else:
            dev = 'cpu'
        return torch.device(dev)

    device = get_device()
    properties = torch.stack([torch.tensor(ins, dtype=torch.float32) for ins in data[1]]).to(device) if use_properties else None
    # properties = torch.concat([torch.tensor(p, dtype=torch.float32) for p in data[1]], dim=1).to(device) if use_properties else None
    # Return
    # -       1. randomized SMILES ids 
    # -       2. canonical SMILES ids
    # -       3. prperties of chemicals
    return (
        [torch.tensor(r_idx, device=device) for r_idx, _ in data[0]],
        [torch.tensor(c_idx, device=device) for _, c_idx in data[0]],
        properties
    )

def keep_largest_fragment(sml):
    mol_frags = Chem.GetMolFrags(Chem.MolFromSmiles(sml), asMols=True)
    largest_mol = None
    largest_mol_size = 0
    for mol in mol_frags:
        size = mol.GetNumAtoms()
        if size > largest_mol_size:
            largest_mol = mol
            largest_mol_size = size
    return Chem.MolToSmiles(largest_mol)

def elements_filter(mol, voc):

    atoms = list(set([atom.GetSymbol() for atom in mol.GetAtoms()]))
    for atom in atoms:
        if not voc.get(atom):
            return False
    return True

def remove_salt_stereo(sml, isomeric=False, canonical=True):
    try:
        remover = SaltRemover()
        sml = Chem.MolToSmiles(remover.StripMol(Chem.MolFromSmiles(sml),
                                                dontRemoveEverything=True),
                               isomericSmiles=isomeric, canonical=canonical)
        if '.' in sml:
            sml = keep_largest_fragment(sml)
    except:

        sml = None
    return sml

def canonical_smile(sml):
    return Chem.MolToSmiles(Chem.MolFromSmiles(sml), canonical=True)

def idx_to_smiles(idx_list: torch.Tensor):

    return ''.join([VOC[c.item()] for c in idx_list])

def randomize_smiles(smiles, canonical=True, isomericSmiles=False):
    """Perform a randomization of a SMILES string
    must be RDKit sanitizable"""
    m = Chem.MolFromSmiles(smiles)
    ans = list(range(m.GetNumAtoms()))
    np.random.shuffle(ans)
    nm = Chem.RenumberAtoms(m, ans)
    return Chem.MolToSmiles(nm, canonical=canonical, isomericSmiles=isomericSmiles)
