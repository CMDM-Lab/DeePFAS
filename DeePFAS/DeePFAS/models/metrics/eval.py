import os
import pickle
import sys

import torch.nn.functional as F
from IPython.display import SVG
from pyteomics import mgf
from rdkit import Chem, DataStructs
from rdkit.Chem import (AllChem, Descriptors, Draw, RDConfig, rdDepictor,
                        rdFMCS, rdMolDescriptors)
from rdkit.Chem.Draw import IPythonConsole, rdMolDraw2D

from DeePFAS.utils.smiles_process import (ELEMENTS, get_elements_chempy,
                                          get_elements_chempy_map,
                                          get_elements_chempy_umap,
                                          remove_salt_stereo)

sys.path.append(os.path.join(RDConfig.RDContribDir, 'SA_Score'))
import sascorer
import torch.nn as nn
from tqdm import tqdm


class CosSimLoss(nn.Module):
    def __init__(self, reduction=False):
        super(CosSimLoss, self).__init__()
        self.reduction = reduction
    def forward(self, inputs, targets):
        if self.reduction:
            return 1 - F.cosine_similarity(inputs, targets)
        return 1 - F.cosine_similarity(inputs, targets).mean()

def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

def get_mcs(candidates, original):
    """
    MCS-based metrics of the prediction with the max number of atoms in the maximum common substructure
    Args:
        pred_list: list of predicted smiles
        original: reference smiles 
    Output:
        closest_smiles: smiles with the maximum number of atoms in the MCS 
        mcs ratio for the smiles with the max number of atoms in the MCS 
        mcs tanimoto >>
        mcs coefficient  >>
    """
    smiles_list = [c['smiles'] for c in candidates]
    removed = 0
    original_mol = Chem.MolFromSmiles(original)
    original_atoms = original_mol.GetNumAtoms()
    max_mcs = 0
    min_mcs = 1e5
    mcs_list = []
    num_atoms_list = []
    closest_smiles = ''
    farest_smiles = ''
    for smi in tqdm(smiles_list):
        try:
            mols = [Chem.MolFromSmiles(original), Chem.MolFromSmiles(smi)]
            mcs = rdFMCS.FindMCS(mols, ringMatchesRingOnly=True, atomCompare=Chem.rdFMCS.AtomCompare.CompareElements, bondCompare=Chem.rdFMCS.BondCompare.CompareOrder, timeout=60)
            mcs_atoms = mcs.numAtoms
            if mcs_atoms > max_mcs:
                max_mcs = mcs_atoms
                closest_smiles = smi
            if mcs_atoms < min_mcs:
                min_mcs = mcs_atoms
                farest_smiles = smi
            mcs_list.append(mcs_atoms)
            num_atoms_list.append(Chem.MolFromSmiles(smi).GetNumAtoms())
        except:
            removed = removed + 1
    closest_mol = Chem.MolFromSmiles(closest_smiles)
    closest_atoms = closest_mol.GetNumAtoms()
    farest_mol = Chem.MolFromSmiles(farest_smiles)
    farest_atoms = farest_mol.GetNumAtoms()
    if not mcs_list:
        return None
    else:
        max_mcs_ratio = max_mcs / original_atoms
        max_mcs_tan = max_mcs / (original_atoms + closest_atoms - max_mcs)
        max_mcs_coef = max_mcs / min(original_atoms, closest_atoms)
        min_mcs_ratio = min_mcs / original_atoms
        min_mcs_tan = min_mcs / (original_atoms + farest_atoms - min_mcs)
        min_mcs_coef = min_mcs / min(original_atoms, farest_atoms)
        avg_mcs = sum(mcs_list) / len(mcs_list)
        avg_mcs_ratio = avg_mcs / original_atoms
        avg_atoms = sum(num_atoms_list) / len(num_atoms_list)
        avg_mcs_tan = avg_mcs / (original_atoms + avg_atoms - avg_mcs)
        avg_mcs_coef = avg_mcs / min(original_atoms, avg_atoms)
    return {
        'closest_smiles': closest_smiles, 
        'max_mcs_ratio': max_mcs_ratio,
        'max_mcs_tan': max_mcs_tan,
        'max_mcs_coef': max_mcs_coef,
        'farest_smiles': farest_smiles,
        'min_mcs_ratio': min_mcs_ratio,
        'min_mcs_tan': min_mcs_tan,
        'min_mcs_coef': min_mcs_coef,
        'avg_mcs_ratio': avg_mcs_ratio,
        'avg_mcs_tan': avg_mcs_tan,
        'avg_mcs_coef': avg_mcs_coef
    }

def get_sim(smi1, smi2):
    mol, mol2 = Chem.MolFromSmiles(smi1), Chem.MolFromSmiles(smi2)
    return DataStructs.FingerprintSimilarity(Chem.RDKFingerprint(mol), Chem.RDKFingerprint(mol2))

def get_4096_Morgan_sim(candidates, original):
    """
    Fingerprint similarity based on RDKFingerprint
    """
    smiles_list = [c['smiles'] for c in candidates]
    max_sim = -1
    avg_sim = 0
    min_sim = 1e5
    invalid_pred = 0
    mol_original = Chem.MolFromSmiles(original)
    fp_original = Chem.AllChem.GetMorganFingerprintAsBitVect(mol_original, radius=3, nBits=4096)
    closest_smiles = ''
    farest_smiles = ''
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            invalid_pred += 1
            continue
        sim = DataStructs.FingerprintSimilarity(
            fp_original,
            Chem.AllChem.GetMorganFingerprintAsBitVect(mol, radius=3, nBits=4096),
        )
        if sim > max_sim:
            max_sim = sim
            closest_smiles = smi
        if sim < min_sim:
            min_sim = sim
            farest_smiles = smi
        avg_sim += sim
    if not avg_sim:
        return None
    avg_sim = avg_sim / (len(smiles_list) - invalid_pred)
    return {
        'max_sim': max_sim,
        'min_sim': min_sim,
        'avg_sim': avg_sim,
        'closest_smiles': closest_smiles,
        'farest_smiles': farest_smiles
    }

def get_RDFsim(candidates, original):
    """
    Fingerprint similarity based on RDKFingerprint
    """
    smiles_list = [c['smiles'] for c in candidates]
    max_sim = -1
    avg_sim = 0
    min_sim = 1e5
    invalid_pred = 0
    mol_original = Chem.MolFromSmiles(original)
    fp_original = Chem.RDKFingerprint(mol_original)
    closest_smiles = ''
    farest_smiles = ''
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            invalid_pred += 1
            continue
        sim = DataStructs.FingerprintSimilarity(fp_original, Chem.RDKFingerprint(mol))
        if sim > max_sim:
            max_sim = sim
            closest_smiles = smi
        if sim < min_sim:
            min_sim = sim
            farest_smiles = smi
        avg_sim += sim
    if not avg_sim:
        return None
    avg_sim = avg_sim / (len(smiles_list) - invalid_pred)
    return {
        'max_sim': max_sim,
        'min_sim': min_sim,
        'avg_sim': avg_sim,
        'closest_smiles': closest_smiles,
        'farest_smiles': farest_smiles
    }

def get_weight_diff(candidates, original):
    smiles_list = [c['smiles'] for c in candidates]
    mol = Chem.MolFromSmiles(original)
    original_mw = Descriptors.ExactMolWt(mol)
    min_diff = 1e5
    closest_smiles = ''
    diff_list = []
    for smi in smiles_list:
        mw_diff = abs(Descriptors.ExactMolWt(Chem.MolFromSmiles(smi)) - original_mw)
        diff_list.append(mw_diff)
        if mw_diff < min_diff:
            min_diff = mw_diff
            closest_smiles = smi
    return {
        'min_dmw': min_diff,
        'avg_dmw': sum(diff_list) / len(diff_list),
        'closest_smiles': closest_smiles,
        'closest_mw': Descriptors.ExactMolWt(Chem.MolFromSmiles(closest_smiles))
    }
def get_formula_diff(candidates, original):

    smiles_list = [c['smiles'] for c in candidates]
    mol = Chem.MolFromSmiles(original)
    elements_chempy = get_elements_chempy(ELEMENTS.keys())
    elements_chempy_map = get_elements_chempy_map(elements_chempy)
    fo = rdMolDescriptors.CalcMolFormula(mol)
    fo_map = get_elements_chempy_umap(fo)
    original_fo = [0] * len(ELEMENTS)
    for chempy_idx, count in fo_map.items():
        if chempy_idx:  # 0 -> charge ['+', '-']
            original_fo[elements_chempy_map[chempy_idx]] = count
    diff_list = []
    min_diff = 1e5
    closest_smiles = ''
    for smi in smiles_list:
        try:
            mol = Chem.MolFromSmiles(smi)
            fo = rdMolDescriptors.CalcMolFormula(mol)
            fo_map = get_elements_chempy_umap(fo)
            fo = [0] * len(ELEMENTS)
            for chempy_idx, count in fo_map.items():
                if chempy_idx:  # 0 -> charge ['+', '-']
                    fo[elements_chempy_map[chempy_idx]] = count
            diff = sum([abs(d - od) for d, od in zip(fo, original_fo)])
            diff_list.append(diff)
            if diff < min_diff:
                min_diff = diff
                closest_smiles = smi

        except:
            continue

    return {
        'min_dmf': min_diff,
        'avg_dmf': sum(diff_list) / len(diff_list),
        'closest_smiles': closest_smiles,
        'closest_formula': rdMolDescriptors.CalcMolFormula(Chem.MolFromSmiles(closest_smiles))
    }

def cal_form_acc(correct, candidates):
    mol = Chem.MolFromSmiles(correct)
    correct_fo = rdMolDescriptors.CalcMolFormula(mol)
    cnt = 0
    for c in candidates:
        correct = False
        for o in c:
            mol = Chem.MolFromSmiles(o['smiles'])
            fo = rdMolDescriptors.CalcMolFormula(mol)
            if fo == correct_fo:
                correct = True
                break
        cnt += correct
    return {
        'hit formula topk': cnt > 0,
        'num of hit topk': cnt,
    }

def cal_ma_fa(statistic):
    tot_hit_topk = sum([v['topk'] > 0 for k, v in statistic.items()])
    tot = len(statistic)
    tot_spec = sum([v['topk'] + v['not hit topk'] for k, v in statistic.items()])
    tot_spec_hit_topk = sum([v['topk'] for k, v in statistic.items()])
    ma = tot_hit_topk / tot
    mas = tot_spec_hit_topk / tot_spec

    tot_form_hit_topk = sum([v['num formula hit topk'] > 0 for k, v in statistic.items()])
    tot_spec_form_hit_topk = sum([v['num formula hit topk'] for k, v in statistic.items()])
    fa = tot_form_hit_topk / tot
    fas = tot_spec_form_hit_topk / tot_spec
    return {
        'tot mols': tot,
        'tot mols hit topk (at least one)': tot_hit_topk,
        'ma': ma,
        'mas': mas,
        'tot spec': tot_spec,
        'tot spec hit topk': tot_spec_hit_topk,
        'tot formula hit topk (at least one)': tot_form_hit_topk,
        'tot formula spec hit topk': tot_spec_form_hit_topk,
        'fa': fa,
        'fas': fas
    }

def cal_global_statistic(statistic):
    attrs = {
        k: 0 for k, v in statistic[list(statistic.keys())[0]].items()
    }
    for smi, meta in statistic.items():
        for k, v in meta.items():
            attrs[k] += v
    return {
        k: v / len(statistic) for k, v in attrs.items()
    }

def cal_confidence_score(cands, use_string=False):
    tots = [len(c) for c in cands]
    is_pfas = [sum([int(c_o['is pfas']) for c_o in c]) for c in cands]
    if use_string:
        return [f'{(i_p / t * 100.0):2f}%' for t, i_p in zip(tots, is_pfas)]
    else:
        return [round(i_p / t, 2) for t, i_p in zip(tots, is_pfas)]

def eval_statistic(pkl_path):

    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)
    smiles_list = data.keys()
    statistic = {
        smi: {
            'avg_loss': 0,
            'min_loss': 0,
            'max_loss': 0,
            'min_sim': 0,
            'max_sim': 0,
            'avg_sim': 0,
            'max_mcs_coef': 0,
            'min_mcs_coef': 0,
            'avg_mcs_coef': 0,
            'max_mcs_tan': 0,
            'min_mcs_tan': 0,
            'avg_mcs_tan': 0,
            'max_mcs_ratio': 0,
            'min_mcs_ratio': 0,
            'avg_mcs_tan': 0,     
            'topk': data[smi]['hit topk'],
            'not hit topk': len(data[smi]['candidates']) - data[smi]['hit topk'],
            'hit topk': data[smi]['hit topk'] > 0,
        } for smi in smiles_list
    }
    for smi in smiles_list:
        max_mcs_ratio = [o['max_mcs_ratio'] for o in data[smi]['mcs']]
        max_mcs_ratio = sum(max_mcs_ratio) / len(max_mcs_ratio)
        max_mcs_coef = [o['max_mcs_coef'] for o in data[smi]['mcs']]
        max_mcs_coef = sum(max_mcs_coef) / len(max_mcs_coef)
        max_mcs_tan = [o['max_mcs_tan'] for o in data[smi]['mcs']]
        max_mcs_tan = sum(max_mcs_tan) / len(max_mcs_tan)

        min_mcs_ratio = [o['min_mcs_ratio'] for o in data[smi]['mcs']]
        min_mcs_ratio = sum(min_mcs_ratio) / len(min_mcs_ratio)
        min_mcs_coef = [o['min_mcs_coef'] for o in data[smi]['mcs']]
        min_mcs_coef = sum(min_mcs_coef) / len(min_mcs_coef)
        min_mcs_tan = [o['min_mcs_tan'] for o in data[smi]['mcs']]
        min_mcs_tan = sum(min_mcs_tan) / len(min_mcs_tan)

        avg_mcs_ratio = [o['avg_mcs_ratio'] for o in data[smi]['mcs']]
        avg_mcs_ratio = sum(avg_mcs_ratio) / len(avg_mcs_ratio)
        avg_mcs_coef = [o['avg_mcs_coef'] for o in data[smi]['mcs']]
        avg_mcs_coef = sum(avg_mcs_coef) / len(avg_mcs_coef)
        avg_mcs_tan = [o['avg_mcs_tan'] for o in data[smi]['mcs']]
        avg_mcs_tan = sum(avg_mcs_tan) / len(avg_mcs_tan)

        max_sim = [o['max_sim'] for o in data[smi]['similarity']]
        max_sim = sum(max_sim) / len(max_sim)
        min_sim = [o['min_sim'] for o in data[smi]['similarity']]
        min_sim = sum(min_sim) / len(min_sim)
        avg_sim = [o['avg_sim'] for o in data[smi]['similarity']]
        avg_sim = sum(avg_sim) / len(avg_sim)

        num_spec = len(data[smi]['candidates'])
        num_candidates = len(data[smi]['candidates'][0])
        avg_loss = 0
        max_loss = 0
        min_loss = 0
        for spec in data[smi]['candidates']:
            tmax = 0
            tmin = 10000
            for o in spec:
                loss = abs(o['loss'])
                avg_loss += loss
                tmax = max(tmax, loss)
                tmin = min(tmin, loss)
            max_loss += tmax
            min_loss += tmin
        if avg_loss > 0:
            avg_loss = avg_loss / (num_spec * num_candidates)
        if min_loss > 0:
            min_loss = min_loss / num_spec
        if max_loss > 0:
            max_loss = max_loss / num_spec

        cal_form_results = cal_form_acc(smi, data[smi]['candidates'])
        correct_formula_hit_topk = cal_form_results['hit formula topk']
        num_formula_hit_topk = cal_form_results['num of hit topk']
        statistic[smi]['avg_loss'] = round(avg_loss, 2)
        statistic[smi]['min_loss'] = round(min_loss, 2)
        statistic[smi]['max_loss'] = round(max_loss, 2)
        statistic[smi]['max_mcs_ratio'] = round(max_mcs_ratio, 2)
        statistic[smi]['min_mcs_ratio'] = round(min_mcs_ratio, 2)
        statistic[smi]['avg_mcs_ratio'] = round(avg_mcs_ratio, 2)
        statistic[smi]['max_mcs_tan'] = round(max_mcs_tan, 2)
        statistic[smi]['min_mcs_tan'] = round(min_mcs_tan, 2)
        statistic[smi]['avg_mcs_tan'] = round(avg_mcs_tan, 2)
        statistic[smi]['max_mcs_coef'] = round(max_mcs_coef, 2)
        statistic[smi]['min_mcs_coef'] = round(min_mcs_coef, 2)
        statistic[smi]['avg_mcs_coef'] = round(avg_mcs_coef, 2)
        statistic[smi]['max_sim'] = round(max_sim, 2)
        statistic[smi]['min_sim'] = round(min_sim, 2)
        statistic[smi]['avg_sim'] = round(avg_sim, 2)
        statistic[smi]['formula hit topk'] = correct_formula_hit_topk
        statistic[smi]['num formula hit topk'] = num_formula_hit_topk

    fa_ma = cal_ma_fa(statistic)
    global_statistic = cal_global_statistic(statistic)
    global_statistic.update(fa_ma)

    for smi in smiles_list:
        statistic[smi]['pfas_confidence_level'] = (
            cal_confidence_score(data[smi]['candidates'], use_string=True) 
            if data[smi]['candidates'][0][0].get('is pfas') is not None else None
        )
        statistic[smi]['pfas_confidence_level_float'] = (
            cal_confidence_score(data[smi]['candidates'], use_string=False) 
            if data[smi]['candidates'][0][0].get('is pfas') is not None else None  
        )
        statistic[smi]['avg_pfas_confidence_level_float'] = round(
            sum(statistic[smi]['pfas_confidence_level_float']) / len(statistic[smi]['pfas_confidence_level_float']),
            2
        )
        statistic[smi]['avg_pfas_confidence_level'] = f"{(statistic[smi]['avg_pfas_confidence_level_float'] * 100.0):.2f}%"
    return statistic, global_statistic