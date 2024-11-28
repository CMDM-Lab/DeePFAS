import os
import pickle
import sys

from IPython.display import SVG
from pyteomics import mgf
from rdkit import Chem, DataStructs
from rdkit.Chem import (AllChem, Descriptors, Draw, RDConfig, rdDepictor,
                        rdFMCS, rdMolDescriptors)
from rdkit.Chem.Draw import IPythonConsole, rdMolDraw2D

from .utils.smiles_process import (ELEMENTS, get_elements_chempy,
                                   get_elements_chempy_map,
                                   get_elements_chempy_umap,
                                   remove_salt_stereo)

sys.path.append(os.path.join(RDConfig.RDContribDir, 'SA_Score'))
import sascorer
from tqdm import tqdm


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

def eval_statistic(pkl_path):

    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)
    smiles_list = data.keys()
    statistic = {
        smi: {
            'avg_loss': 0,
            'min_loss': 0,
            'max_loss': 0,
            'min_dmf': 0,
            'max_dmf': 0,
            'avg_dmf': 0,
            'min_dmw': 0,
            'max_dmw': 0,
            'avg_dmw': 0,
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
            'top20': data[smi]['hit top20']
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

        min_dmf = [o['min_dmf'] for o in data[smi]['dmf']]
        min_dmf = sum(min_dmf) / len(min_dmf)
        avg_dmf = [o['avg_dmf'] for o in data[smi]['dmf']]
        avg_dmf = sum(avg_dmf) / len(avg_dmf)

        min_dmw = [o['min_dmw'] for o in data[smi]['dmw']]
        min_dmw = sum(min_dmw) / len(min_dmw)
        avg_dmw = [o['avg_dmw'] for o in data[smi]['dmw']]
        avg_dmw = sum(avg_dmw) / len(avg_dmw)

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
        statistic[smi]['min_dmf'] = round(min_dmf, 2)
        statistic[smi]['avg_dmf'] = round(avg_dmf, 2)
        statistic[smi]['min_dmw'] = round(min_dmw, 2)
        statistic[smi]['avg_dmw'] = round(avg_dmw, 2)
    return statistic
