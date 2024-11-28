import copy
import pickle
import random

from pyteomics import mgf
from tqdm import tqdm


def split_dataset(path, out_train_path, out_test_path, test_num):
    with mgf.read(path) as f:
        mol_map = {}
        for o in tqdm(f):
            smiles = o['params']['canonicalsmiles']
            if mol_map.get(smiles) is None:
                mol_map[smiles] = []
            mol_map[smiles].append(o)
        train_data, test_data = [], []
        cnt = 0
        mol_map = [(k, v) for k, v in mol_map.items()]
        for k, v in mol_map[:-test_num]:
            for o in v:
                c = copy.deepcopy(o)
                c['params']['title'] = cnt
                cnt += 1
                train_data.append(c)
        cnt = 0
        for k, v in mol_map[-test_num:]:
            for o in v:
                c = copy.deepcopy(o)
                c['params']['title'] = cnt
                cnt += 1
                test_data.append(c)
        mgf.write(train_data, out_train_path, file_mode='w', write_charges=False)
        mgf.write(train_data, out_test_path, file_mode='w', write_charges=False)

def split_dataset_2():
    with open('./embeddings/randomizedsmiles.pkl', 'rb') as f:
        valid_mol = pickle.load(f)
    with mgf.read('../DATASET/all_gnps_nist20.mgf') as f:
        train_data, test_data = [], []
        mol_map = {}
        for o in tqdm(f):
            smiles = o['params']['canonicalsmiles']
            if valid_mol.get(smiles) is None:
                continue
            if mol_map.get(smiles) is None:
                mol_map[smiles] = []
            mol_map[smiles].append(o)
        mol_map = [(k, v) for k, v in mol_map.items()]
        random.shuffle(mol_map)
        train_data = [o for k, v in mol_map[:-500] for o in v]
        test_data = [o for k, v in mol_map[-500:] for o in v]
        with mgf.read('../DATASET/150_filted.mgf') as f2:
            for o in tqdm(f2):
                smiles = o['params']['canonicalsmiles']
                if smiles in ['CN(CCO)S(=O)(=O)C(F)(F)C(F)(F)C(F)(F)C(F)(F)F', 'O=S(=O)(N(CCO)CCO)C(F)(F)C(F)(F)C(F)(F)C(F)(F)F']:
                    test_data.append(o)
                else:
                    train_data.append(o)
        random.shuffle(train_data)
        random.shuffle(test_data)
        for i, o in enumerate(train_data):
            train_data[i]['params']['title'] = i
        for i, o in enumerate(test_data):
            test_data[i]['params']['title'] = i
        mgf.write(train_data, '../DATASET/all_train.mgf', file_mode='w', write_charges=False)
        mgf.write(test_data, '../DATASET/all_test.mgf', file_mode='w', write_charges=False)

# def split_MoNA(path):
#     with open(path, 'r') as f:

def pickle_smiles(in_path, out_path):
    offsets = [0]
    with open(in_path, 'r', encoding='utf-8') as fp:
        while fp.readline() != '':
            offsets.append(fp.tell())
    offsets.pop()
    with open(out_path, 'wb') as f:
        pickle.dump(offsets, f)
    print(len(offsets))

