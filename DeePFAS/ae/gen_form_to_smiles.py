from tqdm import tqdm
import h5py

from rdkit.Chem import AllChem, Descriptors, RDConfig, rdMolDescriptors
from rdkit import Chem
import numpy as np
import pickle
def gen_form_to_smiles(in_pth, out_pth, compression_opts):


    form_to_smiles = {}
    with open(in_pth, 'r') as f:
        while True:
            line = f.readline().replace('\n', '').strip()
            if line is None or not line:
                break
            mol = Chem.MolFromSmiles(line)
            fo = rdMolDescriptors.CalcMolFormula(mol)
            form_to_smiles.setdefault(fo, [])
            form_to_smiles[fo].append(line)
    print(f'num of keys: {len(form_to_smiles)}')
    all_smiles = []
    index_map = {}

    start_idx = 0
    for mf, smiles_list in tqdm(form_to_smiles.items()):
        index_map[mf] = (start_idx, start_idx + len(smiles_list))
        all_smiles.extend(smiles_list)
        start_idx += len(smiles_list)

    with h5py.File(out_pth, 'w') as h5f:
        max_len = max(len(s) for s in all_smiles)
        dt = h5py.string_dtype(length=max_len)
        dataset = h5f.create_dataset(
            "SMILES",
            data=np.array(all_smiles, dtype=dt),
            compression="gzip",
            compression_opts=compression_opts,
            
        )

        mf_grp = h5f.create_group("MF_Index")
        for mf, (start, end) in index_map.items():
            mf_grp.attrs[mf] = (start, end)
def gen_form_to_smiles_pkl(in_pth, out_pth):
    form_to_smiles = {}
    with open(in_pth, 'r') as f:
        while True:
            line = f.readline().replace('\n', '').strip()
            if line is None or not line:
                break
            mol = Chem.MolFromSmiles(line)
            fo = rdMolDescriptors.CalcMolFormula(mol)
            form_to_smiles.setdefault(fo, [])
            form_to_smiles[fo].append(line)
    print(f'num of keys: {len(form_to_smiles)}')
    with open(out_pth, 'wb') as f:
        pickle.dump(form_to_smiles, f)
if __name__ == '__main__':
    in_pth = '../DATASET/mol_database.tsv'
    out_pth = '../DATASET/form_to_smiles.pkl'
    compression_opts = 9
    # gen_form_to_smiles(in_pth, out_pth, compression_opts)
    gen_form_to_smiles_pkl(in_pth, out_pth)
