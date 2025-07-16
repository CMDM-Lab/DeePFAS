from tqdm import tqdm
from pyteomics import mgf
import copy
import random
def oversampling(file, out_file, max_num=None):
    mol_map = {}
    data = []
    with mgf.read(file) as f:
        for o in tqdm(f):
            smiles = o['params']['canonicalsmiles']
            if mol_map.get(smiles) is None:
                mol_map[smiles] = []
            mol_map[smiles].append(o)
    max_num = min(max_num if max_num else 1e10, max([len(v) for k, v in mol_map.items()]))

    for _, v in mol_map.items():
        tot = int(max_num)
        while tot > len(v):
            data.extend(v)
            tot -= len(v)
        if tot > 0:
            data.extend(random.sample(v, k=tot))

    random.shuffle(data)
    for i, o in enumerate(data):
        copy_o = copy.deepcopy(o)
        copy_o['params']['title'] = i
        data[i] = copy_o
    print(f'Total num of data: {len(data)}')
    mgf.write(data, out_file, file_mode='w', write_charges=False)

if __name__ == '__main__':
    file = '../DATASET/all_neg_hydrogen_nist_150_train.mgf'
    out_file = '../DATASET/all_neg_hydrogen_nist_150_train_over.mgf'
    random.seed(57)
    max_num = 50
    oversampling(file, out_file, max_num)
