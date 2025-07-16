
from pyteomics import mgf
from tqdm import tqdm
from DeePFAS.utils.smiles_process import remove_salt_stereo
from rdkit import Chem
def check_q(pth, adduct):

    m_h_cnt = 0
    q_map = {
        '1': 0,
        '2': 0,
        '3': 0
    }
    with open(pth, 'r') as f:
        while True:
            line = f.readline()
            if not line or line is None:
                break
            line = line.replace('\n', '').strip()
            if line == 'BEGIN IONS':
                for i in range(9):
                    line = f.readline().replace('\n', '').strip()
                if line.split(' ')[-1] == adduct:
                    m_h_cnt += 1
                    for i in range(8):
                        line = f.readline().replace('\n', '').strip()
                    q_map[line[-1]] += 1
        print(f'total M-H spectra: {m_h_cnt}\n')
        print(q_map)

def write_mgf(pth, out_pth):
    out_f = open(out_pth, 'w')
    title = 0
    with open(pth, 'r') as f:
        while True:
            line = f.readline()
            if not line or line is None:
                break
            tmp = line.replace('\n', '').strip()
            out_f.write(line)
            if tmp == 'BEGIN IONS':
                out_f.write(f'TITLE={title}\n')
                title += 1
    out_f.close()
def cal_info(pth, adduct):
    q_map = {
        '1': 0,
        '2': 0,
        '3': 0
    }
    mol_set = set()
    ce_map = {

    }
    with mgf.read(pth) as f:
        for o in tqdm(f):
            name_splits = o['params']['name'].split(' ')
            # if name_splits[-1] != adduct or name_splits[-2][:3] != 'Col':
            #     continue
            # ce = name_splits[-2]
            # if ce_map.get(ce) is None:
            #     ce_map[ce] = 1
            # else:
            #     ce_map[ce] += 1
            if name_splits[-1] != adduct:
                continue
            if q_map.get(o['params']['libraryquality']) is None:
                q_map[o['params']['libraryquality']] = 1
            else:
                q_map[o['params']['libraryquality']] += 1

            try:
                smiles = remove_salt_stereo(o['params']['smiles'], isomeric=False, canonical=True)
            except:
                continue
            mol_set.add(smiles)
    print(f'num mols: {len(mol_set)}')
    print(ce_map)
    print(q_map)

def check_resource(pth, adduct):
    q_map = {}
    
    with mgf.read(pth) as f:
        for o in tqdm(f):
            name_splits = o['params']['name'].split(' ')
            if name_splits[-1] != adduct:
                continue
            # source_instrument = o['params']['source_instrument']

            if q_map.get(o['params']['source_instrument']) is None:
                q_map[o['params']['source_instrument']] = 1
            else:
                q_map[o['params']['source_instrument']] += 1

    print(q_map)

if __name__ == '__main__':
    # pth = '../DATASET/GNPS-LIBRARY.mgf'
    pth = '../DATASET/ALL_GNPS_NO_PROPOGATED.mgf'
    out_pth = '../DATASET/ALL_GNPS_NO_PROPOGATED_with_title.mgf'
    adduct = 'M-H'
    # check_q(pth, adduct)
    # write_mgf(pth, out_pth)
    # cal_info(out_pth, adduct)
    check_resource(out_pth, adduct)