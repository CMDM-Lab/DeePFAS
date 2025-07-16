from pyteomics import mgf
from tqdm import tqdm
import copy

match_smiles_data = []
match_smiles_map = {}
match_data = []
match_map = {}
remain_data = []
with open('../DATASET/WWTP3_name.txt') as f:
    while True:
        line = f.readline().replace('\n', '').strip()
        if line is None or not line:
            break
        name, smiles, pepmass, pubchemcid, confi = line.split('|')

        metadata = {
                'name': name,
                'smiles': smiles,
        }
        if smiles.find('F') != -1 and pubchemcid.find('unknown') == -1:
            match_smiles_map[float(pepmass)] = metadata
        else:
            match_map[float(pepmass)] = metadata

with mgf.read('../DATASET/new_cali_T3_ms2.mgf') as f:
    for o in tqdm(f):
        copy_o = copy.deepcopy(o)
        copy_o['params']['original title'] = copy_o['params']['title']
        hit = False
        for k, v in match_smiles_map.items():
            if abs(k - o['params']['pepmass'][0]) / k * 1e6 <= 5.0:
                copy_o['params']['canonicalsmiles'] = v['smiles']
                copy_o['params']['randomizedsmiles'] = v['smiles']
                copy_o['params']['name'] = v['name']
                copy_o['params']['title'] = len(match_smiles_data)
                hit = True
                match_smiles_data.append(copy_o)
                break
        if not hit:
            for k, v in match_map.items():
                if abs(k - o['params']['pepmass'][0]) / k * 1e6 <= 5.0:
                    copy_o['params']['canonicalsmiles'] = v['smiles']
                    copy_o['params']['name'] = v['name']
                    copy_o['params']['title'] = len(match_data)
                    match_data.append(copy_o)
                    hit = True
                    break
        if not hit:
            copy_o['params']['title'] = len(remain_data)
            remain_data.append(copy_o)

print(len(match_smiles_data))
print(len(match_data))
print(len(remain_data))

mgf.write(match_smiles_data, '../DATASET/wwtp3_hit_smiles_std.mgf', file_mode='w', write_charges=False)
mgf.write(match_data, '../DATASET/wwtp3_hit_std.mgf', file_mode='w', write_charges=False)
mgf.write(remain_data, '../DATASET/wwtp3_remain.mgf', file_mode='w', write_charges=False)
