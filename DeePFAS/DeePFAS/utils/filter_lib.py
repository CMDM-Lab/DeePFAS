from ae.utils.smiles_process import *
from tqdm import tqdm
from rdkit import Chem
from rdkit.Chem import Descriptors
def filter_(data_path, out_path, thres_hold, file_len):
    with open(data_path, 'r') as f:
        out_f = open(out_path, 'a')
        repl_map = {}
        for _ in tqdm(range(file_len)):
            line = f.readline()
            if line is None or line:
                break
            o_smiles = line.split('\t')[-1].replace('\n', '').strip()
            smiles = remove_salt_stereo(o_smiles, False, True)
            if smiles is None:
                print(o_smiles)
                continue
            mol = Chem.MolFromSmiles(smiles)
            mw = Descriptors.ExactMolWt(mol)
            if mw > 1000 or elements_filter(mol, ELEMENTS):
                continue
            if smiles and len(tokens_from_smiles([smiles])) <= thres_hold:
                if not repl_map.get(smiles):
                    repl_map[smiles] = 1
                    out_f.write(f'{smiles}\n')
    out_f.close()
    print(len(repl_map))

if __name__ == '__main__':
    data_path = '../DATASET/CID-SMILES'
    out_path = '../DATASET/PubChem_SMILES.tsv'
    thres_hold = 110
    file_len = 170855227

