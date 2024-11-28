# DeePFAS

## Overview

From spectrum predict molecular structure (SMILES)

## Usage

### Option1: import M2S like python package

Examples:

```python


from M2S.inference import inference

ignore_MzRange = 'not ignore' # Whether to ignore the input spectrum mz peaks range (50 ~ 1000 Da) limitation ('not ignore', 'ignore')
ignore_CE = 'not ignore' # Whether to ignore the input spectrum collision energy (10 ~ 50 eV) limitation ('not ignore', 'ignore')

# Eval mode (the compounds corresponding to the spectra are known, and Canonical SMILES must be provided in the spectral files for result evaluation)
results = inference(dataset_path='./randomizedsmiles.tsv',
                    data_id_path='./randomizedsmiles.id',
                    data_file='testdata.mgf',
                    mode='eval',
                    topk=20,
                    out_dir='./results',
                    ignore_MzRange=ignore_MzRange,
                    ignore_CE=ignore_CE)

# Inference mode (unknown compounds, no need to provide Canonical SMILES)
results = inference(dataset_path='./randomizedsmiles.tsv',
                    data_id_path='./randomizedsmiles.id',
                    data_file='testdata.mgf',
                    mode='inference',
                    topk=20,
                    out_dir='./results',
                    ignore_MzRange=ignore_MzRange,
                    ignore_CE=ignore_CE)
print(results)

# statistic.pkl store results (python map)
# statistic.json store json format results
import pickle
with open('./results/statistic.pkl', 'rb') as f:
    statistic = pickle.load(f)
print(statistic)

```

### Option2: script

Examples:

```shell

python3 ./M2S/script.py \
    --dataset_path './randomizedsmiles.tsv' \
    --dataset_id_path './randomizedsmiles.id' \
    --topk 20 \
    --data_file 'testdata.mgf' \
    --mode inference \
    --out_dir './results' \
    --ignore_MzRange 'not ignore' \
    --ignore_CE 'not ignore'
```

### Example of generateing candidates dataset

The example shows how to generate pickle file used in obtaining position of each SMILES in customized candidate dataset.
See files in directory `dataset`.

```python

def pickle_smiles(in_path, out_path):
    offsets = [0]
    with open(in_path, 'r', encoding='utf-8') as fp:
        while fp.readline() != '':
            offsets.append(fp.tell())
    offsets.pop()
    with open(out_path, 'wb') as f:
        pickle.dump(offsets, f)

pickle_smiles('./randomizedsmiles.tsv', './randomizedsmiles.id')

```
### Example of statistic.json

This is an example of top3 candidates.
Title in spectra data is key of .json file.

```json

{
    "0": [
        {
            "loss": -20.14383316040039,
            "smiles": "CC(NC(=O)C(N)Cc1ccc(O)cc1)C(=O)O",
            "mw": 252.111006992,
            "mw_diff": 4.982934008000029
        },
        {
            "loss": -20.01431655883789,
            "smiles": "NC(Cc1ccccc1)C(=O)NC(CO)C(=O)O",
            "mw": 252.111006992,
            "mw_diff": 4.982934008000029
        },
        {
            "loss": -20.248918533325195,
            "smiles": "Nc1ccc(S(=O)(=O)Nc2ccccn2)cc1",
            "mw": 249.057197592,
            "mw_diff": 8.036743408000035
        }
    ]
}
```

### Construct input spectra data (.mgf)

Example:

```python

from pyteomics import mgf
import numpy as np
data = []

intensity = [0.1, 1.0, 0.3, 0.4]
m_z = [11.1, 23.23, 111.44, 55.2]
spectra = {
    'params': {
        # identifier of spectra in .mgf file (necessary)
        'title': 0,
        # ms level (necessary)
        'mslevel': 2,
        # precursor m/z (necessary)
        'pepmass': 562.957580566406,
        # adduct type (necessary)
        'precursor_type': '[M-H]-',
        # In eval mode, canonicalsmiless is necessary (unnecessary)
        'canonicalsmiles': 'O=C(O)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)F',
        # collision energy (necessary)
        # absolute collision energy (ACE) format: 'collision_energy': 12
        # normalized collision energy (NCE) format: 'collision_energy': 'NCE=37.5%'
        'collision_energy': 'NCE=37.5%'
    },
    # m/z array (necessary)
    'm/z array': np.array(intensity), 
    # intensity array (necessary)
    'intensity array': np.array(m_z)
}

data.append(spectra)
mgf.write(data, 'spectra.mgf', file_mode='w', write_charges=False)

```