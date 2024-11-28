# DeePFAS: Deep Learning-Enabled Rapid Annotation of PFAS: Enhancing Non-Targeted Screening through Spectral Encoding and Latent Space Analysis

This repository provides implementations and code examples for DeePFAS: Deep Learning-Enabled Rapid Annotation of PFAS: Enhancing Non-Targeted Screening through Spectral Encoding and Latent Space Analysis. DeePFAS projects raw MS/MS data into the latent space of chemical structures for PFAS identification, facilitating the inference of structurally similar compounds by comparing spectra to multiple candidate molecules within this latent chemical space.

## Quickstart


### Option1: import M2S like python package

```python


from DeePFAS.inference import inference

ignore_MzRange = 'not ignore' # Whether to ignore the input spectrum mz peaks range (< 1000 Da) limitation ('not ignore', 'ignore')
ignore_CE = 'not ignore' # Whether to ignore the input spectrum collision energy (10 ~ 50 eV) limitation ('not ignore', 'ignore')

# Eval mode (the compounds corresponding to the spectra are known, and Canonical SMILES must be provided in the spectral files for result evaluation)
results = inference(dataset_path='./smiles_dataset.tsv',
                    data_id_path='./smiles_dataset.id',
                    data_file='./example/testdata.mgf',
                    mode='eval',
                    topk=20,
                    out_dir='./results',
                    ignore_MzRange=ignore_MzRange,
                    ignore_CE=ignore_CE)

# Inference mode (unknown compounds, Canonical SMILES is not necessary)
results = inference(dataset_path='./smiles_dataset.tsv',
                    data_id_path='./smiles_dataset.id',
                    data_file='./examplt/testdata.mgf',
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

### Option2: run script

```shell

python3 ./M2S/script.py \
    --dataset_path './randomizedsmiles.tsv' \
    --dataset_id_path './randomizedsmiles.id' \
    --topk 20 \
    --data_file './example/testdata.mgf' \
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
            "loss": 11.045734405517578, 
            "smiles": "O=S(=O)([O-])C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)F",
            "mw": 348.93979885991,
            "mw_diff": 5.946375631272986
        },
        {
            "loss": 9.646349906921387,
            "smiles": "O=S(=O)(O)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)F",
            "mw": 349.94707531200004,
            "mw_diff": 6.953652083363011
        },
        {
            "loss": 9.632779121398926,
            "smiles": "O=C(O)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)F",
            "mw": 363.97689613200004,
            "mw_diff": 20.98347290336301
        },
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