import time
import h5py

from DeePFAS.models.metrics.eval import epoch_time


chunk_size = 400000
with h5py.File('../DATASET/PubChem_chemical_embbeddings_chunk_size_uncompressed.hdf5', 'r') as f:
    start_time = time.time()
    a = f['CANONICALSMILES'][:chunk_size]
    b = f['chemical_emb'][:chunk_size]
    del a
    del b
    end_time = time.time()
    mins, secs = epoch_time(start_time, end_time)
    print(f'Load hdf5 Time: {mins}m {secs}s')

with h5py.File('../DATASET/PubChem_chemical_embbeddings_chunk_size_1600000.hdf5', 'r') as f:
    start_time = time.time()
    a = f['CANONICALSMILES'][:chunk_size * 2]
    b = f['chemical_emb'][:chunk_size * 2]
    del a
    del b
    end_time = time.time()
    mins, secs = epoch_time(start_time, end_time)
    print(f'Load hdf5 Time: {mins}m {secs}s')