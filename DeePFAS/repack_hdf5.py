import h5py
from tqdm import tqdm



def repack_hdf5(in_hdf5, out_hdf5, chunk_size):

    f = h5py.File(in_hdf5, 'r')
    dataset_size = len(f['CANONICALSMILES'])
    with h5py.File(out_hdf5, 'w') as h5f:
        emb_dataset = h5f.create_dataset(
            "chemical_emb",
            shape=(dataset_size, 512),
            chunks=(chunk_size, 512),
            # compression="gzip",
            # compression_opts=9,
        )
        smiles_dataset = h5f.create_dataset(
            "CANONICALSMILES",
            shape=(dataset_size, ),
            dtype=h5py.string_dtype(length=130),
            chunks=(chunk_size, ),
            # compression="gzip",
            # compression_opts=9,
        )
        chunk_size = chunk_size
        tot_chunk = dataset_size // chunk_size + (dataset_size % chunk_size > 0)
        for i in tqdm(range(tot_chunk)):
            emb_dataset[i * chunk_size: min(dataset_size, (i + 1) * chunk_size)] = f['chemical_emb'][i * chunk_size: min(dataset_size, (i + 1) * chunk_size)]
            smiles_dataset[i * chunk_size: min(dataset_size, (i + 1) * chunk_size)] = f['CANONICALSMILES'][i * chunk_size: min(dataset_size, (i + 1) * chunk_size)]

if __name__ == '__main__':
    chunk_size = 400000
    out_hdf5=  '../DATASET/PubChem_chemical_embbeddings_chunk_size_uncompressed.hdf5'
    in_hdf5 = '../DATASET/PubChem_chemical_embbeddings_chunk_size_1600000.hdf5'
    repack_hdf5(in_hdf5=in_hdf5, out_hdf5=out_hdf5, chunk_size=chunk_size)