from ae.models.mol_extracter import MolExtracter
from ae.utils.data_modules.mol_modules import MolProcess, MultiprocessModule
import pytorch_lightning as pl
from ae.config.models import MolExtractorConfig
from ae.utils.smiles_process import batch_buffer_collate_fn, batch_to_device_collate_fn, idx_to_smiles
from functools import partial
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger
from pathlib import Path
from DeePFAS.config.models import DeePFASConfig
from DeePFAS.models.model.deepfas import DeePFAS
import torch
import wandb
import argparse
import h5py
import pickle
import multiprocessing
import numpy as np
from tqdm import tqdm
torch.set_printoptions(profile='full')
torch.set_float32_matmul_precision('high')

def arg_parser():
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument('--ae_config_pth', type=str, required=True)
    parser.add_argument('--deepfas_config_pth', type=str, required=True)
    parser.add_argument('--latent_space_out_pth', type=str, required=True)
    parser.add_argument('--chunk_size', type=int, required=True)
    parser.add_argument('--compression_level', type=int, required=True)
    return parser.parse_args()


def search_latent_space(model, dataloader, dataset_size, chunk_size, compression_level, out_pth):

    mol_map = {}
    dataiter = iter(dataloader)

    with h5py.File(out_pth, 'w') as h5f:
        emb_dataset = h5f.create_dataset(
            "chemical_emb",
            shape=(dataset_size, 512),
            compression="gzip",
            compression_opts=compression_level,
        )
        smiles_dataset = h5f.create_dataset(
            "CANONICALSMILES",
            shape=(dataset_size, ),
            dtype=h5py.string_dtype(length=130),
            compression="gzip",
            compression_opts=compression_level,
        )
        smiles_chunk = []
        chunk_id = 0
        cnt = 0
        for _ in tqdm(range(len(dataloader))):
            batch = next(dataiter)
            randomized_smiles, canonical_smiles, properties = batch
            _, _, z, _ = model.forward_encoder(randomized_smiles)
            z = z.cpu().detach().numpy()
            # z = z.detach()
            for i, z_i in enumerate(z):
                c_smiles = idx_to_smiles(canonical_smiles[i][1:-1])
                if mol_map.get(c_smiles) is None:
                    mol_map[c_smiles] = 1
                    smiles_chunk.append((c_smiles, z_i))
                    if (cnt + 1) % chunk_size == 0:
                        emb_dataset[chunk_id * chunk_size: (chunk_id + 1) * chunk_size] = [e for s, e in smiles_chunk]
                        smiles_dataset[chunk_id * chunk_size: (chunk_id + 1) * chunk_size] = [s for s, e in smiles_chunk]
                        del smiles_chunk
                        smiles_chunk = []
                        chunk_id += 1
                    cnt += 1
        if len(smiles_chunk) > 0:
            emb_dataset[chunk_id * chunk_size: chunk_id * chunk_size + len(smiles_chunk)] = [e for s, e in smiles_chunk]
            smiles_dataset[chunk_id * chunk_size: chunk_id * chunk_size + len(smiles_chunk)] = [s for s, e in smiles_chunk]
        print(f'Total write data: {cnt}')

def gen_latent_space(args: str = None):

    multiprocessing.set_start_method('fork', force=True)
    if args is None:
        args = arg_parser()
    device_name = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device_name)

    ae_args = MolExtractorConfig.from_json_path(args.ae_config_pth)

    deepfas_args = DeePFASConfig.from_json_path(
        args.deepfas_config_pth,
    )
    out_pth = args.latent_space_out_pth
    chunk_size = args.chunk_size
    compression_level = args.compression_level
    seed_everything(ae_args.hparams.seed)

    mol_processor = MolProcess(
        max_len=ae_args.hparams.max_len,
        use_properties=ae_args.hparams.use_properties,
        contrastive_learning_decoys=None,
        num_decoys=ae_args.hparams.num_decoys,
        randomized=ae_args.hparams.randomized,
    )

    if Path(ae_args.path.dataset_path).suffix == '.hdf5':
        with h5py.File(ae_args.path.dataset_path, 'r') as f:
            if ae_args.hparams.debug:
                dataset = f['CANONICALSMILES'][:1000]
            else:
                dataset = f['CANONICALSMILES'][:]
            dataset = np.array(dataset)
    else:
        dataset = []
        with open(ae_args.path.dataset_path, 'rb') as f:
            while True:
                line = f.readline().replace(b'\n', b'').strip()
                if line is None or not line:
                    break
                dataset.append(line)

    dataset_size = len(dataset)
    data_modules = MultiprocessModule(
        hparams=ae_args.hparams,
        dataset=dataset,
        mol_processor=mol_processor,
        batch_collate_fn=partial(
            batch_buffer_collate_fn,
            use_properties=ae_args.hparams.use_properties,
        ),
        to_device_collate=partial(batch_to_device_collate_fn, device=device),
        retrieval=True
    )

    deepfas_model_pth = deepfas_args.path.save_model_path
    ae_model_pth = ae_args.path.save_model_path
    if Path(ae_model_pth).suffix == '.pt':
        model = MolExtracter.load_translator(ae_model_pth, ae_args, device)
    elif Path(deepfas_model_pth).suffix == '.ckpt':
        model = DeePFAS.load_from_checkpoint(
            checkpoint_path=deepfas_model_pth,
            map_location=device,
        ).ae

    retrieval_dataloaders = data_modules.test_dataloader()
    search_latent_space(model, retrieval_dataloaders, dataset_size, chunk_size, compression_level, out_pth)
    retrieval_dataloaders.kill_process()

if __name__ == '__main__':
    gen_latent_space()
