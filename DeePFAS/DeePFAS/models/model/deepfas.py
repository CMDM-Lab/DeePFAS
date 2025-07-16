import pytorch_lightning as pl

from DeePFAS.models.model.attentionConv2D import AttentionConv2D
from ae.models.mol_extracter import MolExtracter
from ae.config.models import MolExtractorConfig
from DeePFAS.config.models import DeePFASConfig
import torch
import torch.nn.functional as F
from ae.utils.smiles_process import idx_to_smiles
import pandas as pd
from pathlib import Path
from DeePFAS.models.metrics.copy_inference_topk import inference_topk, search_topk
from DeePFAS.models.metrics.eval import get_RDFsim, get_mcs
import os
import pickle
import json
def smiles_acc(pred, trg):
    cnt = 0
    pred = torch.argmax(pred, dim=2)
    for i in range(len(pred)):
        cnt += 1 if not torch.sum(pred[i, :len(trg[i]) - 1] ^ trg[i][1:len(trg[i])]).item() else 0
    return cnt

class DeePFAS(pl.LightningModule):
    def __init__(self,
                 deepfas_args: DeePFASConfig,
                 ae_args: MolExtractorConfig,
                 device,
                 src_pad_idx=0,
                 train_step_check_result=None,
                 val_step_check_result=None):
        super(DeePFAS, self).__init__()

        self.save_hyperparameters()
        self.src_pad_idx = src_pad_idx
        self.deepfas_config = deepfas_args
        self.ae_config = ae_args

        self.model = AttentionConv2D(device=device, config=self.deepfas_config, src_pad_idx=self.src_pad_idx)

        if Path(self.ae_config.path.save_model_path).suffix == '.ckpt':
            self.ae = MolExtracter.load_from_checkpoint(
                checkpoint_path=self.ae_config.path.save_model_path,
                map_location=device,
            ).translator
        elif Path(self.ae_config.path.save_model_path).suffix == '.pt':
            self.ae = MolExtracter.load_translator(self.ae_config.path.save_model_path, self.ae_config, device)
        else:
            raise ValueError('Unknown suffix of molextractor')

        self.ae.freeze()
        self.train_step_check_result = train_step_check_result
        self.val_step_check_result = val_step_check_result
        self.initialize_memory()

    @staticmethod
    def load_AttConv2D(pth, args, device):
        model = AttentionConv2D(device, args)
        model.load_state_dict(torch.load(pth, map_location=device, weights_only=True))
        print('successfully load previous best AttentionConv2D parameters')
        return model

    def store_attConv2D(self, pth):
        torch.save(self.model.state_dict(), pth)
        
    def initialize_memory(self):
        if hasattr(self, "statistic_map"):
            del self.statistic_map
        if hasattr(self, "val_results"):
            del self.val_results
        self.statistic_map = {}
        self.val_results = {
            "REAL_CAN": [],
            "GENERATED": [],
            "RANDOMIZED_CONVERTED": [],
        }

    def forward(self, m_z, intensity):
        return self.model(m_z, intensity)

    def training_step(self, batch, batch_idx):
        randomized_smiles, canonical_smiles, m_z, intensity = batch
        _, _, z, _ = self.ae.forward_encoder(randomized_smiles)
        z_p = self.model(m_z, intensity)
        loss = F.mse_loss(z_p, z)
        if batch_idx % self.deepfas_config.hparams.log_every_n_steps == 0:
            self.log('Train loss', float(loss), sync_dist=True, on_step=True, prog_bar=True, batch_size=len(randomized_smiles))

        if batch_idx % self.train_step_check_result == 0:
            pred = self.ae.sample(
                len(z_p),
                max_len=self.ae_config.hparams.max_len,
                z=z_p,
                device=self.device,
            )
            pred_2 = self.ae.sample(
                len(z),
                max_len=self.ae_config.hparams.max_len,
                z=z,
                device=self.device,
            )
            batch_acc = sum([r == idx_to_smiles(p) for r, p in zip(pred, canonical_smiles)])
            print(f'Real canonical smiles: {idx_to_smiles(canonical_smiles[0])}\n')
            print(f'Pred smiles: {pred[0]}\n')
            print(f'Batch Acc: {batch_acc / self.ae_config.hparams.batch_size}\n')
            print(f'AE smiles: {pred_2[0]}\n')

        return loss

    def validation_step(self, batch, batch_idx):
        randomized_smiles, canonical_smiles, m_z, intensity = batch
        z_p = self.model(m_z, intensity)
        _, _, z, _ = self.ae.forward_encoder(randomized_smiles)
        loss = F.mse_loss(z_p, z)
        current_samples = self.ae.sample(
            len(z_p),
            max_len=self.ae_config.hparams.max_len,
            z=z_p,
            device=self.device,
        )
        if batch_idx % self.deepfas_config.hparams.log_every_n_steps == 0:
            self.log('Val loss', float(loss), sync_dist=True, on_step=True, prog_bar=True, batch_size=len(batch[0]))
        self.val_results['RANDOMIZED_CONVERTED'].extend(randomized_smiles)
        self.val_results['GENERATED'].extend(current_samples)
        self.val_results['REAL_CAN'].extend(canonical_smiles)

        return loss

    def on_validation_epoch_end(self):
        self.val_results['RANDOMIZED_CONVERTED'] = [idx_to_smiles(x) for x in self.val_results['RANDOMIZED_CONVERTED']]
        self.val_results['REAL_CAN'] = [idx_to_smiles(x) for x in self.val_results['REAL_CAN']]
        samples = pd.DataFrame(self.val_results)
        samples['MATCH'] = samples['REAL_CAN'] == samples['GENERATED']
        total = len(samples)
        match = samples['MATCH'].sum()
        pct_match = samples['MATCH'].mean()
        print(samples.head())
        print(f'Total: {total} Matched: {match} Percent Matched {pct_match}\n')
        self.log("Percent Matched", pct_match, on_epoch=True, sync_dist=True)
        self.initialize_memory()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.deepfas_config.hparams.lr,
            weight_decay=self.deepfas_config.hparams.weight_decay,
            eps=self.deepfas_config.hparams.adam_eps,
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer=optimizer,
            factor=self.deepfas_config.hparams.factor,
            patience=self.deepfas_config.hparams.patience,
            verbose=self.deepfas_config.hparams.verbose,
            min_lr=self.deepfas_config.hparams.min_lr,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "Val loss",
                "frequency": 1,
                "interval": "epoch"
            },
        }

class InfernceDeePFAS(pl.LightningModule):
    def __init__(self,
                 ae_config: MolExtractorConfig,
                 deepfas_config: DeePFASConfig,
                 deepfas_model_pth,
                 retrieval_results_pth,
                 retrieval_dataloader,
                 mode,
                 topk,
                 device):
        super(InfernceDeePFAS, self).__init__()

        if Path(deepfas_model_pth).suffix == '.pt':
            self.model = DeePFAS.load_AttConv2D(deepfas_model_pth, deepfas_config, device)
            self.ae = MolExtracter.load_translator(ae_config.path.save_model_path, ae_config, device)
        elif Path(deepfas_model_pth).suffix == '.ckpt':
            self.model = DeePFAS.load_from_checkpoint(
                checkpoint_path=deepfas_model_pth,
                map_location=device,
            )
            self.ae = self.model.ae
        else:
            raise ValueError('Unkwon Model !!')

        self.ae_config = ae_config
        self.deepfas_config = deepfas_config
        self.retrieval_dataloader = retrieval_dataloader
        self.retrieval_results_pth = retrieval_results_pth
        self.topk = topk
        self.mode = mode
        self.initialize_memory()

    def initialize_memory(self):
        if hasattr(self, "statistic_map"):
            del self.statistic_map
        self.statistic_map = {}
        if self.mode == 'eval':
            if hasattr(self, "test_results"):
                del self.test_results
            self.test_results = {
                "REAL_CAN": [],
                "GENERATED": [],
                "RANDOMIZED_CONVERTED": [],
            }

    def test_step(self, batch, batch_idx):
        randomized_smiles, canonical_smiles, m_z, intensity, pepmass, precursor_type, titles = batch
        z_p = self.model(m_z, intensity)
        if self.mode == 'eval':
            _, _, z, _ = self.ae.forward_encoder(randomized_smiles)
            loss = F.mse_loss(z_p, z)
            current_samples = self.ae.sample(
                len(z_p),
                max_len=self.ae_config.hparams.max_len,
                z=z_p,
                device=self.device,
            )
            if batch_idx % self.deepfas_config.hparams.log_every_n_steps == 0:
                self.log('Test loss', float(loss), sync_dist=True, on_step=True, prog_bar=True, batch_size=len(batch[0]))
            self.test_results['RANDOMIZED_CONVERTED'].extend(randomized_smiles)
            self.test_results['GENERATED'].extend(current_samples)
            self.test_results['REAL_CAN'].extend(canonical_smiles)

        search_topk(
            z_p,
            titles,
            pepmass,
            precursor_type,
            [ids[1:-1] for ids in canonical_smiles] if self.mode == 'eval' else None,
            self.retrieval_dataloader,
            topk=self.topk,
            loss_fn='mse',
            test_retrieval_out_dir=self.retrieval_results_pth,
            statistic_map=self.statistic_map,
            mode=self.mode
        )
        if self.mode == 'eval':
            return loss

    def on_test_epoch_end(self):
        if self.mode == 'eval':
            for k, v in self.statistic_map.items():
                self.statistic_map[k]['similarity'] = [get_RDFsim(c, k) for c in v['candidates']]
                self.statistic_map[k]['mcs'] = [get_mcs(c, k) for c in v['candidates']]
            self.test_results['RANDOMIZED_CONVERTED'] = [idx_to_smiles(x) for x in self.test_results['RANDOMIZED_CONVERTED']]
            self.test_results['REAL_CAN'] = [idx_to_smiles(x) for x in self.test_results['REAL_CAN']]
            samples = pd.DataFrame(self.test_results)
            samples['MATCH'] = samples['REAL_CAN'] == samples['GENERATED']
            total = len(samples)
            match = samples['MATCH'].sum()
            pct_match = samples['MATCH'].mean()
            print(samples.head())
            print(f'Total: {total} Matched: {match} Percent Matched {pct_match}')
            samples.to_csv(os.path.join(self.retrieval_results_pth, 'denovo_generated.csv'), sep='\t')
            self.log("Percent Matched", pct_match, on_epoch=True, sync_dist=True)

        with open(os.path.join(self.retrieval_results_pth,'statistic.json'), 'w') as f:
            json.dump(self.statistic_map, f, indent=4)
        with open(os.path.join(self.retrieval_results_pth,'statistic.pkl'), 'wb') as f:
            pickle.dump(self.statistic_map, f)
        self.initialize_memory()

