
from ae.models.gru_autoencoder import TranslationModel
from ae.models.chemical_model import PropModel
from ae.config.models import MolExtractorConfig
from ae.utils.smiles_process.functions import idx_to_smiles
import pytorch_lightning as pl
import torch
import pandas as pd
import torch.nn as nn

def initialize_weights(model):
    for m in model:
        if isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight)

def smiles_acc(pred, trg):
    cnt = 0
    pred = torch.argmax(pred, dim=2)
    for i in range(len(pred)):
        cnt += 1 if not torch.sum(pred[i, :len(trg[i]) - 1] ^ trg[i][1:len(trg[i])]).item() else 0
    return cnt

class MolExtracter(pl.LightningModule):
    def __init__(self, 
                 args: MolExtractorConfig):

        super(MolExtracter, self).__init__()

        self.save_hyperparameters()
        self.translator = TranslationModel(args)
        self.use_properties = args.hparams.use_properties
        if self.use_properties:
            self.prop_model = PropModel(prop_dim=args.prop.prop_dim,
                                        hidden_dim=args.prop.hidden_dim)

        self.config = args
        self.initialize_memory()

    @staticmethod
    def load_translator(pth, args, device):
        model = TranslationModel(args)
        model.load_state_dict(torch.load(pth, map_location=device, weights_only=True))
        print('successfully load previous best AE parameters')
        return model.to(device)

    def store_translator(self, pth):
        torch.save(self.translator.state_dict(), pth)
        print('successfully store previous best AE parameters')

    def configure_optimizers(self):

        initialize_weights(self.modules())
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.config.hparams.lr,
            weight_decay=self.config.hparams.weight_decay,
            eps=self.config.hparams.adam_eps,
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer=optimizer,
            factor=self.config.hparams.factor,
            patience=self.config.hparams.patience,
            verbose=self.config.hparams.verbose,
            min_lr=self.config.hparams.min_lr,
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

    def initialize_memory(self):
        self.val_results = {
            "REAL_CAN": [],
            "GENERATED": [],
            "RANDOMIZED_CONVERTED": [],
        }

    def training_step(self, batch, batch_idx):
        randomized_smiles, canonical_smiles, properties = batch
        mu, logvar, z, kl_loss, recon_loss, pred = self.translator(
            randomized_smiles, canonical_smiles,
        )
        p_loss = 0
        if self.use_properties:
            p_loss = self.prop_model(z, properties) if self.config.hparams.use_properties else 0
            p_loss *= self.config.hparams.regression_loss_weight

        loss = (
            recon_loss +
            p_loss
        )

        metric = {
            "Recon loss": float(recon_loss),
            "Prop loss": float(p_loss),
            "Train loss": float(loss),
        }
        # if self.global_step % self.config.hparams.log_every_n_steps == 0:
        self.log_dict(metric, sync_dist=True, on_step=True, prog_bar=True, batch_size=len(batch[0]))

        return loss

    def validation_step(self, batch, batch_idx):
        randomized_smiles, canonical_smiles, properties, neg_samples, aux_masks = batch
        mu, logvar, z, kl_loss, recon_loss, _ = self.translator(
            randomized_smiles, canonical_smiles,
        )
        current_samples = self.translator.sample(len(mu), max_len=self.config.hparams.max_len, z=mu, device=mu.device)
        p_loss = 0
        if self.use_properties:
            p_loss = self.prop_model(z, properties) if self.config.hparams.use_properties else 0
            p_loss *= self.config.hparams.regression_loss_weight

        loss = (
            recon_loss +
            p_loss
        )
        metric = {
            "Recon loss": float(recon_loss),
            "Prop loss": float(p_loss),
            "Val loss": float(loss),
        }
        # if self.global_step % self.config.hparams.log_every_n_steps == 0:
        self.log_dict(metric, sync_dist=True, on_step=True, prog_bar=True, batch_size=len(batch[0]))

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
