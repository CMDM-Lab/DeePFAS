import json
import math
import os
from typing import Optional

from pydantic import BaseModel
from pyteomics import mgf


def get_absolute_path(relative_path):
    current_file_dir = os.path.dirname(os.path.abspath(__file__))
    absolute_path = os.path.join(current_file_dir, relative_path)
    normalized_path = os.path.normpath(absolute_path)
    
    return normalized_path

class PathConfig(BaseModel):
    model_dir: str
    save_model_path: str
    dataset_path: str
    decoys_dataset_pth: str

    class Config:
        # model_ conflict
        protected_namespaces = ()

class Hparams(BaseModel):
    max_len: int
    batch_size: int
    val_frac: float
    train_num_workers: int
    val_num_workers: int
    drop_last: bool
    lr: float
    factor: float
    adam_eps: float
    patience: int
    min_lr: float
    max_epoch: int
    clip: float
    seed: int
    weight_decay: float
    regression_loss_weight: float
    use_properties: bool
    use_contrastive_learning: bool
    randomized: bool
    ignore_vae: bool
    log_every_n_steps: int
    precision: int
    num_devices: int
    val_check_interval: float
    accumulate_grad_batches: int
    num_sanity_val_steps: int
    save_top_k: int
    save_last: bool
    save_every_n_train_steps: int
    save_every_n_train_epochs: int
    num_decoys: int
    verbose: bool
    instances_buffer_size: int
    buffer_max_batch: int
    debug: bool

class EmbeddingConfig(BaseModel):
    e_dim: int
    z_dim: int
    enc_voc_size: int
    dec_voc_size: int

class EncoderConfig(BaseModel):
    in_dim: int
    hid_dim: int
    n_layers: int
    drop_prob: float
    bidir: bool

class DecoderConfig(BaseModel):
    in_dim: int
    hid_dim: int
    n_layers: int
    drop_prob: float

class LoggerConfig(BaseModel):
    project_name: str
    run_name: str
    wandb_entity_name: str
    job_key: str

class PropConfig(BaseModel):
    hidden_dim: int
    prop_dim: int
    fp_dim: int

class MolExtractorConfig(BaseModel):
    embedding: EmbeddingConfig
    encoder: EncoderConfig
    decoder: DecoderConfig
    path: PathConfig
    hparams: Hparams
    prop: Optional[PropConfig] = None
    logger: LoggerConfig

    @classmethod
    def from_json_path(cls, json_path):
        with open(json_path, 'r') as f:
            config = json.load(f)
            run_name = ''.join([
                f"lr_{config['hparams']['lr']}_",
                f"batch_size_{config['hparams']['batch_size']}_",
                f"accumulated_gradient_batch_{config['hparams']['accumulate_grad_batches']}_",
                f"{config['logger']['project_name']}",

            ])
            job_key = run_name
            config['logger']['run_name'] = run_name
            config['logger']['job_key'] = job_key
        return cls(**config)
