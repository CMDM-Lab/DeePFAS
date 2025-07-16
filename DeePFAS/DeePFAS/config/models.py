import json
import math
from typing import Optional

from pydantic import BaseModel
from pyteomics import mgf

class PathConfig(BaseModel):
    model_dir: str
    save_model_path: str

    class Config:
        # model_ conflict
        protected_namespaces = ()

class Hparams(BaseModel):
    max_len: int
    batch_size: int
    train_num_workers: int
    val_num_workers: int
    lr: float
    factor: float
    adam_eps: float
    patience: int
    min_lr: float
    max_epoch: int
    clip: float
    seed: int
    weight_decay: float
    log_every_n_steps: int
    num_devices: int
    val_check_interval: float
    accumulate_grad_batches: int
    num_sanity_val_steps: int
    precision: int
    save_top_k: int
    save_last: bool
    save_every_n_train_steps: int
    save_every_n_train_epochs: int
    verbose: bool
    instances_buffer_size: int
    buffer_max_batch: int
    train_check_result_ratio: float
    val_check_result_ratio: float
    debug: bool

class EncoderConfig(BaseModel):
    d_model: int
    n_head: int
    n_layers: int
    drop_prob: float
    ffn_hidden: int
    voc_size: int

class LoggerConfig(BaseModel):
    project_name: str
    run_name: str
    wandb_entity_name: str
    job_key: str

class SpectrumConfig(BaseModel):
    max_mz: float
    min_mz: float
    min_num_peaks: int
    max_num_peaks: int
    loss_mz_from: Optional[int] = None
    loss_mz_to: Optional[int] = None
    resolution: int
    tokenized: bool
    use_neutral_loss: bool
    neg: Optional[bool] = None

class MsPathConfig(PathConfig):
    train_data_path: str
    val_data_path: str

class DeePFASConfig(BaseModel):
    spectrum: SpectrumConfig
    hparams: Hparams
    encoder: EncoderConfig
    path: MsPathConfig
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
