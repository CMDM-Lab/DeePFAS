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

class DecoderConfig(BaseModel):
    d_model: int
    n_head: int
    n_layers: int
    drop_prob: float
    ffn_hidden: int
    voc_size: int

class EncoderConfig(BaseModel):
    d_model: int
    n_head: int
    n_layers: int
    drop_prob: float
    ffn_hidden: int
    voc_size: int

class PathConfig(BaseModel):
    model_dir: str
    save_model_path: str
    save_loss_path: str
    save_best_pct: str
    figure_path: str
    loss_dir: str
    data_path: str
    data_id_path: str
    properties_model_path: Optional[str] = None

    class Config:
        # model_ conflict
        protected_namespaces = ()

class PoolingConfig(BaseModel):
    in_dim: int
    hid_dim: int
    out_dim: int
    n_layers: int

class Hparams(BaseModel):
    max_len: int
    batch_size: int
    beam_size: int
    topk: int
    sample_size: int
    train_ratio: float
    train_num_workers: int
    val_num_workers: int
    drop_last: bool
    lr: float
    factor: float
    adam_eps: float
    patience: int
    warmup: int
    min_lr: float
    epoch: int
    clip: float
    weight_decay: float
    buffer_max_batch: int  # 200
    instances_buffer_size: int  # 128000
    train_steps: int
    val_steps: int
    use_record_idx: bool
    regression_loss_weight: Optional[float] = None
    train_step_check_result: Optional[int] = None
    val_step_check_result: Optional[int] = None
    train_check_result_ratio: Optional[int] = None
    val_check_result_ratio: Optional[int] = None
    step_evaluate: Optional[int] = None
    use_properties: Optional[bool] = None
    ignore_vae: Optional[bool] = None
    use_hydrogen: Optional[bool] = None
    use_formula: Optional[bool] = None

class SpectrumConfig(BaseModel):
    max_mz: float
    min_mz: float
    min_num_peaks: int
    max_num_peaks: int
    resolution: int
    neg: Optional[bool] = None

class MsPathConfig(PathConfig):
    test_data_path: str
    train_data_path: str
    smiles_embedding_path: Optional[str] = None

class MsConvConfig(BaseModel):
    channels_med_1: int
    channels_out: int
    channels_in: int
    conv_kernel_dim_1: int
    conv_kernel_dim_2: int
    conv_stride_1: int
    conv_stride_2: int
    conv_dilation: int
    conv_padding_1: int
    conv_padding_2: int
    pool_kernel_dim_1: int
    pool_kernel_dim_2: int
    pool_stride_1: int
    pool_stride_2: int
    pool_dilation: int
    pool_padding_1: int
    pool_padding_2: int
    fc_dim_1: int
    n_layers: int
    emb_dim: int

class Ms2VecConfig(BaseModel):
    spectrum: SpectrumConfig
    hparams: Hparams
    encoder: EncoderConfig
    conv: MsConvConfig
    path: MsPathConfig

    @classmethod
    def load_config(
        cls,
        config_path: str,
        train_data_path: str,
        test_data_path: str,
        smiles_embedding_path: str = None
    ):
        if smiles_embedding_path:
            smiles_embedding_path = get_absolute_path(smiles_embedding_path)
        with open(config_path, 'r') as f:
            data = json.load(f)
            with mgf.read(train_data_path) as m_f:
                data['hparams']['train_steps'] = math.ceil(len(m_f) / data['hparams']['batch_size'])
            with mgf.read(test_data_path) as m_f:
                data['hparams']['val_steps'] = math.ceil(len(m_f) / data['hparams']['batch_size'])
            data['path']['train_data_path'] = train_data_path
            data['path']['test_data_path'] = test_data_path
            data['hparams']['train_step_check_result'] = data['hparams']['train_steps'] // data['hparams']['train_check_result_ratio']
            data['hparams']['val_step_check_result'] = data['hparams']['val_steps'] // data['hparams']['val_check_result_ratio']
            if smiles_embedding_path:
                data['path']['smiles_embedding_path'] = smiles_embedding_path

        return cls(**data)

class GruEmbeddingConfig(BaseModel):
    e_dim: int
    z_dim: int
    enc_voc_size: int
    dec_voc_size: int

class GruEncoderConfig(BaseModel):
    in_dim: int
    hid_dim: int
    n_layers: int
    drop_prob: float
    bidir: bool

class GruDecoderConfig(BaseModel):
    in_dim: int
    hid_dim: int
    n_layers: int
    drop_prob: float

class GruOptimConfig(BaseModel):
    kl_start: float
    kl_w_start: float
    kl_w_end: float
    kr_n_period: int
    lr_start: float
    lr_n_period: float
    lr_n_mult: float
    lr_end: float
    lr_n_restarts: int

class GruAutoEncoderConfig(BaseModel):
    embedding: GruEmbeddingConfig
    encoder: GruEncoderConfig
    decoder: GruDecoderConfig
    path: PathConfig
    hparams: Hparams
    optim: GruOptimConfig

    @classmethod
    def from_json_path(cls, json_path):
        with open(json_path) as f:
            config = json.load(f)
        return cls(**config)

    @classmethod
    def load_config(
        cls,
        config_path: str,
        data_path: str,
        data_id_path: str,
        use_properties=None
    ):
        with open(config_path, 'r') as f:
            data = json.load(f)
            train_set_size = int(data['hparams']['sample_size'] * data['hparams']['train_ratio'])
            val_set_size = data['hparams']['sample_size'] - train_set_size
            data['hparams']['train_steps'] = math.ceil(train_set_size / data['hparams']['batch_size'])
            data['hparams']['val_steps'] = math.ceil(val_set_size / data['hparams']['batch_size'])
            data['path']['data_path'] = data_path
            data['path']['data_id_path'] = data_id_path
            data['hparams']['train_step_check_result'] = data['hparams']['train_steps'] // data['hparams']['train_check_result_ratio']
            data['hparams']['val_step_check_result'] = data['hparams']['val_steps'] // data['hparams']['val_check_result_ratio']
            if use_properties is not None:
                data['hparams']['use_properties'] = use_properties
        return cls(**data)

