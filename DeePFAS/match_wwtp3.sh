
CURRENT_DIR=$(pwd)

export PYTHONPATH="${CURRENT_DIR}:${PYTHONPATH}"

echo $PYTHONPATH

python3 DeePFAS/test_deepfas.py \
 --deepfas_config_pth DeePFAS/config/deepfas_config.json \
 --ae_config_pth ae/config/ae_config.json \
 --test_data_pth ../DATASET/wwtp3_remain.mgf \
 --retrieval_data_pth ../DATASET/PubChem_chemical_embbeddings_chunk_size_uncompressed.hdf5 \
 --results_dir ./WWTP3_remain_results \
 --topk 20 \
 --mode inference
