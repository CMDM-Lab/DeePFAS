
CURRENT_DIR=$(pwd)

export PYTHONPATH="${CURRENT_DIR}:${PYTHONPATH}"

echo $PYTHONPATH

python3 DeePFAS/test_deepfas.py \
 --deepfas_config_pth DeePFAS/config/deepfas_config.json \
 --ae_config_pth ae/config/ae_config.json \
 --test_data_pth spec_dataset/test_spec.mgf \
 --retrieval_data_pth mol_dataset/5w_chemical_embbeddings.hdf5 \
 --results_dir ./test_results \
 --topk 20 \
 --mode inference
