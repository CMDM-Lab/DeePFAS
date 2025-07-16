CURRENT_DIR=$(pwd)

export PYTHONPATH="${CURRENT_DIR}:${PYTHONPATH}"

echo $PYTHONPATH

python3 ae/gen_latent_space.py \
 --deepfas_config_pth DeePFAS/config/deepfas_config.json \
 --ae_config_pth ae/config/gen_latent_space_config.json \
 --latent_space_out_pth customized_mol_database.hdf5 \
 --chunk_size 100000 \
 --compression_level 9