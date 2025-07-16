
CURRENT_DIR=$(pwd)

export PYTHONPATH="${CURRENT_DIR}:${PYTHONPATH}"

echo $PYTHONPATH

python3 DeePFAS/train_deepfas.py \
 --deepfas_config_pth DeePFAS/config/deepfas_config.json \
 --ae_config_pth ae/config/ae_config.json \
