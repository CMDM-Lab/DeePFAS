
CURRENT_DIR=$(pwd)

export PYTHONPATH="${CURRENT_DIR}:${PYTHONPATH}"

echo $PYTHONPATH

python3 ae/train_ae.py \
 --config_pth ae/config/ae_config.json