
CURRENT_DIR=$(pwd)

export PYTHONPATH="${CURRENT_DIR}:${PYTHONPATH}"

echo $PYTHONPATH

python3 aggregate_results.py \
 --dir_path ./2_26_3layers_256_filters_20_50_suspect_watersamples_PubChem_results \
 --out_path ./2_26_3layers_256_filters_20_50_suspect_watersamples_PubChem_results/statistic.json \
