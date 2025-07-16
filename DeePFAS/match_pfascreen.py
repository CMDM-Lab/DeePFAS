
import pandas as pd



def match_wwtp3(wwtp3_names, pfascreen_preds):
    wwtp3_stds = {}
    wwtp3_remain = {}
    with open(wwtp3_names, 'r') as f:

        while True:
            line = f.readline().replace('\n', '').strip()
            if line is None or not line:
                break
            name, smiles, pepmass, cid, cf = line.split('|')
            pepmass = float(pepmass)
            if cf in ['1a', '1b']:
                wwtp3_stds[pepmass] = 1
            else:
                wwtp3_remain[pepmass] = 1


    preds = pd.read_csv(pfascreen_preds)

    num_matched_stds = 0
    num_matched_remains = 0
    matched_stds_pfascreen = {}
    matched_remains_pfascreen = {}
    for mz in preds['mz']:
        for k, v in wwtp3_stds.items():
            if abs(k - mz) / k * 1e6 <= 5.0 and matched_stds_pfascreen.get(k) is None:
                num_matched_stds += 1
                matched_stds_pfascreen[k] = 1
                break

        for k, v in wwtp3_remain.items():
            if abs(k - mz) / k * 1e6 <= 5.0 and matched_remains_pfascreen.get(k) is None and matched_stds_pfascreen.get(k) is None:
                num_matched_remains += 1
                matched_remains_pfascreen[k] = 1
                break

    print(matched_stds_pfascreen)
    print(matched_remains_pfascreen)
    print(f'total stds: {len(wwtp3_stds)}\n')
    print(f'matched stds: {num_matched_stds}\n')
    print(f'total remains: {len(wwtp3_remain)}\n')
    print(f'matched remains: {num_matched_remains}\n')

if __name__ == '__main__':
    wwtp3_names = '../DATASET/WWTP3_name.txt'
    # pfascreen_preds = '../PFAScreen/PFAScreen_new_cali_T3/Results_PFAScreen_new_cali_T3.csv'
    # pfascreen_preds = '../PFAScreen/PFAScreen_new_cali_T3/ms2_metadata.csv'
    # pfascreen_preds = '../PFAScreen/PFAScreen_new_cali_T3/only_ms2_results.csv'
    pfascreen_preds = '../PFAScreen/PFAScreen_new_cali_T3/only_ms2_n_dias_diffs_results.csv'
    # pfascreen_preds = '../PFAScreen/PFAScreen_new_cali_T3/only_ms2_and_smiles_n_dias_diffs_results.csv'
    match_wwtp3(wwtp3_names, pfascreen_preds)