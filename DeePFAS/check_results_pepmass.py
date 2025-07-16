from pyteomics import mgf
import json
from tqdm import tqdm
import pandas as pd

with open('./T3_results/statistic.json', 'r') as f:
# with open('../Dual_Mass/2_26_3layers_256_filters_20_50_watersamples_aggregated_results/statistic.json', 'r') as f:
    T3_results = json.load(f)


pfascreen_results = pd.read_csv('../PFAScreen/PFAScreen_new_cali_T3/only_ms2_results.csv')
pfascreen_pepmass_map = {}
for ms1 in pfascreen_results['mz']:
    hit = False
    for k, v in pfascreen_pepmass_map.items():
        if abs(k - ms1) / k * 1e6 <= 5.0:
            pfascreen_pepmass_map[k] += 1
            hit = True
            break
    if not hit:
        pfascreen_pepmass_map[ms1] = 1


title_map = {k:1 for k, v in T3_results.items() if v['pfas_confidence_level_float'] >= 0.799}
pepmass_map = {}
total_pepmass_map = {}

with mgf.read('../DATASET/new_cali_T3_ms2.mgf') as f:
# with mgf.read('../DATASET/peak_spectra.mgf') as f:
    for o in tqdm(f):
        title = str(o['params']['title'])
        pepmass = float(o['params']['pepmass'][0])
        hit = False
        hit_pepmass = -1
        for k, v in pepmass_map.items():
            if abs(pepmass - k) / k * 1e6 <= 5.0:
                total_pepmass_map[k] += 1
                hit = True
                hit_pepmass = k
                break
        if not hit:
            total_pepmass_map[pepmass] = 1
        if title_map.get(title) is not None:
            if hit:
                pepmass_map[hit_pepmass] += 1
            else:
                pepmass_map[pepmass] = 1
        # total_pepmass_map.setdefault(round(pepmass, 3), 0)
        # total_pepmass_map[round(pepmass, 3)] += 1
        # if title_map.get(title) is not None:
        #     pepmass_map.setdefault(round(pepmass, 3), 0)
        #     pepmass_map[round(pepmass, 3)] += 1

pepmass_tuple = [(k, v) for k, v in pepmass_map.items()]
pepmass_tuple_high_confidence_tupe = []
for i, (t1, t2) in enumerate(pepmass_tuple):
    tot = total_pepmass_map[t1]
    pepmass_tuple[i] = (t1, t2, tot, f'{round(t2 / tot * 1e2, 2)}%', round(t2 / tot * 1e2, 2))
    pepmass_tuple_high_confidence_tupe.append(t1)

pepmass_tuple.sort(key=lambda x: x[0], reverse=True)


cnt_statistic = {}
percentage_statistic = {}
for t1, t2, tot, percentage, rate in pepmass_tuple:
    cnt_statistic.setdefault(tot, 0)
    cnt_statistic[tot] += 1
    percentage_statistic.setdefault(percentage, 0)
    percentage_statistic[percentage] += 1
cnt_statistic_tuple = [(k, v) for k, v in cnt_statistic.items()]
percentage_statistic_tuple = [(k, v) for k, v in percentage_statistic.items()]
cnt_statistic_tuple.sort(key=lambda x: x[1], reverse=True)
percentage_statistic_tuple.sort(key=lambda x: x[1], reverse=True)

total_consistent_classes = 0
for k, v in pfascreen_pepmass_map.items():
    for k2, v2 in pepmass_map.items():
        if abs(k - k2) / max(k, k2) * 1e6 <= 5.0:
            total_consistent_classes += 1
            break
    # for ms1 in pepmass_tuple_high_confidence_tupe:
    #     if  abs(ms1 - k) / max(ms1, k) * 1e6 <= 5.0:
    #         total_consistent_classes += 1
    #         break

print(f'total num class: {len(total_pepmass_map)}\n')
print(f'total num of spectra with confidence level: {len(title_map)}\n')
print(f'total num class {len(pepmass_map)}\n')
# print(f'total num class {len(pepmass_tuple_high_confidence_tupe)}\n')
print(f'total num class (pfascreen): {len(pfascreen_pepmass_map)}\n')
print(f'total num consistent class: {total_consistent_classes}\n')
print(pepmass_tuple)
print(cnt_statistic_tuple)
percentage_statistic_sum = sum([b for a, b in percentage_statistic_tuple])
print(f'sum percentage: {percentage_statistic_sum}\n')
print(percentage_statistic_tuple)