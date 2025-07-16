from pyteomics import mgf
from tqdm import tqdm
import json
import numpy as np
from DeePFAS.utils.spectra_process.functions import cal_losses
def match_peaks(spec_f, preds, ref_files, add_loss=False):

    ref_pfas_spec = []
    for ref_f in ref_files:
        with mgf.read(ref_f) as mgf_f:
            for o in tqdm(mgf_f):
                pepmass = o['params']['pepmass'][0]
                m_z, intensity = o['m/z array'], o['intensity array']
                if add_loss:
                    losses_mz, losses_intensity = cal_losses(
                        pepmass,
                        m_z,
                        intensity,
                        loss_mz_from=1.0,
                        loss_mz_to=1000.0,
                    )
                    m_z = np.concatenate((m_z, losses_mz))
                ref_pfas_spec.append(m_z)


    with open(preds, 'r') as f:
        results = json.load(f)

    cnt_map = {}

    # original_title_map = {}
    # with mgf.read(spec_f) as mgf_f:
    #     for o in tqdm(mgf_f):
    #         original_title_map[o['params']['original title']] = 1

    with mgf.read(spec_f) as mgf_f:
        for i, (k, v) in tqdm(enumerate(results.items())):
            max_peak_matched = 0
            if v["pfas_confidence_level"] != "0.00%":
                continue
            spec = mgf_f[int(k) - 1]['m/z array']

            # if v["pfas_confidence_level"] != "100.00%" or original_title_map.get(k) is None:
            #     continue
            # spec = mgf_f[i]['m/z array']

            spec = np.array(spec)
            for o in ref_pfas_spec:
                # matched_peak = (spec.reshape(-1, 1).round(2) * 1e2).astype(np.int32) == (o.reshape(1, -1).round(2) * 1e2).astype(np.int32)
                matched_peak = abs((spec.reshape(-1, 1) - o.reshape(1, -1)) / o.reshape(1, -1) * 1e6) <= 5.0
                matched_cnt = matched_peak.any(axis=1).sum()
                max_peak_matched = max(max_peak_matched, matched_cnt)
            cnt_map.setdefault(max_peak_matched, 0)
            cnt_map[max_peak_matched] += 1
    print(cnt_map)

if __name__ == '__main__':
    # spec_f = '../DATASET/peak_spectra.mgf'
    spec_f = '../DATASET/new_cali_T3_ms2.mgf'
    # spec_f = '../DATASET/new_picked_spectra.mgf'
    # preds = '../Dual_Mass/2_26_3layers_256_filters_20_50_watersamples_aggregated_results/statistic.json'
    preds = './T3_results/statistic.json'
    ref_files = [
        # '../DATASET/half_150_test.mgf',
        # '../DATASET/half_150_train.mgf',
        # '../DATASET/half_nist_pfas_test.mgf',
        # '../DATASET/half_nist_pfas_train.mgf',
        '../DATASET/new_cali_std_150.mgf'
    ]
    match_peaks(spec_f, preds, ref_files)
            