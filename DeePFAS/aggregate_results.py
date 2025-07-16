

import re
from pathlib import Path
from ae.utils.OECD import oecd_pfas
from tqdm import tqdm
import json
import argparse


def arg_parser():
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument('--dir_path', type=str, required=True)
    parser.add_argument('--out_path', type=str, required=True)
    return parser.parse_args()


def extract_candidates_from_txt(dir_path, out_path):


    # 定義正則表達式來提取候選物質的資料
    candidate_pattern = r'candidate (\d+)\s+loss:([0-9.-]+)\s+([A-Za-z0-9#\[\]\(\)=\+\-\.\s]+)\s+mw:([0-9.-]+)\s+mw diff:([0-9.-]+)'
    spec_to_metadata = {}
    with open(out_path, 'w') as out_f:
        for file in tqdm(Path(dir_path).rglob('*')):
            if file.is_file() and file.name != 'statistic.json':
                with open(file, 'r') as f:
                    text = f.read()
                    extracted_data = []
                    # 用正則表達式匹配所有候選物質
                    candidates = re.findall(candidate_pattern, text)
                    if len(candidates) < 20:
                        print(candidates)
                        print(file.name)
                        print(len(candidates))
                    tot_is_pfas = 0
                    # 處理提取的資料
                    for candidate in candidates:
                        is_pfas = oecd_pfas(candidate[2])
                        tot_is_pfas += int(is_pfas)
                        candidate_data = {
                            'candidate': int(candidate[0]),
                            'loss': float(candidate[1]),
                            'smiles': candidate[2],
                            'mw': float(candidate[3]),
                            'mw_diff': float(candidate[4]),
                            'is_pfas': is_pfas
                        }
                        extracted_data.append(candidate_data)

                    spec_to_metadata[file.name.split('_')[0]] = {
                        'pfas_confidence_level': f'{(tot_is_pfas / len(extracted_data) * 100):.2f}%',
                        'pfas_confidence_level_float': round(tot_is_pfas / len(extracted_data), 2),
                        'candidates': extracted_data
                    }
                    
        json.dump(spec_to_metadata, out_f)
    return spec_to_metadata


if __name__ == '__main__':
    args = arg_parser()
    data = extract_candidates_from_txt(args.dir_path, args.out_path)
