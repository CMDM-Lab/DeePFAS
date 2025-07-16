import json
import os
import time
from queue import PriorityQueue
import heapq
import torch
import torch.nn.functional as F
from colorama import Fore
from colorama import Style
from DeePFAS.models.metrics.eval import (CosSimLoss, epoch_time,
                                         get_4096_Morgan_sim, get_formula_diff,
                                         get_mcs, get_RDFsim)
from DeePFAS.utils.smiles_process import idx_to_smiles, remove_salt_stereo
from DeePFAS.utils.spectra_process import cal_mw
from DeePFAS.utils.OECD import oecd_pfas
from rdkit.Chem import Descriptors
from rdkit import Chem
from tqdm import tqdm

def search_topk(
    preds,
    titles,
    pepmass,
    precursor_type,
    canonicalsmiles_list,
    retrieval_dataloader,
    loss_fn,
    topk,
    test_retrieval_out_dir,
    statistic_map,
    mode
):
    mw_list = [
        cal_mw(
            float(pe),
            pr,
        ) for pe, pr in zip(pepmass, precursor_type)
    ]

    if mode == 'eval':
        real_smiles = [remove_salt_stereo(idx_to_smiles(c_ids)) for c_ids in canonicalsmiles_list]
    title_list = [int(title) for title in titles]
    with torch.no_grad():
        start_time = time.time()
        pq = [PriorityQueue(topk) for _ in range(len(mw_list))] 
        mol_set = [set() for _ in range(len(mw_list))]
        retrieval_iter = iter(retrieval_dataloader)
        for _ in tqdm(range(len(retrieval_dataloader))):
            batch = next(retrieval_iter)
            canonical_smiles, embeddings = batch
            for idx, (mw, pred) in enumerate(zip(mw_list, preds)):

                if loss_fn == 'mse':
                    batch_loss = F.mse_loss(pred, embeddings, reduction='none')
                elif loss_fn == 'cos':
                    batch_loss = CosSimLoss(reduction=True)(pred, embeddings)
                else:
                    raise ValueError('Wrong loss functions')
                batch_loss = torch.sum(batch_loss, dim=1).unsqueeze(1)
                for i, loss in enumerate(batch_loss):
                    loss_value = loss.item()
                    if canonical_smiles[i] in mol_set[idx]:
                        continue
                    if pq[idx].qsize() < topk:
                        try:
                            mol = Chem.MolFromSmiles(canonical_smiles[i])
                        except:
                            continue
                        mol_set[idx].add(canonical_smiles[i])
                        # mol_map[canonical_smiles[i]] = 1
                        pq[idx].put((-loss.item(), canonical_smiles[i]))
                    else:
                        try:
                            tail = pq[idx].get()
                            # mol_map[tail[1]] = 0
                        except:
                            print(tail)
                            raise ValueError('Error')
                        if loss.item() < -tail[0]:
                            try:
                                mol = Chem.MolFromSmiles(canonical_smiles[i])
                                candidate_mw = Descriptors.ExactMolWt(mol)
                            except:
                                pq[idx].put(tail)
                                continue
                            pq[idx].put((-loss.item(), canonical_smiles[i]))
                            mol_set[idx].add(canonical_smiles[i])
                            mol_set[idx].remove(tail[1])
                            # mol_map[canonical_smiles[i]] = 1
                        else:
                            pq[idx].put(tail)
                            # mol_map[canonical_smiles[i]] = 1
        end_time = time.time()
        mins, secs = epoch_time(start_time, end_time)
        print(f'Inference Time: {mins}m {secs}s')

        for idx, (mw, pred) in enumerate(zip(mw_list, preds)):
            candidates = []
            while not pq[idx].empty():
                try:
                    loss, smiles = pq[idx].get()
                except:
                    print(loss)
                    print(smiles)
                    raise ValueError('Error')
                mol = Chem.MolFromSmiles(smiles)
                candidate_mw = Descriptors.ExactMolWt(mol)
                is_pfas = oecd_pfas(smiles)
                candidates.append(
                    {
                        'loss': abs(loss),
                        'smiles': smiles,
                        'mw': candidate_mw,
                        'mw_diff': abs(mw - candidate_mw),
                        'is pfas': is_pfas
                    }
                )
            candidates.sort(key=lambda x: (x['loss'], x['mw_diff']))
            out_file = os.path.join(test_retrieval_out_dir, f'{title_list[idx]}_pred.txt')
            if mode == 'eval':
                hit = real_smiles[idx] in [candidate['smiles'] for candidate in candidates]
                if hit:
                    print(f'\ntarget compound {Fore.GREEN}in{Style.RESET_ALL} candidates')
                else:
                    print(f'\ntarget compound {Fore.RED}not in{Style.RESET_ALL} candidates')
                print(f'Target Compound: {real_smiles[idx]}\n')
                print(candidates)
                with open(out_file, 'a') as f:
                    if real_smiles:
                        rank = []
                        f.write(f'Target Compound: {real_smiles[idx]}\n')
                        if hit:
                            f.write(f'Target compound in candidates\n')
                            for j, candidate in enumerate(candidates):
                                if real_smiles[idx] == candidate['smiles']:
                                    rank.append(j+1)
                        else:
                            f.write(f'Target compound not in candidates\n')
                    for j, candidate in enumerate(candidates):
                        f.write(f"\ncandidate {j+1}\nloss:{candidate['loss']}\nSMILES:{candidate['smiles']}\nmw:{candidate['mw']}\nmw diff:{candidate['mw_diff']}\n")
                if statistic_map.get(real_smiles[idx]) is None:
                    statistic_map[real_smiles[idx]] = {
                        'hit topk': int(hit),
                        'title': [title_list[idx]],
                        'rank': rank if hit else ['()'],
                        'not hit topk': int(not hit),
                        'candidates': [candidates],
                        # 'similarity': [get_RDFsim(candidates, real_smiles[idx])],
                        # 'mcs': [get_mcs(candidates, real_smiles[idx])],
                    }
                else:
                    statistic_map[real_smiles[idx]]['candidates'].append(candidates)
                    statistic_map[real_smiles[idx]]['hit topk'] += int(hit)
                    statistic_map[real_smiles[idx]]['not hit topk'] += int(not hit)
                    statistic_map[real_smiles[idx]]['rank'].append(f"({', '.join([str(r) for r in rank])})")
                    # statistic_map[real_smiles[idx]]['similarity'].append(get_RDFsim(candidates, real_smiles[idx]))
                    # statistic_map[real_smiles[idx]]['mcs'].append(get_mcs(candidates, real_smiles[idx]))
                    statistic_map[real_smiles[idx]]['title'].append(title_list[idx])
            else:
                print(f'\n{Fore.GREEN}Title {title_list[idx]}{Style.RESET_ALL} spectrum {Fore.BLUE}top{topk} {Style.RESET_ALL}candidates:\n')
                print(candidates)
                tot_is_pfas = sum([int(c['is pfas']) for c in candidates])
                statistic_map[title_list[idx]] = {
                    'candidates': candidates,
                    'pfas_confidence_level': f'{(tot_is_pfas / len(candidates) * 100):.2f}%',
                    'pfas_confidence_level_float': round(tot_is_pfas / len(candidates), 2),
                }
                with open(out_file, 'a') as f:
                    for j, candidate in enumerate(candidates):
                        f.write(f"\ncandidate {j+1}\nloss:{candidate['loss']}\n{candidate['smiles']}\nmw:{candidate['mw']}\nmw diff:{candidate['mw_diff']}\n")

def inference_topk(
    preds,
    titles,
    pepmass,
    precursor_type,
    canonicalsmiles_list,
    AE,
    retrieval_dataloader,
    loss_fn,
    topk,
    test_retrieval_out_dir,
    statistic_map,
    ):

    mw_list = [
        cal_mw(
            float(pe),
            pr,
        ) for pe, pr in zip(pepmass, precursor_type)
    ]

    real_smiles = [remove_salt_stereo(idx_to_smiles(c_ids)) for c_ids in canonicalsmiles_list]
    title_list = [int(title) for title in titles]
    with torch.no_grad():
        start_time = time.time()
        for idx, (mw, pred) in enumerate(zip(mw_list, preds)):
            pq = PriorityQueue(topk)
            mol_map = {}
            min_loss = 1e5
            retrieval_iter = iter(retrieval_dataloader)
            for _ in tqdm(range(len(retrieval_dataloader))):
                batch = next(retrieval_iter)
                randomized_smiles, canonical_smiles, properties, neg_samples, aux_masks = batch
                _, _, z, _ = AE.forward_encoder(randomized_smiles)
                if loss_fn == 'mse':
                    batch_loss = F.mse_loss(pred, z, reduction='none')
                elif loss_fn == 'cos':
                    batch_loss = CosSimLoss(reduction=True)(pred, z)
                else:
                    raise ValueError('Wrong loss functions')
                batch_loss = torch.sum(batch_loss, dim=1).unsqueeze(1)
                for i, loss in enumerate(batch_loss):
                    min_loss = min(loss.item(), min_loss)
                    smiles_str = idx_to_smiles(canonical_smiles[i][1:-1])
                    if mol_map.get(smiles_str) == 1:
                        continue
                    if pq.qsize() < topk:
                        mol_map[smiles_str] = 1
                        pq.put((-loss.item(), smiles_str))
                    else:
                        try:
                            tail = pq.get()
                            mol_map[tail[1]] = 0
                        except:
                            print(tail)
                            raise ValueError('Error')
                        if loss.item() < -tail[0]:
                            pq.put((-loss.item(), smiles_str))
                            mol_map[smiles_str] = 1
                        else:
                            pq.put(tail)
                            mol_map[smiles_str] = 1
            end_time = time.time()
            mins, secs = epoch_time(start_time, end_time)
            print(f'Inference Time: {mins}m {secs}s')
            candidates = []
            
            while not pq.empty():
                try:
                    loss, smiles = pq.get()
                except:
                    print(loss)
                    print(smiles)
                    raise ValueError('Error')
                mol = Chem.MolFromSmiles(smiles)
                candidate_mw = Descriptors.ExactMolWt(mol)
                candidates.append({'loss': abs(loss), 'smiles': smiles, 'mw': candidate_mw, 'mw_diff': abs(mw - candidate_mw)})
            out_file = os.path.join(test_retrieval_out_dir, f'{idx+1}_pred.txt')
            hit = real_smiles[idx] in [candidate['smiles'] for candidate in candidates]
            if hit:
                print('target compound in candidates')
            else:
                print('target compound not in candidates')
            print(f'Target Compound: {real_smiles[idx]}\n')
            print(candidates)
            with open(out_file, 'a') as f:
                if real_smiles:
                    rank = []
                    f.write(f'Target Compound: {real_smiles[idx]}\n')
                    if hit:
                        f.write(f'Target compound in candidates\n')
                        for j, candidate in enumerate(candidates):
                            if real_smiles[idx] == candidate['smiles']:
                                rank.append(j+1)
                    else:
                        f.write(f'Target compound not in candidates\n')
                for j, candidate in enumerate(candidates):
                    f.write(f"candidate {j+1}\nloss:{candidate['loss']} {candidate['smiles']}\n")
            if statistic_map.get(real_smiles[idx]) is None:
                statistic_map[real_smiles[idx]] = {
                    'hit topk': int(hit),
                    'title': [title_list[idx]],
                    'rank': rank if hit else ['()'],
                    'not hit topk': int(not hit),
                    'candidates': [candidates],
                    'similarity': [get_RDFsim(candidates, real_smiles[idx])],
                    'mcs': [get_mcs(candidates, real_smiles[idx])],
                }
            else:
                statistic_map[real_smiles[idx]]['candidates'].append(candidates)
                statistic_map[real_smiles[idx]]['hit topk'] += int(hit)
                statistic_map[real_smiles[idx]]['not hit topk'] += int(not hit)
                statistic_map[real_smiles[idx]]['rank'].append(f"({', '.join([str(r) for r in rank])})")
                statistic_map[real_smiles[idx]]['similarity'].append(get_RDFsim(candidates, real_smiles[idx]))
                statistic_map[real_smiles[idx]]['mcs'].append(get_mcs(candidates, real_smiles[idx]))
                statistic_map[real_smiles[idx]]['title'].append(title_list[idx])
