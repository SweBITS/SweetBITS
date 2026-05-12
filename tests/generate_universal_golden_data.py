import os
import csv
import math
import random
import numpy as np
from pathlib import Path
from joltax import JolTree

# --- CONFIGURATION ---
SAMPLES = [
    {"id": f"Ki-2024_{i:02d}_001", "year": 2024, "week": i} for i in range(1, 11)
]

CANONICAL_RANKS = ["superkingdom", "phylum", "class", "order", "family", "genus", "species"]

def get_weighted_stats(data, value_col, weight_col):
    if not data:
        return {'mean': None, 'median': None, 'cv': None, 'p05': None, 'p95': None}
    sorted_data = sorted(data, key=lambda x: x[value_col])
    total_weight = sum(x[weight_col] for x in sorted_data)
    if total_weight == 0:
        return {'mean': None, 'median': None, 'cv': None, 'p05': None, 'p95': None}
    mean_val = sum(x[value_col] * x[weight_col] for x in sorted_data) / total_weight
    def get_quantile(q):
        target = total_weight * q
        cum_weight = 0
        for x in sorted_data:
            cum_weight += x[weight_col]
            if cum_weight >= target: return x[value_col]
        return sorted_data[-1][value_col]
    p05, median, p95 = get_quantile(0.05), get_quantile(0.50), get_quantile(0.95)
    if total_weight > 1:
        var = sum(x[weight_col] * ((x[value_col] - mean_val) ** 2) for x in sorted_data) / (total_weight - 1)
        stdev = math.sqrt(var)
    else: stdev = None
    cv = (stdev / mean_val) if stdev is not None and mean_val != 0 else None
    return {'mean': mean_val, 'median': median, 'cv': cv, 'p05': p05, 'p95': p95}

def main():
    random.seed(42)
    base_dir = Path("test_data/universal_golden")
    input_dir = base_dir / "inputs"
    truth_dir = base_dir / "ground_truth"
    input_dir.mkdir(parents=True, exist_ok=True)
    truth_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Load Taxonomy
    tree = JolTree.load("test_data/joltax_cache")
    node_lookup = {}
    for i, tid in enumerate(tree._index_to_id):
        node_lookup[int(tid)] = {
            'entry': tree.entry_times[i], 'exit': tree.exit_times[i],
            'depth': tree.depths[i], 'rank': tree.rank_names[tree.ranks[i]],
            'parent': int(tree._index_to_id[tree.parents[i]]),
            'name': tree.get_name(int(tid))
        }

    def is_descendant(child_tid, parent_tid):
        if child_tid == 0: return False
        if child_tid == parent_tid: return True
        try:
            lin = tree.get_lineage(child_tid)
            return parent_tid in lin
        except: return False

    db_minimizers = {int(tid): random.randint(10000, 100000) for tid in tree._index_to_id}

    # 2. Define Biological Scenario
    print("Phase 2: Generating complex biological scenarios and reads...")
    species_idx = tree.rank_names.index("species")
    all_species = [int(tree._index_to_id[i]) for i, r in enumerate(tree.ranks) if r == species_idx]
    
    CORE_TEST_TIDS = [9606, 562, 694009]
    remaining_species = [tid for tid in all_species if tid not in CORE_TEST_TIDS]
    sampled_species = CORE_TEST_TIDS + random.sample(remaining_species, min(47, len(remaining_species)))
    
    higher_ranks = [i for i, n in enumerate(tree.rank_names) if n in ["genus", "family", "order", "phylum"]]
    all_higher = [int(tree._index_to_id[i]) for i, r in enumerate(tree.ranks) if r in higher_ranks]
    sampled_higher = random.sample(all_higher, min(20, len(all_higher)))
    
    target_nodes = sampled_species + sampled_higher + [1]
    all_tids_list = [int(x) for x in tree._index_to_id]
    
    all_reads = []
    for sample in SAMPLES:
        print(f"  -> Processing sample {sample['id']}...")
        sample_targets = set(random.sample(target_nodes, random.randint(40, len(target_nodes))))
        for tid in CORE_TEST_TIDS: sample_targets.add(tid)
        
        num_u = random.randint(20, 50)
        for _ in range(num_u):
            all_reads.append({
                "sample_id": sample["id"], "year": sample["year"], "week": sample["week"],
                "status": "U", "t_id": 0, "kmer_str": "0:35"
            })

        for tid in sample_targets:
            num_reads = random.randint(5, 50)
            lineage = tree.get_lineage(tid)
            clade = tree.get_clade(tid)
            
            # Weighted pool for hits
            pool_tids = list(lineage) + random.sample(all_tids_list, 10)
            weights = [1.0] * len(lineage) + [0.05] * 10
            
            for _ in range(num_reads):
                hit_tid = random.choices(pool_tids, weights=weights, k=1)[0]
                all_reads.append({
                    "sample_id": sample["id"], "year": sample["year"], "week": sample["week"],
                    "status": "C", "t_id": tid, "kmer_str": f"{hit_tid}:35"
                })

    # 3. Write Mock Kraken Files
    print("Phase 3: Writing mock Kraken files...")
    for sample in SAMPLES:
        s_reads = [r for r in all_reads if r["sample_id"] == sample["id"]]
        with open(input_dir / f"{sample['id']}.kraken", "w") as f:
            for r in s_reads:
                f.write(f"{r['status']}\tREAD\t{r['t_id']}\t150\t0\t{r['kmer_str']}\n")

    # 4. Extract Ground Truth Features
    print("Phase 4: Calculating ground truth features...")
    kmer_truth = []
    
    # Pooled hit counts across all samples per t_id
    species_map = {}
    for r in all_reads:
        if r["status"] == "U": continue
        tid = r["t_id"]
        hit_tid, hit_count = r["kmer_str"].split(":")
        hit_tid, hit_count = int(hit_tid), int(hit_count)
        if tid not in species_map: species_map[tid] = {}
        species_map[tid][hit_tid] = species_map[tid].get(hit_tid, 0) + hit_count
        
    for s_tid, hits in species_map.items():
        s_meta = node_lookup[s_tid]
        res = {
            't_id': s_tid, 
            'kmers_global_clade_count': 0, 
            'kmers_global_lineage_count': 0, 
            'kmers_global_misclassified_count': 0, 
            'kmers_global_root_count': 0, 
            'kmers_global_total_count': sum(hits.values())
        }
        dist_data, lineage_dist_data, misc_hits, exc_hits = [], [], {}, {}
        for k_tid, count in hits.items():
            if k_tid == 0: continue
            k_meta = node_lookup[k_tid]; is_clade = is_descendant(k_tid, s_tid); is_lineage = is_descendant(s_tid, k_tid) and (k_tid != s_tid)
            if is_clade: res['kmers_global_clade_count'] += count
            else:
                res['kmers_global_exclade_count'] = res.get('kmers_global_exclade_count', 0) + count; exc_hits[k_tid] = exc_hits.get(k_tid, 0) + count
                if is_lineage:
                    res['kmers_global_lineage_count'] += count; lineage_dist_data.append({'val': k_meta['depth'] / max(int(s_meta['depth']) - 1, 1), 'weight': count})
                else:
                    res['kmers_global_misclassified_count'] += count; misc_hits[k_tid] = misc_hits.get(k_tid, 0) + count
                    lca_id = tree.get_lca(s_tid, k_tid); lca_meta = node_lookup[lca_id]
                    dist = (int(s_meta['depth']) - int(lca_meta['depth'])) + (int(k_meta['depth']) - int(lca_meta['depth']))
                    dist_data.append({'distance': dist, 'kmer_depth': k_meta['depth'], 'relative_lca_depth': lca_meta['depth'] / max(int(s_meta['depth']) - 1, 1), 'weight': count})
            if k_tid == 1 and not is_clade: res['kmers_global_root_count'] += count
        res['kmers_global_classified_count'] = res['kmers_global_clade_count'] + sum(exc_hits.values())
        res['kmers_global_unclassified_count'] = res['kmers_global_total_count'] - res['kmers_global_classified_count']
        c, d_class, d_tot, d_exc, d_misc = res, max(res['kmers_global_classified_count'], 1), max(res['kmers_global_total_count'], 1), max(res.get('kmers_global_exclade_count', 0), 1), max(res['kmers_global_misclassified_count'], 1)
        res.update({
            'kmers_global_cladeVSclassified_ratio': c['kmers_global_clade_count'] / d_class, 
            'kmers_global_lineageVSclassified_ratio': c['kmers_global_lineage_count'] / d_class, 
            'kmers_global_misclassifiedVSclassified_ratio': c['kmers_global_misclassified_count'] / d_class, 
            'kmers_global_rootVSclassified_ratio': c['kmers_global_root_count'] / d_class, 
            
            'kmers_global_cladeVStotal_ratio': c['kmers_global_clade_count'] / d_tot, 
            'kmers_global_classifiedVStotal_ratio': c['kmers_global_classified_count'] / d_tot, 
            'kmers_global_lineageVStotal_ratio': c['kmers_global_lineage_count'] / d_tot, 
            'kmers_global_rootVStotal_ratio': c['kmers_global_root_count'] / d_tot, 
            'kmers_global_misclassifiedVStotal_ratio': c['kmers_global_misclassified_count'] / d_tot, 
            'kmers_global_excladeVStotal_ratio': c.get('kmers_global_exclade_count', 0) / d_tot, 
            'kmers_global_supportingVStotal_ratio': (c['kmers_global_clade_count'] + c['kmers_global_lineage_count']) / d_tot,
            
            'kmers_global_rootVSexclade_ratio': c['kmers_global_root_count'] / d_exc, 
            'kmers_global_lineageVSexclade_ratio': c['kmers_global_lineage_count'] / d_exc,
        })
        for m in ['dist', 'depth', 'relative_lca_depth']:
            map_m = {'dist': 'distance', 'depth': 'kmer_depth', 'relative_lca_depth': 'relative_lca_depth'}
            v_col = map_m[m]
            s = get_weighted_stats([{'v': x[v_col], 'w': x['weight']} for x in dist_data], 'v', 'w')
            res.update({
                f'kmers_global_misclassified_{m}_mean': s['mean'], 
                f'kmers_global_misclassified_{m}_median': s['median'], 
                f'kmers_global_misclassified_{m}_cv': s['cv'], 
                f'kmers_global_misclassified_{m}_p05': s['p05'], 
                f'kmers_global_misclassified_{m}_p95': s['p95']
            })
        lin_res = get_weighted_stats(lineage_dist_data, 'val', 'weight')
        res.update({
            'kmers_global_lineage_relative_depth_mean': lin_res['mean'], 
            'kmers_global_lineage_relative_depth_median': lin_res['median'], 
            'kmers_global_lineage_relative_depth_cv': lin_res['cv'], 
            'kmers_global_lineage_relative_depth_p05': lin_res['p05'], 
            'kmers_global_lineage_relative_depth_p95': lin_res['p95']
        })
        def fmt_top(h_dict, total_sum):
            sorted_h = sorted(h_dict.items(), key=lambda x: (x[1], -x[0]), reverse=True)[:5]
            shares = [round(c / total_sum, 4) for _, c in sorted_h]
            shares_str = ";".join([str(x) if x != int(x) else str(int(x)) for x in shares])
            return ";".join([str(k) for k, _ in sorted_h]), ";".join([tree.get_name(k) or "Unknown" for k, _ in sorted_h]), shares_str
        res['kmers_global_misclassified_top5_taxids'], res['kmers_global_misclassified_top5_names'], res['kmers_global_misclassified_top5_shares'] = fmt_top(misc_hits, d_misc)
        res['kmers_global_exclade_top5_taxids'], res['kmers_global_exclade_top5_names'], res['kmers_global_exclade_top5_shares'] = fmt_top(exc_hits, d_exc)
        kmer_truth.append(res)
    kmer_truth.sort(key=lambda x: x['t_id'])
    if kmer_truth:
        all_keys = set()
        for r in kmer_truth: all_keys.update(r.keys())
        headers = sorted(list(all_keys)) # Sorted for consistency
        with open(truth_dir / "golden_kmer_features.csv", "w", newline='') as f:
            writer = csv.DictWriter(f, fieldnames=headers); writer.writeheader()
            for r in kmer_truth:
                row_copy = {h: "" for h in headers}
                for k, v in r.items():
                    if isinstance(v, float) and not math.isnan(v): row_copy[k] = f"{v:.6f}"
                    elif v is None: row_copy[k] = ""
                    else: row_copy[k] = v
                writer.writerow(row_copy)

    # 5. Extract Sample Ground Truth Features
    print("Phase 5: Calculating sample ground truth features...")
    sample_truth = []
    
    sample_species_map = {}
    for r in all_reads:
        if r["status"] == "U": continue
        s_id = r["sample_id"]
        raw_tid = r["t_id"]
        
        lineage = tree.get_lineage(raw_tid)
        species_idx = tree.rank_names.index('species') if 'species' in tree.rank_names else -1
        tid = None
        for anc in lineage:
            idx = tree._get_indices(np.array([anc], dtype=np.uint32))[0]
            if idx != -1 and tree.ranks[idx] == species_idx:
                tid = anc
                break
                
        if tid is None:
            continue
            
        hit_tid, hit_count = r["kmer_str"].split(":")
        hit_tid, hit_count = int(hit_tid), int(hit_count)
        
        if s_id not in sample_species_map: sample_species_map[s_id] = {}
        if tid not in sample_species_map[s_id]: sample_species_map[s_id][tid] = {}
        sample_species_map[s_id][tid][hit_tid] = sample_species_map[s_id][tid].get(hit_tid, 0) + hit_count

    for s_id, taxa_hits in sample_species_map.items():
        for s_tid, hits in taxa_hits.items():
            s_meta = node_lookup[s_tid]
            res = {
                'sample_id': s_id,
                't_id': s_tid, 
                'kmers_sample_clade_count': 0, 
                'kmers_sample_lineage_count': 0, 
                'kmers_sample_misclassified_count': 0, 
                'kmers_sample_root_count': 0, 
                'kmers_sample_total_count': sum(hits.values())
            }
            dist_data, lineage_dist_data, misc_hits, exc_hits = [], [], {}, {}
            for k_tid, count in hits.items():
                if k_tid == 0: continue
                k_meta = node_lookup[k_tid]; is_clade = is_descendant(k_tid, s_tid); is_lineage = is_descendant(s_tid, k_tid) and (k_tid != s_tid)
                if is_clade: res['kmers_sample_clade_count'] += count
                else:
                    res['kmers_sample_exclade_count'] = res.get('kmers_sample_exclade_count', 0) + count; exc_hits[k_tid] = exc_hits.get(k_tid, 0) + count
                    if is_lineage:
                        res['kmers_sample_lineage_count'] += count; lineage_dist_data.append({'val': k_meta['depth'] / max(int(s_meta['depth']) - 1, 1), 'weight': count})
                    else:
                        res['kmers_sample_misclassified_count'] += count; misc_hits[k_tid] = misc_hits.get(k_tid, 0) + count
                        lca_id = tree.get_lca(s_tid, k_tid); lca_meta = node_lookup[lca_id]
                        dist = (int(s_meta['depth']) - int(lca_meta['depth'])) + (int(k_meta['depth']) - int(lca_meta['depth']))
                        dist_data.append({'distance': dist, 'kmer_depth': k_meta['depth'], 'relative_lca_depth': lca_meta['depth'] / max(int(s_meta['depth']) - 1, 1), 'weight': count})
                if k_tid == 1 and not is_clade: res['kmers_sample_root_count'] += count
            
            res['kmers_sample_classified_count'] = res['kmers_sample_clade_count'] + sum(exc_hits.values())
            res['kmers_sample_unclassified_count'] = res['kmers_sample_total_count'] - res['kmers_sample_classified_count']
            
            c, d_class, d_tot, d_exc = res, res['kmers_sample_classified_count'], res['kmers_sample_total_count'], res.get('kmers_sample_exclade_count', 0)
            
            res.update({
                'kmers_sample_cladeVSclassified_ratio': c['kmers_sample_clade_count'] / d_class if d_class > 0 else None, 
                'kmers_sample_lineageVSclassified_ratio': c['kmers_sample_lineage_count'] / d_class if d_class > 0 else None, 
                'kmers_sample_misclassifiedVSclassified_ratio': c['kmers_sample_misclassified_count'] / d_class if d_class > 0 else None, 
                'kmers_sample_rootVSclassified_ratio': c['kmers_sample_root_count'] / d_class if d_class > 0 else None, 
                
                'kmers_sample_cladeVStotal_ratio': c['kmers_sample_clade_count'] / d_tot if d_tot > 0 else None, 
                'kmers_sample_classifiedVStotal_ratio': c['kmers_sample_classified_count'] / d_tot if d_tot > 0 else None, 
                'kmers_sample_lineageVStotal_ratio': c['kmers_sample_lineage_count'] / d_tot if d_tot > 0 else None, 
                'kmers_sample_rootVStotal_ratio': c['kmers_sample_root_count'] / d_tot if d_tot > 0 else None, 
                'kmers_sample_misclassifiedVStotal_ratio': c['kmers_sample_misclassified_count'] / d_tot if d_tot > 0 else None, 
                'kmers_sample_excladeVStotal_ratio': c.get('kmers_sample_exclade_count', 0) / d_tot if d_tot > 0 else None, 
                'kmers_sample_supportingVStotal_ratio': (c['kmers_sample_clade_count'] + c['kmers_sample_lineage_count']) / d_tot if d_tot > 0 else None,
                
                'kmers_sample_rootVSexclade_ratio': c['kmers_sample_root_count'] / d_exc if d_exc > 0 else None, 
                'kmers_sample_lineageVSexclade_ratio': c['kmers_sample_lineage_count'] / d_exc if d_exc > 0 else None,
            })
            
            for m in ['dist', 'depth', 'relative_lca_depth']:
                map_m = {'dist': 'distance', 'depth': 'kmer_depth', 'relative_lca_depth': 'relative_lca_depth'}
                v_col = map_m[m]
                s = get_weighted_stats([{'v': x[v_col], 'w': x['weight']} for x in dist_data], 'v', 'w')
                res.update({
                    f'kmers_sample_misclassified_{m}_mean': s['mean'], 
                    f'kmers_sample_misclassified_{m}_median': s['median'], 
                    f'kmers_sample_misclassified_{m}_cv': s['cv'], 
                    f'kmers_sample_misclassified_{m}_p05': s['p05'], 
                    f'kmers_sample_misclassified_{m}_p95': s['p95']
                })
                
            lin_res = get_weighted_stats(lineage_dist_data, 'val', 'weight')
            res.update({
                'kmers_sample_lineage_relative_depth_mean': lin_res['mean'], 
                'kmers_sample_lineage_relative_depth_median': lin_res['median'], 
                'kmers_sample_lineage_relative_depth_cv': lin_res['cv'], 
                'kmers_sample_lineage_relative_depth_p05': lin_res['p05'], 
                'kmers_sample_lineage_relative_depth_p95': lin_res['p95']
            })
            
            def fmt_top(h_dict, total_sum):
                sorted_h = sorted(h_dict.items(), key=lambda x: (x[1], -x[0]), reverse=True)[:3]
                shares = [round(c / total_sum, 4) for _, c in sorted_h]
                shares_str = ";".join([str(x) if x != int(x) else str(int(x)) for x in shares])
                return ";".join([str(k) for k, _ in sorted_h]), ";".join([tree.get_name(k) or "Unknown" for k, _ in sorted_h]), shares_str
            
            res['kmers_sample_misclassified_top3_taxids'], res['kmers_sample_misclassified_top3_names'], res['kmers_sample_misclassified_top3_shares'] = fmt_top(misc_hits, max(c['kmers_sample_misclassified_count'], 1))
            res['kmers_sample_exclade_top3_taxids'], res['kmers_sample_exclade_top3_names'], res['kmers_sample_exclade_top3_shares'] = fmt_top(exc_hits, max(c.get('kmers_sample_exclade_count', 0), 1))
            
            sample_truth.append(res)
            
    sample_truth.sort(key=lambda x: (x['sample_id'], x['t_id']))
    if sample_truth:
        all_keys = set()
        for r in sample_truth: all_keys.update(r.keys())
        headers = sorted(list(all_keys))
        with open(truth_dir / "golden_kmer_sample_features.csv", "w", newline='') as f:
            writer = csv.DictWriter(f, fieldnames=headers); writer.writeheader()
            for r in sample_truth:
                row_copy = {h: "" for h in headers}
                for k, v in r.items():
                    if isinstance(v, float) and not math.isnan(v): row_copy[k] = f"{v:.6f}"
                    elif v is None: row_copy[k] = ""
                    else: row_copy[k] = v
                writer.writerow(row_copy)

    # 6. Extract Stability Ground Truth Features
    print("Phase 6: Calculating stability ground truth features...")
    stability_truth = []
    
    TARGET_RATIOS = [
        "kmers_sample_cladeVSclassified_ratio",
        "kmers_sample_lineageVSclassified_ratio",
        "kmers_sample_misclassifiedVSclassified_ratio",
        "kmers_sample_rootVSclassified_ratio",
        "kmers_sample_cladeVStotal_ratio",
        "kmers_sample_classifiedVStotal_ratio",
        "kmers_sample_lineageVStotal_ratio",
        "kmers_sample_rootVStotal_ratio",
        "kmers_sample_misclassifiedVStotal_ratio",
        "kmers_sample_excladeVStotal_ratio",
        "kmers_sample_supportingVStotal_ratio",
        "kmers_sample_rootVSexclade_ratio",
        "kmers_sample_lineageVSexclade_ratio"
    ]
    
    total_samples = len(SAMPLES)
    
    taxon_sample_metrics = {}
    for r in sample_truth:
        tid = r['t_id']
        if tid not in taxon_sample_metrics:
            taxon_sample_metrics[tid] = {ratio: [] for ratio in TARGET_RATIOS}
            taxon_sample_metrics[tid]['samples'] = []
        taxon_sample_metrics[tid]['samples'].append(r['sample_id'])
        for ratio in TARGET_RATIOS:
            val = r.get(ratio)
            if val is not None:
                taxon_sample_metrics[tid][ratio].append(val)
                
    for tid, data in taxon_sample_metrics.items():
        res = {
            't_id': tid,
            'kmers_stability_occupancy_ratio': len(data['samples']) / total_samples
        }
        
        for ratio in TARGET_RATIOS:
            base_name = ratio.replace("kmers_sample_", "kmers_stability_")
            vals = data[ratio]
            valid_n = len(vals)
            
            res[f'{base_name}_presence'] = valid_n / len(data['samples']) if len(data['samples']) > 0 else 0.0
            
            if valid_n > 0:
                s_vals = sorted(vals)
                mean_val = sum(s_vals) / valid_n
                
                p05 = np.percentile(s_vals, 5, method='nearest') if valid_n > 0 else None
                median = np.median(s_vals) if valid_n > 0 else None
                p95 = np.percentile(s_vals, 95, method='nearest') if valid_n > 0 else None
                
                res[f'{base_name}_mean'] = mean_val
                res[f'{base_name}_median'] = float(median)
                res[f'{base_name}_p05'] = float(p05)
                res[f'{base_name}_p95'] = float(p95)
                
                if valid_n > 1:
                    var = sum((x - mean_val) ** 2 for x in s_vals) / (valid_n - 1)
                    stdev = math.sqrt(var)
                else:
                    stdev = None
                    
                res[f'{base_name}_cv'] = (stdev / mean_val) if stdev is not None and mean_val != 0 else None
            else:
                res[f'{base_name}_mean'] = None
                res[f'{base_name}_median'] = None
                res[f'{base_name}_p05'] = None
                res[f'{base_name}_p95'] = None
                res[f'{base_name}_cv'] = None
                
        stability_truth.append(res)
        
    stability_truth.sort(key=lambda x: x['t_id'])
    if stability_truth:
        all_keys = set()
        for r in stability_truth: all_keys.update(r.keys())
        headers = sorted(list(all_keys))
        with open(truth_dir / "golden_kmer_stability_features.csv", "w", newline='') as f:
            writer = csv.DictWriter(f, fieldnames=headers); writer.writeheader()
            for r in stability_truth:
                row_copy = {h: "" for h in headers}
                for k, v in r.items():
                    if isinstance(v, float) and not math.isnan(v): row_copy[k] = f"{v:.6f}"
                    elif v is None: row_copy[k] = ""
                    else: row_copy[k] = v
                writer.writerow(row_copy)

    # 7. Extract Abundance Ground Truth Features
    print("Phase 7: Calculating abundance ground truth features...")
    
    # First, we need a mock abundance table. 
    # We'll use the clade_reads counts from the reports as a base, then CLR-transform them.
    # Actually, the user said it could be proportions too. Let's use simple proportions
    # to avoid complex zero-replacement logic in the mock generator.
    
    # 7.1 Calculate proportions
    sample_totals = {}
    for r in all_reads:
        sid = r["sample_id"]
        sample_totals[sid] = sample_totals.get(sid, 0) + 1
        
    abundance_data = {} # {tid: {sid: proportion}}
    for r in all_reads:
        if r["status"] == "U": continue
        tid = r["t_id"]
        sid = r["sample_id"]
        if tid not in abundance_data: abundance_data[tid] = {}
        abundance_data[tid][sid] = abundance_data[tid].get(sid, 0) + (1.0 / sample_totals[sid])
        
    all_tids_in_abundance = sorted(list(abundance_data.keys()))
    all_sids = [s["id"] for s in SAMPLES]
    
    # Write mock abundance table
    with open(input_dir / "golden_abundance_table.csv", "w", newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["t_id"] + all_sids)
        for tid in all_tids_in_abundance:
            row = [tid]
            for sid in all_sids:
                row.append(f"{abundance_data[tid].get(sid, 0.0):.8f}")
            writer.writerow(row)
            
    # 7.2 Calculate stats
    abundance_truth = []
    for tid in all_tids_in_abundance:
        vals = [abundance_data[tid].get(sid, 0.0) for sid in all_sids]
        s_vals = sorted(vals)
        n = len(s_vals)
        mean_val = sum(s_vals) / n
        median = np.median(s_vals)
        p05 = np.percentile(s_vals, 5, method='nearest')
        p95 = np.percentile(s_vals, 95, method='nearest')
        
        var = sum((x - mean_val) ** 2 for x in s_vals) / (n - 1) if n > 1 else 0.0
        stdev = math.sqrt(var)
        cv = (stdev / mean_val) if mean_val != 0 else 0.0
        
        abundance_truth.append({
            't_id': tid,
            'abundance_global_mean': mean_val,
            'abundance_global_median': float(median),
            'abundance_global_p05': float(p05),
            'abundance_global_p95': float(p95),
            'abundance_global_cv': cv
        })
        
    if abundance_truth:
        headers = ['t_id', 'abundance_global_mean', 'abundance_global_median', 'abundance_global_p05', 'abundance_global_p95', 'abundance_global_cv']
        with open(truth_dir / "golden_abundance_features.csv", "w", newline='') as f:
            writer = csv.DictWriter(f, fieldnames=headers); writer.writeheader()
            for r in abundance_truth:
                row_copy = {}
                for k, v in r.items():
                    if k == 't_id': row_copy[k] = v
                    elif isinstance(v, float): row_copy[k] = f"{v:.6f}"
                    else: row_copy[k] = v
                writer.writerow(row_copy)

    print(f"Universal Golden Dataset built successfully in {base_dir}")

if __name__ == "__main__":
    main()
