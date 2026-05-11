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
            'kmers_global_supportingVSmisclassified_ratio': (c['kmers_global_clade_count'] + c['kmers_global_lineage_count']) / d_misc if c['kmers_global_misclassified_count'] > 0 else 1.0,
            
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
    print(f"Universal Golden Dataset built successfully in {base_dir}")

if __name__ == "__main__":
    main()
