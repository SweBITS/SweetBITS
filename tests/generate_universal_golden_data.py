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
        c, p = node_lookup[child_tid], node_lookup[parent_tid]
        return (c['entry'] >= p['entry']) and (c['exit'] <= p['exit'])

    # Pre-define fixed DB minimizer counts for inspect simulation
    db_minimizers = {int(tid): random.randint(10000, 100000) for tid in tree._index_to_id}

    # 2. Define Biological Scenario
    print("Phase 2: Generating complex biological scenarios and reads...")
    species_idx = tree.rank_names.index("species")
    all_species = [int(tree._index_to_id[i]) for i, r in enumerate(tree.ranks) if r == species_idx]
    
    # ALWAYS include some stable taxa for test assertions
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
        # Increase target count to ensure core species are likely present in most samples
        sample_targets = set(random.sample(target_nodes, random.randint(40, len(target_nodes))))
        # Force inclusion of core species in every sample for correlation
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
            
            lineage_set = set(lineage)
            clade_set = set(clade)
            
            for _ in range(num_reads):
                k_hits = []
                for _ in range(random.randint(1, 2)):
                    k_hits.append((random.choice(clade), random.randint(10, 30)))
                for _ in range(random.randint(1, 3)):
                    k_hits.append((random.choice(lineage), random.randint(2, 10)))
                
                if tid != 1:
                    for _ in range(random.randint(1, 3)):
                        misc_tid = random.choice(all_tids_list)
                        attempts = 0
                        while (misc_tid in lineage_set or misc_tid in clade_set) and attempts < 100:
                            misc_tid = random.choice(all_tids_list)
                            attempts += 1
                        k_hits.append((misc_tid, random.randint(1, 5)))
                
                k_hits.append((0, random.randint(1, 10)))
                random.shuffle(k_hits)
                l1, l2 = random.randint(50, 150), random.randint(50, 150)
                
                all_reads.append({
                    "sample_id": sample["id"], "year": sample["year"], "week": sample["week"],
                    "status": "C", "t_id": tid, "length": f"{l1}|{l2}", "mhg": random.randint(1, 20),
                    "kmer_str": " ".join([f"{k}:{c}" for k, c in k_hits]),
                    "kmer_hits_list": k_hits
                })

    # 3. Write Input Files
    print("Phase 3: Writing Kraken classification and report files...")
    for sample in SAMPLES:
        s_reads = [r for r in all_reads if r["sample_id"] == sample["id"]]
        
        with open(input_dir / f"{sample['id']}.kraken", "w") as f:
            for i, r in enumerate(s_reads):
                if r["status"] == "U":
                    f.write(f"U\tR{i}\t0\t150|150\t0\t0:35\n")
                else:
                    f.write(f"C\tR{i}\t{r['t_id']}\t{r['length']}\t{r['mhg']}\t{r['kmer_str']}\n")
        
        counts = {}
        for r in s_reads:
            if r["status"] == "C":
                t = r["t_id"]
                counts[t] = counts.get(t, 0) + 1
        
        report_clade = {}
        for tid, count in counts.items():
            lin = tree.get_lineage(tid)
            for node in lin:
                report_clade[node] = report_clade.get(node, 0) + count
        
        with open(input_dir / f"{sample['id']}.report", "w") as f:
            for node in sorted(report_clade.keys()):
                m = node_lookup[node]
                # To test correlations, we need mm_uniq to have a relationship with clade_reads
                # Let's say mm_uniq = 0.05 * clade_reads (clamped by DB total)
                expected_uniq = int(min(report_clade[node] * 0.05, db_minimizers[node] * 0.9))
                # Add some small deterministic noise based on sample_id and tid for variety
                noise = (hash(sample['id']) + node) % 5
                mm_uniq = max(1, expected_uniq + noise)
                mm_tot = mm_uniq * 2
                f.write(f"0.0\t{report_clade[node]}\t{counts.get(node, 0)}\t{mm_tot}\t{mm_uniq}\t{m['rank']}\t{node}\t{m['name']}\n")

    # 4. Generate Ground Truth Tables
    print("Phase 4: Generating ground truth tables...")
    periods = sorted(list(set(f"{s['year']}_{s['week']:02d}" for s in SAMPLES)))
    
    clade_sums_truth = {}
    for r in all_reads:
        if r["status"] == "U": continue
        p = f"{r['year']}_{r['week']:02d}"
        lin = tree.get_lineage(r["t_id"])
        for node in lin:
            key = (node, p)
            clade_sums_truth[key] = clade_sums_truth.get(key, 0) + 1

    def write_csv(name, data_dict, filter_empty_cols=True):
        if not data_dict: return
        active_periods = periods
        if filter_empty_cols:
            active_periods = [p for p in periods if any(data_dict[tid].get(p, 0) > 0 for tid in data_dict)]
            if not active_periods: active_periods = [periods[0]]
        with open(truth_dir / name, "w", newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["t_id"] + active_periods)
            for tid in sorted(data_dict.keys()):
                writer.writerow([tid] + [data_dict[tid].get(p, 0) for p in active_periods])

    all_present_nodes = sorted(list(set(k[0] for k in clade_sums_truth.keys())))
    taxon_raw = {}
    direct_counts = {}
    for r in all_reads:
        if r["status"] == "U": continue
        key = (r['t_id'], f"{r['year']}_{r['week']:02d}")
        direct_counts[key] = direct_counts.get(key, 0) + 1

    for tid in all_present_nodes:
        taxon_raw[tid] = {p: direct_counts.get((tid, p), 0) for p in periods}
    write_csv("abundance_taxon.csv", taxon_raw, filter_empty_cols=False)

    clade_raw = {}
    for tid in all_present_nodes:
        clade_raw[tid] = {p: clade_sums_truth.get((tid, p), 0) for p in periods}
    write_csv("abundance_clade.csv", clade_raw, filter_empty_cols=False)

    canonical_raw = {}
    for r in all_reads:
        if r["status"] == "U": continue
        p = f"{r['year']}_{r['week']:02d}"
        lin = tree.get_lineage(r["t_id"])
        target = None
        for node in reversed(lin):
            if node_lookup[node]['rank'] in CANONICAL_RANKS:
                target = node
                break
        if target:
            if target not in canonical_raw: canonical_raw[target] = {}
            canonical_raw[target][p] = canonical_raw[target].get(p, 0) + 1
    write_csv("abundance_canonical.csv", canonical_raw, filter_empty_cols=False)

    def generate_filtered_truth(filter_type, threshold):
        current_counts = {tid: {p: direct_counts.get((tid, p), 0) for p in periods} for tid in all_present_nodes}
        leaf_to_root = sorted(all_present_nodes, key=lambda x: node_lookup[x]['depth'], reverse=True)
        final_kept = {}
        for tid in leaf_to_root:
            node_clade_counts = clade_raw[tid]
            passes = False
            if filter_type == "reads":
                passes = any(node_clade_counts.get(p, 0) >= threshold for p in periods)
            elif filter_type == "obs":
                obs_count = sum(1 for p in periods if node_clade_counts.get(p, 0) > 0)
                passes = obs_count >= threshold
            if passes or tid == 1:
                final_kept[tid] = current_counts[tid]
            else:
                parent_id = node_lookup[tid]['parent']
                if parent_id in current_counts:
                    for p in periods:
                        current_counts[parent_id][p] += current_counts[tid][p]
        return final_kept

    write_csv("abundance_read_filtered_20.csv", generate_filtered_truth("reads", 20), filter_empty_cols=False)
    write_csv("abundance_obs_filtered_2.csv", generate_filtered_truth("obs", 2), filter_empty_cols=False)

    # G. Inspect Truth
    with open(truth_dir / "kraken_inspect.csv", "w", newline='') as f:
        writer = csv.writer(f, delimiter=';')
        writer.writerow(["tax_id", "clade_minimizers"])
        for tid, mm in db_minimizers.items():
            writer.writerow([tid, mm])

    # 5. Global K-mer Features Truth
    print("Phase 5: Generating global k-mer feature truth...")
    species_map = {}
    for r in all_reads:
        if r["status"] == "U": continue
        target_tid = r["t_id"]
        lin = tree.get_lineage(target_tid)
        species_root = None
        for node in reversed(lin):
            if node_lookup[node]['rank'] == "species":
                species_root = node
                break
        if not species_root: continue
        if species_root not in species_map: species_map[species_root] = {}
        for k_tid, k_count in r["kmer_hits_list"]:
            species_map[species_root][k_tid] = species_map[species_root].get(k_tid, 0) + k_count
    
    kmer_truth = []
    for s_tid, hits in species_map.items():
        s_meta = node_lookup[s_tid]
        res = {'t_id': s_tid, 'grand_clade_kmers': 0, 'grand_lineage_kmers': 0, 'grand_misclassified_kmers': 0, 
               'grand_root_kmers': 0, 'grand_total_kmers': 0}
        for k_tid, count in hits.items():
             res['grand_total_kmers'] += count
        dist_data, lineage_dist_data = [], []
        misc_hits, exc_hits = {}, {}
        for k_tid, count in hits.items():
            if k_tid == 0: continue
            k_meta = node_lookup[k_tid]
            is_clade = is_descendant(k_tid, s_tid)
            is_lineage = is_descendant(s_tid, k_tid) and (k_tid != s_tid)
            if is_clade: res['grand_clade_kmers'] += count
            else:
                res['grand_exclade_kmers'] = res.get('grand_exclade_kmers', 0) + count
                exc_hits[k_tid] = exc_hits.get(k_tid, 0) + count
                if is_lineage:
                    res['grand_lineage_kmers'] += count
                    lineage_dist_data.append({'val': k_meta['depth'] / max(int(s_meta['depth']) - 1, 1), 'weight': count})
                else:
                    res['grand_misclassified_kmers'] += count
                    misc_hits[k_tid] = misc_hits.get(k_tid, 0) + count
                    lca_id = tree.get_lca(s_tid, k_tid)
                    lca_meta = node_lookup[lca_id]
                    dist = (int(s_meta['depth']) - int(lca_meta['depth'])) + (int(k_meta['depth']) - int(lca_meta['depth']))
                    dist_data.append({'distance': dist, 'kmer_depth': k_meta['depth'], 
                                    'relative_lca_depth': lca_meta['depth'] / max(int(s_meta['depth']) - 1, 1), 'weight': count})
            if k_tid == 1 and not is_clade: res['grand_root_kmers'] += count
        res['grand_classified_kmers'] = res['grand_clade_kmers'] + sum(exc_hits.values())
        res['grand_unclassified_kmers'] = res['grand_total_kmers'] - res['grand_classified_kmers']
        c = res
        d_class, d_tot, d_exc, d_misc = max(c['grand_classified_kmers'], 1), max(c['grand_total_kmers'], 1), max(c['grand_exclade_kmers'], 1), max(c['grand_misclassified_kmers'], 1)
        res.update({
            'grand_clade_to_classified_kmer_ratio': c['grand_clade_kmers'] / d_class,
            'grand_lineage_to_classified_kmer_ratio': c['grand_lineage_kmers'] / d_class,
            'grand_misclassified_to_classified_kmer_ratio': c['grand_misclassified_kmers'] / d_class,
            'grand_root_to_classified_kmer_ratio': c['grand_root_kmers'] / d_class,
            'grand_supporting_to_misclassified_kmer_ratio': (c['grand_clade_kmers'] + c['grand_lineage_kmers']) / d_misc if c['grand_misclassified_kmers'] > 0 else 1.0,
            'grand_clade_to_total_kmer_ratio': c['grand_clade_kmers'] / d_tot,
            'grand_classified_to_total_kmer_ratio': c['grand_classified_kmers'] / d_tot,
            'grand_lineage_to_total_kmer_ratio': c['grand_lineage_kmers'] / d_tot,
            'grand_root_to_total_kmer_ratio': c['grand_root_kmers'] / d_tot,
            'grand_misclassified_to_total_kmer_ratio': c['grand_misclassified_kmers'] / d_tot,
            'grand_exclade_to_total_kmer_ratio': c['grand_exclade_kmers'] / d_tot,
            'grand_supporting_to_total_kmer_ratio': (c['grand_clade_kmers'] + c['grand_lineage_kmers']) / d_tot,
            'grand_root_to_exclade_kmer_ratio': c['grand_root_kmers'] / d_exc,
            'grand_lineage_to_exclade_kmer_ratio': c['grand_lineage_kmers'] / d_exc,
        })
        def prep_stats(data_list, val_col, out_suffix):
            r = get_weighted_stats(data_list, val_col, 'weight')
            return {f"mean_grand_misclassified_kmer_{out_suffix}": r['mean'], f"median_grand_misclassified_kmer_{out_suffix}": r['median'], f"cv_grand_misclassified_kmer_{out_suffix}": r['cv'], f"p05_grand_misclassified_kmer_{out_suffix}": r['p05'], f"p95_grand_misclassified_kmer_{out_suffix}": r['p95']}
        res.update(prep_stats(dist_data, 'distance', 'distance'))
        res.update(prep_stats(dist_data, 'kmer_depth', 'depth'))
        res.update(prep_stats(dist_data, 'relative_lca_depth', 'relative_lca_depth'))
        lin_res = get_weighted_stats(lineage_dist_data, 'val', 'weight')
        res.update({'mean_grand_lineage_kmer_relative_depth': lin_res['mean'], 'median_grand_lineage_kmer_relative_depth': lin_res['median'], 'cv_grand_lineage_kmer_relative_depth': lin_res['cv'], 'p05_grand_lineage_kmer_relative_depth': lin_res['p05'], 'p95_grand_lineage_kmer_relative_depth': lin_res['p95']})
        def fmt_top(h_dict, total_sum):
            sorted_h = sorted(h_dict.items(), key=lambda x: (x[1], -x[0]), reverse=True)[:5]
            return ";".join([str(k) for k, _ in sorted_h]), ";".join([tree.get_name(k) or "Unknown" for k, _ in sorted_h]), ";".join([str(round(c/total_sum, 4)) for _, c in sorted_h])
        res['grand_top_5_misclassified_kmer_tax_ids'], res['grand_top_5_misclassified_kmer_names'], res['grand_top_5_misclassified_kmer_shares'] = fmt_top(misc_hits, d_misc)
        res['grand_top_5_exclade_kmer_tax_ids'], res['grand_top_5_exclade_kmer_names'], res['grand_top_5_exclade_kmer_shares'] = fmt_top(exc_hits, d_exc)
        kmer_truth.append(res)
    kmer_truth.sort(key=lambda x: x['t_id'])
    if kmer_truth:
        headers = list(kmer_truth[0].keys())
        with open(truth_dir / "golden_kmer_features.csv", "w", newline='') as f:
            writer = csv.DictWriter(f, fieldnames=headers)
            writer.writeheader()
            for r in kmer_truth:
                row_copy = r.copy()
                for k, v in row_copy.items():
                    if isinstance(v, float) and not math.isnan(v): row_copy[k] = f"{v:.6f}"
                    elif v is None: row_copy[k] = ""
                writer.writerow(row_copy)
    print(f"Universal Golden Dataset built successfully in {base_dir}")

if __name__ == "__main__":
    main()
