import os
import random
import math
import csv
import numpy as np
from pathlib import Path
from joltax import JolTree

def get_weighted_stats(data, value_col, weight_col):
    if len(data) == 0:
        return {
            'mean': None, 'median': None, 'cv': None, 
            'p05': None, 'p95': None
        }
        
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
            if cum_weight >= target:
                return x[value_col]
        return sorted_data[-1][value_col]
    
    p05 = get_quantile(0.05)
    median = get_quantile(0.50)
    p95 = get_quantile(0.95)
    
    if total_weight > 1:
        variance = sum(x[weight_col] * ((x[value_col] - mean_val) ** 2) for x in sorted_data) / (total_weight - 1)
        stdev = math.sqrt(variance)
    else:
        stdev = None
        
    cv = (stdev / mean_val) if stdev is not None and mean_val != 0 else None
    
    return {
        'mean': mean_val,
        'median': median,
        'cv': cv,
        'p05': p05,
        'p95': p95
    }

def main():
    random.seed(42)
    tax_dir = Path("test_data/joltax_cache")
    out_dir = Path("tests/golden_data")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    tree = JolTree.load(str(tax_dir))
    
    # Get 20 species
    species_idx = tree.rank_names.index("species")
    species_tids = [tree._index_to_id[i] for i, rank in enumerate(tree.ranks) if rank == species_idx]
    target_tids = random.sample(species_tids, min(20, len(species_tids)))
    
    samples = [f"Ki-2024_01_{i:03d}" for i in range(1, 6)]
    sample_files = {s: open(out_dir / f"{s}.kraken", "w") for s in samples}
    
    all_hits = []
    
    read_id_counter = 1
    for t_id in target_tids:
        t_id = int(t_id)
        lineage = tree.get_lineage(t_id)
        
        for sample in samples:
            # Generate 3-10 reads per sample for this target
            num_reads = random.randint(3, 10)
            for _ in range(num_reads):
                kmer_hits = []
                
                # Clade hit
                clade_count = random.randint(10, 50)
                kmer_hits.append((t_id, clade_count))
                
                # Lineage hit (Root or random ancestor)
                lin_target = random.choice(lineage[:-1]) if len(lineage) > 1 else 1
                lin_count = random.randint(2, 10)
                kmer_hits.append((lin_target, lin_count))
                
                # Misclassified hit (can hit ANY node in the tree: species, phylum, domain, etc.)
                all_tids = tree._index_to_id
                misc_target = int(random.choice(all_tids))
                while misc_target in lineage:
                    misc_target = int(random.choice(all_tids))
                misc_count = random.randint(1, 5)
                kmer_hits.append((misc_target, misc_count))
                
                # Unclassified hit (0)
                unclass_count = random.randint(0, 5)
                if unclass_count > 0:
                    kmer_hits.append((0, unclass_count))
                    
                # Format kmer string
                kmer_str = " ".join([f"{k}:{c}" for k, c in kmer_hits])
                
                # Write to file
                sample_files[sample].write(f"C\tR{read_id_counter}\t{t_id}\t150|150\t10\t{kmer_str}\n")
                read_id_counter += 1
                
                # Store for independent verification
                for k_tid, k_count in kmer_hits:
                    all_hits.append({
                        't_id': t_id,
                        'k_tid': k_tid,
                        'count': k_count
                    })
                    
    for f in sample_files.values():
        f.close()
        
    # --- INDEPENDENT VERIFICATION MATH ---
    # Aggregate hits across samples (like Phase 2)
    agg_hits = {}
    for hit in all_hits:
        key = (hit['t_id'], hit['k_tid'])
        agg_hits[key] = agg_hits.get(key, 0) + hit['count']
        
    # Group by t_id
    grouped_hits = {}
    for (t_id, k_tid), count in agg_hits.items():
        if t_id not in grouped_hits:
            grouped_hits[t_id] = []
        grouped_hits[t_id].append({'k_tid': k_tid, 'count': count})
    
    # Get node mapping data for O(1) checks
    # entry/exit times allow determining ancestry instantly
    indices = tree._get_indices(np.array(tree._index_to_id, dtype=np.uint32))
    node_lookup = {}
    for i, tid in enumerate(tree._index_to_id):
        node_lookup[int(tid)] = {
            'entry': tree.entry_times[i],
            'exit': tree.exit_times[i],
            'depth': tree.depths[i]
        }
    
    golden_results = []
    
    for t_id, hits in grouped_hits.items():
        t_meta = node_lookup[t_id]
        
        counts = {
            'grand_clade_kmers': 0,
            'grand_exclade_kmers': 0,
            'grand_lineage_kmers': 0,
            'grand_root_kmers': 0,
            'grand_total_kmers': 0,
        }
        
        dist_data = []
        lineage_dist_data = []
        misclassified_data = [] # For top hits
        exclade_data = []
        
        for row in hits:
            k_tid = int(row['k_tid'])
            count = int(row['count'])
            
            counts['grand_total_kmers'] += count
            
            if k_tid == 0:
                continue # Unclassified
                
            k_meta = node_lookup.get(k_tid)
            if not k_meta:
                continue # Should not happen with valid cache
                
            # is_in_clade: k_tid is descendant of t_id (or t_id itself)
            is_in_clade = (k_meta['entry'] >= t_meta['entry']) and (k_meta['exit'] <= t_meta['exit'])
            
            # is_in_lineage: t_id is descendant of k_tid (k_tid is ancestor)
            # strictly ancestor: k_tid != t_id
            is_in_lineage = (t_meta['entry'] >= k_meta['entry']) and (t_meta['exit'] <= k_meta['exit']) and (k_tid != t_id)
            
            if is_in_clade:
                counts['grand_clade_kmers'] += count
            else:
                counts['grand_exclade_kmers'] += count
                exclade_data.append({'kmer_tax_id': k_tid, 'kmer_count': count})
                if is_in_lineage:
                    counts['grand_lineage_kmers'] += count
                    lineage_dist_data.append({'val': k_meta['depth'] / max(t_meta['depth'] - 1, 1), 'weight': count})
                else:
                    misclassified_data.append({'kmer_tax_id': k_tid, 'kmer_count': count})
                    
                    # LCA
                    lca_id = tree.get_lca(t_id, k_tid)
                    lca_meta = node_lookup[lca_id]
                    
                    distance = (t_meta['depth'] - lca_meta['depth']) + (k_meta['depth'] - lca_meta['depth'])
                    rel_lca = lca_meta['depth'] / max(t_meta['depth'] - 1, 1)
                    
                    dist_data.append({
                        'distance': distance,
                        'kmer_depth': k_meta['depth'],
                        'relative_lca_depth': rel_lca,
                        'weight': count
                    })
                    
            if k_tid == 1 and not is_in_clade:
                counts['grand_root_kmers'] += count
                
        # Ratios
        counts['grand_classified_kmers'] = counts['grand_clade_kmers'] + counts['grand_exclade_kmers']
        counts['grand_misclassified_kmers'] = counts['grand_exclade_kmers'] - counts['grand_lineage_kmers']
        counts['grand_unclassified_kmers'] = counts['grand_total_kmers'] - counts['grand_classified_kmers']
        
        c = counts
        denom_class = c['grand_classified_kmers'] if c['grand_classified_kmers'] > 0 else 1
        denom_tot = c['grand_total_kmers'] if c['grand_total_kmers'] > 0 else 1
        denom_exc = c['grand_exclade_kmers'] if c['grand_exclade_kmers'] > 0 else 1
        denom_misc = c['grand_misclassified_kmers'] if c['grand_misclassified_kmers'] > 0 else 1
        
        ratios = {
            'grand_clade_to_classified_kmer_ratio': c['grand_clade_kmers'] / denom_class,
            'grand_lineage_to_classified_kmer_ratio': c['grand_lineage_kmers'] / denom_class,
            'grand_misclassified_to_classified_kmer_ratio': c['grand_misclassified_kmers'] / denom_class,
            'grand_root_to_classified_kmer_ratio': c['grand_root_kmers'] / denom_class,
            'grand_supporting_to_misclassified_kmer_ratio': (c['grand_clade_kmers'] + c['grand_lineage_kmers']) / denom_misc if c['grand_misclassified_kmers'] > 0 else 1.0,
            
            'grand_clade_to_total_kmer_ratio': c['grand_clade_kmers'] / denom_tot,
            'grand_classified_to_total_kmer_ratio': c['grand_classified_kmers'] / denom_tot,
            'grand_lineage_to_total_kmer_ratio': c['grand_lineage_kmers'] / denom_tot,
            'grand_root_to_total_kmer_ratio': c['grand_root_kmers'] / denom_tot,
            'grand_misclassified_to_total_kmer_ratio': c['grand_misclassified_kmers'] / denom_tot,
            'grand_exclade_to_total_kmer_ratio': c['grand_exclade_kmers'] / denom_tot,
            'grand_supporting_to_total_kmer_ratio': (c['grand_clade_kmers'] + c['grand_lineage_kmers']) / denom_tot,
            
            'grand_root_to_exclade_kmer_ratio': c['grand_root_kmers'] / denom_exc,
            'grand_lineage_to_exclade_kmer_ratio': c['grand_lineage_kmers'] / denom_exc,
        }
        
        # Stats
        def prep_stats(data_list, val_col, out_suffix):
            res = get_weighted_stats(data_list, val_col, 'weight')
            return {f"mean_grand_misclassified_kmer_{out_suffix}": res['mean'],
                    f"median_grand_misclassified_kmer_{out_suffix}": res['median'],
                    f"cv_grand_misclassified_kmer_{out_suffix}": res['cv'],
                    f"p05_grand_misclassified_kmer_{out_suffix}": res['p05'],
                    f"p95_grand_misclassified_kmer_{out_suffix}": res['p95']}
                    
        stats_out = {}
        stats_out.update(prep_stats(dist_data, 'distance', 'distance'))
        stats_out.update(prep_stats(dist_data, 'kmer_depth', 'depth'))
        stats_out.update(prep_stats(dist_data, 'relative_lca_depth', 'relative_lca_depth'))
        
        lin_res = get_weighted_stats(lineage_dist_data, 'val', 'weight')
        stats_out.update({
            'mean_grand_lineage_kmer_relative_depth': lin_res['mean'],
            'median_grand_lineage_kmer_relative_depth': lin_res['median'],
            'cv_grand_lineage_kmer_relative_depth': lin_res['cv'],
            'p05_grand_lineage_kmer_relative_depth': lin_res['p05'],
            'p95_grand_lineage_kmer_relative_depth': lin_res['p95'],
        })
        
        # Top Hits
        def get_top_5(hits_data, prefix):
            if not hits_data:
                return {f'{prefix}_tax_ids': '', f'{prefix}_names': '', f'{prefix}_shares': ''}
                
            agg_h = {}
            for h in hits_data:
                agg_h[h['kmer_tax_id']] = agg_h.get(h['kmer_tax_id'], 0) + h['kmer_count']
                
            sorted_h = sorted(agg_h.items(), key=lambda x: (x[1], -x[0]), reverse=True)[:5]
            
            if prefix == 'grand_top_5_misclassified_kmer':
                global_sum = counts['grand_misclassified_kmers']
            else:
                global_sum = counts['grand_exclade_kmers']
                
            shares_str = ";".join([str(round(c / global_sum, 4)) for _, c in sorted_h])
            ids_str = ";".join([str(k) for k, _ in sorted_h])
            names_str = ";".join([tree.get_name(k) or "Unknown" for k, _ in sorted_h])
            return {f'{prefix}_tax_ids': ids_str, f'{prefix}_names': names_str, f'{prefix}_shares': shares_str}
            
        top_out = {}
        top_out.update(get_top_5(misclassified_data, 'grand_top_5_misclassified_kmer'))
        top_out.update(get_top_5(exclade_data, 'grand_top_5_exclade_kmer'))
        
        res_row = {'t_id': t_id}
        res_row.update(counts)
        res_row.update(ratios)
        res_row.update(stats_out)
        res_row.update(top_out)
        
        golden_results.append(res_row)
        
    golden_results.sort(key=lambda x: x['t_id'])
    
    if not golden_results:
        print("No results generated.")
        return
        
    headers = list(golden_results[0].keys())
    with open(out_dir / "golden_kmer_features.csv", "w", newline='') as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        for row in golden_results:
            # Format floats
            for k, v in row.items():
                if isinstance(v, float) and not math.isnan(v):
                    row[k] = f"{v:.6f}"
                elif v is None or (isinstance(v, float) and math.isnan(v)):
                    row[k] = ""
            writer.writerow(row)
            
    print("Golden dataset generated successfully.")

if __name__ == "__main__":
    main()
