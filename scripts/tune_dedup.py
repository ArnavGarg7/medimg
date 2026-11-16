"""Re-tune near-duplicate detection thresholds on real dermoscopic images.
Sweeps Hamming distance and MAD (Mean Absolute Difference) thresholds and reports findings.
"""
import os, json
import numpy as np, pandas as pd
from medimg.prepare_splits import thumb, ahash, sha, UF

def sweep_dedup_thresholds(metadata_path, root_dir, hamming_values=(4, 8, 12, 16, 20), mad_values=(0.01, 0.02, 0.03, 0.05)):
    df = pd.read_csv(metadata_path)
    ids = df.image_id.tolist()
    print(f"Loaded {len(ids)} images for deduplication threshold tuning.")

    # Compute thumbnails, hashes, and sha256 checksums once
    print("Computing thumbnails and 16x16 aHash (256-bit)...")
    thumbs = [thumb(os.path.join(root_dir, p)) for p in df.image_path]
    hashes = [ahash(t) for t in thumbs]
    shas = [sha(os.path.join(root_dir, p)) for p in df.image_path]
    H = np.stack(hashes)

    # Exact duplicate links (SHA-256)
    seen = {}
    n_exact = 0
    for i, s in enumerate(shas):
        if s in seen:
            n_exact += 1
        else:
            seen[s] = i

    results = []
    print("\n--- Sweeping Thresholds ---")
    for h in hamming_values:
        for m in mad_values:
            uf = UF()
            # 1. Base grouping by patient_id / lesion_id
            for _, r in df.iterrows():
                uf.union("img:" + r.image_id, "pat:" + str(r.patient_id))
            
            # Exact duplicates
            seen_exact = {}
            for i, s in enumerate(shas):
                if s in seen_exact:
                    uf.union("img:" + ids[i], "img:" + ids[seen_exact[s]])
                else:
                    seen_exact[s] = i

            n_near = 0
            for i in range(len(ids)):
                d = (H[i + 1:] != H[i]).sum(1)
                for j in np.where(d <= h)[0]:
                    if np.abs(thumbs[i] - thumbs[i + 1 + j]).mean() < m:
                        uf.union("img:" + ids[i], "img:" + ids[i + 1 + j])
                        n_near += 1
            
            groups = [uf.find("img:" + i) for i in ids]
            group_series = pd.Series(groups)
            n_groups = int(group_series.nunique())
            largest_group_frac = float(group_series.value_counts().iloc[0] / len(df))
            degenerate = largest_group_frac > 0.5

            res = {
                "hamming": h,
                "mad": m,
                "exact_duplicates": n_exact,
                "near_duplicates": n_near,
                "n_groups": n_groups,
                "largest_group_frac": round(largest_group_frac, 4),
                "degenerate": degenerate
            }
            results.append(res)
            print(f"Hamming={h:2d}, MAD={m:.2f} -> Near Dups: {n_near:4d}, Total Groups: {n_groups:4d}, Largest Group: {largest_group_frac*100:5.2f}% (Degenerate: {degenerate})")

    out_file = os.path.join(root_dir, "dedup_tuning_report.json")
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved tuning report to {out_file}")
    return results

if __name__ == "__main__":
    import sys
    meta = "data/ham10000/metadata.csv"
    root = "data/ham10000"
    sweep_dedup_thresholds(meta, root)
