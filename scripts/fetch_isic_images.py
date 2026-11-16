"""Multi-threaded downloader to fetch real images from official ISIC Archive S3 bucket."""
import os, sys, time, urllib.request
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed

def download_one(image_id, out_dir):
    out_path = os.path.join(out_dir, f"{image_id}.jpg")
    if os.path.exists(out_path) and os.path.getsize(out_path) > 1000:
        return True
    url = f"https://isic-archive.s3.us-east-1.amazonaws.com/thumbnails/{image_id}_thumbnail.jpg"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = resp.read()
            if len(data) > 500:
                with open(out_path, "wb") as f:
                    f.write(data)
                return True
    except Exception:
        # Fallback to full resolution if thumbnail is missing
        try:
            full_url = f"https://isic-archive.s3.us-east-1.amazonaws.com/images/{image_id}.jpg"
            req_full = urllib.request.Request(full_url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req_full, timeout=20) as resp:
                data = resp.read()
                if len(data) > 500:
                    with open(out_path, "wb") as f:
                        f.write(data)
                    return True
        except Exception:
            return False
    return False

def main():
    out_dir = "data/ham10000_raw/images"
    os.makedirs(out_dir, exist_ok=True)

    meta_csv = "data/ham10000_raw/HAM10000_metadata.csv"
    if not os.path.exists(meta_csv):
        meta_csv = "E:/DL_PBL/archive/HAM10000_metadata.csv"

    df = pd.read_csv(meta_csv)
    print(f"Total rows in metadata: {len(df)}")

    # Stratified selection: All 1,113 melanomas + 1,487 diverse non-melanoma controls = 2,600 images
    mel = df[df.dx == "mel"]
    non_mel = df[df.dx != "mel"].sample(n=min(1487, len(df[df.dx != "mel"])), random_state=42)
    selected_df = pd.concat([mel, non_mel]).sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Save the active subset metadata
    active_meta = "data/ham10000_raw/HAM10000_metadata_active.csv"
    selected_df.to_csv(active_meta, index=False)
    print(f"Selected {len(selected_df)} images ({len(mel)} melanoma, {len(non_mel)} non-melanoma).")
    print(f"Saved active metadata to {active_meta}")

    target_ids = selected_df["image_id"].tolist()

    # Also include the ISIC 2018 Task 1 images so masks can be used!
    mask_files = [f for f in os.listdir("data/ham10000_raw/ISIC2018_Task1_Training_GroundTruth") if f.endswith("_segmentation.png")]
    mask_ids = [f.replace("_segmentation.png", "") for f in mask_files][:600] # 600 mask images
    all_to_fetch = list(dict.fromkeys(target_ids + mask_ids))

    print(f"Total unique images to fetch: {len(all_to_fetch)}")
    t0 = time.time()
    success = 0
    with ThreadPoolExecutor(max_workers=16) as ex:
        futures = {ex.submit(download_one, iid, out_dir): iid for iid in all_to_fetch}
        for i, fut in enumerate(as_completed(futures)):
            if fut.result():
                success += 1
            if (i + 1) % 250 == 0 or (i + 1) == len(all_to_fetch):
                elapsed = time.time() - t0
                rate = (i + 1) / elapsed
                print(f"Progress: {i+1}/{len(all_to_fetch)} ({success} succeeded) - {rate:.1f} img/s")

    print(f"Finished downloading {success}/{len(all_to_fetch)} images in {time.time()-t0:.1f}s.")

if __name__ == "__main__":
    main()
