import os
import pandas as pd

def parse_split(split_dir, out_csv):
    image_paths, genders = [], []
    for gender_folder in ["male", "female"]:
        gender_val = 1 if gender_folder == "male" else 0
        folder_path = os.path.join(split_dir, gender_folder)
        if not os.path.isdir(folder_path):
            continue
        for fname in os.listdir(folder_path):
            if fname.lower().endswith((".jpg", ".jpeg", ".png")):
                image_paths.append(os.path.join(folder_path, fname))
                genders.append(gender_val)
    df = pd.DataFrame({
        "image_path": image_paths,
        "gender": genders
    })
    df.to_csv(out_csv, index=False)
    print(f"✅ Processed {split_dir} -> {out_csv}")

def parse_identities(identities_dir, out_csv):
    image_paths, identities = [], []
    for identity in os.listdir(identities_dir):
        id_folder = os.path.join(identities_dir, identity)
        if not os.path.isdir(id_folder):
            continue
        # Recursively search for images in all subfolders
        for root, _, files in os.walk(id_folder):
            for fname in files:
                if fname.lower().endswith((".jpg", ".jpeg", ".png")):
                    image_paths.append(os.path.join(root, fname))
                    identities.append(identity)
    df = pd.DataFrame({
        "image_path": image_paths,
        "identity": identities
    })
    df.to_csv(out_csv, index=False)
    print(f"✅ Processed {identities_dir} -> {out_csv}")

def parse_distorted(distorted_dir, out_csv):
    image_paths = []
    if not os.path.isdir(distorted_dir):
        print(f"⚠ Skipped {distorted_dir} (folder not found)")
        return
    for fname in os.listdir(distorted_dir):
        if fname.lower().endswith((".jpg", ".jpeg", ".png")):
            image_paths.append(os.path.join(distorted_dir, fname))
    df = pd.DataFrame({"image_path": image_paths})
    df.to_csv(out_csv, index=False)
    print(f"✅ Processed {distorted_dir} -> {out_csv}")

if __name__ == "__main__":
    # Task A
    parse_split("data/raw/train", "train.csv")
    parse_split("data/raw/val", "val.csv")
    # Task B (update to match your actual folder structure)
    parse_identities("data/raw/Task_B/train", "identities_train.csv")
    parse_identities("data/raw/Task_B/val", "identities_val.csv")
    parse_distorted("data/raw/Task_B/distorted", "distorted.csv")
