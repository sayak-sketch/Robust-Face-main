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

if __name__ == "__main__":
    parse_split("data/raw/train", "train.csv")
    parse_split("data/raw/val", "val.csv")
