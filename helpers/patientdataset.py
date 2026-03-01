import os
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader

from helpers.augments import make_transform

def load_df(csv_name, data_dir):
    df = pd.read_csv(os.path.join(data_dir, csv_name), dtype=str, header=None, names=["image_path"])
    df["label"]     = df["image_path"].str.contains("positive").astype(int)
    df["category"]  = df["image_path"].str.split("/").str[2]
    df["patientId"] = df["image_path"].str.split("/").str[3].str.replace("patient", "")
    return df

class PatientDataset(Dataset):
    """
    Each item is one patient's full set of X-ray images plus their label.
    """
    def __init__(self, df, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform

        # extract study id from path (once, before grouping)
        df = df.copy()
        df["studyId"] = df["image_path"].apply(lambda x: x.split("/")[4])  # study1, study2..

        self.patients = (
            df.groupby(["patientId", "category", "studyId"])
            .agg(image_paths=("image_path", list), label=("label", "first"))
            .reset_index()
        )

    def __len__(self):
        return len(self.patients)

    def __getitem__(self, idx):
        row = self.patients.iloc[idx]
        imgs = []
        for p in row["image_paths"]:
            img = Image.open(os.path.join(self.root_dir, p)).convert("L")
            if self.transform:
                img = self.transform(img)
            imgs.append(img)

        # (N, 1, H, W) — variable N per patient
        images = torch.stack(imgs, dim=0)
        label = torch.tensor(row["label"], dtype=torch.float32)
        return images, label


def patient_collate_fn(batch):
    """
    Returns a list of image tensors (one per patient) and a stacked label tensor.
    """
    image_list = [item[0] for item in batch]   # list of (N_i, 1, H, W)
    labels = torch.stack([item[1] for item in batch])  # (B,)
    return image_list, labels

def make_loader(df, augment, parent_dir, **kwargs):
    return DataLoader(PatientDataset(df, parent_dir, make_transform(augment)),
                      collate_fn=patient_collate_fn, **kwargs)
