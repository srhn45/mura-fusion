import os
from kaggle import api

os.makedirs("data", exist_ok=True)

api.dataset_download_files(
    "cjinny/mura-v11",
    path="data",
    unzip=True
)