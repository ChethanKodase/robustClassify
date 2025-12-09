import sys
from datasets import load_dataset
sys.path.append('..')

from datasets import load_dataset

dataset = load_dataset("parquet", data_files="imagenetparaquet/train-*.parquet", split="train")

for i, item in enumerate(dataset):
    image = item["image"]
    image.save(f"imagenetDataSubset/{i}.jpg")

