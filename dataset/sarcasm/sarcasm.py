# -*- coding: utf-8 -*-
"""Sarcasm

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/15_wDQ9RJXwyxbomu2F1k0pK9H7XZ1cuT
"""
import geopandas
import matplotlib.pyplot as plt
import seaborn as sns
from shapely.geometry import Point
import pandas as pd
import geopandas as gpd
from datasets import (
    GeneratorBasedBuilder, Version, DownloadManager, SplitGenerator, Split,
    Features, Value, DatasetInfo
)

# URL definitions
_URLS = {
    "csv_file": "https://drive.google.com/uc?export=download&id=1WcPqVZasDy1nmGcildLS-uw_-04I9Max",
}

class Sarcasm(GeneratorBasedBuilder):
    VERSION = Version("1.0.0")

    def _info(self):
        return DatasetInfo(
            description="This dataset contains sarcastic comments.",
            features=Features({
                "comments": Value("string"),
                "contains_slash_s": Value("int64"),
            }),
            supervised_keys=None,
            homepage="https://github.com/AuraMa111?tab=repositories",
            citation="Citation for the combined dataset",
        )

    def _split_generators(self, dl_manager):
        downloaded_files = dl_manager.download_and_extract(_URLS)
        data_file_path = downloaded_files["csv_file"]

        # Debug information
        print(f"下载的文件路径: {data_file_path}")
        print("文件内容预览:")
        print(pd.read_csv(data_file_path).head())

        num_examples = pd.read_csv(data_file_path).shape[0]
        train_size = int(0.6 * num_examples)
        val_size = int(0.2 * num_examples)
        test_size = num_examples - train_size - val_size

        return [
            SplitGenerator(
                name=Split.TRAIN,
                gen_kwargs={"data_file_path": data_file_path, "split": Split.TRAIN, "size": train_size, "val_size": val_size}
            ),
            SplitGenerator(
                name=Split.VALIDATION,
                gen_kwargs={"data_file_path": data_file_path, "split": Split.VALIDATION, "size": val_size, "val_size": val_size}
            ),
            SplitGenerator(
                name=Split.TEST,
                gen_kwargs={"data_file_path": data_file_path, "split": Split.TEST, "size": test_size, "val_size": val_size}
            ),
        ]

    def _generate_examples(self, data_file_path, split, size, val_size):
        data = pd.read_csv(data_file_path)
        if split == Split.TRAIN:
            subset_data = data[:size]
        elif split == Split.VALIDATION:
            subset_data = data[size:size + val_size]
        elif split == Split.TEST:
            subset_data = data[size + val_size:]

        for index, row in subset_data.iterrows():
            example = {
                "comments": row["comments"],
                "contains_slash_s": row["contains_slash_s"]
            }
            yield index, example

# Instantiate your dataset class
sarcasm = Sarcasm()

# Build the datasets
print("开始下载和准备数据集...")
sarcasm.download_and_prepare()
print("数据集下载和准备完成。")

# Access the datasets for training, validation, and testing
dataset_train = sarcasm.as_dataset(split='train')
print("训练数据集:", dataset_train)

dataset_validation = sarcasm.as_dataset(split='validation')
print("验证数据集:", dataset_validation)

dataset_test = sarcasm.as_dataset(split='test')
print("测试数据集:", dataset_test)