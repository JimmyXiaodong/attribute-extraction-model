import os

from typing import Tuple
from datasets import load_dataset, Dataset


def get_tokenized_dataset(
        tokenizer, data_files, data_args
) -> Tuple[Dataset, Dataset]:
    """
    get and preprocess all datasets used to train and evaluate the model
    :param tokenizer: tokenizer function
    :param data_files: the path of data
    :param data_args:  arguments
    :return: tokenized train dataset and test dataset
    """

    def preprocess_function(examples):
        if data_args.pad_to_max_length:
            return tokenizer(examples["title"], padding="max_length",
                             max_length=data_args.max_seq_length,
                             truncation=True)
        else:
            return tokenizer(examples["title"], padding=True,
                             max_length=data_args.max_seq_length,
                             truncation=True)

    map_cache_files = {
        "train": os.path.join(data_args.data_path,
                              data_files["train"].split(".")[0] + "_map.arrow"),
        "test": os.path.join(data_args.data_path,
                             data_files["test"].split(".")[0] + "_map.arrow")
    }
    data_files = {k: os.path.join(data_args.data_path, v) for k, v in
                  data_files.items()}

    datasets = load_dataset(
        "csv",
        cache_dir=data_args.data_path,
        data_files=data_files)
    train_dataset = datasets["train"].map(
        preprocess_function,
        load_from_cache_file=not data_args.overwrite_cache,
        cache_file_name=map_cache_files["train"],
        batched=True)
    test_dataset = datasets["test"].map(
        preprocess_function,
        load_from_cache_file=not data_args.overwrite_cache,
        cache_file_name=map_cache_files["test"],
        batched=True)
    return train_dataset, test_dataset
