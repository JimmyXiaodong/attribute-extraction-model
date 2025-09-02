import json
import logging
import os
from typing import Tuple, List

import datasets
from datasets import Dataset, DatasetDict, Features

logger = logging.getLogger(__name__)


def generate_dataset(data_files, data_args) -> DatasetDict:
    """
    generate model dataset from input data file
    :param data_files: the path of data
    :param data_args:  arguments
    :return: a DatasetDict
    """

    def data_generator(data_file):
        with open(data_file, "r") as f:
            current_tokens = []
            current_labels = []
            sentence_counter = 0
            for row in f:
                row = row.rstrip()
                if row:
                    token, label = row.split(" ")
                    current_tokens.append(token)
                    current_labels.append(label)
                else:
                    if not current_tokens:
                        continue
                    assert len(current_tokens) == len(current_labels), "mismatch between len of tokens & labels"
                    sentence = {
                        "id": str(sentence_counter),
                        "tokens": current_tokens,
                        "ner_tags": current_labels,
                    }
                    sentence_counter += 1
                    current_tokens = []
                    current_labels = []
                    yield sentence

                    if current_tokens:
                        yield {
                            "id": str(sentence_counter),
                            "tokens": current_tokens,
                            "ner_tags": current_labels,
                        }

    return DatasetDict({
        "train": Dataset.from_generator(data_generator, gen_kwargs={"data_file": data_files["train"]},
                                        cache_dir=data_args.data_cache_dir, ),
        "test": Dataset.from_generator(data_generator, gen_kwargs={"data_file": data_files["test"]},
                                       cache_dir=data_args.data_cache_dir, ),
        "validation": Dataset.from_generator(data_generator, gen_kwargs={"data_file": data_files["validation"]},
                                             cache_dir=data_args.data_cache_dir, )
    })


def get_tokenized_dataset(
        tokenizer, data_files, data_args
) -> Tuple[Dataset, Dataset, Dataset, List]:
    """
    get and preprocess all datasets used to train and evaluate the model
    :param tokenizer: tokenizer function
    :param data_files: the path of data
    :param data_args:  arguments
    :return: tokenized train dataset, validation dataset, test dataset and token label list
    """
    model_datasets = generate_dataset(data_files, data_args)

    dataset_features = Features(
        {
            "id": datasets.Value("string"),
            "tokens": datasets.Sequence(datasets.Value("string")),
            "ner_tags": datasets.Sequence(
                datasets.features.ClassLabel(
                    names=json.load(open(data_args.token_label_file, "r"))
                )
            ),
        })

    model_datasets = model_datasets.cast(dataset_features)

    def tokenize_and_align_labels(examples):
        tokenized_inputs = tokenizer(examples["tokens"], truncation=True, is_split_into_words=True, max_length=128)

        labels = []
        for i, label in enumerate(examples[f"ner_tags"]):
            word_ids = tokenized_inputs.word_ids(batch_index=i)  # Map tokens to their respective word.
            previous_word_idx = None
            label_ids = []
            for word_idx in word_ids:  # Set the special tokens to -100.
                if word_idx is None:
                    label_ids.append(-100)
                elif word_idx != previous_word_idx:  # Only label the first token of a given word.
                    label_ids.append(label[word_idx])
                else:
                    label_ids.append(-100)
                previous_word_idx = word_idx
            labels.append(label_ids)

        tokenized_inputs["labels"] = labels
        return tokenized_inputs

    map_cache_files = {
        "train": os.path.join(
            data_args.data_cache_dir,
            data_args.train_file.split("/")[-1].split(".")[0] + "_map.arrow"),
        "validation": os.path.join(
            data_args.data_cache_dir,
            data_args.validation_file.split("/")[-1].split(".")[0] + "_map.arrow"),
        "test": os.path.join(
            data_args.data_cache_dir,
            data_args.test_file.split("/")[-1].split(".")[0] + "_map.arrow"),
    }

    label_list = model_datasets["train"].features["ner_tags"].feature.names

    train_dataset = model_datasets["train"].map(
        tokenize_and_align_labels,
        batched=True,
        num_proc=data_args.preprocessing_num_workers,
        remove_columns=["tokens", "ner_tags"],
        load_from_cache_file=not data_args.overwrite_cache,
        cache_file_name=map_cache_files["train"]
    )

    validation_dataset = model_datasets["validation"].map(
        tokenize_and_align_labels,
        batched=True,
        num_proc=data_args.preprocessing_num_workers,
        remove_columns=["tokens", "ner_tags"],
        load_from_cache_file=not data_args.overwrite_cache,
        cache_file_name=map_cache_files["validation"]
    )

    test_dataset = model_datasets["test"].map(
        tokenize_and_align_labels,
        batched=True,
        num_proc=data_args.preprocessing_num_workers,
        remove_columns=["tokens", "ner_tags"],
        load_from_cache_file=not data_args.overwrite_cache,
        cache_file_name=map_cache_files["test"]
    )

    return train_dataset, validation_dataset, test_dataset, label_list
