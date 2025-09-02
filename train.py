import logging
import os
from datetime import datetime

import numpy as np
import transformers
from datasets import load_metric
from transformers import (
    set_seed,
    AutoTokenizer,
    HfArgumentParser,
    DataCollatorForTokenClassification,
    Trainer,
    AutoModelForTokenClassification
)
from transformers.trainer_utils import is_main_process

from models.attribute_extraction_model.definition.data import get_tokenized_dataset
from models.attribute_extraction_model.definition.model_arguments import (
    AttributeExtractionModelArguments,
    AttributeExtractionModelDataTrainingArguments,
    AttributeExtractionModelTrainingArguments)

os.environ["CUDA_VISIBLE_DEVICES"] = "2，3"
os.environ["NCCL_DEBUG"] = "INFO"

logger = logging.getLogger(__name__)
timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")


def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((
        AttributeExtractionModelArguments, AttributeExtractionModelDataTrainingArguments,
        AttributeExtractionModelTrainingArguments))

    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if is_main_process(
            training_args.local_rank) else logging.WARN,
    )

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, "
        f"device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f" distributed training: {bool(training_args.local_rank != -1)}, "
          f"16-bits training: {training_args.fp16}"
    )
    # Set the verbosity to info of the Transformers logger (
    # on main process only):
    if is_main_process(training_args.local_rank):
        transformers.utils.logging.set_verbosity_info()
        transformers.utils.logging.enable_default_handler()
        transformers.utils.logging.enable_explicit_format()
    logger.info("Training/evaluation parameters %s", training_args)

    # Set seed before initializing model.
    set_seed(training_args.seed)

    # Get the datasets: you can either provide your own CSV/JSON/TXT training
    # and evaluation files (see below)
    #
    # For CSV/JSON files, this script will use the column called 'text' or the
    # first column. You can easily tweak this behavior (see below)
    #
    # In distributed training, the load_dataset function guarantee that only
    # one local process can concurrently download the dataset.
    logger.info("Data parameters %s", data_args)
    data_files = {}
    if data_args.train_file is not None:
        data_files["train"] = data_args.train_file
    if data_args.validation_file is not None:
        data_files["validation"] = data_args.validation_file
    if data_args.test_file is not None:
        data_files["test"] = data_args.test_file

    # Load pretrained model and tokenizer
    #
    # Distributed training:
    # The .from_pretrained methods guarantee that only one local process can
    # concurrently
    logger.info("Model parameters %s", model_args)
    if not model_args.model_name_or_path:
        raise ValueError(
            "The parameter of model name or path must be specified."
        )

    tokenizer_kwargs = {
        "use_fast": model_args.use_fast_tokenizer
    }

    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path,
                                              **tokenizer_kwargs)

    train_dataset, validation_dataset, test_dataset, label_list = get_tokenized_dataset(
        tokenizer, data_files, data_args)
    logger.info(train_dataset)
    logger.info(validation_dataset)
    logger.info(test_dataset)

    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

    id2label = {i: label for i, label in enumerate(label_list)}
    label2id = {label: i for i, label in enumerate(label_list)}

    model = AutoModelForTokenClassification.from_pretrained(
        model_args.model_name_or_path,
        num_labels=len(id2label.keys()),
        id2label=id2label,
        label2id=label2id,
        cache_dir=model_args.cache_dir)

    training_args.output_dir = os.path.join(
        training_args.output_dir, training_args.dataset_date, timestamp
    )
    logging.info("The output path: {}".format(training_args.output_dir))

    def compute_sequence_metrics(eval_pred):
        """
        compute accuracy, recall and f1 score for sequence labeling model
        :param eval_pred: prediction(logit) and label
        :return: accuracy, recall and f1 score
        """
        predictions, labels = eval_pred
        sequence_metric = load_metric(training_args.metric_path)
        predictions = np.argmax(predictions, axis=2)
        true_predictions = [
            [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]
        true_labels = [
            [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]

        metric_results = sequence_metric.compute(predictions=true_predictions, references=true_labels)
        return {
            "precision": metric_results["overall_precision"],
            "recall": metric_results["overall_recall"],
            "f1": metric_results["overall_f1"],
            "accuracy": metric_results["overall_accuracy"],
        }

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=validation_dataset if training_args.do_eval else None,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_sequence_metrics,
    )

    # Training
    if training_args.do_train:
        train_result = trainer.train()
        trainer.save_model()  # Saves the tokenizer too for easy upload

        output_train_file = os.path.join(training_args.output_dir,
                                         "train_results.txt")
        if trainer.is_world_process_zero():
            with open(output_train_file, "w") as writer:
                logger.info("***** Train results *****")
                for key, value in sorted(train_result.metrics.items()):
                    logger.info(f"  {key} = {value}")
                    writer.write(f"{key} = {value}\n")

            # Need to save the state, since Trainer.save_model saves only
            # the tokenizer with the model
            trainer.state.save_to_json(
                os.path.join(training_args.output_dir, "trainer_state.json"))

    results = {}
    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        results = trainer.evaluate(eval_dataset=test_dataset)

        output_eval_file = os.path.join(training_args.output_dir,
                                        "eval_results.txt")
        if trainer.is_world_process_zero():
            with open(output_eval_file, "w") as writer:
                for key, value in sorted(results.items()):
                    writer.write(f"{key} = {value}\n")

    return results


if __name__ == "__main__":
    main()
