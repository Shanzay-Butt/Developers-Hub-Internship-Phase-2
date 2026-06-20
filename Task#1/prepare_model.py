from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments, DataCollatorWithPadding
import evaluate
import numpy as np

MODEL_DIR = "news_classifier_model"
CHECKPOINT = "bert-base-uncased"
MAX_LENGTH = 128


def preprocess_function(examples):
    return tokenizer(
        examples["text"],
        truncation=True,
        padding="max_length",
        max_length=MAX_LENGTH,
    )


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    accuracy = accuracy_metric.compute(predictions=predictions, references=labels)["accuracy"]
    f1_macro = f1_metric.compute(predictions=predictions, references=labels, average="macro")["f1"]
    return {
        "accuracy": accuracy,
        "f1_macro": f1_macro,
    }


if __name__ == "__main__":
    raw_datasets = load_dataset("ag_news")
    label_names = raw_datasets["train"].features["label"].names
    tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT)

    processed_datasets = raw_datasets.map(
        preprocess_function,
        batched=True,
        remove_columns=["text"],
    )
    processed_datasets = processed_datasets.rename_column("label", "labels")
    processed_datasets.set_format("torch")

    train_dataset = processed_datasets["train"].shuffle(seed=42).select(range(20000))
    eval_dataset = processed_datasets["test"].shuffle(seed=42).select(range(5000))

    model = AutoModelForSequenceClassification.from_pretrained(CHECKPOINT, num_labels=len(label_names))
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    accuracy_metric = evaluate.load("accuracy")
    f1_metric = evaluate.load("f1")

    training_args = TrainingArguments(
        output_dir=MODEL_DIR,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        num_train_epochs=2,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        save_total_limit=2,
        logging_steps=100,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    trainer.evaluate()
    trainer.save_model(MODEL_DIR)
    tokenizer.save_pretrained(MODEL_DIR)
