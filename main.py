import logging
from src import get_from_pretrained, get_from_config, get_dataset, tokenize, masking, batching, get_model
from src.utils import get_logger, collator
from transformers import Trainer, TrainingArguments

def main():
    # Step 0 : dataset
    dataset = get_dataset()
    # TODO: User word interface

    # Step 1 : Tokenizer
    # user word should be a list of string type userword
    tokenizer = get_from_config()

    # Step 2 : Data Process
    tokenized_datasets = dataset.map(
        tokenize, 
        batched=True,
        fn_kwargs={'tokenizer': tokenizer}, 
        )

    tokenized_datasets = tokenized_datasets.map(
        batching, 
        batched=True, 
        fn_kwargs={'_ctx_len': context_length}, 
        remove_columns=tokenized_datasets['train'].column_names, 
        oad_from_cache_file=False)

    tokenized_datasets.set_format(type='torch', columns=['input_ids'])
    tokenized_datasets = tokenized_datasets.map(
        masking, 
        batched=True, 
        fn_kwargs={'mask_prob': 0.15, 'tokenizer': tokenizer})

    # Step 3 : Model
    # NOTE: can be loaded from pretrained model
    model = get_model(name_or_path=model_name, tokenizer=tokenizer)

    # Step 4 : Train
    # TODO: train argument custom
    args = TrainingArguments(
        # full_determinism=True,
        output_dir="./",
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
        evaluation_strategy="steps",
        # eval_steps=5_000,
        logging_steps=20,
        num_train_epochs=1,
        weight_decay=0.1,
        # warmup_steps=1_000,
        lr_scheduler_type="cosine",
        learning_rate=5e-4,
        save_steps=5_000,
    )

    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        args=args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["test"],
        data_collator=collator
    )

    trainer.train()

    # TODO: save model 




if __name__ == '__main__':
    # These are can be replace with argumentParser
    output_dir = './output'
    context_length = 128
    model_name = 'bert-base-uncased'

    # TODO: logger 
    logger = get_logger('main', output_dir)

    # TODO: test code
    main()