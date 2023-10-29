import os
import torch
import logging

def get_logger(name, log_dir):
    os.makedirs(log_dir, exist_ok=True)

    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    # file writer
    formatter = logging.Formatter("[%(asctime)s, %(levelname)s] : %(message)s")
    handler = logging.FileHandler(f'{log_dir}/_debug.log')
    handler.setFormatter(formatter)
    logger.setLevel(logging.DEBUG)
    logger.addHandler(handler)

    return logger


def collator(inputs):
    result = {
        'input_ids': torch.concat([inp['input_ids'].unsqueeze(-1) for inp in inputs], 1),
        'labels': torch.concat([inp['labels'].unsqueeze(-1) for inp in inputs], 1)
    }
    return result