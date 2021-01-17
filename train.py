import torch
import pandas as pd
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch.optim import lr_scheduler

from sklearn import model_selection
import transformers
import tokenizers
from transformers import AdamW
from transformers import get_linear_schedule_with_warmup
from tqdm.autonotebook import tqdm
import utils
from sklearn.model_selection import train_test_split

from train_helper import train_fn, eval_fn
from ExtractiveDataloader import ExtractionDataset
import datasets

def train(config, dataset):

    train_dataset = ExtractionDataset(
        context=dataset['train']['context'],
        question=dataset['train']['question'],
        answer=dataset['train']['answers'],
        config = config
    )

    train_data_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size = config.TRAIN_BATCH_SIZE,
        num_workers=4
    )

    valid_dataset = ExtractionDataset(
        context=dataset['validation']['context'],
        question=dataset['validation']['question'],
        answer=dataset['validation']['answers'],
        config = config
    )

    valid_data_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size = config.VALID_BATCH_SIZE,
        num_workers=2
    )

    device = torch.device("cuda")

    model = transformers.RobertaForQuestionAnswering.from_pretrained(config.TOKENIZER)
    model.to(device)

    num_train_steps = int(len(dataset['train']) / config.TRAIN_BATCH_SIZE * EPOCHS)
    param_optimizer = list(model.named_parameters())
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    optimizer_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.001},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0},
    ]
    optimizer = AdamW(optimizer_parameters, lr=3e-5)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=0, 
        num_training_steps=num_train_steps
    )

    es = utils.EarlyStopping(patience=5, mode="max")
    print(f"Training is Starting:")
    
    for epoch in range(EPOCHS):
        train_fn(train_data_loader, model, optimizer, device, scheduler=scheduler)
        jaccard = eval_fn(valid_data_loader, model, device)
        print(f"Jaccard Score = {jaccard}")
        es(jaccard, model, model_path=config.DIRECTORY)
        if es.early_stop:
            print("Early stopping")
            break