import numpy as np
import logging, os, sys, torch
import torch.nn.functional as F
from processor import *
from model import *
from pytorch_transformers import WEIGHTS_NAME, AdamW, BertConfig, BertTokenizer, WarmupLinearSchedule
from torch import nn
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from tqdm import tqdm, trange

from seqeval.metrics import classification_report

def main():
    raw_data_path = 'data/'
    model_path = 'model'
    max_seq_length = 128
    do_train = False
    do_eval = True
    train_batch_size = 32
    eval_batch_size = 8
    num_train_epochs = 3
    max_grad_norm = 1
    gradient_accumulation_steps = 1
    device = torch.device("cuda")
    if not os.path.exists(model_path):
        os.makedirs(model_path)

    # sentence = 'As officers secured the building, more than a dozen people were led out of the supermarket, a King Soopers in a residential area a couple of miles south of the campus of the University of Colorado.'
    sentence = 'At the Oval , Surrey captain Chris Lewis , another man dumped by England , continued to silence his critics as he followed his four for 45 on Thursday with 80 not out on Friday in the match against Warwickshire .'
    sentence_list = []
    sentence_list.append(SampleRaw(id='dev-0',text=sentence,label=['O' for i in range(41)]))

    processor = DataProcessor()
    label_list = ["O", "B-MISC", "I-MISC",  "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC", "[CLS]", "[SEP]"]
    label_map = {i : label for i, label in enumerate(label_list,1)}
    # Instantiate a pretrained pytorch model from a pre-trained model configuration.
    tokenizer = BertTokenizer.from_pretrained('bert-base-cased', do_lower_case=False)

    # This is the configuration class to store the configuration of a BertModel
    config = BertConfig.from_pretrained('bert-base-cased', num_labels=len(label_list) + 1, finetuning_task='ner')
    model = Ner.from_pretrained('bert-base-cased', from_tf = False, config = config)

    model.to(device)

    param_optimizer = list(model.named_parameters())
    no_decay = ['bias','LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}        ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=5e-5, eps=1e-8)
    # warmup_steps = warmup_proportion * num_train_optimization_steps,
    # t_total = train_examples / train_batch_size / gradient_accumulation_steps * num_train_epochs
    scheduler = WarmupLinearSchedule(optimizer, warmup_steps=131, t_total=1314)

    eval_features = from_raw_to_feature(sentence_list, label_list, max_seq_length, tokenizer)
    # eval_features = from_raw_to_feature(processor.get_dev_sample(raw_data_path), label_list, max_seq_length, tokenizer)
    eval_data = TensorDataset(torch.tensor([f.token_id for f in eval_features], dtype=torch.long), 
                                    torch.tensor([f.token_mask for f in eval_features], dtype=torch.long), 
                                    torch.tensor([f.segment_id for f in eval_features], dtype=torch.long), 
                                    torch.tensor([f.label_id for f in eval_features], dtype=torch.long), 
                                    torch.tensor([f.valid_id for f in eval_features], dtype=torch.long), 
                                    torch.tensor([f.label_mask for f in eval_features], dtype=torch.long))
        
    eval_dataloader = DataLoader(eval_data, sampler=SequentialSampler(eval_data), batch_size=eval_batch_size)
    model.eval()
    y_true = []
    y_pred = []
    label_map = {i : label for i, label in enumerate(label_list,1)}
    for token_id, token_mask, segment_id, label_id, valid_id, l_mask in tqdm(eval_dataloader, desc="Evaluating"):
        token_id = token_id.to(device)
        token_mask = token_mask.to(device)
        segment_id = segment_id.to(device)
        valid_id = valid_id.to(device)
        label_id = label_id.to(device)
        l_mask = l_mask.to(device)

        with torch.no_grad():
            network_output = model(token_id, segment_id, token_mask,valid_id=valid_id,attention_mask_label=l_mask)

        network_output = torch.argmax(F.log_softmax(network_output,dim=2),dim=2)
        network_output = network_output.detach().cpu().numpy()
        label_id = label_id.to('cpu').numpy()
        token_mask = token_mask.to('cpu').numpy()

        for i, label in enumerate(label_id):
            sentence_label_true = []
            sentence_label_pred = []
            for j,m in enumerate(label):
                if j == 0:
                    continue
                elif label_id[i][j] == len(label_map):
                    y_true.append(sentence_label_true)
                    y_pred.append(sentence_label_pred)
                    break
                else:
                    sentence_label_true.append(label_map[label_id[i][j]])
                    sentence_label_pred.append(label_map[network_output[i][j]])

    print(y_true)
    print(y_pred)


if __name__ == "__main__":
    main()