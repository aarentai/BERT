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

    if do_train:
        train_features = from_raw_to_feature(processor.get_train_sample(raw_data_path), label_list, max_seq_length, tokenizer)
        # wrapping tensors
        train_data = TensorDataset(torch.tensor([f.token_id for f in train_features], dtype=torch.long), 
                                    torch.tensor([f.token_mask for f in train_features], dtype=torch.long), 
                                    torch.tensor([f.segment_id for f in train_features], dtype=torch.long), 
                                    torch.tensor([f.label_id for f in train_features], dtype=torch.long),
                                    torch.tensor([f.valid_id for f in train_features], dtype=torch.long),
                                    torch.tensor([f.label_mask for f in train_features], dtype=torch.long))
        # provides an iterable over the given dataset
        train_dataloader = DataLoader(train_data, sampler=RandomSampler(train_data), batch_size=train_batch_size)

        model.train()
        for _ in trange(int(num_train_epochs), desc="Epoch"):
            for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
                batch = tuple(t.to(device) for t in batch)
                token_id, token_mask, segment_id, label_id, valid_id, l_mask = batch
                loss = model(token_id, segment_id, token_mask, label_id, valid_id, l_mask)

                loss.backward()
                # Total norm of the parameters
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

                # The optimizer dictates the "update rule"--how the parameters are
                # modified based on their gradients, the learning rate, etc.
                optimizer.step()
                # Update the learning rate
                scheduler.step()
                # reset the grad to zero after backpropagation
                model.zero_grad()

        model.save_pretrained(model_path)
        tokenizer.save_pretrained(model_path)
        label_map = {i : label for i, label in enumerate(label_list,1)}
    else:
        model = Ner.from_pretrained(model_path)
        tokenizer = BertTokenizer.from_pretrained(model_path, do_lower_case=False)

    model.to(device)

    if do_eval:
        eval_features = from_raw_to_feature(processor.get_dev_sample(raw_data_path), label_list, max_seq_length, tokenizer)
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
                '''without label_id here'''
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
        y_true_flatten = [[true for sentence in y_true for true in sentence]]
        y_pred_flatten = [['O' if pred=='[SEP]' else pred for sentence in y_pred for pred in sentence]]
        with open('y_pred.txt','w') as f:
            for item in y_pred:
                f.write("%s\n" % item)
        with open('y_true.txt','w') as f:
            for item in y_true:
                f.write("%s\n" % item)

        report = classification_report(y_true_flatten, y_pred_flatten, digits=4)
        print(f'True label: ', y_true_flatten)
        print(f'Pred label: ', y_pred_flatten)
        # phase_eval('PER', y_pred, y_true)
        # phase_eval('ORG', y_pred, y_true)
        # phase_eval('LOC', y_pred, y_true)
        # phase_eval('MISC', y_pred, y_true)
        from ner_evaluation.ner_eval import Evaluator
        evaluator = Evaluator(y_true, y_pred, ['LOC', 'MISC', 'PER', 'ORG'])
        # template = "{0:8}|{1:8}|{2:8}|{3:8}" # column widths: 8, 8, 8, 8
        # print(template.format("", "precision", "recall", "f1-score"))
        # for category in ['LOC', 'MISC', 'PER', 'ORG']:
        #     print(category, format(evaluator.evaluate()[category]['strict']['precision'],'.4f'), format(evaluator.evaluate()[category]['strict']['recall'],'.4f'), format(evaluator.evaluate()[category]['strict']['f1'],'.4f'))

        logging.basicConfig(level = logging.INFO)
        logger = logging.getLogger(__name__)
        logger.info("\n%s", report)


if __name__ == "__main__":
    main()
