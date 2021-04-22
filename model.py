
from pytorch_transformers import BertForTokenClassification
import torch
from torch import nn

class Ner(BertForTokenClassification):
    def forward(self, token_id, token_type_ids=None, attention_mask=None, labels=None, valid_id=None, attention_mask_label=None):
        sequence_output = self.bert(token_id, token_type_ids, attention_mask, head_mask=None)[0]
        batch_size,max_len,feat_dim = sequence_output.shape
        valid_output = torch.zeros(batch_size,max_len,feat_dim,dtype=torch.float32,device='cuda')
        # squeeze sequence_output to valid_output by valid_id
        for i in range(batch_size):
            k = -1
            for j in range(max_len):
                if valid_id[i][j].item() == 1:
                    k += 1
                    valid_output[i][k] = sequence_output[i][j]
        sequence_output = self.dropout(valid_output)
        network_output = self.classifier(sequence_output)

        if labels is not None:
            loss_function = nn.CrossEntropyLoss(ignore_index=0)
            loss = loss_function(network_output.view(-1, self.num_labels), labels.view(-1))
            return loss
        else:
            return network_output