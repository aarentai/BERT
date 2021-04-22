import os
            

def phase_eval(label, pred, true):
    total_pred_item = 0
    total_true_item = 0
    true_positive_item = 0
    for i in range(len(pred)):
        pred[i].append('O')
        for j in range(len(true[i])):
            illegal = False
            if pred[i][j] == 'B-'+label:
                total_pred_item += 1
                if pred[i][j] == true[i][j] and 'O' in pred[i][j:] and 'O' in true[i][j:] and pred[i][j:].index('O')==true[i][j:].index('O'):
                    for k in range(j,pred[i][j:].index('O')):
                        if label not in pred[i][k]:
                            illegal = True
                            break
                    if not illegal:
                        true_positive_item += 1
            if true[i][j] == 'B-'+label:
                total_true_item += 1

    precision = true_positive_item/total_pred_item
    recall = true_positive_item/total_true_item
    f1 = 2*precision*recall/(precision+recall)
    print(f'{label} precision {true_positive_item}/{total_pred_item}={precision}')
    print(f'{label} recall    {true_positive_item}/{total_true_item}={recall}')
    print(f'{label} F1        ={f1}')


class SampleRaw(object):
    def __init__(self, id, text, label=None):
        self.id = id
        self.text = text
        self.label = label


class SampleFeatures(object):
    def __init__(self, token_id, token_mask, segment_id, label_id, valid_id=None, label_mask=None):
        self.token_id = token_id
        self.token_mask = token_mask
        self.label_id = label_id
        self.label_mask = label_mask
        self.segment_id = segment_id
        self.valid_id = valid_id


class DataProcessor(object):
    def get_train_sample(self, data_dir):
        return self._init_sample_obj(self._read_file(os.path.join(data_dir, "train.txt")), "train")

    def get_dev_sample(self, data_dir):
        return self._init_sample_obj(self._read_file(os.path.join(data_dir, "demo.txt")), "dev")

    def get_test_sample(self, data_dir):
        return self._init_sample_obj(self._read_file(os.path.join(data_dir, "test.txt")), "test")

    def _init_sample_obj(self,lines,set_type):
        samples = []
        for i,(sentence,label) in enumerate(lines):
            id = f"{set_type}-{i}"
            text = ' '.join(sentence)
            samples.append(SampleRaw(id=id,text=text,label=label))
        return samples
    
    def _read_file(self, input_file, quotechar=None):
        f = open(input_file)
        data, sentence, label = [], [], []
        for line in f:
            if len(line)==0 or line.startswith('-DOCSTART') or line[0]=="\n":
                if len(sentence) > 0:
                    data.append((sentence,label))
                    sentence, label= [], []
                continue
            splits = line.split(' ')
            sentence.append(splits[0])
            label.append(splits[-1][:-1])

        if len(sentence) >0:
            data.append((sentence,label))
            sentence, label= [], []
        return data


def from_raw_to_feature(samples, label_list, max_seq_length, tokenizer):
    label_map = {label : i for i, label in enumerate(label_list,1)}

    features = []
    for (_,sample) in enumerate(samples):
        textlist = sample.text.split(' ')
        labellist = sample.label
        tokens, labels, valid, label_mask = [], [], [], []
        for i, word in enumerate(textlist):
            token = tokenizer.tokenize(word)
            tokens.extend(token)
            label_1 = labellist[i]
            for j in range(len(token)):
                if j == 0:
                    labels.append(label_1)
                    valid.append(1)
                    label_mask.append(1)
                else:
                    valid.append(0)
        # if tokenized sequence is larger than 128, crop the end of it to fit the network
        if len(tokens) >= max_seq_length - 1:
            tokens = tokens[0:(max_seq_length - 2)]
            labels = labels[0:(max_seq_length - 2)]
            valid = valid[0:(max_seq_length - 2)]
            label_mask = label_mask[0:(max_seq_length - 2)]
        ntokens, segment_id, label_id = [], [], []
        ntokens.append("[CLS]")

        # corresponding to [CLS]
        segment_id.append(0)
        valid.insert(0,1)
        label_mask.insert(0,1)
        label_id.append(label_map["[CLS]"])

        for i, token in enumerate(tokens):
            ntokens.append(token)
            segment_id.append(0)
            if len(labels) > i:
                label_id.append(label_map[labels[i]])
        ntokens.append("[SEP]")

        # corresponding to [SEP]
        segment_id.append(0)
        valid.append(1)
        label_mask.append(1)
        label_id.append(label_map["[SEP]"])

        token_id = tokenizer.convert_tokens_to_ids(ntokens)
        token_mask = [1] * len(token_id)
        label_mask = [1] * len(label_id)
        # pad all list to max_seq_length
        while len(token_id) < max_seq_length:
            token_id.append(0)
            token_mask.append(0)
            segment_id.append(0)
            label_id.append(0)
            valid.append(1)
            label_mask.append(0)
        while len(label_id) < max_seq_length:
            label_id.append(0)
            label_mask.append(0)

        features.append(
                SampleFeatures(token_id=token_id,
                              token_mask=token_mask,
                              segment_id=segment_id,
                              label_id=label_id,
                              valid_id=valid,
                              label_mask=label_mask))
    return features
