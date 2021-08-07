import os
import torch
import random
import logging
import argparse
import torch.nn as nn
import numpy as np

from util import tools
from eval_bert import evaluate
from tqdm import trange
from torch.nn.utils.rnn import pad_sequence
from transformers import BertTokenizer
from transformers.optimization import get_linear_schedule_with_warmup, AdamW
from transformers.modeling_bert import *


class DataLoader(object):
    def __init__(self, data_dir, bert_class, params, token_pad_idx=0, tag_pad_idx=-1):
        self.data_dir = data_dir
        self.batch_size = params.batch_size
        self.max_len = params.max_len
        self.device = params.device
        self.token_pad_idx = token_pad_idx
        self.tag_pad_idx = tag_pad_idx

        # Load all tags from tags.txt
        tags = self.load_tags()

        # Get tag-id and id-tag dictionaries
        self.tag2idx = {tag: idx for idx, tag in enumerate(tags)}
        self.idx2tag = {idx: tag for idx, tag in enumerate(tags)}
        params.tag2idx = self.tag2idx
        params.idx2tag = self.idx2tag

        # Get tokens
        self.tokenizer = BertTokenizer.from_pretrained(
            bert_class, do_lower_case=False)

    def load_tags(self):
        '''
        Load all tags from tags.txt
        Return:
            tags: a list of all tags
                "['B-ANA', 'B-DIS', ...]"
        '''
        tags = []
        file_path = os.path.join(self.data_dir, 'tags.txt')
        with open(file_path, 'r') as file:
            for tag in file:
                tags.append(tag.strip())
        return tags

    def load_sentences_tags(self, sentences_file, tags_file, d):
        """
        Loads sentences and tags from their corresponding files. 
        Maps tokens and tags to their indices and stores them in the provided dict d.
        """
        sentences = []
        tags = []

        with open(sentences_file, 'r') as file:
            for line in file:
                # replace each token by its index
                tokens = line.strip().split(' ')
                subwords = list(map(self.tokenizer.tokenize, tokens))
                subword_lengths = list(map(len, subwords))
                subwords = ['[CLS]'] + \
                    [item for indices in subwords for item in indices]
                token_start_idxs = 1 + np.cumsum([0] + subword_lengths[:-1])
                sentences.append(
                    (self.tokenizer.convert_tokens_to_ids(subwords), token_start_idxs))
        if tags_file != None:
            with open(tags_file, 'r') as file:
                for line in file:
                    # replace each tag by its index
                    tag_seq = [self.tag2idx.get(tag)
                               for tag in line.strip().split(' ')]
                    tags.append(tag_seq)

            # checks to ensure there is a tag for each token
            assert len(sentences) == len(tags)
            for i in range(len(sentences)):
                assert len(tags[i]) == len(sentences[i][-1])

            d['tags'] = tags

        # storing sentences and tags in dict d
        d['data'] = sentences
        d['size'] = len(sentences)

    def load_data(self, data_type):
        """
        Loads the data for each type in types from data_dir.
        Params:
            data_type: (str) has one of 'train', 'val', 'test' depending on which data is required.
        Return:
            data: (dict) contains the data with tags for each type in types.
        """
        data = {}
        if data_type in ['train', 'val', 'test']:
            print('Loading ' + data_type)
            sentences_path = os.path.join(
                self.data_dir, data_type, 'sentences.txt')
            tags_path = os.path.join(self.data_dir, data_type, 'tags.txt')
            self.load_sentences_tags(sentences_path, tags_path, data)
        else:
            raise ValueError("data type not in ['train', 'val', 'test']")
        return data

    def data_iterator(self, data, shuffle=False):
        """Returns a generator that yields batches data with tags.

        Args:
            data: (dict) contains data which has keys 'data', 'tags' and 'size'
            shuffle: (bool) whether the data should be shuffled

        Yields:
            batch_data: (tensor) shape: (batch_size, max_len)
            batch_tags: (tensor) shape: (batch_size, max_len)
        """

        # make a list that decides the order in which we go over the data- this avoids explicit shuffling of data
        order = list(range(data['size']))
        if shuffle:
            random.shuffle(order)

        interMode = False if 'tags' in data else True

        if data['size'] % self.batch_size == 0:
            BATCH_NUM = data['size']//self.batch_size
        else:
            BATCH_NUM = data['size']//self.batch_size + 1

        # one pass over data
        for i in range(BATCH_NUM):
            # fetch sentences and tags
            if i * self.batch_size < data['size'] < (i+1) * self.batch_size:
                sentences = [data['data'][idx]
                             for idx in order[i*self.batch_size:]]
                if not interMode:
                    tags = [data['tags'][idx]
                            for idx in order[i*self.batch_size:]]
            else:
                sentences = [data['data'][idx]
                             for idx in order[i*self.batch_size:(i+1)*self.batch_size]]
                if not interMode:
                    tags = [data['tags'][idx]
                            for idx in order[i*self.batch_size:(i+1)*self.batch_size]]

            # batch length
            batch_len = len(sentences)

            # compute length of longest sentence in batch
            batch_max_subwords_len = max([len(s[0]) for s in sentences])
            max_subwords_len = min(batch_max_subwords_len, self.max_len)
            max_token_len = 0

            # prepare a numpy array with the data, initialising the data with pad_idx
            batch_data = self.token_pad_idx * \
                np.ones((batch_len, max_subwords_len))
            batch_token_starts = []

            # copy the data to the numpy array
            for j in range(batch_len):
                cur_subwords_len = len(sentences[j][0])
                if cur_subwords_len <= max_subwords_len:
                    batch_data[j][:cur_subwords_len] = sentences[j][0]
                else:
                    batch_data[j] = sentences[j][0][:max_subwords_len]
                token_start_idx = sentences[j][-1]
                token_starts = np.zeros(max_subwords_len)
                token_starts[[
                    idx for idx in token_start_idx if idx < max_subwords_len]] = 1
                batch_token_starts.append(token_starts)
                max_token_len = max(int(sum(token_starts)), max_token_len)

            if not interMode:
                batch_tags = self.tag_pad_idx * \
                    np.ones((batch_len, max_token_len))
                for j in range(batch_len):
                    cur_tags_len = len(tags[j])
                    if cur_tags_len <= max_token_len:
                        batch_tags[j][:cur_tags_len] = tags[j]
                    else:
                        batch_tags[j] = tags[j][:max_token_len]

            # since all data are indices, we convert them to torch LongTensors
            batch_data = torch.tensor(batch_data, dtype=torch.long)
            batch_token_starts = torch.tensor(
                batch_token_starts, dtype=torch.long)
            if not interMode:
                batch_tags = torch.tensor(batch_tags, dtype=torch.long)

            # shift tensors to GPU if available
            batch_data, batch_token_starts = batch_data.to(
                self.device), batch_token_starts.to(self.device)
            if not interMode:
                batch_tags = batch_tags.to(self.device)
                yield batch_data, batch_token_starts, batch_tags
            else:
                yield batch_data, batch_token_starts


class BertForSequenceTagging(BertPreTrainedModel):
    def __init__(self, config):
        super(BertForSequenceTagging, self).__init__(config)
        self.num_labels = config.num_labels

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        self.init_weights()

    def forward(self, input_data, token_type_ids=None, attention_mask=None, labels=None,
                position_ids=None, inputs_embeds=None, head_mask=None):
        input_ids, input_token_starts = input_data
        #print("input_ids", input_ids.shape)
        #print("input_token_starts", input_token_starts.shape)
        #print("attention_mask", attention_mask.shape)
        #print("labels", labels.shape)
        outputs = self.bert(input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids,
                            position_ids=position_ids,
                            head_mask=head_mask,
                            inputs_embeds=inputs_embeds)
        sequence_output = outputs[0]
        #print("sequence_output", sequence_output.shape)

        #### 'X' label Issue Start ####
        # obtain original token representations from sub_words representations (by selecting the first sub_word)
        origin_sequence_output = [
            layer[starts.nonzero().squeeze(1)]
            for layer, starts in zip(sequence_output, input_token_starts)]
        padded_sequence_output = pad_sequence(
            origin_sequence_output, batch_first=True)
        #print("padded_sequence_output", padded_sequence_output.shape)
        padded_sequence_output = self.dropout(padded_sequence_output)
        #### 'X' label Issue End ####

        logits = self.classifier(padded_sequence_output)

        outputs = (logits,)
        if labels is not None:
            loss_mask = labels.gt(-1)
            loss_fct = CrossEntropyLoss()
            # Only keep active parts of the loss
            if loss_mask is not None:
                active_loss = loss_mask.view(-1) == 1
                active_logits = logits.view(-1, self.num_labels)[active_loss]
                active_labels = labels.view(-1)[active_loss]
                loss = loss_fct(active_logits, active_labels)
            else:
                loss = loss_fct(
                    logits.view(-1, self.num_labels), labels.view(-1))
            outputs = (loss,) + outputs

        return outputs  # (loss), scores


def train_epoch(model, data_iterator, optimizer, scheduler, params):
    """
    Train the model on `steps` batches
    """
    # set model to training mode
    model.train()

    # a running average object for loss
    loss_avg = tools.RunningAverage()

    # Use tqdm for progress bar
    one_epoch = trange(params.train_steps)
    for batch in one_epoch:
        # fetch the next training batch
        batch_data, batch_token_starts, batch_tags = next(data_iterator)
        batch_masks = batch_data.gt(0)  # get padding mask

        # compute model output and loss
        loss = model((batch_data, batch_token_starts), token_type_ids=None,
                     attention_mask=batch_masks, labels=batch_tags)[0]

        # clear previous gradients, compute gradients of all variables wrt loss
        model.zero_grad()
        loss.backward()

        # gradient clipping
        nn.utils.clip_grad_norm_(
            parameters=model.parameters(), max_norm=params.clip_grad)

        # performs updates using calculated gradients
        optimizer.step()
        scheduler.step()

        # update the average loss
        loss_avg.update(loss.item())
        one_epoch.set_postfix(loss='{:05.3f}'.format(loss_avg()))


def train_and_evaluate(model, train_data, val_data, optimizer, scheduler, params, model_dir, restore_dir=None):
    """
    Train the model and evaluate every epoch.
    """
    # reload weights from restore_dir if specified
    if restore_dir is not None:
        model = BertForSequenceTagging.from_pretrained(tagger_model_dir)

    best_val_f1 = 0.0
    patience_counter = 0

    for epoch in range(1, params.epoch_num + 1):
        # Run one epoch
        logging.info("Epoch {}/{}".format(epoch, params.epoch_num))

        # Compute number of batches in one epoch
        params.train_steps = params.train_size // params.batch_size
        params.val_steps = params.val_size // params.batch_size

        # data iterator for training
        train_data_iterator = data_loader.data_iterator(
            train_data, shuffle=True)

        # Train for one epoch on training set
        train_epoch(model, train_data_iterator, optimizer, scheduler, params)

        # data iterator for evaluation
        # train_data_iterator = data_loader.data_iterator(train_data, shuffle=False)
        val_data_iterator = data_loader.data_iterator(val_data, shuffle=False)

        # Evaluate for one epoch on training set and validation set
        # params.eval_steps = params.train_steps
        # train_metrics = evaluate(model, train_data_iterator, params, mark='Train') # callback train f1
        params.eval_steps = params.val_steps
        val_metrics = evaluate(model, val_data_iterator, params, mark='Val')

        val_f1 = val_metrics['f1']
        improve_f1 = val_f1 - best_val_f1
        if improve_f1 > 1e-5:
            logging.info("- Found new best F1")
            best_val_f1 = val_f1
            model.save_pretrained(model_dir)
            if improve_f1 < params.patience:
                patience_counter += 1
            else:
                patience_counter = 0
        else:
            patience_counter += 1

        # Early stopping and logging best f1
        if (patience_counter >= params.patience_num and epoch > params.min_epoch_num) or epoch == params.epoch_num:
            logging.info("Best val f1: {:05.2f}".format(best_val_f1))
            break


def getParam(model_dir):
    '''
    Initialize tagger model parameters
    Param:
        model_dir: directory of tagger model. Two files should be contained in
                   this directory: config.json and pytorch_model.bin
    Return:
        params: all parameters in config.json file
    '''
    json_path = os.path.join(model_dir, 'params.json')
    assert os.path.isfile(
        json_path), "No json configuration file found at {}".format(json_path)
    params = tools.Params(json_path)
    return params


def main():
    # Hyper-parameters
    tagger_model_dir = args.tagger_model_dir
    data_dir = args.data_dir
    bert_class = args.pretrain_model
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    # Load the parameters from json file
    params = getParam(tagger_model_dir)

    # Use GPUs if available
    params.device = torch.device(
        'cuda:0' if torch.cuda.is_available() else 'cpu')

    # Set the logger
    tools.set_logger(os.path.join(tagger_model_dir, 'train.log'))
    logging.info("device: {}".format(params.device))

    # Load data
    data_loader = DataLoader(data_dir, bert_class, params)

    # Start loading datasets
    logging.info("Loading the datasets...")

    # Load training data and test data
    train_data = data_loader.load_data('train')
    val_data = data_loader.load_data('val')

    # Specify the training and validation dataset sizes
    params.train_size = train_data['size']
    params.val_size = val_data['size']

    logging.info("Loading BERT model...")

    # Prepare model
    model = BertForSequenceTagging.from_pretrained(
        bert_class, num_labels=len(params.tag2idx))
    model.to(params.device)

    # Prepare optimizer
    if params.full_finetuning:
        param_optimizer = list(model.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
             'weight_decay': params.weight_decay},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
             'weight_decay': 0.0}
        ]
    else:  # only finetune the head classifier
        param_optimizer = list(model.classifier.named_parameters())
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer]}]

    optimizer = AdamW(optimizer_grouped_parameters,
                      lr=params.learning_rate, correct_bias=False)
    train_steps_per_epoch = params.train_size // params.batch_size
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=train_steps_per_epoch, num_training_steps=params.epoch_num * train_steps_per_epoch)

    # Train and evaluate the model
    logging.info("Starting training for {} epoch(s)".format(params.epoch_num))
    train_and_evaluate(model, train_data, val_data, optimizer,
                       scheduler, params, tagger_model_dir, args.restore_dir)


if __name__ == '__main__':
    # Set parser
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default=none,
                        required=true, help="name of the directory of data")
    parser.add_argument("--pretrain_model", type=str, default="bert-base-chinese",
                        required=false, help="name of the directory of tagger model")
    parser.add_argument("--tagger_model_dir", type=str, default=None,
                        required=True, help="name of the directory of tagger model")
    parser.add_argument('--restore_dir', default=None,
                        help="Optional, name of the directory containing weights to reload before training, e.g., 'experiments/conll/'")

    # Active parser
    args = parser.parse_args()

    # Main function
    main()
