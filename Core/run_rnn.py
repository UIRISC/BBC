# Author: StevenChaoo
# -*- coding:UTF-8 -*-
import argparse
import json
import pickle
import codecs
import copy
import torch
import torch.nn as nn
import torch.optim as optim

from util import tools

import logging

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO
)
logger = logging.getLogger("root")


class Data_preprocess():
    def __init__(self, batch_size=64, data_type='train'):
        self.batch_size = batch_size
        self.data_type = data_type
        self.data = []
        self.batch_data = []
        self.vocab = {'unk': 0}
        self.tags_map = {'<START>': 0, '<STOP>': 1}

        # Load train / dev / test data
        if data_type == 'train':
            self.data_path = '{}/train.txt'.format(args.raw_data_dir)
        elif data_type == 'dev':
            self.data_path = '{}/dev.txt'.format(args.raw_data_dir)
            self.load_data_map()
        elif data_type == 'test':
            self.data_path = '{}/test.txt'.format(args.raw_data_dir)
            self.load_data_map()

        # Format self.data, self.vocab, self.tags_map
        self.load_data()

        # Pad self.data
        self.prepare_batch()

    def load_data_map(self):
        '''
        Load data information builded in training stage when loading dev and
        test data
        Return:
            self.vocab: training stage token dictionary
            self.tags_map: training stage label dictionary
            self.tags: training stage labels
        '''
        with codecs.open('{}/data.pkl'.format(args.save_dir), 'rb') as f:
            self.data_map = pickle.load(f)
            self.vocab = self.data_map.get('vocab', {})
            self.tags_map = self.data_map.get('tags_map', {})
            self.tags = self.data_map.keys()

    def load_data(self):
        '''
        Load data and print some INFO
        Return:
            self.vocab: a dictionary of all tokens with value of token id
                "{"unk": 0, "我": 1, ...}"
            self.tags_map: a dictionary of all labels
                "{"<START>": 0, "<STOP>": 1, "B-ANA": 2, ...}"
            self.data: a list of sentence-labels list
                "[[["我", "是", ...], ["O", "O", ...]], ...]"
        '''
        # Initialize sentence list and label list
        sentence = []
        target = []

        # Load data from self.data_path
        with codecs.open(self.data_path, 'r', encoding="utf-8") as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip()

                # Save when one sentence loaded
                if len(line) == 0:
                    self.data.append([sentence, target])
                    sentence = []
                    target = []
                    continue

                # Decrease negative influence of data annotation
                elif len(line.split()) == 1:
                    continue

                # Get token and label
                word, tag = line.split()

                # Only append in training stage
                if word not in self.vocab and self.data_type == 'train':
                    self.vocab[word] = len(self.vocab)
                if tag not in self.tags_map and self.data_type == 'train':
                    self.tags_map[tag] = len(self.tags_map)

                # Append token and label to sentence and target list
                sentence.append(self.vocab.get(word, 0))
                target.append(self.tags_map.get(tag, 0))

        # Print some INFO
        logger.info(self.tags_map)
        self.input_size = len(self.vocab)
        logger.info("{} data: {}".format(self.data_type, self.data))
        logger.info("vocab size: {}".format(self.input_size))
        logger.info("unique tag: {}".format(self.tags_map))

    def prepare_batch(self):
        '''
        Seperate data under batch_size
        '''
        index = 0
        while True:

            # If the length of remain part is shorter than batch_size
            if index + self.batch_size >= len(self.data):
                pad_data = self.pad_data(self.data[-self.batch_size:])
                self.batch_data.append(pad_data)
                break

            # If the length of remain part is longer than batch_size
            else:
                pad_data = self.pad_data(
                    self.data[index: index+self.batch_size])
                index += self.batch_size
                self.batch_data.append(pad_data)

    def pad_data(self, data):
        '''
        Pad data with "0" to the max length of sentence in current batch
        Param:
            data: the sentence in current batch
                "[[["I", "come", "from", "China"],
                   ["O", "O", "O", "S"]],
                  [["He", "comes", "from", "the", "United", "States"],
                   ["O", "O", "O", "B", "M", "E"]]]"
        Return:
            c_data: padded data
                "[[["I", "come", "from", "China", 0, 0],
                   ["O", "O", "O", "S", 0, 0],
                   4],
                  [["He", "comes", "from", "the", "United", "States"],
                   ["O", "O", "O", "B", "M", "E"],
                   6]]"
        '''
        # Deep copy
        c_data = copy.deepcopy(data)

        # Get the max length of sentence in current batch
        max_length = max([len(i[0]) for i in c_data])

        # Padding
        for c in c_data:
            c.append(len(c[0]))
            c[0] += (max_length - len(c[0])) * [0]
            c[1] += (max_length - len(c[1])) * [0]

        # Return results
        return c_data

    def iteration(self):
        '''
        Push batches of data into iterator
        '''
        idx = 0
        while True:
            yield self.batch_data[idx]
            idx += 1
            if idx > len(self.batch_data)-1:
                idx = 0

    def get_batch(self):
        '''
        Push data into iterator
        '''
        for data in self.batch_data:
            yield data


class BiLSTM_CRF(nn.Module):
    START_TAG = "<START>"
    STOP_TAG = "<STOP>"
    BATCH_SIZE= 128
    EMBEDDING_SIZE = 100
    HIDDEN_SIZE = 128
    DROPOUT = 1.0

    def __init__(self, vocab_size, tag_to_ix,
                 embedding_dim=100, hidden_dim=128,
                 batch_size=64):
        START_TAG = "<START>"
        STOP_TAG = "<STOP>"
        BATCH_SIZE = 128
        EMBEDDING_SIZE = 100
        HIDDEN_SIZE = 128
        DROPOUT = 1.0

        super(BiLSTM_CRF, self).__init__()
        # Hyper-parameters
        self.vocab_size = vocab_size
        self.tag_to_ix = tag_to_ix
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.tagset_size = len(tag_to_ix)
        self.batch_size = batch_size
        # Build embedding
        self.word_embeds = nn.Embedding(self.vocab_size, self.embedding_dim)
        # Build LSTM layer
        self.lstm = nn.LSTM(self.embedding_dim, self.hidden_dim//2,
                            num_layers=1, bidirectional=True,
                            batch_first=True)
        # Maps the output of the LSTM into tag space.
        self.hidden2tag = nn.Linear(self.hidden_dim, self.tagset_size)
        # Matrix of transition parameters. Entry i,j is
        # the score of transitioning from i to j
        self.transitions = nn.Parameter(torch.randn(self.tagset_size,
                                                    self.tagset_size))
        # These two statements enforce the constraint
        # that we never transfer to the start tag and
        # we never transfer from the stop tag
        self.transitions.data[:, self.tag_to_ix[START_TAG]] = -10000.
        self.transitions.data[self.tag_to_ix[STOP_TAG], :] = -10000.
        # Initialize hidden layer
        self.hidden = self.init_hidden()

    def log_sum_exp(self, vec):
        '''
        Official document
        '''
        # compute log sum exp in a numerically stable way for the forward algorithm
        max_score = torch.max(vec, 0)[0].unsqueeze(0)
        max_score_broadcast = max_score.expand(vec.size(1), vec.size(1))
        result = max_score + \
            torch.log(
                torch.sum(torch.exp(vec - max_score_broadcast), 0)).unsqueeze(0)
        return result.squeeze(1)

    def init_hidden(self):
        '''
        Initialize hidden layer
        '''
        return (torch.randn(2, self.batch_size, self.hidden_dim//2),
                torch.randn(2, self.batch_size, self.hidden_dim//2))

    def _get_lstm_features(self, sentence):
        '''
        Official document
        '''
        self.hidden = self.init_hidden()
        seq_len = sentence.size(1)
        embeds = self.word_embeds(sentence).view(
            self.batch_size, seq_len, self.embedding_dim)
        lstm_out, self.hidden = self.lstm(embeds, self.hidden)
        lstm_out = lstm_out.view(self.batch_size, -1, self.hidden_dim)
        lstm_feats = self.hidden2tag(lstm_out)
        return lstm_feats

    def _forward_alg(self, emissions):
        '''
        Official document
        '''
        START_TAG = "<START>"
        STOP_TAG = "<STOP>"
        BATCH_SIZE = 128
        EMBEDDING_SIZE = 100
        HIDDEN_SIZE = 128
        DROPOUT = 1.0

        previous = torch.full((1, self.tagset_size), 0)
        for index in range(len(emissions)):
            previous = torch.transpose(previous.expand(
                self.tagset_size, self.tagset_size), 0, 1)
            obs = emissions[index].view(
                1, -1).expand(self.tagset_size, self.tagset_size)
            scores = previous + obs + self.transitions
            previous = self.log_sum_exp(scores)
        previous = previous + self.transitions[:, self.tag_to_ix[STOP_TAG]]
        # calculate total_scores
        total_scores = self.log_sum_exp(torch.transpose(previous, 0, 1))[0]
        return total_scores

    def _score_sentences(self, emissions, tags):
        '''
        Official document
        '''
        START_TAG = "<START>"
        STOP_TAG = "<STOP>"
        BATCH_SIZE = 128
        EMBEDDING_SIZE = 100
        HIDDEN_SIZE = 128
        DROPOUT = 1.0

        # Gives the score of a provided tag sequence
        # Score = Emission_Score + Transition_Score
        score = torch.zeros(1)
        tags = torch.cat(
            [torch.tensor([self.tag_to_ix[START_TAG]], dtype=torch.long), tags])
        for i, emission in enumerate(emissions):
            score += self.transitions[tags[i], tags[i+1]] + emission[tags[i+1]]
        score += self.transitions[tags[-1], self.tag_to_ix[STOP_TAG]]
        return score

    def neg_log_likelihood(self, sentences, tags, length):
        '''
        Official document
        '''
        self.batch_size = sentences.size(0)
        emissions = self._get_lstm_features(sentences)
        gold_score = torch.zeros(1)
        total_score = torch.zeros(1)
        for emission, tag, len in zip(emissions, tags, length):
            emission = emission[:len]
            tag = tag[:len]
            gold_score += self._score_sentences(emission, tag)
            total_score += self._forward_alg(emission)
        return (total_score - gold_score) / self.batch_size

    def _viterbi_decode(self, emissions):
        '''
        Official document
        '''
        trellis = torch.zeros(emissions.size())
        backpointers = torch.zeros(emissions.size(), dtype=torch.long)
        trellis[0] = emissions[0]
        for t in range(1, len(emissions)):
            v = trellis[t -
                        1].unsqueeze(1).expand_as(self.transitions) + self.transitions
            trellis[t] = emissions[t] + torch.max(v, 0)[0]
            backpointers[t] = torch.max(v, 0)[1]
        viterbi = [torch.max(trellis[-1], -1)[1].cpu().tolist()]
        backpointers = backpointers.cpu().numpy()
        for bp in reversed(backpointers[1:]):
            viterbi.append(bp[viterbi[-1]])
        viterbi.reverse()
        viterbi_score = torch.max(trellis[-1], 0)[0].cpu().tolist()
        return viterbi_score, viterbi

    def forward(self, sentences, lengths=None):
        '''
        Official document
        '''
        sentence = torch.tensor(sentences, dtype=torch.long)
        if not lengths:
            lengths = [sen.size(-1) for sen in sentence]
        self.batch_size = sentence.size(0)
        emissions = self._get_lstm_features(sentence)
        scores = []
        paths = []
        for emission, len in zip(emissions, lengths):
            emission = emission[:len]
            score, path = self._viterbi_decode(emission)
            scores.append(score)
            paths.append(path)
        return scores, paths


class ChineseNER():
    START_TAG = "<START>"
    STOP_TAG = "<STOP>"
    BATCH_SIZE= 128
    EMBEDDING_SIZE = 100
    HIDDEN_SIZE = 128
    DROPOUT = 1.0

    def __init__(self):
        START_TAG = "<START>"
        STOP_TAG = "<STOP>"
        BATCH_SIZE = 128
        EMBEDDING_SIZE = 100
        HIDDEN_SIZE = 128
        DROPOUT = 1.0

        # Set training data
        self.train_manager = Data_preprocess(batch_size=BATCH_SIZE)

        # How many batches need to train
        self.total_size = len(self.train_manager.batch_data)

        # Initialize data dictionary with train_manager
        data = {
            'batch_size': self.train_manager.batch_size,
            'input_size': self.train_manager.input_size,
            'vocab': self.train_manager.vocab,
            'tags_map': self.train_manager.tags_map
        }

        # Save parameters of data to data.pkl
        self.save_params(data)

        # Set dev data
        dev_manager = Data_preprocess(
            batch_size=BATCH_SIZE, data_type='dev')

        # Push dev data into iterator
        self.dev_batch = dev_manager.iteration()

        # Build BiLSTM CRF model
        self.model = BiLSTM_CRF(
            vocab_size=len(self.train_manager.vocab),
            tag_to_ix=self.train_manager.tags_map,
            embedding_dim=EMBEDDING_SIZE,
            hidden_dim=HIDDEN_SIZE,
            batch_size=BATCH_SIZE
        )
        self.model

    def save_params(self, data):
        '''
        Save parameters of data to data.pkl
        Param:
            data:
                "{'batch_size': self.train_manager.batch_size,
                  'input_size': self.train_manager.input_size,
                  'vocab': self.train_manager.vocab,
                  'tags_map': self.train_manager.tags_map}"
        '''
        with codecs.open('{}/data.pkl'.format(args.save_dir), 'wb') as f:
            pickle.dump(data, f)

    def restore_model(self):
        '''
        Load model's weight
        '''
        self.model.load_state_dict(torch.load(
            '{}/params.pkl'.format(args.save_dir)))
        logger.info("model restore success!")

    def load_params(self):
        '''
        Load parameters from data.pkl
        Return:
            data_map:
                "{'batch_size': self.train_manager.batch_size,
                  'input_size': self.train_manager.input_size,
                  'vocab': self.train_manager.vocab,
                  'tags_map': self.train_manager.tags_map}"
        '''
        with codecs.open('{}/data.pkl'.format(args.save_dir), 'rb') as f:
            data_map = pickle.load(f)
        return data_map

    def train(self):
        '''
        Train model
        '''
        # Build optimizer
        optimizer = optim.Adam(self.model.parameters())

        # Epoch is 50
        for epoch in range(50):
            index = 0
            for batch in self.train_manager.get_batch():
                index += 1

                # Clean grad
                self.model.zero_grad()

                # Transfer sentence, tag, and length to tensor type
                sentences, tags, length = zip(*batch)
                sentences_tensor = torch.tensor(
                    sentences, dtype=torch.long)
                tags_tensor = torch.tensor(tags, dtype=torch.long)
                length_tensor = torch.tensor(length, dtype=torch.long)

                # Define loss function
                loss = self.model.neg_log_likelihood(sentences_tensor,
                                                     tags_tensor,
                                                     length_tensor)

                # Print progress
                progress = ('█' * int(index * 25 / self.total_size)).ljust(25)
                print(f'epoch [{epoch+1}] |{progress}| '
                      f'{index}/{self.total_size}\n\t'
                      f'loss {loss.cpu().tolist()[0]:.2f}')
                print('-'*50)

                # Update weights
                loss.backward()
                optimizer.step()

            # Evaluate with dev data
            self.evaluate()
            print('*' * 50)

            # Save model's weight
            torch.save(self.model.state_dict(),
                       '{}/params.pkl'.format(args.save_dir))

    def evaluate(self):
        '''
        Evaluate with dev data
        '''
        sentences, tags, lengths = zip(*self.dev_batch.__next__())
        _, paths = self.model(sentences, lengths)
        logger.info("Evaluation")

        # Calculate f1 score
        pre, cal, f1 = tools.f1_score(tags, paths, lengths)
        logger.info(
            "Precision: {:.2f}, Recall: {:.2f}, F1 Score: {:.2f}".format(pre, cal, f1))


def dataPreparation():
    # Label map
    label_map = {
        "解剖部位": "ANA",
        "手术": "OPE",
        "疾病和诊断": "DIS",
        "药物": "MED",
        "影像检查": "IMA",
        "实验室检验": "LAB"
    }

    # Load raw data
    raw_data_file = open("{}/Raw_data.json".format(args.raw_data_dir),
                         "r", encoding=args.encoding_type)
    raw_data_list = json.load(raw_data_file)

    # Prepare training / test data
    train_data = open("{}/train.txt".format(args.raw_data_dir), "w", encoding="utf-8")
    dev_data = open("{}/dev.txt".format(args.raw_data_dir), "w", encoding="utf-8")
    test_data = open("{}/test.txt".format(args.raw_data_dir), "w", encoding="utf-8")

    # 1-700 for train.txt, 701-800 for dev.txt, 801-1000 for test.txt
    count = 0

    # Write in files
    for raw_data in raw_data_list:
        count += 1
        # raw_data_label_list = labeled(raw_data, label_map)
        raw_data_label_list = tools.crfDataPreparation(raw_data, label_map)
        for raw_data_label in raw_data_label_list:
            data_label_pair = raw_data_label[0] + " " + raw_data_label[1]
            if count <= 700:
                train_data.write(data_label_pair)
                train_data.write("\n")
            elif count <= 800 and count > 700:
                dev_data.write(data_label_pair)
                dev_data.write("\n")
            else:
                test_data.write(data_label_pair)
                test_data.write("\n")
        if count <= 700:
            train_data.write("\n")
        elif count <= 800 and count > 700:
            dev_data.write("\n")
        else:
            test_data.write("\n")

    # Close files
    train_data.close()
    dev_data.close()
    test_data.close()


def main():
    # Hyper-parameters
    START_TAG = "<START>"
    STOP_TAG = "<STOP>"
    BATCH_SIZE= 128
    EMBEDDING_SIZE = 100
    HIDDEN_SIZE = 128
    DROPOUT = 1.0

    # Prepare train & dev & test data
    if args.generate_data:
        dataPreparation()

    # Train & evaluate
    ner = ChineseNER()
    ner.train()


if __name__ == '__main__':
    # Set parser
    parser = argparse.ArgumentParser()
    parser.add_argument("--generate_data", type=bool, default=False,
                        required=True, help="if need to generate data")
    parser.add_argument("--raw_data_dir", type=str, default=None,
                        required=True, help="path to the raw dataset")
    parser.add_argument("--encoding_type", type=str, default=None,
                        required=True, choices=["utf-8", "utf-8-sig"])
    parser.add_argument("--save_dir", type=str, default=None,
                        required=True, help="path where data saved")
    parser.add_argument("--seed", type=int, default=0)

    # Active parser
    args = parser.parse_args()

    # Print args
    logger.info(args)

    # Main function
    main()
