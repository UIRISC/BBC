# Author: StevenChaoo
# -*- coding:UTF-8 -*-


import json
import argparse
import logging
import time
import random
import sys

from sklearn_crfsuite import CRF
from sklearn.metrics import classification_report
from util import tools
from tqdm import tqdm


logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO
)
logger = logging.getLogger("root")


def prepareDate(data_file, label_map, data_type):
    '''
    Label data
    Param:
        data_file: json file
        label_map: map Chinese label in English
            "label_map = {
                 "解剖部位": "ANA",
                 "手术": "OPE",
                 "疾病和诊断": "DIS",
                 "药物": "MED",
                 "影像检查": "IMA",
                 "实验室检验": "LAB"
             }"
        data_type: "train" or "dev" or "test"
    Return:
        word_lists: a list of all sentences seperate in token
            "[["We", "are", ...], ...]"
        label_lists: a list of all sentences seperate in token cooresponding
                     with label
            "[["O", "O", ...], ...]"
   '''
    # Train : dev : test = 7 : 1 : 2
    data_types = {
        "train": {"start": 0, "end": 700},
        "dev": {"start": 700, "end": 800},
        "test": {"start": 800, "end": 1000}
    }

    # Load json file
    all_data_list = json.load(data_file)
    data_list = all_data_list[data_types[data_type]
                              ["start"]:data_types[data_type]["end"]]

    # Label data
    word_lists, label_lists = [], []
    for data in data_list:
        raw_data_label_list = tools.crfDataPreparation(data, label_map)
        word_list, label_list = [], []
        for word_label_pair in raw_data_label_list:
            word_list.append(word_label_pair[0])
            label_list.append(word_label_pair[1])
        word_lists.append(word_list)
        label_lists.append(label_list)

    # Return results
    return word_lists, label_lists


def sentence2feature(sentences):
    '''
    Extract feature from sentence
    Param:
        sentences: a list of sentences
            "[["We", "are", ...], ...]"
    Return:
        features: features of sentences
            "[
               {
                 "w": "We",
                 "w-1": "<start>",
                 "w+1": "are",
                 "w-1:w": <start>We,
                 "w:w+1": Weare
               }, {...}, ...
             ]"
    '''
    # Extract context features
    features = []
    for sentence in tqdm(sentences):
        result = []
        for i in range(len(sentence)):
            # Previous and next word
            word = sentence[i]
            previous_word = '<start>' if i == 0 else sentence[i-1]
            next_word = '<end>' if i == (len(sentence)-1) else sentence[i+1]

            # Contains five features
            feature = {
                "w": word,
                "w-1": previous_word,
                "w+1": next_word,
                "w-1:w": previous_word+word,
                "w:w+1": word+next_word
            }
            result.append(feature)
        features.append(result)

    # Return results
    return features


def normalizationLabel(label_lists):
    '''
    Drop BMESO label
    Param:
        label_lists: a list of labels in sentences
    Return:
        labels: a list of labels
    '''
    labels = []
    for label_list in label_lists:
        for label in label_list:
            if len(label) > 1:
                labels.append(label[2:])
            else:
                labels.append(label)
    return labels


class CRFModel(object):
    '''
    Build CRF model
    Func:
        train(self, features, tag_lists): Fit model
        evaluate(self, features, tagPlists): Evaluate model
    '''

    def __init__(self):
        self.model = CRF(algorithm='l2sgd',
                         c2=0.1,
                         max_iterations=100)

    def train(self, features, tag_lists):
        '''
        Fit model
        Param:
            features: sentences features
            tag_lists: a list of tags of sentences
        '''
        self.model.fit(features, tag_lists)

    def evaluate(self, features, tag_lists):
        '''
        Evaluate model
        Param:
            features: sentences features
            tag_lists: a list of tags of sentences
        '''
        predict_tag = self.model.predict(features)
        real_tag = normalizationLabel(tag_lists)
        pred_tag = normalizationLabel(predict_tag)
        print(classification_report(real_tag, pred_tag))


def main():
    # Label map
    label_map = {
        "解剖部位": "ANA",
        "手术": "OPE",
        "疾病和诊断": "DIS",
        "药物": "MED",
        "影像检查": "IMA",
        "实验室检验": "LAB"
    }

    # Prepare train / test data
    raw_data_file_for_train = open(
        args.data_dir, "r", encoding=args.encoding_type)
    raw_data_file_for_test = open(
        args.data_dir, "r", encoding=args.encoding_type)
    train_word_lists, train_label_lists = prepareDate(
        raw_data_file_for_train, label_map, data_type="train")
    test_word_lists, test_label_lists = prepareDate(
        raw_data_file_for_test, label_map, data_type="test")

    # Extract features
    logger.info("Prepare train data")
    train_features = sentence2feature(train_word_lists)
    logger.info("Prepare test data")
    test_features = sentence2feature(test_word_lists)

    # Build CRF model
    logger.info("Build CRF model")
    crf = CRFModel()
    logger.info("Success!")

    # Train model
    logger.info("Begin training")
    crf.train(train_features, train_label_lists)
    logger.info("Finish training")

    # Evaluate model
    logger.info("Begin evaluating")
    crf.evaluate(test_features, test_label_lists)
    logger.info("Finish evaluating")


if __name__ == "__main__":
    # Set parser
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default=None, required=True,
                        help="path to the raw dataset")
    parser.add_argument("--encoding_type", type=str, default=None,
                        required=True, choices=["utf-8", "utf-8-sig"])
    parser.add_argument("--seed", type=int, default=0)

    # Active parser
    args = parser.parse_args()

    # Print args
    logger.info(args)

    # Main function
    main()
