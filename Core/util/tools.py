# Author: StevenChaoo
# -*- coding:UTF-8 -*-


import os
import json
import torch
import shutil
import logging
import numpy as np


class Params():
    """Class that loads hyperparameters from a json file.

    Example:
    ```
    params = Params(json_path)
    print(params.learning_rate)
    params.learning_rate = 0.5  # change the value of learning_rate in params
    ```
    """

    def __init__(self, json_path):
        with open(json_path) as f:
            params = json.load(f)
            self.__dict__.update(params)

    def save(self, json_path):
        with open(json_path, 'w') as f:
            json.dump(self.__dict__, f, indent=4)

    def update(self, json_path):
        """
        Loads parameters from json file
        Params:
            json_path: directory of config json
        """
        with open(json_path) as f:
            params = json.load(f)
            self.__dict__.update(params)

    @property
    def dict(self):
        """
        Gives dict-like access to Params instance by `params.dict['learning_rate']
        """
        return self.__dict__


class RunningAverage():
    """A simple class that maintains the running average of a quantity

    Example:
    ```
    loss_avg = RunningAverage()
    loss_avg.update(2)
    loss_avg.update(4)
    loss_avg() = 3
    ```
    """

    def __init__(self):
        self.steps = 0
        self.total = 0

    def update(self, val):
        self.total += val
        self.steps += 1

    def __call__(self):
        return self.total / float(self.steps)


def crfDataPreparation(data, label_map):
    '''
    Convert json file to data-label pair.
    Param:
        data: single piece of raw data should be like as followed.
            "{
                'originalText': 入院前4月...,
                'entities': [
                                {'label_type': '解剖部位',
                                 'overlap': 0,
                                 'start_pos': 22,
                                 'end_pos': 23},
                                {...}, ...
                            ]
             }"
        label_map: map Chinese label in English.
            "label_map = {
                 "解剖部位": "ANA",
                 "手术": "OPE",
                 "疾病和诊断": "DIS",
                 "药物": "MED",
                 "影像检查": "IMA",
                 "实验室检验": "LAB"
             }"
    Return:
        label_list: a list contains all data-label pairs of one sentence.
            "[['入', 'O'], ['院', 'O'], ...]"
    '''
    # Obtain the text and entities
    original_text = data["originalText"]
    entities = data["entities"]

    # Firstly, set all labels with "O"
    label_list = []
    for token in original_text:
        label_list.append([token, "O"])

    # Then, fine labels with BMESO scheme
    for entity in entities:
        if entity["start_pos"] - entity["end_pos"] == -1:
            label_list[entity["start_pos"]
                       ][1] = "S-{}".format(label_map[entity["label_type"]])
        else:
            for span in range(entity["start_pos"], entity["end_pos"]):
                if span == entity["start_pos"]:
                    label_list[span][1] = "B-{}".format(
                        label_map[entity["label_type"]])
                elif span == entity["end_pos"]-1:
                    label_list[span][1] = "E-{}".format(
                        label_map[entity["label_type"]])
                else:
                    label_list[span][1] = "M-{}".format(
                        label_map[entity["label_type"]])

    # return results
    return label_list


def f1_score(true_path, predict_path, lengths):
    '''
    Calculate f1 score
    Param:
        true_path: real tag list
        predict_path: predict tag list
        lengths: batch length
    '''
    batch_TP_FP = 0
    batch_TP_FN = 0
    batch_TP = 0
    for true, predict, len in zip(true_path, predict_path, lengths):
        true = true[:len]
        TP_FP = 0
        TP_FN = 0
        TP = 0
        for i in predict:
            if i == 3 or i == 5 or i == 7:
                TP_FP += 1
        for i in true:
            if i == 3 or i == 5 or i == 7:
                TP_FN += 1
        for i, index in enumerate(true):
            if predict[i] == index and index != 2:
                if index == 3 or index == 5 or index == 7:
                    TP += 1
        batch_TP_FP += TP_FP
        batch_TP_FN += TP_FN
        batch_TP += TP
    precision = batch_TP / batch_TP_FP
    recall = batch_TP / batch_TP_FN
    f1 = 2 * precision * recall / (precision + recall + 1)

    # Print results
    results = (f'precision: {precision:.2f}, '
               f'recall: {recall:.2f}, f1:{f1:.2f}')
    return precision, recall, f1


def load_checkpoint(checkpoint, model, optimizer=None):
    """Loads model parameters (state_dict) from file_path. If optimizer is provided, loads state_dict of
    optimizer assuming it is present in checkpoint.

    Args:
        checkpoint: (string) filename which needs to be loaded
        model: (torch.nn.Module) model for which the parameters are loaded
        optimizer: (torch.optim) optional: resume optimizer from checkpoint
    """
    if not os.path.exists(checkpoint):
        raise ("File doesn't exist {}".format(checkpoint))
    checkpoint = torch.load(checkpoint)
    # model.load_state_dict(checkpoint['state_dict'])
    model.load_state_dict(checkpoint['state_dict'])

    if optimizer:
        optimizer.load_state_dict(checkpoint['optim_dict'])

    return checkpoint


def save_checkpoint(state, is_best, checkpoint):
    """Saves model and training parameters at checkpoint + 'last.pth.tar'. If is_best==True, also saves
    checkpoint + 'best.pth.tar'

    Args:
        state: (dict) contains model's state_dict, may contain other keys such as epoch, optimizer state_dict
        is_best: (bool) True if it is the best model seen till now
        checkpoint: (string) folder where parameters are to be saved
    """
    filepath = os.path.join(checkpoint, 'last.pth.tar')
    if not os.path.exists(checkpoint):
        print("Checkpoint Directory does not exist! Making directory {}".format(checkpoint))
        os.mkdir(checkpoint)
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint, 'best.pth.tar'))


def set_logger(log_path):
    """
    Set the logger to log info in terminal and file `log_path`.

    In general, it is useful to have a logger so that every output to the terminal is saved
    in a permanent file. Here we save it to `model_dir/train.log`.

    Example:
    ```
    logging.info("Starting training...")
    ```

    Args:
        log_path: (string) where to log
    """
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        # Logging to a file
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(logging.Formatter(
            '%(asctime)s:%(levelname)s: %(message)s'))
        logger.addHandler(file_handler)

        # Logging to console
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(logging.Formatter('%(message)s'))
        logger.addHandler(stream_handler)
