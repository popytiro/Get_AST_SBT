import tensorflow as tf

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from sklearn.model_selection import train_test_split

import unicodedata
import re
import numpy as np
import os
import io
import time

sbt_start_end_list = []
mehtod_name_start_end_list = []

sbt_sentense = open('C:/Users/acmil/Desktop/boom/replace.test.token.sbt', 'r', encoding="utf").readlines()
method_name_sentense = open('C:/Users/acmil/Desktop/boom/replace.test.token.nl', 'r', encoding="utf").readlines()


def preprocess_sentence(w):
    w_list = w.split()
    w_list.insert(0, '<start> ')
    w_list.append(' <end>')
    w_list.append('\n')
    w_str = ' '.join(w_list)
    return w_str



# test_sentence = '( METHOD DECLARATION ( STATEMENT EXPRESSION ( METHOD INVOCATION ( MEMBER REFERENCE ) MEMBER REFERENCE ) METHOD INVOCATION ) STATEMENT EXPRESSION ( STATEMENT EXPRESSION ( METHOD INVOCATION ( MEMBER REFERENCE ) MEMBER REFERENCE ( MEMBER REFERENCE ) MEMBER REFERENCE ( MEMBER REFERENCE ) MEMBER REFERENCE ) METHOD INVOCATION ) STATEMENT EXPRESSION ( STATEMENT EXPRESSION ( METHOD INVOCATION ( MEMBER REFERENCE ) MEMBER REFERENCE ( MEMBER REFERENCE ) MEMBER REFERENCE ( MEMBER REFERENCE ) MEMBER REFERENCE ) METHOD INVOCATION ) STATEMENT EXPRESSION ) METHOD DECLARATION'

# print(preprocess_sentence(test_sentence))


for sbt_line in sbt_sentense:
    sbt_line = preprocess_sentence(sbt_line)
    sbt_start_end_list.append(sbt_line)
    with open('sbt.start.end', 'a', encoding="utf-8") as w_sbt_f:
        w_sbt_f.write(sbt_line)

for method_name_line in method_name_sentense:
    method_name_line = preprocess_sentence(method_name_line)
    mehtod_name_start_end_list = method_name_line
    with open('method_name.start.end', 'a', encoding="utf-8") as w_method_name_f:
        w_method_name_f.write(method_name_line)


def create_dataset(path, num_examples):
    lines = io.open(path, encoding='UTF-8').read().strip().split('\n')

    word_pairs = [[preprocess_sentence(w) for w in l.split('\t')]  for l in lines[:num_examples]]

    return zip(*word_pairs)