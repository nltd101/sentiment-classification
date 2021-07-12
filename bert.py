from tensorflow.keras import models

import os
from tensorflow.python.framework.ops import disable_eager_execution
disable_eager_execution()
os.environ['TF_KERAS'] = '1'
from keras_bert.layers import Extract
import numpy as np
from keras_bert import Tokenizer
import pickle
import codecs
from tensorflow import keras
from keras_bert import load_trained_model_from_checkpoint
from tensorflow import train
from os.path import join
SEQ_LEN = 256
LR = 2e-5

token_dict = {}


class bert():

    def __init__(self):
        self.create_model()

    def load_data(self, data, sentiments):
        token_dict = {}
        indices = []
        for text in data:
            ids, segments = self.tokenizer.encode(text, max_len=SEQ_LEN)
            indices.append(ids)

        return [indices, np.zeros_like(indices)], np.array(sentiments)

    def classify_sentiment(self, text: str) -> float:
        """classify sentiment of a sentence
        Args:
            text (str): A sentence predicted
        Returns:
            probability[float]: positive probability of the sentence
        """
        print("list", text)
        list_text = [text]
        sample, _ = self.load_data(list_text, [])
        return self.model.predict(sample)

    def create_model(self):

        vocab_path = os.path.join("trained_model", 'vocab.txt')
        with codecs.open(vocab_path, 'rb', 'utf-8') as reader:
            print(vocab_path)
            for line in reader:
                token = line.strip()
                # the first word is the most negative
                token_dict[token] = len(token_dict)
        self.tokenizer = Tokenizer(token_dict, cased=True)

        self.model = load_trained_model_from_checkpoint(
            os.path.join(
                "trained_model", 'bert_config.json'),
            os.path.join(
                "trained_model", 'bert_model.ckpt'),
          
            trainable=True,
            seq_len=SEQ_LEN,
            output_layer_num=4
        )

        inputs = self.model.inputs[:2]
        newout = Extract(index=0)(self.model.output)
        newout = keras.layers.Dense(768, activation='relu')(newout)
        outputs = keras.layers.Dense(units=1, activation='sigmoid')(newout)

        self.model = models.Model(inputs, outputs)
        self.model.load_weights(join("last_weight", "lastweight"))


# a = bert()
# print(a.classify_sentiment("Ngon quá bạn ơi"))
# print(a.classify_sentiment("Ngon quá bạn ơi"))
# print(a.classify_sentiment("Dở quá, đồ ăn không ngon"))
# print(a.classify_sentiment("Nhất quyết không ghé lại"))
