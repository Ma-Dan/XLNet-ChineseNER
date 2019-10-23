# -*- coding:utf-8 -*-
from absl import flags
import json
import math
import pickle as cPickle
import random
import sys

import numpy as np
import tensorflow as tf
import modeling, xlnet, model_utils
import sentencepiece as spm
from prepro_utils import preprocess_text, encode_ids
from tensorflow.contrib import rnn, crf
from tensorflow.contrib.crf import crf_log_likelihood
from tensorflow.contrib.layers.python.layers import initializers

from utils import f1_score, format_result, get_tags, format_tags, new_f1_score

# Model
flags.DEFINE_float("dropout", default=0.1,
      help="Dropout rate.")
flags.DEFINE_float("dropatt", default=0.1,
      help="Attention dropout rate.")
flags.DEFINE_integer("clamp_len", default=-1,
      help="Clamp length.")
flags.DEFINE_bool("use_bfloat16", default=False,
      help="Whether to use bfloat16.")

# Parameter initialization
flags.DEFINE_enum("init", default="normal",
                  enum_values=["normal", "uniform"],
                  help="Initialization method.")
flags.DEFINE_float("init_std", default=0.02,
                   help="Initialization std when init is normal.")
flags.DEFINE_float("init_range", default=0.1,
                   help="Initialization std when init is uniform.")

# TPUs and machines
flags.DEFINE_bool("use_tpu", default=False, help="whether to use TPU.")

# Training
flags.DEFINE_integer("train_batch_size", default=1,
                     help="batch size for training")
flags.DEFINE_integer("train_steps", default=8000,
                     help="Number of training steps")
flags.DEFINE_integer("warmup_steps", default=0, help="number of warmup steps")
flags.DEFINE_integer("save_steps", default=None,
                     help="Save the model for every save_steps. "
                     "If None, not to save any model.")
flags.DEFINE_integer("max_save", default=5,
                     help="Max number of checkpoints to save. "
                     "Use 0 to save all.")
flags.DEFINE_integer("shuffle_buffer", default=2048,
                     help="Buffer size used for shuffle.")

# Optimization
flags.DEFINE_float("learning_rate", default=3e-5, help="initial learning rate")
flags.DEFINE_float("min_lr_ratio", default=0.0,
                   help="min lr ratio for cos decay.")
flags.DEFINE_float("clip", default=1.0, help="Gradient clipping")
flags.DEFINE_float("weight_decay", default=0.00, help="Weight decay rate")
flags.DEFINE_float("adam_epsilon", default=1e-6, help="Adam epsilon")
flags.DEFINE_string("decay_method", default="poly", help="poly or cos")
flags.DEFINE_float("lr_layer_decay_rate", default=0.75,
                   help="Top layer: lr[L] = FLAGS.learning_rate."
                   "Lower layers: lr[l-1] = lr[l] * lr_layer_decay_rate.")

flags.DEFINE_string("data_dir", default="data/", help="tran, dev and test data dir", )
flags.DEFINE_string("xlnet_config", default="chinese_xlnet_base_L-12_H-768_A-12/xlnet_config.json", help="xlnet config file dir")
flags.DEFINE_string("init_checkpoint", default="chinese_xlnet_base_L-12_H-768_A-12/xlnet_model.ckpt", help="xlnet model init checkpoint")
flags.DEFINE_string("spm", default="chinese_xlnet_base_L-12_H-768_A-12/spiece.model", help="spiece model file")
flags.DEFINE_string("entry", default="train", help="operation")

FLAGS = flags.FLAGS

class Model():
    def __init__(self):
        self.nums_tags = 4
        self.lstm_dim = 128
        self.embedding_size = 50
        self.max_epoch = 10
        self.global_steps = tf.Variable(0, trainable=False)
        self.best_dev_f1 = tf.Variable(0.0, trainable=False)
        self.checkpoint_dir = "./model/"
        self.checkpoint_path = "./model/ner.ckpt"
        self.initializer = initializers.xavier_initializer()

        self.is_training = True if FLAGS.entry=="train" else False

    def __creat_model(self):

        # xlnet embbeding layer
        self._init_xlnet_placeholder()
        self.xlnet_layer()

        # bi-Lstm layer
        self.biLSTM_layer()

        # logits_layer
        self.logits_layer()

        # loss_layer
        self.loss_layer()

        # crf_layer
        self.crf_layer()

        # optimizer_layer
        self.xlnet_optimizer_layer()

    def _init_xlnet_placeholder(self):
        self.input_ids = tf.placeholder(
            dtype=tf.int32,
            shape=[None, None],
            name="xlnet_input_ids"
        )
        self.input_mask = tf.placeholder(
            dtype=tf.float32,
            shape=[None, None],
            name="xlnet_input_mask"
        )
        self.segment_ids = tf.placeholder(
            dtype=tf.int32,
            shape=[None, None],
            name="xlnet_segment_ids"
        )
        self.targets = tf.placeholder(
            dtype=tf.int32,
            shape=[None, None],
            name="xlnet_targets"
        )

        self.dropout = tf.placeholder(
            dtype=tf.float32,
            shape=None,
            name="xlnet_dropout"
        )
        used = tf.sign(tf.abs(self.input_ids))
        length = tf.reduce_sum(used, reduction_indices=1)
        self.length = tf.cast(length, tf.int32)
        self.nums_steps = tf.shape(self.input_ids)[-1]

    def xlnet_layer(self):
        xlnet_config = xlnet.XLNetConfig(json_path = FLAGS.xlnet_config)
        run_config = xlnet.create_run_config(self.is_training, True, FLAGS)

        xlnet_model = xlnet.XLNetModel(
            xlnet_config = xlnet_config,
            run_config = run_config,
            input_ids = self.input_ids,
            seg_ids = self.segment_ids,
            input_mask = self.input_mask)

        self.embedded = xlnet_model.get_sequence_output()
        self.model_inputs = tf.nn.dropout(
            self.embedded, self.dropout
        )

    def embedding_layer(self):
        with tf.variable_scope("embedding_layer") as scope:
            sqart3 = math.sqrt(3)

            self.embedding_matrix = tf.get_variable(
                name="embedding_matrix",
                shape=[self.input_size, self.embedding_size],
                initializer=self.initializer,
                dtype=tf.float32,
            )

            self.embedded = tf.nn.embedding_lookup(
                self.embedding_matrix, self.inputs
            )

            self.model_inputs = tf.nn.dropout(
                self.embedded, self.dropout
            )

    def biLSTM_layer(self):
        with tf.variable_scope("bi-LSTM") as scope:
            lstm_cell = {}
            for direction in ["forward", "backward"]:
                with tf.variable_scope(direction):
                    lstm_cell[direction] = rnn.LSTMCell(
                        num_units=self.lstm_dim,
                    )

            outputs, final_states = tf.nn.bidirectional_dynamic_rnn(
                cell_fw=lstm_cell['forward'],
                cell_bw=lstm_cell['backward'],
                inputs=self.model_inputs,
                sequence_length=self.length,
                dtype=tf.float32,
            )
            self.lstm_outputs = tf.concat(outputs, axis=2)

    def logits_layer(self):
        with tf.variable_scope("hidden"):
            w = tf.get_variable("W", shape=[self.lstm_dim*2, self.lstm_dim],
                                dtype=tf.float32, initializer=self.initializer
                                )
            b = tf.get_variable("b", shape=[self.lstm_dim], dtype=tf.float32,
                                initializer=self.initializer
                                )

            output = tf.reshape(self.lstm_outputs, shape=[-1, self.lstm_dim*2])
            hidden = tf.tanh(tf.nn.xw_plus_b(output, w, b))
            self.hidden = hidden

        with tf.variable_scope("logits"):
            w = tf.get_variable("W", shape=[self.lstm_dim, self.nums_tags],
                                initializer=self.initializer, dtype=tf.float32
                                )
            self.test_w = w
            b = tf.get_variable("b", shape=[self.nums_tags], dtype=tf.float32)
            self.test_b = b
            pred = tf.nn.xw_plus_b(hidden, w, b)
            self.logits = tf.reshape(
                pred, shape=[-1, self.nums_steps, self.nums_tags])

    def loss_layer(self):
        with tf.variable_scope("loss_layer"):
            logits = self.logits
            targets = self.targets

            self.trans = tf.get_variable(
                "transitions",
                shape=[self.nums_tags, self.nums_tags],
                initializer=self.initializer
            )

            log_likelihood, self.trans = crf_log_likelihood(
                inputs=logits,
                tag_indices=targets,
                transition_params=self.trans,
                sequence_lengths=self.length
            )
            self.loss = tf.reduce_mean(-log_likelihood)

    def crf_layer(self):
        # CRF decode, pred_ids 是一条最大概率的标注路径
        self.pred_ids, _ = crf.crf_decode(potentials=self.logits, transition_params=self.trans, sequence_length=self.length)

    def xlnet_optimizer_layer(self):
        correct_prediction = tf.equal(
            tf.argmax(self.logits, 2), tf.cast(self.targets, tf.int64))
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        FLAGS.train_steps = int(
            self.train_length / FLAGS.train_batch_size * self.max_epoch)
        FLAGS.warmup_steps = int(FLAGS.train_steps * 0.1)

        self.train_op, self.learning_rate, _ = model_utils.get_train_op(FLAGS, self.loss)
        self.saver = tf.train.Saver(tf.global_variables(), max_to_keep=5)

    def xlnet_step(self, sess, batch):
        try:
            ntokens, tag_ids, inputs_ids, segment_ids, input_mask = zip(*batch)
        except ValueError:
            print("ValueError: not enough values to unpack")
            return 0, 0, 0, 0, 0

        feed = {
            self.input_ids: inputs_ids,
            self.targets: tag_ids,
            self.segment_ids: segment_ids,
            self.input_mask: input_mask,
            self.dropout: 0.5
        }
        embedding, global_steps, loss, _, logits, acc, length = sess.run([self.embedded, self.global_steps, self.loss, self.train_op, self.logits, self.accuracy, self.length], feed_dict=feed)
        return global_steps, loss, logits, acc, length

    def train(self):
        from xlnet_data_utils import XLNetDataUtils
        sp_model = spm.SentencePieceProcessor()
        sp_model.Load(FLAGS.spm)

        self.train_data = XLNetDataUtils(sp_model, batch_size=8)
        self.dev_data = XLNetDataUtils(sp_model, batch_size=8)
        self.dev_batch = self.dev_data.iteration()

        data = {
            "batch_size": self.train_data.batch_size,
            "input_size": self.train_data.input_size,
            "vocab": self.train_data.vocab,
            "tag_map": self.train_data.tag_map,
        }

        f = open("model/data_map.pkl", "wb")
        cPickle.dump(data, f)
        f.close()
        self.batch_size = self.train_data.batch_size
        self.nums_tags = len(self.train_data.tag_map.keys())
        self.tag_map = self.train_data.tag_map
        self.train_length = len(self.train_data.data)

        # save vocab
        print("-"*50)
        print("train data:\t", self.train_length)
        print("nums of tags:\t", self.nums_tags)

        self.__creat_model()
        with tf.Session() as sess:
            with tf.device("/gpu:0"):
                ckpt = tf.train.get_checkpoint_state(self.checkpoint_dir)
                if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
                    print("restore model")
                    self.saver.restore(sess, ckpt.model_checkpoint_path)
                else:
                    sess.run(tf.global_variables_initializer())

                    tvars = tf.trainable_variables()
                    (assignment_map, initialized_variable_names) = model_utils.get_assignment_map_from_checkpoint(tvars, FLAGS.init_checkpoint)
                    tf.train.init_from_checkpoint(FLAGS.init_checkpoint, assignment_map)
                    for var in tvars:
                        init_string = ""
                        if var.name in initialized_variable_names:
                            init_string = ", *INIT_FROM_CKPT*"
                        print("  name = %s, shape = %s%s", var.name, var.shape,
                                        init_string)

                for i in range(self.max_epoch):
                    print("-"*50)
                    print("epoch {}".format(i))

                    steps = 0
                    for batch in self.train_data.get_batch():
                        steps += 1
                        global_steps, loss, logits, acc, length = self.xlnet_step(sess, batch)
                        if steps % 1 == 0:
                            print("[->] step {}/{}\tloss {:.2f}\tacc {:.2f}".format(
                                steps, len(self.train_data.batch_data), loss, acc))
                        if steps % 100 == 0:
                            self.xlnet_evaluate(sess, "ORG")
                            self.xlnet_evaluate(sess, "PER")
                            self.xlnet_evaluate(sess, "LOC")
                        if steps % 1000 == 0:
                            self.saver.save(sess, self.checkpoint_path)
                    self.saver.save(sess, self.checkpoint_path)

    def xlnet_evaluate(self, sess, tag):
        result = []
        trans = self.trans.eval()
        batch = self.dev_batch.__next__()

        ntokens, tag_ids, inputs_ids, segment_ids, input_mask = zip(*batch)
        feed = {
            self.input_ids: inputs_ids,
            self.segment_ids: segment_ids,
            self.targets: tag_ids,
            self.input_mask: input_mask,
            self.dropout: 1
        }
        pre_paths, acc, lengths = sess.run([self.pred_ids, self.accuracy, self.length], feed_dict=feed)

        tar_paths = tag_ids
        recall, precision, f1 = f1_score(tar_paths, pre_paths, tag, self.tag_map)
        best = self.best_dev_f1.eval()
        if f1 > best:
            print("\tnew best f1:")
            print("\trecall {:.2f}\t precision {:.2f}\t f1 {:.2f}".format(recall, precision, f1))
            tf.assign(self.best_dev_f1, f1).eval()

    def prepare_xlnet_pred_data(self, text):
        text.replace('…', '.')
        text.replace('℃', 'C')
        text_preprocessed = preprocess_text(text)
        input_ids = encode_ids(self.sp_model, text_preprocessed)
        input_mask = [1] * len(input_ids)
        segment_ids = [0] * len(input_ids)

        feed = {
            self.input_ids: [input_ids],
            self.segment_ids: [segment_ids],
            self.input_mask: [input_mask],
            self.dropout: 1
        }
        return feed


    def predict(self):
        f = open("model/data_map.pkl", "rb")
        maps = cPickle.load(f)
        f.close()
        self.batch_size = 1
        self.sp_model = spm.SentencePieceProcessor()
        self.sp_model.Load(FLAGS.spm)
        self.train_length = 10

        self.tag_map = maps.get("tag_map", {})
        self.nums_tags = len(self.tag_map.values())
        self.__creat_model()
        with tf.Session() as sess:
            ckpt = tf.train.get_checkpoint_state(self.checkpoint_dir)
            if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
                print("[->] restore model")
                self.saver.restore(sess, ckpt.model_checkpoint_path)
            else:
                print("[->] no model, initializing")
                sess.run(tf.global_variables_initializer())

            trans = self.trans.eval()
            while True:
                text = input(" > ")

                feed = self.prepare_xlnet_pred_data(text)

                paths, length = sess.run([self.pred_ids, self.length], feed_dict=feed)

                print(format_tags(paths[0], self.tag_map))
                org = get_tags(paths[0], "ORG", self.tag_map)
                org_entity = format_result(org, text, "ORG")
                per = get_tags(paths[0], "PER", self.tag_map)
                per_entity = format_result(per, text, "PER")
                loc = get_tags(paths[0], "LOC", self.tag_map)
                loc_entity = format_result(loc, text, "LOC")

                resp = org_entity["entities"] + per_entity["entities"] + loc_entity["entities"]
                print(json.dumps(resp, indent=2, ensure_ascii=False))

def main(_):
    model = Model()
    if FLAGS.entry == "train":
        model.train()
    elif FLAGS.entry == "predict":
        model.predict()

if __name__ == "__main__":
    tf.app.run()
