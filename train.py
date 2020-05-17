#!/usr/bin/python3

from os import mkdir;
from os.path import join, exists;
import tensorflow as tf;
import pandas as pd;
from WaveNet import WaveNet;
from create_dataset import parse_function_generator;

batch_size = 8;

def train():

  category = pd.read_pickle('category.pkl');
  wavenet = WaveNet(use_glob_cond = True, glob_cls_num = len(category), glob_embed_dim = 5);
  optimizer = tf.keras.optimizers.Adam(1e-3);
  # load dataset
  trainset = tf.data.TFRecordDataset(join('dataset', 'trainset.tfrecord')).repeat(-1).map(parse_function_generator()).batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE);
  # restore from existing checkpoint
  if False == exists('checkpoints'): mkdir('checkpoints');
  checkpoint = tf.train.Checkpoint(model = wavenet, optimizer = optimizer);
  checkpoint.restore(tf.train.latest_checkpoint('checkpoints'));
  # create log
  log = tf.summary.create_file_writer('checkpoints');
  # train model
  avg_loss = tf.keras.metrics.Mean(name = 'loss', dtype = tf.float32);
  for audios, labels in trainset:
    target = labels[:,,:]
    with tf.GradientTape() as tape:
      

if __name__ == "__main__":

  assert tf.executing_eagerly();
  train();
