#!/usr/bin/python3

from os import mkdir;
from os.path import join, exists;
import tensorflow as tf;
import pandas as pd;
from WaveNet import WaveNet, calculate_receptive_field;
from create_dataset import parse_function_generator;

batch_size = 8;

def train():

  category = pd.read_pickle('category.pkl');
  dilations = [2**i for i in range(10)] * 5;
  receptive_field = calculate_receptive_field(dilations, 2, 32);
  wavenet = WaveNet(dilations = dilations, use_glob_cond = True, glob_cls_num = len(category), glob_embed_dim = 5);
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
  for audios, person_id in trainset:
    inputs = audios[:,:-1,:]; # inputs.shape = (batch, receptive_field + audio_length - 1, 1)
    target = audios[:,receptive_field:,:]; # target.shape = (batch, audio_length, 1)
    with tf.GradientTape() as tape:
      outputs = wavenet([inputs, person_id]); # outputs.shape = (batch, audio_length, 1)
      loss = tf.keras.losses.SparseCategoricalCrossentropy()(target, outputs);
    avg_loss.update_state(loss);
    # write log
    if tf.equal(optimizer.iterations % 100, 0):
      with log.as_default():
        tf.summary.scalar('loss', avg_loss.result(), step = optimizer.iterations);
      print('Step #%d Loss: %.6f' % (optimizer.iterations, avg_loss.result()));
      if avg_loss.result() < 0.01: break;
      avg_loss.reset_states();
    grads = tape.gradient(loss, wavenet.trainable_variables);
    optimizer.apply_gradients(zip(grads, wavenet.trainable_variables));
  # save the network structure with weights
  if False == exists('model'): mkdir('model');
  wavenet.save(join('model', 'wavenet.h5'));

if __name__ == "__main__":

  assert tf.executing_eagerly();
  train();
