#!/usr/bin/python3 

import sys;
from os.path import exists, join;
import librosa;
import pandas as pd;
import tensorflow as tf;
from WaveNet import GCConv1D, calculate_receptive_field;
from create_dateset import mu_law_decode;

class Sampler(object):

  def __init__(self, ):

    if exists(join('model', 'wavenet.h5')):
      self.wavenet = tf.keras.models.load_model(join('model', 'wavenet.h5'), compile = False, custom_objects = {'GCConv1D': GCConv1D});
    else:
      raise 'train the wavenet first before sampling audios';
    self.receptive_field = calculate_receptive_field();
    self.category = pd.read_pickle('category.pkl');
  
  def sample(self, person_id, length = 10000):

    class_id = self.category[self.category['person_id'] == person_id]['class_id'].iloc[0];
    glob_cond = tf.reshape(class_id, (1,1)); # class_id.shape = (1,1)
    inputs = tf.random.uniform((1, self.receptive_field, 1), minval = 0, maxval = 256, dtype = tf.int32); # inputs.shape = (1, receptive_field, 1)
    samples = list();
    for i in range(length):
      outputs = self.wavenet([inputs, glob_cond]); # outputs.shape = (1, 1, 256)
      index = tf.math.argmax(outputs, axis = -1); # index.shape = (1, 1, 1)
      samples.append(index);
      inputs = tf.concat([inputs[:, 1:, :], index], axis = 1); # inputs.shape = (1, receptive_field, 1);
    samples = tf.squeeze(tf.concat(samples, axis = 1), axis = 0); # samples.shape = (length, 1)
    audio = tf.constant([mu_law_decode(sample) for sample in samples], dtype = tf.float32);
    return audio.numpy();

  def list_person(self, ):
      
    return self.category['person_id'];

if __name__ == "__main__":

  if len(sys.argv) != 2:
    print("Usage: " + sys.argv[0] + " <person_id>");
    exit(1);
  sampler = Sampler();
  audio = sampler.sample(sys.argv[1]);
  librosa.output.write_wave('sample.wav', audio, 16000);
