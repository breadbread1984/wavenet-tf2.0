#!/usr/bin/python3

import sys;
from os import listdir, mkdir;
from os.path import join, exists, splitext;
from re import search;
import librosa;
import pandas as pd;
import numpy as np;
import tensorflow as tf;

def mu_law_encode(audio, quantization_channels = 256):

  mu = tf.constant(quantization_channels - 1, dtype = tf.float32);
  safe_audio_abs = tf.math.minimum(tf.math.abs(audio), 1.0); # min(|x|, 1)
  magnitude = tf.math.log1p(mu * safe_audio_abs) / tf.math.log1p(mu); # log_e(1 + mu * min(|x|, 1)) / log_e(1 + mu)
  signal = tf.math.sign(audio) * magnitude; # sign(x) log_e(1 + mu * min(|x|, 1)) / log_e(1 + mu)
  return tf.cast((signal + 1) / 2 * mu + 0.5, dtype = tf.int32);

def mu_law_decode(output, quantization_channels = 256):

  mu = tf.constant(quantization_channels - 1, dtype = tf.float32);
  signal = 2 * (tf.cast(output, dtype = tf.float32) / mu) - 1;
  magnitude = (1 / mu) * ((1 + mu) ** tf.math.abs(signal) - 1);
  return tf.sign(signal) * magnitude;

def main(root_dir, sample_rate = 16000, silence_threshold = 0.3, dilations = [2**i for i in range(10)] * 5, quantization_channels = 256):

  from WaveNet import calculate_receptive_field;
  receptive_field = calculate_receptive_field(dilations, 2, 32);
  category = dict(); # person_id -> class id
  count = 0;
  if False == exists('dataset'): mkdir('dataset');
  writer = tf.io.TFRecordWriter(join('dataset', 'trainset.tfrecord'));
  for d in listdir(join(root_dir, 'wav48')):
    for f in listdir(join(root_dir, 'wav48', d)):
      result = search(r'p([0-9]+)_([0-9]+)\.wav', f);
      if result is None: continue;
      if False == exists(join(root_dir, 'txt', d, splitext(f)[0] + ".txt")):
        print("can't find corresponding label file!");
        continue;
      # 1) load audio file
      audio_path = join(root_dir, 'wav48', d, f);
      audio, _ = librosa.load(audio_path, sr = sample_rate, mono=True);
      audio = audio.reshape(-1, 1);
      # 2) load label file
      label_path = join(root_dir, 'txt', d, splitext(f)[0] + ".txt");
      label = open(label_path, 'r');
      if label is None:
        print("can't open label file!");
        continue;
      transcript = label.read().strip();
      label.close();
      person_id = int(result[1]);
      record_id = int(result[2]);
      if person_id not in category:
        category[person_id] = count;
        count += 1;
      # 3) trim silence under specific signal to noise ratio
      frame_length = 2048 if audio.size >= 2048 else audio.size;
      energe = librosa.feature.rms(audio, frame_length = frame_length);
      frames = np.nonzero(energe > silence_threshold);
      indices = librosa.core.frames_to_samples(frames)[1];
      audio = audio[indices[0]:indices[-1]] if indices.size else audio[0:0];
      audio = audio.reshape(-1, 1);
      # 4) pad at head
      audio = np.pad(audio, [[receptive_field, 0],[0, 0]], 'constant');
      # 5) quantization 
      quantized = mu_law_encode(audio, 256); # quantized.shape(length, 256)
      # 6) write to file
      trainsample = tf.train.Example(features = tf.train.Features(
        feature = {
          'audio': tf.train.Feature(int64_list = tf.train.Int64List(value = quantized.reshape(-1))),
          'category': tf.train.Feature(int64_list = tf.train.Int64List(value = [category[person_id]])),
          'transcript': tf.train.Feature(bytes_list = tf.train.BytesList(value = [transcript.encode('utf-8')]))
        }
      ));
      writer.write(trainsample.SerializeToString());
  writer.close();
  category = [(class_id, person_id) for person_id, class_id in category.items()];
  category = pd.DateFrame(category, columns = ['class_id', 'person_id']);
  category.to_pickle('category.pkl');

if __name__ == "__main__":

  assert True == tf.executing_eagerly();
  if len(sys.argv) != 2:
    print("Usage: " + sys.argv[0] + " <directory>");
    exit(1);
  main(sys.argv[1]);
