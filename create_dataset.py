#!/usr/bin/python3

import sys;
from os import listdir;
from os.path import join, exists, splitext;
from re import search;
import librosa;
import numpy as np;
import tensorflow as tf;

def main(root_dir, sample_rate = 16000, silence_threshold = 0.3):

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
      label_path = join(root_dir, 'txt', d, splitext(f)[0] + ".txt"));
      label = open(label, 'r');
      if label is None:
        print("can't open label file!");
        continue;
      transcript = label.read().strip();
      label.close();
      person_id = int(result[1]);
      record_id = int(result[2]);
      # 3) trim silence under specific signal to noise ratio
      frame_length = 2048 if audio.size >= 2048 else audio.size;
      energe = librosa.feature.rmse(audio, frame_length = frame_length);
      frames = np.nonzero(energe > threshold);
      indices = librosa.core.frames_to_samples(frames)[1];
      audio = audio[indices[0]:indices[-1]] if indices.size else audio[0:0];
      # TODO

if __name__ == "__main__":

  assert True == tf.executing_eagerly();
  if len(sys.argv) != 2:
    print("Usage: " + sys.argv[0] + " <directory>");
    exit(1);
  main(sys.argv[1]);
