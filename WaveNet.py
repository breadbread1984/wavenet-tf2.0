#!/usr/bin/python3

import tensorflow as tf;

def GlobalConditionalConv1D(channel, glob_embed_dim, filters = 32, kernel_size = 2, dilation = 1, activation = None):

  assert activation in ['sigmoid', 'tanh', None];
  inputs = tf.keras.Input((None, channel)); # inputs1.shape = (batch, length, channel);
  condition = tf.keras.Input((1, glob_embed_dim)); # inputs2.shape = (batch, 1, glob_embed_dim);
  results = tf.keras.layers.Conv1D(filters = filters, kernel_size = kernel_size, padding = 'valid', dilation_rate = dilation)(inputs); # results1.shape = (batch, length, filters)
  cond_results = tf.keras.layers.Conv1D(filters = filters, kernel_size = 1, padding = 'same')(condition); # results2.shape = (batch, 1, filters)
  results = tf.keras.layers.Lambda(lambda x: x[0] + x[1])([results, cond_results]); # results.shape = (batch, length, filters)
  if activation is not None:
    results = tf.keras.layers.Activation(activation)(results);
  return tf.keras.Model(inputs = (inputs, condition), outputs = results);

def calculate_receptive_field(dilations = [2**i for i in range(10)] * 5, kernel_size = 2, initial_kernel = 32):

  # NOTE: passing every conv1d layer, out length = in length - dilation * (kernel size - 1)
  # in order to get output of length 1, calculate how long the input length is
  receptive_field = 1 + (initial_kernel - 1) * 1 + (kernel_size - 1) * sum(dilations);
  return receptive_field;

def WaveNet(initial_kernel = 32, kernel_size = 2, residual_channels = 32, dilation_channels = 32, skip_channels = 512, quantization_channels = 256, dilations = [2**i for i in range(10)] * 5, use_glob_cond = False, glob_cls_num = None, glob_embed_dim = None):

  # 1) embed_gc
  if use_glob_cond:
    glob_cond = tf.keras.Input((1,)); # glob_cond.shape = (batch, 1)
    glob_embed = tf.keras.layers.Embedding(glob_cls_num, glob_embed_dim)(glob_cond); # glob_embed.shape = (batch, 1, glob_embed_dim)
  # 2) create_network
  inputs = tf.keras.Input((None, 1)); # inputs.shape = (batch, length, 1)
  # calculate how long the output is, given the length of input
  output_width = tf.keras.layers.Lambda(lambda x, r: tf.shape(x)[1] - r + 1, arguments = {'r': calculate_receptive_field(dilations, kernel_size, initial_kernel)})(inputs);
  current_layer = tf.keras.layers.Conv1D(filters = residual_channels, kernel_size = initial_kernel, padding = 'valid')(inputs); # current_layer.shape = (batch, length, residual_channels)
  outputs = list();
  for layer_index, dilation in enumerate(dilations):
    # NOTE: out length = in length - dilation * (kernel size - 1)
    if use_glob_cond:
      activation = GlobalConditionalConv1D(channel = current_layer.shape[-1], glob_embed_dim = glob_embed_dim, filters = dilation_channels, kernel_size = kernel_size, dilation = dilation, activation = 'tanh')([current_layer, glob_embed]); # activation.shape = (batch, length - dilation * (kernel size - 1), dilation_channels)
      gate = GlobalConditionalConv1D(channel = current_layer.shape[-1], glob_embed_dim = glob_embed_dim, filters = dilation_channels, kernel_size = kernel_size, dilation = dilation, activation = 'sigmoid')([current_layer, glob_embed]); # gate.shape = (batch, length - dilation * (kernel size - 1), dilation_channels)
    else:
      activation = tf.keras.layers.Conv1D(filters = dilation_channels, kernel_size = kernel_size, dilation_rate = dilation, activation = 'tanh', padding = 'valid')(current_layer); # activation.shape = (batch, length - dilation * (kernel size - 1), dilation_channel)
      gate = tf.keras.layers.Conv1D(filters = dilation_channels, kernel_size = kernel_size, dilation_rate = dilation, activation = 'sigmoid', padding = 'valid')(current_layer); # activation.shape = (batch, length - dilation * (kernel size - 1), dilation_channel)
    gated_activation = tf.keras.layers.Multiply()([activation, gate]); # gated_activation.shape = (batch, length - dilation * (kernel size - 1), dilation_channel)
    # 1) feed forward output + shortened original input
    transformed = tf.keras.layers.Dense(units = residual_channels)(gated_activation); # transformed.shape = (batch, length - dilation * (kernel size - 1), residual_channels)
    shortened_current_layer = tf.keras.layers.Lambda(lambda x: x[0][:, -tf.shape(x[1])[1]:, :])([current_layer, transformed]); # shortened_current_layer.shape = (batch, length - dilation * (kernel size - 1), residual_channels)
    current_layer = tf.keras.layers.Add()([shortened_current_layer, transformed]); # current_layer.shape = (batch, length - dilation * (kernel size - 1), residual_channels)
    # 2) output branch
    out_skip = tf.keras.layers.Lambda(lambda x: x[0][:, -x[1]:, :])([gated_activation, output_width]);
    skip_constribution = tf.keras.layers.Dense(units = skip_channels)(out_skip); # skip_construction.shape = (batch, new_length, skip_channels)
    outputs.append(skip_constribution);
  total = tf.keras.layers.Add()(outputs); # total.shape = (batch, new_length, skip_channels);
  transformed1 = tf.keras.layers.ReLU()(total);
  conv1 = tf.keras.layers.Dense(units = skip_channels)(transformed1); # conv1.shape = (batch, new_length, skip_channels)
  transformed2 = tf.keras.layers.ReLU()(conv1);
  raw_output = tf.keras.layers.Dense(units = quantization_channels)(transformed2); # raw_output.shape = (batch, new_length, quantization_channels)
  # 3) output
  outputs = tf.keras.layers.Softmax(axis = -1)(raw_output);
  if use_glob_cond:
    return tf.keras.Model(inputs = (inputs, glob_cond), outputs = outputs);
  else:
    return tf.keras.Model(inputs = inputs, outputs = outputs);

if __name__ == "__main__":

  assert tf.executing_eagerly();
  wavenet = WaveNet(use_glob_cond = True, glob_cls_num = 100, glob_embed_dim = 5);
  import numpy as np;
  inputs = tf.constant(np.random.randint(low = 0, high = 256, size = (32,calculate_receptive_field([2**i for i in range(10)] * 5) + 100 - 1,1)), dtype = tf.float32);
  gc = tf.constant(np.random.randint(low = 0, high = 100, size = (32, 1)), tf.float32);
  outputs = wavenet([inputs, gc]);
  print(inputs.shape)
  print(outputs.shape)
  wavenet.save('wavenet.h5');
  tf.keras.utils.plot_model(model = wavenet, to_file = 'wavenet.png', show_shapes = True, dpi = 64);
  wavenet = tf.keras.models.load_model('wavenet.h5', compile = False, custom_objects = {'GCConv1D': GCConv1D});
  outputs = wavenet([inputs, gc]);
