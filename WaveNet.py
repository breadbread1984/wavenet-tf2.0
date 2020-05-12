#!/usr/bin/python3

import tensorflow as tf;

class GCConv1D(tf.keras.layers.Layer):

  def __init__(self, filters = 32, kernel_size = 2, use_bias = True, padding = 'valid', strides = 1, dilations = 1, activation = None):
    
    super(GCConv1D, self).__init__();
    self.kernel_size = 2;
    self.filters = filters;
    self.use_bias = use_bias;
    self.padding = padding;
    self.strides = strides;
    self.dilations = dilations;
    self.activation = tf.math.tanh if activation == 'tanh' else (tf.math.sigmoid if activation == 'sigmoid' else (None if activation is None else -1));
    if self.activation == -1:
      raise "unknown activation";

  def build(self, input_shape):

    # input_shape[0] = (batch, length, 1)
    # input_shape[1] = (batch, 1, glob_embed_dim)
    weight_shape = tf.TensorShape((self.kernel_size, input_shape[0][-1], self.filters))
    self.weight = self.add_weight(name = 'weight', shape = weight_shape, trainable = True);
    if self.use_bias:
      bias_shape = tf.TensorShape((self.filters,));
      self.bias = self.add_weight(name = 'bias', shape = bias_shape, trainable = True);
    # global condition kernel and bias weights
    cond_weight_shape = tf.TensorShape((1, input_shape[1][-1], self.filters));
    self.cond_weight = self.add_weight(name = 'cond_weight', shape = cond_weight_shape, trainable = True);
    super(GCConv1D, self).build(input_shape);

  def call(self, inputs):

    conv_inputs = inputs[0];
    cond_inputs = inputs[1];
    outputs = tf.nn.conv1d(conv_inputs, self.weight, strides = self.strides, padding = self.padding, dilations = self.dilations);
    outputs += tf.nn.conv1d(cond_inputs, self.cond_weight, strides = 1, padding = 'same');
    if self.use_bias:
      outputs += self.bias;
    if self.activation is not None:
      outputs = self.activation(outputs);
    # outputs.shape = (batch, new_length, 32)
    return outputs;

def WaveNet(length, initial_kernel = 32, kernel_size = 2, residual_channels = 32, dilation_channels = 32, skip_channels = 512, quantization_channels = 256, use_glob_cond = False, glob_cls_num = None, glob_embed_dim = None):

  dilations = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512,
               1, 2, 4, 8, 16, 32, 64, 128, 256, 512,
               1, 2, 4, 8, 16, 32, 64, 128, 256, 512,
               1, 2, 4, 8, 16, 32, 64, 128, 256, 512,
               1, 2, 4, 8, 16, 32, 64, 128, 256, 512];

  # 1) embed_gc
  if use_glob_cond:
    glob_cond = tf.keras.Input((1,)); # glob_cond.shape = (batch, 1)
    glob_embed = tf.keras.layers.Embedding(glob_cls_num, glob_embed_dim)(glob_cond); # glob_embed.shape = (batch, glob_embed_dim)
    glob_embed = tf.keras.layers.Reshape((1, glob_embed_dim))(glob_embed); # glob_embed.shape = (batch, 1, glob_embed_dim)
  # 2_ create_network
  inputs = tf.keras.Input([length, 1]); # inputs.shape = (batch, length, 1)
  results = tf.keras.layers.Conv1D(filters = residual_channels, kernel_size = initial_kernel, padding = 'valid')(inputs); # results.shape = (batch, new_length, residual_channels)
  outputs = list();
  for layer_index, dilation in enumerate(dilations):
    if use_glob_cond:
      activation = GCConv1D(filters = dilation_channels, kernel_size = kernel_size, dilation = dilation, activation = 'tanh')([results, glob_embed]); # activation.shape = (batch, new_length, dilation_channels)
      gate = GCConv1D(filters = dilation_channels, kernel_size = kernel_size, dilation = dilation, activation = 'sigmoid')([results, glob_embed]); # gate.shape = (batch, new_length, dilation_channels)
    else:
      activation = tf.keras.layers.Conv1D(filters = dilation_channels, kernel_size = kernel_size, dilation_rate = dilation, activation = 'tanh', padding = 'same')(results);
      gate = tf.keras.layers.Conv1D(filters = dilation_channels, kernel_size = kernel_size, dilation_rate = dilation, activation = 'sigmoid', padding = 'same')(results);
    gated_activation = tf.keras.layers.Multiply()([activation, gate]);
    transformed = tf.keras.layers.Dense(units = residual_channels)(gated_activation); # transformed.shape = (batch, new_length, residual_channels)
    skip_constribution = tf.keras.layers.Dense(units = skip_channels)(gated_activation); # skip_construction.shape = (batch, new_length, skip_channels)
    outputs.append(skip_constribution);
    results = tf.keras.layers.Add()([results, transformed]); # results.shape = (batch, new_length, residual_channels)
  total = tf.keras.layers.Add()(outputs); # total.shape = (batch, new_length, skip_channels);
  transformed1 = tf.keras.layers.ReLU()(total);
  conv1 = tf.keras.layers.Dense(units = skip_channels)(transformed1); # conv1.shape = (batch, new_length, skip_channels)
  transformed2 = tf.keras.layers.ReLU()(conv1);
  raw_output = tf.keras.layers.Dense(units = quantization_channels)(transformed2); # raw_output.shape = (batch, new_length, quantization_channels)
  # 3) output
  outputs = tf.keras.layers.Softmax(axis = -1)(raw_output);
  return tf.keras.Model(inputs = inputs, outputs = outputs);

if __name__ == "__main__":

    assert tf.executing_eagerly();
    wavenet = WaveNet(100);
    wavenet.save('wavenet.h5');
    tf.keras.utils.plot_model(model = wavenet, to_file = 'wavenet.png', show_shapes = True, dpi = 64);
