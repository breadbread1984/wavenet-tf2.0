#!/usr/bin/python3

import tensorflow as tf;

class GCConv1D(tf.keras.layers.Layer):

  def __init__(self, filters = 32, kernel_size = 2, use_bias = True, padding = 'valid', stride = 1, dilation = 1, activation = None):
    
    super(GCConv1D, self).__init__();
    self.kernel_size = 2;
    self.filters = filters;
    self.use_bias = use_bias;
    self.padding = padding.upper();
    self.stride = stride;
    self.dilation = dilation;
    activation = activation.lower();
    if activation is None: activation = 'identity';
    assert activation in ['identity', 'tanh', 'sigmoid'];
    self.activation = activation;

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
    outputs = tf.nn.conv1d(conv_inputs, self.weight, stride = self.stride, padding = self.padding, dilations = self.dilation);
    outputs += tf.nn.conv1d(cond_inputs, self.cond_weight, stride = 1, padding = 'SAME');
    if self.use_bias:
      outputs += self.bias;
    if self.activation == 'identity':
      outputs = tf.identity(outputs);
    elif self.activation == 'tanh':
      outputs = tf.math.tanh(outputs);
    else:
      outputs = tf.math.sigmoid(outputs);
    # outputs.shape = (batch, new_length, 32)
    return outputs;

  def get_config(self):

    config = super(GCConv1D, self).get_config();
    config['kernel_size'] = self.kernel_size;
    config['filters'] = self.filters;
    config['use_bias'] = self.use_bias;
    config['padding'] = self.padding;
    config['stride'] = self.stride;
    config['dilation'] = self.dilation;
    config['activation'] = self.activation;
    return config;

  @classmethod
  def from_config(cls, config):
    return cls(config['filters'], config['kernel_size'], config['use_bias'], config['padding'], config['stride'], config['dilation'], config['activation']);

def calculate_receptive_field(dilations = [2**i for i in range(10)] * 5, kernel_size = 2, initial_kernel = 32):

  receptive_field = (kernel_size - 1) * sum(dilations) + 1;
  receptive_field += initial_kernel - 1;
  return receptive_field;

def WaveNet(initial_kernel = 32, kernel_size = 2, residual_channels = 32, dilation_channels = 32, skip_channels = 512, quantization_channels = 256, dilations = [2**i for i in range(10)] * 5, use_glob_cond = False, glob_cls_num = None, glob_embed_dim = None):

  # 1) embed_gc
  if use_glob_cond:
    glob_cond = tf.keras.Input((1,)); # glob_cond.shape = (batch, 1)
    glob_embed = tf.keras.layers.Embedding(glob_cls_num, glob_embed_dim)(glob_cond); # glob_embed.shape = (batch, glob_embed_dim)
    glob_embed = tf.keras.layers.Reshape((1, glob_embed_dim))(glob_embed); # glob_embed.shape = (batch, 1, glob_embed_dim)
  # 2) create_network
  inputs = tf.keras.Input((None, 1)); # inputs.shape = (batch, length, 1)
  output_width = tf.keras.layers.Lambda(lambda x, r: tf.shape(x)[1] - r + 1, arguments = {'r': calculate_receptive_field(dilations, kernel_size, initial_kernel)})(inputs);
  results = tf.keras.layers.Conv1D(filters = residual_channels, kernel_size = initial_kernel, padding = 'valid')(inputs); # results.shape = (batch, new_length, residual_channels)
  current_layer = results;
  outputs = list();
  for layer_index, dilation in enumerate(dilations):
    if use_glob_cond:
      activation = GCConv1D(filters = dilation_channels, kernel_size = kernel_size, dilation = dilation, activation = 'tanh', padding = 'valid')([current_layer, glob_embed]); # activation.shape = (batch, new_length, dilation_channels)
      gate = GCConv1D(filters = dilation_channels, kernel_size = kernel_size, dilation = dilation, activation = 'sigmoid', padding = 'valid')([current_layer, glob_embed]); # gate.shape = (batch, new_length, dilation_channels)
    else:
      activation = tf.keras.layers.Conv1D(filters = dilation_channels, kernel_size = kernel_size, dilation_rate = dilation, activation = 'tanh', padding = 'valid')(current_layer);
      gate = tf.keras.layers.Conv1D(filters = dilation_channels, kernel_size = kernel_size, dilation_rate = dilation, activation = 'sigmoid', padding = 'valid')(current_layer);
    gated_activation = tf.keras.layers.Multiply()([activation, gate]);
    transformed = tf.keras.layers.Dense(units = residual_channels)(gated_activation); # transformed.shape = (batch, new_length, residual_channels)
    out_skip = tf.keras.layers.Lambda(lambda x: x[0][:, tf.shape(x[0])[1] - x[1]:, :])([gated_activation, output_width]);
    skip_constribution = tf.keras.layers.Dense(units = skip_channels)(out_skip); # skip_construction.shape = (batch, new_length, skip_channels)
    input_batch = tf.keras.layers.Lambda(lambda x: x[0][:, tf.shape(x[0])[1] - tf.shape(x[1])[1]:, :])([current_layer, transformed]);
    current_layer = tf.keras.layers.Add()([input_batch, transformed]); # results.shape = (batch, new_length, residual_channels)
    outputs.append(skip_constribution);
  total = tf.keras.layers.Add()(outputs); # total.shape = (batch, new_length, skip_channels);
  transformed1 = tf.keras.layers.ReLU()(total);
  conv1 = tf.keras.layers.Dense(units = skip_channels)(transformed1); # conv1.shape = (batch, new_length, skip_channels)
  transformed2 = tf.keras.layers.ReLU()(conv1);
  raw_output = tf.keras.layers.Dense(units = quantization_channels)(transformed2); # raw_output.shape = (batch, new_length, quantization_channels)
  raw_output = tf.keras.layers.Lambda(lambda x: tf.squeeze(x[:, tf.shape(x)[1] - 1:, :], axis = 1))(raw_output);
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
  inputs = tf.constant(np.random.randint(low = 0, high = 256, size = (32,calculate_receptive_field([2**i for i in range(10)] * 5) + 100,1)), dtype = tf.float32);
  gc = tf.constant(np.random.randint(low = 0, high = 100, size = (32, 1)), tf.float32);
  outputs = wavenet([inputs, gc]);
  print(inputs.shape)
  print(outputs.shape)
  wavenet.save('wavenet.h5');
  tf.keras.utils.plot_model(model = wavenet, to_file = 'wavenet.png', show_shapes = True, dpi = 64);
  wavenet = tf.keras.models.load_model('wavenet.h5', compile = False, custom_objects = {'GCConv1D': GCConv1D});
  outputs = wavenet([inputs, gc]);
