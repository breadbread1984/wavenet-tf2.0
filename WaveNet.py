#!/usr/bin/python3

import tensorflow as tf;

def ResidualBlock(input_shape, global_condition_shape = None, local_condition_shape = None, filters = None, kernel_size = None, dilation_rate = None):

    tf.debugging.Assert(tf.equal(tf.shape(input_shape)[0], 2), [input_shape]);
    if global_condition_shape is not None:
        tf.debugging.Assert(tf.equal(tf.shape(global_condition_shape)[0], 1), [global_condition_shape]);
    if local_condition_shape is not None:
        tf.debugging.Assert(tf.equal(tf.shape(local_condition_shape)[0], 2), [local_condition_shape]);
        tf.debugging.Assert(tf.equal(input_shape[0] % local_condition_shape[0],0), [input_shape, local_condition_shape])
    assert filters is not None;
    assert kernel_size is not None;
    assert dilation_rate is not None;
    # inputs.shape = (batch, length, input channel)
    inputs = tf.keras.Input(input_shape);
    if global_condition_shape is not None:
        # global_condition.shape = (batch, global condition channel)
        global_condition = tf.keras.Input(global_condition_shape);
    if local_condition_shape is not None:
        # local_condition.shape = (batch, subsampled length, local condition channel)
        local_condition = tf.keras.Input(local_condition_shape);
    residual = inputs;
    # activation.shape = (batch, length, output channels = filters)
    activation = tf.keras.layers.Conv1D(filters = filters, kernel_size = kernel_size, dilation_rate = dilation_rate, padding = 'same', activation = tf.math.tanh)(inputs);
    # gate.shape = (batch, length, output channels = filters)
    gate = tf.keras.layers.Conv1D(filters = filters, kernel_size = kernel_size, dilation_rate = dilation_rate, padding = 'same', activation = tf.math.sigmoid)(inputs);
    if global_condition_shape is not None:
        # global_activaiton control.shape = (batch, length, filters)
        global_activation_control = tf.keras.layers.Dense(units = filters)(global_condition);
        global_activation_control = tf.keras.layers.Lambda(lambda x, inputs: tf.tile(tf.expand_dims(x, 1), (1, inputs.shape[1], 1)), arguments = {'inputs': inputs})(global_activation_control);
        # global_gate_control.shape = (batch, length, filters)
        global_gate_control = tf.keras.layers.Dense(units = filters)(global_condition);
        global_gate_control = tf.keras.layers.Lambda(lambda x, inputs: tf.tile(tf.expand_dims(x,1), (1, inputs.shape[1], 1)), arguments = {'inputs': inputs})(global_gate_control);
        # add to activation and gate
        activation = tf.keras.layers.Add()([activation, global_activation_control]);
        gate = tf.keras.layers.Add()([gate, global_gate_control]);
    if local_condition_shape is not None:
        # local_activation_control.shape = (batch, length, filters)
        local_activation_control = tf.keras.layers.Dense(units = filters)(local_condition);
        local_activation_control = tf.keras.layers.Lambda(lambda x, inputs: tf.tile(x, (1, inputs.shape[1] // x.shape[1], 1)), arguments = {'inputs': inputs})(local_activation_control);
        # local_gate_control.shape = (batch, length, filters)
        local_gate_control = tf.keras.layers.Dense(units = filters)(local_condition);
        local_gate_control = tf.keras.layers.Lambda(lambda x, inputs: tf.tile(x, (1, inputs.shape[1] // x.shape[1], 1)), arguments = {'inputs': inputs})(local_gate_control);
        # add to activation and gate
        activation = tf.keras.layers.Add()([activation, local_activation_control]);
        gate = tf.keras.layers.Add()([gate, local_gate_control]);
    gated_activation = tf.keras.layers.Multiply()([activation, gate]);
    filtered = tf.keras.layers.Conv1D(filters = 1, kernel_size = 1, padding = 'same', activation = tf.nn.relu)(gated_activation);
    # results.shape = (batch, length, filters);
    results = tf.keras.layers.Add()([filtered, residual]);
    return tf.keras.Model(inputs = inputs, outputs = (results, filtered));

def WaveNet(length, global_condition_shape = None, local_condition_shape = None):

    inputs = tf.keras.Input([length, 1]);
    # l1.shape = (batch, length, 10)
    l1a, l1b = ResidualBlock(inputs.shape[1:], global_condition_shape, local_condition_shape, 10, 5, 2)(inputs);
    # l2-l5.shape = (batch, length, 1)
    l2a, l2b = ResidualBlock(l1a.shape[1:], global_condition_shape, local_condition_shape, 1, 2, 4)(l1a);
    l3a, l3b = ResidualBlock(l2a.shape[1:], global_condition_shape, local_condition_shape, 1, 2, 8)(l2a);
    l4a, l4b = ResidualBlock(l3a.shape[1:], global_condition_shape, local_condition_shape, 1, 2, 16)(l3a);
    l5a, l5b = ResidualBlock(l4a.shape[1:], global_condition_shape, local_condition_shape, 1, 2, 32)(l4a);
    # l6.shape = (batch, length, 1)
    l6 = tf.keras.layers.Add()([l1b, l2b, l3b, l4b, l5b]);
    l7 = tf.keras.layers.ReLU()(l6);
    l8 = tf.keras.layers.Conv1D(filters = 1, kernel_size = 1, padding = 'same', activation = tf.nn.relu)(l7);
    l9 = tf.keras.layers.Conv1D(filters = 1, kernel_size = 1, padding = 'same')(l8)
    # l10.shape = (batch, length)
    l10 = tf.keras.layers.Flatten()(l9);
    l11 = tf.keras.layers.Dense(units = 1, activation = tf.math.tanh)(l10);
    return tf.keras.Model(inputs = inputs, outputs = l11);

if __name__ == "__main__":

    assert tf.executing_eagerly();
    wavenet = WaveNet(100);

