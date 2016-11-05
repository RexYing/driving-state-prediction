from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tensorflow as tf

import driving_input

FLAGS = tf.app.flags.FLAGS

# Basic model parameters.
tf.app.flags.DEFINE_integer('batch_size', 30,
                            """Number of images to process in a batch.""")
tf.app.flags.DEFINE_string('root_dir', '/orions4-zfs/projects/rexy/kitti/raw',
        """Path to the Kitti root data directory.""")
tf.app.flags.DEFINE_boolean('use_fp16', False,
                            """Train the model using fp16.""")

NUM_CLASSES = 1


VAR_EPS = 1E-7

# Constants describing the training process.
MOVING_AVERAGE_DECAY = 0.9999     # The decay to use for the moving average.
NUM_EPOCHS_PER_DECAY = 50.0      # Epochs after which learning rate decays.
LEARNING_RATE_DECAY_FACTOR = 0.1  # Learning rate decay factor.
INITIAL_LEARNING_RATE = 0.01       # Initial learning rate.


def _variable_on_cpu(name, shape, initializer):
    """Helper to create a Variable stored on CPU memory.

    Args:
      name: name of the variable
      shape: list of ints
      initializer: initializer for Variable
    
    Returns:
      Variable Tensor
    """
    with tf.device('/cpu:0'):
      dtype = tf.float32
      var = tf.get_variable(name, shape, initializer=initializer, dtype=dtype)
    return var



def _variable_with_weight_decay(name, shape, stddev, wd):
  """Helper to create an initialized Variable with weight decay.
    Note that the Variable is initialized with a truncated normal distribution.
    A weight decay is added only if one is specified.
  Args:
    name: name of the variable
    shape: list of ints
    stddev: standard deviation of a truncated Gaussian
    wd: add L2Loss weight decay multiplied by this float. If None, weight
        decay is not added for this Variable.
  Returns:
    Variable Tensor
  """
  dtype = tf.float32
  var = _variable_on_cpu(
      name, shape, tf.truncated_normal_initializer(stddev=stddev, dtype=dtype))
  if wd is not None:
    weight_decay = tf.mul(tf.nn.l2_loss(var), wd, name='weight_loss')
    tf.add_to_collection('losses', weight_decay)
  return var


def _activation_summary(x):
  """Helper to create summaries for activations.
    Creates a summary that provides a histogram of activations.
    Creates a summary that measures the sparsity of activations.

  Args:
    x: Tensor

  Returns:
    nothing
  """
  tensor_name = x.op.name
  tf.histogram_summary(tensor_name + '/activations', x)
  tf.scalar_summary(tensor_name + '/sparsity',
  tf.nn.zero_fraction(x))



def inputs(num_input_channels):
  if not FLAGS.root_dir:
    raise ValueError('Please supply a data_dir')
  return driving_input.kitti_raw_input(FLAGS.root_dir, num_input_channels, FLAGS.batch_size)


def inference(images):
  """Build the simple sensor value estimation model.

  Args:
    images: Image frames returned from distorted_inputs() or inputs().

  Returns:
    Sensor values.
  """

  batch_size = images.get_shape()[0]
  num_frames = images.get_shape()[1]
  height = images.get_shape()[2];
  width = images.get_shape()[3];
  
  # We instantiate all variables using tf.get_variable() instead of
  # tf.Variable() in order to share variables across multiple GPU training runs.
  # If we only ran this model on a single GPU, we could simplify this function
  # by replacing all instances of tf.get_variable() with tf.Variable().
  #

  # conv1
  num_channels = [1, 48, 64, 64, 128]
  with tf.variable_scope('conv1') as scope:
    kernel = _variable_with_weight_decay(
        'weights', shape=[num_frames, 7, 7, num_channels[0], num_channels[1]], stddev=5e-2, wd=0.0)
    conv = tf.nn.conv3d(images, kernel, strides=[1, 1, 2, 4, 1], padding='SAME')
    biases = _variable_on_cpu('biases', [num_channels[1]], tf.constant_initializer(0.0))
    bias = tf.nn.bias_add(conv, biases)
    conv1 = tf.nn.relu(bias, name=scope.name)
    _activation_summary(conv1)

    # pool1
    pool1 = tf.nn.max_pool3d(
        conv1, ksize=[1, 3, 3, 3, 1], strides=[1, 1, 2, 2, 1], padding='SAME', name='pool1')

    # norm1
    #norm1 = tf.nn.lrn(pool1, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='norm1')
    batch_mean, batch_var = tf.nn.moments(pool1, [0])
    offset1  = _variable_on_cpu('offset1', batch_mean.get_shape(), tf.constant_initializer(0.0))
    scale1  = _variable_on_cpu('scale1', batch_mean.get_shape(), tf.constant_initializer(1.0))
    norm1 = tf.nn.batch_normalization(pool1, batch_mean, batch_var, offset1, scale1, VAR_EPS,
            name='batch_norm1')
    #norm1 = pool1
    print(norm1)

  # conv2
  with tf.variable_scope('conv2') as scope:
    kernel = _variable_with_weight_decay(
        'weights', shape=[num_frames, 5, 5, num_channels[1], num_channels[2]], stddev=5e-2, wd=0.0)
    conv = tf.nn.conv3d(norm1, kernel, [1, 3, 2, 2, 1], padding='VALID')
    biases = _variable_on_cpu('biases', [num_channels[2]], tf.constant_initializer(0.1))
    bias = tf.nn.bias_add(conv, biases)
    conv2 = tf.nn.relu(bias, name=scope.name)
    _activation_summary(conv2)
  
    # pool2
    pool2 = tf.nn.max_pool3d(conv2, ksize=[1, 1, 3, 3, 1], strides=[1, 1, 2, 2, 1], padding='SAME',
            name='pool2')
    print(pool2)

    # norm2
    #norm2 = tf.nn.lrn(conv2, 4, bias=1.0, alpha=0.001/9.0, beta=0.75, name='norm2')
    norm2 = pool2
  
  # conv3
  with tf.variable_scope('conv3') as scope:
    kernel = _variable_with_weight_decay(
        'weights', shape=[1, 5, 5, num_channels[2], num_channels[3]], stddev=5e-2, wd=0.0)
    conv = tf.nn.conv3d(norm2, kernel, [1, 1, 2, 2, 1], padding='SAME')
    biases = _variable_on_cpu('biases', [num_channels[3]], tf.constant_initializer(0.1))
    bias = tf.nn.bias_add(conv, biases)
    conv3 = tf.nn.relu(bias, name=scope.name)
    _activation_summary(conv3)

    pool3 = tf.nn.max_pool3d(conv3, ksize=[1, 1, 3, 3, 1], strides=[1, 1, 2, 2, 1], padding='SAME',
            name='pool3')
    print(pool3)

    # norm3
    #norm3 = tf.nn.lrn(conv3, 4, bias=1.0, alpha=0.001/9.0, beta=0.75, name='norm3')
    norm3 = pool3
   
  # conv4
  with tf.variable_scope('conv4') as scope:
    kernel = _variable_with_weight_decay(
        'weights', shape=[1, 5, 5, num_channels[3], num_channels[4]], stddev=5e-2, wd=0.0)
    conv = tf.nn.conv3d(norm3, kernel, [1, 1, 2, 2, 1], padding='SAME')
    biases = _variable_on_cpu('biases', [num_channels[4]], tf.constant_initializer(0.1))
    bias = tf.nn.bias_add(conv, biases)
    conv4 = tf.nn.relu(bias, name=scope.name)
    _activation_summary(conv4)

    # pool4
    pool4 = tf.nn.max_pool3d(conv4, ksize=[1, 1, 3, 3, 1], strides=[1, 1, 2, 2, 1], padding='SAME',
            name='pool4')
    print(pool4)

    # norm4
    #norm4 = tf.nn.lrn(conv4, 4, bias=1.0, alpha=0.001/9.0, beta=0.75, name='norm4')
    norm4 = pool4

  with tf.variable_scope('linear') as scope:
    reshape = tf.reshape(norm4, [FLAGS.batch_size, -1])
    dim = reshape.get_shape()[1].value
    weights = _variable_with_weight_decay('weights', [dim, NUM_CLASSES], stddev=1/192.0, wd=0.0)
    biases = _variable_on_cpu('biases', [NUM_CLASSES], tf.constant_initializer(0.0))
    linear = tf.add(tf.matmul(reshape, weights), biases, name=scope.name)
    print(linear)
    _activation_summary(linear)

    return linear                                                        


def loss(inferred_values, labels):
  """Add L2Loss to all the trainable variables.
    Add summary for "Loss" and "Loss/avg".

  Args:
    inferred_values: Sensor values output from inference().
    labels: Labels. 1-D tensor of shape [batch_size]
  Returns:
    Loss tensor of type float.
  """
  # Calculate the average cross entropy loss across the batch.
  l2_loss = tf.nn.l2_loss(tf.sub(inferred_values, labels), name='l2_per_frame')
  print('l2 loss ' + str(l2_loss))
  l2_loss_mean = tf.reduce_mean(l2_loss, name='l2_mean_loss')
  print('l2 loss mean' + str(l2_loss_mean))
  tf.add_to_collection('losses', l2_loss_mean)

  # The total loss is defined as the l2 diff plus all of the weight decay terms (L2 loss).
  return tf.add_n(tf.get_collection('losses'), name='total_loss')


def _add_loss_summaries(total_loss):
  """Add summaries for losses in CIFAR-10 model.
    Generates moving average for all losses and associated summaries for
    visualizing the performance of the network.
  Args:
    total_loss: Total loss from loss().
  Returns:
    loss_averages_op: op for generating moving averages of losses.
  """
  # Compute the moving average of all individual losses and the total loss.
  loss_averages = tf.train.ExponentialMovingAverage(0.9, name='avg')
  losses = tf.get_collection('losses')
  loss_averages_op = loss_averages.apply(losses + [total_loss])

  # Attach a scalar summary to all individual losses and the total
  # loss; do the
  # same for the averaged version of the losses.
  for l in losses + [total_loss]:
    # Name each loss as '(raw)' and name the moving
    # average version of the loss
    # as the original loss name.
    tf.scalar_summary(l.op.name +' (raw)', l)
    tf.scalar_summary(l.op.name, loss_averages.average(l))

  return loss_averages_op


def train(total_loss, global_step):
  """Train CIFAR-10 model.
    Create an optimizer and apply to all trainable variables. Add moving
    average for all trainable variables.

  Args:
    total_loss: Total loss from loss().
    global_step: Integer Variable counting the number of training steps processed.

  Returns:
    train_op: op for training.
  """
  # Variables that affect learning rate.
  num_batches_per_epoch = driving_input.NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN / FLAGS.batch_size
  decay_steps = int(num_batches_per_epoch * NUM_EPOCHS_PER_DECAY)

  # Decay the learning rate exponentially based on the number of steps.
  lr = tf.train.exponential_decay(
      INITIAL_LEARNING_RATE, global_step, decay_steps, LEARNING_RATE_DECAY_FACTOR, staircase=True)
  tf.scalar_summary('learning_rate', lr)

  # Generate moving averages of all losses and associated summaries.
  loss_averages_op = _add_loss_summaries(total_loss)

  # Compute gradients.
  with tf.control_dependencies([loss_averages_op]):
    #opt = tf.train.GradientDescentOptimizer(lr)
    opt = tf.train.AdamOptimizer(epsilon=1e-3)
    grads = opt.compute_gradients(total_loss)

  # Apply gradients.
  apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)

  # Add histograms for trainable variables.
  for var in tf.trainable_variables():
    tf.histogram_summary(var.op.name, var)

  # Add histograms for gradients.
  for grad, var in grads:
    if grad is not None:
      tf.histogram_summary(var.op.name + '/gradients', grad)

  # Track the moving averages of all trainable variables.
  variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)
  variables_averages_op = variable_averages.apply(tf.trainable_variables())

  with tf.control_dependencies([apply_gradient_op, variables_averages_op]):
    train_op = tf.no_op(name='train')

  return train_op

