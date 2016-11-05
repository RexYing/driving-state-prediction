from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import os
import time
import tensorflow as tf
import numpy as np

import imgseqCNN

FLAGS = tf.app.flags.FLAGS


tf.app.flags.DEFINE_string('train_dir', './log',
                           """Directory where to write event logs """
                           """and checkpoint.""")
tf.app.flags.DEFINE_integer('max_steps', 10000,
                            """Number of batches to run.""")
tf.app.flags.DEFINE_boolean('log_device_placement', False,
                            """Whether to log device placement.""")

NUM_INPUT_CHANNELS = 1
LOG_DEVICE_PLACEMENT = False
GPU_MEM_FRACTION = 0.6
  

def train(): 

  global_step = tf.Variable(0, trainable=False)

  imgseqs, labels = imgseqCNN.inputs(NUM_INPUT_CHANNELS)

  inferred_sensor_values = imgseqCNN.inference(imgseqs)

  loss = imgseqCNN.loss(inferred_sensor_values, labels)

  train_op = imgseqCNN.train(loss, global_step)

  # Create a saver.
  saver = tf.train.Saver(tf.all_variables())

  # Build the summary operation based on the TF collection of Summaries.
  summary_op = tf.merge_all_summaries()

  # Build an initialization operation to run below.
  init = tf.initialize_all_variables()

  # Memory usage options
  config = tf.ConfigProto(log_device_placement=LOG_DEVICE_PLACEMENT)
  config.gpu_options.allow_growth = True
  config.gpu_options.per_process_gpu_memory_fraction = GPU_MEM_FRACTION

  # Start running operations on the Graph.
  sess = tf.Session(config=config)
  sess.run(init)

  # Start the queue runners.
  tf.train.start_queue_runners(sess=sess)

  summary_writer = tf.train.SummaryWriter(FLAGS.train_dir, sess.graph)

  for step in range(FLAGS.max_steps):
    start_time = time.time()
    _, loss_value = sess.run([train_op, loss])
    duration = time.time() - start_time

    assert not np.isnan(loss_value), 'Model diverged with loss = NaN'

    if step % 1 == 0:
      num_examples_per_step = FLAGS.batch_size
      examples_per_sec = num_examples_per_step / duration
      sec_per_batch = float(duration)

      format_str = ('%s: step %d, loss = %.2f (%.1f examples/sec; %.3f sec/batch)')
      print(format_str % (datetime.now(), step, loss_value, examples_per_sec, sec_per_batch))

    if step % 10 == 0:
      summary_str = sess.run(summary_op)
      summary_writer.add_summary(summary_str, step)

    # Save the model checkpoint periodically.
    if step % 100 == 0 or (step + 1) == FLAGS.max_steps:
      checkpoint_path = os.path.join(FLAGS.train_dir, 'model.ckpt')
      saver.save(sess, checkpoint_path, global_step=step)

def main(argv=None):
  if tf.gfile.Exists(FLAGS.train_dir):
    tf.gfile.DeleteRecursively(FLAGS.train_dir)
    tf.gfile.MakeDirs(FLAGS.train_dir)
    train()

if __name__ == '__main__':
  tf.app.run()

