from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import math
import time

import numpy as np
import tensorflow as tf

import imgseqCNN
import driving_input

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('eval_dir', '/tmp/driving_eval',
                               """Directory where to write event logs.""")
tf.app.flags.DEFINE_string('eval_data', 'test',
                               """Either 'test' or 'train_eval'.""")
tf.app.flags.DEFINE_string('checkpoint_dir', '/tmp/driving_train',
                               """Directory where to read model checkpoints.""")
tf.app.flags.DEFINE_string('test_root_dir', '/orions4-zfs/projects/rexy/kitti/rawtest',
                             """Root dir for test data.""")
tf.app.flags.DEFINE_integer('eval_interval_secs', 60 * 5,
                                """How often to run the eval.""")
tf.app.flags.DEFINE_integer('num_examples', 300,
                                """Number of examples to run.""")
tf.app.flags.DEFINE_boolean('run_once', False,
                             """Whether to run eval only once.""")

def to_string_float_list_with_prec(l, num_digits):
  format_str = "{0:0." + str(num_digits) + "f}"
  return [format_str.format(float(i)) for i in l]

def eval_once(saver, summary_writer, sensor_value_op, label_op, summary_op):
    """Run Eval once.
    Args:
      saver: Saver.
      summary_writer: Summary writer.
      top_k_op: Top K op.
      summary_op: Summary op.
    """
    sess = tf.Session()
    ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir)
    if ckpt and ckpt.model_checkpoint_path:
      # Restores from checkpoint
      saver.restore(sess, ckpt.model_checkpoint_path)
      # Assuming model_checkpoint_path looks
      # something like:
      #   /tmp/driving_train/model.ckpt-1000,
      # extract global_step from it(the digits after '-').
      global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
      print('Step No.: ' + global_step)
    else:
      print('No checkpoint file found')
      sess.close()
      return

    coord = tf.train.Coordinator()
    try:
      threads = []
      for qr in tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS):
        threads.extend(qr.create_threads(
            sess, coord=coord, daemon=True, start=True))

      num_iter = int(math.ceil(FLAGS.num_examples / FLAGS.batch_size))
      total_sample_count = num_iter * FLAGS.batch_size
      step = 0
      
      diff = 0

      # Other operations to output
      sensor_value_diff_op = tf.abs(tf.sub(sensor_value_op, label_op))
      sensor_value_percent_diff_op = tf.reduce_mean(tf.div(sensor_value_diff_op, label_op))

      while step < num_iter and not coord.should_stop():
        sensor_value, label, percent_err = sess.run([sensor_value_op, label_op, sensor_value_percent_diff_op])
        print('prediction:  ' + ' '.join(to_string_float_list_with_prec(sensor_value, 2)))
        print('actual:  ' + ' '.join(to_string_float_list_with_prec(label, 2)))
        print('percent_err:  ' + str(percent_err))
        diff += np.sum(percent_err)
        step += 1
      # Compute error.
      precision = diff / num_iter
      print('%s: percentage_err @ 1 = %.3f' % (datetime.now(), precision))

      summary = tf.Summary()
      summary.ParseFromString(sess.run(summary_op))
      summary.value.add(tag='percentage_err @ 1', simple_value=precision)
      summary_writer.add_summary(summary, global_step)
    except Exception as e: 
      coord.request_stop(e)

    coord.request_stop()
    coord.join(threads, stop_grace_period_secs=10)


def evaluate():
  """Eval for a number of steps."""
  with tf.Graph().as_default() as g:
    eval_data = FLAGS.eval_data == 'test'
    images, labels = driving_input.kitti_raw_input(FLAGS.test_root_dir, 1, FLAGS.batch_size)

    inferred = imgseqCNN.inference(images)

    # Restore the moving average version of the learned
    # variables for eval.
    variable_averages = tf.train.ExponentialMovingAverage(imgseqCNN.MOVING_AVERAGE_DECAY)
    variables_to_restore = variable_averages.variables_to_restore()
    saver = tf.train.Saver(variables_to_restore)

    # Build the summary operation based on
    # the TF collection of Summaries.
    summary_op = tf.merge_all_summaries()

    summary_writer = tf.train.SummaryWriter(FLAGS.eval_dir, g)

    while True:
      eval_once(saver, summary_writer, inferred, labels, summary_op)
      if FLAGS.run_once:
        break
      time.sleep(FLAGS.eval_interval_secs)

def main(argv=None):  # pylint: disable=unused-argument
  if tf.gfile.Exists(FLAGS.eval_dir):
    tf.gfile.DeleteRecursively(FLAGS.eval_dir)
    tf.gfile.MakeDirs(FLAGS.eval_dir)
  evaluate()

if __name__ == '__main__':
  tf.app.run()


