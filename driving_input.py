from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import tensorflow as tf
import numpy as np
import cv2

FLAGS = tf.app.flags.FLAGS

KITTI_RAW_IMAGE_HEIGHT = 375
KITTI_RAW_IMAGE_WIDTH = 1242
NUM_INPUT_CHANNELS = 3
NUM_SENSOR_VALS = 30

INPUT_IMAGE_HEIGHT = KITTI_RAW_IMAGE_HEIGHT // 2
INPUT_IMAGE_WIDTH = KITTI_RAW_IMAGE_WIDTH // 2

NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 100
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = 100



# Order: frame. height, weight, channel
FRAME_DIM = 0


def read_kitti_raw_data(sensor_filename_queue, image_filename_queues):
  class KittiRawRecord(object):
    pass
  result = KittiRawRecord()
  
  num_consec_frames = len(image_filename_queues)
  num_prec_frames = num_consec_frames // 2
  num_succ_frames = num_consec_frames - num_prec_frames - 1

  reader = tf.TextLineReader()
  result.key, value = reader.read(sensor_filename_queue)

  # Convert from a string to a vector of uint8 that is record_bytes long.
  #record_bytes = tf.decode_raw(value, tf.uint8)
  record_defaults = [[np.float32(0.0)] for i in range(NUM_SENSOR_VALS)]
  sensor_list = tf.decode_csv(value, record_defaults, field_delim=" ")

  # 8: forward velocity
  result.label=sensor_list[8]

  # The remaining bytes after the label represent the image, which we reshape
  # from [depth * height * width] to [depth, height, width].
  #depth_major = tf.reshape(tf.slice(record_bytes, [label_bytes], [image_bytes]),
  #                         [result.depth, result.height, result.width])
  # Convert from [depth, height, width] to [height, width, depth].
  #result.uint8image = tf.transpose(depth_major, [1, 2, 0])
  img_reader = tf.WholeFileReader()
  
  result.imkey, value = img_reader.read(image_filename_queues[0])
  img = tf.image.decode_png(value, NUM_INPUT_CHANNELS, dtype=tf.uint8)
  result.imgseq = tf.image.rgb_to_grayscale(img)
  # whiten for invariance to contrast
  result.imgseq = tf.image.per_image_whitening(result.imgseq)
  result.imgseq = tf.expand_dims(result.imgseq, FRAME_DIM)

  for i in range(1, num_consec_frames):
    key, value = img_reader.read(image_filename_queues[i])
    img = tf.image.decode_png(value, NUM_INPUT_CHANNELS, dtype=tf.uint8)
    img = tf.image.rgb_to_grayscale(img)
    # whiten for invariance to contrast
    img = tf.image.per_image_whitening(img)
    img = tf.expand_dims(img, FRAME_DIM)
    result.imgseq = tf.concat(FRAME_DIM, [result.imgseq, img])


  return result




def _generate_image_and_label_batch(image, label, min_queue_examples, batch_size, shuffle):
  """Construct a queued batch of images and labels.

  Args:
    image: 4-D Tensor of [frame, height, width, NUM_CHANNELS] of type.float32.
    label: 1-D Tensor of type.int32
    min_queue_examples: int32, minimum number of samples to retain
      in the queue that provides of batches of examples.
    batch_size: Number of images per batch.
    shuffle: boolean indicating whether to use a shuffling queue.

  Returns:
    images: Images. 5D tensor of [batch_size, frame, height, width, NUM_CHANNELS] size.
    labels: Labels. 1D tensor of [batch_size] size.
  """
  # Create a queue that shuffles the examples, and then
  # read 'batch_size' images + labels from the example queue.

  # Set it to 1 because there is racing... WTF?!
  num_preprocess_threads = 1
  if shuffle:
    images, label_batch = tf.train.shuffle_batch(
        [image, label],
        batch_size=batch_size,
        num_threads=num_preprocess_threads,
        capacity=min_queue_examples + 3 * batch_size,
        min_after_dequeue=min_queue_examples)
  else:
    images, label_batch = tf.train.batch(
        [image, label],
        batch_size=batch_size,
        num_threads=num_preprocess_threads,
        capacity=min_queue_examples + 3 * batch_size,
        allow_smaller_final_batch=True)

  # Display the training images in the visualizer.
  tf.image_summary('images0', tf.unpack(images, axis=1)[0])
  tf.image_summary('images1', tf.unpack(images, axis=1)[1])
  tf.image_summary('images2', tf.unpack(images, axis=1)[2])

  return images, tf.reshape(label_batch, [batch_size])


def get_sensor_img_dirs(root_dir):
  """ Get directories that contain sensor value texts as a list for all Kitti raw data.
  """
  sensor_dirs = []
  image_dirs = []
  EXCLUDE_DIRS = ['devkit']
  OXTS_DIR = 'oxts/data'
  IMG_DIR = 'image_03/data'

  dir_list = next(os.walk(FLAGS.root_dir))[1]
  for date_str in dir_list:
    if date_str in EXCLUDE_DIRS:
      continue
    path1 = os.path.join(root_dir, date_str)
    print(path1)
    dir_list1 = next(os.walk(path1))[1]
    for next_dir in dir_list1:
      path2 = os.path.join(path1, next_dir)
      path_sensor = os.path.join(path2, OXTS_DIR)
      sensor_dirs.append(path_sensor)

      path_img = os.path.join(path2, IMG_DIR)
      image_dirs.append(path_img)

  return (sensor_dirs, image_dirs)


def get_sensor_img_filenames(sensor_dirs, img_dirs):
  """ Get a list of sensor filenames, and a lists of image filenames
  The number of lists of image files is equal to number of consecutive frames.

  """
  num_consec_frames = 3
  num_prec_frames = num_consec_frames // 2
  num_succ_frames = num_consec_frames - num_prec_frames - 1

  sensor_filenames = []
  image_filenames_list = [[] for i in range(num_consec_frames)]

  for sensor_dir in sensor_dirs:
    num_files = len([name for name in os.listdir(sensor_dir) if
        os.path.isfile(os.path.join(sensor_dir, name))])
    print('num files ' + str(num_files))
    sensor_filenames = sensor_filenames + [
        os.path.join(sensor_dir, '%010d.txt' % i) 
            for i in range(num_prec_frames, num_files - num_succ_frames)]

  for image_dir in img_dirs:
    num_files = len([name for name in os.listdir(image_dir) if
        os.path.isfile(os.path.join(image_dir, name))])
    for j in range(0, num_consec_frames):
      image_filenames_list[j] = image_filenames_list[j] + [
          os.path.join(image_dir, '%010d.png' % i) 
            for i in range(j, num_files - num_consec_frames + j + 1)]

  for f in image_filenames_list[num_consec_frames - 1]:
    if not tf.gfile.Exists(f):
      raise ValueError('Failed to find file: ' + f)

  return (sensor_filenames, image_filenames_list)



def kitti_raw_input(root_dir, num_channels, batch_size):

  
  sensor_dirs, img_dirs = get_sensor_img_dirs(root_dir)
  sensor_filenames, image_filenames_list = get_sensor_img_filenames(sensor_dirs, img_dirs)
  print(len(sensor_filenames))
  print(len(image_filenames_list[0]))

  num_examples_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN
  
  num_consec_frames = len(image_filenames_list)
  sensor_filename_queue = tf.train.string_input_producer(sensor_filenames, num_epochs=None, shuffle=False)
  image_filename_queues = [tf.train.string_input_producer(image_filenames_list[i], num_epochs=None, shuffle=False) 
      for i in range(num_consec_frames)]
  
  drive_record = read_kitti_raw_data(sensor_filename_queue, image_filename_queues)
   
  image = tf.cast(drive_record.imgseq, tf.float32)
  image = tf.image.resize_images(
          image, 
          [INPUT_IMAGE_HEIGHT, INPUT_IMAGE_WIDTH],
          tf.image.ResizeMethod.AREA)

  image.set_shape([num_consec_frames, INPUT_IMAGE_HEIGHT, INPUT_IMAGE_WIDTH, num_channels])
  
  min_fraction_of_examples_in_queue = 0.4
  min_queue_examples = int(num_examples_per_epoch *
                           min_fraction_of_examples_in_queue)
  
  images, labels = _generate_image_and_label_batch(image, drive_record.label, 
      min_queue_examples, batch_size, shuffle=True)

  
#  config = tf.ConfigProto(log_device_placement=LOG_DEVICE_PLACEMENT)
#  config.gpu_options.allow_growth = True
#  config.gpu_options.per_process_gpu_memory_fraction = 0.4
#  
#  with tf.Session(config=config) as sess:
#    print("start")
#    init = tf.initialize_all_variables()
#    sess.run(init)
#    # Start populating the filename queues.
#    coord = tf.train.Coordinator()
#    threads = tf.train.start_queue_runners(coord=coord)
#    imgs, labs = sess.run([images, labels])
#    #img, lab, k, ik = sess.run([drive_record.imgseq, drive_record.label, drive_record.key,
#    #    drive_record.imkey])
#    
#  
#    coord.request_stop()
#    coord.join(threads)
#    print('finish')
#    im = imgs[0]
#    cv2.imwrite('tmp1.png', np.squeeze(im[0, ...]))
#    cv2.imwrite('tmp2.png', np.squeeze(im[0, ...]))
#    cv2.imwrite('tmp3.png', np.squeeze(im[0, ...]))

  return images, labels


