from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cv2
import os
import tensorflow as tf
import struct

import leveldb as ldb

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('leveldb_dir',
                           '/orions4-zfs/projects/rexy/TORCS/deepdriving/TORCS_Training_1F',
                           """Path to the Kitti root data directory.""")

def read_TORCS(db_path):
  """ Read TORCS dataset stored in leveldb format.

  Args:
    db_path: path to the leveldb folder

  Returns:
    record containing image frame stack and corresponding labels
  """
  db = ldb.LevelDB(db_path)
  it = db.RangeIter()
  for i in range(1):
    key, val = it.next()
    print(key)
    b=bytes(val)
    print(len(b))
    print(struct.unpack('i', val[:4]))

if __name__ == '__main__':
  #DB_PATH = "/orions4-zfs/projects/rexy/TORCS/deepdriving/TORCS_Training_1F"
  read_TORCS(FLAGS.leveldb_dir)
  #tf.app.run()
