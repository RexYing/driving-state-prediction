from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cv2
import os
import tensorflow as tf

import leveldb as ldb

FLAGS = tf.app.flags.FLAGS


def read_TORCS(db_path):
  """ Read TORCS dataset stored in leveldb format.

  Args:
    db_path: path to the leveldb folder

  Returns:
    record containing image frame stack and corresponding labels
  """
  db = ldb.LevelDB(db_path)
  it = db.RangeIter()
  for i in range(3):
    print(it.next())

if __name__ == '__main__':
  DB_PATH = "orions4-zfs/projects/rexy/TORCS/deepdriving/TORCS_training_1F"
  read_TORCS(DB_PATH)
  #tf.app.run()
