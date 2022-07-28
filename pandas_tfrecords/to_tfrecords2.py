import pandas as pd
import tensorflow as tf
import numpy as np
from .to_tfrecords import get_tfrecords, get_schema


def write_tfrecords(df, path, spark_schema=None, compression_type='GZIP', columns=None, compression_level=9):
    compression_ext = '.gz' if compression_type else ''
    opts = {}
    if compression_type:
        opts['options'] = tf.io.TFRecordOptions(
            compression_type=compression_type,
            compression_level=compression_level,
        )
    # convert datetime columns to long timestam
    
    schema = get_schema(df, columns, spark_schema=spark_schema)
    tfrecords = get_tfrecords(df, schema)
    
    with tf.io.TFRecordWriter(path, **opts) as writer:
        for item in tfrecords:
            writer.write(item.SerializeToString())
    return True

