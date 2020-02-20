from .from_tfrecords import from_tfrecords as tfrecords_to_pandas
from .to_tfrecords import to_tfrecords as pandas_to_tfrecords

__all__ = 'pandas_to_tfrecords', 'tfrecords_to_pandas', 'pd2tf', 'tf2pd'

pd2tf = pandas_to_tfrecords
tf2pd = tfrecords_to_pandas
