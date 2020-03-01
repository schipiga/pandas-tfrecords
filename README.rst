**************************
pandas-tfrecords converter
**************************

This project was born under impression from `spark-tensorflow-connector <https://github.com/tensorflow/ecosystem/tree/master/spark/spark-tensorflow-connector>`_ and implements similar functionality in order to save easy pandas dataframe to tfrecords and to restore tfrecords to pandas dataframe.

It can work as with local files as with AWS S3 files. Please keep in mind, that here tensorflow works with local copies of remote files, which are synced via ``s3fs`` with S3. I did this workaround because my tensorflow ``v2.1.0`` didn't work with S3 directly and raised authentication error ``Credentials have expired attempting to repull from EC2 Metadata Service``, but maybe it's fixed already.

===========
Quick start
===========

.. code::

    pip install pandas-tfrecords


.. code:: python

    import pandas as pd
    from pandas_tfrecords import pd2tf, tf2pd

    df = pd.DataFrame({'A': [1, 2, 3], 'B': ['a', 'b', 'c'], 'C': [[1, 2], [3, 4], [5, 6]]})

    # local
    pd2tf(df, './tfrecords')
    my_df = tf2pd('./tfrecords')

    # S3
    pd2tf(df, 's3://my-bucket/tfrecords')
    my_df = tf2pd('s3://my-bucket/tfrecords')

===============
Converted types
===============

-------------------
pandas -> tfrecords
-------------------

.. code::

    bytes, str -> tf.string
    int, np.integer -> tf.int64
    float, np.floating -> tf.float32
    list, np.ndarray of bytes, str, int, np.integer, float, np.floating -> sequence of tf.string, tf.int64, tf.float32

-------------------
tfrecords -> pandas
-------------------

.. code::

    tf.string -> bytes
    tf.int64 -> int
    tf.float32 -> float
    sequence of tf.string, tf.int64, tf.float32 -> list of bytes, int, float

**NB!** Please pay attention it works only with **one-dimentional** arrays. It means ``[1, 2, 3]`` will be converted to both sides, but ``[[1,2,3]]`` **won't** be converted to any side. It works that, because ``spark-tensorflow-connector`` works similar, and I didn't learn yet how to implement nested sequences. In order to work with **nested** sequences you should use ``reshape``.

===
API
===

.. code:: python

    pandas_tfrecords.pandas_to_tfrecords(df, folder, compression_type='GZIP', compression_level=9, columns=None, max_mb=50)

Arguments:

- ``df`` - pandas dataframe. Please keep in mind above info about nested sequences.
- ``folder`` - folder to save tfrecords, local or S3. Please be sure that it doesn't contain other files or folders, if you want to read from this folder then.
- ``compression_type='GZIP'`` - compression types: ``'GZIP'``, ``'ZLIB'``, ``None``. If ``None`` not compressed.
- ``compression_level=9`` - compression level 0...9.
- ``columns=None`` - list of columns to save, if ``None`` all columns will be saved.
- ``max_mb=50`` - maximum size of uncompressed data to save in single file. If dataframe total size is bigger than this limit, then it will be splitted to several files. If ``None`` it isn't limited and single file will be saved.

alias ``pandas_tfrecords.pd2tf``

.. code:: python

    pandas_tfrecords.tfrecords_to_pandas(file_paths, schema=None, compression_type='auto', cast=True)

Arguments:

- ``file_paths`` - One or sequence of file paths or folders, local or S3, to read tfrecords from.
- ``schema=None`` - If ``None`` schema will be detected automatically. But you can specify which columns you want to read only. It should be a dict, which keys are column names and values are column data types: ``str`` (or ``bytes``), ``int``, ``float``, and for sequences it should be wrapped to ``list``: ``[str]`` (or ``[bytes]``), ``[int]``, ``[float]``. For example:

.. code:: python

    df = pd.DataFrame({'A': [1, 2, 3], 'B': ['a', 'b', 'c'], 'C': [[1, 2], [3, 4], [5, 6]]})
    print(df)
       A  B       C
    0  1  a  [1, 2]
    1  2  b  [3, 4]
    2  3  c  [5, 6]

    pd2tf(df, './tfrecords')
    tf2pd('./tfrecords', schema={'A': int, 'C': [int]})
       A       C
    0  1  [1, 2]
    1  2  [3, 4]
    2  3  [5, 6]

- ``compression_type='auto'`` - compression type: ``'auto'``, ``'GZIP'``, ``'ZLIB'``, ``None``.
- ``cast=True`` - if ``True`` it casts ``bytes`` data after converting from ``tf.string``. It tries to cast it to ``int``, ``float`` and ``str`` sequentially. If it's not possible, otherwise keeps as is.

alias ``pandas_tfrecords.tf2pd``
