import os
import tempfile

import pandas as pd
import s3fs
import tensorflow as tf


__all__ = 'from_tfrecords',


s3_fs = s3fs.S3FileSystem()


def from_tfrecords(file_paths, schema=None, compression_type='auto', cast=True):
    file_paths = list(_normalize(file_paths))

    if compression_type == 'auto':
        compression_type = _get_compress_type(file_paths[0])

    dataset = tf.data.TFRecordDataset(
        file_paths, compression_type=compression_type)

    if schema:
        features, feature_lists = parse_schema(schema)
    else:
        features, feature_lists = detect_schema(dataset)

    if feature_lists:
        parser = read_sequence_example(features, feature_lists)
    else:
        parser = read_example(features)

    parsed = dataset.map(parser)
    return to_pandas(parsed, cast=cast)


def to_pandas(tfrecords, cast=True):
    df = None
    casting = _cast if cast else lambda o: o
    for row in tfrecords:

        if df is None:
            df = pd.DataFrame(columns=row.keys())

        row = {key: casting(val.numpy()) for key, val in row.items()}
        df = df.append(row, ignore_index=True)
    return df


def read_example(features):

    def parse(serialized):
        example = tf.io.parse_single_example(
            serialized,
            features=features)
        return example
    
    return parse


def read_sequence_example(features, feature_lists):

    def parse(serialized):
        context, sequence = tf.io.parse_single_sequence_example(
            serialized,
            context_features=features,
            sequence_features=feature_lists)
        context.update(sequence)
        return context
    
    return parse


def parse_schema(schema):
    features = {}
    feature_lists = {}

    for col, col_type in schema.items():
        if isinstance(col_type, list):
            feature_lists[col] = tf.io.FixedLenSequenceFeature(
                (), _get_feature_type(type_=col_type[0]))
        else:
            features[col] = tf.io.FixedLenFeature(
                (), _get_feature_type(type_=col_type))

    return features, feature_lists


def detect_schema(dataset):
    features = {}
    feature_lists = {}

    serialized = next(iter(dataset.map(lambda serialized: serialized)))
    seq_ex = tf.train.SequenceExample.FromString(serialized.numpy())

    if seq_ex.context.feature:
        for key, feature in seq_ex.context.feature.items():
            features[key] = tf.io.FixedLenFeature(
                (), _get_feature_type(feature=feature))

    if seq_ex.feature_lists.feature_list:
        for key, feature_list in seq_ex.feature_lists.feature_list.items():
            feature_lists[key] = tf.io.FixedLenSequenceFeature(
                (), _get_feature_type(feature=feature_list.feature[0]))

    return features, feature_lists


def _get_feature_type(feature=None, type_=None):

    if type_:
        return {
            int: tf.int64,
            float: tf.float32,
            str: tf.string,
            bytes: tf.string,
        }[type_]

    if feature:
        if feature.HasField('int64_list'):
            return tf.int64
        if feature.HasField('float_list'):
            return tf.float32
        if feature.HasField('bytes_list'):
            return tf.string


def _get_compress_type(file_path):
    magic_bytes = {
        b'x\x01': 'ZLIB',
        b'x^': 'ZLIB',
        b'x\x9c': 'ZLIB',
        b'x\xda': 'ZLIB',
        b'\x1f\x8b': 'GZIP'}

    two_bytes = open(file_path, 'rb').read(2)
    return magic_bytes.get(two_bytes)


def _normalize(file_paths):
    if isinstance(file_paths, str):
        file_paths = [file_paths]

    for file_path in file_paths:
        if file_path.startswith('s3'):
            if s3_fs.isdir(file_path):
                for s3_path in s3_fs.ls(file_path):
                    yield _download_s3(s3_path)
            else:
                yield _download_s3(file_path)
        else:
            file_path = os.path.abspath(os.path.expanduser(file_path))
            if os.path.isdir(file_path):
                for file_name in os.listdir(file_path):
                    yield os.path.join(file_path, file_name)
            else:
                yield file_path


def _download_s3(s3_path):
    tmp_path = tempfile.mkstemp()[1]
    s3_fs.get(s3_path, tmp_path)
    return tmp_path


def _cast(val):
    if not isinstance(val, bytes):
        return val
    try:
        return int(val)
    except ValueError:
        try:
            return float(val)
        except ValueError:
            try:
                return val.decode('utf8')
            except UnicodeDecodeError:
                return val
