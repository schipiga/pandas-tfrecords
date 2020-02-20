from setuptools import setup


setup(
    name='pandas_tfrecords',
    version='0.1',
    description='Convertor pandas to tfrecords & tfrecords to pandas',
    url='https://github.com/schipiga/pandas-tfrecords/',
    author='Sergei Chipiga <chipiga86@gmail.com>',
    author_email='chipiga86@gmail.com',
    packages=['pandas_tfrecords'],
    install_requires=[
        's3fs==0.4.0',
        'tensorflow==2.1.0',
        'pandas==1.0.1',
        'numpy==1.18.1',
    ],
)