from setuptools import setup


setup(
    name='pandas-tfrecords',
    version='0.1.4',
    description='Converter pandas to tfrecords & tfrecords to pandas',
    long_description=open('README.rst').read(),
    url='https://github.com/schipiga/pandas-tfrecords/',
    author='Sergei Chipiga',
    author_email='chipiga86@gmail.com',
    license="MIT",
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
    ],
    packages=['pandas_tfrecords'],
    install_requires=[
        's3fs==0.4.0',
        'tensorflow==2.1.0',
        'pandas==1.0.1',
        'numpy==1.18.1',
    ],
)
