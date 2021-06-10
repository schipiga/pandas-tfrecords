from setuptools import setup


setup(
    name='pandas-tfrecords',
    version='0.1.5',
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
        's3fs==2021.6.0',
        'tensorflow==2.5.0',
        'pandas==1.2.4',
        'numpy>=1.16.5',
    ],
)
