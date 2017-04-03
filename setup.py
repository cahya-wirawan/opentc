from setuptools import setup, find_packages
import opentc


def readme():
    with open('README.rst') as f:
        return f.read()

setup(
    name='opentc',
    version=opentc.__version__,
    description='A text classification engine',
    long_description=readme(),
    classifiers=[
        'Development Status :: 3 - Alpha',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.5',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Text Processing :: Filters'
    ],
    keywords='machine learning cnn svm bayesian',
    url='https://github.com/cahya-wirawan/opentc',
    author='Cahya Wirawan',
    author_email='Cahya.Wirawan@gmail.com',
    license='MIT',
    packages=find_packages('.'),
    package_dir={'': '.'},
    install_requires=[
        'numpy',
        'protobuf',
        'pyparsing',
        'PyYAML',
        'scikit-learn',
        'scipy',
        'tensorflow'
    ],
    scripts=['bin/opentc', 'bin/opentcd'],
    data_files=[('/etc/opentc', ['config/opentc.yml', 'config/logging.yml'])],
    include_package_data=True,
    zip_safe=False)