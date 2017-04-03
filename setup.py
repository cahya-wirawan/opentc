from setuptools import setup, find_packages
from setuptools.command.test import test as TestCommand


def readme():
    with open('README.rst') as f:
        return f.read()

setup(
    name='opentc',
    version='0.2.2',
    description='Open Text Classification engine',
    long_description='Really, the funniest around.',
    classifiers=[
        'Development Status :: 1 - Alpha',
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
    package_dir = {'': '.'},
    install_requires=[
        'tensorflow',
        'scikit-learn',
        'scipy',
        'PyYAML'
    ],
    scripts=['bin/opentc', 'bin/opentcd'],
    data_files=[('/etc/opentc', ['config/opentc.yml', 'config/logging.yml'])],
    include_package_data=True,
    zip_safe=False)