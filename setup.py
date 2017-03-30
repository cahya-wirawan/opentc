from setuptools import setup


def readme():
    with open('README.rst') as f:
        return f.read()

setup(
    name='opentc',
    version='0.1',
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
    packages=['opentc'],
    install_requires=[
        'tensorflow',
        'scikit-learn'
    ],
    zip_safe=False)