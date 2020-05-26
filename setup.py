from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(name='acora',
      version='0.1',
      description='Automatic Code-Review Assistant (ACoRA)',
      url='https://github.com/mochodek/acora.git',
      long_description=long_description,
      long_description_content_type="text/markdown",
      author='',
      author_email='',
      license='',
      package_dir={"": "src"},
      packages=find_packages("src"),
      install_requires=[
          'numpy>=1.18.1',
          'pandas>=1.0.2',
          'numpy>=1.18.1',
          'scikit-learn>=0.22.2',
          'scipy>=1.3.1',
          'requests>=2.23.0',
          'pygerrit2>=2.0.12',
          'urllib3>=1.25.8',
          'python-dateutil>=2.8.1',
          'keras-bert>=0.81.0',
          'keras-radam>=0.15.0',
          'matplotlib>=2.2.3',
          'seaborn>=0.10.0',
          'pygit2>=1.1.1',
      ],
      scripts=[
          'scripts/download_commented_lines_from_gerrit.py',
          'scripts/download_lines_from_gerrit.py',
          'scripts/train_bert_comments.py',
          'stripts/test_bert_comments.py',
          'scripts/classify_comments.py',
          'scripts/extract_code_lines_git.py',
          'scripts/extract_vocab_from_code.py',
      ],
      zip_safe=False,
      python_requires='>=3.6',
)
