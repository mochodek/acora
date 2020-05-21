# ACoRA - Automatic Code-Review Assistant

ACoRA is a prototype toolset employing Machine Learning algorithms to support code reviews in Continous Integration. 

## Installation
Download or clone the repository, open terminal in the root directory of the project and run the following command (if you are using conda or other Python virtualization tools, please remember to activate your virtual environment before):
```
pip install -e .
```
The installer should download and install or the required Python packages. However, please keep in mind that the required versions of the packages declared in the setup.py might be over-restrictive (you can try changing the versions and check whether the ACoRA tools execute properly).

## General usage scenario

ACoRA consists of a set of tools helping to build an automated pipeline supporting the code review process. They support data acquisition, models training, and classification. All of the scripts are located in the "scripts" folder. You can run each of them with the -h parameter to learn about the runtime parameter they accept. You can also check the "examples" folder to get examples of batch/bash runtime scripts.

The ACoRA process consists of the following activities:
TBD


## Download reviews from Gerrit

The following scripts supporting downloading review-data from Gerrit are available:
* scripts/download_lines_from_gerrit.py - allows downloading lines of code for a given Gerrit query and storing them in a csv file
* scripts/download_commented_lines_from_gerrit.py - allows downloading lines of code and comments made by reviewers.

## Classify review comments
In order to guide the focus of reviewers, we need to understand what the comments in our historical database of reviews are about. We classify each of the comments by the comment "purpose" (e.g., to request a change or to trigger a discussion) and by the comment "subject" (e.g., the comment is about coding style).

ACoRA provides the following scripts supporting this process:
* scripts/train_bert_comments.py - this script is used to train a BERT-based classifier that simultaneously predicts the purpose and subject of a comment. Please keep it in mind that you need to download a pre-trained BERT model from https://github.com/google-research/bert. The trained model is saved to a file. 

* scripts/test_bert_comments.py - this script allows testing the accuracy of predictions being made by the previously trained review-comments classification model on a given dataset. The script will generate plots and will report quality prediction metrics (e.g., accuracy, precision, recall).

* scripts/classify_comments.py - this script allows classifying new instances of comments using a previously trained review-comments classification model and stores the results to an xlsx or csv file.

