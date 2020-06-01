#!/usr/bin/env python

description="""Train BERT for review comments."""

import argparse
import logging
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}
logging.getLogger("tensorflow").setLevel(logging.INFO)
import json

import pandas as pd
import numpy as np

from scipy import stats

from sklearn.model_selection import train_test_split

from collections import Counter

import warnings  
with warnings.catch_warnings():  
    warnings.filterwarnings("ignore",category=FutureWarning)

    import keras

    from keras_bert import Tokenizer, load_trained_model_from_checkpoint
    from keras_bert.layers.extract import Extract

    from keras_radam import RAdam

    import tensorflow as tf
    
    if tf.__version__.startswith("1."):
            from tensorflow import ConfigProto, Session, set_random_seed
    else:
        from tensorflow.compat.v1 import ConfigProto, Session, set_random_seed
    
    from tensorflow.python.client import device_lib


from acora.vocab import BERTVocab
from acora.comments import default_subject_columns, \
    load_comments_files, CommentPurposeTransformer, CommentSubjectTransformer, \
    plot_purpose_confusion_matrix, plot_subjects_confusion_matrix, \
    report_comment_predictions_accuracy, default_purpose_lables

logger = logging.getLogger(f'acora.{__file__}')
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
logger.addHandler(ch)





if __name__ == '__main__':


    logger.info(f"\n#### Running script: {__file__}")

    parser = argparse.ArgumentParser(description=description)

    parser.add_argument("--training_data_paths",
                        help="a list of paths to training data files (either .csv or .xlsx).", 
                        type=str, nargs="+")

    parser.add_argument("--bert_pretrained_path",
                        help="a path to a pretrained BERT models (see https://github.com/google-research/bert)", 
                        type=str, required=True)

    parser.add_argument("--sep", help="a seprator used to separate columns in a csv file.",
                        default=";", type=str)

    parser.add_argument("--report_comments_lengths", help="to display information about the distribution of comment lengths.",
                        action='store_true')

    parser.add_argument("--message_column", help="a name of the column that stores the comment/message.",
                        type=str, default="message")

    parser.add_argument("--purpose_column", help="a name of the column that stores the decision class for the comment purpose.",
                        type=str, default="purpose")

    parser.add_argument("--purpose_labels", help="a list of possible purpose labels (categories).",
                        type=str, nargs="+", default=default_purpose_lables)

    parser.add_argument("--subject_columns", help="a list of column names that store the decision classes for the comment subjects.",
                        type=str, nargs="+", default=default_subject_columns)

    parser.add_argument("--seq_len", help="a maximum length of a comment (in the number of tokens).",
                        type=int, default=128)

    parser.add_argument("--not_use_gpu", help="to forbid using a GPU if available.",
                        action='store_true')

    parser.add_argument("--weight_instances", help="to balance the dataset by weighting the instances based on the classes frequency.",
                        action='store_true')

    parser.add_argument("--random_seed", help="a random seed used to control the process.",
                        type=int, default=102329)

    parser.add_argument("--learning_rate", help="a learning rate used to train the model.",
                        type=float, default=2e-4)
    
    parser.add_argument("--batch_size", help="a batch size used for training the model.",
                        type=int, default=32)

    parser.add_argument("--epochs", help="a number of epochs to train the model.",
                        type=int, default=50)

    parser.add_argument("--report_trainig_accuracy", help="to report the accuracy on the training dataset.",
                        action='store_true')

    parser.add_argument("--purpose_cm_train_path", help="a path to a file presenting a confusion matrix for the purpose output (for training).",
                        type=str, default="./comment_purpose_train.pdf")
    
    parser.add_argument("--subject_cm_train_path", help="a path to a file presenting a confusion matrix for the subject output (for training).",
                        type=str, default="./comment_subject_train.pdf")

    parser.add_argument("--model_save_path", help="a path to serialize the trained BERT model.",
                        type=str, default="./bert_comments.h5")

    parser.add_argument("--train_size", help="should be between 0.0 and 1.0 and represent the proportion of the dataset to include in the train split.",
                        type=float, default=1.0)

    parser.add_argument("--purpose_cm_train_val_path", help="a path to a file presenting a confusion matrix for the purpose output (for validation).",
                        type=str, default="./comment_purpose_val.pdf")
    
    parser.add_argument("--subject_cm_train_val_path", help="a path to a file presenting a confusion matrix for the subject output (for validation).",
                        type=str, default="./comment_subject_val.pdf")


    args = vars(parser.parse_args())
    logger.info(f"Run parameters: {str(args)}")

    #### Reading run arguments

    bert_pretrained_path = args['bert_pretrained_path']
    training_data_paths = args['training_data_paths']
    sep = args['sep']
    report_comments_lengths = args['report_comments_lengths']
    message_column = args['message_column']
    purpose_column = args['purpose_column']
    purpose_labels = args['purpose_labels']
    subject_columns = args['subject_columns']
    seq_len = args['seq_len']
    not_use_gpu = args['not_use_gpu']
    weight_instances =  args['weight_instances']
    random_seed = args['random_seed']
    lr = args['learning_rate']
    batch_size = args['batch_size']
    epochs = args['epochs']
    report_trainig_accuracy = args['report_trainig_accuracy']
    purpose_cm_path = args['purpose_cm_train_path']
    subject_cm_path = args['subject_cm_train_path']
    model_save_path = args['model_save_path']
    train_size = args['train_size']
    purpose_cm_val_path = args['purpose_cm_train_val_path']
    subject_cm_val_path = args['subject_cm_train_val_path']
    
    ######

    cols = [message_column, purpose_column] + subject_columns

    config_path = os.path.join(bert_pretrained_path, 'bert_config.json')
    checkpoint_path = os.path.join(bert_pretrained_path, 'bert_model.ckpt')
    vocab_path = os.path.join(bert_pretrained_path, 'vocab.txt')
    with open(config_path, "r", encoding='utf', errors='ignore') as json_file:
        bert_config = json.load(json_file)


    gpus = [x.name for x in device_lib.list_local_devices() if x.device_type == 'GPU']
    if not not_use_gpu and len(gpus) == 0:
        logger.error("You don't have a GPU available on your system, it can affect the performance...")

    config = ConfigProto( device_count = {'GPU': 0 if not_use_gpu else len(gpus)}, allow_soft_placement = True )
    sess = Session(config=config) 
    keras.backend.set_session(sess)

    logger.info(f"Loading vocabulary from {vocab_path}")
    vocab = BERTVocab.load_from_file(vocab_path)
    logger.info(f"Loaded {vocab.size:,} vocab entries.")

    logger.info("Initializing a BERT tokenizer...")
    tokenizer = Tokenizer(vocab.token_dict)
    logger.info(f"BERT tokenizer ready, example: 'ACoRA is a nice tool' -> {str(tokenizer.tokenize('ACoRA is a nice tool'))}")

    logger.info("Loading training data...")
    reviews_all_df = load_comments_files(training_data_paths, cols, sep)

    if train_size < 1.0:
        ids_train, ids_val, _, _ = train_test_split(reviews_all_df.index.tolist(), 
                                reviews_all_df.index.tolist(), train_size=train_size, random_state=random_seed)
        logger.info(f"Selecting {len(ids_train)} out of {reviews_all_df.shape[0]} comments for training and {len(ids_val)} out of {reviews_all_df.shape[0]}")
        reviews_val_df = reviews_all_df.loc[ids_val, :]
        reviews_all_df = reviews_all_df.loc[ids_train, :]


    if report_comments_lengths:
        comments_lengths = [len(tokenizer.encode(text)[0]) for text in reviews_all_df[message_column].tolist()]
        logger.info("Message lengths distribution: 90% is {:.0f}, 95% is {:.0f}, 98% is {:.0f}, 99% is {:.0f}, and 100% is {}".format(
                *np.percentile(comments_lengths, [90, 95, 98, 99, 100])))
        logger.info(f"Your selected sequence length corresponds to {stats.percentileofscore(comments_lengths, seq_len):.2f} percentile in the training dataset.")

    logger.info("Tokenizing messages in the training dataset...")
    tokenized_all_messages = [tokenizer.encode(text, max_len=seq_len)[0] for text in reviews_all_df[message_column].tolist()] 
    x_all = [np.array(tokenized_all_messages), np.zeros_like(tokenized_all_messages)]

    logger.info("Transforming data for the purpose output variable...")
    purpose_transformer = CommentPurposeTransformer(reviews_all_df, purpose_column, purpose_labels)
    y_all_purpose_labels = purpose_transformer.get_class_labels()
    y_all_purpose = purpose_transformer.encode()
    message_purpose_labels_cat_mappings = purpose_transformer.message_purpose_labels_cat_mappings

    logger.info(f"Distribution of the purpose types: {Counter(y_all_purpose_labels)}")

    logger.info("Transforming data for the subject output variables...")
    subject_transformer = CommentSubjectTransformer(reviews_all_df, subject_columns)
    y_all_subject = subject_transformer.encode_one_hot_all_subjects()

    np.random.seed(random_seed)
    set_random_seed(random_seed)
  
    logger.info("Loading the pre-trained BERT model...")
    layer_num = bert_config['num_hidden_layers']
    model = load_trained_model_from_checkpoint(
        config_path,
        checkpoint_path,
        training=False,
        use_adapter=True,
        seq_len=seq_len,
        trainable=['Encoder-{}-MultiHeadSelfAttention-Adapter'.format(i + 1) for i in range(layer_num)] +
        ['Encoder-{}-FeedForward-Adapter'.format(i + 1) for i in range(layer_num)] +
        ['Encoder-{}-MultiHeadSelfAttention-Norm'.format(i + 1) for i in range(layer_num)] +
        ['Encoder-{}-FeedForward-Norm'.format(i + 1) for i in range(layer_num)],
    )

    logger.info("Transforming BERT model to classify comments...")
    inputs = model.inputs[:2]
    dense = model.get_layer(f'Encoder-{layer_num}-FeedForward-Norm').output
    dense = Extract(index=0)(dense)

    losses = {
        "purpose_output": "categorical_crossentropy",
    }
    loss_weights = {"purpose_output": 1.0}
    outputs = [keras.layers.Dense(units=y_all_purpose.shape[1], activation='softmax', name="purpose_output")(dense)]
    for i, subject_class in enumerate(subject_columns):
        outputs.append(keras.layers.Dense(units=1, activation='sigmoid', name=f"{subject_class}_output")(dense))
        losses[f"{subject_class}_output"] = "binary_crossentropy"
        loss_weights[f"{subject_class}_output"] = 1.0

    model = keras.models.Model(inputs, outputs)

    model.compile(
        RAdam(learning_rate =lr),
        loss=losses, 
        loss_weights=loss_weights,
        metrics=['accuracy'],
    )

    y_all = {"purpose_output": y_all_purpose}
    for i, subject_class in enumerate(subject_columns):
        y_all[f"{subject_class}_output"] = subject_transformer.encode_binary_single_subject(subject_class).values

    logger.info("Fine-tuning BERT model...")
    if weight_instances:
        purpose_class_weights = purpose_transformer.class_weights()
        logger.info(f"Calculated purpose weights: {purpose_class_weights}")

        subject_class_weights = subject_transformer.class_weights()
        logger.info(f"Calculated subject weights: {subject_class_weights}")

        class_weights_all = {"purpose_output": purpose_class_weights}
        for i, subject_class in enumerate(subject_columns):
            class_weights_all[f"{subject_class}_output"] = subject_class_weights[subject_class]

        history = model.fit(
            x_all,
            y_all,
            epochs=epochs,
            batch_size=batch_size,
            verbose=2,
            shuffle=True,
            class_weight=class_weights_all,
        )
    else:
        history = model.fit(
            x_all,
            y_all,
            epochs=epochs,
            batch_size=batch_size,
            verbose=2,
            shuffle=True,
        )

    logger.info(f"Saving the BERT model to {model_save_path}")
    model.save(model_save_path)

    if report_trainig_accuracy:
        y_all_pred_purpose, *y_all_pred_subject = model.predict(x_all) 

        logger.info("Preparing confusion matrix for the comment purpose.")
        plot_purpose_confusion_matrix(y_all_pred_purpose, y_all_purpose, message_purpose_labels_cat_mappings, purpose_cm_path)

        
        logger.info("Preparing confusion matrix for the comment subjects.")
        plot_subjects_confusion_matrix(y_all_pred_subject, y_all_subject, subject_columns, subject_cm_path)

        logger.info("The accuracy of the predictions on the trainig dataset:")   
        report_comment_predictions_accuracy(y_all_pred_purpose, y_all_purpose, 
                    y_all_pred_subject, y_all_subject, subject_columns)

    if train_size < 1.0:
        logger.info("Tokenizing messages in the validation dataset...")
        tokenized_val_messages = [tokenizer.encode(text, max_len=seq_len)[0] for text in reviews_val_df[message_column].tolist()] 
        x_val = [np.array(tokenized_val_messages), np.zeros_like(tokenized_val_messages)]

        logger.info("Transforming data for the purpose output variable...")
        y_val_purpose_labels = purpose_transformer.transform_class_labels(reviews_val_df)
        y_val_purpose = purpose_transformer.transform_encode(y_val_purpose_labels)

        logger.info(f"Distribution of the purpose types in the validation dataset: {Counter(y_val_purpose_labels)}")

        logger.info("Transforming data for the subject output variables...")
        y_val_subject = subject_transformer.transform_encode_one_hot_all_subjects(reviews_val_df)

        y_val_all = {"purpose_output": y_val_purpose}
        for i, subject_class in enumerate(subject_columns):
            y_val_all[f"{subject_class}_output"] = subject_transformer.transform_encode_binary_single_subject(reviews_val_df, subject_class).values

        y_val_pred_purpose, *y_val_pred_subject = model.predict(x_val) 

        logger.info("Preparing confusion matrix for the comment purpose.")
        plot_purpose_confusion_matrix(y_val_pred_purpose, y_val_purpose, message_purpose_labels_cat_mappings, purpose_cm_val_path)

        
        logger.info("Preparing confusion matrix for the comment subjects.")
        plot_subjects_confusion_matrix(y_val_pred_subject, y_val_subject, subject_columns, subject_cm_val_path)

        logger.info("The accuracy of the predictions on the validation dataset:")   
        report_comment_predictions_accuracy(y_val_pred_purpose, y_val_purpose, 
                    y_val_pred_subject, y_val_subject, subject_columns)


    logger.info("BERT for comment training has been completed.")




