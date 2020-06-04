import numpy as np

import keras
from keras import backend as K
from keras_bert import (get_model, compile_model, get_base_dict, gen_batch_inputs, get_token_embedding,
                        get_custom_objects, set_custom_objects)
from keras_bert.layers import Extract, MaskedGlobalMaxPool1D

from acora.code import CodeTokenizer

POOL_NSP = 'POOL_NSP'
POOL_MAX = 'POOL_MAX'
POOL_AVE = 'POOL_AVE'



def extract_embeddings_generator(model,
                                 texts,
                                 poolings=None,
                                 vocabs=None,
                                 cased=True,
                                 batch_size=4,
                                 cut_embed=True,
                                 output_layer_num=1):
    """Extract embeddings from texts. It is a modified version of the function that comes
    with keras-bert, modified to use CodeTokenizer.

    Parameters:
    -----------
    model: Path to the checkpoint or built model without MLM and NSP.
    texts: Iterable texts.
    poolings: Pooling methods. Word embeddings will be returned if it is None.
                     Otherwise concatenated pooled embeddings will be returned.
    vocabs: A dict should be provided if model is built.
    cased: Whether it is cased for tokenizer.
    batch_size: Batch size.
    cut_embed: The computed embeddings will be cut based on their input lengths.
    output_layer_num: The number of layers whose outputs will be concatenated as a single output.
                             Only available when `model` is a path to checkpoint.
    return: A list of numpy arrays representing the embeddings.
    """

    seq_len = K.int_shape(model.outputs[0])[1]
    tokenizer = CodeTokenizer(vocabs, cased=cased)

    def _batch_generator():
        tokens, segments = [], []

        def _pad_inputs():
            if seq_len is None:
                max_len = max(map(len, tokens))
                for i in range(len(tokens)):
                    tokens[i].extend([0] * (max_len - len(tokens[i])))
                    segments[i].extend([0] * (max_len - len(segments[i])))
            return [np.array(tokens), np.array(segments)]

        for text in texts:
            if isinstance(text, (str, type(u''))):
                token, segment = tokenizer.encode(text, max_len=seq_len)
            else:
                token, segment = tokenizer.encode(text[0], text[1], max_len=seq_len)
            tokens.append(token)
            segments.append(segment)
            if len(tokens) == batch_size:
                yield _pad_inputs()
                tokens, segments = [], []
        if len(tokens) > 0:
            yield _pad_inputs()

    if poolings is not None:
        if isinstance(poolings, (str, type(u''))):
            poolings = [poolings]
        outputs = []
        for pooling in poolings:
            if pooling == POOL_NSP:
                outputs.append(Extract(index=0, name='Pool-NSP')(model.outputs[0]))
            elif pooling == POOL_MAX:
                outputs.append(MaskedGlobalMaxPool1D(name='Pool-Max')(model.outputs[0]))
            elif pooling == POOL_AVE:
                outputs.append(keras.layers.GlobalAvgPool1D(name='Pool-Ave')(model.outputs[0]))
            else:
                raise ValueError('Unknown pooling method: {}'.format(pooling))
        if len(outputs) == 1:
            outputs = outputs[0]
        else:
            outputs = keras.layers.Concatenate(name='Concatenate')(outputs)
        model = keras.models.Model(inputs=model.inputs, outputs=outputs)

    for batch_inputs in _batch_generator():
        outputs = model.predict(batch_inputs)
        for inputs, output in zip(batch_inputs[0], outputs):
            if poolings is None and cut_embed:
                length = 0
                for i in range(len(inputs) - 1, -1, -1):
                    if inputs[i] != 0:
                        length = i + 1
                        break
                output = output[:length]
            yield output


def extract_embeddings(model,
                       texts,
                       poolings=None,
                       vocabs=None,
                       cased=True,
                       batch_size=4,
                       cut_embed=True,
                       output_layer_num=1):
    """Extract embeddings from texts. This is a modified function from keras-bert
    that will use a generator using CodeTokenizer.

    Parameters:
    -----------
    model: Path to the checkpoint or built model without MLM and NSP.
    texts: Iterable texts.
    poolings: Pooling methods. Word embeddings will be returned if it is None.
                     Otherwise concatenated pooled embeddings will be returned.
    vocabs: A dict should be provided if model is built.
    cased: Whether it is cased for tokenizer.
    batch_size: Batch size.
    cut_embed: The computed embeddings will be cut based on their input lengths.
    output_layer_num: The number of layers whose outputs will be concatenated as a single output.
                             Only available when `model` is a path to checkpoint.
    return: A list of numpy arrays representing the embeddings.
    """
    return [embedding for embedding in extract_embeddings_generator(
        model, texts, poolings, vocabs, cased, batch_size, cut_embed, output_layer_num
    )]


class CodeLinesBERTEmbeddingsExtractor():
    """Allows extracting line embeddings using a BERT model trained on code"""

    def __init__(self, base_model, no_layers, token_dict):
        output_layer_num = [-i for i in range(1, no_layers + 1)]

        inputs = base_model.inputs[:2]
        outputs = []

        for layer_index in output_layer_num:
            if layer_index < 0:
                layer_index = no_layers + layer_index
            layer_index += 1
            try:
                layer = base_model.get_layer(name='Encoder-{}-FeedForward-Norm'.format(layer_index))
                outputs.append(layer.output)
            except ValueError:
                break
        if len(outputs) > 1:
            transformed = keras.layers.Concatenate(name='Encoder-Output')(list(reversed(outputs)))
        else:
            transformed = outputs[0]
            
        self.model =  keras.models.Model(inputs, transformed)

        self.token_dict = token_dict



    def extract_embeddings(self, lines):
        """Extracts embeddings from the given lines.
        
        Parameters:
        -----------
        lines : list of str, lines to extract embeddings for.

        return embeddings a list embeddings.
        """
        return extract_embeddings(self.model, 
                                lines, 
                                batch_size=1,
                                poolings=[POOL_NSP, POOL_AVE],
                                vocabs=self.token_dict)

