import re

import unicodedata
from keras_bert import Tokenizer
from keras_bert.bert import TOKEN_CLS, TOKEN_SEP, TOKEN_UNK


default_code_stop_delim = r"([\s\t\(\)\[\]{}!@#$%^&*\/\+\-=;:\\\\|`'\"~,.<>/?\n'])"

class CodeTokenizer(Tokenizer):

    def __init__(self,
                 token_dict,
                 token_cls=TOKEN_CLS,
                 token_sep=TOKEN_SEP,
                 token_unk=TOKEN_UNK,
                 pad_index=0,
                 cased=False,
                 code_stop_delim=default_code_stop_delim):
        """Initialize tokenizer.
        :param token_dict: A dict maps tokens to indices.
        :param token_cls: The token represents classification.
        :param token_sep: The token represents separator.
        :param token_unk: The token represents unknown token.
        :param pad_index: The index to pad.
        :param cased: Whether to keep the case.
        """
        
        super(CodeTokenizer, self).__init__(token_dict, 
                                            token_cls,
                                            token_sep,
                                            token_unk,
                                            pad_index,
                                            cased)
        self._code_stop_delim = code_stop_delim
        
        
    def tokenize_training(self, first, second=None):
        """Split text to tokens for BERT pre-training.
        :param first: First text.
        :param second: Second text.
        :return: A list of strings.
        """
        first_tokens = self._tokenize(first)
        second_tokens = self._tokenize(second) if second is not None else None
        return first_tokens, second_tokens
    
    def _tokenize(self, text):
        """Split text to tokens.
        :param first: First text.
        :param second: Second text.
        :return: A list of strings.
        """
        split_loc = re.split(self._code_stop_delim, text)
        split_loc = list(filter(lambda a: a != '', split_loc))
        
        tokens = []
        for word in split_loc:
            tokens += self._word_piece_tokenize(word)
        return tokens


def generate_code_pairs(file_lines, tokenizer, line_repeat_period=64, logger = None):
    """ Generates pairs of subsequent lines of code that can be used for BERT training."""

    # a line can only appear on each of the sides per given number of samples 
    # to minimize the chance of False negatives pairs while generating 
    left_lines = set()
    right_lines = set()

    no_files = len(file_lines)
    perc = no_files // 10

    code_line_pairs = []
 
    lines_counter = 0
    for i, file in enumerate(file_lines):
        
        if i % perc == 0 and logger is not None:
            logger.info(f"Processing lines in {i+1} / {no_files} of the files.")
            
        for j, left_line in enumerate(file):
            if j + 1 < len(file):
                right_line = file[j+1]
                if left_line not in left_lines and right_line not in right_lines:
                    code_line_pairs.append(tokenizer.tokenize_training(left_line, right_line))
                    left_lines.add(left_line)
                    right_lines.add(right_line)
                
            # handle line period reset
            lines_counter += 1
            if lines_counter == line_repeat_period:
                left_lines = set()
                right_lines = set()
                lines_counter = 0
    
    return code_line_pairs

