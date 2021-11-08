import re

from acora.code import default_code_stop_delim

class BERTVocab(object):
    """Vocabulary containing pairs of words and their ids. 
       To be used with BERT, it should contain special tokens [CLS], [SEP], and [UNK]."""


    def __init__(self, token_dict):
        self._token_dict = token_dict

    @classmethod
    def load_from_file(cls, vocab_path, limit=None):
        token_dict = {}
        with open(vocab_path, 'r', encoding='utf8', errors="ignore") as reader:
            for i, line in enumerate(reader):
                if limit is not None and i == limit:
                    break
                token = line.replace("\n", "").replace("\r", "")
                token_dict[token] = len(token_dict)

        return cls(token_dict)

    @property
    def size(self):
        return len(self._token_dict)

    @property
    def token_dict(self):
        return self._token_dict

    @property
    def token_list(self):
        return list(self._token_dict.keys())


def camel_case_split(identifier):
    """Split an identifier to substrings according to a camel case notation."""

    matches = re.finditer('.+?(?:(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|$)', identifier)
    return [m.group(0) for m in matches]

def code_vocab_tokenize(text, code_stop_delim=default_code_stop_delim, digit_token=True):
    """Uses a BERT approach to tokenize lines of code (the so-called word-piece tokenization)."""

    split_loc = re.split(code_stop_delim, text)
    split_loc = list(filter(lambda a: a != '', split_loc))
    tokens = []
    for token in split_loc:
        split_under = re.split("([_])", token)
        camel_case = camel_case_split(token)
        if len(split_under) > 1:  
            tokens.append(split_under[0])
            tokens.extend([f'##{str(x)}' for x in split_under[1:]])
        elif len(camel_case) > 1:
            tokens.append(camel_case[0])
            tokens.extend([f'##{str(x)}' for x in camel_case[1:]])
        elif token.isdigit() and not digit_token:
            digits = list(str(token))
            tokens.append(digits[0])
            tokens.extend([f'##{str(x)}' for x in digits[1:]])
        elif token.isdigit() and digit_token:
            tokens.append("[NUMBER]")
        else:
            tokens.append(token)
            
    return tokens