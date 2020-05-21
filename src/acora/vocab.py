

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
                if limit and i == limit:
                    break
                token = line.strip()
                token_dict[token] = len(token_dict)

        return cls(token_dict)

    @property
    def size(self):
        return len(self._token_dict)

    @property
    def token_dict(self):
        return self._token_dict