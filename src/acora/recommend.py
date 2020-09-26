import logging
import numpy as np

class CodeReviewFocusRecommender():
    """Detects lines of code on which a review should focus on"""

    def __init__(self, classifier, code_tokenizer, 
                seq_len, similarity_finder, embeddings_extractor,
                review_comments_df, line_column, purpose_column,
                subject_columns, message_column,
                classify_threshold=0.5):
        """
        Parameters:
        -----------
        classifier : keras model, a BERT model trained to detect lines that will be commented on.
        code_tokenizer : acora.code.CodeTokenizer, a tokenizer to use to tokenize the lines for classification.
        seq_len : int, a maximum length of a line of code.
        embeddings_extractor : acora.code_embeddings.CodeLinesBERTEmbeddingsExtractor, used to extract embeddings.
        similarity_finder : acora.code_similarities.SimilarLinesFinder, a finder used to search for reference comments.
        review_comments_df : pandas.DataFrame, data frame with review comments,
        line_column : str, a column name with line contents.
        purpose_column : str, a column name of comment purpose.
        subject_columns : a list of str, the names of a comment subject columns.
        message_column : str, a name of the column storing comments.
        classify_threshold : float, a threshold for classification decision - default is 0.5.
        """
        self.classifier = classifier
        self.tokenizer = code_tokenizer
        self.seq_len = seq_len
        self.embeddings_extractor = embeddings_extractor
        self.similarity_finder = similarity_finder
        self.review_comments_df = review_comments_df
        self.line_column = line_column
        self.purpose_column = purpose_column
        self.subject_columns = subject_columns
        self.message_column = message_column
        self.classify_threshold = classify_threshold

        self.logger = logging.getLogger('acora.recommend')


    def review(self, lines):
        """Reviews the givnen lines and returns the recommendations.
        
        Parameters:
        -----------
        lines : a list of str, lines to review.
        return : a list, each entry in the list consists of the line contents, decsion 
                            (1 - the reviewer should focus on the line, 0 - ignore),
                            and some focus information as a dictionary.
        """
        result = []
        review_decisions = self._detect_lines_to_comment(lines)
        suspicious_lines = list({line for i, line in enumerate(lines) if review_decisions[i] == 1})

        self.logger.debug('Extracting embeddings...')
        embeddings = self.embeddings_extractor.extract_embeddings(suspicious_lines)
        self.logger.debug('Finding similar lines...')
        
        similar_lines_dict = self.similarity_finder.query_to_dict(suspicious_lines, embeddings)

        self.logger.debug("Processing the results of classification")
        for i, decision in enumerate(review_decisions):
            if i % 100 == 0:
                self.logger.debug(f"Processing the line {i+1} of {len(lines)}...")
            line = lines[i]
            recommendations = {}
            if decision == 1:
                similar_lines = set(similar_lines_dict.get(line, []))
                comments_df = self.review_comments_df[self.review_comments_df[self.line_column].isin(similar_lines)]
                recommendations['no_similar_lines'] = len(similar_lines)
                recommendations['no_comments'] = comments_df.shape[0]
                recommendations['purpose'] = comments_df.groupby(self.purpose_column)[self.purpose_column].count().div(comments_df.shape[0]).to_dict()
                recommendations['subject'] = comments_df[self.subject_columns].sum(axis=0, skipna=True).div(comments_df.shape[0]).to_dict()
                recommendations['comments_lines'] = comments_df[self.line_column].tolist()
                recommendations['comments_messages'] = comments_df[self.message_column].tolist()
            
            result.append((line, decision, recommendations))

        return result

    def _detect_lines_to_comment(self, lines):
        """Classifies lines to detect those that should be commented on.

        Paremeters:
        -----------
        lines : a list of str, lines to be classified.
        return : a list of int, a list with decisions for each line: 1 - to comment, 0 - OK
        """
        self.logger.debug("Tokenizing lines...")
        tokenized_all_code_lines = [self.tokenizer.encode(text, max_len=self.seq_len)[0] for text in lines]
        x_all = [np.array(tokenized_all_code_lines), np.zeros_like(tokenized_all_code_lines)]

        self.logger.debug("Classifying lines...")
        y_all_pred = self.classifier.predict(x_all)

        return [1 if y >= self.classify_threshold else 0 for y in y_all_pred]
  


