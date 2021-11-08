
import logging
import numpy as np
import random
from sklearn.neighbors import NearestNeighbors

class SimilarLinesFinder():
    """Allows finding similar lines to the lines given as the reference database."""

    def __init__(self, cut_off_percentile=None, max_similar=None, cut_off_sample=500, at_least_one_result=True):
        """
        Parameters:
        -----------
        cut_off_percentile : int or None, it is used to calculate the maximum similarity threshold
                                    for the lines to be called similar. It is calculated as
                                    the given percentile of the distance between reference lines
                                    and their most similar lines in the reference dataset.
                                    If None is given, the cut off is ignored.
        max_similar : int or None, the maximum number of similar lines to be returned.
        cut_off_sample : int, the number of lines to sample while calculating the cut off threshold.
        at_least_one_result : bool, if set to True it will always return at least one similar line (the most similar one).
        """
        self.cut_off_percentile = cut_off_percentile
        self.max_similar = max_similar
        self.reference_lines = None
        self.reference_lines_embeddings = None
        self.nbrs = None
        self.logger = logging.getLogger('acora.code_similarities')
        self.dist_cut_off = None
        self.cut_off_sample = cut_off_sample
        self.at_least_one_result = at_least_one_result

    def fit(self, reference_lines, reference_lines_embeddings):
        """Fits a simlarity finder model
        
        Parameters:
        -----------
        reference_lines : a list of str, a list of reference lines.
        reference_lines_embeddings : a list of lists of floats, lines embeddings,
                                        it has to correspond index-wise to reference_lines.
        return : itself.
        """
        self.reference_lines = reference_lines
        self.reference_lines_embeddings = reference_lines_embeddings

        self.logger.debug(f"Fitting the NearestNeighbors model using the minkowski p=6 distance...")
        self.nbrs = NearestNeighbors(n_neighbors=self.max_similar if self.max_similar is not None else len(self.reference_lines),
                                    algorithm='ball_tree', metric='minkowski', p=6) 
        self.nbrs.fit(self.reference_lines_embeddings)

        if self.cut_off_percentile is not None:
            self.logger.debug(f"Calcuating the cut off point for similarity...")
            if self.cut_off_sample <= len(self.reference_lines):
                sample_size = self.cut_off_sample
            else:
                sample_size = len(self.reference_lines)
            dist_all, _ = self.nbrs.kneighbors(random.sample(self.reference_lines_embeddings, sample_size ), 
                              n_neighbors=2, 
                              return_distance=True)
            distances = [x[1] for x in dist_all]

            self.dist_cut_off = np.percentile(distances, self.cut_off_percentile)
            self.logger.debug(f"The distance threshold was set to {self.cut_off_percentile} percentile and is equal to {self.dist_cut_off}")
        else:
            self.dist_cut_off = None

        return self
        
    def query(self, lines, lines_embeddings):
        """Query reference lines to find similar lines to the lines given as the parameter.

        Parameters:
        -----------
        lines : a list of str, a list of lines to query for.
        lines_embeddings: a list of lists of floats, line embeddings that will be used to find similar lines
                                among the reference lines.
        return : a list of tuples, each tupple consists of a query line, a similar reference line, and the distance measure.
        """
        if self.dist_cut_off is not None:
            dist, sims_ids = self.nbrs.radius_neighbors(lines_embeddings, 
                                radius=self.dist_cut_off,
                                sort_results=True,
                                return_distance=True)
        else:
            dist, sims_ids = self.nbrs.kneighbors(lines_embeddings, 
                                return_distance=True)
        result = []
        for i, sim_ids in enumerate(sims_ids):
            if self.max_similar is not None:
                sim_ids = sim_ids[:self.max_similar]
       
            for j, sim_id in enumerate(sim_ids):
                result.append((lines[i], self.reference_lines[sim_id], dist[i][j]))
        
        return result
    
    def query_to_dict(self, lines, lines_embeddings):
        """Query reference lines to find similar lines to the lines given as the parameter and returns 
        a dictionary with lines as keys and list of similar lines as values..

        Parameters:
        -----------
        lines : a list of str, a list of lines to query for.
        lines_embeddings: a list of lists of floats, line embeddings that will be used to find similar lines
                                among the reference lines.
        return : a list of tuples, each tupple consists of a query line, a similar reference line, and the distance measure.
        """
        if not self.at_least_one_result and self.dist_cut_off is not None:
            dist, sims_ids = self.nbrs.radius_neighbors(lines_embeddings, 
                                radius=self.dist_cut_off,
                                sort_results=True,
                                return_distance=True)
        else:
            dist, sims_ids = self.nbrs.kneighbors(lines_embeddings, 
                                return_distance=True)
        result = {}
        for i, sim_ids in enumerate(sims_ids):
            line = lines[i]
            if line not in result:
                if self.max_similar is not None:
                    sim_ids = sim_ids[:self.max_similar]

                if not self.at_least_one_result:
                    result[line] = [self.reference_lines[sim_id] for sim_id in sim_ids]
                else:
                    sim_lines = [self.reference_lines[sim_id] for j, sim_id in enumerate(sim_ids) if dist[i][j] <= self.dist_cut_off]
       
                    if len(sim_lines) == 0:
                        sim_lines = [self.reference_lines[sim_ids[0]],]
                    result[line] = sim_lines

        
        return result


        

        