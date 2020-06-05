
import logging
import numpy as np
from sklearn.neighbors import NearestNeighbors

class SimilarLinesFinder():
    """Allows finding similar lines to the lines given as the reference database."""

    def __init__(self, cut_off_percentile=None, max_similar=None):
        """
        Parameters:
        -----------
        cut_off_percentile : int or None, it is used to calculate the maximum similarity threshold
                                    for the lines to be called similar. It is calculated as
                                    the given percentile of the distance between reference lines
                                    and their most similar lines in the reference dataset.
                                    If None is given, the cut off is ignored.
        max_similar : int or None, the maximum number of similar lines to be returned.
        """
        self.cut_off_percentile = cut_off_percentile
        self.max_similar = max_similar
        self.reference_lines = None
        self.reference_lines_embeddings = None
        self.nbrs = None
        self.logger = logging.getLogger('acora.code_similarities')
        self.dist_cut_off = None

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
            dist_all, _ = self.nbrs.kneighbors(self.reference_lines_embeddings, 
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


        

        