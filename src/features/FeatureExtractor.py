__all__ = ["FeatureExtractor"]

import os
import multiprocessing

from tqdm import tqdm

class FeatureExtractor():
    """
    Base class for feature extractors.
    """

    def multi_process(self, function, iterable, num_workers=1 ):
        """multiprocessing wrapper for feature extraction

        Args:
            function (function): the function to multiprocess
            iterable (iterable): iterable to be given to function
            num_workers (int, optional): Amount of jobs to use. Defaults to 1.
        
        TODO add progress bar
        """
        with multiprocessing.Pool(processes = num_workers) as pool:
            pool.map(function, iterable)     
        
    def single_process(self, function, iterable):
        """Single core processing wrapper for feature extraction

        Args:
            function (function): the function to multiprocess
            iterable (iterable): iterable to be given to function

        """
        for i in tqdm(iterable):
            function(i)

    @staticmethod
    def _process_wrapper(paths, function):
        # TODO implement wrapper for checking paths
        raise NotImplementedError

    @staticmethod
    def feature_path_checker(input_path, output_path):
        """
        - Checks if input path exists 
        - Creates output directory if necessary
        - Checks whether output file is already created

        Args:
            input_path (str): path to input file
            output_path (str): path to output file

        Returns:
            input_path_exists (bool): if input file exists
            output_path_exists (bool): if output file already exists
        """

        input_path_exists = os.path.exists(input_path)
        output_path_exists = os.path.exists(output_path)
        if not output_path_exists:
            directory = os.path.dirname(output_path) # returns path to folder of file
            if not os.path.exists(directory):
                os.makedirs(directory) # create all necessary folders

        return input_path_exists, output_path_exists