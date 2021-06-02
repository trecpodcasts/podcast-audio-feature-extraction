# -*- coding: utf-8 -*-

"""Base feature extraction class."""

import os
import multiprocessing
from functools import partial
from tqdm import tqdm


class FeatureExtractor:
    """Base class for feature extractors."""

    def __init__(self, logfile="./log"):
        """Init method for FeatureExtractor."""
        self.log_file = logfile

        # Create the log file
        log_path_exists = os.path.exists(logfile)
        if not log_path_exists:
            directory = os.path.dirname(logfile)
            if not os.path.exists(directory):
                os.makedirs(directory)
            f = open(self.log_file, "w")
            f.write("Files skipped:\n")
            f.close()

    def multi_process(self, function, iterable, num_workers=1):
        """Multiprocessing wrapper for feature extraction.

        Args:
            function (function): the function to multiprocess
            iterable (iterable): iterable to be given to function
            num_workers (int, optional): Amount of jobs to use. Defaults to 1.

        """
        pbar = tqdm(total=len(iterable))
        f = partial(self._process_wrapper, function=function, log=self.log_file)
        with multiprocessing.Pool(processes=num_workers) as pool:
            res = [
                pool.apply_async(f, args=(i,), callback=lambda _: pbar.update(1))
                for i in iterable
            ]
            results = [p.get() for p in res]  # noqa: F841

    def single_process(self, function, iterable):
        """Single core processing wrapper for feature extraction.

        Args:
            function (function): the function to multiprocess
            iterable (iterable): iterable to be given to function

        """
        for i in tqdm(iterable):
            self._process_wrapper(i, function, self.log_file)

    @staticmethod
    def _process_wrapper(argument, function=lambda x: x, log="./log"):
        try:
            function(argument)
        except Exception:
            with open(log, "a") as f:
                f.write(
                    "spotify:episode:{}\n".format(argument[0].split("/")[-1][:-4])
                )  # should produce the uri only

    @staticmethod
    def feature_path_checker(input_path, output_path):
        """Check the input and output paths.

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
            directory = os.path.dirname(output_path)  # returns path to folder of file
            if not os.path.exists(directory):
                os.makedirs(directory)  # create all necessary folders

        return input_path_exists, output_path_exists
