__all__ = ["OpenSmileExtractor"]

import opensmile

from src.features import FeatureExtractor

# TODO change over to eGeMAPSv02 before rerunning on full dataset
SMILE = opensmile.Smile(  # Create the functionals extractor here
    feature_set=opensmile.FeatureSet.eGeMAPSv01b,
    feature_level=opensmile.FeatureLevel.Functionals,
)

class OpenSmileExtractor(FeatureExtractor):
    """Class for feature extraction with opensmile

        example:
        extractor = OpenSmileExtractor():
        extractor.extract(paths, num_workers=2) 
    """

    def extract(self, input_paths, output_paths, num_workers=1):
        """extract eGeMAPS features with opensmile using multiprocessing

        Args:
            input_paths  (str): paths to input files
            output_paths (str): paths to storage location for output of each input file
            num_workers (int, optional): Amount of cores to use. Defaults to 1.
        """
        paths = list(zip(input_paths, output_paths))
        self.multi_process(self._process, paths, num_workers=num_workers)

    @staticmethod
    def _process(paths):
        input_path, output_path = paths
        input_path_exists, output_path_exists = FeatureExtractor.feature_path_checker(input_path, output_path)

        if input_path_exists and not output_path_exists:
            features = SMILE.process_file(input_path, channel=1) 
            features.to_pickle(output_path) # save to pickle file
            del features
            # TODO change the way this data is saved to be similar to vgg?