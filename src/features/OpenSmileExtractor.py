__all__ = ["OpenSmileExtractor"]

import opensmile

from src.features import FeatureExtractor

# TODO change over to eGeMAPSv02 before rerunning on full dataset
SMILE = opensmile.Smile(  # Create the functionals extractor here
    feature_set=opensmile.FeatureSet.eGeMAPSv02,
    feature_level=opensmile.FeatureLevel.Functionals,
)

class OpenSmileExtractor(FeatureExtractor):
    """Class for feature extraction with opensmile

        The configuration for the OpenSmile extractor needs to be altered in the package. In particular the file: "opensmile/core/config/shared/FrameModeFunctionals.conf.inc"
        
        You need to change:
        frameMode = fixed
        frameSize = 0.96
        frameStep = 0.48
        frameCenterSpecial = left


        example:
        extractor = OpenSmileExtractor()
        extractor.extract(paths, num_workers=2) 
    """
    def __init__(self):
        super().__init__(logfile="./log_OpenSmile")


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
            features.reset_index(inplace=True)
            features["time (s)"] = features["start"].dt.total_seconds()
            del features["start"]
            del features["file"]
            del features["end"]
            features.set_index("time (s)", inplace=True)

            features.to_hdf(
                output_path, 
                'OpenSmile_Functionals', 
                mode = 'w',
                complevel=9
                )# save to pickle file
            del features
