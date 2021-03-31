__all__ = [
    "DATA_AUDIO",
    "DATA_OPENSMILE_FUNCTIONALS",
    "DATA_OPENSMILE_FUNCTIONALS_1s",
    "DATA_VGGISH_LOG_MEL",
    "DATA_VGGISH_EMBED",
    "DATA_VGGISH_POSTPROCESSED",
    "DATA_YAMNET_EMBED",
    "DATA_YAMNET_SCORES",
]

# raw data
DATA_AUDIO = "/unix/cdtdisspotify/data/spotify-podcasts-2020/podcasts-audio/"

# OpenSmile features
# DATA_OPENSMILE_FUNCTIONALS = "/mnt/storage/cdtdisspotify/eGeMAPSv01b/intermediate_uri/func/"
DATA_OPENSMILE_FUNCTIONALS_1s = (
    "/mnt/storage/cdtdisspotify/eGeMAPSv01b/intermediate_uri/func_1s/"
)

DATA_OPENSMILE_FUNCTIONALS = "/mnt/storage/cdtdisspotify/features/eGeMAPSv02/"

# VGGish features
DATA_VGGISH_LOG_MEL = "/mnt/storage/cdtdisspotify/vggish/intermediate_uri/log_mel/"
DATA_VGGISH_EMBED = "/mnt/storage/cdtdisspotify/vggish/intermediate_uri/embedding/"
DATA_VGGISH_POSTPROCESSED = (
    "/mnt/storage/cdtdisspotify/vggish/intermediate_uri/postprocess/"
)

# YAMNet features
DATA_YAMNET_EMBED = "/mnt/storage/cdtdisspotify/yamnet/intermediate_uri/embedding/"
DATA_YAMNET_SCORES = "/mnt/storage/cdtdisspotify/yamnet/intermediate_uri/scores/"

DATA_YAMNET_EMBED = "/mnt/storage/cdtdisspotify/features/yamnet/embedding/"
DATA_YAMNET_SCORES = "/mnt/storage/cdtdisspotify/features/yamnet/scores/"
