#! /bin/bash

# ensure that the script has been sourced rather than just executed
if [[ "${BASH_SOURCE[0]}" = "${0}" ]]; then
    echo "Please use 'source' to execute env.sh!"
    exit 1
fi

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
VGGISH_PATH="${DIR}/deps/tf-models/research/audioset/vggish/"
YAMNET_PATH="${DIR}/deps/tf-models/research/audioset/yamnet/"
LAUGHTER_PATH="${DIR}/deps/laughter-detection/"

# Check the directories exist
if [[ ! -d $VGGISH_PATH ]]; then
    echo "Can't find VGGISH_PATH!"
fi

if [[ ! -d $YAMNET_PATH ]]; then
    echo "Can't find YAMNET_PATH!"
fi

if [[ ! -d $LAUGHTER_PATH ]]; then
    echo "Can't find LAUGHTER_PATH!"
fi

# update environment variables
export PODCAST_PATH=$DIR
export VGGISH_PATH=$VGGISH_PATH
export YAMNET_PATH=$YAMNET_PATH
export LAUGHTER_PATH=$LAUGHTER_PATH
export PYTHONPATH=$PYTHONPATH:$VGGISH_PATH:$YAMNET_PATH:$LAUGHTER_PATH
export TOKENIZERS_PARALLELISM="false"
