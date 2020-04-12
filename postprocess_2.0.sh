#!/usr/bin/env bash

set -e

# Directory where intermediate utils will be saved to speed up processing.
util_dir=data/AMR/amr_2.0_utils

# AMR data with **features**
test_data=$1


python3 -u -m stog.data.dataset_readers.amr_parsing.postprocess.postprocess \
    --amr_path ${test_data} \
    --util_dir ${util_dir} \
    --v 2
printf "Done.`date`\n\n"
