#!/usr/bin/env bash

set -e

# Start a Stanford CoreNLP server before running this script.
# https://stanfordnlp.github.io/CoreNLP/corenlp-server.html

# The compound file is downloaded from
# https://github.com/ChunchuanLv/AMR_AS_GRAPH_PREDICTION/blob/master/data/joints.txt
compound_file=data/AMR/amr_2.0_utils/joints.txt
raw_file=$1

python prepare_raw.py ${raw_file}


python -u -m stog.data.dataset_readers.amr_parsing.preprocess.feature_annotator \
    ${raw_file}.raw \
    --compound_file ${compound_file}


# ############### AMR v2.0 ################
# # Directory where intermediate utils will be saved to speed up processing.
util_dir=data/AMR/amr_2.0_utils

# AMR data with **features**
test_data=${raw_file}.raw.features

# ========== Set the above variables correctly ==========

printf "Cleaning inputs...`date`\n"
python -u -m stog.data.dataset_readers.amr_parsing.preprocess.input_cleaner \
    --amr_files ${test_data}
printf "Done.`date`\n\n"

printf "Recategorizing subgraphs...`date`\n"
python -u -m stog.data.dataset_readers.amr_parsing.preprocess.text_anonymizor \
    --amr_file ${test_data}.input_clean \
    --util_dir ${util_dir}
printf "Done.`date`\n\n"

printf "Removing senses...`date`\n"
python -u -m stog.data.dataset_readers.amr_parsing.preprocess.sense_remover \
    --util_dir ${util_dir} \
    --amr_files ${test_data}.input_clean.recategorize
printf "Done.`date`\n\n"

printf "Renaming preprocessed files...`date`\n"
mv ${test_data}.input_clean.recategorize.nosense ${test_data}.preproc
rm ${test_data}*.input_clean*
