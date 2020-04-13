#!/usr/bin/env bash

set -e

echo "Building artifacts."

# ############### AMR v3.0 ################
# # Directory where intermediate utils will be saved to speed up processing.
util_dir=data/AMR/amr_3.0_utils

# AMR data with **features**
data_dir=data/AMR/amr_3.0
train_data=${data_dir}/train.txt.features
dev_data=${data_dir}/dev.txt.features
test_data=${data_dir}/test.txt.features

# ========== Set the above variables correctly ==========
compound_file=${util_dir}/joints.txt
python -u -m stog.data.dataset_readers.amr_parsing.preprocess.feature_annotator \
    ${data_dir}/train.txt \
    --compound_file ${compound_file}

printf "Cleaning inputs...`date`\n"
python -u -m stog.data.dataset_readers.amr_parsing.preprocess.input_cleaner \
    --amr_files ${train_data}
printf "Done.`date`\n\n"

printf "Recategorizing subgraphs...`date`\n"
python -u -m stog.data.dataset_readers.amr_parsing.preprocess.recategorizer \
    --dump_dir ${util_dir} \
    --amr_train_file ${train_data}.input_clean \
    --amr_files ${train_data}.input_clean \
    --build_utils
printf "Done.`date`\n\n"

printf "Removing senses...`date`\n"
python -u -m stog.data.dataset_readers.amr_parsing.node_utils \
    --amr_train_files ${train_data} \
    --dump_dir ${util_dir}
printf "Done.`date`\n\n"

printf "Preprocessing files...`date`\n"
sh preprocess_3.0.sh

printf "Removing wiki...`date`\n"
python -u -m stog.data.dataset_readers.amr_parsing.postprocess.wikification \
    --amr_files ${dev_data}.preproc ${test_data}.preproc  \
    --util_dir ${util_dir} \
    --dump_spotlight_wiki
printf "Done.`date`\n\n"

