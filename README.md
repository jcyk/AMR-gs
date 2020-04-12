# AMR Parsing via Graph-Sequence Iterative Inference

## Requirements

The code has been tested on **Python 3.6** and **PyTorch 1.2.0**.

All other dependencies are listed in [requirements.txt](requirements.txt).

The code has two branches:

1. master branch corresponds to the experiments with graph recategorization
2. no-recategorize branch corresponds to the experiments without graph recategorization

## AMR Parsing with Pretrained Models
0. We are still working on a convenient API for parsing raw sentences. For now, a hacky solution is to convert to your input data into the same format as LDC files, like

   ```
   # ::id 0
   # ::snt This is a sentence. (d / dummy) is used as a placeholder.
   (d / dummy)
   ```

1. Data Preparation [Data Preparation](https://github.com/jcyk/stog#data-preparation) step 3-4. 

2. `sh work.sh` => `{load_path}{output_suffix}.pred`

3. `sh postprocess_2.0.sh` `{load_path}{output_suffix}.pred`=> `{load_path}{output_suffix}.pred.post`

## Download Links

|           Model           | Link |
| :-----------------------: | ---- |
| AMR2.0+BERT+GR+Smatch80.2 |      |



## Train New Parsers

The following instruction assumes that you're training on AMR 2.0 ([LDC2017T10](https://catalog.ldc.upenn.edu/LDC2017T10)). For AMR 1.0, the procedure is similar.

### Data Preparation

0. unzip the corpus to `data/AMR/LDC2017T10`.

1. Prepare training/dev/test splits:

   ```sh prepare_data.sh -v 2 -p data/AMR/LDC2017T10```

2. Download Artifacts:

   ```sh download_artifacts.sh```

3. Feature Annotation:

   We use [Stanford CoreNLP](https://stanfordnlp.github.io/CoreNLP/index.html) (version **3.9.2**) for lemmatizing, POS tagging, etc.

   ```
   sh run_standford_corenlp_server.sh
   sh annotate_features.sh data/AMR/amr_2.0
   ```

4. Data Preprocessing:

   ```sh preprocess_2.0.sh ```

5. Building Vocabs

   ```sh prepare.sh data/AMR/amr_2.0```

### Training 

`sh train.sh data/AMR/amr_2.0`

It is recommended to use [fast smatch](./tools/fast_smatch) for model selection.

### Evaluation

For evaluation, following [Parsing with Pretrained Models](https://github.com/jcyk/stog#amr-parsing-with-pretrained-models) step 2-3, then `sh compute_smatch {load_path}{output_suffix}.pred.post data/AMR/amr_2.0/test.txt`

## Acknowledgements

We adopted the code snippets from [stog](https://github.com/sheng-z/stog) for data preprocessing.
