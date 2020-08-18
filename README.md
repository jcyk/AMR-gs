# AMR Parsing via Graph-Sequence Iterative Inference

Code for our **ACL2020** paper, 

**AMR Parsing via Graph-Sequence Iterative Inference** [[preprint]](http://arxiv.org/abs/2004.05572)

Deng Cai and Wai Lam.

## Introduction

The code has two branches:

1. master branch corresponds to the experiments with graph recategorization.
2. [no-recategorize branch](https://github.com/jcyk/AMR-gs/tree/no-recategorize) corresponds to the experiments without graph recategorization.

## Requirements

The code has been tested on **Python 3.6**. All dependencies are listed in [requirements.txt](requirements.txt).

## AMR Parsing with Pretrained Models
0. We use [Stanford CoreNLP](https://stanfordnlp.github.io/CoreNLP/index.html) (version **3.9.2**) for lemmatizing, POS tagging, etc.

    ```
    sh run_standford_corenlp_server.sh
    ```

    The input file should constain the raw sentences to parse (one sentence per line).

1. Data Preprocessing: `sh preprocess_raw.sh ${input_file}`

2. `sh work.sh` => `{load_path}{output_suffix}.pred`

3. `sh postprocess_2.0.sh` `{load_path}{output_suffix}.pred`=> `{load_path}{output_suffix}.pred.post`

## Download Links

|           Model           | Link |
| :-----------------------: | ---- |
| AMR2.0+BERT+GR=Smatch80.2 |  [amr2.0.bert.gr.tar.gz](https://drive.google.com/open?id=1v1fEoJGIrpM6kRzY796nDy2Ju6eFLs9-)    |
|AMR2.0+BERT=Smatch78.7|[amr2.0.bert.tar.gz](https://drive.google.com/file/d/1S-P6Y6c-_5-uzcqyX7-BixVUvFzN0MG9/view?usp=sharing) |

## Train New Parsers

The following instruction assumes that you're training on AMR 2.0 ([LDC2017T10](https://catalog.ldc.upenn.edu/LDC2017T10)). For AMR 1.0, the procedure is similar.

### Data Preparation

0. Unzip the corpus to `data/AMR/LDC2017T10`.

1. Prepare training/dev/test splits:

   ```sh prepare_data.sh -v 2 -p data/AMR/LDC2017T10```

3. Download Artifacts:

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

The training process will produce many checkpoints and the corresponding output on dev set. To select the best checkpoint, one can evaluate the dev output files (need to do postprocessing first). It is recommended to use [fast smatch](./tools/fast_smatch) for model selection.

### Evaluation

For evaluation, following [Parsing with Pretrained Models](https://github.com/jcyk/AMR-gs#amr-parsing-with-pretrained-models) step 2-3, then `sh compute_smatch {load_path}{output_suffix}.pred.post data/AMR/amr_2.0/test.txt`.

## Notes

1. We adopted the code snippets from [stog](https://github.com/sheng-z/stog) for data preprocessing.

2. The dbpedia-spotlight occasionally does not work. Therefore, we have [disabled it](https://github.com/jcyk/AMR-gs/blob/52d2a95cb3c654d2dcefdd2bc85c5d54b84c027d/stog/data/dataset_readers/amr_parsing/postprocess/wikification.py#L61-L63).

## Contact
For any questions, please drop an email to [Deng Cai](https://jcyk.github.io/).
