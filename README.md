# AMR Parsing via Graph-Sequence Iterative Inference

Code for our **ACL2020** paper, 

**AMR Parsing via Graph-Sequence Iterative Inference** [[preprint]](http://arxiv.org/abs/2004.05572)

Deng Cai and Wai Lam.

## Requirements

The code has been tested on **Python 3.6**.

All dependencies are listed in [requirements.txt](requirements.txt).

The code has two branches:

1. master branch corresponds to the experiments with graph recategorization
2. [no-recategorize branch](https://github.com/jcyk/AMR-gs/tree/no-recategorize) corresponds to the experiments without graph recategorization

## AMR Parsing with Pretrained Models
0. We are still working on a convenient API for parsing raw sentences. For now, a hacky solution is to convert to your input data into the LDC format (e.g., the novel [*The Little Prince*](https://amr.isi.edu/download/amr-bank-struct-v3.0.txt) in LDC format), and pretend it as our test set. You should wrap every sentence like this:

   ```
   # ::id 0
   # ::snt This is a sentence. (d / dummy) is used as a placeholder.
   (d / dummy)
   ```

1. Data Preprocessing: [Data Preparation](https://github.com/jcyk/stog#data-preparation) step 3-4. 

2. `sh work.sh` => `{load_path}{output_suffix}.pred`

3. `sh postprocess_2.0.sh` `{load_path}{output_suffix}.pred`=> `{load_path}{output_suffix}.pred.post`

## Download Links

|           Model           | Link |
| :-----------------------: | ---- |
| AMR2.0+BERT+GR+Smatch80.2 |  [amr2.0.bert.gr.tar.gz](https://drive.google.com/open?id=1v1fEoJGIrpM6kRzY796nDy2Ju6eFLs9-)    |

## Train New Parsers

The following instruction assumes that you're training on AMR 2.0 ([LDC2017T10](https://catalog.ldc.upenn.edu/LDC2017T10)). For AMR 1.0, the procedure is similar.

### Data Preparation

0. unzip the corpus to `data/AMR/LDC2017T10`.

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

It is recommended to use [fast smatch](./tools/fast_smatch) for model selection.

### Evaluation

For evaluation, following [Parsing with Pretrained Models](https://github.com/jcyk/stog#amr-parsing-with-pretrained-models) step 2-3, then `sh compute_smatch {load_path}{output_suffix}.pred.post data/AMR/amr_2.0/test.txt`

## Notes

1. We adopted the code snippets from [stog](https://github.com/sheng-z/stog) for data preprocessing.

2. The dbpedia-spotlight occasionally does not work. Therefore, we have [diabaled it](https://github.com/jcyk/AMR-gs/blob/52d2a95cb3c654d2dcefdd2bc85c5d54b84c027d/stog/data/dataset_readers/amr_parsing/postprocess/wikification.py#L61-L63).

## Contact
For any questions, please drop an email to [Deng Cai](https://jcyk.github.io/).
