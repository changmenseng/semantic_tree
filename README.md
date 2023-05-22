This is the code implementation of our paper "Interpreting Sentiment Composition with Latent Semantic Tree" in ACL2023 Findings.

First, please install the requirements:

```shell
pip install -r requirements.txt
```

Then, please use the following command to run the experiment of the model and dataset you want

```shell
python main.py -d <dataset> -m <model>
```

The script `main.py` will use the config files in the folder `configs`, which should reproduce the results in the paper. The model checkpoints, metric logging file, as well as the model outputs (including predicted labels and generated semantic trees) will be stored in the folder `saved/<dataset>/<model>_scm`. Also, you can check the training procedure using tensorboard:

```shell
tensorboard --logdir saved/<dataset>/<model>_scm
```