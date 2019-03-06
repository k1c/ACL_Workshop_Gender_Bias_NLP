# ACL_Workshop_Gender_Bias_NLP
https://genderbiasnlp.talp.cat/

1) datasets:
    - GAP Coreference
    - Book Corpus
    - UMBC
    
2) dataloaders: 
    - Gap Coreference Dataloader
    - Book Corpus Dataloader
    - UMBC Dataloader

3) experiments:
    - A1_filter: run filter on a small test set.
    - plot_tsne: run TSNE on `Bert_feature_extraction.json` produced by `./pytorch-pretrained-BERT/examples/extract_features.py`
    - allennlp_models.py: Run `[your_dataset].*` through various AllenNLP pre-trained models. Currently supporting Coreference and SRL. 

```
pip3 install allennlp
```

Run script with Coreference model:
```
python3 allennlp_models.py \
    https://s3-us-west-2.amazonaws.com/allennlp/models/coref-model-2018.02.05.tar.gz \
    [your_dataset].* --output-file [your_dataset]_coref.json
```

Run script with Semantic Role Labeling model:
```
python3 allennlp_models.py \
    https://s3-us-west-2.amazonaws.com/allennlp/models/srl-model-2018.02.27.tar.gz \
    [your_dataset].* --output-file [your_dataset]_coref.json
```
