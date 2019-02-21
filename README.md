# ACL_Workshop_Gender_Bias_NLP
https://genderbiasnlp.talp.cat/

1) datasets:
    - Gap Coreference
    - Book Corpus
    - UMBC
    
2) dataloaders: Ensures datasets are in proper format `[your_dataset].*`: one sentence or document per line
    - Gap Coreference Dataloader
    - Book Corpus Dataloader
    - UMBC Dataloader

3) allennlp_models.py: Run `[your_dataset].json` through allenNLP Coreference model. This will output a json file `[your_dataset]_coref.json`

```
pip3 install allennlp
```

Run script with Coreference model:
```
python3 allennlp_models.py \
    https://s3-us-west-2.amazonaws.com/allennlp/models/coref-model-2018.02.05.tar.gz \
    [your_dataset].json --output-file [your_dataset]_coref.json
```

4) A1_filter.py: Run `[your_dataset]_coref.json` through NLTK Part-of-Speech tagging and filter for gender bias type A1. Returns `[your_dataset]_A1.json`      


4) a) nltk_pos.py: Might want to save `[your_dataset]_coref.json` through NLTK Part-of-Speech tagging in a separate file before filtering `[your_dataset]_ntlk_pos.json`
    
%TO-DO:
- remove command line arguments from allennlp_models.py
- make sure the naming of `[your_dataset]_coref.json` is correct
- make whole thing run under one script rather than creating seperate files 
- try allennlp with .txt