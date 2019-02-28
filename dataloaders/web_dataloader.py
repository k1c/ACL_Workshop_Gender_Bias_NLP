import pandas as pd
from torchtext import data
import re
from allennlp.predictors import Predictor
from allennlp.models.archival import load_archive
import nltk

class WebDataset(object):
    def __init__(self, data_path, encoding="utf-8"):
        self.data_path = data_path
        self.encoding = encoding

    def loader(self):
        # Clean data (not sure how to incorporate this for now)
        clean = lambda x: re.sub(r'<.*?>|[^\w\s]|\d+', '', x)  #.split() ?

        df = pd.read_csv(self.data_path, sep=',', header=None, squeeze=True)

        coref_archive = load_archive('https://s3-us-west-2.amazonaws.com/allennlp/models/coref-model-2018.02.05.tar.gz',
                               weights_file=None,
                               overrides="")

        srl_archive = load_archive('https://s3-us-west-2.amazonaws.com/allennlp/models/srl-model-2018.02.27.tar.gz',
                               weights_file=None,
                               overrides="")

        coref_results = []
        srl_results = []
        pos_results = []

        for line in df:
            coref_line = {"document":line.strip()}
            srl_line = {"sentence": line.strip()}
            coref_results.append(Predictor.from_archive(coref_archive, 'coreference-resolution').predict_json(coref_line))
            srl_results.append(Predictor.from_archive(srl_archive, 'semantic-role-labeling').predict_json(srl_line))
            pos_results.append(nltk.pos_tag(nltk.word_tokenize(line.strip())))

        #used for testing
        print("Size coref", len(coref_results))
        print("Size srl", len(srl_results))
        print("Size pos", len(pos_results))

        #Next Step: build a dataset that will hold the 3 results of a sentence per row: https://github.com/pytorch/text/blob/master/torchtext/datasets/sequence_tagging.py



#Used temporarily for testing
if __name__ == '__main__':
    web_data = WebDataset('../datasets/test_dataset.txt')
    web_data.loader()

