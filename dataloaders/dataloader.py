from allennlp.predictors import Predictor
from allennlp.models.archival import load_archive
from nltk.tokenize.treebank import TreebankWordDetokenizer
import pandas as pd
import re
import unidecode
import pprint
from tqdm import tqdm
import nltk
from nltk.tokenize import sent_tokenize
from nltk import ne_chunk, pos_tag
from nltk.tokenize import sent_tokenize, word_tokenize
pp = pprint.PrettyPrinter(indent=1)

#from loaders import load_IMDB, load_gluternberg
from filter import check_remove
from loaders import load_IMDB, load_gutenberg, load_general

#Use the NLTK Downloader to obtain the resources that you need for this script:
# nltk.download('averaged_perceptron_tagger')
# nltk.download('punkt')
# nltk.download('averaged_perceptron_tagger')
# nltk.download('maxent_ne_chunker')
# nltk.download('words')
# nltk.download('tagsets')


class Dataloader(object):
    def __init__(self, output_name, filter, encoding="utf-8"):
        self.output_name = output_name
        self.encoding = encoding
        self.filter = filter
        self.predictor = Predictor.from_archive(
            load_archive('https://s3-us-west-2.amazonaws.com/allennlp/models/coref-model-2018.02.05.tar.gz',
                         weights_file=None,
                         overrides=""), 'coreference-resolution')

# coref_json = {'top_spans': [[0, 0], [0, 8], [2, 5], [2, 8], [7, 8], [12, 12]], 'predicted_antecedents': [-1, -1, -1, -1, -1, 4], 'document': ['This', 'is', 'an', 'action', '-', 'melodrama', 'like', 'the', 'world', 'has', 'never', 'seen', 'it', 'before', '.'], 'clusters': [[[0, 0], [12, 12]]]}
    def filter_to_file(self, data):
        coref_true = []
        f = open(self.output_name, "w+")

        for line in tqdm(data):
            coref_line = {"document":line.strip()}
            try:
                coref_json = self.predictor.predict_json(coref_line)
            except KeyboardInterrupt:
                print("KeyboardInterrup")
                break
            except:
                print("problem sentence: ", line)

            # if coref cluster exists, then add the coref json to coref_true list
            if len(coref_json['clusters']) > 0 and self.filter(TreebankWordDetokenizer().detokenize(coref_json['document']),coref_json['clusters'],"all") is False:
                f.write(TreebankWordDetokenizer().detokenize(coref_json['document']) + "\n")

        f.close()
        print("write to file complete")



#Used temporarily for testing
if __name__ == '__main__':
    input_path = '../datasets/test_datasets/test_dataset.txt'
    output_name = "test_dataset"

    dataloader = Dataloader(output_name, check_remove)
    data = load_general(input_path)
    dataloader.filter_to_file(data)