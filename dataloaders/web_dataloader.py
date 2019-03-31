'''
Purpose: General Purpose dataloader that passes text data through allen-nlp coref, allen-nlp srl, and nltk pos tagging
Usage: python web_dataloader.py input_dataset_path

Input: dataset.tsv (one sentence per line)
Output: dataframe that has the following dataframe structure:
        sentence | coref cluster | srl | pos tags |

There's also a function called preprocess() that cleans sentences (depending on your data, uncomment what you need
'''

from allennlp.predictors import Predictor
from allennlp.models.archival import load_archive
import pandas as pd
import re
import unidecode
import pprint
pp = pprint.PrettyPrinter(indent=1)

#Use the NLTK Downloader to obtain the resources that you need for this script:
import nltk
#nltk.download('averaged_perceptron_tagger')
#nltk.download('punkt')
#nltk.download('averaged_perceptron_tagger')
#nltk.download('maxent_ne_chunker')
#nltk.download('words')
#nltk.download('tagsets')


class WebDataset(object):
    def __init__(self, data_path, encoding="utf-8"):
        self.data_path = data_path
        self.encoding = encoding

    def preprocess(self, sentences):
        new_data = []
        for sentence in sentences:
            new_sentence = re.sub('"', '', sentence)  # removes quotation marks
            new_sentence = unidecode.unidecode(new_sentence)  # removes accents and represents any unicode to closest ascii
            # new_sentence = re.sub('<.*?>', '', new_sentence)  # remove HTML tags (in case, should be clean)
            # new_sentence = re.sub(r'[^\w\s]', '', new_sentence) # remove punctuation
            # new_sentence = new_sentence.lower() # convert to lower case
            wordcount = len(new_sentence.split())
            if new_sentence != '' and wordcount > 1: # wordcount of 1 breaks allennlp
                new_data.append(new_sentence)
        return new_data

    def loader(self):
        df = pd.read_csv(self.data_path,delimiter='\t', encoding='utf-8', header=None, squeeze=True)

        coref_archive = load_archive('https://s3-us-west-2.amazonaws.com/allennlp/models/coref-model-2018.02.05.tar.gz',
                               weights_file=None,
                               overrides="")

        srl_archive = load_archive('https://s3-us-west-2.amazonaws.com/allennlp/models/srl-model-2018.02.27.tar.gz',
                               weights_file=None,
                               overrides="")

        coref_results = []
        srl_results = []
        pos_results = []

        df_clean = self.preprocess(df)
        print(len(df_clean))
        for line in df_clean:
            coref_line = {"document":line.strip()}
            srl_line = {"sentence": line.strip()}
            coref_results.append(Predictor.from_archive(coref_archive, 'coreference-resolution').predict_json(coref_line))
            srl_results.append(Predictor.from_archive(srl_archive, 'semantic-role-labeling').predict_json(srl_line))
            pos_results.append(nltk.pos_tag(nltk.word_tokenize(line.strip())))

        #Next Step: build a dataset that will hold the 3 results of a sentence per row: https://github.com/pytorch/text/blob/master/torchtext/datasets/sequence_tagging.py



#Used temporarily for testing
if __name__ == '__main__':
    input_path = '../datasets/test_datasets/test_dataset.txt'

    web_data = WebDataset(input_path)
    web_data.loader()

