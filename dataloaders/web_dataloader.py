import pandas as pd
from torchtext import data
import re
from allennlp.predictors import Predictor
from allennlp.models.archival import load_archive
import nltk
import pprint
import pandas as pd
import re
import nltk
nltk.download('averaged_perceptron_tagger')
#Use the NLTK Downloader to obtain the resources that you need for this script:
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('maxent_ne_chunker')
nltk.download('words')
nltk.download('tagsets')

import unidecode
pp = pprint.PrettyPrinter(indent=1)

class WebDataset(object):
    def __init__(self, data_path, encoding="utf-8"):
        self.data_path = data_path
        self.encoding = encoding

    def preprocess(self, sentences):
        new_data = []
        for sentence in sentences:
            new_sentence = re.sub('"', '', sentence)  # remove quptation marks
            new_sentence = unidecode.unidecode(new_sentence)  # removes accents and represents any unicode to closest ascii
            # bnew_sentence = re.sub('<.*?>', '', new_sentence)  # remove HTML tags (in case, should be clean)
            # new_sentence = re.sub(r'[^\w\s]', '', new_sentence) # remove punctuation
            # new_sentence = new_sentence.lower() # convert to lower case
            if new_sentence != '':
                new_data.append(new_sentence)
        return new_data

    def loader(self):
       #df = pd.read_csv(self.data_path, delimiter='\t', encoding='utf-8', header=None)
        df = pd.read_csv(self.data_path,delimiter='\t', encoding='utf-8', header=None, squeeze=True)# sep=','

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

        print('complete')
        #Next Step: build a dataset that will hold the 3 results of a sentence per row: https://github.com/pytorch/text/blob/master/torchtext/datasets/sequence_tagging.py



#Used temporarily for testing
if __name__ == '__main__':
    web_data = WebDataset('../datasets/test_datasets/biasly_data.tsv')
    web_data.loader()

