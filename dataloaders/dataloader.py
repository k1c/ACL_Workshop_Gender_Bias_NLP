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
from filter import remove_sentence

#Use the NLTK Downloader to obtain the resources that you need for this script:
# nltk.download('averaged_perceptron_tagger')
# nltk.download('punkt')
# nltk.download('averaged_perceptron_tagger')
# nltk.download('maxent_ne_chunker')
# nltk.download('words')
# nltk.download('tagsets')


class Dataloader(object):
    def __init__(self, data_path, output_name, filter, encoding="utf-8"):
        self.data_path = data_path
        self.output_name = output_name
        self.encoding = encoding
        self.filter = filter
        self.predictor = Predictor.from_archive(
            load_archive('https://s3-us-west-2.amazonaws.com/allennlp/models/coref-model-2018.02.05.tar.gz',
                         weights_file=None,
                         overrides=""), 'coreference-resolution')

    def preprocess_IMDB(self, data):
        new_data = []
        for line in data:
            new_line = re.sub('<.*?>', ' ', line)  # remove HTML tags and replace with space
            sentences = sent_tokenize(new_line)  # sentence tokenization
            for sentence in sentences:
                new_sentence = re.sub('"', '', sentence)  # remove quotation marks
                new_sentence = unidecode.unidecode(new_sentence)  # removes accents and represents any unicode to closest ascii
                new_sentence = re.sub(r'\s+', ' ', new_sentence)  # Eliminate duplicate whitespaces
                # new_sentence = re.sub(r'[^\w\s]', '', new_sentence) # remove punctuation
                # new_sentence = new_sentence.lower() # convert to lower case
                wordcount = len(new_sentence.split())
                if new_sentence != '' and wordcount > 2: # wordcount of 2 or lower doesn't make sense for coref
                    new_data.append(new_sentence)
        return new_data

    # loads IMDB dataset and returns a dataframe that contains one review per line
    def load_IMDB(self):

        # read in input_file
        df = pd.read_csv(self.data_path, delimiter='\t', encoding='utf-8', header=None, squeeze=True)

        #info about the dataset
        print("Size of Dataset: ", len(df))
        #print("Columns :", df.columns)

        # cleaning sentences
        df_clean = self.preprocess_IMDB(df)
        print("Size of Dataset after clean: ", len(df_clean))

        return df_clean


# coref_json = {'top_spans': [[0, 0], [0, 8], [2, 5], [2, 8], [7, 8], [12, 12]], 'predicted_antecedents': [-1, -1, -1, -1, -1, 4], 'document': ['This', 'is', 'an', 'action', '-', 'melodrama', 'like', 'the', 'world', 'has', 'never', 'seen', 'it', 'before', '.'], 'clusters': [[[0, 0], [12, 12]]]}
    def to_file(self, data):
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
            if len(coref_json['clusters']) > 0 and self.filter(TreebankWordDetokenizer().detokenize(coref_json['document']),coref_json['clusters'],"all") == False:
                f.write(TreebankWordDetokenizer().detokenize(coref_json['document']) + "\n")

        f.close()
        print("write to file complete")



#Used temporarily for testing
if __name__ == '__main__':
    input_path = '../datasets/test_datasets/test_dataset.txt'
    output_name = "COREF2"

    dataloader = Dataloader(input_path, output_name, remove_sentence)
    data = dataloader.load_IMDB()
    dataloader.to_file(data)