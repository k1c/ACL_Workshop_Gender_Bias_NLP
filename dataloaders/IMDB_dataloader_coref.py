'''
Purpose:    Dataloader that takes IMDB data and passes it through allen-nlp coref or A1 filter and writes the results to file
            There's also a function called preprocess() that cleans sentences

usage example:
    input_path = '../datasets/test_datasets/IMDB-train.txt'
    output_name = "IMDB"

    dataloader = Dataloader(input_path, output_name)
    data = dataloader.load_IMDB()
    dataloader.coref_true_to_file(data)
'''

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
pp = pprint.PrettyPrinter(indent=1)


#Use the NLTK Downloader to obtain the resources that you need for this script:
nltk.download('averaged_perceptron_tagger')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('maxent_ne_chunker')
nltk.download('words')
nltk.download('tagsets')


class Dataloader(object):
    def __init__(self, data_path, output_name, encoding="utf-8"):
        self.data_path = data_path
        self.output_name = output_name
        self.encoding = encoding
        self.predictor = Predictor.from_archive(
            load_archive('https://s3-us-west-2.amazonaws.com/allennlp/models/coref-model-2018.02.05.tar.gz',
                         weights_file=None,
                         overrides=""), 'coreference-resolution')

    def preprocess(self, data):
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

# load general purpose text dataset that has one data text per line
    def load_general(self):

        # read in input_file
        df = pd.read_csv(self.data_path, delimiter='\t', encoding='utf-8', header=None, squeeze=True)

        #info about the dataset
        print("Size of Dataset: ", len(df))

        # cleaning sentences
        df_clean = self.preprocess(df)
        print("Size of Dataset after clean: ", len(df_clean))

        return df_clean

# loads IMDB dataset and returns a dataframe that contains one review per line
    def load_IMDB(self):

        # read in input_file
        df = pd.read_csv(self.data_path, delimiter='\t', encoding='utf-8', header=None, squeeze=True)

        #info about the dataset
        print("Size of Dataset: ", len(df))
        print("Columns :", df.columns)

        # cleaning sentences
        df_clean = self.preprocess(df[0])
        print("Size of Dataset after clean: ", len(df_clean))

        return df_clean

#Purpose: writes text data that contains coref-resolution in them to file (one per line)
    # data input should be in the form of one text data per line, where text data can be sentence or paragraph

    def coref_true_to_file(self, data):
        # write the coref results to file
        f = open(self.output_name + "_coref_true.tsv", "w+")
        for line in tqdm(data):
            coref_line = {"document":line.strip()}
            try:
                json = self.predictor.predict_json(coref_line)
            except KeyboardInterrupt:
                print("KeyboardInterrup")
                break
            except:
                print("problem sentence: ", line)
            if len(json['clusters']) > 0:
                f.write(TreebankWordDetokenizer().detokenize(json['document'])+"\n")
        f.close()

        print("write to file complete")

# Purpose: writes the sentences/paragraphs that have passed the A1 filter to file (one sentence per line)
    def A1_filter_to_file(self, data):
        pass

#Used temporarily for testing
if __name__ == '__main__':
    input_path_imdb_train = '../datasets/IMDB/IMDB-train.txt'
    output_name_train = "IMDB-train"

    dataloader = Dataloader(input_path_imdb_train, output_name_train)
    data_imdb_train = dataloader.load_IMDB()
    dataloader.coref_true_to_file(data_imdb_train)

    # input_path_imdb_train_subset = '../datasets/IMDB/imbd_subset.txt'
    # output_name_train_subset = "imbd_subset_new_new_new"
    #
    # dataloader = Dataloader(input_path_imdb_train_subset, output_name_train_subset)
    # data_imdb_train_subset = dataloader.load_general()
    # dataloader.coref_true_to_file(data_imdb_train_subset)

