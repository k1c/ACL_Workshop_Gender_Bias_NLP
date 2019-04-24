'''
Purpose:    Dataloader that takes IMDB data and passes it through allen-nlp coref or A1 filter and writes the results to file
            There's also a function called preprocess() that cleans sentences.

Filters:    There are 4 filters and each fitler is embedded into one another in order. For example in the coref filter
            all sentences go through coreference. In the GP filter, all sentences do coreference and then GP filter. GP Link
            is the coref, GP presence and linkage of potential coreference clusters which may only have gendered pronouns.


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
from nltk import ne_chunk, pos_tag
from nltk.tokenize import sent_tokenize, word_tokenize
pp = pprint.PrettyPrinter(indent=1)


#Use the NLTK Downloader to obtain the resources that you need for this script:
# nltk.download('averaged_perceptron_tagger')
# nltk.download('punkt')
# nltk.download('averaged_perceptron_tagger')
# nltk.download('maxent_ne_chunker')
# nltk.download('words')
# nltk.download('tagsets')


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
        clean_sentences = []
        with open(data) as f:
            text = f.read()
        text = re.sub(r'(M\w{1,2})\.', r'\1', text)  # removes the '.' in Mr. and Mrs.
        sentences = sent_tokenize(text)
        # sentences = re.split(r' *[\.\?!][\'"\)\]]* *', text)
        new_sentences = [re.sub(r'\n+', ' ', s) for s in sentences]
        #clean_sentences.append(new_sentences)

        for sentence in new_sentences:
            wordcount = len(sentence.split())
            if sentence != '' and wordcount > 2: # wordcount of 2 or lower doesn't make sense for coref
                clean_sentences.append(sentence)
        return clean_sentences


    def load_gutenberg(self):
        # read in input_file
        df = self.data_path
        #info about the dataset
        print("Size of Dataset: ", len(df))
        # cleaning sentences
        df_clean = self.preprocess(df)
        print("Size of Dataset after clean: ", len(df_clean))

        smalldf = df_clean[0:200]  # subset
        print("size of small subset for testing: ", len(smalldf))
        return smalldf

#Purpose: writes text data that contains coref-resolution in them to file (one per line)
    # data input should be in the form of one text data per line, where text data can be sentence or paragraph

    def coref_true_to_file(self, data):
        # write the coref results to file
        corefCount = 0
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
                corefCount += 1
                f.write(TreebankWordDetokenizer().detokenize(json['document'])+"\n")
        f.close()
        print("Coref count: ", corefCount)
        print("write to file complete")

# Purpose: writes text data that contains coref-resolution in them and have passed the Gender Pronoun filter to file (one sentence per line)
    # data input should be in the form of one text data per line, where text data can be sentence or paragraph
    def GP_filter_to_file(self, data):
        # write the coref results to file
        GENDER_PRONOUNS = ['he','she','him','her','his','hers','himself','herself','He','She','Him','Her','His','Hers','Himself','Herself']
        corefCount = 0
        gpCount = 0
        candidates = set()

        for line in tqdm(data):
            coref_line = {"document":line.strip()}
            try:
                json = self.predictor.predict_json(coref_line)
            except KeyboardInterrupt:
                print("KeyboardInterrupt")
                break
            except:
                print("problem sentence: ", line)

            if len(json['clusters']) > 0:
                corefCount += 1
                for clusters in json['clusters']:
                    if any([ any(json['document'][i] in GENDER_PRONOUNS for i in range(start_idx, end_idx+1)) for (start_idx, end_idx) in clusters ]):
                        gpCount += 1
                        candidates.add(TreebankWordDetokenizer().detokenize(json['document']))

        with open(self.output_name + "_GP_filter.tsv", "w+") as f:
            for line in candidates:
                f.write(line + "\n")

        print("Coref count: ", corefCount)
        print("GP Filter count: ", gpCount)
        print("write to file complete")

# Purpose: writes text data that contains coref-resolution, Gender pronouns and have passed the Gender Pronoun link filter to file (one sentence per line)
# data input should be in the form of one text data per line, where text data can be sentence or paragraph
    def GPLink_filter(self, data):
        # write the coref results to file
        GENDER_PRONOUNS = ['he','she','him','her','his','hers','himself','herself','He','She','Him','Her','His','Hers','Himself','Herself']
        corefCount = 0
        gpCount = 0
        gpLink = 0
        gpNoLink = 0
        candidates = set()

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
                corefCount += 1
                for clusters in json['clusters']:
                    if any([ any(json['document'][i] in GENDER_PRONOUNS for i in range(start_idx, end_idx+1)) for (start_idx, end_idx) in clusters ]): # Identify if GP is in a cluster
                        gpCount += 1
                        for cluster in ['document']:
                            if any([any(json['document'][i] in GENDER_PRONOUNS for i in ([((c[0] == c[1]) and (c[0]) in GENDER_PRONOUNS) for c in cluster]))]): # indentify the GP clustered with another GP
                                gpLink += 1
                            else:                            # GP with no link store accordingly
                                gpNoLink += 1
                                candidates.add(TreebankWordDetokenizer().detokenize(json['document']))

        with open(self.output_name + "_GPLink_filter.tsv", "w+") as f:
            for line in candidates:
                f.write(line + "\n")

        print("Coref count: ", corefCount)
        print("Coref and GP filter", gpCount)
        print("Coref + GP + GP Link Filter + GP Link Filter count: ", gpLink)
        print("Coref with GP but with no GP to GP link: ", gpNoLink)
        print("write to file complete")

# Purpose: writes text data that contains coref-resolution, Gender pronouns, Gender Pronoun link filter and the Human name filter to file (one sentence per line)
# data input should be in the form of one text data per line, where text data can be sentence or paragraph
    def HumanName_filter(self, data):
        # write the coref results to file
        GENDER_PRONOUNS = ['he','she','him','her','his','hers','himself','herself','He','She','Him','Her','His','Hers','Himself','Herself']
        corefCount = 0
        gpCount = 0
        gpLink = 0
        gpNoLink = 0
        human = 0
        nohuman = 0
        namelist = []
        candidates = set()

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
                corefCount += 1
                for clusters in json['clusters']:
                    if any([ any(json['document'][i] in GENDER_PRONOUNS for i in range(start_idx, end_idx+1)) for (start_idx, end_idx) in clusters ]): # Identify if GP is in a cluster
                        gpCount += 1
                        for cluster in ['document']:
                            if any([any(json['document'][c] in GENDER_PRONOUNS for c in ([((c[0] == c[1]) and (c[0]) in GENDER_PRONOUNS) for c in clusters]))]): # indentify the GP clustered with another GP
                                gpLink += 1
                            else:                            # GP with no link store accordingly
                                gpNoLink += 1
                                candidates.add(TreebankWordDetokenizer().detokenize(json['document']))
                                for sent in ne_chunk(pos_tag(json['document'])):
                                    if hasattr(sent, 'label'):
                                        if sent.label() == "PERSON":
                                            namelist.append(' '.join(c[0] for c in sent))

                                for cluster in ['document']:
                                    if any(' '.join(w for w in json['document'][c[0]:c[1]+1]) in namelist for c in clusters):
                                        human += 1
                                    else:
                                        nohuman += 1
                                        candidates.add(TreebankWordDetokenizer().detokenize(json['document']))

        with open(self.output_name + "_HumanName_filter.tsv", "w+") as f:
            for line in candidates:
                f.write(line + "\n")

        print("Coref count: ", corefCount)
        print("Coref and GP filter", gpCount)
        print("Coref + GP + GP Link Filter count: ", gpLink)
        print("Coref with GP but with no GP to GP link: ", gpNoLink)
        print("Coref + with GP and GP Link + Human name filter count", human)
        print("Coref + with GP + no GP Link + no Human name", nohuman)
        print("write to file complete")


#Used temporarily for testing
if __name__ == '__main__':
    input_path = '../datasets/test_datasets/pg6167.txt'
    output_name= "businesshints"

    dataloader = Dataloader(input_path, output_name)
    data_business = dataloader.load_gutenberg()
    #dataloader.coref_true_to_file(data_imdb_train)
    #dataloader.GP_filter_to_file(data_imdb_train)
    dataloader.HumanName_filter(data_business)
    #dataloader.HumanName_filter(data_imdb_train)

    # input_path_imdb_train_subset = '../datasets/IMDB/imbd_subset.txt'
    # output_name_train_subset = "imbd_subset_new_new_new"
    #
    # dataloader = Dataloader(input_path_imdb_train_subset, output_name_train_subset)
    # data_imdb_train_subset = dataloader.load_general()
    # dataloader.coref_true_to_file(data_imdb_train_subset)