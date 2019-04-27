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
from filter import check_remove, filter_by_corpus
from loaders import load_IMDB, load_gutenberg, load_general

#Use the NLTK Downloader to obtain the resources that you need for this script:
# nltk.download('averaged_perceptron_tagger')
# nltk.download('punkt')
# nltk.download('averaged_perceptron_tagger')
# nltk.download('maxent_ne_chunker')
# nltk.download('words')
# nltk.download('tagsets')

""" If you are using a gutenberg dataset, choose the load_gutenberg loader 
    If you are using IMDB dataset, choose the load_IMDB loader
    If you have a clean txt that is ready for testing, use the load_general loader
"""
class Dataloader(object):
    def __init__(self, output_candidates, filter_by_corpus, output_df, encoding="utf-8"):
        self.output_candidates = output_candidates
        self.encoding = encoding
        self.output_df = output_df
        self.filter_by_corpus = filter_by_corpus
        self.predictor = Predictor.from_archive(
            load_archive('https://s3-us-west-2.amazonaws.com/allennlp/models/coref-model-2018.02.05.tar.gz',
                weights_file=None,overrides=""), 'coreference-resolution')

    def filter_to_file(self, data):
        coref_true = []
        f = open(self.output_candidates, "w+")

        GENDER_PRONOUNS = ['he','she','him','her','his','hers','himself','herself']
        coref_output = []
        gp_output = []
        coref_range = []
        final_sentences = []
        tok_sent = []
        test_gp = []
        no_coref_output = []
        #coref_count = 0

        # check if a sentence has coref link
        for line in tqdm(data):
            coref_line = {"document":line.strip()}
            try:
                coref_json = self.predictor.predict_json(coref_line)
            except KeyboardInterrupt:
                print("KeyboardInterrup")
                break
            except:
                #print("problem sentence: ", line)
                no_coref_output.append(line)

            coref_range.append(coref_json['clusters'])
            tok_sent.append(coref_json['document'])

            if len(coref_json['clusters']) > 0:
                coref_output.append(1) # coref cluster exists

            else:
                coref_output.append(0) # coref cluster does not exist

        # check 
        for i in range(0, len(data)):
            if coref_output[i] == 1:
                for cluster in coref_range[i]:
                    test_gp = []
                    if any([((c[0] == c[1]) and (tok_sent[i][c[0]]).lower() in GENDER_PRONOUNS) for c in cluster]):
                        test_gp.append(True)
                        #gp_output.append(1)
                    else:
                        #gp_output.append(0)
                        test_gp.append(False) # gp pronoun exists
                       
                if any(test_gp):
                    gp_output.append(1)
                else:
                    gp_output.append(0)
                
            else:
                gp_output.append(0) # coref cluster doesn't exists so don't look for gp pronoun

        assert (len(data) == len(coref_output) == len(gp_output) == len(coref_range)), "DIM OF COREF & GP OUT NOT SAME"

        print("gp_output length", len(gp_output))

        pronoun_link = self.filter_by_corpus(data, coref_range, gp_output, "pro")
        human_name = self.filter_by_corpus(data, coref_range, gp_output, "name")
        gendered_term = self.filter_by_corpus(data, coref_range, gp_output, "term")
        final_candidates = self.filter_by_corpus(data,coref_range, gp_output,"all")

        assert (len(data) == len(human_name) == len(final_candidates) == len(gendered_term) == len(pronoun_link)),"DIM OF FILTER OUT NOT SAME"


        building_df = {'Sentences': data, 'Coreference': coref_output, 'Gender pronoun': gp_output, 'Gender link': pronoun_link,'Human Name': human_name,
                        'Gendered term': gendered_term, 'Final candidates': final_candidates}

        plotting_df = pd.DataFrame(building_df)

        #write to csv
        plotting_df[plotting_df['Final candidates'] == 1]['Sentences'].to_csv(self.output_candidates, header=False, index=None)
        plotting_df.to_csv(self.output_df, header=True, index=None )

        return plotting_df


#Used temporarily for testing
if __name__ == '__main__':
    input_path = '../datasets/gutenberg/pg6167.txt'  ## set up for gutenberg
    output_candidates = "winnerHOPE"
    output_df = "AllFilterData"

    dataloader = Dataloader(output_candidates, filter_by_corpus, output_df)
    data = load_gutenberg(input_path)
    dataloader.filter_to_file(data)
