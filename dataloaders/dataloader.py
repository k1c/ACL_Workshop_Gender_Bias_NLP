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


class Dataloader(object):
    def __init__(self, output_name, filter_by_corpus, encoding="utf-8"):
        self.output_name = output_name
        self.encoding = encoding
        self.filter_by_corpus = filter_by_corpus
        self.predictor = Predictor.from_archive(
            load_archive('https://s3-us-west-2.amazonaws.com/allennlp/models/coref-model-2018.02.05.tar.gz',
                         weights_file=None,
                         overrides=""), 'coreference-resolution')

    def filter_to_file(self, data):
        coref_true = []
        f = open(self.output_name, "w+")

        GENDER_PRONOUNS = ['he','she','him','her','his','hers','himself','herself']
        coref_output = []
        gp_output = []
        coref_range = []
        final_sentences = []
        tok_sent = []
        test_gp = []
        #coref_count = 0
        for line in tqdm(data):
            coref_line = {"document":line.strip()}
            try:
                coref_json = self.predictor.predict_json(coref_line)
            except KeyboardInterrupt:
                print("KeyboardInterrup")
                break
            except:
                print("problem sentence: ", line)

            coref_range.append(coref_json['clusters'])
            tok_sent.append(coref_json['document'])

            if len(coref_json['clusters']) > 0:
                coref_output.append(1) # coref cluster exists

            else:
                coref_output.append(0) # coref cluster does not exist

        for i in range(0, len(data)):
            if coref_output[i] == 1:
                for cluster in coref_range[i]:
                    test_gp = []
                    if any([((c[0] == c[1]) and (tok_sent[i][c[0]]).lower() in GENDER_PRONOUNS) for c in cluster]):
                        test_gp.append(True)

                    else:
                        test_gp.append(False) # gp pronoun exists
                if any(test_gp):
                    gp_output.append(1)
                else:
                    gp_output.append(0)
            else:
                gp_output.append(0) # coref cluster doesn't exists so don't look for gp pronoun

        assert (len(data) == len(coref_output) == len(gp_output) == len(coref_range)), "arrays not same size"
        print("gp array", gp_output)
        print("gp_output length", len(gp_output))
        #print(gp_output)
        pronoun_link = self.filter_by_corpus(data, coref_range, gp_output, "pro")
        human_name = self.filter_by_corpus(data, coref_range, gp_output, "name")
        gendered_term = self.filter_by_corpus(data, coref_range, gp_output, "term")
        final_candidates = self.filter_by_corpus(data,coref_range, gp_output,"all")

        assert (len(data) == len(human_name) == len(final_candidates) == len(gendered_term) == len(pronoun_link)), "arrays not same size"

        building_df = {'Sentences': data, 'Coreference': coref_output, 'Gender pronoun': gp_output, 'Gender link': pronoun_link,'Human Name': human_name,
                        'Gendered term': gendered_term, 'Final candidates': final_candidates}
        print("pronoun link", pronoun_link)
        print("human name", human_name)
        print("gendered term", gendered_term)
        print("final_candidates", final_candidates)
        plotting_df = pd.DataFrame(building_df)

        print(plotting_df.values)
        with open(self.output_name , "w+") as f:
            for j in plotting_df['Final candidates']:
                if j == 1:
                    f.write(plotting_df.loc[plotting_df['Final candidates'], 'Sentences'].iloc[j] + "\n")
            #for final_candidate in plotting_df['Final candidates']:
               # for j in range(0 , len(data)):
                    #if final_candidate == 1:
                        #f.write(data[j] + "\n")

        print("write to file complete")

        return plotting_df


#Used temporarily for testing
if __name__ == '__main__':
    input_path = '../datasets/test_datasets/test_dataset.txt'
    output_name = "winnerHOPE"

    dataloader = Dataloader(output_name, filter_by_corpus)
    data = load_general(input_path)
    df = dataloader.filter_to_file(data)
    #print(df)