'''
Purpose: General Purpose dataloader that passes text data through allen-nlp coref and also cleans the text data
Usage: python web_dataloader.py input_dataset_path output_dataset_name

Input: dataset.tsv (one sentence per line)
Output: coref results from allen-nlp (one json per line)

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
    def __init__(self, data_path, output_name, encoding="utf-8"):
        self.data_path = data_path
        self.output_name = output_name
        self.encoding = encoding

    def preprocess(self, sentences):
        new_data = []
        for sentence in sentences:
            new_sentence = re.sub('"', '', sentence)  # remove quotation marks
            new_sentence = unidecode.unidecode(new_sentence)  # removes accents and represents any unicode to closest ascii
            new_sentence = re.sub('<.*?>', '', new_sentence)  # remove HTML tags (in case, should be clean)
            new_sentence = re.sub(r'[^\w\s]', '', new_sentence) # remove punctuation
            # new_sentence = new_sentence.lower() # convert to lower case
            wordcount = len(new_sentence.split())
            if new_sentence != '' and wordcount > 2: # wordcount of 2 or lower doesn't make sense for coref
                new_data.append(new_sentence)
        return new_data

    def loader(self):

        # Allen NLP coref settings using pre-trained model
        coref_archive = load_archive('https://s3-us-west-2.amazonaws.com/allennlp/models/coref-model-2018.02.05.tar.gz',
                               weights_file=None,
                               overrides="")

        predictor = Predictor.from_archive(coref_archive, 'coreference-resolution')

        # read in input_file
        df = pd.read_csv(self.data_path, delimiter='\t', encoding='utf-8', header=None, squeeze=True)
        print("Size of Dataset: ", len(df))

        # cleaning sentences
        df_clean = self.preprocess(df)
        print("Size of Dataset after clean: ", len(df_clean))

        # get the coref results from allen-nlp
        coref_results = []
        for line in df_clean:
            coref_line = {"document":line.strip()}
            try:
                coref_results.append(predictor.predict_json(coref_line))
            except:
                print("problem sentence: ", line)

        # write the coref results to file
        f = open(self.output_name + ".json", "w+")
        for i in range(len(coref_results)):
           string_output = predictor.dump_line(coref_results[i])
           f.write(string_output)
        f.close()

        print("Size of Dataset after coref: ", len(coref_results))

#Used temporarily for testing
if __name__ == '__main__':
    input_path = '../datasets/test_datasets/biasly_data_clean3.tsv'
    output_name = "biasly_data_testing123"
    web_data = WebDataset(input_path, output_name)
    web_data.loader()