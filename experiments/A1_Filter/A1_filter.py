import nltk
import pandas as pd
import json
from nltk.tokenize.treebank import TreebankWordDetokenizer
import pprint
pp = pprint.PrettyPrinter(indent=1)
#Use the NLTK Downloader to obtain the resources that you need for this script:
#nltk.download('punkt')
#nltk.download('averaged_perceptron_tagger')
#nltk.download('maxent_ne_chunker')
#nltk.download('words')
#nltk.download('tagsets')

#returns a dataframe
df = pd.read_json("../datasets/test_datasets/test_dataset_coref.json",lines=True,encoding='utf-8')

print("AllenNLP Coreference column names: ", df.columns)
print(df.size)
print(df.shape)
#print(df.dtypes)

clusters = list(df["clusters"])

documents = list(df["document"])


#documents is a list of tokenized documents (uncomment below to see)
#print("Documents: ",documents)

print("Number of clusters: ",len(clusters))
print("Number of documents: ", len(documents))

docs_coref = []
clusters_coref = []

DETERM_LIST = ['A', 'a', 'all', 'All', 'Every', 'every', 'the']
PRPS_LIST = ['her', 'his']
A1_candidates = []
A1_candidates_pos = []
for i in range(len(clusters)):
    if(len(clusters[i])>0): #if cluster exists, this means coreference picked it up therefore there's a chance that its A1
        doc_pos_tagged = nltk.pos_tag(documents[i])
        for cluster in clusters[i]:
            if any( [ any(doc_pos_tagged[i][0] in DETERM_LIST for i in range(start_idx, end_idx+1)) for (start_idx, end_idx) in cluster] ) and \
                    any( [ any(doc_pos_tagged[i][1] == "PRP" or doc_pos_tagged[i][0] in PRPS_LIST for i in range(start_idx, end_idx+1)) for (start_idx, end_idx) in cluster]):
                A1_candidates_pos.append(doc_pos_tagged)
                A1_candidates.append(documents[i])

print("Part-of-Speech of full sentence that passes coref, (size: %d): " % (len(doc_pos_tagged)))
print("A1 Candidates, (size: %d): " % (len(A1_candidates)))
#pp.pprint(A1_candidates)
#print("\nConclusion: 42 sentences in total, 22 get tagged by coref. Should have 9 A1, 3 didn't get picked up by coref, and there's 1 too many ([a woman],[she]), for a total of 7.")

# check duplicates
f = open("A1_candidates_cleaner","w+")
for i in range(len(A1_candidates)):
    sentence = TreebankWordDetokenizer().detokenize(A1_candidates[i])
    #sentence = sentence.encode('ascii', 'ignore').decode('ascii') #silently dropping unicode, workaround until our dataset is clean
    f.write(sentence+"\n")
f.close()