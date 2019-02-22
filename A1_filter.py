import nltk
import pandas as pd
import pprint
pp = pprint.PrettyPrinter(indent=1)
#Use the NLTK Downloader to obtain the resources that you need for this script:
#nltk.download('punkt')
#nltk.download('averaged_perceptron_tagger')
#nltk.download('maxent_ne_chunker')
#nltk.download('words')
#nltk.download('tagsets')

#returns a dataframe
df = pd.read_json("./test_dataset_coref.json",lines=True)

print("AllenNLP Coreference column names: ",df.columns)
print(df.size)
print(df.shape)
#print(df.dtypes)

clusters = list(df["clusters"])

documents = list(df["document"])


#documents is a list of tokenized documents (uncomment below to see)
#print("Documents: ",documents)

print("Number of clusters: ",len(clusters))
print("Number of documents: ", len(documents))

print(len(clusters))
docs_coref = []
clusters_coref = []

#tuple (nltk_pos(document),cluster))
DETERM_LIST = ['A', 'a', 'all', 'All', 'Every', 'every', 'the']
#PRP$_LIST = ['her','his']
candidates = []
for i in range(len(clusters)):
    if(len(clusters[i])>0): #if cluster exists, this means coreference picked it up therefore there's a chance that its A1
        doc_pos_tagged = nltk.pos_tag(documents[i])
        for cluster in clusters[i]:
            if any( [ any(doc_pos_tagged[i][0] in DETERM_LIST for i in range(start_idx, end_idx+1)) for (start_idx, end_idx) in cluster] ) and \
                    any( [ any(doc_pos_tagged[i][1] == "PRP" or doc_pos_tagged[i][1] == "PRP$" for i in range(start_idx, end_idx+1)) for (start_idx, end_idx) in cluster]):
                candidates.append(doc_pos_tagged)


print("Candidates, (size: %d): " % (len(candidates)))
pp.pprint(candidates)


