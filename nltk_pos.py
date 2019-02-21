import nltk
import pandas as pd

#Use the NLTK Downloader to obtain the resources that you need for this script:
#nltk.download('punkt')
#nltk.download('averaged_perceptron_tagger')
#nltk.download('maxent_ne_chunker')
#nltk.download('words')
#nltk.download('tagsets')

#returns a data frame
df = pd.read_json("./test_dataset_coref.json",lines=True)

print(df.columns)
print(df.size)
print(df.shape)
#print(df.dtypes)

clusters = list(df["clusters"])

sentences = list(df["document"])


#sentences is a list of tokenized sentences (uncomment below to see)
#print("Sentences: ",sentences)

# print(sentences[0])
print(len(clusters))
print(len(sentences))


tagged = []

for i in range(len(clusters)):
    if(len(clusters[i])>0): #if cluster exists, this means coreference picked it up therefore there's a chance that its A1
        tagged.append(nltk.pos_tag(sentences[i]))

#tagged has been through coreference and has been tagged with POS
#tagged is a list of list of tupples
       #sentence1                                                   sentence2
#[[('word','tag'),('word','tag'),('word','tag')],[('word','tag'),('word','tag'),('word','tag'),('word','tag')]]
#print("Tagged: ",tagged)
entities = nltk.chunk.ne_chunk(tagged[0])
#print("Entities: ",entities)



#Name

""" 
Output of named-entity recognition on coreference. 
As you can see, don't pick up nouns like person as being people
ORGANIZATION Territory
GPE Pink
GPE Girls
GPE Protocol
GPE Antarctica
GPE New York
GPE Men
GPE Literature
PERSON Lev
PERSON Hershberg
ORGANIZATION Qatar Airways
PERSON Baker
ORGANIZATION Position
GPE Men
GPE Men
GPE Men
"""

#use this if tokenized and tagged already
#tagged:[[('A', 'DT'), ('person', 'NN'), ('must', 'MD'), ('reside', 'VB'), ('continuously', 'RB'), ('in', 'IN'), ('the', 'DT'), ('Territory', 'NNP'), ('for', 'IN'), ('20', 'CD'), ('years', 'NNS'), ('before', 'IN'), ('she', 'PRP'), ('may', 'MD'), ('apply', 'VB'), ('for', 'IN'), ('permanent', 'JJ'), ('residence', 'NN'), ('.', '.')], [('A', 'DT'), ('staff', 'NN'), ('member', 'NN'), ('in', 'IN'), ('Antarctica', 'NNP'), ('earns', 'VBZ'), ('less', 'JJR'), ('than', 'IN'), ('he', 'PRP'), ('would', 'MD'), ('in', 'IN'), ('New', 'NNP'), ('York', 'NNP'), ('.', '.')], [('A', 'DT'), ('bedazzled', 'JJ'), ('ninja', 'JJ'), ('turtle', 'NN'), ('or', 'CC'), ('a', 'DT'), ('feature', 'NN'), ('film', 'NN'), ('about', 'IN'), ('a', 'DT'), ('peasant', 'JJ'), ('boy', 'NN'), ('who', 'WP'), ('falls', 'VBZ'), ('hopelessly', 'RB'), ('in', 'IN'), ('love', 'NN'), ('with', 'IN'), ('princess', 'NN'), ('would', 'MD'), ('help', 'VB'), ('all', 'DT'), ('children', 'NNS'), ('feel', 'VBP'), ('more', 'RBR'), ('emboldened', 'VBN'), ('by', 'IN'), ('their', 'PRP$'), ('girlier', 'NN'), ('proclivities', 'NNS'), ('.', '.')], [('In', 'IN'), ('general', 'JJ'), (',', ','), ('a', 'DT'), ('nanny', 'NN'), ('is', 'VBZ'), ('concerned', 'VBN'), ('about', 'IN'), ('her', 'PRP$'), ('reputation', 'NN'), ('amongst', 'NN'), ('parents', 'NNS'), ('.', '.')], [('This', 'DT'), ('is', 'VBZ'), ('also', 'RB'), ('why', 'WRB'), ('you', 'PRP'), ('should', 'MD'), ('never', 'RB'), (',', ','), ('ever', 'RB'), (',', ','), ('ever', 'RB'), ('hit', 'VBN'), ('on', 'IN'), ('a', 'DT'), ('woman', 'NN'), ('while', 'IN'), ('she', 'PRP'), ('’s', 'VBZ'), ('at', 'IN'), ('work', 'NN'), ('.', '.')], [('A', 'DT'), ('substitute', 'NN'), ('judge', 'NN'), ('must', 'MD'), ('certify', 'VB'), ('that', 'IN'), ('he', 'PRP'), ('has', 'VBZ'), ('familiarized', 'VBN'), ('himself', 'PRP'), ('with', 'IN'), ('the', 'DT'), ('record', 'NN'), ('of', 'IN'), ('the', 'DT'), ('proceedings', 'NNS'), ('.', '.')], [('When', 'WRB'), ('a', 'DT'), ('student', 'NN'), ('does', 'VBZ'), ('not', 'RB'), ('plan', 'VB'), ('ahead', 'RB'), ('effectively', 'RB'), (',', ','), ('he', 'PRP'), ('may', 'MD'), ('face', 'VB'), ('problems', 'NNS'), ('with', 'IN'), ('time', 'NN'), ('management', 'NN'), ('that', 'WDT'), ('will', 'MD'), ('be', 'VB'), ('reflected', 'VBN'), ('in', 'IN'), ('his', 'PRP$'), ('academic', 'JJ'), ('work', 'NN'), ('.', '.')], [('No', 'DT'), ('man', 'NN'), ('succeeds', 'VBZ'), ('without', 'IN'), ('a', 'DT'), ('good', 'JJ'), ('woman', 'NN'), ('besides', 'IN'), ('him', 'PRP'), ('.', '.'), ('Wife', "''"), ('or', 'CC'), ('mother', 'NN'), ('.', '.'), ('If', 'IN'), ('it', 'PRP'), ('is', 'VBZ'), ('both', 'DT'), (',', ','), ('he', 'PRP'), ('is', 'VBZ'), ('twice', 'RB'), ('as', 'RB'), ('blessed', 'JJ')], [('Ask', 'VB'), ('the', 'DT'), ('student', 'NN'), ('whether', 'IN'), ('he', 'PRP'), ('is', 'VBZ'), ('prepared', 'JJ'), ('to', 'TO'), ('give', 'VB'), ('a', 'DT'), ('presentation', 'NN'), ('.', '.')], [('Women', 'NNS'), ('do', 'VBP'), ("n't", 'RB'), ('want', 'VB'), ('perfect', 'JJ'), ('partners', 'NNS'), (';', ':'), ('they', 'PRP'), ('want', 'VBP'), ('men', 'NNS'), ('who', 'WP'), ('are', 'VBP'), ('striving', 'VBG'), ('to', 'TO'), ('be', 'VB'), ('their', 'PRP$'), ('best', 'JJS'), ('selves', 'NNS'), ('.', '.')], [('A', 'DT'), ('good', 'JJ'), ('woman', 'NN'), ('understands', 'VBZ'), ('that', 'IN'), ('a', 'DT'), ('man', 'NN'), ('does', 'VBZ'), ('n’t', 'RB'), ('have', 'VB'), ('to', 'TO'), ('be', 'VB'), ('all', 'DT'), ('up', 'RP'), ('under', 'IN'), ('you', 'PRP'), ('to', 'TO'), ('be', 'VB'), ('in', 'IN'), ('love', 'NN'), ('with', 'IN'), ('you', 'PRP'), ('.', '.')], [('She', 'PRP'), ('is', 'VBZ'), ('a', 'DT'), ('girl', 'NN'), (',', ','), ('she', 'PRP'), ('ca', 'MD'), ("n't", 'RB'), ('be', 'VB'), ('an', 'DT'), ('engineer', 'NN'), ('!', '.')], [('As', 'IN'), ('a', 'DT'), ('rule', 'NN'), (',', ','), ('the', 'DT'), ('warrior', 'NN'), ('who', 'WP'), ('inspired', 'VBD'), ('the', 'DT'), ('greatest', 'JJS'), ('terror', 'NN'), ('in', 'IN'), ('the', 'DT'), ('hearts', 'NNS'), ('of', 'IN'), ('his', 'PRP$'), ('enemies', 'NNS'), ('was', 'VBD'), ('a', 'DT'), ('man', 'NN'), ('of', 'IN'), ('the', 'DT'), ('most', 'RBS'), ('exemplary', 'JJ'), ('gentleness', 'NN'), (',', ','), ('and', 'CC'), ('almost', 'RB'), ('feminine', 'JJ'), ('refinement', 'NN'), (',', ','), ('among', 'IN'), ('his', 'PRP$'), ('family', 'NN'), ('and', 'CC'), ('friends', 'NNS'), ('.', '.')], [('When', 'WRB'), ('women', 'NNS'), ('demand', 'VBP'), ('more', 'JJR'), ('intensity', 'NN'), ('than', 'IN'), ('her', 'PRP$'), ('man', 'NN'), ('can', 'MD'), ('comfortably', 'RB'), ('offer', 'VB'), (',', ','), ('he', 'PRP'), ('withdraws', 'VBZ'), ('.', '.')], [('Remind', 'VB'), ('your', 'PRP$'), ('partner', 'NN'), ('that', 'IN'), ('you', 'PRP'), ('love', 'VBP'), ('her', 'PRP'), ('.', '.')], [('Do', 'VBP'), ("n't", 'RB'), ('listen', 'VB'), ('to', 'TO'), ('her', 'PRP$'), ('she', 'PRP'), ('is', 'VBZ'), ('probably', 'RB'), ('on', 'IN'), ('her', 'PRP'), ('period', 'NN')], [('Women', 'NNS'), ('are', 'VBP'), ('quiet', 'JJ'), (',', ','), ('they', 'PRP'), ('should', 'MD'), ('not', 'RB'), ('speak', 'VB'), ('up', 'RP'), ('and', 'CC'), ('argue', 'VB')], [('And', 'CC'), ('we', 'PRP'), ('want', 'VBP'), ('our', 'PRP$'), ('girls', 'NNS'), ('to', 'TO'), ('be', 'VB'), ('more', 'RBR'), ('like', 'IN'), ('boys', 'NNS'), ('for', 'IN'), ('the', 'DT'), ('same', 'JJ'), ('reason', 'NN'), ('.', '.')], [('Lev', 'NNP'), ('Hershberg', 'NNP'), ('says', 'VBZ'), ('that', 'IN'), ('if', 'IN'), ('he', 'PRP'), ('were', 'VBD'), ('a', 'DT'), ('girl', 'NN'), (',', ','), ('he', 'PRP'), ('would', 'MD'), ("n't", 'RB'), ('like', 'VB'), ('computers', 'NNS'), ('.', '.')], [('Men', 'NN'), (',', ','), ('more', 'RBR'), ('often', 'RB'), ('than', 'IN'), ('not', 'RB'), (',', ','), ('connect', 'VBP'), ('through', 'IN'), ('indicators', 'NNS'), ('of', 'IN'), ('sexual', 'JJ'), ('access', 'NN'), ('just', 'RB'), ('as', 'RB'), ('much', 'JJ'), ('as', 'IN'), ('they', 'PRP'), ('do', 'VBP'), ('through', 'IN'), ('sex', 'NN'), ('.', '.')], [('Men', 'NN'), ('can', 'MD'), ('sometimes', 'RB'), ('view', 'VB'), ('unsolicited', 'JJ'), ('assistance', 'NN'), ('as', 'IN'), ('an', 'DT'), ('undermining', 'NN'), ('of', 'IN'), ('their', 'PRP$'), ('effort', 'NN'), ('to', 'TO'), ('solve', 'VB'), ('problems', 'NNS'), ('alone', 'RB'), ('while', 'IN'), ('women', 'NNS'), ('value', 'NN'), ('assistance', 'NN'), (',', ','), ('and', 'CC'), ('thus', 'RB'), ('view', 'NN'), ('unsolicited', 'JJ'), ('solutions', 'NNS'), ('as', 'IN'), ('undermining', 'VBG'), ('their', 'PRP$'), ('effort', 'NN'), ('to', 'TO'), ('proceed', 'VB'), ('interactively', 'RB'), ('.', '.')], [('Men', 'NN'), (',', ','), ('more', 'RBR'), ('often', 'RB'), ('than', 'IN'), ('not', 'RB'), (',', ','), ('connect', 'VBP'), ('through', 'IN'), ('indicators', 'NNS'), ('of', 'IN'), ('sexual', 'JJ'), ('access', 'NN'), ('just', 'RB'), ('as', 'RB'), ('much', 'JJ'), ('as', 'IN'), ('they', 'PRP'), ('do', 'VBP'), ('through', 'IN'), ('sex', 'NN'), ('.', '.')]]

for i in range(len(tagged)):
    for chunk in nltk.ne_chunk(tagged[i]):
        if hasattr(chunk, 'label'):
            print(chunk.label(), ' '.join(c[0] for c in chunk))


#use this if your sentence is tokenized already
#Sentences:  [['A', 'person', 'must', 'reside', 'continuously', 'in', 'the', 'Territory', 'for', '20', 'years', 'before', 'she', 'may', 'apply', 'for', 'permanent', 'residence', '.'], ['Pink', 'is', 'a', 'girl', "'s", 'color', '.'], ['Girls', 'love', 'shopping', '.'], ['Every', 'Permanent', 'Representative', 'must', 'submit', 'his', 'credentials', 'to', 'Protocol', '.'], ['A', 'staff', 'member', 'in', 'Antarctica', 'earns', 'less', 'than', 'he', 'would', 'in', 'New', 'York', '.'], ['If', 'a', 'complainant', 'is', 'not', 'satisfied', 'with', 'the', 'board', '’s', 'decision', ',', 'he', 'can', 'ask', 'for', 'a', 'rehearing', '.'], ['A', 'bedazzled', 'ninja', 'turtle', 'or', 'a', 'feature', 'film', 'about', 'a', 'peasant', 'boy', 'who', 'falls', 'hopelessly', 'in', 'love', 'with', 'princess', 'would', 'help', 'all', 'children', 'feel', 'more', 'emboldened', 'by', 'their', 'girlier', 'proclivities', '.'], ['In', 'general', ',', 'a', 'nanny', 'is', 'concerned', 'about', 'her', 'reputation', 'amongst', 'parents', '.'], ['This', 'is', 'also', 'why', 'you', 'should', 'never', ',', 'ever', ',', 'ever', 'hit', 'on', 'a', 'woman', 'while', 'she', '’s', 'at', 'work', '.'], ['A', 'substitute', 'judge', 'must', 'certify', 'that', 'he', 'has', 'familiarized', 'himself', 'with', 'the', 'record', 'of', 'the', 'proceedings', '.'], ['When', 'a', 'student', 'does', 'not', 'plan', 'ahead', 'effectively', ',', 'he', 'may', 'face', 'problems', 'with', 'time', 'management', 'that', 'will', 'be', 'reflected', 'in', 'his', 'academic', 'work', '.'], ['Boys', 'are', 'usually', 'aggressive', '.'], ['No', 'man', 'succeeds', 'without', 'a', 'good', 'woman', 'besides', 'him', '.', 'Wife', 'or', 'mother', '.', 'If', 'it', 'is', 'both', ',', 'he', 'is', 'twice', 'as', 'blessed'], ['Ask', 'the', 'student', 'whether', 'he', 'is', 'prepared', 'to', 'give', 'a', 'presentation', '.'], ['Women', 'do', "n't", 'want', 'perfect', 'partners', ';', 'they', 'want', 'men', 'who', 'are', 'striving', 'to', 'be', 'their', 'best', 'selves', '.'], ['A', 'good', 'woman', 'understands', 'that', 'a', 'man', 'does', 'n’t', 'have', 'to', 'be', 'all', 'up', 'under', 'you', 'to', 'be', 'in', 'love', 'with', 'you', '.'], ['Men', 'would', 'find', 'difficult', 'to', 'accept', 'a', 'young', ',', 'pretty', 'woman', 'as', 'an', 'authority', 'figure', '.'], ['She', 'is', 'a', 'girl', ',', 'she', 'ca', "n't", 'be', 'an', 'engineer', '!'], ['Literature', 'is', 'a', 'girls', 'major'], ['The', 'average', 'teenager', 'worries', 'about', 'his', 'physical', 'fitness', '.'], ['As', 'a', 'rule', ',', 'the', 'warrior', 'who', 'inspired', 'the', 'greatest', 'terror', 'in', 'the', 'hearts', 'of', 'his', 'enemies', 'was', 'a', 'man', 'of', 'the', 'most', 'exemplary', 'gentleness', ',', 'and', 'almost', 'feminine', 'refinement', ',', 'among', 'his', 'family', 'and', 'friends', '.'], ['When', 'women', 'demand', 'more', 'intensity', 'than', 'her', 'man', 'can', 'comfortably', 'offer', ',', 'he', 'withdraws', '.'], ['When', 'men', 'do', 'housework', ',', 'it', 'is', 'considered', 'a', 'nice', 'favor', ',', 'something', 'to', 'be', 'actively', 'appreciated', '.'], ['Remind', 'your', 'partner', 'that', 'you', 'love', 'her', '.'], ['All', 'men', 'enjoy', 'working', 'on', 'cars', '.'], ['Do', "n't", 'listen', 'to', 'her', 'she', 'is', 'probably', 'on', 'her', 'period'], ['Computer', 'science', 'is', 'for', 'boys'], ['Women', 'are', 'quiet', ',', 'they', 'should', 'not', 'speak', 'up', 'and', 'argue'], ['But', 'to', 'have', 'a', 'friend', ',', 'and', 'to', 'be', 'true', 'under', 'any', 'and', 'all', 'trials', ',', 'is', 'the', 'mark', 'of', 'a', 'man', '!'], ['And', 'we', 'want', 'our', 'girls', 'to', 'be', 'more', 'like', 'boys', 'for', 'the', 'same', 'reason', '.'], ['Lev', 'Hershberg', 'says', 'that', 'if', 'he', 'were', 'a', 'girl', ',', 'he', 'would', "n't", 'like', 'computers', '.'], ['“', 'Well', ',', 'it', '’s', 'not', 'in', 'Qatar', 'Airways', ',', '”', 'Al', 'Baker', 'said', 'of', 'gender', 'inequality', 'in', 'aviation', ',', 'to', 'which', 'the', 'reporter', 'responded', ':', '“', 'Well', ',', 'certainly', 'it', '’s', 'being', 'led', 'by', 'a', 'man', '?'], ['But', 'the', 'last', 'line', 'reads', ':', '“', 'Please', 'note', 'that', 'the', 'Position', 'requires', 'filling', 'in', 'the', 'responsibilities', 'of', 'a', 'receptionist', ',', 'so', 'female', 'candidates', 'are', 'preferred', '”', '.'], ['Even', 'if', 'it', 'is', 'true', 'that', 'men', 'have', 'always', 'dominated', ',', 'husbands', 'have', 'always', 'ruled', 'the', 'home', ',', 'what', 'has', 'been', 'forever', 'is', 'not', 'what', 'can', 'persist', 'forever', '.'], ['Men', 'have', 'infamously', 'tender', 'egos', '.'], ['Men', ',', 'more', 'often', 'than', 'not', ',', 'connect', 'through', 'indicators', 'of', 'sexual', 'access', 'just', 'as', 'much', 'as', 'they', 'do', 'through', 'sex', '.'], ['While', 'women', 'connect', 'better', 'through', 'the', 'act', 'of', 'communication', ',', 'men', 'are', 'known', 'to', 'connect', 'better', 'through', 'the', 'act', 'of', 'physical', 'intimacy', '.'], ['Men', 'can', 'sometimes', 'view', 'unsolicited', 'assistance', 'as', 'an', 'undermining', 'of', 'their', 'effort', 'to', 'solve', 'problems', 'alone', 'while', 'women', 'value', 'assistance', ',', 'and', 'thus', 'view', 'unsolicited', 'solutions', 'as', 'undermining', 'their', 'effort', 'to', 'proceed', 'interactively', '.'], ['Men', 'have', 'infamously', 'tender', 'egos', '.'], ['Men', ',', 'more', 'often', 'than', 'not', ',', 'connect', 'through', 'indicators', 'of', 'sexual', 'access', 'just', 'as', 'much', 'as', 'they', 'do', 'through', 'sex', '.'], ['This', 'lack', 'of', 'awareness', 'around', 'women', 'needing', 'to', 'connect', 'through', 'words', 'and', 'men', 'needing', 'to', 'connect', 'through', 'sex', 'can', 'sometimes', 'turn', 'into', 'an', 'unfortunate', 'and', 'rapid', 'downward', 'spiral', '.'], ['Obviously', ',', 'men', 'do', 'care', 'but', 'fail', 'to', 'show', '.']]

# for sent in sentences:
#    for chunk in nltk.ne_chunk(nltk.pos_tag(sent)):
#       if hasattr(chunk, 'label'):
#          print(chunk.label(), ' '.join(c[0] for c in chunk))


#use this if your sentence is not tokenized

# for sent in nltk.sent_tokenize(sentence):
#    for chunk in nltk.ne_chunk(nltk.pos_tag(nltk.word_tokenize(sent))):
#       if hasattr(chunk, 'label'):
#          print(chunk.label(), ' '.join(c[0] for c in chunk))




#print(tagged)

#nltk.download('brown')
#nltk.download('universal_tagset')
from nltk.corpus import brown
#brown_news_tagged = brown.tagged_words(categories='news', tagset='universal')



# brown_news_tagged = tagged[0][0].tagged_words(tagset='universal')
tag_fd = nltk.FreqDist(tagged[0])


#print([tag for (tag,_) in pairs.most_common()])

# tagged_fd = nltk.FreqDist(tagged[0])
# print(sentences[0])
# print(tagged_fd.most_common())

count2=0
for cluster in clusters:
    if(cluster):
        count2+=1
        #nltk.pos_tag(sentences[0])


# test = df.select_dtypes([clusters])
# print(test)

#print(df.head(10))

#print(df.columns)

# for cluster in df.clusters:
#    if (cluster): #(len(cluster) > 0)
#        print(cluster)


# for key in df.keys():
#     print(key)

# for key in df.values():
#     print(key)


# for antecedent_indices,clusters,document,predicted_antecendents,top_spans in df.items():
#     if (clusters): #(len(cluster) > 0)
#         print(clusters)
#         pass

#for k,v in df:
  #  print(k)
    #print(v)

#Tokenize and tag
tokens = nltk.word_tokenize("""At eight o'clock on Thursday morning
... Arthur didn't feel very good.""")

#print(tokens)

tagged = nltk.pos_tag(tokens)

#print(tagged)

# print(tagged)
# print(core_index[0])

#Identify named entities

# entities = nltk.chunk.ne_chunk(tagged)
#print(entities)

#All possible POS tags of NLTK
#nltk.help.upenn_tagset()

