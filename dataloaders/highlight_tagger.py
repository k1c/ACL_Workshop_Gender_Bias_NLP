"""
    Input: (tok_sentence, word_range)
        tok_sentence - one tokenized sentence ie. ["A","nurse","must","always","take","care","of","her","patients"]
        word_range - a nested list of clusters with their ranges ie. [[[0,1],[7,7]]]

    Returns:
        sentence with <mark> </mark> tags at the start and end of the word range
        ie. ['<mark>A', 'nurse</mark>', 'must', 'always', 'take', 'care', 'of', '<mark>her</mark>', 'patients']

"""
def insert_tags(tok_sent, word_range):
    for cluster in word_range:
        for (start_index, end_index) in cluster:
            tok_sent[start_index]="<mark>"+tok_sent[start_index]
            tok_sent[end_index]=tok_sent[end_index]+"</mark>"