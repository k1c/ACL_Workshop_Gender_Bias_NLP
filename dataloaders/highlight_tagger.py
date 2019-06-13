#TODO update comments to relfect new highlight
"""
    Input: (tok_sentence, word_range)
        tok_sentence - one tokenized sentence ie. ["A","nurse","must","always","take","care","of","her","patients"]
        word_range - a nested list of clusters with their ranges ie. [[[0,1],[7,7]]]

    Returns:
        sentence with <mark> </mark> tags at the start and end of the word range
        ie. ['<mark>A', 'nurse</mark>', 'must', 'always', 'take', 'care', 'of', '<mark>her</mark>', 'patients']

"""
def insert_tags(tok_sent, word_range, GENDER_PRONOUNS):
    background_colors = ["FF0099","7FDBFF","01FF70"] #yellow, aqua, lime TODO: add more colors and an assert
    index = 0
    for cluster in word_range:
        if any([((c[0] == c[1]) and (tok_sent[c[0]]).lower() in GENDER_PRONOUNS) for c in cluster]):
            for (start_index, end_index) in cluster:
                tok_sent[start_index]="<span style='background-color:#"+background_colors[index]+"'>"+tok_sent[start_index]
                tok_sent[end_index]=tok_sent[end_index]+"</span>"
            index += 1
    return tok_sent