"""
JSON line scheme:

dict {
> 'linex_index' :
  '0'
> 'features' :
  list [
    dict {
      > 'layers' :
        list [
          dict {
            > 'index' :
              number [min: -4, mean: -2.5, max: -1]
            > 'values' :
              list [
                number [min: -12.17, mean: -0.02341, max: 4.246]
                sublist.lengths: 768
              ]
          }
          sublist.lengths: 4
        ]
      > 'token' :
        string
          'for' x 2
          19 other elements
    }
  ]
}


"""

import os
import json
import numpy as np
from MulticoreTSNE import MulticoreTSNE as TSNE
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

def pool_sentence_embs(path, pooling_layer=-1, pooling_strategy='mean'):
    embs = []
    sents = []
    with open(path, 'r') as f:
        for line in f.readlines():
            line = line.strip()
            dic = json.loads(line)
            token_embs = []
            tokens = []
            for index in range(len(dic['features'])): #index is word in sent
                token_embs.append(dic['features'][index]['layers'][pooling_layer]['values'])
                tokens.append(dic['features'][index]['token'])
            sents.append(" ".join(tokens))
            token_embs = np.asarray(token_embs) #length of sentence X hidden dimension 768
            if pooling_strategy == "mean":
                embs.append(token_embs.mean(0))

        embs = np.asarray(embs)
        print(embs.shape)
        #print (sents)
    return embs
            # sent = dic[]

def main(path):
    embs = pool_sentence_embs(path)
    print("Dimension", embs.shape) #number sentences X BERT hidden dimension (768)
    df = pd.read_csv('results_1584_gutenberg.csv', encoding ='utf - 8', index_col = False)
    target = df['label']
    print(len(target))
    assert (len(target) == embs.shape[0]), "length of target from input file must match BERT embedding length"

    # Step size of the mesh. Decrease to increase the quality of the VQ.
    h = .02  # point in the mesh [x_min, x_max]x[y_min, y_max].

    # Reduce data dimentionality using PCA
    reduced_data = PCA(n_components=2).fit_transform(embs)

    num_clusters = [2,3,4,5,6,7,8,9,10]

    # run kmeans for each k in num_clusters
    for k in num_clusters:
        kmeans = KMeans(init='k-means++', n_clusters=k, n_init=10)
        kmeans.fit(reduced_data)

        # Plot the decision boundary. For that, we will assign a color to each
        x_min, x_max = reduced_data[:, 0].min() - 1, reduced_data[:, 0].max() + 1
        y_min, y_max = reduced_data[:, 1].min() - 1, reduced_data[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

        # Obtain labels for each point in mesh. Use last trained model.
        Z = kmeans.predict(np.c_[xx.ravel(), yy.ravel()])

        # Put the result into a color plot
        Z = Z.reshape(xx.shape)
        plt.figure(1)
        plt.clf()
        plt.imshow(Z, interpolation='nearest',
                extent=(xx.min(), xx.max(), yy.min(), yy.max()),
                cmap=plt.cm.Paired,
                aspect='auto', origin='lower')

        # Plot the centroids as a white X
        centroids = kmeans.cluster_centers_
        plt.scatter(centroids[:, 0], centroids[:, 1],
                    marker='x', s=169, linewidths=3,
                    color='w', zorder=10)
        plt.scatter(reduced_data[:, 0], reduced_data[:, 1], c=target, cmap=ListedColormap(["blue", "red"]), marker='.', s=50)
        title = "k-means clustering (PCA-reduced data) with "+str(k)+" clusters. (0=blue, 1=red)"
        plt.title(title+'\n'
                  'Centroids are marked with white cross')
        plt.xlim(x_min, x_max)
        plt.ylim(y_min, y_max)
        plt.xticks(())
        plt.yticks(())
        plt.savefig(title+".png",bbox_inches='tight')
        plt.ioff()
        #plt.show()

if __name__ == '__main__':
    main('1584_plotting.json')
