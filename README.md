# Document-Clustering-using-HAC
Hierarchical (agglomerative/bottom-up) clustering HAC (in Python)
Represent each document as a unit vector. The unit vector of a document is obtained from tf*idf vector of the document, normalized (divided) by its Euclidean length. Recall that
tf is term frequency (# of occurrences of word in the document) and idf is given by log 𝑁+1, where N df+1
is the number of documents in the given collection and df is the number of documents where the word appears.
