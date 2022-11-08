# inter-word-embedding
inter-sampling is a non-neural network approach for semantic word embedding that uses context distribution along with euclidean distance and vocab sampling. Now we are using SVD for dimension reduction and concurrency. Methods and formulas will be documented in the first opportunity.
> note: code is dirty, without comment, and not user-friendly.

Part of a pre-trained model uploaded. Just run the notebook and extract data/text5-AC-MODEL-4000.tar.gz. If one wants to train another model, it just needs to open the inter-sampling.go and change the "data/text5" to whatever you want. Other variables are as follow:

 * DATASET_FILE: a filename contained with text data. Each line, one document.
 * WINDOW: number of vocabs that script should use to be base of judgment.
 * MAX_RATE: Do not change it. It defines the max rate of vectors. Eight means the vectors are between 0-8.
 * DIMENSION: Dimension specifier. default 100. The higher, the better and slower.
 * SAMPLE_ACCURACY: number of samples that specifies vectors. The higher, the better and slower.

