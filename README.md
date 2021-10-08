# inter-word-embedding
a non-neural network approach for semantic word embedding that used contex distribution along with euclidean distance and vocab sampling. The concepts, and formulas will be documented in the first opportunity.
> note: code is dirty, without comment, without using concurrency(wich makes the script so lazy), and not user-friendly.

Part of a pre-trained model uploaded. Just run the notebook and extract data/text5-AC-MODEL-4000.tar.gz. If you want to train your model, you need to open the inter-sampling.go and change the "data/text5" to whatever you want. Other variables are as follow:

 * DATASET_FILE: a filename contained with text data. Each line, one document.
 * WINDOW: number of vocabs that you want to be base of judgement.
 * MAX_RATE: Do not change it. It defines the max rate of vectors. 8 means the vectors are between 0-8.
 * DIMENSION: Dimenstion specifier. default 100. The higher, the better and slower.
 * SAMPLE_ACCURACY: number of samples that specifies vectors. The higher, the better and slower.

