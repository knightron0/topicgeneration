# Topic Generation
Finding out the current topic from caption. This uses a Latent Dirichlet Allocation (LDA) Model in order to predict the topic from captions. LDA is known to be most efficient way to do Topic Modelling

# Requirements
- nltk
- gensim
- sklearn

# How to use?
Clone the repository and open it in the terminal. Enter the following command:-
```
python main.py /pth/to/file
```
Note: The file should contain the paragraph for which the topic needs to be generated.

# Data Preprocessing
The most important part in Natural Language Processing or any type of Machine Learning is Data Preparation. For this project, in order to get the data ready for the model, the following steps were carried out:
- Tokenization: The document is split into sentences and the sentences are split into words. These words are further turned into lowercase and the punctuation is removed.
- Stopwords are removed and words which have length <= 3 are also removed.
- Lemmatization: The versbs are changed from past and future tense to present tense.
- Stemming: The words are stemmed and are converted to their root form.

After applying these steps, the dataset is then converted into a dictionary, which contains the word as the key and the number of occurences as the value. This is what is fed into the model.

