# twitter_sentiment_analysis
The aim of this project is to develop a neural network using python libraries.  
These NNs will analyse tweets and detect whether they are good or not(polarity analysis).

We use [Sentiment 140 dataset](https://www.kaggle.com/kazanova/sentiment140). It contains 1,600,000 tweets extracted using the twitter api.

For the 2nd part of this project we use [Hate Speech dataset](https://www.kaggle.com/arkhoshghalb/twitter-sentiment-analysis-hatred-speech). It is composed of "only" 32k training tweet where only 7% of them are hate speech.

In order to analyse we use Bi-LSTM NNs and currently the best results comes with Word2Vec embedding.

Most of the project is running with **Keras** and **Tensorflow 2.3** background.  

# Disclaimer
Most of the proposed scripts need high configurations with decent GPU.  
Scripts 3, 4, 5 and bonus need also at least 32GB of RAM. More is recommended (was running fine with 52GB). For the other scripts 16GB of RAM was fine with an Nvidia GTX1070 GPU.  

The Word2Vec model used is nearly 5GB and the FastText one is 15GB. They may be manually from [Frederic Godin GitHub's](https://github.com/FredericGodin/TwitterEmbeddings)
A command is provided to download them automatically

# How to get the project work
## Setup a conda environment with python 3.7 and ML libraries
### Create the virtual env
`conda create -n yourenvname python=3.7`

### Navigate to the env
`conda activate yourenvname`

### Install libraries using Pypi (pip)
For Scripts 1 to 7  
`pip install scikit-learn pandas numpy scipy tensorflow keras gensim matplotlib`

For bonus script it is recommended to setup an environment for itself because it use ktrain which have compatibility issue with keras.  

## Downloads
### Datasets
They are pretty "light"(~350MB and 10MB unzipped) and can be found an Kaggle.  
Links : [Sentiment 140](https://www.kaggle.com/kazanova/sentiment140), [Hate Speech](https://www.kaggle.com/arkhoshghalb/twitter-sentiment-analysis-hatred-speech)  
  
### Word2Vec and Gensim models
As I said previously in the disclaimer part they are respectively 5GB and 15GB.  
They are stored in a Google Drive. Because they are heavy files they need a confirmation to be downloaded.
Thanks to beliys comment into this [GitHub](https://gist.github.com/iamtekeste/3cdfd0366ebfd2c0d805#gistcomment-2316906) we can bypass the confirmation.

Using jupyter terminal or any linux-based terminal we are now able to download each file from a single command-line using `wget`.  
For the Word2Vec model :  
`wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1lw5Hr6Xw0G0bMT1ZllrtMqEgCTrM7dzc' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1lw5Hr6Xw0G0bMT1ZllrtMqEgCTrM7dzc" -O word2vec_twitter_tokens.bin && rm -rf /tmp/cookies.txt`

For the FastText model :
`wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=15zXlbO3bxSYTPt71Fon5-0-oCB8iMSno' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=15zXlbO3bxSYTPt71Fon5-0-oCB8iMSno" -O fasttext.bin && rm -rf /tmp/cookies.txt`

## Clean the data
In order not to repeat a 15min data cleansing every time we run one of the Sentiment 140 scripts I've created a jupyter notebook(`0 - Data cleansing - Sentiment140.ipynb`) which only perform data cleansing for the Sentiment140 dataset.

## You are now ready to launch every scripts to try them by yourself

