import string
import pandas as pd
import nltk
import re

nltk.download('stopwords')
nltk.download('wordnet')


def clean_text(text):
    # remove users tags and url
    text = ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)", " ", text).split())
    # remove punctuation
    text = "".join([char for char in text if char not in string.punctuation])
    text = re.sub('[0-9]+', '', text)
    # tokenize
    text = re.split('\W+', text)
    # remove stop-word
    stopword = nltk.corpus.stopwords.words('english')
    text = [word for word in text if word not in stopword]
    # Stemming
    ps = nltk.PorterStemmer()
    text = [ps.stem(word) for word in text]
    # Lammitization
    wn = nltk.WordNetLemmatizer()
    text = [wn.lemmatize(word) for word in text]
    # remove empty words
    text = [word for word in text if word != '']
    # rejoin for easier one-hot extraction
    text = ' '.join(text)
    return text


# Import CSV
# data = pd.read_csv(
#     "C:/Users/HENAFF/Documents/Cours Polytech/S9 en Roumanie/Machine Learning - ML/data/training.1600000.processed.noemoticon.csv",
#     header=None,
#     encoding='latin-1',
#     usecols=[0, 5])

# available columns are [0,1,2,4,5]=['polarity', 'id', 'date', 'user', 'text']
# data.columns = ['polarity', 'text']
# data['polarity'] = pd.to_numeric(data['polarity'], downcast='integer')
# 0 ->[1,0] negative ou 0, 4 ->[0,1] positive ou 1
# data.polarity = data.polarity.replace({0: 0, 4: 1})

# Cleaning all the data
# data['clean_text'] = data['text'].apply(lambda x: clean_text(x))
# data.to_csv(r'C:/Users/HENAFF/Documents/Cours Polytech/S9 en Roumanie/Machine Learning - ML/data/cleaned_data.csv', index = False)


# The goal is to have a smaller dataset for training
def create_small_dataset(n, dataset):
    return pd.concat([dataset.head(n // 2), dataset.tail(n // 2)], ignore_index=True)


data2 = pd.read_csv(
    "C:/Users/HENAFF/Documents/Cours Polytech/S9 en Roumanie/Machine Learning - ML/data/cleaned_data.csv",
    encoding='latin-1')
data2['clean_text'] = data2.clean_text.astype(str)

print("start creating a small dataset")
small_df = create_small_dataset(100000, data2)
print(small_df['polarity'].value_counts())
print("finishing creating a small dataset")
small_df.to_csv(
    r'C:/Users/HENAFF/Documents/Cours Polytech/S9 en Roumanie/Machine Learning - ML/data/mid_cleaned_data.csv',
    index=False)
print("finishing saving it")

