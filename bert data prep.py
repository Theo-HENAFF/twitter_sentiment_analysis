import pandas as pd

from sklearn.model_selection import train_test_split

# ------------------------------------------------------
# Import cleaned_data CSV
# ------------------------------------------------------
df = pd.read_csv(
    # "C:/Users/Théo/Documents/twitter_sentiment_analysis/data/cleaned_data.csv",
    "C:/Users/HENAFF/Documents/Cours Polytech/S9 en Roumanie/Machine Learning - ML/data/mid_cleaned_data.csv",
    # nrows=20000,
    encoding='latin-1')
df['clean_text'] = df.clean_text.astype(str)

test_size = 0.25

train, test = train_test_split(df[['clean_text', 'polarity']], test_size=test_size, random_state=1, shuffle=True)

train_0 = train[train['polarity'] == 0]
train_1 = train[train['polarity'] == 1]

test_0 = test[test['polarity'] == 0]
test_1 = test[test['polarity'] == 1]

train_0.to_csv(r'C:/Users/HENAFF/Documents/Cours Polytech/S9 en Roumanie/Machine Learning - ML/data/train/0/train0.csv', index=False)
train_1.to_csv(r'C:/Users/HENAFF/Documents/Cours Polytech/S9 en Roumanie/Machine Learning - ML/data/train/1/train1.csv', index=False)

test_0.to_csv(r'C:/Users/HENAFF/Documents/Cours Polytech/S9 en Roumanie/Machine Learning - ML/data/test/0/test0.csv', index=False)
test_1.to_csv(r'C:/Users/HENAFF/Documents/Cours Polytech/S9 en Roumanie/Machine Learning - ML/data/test/1/test1.csv', index=False)

