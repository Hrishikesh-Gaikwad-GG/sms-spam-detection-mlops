from nltk.stem.porter import PorterStemmer
import pandas as pd
import nltk
from nltk.corpus import stopwords
import string
from pathlib import Path



def transform_text(text: str) -> pd.DataFrame:

    """
    Takes a string(sentence) and performs the following operations:
    - lowercase
    - Breaks into words (tokenization)
    - removes non - alnum characters
    - removes punctuations and stopwords
    - converts to its stem word

    :param text: SMS
    :type text: str
    :return: lowercase string converted to root word with punctuation and stopwords removed.
    :rtype: DataFrame

    """

    text = text.lower()
    text = nltk.word_tokenize(text)
 
    y = [i for i in text if i.isalnum()]
    
    y = [i for i in y if i not in stopwords.words('english') and i not in string.punctuation] 

    y = [PorterStemmer().stem(i) for i in y]

    return " ".join(y)


def make_features(df : pd.DataFrame) -> pd.DataFrame:

    df['num_characters'] = df['text'].apply(len)
    df['num_words'] = df['text'].apply(lambda x: len(nltk.word_tokenize(x)))
    df['num_sentences'] = df['text'].apply(lambda x: len(nltk.sent_tokenize(x)))


    return df