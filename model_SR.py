from nltk.tokenize import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import wordnet
from nltk.corpus import stopwords
import xgboost
import pickle
import pandas as pd
import numpy as np
import re
import string
import nltk


# nltk.download('stopwords')
# nltk.download('punkt')
# nltk.download('averaged_perceptron_tagger')
# nltk.download('wordnet')
# nltk.download('omw-1.4')


class SentimentRecommenderModel:
    ROOT_PATH = "C:\\Users\\MRBANER\\Documents\\Personal Docs\\Upgrad\\10_DEPLOYMENT\\Submission\\"
    MODEL_NAME = "pickle_sentiment-classification-xg-boost-model.pkl"
    VECTORIZER = "pickle_tfidf-vectorizer.pkl"
    RECOMMENDER = "pickle_user_final_rating.pkl"
    CLEANED_DATA = "pickle_cleaned-data.pkl"

    def __init__(self):
        self.model = pickle.load(open(SentimentRecommenderModel.ROOT_PATH + SentimentRecommenderModel.MODEL_NAME, 'rb'))
        self.vectorizer = pd.read_pickle(
            SentimentRecommenderModel.ROOT_PATH + SentimentRecommenderModel.VECTORIZER)
        self.user_final_rating = pickle.load(open(
            SentimentRecommenderModel.ROOT_PATH + SentimentRecommenderModel.RECOMMENDER, 'rb'))
        self.data = pd.read_csv("C:\\Users\\MRBANER\\Documents\\Personal Docs\\Upgrad\\10_DEPLOYMENT\\SENTIMENT BASED "
                                "RECOMMENDATION SYSTEM\\sample30_IS.csv")
        self.cleaned_data = pickle.load(open(
            SentimentRecommenderModel.ROOT_PATH + SentimentRecommenderModel.CLEANED_DATA, 'rb'))
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))

    # This is the function to derive the top 20 recommendations for the user

    def RecommendationForUser(self, user):
        recommendations = []
        return list(self.user_final_rating.loc[user].sort_values(ascending=False)[0:20].index)

    # This is the function to finally filter the recommendations to top 5 via the sentiment model

    def SentiRecomm(self, user):
        if user in self.user_final_rating.index:
            # Recommendation from the sentiment model
            recommendations = list(
                self.user_final_rating.loc[user].sort_values(ascending=False)[0:20].index)

            # preprocess the text before transforming and predicting
            filtered_data = self.cleaned_data[self.cleaned_data.id.isin(
                recommendations)]

            # transform the input data using saved tf-idf vectorizer
            X = self.vectorizer.transform(
                filtered_data["reviews_text_cleaned"].values.astype(str))
            filtered_data["predicted_sentiment"] = self.model.predict(X)
            temp = filtered_data[['id', 'predicted_sentiment']]
            temp_grouped = temp.groupby('id', as_index=False).count()
            temp_grouped["pos_review_count"] = temp_grouped.id.apply(lambda x: temp[(
                                                                                            temp.id == x) & (
                                                                                            temp.predicted_sentiment == 1)][
                "predicted_sentiment"].count())
            temp_grouped["total_review_count"] = temp_grouped['predicted_sentiment']
            temp_grouped['pos_sentiment_percent'] = np.round(
                temp_grouped["pos_review_count"] / temp_grouped["total_review_count"] * 100, 2)
            sorted_products = temp_grouped.sort_values(
                'pos_sentiment_percent', ascending=False)[0:5]
            return pd.merge(self.data, sorted_products, on="id")[
                ["name", "brand", "manufacturer", "pos_sentiment_percent"]].drop_duplicates().sort_values(
                ['pos_sentiment_percent', 'name'], ascending=[False, True])

        else:
            print(f"User name {user} doesn't exist")
            return None

    # This function is created to classify the sentiment

    def classify_sentiment(self, review_text):
        review_text = self.preprocess_text(review_text)
        X = self.vectorizer.transform([review_text])
        y_pred = self.model.predict(X)
        return y_pred

    # Before sending the corpus to the model, this function will help preprocess it

    def preprocess_text(self, text):

        # cleaning the review text (lower, removing punctuation, numerical, whitespaces)
        text = text.lower().strip()
        text = re.sub("\[\s*\w*\s*\]", "", text)
        dictionary = "abc".maketrans('', '', string.punctuation)
        text = text.translate(dictionary)
        text = re.sub("\S*\d\S*", "", text)

        # remove stop-words and convert it to lemma
        text = self.lemma_text(text)
        return text

    # This function helps us get the PoS tag

    def get_wordnet_pos(self, tag):
        if tag.startswith('J'):
            return wordnet.ADJ
        elif tag.startswith('V'):
            return wordnet.VERB
        elif tag.startswith('N'):
            return wordnet.NOUN
        elif tag.startswith('R'):
            return wordnet.ADV
        else:
            return wordnet.NOUN

    # This function helps in removing the stopwords from the text

    def remove_stopword(self, text):
        words = [word for word in text.split() if word.isalpha()
                 and word not in self.stop_words]
        return " ".join(words)

    # This function helps in generating the base lemma form with the help of the PoS tags

    def lemma_text(self, text):
        word_pos_tags = nltk.pos_tag(word_tokenize(
            self.remove_stopword(text)))  # Get position tags
        # Map the position tag and lemmatize the word/token
        words = [self.lemmatizer.lemmatize(tag[0], self.get_wordnet_pos(
            tag[1])) for idx, tag in enumerate(word_pos_tags)]
        return " ".join(words)
