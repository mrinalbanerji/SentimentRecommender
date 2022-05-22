from flask import Flask, request, render_template
from model_SR import SentimentRecommenderModel

app = Flask(__name__, template_folder='template')

sentiment_model = SentimentRecommenderModel()


@app.route('/')
def home():
    return render_template('index_SR.html')


@app.route('/predict', methods=['POST'])
def prediction():
    # get user from the html form
    user = request.form['userName']
    # convert text to lowercase
    user = user.lower()
    items = sentiment_model.SentiRecomm(user)

    if (not (items is None)):
        print(f"retrieving items....{len(items)}")
        print(items)
        return render_template("index_SR.html", column_names=items.columns.values, row_data=list(items.values.tolist()),
                               zip=zip)
    else:
        return render_template("index_SR.html",
                               message="User Name doesn't exists, No product recommendations at this point of time!")


@app.route('/predictSentiment', methods=['POST'])
def predict_sentiment():
    # get the review text from the html form
    review_text = request.form["reviewText"]
    print(review_text)
    pred_sentiment = sentiment_model.classify_sentiment(review_text)
    print(pred_sentiment)
    return render_template("index_SR.html", sentiment=pred_sentiment)


if __name__ == '__main__':
    app.run()
