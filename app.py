from flask import Flask,request,jsonify, render_template
import numpy as np
from gensim.models import Word2Vec
from tensorflow.keras.models import load_model
import re
import os

stop_words = {
    "a","an","the","and","or","but","if","is","are","was","were","am",
    "this","that","to","of","in","on","for","with","as","at","by","from"
}



model = load_model("sentiment_rnn_model.h5")
w2v_model = Word2Vec.load("word2vec.model")

app=Flask(__name__)

MAX_LEN = 30
VECTOR_SIZE = 100


def review_tokenize(review):
     review=review.lower()
     review = re.sub(r'<.*?>', '', review)
     review = re.sub(r'[^\w\s]', '', review)
     words= review.split()
     words=[word for word in words if word not in stop_words]
     return words
    



def review_to_vectors(review):
    vectors = []

    for word in review:
        if word in w2v_model.wv:
            vectors.append(w2v_model.wv[word])

    while len(vectors) < MAX_LEN:
        vectors.append(np.zeros(VECTOR_SIZE))

    return np.array(vectors[:MAX_LEN])


@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None
    confidence = None
    user_text = ""

    if request.method == "POST":
        user_text = request.form["review"]

        tokens = review_tokenize(user_text)
        vec = review_to_vectors(tokens)
        vec = vec.reshape(1, MAX_LEN, VECTOR_SIZE)

        prob = float(model.predict(vec)[0][0])
        label = 1 if prob >= 0.45 else 0

        prediction = "Positive ✅" if label == 1 else "Negative ❌"
        confidence = round(prob * 100, 2)

    return render_template(
        "index.html",
        prediction=prediction,
        confidence=confidence,
        user_text=user_text
    )

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
