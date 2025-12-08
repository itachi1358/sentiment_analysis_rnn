from flask import Flask,request,jsonify
import numpy as np
from gensim.models import Word2Vec
from tensorflow.keras.models import load_model
import re

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

@app.route("/", methods=["GET"])
def home():
    return jsonify({"status": "Sentiment API is running"})

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()                  # Read JSON body from POST request

    if not data or "review" not in data:
        return jsonify({"error": "Missing 'review' field"}), 400
    
    review = review_tokenize(data["review"])                    # Must be a list of tokens

    
    vec = review_to_vectors(review)            # Convert → (30, 100)
    vec = vec.reshape(1, MAX_LEN, VECTOR_SIZE) # Reshape → (1, 30, 100)

    prob = float(model.predict(vec)[0][0])     # Get prediction probability
    label = 1 if prob >= 0.5 else 0             # Convert to binary label

    return jsonify({
        "prediction": label,
        "score": prob
    })


if __name__=="__main__":
    app.run(debug=True)
