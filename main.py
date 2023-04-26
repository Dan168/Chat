from flask import Flask, render_template, request
import nltk
from nltk.stem.lancaster import LancasterStemmer
import json
import tensorflow as tf
import pickle
import numpy as np
import random
nltk.download('punkt')

stemmer = LancasterStemmer()

user_log = []
bot_log = []

with open("intents.json") as file:
  data = json.load(file)
app = Flask(__name__)

model = tf.keras.models.load_model('model/modelV2')

with open('data.pickle', 'rb') as f:
    words, labels, training, output = pickle.load(f)


def bag_of_words(s, words):
  bag = [0 for _ in range(len(words))]
  s_words = nltk.word_tokenize(s)
  s_words = [stemmer.stem(word.lower()) for word in s_words]

  for se in s_words:
    for i, w in enumerate(words):
      if w == se:
        bag[i] = 1

  return np.array(bag)


@app.route("/", methods=["GET", "POST"])
def home():
    response = ""
    if request.method == "POST":
        inp = request.form["user_input"]
        user_log.append(inp)

        # Convert input text to bag-of-words vector
        bow = bag_of_words(inp, words)

        # Reshape bag-of-words vector to match expected input shape of model
        bow = bow.reshape(1, -1)

        # Make prediction using Keras model
        results = model.predict(bow)
        results_index = np.argmax(results)
        tag = labels[results_index]

        # Find corresponding response based on predicted tag
        for tg in data['intents']:
            if tg['tag'] == tag:
                responses = tg['responses']

        response = random.choice(responses)
        bot_log.append(response)

    return render_template("index.html", response=response, bot_log=bot_log, user_log=user_log)


if __name__ == "__main__":
    app.run()
