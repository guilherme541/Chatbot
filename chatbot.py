# this class was based on Jere Xu article available at:
# https://towardsdatascience.com/how-to-create-a-chatbot-with-python-deep-learning-in-less-than-an-hour-56a063bdfc44
# thanks to Jere Xu ;)

import numpy as np
import json
import pickle

import nltk
import ssl
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

#nltk.download()

from nltk.stem import WordNetLemmatizer

from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from tensorflow.keras.optimizers import SGD

import random

class ChatBot:
    words = []
    classes = []
    documents = []
    intents = []
    model = []
    ignore_words = ['?', '!']
    lemmatizer = WordNetLemmatizer()

    def createModel(self):
        nltk.download('punkt')
        nltk.download('punkt_tab')
        nltk.download('wordnet')
        nltk.download('omw-1.4')

        data_file = open('intents.json', encoding="utf8").read()
        self.intents = json.loads(data_file)
        for intent in self.intents['intents']:
            for pattern in intent['patterns']:
                w = nltk.word_tokenize(pattern)
                self.words.extend(w)
                self.documents.append((w, intent['tag']))
                if intent['tag'] not in self.classes:
                    self.classes.append(intent['tag'])

        self.words = [self.lemmatizer.lemmatize(w.lower()) for w in self.words if w not in self.ignore_words]
        self.words = sorted(list(set(self.words)))

        self.classes = sorted(list(set(self.classes)))

        print(len(self.documents), "documents")
        print(len(self.classes), "classes", self.classes)
        print(len(self.words), "unique lemmatized words", self.words)

        pickle.dump(self.words, open('words.pkl', 'wb'))
        pickle.dump(self.classes, open('classes.pkl', 'wb'))

        training = []
        output_empty = [0] * len(self.classes)
        for doc in self.documents:
            bag = []
            pattern_words = doc[0]
            pattern_words = [self.lemmatizer.lemmatize(word.lower()) for word in pattern_words]
            for w in self.words:
                bag.append(1) if w in pattern_words else bag.append(0)

            output_row = list(output_empty)
            output_row[self.classes.index(doc[1])] = 1

            training.append([bag, output_row])

        random.shuffle(training)
        training = np.array(training,dtype=object)
        train_x = list(training[:, 0])
        train_y = list(training[:, 1])
        print("Training data created")

        model = Sequential()
        model.add(Dense(128, input_shape=(len(train_x[0]),), activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(64, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(len(train_y[0]), activation='softmax'))

        sgd = SGD(learning_rate=0.01, decay=1e-6, momentum=0.9, nesterov=True)
        model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

        hist = model.fit(np.array(train_x), np.array(train_y), epochs=2000, batch_size=5, verbose=1)
        model.save('chatbot_model.h5', hist)

        self.model = model
        print("model created")

    def loadModel(self):
        from keras.models import load_model
        self.model = load_model('chatbot_model.h5')
        self.intents = json.loads(open('intents.json').read())
        self.words = pickle.load(open('words.pkl', 'rb'))
        self.classes = pickle.load(open('classes.pkl', 'rb'))

    def clean_up_sentence(self, sentence):
        sentence_words = nltk.word_tokenize(sentence)
        sentence_words = [self.lemmatizer.lemmatize(word.lower()) for word in sentence_words]
        return sentence_words

    def bow(self,sentence, words, show_details=True):
        sentence_words = self.clean_up_sentence(sentence)
        bag = [0]*len(words)
        for s in sentence_words:
            for i,w in enumerate(words):
                if w == s:
                    bag[i] = 1
                    if show_details:
                        print("found in bag: %s" % w)
        return (np.array(bag))

    def predict_class(self,sentence, model):
        p = self.bow(sentence, self.words,show_details=False)
        res = model.predict(np.array([p]))[0]
        ERROR_THRESHOLD = 0.25
        results = [[i,r] for i,r in enumerate(res) if r>ERROR_THRESHOLD]
        results.sort(key=lambda x: x[1], reverse=True)
        return_list = []
        for r in results:
            return_list.append({"intent": self.classes[r[0]], "probability": str(r[1])})
        return return_list

    def getResponse(self, ints, intents_json):
        tag = ints[0]['intent']
        list_of_intents = intents_json['intents']
        for i in list_of_intents:
            if(i['tag']== tag):
                result = random.choice(i['responses'])
                break
        return result

    def chatbot_response(self, msg):
        ints = self.predict_class(msg, self.model)
        res = self.getResponse(ints, self.intents)
        return res, ints