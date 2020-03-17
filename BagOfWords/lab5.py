from sklearn import preprocessing
from sklearn.svm import SVC
from sklearn.metrics import f1_score, accuracy_scoreg
import numpy as np

train_sentences = np.load('data/dataLab5/training_sentences.npy', allow_pickle = True)
train_labels = np.load('data/dataLab5/training_labels.npy')
test_sentences = np.load('data/dataLab5/test_sentences.npy', allow_pickle = True)
test_labels = np.load('data/dataLab5/test_labels.npy')

def normalize_data(train_data, test_data, type=None):
    if type == "standard":
        scaler = preprocessing.StandardScaler()
    elif type == "minmax":
        scaler = preprocessing.MinMaxScaler()
    elif type == "l1":
        scaler = preprocessing.Normalizer(norm='l1')
    elif type == "l2":
        scaler = preprocessing.Normalizer(norm='l2')
    scaler.fit(train_data)
    scaled_train = scaler.transform(train_data)
    scaled_data = scaler.transform(test_data)
    return scaled_train, scaled_data

class Bow:
    train_sentences = np.load('data/dataLab5/training_sentences.npy', allow_pickle = True)
    def __init__(self):
        self.vocab = {}
        self.words = []
        self.len = 0
    def buildVocab(self, data):
        for doc in data:
            for word in doc:
                if word  not in  self.vocab:
                    self.vocab[word] = len(self.words)
                    self.words.append(word)
        self.len = len(self.words)
    def getFeatures(self, data):
        features = np.zeros((data.shape[0], self.len))
        for id, sentence in enumerate(data):
            for cuv in sentence:
                if cuv in self.vocab:
                    features[id, self.vocab[cuv]] += 1
        return features

obj = Bow()
obj.buildVocab(train_sentences)

trainingF = obj.getFeatures(train_sentences)
testF = obj.getFeatures(test_sentences)

normalized_train, normalized_test = normalize_data(trainingF, testF, type = 'l2')

classifier = SVC(C = 1, kernel='linear')
classifier.fit(normalized_train, train_labels)

prediction = classifier.predict(normalized_test)
print(accuracy_score(test_labels, prediction))
print(f1_score(test_labels, prediction))

weights = np.squeeze(classifier.coef_)
index = np.argsort(weights)
cuvinte = np.array(obj.words)

print("Primele 10 cuvinte negative:", cuvinte[index[-10:]])
print("Primele 10 cuvinte positive:", cuvinte[index[:10]])








