import pandas as pd
import numpy as np
import re
import tensorflow as tf
from sklearn.model_selection import train_test_split
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)
# Dir of csv's 
csv_dir = 'Toxic-comments/'

# Let's read the csv's belong to Toxic comments
sample_submission = pd.read_csv(csv_dir+'sample_submission.csv')
test = pd.read_csv(csv_dir+'test.csv')
test_labels = pd.read_csv(csv_dir+'test_labels.csv')
train = pd.read_csv(csv_dir+'train.csv')

#print(train.head())
#print(train["comment_text"][168])
#print("toxic:",train["toxic"][168],"severe_toxic:",train["severe_toxic"][168],"insult:",train["insult"][168])

toxic_comments_labels = train[["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]]
#print(toxic_comments_labels.head())

def preprocess_text(sen):
    # Remove punctuations and numbers
    sentence = re.sub('[^a-zA-Z]', ' ', sen)

    # Single character removal
    sentence = re.sub(r"\s+[a-zA-Z]\s+", ' ', sentence)

    # Removing multiple spaces
    sentence = re.sub(r'\s+', ' ', sentence)

    return sentence

X = []
sentences = list(train["comment_text"])
print(sentences[:5])
for sen in sentences:
    X.append(preprocess_text(sen))
y = toxic_comments_labels.values

train_sentence,test_sentence,train_labels,test_labels = train_test_split(X,y,test_size=0.2,random_state=42)

tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=5000)
tokenizer.fit_on_texts(train_sentence)

X_train = tokenizer.texts_to_sequences(train_sentence)
X_test = tokenizer.texts_to_sequences(test_sentence)

vocab_size = len(tokenizer.word_index) + 1

maxlen = 200

X_train = tf.keras.preprocessing.sequence.pad_sequences(X_train, padding='post', maxlen=maxlen)
X_test = tf.keras.preprocessing.sequence.pad_sequences(X_test, padding='post', maxlen=maxlen)

# Let's tokenize and vectorize and embedding
max_vocab_length = len(tokenizer.word_index) + 1
max_length = 5000

embedding = tf.keras.layers.Embedding(input_dim=max_vocab_length,
                                      output_dim = 100,
                                       input_length=max_length)

#Build model with the Functional API
inputs = tf.keras.layers.Input(shape=(200,))
x = embedding(inputs)
lstm = tf.keras.layers.LSTM(128)(x)
outputs = tf.keras.layers.Dense(6,activation="sigmoid")(lstm)

# Create model, compile and fit
model_0 = tf.keras.Model(inputs,outputs)
model_0.compile(optimizer=tf.keras.optimizers.Adam(),
	loss=tf.keras.losses.BinaryCrossentropy(),
	metrics=["accuracy"])
model_0.summary()
history = model_0.fit(X_train,train_labels,epochs=1)

score = model_0.evaluate(X_test, test_labels, verbose=1)

print("Test Score:", score[0])
print("Test Accuracy:", score[1])

df1 = pd.DataFrame(history.history)
plt.figure(figsize=(10,7))
plt.plot(df1)
plt.show()