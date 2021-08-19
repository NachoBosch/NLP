import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

#.---Functions---.#
def calculate_results(y_true,y_pred):
  model_accuracy = accuracy_score(y_true,y_pred)
  model_precision, model_recall, model_f1,_ = precision_recall_fscore_support(y_true,y_pred,average="weighted")
  model_results = {"accuracy": model_accuracy,
                   "precision":model_precision,
                   "recall":model_recall,
                   "f1":model_f1}
  return model_results  


'''chekear labels de test creo que estan en cero'''

# Dir of csv's 
csv_dir = 'Toxic-comments/'

# Let's read the csv's belong to Toxic comments
sample_submission = pd.read_csv(csv_dir+'sample_submission.csv')
test = pd.read_csv(csv_dir+'test.csv')
test_labels = pd.read_csv(csv_dir+'test_labels.csv')
train = pd.read_csv(csv_dir+'train.csv')
print(train[:50])

# Lets clean some dataframes
test.drop('id', inplace=True, axis=1)
test_labels.drop('id', inplace=True, axis=1)
train.drop('id', inplace=True, axis=1)
train_data = train['comment_text']
train.drop('comment_text',inplace=True,axis=1)
train_labels=train
print("#------------------------------#")
print(type(train_data), train_data.shape)
print(type(train_labels), train_labels.shape)
print("#--------------___-------------#")


# Let's split data into train and test
train_sentence,test_sentence,train_labels,test_labels = train_test_split(train_data.to_numpy(),train_labels.to_numpy(),test_size=0.1,random_state=42)


# Let's tokenize and vectorize and embedding
max_vocab_length = 10000
max_length = 5000
text_vectorizer = tf.keras.layers.experimental.preprocessing.TextVectorization(max_tokens=max_vocab_length,
                                                                               output_mode='int',
                                                                               output_sequence_length=max_length)
embedding = tf.keras.layers.Embedding(input_dim=max_vocab_length,
                                      output_dim = 300,
                                       input_length=max_length)

#Build model with the Functional API
inputs = tf.keras.layers.Input(shape=(1,),dtype=tf.string)
x = text_vectorizer(inputs)
x = embedding(x)
x = tf.keras.layers.GlobalMaxPool1D()(x)
outputs = tf.keras.layers.Dense(6,activation="softmax")(x)

# Create model, compile and fit
model_0 = tf.keras.Model(inputs,outputs)
model_0.compile(optimizer=tf.keras.optimizers.Adam(),
	loss=tf.keras.losses.CategoricalCrossentropy(),
	metrics=["accuracy"])
model_0.summary()
#history = model_0.fit(train_sentence,train_labels,epochs=1)
#model_0.save('Models/model_0.h5')
#model_0.save_weights('Models/model_0_weights.h5')
#pred_model_0=model_0.predict(test_sentence)
#print("Preds:",pred_model_0)

#.---------------------------------------------------------------.#
# model with sklearn
'''
model_1 = Pipeline([
                    ("tfidf",TfidfVectorizer()),
                    ("clf",MultinomialNB())])
# Fit the pipeline to the training data
history=model_1.fit(train_sentence,train_labels)
baseline_score = model_1.score(test_sentence, test_labels)
print(f"Our baseline model achieves an accuracy of: {baseline_score*100:.2f}%")
pred_model_1=model_1.predict(test_sentence)
'''
#.--------------------------------------------------------------.#
#Calculate preds error in model_0 and model_1
#model_0_results = calculate_results(test_labels,pred_model_0)
#model_1_results = calculate_results(test_labels,pred_model_1)
#.--------------------------------------------------------------.#
#Plot the differences between models
#all_model_results = pd.DataFrame({"Model_0":model_0_results})
#all_model_results.plot(kind='bar',figsize=(10,7)).legend()