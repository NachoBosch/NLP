import numpy as np

samples = ['The cat sat on the mat.','the dog ate my homework.']

token_index = {}
for sample in samples:
    for word in sample.split():
        if word not in token_index:# esta condicion es para evitar que se sobre-escriban los datos
            token_index[word] = len(token_index)+1
#print(token_index)
max_length = len(token_index)
results = np.zeros(shape=(len(samples),max_length,max(token_index.values())+1))
print(results.shape)
for i, sample in enumerate(samples):
    for j,word in list(enumerate(sample.split()))[:max_length]:
        index = token_index.get(word)
        results[i,j,index]=1.
print(results)
