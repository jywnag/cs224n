import numpy as np

text = 'You say goodbye and I say hello.'

text = text.lower()
text = text.replace('.', ' .')

words = text.split(' ')
word_to_id = {}
id_to_word = {}

for word in words:
    if word not in word_to_id:
        new_id = len(word_to_id)
        word_to_id[word] = new_id
        id_to_word[new_id] = word
print(word_to_id['hello'])
corpus = [word_to_id[w] for w in words]
print(corpus)
corpus = np.array(corpus)
print(corpus)