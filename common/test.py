import sys
import sys
sys.path.append('..')
from common.trainer import Trainer
from common.optimizer import Adam
from common.util import preprocess, create_contexts_target,convert_one_hot
import numpy as np

sys.path.append('..')
from common.util import preprocess, create_co_matrix, cos_similarity
from common.layers import MatMul
text = 'You say goodbye and I say hello.'
corpus, word_to_id, id_to_word = preprocess(text)
print(corpus)
window_size = 1
target = corpus[window_size:-window_size]
contexts = []

for idx in range(window_size, len(corpus) - window_size):
    cs = []
    for t in range(-window_size, window_size + 1):
        if t == 0:
            continue
        cs.append(corpus[idx + t])
    contexts.append(cs)
h = np.array([[0, 1,2],
            [3 ,4, 5],
            [6, 7, 8]])
target_W = np.array([[ 0 ,1, 4],
        [27, 40, 55],
         [18, 28 ,40]])

c = target_W * h
print(c)
# print(np.array(contexts))
# print(np.array(target))
# for i in corpus:
#     print(id_to_word[i])
# 0.707106769115479