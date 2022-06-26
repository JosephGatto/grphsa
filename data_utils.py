import numpy as np
import spacy
import pickle

from spacy.tokens import Doc

class Tokenizer(object):
    def __init__(self, word2idx=None):
        if word2idx is None:
            self.word2idx = {}
            self.idx2word = {}
            self.idx = 0
            self.word2idx['<pad>'] = self.idx
            self.idx2word[self.idx] = '<pad>'
            self.idx += 1
            self.word2idx['<unk>'] = self.idx
            self.idx2word[self.idx] = '<unk>'
            self.idx += 1
        else:
            self.word2idx = word2idx
            self.idx2word = {v:k for k,v in word2idx.items()}

    def fit_on_text(self, text):
        text = text.lower()
        words = text.split()
        for word in words:
            if word not in self.word2idx:
                self.word2idx[word] = self.idx
                self.idx2word[self.idx] = word
                self.idx += 1
    
    def text_to_sequence(self, text):
        text = text.lower()
        words = text.split()
        unknownidx = 1
        sequence = [self.word2idx[w] if w in self.word2idx else unknownidx for w in words]
        if len(sequence) == 0:
            sequence = [0]
        return sequence




class WhitespaceTokenizer(object):
    def __init__(self, vocab):
        self.vocab = vocab

    def __call__(self, text):
        words = text.split()
        # All tokens 'own' a subsequent space character in this tokenizer
        spaces = [True] * len(words)
        return Doc(self.vocab, words=words, spaces=spaces)

nlp = spacy.load('en_core_web_sm')
nlp.tokenizer = WhitespaceTokenizer(nlp.vocab)

def dependency_adj_matrix(text):
    # https://spacy.io/docs/usage/processing-text
    tokens = nlp(text)
    words = text.split()
    matrix = np.zeros((len(words), len(words))).astype('float32')
    assert len(words) == len(list(tokens))

    for token in tokens:
        matrix[token.i][token.i] = 1
        for child in token.children:
            matrix[token.i][child.i] = 1
            matrix[child.i][token.i] = 1

    return matrix

import penman 
def amr_adj_matrix(text, amr):
    graph = penman.decode(amr)
    alignments = penman.surface.alignments(graph)
    # print(text.split())
    adj = np.zeros((len(text.split()),len(text.split()))) 
    node_token_map = {}
    for id, token in enumerate(text.split()): 
      for key, value in alignments.items():
        if id in list(value.indices):
          node = key[0]
          node_token_map[node] = id 

    for id, edge in enumerate(graph.edges()):
      if edge.source in node_token_map.keys() and edge.target in node_token_map.keys():
        s = node_token_map[edge.source]
        t = node_token_map[edge.target]
        adj[s][t] = 1 ## 1
        adj[t][s] = 1 ## 1

    for id, att in enumerate(graph.attributes()):
      if att.source in node_token_map.keys() and att.target in node_token_map.keys():
        s = node_token_map[att.source]
        t = node_token_map[att.target]
        adj[s][t] = 1 
        adj[t][s] = 1 
    
    for id in range(adj.shape[0]):
      adj[id][id] = 1

    return adj 

def combo_matrix(text, amr):
  amr_mat = amr_adj_matrix(text, amr)
  adj_mat = dependency_adj_matrix(text)

  assert amr_mat.shape == adj_mat.shape 
  out = amr_mat + adj_mat 
  twos = np.where(out==2) 
  out[twos] = 1
  return out 


import os
import pickle
import numpy as np

def load_word_vec(path, word2idx=None, embed_dim=300):
    fin = open(path, 'r', encoding='utf-8', newline='\n', errors='ignore')
    word_vec = {}
    for line in fin:
        tokens = line.rstrip().split()
        word, vec = ' '.join(tokens[:-embed_dim]), tokens[-embed_dim:]
        if word in word2idx.keys():
            word_vec[word] = np.asarray(vec, dtype='float32')
    return word_vec


def build_embedding_matrix(word2idx, embed_dim, fname = '/content/drive/MyDrive/glove_vecs/glove.6B.300d.txt', type=None):

    print('loading word vectors ...')
    embedding_matrix = np.zeros((len(word2idx), embed_dim))  # idx 0 and 1 are all-zeros
    embedding_matrix[1, :] = np.random.uniform(-1/np.sqrt(embed_dim), 1/np.sqrt(embed_dim), (1, embed_dim))
    
    word_vec = load_word_vec(fname, word2idx=word2idx, embed_dim=embed_dim)
    # print('building embedding_matrix:', embedding_matrix_file_name)
    for word, i in word2idx.items():
        vec = word_vec.get(word)
        if vec is not None:
            # words not found in embedding index will be all-zeros.
            embedding_matrix[i] = vec
    # pickle.dump(embedding_matrix, open('/content/drive/MyDrive/glove_vecs/embed_matrix_absa', 'wb'))
    return embedding_matrix

