import numpy as np
import pickle as pk
from tqdm import tqdm
import glob
from nltk.tokenize import word_tokenize
import torch

# Class for a memory-friendly iterator over the dataset
class GenSentences(object):
    def __init__(self, filename):
        self.filename = filename

        with open(self.filename, 'r') as para_file:
            self.para = para_file.read()

vocab_file = '../vocab.pkl'
vocab = pk.load(open(vocab_file,'rb'))
print("Vocab size: %d"%(len(vocab)))

vocab_file = '../vocab_code.pkl'
vocab_code = pk.load(open(vocab_file,'rb'))
print("Code Vocab size: %d"%(len(vocab_code)))

train_docs = []
dev_docs = []
test_docs = []

numbr_docs = len(glob.glob("../docs/*.txt"))
count_train = 0
count_dev = 0
count_test = 0
code_len = 0
numbr_test_tokens = 0
for idx in tqdm(range(numbr_docs), desc='Reading Doc Files..'):
    doc_file = '../docs/%d.txt'%(idx)

    para = GenSentences(doc_file).para
    snippet = GenSentences(code_file).para

    tokens = word_tokenize(para)
    doc = []
    doc_mallet = []
    for token in tokens:
        if token in vocab:
            doc.append(vocab[token])
            doc_mallet.append(token)
        else:
            doc.append(vocab['<unk>'])

    if idx <= int(0.85 * numbr_docs):
        #torch.save(torch.LongTensor(np.array(doc)), '../train_docs/%d.pt'%(count_train))
        #torch.save(torch.LongTensor(np.array(code)), '../train_code/%d.pt'%(count_train))
        with open('../train_mallet/%d.txt'%(count_train), 'w') as train_docu_mallet:
            train_docu_mallet.write(' '.join(doc_mallet))
        count_train = count_train + 1

    if int(0.85 * numbr_docs) < idx <= int(0.9 * numbr_docs):
        #torch.save(torch.LongTensor(np.array(doc)), '../dev_docs/%d.pt'%(count_dev))
        #torch.save(torch.LongTensor(np.array(code)), '../dev_code/%d.pt'%(count_dev))
        with open('../dev_mallet/%d.txt'%(count_dev), 'w') as dev_docu_mallet:
            dev_docu_mallet.write(' '.join(doc_mallet))
        count_dev = count_dev + 1

    if int(0.9 * numbr_docs) < idx <= numbr_docs:
        #torch.save(torch.LongTensor(np.array(doc)), '../test_docs/%d.pt'%(count_test))
        #torch.save(torch.LongTensor(np.array(code)), '../test_code/%d.pt'%(count_test))
        numbr_test_tokens = numbr_test_tokens + len(set(doc_mallet))
        with open('../test_mallet/%d.txt'%(count_test), 'w') as test_docu_mallet:
            test_docu_mallet.write(' '.join(doc_mallet))
        count_test = count_test + 1

#print("Avg. Code Length: %d"%(code_len//numbr_docs))
print("Number of Test Tokens: %d"%(numbr_test_tokens))
