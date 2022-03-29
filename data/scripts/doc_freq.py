from nltk.tokenize import word_tokenize
import glob
import pickle as pk
from tqdm import tqdm
import math


vocab = set()

# Class for a memory-friendly iterator over the dataset
class GenSentences(object):
    def __init__(self, filename):
        self.filename = filename

        with open(self.filename, 'r') as para_file:
            self.para = para_file.read()

idf_dict = {}
numbr_docs = len(glob.glob("../docs/*.txt"))

#count = 0
for file in tqdm(glob.glob("../docs/*.txt"), desc='Reading Doc Files..'):
    #count = count + 1
    para = GenSentences(file).para
    tokens = word_tokenize(para)

    for token in set(tokens):
        if token.isalpha():
            if token not in idf_dict.keys():
                idf_dict[token] = 1
            else:
                idf_dict[token] = idf_dict[token] + 1

    # if count == 3:
    #     print(freq_dict)
    #     quit()
print("Vocab size: %d"%(len(idf_dict)), flush=True)
for key in idf_dict:
    idf_dict[key] = math.log(float(numbr_docs/idf_dict[key]))

with open('../doc_freq.pkl','wb') as docfreq_file:
    pk.dump(idf_dict, docfreq_file)
