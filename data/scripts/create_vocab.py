from nltk.corpus import stopwords
import pickle as pk
import operator

stop_words = set(stopwords.words('english'))

freqdict_file = '../doc_freq.pkl'
freq_dict = pk.load(open(freqdict_file,'rb'))
sorted_freq_list = sorted(freq_dict.items(), key=operator.itemgetter(1),reverse=True)

vocab_without_stopwords = {}
vocab_without_stopwords['<eos>'] = 0
vocab_without_stopwords['<sos>'] = 1
vocab_without_stopwords['<unk>'] = 2

vocab_with_stopwords = {}
vocab_with_stopwords['<eos>'] = 0
vocab_with_stopwords['<sos>'] = 1
vocab_with_stopwords['<unk>'] = 2

idx = 2
idx_stop = 2

for word, freq in sorted_freq_list:

    idx_stop = idx_stop + 1
    vocab_with_stopwords[word] = idx_stop
    if word not in stop_words:
        idx = idx + 1
        vocab_without_stopwords[word] = idx

print("Vocab with stopwords: %d"%(len(vocab_with_stopwords)))
print("Vocab without stopwords: %d"%(len(vocab_without_stopwords)))

with open('../vocab.pkl', 'wb') as vocab_file:
    pk.dump(vocab_without_stopwords, vocab_file)

with open('../vocab_with_stopwords.pkl', 'wb') as vocab_file:
    pk.dump(vocab_with_stopwords, vocab_file)
