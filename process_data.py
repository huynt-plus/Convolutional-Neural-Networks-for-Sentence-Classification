import numpy as np
import cPickle
from collections import defaultdict
import sys, re
import pandas as pd
import codecs

class Line(object):
    def __init__(self, raw, line_no):
        parts = raw.split(',')
        self.line_no = line_no
        self.tid = None
        self.txt = None
        self.label = None
        self.error = False
        if len(parts) != 6:
            bad_parts = parts[5:]
            new_str = ', '.join(bad_parts)
            parts = parts[:5] + [new_str]
        if len(parts) != 6:
            print('parsing: bad line @ %d (parts len = %d): %s' % (self.line_no, len(parts), raw))
            # print(parts)
            self.error = True
        else:
            self.tid = parts[1].strip('"')
            self.txt = parts[5].strip('"')
            tweet = self.txt
            tweet = tweet.lower()
            # Convert https?://* to URL
            tweet = re.sub('((www\.[\s]+)|(https?://[^\s]+))', 'URL', tweet)
            # Convert @username to AT_USER
            tweet = re.sub('@[^\s]+', 'AT_USER', tweet)
            # Remove additional white spaces
            tweet = re.sub('[\s]+', ' ', tweet)
            # Replace #word with word
            tweet = re.sub(r'#([^\s]+)', r'\1', tweet)
            tweet = re.sub('[!\?.,#]', '', tweet)
            # trim
            tweet = tweet.strip()
            # remove first/last " or 'at string end
            tweet = tweet.rstrip('\'"')
            tweet = tweet.lstrip('\'"')
            tweet = tweet.strip('!')
            tweet = tweet.strip('.')
            tweet = tweet.strip(')')
            tweet = tweet.strip('(')
            tweet = tweet.strip(':')
            tweet = tweet.strip('?')
            tweet = tweet.strip(',')
            self.txt = tweet.strip('#')
            try:
                self.label = int(parts[0].strip('"'))
            except ValueError:
                pass
        self.tokens = None

    def __repr__(self):
        if self.txt is not None:
            return 'Line(%s)' % self.txt
        else:
            return 'Line(ERROR)'

def process(filename):
    print('processing %s' % filename)
    unicode_errors = 0
    parse_errors = 0
    lines = []
    count = 0
    with codecs.open(filename, 'rU', 'utf-8', 'ignore') as f:
        for line in f:
            parsed = Line(line.rstrip('\n'), count)
            if parsed.error:
                parse_errors += 1
            else:
                lines.append(parsed)
            count += 1
    print('%d / %d lines with unicode errors' % (unicode_errors, count))
    print('%d / %d lines with parse errors' % (parse_errors, count))

    return lines


def build_data_cv(data_folder, cv=10, clean_string=True):
    """
    Loads data and split into 10 folds.
    """
    revs = []
    pos_file = data_folder[0]
    neg_file = data_folder[1]
    vocab = defaultdict(float)
    with open(pos_file, "rb") as f:
        for line in f:       
            rev = []
            rev.append(line.strip())
            if clean_string:
                orig_rev = clean_str(" ".join(rev))
            else:
                orig_rev = " ".join(rev).lower()
            words = set(orig_rev.split())
            for word in words:
                vocab[word] += 1
            datum  = {"y":1, 
                      "text": orig_rev,                             
                      "num_words": len(orig_rev.split()),
                      "split": np.random.randint(0,cv)}
            revs.append(datum)
    with open(neg_file, "rb") as f:
        for line in f:       
            rev = []
            rev.append(line.strip())
            if clean_string:
                orig_rev = clean_str(" ".join(rev))
            else:
                orig_rev = " ".join(rev).lower()
            words = set(orig_rev.split())
            for word in words:
                vocab[word] += 1
            datum  = {"y":0, 
                      "text": orig_rev,                             
                      "num_words": len(orig_rev.split()),
                      "split": np.random.randint(0,cv)}
            revs.append(datum)
    return revs, vocab
    
def get_W(word_vecs, k=300):
    """
    Get word matrix. W[i] is the vector for word indexed by i
    """
    vocab_size = len(word_vecs)
    word_idx_map = dict()
    W = np.zeros(shape=(vocab_size+1, k), dtype='float32')            
    W[0] = np.zeros(k, dtype='float32')
    i = 1
    for word in word_vecs:
        W[i] = word_vecs[word]
        word_idx_map[word] = i
        i += 1
    return W, word_idx_map

def load_bin_vec(fname, vocab):
    """
    Loads 300x1 word vecs from Google (Mikolov) word2vec
    """
    word_vecs = {}
    with open(fname, "rb") as f:
        header = f.readline()
        vocab_size, layer1_size = map(int, header.split())
        binary_len = np.dtype('float32').itemsize * layer1_size
        for line in xrange(vocab_size):
            word = []
            while True:
                ch = f.read(1)
                if ch == ' ':
                    word = ''.join(word)
                    break
                if ch != '\n':
                    word.append(ch)   
            if word in vocab:
               word_vecs[word] = np.fromstring(f.read(binary_len), dtype='float32')  
            else:
                f.read(binary_len)
    return word_vecs

def add_unknown_words(word_vecs, vocab, min_df=1, k=300):
    """
    For words that occur in at least min_df documents, create a separate word vector.    
    0.25 is chosen so the unknown vectors have (approximately) same variance as pre-trained ones
    """
    for word in vocab:
        if word not in word_vecs and vocab[word] >= min_df:
            word_vecs[word] = np.random.uniform(-0.25,0.25,k)  

def clean_str(string, TREC=False):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Every dataset is lower cased except for TREC
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)     
    string = re.sub(r"\'s", " \'s", string) 
    string = re.sub(r"\'ve", " \'ve", string) 
    string = re.sub(r"n\'t", " n\'t", string) 
    string = re.sub(r"\'re", " \'re", string) 
    string = re.sub(r"\'d", " \'d", string) 
    string = re.sub(r"\'ll", " \'ll", string) 
    string = re.sub(r",", " , ", string) 
    string = re.sub(r"!", " ! ", string) 
    string = re.sub(r"\(", " \( ", string) 
    string = re.sub(r"\)", " \) ", string) 
    string = re.sub(r"\?", " \? ", string) 
    string = re.sub(r"\s{2,}", " ", string)    
    return string.strip() if TREC else string.strip().lower()

def clean_str_sst(string):
    """
    Tokenization/string cleaning for the SST dataset
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)   
    string = re.sub(r"\s{2,}", " ", string)    
    return string.strip().lower()

if __name__=="__main__":    
    w2v_file = sys.argv[1]

    data_folder = ["polarity.pos","polarity.neg"]
    print "loading data...",        
    revs, vocab = build_data_cv(data_folder, cv=10, clean_string=True)
    max_l = np.max(pd.DataFrame(revs)["num_words"])
    print "data loaded!"
    print "number of sentences: " + str(len(revs))
    print "vocab size: " + str(len(vocab))
    print "max sentence length: " + str(max_l)
    print "loading word2vec vectors...",
    w2v = load_bin_vec(w2v_file, vocab)
    print "word2vec loaded!"
    print "num words already in word2vec: " + str(len(w2v))
    add_unknown_words(w2v, vocab)
    W, word_idx_map = get_W(w2v)
    rand_vecs = {}
    add_unknown_words(rand_vecs, vocab)
    W2, _ = get_W(rand_vecs)
    cPickle.dump([revs, W, W2, word_idx_map, vocab], open("processed-data", "wb"))
    print "dataset created!"
    
