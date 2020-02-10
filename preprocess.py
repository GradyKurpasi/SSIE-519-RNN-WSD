# Preprocessing to prepare datasets and write them to disk
# Only intended to run once

# from gensim.test.utils import datapath
from gensim import utils
import gensim.models
from nltk.corpus import semcor
import pickle
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

####################################################################################################
#
# Create and Store Word2Vec distributional word vectors
#
# GENSIM code adapted from GENSIM documentation found at 
# https://radimrehurek.com/gensim/auto_examples/tutorials/run_word2vec.html#sphx-glr-download-auto-examples-tutorials-run-word2vec-py

class MyCorpus(object):
    """An interator that yields sentences (lists of str)."""
    

    def __iter__(self):

        sents = semcor.sents() #subset of the Brown Corpus, tokenized, but not tagged or chunked
        for s in sents :
            # ss = ' '.join(list(s))
            # temp = utils.simple_preprocess(' '.join((list(s))
            yield utils.simple_preprocess(' '.join((list(s))))

print('Creating Word Vectors')
sentences = MyCorpus()
model = gensim.models.Word2Vec(sentences=sentences, size=300) #creates distributional word vectors with dimensionality = 300

import tempfile
with tempfile.NamedTemporaryFile(prefix='gensim-model-', delete=False) as tmp:
    temporary_filepath = tmp.name #'C:\\Users\\gkurp\\AppData\\Local\\Temp\\gensim-model-zviw4kpu'
    model.save(temporary_filepath)

# with tempfile.NamedTemporaryFile(prefix='gensim-model-', delete=False) as tmp:
#     temporary_filepath = 'gensim-model-zviw4kpu'
#     new_model = gensim.models.Word2Vec.load(temporary_filepath)
print('done')


####################################################################################################
#
# Create and Store SemCor word / synset dictionaries
#
# SEMCOR documentation found at
# https://www.nltk.org/_modules/nltk/corpus/reader/semcor.html
# NLTK Corpus howto found at
# http://www.nltk.org/howto/corpus.html

def processChunk(sentence_chunk):

    if "Lemma" in str(sentence_chunk.label()):
        # if "Lemma" in chunk label, chunk represents a sysnset encoding
        descriptor = str(sentence_chunk.label())
        word = str(sentence_chunk[0][0]) if type(sentence_chunk[0][0]) == str else " ".join(sentence_chunk[0][0])
        # if synset encountered, sentence chunk is a tree w subtree
        # sentence_chunk[0] is a tree with POS and [lemma]
        # if [lemma] are only 1 word, type(sentence_chunk[0][0]) is str
        # if [lemma] are more than 1 word type(sentence_chunk[0][0]) is nltk.tree
        # if [lemma] are more than 1 word, str array is joined with " " separator
    else: 
        # chunk represents a regular tree (POS : stop word or punctuation)
        word = str(sentence_chunk[0])
        descriptor = str(sentence_chunk.label())

    if word in lemmas:
        lemmas[word][descriptor] = lemmas[word][descriptor] + 1 if descriptor in lemmas[word] else 1
    else:
        lemmas[word] = {descriptor: 1}  # this else statement prevents keyerror lookups on lemmas[word][synset]
    return word

print("Importing Lemma and Synsets")
lemmas = dict()
    # lemmas is a dict of dict,
    # lemmas[word] = dictionary of { synsets:frequency of synset when associated with a 'word' }
    # lemmas[word][synset] is a count of how many times a synset appears for each word
    # *** len(lemmas[word]) = the number of different senses a 'word' has in the corpus
taggedsentences = semcor.tagged_sents(tag='both')
    # all sentences, fully tagged from SEMCOR
plaintextsentences = semcor.sents()
    # all sentences from SEMCOR
targetsentences = {}
    # sentences containing 'point'
pos = dict()
    # list of part of speech tags from the corpus
max_sentence_len = 0
lemmacount = {}


# find all sentences including exactly 1 occurence of 'back'
# not all of these sentences are related to the synsets we are looking for
# e.g. goes back relates to the verb go instead of back
for i, s in enumerate(plaintextsentences) :
    ss = ' '.join(list(s))
    if ss.count(' back ') == 1: 
        targetsentences[ss] = i
    # temp = utils.simple_preprocess(' '.join((list(s))


# find all lemma and synsets associated with them.
for sentence in taggedsentences:
    # Prepare:
    # synset inventory and count by lemma
    # lemma inventory and count by synset
    for sentence_chunk in sentence:
        processChunk(sentence_chunk)
    if len(sentence) > max_sentence_len : max_sentence_len = len(sentence)

# find lemma with most different senses
# for lemma in lemmas:
#    lemmacount[lemma] = len(lemmas[lemma])
# high_lemma = {i:j for i, j in lemmacount.items() if j > 5}
# high_lemmas ={}
# for a  in high_lemma.keys():
#     high_lemmas[a] = lemmas[a]

with open('lemma.p', 'wb') as f:
     pickle.dump(lemmas, f)

targetsynset = lemmas['back']
with open('targetsynset.p', 'wb') as f:
    pickle.dump(targetsynset, f)

print("Done")


##################################################################################
#
# create and store training / test data
#
# train / test data is dictionary of 
# { sentence : index of sense in target synset}

print("Creating Training/Testing data")
trainingsentences = {}
notfound = 0 
for i, sent in enumerate(targetsentences):
  idx = targetsentences[sent]
  tagsent = taggedsentences[idx]
  for token in tagsent:
    if str(token.label()) in tgtsynsetlist: 
      trainingsentences[sent] = tgtsynsetlist.index(str(token.label())) 
      #print(sent, str(token.label()))
      break
  else:
    notfound += 1
  
print(notfound)
print(len(trainingsentences))
begintestdata = round(len(trainingsentences) * .75)

train_data = dict(list(trainingsentences.items())[:begintestdata])
test_data = dict(list(trainingsentences.items())[begintestdata:])
with open('train_data.p', 'wb') as f:
    pickle.dump(train_data, f)
with open('test_data.p', 'wb') as f:
    pickle.dump(test_data, f)
  
print("Done")