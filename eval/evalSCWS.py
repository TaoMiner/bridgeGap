import codecs
import struct
import numpy as np
from Word import Word
from Title import Title
from Map import Map
import math
from scipy import stats
from scipy import spatial
import regex as re
import string

class Evaluator:
    def __init__(self):
        self.standard = []
        self.punc = re.compile('[%s]' % re.escape(string.punctuation))
        self.skip = 0
        self.half_window = 5
        self.data_size = 0
        self.avg_sense_num = 0

    def loadData(self,file):
        self.data = map(lambda x: x.lower().split('\t'), open(file).readlines())
        self.standard = [float(d[7]) for d in self.data]
        self.data_size = len(self.standard)

    def loadWordEmbeddings(self, word):
        self.tr_word = word
        self.tr_title = None
        self.tr_map = None

    def maprule(self, str):
        # possessive case 's
        # tmp_line = re.sub(r' s |\'s', ' ', str)
        # following clean wiki xml, punctuation, numbers, and lower case
        tmp_line = self.punc.sub(' ', str)
        tmp_line = tmp_line.replace('\t', ' ')
        tmp_line = re.sub(r'[\s]+', ' ', tmp_line)
        tmp_line = re.sub(r'(?<=\s)(\d+)(?=($|\s))', 'dddddd', tmp_line)
        tmp_line = re.sub(r'(?<=^)(\d+)(?=($|\s))', 'dddddd', tmp_line).lower().strip()
        return tmp_line

    def loadEmbeddings(self, word, title):
        self.tr_word = word
        self.tr_title = title

    def loadMentionMap(self, mention_map):
        self.tr_map = mention_map

    def cosSim(self, v1, v2):
        return 1 - spatial.distance.cosine(v1,v2)

    def getSenseVec(self,sent):
        items = re.split(r' ', sent)
        word_pos = -2
        word = ''
        context = []
        for item in items:
            if item == '<b>':
                word_pos = -1
                continue
            if word_pos == -1:
                word = self.maprule(item)
                if word not in self.tr_word.vectors: return None, None
                if not self.tr_map:
                    return [self.tr_word.vectors[word]], [1.0]
                if word not in self.tr_map.names:
                    return None, None
                word_pos = len(context)
                context.append(word)
                continue
            if item == '<\b>': continue
            item = self.maprule(item)
            if len(item) < 1 or item not in self.tr_word.vectors: continue
            context.append(item)

        context_vec = [0 for i in xrange(self.tr_word.layer_size)]
        if len(context) > 1:
            count = 0
            for i in xrange(max(0, word_pos - self.half_window), word_pos):
                for j in xrange(self.tr_word.layer_size):
                    context_vec[j] += self.tr_word.vectors[context[i]][j]
                count += 1
            if word_pos < len(context)-1:
                for i in xrange(word_pos + 1, min(word_pos + self.half_window, len(context))):
                    for j in xrange(self.tr_word.layer_size):
                        context_vec[j] += self.tr_word.vectors[context[i]][j]
                    count += 1
            if count > 0:
                for j in xrange(self.tr_word.layer_size):
                    context_vec[j] /= count
            else: return None, None

        re_vec = []
        re_w = []
        for m in self.tr_map.names[word]:
            if m in self.tr_title.ent_vectors:
                re_vec.append(self.tr_title.ent_vectors[m])
                re_w.append(self.cosSim(context_vec,self.tr_title.ent_mu[m]))
            if m in self.tr_title.ent_title:
                tmp_t = self.tr_title.ent_title[m]
                if tmp_t in self.tr_title.vectors:
                    for i in xrange(self.tr_title.vectors[tmp_t].size):
                        re_vec.append(self.tr_title.vectors[tmp_t].vector[i])
                        re_w.append(self.cosSim(context_vec,self.tr_title.vectors[tmp_t].mu[i]))

        weight_Z = sum(re_w)
        if weight_Z == 0:
            weight_Z = 0.000001
        re_w = [w / weight_Z for w in re_w]

        return re_vec, re_w

    def evaluate(self):
        avg_res2 = []
        max_res2 = []
        skip_standard = []
        count = -1
        for d in self.data:
            count += 1
            # if count%100 == 0:
            # 	print "evaluating number: ", count
            emb1, wei1 = self.getSenseVec(d[5])
            if not emb1:
                self.skip += 1
                continue
            emb2, wei2 = self.getSenseVec(d[6])
            if not emb2:
                self.skip += 1
                continue
            self.avg_sense_num += len(emb1)
            self.avg_sense_num += len(emb2)
            skip_standard.append(self.standard[count])
            score = 0.0
            for i in xrange(len(emb1)):
                for j in xrange(len(emb2)):
                    score += wei1[i] * wei2[j] * self.cosSim(emb1[i], emb2[j])
            avg_res2.append(score)

            i = wei1.index(max(wei1))
            j = wei2.index(max(wei2))
            score = self.cosSim(emb1[i], emb2[j])
            max_res2.append(score)
        return stats.spearmanr(skip_standard, avg_res2), stats.spearmanr(skip_standard, max_res2)

if __name__ == '__main__':
    word_vector_file = '/data/m1/cyx/etc/output/exp10/vectors_word10.dat'
    title_vector_file = '/data/m1/cyx/etc/output/exp10/vectors_title10.dat'
    mention_name_file = '/data/m1/cyx/etc/enwiki/mention_names'
    scws_file = '/data/m1/cyx/etc/expdata/ratings.txt'
    output_file = '/data/m1/cyx/etc/expdata/log_scws'
    wiki_id_file = '/data/m1/cyx/etc/enwiki/wiki_title_cl'
    has_title = True

    wiki_word = Word()
    wiki_word.loadVector(word_vector_file)
    print 'load %d words!' % len(wiki_word.vectors)
    if has_title:
        wiki_title = Title()
        wiki_title.loadVector(title_vector_file)
        print 'load %d titles with %d entity senses!' % (wiki_title.vocab_size,wiki_title.ent_vocab_size)

        mention_map = Map()
        mention_map.loadWikiID(wiki_id_file)
        mention_map.load(mention_name_file)
        print 'load %d mention names!' % len(mention_map.names)

    eval = Evaluator()
    eval.loadData(scws_file)
    if has_title:
        eval.loadEmbeddings(wiki_word, wiki_title)
        eval.loadMentionMap(mention_map)
    else:
        eval.loadWordEmbeddings(wiki_word)
    avg_res, max_res = eval.evaluate()
    output = open(output_file, 'a')
    output.write('\n*****************************************************')
    if len(eval.standard)-eval.skip > 0:
        output.write('\ntotal %d pairs, skip %d pairs, %d sense num avg!' % (len(eval.standard), eval.skip, eval.avg_sense_num/2/(len(eval.standard)-eval.skip)))
    if has_title:
        output.write('\n'+scws_file+'\n' + word_vector_file + '\n' + title_vector_file +'\n' + 'avg: ' + str(avg_res)+ 'max: ' + str(max_res))
    else:
        output.write('\n' +scws_file+'\n'+ word_vector_file + '\n' + 'avg: ' + str(avg_res) )
    output.write('\n*****************************************************')
    output.close()
