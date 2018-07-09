import codecs
import struct
import numpy as np
from Word import Word
from Title import Title
from Map import Map
import math
from scipy import stats
from scipy import spatial

class Evaluator:
    def __init__(self):
        self.standard = []

    def loadData(self,file):
        self.data = map(lambda x: x.strip().lower().split('\t'), open(file).readlines())
        self.standard = [float(d[2]) for d in self.data]

    def loadWordEmbeddings(self, word):
        self.tr_word = word
        self.tr_title = None
        self.tr_map = None

    def loadEmbeddings(self, word, title):
        self.tr_word = word
        self.tr_title = title

    def loadMentionMap(self, mention_map):
        self.tr_map = mention_map

    def cosSim(self, v1, v2):
	return 1 - spatial.distance.cosine(v1,v2)

    def getSenseVec(self,word):
        if not self.tr_map or word not in self.tr_map.names or not self.tr_title:
            return [self.tr_word.vectors[word]]
        re_vec = [self.tr_word.vectors[word]]
        for m in self.tr_map.names[word]:
            if m in self.tr_title.ent_vectors:
                re_vec.append(self.tr_title.ent_vectors[m])
        return re_vec

    def evaluate(self):
        glb = []
        avg = []
        simmax = []
        for d in self.data:
            if d[1] not in self.tr_word.vectors or d[0] not in self.tr_word.vectors:
                glb.append(0)
                avg.append(0)
                simmax.append(0)
            else:
                sim = self.cosSim(self.tr_word.vectors[d[1]], self.tr_word.vectors[d[0]])
                glb.append(sim)
                if self.tr_map:
                    sumsim, tmp_sim, count, largest_sim = 0, 0, 0, -1.0
                    for i in self.getSenseVec(d[1]):
                        for j in self.getSenseVec(d[0]):
                            tmp_sim = self.cosSim(i, j)
                            if tmp_sim > largest_sim:
                                largest_sim = tmp_sim
                            sumsim += tmp_sim
                            count += 1
                    avg.append(sumsim / count)
                    simmax.append(largest_sim)
                else:
                    avg.append(sim)
                    simmax.append(sim)
        output = open('/data/m1/cyx/etc/expdata/wordsim353_pred', 'w')
        for i in xrange(len(self.data)):
            output.write('%s\t%s\t%f\t%f\n' % (self.data[i][0], self.data[i][1], self.standard[i], glb[i]*10 ))
        output.close()
        return stats.spearmanr(self.standard, glb), stats.spearmanr(self.standard, avg), stats.spearmanr(self.standard, simmax)

if __name__ == '__main__':
    word_vector_file = '/data/m1/cyx/etc/output/exp3/vectors_word10.dat'
    title_vector_file = '/data/m1/cyx/etc/output/exp10/vectors_title10.dat'
    mention_name_file = '/data/m1/cyx/etc/enwiki/mention_names'
    word_sim_file = '/data/m1/cyx/etc/expdata/wordsim_similarity_goldstandard.txt'
    output_file = '/data/m1/cyx/etc/expdata/log_wordsim353'
    wiki_id_file = '/data/m1/cyx/etc/enwiki/wiki_title_cl'
    has_title = False

    wiki_word = Word()
    wiki_word.loadVector(word_vector_file)
    print len(wiki_word.vectors)
    if has_title:
        wiki_title = Title()
        wiki_title.loadVector(title_vector_file)
        print len(wiki_title.vectors)

        mention_map = Map()
        mention_map.loadWikiID(wiki_id_file)
        mention_map.load(mention_name_file)
	   print len(mention_map.names)

    eval = Evaluator()
    eval.loadData(word_sim_file)
    if has_title:
        eval.loadEmbeddings(wiki_word, wiki_title)
        eval.loadMentionMap(mention_map)
    else:
        eval.loadWordEmbeddings(wiki_word)
    gs, avgs, maxs = eval.evaluate()
    output = open(output_file, 'a')
    output.write('\n*****************************************************')
    if has_title:
        output.write('\n'+word_sim_file+'\n' + word_vector_file + '\n' + title_vector_file +'\n' + 'glb: ' + str(gs)+ 'avg: ' + str(avgs)+ 'glb: ' + str(maxs))
    else:
        output.write('\n' +word_sim_file+'\n'+ word_vector_file + '\n' + 'glb: ' + str(gs) )
    output.write('\n*****************************************************')
    output.close()

