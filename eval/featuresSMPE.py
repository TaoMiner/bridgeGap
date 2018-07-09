import codecs
import regex as re
import Levenshtein
from Entity import Entity
from Word import Word
from Title import Title
import numpy as np
from scipy import spatial
import os
import pandas as pd
import string

class Features:
    "construct (mention, entity) pair's feature vectors, including base feature, \
    contextual feature and string feature"
    def __init__(self):
        self.window = 5
        self.entity_prior = {}
        self.me_prob = {}
        self.mention_cand = {}
        self.res1 = []
        self.skip = 0
        self.m_count = {}
        self.punc = re.compile('[%s]' % re.escape(string.punctuation))

    def loadAllVec(self, word, entity, title):
        self.tr_word = word
        self.tr_entity = entity
        self.tr_title = title
        self.id_wiki = self.tr_entity.id_wiki

    def loadWEVec(self, word, entity):
        self.tr_word = word
        self.tr_entity = entity
        self.id_wiki = self.tr_entity.id_wiki
        self.tr_title = None

    def loadPrior(self, filename):
        with codecs.open(filename, 'r', encoding='UTF-8') as fin:
            total_anchor_num = 0
            for line in fin:
                ent_anchor_num = 0
                items = re.split(r'\t', line.strip())
                if len(items) < 3 : continue
                for mention in items[2:]:
                    tmp_items = re.split(r'::=', mention)
                    if len(tmp_items)!=2: continue
                    tmp_count = int(tmp_items[1])
                    ent_anchor_num += tmp_count
                    tmp_entity_count = self.me_prob[tmp_items[0]] if tmp_items[0] in self.me_prob else {}
                    if items[0] in tmp_entity_count:
                        tmp_entity_count[items[0]] += tmp_count
                    else:
                        tmp_entity_count[items[0]] = tmp_count
                    self.me_prob[tmp_items[0]] = tmp_entity_count
                self.entity_prior[items[0]] = float(ent_anchor_num)
                total_anchor_num += ent_anchor_num
        for ent in self.entity_prior:
            self.entity_prior[ent] /= total_anchor_num
            self.entity_prior[ent] *= 100
        for m in self.me_prob:
            self.m_count[m] = sum([self.me_prob[m][k] for k in self.me_prob[m]])

    def savePrior(self, file):
        with codecs.open(file, 'w', encoding='UTF-8') as fout:
            for ent in self.entity_prior:
                fout.write('%s\t%f\n' % (ent,self.entity_prior[ent]))

    def loadCand(self, filename):
        with codecs.open(filename, 'r', encoding='UTF-8') as fin:
            can_sum = 0
            for line in fin:
                items = re.split(r'\t', line.strip())
                tmp_set = set()
                for i in items[1:]:
                    if i in self.id_wiki:
                        tmp_set.add(i)
                if len(tmp_set) > 0:
                    self.mention_cand[items[0]] = tmp_set
                    can_sum += len(tmp_set)
        print("load %d mentions with %d candidates!" % (len(features.mention_cand),can_sum))

    # load the predicted entity in the first step, or return None
    def loadResult(self, filename):
        if os.path.isfile(filename):
            with codecs.open(filename, 'r', encoding='UTF-8') as fin:
                for line in fin:
                    items = re.split(r'\t', line.strip())
        return self.res1

    def nearestSenseMu(self, cvec, sense):
        nearest_index = -1
        cloest_sim = -1.0
        if cvec[0] != 0 and cvec[-1] != 0:
            for i in xrange(sense.size):
                sim = self.cosSim(cvec, sense.mu[i])
                if sim > cloest_sim :
                    cloest_sim = sim
                    nearest_index = i
        return nearest_index

    def cosSim(self, v1, v2):
        res = spatial.distance.cosine(v1,v2)
        if res == np.nan or res == np.inf or res >1 or res <-1:
            res = 1
        return 1-res

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

    def getTitle(self, entity_label):
        return re.sub(r'\(.*?\)$', '', entity_label).strip()

        # doc=[w,..,w], mentions = [[doc_index, ment_name, wiki_id],...], c_entities = [wiki_id, ...]
    def getFVec(self, doc_id, doc, mentions, c_entities = []):
        vec = []
        largest_pe = -1.0
        mention_index = -1
        isFirstStep = True if len(c_entities) < 1 else False
        for m in mentions:      #m [doc_index, sentence_index, wiki_id]
            mention_index += 1
            ment_name = m[1]
            if ment_name not in self.mention_cand:
                self.skip += 1
                continue
            cand_set = self.mention_cand[ment_name]
            cand_size = len(cand_set)
            norm_ment_name = doc[m[0]]

            for cand_id in cand_set:        #cand_id: m's candidates wiki id
                entity_label = self.id_wiki[cand_id]
                entity_title = self.getTitle(entity_label)
                tmp_mc_vec = [doc_id]
                #get base features
                pem = float(self.me_prob[norm_ment_name][cand_id])/self.m_count[norm_ment_name] if norm_ment_name in self.me_prob and cand_id in self.me_prob[norm_ment_name] else 0
                pe = self.entity_prior[cand_id] if cand_id in self.entity_prior else 0
                largest_pe = pe if largest_pe < pe else largest_pe
                tmp_mc_vec.extend([mention_index, m[2], cand_id, cand_size, pem, pe, 0])
                #get string features
                str_sim1 = Levenshtein.distance(ment_name, entity_title)
                str_sim2 = 1 if ment_name == entity_title else 0
                str_sim3 = 1 if entity_title.find(ment_name) else 0
                str_sim4 = 1 if entity_title.startswith(ment_name) else 0
                str_sim5 = 1 if entity_title.endswith(ment_name) else 0
                tmp_mc_vec.extend([str_sim1, str_sim2, str_sim3, str_sim4, str_sim5])

                has_word = True if norm_ment_name in self.tr_word.vectors else False
                has_entity = True if entity_label in self.tr_entity.vectors else False
                has_title = True if entity_title in self.tr_title.vectors else False

                # similarity between word and entity
                contextual_sim1 = 0
                rank1 = 0
                if has_word and has_entity:
                    contextual_sim1 = self.cosSim(self.tr_word.vectors[norm_ment_name],
                                                  self.tr_entity.vectors[entity_label])
                # similarity between word and title
                contextual_sim2 = 0
                rank2 = 0
                if has_word and has_title:
                    contextual_sim2 = self.cosSim(self.tr_word.vectors[norm_ment_name],
                                                  self.tr_title.vectors[entity_title])
                # similarity between entity's mu and mention's context vec
                contextual_sim3 = 0
                rank3 = 1
                # similarity between nearest mention sense mu and context
                contextual_sim4 = 0
                rank4 = 1
                # similarity between mention name and mention sense
                contextual_sim5 = 0
                rank5 = 1

                # similarity of contextual entities
                contextual_sim6 = 0
                rank6 = 0

                tmp_mc_vec.extend([contextual_sim1, rank1, contextual_sim2, rank2, contextual_sim3, rank3, contextual_sim4, rank4, contextual_sim5, rank5, contextual_sim6, rank6])
                vec.append(tmp_mc_vec)
                if isFirstStep and pem > 0.9:
                    c_entities.append(entity_label)
        df_vec = pd.DataFrame(vec, columns = ['doc_id', 'mention_id', 'wiki_id', 'cand_id', 'cand_size', 'pem', 'pe','largest_pe',\
                                      'str_sim1', 'str_sim2','str_sim3','str_sim4','str_sim5',\
                                      'c_sim1', 'rank1', 'c_sim2', 'rank2','c_sim3', 'rank3','c_sim4', 'rank4','c_sim5', 'rank5','c_sim6', 'rank6'])

        cand_count = 0
        cand_size = 0
        last_mention = -1
        for row in df_vec.itertuples():
            # update feature vector for contextual feature's rank and largest pe
            if last_mention != row[2]:
                last_mention = row[2]
                cand_count = 0
                cand_size = row[5]
            entity_label = self.id_wiki[row[4]]
            cand_count += 1
            #update the largest pe in base features
            df_vec.loc[row[0], 'largest_pe'] = largest_pe
            #update context entity features
            if len(c_entities) > 0 and entity_label in self.tr_entity.vectors :
                c_ent_vec = np.zeros(self.tr_word.layer_size)
                for ent in c_entities:
                    if ent in self.tr_entity.vectors:
                        c_ent_vec += self.tr_entity.vectors[ent]
                entity_vec = self.tr_entity.vectors[entity_label]
                df_vec.loc[row[0], 'c_sim6'] = self.cosSim(c_ent_vec, entity_vec)

            if cand_count == cand_size and cand_size > 0:
                #compute last mention's candidate rank, 12->13, 14->15, 16->17, 18->19
                t = -df_vec.loc[ row[0]-cand_size+1:row[0], 'c_sim1']
                ranks = t.rank(method = 'min')
                for i in ranks.index:
                    df_vec.loc[i, 'rank1'] = ranks[i]

                t = -df_vec.loc[ row[0]-cand_size+1:row[0], 'c_sim2']
                ranks = t.rank(method = 'min')
                for i in ranks.index:
                    df_vec.loc[i, 'rank2'] = ranks[i]



                t = -df_vec.loc[row[0] - cand_size + 1:row[0], 'c_sim6']
                ranks = t.rank(method='min')
                for i in ranks.index:
                    df_vec.loc[i, 'rank6'] = ranks[i]

                cand_size = 0
        return df_vec

    def extFeatures(self, doc_file, feature_file):
        with codecs.open(doc_file, 'r', encoding='UTF-8') as fin:
            doc_id = 0
            doc = []
            mentions = []
            for line in fin:
                line = line.strip()
                if line.startswith('-DOCSTART-'):
                    if doc_id > 0 :
                        vec = self.getFVec(doc_id, doc, mentions)
                        vec.to_csv(feature_file, mode='a', header=False, index=False)
                    doc_id += 1
                    if doc_id % 20 ==0:
                        print("has processed:%d" % doc_id)
                    del doc[:]
                    del mentions[:]
                    continue
                elif len(line)<1:
                    continue
                else:
                    items = re.split(r'\t', line)
                    if len(items) < 2 :
                        tmp_w = self.maprule(items[0])
                        if tmp_w in self.tr_word.vectors:
                            doc.append(tmp_w)
                        continue
                    if items[1] == 'I':
                        continue
                    if items[1] == 'B' and len(items)>4 and items[2] in self.mention_cand:
                        mentions.append([len(doc), items[2], items[5]])
                        tmp_w = self.maprule(items[2])
                        doc.append(tmp_w)
            if len(doc) > 0:
                vec = self.getFVec(doc_id, doc, mentions)
                vec.to_csv(feature_file, mode='a', header=False, index=False)


if __name__ == '__main__':
    aida_file = '/data/m1/cyx/etc/expdata/conll/AIDA-YAGO2-dataset.tsv'
    candidate_file = '/data/m1/cyx/etc/ppr/ppr_candidate'
    wiki_id_file = '/data/m1/cyx/etc/enwiki/wiki_title_cl'
    count_mention_file = '/data/m1/cyx/etc/enwiki/count_mentions'
    me_prob_file = '/data/m1/cyx/etc/expdata/conll/prior'
    output_file = '/data/m1/cyx/etc/expdata/conll/ppr_conll_file_nosense.csv'
    entity_vector_file = '/data/m1/cyx/etc/output/exp3/vectors_entity10.dat'
    word_vector_file = '/data/m1/cyx/etc/output/exp3/vectors_word10.dat'
    title_vector_file = '/data/m1/cyx/etc/output/exp3/vectors_title10.dat'

    wiki_word = Word()
    wiki_word.loadVector(word_vector_file)
    print 'load %d words!' % wiki_word.vocab_size

    wiki_entity = Entity()
    wiki_entity.loadWikiId(wiki_id_file)
    wiki_entity.loadVector(entity_vector_file)
    print 'load %d entities!' % wiki_entity.vocab_size

    wiki_title = Word()
    wiki_title.loadVector(title_vector_file)
    print 'load %d titles!' % wiki_title.vocab_size


    features = Features()
    features.loadAllVec(wiki_word, wiki_entity, wiki_title)
    #mention's candidate entities {apple:{wiki ids}, ...}
    features.loadCand(candidate_file)
    #p(e)
    features.loadPrior(count_mention_file)
    print("load %d entities' priors!" % len(features.entity_prior))
    #{m:{e1:1, e2:3, ...}} for calculating p(e|m)
    print("load %d mention names with prob !" % len(features.me_prob))
    features.extFeatures(aida_file, output_file)