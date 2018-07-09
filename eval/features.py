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
import math

class Features:
    "construct (mention, entity) pair's feature vectors, including base feature, \
    contextual feature and string feature"
    def __init__(self):
        self.window = 5
        self.entity_prior = {}
        self.me_prob = {}
        self.mention_cand = {}
        self.res1 = {}
        self.skip = 0
        self.m_count = {}
        self.punc = re.compile('[%s]' % re.escape(string.punctuation))
        self.is_mpme = False
        self.is_me = False
        self.log_file = ''

    def loadWEVec(self, word, entity):
        self.tr_word = word
        self.tr_entity = entity
        self.id_wiki = self.tr_entity.id_wiki

    def loadTitle(self, title, has_sense=True):
        self.tr_title = title
        if has_sense:
            self.is_mpme = True
        else:
            self.is_me = True

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
        print("load %d mentions with %d candidates!" % (len(self.mention_cand),can_sum))

    # load the predicted entity in the first step, or return None
    def loadResult(self, filename):
        count = 0
        if os.path.isfile(filename):
            with codecs.open(filename, 'r', encoding='UTF-8') as fin:
                for line in fin:
                    items = re.split(r',', line.strip())
                    if len(items) < 6 : continue
                    tmp_ans = self.res1[items[2]] if items[2] in self.res1 else set()
                    if items[5] in self.id_wiki:
                        tmp_ans.add(self.id_wiki[items[5]])
                        self.res1[items[2]] = tmp_ans

    def cosSim(self, v1, v2):
        res = spatial.distance.cosine(v1,v2)
        if math.isnan(res) or math.isinf(res) or res >1 or res <-1:
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

    #doc=[w,..,w], mentions = [[doc_pos, ment_name, wiki_id],...], c_entities = [wiki_id, ...]
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
                if len(self.log_file) > 0:
                    self.fout_log.write('miss mention:%s, for entity: %s\n' % (m[1], m[2]))
                continue
            cand_set = self.mention_cand[ment_name]
            cand_size = len(cand_set)
            norm_ment_name = doc[m[0]]

            for cand_id in cand_set:        #cand_id: m's candidates wiki id
                entity_label = self.id_wiki[cand_id]    # check when load candidates
                entity_title = self.getTitle(entity_label)
                tmp_mc_vec = [doc_id]
                #get base features
                pem = 0.0
                if norm_ment_name not in self.me_prob:
                    if len(self.log_file) > 0:
                        self.fout_log.write('miss norm mention: %s, for %s, in anchors!\n' % (norm_ment_name, ment_name) )
                elif cand_id not in self.me_prob[norm_ment_name]:
                    if len(self.log_file) > 0 :
                        self.fout_log.write('miss entity %s, for %s!\n' % (cand_id, norm_ment_name))
                else:
                    pem = float(self.me_prob[norm_ment_name][cand_id])/self.m_count[norm_ment_name]
                pe = 0.0
                if cand_id not in self.entity_prior:
                    if len(self.log_file) > 0:
                        self.fout_log.write('miss entity %s in prior!\n' % cand_id)
                else:
                    pe = self.entity_prior[cand_id]
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
                has_sense = False
                has_etitle = False
                if self.is_mpme and entity_label in self.tr_title.ent_vectors:
                    has_sense = True
                if self.is_me and entity_title in self.tr_title.vectors:
                    has_etitle = True

                # 3 embedding similarity: esim1:(w,e), esim2:(tl,e) [only me], esim3:(w,tls) [only mpme]
                esim1 = 0
                erank1 = 0
                if has_word and has_entity:
                    esim1 = self.cosSim(self.tr_word.vectors[norm_ment_name],self.tr_entity.vectors[entity_label])

                esim2 = 0
                erank2 = 0
                if has_etitle and has_entity:
                    esim2 = self.cosSim(self.tr_title.vectors[entity_title], self.tr_entity.vectors[entity_label])

                esim3 = 0
                erank3 = 0
                if has_sense and has_word:
                    esim3 = self.cosSim(self.tr_word.vectors[norm_ment_name], self.tr_title.ent_vectors[entity_label])

                # 4 contexual similarities for align, me and mpme: csim1:(c(w),e), csim2:(N(e),e), csim3:(tls,c(w)) [only mpme], csim4:(mu(tls),c(w)) [only mpme], csim5:(N(tls),tls)
                # context vec
                c_w_actual = 0
                if has_sense or has_entity:
                    context_vec = np.zeros(self.tr_word.layer_size)
                    begin_pos = m[0] - self.window if m[0] - self.window > 0 else 0
                    end_pos = m[0] + self.window if m[0] + self.window < len(doc) else len(doc) - 1
                    for i in xrange(begin_pos, end_pos + 1):
                        if i == m[0]: continue
                        if doc[i] in self.tr_word.vectors:
                            context_vec += self.tr_word.vectors[doc[i]]
                            c_w_actual += 1
                    if c_w_actual > 0:
                        context_vec /= c_w_actual
                csim1 = 0
                crank1 = 0
                if c_w_actual>0 and has_entity:
                    csim1 = self.cosSim(context_vec,
                                                  self.tr_entity.vectors[entity_label])

                csim2 = 0
                crank2 = 0

                # similarity between entity's mu and mention's context vec
                csim3 = 0
                crank3 = 0
                if c_w_actual > 0 and has_sense:
                    csim3 = self.cosSim(context_vec, self.tr_title.ent_vectors[entity_label])

                csim4 = 0
                crank4 = 0
                if has_sense and c_w_actual>0:
                    csim4 = self.cosSim(context_vec, self.tr_title.ent_mu[entity_label])

                csim5 = 0
                crank5 = 0

                tmp_mc_vec.extend([esim1, erank1, esim2, erank2, esim3, erank3, csim1, crank1, csim2, crank2, csim3, crank3, csim4, crank4, csim5, crank5])
                vec.append(tmp_mc_vec)
                # add entities without ambiguous as truth
                if isFirstStep and pem > 0.95 :
                    c_entities.append(entity_label)
        df_vec = pd.DataFrame(vec, columns = ['doc_id', 'mention_id', 'wiki_id', 'cand_id',\
                                              'cand_size', 'pem', 'pe','largest_pe',\
                                                'str_sim1', 'str_sim2','str_sim3','str_sim4','str_sim5',\
                                                'esim1', 'erank1', 'esim2', 'erank2','esim3', 'erank3',\
                                                'csim1', 'crank1','csim2', 'crank2','csim3', 'crank3','csim4', 'crank4', 'csim5', 'crank5'])

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
                c_ent_num = 0
                for ent in c_entities:
                    if ent in self.tr_entity.vectors:
                        c_ent_vec += self.tr_entity.vectors[ent]
                        c_ent_num += 1
                entity_vec = self.tr_entity.vectors[entity_label]
                if c_ent_num > 0:
                    c_ent_vec /= c_ent_num
                    df_vec.loc[row[0], 'csim2'] = self.cosSim(c_ent_vec, entity_vec)

            # update context entity features by entity title
            if self.is_mpme and len(c_entities) > 0 and entity_label in self.tr_title.ent_vectors:
                c_title_vec = np.zeros(self.tr_word.layer_size)
                c_title_num = 0
                for ent in c_entities:
                    if ent in self.tr_title.ent_vectors:
                        c_title_vec += self.tr_title.ent_vectors[ent]
                        c_title_num += 1
                title_vec = self.tr_title.ent_vectors[entity_label]
                if c_title_num > 0:
                    c_title_vec /= c_title_num
                    df_vec.loc[row[0], 'csim5'] = self.cosSim(c_title_vec, title_vec)

            if cand_count == cand_size and cand_size > 0:
                #compute last mention's candidate rank
                t = -df_vec.loc[ row[0]-cand_size+1:row[0], 'esim1']
                ranks = t.rank(method = 'min')
                for i in ranks.index:
                    df_vec.loc[i, 'erank1'] = ranks[i]

                t = -df_vec.loc[row[0] - cand_size + 1:row[0], 'esim2']
                ranks = t.rank(method='min')
                for i in ranks.index:
                    df_vec.loc[i, 'erank2'] = ranks[i]

                t = -df_vec.loc[row[0] - cand_size + 1:row[0], 'esim3']
                ranks = t.rank(method='min')
                for i in ranks.index:
                    df_vec.loc[i, 'erank3'] = ranks[i]

                t = -df_vec.loc[ row[0]-cand_size+1:row[0], 'csim1']
                ranks = t.rank(method = 'min')
                for i in ranks.index:
                    df_vec.loc[i, 'crank1'] = ranks[i]

                t = -df_vec.loc[ row[0]-cand_size+1:row[0], 'csim2']
                ranks = t.rank(method = 'min')
                for i in ranks.index:
                    df_vec.loc[i, 'crank2'] = ranks[i]

                t = -df_vec.loc[ row[0]-cand_size+1:row[0], 'csim3']
                ranks = t.rank(method = 'min')
                for i in ranks.index:
                    df_vec.loc[i, 'crank3'] = ranks[i]

                t = -df_vec.loc[ row[0] - cand_size + 1:row[0], 'csim4']
                ranks = t.rank(method='min')
                for i in ranks.index:
                    df_vec.loc[i, 'crank4'] = ranks[i]

                t = -df_vec.loc[row[0] - cand_size + 1:row[0], 'csim5']
                ranks = t.rank(method='min')
                for i in ranks.index:
                    df_vec.loc[i, 'crank5'] = ranks[i]

                cand_size = 0
        return df_vec

    def extFeatures(self, doc_file, feature_file):
        if len(self.log_file) > 0:
            self.fout_log = codecs.open(self.log_file, 'w', encoding='UTF-8')
        with codecs.open(doc_file, 'r', encoding='UTF-8') as fin:
            doc_id = 0
            doc = []
            mentions = []
            is_mention = False
            for line in fin:
                line = line.strip()
                if line.startswith('-DOCSTART-'):
                    if doc_id > 0 and len(mentions) > 0:
                        if doc_id in self.res1:
                            vec = self.getFVec(doc_id, doc, mentions, self.res1[doc_id])
                        else:
                            vec = self.getFVec(doc_id, doc, mentions)
                        vec.to_csv(feature_file, mode='a', header=False, index=False)
                    doc_id += 1
                    if doc_id % 20 ==0:
                        print 'has processed %d docs!' % doc_id
                    del doc[:]
                    del mentions[:]
                    is_mention = False
                    continue
                elif len(line) < 1:
                    is_mention = False
                    continue
                else:
                    items = re.split(r'\t', line)
                    if len(items) > 4 and items[1] == 'B' and items[2] in self.mention_cand and items[5] in self.mention_cand[items[2]]:
                        mentions.append([len(doc), items[2], items[5]])
                        doc.append(self.maprule(items[2]))
                        is_mention = True
                    elif is_mention and len(items) > 2 and items[1] == 'I':
                        continue
                    else:
                        tmp_w = self.maprule(items[0])
                        if tmp_w in self.tr_word.vectors:
                            doc.append(tmp_w)
                        is_mention = False
                        continue
            if len(doc) > 0:
                if doc_id in self.res1:
                    vec = self.getFVec(doc_id, doc, mentions, self.res1[doc_id])
                else:
                    vec = self.getFVec(doc_id, doc, mentions)
                vec.to_csv(feature_file, mode='a', header=False, index=False)
        if len(self.log_file) > 0:
            self.fout_log.close()


if __name__ == '__main__':
    aida_file = '/data/m1/cyx/etc/expdata/conll/AIDA-YAGO2-dataset.tsv'
    candidate_file = '/data/m1/cyx/etc/ppr/ppr_candidate'
    wiki_id_file = '/data/m1/cyx/etc/enwiki/wiki_title_cl'
    count_mention_file = '/data/m1/cyx/etc/enwiki/count_mentions'
    output_file = '/data/m1/cyx/etc/expdata/conll/ppr_conll_file2.csv'
    input_path = '/data/m1/cyx/etc/output/exp10/'
    entity_vector_file = input_path + 'vectors_entity10.dat'
    word_vector_file = input_path + 'vectors_word10.dat'
    title_vector_file = input_path + 'vectors_title10.dat'
    log_file = '/data/m1/cyx/etc/expdata/conll/log/log_feature'
    res_file = '/data/m1/cyx/etc/expdata/conll/log/conll_pred.mpme'
    has_title = True
    has_sense = True

    wiki_word = Word()
    wiki_word.loadVector(word_vector_file)

    wiki_entity = Entity()
    wiki_entity.loadWikiId(wiki_id_file)
    wiki_entity.loadVector(entity_vector_file)

    if has_title and has_sense:
        wiki_title = Title()
        wiki_title.loadVector(title_vector_file)

    if has_title and not has_sense:
        wiki_title = Word()
        wiki_title.loadVector(title_vector_file)

    features = Features()
    features.log_file = log_file
    features.loadResult(res_file)
    features.loadWEVec(wiki_word, wiki_entity)

    if has_title:
        features.loadTitle(wiki_title, has_sense)
    #mention's candidate entities {apple:{wiki ids}, ...}
    features.loadCand(candidate_file)
    #p(e)
    features.loadPrior(count_mention_file)
    print("load %d entities' priors!" % len(features.entity_prior))
    #{m:{e1:1, e2:3, ...}} for calculating p(e|m)
    print("load %d mention names with prob !" % len(features.me_prob))
    features.extFeatures(aida_file, output_file)

