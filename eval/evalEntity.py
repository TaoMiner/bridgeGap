import numpy as np
import regex as re
import codecs
import struct
import rank_metrics as rm
import pdb

eval_file = '/data/m1/cyx/etc/expdata/test_relatedness_id.dat'
entity_dic_file = '/data/m1/cyx/etc/enwiki/wiki_title_cl'
entity_vec_file = '/data/m1/cyx/etc/output/exp10/vectors_entity10.dat'
log_file = '/data/m1/cyx/etc/expdata/log_entity'

ent_id_dic = {}
id_ent_dic = {}
ent_vec = {}
eval_query = {}
relatedness_pair_num = 0
vocab_size = 0
layer_size = 0

def loadEntityDic():
    with codecs.open(entity_dic_file, 'r', encoding='UTF-8') as fin_id:
        for line in fin_id:
            ents = re.split(r'\t', line.strip())
            if len(ents)==2 and ents[0]!="" and ents[0]!=" ":
                t_ent_label = ents[0]
                t_ent_id = ents[1]
                ent_id_dic[t_ent_label] = t_ent_id
                id_ent_dic[t_ent_id] = t_ent_label
        print("successfully load %d entities from wiki entity!" % len(ent_id_dic))

def loadEvalFile():
    with codecs.open(eval_file, 'r', encoding='UTF-8') as fin_eval:
        global relatedness_pair_num
        for line in fin_eval:
            tmp_q = re.split(r'\t', line.strip())
            if len(tmp_q)==3:
                e_id = tmp_q[0]
                c_id = tmp_q[1]
                label = int(tmp_q[2])
		if e_id not in ent_vec or c_id not in ent_vec:
			continue
                if e_id in eval_query and c_id not in eval_query[e_id]:
                    eval_query[e_id][c_id] = label
                else:
                    eval_query[e_id] = {c_id:label}
                relatedness_pair_num += 1
        print("successfully load %d entities with %d candidate entities on average from relatedness file!" % (len(eval_query), relatedness_pair_num/len(eval_query)))

def readEntityId(fin):
    char_set = []
    while True:
        ch = struct.unpack('c',fin.read(1))[0]
        if ch=='\t':
            break
        char_set.append(ch)
    label = "".join(char_set).decode('ISO-8859-1')
    if label in ent_id_dic:
        return ent_id_dic[label]
    else:
        readEntityVector(fin)
        return None

def readEntityVector(fin):
    global layer_size, p_struct_fmt
    tmp_struct_fmt = []
    for i in xrange(layer_size):
        tmp_struct_fmt.append('f')
    p_struct_fmt = "".join(tmp_struct_fmt)
    vec = np.array(struct.unpack(p_struct_fmt, fin.read(4*layer_size)), dtype=float)
    fin.read(1)
    return vec

def readFileHead(fin):
    char_set = []
    vocab_size = 0
    layer_size = 0
    while True:
        ch = fin.read(1)
        if ch==' ':
            vocab_size = (int)("".join(char_set))
            del char_set[:]
            continue
        if ch=='\n':
            layer_size = (int)("".join(char_set))
            break
        char_set.append(ch)
    return [vocab_size, layer_size]

def loadEntityVec():
    with codecs.open(entity_vec_file, 'rb') as fin_vec:
        global vocab_size, layer_size
        [vocab_size, layer_size] = readFileHead(fin_vec)
        for i in xrange(vocab_size):
            tmp_id = readEntityId(fin_vec)
            if tmp_id:
                ent_vec[tmp_id] = readEntityVector(fin_vec)
        print("successfully load %d entities vectors with %d dimensions!" % (vocab_size, layer_size))


loadEntityDic()
loadEntityVec()
loadEvalFile()
ent_skip_count = 0
can_count = 0
ndcg1_sum = 0
ndcg5_sum = 0
ndcg10_sum = 0
map_sum = 0
for ent in eval_query:
    sim = []
    if ent not in ent_vec:
        ent_skip_count += 1
    else:
        tmp_can_count = 0
        for can in eval_query[ent]:
            if can in ent_vec:
                tmp_can_count += 1
                a = ent_vec[ent]*ent_vec[can]
                sim.append((can, a.sum()))
        if tmp_can_count > 1:
            sim_rank = sorted(sim, key=lambda sim : sim[1], reverse=True)
            r = []
            for item in sim_rank:
                r.append(eval_query[ent][item[0]])
            if len(r) >1:
                tmp_n1 = rm.ndcg_at_k(r, 1, 1)
            else:
                tmp_n1 = rm.ndcg_at_k(r, len(r), 1)
            if len(r) >5:
                tmp_n5 = rm.ndcg_at_k(r, 5, 1)
            else:
                tmp_n5 = rm.ndcg_at_k(r, len(r), 1)
            if len(r) >10:
                tmp_n10 = rm.ndcg_at_k(r, 10, 1)
            else:
                tmp_n10 = rm.ndcg_at_k(r, len(r), 1)
            tmp_ap = rm.average_precision(r)
            ndcg1_sum += tmp_n1
            ndcg5_sum += tmp_n5
            ndcg10_sum += tmp_n10
            map_sum += tmp_ap
            can_count += tmp_can_count
        else:
            ent_skip_count +=1
act_ent_count = len(eval_query)-ent_skip_count

with codecs.open(log_file, 'a', encoding='UTF-8') as fout_log:
    fout_log.write("**********************************\n")
    fout_log.write("eval %d(%d) entities with %d(%d) candidate entities for %s!\n" % (act_ent_count,len(eval_query),can_count/act_ent_count,relatedness_pair_num/len(eval_query), entity_vec_file))
    fout_log.write("ndcg1 : %f, ndcg5 : %f, ndcg10 : %f, map : %f\n" % (float(ndcg1_sum/act_ent_count),float(ndcg5_sum/act_ent_count),float(ndcg10_sum/act_ent_count),float(map_sum/act_ent_count)))
    fout_log.write("**********************************\n")
