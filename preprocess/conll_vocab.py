import codecs
import regex as re
import string

vocab_word = set()
vocab_entity = set()
vocab_mention = set()
id_wiki = {}
mention_cand = {}
punc = re.compile('[%s]' % re.escape(string.punctuation))


def loadCand(filename):
    with codecs.open(filename, 'r', encoding='UTF-8') as fin:
        can_sum = 0
        for line in fin:
            items = re.split(r'\t', line.strip())
            tmp_set = set()
            for i in items[1:]:
                if i in id_wiki:
                    tmp_set.add(id_wiki[i])
            if len(tmp_set) > 0:
                mention_cand[items[0]] = tmp_set
                can_sum += len(tmp_set)
    print("load %d mentions with %d candidates!" % (len(mention_cand), can_sum))

def loadWiki(file):
    with codecs.open(file, 'r', encoding='UTF-8') as fin:
        for line in fin:
            items = re.split(r'\t', line.strip())
            if len(items) < 2 or items[0] == "" or items[0] == " ": continue
            id_wiki[items[1]] = items[0]
    print 'load %d wikis!\n' % len(id_wiki)

def maprule(str):
    # possessive case 's
    # tmp_line = re.sub(r' s |\'s', ' ', str)
    # following clean wiki xml, punctuation, numbers, and lower case
    tmp_line = punc.sub(' ', str)
    tmp_line = tmp_line.replace('\t', ' ')
    tmp_line = re.sub(r'[\s]+', ' ', tmp_line)
    tmp_line = re.sub(r'(?<=\s)(\d+)(?=($|\s))', 'dddddd', tmp_line)
    tmp_line = re.sub(r'(?<=^)(\d+)(?=($|\s))', 'dddddd', tmp_line).lower().strip()
    return tmp_line

def buildVocab(file):
    with codecs.open(file, 'r', encoding='UTF-8') as fin:
        doc_id = 0
        doc = []
        mentions = []
        train_num = 0
        testa_num = 0
        testb_num = 0
        train_cand_skip = 0
        testa_cand_skip = 0
        testb_cand_skip = 0
        for line in fin:
            line = line.strip()
            if line.startswith('-DOCSTART-'):
                doc_id += 1
                del doc[:]
                del mentions[:]
                continue
            elif len(line) < 1:
                continue
            else:
                items = re.split(r'\t', line)
                if len(items) < 2:
                    tmp_w = maprule(items[0])
                    continue
                if items[1] == 'I':
                    continue
                if items[1] == 'B' and len(items) > 4:
                    mentions.append([len(doc), items[2], items[5]])
                    if items[2] in mention_cand:
                        vocab_entity.update(mention_cand[items[2]])
                        if doc_id > 0 and doc_id <= 946:
                            train_num += len(mention_cand[items[2]])
                        elif doc_id <= 1162:
                            testa_num += len(mention_cand[items[2]])
                        else:
                            testb_num += len(mention_cand[items[2]])
                    else:
                        if doc_id > 0 and doc_id <= 946:
                            train_cand_skip += 1
                        elif doc_id <= 1162:
                            testa_cand_skip += 1
                        else:
                            testb_cand_skip += 1
                    if items[5] in id_wiki:
                        vocab_entity.add(id_wiki[items[5]])
                    tmp_w = maprule(items[2])
                    doc.append(tmp_w)
    print '%d train candidates, skip %d candidates!\n' % (train_num, train_cand_skip)
    print '%d testa candidates, skip %d candidates!\n' % (testa_num, testa_cand_skip)
    print '%d testb candidates, skip %d candidates!\n' % (testb_num, testb_cand_skip)
    print 'total %d entities!\n' % len(vocab_entity)

def output(word_file, entity_file, mention_file):
    with codecs.open(word_file, 'w', encoding='UTF-8') as fout:
        fout.write('%s' % '\n'.join(vocab_word))
    with codecs.open(entity_file, 'w', encoding='UTF-8') as fout:
        fout.write('%s' % '\n'.join(vocab_entity))
    with codecs.open(mention_file, 'w', encoding='UTF-8') as fout:
        fout.write('%s' % '\n'.join(vocab_mention))

wiki_file = '../etc/wiki_title_cl'
data_file = '../etc/AIDA-YAGO2-dataset.tsv'
candidate_file = '../etc/ppr_candidate'
word_file = '../etc/conll_word_vocab'
entity_file = '../etc/conll_entity_vocab'
mention_file = '../etc/conll_mention_vocab'
loadWiki(wiki_file)
loadCand(candidate_file)
buildVocab(data_file)
output(word_file, entity_file, mention_file)
