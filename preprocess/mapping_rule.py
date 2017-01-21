import  codecs
import regex as re
import string
from nltk.corpus import stopwords

wiki_id_file = '/Volumes/cyx/document_rpi/wiki20160305wiki_title_cl'
vocab_title = '/Volumes/cyx/document_rpi/wiki20160305/vocab_title.txt'
map_file = '/Volumes/cyx/document_rpi/wiki20160305lower_map'
count_file = '/Volumes/cyx/document_rpi/wiki20160305count_mentions'
mapping_file = '/Volumes/cyx/document_rpi/wiki20160305mapping_pair'
mention_list = '/Volumes/cyx/document_rpi/wiki20160305mention_list'

titles = set()
wiki_tilte = {}
wiki_map = {}
mentions = {}
stop_words = set(stopwords.words('english'))

punc = re.compile('[%s]' % re.escape(string.punctuation))

def loadVocab(file):
    with codecs.open(file, 'r', 'utf-8') as fin:
        for line in fin:
            line = line.strip()
            items = re.split(r'\t', line)
            titles.add(items[0])
    print 'successfully load %d vocab titles!' % len(titles)

def loadWikiTitle(file):
    with codecs.open(file, 'r', 'utf-8') as fin:
        for line in fin:
            line = line.strip()
            items = re.split(r'\t', line)
            if len(items[0]) > 0:
                tmp_t = re.sub(r'\(.*?\)$', '', items[0])
                if tmp_t in titles:
                    wiki_tilte[items[0]] = tmp_t
        print 'successfully load %d wiki tities!' % len(wiki_tilte)

def loadMap(file):
    with codecs.open(file, 'r', 'utf-8') as fin:
        for line in fin:
            line = line.strip()
            items = re.split(r'\t', line)
            if len(items) == 2 and items[1] in wiki_tilte and len(items[0]) > 0:
                items[0] = rule(items[0])
                tmp_ments = set() if items[0] not in mentions else mentions[items[0]]
                tmp_ments.add(wiki_tilte[items[1]])
                mentions[items[0]] = tmp_ments
        print 'successfully load %d lower maps!' % len(mentions)

def rule(str):
    # possessive case 's
    # tmp_line = re.sub(r' s |\'s', ' ', str)
    # following clean wiki xml, punctuation, numbers, and lower case
    tmp_line = punc.sub(' ', str)
    tmp_line = tmp_line.replace('\t', ' ')
    tmp_line = re.sub(r'[\s]+', ' ', tmp_line)
    tmp_line = re.sub(r'(?<=\s)(\d+)(?=($|\s))', 'dddddd', tmp_line)
    tmp_line = re.sub(r'(?<=^)(\d+)(?=($|\s))', 'dddddd', tmp_line).lower().strip()
    return tmp_line

def loadCount(file):
    with codecs.open(file, 'r', 'utf-8') as fin:
        for line in fin:
            line = line.strip()
            items = re.split(r'\t', line)
            if items[1] not in wiki_tilte or len(items)<3: continue
            else:
                tmp_title = wiki_tilte[items[1]]
                for m in items[2:]:
                    tmp_m = rule(m)
                    tmp_mset = set() if tmp_m not in mentions else mentions[tmp_m]
                    tmp_mset.add(tmp_title)
                    mentions[tmp_m] = tmp_mset
    sum_m = 0
    for m in mentions:
        sum_m += len(mentions[m])
    print 'successfully extract %d mentions! including %d titles!' % (len(mentions),sum_m)

#only mapping title
with codecs.open('./mapping_title', 'w', 'utf-8') as fout:
    with codecs.open(vocab_title, 'r', 'utf-8') as fin:
        for line in fin:
            line = line.strip()
            items = re.split(r'\t', line)
            tmp_ment = rule(items[0])
            if tmp_ment not in stop_words and len(tmp_ment) >2 and tmp_ment!='dddddd':
                fout.write('%s\t%s\n' % (tmp_ment,items[0]))
'''
loadVocab(vocab_title)
loadWikiTitle(wiki_id_file)
loadMap(map_file)
loadCount(count_file)
with codecs.open(mapping_file, 'w', 'utf-8') as fout:
    with codecs.open(mention_list, 'w', 'utf-8') as fout_list:
        for m in mentions:
            fout.write('%s\t%s\n' % (m,'\t'.join(mentions[m])))
            fout_list.write(m)
'''