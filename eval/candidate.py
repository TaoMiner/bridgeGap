import codecs
import regex as re
import os
try:
    import xml.etree.cElementTree as ET
except ImportError:
    import xml.etree.ElementTree as ET


class Candidate:
    "candidate generation"

    candidate = {}      #{mention:{wiki_id\t...},...}
    ids = set()
    wiki_id = {}

    def loadWikiId(self, filename):
        with codecs.open(filename, 'r', encoding='UTF-8') as fin:
            for line in fin:
                items = re.split(r'\t', line.strip())
                if len(items) < 2 : continue
                self.ids.add(items[1])
                self.wiki_id[items[0]] = items[1]

    def findPPRCand(self, path):
        if not os.path.isdir(path):
            return
        for root, dirs, list in os.walk(path):
            for dir in dirs:
                self.findPPRCand(os.path.join(root, dir))
            for l in list:
                if not l.startswith('.'):
                    with codecs.open(os.path.join(root, l), 'r', encoding='UTF-8') as fin:
                        mention_name = ''
                        for line in fin:
                            line = line.strip()
                            if line.startswith('ENTITY'):
                                mention_name = re.search(r'(?<=text:).*?(?=\t)', line).group()
                                if mention_name not in self.candidate:
                                    self.candidate[mention_name] = set()
                            if line.startswith('CANDIDATE') and mention_name != '':
                                id = re.search(r'(?<=id:).*?(?=\t)', line).group()
                                if id in self.ids:
                                    self.candidate[mention_name].add(id)

    def findYagoCand(self, filename):
        with codecs.open(filename, 'r', encoding='UTF-8') as fin:
            for line in fin:
                items = re.split(r'\t',line.strip().lower().decode('unicode_escape'))
                items[1] = items[1].replace('_',' ')
                if items[1] in self.wiki_id:
                    tmp_id = self.wiki_id[items[1]]
                    tmp_m = re.search(r'(?<=").*?(?=")', items[0]).group()
                    if tmp_m in self.candidate:
                        cand = self.candidate[tmp_m]
                    else:
                        cand = set()
                    cand.add(tmp_id)
                    self.candidate[tmp_m] = cand

    def findWikiCand(self, anchor_file, redirect_file, wiki_file):
        with codecs.open(anchor_file, 'r', encoding='UTF-8') as fin:
            for line in fin:
                items = re.split(r'\t', line.strip().lower())
                if len(items) <= 2: continue
                for m in items[2:]:
                    m_text = m.replace('=[\d]+$', '')
                    tmp_cand = self.candidate[m_text] if m_text in self.candidate else set()
                    tmp_cand.add(items[0])
                    self.candidate[m_text] = tmp_cand
            print("load %d mention from anchors!" % len(self.candidate))
        if len(self.wiki_id) < 1:
            self.loadWikiId(wiki_file)
        with codecs.open(redirect_file, 'r', encoding='UTF-8') as fin:
            for line in fin:
                items = re.split(r'\t\t', line.strip().lower())
                if len(items)<2:continue
                tmp_cand = self.candidate[items[0]] if items[0] in self.candidate else set()
                if items[1] not in self.wiki_id:
                    items[1] = items[1].replace('_', ' ')
                    if items[1] not in self.wiki_id: continue
                tmp_cand.add(self.wiki_id[items[1]])
                self.candidate[items[0]] = tmp_cand
            print("load %d mention from redirect pages!" % len(self.candidate))
        with codecs.open(wiki_file, 'r', encoding='UTF-8') as fin:
            for line in fin:
                items = re.split(r'\t\t', line.strip().lower())
                tmp_cand = self.candidate[items[0]] if items[0] in self.candidate else set()
                tmp_cand.add(items[1])
                self.candidate[items[0]] = tmp_cand
            print("load %d mention from wiki pages!" % len(self.candidate))

    def saveCandidates(self, filename):
        with codecs.open(filename, 'w', encoding='UTF-8') as fout:
            count = 0
            for mention in self.candidate:
                if len(self.candidate[mention])>0 and len(mention)>1:
                    count += len(self.candidate[mention])
                    fout.write("%s\t%s\n" % (mention, '\t'.join(self.candidate[mention])))
        return count

if __name__ == '__main__':
    wiki_id_file = '/Users/ethan/Downloads/datasets/wiki/enwiki-ID.dat'
    conll_candidate = Candidate()
    conll_candidate.loadWikiId(wiki_id_file)
    print("successfully load %d wiki id!" %  len(conll_candidate.wiki_id))
    #yago_candidate
    '''
    output_file = './yago_candidate'
    aida_mean_file = ''
    conll_candidate.findYagoCand(aida_mean_file)
    print("successfully load %d mentions" %  len(conll_candidate.candidate))
    count = conll_candidate.saveCandidates(output_file)
    print("total %d candidates, each mention has %d candidates on average." % (count, count/len(conll_candidate.candidate)))
    '''
    #ppr_candidate
    output_file = './ppr_candidate'
    ppr_path = ''
    conll_candidate.findPPRCand(ppr_path)
    print("successfully load %d mentions" %  len(conll_candidate.candidate))
    count = conll_candidate.saveCandidates(output_file)
    print("total %d candidates, each mention has %d candidates on average." % (count, count/len(conll_candidate.candidate)))
    #wiki_candidate
    '''
    output_file = './wiki_candidate'
    anchor_file = ''
    redirect_file = ''
    conll_candidate.findWikiCand(anchor_file, redirect_file, wiki_id_file)
    print("successfully load %d mentions" %  len(conll_candidate.candidate))
    count = conll_candidate.saveCandidates(output_file)
    print("total %d candidates, each mention has %d candidates on average." % (count, count/len(conll_candidate.candidate)))
    '''

