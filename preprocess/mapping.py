import codecs
import regex as re

class Map:

    def __init__(self):
        self.mention_names = {}
        self.wiki_id = {}
        self.id_wiki = {}

    def loadWikiID(self, file):
        with codecs.open(file, 'r', 'utf-8') as fin:
            for line in fin:
                items = re.split(r'\t', line.strip())
                if len(items) != 2: continue
                self.wiki_id[items[0]] = items[1]
                self.id_wiki[items[1]]  =items[0]

    def loadMap(self, file):
        with codecs.open(file, 'r', 'utf-8') as fin:
            for line in fin:
                items = re.split(r'\t', line.strip())
                if len(items) == 2 and items[1] in self.wiki_id and len(items[0]) > 0:
                    tmp_ments = set() if items[0] not in self.mention_names else self.mention_names[items[0]]
                    tmp_ments.add(self.wiki_id[items[1]])
                    self.mention_names[items[0]] = tmp_ments
            print 'successfully load %d lower maps!' % len(self.mention_names)

    def loadAnchor(self, file):
        with codecs.open(file, 'r', 'utf-8') as fin:
            for line in fin:
                items = re.split(r'\t', line.strip())
                if len(items) < 3 or items[1] not in self.wiki_id: continue
                for item in items[2:]:
                    tmp_items = re.split(r'::=', item)
                    if len(tmp_items) != 2 or int(tmp_items[1]) < 5: continue
                    tmp_ments = set() if tmp_items[0] not in self.mention_names else self.mention_names[tmp_items[0]]
                    tmp_ments.add(items[0])
                    self.mention_names[tmp_items[0]] = tmp_ments
        print 'successfully load %d anchor mentions!' % len(self.mention_names)

    def build(self, output_file):
        wiki_id_file = '/data/m1/cyx/etc/enwiki/wiki_title_cl'
        map_file = '/data/m1/cyx/etc/enwiki/lower_map'
        anchor_count_file = '/data/m1/cyx/etc/enwiki/count_mentions'
        self.loadWikiID(wiki_id_file)
        self.loadMap(map_file)
        self.loadAnchor(anchor_count_file)
        with codecs.open(output_file, 'w', 'utf-8') as fout:
            for m in self.mention_names:
                if len(self.mention_names[m]) > 0:
                    fout.write('%s\t%s\n' % (m, '\t'.join(self.mention_names[m])))

    def load(self, file):
        with codecs.open(file, 'r', 'utf-8') as fin:
            for line in fin:
                items = re.split(r'\t',line.strip)
                if len(items)<2: continue
                tmp_set = set()
                for item in items[1:]:
                    if item in self.id_wiki:
                        tmp_set.add(self.id_wiki[item])
                self.mention_names[items[0]] = tmp_set

if __name__ == '__main__':
    mention_name_file = ''
    mention_map = Map()
    mention_map.load(mention_name_file)