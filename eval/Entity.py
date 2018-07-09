import codecs
import struct
import numpy as np
import regex as re

class Entity():

    def __init__(self):
        self.vocab_size = 0
        self.layer_size = 0
        self.wiki_id = {}
        self.id_wiki = {}
        self.vectors = {}


    def loadWikiId(self, filename):
        with codecs.open(filename, 'r', encoding='UTF-8') as fin:
            for line in fin:
                items = re.split(r'\t', line.strip())
                if len(items) < 2 or items[0]=="" or items[0] ==" ": continue
                self.wiki_id[items[0]] = items[1]
                self.id_wiki[items[1]] = items[0]
        print 'load %d wiki id!' % len(self.wiki_id)

    def initVectorFormat(self, size):
        tmp_struct_fmt = []
        for i in xrange(size):
            tmp_struct_fmt.append('f')
        p_struct_fmt = "".join(tmp_struct_fmt)
        return p_struct_fmt

    def readVocab(self, file):
        vocab = set()
        with codecs.open(file, 'r', encoding='UTF-8') as fin:
            for line in fin:
                items = re.split(r'\t', line.strip())
                if items[0] in self.vectors:
                    vocab.add(items[0])
        print 'read %d entities!' % len(vocab)
        return vocab

    def sample(self, file, input_file,  output_file):
        self.loadVector(input_file)
        new_vocab = self.readVocab(file)
        new_vocab_size = len(new_vocab)
        new_word_count = 0
        print 'new vocab size: %d\n' % new_vocab_size
        with codecs.open(input_file, 'rb') as fin_vec:
            with codecs.open(output_file, 'wb') as fout_vec:
                # read file head: vocab size and layer size
                char_set = []
                while True:
                    ch = fin_vec.read(1)
                    if ch == ' ':
                        vocab_size = (int)("".join(char_set))
                        del char_set[:]
                        continue
                    if ch == '\n':
                        layer_size = (int)("".join(char_set))
                        break
                    char_set.append(ch)
                fout_vec.write("%d %d\n" % (new_vocab_size, layer_size))
                for i in xrange(vocab_size):
                    # read entity label
                    del char_set[:]
                    while True:
                        ch = struct.unpack('c', fin_vec.read(1))[0]
                        if ch == '\t':
                            break
                        char_set.append(ch)
                    label = "".join(char_set).decode('ISO-8859-1')
                    if label in new_vocab:
                        new_word_count += 1
                        fout_vec.write("%s\t" % label.encode('ISO-8859-1'))
                        fout_vec.write(fin_vec.read(4 * layer_size))
                        fout_vec.write(fin_vec.read(1))
                    else:
                        fin_vec.read(4 * layer_size)
                        fin_vec.read(1)  # \n
        print 'sample %d entities!' % new_word_count

    def loadVector(self, filename):
        with codecs.open(filename, 'rb') as fin_vec:
            # read file head: vocab size and layer size
            char_set = []
            while True:
                ch = fin_vec.read(1)
                if ch==' ' or ch=='\t':
                    self.vocab_size = (int)("".join(char_set))
                    del char_set[:]
                    continue
                if ch=='\n':
                    self.layer_size = (int)("".join(char_set))
                    break
                char_set.append(ch)
            p_struct_fmt = self.initVectorFormat(self.layer_size)
            for i in xrange(self.vocab_size):
                # read entity label
                del char_set[:]
                while True:
                    tmp_c = fin_vec.read(1)
                    if not tmp_c : break
                    ch = struct.unpack('c', tmp_c)[0]
                    if ch=='\t':
                        break
                    char_set.append(ch)
                if len(char_set) < 1: break
                label = "".join(char_set).decode('ISO-8859-1')
                self.vectors[label] = np.array(struct.unpack(p_struct_fmt, fin_vec.read(4*self.layer_size)), dtype=float)
                fin_vec.read(1)     #\n
            self.vocab_size = len(self.vectors)
        print 'load %d entity vectors!' % len(self.vectors)

if __name__ == '__main__':
    entity_vector_file = '/data/m1/cyx/etc/output/exp10/vectors_entity10.dat'
    sample_file = '/data/m1/cyx/etc/expdata/conll/vectors_entity.sample'
    vocab_conll_file = '/data/m1/cyx/etc/expdata/conll/conll_entity_vocab'
    wiki_entity = Entity()
    wiki_entity.sample(vocab_conll_file, entity_vector_file, sample_file)