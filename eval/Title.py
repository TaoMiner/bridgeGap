import codecs
import struct
import numpy as np
import regex as re

class Sense():

    def __init__(self, size):
        self.size = size
        self.vector = []
        self.mu = []        #sense cluster center

class Title():

    def __init__(self):
        self.vocab_size = 0
        self.layer_size = 0
        self.max_sense_num = 0
        self.ent_vocab_size = 0
        self.title_ent = {}
        self.ent_title = {}
        self.vectors = {}
        self.ent_vectors = {}
        self.ent_mu = {}

    def initVectorFormat(self, size):
        tmp_struct_fmt = []
        for i in xrange(size):
            tmp_struct_fmt.append('f')
        p_struct_fmt = "".join(tmp_struct_fmt)
        return p_struct_fmt

    # return all titles with input entities
    def readEntityVocab(self, file):
        vocab = set()
        with codecs.open(file, 'r', encoding='UTF-8') as fin:
            for line in fin:
                items = re.split(r'\t', line.strip())
                if items[0] in self.ent_title:
                    vocab.add(self.ent_title[items[0]])
        print 'read %d titles for entities!' % len(vocab)
        return vocab

    def readTitleVocab(self, file):
        vocab = set()
        with codecs.open(file, 'r', encoding='UTF-8') as fin:
            for line in fin:
                items = re.split(r'\t', line.strip())
                if items[0] in self.title_ent:
                    vocab.add(items[0])
        print 'read %d titles!' % len(vocab)
        return vocab

    def sample(self, entity_file, mention_file, input_file,  output_file):
        self.loadVector(input_file)
        new_title_vocab = self.readTitleVocab(mention_file)
        print 'new title vocab size %d!' % len(new_title_vocab)
        new_title_vocab.update(self.readEntityVocab(entity_file))
        print 'new title vocab size %d add entity title!' % len(new_title_vocab)
        new_vocab_size = len(new_title_vocab)
        new_title_count = 0
        with codecs.open(input_file, 'rb') as fin_vec:
            with codecs.open(output_file, 'wb') as fout_vec:
                # read file head: vocab size and layer size
                char_set = []
                var_pos = 0
                while True:
                    ch = fin_vec.read(1)
                    if ch == ' ' or ch == '\n':
                        var_pos += 1
                        if var_pos == 1:
                            vocab_size = (int)("".join(char_set))
                        elif var_pos == 2:
                            layer_size = (int)("".join(char_set))
                        else:
                            max_sense_num = (int)("".join(char_set))
                        del char_set[:]
                        if ch == '\n': break
                        continue
                    char_set.append(ch)
                fout_vec.write("%d %d %d\n" % (new_vocab_size, layer_size, max_sense_num))
                for i in xrange(vocab_size):
                    # read title label
                    del char_set[:]
                    var_pos = 0
                    title = ''
                    sense_num = 0
                    entity_num = 0
                    while True:
                        ch = struct.unpack('c', fin_vec.read(1))[0]
                        if ch == '\t':
                            var_pos += 1
                            if var_pos == 1:
                                title = "".join(char_set).decode('ISO-8859-1')
                                del char_set[:]
                                continue
                            elif var_pos == 2:
                                sense_num = int("".join(char_set))
                                del char_set[:]
                            else:
                                entity_num = int("".join(char_set))
                                break
                        char_set.append(ch)
                    if title in new_title_vocab:
                        fout_vec.write("%s\t%d\t%d\t\n" % (title.encode('ISO-8859-1'), 0, entity_num))
                        new_title_count += 1
                    # read sense
                    for j in xrange(sense_num):
                        fin_vec.read(4 * self.layer_size)
                        fin_vec.read(4 * self.layer_size)
                    fin_vec.read(1)  # \n
                    # read entity sense
                    # read entity label
                    for j in xrange(entity_num):
                        del char_set[:]
                        entity_label = ''
                        while True:
                            tmp_c = fin_vec.read(1)
                            if not tmp_c: break
                            ch = struct.unpack('c', tmp_c)[0]
                            if ch == '\t':
                                break
                            char_set.append(ch)
                        entity_label = "".join(char_set).decode('ISO-8859-1')
                        if title in new_title_vocab:
                            fout_vec.write("%s\t" % entity_label.encode('ISO-8859-1'))
                            fout_vec.write(fin_vec.read(4 * layer_size))
                            fout_vec.write(fin_vec.read(4 * layer_size))
                            fout_vec.write(fin_vec.read(1))
                        else:
                            fin_vec.read(4 * layer_size)
                            fin_vec.read(4 * layer_size)
                            fin_vec.read(1)  # \n
        print 'sample %d titles!' % new_title_count

    def loadVector(self, filename):
        with codecs.open(filename, 'rb') as fin_vec:
            # read file head: vocab size, layer size and max_sense_num
            char_set = []
            var_pos = 0
            while True:
                ch = fin_vec.read(1)
                if ch==' ' or ch=='\t' or ch == '\n':
                    var_pos += 1
                    if var_pos == 1:
                        self.vocab_size = (int)("".join(char_set))
                    elif var_pos == 2:
                        self.layer_size = (int)("".join(char_set))
                    else:
                        self.max_sense_num = (int)("".join(char_set))
                    del char_set[:]
                    if ch == '\n' : break
                    continue
                char_set.append(ch)
            p_struct_fmt = self.initVectorFormat(self.layer_size)
            for i in xrange(self.vocab_size):
                # read title label
                del char_set[:]
                var_pos = 0
                label = ''
                sense_num = 0
                entity_num = 0
                while True:
                    ch = struct.unpack('c',fin_vec.read(1))[0]
                    if ch=='\t' or ch=='\n':
                        var_pos += 1
                        if var_pos == 1:
                            label = "".join(char_set).decode('ISO-8859-1')
                            del char_set[:]
                            continue
                        elif var_pos == 2:
                            sense_num = int("".join(char_set))
                            sense = Sense(sense_num)
                            del char_set[:]
                        else:
                            entity_num = int("".join(char_set))
                            break
                    char_set.append(ch)
                # read sense
                sense_skip = 0
                for j in xrange(sense.size):
                    tmp_vec = np.array(struct.unpack(p_struct_fmt, fin_vec.read(4*self.layer_size)), dtype=float)
                    tmp_mu = np.array(struct.unpack(p_struct_fmt, fin_vec.read(4*self.layer_size)), dtype=float)
                    if not np.isnan(tmp_vec).any() and not np.isinf(tmp_mu).any():
                        sense.vector.append(tmp_vec)
                        sense.mu.append(tmp_mu)
                    else:
                        sense_skip +=1
                sense.size -= sense_skip
                if sense.size > 0:
                    self.vectors[label] = sense
                fin_vec.read(1)     #\n
                #read entity sense
                #read entity label
                title_entity_set = set()
                for j in xrange(entity_num):
                    del char_set[:]
                    entity_label = ''
                    while True:
                        tmp_c = fin_vec.read(1)
                        if not tmp_c : break
                        ch = struct.unpack('c', tmp_c)[0]
                        if ch=='\t':
                            break
                        char_set.append(ch)
                    if len(char_set) < 1:
                        print 'error load entity label of title: %s!' % label
                        break
                    entity_label = "".join(char_set).decode('ISO-8859-1')
                    tmp_vec = np.array(struct.unpack(p_struct_fmt, fin_vec.read(4 * self.layer_size)), dtype=float)
                    tmp_mu = np.array(struct.unpack(p_struct_fmt, fin_vec.read(4 * self.layer_size)), dtype=float)
                    #if not np.isnan(tmp_vec).any() and not np.isinf(tmp_mu).any():
                    title_entity_set.add(entity_label)
                    self.ent_title[entity_label] = label
                    self.ent_vectors[entity_label] = tmp_vec
                    self.ent_mu[entity_label] = tmp_mu
                    fin_vec.read(1)  # \n
                if len(title_entity_set) > 0:
                    self.title_ent[label] = title_entity_set
            self.vocab_size = len(self.title_ent)
            self.ent_vocab_size = len(self.ent_vectors)
        print 'load %d title vectors with %d mention sense vectors!' % (self.vocab_size, self.ent_vocab_size)

if __name__ == '__main__':
    title_vector_file = '/data/m1/cyx/etc/output/exp10/vectors_title10.dat'
    sample_file = '/data/m1/cyx/etc/expdata/conll/vectors_title.sample'
    entity_conll_file = '/data/m1/cyx/etc/expdata/conll/conll_entity_vocab'
    title_conll_file = '/data/m1/cyx/etc/expdata/conll/conll_mention_vocab'
    wiki_title = Title()
    wiki_title.sample(entity_conll_file, title_conll_file, title_vector_file, sample_file)