import os
import codecs
import regex as re
import string
punc = re.compile('[%s]' % re.escape(string.punctuation))

def processFile(file):
    with codecs.open(file, 'r', 'utf-8') as fin:
        doc = []
        for line in fin:
            tmp_line = []
            items = re.split(r' ',line.strip())
            if len(items) < 5: continue
            for item in items:
                tmp_word = ''
                tmp_items = re.split(r'#',item)
                if len(tmp_items)!=2:continue
                if tmp_items[1] == 'NUMBER':
                    tmp_word = 'dddddd'
                else:
                    tmp_word = re.sub(r'_',' ',tmp_items[0])
                    tmp_word = punc.sub(' ', tmp_word)
                    tmp_word = tmp_word.replace('\t', ' ')
                    tmp_word = re.sub(r'[\s]+', ' ', tmp_word)
                    tmp_word = tmp_word.strip()
                if tmp_word == 'dddddd' and tmp_line[-1] == 'dddddd':
                    continue
                tmp_line.append(tmp_word)
            if len(tmp_line) > 4:
                doc.append(' '.join(tmp_line))
    return '\n'.join(doc)

def browser(path, fout):
    if not os.path.isdir(path):
        return
    for root, dirs, list in os.walk(path):
        for d in dirs:
            browser(os.path.join(root,d),fout)
        for i in list:
            file = os.path.join(root, i)
            fout.write('%s\n' % processFile(file))

output_file = '/data/m1/cyx/etc/giga/giga_corpus.dat'
input_path = '/data/m1/en_giga_processed/'
with codecs.open(output_file, 'w', 'utf-8') as fout:
    browser(input_path,fout)