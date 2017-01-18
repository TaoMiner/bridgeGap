import  codecs
import regex as re
from itertools import izip, izip_longest

wiki_id_file = '/data/m1/cyx/etc/enwiki2/wiki_title_cl'
wiki_redirected_file = '/data/m1/cyx/etc/enwiki2/wiki_redirect_cl'
map_file = '/data/m1/cyx/etc/enwiki2/lower_map'
outlink_file = '/data/m1/cyx/etc/enwiki2/enwiki-outlink.dat'
text_file = '/data/m1/cyx/etc/enwiki2/train_text'
anchor_file = '/data/m1/cyx/etc/enwiki2/train_anchor'
wiki_tilte = {}
wiki_redirects = {}
wiki_map = {}
redirect_id = {}
mentions = {}

def loadWikiTitle(file):
    with codecs.open(file, 'r', 'utf-8') as fin:
        for line in fin:
            line = line.strip()
            items = re.split(r'\t', line)
            if len(items[0]) > 0:
                wiki_tilte[items[0]] = items[1]
        print 'successfully load %d wiki tities!' % len(wiki_tilte)

def loadRedirect(file):
    with codecs.open(file, 'r', 'utf-8') as fin:
        for line in fin:
            line = line.strip()
            items = re.split(r'\t', line)
            if len(items) == 3 and items[2] in wiki_tilte and len(items[1]) > 0:
                if items[1] not in wiki_redirects:
                    wiki_redirects[items[1]] = items[2]
                    redirect_id[items[1]] = items[0]
        print 'successfully load %d wiki redirects!' % len(wiki_redirects)

def loadMap(file):
    with codecs.open(file, 'r', 'utf-8') as fin:
        for line in fin:
            line = line.strip()
            items = re.split(r'\t', line)
            if len(items) == 2 and items[1] in wiki_tilte and len(items[0]) > 0:
                if items[0] not in wiki_map:
                    wiki_map[items[0]] = items[1]
        print 'successfully load %d lower maps!' % len(wiki_map)

def findBalanced(text, openDelim=['[['], closeDelim=[']]']):
    """
    Assuming that text contains a properly balanced expression using
    :param openDelim: as opening delimiters and
    :param closeDelim: as closing delimiters.
    :return: an iterator producing pairs (start, end) of start and end
    positions in text containing a balanced expression.
    """
    openPat = '|'.join([re.escape(x) for x in openDelim])
    afterPat = dict()
    for o,c in izip(openDelim, closeDelim):
        afterPat[o] = re.compile(openPat + '|' + c, re.DOTALL)
    stack = []
    start = 0
    cur = 0
    # end = len(text)
    startSet = False
    startPat = re.compile(openPat)
    nextPat = startPat
    while True:
        next = nextPat.search(text, cur)
        if not next:
            return
        if not startSet:
            start = next.start()
            startSet = True
        delim = next.group(0)
        if delim in openDelim:
            stack.append(delim)
            nextPat = afterPat[delim]
        else:
            opening = stack.pop()
            # assert opening == openDelim[closeDelim.index(next.group(0))]
            if stack:
                nextPat = afterPat[stack[-1]]
            else:
                yield start, next.end()
                nextPat = startPat
                start = next.end()
                startSet = False
        cur = next.end()

def processAnchors(file, output_file, count_file):
    anchor_count = 0
    with codecs.open(file, 'r', 'utf-8') as fin:
        with codecs.open(output_file, 'w', 'utf-8') as fout:
            with codecs.open(anchor_file, 'w', 'utf-8') as fout_anchor:
                for line in fin:
                    cur = 0
                    res = ''
                    res_anchor = ''
                    line = line.strip()
                    for s, e in findBalanced(line):
                        res += line[cur:s]
                        res_anchor += line[cur:s]
                        tmp_anchor = line[s:e]
                        # extract title and label
                        tmp_vbar = tmp_anchor.find('|')
                        tmp_title = ''
                        tmp_label = ''
                        if tmp_vbar > 0:
                            tmp_title = tmp_anchor[2:tmp_vbar]
                            tmp_label = tmp_anchor[tmp_vbar+1:-2]
                        else:
                            tmp_title = tmp_anchor[2:-2]
                            tmp_label = tmp_title
                        # map the right title
                        if tmp_title not in wiki_tilte and tmp_title not in wiki_redirects and tmp_title.lower() not in wiki_map:
                            tmp_anchor = tmp_label
                        else:
                            if tmp_title in wiki_redirects:
                                tmp_title = wiki_redirects[tmp_title]
                            elif tmp_title.lower() in wiki_map:
                                tmp_title = wiki_map[tmp_title.lower()]
                            if tmp_title == tmp_label:
                                tmp_anchor = '[[' + tmp_title + ']]'
                            else:
                                tmp_anchor = '[[' + tmp_title + '|' + tmp_label + ']]'
                            anchor_count += 1
                            if anchor_count % 100000 == 0:
                                print 'has processed %d anchors!' % anchor_count
                            # count the mentions
                            tmp_mention = {} if tmp_title not in mentions else mentions[tmp_title]
                            if tmp_label in tmp_mention:
                                tmp_mention[tmp_label] += 1
                            else:
                                tmp_mention[tmp_label] = 1
                            mentions[tmp_title] = tmp_mention

                        res += tmp_anchor
                        res_anchor += tmp_label
                        cur = e
                    res += line[cur:] + '\n'
                    res_anchor += line[cur:] + '\n'
                    if len(res) > 10:
                        fout.write(res)
                    if len(res_anchor) > 10:
                        fout_anchor.write(res_anchor)
    print 'process train text finished! start count %d mentions ...' % anchor_count
    with codecs.open(count_file, 'w', 'utf-8') as fout:
        out_list = []
        for t in mentions:
            out_list.append(wiki_tilte[t] + '\t' + t + "\t" + "\t".join(
                ["%s::=%s" % (k, v) for k, v in mentions[t].items()]) + "\n")
            if len(out_list) >= 10000:
                fout.writelines(out_list)
                del out_list[:]
        if len(out_list) > 0:
            fout.writelines(out_list)
    print 'count mentions finished!'




def processOutlinks(file, output_file):
    with codecs.open(file, 'r', 'utf-8') as fin:
        with codecs.open(output_file, 'w', 'utf-8') as fout:
            for line in fin:
                out_list = []
                line = line.strip()
                items = re.split(r'\t|;', line)
                if len(items) <= 1: continue
                if items[0] not in wiki_tilte and items[0] not in wiki_redirects and items[0].lower() not in wiki_map:
                    continue
                else:
                    for item in items:
                        if item in wiki_redirects:
                            item = wiki_redirects[item]
                        elif item.lower() in wiki_map:
                            item = wiki_map[item.lower()]
                        elif item not in wiki_tilte:
                            continue
                        out_list.append(item)
                if len(out_list) > 1:
                    fout.write('%s\n' % '\t'.join(out_list))
    print 'outlinks processing finished!'

loadWikiTitle(wiki_id_file)
loadRedirect(wiki_redirected_file)
loadMap(map_file)
#processAnchors(text_file,'/data/m1/cyx/etc/enwiki2/train_text_cl', '/data/m1/cyx/etc/enwiki2/count_mentions')
processOutlinks(outlink_file, '/data/m1/cyx/etc/enwiki2/train_kg_cl2')
