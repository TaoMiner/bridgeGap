import  codecs
import regex as re

wiki_id_file = '/Volumes/cyx/document_rpi/wiki20160305/wiki_id'
wiki_redirected_file = '/Volumes/cyx/document_rpi/wiki20160305/redirect-en'
outlink_file = ''
text_file = ''
wiki_tilte = {}
wiki_redirects = {}
wiki_map = {}
del_title = set()
redirect_id = {}

def loadWikiTitle(file):
    count = 0
    del_count = 0
    ignored_wiki_title = {}
    with codecs.open(wiki_id_file, 'r', 'utf-8') as fin:
        for line in fin:
            line = line.strip()
            items = re.split(r'\t', line)
            if re.match(r'(.*):(.*)', items[0]):
                if items[0] not in ignored_wiki_title:
                    ignored_wiki_title[items[0]] = items[1]
                count += 1
                continue
            if items[0] not in wiki_tilte:
                wiki_tilte[items[0]] = items[1]
            else:
                print 'error!%s' % line
            lowered_title = items[0].lower()
            if lowered_title in del_title:
                continue
            if lowered_title not in wiki_map:
                wiki_map[lowered_title] = items[0]
            elif wiki_map[lowered_title] != items[0]:
                del wiki_map[lowered_title]
                del_count +=1
                del_title.add(lowered_title)
        print 'successfully load %d wiki tities, ignored %d entries!' % (len(wiki_tilte), count)
    print 'learned %d maps!' % len(wiki_map)
    with codecs.open('wiki_title_cl', 'w', 'utf-8') as fout:
        for t in wiki_tilte:
            fout.write('%s\t%s\n' % (t, wiki_tilte[t]))
    with codecs.open('wiki_title_ignored', 'w', 'utf-8') as fout:
        for it in ignored_wiki_title:
            fout.write('%s\t%s\n' % (it, ignored_wiki_title[it]))


def loadRedirect(file):
    count = 0
    with codecs.open(wiki_redirected_file, 'r', 'utf-8') as fin:
        for line in fin:
            line = line.strip()
            items = re.split(r'\t', line)
            if len(items) == 3 and items[2] in wiki_tilte:
                if items[1] not in wiki_redirects:
                    wiki_redirects[items[1]] = items[2]
                    redirect_id[items[1]] = items[0]
                elif wiki_redirects[items[1]] != items[2]:
                    count += 1
                    del wiki_redirects[items[1]]
        print 'successfully load %d wiki redirects, ignored %d entries!' % (len(wiki_redirects), count)
    with codecs.open('wiki_redirect_cl', 'w', 'utf-8') as fout:
        for rt in wiki_redirects:
            fout.write('%s\t%s\t%s\n' % (redirect_id[rt], rt, wiki_redirects[rt]))
    for rt in wiki_redirects:
        lowered_title = rt.lower()
        if lowered_title in del_title:
            continue
        if lowered_title not in wiki_map:
            wiki_map[lowered_title] = items[0]
        elif wiki_map[lowered_title] != items[0]:
            del wiki_map[lowered_title]
            del_title.add(lowered_title)
    print 'learned %d maps!' % len(wiki_map)

loadWikiTitle(wiki_id_file)
loadRedirect(wiki_redirected_file)
with codecs.open('lower_map', 'w', 'utf-8') as fout:
    for m in wiki_map:
        fout.write('%s\t%s\n' % (m, wiki_map[m]))
