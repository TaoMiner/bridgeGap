import  codecs
import regex as re
import numpy as np

wiki_id_file = '/data/m1/cyx/etc/enwiki2/wiki_title_cl'
wiki_redirected_file = '/data/m1/cyx/etc/enwiki2/wiki_redirect_cl'
map_file = '/data/m1/cyx/etc/enwiki2/lower_map'
count_file = '/data/m1/cyx/etc/enwiki2/count_mentions'
wiki_tilte = {}
wiki_redirects = {}
wiki_map = {}
redirect_id = {}
mentions = {}
title_count = {}

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
                    tmp_ments = set() if items[2] not in mentions else mentions[items[2]]
                    tmp_ments.add(items[1])
                    mentions[items[2]] = tmp_ments
        print 'successfully load %d wiki redirects!' % len(wiki_redirects)

def loadMap(file):
    with codecs.open(file, 'r', 'utf-8') as fin:
        for line in fin:
            line = line.strip()
            items = re.split(r'\t', line)
            if len(items) == 2 and items[1] in wiki_tilte and len(items[0]) > 0:
                if items[0] not in wiki_map:
                    wiki_map[items[0]] = items[1]
                    tmp_ments = set() if items[1] not in mentions else mentions[items[1]]
                    tmp_ments.add(items[0])
                    mentions[items[1]] = tmp_ments
        print 'successfully load %d lower maps!' % len(wiki_map)

def loadCount(file):
    with codecs.open(file, 'r', 'utf-8') as fin:
        for line in fin:
            count = 0
            line = line.strip()
            items = re.split(r'\t', line)
            tmp_ments = set() if items[1] not in mentions else mentions[items[1]]
            for item in items[2:]:
                tmp_items = re.split(r'::=', item)
                if len(tmp_items) == 2:
                    tmp_ments.add(tmp_items[0])
                    count += int(tmp_items[1])
            mentions[items[1]] = tmp_ments
            if items[1] in title_count:
                title_count[items[1]] += count
            else:
                title_count[items[1]] = count

def statistic():
    # 0: total mention num, 1: total anchors, 2: total entities
    stat = np.zeros([18,3], dtype=int)
    for t in mentions:
        tmp_num = title_count[t] if t in title_count else 0
        if len(mentions[t]) > 170:
            stat[17][0] += len(mentions[t])
            stat[17][1] += tmp_num
            stat[17][2] += 1
        elif len(mentions[t]) <= 170 and len(mentions[t]) > 160:
            stat[16][0] += len(mentions[t])
            stat[16][1] += tmp_num
            stat[16][2] += 1
        elif len(mentions[t]) <= 160 and len(mentions[t]) > 150:
            stat[15][0] += len(mentions[t])
            stat[15][1] += tmp_num
            stat[15][2] += 1
        elif len(mentions[t]) <= 150 and len(mentions[t]) > 140:
            stat[14][0] += len(mentions[t])
            stat[14][1] += tmp_num
            stat[14][2] += 1
        elif len(mentions[t]) <= 140 and len(mentions[t]) > 130:
            stat[13][0] += len(mentions[t])
            stat[13][1] += tmp_num
            stat[13][2] += 1
        elif len(mentions[t]) <= 130 and len(mentions[t]) > 120:
            stat[12][0] += len(mentions[t])
            stat[12][1] += tmp_num
            stat[12][2] += 1
        elif len(mentions[t]) <= 120 and len(mentions[t]) > 110:
            stat[11][0] += len(mentions[t])
            stat[11][1] += tmp_num
            stat[11][2] += 1
        elif len(mentions[t]) <= 110 and len(mentions[t]) > 100:
            stat[10][0] += len(mentions[t])
            stat[10][1] += tmp_num
            stat[10][2] += 1
        elif len(mentions[t]) <= 100 and len(mentions[t]) > 90:
            stat[9][0] += len(mentions[t])
            stat[9][1] += tmp_num
            stat[9][2] += 1
        elif len(mentions[t]) <= 90 and len(mentions[t]) > 80:
            stat[8][0] += len(mentions[t])
            stat[8][1] += tmp_num
            stat[8][2] += 1
        elif len(mentions[t]) <= 80 and len(mentions[t]) > 70:
            stat[7][0] += len(mentions[t])
            stat[7][1] += tmp_num
            stat[7][2] += 1
        elif len(mentions[t]) <= 70 and len(mentions[t]) > 60:
            stat[6][0] += len(mentions[t])
            stat[6][1] += tmp_num
            stat[6][2] += 1
        elif len(mentions[t]) <= 60 and len(mentions[t]) > 50:
            stat[5][0] += len(mentions[t])
            stat[5][1] += tmp_num
            stat[5][2] += 1
        elif len(mentions[t]) <= 50 and len(mentions[t]) > 40:
            stat[4][0] += len(mentions[t])
            stat[4][1] += tmp_num
            stat[4][2] += 1
        elif len(mentions[t]) <= 40 and len(mentions[t]) > 30:
            stat[3][0] += len(mentions[t])
            stat[3][1] += tmp_num
            stat[3][2] += 1
        elif len(mentions[t]) <= 30 and len(mentions[t]) > 20:
            stat[2][0] += len(mentions[t])
            stat[2][1] += tmp_num
            stat[2][2] += 1
        elif len(mentions[t]) <= 20 and len(mentions[t]) > 10:
            stat[1][0] += len(mentions[t])
            stat[1][1] += tmp_num
            stat[1][2] += 1
        elif len(mentions[t]) <= 10 and len(mentions[t]) > 0:
            stat[0][0] += len(mentions[t])
            stat[0][1] += tmp_num
            stat[0][2] += 1
    print stat

loadWikiTitle(wiki_id_file)
loadRedirect(wiki_redirected_file)
loadMap(map_file)
loadCount(count_file)
statistic()