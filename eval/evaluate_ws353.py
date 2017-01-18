# coding: utf-8
from scipy import stats
from nltk.stem import WordNetLemmatizer
import math
import sys
import numpy
import random

WORD_FILE = sys.argv[1]
GLOBAL_FILE = sys.argv[2]

DIC_FILE = sys.argv[3]
dimension = int(sys.argv[4])

OUT_FILE = './ws353.output'
STD_FILE = './wordsim353/combined.tab'
SIM_TYPE = 'cos'
isLemma = True



wnl = WordNetLemmatizer() if isLemma else null
data = map(lambda x: x.lower().split('\t'), open(STD_FILE).readlines()[1:])
dic = {}
word_emb = {}
global_vec = []


def readFromFiles():
	global global_vec
	print 'read DIC_FILE...(first line must be UNKnown)'
	count = 0
	for line in open(DIC_FILE):
		if count > 0:
			dic[line.strip()] = count
			# 把UNKnown排除在外，剩下的词从1开始编号
		count += 1

	print 'read WORD_FILE...'
	one_word = []
	count = 0
	for line in open(WORD_FILE):
		if 'w' == line[0]:
			word_emb[count] = one_word # 0对应的是UNKnown，一个空的[]
			one_word = []
			count = int(line.split()[1])
		else:
			ls = line.strip().split()
			one_word.append([float(ls[0]), map(lambda x: float(x), ls[1:])])
	word_emb[count] = one_word
	# for line in open(WORD_FILE):
	# 	line = line.strip()
	# 	if 'w' == line[0]:
	# 		word_emb[count] = one_word
	# 		one_word = []
	# 		count = int(line.split()[1])
	# 	elif 's' == line[0]:
	# 		possibility = float(line.split()[1])
	# 	else:
	# 		one_word.append([possibility, map(lambda x: float(x), line.split())])
	# word_emb[count] = one_word

	print 'read GLOBAL_FILE...'
	global_vec = [map(lambda x: float(x), line.strip().split()) for line in open(GLOBAL_FILE).readlines()]
	# global_vec = [map(lambda x: float(x), line.strip().split(',')) for line in open(GLOBAL_FILE).readlines()]
	print len(global_vec)

def sim(v1, v2):
	if SIM_TYPE == 'cos':
		b = math.sqrt(sum(map(lambda x: x*x, v1)))
		c = math.sqrt(sum(map(lambda x: x*x, v2)))
		if b==0 or c==0 or len(v2)==0 or len(v1)==0:
			return 0
		a = sum([v1[i]*v2[i] for i in xrange(min(len(v1), len(v2)))])
		return a / b / c
	if SIM_TYPE == 'sig':
		a = sum([v1[i]*v2[i] for i in xrange(min(len(v1), len(v2)))])
		if len(v1)==0 or len(v2)==0:
			a = 0
		return 1.0/(1+math.exp(-a))


def evaluate():
	standard = [float(d[2]) for d in data]
	glb = []
	avg = []
	for d in data:
		w1 = wnl.lemmatize(d[0])
		w2 = wnl.lemmatize(d[1])
		if w1 not in dic or w2 not in dic:
			glb.append(0)
			avg.append(0)
		else:
			glb.append(sim(global_vec[dic[w1]], global_vec[dic[w2]]))
			sumsim, count = 0, 0
			for i in word_emb[dic[w1]]:
				for j in word_emb[dic[w2]]:
					sumsim += sim(i[1], j[1])
					count += 1
			avg.append(sumsim / count)
	return stats.spearmanr(standard, glb), stats.spearmanr(standard, avg)

if __name__ == '__main__':
	readFromFiles()
	output = open(OUT_FILE, 'a')
	glb_res, avg_res = evaluate()
	output.write('\n' + WORD_FILE + 'glb: ' + str(glb_res))
	output.write('\n' + WORD_FILE + 'avg: ' + str(avg_res))
	output.close()







