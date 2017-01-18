# coding: utf-8
from scipy import stats
from nltk.stem import WordNetLemmatizer
import math
import re
import sys



SCWS_FILE = './SCWS/ratings.txt'
WORD_FILE = sys.argv[1] + '/CRP_' + sys.argv[2] + '_word_sense'
CONT_FILE = sys.argv[1] + '/CRP_' + sys.argv[2] + '_word_context'
VECT_FILE = sys.argv[1] + '/CRP_' + sys.argv[2] + '_vect_global'
DIC_FILE = sys.argv[3]

isLemma = True
SIM_TYPE = 'cos'
dimension = 300
half_window = 5
OUTPUT_FILE = './output.tmp'



wnl = WordNetLemmatizer() if isLemma else null
data = map(lambda x: x.lower().split('\t'), open(SCWS_FILE).readlines())
dic = {}
word_emb = {}
cont_emb = {}
vect_emb = []

def readFromFiles():
	# print 'read DIC_FILE...(first line must be UNKnown)'
	count = 0
	for line in open(DIC_FILE):
		if count > 0:
			dic[line.strip()] = count
			# 把UNKnown排除在外，剩下的词从1开始编号
		count += 1

	# print 'read WORD_FILE...'
	one_word = []
	count = 0
	for line in open(WORD_FILE):
		if 'w' == line[0]:
			word_emb[count] = one_word # 0对应的是UNKnown，一个空的[]
			one_word = []
			count = int(line.split()[1])
			dimension = int(line.split()[3])
		else:
			one_word.append(map(lambda x: float(x), line.strip().split()[1:]))
	word_emb[count] = one_word

	# print 'read VECT_FILE...'
	for line in open(VECT_FILE):
		vect_emb.append(map(lambda x: float(x), line.strip().split()))

	# print 'read CONT_FILE...'
	one_cont = []
	count = 0
	for line in open(CONT_FILE):
		if 'w' == line[0]:
			cont_emb[count] = one_cont # 0对应的是UNKnown
			one_cont = []
			count = int(line.split()[1])
		else:
			one_cont.append(map(lambda x: float(x), line.strip().split()[1:]))
	cont_emb[count] = one_cont

	# print 'finish reading all files.'

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

replace_reg = re.compile('[^a-z0-9\\.<>/]')
def getEmbedding(s):
	sentences = replace_reg.sub(' ', s.lower().replace('!', '.').replace('?', '.')).split('.')
	sentence = ''
	for tmp in sentences:
		if '<b>' in tmp:
			sentence = tmp
			break
	sentence = sentence.split()
	sen, target, exist = [], -1, False
	for i in xrange(len(sentence)):
		if '<b>' == sentence[i]:
			w = wnl.lemmatize(sentence[i+1]) if isLemma else sentence[i+1]
			if w in dic and dic[w] in word_emb:
				sen.append(dic[w])
				target = len(sen) - 1
				exist = True
			else:
				target = len(sen)
			i += 2
		else:
			w = wnl.lemmatize(sentence[i]) if isLemma else sentence[i]
			if w in dic and dic[w] in word_emb:
				sen.append(dic[w])
	
	context = [0 for i in xrange(dimension)]
	if len(sen) > 0:
		count = 0
		for i in xrange(max(0, target-half_window), target):
			for j in xrange(dimension):
				context[j] += vect_emb[sen[i]][j]
			count += 1
		for i in xrange(target + (1 if exist else 0), min(target+half_window, len(sen))):
			for j in xrange(dimension):
				context[j] += vect_emb[sen[i]][j]
			count += 1
		for j in xrange(dimension):
			context[j] /= count
	results = []
	weights = []
	if exist:
		for i in xrange(len(cont_emb[sen[target]])):
			weights.append(sim(context, cont_emb[sen[target]][i]) + 1.0)
			results.append(word_emb[sen[target]][i])
		weight_Z = sum(weights)
		if weight_Z == 0:
			weight_Z = 0.000001
		weights = [w/weight_Z for w in weights]
	else:
		results.append(context)
		weights.append(1.0)
	return results, weights

def evaluate():
	res1 = [d[7] for d in data]
	avg_res2 = []
	max_res2 = []
	# count = 0
	for d in data:
		# count += 1
		# if count%100 == 0:
		# 	print "evaluating number: ", count
		emb1, wei1 = getEmbedding(d[5])
		emb2, wei2 = getEmbedding(d[6])
		score = 0.0
		for i in xrange(len(emb1)):
			for j in xrange(len(emb2)):
				score += wei1[i] * wei2[j] * sim(emb1[i], emb2[j])
		avg_res2.append(score)


		i = wei1.index(max(wei1))
		j = wei2.index(max(wei2))
		score = sim(emb1[i], emb2[j])
		max_res2.append(score)
	return stats.spearmanr(res1, avg_res2), stats.spearmanr(res1, max_res2)

if __name__ == '__main__':
	readFromFiles()
	output = open(OUTPUT_FILE, 'a')
	avg_res, max_res = evaluate()
	output.write(sys.argv[1] + ' ' + sys.argv[2] + ' avg: ' + str(avg_res) + '\n')
	output.write(sys.argv[1] + ' ' + sys.argv[2] + ' max: ' + str(max_res) + '\n')
	output.close()

