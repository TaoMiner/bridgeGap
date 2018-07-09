import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
import datetime
import codecs

class Evaluator:
    def __init__(self):
        self.ans_file = ''
        self.predict_file = ''
        self.log_file = ''
        self.feature_file = ''
        self.feature_list = ['cand_size', 'pem', 'pe','largest_pe','str_sim1', 'str_sim2','str_sim3','str_sim4','str_sim5','csim1', 'crank1','csim2', 'crank2','csim3', 'crank3', 'csim4', 'crank4']

    def loadFeatures(self):
        self.features = pd.read_csv(self.feature_file, header = None, names = ['doc_id', 'mention_id', 'wiki_id', 'cand_id',\
                                              'cand_size', 'pem', 'pe','largest_pe',\
                                                'str_sim1', 'str_sim2','str_sim3','str_sim4','str_sim5',\
                                                'esim1', 'erank1', 'esim2', 'erank2','esim3', 'erank3',\
                                                'csim1', 'crank1','csim2', 'crank2','csim3', 'crank3','csim4', 'crank4','csim5', 'crank5'])
        print 'load finished!'
        self.features = self.features.fillna(0)

    def formatter(self):
        label = []
        for row in self.features.loc[:,['wiki_id', 'cand_id']].itertuples():
            label.append(1 if row[1] == row[2] else 0)
        self.features.insert(0, 'label', label)

        self.train = self.features[self.features.doc_id < 947]
        tmp = self.features[self.features.doc_id >= 947]
        self.testa = tmp[tmp.doc_id < 1163]
        #self.testb = tmp[tmp.doc_id >= 1163]
        # predict full dataset
        self.testb = self.features

    def gbdt(self):
        gbdt=GradientBoostingRegressor(
                                          loss='ls'
                                        , learning_rate=0.02
                                        , n_estimators=10000
                                        , subsample=1
                                        , min_samples_split=2
                                        , min_samples_leaf=1
                                        , max_depth=4
                                        , init=None
                                        , random_state=None
                                        , max_features=None
                                        , alpha=0.9
                                        , verbose=0
                                        , max_leaf_nodes=None
                                        , warm_start=False
                                        )
        #self.train_x = self.train.loc[:, 'cand_size':'crank5'].values
        self.train_x = self.train.loc[:, self.feature_list].values
        self.train_y = self.train.loc[:, 'label'].values

        self.testb_x = self.testb.loc[:, self.feature_list].values
        self.testb_y = self.testb.loc[:, 'label'].values

        gbdt.fit(self.train_x, self.train_y)
        print("train finished!")
        pred=gbdt.predict(self.testb_x)
        self.testb.insert(0,'score', pred)

        #testa's doc_id 947-1162
        total_p = 0.0
        total_mention_tp = 0
        total_doc_num = 0
        total_mention_num = 0
        # for i in xrange(1163,1394,1) only for testb
        for i in xrange(1, 1394, 1):
            df_doc = self.testb[self.testb.doc_id == i]
            if df_doc.shape[0]==0:continue
            d_mention_num = df_doc['mention_id'].iloc[-1]+1
            #max score's index of mention's candidates
            idx = df_doc.groupby('mention_id')['score'].idxmax()
            # restore predicted result
            tmp_res = df_doc.loc[idx]
            if len(self.predict_file) > 0:
                tmp_res.to_csv(self.predict_file, mode='a', header=False, index=False)
            # record answers
            if len(self.ans_file) > 0:
                ans = df_doc.loc[df_doc[df_doc.label == 1].index]
                ans.to_csv(self.ans_file, mode='a', header=False, index=False)

            #num of label with 1 with max score
            d_tp = tmp_res[tmp_res.label == 1].shape[0]
            total_p += float(d_tp)/d_mention_num
            total_mention_tp += d_tp
            total_doc_num += 1
            total_mention_num += d_mention_num
        micro_p = float(total_mention_tp)/total_mention_num
        macro_p = total_p/total_doc_num
        print("micro precision : %f(%d/%d), macro precision : %f" % (micro_p, total_mention_tp, total_mention_num, macro_p))
        if len(self.log_file) > 0:
            with codecs.open(self.log_file, 'a', encoding='UTF-8') as fout:
                fout.write('*******************************************************************************************\n')
                fout.write('feature file:%s, answer file:%s, predicted file:%s\n' % (self.feature_file, self.predict_file, self.ans_file))
                fout.write('%s\n' % ','.join(self.feature_list))
                fout.write("micro precision : %f(%d/%d), macro precision : %f\n" % (micro_p, total_mention_tp, total_mention_num, macro_p))
                fout.write("*******************************************************************************************\n")

if __name__ == '__main__':

    intput_path = '/data/m1/cyx/etc/expdata/conll/'
    output_path = '/data/m1/cyx/etc/expdata/conll/log/'
    conll_file = intput_path + 'ppr_conll_file.csv'
    ans_file = output_path + 'conll_ans.mpme'
    predict_file = output_path + 'conll_pred.mpme'
    log_file = output_path + 'conll_log'
    eval = Evaluator()
    eval.ans_file = ans_file
    eval.predict_file = predict_file
    eval.log_file = log_file
    eval.feature_file = conll_file
    eval.loadFeatures()
    eval.formatter()
    starttime = datetime.datetime.now()
    eval.gbdt()
    endtime = datetime.datetime.now()
    print (endtime - starttime).seconds
