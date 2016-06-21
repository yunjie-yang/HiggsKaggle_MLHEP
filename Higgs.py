import matplotlib.pyplot as plt

import pandas
import numpy

from sklearn.cross_validation import train_test_split
from sklearn.metrics import roc_auc_score

data = pandas.read_csv('train_10000.csv')

#data_raw = pandas.DataFrame(data_pre)

'''
indices = ['event_id','target', 'lepton_pt', 'lepton_eta','lepton_phi','mem_pt','mem_phi',
           'jet1_pt','jet1_eta','jet1_phi','jet1_btag',
           'jet2_pt','jet2_eta','jet2_phi','jet2_btag',
           'jet3_pt','jet3_eta','jet3_phi','jet3_btag',
           'jet4_pt','jet4_eta','jet4_phi','jet4_btag',
            'm_jj',
            'm_jjj',
            'm_lv',
            'm_jlv',
            'm_bb',
            'm_wbb',
            'm_wwbb']
'''

#print data

features = list(set(data.columns) - {'event_id', 'target'})

high_level_features = ['m_jj', 'm_jjj', 'm_jlv', 'm_wwbb', 'm_bb', 'm_wbb', 'm_lv']
#print features

hist_params = {'normed': True, 'bins': 60, 'alpha': 0.4}
# create the figure
plt.figure(figsize=(16, 25))
for n, feature in enumerate(high_level_features):
    # add sub plot on our figure
    plt.subplot(len(features) // 5 + 1, 3, n+1)
    # define range for histograms by cutting 1% of data from both ends
    min_value, max_value = numpy.percentile(data[feature], [1, 99])
    plt.hist(data.ix[data.target.values == 0, feature].values, range=(min_value, max_value), 
             label='class 0', **hist_params)
    plt.hist(data.ix[data.target.values == 1, feature].values, range=(min_value, max_value), 
             label='class 1', **hist_params)
    plt.legend(loc='best')
    plt.title(feature)


training_data, validation_data = train_test_split(data, random_state=11, train_size=0.66)

high_level_features = ['m_jj', 'm_jjj', 'm_jlv', 'm_wwbb', 'm_bb', 'm_wbb', 'm_lv']

from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=7)
knn.fit(training_data[high_level_features],training_data.target)

proba = knn.predict_proba(validation_data[high_level_features])

print roc_auc_score(validation_data.target,proba[:,1])



