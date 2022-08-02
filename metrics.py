import numpy as np
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score ,v_measure_score, adjusted_mutual_info_score, accuracy_score
from scipy.optimize import linear_sum_assignment
from sklearn import metrics

def purity_score(y_true, y_pred):
    # compute contingency matrix (also called confusion matrix)
    contingency_matrix = metrics.cluster.contingency_matrix(y_true, y_pred)
    # return purity
    return np.sum(np.amax(contingency_matrix, axis=0)) / np.sum(contingency_matrix) 

nmi = normalized_mutual_info_score
ari = adjusted_rand_score
accuracy = accuracy_score

def clustering_accuracy(y_true, y_pred):
    """
    Calculate clustering accuracy. Require scikit-learn installed
    # Arguments
        y: true labels, numpy.array with shape `(n_samples,)`
        y_pred: predicted labels, numpy.array with shape `(n_samples,)`
    # Return
        accuracy, in [0,1]
    """
    y_true = y_true.astype(np.int64)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    ind = linear_sum_assignment(w.max() - w)
    ind = np.asarray(ind)
    ind = np.transpose(ind)
    return sum([w[i, j] for i, j in ind]) * 1.0 / y_pred.size

def tracker(label,pred,n_clusters):
    ###
    label_true = []
    label_pred_km = []
    label_pred_last = []

    if len(label) != len(pred):
        print("size error {0} != {1}".format(len(label),len(pred)))

    for i,classnum in enumerate(label):
        if classnum == 'EQ':
            label_true.append('eq')
        elif classnum == 'RF':
            label_true.append('rf')
        elif classnum == 'EN':
            label_true.append('en')
        elif classnum == 'car':
            label_true.append('car')
    for i,classnum in enumerate(pred):
        if label[i] != 'Check':
            label_pred_km.append(classnum)


    label_pred = np.copy(label_pred_km)

    ######
    for i in range(n_clusters):
        check = []
        for n,classnum in enumerate(pred):
            if classnum == i and label[n] != 'Check':
                check.append(label[n])
        if len(check) > 0 :  
            print ("in cluster{0} : EQnum={1},RFnum={2},ENnum={3},carnum={4}".format(i,check.count('EQ'),check.count('RF'),check.count('EN'),check.count('car')))
            if max(check,key=check.count) == 'EQ':
                print ("clster{}->EQclass".format(i))
                label_pred = [str('eq') if x==i else x for x in label_pred]
            elif max(check,key=check.count) == 'RF':
                print ("clster{}->RFclass".format(i))
                label_pred = [str('rf') if x==i else x for x in label_pred]
            elif max(check,key=check.count) == 'EN':
                print ("clster{}->ENclass".format(i))
                label_pred = [str('en') if x==i else x for x in label_pred]
            elif max(check,key=check.count) == 'car':
                print ("clster{}->ENclass".format(i))
                label_pred = [str('car') if x==i else x for x in label_pred]
        else :print ("no tracker in cluster="+str(i))

    return np.array(label_true),np.array(label_pred),np.array(label_pred_km)

def Clustering_metrics(label,pred,n_clusters,datasettype='check') :
    lists = ["nmi","ari","purity"]
    truelist,predlist,predlist_km = tracker(label,pred,n_clusters,datasettype='check')
    nmi = nmi(truelist,predlist_km)
    ari = ari(truelist,predlist_km)
    purity = purity_score(truelist,predlist_km)
    metric=[nmi,ari,purity]
    lists.append(metric)

    return lists
