from NaiveBayes import NaiveBayes
import json
import numpy

def use_nb_indiv(datas_train, datas_valid, reTrain = True):
    # The Naive Bayes Model
    nb = NaiveBayes()
    # Record all of the predictions
    predicts_all = []
    # Number of all of the correctly predicted articles
    correct_all = 0
    # Correctly predicted ratio in each part
    correct_ratios = []
    # First train, if reTrain = False, only train
    nb.train(datas_train)
    for cur_part in range(1,partition+1):
        # If no retrain
        if reTrain:
            nb.train(datas_train)
        # All of the predictions in this part
        predicts = [nb.predict(data['content'])[0] for data in datas_valid]
        # Append to all of the predictions
        predicts_all += predicts
        # Number of correctly predicted articles in this part
        correct = 0
        for i in range(len(predicts)):
            if predicts[i] == datas_valid[i]['category']:
                correct += 1
        # Add to all of the correctly predicted articles
        correct_all += correct
        # Add to all of the ratios
        correct_ratios.append(correct/len(datas_valid))
        print("Correct: ", correct, "out of ", len(datas_valid))
        print(datas_valid[-1]['date'])
        #
        datas_train = datas[min(len(datas), plen*cur_part) : min(len(datas), plen*(cur_part+1))]
        datas_valid = datas[min(len(datas), plen*(cur_part+1)) : min(len(datas), plen*(cur_part+2))]
        #
        if reTrain:
            for i in range(len(datas_train)):
                datas_train[i]['category'] = predicts[i]
    #
    res = "Correct: %d out of %d"%(correct_all, len(predicts_all))
    print(res)
    return correct_ratios
    #return res 

with open('../datas/bbc_preprocessed.json') as f:
    datas = json.load(f)

partition = 100
plen = len(datas)//partition
d1 = datas[:plen]
d2 = datas[plen:plen*2]

res1 = use_nb_indiv(d1, d2, reTrain = False)
res2 = use_nb_indiv(d1, d2, reTrain = True)
