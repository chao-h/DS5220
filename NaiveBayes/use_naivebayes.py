from NaiveBayes import NaiveBayes
import json
import numpy
with open('../datas/bbc_preprocessed.json') as fin:
    datas = json.load(fin)

def use_nb(datas_train, datas_valid):
    nb = NaiveBayes()
    nb.train(datas_train)
    #
    predicts = []
    for i in range(len(datas_valid)):
        if (i%1000) == 0:
            print("Getting prediction of", i, "documents out of", len(datas_valid))
        predicts.append(nb.predict(datas_valid[i]['content']))
    #
    pp = [p[0] for p in predicts]
    #
    correct = 0
    for i in range(len(predicts)):
        if predicts[i][0] == datas_valid[i]['category']:
            correct += 1
    #
    res = "Correct: %d out of %d"%(correct, len(datas_valid))
    print(res)
    return res 
    

#use_nb(datas_train, datas_valid)

partition = 100
plen = len(datas)//partition
datas_train = datas[:plen]
datas_valid = datas[plen:plen*2]

res1 = use_nb(datas[:plen], datas[plen:])

def use_nb2(datas_train, datas_valid):
    nb = NaiveBayes()
    predicts_all = []
    correct_all = 0
    #
    for cur_part in range(1,partition+1):
        nb.train(datas_train)
        predicts = [nb.predict(data['content'])[0] for data in datas_valid]
        predicts_all += predicts
        correct = 0
        for i in range(len(predicts)):
            if predicts[i] == datas_valid[i]['category']:
                correct += 1
        correct_all += correct
        print("Correct: ", correct, "out of ", len(datas_valid))
        print(datas_valid[-1]['date'])
        #
        datas_train = datas[min(len(datas), plen*cur_part) : min(len(datas), plen*(cur_part+1))]
        datas_valid = datas[min(len(datas), plen*(cur_part+1)) : min(len(datas), plen*(cur_part+2))]
        #
        for i in range(len(datas_train)):
            datas_train[i]['category'] = predicts[i]
    #
    res = "Correct: %d out of %d"%(correct_all, len(predicts_all))
    print(res)
    return res 

res2 = use_nb2(datas_train, datas_valid)

print(res1)
print(res2)