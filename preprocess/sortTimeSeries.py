import json
import numpy
from datetime import datetime

finFile = '../datas/bbc_removeStopwords.json'
foutFile = '../datas/bbc_sortTimeseries.json'

with open(finFile) as f:
    datas=json.load(f)

numpy.random.seed(123)
numpy.random.shuffle(datas)
datas.sort(key = lambda data:  datetime.strptime(data['date'], '%d %B %Y'))

with open(foutFile, 'w') as fout:
    json.dump(datas, fout, separators = (', \n', ': '))

