import json
from collections import defaultdict
from collections import OrderedDict
from datetime import datetime

fin = '../datas/bbc_preprocessed.json'
fout = '../datas/bbc_classifyTimeSeries.json'

with open(fin) as f:
    datas = json.load(f)

timeSeries = defaultdict(list)

for data in datas:
    date = data['date']
    timeSeries[date].append(data)

timeSeries = OrderedDict(sorted(timeSeries.items(), key = lambda kv:  datetime.strptime(kv[0], '%d %B %Y')))

datass = []
for k, v in timeSeries.items():
    datass.append({
        'date': k,
        'datas': v
    })

with open(fout, 'w') as f:
    json.dump(datass, f, separators = (', \n', ': '))
