import string
import json

finFile = '../datas/bbc.json'
foutFile = '../datas/bbc_filterCategory.json'

with open(finFile) as fin:
  datas = json.load(fin)

cat_set = set(['Entertainment & Arts', 'Business', 'Technology', 'Health', 'Science & Environment'])

datas = [data for data in datas if data['category'] in cat_set]

with open(foutFile, 'w') as fout:
  json.dump(datas, fout, separators = (', \n', ': '))
  