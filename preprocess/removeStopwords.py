import json

finFile = 'bbc_removeNonPrintable.json'
foutFile = 'bbc_removeStopwords.json'

f = open('stopwords.txt')
l = f.read().replace('"', '').split(',')

fin = open(finFile)
datas = json.load(fin)

for data in datas:
    data['title'] = 