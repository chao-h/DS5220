import string
import json

finFile = '../datas/bbc_filterCategory.json'
foutFile = '../datas/bbc_removeNonPrintable.json'

with open(finFile) as fin:
  datas = json.load(fin)

# The set of all printable characters
printable = set(string.printable)

def isPrintable(x):
  return (x in printable)

# Input: original string, Output: printable string
def removeNonPrintable(s):
  return (''.join(list(filter(isPrintable, s))))

# remove all nonprintable characters in title and content  
for data in datas:
  data['title'] = removeNonPrintable(data['title'])
  data['content'] = removeNonPrintable(data['content'])

with open(foutFile, 'w') as fout:
  json.dump(datas, fout, separators = (', \n', ': '))
  
