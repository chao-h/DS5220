files = [
    "filterCategory.py",
    "removeNonPrintable.py",
    "stem.py",
    "removeStopwords.py",
    "sortTimeSeries.py"
]

for file_name in files:
    print("Executing:", file_name)
    exec(open(file_name).read())

foutFile = '../datas/bbc_preprocessed.json'

with open(foutFile, 'w') as fout:
    json.dump(datas, fout, separators = (', \n', ': '))

print("Finish preprocess")

