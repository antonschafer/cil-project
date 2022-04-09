import pandas as pd 

e = pd.read_csv('ok.csv')
b = pd.read_csv('output.csv')

print(b["Prediction"] == e['Prediction'])
