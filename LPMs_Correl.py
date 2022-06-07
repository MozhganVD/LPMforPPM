import glob
import os
import pandas as pd
from sklearn.metrics import matthews_corrcoef


def transfer(x):
    if x == False:
        return 0
    elif x == True:
        return 1
    elif x == 'None':
        return 0
    elif x == 'pattern1':
        return 1
    elif x == 'regular':
        return 1
    elif x == 'deviant':
        return 0
    else:
        return x


folder_address = "../../Datasets/000_Experimemts/TrafficFine/"
data = pd.read_csv(folder_address + "TrafficFines_Trunc10_Complete_Agg.csv", sep=';')
trace_data = pd.DataFrame(columns=['case:concept:name', 'LPMs', 'label'])
for idx, case, in enumerate(data['case:concept:name'].unique()):
    trace_data.loc[idx, 'case:concept:name'] = case
    trace_data.loc[idx, 'LPMs'] = max(data[data['case:concept:name'] == case]['LPMs_binary'].values)
    trace_data.loc[idx, 'label'] = data[data['case:concept:name'] == case]['label'].values[0]


trace_data['label'] = trace_data['label'].apply(transfer)
Mattiew_Coef = matthews_corrcoef(list(trace_data['label']), list(trace_data['LPMs']))
print(Mattiew_Coef)


