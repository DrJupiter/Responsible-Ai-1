import numpy as np
import pandas as pd

df = pd.read_csv('data/catalan-juvenile-recidivism-subset.csv')
print(f"Attributes {df.columns}")


df_dummy = df.loc[:10]
df_dummy.loc[0:3, 'V115_RECID2015_recid'] = 0 # alter values to have different recidivism
fake_preds = [1, 1, 0, 0, 1, 1, 1, 0, 1, 0]
group = 'V4_area_origin'

print(df_dummy['V4_area_origin'].unique())

def assessIndependence(df, preds, group='V4_area_origin'):
    # unique values
    groups = df[group].unique()

    out = {e : 0 for e in groups}

    for idx, pred in enumerate(preds):

        if df.loc[idx]['V115_RECID2015_recid'] == pred:
            out[df.loc[idx][group]] += 1
    
    for g in groups:
        out[g] = out[g]/len(preds)

    return out

out = assessIndependence(df_dummy, fake_preds)


def assessSeperation(x, y, z):
    pass

def assessSufficiency(x, y, z):
    pass




    


