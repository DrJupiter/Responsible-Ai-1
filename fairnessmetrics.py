import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix



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


def assessSeperation(df, preds, group = 'V4_area_origin'):
    groups = df[group].unique()

    y_pred = {e: [] for e in groups}
    y_true = {e: [] for e in groups}

    for idx, pred in enumerate(preds):
        #lists of predictions and true values across groups
        y_pred[df.loc[idx][group]].append(pred)
        y_true[df.loc[idx][group]].append(df.loc[idx]['V115_RECID2015_recid'])

    out = {e: (0,0) for e in groups}

    for group in groups:
        #calculating the true positive rate and false positive rate across groups
        tn, fp, fn, tp = confusion_matrix(y_true[group], y_pred[group]).ravel()
        tpr = tp / (tp + fn)
        fpr = fp / (fp + tn)
        out[group] = (tpr, fpr)

    return out

def assessSufficiency(df, preds, group = 'V4_area_origin'):
    groups = df[group].unique()

    # diving into groups and into the different predictors r = 1 and r = 0
    r_1 = {e: [0, 0] for e in groups}
    r_0 = {e: [0, 0] for e in groups}

    for idx, pred in enumerate(preds):
        if pred == 1:
            # getting number of predictions for r=1
            r_1[df.loc[idx][group]][0] += 1
            if df.loc[idx]['V115_RECID2015_recid'] == 1:
                # Getting number of true values for y=1, given the model predicts r=1
                r_1[df.loc[idx][group]][1] += 1
        else:
            # getting number of predictions for r=0
            r_0[df.loc[idx][group]][0] += 1
            if df.loc[idx]['V115_RECID2015_recid'] == 1:
                # Getting number of true values for y=1, given the model predicts r=0
                r_0[df.loc[idx][group]][1] += 1

    out = {e: (0, 0) for e in groups}

    # Getting the sufficiency rate for predictors r=0 and r=1 for each group
    for group in groups:
        try:
            y1r1 = r_1[group][1] / r_1[group][0]
        except ZeroDivisionError:
            y1r1 = 0

        try:
            y1r0 = r_0[group][1] / r_0[group][0]
        except ZeroDivisionError:
            y1r0 = 0

        out[group] = (y1r1, y1r0)

    return out


def test_fairness(dataframe, predictions, log=True, print=True):
    out1 = assessIndependence(dataframe, predictions)
    out2 = assessSeperation(dataframe, predictions)
    out3 = assessSufficiency(dataframe, predictions)

    print("\n Independency test")
    for group in out1.keys():
        print("P( y=1 |", group, ") :", out1[group])

    print("\n Separation test")
    for group in out2.keys():
        print("P( y_hat=1 | y=1 ,", group, ") (TPR):", out2[group][0])
        print("P( y_hat=1 | y=0 ,", group, ") (FPR):", out2[group][1])

    print("\n Sufficiency test")
    for group in out3.keys():
        print("P( y=1 | r=1 ,", group, ") :", out3[group][0])
        print("P( y=1 | r=0 ,", group, ") :", out3[group][1])


    


if __name__ == "__main__":
    df = pd.read_csv('data/catalan-juvenile-recidivism-subset.csv')
    print(f"Attributes {df.columns}")


    df_dummy = df.loc[:10]
    df_dummy.loc[0:3, 'V115_RECID2015_recid'] = 0 # alter values to have different recidivism
    fake_preds = [1, 1, 0, 0, 1, 1, 1, 0, 1, 0, 1]
    group = 'V4_area_origin'

    print(df_dummy['V4_area_origin'].unique())
    test_fairness(df_dummy, fake_preds)