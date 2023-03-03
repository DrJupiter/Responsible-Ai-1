#%%
import pandas as pd
import numpy as np
#%%
df = pd.read_csv("data\catalan-juvenile-recidivism-subset.csv")

dummy_df = df.loc[:10]

# %%
dummy_df.loc[1:3,"V115_RECID2015_recid"] = 0

# %%

dummy_df["predictions"] = [0,1,0,1,0,1,0,1,0,1,0]

dummy_df.loc[:,["V4_area_origin","V115_RECID2015_recid","predictions"]]


# %%
s_df = dummy_df.loc[dummy_df['V4_area_origin'] == "Spain"]
m_df = dummy_df.loc[dummy_df['V4_area_origin'] == "Maghreb"]
# %%

s_acc = np.mean((s_df["V115_RECID2015_recid"]-s_df["predictions"])**2)
m_acc = np.mean((m_df["V115_RECID2015_recid"]-m_df["predictions"])**2)
s_acc, m_acc