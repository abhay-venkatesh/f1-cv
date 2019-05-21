import pandas as pd
df = pd.read_csv('data/beta_0001_mu_003', sep=',', header=None)
df = df.drop(0, 1)
df.columns = ["b0001m003"]

df_ = pd.read_csv('data/beta_0001_mu_005', sep=',', header=None)
df_ = df_.drop(0, 1)
df_.columns = ["b0001m005"]
df = pd.concat([df, df_], axis=1)

print(df)