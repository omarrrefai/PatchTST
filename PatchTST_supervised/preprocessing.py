import pandas as pd
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("/home/omaralrefai/dev/PatchTST/.dataset/ercot/ercot_15min.csv", parse_dates=["date"])
cols = [c for c in df.columns if c!="date"]

# Clip extreme outliers
df[cols] = df[cols].clip(lower=-50, upper=500)

# Scale
scaler = StandardScaler()
df[cols] = scaler.fit_transform(df[cols])

df.to_csv("/home/omaralrefai/dev/PatchTST/.dataset/ercot/ercot_15min_scaled.csv", index=False)
