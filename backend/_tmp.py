import matplotlib.pyplot as mp
import pandas as pd
import numpy as np
import seaborn as sb
import xgboost as xgb

if __name__ == "__main__":
    df = pd.read_csv("all_games_preproc.csv", low_memory=False)
    df = df[df["goalDiffFor"] != 0]
    X = df.loc[:, df.columns.str.contains("ema") + df.columns.str.contains("elo")]
    y_ml = df.loc[:,"winner"]

    rf = xgb.XGBClassifier()
    rf.fit(X, y_ml)
    important = pd.DataFrame({'Feature': X.columns, 'Importance':rf.feature_importances_})
    important.sort_values(by='Importance', ascending=False, inplace=True)
    top_features = important['Feature'].head(10)
    
    print(X.loc[:, top_features].corr())
    dataplot = sb.heatmap(X.loc[:, top_features].corr(), cmap="YlGnBu",annot=True)

    mp.show()
    