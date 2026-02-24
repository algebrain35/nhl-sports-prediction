import os
import matplotlib.pyplot as plt
import pandas as pd
import xgboost as xgb

if __name__ == "__main__":
    df = pd.read_csv("all_games_preproc.csv")
    df = df.dropna()
    df = df[df["goalDiffFor"] != 0]
    X = X.loc[:, df.columns.str.contains('elo') | df.columns.str.contains('ema')]
##    df["gameId"] = df["gameId"] % 200000
##    season_19 = df[ (df["season"].astype(int) == 2019) & (df["team"] == "ANA")]
##
##    x1 = season_19["gameId"]
##    y1 = season_19["fenwickPercentage_For"]
##    y2 = season_19["fenwickPercentage_For_seasonal_ema_span_8"]
##
##
####    plt.figure(figsize=(10,8))
##    plt.plot(x1, y1, color='black', label='fenwick_Per_For')
##    plt.plot(x1, y2, color='green', label='fenwick_Per_ema_8')
##    plt.title("fenwickPercentage_For vs EMA span 8 ANA 2019")
##
##    plt.xlabel('ANA_2019 Games')
##    plt.ylabel('fenwickPercentage_For')
##
##    plt.legend()
##
##    plt.show()
    y3 = df[df["goalDiffFor" > 1.5]]
    y2 = df[df["goalsFor"] + df["goalsAgainst"] > 6.5]
    y1 = df.loc[:, "winner"]
            
    
    
