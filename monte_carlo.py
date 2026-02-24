import os
import pandas as pd
import numpy as np
import random

N_TEAMS = 30
GAMES_PER_TEAM = 82

def simulate_season(strengths, skill_percent):
    games = np.zeros(N_TEAMS)
    wins = np.zeros(N_TEAMS)

    matchups = []
    for i in range(N_TEAMS):
        opponents = np.random.choice(
                [j for j in range(N_TEAMS) if j != i],
                size=GAMES_PER_TEAM,
                replace=True
                )
        for opp in opponents:
            matchups.append((i, opp))
    for team_a, team_b in matchups:
        roll = np.random.randint(1, 101)
        if roll <= (100 - skill_percent):
            winner = np.random.choice([team_a, team_b])
        else:
            if strengths[team_a] > strengths[team_b]:
                winner = team_a
            else:
                winner = team_b
        wins[winner] += 1
        games[team_a] += 1
        games[team_b] += 1
    win_pct = wins / GAMES_PER_TEAM

    return np.std(win_pct)

def find_closest_skill_pct(skill_levels, std):
    best = -1e10
    best_pct = 0
    for i in range(3):
        for skill in skill_levels:
            curr = simulate_season(strengths, skill)
            if abs(curr - std) < abs(best - std):
                best = curr
                best_pct = skill
    return best_pct, best


        


if __name__ == "__main__":
    df = pd.read_csv("all_games_preproc.csv")
    win_pct = df.groupby(["season", "team"]).apply(lambda x: x["winRateFor"].tail(1))
    win_pct_std = win_pct.std()

    strengths = np.random.permutation(np.arange(1, N_TEAMS + 1))
    skill_levels = np.arange(20, 30, 0.01)
    
    closest, best_std = find_closest_skill_pct(skill_levels, win_pct_std)
    print(f"Skill: {closest}\t\tUpper Bound: {closest + (100 -closest) / 2}\t\tDifference:{abs(best_std - win_pct_std)}")
    
    
    
