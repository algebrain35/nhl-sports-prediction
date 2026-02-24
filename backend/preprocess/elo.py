import numpy as np
import math

class Elo:
    def __init__(self, k: int, decay: int):
        self.rating_dict = {}
        self.season = 0.0
        self.k = k
        self.decay = decay

    def __getitem__(self, player: str) -> float:
        return self.rating_dict[player]
    def __setitem__(self, player: str, data: float) -> None:
        self.rating_dict[player] = data
    def get_margin_factor(self, score: float) -> float:
        return np.log2(math.fabs(score) + 1)
    def get_inflation_factor(self, r_win: float, r_lose: float) -> float:
        return 1 / (1 - ((r_lose - r_win) / 2200))
    def add_team(self, name: str, rating: float = 1500.):
        self.rating_dict[name] = rating
    def update_ratings(self, winner: str, loser: str, decay: float, margin: float, inflation: float) -> None:
        expected_result = self.get_expect_result(
            self.rating_dict[winner], self.rating_dict[loser]
        )
        self.rating_dict[winner] = max(100, self.rating_dict[winner] + self.k *margin*inflation* (1 - expected_result))
        self.rating_dict[loser] = max(100, self.rating_dict[loser] + self.k *margin*inflation* (-1 + expected_result))
    def get_expect_result(self, p1: float, p2: float) -> float:
        exp = (p2 - p1) / 400.0
        return 1 / ((10.0 ** (exp)) + 1)
    def get_season(self) -> float:
        return self.season
    def set_season(self, season, team_set) -> None:
        self.season = season
        for k,v in self.rating_dict.items():
            self.rating_dict[k] = (1 - self.decay) * v
