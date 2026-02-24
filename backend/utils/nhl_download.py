from tqdm import tqdm
import requests
import threading
import os
from dotenv import load_dotenv

def get_nhl_teams(fpath):
    team_names = []
    try:
        with open(fpath, 'r') as f:
            team_names = [l.strip('\n') for l in f.readlines()]
    except:
        return Error("Failed to read NHL team file")
    return team_names


def update_team_csv():


def download_file(url, subject, gametype, path=None):
    if path == None:
        _dir = "./backend/data/NHL/{1}/{2}".format(subject, gametype)
        path = "./backend/data/NHL/{1}/{2}/{3}".format(subject, gametype, url.split("/")[-1])
    try:
        r = requests.get(url)
        if r.status_code == 404:
            return
        content = requests.get(url, stream = True)
        if not os.path.exists(_dir):
            os.makedirs(_dir)
        with open(path, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024):
                if chunk:
                    f.write(chunk)
    except:
        print("Downloading {0} failed...".format(url))
def fetch_team_data(team_name: str, regular: bool, playoff: bool):
    regular_base_url = "https://moneypuck.com/moneypuck/playerData/careers/gameByGame/regular/teams/"
    playoff_base_url = "https://moneypuck.com/moneypuck/playerData/careers/gameByGame/playoffs/teams/"

    try:
        t1, t2 = (None, None)
        team = team_name.strip()
        url1 = regular_base_url + f"{team_name}.csv"
        url2 = playoff_base_url + f"{team_name}.csv"

        if regular:
            t1 = threading.Thread(target=download_team_file, args=((url1, "teams", "regular")))
            t1.start()
        if playoff:
            t2 = threading.Thread(target=download_team_file, args=((url2, "teams", "playoff")))
            t2.start()
        if t1 != None:
            t1.join()
        if t2 != None:
            t2.join()
    except Exception as e:
        print(e)


def fetch_all_team_data(regular=True, playoff=True):
    try:
        nhl_teams = get_nhl_teams('./team_files')
        regular_base_url = "https://moneypuck.com/moneypuck/playerData/careers/gameByGame/regular/teams/"
        playoff_base_url = "https://moneypuck.com/moneypuck/playerData/careers/gameByGame/playoffs/teams/"

        for team in tqdm(nhl_teams):
            fetch_nhl_team_data(team, regular, playoff)
    except:
        raise Exception("Error occurred downloading team data ...")

        return True