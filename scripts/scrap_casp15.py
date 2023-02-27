import requests
from bs4 import BeautifulSoup
from tqdm.auto import tqdm

url = "https://predictioncenter.org/casp15"
save_dir = "/scratch/09101/whatever/data/casp15"

page = requests.get(f"{url}/targetlist.cgi")
soup = BeautifulSoup(page.content, "html.parser")


name2link = {
    link.text: link.get("href")
    for link in soup.find_all("a")
    if link.get("href").startswith("target.cgi?id=")
}
name2detail = {}
for n, link in tqdm(name2link.items()):
    details = BeautifulSoup(
        requests.get(f"{url}/{link}").content, "html.parser"
    ).find_all("textarea")
    seq = details[0].text
    pdb = details[1].text
    name2detail[n] = [seq, pdb]

with open(f"{save_dir}/fasta.fa", "w") as f:
    for n, detail in name2detail.items():
        print(detail[0], file=f, end="")
        with open(f"{save_dir}/data/{n}.pdb", "w") as f_pdb:
            print(detail[1], file=f_pdb, end="")
