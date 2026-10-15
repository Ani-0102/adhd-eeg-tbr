"""Literature check: search PubMed for TBR-in-ADHD papers and save a small table.
No API key and no LLM: it only lists what PubMed returns, newest first."""
import json
import ssl
import time
import urllib.parse
import urllib.request

import certifi
import pandas as pd

BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
QUERIES = [
    "theta beta ratio ADHD meta-analysis",
    "theta/beta ratio ADHD diagnostic accuracy",
    "theta beta ratio ADHD children EEG",
]
PER_QUERY = 15
OUT_PATH = "data/literature_check.csv"
# certifi's CA list, because some Python installs lack the system certificates
SSL_CONTEXT = ssl.create_default_context(cafile=certifi.where())


def get_json(endpoint, params):
    url = BASE + endpoint + "?" + urllib.parse.urlencode(params)
    with urllib.request.urlopen(url, timeout=30, context=SSL_CONTEXT) as response:
        time.sleep(0.4)  # NCBI allows about 3 requests per second without a key
        return json.loads(response.read())


def search_ids(query):
    result = get_json("esearch.fcgi", {"db": "pubmed", "term": query, "retmode": "json",
                                       "retmax": PER_QUERY, "sort": "relevance"})
    return result["esearchresult"]["idlist"]


def summaries(ids):
    result = get_json("esummary.fcgi", {"db": "pubmed", "id": ",".join(ids), "retmode": "json"})
    rows = []
    for pmid in ids:
        item = result["result"][pmid]
        rows.append({
            "pmid": pmid,
            "year": item["pubdate"][:4],
            "title": item["title"],
            "journal": item["source"],
            "types": ", ".join(item.get("pubtype", [])),
            "url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
        })
    return rows


if __name__ == "__main__":
    ids = []
    for query in QUERIES:
        for pmid in search_ids(query):
            if pmid not in ids:
                ids.append(pmid)
    table = pd.DataFrame(summaries(ids)).sort_values("year", ascending=False)
    table.to_csv(OUT_PATH, index=False)
    print(len(table), "papers saved to", OUT_PATH)
    print(table[table.types.str.contains("Meta-Analysis|Systematic Review")]
          [["pmid", "year", "journal", "title"]].to_string(index=False))
