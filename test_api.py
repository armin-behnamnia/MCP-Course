import requests
import json
PROXY = "http://192.168.10.2:3129" 
title = "Inverse Reinforcement Learning"
params = {"query.title": title, "rows": 1}
response = requests.get("https://api.crossref.org/works", timeout=10, params=params, proxies={"http": PROXY, "https": PROXY})
data = json.loads(response.text)
status = data['status']
message = data['message']
query = message['query']
results = message['items']
final_result = {
    "status": status,
    "results": [
        {
            'publisher': result['publisher'],
            'doi': result['DOI'],
            'source': result['source'],
            'title': result['title'][0],
            'author': result['author'],
            'year': result['created']['date-parts'][0][0]
        }
        for result in results
    ]
}
print(final_result)
