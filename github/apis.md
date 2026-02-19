# API

## Headers
```python
headers = {
    'Authorization': f"token {os.getenv('GITHUB_TOKEN')}",
    'Accept': 'application/vnd.github.v3+json'
}
```

## Issues
```python
url = f"https://api.github.com/repos/{owner}/{repo}/issues"
params = {"per_page": per_page, "page": 1, "sort": "updated", "direction": "desc"}
response = requests.get(url, headers=headers, params=params)
```

## Comments
```python
url = f"https://api.github.com/repos/{owner}/{repo}/issues/{issue_number}/comments"
params = {"per_page": per_page, "page": 1, "sort": "updated", "direction": "desc"}
response = requests.get(url, headers=headers, params=params)
```