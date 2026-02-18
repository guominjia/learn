# Azure

## Authentication
1. Microsoft Authentication Library, key APIs
    - **acquire_token_for_clients**
    - **initiate_auth_code_flow**
    - **acquire_token_by_auth_code_flow**
```python
import msal

app = msal.ConfidentialClientApplication(
    CLIENT_ID,
    authority=AUTHORITY,
    client_credential=CLIENT_SECRET,
)

result = app.acquire_token_for_client(scopes=SCOPE)

access_token = result['access_token']
headers = {
    'Authorization': f'Bearer {access_token}',
    'Content-Type': 'application/json'
}

flow = msal.initiate_auth_code_flow(SCOPE, redirect_uri=REDIRECT_URI)
prin(flow['auth_uri']
result = app.acquire_token_by_auth_code_flow(flow, request.args)

for account in msal.get_accounts():
    result = msal.acquire_token_silent(SCOPE, account=account)
    print(result['access_token'])
```

Or

```python
import msal

app = msal.PublicClientApplication(
    CLIENT_ID,
    authority=AUTHORITY,
)

auth_url = app.get_authorization_request_url(SCOPE, redirect_uri=REDIRECT_URI)

webbrowser.open(auth_url)

result = app.acquire_token_by_authorization_code(auth_code, scopes=SCOPE, redirect_uri=REDIRECT_URI)

access_token = result['access_token']
headers = {
    'Authorization': f'Bearer {access_token}',
    'Content-Type': 'application/json'
}
```

## References
- <https://learn.microsoft.com/en-us/graph/api/user-list-messages?view=graph-rest-1.0&tabs=http>