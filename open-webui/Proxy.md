# Proxy
Discuss proxy in open-webui

## How to
1. Create necessary directory
```bash
mkdir -p nginx/conf.d nginx/ssl
```

2. Create config file `nginx/conf.d/app.conf`

```conf
server {
    listen 80;
    server_name your-domain.com;
    return 301 https://$host$request_uri;
}

server {
    listen 443 ssl;
    server_name your-domain.com;

    ssl_certificate /etc/nginx/ssl/cert.crt;
    ssl_certificate_key /etc/nginx/ssl/cert.key;

    # SSL config
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers HIGH:!aNULL:!MD5;
    ssl_prefer_server_ciphers on;

    # CORS config
    add_header 'Access-Control-Allow-Origin' '*' always;
    add_header 'Access-Control-Allow-Methods' '*' always;
    add_header 'Access-Control-Allow-Headers' '*' always;

    # Reverse config
    location / {
        if ($request_method = OPTIONS) {
            add_header 'Access-Control-Allow-Origin' '*';
            add_header 'Access-Control-Allow-Methods' '*';
            add_header 'Access-Control-Allow-Headers' '*';
            add_header 'Content-Type' 'text/plain; charset=utf-8';
            add_header 'Content-Length' 0;
            return 204;
        }

        proxy_pass http://host.docker.internal:3000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        
        # WebSocket proxy
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
    }
}
```

3. Run docker cli
```bash
docker run -d -p 80:80 -p 443:443 \
  -v $(pwd)/nginx/conf.d:/etc/nginx/conf.d \
  -v $(pwd)/nginx/ssl:/etc/nginx/ssl \
  --name nginx-proxy \
  --add-host=host.docker.internal:host-gateway \
  nginx
```

## Reverse Proxy
Should properly set http and websocket proxy, otherwise, stream(for example AI stream output) will fail.

### Apache use below configuration to forward request
```conf
# Forward HTTP requests  
ProxyPass / http://localhost:3000/ nocanon  
ProxyPassReverse / http://localhost:3000/  
  
# Forward WebSocket connections
ProxyPass / ws://localhost:3000/ nocanon  
ProxyPassReverse / ws://localhost:3000/
```

### Nginx covert to below
```conf
# Primary proxy
proxy_pass http://localhost:3000;
proxy_set_header Host $host;
proxy_set_header X-Real-IP $remote_addr;
proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;

# WebSocket proxy
proxy_set_header Upgrade $http_upgrade;
proxy_set_header Connection "upgrade";
proxy_http_version 1.1;
```

## Websearch Proxy
Should enable `WEB_SEARCH_TRUST_ENV=true` if in proxy environment, otherwise web search will fail

## Refer
- <https://deepwiki.com/search/webui-display-error-unexpected_64a107ad-0a83-41a6-b7e0-c2b6f239fec5?mode=fast>
- <https://github.com/open-webui/open-webui/blob/main/docs/apache.md>
- <https://github.com/open-webui/open-webui/blob/main/src/lib/components/admin/Settings/WebSearch.svelte>