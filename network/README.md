# Network

```python
def mount(user=user, passw=passw, rpath=rpath):
    subprocess.run(f'net use T: /delete', shell=True)
    subprocess.run(f'net use /User:{user} T: {rpath} {passw}', shell=True)
```