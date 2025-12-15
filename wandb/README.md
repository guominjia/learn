# Weights & Biases

## Some commands to enable wandb
1. Run the following commands to make sure that proxy and docker are working normally

```bash
cat /etc/systemd/system/docker.service.d/http-proxy.conf 
cat /etc/systemd/system/docker.service.d/https-proxy.conf 
systemctl status docker
```

2. Run the following commands to pull wandb and start wandb server

```bash
docker pull wandb/local
docker search wandb/local

export WANDB_MODE=online
wandb server start
wandb local
```

3. The above command may be fail with `image_id_from_registry NoneType have no split attribute`, the following commands can check the method and configuration

```bash
find /usr/local/lib/python3.10/dist-packages/wandb/ -name "*.py"|xargs grep "def image_id_from_registry"
cat ~/.docker/config.json
```

4. Run the following command to check method one by one and will found `auth_token` will fail with `ssl/tls` protocal

```python
python -c "import wandb;print(wandb.docker.image_id('wandb/local'))"
python -c "import wandb;print(wandb.docker.image_id_from_registry('wandb/local'))"
python -c "import wandb;print(wandb.docker.auth.load_config())"
python -c "import wandb;print(wandb.docker.parse('wandb/local'))"
python -c "import wandb;print(wandb.docker.auth_token('index.docker.io', 'wandb/local'))"
```

5. Run the following command to double confirm that `ssl/tls` have some issue. I haven't resolved it yet

```bash
curl -v https://index.docker.io
```

6. I checked the `image_id_from_registry` and extract the docker command to run it directly

```bash
docker run -d --rm -v wandb:/vol -p 8080:8080 --name wandb-local wandb/local
docker logs wandb-local
ls /var/lib/docker/volumes
```

7. Run the following command to cli login

```bash
wandb login --relogin --host=http://shsse006.sh.intel.com:8080
```

8. Run below command to synchronize offline run directory

```bash
export WANDB_BASE_URL=http://localhost:8080
wandb sync ./path/to/offline-run-directory/
```