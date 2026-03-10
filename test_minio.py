
from minio import Minio

client = Minio(
    "127.0.0.1:9000",
    access_key="MINIO_ACCESS_KEY",
    secret_key="MINIO_SECRET_KEY",
    secure=False,  # True if HTTPS
)

bucket = "my-bucket"
obj = "path/file.json"

response = client.get_object(bucket, obj)
try:
    data = response.read()
    print(data.decode("utf-8"))
finally:
    response.close()
    response.release_conn()