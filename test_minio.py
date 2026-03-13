from minio import Minio
import os

client = Minio(
    f"{os.getenv('MINIO_HOST')}:{int(os.getenv('MINIO_PORT'))}",
    access_key=os.getenv("MINIO_ACCESS_KEY"),
    secret_key=os.getenv("MINIO_SECRET_KEY"),
    secure=False,  # True if HTTPS
)

bucket = os.getenv("MINIO_BUCKET")
obj = os.getenv("MINIO_OBJECT")

response = client.get_object(bucket, obj)
try:
    data = response.read()
    print("Data Length from MinIO object:", len(data))
    print("First 20 characters of MinIO object:", data[:20])
finally:
    response.close()
    response.release_conn()