import os
import boto3
from botocore.exceptions import ClientError

R2_ENDPOINT = os.getenv("R2_ENDPOINT")
R2_ACCESS_KEY = os.getenv("R2_ACCESS_KEY")
R2_SECRET_KEY = os.getenv("R2_SECRET_KEY")
R2_BUCKET = os.getenv("R2_BUCKET")

s3 = boto3.client(
    "s3",
    endpoint_url=R2_ENDPOINT,
    aws_access_key_id=R2_ACCESS_KEY,
    aws_secret_access_key=R2_SECRET_KEY,
    region_name="auto"
)

def upload_file(local_path: str, remote_path: str):
    s3.upload_file(local_path, R2_BUCKET, remote_path)

def download_file(remote_path: str, local_path: str):
    try:
        s3.download_file(R2_BUCKET, remote_path, local_path)
        return True
    except ClientError:
        return False
