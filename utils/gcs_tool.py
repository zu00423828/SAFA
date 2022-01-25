from google.cloud import storage
import os
from datetime import timedelta
storage_client = storage.Client()
bucket_name = "test-lip-data" if os.environ.get(
    'GCS_BUCKET_NAME') is None else os.environ.get('GCS_BUCKET_NAME')
bucket = storage_client.bucket(bucket_name)


def upload_to_gcs(path, gcs_path):
    filename = os.path.basename(path)
    blob = bucket.blob(gcs_path)  # gcs path
    blob.upload_from_filename(path)  # local path


def get_blob_list():
    # bucket = storage_client.get_bucket(bucket_name)
    for blob in bucket.list_blobs():
        print(blob.name, blob.generate_signed_url(expiration=timedelta(3)))


def delete_blobs():
    for blob in bucket.list_blobs():
        bucket.delete_blob(blob.name)


def download_gcs(filename, save_dir):
    blob = bucket.get_blob(filename)
    save_path = os.path.join(save_dir, os.path.basename(filename))
    blob.download_to_filename(save_path)
