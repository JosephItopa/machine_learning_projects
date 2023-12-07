
import os
import time
import json
from google.cloud import storage

try:
    with open('deep-contact-credentials.json', 'r') as f:
      client = storage.Client.from_service_account_json('deep-contact-credentials.json')
except Exception as e:
    print(e)

def create_bucket_class_location(bucket_name):
    """
    Create a new bucket in the US region
    """
    bucket = ""
    try:
        # Creates the new bucket
        bucket = client.create_bucket(bucket_name, location='US-EAST1')
        print(f"Bucket {bucket.name} created.")
    except Exception as e:
       print(e)

    return bucket

def create_folder(bucket_name, folder_name):
    try:
       # Create a bucket object
       bucket = client.bucket(bucket_name)
       # Set the name of the folder you want to create
       folder = folder_name #'extracted-text', 'processed-img'
       # Create a blob object for the folder
       folder_ = bucket.blob(folder)
       # Create the folder
       folder_.upload_from_string('')
       print(f'Folder {folder_name} created.')
    except Exception as e:
       print(e)

def upload_files(bucket_name, local_directory):
    # Create a bucket object
    bucket = client.bucket(bucket_name)
    # Uupload the files in the local directory
    #with client.batch():
    for root, dirs, files in os.walk(local_directory):
       for file in files:
        local_path = os.path.join(root, file)
        remote_path = "new-models/"+file # dict 
        blob = bucket.blob(remote_path)
        print(f'Uploading file: {local_path}')
        try:
           blob.upload_from_filename(local_path)
           os.remove(local_path)
           print("file uploaded, and deleted from local directory")
        except Exception as e:
           print(f'Error uploading file: {e}')
           continue
    print('Files uploaded successfully')


def delete_folder(bucket_name, folder_name):
    try:
        # Create a bucket object
        bucket = client.bucket(bucket_name)
        # Delete the folder itself
        folder_ = bucket.blob(folder_name)
        folder_.delete()
        print(f'Folder {folder_name} deleted.')
    except Exception as e:
       print(e)


def delete_file(bucket_name, file_name):
    try:
       # Create a bucket object
       bucket = client.bucket(bucket_name)
       # Create a blob object from the file to object storage
       blob = bucket.blob(file_name)
       # Delete the file from the bucket
       blob.delete()
       print(f'File {file_name} deleted.')
    except Exception as e:
       print(e)

def download_files(bucket_name, local_dir, remote_folder):
    # Create a bucket object
    bucket = client.bucket(bucket_name)
    blob_names = [blob.name for blob in bucket.list_blobs(prefix=remote_folder,max_results=1000)]

    for name in blob_names:
        print(name)
        #remote_path = remote_folder+name
        blob = bucket.blob(name)
        # Download the file to a local directory
        local_path = os.path.join(local_dir, os.path.basename(name))
        blob.download_to_filename(local_path)

    print('Files downloaded successfully')


if __name__ == "__main__":
    # The name for the new bucket
    bucket_name = "df-ml-models" #"receipt-img-text-extract"
    #create_bucket_class_location(bucket_name)
    # create_folder(bucket_name, "dict") # new-models, bucket_name, folder_name; raw, extracted_text
    #upload_files(bucket_name, "./data/dict") # "sample-bucket-img2023
    download_files(bucket_name, "./data/dict", "dict")