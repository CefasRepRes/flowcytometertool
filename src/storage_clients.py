# storage_clients.py
from urllib.parse import urlparse
from azure.storage.blob import BlobServiceClient, ContainerClient, BlobClient
from auth import get_credential

def _split_blob_url(url: str):
    """
    Accepts: https://<account>.blob.core.windows.net/<container>/<optional/prefix>
    Returns: (account_url, container_name, prefix)
    """
    p = urlparse(url)
    account_url = f"https://{p.netloc}"
    path = p.path.strip("/").split("/", 1)
    container_name = path[0] if path and path[0] else ""
    prefix = path[1] if len(path) > 1 else ""
    return account_url, container_name, prefix

def get_blob_service_client(account_url: str) -> BlobServiceClient:
    return BlobServiceClient(account_url=account_url, credential=get_credential())

def get_container_client(account_url: str, container_name: str, anonymous: bool = False) -> ContainerClient:
    if anonymous:
        return ContainerClient(account_url=account_url, container_name=container_name)  # public container
    return get_blob_service_client(account_url).get_container_client(container_name)

def get_blob_client(account_url: str, container_name: str, blob_name: str, anonymous: bool = False) -> BlobClient:
    if anonymous:
        return BlobClient(account_url=account_url, container_name=container_name, blob_name=blob_name)
    return get_blob_service_client(account_url).get_blob_client(container=container_name, blob=blob_name)
