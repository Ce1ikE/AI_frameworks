# Copyright 2025 Yakhyokhuja Valikhujaev
# Author: Yakhyokhuja Valikhujaev
# GitHub: https://github.com/yakhyo/uniface

import os
import hashlib
import requests
from enum import Enum
from typing import Dict

import logging

logger = logging.getLogger(__name__)

def verify_model_weights(
    model_name: str, 
    root: str = '~/.arcface/models',
    model_urls: Dict[Enum, str] = None,
    model_sha256: Dict[Enum, str] = None,
    chunk_size: int = 8192
) -> str:

    root = os.path.expanduser(root)
    os.makedirs(root, exist_ok=True)
    model_path = os.path.join(root, f'{model_name}.onnx')

    if not os.path.exists(model_path):
        url = model_urls.get(model_name)
        if not url:
            logger.error(f"No URL found for model '{model_name}'")
            raise ValueError(f"No URL found for model '{model_name}'")

        logger.info(f"Downloading model '{model_name}' from {url}")
        download_file(url, model_path, chunk_size)
        logger.info(f"Successfully downloaded '{model_name}' to {os.path.normpath(model_path)}")
    else:
        logger.info(f"Model '{model_name}' already exists at {os.path.normpath(model_path)}")

    expected_hash = model_sha256.get(model_name)
    if expected_hash and not verify_file_hash(model_path, expected_hash, chunk_size):
        os.remove(model_path)  # Remove corrupted file
        logger.warning("Corrupted weight detected. Removing...")
        raise ValueError(f"Hash mismatch for '{model_name}'. The file may be corrupted; please try downloading again.")

    return model_path


def download_file(url: str, dest_path: str, chunk_size: int) -> None:
    """Download a file from a URL in chunks and save it to the destination path."""
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        with open(dest_path, "wb") as file:
            for chunk in response.iter_content(chunk_size=chunk_size):
                logger.debug(f"Downloading chunk of size {len(chunk)} bytes")
                if chunk:
                    file.write(chunk)
    except requests.RequestException as e:
        raise ConnectionError(f"Failed to download file from {url}. Error: {e}")


def verify_file_hash(file_path: str, expected_hash: str, chunk_size: int) -> bool:
    """Compute the SHA-256 hash of the file and compare it with the expected hash."""
    file_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            file_hash.update(chunk)
    actual_hash = file_hash.hexdigest()
    if actual_hash != expected_hash:
        logger.warning(f"Expected hash: {expected_hash}, but got: {actual_hash}")
    return actual_hash == expected_hash
