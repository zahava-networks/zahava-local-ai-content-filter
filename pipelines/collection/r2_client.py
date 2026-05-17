"""Cloudflare R2 client (S3-compatible via boto3)."""
from __future__ import annotations

import io
from functools import lru_cache
from typing import BinaryIO

import boto3
from botocore.client import Config

from ..common import load_config, require_env


@lru_cache(maxsize=1)
def get_client():
    cfg = load_config()["storage"]["r2"]
    return boto3.client(
        "s3",
        endpoint_url=require_env(cfg["endpoint_env"]),
        aws_access_key_id=require_env(cfg["access_key_env"]),
        aws_secret_access_key=require_env(cfg["secret_key_env"]),
        config=Config(signature_version="s3v4", retries={"max_attempts": 5}),
        region_name="auto",
    )


def bucket() -> str:
    cfg = load_config()["storage"]["r2"]
    return require_env(cfg["bucket_env"])


def upload_bytes(key: str, data: bytes, content_type: str = "image/webp") -> None:
    get_client().put_object(Bucket=bucket(), Key=key, Body=data, ContentType=content_type)


def upload_fileobj(key: str, fileobj: BinaryIO, content_type: str = "image/webp") -> None:
    get_client().upload_fileobj(fileobj, bucket(), key, ExtraArgs={"ContentType": content_type})


def exists(key: str) -> bool:
    try:
        get_client().head_object(Bucket=bucket(), Key=key)
        return True
    except Exception:
        return False


def download_bytes(key: str) -> bytes:
    buf = io.BytesIO()
    get_client().download_fileobj(bucket(), key, buf)
    return buf.getvalue()


def presigned_get_url(key: str, expires_in: int = 600) -> str:
    return get_client().generate_presigned_url(
        "get_object",
        Params={"Bucket": bucket(), "Key": key},
        ExpiresIn=expires_in,
    )


def list_keys(prefix: str, limit: int | None = None):
    paginator = get_client().get_paginator("list_objects_v2")
    n = 0
    for page in paginator.paginate(Bucket=bucket(), Prefix=prefix):
        for obj in page.get("Contents", []):
            yield obj["Key"]
            n += 1
            if limit and n >= limit:
                return
