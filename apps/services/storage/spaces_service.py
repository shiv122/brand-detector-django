"""
DigitalOcean Spaces (S3-compatible) storage.

Detection uploads each annotated frame here and stores the object's public URL
on the Frame row; OCR then sends that URL to the GLM box (no base64). When the
DO_* env is unset, `is_configured()` is False and detection falls back to local
/static frame files (dev only).
"""

from __future__ import annotations

import logging
from urllib.parse import urlsplit

from config.app_config import AppConfig

logger = logging.getLogger(__name__)


class SpacesService:
    def __init__(self, config: AppConfig):
        self.config = config
        self._client = None
        self._public_base = self._compute_public_base()

    def is_configured(self) -> bool:
        c = self.config
        return bool(
            c.spaces_endpoint
            and c.spaces_bucket
            and c.spaces_access_key_id
            and c.spaces_secret_access_key
        )

    def _compute_public_base(self) -> str:
        if self.config.spaces_public_base_url:
            return self.config.spaces_public_base_url.rstrip("/")
        endpoint = self.config.spaces_endpoint
        if not endpoint:
            return ""
        # Virtual-hosted form: https://<bucket>.<host>/<key>
        parts = urlsplit(endpoint)
        return f"{parts.scheme}://{self.config.spaces_bucket}.{parts.netloc}"

    @property
    def client(self):
        if self._client is None:
            import boto3
            from botocore.config import Config as BotoConfig

            self._client = boto3.client(
                "s3",
                endpoint_url=self.config.spaces_endpoint,
                region_name=self.config.spaces_region or None,
                aws_access_key_id=self.config.spaces_access_key_id,
                aws_secret_access_key=self.config.spaces_secret_access_key,
                config=BotoConfig(
                    s3={"addressing_style": "virtual"},
                    retries={"max_attempts": 3, "mode": "standard"},
                ),
            )
        return self._client

    def public_url(self, key: str) -> str:
        return f"{self._public_base}/{key.lstrip('/')}"

    def presigned_put_url(self, key: str, expires: int = 3600) -> str:
        """A time-limited URL the browser can PUT raw bytes to directly.

        Signed with only Bucket+Key so the client can send any Content-Type
        without breaking the signature (Content-Type isn't a signed header).
        """
        return self.client.generate_presigned_url(
            "put_object",
            Params={"Bucket": self.config.spaces_bucket, "Key": key},
            ExpiresIn=expires,
        )

    def presigned_get_url(self, key: str, expires: int = 3600) -> str:
        """A time-limited URL to fetch the object — handed to the detector's
        file_url download flow so the uploaded video need not be public."""
        return self.client.generate_presigned_url(
            "get_object",
            Params={"Bucket": self.config.spaces_bucket, "Key": key},
            ExpiresIn=expires,
        )

    def upload_bytes(
        self,
        key: str,
        data: bytes,
        content_type: str = "image/jpeg",
        public: bool = True,
    ) -> str:
        """Upload bytes and return the object's public URL.

        Raises on failure — callers decide whether to fall back to local
        storage. Objects are written public-read so both the dashboard and the
        GLM OCR box can fetch them by a stable URL.
        """
        extra = {"ContentType": content_type, "CacheControl": "public, max-age=31536000, immutable"}
        if public:
            extra["ACL"] = "public-read"
        self.client.put_object(
            Bucket=self.config.spaces_bucket,
            Key=key,
            Body=data,
            **extra,
        )
        return self.public_url(key)
