from __future__ import annotations


def decompress_body(body: bytes, encoding: str) -> bytes:
    """Decompress request body if Content-Encoding is set.

    Supported: gzip, deflate, br (brotli), zstd.
    Returns original body if encoding is not supported or empty.
    """
    encoding = encoding.lower()
    if encoding == "gzip":
        import gzip
        return gzip.decompress(body)
    elif encoding == "deflate":
        import zlib
        return zlib.decompress(body)
    elif encoding == "br":
        import brotli
        return brotli.decompress(body)
    elif encoding == "zstd":
        import zstandard
        return zstandard.decompress(body)
    return body
