# SPDX-License-Identifier: Apache-2.0
"""Hardware-compression contracts for distributed KV storage."""

# First Party
from lmcache.v1.distributed.compress_adapters.format import (
    MAX_RECORD_HEADER_SIZE,
    RECORD_FORMAT_VERSION,
    RECORD_MAGIC,
    CompressedChunkDescriptor,
    CompressedRecordFormatError,
    CompressedRecordHeader,
    CompressionCodec,
    CompressionFraming,
    PostDecompressTransform,
    StoredCompressionFormat,
    crc32_ieee,
    encode_record_header,
    parse_record_header,
)

__all__ = [
    "MAX_RECORD_HEADER_SIZE",
    "RECORD_FORMAT_VERSION",
    "RECORD_MAGIC",
    "CompressedChunkDescriptor",
    "CompressedRecordFormatError",
    "CompressedRecordHeader",
    "CompressionCodec",
    "CompressionFraming",
    "PostDecompressTransform",
    "StoredCompressionFormat",
    "crc32_ieee",
    "encode_record_header",
    "parse_record_header",
]
