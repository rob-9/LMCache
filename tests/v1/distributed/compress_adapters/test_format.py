# SPDX-License-Identifier: Apache-2.0
"""Public-contract tests for portable compression-format identity."""

# Standard
from dataclasses import FrozenInstanceError
import struct

# Third Party
import pytest

# First Party
from lmcache.v1.distributed.compress_adapters import (
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

_V1_ONE_CHUNK_HEADER = bytes.fromhex(
    "4c4d435201010100"
    "40000000"
    "4300000000000000"
    "0500000000000000"
    "01000000"
    "abda4916"
    "00000000"
    "4000000000000000"
    "03000000"
    "05000000"
    "86a61036"
    "00000000"
)


def _rewrite_header_crc(encoded: bytearray) -> None:
    """Recompute the draft header checksum after a deliberate mutation."""
    struct.pack_into("<I", encoded, 32, 0)
    struct.pack_into("<I", encoded, 32, crc32_ieee(encoded))


def _sample_header() -> CompressedRecordHeader:
    """Return a valid two-chunk header with an alignment gap."""
    return CompressedRecordHeader(
        stored_format=StoredCompressionFormat(
            codec=CompressionCodec.DEFLATE,
            framing=CompressionFraming.RAW,
        ),
        chunks=(
            CompressedChunkDescriptor(
                payload_offset=88,
                compressed_size=4,
                uncompressed_size=8,
                uncompressed_crc32=0x12345678,
            ),
            CompressedChunkDescriptor(
                payload_offset=96,
                compressed_size=3,
                uncompressed_size=5,
                uncompressed_crc32=0x90ABCDEF,
            ),
        ),
        record_size=99,
    )


def test_format_requires_explicit_codec_and_framing() -> None:
    """A stored format names its codec and framing without a backend."""
    stored_format = StoredCompressionFormat(
        codec=CompressionCodec.DEFLATE,
        framing=CompressionFraming.RAW,
    )

    assert stored_format.codec is CompressionCodec.DEFLATE
    assert stored_format.framing is CompressionFraming.RAW
    assert stored_format.post_decompress_transform is PostDecompressTransform.NONE


def test_format_exposes_only_a_fully_specified_transform() -> None:
    """Version 1 does not advertise an underspecified byte-reorder mode."""
    assert list(PostDecompressTransform) == [PostDecompressTransform.NONE]


def test_format_is_immutable() -> None:
    """Stored-format identity cannot change after construction."""
    stored_format = StoredCompressionFormat(
        codec=CompressionCodec.DEFLATE,
        framing=CompressionFraming.RAW,
    )

    with pytest.raises(FrozenInstanceError):
        stored_format.framing = CompressionFraming.GZIP  # type: ignore[misc]


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("codec", "deflate", "codec must be a CompressionCodec"),
        ("framing", "raw", "framing must be a CompressionFraming"),
        (
            "post_decompress_transform",
            "none",
            "post_decompress_transform must be a PostDecompressTransform",
        ),
    ],
)
def test_format_rejects_untyped_values(
    field: str,
    value: str,
    message: str,
) -> None:
    """Configuration strings must be parsed before format construction."""
    kwargs: dict[str, object] = {
        "codec": CompressionCodec.DEFLATE,
        "framing": CompressionFraming.RAW,
    }
    kwargs[field] = value

    with pytest.raises(TypeError, match=message):
        StoredCompressionFormat(**kwargs)  # type: ignore[arg-type]


def test_record_header_round_trip_is_deterministic() -> None:
    """Encoding is stable and parsing recovers the public header value."""
    header = _sample_header()

    encoded_once = encode_record_header(header)
    encoded_twice = encode_record_header(header)

    assert encoded_once == encoded_twice
    assert len(encoded_once) == header.header_size == 88
    assert encoded_once[:4] == RECORD_MAGIC
    assert encoded_once[4] == RECORD_FORMAT_VERSION
    assert parse_record_header(encoded_once) == header


def test_v1_wire_format_matches_frozen_vector() -> None:
    """Version 1 field order, IDs, widths, and CRC remain byte-for-byte stable."""
    header = CompressedRecordHeader(
        stored_format=StoredCompressionFormat(
            codec=CompressionCodec.DEFLATE,
            framing=CompressionFraming.RAW,
        ),
        chunks=(
            CompressedChunkDescriptor(
                payload_offset=64,
                compressed_size=3,
                uncompressed_size=5,
                uncompressed_crc32=0x3610A686,
            ),
        ),
        record_size=67,
    )

    assert encode_record_header(header) == _V1_ONE_CHUNK_HEADER
    assert parse_record_header(_V1_ONE_CHUNK_HEADER) == header


def test_header_rejects_invalid_chunk_element_with_type_error() -> None:
    """Invalid descriptor elements fail the documented type contract first."""
    stored_format = StoredCompressionFormat(
        codec=CompressionCodec.DEFLATE,
        framing=CompressionFraming.RAW,
    )

    with pytest.raises(
        TypeError,
        match=r"chunks\[0\] must be a CompressedChunkDescriptor",
    ):
        CompressedRecordHeader(
            stored_format=stored_format,
            chunks=(object(),),  # type: ignore[arg-type]
            record_size=64,
        )


def test_crc32_ieee_matches_the_standard_check_value() -> None:
    """CRC semantics match CRC-32/IEEE as implemented by ``zlib``."""
    assert crc32_ieee(b"123456789") == 0xCBF43926


def test_encoded_header_contains_crc32_ieee() -> None:
    """The stored header checksum covers fixed metadata and descriptors."""
    encoded = bytearray(encode_record_header(_sample_header()))
    stored_crc32 = struct.unpack_from("<I", encoded, 32)[0]
    struct.pack_into("<I", encoded, 32, 0)

    assert stored_crc32 == crc32_ieee(encoded)


def test_parser_rejects_header_checksum_mismatch() -> None:
    """A valid-looking metadata bit flip fails header integrity validation."""
    encoded = bytearray(encode_record_header(_sample_header()))
    encoded[48] ^= 1

    with pytest.raises(CompressedRecordFormatError, match="CRC-32/IEEE mismatch"):
        parse_record_header(encoded)


def test_empty_record_header_round_trip() -> None:
    """An empty record contains only the fixed header and no descriptors."""
    header = CompressedRecordHeader(
        stored_format=StoredCompressionFormat(
            codec=CompressionCodec.DEFLATE,
            framing=CompressionFraming.GZIP,
        ),
        chunks=(),
        record_size=40,
    )

    encoded = encode_record_header(header)

    assert len(encoded) == 40
    assert header.uncompressed_size == 0
    assert parse_record_header(encoded) == header


@pytest.mark.parametrize(
    ("byte_offset", "replacement", "message"),
    [
        (0, ord("X"), "invalid record magic"),
        (4, RECORD_FORMAT_VERSION + 1, "unsupported record version"),
        (5, 0xFF, "unknown compression codec id"),
        (6, 0xFF, "unknown compression framing id"),
        (7, 0xFF, "unknown post-decompression transform id"),
    ],
)
def test_parser_rejects_unknown_fixed_header_values(
    byte_offset: int,
    replacement: int,
    message: str,
) -> None:
    """Magic, version, and enum identifiers fail closed when unknown."""
    encoded = bytearray(encode_record_header(_sample_header()))
    encoded[byte_offset] = replacement
    _rewrite_header_crc(encoded)

    with pytest.raises(CompressedRecordFormatError, match=message):
        parse_record_header(encoded)


@pytest.mark.parametrize("size", [0, 1, 39, 87])
def test_parser_rejects_truncated_headers(size: int) -> None:
    """Both the fixed header and descriptor table must be complete."""
    encoded = encode_record_header(_sample_header())

    with pytest.raises(CompressedRecordFormatError, match="truncated"):
        parse_record_header(encoded[:size])


def test_parser_rejects_inconsistent_header_size() -> None:
    """The declared header size must match the descriptor count exactly."""
    encoded = bytearray(encode_record_header(_sample_header()))
    struct.pack_into("<I", encoded, 8, 32)

    with pytest.raises(CompressedRecordFormatError, match="header_size"):
        parse_record_header(encoded)


def test_parser_rejects_chunk_beyond_record_size() -> None:
    """A descriptor cannot address bytes beyond the declared record."""
    encoded = bytearray(encode_record_header(_sample_header()))
    struct.pack_into("<Q", encoded, 12, 98)
    _rewrite_header_crc(encoded)

    with pytest.raises(CompressedRecordFormatError, match="beyond record_size"):
        parse_record_header(encoded)


def test_parser_rejects_overlapping_chunks() -> None:
    """Ordered descriptors may contain alignment gaps but cannot overlap."""
    encoded = bytearray(encode_record_header(_sample_header()))
    struct.pack_into("<Q", encoded, 64, 91)
    _rewrite_header_crc(encoded)

    with pytest.raises(CompressedRecordFormatError, match="overlaps"):
        parse_record_header(encoded)


def test_parser_rejects_inconsistent_uncompressed_total() -> None:
    """The fixed-header output total must equal the descriptor sum."""
    encoded = bytearray(encode_record_header(_sample_header()))
    struct.pack_into("<Q", encoded, 20, 14)
    _rewrite_header_crc(encoded)

    with pytest.raises(
        CompressedRecordFormatError,
        match="declared uncompressed size",
    ):
        parse_record_header(encoded)


def test_parser_rejects_reserved_chunk_flags() -> None:
    """Unknown per-chunk flags are rejected instead of silently ignored."""
    encoded = bytearray(encode_record_header(_sample_header()))
    struct.pack_into("<I", encoded, 60, 1)
    _rewrite_header_crc(encoded)

    with pytest.raises(CompressedRecordFormatError, match="unsupported flags"):
        parse_record_header(encoded)


def test_parser_rejects_reserved_header_flags() -> None:
    """Unknown fixed-header flags are rejected instead of silently ignored."""
    encoded = bytearray(encode_record_header(_sample_header()))
    struct.pack_into("<I", encoded, 36, 1)
    _rewrite_header_crc(encoded)

    with pytest.raises(CompressedRecordFormatError, match="header uses unsupported"):
        parse_record_header(encoded)


def test_parser_rejects_impractical_chunk_table() -> None:
    """Wire-valid counts cannot force allocation or iteration past the limit."""
    header = CompressedRecordHeader(
        stored_format=StoredCompressionFormat(
            codec=CompressionCodec.DEFLATE,
            framing=CompressionFraming.RAW,
        ),
        chunks=(),
        record_size=40,
    )
    encoded = bytearray(encode_record_header(header))
    excessive_chunk_count = (MAX_RECORD_HEADER_SIZE - 40) // 24 + 1
    struct.pack_into("<I", encoded, 8, 40 + excessive_chunk_count * 24)
    struct.pack_into("<I", encoded, 28, excessive_chunk_count)

    with pytest.raises(CompressedRecordFormatError, match="exceeds version-1"):
        parse_record_header(encoded)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("payload_offset", True, "payload_offset must be an int"),
        ("payload_offset", -1, "payload_offset must be in"),
        ("compressed_size", 0, "compressed_size must be greater than zero"),
        ("uncompressed_size", 0, "uncompressed_size must be greater than zero"),
        ("uncompressed_crc32", 1 << 32, "uncompressed_crc32 must be in"),
    ],
)
def test_chunk_descriptor_validates_wire_ranges(
    field: str,
    value: int,
    message: str,
) -> None:
    """Descriptors reject values that cannot be encoded or decoded safely."""
    kwargs = {
        "payload_offset": 32,
        "compressed_size": 1,
        "uncompressed_size": 1,
        "uncompressed_crc32": 0,
    }
    kwargs[field] = value

    with pytest.raises((TypeError, ValueError), match=message):
        CompressedChunkDescriptor(**kwargs)


def test_header_allows_alignment_gaps_between_chunks() -> None:
    """Payload alignment padding does not become part of a chunk."""
    header = _sample_header()

    assert header.chunks[0].payload_offset + header.chunks[0].compressed_size == 92
    assert header.chunks[1].payload_offset == 96
    assert parse_record_header(encode_record_header(header)) == header
