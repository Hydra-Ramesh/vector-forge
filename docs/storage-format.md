# VectorForge Binary Storage Format

The binary storage format uses little-endian byte order.

## File Header (24 bytes)

| Offset | Type     | Name      | Description |
|--------|----------|-----------|-------------|
| 0      | char[8]  | Magic     | "VFORGE01"  |
| 8      | uint32_t | Version   | 1           |
| 12     | uint32_t | Dimension | Vector dimensionality (e.g. 128) |
| 16     | uint64_t | Count     | Number of vectors in the file |

## Payload

The payload follows immediately after the header.

1. **IDs Array**: `Count` x `uint64_t` (8 bytes each). Represents the Vector IDs. Total size: `Count * 8` bytes.
2. **Vectors Array**: `Count * Dimension` x `float` (4 bytes each). Represents the flattened vector data. Total size: `Count * Dimension * 4` bytes.

### Notes
- Storing vectors sequentially rather than interleaved with IDs allows for cache-friendly distance calculations.
