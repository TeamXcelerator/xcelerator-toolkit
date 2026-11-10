# Historical deterministic ZIP fixtures

These immutable archives were encoded with zip-rs 2.4.2, flate2 1.1.9,
miniz_oxide 0.8.9, and compression level 6. They use the fixed 1980 timestamp,
Unix creator metadata, mode 0644, and the original ordered payload bytes.
The V1 multi-entry archive omits ZIP64 for its small entries; the V1
single-entry and V2 multi-entry archives request ZIP64 unconditionally.

The compatibility test reads these actual historical encoder bytes with the
current decoder and preserves their original transport profiles. Do not
regenerate them with the current encoder.

| File | Bytes | SHA-256 |
| --- | ---: | --- |
| `deterministic-v1-multi.zip` | 286 | `7f055ac842984a82133d8f55ad0320da1fab69ff838c1ba49d2dd839dad22da0` |
| `deterministic-v1-single.zip` | 180 | `55bb83e72b1489e33353c45eba5a65acccd222c6b760b6e7b40726f238cab563` |
| `deterministic-v2-multi.zip` | 366 | `2023bee9a5e03a2082e5ad31d34eafe5fd5be0314fb07c141af1a831fc908b21` |
