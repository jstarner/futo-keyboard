# XZ Embedded

Decoder-only subset of [XZ Embedded](https://github.com/tukaani-project/xz-embedded)
(tag `v2024-12-30`, commit `ae63ae3a36ed01724674e8f3d750dc47bf125410`), used to
unpack the LZMA-compressed side blobs of Bark `.bark` speech models.

Files are copied unchanged from `linux/include/linux`, `linux/lib/xz` and
`userspace`. The build defines `XZ_USE_CRC64`, since the blobs use CRC64 checks.

License: 0BSD (see `LICENSE`).
