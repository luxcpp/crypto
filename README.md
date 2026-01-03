# lux-crypto

High-performance C++ cryptography library for the Lux Network.

## Features

- **BLS12-381** - Pairing-friendly curves for threshold signatures
- **ML-DSA** - Post-quantum digital signatures (CRYSTALS-Dilithium)
- **ML-KEM** - Post-quantum key encapsulation (CRYSTALS-Kyber)
- **secp256k1** - Ethereum-compatible ECDSA

## Installation

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
cmake --install build --prefix /usr/local
```

## Usage

### CMake

```cmake
find_package(lux-crypto REQUIRED)
target_link_libraries(myapp PRIVATE lux::crypto)
```

### pkg-config (for CGO)

```bash
export CGO_CFLAGS=$(pkg-config --cflags lux-crypto)
export CGO_LDFLAGS=$(pkg-config --libs lux-crypto)
```

## Go Bindings

See [github.com/luxfi/crypto](https://github.com/luxfi/crypto) for Go bindings.

## Documentation

- [Full Documentation](https://luxfi.github.io/crypto/docs/cpp-libraries)
- [GPU Acceleration](https://luxfi.github.io/crypto/docs/gpu-acceleration)
- [Post-Quantum Crypto](https://luxfi.github.io/crypto/docs/post-quantum)

## License

BSD-3-Clause - See [LICENSE](LICENSE)
