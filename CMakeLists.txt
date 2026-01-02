cmake_minimum_required(VERSION 3.20)
project(lux-crypto VERSION 1.0.0 LANGUAGES CXX OBJCXX)

# =============================================================================
# Lux Crypto Library - GPU-Accelerated Cryptographic Primitives
# =============================================================================
#
# Contract: github.com/luxfi/luxcpp/LIBRARY_CONTRACT.md
#   - CMake package: lux-crypto
#   - Target: lux::crypto
#   - Library: libluxcrypto.{dylib,so}
#   - Headers: include/lux/crypto/
#   - Resources: share/lux/crypto/ (Metal shaders)
#   - Relocatable: @rpath on macOS, $ORIGIN on Linux
#
# Depends on: luxcpp/lattice (for NTT operations)
#             luxcpp/gpu (optional, for MLX acceleration)
#
# Build Options:
#   -DWITH_GPU=ON       Enable GPU acceleration via luxcpp/gpu (MLX)
#   -DWITH_METAL=ON     Enable Metal compute shaders for BLS12-381 (macOS)
#   -DBUILD_SHARED_LIBS=ON Build shared libraries (default)
#
# Usage:
#   mkdir build && cd build
#   cmake -DWITH_METAL=ON -DCMAKE_PREFIX_PATH=../lattice/build ..
#   make -j$(sysctl -n hw.ncpu)
#
# =============================================================================

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
include(LuxLibrary)
include(GNUInstallDirs)

# Options
option(WITH_GPU "Build with GPU acceleration via luxcpp/gpu (MLX)" OFF)
option(WITH_METAL "Build with Metal compute shaders for BLS12-381" ON)

# =============================================================================
# Metal BLS12-381 Shader Compilation (macOS)
# =============================================================================

set(METAL_RESOURCES)
if(APPLE AND WITH_METAL)
    message(STATUS "Enabling Metal BLS12-381 compute shaders")

    find_program(XCRUN xcrun)
    if(XCRUN)
        set(METAL_SHADER "${CMAKE_CURRENT_SOURCE_DIR}/src/metal/bls12_381.metal")
        set(METAL_AIR "${CMAKE_CURRENT_BINARY_DIR}/bls12_381.air")
        set(METAL_LIB "${CMAKE_CURRENT_BINARY_DIR}/bls12_381.metallib")

        if(EXISTS "${METAL_SHADER}")
            # Compile .metal -> .air
            add_custom_command(
                OUTPUT ${METAL_AIR}
                COMMAND ${XCRUN} -sdk macosx metal -c ${METAL_SHADER} -o ${METAL_AIR}
                DEPENDS ${METAL_SHADER}
                COMMENT "Compiling Metal shader: bls12_381.metal"
            )

            # Link .air -> .metallib
            add_custom_command(
                OUTPUT ${METAL_LIB}
                COMMAND ${XCRUN} -sdk macosx metallib ${METAL_AIR} -o ${METAL_LIB}
                DEPENDS ${METAL_AIR}
                COMMENT "Linking Metal library: bls12_381.metallib"
            )

            add_custom_target(bls_metal_shaders DEPENDS ${METAL_LIB})
            set(METAL_RESOURCES ${METAL_LIB})
            message(STATUS "Metal shader: ${METAL_SHADER}")
        else()
            message(WARNING "Metal shader not found: ${METAL_SHADER}")
        endif()
    else()
        message(WARNING "xcrun not found, Metal shaders will be compiled at runtime")
    endif()
endif()

# =============================================================================
# Source Files
# =============================================================================

set(CRYPTO_SOURCES
    src/crypto.cpp
)

set(CRYPTO_HEADERS
    include/lux/crypto/crypto.h
)

if(APPLE AND WITH_METAL)
    list(APPEND CRYPTO_SOURCES src/metal_bls.mm)
    list(APPEND CRYPTO_HEADERS include/lux/crypto/metal_bls.h)
endif()

# =============================================================================
# Dependencies
# =============================================================================

set(CRYPTO_DEPS)
if(WITH_GPU)
    list(APPEND CRYPTO_DEPS lux-gpu)
endif()
# lux-lattice is optional dependency for NTT operations
find_package(lux-lattice CONFIG QUIET)
if(lux-lattice_FOUND)
    list(APPEND CRYPTO_DEPS lux-lattice)
endif()

# Frameworks
set(CRYPTO_FRAMEWORKS)
if(APPLE)
    list(APPEND CRYPTO_FRAMEWORKS Metal Foundation Security)
endif()

# =============================================================================
# Library Target (using shared LuxLibrary helper)
# =============================================================================

lux_add_library(
    NAME crypto
    VERSION ${PROJECT_VERSION}
    DESCRIPTION "GPU-accelerated cryptographic primitives for Lux (BLS12-381, ML-DSA)"
    SOURCES ${CRYPTO_SOURCES}
    HEADERS ${CRYPTO_HEADERS}
    DEPENDENCIES ${CRYPTO_DEPS}
    FRAMEWORKS ${CRYPTO_FRAMEWORKS}
    RESOURCES ${METAL_RESOURCES}
    SHARED
)

set(TARGET_NAME "lux-crypto")

# Add dependency on Metal shader compilation
if(TARGET bls_metal_shaders)
    add_dependencies(${TARGET_NAME} bls_metal_shaders)
endif()

# Compile definitions
if(APPLE AND WITH_METAL)
    target_compile_definitions(${TARGET_NAME} PRIVATE WITH_METAL)
endif()

if(WITH_GPU)
    target_compile_definitions(${TARGET_NAME} PRIVATE WITH_GPU WITH_MLX)
endif()

# =============================================================================
# CI Relocatability Check
# =============================================================================

lux_verify_relocatable(${TARGET_NAME})

# =============================================================================
# Summary
# =============================================================================

message(STATUS "=================================================")
message(STATUS "Lux Crypto Library Configuration:")
message(STATUS "  Package:        lux-crypto")
message(STATUS "  Target:         lux::crypto")
message(STATUS "  Library:        libluxcrypto.{dylib,so}")
message(STATUS "  Version:        ${PROJECT_VERSION}")
message(STATUS "  Metal BLS:      ${WITH_METAL}")
message(STATUS "  GPU (lux-gpu):  ${WITH_GPU}")
message(STATUS "  lux-lattice:    ${lux-lattice_FOUND}")
message(STATUS "  Shared:         ${BUILD_SHARED_LIBS}")
message(STATUS "  Install Prefix: ${CMAKE_INSTALL_PREFIX}")
message(STATUS "=================================================")
