#----------------------------------------------------------------
# Generated CMake target import file.
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "lux::crypto" for configuration ""
set_property(TARGET lux::crypto APPEND PROPERTY IMPORTED_CONFIGURATIONS NOCONFIG)
set_target_properties(lux::crypto PROPERTIES
  IMPORTED_LOCATION_NOCONFIG "${_IMPORT_PREFIX}/lib/libluxcrypto.1.0.0.dylib"
  IMPORTED_SONAME_NOCONFIG "@rpath/libluxcrypto.1.0.0.dylib"
  )

list(APPEND _cmake_import_check_targets lux::crypto )
list(APPEND _cmake_import_check_files_for_lux::crypto "${_IMPORT_PREFIX}/lib/libluxcrypto.1.0.0.dylib" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
