
# Package config for lux-crypto
get_filename_component(PACKAGE_PREFIX_DIR "${CMAKE_CURRENT_LIST_DIR}/../../../" ABSOLUTE)

macro(set_and_check _var _file)
  set(${_var} "${_file}")
  if(NOT EXISTS "${_file}")
    message(FATAL_ERROR "File or directory ${_file} referenced by variable ${_var} does not exist !")
  endif()
endmacro()

macro(check_required_components _NAME)
  foreach(comp ${_FIND_COMPONENTS})
    if(NOT ${_NAME}_${comp}_FOUND)
      if(${_NAME}_FIND_REQUIRED_${comp})
        set(${_NAME}_FOUND FALSE)
      endif()
    endif()
  endforeach()
endmacro()

include(CMakeFindDependencyMacro)

find_dependency(lux-lattice CONFIG)

include("${CMAKE_CURRENT_LIST_DIR}/lux-cryptoTargets.cmake")

# Resource directory for runtime assets
set(LUX_crypto_RESOURCE_DIR "${PACKAGE_PREFIX_DIR}/share/lux/crypto")

check_required_components(lux-crypto)
