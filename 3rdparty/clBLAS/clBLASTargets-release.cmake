#----------------------------------------------------------------
# Generated CMake target import file for configuration "Release".
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "clBLAS" for configuration "Release"
set_property(TARGET clBLAS APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(clBLAS PROPERTIES
  IMPORTED_IMPLIB_RELEASE "${_IMPORT_PREFIX}/lib64/import/clBLAS.lib"
  IMPORTED_LINK_INTERFACE_LIBRARIES_RELEASE "C:/Program Files (x86)/AMD APP SDK/3.0/lib/x86_64/OpenCL.lib"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/bin/clBLAS.dll"
  )

list(APPEND _IMPORT_CHECK_TARGETS clBLAS )
list(APPEND _IMPORT_CHECK_FILES_FOR_clBLAS "${_IMPORT_PREFIX}/lib64/import/clBLAS.lib" "${_IMPORT_PREFIX}/bin/clBLAS.dll" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
