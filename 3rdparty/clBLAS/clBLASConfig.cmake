include(${CMAKE_CURRENT_LIST_DIR}/clBLASTargets.cmake)
get_filename_component(CLBLAS_INCLUDE_DIRS ${CMAKE_CURRENT_LIST_DIR}/../include ABSOLUTE)
set(CLBLAS_LIBRARIES clBLAS)
