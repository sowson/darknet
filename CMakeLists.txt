cmake_minimum_required(VERSION 3.7.2 FATAL_ERROR)

if (${CMAKE_SYSTEM_NAME} MATCHES "Unix")
    set(CMAKE_C_COMPILER "gcc")
endif()
set(CMAKE_CXX_COMPILER "g++")

set(CMAKE_CXX_STANDARD 11)

if (${CMAKE_SYSTEM_NAME} MATCHES "Unix")
    project(darknet LANGUAGES C CXX)
else()
    project(darknet LANGUAGES CXX)
endif()

option(DARKNET_CPU "Enable CPU support" OFF)
option(DARKNET_OPENCV "Enable OpenCV support" ON)
option(DARKNET_OPENMP "Enable OpenMP support" OFF)
option(DARKNET_GPU "Enable OpenCL support" ON)
option(DARKNET_GPU_FAST "Enable OpenCL fast kernels support" ON)
option(DARKNET_GPU_MULTI "Enable OpenCL multi-GPU support" OFF)
option(DARKNET_ARM "Enable ARM support" OFF)
option(DARKNET_BENCHMARK "Some Benchmark Stats" OFF)
option(DARKNET_LOSS_ONLY "Loss Only Stats" OFF)
option(DARKNET_TESTING "Build the unit tests" OFF)
option(DARKNET_DEBUG "Enable debug mode" OFF)
option(DARKNET_STATIC "Build static library" ON)
option(DARKNET_SHARED "Build shared library" ON)
option(DARKNET_NO_BINARY "Do not build binary" OFF)

include(GNUInstallDirs)

include_directories("include" "src")

set (DARKNET_DEFINITIONS "")
set (DARKNET_BINARY bindarknet)
set (DARKNET_LIBRARY libdarknet)
set (DARKNET_LIBRARY_STATIC libdarknet_s)

message(STATUS ${CMAKE_SYSTEM_NAME})

if (DARKNET_DEBUG OR DARKNET_TESTING)
    list (APPEND DARKNET_CFLAGS "-g")
else()
	if (DARKNET_OPENMP)
		list (APPEND DARKNET_CFLAGS "-openmp")
	endif()
    if (${CMAKE_SYSTEM_NAME} MATCHES "Linux") 
	    list (APPEND DARKNET_CFLAGS "-Wno-unused-variable")
	    list (APPEND DARKNET_CFLAGS "-Wno-unused-result")
	    list (APPEND DARKNET_CFLAGS "-Wno-deprecated-declarations")
	    list (APPEND DARKNET_CFLAGS "-Wno-return-type-c-linkage")
	    list (APPEND DARKNET_CFLAGS "-O3")
    else()
        # this is for test
        list(APPEND DARKNET_LINK_CFLAGS "/Debug")
        list(APPEND DARKNET_CFLAGS "/ZI")
        list(APPEND DARKNET_CFLAGS "/Od")

        add_library(POSIX INTERFACE IMPORTED)
        set_property(TARGET POSIX PROPERTY INTERFACE_INCLUDE_DIRECTORIES "D:/workspace/libunistd/unistd")
        set_property(TARGET POSIX PROPERTY INTERFACE_LINK_DIRECTORIES "D:/workspace/libunistd/build/Release")
        set_property(TARGET POSIX PROPERTY INTERFACE_LINK_LIBRARIES "libunistd.lib")
        add_library(pthread INTERFACE IMPORTED)
        set_property(TARGET pthread PROPERTY INTERFACE_INCLUDE_DIRECTORIES ${PROJECT_SOURCE_DIR}/3rdparty/pthreads/include)
        set_property(TARGET pthread PROPERTY INTERFACE_LINK_DIRECTORIES ${PROJECT_SOURCE_DIR}/3rdparty/pthreads/lib)
        set_property(TARGET pthread PROPERTY INTERFACE_LINK_LIBRARIES "pthreadVC2.lib")
        add_library(stb INTERFACE IMPORTED)
        set_property(TARGET stb PROPERTY INTERFACE_INCLUDE_DIRECTORIES ${PROJECT_SOURCE_DIR}/3rdparty/stb/include)
    endif()
endif()

# Find OpenCV
if (DARKNET_OPENCV)
    set(OpenCV_DIV "unset" CACHE STRING "")
	find_package(OpenCV 3.1 REQUIRED PATHS ${OpenCV_DIR})
	add_library(opencv INTERFACE IMPORTED)
    message(OPENCV_INCLUDE_DIRS: ${OpenCV_INCLUDE_DIRS})
	set_property(TARGET opencv PROPERTY
		INTERFACE_INCLUDE_DIRECTORIES ${OpenCV_INCLUDE_DIRS})
    message(OPENCV_LIBRARIES: ${OpenCV_LIBS})
    set_property(TARGET opencv PROPERTY
        INTERFACE_LINK_LIBRARIES ${OpenCV_LIBRARIES})
    message(OPENCV_LIB_PATH: ${OpenCV_LIB_PATH})
    set_property(TARGET opencv PROPERTY INTERFACE_LINK_DIRECTORIES ${OpenCV_LIB_PATH})
    set_property(TARGET opencv PROPERTY INTERFACE_INCLUDE_DIRECTORIES ${OpenCV_DIR}/include)
    set_property(TARGET opencv PROPERTY INTERFACE_LINK_DIRECTORIES ${OpenCV_DIR}/lib ${OpenCV_DIR}/staticlib)
    set_property(TARGET opencv PROPERTY INTERFACE_LINK_LIBRARIES
    opencv_calib3d310.lib
    opencv_core310.lib
    opencv_features2d310.lib
    opencv_flann310.lib
    opencv_highgui310.lib
    opencv_imgcodecs310.lib
    opencv_imgproc310.lib
    opencv_ml310.lib
    opencv_objdetect310.lib
    opencv_photo310.lib
    opencv_shape310.lib
    opencv_stitching310.lib
    opencv_superres310.lib
    opencv_ts310.lib
    opencv_video310.lib
    opencv_videoio310.lib
    opencv_videostab310.lib)
    list (APPEND DARKNET_DEFINITIONS OPENCV)
endif()

# Find OpenCL
if (DARKNET_GPU)
    set(OpenCL_DIR "unset" CACHE STRING "")
	find_package(OpenCL REQUIRED)
    add_library(opencl INTERFACE IMPORTED)
    message(CL_INCLUDE_DIR: ${OpenCL_INCLUDE_DIRS})
    set_property(TARGET opencl PROPERTY
    	INTERFACE_INCLUDE_DIRECTORIES ${OpenCL_INCLUDE_DIRS})
    message(CL_LIBS: ${OpenCL_LIBRARY})
    set_property(TARGET opencl PROPERTY
        INTERFACE_LINK_LIBRARIES ${OpenCL_LIBRARY})
    list (APPEND DARKNET_DEFINITIONS GPU CL_TARGET_OPENCL_VERSION=120)
	# Raspberry PI
	if (DARKNET_ARM)
		list (APPEND DARKNET_DEFINITIONS ARM)
	else()
        set(clBLAS_DIR "unset" CACHE STRING "")
		find_package(clBLAS REQUIRED PATHS "${clBLAS_DIR}")
		add_library(clblas INTERFACE IMPORTED)
        message(clblas_INCLUDE_DIRS:${CLBLAS_INCLUDE_DIRS})
		set_property(TARGET clblas PROPERTY
			INTERFACE_INCLUDE_DIRECTORIES ${CLBLAS_INCLUDE_DIRS})
        message(clblas_LIBRARIES:${CLBLAS_LIBRARIES})
		set_property(TARGET clblas PROPERTY
			INTERFACE_LINK_LIBRARIES ${CLBLAS_LIBRARIES})
	endif()
endif()

# GPU Training
if (DARKNET_GPU)
	list (APPEND DARKNET_DEFINITIONS GPU)
endif()

# Fast Training
if (DARKNET_GPU_FAST)
	list (APPEND DARKNET_DEFINITIONS GPU_FAST)
endif()

# Multi-GPU Training
if (DARKNET_GPU_MULTI)
	list (APPEND DARKNET_DEFINITIONS GPU_MULTI)
endif()

# Benchmark Stats
if (DARKNET_BENCHMARK)
	list (APPEND DARKNET_DEFINITIONS BENCHMARK)
endif()

# Loss Stats
if (DARKNET_LOSS_ONLY)
	list (APPEND DARKNET_DEFINITIONS LOSS_ONLY)
endif()

# Find Catch
if (DARKNET_UNIT)
	set (DARKNET_CPU OFF)
	find_package(Catch REQUIRED)
	add_library(catch INTERFACE IMPORTED)
	set_property(TARGET catch PROPERTY
		INTERFACE_INCLUDE_DIRECTORIES ${Catch_INCLUDE_DIRS})
endif()

set (DARKNET_SOURCES
    "src/activation_layer.c"
    "src/activations.c"
    "examples/art.c"
    "examples/attention.c"
    "src/avgpool_layer.c"
    "src/batchnorm_layer.c"
    "src/blas.c"
    "src/box.c"
    "examples/captcha.c"
    "examples/cifar.c"
    "examples/classifier.c"
    "examples/coco.c"
    "src/col2im.c"
    "src/compare.c"
    "src/connected_layer.c"
    "src/convolutional_layer.c"
    "src/cost_layer.c"
    "src/crnn_layer.c"
    "src/crop_layer.c"
    "src/data.c"
    "src/deconvolutional_layer.c"
    "src/demo.c"
    "src/detection_layer.c"
    "examples/detector.c"
    "examples/dice.c"
    "src/dropout_layer.c"
    "src/gemm.c"
    "examples/go.c"
    "src/gru_layer.c"
    "src/im2col.c"
    "src/image.c"
    "src/iseg_layer.c"
    "examples/instance-segmenter.c"
    "src/layer.c"
    "src/list.c"
    "src/local_layer.c"
    "src/lstm_layer.c"
    "examples/lsd.c"
    "examples/cgan.c"
    "src/matrix.c"
    "src/maxpool_layer.c"
    "src/network.c"
    "examples/nightmare.c"
    "src/normalization_layer.c"
    "src/option_list.c"
    "src/parser.c"
    "src/region_layer.c"
    "examples/regressor.c"
    "src/reorg_layer.c"
    "examples/rnn.c"
    "src/rnn_layer.c"
    "examples/rnn_vid.c"
    "src/route_layer.c"
    "src/shortcut_layer.c"
    "src/softmax_layer.c"
    "examples/segmenter.c"
    "examples/super.c"
    "examples/swag.c"
    "examples/tag.c"
    "src/tree.c"
    "src/utils.c"
    "examples/voxel.c"
    "examples/writing.c"
    "examples/yolo.c"
    "src/yolo_layer.c"
    "src/yolo4_layer.c"
    "src/upsample_layer.c"
    "src/logistic_layer.c"
    "src/l2norm_layer.c"
)

if (${CMAKE_SYSTEM_NAME} MATCHES "Windows")
    set_source_files_properties(${DARKNET_SOURCES} PROPERTIES LANGUAGE "CXX")
endif()

set (DARKNET_PUBLIC_HEADERS
    "include/darknet.h"
    "src/activation_layer.h"
    "src/activations.h"
    "src/avgpool_layer.h"
    "src/batchnorm_layer.h"
    "src/blas.h"
    "src/box.h"
   #"src/bugfixes.h"
    "src/classifier.h"
    "src/col2im.h"
    "src/connected_layer.h"
    "src/convolutional_layer.h"
    "src/cost_layer.h"
    "src/crnn_layer.h"
    "src/crop_layer.h"
    "src/opencl.h"
    "src/data.h"
    "src/deconvolutional_layer.h"
    "src/demo.h"
    "src/detection_layer.h"
    "src/dropout_layer.h"
    "src/gemm.h"
    "src/gru_layer.h"
    "src/im2col.h"
    "src/image.h"
    "src/iseg_layer.h"
    "src/layer.h"
    "src/list.h"
    "src/local_layer.h"
    "src/lstm_layer.h"
    "src/matrix.h"
    "src/maxpool_layer.h"
    "src/network.h"
    "src/normalization_layer.h"
    "src/option_list.h"
    "src/parser.h"
    "src/region_layer.h"
    "src/reorg_layer.h"
    "src/rnn_layer.h"
    "src/route_layer.h"
    "src/shortcut_layer.h"
    "src/softmax_layer.h"
    "src/stb_image.h"
    "src/stb_image_write.h"
    "src/tree.h"
   #"src/unit.h"
    "src/utils.h"
    "src/opencl.h"
    "src/yolo_layer.h"
    "src/yolo4_layer.h"
    "src/upsample_layer.h"
    "src/logistic_layer.h"
    "src/l2norm_layer.h"
)

set (DARKNET_MAIN_SOURCE
    "examples/darknet.c"
)

if (${CMAKE_SYSTEM_NAME} MATCHES "Windows")
    set_source_files_properties(${DARKNET_MAIN_SOURCE} PROPERTIES LANGUAGE "CXX")
endif()

set (DARKNET_SOURCES_CXX
    "src/image_opencv.cpp"
)

if (DARKNET_CPU)
	set (DARKNET_CPU_SOURCES
			"src/cpu.c"
			)
    if (${CMAKE_SYSTEM_NAME} MATCHES "Windows")
        set_source_files_properties(${DARKNET_CPU} PROPERTIES LANGUAGE "CXX")
    endif()
endif()

if (DARKNET_GPU)
	set (DARKNET_GPU_SOURCES
	    "src/activation_kernels.c"
	    #"src/activation_kernels.cl"
	    "src/avgpool_layer_kernels.c"
	    #"src/avgpool_layer_kernels.cl"
	    "src/blas_kernels.c"
	    #"src/blas_kernels.cl"
	    "src/col2im_kernels.c"
	    #"src/col2im_kernels.cl"
	    "src/convolutional_kernels.c"
	    #"src/convolutional_kernels.cl"
	    "src/crop_layer_kernels.c"
	    #"src/crop_layer_kernels.cl"
	    "src/deconvolutional_kernels.c"
	    "src/dropout_layer_kernels.c"
	    #"src/dropout_layer_kernels.cl"
	    "src/im2col_kernels.c"
	    #"src/im2col_kernels.cl"
	    "src/maxpool_layer_kernels.c"
	    #"src/maxpool_layer_kernels.cl"
	    "src/opencl.c"
	)
    if (${CMAKE_SYSTEM_NAME} MATCHES "Windows")
        set_source_files_properties(${DARKNET_GPU_SOURCES} PROPERTIES LANGUAGE "CXX")
    endif()
endif()

if (DARKNET_UNIT)
	set (DARKNET_UNIT_SOURCES
	    "src/unit.c"
	    "src/unit.h"
	    "src/blas_unit.c"
	    "src/col2im_unit.c"
	    "src/convolutional_unit.c"
	    "src/gemm_unit.c"
	    "src/maxpool_unit.c"
	    "src/network_unit.c"
	    "src/region_unit.c"
	)
endif()

function (ConfigureBinary target)
    target_compile_definitions(${target} PUBLIC ${DARKNET_DEFINITIONS})
    target_compile_options(${target} PRIVATE ${DARKNET_CFLAGS})
    target_link_options(${target} PRIVATE ${DARKNET_LINK_CFLAGS})#debug

    if (${CMAKE_SYSTEM_NAME} MATCHES "Windows") 
        target_link_libraries(${target} POSIX)
        target_link_libraries(${target} pthread)
        target_link_libraries(${target} stb)
    endif()

    if (DARKNET_ARM)
        target_link_libraries(${target} opencl)
    elseif (DARKNET_GPU)
        target_link_libraries(${target} opencl clblas)
    endif()

    if (DARKNET_OPENCV)
        target_link_libraries(${target} opencv)
    endif()

#    if (DARKNET_TESTING)
#        target_link_libraries(${target} catch)
#    endif()

    if (${CMAKE_SYSTEM_NAME} MATCHES "Unix")
        target_link_libraries(${target} "m" "pthread")
    endif()
endfunction()

function (ConfigureLibrary target)
    target_include_directories(${target} PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src>
        $<INSTALL_INTERFACE:include>
    )
    install(TARGETS ${target} EXPORT ${CMAKE_PROJECT_NAME}
        ARCHIVE  DESTINATION ${CMAKE_INSTALL_LIBDIR}
        LIBRARY  DESTINATION ${CMAKE_INSTALL_LIBDIR}
        RUNTIME  DESTINATION ${CMAKE_INSTALL_BINDIR})
endfunction()

if (DARKNET_SHARED)
    add_library(${DARKNET_LIBRARY} SHARED ${DARKNET_SOURCES} ${DARKNET_CPU_SOURCES}
	   ${DARKNET_GPU_SOURCES} ${DARKNET_WINDOWS_SOURCES} ${DARKNET_SOURCES_CXX} ${DARKNET_PUBLIC_HEADERS})
    set_target_properties(${DARKNET_LIBRARY} PROPERTIES OUTPUT_NAME "yolo_cl_cpp_dll")
    ConfigureBinary(${DARKNET_LIBRARY})
    ConfigureLibrary(${DARKNET_LIBRARY})
endif()

if (DARKNET_STATIC)
    add_library(${DARKNET_LIBRARY_STATIC} STATIC ${DARKNET_SOURCES} ${DARKNET_CPU_SOURCES}
       ${DARKNET_GPU_SOURCES} ${DARKNET_WINDOWS_SOURCES} ${DARKNET_SOURCES_CXX})
    set_target_properties(${DARKNET_LIBRARY_STATIC} PROPERTIES OUTPUT_NAME "yolo_cl_cpp_dll_s")    
    ConfigureBinary(${DARKNET_LIBRARY_STATIC})
    ConfigureLibrary(${DARKNET_LIBRARY_STATIC})
endif()


if (NOT DARKNET_NO_BINARY)
    add_executable(${DARKNET_BINARY} ${DARKNET_SOURCES} ${DARKNET_CPU_SOURCES}
        ${DARKNET_GPU_SOURCES} ${DARKNET_WINDOWS_SOURCES} ${DARKNET_SOURCES_CXX}
        ${DARKNET_MAIN_SOURCE} ${DARKNET_PUBLIC_HEADERS})
    set_target_properties(${DARKNET_BINARY} PROPERTIES OUTPUT_NAME "darknet")
    target_compile_definitions(${DARKNET_BINARY} PRIVATE DARKNET_BIN)

    ConfigureBinary(${DARKNET_BINARY})
endif()

install(FILES ${DARKNET_PUBLIC_HEADERS} ${DARKNET_WINDOWS_PUBLIC_HEADERS} DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/${CMAKE_PROJECT_NAME})

install(EXPORT ${CMAKE_PROJECT_NAME} DESTINATION share/${CMAKE_PROJECT_NAME}/cmake)

if (DARKNET_SHARED)
    export(TARGETS ${DARKNET_LIBRARY} FILE darknetConfig.cmake)
endif()
