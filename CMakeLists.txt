
cmake_minimum_required(VERSION 3.15.0)

project(demo )



find_package(PopSift CONFIG REQUIRED)
if(PopSift_FOUND)
    message(STATUS "popsift found")
else()
    message(WARNING "popsift not found")
endif()

find_package(CUDA REQUIRED)
if(CUDA_FOUND)
    message(STATUS "cuda found")
else()
    message(WARNING "cuda not found")
endif()

set(pybind11_dir "../pybind11")
set(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR})

add_subdirectory(${pybind11_dir} pybind11.out)

find_package(PythonInterp REQUIRED)

include_directories( ${PopSift_INCLUDE_DIRS} ${CUDA_INCLUDE_DIRS})


add_library(ZHSift
  MODULE
    ZHSift.cpp
  )

target_link_libraries(ZHSift  PopSift::popsift pybind11::module ${CUDA_LIBS})
set_target_properties(ZHSift
  PROPERTIES
    PREFIX "${PYTHON_MODULE_PREFIX}"
    SUFFIX "${PYTHON_MODULE_EXTENSION}"
  )