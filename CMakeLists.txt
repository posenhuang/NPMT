# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.
#
# Copyright (c) Microsoft Corporation. All rights reserved
# Licensed under the BSD License

CMAKE_MINIMUM_REQUIRED(VERSION 2.8 FATAL_ERROR)
CMAKE_POLICY(VERSION 2.8)

FIND_PACKAGE(Torch REQUIRED)
FIND_PACKAGE(OpenMP)
FIND_PACKAGE(CUDA REQUIRED)

SET(CMAKE_CXX_FLAGS "-std=c++11 -Ofast")
IF(OpenMP_FOUND)
  SET(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
  SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
ENDIF()

SET(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -O3 -shared -Xcompiler -fPIC
    -gencode arch=compute_30,code=sm_30 -gencode arch=compute_35,code=sm_35
    -gencode arch=compute_37,code=sm_37 -gencode arch=compute_50,code=sm_50
    -gencode arch=compute_52,code=sm_52 -gencode arch=compute_52,code=compute_52")

INCLUDE_DIRECTORIES(${CMAKE_PREFIX_PATH}/include/THC)

# C++ library
IF(APPLE)
  SET(CMAKE_SHARED_LIBRARY_SUFFIX ".so")
ENDIF(APPLE)
FILE(GLOB CPPSRC fairseq/clib/*.cpp)
ADD_LIBRARY(fairseq_clib SHARED ${CPPSRC})
INSTALL(TARGETS fairseq_clib DESTINATION "${ROCKS_LIBDIR}")

ADD_LIBRARY(dp_lib SHARED fairseq/models/c_sample_dp.cc)
INSTALL(TARGETS dp_lib DESTINATION "${ROCKS_LIBDIR}")

CUDA_ADD_LIBRARY(compute_logpy_lib SHARED fairseq/models/compute_logpy.cu)
INSTALL(TARGETS compute_logpy_lib DESTINATION "${ROCKS_LIBDIR}")

TARGET_LINK_LIBRARIES(compute_logpy_lib ${CUDA_LIBRARIES})

# Lua library
INSTALL(DIRECTORY "fairseq" DESTINATION "${ROCKS_LUADIR}" FILES_MATCHING PATTERN "*.lua")

# Scripts and main executable
FOREACH(SCRIPT preprocess train tofloat generate generate-lines score optimize-fconv help)
  INSTALL(FILES "${SCRIPT}.lua" DESTINATION "${ROCKS_LUADIR}/fairseq/scripts")
ENDFOREACH(SCRIPT)
INSTALL(FILES "run.lua" DESTINATION "${ROCKS_BINDIR}" RENAME "fairseq")
