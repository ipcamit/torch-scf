cmake_minimum_required(VERSION 3.10)
project(pybind11-scf)

set(CMAKE_CXX_STANDARD 17)

if (NOT DEFINED Enzyme_DIR)
    set(Enzyme_DIR /opt/enzyme/enzyme/build/Enzyme)
endif()

set(CMAKE_BUILD_TYPE Release)

string(REGEX MATCH "^[0-9]+" LLVM_MAJOR_VERSION ${CMAKE_CXX_COMPILER_VERSION})

find_package(pybind11 REQUIRED)

if (NOT DEFINED LLD)
    set(LLD "lld")
endif()

pybind11_add_module(scf integrals.cpp)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fuse-ld=${LLD} -flto -Xclang -new-struct-path-tbaa")

if (${LLVM_MAJOR_VERSION} LESS_EQUAL 13)
    target_link_options(scf PRIVATE "LINKER:--lto-legacy-pass-manager")
    target_link_options(scf PRIVATE "LINKER:-mllvm=-load=${Enzyme_DIR}/LLDEnzyme-${LLVM_MAJOR_VERSION}.so")
else()
    target_link_options(scf PRIVATE "LINKER:--load-pass-plugin=${Enzyme_DIR}/LLDEnzyme-${LLVM_MAJOR_VERSION}.so")
endif()
