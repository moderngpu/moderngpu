# We can lower this if needed.
cmake_minimum_required(VERSION 3.19 FATAL_ERROR)

# begin /* Update MODERNGPU version */
set(MODERNGPU_VERSION_MAJOR 2)
set(MODERNGPU_VERSION_MINOR 13)
set(MODERNGPU_VERSION_PATCH 0)
# end /* Update MODERNGPU version */

set(MODERNGPU_VERSION "${MODERNGPU_VERSION_MAJOR}.${MODERNGPU_VERSION_MINOR}.${MODERNGPU_VERSION_PATCH}")

project(MODERNGPU
    VERSION ${MODERNGPU_VERSION}
    LANGUAGES CXX CUDA
)

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

set(PROJECT_DEPS_DIR externals)

set(EXECUTABLE_OUTPUT_PATH
  ${PROJECT_BINARY_DIR}/bin
  CACHE PATH
  "Directory where all executables will be stored")
set(LIBRARY_OUTPUT_PATH
  ${PROJECT_BINARY_DIR}/lib
  CACHE PATH
  "Directory where all libraries will be stored")
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_VERBOSE_MAKEFILE ON)

add_library(MODERNGPU INTERFACE)
set(MODERNGPU_SRC ${PROJECT_SOURCE_DIR}/src)

set(MODERNGPU_CUDA_ARCHITECTURES "60;61;70;75;80" CACHE STRING
    "CUDA architectures generated for moderngpu consumers")
option(MODERNGPU_BUILD_DEMO "Build moderngpu demos" ON)
option(MODERNGPU_BUILD_TUTORIAL "Build moderngpu tutorials" ON)
option(MODERNGPU_BUILD_TESTS "Build moderngpu unit tests" ON)

set_target_properties(MODERNGPU
    PROPERTIES
        CXX_STANDARD 11
        CXX_STANDARD_REQUIRED ON
        CXX_EXTENSIONS OFF
        CUDA_STANDARD 11
        CUDA_STANDARD_REQUIRED ON
        CUDA_EXTENSIONS OFF
        CUDA_RESOLVE_DEVICE_SYMBOLS ON
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_ARCHITECTURES "${MODERNGPU_CUDA_ARCHITECTURES}"
)

target_compile_definitions(MODERNGPU INTERFACE MODERNGPUVERSION=${MODERNGPU_VERSION})
target_compile_features(MODERNGPU INTERFACE cxx_std_11)
target_include_directories(MODERNGPU INTERFACE
    $<BUILD_INTERFACE:${MODERNGPU_SRC}>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
    $<BUILD_INTERFACE:${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}>
)
target_link_libraries(MODERNGPU INTERFACE cuda)

set(CXX_FLAGS
  $<$<CXX_COMPILER_ID:MSVC>:-D_SCL_SECURE_NO_WARNINGS>
  $<$<CXX_COMPILER_ID:GNU>:-Werror -Wundef>
)
set(CUDA_FLAGS
  --expt-extended-lambda
  --expt-relaxed-constexpr
  --use_fast_math
  --ptxas-options -v
  --generate-line-info
  $<$<CXX_COMPILER_ID:GNU>:-O3>
)
target_compile_options(MODERNGPU INTERFACE
    $<$<COMPILE_LANGUAGE:CXX>:${CXX_FLAGS}>
    $<$<COMPILE_LANGUAGE:CUDA>:${CUDA_FLAGS}>
)

if(MODERNGPU_BUILD_DEMO)
    add_subdirectory(demo)
endif()
if(MODERNGPU_BUILD_TUTORIAL)
    add_subdirectory(tutorial)
endif()
if(MODERNGPU_BUILD_TESTS)
    add_subdirectory(tests)
endif()

# ModernGPU public headers include .hxx files, so install them alongside
# .h/.hpp headers for downstream find_package() consumers.
install(DIRECTORY ${MODERNGPU_SRC}/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
    FILES_MATCHING PATTERN "*.h" PATTERN "*.hpp" PATTERN "*.hxx")
install(TARGETS MODERNGPU EXPORT ModerngpuTargets)
install(EXPORT ModerngpuTargets
    FILE ModerngpuTargets.cmake
    NAMESPACE Moderngpu::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/Moderngpu)

configure_package_config_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/cmake/ModerngpuConfig.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/ModerngpuConfig.cmake
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/Moderngpu
)
write_basic_package_version_file(
    ${CMAKE_CURRENT_BINARY_DIR}/ModerngpuConfigVersion.cmake
    VERSION ${MODERNGPU_VERSION}
    COMPATIBILITY SameMajorVersion
)
install(FILES
    ${CMAKE_CURRENT_BINARY_DIR}/ModerngpuConfig.cmake
    ${CMAKE_CURRENT_BINARY_DIR}/ModerngpuConfigVersion.cmake
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/Moderngpu
)
