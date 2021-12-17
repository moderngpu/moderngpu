# We can lower this if needed.
cmake_minimum_required(VERSION 3.19 FATAL_ERROR)

# begin /* Update MODERNGPU version */
set(MODERNGPU_VERSION_MAJOR 2)
set(MODERNGPU_VERSION_MINOR 1)
set(MODERNGPU_VERSION_PATCH 3)
# end /* Update MODERNGPU version */

set(MODERNGPU_VERSION "${MODERNGPU_VERSION_MAJOR}.${MODERNGPU_VERSION_MINOR}.${MODERNGPU_VERSION_PATCH}")

project(MODERNGPU
    VERSION ${MODERNGPU_VERSION}
    LANGUAGES CXX CUDA
)

# begin /* Dependencies directory */
set(PROJECT_DEPS_DIR externals)
# end /* Dependencies directory */

## Set the directory where the binaries will be stored
set(EXECUTABLE_OUTPUT_PATH
  ${PROJECT_BINARY_DIR}/bin
  CACHE PATH
  "Directory where all executables will be stored")

## Set the directory where the libraries will be stored
set(LIBRARY_OUTPUT_PATH
  ${PROJECT_BINARY_DIR}/lib
  CACHE PATH
  "Directory where all the libraries will be stored")

## Export compile commands
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_VERBOSE_MAKEFILE ON)

############ ADD LIBRARY: MODERNGPU (HEADER-ONLY) ############
add_library(MODERNGPU INTERFACE)

####################################################
############### SET TARGET PROPERTIES ##############
####################################################
set_target_properties(MODERNGPU 
    PROPERTIES
        CXX_STANDARD 11
        CXX_STANDARD_REQUIRED ON
        CXX_EXTENSIONS OFF # Should this be turned on for MSVC?
        CUDA_STANDARD 11
        CUDA_STANDARD_REQUIRED ON
        CUDA_EXTENSIONS OFF
        CUDA_RESOLVE_DEVICE_SYMBOLS ON
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_ARCHITECTURES 60 61 70 75 80 # Set required architecture.
        # CUDA_PTX_COMPILATION ON # Can only be applied to OBJ.
)

# Use for later.
get_target_property(MODERNGPU_ARCHITECTURES 
    MODERNGPU 
        CUDA_ARCHITECTURES
)

####################################################
############ TARGET COMPILER DEFINITIONS ###########
####################################################
target_compile_definitions(MODERNGPU
  INTERFACE 
      MODERNGPUVERSION=${MODERNGPU_VERSION}
    #   MGPU_SM_TAG=${MODERNGPU_ARCHITECTURES}
)

####################################################
############ TARGET COMPILE FEATURES ###############
####################################################
# Turn C++ Standard 17 ON.
target_compile_features(MODERNGPU INTERFACE cxx_std_11)

####################################################
############ TARGET INCLUDE DIRECTORIES ############
####################################################
set(MODERNGPU_SRC ${PROJECT_SOURCE_DIR}/src)
target_include_directories(MODERNGPU
    INTERFACE ${MODERNGPU_SRC}
    INTERFACE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
)

####################################################
############ TARGET LINK LIBRARIES #################
####################################################
target_link_libraries(MODERNGPU
    # INTERFACE curand
    INTERFACE cuda
)

####################################################
################# TARGET SOURCES ###################
####################################################
# target_sources(MODERNGPU
#     INTERFACE "${CMAKE_CURRENT_SOURCE_DIR}/path/to/file.cpp"
# )

####################################################
############## SET CXX & CUDA FLAGS ################
####################################################
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
  $<$<CXX_COMPILER_ID:GNU>:-O3> # Host optimize-level
  # --verbose
  # --debug # Host debug
  # --device-debug # Device debug
)

####################################################
############ TARGET COMPILE OPTIONS ################
####################################################
target_compile_options(MODERNGPU INTERFACE 
    $<$<COMPILE_LANGUAGE:CXX>:${CXX_FLAGS}>
    $<$<COMPILE_LANGUAGE:CUDA>:${CUDA_FLAGS}>
)

####################################################
############# BUILD DEMO APPLICATIONS ##############
####################################################
option(MODERNGPU_BUILD_DEMO
  "If on, builds the moderngpu demo."
  OFF)

option(MODERNGPU_BUILD_TUTORIAL
  "If on, builds the moderngpu tutorials."
  OFF)

# Subdirectories for examples, testing and documentation
if(MODERNGPU_BUILD_DEMO)
    message(ERROR "Not implemented yet")
    add_subdirectory(demo)
endif(MODERNGPU_BUILD_DEMO)

if(MODERNGPU_BUILD_TUTORIAL)
    message(ERROR "Not implemented yet")
    add_subdirectory(tutorial)
endif(MODERNGPU_BUILD_TUTORIAL)

####################################################
################ BUILD UNIT TESTS  #################
####################################################
option(MODERNGPU_BUILD_TESTS
    "If on, builds the unit tests."
    ON)

# Subdirectories for examples, testing and documentation
if(MODERNGPU_BUILD_TESTS)
    add_subdirectory(tests)
endif(MODERNGPU_BUILD_TESTS)
