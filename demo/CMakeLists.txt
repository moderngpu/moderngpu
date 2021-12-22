# This is not the recommended method.
# See: https://stackoverflow.com/a/14307434/5729690
file(GLOB DEMO_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/*.cu)
foreach(SOURCE_FILE ${DEMO_SOURCES})
    get_filename_component(DEMO_NAME ${SOURCE_FILE} NAME_WLE)
    add_executable(${DEMO_NAME} ${SOURCE_FILE} graph.cxx)
    target_link_libraries(${DEMO_NAME} PRIVATE MODERNGPU)
    get_target_property(MODERNGPU_ARCHITECTURES MODERNGPU CUDA_ARCHITECTURES)
    set_target_properties(${DEMO_NAME} 
        PROPERTIES 
            CUDA_ARCHITECTURES ${MODERNGPU_ARCHITECTURES}
    ) # XXX: Find a better way to inherit MODERNGPU properties.
    message(STATUS "Demo Added: ${DEMO_NAME}")
endforeach(SOURCE_FILE ${DEMO_SOURCES})