# This is not the recommended method.
# See: https://stackoverflow.com/a/14307434/5729690
file(GLOB TUTORIAL_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/*.cu)
foreach(SOURCE_FILE ${TUTORIAL_SOURCES})
    get_filename_component(TUTORIAL_NAME ${SOURCE_FILE} NAME_WLE)
    add_executable(${TUTORIAL_NAME} ${SOURCE_FILE})
    target_link_libraries(${TUTORIAL_NAME} PRIVATE MODERNGPU)
    get_target_property(MODERNGPU_ARCHITECTURES MODERNGPU CUDA_ARCHITECTURES)
    set_target_properties(${TUTORIAL_NAME} 
        PROPERTIES 
            CUDA_ARCHITECTURES ${MODERNGPU_ARCHITECTURES}
    ) # XXX: Find a better way to inherit MODERNGPU properties.
    message(STATUS "Tutorial Added: ${TUTORIAL_NAME}")
endforeach(SOURCE_FILE ${TUTORIAL_SOURCES})