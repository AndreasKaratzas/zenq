# Collect source files
file(GLOB_RECURSE CPU_SOURCES 
    "*.cpp"
)

file(GLOB_RECURSE CUDA_SOURCES 
    "*.cu"
)

# Create library
add_library(kernel_compute
    ${CPU_SOURCES}
    ${CUDA_SOURCES}
)

# Set include directories
target_include_directories(kernel_compute
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
    PRIVATE
        ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
)

# Link dependencies
target_link_libraries(kernel_compute
    PUBLIC
        CUDA::cudart
        CUDA::cublas
        CUDA::cusparse
)

# Installation
install(TARGETS kernel_compute
    EXPORT kernel_compute-targets
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)