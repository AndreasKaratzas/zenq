# Add test executable
add_executable(kernel_test_driver
    driver.cpp
)

# Link against main library
target_link_libraries(kernel_test_driver
    PRIVATE
        kernel_compute
)