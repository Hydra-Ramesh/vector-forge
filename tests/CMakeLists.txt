include(FetchContent)
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/03597a01ee50ed33e9dfd640b249b4be3799d395.zip
)
# For Windows: Prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

add_executable(vectorforge_tests
    test_dataset.cpp
    test_brute_force_index.cpp
    test_storage.cpp
    test_ivf_index.cpp
    test_ivfpq_index.cpp
    test_vamana_index.cpp
)
target_link_libraries(vectorforge_tests PRIVATE vectorforge_lib GTest::gtest_main)

include(GoogleTest)
gtest_discover_tests(vectorforge_tests)
