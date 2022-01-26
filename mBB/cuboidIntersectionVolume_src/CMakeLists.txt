cmake_minimum_required(VERSION 3.11)

project(cuboidIntersectionVolume)

find_package(PythonLibs 3.8 REQUIRED)
find_package(pybind11 REQUIRED)

# execute_process(COMMAND python3-config --extension-suffix
#     OUTPUT_VARIABLE python_suffix
#     OUTPUT_STRIP_TRAILING_WHITESPACE
# )
# 
# add_library(cuboidIntersectionVolume SHARED
#     cuboidIntersectionVolume.cpp
#     gpc/gpc.c
# )
# target_link_libraries(cuboidIntersectionVolume
#     ${PYTHON_LIBRARIES}
#     pybind11::module
# )
# target_include_directories(cuboidIntersectionVolume
#     PRIVATE gpc
#     PRIVATE ${PYTHON_INCLUDE_PATH}
# )
# set_target_properties(cuboidIntersectionVolume PROPERTIES SUFFIX "${python_suffix}")
# set_target_properties(cuboidIntersectionVolume PROPERTIES PREFIX "")
pybind11_add_module(cuboidIntersectionVolume MODULE
    cuboidIntersectionVolume.cpp
    gpc/gpc.c
)
set_source_files_properties(gpc/gpc.c PROPERTIES LANGUAGE CXX)
target_include_directories(cuboidIntersectionVolume
    PRIVATE gpc
)


# install(TARGETS cuboidIntersectionVolume
#     LIBRARY DESTINATION ..)


add_executable(test
    test.cpp
)
target_link_libraries(test
    pthread
    util
    dl
    ${PYTHON_LIBRARIES}
    pybind11::embed
)
target_include_directories(test
    PRIVATE ${PYTHON_INCLUDE_PATH}
)