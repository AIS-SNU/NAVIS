set(CMAKE_CXX_STANDARD 17)

file(GLOB PROJECT_CPP *.cpp search/*.cpp update/*.cpp utils/*.cpp v2/*.cpp)
add_library(${PROJECT_NAME} ${PROJECT_CPP})

