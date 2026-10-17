cmake_minimum_required(VERSION 3.16)
project(rk3588s_video_struct LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 板端目标。交叉编译示例：
#   aarch64-linux-gnu-g++ 在开发容器里；MPP / RGA / RKNN 头文件和库来自板子 SDK。
add_executable(video_struct src/main.cpp)
