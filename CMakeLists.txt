cmake_minimum_required(VERSION 3.10)
project(rime-wanxiang)

set(CMAKE_CXX_STANDARD 17)

# ==========================================
# 1. 寻找 OpenMP (标准做法)
# ==========================================
find_package(OpenMP REQUIRED)

# ==========================================
# 2. 引擎路径标准化 (解耦核心)
# ==========================================
# 允许从外部传入 -DNCNN_ROOT=...，默认为你本地的路径
if(NOT NCNN_ROOT)
    set(NCNN_ROOT "/home/amz/Desktop/ncnn/build/install")
endif()

message(STATUS "[Wanxiang] Using NCNN_ROOT: ${NCNN_ROOT}")

# 标准化寻找头文件路径
find_path(NCNN_INCLUDE_DIR ncnn/net.h
    PATHS "${NCNN_ROOT}/include"
    NO_DEFAULT_PATH
)

# 标准化寻找静态库文件 (强制锁定 .a)
find_library(NCNN_LIBRARY NAMES ncnn libncnn.a
    PATHS "${NCNN_ROOT}/lib"
    NO_DEFAULT_PATH
)

if(NOT NCNN_LIBRARY OR NOT NCNN_INCLUDE_DIR)
    message(FATAL_ERROR "[Wanxiang] 找不到静态 NCNN 组件！请检查 NCNN_ROOT 路径。")
endif()

# ==========================================
# 3. 插件对象编译
# ==========================================
include_directories(src ${NCNN_INCLUDE_DIR} ${PROJECT_SOURCE_DIR}/src)

file(GLOB wanxiang_src "src/*.cc")
add_library(rime-wanxiang-objs OBJECT ${wanxiang_src})

# 编译参数：注入 OpenMP 标志
target_compile_options(rime-wanxiang-objs PRIVATE ${OpenMP_CXX_FLAGS})

# 确保位置无关代码 (对于编译成 .so 插件至关重要)
if(BUILD_SHARED_LIBS OR TRUE)
  set_target_properties(rime-wanxiang-objs PROPERTIES POSITION_INDEPENDENT_CODE ON)
endif()

# ==========================================
# 4. 变量传递给 Rime 主控 (插件标准接口)
# ==========================================
set(plugin_name rime-wanxiang PARENT_SCOPE)
set(plugin_objs $<TARGET_OBJECTS:rime-wanxiang-objs> PARENT_SCOPE)

# 💡 核心依赖包 (零耦合方案):
# 1. ${rime_library}: Rime 核心
# 2. ${NCNN_LIBRARY}: 强制路径的静态库 (.a)
# 3. -pthread: 底层线程支持
# 4. ${OpenMP_CXX_LIBRARIES}: OpenMP 运行库支持
set(plugin_deps 
    ${rime_library} 
    "${NCNN_LIBRARY}" 
    "pthread" 
    ${OpenMP_CXX_LIBRARIES} 
    PARENT_SCOPE
)

set(plugin_modules "wanxiang" PARENT_SCOPE)