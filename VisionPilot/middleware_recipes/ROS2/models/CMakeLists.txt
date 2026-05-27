cmake_minimum_required(VERSION 3.14)
project(models)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# Check the ROS_DISTRO environment variable to set a compile-time macro
# It's used to fix the cv_bridge compatibility issue with ROS Humble and later versions
set(ROS_DISTRO_ENV $ENV{ROS_DISTRO})
if(ROS_DISTRO_ENV STREQUAL "humble")
    add_definitions(-DROS_HUMBLE)
    message(STATUS "ROS_DISTRO is 'humble'. Defining MACRO: ROS_HUMBLE")
else()
    message(STATUS "ROS_DISTRO is set to '${ROS_DISTRO_ENV}'. ROS_HUMBLE MACRO not defined.")
endif()

# Find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclcpp_components REQUIRED)
find_package(image_transport REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(OpenCV 4 REQUIRED)
find_package(CUDA REQUIRED)

# Optional: vision_msgs for AutoSpeed detection node
find_package(vision_msgs QUIET)
if(vision_msgs_FOUND)
  message(STATUS "vision_msgs found - AutoSpeed node will be built")
else()
  message(WARNING "vision_msgs not found - AutoSpeed node will be skipped. Install with: sudo apt install ros-$ENV{ROS_DISTRO}-vision-msgs")
endif()

# --- HIP Configuration (Optional) ---
# Check if HIP compiler actually exists before trying to use it
if(EXISTS "/opt/rocm/bin/hipcc" OR EXISTS "/opt/rocm-6.2.4/bin/hipcc")
  find_package(HIP QUIET)
  if(HIP_FOUND)
    message(STATUS "HIP found: ${HIP_VERSION}")
    enable_language(HIP)
    message(STATUS "HIP language enabled")
  else()
    message(STATUS "HIP not found, HIP kernels will be disabled")
  endif()
else()
  message(STATUS "HIP compiler not found, skipping HIP support")
  set(HIP_FOUND FALSE)
endif()

# --- TensorRT Configuration ---
find_path(TENSORRT_INCLUDE_DIR NvInfer.h HINTS /usr/include/x86_64-linux-gnu/ /opt/tensorrt/include)
find_library(TENSORRT_LIBRARY nvinfer HINTS /usr/lib/x86_64-linux-gnu/)
find_library(TENSORRT_ONNXPARSER_LIBRARY nvonnxparser HINTS /usr/lib/x86_64-linux-gnu/)
if(NOT TENSORRT_INCLUDE_DIR OR NOT TENSORRT_LIBRARY)
    message(FATAL_ERROR "Failed to find TensorRT.")
endif()

# --- ONNX Runtime Configuration ---
option(ONNXRUNTIME_ROOTDIR "Path to ONNX Runtime root directory" "")
if(NOT ONNXRUNTIME_ROOTDIR)
    message(FATAL_ERROR "You must provide the ONNX Runtime root directory via -DONNXRUNTIME_ROOTDIR")
endif()

include_directories(
  "${ONNXRUNTIME_ROOTDIR}/include"
)
link_directories("${ONNXRUNTIME_ROOTDIR}/lib")

# --- MemryX MX3 Configuration (optional) ---
find_path(MEMRYX_INCLUDE_DIR memx/accl/MxAccl.h HINTS /usr/include /usr/local/include)
find_library(MEMRYX_LIBRARY mx_accl HINTS /usr/lib/x86_64-linux-gnu /usr/local/lib)
if(MEMRYX_INCLUDE_DIR AND MEMRYX_LIBRARY)
  message(STATUS "MemryX SDK found: ${MEMRYX_LIBRARY}")
  set(MEMRYX_FOUND TRUE)
else()
  message(STATUS "MemryX SDK not found (install memx-accl for MemryX backend support)")
  set(MEMRYX_FOUND FALSE)
endif()

# --- DeepX M1 Configuration (optional) ---
find_path(DEEPX_INCLUDE_DIR dxrt/dxrt_api.h HINTS /usr/local/include /usr/include)
find_library(DEEPX_LIBRARY dxrt HINTS /usr/local/lib /usr/lib /usr/lib/x86_64-linux-gnu)
if(DEEPX_INCLUDE_DIR AND DEEPX_LIBRARY)
  message(STATUS "DeepX SDK found: ${DEEPX_LIBRARY}")
  set(DEEPX_FOUND TRUE)
else()
  message(STATUS "DeepX SDK not found (install dx-runtime for DeepX backend support)")
  set(DEEPX_FOUND FALSE)
endif()

# --- Hailo-8 Configuration (optional) ---
find_path(HAILORT_INCLUDE_DIR hailo/hailort.hpp HINTS /usr/include /usr/local/include)
find_library(HAILORT_LIBRARY hailort HINTS /usr/lib /usr/local/lib /usr/lib/x86_64-linux-gnu)
if(HAILORT_INCLUDE_DIR AND HAILORT_LIBRARY)
  message(STATUS "HailoRT SDK found: ${HAILORT_LIBRARY}")
  set(HAILO_FOUND TRUE)
else()
  message(STATUS "HailoRT SDK not found (install hailort for Hailo-8 backend support)")
  set(HAILO_FOUND FALSE)
endif()

# --- Axelera Metis Configuration (optional) ---
find_path(AXELERA_INCLUDE_DIR axruntime/axruntime.h
  HINTS /opt/axelera/runtime-1.5.3-1/include /opt/axelera/runtime/include /usr/local/include /usr/include)
find_library(AXELERA_LIBRARY axruntime
  HINTS /opt/axelera/runtime-1.5.3-1/lib /opt/axelera/runtime/lib /usr/local/lib /usr/lib /usr/lib/x86_64-linux-gnu)
if(AXELERA_INCLUDE_DIR AND AXELERA_LIBRARY)
  message(STATUS "Axelera runtime SDK found: ${AXELERA_LIBRARY}")
  set(AXELERA_FOUND TRUE)
else()
  message(STATUS "Axelera runtime SDK not found (install runtime-1.5.x-1 for Axelera backend support)")
  set(AXELERA_FOUND FALSE)
endif()

# --- Inference Backend Library ---
set(INFERENCE_BACKEND_SOURCES
  ../../common/backends/onnx_runtime_backend.cpp
  ../../common/backends/tensorrt_backend.cpp
  ../../common/benchmark/fps_timer.cpp
  ../../common/backends/autospeed/tensorrt_engine.cpp
  ../../common/backends/autospeed/tensorrt_engine.hpp
)
if(MEMRYX_FOUND)
  list(APPEND INFERENCE_BACKEND_SOURCES ../../common/backends/memryx_backend.cpp)
endif()
if(DEEPX_FOUND)
  list(APPEND INFERENCE_BACKEND_SOURCES ../../common/backends/deepx_backend.cpp)
endif()
if(HAILO_FOUND)
  list(APPEND INFERENCE_BACKEND_SOURCES ../../common/backends/hailo_backend.cpp)
endif()
if(AXELERA_FOUND)
  list(APPEND INFERENCE_BACKEND_SOURCES ../../common/backends/axelera_backend.cpp)
endif()

add_library(inference_backends SHARED ${INFERENCE_BACKEND_SOURCES})
target_include_directories(inference_backends PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../../common/include>
  $<INSTALL_INTERFACE:include>
  ${OpenCV_INCLUDE_DIRS}
  "${ONNXRUNTIME_ROOTDIR}/include"
  ${CUDA_INCLUDE_DIRS}
  ${TENSORRT_INCLUDE_DIR}
)
target_link_libraries(inference_backends
  PUBLIC
    onnxruntime
    ${OpenCV_LIBS}
    rclcpp::rclcpp
    ${TENSORRT_LIBRARY}
    ${TENSORRT_ONNXPARSER_LIBRARY}
    ${CUDA_LIBRARIES}
)
if(MEMRYX_FOUND)
  target_include_directories(inference_backends PUBLIC ${MEMRYX_INCLUDE_DIR})
  target_link_libraries(inference_backends PUBLIC ${MEMRYX_LIBRARY})
  target_compile_definitions(inference_backends PUBLIC MEMRYX_FOUND)
endif()
if(DEEPX_FOUND)
  target_include_directories(inference_backends PUBLIC ${DEEPX_INCLUDE_DIR})
  target_link_libraries(inference_backends PUBLIC ${DEEPX_LIBRARY})
  target_compile_definitions(inference_backends PUBLIC DEEPX_FOUND)
endif()
if(HAILO_FOUND)
  target_include_directories(inference_backends PUBLIC ${HAILORT_INCLUDE_DIR})
  target_link_libraries(inference_backends PUBLIC ${HAILORT_LIBRARY})
  target_compile_definitions(inference_backends PUBLIC HAILO_FOUND)
endif()
if(AXELERA_FOUND)
  target_include_directories(inference_backends PUBLIC ${AXELERA_INCLUDE_DIR})
  target_link_libraries(inference_backends PUBLIC ${AXELERA_LIBRARY})
  target_compile_definitions(inference_backends PUBLIC AXELERA_FOUND)
endif()

# --- ROS2 Model Node (SceneSeg/Scene3D/DomainSeg) ---
add_library(run_model_node SHARED 
  src/run_model_node.cpp
)

# Add HIP source files if HIP is available
if(HIP_FOUND)
  target_sources(run_model_node PRIVATE
    ../../common/visualizers/masks_viz.hip.cpp
  )
  target_include_directories(run_model_node PUBLIC
    ${HIP_INCLUDE_DIRS}
  )
  target_link_libraries(run_model_node
    ${HIP_LIBRARIES}
  )
endif()

target_include_directories(run_model_node PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../../common/include>
  $<INSTALL_INTERFACE:include>
)
ament_target_dependencies(run_model_node
  rclcpp rclcpp_components cv_bridge sensor_msgs image_transport
)
target_link_libraries(run_model_node
  inference_backends
)

rclcpp_components_register_node(
  run_model_node
  PLUGIN "autoware_pov::vision::RunModelNode"
  EXECUTABLE models_node_exe
)

# --- ROS2 AutoSpeed Node (Object Detection) ---
# Only build if vision_msgs is available
if(vision_msgs_FOUND)
  add_library(run_autospeed_node SHARED 
    src/run_autospeed_node.cpp
  )

  target_include_directories(run_autospeed_node PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../../common/include>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../../common/backends/autospeed>
    $<INSTALL_INTERFACE:include>
  )
  ament_target_dependencies(run_autospeed_node
    rclcpp rclcpp_components cv_bridge sensor_msgs vision_msgs image_transport
  )
  target_link_libraries(run_autospeed_node
    inference_backends
  )

  rclcpp_components_register_node(
    run_autospeed_node
    PLUGIN "autoware_pov::vision::RunAutoSpeedNode"
    EXECUTABLE autospeed_node_exe
  )
  
  message(STATUS "AutoSpeed node configured successfully")
endif()

# Install targets
set(INSTALL_TARGETS inference_backends run_model_node)
if(vision_msgs_FOUND)
  list(APPEND INSTALL_TARGETS run_autospeed_node)
endif()

install(
  TARGETS ${INSTALL_TARGETS}
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
)

install(
  DIRECTORY include/
  DESTINATION include
)

install(
  DIRECTORY launch
  DESTINATION share/${PROJECT_NAME}
)

install(
  DIRECTORY config
  DESTINATION share/${PROJECT_NAME}
)

ament_package()