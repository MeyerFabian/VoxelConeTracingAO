cmake_minimum_required(VERSION 2.8)

# Module path
set(CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake CACHE PATH "Project specific path. Set manually if it was not found.")

IF(WIN32)
	set(CMAKE_USER_MAKE_RULES_OVERRIDE ${CMAKE_MODULE_PATH}/c_flag_overrides.cmake)
	set(CMAKE_USER_MAKE_RULES_OVERRIDE_CXX ${CMAKE_MODULE_PATH}/cxx_flag_overrides.cmake)
ENDIF(WIN32)

# Setup
project(RealtimeRendering)
set(APPNAME VoxelConeTracing)

# Activate C++11 in GCC
IF(CMAKE_COMPILER_IS_GNUCC)
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
ENDIF(CMAKE_COMPILER_IS_GNUCC)

# Make sure we dont propagate C++11 to cuda (problems with ubuntu (maybe because of old cuda version))
set(CUDA_PROPAGATE_HOST_FLAGS OFF)

# Fuck ubuntu
if("${CMAKE_SYSTEM}" MATCHES "Linux")
    set (CMAKE_CXX_LINK_EXECUTABLE "${CMAKE_CXX_LINK_EXECUTABLE} -ldl")
endif()
include_directories("${PROJECT_SOURCE_DIR}")
include_directories("${PROJECT_SOURCE_DIR}/src")

# Collect files
file(GLOB SOURCES
    "src/*.cpp"
    "src/*.c")

file(GLOB HEADERS
    "src/*.h"
    "src/*.hpp")

# Used to find shaders within c++ code
set(SHADERS_PATH ${CMAKE_SOURCE_DIR}/shaders CACHE PATH "Project specific path. Set manually if it was not found.")
add_definitions(-DSHADERS_PATH="${SHADERS_PATH}")

# Used to find meshes within c++ code
set(MESHES_PATH ${CMAKE_SOURCE_DIR}/meshes CACHE PATH "Project specific path. Set manually if it was not found.")
add_definitions(-DMESHES_PATH="${MESHES_PATH}")

# Used to find textures within c++ code
set(TEXTURES_PATH ${CMAKE_SOURCE_DIR}/textures CACHE PATH "Project specific path. Set manually if it was not found.")
add_definitions(-DTEXTURES_PATH="${TEXTURES_PATH}")


add_subdirectory("${PROJECT_SOURCE_DIR}/shaders")

add_subdirectory("${PROJECT_SOURCE_DIR}/src/SparseOctree")
add_subdirectory("${PROJECT_SOURCE_DIR}/src/Utilities")
add_subdirectory("${PROJECT_SOURCE_DIR}/src/Scene")
add_subdirectory("${PROJECT_SOURCE_DIR}/src/Rendering")
add_subdirectory("${PROJECT_SOURCE_DIR}/src/Voxelization")
add_subdirectory("${PROJECT_SOURCE_DIR}/src/PointCloud")
add_subdirectory("${PROJECT_SOURCE_DIR}/src/OctreeRaycaster")
add_subdirectory("${PROJECT_SOURCE_DIR}/src/VoxelCubes")

# Directory of external code
set(EXTERNAL_DIR "${CMAKE_CURRENT_SOURCE_DIR}/externals")

# Cuda kernels
set(CUDAKERNEL_PATH "${CMAKE_SOURCE_DIR}/kernels")
add_subdirectory("${CUDAKERNEL_PATH}")

# Add GLM
file(GLOB_RECURSE GLM
    "${EXTERNAL_DIR}/GLM/glm/*.hpp")
include_directories("${EXTERNAL_DIR}/GLM")

# Add gl3w
file(GLOB_RECURSE GL3W
    "${EXTERNAL_DIR}/gl3w/src/*")
include_directories("${EXTERNAL_DIR}/gl3w/include")

# Add stb
file(GLOB_RECURSE STB
    "${EXTERNAL_DIR}/stb/*")

# Add ImGui
file(GLOB_RECURSE IM_GUI
    "${EXTERNAL_DIR}/ImGui/*.h"
    "${EXTERNAL_DIR}/ImGui/*.cpp")
include_directories("${EXTERNAL_DIR}/ImGui")

# Collect all code
set(ALL_CODE
	${SOURCES}
	${HEADERS}
	${GLM}
	${GL3W}
	${STB}
	${IM_GUI})

# Filtering for Visual Studio
IF(MSVC)

	# http://stackoverflow.com/questions/9701387/cmake-source-group-multiple-files
	foreach(f ${ALL_CODE})
		# Get the path of the file relative to ${CMAKE_CURRENT_SOURCE_DIR},
		# then alter it (not compulsory)
		file(RELATIVE_PATH SRCGR "${CMAKE_CURRENT_SOURCE_DIR}" ${f})
		set(SRCGR "${APPNAME}/${SRCGR}")

		# Extract the folder, ie remove the filename part
		string(REGEX REPLACE "(.*)(/[^/]*)$" "\\1" SRCGR ${SRCGR})

		# Source_group expects \\ (double antislash), not / (slash)
		string(REPLACE / \\ SRCGR ${SRCGR})
		source_group("${SRCGR}" FILES ${f})
	endforeach()

ENDIF(MSVC)

# Find and add other libraries
find_package(GLFW3 REQUIRED)
find_package(OpenGL REQUIRED)
find_package(CUDA REQUIRED)
find_package(ASSIMP REQUIRED)

include_directories(${GLFW3_INCLUDE_PATH}
					${OPENGL_INCLUDE_DIR}
					${CUDA_TOOLKIT_INCLUDE}
					${ASSIMP_INCLUDE_DIRS})


# Creation of executeable
cuda_add_executable(${APPNAME} ${ALL_CODE} ${CUDAHEADERS} ${CUDAKERNELS})

# Linking
target_link_libraries(${APPNAME} ${OPENGL_LIBRARIES} ${GLFW3_LIBRARY} ${ASSIMP_LIBRARIES} ${CUDA_CUDA_LIBRARY} ${CUDA_CUDART_LIBRARY})

IF(WIN32)
     SET(CMAKE_EXE_LINKER_FLAGS /NODEFAULTLIB:\")
     set(CMAKE_CONFIGURATION_TYPES Release RelWithDebInfo CACHE TYPE  
INTERNAL FORCE )
    # Copy dlls
    add_custom_command(TARGET ${APPNAME} POST_BUILD
        COMMAND ${CMAKE_COMMAND}
        -E copy_if_different
        "${ASSIMP_DLL_RELEASE}/assimp.dll"
        "${PROJECT_BINARY_DIR}/$<CONFIGURATION>/assimp.dll"
	)

     add_custom_command(TARGET ${APPNAME} POST_BUILD
        COMMAND ${CMAKE_COMMAND}
        -E copy_if_different
        "${GLFW_DLL}/glfw3.dll"
        "${PROJECT_BINARY_DIR}/$<CONFIGURATION>/glfw3.dll"
        )

ENDIF(WIN32)
