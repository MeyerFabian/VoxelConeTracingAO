cmake_minimum_required(VERSION 2.8)

project(AnimSim)
set(APPNAME VoxelConeTracing)

# Activate C++11 in GCC
IF(CMAKE_COMPILER_IS_GNUCC)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
ENDIF(CMAKE_COMPILER_IS_GNUCC)

include_directories(.)

# Collect files
file(GLOB SOURCES
    "*.cpp"
    "*.c")

file(GLOB HEADERS
    "*.h")

# Directory of external code
set(EXTERNAL_DIR ${CMAKE_CURRENT_SOURCE_DIR}/External)

# Add GLM
file(GLOB_RECURSE GLM
    "${EXTERNAL_DIR}/GLM/glm/*.hpp")
include_directories(${EXTERNAL_DIR}/GLM)

# Add OpenGLLoader
file(GLOB_RECURSE OPEN_GL_LOADER
    "${EXTERNAL_DIR}/OpenGLLoader/*")

# Add picoPNG
file(GLOB_RECURSE PICO_PNG
    "${EXTERNAL_DIR}/picoPNG/*")

# Collect all code
set(ALL_CODE
	${SOURCES}
	${HEADERS}
	${GLM}
	${OPEN_GL_LOADER}
	${PICO_PNG})

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

# Find other libraries
IF(WIN32)

	# GLFW 3
	set(GLFW_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/External/GLFW")
	include_directories("${GLFW_DIRECTORY}/include")

	IF(MSVC10)
		set(GLFW_LIBRARIES_DIRECTORY "${GLFW_DIRECTORY}/lib-vc2012")
	ELSEIF(MSVC11)
		set(GLFW_LIBRARIES_DIRECTORY "${GLFW_DIRECTORY}/lib-vc2013")
	ELSE()
		set(GLFW_LIBRARIES_DIRECTORY "${GLFW_DIRECTORY}/lib-vc2015")
	ENDIF()

	find_library(GLFW_LIBRARIES
		NAMES glfw glfw3
		HINTS "${GLFW_LIBRARIES_DIRECTORY}")

	set(GLFW_BINARIES_DIRECTORY "${GLFW_LIBRARIES_DIRECTORY}")

	# OpenGL
	# ${OPENGL_LIBRARIES}
	# ${OPENGL_INCLUDE_DIR}
	find_package(OpenGL REQUIRED)
	include_directories(${OPENGL_INCLUDE_DIR})

	# CUDA
	find_package(CUDA REQUIRED)
	include_directories(${CUDA_TOOLKIT_INCLUDE})
ELSE(WIN32)

	# GLFW 3 (needs to be installed via package manager)
	find_package(PkgConfig REQUIRED)
	pkg_search_module(GLFW REQUIRED glfw3)
	include_directories(${GLFW_INCLUDE_DIRS})

	# OpenGL
	find_package(OpenGL REQUIRED)
	include_directories(${OpenGL_INCLUDE_DIRS})
	link_directories(${OpenGL_LIBRARY_DIRS})

	# CUDA
	find_package(CUDA REQUIRED)
	include_directories(${CUDA_TOOLKIT_INCLUDE})
ENDIF(WIN32)

# Creation of executeable
add_executable(${APPNAME} ${ALL_CODE})

# Linking
target_link_libraries(${APPNAME} ${OPENGL_LIBRARIES} ${GLFW_LIBRARIES} ${CUDA_CUDA_LIBRARY} ${CUDA_CUDART_LIBRARY})

IF(WIN32)

    # Copy dlls
		add_custom_command(TARGET ${APPNAME} POST_BUILD
			COMMAND ${CMAKE_COMMAND} -E copy_if_different
			"${GLFW_BINARIES_DIRECTORY}/glfw3.dll"
			"${PROJECT_BINARY_DIR}/glfw3.dll")

ENDIF(WIN32)

# Copy content
add_custom_command(TARGET ${APPNAME} PRE_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
    ${CMAKE_SOURCE_DIR}/Content Content)

