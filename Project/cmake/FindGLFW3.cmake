#
# Try to find GLFW3 library and include path.
# Once done this will define
#
# GLFW3_FOUND
# GLFW3_INCLUDE_PATH
# GLFW3_LIBRARY
#

SET(GLFW3_SEARCH_PATHS
	$ENV{GLFW3_ROOT}
	${DEPENDENCIES_ROOT}
	/usr			# APPLE
	/usr/local		# APPLE
	/opt/local		# APPLE
	$ENV{PROGRAMFILES}/GLFW/
	C:/GLFW
        ${CMAKE_CURRENT_SOURCE_DIR}/externals/GLFW
)

IF (MSVC)
    FIND_PATH(GLFW_INCLUDE_DIRS
        NAMES
                GLFW/glfw3.h
        PATHS
                ${GLFW3_SEARCH_PATHS}
        PATH_SUFFIXES
                include
        DOC
                "The directory where GLFW/glfw3.h resides"
    )

    IF(MSVC11)
        set(SUFFIX "lib-vc2012")
    ELSEIF(MSVC12)
        set(SUFFIX "lib-vc2013")
    ELSE()
        set(SUFFIX "lib-vc2015")
    ENDIF()

    FIND_PATH( GLFW_DLL

        NAMES
                glfw3.dll
        PATHS
                ${GLFW3_SEARCH_PATHS}
        PATH_SUFFIXES
                ${SUFFIX}
        DOC
                "The glfw3.dll library."
	)

    FIND_LIBRARY( GLFW_LIBRARIES

        NAMES
                glfw3dll.lib
        PATHS
                ${GLFW3_SEARCH_PATHS}
        PATH_SUFFIXES
                ${SUFFIX}
        DOC
                "The glfw3dll.lib library."
    )

else()
    FIND_PATH(GLFW_INCLUDE_DIRS GLFW/glfw3.h)
    FIND_LIBRARY(GLFW_LIBRARIES
                NAMES glfw3 glfw
                PATH_SUFFIXES dynamic)
ENDIF ()


SET(GLFW3_FOUND "NO")
IF (GLFW_INCLUDE_DIRS AND GLFW_LIBRARIES AND GLFW_DLL)
	SET(GLFW3_FOUND "YES")
    message("EXTERNAL LIBRARY 'GLFW3' FOUND")
ELSE()
    message("ERROR: EXTERNAL LIBRARY 'GLFW3' NOT FOUND")
ENDIF (GLFW_INCLUDE_DIRS AND GLFW_LIBRARIES AND GLFW_DLL)
