# Based on the FindPhysFS.cmake scipt
# - Try to find Assimp
# Once done this will define
#
#  ASSIMP_FOUND - system has Assimp
#  ASSIMP_INCLUDE_DIRS - the Assimp include directory
#  ASSIMP_LIBRARIES - Link these to use Assimp
#  ASSIMP_DLL

SET(ASSIMP_SEARCH_PATHS
		$ENV{ASSIMP_ROOT}
		${DEPENDENCIES_ROOT}
		/usr			# APPLE
		/usr/local		# APPLE
		/opt/local		# APPLE
		$ENV{PROGRAMFILES}/assimp/
		C:/assimp
		${CMAKE_CURRENT_SOURCE_DIR}/External/assimp
		$ENV{ASSIMPSDIR}
		/sw
		/opt/csw
		/opt
		${_assimp_LIB_SEARCH_DIRS_SYSTEM}
)

IF (MSVC)
        FIND_PATH(ASSIMP_INCLUDE_DIRS
                NAMES
                        assimp/ai_assert.h
                PATHS
                        ${ASSIMP_SEARCH_PATHS}
                PATH_SUFFIXES
                        include
                DOC
                        "The directory where assimp/ai_assert.h resides"
	)

	FIND_LIBRARY(ASSIMP_LIBRARY_RELEASE
		NAMES
                        assimp.lib
		PATHS
                        ${ASSIMP_SEARCH_PATHS}
		PATH_SUFFIXES
			lib/Release
		DOC
                        "The assimp.lib library."
	)
	FIND_LIBRARY(ASSIMP_LIBRARY_DEBUG
		NAMES
			assimpd.lib
		PATHS
                        ${ASSIMP_SEARCH_PATHS}
		PATH_SUFFIXES
			lib/Debug
		DOC
                        "The assimpd.lib library."
	)

	FIND_PATH( ASSIMP_DLL_DEBUG
		NAMES
			assimpd.dll
		PATHS
			${ASSIMP_SEARCH_PATHS}
		PATH_SUFFIXES
			bin/Debug
		DOC
			"The assimpd.dll library."
	)

	FIND_PATH( ASSIMP_DLL_RELEASE
		NAMES
			assimp.dll
		PATHS
			${ASSIMP_SEARCH_PATHS}
		PATH_SUFFIXES
			bin/Release
		DOC
			"The assimp.dll library."
	)
ENDIF ()


SET(ASSIMP_FOUND "NO")
IF (ASSIMP_INCLUDE_DIRS AND ASSIMP_LIBRARY_DEBUG AND ASSIMP_LIBRARY_RELEASE AND ASSIMP_DLL_RELEASE AND ASSIMP_DLL_DEBUG)
        SET(ASSIMP_FOUND "YES")
        message("EXTERNAL LIBRARY 'ASSIMP' FOUND")
ELSE()
        message("ERROR: EXTERNAL LIBRARY 'ASSIMP' NOT FOUND")
ENDIF (ASSIMP_INCLUDE_DIRS AND ASSIMP_LIBRARY_DEBUG AND ASSIMP_LIBRARY_RELEASE AND ASSIMP_DLL_RELEASE AND ASSIMP_DLL_DEBUG)

SET(ASSIMP_FOUND "NO")
IF(ASSIMP_INCLUDE_DIRS AND ASSIMP_LIBRARY_DEBUG AND ASSIMP_LIBRARY_RELEASE AND ASSIMP_DLL_RELEASE AND ASSIMP_DLL_DEBUG)
	SET(ASSIMP_FOUND "YES")
	SET(ASSIMP_LIBRARY debug ${ASSIMP_LIBRARY_DEBUG} optimized ${ASSIMP_LIBRARY_RELEASE})
ENDIF(ASSIMP_INCLUDE_DIRS AND ASSIMP_LIBRARY_DEBUG AND ASSIMP_LIBRARY_RELEASE AND ASSIMP_DLL_RELEASE AND ASSIMP_DLL_DEBUG)

if(ASSIMP_DEBUG)
	message(STATUS "assimp inc: ${ASSIMP_INCLUDE_DIRS}")
	message(STATUS "assimp lib: ${ASSIMP_LIBRARY}")
ENDIF(ASSIMP_DEBUG)

if(AssImp_FIND_REQUIRED AND NOT (ASSIMP_LIBRARY AND ASSIMP_INCLUDE_DIRS))
	message(FATAL_ERROR "Could not find assimp")
ENDIF(AssImp_FIND_REQUIRED AND NOT (ASSIMP_LIBRARY AND ASSIMP_INCLUDE_DIRS))

mark_as_advanced(ASSIMP_LIBRARY_DEBUG ASSIMP_LIBRARY_RELEASE ASSIMP_INCLUDE_DIRS)
