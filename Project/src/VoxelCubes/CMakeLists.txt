file(GLOB FOLDER_SOURCES
        "*.cpp"
        "*.c")

file(GLOB FOLDER_HEADERS
        "*.h"
        "*.hpp")

set(SOURCES
        ${SOURCES}
        ${FOLDER_SOURCES}
        PARENT_SCOPE
)

set(HEADERS
        ${HEADERS}
        ${FOLDER_HEADERS}
        PARENT_SCOPE
)