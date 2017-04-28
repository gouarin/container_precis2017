file(GLOB SRC . src/*.c)
include_directories(include)

ADD_EXECUTABLE(poissonC ${SRC})
TARGET_LINK_LIBRARIES(poissonC ${PETSC_LIBRARY})
