# parnav extensions: PARS measurement factors and preintegration helpers.
# Built as an OBJECT library so downstream targets can link the object files
# directly without producing a standalone library artifact.

add_library(parnav_extensions OBJECT
    RangeFactor.cpp
    AzimuthFactor.cpp
    ElevationFactor.cpp
    BaroFactor.cpp
    BaroFactorKnownPoint.cpp
    CompassFactor.cpp
    GNSSAttFactor.cpp
    PreintegrationHelpers.cpp
    MultiModalSimLoader.cpp
    SelfTrust.cpp
    MarkerAssociation.cpp
    MarkerAzimuthFactor.cpp
)

target_include_directories(parnav_extensions PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(parnav_extensions PUBLIC gtsam)

# Optional dependencies for the multimodal-sim 3D loader. Required only by
# downstream targets that use MultiModalSimLoader; left as PUBLIC so program
# executables linking parnav_extensions inherit the include / link flags.
find_package(cnpy QUIET)
if(cnpy_FOUND)
    target_link_libraries(parnav_extensions PUBLIC cnpy)
else()
    # Header-+lib install without a CMake config: fall back to find_path/find_library.
    find_path(CNPY_INCLUDE_DIR cnpy.h)
    find_library(CNPY_LIBRARY cnpy)
    if(CNPY_INCLUDE_DIR AND CNPY_LIBRARY)
        target_include_directories(parnav_extensions PUBLIC ${CNPY_INCLUDE_DIR})
        target_link_libraries(parnav_extensions PUBLIC ${CNPY_LIBRARY} z)
    else()
        message(FATAL_ERROR "cnpy not found; install rogersce/cnpy or set CNPY_INCLUDE_DIR / CNPY_LIBRARY")
    endif()
endif()

find_package(nlohmann_json REQUIRED)
target_link_libraries(parnav_extensions PUBLIC nlohmann_json::nlohmann_json)

gtsam_apply_build_flags(parnav_extensions)
