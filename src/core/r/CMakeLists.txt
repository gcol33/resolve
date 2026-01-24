# R bindings using Rcpp
# This is primarily for reference - R packages are built with R CMD build

find_package(R QUIET)

if(R_FOUND)
    message(STATUS "R found: ${R_COMMAND}")

    # Create R registration file
    add_custom_target(r_package
        COMMAND R CMD INSTALL .
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        DEPENDS resolve_core
        COMMENT "Building R package"
    )
else()
    message(STATUS "R not found - R bindings will not be built")
endif()
