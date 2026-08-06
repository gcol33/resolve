# RESOLVE CLI Application

# No argument-parsing dependency: arg_parser.hpp is a header-only declarative
# flag table (issue #104) and cli_spec.hpp holds one table per subcommand. Both
# are listed so they show up in generated IDE projects.
set(CLI_SOURCES
    main.cpp
    train_cmd.cpp
    predict_cmd.cpp
    info_cmd.cpp
    arg_parser.hpp
    cli_spec.hpp
)

add_executable(resolve_cli ${CLI_SOURCES})

target_link_libraries(resolve_cli PRIVATE resolve_core ${TORCH_LIBRARIES} resolve_warnings)

target_include_directories(resolve_cli PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../include
)

# Set output name to just "resolve"
set_target_properties(resolve_cli PROPERTIES
    OUTPUT_NAME "resolve"
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)

# Install CLI
install(TARGETS resolve_cli
    RUNTIME DESTINATION bin
)
