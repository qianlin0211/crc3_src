
add_executable(check_stop_node src/check_stop_node.cpp)
add_dependencies(check_stop_node ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
target_link_libraries(check_stop_node
    ${catkin_LIBRARIES}
    )
