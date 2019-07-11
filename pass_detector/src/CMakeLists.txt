
add_executable(pass_detection_node src/pass_detection_node.cpp)
add_dependencies(pass_detection_node ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
target_link_libraries(pass_detection_node
    ${catkin_LIBRARIES} ${OpenCV_LIBRARIES}
    )
