
add_executable(sign_detection_node src/sign_detection/sign_detection_node.cpp)
add_dependencies(sign_detection_node ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
target_link_libraries(sign_detection_node
    ${catkin_LIBRARIES} ${OpenCV_LIBRARIES}
    )
