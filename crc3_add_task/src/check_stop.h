#ifndef SCREW_DETECTION_H
#define SCREW_DETECTION_H

#include <pass_detector/detection.h>
#include <ros/package.h>
#include <ros/ros.h>
#include <sensor_msgs/CameraInfo.h>
#include <std_msgs/Header.h>
#include <std_msgs/String.h>
#include <string>
#include <vector>

#include <fstream>
#include <iostream>
#include <sstream>

#include <math.h>

#include <image_geometry/pinhole_camera_model.h>
#include <tf/transform_broadcaster.h>
#include <tf/transform_listener.h>

using namespace std;

class CheckStop {

public:
    CheckStop(ros::NodeHandle& node_handle);

private:
    std_msgs::String str_msg;
    image_geometry::PinholeCameraModel depth_camera_model_;
    tf::StampedTransform lt_transform_;
    tf::TransformBroadcaster br_;
    tf::TransformListener lt_;
    float pass_x;
    float pass_y;
    float last_y = 0.0;
    ros::NodeHandle node_handle_;
    ros::Publisher result_pub_;
    ros::Subscriber info_sub_;

    void Callback(const sensor_msgs::Image::ConstPtr& image_color_msg, const sensor_msgs::Image::ConstPtr& image_depth_msg);
    void infoCb(const sensor_msgs::CameraInfo::ConstPtr& msg);
};
#endif
