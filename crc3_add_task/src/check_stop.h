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

#include <crc3_add_task/DistanceConfig.h>
#include <dynamic_reconfigure/server.h>
using namespace std;

class CheckStop {

public:
    CheckStop(ros::NodeHandle& node_handle);

private:
    dynamic_reconfigure::Server<crc3_add_task::DistanceConfig> server;
    dynamic_reconfigure::Server<crc3_add_task::DistanceConfig>::CallbackType f;
    std_msgs::String str_msg;
    image_geometry::PinholeCameraModel depth_camera_model_;
    tf::StampedTransform lt_transform_;
    tf::TransformBroadcaster br_;
    tf::TransformListener lt_;
    float pass_x;
    float pass_y;
    float last_x;
    ros::NodeHandle node_handle_;
    ros::Publisher result_pub_;
    ros::Subscriber info_sub_;
    ros::Subscriber pos_sub_;
    float x_min;
    float x_max;
    float movement;
    float dis_stop;
    void Callback(const pass_detector::detection::ConstPtr& msg);
    void infoCb(const sensor_msgs::CameraInfo::ConstPtr& msg);
    void dynamic_callback(crc3_add_task::DistanceConfig& config, uint32_t level);
};
#endif
