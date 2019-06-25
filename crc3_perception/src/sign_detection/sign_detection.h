#ifndef SCREW_DETECTION_H
#define SCREW_DETECTION_H

#include <algorithm>
#include <crc3_perception/detection.h>
#include <cv_bridge/cv_bridge.h>
#include <map>
#include <ros/package.h>
#include <ros/ros.h>
#include <sensor_msgs/CameraInfo.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/image_encodings.h>
#include <std_msgs/Header.h>
#include <std_msgs/String.h>
#include <string>
#include <vector>

#include <fstream>
#include <iostream>
#include <sstream>

#include <opencv2/dnn/dnn.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/opencv.hpp>

#include <math.h>
#include <message_filters/subscriber.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <message_filters/synchronizer.h>

#include <crc3_perception/DistanceConfig.h>
#include <dynamic_reconfigure/server.h>

#include <image_geometry/pinhole_camera_model.h>
#include <tf/transform_broadcaster.h>
#include <tf/transform_listener.h>

using namespace std;
using namespace cv;
using namespace dnn;
using namespace message_filters;

class SignDetection {

public:
    SignDetection(ros::NodeHandle& node_handle);

private:
    std_msgs::String str_msg;
    dynamic_reconfigure::Server<crc3_perception::DistanceConfig> server;
    dynamic_reconfigure::Server<crc3_perception::DistanceConfig>::CallbackType f;
    float dynamic_dis;
    float dis = 0.0;
    int cx = 0;
    int cy = 0;
    image_geometry::PinholeCameraModel depth_camera_model_;
    tf::StampedTransform br_transform_;
    tf::StampedTransform lt_transform_;
    tf::TransformBroadcaster br_;
    tf::TransformListener lt_;
    float pass_x;
    float pass_y;
    float last_y = 0.0;
    vector<string> str_vec;
    ros::NodeHandle node_handle_;
    ros::Publisher result_pub_;
    ros::Publisher detected_image_pub_;
    cv::Mat image_depth_;
    message_filters::Subscriber<sensor_msgs::Image> image_color_sub_;
    message_filters::Subscriber<sensor_msgs::CameraInfo> cameraInfo_sub_;
    message_filters::Subscriber<sensor_msgs::Image> image_depth_sub_;
    typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::Image, sensor_msgs::Image> MySyncPolicy;
    message_filters::Synchronizer<MySyncPolicy> sync;
    // const string pro_dir_ = "/home/wu/kal_ws/anicar3_kal3/src/crc3_src/crc3_perception/src/sign_detection";
    // const String modelConfiguration_ = pro_dir_ + "/data_file/yolov3.cfg";
    // const string classesFile_ = pro_dir_ + "/data_file/coco.names";
    // const String modelWeights_ = pro_dir_ + "/data_file/yolov3.weights";
    const string pro_dir_
        = ros::package::getPath("crc3_perception");
    const String modelConfiguration_ = pro_dir_ + "/src/sign_detection/data_file/passenger_tiny.cfg";
    const string classesFile_ = pro_dir_ + "/src/sign_detection/data_file/passenger.names";
    const String modelWeights_ = pro_dir_ + "/src/sign_detection/data_file/passenger_tiny_final.weights";

    const float confThreshold_ = 0.5; // Confidence threshold
    const float nmsThreshold_ = 0.4;  // Non-maximum suppression threshold
    const int inpWidth_ = 416;        // Width of network's input image
    const int inpHeight_ = 416;       // Height of network's input image
    vector<string> classes_;

    void Callback(const sensor_msgs::Image::ConstPtr& image_color_msg, const sensor_msgs::Image::ConstPtr& image_depth_msg, const sensor_msgs::CameraInfo::ConstPtr& camera_info_msg);
    void drawPred(int classId, float conf, int left, int top, int right, int bottom, Mat& frame, float distance);
    vector<String> getOutputsNames(const Net& net);
    void detect_image(Mat& cvframe, string modelWeights, string modelConfiguration, string classesFile, std_msgs::Header header);
    void postprocess(Mat& frame, const vector<Mat>& outs);
    float CaculateDepth(int c_x, int c_y, int w, int h);
    string find_max(const vector<string>& invec);
    void dynamic_callback(crc3_perception::DistanceConfig& config, uint32_t level);
};
#endif
