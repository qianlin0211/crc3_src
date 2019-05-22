#ifndef SCREW_DETECTION_H
#define SCREW_DETECTION_H

#include <ros/package.h>
#include <ros/ros.h>
#include <sensor_msgs/Image.h>
#include <std_msgs/Bool.h>
#include <string>

#include <fstream>
#include <iostream>
#include <sstream>

#include <opencv2/dnn/dnn.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using namespace std;
using namespace cv;
using namespace dnn;

class SignDetection {

public:
    SignDetection(ros::NodeHandle& node_handle);

private:
    ros::NodeHandle node_handle_;
    ros::Publisher result_pub_;
    ros::Publisher detected_image_pub_;
    ros::Subscriber image_sub_;

    //const string pro_dir_ = "/home/wu/kal_ws/anicar3_kal3/src/crc3_src/crc3_perception/src/sign_detection";
    //const String modelConfiguration_ = pro_dir_ + "/data_file/yolov3.cfg";
    //const string classesFile_ = pro_dir_ + "/data_file/coco.names";
    //const String modelWeights_ = pro_dir_ + "/data_file/yolov3.weights";
    const string pro_dir_ = ros::package::getPath("crc3_perception");
    const String modelConfiguration_ = pro_dir_ + "/src/sign_detection/data_file/yolov3.cfg";
    const string classesFile_ = pro_dir_ + "/src/sign_detection/data_file/coco.names";
    const String modelWeights_ = pro_dir_ + "/src/sign_detection/data_file/yolov3.weights";

    const float confThreshold_ = 0.5; // Confidence threshold
    const float nmsThreshold_ = 0.4;  // Non-maximum suppression threshold
    const int inpWidth_ = 416;        // Width of network's input image
    const int inpHeight_ = 416;       // Height of network's input image
    vector<string> classes_;

    void imageCb(const sensor_msgs::Image::ConstPtr& msg);
    void drawPred(int classId, float conf, int left, int top, int right, int bottom, Mat& frame);
    vector<String> getOutputsNames(const Net& net);
    void detect_image(Mat& cvframe, string modelWeights, string modelConfiguration, string classesFile);
    void postprocess(Mat& frame, const vector<Mat>& outs);
    int encoding2mat_type(const std::string& encoding);
};
#endif
