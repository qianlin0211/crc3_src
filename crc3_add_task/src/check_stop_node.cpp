#include "check_stop.h"
#include <cstdio>

CheckStop::CheckStop(ros::NodeHandle& node_handle)
    : node_handle_(node_handle)
    , last_y(0.0)
{

    result_pub_ = node_handle_.advertise<std_msgs::String>("/go_stop", 1);
    info_sub_ = node_handle_.subscribe("/kinect2/qhd/camera_info", 1, &CheckStop::infoCb, this);
    pos_sub_ = node_handle_.subscribe("/position", 1, &CheckStop::Callback, this);
}
void CheckStop::infoCb(const sensor_msgs::CameraInfo::ConstPtr& msg)
{

    depth_camera_model_.fromCameraInfo(msg);
}

void CheckStop::Callback(const pass_detector::detection::ConstPtr& msg)
{
    //static const std::string OPENCV_WINDOW = "Image window";
    //cv::namedWindow(OPENCV_WINDOW);
    //cv::imshow(OPENCV_WINDOW, image_gray_);
    //cv::waitKey(3);
    float dis = msg->dis;
    int cx = msg->cx;
    int cy = msg->cy;
    if (dis != 0.0) {
        cv::Point2d pt_cv(cy, cx);
        cv::Point3d xyz = depth_camera_model_.projectPixelTo3dRay(pt_cv);
        xyz *= (dis / xyz.z);
        tf::Transform transform;
        transform.setOrigin(tf::Vector3(xyz.x, xyz.y, xyz.z));
        tf::Quaternion q;
        q.setRPY(0, 0, 0);
        transform.setRotation(q);
        br_.sendTransform(tf::StampedTransform(transform, ros::Time::now(), depth_camera_model_.tfFrame().c_str(), "/passenger_frame"));
        //        lt_.lookupTransform("/stargazer", "/passenger_frame", ros::Time(0), lt_transform_);
        //        pass_x = lt_transform_.getOrigin().x();
        //        pass_y = lt_transform_.getOrigin().y();
        //        if (last_y == 0.0) {
        //            last_y = pass_y;
        //        }
        //        float move = pass_y - last_y;
        //        last_y = pass_y;
        //        if (move > 0.2 && pass_y > 0 && pass_y < 0.5 && dis < 1.5) {
        //            str_msg.data = "stop";
        //        } else {
        //            str_msg.data = "go";
        //        }
        //    } else {
        //        str_msg.data = "go";
    }
    //    result_pub_.publish(str_msg);
}
int main(int argc, char** argv)
{
    ros::init(argc, argv, "check_stop_node");
    ros::NodeHandle nh("~");

    CheckStop node(nh);
    ros::spin();
    //cv::waitKey(0);
    return 0;
}
