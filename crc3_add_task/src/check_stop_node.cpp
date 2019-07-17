#include "check_stop.h"
#include <cstdio>

CheckStop::CheckStop(ros::NodeHandle& node_handle)
    : node_handle_(node_handle)
    , last_y(0.0)
{

    result_pub_ = node_handle_.advertise<std_msgs::String>("/go_stop", 1);
    info_sub_ = node_handle_.subscribe("/kinect2/qhd/camera_info", 1, &CheckStop::infoCb, this);
    pos_sub_ = node_handle_.subscribe("/position", 1, &CheckStop::Callback, this);
    f = boost::bind(&CheckStop::dynamic_callback, this, _1, _2);
    server.setCallback(f);
}
void CheckStop::dynamic_callback(crc3_add_task::DistanceConfig& config, uint32_t level)
{
    y_min = config.y_min;
    y_max = config.y_max;
    movement = config.movement;
    dis_stop = config.dis_stop;
    mal = config.mal;
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
    float dis_c;
    if (dis != 0.0) {
        dis_c = dis + float(abs(cx - 480)) / 500 * (1 / dis) * mal;
        cout << "cx: " << cx << endl;
        cv::Point2d pt_cv(cy, cx);
        cv::Point3d xyz = depth_camera_model_.projectPixelTo3dRay(pt_cv);
        xyz *= (dis_c / xyz.z);
        tf::Transform transform;
        transform.setOrigin(tf::Vector3(xyz.x, xyz.y, xyz.z));
        tf::Quaternion q;
        q.setRPY(0, 0, 0);
        transform.setRotation(q);
        br_.sendTransform(tf::StampedTransform(transform, ros::Time::now(), "/kinect2_ir_optical_frame", "/passenger_frame"));
        try {
            lt_.lookupTransform("/world", "/passenger_frame", ros::Time(0), lt_transform_);
        } catch (tf::TransformException& ex) {
            ROS_INFO("%s", ex.what());
            ros::Duration(1.0).sleep();
            ros::spinOnce();
        }
        pass_x = lt_transform_.getOrigin().x();
        pass_y = lt_transform_.getOrigin().y();
        pass_z = lt_transform_.getOrigin().z();

        if (last_y == 0.0) {
            last_y = pass_y;
        }
        float move = (last_y - pass_y);
        cout << "passenger_x:" << pass_x << ", "
             << "passenger_y:" << pass_y << " ,pass_z:" << pass_z << ", movement:" << move << ", distance:" << dis << endl;
        last_y = pass_y;
        if (move > movement && dis < dis_stop || pass_y > y_min && pass_y < y_max && dis < dis_stop) {
            str_msg.data = "stop";
        } else {
            str_msg.data = "go";
        }
    } else {
        str_msg.data = "go";
    }
    result_pub_.publish(str_msg);
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
