#include "check_stop.h"
#include <cstdio>

CheckStop::CheckStop(ros::NodeHandle& node_handle)
    : node_handle_(node_handle)
    , last_y(0.0)
{

    result_pub_ = node_handle_.advertise<std_msgs::String>("/go_stop", 1);
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
}

void CheckStop::Callback(const pass_detector::detection::ConstPtr& msg)
{
    //static const std::string OPENCV_WINDOW = "Image window";
    //cv::namedWindow(OPENCV_WINDOW);
    //cv::imshow(OPENCV_WINDOW, image_gray_);
    //cv::waitKey(3);
    float dis = msg->dis;
    pass_x = msg->x;
    pass_y = msg->y;
    pass_z = msg->z;
    float dis_c;
    if (dis != 0.0) {

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
