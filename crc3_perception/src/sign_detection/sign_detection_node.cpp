#include "sign_detection.h"
#include <cstdio>

SignDetection::SignDetection(ros::NodeHandle& node_handle)
    : node_handle_(node_handle)
    , image_color_sub_(node_handle_, "/kinect2/qhd/image_color_rect", 1)
    , image_depth_sub_(node_handle_, "/kinect2/qhd/image_depth_rect", 1)
    , sync(MySyncPolicy(10), image_color_sub_, image_depth_sub_)
{

    result_pub_ = node_handle_.advertise<crc3_perception::detection>("/perception", 1);
    detected_image_pub_ = node_handle_.advertise<sensor_msgs::Image>("/detected_image", 1);
    sync.registerCallback(boost::bind(&SignDetection::Callback, this, _1, _2));
    f = boost::bind(&SignDetection::dynamic_callback, this, _1, _2);
    server.setCallback(f);
}

void SignDetection::dynamic_callback(crc3_perception::DistanceConfig& config, uint32_t level)
{
    dynamic_dis = config.distance_param;
    dynamic_depth = config.depth_param;
    dynamic_box = config.box_param;
}
float SignDetection::getAngelOfTwoVector(Point2f& pt1, Point2f& pt2, Point2f& c)
{
    float theta = atan2(pt1.x - c.x, pt1.y - c.y) - atan2(pt2.x - c.x, pt2.y - c.y);
    if (theta > CV_PI)
        theta -= 2 * CV_PI;
    if (theta < -CV_PI)
        theta += 2 * CV_PI;

    theta = theta * 180.0 / CV_PI;
    return theta;
}
int SignDetection::CaculateDirection(int c_x, int c_y, int w, int h)
{
    int mal = 0;
    int l_x = c_x - w / 2;
    int r_x = c_x + w / 2;
    int t_y = c_y - h / 2;
    int b_y = c_y + h / 2;
    int sum_white_x = 0;
    int sum_white_y = 0;

    for (int i = l_x; i < r_x; ++i) {
        for (int j = t_y; j < b_y; ++j) {
            cv::Vec3b value = image_color_.at<cv::Vec3b>(cv::Point(i, j));
            if (value[0] > 180 && value[1] > 180 && value[2] > 180) {
                sum_white_x += i;
                sum_white_y += j;
                mal += 1;
            }
        }
    }
    if (mal > 0) {
        int white_x = (int)sum_white_x / mal;
        int white_y = (int)sum_white_y / mal;
        Point2f c(c_x, c_y);
        Point2f pt1(c_x + 1, c_y);
        Point2f pt2(white_x, white_y);

        float theta = getAngelOfTwoVector(pt1, pt2, c);
        if (theta < 70.0 && theta > -90.0) {
            return 0;
        }
        if (theta < 110.0 && theta > 70.0) {
            return 2;
        }
        if (theta < -90.0 || theta > 110.0) {
            return 3;
        }
    }
    return 2;

    //std::cout << white_x - c_x << std::endl;
    //std::cout << white_y - c_y << std::endl;
}
int SignDetection::CaculateDirectionNeu(int c_x, int c_y, int w, int h, float d)
{
    int sum_x = 0;
    int sum_y = 0;
    int mal = 0;
    int sum_right = 0;
    int sum_left = 0;
    int sum_top = 0;
    int sum_bottom = 0;
    int mal_right = 0;
    int mal_left = 0;
    int mal_top = 0;
    int mal_bottom = 0;
    int l_x = c_x - w / dynamic_box;
    int r_x = c_x + w / dynamic_box;
    int t_y = c_y - h / dynamic_box;
    int b_y = c_y + h / dynamic_box;
    for (int u = l_x; u < r_x; ++u) {
        for (int v = t_y; v < b_y; ++v) {
            float depth = image_depth_.at<short int>(cv::Point(u, v)) / 1000.0;
            if (depth > d - dynamic_depth && depth < d + dynamic_depth) {
                sum_x += u;
                sum_y += v;
                mal++;
            }
        }
    }
    if (mal > 0) {
        int x = (int)sum_x / mal;
        int y = (int)sum_y / mal;

        for (int i = l_x; i < r_x; ++i) {
            for (int j = t_y; j < b_y; ++j) {
                float depthN = image_depth_.at<short int>(cv::Point(i, j)) / 1000.0;
                if (depthN > d - dynamic_depth && depthN < d + dynamic_depth) {
                    cv::Vec3b value = image_color_.at<cv::Vec3b>(cv::Point(i, j));
                    if (i > x) {
                        sum_right += (value[0] + value[1] + value[2]);
                        mal_right += 1;
                    }
                    if (i < x) {
                        sum_left += (value[0] + value[1] + value[2]);
                        mal_left += 1;
                    }
                    if (j < y) {
                        sum_top += (value[0] + value[1] + value[2]);
                        mal_top += 1;
                    }
                    if (j > y) {
                        sum_bottom += (value[0] + value[1] + value[2]);
                        mal_bottom += 1;
                    }
                }
            }
        }
        if (mal_right > 0 && mal_left > 0 && mal_top > 0 && mal_bottom > 0) {
            float right = sum_right / mal_right;
            float left = sum_left / mal_left;
            float top = sum_top / mal_top;
            float bottom = sum_bottom / mal_bottom;
            float right_left = 2 * (right - left);
            float top_bottom = 2 * (top - bottom);
            if (right_left > dynamic_dis) {
                return 1;
            } else if (right_left < -dynamic_dis) {
                return 0;
            } else {
                if (top_bottom > dynamic_dis) {
                    return 3;
                } else if (top_bottom < -dynamic_dis) {
                    return 4;
                } else
                    return 3;
            }
        } else {

            std::cout << "mal_right =0" << std::endl;
            return 4;
        }

    } else {
        std::cout << "mal =0" << std::endl;
        return 4;
    }
}
float SignDetection::CaculateDepth(int c_x, int c_y, int w, int h)
{
    int mal = 0;
    int l_x = c_x - w / 3;
    int r_x = c_x + w / 3;
    int t_y = c_y - h / 3;
    int b_y = c_y + h / 3;
    float sum_depth = 0.0;

    for (int i = l_x; i < r_x; ++i) {
        for (int j = t_y; j < b_y; ++j) {
            float depth = image_depth_.at<short int>(cv::Point(i, j)) / 1000.0;
            if (depth > 0) {
                sum_depth += depth;
                mal++;
            }
        }
    }
    if (mal > 0) {
        return (sum_depth / mal);
    }
    return 0.0;
}
void SignDetection::Callback(const sensor_msgs::Image::ConstPtr& msg, const sensor_msgs::Image::ConstPtr& image_depth_msg)
{
    cv::Mat cvframe = cv_bridge::toCvCopy(msg)->image;
    image_color_ = cvframe.clone();
    //static const std::string OPENCV_WINDOW = "Image window";
    //cv::namedWindow(OPENCV_WINDOW);
    //cv::imshow(OPENCV_WINDOW, image_gray_);
    //cv::waitKey(3);
    try {
        image_depth_ = cv_bridge::toCvCopy(image_depth_msg)->image;
    } catch (cv_bridge::Exception& e) {
        ROS_ERROR("cv_bridge exception:%s", e.what());
        return;
    }

    detect_image(cvframe, modelWeights_, modelConfiguration_, classesFile_, msg->header);
}

void SignDetection::detect_image(Mat& cvframe, string modelWeights, string modelConfiguration, string classesFile, std_msgs::Header header)
{

    // Load names of classes
    ifstream ifs(classesFile.c_str());
    string line;
    while (getline(ifs, line))
        classes_.push_back(line);

    // Load the network
    Net net = readNetFromDarknet(modelConfiguration, modelWeights);
    net.setPreferableBackend(DNN_BACKEND_OPENCV);
    net.setPreferableTarget(DNN_TARGET_OPENCL);

    // Open a video file or an image file or a camera stream.
    string str, outputFile;
    cv::Mat frame = cvframe;
    // Create a window

    // Stop the program if reached end of video
    // Create a 4D blob from a frame.
    Mat blob;
    blobFromImage(frame, blob, 1 / 255.0, cvSize(inpWidth_, inpHeight_), Scalar(0, 0, 0), true, false);

    // Sets the input to the network
    net.setInput(blob);

    // Runs the forward pass to get output of the output layers
    vector<Mat> outs;
    net.forward(outs, getOutputsNames(net));

    // Remove the bounding boxes with low confidence
    postprocess(frame, outs);
    // Put efficiency information. The function getPerfProfile returns the overall time for inference(t) and the timings
    // for each of the layers(in layersTimes)
    // vector<double> layersTimes;
    // double freq = getTickFrequency() / 1000;
    // double t = net.getPerfProfile(layersTimes) / freq;
    // string label = format("Inference time for a frame : %.2f ms", t);
    // putText(frame, label, Point(0, 15), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 0, 255));
    // Write the frame with the detection boxes
    //imshow(kWinName, frame);
    cv_bridge::CvImage out_msg;
    out_msg.header = header;
    out_msg.encoding = sensor_msgs::image_encodings::BGR8;
    out_msg.image = frame;
    // sensor_msgs::msg::Image img_msg; // >> message to be sent

    //std_msgs::msg::Header header;       // empty header
    //header.seq = counter;               // user defined counter
    //header.stamp = rclcpp::Time::now(); // time
    //out_msg = cv_bridge::CvImage(header, sensor_msgs::image_encodings::RGB8, frame);
    // img_bridge.toImageMsg(img_msg); // from cv_bridge to sensor_msgs::Image
    detected_image_pub_.publish(out_msg.toImageMsg());

    //cv::waitKey(30);
}

void SignDetection::postprocess(Mat& frame, const vector<Mat>& outs)
{

    vector<int> classIds;
    vector<float> confidences;
    vector<Rect> boxes;
    vector<float> depth_vec;

    for (size_t i = 0; i < outs.size(); ++i) {
        // Scan through all the bounding boxes output from the network and keep only the
        // ones with high confidence scores. Assign the box's class label as the class
        // with the highest score for the box.
        float* data = (float*)outs[i].data;
        for (int j = 0; j < outs[i].rows; ++j, data += outs[i].cols) {
            Mat scores = outs[i].row(j).colRange(5, outs[i].cols);
            Point classIdPoint;
            double confidence;
            // Get the value and location of the maximum score
            minMaxLoc(scores, 0, &confidence, 0, &classIdPoint);
            if (confidence > confThreshold_) {
                int centerX = (int)(data[0] * frame.cols);
                int centerY = (int)(data[1] * frame.rows);
                int width = (int)(data[2] * frame.cols);
                int height = (int)(data[3] * frame.rows);
                int left = centerX - width / 2;
                int top = centerY - height / 2;
                float depth = CaculateDepth(centerX, centerY, width, height);
                if (depth <= 3.0) {
                    if (classIdPoint.x == 0 || classIdPoint.x == 1) {
                        //add direction caculate finktion
                        int directId = CaculateDirectionNeu(centerX, centerY, width, height, depth);

                        classIds.push_back(directId);
                        confidences.push_back((float)confidence);
                        boxes.push_back(Rect(left, top, width, height));
                        depth_vec.push_back(depth);
                    }
                    classIds.push_back(classIdPoint.x);
                    confidences.push_back((float)confidence);
                    boxes.push_back(Rect(left, top, width, height));
                    depth_vec.push_back(depth);
                }
            }
        }
    }

    // Perform non maximum suppression to eliminate redundant overlapping boxes with
    // lower confidences
    detect_msg.stop_sign_found = false;
    detect_msg.dist_to_stop = -0.0;
    vector<int> indices;
    float last_dep = 1000.0;
    float last_dep_stop = 1000.0;
    string str_push;
    classId_target = 4;
    NMSBoxes(boxes, confidences, confThreshold_, nmsThreshold_, indices);
    for (size_t i = 0; i < indices.size(); ++i) {
        int idx = indices[i];
        Rect box = boxes[idx];
        float dep = depth_vec[idx];
        float dep_stop = depth_vec[idx];
        drawPred(classIds[idx], confidences[idx], box.x, box.y, box.x + box.width, box.y + box.height, frame, dep);
        if (dep_stop <= last_dep_stop && classIds[idx] == 2 && dep_stop > 0.0) {
            detect_msg.stop_sign_found = true;
            detect_msg.dist_to_stop = dep_stop;
            last_dep_stop = dep_stop;
        }
        if (dep <= last_dep && classIds[idx] != 2) {
            classId_target = classIds[idx];
            last_dep = dep;
        }
    }
    str_push = classes_[classId_target];
    detect_msg.direction = str_push;
    result_pub_.publish(detect_msg);
}

void SignDetection::drawPred(int classId, float conf, int left, int top, int right, int bottom, Mat& frame, float distance)
{

    // Draw a rectangle displaying the bounding box

    rectangle(frame, Point(left, top), Point(right, bottom), Scalar(0, 255, 0), 2);

    // Get the label for the class name and its confidence
    string label = format("%.2f", conf);
    if (!classes_.empty()) {
        CV_Assert(classId < (int)classes_.size());
        label = classes_[classId] + ":" + label;
    }

    string label_depth = format("%.2f", distance) + "m";

    // Display the label at the top of the bounding box
    //int baseLine;
    //Size labelSize = getTextSize(label, FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
    //top = max(top, labelSize.height);
    //rectangle(frame,
    //    Point(left, top - round(1.5 * labelSize.height)),
    //    Point(left + round(1.5 * labelSize.width), top + baseLine),
    //    Scalar(255, 255, 255),
    //    FILLED);

    putText(frame, label, Point(left, top - 7), FONT_HERSHEY_SIMPLEX, 1.0, Scalar(0, 255, 0), 2.0);
    putText(frame, label_depth, Point(left, top - 30), FONT_HERSHEY_SIMPLEX, 0.8, Scalar(255, 178, 0), 2.0);
}

vector<String> SignDetection::getOutputsNames(const Net& net)
{
    static vector<String> names;
    if (names.empty()) {
        // Get the indices of the output layers, i.e. the layers with unconnected outputs
        vector<int> outLayers = net.getUnconnectedOutLayers();

        // get the names of all the layers in the network
        vector<String> layersNames = net.getLayerNames();

        // Get the names of the output layers in names
        names.resize(outLayers.size());
        for (size_t i = 0; i < outLayers.size(); ++i)
            names[i] = layersNames[outLayers[i] - 1];
    }
    return names;
}
int main(int argc, char** argv)
{
    ros::init(argc, argv, "kal3_traffic_sign_detection");
    ros::NodeHandle nh("~");

    SignDetection node(nh);
    ros::spin();
    //cv::waitKey(0);
    return 0;
}
