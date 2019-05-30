#include "sign_detection.h"
#include <cstdio>

SignDetection::SignDetection(ros::NodeHandle& node_handle)
    : node_handle_(node_handle)
{

    result_pub_ = node_handle_.advertise<std_msgs::String>("/command", 1);
    detected_image_pub_ = node_handle_.advertise<sensor_msgs::Image>("/detected_image", 1);
    depth_image_sub_ = node_handle_.subscribe("/kinect2/sd/image_depth_rect", 1, &SignDetection::depthImageCb, this);
    image_sub_ = node_handle_.subscribe("/kinect2/hd/image_color", 1, &SignDetection::imageCb, this);
}
void SignDetection::depthImageCb(const sensor_msgs::Image::ConstPtr& msg)
{
    try {
        cv_depth_ptr_ = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::TYPE_16UC1);
    } catch (cv_bridge::Exception& e) {
        ROS_ERROR("cv_bridge exception:%s", e.what());
        return;
    }
}

float SignDetection::CaculateDepth(int center_x, int center_y)
{
    float depth = cv_depth_ptr_->image.at<float>(cv::Point(center_x, center_y));
    return depth;
}
void SignDetection::imageCb(const sensor_msgs::Image::ConstPtr& msg)
{

    cv::Mat cvframe(msg->height,
        msg->width,
        encoding2mat_type(msg->encoding),
        const_cast<unsigned char*>(msg->data.data()),
        msg->step);

    if (msg->encoding == "rgb8") {
        cv::cvtColor(cvframe, cvframe, cv::COLOR_RGB2BGR);
    }

    detect_image(cvframe, modelWeights_, modelConfiguration_, classesFile_, msg->header);
}
int SignDetection::encoding2mat_type(const std::string& encoding)
{
    if (encoding == "mono8") {
        return CV_8UC1;
    } else if (encoding == "bgr8") {
        return CV_8UC3;
    } else if (encoding == "mono16") {
        return CV_16SC1;
    } else if (encoding == "rgba8") {
        return CV_8UC4;
    } else if (encoding == "bgra8") {
        return CV_8UC4;
    } else if (encoding == "32FC1") {
        return CV_32FC1;
    } else if (encoding == "rgb8") {
        return CV_8UC3;
    } else {
        throw std::runtime_error("Unsupported encoding type");
    }
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
    static const string kWinName = "Deep learning object detection in OpenCV";
    namedWindow(kWinName, WINDOW_NORMAL);

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
                float depth = CaculateDepth(centerX, centerY);
                if (depth <= 4.0) {
                    int width = (int)(data[2] * frame.cols);
                    int height = (int)(data[3] * frame.rows);
                    int left = centerX - width / 2;
                    int top = centerY - height / 2;
                    //add direction caculate finktion
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
    vector<int> indices;
    float last_dep = 1000.0;
    int classId_target = 3;
    std_msgs::String str_msg;
    NMSBoxes(boxes, confidences, confThreshold_, nmsThreshold_, indices);
    for (size_t i = 0; i < indices.size(); ++i) {
        int idx = indices[i];
        Rect box = boxes[idx];
        float dep = depth_vec[idx];
        drawPred(classIds[idx], confidences[idx], box.x, box.y, box.x + box.width, box.y + box.height, frame, dep);
        if (dep < last_dep) {
            classId_target = classIds[idx];
        }
        last_dep = dep;
    }
    str_msg.data = classes_[classId_target];
    result_pub_.publish(str_msg);
}

void SignDetection::drawPred(int classId, float conf, int left, int top, int right, int bottom, Mat& frame, float distance)
{

    // Draw a rectangle displaying the bounding box

    rectangle(frame, Point(left, top), Point(right, bottom), Scalar(0, 255, 0), 3);

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

    putText(frame, label, Point(left, top - 10), FONT_HERSHEY_SIMPLEX, 1.2, Scalar(0, 255, 0), 2);
    putText(frame, label_depth, Point(left, top - 40), FONT_HERSHEY_SIMPLEX, 1.0, Scalar(255, 178, 0), 2);
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
