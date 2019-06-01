#include "sign_detection.h"
#include <cstdio>

SignDetection::SignDetection(ros::NodeHandle& node_handle)
    : node_handle_(node_handle)
    , image_color_sub_(node_handle_, "/kinect2/qhd/image_color_rect", 1)
    , image_depth_sub_(node_handle_, "/kinect2/qhd/image_depth_rect", 1)
    , sync(MySyncPolicy(10), image_color_sub_, image_depth_sub_)
{

    result_pub_ = node_handle_.advertise<std_msgs::String>("/command", 1);
    detected_image_pub_ = node_handle_.advertise<sensor_msgs::Image>("/detected_image", 1);
    sync.registerCallback(boost::bind(&SignDetection::Callback, this, _1, _2));
}

int SignDetection::CaculateDirection(int c_x, int c_y, int w, int h)
{
}
float SignDetection::CaculateDepth(int c_x, int c_y, int w, int h)
{
    int mal = 0;
    int l_x = c_x - w / 4;
    int r_x = c_x + w / 4;
    int t_y = c_y - h / 4;
    int b_y = c_y + h / 4;
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
    cvtColor(cvframe, image_gray_, CV_BGR2GRAY);
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
                float depth = CaculateDepth(centerX, centerY, width, height);
                if (depth <= 10000.0) {
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

    putText(frame, label, Point(left, top - 7), FONT_HERSHEY_SIMPLEX, 0.8, Scalar(0, 255, 0), 1.5);
    putText(frame, label_depth, Point(left, top - 30), FONT_HERSHEY_SIMPLEX, 0.6, Scalar(255, 178, 0), 1.5);
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
