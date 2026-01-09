#ifndef YOLO_ONNX_HPP
#define YOLO_ONNX_HPP

#include <onnxruntime_cxx_api.h>
#include <opencv2/opencv.hpp>
#include <vector>
#include <string>
#include <iostream>

using namespace std;
using namespace Ort;

struct Detection {
    cv::Rect box;
    float conf;
    int class_id;
};

class YoloONNX {
private:
    Ort::Env env;
    Ort::Session session;
    Ort::SessionOptions session_options;
    
    int input_width = 416;   // fixed onnx model size
    int input_height = 416;
    int blob_size = 416;     // dynamic processing size
    
    void printModelInfo();

public:
    YoloONNX(const string& model_path);
    vector<Detection> infer(const cv::Mat& image);
    void setInputSize(int size);  // set dynamic blobsize
};

#endif
