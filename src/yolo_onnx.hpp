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
    
    int input_width = 416;
    int input_height = 416;
    int blob_size = 416;
    
    // cache untuk hindari rekalkulasi saat blobsize sama
    int last_blob_size = -1;
    float cached_scale = 1.0f;
    int cached_new_w = 0;
    int cached_new_h = 0;
    int cached_pad_x = 0;
    int cached_pad_y = 0;
    
    // pre-allocated buffers
    cv::Mat padded_buffer;
    cv::Mat float_buffer;
    std::vector<float> input_tensor_values;
    
    void printModelInfo();

public:
    YoloONNX(const string& model_path);
    vector<Detection> infer(const cv::Mat& image);
    void setInputSize(int size);
};

#endif
