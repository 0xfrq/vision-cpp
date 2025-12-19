#pragma once
#include <onnxruntime_cxx_api.h>
#include <opencv2/opencv.hpp>
#include <vector>
#include <string>

struct Detection {
    cv::Rect box;
    float conf;
    int class_id;
};

class YoloONNX {
public:
    YoloONNX(const std::string& model_path);
    std::vector<Detection> infer(const cv::Mat& image);

private:
    Ort::Env env;
    Ort::Session session{nullptr};
    Ort::SessionOptions session_options;

    int input_width = 416;
    int input_height = 416;
};

