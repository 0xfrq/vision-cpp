#include "yolo_onnx.hpp"
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

int main() {
    std::cout << "Loading YOLO model..." << std::endl;
    YoloONNX yolo("best.onnx");
    std::cout << "Model loaded successfully!" << std::endl;
    
    std::cout << "Opening camera..." << std::endl;
    cv::VideoCapture cap(0);
    
    if (!cap.isOpened()) {
        std::cerr << "ERROR: Cannot open camera!" << std::endl;
        return -1;
    }
    std::cout << "Camera opened successfully!" << std::endl;
    
    cap.set(CV_CAP_PROP_FRAME_WIDTH, 320);
    cap.set(CV_CAP_PROP_FRAME_HEIGHT, 240);
    
    while (true) {
        cv::Mat frame;
        cap >> frame;
        
        if (frame.empty()) {
            std::cerr << "ERROR: Empty frame!" << std::endl;
            break;
        }
        
        std::cout << "Processing frame..." << std::endl;
        auto detections = yolo.infer(frame);
        std::cout << "Detections: " << detections.size() << std::endl;
        
        for (auto& det : detections) {
            cv::rectangle(frame, det.box, cv::Scalar(0,255,0), 2);
            cv::Point center(
                det.box.x + det.box.width / 2,
                det.box.y + det.box.height / 2
            );
            cv::circle(frame, center, 3, cv::Scalar(0,0,255), -1);
            std::cout << "Ball @ (" << center.x << "," << center.y
                      << ") area=" << det.box.area() << std::endl;
        }
        
        cv::imshow("VISION_CPP", frame);
        if (cv::waitKey(1) == 27) break;
    }
    
    std::cout << "Program ended." << std::endl;
    return 0;
}
