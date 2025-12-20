#include "yolo_onnx.hpp"
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <chrono>
#include <iomanip>

using namespace std;
using namespace cv;

int main() {
    // load model yolo
    cout << "Loading YOLO model..." << endl;
    YoloONNX yolo("best.onnx");
    cout << "Model loaded successfully!" << endl;
    
    // buka kamera
    cout << "Opening camera..." << endl;
    VideoCapture cap(0);
    
    if (!cap.isOpened()) {
        cerr << "ERROR: Cannot open camera!" << endl;
        return -1;
    }
    cout << "Camera opened successfully!" << endl;
    
    // set resolusi kamera lebih rendah untuk fps lebih tinggi
    cap.set(CV_CAP_PROP_FRAME_WIDTH, 320);
    cap.set(CV_CAP_PROP_FRAME_HEIGHT, 240);
    
    double fps = 0;
    
    while (true) {
        auto frame_start = chrono::high_resolution_clock::now();
        
        Mat frame;
        cap >> frame;
        
        // jalankan inferensi deteksi
        auto detections = yolo.infer(frame);
        
        // gambar hasil deteksi
        for (auto& det : detections) {
            // clamping bounding box ke batas frame
            int x = max(0, det.box.x);
            int y = max(0, det.box.y);
            int w = min(det.box.width, frame.cols - x);
            int h = min(det.box.height, frame.rows - y);
            
            // gambar bounding box hijau (tipis untuk kecepatan)
            rectangle(frame, Rect(x, y, w, h), Scalar(0, 255, 0), 1);
            
            // gambar confidence value simple
            char conf_str[16];
            snprintf(conf_str, sizeof(conf_str), "%.0f%%", det.conf * 100);
            putText(frame, conf_str, Point(x, y - 5),
                   FONT_HERSHEY_SIMPLEX, 0.3, Scalar(0, 255, 0), 1);
        }
        // hitung fps
        auto frame_end = chrono::high_resolution_clock::now();
        chrono::duration<double> frame_duration = frame_end - frame_start;
        
        if (frame_duration.count() > 0) {
            fps = 0.9 * fps + 0.1 * (1.0 / frame_duration.count());
        }
        
        // tampilkan fps dengan format simple
        char fps_str[32];
        snprintf(fps_str, sizeof(fps_str), "FPS: %.1f", fps);
        putText(frame, fps_str, Point(10, 20), 
                FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 255, 0), 1);
        
        imshow("VISION_CPP - Object Detection", frame);
        if (waitKey(1) == 27) break;
    }
    
    cout << "Program ended." << endl;
    cap.release();
    destroyAllWindows();
    return 0;
}
