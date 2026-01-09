// deteksi bola yolov5 + hsv tracking untuk robot sepak bola
#include <ros/ros.h>
#include <ros/package.h>
#include <v2_detection/BallState.h>
#include <v2_detection/BallCoordinate.h>
#include <v2_detection/Ballarea.h>
#include <opencv2/opencv.hpp>
#include <thread>
#include <mutex>
#include <algorithm>
#include "yolo_onnx.hpp"

using namespace cv;
using namespace std;

// fungsi utilitas
template<typename T>
inline T clamp(T v, T lo, T hi) { return (v < lo) ? lo : (v > hi) ? hi : v; }

double map_value(double v, double smin, double smax, double tmin, double tmax) {
    v = clamp(v, min(smin, smax), max(smin, smax));
    return tmin + (v - smin) * (tmax - tmin) / (smax - smin);
}

// status deteksi
enum DetectState { NOTFOUND = 0, FOUND = 1 };
DetectState detect_status = NOTFOUND;

// hsv range untuk tracking
int min_h=0, min_s=0, min_v=0;
int max_h=0, max_s=0, max_v=0;

// data bola
int x_ball=0, y_ball=0, w_ball=0, h_ball=0;
int ball_area = 0;
Point2f center_ball(0,0);
vector<int> scan_area(2,0);
Rect last_box;

// timing dan fps
double waktu_sebelum = 0.0;
int frame_counter = 0;
double fps_start_time = 0.0;
double fps = 0.0;

// ukuran frame dan blobsize dinamis
const int framesize[2] = {320, 240};
int blobsize = 416;

// hitung fps setiap 0.2 detik
void calculate_fps() {
    frame_counter++;
    double current_time = ros::Time::now().toSec();
    double elapsed = current_time - fps_start_time;
    if(elapsed >= 0.2) {
        fps = frame_counter / elapsed;
        frame_counter = 0;
        fps_start_time = current_time;
    }
}

// ambil nilai hsv dari titik-titik di dalam bounding box bola
void get_hsv_val(const Mat& img) {
    int x1 = x_ball, y1 = y_ball;
    int x2 = x_ball + w_ball, y2 = y_ball + h_ball;
    
    // titik-titik sampling seperti python
    Point dot_tengah((x1+x2)/2, (y1+y2)/2);
    Point dot_sepertiga_kanan((x1+dot_tengah.x)/2, (y1+dot_tengah.y)/2);
    Point dot_duapertiga_kanan((x2+dot_tengah.x)/2, (y2+dot_tengah.y)/2);
    Point dot_sepertiga_kiri((dot_tengah.x+(x1+(x2-x1)))/2, (dot_tengah.y+y1)/2);
    Point dot_duapertiga_kiri((x1+dot_tengah.x)/2, (y1+(y2-y1)+dot_tengah.y)/2);
    Point dot_bawah_tengah(dot_tengah.x, (dot_tengah.y+y2)/2);
    
    vector<Point> dots = {dot_tengah, dot_sepertiga_kanan, dot_duapertiga_kanan, 
                          dot_sepertiga_kiri, dot_duapertiga_kiri, dot_bawah_tengah};
    
    vector<int> H, S, V;
    
    for(auto& p : dots) {
        int px = clamp(p.x, 0, img.cols-1);
        int py = clamp(p.y, 0, img.rows-1);
        Vec3b bgr = img.at<Vec3b>(py, px);
        Mat hsv;
        cvtColor(Mat(1,1,CV_8UC3,Scalar(bgr[0],bgr[1],bgr[2])), hsv, COLOR_BGR2HSV);
        Vec3b hv = hsv.at<Vec3b>(0,0);
        H.push_back(hv[0]); S.push_back(hv[1]); V.push_back(hv[2]);
    }
    
    min_h = *min_element(H.begin(), H.end());
    max_h = min(*max_element(H.begin(), H.end()), 33);
    min_s = max(*min_element(S.begin(), S.end()), 160);
    max_s = *max_element(S.begin(), S.end());
    min_v = *min_element(V.begin(), V.end());
    max_v = *max_element(V.begin(), V.end());
    
    ROS_INFO("hsv: h[%d,%d] s[%d,%d] v[%d,%d]", min_h, max_h, min_s, max_s, min_v, max_v);
}

// threaded video capture seperti python webcamvideostream
class ThreadedCapture {
private:
    VideoCapture cap;
    Mat frame;
    bool stopped;
    mutex frameMutex;
    thread captureThread;
    
    void update() {
        while(!stopped) {
            Mat temp;
            cap >> temp;
            if(!temp.empty()) {
                lock_guard<mutex> lock(frameMutex);
                frame = temp.clone();
            }
        }
    }
    
public:
    ThreadedCapture(int src) : stopped(false) {
        cap.open(src);
        cap.set(CAP_PROP_FPS, 60);
        cap.set(CAP_PROP_FRAME_WIDTH, framesize[0]);
        cap.set(CAP_PROP_FRAME_HEIGHT, framesize[1]);
        if(!cap.isOpened()) {
            ROS_ERROR("kamera gagal dibuka");
            return;
        }
        cap >> frame;
        captureThread = thread(&ThreadedCapture::update, this);
    }
    
    Mat read() {
        lock_guard<mutex> lock(frameMutex);
        return frame.clone();
    }
    
    void stop() {
        stopped = true;
        if(captureThread.joinable()) captureThread.join();
        cap.release();
    }
    
    ~ThreadedCapture() { stop(); }
};

// ekstrak area lapangan hijau
Mat extractField(const Mat& img) {
    Mat hsv, mask, result;
    cvtColor(img, hsv, COLOR_BGR2HSV);
    
    // range hsv lapangan hijau
    inRange(hsv, Scalar(35, 40, 40), Scalar(85, 255, 255), mask);
    
    Mat kernel = getStructuringElement(MORPH_RECT, Size(5,5));
    erode(mask, mask, kernel, Point(-1,-1), 2);
    dilate(mask, mask, kernel, Point(-1,-1), 5);
    
    vector<vector<Point>> contours;
    findContours(mask, contours, RETR_TREE, CHAIN_APPROX_NONE);
    
    Mat fieldMask = Mat::zeros(img.size(), CV_8UC1);
    if(!contours.empty()) {
        auto maxContour = *max_element(contours.begin(), contours.end(),
            [](const vector<Point>&a, const vector<Point>&b){ return contourArea(a) < contourArea(b); });
        vector<Point> hull;
        convexHull(maxContour, hull);
        fillConvexPoly(fieldMask, hull, Scalar(255));
    }
    
    bitwise_and(img, img, result, fieldMask);
    return result;
}

int main(int argc, char** argv) {
    ros::init(argc, argv, "vision_yolo_cpp");
    ros::NodeHandle nh;

    // publisher ros
    auto pub_state = nh.advertise<v2_detection::BallState>("/DEWO/image_processing/deteksi_bola/ball_state", 10);
    auto pub_coord = nh.advertise<v2_detection::BallCoordinate>("/DEWO/image_processing/deteksi_bola/coordinate", 10);
    auto pub_area = nh.advertise<v2_detection::Ballarea>("/DEWO/image_processing/deteksi_bola/ball_area", 10);

    // load model yolo
    string pkg = ros::package::getPath("vision_cpp");
    YoloONNX yolo(pkg + "/src/best.onnx");

    // inisialisasi kamera threaded
    ROS_INFO("memulai kamera...");
    ThreadedCapture capture(0);
    
    fps_start_time = ros::Time::now().toSec();
    waktu_sebelum = ros::Time::now().toSec();
    
    ROS_INFO("sistem siap");

    while(ros::ok()) {
        Mat img = capture.read();
        if(img.empty()) { ros::spinOnce(); continue; }
        
        Mat img_result = img.clone();

        // mode tracking hsv
        if(detect_status == FOUND) {
            // ekstrak lapangan
            Mat field_img = extractField(img);
            
            // konversi ke hsv dan buat mask bola
            Mat hsv, binary_ball;
            cvtColor(field_img, hsv, COLOR_BGR2HSV);
            inRange(hsv, Scalar(min_h, min_s, min_v), Scalar(max_h, max_s, max_v), binary_ball);
            
            // morphology untuk bersihkan noise
            Mat kernel = Mat::ones(5, 5, CV_8U);
            morphologyEx(binary_ball, binary_ball, MORPH_CLOSE, kernel);
            morphologyEx(binary_ball, binary_ball, MORPH_OPEN, kernel);
            
            // cari kontur bola
            vector<vector<Point>> contours;
            findContours(binary_ball, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
            
            bool objek_ditemukan = false;
            
            for(auto& contour : contours) {
                double area = contourArea(contour);
                
                // filter area seperti python: area > ball_area/5 dan area < ball_area*1.1
                if(area > ball_area/5 && area < ball_area*1.1 && contours.size() > 0) {
                    Rect r = boundingRect(contour);
                    float x_center = r.x + r.width/2.0f;
                    
                    // cek apakah dalam scan area dan area cukup besar
                    if(scan_area[0] <= x_center && x_center <= scan_area[1] && (r.width*r.height) > 3000) {
                        // update posisi bola
                        int area_bola = r.width * r.height;
                        float x_center_rect = r.x + r.width/2.0f;
                        float y_center_rect = r.y + r.height/2.0f;
                        center_ball = Point2f(x_center_rect, y_center_rect);
                        last_box = r;
                        
                        // gambar bounding box kuning
                        rectangle(img_result, r, Scalar(0, 255, 255), 2);
                        
                        // kirim data ros
                        v2_detection::BallCoordinate bc;
                        bc.pos_x = clamp(x_center_rect/framesize[0]*2-1, -1.0f, 1.0f);
                        bc.pos_y = clamp(y_center_rect/framesize[1]*2-1, -1.0f, 1.0f);
                        bc.obj_size = area_bola;
                        pub_coord.publish(bc);
                        
                        v2_detection::Ballarea ba;
                        ba.ballarea = area_bola;
                        pub_area.publish(ba);
                        
                        v2_detection::BallState bs;
                        bs.ball_status = "FOUND";
                        pub_state.publish(bs);
                        
                        objek_ditemukan = true;
                        
                        // cek waktu untuk reset ke yolo
                        double waktu_detect = map_value(area_bola, 0, 76800, 0.5, 80);
                        double waktu_sesudah = ros::Time::now().toSec();
                        double delta = waktu_sesudah - waktu_sebelum;
                        
                        ROS_INFO("ball area result: %d", area_bola);
                        
                        if(delta >= waktu_detect) {
                            waktu_sebelum = ros::Time::now().toSec();
                            detect_status = NOTFOUND;
                        }
                        break;
                    }
                    break;
                }
            }
            
            // jika tidak ditemukan, kembali ke yolo
            if(!objek_ditemukan) {
                detect_status = NOTFOUND;
            }
        }

        // mode pencarian yolo
        if(detect_status == NOTFOUND) {
            // set blobsize dinamis seperti python
            yolo.setInputSize(blobsize);
            
            auto dets = yolo.infer(img);
            
            if(dets.empty()) {
                // tidak ada deteksi
                ROS_INFO("yolo searching");
                v2_detection::BallState bs;
                bs.ball_status = "NOTFOUND";
                pub_state.publish(bs);
                
                // reset blobsize ke 416 untuk search
                blobsize = 416;
                ROS_INFO("blobsize: %d", blobsize);
            } else {
                // proses deteksi
                for(auto& d : dets) {
                    if(d.class_id != 0 || d.conf < 0.5f) continue;
                    
                    Rect b = d.box;
                    x_ball = b.x;
                    y_ball = b.y;
                    w_ball = b.width;
                    h_ball = b.height;
                    
                    float x_center_ball = x_ball + w_ball/2.0f;
                    float y_center_ball = y_ball + h_ball/2.0f;
                    center_ball = Point2f(x_center_ball, y_center_ball);
                    int obj_size_ball = w_ball * h_ball;
                    ball_area = obj_size_ball;
                    last_box = b;
                    
                    // hitung scan area seperti python
                    int in_area_ball;
                    if(ball_area <= 5000) {
                        in_area_ball = w_ball * 3;
                    } else {
                        in_area_ball = w_ball + 35;
                    }
                    scan_area = {int(x_center_ball - in_area_ball), int(x_center_ball + in_area_ball)};
                    
                    // gambar bounding box biru dengan label
                    rectangle(img_result, b, Scalar(255, 0, 0), 1);
                    char text[50];
                    snprintf(text, sizeof(text), "bola: %.2f", d.conf);
                    putText(img_result, text, Point(b.x, b.y-5), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(255,0,0), 1);
                    
                    // kirim data ros
                    v2_detection::BallCoordinate bc;
                    bc.pos_x = clamp(x_center_ball/framesize[0]*2-1, -1.0f, 1.0f);
                    bc.pos_y = clamp(y_center_ball/framesize[1]*2-1, -1.0f, 1.0f);
                    bc.obj_size = obj_size_ball;
                    pub_coord.publish(bc);
                    
                    v2_detection::Ballarea ba;
                    ba.ballarea = obj_size_ball;
                    pub_area.publish(ba);
                    
                    v2_detection::BallState bs;
                    bs.ball_status = "FOUND";
                    pub_state.publish(bs);
                    
                    ROS_INFO("bola [%.0f%%]", d.conf * 100);
                    
                    // ambil hsv dari bola
                    get_hsv_val(img);
                    detect_status = FOUND;
                    waktu_sebelum = ros::Time::now().toSec();
                    
                    // update blobsize berdasarkan jarak bola
                    if(obj_size_ball <= 2800) {
                        blobsize = 320;
                    } else {
                        blobsize = 224;
                    }
                    break;
                }
                
                // jika tidak ada bola class 0
                if(detect_status != FOUND) {
                    v2_detection::BallCoordinate bc;
                    bc.pos_x = 0; bc.pos_y = 0; bc.obj_size = 0;
                    pub_coord.publish(bc);
                    
                    v2_detection::BallState bs;
                    bs.ball_status = "NOTFOUND";
                    pub_state.publish(bs);
                    
                    ROS_INFO("bola tidak ditemukan");
                }
            }
        }

        // hitung dan tampilkan fps
        calculate_fps();
        
        // tampilkan info di layar
        char fps_text[20];
        snprintf(fps_text, sizeof(fps_text), "FPS: %.1f", fps);
        putText(img_result, fps_text, Point(5, 15), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 255, 0), 2);
        
        char blob_text[10];
        snprintf(blob_text, sizeof(blob_text), "%d", blobsize);
        putText(img_result, blob_text, Point(280, 15), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 255, 0), 2);
        
        // tampilkan status deteksi
        ROS_INFO("\n%s", detect_status == FOUND ? "FOUND" : "NOTFOUND");
        
        imshow("VISION_CPP", img_result);
        waitKey(1);
        
        ros::spinOnce();
    }
    
    capture.stop();
    ROS_INFO("sistem berhenti");
    return 0;
}
