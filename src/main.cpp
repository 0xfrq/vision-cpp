#include <ros/ros.h>
#include <ros/package.h>

#include <v2_detection/BallState.h>
#include <v2_detection/BallCoordinate.h>
#include <v2_detection/Ballarea.h>

#include <opencv2/opencv.hpp>
#include <chrono>
#include <vector>
#include <algorithm>
#include <cmath>
#include <thread>
#include <mutex>

#include "yolo_onnx.hpp"

using namespace cv;
using namespace std;

/* =========================
   UTIL
   ========================= */
template<typename T>
inline T clamp(T v, T lo, T hi) {
    return (v < lo) ? lo : (v > hi) ? hi : v;
}

double map_value(double v, double smin, double smax, double tmin, double tmax) {
    v = clamp(v, min(smin, smax), max(smin, smax));
    return tmin + (v - smin) * (tmax - tmin) / (smax - smin);
}

/* =========================
   STATE
   ========================= */
enum DetectState { NOTFOUND = 0, FOUND = 1 };
DetectState state = NOTFOUND;

/* HSV */
int min_h=0, min_s=0, min_v=0;
int max_h=0, max_s=0, max_v=0;

/* Tracking */
Point2f center(0,0);
Point2f smooth_center(0,0);
Point2f velocity(0,0);
bool initialized = false;

int ball_area = 0;
int smooth_area = 0;
Rect last_box;
vector<int> scan_x(2,0);

/* Timing - matching Python */
double last_seen_time = 0.0;
double timeout_duration = 0.0;

/* Params - exact Python behavior */
constexpr double FISHEYE = 1.0;  // Set to 0.34 if using fisheye lens
constexpr int MIN_AREA_THRESHOLD = (int)(3000 * FISHEYE);

/* FPS Tracking */
int frame_counter = 0;
double fps_start_time = 0.0;
double fps = 0.0;
constexpr double FPS_DISPLAY_INTERVAL = 0.2;

/* =========================
   FPS CALCULATOR
   ========================= */
void calculate_fps() {
    frame_counter++;
    double current_time = ros::Time::now().toSec();
    double elapsed = current_time - fps_start_time;
    
    if(elapsed >= FPS_DISPLAY_INTERVAL) {
        fps = frame_counter / elapsed;
        frame_counter = 0;
        fps_start_time = current_time;
    }
}

/* =========================
   HSV SAMPLING (PY PORT)
   ========================= */
void extractHSV(const Mat& img, const Rect& b) {

    vector<Point> pts;
    int x1=b.x, y1=b.y, x2=b.x+b.width, y2=b.y+b.height;
    Point mid((x1+x2)/2, (y1+y2)/2);

    pts = {
        mid,
        {(x1+mid.x)/2,(y1+mid.y)/2},
        {(x2+mid.x)/2,(y2+mid.y)/2},
        {(mid.x+x2)/2,(mid.y+y1)/2},
        {(x1+mid.x)/2,(y1+y2+mid.y)/2},
        {mid.x,(mid.y+y1+b.height/5)/2},
        {mid.x,(mid.y+y2)/2}
    };

    vector<int> H,S,V;

    for (auto&p:pts){
        int px=clamp(p.x,0,img.cols-1);
        int py=clamp(p.y,0,img.rows-1);
        Vec3b bgr=img.at<Vec3b>(py,px);
        Mat hsv;
        cvtColor(Mat(1,1,CV_8UC3,Scalar(bgr[0],bgr[1],bgr[2])),hsv,COLOR_BGR2HSV);
        Vec3b hv=hsv.at<Vec3b>(0,0);
        H.push_back(hv[0]); S.push_back(hv[1]); V.push_back(hv[2]);
    }

    min_h=*min_element(H.begin(),H.end());
    max_h=min(*max_element(H.begin(),H.end()),33);

    min_s=max(*min_element(S.begin(),S.end()),160);
    max_s=*max_element(S.begin(),S.end());

    min_v=*min_element(V.begin(),V.end());
    max_v=*max_element(V.begin(),V.end());
    
    // Debug output (matching Python)
    ROS_INFO_THROTTLE(2.0, "HSV Range - H:[%d,%d] S:[%d,%d] V:[%d,%d]", 
                     min_h, max_h, min_s, max_s, min_v, max_v);
}

/* =========================
   THREADED VIDEO CAPTURE (matching Python WebcamVideoStream)
   ========================= */
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
        cap.set(CAP_PROP_FRAME_WIDTH, 320);
        cap.set(CAP_PROP_FRAME_HEIGHT, 240);
        
        if(!cap.isOpened()) {
            ROS_ERROR("Camera failed to open");
            return;
        }
        
        cap >> frame;  // grab first frame
        captureThread = thread(&ThreadedCapture::update, this);
    }
    
    Mat read() {
        lock_guard<mutex> lock(frameMutex);
        return frame.clone();
    }
    
    void stop() {
        stopped = true;
        if(captureThread.joinable()) {
            captureThread.join();
        }
        cap.release();
    }
    
    ~ThreadedCapture() {
        stop();
    }
};

/* =========================
   FIELD MASKING
   ========================= */
Mat extractField(const Mat& img) {
    // Green field detection (adjust HSV values as needed)
    Mat hsv, mask, result;
    cvtColor(img, hsv, COLOR_BGR2HSV);
    
    // Green field HSV range - adjust these values for your field
    Scalar lower(35, 40, 40);
    Scalar upper(85, 255, 255);
    
    inRange(hsv, lower, upper, mask);
    
    Mat kernel = getStructuringElement(MORPH_RECT, Size(5,5));
    erode(mask, mask, kernel, Point(-1,-1), 2);
    dilate(mask, mask, kernel, Point(-1,-1), 5);
    
    // Find largest contour (field)
    vector<vector<Point>> contours;
    findContours(mask, contours, RETR_TREE, CHAIN_APPROX_NONE);
    
    Mat fieldMask = Mat::zeros(img.size(), CV_8UC1);
    if(!contours.empty()){
        auto maxContour = *max_element(contours.begin(), contours.end(),
            [](const vector<Point>&a, const vector<Point>&b){
                return contourArea(a) < contourArea(b);
            });
        vector<Point> hull;
        convexHull(maxContour, hull);
        fillConvexPoly(fieldMask, hull, Scalar(255));
    }
    
    bitwise_and(img, img, result, fieldMask);
    return result;
}

/* =========================
   MAIN
   ========================= */
int main(int argc,char**argv){

    ros::init(argc,argv,"vision_yolo_cpp");
    ros::NodeHandle nh;

    auto pub_state = nh.advertise<v2_detection::BallState>(
        "/DEWO/image_processing/deteksi_bola/ball_state",10);
    auto pub_coord = nh.advertise<v2_detection::BallCoordinate>(
        "/DEWO/image_processing/deteksi_bola/coordinate",10);
    auto pub_area  = nh.advertise<v2_detection::Ballarea>(
        "/DEWO/image_processing/deteksi_bola/ball_area",10);

    string pkg=ros::package::getPath("vision_cpp");
    YoloONNX yolo(pkg+"/src/best.onnx");

    // Use threaded capture (matching Python WebcamVideoStream)
    ROS_INFO("Initializing threaded camera capture...");
    ThreadedCapture capture(0);
    
    fps_start_time = ros::Time::now().toSec();
    last_seen_time = ros::Time::now().toSec();
    
    ROS_INFO("Vision system ready - starting main loop");

    while(ros::ok()){

        Mat frame = capture.read();
        if(frame.empty()) {
            ros::spinOnce();
            continue;
        }
        
        Mat display_frame = frame.clone();

        /* ===== YOLO ===== */
        if(state==NOTFOUND){
            auto dets=yolo.infer(frame);
            
            for(auto&d:dets){
                if(d.class_id!=0 || d.conf<0.5f) continue;

                Rect b=d.box;
                Point2f nc(b.x+b.width/2.f,b.y+b.height/2.f);
                int na=b.area();

                // Direct assignment - exact Python
                center=nc;
                smooth_center=nc;
                ball_area=na;
                smooth_area=na;
                initialized=true;
                last_box=b;

                // Extract HSV from 7 sample points (matching Python)
                extractHSV(frame,b);
                
                // Calculate timeout based on ball area (matching Python)
                timeout_duration = map_value(ball_area, 0, 76800, 0.5, 80);
                last_seen_time = ros::Time::now().toSec();
                
                // Calculate scan area (matching Python logic)
                int in_area_ball;
                if(ball_area <= (int)(5000*FISHEYE)) {
                    in_area_ball = b.width * 3;
                } else {
                    in_area_ball = b.width + (int)(35*FISHEYE);
                }
                scan_x[0] = center.x - in_area_ball;
                scan_x[1] = center.x + in_area_ball;
                
                state=FOUND;
                
                // Publish immediately
                v2_detection::BallState bs;
                v2_detection::BallCoordinate bc;
                v2_detection::Ballarea ba;
                
                bs.ball_status="FOUND";
                bc.pos_x=clamp(center.x/frame.cols*2-1,-1.f,1.f);
                bc.pos_y=clamp(center.y/frame.rows*2-1,-1.f,1.f);
                bc.obj_size=ball_area;
                ba.ballarea=ball_area;
                
                pub_state.publish(bs);
                pub_coord.publish(bc);
                pub_area.publish(ba);
                
                ROS_INFO("bola [%.0f%%]", d.conf*100);
                break;
            }
        }

        /* ===== HSV TRACK ===== */
        else{
            // Field masking (matching Python)
            Mat field_frame = extractField(frame);

            Mat hsv,mask;
            cvtColor(field_frame,hsv,COLOR_BGR2HSV);
            inRange(hsv,Scalar(min_h,min_s,min_v),Scalar(max_h,max_s,max_v),mask);

            Mat kernel = Mat::ones(5,5,CV_8U);
            morphologyEx(mask,mask,MORPH_CLOSE,kernel);
            morphologyEx(mask,mask,MORPH_OPEN ,kernel);

            vector<vector<Point>> contours;
            findContours(mask,contours,RETR_EXTERNAL,CHAIN_APPROX_SIMPLE);

            bool found=false;

            for(auto&c:contours){
                double a=contourArea(c);
                
                // Area filtering: 20% to 110% (matching Python)
                if(a < ball_area/5.0 || a > ball_area*1.1) continue;
                if((int)a < MIN_AREA_THRESHOLD) continue;

                Rect r=boundingRect(c);
                int cx=r.x+r.width/2;
                
                // Scan area check (matching Python)
                if(cx < scan_x[0] || cx > scan_x[1]) continue;

                // Direct assignment (matching Python)
                Point2f nc(cx, r.y+r.height/2.f);
                center=nc;
                smooth_center=nc;
                ball_area=(int)a;
                smooth_area=(int)a;
                last_box=r;

                found=true;
                
                // Publish immediately
                v2_detection::BallCoordinate bc;
                v2_detection::BallState bs;
                v2_detection::Ballarea ba;
                
                bc.pos_x=clamp(center.x/frame.cols*2-1,-1.f,1.f);
                bc.pos_y=clamp(center.y/frame.rows*2-1,-1.f,1.f);
                bc.obj_size=ball_area;
                bs.ball_status="FOUND";
                ba.ballarea=ball_area;
                
                pub_coord.publish(bc);
                pub_state.publish(bs);
                pub_area.publish(ba);
                
                ROS_INFO("Ball Area Result : %d", ball_area);
                break;
            }

            // Timeout check (matching Python)
            if(found){
                last_seen_time = ros::Time::now().toSec();
            } else {
                double current_time = ros::Time::now().toSec();
                double delta = current_time - last_seen_time;
                if(delta >= timeout_duration){
                    state=NOTFOUND;
                    initialized=false;
                    ROS_INFO("Timeout after %.2fs - switching to YOLO", delta);
                }
            }
        }

        /* ===== DISPLAY ===== */
        if(state==FOUND){
            rectangle(display_frame,last_box,Scalar(0,255,255),2);
            circle(display_frame, Point(int(center.x), int(center.y)), 5, Scalar(0,0,255), -1);
            
            // Draw scan area (yellow vertical lines)
            line(display_frame, Point(scan_x[0],0), Point(scan_x[0],display_frame.rows), 
                 Scalar(0,255,255), 1);
            line(display_frame, Point(scan_x[1],0), Point(scan_x[1],display_frame.rows), 
                 Scalar(0,255,255), 1);
        }
        
        // Calculate and display FPS
        calculate_fps();
        
        // Draw FPS and status on display (matching Python)
        char fps_text[64];
        snprintf(fps_text, sizeof(fps_text), "FPS: %.1f | %s", fps, 
                 state==FOUND ? "FOUND" : "NOTFOUND");
        putText(display_frame, fps_text, Point(5, 15), 
                FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 255, 0), 2);
        
        imshow("VISION_CPP", display_frame);
        waitKey(1);

        ros::spinOnce();
    }
    
    capture.stop();
    ROS_INFO("Vision system shutdown");
    return 0;
}
