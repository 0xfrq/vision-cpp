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

/* Timing */
double last_seen = 0.0;
int hsv_fail = 0;
int yolo_skip_counter = 0;  // Skip YOLO frames when tracking
Point2f last_velocity(0,0);  // For prediction
int consecutive_found = 0;   // Lock-in counter
int consecutive_lost = 0;    // Loss counter

/* Params - Matching Python's direct tracking behavior */
constexpr int HSV_FAIL_MAX = 10;        // Increased tolerance
constexpr float POS_ALPHA = 0.0f;       // Direct position (no smoothing like Python)
constexpr float AREA_ALPHA = 0.0f;      // Direct area (no smoothing like Python)  
constexpr float VEL_ALPHA = 0.0f;       // No velocity prediction (match Python)
constexpr int YOLO_SKIP_FRAMES = 3;     // Skip YOLO when tracking well
constexpr int LOCK_IN_THRESHOLD = 3;    // Frames needed to lock tracking
constexpr int LOCK_OUT_THRESHOLD = 15;  // Frames needed to lose tracking

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
    last_seen=ros::Time::now().toSec();
    
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
            bool yolo_found = false;
            
            for(auto&d:dets){
                // Lower confidence for faster detection (0.4 instead of 0.5)
                if(d.class_id!=0 || d.conf<0.4f) continue;

                Rect b=d.box;
                Point2f nc(b.x+b.width/2.f,b.y+b.height/2.f);
                int na=b.area();

                // Direct assignment (no smoothing) - matching Python
                center=nc;
                smooth_center=nc;
                ball_area=na;
                smooth_area=na;
                velocity=Point2f(0,0);
                last_velocity=Point2f(0,0);
                initialized=true;
                last_box=b;

                // Calculate scan area based on ball size (matching Python)
                int in_area;
                if(ball_area <= 5000){
                    in_area = int(b.width * 3);
                }else{
                    in_area = b.width + 35;
                }
                scan_x={int(center.x-in_area),int(center.x+in_area)};

                extractHSV(frame,b);
                last_seen=ros::Time::now().toSec();
                hsv_fail=0;
                consecutive_found=0;
                consecutive_lost=0;
                yolo_skip_counter=0;
                state=FOUND;
                yolo_found=true;
                
                // Immediate publish on detection for fast response
                v2_detection::BallState bs_immediate;
                v2_detection::BallCoordinate bc_immediate;
                v2_detection::Ballarea ba_immediate;
                
                bs_immediate.ball_status="FOUND";
                bc_immediate.pos_x=clamp(center.x/frame.cols*2-1,-1.f,1.f);
                bc_immediate.pos_y=clamp(center.y/frame.rows*2-1,-1.f,1.f);
                bc_immediate.obj_size=ball_area;
                ba_immediate.ballarea=ball_area;
                
                pub_state.publish(bs_immediate);
                pub_coord.publish(bc_immediate);
                pub_area.publish(ba_immediate);
                
                ROS_INFO("YOLO: Ball locked - Area:%d, Conf:%.2f", ball_area, d.conf);
                break;
            }
            
            if(!yolo_found) {
                // Publish NOTFOUND only if really not detected
                v2_detection::BallState bs_nf;
                bs_nf.ball_status="NOTFOUND";
                pub_state.publish(bs_nf);
            }
        }

        /* ===== HSV TRACK ===== */
        else{
            // Skip field masking for speed when tracking is stable
            Mat track_frame = (consecutive_found > 5) ? frame : extractField(frame);

            Mat hsv,mask;
            cvtColor(track_frame,hsv,COLOR_BGR2HSV);
            inRange(hsv,Scalar(min_h,min_s,min_v),Scalar(max_h,max_s,max_v),mask);

            Mat kernel = Mat::ones(5,5,CV_8U);
            morphologyEx(mask,mask,MORPH_CLOSE,kernel);
            morphologyEx(mask,mask,MORPH_OPEN ,kernel);

            vector<vector<Point>> contours;
            findContours(mask,contours,RETR_EXTERNAL,CHAIN_APPROX_SIMPLE);

            bool found=false;
            double best_score = -1;
            Point2f best_center;
            int best_area = 0;
            Rect best_box;

            // Predict next position based on last velocity
            Point2f predicted_center = center + last_velocity;

            for(auto&c:contours){
                double a=contourArea(c);
                // More lenient area filtering: 15% to 130%
                if(a<ball_area*0.15||a>ball_area*1.3) continue;
                if(a<2000) continue; // Lower minimum threshold

                Rect r=boundingRect(c);
                int cx=r.x+r.width/2;
                int cy=r.y+r.height/2;
                
                // Expanded scan area for fast balls
                int expanded_scan = (scan_x[1]-scan_x[0])*1.5;
                int scan_center = (scan_x[0]+scan_x[1])/2;
                if(cx<scan_center-expanded_scan||cx>scan_center+expanded_scan) continue;

                Point2f nc(cx,cy);
                
                // Score based on distance to predicted position + area similarity
                float dist = norm(nc - predicted_center);
                float area_diff = abs(a - ball_area) / (float)ball_area;
                float score = 1.0f / (1.0f + dist/100.0f + area_diff*2.0f);
                
                if(score > best_score) {
                    best_score = score;
                    best_center = nc;
                    best_area = int(a);
                    best_box = r;
                    found = true;
                }
            }

            if(found){
                // Calculate velocity for next frame prediction
                last_velocity = best_center - center;
                
                // Direct assignment
                center=best_center;
                smooth_center=best_center;
                ball_area=best_area;
                smooth_area=best_area;
                last_box=best_box;

                last_seen=ros::Time::now().toSec();
                hsv_fail=0;
                consecutive_found++;
                consecutive_lost=0;
                
                // Immediate publish during tracking
                v2_detection::BallCoordinate bc_track;
                v2_detection::BallState bs_track;
                v2_detection::Ballarea ba_track;
                
                bc_track.pos_x=clamp(center.x/frame.cols*2-1,-1.f,1.f);
                bc_track.pos_y=clamp(center.y/frame.rows*2-1,-1.f,1.f);
                bc_track.obj_size=ball_area;
                bs_track.ball_status="FOUND";
                ba_track.ballarea=ball_area;
                
                pub_coord.publish(bc_track);
                pub_state.publish(bs_track);
                pub_area.publish(ba_track);
                
                // Print like Python
                ROS_INFO_THROTTLE(0.1, "\nFOUND\nBall Area Result : %d\n", ball_area);
            }else{
                hsv_fail++;
                consecutive_lost++;
                consecutive_found=0;
                
                // Lock-in mechanism: don't lose tracking immediately
                if(consecutive_lost > LOCK_OUT_THRESHOLD){
                    state=NOTFOUND;
                    initialized=false;
                    last_velocity=Point2f(0,0);
                    ROS_INFO("Track lost after %d frames - switching to YOLO", consecutive_lost);
                } else {
                    // Try prediction during temporary loss
                    center = center + last_velocity;
                    center.x = clamp(center.x, 0.f, (float)frame.cols-1);
                    center.y = clamp(center.y, 0.f, (float)frame.rows-1);
                    
                    // Still publish during prediction
                    v2_detection::BallCoordinate bc_pred;
                    bc_pred.pos_x=clamp(center.x/frame.cols*2-1,-1.f,1.f);
                    bc_pred.pos_y=clamp(center.y/frame.rows*2-1,-1.f,1.f);
                    bc_pred.obj_size=ball_area;
                    pub_coord.publish(bc_pred);
                }
            }
        }

        /* ===== ROS OUTPUT & DISPLAY ===== */
        v2_detection::BallState bs;
        v2_detection::BallCoordinate bc;
        v2_detection::Ballarea ba;

        if(state==FOUND){
            bs.ball_status="FOUND";
            bc.pos_x=clamp(center.x/frame.cols*2-1,-1.f,1.f);
            bc.pos_y=clamp(center.y/frame.rows*2-1,-1.f,1.f);
            bc.obj_size=ball_area;
            ba.ballarea=ball_area;
            pub_coord.publish(bc);
            pub_area.publish(ba);
            pub_state.publish(bs);
            
            rectangle(display_frame,last_box,Scalar(0,255,255),2);
            circle(display_frame, Point(int(center.x), int(center.y)), 5, Scalar(0,0,255), -1);
            
            // Velocity vector visualization
            if(norm(last_velocity) > 1.0) {
                Point2f vel_end = center + last_velocity*3.0f;
                arrowedLine(display_frame, Point(center), Point(vel_end), 
                           Scalar(255,0,255), 2);
            }
        }else{
            bs.ball_status="NOTFOUND";
            pub_state.publish(bs);
        }
        
        // Calculate and display FPS
        calculate_fps();
        
        // Draw FPS and status on display
        char fps_text[50];
        snprintf(fps_text, sizeof(fps_text), "FPS: %.1f", fps);
        putText(display_frame, fps_text, Point(5, 15), 
                FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 255, 0), 2);
        
        char status_text[50];
        snprintf(status_text, sizeof(status_text), "%s", 
                 state==FOUND ? "TRACKING" : "SEARCHING");
        putText(display_frame, status_text, Point(5, 35), 
                FONT_HERSHEY_SIMPLEX, 0.5, 
                state==FOUND ? Scalar(0, 255, 0) : Scalar(0, 0, 255), 2);
        
        imshow("VISION_CPP", display_frame);
        waitKey(1);

        ros::spinOnce();
    }
    
    capture.stop();
    ROS_INFO("Vision system shutdown");
    return 0;
}
