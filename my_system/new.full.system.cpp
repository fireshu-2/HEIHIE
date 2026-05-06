#include <iostream>
#include <thread>
#include <mutex>
#include <atomic>
#include <chrono>
#include "include/common.hpp"
#include "include/utils.hpp"
#include "include/yolo.hpp"
#include <unistd.h>
#include <sys/time.h>

std::mutex g_mtx;
std::atomic<bool> g_exit_flag(false);

void keyboard_listener() {
    char c;
    while (!g_exit_flag) {
        c = getchar();
        if (c == 'q' || c == 'Q') {
            g_exit_flag = true;
            break;
        }
    }
}

bool check_exit_flag() {
    return g_exit_flag;
}

double get_current_time() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec + tv.tv_usec / 1000000.0;
}

int main() {
    int img_width = 1920;
    int img_height = 1080;

    std::array<int, 2> cls_shape = {640, 640};
    std::string path_cls = "/app_demo/app_assets/models/yolov8n.m1model";

    if (ssne_initial()) {
        fprintf(stderr, "SSNE initialization failed!\n");
        return -1;
    }

    std::array<int, 2> img_shape = {img_width, img_height};

    IMAGEPROCESSOR processor;
    processor.Initialize(&img_shape);

    YOLO_DETECTOR detector;
    detector.Initialize(path_cls, &img_shape, &cls_shape);

    VISUALIZER visualizer;
    visualizer.Initialize(img_shape, "");
    sleep(1);

    std::array<Point2f, 2> line_pts = {{{220, 260}, {580, 260}}};
    std::vector<Point2f> zone_polygon = {{180, 280}, {620, 280}, {620, 460}, {180, 460}};

    // Draw the hardcoded zones using visualizer lines.
    // The VISUALIZER currently doesn't have a draw line method out of the box, we will draw the line using Draw method with very thin rectangles or we can just draw fixed square for zone.
    visualizer.DrawFixedSquare(180, 280, 620, 460, 1);

    IntrusionAnalyzer analyzer(line_pts, zone_polygon, 3.0f);
    SimpleZoneTrackManager tracker(3, 0.20f);

    ssne_tensor_t img_sensor;
    std::thread listener_thread(keyboard_listener);

    double prev_time = get_current_time();

    while (!check_exit_flag()) {
        processor.GetImage(&img_sensor);

        double now = get_current_time();

        std::vector<Detection> detections;
        detector.Predict(&img_sensor, detections, 0.38f);

        std::vector<Detection> valid_detections;
        for (const auto& det : detections) {
            if (analyzer.bbox_in_zone(det.box)) {
                valid_detections.push_back(det);
            }
        }

        std::vector<int> removed_ids = tracker.update(valid_detections);
        for (int rid : removed_ids) {
            analyzer.clear_track(rid);
        }

        std::vector<std::array<float, 4>> boxes_to_draw;
        for (const auto& pair : tracker.get_active()) {
            int tid = pair.first;
            TrackRecord trk = pair.second;

            if (trk.hits < 2 || trk.lost > 1) continue;

            float cx = (trk.box[0] + trk.box[2]) / 2.0f;
            float cy = trk.box[3];
            Point2f center = {cx, cy};

            bool box_in_zone = analyzer.bbox_in_zone(trk.box);
            auto state = analyzer.update_track(tid, center, now, box_in_zone);

            if (state["crossed"]) {
                printf("[ALARM] LineCross ID: %d\n", tid);
            }
            if (state["in_zone"] && state["dwell_alarm"]) {
                printf("[ALARM] DwellAlarm ID: %d\n", tid);
            }

            boxes_to_draw.push_back(trk.box);
        }

        visualizer.Draw(boxes_to_draw);

        double fps = 1.0 / std::max(now - prev_time, 1e-6);
        prev_time = now;

        // Print FPS periodically (every ~30 frames)
        static int frame_idx = 0;
        if (frame_idx++ % 30 == 0) {
            printf("[INFO] FPS: %.1f\n", fps);
        }
    }

    if (listener_thread.joinable()) {
        listener_thread.join();
    }

    detector.Release();
    processor.Release();
    visualizer.Release();

    if (ssne_release()) {
        fprintf(stderr, "SSNE release failed!\n");
        return -1;
    }

    return 0;
}
