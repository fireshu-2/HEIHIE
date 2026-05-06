#pragma once

#include <vector>
#include <array>
#include <string>
#include <map>
#include <deque>
#include "common.hpp"

// Detection output structure
struct Detection {
    std::array<float, 4> box; // xmin, ymin, xmax, ymax
    float score;
    int class_id;
};

// YOLO Detector Class
class YOLO_DETECTOR {
public:
    std::string ModelName() const { return "yolov8_detector"; }

    void Initialize(std::string& model_path, std::array<int, 2>* in_img_shape,
                    std::array<int, 2>* in_cls_shape);

    void Predict(ssne_tensor_t* img_in, std::vector<Detection>& detections, float conf_thres = 0.38f);

    void Release();

private:
    uint16_t model_id = 0;
    ssne_tensor_t inputs[1];
    ssne_tensor_t outputs[1];
    AiPreprocessPipe pipe_offline = GetAIPreprocessPipe();

    std::array<int, 2> img_shape;
    std::array<int, 2> cls_shape;

    std::vector<Detection> NonMaximumSuppression(const std::vector<Detection>& input_detections, float iou_thres = 0.60f);
};


struct TrackRecord {
    std::array<float, 4> box;
    float conf;
    int hits;
    int lost;
};

class SimpleZoneTrackManager {
public:
    SimpleZoneTrackManager(int max_lost = 3, float iou_thres = 0.20f)
        : max_lost(max_lost), iou_thres(iou_thres), next_id(1) {}

    std::vector<int> update(const std::vector<Detection>& detections);
    std::map<int, TrackRecord> get_active() const { return tracks; }
    bool empty() const { return tracks.empty(); }

private:
    int max_lost;
    float iou_thres;
    int next_id;
    std::map<int, TrackRecord> tracks;

    static float compute_iou(const std::array<float, 4>& a, const std::array<float, 4>& b);
};

struct Point2f {
    float x;
    float y;
};

class IntrusionAnalyzer {
public:
    IntrusionAnalyzer(const std::array<Point2f, 2>& line, const std::vector<Point2f>& polygon, float dwell_seconds = 3.0f)
        : line_pts(line), zone_polygon(polygon), dwell_seconds(dwell_seconds) {}

    bool bbox_in_zone(const std::array<float, 4>& box);

    std::map<std::string, bool> update_track(int track_id, const Point2f& center, float now, bool box_in_zone_now);

    void clear_track(int track_id);

    // Add point polygon test logic
    static float pointPolygonTest(const std::vector<Point2f>& polygon, const Point2f& pt);

private:
    std::array<Point2f, 2> line_pts;
    std::vector<Point2f> zone_polygon;
    float dwell_seconds;

    std::map<int, Point2f> prev_centers;
    std::map<int, float> in_zone_since;
    std::map<int, bool> zone_state;
    std::map<int, int> zone_in_frames;
    std::map<int, int> zone_out_frames;

    float side_of_line(const Point2f& p, const Point2f& a, const Point2f& b);
};
