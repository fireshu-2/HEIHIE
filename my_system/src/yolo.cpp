#include "../include/yolo.hpp"
#include <iostream>
#include <cmath>
#include <algorithm>

void YOLO_DETECTOR::Initialize(std::string& model_path, std::array<int, 2>* in_img_shape,
                               std::array<int, 2>* in_cls_shape) {
    img_shape = *in_img_shape;
    cls_shape = *in_cls_shape;

    char* model_path_char = const_cast<char*>(model_path.c_str());
    model_id = ssne_loadmodel(model_path_char, SSNE_STATIC_ALLOC);

    uint32_t cls_width = static_cast<uint32_t>(cls_shape[0]);
    uint32_t cls_height = static_cast<uint32_t>(cls_shape[1]);
    inputs[0] = create_tensor(cls_width, cls_height, SSNE_RGB, SSNE_BUF_AI);

    // No crop, just resize for YOLO full frame
    // SetNormalize is automatically read from model
    SetNormalize(pipe_offline, model_id);

    printf("[INFO] YOLO detector initialized with input shape [%d, %d]\n", cls_shape[0], cls_shape[1]);
}

void YOLO_DETECTOR::Release() {
    release_tensor(inputs[0]);
    release_tensor(outputs[0]);
    ReleaseAIPreprocessPipe(pipe_offline);
}

void YOLO_DETECTOR::Predict(ssne_tensor_t* img_in, std::vector<Detection>& detections, float conf_thres) {
    detections.clear();

    int ret = RunAiPreprocessPipe(pipe_offline, *img_in, inputs[0]);
    if (ret != 0) {
        printf("[ERROR] Failed to run AI preprocess pipe!\n");
        return;
    }

    int dtype = -1;
    ssne_get_model_input_dtype(model_id, &dtype);
    set_data_type(inputs[0], dtype);

    if (ssne_inference(model_id, 1, inputs)) {
        fprintf(stderr, "ssne inference fail!\n");
        return;
    }

    ssne_getoutput(model_id, 1, outputs);
    float* data = (float*)get_data(outputs[0]);
    int num_dims = 0;
    int* dims = nullptr;
    ssne_get_tensor_shape(outputs[0], &num_dims, &dims);

    // Assume typical YOLOv8 output: [1, 84, 8400]
    int num_classes = dims[1] - 4; // usually 80 classes, dim is 4 (bbox) + classes
    int num_anchors = dims[2];

    std::vector<Detection> raw_detections;

    float scale_x = (float)img_shape[0] / cls_shape[0];
    float scale_y = (float)img_shape[1] / cls_shape[1];

    for (int i = 0; i < num_anchors; ++i) {
        float max_class_score = 0;
        int class_id = -1;
        for (int c = 0; c < num_classes; ++c) {
            float score = data[(4 + c) * num_anchors + i];
            if (score > max_class_score) {
                max_class_score = score;
                class_id = c;
            }
        }

        if (max_class_score >= conf_thres && class_id == 0) { // Only person (class 0)
            float cx = data[0 * num_anchors + i];
            float cy = data[1 * num_anchors + i];
            float w = data[2 * num_anchors + i];
            float h = data[3 * num_anchors + i];

            float xmin = (cx - w / 2.0f) * scale_x;
            float ymin = (cy - h / 2.0f) * scale_y;
            float xmax = (cx + w / 2.0f) * scale_x;
            float ymax = (cy + h / 2.0f) * scale_y;

            Detection det;
            det.box = {xmin, ymin, xmax, ymax};
            det.score = max_class_score;
            det.class_id = class_id;
            raw_detections.push_back(det);
        }
    }

    detections = NonMaximumSuppression(raw_detections, 0.60f);
}

std::vector<Detection> YOLO_DETECTOR::NonMaximumSuppression(const std::vector<Detection>& input_detections, float iou_thres) {
    std::vector<Detection> result;
    std::vector<Detection> dets = input_detections;

    std::sort(dets.begin(), dets.end(), [](const Detection& a, const Detection& b) {
        return a.score > b.score;
    });

    std::vector<bool> suppressed(dets.size(), false);

    for (size_t i = 0; i < dets.size(); ++i) {
        if (suppressed[i]) continue;
        result.push_back(dets[i]);

        for (size_t j = i + 1; j < dets.size(); ++j) {
            if (suppressed[j]) continue;
            float iou = SimpleZoneTrackManager::compute_iou(dets[i].box, dets[j].box);
            if (iou > iou_thres) {
                suppressed[j] = true;
            }
        }
    }
    return result;
}

float SimpleZoneTrackManager::compute_iou(const std::array<float, 4>& a, const std::array<float, 4>& b) {
    float x1 = std::max(a[0], b[0]);
    float y1 = std::max(a[1], b[1]);
    float x2 = std::min(a[2], b[2]);
    float y2 = std::min(a[3], b[3]);

    float w = std::max(0.0f, x2 - x1);
    float h = std::max(0.0f, y2 - y1);
    float inter = w * h;

    float area_a = std::max(0.0f, a[2] - a[0]) * std::max(0.0f, a[3] - a[1]);
    float area_b = std::max(0.0f, b[2] - b[0]) * std::max(0.0f, b[3] - b[1]);

    return inter / (area_a + area_b - inter + 1e-6f);
}

std::vector<int> SimpleZoneTrackManager::update(const std::vector<Detection>& detections) {
    std::vector<int> removed;
    std::vector<bool> det_matched(detections.size(), false);
    std::vector<int> track_matched;

    for (auto& pair : tracks) {
        int tid = pair.first;
        float best_iou = 0.0f;
        int best_det_idx = -1;

        for (size_t i = 0; i < detections.size(); ++i) {
            if (det_matched[i]) continue;
            float iou = compute_iou(tracks[tid].box, detections[i].box);
            if (iou > best_iou) {
                best_iou = iou;
                best_det_idx = i;
            }
        }

        if (best_det_idx != -1 && best_iou >= iou_thres) {
            tracks[tid].box = detections[best_det_idx].box;
            tracks[tid].conf = detections[best_det_idx].score;
            tracks[tid].lost = 0;
            tracks[tid].hits++;
            det_matched[best_det_idx] = true;
            track_matched.push_back(tid);
        } else {
            tracks[tid].lost++;
            if (tracks[tid].lost > max_lost) {
                removed.push_back(tid);
            }
        }
    }

    for (int tid : removed) tracks.erase(tid);

    for (size_t i = 0; i < detections.size(); ++i) {
        if (!det_matched[i]) {
            int tid = next_id++;
            tracks[tid] = {detections[i].box, detections[i].score, 1, 0};
        }
    }

    return removed;
}

float IntrusionAnalyzer::pointPolygonTest(const std::vector<Point2f>& polygon, const Point2f& pt) {
    bool c = false;
    int i, j;
    int nvert = polygon.size();
    for (i = 0, j = nvert - 1; i < nvert; j = i++) {
        if (((polygon[i].y > pt.y) != (polygon[j].y > pt.y)) &&
            (pt.x < (polygon[j].x - polygon[i].x) * (pt.y - polygon[i].y) / (polygon[j].y - polygon[i].y + 1e-6f) + polygon[i].x))
            c = !c;
    }
    return c ? 1.0f : -1.0f;
}

bool IntrusionAnalyzer::bbox_in_zone(const std::array<float, 4>& box) {
    float x1 = box[0], y1 = box[1], x2 = box[2], y2 = box[3];
    float w = std::max(1.0f, x2 - x1);
    float h = std::max(1.0f, y2 - y1);

    std::vector<Point2f> points = {
        {(x1 + x2) / 2.0f, y2},
        {x1 + 0.30f * w, y2},
        {x1 + 0.70f * w, y2},
        {(x1 + x2) / 2.0f, y1 + 0.85f * h},
        {x1 + 0.35f * w, y1 + 0.85f * h},
        {x1 + 0.65f * w, y1 + 0.85f * h},
        {(x1 + x2) / 2.0f, y1 + 0.70f * h},
        {x1 + 0.35f * w, y1 + 0.70f * h},
        {x1 + 0.65f * w, y1 + 0.70f * h}
    };

    int inside_count = 0;
    for (const auto& p : points) {
        if (pointPolygonTest(zone_polygon, p) > 0) {
            inside_count++;
        }
    }

    return inside_count >= 2;
}

float IntrusionAnalyzer::side_of_line(const Point2f& p, const Point2f& a, const Point2f& b) {
    return (b.x - a.x) * (p.y - a.y) - (b.y - a.y) * (p.x - a.x);
}

std::map<std::string, bool> IntrusionAnalyzer::update_track(int track_id, const Point2f& center, float now, bool box_in_zone_now) {
    std::map<std::string, bool> state;
    state["crossed"] = false;

    if (prev_centers.count(track_id)) {
        Point2f prev_p = prev_centers[track_id];
        float s1 = side_of_line(prev_p, line_pts[0], line_pts[1]);
        float s2 = side_of_line(center, line_pts[0], line_pts[1]);

        if (s1 == 0.0f) s1 = 1e-6f;
        if (s2 == 0.0f) s2 = 1e-6f;

        if (s1 * s2 < 0) {
            state["crossed"] = true;
        }
    }

    if (box_in_zone_now) {
        zone_in_frames[track_id]++;
        zone_out_frames[track_id] = 0;
    } else {
        zone_out_frames[track_id]++;
        zone_in_frames[track_id] = 0;
    }

    if (box_in_zone_now && !zone_state[track_id] && zone_in_frames[track_id] >= 1) {
        zone_state[track_id] = true;
        in_zone_since[track_id] = now;
    } else if (!box_in_zone_now && zone_state[track_id] && zone_out_frames[track_id] >= 3) {
        zone_state[track_id] = false;
        in_zone_since.erase(track_id);
    }

    state["in_zone"] = zone_state[track_id];
    state["dwell_alarm"] = false;

    if (state["in_zone"]) {
        float dwell_time = now - (in_zone_since.count(track_id) ? in_zone_since[track_id] : now);
        if (dwell_time >= dwell_seconds) {
            state["dwell_alarm"] = true;
        }
    }

    prev_centers[track_id] = center;
    return state;
}

void IntrusionAnalyzer::clear_track(int track_id) {
    prev_centers.erase(track_id);
    in_zone_since.erase(track_id);
    zone_state.erase(track_id);
    zone_in_frames.erase(track_id);
    zone_out_frames.erase(track_id);
}
