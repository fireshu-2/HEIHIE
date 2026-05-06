/*
 * @Filename: demo_rps_game.cpp
 * @Author: Hongying He
 * @Email: hongying.he@smartsenstech.com
 * @Date: 2026-04-20
 * @Copyright (c) 2026 SmartSens
 * @Description: RPS分类游戏演示程序
 */
#include <fstream>
#include <iostream>
#include <cstring>
#include <thread>
#include <mutex>
#include <fcntl.h>
#include <regex>
#include <dirent.h>
#include <unistd.h>
#include <cstdlib>
#include <ctime>
#include "include/utils.hpp"

using namespace std;

// 游戏阶段枚举
enum GamePhase {
    PHASE_READY,      // 准备阶段，显示 ready，等待 'a' 触发
    PHASE_COUNTDOWN,  // 倒计时阶段，顺序显示 1->2->3
    PHASE_BATTLE      // 比赛阶段，显示随机 r/p/s，推理判定胜负
};

// 全局退出标志（线程安全）
bool g_exit_flag = false;
// 保护退出标志的互斥锁
std::mutex g_mtx;

// 游戏状态（线程安全）
std::mutex g_game_mtx;
GamePhase g_phase = PHASE_READY;
bool g_trigger_battle = false;  // 'a' 按键触发标志

// OSD 贴图结构体
struct osdInfo {
    std::string filename; // OSD 文件名
    uint16_t x;           // 起始坐标 x
    uint16_t y;           // 起始坐标 y
};

/**
 * @brief 键盘监听程序，用于结束demo或触发游戏
 */
void keyboard_listener() {
    std::string input;
    std::cout << "输入 'a' 开始游戏，'q' 退出程序..." << std::endl;

    while (true) {
        // 读取键盘输入（会阻塞直到有输入）
        std::cin >> input;

        // 加锁修改退出标志
        std::lock_guard<std::mutex> lock(g_mtx);
        if (input == "q" || input == "Q") {
            g_exit_flag = true;
            std::cout << "检测到退出指令，通知主线程退出..." << std::endl;
            break;
        } else if (input == "a" || input == "A") {
            std::lock_guard<std::mutex> game_lock(g_game_mtx);
            if (g_phase == PHASE_READY) {
                g_trigger_battle = true;
                std::cout << "开始游戏!" << std::endl;
            }
        } else {
            std::cout << "输入无效（'a' 开始游戏，'q' 退出），请重新输入：" << std::endl;
        }
    }
}

/**
 * @brief 检查退出标志的辅助函数（线程安全）
 * @return 是否需要退出
 */
bool check_exit_flag() {
    std::lock_guard<std::mutex> lock(g_mtx);
    return g_exit_flag;
}

/**
 * @brief RPS分类游戏演示程序主函数
 * @return 执行结果，0表示成功
 */
int main() {
    /******************************************************************************************
     * 1. 参数配置
     ******************************************************************************************/

    // 图像尺寸配置（根据镜头参数修改）
    int img_width = 1920;    // 输入图像宽度
    int img_height = 1080;   // 输入图像高度

    // 模型配置参数
    array<int, 2> cls_shape = {320, 320};  // 分类模型输入尺寸
    string path_cls = "/app_demo/app_assets/models/model_rps.m1model";  // 石头剪刀布分类模型路径

    // OSD 位图信息表
    // 0:background, 1:r, 2:p, 3:s, 4:1, 5:2, 6:3, 7:ready
    static osdInfo osds[8] = {
        {"background.ssbmp", 0, 0},
        {"r.ssbmp", 960, 300},
        {"p.ssbmp", 960, 300},
        {"s.ssbmp", 960, 300},
        {"1.ssbmp", 1080, 270},
        {"2.ssbmp", 1080, 270},
        {"3.ssbmp", 1080, 270},
        {"ready.ssbmp", 960, 270}
    };

    /******************************************************************************************
     * 2. 系统初始化
     ******************************************************************************************/

    // 初始化随机数种子
    srand(static_cast<unsigned int>(time(nullptr)));

    // SSNE初始化
    if (ssne_initial()) {
        fprintf(stderr, "SSNE initialization failed!\n");
    }

    // 图像处理器初始化
    array<int, 2> img_shape = {img_width, img_height};  // 原始图像尺寸 1920×1080

    IMAGEPROCESSOR processor;
    processor.Initialize(&img_shape);  // 初始化图像处理器（配置原图尺寸）

    // 石头剪刀布分类模型初始化
    RPS_CLASSIFIER classifier;
    classifier.Initialize(path_cls, &img_shape, &cls_shape);  // 初始化分类器

    // OSD可视化器初始化
    VISUALIZER visualizer;
    visualizer.Initialize(img_shape, "shared_colorLUT.sscl");  // 初始化可视化器
    // 系统稳定等待
    cout << "sleep for 0.2 second!" << endl;
    sleep(0.2);  // 等待系统稳定

    // 绘制背景位图（layer 2，最底层位图图层，初始化时绘制，一直存在不清除）
    visualizer.DrawBitmap(osds[0].filename, "shared_colorLUT.sscl", osds[0].x, osds[0].y, 2);

    ssne_tensor_t img_sensor;  // 图像tensor定义

    // 创建键盘监听线程
    std::thread listener_thread(keyboard_listener);

    // 游戏状态变量
    uint16_t frame_counter = 0;   // 帧计数器
    int countdown_idx = 0;        // 倒计时索引 0/1/2 -> 1/2/3
    int random_rps = 0;           // 随机出的 r/p/s 索引 0=r,1=p,2=s
    std::string last_label;       // 上一帧分类结果
    float last_score = 0.0f;      // 上一帧分类得分

    // 比赛阶段10帧平均分类结果变量
    float accum_scores[3] = {0.0f, 0.0f, 0.0f};  // 累计得分 [P, R, S]
    std::string final_label = "NoTarget";          // 10帧后确定的最终标签

    /******************************************************************************************
     * 3. 主处理循环
     ******************************************************************************************/
    while (!check_exit_flag()) {

        // 从sensor获取图像（原图1920×1080）
        processor.GetImage(&img_sensor);

        // RPS分类模型推理（每帧都执行）
        std::string label;
        float score;
        float scores[3];
        classifier.Predict(&img_sensor, label, score, scores);
        last_label = label;
        last_score = score;

        // 游戏状态机处理
        {
            std::lock_guard<std::mutex> game_lock(g_game_mtx);

            switch (g_phase) {
                /******************************************************************************
                 * 阶段一：准备阶段
                 ******************************************************************************/
                case PHASE_READY: {
                    // 清理上一阶段可能残留的位图（layer 3/4），然后显示 ready
                    // visualizer.ClearLayer(3);
                    // visualizer.ClearLayer(4);
                    visualizer.DrawBitmap(osds[7].filename, "shared_colorLUT", osds[7].x, osds[7].y, 3);

                    // 检查是否触发进入倒计时
                    if (g_trigger_battle) {
                        g_trigger_battle = false;
                        g_phase = PHASE_COUNTDOWN;
                        frame_counter = 0;
                        countdown_idx = 0;
                    }
                    break;
                }

                /******************************************************************************
                 * 阶段二：倒计时阶段
                 ******************************************************************************/
                case PHASE_COUNTDOWN: {
                    // 每20帧切换一个数字：0~19显示1，20~39显示2，40~59显示3
                    if (frame_counter < 20) {
                        visualizer.DrawBitmap(osds[6].filename, "shared_colorLUT.sscl", osds[4].x, osds[4].y, 3);
                    } else if (frame_counter < 40) {
                        visualizer.DrawBitmap(osds[5].filename, "shared_colorLUT.sscl", osds[5].x, osds[5].y, 3);
                    } else if (frame_counter < 60) {
                        visualizer.DrawBitmap(osds[4].filename, "shared_colorLUT.sscl", osds[6].x, osds[6].y, 3);
                    } else {
                        // 倒计时结束，进入比赛阶段
                        g_phase = PHASE_BATTLE;
                        frame_counter = 0;
                        random_rps = rand() % 3;  // 随机选择 0=r, 1=p, 2=s
                        // 重置累计得分和最终标签
                        accum_scores[0] = accum_scores[1] = accum_scores[2] = 0.0f;
                        final_label = "NoTarget";
                        break;
                    }
                    frame_counter++;
                    break;
                }

                /******************************************************************************
                 * 阶段三：比赛阶段
                 ******************************************************************************/
                case PHASE_BATTLE: {
                    // 显示随机出的 r/p/s 位图（layer 3）
                    if (random_rps == 0) {
                        visualizer.DrawBitmap(osds[1].filename, "shared_colorLUT", osds[1].x, osds[1].y, 3);
                    } else if (random_rps == 1) {
                        visualizer.DrawBitmap(osds[2].filename, "shared_colorLUT", osds[2].x, osds[2].y, 3);
                    } else {
                        visualizer.DrawBitmap(osds[3].filename, "shared_colorLUT", osds[3].x, osds[3].y, 3);
                    }

                    // 前10帧收集分类得分，计算平均后确定最终标签
                    if (frame_counter < 5) {
                        accum_scores[0] += scores[0];
                        accum_scores[1] += scores[1];
                        accum_scores[2] += scores[2];

                        if (frame_counter == 4) {
                            // 10帧后计算平均得分并确定最终标签
                            float avg_p = accum_scores[0] / 5.0f;
                            float avg_r = accum_scores[1] / 5.0f;
                            float avg_s = accum_scores[2] / 5.0f;
                            float avg_scores[3] = {avg_p, avg_r, avg_s};

                            int max_idx = 0;
                            float max_score = avg_scores[0];
                            for (int i = 1; i < 3; i++) {
                                if (avg_scores[i] > max_score) {
                                    max_score = avg_scores[i];
                                    max_idx = i;
                                }
                            }

                            const char* labels[] = {"P", "R", "S"};
                            if (max_score > 0.6f) {
                                final_label = labels[max_idx];
                            } else {
                                final_label = "NoTarget";
                            }
                            printf("[BATTLE] 5帧平均结果: P=%.4f, R=%.4f, S=%.4f -> 最终标签: %s\n",
                                   avg_p, avg_r, avg_s, final_label.c_str());
                        }
                    }
                    if (frame_counter >= 4 && final_label != "NoTarget") {
                        bool is_win =
                            (final_label == "P" && random_rps == 0) ||  // P wins R
                            (final_label == "R" && random_rps == 2) ||  // R wins S
                            (final_label == "S" && random_rps == 1);    // S wins P

                        bool is_lose =
                            (final_label == "R" && random_rps == 1) ||  // R loses to P
                            (final_label == "S" && random_rps == 0) ||  // S loses to R
                            (final_label == "P" && random_rps == 2);    // P loses to S

                        if (is_win) {
                            // 赢：绘制在一个位置
                            visualizer.DrawBitmap("win.ssbmp", "shared_colorLUT", 510, 100, 4);
                        } else if (is_lose) {
                            // 输：绘制在另一个位置
                            visualizer.DrawBitmap("win.ssbmp", "shared_colorLUT", 1300, 100, 4);
                        }
                    }

                    frame_counter++;
                    // 持续80帧后回到准备阶段
                    if (frame_counter >= 80) {
                        g_phase = PHASE_READY;
                        // 清理比赛阶段位图
                        visualizer.ClearLayer(3);
                        visualizer.ClearLayer(4);
                    }
                    break;
                }
            }
        }
    }

    // 等待监听线程退出，释放资源
    if (listener_thread.joinable()) {
        listener_thread.join();
    }

    /******************************************************************************************
     * 4. 资源释放
     ******************************************************************************************/

    classifier.Release();  // 释放分类器资源
    processor.Release();  // 释放图像处理器资源
    visualizer.Release();  // 释放可视化器资源

    if (ssne_release()) {
        fprintf(stderr, "SSNE release failed!\n");
        return -1;
    }

    return 0;
}
