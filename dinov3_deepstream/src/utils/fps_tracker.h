#pragma once

#include <chrono>
#include <mutex>
#include <string>
#include <unordered_map>
#include <iostream>
#include <iomanip>

class FPSTracker {
public:
    static FPSTracker& getInstance() {
        static FPSTracker instance;
        return instance;
    }

    void update(const std::string& task_name, int count = 1) {
        std::lock_guard<std::mutex> lock(mtx);
        auto now = std::chrono::steady_clock::now();
        
        if (tasks.find(task_name) == tasks.end()) {
            tasks[task_name] = TaskStats{0};
        }
        tasks[task_name].frame_count += count;

        auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(now - last_print_time).count();
        if (elapsed >= 5000) { // 5 seconds
            float elapsed_sec = elapsed / 1000.0f;
            std::cout << "\n==================== [FPS MONITOR] ====================\n";
            for (const auto& pair : tasks) {
                float fps = static_cast<float>(pair.second.frame_count) / elapsed_sec;
                std::cout << "  Task: " << std::left << std::setw(15) << pair.first
                          << " | FPS: " << std::right << std::setw(6) << std::fixed << std::setprecision(2) << fps
                          << " (frames: " << pair.second.frame_count << ")\n";
            }
            // Reset frame counts for all tasks
            for (auto& pair : tasks) {
                pair.second.frame_count = 0;
            }
            std::cout << "=======================================================\n\n";
            last_print_time = now;
        }
    }

private:
    FPSTracker() : last_print_time(std::chrono::steady_clock::now()) {}
    
    struct TaskStats {
        int frame_count;
    };

    std::mutex mtx;
    std::unordered_map<std::string, TaskStats> tasks;
    std::chrono::steady_clock::time_point last_print_time;
};
