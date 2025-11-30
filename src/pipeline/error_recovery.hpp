/**
 * @file error_recovery.hpp
 * @brief Error recovery and logging framework for TurboLoader v1.8.0
 *
 * Provides:
 * - Graceful handling of corrupted files
 * - Configurable error thresholds
 * - Structured error logging
 * - Performance profiling support
 *
 * Usage:
 *   ErrorRecovery recovery(config);
 *   if (!recovery.handle_error("decode_error", "corrupted.jpg", "Invalid JPEG")) {
 *       // Max errors exceeded, should stop processing
 *   }
 */

#pragma once

#include <string>
#include <atomic>
#include <mutex>
#include <fstream>
#include <chrono>
#include <functional>
#include <vector>
#include <unordered_map>
#include <memory>
#include <iostream>
#include <iomanip>
#include <sstream>

namespace turboloader {
namespace pipeline {

/**
 * @brief Error severity levels
 */
enum class ErrorSeverity {
    DEBUG,      // Verbose debug information
    INFO,       // General information
    WARNING,    // Non-fatal issues (corrupted files skipped)
    ERROR,      // Errors that don't stop processing
    CRITICAL    // Fatal errors that stop processing
};

/**
 * @brief Single error record
 */
struct ErrorRecord {
    std::chrono::system_clock::time_point timestamp;
    ErrorSeverity severity;
    std::string error_type;
    std::string filename;
    std::string message;
    size_t worker_id;
};

/**
 * @brief Callback for error handling
 */
using ErrorCallback = std::function<void(const ErrorRecord&)>;

/**
 * @brief Error recovery configuration
 */
struct ErrorRecoveryConfig {
    bool skip_corrupted = true;        // Skip corrupted files
    size_t max_errors = 100;           // Max errors before failing (0 = unlimited)
    bool log_errors = true;            // Enable error logging
    std::string log_path = "";         // Log file path (empty = stderr)
    ErrorSeverity min_level = ErrorSeverity::WARNING;  // Minimum log level
    ErrorCallback callback = nullptr;  // Optional callback on errors
};

/**
 * @brief Thread-safe error recovery manager
 */
class ErrorRecovery {
public:
    explicit ErrorRecovery(const ErrorRecoveryConfig& config = {})
        : config_(config), error_count_(0), warning_count_(0) {

        if (!config_.log_path.empty()) {
            log_file_ = std::make_unique<std::ofstream>(
                config_.log_path, std::ios::app
            );
            if (!log_file_->is_open()) {
                std::cerr << "[TurboLoader] Warning: Could not open log file: "
                         << config_.log_path << std::endl;
                log_file_.reset();
            }
        }
    }

    ~ErrorRecovery() {
        if (log_file_ && log_file_->is_open()) {
            log_file_->close();
        }
    }

    /**
     * @brief Handle an error and decide whether to continue
     * @return true if processing should continue, false if max errors exceeded
     */
    bool handle_error(const std::string& error_type,
                      const std::string& filename,
                      const std::string& message,
                      size_t worker_id = 0,
                      ErrorSeverity severity = ErrorSeverity::WARNING) {

        // Create error record
        ErrorRecord record{
            std::chrono::system_clock::now(),
            severity,
            error_type,
            filename,
            message,
            worker_id
        };

        // Log the error
        if (config_.log_errors && severity >= config_.min_level) {
            log_error(record);
        }

        // Invoke callback if set
        if (config_.callback) {
            config_.callback(record);
        }

        // Track error counts
        if (severity >= ErrorSeverity::ERROR) {
            error_count_++;
        } else if (severity == ErrorSeverity::WARNING) {
            warning_count_++;
        }

        // Store error for later retrieval
        {
            std::lock_guard<std::mutex> lock(errors_mutex_);
            errors_.push_back(record);

            // Limit stored errors to prevent memory issues
            if (errors_.size() > 10000) {
                errors_.erase(errors_.begin(), errors_.begin() + 1000);
            }
        }

        // Check if max errors exceeded
        if (config_.max_errors > 0 && error_count_ >= config_.max_errors) {
            log_error({
                std::chrono::system_clock::now(),
                ErrorSeverity::CRITICAL,
                "max_errors_exceeded",
                "",
                "Maximum error threshold (" + std::to_string(config_.max_errors) +
                    ") exceeded. Stopping processing.",
                0
            });
            return false;
        }

        return config_.skip_corrupted;
    }

    /**
     * @brief Get total error count
     */
    size_t error_count() const { return error_count_.load(); }

    /**
     * @brief Get warning count
     */
    size_t warning_count() const { return warning_count_.load(); }

    /**
     * @brief Get all recorded errors
     */
    std::vector<ErrorRecord> get_errors() const {
        std::lock_guard<std::mutex> lock(errors_mutex_);
        return errors_;
    }

    /**
     * @brief Clear error history
     */
    void clear_errors() {
        std::lock_guard<std::mutex> lock(errors_mutex_);
        errors_.clear();
        error_count_ = 0;
        warning_count_ = 0;
    }

    /**
     * @brief Check if processing should continue
     */
    bool should_continue() const {
        if (config_.max_errors == 0) return true;
        return error_count_ < config_.max_errors;
    }

    /**
     * @brief Get error summary string
     */
    std::string get_summary() const {
        std::ostringstream ss;
        ss << "Errors: " << error_count_.load()
           << ", Warnings: " << warning_count_.load();

        if (!errors_.empty()) {
            std::lock_guard<std::mutex> lock(errors_mutex_);

            // Count by type
            std::unordered_map<std::string, size_t> by_type;
            for (const auto& e : errors_) {
                by_type[e.error_type]++;
            }

            ss << "\nBy type:";
            for (const auto& [type, count] : by_type) {
                ss << "\n  " << type << ": " << count;
            }
        }

        return ss.str();
    }

private:
    void log_error(const ErrorRecord& record) {
        std::ostringstream ss;

        // Format timestamp
        auto time_t = std::chrono::system_clock::to_time_t(record.timestamp);
        ss << std::put_time(std::localtime(&time_t), "%Y-%m-%d %H:%M:%S");

        // Severity
        ss << " [" << severity_to_string(record.severity) << "]";

        // Error details
        ss << " [" << record.error_type << "]";

        if (!record.filename.empty()) {
            ss << " " << record.filename << ":";
        }

        ss << " " << record.message;

        if (record.worker_id > 0) {
            ss << " (worker " << record.worker_id << ")";
        }

        std::string log_line = ss.str();

        // Write to appropriate destination
        std::lock_guard<std::mutex> lock(log_mutex_);

        if (log_file_ && log_file_->is_open()) {
            *log_file_ << log_line << std::endl;
            log_file_->flush();
        } else {
            std::cerr << "[TurboLoader] " << log_line << std::endl;
        }
    }

    static const char* severity_to_string(ErrorSeverity severity) {
        switch (severity) {
            case ErrorSeverity::DEBUG: return "DEBUG";
            case ErrorSeverity::INFO: return "INFO";
            case ErrorSeverity::WARNING: return "WARNING";
            case ErrorSeverity::ERROR: return "ERROR";
            case ErrorSeverity::CRITICAL: return "CRITICAL";
            default: return "UNKNOWN";
        }
    }

    ErrorRecoveryConfig config_;
    std::atomic<size_t> error_count_;
    std::atomic<size_t> warning_count_;

    mutable std::mutex errors_mutex_;
    std::vector<ErrorRecord> errors_;

    std::mutex log_mutex_;
    std::unique_ptr<std::ofstream> log_file_;
};

/**
 * @brief Performance profiler for pipeline operations
 */
class PipelineProfiler {
public:
    struct TimingStats {
        std::string operation;
        size_t count;
        double total_ms;
        double min_ms;
        double max_ms;
        double avg_ms() const { return count > 0 ? total_ms / count : 0; }
    };

    PipelineProfiler() : enabled_(false) {}

    void enable() { enabled_ = true; }
    void disable() { enabled_ = false; }
    bool is_enabled() const { return enabled_; }

    /**
     * @brief Start timing an operation
     */
    void start(const std::string& operation) {
        if (!enabled_) return;

        std::lock_guard<std::mutex> lock(mutex_);
        start_times_[operation] = std::chrono::high_resolution_clock::now();
    }

    /**
     * @brief End timing an operation
     */
    void end(const std::string& operation) {
        if (!enabled_) return;

        auto end_time = std::chrono::high_resolution_clock::now();

        std::lock_guard<std::mutex> lock(mutex_);

        auto it = start_times_.find(operation);
        if (it == start_times_.end()) return;

        double elapsed_ms = std::chrono::duration<double, std::milli>(
            end_time - it->second
        ).count();

        auto& stats = stats_[operation];
        stats.operation = operation;
        stats.count++;
        stats.total_ms += elapsed_ms;

        if (stats.count == 1) {
            stats.min_ms = stats.max_ms = elapsed_ms;
        } else {
            stats.min_ms = std::min(stats.min_ms, elapsed_ms);
            stats.max_ms = std::max(stats.max_ms, elapsed_ms);
        }

        start_times_.erase(it);
    }

    /**
     * @brief RAII timer helper
     */
    class ScopedTimer {
    public:
        ScopedTimer(PipelineProfiler& profiler, const std::string& operation)
            : profiler_(profiler), operation_(operation) {
            profiler_.start(operation_);
        }
        ~ScopedTimer() {
            profiler_.end(operation_);
        }
    private:
        PipelineProfiler& profiler_;
        std::string operation_;
    };

    /**
     * @brief Get timing statistics
     */
    std::vector<TimingStats> get_stats() const {
        std::lock_guard<std::mutex> lock(mutex_);
        std::vector<TimingStats> result;
        for (const auto& [op, stats] : stats_) {
            result.push_back(stats);
        }
        return result;
    }

    /**
     * @brief Get profiling report
     */
    std::string report() const {
        std::ostringstream ss;
        ss << "Performance Profile:\n";
        ss << std::setw(30) << "Operation"
           << std::setw(10) << "Count"
           << std::setw(12) << "Total(ms)"
           << std::setw(10) << "Avg(ms)"
           << std::setw(10) << "Min(ms)"
           << std::setw(10) << "Max(ms)"
           << "\n";
        ss << std::string(82, '-') << "\n";

        std::lock_guard<std::mutex> lock(mutex_);
        for (const auto& [op, stats] : stats_) {
            ss << std::setw(30) << op
               << std::setw(10) << stats.count
               << std::setw(12) << std::fixed << std::setprecision(2) << stats.total_ms
               << std::setw(10) << stats.avg_ms()
               << std::setw(10) << stats.min_ms
               << std::setw(10) << stats.max_ms
               << "\n";
        }

        return ss.str();
    }

    /**
     * @brief Clear all statistics
     */
    void clear() {
        std::lock_guard<std::mutex> lock(mutex_);
        stats_.clear();
        start_times_.clear();
    }

private:
    bool enabled_;
    mutable std::mutex mutex_;
    std::unordered_map<std::string, TimingStats> stats_;
    std::unordered_map<std::string, std::chrono::high_resolution_clock::time_point> start_times_;
};

/**
 * @brief Global logger singleton
 */
class Logger {
public:
    static Logger& instance() {
        static Logger instance;
        return instance;
    }

    void set_level(ErrorSeverity level) { min_level_ = level; }
    ErrorSeverity get_level() const { return min_level_; }

    void enable() { enabled_ = true; }
    void disable() { enabled_ = false; }
    bool is_enabled() const { return enabled_; }

    void set_output(const std::string& path) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (!path.empty()) {
            file_ = std::make_unique<std::ofstream>(path, std::ios::app);
        } else {
            file_.reset();
        }
    }

    void log(ErrorSeverity level, const std::string& message) {
        if (!enabled_ || level < min_level_) return;

        std::lock_guard<std::mutex> lock(mutex_);

        auto now = std::chrono::system_clock::now();
        auto time_t = std::chrono::system_clock::to_time_t(now);

        std::ostringstream ss;
        ss << std::put_time(std::localtime(&time_t), "%Y-%m-%d %H:%M:%S");
        ss << " [" << severity_to_string(level) << "] " << message;

        std::string line = ss.str();

        if (file_ && file_->is_open()) {
            *file_ << line << std::endl;
        } else {
            std::cerr << "[TurboLoader] " << line << std::endl;
        }
    }

    void debug(const std::string& msg) { log(ErrorSeverity::DEBUG, msg); }
    void info(const std::string& msg) { log(ErrorSeverity::INFO, msg); }
    void warning(const std::string& msg) { log(ErrorSeverity::WARNING, msg); }
    void error(const std::string& msg) { log(ErrorSeverity::ERROR, msg); }
    void critical(const std::string& msg) { log(ErrorSeverity::CRITICAL, msg); }

private:
    Logger() : enabled_(false), min_level_(ErrorSeverity::WARNING) {}

    static const char* severity_to_string(ErrorSeverity severity) {
        switch (severity) {
            case ErrorSeverity::DEBUG: return "DEBUG";
            case ErrorSeverity::INFO: return "INFO";
            case ErrorSeverity::WARNING: return "WARNING";
            case ErrorSeverity::ERROR: return "ERROR";
            case ErrorSeverity::CRITICAL: return "CRITICAL";
            default: return "UNKNOWN";
        }
    }

    bool enabled_;
    ErrorSeverity min_level_;
    std::mutex mutex_;
    std::unique_ptr<std::ofstream> file_;
};

// Convenience macros
#define TURBO_LOG_DEBUG(msg) turboloader::pipeline::Logger::instance().debug(msg)
#define TURBO_LOG_INFO(msg) turboloader::pipeline::Logger::instance().info(msg)
#define TURBO_LOG_WARNING(msg) turboloader::pipeline::Logger::instance().warning(msg)
#define TURBO_LOG_ERROR(msg) turboloader::pipeline::Logger::instance().error(msg)
#define TURBO_LOG_CRITICAL(msg) turboloader::pipeline::Logger::instance().critical(msg)

} // namespace pipeline
} // namespace turboloader
