#pragma once

#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <sstream>
#include <string>

// OS-specific headers for color support
#ifdef _WIN32
    #include <windows.h>
#endif

namespace hpc::logging {

// ANSI color codes for terminal output
struct TerminalColors {
    // ANSI color code constants
    static constexpr const char* RESET     = "\033[0m";
    static constexpr const char* BLACK     = "\033[30m";
    static constexpr const char* RED       = "\033[31m";
    static constexpr const char* GREEN     = "\033[32m";
    static constexpr const char* YELLOW    = "\033[33m";
    static constexpr const char* BLUE      = "\033[34m";
    static constexpr const char* MAGENTA   = "\033[35m";
    static constexpr const char* CYAN      = "\033[36m";
    static constexpr const char* WHITE     = "\033[37m";
    static constexpr const char* BOLD      = "\033[1m";
    static constexpr const char* UNDERLINE = "\033[4m";

    // Background colors
    static constexpr const char* BG_BLACK   = "\033[40m";
    static constexpr const char* BG_RED     = "\033[41m";
    static constexpr const char* BG_GREEN   = "\033[42m";
    static constexpr const char* BG_YELLOW  = "\033[43m";
    static constexpr const char* BG_BLUE    = "\033[44m";
    static constexpr const char* BG_MAGENTA = "\033[45m";
    static constexpr const char* BG_CYAN    = "\033[46m";
    static constexpr const char* BG_WHITE   = "\033[47m";

    // Enable Windows console color support
    static void setupConsole() {
#ifdef _WIN32
        // Enable ANSI escape sequences in Windows console
        HANDLE hOut   = GetStdHandle(STD_OUTPUT_HANDLE);
        DWORD  dwMode = 0;
        GetConsoleMode(hOut, &dwMode);
        dwMode |= 0x0004; // ENABLE_VIRTUAL_TERMINAL_PROCESSING
        SetConsoleMode(hOut, dwMode);
#endif
    }
};

// Logging levels
enum class LogLevel {
    None,  // No logging
    Error, // Only errors
    Warn,  // Warnings and errors
    Info,  // General information plus warnings and errors
    Debug, // Detailed debug information
    Trace  // Most verbose level
};

class OptimizationLogger {
public:
    // Initialize logger with default values
    static void init(LogLevel           level          = LogLevel::Info,
                     bool               enable_console = true,
                     const std::string& log_file       = "") {
        instance().level_          = level;
        instance().enable_console_ = enable_console;

        // Setup console colors if using console output
        if (enable_console) {
            TerminalColors::setupConsole();
        }

        if (!log_file.empty()) {
            instance().log_file_.open(log_file, std::ios::out | std::ios::app);
            instance().enable_file_ = instance().log_file_.is_open();
        }

        // Log initialization message
        if (enable_console || instance().enable_file_) {
            log(LogLevel::Info, "Logging system initialized: level=", levelToString(level));
        }
    }

    // Set logging level
    static void setLevel(LogLevel level) {
        instance().level_ = level;
        log(LogLevel::Info, "Log level changed to ", levelToString(level));
    }

    // Enable/disable optimization logging
    static void enableOptimizationLogging(bool enable) {
        instance().optimization_logging_enabled_ = enable;
        log(LogLevel::Info, "Optimization logging ", enable ? "enabled" : "disabled");
    }

    // Log optimization usage with colored output
    static void logOptimization(const std::string& component,
                                const std::string& optimization,
                                const std::string& details = "") {
        if (!instance().optimization_logging_enabled_ || instance().level_ < LogLevel::Info) {
            return;
        }

        std::stringstream ss;
        // Use magenta color for optimization logs
        ss << TerminalColors::MAGENTA << TerminalColors::BOLD << "[OPTIMIZATION]"
           << TerminalColors::RESET << " ";
        ss << TerminalColors::CYAN << component << TerminalColors::RESET << ": Using ";
        ss << TerminalColors::GREEN << TerminalColors::BOLD << optimization
           << TerminalColors::RESET;

        if (!details.empty()) {
            ss << " - " << details;
        }

        log(LogLevel::Info, ss.str());
    }

    // Log performance measurement with colored output
    static void logPerformance(const std::string& operation,
                               double             time_ms,
                               const std::string& details = "") {
        if (!instance().optimization_logging_enabled_ || instance().level_ < LogLevel::Info) {
            return;
        }

        std::stringstream ss;
        // Use blue color for performance logs
        ss << TerminalColors::BLUE << TerminalColors::BOLD << "[PERFORMANCE]"
           << TerminalColors::RESET << " ";
        ss << TerminalColors::CYAN << operation << TerminalColors::RESET << ": ";
        ss << TerminalColors::YELLOW << std::fixed << std::setprecision(3) << time_ms << " ms"
           << TerminalColors::RESET;

        if (!details.empty()) {
            ss << " - " << details;
        }

        log(LogLevel::Info, ss.str());
    }

    // General logging function
    template <typename... Args>
    static void log(LogLevel level, const Args&... args) {
        if (level > instance().level_) {
            return;
        }

        std::lock_guard<std::mutex> lock(instance().mutex_);

        std::stringstream ss;
        std::string       level_color;
        std::string       level_prefix;

        // Add timestamp
        auto now        = std::chrono::system_clock::now();
        auto now_time_t = std::chrono::system_clock::to_time_t(now);
        auto now_ms =
            std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()) % 1000;

        ss << TerminalColors::WHITE
           << std::put_time(std::localtime(&now_time_t), "%Y-%m-%d %H:%M:%S") << '.'
           << std::setfill('0') << std::setw(3) << now_ms.count() << TerminalColors::RESET << " ";

        // Add colored log level prefix
        switch (level) {
        case LogLevel::Error:
            level_color  = TerminalColors::RED;
            level_prefix = "[ERROR]";
            break;
        case LogLevel::Warn:
            level_color  = TerminalColors::YELLOW;
            level_prefix = "[WARN]";
            break;
        case LogLevel::Info:
            level_color  = TerminalColors::GREEN;
            level_prefix = "[INFO]";
            break;
        case LogLevel::Debug:
            level_color  = TerminalColors::CYAN;
            level_prefix = "[DEBUG]";
            break;
        case LogLevel::Trace:
            level_color  = TerminalColors::WHITE;
            level_prefix = "[TRACE]";
            break;
        default:
            level_color  = "";
            level_prefix = "";
            break;
        }

        ss << level_color << TerminalColors::BOLD << level_prefix << TerminalColors::RESET << " ";

        // Fold arguments into stream
        std::stringstream content;
        (content << ... << args);
        ss << content.str();

        // Plain content without color codes for file output
        std::stringstream time_ss;
        time_ss << std::put_time(std::localtime(&now_time_t), "%Y-%m-%d %H:%M:%S") << '.'
                << std::setfill('0') << std::setw(3) << now_ms.count();

        std::string plain_content =
            removeColorCodes(time_ss.str() + " " + level_prefix + " " + content.str());

        // Output to console if enabled
        if (instance().enable_console_) {
            std::cout << ss.str() << std::endl;
        }

        // Output to file if enabled (without color codes)
        if (instance().enable_file_ && instance().log_file_.is_open()) {
            instance().log_file_ << plain_content << std::endl;
            instance().log_file_.flush();
        }
    }

    // Performance timer class for measuring execution time
    class Timer {
    public:
        Timer(const std::string& operation, const std::string& details = "")
            : operation_(operation), details_(details),
              start_(std::chrono::high_resolution_clock::now()) {}

        ~Timer() {
            auto end = std::chrono::high_resolution_clock::now();
            auto duration =
                std::chrono::duration_cast<std::chrono::microseconds>(end - start_).count() /
                1000.0;
            logPerformance(operation_, duration, details_);
        }

    private:
        std::string                                                 operation_;
        std::string                                                 details_;
        std::chrono::time_point<std::chrono::high_resolution_clock> start_;
    };

private:
    // Helper method to convert log level to string
    static std::string levelToString(LogLevel level) {
        switch (level) {
        case LogLevel::None:
            return "None";
        case LogLevel::Error:
            return "Error";
        case LogLevel::Warn:
            return "Warning";
        case LogLevel::Info:
            return "Info";
        case LogLevel::Debug:
            return "Debug";
        case LogLevel::Trace:
            return "Trace";
        default:
            return "Unknown";
        }
    }

    // Helper method to remove ANSI color codes from strings for file output
    static std::string removeColorCodes(const std::string& input) {
        std::string result;
        bool        in_escape_sequence = false;

        for (char c : input) {
            if (c == '\033') {
                in_escape_sequence = true;
                continue;
            }

            if (in_escape_sequence) {
                if (c == 'm') {
                    in_escape_sequence = false;
                }
                continue;
            }

            result += c;
        }

        return result;
    }

    // Singleton implementation
    static OptimizationLogger& instance() {
        static OptimizationLogger instance;
        return instance;
    }

    OptimizationLogger()
        : level_(LogLevel::None), enable_console_(false), enable_file_(false),
          optimization_logging_enabled_(false) {}

    ~OptimizationLogger() {
        if (log_file_.is_open()) {
            log_file_.close();
        }
    }

    std::mutex    mutex_;                        // Mutex for thread safety
    LogLevel      level_;                        // Current log level
    bool          enable_console_;               // Whether to output to console
    bool          enable_file_;                  // Whether to output to file
    std::ofstream log_file_;                     // Log file stream
    bool          optimization_logging_enabled_; // Flag to enable/disable optimization logging
};

// Convenience macros for logging - with full namespace qualification
#define LOG_ERROR(...)                                                                             \
    ::hpc::logging::OptimizationLogger::log(::hpc::logging::LogLevel::Error, __VA_ARGS__)
#define LOG_WARN(...)                                                                              \
    ::hpc::logging::OptimizationLogger::log(::hpc::logging::LogLevel::Warn, __VA_ARGS__)
#define LOG_INFO(...)                                                                              \
    ::hpc::logging::OptimizationLogger::log(::hpc::logging::LogLevel::Info, __VA_ARGS__)
#define LOG_DEBUG(...)                                                                             \
    ::hpc::logging::OptimizationLogger::log(::hpc::logging::LogLevel::Debug, __VA_ARGS__)
#define LOG_TRACE(...)                                                                             \
    ::hpc::logging::OptimizationLogger::log(::hpc::logging::LogLevel::Trace, __VA_ARGS__)

// Optimization logging macros
#define LOG_OPTIMIZATION(component, optimization, details)                                         \
    ::hpc::logging::OptimizationLogger::logOptimization(component, optimization, details)

// Performance timing macro
#define TIME_OPERATION(operation, details)                                                         \
    ::hpc::logging::OptimizationLogger::Timer timer##__LINE__(operation, details)

} // namespace hpc::logging