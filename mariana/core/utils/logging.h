/*
 *        (C) COPYRIGHT LeiNao Limited.
 *             ALL RIGHTS RESERVED
 *
 * File       : core/utils/logging.h
 * Authors    : wangliquan@zkln
 * Create Time: 2022-12-01:09:23:49
 * Description:
 *
 */

#ifndef __CORE_UTILS_LOGGING_H__
#define __CORE_UTILS_LOGGING_H__

#include <glog/logging.h>
#include <core/utils/sys.h>

namespace mariana {

    extern const char LOG_DIR[];

    class LoggerBase {
    public:    
        LoggerBase() {
            create_folders(LOG_DIR);
        }
        virtual void init()=0;
        virtual ~LoggerBase()=default;
    };

    class GolgLogger final : public LoggerBase {
    public:
    GolgLogger() : LoggerBase() {
            init();
        }
        void init() override {
            FLAGS_log_dir = LOG_DIR;
            FLAGS_colorlogtostderr = true;
            google::SetLogFilenameExtension(".log");
            google::InitGoogleLogging("mariana");
            google::EnableLogCleaner(3); // Keep the log alive 3 days.
        }
        ~GolgLogger() {
            google::ShutdownGoogleLogging();
        }
    };

}
using google::WARNING;
using google::ERROR;
using google::FATAL;
using google::INFO;
#define MLOG(severity)                          \
    LOG(severity)

#define MVLOG(verboselevel)                     \
    VLOG(verboselevel)

#define MCHECK(condition)                       \
    CHECK(condition)

#define MLOG_IF(severity, condition)            \
    LOG_IF(severity, condition)

#define MLOG_EVERY_N(severity, n)               \
    LOG_EVERY_N(severity, n)

#define MLOG_IF_EVERY_N(severity, condition, n) \
    LOG_IF_EVERY_N(severity, condition, n)

#define MLOG_EVERY_T(severity, T)               \
    LOG_EVERY_T(severity, T)

#define MCHECK(condition)                       \
    CHECK(condition)

#define MCHECK_EQ(val1, val2)                   \
    CHECK_EQ(val1, val2)
#define MCHECK_NE(val1, val2)                   \
    CHECK_NE(val1, val2)
#define MCHECK_LE(val1, val2)                   \
    CHECK_LE(val1, val2)
#define MCHECK_LT(val1, val2)                   \
    CHECK_LT(val1, val2)
#define MCHECK_GE(val1, val2)                   \
    CHECK_GE(val1, val2)
#define MCHECK_GT(val1, val2)                   \
    CHECK_GT(val1, val2)

#define MCHECK_NOTNULL(val)                     \
    CHECK_NOTNULL(val)

#define MCHECK_STREQ(s1, s2)                    \
    CHECK_STREQ(s1, s2)

#define MCHECK_STRNE(s1, s2)                    \
    CHECK_STRNE(s1, s2)

#define MCHECK_STRCASEEQ(s1, s2)                \
    CHECK_STRCASEEQ(s1, s2)

#define MCHECK_STRCASENE(s1, s2)                \
    CHECK_STRCASENE(s1, s2)
    
#define MCHECK_INDEX(I,A)                       \
    CHECK_INDEX(I,A)

#define MCHECK_BOUND(B,A)                       \
    CHECK_BOUND(B,A)

#define MCHECK_DOUBLE_EQ(val1, val2)            \
    CHECK_DOUBLE_EQ(val1, val2)

#define MCHECK_NEAR(val1, val2, margin)         \
    CHECK_NEAR(val1, val2, margin)

#endif /* __CORE_UTILS_LOGGING_H__ */

