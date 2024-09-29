#pragma once

#include "utils/cuda_utils.h"

#include "stdlib.h"
#include <cuda_fp16.h>
#include <cuda_runtime_api.h>
#include <dirent.h>
#include <numeric>
#include <stdlib.h>
#include <string>
#include <sys/stat.h>
#include <sys/types.h>
#include <unordered_map>
#include <vector>

namespace fastertransformer {

typedef enum datatype_enum {
    TYPE_INVALID,
    TYPE_BOOL,
    TYPE_UINT8,
    TYPE_UINT16,
    TYPE_UINT32,
    TYPE_UINT64,
    TYPE_INT8,
    TYPE_INT16,
    TYPE_INT32,
    TYPE_INT64,
    TYPE_FP16,
    TYPE_FP32,
    TYPE_FP64,
    TYPE_BYTES,
    TYPE_BF16,
    TYPE_FP8_E4M3,
    TYPE_STR,
    TYPE_VOID,
} DataType;

typedef enum memorytype_enum {
    MEMORY_CPU,
    MEMORY_CPU_PINNED,
    MEMORY_GPU
} MemoryType;

struct Tensor {
    const MemoryType where;
    const DataType type;
    const std::vector<size_t> shape;
    const void *data; // TODO(bhseuh) modify from const void* to void* const
    const std::vector<size_t> offsets = std::vector<size_t>{};

    Tensor();
    Tensor(const MemoryType _where, const DataType _type, const std::vector<size_t> _shape, const void *_data);
    Tensor(const MemoryType _where,
           const DataType _type,
           const std::vector<size_t> _shape,
           const void *_data,
           const std::vector<size_t> _offset);

    size_t size() const;
    size_t sizeBytes() const;

    std::string whereToString() const;
    std::string toString() const;
    std::string getNumpyTypeDesc(DataType type) const;

    void saveNpy(const std::string &filename) const;
    static Tensor loadNpy(const std::string &npy_file, const MemoryType where);

    static DataType typeFromNumpyDesc(std::string type);
    static size_t getTypeSize(DataType type);

    template <typename T>
    inline T getVal(size_t index) const {
        FT_LOG_DEBUG("%s start", __PRETTY_FUNCTION__);
        FT_CHECK(where == MEMORY_CPU);
        FT_CHECK(data != nullptr);
        FT_CHECK_WITH_INFO(index < size(), "index is larger than buffer size");

        if (getTensorType<T>() != type) {
            FT_LOG_DEBUG("getVal with type %s, but data type is: %s",
                         getNumpyTypeDesc(getTensorType<T>()).c_str(),
                         getNumpyTypeDesc(type).c_str());
        }
        return ((T *)data)[index];
    }

    template <typename T>
    inline T getVal() const {
        FT_LOG_DEBUG("%s start", __PRETTY_FUNCTION__);
        if (getTensorType<T>() != type) {
            FT_LOG_DEBUG("getVal with type %s, but data type is: %s",
                         getNumpyTypeDesc(getTensorType<T>()).c_str(),
                         getNumpyTypeDesc(type).c_str());
        }
        return getVal<T>(0);
    }

    template <typename T>
    inline T *getPtr() const {
        FT_LOG_DEBUG("%s start", __PRETTY_FUNCTION__);
        if (getTensorType<T>() != type) {
            FT_LOG_DEBUG("getPtr with type %s, but data type is: %s",
                         getNumpyTypeDesc(getTensorType<T>()).c_str(),
                         getNumpyTypeDesc(type).c_str());
        }
        return (T *)data;
    }

    inline void *getPtrWithOffset(size_t offset) const {
        FT_LOG_DEBUG("%s start", __PRETTY_FUNCTION__);
        if (data == nullptr) {
            return (void *)data;
        } else {
            FT_CHECK_WITH_INFO(offset < size(), "offset is larger than buffer size");
            return (void *)((char *)data + offset * Tensor::getTypeSize(type));
        }
    }

    template <typename T>
    inline T *getPtrWithOffset(size_t offset) const {
        FT_LOG_DEBUG("%s start", __PRETTY_FUNCTION__);
        if (getTensorType<T>() != type) {
            FT_LOG_DEBUG("getVal with type %s, but data type is: %s",
                         getNumpyTypeDesc(getTensorType<T>()).c_str(),
                         getNumpyTypeDesc(type).c_str());
        }
        if (data == nullptr) {
            return (T *)data;
        } else {
            FT_CHECK_WITH_INFO(offset < size(),
                               fmtstr("offset (%lu) is larger than buffer size (%lu)", offset, size()));
            return ((T *)data) + offset;
        }
    }

    template <typename T>
    T max() const {
        if (getTensorType<T>() != type) {
            FT_LOG_DEBUG("getVal with type %s, but data type is: %s",
                         getNumpyTypeDesc(getTensorType<T>()).c_str(),
                         getNumpyTypeDesc(type).c_str());
        }
        FT_CHECK_WITH_INFO(shape.size() > 0 && data != nullptr, "Should be a non-empty tensor.");
        FT_CHECK_WITH_INFO(where == MEMORY_CPU || where == MEMORY_CPU_PINNED,
                           "max() supports MEMORY_CPU or MEMORY_CPU_PINNED tensor.");
        size_t max_idx = 0;
        T max_val = getVal<T>(max_idx);
        for (size_t i = 1; i < size(); ++i) {
            T val = getVal<T>(i);
            if (val > max_val) {
                max_idx = i;
                max_val = val;
            }
        }
        return max_val;
    }

    template <typename T>
    T min() const {
        if (getTensorType<T>() != type) {
            FT_LOG_DEBUG("getVal with type %s, but data type is: %s",
                         getNumpyTypeDesc(getTensorType<T>()).c_str(),
                         getNumpyTypeDesc(type).c_str());
        }
        FT_CHECK_WITH_INFO(shape.size() > 0 && data != nullptr, "Should be a non-empty tensor.");
        FT_CHECK_WITH_INFO(where == MEMORY_CPU || where == MEMORY_CPU_PINNED,
                           "min() supports MEMORY_CPU or MEMORY_CPU_PINNED tensor.");
        size_t min_idx = 0;
        T min_val = getVal<T>(min_idx);
        for (size_t i = 1; i < size(); ++i) {
            T val = getVal<T>(i);
            if (val < min_val) {
                min_idx = i;
                min_val = val;
            }
        }
        return min_val;
    }

    template <typename T>
    T any(T val) const {
        if (getTensorType<T>() != type) {
            FT_LOG_DEBUG("getVal with type %s, but data type is: %s",
                         getNumpyTypeDesc(getTensorType<T>()).c_str(),
                         getNumpyTypeDesc(type).c_str());
        }
        FT_CHECK_WITH_INFO(shape.size() > 0 && data != nullptr, "Should be a non-empty tensor.");
        FT_CHECK_WITH_INFO(where == MEMORY_CPU || where == MEMORY_CPU_PINNED,
                           "any() supports MEMORY_CPU or MEMORY_CPU_PINNED tensor.");
        for (size_t i = 0; i < size(); ++i) {
            if (getVal<T>(i) == val) {
                return true;
            }
        }
        return false;
    }

    template <typename T>
    T all(T val) const {
        if (getTensorType<T>() != type) {
            FT_LOG_DEBUG("getVal with type %s, but data type is: %s",
                         getNumpyTypeDesc(getTensorType<T>()).c_str(),
                         getNumpyTypeDesc(type).c_str());
        }
        FT_CHECK_WITH_INFO(shape.size() > 0 && data != nullptr, "Should be a non-empty tensor.");
        FT_CHECK_WITH_INFO(where == MEMORY_CPU || where == MEMORY_CPU_PINNED,
                           "all() supports MEMORY_CPU or MEMORY_CPU_PINNED tensor.");
        for (size_t i = 0; i < size(); ++i) {
            if (getVal<T>(i) != val) {
                return false;
            }
        }
        return true;
    }

    void updateShape(size_t idx, size_t val) {
        // TODO: find a better way to update the shape
        std::vector<size_t> &shape_ref = const_cast<std::vector<size_t> &>(shape);
        shape_ref[idx] = val;
    }

    Tensor slice(std::vector<size_t> shape, size_t offset = 0) const;

private:
    static void parseNpyIntro(FILE *&f_ptr, uint32_t &header_len, uint32_t &start_data);
    static int parseNpyHeader(FILE *&f_ptr, uint32_t header_len, DataType &type, std::vector<size_t> &shape);
};

} // namespace fastertransformer
