#pragma once
#include <cstdlib>
#include <cstring>
#include <new>

namespace softflow {

// ── Aligned memory allocation/deallocation ──────────────────────────
// Ensures 64-byte alignment for optimal cache line and SIMD performance.
// All LBM hot-path arrays should use these.

constexpr size_t CACHE_LINE = 64;

inline void* aligned_malloc(size_t size) {
    void* ptr = nullptr;
#if defined(_WIN32)
    ptr = _aligned_malloc(size, CACHE_LINE);
#else
    if (posix_memalign(&ptr, CACHE_LINE, size) != 0) ptr = nullptr;
#endif
    if (!ptr && size > 0) throw std::bad_alloc();
    return ptr;
}

inline void aligned_free(void* ptr) {
#if defined(_WIN32)
    _aligned_free(ptr);
#else
    free(ptr);
#endif
}

// RAII wrapper for aligned arrays
template<typename T>
class AlignedArray {
public:
    AlignedArray() : data_(nullptr), size_(0) {}

    explicit AlignedArray(size_t count, T init = T{})
        : size_(count)
    {
        if (count == 0) { data_ = nullptr; return; }
        data_ = static_cast<T*>(aligned_malloc(count * sizeof(T)));
        for (size_t i = 0; i < count; ++i) data_[i] = init;
    }

    ~AlignedArray() { if (data_) aligned_free(data_); }

    // Move semantics
    AlignedArray(AlignedArray&& o) noexcept : data_(o.data_), size_(o.size_) {
        o.data_ = nullptr; o.size_ = 0;
    }
    AlignedArray& operator=(AlignedArray&& o) noexcept {
        if (this != &o) {
            if (data_) aligned_free(data_);
            data_ = o.data_; size_ = o.size_;
            o.data_ = nullptr; o.size_ = 0;
        }
        return *this;
    }

    // No copy
    AlignedArray(const AlignedArray&) = delete;
    AlignedArray& operator=(const AlignedArray&) = delete;

    T*       data()       { return data_; }
    const T* data() const { return data_; }
    size_t   size() const { return size_; }

    T&       operator[](size_t i)       { return data_[i]; }
    const T& operator[](size_t i) const { return data_[i]; }

    T*       begin()       { return data_; }
    const T* begin() const { return data_; }
    T*       end()         { return data_ + size_; }
    const T* end()   const { return data_ + size_; }

    void fill(T val) {
        for (size_t i = 0; i < size_; ++i) data_[i] = val;
    }

    void resize(size_t count, T init = T{}) {
        if (count == size_) return;
        T* new_data = nullptr;
        if (count > 0) {
            new_data = static_cast<T*>(aligned_malloc(count * sizeof(T)));
            size_t copy_count = (count < size_) ? count : size_;
            if (data_ && copy_count > 0) {
                std::memcpy(new_data, data_, copy_count * sizeof(T));
            }
            for (size_t i = copy_count; i < count; ++i) new_data[i] = init;
        }
        if (data_) aligned_free(data_);
        data_ = new_data;
        size_ = count;
    }

    // Swap for pointer-swap streaming
    void swap(AlignedArray& other) noexcept {
        T* tmp = data_;       data_ = other.data_;       other.data_ = tmp;
        size_t ts = size_;    size_ = other.size_;        other.size_ = ts;
    }

private:
    T*     data_;
    size_t size_;
};

} // namespace softflow
