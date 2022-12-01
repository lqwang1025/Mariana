/*
 *        (C) COPYRIGHT Daniel Limited.
 *             ALL RIGHTS RESERVED
 *
 * File       : core/utils/arrary_ref.h
 * Authors    : wangliquan@zkln
 * Create Time: 2022-12-01:11:58:32
 * Description:
 *
 */

#ifndef __CORE_UTILS_ARRARY_REF_H__
#define __CORE_UTILS_ARRARY_REF_H__

#include <array>
#include <vector>
#include <iterator>

#include <core/utils/logging.h>

namespace mariana {

template <typename T>
class ArrayRef final {
public:
    using iterator = const T*;
    using const_iterator = const T*;
    using value_type = T;
    using reverse_iterator = std::reverse_iterator<iterator>;
private:
    const T* data_;
    size_t length_;
    void _check() {
        MLOG_IF(WARNING, data_==nullptr || length_==0)<<
            "Created ArrayRef with nullptr and non-zero length!";
    }
public:
    ArrayRef() : data_(nullptr), length_(0) {}
    constexpr ArrayRef(const T& OneElt) : data_(&OneElt), length_(1) {}
    ArrayRef(const T* data, size_t length) : data_(data), length_(length) {
        _check();
    }
    ArrayRef(const T* begin, const T* end) : data_(begin), length_(end-begin) {
        _check();
    }
    template <typename A>
        ArrayRef(const std::vector<T, A>& vec)
        : data_(vec.data()), length_(vec.size()) {
        static_assert(!std::is_same<T, bool>::value, "ArrayRef<bool> cannot be "
                      "constructed from a std::vector<bool> bitfield.");
    }
    template <size_t N>
        constexpr ArrayRef(const std::array<T, N>& Arr) : data_(Arr.data()), length_(N) {}
    template <size_t N>
        constexpr ArrayRef(const T (&Arr)[N]) : data_(Arr), length_(N) {}
    constexpr ArrayRef(const std::initializer_list<T>& Vec)
        : data_(std::begin(Vec) == std::end(Vec) ? static_cast<T*>(nullptr)
                : std::begin(Vec)), length_(Vec.size()) {}

    constexpr iterator begin() const {
        return data_;
    }
    
    constexpr iterator end() const {
        return data_+length_;
    }

    constexpr const_iterator cbegin() const {
        return data_;
    }
    
    constexpr const_iterator cend() const {
        return data_+length_;
    }

    constexpr reverse_iterator rbegin() const {
        return reverse_iterator(end());
    }
    
    constexpr reverse_iterator rend() const {
        return reverse_iterator(begin());
    }
    
    constexpr bool empty() const {
        return length_==0;
    }
    
    constexpr const T* data() const {
        return data_;
    }

    /// size - Get the array size.
    constexpr size_t size() const {
        return length_;
    }

    /// front - Get the first element.
    constexpr const T& front() const {
        MCHECK(!empty())<<"ArrayRef: attempted to access front() of empty list";
        return data_[0];
    }

    /// back - Get the last element.
    constexpr const T& back() const {
        MCHECK(!empty())<<"ArrayRef: attempted to access front() of empty list";
        return data_[length_ - 1];
    }

    /// equals - Check for element-wise equality.
    constexpr bool equals(ArrayRef RHS) const {
        return length_ == RHS.length_ && std::equal(begin(), end(), RHS.begin());
    }

    /// slice(n, m) - Take M elements of the array starting at element N
    constexpr ArrayRef<T> slice(size_t N, size_t M) const {
        MCHECK(N+M<=size())<<"ArrayRef: invalid slice, N = "
                           <<N<<"; M = "<<M<<"; size = "<<size();
        return ArrayRef<T>(data() + N, M);
    }

    /// slice(n) - Chop off the first N elements of the array.
    constexpr ArrayRef<T> slice(size_t N) const {
        return slice(N, size() - N);
    }

    constexpr const T& operator[](size_t Index) const {
        return data_[Index];
    }

    /// Vector compatibility
    constexpr const T& at(size_t Index) const {
        MCHECK(Index<length_)<<"ArrayRef: invalid index Index = "
                             <<Index<<"; Length = "<<length_;
        return data[Index];
    }

    /// Disallow accidental assignment from a temporary.
    ///
    /// The declaration here is extra complicated so that "arrayRef = {}"
    /// continues to select the move assignment operator.
    template <typename U>
        typename std::enable_if<std::is_same<U, T>::value, ArrayRef<T>>::type&
        operator=(U&& Temporary) = delete;

    template <typename U>
        typename std::enable_if<std::is_same<U, T>::value, ArrayRef<T>>::type&
        operator=(std::initializer_list<U>) = delete;

    std::vector<T> vec() const {
        return std::vector<T>(data_, data_ + length_);
    }
};

template <typename T>
std::ostream& operator<<(std::ostream& out, ArrayRef<T> list) {
    int i = 0;
    out << "[";
    for (auto e : list) {
        if (i++ > 0)
            out << ", ";
        out << e;
    }
    out << "]";
    return out;
}

/// Construct an ArrayRef from a single element.
template <typename T>
ArrayRef<T> make_arrayref(const T& OneElt) {
    return OneElt;
}

/// Construct an ArrayRef from a pointer and length.
template <typename T>
ArrayRef<T> make_arrayref(const T* data, size_t length) {
    return ArrayRef<T>(data, length);
}

/// Construct an ArrayRef from a range.
template <typename T>
ArrayRef<T> make_arrayref(const T* begin, const T* end) {
    return ArrayRef<T>(begin, end);
}

/// Construct an ArrayRef from a std::vector.
template <typename T>
ArrayRef<T> make_arrayref(const std::vector<T>& Vec) {
    return Vec;
}

/// Construct an ArrayRef from a std::array.
template <typename T, std::size_t N>
ArrayRef<T> make_arrayref(const std::array<T, N>& Arr) {
    return Arr;
}

/// Construct an ArrayRef from an ArrayRef (no-op) (const)
template <typename T>
ArrayRef<T> make_arrayref(const ArrayRef<T>& Vec) {
    return Vec;
}

/// Construct an ArrayRef from an ArrayRef (no-op)
template <typename T>
ArrayRef<T>& make_arrayref(ArrayRef<T>& Vec) {
    return Vec;
}

/// Construct an ArrayRef from a C array.
template <typename T, size_t N>
ArrayRef<T> make_arrayref(const T (&Arr)[N]) {
    return ArrayRef<T>(Arr);
}

// WARNING: Template instantiation will NOT be willing to do an implicit
// conversions to get you to an ArrayRef, which is why we need so
// many overloads.
template <typename T>
bool operator==(ArrayRef<T> a1, ArrayRef<T> a2) {
    return a1.equals(a2);
}

template <typename T>
bool operator!=(ArrayRef<T> a1, ArrayRef<T> a2) {
    return !a1.equals(a2);
}

template <typename T>
bool operator==(const std::vector<T>& a1, ArrayRef<T> a2) {
    return ArrayRef<T>(a1).equals(a2);
}

template <typename T>
bool operator!=(const std::vector<T>& a1, ArrayRef<T> a2) {
    return !ArrayRef<T>(a1).equals(a2);
}

template <typename T>
bool operator==(ArrayRef<T> a1, const std::vector<T>& a2) {
    return a1.equals(ArrayRef<T>(a2));
}

template <typename T>
bool operator!=(ArrayRef<T> a1, const std::vector<T>& a2) {
    return !a1.equals(ArrayRef<T>(a2));
}

using IntArrayRef = ArrayRef<int64_t>;

} // namespace mariana

#endif /* __CORE_UTILS_ARRARY_REF_H__ */

