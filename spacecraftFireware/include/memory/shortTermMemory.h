//
// Created by DevAccount on 30/04/2025.
//

#ifndef SHORT_TERM_MEMORY_H
#define SHORT_TERM_MEMORY_H

#include <vector>
#include <cstdint>
#include <cstring>
#include <cstddef>
#include <vector>
#include <mutex>

class shortTermMemory {
public:
    explicit shortTermMemory();  // Default constructor
    bool write(size_t offset, const void* data, size_t size);
    size_t read(size_t offset, void* buffer, size_t size) const;
    void clear();

private:
    static constexpr size_t MEMORY_SIZE = 4096;
    std::vector<char> memory;
    mutable std::mutex memoryMutex;
};
#endif
