//
// Created by DevAccount on 30/04/2025.
//

#include <memory/shortTermMemory.h>


shortTermMemory::shortTermMemory() : memory(MEMORY_SIZE, 0) {}

bool shortTermMemory::write(size_t offset, const void* data, size_t size) {
    if (offset + size > MEMORY_SIZE) {
        return false;
    }

    std::lock_guard<std::mutex> lock(memoryMutex);
    const char* src = static_cast<const char*>(data);
    std::copy(src, src + size, memory.begin() + offset);
    return true;
}

size_t shortTermMemory::read(size_t offset, void* buffer, size_t size) const {
    if (offset >= MEMORY_SIZE) {
        return 0;
    }

    std::lock_guard<std::mutex> lock(memoryMutex);
    size_t readable = std::min(size, MEMORY_SIZE - offset);
    char* dest = static_cast<char*>(buffer);
    std::copy(memory.begin() + offset, memory.begin() + offset + readable, dest);
    return readable;
}

void shortTermMemory::clear() {
    std::lock_guard<std::mutex> lock(memoryMutex);
    std::fill(memory.begin(), memory.end(), 0);
}