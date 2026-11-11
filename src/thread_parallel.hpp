#ifndef PSUM_THREAD_PARALLEL_HPP
#define PSUM_THREAD_PARALLEL_HPP

#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <deque>
#include <functional>
#include <mutex>
#include <thread>
#include <vector>

namespace psum {

namespace thread_parallel {

    constexpr unsigned max_thread_count = 16;
    constexpr unsigned fallback_thread_count = 4;
    constexpr size_t min_parallel_grain = 16384;

    inline unsigned default_thread_count() {
        const unsigned hw = std::thread::hardware_concurrency();
        return hw ? std::min(hw, max_thread_count) : fallback_thread_count;
    }

    // A positive requested count asks for an exact thread count, otherwise default.
    // Small workloads shrink to one thread on the caller.
    inline unsigned resolve_thread_count(int requested, size_t n) {
        int count = requested > 0 ? requested : static_cast<int>(default_thread_count());
        count = std::min<int>(count, static_cast<int>((n + min_parallel_grain - 1) / min_parallel_grain));
        return static_cast<unsigned>(std::max(1, count));
    }

    // Resident thread pool: workers are spawned once, blocks are dispatched
    // through a task queue with a completion latch. Nested dispatches from
    // inside a pool worker fall back to the calling thread.
    class thread_pool {
    private:
        std::vector<std::thread> workers_;
        std::deque<std::function<void()>> tasks_;
        std::mutex mutex_;
        std::condition_variable task_cv_;
        std::condition_variable done_cv_;
        std::atomic<unsigned> pending_{0};
        bool stop_ = false;
        static inline thread_local bool pool_thread_ = false;

        void worker_entry_() {
            pool_thread_ = true;
            for (;;) {
                std::function<void()> task;
                {
                    std::unique_lock<std::mutex> lock(mutex_);
                    task_cv_.wait(lock, [this] { return stop_ || !tasks_.empty(); });
                    if (stop_ && tasks_.empty())
                        return;
                    task = std::move(tasks_.front());
                    tasks_.pop_front();
                }
                task();
                if (pending_.fetch_sub(1u) == 1u) {
                    std::lock_guard<std::mutex> lock(mutex_);
                    done_cv_.notify_all();
                }
            }
        }

        explicit thread_pool(unsigned n) {
            workers_.reserve(n);
            for (unsigned i = 0; i < n; ++i)
                workers_.emplace_back(&thread_pool::worker_entry_, this);
        }

        ~thread_pool() {
            {
                std::lock_guard<std::mutex> lock(mutex_);
                stop_ = true;
            }
            task_cv_.notify_all();
            for (auto& worker : workers_)
                worker.join();
        }

        thread_pool(const thread_pool&) = delete;
        thread_pool& operator=(const thread_pool&) = delete;

    public:
        static thread_pool& instance() {
            static thread_pool pool(default_thread_count());
            return pool;
        }

        static bool in_pool_thread() {
            return pool_thread_;
        }

        // Submit n_chunks blocks of [lo, hi); block 0 runs on the caller,
        // the rest are queued; returns when all blocks are complete.
        template <class F>
        void run_chunks(size_t n, unsigned n_chunks, F&& chunk) {
            if (n_chunks <= 1 || pool_thread_) {
                chunk(0u, 0, n);
            } else {
                {
                    std::lock_guard<std::mutex> lock(mutex_);
                    pending_.store(n_chunks - 1u);
                    for (unsigned t = 1; t < n_chunks; ++t) {
                        const size_t lo = n * t / n_chunks;
                        const size_t hi = n * (t + 1) / n_chunks;
                        tasks_.emplace_back([&chunk, t, lo, hi] { chunk(t, lo, hi); });
                    }
                }
                task_cv_.notify_all();
                chunk(0u, 0, n / n_chunks);
                std::unique_lock<std::mutex> lock(mutex_);
                done_cv_.wait(lock, [this] { return pending_.load() == 0u; });
            }
        }
    };

    // Apply f(size_t i) to each element; the loop over a block stays inside
    // this layer so callers never deal with lo/hi ranges.
    template <class Fn>
    inline void parallel_for(size_t n, int requested, Fn&& f) {
        const unsigned n_threads = resolve_thread_count(requested, n);
        if (n_threads <= 1 || thread_pool::in_pool_thread()) {
            for (size_t i = 0; i < n; ++i)
                f(i);
        } else {
            thread_pool::instance().run_chunks(
                n, n_threads, [&](unsigned, size_t lo, size_t hi) {
                    for (size_t i = lo; i < hi; ++i)
                        f(i);
                }
            );
        }
    }

    // Apply f(unsigned tid, size_t i) to each element; tid identifies the
    // executing thread so callers can keep per-thread scratch buffers.
    template <class Fn>
    inline void parallel_for_tid(size_t n, int requested, Fn&& f) {
        const unsigned n_threads = resolve_thread_count(requested, n);
        if (n_threads <= 1 || thread_pool::in_pool_thread()) {
            for (size_t i = 0; i < n; ++i)
                f(0u, i);
        } else {
            thread_pool::instance().run_chunks(
                n, n_threads, [&](unsigned tid, size_t lo, size_t hi) {
                    for (size_t i = lo; i < hi; ++i)
                        f(tid, i);
                }
            );
        }
    }

}

}

#endif
