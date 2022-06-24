#pragma once

#include <condition_variable>
#include <functional>
#include <future>
#include <memory>
#include <mutex>
#include <queue>
#include <thread>
#include <vector>

class ThreadPool
{
public:
    ThreadPool() = default;

    void start(std::size_t threadCount);

    template<typename F, typename... Args>
    std::future<typename std::result_of_t<F(Args...)>> enqueue(F&& f, Args&&... args);

    ~ThreadPool();

private:
    std::vector<std::thread> m_workers;
    std::queue<std::move_only_function<void()>> m_taskQueue;

    std::mutex m_queueMutex;
    std::condition_variable m_condition;
    bool m_stop = false;
};

void ThreadPool::start(std::size_t threadCount)
{
    m_workers.reserve(threadCount);
    for (std::size_t i = 0; i < threadCount; i++)
    {
        m_workers.emplace_back(
            [this]
            {
                while (true)
                {
                    std::move_only_function<void()> task;
                    {
                        std::unique_lock<std::mutex> lock(m_queueMutex);
                        m_condition.wait(lock, [this] { return m_stop || !m_taskQueue.empty(); });
                        if (m_stop && m_taskQueue.empty())
                        {
                            return;
                        }
                        task = std::move(m_taskQueue.front());
                        m_taskQueue.pop();
                    }

                    task();
                }
            });
    }
}

template<class F, class... Args>
std::future<typename std::result_of_t<F(Args...)>> ThreadPool::enqueue(F&& f, Args&&... args)
{
    using ReturnType = typename std::result_of_t<F(Args...)>;

    auto task = std::make_unique<std::packaged_task<ReturnType()>>(std::bind(std::forward<F>(f), std::forward<Args>(args)...));

    std::future<ReturnType> future = task->get_future();
    {
        std::unique_lock<std::mutex> lock(m_queueMutex);
        m_taskQueue.emplace([task = std::move(task)]() { (*task)(); });
    }
    m_condition.notify_one();
    return future;
}

ThreadPool::~ThreadPool()
{
    {
        std::unique_lock<std::mutex> lock(m_queueMutex);
        m_stop = true;
    }
    m_condition.notify_all();

    for (auto& worker : m_workers)
    {
        worker.join();
    }
}
