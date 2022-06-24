# Simple multithreaded graph scheduler

This is a simple graph scheduler that analyzes node dependencies from their IO with the help of boost::pfr, and runs them in parralel.

This is my attempt at making a [taskflow](https://github.com/taskflow/taskflow)-like scheduler.

# TODOs
1. Error handling instead of asserts
2. Use std::source_location
3. Stop using requires and make a concept
4. Better integration of the thread pool in order to make a single system call when new work is available
5. Stop using std::move_only_function as we know the type in advance and don't need to make memory allocations

# In order to compile
1. Get PFR (non-boost version)
2. Have a C++23 compiler (for std::move_only_function)
3. make release=1 run

