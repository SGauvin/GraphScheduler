# Simple multithreaded graph scheduler

This is a simple graph scheduler that analyzes node dependencies from their IO with the help of boost::pfr, and runs them in parralel.

This is my attempt at making a [taskflow](https://github.com/taskflow/taskflow)-like scheduler.

# TODOs
1. Error handling instead of asserts
2. Use std::source_location
3. Stop using requires and make a concept

# In order to compile
1. Have a C++20 compiler
2. make release=1 run
