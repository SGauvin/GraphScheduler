#pragma once

#include <cassert>
#include <concepts>
#include <condition_variable>
#include <cstring>
#include <functional>
#include <latch>
#include <memory>
#include <mutex>
#include <type_traits>
#include <typeinfo>
#include <unordered_map>
#include <unordered_set>
#include <variant>
#include <vector>
#include "CpuNodeIo.h"
#include "Error.h"
#include "pfr.hpp"

class GraphWithoutRtti
{
};

template<typename PlaceHolder>
class GraphWithRtti
{
protected:
    std::unordered_map<PlaceHolder, std::reference_wrapper<const std::type_info>> m_placeHolderToTypeInfo;
};

// TODO: Error handling instead of asserts
// TODO: Use std::source_location
// TODO: Stop using requires and make a concept

// clang-format off
template<typename T, Hashable PlaceHolder = std::string_view, bool RttiEnabled = true, bool ErrorsEnabled = true>
requires(std::equality_comparable<PlaceHolder>)
class Graph : private std::conditional_t<RttiEnabled, GraphWithRtti<PlaceHolder>, GraphWithoutRtti>
// clang-format on
{
public:
    static_assert(!(RttiEnabled == true && ErrorsEnabled == false), "Can't have RTTI checking while disabling errors");

    using MemoryType = T;
    using Error = std::conditional_t<ErrorsEnabled, ErrorReal, ErrorFake>;

    Graph();
    ~Graph();
    Graph(const Graph&) = delete;
    Graph(Graph&&) = delete;
    Graph& operator=(const Graph&) = delete;
    Graph& operator=(Graph&&) = delete;

    template<typename Node>
    Node* createNode(std::string_view nodeName, NodeInput<Node, PlaceHolder>&& inputs, NodeOutput<Node, PlaceHolder>&& outputs);

    // clang-format off
    template<typename Node, typename... Args>
    requires(pfr::tuple_size_v<typename Node::InputType> == sizeof...(Args) && std::is_trivially_constructible_v<typename Node::InputType>)
    NodeInput<Node, PlaceHolder> createInputs(Args&&... args);
    // clang-format on

    // clang-format off
    template<typename Node, typename... Args>
    requires(pfr::tuple_size_v<typename Node::OutputType> == sizeof...(Args) && std::is_trivially_constructible_v<typename Node::OutputType>)
    NodeOutput<Node, PlaceHolder> createOutputs(Args&&... args);
    // clang-format on

    template<typename PlaceHolderUnderlyingType>
    void setPlaceHolderAddress(const PlaceHolder& placeHolder, PlaceHolderUnderlyingType* address);

    template<typename PlaceHolderUnderlyingType>
    void setPlaceHolderValue(const PlaceHolder& placeHolder, const PlaceHolderUnderlyingType& value);

    MemoryType& memory() { return m_memory; }

    void connectNodes(NodeConvertible auto& upstream, NodeConvertible auto& downstream);

    void build(std::size_t numThreads);

    template<typename Node>
    Node* getNode(std::string_view nodeName);

    void execute();

private:
    template<typename NodeIO>
    static void checkEachFieldIsDoublePtr();

    template<std::size_t N, typename StructType, typename Arg>
    void expandPointers(std::vector<const void*>& vector, std::unordered_map<std::size_t, PlaceHolder>& indexToPlaceHolder, Arg&& arg);

    template<std::size_t N, typename StructType, typename Arg, typename... Args>
    void expandPointers(std::vector<const void*>& vector, std::unordered_map<std::size_t, PlaceHolder>& indexToPlaceHolder, Arg&& arg,
                        Args&&... args);

    template<std::size_t N, typename StructType, typename... Args>
    std::pair<std::vector<const void*>, std::unordered_map<std::size_t, PlaceHolder>> expandPointers(Args&&... args);

    template<bool IsOutput, typename StructType, typename... Args>
    void parseStruct(std::vector<std::size_t>& structsBufferIndices, std::vector<PlaceHolder>& localPlaceHolders, Args&&... args);

    void connectNodes(CpuNode* upstream, CpuNode* downstream);

    CpuNode* getNode(NodeConvertible auto& nodeIdentifier);

    MemoryType m_memory;

    std::vector<std::unique_ptr<CpuNode>> m_nodes;
    std::vector<std::pair<void*, std::vector<std::size_t>>> m_IOBufferIndices;

    std::unordered_map<CpuNode*, std::vector<std::size_t>> m_nodePointerInputsIndices;
    std::unordered_map<CpuNode*, std::vector<PlaceHolder>> m_nodePlaceHolderInputs;
    std::unordered_map<CpuNode*, std::vector<std::size_t>> m_nodePointerOutputsIndices;
    std::unordered_map<CpuNode*, std::vector<PlaceHolder>> m_nodePlaceHolderOutputs;

    std::unordered_map<std::string_view, CpuNode*> m_nodeNameToPtr;

    std::unordered_set<PlaceHolder> m_outputPlaceHolders;
    std::unordered_map<PlaceHolder, std::size_t> m_placeHolderToBufferIndex;
    std::unordered_map<const void*, std::size_t> m_addressToBufferIndex;

    std::vector<std::variant<const void*, void*>> m_IOPointers;

    std::unordered_map<CpuNode*, std::atomic<std::size_t>> m_nodeUpstreamDependencyCount;
    std::unordered_map<CpuNode*, std::atomic<std::size_t>> m_nodeUpstreamDependencyCountCopy;
    std::unordered_map<CpuNode*, std::vector<CpuNode*>> m_nodeDownstreamDependencies;
    std::vector<CpuNode*> m_nodesWithoutUpstreamDependencies;

    bool m_stop = false;
    alignas(std::latch) std::array<std::byte, sizeof(std::latch)> m_latchBuffer;
    std::latch& m_latch = reinterpret_cast<std::latch&>(m_latchBuffer);
    std::vector<std::thread> m_workers;
    std::vector<CpuNode*> m_unblockedNodes;
    std::mutex m_mutex;
    std::condition_variable m_condition;
};

#include "Graph.inl"
