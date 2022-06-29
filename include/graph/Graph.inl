template<typename T, Hashable P, bool R, bool E>
requires(std::equality_comparable<P>) Graph<T, P, R, E>::Graph()
{
    m_IOPointers.reserve(sizeof(T) / sizeof(void**));
}

template<typename T, Hashable P, bool R, bool E>
requires(std::equality_comparable<P>) Graph<T, P, R, E>::~Graph()
{
    {
        std::unique_lock<std::mutex> lock(m_mutex);
        m_stop = true;
    }
    m_condition.notify_all();

    for (auto& worker : m_workers)
    {
        worker.join();
    }
}

template<typename T, Hashable P, bool R, bool E>
requires(std::equality_comparable<P>) template<typename Node>
Node* Graph<T, P, R, E>::createNode(std::string_view nodeName, NodeInput<Node, P>&& inputs, NodeOutput<Node, P>&& outputs)
{
    assert(!m_nodeNameToPtr.contains(nodeName));
    // Todo add requirements so that node can be constructed this way
    auto node = std::make_unique<Node>(nodeName, std::move(inputs), std::move(outputs));
    auto returnNode = node.get();
    m_nodes.push_back(std::move(node));
    m_nodeNameToPtr.emplace(nodeName, returnNode);

    m_nodePointerInputsIndices.emplace(returnNode, inputs.getBufferIndices());
    m_nodePlaceHolderInputs.emplace(returnNode, inputs.getPlaceHolders());
    m_nodePointerOutputsIndices.emplace(returnNode, outputs.getBufferIndices());
    m_nodePlaceHolderOutputs.emplace(returnNode, outputs.getPlaceHolders());

    m_IOBufferIndices.push_back({&returnNode->m_inputs, inputs.getBufferIndices()});
    m_IOBufferIndices.push_back({&returnNode->m_outputs, outputs.getBufferIndices()});

    return returnNode;
}

template<typename T, Hashable P, bool R, bool E>
requires(std::equality_comparable<P>) template<typename Node, typename... Args>
requires(pfr::tuple_size_v<typename Node::InputType> == sizeof...(Args) &&
         std::is_trivially_constructible_v<typename Node::InputType>) NodeInput<Node, P> Graph<T, P, R, E>::createInputs(Args&&... args)
{
    using InputType = typename Node::InputType;
    checkEachFieldIsDoublePtr<InputType>(); // static_assert if one of the types isn't a double pointer

    // static_assert if one of the types isn't const
    pfr::for_each_field(InputType{},
                        [](auto& field)
                        {
                            using FieldTypeDoublePtr = std::remove_reference_t<decltype(field)>;
                            using FieldTypePtr = std::remove_pointer_t<FieldTypeDoublePtr>;
                            using FieldType = std::remove_pointer_t<FieldTypePtr>;
                            static_assert(std::is_const_v<FieldType>, "Every field of an Input struct must be const");
                        });

    std::vector<std::size_t> inputBufferIndices;
    std::vector<P> placeHolderInputs;
    parseStruct<false, InputType>(inputBufferIndices, placeHolderInputs, std::forward<Args>(args)...);
    return NodeInput<Node, P>(inputBufferIndices, placeHolderInputs);
}

template<typename T, Hashable P, bool R, bool E>
requires(std::equality_comparable<P>) template<typename Node, typename... Args>
requires(pfr::tuple_size_v<typename Node::OutputType> == sizeof...(Args) &&
         std::is_trivially_constructible_v<typename Node::OutputType>) NodeOutput<Node, P> Graph<T, P, R, E>::createOutputs(Args&&... args)
{
    using OutputType = typename Node::OutputType;
    checkEachFieldIsDoublePtr<OutputType>(); // static_assert if one of the types isn't a double pointer

    // static_assert if one of the types is const
    pfr::for_each_field(OutputType{},
                        [](auto& field)
                        {
                            using FieldTypeDoublePtr = std::remove_reference_t<decltype(field)>;
                            using FieldTypePtr = std::remove_pointer_t<FieldTypeDoublePtr>;
                            using FieldType = std::remove_pointer_t<FieldTypePtr>;
                            static_assert(!std::is_const_v<FieldType>, "Every field of an Output struct must be non-const");
                        });

    std::vector<std::size_t> outputBufferIndices;
    std::vector<P> placeHolderOutputs;
    parseStruct<true, OutputType>(outputBufferIndices, placeHolderOutputs, std::forward<Args>(args)...);
    return NodeOutput<Node, P>(outputBufferIndices, placeHolderOutputs);
}

template<typename T, Hashable P, bool R, bool E>
requires(std::equality_comparable<P>) template<typename PlaceHolderUnderlyingType>
void Graph<T, P, R, E>::setPlaceHolderAddress(const P& placeHolder, PlaceHolderUnderlyingType* address)
{
    auto it = m_placeHolderToBufferIndex.find(placeHolder);
    assert(it != m_placeHolderToBufferIndex.cend());

    if constexpr (R)
    {
        if (this->m_placeHolderToTypeInfo.at(placeHolder) != typeid(PlaceHolderUnderlyingType))
        {
            std::cout << "Error while setting placeholder's address. Type mismatch!\n"; // TODO add std::source_location
            assert(false);
        }
    }

    if constexpr (std::is_const_v<PlaceHolderUnderlyingType>)
    {
        if (m_outputPlaceHolders.contains(placeHolder))
        {
            std::cout << "Can't assign a const pointer to a placeholder of an output.\n";
            assert(false);
        }
        m_IOPointers[it->second].template emplace<0>(address);
    }
    else
    {
        m_IOPointers[it->second].template emplace<1>(address);
    }
}

template<typename T, Hashable P, bool R, bool E>
requires(std::equality_comparable<P>) template<typename PlaceHolderUnderlyingType>
void Graph<T, P, R, E>::setPlaceHolderValue(const P& placeHolder, const PlaceHolderUnderlyingType& value)
{
    auto it = m_placeHolderToBufferIndex.find(placeHolder);
    assert(it != m_placeHolderToBufferIndex.cend());

    if constexpr (R)
    {
        if (this->m_placeHolderToTypeInfo.at(placeHolder) != typeid(PlaceHolderUnderlyingType))
        {
            std::cout << "Error while setting placeholder's value. Type mismatch!\n";
            assert(false);
        }
    }

    if (!std::holds_alternative<void*>(m_IOPointers[it->second]))
    {
        std::cout << "Can't assign the value to a non const pointer.\n";
        assert(false);
    }
    *reinterpret_cast<PlaceHolderUnderlyingType*>(std::get<void*>(m_IOPointers[it->second])) = value;
}

template<typename T, Hashable P, bool R, bool E>
requires(std::equality_comparable<P>) void Graph<T, P, R, E>::connectNodes(NodeConvertible auto& upstream, NodeConvertible auto& downstream)
{
    CpuNode* upstreamNodePtr = getNode(upstream);
    CpuNode* downstreamNodePtr = getNode(downstream);

    assert(upstreamNodePtr != nullptr && downstreamNodePtr != nullptr);
    connectNodes(upstreamNodePtr, downstreamNodePtr);
}

static void nodeExecutor(const bool& stop, std::latch& latch, std::mutex& mutex, std::condition_variable& condition,
                         std::vector<CpuNode*>& unblockedNodes,
                         std::unordered_map<CpuNode*, std::atomic<std::size_t>>& upstreamDependencyCount,
                         const std::unordered_map<CpuNode*, std::atomic<std::size_t>>& upstreamDependencyCountOrig,
                         const std::unordered_map<CpuNode*, std::vector<CpuNode*>>& downstreamDependencies)
{
    CpuNode* nodeToRun;
    std::vector<CpuNode*> currentUnblockedNodes;
    currentUnblockedNodes.reserve(unblockedNodes.capacity());
    while (true)
    {
        {
            std::unique_lock<std::mutex> lock(mutex);
            condition.wait(lock, [&unblockedNodes, &stop] { return !unblockedNodes.empty() || stop; });
            if (stop)
            {
                return;
            }

            nodeToRun = unblockedNodes.back();
            unblockedNodes.pop_back();
        }

        nodeToRun->execute();

        auto it = downstreamDependencies.find(nodeToRun);
        if (it != downstreamDependencies.cend())
        {
            const std::vector<CpuNode*>& downstreamNodes = it->second;
            for (CpuNode* node : downstreamNodes)
            {
                auto& atomicCount = upstreamDependencyCount.find(node)->second;
                if (atomicCount.fetch_sub(1) - 1 == 0)
                {
                    currentUnblockedNodes.push_back(node);
                    atomicCount.store(upstreamDependencyCountOrig.find(node)->second.load());
                }
            }

            if (!currentUnblockedNodes.empty())
            {
                // We could avoid blocking here if the container was lockfree
                std::unique_lock<std::mutex> lock(mutex);
                for (CpuNode* node : currentUnblockedNodes)
                {
                    unblockedNodes.push_back(node);
                    condition.notify_one();
                }
            }
            currentUnblockedNodes.clear();
        }

        latch.count_down();
    }
}

template<typename T, Hashable P, bool R, bool E>
requires(std::equality_comparable<P>) void Graph<T, P, R, E>::build(const std::size_t numThreads)
{
    for (const auto& structToFill : m_IOBufferIndices)
    {
        auto& [address, indices] = structToFill;
        std::vector<const void* const*> doublePtrBuffer;
        doublePtrBuffer.reserve(indices.size());
        for (const auto& index : indices)
        {
            const void* const* ptr = std::holds_alternative<const void*>(m_IOPointers[index]) ?
                                         &std::get<const void*>(m_IOPointers[index]) :
                                         &std::get<void*>(m_IOPointers[index]);
            doublePtrBuffer.push_back(ptr);
        }

        memcpy(address, doublePtrBuffer.data(), doublePtrBuffer.size() * sizeof(void**));
    }

    m_IOBufferIndices.clear();
    m_IOBufferIndices.shrink_to_fit();

    for (const auto& node : m_nodes)
    {
        if (!m_nodeUpstreamDependencyCount.contains(node.get()))
        {
            m_nodesWithoutUpstreamDependencies.push_back(node.get());
        }
    }

    m_unblockedNodes.reserve(m_nodes.size());

    for (const auto& [key, value] : m_nodeUpstreamDependencyCount)
    {
        m_nodeUpstreamDependencyCountCopy[key].store(value.load());
    }

    m_workers.reserve(numThreads);
    for (std::size_t thread = 0; thread < numThreads; thread++)
    {
        m_workers.push_back(std::thread(nodeExecutor,
                                        std::ref(m_stop),
                                        std::ref(m_latch),
                                        std::ref(m_mutex),
                                        std::ref(m_condition),
                                        std::ref(m_unblockedNodes),
                                        std::ref(m_nodeUpstreamDependencyCountCopy),
                                        std::ref(m_nodeUpstreamDependencyCount),
                                        std::ref(m_nodeDownstreamDependencies)));
    }
}

template<typename T, Hashable P, bool R, bool E>
requires(std::equality_comparable<P>) template<typename Node>
Node* Graph<T, P, R, E>::getNode(std::string_view nodeName)
{
    auto it = m_nodeNameToPtr.find(nodeName);
    if (it == m_nodeNameToPtr.cend())
    {
        assert(false);
    }

    if constexpr (R)
    {
        Node* node = dynamic_cast<Node*>(it->second);
        assert(node != nullptr);
        return node;
    }
    else
    {
        Node* node = static_cast<Node*>(it->second);
        return node;
    }
}

template<typename T, Hashable P, bool R, bool E>
requires(std::equality_comparable<P>) void Graph<T, P, R, E>::execute()
{
    std::construct_at(reinterpret_cast<std::latch*>(&m_latchBuffer), m_nodes.size());
    for (CpuNode* node : m_nodesWithoutUpstreamDependencies)
    {
        m_unblockedNodes.push_back(node);
    }

    m_condition.notify_all();
    m_latch.wait();
    std::destroy_at(reinterpret_cast<std::latch*>(&m_latchBuffer));
}

template<typename T, Hashable P, bool R, bool E>
requires(std::equality_comparable<P>) template<typename NodeIO>
void Graph<T, P, R, E>::checkEachFieldIsDoublePtr()
{
    pfr::for_each_field(NodeIO{},
                        [](auto& field)
                        {
                            using FieldTypeDoublePtr = std::remove_reference_t<decltype(field)>;
                            using FieldTypePtr = std::remove_pointer_t<FieldTypeDoublePtr>;
                            // using FieldType = std::remove_pointer_t<FieldTypePtr>;
                            static_assert(std::is_pointer_v<FieldTypeDoublePtr>, "Every field in IO structs must be double pointers");
                            static_assert(std::is_pointer_v<FieldTypePtr>, "Every field in IO structs must be double pointers");
                            // static_assert(!std::is_pointer_v<FieldType>, "Every field in IO structs must be double pointers");
                        });
}

template<typename T, Hashable P, bool R, bool E>
requires(std::equality_comparable<P>) template<std::size_t N, typename StructType, typename Arg>
void Graph<T, P, R, E>::expandPointers(std::vector<const void*>& vector, std::unordered_map<std::size_t, P>& indexToPlaceHolder, Arg&& arg)
{
    assert(arg != nullptr);
    using FieldType = std::remove_const_t<std::remove_pointer_t<std::remove_pointer_t<decltype(pfr::get<N>(StructType{}))>>>;
    using ArgType = std::remove_const_t<std::remove_pointer_t<std::remove_reference_t<decltype(arg)>>>;

    constexpr bool isGoodType = std::is_pointer_v<Arg> && std::is_same_v<ArgType, FieldType>;
    constexpr bool isPlaceHolder =
        std::is_same_v<Arg, P> ||
        (std::is_array_v<std::remove_reference_t<Arg>> &&
         std::is_same_v<std::remove_extent_t<std::remove_reference_t<std::remove_const_t<Arg>>>, std::remove_pointer_t<P>>);

    static_assert(isGoodType || (isPlaceHolder), "IO must be a pointer to its memory address (of the good type) or a placeholder");
    static_assert(!(isGoodType && isPlaceHolder), "Ambiguous type, can't decide if type or placeholder");

    if constexpr (isGoodType)
    {
        vector.push_back(std::forward<Arg>(arg));
    }
    else
    {
        if constexpr (R)
        {
            auto it = this->m_placeHolderToTypeInfo.find(std::forward<Arg>(arg));
            if (it == this->m_placeHolderToTypeInfo.end())
            {
                this->m_placeHolderToTypeInfo.emplace(arg, typeid(FieldType));
            }
            else if (it->second.get() != typeid(FieldType))
            {
                std::cout << "Error while adding placeholder. Type mismatch!\n"; // TODO add std::source_location
                assert(false);
            }
        }

        vector.push_back(nullptr);
        indexToPlaceHolder.emplace(N, std::forward<Arg>(arg));
    }
}

template<typename T, Hashable P, bool R, bool E>
requires(std::equality_comparable<P>) template<std::size_t N, typename StructType, typename Arg, typename... Args>
void Graph<T, P, R, E>::expandPointers(std::vector<const void*>& vector, std::unordered_map<std::size_t, P>& indexToPlaceHolder, Arg&& arg,
                                       Args&&... args)
{
    expandPointers<N, StructType>(vector, indexToPlaceHolder, std::forward<Arg>(arg));
    expandPointers<N + 1, StructType>(vector, indexToPlaceHolder, std::forward<Args>(args)...);
}

template<typename T, Hashable P, bool R, bool E>
requires(std::equality_comparable<P>) template<std::size_t N, typename StructType, typename... Args>
std::pair<std::vector<const void*>, std::unordered_map<std::size_t, P>> Graph<T, P, R, E>::expandPointers(Args&&... args)
{
    std::unordered_map<std::size_t, P> indexToPlaceHolder;
    std::vector<const void*> vector;
    vector.reserve(sizeof...(Args));
    expandPointers<N, StructType>(vector, indexToPlaceHolder, std::forward<Args>(args)...);
    return {vector, indexToPlaceHolder};
}

template<typename T, Hashable P, bool R, bool E>
requires(std::equality_comparable<P>) template<bool IsOutput, typename StructType, typename... Args>
void Graph<T, P, R, E>::parseStruct(std::vector<std::size_t>& structsBufferIndices, std::vector<P>& localPlaceHolders, Args&&... args)
{
    auto [localIO, indexToPlaceHolder] = expandPointers<0, StructType>(std::forward<Args>(args)...);
    structsBufferIndices.resize(sizeof...(Args));

    for (std::size_t i = 0; i < localIO.size(); i++)
    {
        if (localIO[i] != nullptr)
        {
            auto it = m_addressToBufferIndex.find(localIO[i]);
            if (it == m_addressToBufferIndex.cend())
            {
                structsBufferIndices[i] = m_IOPointers.size();
                m_addressToBufferIndex.emplace(localIO[i], m_IOPointers.size());
                m_IOPointers.push_back(localIO[i]);
            }
            else
            {
                const std::size_t index = it->second;
                structsBufferIndices[i] = index;
            }

            continue;
        }
        // If our pointer is nullptr, it means it was a placeholder
        // Make sure all registered placeholders point to the same place
        // For new placeholders, add a ptr to the buffer and save its index
        auto placeHolder = indexToPlaceHolder[i];

        localPlaceHolders.push_back(placeHolder);
        auto it = m_placeHolderToBufferIndex.find(placeHolder);
        if (it == m_placeHolderToBufferIndex.end())
        {
            structsBufferIndices[i] = m_IOPointers.size();
            m_placeHolderToBufferIndex.emplace(placeHolder, m_IOPointers.size());
            m_IOPointers.push_back(static_cast<void*>(nullptr));
        }
        else
        {
            const std::size_t index = it->second;
            structsBufferIndices[i] = index;
        }

        if constexpr (IsOutput)
        {
            m_outputPlaceHolders.emplace(placeHolder);
        }
    }
}

template<typename T, Hashable P, bool R, bool E>
requires(std::equality_comparable<P>) void Graph<T, P, R, E>::connectNodes(CpuNode* upstream, CpuNode* downstream)
{
    auto upstreamPointerOutputIt = m_nodePointerOutputsIndices.find(upstream);
    auto downstreamPointerInputIt = m_nodePointerInputsIndices.find(downstream);
    assert(!(upstreamPointerOutputIt == m_nodePointerOutputsIndices.end() || downstreamPointerInputIt == m_nodePointerInputsIndices.end()));

    const auto& upstreamPointerOutputsIndices = upstreamPointerOutputIt->second;
    const auto& downstreamPointerInputsIndices = downstreamPointerInputIt->second;

    bool hasMatch = false;
    // Check ptrs ...
    for (const auto& output : upstreamPointerOutputsIndices)
    {
        // O(n^2) could be reduced with the usage of a map
        if (std::find(downstreamPointerInputsIndices.cbegin(), downstreamPointerInputsIndices.cend(), output) !=
            downstreamPointerInputsIndices.cend())
        {
            hasMatch = true;
            break;
        }
    }

    if (hasMatch == false)
    {
        auto upstreamPlaceHolderOutputIt = m_nodePlaceHolderOutputs.find(upstream);
        auto upstreamPlaceHolderInputIt = m_nodePlaceHolderInputs.find(downstream);
        assert(!(upstreamPlaceHolderOutputIt == m_nodePlaceHolderOutputs.end() ||
                 upstreamPlaceHolderInputIt == m_nodePlaceHolderInputs.end()));

        const auto& upstreamPlaceHolderOuputs = upstreamPlaceHolderOutputIt->second;
        const auto& downstreamPlaceHolderInputs = upstreamPlaceHolderInputIt->second;

        for (const auto& output : upstreamPlaceHolderOuputs)
        {
            if (std::find(downstreamPlaceHolderInputs.cbegin(), downstreamPlaceHolderInputs.cend(), output) !=
                downstreamPlaceHolderInputs.cend())
            {
                hasMatch = true;
                break;
            }
        }
    }

    assert(hasMatch);

    auto it = m_nodeUpstreamDependencyCount.emplace(downstream, 0).first;
    it->second++;
    m_nodeDownstreamDependencies[upstream].push_back(downstream);
}

template<typename T, Hashable P, bool R, bool E>
requires(std::equality_comparable<P>) CpuNode* Graph<T, P, R, E>::getNode(NodeConvertible auto& nodeIdentifier)
{
    if constexpr (std::is_convertible_v<decltype(nodeIdentifier), std::string_view>)
    {
        auto it = m_nodeNameToPtr.find(std::string_view(nodeIdentifier));
        if (it == m_nodeNameToPtr.end())
        {
            return nullptr;
        }
        return it->second;
    }
    else // Better have an else with if constexpr
    {
        return nodeIdentifier;
    }
}
