#pragma once

#include <type_traits>
#include <vector>
#include "CpuNode.h"

enum class NodeIoType : std::uint8_t
{
    Input = 0,
    Output,
};

template<typename T, typename PlaceHolder, NodeIoType Type>
class NodeIO
{
public:
    template<typename... Args>
    explicit NodeIO(std::vector<std::size_t>& bufferIndices, std::vector<PlaceHolder> placeHolderOutputs)
        : m_value()
        , m_bufferIndices(bufferIndices)
        , m_placeHolderOutputs(std::move(placeHolderOutputs))
    {
    }

    operator T&&() { return std::move(m_value); }

    std::vector<std::size_t>& getBufferIndices() { return m_bufferIndices; }
    std::vector<PlaceHolder>& getPlaceHolders() { return m_placeHolderOutputs; }

private:
    T m_value;
    std::vector<std::size_t> m_bufferIndices;
    std::vector<PlaceHolder> m_placeHolderOutputs;
};

template<typename T, typename PlaceHolder>
using NodeInput = NodeIO<typename T::InputType, PlaceHolder, NodeIoType::Input>;

template<typename T, typename PlaceHolder>
using NodeOutput = NodeIO<typename T::OutputType, PlaceHolder, NodeIoType::Output>;

template<class T>
concept NodeConvertible = std::is_convertible_v<T, std::string_view> || std::is_convertible_v<T, CpuNode*>;

template<typename T>
concept Hashable = requires(T a)
{
    // clang-format off
    { std::hash<T>{}(a) } -> std::convertible_to<std::size_t>;
    // clang-format on
};

template<typename InputClass, typename OutputClass>
class CpuNodeIo : public CpuNode
{
public:
    using InputType = InputClass;
    using OutputType = OutputClass;

    template<typename, Hashable PlaceHolder, bool, bool>
    requires(std::equality_comparable<PlaceHolder>&& Hashable<PlaceHolder>) friend class Graph;

    CpuNodeIo(const std::string_view nodeName, const InputClass& inputs, const OutputClass& outputs)
        : CpuNode(nodeName)
        , m_inputs(inputs)
        , m_outputs(outputs)
    {
    }

    virtual ~CpuNodeIo() {}

    const InputClass& inputs() const { return m_inputs; }
    OutputClass& outputs() { return m_outputs; }

    virtual void execute() = 0;

private:
    InputClass m_inputs;
    OutputClass m_outputs;
};
