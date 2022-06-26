#pragma once

#include <string_view>

class CpuNode
{
public:
    CpuNode(std::string_view nodeName)
        : m_nodeName(nodeName)
    {
    }

    CpuNode(const CpuNode&) = delete;
    CpuNode(CpuNode&&) = delete;
    CpuNode& operator=(const CpuNode&) = delete;
    CpuNode& operator=(CpuNode&&) = delete;

    virtual ~CpuNode() {}

    virtual void execute() = 0;

    const std::string_view getNodeName() const { return m_nodeName; }

private:
    std::string_view m_nodeName;
};
