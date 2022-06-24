#pragma once

#include <iostream>
#include <numeric>
#include "graph/CpuNodeIo.h"
#include "graph/Graph.h"

void busyWait(std::int64_t us)
{
    auto start = std::chrono::high_resolution_clock::now();
    while (true)
    {
        auto now = std::chrono::high_resolution_clock::now();
        if  (std::chrono::duration_cast<std::chrono::microseconds>(now - start).count() >= us)
        {
            break;
        }
    }
}

namespace Node1
{
    struct Inputs
    {
        const std::size_t** a;
        const std::size_t** b;
        const std::size_t** c;
        const std::size_t** d;
    };

    struct Outputs
    {
        std::size_t** out;
    };

    class Node : public CpuNodeIo<Inputs, Outputs>
    {
    public:
        Node(const std::string_view nodeName, const Inputs& inputs, const Outputs& outputs)
            : CpuNodeIo<Inputs, Outputs>(nodeName, inputs, outputs)
        {
        }

        void execute() override
        {
            busyWait(10000);
            **outputs().out = **inputs().a + **inputs().b + **inputs().c + **inputs().d;
        }
    };
} // namespace Node1

namespace Node2
{
    struct Inputs
    {
        const float** coconut;
        const float** cocoa;
        const std::size_t** shampoo;
        const std::size_t** cake;
    };

    struct Outputs
    {
        std::int64_t** out;
    };

    class Node : public CpuNodeIo<Inputs, Outputs>
    {
    public:
        Node(const std::string_view nodeName, const Inputs& inputs, const Outputs& outputs)
            : CpuNodeIo<Inputs, Outputs>(nodeName, inputs, outputs)
        {
        }

        void execute() override
        {
            busyWait(7000);
            **outputs().out = static_cast<std::int64_t>(**inputs().coconut * **inputs().cocoa * **inputs().shampoo * **inputs().cake);
        }
    };
} // namespace Node2

namespace Node3
{
    struct Inputs
    {
        const std::size_t** node1Out;
        const std::int64_t** node2Out;
    };

    struct Outputs
    {
        char*** out;
    };

    class Node : public CpuNodeIo<Inputs, Outputs>
    {
    public:
        Node(const std::string_view nodeName, const Inputs& inputs, const Outputs& outputs)
            : CpuNodeIo<Inputs, Outputs>(nodeName, inputs, outputs)
        {
        }

        void execute() override
        {
            busyWait(3000);
            auto str = std::to_string(**inputs().node1Out) + " " + std::to_string(**inputs().node2Out);
            **outputs().out = strcpy(**outputs().out, str.c_str());
        }
    };
} // namespace Node3

namespace test1
{
    struct Mem
    {
        std::size_t unicorn;
        std::size_t rainbow;
        std::size_t candy;
        float milk;
    };

    void test()
    {
        Graph<Mem, const char*> graph;

        char str[64];
        char* decay = str;
        {
            auto start = std::chrono::high_resolution_clock::now();
            graph.createNode<Node1::Node>("Node1",
                                          graph.createInputs<Node1::Node>(&graph.memory().unicorn,
                                                                          &graph.memory().rainbow,
                                                                          &graph.memory().candy,
                                                                          // If we don't have the memory yet, put a placeholder
                                                                          "VeryCoolPlaceHolder"),
                                          graph.createOutputs<Node1::Node>("Node1OutPlaceHolder"));

            float outOfGraphMemory = 12.f;
            auto node2 = graph.createNode<Node2::Node>("Node2",
                                                       graph.createInputs<Node2::Node>(&outOfGraphMemory,
                                                                                       &graph.memory().milk,
                                                                                       "VeryCoolPlaceHolder",
                                                                                       &graph.memory().unicorn),
                                                       graph.createOutputs<Node2::Node>("Node2OutPlaceHolder"));

            auto node3 = graph.createNode<Node3::Node>("Node3",
                                                       graph.createInputs<Node3::Node>("Node1OutPlaceHolder", "Node2OutPlaceHolder"),
                                                       graph.createOutputs<Node3::Node>("Result!!!"));

            graph.connectNodes("Node1", node3);
            graph.connectNodes(node2, node3);

            graph.memory().unicorn = 4;
            graph.memory().rainbow = 5;
            graph.memory().candy = 2;
            graph.memory().milk = 1.2f;

            const std::size_t node1LastInput = 32;
            auto placeholder = "VeryCoolPlaceHolder";
            graph.setPlaceHolderAddress(placeholder, &node1LastInput);

            std::size_t node1Out;
            graph.setPlaceHolderAddress("Node1OutPlaceHolder", &node1Out);

            std::int64_t node2Out = 5;
            graph.setPlaceHolderAddress("Node2OutPlaceHolder", &node2Out);

            graph.setPlaceHolderAddress("Result!!!", &decay);
            auto end = std::chrono::high_resolution_clock::now();
            std::cout << "Time to create nodes (us): " << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count()
                      << std::endl;
        }

        {
            auto start = std::chrono::high_resolution_clock::now();
            graph.build(2);
            auto end = std::chrono::high_resolution_clock::now();
            std::cout << "Time to build graph (us): " << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count()
                      << std::endl;
        }

        std::vector<std::size_t> durations;
        for (std::size_t i = 0; i < 100; i++)
        {
            auto start = std::chrono::high_resolution_clock::now();
            graph.execute();
            auto end = std::chrono::high_resolution_clock::now();

            durations.push_back(std::chrono::duration_cast<std::chrono::microseconds>(end - start).count());
        }
        std::cout << "Average execute time (us): " << std::accumulate(durations.begin(), durations.end(), 0UL) / durations.size()
                  << std::endl;
        std::cout << decay << std::endl;
    }
} // namespace test1
