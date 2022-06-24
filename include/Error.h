#pragma once

class ErrorFake
{
public:
    consteval ErrorFake() {}
    consteval bool isError() const { return false; }
    consteval operator bool() const { return true; }
    consteval const char* text() const { return nullptr; }
};

class [[nodiscard]] ErrorReal
{
public:
    constexpr ErrorReal() {}
    constexpr ErrorReal(const char* text)
        : m_text(text)
    {
    }
    constexpr bool isError() const { return m_text == nullptr; }
    constexpr operator bool() const { return !isError(); }
    constexpr const char* text() const { return m_text; }

private:
    const char* m_text = nullptr;
};
