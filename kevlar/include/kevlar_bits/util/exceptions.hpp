#pragma once
#include <string>
#include <exception>

namespace kevlar {

struct kevlar_error : std::exception {};
struct thread_error : kevlar_error
{ 
    thread_error(const char* msg)
        : msg_(msg)
    {}

    thread_error(const std::string& msg)
        : msg_(msg)
    {}

    const char* what() const noexcept override { return msg_.data(); }

private:
    std::string msg_;
};

struct min_lmda_reached_error : kevlar_error
{
    min_lmda_reached_error()
        : msg_("Min lmda reached. Try a grid of lambda with lower values.")
    {}

    const char* what() const noexcept override { return msg_.data(); }

private:
    std::string msg_;
};

} // namespace kevlar
