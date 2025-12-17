#pragma once

// CORTEXT_EXPORT macro for controlling symbol visibility.
//
// When building cortext as a shared library (CORTEXT_BUILDING_SHARED defined),
// symbols marked with CORTEXT_EXPORT will be visible to consumers.
// All other symbols are hidden by default via -fvisibility=hidden.

#if defined(_WIN32)
#if defined(CORTEXT_BUILDING_SHARED)
#define CORTEXT_EXPORT __declspec(dllexport)
#elif defined(CORTEXT_USING_SHARED)
#define CORTEXT_EXPORT __declspec(dllimport)
#else
#define CORTEXT_EXPORT
#endif
#elif defined(__GNUC__) && __GNUC__ >= 4
#define CORTEXT_EXPORT __attribute__((visibility("default")))
#else
#define CORTEXT_EXPORT
#endif
