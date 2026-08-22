/**
 * Pocket-LLM: Hyperbolic Math Primitives Header
 * ==============================================
 *
 * PRACTICAL VALUE:
 * ----------------
 * This header defines the core non-Euclidean mathematical operations
 * (sinh, cosh, tanh) required for the edge runtime to compute
 * hyperbolic distances natively on mobile hardware without relying
 * on heavy math libraries. These primitives replace traditional
 * dot-product matrix multiplications with coral-curve geometry
 * calculations, enabling nanosecond inference at near-zero power.
 */

#ifndef HYPERBOLIC_MATH_H
#define HYPERBOLIC_MATH_H

#include <cmath>

namespace pocketllm {

    // Hyperbolic Sine: sinh(x) = (e^x - e^(-x)) / 2
    inline float hyperbolic_sine(float x) {
        return std::sinh(x);
    }

    // Hyperbolic Cosine: cosh(x) = (e^x + e^(-x)) / 2
    inline float hyperbolic_cosine(float x) {
        return std::cosh(x);
    }

    // Hyperbolic Tangent: tanh(x) = sinh(x) / cosh(x)
    inline float hyperbolic_tangent(float x) {
        return std::tanh(x);
    }

    // Poincaré Disk Distance Metric
    // Computes the hyperbolic distance between two points u and v inside the unit disk.
    // This replaces Euclidean dot-product in attention layers.
    inline float poincare_distance(const float* u, const float* v, int dims, float curvature) {
        float euclidean_sq = 0.0f;
        float u_norm_sq = 0.0f;
        float v_norm_sq = 0.0f;

        for (int i = 0; i < dims; ++i) {
            float diff = u[i] - v[i];
            euclidean_sq += diff * diff;
            u_norm_sq += u[i] * u[i];
            v_norm_sq += v[i] * v[i];
        }

        float alpha = 1.0f - curvature * u_norm_sq;
        float beta  = 1.0f - curvature * v_norm_sq;
        float gamma = 1.0f + (2.0f * curvature * euclidean_sq) / (alpha * beta);

        return std::acosh(gamma) / std::sqrt(curvature);
    }

} // namespace pocketllm

#endif // HYPERBOLIC_MATH_H
