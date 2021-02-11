struct SampleVariance{T} <: ConditionalVarianceModel{T}
    σ̄²::T
end

function SampleVariance(coef::Vector)
    SampleVariance(coef[])
end

function is_valid(model::SampleVariance)
    model.σ̄² > 0
end

function nparams(::Type{SampleVariance})
    1
end

function presamples(::SampleVariance)
    0
end

function initial_coefficients(::Type{SampleVariance}, y)
    [var(y)]
end

function conditional_variance(model::SampleVariance{T}, y::Vector) where {T}
    T(model.σ̄²)
end

function simulate(model::SampleVariance, t)
    fill(model.σ̄², t)
end

function predict(model::SampleVariance, t=1)
    model.σ̄²
end
