struct ConstantVariance{T} <: ConditionalVarianceModel{T}
    σ̄²::T
end

function ConstantVariance(coef::Vector)
    ConstantVariance(coef[])
end

function is_valid(model::ConstantVariance)
    model.σ̄² > 0
end

function nparams(::Type{ConstantVariance})
    1
end

function presamples(::ConstantVariance)
    0
end

function initial_coefficients(::Type{ConstantVariance}, y)
    [var(y)]
end

function conditional_variance(model::ConstantVariance{T}, y::Vector, σ) where {T}
    T(model.σ̄²)
end

function simulate(model::ConstantVariance, t)
    fill(model.σ̄², t)
end

function predict(model::ConstantVariance, t=1)
    model.σ̄²
end
