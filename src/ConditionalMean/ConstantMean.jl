struct ConstantMean{T} <: ConditionalMeanModel{T}
    μ̄::T
end

function ConstantMean(coef::Vector)
    ConstantMean(coef[])
end

function is_valid(::ConstantMean)
    true
end

function nparams(::Type{ConstantMean})
    1
end

function presamples(::ConstantMean)
    0
end

function initial_coefficients(::Type{ConstantMean}, y)
    [mean(y)]
end

function conditional_mean(model::ConstantMean{T}, y::Vector, ϵ) where {T}
    T(model.μ̄)
end

function simulate(model::ConstantMean, t)
    fill(model.μ̄, t)
end

function predict(model::ConstantMean, t=1)
    model.μ̄
end
