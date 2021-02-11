struct SampleMean{T} <: ConditionalMeanModel{T}
    μ̄::T
end

function SampleMean(coef::Vector)
    SampleMean(coef[])
end

function is_valid(::SampleMean)
    true
end

function nparams(::Type{SampleMean})
    1
end

function presamples(::SampleMean)
    0
end

function initial_coefficients(::Type{SampleMean}, y)
    [mean(y)]
end

function conditional_mean(model::SampleMean{T}, y::Vector) where {T}
    T(model.μ̄)
end

function simulate(model::SampleMean, t)
    fill(model.μ̄, t)
end

function predict(model::SampleMean, t=1)
    model.μ̄
end
