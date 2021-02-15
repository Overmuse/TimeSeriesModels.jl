struct VolInMean{T} <: ConditionalMeanModel{T}
    μ::T
    λ::T
end

function VolInMean(coef::Vector)
    VolInMean(coef[1], coef[2])
end

function is_valid(::VolInMean)
    true
end

function nparams(::Type{VolInMean})
    2
end

function presamples(::VolInMean)
    0
end

function initial_coefficients(::Type{VolInMean}, y)
    [mean(y), 1.0]
end

function conditional_mean(model::VolInMean{T}, y, ϵ, σ) where {T}
    T(model.μ + model.λ * σ[end])
end
