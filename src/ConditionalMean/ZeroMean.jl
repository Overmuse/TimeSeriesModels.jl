struct ZeroMean{T} <: ConditionalMeanModel{T} end

function ZeroMean(coefs::Vector{T}) where {T}
    ZeroMean{T}()
end

function is_valid(::ZeroMean)
    true
end

function nparams(::Type{ZeroMean})
    0
end

function presamples(::ZeroMean)
    0
end

function initial_coefficients(::Type{ZeroMean}, y::Vector{T}) where {T}
    T[]
end

function conditional_mean(::ZeroMean{T}, y, ϵ, σ) where {T}
    T(0.0)
end
