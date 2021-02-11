struct ZeroMean{T} <: ConditionalMeanModel{T}
    coefs::Vector{T}

    function ZeroMean{T}(coefs::Vector) where {T}
        new{T}(coefs)
    end
end

function is_valid(::ZeroMean)
    true
end

function nparams(::Type{ZeroMean{T}}) where {T}
    0
end

function presamples(::ZeroMean)
    0
end

function initial_coefficients(::Type{ZeroMean{T}}, y) where T
    T[]
end

function conditional_mean(model::ZeroMean{T}, y::Vector) where {T}
    T(0.0)
end
