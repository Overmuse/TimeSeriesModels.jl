struct ARMA{p, q, T} <: ConditionalMeanModel{T}
    C::T
    ϕ::Vector{T}
    θ::Vector{T}

    function ARMA{p, q, T}(C, ϕ, θ) where {p, q, T}
        @assert length(ϕ) == p
        @assert length(θ) == q
        new{p, q, T}(C, ϕ, θ)
    end
end

const AR{p, T} = ARMA{p, 0, T}
const MA{q, T} = ARMA{0, q, T}
const ARMA{p, q, T} = ARMA{p, q, T}

function ARMA{p, q}(coefficients::Vector{T}) where {p, q, T}
    @assert length(coefficients) == (p + q + 1)
    ARMA{p, q, T}(coefficients[1], coefficients[2:p+1], coefficients[p+2:end])
end

function is_valid(model::ARMA{p, q}) where {p, q}
    all(<=(1), abs.(model.ϕ)) && all(<=(1), abs.(model.θ))
end

function nparams(::Type{ARMA{p, q}}) where {p, q}
    1 + p + q
end

function presamples(model::ARMA{p, q}) where {p, q}
    max(p, q)
end

function initial_coefficients(::Type{ARMA{p, q}}, y::Vector{T}) where {p, q, T}
    N = length(y)
    X = Matrix{T}(undef, N - p, p + 1)
    X[:, 1] .= ones(T)
    for i in 1:p
        X[:, i + 1] .= y[p - i + 1:N - i]
    end
    ϕ = X \ y[(p + 1):end]
    vcat(ϕ, zeros(T, q))
end

function conditional_mean(model::ARMA{p, q, T}, y, ϵ) where {p, q, T}
    ŷ = zeros(T, length(y))
    ŷ .= model.C
    for t in 2:length(y)
        if t > p
            for (i, φ) in enumerate(model.ϕ)
                ŷ[t] += y[t-i] * φ
            end
        end
        if t > q
            for (i, ϑ) in enumerate(model.θ)
                ŷ[t] += ϵ[t-i] * ϑ
            end
        end
    end
    ŷ
end

function residuals(model::ARMA, y)
    y .- filter(model, y)
end

function simulate(model::ARMA{p, q}, t) where {p, q}
    y = zeros(t)
    ϵ = zeros(t)
    y[1] = model.C
    for t in 2:length(y)
        ϵ[t-1] = rand()
        y[t] += model.C
        for (i, φ) in enumerate(model.ϕ)
            y[t] += y[t-i] * φ
        end
        for (i, ϑ) in enumerate(model.θ)
            y[t] += ϵ[t-1] * ϑ
        end
    end
    y
end
