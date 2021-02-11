struct ARIMA{p, d, q, T} <: ConditionalMeanModel{T}
    C::T
    ϕ::Vector{T}
    θ::Vector{T}

    function ARIMA{p, d, q, T}(C, ϕ, θ) where {p, d, q, T}
        @assert length(ϕ) == p
        @assert length(θ) == q
        new{p, d, q, T}(C, ϕ, θ)
    end
end

const AR{p, T} = ARIMA{p, 0, 0, T}
const MA{q, T} = ARIMA{0, 0, q, T}
const ARMA{p, q, T} = ARIMA{p, 0, q, T}

function ARIMA{p, d, q}(coefficients::Vector{T}) where {p, d, q, T}
    @assert length(coefficients) == (1 + p + q)
    ARIMA{p, d, q, T}(coefficients[1], coefficients[2:p+1], coefficients[p+2:end])
end

function is_valid(model::ARIMA{p, d, q}) where {p, d, q}
    all(<=(1), abs.(model.ϕ)) && all(<=(1), abs.(model.θ))
end

function nparams(::Type{ARIMA{p, d, q}}) where {p, d, q}
    p + q + 1
end

function difference(D, y)
    if D == 0
        y
    else
        difference(D-1, y[2:end] .- y[1:end-1])
    end
end

function presamples(model::ARIMA{p, d, q}) where {p, d, q}
    d + max(p, q)
end

function initial_coefficients(::Type{ARIMA{p, d, q}}, y::Vector{T}) where {p, d, q, T}
    N = length(y)
    X = Matrix{T}(undef, N - p, p + 1)
    X[:, 1] .= ones(T)
    for i in 1:p
        X[:, i + 1] .= y[p - i + 1:N - i]
    end
    ϕ = X \ y[(p + 1):end]
    vcat(ϕ, zeros(T, q))
end

function conditional_mean(model::ARIMA{p, d, q, T}, y, ϵ) where {p, d, q, T}
    ŷ = zeros(T, length(y) - d)
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

function residuals(model::ARIMA, y)
    y .- filter(model, y)
end

function simulate(model::ARIMA{p, d, q}, t) where {p, d, q}
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
