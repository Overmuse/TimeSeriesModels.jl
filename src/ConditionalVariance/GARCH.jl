struct GARCH{p, q, T} <: ConditionalVarianceModel{T}
    ω::T
    alphas::Vector{T}
    betas::Vector{T}

    function GARCH{p, q, T}(ω, alphas, betas) where {p, q, T}
        @assert(p == length(alphas))
        @assert(q == length(betas))
        new{p, q, T}(ω, alphas, betas)
    end
end

function GARCH{p, q}(coef::Vector{T}) where {p, q, T}
    GARCH{p, q, T}(coef[1], coef[2:1+p], coef[2+p:end])
end

const ARCH{p, T} = GARCH{p, 0, T}

function is_valid(model::GARCH)
    model.ω > 0 && all(x -> 0 <= x <= 1, model.alphas) && all(x -> 0 <= x <= 1, model.betas)
end

function nparams(::Type{GARCH{p, q}}) where {p, q}
    1 + p + q
end

function presamples(::GARCH{p, q}) where {p, q}
    max(p, q)
end

function initial_coefficients(::Type{GARCH{p, q}}, y) where {p, q}
    alphas = fill(0.9 / p, p)
    betas = fill(0.05 / q, q)
    ω = var(y) * (1.0 - sum(alphas) - sum(betas))
    vcat(ω, alphas, betas)
end

function conditional_variance(model::GARCH{p, q, T}, y::Vector) where {p, q, T}
    t = length(y)
    σ² = zeros(T, t)
    n = presamples(model)

    σ²[1] = unconditional_variance(model)
    for i in (n+1):t
        σ²[i] = model.ω
        for (j, α) in enumerate(model.alphas)
            σ²[i] += α * y[i - j] ^ 2
        end
        for (j, β) in enumerate(model.betas)
            σ²[i] += β * σ²[i - j]
        end
    end
    σ²
end

function unconditional_variance(model::GARCH{p, q}) where {p, q}
    unconditional_variance(model.ω, model.alphas, model.betas)
end

function unconditional_variance(ω, alphas, betas)
    ω / (1 - sum(alphas) - sum(betas))
end
