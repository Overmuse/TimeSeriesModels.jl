struct GARCH{p, q, T} <: ConditionalVarianceModel{T}
    ω::T
    betas::Vector{T}
    alphas::Vector{T}

    function GARCH{p, q, T}(ω, betas, alphas) where {p, q, T}
        @assert(q == length(alphas))
        @assert(p == length(betas))
        new{p, q, T}(ω, betas, alphas)
    end
end

function GARCH{p, q}(coef::Vector{T}) where {p, q, T}
    GARCH{p, q, T}(coef[1], coef[2:1+p], coef[2+p:end])
end

const ARCH{q, T} = GARCH{0, q, T}

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
    betas = fill(0.9 / p, p)
    alphas = fill(0.05 / q, q)
    ω = var(y) * (1.0 - sum(alphas) - sum(betas))
    vcat(ω, betas, alphas)
end

function conditional_variance(model::GARCH{p, q, T}, ϵ::Vector, σ̂) where {p, q, T}
    # TODO: Remove σ₀ parameter somehow
    t = length(ϵ)
    σ² = zeros(T, t)
    n = presamples(model)

    σ² .= model.ω
    for i in (n+1):t
        for j in 1:p
            σ²[i] += model.betas[j] * σ̂[i - j] ^ 2
        end
        for j in 1:q
            σ²[i] += model.alphas[j] * ϵ[i - j] ^ 2
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
