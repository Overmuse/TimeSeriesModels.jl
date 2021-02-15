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

function conditional_variance(model::GARCH{p, q, T}, ϵ, σ̂) where {p, q, T}
    σ² = model.ω
    @inbounds for j in 1:p
        σ² += model.betas[j] * σ̂[end - j + 1] ^ 2
    end
    @inbounds for j in 1:q
        σ² += model.alphas[j] * ϵ[end - j + 1] ^ 2
    end
    σ²
end

function unconditional_variance(model::GARCH{p, q}) where {p, q}
    unconditional_variance(model.ω, model.alphas, model.betas)
end

function unconditional_variance(ω, alphas, betas)
    ω / (1 - sum(alphas) - sum(betas))
end
