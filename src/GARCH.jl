struct GARCH{p, q}
    ω::Float64
    alphas::Vector{Float64}
    betas::Vector{Float64}

    function GARCH{p, q}(ω, alphas, betas) where {p, q}
        @assert(p == length(alphas))
        @assert(q == length(betas))
        new{p, q}(ω, alphas, betas)
    end
end

function coefficients(model)
    [model.ω, model.alphas, model.betas]
end

function unconditional_variance(model::GARCH{p, q}) where {p, q}
    unconditional_variance(model.ω, model.alphas, model.betas)
end

function unconditional_variance(ω, alphas, betas)
    ω / (1 - sum(alphas) - sum(betas))
end

function required_samples(model::GARCH{p, q}) where {p, q}
    max(p, q)
end

function filter(model::GARCH, ϵ)
    filter(model.ω, model.alphas, model.betas, ϵ)
end

function filter(ω, alphas, betas, ϵ)
    T = length(ϵ)
    σ² = zeros(T)
    n = max(length(alphas), length(betas))

    σ²[1] = unconditional_variance(ω, alphas, betas)
    for t in (n+1):T
        σ²[t] = ω
        for (i, α) in enumerate(alphas)
            σ²[t] += α * ϵ[t - i] ^ 2
        end
        for (i, β) in enumerate(betas)
            σ²[t] += β * σ²[t-i]
        end
    end
    σ²
end

function log_likelihood(model, ϵ)
    log_likelihood(model.ω, model.alphas, model,betas, ϵ)
end

function log_likelihood(ω, alphas, betas, ϵ)
    σ² = filter(ω, alphas, betas, ϵ)
    sum(-log.(σ²) .- ϵ.^2 ./ σ²)
end

function fit(model::GARCH{p, q}, ϵ) where {p, q}
    opt = coeffs -> log_likelihood(extract_coeffs(coeffs)..., ϵ)
    maximize(opt, vcat(model.ω, model.alphas, model.betas))
end
