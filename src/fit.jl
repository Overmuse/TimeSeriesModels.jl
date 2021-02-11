function process_mean(model, y, ϵ, t)
    if t <= presamples(model)
        mean(y)
    else
        conditional_mean(model, y[1:t], ϵ[1:t])
    end
end
 
function process_variance(model, y, t)
    if t <= presamples(model)
        var(y)
    else
        conditional_variance(model, y[1:t])
    end
end

function log_likelihood(mean_model::ConditionalMeanModel{T}, variance_model::ConditionalVarianceModel{T}, y::Vector) where {T}
    N = length(y)
    ϵ = zeros(T, N)
    (is_valid(mean_model) && is_valid(variance_model)) || return T(-Inf)

    LL = 0.0
    for t in 1:N
        μ̂ = process_mean(mean_model, y, ϵ, t)
        σ̂ = sqrt(process_variance(variance_model, y, t)[end])
        ϵ[t] = y[t] - μ̂[end]
        LL += logpdf(Normal(0, σ̂), ϵ[t])
    end
    LL
end

function log_likelihood(mean_model, variance_model, coef, y)
    mean_nparams = nparams(mean_model)
    log_likelihood(mean_model(coef[1:mean_nparams]), variance_model(coef[mean_nparams+1:end]), y)
end

function fit(mean_model::Type{<:ConditionalMeanModel}, variance_model::Type{<:ConditionalVarianceModel}, y)
    opt = coef -> -log_likelihood(mean_model, variance_model, coef, y)
    init = mapreduce(model -> initial_coefficients(model, y), vcat, [mean_model, variance_model])
    solve = optimize(opt, init, method = BFGS(), autodiff = :forward)
    coef = minimizer(solve)
    mean_nparams = nparams(mean_model)
    (mean_model(coef[1:mean_nparams]), variance_model(coef[mean_nparams+1:end]))
end

function fit(model::Type{<:ConditionalMeanModel}, y)
    fit(model, SampleVariance, y)
end

function fit(model::Type{<:ConditionalVarianceModel}, y)
    fit(SampleMean, model, y)
end
