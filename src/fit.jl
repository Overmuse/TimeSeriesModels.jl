function process_mean(model, y, ϵ, σ̂, t)
    if t <= presamples(model)
        mean(y)
    else
        conditional_mean(model, view(y, 1:t), ϵ, σ̂)
    end
end
 
function process_variance(model, y, ϵ, σ̂, t)
    if t <= presamples(model)
        var(y)
    else
        conditional_variance(model, ϵ, σ̂)
    end
end

function loglikelihood(mean_model::ConditionalMeanModel{T}, variance_model::ConditionalVarianceModel{T}, dist::Distribution, y::Vector) where {T}
    std_dist = standardized(dist)
    (is_valid(mean_model) && is_valid(variance_model) && is_valid(std_dist)) || return T(-Inf)
    mean_samples = max(1, presamples(mean_model))
    variance_samples = max(1, presamples(variance_model))
    μ̂ = CircularBuffer{T}(mean_samples)
    σ̂ = CircularBuffer{T}(variance_samples)
    ϵ = CircularBuffer{T}(max(mean_samples, variance_samples))

    LL = 0.0
    for t in 1:length(y)
        push!(σ̂, sqrt(process_variance(variance_model, y, ϵ, σ̂, t)))
        push!(μ̂, process_mean(mean_model, y, ϵ, σ̂, t))
        push!(ϵ, y[t] - μ̂[end])
        LL += logpdf(LocationScale(0, σ̂[end], std_dist), ϵ[end])
    end
    LL
end

function loglikelihood(model::UnivariateModel, y::Vector)
    loglikelihood(model.mean_model, model.variance_model, model.dist, y)
end

function loglikelihood(mean_model, variance_model, dist, coef, y)
    idx₁ = nparams(mean_model)
    idx₂ = idx₁ + nparams(variance_model)
    loglikelihood(mean_model(coef[1:idx₁]), variance_model(coef[idx₁+1:idx₂]), dist(coef[idx₂+1:end]...; check_args=false), y)
end

function fit(mean_model::Type{<:ConditionalMeanModel}, variance_model::Type{<:ConditionalVarianceModel}, y; dist = Normal)
    opt = coef -> -loglikelihood(mean_model, variance_model, dist, coef, y)
    init = mapreduce(model -> initial_coefficients(model, y), vcat, [mean_model, variance_model, dist])
    solve = optimize(opt, init, method = BFGS(), autodiff = :forward)
    coef = minimizer(solve)
    idx₁ = nparams(mean_model)
    idx₂ = idx₁ + nparams(variance_model)
    UnivariateModel(mean_model(coef[1:idx₁]), variance_model(coef[idx₁+1:idx₂]), dist(coef[idx₂+1:end]...))
end

function fit(model::Type{<:ConditionalMeanModel}, y; dist = Normal)
    fit(model, ConstantVariance, y; dist)
end

function fit(model::Type{<:ConditionalVarianceModel}, y; dist = Normal)
    fit(ConstantMean, model, y; dist)
end
