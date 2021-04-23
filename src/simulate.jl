function simulate(model, T)
    dist = standardized(model.dist)
    mean_model = model.mean_model
    variance_model = model.variance_model
    mean_samples = max(1, presamples(mean_model))
    variance_samples = max(1, presamples(variance_model))
    μ = CircularBuffer{Float64}(mean_samples)
    σ = CircularBuffer{Float64}(variance_samples)
    ϵ = CircularBuffer{Float64}(max(mean_samples, variance_samples))
    y = zeros(T)
    for t in 1:T
        if t <= presamples(variance_model)
            push!(σ, sqrt(unconditional_variance(variance_model)))
        else
            push!(σ, sqrt(conditional_variance(variance_model, ϵ, σ)))
        end
        if t <= presamples(mean_model)
            push!(μ, unconditional_mean(mean_model))
        else
            push!(μ, conditional_mean(mean_model, view(y, 1:t), ϵ, σ))
        end
        push!(ϵ, rand(dist) * σ[end])
        y[t] = μ[end] + ϵ[end]
    end
    y
end
