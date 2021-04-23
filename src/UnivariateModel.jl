struct UnivariateModel{T}
    mean_model::ConditionalMeanModel{T}
    variance_model::ConditionalVarianceModel{T}
    dist::Distribution
end
