module TimeSeriesModels

using Distributions: Normal, logpdf
using Optim: BFGS, optimize, minimizer
using Statistics: mean, var

export
    UnivariateModel,
    AR,
    MA,
    ARMA,
    ARCH,
    GARCH,
    SampleMean,
    SampleVariance,
    fit

abstract type ConditionalMeanModel{T} end
abstract type ConditionalVarianceModel{T} end

include("UnivariateModel.jl")
include("ConditionalMean/ZeroMean.jl")
include("ConditionalMean/SampleMean.jl")
include("ConditionalMean/ARMA.jl")
include("ConditionalVariance/SampleVariance.jl")
include("ConditionalVariance/GARCH.jl")
include("fit.jl")

end # module
