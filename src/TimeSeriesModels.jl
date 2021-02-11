module TimeSeriesModels

using Distributions: Normal, logpdf
using Optim: BFGS, optimize, minimizer
using Statistics: mean, var

export
    AR,
    MA,
    ARMA,
    ARIMA,
    GARCH,
    SampleMean,
    SampleVariance,
    fit

abstract type ConditionalMeanModel{T} end
abstract type ConditionalVarianceModel{T} end

include("ZeroMean.jl")
include("SampleMean.jl")
include("SampleVariance.jl")
include("fit.jl")
include("ARIMA.jl")
#include("GARCH.jl")

end # module
