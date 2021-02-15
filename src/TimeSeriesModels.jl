module TimeSeriesModels

using DataStructures: CircularBuffer
using Distributions: Normal, logpdf
using Optim: BFGS, optimize, minimizer
using Statistics: mean, var

export
    UnivariateModel,

    # Mean models
    ZeroMean,
    ConstantMean,
    AR,
    MA,
    ARMA,
    VolInMean,

    # Variance models
    ARCH,
    GARCH,
    ConstantVariance,

    # Functions
    fit

abstract type ConditionalMeanModel{T} end
abstract type ConditionalVarianceModel{T} end

include("UnivariateModel.jl")
include("ConditionalMean/ZeroMean.jl")
include("ConditionalMean/ConstantMean.jl")
include("ConditionalMean/VolInMean.jl")
include("ConditionalMean/ARMA.jl")
include("ConditionalVariance/ConstantVariance.jl")
include("ConditionalVariance/GARCH.jl")
include("fit.jl")

end # module
