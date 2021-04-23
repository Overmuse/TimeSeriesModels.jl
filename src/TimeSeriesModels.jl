module TimeSeriesModels

using DataStructures: CircularBuffer
using Distributions: Distribution, Normal, TDist, LocationScale, logpdf
using Optim: BFGS, optimize, minimizer
using Statistics: mean, std, var
import StatsBase: fit, loglikelihood

export
    UnivariateModel,

    # Mean models
    AR,
    ARMA,
    ConstantMean,
    MA,
    VolInMean,
    ZeroMean,

    # Variance models
    ARCH,
    GARCH,
    ConstantVariance,

    # Functions
    fit,
    loglikelihood,
    simulate

abstract type ConditionalMeanModel{T} end
abstract type ConditionalVarianceModel{T} end

include("UnivariateModel.jl")
include("ConditionalMean/ZeroMean.jl")
include("ConditionalMean/ConstantMean.jl")
include("ConditionalMean/VolInMean.jl")
include("ConditionalMean/ARMA.jl")
include("ConditionalVariance/ConstantVariance.jl")
include("ConditionalVariance/GARCH.jl")
include("distributions.jl")
include("fit.jl")
include("simulate.jl")

end # module
