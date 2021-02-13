using TimeSeriesModels, Test

@testset "All tests" begin
    (mean, variance) = fit(SampleMean, [0.1, 0.2, 0.3])
    @test mean.μ̄ ≈ 0.2
end
