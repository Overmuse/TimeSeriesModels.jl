using TimeSeriesModels, Test

@testset "All tests" begin
    model = fit(ConstantMean, [0.1, 0.2, 0.3])
    @test model.mean_model.μ̄ ≈ 0.2
end
