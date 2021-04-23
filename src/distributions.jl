standardized(d::Normal) = Normal(0.0, 1.0)
standardized(d::Distribution) = LocationScale(-mean(d), 1/std(d), d; check_args=false)

initial_coefficients(::Type{Normal}, y::Vector{T}) where {T} = T[0.0, 1.0]
initial_coefficients(::Type{TDist}, y::Vector{T}) where {T} = T(length(y))

function is_valid(d::TDist)
    d.ν > 0
end
function is_valid(d::Distribution)
    if d isa LocationScale
        is_valid(d.ρ) && (mean(d) ≈ 0) && (std(d) ≈ 1) 
    else
        (mean(d) ≈ 0) && (std(d) ≈ 1) 
    end
end
