@isdefined(AxSDPJointALMDesignQCQPData) || include("BnB_PEP_axsdp_joint_design.jl")


function _check_param_template_bundles(
    bundles::Vector{Vector{Matrix{Float64}}},
    nG::Int,
    num_params::Int,
    name::AbstractString,
)
    for mats in bundles
        length(mats) == num_params || error("Each template bundle in $name must have $num_params matrices.")
        for A in mats
            size(A) == (nG, nG) || error("Every matrix in $name must be of size $nG-by-$nG.")
        end
    end
    return nothing
end


function _add_param_bundle_sequence_entry(
    expr,
    vars,
    const_mats::Vector{Matrix{Float64}},
    param_bundles::Vector{Vector{Matrix{Float64}}},
    param_vars,
    i::Int,
    j::Int,
)
    for t in eachindex(const_mats)
        coeff_const = const_mats[t][i, j]
        if coeff_const != 0.0
            expr += coeff_const * vars[t]
        end

        mats_t = param_bundles[t]
        for s in eachindex(mats_t)
            coeff_param = mats_t[s][i, j]
            if coeff_param != 0.0
                expr += coeff_param * vars[t] * param_vars[s]
            end
        end
    end
    return expr
end


function _default_box_upper(values; min_bound::Real = 1.0, scale::Real = 2.0)
    isempty(values) && return float(min_bound)
    return max(float(min_bound), float(scale) * maximum(abs, values))
end


function _triangular_pairs_zero_based(N::Int)
    pairs = Tuple{Int, Int}[]
    for k in 1:N
        for i in 0:(k - 1)
            push!(pairs, (k, i))
        end
    end
    return pairs
end


function _triangular_pairs_one_based(N::Int)
    pairs = Tuple{Int, Int}[]
    for k in 1:N
        for i in 1:k
            push!(pairs, (k, i))
        end
    end
    return pairs
end


function _set_flat_start!(vars, values)
    isnothing(values) && return nothing
    length(vars) == length(values) || error("Warm-start length mismatch.")
    for i in eachindex(vars)
        set_start_value(vars[i], values[i])
    end
    return nothing
end


function _vector_or_nothing(vars, has_solution::Bool)
    return has_solution ? value.(vars) : nothing
end


function _matrix_or_nothing(vars, has_solution::Bool)
    return has_solution ? value.(vars) : nothing
end

