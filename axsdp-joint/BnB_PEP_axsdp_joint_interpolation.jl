using JuMP, LinearAlgebra, Gurobi
import Clarabel

const MOI = JuMP.MOI

const HAS_MOSEK = let
    try
        @eval using MosekTools
        @eval using Mosek
        true
    catch err
        @warn "Mosek is unavailable; using Clarabel for SDP solves." exception = (err, catch_backtrace())
        false
    end
end

make_sdp_model() = HAS_MOSEK ? Model(optimizer_with_attributes(Mosek.Optimizer)) : Model(Clarabel.Optimizer)

# Reuse the same PSD-to-Cholesky helper already used by the function-value BnB-PEP code.
include(joinpath(@__DIR__, "..", "function-value", "code_to_compute_pivoted_cholesky.jl"))


"""
Data for the AxSDP joint-interpolation model.

The slides define the operator block with

    X_Q = [x_star, x_0, ..., x_N, xbar, y_star, y_0, ..., y_N]
    Y_Q = [0, Ax_0, ..., Ax_N, A xbar, -g_star, Ay_0, ..., Ay_N].

In the Julia implementation below we introduce two explicit auxiliary atoms,
`Ax_star` and `Ay_star`, and impose the linear identities

    Ax_star = 0,
    g_star + Ay_star = 0.

This keeps the dual/operator formulas identical to the older AxSDP code while
remaining exactly equivalent to the joint-interpolation statement in the slides.
"""
Base.@kwdef struct AxSDPJointDualQCQPData
    C::Matrix{Float64}
    c::Vector{Float64}
    Aeq_x::Vector{Matrix{Float64}}
    Aeq_y::Vector{Matrix{Float64}}
    Aeq_barx::Vector{Matrix{Float64}}
    Aeq_Abarx::Vector{Matrix{Float64}}
    Aeq_Ax::Vector{Matrix{Float64}} = Matrix{Float64}[]
    Aeq_gAy::Vector{Matrix{Float64}} = Matrix{Float64}[]
    Aint::Vector{Matrix{Float64}}
    aint::Vector{Vector{Float64}}
    Ax0::Matrix{Float64}
    Ay0::Matrix{Float64}
    U_idx::Vector{Int}
    AU_idx::Vector{Int}
    mu_A::Float64
    L_A::Float64
    Rx2::Float64 = 1.0
    Ry2::Float64 = 1.0
end


Base.@kwdef struct AxSDPJointInstanceMetadata
    N::Int
    nG::Int
    nf::Int
    atom_index::NamedTuple
    value_index::NamedTuple
end


"""
Build one concrete ALM-like coefficient family.

The recurrence is

    x_1 = x_0 - (1 / eta) g_0 + (1 / eta) Ay_0
    x_k = x_{k-1} - (1 / eta) g_{k-1} + (2 / eta) Ay_{k-1} - (1 / eta) Ay_{k-2}
    y_k = y_{k-1} - rho_dual * Ax_k
    xbar = (1 / N) * sum_{k=1}^N x_k.
"""
function build_alm_like_coefficients(; N::Int, eta::Real, rho_dual::Real)
    N >= 1 || error("N must be at least 1.")
    eta > 0 || error("eta must be positive.")
    rho_dual > 0 || error("rho_dual must be positive.")

    alpha = zeros(Float64, N + 1, N + 1)
    rho_xy = zeros(Float64, N + 1, N + 1)
    beta = zeros(Float64, N + 1, N + 1)

    for k in 1:N
        alpha[k + 1, :] .= alpha[k, :]
        rho_xy[k + 1, :] .= rho_xy[k, :]

        # Column `i + 1` corresponds to gradient/operator atom with iteration index `i`.
        alpha[k + 1, k] += -1.0 / eta
        if k == 1
            rho_xy[k + 1, 1] += 1.0 / eta
        else
            rho_xy[k + 1, k] += 2.0 / eta
            rho_xy[k + 1, k - 1] += -1.0 / eta
        end

        # y_k = y_0 - rho_dual * sum_{i = 1}^k Ax_i.
        for i in 1:k
            beta[k + 1, i + 1] = -rho_dual
        end
    end

    omega = [0.0; fill(1.0 / N, N)]
    return alpha, rho_xy, beta, omega
end


"""
Build a small default AxSDP instance that mirrors the earlier ALM examples.
"""
function build_default_alm_joint_instance(;
    N::Int = 1,
    smoothness_L::Real = 1.0,
    mu_A::Real = 0.1,
    L_A::Real = 1.0,
    rho_dual::Real = 1.0,
    eta::Union{Nothing, Real} = nothing,
    Rx2::Real = 1.0,
    Ry2::Real = 1.0,
)
    eta_value = isnothing(eta) ? max(2.0 * rho_dual * L_A, 2.0 * smoothness_L) : float(eta)
    alpha, rho_xy, beta, omega = build_alm_like_coefficients(
        N = N,
        eta = eta_value,
        rho_dual = rho_dual,
    )
    return build_axsdp_joint_dual_qcqp_data(
        alpha = alpha,
        rho_xy = rho_xy,
        beta = beta,
        omega = omega,
        smoothness_L = smoothness_L,
        mu_A = mu_A,
        L_A = L_A,
        Rx2 = Rx2,
        Ry2 = Ry2,
    )
end


"""
Assemble the actual linear-algebra data for the joint-interpolation AxSDP model.
"""
function build_axsdp_joint_dual_qcqp_data(;
    alpha,
    rho_xy,
    beta,
    omega,
    smoothness_L::Real,
    mu_A::Real,
    L_A::Real,
    Rx2::Real = 1.0,
    Ry2::Real = 1.0,
)
    smoothness_L > 0 || error("smoothness_L must be positive.")
    mu_A >= 0 || error("mu_A must be nonnegative.")
    L_A > 0 || error("L_A must be positive.")
    mu_A <= L_A || error("mu_A must be at most L_A.")

    omega_vec = Float64.(collect(omega))
    length(omega_vec) >= 2 || error("omega must have length at least 2, corresponding to x_0, ..., x_N.")

    N = length(omega_vec) - 1
    alpha_mat = _to_dense_square(alpha, N, "alpha")
    rho_mat = _to_dense_square(rho_xy, N, "rho_xy")
    beta_mat = _to_dense_square(beta, N, "beta")

    atom_index = _build_joint_atom_index(N)
    value_index = _build_value_index(N)
    nG = atom_index.nG
    nf = value_index.nf

    C = zeros(Float64, nG, nG)
    c = zeros(Float64, nf)
    c[value_index.f_star] = -1.0
    c[value_index.f_bar] = 1.0

    # Objective: f(xbar) - f(x_star) - <A xbar, y_star>.
    _add_sym_entry!(C, atom_index.Abarx, atom_index.y_star, -1.0)

    Aeq_x = Matrix{Float64}[]
    Aeq_y = Matrix{Float64}[]
    Aeq_barx = Matrix{Float64}[]
    Aeq_Abarx = Matrix{Float64}[]
    Aeq_Ax = Matrix{Float64}[]
    Aeq_gAy = Matrix{Float64}[]

    # Each vector identity is tested against every atom q, which turns it into
    # one linear equation <A, G> = 0.
    for k in 1:N
        x_k = atom_index.x_iter[k + 1]
        x_0 = atom_index.x_iter[1]
        y_k = atom_index.y_iter[k + 1]
        y_0 = atom_index.y_iter[1]

        for q in 1:nG
            A = zeros(Float64, nG, nG)
            _add_sym_entry!(A, q, x_k, +1.0)
            _add_sym_entry!(A, q, x_0, -1.0)
            for i in 0:(k - 1)
                _add_sym_entry!(A, q, atom_index.g_iter[i + 1], -alpha_mat[k + 1, i + 1])
                _add_sym_entry!(A, q, atom_index.Ay_iter[i + 1], -rho_mat[k + 1, i + 1])
            end
            push!(Aeq_x, A)

            A = zeros(Float64, nG, nG)
            _add_sym_entry!(A, q, y_k, +1.0)
            _add_sym_entry!(A, q, y_0, -1.0)
            for i in 1:k
                _add_sym_entry!(A, q, atom_index.Ax_iter[i + 1], -beta_mat[k + 1, i + 1])
            end
            push!(Aeq_y, A)
        end
    end

    for q in 1:nG
        # Explicitly enforce the two anchor columns in Y_Q:
        #   first column  = 0     <=> Ax_star = 0,
        #   star-y column = -g_*  <=> Ay_star = -g_star.
        A = zeros(Float64, nG, nG)
        _add_sym_entry!(A, q, atom_index.Ax_star, +1.0)
        push!(Aeq_Ax, A)

        A = zeros(Float64, nG, nG)
        _add_sym_entry!(A, q, atom_index.g_star, +1.0)
        _add_sym_entry!(A, q, atom_index.Ay_star, +1.0)
        push!(Aeq_gAy, A)

        A = zeros(Float64, nG, nG)
        _add_sym_entry!(A, q, atom_index.x_bar, +1.0)
        for i in 0:N
            _add_sym_entry!(A, q, atom_index.x_iter[i + 1], -omega_vec[i + 1])
        end
        push!(Aeq_barx, A)

        A = zeros(Float64, nG, nG)
        _add_sym_entry!(A, q, atom_index.Abarx, +1.0)
        for i in 0:N
            _add_sym_entry!(A, q, atom_index.Ax_iter[i + 1], -omega_vec[i + 1])
        end
        push!(Aeq_Abarx, A)
    end

    # Smooth convex interpolation on J = {star, 0, ..., N, bar}.
    Aint = Matrix{Float64}[]
    aint = Vector{Float64}[]
    xJ = [atom_index.x_star; atom_index.x_iter; atom_index.x_bar]
    gJ = [atom_index.g_star; atom_index.g_iter; atom_index.g_bar]
    fJ = [value_index.f_star; value_index.f_iter; value_index.f_bar]

    for i in eachindex(xJ)
        for j in eachindex(xJ)
            A = zeros(Float64, nG, nG)
            a = zeros(Float64, nf)

            a[fJ[i]] += 1.0
            a[fJ[j]] -= 1.0

            _add_sym_entry!(A, gJ[j], xJ[i], -1.0)
            _add_sym_entry!(A, gJ[j], xJ[j], +1.0)
            _add_sym_entry!(A, gJ[i], gJ[i], -1.0 / (2.0 * smoothness_L))
            _add_sym_entry!(A, gJ[i], gJ[j], +1.0 / smoothness_L)
            _add_sym_entry!(A, gJ[j], gJ[j], -1.0 / (2.0 * smoothness_L))

            push!(Aint, A)
            push!(aint, a)
        end
    end

    # Radius bounds ||x_0 - x_star||^2 <= Rx2 and ||y_0 - y_star||^2 <= Ry2.
    Ax0 = zeros(Float64, nG, nG)
    _add_sym_entry!(Ax0, atom_index.x_iter[1], atom_index.x_iter[1], +1.0)
    _add_sym_entry!(Ax0, atom_index.x_star, atom_index.x_star, +1.0)
    _add_sym_entry!(Ax0, atom_index.x_iter[1], atom_index.x_star, -2.0)

    Ay0 = zeros(Float64, nG, nG)
    _add_sym_entry!(Ay0, atom_index.y_iter[1], atom_index.y_iter[1], +1.0)
    _add_sym_entry!(Ay0, atom_index.y_star, atom_index.y_star, +1.0)
    _add_sym_entry!(Ay0, atom_index.y_iter[1], atom_index.y_star, -2.0)

    # These are the operator-interpolation blocks used in the old AxSDP dual:
    #
    #   U  = [x_star, x_0, ..., x_N, xbar, y_star, y_0, ..., y_N]
    #   AU = [Ax_star, Ax_0, ..., Ax_N, A xbar, Ay_star, Ay_0, ..., Ay_N].
    #
    # Because Ax_star = 0 and Ay_star = -g_star are enforced separately, this
    # AU block is exactly equivalent to the slide-level joint block Y_Q.
    U_idx = [atom_index.x_star; atom_index.x_iter; atom_index.x_bar; atom_index.y_star; atom_index.y_iter]
    AU_idx = [atom_index.Ax_star; atom_index.Ax_iter; atom_index.Abarx; atom_index.Ay_star; atom_index.Ay_iter]

    data = AxSDPJointDualQCQPData(
        C = C,
        c = c,
        Aeq_x = Aeq_x,
        Aeq_y = Aeq_y,
        Aeq_barx = Aeq_barx,
        Aeq_Abarx = Aeq_Abarx,
        Aeq_Ax = Aeq_Ax,
        Aeq_gAy = Aeq_gAy,
        Aint = Aint,
        aint = aint,
        Ax0 = Ax0,
        Ay0 = Ay0,
        U_idx = U_idx,
        AU_idx = AU_idx,
        mu_A = float(mu_A),
        L_A = float(L_A),
        Rx2 = float(Rx2),
        Ry2 = float(Ry2),
    )
    meta = AxSDPJointInstanceMetadata(
        N = N,
        nG = nG,
        nf = nf,
        atom_index = atom_index,
        value_index = value_index,
    )
    return data, meta
end


"""
Return a compact size summary for quick sanity checks.
"""
function axsdp_joint_instance_summary(data::AxSDPJointDualQCQPData, meta::AxSDPJointInstanceMetadata)
    return (
        N = meta.N,
        nG = meta.nG,
        nf = meta.nf,
        m = length(data.U_idx),
        num_Aeq_x = length(data.Aeq_x),
        num_Aeq_y = length(data.Aeq_y),
        num_Aeq_barx = length(data.Aeq_barx),
        num_Aeq_Abarx = length(data.Aeq_Abarx),
        num_Aeq_Ax = length(data.Aeq_Ax),
        num_Aeq_gAy = length(data.Aeq_gAy),
        num_Aint = length(data.Aint),
    )
end


"""
Solve the direct convex dual SDP.

This is the cleanest way to verify the AxSDP formulation for fixed algorithm
coefficients. The QCQP/BnB-PEP-style model below is built on top of this
solution and uses it as a warm-start source.
"""
function solve_axsdp_joint_dual_sdp(data::AxSDPJointDualQCQPData; show_output::Symbol = :off)
    ctx = _validate_input_data(data)
    model = make_sdp_model()
    _apply_show_output!(model, show_output)

    lambda_x = _add_free_vector(model, length(data.Aeq_x), "lambda_x")
    lambda_y = _add_free_vector(model, length(data.Aeq_y), "lambda_y")
    lambda_barx = _add_free_vector(model, length(data.Aeq_barx), "lambda_barx")
    lambda_Abarx = _add_free_vector(model, length(data.Aeq_Abarx), "lambda_Abarx")
    lambda_Ax = _add_free_vector(model, length(data.Aeq_Ax), "lambda_Ax")
    lambda_gAy = _add_free_vector(model, length(data.Aeq_gAy), "lambda_gAy")
    nu = _add_nonnegative_vector(model, length(data.Aint), "nu")

    @variable(model, mu_x >= 0)
    @variable(model, mu_y >= 0)
    @variable(model, M[1:ctx.m, 1:ctx.m])
    @variable(model, Z_minus[1:ctx.m, 1:ctx.m], PSD)
    @variable(model, Z_plus[1:ctx.m, 1:ctx.m], PSD)
    @variable(model, W[1:ctx.nG, 1:ctx.nG], PSD)

    for r in 1:ctx.nf
        expr = data.c[r]
        for t in eachindex(data.Aint)
            coeff = data.aint[t][r]
            if coeff != 0.0
                expr += coeff * nu[t]
            end
        end
        @constraint(model, expr == 0)
    end

    for i in 1:ctx.nG
        for j in 1:i
            expr = data.C[i, j]
            expr = _add_matrix_sequence_entry(expr, lambda_x, data.Aeq_x, i, j)
            expr = _add_matrix_sequence_entry(expr, lambda_y, data.Aeq_y, i, j)
            expr = _add_matrix_sequence_entry(expr, lambda_barx, data.Aeq_barx, i, j)
            expr = _add_matrix_sequence_entry(expr, lambda_Abarx, data.Aeq_Abarx, i, j)
            expr = _add_matrix_sequence_entry(expr, lambda_Ax, data.Aeq_Ax, i, j)
            expr = _add_matrix_sequence_entry(expr, lambda_gAy, data.Aeq_gAy, i, j)
            expr = _add_matrix_sequence_entry(expr, nu, data.Aint, i, j)

            if data.Ax0[i, j] != 0.0
                expr += -data.Ax0[i, j] * mu_x
            end
            if data.Ay0[i, j] != 0.0
                expr += -data.Ay0[i, j] * mu_y
            end

            expr += _e_star_entry(M, ctx.u_pos, ctx.au_pos, i, j)
            expr += _l_minus_entry(Z_minus, ctx.u_pos, ctx.au_pos, data.mu_A, i, j)
            expr += _l_plus_entry(Z_plus, ctx.u_pos, ctx.au_pos, data.L_A, i, j)
            expr += W[i, j]

            @constraint(model, expr == 0)
        end
    end

    @objective(model, Min, data.Rx2 * mu_x + data.Ry2 * mu_y)
    optimize!(model)

    solution = _collect_sdp_solution(
        model,
        lambda_x,
        lambda_y,
        lambda_barx,
        lambda_Abarx,
        lambda_Ax,
        lambda_gAy,
        nu,
        mu_x,
        mu_y,
        M,
        Z_minus,
        Z_plus,
        W,
    )
    return (model = model, solution = solution)
end


"""
Solve the BnB-PEP-style nonconvex QCQP with explicit PSD slacks and explicit
Cholesky-style factors.

This mirrors the structure of the Julia BnB-PEP codes:

1. keep explicit symmetric variables `Z_minus`, `Z_plus`, `W`,
2. keep lower-triangular factors `P_minus`, `P_plus`, `S`,
3. connect them through quadratic equalities, and
4. optionally add simple linear PSD-valid cuts.
"""
function solve_axsdp_joint_dual_qcqp(
    data::AxSDPJointDualQCQPData;
    show_output::Symbol = :off,
    factor_bound::Union{Nothing, Real} = nothing,
    psd_bound::Union{Nothing, Real} = nothing,
    equality_tolerance::Real = 0.0,
    add_psd_cuts::Bool = true,
    warm_start = nothing,
    gurobi_params = Dict{String, Any}(),
)
    ctx = _validate_input_data(data)
    equality_tolerance >= 0 || error("equality_tolerance must be nonnegative.")
    isnothing(factor_bound) || factor_bound > 0 || error("factor_bound must be positive when provided.")
    isnothing(psd_bound) || psd_bound > 0 || error("psd_bound must be positive when provided.")

    model = Model(Gurobi.Optimizer)
    if show_output == :off && !haskey(gurobi_params, "OutputFlag")
        set_attribute(model, "OutputFlag", 0)
    end
    for (key, value) in pairs(gurobi_params)
        set_attribute(model, key, value)
    end
    set_attribute(model, "NonConvex", 2)

    lambda_x = _add_free_vector(model, length(data.Aeq_x), "lambda_x")
    lambda_y = _add_free_vector(model, length(data.Aeq_y), "lambda_y")
    lambda_barx = _add_free_vector(model, length(data.Aeq_barx), "lambda_barx")
    lambda_Abarx = _add_free_vector(model, length(data.Aeq_Abarx), "lambda_Abarx")
    lambda_Ax = _add_free_vector(model, length(data.Aeq_Ax), "lambda_Ax")
    lambda_gAy = _add_free_vector(model, length(data.Aeq_gAy), "lambda_gAy")
    nu = _add_nonnegative_vector(model, length(data.Aint), "nu")

    @variable(model, mu_x >= 0)
    @variable(model, mu_y >= 0)
    @variable(model, M[1:ctx.m, 1:ctx.m], base_name = "M")

    z_operator_bound = _resolve_psd_bound(psd_bound, factor_bound, ctx.m)
    w_bound = _resolve_psd_bound(psd_bound, factor_bound, ctx.nG)

    Z_minus = _add_symmetric_matrix_vars(model, ctx.m, "Z_minus", z_operator_bound)
    Z_plus = _add_symmetric_matrix_vars(model, ctx.m, "Z_plus", z_operator_bound)
    W = _add_symmetric_matrix_vars(model, ctx.nG, "W", w_bound)

    P_minus = _add_lower_triangular_factor(model, ctx.m, "P_minus", factor_bound)
    P_plus = _add_lower_triangular_factor(model, ctx.m, "P_plus", factor_bound)
    S = _add_lower_triangular_factor(model, ctx.nG, "S", factor_bound)

    if add_psd_cuts
        _add_psd_valid_cuts(model, Z_minus)
        _add_psd_valid_cuts(model, Z_plus)
        _add_psd_valid_cuts(model, W)
    end

    _add_cholesky_formula_constraints(model, Z_minus, P_minus, equality_tolerance)
    _add_cholesky_formula_constraints(model, Z_plus, P_plus, equality_tolerance)
    _add_cholesky_formula_constraints(model, W, S, equality_tolerance)

    _apply_qcqp_warm_start!(
        lambda_x,
        lambda_y,
        lambda_barx,
        lambda_Abarx,
        lambda_Ax,
        lambda_gAy,
        nu,
        mu_x,
        mu_y,
        M,
        Z_minus,
        Z_plus,
        W,
        P_minus,
        P_plus,
        S,
        warm_start,
    )

    for r in 1:ctx.nf
        expr = data.c[r]
        for t in eachindex(data.Aint)
            coeff = data.aint[t][r]
            if coeff != 0.0
                expr += coeff * nu[t]
            end
        end
        _add_scalar_equality!(model, expr, equality_tolerance)
    end

    for i in 1:ctx.nG
        for j in 1:i
            expr = data.C[i, j]
            expr = _add_matrix_sequence_entry(expr, lambda_x, data.Aeq_x, i, j)
            expr = _add_matrix_sequence_entry(expr, lambda_y, data.Aeq_y, i, j)
            expr = _add_matrix_sequence_entry(expr, lambda_barx, data.Aeq_barx, i, j)
            expr = _add_matrix_sequence_entry(expr, lambda_Abarx, data.Aeq_Abarx, i, j)
            expr = _add_matrix_sequence_entry(expr, lambda_Ax, data.Aeq_Ax, i, j)
            expr = _add_matrix_sequence_entry(expr, lambda_gAy, data.Aeq_gAy, i, j)
            expr = _add_matrix_sequence_entry(expr, nu, data.Aint, i, j)

            if data.Ax0[i, j] != 0.0
                expr += -data.Ax0[i, j] * mu_x
            end
            if data.Ay0[i, j] != 0.0
                expr += -data.Ay0[i, j] * mu_y
            end

            expr += _e_star_entry(M, ctx.u_pos, ctx.au_pos, i, j)
            expr += _l_minus_entry(Z_minus, ctx.u_pos, ctx.au_pos, data.mu_A, i, j)
            expr += _l_plus_entry(Z_plus, ctx.u_pos, ctx.au_pos, data.L_A, i, j)
            expr += W[i, j]

            _add_scalar_equality!(model, expr, equality_tolerance)
        end
    end

    @objective(model, Min, data.Rx2 * mu_x + data.Ry2 * mu_y)
    optimize!(model)

    solution = _collect_qcqp_solution(
        model,
        lambda_x,
        lambda_y,
        lambda_barx,
        lambda_Abarx,
        lambda_Ax,
        lambda_gAy,
        nu,
        mu_x,
        mu_y,
        M,
        Z_minus,
        Z_plus,
        W,
        P_minus,
        P_plus,
        S,
    )
    return (model = model, solution = solution)
end


"""
Convert the SDP solution into a QCQP warm start and estimate reasonable bounds
for the factorized model.
"""
function build_qcqp_warm_start_from_sdp_solution(
    sdp_solution;
    chol_tol::Real = 1e-8,
    scale::Real = 1.01,
    min_bound::Real = 1.0,
)
    sdp_solution.objective === nothing && error("The SDP solution does not contain a primal point.")

    P_minus = compute_pivoted_cholesky_L_mat(sdp_solution.Z_minus; ϵ_tol = chol_tol)
    P_plus = compute_pivoted_cholesky_L_mat(sdp_solution.Z_plus; ϵ_tol = chol_tol)
    S = compute_pivoted_cholesky_L_mat(sdp_solution.W; ϵ_tol = chol_tol)

    factor_bound = scale * maximum((
        float(min_bound),
        maximum(abs, P_minus),
        maximum(abs, P_plus),
        maximum(abs, S),
    ))
    psd_bound = scale * maximum((
        float(min_bound),
        maximum(abs, sdp_solution.Z_minus),
        maximum(abs, sdp_solution.Z_plus),
        maximum(abs, sdp_solution.W),
    ))

    return (
        lambda_x = copy(sdp_solution.lambda_x),
        lambda_y = copy(sdp_solution.lambda_y),
        lambda_barx = copy(sdp_solution.lambda_barx),
        lambda_Abarx = copy(sdp_solution.lambda_Abarx),
        lambda_Ax = copy(sdp_solution.lambda_Ax),
        lambda_gAy = copy(sdp_solution.lambda_gAy),
        nu = copy(sdp_solution.nu),
        mu_x = sdp_solution.mu_x,
        mu_y = sdp_solution.mu_y,
        M = copy(sdp_solution.M),
        Z_minus = copy(sdp_solution.Z_minus),
        Z_plus = copy(sdp_solution.Z_plus),
        W = copy(sdp_solution.W),
        P_minus = P_minus,
        P_plus = P_plus,
        S = S,
        factor_bound = factor_bound,
        psd_bound = psd_bound,
    )
end


function estimate_qcqp_bounds_from_sdp_solution(sdp_solution; kwargs...)
    warm = build_qcqp_warm_start_from_sdp_solution(sdp_solution; kwargs...)
    return (factor_bound = warm.factor_bound, psd_bound = warm.psd_bound)
end


function _build_joint_atom_index(N::Int)
    cursor = 1

    x_star = cursor
    cursor += 1

    x_iter = collect(cursor:(cursor + N))
    cursor += N + 1

    x_bar = cursor
    cursor += 1

    g_star = cursor
    cursor += 1

    g_iter = collect(cursor:(cursor + N))
    cursor += N + 1

    g_bar = cursor
    cursor += 1

    y_star = cursor
    cursor += 1

    y_iter = collect(cursor:(cursor + N))
    cursor += N + 1

    # These two auxiliary atoms are not part of the slide-level atom set. They
    # are introduced only to keep the operator-dual formulas identical to the
    # older AxSDP derivation.
    Ax_star = cursor
    cursor += 1

    Ax_iter = collect(cursor:(cursor + N))
    cursor += N + 1

    Abarx = cursor
    cursor += 1

    Ay_star = cursor
    cursor += 1

    Ay_iter = collect(cursor:(cursor + N))
    cursor += N + 1

    return (
        x_star = x_star,
        x_iter = x_iter,
        x_bar = x_bar,
        g_star = g_star,
        g_iter = g_iter,
        g_bar = g_bar,
        y_star = y_star,
        y_iter = y_iter,
        Ax_star = Ax_star,
        Ax_iter = Ax_iter,
        Abarx = Abarx,
        Ay_star = Ay_star,
        Ay_iter = Ay_iter,
        nG = cursor - 1,
    )
end


function _build_value_index(N::Int)
    f_star = 1
    f_iter = collect(2:(N + 2))
    f_bar = N + 3
    return (
        f_star = f_star,
        f_iter = f_iter,
        f_bar = f_bar,
        nf = N + 3,
    )
end


function _to_dense_square(coeffs, N::Int, name::AbstractString)
    if coeffs isa AbstractMatrix
        A = Matrix{Float64}(coeffs)
    elseif coeffs isa AbstractVector
        length(coeffs) == N + 1 || error("$name must have $(N + 1) rows.")
        A = zeros(Float64, N + 1, N + 1)
        for i in 1:(N + 1)
            row = collect(coeffs[i])
            length(row) == N + 1 || error("Each row of $name must have length $(N + 1).")
            A[i, :] .= Float64.(row)
        end
    else
        error("$name must be a square matrix or a vector of rows.")
    end
    size(A) == (N + 1, N + 1) || error("$name must be of size $(N + 1)-by-$(N + 1).")
    return A
end


function _add_sym_entry!(A::Matrix{Float64}, i::Int, j::Int, value::Real)
    value_f = float(value)
    if i == j
        A[i, j] += value_f
    else
        half = 0.5 * value_f
        A[i, j] += half
        A[j, i] += half
    end
    return A
end


function _validate_input_data(data::AxSDPJointDualQCQPData)
    nG = size(data.C, 1)
    nG > 0 || error("C must be nonempty.")
    size(data.C, 2) == nG || error("C must be square.")

    nf = length(data.c)
    size(data.Ax0) == (nG, nG) || error("Ax0 must be an nG-by-nG matrix.")
    size(data.Ay0) == (nG, nG) || error("Ay0 must be an nG-by-nG matrix.")

    for (name, mats) in (
        ("Aeq_x", data.Aeq_x),
        ("Aeq_y", data.Aeq_y),
        ("Aeq_barx", data.Aeq_barx),
        ("Aeq_Abarx", data.Aeq_Abarx),
        ("Aeq_Ax", data.Aeq_Ax),
        ("Aeq_gAy", data.Aeq_gAy),
        ("Aint", data.Aint),
    )
        for A in mats
            size(A) == (nG, nG) || error("Every matrix in $name must be of size $nG-by-$nG.")
        end
    end

    length(data.Aint) == length(data.aint) || error("Aint and aint must have the same length.")
    for a in data.aint
        length(a) == nf || error("Every vector in aint must have the same length as c.")
    end

    length(data.U_idx) == length(data.AU_idx) || error("U_idx and AU_idx must have the same length.")
    !isempty(data.U_idx) || error("U_idx and AU_idx must be nonempty.")
    all(idx -> 1 <= idx <= nG, data.U_idx) || error("U_idx contains an invalid atom index.")
    all(idx -> 1 <= idx <= nG, data.AU_idx) || error("AU_idx contains an invalid atom index.")
    length(unique(data.U_idx)) == length(data.U_idx) || error("U_idx must not contain duplicates.")
    length(unique(data.AU_idx)) == length(data.AU_idx) || error("AU_idx must not contain duplicates.")

    data.mu_A >= 0 || error("mu_A must be nonnegative.")
    data.L_A > 0 || error("L_A must be positive.")
    data.mu_A <= data.L_A || error("mu_A must be at most L_A.")
    data.Rx2 >= 0 || error("Rx2 must be nonnegative.")
    data.Ry2 >= 0 || error("Ry2 must be nonnegative.")

    u_pos = Dict(data.U_idx[p] => p for p in eachindex(data.U_idx))
    au_pos = Dict(data.AU_idx[p] => p for p in eachindex(data.AU_idx))
    return (nG = nG, nf = nf, m = length(data.U_idx), u_pos = u_pos, au_pos = au_pos)
end


function _apply_show_output!(model::Model, show_output::Symbol)
    if show_output == :off
        set_silent(model)
    elseif show_output != :on
        error("show_output must be either :on or :off.")
    end
    return nothing
end


function _add_free_vector(model::Model, n::Int, base_name::AbstractString)
    if n == 0
        return VariableRef[]
    end
    return @variable(model, [1:n], base_name = base_name)
end


function _add_nonnegative_vector(model::Model, n::Int, base_name::AbstractString)
    if n == 0
        return VariableRef[]
    end
    return @variable(model, [1:n], lower_bound = 0.0, base_name = base_name)
end


function _add_symmetric_matrix_vars(
    model::Model,
    n::Int,
    prefix::AbstractString,
    bound::Union{Nothing, Real},
)
    X = Matrix{VariableRef}(undef, n, n)
    lower = isnothing(bound) ? -Inf : -float(bound)
    upper = isnothing(bound) ? Inf : float(bound)

    for i in 1:n
        for j in 1:i
            var = @variable(
                model,
                lower_bound = lower,
                upper_bound = upper,
                base_name = "$(prefix)_$(i)_$(j)",
            )
            X[i, j] = var
            X[j, i] = var
        end
    end
    return X
end


function _add_lower_triangular_factor(
    model::Model,
    n::Int,
    prefix::AbstractString,
    bound::Union{Nothing, Real},
)
    P = Matrix{Union{Nothing, VariableRef}}(undef, n, n)
    fill!(P, nothing)

    for i in 1:n
        for j in 1:i
            if i == j
                lower = 0.0
                upper = isnothing(bound) ? Inf : float(bound)
            else
                lower = isnothing(bound) ? -Inf : -float(bound)
                upper = isnothing(bound) ? Inf : float(bound)
            end
            P[i, j] = @variable(
                model,
                lower_bound = lower,
                upper_bound = upper,
                base_name = "$(prefix)_$(i)_$(j)",
            )
        end
    end
    return P
end


function _add_psd_valid_cuts(model::Model, Z::Matrix{VariableRef})
    n = size(Z, 1)
    for i in 1:n
        @constraint(model, Z[i, i] >= 0)
    end
    for i in 2:n
        for j in 1:(i - 1)
            @constraint(model, Z[i, j] <= 0.5 * (Z[i, i] + Z[j, j]))
            @constraint(model, Z[i, j] >= -0.5 * (Z[i, i] + Z[j, j]))
        end
    end
    return nothing
end


function _add_cholesky_formula_constraints(
    model::Model,
    Z::Matrix{VariableRef},
    P::Matrix{Union{Nothing, VariableRef}},
    tolerance::Real,
)
    n = size(Z, 1)

    for j in 1:n
        expr = sum((P[j, k]::VariableRef)^2 for k in 1:j) - Z[j, j]
        _add_scalar_equality!(model, expr, tolerance)
    end

    for i in 2:n
        for j in 1:(i - 1)
            expr = sum((P[i, k]::VariableRef) * (P[j, k]::VariableRef) for k in 1:j) - Z[i, j]
            _add_scalar_equality!(model, expr, tolerance)
        end
    end
    return nothing
end


function _add_scalar_equality!(model::Model, expr, tolerance::Real)
    if tolerance == 0.0
        @constraint(model, expr == 0)
    else
        @constraint(model, expr <= tolerance)
        @constraint(model, expr >= -tolerance)
    end
    return nothing
end


function _add_matrix_sequence_entry(expr, vars, mats::Vector{Matrix{Float64}}, i::Int, j::Int)
    for t in eachindex(mats)
        coeff = mats[t][i, j]
        if coeff != 0.0
            expr += coeff * vars[t]
        end
    end
    return expr
end


function _e_star_entry(M, u_pos::Dict{Int, Int}, au_pos::Dict{Int, Int}, i::Int, j::Int)
    expr = 0.0
    if haskey(u_pos, i) && haskey(au_pos, j)
        a = u_pos[i]
        b = au_pos[j]
        expr += 0.5 * M[a, b]
        expr += -0.5 * M[b, a]
    end
    if haskey(au_pos, i) && haskey(u_pos, j)
        a = au_pos[i]
        b = u_pos[j]
        expr += 0.5 * M[b, a]
        expr += -0.5 * M[a, b]
    end
    return expr
end


function _l_minus_entry(Z_minus, u_pos::Dict{Int, Int}, au_pos::Dict{Int, Int}, mu_A::Real, i::Int, j::Int)
    expr = 0.0
    if haskey(au_pos, i) && haskey(au_pos, j)
        expr += Z_minus[au_pos[i], au_pos[j]]
    end
    if haskey(u_pos, i) && haskey(au_pos, j)
        expr += -0.5 * float(mu_A) * Z_minus[u_pos[i], au_pos[j]]
    end
    if haskey(au_pos, i) && haskey(u_pos, j)
        expr += -0.5 * float(mu_A) * Z_minus[au_pos[i], u_pos[j]]
    end
    return expr
end


function _l_plus_entry(Z_plus, u_pos::Dict{Int, Int}, au_pos::Dict{Int, Int}, L_A::Real, i::Int, j::Int)
    expr = 0.0
    if haskey(u_pos, i) && haskey(au_pos, j)
        expr += 0.5 * float(L_A) * Z_plus[u_pos[i], au_pos[j]]
    end
    if haskey(au_pos, i) && haskey(u_pos, j)
        expr += 0.5 * float(L_A) * Z_plus[au_pos[i], u_pos[j]]
    end
    if haskey(au_pos, i) && haskey(au_pos, j)
        expr += -Z_plus[au_pos[i], au_pos[j]]
    end
    return expr
end


function _resolve_psd_bound(
    psd_bound::Union{Nothing, Real},
    factor_bound::Union{Nothing, Real},
    dimension::Int,
)
    if !isnothing(psd_bound)
        return float(psd_bound)
    end
    if isnothing(factor_bound)
        return nothing
    end
    return float(dimension) * float(factor_bound)^2
end


function _apply_qcqp_warm_start!(
    lambda_x,
    lambda_y,
    lambda_barx,
    lambda_Abarx,
    lambda_Ax,
    lambda_gAy,
    nu,
    mu_x,
    mu_y,
    M,
    Z_minus,
    Z_plus,
    W,
    P_minus,
    P_plus,
    S,
    warm_start,
)
    isnothing(warm_start) && return nothing

    _set_vector_start!(lambda_x, _warm_get(warm_start, :lambda_x))
    _set_vector_start!(lambda_y, _warm_get(warm_start, :lambda_y))
    _set_vector_start!(lambda_barx, _warm_get(warm_start, :lambda_barx))
    _set_vector_start!(lambda_Abarx, _warm_get(warm_start, :lambda_Abarx))
    _set_vector_start!(lambda_Ax, _warm_get(warm_start, :lambda_Ax))
    _set_vector_start!(lambda_gAy, _warm_get(warm_start, :lambda_gAy))
    _set_vector_start!(nu, _warm_get(warm_start, :nu))

    _set_scalar_start!(mu_x, _warm_get(warm_start, :mu_x))
    _set_scalar_start!(mu_y, _warm_get(warm_start, :mu_y))
    _set_dense_matrix_start!(M, _warm_get(warm_start, :M))
    _set_symmetric_matrix_start!(Z_minus, _warm_get(warm_start, :Z_minus))
    _set_symmetric_matrix_start!(Z_plus, _warm_get(warm_start, :Z_plus))
    _set_symmetric_matrix_start!(W, _warm_get(warm_start, :W))
    _set_lower_triangular_start!(P_minus, _warm_get(warm_start, :P_minus))
    _set_lower_triangular_start!(P_plus, _warm_get(warm_start, :P_plus))
    _set_lower_triangular_start!(S, _warm_get(warm_start, :S))
    return nothing
end


function _warm_get(warm_start, key::Symbol)
    if warm_start isa NamedTuple
        return key in keys(warm_start) ? getproperty(warm_start, key) : nothing
    elseif warm_start isa AbstractDict
        if haskey(warm_start, key)
            return warm_start[key]
        elseif haskey(warm_start, String(key))
            return warm_start[String(key)]
        else
            return nothing
        end
    else
        return hasproperty(warm_start, key) ? getproperty(warm_start, key) : nothing
    end
end


function _set_scalar_start!(var::VariableRef, value_start)
    isnothing(value_start) && return nothing
    set_start_value(var, float(value_start))
    return nothing
end


function _set_vector_start!(vars, values)
    isnothing(values) && return nothing
    length(vars) == length(values) || error("Warm-start vector has the wrong length.")
    for i in eachindex(vars)
        set_start_value(vars[i], float(values[i]))
    end
    return nothing
end


function _set_dense_matrix_start!(X, values)
    isnothing(values) && return nothing
    size(X) == size(values) || error("Warm-start matrix has the wrong size.")
    for i in axes(X, 1)
        for j in axes(X, 2)
            set_start_value(X[i, j], float(values[i, j]))
        end
    end
    return nothing
end


function _set_symmetric_matrix_start!(X::Matrix{VariableRef}, values)
    isnothing(values) && return nothing
    size(X) == size(values) || error("Warm-start symmetric matrix has the wrong size.")
    for i in axes(X, 1)
        for j in 1:i
            set_start_value(X[i, j], float(values[i, j]))
        end
    end
    return nothing
end


function _set_lower_triangular_start!(P::Matrix{Union{Nothing, VariableRef}}, values)
    isnothing(values) && return nothing
    size(P) == size(values) || error("Warm-start factor matrix has the wrong size.")
    for i in axes(P, 1)
        for j in 1:i
            var = P[i, j]
            if !isnothing(var)
                set_start_value(var, float(values[i, j]))
            end
        end
    end
    return nothing
end


function _has_primal_solution(model::Model)
    return result_count(model) > 0 && primal_status(model) != MOI.NO_SOLUTION
end


function _safe_objective_bound(model::Model)
    try
        return objective_bound(model)
    catch
        return nothing
    end
end


function _collect_sdp_solution(
    model::Model,
    lambda_x,
    lambda_y,
    lambda_barx,
    lambda_Abarx,
    lambda_Ax,
    lambda_gAy,
    nu,
    mu_x,
    mu_y,
    M,
    Z_minus,
    Z_plus,
    W,
)
    has_solution = _has_primal_solution(model)
    return (
        termination_status = termination_status(model),
        primal_status = primal_status(model),
        dual_status = dual_status(model),
        raw_status = raw_status(model),
        solve_time_sec = solve_time(model),
        result_count = result_count(model),
        objective = has_solution ? objective_value(model) : nothing,
        objective_bound = _safe_objective_bound(model),
        lambda_x = has_solution ? value.(lambda_x) : nothing,
        lambda_y = has_solution ? value.(lambda_y) : nothing,
        lambda_barx = has_solution ? value.(lambda_barx) : nothing,
        lambda_Abarx = has_solution ? value.(lambda_Abarx) : nothing,
        lambda_Ax = has_solution ? value.(lambda_Ax) : nothing,
        lambda_gAy = has_solution ? value.(lambda_gAy) : nothing,
        nu = has_solution ? value.(nu) : nothing,
        mu_x = has_solution ? value(mu_x) : nothing,
        mu_y = has_solution ? value(mu_y) : nothing,
        M = has_solution ? value.(M) : nothing,
        Z_minus = has_solution ? Matrix(value.(Z_minus)) : nothing,
        Z_plus = has_solution ? Matrix(value.(Z_plus)) : nothing,
        W = has_solution ? Matrix(value.(W)) : nothing,
    )
end


function _collect_qcqp_solution(
    model::Model,
    lambda_x,
    lambda_y,
    lambda_barx,
    lambda_Abarx,
    lambda_Ax,
    lambda_gAy,
    nu,
    mu_x,
    mu_y,
    M,
    Z_minus,
    Z_plus,
    W,
    P_minus,
    P_plus,
    S,
)
    has_solution = _has_primal_solution(model)
    return (
        termination_status = termination_status(model),
        primal_status = primal_status(model),
        dual_status = dual_status(model),
        raw_status = raw_status(model),
        solve_time_sec = solve_time(model),
        result_count = result_count(model),
        objective = has_solution ? objective_value(model) : nothing,
        objective_bound = _safe_objective_bound(model),
        lambda_x = has_solution ? value.(lambda_x) : nothing,
        lambda_y = has_solution ? value.(lambda_y) : nothing,
        lambda_barx = has_solution ? value.(lambda_barx) : nothing,
        lambda_Abarx = has_solution ? value.(lambda_Abarx) : nothing,
        lambda_Ax = has_solution ? value.(lambda_Ax) : nothing,
        lambda_gAy = has_solution ? value.(lambda_gAy) : nothing,
        nu = has_solution ? value.(nu) : nothing,
        mu_x = has_solution ? value(mu_x) : nothing,
        mu_y = has_solution ? value(mu_y) : nothing,
        M = has_solution ? value.(M) : nothing,
        Z_minus = has_solution ? _value_symmetric_matrix(Z_minus) : nothing,
        Z_plus = has_solution ? _value_symmetric_matrix(Z_plus) : nothing,
        W = has_solution ? _value_symmetric_matrix(W) : nothing,
        P_minus = has_solution ? _value_lower_triangular_matrix(P_minus) : nothing,
        P_plus = has_solution ? _value_lower_triangular_matrix(P_plus) : nothing,
        S = has_solution ? _value_lower_triangular_matrix(S) : nothing,
    )
end


function _value_symmetric_matrix(X::Matrix{VariableRef})
    n = size(X, 1)
    out = zeros(Float64, n, n)
    for i in 1:n
        for j in 1:i
            val = value(X[i, j])
            out[i, j] = val
            out[j, i] = val
        end
    end
    return out
end


function _value_lower_triangular_matrix(P::Matrix{Union{Nothing, VariableRef}})
    n = size(P, 1)
    out = zeros(Float64, n, n)
    for i in 1:n
        for j in 1:i
            out[i, j] = value(P[i, j]::VariableRef)
        end
    end
    return out
end
