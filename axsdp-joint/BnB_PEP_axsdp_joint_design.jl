include("BnB_PEP_axsdp_joint_interpolation.jl")
using Ipopt


"""
Design-data container for the ALM-like AxSDP family where the algorithm
parameters are optimized jointly with the dual certificate.

Unlike `AxSDPJointDualQCQPData`, which assumes that `alpha`, `rho_xy`, `beta`,
and `omega` are fixed inputs, this struct stores a template decomposition:

    Aeq_x(tau)      = Aeq_x_const      + tau      * Aeq_x_tau
    Aeq_y(rho)      = Aeq_y_const      + rho      * Aeq_y_rho
    Aeq_barx(omega) = Aeq_barx_const   + sum_i omega_i * Aeq_barx_omega[i]
    Aeq_Abarx(...)  = Aeq_Abarx_const  + sum_i omega_i * Aeq_Abarx_omega[i]

This lets the final QCQP optimize both the AxSDP dual variables and the
algorithm parameters `(tau, rho_dual, omega)`.
"""
Base.@kwdef struct AxSDPJointALMDesignQCQPData
    N::Int
    smoothness_L::Float64
    mu_A::Float64
    L_A::Float64
    Rx2::Float64
    Ry2::Float64
    C::Matrix{Float64}
    c::Vector{Float64}
    Aeq_x_const::Vector{Matrix{Float64}}
    Aeq_x_tau::Vector{Matrix{Float64}}
    Aeq_y_const::Vector{Matrix{Float64}}
    Aeq_y_rho::Vector{Matrix{Float64}}
    Aeq_barx_const::Vector{Matrix{Float64}}
    Aeq_barx_omega::Vector{Vector{Matrix{Float64}}}
    Aeq_Abarx_const::Vector{Matrix{Float64}}
    Aeq_Abarx_omega::Vector{Vector{Matrix{Float64}}}
    Aeq_Ax::Vector{Matrix{Float64}}
    Aeq_gAy::Vector{Matrix{Float64}}
    Aint::Vector{Matrix{Float64}}
    aint::Vector{Vector{Float64}}
    Ax0::Matrix{Float64}
    Ay0::Matrix{Float64}
    U_idx::Vector{Int}
    AU_idx::Vector{Int}
    tau0::Float64
    rho_dual0::Float64
    omega0::Vector{Float64}
end


Base.@kwdef struct AxSDPJointALMDesignMetadata
    N::Int
    nG::Int
    nf::Int
    atom_index::NamedTuple
    value_index::NamedTuple
end


"""
Build the design-data templates for the ALM-like AxSDP family.

Algorithm family:

    x_1 = x_0 - tau * g_0 + tau * Ay_0
    x_k = x_{k-1} - tau * g_{k-1} + 2 tau * Ay_{k-1} - tau * Ay_{k-2},  k >= 2
    y_k = y_{k-1} - rho_dual * Ax_k
    xbar = sum_{i=0}^N omega_i x_i

Here `tau = 1 / eta`, `rho_dual` is the dual stepsize, and `omega` is the
averaging vector. These are the quantities that will become decision variables
in the final design QCQP.
"""
function build_axsdp_joint_alm_design_data(;
    N::Int,
    smoothness_L::Real,
    mu_A::Real,
    L_A::Real,
    Rx2::Real = 1.0,
    Ry2::Real = 1.0,
    tau0::Real,
    rho_dual0::Real,
    omega0,
)
    N >= 1 || error("N must be at least 1.")
    smoothness_L > 0 || error("smoothness_L must be positive.")
    mu_A >= 0 || error("mu_A must be nonnegative.")
    L_A > 0 || error("L_A must be positive.")
    mu_A <= L_A || error("mu_A must be at most L_A.")
    tau0 >= 0 || error("tau0 must be nonnegative.")
    rho_dual0 >= 0 || error("rho_dual0 must be nonnegative.")

    omega0_vec = Float64.(collect(omega0))
    length(omega0_vec) == N + 1 || error("omega0 must have length N + 1.")
    minimum(omega0_vec) >= -1e-10 || error("omega0 must be componentwise nonnegative.")
    abs(sum(omega0_vec) - 1.0) <= 1e-8 || error("omega0 must sum to 1.")

    atom_index = _build_joint_atom_index(N)
    value_index = _build_value_index(N)
    nG = atom_index.nG
    nf = value_index.nf

    C = zeros(Float64, nG, nG)
    c = zeros(Float64, nf)
    c[value_index.f_star] = -1.0
    c[value_index.f_bar] = 1.0
    _add_sym_entry!(C, atom_index.Abarx, atom_index.y_star, -1.0)

    Aeq_x_const = Matrix{Float64}[]
    Aeq_x_tau = Matrix{Float64}[]
    Aeq_y_const = Matrix{Float64}[]
    Aeq_y_rho = Matrix{Float64}[]
    Aeq_barx_const = Matrix{Float64}[]
    Aeq_barx_omega = Vector{Vector{Matrix{Float64}}}()
    Aeq_Abarx_const = Matrix{Float64}[]
    Aeq_Abarx_omega = Vector{Vector{Matrix{Float64}}}()
    Aeq_Ax = Matrix{Float64}[]
    Aeq_gAy = Matrix{Float64}[]

    # x-residuals:
    #
    #   r_1^(x) = x_1 - x_0 + tau * g_0 - tau * Ay_0
    #   r_k^(x) = x_k - x_{k-1} + tau * g_{k-1} - 2 tau * Ay_{k-1} + tau * Ay_{k-2}
    #
    # Each tested residual <q, r_k^(x)> = 0 produces one linear-equality
    # multiplier `lambda_x`.
    for k in 1:N
        x_k = atom_index.x_iter[k + 1]
        x_prev = atom_index.x_iter[k]
        g_prev = atom_index.g_iter[k]

        for q in 1:nG
            Aconst = zeros(Float64, nG, nG)
            Atau = zeros(Float64, nG, nG)

            _add_sym_entry!(Aconst, q, x_k, +1.0)
            _add_sym_entry!(Aconst, q, x_prev, -1.0)
            _add_sym_entry!(Atau, q, g_prev, +1.0)

            if k == 1
                _add_sym_entry!(Atau, q, atom_index.Ay_iter[1], -1.0)
            else
                _add_sym_entry!(Atau, q, atom_index.Ay_iter[k], -2.0)
                _add_sym_entry!(Atau, q, atom_index.Ay_iter[k - 1], +1.0)
            end

            push!(Aeq_x_const, Aconst)
            push!(Aeq_x_tau, Atau)
        end
    end

    # y-residuals:
    #
    #   r_k^(y) = y_k - y_{k-1} + rho_dual * Ax_k
    #
    # Again, each tested residual <q, r_k^(y)> = 0 gets its own multiplier.
    for k in 1:N
        y_k = atom_index.y_iter[k + 1]
        y_prev = atom_index.y_iter[k]
        Ax_k = atom_index.Ax_iter[k + 1]

        for q in 1:nG
            Aconst = zeros(Float64, nG, nG)
            Arho = zeros(Float64, nG, nG)

            _add_sym_entry!(Aconst, q, y_k, +1.0)
            _add_sym_entry!(Aconst, q, y_prev, -1.0)
            _add_sym_entry!(Arho, q, Ax_k, +1.0)

            push!(Aeq_y_const, Aconst)
            push!(Aeq_y_rho, Arho)
        end
    end

    # Anchor equalities that make the explicit auxiliary atoms equivalent to the
    # joint-interpolation columns Y_Q[:, star] = 0 and Y_Q[:, y_star] = -g_star.
    for q in 1:nG
        A = zeros(Float64, nG, nG)
        _add_sym_entry!(A, q, atom_index.Ax_star, +1.0)
        push!(Aeq_Ax, A)

        A = zeros(Float64, nG, nG)
        _add_sym_entry!(A, q, atom_index.g_star, +1.0)
        _add_sym_entry!(A, q, atom_index.Ay_star, +1.0)
        push!(Aeq_gAy, A)

        A = zeros(Float64, nG, nG)
        _add_sym_entry!(A, q, atom_index.x_bar, +1.0)
        push!(Aeq_barx_const, A)

        Aweights = Matrix{Float64}[]
        for i in 0:N
            Aω = zeros(Float64, nG, nG)
            _add_sym_entry!(Aω, q, atom_index.x_iter[i + 1], -1.0)
            push!(Aweights, Aω)
        end
        push!(Aeq_barx_omega, Aweights)

        A = zeros(Float64, nG, nG)
        _add_sym_entry!(A, q, atom_index.Abarx, +1.0)
        push!(Aeq_Abarx_const, A)

        Aweights = Matrix{Float64}[]
        for i in 0:N
            Aω = zeros(Float64, nG, nG)
            _add_sym_entry!(Aω, q, atom_index.Ax_iter[i + 1], -1.0)
            push!(Aweights, Aω)
        end
        push!(Aeq_Abarx_omega, Aweights)
    end

    # Smooth interpolation constraints are unchanged: they depend on the atoms
    # and function values, not directly on `(tau, rho_dual, omega)`.
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

    Ax0 = zeros(Float64, nG, nG)
    _add_sym_entry!(Ax0, atom_index.x_iter[1], atom_index.x_iter[1], +1.0)
    _add_sym_entry!(Ax0, atom_index.x_star, atom_index.x_star, +1.0)
    _add_sym_entry!(Ax0, atom_index.x_iter[1], atom_index.x_star, -2.0)

    Ay0 = zeros(Float64, nG, nG)
    _add_sym_entry!(Ay0, atom_index.y_iter[1], atom_index.y_iter[1], +1.0)
    _add_sym_entry!(Ay0, atom_index.y_star, atom_index.y_star, +1.0)
    _add_sym_entry!(Ay0, atom_index.y_iter[1], atom_index.y_star, -2.0)

    U_idx = [atom_index.x_star; atom_index.x_iter; atom_index.x_bar; atom_index.y_star; atom_index.y_iter]
    AU_idx = [atom_index.Ax_star; atom_index.Ax_iter; atom_index.Abarx; atom_index.Ay_star; atom_index.Ay_iter]

    data = AxSDPJointALMDesignQCQPData(
        N = N,
        smoothness_L = float(smoothness_L),
        mu_A = float(mu_A),
        L_A = float(L_A),
        Rx2 = float(Rx2),
        Ry2 = float(Ry2),
        C = C,
        c = c,
        Aeq_x_const = Aeq_x_const,
        Aeq_x_tau = Aeq_x_tau,
        Aeq_y_const = Aeq_y_const,
        Aeq_y_rho = Aeq_y_rho,
        Aeq_barx_const = Aeq_barx_const,
        Aeq_barx_omega = Aeq_barx_omega,
        Aeq_Abarx_const = Aeq_Abarx_const,
        Aeq_Abarx_omega = Aeq_Abarx_omega,
        Aeq_Ax = Aeq_Ax,
        Aeq_gAy = Aeq_gAy,
        Aint = Aint,
        aint = aint,
        Ax0 = Ax0,
        Ay0 = Ay0,
        U_idx = U_idx,
        AU_idx = AU_idx,
        tau0 = float(tau0),
        rho_dual0 = float(rho_dual0),
        omega0 = omega0_vec,
    )
    meta = AxSDPJointALMDesignMetadata(
        N = N,
        nG = nG,
        nf = nf,
        atom_index = atom_index,
        value_index = value_index,
    )
    return data, meta
end


"""
Default design instance matching the current fixed-parameter ALM-like family.

The warm-start/default algorithm is exactly the one used by
`build_default_alm_joint_instance`, but here the final QCQP is allowed to move
away from that point by changing `(tau, rho_dual, omega)`.
"""
function build_default_alm_joint_design_instance(;
    N::Int = 1,
    smoothness_L::Real = 1.0,
    mu_A::Real = 0.1,
    L_A::Real = 1.0,
    rho_dual0::Real = 1.0,
    eta0::Union{Nothing, Real} = nothing,
    Rx2::Real = 1.0,
    Ry2::Real = 1.0,
    allow_weight_on_x0::Bool = false,
)
    eta_value = isnothing(eta0) ? max(2.0 * rho_dual0 * L_A, 2.0 * smoothness_L) : float(eta0)
    tau0 = 1.0 / eta_value

    omega0 =
        if allow_weight_on_x0
            fill(1.0 / (N + 1), N + 1)
        else
            [0.0; fill(1.0 / N, N)]
        end

    return build_axsdp_joint_alm_design_data(
        N = N,
        smoothness_L = smoothness_L,
        mu_A = mu_A,
        L_A = L_A,
        Rx2 = Rx2,
        Ry2 = Ry2,
        tau0 = tau0,
        rho_dual0 = rho_dual0,
        omega0 = omega0,
    )
end


function axsdp_joint_design_instance_summary(
    data::AxSDPJointALMDesignQCQPData,
    meta::AxSDPJointALMDesignMetadata,
)
    return (
        N = meta.N,
        nG = meta.nG,
        nf = meta.nf,
        m = length(data.U_idx),
        num_Aeq_x = length(data.Aeq_x_const),
        num_Aeq_y = length(data.Aeq_y_const),
        num_Aeq_barx = length(data.Aeq_barx_const),
        num_Aeq_Abarx = length(data.Aeq_Abarx_const),
        num_Aeq_Ax = length(data.Aeq_Ax),
        num_Aeq_gAy = length(data.Aeq_gAy),
        num_Aint = length(data.Aint),
        tau0 = data.tau0,
        rho_dual0 = data.rho_dual0,
        omega0 = data.omega0,
    )
end


"""
Return the ALM-like coefficient arrays implied by `(tau, rho_dual, omega)`.

This is the design-stage counterpart of `build_alm_like_coefficients`. It is
useful when we want to interpret the final optimized AxSDP-QCQP solution as a
concrete algorithm written in the slide notation `(alpha, rho_xy, beta, omega)`.
"""
function alm_coefficients_from_design_parameters(
    N::Int,
    tau::Real,
    rho_dual::Real,
    omega,
)
    tau >= 0 || error("tau must be nonnegative.")
    rho_dual >= 0 || error("rho_dual must be nonnegative.")

    alpha = zeros(Float64, N + 1, N + 1)
    rho_xy = zeros(Float64, N + 1, N + 1)
    beta = zeros(Float64, N + 1, N + 1)

    for k in 1:N
        alpha[k + 1, :] .= alpha[k, :]
        rho_xy[k + 1, :] .= rho_xy[k, :]

        alpha[k + 1, k] += -float(tau)
        if k == 1
            rho_xy[k + 1, 1] += float(tau)
        else
            rho_xy[k + 1, k] += 2.0 * float(tau)
            rho_xy[k + 1, k - 1] += -float(tau)
        end

        for i in 1:k
            beta[k + 1, i + 1] = -float(rho_dual)
        end
    end

    return (
        alpha = alpha,
        rho_xy = rho_xy,
        beta = beta,
        omega = Float64.(collect(omega)),
    )
end


"""
Build a design-QCQP warm start by augmenting the fixed-algorithm SDP warm start
with the current algorithm-parameter guess `(tau0, rho_dual0, omega0)`.
"""
function build_design_qcqp_warm_start_from_fixed_sdp_solution(
    sdp_solution,
    design_data::AxSDPJointALMDesignQCQPData;
    chol_tol::Real = 1e-8,
    scale::Real = 1.01,
    min_bound::Real = 1.0,
)
    warm = build_qcqp_warm_start_from_sdp_solution(
        sdp_solution;
        chol_tol = chol_tol,
        scale = scale,
        min_bound = min_bound,
    )
    return merge(
        warm,
        (
            tau = design_data.tau0,
            rho_dual = design_data.rho_dual0,
            omega = copy(design_data.omega0),
        ),
    )
end


"""
Solve the AxSDP design QCQP where `(tau, rho_dual, omega)` are decision
variables.

This is the AxSDP analogue of the function-value BnB-PEP workflow:

1. start from one feasible fixed algorithm,
2. warm-start the dual/QCQP variables from its SDP solution, and then
3. jointly optimize the certificate and the algorithm parameters.

The current implementation focuses on the "design QCQP" stage. It does not yet
replicate the full lifted-SDP bound-generation machinery from the function-value
 code, but it already exposes the core optimization problem whose solution gives
 the designed algorithm parameters.
"""
function solve_axsdp_joint_alm_design_qcqp(
    data::AxSDPJointALMDesignQCQPData;
    show_output::Symbol = :off,
    factor_bound::Union{Nothing, Real} = nothing,
    psd_bound::Union{Nothing, Real} = nothing,
    equality_tolerance::Real = 0.0,
    add_psd_cuts::Bool = true,
    warm_start = nothing,
    gurobi_params = Dict{String, Any}(),
    tau_lower::Real = 0.0,
    tau_upper::Union{Nothing, Real} = nothing,
    rho_dual_lower::Real = 0.0,
    rho_dual_upper::Real = 2.0,
    allow_weight_on_x0::Bool = false,
    enforce_default_stability::Bool = true,
)
    ctx = _validate_design_input_data(data)
    equality_tolerance >= 0 || error("equality_tolerance must be nonnegative.")
    isnothing(factor_bound) || factor_bound > 0 || error("factor_bound must be positive when provided.")
    isnothing(psd_bound) || psd_bound > 0 || error("psd_bound must be positive when provided.")
    tau_lower >= 0 || error("tau_lower must be nonnegative.")
    rho_dual_lower >= 0 || error("rho_dual_lower must be nonnegative.")
    rho_dual_upper > rho_dual_lower || error("rho_dual_upper must be strictly larger than rho_dual_lower.")

    tau_upper_value = isnothing(tau_upper) ? 1.0 / (2.0 * data.smoothness_L) : float(tau_upper)
    tau_upper_value > tau_lower || error("tau_upper must be strictly larger than tau_lower.")

    model = Model(Gurobi.Optimizer)
    if show_output == :off && !haskey(gurobi_params, "OutputFlag")
        set_attribute(model, "OutputFlag", 0)
    end
    for (key, value) in pairs(gurobi_params)
        set_attribute(model, key, value)
    end
    set_attribute(model, "NonConvex", 2)

    # Algorithm parameters to be designed.
    @variable(model, tau_lower <= tau <= tau_upper_value)
    @variable(model, rho_dual_lower <= rho_dual <= rho_dual_upper)
    @variable(model, omega[1:(data.N + 1)] >= 0)

    @constraint(model, sum(omega) == 1.0)
    if !allow_weight_on_x0
        @constraint(model, omega[1] == 0.0)
    end

    # These two bounds are exactly the conditions behind the default warm-start
    # family eta = max(2 rho_dual L_A, 2 L_f), rewritten in terms of tau = 1/eta.
    if enforce_default_stability
        @constraint(model, data.smoothness_L * tau <= 0.5)
        @constraint(model, data.L_A * tau * rho_dual <= 0.5)
    end

    lambda_x = _add_free_vector(model, length(data.Aeq_x_const), "lambda_x")
    lambda_y = _add_free_vector(model, length(data.Aeq_y_const), "lambda_y")
    lambda_barx = _add_free_vector(model, length(data.Aeq_barx_const), "lambda_barx")
    lambda_Abarx = _add_free_vector(model, length(data.Aeq_Abarx_const), "lambda_Abarx")
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
    _set_scalar_start!(tau, _warm_get(warm_start, :tau))
    _set_scalar_start!(rho_dual, _warm_get(warm_start, :rho_dual))
    _set_vector_start!(omega, _warm_get(warm_start, :omega))

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

            expr = _add_affine_parameter_sequence_entry(
                expr,
                lambda_x,
                data.Aeq_x_const,
                data.Aeq_x_tau,
                tau,
                i,
                j,
            )
            expr = _add_affine_parameter_sequence_entry(
                expr,
                lambda_y,
                data.Aeq_y_const,
                data.Aeq_y_rho,
                rho_dual,
                i,
                j,
            )
            expr = _add_weighted_sequence_entry(
                expr,
                lambda_barx,
                data.Aeq_barx_const,
                data.Aeq_barx_omega,
                omega,
                i,
                j,
            )
            expr = _add_weighted_sequence_entry(
                expr,
                lambda_Abarx,
                data.Aeq_Abarx_const,
                data.Aeq_Abarx_omega,
                omega,
                i,
                j,
            )

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

    solution = _collect_design_qcqp_solution(
        model,
        data,
        tau,
        rho_dual,
        omega,
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
Local design stage for AxSDP, analogous to the local-optimization phase in the
function-value BnB-PEP workflow.

This uses Ipopt to optimize `(tau, rho_dual, omega)` jointly with the dual
certificate variables, starting from the fixed-algorithm SDP warm start.
Compared with the global Gurobi QCQP, this stage is usually much better at
producing an actual candidate algorithm quickly for larger `N`.
"""
function solve_axsdp_joint_alm_design_local(
    data::AxSDPJointALMDesignQCQPData;
    show_output::Symbol = :off,
    factor_bound::Union{Nothing, Real} = nothing,
    psd_bound::Union{Nothing, Real} = nothing,
    equality_tolerance::Real = 0.0,
    add_psd_cuts::Bool = true,
    warm_start = nothing,
    ipopt_params = Dict{String, Any}(),
    tau_lower::Real = 0.0,
    tau_upper::Union{Nothing, Real} = nothing,
    rho_dual_lower::Real = 0.0,
    rho_dual_upper::Real = 2.0,
    allow_weight_on_x0::Bool = false,
    enforce_default_stability::Bool = true,
)
    ctx = _validate_design_input_data(data)
    equality_tolerance >= 0 || error("equality_tolerance must be nonnegative.")
    isnothing(factor_bound) || factor_bound > 0 || error("factor_bound must be positive when provided.")
    isnothing(psd_bound) || psd_bound > 0 || error("psd_bound must be positive when provided.")
    tau_lower >= 0 || error("tau_lower must be nonnegative.")
    rho_dual_lower >= 0 || error("rho_dual_lower must be nonnegative.")
    rho_dual_upper > rho_dual_lower || error("rho_dual_upper must be strictly larger than rho_dual_lower.")

    tau_upper_value = isnothing(tau_upper) ? 1.0 / (2.0 * data.smoothness_L) : float(tau_upper)
    tau_upper_value > tau_lower || error("tau_upper must be strictly larger than tau_lower.")

    model = Model(Ipopt.Optimizer)
    if show_output == :off
        set_silent(model)
        set_attribute(model, "print_level", 0)
    elseif show_output != :on
        error("show_output must be either :on or :off.")
    end
    for (key, value) in pairs(ipopt_params)
        set_attribute(model, key, value)
    end

    @variable(model, tau_lower <= tau <= tau_upper_value)
    @variable(model, rho_dual_lower <= rho_dual <= rho_dual_upper)
    @variable(model, omega[1:(data.N + 1)] >= 0)

    @constraint(model, sum(omega) == 1.0)
    if !allow_weight_on_x0
        @constraint(model, omega[1] == 0.0)
    end

    if enforce_default_stability
        @constraint(model, data.smoothness_L * tau <= 0.5)
        @constraint(model, data.L_A * tau * rho_dual <= 0.5)
    end

    lambda_x = _add_free_vector(model, length(data.Aeq_x_const), "lambda_x")
    lambda_y = _add_free_vector(model, length(data.Aeq_y_const), "lambda_y")
    lambda_barx = _add_free_vector(model, length(data.Aeq_barx_const), "lambda_barx")
    lambda_Abarx = _add_free_vector(model, length(data.Aeq_Abarx_const), "lambda_Abarx")
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
    _set_scalar_start!(tau, _warm_get(warm_start, :tau))
    _set_scalar_start!(rho_dual, _warm_get(warm_start, :rho_dual))
    _set_vector_start!(omega, _warm_get(warm_start, :omega))

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

            expr = _add_affine_parameter_sequence_entry(
                expr,
                lambda_x,
                data.Aeq_x_const,
                data.Aeq_x_tau,
                tau,
                i,
                j,
            )
            expr = _add_affine_parameter_sequence_entry(
                expr,
                lambda_y,
                data.Aeq_y_const,
                data.Aeq_y_rho,
                rho_dual,
                i,
                j,
            )
            expr = _add_weighted_sequence_entry(
                expr,
                lambda_barx,
                data.Aeq_barx_const,
                data.Aeq_barx_omega,
                omega,
                i,
                j,
            )
            expr = _add_weighted_sequence_entry(
                expr,
                lambda_Abarx,
                data.Aeq_Abarx_const,
                data.Aeq_Abarx_omega,
                omega,
                i,
                j,
            )

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

    solution = _collect_design_qcqp_solution(
        model,
        data,
        tau,
        rho_dual,
        omega,
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


function _validate_design_input_data(data::AxSDPJointALMDesignQCQPData)
    nG = size(data.C, 1)
    nG > 0 || error("C must be nonempty.")
    size(data.C, 2) == nG || error("C must be square.")

    nf = length(data.c)
    data.N >= 1 || error("N must be at least 1.")
    size(data.Ax0) == (nG, nG) || error("Ax0 must be an nG-by-nG matrix.")
    size(data.Ay0) == (nG, nG) || error("Ay0 must be an nG-by-nG matrix.")

    _check_matrix_list_shape(data.Aeq_x_const, nG, "Aeq_x_const")
    _check_matrix_list_shape(data.Aeq_x_tau, nG, "Aeq_x_tau")
    length(data.Aeq_x_const) == length(data.Aeq_x_tau) || error("Aeq_x_const and Aeq_x_tau must have the same length.")

    _check_matrix_list_shape(data.Aeq_y_const, nG, "Aeq_y_const")
    _check_matrix_list_shape(data.Aeq_y_rho, nG, "Aeq_y_rho")
    length(data.Aeq_y_const) == length(data.Aeq_y_rho) || error("Aeq_y_const and Aeq_y_rho must have the same length.")

    _check_matrix_list_shape(data.Aeq_barx_const, nG, "Aeq_barx_const")
    _check_weighted_templates(data.Aeq_barx_omega, nG, data.N + 1, "Aeq_barx_omega")
    length(data.Aeq_barx_const) == length(data.Aeq_barx_omega) || error("Aeq_barx_const and Aeq_barx_omega must have the same length.")

    _check_matrix_list_shape(data.Aeq_Abarx_const, nG, "Aeq_Abarx_const")
    _check_weighted_templates(data.Aeq_Abarx_omega, nG, data.N + 1, "Aeq_Abarx_omega")
    length(data.Aeq_Abarx_const) == length(data.Aeq_Abarx_omega) || error("Aeq_Abarx_const and Aeq_Abarx_omega must have the same length.")

    _check_matrix_list_shape(data.Aeq_Ax, nG, "Aeq_Ax")
    _check_matrix_list_shape(data.Aeq_gAy, nG, "Aeq_gAy")
    _check_matrix_list_shape(data.Aint, nG, "Aint")
    length(data.Aint) == length(data.aint) || error("Aint and aint must have the same length.")
    for a in data.aint
        length(a) == nf || error("Every vector in aint must have length nf.")
    end

    length(data.U_idx) == length(data.AU_idx) || error("U_idx and AU_idx must have the same length.")
    !isempty(data.U_idx) || error("U_idx and AU_idx must be nonempty.")
    all(idx -> 1 <= idx <= nG, data.U_idx) || error("U_idx contains an invalid atom index.")
    all(idx -> 1 <= idx <= nG, data.AU_idx) || error("AU_idx contains an invalid atom index.")

    length(data.omega0) == data.N + 1 || error("omega0 must have length N + 1.")
    minimum(data.omega0) >= -1e-10 || error("omega0 must be componentwise nonnegative.")
    abs(sum(data.omega0) - 1.0) <= 1e-8 || error("omega0 must sum to 1.")

    u_pos = Dict(data.U_idx[p] => p for p in eachindex(data.U_idx))
    au_pos = Dict(data.AU_idx[p] => p for p in eachindex(data.AU_idx))
    return (nG = nG, nf = nf, m = length(data.U_idx), u_pos = u_pos, au_pos = au_pos)
end


function _check_matrix_list_shape(mats::Vector{Matrix{Float64}}, nG::Int, name::AbstractString)
    for A in mats
        size(A) == (nG, nG) || error("Every matrix in $name must be of size $nG-by-$nG.")
    end
    return nothing
end


function _check_weighted_templates(
    templates::Vector{Vector{Matrix{Float64}}},
    nG::Int,
    num_weights::Int,
    name::AbstractString,
)
    for mats in templates
        length(mats) == num_weights || error("Each template bundle in $name must have $num_weights weight matrices.")
        for A in mats
            size(A) == (nG, nG) || error("Every matrix in $name must be of size $nG-by-$nG.")
        end
    end
    return nothing
end


function _add_affine_parameter_sequence_entry(
    expr,
    vars,
    const_mats::Vector{Matrix{Float64}},
    param_mats::Vector{Matrix{Float64}},
    parameter_var,
    i::Int,
    j::Int,
)
    for t in eachindex(const_mats)
        coeff_const = const_mats[t][i, j]
        if coeff_const != 0.0
            expr += coeff_const * vars[t]
        end

        coeff_param = param_mats[t][i, j]
        if coeff_param != 0.0
            expr += coeff_param * vars[t] * parameter_var
        end
    end
    return expr
end


function _add_weighted_sequence_entry(
    expr,
    vars,
    const_mats::Vector{Matrix{Float64}},
    weight_mats::Vector{Vector{Matrix{Float64}}},
    omega_vars,
    i::Int,
    j::Int,
)
    for t in eachindex(const_mats)
        coeff_const = const_mats[t][i, j]
        if coeff_const != 0.0
            expr += coeff_const * vars[t]
        end

        mats_t = weight_mats[t]
        for s in eachindex(mats_t)
            coeff_weight = mats_t[s][i, j]
            if coeff_weight != 0.0
                expr += coeff_weight * vars[t] * omega_vars[s]
            end
        end
    end
    return expr
end


function _collect_design_qcqp_solution(
    model::Model,
    data::AxSDPJointALMDesignQCQPData,
    tau,
    rho_dual,
    omega,
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

    tau_value = has_solution ? value(tau) : nothing
    rho_value = has_solution ? value(rho_dual) : nothing
    omega_value = has_solution ? value.(omega) : nothing
    coeffs =
        if has_solution
            alm_coefficients_from_design_parameters(data.N, tau_value, rho_value, omega_value)
        else
            nothing
        end

    return (
        termination_status = termination_status(model),
        primal_status = primal_status(model),
        dual_status = dual_status(model),
        raw_status = raw_status(model),
        solve_time_sec = solve_time(model),
        result_count = result_count(model),
        objective = has_solution ? objective_value(model) : nothing,
        objective_bound = _safe_objective_bound(model),
        tau = tau_value,
        eta = has_solution ? (tau_value == 0.0 ? Inf : 1.0 / tau_value) : nothing,
        rho_dual = rho_value,
        omega = omega_value,
        coefficients = coeffs,
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
