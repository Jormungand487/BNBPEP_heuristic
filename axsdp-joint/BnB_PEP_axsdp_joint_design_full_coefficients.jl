include("BnB_PEP_axsdp_joint_design_general_helpers.jl")


Base.@kwdef struct AxSDPJointFullCoeffDesignQCQPData
    N::Int
    smoothness_L::Float64
    mu_A::Float64
    L_A::Float64
    Rx2::Float64
    Ry2::Float64
    C::Matrix{Float64}
    c::Vector{Float64}
    Aeq_x_const::Vector{Matrix{Float64}}
    Aeq_x_params::Vector{Vector{Matrix{Float64}}}
    Aeq_y_const::Vector{Matrix{Float64}}
    Aeq_y_params::Vector{Vector{Matrix{Float64}}}
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
    alpha_pairs::Vector{Tuple{Int, Int}}
    rho_pairs::Vector{Tuple{Int, Int}}
    beta_pairs::Vector{Tuple{Int, Int}}
    alpha0_flat::Vector{Float64}
    rho0_flat::Vector{Float64}
    beta0_flat::Vector{Float64}
    omega0::Vector{Float64}
end


const AxSDPJointFullCoeffDesignMetadata = AxSDPJointALMDesignMetadata


function _flatten_coefficients_by_pairs(A::AbstractMatrix, pairs)
    values = Float64[]
    for (k, i) in pairs
        push!(values, float(A[k + 1, i + 1]))
    end
    return values
end


function _flatten_beta_by_pairs(B::AbstractMatrix, pairs)
    values = Float64[]
    for (k, i) in pairs
        push!(values, float(B[k + 1, i + 1]))
    end
    return values
end


function _decode_full_coefficients(
    N::Int,
    alpha_pairs,
    rho_pairs,
    beta_pairs,
    alpha_flat,
    rho_flat,
    beta_flat,
    omega,
)
    alpha = zeros(Float64, N + 1, N + 1)
    rho_xy = zeros(Float64, N + 1, N + 1)
    beta = zeros(Float64, N + 1, N + 1)

    for (idx, (k, i)) in enumerate(alpha_pairs)
        alpha[k + 1, i + 1] = float(alpha_flat[idx])
    end
    for (idx, (k, i)) in enumerate(rho_pairs)
        rho_xy[k + 1, i + 1] = float(rho_flat[idx])
    end
    for (idx, (k, i)) in enumerate(beta_pairs)
        beta[k + 1, i + 1] = float(beta_flat[idx])
    end

    return (
        alpha = alpha,
        rho_xy = rho_xy,
        beta = beta,
        omega = Float64.(collect(omega)),
    )
end


function build_axsdp_joint_full_coeff_design_data(;
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
    length(omega_vec) >= 2 || error("omega must have length at least 2.")
    N = length(omega_vec) - 1

    alpha_mat = _to_dense_square(alpha, N, "alpha")
    rho_mat = _to_dense_square(rho_xy, N, "rho_xy")
    beta_mat = _to_dense_square(beta, N, "beta")

    atom_index = _build_joint_atom_index(N)
    value_index = _build_value_index(N)
    nG = atom_index.nG
    nf = value_index.nf

    alpha_pairs = _triangular_pairs_zero_based(N)
    rho_pairs = _triangular_pairs_zero_based(N)
    beta_pairs = _triangular_pairs_one_based(N)

    n_alpha = length(alpha_pairs)
    n_rho = length(rho_pairs)
    n_beta = length(beta_pairs)

    C = zeros(Float64, nG, nG)
    c = zeros(Float64, nf)
    c[value_index.f_star] = -1.0
    c[value_index.f_bar] = 1.0
    _add_sym_entry!(C, atom_index.Abarx, atom_index.y_star, -1.0)

    Aeq_x_const = Matrix{Float64}[]
    Aeq_x_params = Vector{Vector{Matrix{Float64}}}()
    Aeq_y_const = Matrix{Float64}[]
    Aeq_y_params = Vector{Vector{Matrix{Float64}}}()
    Aeq_barx_const = Matrix{Float64}[]
    Aeq_barx_omega = Vector{Vector{Matrix{Float64}}}()
    Aeq_Abarx_const = Matrix{Float64}[]
    Aeq_Abarx_omega = Vector{Vector{Matrix{Float64}}}()
    Aeq_Ax = Matrix{Float64}[]
    Aeq_gAy = Matrix{Float64}[]

    for k in 1:N
        x_k = atom_index.x_iter[k + 1]
        x_0 = atom_index.x_iter[1]
        y_k = atom_index.y_iter[k + 1]
        y_0 = atom_index.y_iter[1]

        for q in 1:nG
            A = zeros(Float64, nG, nG)
            bundle = [zeros(Float64, nG, nG) for _ in 1:(n_alpha + n_rho)]
            _add_sym_entry!(A, q, x_k, +1.0)
            _add_sym_entry!(A, q, x_0, -1.0)

            for (idx, (row_k, i)) in enumerate(alpha_pairs)
                if row_k == k
                    _add_sym_entry!(bundle[idx], q, atom_index.g_iter[i + 1], -1.0)
                end
            end
            for (idx, (row_k, i)) in enumerate(rho_pairs)
                if row_k == k
                    _add_sym_entry!(bundle[n_alpha + idx], q, atom_index.Ay_iter[i + 1], -1.0)
                end
            end
            push!(Aeq_x_const, A)
            push!(Aeq_x_params, bundle)

            A = zeros(Float64, nG, nG)
            bundle = [zeros(Float64, nG, nG) for _ in 1:n_beta]
            _add_sym_entry!(A, q, y_k, +1.0)
            _add_sym_entry!(A, q, y_0, -1.0)
            for (idx, (row_k, i)) in enumerate(beta_pairs)
                if row_k == k
                    _add_sym_entry!(bundle[idx], q, atom_index.Ax_iter[i + 1], -1.0)
                end
            end
            push!(Aeq_y_const, A)
            push!(Aeq_y_params, bundle)
        end
    end

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

        omega_bundle = Matrix{Float64}[]
        for i in 0:N
            Aω = zeros(Float64, nG, nG)
            _add_sym_entry!(Aω, q, atom_index.x_iter[i + 1], -1.0)
            push!(omega_bundle, Aω)
        end
        push!(Aeq_barx_omega, omega_bundle)

        A = zeros(Float64, nG, nG)
        _add_sym_entry!(A, q, atom_index.Abarx, +1.0)
        push!(Aeq_Abarx_const, A)

        omega_bundle = Matrix{Float64}[]
        for i in 0:N
            Aω = zeros(Float64, nG, nG)
            _add_sym_entry!(Aω, q, atom_index.Ax_iter[i + 1], -1.0)
            push!(omega_bundle, Aω)
        end
        push!(Aeq_Abarx_omega, omega_bundle)
    end

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

    data = AxSDPJointFullCoeffDesignQCQPData(
        N = N,
        smoothness_L = float(smoothness_L),
        mu_A = float(mu_A),
        L_A = float(L_A),
        Rx2 = float(Rx2),
        Ry2 = float(Ry2),
        C = C,
        c = c,
        Aeq_x_const = Aeq_x_const,
        Aeq_x_params = Aeq_x_params,
        Aeq_y_const = Aeq_y_const,
        Aeq_y_params = Aeq_y_params,
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
        alpha_pairs = alpha_pairs,
        rho_pairs = rho_pairs,
        beta_pairs = beta_pairs,
        alpha0_flat = _flatten_coefficients_by_pairs(alpha_mat, alpha_pairs),
        rho0_flat = _flatten_coefficients_by_pairs(rho_mat, rho_pairs),
        beta0_flat = _flatten_beta_by_pairs(beta_mat, beta_pairs),
        omega0 = omega_vec,
    )
    meta = AxSDPJointFullCoeffDesignMetadata(
        N = N,
        nG = nG,
        nf = nf,
        atom_index = atom_index,
        value_index = value_index,
    )
    return data, meta
end


function build_default_full_coeff_design_instance(;
    N::Int = 1,
    smoothness_L::Real = 1.0,
    mu_A::Real = 0.1,
    L_A::Real = 1.0,
    rho_dual0::Real = 1.0,
    eta0::Union{Nothing, Real} = nothing,
    Rx2::Real = 1.0,
    Ry2::Real = 1.0,
)
    eta_value = isnothing(eta0) ? max(2.0 * rho_dual0 * L_A, 2.0 * smoothness_L) : float(eta0)
    alpha0, rho0, beta0, omega0 = build_alm_like_coefficients(
        N = N,
        eta = eta_value,
        rho_dual = rho_dual0,
    )
    return build_axsdp_joint_full_coeff_design_data(
        alpha = alpha0,
        rho_xy = rho0,
        beta = beta0,
        omega = omega0,
        smoothness_L = smoothness_L,
        mu_A = mu_A,
        L_A = L_A,
        Rx2 = Rx2,
        Ry2 = Ry2,
    )
end


function build_full_coeff_qcqp_warm_start_from_fixed_sdp_solution(
    sdp_solution,
    design_data::AxSDPJointFullCoeffDesignQCQPData;
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
            alpha = copy(design_data.alpha0_flat),
            rho_xy = copy(design_data.rho0_flat),
            beta = copy(design_data.beta0_flat),
            omega = copy(design_data.omega0),
        ),
    )
end


function _validate_full_coeff_design_input_data(data::AxSDPJointFullCoeffDesignQCQPData)
    nG = size(data.C, 1)
    nG > 0 || error("C must be nonempty.")
    size(data.C, 2) == nG || error("C must be square.")
    nf = length(data.c)

    _check_matrix_list_shape(data.Aeq_x_const, nG, "Aeq_x_const")
    _check_param_template_bundles(data.Aeq_x_params, nG, length(data.alpha_pairs) + length(data.rho_pairs), "Aeq_x_params")
    _check_matrix_list_shape(data.Aeq_y_const, nG, "Aeq_y_const")
    _check_param_template_bundles(data.Aeq_y_params, nG, length(data.beta_pairs), "Aeq_y_params")
    _check_matrix_list_shape(data.Aeq_barx_const, nG, "Aeq_barx_const")
    _check_param_template_bundles(data.Aeq_barx_omega, nG, data.N + 1, "Aeq_barx_omega")
    _check_matrix_list_shape(data.Aeq_Abarx_const, nG, "Aeq_Abarx_const")
    _check_param_template_bundles(data.Aeq_Abarx_omega, nG, data.N + 1, "Aeq_Abarx_omega")
    _check_matrix_list_shape(data.Aeq_Ax, nG, "Aeq_Ax")
    _check_matrix_list_shape(data.Aeq_gAy, nG, "Aeq_gAy")
    _check_matrix_list_shape(data.Aint, nG, "Aint")
    length(data.Aint) == length(data.aint) || error("Aint and aint must have the same length.")
    for a in data.aint
        length(a) == nf || error("Every vector in aint must have length nf.")
    end

    u_pos = Dict(data.U_idx[p] => p for p in eachindex(data.U_idx))
    au_pos = Dict(data.AU_idx[p] => p for p in eachindex(data.AU_idx))
    return (nG = nG, nf = nf, m = length(data.U_idx), u_pos = u_pos, au_pos = au_pos)
end


function _collect_full_coeff_design_solution(
    model::Model,
    data::AxSDPJointFullCoeffDesignQCQPData,
    alpha_vars,
    rho_vars,
    beta_vars,
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

    alpha_flat = has_solution ? value.(alpha_vars) : nothing
    rho_flat = has_solution ? value.(rho_vars) : nothing
    beta_flat = has_solution ? value.(beta_vars) : nothing
    omega_val = has_solution ? value.(omega) : nothing
    coeffs =
        if has_solution
            _decode_full_coefficients(
                data.N,
                data.alpha_pairs,
                data.rho_pairs,
                data.beta_pairs,
                alpha_flat,
                rho_flat,
                beta_flat,
                omega_val,
            )
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
        alpha_flat = alpha_flat,
        rho_xy_flat = rho_flat,
        beta_flat = beta_flat,
        omega = omega_val,
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


function solve_axsdp_joint_full_coeff_design_qcqp(
    data::AxSDPJointFullCoeffDesignQCQPData;
    show_output::Symbol = :off,
    factor_bound::Union{Nothing, Real} = nothing,
    psd_bound::Union{Nothing, Real} = nothing,
    equality_tolerance::Real = 0.0,
    add_psd_cuts::Bool = true,
    warm_start = nothing,
    gurobi_params = Dict{String, Any}(),
    alpha_bound::Union{Nothing, Real} = nothing,
    rho_bound::Union{Nothing, Real} = nothing,
    beta_bound::Union{Nothing, Real} = nothing,
    allow_weight_on_x0::Bool = false,
)
    ctx = _validate_full_coeff_design_input_data(data)
    equality_tolerance >= 0 || error("equality_tolerance must be nonnegative.")
    isnothing(factor_bound) || factor_bound > 0 || error("factor_bound must be positive when provided.")
    isnothing(psd_bound) || psd_bound > 0 || error("psd_bound must be positive when provided.")

    alpha_bound_value = isnothing(alpha_bound) ? _default_box_upper(data.alpha0_flat) : float(alpha_bound)
    rho_bound_value = isnothing(rho_bound) ? _default_box_upper(data.rho0_flat) : float(rho_bound)
    beta_bound_value = isnothing(beta_bound) ? _default_box_upper(data.beta0_flat) : float(beta_bound)

    model = Model(Gurobi.Optimizer)
    if show_output == :off && !haskey(gurobi_params, "OutputFlag")
        set_attribute(model, "OutputFlag", 0)
    end
    for (key, value) in pairs(gurobi_params)
        set_attribute(model, key, value)
    end
    set_attribute(model, "NonConvex", 2)

    @variable(model, -alpha_bound_value <= alpha_vars[1:length(data.alpha_pairs)] <= alpha_bound_value)
    @variable(model, -rho_bound_value <= rho_vars[1:length(data.rho_pairs)] <= rho_bound_value)
    @variable(model, -beta_bound_value <= beta_vars[1:length(data.beta_pairs)] <= beta_bound_value)
    @variable(model, omega[1:(data.N + 1)] >= 0)
    @constraint(model, sum(omega) == 1.0)
    if !allow_weight_on_x0
        @constraint(model, omega[1] == 0.0)
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
        lambda_x, lambda_y, lambda_barx, lambda_Abarx, lambda_Ax, lambda_gAy,
        nu, mu_x, mu_y, M, Z_minus, Z_plus, W, P_minus, P_plus, S, warm_start,
    )
    _set_vector_start!(alpha_vars, _warm_get(warm_start, :alpha))
    _set_vector_start!(rho_vars, _warm_get(warm_start, :rho_xy))
    _set_vector_start!(beta_vars, _warm_get(warm_start, :beta))
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

    x_param_vars = vcat(collect(alpha_vars), collect(rho_vars))
    y_param_vars = collect(beta_vars)
    for i in 1:ctx.nG
        for j in 1:i
            expr = data.C[i, j]
            expr = _add_param_bundle_sequence_entry(expr, lambda_x, data.Aeq_x_const, data.Aeq_x_params, x_param_vars, i, j)
            expr = _add_param_bundle_sequence_entry(expr, lambda_y, data.Aeq_y_const, data.Aeq_y_params, y_param_vars, i, j)
            expr = _add_param_bundle_sequence_entry(expr, lambda_barx, data.Aeq_barx_const, data.Aeq_barx_omega, omega, i, j)
            expr = _add_param_bundle_sequence_entry(expr, lambda_Abarx, data.Aeq_Abarx_const, data.Aeq_Abarx_omega, omega, i, j)
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

    solution = _collect_full_coeff_design_solution(
        model, data, alpha_vars, rho_vars, beta_vars, omega,
        lambda_x, lambda_y, lambda_barx, lambda_Abarx, lambda_Ax, lambda_gAy,
        nu, mu_x, mu_y, M, Z_minus, Z_plus, W, P_minus, P_plus, S,
    )
    return (model = model, solution = solution)
end


function solve_axsdp_joint_full_coeff_design_local(
    data::AxSDPJointFullCoeffDesignQCQPData;
    show_output::Symbol = :off,
    factor_bound::Union{Nothing, Real} = nothing,
    psd_bound::Union{Nothing, Real} = nothing,
    equality_tolerance::Real = 0.0,
    add_psd_cuts::Bool = true,
    warm_start = nothing,
    ipopt_params = Dict{String, Any}(),
    alpha_bound::Union{Nothing, Real} = nothing,
    rho_bound::Union{Nothing, Real} = nothing,
    beta_bound::Union{Nothing, Real} = nothing,
    allow_weight_on_x0::Bool = false,
)
    ctx = _validate_full_coeff_design_input_data(data)
    equality_tolerance >= 0 || error("equality_tolerance must be nonnegative.")
    isnothing(factor_bound) || factor_bound > 0 || error("factor_bound must be positive when provided.")
    isnothing(psd_bound) || psd_bound > 0 || error("psd_bound must be positive when provided.")

    alpha_bound_value = isnothing(alpha_bound) ? _default_box_upper(data.alpha0_flat) : float(alpha_bound)
    rho_bound_value = isnothing(rho_bound) ? _default_box_upper(data.rho0_flat) : float(rho_bound)
    beta_bound_value = isnothing(beta_bound) ? _default_box_upper(data.beta0_flat) : float(beta_bound)

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

    @variable(model, -alpha_bound_value <= alpha_vars[1:length(data.alpha_pairs)] <= alpha_bound_value)
    @variable(model, -rho_bound_value <= rho_vars[1:length(data.rho_pairs)] <= rho_bound_value)
    @variable(model, -beta_bound_value <= beta_vars[1:length(data.beta_pairs)] <= beta_bound_value)
    @variable(model, omega[1:(data.N + 1)] >= 0)
    @constraint(model, sum(omega) == 1.0)
    if !allow_weight_on_x0
        @constraint(model, omega[1] == 0.0)
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
        lambda_x, lambda_y, lambda_barx, lambda_Abarx, lambda_Ax, lambda_gAy,
        nu, mu_x, mu_y, M, Z_minus, Z_plus, W, P_minus, P_plus, S, warm_start,
    )
    _set_vector_start!(alpha_vars, _warm_get(warm_start, :alpha))
    _set_vector_start!(rho_vars, _warm_get(warm_start, :rho_xy))
    _set_vector_start!(beta_vars, _warm_get(warm_start, :beta))
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

    x_param_vars = vcat(collect(alpha_vars), collect(rho_vars))
    y_param_vars = collect(beta_vars)
    for i in 1:ctx.nG
        for j in 1:i
            expr = data.C[i, j]
            expr = _add_param_bundle_sequence_entry(expr, lambda_x, data.Aeq_x_const, data.Aeq_x_params, x_param_vars, i, j)
            expr = _add_param_bundle_sequence_entry(expr, lambda_y, data.Aeq_y_const, data.Aeq_y_params, y_param_vars, i, j)
            expr = _add_param_bundle_sequence_entry(expr, lambda_barx, data.Aeq_barx_const, data.Aeq_barx_omega, omega, i, j)
            expr = _add_param_bundle_sequence_entry(expr, lambda_Abarx, data.Aeq_Abarx_const, data.Aeq_Abarx_omega, omega, i, j)
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

    solution = _collect_full_coeff_design_solution(
        model, data, alpha_vars, rho_vars, beta_vars, omega,
        lambda_x, lambda_y, lambda_barx, lambda_Abarx, lambda_Ax, lambda_gAy,
        nu, mu_x, mu_y, M, Z_minus, Z_plus, W, P_minus, P_plus, S,
    )
    return (model = model, solution = solution)
end

