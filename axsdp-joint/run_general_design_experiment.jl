function _normalize_design_mode(raw_mode)
    mode = lowercase(strip(String(raw_mode)))
    if mode in ("single_tau", "single-tau", "alm", "default", "baseline")
        return :single_tau
    elseif mode in ("multi_tau", "multi-tau", "vector_tau", "vector-tau")
        return :multi_tau
    elseif mode in ("abcd", "four_block", "four-block")
        return :abcd
    elseif mode in ("full_coeff", "full-coeff", "full_coefficients", "full-coefficients")
        return :full_coeff
    end
    error(
        "Unknown design mode '$raw_mode'. Supported modes: " *
        "single_tau, multi_tau, abcd, full_coeff.",
    )
end


const DESIGN_MODE = _normalize_design_mode(
    get(ENV, "AXSDP_DESIGN_MODE", length(ARGS) >= 1 ? ARGS[1] : "single_tau"),
)

if DESIGN_MODE == :single_tau
    include("BnB_PEP_axsdp_joint_design.jl")
elseif DESIGN_MODE == :multi_tau
    include("BnB_PEP_axsdp_joint_design_multi_tau.jl")
elseif DESIGN_MODE == :abcd
    include("BnB_PEP_axsdp_joint_design_abcd.jl")
elseif DESIGN_MODE == :full_coeff
    include("BnB_PEP_axsdp_joint_design_full_coefficients.jl")
else
    error("Unsupported design mode: $DESIGN_MODE")
end

using Dates
using Serialization


function _design_mode_label(mode::Symbol)
    if mode == :single_tau
        return "single_tau"
    elseif mode == :multi_tau
        return "multi_tau"
    elseif mode == :abcd
        return "abcd"
    elseif mode == :full_coeff
        return "full_coeff"
    end
    return string(mode)
end


function _parse_bool_env(name::AbstractString, default::Bool)
    raw = lowercase(strip(get(ENV, name, default ? "1" : "0")))
    if raw in ("1", "true", "yes", "on")
        return true
    elseif raw in ("0", "false", "no", "off")
        return false
    end
    error("Environment variable $name must be a boolean-like value, got '$raw'.")
end


function _parse_float_env(name::AbstractString, default::Real)
    raw = get(ENV, name, string(default))
    try
        return parse(Float64, raw)
    catch err
        error("Environment variable $name must be a Float64, got '$raw'. Original error: $err")
    end
end


function _parse_int_env(name::AbstractString, default::Integer)
    raw = get(ENV, name, string(default))
    try
        return parse(Int, raw)
    catch err
        error("Environment variable $name must be an Int, got '$raw'. Original error: $err")
    end
end


function _parse_optional_float_env(name::AbstractString)
    raw = strip(get(ENV, name, ""))
    isempty(raw) && return nothing
    try
        return parse(Float64, raw)
    catch err
        error("Environment variable $name must be empty or Float64, got '$raw'. Original error: $err")
    end
end


function _runtime_options()
    thread_default = try
        parse(Int, get(ENV, "JULIA_NUM_THREADS", "4"))
    catch
        4
    end

    return (
        resume_enabled = _parse_bool_env("AXSDP_RESUME", true),
        allow_weight_on_x0 = _parse_bool_env("AXSDP_ALLOW_WEIGHT_ON_X0", false),
        enforce_default_stability = _parse_bool_env("AXSDP_ENFORCE_DEFAULT_STABILITY", true),
        add_psd_cuts = _parse_bool_env("AXSDP_ADD_PSD_CUTS", true),
        show_solver_output = _parse_bool_env("AXSDP_SHOW_SOLVER_OUTPUT", true),
        equality_tolerance = _parse_float_env("AXSDP_EQUALITY_TOL", 1e-6),
        local_max_iter = _parse_int_env("AXSDP_LOCAL_MAX_ITER", 3000),
        local_tol = _parse_float_env("AXSDP_LOCAL_TOL", 1e-7),
        local_acceptable_tol = _parse_float_env("AXSDP_LOCAL_ACCEPTABLE_TOL", 1e-6),
        gurobi_threads = _parse_int_env("AXSDP_GUROBI_THREADS", max(thread_default, 1)),
        gurobi_mipfocus = _parse_int_env("AXSDP_GUROBI_MIPFOCUS", 1),
        rho_dual_upper = _parse_float_env("AXSDP_RHO_DUAL_UPPER", 2.0),
        smoothness_L = _parse_float_env("AXSDP_SMOOTHNESS_L", 1.0),
        mu_A = _parse_float_env("AXSDP_MU_A", 0.1),
        L_A = _parse_float_env("AXSDP_L_A", 1.0),
        rho_dual0 = _parse_float_env("AXSDP_RHO_DUAL0", 1.0),
        eta0 = _parse_optional_float_env("AXSDP_ETA0"),
        Rx2 = _parse_float_env("AXSDP_RX2", 1.0),
        Ry2 = _parse_float_env("AXSDP_RY2", 1.0),
    )
end


function _println_summary_pair(io, key, value)
    println(io, rpad(string(key), 24), " = ", value)
end


function _print_if_hasproperty(io, sol, field::Symbol)
    if hasproperty(sol, field)
        _println_summary_pair(io, field, getproperty(sol, field))
    end
end


function _write_solution_block(io, title, sol)
    println(io, title)
    println(io, repeat("-", length(title)))
    _println_summary_pair(io, "termination_status", sol.termination_status)
    _println_summary_pair(io, "primal_status", sol.primal_status)
    _println_summary_pair(io, "dual_status", sol.dual_status)
    _println_summary_pair(io, "raw_status", sol.raw_status)
    _println_summary_pair(io, "solve_time_sec", sol.solve_time_sec)
    _println_summary_pair(io, "result_count", sol.result_count)
    _println_summary_pair(io, "objective", sol.objective)
    _println_summary_pair(io, "objective_bound", sol.objective_bound)

    for field in (:tau, :eta, :rho_dual, :omega, :a, :b, :c, :d, :alpha_flat, :rho_xy_flat, :beta_flat)
        _print_if_hasproperty(io, sol, field)
    end

    if hasproperty(sol, :coefficients) && sol.coefficients !== nothing
        println(io)
        println(io, "alpha =")
        show(io, "text/plain", sol.coefficients.alpha)
        println(io)
        println(io)
        println(io, "rho_xy =")
        show(io, "text/plain", sol.coefficients.rho_xy)
        println(io)
        println(io)
        println(io, "beta =")
        show(io, "text/plain", sol.coefficients.beta)
        println(io)
        println(io)
        println(io, "omega_coeff =")
        show(io, "text/plain", sol.coefficients.omega)
        println(io)
    end

    println(io)
end


function _stage_block(io, title, solution)
    if isnothing(solution)
        println(io, title)
        println(io, repeat("-", length(title)))
        println(io, "pending")
        println(io)
        return
    end
    _write_solution_block(io, title, solution)
end


function _checkpoint_paths(run_dir)
    return NamedTuple{
        (:fixed, :local, :global_qcqp, :state, :summary),
    }(
        (
            joinpath(run_dir, "fixed_sdp_solution.bin"),
            joinpath(run_dir, "local_design_solution.bin"),
            joinpath(run_dir, "global_design_solution.bin"),
            joinpath(run_dir, "run_state.txt"),
            joinpath(run_dir, "summary.txt"),
        ),
    )
end


function _save_checkpoint(path, value)
    open(path, "w") do io
        serialize(io, value)
    end
end


function _load_checkpoint(path)
    open(path, "r") do io
        return deserialize(io)
    end
end


function _write_run_state(
    path;
    run_dir,
    design_mode,
    N,
    qcqp_time_limit,
    resume_enabled,
    run_status,
    stage_fixed,
    stage_local,
    stage_global,
    fixed_checkpoint,
    local_checkpoint,
    global_checkpoint,
)
    open(path, "w") do io
        println(io, "last_update = ", now())
        println(io, "run_dir = ", run_dir)
        println(io, "design_mode = ", _design_mode_label(design_mode))
        println(io, "N = ", N)
        println(io, "qcqp_time_limit_sec = ", qcqp_time_limit)
        println(io, "resume_enabled = ", resume_enabled)
        println(io, "run_status = ", run_status)
        println(io, "stage_fixed = ", stage_fixed)
        println(io, "stage_local = ", stage_local)
        println(io, "stage_global = ", stage_global)
        println(io, "fixed_checkpoint = ", fixed_checkpoint)
        println(io, "local_checkpoint = ", local_checkpoint)
        println(io, "global_checkpoint = ", global_checkpoint)
    end
end


function _write_summary(
    path,
    fixed_summary,
    design_summary;
    run_dir,
    design_mode,
    N,
    qcqp_time_limit,
    run_status,
    fixed_solution = nothing,
    local_solution = nothing,
    global_solution = nothing,
    note = nothing,
)
    open(path, "w") do io
        println(io, "AxSDP design run summary")
        println(io, "=======================")
        println(io, "run_dir = ", run_dir)
        println(io, "timestamp = ", now())
        println(io, "design_mode = ", _design_mode_label(design_mode))
        println(io, "N = ", N)
        println(io, "qcqp_time_limit_sec = ", qcqp_time_limit)
        println(io, "run_status = ", run_status)
        println(io)
        println(io, "fixed_summary = ", fixed_summary)
        println(io, "design_summary = ", design_summary)
        println(io)

        if !isnothing(note)
            println(io, "note = ", note)
            println(io)
        end

        _stage_block(io, "Fixed SDP", fixed_solution)
        _stage_block(io, "Local Design (Ipopt)", local_solution)
        _stage_block(io, "Global Design (Gurobi QCQP)", global_solution)
    end
end


function _base_design_instance_summary(data, meta)
    summary = (
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
    )

    if hasproperty(data, :tau0)
        return merge(
            summary,
            (
                tau0 = data.tau0,
                rho_dual0 = data.rho_dual0,
                omega0 = data.omega0,
            ),
        )
    elseif hasproperty(data, :a0)
        return merge(
            summary,
            (
                a0 = data.a0,
                b0 = data.b0,
                c0 = data.c0,
                d0 = data.d0,
                omega0 = data.omega0,
            ),
        )
    elseif hasproperty(data, :alpha0_flat)
        return merge(
            summary,
            (
                num_alpha = length(data.alpha0_flat),
                num_rho_xy = length(data.rho0_flat),
                num_beta = length(data.beta0_flat),
                omega0 = data.omega0,
            ),
        )
    end

    return summary
end


function _build_design_instance(mode::Symbol, N::Int, opts)
    kwargs = (
        N = N,
        smoothness_L = opts.smoothness_L,
        mu_A = opts.mu_A,
        L_A = opts.L_A,
        rho_dual0 = opts.rho_dual0,
        eta0 = opts.eta0,
        Rx2 = opts.Rx2,
        Ry2 = opts.Ry2,
    )

    if mode == :single_tau
        data, meta = build_default_alm_joint_design_instance(;
            kwargs...,
            allow_weight_on_x0 = opts.allow_weight_on_x0,
        )
        return data, meta, axsdp_joint_design_instance_summary(data, meta)
    elseif mode == :multi_tau
        data, meta = build_default_multi_tau_design_instance(;
            kwargs...,
            allow_weight_on_x0 = opts.allow_weight_on_x0,
        )
        return data, meta, axsdp_joint_multi_tau_instance_summary(data, meta)
    elseif mode == :abcd
        data, meta = build_default_abcd_design_instance(;
            kwargs...,
            allow_weight_on_x0 = opts.allow_weight_on_x0,
        )
        return data, meta, _base_design_instance_summary(data, meta)
    elseif mode == :full_coeff
        data, meta = build_default_full_coeff_design_instance(; kwargs...)
        return data, meta, _base_design_instance_summary(data, meta)
    end

    error("Unsupported design mode: $mode")
end


function _build_design_warm_start(mode::Symbol, fixed_solution, design_data)
    if mode == :single_tau
        return build_design_qcqp_warm_start_from_fixed_sdp_solution(
            fixed_solution,
            design_data;
            scale = 1.01,
        )
    elseif mode == :multi_tau
        return build_multi_tau_qcqp_warm_start_from_fixed_sdp_solution(
            fixed_solution,
            design_data;
            scale = 1.01,
        )
    elseif mode == :abcd
        return build_abcd_qcqp_warm_start_from_fixed_sdp_solution(
            fixed_solution,
            design_data;
            scale = 1.01,
        )
    elseif mode == :full_coeff
        return build_full_coeff_qcqp_warm_start_from_fixed_sdp_solution(
            fixed_solution,
            design_data;
            scale = 1.01,
        )
    end
    error("Unsupported design mode: $mode")
end


function _solve_local_design(mode::Symbol, design_data, warm, opts)
    show_output = opts.show_solver_output ? :on : :off
    common_kwargs = (
        show_output = show_output,
        factor_bound = warm.factor_bound,
        psd_bound = warm.psd_bound,
        equality_tolerance = opts.equality_tolerance,
        add_psd_cuts = opts.add_psd_cuts,
        warm_start = warm,
        ipopt_params = Dict(
            "max_iter" => opts.local_max_iter,
            "tol" => opts.local_tol,
            "acceptable_tol" => opts.local_acceptable_tol,
        ),
    )

    if mode == :single_tau
        return solve_axsdp_joint_alm_design_local(
            design_data;
            common_kwargs...,
            rho_dual_upper = opts.rho_dual_upper,
            allow_weight_on_x0 = opts.allow_weight_on_x0,
            enforce_default_stability = opts.enforce_default_stability,
        )
    elseif mode == :multi_tau
        return solve_axsdp_joint_multi_tau_design_local(
            design_data;
            common_kwargs...,
            rho_dual_upper = opts.rho_dual_upper,
            allow_weight_on_x0 = opts.allow_weight_on_x0,
            enforce_default_stability = opts.enforce_default_stability,
        )
    elseif mode == :abcd
        return solve_axsdp_joint_abcd_design_local(
            design_data;
            common_kwargs...,
            allow_weight_on_x0 = opts.allow_weight_on_x0,
            enforce_default_stability = opts.enforce_default_stability,
        )
    elseif mode == :full_coeff
        return solve_axsdp_joint_full_coeff_design_local(
            design_data;
            common_kwargs...,
            allow_weight_on_x0 = opts.allow_weight_on_x0,
        )
    end

    error("Unsupported design mode: $mode")
end


function _prepare_global_warm_start(mode::Symbol, base_warm, local_solution)
    if isnothing(local_solution) || local_solution.objective === nothing
        return base_warm
    end

    merged_local = merge(base_warm, local_solution)

    if mode == :full_coeff
        return merge(
            merged_local,
            (
                alpha = local_solution.alpha_flat,
                rho_xy = local_solution.rho_xy_flat,
                beta = local_solution.beta_flat,
            ),
        )
    end

    return merged_local
end


function _solve_global_design(mode::Symbol, design_data, warm, opts, qcqp_time_limit::Float64)
    show_output = opts.show_solver_output ? :on : :off
    common_kwargs = (
        show_output = show_output,
        factor_bound = warm.factor_bound,
        psd_bound = warm.psd_bound,
        equality_tolerance = opts.equality_tolerance,
        add_psd_cuts = opts.add_psd_cuts,
        warm_start = warm,
        gurobi_params = Dict(
            "TimeLimit" => qcqp_time_limit,
            "Threads" => opts.gurobi_threads,
            "MIPFocus" => opts.gurobi_mipfocus,
        ),
    )

    if mode == :single_tau
        return solve_axsdp_joint_alm_design_qcqp(
            design_data;
            common_kwargs...,
            rho_dual_upper = opts.rho_dual_upper,
            allow_weight_on_x0 = opts.allow_weight_on_x0,
            enforce_default_stability = opts.enforce_default_stability,
        )
    elseif mode == :multi_tau
        return solve_axsdp_joint_multi_tau_design_qcqp(
            design_data;
            common_kwargs...,
            rho_dual_upper = opts.rho_dual_upper,
            allow_weight_on_x0 = opts.allow_weight_on_x0,
            enforce_default_stability = opts.enforce_default_stability,
        )
    elseif mode == :abcd
        return solve_axsdp_joint_abcd_design_qcqp(
            design_data;
            common_kwargs...,
            allow_weight_on_x0 = opts.allow_weight_on_x0,
            enforce_default_stability = opts.enforce_default_stability,
        )
    elseif mode == :full_coeff
        return solve_axsdp_joint_full_coeff_design_qcqp(
            design_data;
            common_kwargs...,
            allow_weight_on_x0 = opts.allow_weight_on_x0,
        )
    end

    error("Unsupported design mode: $mode")
end


function main()
    design_mode = DESIGN_MODE
    arg_offset = length(ARGS) >= 1 ? 1 : 0
    N = length(ARGS) >= arg_offset + 1 ? parse(Int, ARGS[arg_offset + 1]) : 3
    qcqp_time_limit = length(ARGS) >= arg_offset + 2 ? parse(Float64, ARGS[arg_offset + 2]) : 300.0
    opts = _runtime_options()

    run_dir =
        if length(ARGS) >= arg_offset + 3
            abspath(ARGS[arg_offset + 3])
        else
            timestamp = Dates.format(now(), "yyyymmdd_HHMMSS")
            joinpath(
                pwd(),
                "outputs",
                "axsdp_design_runs",
                string(_design_mode_label(design_mode), "_N", N, "_", timestamp),
            )
        end

    mkpath(run_dir)
    checkpoints = _checkpoint_paths(run_dir)

    fixed_data, fixed_meta = build_default_alm_joint_instance(
        N = N,
        smoothness_L = opts.smoothness_L,
        mu_A = opts.mu_A,
        L_A = opts.L_A,
        rho_dual = opts.rho_dual0,
        Rx2 = opts.Rx2,
        Ry2 = opts.Ry2,
    )
    design_data, design_meta, design_summary = _build_design_instance(design_mode, N, opts)
    fixed_summary = axsdp_joint_instance_summary(fixed_data, fixed_meta)

    open(joinpath(run_dir, "run_config.txt"), "w") do io
        println(io, "run_dir = ", run_dir)
        println(io, "timestamp = ", now())
        println(io, "design_mode = ", _design_mode_label(design_mode))
        println(io, "N = ", N)
        println(io, "qcqp_time_limit_sec = ", qcqp_time_limit)
        println(io, "resume_enabled = ", opts.resume_enabled)
        println(io, "cwd = ", pwd())
        println(io)
        println(io, "runtime_options = ", opts)
        println(io)
        println(io, "fixed_summary = ", fixed_summary)
        println(io, "design_summary = ", design_summary)
    end

    _write_run_state(
        checkpoints.state;
        run_dir = run_dir,
        design_mode = design_mode,
        N = N,
        qcqp_time_limit = qcqp_time_limit,
        resume_enabled = opts.resume_enabled,
        run_status = "initialized",
        stage_fixed = isfile(checkpoints.fixed) ? "checkpoint_found" : "pending",
        stage_local = isfile(checkpoints.local) ? "checkpoint_found" : "pending",
        stage_global = isfile(checkpoints.global_qcqp) ? "checkpoint_found" : "pending",
        fixed_checkpoint = checkpoints.fixed,
        local_checkpoint = checkpoints.local,
        global_checkpoint = checkpoints.global_qcqp,
    )
    _write_summary(
        checkpoints.summary,
        fixed_summary,
        design_summary;
        run_dir = run_dir,
        design_mode = design_mode,
        N = N,
        qcqp_time_limit = qcqp_time_limit,
        run_status = "initialized",
    )

    println("Run directory:   ", run_dir)
    println("Design mode:     ", _design_mode_label(design_mode))
    println("Resume enabled:  ", opts.resume_enabled)
    println()
    println("Fixed-instance summary:")
    for (key, value) in pairs(fixed_summary)
        println("  ", rpad(string(key), 16), " = ", value)
    end
    println()
    println("Design-instance summary:")
    for (key, value) in pairs(design_summary)
        println("  ", rpad(string(key), 16), " = ", value)
    end

    fixed_solution =
        if opts.resume_enabled && isfile(checkpoints.fixed)
            println()
            println("Loading fixed SDP checkpoint: ", checkpoints.fixed)
            _load_checkpoint(checkpoints.fixed)
        else
            _write_run_state(
                checkpoints.state;
                run_dir = run_dir,
                design_mode = design_mode,
                N = N,
                qcqp_time_limit = qcqp_time_limit,
                resume_enabled = opts.resume_enabled,
                run_status = "running_fixed_sdp",
                stage_fixed = "running",
                stage_local = isfile(checkpoints.local) ? "checkpoint_found" : "pending",
                stage_global = isfile(checkpoints.global_qcqp) ? "checkpoint_found" : "pending",
                fixed_checkpoint = checkpoints.fixed,
                local_checkpoint = checkpoints.local,
                global_checkpoint = checkpoints.global_qcqp,
            )
            fixed_sdp_result = solve_axsdp_joint_dual_sdp(fixed_data; show_output = :off)
            _save_checkpoint(checkpoints.fixed, fixed_sdp_result.solution)
            fixed_sdp_result.solution
        end

    println()
    _write_solution_block(stdout, "Fixed SDP", fixed_solution)

    _write_run_state(
        checkpoints.state;
        run_dir = run_dir,
        design_mode = design_mode,
        N = N,
        qcqp_time_limit = qcqp_time_limit,
        resume_enabled = opts.resume_enabled,
        run_status = "fixed_sdp_completed",
        stage_fixed = "completed",
        stage_local = isfile(checkpoints.local) ? "checkpoint_found" : "pending",
        stage_global = isfile(checkpoints.global_qcqp) ? "checkpoint_found" : "pending",
        fixed_checkpoint = checkpoints.fixed,
        local_checkpoint = checkpoints.local,
        global_checkpoint = checkpoints.global_qcqp,
    )
    _write_summary(
        checkpoints.summary,
        fixed_summary,
        design_summary;
        run_dir = run_dir,
        design_mode = design_mode,
        N = N,
        qcqp_time_limit = qcqp_time_limit,
        run_status = "fixed_sdp_completed",
        fixed_solution = fixed_solution,
    )

    if fixed_solution.objective === nothing
        println()
        println("The fixed SDP did not return a primal point, so the design stages were skipped.")
        _write_run_state(
            checkpoints.state;
            run_dir = run_dir,
            design_mode = design_mode,
            N = N,
            qcqp_time_limit = qcqp_time_limit,
            resume_enabled = opts.resume_enabled,
            run_status = "stopped_after_fixed_sdp",
            stage_fixed = "completed",
            stage_local = "skipped",
            stage_global = "skipped",
            fixed_checkpoint = checkpoints.fixed,
            local_checkpoint = checkpoints.local,
            global_checkpoint = checkpoints.global_qcqp,
        )
        _write_summary(
            checkpoints.summary,
            fixed_summary,
            design_summary;
            run_dir = run_dir,
            design_mode = design_mode,
            N = N,
            qcqp_time_limit = qcqp_time_limit,
            run_status = "stopped_after_fixed_sdp",
            fixed_solution = fixed_solution,
            note = "Design stages skipped because the fixed SDP returned no primal point.",
        )
        return
    end

    warm = _build_design_warm_start(design_mode, fixed_solution, design_data)

    println()
    println("Warm-start bounds:")
    println("  factor_bound = ", warm.factor_bound)
    println("  psd_bound    = ", warm.psd_bound)

    local_solution =
        if opts.resume_enabled && isfile(checkpoints.local)
            println()
            println("Loading local design checkpoint: ", checkpoints.local)
            _load_checkpoint(checkpoints.local)
        else
            _write_run_state(
                checkpoints.state;
                run_dir = run_dir,
                design_mode = design_mode,
                N = N,
                qcqp_time_limit = qcqp_time_limit,
                resume_enabled = opts.resume_enabled,
                run_status = "running_local_design",
                stage_fixed = "completed",
                stage_local = "running",
                stage_global = isfile(checkpoints.global_qcqp) ? "checkpoint_found" : "pending",
                fixed_checkpoint = checkpoints.fixed,
                local_checkpoint = checkpoints.local,
                global_checkpoint = checkpoints.global_qcqp,
            )
            local_result = _solve_local_design(design_mode, design_data, warm, opts)
            _save_checkpoint(checkpoints.local, local_result.solution)
            local_result.solution
        end

    println()
    _write_solution_block(stdout, "Local Design (Ipopt)", local_solution)

    _write_run_state(
        checkpoints.state;
        run_dir = run_dir,
        design_mode = design_mode,
        N = N,
        qcqp_time_limit = qcqp_time_limit,
        resume_enabled = opts.resume_enabled,
        run_status = "local_design_completed",
        stage_fixed = "completed",
        stage_local = "completed",
        stage_global = isfile(checkpoints.global_qcqp) ? "checkpoint_found" : "pending",
        fixed_checkpoint = checkpoints.fixed,
        local_checkpoint = checkpoints.local,
        global_checkpoint = checkpoints.global_qcqp,
    )
    _write_summary(
        checkpoints.summary,
        fixed_summary,
        design_summary;
        run_dir = run_dir,
        design_mode = design_mode,
        N = N,
        qcqp_time_limit = qcqp_time_limit,
        run_status = "local_design_completed",
        fixed_solution = fixed_solution,
        local_solution = local_solution,
    )

    global_warm = _prepare_global_warm_start(design_mode, warm, local_solution)

    global_solution =
        if opts.resume_enabled && isfile(checkpoints.global_qcqp)
            println()
            println("Loading global design checkpoint: ", checkpoints.global_qcqp)
            _load_checkpoint(checkpoints.global_qcqp)
        else
            _write_run_state(
                checkpoints.state;
                run_dir = run_dir,
                design_mode = design_mode,
                N = N,
                qcqp_time_limit = qcqp_time_limit,
                resume_enabled = opts.resume_enabled,
                run_status = "running_global_design",
                stage_fixed = "completed",
                stage_local = "completed",
                stage_global = "running",
                fixed_checkpoint = checkpoints.fixed,
                local_checkpoint = checkpoints.local,
                global_checkpoint = checkpoints.global_qcqp,
            )
            global_result = _solve_global_design(
                design_mode,
                design_data,
                global_warm,
                opts,
                qcqp_time_limit,
            )
            _save_checkpoint(checkpoints.global_qcqp, global_result.solution)
            global_result.solution
        end

    println()
    _write_solution_block(stdout, "Global Design (Gurobi QCQP)", global_solution)

    _write_run_state(
        checkpoints.state;
        run_dir = run_dir,
        design_mode = design_mode,
        N = N,
        qcqp_time_limit = qcqp_time_limit,
        resume_enabled = opts.resume_enabled,
        run_status = "completed",
        stage_fixed = "completed",
        stage_local = "completed",
        stage_global = "completed",
        fixed_checkpoint = checkpoints.fixed,
        local_checkpoint = checkpoints.local,
        global_checkpoint = checkpoints.global_qcqp,
    )
    _write_summary(
        checkpoints.summary,
        fixed_summary,
        design_summary;
        run_dir = run_dir,
        design_mode = design_mode,
        N = N,
        qcqp_time_limit = qcqp_time_limit,
        run_status = "completed",
        fixed_solution = fixed_solution,
        local_solution = local_solution,
        global_solution = global_solution,
    )

    println()
    println("Saved summary: ", checkpoints.summary)
    println("Saved state:   ", checkpoints.state)
    println("Saved config:  ", joinpath(run_dir, "run_config.txt"))
end


main()
