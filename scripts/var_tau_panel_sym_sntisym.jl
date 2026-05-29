using DifferentialEquations
using LinearAlgebra
using Random
using Statistics
using LaTeXStrings
using Plots

gr()
default(fontfamily = "Computer Modern", linewidth = 2, label = nothing,
        grid = false, framestyle = :box)

# ==============================================================================
# OUTPUT DIRECTORY
# ==============================================================================
const BASE_OUT_DIR = "figures/kuramoto_tau_sweep"

# ==============================================================================
# 1. DYNAMICAL SYSTEM
# ==============================================================================
function dynamic_kuramoto!(dy, y, p, t)
    ω, A, B, K1, K2, τ, N = p

    θ  = @view y[1:N]
    u  = reshape(@view(y[N+1:end]), N, N)
    dθ = @view dy[1:N]
    du = reshape(@view(dy[N+1:end]), N, N)

    # Phase dynamics (Normalized by N)
    for i in 1:N
        dθ[i] = ω[i]
        for j in 1:N
            dθ[i] += u[i, j] * sin(θ[j] - θ[i]) / N        
        end
    end

    # Transmission dynamics (Normalized by N)
    for i in 1:N
        for j in 1:N
            local_field = 0.0
            for k in 1:N
                local_field += B[i, j, k] * cos(θ[k] - θ[i])
            end
            driving  = (K1 * A[i, j] + K2 * local_field / N) 
            du[i, j] = (-u[i, j] + driving) / τ
        end
    end
end

# ==============================================================================
# 2. NETWORK GENERATION (SYMMETRIC & ANTISYMMETRIC)
# ==============================================================================
function build_network(N::Int, sym_type::Symbol, rng::AbstractRNG)
    A = ones(Float64, N, N)
    B = zeros(Float64, N, N, N)

    for i in 1:N
        A[i, i] = 0.0
    end

    for i in 1:N, j in 1:N, k in (j+1):N
        if i != j && i != k
            B[i, j, k] = 1.0
            if sym_type == :symmetric
                B[i, k, j] = 1.0
            elseif sym_type == :antisymmetric
                B[i, k, j] = -1.0
            end
        end
    end

    return A, B
end

# ==============================================================================
# 3. ORDER PARAMETERS
# ==============================================================================
function order_parameter(θ::AbstractVector{<:Real}, q::Int)
    return abs(sum(exp.(im * q .* θ)) / length(θ))
end

# ==============================================================================
# 4. SWEEP EXECUTION & PLOTTING
# ==============================================================================
function run_and_plot_combined_sweep()
    mkpath(BASE_OUT_DIR)
    
    N = 20
    τ_vals = 10.0 .^ range(-1.1, 2.0, length=100)
    K_pairs = [(0.5, 0.0), (0.5, 2.0), (0.5, 10.0)]
    
    tspan = (0.0, 500.0)
    t_eq  = 300.0
    seed  = 42

    rng = MersenneTwister(seed)
    ω = randn(rng, N) .* 0.1
    
    # Generate identical base topologies for strict 1-to-1 comparison
    A, B_sym  = build_network(N, :symmetric, rng)
    _, B_anti = build_network(N, :antisymmetric, rng)

    θ0_base = range(0, 2π, length=N)
    y0_init = vcat(θ0_base, zeros(N * N))

    # Color palette
    c_R1 = :steelblue; c_R2 = :forestgreen; c_R4 = :darkorange; c_R8 = :crimson

    println("Running combined symmetry sweep across $(length(τ_vals)) τ points | N=$(N)")

    for (p_idx, (K1, K2)) in enumerate(K_pairs)
        println("  -> Processing Configuration: K1=$K1, K2=$K2")
        
        # Preallocate result arrays for this specific K pair
        res_sym  = Dict(q => zeros(length(τ_vals)) for q in [1, 2, 4, 8])
        res_anti = Dict(q => zeros(length(τ_vals)) for q in [1, 2, 4, 8])

        Threads.@threads for t_idx in 1:length(τ_vals)
            τ = τ_vals[t_idx]
            
            # 1. Solve Symmetric
            p_sym = (ω, A, B_sym, K1, K2, τ, N)
            prob_sym = ODEProblem(dynamic_kuramoto!, y0_init, tspan, p_sym)
            sol_sym = solve(prob_sym, Tsit5(), saveat=0.5, reltol=1e-6, abstol=1e-6)
            eq_idx_sym = findall(t -> t >= t_eq, sol_sym.t)
            
            res_sym[1][t_idx] = mean([order_parameter(sol_sym.u[idx][1:N], 1) for idx in eq_idx_sym])
            res_sym[2][t_idx] = mean([order_parameter(sol_sym.u[idx][1:N], 2) for idx in eq_idx_sym])
            res_sym[4][t_idx] = mean([order_parameter(sol_sym.u[idx][1:N], 4) for idx in eq_idx_sym])
            res_sym[8][t_idx] = mean([order_parameter(sol_sym.u[idx][1:N], 8) for idx in eq_idx_sym])

            # 2. Solve Antisymmetric
            p_anti = (ω, A, B_anti, K1, K2, τ, N)
            prob_anti = ODEProblem(dynamic_kuramoto!, y0_init, tspan, p_anti)
            sol_anti = solve(prob_anti, Tsit5(), saveat=0.5, reltol=1e-6, abstol=1e-6)
            eq_idx_anti = findall(t -> t >= t_eq, sol_anti.t)
            
            res_anti[1][t_idx] = mean([order_parameter(sol_anti.u[idx][1:N], 1) for idx in eq_idx_anti])
            res_anti[2][t_idx] = mean([order_parameter(sol_anti.u[idx][1:N], 2) for idx in eq_idx_anti])
            res_anti[4][t_idx] = mean([order_parameter(sol_anti.u[idx][1:N], 4) for idx in eq_idx_anti])
            res_anti[8][t_idx] = mean([order_parameter(sol_anti.u[idx][1:N], 8) for idx in eq_idx_anti])
        end

        # --- Generate 2-Panel Plot for current K pair ---
        base_plot_args = (
            xscale = :log10,
            xlims  = (minimum(τ_vals), maximum(τ_vals)),
            ylims  = (0.0, 1.05),
            xlabel = L"\tau \text{ (Inertia)}",
        )

        # Panel 1: Symmetric
        p1 = plot(title=L"\text{Symmetric } (B_{ijk} = B_{ikj})", ylabel="Coherence "*L"(R_q)", legend=:topright; base_plot_args...)
        plot!(p1, τ_vals, res_sym[1], label=L"R_1", color=c_R1, linestyle=:solid)
        plot!(p1, τ_vals, res_sym[2], label=L"R_2", color=c_R2, linestyle=:dash)
        plot!(p1, τ_vals, res_sym[4], label=L"R_4", color=c_R4, linestyle=:dot)
        # Uncomment to include R8
         plot!(p1, τ_vals, res_sym[8], label=L"R_8", color=c_R8, linestyle=:dashdot)

        # Panel 2: Antisymmetric
        p2 = plot(title=L"\text{Antisymmetric } (B_{ijk} = -B_{ikj})", legend=false; base_plot_args...)
        plot!(p2, τ_vals, res_anti[1], color=c_R1, linestyle=:solid)
        plot!(p2, τ_vals, res_anti[2], color=c_R2, linestyle=:dash)
        plot!(p2, τ_vals, res_anti[4], color=c_R4, linestyle=:dot)
        # Uncomment to include R8
         plot!(p2, τ_vals, res_anti[8], color=c_R8, linestyle=:dashdot)

        combined_plot = plot(p1, p2, 
                             layout = (1, 2), 
                             size = (900, 400), 
                             left_margin = 5Plots.mm, 
                             bottom_margin = 5Plots.mm,
                             plot_title = L"K_1=%$K1, \ K_2=%$K2")

        # Save
        k_label = "K1_$(replace(string(K1), "."=>"p"))_K2_$(replace(string(K2), "."=>"p"))"
        out_file = joinpath(BASE_OUT_DIR, "tau_sweep_sym_vs_anti_$(k_label).png")
        savefig(combined_plot, out_file)
        println("    [✓] Saved Panel → $out_file")
    end
end

# Execute
run_and_plot_combined_sweep()