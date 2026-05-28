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
            dθ[i] += u[i, j] / N        
        end
    end

    # Transmission dynamics (Normalized by N)
    for i in 1:N
        for j in 1:N
            local_field = 0.0
            for k in 1:N
                local_field += B[i, j, k] * cos(θ[k] - θ[i])
            end
            driving  = (K1 * A[i, j] + K2 * local_field) * sin(θ[j] - θ[i])
            du[i, j] = (-u[i, j] + driving) / τ
        end
    end
end

# ==============================================================================
# 2. NETWORK GENERATION (SYMMETRIC ONLY)
# ==============================================================================
function build_symmetric_network(N::Int, rng::AbstractRNG)
    A = ones(Float64, N, N)
    B = zeros(Float64, N, N, N)

    for i in 1:N
        A[i, i] = 0.0
    end

    # Higher-order tensor B (Symmetric)
    for i in 1:N, j in 1:N, k in (j+1):N
        if i != j && i != k
            B[i, j, k] = 1.0
            B[i, k, j] = 1.0
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
# 4. SWEEP EXECUTION
# ==============================================================================
function run_tau_sweep(N, τ_vals, K_pairs; seed=42)
    n_tau = length(τ_vals)
    n_panels = length(K_pairs)

    R1_res = zeros(Float64, n_panels, n_tau)
    R2_res = zeros(Float64, n_panels, n_tau)
    R4_res = zeros(Float64, n_panels, n_tau)
    R8_res = zeros(Float64, n_panels, n_tau)

    tspan = (0.0, 350.0)
    t_eq  = 200.0

    rng = MersenneTwister(seed)
    ω = randn(rng, N) .* 0.1
    A, B = build_symmetric_network(N, rng)

    println("Sweeping τ across $(n_tau) points | N=$(N)")

    # Generate fixed initial conditions so the manifold comparison is exact
    θ0_base = range(0, 2π, length=N)
    y0_init = vcat(θ0_base, zeros(N * N))

    for (p_idx, (K1, K2)) in enumerate(K_pairs)
        println("  -> Panel $p_idx: K1=$K1, K2=$K2")
        
        Threads.@threads for (t_idx, τ) in collect(enumerate(τ_vals))
            p_sys = (ω, A, B, K1, K2, τ, N)
            prob = ODEProblem(dynamic_kuramoto!, y0_init, tspan, p_sys)
            sol = solve(prob, Tsit5(), saveat=0.5, reltol=1e-6, abstol=1e-6)

            eq_idx = findall(t -> t >= t_eq, sol.t)
            
            R1_res[p_idx, t_idx] = mean([order_parameter(sol.u[idx][1:N], 1) for idx in eq_idx])
            R2_res[p_idx, t_idx] = mean([order_parameter(sol.u[idx][1:N], 2) for idx in eq_idx])
            R4_res[p_idx, t_idx] = mean([order_parameter(sol.u[idx][1:N], 4) for idx in eq_idx])
            R8_res[p_idx, t_idx] = mean([order_parameter(sol.u[idx][1:N], 8) for idx in eq_idx])
        end
    end

    return R1_res, R2_res, R4_res, R8_res
end

# ==============================================================================
# 5. PLOTTING
# ==============================================================================
function generate_tau_panels()
    mkpath(BASE_OUT_DIR)
    
    N = 20
    τ_vals = 10.0 .^ range(-1.1, 2, length=100)
    
    # 3 Configurations: Pairwise dominant, Mixed, High Coupling
    K_pairs = [(0.2, 0.0), (0.2, 0.2), (0.6, 0.8)]
    
    R1_res, R2_res, R4_res, R8_res = run_tau_sweep(N, τ_vals, K_pairs)

    plot_list = []
    
    # Color palette for distinct q values
    c_R1 = :steelblue
    c_R2 = :forestgreen
    c_R4 = :darkorange
    c_R8 = :crimson

    for (p_idx, (K1, K2)) in enumerate(K_pairs)
        plt = plot(
            xscale = :log10,
            xlims  = (minimum(τ_vals), maximum(τ_vals)),
            ylims  = (0.0, 1.0),
            xlabel = p_idx == 3 ? L"\tau" : "",
            ylabel = "Coherence "*L"(R_q)",
            title  = L"K_1=%$K1, K_2=%$K2",
            legend = p_idx == 1 ? :topright : false
        )

        # R1
        plot!(plt, τ_vals, R1_res[p_idx, :];
              label = L"R_1", color = c_R1, linestyle = :solid)
        # R2
        plot!(plt, τ_vals, R2_res[p_idx, :];
              label = L"R_2", color = c_R2, linestyle = :dash)
        # R4
        plot!(plt, τ_vals, R4_res[p_idx, :];
              label = L"R_4", color = c_R4, linestyle = :dot)
        # R8
        #plot!(plt, τ_vals, R8_res[p_idx, :];
        #      label = L"R_8", color = c_R8, linestyle = :dashdot)
              
        push!(plot_list, plt)
    end

    # Combine into 3x1 Layout
    panel_plot = plot(plot_list..., 
                      layout = (3, 1), 
                      size = (500, 600), 
                      left_margin = 5Plots.mm, 
                      bottom_margin = 5Plots.mm)

    out_file = joinpath(BASE_OUT_DIR, "tau_sweep_panels_N$(N)_3x1.png")
    savefig(panel_plot, out_file)
    println("\n  [✓] Saved 3x1 Panel → $out_file")
end

# Execute
generate_tau_panels()