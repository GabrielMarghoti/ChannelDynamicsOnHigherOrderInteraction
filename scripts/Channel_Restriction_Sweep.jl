using DifferentialEquations
using LinearAlgebra
using Statistics
using LaTeXStrings
using Plots
using Random

gr()
default(fontfamily = "Computer Modern", linewidth = 2, label = nothing, dpi=300,
        grid = false, framestyle = :box)

# ==============================================================================
# SETTINGS
# ==============================================================================
const BASE_OUT_DIR = "figures/kuramoto_channel_sweep"
const N            = 20
const SEED         = 42

# ==============================================================================
# 1. DYNAMICAL SYSTEM
# ==============================================================================
function dynamic_kuramoto_restricted!(dy, y, p, t)
    ω, channel_mask, B, K1, K2, τ, N = p

    θ  = @view y[1:N]
    u  = reshape(@view(y[N+1:end]), N, N)
    dθ = @view dy[1:N]
    du = reshape(@view(dy[N+1:end]), N, N)

    for i in 1:N
        s = ω[i]
        for j in 1:N
            s += u[i, j]
        end
        dθ[i] = s
    end

    for i in 1:N, j in 1:N
        if channel_mask[i, j] == 0.0
            du[i, j] = -u[i, j] / τ
            continue
        end
        
        lf = 0.0
        for k in 1:N
            lf += B[i, j, k] * cos(θ[k] - θ[i])
        end
        
        du[i, j] = (-u[i, j] + (K1 + K2 * lf) * sin(θ[j] - θ[i])) / τ
    end
end

function R1_order(θ::AbstractVector{<:Real})
    return abs(sum(exp.(im .* θ)) / length(θ))
end

function build_symmetric_B(N::Int)
    B = ones(Float64, N, N, N)
    for i in 1:N
        B[i, i, :] .= 0.0
        B[i, :, i] .= 0.0
    end
    return B
end

# ==============================================================================
# 2. SWEEP EXECUTION
# ==============================================================================
function run_sweep_for_tau(τ::Float64, n_points::Int)
    rng = MersenneTwister(SEED)
    ω   = randn(rng, N) .* 0.1
    B   = build_symmetric_B(N)
    
    # Log-spaced fractions from 1% to 100%
    fractions = 10.0 .^ range(log10(0.01), log10(1.0), length=n_points)
    
    K1_base = 0.1
    configs = [
        (K1_base, K1_base * 0.1, L"K_2/K_1 = 0.1", :steelblue),
        (K1_base, K1_base * 1.0, L"K_2/K_1 = 1.0", :forestgreen),
        (K1_base, K1_base * 5.0, L"K_2/K_1 = 5.0", :crimson)
    ]
    
    results = Dict(label => zeros(n_points) for (_, _, label, _) in configs)
    tspan = (0.0, 150.0)
    t_eq  = 100.0

    for (i, f) in enumerate(fractions)
        num_active = max(1, round(Int, f * N^2))
        
        mask_flat = zeros(Float64, N^2)
        mask_flat[1:num_active] .= 1.0
        shuffle!(rng, mask_flat)
        mask = reshape(mask_flat, N, N)
        
        for (K1, K2, label, _) in configs
            θ0 = rand(rng, N) .* 2π
            y0 = vcat(θ0, zeros(N * N))
            
            p    = (ω, mask, B, K1, K2, τ, N)
            prob = ODEProblem(dynamic_kuramoto_restricted!, y0, tspan, p)
            sol  = solve(prob, Tsit5(), saveat=0.5, reltol=1e-6, abstol=1e-6)
            
            eq_idx = findall(t -> t >= t_eq, sol.t)
            R1_avg = mean([R1_order(sol.u[idx][1:N]) for idx in eq_idx])
            results[label][i] = R1_avg
        end
    end
    
    return fractions, results, configs
end

# ==============================================================================
# 3. PLOTTING & SAVE
# ==============================================================================
function main()
    mkpath(BASE_OUT_DIR)
    println("Running sweep for Adiabatic Limit (τ = 0.001)...")
    f_vals, res_ad, configs = run_sweep_for_tau(0.001, 25)
    
    println("Running sweep for Dynamic Limit (τ = 1.0)...")
    _, res_dyn, _ = run_sweep_for_tau(1.0, 25)

    # Plot settings
    p_args = (
        xscale = :log10,
        xlims  = (0.01, 1.0),
        ylims  = (0.0, 1.0),
        xlabel = L"\text{Fraction } (N_u/N_\theta^2)",
        ylabel = L"R_1",
        legend = :topleft
    )

    # Adiabatic Panel
    p1 = plot(title=L"\tau = 0.001 \text{ (Adiabatic)}"; p_args...)
    for (_, _, label, color) in configs
        plot!(p1, f_vals, res_ad[label]; label=label, color=color, marker=:circle, markersize=3, markerstrokewidth=0)
    end

    # Dynamic Panel
    p2 = plot(title=L"\tau = 1.0 \text{ (Dynamic)}"; p_args...)
    for (_, _, label, color) in configs
        plot!(p2, f_vals, res_dyn[label]; label=label, color=color, marker=:circle, markersize=3, markerstrokewidth=0)
    end

    panel = plot(p1, p2, layout=(1, 2), size=(1000, 450), left_margin=5Plots.mm, bottom_margin=5Plots.mm)
    
    out_path = joinpath(BASE_OUT_DIR, "tau_comparison_panel.png")
    savefig(panel, out_path)
    println("  [✓] Saved → $out_path")
end

main()