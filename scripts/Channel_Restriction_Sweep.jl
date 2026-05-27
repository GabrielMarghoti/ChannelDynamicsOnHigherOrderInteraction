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
const BASE_OUT_DIR = "figures/kuramoto_small_world"
const N            = 12       
const τ            = 0.01     
const SEED         = 42

# ==============================================================================
# 1. DYNAMICAL SYSTEM
# ==============================================================================
function dynamic_kuramoto_sw!(dy, y, p, t)
    ω, A, B, K1, K2, τ, N = p

    θ  = @view y[1:N]
    u  = reshape(@view(y[N+1:end]), N, N)
    dθ = @view dy[1:N]
    du = reshape(@view(dy[N+1:end]), N, N)

    for i in 1:N
        s = ω[i]
        for j in 1:N
            s += u[i, j]
        end
        dθ[i] = s/N
    end

    for i in 1:N, j in 1:N
        lf = 0.0
        for k in 1:N
            lf += B[i, j, k] * cos(θ[k] - θ[i])
        end
        
        du[i, j] = (-u[i, j] + (K1 * A[i, j] + K2 * lf/N) * sin(θ[j] - θ[i])) /(τ)
    end
end

function Rq_order(θ::AbstractVector{<:Real}, q::Int)
    return abs(sum(exp.(im * q .* θ)) / length(θ))
end

# ==============================================================================
# 2. SMALL WORLD NETWORK CONSTRUCTION & SYMMETRY
# ==============================================================================
function build_sw_network(N::Int, p_shortcut::Float64, sym_type::Symbol, rng::AbstractRNG)
    L = zeros(Float64, N, N)
    for i in 1:N
        for d in 1:2
            right = mod1(i + d, N)
            L[i, right] = 1.0
            L[right, i] = 1.0
        end
    end

    for i in 1:N, j in i+1:N
        if L[i,j] == 0.0 && rand(rng) < p_shortcut
            L[i,j] = 1.0
            L[j,i] = 1.0
        end
    end

    A = zeros(Float64, N, N)
    B = zeros(Float64, N, N, N)

    for i in 1:N
        neighbors = findall(x -> x > 0, L[:, i])
        
        for j in neighbors
            if sym_type == :symmetric
                A[i, j] = 1.0
            elseif sym_type == :antisymmetric
                A[i, j] = i < j ? 1.0 : -1.0
            elseif sym_type == :asymmetric
                A[i, j] = rand(rng) > 0.5 ? 1.0 : 0.0
            end
            
            for k in neighbors
                if j >= k
                    continue
                end
                
                if sym_type == :symmetric
                    B[i, j, k] = 1.0
                    B[i, k, j] = 1.0
                elseif sym_type == :antisymmetric
                    B[i, j, k] = 1.0
                    B[i, k, j] = -1.0
                elseif sym_type == :asymmetric
                    if rand(rng) > 0.5
                        B[i, j, k] = 1.0
                        B[i, k, j] = 0.0
                    else
                        B[i, j, k] = 0.0
                        B[i, k, j] = 1.0
                    end
                end
            end
        end
    end
    return A, B
end

# ==============================================================================
# 3. SWEEP EXECUTION
# ==============================================================================
function run_topology_sweep()
    rng = MersenneTwister(SEED)
    ω   = randn(rng, N) .* 0.1
    
    p_vals = 10.0 .^ range(log10(0.001), log10(1.0), length=11)
    
    panels = [
        (1.0, 0.0, L"\text{Pairwise Only } (K_1>0, K_2=0)"),
        (0.0, 1.0, L"\text{High-Order Only } (K_1=0, K_2>0)"),
        (1.0, 1.0, L"\text{Mixed Dynamics } (K_1=K_2)")
    ]
    
    sym_cases = [
        (:symmetric,     "Symmetric",     :steelblue),
        (:antisymmetric, "Antisymmetric", :crimson),
        (:asymmetric,    "Mixed (Asym)",  :forestgreen)
    ]
    
    results_R1 = Dict(i => Dict(sym => zeros(length(p_vals)) for (sym, _, _) in sym_cases) for i in 1:3)
    results_R2 = Dict(i => Dict(sym => zeros(length(p_vals)) for (sym, _, _) in sym_cases) for i in 1:3)
    
    tspan = (0.0, 150.0)
    t_eq  = 100.0

    println("Sweeping Small-World Shortcuts sequentially (N=$(N))...")

    # Inverted loops: configurations outer, parameter sweep inner
    for (sym_type, _, _) in sym_cases
        for (panel_idx, (K1, K2, _)) in enumerate(panels)
            
            local_rng = MersenneTwister(SEED + panel_idx)
            # Initialize state once per configuration
            θ0 = rand(local_rng, N) .* 2π
            y0 = vcat(θ0, zeros(N * N))
            
            for p_idx in 1:length(p_vals)
                p_shortcut = p_vals[p_idx]
                
                A, B = build_sw_network(N, p_shortcut, sym_type, local_rng)
                p_sys = (ω, A, B, K1, K2, τ, N)
                
                prob  = ODEProblem(dynamic_kuramoto_sw!, y0, tspan, p_sys)
                sol   = solve(prob, Tsit5(), saveat=0.5, reltol=1e-5, abstol=1e-5)
                
                eq_idx = findall(t -> t >= t_eq, sol.t)
                R1_avg = mean([Rq_order(sol.u[idx][1:N], 1) for idx in eq_idx])
                R2_avg = mean([Rq_order(sol.u[idx][1:N], 2) for idx in eq_idx])
                
                results_R1[panel_idx][sym_type][p_idx] = R1_avg
                results_R2[panel_idx][sym_type][p_idx] = R2_avg
                
                # State continuation: update initial condition for next step
                #y0 = sol.u[end]
            end
        end
    end
    
    return p_vals, results_R1, results_R2, panels, sym_cases
end

# ==============================================================================
# 4. PLOTTING & SAVE
# ==============================================================================
function plot_results(p_vals, res_R1, res_R2, panels, sym_cases)
    mkpath(BASE_OUT_DIR)
    
    plot_list = []
    
    for (panel_idx, (_, _, title_str)) in enumerate(panels)
        plt = plot(
            title  = title_str,
            xscale = :log10,
            xlims  = (0.001, 1.0),
            ylims  = (0.0, 1.0),
            xlabel = panel_idx == 2 ? L"\text{Shortcut Probability } (p)" : "",
            ylabel = panel_idx == 1 ? L"\text{Coherence } (R_1, R_2)" : "",
            legend = panel_idx == 3 ? :bottomright : false,
            titlefontsize = 11
        )
        
        for (sym_type, label_str, color) in sym_cases
            # Plot R1 (Solid)
            plot!(plt, p_vals, res_R1[panel_idx][sym_type];
                  label = label_str,
                  color = color,
                  linestyle = :solid,
                  marker = :circle,
                  markersize = 3,
                  markerstrokewidth = 0)
                  
            # Plot R2 (Dashed, no legend label to avoid clutter)
            plot!(plt, p_vals, res_R2[panel_idx][sym_type];
                  label = "",
                  color = color,
                  linestyle = :dash,
                  marker = :utriangle,
                  markersize = 3,
                  markerstrokewidth = 0)
        end
        push!(plot_list, plt)
    end
    
    panel_fig = plot(plot_list..., layout=(1, 3), size=(1200, 400), 
                     left_margin=5Plots.mm, bottom_margin=5Plots.mm)
                     
    out_path = joinpath(BASE_OUT_DIR, "small_world_symmetry_panel.png")
    savefig(panel_fig, out_path)
    println("  [✓] Saved → $out_path")
end

p_vals, res_R1, res_R2, panels, sym_cases = run_topology_sweep()
plot_results(p_vals, res_R1, res_R2, panels, sym_cases)