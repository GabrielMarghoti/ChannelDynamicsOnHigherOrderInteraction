using DifferentialEquations
using LinearAlgebra
using Random
using Statistics
using LaTeXStrings
using Plots

gr() 
default(fontfamily="Computer Modern", linewidth=2, label=nothing, grid=false, framestyle=:box)

# Base output directory
base_out_dir = "figures/kuramoto_var_K1_K2_simmetric_Bijk"

# ==============================================================================
# 1. Dynamical System Definition
# ==============================================================================

function dynamic_kuramoto!(dy, y, p, t)
    ω, A, B, K1, K2, τ, N = p
    
    # Memory views for phases (first N) and transmission variables (N x N)
    θ = @view y[1:N]
    u = reshape(@view(y[N+1:end]), N, N)
    
    dθ = @view dy[1:N]
    du = reshape(@view(dy[N+1:end]), N, N)
    
    # --------------------------------------------------------------------------
    # Phase Dynamics
    # \dot{\theta}_i = \omega_i + \sum_j u_{ij} \sin(\theta_j - \theta_i)
    # --------------------------------------------------------------------------
    for i in 1:N
        dθ[i] = ω[i]
        for j in 1:N
            dθ[i] += u[i,j] * sin(θ[j] - θ[i])
        end
    end
    
    # --------------------------------------------------------------------------
    # Transmission Channel Dynamics (Non-Equilibrium Inertia)
    # \tau \dot{u}_{ij} = -u_{ij} + K_1 A_{ij} + K_2 \sum_k B_{ijk} \cos(\theta_k - \theta_i)
    # --------------------------------------------------------------------------
    for i in 1:N
        for j in 1:N
            local_field = 0.0
            for k in 1:N
                if B[i,j,k] != 0.0
                    local_field += B[i,j,k] * cos(θ[k] - θ[i])
                end
            end
            du[i,j] = (-u[i,j] + K1 * A[i,j] + K2 * local_field) / τ
        end
    end
end

# ==============================================================================
# 2. Order Parameter Calculation
# ==============================================================================

function kuramoto_order_parameter(θ_array)
    # R = | (1/N) * \sum e^{i \theta_j} |
    N = length(θ_array)
    return abs(mean(exp.(im .* θ_array)))
end

# ==============================================================================
# 3. Main Simulation and Parameter Sweep
# ==============================================================================

function run_parameter_sweep(N = 20, τ = 0.05)
    Random.seed!(42) # For reproducibility
    
    # System Parameters
    tspan = (0.0, 100.0)  # Simulation time
    t_eq = 50.0           # Time to discard as transient
    
    # Frequencies drawn from a standard normal distribution
    ω = randn(N) * 0.1 
    
    # Structural Topology (Random Erdős–Rényi)
    p_edge = 0.3
    A = float.(rand(N, N) .< p_edge) 
    B = float.(rand(N, N, N) .< (p_edge)) 
    
    # Symmetrize environmental influence
    for i in 1:N, j in 1:N, k in 1:N
        B[i,j,k] = B[i,k,j]
    end

    # Sweep Grids
    K1_vals = 10.0 .^range(-3, 0, length=11)#range(0.0, 1.0, length=11)
    K2_vals = 10.0 .^range(-3, 0, length=11)#range(0.0, 1.0, length=11)
    
    # Result Matrix
    R_matrix = zeros(length(K1_vals), length(K2_vals))
    
    println("  -> Grid: $(length(K1_vals))x$(length(K2_vals))")
    
    for (i, K1) in enumerate(K1_vals)
        for (j, K2) in enumerate(K2_vals)
            
            # Initial conditions: random phases in [0, 2π], initial coupling at 0
            θ0 = rand(N) .* 2π
            u0 = zeros(N * N)
            y0 = vcat(θ0, u0)
            
            p = (ω, A, B, K1, K2, τ, N)
            prob = ODEProblem(dynamic_kuramoto!, y0, tspan, p)
            
            # Solve the system
            sol = solve(prob, Tsit5(), saveat=0.5, reltol=1e-6, abstol=1e-6)
            
            # Extract dynamics post-equilibration
            eq_indices = findall(x -> x >= t_eq, sol.t)
            
            # Time-average the order parameter
            R_avg = 0.0
            for idx in eq_indices
                θ_state = sol.u[idx][1:N]
                R_avg += kuramoto_order_parameter(θ_state)
            end
            R_avg /= length(eq_indices)
            
            R_matrix[i, j] = R_avg
        end
        println("  -> Completed K1 = $(round(K1, digits=2))")
    end
    
    println("  -> Sweep complete.")
    return K1_vals, K2_vals, R_matrix
end

# ==============================================================================
# 4. Execution Across Tau Regimes
# ==============================================================================

# Define the tau values for the sweep
tau_configs = [
    ("low_tau", 0.001),         # Adiabatic limit (Recovers HOI topology)
]

for (label, τ_val) in tau_configs
    println("\n=======================================================")
    println("Running sweep for regime: $label (τ = $τ_val)")
    println("=======================================================\n")
    
    # Create specific subfolder (mkpath creates parent directories if needed)
    out_dir = joinpath(base_out_dir, label)
    mkpath(out_dir)
    
    # Execute the sweep
    K1_grid, K2_grid, R_results = run_parameter_sweep(20, τ_val)
    
    # Generate the heatmap
    p = heatmap(K1_grid, K2_grid, R_results',
                color = :viridis,
                xlabel = L"K_1",
                ylabel = L"K_2",
                title = L"\tau = %$τ_val",
                colorbar_title = "R",
                framestyle = :box,
                size = (600, 500),
                xscale = :log10,
                yscale = :log10,
                right_margin = 5Plots.mm)
    
    # Save as publication-ready PNG and PDF
    out_path_png = joinpath(out_dir, "heatmap_tau_$(τ_val).png")
    out_path_pdf = joinpath(out_dir, "heatmap_tau_$(τ_val).pdf")
    
    savefig(p, out_path_png)
    savefig(p, out_path_pdf)
    
    println("\n[Success] Heatmaps saved to:")
    println(" - $out_path_png")
    println(" - $out_path_pdf")
end