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
BASE_OUT_DIR = "figures/kuramoto_symmetry_panel_var_K1_K2_normalized"

# ==============================================================================
# 1. DYNAMICAL SYSTEM
# ==============================================================================
function dynamic_kuramoto!(dy, y, p, t)
    ω, A, B, K1, K2, τ, N = p

    θ  = @view y[1:N]
    u  = reshape(@view(y[N+1:end]), N, N)
    dθ = @view dy[1:N]
    du = reshape(@view(dy[N+1:end]), N, N)

    # --- Phase dynamics (Normalized by N) ---
    for i in 1:N
        dθ[i] = ω[i]
        for j in 1:N
            dθ[i] += u[i, j]  * sin(θ[j] - θ[i]) / N        
        end
    end

    # --- Transmission dynamics (Normalized by N) ---
    for i in 1:N
        for j in 1:N
            local_field = 0.0
            for k in 1:N
                local_field += B[i, j, k] * cos(θ[k] - θ[i])
            end
            driving  = (K1 * A[i, j] + K2 * local_field/N )
            du[i, j] = (-u[i, j] + driving) / τ
        end
    end
end

# ==============================================================================
# 2. NETWORK GENERATION (SYMMETRIC & ANTISYMMETRIC)
# ==============================================================================
function build_network(N::Int, p_edge::Float64, sym_type::Symbol, rng::AbstractRNG)
    A = zeros(Float64, N, N)
    B = zeros(Float64, N, N, N)

    # Pairwise adjacency A
    for i in 1:N, j in 1:N
        if i != j && rand(rng) < p_edge
            A[i, j] = 1.0
        end
    end

    # Higher-order tensor B 
    for i in 1:N, j in 1:N, k in (j+1):N
        (i == j || i == k) && continue
        
        if rand(rng) < p_edge
            if sym_type == :symmetric
                B[i, j, k] = 1.0
                B[i, k, j] = 1.0
            elseif sym_type == :antisymmetric
                B[i, j, k] = 1.0
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
# 4. PARAMETER SWEEP
# ==============================================================================
function run_sweep(N, p_edge, τ, K1_vals, K2_vals; seed=42)
    nK1 = length(K1_vals)
    nK2 = length(K2_vals)

    R1_sym  = zeros(Float64, nK1, nK2)
    R2_sym  = zeros(Float64, nK1, nK2)
    R1_anti = zeros(Float64, nK1, nK2)
    R2_anti = zeros(Float64, nK1, nK2)

    tspan = (0.0, 150.0)
    t_eq  = 75.0

    # Ensure same base topology and frequencies for fair comparison
    ω = randn(MersenneTwister(seed), N) .* 0.1
    A, B_sym  = build_network(N, p_edge, :symmetric, MersenneTwister(seed))
    _, B_anti = build_network(N, p_edge, :antisymmetric, MersenneTwister(seed))

    println("Sweeping K1×K2 grid ($(nK1)×$(nK2)) | N=$(N) | τ=$(τ)")

    # Initial condition (Consistent across all runs)
    θ0_base = range(0, 1, length=N) .* 2π
    y0 = vcat(θ0_base, zeros(N * N))

    Threads.@threads for i in 1:nK1
        K1 = K1_vals[i]
        for j in 1:nK2
            K2 = K2_vals[j]

            # Solve Symmetric Case
            p_sym = (ω, A, B_sym, K1, K2, τ, N)
            prob_sym = ODEProblem(dynamic_kuramoto!, y0, tspan, p_sym)
            sol_sym = solve(prob_sym, Tsit5(), saveat=0.5, reltol=1e-5, abstol=1e-5)

            # Solve Antisymmetric Case
            p_anti = (ω, A, B_anti, K1, K2, τ, N)
            prob_anti = ODEProblem(dynamic_kuramoto!, y0, tspan, p_anti)
            sol_anti = solve(prob_anti, Tsit5(), saveat=0.5, reltol=1e-5, abstol=1e-5)

            # Time-averaging
            eq_idx_sym  = findall(t -> t >= t_eq, sol_sym.t)
            eq_idx_anti = findall(t -> t >= t_eq, sol_anti.t)

            R1_sym[i, j] = mean([order_parameter(sol_sym.u[idx][1:N], 1) for idx in eq_idx_sym])
            R2_sym[i, j] = mean([order_parameter(sol_sym.u[idx][1:N], 2) for idx in eq_idx_sym])
            
            R1_anti[i, j] = mean([order_parameter(sol_anti.u[idx][1:N], 1) for idx in eq_idx_anti])
            R2_anti[i, j] = mean([order_parameter(sol_anti.u[idx][1:N], 2) for idx in eq_idx_anti])
        end
        println("  -> K1 progress: $(round(100 * i / nK1, digits=1))%")
    end

    return R1_sym, R2_sym, R1_anti, R2_anti
end

# ==============================================================================
# 5. PLOTTING (2x2 PANEL)
# ==============================================================================
function make_heatmap(K1_vals, K2_vals, Z, title_str, cbar_title; colormap=:viridis)
    heatmap(K1_vals, K2_vals, Z';
            color          = colormap,
            xlabel         = L"K_1",
            ylabel         = L"K_2",
            title          = title_str,
            colorbar_title = cbar_title,
            clims          = (0.0, 1.0),
            framestyle     = :box,
            xscale         = :log10,
            yscale         = :log10
            )
end

function make_smooth_surface(K1_vals, K2_vals, Z, title_str, cbar_title; colormap=:viridis, levels=30)
    contourf(K1_vals, K2_vals, Z';
             color          = colormap,
             xlabel         = L"K_1",
             ylabel         = L"K_2",
             title          = title_str,
             colorbar_title = cbar_title,
             clims          = (0.0, 1.0),
             framestyle     = :box,
             xscale         = :log10,
             yscale         = :log10,
             levels         = levels,
             legend         = false,
             alpha          = 0.1,
             )
end

function generate_four_panel_plot()
    mkpath(BASE_OUT_DIR)
    
    # Parameters
    N      = 50

    p_edge = 1.0
    τ      = 1.0 # Adiabatic limit to highlight HOI differences
    OUT_DIR = BASE_OUT_DIR*"/N$(N)_τ$(τ)"
    mkpath(OUT_DIR)
    
    K1_vals = 10.0 .^ range(-1.8, 1.8, length=32)
    K2_vals = 10.0 .^ range(-1.8, 1.8, length=32)

    R1_s, R2_s, R1_a, R2_a = run_sweep(N, p_edge, τ, K1_vals, K2_vals)

    # Construct individual panels
    p1 = make_heatmap(K1_vals, K2_vals, R1_s, L"\text{Symmetric } (B_{ijk}=B_{ikj})", L"R_1"; colormap=:viridis)
    p2 = make_heatmap(K1_vals, K2_vals, R2_s, L"\text{Symmetric } (B_{ijk}=B_{ikj})", L"R_2"; colormap=:viridis)
    
    p3 = make_heatmap(K1_vals, K2_vals, R1_a, L"\text{Antisymmetric } (B_{ijk}=-B_{ikj})", L"R_1"; colormap=:viridis)
    p4 = make_heatmap(K1_vals, K2_vals, R2_a, L"\text{Antisymmetric } (B_{ijk}=-B_{ikj})", L"R_2"; colormap=:viridis)

    # Combine into 2x2 Layout
    panel_plot = plot(p1, p2, p3, p4, 
                      layout = (2, 2), 
                      size = (1000, 850), 
                      left_margin = 5Plots.mm, 
                      bottom_margin = 5Plots.mm,
                      plot_title = L"\text{Dynamical Response: } \tau=%$τ, \text{ Network } N=%$N")

    out_file = joinpath(OUT_DIR, "four_panel_symmetry_N$(N).png")
    savefig(panel_plot, out_file)
    println("\n  [✓] Saved 2x2 Panel → $out_file")


    # Construct individual panels
    p1 = make_smooth_surface(K1_vals, K2_vals, R1_s, L"\text{Symmetric } (B_{ijk}=B_{ikj})", L"R_1"; colormap=:viridis)
    p2 = make_smooth_surface(K1_vals, K2_vals, R2_s, L"\text{Symmetric } (B_{ijk}=B_{ikj})", L"R_2"; colormap=:viridis)
    
    p3 = make_smooth_surface(K1_vals, K2_vals, R1_a, L"\text{Antisymmetric } (B_{ijk}=-B_{ikj})", L"R_1"; colormap=:viridis)
    p4 = make_smooth_surface(K1_vals, K2_vals, R2_a, L"\text{Antisymmetric } (B_{ijk}=-B_{ikj})", L"R_2"; colormap=:viridis)

    # Combine into 2x2 Layout
    panel_plot = plot(p1, p2, p3, p4, 
                      layout = (2, 2), 
                      size = (1000, 850), 
                      left_margin = 5Plots.mm, 
                      bottom_margin = 5Plots.mm,
                      plot_title = L"\text{Dynamical Response: } \tau=%$τ, \text{ Network } N=%$N")

    out_file = joinpath(OUT_DIR, "four_panel_symmetry_N$(N)_smooth.png")
    savefig(panel_plot, out_file)
    println("\n  [✓] Saved 2x2 Panel → $out_file")
end

# Execute
generate_four_panel_plot()