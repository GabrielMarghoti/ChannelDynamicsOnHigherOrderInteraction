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

base_out_dir = "figures/kuramoto_cluster_sweep"

# ==============================================================================
# 1. DYNAMICAL SYSTEM
# ==============================================================================
# Full model with latent transmission variable u_{ij}(t):
#
#   θ̇_i = ω_i + Σ_j u_{ij}(t) 
#
#   τ u̇_{ij} = -u_{ij} + K₁ A_{ij} sin(θ_j - θ_i)
#              + K₂ [ Σ_k B_{ijk} cos(θ_k - θ_i) ] sin(θ_j - θ_i)
# ==============================================================================

function dynamic_kuramoto!(dy, y, p, t)
    ω, A, B, K1, K2, τ, N = p

    θ = @view y[1:N]
    u = reshape(@view(y[N+1:end]), N, N)

    dθ = @view dy[1:N]
    du = reshape(@view(dy[N+1:end]), N, N)

    # Phase dynamics
    for i in 1:N
        dθ[i] = ω[i]
        for j in 1:N
            dθ[i] += u[i, j]
        end
    end

    # Transmission variable dynamics
    for i in 1:N
        for j in 1:N
            local_field = 0.0
            for k in 1:N
                if B[i, j, k] != 0.0
                    local_field += B[i, j, k] * cos(θ[k] - θ[i])
                end
            end
            driving = K1 * A[i, j] * sin(θ[j] - θ[i]) + K2 * local_field * sin(θ[j] - θ[i])
            du[i, j] = (-u[i, j] + driving) / τ
        end
    end
end

# ==============================================================================
# 2. ORDER PARAMETERS
# ==============================================================================

function kuramoto_order(theta; qs=1)
    N = length(theta)

    if isa(qs, Int)
        q = qs
        return sum(exp.(im * q .* theta)) / N
    else
        z = Dict{Int, ComplexF64}()
        for q in qs
            z[q] = sum(exp.(im * q .* theta)) / N
        end
        return z
    end
end

# ==============================================================================
# 3. PARAMETER SWEEP
# ==============================================================================

function run_parameter_sweep(N, τ, symmetrize_B::Bool;
                              K1_vals, K2_vals,
                              tspan = (0.0, 150.0),
                              t_eq  = 75.0,
                              saveat_dt = 0.5,
                              seed = 42)

    Random.seed!(seed)

    # Natural frequencies ~ N(0, σ_ω²)
    ω = randn(N) .* 0.1

    # Erdős–Rényi topology for A and B
    p_edge = 1 # 0.3
    A = float.(rand(N, N) .< p_edge)
    B = float.(rand(N, N, N) .< p_edge)

    # Remove self-loops
    for i in 1:N
        A[i, i] = 0.0
        for j in 1:N
            B[i, j, j] = 0.0
            B[i, i, j] = 0.0
        end
    end

    # Optional: symmetrize B_{ijk} = B_{ikj} (adiabatic HOI limit)
    if symmetrize_B
        for i in 1:N, j in 1:N, k in 1:N
            B[i, k, j] = B[i, j, k]
        end
    end

    nK1 = length(K1_vals)
    nK2 = length(K2_vals)

    R_matrix  = zeros(nK1, nK2)   # Kuramoto order parameter
    Qc_matrix = zeros(nK1, nK2)   # Cluster order parameter

    println("  -> Grid: $(nK1) × $(nK2)  |  N = $N  |  τ = $τ  |  sym_B = $symmetrize_B")

    rngs = [MersenneTwister(seed + t) for t in 1:Threads.nthreads()]

    for i in 1:nK1
        K1 = K1_vals[i]
        rng = rngs[Threads.threadid()]

        for j in 1:nK2
            K2 = K2_vals[j]

            θ0 = rand(rng, N) .* 2π
            u0 = zeros(N * N)
            y0 = vcat(θ0, u0)

            p = (ω, A, B, K1, K2, τ, N)
            prob = ODEProblem(dynamic_kuramoto!, y0, tspan, p)
            sol  = solve(prob, Tsit5(), saveat = saveat_dt,
                         reltol = 1e-6, abstol = 1e-6, maxiters = 1_000_000)

            eq_idx = findall(t -> t >= t_eq, sol.t)

            R_sum  = 0.0
            Qc_sum = 0.0
            cnt    = length(eq_idx)

            for idx in eq_idx
                θ_t      = sol.u[idx][1:N]
                zs = kuramoto_order(θ_t, qs=[1,2])
                R_sum   += abs(zs[1])
                Qc_sum  +=  abs(zs[2])
            end

            R_matrix[i, j]  = cnt > 0 ? R_sum  / cnt : 0.0
            Qc_matrix[i, j] = cnt > 0 ? Qc_sum / cnt : 0.0
        end

        pct = round(100 * i / nK1, digits = 1)
        println("  -> K1 = $(round(K1, sigdigits=3))  [$pct%]")
    end

    println("  -> Sweep complete.")
    return R_matrix, Qc_matrix
end

# ==============================================================================
# 4. PLOTTING
# ==============================================================================

function make_heatmap(K1_vals, K2_vals, Z, title_str, cbar_title;
                      colormap = :viridis, clims = (0.0, 1.0))
    heatmap(K1_vals, K2_vals, Z';
            color          = colormap,
            xlabel         = L"K_1",
            ylabel         = L"K_2",
            title          = title_str,
            colorbar_title = cbar_title,
            clims          = clims,
            framestyle     = :box,
            size           = (520, 450),
            xscale         = :log10,
            yscale         = :log10,
            right_margin   = 7Plots.mm,
            bottom_margin  = 5Plots.mm)
end

# Scatter R_2 vs R to reveal the dynamical regimes ----------------------------
function make_scatter(R_mat, Qc_mat, title_str)
    R_vec  = vec(R_mat)
    Qc_vec = vec(Qc_mat)
    scatter(R_vec, Qc_vec;
            xlabel      = L"R_1 \; \mathrm{(global \; order \; parameter)}",
            ylabel      = L"R_2 \; \mathrm{(cluster \; order \; parameter)}",
            title       = title_str,
            markersize  = 4,
            markeralpha = 0.7,
            markerstrokewidth = 0,
            color       = :steelblue,
            xlims       = (0, 1),
            ylims       = (0, 1),
            size        = (500, 450))
end

# ==============================================================================
# 5. MAIN EXECUTION
# ==============================================================================

# Grid resolution – increase length for publication-quality figures
K1_vals = 10.0 .^ range(-1, 0, length = 11)
K2_vals = 10.0 .^ range(-1, 0, length = 11)

N = 10  # Number of oscillators

# Configurations: (label, τ, symmetrize_B)
# τ ≪ 1 → adiabatic 
configs = [
    ("adiabatic_asymB",  0.001, false),   # τ → 0, general B_{ijk} ≠ B_{ikj}
    ("adiabatic_symB",   0.001, true),    # τ → 0, B_{ijk} = B_{ikj} (HOI limit)
    ("intermediate",     0.05,  false),   # intermediate τ
]

for (label, τ_val, sym_B) in configs
    println("\n" * "="^65)
    println("Config: $label   (τ = $τ_val, symmetric_B = $sym_B)")
    println("="^65)

    out_dir = joinpath(base_out_dir, label)
    mkpath(out_dir)

    R_mat, Qc_mat = run_parameter_sweep(N, τ_val, sym_B;
                                         K1_vals = K1_vals,
                                         K2_vals = K2_vals)

    τ_str  = L"\tau = %$(τ_val)"
    sym_str = sym_B ? L",\; B_{ijk}=B_{ikj}" : L",\; B_{ijk}\neq B_{ikj}"

    # ── Heatmap: Kuramoto R ────────────────────────────────────────────────
    p_R = make_heatmap(K1_vals, K2_vals, R_mat,
                       τ_str * sym_str,
                       L"R_1",
                       colormap = :viridis)

    # ── Heatmap: Cluster order parameter R_2 ──────────────────────────────
    p_R2 = make_heatmap(K1_vals, K2_vals, Qc_mat,
                        τ_str * sym_str,
                        L"R_2",
                        colormap = :inferno)

    # ── Side-by-side panel ────────────────────────────────────────────────
    p_panel = plot(p_R, p_R2;
                   layout = (1, 2),
                   size   = (1050, 450),
                   plot_title = "Parameter space  ($label)")

    # ── Scatter: R_2 vs R (reveals dynamical regimes) ─────────────────────
    p_scatter = make_scatter(R_mat, Qc_mat, τ_str * sym_str)

    # Save all figures
    for (fname, fig) in [
            ("heatmap_R",      p_R),
            ("heatmap_R2",     p_R2),
            ("panel_R_Qc",     p_panel),
            ("scatter_Qc_R",   p_scatter),
        ]
        for ext in ["png", "pdf"]
            path = joinpath(out_dir, "$(fname).$(ext)")
            savefig(fig, path)
        end
    end

    println("\n[✓] Figures saved to: $out_dir")
end

println("\n[Done] All sweeps completed.")