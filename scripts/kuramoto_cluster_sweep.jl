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

const BASE_OUT_DIR = "figures/kuramoto_cluster_sweep"

# ==============================================================================
# 1. DYNAMICAL SYSTEM
# ==============================================================================
#
#   θ̇_i  =  ω_i  +  Σ_j u_{ij}(t)                               [Eq. 1]
#
#   τ u̇_{ij}  =  −u_{ij}
#                + K₁ A_{ij} sin(θ_j − θ_i)
#                + K₂ [Σ_k B_{ijk} cos(θ_k − θ_i)] sin(θ_j − θ_i)   [Eq. 2]
#
# Note: u_{ij} already carries the coupling kernel, so the phase equation
# sums u directly (no extra sin factor at the θ level).
# ==============================================================================

function dynamic_kuramoto!(dy, y, p, t)
    ω, A, B, K1, K2, τ, N = p

    θ = @view y[1:N]
    u = reshape(@view(y[N+1:end]), N, N)
    dθ = @view dy[1:N]
    du = reshape(@view(dy[N+1:end]), N, N)

    # --- Phase dynamics [Eq. 1] -----------------------------------------------
    for i in 1:N
        dθ[i] = ω[i]
        for j in 1:N
            dθ[i] += u[i, j] /N        
        end
    end

    # --- Transmission variable dynamics [Eq. 2] --------------------------------
    for i in 1:N
        for j in 1:N
            local_field = 0.0
            for k in 1:N
                local_field += B[i, j, k] * cos(θ[k] - θ[i])
            end
            driving  = (K1 * A[i, j] + K2 * local_field/N) * sin(θ[j] - θ[i])
            du[i, j] = (-u[i, j] + driving) / τ
        end
    end
end

# ==============================================================================
# 2. NETWORK GENERATION
# ==============================================================================

function build_network(N::Int, p_edge::Float64, symmetrize_B::Bool, rng::AbstractRNG)

    # --- Pairwise adjacency A (Erdős–Rényi, no self-loops) --------------------
    A = zeros(Float64, N, N)
    for i in 1:N, j in 1:N
        i != j && rand(rng) < p_edge && (A[i, j] = 1.0)
    end

    # --- Higher-order tensor B (Erdős–Rényi, no degenerate indices) -----------
    # B[i,j,k] weights how oscillator k modulates the j→i channel.
    # Forbidden: j==i, k==i, j==k (self-modulation or repeated index).
    B = zeros(Float64, N, N, N)
    for i in 1:N, j in 1:N, k in 1:N
        (i == j || i == k || j == k) && continue
        rand(rng) < p_edge && (B[i, j, k] = 1.0)
    end

    # --- Optional symmetrization B_{ijk} = B_{ikj} (enables HOI reduction) ---
    # Bug-safe: iterate only over j < k (upper triangle) so each pair {j,k}
    # is visited exactly once. Iterating all (j,k) would cause each assignment
    # to be immediately overwritten by the symmetric pass.
    if symmetrize_B
        for i in 1:N, j in 1:N, k in (j+1):N
            sym_val     = (B[i, j, k] + B[i, k, j]) / 2.0
            B[i, j, k]  = sym_val
            B[i, k, j]  = sym_val
        end
    end

    return A, B
end

# ==============================================================================
# 3. ORDER PARAMETERS
# ==============================================================================

# Generalized Kuramoto order parameter R_q:
#
#   z_q = (1/N) Σ_j exp(i·q·θ_j),    R_q = |z_q|
#
#   q = 1  →  R₁: global synchrony
#             → 1 when all phases coincide
#             → 0 when phases are uniformly distributed
#
#   q = 2  →  R₂: 2-cluster synchrony (π clusters)
#             → 1 when oscillators form two groups separated by π
#             → 0 when phases are uniformly distributed
#
# Interpretation of the joint (R₁, R₂) pair:
#   R₁ ≈ 1, R₂ ≈ 1  →  global synchrony (both modes agree)
#   R₁ ≈ 0, R₂ ≈ 1  →  pure 2-cluster state
#   R₁ ≈ 0, R₂ ≈ 0  →  incoherent
# --------------------------------------------------------------------------
function order_parameter(θ::AbstractVector{<:Real}, q)
    return abs(sum(exp.(im * q .* θ)) / length(θ))
end

# ==============================================================================
# 4. PARAMETER SWEEP
# ==============================================================================

"""
    run_parameter_sweep(cfg; K1_vals, K2_vals, kwargs...)

Sweep K₁ × K₂ for a single configuration tuple
    cfg = (label, τ, sym_B, N, p_edge)
Returns `(R1_matrix, R2_matrix)`, each of size `(nK1, nK2)`.
"""
function run_parameter_sweep(cfg;
                              K1_vals,
                              K2_vals,
                              tspan     = (0.0, 150.0),
                              t_eq      = 75.0,
                              saveat_dt = 0.5,
                              seed      = 42)

    label, τ, sym_B, N, p_edge = cfg

    rng = MersenneTwister(seed)

    # Fixed network for this configuration (shared across all K1, K2 points)
    ω    = randn(rng, N) .* 0.1
    A, B = build_network(N, p_edge, sym_B, rng)

    nK1 = length(K1_vals)
    nK2 = length(K2_vals)

    R1_matrix = zeros(Float64, nK1, nK2)
    R2_matrix = zeros(Float64, nK1, nK2)
    R4_matrix = zeros(Float64, nK1, nK2)

    println("  -> Grid $(nK1)×$(nK2) | N=$N | p=$p_edge | τ=$τ | sym_B=$sym_B")

    for i in 1:nK1
        K1 = K1_vals[i]

        for j in 1:nK2
            K2 = K2_vals[j]

            # Fresh random initial phases; u starts at rest
            θ0 = range(0,1,length=N) .* 2π # rand(rng, N) .* 2π
            y0 = vcat(θ0, zeros(N * N))

            p    = (ω, A, B, K1, K2, τ, N)
            prob = ODEProblem(dynamic_kuramoto!, y0, tspan, p)
            sol  = solve(prob, Tsit5(),
                         saveat   = saveat_dt,
                         reltol   = 1e-6,
                         abstol   = 1e-6,
                         maxiters = 1_000_000)

            # Time-average over the post-transient window
            eq_idx = findall(t -> t >= t_eq, sol.t)
            cnt    = length(eq_idx)

            R1_sum = 0.0
            R2_sum = 0.0
            R4_sum = 0.0
            for idx in eq_idx
                θ_t    = sol.u[idx][1:N]
                R1_sum += order_parameter(θ_t, 1)
                R2_sum += order_parameter(θ_t, 2)
                R4_sum += order_parameter(θ_t, 4)
            end

            R1_matrix[i, j] = cnt > 0 ? R1_sum / cnt : 0.0
            R2_matrix[i, j] = cnt > 0 ? R2_sum / cnt : 0.0
            R4_matrix[i, j] = cnt > 0 ? R4_sum / cnt : 0.0

        end

        pct = round(100 * i / nK1, digits = 1)
        println("  -> K1 = $(round(K1, sigdigits=3))  [$pct %]")
    end

    println("  -> Sweep complete.")
    return R1_matrix, R2_matrix, R4_matrix
end

# ==============================================================================
# 5. PLOTTING
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

function make_scatter(R1_mat, R2_mat, title_str)
    scatter(vec(R1_mat), vec(R2_mat);
            xlabel            = L"R_1 \;\mathrm{(global\;synchrony)}",
            ylabel            = L"R_2 \;\mathrm{(2\text{-}cluster\;synchrony)}",
            title             = title_str,
            markersize        = 4,
            markeralpha       = 0.7,
            markerstrokewidth = 0,
            color             = :steelblue,
            xlims             = (0.0, 1.0),
            ylims             = (0.0, 1.0),
            size              = (500, 450))
end

# ==============================================================================
# 6. SAVE HELPER
# ==============================================================================

"""
    save_results(cfg, K1_vals, K2_vals, R1_mat, R2_mat)

Output directory tree:

    BASE_OUT_DIR/
    └── N{N}_p{p_edge}/
        └── {label}/
            ├── heatmap_R1.{png,pdf}
            ├── heatmap_R2.{png,pdf}
            ├── panel_R1_R2.{png,pdf}
            └── scatter_R2_vs_R1.{png,pdf}

Network parameters (N, p_edge) form the parent folder so that sweeps with
different network sizes are kept separate from those varying τ or B symmetry.
"""
function save_results(cfg, K1_vals, K2_vals, R1_mat, R2_mat, R4_mat)
    label, τ, sym_B, N, p_edge = cfg

    # e.g.  N20_p0p30
    p_str   = replace(string(round(p_edge, digits = 2)), "." => "p")
    net_dir = "N$(N)_p$(p_str)"
    out_dir = joinpath(BASE_OUT_DIR, net_dir, label)
    mkpath(out_dir)

    # LaTeX title string
    τ_str   = L"\tau = %$(τ)"
    sym_str = sym_B ? L",\; B_{ijk}=B_{ikj}" : L",\; B_{ijk}\neq B_{ikj}"
    full_title = τ_str * sym_str

    p_R1 = make_heatmap(K1_vals, K2_vals, R1_mat,
                        full_title, L"R_1"; colormap = :viridis)


    p_R2 = make_heatmap(K1_vals, K2_vals, R2_mat,
                        full_title, L"R_2"; colormap = :inferno)

    p_R4 = make_heatmap(K1_vals, K2_vals, R4_mat,
                         full_title, L"R_{4}"; colormap = :plasma)
    p_panel = plot(p_R1, p_R2, p_R4;
                   layout     = (1, 3),
                   size       = (1250, 450),
                   plot_title = "$(label)  |  N=$(N)  p=$(p_edge)")

    p_scatter = make_scatter(R1_mat, R2_mat, full_title)
    p_scatter = make_scatter(R1_mat, R4_mat, full_title * L"  |  R_1 \mathrm{\ vs.\ } R_4")

    for (fname, fig) in [
            ("heatmap_R1",       p_R1), 
            ("heatmap_R2",       p_R2),
            ("heatmap_R4",      p_R4),   
            ("panel_R1_R2",      p_panel),
            ("scatter_R2_vs_R1", p_scatter),
        ]
        for ext in ("png", "pdf")
            savefig(fig, joinpath(out_dir, "$(fname).$(ext)"))
        end
    end

    println("\n  [✓] Saved → $out_dir")
end

# ==============================================================================
# 7. CONFIGURATIONS AND MAIN EXECUTION
# ==============================================================================

# K₁ / K₂ sweep grid (log-spaced)
K1_vals = 10.0 .^ range(-2, 0, length = 10)
K2_vals = 10.0 .^ range(-2, 0, length = 10)

# Configuration tuples:  (label, τ, symmetrize_B, N, p_edge)
#
#   label        – descriptive string; becomes the leaf folder name
#   τ            – transmission timescale
#                    τ ≪ 1  →  adiabatic / HOI limit
#                    τ ~ 1  →  intermediate inertia
#                    τ ≫ 1  →  pairwise (standard Kuramoto) limit
#   symmetrize_B – true enforces B_{ijk} = B_{ikj}, enabling the exact
#                  (1,1,−2) HOI reduction; false is the general physical case
#   N            – number of oscillators
#   p_edge       – Erdős–Rényi edge probability for both A and B
#
configs = [
    # ── Adiabatic regime ──────────────────────────────────────────────────────
    ("adiabatic_asymB",  0.001, false, 10, 1.0),
    ("adiabatic_symB",   0.001, true,  10, 1.0),
    # ── Intermediate inertia ──────────────────────────────────────────────────
    ("intermediate",     0.05,  false, 10, 1.0),
]

for cfg in configs
    label, τ_val, sym_B, N, p_edge = cfg

    println("\n" * "="^65)
    println("Config : $label")
    println("Params : τ=$τ_val  |  sym_B=$sym_B  |  N=$N  |  p=$p_edge")
    println("="^65)

    R1_mat, R2_mat, R4_mat = run_parameter_sweep(cfg;
                                          K1_vals = K1_vals,
                                          K2_vals = K2_vals)

    save_results(cfg, K1_vals, K2_vals, R1_mat, R2_mat, R4_mat)
end

println("\n" * "="^65)
println("[Done]  All sweeps completed.")
println("="^65)