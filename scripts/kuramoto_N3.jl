using DifferentialEquations
using LinearAlgebra
using Statistics
using LaTeXStrings
using Plots

gr()
default(fontfamily = "Computer Modern", linewidth = 2, label = nothing, dpi=300,
        grid = false, framestyle = :box)

# ==============================================================================
# SETTINGS
# ==============================================================================

const BASE_OUT_DIR = "figures/kuramoto_N3"
const N            = 3          # fixed network size for this script

# ==============================================================================
# 1.  DYNAMICAL SYSTEM
# ==============================================================================
#
#   θ̇_i  =  ω_i  +  Σ_j u_{ij}(t)                                    [Eq. 1]
#
#   τ u̇_{ij}  =  −u_{ij}
#                + K₁ A_{ij} sin(θ_j − θ_i)
#                + K₂ [Σ_k B_{ijk} cos(θ_k − θ_i)] sin(θ_j − θ_i)   [Eq. 2]
#
# In the adiabatic limit (τ → 0), the quasi-steady u feeds back into the
# phase equation and generates the higher-order term:
#
#   Σ_HOI = (K₂/2) Σ_{j,k} [ C_{ijk} sin(θ_j+θ_k−2θ_i)    ← 3-body (standard)
#                             + D_{ijk} sin(θ_j−θ_k)  ]      ← pairwise correction
#
# where B_{ijk} = C_{ijk} + D_{ijk}  and  B_{ikj} = C_{ijk} − D_{ijk}
#   C_{ijk} = (B_{ijk}+B_{ikj})/2  (symmetric part)
#   D_{ijk} = (B_{ijk}−B_{ikj})/2  (antisymmetric part)
#
# D = 0  →  standard (1,1,−2) HOI fully recovered
# D ≠ 0  →  extra pairwise correction spoils the hyperedge reduction
# ==============================================================================

function dynamic_kuramoto!(dy, y, p, t)
    ω, A, B, K1, K2, τ = p

    θ  = @view y[1:N]
    u  = reshape(@view(y[N+1:end]), N, N)
    dθ = @view dy[1:N]
    du = reshape(@view(dy[N+1:end]), N, N)

    # [Eq. 1] phase dynamics
    for i in 1:N
        s = ω[i]
        for j in 1:N
            s += u[i, j]
        end
        dθ[i] = s
    end

    # [Eq. 2] transmission variable dynamics
    for i in 1:N, j in 1:N
        lf = 0.0
        for k in 1:N
            lf += B[i, j, k] * cos(θ[k] - θ[i])
        end
        du[i, j] = (-u[i, j] + (K1 * A[i, j] + K2 * lf) * sin(θ[j] - θ[i])) / τ
    end
end

# ==============================================================================
# 2.  NETWORK CONSTRUCTION  (N = 3, fully connected)
# ==============================================================================

# A: fully connected pairwise adjacency (off-diagonal = 1, diagonal = 0)
const A_FC = begin
    M = ones(Float64, N, N)
    for i in 1:N; M[i, i] = 0.0; end
    M
end

# B: explicit construction based on the C / D decomposition.
#
# For N = 3 there is exactly one unordered pair {j,k} per target i:
#   i=1 → {j,k} = {2,3},   i=2 → {j,k} = {1,3},   i=3 → {j,k} = {1,2}
#
# We keep the SYMMETRIC PART C constant across both cases so that the only
# difference between D=0 and D≠0 is the presence of the antisymmetric term:
#
#   D = 0  (sym_D = true)  :  B_{ijk} = B_{ikj} = C = 0.5
#                              → Σ_HOI = (K₂/2) Σ C_{ijk} sin(θ_j+θ_k−2θ_i)
#
#   D ≠ 0  (sym_D = false) :  B_{ijk} = 1.0,  B_{ikj} = 0.0
#                              → C = D = 0.5 (same C as above, D is added)
#                              → Σ_HOI = (K₂/2) Σ [ C_{ijk} sin(θ_j+θ_k−2θ_i)
#                                                   + D_{ijk} sin(θ_j−θ_k) ]
#
# Both cases have the same Frobenius norm of C; D adds a purely directed
# pairwise-like correction that has no analogue in static hyperedge models.
function build_B(sym_D::Bool)
    B = zeros(Float64, N, N, N)
    if sym_D                      # D = 0 : B_{ijk} = B_{ikj} = C = 0.5
        B[1,2,3] = B[1,3,2] = 0.5
        B[2,1,3] = B[2,3,1] = 0.5
        B[3,1,2] = B[3,2,1] = 0.5
    else                          # D ≠ 0 : B_{ijk}=1, B_{ikj}=0 → C=D=0.5
        B[1,2,3] = 1.0            # B[1,3,2] stays 0
        B[2,1,3] = 1.0            # B[2,3,1] stays 0
        B[3,1,2] = 1.0            # B[3,2,1] stays 0
    end
    return B
end

# Compute ||D||_F as a sanity check printed at runtime
function norm_D(B::Array{Float64,3})
    s = 0.0
    for i in 1:N, j in 1:N, k in 1:N
        s += ((B[i,j,k] - B[i,k,j]) / 2)^2
    end
    return sqrt(s)
end

# ==============================================================================
# 3.  ORDER PARAMETERS
# ==============================================================================
#
# Generalized Kuramoto order parameter:
#   R_q = |(1/N) Σ_j exp(i q θ_j)|
#
#   q = 1 → R₁: global synchrony          (R₁ → 1 iff all θ_j equal)
#   q = 2 → R₂: 2-cluster synchrony       (R₂ → 1 iff phases split into
#                                            two groups separated by π)
#
# For N = 3 the joint (R₁, R₂) pair identifies four regimes:
#   R₁ ≈ 1, R₂ ≈ 1    global sync
#   R₁ ≈ 1/3, R₂ ≈ 1  2+1 cluster  ({φ,φ,φ+π})
#   R₁ ≈ 0, R₂ ≈ 0    splay / incoherent
# --------------------------------------------------------------------------
function Rq(θ::AbstractVector{<:Real}, q::Int)
    return abs(sum(exp.(im * q .* θ)) / N)
end

# ==============================================================================
# 4.  PARAMETER SWEEP
# ==============================================================================

"""
    run_sweep(τ, sym_D, ω, K1_vals, K2_vals; kwargs...)

K₁ × K₂ grid sweep for a single (τ, sym_D) pair.
Returns `(R1_mat, R2_mat)`, each `(nK1, nK2)`.
"""
function run_sweep(τ::Float64, sym_D::Bool, ω::Vector{Float64},
                   K1_vals::AbstractVector, K2_vals::AbstractVector;
                   tspan     = (0.0, 200.0),
                   t_eq      = 100.0,
                   saveat_dt = 0.5)

    B     = build_B(sym_D)
    nK1   = length(K1_vals)
    nK2   = length(K2_vals)
    R1    = zeros(Float64, nK1, nK2)
    R2    = zeros(Float64, nK1, nK2)

    label = sym_D ? "D=0 " : "D≠0 "
    println("  [$label]  ‖D‖_F = $(round(norm_D(B), digits=4))")

    # Splay-state initial condition: θ₀ = [0, 2π/3, 4π/3]
    # Evenly spaced → neutral starting point (neither sync nor anti-sync biased)
    θ0 = [2π * (n - 1) / N for n in 1:N]
    y0 = vcat(θ0, zeros(Float64, N * N))

    for i in 1:nK1, j in 1:nK2
        K1 = K1_vals[i]
        K2 = K2_vals[j]

        p    = (ω, A_FC, B, K1, K2, τ)
        prob = ODEProblem(dynamic_kuramoto!, y0, tspan, p)
        sol  = solve(prob, Tsit5();
                     saveat   = saveat_dt,
                     reltol   = 1e-8,
                     abstol   = 1e-8,
                     maxiters = 2_000_000)

        eq_idx = findall(t -> t >= t_eq, sol.t)
        cnt    = length(eq_idx)
        s1 = s2 = 0.0
        for idx in eq_idx
            θt  = sol.u[idx][1:N]
            s1 += Rq(θt, 1)
            s2 += Rq(θt, 2)
        end
        R1[i, j] = cnt > 0 ? s1 / cnt : 0.0
        R2[i, j] = cnt > 0 ? s2 / cnt : 0.0
    end

    pct = round(100 * (nK1 * nK2) / (nK1 * nK2), digits = 0)
    println("     done ($(nK1*nK2) points).")
    return R1, R2
end

# ==============================================================================
# 5.  PLOTTING
# ==============================================================================

# Single heatmap
function make_heatmap(K1v, K2v, Z, title_str, cbar_str; cmap = :viridis)
    heatmap(K1v, K2v, Z';
            color          = cmap,
            xlabel         = L"K_1",
            ylabel         = L"K_2",
            title          = title_str,
            colorbar_title = cbar_str,
            clims          = (0.0, 1.0),
            xscale         = :log10,
            yscale         = :log10,
            framestyle     = :box,
            right_margin   = 8Plots.mm,
            bottom_margin  = 5Plots.mm,
            left_margin    = 5Plots.mm,
            size           = (480, 420))
end

# 2 × 2 comparison panel
#
#           │        R₁  (viridis)         R₂  (inferno)
#  ─────────┼──────────────────────────────────────────────
#   D = 0   │    h[1,1]                  h[1,2]
#   D ≠ 0   │    h[2,1]                  h[2,2]
#
function make_panel(K1v, K2v,
                    R1_sym, R2_sym, R1_asym, R2_asym,
                    τ_val)

    τ_str = latexstring("\\tau = $τ_val")

    h = [
        make_heatmap(K1v, K2v, R1_sym,  L"D=0,\quad R_1",          L"R_1"; cmap=:viridis),
        make_heatmap(K1v, K2v, R2_sym,  L"D=0,\quad R_2",          L"R_2"; cmap=:inferno),
        make_heatmap(K1v, K2v, R1_asym, L"D\neq 0,\quad R_1",      L"R_1"; cmap=:viridis),
        make_heatmap(K1v, K2v, R2_asym, L"D\neq 0,\quad R_2",      L"R_2"; cmap=:inferno),
    ]

    panel = plot(h...;
                 layout     = (2, 2),
                 size       = (1020, 840),
                 plot_title = τ_str)

    return h, panel
end

# Scatter R₂ vs R₁ with sym and asym overlaid
function make_scatter(R1_sym, R2_sym, R1_asym, R2_asym, τ_val)
    τ_str = latexstring("\\tau = $τ_val")
    p = scatter(vec(R1_sym), vec(R2_sym);
                label             = L"D = 0",
                color             = :steelblue,
                markersize        = 4,
                markeralpha       = 0.65,
                markerstrokewidth = 0,
                xlabel            = L"R_1",
                ylabel            = L"R_2",
                title             = τ_str,
                legend            = :topleft,
                xlims             = (0.0, 1.0),
                ylims             = (0.0, 1.0),
                size              = (500, 450))
    scatter!(p, vec(R1_asym), vec(R2_asym);
             label             = L"D \neq 0",
             color             = :crimson,
             markersize        = 4,
             markeralpha       = 0.65,
             markerstrokewidth = 0)
    return p
end

# ==============================================================================
# 6.  SAVE HELPER
# ==============================================================================
#
# Output tree:
#   BASE_OUT_DIR/
#   └── tau_{τ_str}/
#       ├── panel_2x2.{png,pdf}           ← main comparison figure
#       ├── scatter_R2_vs_R1.{png,pdf}
#       ├── heatmap_sym_R1.{png,pdf}      ← individual panels for reuse
#       ├── heatmap_sym_R2.{png,pdf}
#       ├── heatmap_asym_R1.{png,pdf}
#       └── heatmap_asym_R2.{png,pdf}
function save_all(τ_val, K1v, K2v,
                  R1_sym, R2_sym, R1_asym, R2_asym)

    τ_str   = replace(string(τ_val), "." => "p")    # e.g. 0.001 → "0p001"
    out_dir = joinpath(BASE_OUT_DIR, "tau_$τ_str")
    mkpath(out_dir)

    h_list, panel = make_panel(K1v, K2v, R1_sym, R2_sym, R1_asym, R2_asym, τ_val)
    p_scatter     = make_scatter(R1_sym, R2_sym, R1_asym, R2_asym, τ_val)

    figures = [
        ("panel_2x2",         panel),
        ("scatter_R2_vs_R1",  p_scatter),
        ("heatmap_sym_R1",    h_list[1]),
        ("heatmap_sym_R2",    h_list[2]),
        ("heatmap_asym_R1",   h_list[3]),
        ("heatmap_asym_R2",   h_list[4]),
    ]

    for (fname, fig) in figures, ext in ("png", "pdf")
        savefig(fig, joinpath(out_dir, "$fname.$ext"))
    end

    println("  [✓]  Saved → $out_dir")
end

# ==============================================================================
# 7.  MAIN EXECUTION
# ==============================================================================

# K₁ / K₂ log-spaced grid
K1_vals = 10.0 .^ range(-2, 0, length = 30)
K2_vals = 10.0 .^ range(-2, 0, length = 30)

# Fixed natural frequencies for N = 3:  zero mean removes uniform co-rotation
const ω0 = [0.08, -0.05, -0.03]

# τ regimes
#   τ ≪ 1  →  adiabatic limit  (emergent HOI, C and D both active)
#   τ ~ 1  →  intermediate inertia (memory / transient effects visible)

const TAU_REGIMES = [0.001, 0.05]

for τ in TAU_REGIMES
    println("\n" * "="^65)
    println("τ = $τ")
    println("="^65)

    R1_sym,  R2_sym  = run_sweep(τ, true,  ω0, K1_vals, K2_vals)
    R1_asym, R2_asym = run_sweep(τ, false, ω0, K1_vals, K2_vals)

    save_all(τ, K1_vals, K2_vals, R1_sym, R2_sym, R1_asym, R2_asym)
end

println("\n" * "="^65)
println("[Done]")
println("="^65)