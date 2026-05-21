using DifferentialEquations
using LinearAlgebra
using Statistics
using LaTeXStrings
using Plots

gr()
default(fontfamily = "Computer Modern", linewidth = 2, dpi=300, grid = false, framestyle = :box)

const BASE_OUT_DIR = "figures/kuramoto_u0_dependence"
const N = 3

# Threshold on R1 above which a trial is considered synchronized
const SYNC_THRESHOLD = 0.9

# ==============================================================================
# 1. DYNAMICAL SYSTEM AND NETWORK
# ==============================================================================

function dynamic_kuramoto!(dy, y, p, t)
    ω, A, B, K1, K2, τ = p
    θ  = @view y[1:N]
    u  = reshape(@view(y[N+1:end]), N, N)
    dθ = @view dy[1:N]
    du = reshape(@view(dy[N+1:end]), N, N)

    for i in 1:N
        s = ω[i]
        for j in 1:N; s += u[i, j]; end
        dθ[i] = s
    end

    for i in 1:N, j in 1:N
        lf = 0.0
        for k in 1:N; lf += B[i, j, k] * cos(θ[k] - θ[i]); end
        du[i, j] = (-u[i, j] + (K1 * A[i, j] + K2 * lf) * sin(θ[j] - θ[i])) / τ
    end
end

const A_FC = ones(Float64, N, N) - I

function build_B(sym_D::Bool)
    B = zeros(Float64, N, N, N)
    if sym_D
        B[1,2,3] = B[1,3,2] = 0.5
        B[2,1,3] = B[2,3,1] = 0.5
        B[3,1,2] = B[3,2,1] = 0.5
    else
        B[1,2,3] =  0.5
        B[1,3,2] = -0.5
        B[2,1,3] =  0.5
        B[2,3,1] = -0.5
        B[3,1,2] =  0.5
        B[3,2,1] = -0.5
    end
    return B
end

function Rq(θ::AbstractVector{<:Real}, q::Int)
    return abs(sum(exp.(im * q .* θ)) / N)
end

# ==============================================================================
# 2. SWEEP OVER RANDOM INITIAL CONDITIONS u(0) AND θ(0)
# ==============================================================================

"""
    run_ic_sweep(τ_vals, sym_D, ω, K1, K2, num_trials; tspan, t_eq)

For each value of τ, runs `num_trials` simulations with:
  - θ(0) sampled uniformly in (0, 2π)  (N independent draws per trial)
  - u(0) sampled uniformly in (-1, 1)  (N×N independent draws per trial)

Returns a vector of length `n_tau` with the fraction of trials that reached
an asymptotically synchronized state, defined as time-averaged R₁ ≥ SYNC_THRESHOLD
over the interval [t_eq, tspan[2]].
"""
function run_ic_sweep(τ_vals::AbstractVector, sym_D::Bool, ω::Vector{Float64},
                      K1::Float64, K2::Float64, num_trials::Int;
                      tspan = (0.0, 300.0), t_eq = 200.0)

    B     = build_B(sym_D)
    n_tau = length(τ_vals)
    synced = zeros(Int, num_trials, n_tau)   # 1 if trial i synchronized at τ j

    for (j, τ) in enumerate(τ_vals)
        p = (ω, A_FC, B, K1, K2, τ)
        for i in 1:num_trials
            # Random initial conditions: θ ∈ (0, 2π),  u ∈ (-1, 1)
            θ0 = 2π .* rand(Float64, N)
            u0 = 2.0 .* rand(Float64, N * N) .- 1.0
            y0 = vcat(θ0, u0)

            prob = ODEProblem(dynamic_kuramoto!, y0, tspan, p)
            sol  = solve(prob, Tsit5(); saveat=1.0, reltol=1e-6, abstol=1e-6)

            # Compute time-averaged R₁ over the equilibration window
            eq_idx = findall(t -> t >= t_eq, sol.t)
            cnt    = length(eq_idx)
            R1_mean = cnt > 0 ? mean(Rq(sol.u[idx][1:N], 1) for idx in eq_idx) : 0.0

            synced[i, j] = R1_mean >= SYNC_THRESHOLD ? 1 : 0
        end
    end

    # Fraction of trials that synchronized for each τ
    return vec(mean(synced, dims=1))
end

# ==============================================================================
# 3. PLOTTING AND EXECUTION
# ==============================================================================

tau_vals   = 10.0 .^ range(-3, 1, length=20)
num_trials = 100

const ω0       = [-0.1, 0, 0.1]
const K1_fixed = 0.1
const K2_fixed = 0.1

println("Simulating D=0  (symmetric D)...")
frac_sync_sym  = run_ic_sweep(tau_vals, true,  ω0, K1_fixed, K2_fixed, num_trials)

println("Simulating D≠0  (asymmetric D)...")
frac_sync_asym = run_ic_sweep(tau_vals, false, ω0, K1_fixed, K2_fixed, num_trials)

# Output directory
out_dir = joinpath(BASE_OUT_DIR, "K1_$(K1_fixed)_K2_$(K2_fixed)")
mkpath(out_dir)

function plot_sync_fraction(tau_vals, frac_sym, frac_asym)
    p = plot(
        xscale  = :log10,
        xlabel  = L"\tau",
        ylabel  = L"f_{\mathrm{sync}}",
        ylims   = (0, 1.05),
        legend  = :topright,
    )
    plot!(p, tau_vals, frac_sym,  label=L" C \neq 0, D = 0",    color=:blue,  marker=:circle)
    plot!(p, tau_vals, frac_asym, label=L"C = 0, D \neq 0", color=:red,   marker=:square)
    return p
end

panel = plot_sync_fraction(tau_vals, frac_sync_sym, frac_sync_asym)
panel = plot(panel; size=(700, 480), bottom_margin=5Plots.mm, left_margin=5Plots.mm)

savefig(panel, joinpath(out_dir, "ic_sync_fraction_vs_tau.pdf"))
savefig(panel, joinpath(out_dir, "ic_sync_fraction_vs_tau.png"))

println("Done. Figures saved to: $out_dir")