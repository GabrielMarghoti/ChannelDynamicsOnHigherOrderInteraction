using DifferentialEquations
using LinearAlgebra
using Statistics
using LaTeXStrings
using Plots

gr()
default(fontfamily = "Computer Modern", linewidth = 2, dpi=300, grid = false, framestyle = :box)

const BASE_OUT_DIR = "figures/kuramoto_u0_dependence"
const N = 3

# ==============================================================================
# 1. SISTEMA DINÂMICO E REDE
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
        B[1,2,3] = 1.0; B[2,1,3] = 1.0; B[3,1,2] = 1.0            
    end
    return B
end

function Rq(θ::AbstractVector{<:Real}, q::Int)
    return abs(sum(exp.(im * q .* θ)) / N)
end

# ==============================================================================
# 2. VARREDURA DE CONDIÇÕES INICIAIS u(0) E FASE θ(0)
# ==============================================================================

function run_u0_sweep(τ_vals::AbstractVector, sym_D::Bool, ω::Vector{Float64}, 
                      K1::Float64, K2::Float64, θ0::Vector{Float64}, num_trials::Int;
                      tspan = (0.0, 300.0), t_eq = 200.0)
    
    B = build_B(sym_D)
    n_tau = length(τ_vals)
    R1_results = zeros(Float64, num_trials, n_tau)

    for (j, τ) in enumerate(τ_vals)
        p = (ω, A_FC, B, K1, K2, τ)
        for i in 1:num_trials
            u0 = 2.0 .* rand(Float64, N * N) .- 1.0
            y0 = vcat(θ0, u0)

            prob = ODEProblem(dynamic_kuramoto!, y0, tspan, p)
            sol  = solve(prob, Tsit5(); saveat=1.0, reltol=1e-6, abstol=1e-6)

            eq_idx = findall(t -> t >= t_eq, sol.t)
            cnt = length(eq_idx)
            s1 = 0.0
            for idx in eq_idx
                s1 += Rq(sol.u[idx][1:N], 1)
            end
            R1_results[i, j] = cnt > 0 ? s1 / cnt : 0.0
        end
    end
    
    return vec(mean(R1_results, dims=1))
end

# ==============================================================================
# 3. PLOTAGEM E EXECUÇÃO
# ==============================================================================

tau_vals = 10.0 .^ range(-3, 1, length=20)
num_trials = 50

const ω0 = [0.08, -0.05, -0.03]
const K1_fixed = 0.3
const K2_fixed = 0.7

# Condições Iniciais de Fase
θ0_sync  = [0.0, 0.0, 0.0]
θ0_splay = [0.0, 2π/3, 4π/3]

println("Simulando D=0...")
R1_mean_sym_sync  = run_u0_sweep(tau_vals, true, ω0, K1_fixed, K2_fixed, θ0_sync, num_trials)
R1_mean_sym_splay = run_u0_sweep(tau_vals, true, ω0, K1_fixed, K2_fixed, θ0_splay, num_trials)

println("Simulando D≠0...")
R1_mean_asym_sync  = run_u0_sweep(tau_vals, false, ω0, K1_fixed, K2_fixed, θ0_sync, num_trials)
R1_mean_asym_splay = run_u0_sweep(tau_vals, false, ω0, K1_fixed, K2_fixed, θ0_splay, num_trials)

# Preparação de Pastas baseadas nos parâmetros
out_dir = joinpath(BASE_OUT_DIR, "K1_$(K1_fixed)_K2_$(K2_fixed)")
mkpath(out_dir)

function plot_means(tau_vals, R_sync, R_splay, title_str)
    p = plot(xscale=:log10, xlabel=L"\tau", ylabel=L"\langle R_1 \rangle", title=title_str, ylims=(0, 1.05))
    plot!(p, tau_vals, R_sync,  label=L"\theta(0) \text{ síncrono}", color=:blue,  marker=:circle)
    plot!(p, tau_vals, R_splay, label=L"\theta(0) \text{ splay}",    color=:red,   marker=:square)
    return p
end

p1 = plot_means(tau_vals, R1_mean_sym_sync, R1_mean_sym_splay, L"D=0")
p2 = plot_means(tau_vals, R1_mean_asym_sync, R1_mean_asym_splay, L"D \neq 0")

panel = plot(p1, p2, layout=(1,2), size=(1000, 450), bottom_margin=5Plots.mm, left_margin=5Plots.mm)

savefig(panel, joinpath(out_dir, "u0_dependence_mean_R1.pdf"))
savefig(panel, joinpath(out_dir, "u0_dependence_mean_R1.png"))

println("Finalizado. Gráficos salvos em: $out_dir")