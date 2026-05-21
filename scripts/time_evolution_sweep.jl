using DifferentialEquations
using LinearAlgebra
using Statistics
using Plots
using LaTeXStrings

gr()
default(fontfamily = "Computer Modern", linewidth = 1.5, dpi=300, grid = false, framestyle = :box)

const N = 3
const ω0 = [0.9, 1.0, 1.1]
const K1_fixed = 0.1
const K2_fixed = 0.5
const A_FC = ones(Float64, N, N) - I

# 1. Dinâmica do Sistema
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

# 2. Configuração de Simulação
# Condições iniciais idênticas fixadas para expor o efeito exclusivo de tau
θ0 = [0.5, 2.5, 4.5]
u0 = [0.8, -0.5, 0.2, 0.1, -0.9, 0.6, -0.3, 0.4, -0.1]
y0 = vcat(θ0, u0)

B_asym = build_B(false) # Caso assimétrico (D ≠ 0)
tspan  = (0.0, 100.0)
taus   = [0.001, 0.1, 2.0] # Adiabático, Intermediário e Inercial

plot_list = []

for τ in taus
    p = (ω0, A_FC, B_asym, K1_fixed, K2_fixed, τ)
    prob = ODEProblem(dynamic_kuramoto!, y0, tspan, p)
    sol = solve(prob, Tsit5(); saveat=0.02, reltol=1e-7, abstol=1e-7)
    
    t = sol.t
    theta_mat = hcat([sol.u[idx][1:N] for idx in 1:length(t)]...)'
    u_mat     = hcat([sol.u[idx][N+1:end] for idx in 1:length(t)]...)'
    
    # Coluna 1: Evolução das fases (seno para visualização de sincronização bounded)
    p_θ = plot(t, sin.(theta_mat), label=[L"\theta_1" L"\theta_2" L"\theta_3"], 
               xlabel=L"t", ylabel=L"\sin(\theta_i)", 
               title=latexstring("\\tau = $τ"), legend=:bottomright)
    
    # Coluna 2: Evolução das variáveis de canal u_ij
    p_u = plot(t, u_mat[:, [2, 3, 6]], label=[L"u_{12}" L"u_{13}" L"u_{23}"], 
               xlabel=L"t", ylabel=L"u_{ij}", 
               title=latexstring("\\tau = $τ"), legend=:bottomright)
               
    push!(plot_list, p_θ, p_u)
end

# 3. Geração e salvamento do painel 3x2
out_dir = 
out_dir = joinpath("figures/kuramoto_u0_dependence", "K1_$(K1_fixed)_K2_$(K2_fixed)")
mkpath(out_dir)
canvas = plot(plot_list..., layout=(3, 2), size=(1000, 1100), margin=5Plots.mm)
savefig(canvas, joinpath(out_dir, "time_evolution_3x2.png"))

println("Concluído. Painel salvo em: ", joinpath(out_dir, "time_evolution_3x2.png"))