using DifferentialEquations
using LinearAlgebra
using Statistics
using Random
using StatsPlots
using LaTeXStrings

gr(fontfamily="Computer Modern", linewidth=2, grid=false, framestyle=:box, dpi=300)

# 1. Sistema Dinâmico Normalizado (Invariância de Escala)
function dynamic_kuramoto_normalized!(dy, y, p, t)
    ω, A, B, K1, K2, τ, N = p
    θ  = @view y[1:N]
    u  = reshape(@view(y[N+1:end]), N, N)
    dθ = @view dy[1:N]
    du = reshape(@view(dy[N+1:end]), N, N)

    for i in 1:N
        dθ[i] = ω[i]
        for j in 1:N
            dθ[i] += u[i, j] * sin(θ[j] - θ[i]) / N        
        end
    end

    for i in 1:N
        for j in 1:N
            local_field = 0.0
            for k in 1:N
                local_field += B[i, j, k] * cos(θ[k] - θ[i])
            end
            du[i, j] = (-u[i, j] + K1 * A[i, j] + K2 * local_field / N) / τ
        end
    end
end

# 2. Construtor de Rede (Simétrico e Antissimétrico)
function build_network(N, sym_type)
    A = ones(Float64, N, N) - I
    B = zeros(Float64, N, N, N)
    for i in 1:N, j in 1:N, k in (j+1):N
        if i != j && i != k
            B[i, j, k] = 1.0
            if sym_type == :symmetric
                B[i, k, j] = 1.0
            elseif sym_type == :antisymmetric
                B[i, k, j] = -1.0
            end
        end
    end
    return A, B
end

function order_parameter(θ, q)
    return abs(mean(exp.(im .* q .* θ)))
end

# 3. Varredura Estocástica e Simulação
function run_panel_sweep()
    N_vals = [10, 20, 30]#, 50, 100]
    num_trials = 1000
    
    K1_fixed = 0.2
    K2_fixed = 20.0
    τ = 10 # Regime de inércia alta
    
    # Tempos estendidos para garantir relaxação com tau=10
    tspan = (0.0, 300.0)
    t_eq = 200.0

    R1_sym = zeros(length(N_vals), num_trials)
    R2_sym = zeros(length(N_vals), num_trials)
    R1_anti = zeros(length(N_vals), num_trials)
    R2_anti = zeros(length(N_vals), num_trials)

    for (n_idx, N) in enumerate(N_vals)
        println("Simulando N = $N...")
        A_sym, B_sym = build_network(N, :symmetric)
        A_anti, B_anti = build_network(N, :antisymmetric)
        
        Threads.@threads for trial in 1:num_trials

            ω = randn(N) .* 0.1
            
            θ0 = rand(N) .* 2π
            u0 = zeros(N * N)
            y0 = vcat(θ0, u0)
            
            # Simétrico
            p_sym = (ω, A_sym, B_sym, K1_fixed, K2_fixed, τ, N)
            sol_sym = solve(ODEProblem(dynamic_kuramoto_normalized!, y0, tspan, p_sym), Tsit5(), saveat=0.5, reltol=1e-5, abstol=1e-5)
            
            # Antissimétrico
            p_anti = (ω, A_anti, B_anti, K1_fixed, K2_fixed, τ, N)
            sol_anti = solve(ODEProblem(dynamic_kuramoto_normalized!, y0, tspan, p_anti), Tsit5(), saveat=0.5, reltol=1e-5, abstol=1e-5)
            
            # Extração
            eq_idx_sym = findall(t -> t >= t_eq, sol_sym.t)
            eq_idx_anti = findall(t -> t >= t_eq, sol_anti.t)
            
            R1_sym[n_idx, trial] = mean([order_parameter(sol_sym.u[idx][1:N], 1) for idx in eq_idx_sym])
            R2_sym[n_idx, trial] = mean([order_parameter(sol_sym.u[idx][1:N], 2) for idx in eq_idx_sym])
            
            R1_anti[n_idx, trial] = mean([order_parameter(sol_anti.u[idx][1:N], 1) for idx in eq_idx_anti])
            R2_anti[n_idx, trial] = mean([order_parameter(sol_anti.u[idx][1:N], 2) for idx in eq_idx_anti])
        end
    end

    # 4. Plotagem (Painel Duplo)
    x_indices = 1:length(N_vals)
    x_ticks = (x_indices, string.(N_vals))
    
    plt = plot(layout=(2, 1), size=(450, 400), link=:x, left_margin=2Plots.mm)

    plot!(plt[1], ylabel=L"R_q", ylims=(-0.05, 1.05), xticks=x_ticks, bottom_margin=-2Plots.mm, legend=:topright)
    plot!(plt[2], ylabel=L"R_q", xlabel=L"N", ylims=(-0.05, 1.05), xticks=x_ticks, bottom_margin=-2Plots.mm)

    annotate!(plt[1], 0.7, 0.95, text("(a)", :left, font("Computer Modern", 10)))
    annotate!(plt[2], 0.7, 0.95, text("(b)", :left, font("Computer Modern", 10)))

    for n_idx in x_indices
        x_rep = fill(n_idx, num_trials)
        
        # Painel Superior (Simétrico)
        violin!(plt[1], x_rep, R1_sym[n_idx, :], side=:left, color=:steelblue, alpha=0.8, width=0.4, linewidth=0, label=(n_idx==1 ? L"R_1" : ""))
        violin!(plt[1], x_rep, R2_sym[n_idx, :], side=:right, color=:forestgreen, alpha=0.8, width=0.4, linewidth=0, label=(n_idx==2 ? L"R_2" : ""))

        
        # Painel Inferior (Antissimétrico)
        violin!(plt[2], x_rep, R1_anti[n_idx, :], side=:left, color=:steelblue, alpha=0.8, width=0.4, linewidth=0, label="")
        violin!(plt[2], x_rep, R2_anti[n_idx, :], side=:right, color=:forestgreen, alpha=0.8, width=0.4, linewidth=0, label="")  
    end
    
    savefig(plt, "figures/painel_distribuicao_sim_anti_tau$(τ).png")
    println("Figura salva: painel_distribuicao_sim_anti_tau$(τ).png")
    return plt
end

run_panel_sweep()