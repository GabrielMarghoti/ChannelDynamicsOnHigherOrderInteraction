using DifferentialEquations
using LinearAlgebra
using Random
using Statistics
using LaTeXStrings
using StatsPlots

gr()
default(fontfamily = "Computer Modern", linewidth = 2, grid = false, framestyle = :box)

const BASE_OUT_DIR = "figures/kuramoto_tau_sweep"

function dynamic_kuramoto!(dy, y, p, t)
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
            driving  = (K1 * A[i, j] + K2 * local_field / N) 
            du[i, j] = (-u[i, j] + driving) / τ
        end
    end
end

function build_network(N::Int, sym_type::Symbol, rng::AbstractRNG)
    A = ones(Float64, N, N)
    B = zeros(Float64, N, N, N)

    for i in 1:N; A[i, i] = 0.0; end

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

function order_parameter(θ::AbstractVector{<:Real}, q::Int)
    return abs(sum(exp.(im * q .* θ)) / length(θ))
end

function run_and_plot_combined_sweep()
    mkpath(BASE_OUT_DIR)
    
    N = 22
    # Resolução moderada para acomodar os violinos sem sobreposição excessiva
    τ_vals = 10.0 .^ range(-1.1, 2.0, length=10)
    log_τ_vals = log10.(τ_vals) 
    
    K_pairs = [(0.5, 0.0), (0.5, 2.0), (0.1, 10.0)]
    tspan = (0.0, 500.0)
    t_eq  = 300.0
    seed  = 42
    num_trials = 50

    rng = MersenneTwister(seed)
    ω = randn(rng, N) .* 0.1
    A, B_sym  = build_network(N, :symmetric, rng)
    _, B_anti = build_network(N, :antisymmetric, rng)

    c_R1 = :steelblue; c_R2 = :forestgreen; c_R4 = :darkorange
    
    # Rótulos customizados para o eixo X com major ticks labelados e minor ticks sem label
    tick_idx = [1, max(1, floor(Int, length(log_τ_vals) / 3) + 1),
                max(1, floor(Int, 2 * length(log_τ_vals) / 3) + 1), length(log_τ_vals)]
    x_ticks_vals = log_τ_vals
    x_ticks_labs = fill("", length(log_τ_vals))
    x_ticks_labs[tick_idx] .= [L"10^{-1}", L"10^0", L"10^1", L"10^2"]

    println("Running sweep: $(length(τ_vals)) τ points | N=$(N) | Trials=$(num_trials)")

    for (K1, K2) in K_pairs
        println("  -> Processing: K1=$K1, K2=$K2")
        
        res_sym  = Dict(q => zeros(length(τ_vals), num_trials) for q in [1, 2, 4])
        res_anti = Dict(q => zeros(length(τ_vals), num_trials) for q in [1, 2, 4])

        Threads.@threads for t_idx in 1:length(τ_vals)
            τ = τ_vals[t_idx]
            
            for trial in 1:num_trials
                θ0 = rand(N) .* 2π
                y0_init = vcat(θ0, zeros(N * N))

                # Symmetric
                p_sym = (ω, A, B_sym, K1, K2, τ, N)
                sol_sym = solve(ODEProblem(dynamic_kuramoto!, y0_init, tspan, p_sym), Tsit5(), saveat=0.5, reltol=1e-6, abstol=1e-6)
                eq_idx_sym = findall(t -> t >= t_eq, sol_sym.t)
                res_sym[1][t_idx, trial] = mean([order_parameter(sol_sym.u[idx][1:N], 1) for idx in eq_idx_sym])
                res_sym[2][t_idx, trial] = mean([order_parameter(sol_sym.u[idx][1:N], 2) for idx in eq_idx_sym])
                res_sym[4][t_idx, trial] = mean([order_parameter(sol_sym.u[idx][1:N], 4) for idx in eq_idx_sym])

                # Antisymmetric
                p_anti = (ω, A, B_anti, K1, K2, τ, N)
                sol_anti = solve(ODEProblem(dynamic_kuramoto!, y0_init, tspan, p_anti), Tsit5(), saveat=0.5, reltol=1e-6, abstol=1e-6)
                eq_idx_anti = findall(t -> t >= t_eq, sol_anti.t)
                res_anti[1][t_idx, trial] = mean([order_parameter(sol_anti.u[idx][1:N], 1) for idx in eq_idx_anti])
                res_anti[2][t_idx, trial] = mean([order_parameter(sol_anti.u[idx][1:N], 2) for idx in eq_idx_anti])
                res_anti[4][t_idx, trial] = mean([order_parameter(sol_anti.u[idx][1:N], 4) for idx in eq_idx_anti])
            end
        end

# Layout ajustado: painéis próximos (margens negativas) e eixo X compartilhado
        combined_plot = plot(layout = (2, 1), size = (450, 400), 
                             left_margin = 5Plots.mm, right_margin = 5Plots.mm, link=:x)

        # Painel Superior (Symmetric) - sem título, sem ticks no X (para grudar os gráficos)
        plot!(combined_plot[1], ylabel="Coherence "*L"(R_q)", ylims=(-0.05, 1.05), 
              xticks=false, bottom_margin=-3Plots.mm, legend=:topleft)
        annotate!(combined_plot[1], log_τ_vals[1], 0.95, text(" Symmetric", :left, 10))

        # Painel Inferior (Antisymmetric) - sem título
        plot!(combined_plot[2], ylabel="Coherence "*L"(R_q)", xlabel=L"\tau \text{ (Inertia)}", 
              ylims=(-0.05, 1.05), xticks=(x_ticks_vals, x_ticks_labs), top_margin=-3Plots.mm, legend=false)
        annotate!(combined_plot[2], log_τ_vals[1], 0.95, text(" Antisymmetric", :left, 10))

        x_rep = repeat(log_τ_vals, inner=num_trials)

        for (sym_dict, p_idx) in [(res_sym, 1), (res_anti, 2)]
            y_r1 = vec(sym_dict[1]')
            y_r2 = vec(sym_dict[2]')
            
            # width=0.15 restringe a amplitude e impede sobreposição no eixo X contínuo
            violin!(combined_plot[p_idx], x_rep, y_r1, side=:left, color=c_R1, alpha=0.9, 
                    width=0.15, label=p_idx==1 ? L"R_1" : "")
            violin!(combined_plot[p_idx], x_rep, y_r2, side=:right, color=c_R2, alpha=0.9, 
                    width=0.15, label=p_idx==1 ? L"R_2" : "")
            
            # Boxplots (se for usá-los, ajuste também o width)
            #boxplot!(combined_plot[p_idx], x_rep, y_r1, side=:left, color=c_R1, fillalpha=0.6, width=0.04, whisker_range=Inf, outliers=false, label="")
            #boxplot!(combined_plot[p_idx], x_rep, y_r2, side=:right, color=c_R2, fillalpha=0.6, width=0.04, whisker_range=Inf, outliers=false, label="")

            # R4 (mantido igual)
            r4_med = vec(median(sym_dict[4], dims=2))
            r4_min = vec(minimum(sym_dict[4], dims=2))
            r4_max = vec(maximum(sym_dict[4], dims=2))
            
            #scatter!(combined_plot[p_idx], log_τ_vals .+ 0.05, r4_med, yerror=(r4_med .- r4_min, r4_max .- r4_med), color=c_R4, markersize=4, markerstrokewidth=1, label=p_idx==1 ? L"R_4" : "")
        end

        k_label = "K1_$(replace(string(K1), "."=>"p"))_K2_$(replace(string(K2), "."=>"p"))"
        out_file = joinpath(BASE_OUT_DIR, "tau_sweep_violin_split_$(k_label).png")
        savefig(combined_plot, out_file)
        println("    [✓] Saved Panel → $out_file")
    end
end

run_and_plot_combined_sweep()