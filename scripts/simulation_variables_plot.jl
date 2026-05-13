# Requires: DifferentialEquations, Plots, JLD2
using DifferentialEquations
using Plots
using JLD2
using LinearAlgebra
using Random

gr() 
default(fontfamily="Computer Modern", linewidth=1.5, label=nothing, grid=false, framestyle=:box)

# Base Parameters
const N = 20
const tspan = (0.0, 50.0)

# Topology Initialization (Fixed across sweeps for fair comparison)
Random.seed!(42)
const omega = randn(N) * 0.5
const A = Float64.(rand(N, N) .< 0.3)
const B = Float64.(rand(N, N, N) .< 0.1)

# Dynamical System Function
function kuramoto_dynamics!(du, x, p, t)
    N_osc = p.N
    θ = @view x[1:N_osc]
    U = reshape(@view(x[N_osc+1:end]), N_osc, N_osc)
    
    dθ = @view du[1:N_osc]
    dU = reshape(@view(du[N_osc+1:end]), N_osc, N_osc)
    
    for i in 1:N_osc
        phase_sum = 0.0
        for j in 1:N_osc
            phase_sum += U[i, j] * sin(θ[j] - θ[i])
        end
        dθ[i] = p.omega[i] + phase_sum
        
        for j in 1:N_osc
            field_sum = 0.0
            for k in 1:N_osc
                field_sum += p.B[i, j, k] * cos(θ[k] - θ[i])
            end
            dU[i, j] = (-U[i, j] + p.K1 * p.A[i, j] + p.K2 * field_sum) / p.tau
        end
    end
end

# Parameter Arrays
K1_vals = [0.0, 0.01, 0.1, 0.2, 0.5, 1.0]
K2_vals = [0.0, 0.01, 0.05, 0.1]
tau_vals = [0.01, 0.5, 1.0, 5.0]

for K1 in K1_vals, K2 in K2_vals, tau in tau_vals
    println("Analyzing: K1=$K1 | K2=$K2 | tau=$tau")
    
    config_name = "K1_$(K1)_K2_$(K2)_tau_$(tau)"
    data_dir = joinpath("data", config_name)
    fig_dir = joinpath("figures", config_name)
    ts_dir = joinpath(fig_dir, "timeseries") # Specific folder for time series
    
    mkpath(data_dir)
    mkpath(fig_dir)
    mkpath(ts_dir)

    cache_path = joinpath(data_dir, "simulation_cache.jld2")
    run_sim = true
    
    # Strict cache validation
    if isfile(cache_path)
        @load cache_path tempos saved_params
        
        if tempos[end] >= tspan[2] && saved_params.K1 == K1 && saved_params.K2 == K2 && saved_params.tau == tau
            println(" -> Valid cache found. Skipping numerical integration.")
            @load cache_path theta_sol U_media_in
            run_sim = false
        else
            println(" -> Cache obsolete or divergent. Rerunning simulation.")
        end
    end

    if run_sim
        params = (N=N, omega=omega, A=A, B=B, K1=K1, K2=K2, tau=tau)
        theta_0 = rand(N) .* 2π
        U_0 = K1 .* A 
        x0 = vcat(theta_0, vec(U_0))

        prob = ODEProblem(kuramoto_dynamics!, x0, tspan, params)
        sol = solve(prob, AutoTsit5(Rosenbrock23()), saveat=0.1)

        tempos = sol.t
        theta_sol = zeros(N, length(tempos))
        U_media_in = zeros(N, length(tempos))

        for (idx, t) in enumerate(tempos)
            state = sol.u[idx]
            theta_sol[:, idx] .= sin.(state[1:N]) 
            
            U_matrix = reshape(state[N+1:end], N, N)
            U_media_in[:, idx] .= dropdims(sum(U_matrix, dims=2), dims=2) ./ N
        end

        saved_params = params
        @save cache_path tempos theta_sol U_media_in saved_params
    end

    # Plotting: Heatmaps
    p_heat1 = heatmap(tempos, 1:N, theta_sol, color=:viridis, title="Phases sin(θ)", clims=(-1, 1))
    p_heat2 = heatmap(tempos, 1:N, U_media_in, color=:inferno, title="Mean U")
    plt_heat = plot(p_heat1, p_heat2, layout=(2, 1), size=(800, 600))
    savefig(plt_heat, joinpath(fig_dir, "raster_plots.png"))

    # Plotting: Time Series (Lines)
    # Transposing matrices so each column is a separate line (oscillator)
    p_line1 = plot(tempos, theta_sol', title="Time Series: Phases sin(θ)", xlabel="Time", ylabel="sin(θ)", palette=:tab20)
    p_line2 = plot(tempos, U_media_in', title="Time Series: Mean Incoming U", xlabel="Time", ylabel="Mean U", palette=:tab20)
    
    plt_line = plot(p_line1, p_line2, layout=(2, 1), size=(800, 600))
    savefig(plt_line, joinpath(ts_dir, "timeseries_plots.png"))
end

println("Parameter sweep completed.")