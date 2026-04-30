using WaterLily
using StaticArrays
using LinearAlgebra
using CUDA
using GLMakie 

# ==========================================
# 1. Geometry Math (Float32 / GPU Safe)
# ==========================================

function naca0022(x)
    t = 0.22
    return 5.0 * t * (0.2969*sqrt(x) - 0.1260*x - 0.3516*x^2 + 0.2843*x^3 - 0.1015*x^4)
end

function sdSegment(p, a, b)
    pa = p - a
    ba = b - a
    T = eltype(p) 
    h = clamp(dot(pa, ba) / dot(ba, ba), T(0.0), T(1.0))
    return norm(pa - ba * h)
end

function sdSkin(p, points)
    d = eltype(p)(Inf) 
    for i in 1:(length(points)-1)
        d = min(d, sdSegment(p, points[i], points[i+1]))
    end
    return d
end

function make_wing_body(L; thickness=1.5, cut_fraction=0.12, T=Float32)
    theta_top = range(0, pi, length=40)
    x_top = 0.5 .* (1.0 .- cos.(theta_top))
    top_pts = Tuple(SVector{2,T}(T(x * L), T(naca0022(x) * L)) for x in x_top)

    theta_start = acos(1.0 - 2.0 * cut_fraction)
    theta_bot = range(theta_start, pi, length=40)
    x_bot = 0.5 .* (1.0 .- cos.(theta_bot))
    bot_pts = Tuple(SVector{2,T}(T(x * L), T(-naca0022(x) * L)) for x in x_bot)

    thick_T = T(thickness)
    sdf = (p, t) -> min(sdSkin(p, top_pts), sdSkin(p, bot_pts)) - thick_T
    return sdf
end

# ==========================================
# 2. Setup Simulation
# ==========================================

function setup_simulation(; L=128, Re=1*10^6, AoA=4.0, cut_fraction=0.12, mem=CuArray, T=Float32)
    width, height = 4 * L, 3 * L 
    center = SVector{2,T}(T(1.5 * L), T(1.5 * L)) 
    
    sdf = make_wing_body(L, cut_fraction=cut_fraction, T=T)
    
    θ = T(-AoA * π / 180.0) 
    R = SMatrix{2,2,T}(cos(θ), -sin(θ), sin(θ), cos(θ))
    
    function map(x, t)
        return R * (x - center)
    end
    
    body = AutoBody(sdf, map)
    return Simulation((width, height), (T(1.0), T(0.0)), L; ν=T(1.0/Re), body=body, T=T, mem=mem)
end

# ==========================================
# 3. Helpers for Math
# ==========================================

function calc_velocity(u_cpu, N, M)
    return [sqrt(u_cpu[x, y, 1]^2 + u_cpu[x, y, 2]^2) for x in 1:N, y in 1:M]
end

function calc_vorticity(u_cpu, N, M)
    vort = zeros(Float32, N, M)
    for x in 2:N-1, y in 2:M-1
        vort[x, y] = (u_cpu[x+1, y, 2] - u_cpu[x-1, y, 2]) - (u_cpu[x, y+1, 1] - u_cpu[x, y-1, 1])
    end
    return vort
end

# ==========================================
# 4. GLMakie 6-Panel Real-Time Dashboard
# ==========================================

L_chord = 128  
# Real-time pacing: 1/30th of a second per frame
t_step = 1.0 / 30.0   
# 300 frames at 30fps = 10 seconds of video
n_frames = 300 

println("Initializing Open & Closed GPU Simulations...")

# Create BOTH simulations
sim_open = setup_simulation(L=L_chord, cut_fraction=0.12, mem=CuArray)
sim_closed = setup_simulation(L=L_chord, cut_fraction=0.0, mem=CuArray)
N_grid, M_grid = size(sim_open.flow.p)

println("Mapping geometries...")
body_map_open = [WaterLily.sdf(sim_open.body, SVector{2,Float32}(Float32(x), Float32(y)), 0f0) for x in 1:N_grid, y in 1:M_grid]
body_map_closed = [WaterLily.sdf(sim_closed.body, SVector{2,Float32}(Float32(x), Float32(y)), 0f0) for x in 1:N_grid, y in 1:M_grid]

# ---------------------------------------------------------
# A. BUILD THE FIGURE
# ---------------------------------------------------------
println("Building GLMakie window...")

# Open Wing Observables
u_obs_op = Observable(zeros(Float32, N_grid, M_grid))
p_obs_op = Observable(zeros(Float32, N_grid, M_grid))
v_obs_op = Observable(zeros(Float32, N_grid, M_grid))

# Closed Wing Observables
u_obs_cl = Observable(zeros(Float32, N_grid, M_grid))
p_obs_cl = Observable(zeros(Float32, N_grid, M_grid))
v_obs_cl = Observable(zeros(Float32, N_grid, M_grid))

time_title = Observable("t = 0.0s")

fig = Figure(size = (1600, 800))

# ROW 1: OPEN WING
ax_vel_op = Axis(fig[1, 1], title = lift(t -> "Open (Ram-Air) - Velocity ($t)", time_title), aspect = DataAspect())
ax_pre_op = Axis(fig[1, 2], title = lift(t -> "Open (Ram-Air) - Pressure ($t)", time_title), aspect = DataAspect())
ax_vor_op = Axis(fig[1, 3], title = lift(t -> "Open (Ram-Air) - Vorticity ($t)", time_title), aspect = DataAspect())

# ROW 2: CLOSED WING
ax_vel_cl = Axis(fig[2, 1], title = lift(t -> "Closed (Solid) - Velocity ($t)", time_title), aspect = DataAspect())
ax_pre_cl = Axis(fig[2, 2], title = lift(t -> "Closed (Solid) - Pressure ($t)", time_title), aspect = DataAspect())
ax_vor_cl = Axis(fig[2, 3], title = lift(t -> "Closed (Solid) - Vorticity ($t)", time_title), aspect = DataAspect())

# HEATMAPS
heatmap!(ax_vel_op, u_obs_op, colormap = :turbo, colorrange = (0.0, 2.0))
heatmap!(ax_pre_op, p_obs_op, colormap = :balance, colorrange = (-0.5, 0.5)) 
heatmap!(ax_vor_op, v_obs_op, colormap = :PiYG, colorrange = (-0.5, 0.5))

heatmap!(ax_vel_cl, u_obs_cl, colormap = :turbo, colorrange = (0.0, 2.0))
heatmap!(ax_pre_cl, p_obs_cl, colormap = :balance, colorrange = (-0.5, 0.5)) 
heatmap!(ax_vor_cl, v_obs_cl, colormap = :PiYG, colorrange = (-0.5, 0.5))

# OUTLINES
contour!(ax_vel_op, body_map_open, levels = [0.0], color = :white, linewidth = 2.0)
contour!(ax_pre_op, body_map_open, levels = [0.0], color = :black, linewidth = 2.0)
contour!(ax_vor_op, body_map_open, levels = [0.0], color = :black, linewidth = 2.0)

contour!(ax_vel_cl, body_map_closed, levels = [0.0], color = :white, linewidth = 2.0)
contour!(ax_pre_cl, body_map_closed, levels = [0.0], color = :black, linewidth = 2.0)
contour!(ax_vor_cl, body_map_closed, levels = [0.0], color = :black, linewidth = 2.0)

# Zoom all cameras
for ax in [ax_vel_op, ax_pre_op, ax_vor_op, ax_vel_cl, ax_pre_cl, ax_vor_cl]
    limits!(ax, 0.5 * L_chord, 3.0 * L_chord, 0.5 * L_chord, 2.5 * L_chord)
    hidedecorations!(ax)
end

# ---------------------------------------------------------
# B. RUN PHYSICS & RECORD
# ---------------------------------------------------------
println("Running dual-simulation...")

Makie.record(fig, "wing_comparison.mp4", 1:n_frames; framerate = 30) do i
    
    current_time = Float32(i * t_step)
    
    # 1. Step BOTH GPUs
    sim_step!(sim_open, current_time)
    sim_step!(sim_closed, current_time)
    
    # 2. Pull Data
    u_cpu_op = Array(sim_open.flow.u)
    p_cpu_op = Array(sim_open.flow.p)
    u_cpu_cl = Array(sim_closed.flow.u)
    p_cpu_cl = Array(sim_closed.flow.p)
    
    # THE PRESSURE FIX: Anchor far-field pressure to exactly 0.0
    p_cpu_op .-= p_cpu_op[2, 2]
    p_cpu_cl .-= p_cpu_cl[2, 2]
    
    # 3. Update Observables
    u_obs_op[] = calc_velocity(u_cpu_op, N_grid, M_grid)
    p_obs_op[] = p_cpu_op
    v_obs_op[] = calc_vorticity(u_cpu_op, N_grid, M_grid)
    
    u_obs_cl[] = calc_velocity(u_cpu_cl, N_grid, M_grid)
    p_obs_cl[] = p_cpu_cl
    v_obs_cl[] = calc_vorticity(u_cpu_cl, N_grid, M_grid)
    
    time_title[] = "t = $(round(current_time, digits=1))s"
    
    if i % 10 == 0
        println("Rendered frame $i / $n_frames")
    end
end

println("Done! Check your folder for 'wing_comparison.mp4'")