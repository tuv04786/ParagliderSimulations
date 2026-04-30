using WaterLily
using StaticArrays
using LinearAlgebra
using CUDA
using GLMakie 

# ==========================================
# 1. Geometry Math (Pure Math, GPU-Safe)
# ==========================================

function naca0022(x::T) where T
    t = T(0.22)
    return T(5.0) * t * (T(0.2969)*sqrt(abs(x)) - T(0.1260)*x - T(0.3516)*x^2 + T(0.2843)*x^3 - T(0.1015)*x^4)
end

function sdSegment(p, a, b)
    pa = p - a
    ba = b - a
    T = eltype(p) 
    h = clamp(dot(pa, ba) / dot(ba, ba), T(0.0), T(1.0))
    return norm(pa - ba * h)
end

function get_fabric_starts(cut_fraction, cut_angle_deg)
    xs = range(0, 1, length=1000)
    nx = sin(cut_angle_deg * π / 180.0)
    ny = -cos(cut_angle_deg * π / 180.0)
    
    idx_top = findfirst(x -> (x - cut_fraction)*nx + Float64(naca0022(Float32(x)))*ny > 0, xs)
    idx_bot = findfirst(x -> (x - cut_fraction)*nx + Float64(-naca0022(Float32(x)))*ny > 0, xs)
    
    x_top = isnothing(idx_top) ? 0.0 : xs[idx_top]
    x_bot = isnothing(idx_bot) ? 0.0 : xs[idx_bot]
    
    return x_top, x_bot
end

# THE UPGRADE: Fully Decoupled Array-Free SDF
function make_wing_body(L; thickness=1.5, cut_fraction=0.0, cut_angle_deg=90.0, T=Float32)
    x_top, x_bot = get_fabric_starts(cut_fraction, cut_angle_deg)
    
    # Pre-calculate EVERYTHING as raw values in the outer scope
    ts_top = T(acos(clamp(1.0 - 2.0 * x_top, -1.0, 1.0)))
    ts_bot = T(acos(clamp(1.0 - 2.0 * x_bot, -1.0, 1.0)))
    thick_v = T(thickness)
    L_v = T(L)
    pi_v = T(pi)
    inf_v = T(Inf)
    one_v = T(1.0)
    half_v = T(0.5)
    
    # Pre-calculate step sizes
    d_theta_top = (pi_v - ts_top) / T(40)
    d_theta_bot = (pi_v - ts_bot) / T(40)

    # The truly clean closure for the GPU
    sdf = (p, t) -> begin
        # Extract the vector type directly from the point 'p' natively on the GPU
        PT = typeof(p)     

        d_top = inf_v
        d_bot = inf_v
        
        # --- Top Surface ---
        theta_a_t = ts_top
        x_a_t = half_v * (one_v - cos(theta_a_t))
        pt_a_t = PT(x_a_t * L_v, naca0022(x_a_t) * L_v)
        
        for i in 1:40
            # Explicit cast of loop index to Float32 to avoid Int to Float promotion errors
            theta_b = ts_top + Float32(i) * d_theta_top
            x_b = half_v * (one_v - cos(theta_b))
            pt_b = PT(x_b * L_v, naca0022(x_b) * L_v)
            
            d_top = min(d_top, sdSegment(p, pt_a_t, pt_b))
            pt_a_t = pt_b
        end
        
        # --- Bottom Surface ---
        theta_a_b = ts_bot
        x_a_b = half_v * (one_v - cos(theta_a_b))
        pt_a_b = PT(x_a_b * L_v, -naca0022(x_a_b) * L_v)
        
        for i in 1:40
            theta_b = ts_bot + Float32(i) * d_theta_bot
            x_b = half_v * (one_v - cos(theta_b))
            pt_b = PT(x_b * L_v, -naca0022(x_b) * L_v)
            
            d_bot = min(d_bot, sdSegment(p, pt_a_b, pt_b))
            pt_a_b = pt_b
        end
        
        return min(d_top, d_bot) - thick_v
    end
    return sdf
end

# ==========================================
# 2. Setup Simulation
# ==========================================

function setup_simulation(; L=128, Re=1*10^6, AoA=4.0, cut_fraction=0.0, cut_angle_deg=90.0, mem=CuArray, T=Float32)
    width, height = 4 * L, 3 * L 
    center = SVector{2,T}(T(1.5 * L), T(1.5 * L)) 
    
    sdf = make_wing_body(L, cut_fraction=cut_fraction, cut_angle_deg=cut_angle_deg, T=T)
    
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

function get_grid_idx(xc, yc, L, center, θ_sim, N_grid, M_grid)
    R_inv = SMatrix{2,2,Float32}(cos(-θ_sim), -sin(-θ_sim), sin(-θ_sim), cos(-θ_sim))
    p_chord = SVector{2,Float32}(xc * L, yc * L)
    p_grid = R_inv * p_chord + center
    
    ix = clamp(round(Int, p_grid[1]), 1, N_grid)
    iy = clamp(round(Int, p_grid[2]), 1, M_grid)
    return ix, iy
end

# ==========================================
# 4. Initialization & Sensor Mapping
# ==========================================

L_chord = 512  
t_step = 1.0 / 30.0   
n_frames = 600 

AoA_deg = 0.0          
cut_frac = 0.050         # Absolute front
cut_angle = 90.0       # Perfectly vertical cut

println("Initializing Array-Free GPU Simulation...")

sim = setup_simulation(L=L_chord, Re=1*10^4, AoA=AoA_deg, cut_fraction=cut_frac, cut_angle_deg=cut_angle, mem=CuArray, T=Float32)
N_grid, M_grid = size(sim.flow.p)
center_pt = SVector{2,Float32}(1.5 * L_chord, 1.5 * L_chord)
θ_sim = Float32(-AoA_deg * π / 180.0)

body_map = [WaterLily.sdf(sim.body, SVector{2,Float32}(Float32(x), Float32(y)), 0f0) for x in 1:N_grid, y in 1:M_grid]

x_start_top, x_start_bot = get_fabric_starts(cut_frac, cut_angle)
offset = 3.0 / L_chord 

x_top_arr = collect(range(x_start_top + 0.01, 0.98, length=80))
idx_out_top = [get_grid_idx(x, naca0022(Float32(x)) + offset, L_chord, center_pt, θ_sim, N_grid, M_grid) for x in x_top_arr]
idx_in_top  = [get_grid_idx(x, naca0022(Float32(x)) - offset, L_chord, center_pt, θ_sim, N_grid, M_grid) for x in x_top_arr]

x_bot_arr = collect(range(x_start_bot + 0.01, 0.98, length=70))
idx_out_bot = [get_grid_idx(x, -naca0022(Float32(x)) - offset, L_chord, center_pt, θ_sim, N_grid, M_grid) for x in x_bot_arr]
idx_in_bot  = [get_grid_idx(x, -naca0022(Float32(x)) + offset, L_chord, center_pt, θ_sim, N_grid, M_grid) for x in x_bot_arr]

p_out_top_sum = zeros(Float32, length(x_top_arr))
p_in_top_sum  = zeros(Float32, length(x_top_arr))
p_out_bot_sum = zeros(Float32, length(x_bot_arr))
p_in_bot_sum  = zeros(Float32, length(x_bot_arr))
avg_frames = 0

# ==========================================
# 5. Build Live Video Dashboard
# ==========================================
println("Building GLMakie Video Window...")

u_obs = Observable(zeros(Float32, N_grid, M_grid))
p_obs = Observable(zeros(Float32, N_grid, M_grid))
v_obs = Observable(zeros(Float32, N_grid, M_grid))

p_out_top_inst = Observable(zeros(Float32, length(x_top_arr)))
p_in_top_inst  = Observable(zeros(Float32, length(x_top_arr)))
p_out_bot_inst = Observable(zeros(Float32, length(x_bot_arr)))
p_in_bot_inst  = Observable(zeros(Float32, length(x_bot_arr)))

time_title = Observable("t = 0.0s")

fig_live = Figure(size = (1600, 1000))

ax_vel = Axis(fig_live[1, 1], title = lift(t -> "Velocity ($t)", time_title), aspect = DataAspect())
ax_pre = Axis(fig_live[1, 2], title = lift(t -> "Pressure ($t)", time_title), aspect = DataAspect())
ax_vor = Axis(fig_live[2, 1], title = lift(t -> "Vorticity ($t)", time_title), aspect = DataAspect())
ax_graph = Axis(fig_live[2, 2], title = "Live Instantaneous Cp", xlabel = "x/c", ylabel = "Cp", yreversed = true)

heatmap!(ax_vel, u_obs, colormap = :turbo, colorrange = (0.0, 2.0))
heatmap!(ax_pre, p_obs, colormap = :balance, colorrange = (-0.5, 0.5)) 
heatmap!(ax_vor, v_obs, colormap = :PiYG, colorrange = (-0.5, 0.5))

contour!(ax_vel, body_map, levels = [0.0], color = :white, linewidth = 2.0)
contour!(ax_pre, body_map, levels = [0.0], color = :black, linewidth = 2.0)
contour!(ax_vor, body_map, levels = [0.0], color = :black, linewidth = 2.0)

lines!(ax_graph, x_top_arr, p_out_top_inst, color = :darkblue, linewidth = 2, label = "Outer Top")
lines!(ax_graph, x_top_arr, p_in_top_inst, color = :deepskyblue, linewidth = 2, label = "Inner Top")
lines!(ax_graph, x_bot_arr, p_out_bot_inst, color = :red, linewidth = 2, label = "Outer Bottom")
lines!(ax_graph, x_bot_arr, p_in_bot_inst, color = :orange, linewidth = 2, label = "Inner Bottom")

axislegend(ax_graph, position = :rt)
xlims!(ax_graph, 0.0, 1.0)

for ax in [ax_vel, ax_pre, ax_vor]
    limits!(ax, 0.5 * L_chord, 3.0 * L_chord, 0.5 * L_chord, 2.5 * L_chord)
    hidedecorations!(ax)
end

# ==========================================
# 6. Run Physics & Record Video
# ==========================================
println("Running simulation and rendering video...")

Makie.record(fig_live, "ram_air_live_dynamics.mp4", 1:n_frames; framerate = 30) do i
    
    current_time = Float32(i * t_step)
    sim_step!(sim, current_time)
    
    u_cpu = Array(sim.flow.u)
    p_cpu = Array(sim.flow.p)
    p_cpu .-= p_cpu[2, 2] 
    
    u_obs[] = calc_velocity(u_cpu, N_grid, M_grid)
    p_obs[] = p_cpu
    v_obs[] = calc_vorticity(u_cpu, N_grid, M_grid)
    
    cp_out_top = [2.0f0 * p_cpu[idx[1], idx[2]] for idx in idx_out_top]
    cp_in_top  = [2.0f0 * p_cpu[idx[1], idx[2]] for idx in idx_in_top]
    cp_out_bot = [2.0f0 * p_cpu[idx[1], idx[2]] for idx in idx_out_bot]
    cp_in_bot  = [2.0f0 * p_cpu[idx[1], idx[2]] for idx in idx_in_bot]
    
    p_out_top_inst[] = cp_out_top
    p_in_top_inst[]  = cp_in_top
    p_out_bot_inst[] = cp_out_bot
    p_in_bot_inst[]  = cp_in_bot
    
    global avg_frames += 1
    p_out_top_sum .+= cp_out_top
    p_in_top_sum  .+= cp_in_top
    p_out_bot_sum .+= cp_out_bot
    p_in_bot_sum  .+= cp_in_bot
    
    autolimits!(ax_graph) 
    xlims!(ax_graph, 0.0, 1.0)
    
    time_title[] = "t = $(round(current_time, digits=1))s"
    
    if i % 10 == 0
        println("Rendered frame $i / $n_frames")
    end
end
println("Video complete: 'ram_air_live_dynamics.mp4'")

# ==========================================
# 7. Aerodynamic Integration & Final Graph
# ==========================================
println("Calculating force coefficients and stagnation point...")

avg_Cp_out_top = p_out_top_sum ./ avg_frames
avg_Cp_in_top  = p_in_top_sum ./ avg_frames
avg_Cp_out_bot = p_out_bot_sum ./ avg_frames
avg_Cp_in_bot  = p_in_bot_sum ./ avg_frames

max_vals = [maximum(avg_Cp_out_top), maximum(avg_Cp_in_top), maximum(avg_Cp_out_bot), maximum(avg_Cp_in_bot)]
max_idxs = [argmax(avg_Cp_out_top), argmax(avg_Cp_in_top), argmax(avg_Cp_out_bot), argmax(avg_Cp_in_bot)]
x_arrays = [x_top_arr, x_top_arr, x_bot_arr, x_bot_arr]

global_max_val, global_max_idx = findmax(max_vals)
stag_cp = global_max_val
stag_x = x_arrays[global_max_idx][max_idxs[global_max_idx]]

# Accurate force coefficient calculation using pressure distribution
# For 2D airfoil: Cp = (p - p_inf) / (0.5 * rho * U^2), normalized units here
# Force coefficients are integrated per unit span

# Calculate pressure difference (inner - outer, i.e., lower - upper surface pressure)
# Note: For a ram air wing, pressure is higher on the lower surface (inside)
Δcp_top = avg_Cp_in_top .- avg_Cp_out_top  # Pressure coefficient difference on upper surface
Δcp_bot = avg_Cp_out_bot .- avg_Cp_in_bot  # Pressure coefficient difference on lower surface (ram air inside)

# Calculate surface geometry for proper normal vector integration
y_top = naca0022.(Float32.(x_top_arr))
y_bot = -naca0022.(Float32.(x_bot_arr))

# Use trapezoidal integration for more accurate results
dx_top = x_top_arr[2] - x_top_arr[1]
dx_bot = x_bot_arr[2] - x_bot_arr[1]

# Normal force coefficient (perpendicular to freestream)
# CN = integral of Δcp along the surface
# Using trapezoidal rule for better accuracy
function trapz(x_vals, y_vals)
    if length(x_vals) != length(y_vals) || length(x_vals) < 2
        return 0.0
    end
    dx = diff(x_vals)
    integral = sum((y_vals[1:end-1] .+ y_vals[2:end]) .* dx ./ 2.0)
    return integral
end

# Compute surface slopes (dy/dx) for axial force calculation
# Using central differences for better accuracy
dy_dx_top = zeros(Float32, length(x_top_arr))
dy_dx_bot = zeros(Float32, length(x_bot_arr))

for i in 2:length(x_top_arr)-1
    dy_dx_top[i] = (y_top[i+1] - y_top[i-1]) / (2.0 * dx_top)
end
dy_dx_top[1] = (y_top[2] - y_top[1]) / dx_top
dy_dx_top[end] = (y_top[end] - y_top[end-1]) / dx_top

for i in 2:length(x_bot_arr)-1
    dy_dx_bot[i] = (y_bot[i+1] - y_bot[i-1]) / (2.0 * dx_bot)
end
dy_dx_bot[1] = (y_bot[2] - y_bot[1]) / dx_bot
dy_dx_bot[end] = (y_bot[end] - y_bot[end-1]) / dx_bot

# Normal force: CN = integral(Δcp dx/c) from leading edge to trailing edge
CN_top = trapz(x_top_arr, Δcp_top)
CN_bot = trapz(x_bot_arr, Δcp_bot)
CN = CN_top + CN_bot

# Axial force: CA = integral(Δcp * tan(θ) dx/c) ≈ integral(Δcp * dy/dx dx/c)
# This accounts for the pressure acting on the surface slope
CA_top = trapz(x_top_arr, Δcp_top .* dy_dx_top)
CA_bot = trapz(x_bot_arr, Δcp_bot .* dy_dx_bot)
CA = CA_top + CA_bot

# Convert to lift and drag using angle of attack
alpha_rad = AoA_deg * π / 180.0
CL_total = CN * cos(alpha_rad) - CA * sin(alpha_rad)
CD_total = CN * sin(alpha_rad) + CA * cos(alpha_rad)

# Account for ram air inlet effects: the inlet creates additional drag
# The cut surface (ram air inlet) experiences pressure recovery
inlet_area_ratio = 1.0 - x_start_top  # Fraction of chord affected by inlet
inlet_pressure_recovery = 0.8  # Estimated pressure recovery factor (0-1)
CD_inlet_penalty = inlet_area_ratio * (1.0 - inlet_pressure_recovery) * 0.05  # Small pressure drag contribution
CD_total += CD_inlet_penalty

println("=== Aerodynamic Coefficients ===")
println("CN (Normal Force) = $(round(CN, digits=4))")
println("CA (Axial Force) = $(round(CA, digits=4))")
println("CL (Lift) = $(round(CL_total, digits=4))")
println("CD (Drag) = $(round(CD_total, digits=4))")
println("Inlet drag penalty = $(round(CD_inlet_penalty, digits=5))")
println("Stagnation Point: Cp = $(round(stag_cp, digits=4)) at x/c = $(round(stag_x, digits=4))")

println("Generating Final Time-Averaged Report Image...")

fig_final = Figure(size = (1000, 700))

# Main pressure coefficient plot
ax_final = Axis(fig_final[1:2, 1:2], 
                title = "Time-Averaged Pressure Coefficient Distribution\n NACA0022 Ram Air Wing (Cut at $(round(x_start_top, digits=3)))", 
                xlabel = "x/c (Chord Fraction)", 
                ylabel = "Cp (Pressure Coefficient)", 
                yreversed = true)

lines!(ax_final, x_top_arr, avg_Cp_out_top, color = :darkblue, linewidth = 2.5, label = "Outer Top (Suction)")
lines!(ax_final, x_top_arr, avg_Cp_in_top, color = :deepskyblue, linewidth = 2.5, label = "Inner Top (Stagnation)")
lines!(ax_final, x_bot_arr, avg_Cp_out_bot, color = :red, linewidth = 2.5, label = "Outer Bottom (Ram Pressure)")
lines!(ax_final, x_bot_arr, avg_Cp_in_bot, color = :orange, linewidth = 2.5, label = "Inner Bottom (Ram Stagnation)")

scatter!(ax_final, [stag_x], [stag_cp], color = :gold, markersize = 12, marker = :star5, label = "Stagnation Point")
text!(ax_final, stag_x + 0.05, stag_cp - 0.05, text = "Max Cp: $(round(stag_cp, digits=2))", align = (:left, :center), color = :black, fontsize=11)

# Mark the cut location
vlines!(ax_final, [x_start_top], color = :purple, linewidth = 2.5, linestyle = :dash, label = "Inlet Cut")
vlines!(ax_final, [x_start_bot], color = :purple, linewidth = 2.5, linestyle = :dash)

axislegend(ax_final, position = :rb, fontsize = 10)
xlims!(ax_final, 0.0, 1.0)

# Force coefficients summary panel
ax_text = Axis(fig_final[1, 3], aspect = DataAspect())
hidedecorations!(ax_text)

textstr = """
AERODYNAMIC COEFFICIENTS
━━━━━━━━━━━━━━━━━━━━━━━
Angle of Attack: $(AoA_deg)°
Reynolds Number: 1.0×10⁴
Cut Position: $(round(x_start_top, digits=3))

FORCES (per unit span):
CN = $(round(CN, digits=4))
CA = $(round(CA, digits=4))

TOTAL COEFFICIENTS:
CL = $(round(CL_total, digits=4))
CD = $(round(CD_total, digits=4))
L/D = $(round(CL_total/max(abs(CD_total), 1e-6), digits=2))

STAGNATION POINT:
Cp,max = $(round(stag_cp, digits=4))
x/c = $(round(stag_x, digits=4))
"""

text!(ax_text, 0.5, 0.5, text = textstr, fontsize = 10, align = (:center, :center), color = :black)

# Pressure difference (Δcp) plot showing net effect
ax_delta = Axis(fig_final[2, 3], title = "ΔCp (Lower - Upper)", xlabel = "x/c", ylabel = "ΔCp")
lines!(ax_delta, x_top_arr, Δcp_top, color = :blue, linewidth = 2, label = "Upper Surface")
lines!(ax_delta, x_bot_arr, Δcp_bot, color = :red, linewidth = 2, label = "Lower Surface (Ram)")
axislegend(ax_delta, position = :rt, fontsize = 9)
xlims!(ax_delta, 0.0, 1.0)
hlines!(ax_delta, [0.0], color = :gray, linewidth = 1, linestyle = :dash)

save("final_averaged_Cp.png", fig_final)
println("Report Image Saved: 'final_averaged_Cp.png'")