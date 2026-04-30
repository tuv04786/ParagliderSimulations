using WaterLily
using StaticArrays
using LinearAlgebra
using CUDA
using GLMakie
using DelimitedFiles 

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

function make_wing_body(L; thickness=1.5, cut_fraction=0.05, cut_angle_deg=90.0, T=Float32)
    x_start_top, x_start_bot = get_fabric_starts(cut_fraction, cut_angle_deg)
    
    ts_top = T(acos(clamp(1.0 - 2.0 * x_start_top, -1.0, 1.0)))
    ts_bot = T(acos(clamp(1.0 - 2.0 * x_start_bot, -1.0, 1.0)))

    theta_top = range(ts_top, pi, length=40)
    x_top = 0.5 .* (1.0 .- cos.(theta_top))
    top_pts = Tuple(SVector{2,T}(T(x * L), T(naca0022(x) * L)) for x in x_top)

    theta_bot = range(ts_bot, pi, length=40)
    x_bot = 0.5 .* (1.0 .- cos.(theta_bot))
    bot_pts = Tuple(SVector{2,T}(T(x * L), T(-naca0022(x) * L)) for x in x_bot)

    thick_T = T(thickness)
    sdf = (p, t) -> min(sdSkin(p, top_pts), sdSkin(p, bot_pts)) - thick_T
    return sdf
end

# ==========================================
# 2. Setup Simulation
# ==========================================
function setup_simulation(; L=128, Re=5*10^5, AoA=4.0, cut_fraction=0.05, cut_angle_deg=90.0, mem=CuArray, T=Float32)
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
# 3. Helpers for Math & Sensors
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
# 4. MASTER SCRIPT CONFIGURATION
# ==========================================
L_chord = 128  
t_step = 1.0 / 30.0   
n_frames = 300 

# Parameter Sweeps
AoA_list = -20:2:20
cut_frac_list = [0.0, 0.01, 0.025, 0.05, 0.075, 0.10] 
cut_angles = [90.0, 100.0, 110.0, 120.0, 130.0, 140.0, 150.0] 

root_dir = "Overall_Category_Data"

# --- UPDATED: INITIALIZE MASTER RESULTS MATRIX ---
# Now tracking: (AoA, CutAngle, OpeningPercent, Pass/Fail, CL, CD, L/D)
summary_matrix = Tuple{Float64, Float64, Float64, String, Float64, Float64, Float64}[]

# --- NESTED LOOPS ---
for c_angle in cut_angles
    sweep_dir = joinpath(root_dir, "Sweep_$(Int(c_angle))deg")
    
    for c_frac in cut_frac_list
        pct_str = replace(string(c_frac * 100), "." => "_")
        open_dir = joinpath(sweep_dir, "Opening_$(pct_str)pct")
        
        mkpath(open_dir) 
        
        for current_AoA in AoA_list
            println("\n==================================================")
            println("RUNNING: Sweep $(c_angle)° | Opening $(c_frac*100)% | AoA $(current_AoA)°")
            println("==================================================")
            
            # ---------------------------------------------------------
            # 1. SETUP SINGLE SIMULATION
            # ---------------------------------------------------------
            sim = setup_simulation(L=L_chord, Re=5*10^5, AoA=current_AoA, 
                                   cut_fraction=c_frac, cut_angle_deg=c_angle, mem=CuArray)
            N_grid, M_grid = size(sim.flow.p)

            body_map = [WaterLily.sdf(sim.body, SVector{2,Float32}(Float32(x), Float32(y)), 0f0) for x in 1:N_grid, y in 1:M_grid]

            θ_sim = Float32(-current_AoA * π / 180.0)
            center_pt = SVector{2,Float32}(1.5 * L_chord, 1.5 * L_chord)
            offset = 3.0 / L_chord

            x_start_top, x_start_bot = get_fabric_starts(c_frac, c_angle)
            tap_x_top = collect(range(x_start_top + 0.01, 0.90, length=80))
            tap_x_bot = collect(range(x_start_bot + 0.01, 0.90, length=70))
            
            idx_out_top = [get_grid_idx(x, naca0022(Float32(x)) + offset, L_chord, center_pt, θ_sim, N_grid, M_grid) for x in tap_x_top]
            idx_in_top  = [get_grid_idx(x, naca0022(Float32(x)) - offset, L_chord, center_pt, θ_sim, N_grid, M_grid) for x in tap_x_top]
            idx_out_bot = [get_grid_idx(x, -naca0022(Float32(x)) - offset, L_chord, center_pt, θ_sim, N_grid, M_grid) for x in tap_x_bot]
            idx_in_bot  = [get_grid_idx(x, -naca0022(Float32(x)) + offset, L_chord, center_pt, θ_sim, N_grid, M_grid) for x in tap_x_bot]

            p_out_top_sum = zeros(Float32, length(tap_x_top))
            p_in_top_sum  = zeros(Float32, length(tap_x_top))
            p_out_bot_sum = zeros(Float32, length(tap_x_bot))
            p_in_bot_sum  = zeros(Float32, length(tap_x_bot))

            native_Cd_hist = Float32[]
            native_Cl_hist = Float32[]

            # ---------------------------------------------------------
            # 2. BUILD THE DYNAMIC FIGURE
            # ---------------------------------------------------------
            u_obs = Observable(zeros(Float32, N_grid, M_grid))
            p_obs = Observable(zeros(Float32, N_grid, M_grid))
            v_obs = Observable(zeros(Float32, N_grid, M_grid))

            time_title = Observable("AoA: $(current_AoA)° | t = 0.0s")
            fig = Figure(size = (1600, 400))

            ax_vel = Axis(fig[1, 1], title = lift(t -> "Velocity ($t)", time_title), aspect = DataAspect())
            ax_pre = Axis(fig[1, 2], title = lift(t -> "Pressure ($t)", time_title), aspect = DataAspect())
            ax_vor = Axis(fig[1, 3], title = lift(t -> "Vorticity ($t)", time_title), aspect = DataAspect())

            heatmap!(ax_vel, u_obs, colormap = :turbo, colorrange = (0.0, 2.0))
            heatmap!(ax_pre, p_obs, colormap = :balance, colorrange = (-0.5, 0.5)) 
            heatmap!(ax_vor, v_obs, colormap = :PiYG, colorrange = (-0.5, 0.5))

            contour!(ax_vel, body_map, levels = [0.0], color = :white, linewidth = 2.0)
            contour!(ax_pre, body_map, levels = [0.0], color = :black, linewidth = 2.0)
            contour!(ax_vor, body_map, levels = [0.0], color = :black, linewidth = 2.0)

            for ax in [ax_vel, ax_pre, ax_vor]
                limits!(ax, 0.5 * L_chord, 3.0 * L_chord, 0.5 * L_chord, 2.5 * L_chord)
                hidedecorations!(ax)
            end

            # ---------------------------------------------------------
            # 3. RUN PHYSICS & RECORD
            # ---------------------------------------------------------
            video_filename = joinpath(open_dir, "Video_AoA_$(current_AoA).mp4")
            
            Makie.record(fig, video_filename, 1:n_frames; framerate = 30) do frame_idx
                current_time = Float32(frame_idx * t_step)
                
                sim_step!(sim, current_time)
                
                u_cpu = Array(sim.flow.u)
                p_cpu = Array(sim.flow.p)
                
                p_cpu .-= p_cpu[2, div(M_grid, 2)]
                
                u_obs[] = calc_velocity(u_cpu, N_grid, M_grid)
                p_obs[] = p_cpu
                v_obs[] = calc_vorticity(u_cpu, N_grid, M_grid)
                
                cp_out_top = [2.0f0 * p_cpu[idx[1], idx[2]] for idx in idx_out_top]
                cp_in_top  = [2.0f0 * p_cpu[idx[1], idx[2]] for idx in idx_in_top]
                cp_out_bot = [2.0f0 * p_cpu[idx[1], idx[2]] for idx in idx_out_bot]
                cp_in_bot  = [2.0f0 * p_cpu[idx[1], idx[2]] for idx in idx_in_bot]
                
                p_out_top_sum .+= cp_out_top
                p_in_top_sum  .+= cp_in_top
                p_out_bot_sum .+= cp_out_bot
                p_in_bot_sum  .+= cp_in_bot
                
                force = WaterLily.pressure_force(sim)
                q_ref = 0.5f0 * 1.0f0^2 * Float32(L_chord)
                
                push!(native_Cd_hist, -force[1] / q_ref)
                push!(native_Cl_hist, -force[2] / q_ref)
                
                time_title[] = "AoA: $(current_AoA)° | t = $(round(current_time, digits=1))s"
            end

            # ---------------------------------------------------------
            # 4. GENERATE FINAL AVERAGE PRESSURE DIAGRAMS
            # ---------------------------------------------------------
            fig_final = Figure(size = (1200, 650))

            start_idx = div(length(native_Cd_hist), 2)
            avg_CD = sum(native_Cd_hist[start_idx:end]) / length(native_Cd_hist[start_idx:end])
            avg_CL = sum(native_Cl_hist[start_idx:end]) / length(native_Cl_hist[start_idx:end])
            avg_L_over_D = avg_CL / max(abs(avg_CD), 1e-6)

            avg_Cp_out_top = p_out_top_sum ./ n_frames
            avg_Cp_in_top  = p_in_top_sum ./ n_frames
            avg_Cp_out_bot = p_out_bot_sum ./ n_frames
            avg_Cp_in_bot  = p_in_bot_sum ./ n_frames

            ax_cp = Axis(fig_final[1:2, 1], 
                         title = "Time-Averaged Cp | AoA: $(current_AoA)° | Cut: $(c_angle)° | Open: $(c_frac*100)%", 
                         xlabel = "x/c", ylabel = "Cp", yreversed = true)

            lines!(ax_cp, tap_x_top, avg_Cp_out_top, color = :darkblue, linewidth = 2.5, label = "Outer Top")
            lines!(ax_cp, tap_x_top, avg_Cp_in_top, color = :deepskyblue, linewidth = 2.5, label = "Inner Top")
            lines!(ax_cp, tap_x_bot, avg_Cp_out_bot, color = :red, linewidth = 2.5, label = "Outer Bottom")
            lines!(ax_cp, tap_x_bot, avg_Cp_in_bot, color = :orange, linewidth = 2.5, label = "Inner Bottom")

            # --- STAGNATION POINT & PASS/FAIL LOGIC ---
            max_out_top = maximum(avg_Cp_out_top)
            max_in_top  = maximum(avg_Cp_in_top)
            max_out_bot = maximum(avg_Cp_out_bot)
            max_in_bot  = maximum(avg_Cp_in_bot)

            stag_Cp = maximum([max_out_top, max_in_top, max_out_bot, max_in_bot])
            
            # Ram-air physics: If the stagnation point hits the exterior skin, the wing deflates
            is_fail = (stag_Cp == max_out_top) || (stag_Cp == max_out_bot)
            
            if is_fail
                ax_cp.backgroundcolor = :mistyrose 
                pass_status = "FAIL"
            else
                pass_status = "PASS"
            end

            # --- UPDATED: Record to Master Matrix ---
            push!(summary_matrix, (current_AoA, c_angle, c_frac * 100, pass_status, avg_CL, avg_CD, avg_L_over_D))

            all_cps = [avg_Cp_out_top; avg_Cp_in_top; avg_Cp_out_bot; avg_Cp_in_bot]
            all_xs  = [tap_x_top; tap_x_top; tap_x_bot; tap_x_bot]
            stag_idx = argmax(all_cps)
            stag_x = all_xs[stag_idx]
            
            scatter!(ax_cp, [stag_x], [stag_Cp], 
                     marker = :star5, markersize = 25, 
                     color = :gold, strokecolor = :black, strokewidth = 1.5, 
                     label = "Stag Point ($(pass_status))")

            xlims!(ax_cp, 0.0, 1.0)

            # --- LAYOUT: SIDEBAR PANELS ---
            Legend(fig_final[1, 2], ax_cp, "Surfaces")

            force_text = "NATIVE FORCES\n\nAoA: $(current_AoA)°\nCL: $(round(avg_CL, digits=3))\nCD: $(round(avg_CD, digits=3))\nL/D: $(round(avg_L_over_D, digits=1))"
            Box(fig_final[2, 2], color = :gray95, strokecolor = :black, strokewidth = 1)
            Label(fig_final[2, 2], force_text, 
                  fontsize = 18, font = :bold, color = :black,
                  justification = :left, halign = :center, valign = :center)

            colsize!(fig_final.layout, 1, Relative(0.75))

            image_filename = joinpath(open_dir, "Plot_Cp_AoA_$(current_AoA).png")
            save(image_filename, fig_final)
            
            GC.gc(true) 
            CUDA.reclaim()
        end
    end
end

println("\nAll 882 iterations complete!")

# ---------------------------------------------------------
# 5. SAVE AND DISPLAY SUMMARY MATRIX
# ---------------------------------------------------------
println("\n==================================================")
println("FINAL PASS/FAIL SUMMARY")
println("==================================================")

csv_path = joinpath(root_dir, "Wing_Pass_Fail_Summary.csv")

# --- UPDATED: Header for CSV ---
header = ["Angle_of_Attack", "Cut_Angle", "Opening_Percent", "Status", "CL", "CD", "L_over_D"]
data_to_write = vcat(permutedims(header), summary_matrix)

writedlm(csv_path, data_to_write, ',')
println("Summary matrix successfully saved to: $csv_path")