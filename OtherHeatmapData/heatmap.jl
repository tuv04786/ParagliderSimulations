using GLMakie, DelimitedFiles

println("Loading Master CSV Data...")
# 1. Load the data (Ensure this matches the path of your completed sweep)
csv_path = joinpath("Wing_Pass_Fail_Summary_Single_AoA copy.csv")
raw_data, headers = readdlm(csv_path, ',', header=true)

# Extract and convert columns from the Any matrix
aoas       = Float64.(raw_data[:, 1])
cut_angles = Float64.(raw_data[:, 2])
openings   = Float64.(raw_data[:, 3])
statuses   = string.(raw_data[:, 4])
CLs        = Float64.(raw_data[:, 5])
CDs        = Float64.(raw_data[:, 6])
LDs        = Float64.(raw_data[:, 7])

# 2. Extract unique axes
unique_aoas  = sort(unique(aoas))
unique_cuts  = sort(unique(cut_angles))
unique_opens = sort(unique(openings))

# Create a folder for the new images
out_dir = joinpath("Overall_Category_Data", "AoA_Heatmaps")
mkpath(out_dir)

println("Found $(length(unique_aoas)) unique Angles of Attack. Generating plots...")

# 3. Generate a 2x2 Heatmap dashboard for each AoA
for aoa in unique_aoas
    # Initialize 2D matrices (Rows: Cut Angles, Cols: Opening %)
    # Filled with NaN so any missing data just shows up as a blank square
    mat_status = fill(NaN, length(unique_cuts), length(unique_opens))
    mat_CL     = fill(NaN, length(unique_cuts), length(unique_opens))
    mat_CD     = fill(NaN, length(unique_cuts), length(unique_opens))
    mat_LD     = fill(NaN, length(unique_cuts), length(unique_opens))
    
    # Filter rows for this specific AoA
    idx = findall(x -> x == aoa, aoas)
    
    # Populate the matrices
    for i in idx
        c_idx = findfirst(x -> x == cut_angles[i], unique_cuts)
        o_idx = findfirst(x -> x == openings[i], unique_opens)
        
        # Map "PASS" to 1.0 and "FAIL" to 0.0 for the color map
        mat_status[c_idx, o_idx] = statuses[i] == "PASS" ? 1.0 : 0.0 
        mat_CL[c_idx, o_idx]     = CLs[i]
        mat_CD[c_idx, o_idx]     = CDs[i]
        mat_LD[c_idx, o_idx]     = LDs[i]
    end

    # --- Build the Makie Figure with strict 4-column layout ---
    fig = Figure(size = (1200, 1000), fontsize = 18)
    Label(fig[0, :], "Design Space Heatmaps | Angle of Attack: $(aoa)°", font = :bold, fontsize = 24)

    # 1. Status (Pass/Fail) -> Col 1 & 2
    ax_stat = Axis(fig[1, 1], title = "Stagnation Status", xlabel = "Cut Angle [°]", ylabel = "Opening [%]")
    hm_stat = heatmap!(ax_stat, unique_cuts, unique_opens, mat_status, colormap = [:mistyrose, :lightgreen], colorrange = (0, 1))
    Colorbar(fig[1, 2], hm_stat, ticks = ( [0.25, 0.75], ["Fail", "Pass"] ))

    # 2. Lift Coefficient (CL) -> Col 3 & 4
    ax_CL = Axis(fig[1, 3], title = "Lift Coefficient (CL)", xlabel = "Cut Angle [°]", ylabel = "Opening [%]")
    hm_CL = heatmap!(ax_CL, unique_cuts, unique_opens, mat_CL, colormap = :viridis)
    Colorbar(fig[1, 4], hm_CL)

    # 3. Drag Coefficient (CD) -> Col 1 & 2
    ax_CD = Axis(fig[2, 1], title = "Drag Coefficient (CD)", xlabel = "Cut Angle [°]", ylabel = "Opening [%]")
    hm_CD = heatmap!(ax_CD, unique_cuts, unique_opens, mat_CD, colormap = :plasma)
    Colorbar(fig[2, 2], hm_CD) 

    # 4. Lift-to-Drag Ratio (L/D) -> Col 3 & 4
    ax_LD = Axis(fig[2, 3], title = "L/D Ratio", xlabel = "Cut Angle [°]", ylabel = "Opening [%]")
    hm_LD = heatmap!(ax_LD, unique_cuts, unique_opens, mat_LD, colormap = :turbo)
    Colorbar(fig[2, 4], hm_LD)

    # Save the figure
    filename = joinpath(out_dir, "Heatmap_AoA_1_$(aoa)deg.png")
    save(filename, fig)
    println("Saved heatmap: $filename")
end

println("\nAll AoA heatmaps generated successfully!")