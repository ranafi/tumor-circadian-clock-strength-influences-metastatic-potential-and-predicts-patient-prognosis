using Random, Statistics, CSV, DataFrames, PyPlot, MultivariateStats, Distributions, StatsBase

project_path = joinpath(homedir(), "Library/CloudStorage/Box-Box/PhD_Jan_Hammarlund/Specific Aim 1/Data")
BA11_path = joinpath(project_path, "Annotated_Unlogged_BA11Data.csv")

BA11 = CSV.read(BA11_path, DataFrame)
BA11_matrix = Matrix(BA11[:,2:end])

BA11_fit_path = "/Users/janhammarlund/Library/CloudStorage/Box-Box/PhD_Jan_Hammarlund/Specific Aim 1/Individual_Training/Original_Training_Run_Parameters_Rerun_05_05_2023/2023-05-05T12_58_00_align_p_cutoff_0_1_eigen_contr_var_0_03_eigen_max_30_seed_max_CV_0_9_train_collection_time_balance_0_5_train_collection_times_false/Fits/Fit_Output_2023-05-05T12_58_00.csv"

BA11_fit = CSV.read(BA11_fit_path, DataFrame)

BA11_cosine_fit_path = "/Users/janhammarlund/Library/CloudStorage/Box-Box/PhD_Jan_Hammarlund/Specific Aim 1/Individual_Training/Original_Training_Run_Parameters_Rerun_05_05_2023/2023-05-05T12_58_00_align_p_cutoff_0_1_eigen_contr_var_0_03_eigen_max_30_seed_max_CV_0_9_train_collection_time_balance_0_5_train_collection_times_false/Fits/Genes_of_Interest_Aligned_Cosine_Fit_2023-05-05T12_58_00.csv"

BA11_cosine_fit = CSV.read(BA11_cosine_fit_path, DataFrame)

#####################################
# PDF Function for Plot Comparisons #
#####################################
# Probability density function: y = 1/(sigma*sqrt(2pi))*exp((-1/2*((x-mu)/sigma)^2))
function make_pdf_line(sigma, mu, x_min, x_max; n_xs=1000)
    xs = LinRange(x_min,x_max,n_xs)
    ys = (1 ./(sigma .*sqrt(2pi))) .*exp.((-1/2 .*((xs .- mu) ./sigma) .^2))
    return xs, ys
end

#########################
# Cycling Genes in BA11 #
#########################
# Using BHQ corrected P-Value
p_value_cutoff = 0.05
amplitude_ratio_cutoff = 0.25

BA11_expression = BA11[3:end, 4:end] |> Matrix
BA11_expression = map!(x -> typeof(x) <: AbstractString ? parse(Float32, x) : x, BA11_expression, BA11_expression) |> Array{Float32,2}
BA11_mean_expression = mapslices(mean, BA11_expression, dims = 2)
BA11_relative_amplitude = BA11_cosine_fit[:, :Amplitude] ./ BA11_mean_expression

BA11_cycling_genes_logical = (BA11_cosine_fit[:, :P_Statistic] .< p_value_cutoff)
BA11_non_cycling_genes_logical = .!(BA11_cycling_genes_logical)
total_BA11_cyc_genes = sum(BA11_cycling_genes_logical) # Total number of cycling genes

#####################################################
# Amplitudes and Means of Overlapping Cycling Genes #
#####################################################
# Fit Averages of Cycling Genes
BA11_means = BA11_cosine_fit[BA11_cycling_genes_logical,:Fit_Average]

# Amplitudes of Cycling Genes
BA11_amps = BA11_cosine_fit[BA11_cycling_genes_logical,:Amplitude]

###########################################
# Variance and Means of Non-Cycling Genes #
###########################################
# Means of Non-Cycling Genes
BA11_mean = mean(Matrix(BA11_expression[BA11_non_cycling_genes_logical,2:end]),dims=2)

# Variance of Non-Cycling Genes
BA11_variances = var(Matrix(BA11_expression[BA11_non_cycling_genes_logical,2:end]),dims=2)
NON_CYCLING_GENE_INFO = DataFrame(Gene_Symbols=BA11[vcat(false,false,BA11_non_cycling_genes_logical),2],Non_Cycling_Gene_Means=BA11_mean[:,1], Non_Cycling_Gene_Variances=BA11_variances[:,1])
NON_CYCLING_GENE_INFO_PATH = "/Users/janhammarlund/Library/CloudStorage/Box-Box/PhD_Jan_Hammarlund/Specific Aim 2/PaperFigures/Supplement/benchmarking/BA11_Synthetic_Data_From_Cycling_Parameters/BA11_NON_CYCLING_GENE_INFO.csv"
CSV.write(NON_CYCLING_GENE_INFO_PATH, NON_CYCLING_GENE_INFO)

###############################################
# Get Residuals about cosine line of best fit #
###############################################
# Lines of Best fit for cycling genes
BA11_lobf = BA11_cosine_fit[BA11_cycling_genes_logical,:Amplitude] .* cos.(BA11_fit[:,:Phases_MA]' .- BA11_cosine_fit[BA11_cycling_genes_logical,:Acrophase]) .+ BA11_cosine_fit[BA11_cycling_genes_logical,:Fit_Average]
BA11_cg_expression = BA11_expression[BA11_cycling_genes_logical,:]
BA11_cg_residuals = BA11_cg_expression .- BA11_lobf

#############################
# Get Variance of Residuals #
#############################
# Variance of Residuals of Cycling genes
BA11_cg_res_var = var(BA11_cg_residuals, dims=2)
CYCLING_GENE_INFO = hcat(DataFrame(Gene_Symbols=BA11[vcat(false, false, BA11_cycling_genes_logical), 2], Cycling_Gene_Variances=BA11_cg_res_var[:,1]), BA11_cosine_fit[BA11_cycling_genes_logical,[:Amplitude, :Acrophase, :Fit_Average]])
CYCLING_GENE_INFO_PATH = "/Users/janhammarlund/Library/CloudStorage/Box-Box/PhD_Jan_Hammarlund/Specific Aim 2/PaperFigures/Supplement/benchmarking/BA11_Synthetic_Data_From_Cycling_Parameters/BA11_CYCLING_GENE_INFO.csv"
CSV.write(CYCLING_GENE_INFO_PATH, CYCLING_GENE_INFO)
