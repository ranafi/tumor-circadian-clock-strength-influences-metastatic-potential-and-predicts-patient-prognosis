using Random, Statistics, CSV, DataFrames, PyPlot, MultivariateStats, Distributions, StatsBase

# BA11 Non-Cycler Info
BA11_NON_CYCLER_INFO = CSV.read("/Users/janhammarlund/Library/CloudStorage/Box-Box/PhD_Jan_Hammarlund/Specific Aim 2/PaperFigures/Supplement/benchmarking/BA11_Synthetic_Data_From_Cycling_Parameters/BA11_NON_CYCLING_GENE_INFO.csv", DataFrame);
# BA11 Cycler Info
BA11_CYCLER_INFO = CSV.read("/Users/janhammarlund/Library/CloudStorage/Box-Box/PhD_Jan_Hammarlund/Specific Aim 2/PaperFigures/Supplement/benchmarking/BA11_Synthetic_Data_From_Cycling_Parameters/BA11_CYCLING_GENE_INFO.csv", DataFrame);
# BA11 Expression Data
BA11_full_raw = CSV.read("/Users/janhammarlund/Library/CloudStorage/Box-Box/PhD_Jan_Hammarlund/Specific Aim 1/Data/Annotated_Unlogged_BA11Data.csv", DataFrame)
BA11 = BA11_full_raw |>
    x -> x[3:end,3:end] |>
    x -> rename!(x, vcat("Gene.Symbol", names(x)[2:end]));

BA11_expression = BA11[:,2:end] |>
    Matrix |>
    x -> map(y -> (typeof(y) <: AbstractString) ? parse(Float64, y) : y, x)

BA11_sample_times = BA11_full_raw[2, 4:end] |> 
    Vector |>
    x -> x[x .!= "NA"] |>
    x -> Float64.(x) |>
    x -> mod.(x ./ 12 .* pi, 2pi)

BA11_sample_time_ids = names(BA11_full_raw)[4:end] |>
    x -> x[Vector(BA11_full_raw[2, 4:end]) .!= "NA"]

BA11_Non_Cycling_Means_1 = BA11_NON_CYCLER_INFO[:,:Non_Cycling_Gene_Means];
BA11_Non_Cycling_Variances_2 = BA11_NON_CYCLER_INFO[:,:Non_Cycling_Gene_Variances];
BA11_Cycling_Means_3 = BA11_CYCLER_INFO[:,:Fit_Average];
BA11_Cycling_Amplitudes_4 = BA11_CYCLER_INFO[:,:Amplitude];
BA11_Cycling_Residual_Variances_5 = BA11_CYCLER_INFO[:,:Cycling_Gene_Variances];
########################################################################################################################################
Non_Cycling_Gene_Mean_Ratios_1 = exp.(rand(Normal(0.1578896692916177, 0.1549061022160739), length(BA11_Non_Cycling_Means_1)));
Non_Cycling_Relative_Variance_Ratios_2 = exp.(rand(Normal(0.9575424194864084, 0.5284632214501275), length(BA11_Non_Cycling_Variances_2)));
Cycling_Gene_Mean_Ratios_3 = exp.(rand(Normal(0.20939302657582615, 0.4311302064266684), length(BA11_Cycling_Means_3)));
Cycling_Relative_Amplitude_Ratios_4 = exp.(rand(Normal(0.3368763449102559, 0.30626576237626574), length(BA11_Cycling_Amplitudes_4)));
Cycling_Relative_Residual_Variance_Ratio_5 = exp.(rand(Normal(0.4415190280981823, 0.4863923309359499), length(BA11_Cycling_Residual_Variances_5)));
########################################################################################################################################
Laval_Non_Cycling_Means_1 = BA11_Non_Cycling_Means_1 ./ Non_Cycling_Gene_Mean_Ratios_1;
Laval_Non_Cycling_Variances_2 = BA11_Non_Cycling_Means_1 ./ (Non_Cycling_Relative_Variance_Ratios_2 .* Non_Cycling_Gene_Mean_Ratios_1);
Laval_Cycling_Means_3 = BA11_Cycling_Means_3 ./ Cycling_Gene_Mean_Ratios_3;
Laval_Cycling_Amplitude_4 = BA11_Cycling_Amplitudes_4 ./ (Cycling_Relative_Amplitude_Ratios_4 .* Cycling_Gene_Mean_Ratios_3);
Laval_Cycling_Residuals_5 = BA11_Cycling_Residual_Variances_5 ./ (Cycling_Relative_Residual_Variance_Ratio_5 .* Cycling_Relative_Amplitude_Ratios_4 .* Cycling_Gene_Mean_Ratios_3);
########################################################################################################################################
Laval_Non_Cycling_Gene_Expressions = vcat(map(Laval_Non_Cycling_Means_1, Laval_Non_Cycling_Variances_2) do X, Y
    sqrt(Y) .* randn(1, 101) .+ X
end...);
test_laval_non_cyc_means = mean(Laval_Non_Cycling_Gene_Expressions, dims=2); # log.(abs.(Laval_Non_Cycling_Means_1 .- test_laval_non_cyc_means) ./ Laval_Non_Cycling_Means_1)
test_laval_non_cyc_vars = var(Laval_Non_Cycling_Gene_Expressions, dims=2); # log.(abs.(Laval_Non_Cycling_Variances_2 .- test_laval_non_cyc_vars) ./ Laval_Non_Cycling_Variances_2)

########################################################################################################################################
########################################################################################################################################
########################################################################################################################################
# Get Sample Distributions

# 24 hours in 2pi
# 12 hours in pi
# 10 hours in 5pi/6 ==> (12/6-5/6)/2 ==> 7/12
# 8 hours in 2pi/3 ==> (12/6-4/6)/2 ==> 2/3 OR 4/6 OR 8/12
# 6 hours in pi/2

my_rand_sample_phases = [rand(101) for ii in 1:5];
uniform_sample_dists = [2pi*my_rand_sample_phases[ii] for ii in 1:5];
_12_hour_dists = [pi*my_rand_sample_phases[ii] .+ pi/2 for ii in 1:5];
_10_hour_dists = [5pi/6*my_rand_sample_phases[ii] .+ 7pi/12 for ii in 1:5];
_8_hour_dists = [2pi/3*my_rand_sample_phases[ii] .+ 2pi/3 for ii in 1:5];
_6_hour_dists = [pi/2*my_rand_sample_phases[ii] .+ 3pi/4 for ii in 1:5];

########################################################################################################################################
########################################################################################################################################
########################################################################################################################################
# Make Synthetic Data
Random.seed!(12345)

# Laval Cycling Data
# Cycle through the 5 different collection biases
Laval_all_sample_dist_cycler_expression = map([uniform_sample_dists, _12_hour_dists, _10_hour_dists, _8_hour_dists, _6_hour_dists]) do sample_dist_group 
    # Then cycle through each version of each collection bias
    map(sample_dist_group) do sample_dist
        # Cycle through each gene's mean, amplitude, acrophase and residual variance and vertically concatenate
        vcat(map(Laval_Cycling_Means_3, Laval_Cycling_Amplitude_4, BA11_CYCLER_INFO[:,:Acrophase], Laval_Cycling_Residuals_5) do Gene_Mean, Gene_Amplitude, Gene_Acrophase, Gene_Variance
            # A * cos(x - phi) + B + gamma
            Gene_Amplitude .* cos.(sample_dist' .- Gene_Acrophase) .+ Gene_Mean .+ (sqrt(Gene_Variance) .* randn(1,101))
        end...)
    end
end;
########################################################################################################################################
# Concatenate BA11 and Laval Data
BA11_sample_ids = names(BA11_full_raw)[4:end];
laval_sample_ids = vcat(map(x -> join(["sample", x, "Synthetic"], "_"), 1:101));
sample_ids = vcat(BA11_sample_ids, laval_sample_ids);
# sample_ids = vcat(map(y -> map(x -> join(["sample", x, y], "_"), 1:400), 1:2)...);
batch_ids = hcat(repeat(["Real_Batch", "Real_Batch", "Synthetic_Batch"], inner=101)...);
All_Gene_Symbols = vcat(BA11_NON_CYCLER_INFO[:,:Gene_Symbols], BA11_CYCLER_INFO[:,:Gene_Symbols]);

matching_gene_symbol_order = map(x -> findall(in([x]), All_Gene_Symbols), BA11[:,:("Gene.Symbol")]) |> unique |> x -> vcat(x...); # BA11[:,:("Gene.Symbol")] == All_Gene_Symbols[matching_gene_symbol_order] # This only works because duplicate gene symbols appear right after each other in the BA11 dataset.

Laval_Synthetic_Data = map(Laval_all_sample_dist_cycler_expression) do sample_distribution_type
    map(sample_distribution_type) do distribution_version
        vcat(Laval_Non_Cycling_Gene_Expressions, distribution_version)
    end
end;

data_path = joinpath(homedir(), "Library/CloudStorage/Box-Box/PhD_Jan_Hammarlund/Specific Aim 1/Data/BA11_with_Synthetic_BA11_Data_with_Windowed_Time_Bias_05_07_2023");

map(Laval_Synthetic_Data, ["uniform_window", "12_hour_window", "10_hour_window", "8_hour_window", "6_hour_window"]) do specific_time_distribution, distribution_name
    map(specific_time_distribution, ["V1", "V2", "V3", "V4", "V5"]) do distribution_version, version_name
        output_file = hcat(DataFrame(Gene_Symbols=vcat("site_D", All_Gene_Symbols[matching_gene_symbol_order])), DataFrame(vcat(batch_ids, hcat(BA11_expression, distribution_version)), sample_ids))
        CSV.write(joinpath(data_path, distribution_name, join([distribution_name, version_name, "csv"], "_", ".")), output_file)
    end
end;

data_path_2 = joinpath(homedir(), "Library/CloudStorage/Box-Box/PhD_Jan_Hammarlund/Specific Aim 1/Data/BA11_with_Synthetic_BA11_Windowed_Time_Bias_05_07_2023");

map([uniform_sample_dists, _12_hour_dists, _10_hour_dists, _8_hour_dists, _6_hour_dists], ["uniform_window", "12_hour_window", "10_hour_window", "8_hour_window", "6_hour_window"]) do specific_time_distribution, distribution_name
    map(specific_time_distribution, ["V1", "V2", "V3", "V4", "V5"]) do distribution_version, version_name
        CSV.write(joinpath(data_path_2, distribution_name, join([distribution_name, version_name, "csv"], "_", ".")), DataFrame(hcat(BA11_sample_times..., distribution_version...), vcat(BA11_sample_time_ids, laval_sample_ids)))
    end
end;
