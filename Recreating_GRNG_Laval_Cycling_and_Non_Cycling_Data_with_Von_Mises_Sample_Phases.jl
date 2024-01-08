using Random, Statistics, CSV, DataFrames, PyPlot, MultivariateStats, Distributions, StatsBase

# GRNG Non-Cycler Info
GRNG_NON_CYCLER_INFO = CSV.read(joinpath(homedir(), "Library/CloudStorage/Box-Box/PhD_Jan_Hammarlund/Specific Aim 1/Data/GRNG_Laval_Synthetic_Data_06_07_22/GRNG_NON_CYCLING_GENE_INFO.csv"), DataFrame);
#GRNG Cycler Info
GRNG_CYCLER_INFO = CSV.read(joinpath(homedir(), "Library/CloudStorage/Box-Box/PhD_Jan_Hammarlund/Specific Aim 1/Data/GRNG_Laval_Synthetic_Data_06_07_22/GRNG_CYCLING_GENE_INFO.csv"), DataFrame);

GRNG_Non_Cycling_Means_1 = GRNG_NON_CYCLER_INFO[:,:Non_Cycling_Gene_Means];
GRNG_Non_Cycling_Variances_2 = GRNG_NON_CYCLER_INFO[:,:Variances];
GRNG_Cycling_Means_3 = GRNG_CYCLER_INFO[:,:Fit_Average];
GRNG_Cycling_Amplitudes_4 = GRNG_CYCLER_INFO[:,:Amplitude];
GRNG_Cycling_Residual_Variances_5 = GRNG_CYCLER_INFO[:,:GRNG_Cycling_Gene_Variances];
########################################################################################################################################
Non_Cycling_Gene_Mean_Ratios_1 = exp.(rand(Normal(0.1578896692916177, 0.1549061022160739), length(GRNG_Non_Cycling_Means_1)));
Non_Cycling_Relative_Variance_Ratios_2 = exp.(rand(Normal(0.9575424194864084, 0.5284632214501275), length(GRNG_Non_Cycling_Variances_2)));
Cycling_Gene_Mean_Ratios_3 = exp.(rand(Normal(0.20939302657582615, 0.4311302064266684), length(GRNG_Cycling_Means_3)));
Cycling_Relative_Amplitude_Ratios_4 = exp.(rand(Normal(0.3368763449102559, 0.30626576237626574), length(GRNG_Cycling_Amplitudes_4)));
Cycling_Relative_Residual_Variance_Ratio_5 = exp.(rand(Normal(0.4415190280981823, 0.4863923309359499), length(GRNG_Cycling_Residual_Variances_5)));
########################################################################################################################################
Laval_Non_Cycling_Means_1 = GRNG_Non_Cycling_Means_1 ./ Non_Cycling_Gene_Mean_Ratios_1;
Laval_Non_Cycling_Variances_2 = GRNG_Non_Cycling_Means_1 ./ (Non_Cycling_Relative_Variance_Ratios_2 .* Non_Cycling_Gene_Mean_Ratios_1);
Laval_Cycling_Means_3 = GRNG_Cycling_Means_3 ./ Cycling_Gene_Mean_Ratios_3;
Laval_Cycling_Amplitude_4 = GRNG_Cycling_Amplitudes_4 ./ (Cycling_Relative_Amplitude_Ratios_4 .* Cycling_Gene_Mean_Ratios_3);
Laval_Cycling_Residuals_5 = GRNG_Cycling_Residual_Variances_5 ./ (Cycling_Relative_Residual_Variance_Ratio_5 .* Cycling_Relative_Amplitude_Ratios_4 .* Cycling_Gene_Mean_Ratios_3);
########################################################################################################################################
GRNG_non_cycling_gene_expression = vcat(map(GRNG_Non_Cycling_Means_1, GRNG_Non_Cycling_Variances_2) do X, Y
    sqrt(Y) .* randn(1, 300) .+ X
end...);
test_grng_non_cyc_means = mean(GRNG_non_cycling_gene_expression, dims=2); # log.(abs.(GRNG_Non_Cycling_Means_1 .- test_grng_non_cyc_means) ./ GRNG_Non_Cycling_Means_1)
test_grng_non_cyc_vars = var(GRNG_non_cycling_gene_expression, dims=2); # log.(abs.(GRNG_Non_Cycling_Variances_2 .- test_grng_non_cyc_vars) ./ GRNG_Non_Cycling_Variances_2)

Laval_Non_Cycling_Gene_Expressions = vcat(map(Laval_Non_Cycling_Means_1, Laval_Non_Cycling_Variances_2) do X, Y
    sqrt(Y) .* randn(1, 100) .+ X
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

uniform_sample_dist = 2pi*rand(300);

my_rand_sample_phases = [rand(100) for ii in 1:5];
uniform_sample_dists = [2pi*my_rand_sample_phases[ii] for ii in 1:5];
pi_over_four_kappa = [pi*my_rand_sample_phases[ii] .+ pi/2 for ii in 1:5];
pi_over_two_kappa = [5pi/6*my_rand_sample_phases[ii] .+ 7pi/12 for ii in 1:5];
three_pi_over_four_kappa = [2pi/3*my_rand_sample_phases[ii] .+ 2pi/3 for ii in 1:5];
pi_kappa = [pi/2*my_rand_sample_phases[ii] .+ 3pi/4 for ii in 1:5];

########################################################################################################################################
########################################################################################################################################
########################################################################################################################################
# Make Synthetic Data
# GRNG Cycling Data
GRNG_cycler_expression = vcat(map(GRNG_Cycling_Means_3, GRNG_Cycling_Amplitudes_4, GRNG_CYCLER_INFO[:,:Acrophase], GRNG_Cycling_Residual_Variances_5) do Gene_Mean, Gene_Amplitude, Gene_Acrophase, Gene_Variance
    Gene_Amplitude .* cos.(uniform_sample_dist' .- Gene_Acrophase) .+ Gene_Mean .+ (sqrt(Gene_Variance) .* randn(1, 300))
end...)

Random.seed!(12345)
gene_residuals = map(x -> randn(1,100), 1:3500);

# Laval Cycling Data
# Cycler through the 5 different collection biases
Laval_all_sample_dist_cycler_expression = map([uniform_sample_dists, pi_over_four_kappa, pi_over_two_kappa, three_pi_over_four_kappa, pi_kappa]) do sample_dist_group 
    # Then cycle through each version of each collection bias
    map(sample_dist_group) do sample_dist
        # Cycle through each gene's mean, amplitude, acrophase and residual variance and vertically concatenate
        vcat(map(Laval_Cycling_Means_3, Laval_Cycling_Amplitude_4, GRNG_CYCLER_INFO[:,:Acrophase], Laval_Cycling_Residuals_5, gene_residuals) do Gene_Mean, Gene_Amplitude, Gene_Acrophase, Gene_Variance, Gene_Residuals
            # A * cos(x - phi) + B + gamma
            Gene_Amplitude .* cos.(sample_dist' .- Gene_Acrophase) .+ Gene_Mean .+ (sqrt(Gene_Variance) .* Gene_Residuals)
        end...)
    end
end;
close(); scatter(Laval_all_sample_dist_cycler_expression[1][1][1,:],Laval_all_sample_dist_cycler_expression[5][1][1,:])
########################################################################################################################################
# Concatenate GRNG and Laval Data
grng_sample_ids = vcat(map(x -> join(["sample", x, "1"], "_"), 1:300));
laval_sample_ids = vcat(map(x -> join(["sample", x, "2"], "_"), 1:100));
sample_ids = vcat(grng_sample_ids, laval_sample_ids);
# sample_ids = vcat(map(y -> map(x -> join(["sample", x, y], "_"), 1:400), 1:2)...);
batch_ids = hcat(repeat(["Batch_1", "Batch_1", "Batch_1", "Batch_2"], inner=100)...);
GRNG_synthetic_expression = vcat(GRNG_non_cycling_gene_expression, GRNG_cycler_expression);
All_Gene_Symbols = vcat(GRNG_NON_CYCLER_INFO[:,:Gene_Symbols], GRNG_CYCLER_INFO[:,:Gene_Symbols]);
GRNG_Synthetic_Data = vcat(GRNG_non_cycling_gene_expression, GRNG_cycler_expression);

Laval_Synthetic_Data = map(Laval_all_sample_dist_cycler_expression) do sample_distribution_type
    map(sample_distribution_type) do distribution_version
        vcat(Laval_Non_Cycling_Gene_Expressions, distribution_version)
    end
end;

data_path = joinpath(homedir(), "Library/CloudStorage/Box-Box/PhD_Jan_Hammarlund/Specific Aim 1/Data/Fully_Synthetic_Data_05_03_2023");

map(Laval_Synthetic_Data, ["Uniform", "pi_over_four_kappa", "pi_over_two_kappa", "three_pi_over_four_kappa", "pi_kappa"]) do specific_time_distribution, distribution_name
    map(specific_time_distribution, ["V1", "V2", "V3", "V4", "V5"]) do distribution_version, version_name
        output_file = hcat(DataFrame(Gene_Symbols=vcat("site_D", All_Gene_Symbols)), DataFrame(vcat(batch_ids, hcat(GRNG_Synthetic_Data, distribution_version)), sample_ids))
        CSV.write(joinpath(data_path, distribution_name, join([distribution_name, version_name, "csv"], "_", ".")), output_file)
    end
end;

data_path_2 = joinpath(homedir(), "Library/CloudStorage/Box-Box/PhD_Jan_Hammarlund/Specific Aim 1/Data/Fully_Synthetic_Time_Bias_05_03_2023");

map([uniform_sample_dists, pi_over_four_kappa, pi_over_two_kappa, three_pi_over_four_kappa, pi_kappa], ["Uniform", "pi_over_four_kappa", "pi_over_two_kappa", "three_pi_over_four_kappa", "pi_kappa"]) do specific_time_distribution, distribution_name
    map(specific_time_distribution, ["V1", "V2", "V3", "V4", "V5"]) do distribution_version, version_name
        CSV.write(joinpath(data_path_2, distribution_name, join([distribution_name, version_name, "csv"], "_", ".")), DataFrame(hcat(uniform_sample_dist..., distribution_version...), sample_ids))
    end
end;
