using Random, Statistics, CSV, DataFrames, PyPlot, MultivariateStats, Distributions, StatsBase

project_path = joinpath(homedir(), "Library/CloudStorage/Box-Box/PhD_Jan_Hammarlund/Specific Aim 1/Data")
GRNG_path = joinpath(project_path, "GRNG.csv")
Laval_path = joinpath(project_path, "Laval.csv")
Seed_path = joinpath(project_path, "LungCyclerHomologues.csv")

GRNG = CSV.read(GRNG_path, DataFrame)
Laval = CSV.read(Laval_path, DataFrame)
seed = CSV.read(Seed_path, DataFrame).x

GRNG_matrix = Matrix(GRNG)
GRNG_gene_means = mapslices(GRNG_matrix[:,2:end], dims=2) do each_gene
    mean(each_gene)
end

GRNG_sorted_gene_means = abs.(sort(-GRNG_gene_means, dims=1))
GRNG_seed_gene_mean_cutoff = GRNG_sorted_gene_means[10000]

Laval_matrix = Matrix(Laval)
Laval_gene_means = mapslices(Laval_matrix[:,2:end], dims=2) do each_gene
    mean(each_gene)
end

Laval_sorted_gene_means = abs.(sort(-Laval_gene_means, dims=1))
Laval_seed_gene_mean_cutoff = Laval_sorted_gene_means[10000]

seed_index = vcat([findall(in([x]), GRNG[:,1]) for x in seed]...)
# Laval_seed_index = vcat([findall(in([x]), Laval[:,1]) for x in seed]...) # seed_index == Laval_seed_index # true

GRNG_seed = GRNG[seed_index,:]
Laval_seed = Laval[seed_index, :]

GRNG_seed_matrix = Matrix(GRNG_seed)
Laval_seed_matrix = Matrix(Laval_seed)

GRNG_seed_gene_means = mapslices(GRNG_seed_matrix[:,2:end], dims=2) do each_seed_gene
    mean(each_seed_gene)
end

GRNG_seed_gene_std = mapslices(GRNG_seed_matrix[:,2:end], dims=2) do each_seed_gene
    std(each_seed_gene)
end

Laval_seed_gene_means = mapslices(Laval_seed_matrix[:,2:end], dims=2) do each_seed_gene
    mean(each_seed_gene)
end

Laval_seed_gene_std = mapslices(Laval_seed_matrix[:,2:end], dims=2) do each_seed_gene
    std(each_seed_gene)
end

GRNG_seed_genes_over_min = GRNG_seed_gene_means .> GRNG_seed_gene_mean_cutoff
Laval_seed_genes_over_min = Laval_seed_gene_means .> Laval_seed_gene_mean_cutoff

GL_seed_genes_over_min = GRNG_seed_genes_over_min .& Laval_seed_genes_over_min
comparable_seed_genes = GRNG_seed[GL_seed_genes_over_min[:,1],1]

GRNG_seed = GRNG_seed[GRNG_seed_genes_over_min[:,1],:]
Laval_seed = Laval_seed[Laval_seed_genes_over_min[:,1],:]

# CSV.write(joinpath(project_path, "GRNG_seed_data.csv"), GRNG_seed)
# CSV.write(joinpath(project_path, "Laval_seed_data.csv"), Laval_seed)




GRNG_first_values = GRNG[:,2]
Laval_first_values = Laval[:,2]

GRNG_seed_first_values = GRNG_seed[:,2]
Laval_seed_first_values = Laval_seed[:,2]

GRNG_sg_original_row = map(GRNG_seed_first_values) do ii
    findall(in([ii]), GRNG_first_values)[1]
end

Laval_sg_original_row = map(Laval_seed_first_values) do ii
    findall(in([ii]), Laval_first_values)[1]
end

comparable_sg = intersect(GRNG_sg_original_row, Laval_sg_original_row)

GRNG_sg_sg_index = map(comparable_sg) do find_sg
    findall(in([find_sg]), GRNG_sg_original_row)[1]
end

Laval_sg_sg_index = map(comparable_sg) do find_sg
    findall(in([find_sg]), Laval_sg_original_row)[1]
end

GRNG_comparable_seed = GRNG_seed[GRNG_sg_sg_index,:]
Laval_comparable_seed = Laval_seed[Laval_sg_sg_index,:]

GRNG_comparable_seed_means = mapslices(Matrix(GRNG_comparable_seed[:,2:end]), dims=2) do each_seed_gene
    mean(each_seed_gene)
end

Laval_comparable_seed_means = mapslices(Matrix(Laval_comparable_seed[:,2:end]), dims=2) do each_seed_gene
    mean(each_seed_gene)
end

GRNG_comparable_seed_std = mapslices(Matrix(GRNG_comparable_seed[:,2:end]), dims=2) do each_seed_gene
    std(each_seed_gene)
end

Laval_comparable_seed_std = mapslices(Matrix(Laval_comparable_seed[:,2:end]), dims=2) do each_seed_gene
    std(each_seed_gene)
end

GRNG_Laval_mean_ratio = GRNG_comparable_seed_means./Laval_comparable_seed_means
mean_GRNG_Laval_mean_ratio = median(GRNG_Laval_mean_ratio)
# On average, the seed gene means in GRNG are 1.12 times the seed gene means of Laval
mean_ratio_log_normal = fit(LogNormal, GRNG_Laval_mean_ratio)
# mu = 0.128, sigma=0.177
refit_mean_ratio_dist = rand(mean_ratio_log_normal, 1149)

GRNG_Laval_std_ratio = GRNG_comparable_seed_std./Laval_comparable_seed_std
mean_GRNG_Laval_std_ratio = median(GRNG_Laval_std_ratio)
# On average, the seed gene standard deviations in GRNG are 1.6 times the seed gene standard deviations of Laval
std_ratio_log_normal = fit(LogNormal, GRNG_Laval_std_ratio)
# mu = 0.5, sigma=0.21
refit_std_ratio_dist = rand(std_ratio_log_normal, 1149)

# Relationship between
cor(GRNG_Laval_mean_ratio, GRNG_Laval_std_ratio)
cov(GRNG_Laval_mean_ratio, GRNG_Laval_std_ratio)
mean_std_lin_coeff = llsq(GRNG_Laval_std_ratio, GRNG_Laval_mean_ratio) # The linear relationship of GRNG_Laval_mean_ratio as a function of GRNG_Laval_std_ratio
# mu = 0.31, sigma = 0.633

lin_mean_ratio = GRNG_Laval_std_ratio .* mean_std_lin_coeff[1] .+ mean_std_lin_coeff[2] # Line of best fit values values of GRNG_Laval_mean_ratio for GRNG_Laval_std_ratio
refit_lin_mean_ratio = refit_std_ratio_dist .* mean_std_lin_coeff[1] .+ mean_std_lin_coeff[2] # Estimated mean ratios based on the distribution of estimated std ratios and the linear correlation as seen in real data

#################################################################################################################################################################
# Plot the distributions of the real and estimated mean ratios and std ratios, and the line of best fit for the real mean ratio as a function of real std ratio #
#################################################################################################################################################################
fig=figure(figsize=(20,10))
subplot(2,3,1)
hist(GRNG_Laval_mean_ratio, bins=LinRange(0,4,20), alpha = 0.5, label = "Real")
hist(refit_mean_ratio_dist, bins=LinRange(0,4,20), alpha = 0.5, label = "Estimated")
legend(loc=1)
xlabel("GRNG mean over Laval mean")
ylabel("Count")
title("Offset\nEstimates ~ LogNorm(μ=0.128, σ=0.177)")
subplot(2,3,4)
# Change CDF Plots to Q-Q Plots
scatter(GRNG_Laval_mean_ratio, map(x -> sum(GRNG_Laval_mean_ratio.<=x)/length(GRNG_Laval_mean_ratio), GRNG_Laval_mean_ratio), label="Real", alpha=0.5)
scatter(refit_mean_ratio_dist, map(x -> sum(refit_mean_ratio_dist.<=x)/length(refit_mean_ratio_dist), refit_mean_ratio_dist), label="Estimated", alpha=0.5)
legend(loc=4)
ylabel("Cumulative Probability") # <-- Becomes "Estimated Quantiles"
xlabel("Mean Ratio") # <-- Becomes "Real Quantiles"
title("Offset CDF") # <-- Becomes "Mean Ratio Q-Q"
subplot(2,3,2)
hist(GRNG_Laval_std_ratio, bins=LinRange(0,4,20), alpha = 0.5, label = "Real")
hist(refit_std_ratio_dist, bins=LinRange(0,4,20), alpha = 0.5, label = "Estimated")
legend(loc=1)
xlabel("GRNG std over Laval std")
ylabel("Count")
title("Scale\nEstimates ~ LogNorm(μ=0.5, σ=0.21)")
subplot(2,3,5)
# Change CDF Plots to Q-Q Plos
scatter(GRNG_Laval_std_ratio, map(x -> sum(GRNG_Laval_std_ratio.<=x)/length(GRNG_Laval_std_ratio), GRNG_Laval_std_ratio), label="Real", alpha = 0.5)
scatter(refit_std_ratio_dist, map(x -> sum(refit_std_ratio_dist.<=x)/length(refit_std_ratio_dist), refit_std_ratio_dist), label="Estimated", alpha = 0.5)
legend(loc=4)
ylabel("Cumulative Probability") # <-- Becomes "Estimated Quantiles"
xlabel("STD Ratio") # <-- Becomes "Real Quantiles"
title("Scale CDF") # <-- Becomes "STD Ratio Q-Q"
subplot(2,3,3)
scatter(GRNG_Laval_std_ratio, GRNG_Laval_mean_ratio, alpha = 0.5, label = "Real")
plot(GRNG_Laval_std_ratio, lin_mean_ratio, alpha = 0.9, color="red", label = "line of best fit")
xlabel("Scale")
ylabel("Offset")
title("From GRNG and Laval\nOffset=0.31*Scale+0.63")
legend(loc=4)
subplot(2,3,6)
scatter(refit_std_ratio_dist, refit_mean_ratio_dist, alpha = 0.5, label = "Estimated", c="orange")
xlabel("Scale")
ylabel("Offset")
title("From Estimates")
legend(loc=4)
suptitle("Generating Random Scale and Offset Terms From two Independent Log Normal Distributions")
subplots_adjust(left=0.04,right=0.99,bottom=0.05,hspace=0.23,top=0.92)
# As seen in the 3rd panel from the left, on the bottom (subplot(236)), our Estimated Mean and STD ratios do not have a linear correlation
# In the panel above (subplot(233)) we see that there is a linear correlation, with slope of 0.31 and y-intercept of 0.63
# Thus, we can try to estimate the distribution that best fits the residuals of this line of best fit
##############################################
# NOTE!!! SAVE ABOVE FIGURE FOR REFERENCE!!! #
##############################################

lin_mean_ratio_residual = GRNG_Laval_mean_ratio .- lin_mean_ratio # Line of best fit residuals
fig = figure(figsize=(10,10))
hist(lin_mean_ratio_residual, bins=LinRange(-1,2,30)) # Histogram of the residuals of about the line of best fit for Mean Ratio as a function of STD Ratio
# The histogram revelas that the distribution look somewhat log normal but it has negative values.
# We can subtract the smallest negative value and add 0.1 to all values to be able to estimate a log normal distribution

##########################################
# Fitting distributions to the Residuals #
##########################################
mean_ratio_residual_normal_fit = fit(Normal, lin_mean_ratio_residual) # The normal distribution that best describes the residuals of GRNG_Laval_mean_ratio as a function of GRNG_Laval_std_ratio
mean_ratio_residual_lognormal_fit = fit(LogNormal, lin_mean_ratio_residual .- minimum(lin_mean_ratio_residual) .+ 0.1) # The log normal distribution that best describes 0.1 plus the difference of the residuals of GRNG_Laval_mean_ratio as a function of GRNG_Laval_std_ratio and the minimum of GRNG_Laval_mean_ratio
mean_ratio_exp_residual_lognormal_fit = fit(LogNormal, exp.(lin_mean_ratio_residual)) # The log normal distribution that best describes the exponent of the residuals

mean_ratio_residual_normal_dist = rand(mean_ratio_residual_normal_fit, length(lin_mean_ratio_residual))
mean_ratio_residual_lognormal_dist = rand(mean_ratio_residual_lognormal_fit, length(lin_mean_ratio_residual)) .- 0.1 .+ minimum(lin_mean_ratio_residual)
mean_ratio_exp_residual_lognormal_dist = log.(rand(mean_ratio_exp_residual_lognormal_fit, length(lin_mean_ratio_residual)))


###################################################################
# NOTE!!! CHANGE CDF PLOTS TO Q-Q PLOTS AND SAVE AS REFERENCE !!! #
###################################################################
#=
fig = figure(figsize=(20,10))
subplot(2,3,1)
hist(lin_mean_ratio_residual, bins=LinRange(-1,2,30), alpha=0.5, label="Residual")
hist(mean_ratio_residual_normal_dist, bins=LinRange(-1,2,30), alpha=0.5, label="Normal Distribution")
legend(loc=1)
subplot(2,3,2)
hist(lin_mean_ratio_residual, bins=LinRange(-1,2,30), alpha=0.5, label="Residual")
hist(mean_ratio_residual_lognormal_dist, bins=LinRange(-1,2,30), alpha=0.5, label="Log Normal Distribution")
legend(loc=1)
subplot(2,3,3)
hist(lin_mean_ratio_residual, bins=LinRange(-1,2,30), alpha=0.5, label="Residual")
hist(mean_ratio_exp_residual_lognormal_dist, bins=LinRange(-1,2,30), alpha=0.5, label="Log of Log Normal of Exp, Distribution")
legend(loc=1)
subplot(2,3,4)
scatter(lin_mean_ratio_residual, map(x -> sum(lin_mean_ratio_residual.<=x)/length(lin_mean_ratio_residual), lin_mean_ratio_residual), label="Residual", alpha=0.5)
scatter(mean_ratio_residual_normal_dist, map(x -> sum(mean_ratio_residual_normal_dist.<=x)/length(mean_ratio_residual_normal_dist), mean_ratio_residual_normal_dist), label="Estimated Residual", alpha=0.5)
legend(loc=4)
subplot(2,3,5)
scatter(lin_mean_ratio_residual, map(x -> sum(lin_mean_ratio_residual.<=x)/length(lin_mean_ratio_residual), lin_mean_ratio_residual), label="Residual", alpha=0.5)
scatter(mean_ratio_residual_lognormal_dist, map(x -> sum(mean_ratio_residual_lognormal_dist.<=x)/length(mean_ratio_residual_lognormal_dist), mean_ratio_residual_lognormal_dist), label="Estimated Residual", alpha = 0.5)
legend(loc=4)
subplot(2,3,6)
scatter(lin_mean_ratio_residual, map(x -> sum(lin_mean_ratio_residual.<=x)/length(lin_mean_ratio_residual), lin_mean_ratio_residual), label="Residual", alpha=0.5)
scatter(mean_ratio_exp_residual_lognormal_dist, map(x -> sum(mean_ratio_exp_residual_lognormal_dist.<=x)/length(mean_ratio_exp_residual_lognormal_dist), mean_ratio_exp_residual_lognormal_dist), label="Estimated Residual", alpha = 0.5)
legend(loc=4)
=#
# The Log normal distribution visually fits the residuals better, therefore I will generate residuals from a log normal distribution

#####################################################################################
# Linearly correlated mean ratios plus the log normal distribution of the residuals #
#####################################################################################
linearly_correlated_lin_mean_ratio = refit_lin_mean_ratio .+ mean_ratio_residual_lognormal_dist

fig = figure(figsize=(10,10))
hist(GRNG_Laval_mean_ratio, bins=LinRange(0,4,20), alpha=0.5, label="Mean Ratio")
hist(linearly_correlated_lin_mean_ratio, bins=LinRange(0,4,20), alpha=0.5, label="Estimated Mean Ratio")
# The estimated mean ratio is is based on the distribution of residuals of GRNG_Laval_mean_ratio as a function of GRNG_Laval_std_ratio
llsq(reshape(refit_std_ratio_dist,:,1), linearly_correlated_lin_mean_ratio)

######################################################################################
# Figure showing real and estimated mean ratios and std ratios, as well as Q-Q Plots #
######################################################################################
fig=figure(figsize=(30,10))
subplot(2,4,1)
# Scatter of Offset (y) as a function of Scale (x) from real data
scatter(GRNG_Laval_std_ratio, GRNG_Laval_mean_ratio, alpha = 0.5, label = "Real")
# Y=mX+B for mean ratio (y) as a function of scale ratio (x) from real data
plot(GRNG_Laval_std_ratio, lin_mean_ratio, alpha = 0.9, color="red", label = "line of best fit")
legend(loc=4)
xlabel("Scale Ratio")
ylabel("Offset Ratio")
title("From GRNG and Laval\nOffset=0.31*Scale + 0.63")
subplot(2,4,2)
# Histrogram of residuals from line of best fit explaining Offset (y) as a function of Scale (x) from real data
hist(lin_mean_ratio_residual, bins=LinRange(-1,2,30), alpha=0.5, label="Real")
# Histrogram of residuals from lin eof best fit explaining Offset (y) as a function of Scale (x) from estimated distributions
hist(mean_ratio_residual_lognormal_dist, bins=LinRange(-1,2,30), alpha=0.5, label="Estimated", color="orange")
legend(loc=1)
xlabel("Residual")
ylabel("Count")
title("Residuals")
subplot(2,4,3)
# Histrogram of the real scale ratios
hist(GRNG_Laval_std_ratio, bins=LinRange(0,4,20), alpha = 0.5, label = "Real")
# Histrogram of the estimated scale ratios
hist(refit_std_ratio_dist, bins=LinRange(0,4,20), alpha = 0.5, label = "Estimated", color="orange")
legend(loc=1)
xlabel("Scale Ratio")
ylabel("Count")
title("Scale Ratio\nEstimates ~ LogNormal(μ=0.50, σ=0.21)")
xlim(-0.1,4.1)
subplot(2,4,4)
# Histogram of the real mean ratios
hist(GRNG_Laval_mean_ratio, bins=LinRange(0,4,20), alpha = 0.5, label = "Real")
# Histrogram of the mean ratios generated from linear relationship between scale and offset and the distribution of residuals from real data
hist(linearly_correlated_lin_mean_ratio, bins=LinRange(0,4,20), alpha = 0.5, label = "Estimated", color = "orange")
legend(loc=1)
xlabel("Offset Ratio")
ylabel("Count")
title("Offset Ratio\nEstimates ~ Estimated Scale Ratio * 0.31\n+ LogNormal(μ=-0.33, σ=0.24)")
xlim(-0.1,4.1)
subplot(2,4,5)
# Scatter of Offset (y) as a function of Scale (x) from estimated distributions
scatter(refit_std_ratio_dist, linearly_correlated_lin_mean_ratio, alpha = 0.5, label = "Estimated", c="orange")
# Y=mX+B for mean ratio (y) as a function of scale ratio (x) from estimated distributions
plot(refit_std_ratio_dist, refit_lin_mean_ratio, alpha = 0.9, label = "line of best fit", c="g")
legend(loc=4)
xlabel("Scale Ratio")
ylabel("Offset Ratio")
title("From Estimates\nOffset=0.31*Scale + 0.63")
subplot(2,4,6)
# Q-Q plot of the real Residual distribution for reference
plot(sort(map(x -> sum(lin_mean_ratio_residual.<=x)/length(lin_mean_ratio_residual), lin_mean_ratio_residual),dims=1), sort(map(x -> sum(lin_mean_ratio_residual.<=x)/length(lin_mean_ratio_residual), lin_mean_ratio_residual),dims=1), label="Real Q-Q", alpha = 1)
# Q-Q plot of the estimated Residual distribution against the real
scatter(map(x -> sum(lin_mean_ratio_residual.<=x)/length(lin_mean_ratio_residual), lin_mean_ratio_residual), map(x -> sum(mean_ratio_residual_lognormal_dist.<=x)/length(mean_ratio_residual_lognormal_dist), lin_mean_ratio_residual), label="Estimated Q-Q", alpha = 0.5, color = "orange")
legend(loc=4)
xlabel("Real Quantiles")
ylabel("Estimated Quantiles")
title("Residual Q-Q")
subplot(2,4,7)
# Q-Q plot of the real Mean distribution for reference
plot(sort(map(x -> sum(GRNG_Laval_std_ratio.<=x)/length(GRNG_Laval_std_ratio), GRNG_Laval_std_ratio),dims=1), sort(map(x -> sum(GRNG_Laval_std_ratio.<=x)/length(GRNG_Laval_std_ratio), GRNG_Laval_std_ratio),dims=1), label="Real Q-Q", alpha = 1)
# Q-Q plot of the estimated Mean distribution against the real
scatter(map(x -> sum(GRNG_Laval_std_ratio.<=x)/length(GRNG_Laval_std_ratio), GRNG_Laval_std_ratio), map(x -> sum(refit_std_ratio_dist.<=x)/length(refit_std_ratio_dist), GRNG_Laval_std_ratio), label="Estimated Q-Q", alpha = 0.5, color = "orange")
legend(loc=4)
ylabel("Estimated Quantiles")
xlabel("Real Quantiles")
title("Scale Ratio Q-Q")
# xlim(-0.1,4.1)
subplot(2,4,8)
# Q-Q plot of the real Offset distribution for reference
offset_qq = sort(map(x -> sum(GRNG_Laval_mean_ratio.<=x)/length(GRNG_Laval_mean_ratio), GRNG_Laval_mean_ratio),dims=1)
plot(offset_qq, offset_qq, label="Real Q-Q", alpha=1)
# Q-Q plot of the estimated offset distribution against the real
scatter(map(x -> sum(GRNG_Laval_mean_ratio.<=x)/length(GRNG_Laval_mean_ratio), GRNG_Laval_mean_ratio), map(x -> sum(linearly_correlated_lin_mean_ratio.<=x)/length(linearly_correlated_lin_mean_ratio), GRNG_Laval_mean_ratio), label="Estimated Q-Q", alpha=0.5, c="orange")
legend(loc=4)
ylabel("Estimated Quantiles")
xlabel("Real Quantiles")
title("Offset Ratio Q-Q")
# xlim(-0.1,4.1)
suptitle("Generating Correlated Random Scale and Offset Ratios for Semi-Synthetic Data")
subplots_adjust(left=0.04,right=0.99,bottom=0.05,hspace=0.345,wspace=0.226,top=0.871)
savefig()
#####################################
# NOTE!!! SAVE PLOT AS REFERENCE!!! #
#####################################

#=
Estimated Scale factors ~ LogNormal(μ=0.5, σ=0.21)^n
And Estimated Offset factors ~ (Estimated Scale factors * 0.31 + LogNormal(μ=-0.33, σ=0.24) - 0.01)^n
where n is the multiplicative influence factor
If n = 1, the influence is the same as seen in GRNG-Laval
If n = 0.5, the influence is half that see in GRNG-Laval
If n = 2, the influence is twice that see in GRNG-Laval
=#
