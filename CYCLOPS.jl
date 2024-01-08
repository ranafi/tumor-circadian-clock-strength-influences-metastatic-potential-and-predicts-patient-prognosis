module CYCLOPS

export Covariates
export Fit
export GeneTracing # Currently named GOIExpressionTracing
export Acrophase # Currently named AcrophasePlot
export CosineFit # Currently named CYCLOPS_original_post_process_no_covariates and CYCLOPS_new_post_process_covariates

using DataFrames, Statistics, StatsBase, LinearAlgebra, MultivariateStats, Flux, PyPlot, Distributed, Random, CSV, Revise, Distributions, Dates, MultipleTesting
import Flux: onehot, onehotbatch, mse

const mouse_acrophases = [0, 0.0790637050481884, 0.151440116812406, 2.29555301890004, 2.90900605826091, 2.98706493493206, 2.99149022777511, 3.00769248308471, 3.1219769314524, 3.3058682224604, 3.31357155959037, 3.42557704861225, 3.50078722833753, 3.88658015146741, 4.99480367551318, 5.04951134876313, 6.00770260397838]
const mouse_gene_symbol = ["Arntl", "Clock", "Npas2", "Nr1d1", "Bhlhe41", "Nr1d2", "Dbp", "Ciart", "Per1", "Per3", "Tef", "Hlf", "Cry2", "Per2", "Cry1", "Rorc", "Nfil3"]
const human_homologue_gene_symbol = uppercase.(mouse_gene_symbol)
const subfolders = ["Plots", "Fits", "Models", "Parameters"]

#############
# Variables #
#############
function DefaultDict()
	theDefaultDictionary = Dict(:regex_cont => r".*_C",			# What is the regex match for continuous covariates in the data file
	:regex_disc => r".*_D",							# What is the regex match for discontinuous covariates in the data file

	:blunt_percent => 0.975, 						# What is the percentile cutoff below (lower) and above (upper) which values are capped

	:seed_min_CV => 0.14, 							# The minimum coefficient of variation a gene of interest may have to be included in eigen gene transformation
	:seed_max_CV => 0.7, 							# The maximum coefficient of a variation a gene of interest may have to be included in eigen gene transformation
	:seed_mth_Gene => 10000, 						# The minimum mean a gene of interest may have to be included in eigen gene transformation

	:norm_gene_level => true, 						# Does mean normalization occur at the seed gene level
	:norm_disc => false, 							# Does batch mean normalization occur at the seed gene level
	:norm_disc_cov => 1, 							# Which discontinuous covariate is used to mean normalize seed level data

	:eigen_reg => true, 							# Does regression again a covariate occur at the eigen gene level
	:eigen_reg_disc_cov => 1, 						# Which discontinous covariate is used for regression
	#~~~New Addition~~~#
	:eigen_reg_exclude => false,					# Are eigen genes with r squared greater than cutoff removed from final eigen data output
	#~~~~~~~~~~~~~~~~~~#
	:eigen_reg_r_squared_cutoff => 0.6,				# This cutoff is used to determine whether an eigen gene is excluded from final eigen data used for training
	:eigen_reg_remove_correct => false,				# Is the first eigen gene removed (true --> default) or it's contributed variance of the first eigne gene corrected by batch regression (false)

	:eigen_first_var => false, 						# Is a captured variance cutoff on the first eigen gene used
	:eigen_first_var_cutoff => 0.85, 				# Cutoff used on captured variance of first eigen gene

	:eigen_total_var => 0.85, 						# Minimum amount of variance required to be captured by included dimensions of eigen gene data
	:eigen_contr_var => 0.06, 						# Minimum amount of variance required to be captured by a single dimension of eigen gene data
	:eigen_var_override => false,					# Is the minimum amount of contributed variance ignored
	:eigen_max => 5, 								# Maximum number of dimensions allowed to be kept in eigen gene data

	:out_covariates => true, 						# Are covariates included in eigen gene data
	:out_use_disc_cov => true,						# Are discontinuous covariates included in eigen gene data
	:out_all_disc_cov => true, 						# Are all discontinuous covariates included if included in eigen gene data
	:out_disc_cov => 1,								# Which discontinuous covariates are included at the bottom of the eigen gene data, if not all discontinuous covariates
	:out_use_cont_cov => false,						# Are continuous covariates included in eigen data
	:out_all_cont_cov => true,						# Are all continuous covariates included in eigen gene data
	:out_use_norm_cont_cov => false,				# Are continuous covariates Normalized
	:out_all_norm_cont_cov => true,					# Are all continuous covariates normalized
	:out_cont_cov => 1,								# Which continuous covariates are included at the bottom of the eigen gene data, if not all continuous covariates, or which continuous covariates are normalized if not all
	:out_norm_cont_cov => 1,						# Which continuous covariates are normalized if not all continuous covariates are included, and only specific ones are included

	:init_scale_change => true,						# Are scales changed
	:init_scale_1 => false,							# Are all scales initialized such that the model sees them all as having scale 1
													# Or they'll be initilized halfway between 1 and their regression estimate.

	:train_n_models => 80, 							# How many models are being trained
	:train_μA => 0.001, 							# Learning rate of ADAM optimizer
	:train_β => (0.9, 0.999), 						# β parameter for ADAM optimizer
	#:train_optimizer => ADAM(0.0001, (0.9, 0.999)),	# optimizer to be used in training of model
	:train_min_steps => 1500, 						# Minimum number of training steps per model
	:train_max_steps => 2050, 						# Maximum number of training steps per model
	:train_μA_scale_lim => 1000, 					# Factor used to divide learning rate to establish smallest the learning rate may shrink to
	:train_circular => false,						# Train symmetrically
	:train_collection_times => true,						# Train using known times
	:train_collection_time_balance => 0.1,					# How is the true time loss rescaled
	# :train_sample_id => Array{Symbol, 1},
	# :train_sample_phase => Array{Number, 1},

	:cosine_shift_iterations => 192,				# How many different shifts are tried to find the ideal shift
	:cosine_covariate_offset => true,				# Are offsets calculated by covariates

	:align_p_cutoff => 0.05,						# When aligning the acrophases, what genes are included according to the specified p-cutoff
	:align_base => "radians",						# What is the base of the list (:align_acrophases or :align_phases)? "radians" or "hours"
	:align_disc => false,							# Is a discontinuous covariate used to align (true or false)
	:align_disc_cov => 1,							# Which discontinuous covariate is used to choose samples to separately align (is an integer)
	:align_other_covariates => false,				# Are other covariates included
	:align_batch_only => false,
	# :align_genes => Array{String, 1},				# A string array of genes used to align CYCLOPS fit output. Goes together with :align_acrophases
	# :align_acrophases => Array{<: Number, 1}, 	# A number array of acrophases for each gene used to align CYCLOPS fit output. Goes together with :align_genes
	# :align_samples => Array{T, 1},				# Where T is either Bool—the length of the array is the same as the total number of samples in the data set–or Int—referring to index. Goes together with :align_phases
	# :align_phases => Array{<: Number, 1}			# A number array of phases for each sample used to align CYCLOPS fit output. Goes together with :align_gsamples

	:X_Val_k => 10,									# How many folds used in cross validation.
	:X_Val_omit_size => 0.1,						# What is the fraction of samples left out per fold

	:plot_use_o_cov => true,
	:plot_correct_batches => false,
	:plot_disc => false,
	:plot_disc_cov => 1,
	:plot_separate => false,
	:plot_color => ["b", "orange", "g", "r", "m", "y", "k"],
	:plot_only_color => true,
	:plot_p_cutoff => 0.05)									
	return theDefaultDictionary
end

function DefaultDict(alternateDict::Dict{Symbol, Any}; display_changes::Bool = false)
	theDefaultDictionary = DefaultDict()
	for changes in keys(alternateDict)
		if haskey(theDefaultDictionary, changes)
			if theDefaultDictionary[changes] !== alternateDict[changes]
				show_change_made = display_changes ? " = $(theDefaultDictionary[changes]) for $(alternateDict[changes])" : ""
				println(""); @warn "Replacing existing key value: $(changes)$(show_change_made)."; println("");
			end
		else
			additional_keys = [:align_genes, :align_acrophases, :align_samples, :align_phases, :train_sample_id, :train_sample_phase]
			if !in(changes, additional_keys)
				println(""); throw(ArgumentError("$(changes) IS NOT A KEY KNOWN TO CYCLOPS. PLEASE REVISE OR REMOVE KEYS GIVEN IN alternateOps BEFORE PROCEEDING.")); println("");
			end
			show_change_made = display_changes ? " => $(alternateDict[changes])" : ""
			println(""); @warn "Adding :$(changes)$(show_change_made). Delete this key if you do not intend to use it for alignment."; println("");
		end
		theDefaultDictionary[changes] = alternateDict[changes]
	end

	return theDefaultDictionary
end
###########################
# CYCLOPS Layer Functions #
###########################
function CircularNode(x::Array{Float32, 1})
    length(x) == 2 || error("Invalid length of input that should be 2 but is $(length(x)).")
	sqrt(sum(x .^2)) != .0 || error("Both inputs to the circular node are 0.")
	sqrt(sum(x .^2)) != Inf || error("One or both of the inputs to the circular node are Inf.")
	!isnan(sqrt(sum(x .^2))) || error("One or both of the inputs to the circular node are NaN.")
    return x./sqrt(sum(x .^2)) # Convert x and y coordinates to points on a circle
end

function CircularNode(x::Array{Float64, 1})
    length(x) == 2 || error("Invalid length of input that should be 2 but is $(length(x)).")
	sqrt(sum(x .^2)) != .0 || error("Both inputs to the circular node are 0.")
	sqrt(sum(x .^2)) != Inf || error("One or both of the inputs to the circular node are Inf.")
	!isnan(sqrt(sum(x .^2))) || error("One or both of the inputs to the circular node are NaN.")
    return x./sqrt(sum(x .^2)) # Convert x and y coordinates to points on a circle
end

function CircularNode(x::Array{Float32, 2})
	return mapslices(y::Array{Float32, 1} -> CircularNode(y), x, dims = 1)[:]
end

function CircularNode(x::Array{Float64, 2})
	return mapslices(y::Array{Float64, 1} -> CircularNode(y), x, dims = 1)[:]
end

function CircularNode(x::Array{Array{Float32, 1},1})
	return map(y::Array{Float32, 1} -> CircularNode(y), x)
end

function CircularNode(x::Array{Array{Float64, 1},1})
	return map(y::Array{Float64, 1} -> CircularNode(y), x)
end

function CircularNode(x::Array{Array{Float32, 1},2})
	return map(y::Array{32, 1} -> CircularNode(y), x[:])
end

function CircularNode(x::Array{Array{Float64, 1},2})
	return map(y::Array{64, 1} -> CircularNode(y), x[:])
end

struct DenseFunction
	W
	b
end

function (m::DenseFunction)(x::Array{Float32,1})
	return m.W * x + m.b
end

function (m::DenseFunction)(x::Array{Float32,2})
	return mapslices(y::Array{Float32,1} -> m(y), x, dims=1)
end

function (m::DenseFunction)(x::Array{Array{Float32,1},1})
	return map(y::Array{Float32,1} -> m(y), x)
end

function (m::DenseFunction)(x::Array{Array{Float32,1},2})
	return map(y::Array{Float32,1} -> m(y), x[:])
end
#################
# CYCLOPS Types #
#################
struct Order
    L1  # Linear layer from # eigengenes to 2
    c   # Circ layer
    L2  # Linear layer from 2 to # eigengenes
    o   # output dimension used for oldcyclops
end

struct Covariates
	S_OH	# Scaling factor for OH (encoding). Trainable
	B		# Bias factor applied to all data. Trainable
	B_OH	# Bias factor for OH (encoding). Trainable
	L1  	# First linear layer (Dense). Reduced to at least 2 layers for the circ layer but can be reduced to only 3 to add one linear/non-linear layer. L1.W & L1.b are trainable
	# C   	# Circular layer (circ(x)). Non-trainable paremeter
	L2  	# Second linear layer (Dense). Takes output from circ and any additional linear layers and expands to number of eigengenes. L2.W & L2.b are trainable
	o   	# output dimensions (out). Non-trainable parameter
end

struct Covariates
	S_OH	
	B		
	B_OH	
	L1  	
	L2  	
	o   	
end

struct SimpleCovariates
	L1
	L1_OH
	C
	L2
	o
end
#####################
# CYCLOPS Functions #
#####################
function (m::Order)(x::Array{Float32,1})
    Fully_Connected_Encoding = m.L1(x[1:m.o]) # Function for linear (dense, or fully connected) input layer
    Circular_Layer = CircularNode(Fully_Connected_Encoding) # Circular bottleneck
    Fully_Connected_Decoding = m.L2(Circular_Layer) # Function for linear (dense, or fully connected) output layer
	return Fully_Connected_Decoding
end

function (m::Order)(x::Array{Float32,2})
	return mapslices(y::Array{Float32,1} -> [m(y)], x, dims=1)[:]
end

function (m::Order)(x::Array{Array{Float32,1},1})
	return map(y::Array{Float32,1} -> [m(y)], x)
end

function (m::Order)(x::Array{Array{Float32,1},2})
	return map(y::Array{Float32,1} -> [m(y)], x[:])
end

function OrderEncodingDense(x)
	return m.L1(x)
end
#=
function Order
end
=#
# Currently in Use for Single Module

function (m::Covariates)(x::Array{Float32,1})
    Encoding_Onehot = x[1:m.o] .* (1 .+ (m.S_OH * x[m.o + 1:end])) .+ (m.B_OH * x[m.o + 1:end]) .+ m.B
    Fully_Connected_Encoding = m.L1(Encoding_Onehot)
	Cicular_Layer = CircularNode(Fully_Connected_Encoding)
	Fully_Connected_Decoding = m.L2(Cicular_Layer)
	Decoding_Onehot = ((Fully_Connected_Decoding .- (m.B_OH * x[m.o + 1:end]) .- m.B) ./ (1 .+ (m.S_OH * x[m.o + 1:end])))
	return Decoding_Onehot
end

function (m::Covariates)(x::Array{Float32,2})
	return mapslices(y::Array{Float32,1} -> [m(y)], x, dims=1)[:]
end

function (m::Covariates)(x::Array{Array{Float32,1},1})
	return map(y::Array{Float32,1} -> [m(y)], x)
end

function (m::Covariates)(x::Array{Array{Float32,1},2})
	return map(y::Array{Float32,1} -> [m(y)], x[:])
end
#=
function(m::Covariates)(x)
	# local m
	# global m
	Encoding_Onehot = CovariatesEncodingOH(x)
	Encoding_Dense_Layer = CovariatesEncodingDense(Encoding_Onehot)
	Circular_Layer = CircularNode(Encoding_Dense_Layer)
	Decoding_Dense_Layer = CovariatesDecodingDense(Circular_Layer)
	Decoding_Onehot = CovariatesDecodingOH(x, Decoding_Dense_Layer)
	return Decoding_Onehot
end
=#
function CovariatesEncodingOH(x::Array{Float32,1}, m::Covariates)
	return x[1:m.o] .* (1 .+ (m.S_OH * x[m.o + 1:end])) .+ (m.B_OH * x[m.o + 1:end]) .+ m.B
end

function CovariatesEncodingOH(x::Array{Float32,2}, m::Covariates)
	return mapslices(y::Array{Float32,1} -> [CovariatesEncodingOH(y, m)], x, dims=1)[:]
end

function CovariatesEncodingOH(x::Array{Array{Float32,1},1}, m::Covariates)
	return map(y::Array{Float32,1} -> [CovariatesEncodingOH(y, m)], x)
end

function CovariatesEncodingOH(x::Array{Array{Float32,1},2}, m::Covariates)
	return map(y::Array{Float32,1} -> [CovariatesEncodingOH(y, m)], x[:])
end

function CovariatesDecodingOH(x::Array{Float32,1}, y::Array{Float32,1}, m::Covariates)
	return ((y .- (m.B_OH * x[m.o + 1:end]) .- m.B) ./ (1 .+ (m.S_OH * x[m.o + 1:end])))
end

function CovariatesDecodingOH(x::Array{Float32,2}, y::Array{Float32,2}, m::Covariates)
	return map(mapslices(x_v::Array{Float32,1} -> [x_v], x, dims=1)[:], mapslices(y_v::Array{Float32,1} -> [y_v], y, dims=1)[:]) do x_v_i, y_v_i; [CovariatesDecodingOH(x_v_i, y_v_i, m)] end
end

function CovariatesDecodingOH(x::Array{Array{Float32,1},1}, y::Array{Array{Float32,1},1}, m::Covariates)
	return map(x, y) do x_i::Array{Float32,1}, y_i::Array{Float32,1}; [CovariatesDecodingOH(x_i, y_i)] end
end

function CovariatesDecodingOH(x::Array{Array{Float32,1},1}, y::Array{Array{Float32,1},2}, m::Covariates)
	return map(x, y[:]) do x_i::Array{Float32,1}, y_i::Array{Float32,1}; [CovariatesDecodingOH(x_i, y_i)] end
end

function CovariatesDecodingOH(x::Array{Array{Float32,1},2}, y::Array{Array{Float32,1},2}, m::Covariates)
	return map(x[:], y[:]) do x_i::Array{Float32,1}, y_i::Array{Float32,1}; [CovariatesDecodingOH(x_i, y_i)] end
end

function CovariatesDecodingOH(x::Array{Array{Float32,1},2}, y::Array{Array{Float32,1},1}, m::Covariates)
	return map(x[:], y) do x_i::Array{Float32,1}, y_i::Array{Float32,1}; [CovariatesDecodingOH(x_i, y_i)] end
end

function CovariatesDecodingOH(x::Array{Float32,2}, y::Array{Array{Float32,1},2}, m::Covariates)
	return map(mapslices(z::Array{Float32,1} -> [z], x, dims=1)[:], y[:]) do x_i::Array{Float32,1}, y_i::Array{Float32,1}; [CovariatesDecodingOH(x_i, y_i, m)] end
end

function CovariatesDecodingOH(x::Array{Float32,2}, y::Array{Array{Float32,1},1}, m::Covariates)
	return map(mapslices(z::Array{Float32,1} -> [z], x, dims=1)[:], y) do x_i::Array{Float32,1}, y_i::Array{Float32,1}; [CovariatesDecodingOH(x_i, y_i, m)] end
end

function CovariatesDecodingOH(x::Array{Array{Float32,1},1}, y::Array{Float32,2}, m::Covariates)
	return map(x, mapslices(z::Array{Float32,1} -> [z], y, dims=1)[:]) do x_i::Array{Float32,1}, y_i::Array{Float32,1}; [CovariatesDecodingOH(x_i, y_i, m)] end
end

function CovariatesDecodingOH(x::Array{Array{Float32,1}, 2}, y::Array{Float32,2}, m::Covariates)
	return map(x[:], mapslices(z::Array{Float32,1} -> [z], y, dims=1)[:]) do x_i::Array{Float32,1}, y_i::Array{Float32,1}; [CovariatesDecodingOH(x_i, y_i, m)] end
end

function CovariatesEncodingDense(x::Array{Float32,1}, m::Covariates)
	return m.L1(x)
end

function CovariatesEncodingDense(x::Array{Float32,2}, m::Covariates)
	return mapslices(y::Array{Float32,1} -> [CovariatesEncodingDense(y, m)], x, dims=1)[:]
end

function CovariatesEncodingDense(x::Array{Array{Float32,1},2}, m::Covariates)
	return map(y::Array{Float32,1} -> [CovariatesEncodingDense(y, m)], x[:])
end

function CovariatesEncodingDense(x::Array{Array{Float32,1},1}, m::Covariates)
	return map(y::Array{Float32,1} -> [CovariatesEncodingDense(y, m)], x)
end

function CovariatesDecodingDense(x::Array{Float32,1}, m::Covariates)
	return m.L2(x)
end

function CovariatesDecodingDense(x::Array{Float32,2}, m::Covariates)
	return mapslices(y::Array{Float32,1} -> [CovariatesDecodingDense(y, m)], x, dims=1)[:]
end

function CovariatesDecodingDense(x::Array{Array{Float32,1},2}, m::Covariates)
	return map(y::Array{Float32,1} -> [CovariatesDecodingDense(y, m)], x[:])
end

function CovariatesDecodingDense(x::Array{Array{Float32,1},1}, m::Covariates)
	return map(y::Array{Float32,1} -> [CovariatesDecodingDense(y, m)], x)
end

function CovariatesCircularNode(x, m)
	return m.C(x)
end

function CovariatesPhase(x::Array{Float32,1}, m::Covariates)
	CircularOutput = CovariatesThroughCircularNode(x, m)
	return mod(atan(CircularOutput[2], CircularOutput[1]), 2pi)
end

function OrderPhase(x::Array{Float32,1}, m::Order)
	CircularOutput = OrderThroughCircularNode(x, m)
	return mod(atan(CircularOutput[2], CircularOutput[1]), 2pi)
end

function CovariatesPhase(x::Array{Float32,2}, m::Covariates)
	return mapslices(y::Array{Float32,1} -> CovariatesPhase(y, m), x, dims=1)[:]
end

function OrderPhase(x::Array{Float32,2}, m::Order)
	return mapslices(y::Array{Float32,1} -> OrderPhase(y, m), x, dims=1)[:]
end

function CovariatesPhase(x::Array{Array{Float32,1},1}, m::Covariates)
	return map(y::Array{Float32,1} -> CovariatesPhase(y, m), x)
end

function OrderPhase(x::Array{Array{Float32,1},1}, m::Order)
	return map(y::Array{Float32,1} -> OrderPhase(y, m), x)
end

function CovariatesPhase(x::Array{Array{Float32,1},2}, m::Covariates)
	return map(y::Array{Float32,1} -> CovariatesPhase(y, m), x[:])
end

function OrderPhase(x::Array{Array{Float32,1},2}, m::Order)
	return map(y::Array{Float32,1} -> OrderPhase(y, m), x[:])
end

function CovariatesThroughEncodingDense(x::Array{Float32,1}, m::Covariates)
	return CovariatesEncodingDense(CovariatesEncodingOH(x, m), m)
end

function OrderThroughEncodingDense(x::Array{Float32,1}, m::Order)
	return m.L1(x)
end

function CovariatesThroughEncodingDense(x::Array{Float32,2}, m::Covariates)
	return mapslices(y::Array{Float32,1} -> [CovariatesThroughEncodingDense(y, m)], x, dims=1)[:]
end

function OrderThroughEncodingDense(x::Array{Float32,2}, m::Order)
	return mapslices(y::Array{Float32,1} -> [OrderThroughEncodingDense(y, m)], x, dims=1)[:]
end

function CovariatesThroughEncodingDense(x::Array{Array{Float32,1},1}, m::Covariates)
	return map(y::Array{Float32,1} -> [CovariatesThroughEncodingDense(y, m)], x)
end

function OrderThroughEncodingDense(x::Array{Array{Float32,1},1}, m::Order)
	return map(y::Array{Float32,1} -> [OrderThroughEncodingDense(y, m)], x)
end

function CovariatesThroughEncodingDense(x::Array{Array{Float32,1},2}, m::Covariates)
	return map(y::Array{Float32,1} -> [CovariatesThroughEncodingDense(y, m)], x[:])
end

function OrderThroughEncodingDense(x::Array{Array{Float32,1},2}, m::Order)
	return map(y::Array{Float32,1} -> [OrderThroughEncodingDense(y, m)], x[:])
end

function CovariatesThroughCircularNode(x::Array{Float32,1}, m::Covariates)
	return CircularNode(CovariatesThroughEncodingDense(x, m))
end

function OrderThroughCircularNode(x::Array{Float32,1}, m::Order)
	return CircularNode(OrderThroughEncodingDense(x, m))
end

function CovariatesThroughCircularNode(x::Array{Float32,2}, m::Covariates)
	return mapslices(y::Array{Float32,1} -> [CovariatesThroughCircularNode(y, m)], x, dims=1)[:]
end

function OrderThroughCircularNode(x::Array{Float32,2}, m::Order)
	return mapslices(y::Array{Float32,1} -> [OrderThroughCircularNode(y, m)], x, dims=1)[:]
end

function CovariatesThroughCircularNode(x::Array{Array{Float32,1},1}, m::Covariates)
	return map(y::Array{Float32,1} -> [CovariatesThroughCircularNode(y, m)], x)
end

function OrderThroughCircularNode(x::Array{Array{Float32,1},1}, m::Order)
	return map(y::Array{Float32,1} -> [OrderThroughCircularNode(y, m)], x)
end

function CovariatesThroughCircularNode(x::Array{Array{Float32,1},2}, m::Covariates)
	return map(y::Array{Float32,1} -> [CovariatesThroughCircularNode(y, m)], x[:])
end

function OrderThroughCircularNode(x::Array{Array{Float32,1},2}, m::Order)
	return map(y::Array{Float32,1} -> [OrderThroughCircularNode(y, m)], x[:])
end

function CovariatesThroughDecodingDense(x::Array{Float32,1}, m::Covariates)
	return CovariatesDecodingDense(CovariatesThroughCircularNode(x, m), m)
end

function CovariatesThroughDecodingDense(x::Array{Float32,2}, m::Covariates)
	return mapslices(y::Array{Float32,1} -> [CovariatesThroughDecodingDense(y, m)], x, dims=1)[:]
end

function CovariatesThroughDecodingDense(x::Array{Array{Float32,1},1}, m::Covariates)
	return map(y::Array{Float32,1} -> [CovariatesThroughDecodingDense(y, m)], x)
end

function CovariatesThroughDecodingDense(x::Array{Array{Float32,1},2}, m::Covariates)
	return map(y::Array{Float32,1} -> [CovariatesThroughDecodingDense(y, m)], x[:])
end

function CovariatesSkipCircularNodeDecodingDense(x::Array{Float32,1}, m::Covariates)
	return CovariatesDecodingDense(CovariatesThroughEncodingDense(x, m), m)
end

function OrderSkipCircularNode(x::Array{Float32,1}, m::Order)
	return m.L2(m.L1(x))
end

function CovariatesSkipCircularNodeDecodingDense(x::Array{Float32,2}, m::Covariates)
	return mapslices(y::Array{Float32,1} -> [CovariatesSkipCircularNodeDecodingDense(y, m)], x, dims=1)[:]
end

function OrderSkipCircularNode(x::Array{Float32,2}, m::Order)
	return mapslices(y::Array{Float32,1} -> [OrderSkipCircularNode(y, m)], x, dims=1)[:]
end

function CovariatesSkipCircularNodeDecodingDense(x::Array{Array{Float32,1},1}, m::Covariates)
	return map(y::Array{Float32,1} -> [CovariatesSkipCircularNodeDecodingDense(y, m)], x)
end

function OrderSkipCircularNode(x::Array{Array{Float32,1},1}, m::Order)
	return map(y::Array{Float32,1} -> [OrderSkipCircularNode(y, m)], x)
end

function CovariatesSkipCircularNodeDecodingDense(x::Array{Array{Float32,1},2}, m::Covariates)
	return map(y::Array{Float32,1} -> [CovariatesSkipCircularNodeDecodingDense(y, m)], x[:])
end

function OrderSkipCircularNode(x::Array{Array{Float32,1},2}, m::Order)
	return map(y::Array{Float32,1} -> [OrderSkipCircularNode(y, m)], x[:])
end

function CovariatesSkipCircularNodeDecodingDenseMagnitude(x::Array{Float32,1}, m::Covariates)
	return sqrtsumofsquares(CovariatesSkipCircularNodeDecodingDense(x, m))
end

function OrderSkipCircularNodeMagnitude(x::Array{Float32,1}, m::Order)
	return sqrtsumofsquares(OrderSkipCircularNode(x, m))
end

function CovariatesSkipCircularNodeDecodingDenseMagnitude(x::Array{Float32,2}, m::Covariates)
	return mapslices(y::Array{Float32,1} -> [sqrtsumofsquares(CovariatesSkipCircularNodeDecodingDense(y, m))], x, dims=1)[:]
end

function OrderSkipCircularNodeMagnitude(x::Array{Float32,2}, m::Order)
	return mapslices(y::Array{Float32,1} -> [sqrtsumofsquares(OrderSkipCircularNode(y, m))], x, dims=1)[:]
end

function CovariatesSkipCircularNodeDecodingDenseMagnitude(x::Array{Array{Float32,1},1}, m::Covariates)
	return map(y::Array{Float32,1} -> [sqrtsumofsquares(CovariatesSkipCircularNodeDecodingDense(y, m))], x)
end

function OrderSkipCircularNodeMagnitude(x::Array{Array{Float32,1},1}, m::Order)
	return map(y::Array{Float32,1} -> [sqrtsumofsquares(OrderSkipCircularNode(y, m))], x)
end

function CovariatesSkipCircularNodeDecodingDenseMagnitude(x::Array{Array{Float32,1},2}, m::Covariates)
	return map(y::Array{Float32,1} -> [sqrtsumofsquares(CovariatesSkipCircularNodeDecodingDense(y, m))], x[:])
end

function OrderSkipCircularNodeMagnitude(x::Array{Array{Float32,1},2}, m::Order)
	return map(y::Array{Float32,1} -> [sqrtsumofsquares(OrderSkipCircularNode(y, m))], x[:])
end

function CovariatesSkipCircularNodeDecodingOH(x::Array{Float32,1}, m::Covariates)
	return CovariatesDecodingOH(x, CovariatesSkipCircularNodeDecodingDense(x, m), m)
end

function CovariatesSkipCircularNodeDecodingOH(x::Array{Float32,2}, m::Covariates)
	return CovariatesDecodingOH(x, CovariatesSkipCircularNodeDecodingDense(x, m), m)
end

function CovariatesSkipCircularNodeDecodingOH(x::Array{Array{Float32,1},1}, m::Covariates)
	return CovariatesDecodingOH(x, CovariatesSkipCircularNodeDecodingDense(x, m), m)
end

function CovariatesSkipCircularNodeDecodingOH(x::Array{Array{Float32,1},2}, m::Covariates)
	return CovariatesDecodingOH(x, CovariatesSkipCircularNodeDecodingDense(x, m), m)
end

function CovariatesSkipCircularNodeDecodingOHMagnitude(x::Array{Float32,1}, m::Covariates)
	return sqrtsumofsquares(CovariatesSkipCircularNodeDecodingOH(x, m))
end

function CovariatesSkipCircularNodeDecodingOHMagnitude(x::Array{Float32,2}, m::Covariates)
	return sqrtsumofsquares(vcat(CovariatesSkipCircularNodeDecodingOH(x, m)...))
end

function CovariatesSkipCircularNodeDecodingOHMagnitude(x::Array{Array{Float32,1},1}, m::Covariates)
	return sqrtsumofsquares(CovariatesSkipCircularNodeDecodingOH(x, m))
end

function CovariatesSkipCircularNodeDecodingOHMagnitude(x::Array{Array{Float32,1},2}, m::Covariates)
	return sqrtsumofsquares(CovariatesSkipCircularNodeDecodingOH(x, m))
end

function sqrtsumofsquares(x::Array{Float32,1})
	return sqrt(sum(x .^ 2))
end

function sqrtsumofsquares(x::Array{Float32,2})
	return mapslices(y::Array{Float32,1} -> sqrtsumofsquares(y), x, dims=1)[:]
end

function sqrtsumofsquares(x::Array{Array{Float32,1},1})
	return map(y::Array{Float32,1} -> sqrtsumofsquares(y), x)
end

function sqrtsumofsquares(x::Array{Array{Float32,},2})
	return map(y::Array{Float32,1} -> sqrtsumofsquares(y), x[:])
end

function sqrtsumofsquares(x::Array{Float32,2}, m::Covariates)
	return sqrtsumofsquares(m(x))
end

function sqrtsumofsquares(x::Array{Float32,2}, m::Order)
	return sqrtsumofsquares(m(x))
end

function sqrtsumofsquares(x::Array{Array{Float32,1},1}, m::Covariates)
	return sqrtsumofsquares(m(x))
end

function sqrtsumofsquares(x::Array{Array{Float32,1},1}, m::Order)
	return sqrtsumofsquares(m(x))
end

function sqrtsumofsquares(x::Array{Array{Float32,1},2}, m::Covariates)
	return sqrtsumofsquares(m(x))
end

function sqrtsumofsquares(x::Array{Array{Float32,1},2}, m::Order)
	return sqrtsumofsquares(m(x))
end

function CovariatesEncodingDenseMagnitude(x::Array{Float32,1}, m::Covariates)
	return sqrtsumofsquares(CovariatesEncodingOH(x, m))
end

function CovariatesEncodingDenseMagnitude(x::Array{Float32,2}, m::Covariates)
	return mapslices(y::Array{Float32,1} -> CovariatesEncodingDenseMagnitude(y, m), x, dims=1)[:]
end

function CovariatesEncodingDenseMagnitude(x::Array{Array{Float32,1},1}, m::Covariates)
	return mapslices(y::Array{Float32,1} -> CovariatesEncodingDenseMagnitude(y, m), x, dims=1)[:]
end

function CovariatesEncodingDenseMagnitude(x::Array{Float32,2}, m::Covariates)
	return mapslices(y::Array{Float32,1} -> CovariatesEncodingDenseMagnitude(y, m), x, dims=1)[:]
end

function CovariatesDecodingDenseMagnitude(x::Array{Float32,1}, m::Covariates)
	return sqrtsumofsquares(CovariatesThroughDecodingDense(x, m))
end

function CovariatesDecodingDenseMagnitude(x::Array{Float32,2}, m::Covariates)
	return mapslices(y::Array{Float32,1} -> CovariatesDecodingDenseMagnitude(y, m), x, dims=1)[:]
end

function get_inner_mse(the_eigen_data::Array{Float32,2}, the_model::Covariates)
    return mapslices(x::Array{Float32,1} -> get_inner_mse(x, the_model), the_eigen_data, dims=1)[:]
end

function get_inner_mse(the_eigen_data::Array{Array{Float32,1},1}, the_model::Covariates)
    return map(x::Array{Float32,1} -> get_inner_mse(x, the_model), the_eigen_data)
end

function get_inner_mse(the_eigen_data::Array{Array{Float32,1},2}, the_model::Covariates)
    return map(x::Array{Float32,1} -> get_inner_mse(x, the_model), the_eigen_data[:])
end

function get_inner_mse(sample_eigen_expression::Array{Float32,1}, the_model::Covariates)
    return mse(CovariatesEncodingOH(sample_eigen_expression, the_model), CovariatesThroughDecodingDense(sample_eigen_expression, the_model))
end

function get_skip_circle_inner_mse(sample_eigen_expression::Array{Float32,2}, the_model::Covariates)
	return mapslices(x::Array{Float32,1} -> get_skip_circle_inner_mse(x, the_model), sample_eigen_expression, dims=1)[:]
end

function get_skip_circle_inner_mse(sample_eigen_expression::Array{Array{Float32,1},1}, the_model::Covariates)
	return map(x::Array{Float32,1} -> get_skip_circle_inner_mse(x, the_model), sample_eigen_expression)
end

function get_skip_circle_inner_mse(sample_eigen_expression::Array{Array{Float32,1},2}, the_model::Covariates)
	return map(x::Array{Float32,1} -> get_skip_circle_inner_mse(x, the_model), sample_eigen_expression[:])
end

function get_skip_circle_inner_mse(sample_eigen_expression::Array{Float32,1}, the_model::Covariates)
	return mse(CovariatesEncodingOH(sample_eigen_expression, the_model), CovariatesSkipCircularNodeDecodingDense(sample_eigen_expression, the_model))
end

function get_out_of_plane_error(sample_eigen_expression::Array{Float32,1}, the_model::Covariates)
	return mse(CovariatesThroughDecodingDense(sample_eigen_expression, the_model), CovariatesSkipCircularNodeDecodingDense(sample_eigen_expression, the_model))
end

function get_out_of_plane_error(sample_eigen_expression::Array{Float32,2}, the_model::Covariates)
	return mapslices(x::Array{Float32,1} -> get_out_of_plane_error(x, the_model), sample_eigen_expression, dims=1)[:]
end

function get_out_of_plane_error(sample_eigen_expression::Array{Array{Float32,1},1}, the_model::Covariates)
	return map(x::Array{Float32,1} -> get_out_of_plane_error(x, the_model), sample_eigen_expression)
end

function get_out_of_plane_error(sample_eigen_expression::Array{Array{Float32,1},2}, the_model::Covariates)
	return map(x::Array{Float32,1} -> get_out_of_plane_error(x, the_model), sample_eigen_expression[:])
end

function get_out_of_plane_reconstruction_error(sample_eigen_expression::Array{Float32,1}, the_model::Covariates)
	return mse(the_model(sample_eigen_expression), CovariatesSkipCircularNodeDecodingOH(sample_eigen_expression, the_model))
end

function get_out_of_plane_reconstruction_error(sample_eigen_expression::Array{Float32,1}, the_model::Order)
	return mse(the_model(sample_eigen_expression), OrderSkipCircularNode(sample_eigen_expression, the_model))
end

function get_out_of_plane_reconstruction_error(sample_eigen_expression::Array{Float32,2}, the_model::Covariates)
	return mapslices(x::Array{Float32,1} -> get_out_of_plane_reconstruction_error(x, the_model), sample_eigen_expression, dims=1)[:]
end

function get_out_of_plane_reconstruction_error(sample_eigen_expression::Array{Float32,2}, the_model::Order)
	return mapslices(x::Array{Float32,1} -> get_out_of_plane_reconstruction_error(x, the_model), sample_eigen_expression, dims=1)[:]
end

function get_out_of_plane_reconstruction_error(sample_eigen_expression::Array{Array{Float32,1},1}, the_model::Covariates)
	return map(x::Array{Float32,1} -> get_out_of_plane_reconstruction_error(x, the_model), sample_eigen_expression)
end

function get_out_of_plane_reconstruction_error(sample_eigen_expression::Array{Array{Float32,1},1}, the_model::Order)
	return map(x::Array{Float32,1} -> get_out_of_plane_reconstruction_error(x, the_model), sample_eigen_expression)
end

function get_out_of_plane_reconstruction_error(sample_eigen_expression::Array{Array{Float32,1},2}, the_model::Covariates)
	return map(x::Array{Float32,1} -> get_out_of_plane_reconstruction_error(x, the_model), sample_eigen_expression[:])
end

function get_out_of_plane_reconstruction_error(sample_eigen_expression::Array{Array{Float32,1},2}, the_model::Order)
	return map(x::Array{Float32,1} -> get_out_of_plane_reconstruction_error(x, the_model), sample_eigen_expression[:])
end

function get_skip_circle_mse(sample_eigen_expression::Array{Float32,2}, the_model::Covariates)
	return mapslices(x::Array{Float32,1} -> get_skip_circle_mse(x, the_model), sample_eigen_expression, dims=1)[:]
end

function get_skip_circle_mse(sample_eigen_expression::Array{Float32,2}, the_model::Order)
	return mapslices(x::Array{Float32,1} -> get_skip_circle_mse(x, the_model), sample_eigen_expression, dims=1)[:]
end

function get_skip_circle_mse(sample_eigen_expression::Array{Array{Float32,1},1}, the_model::Covariates)
	return map(x::Array{Float32,1} -> get_skip_circle_mse(x, the_model), sample_eigen_expression)
end

function get_skip_circle_mse(sample_eigen_expression::Array{Array{Float32,1},1}, the_model::Order)
	return map(x::Array{Float32,1} -> get_skip_circle_mse(x, the_model), sample_eigen_expression)
end

function get_skip_circle_mse(sample_eigen_expression::Array{Array{Float32,1},2}, the_model::Covariates)
	return map(x::Array{Float32,1} -> get_skip_circle_mse(x, the_model), sample_eigen_expression[:])
end

function get_skip_circle_mse(sample_eigen_expression::Array{Array{Float32,1},2}, the_model::Order)
	return map(x::Array{Float32,1} -> get_skip_circle_mse(x, the_model), sample_eigen_expression[:])
end

function get_skip_circle_mse(sample_eigen_expression::Array{Float32,1}, the_model::Covariates)
	return mse(sample_eigen_expression[1:the_model.o], CovariatesSkipCircularNodeDecodingOH(sample_eigen_expression, the_model))
end

function get_skip_circle_mse(sample_eigen_expression::Array{Float32,1}, the_model::Order)
	return mse(sample_eigen_expression[1:the_model.o], OrderSkipCircularNode(sample_eigen_expression, the_model))
end

function get_circ_mse(the_eigen_data::Array{Float32,2}, the_model::Covariates)
    return mapslices(x::Array{Float32,1} -> get_circ_mse(x, the_model), the_eigen_data, dims=1)[:]
end

function get_circ_mse(the_eigen_data::Array{Float32,2}, the_model::Order)
    return mapslices(x::Array{Float32,1} -> get_circ_mse(x, the_model), the_eigen_data, dims=1)[:]
end

function get_circ_mse(the_eigen_data::Array{Array{Float32,1},1}, the_model::Covariates)
    return map(x::Array{Float32,1} -> get_circ_mse(x, the_model), the_eigen_data)
end

function get_circ_mse(the_eigen_data::Array{Array{Float32,1},1}, the_model::Order)
    return map(x::Array{Float32,1} -> get_circ_mse(x, the_model), the_eigen_data)
end

function get_circ_mse(the_eigen_data::Array{Array{Float32,1},2}, the_model::Covariates)
    return map(x::Array{Float32,1} -> get_circ_mse(x, the_model), the_eigen_data[:])
end

function get_circ_mse(the_eigen_data::Array{Array{Float32,1},2}, the_model::Order)
    return map(x::Array{Float32,1} -> get_circ_mse(x, the_model), the_eigen_data[:])
end

function get_circ_mse(sample_eigen_expression::Array{Float32,1}, the_model::Covariates)
    return mse(CovariatesThroughEncodingDense(sample_eigen_expression, the_model), CovariatesThroughCircularNode(sample_eigen_expression, the_model))
end

function get_circ_mse(sample_eigen_expression::Array{Float32,1}, the_model::Order)
    return mse(OrderThroughEncodingDense(sample_eigen_expression, the_model), OrderThroughCircularNode(sample_eigen_expression, the_model))
end

function (m::SimpleCovariates)(x)
	EncodingDenseOutput = m.L1(x[1:m.o])
	EncodingOHDenseOutput = m.L1_OH(EncodingDenseOutput) * x[m.o:end]
	CircularNodeOutput = m.C(EncodingOHDenseOutput)
	DecodingDenseOutput = m.L2(CircularNodeOutput)
	return DecodingDenseOutput
end

##########################
# CYCLOPS Initialization #
##########################
function InitializeModel(eigen_data, options)
	if size(eigen_data, 1) == options[:o_svd_n_dims]
		output_models = [Order(Dense(options[:o_svd_n_dims], 2), CircularNode, Dense(2, options[:o_svd_n_dims]), options[:o_svd_n_dims]) for ii in 1:options[:train_n_models]]
		return output_models
	else
		x_terms = permutedims(eigen_data[options[:o_svd_n_dims]+1:end, 1:end])
		y_terms = eigen_data[1:options[:o_svd_n_dims], 1:end]
		y_terms_t = permutedims(y_terms)
		llsq_coeffs = -permutedims(llsq(x_terms, y_terms_t, bias = true))
		B = llsq_coeffs[:, end]
		B_OH = llsq_coeffs[:,1:end-1]
		scaled_B_OH = (1 .+ options[:S_OH]) .* (B_OH .+ B) .- B
		options[:B] = B 
		options[:B_OH] = scaled_B_OH
		output_models = Array{Any}([])
		for ii in 1:options[:train_n_models]
			use_S_OH = rand(size(options[:S_OH])...) .* options[:S_OH]
			use_B = rand(size(B)...) .* B
			use_B_OH = rand(size(B_OH)...) .* B_OH
			append!(output_models, [Covariates(Array{Float32}(use_S_OH), Array{Float32}(use_B), Array{Float32}(use_B_OH), Dense(options[:o_svd_n_dims], 2), Dense(2, options[:o_svd_n_dims]), options[:o_svd_n_dims])])
		end
		# output_models = [Covariates(Array{Float32}(use_S_OH), Array{Float32}(use_B), Array{Float32}(use_B_OH), Dense(options[:o_svd_n_dims], 2), CircularNode, Dense(2, options[:o_svd_n_dims]), options[:o_svd_n_dims]) for ii in 1:options[:train_n_models]] # soon to be depricated
		return output_models
	end
end

# Maybe another function to just intialize "blank" covariates model
####################
# CYCLOPS Decoding #
####################
function Decoder(data_matrix, model, n_circs::Integer)
    points = size(data_matrix, 2) # Determine the total number of samples
    phases = Array{Float32}(zeros(n_circs, points)) # Initialize list of phases
    base = 0
    for c in 1:n_circs # how many circular bottleneck layers are there (usually 1)
        for n in 1:points # for each sample
            pos = model(data_matrix[:, n]) # pos is a 2 element array
            phases[c, n] = atan(pos[2 + base], pos[1 + base])# the inverse tangent of the model output is an angle, stored to phases
        end
        base += 2 # move to next circular layer (will in a future version be replaced by map())
    end
	output_phases = mod.(vec(phases), 2π)
	return output_phases
end

function CovariatesProjection(model, ohEigenData, OUT_TYPE = Float64)
	m = model
	OH(x) = x[1:m.o] .* (1 .+ (m.S_OH * x[m.o + 1:end])) .+ (m.B_OH * x[m.o + 1:end]) .+ m.B
	Lin(x) = m.L1(x)
	M(x) = Lin(OH(x))
	projections = zeros(size(ohEigenData, 2), 2)
	for ii in 1:size(projections, 1)
		projections[ii, :] = collect(M(ohEigenData[:, ii]))
	end
	output_projections = Array{OUT_TYPE, 2}(projections)
	return output_projections
end

function CovariatesMagnitude(model, ohEigenData, OUT_TYPE = Float64)
	m = model
	OH(x) = x[1:m.o] .* (1 .+ (m.S_OH * x[m.o + 1:end])) .+ (m.B_OH * x[m.o + 1:end]) .+ m.B
	Lin(x) = m.L1(x)
	M(x) = Lin(OH(x))
	magnitudes = zeros(size(ohEigenData, 2))
	for ii in 1:length(magnitudes)
		magnitudes[ii] = sqrt(sum(collect(M(ohEigenData[:, ii])) .^2))
	end
	output_magnitudes = Array{OUT_TYPE}(magnitudes)
	return output_magnitudes
end

function OrderProjection(model, ohEigenData, OUT_TYPE = Float64)
	m = model
	M(x) = m.L1(x[1:m.o])
	projections = zeros(size(ohEigenData, 2), 2)
	for ii in 1:size(projections, 1)
		projections[ii, :] = collect(M(ohEigenData[:, ii]))
	end
	output_projections = Array{OUT_TYPE, 2}(projections)
	return output_projections
end

function OrderMagnitude(model, ohEigenData, OUT_TYPE = Float64)
	m = model
	M(x) = m.L1(x[1:m.o])
	magnitudes = zeros(size(ohEigenData, 2))
	for ii in 1:length(magnitudes)
		magnitudes[ii] = sqrt(sum(collect(M(ohEigenData[:, ii])) .^2))
	end
	output_magnitudes = Array{OUT_TYPE}(magnitudes)
	return output_magnitudes
end

function CovariatesDecoder(trained_models_and_errors::Array{Any, 1}, gea::Array{Float32,2}, OUT_TYPE = Float64)
	losses = map(x -> x[end], trained_models_and_errors)
	losses[isnan.(losses)] .= Inf
	lowest_loss_index = findmin(losses)[2]

	models = map(x -> x[1], trained_models_and_errors)
	best_model = models[lowest_loss_index]
	
	eigenspace_magnitudes = sqrtsumofsquares(gea[1:best_model.o,:]) # sqrt(sum(A^2))
	encoding_dense_magnitudes = CovariatesEncodingDenseMagnitude(gea, best_model) # sqrt(sum(B^2))
	magnitudes = CovariatesMagnitude(best_model, gea, Float32) # sqrt(sum(C^2))
	decoding_dense_magnitudes = CovariatesDecodingDenseMagnitude(gea, best_model) # sqrt(sum(E^2))
	decoding_eigenspace_magnitudes = sqrtsumofsquares(gea, best_model) # sqrt(sum(F^2))
	
	decoding_dense_skip_circle_magnitudes = CovariatesSkipCircularNodeDecodingDenseMagnitude(gea, best_model) # sqrt(sum(E_b^2))
	decoding_OH_skip_circle_magnitudes = CovariatesSkipCircularNodeDecodingOHMagnitude(gea, best_model) # sqrt(sum(F_b^2))
	
	projections = vcat(permutedims.(CovariatesThroughEncodingDense(gea, best_model))...)
	phases = CovariatesPhase(gea, best_model)
	
	model_errors = MSELoss(gea, best_model) # sqrt((A-F)^2)
	inner_errors = get_inner_mse(gea, best_model) # sqrt((B-E)^2)
	circ_errors = get_circ_mse(gea, best_model) # sqrt((C-D)^2)
	skip_circle_inner_errors = get_skip_circle_inner_mse(gea, best_model) # sqrt((B-E_b)^2)
	skip_circle_errors = get_skip_circle_mse(gea, best_model) # sqrt((A-F_b^2))

	out_of_plane_errors = get_out_of_plane_error(gea, best_model)
	out_of_plane_reconstruction_errors = get_out_of_plane_reconstruction_error(gea, best_model)

	return best_model, hcat(phases, model_errors, magnitudes, projections, inner_errors, circ_errors, skip_circle_inner_errors, skip_circle_errors, out_of_plane_errors, out_of_plane_reconstruction_errors, eigenspace_magnitudes, encoding_dense_magnitudes, decoding_dense_magnitudes, decoding_eigenspace_magnitudes, decoding_dense_skip_circle_magnitudes, decoding_OH_skip_circle_magnitudes)
end

function CovariatesDecoder(best_model, gea::Array{Float32,2}, OUT_TYPE = Float64)
	eigenspace_magnitudes = sqrtsumofsquares(gea[1:best_model.o,:]) # sqrt(sum(A^2))
	encoding_dense_magnitudes = CovariatesEncodingDenseMagnitude(gea, best_model) # sqrt(sum(B^2))
	magnitudes = CovariatesMagnitude(best_model, gea, Float32) # sqrt(sum(C^2))
	decoding_dense_magnitudes = CovariatesDecodingDenseMagnitude(gea, best_model) # sqrt(sum(E^2))
	decoding_eigenspace_magnitudes = sqrtsumofsquares(gea, best_model) # sqrt(sum(F^2))
	
	decoding_dense_skip_circle_magnitudes = CovariatesSkipCircularNodeDecodingDenseMagnitude(gea, best_model) # sqrt(sum(E_b^2))
	decoding_OH_skip_circle_magnitudes = CovariatesSkipCircularNodeDecodingOHMagnitude(gea, best_model) # sqrt(sum(F_b^2))
	
	projections = vcat(permutedims.(CovariatesThroughEncodingDense(gea, best_model))...)
	phases = CovariatesPhase(gea, best_model)
	
	model_errors = MSELoss(gea, best_model) # sqrt((A-F)^2)
	inner_errors = get_inner_mse(gea, best_model) # sqrt((B-E)^2)
	circ_errors = get_circ_mse(gea, best_model) # sqrt((C-D)^2)
	skip_circle_inner_errors = get_skip_circle_inner_mse(gea, best_model) # sqrt((B-E_b)^2)
	skip_circle_errors = get_skip_circle_mse(gea, best_model) # sqrt((A-F_b^2))

	out_of_plane_errors = get_out_of_plane_error(gea, best_model)
	out_of_plane_reconstruction_errors = get_out_of_plane_reconstruction_error(gea, best_model)

	return best_model, hcat(phases, model_errors, magnitudes, projections, inner_errors, circ_errors, skip_circle_inner_errors, skip_circle_errors, out_of_plane_errors, out_of_plane_reconstruction_errors, eigenspace_magnitudes, encoding_dense_magnitudes, decoding_dense_magnitudes, decoding_eigenspace_magnitudes, decoding_dense_skip_circle_magnitudes, decoding_OH_skip_circle_magnitudes)
end

function OrderDecoder(m_array::Array{Any,1}, gea::Array{Float32,2}, OUT_TYPE = Float64)
	losses = [mean([mse(m_array[jj](gea[:, ii]), gea[1:m_array[jj].o, ii]) for ii in 1:size(gea, 2)]) for jj in 1:length(m_array)]
	m = m_array[findmin(losses)[2]]
	eigenspace_magnitudes = sqrtsumofsquares(gea)
	magnitudes = OrderMagnitude(m, gea, Float32)
	decoding_eigenspace_magnitudes = sqrtsumofsquares(gea, m) # make function for order model
	
	decoding_eigenspace_skip_circle_magnitudes = OrderSkipCircularNodeMagnitude(gea, m) # make function for order model
	
	projections = vcat(permutedims.(OrderThroughEncodingDense(gea, m))...)
	phases = OrderPhase(gea, m)

	model_errors = MSELoss(gea, m)
	circ_errors = get_circ_mse(gea, m)

	skip_circle_errors = get_skip_circle_mse(gea, m)

	out_of_plane_errors = get_out_of_plane_reconstruction_error(gea, m)

	return m, hcat(phases, model_errors, magnitudes, projections, circ_errors, skip_circle_errors, out_of_plane_errors, eigenspace_magnitudes, decoding_eigenspace_magnitudes, decoding_eigenspace_skip_circle_magnitudes)
end

function OrderDecoder(best_model, gea::Array{Float32,2}, OUT_TYPE = Float64)
	eigenspace_magnitudes = sqrtsumofsquares(gea)
	magnitudes = OrderMagnitude(best_model, gea, Float32)
	decoding_eigenspace_magnitudes = sqrtsumofsquares(gea, best_model) # make function for order model
	
	decoding_eigenspace_skip_circle_magnitudes = OrderSkipCircularNodeMagnitude(gea, best_model) # make function for order model
	
	projections = vcat(permutedims.(OrderThroughEncodingDense(gea, best_model))...)
	phases = OrderPhase(gea, best_model)

	model_errors = MSELoss(gea, best_model)
	circ_errors = get_circ_mse(gea, best_model)

	skip_circle_errors = get_skip_circle_mse(gea, best_model)

	out_of_plane_errors = get_out_of_plane_reconstruction_error(gea, best_model)

	return best_model, hcat(phases, model_errors, magnitudes, projections, circ_errors, skip_circle_errors, out_of_plane_errors, eigenspace_magnitudes, decoding_eigenspace_magnitudes, decoding_eigenspace_skip_circle_magnitudes)
end

#################
# Saving Models #
#################
function UniversalModelSaver(model; dir=pwd(), name::String="Model1")
	all_model_field_names_before = vcat(propertynames(model)...)
	all_model_field_names = vcat(propertynames(model)...)
	all_model_field_values = map(x -> getfield(model, x), all_model_field_names)
	dense_layer_logical = map(x -> x <: Dense, typeof.(all_model_field_values))
	dense_layer_W_and_b = vcat(map(x -> [x.weight, x.bias], all_model_field_values[dense_layer_logical])...)
	dense_layer_key_names = vcat(map(x -> Symbol.(["$(string(x))_W", "$(string(x))_b"]) , all_model_field_names[dense_layer_logical])...)
	append!(all_model_field_names, dense_layer_key_names)
	append!(all_model_field_values, dense_layer_W_and_b)
	model_dict = Dict(all_model_field_names .=> all_model_field_values)
	map(x -> delete!(model_dict, x), all_model_field_names_before[dense_layer_logical])
	CSV.write(joinpath(dir, "$(name).csv"), model_dict)
	# return model_dict, all_model_field_names, all_model_field_values
end

function get_expression_head(input_string::String)
	ex = Meta.parse(input_string)
	if isa(ex, Expr)
		if ex.head == :ref
			next_head = ex.head
			return ex.args[1]
		end
		return ex.head
	elseif isa(ex, Symbol)
		return :String
	elseif isa(ex, Bool)
		return :Bool
	elseif isa(ex, Number)
		return Symbol(typeof(ex))
	end
end

function safely_eval_string(input_string::String)
	ex = Meta.parse(input_string)

end

function UniversalModelLoader(path)
	model_csv = CSV.read(path, DataFrame)
	csv_model_symbols = Symbol.(model_csv[:, 1])
	csv_model_values = model_csv[:, 2]
	if length(csv_model_symbols) == 8
		model_parameters = fieldnames(Covariates)
		covariate_parameter_symbols = [:S_OH, :B, :B_OH]
		covariate_parameter_indices_in_model_parameters = findXinY(covariate_parameter_symbols, model_parameters)
	else
		model_parameters = fieldnames(Order)
	end
	L1_dense_parameters = [:L1_W, :L1_b]
	L2_dense_parameters = [:L2_W, :L2_b]
	L1_model_value_indices = findXinY(L1_dense_parameters, csv_model_symbols)
	L2_model_value_indices = findXinY(L2_dense_parameters, csv_model_symbols)
	L1_model_values_strings = csv_model_values[L1_model_value_indices]
	L2_model_values_strings = csv_model_values[L2_model_value_indices]

	L1_dense = DenseFunction(L1_model_values...)
	L2_dense

	# dense_parameter_indices_in_model_parameter = findXinY(dense_parameters, model_parameters)
end

function SaveCovariatesModel(x; dir=pwd(), name::String="Model1")
    S_OH = x.S_OH
    B = x.B
	B_OH = x.B_OH
    L1W = x.L1.W
	L1W_store = reshape(L1W, size(L1W, 2), size(L1W, 1))
    L1b = x.L1.b
    L2W = x.L2.W
    L2b = x.L2.b
    L1b_store = Array{Any}(zeros(size(L1W, 2)))
    L1b_store[1:2] = L1b
    model_store = DataFrame(hcat(S_OH, B, B_OH, L1W_store, L1b_store, L2W, L2b))
    CSV.write(joinpath(dir, string("Covariates_", name, ".csv")), model_store)
	println("Model stored as: ", joinpath(dir, string("Covariates_", name, ".csv")))
end

function SaveOrderModel(x; dir=pwd(), name::String="Model1")
    L1W = x.L1.W
    L1b = x.L1.b
    L2W = x.L2.W
    L2b = x.L2.b
    L1b_store = Array{Any}(zeros(size(L1W, 2)))
    L1b_store[1:2] = L1b
    model_store = DataFrame(hcat(reshape(L1W, size(L1W, 2), size(L1W, 1)), L1b_store, L2W, L2b))
	CSV.write(joinpath(dir, string("Order_", name, ".csv")), model_store)
	println("Model stored as: ", joinpath(dir, string("Order_", name, ".csv")))
end
##################
# Loading Models #
##################
function LoadOrder(x::DataFrame)
    neig = size(x, 1)
    L2b = x[:, end]
    L2W = Array{Float32, 2}(x[:, end-2:end-1])
    L1b = x[1:2, end-3]
    L1W_misshaped = Array{Float32, 2}(x[:, end-5:end-4])
    L1W = reshape(L1W_misshaped, size(L1W_misshaped, 2), size(L1W_misshaped, 1))
	L1 = DenseFunction(L1W, L1b)
	L2 = DenseFunction(L2W, L2b)
	output_model = Order(L1, CircularNode, L2, neig)
	return output_model
end

function LoadCovariates(x::DataFrame)
	neig = size(x, 1)
	L2b = x[:, end]
	L2W = Array{Float32, 2}(x[:, end-2:end-1])
	L1b = x[1:2, end-3]
	L1W_misshaped = Array{Float32, 2}(x[:, end-5:end-4])
	L1W = reshape(L1W_misshaped, size(L1W_misshaped, 2), size(L1W_misshaped, 1))
	L1 = DenseFunction(L1W, L1b)
	L2 = DenseFunction(L2W, L2b)
	OHLayer = Array{Float32, 2}(x[:, 1:end-6])
	nb = Int((size(OHLayer, 2) .- 1)/2)
	# in_dim = neig + nb
	S_OH = OHLayer[:, 1:nb]
	B = OHLayer[:, nb+1]
	B_OH = OHLayer[:, nb+2:end]
	output_model = Covariates(S_OH, B, B_OH, L1, CircularNode, L2, neig)
	return output_model
end

function LoadModel(name::String; dir=pwd())
    m_stored = CSV.read(joinpath(dir, string(name, ".csv")))

	if size(m_stored, 2) > 6

		output_model = LoadCovariates(m_stored)
	else
		output_model = LoadOrder(m_stored)
	end
	return output_model
end

function build_model(the_path::String)
    model_csv = CSV.read(the_path, DataFrame)
    model_S_OH = eval(Meta.parse(model_csv[2,2]))
    model_B = eval(Meta.parse(model_csv[6,2]))
    model_B_OH = eval(Meta.parse(model_csv[4,2]))
    model_L1_w = eval(Meta.parse(model_csv[3,2]))
    model_L1_b = eval(Meta.parse(model_csv[5,2]))
    model_L1 = CYCLOPS.DenseFunction(model_L1_w, model_L1_b)
    model_L2_w = eval(Meta.parse(model_csv[7,2]))
    model_L2_b = eval(Meta.parse(model_csv[8,2]))
    model_L2 = CYCLOPS.DenseFunction(model_L2_w, model_L2_b)
    model_o = parse(Int64, model_csv[1,2])
    return CYCLOPS.Covariates(model_S_OH, model_B, model_B_OH, model_L1, model_L2, model_o)
end

function build_model_2(the_path::String)
	model_csv = CSV.read(the_path, DataFrame)
	model_S_OH = eval(Meta.parse(model_csv[2,2]))
	model_B_OH = eval(Meta.parse(model_csv[3,2]))
	model_L2 = eval(Meta.parse(model_csv[4,2]))
	model_B = eval(Meta.parse(model_csv[5,2]))
	model_L1 = eval(Meta.parse(model_csv[6,2]))
	model_o = parse(Int64, model_csv[1,2])
	return CYCLOPS.Covariates(model_S_OH, model_B, model_B_OH, model_L1, model_L2, model_o)
end

##################
# Loss Functions #
##################
function MSELoss(x_l::Array{Float32,1}, m_l)
	return mse(m_l(x_l), x_l[1:m_l.o])
end

function MSELoss(x_l::Array{Float32,2}, m_l)
	return Flux.mse.(m_l(x_l), mapslices(y::Array{Float32,1} -> [y], x_l[1:m_l.o,:], dims=1)[:])
end

function MSELoss(x_l::Array{Array{Float32,1},1}, m_l)
	return Flux.mse.(m_l(x_l), x_l)
end

function MSELoss(x_l::Array{Array{Float32,1},2}, m_l)
	return Flux.mse.(m_l(x_l), x_l[:])
end

function TrueTimeLoss(x_l, y_l, m_l)
	circular_coordinates = CovariatesThroughCircularNode(x_l, m_l)
	true_coordinates = [cos(y_l[2]), sin(y_l[2])]
	coordinate_distance = circular_coordinates .- true_coordinates
	return y_l[1] * sqrt(sum(coordinate_distance .^2))
end

function TrueTimeLoss2(l_sample_eigendata, l_true_radian, m_l)
	cyclops_projection = CovariatesThroughCircularNode(l_sample_eigendata, m_l)
	cyclops_phase=atan(cyclops_projection[2],cyclops_projection[1])
	cos_distance= 1-cos(cyclops_phase-l_true_radians)
	return 2*cos_distance * l_true_radian[1]
end

function MSETrueTimeLoss(x_l, y_l, m_l)
	return MSELoss(x_l, m_l) + TrueTimeLoss(x_l, y_l, m_l)
end

function MSETrueTimeLoss2(x_l, y_l, m_l, fudge_factor)
	return MSELoss(x_l, m_l) + TrueTimeLoss2(x_l, y_l, m_l) * fudge_factor
end

function CircularNodeLoss(x_l, m_l)
	InputToEncodingDense = CovariatesEncodingOH(x_l, m_l)
	OutputOfDecodingDense = CovariatesThroughDecodingDense(x_l, m_l)
	return mse(InputToEncodingDense, OutputOfDecodingDense)
end

######################
# Training Functions #
######################
function TrainCovariates(m, gea_vectorized; MinSteps::Int = 250, MaxSteps::Int = 1000, μA = 0.0001, β = (0.9, 0.999), cutoff::Int = 1000)
	local c1 = 0
	local c2 = 0
	local c3 = 0
	local c4 = 0
	local c5 = 0
	local c6 = 0
	local after = 0

    try
		μA_original = μA
		while c1 < MinSteps
			c1 += 1
			Flux.train!(x->mse(m(x), x[1:m.o]), Flux.params(m.S_OH, m.B, m.B_OH, m.L1, m.L2), zip(gea_vectorized), ADAM(μA, β), cb = () -> ())
		end
		
		smallest_μA = μA
		largest_μA = μA
		
		while (c2 < MaxSteps) & (μA > μA_original/cutoff)
			
			c2 += 1
			μA = μA * 1.05
			before = mean(map(x -> mse(m(x), x[1:m.o]), gea_vectorized))
			before_m = deepcopy(m)
			Flux.train!(x->mse(m(x), x[1:m.o]), Flux.params(m.S_OH, m.B, m.B_OH, m.L1, m.L1, m.L2), zip(gea_vectorized), ADAM(μA, β), cb = () -> ())
			after = mean(map(x -> mse(m(x), x[1:m.o]), gea_vectorized))
			change = before - after

			while (change <= 0) & (μA > μA_original/cutoff)
				c3 += 1
				c4 += 1
				μA = μA * 0.5
				m = before_m
				Flux.train!(x->mse(m(x), x[1:m.o]), Flux.params(m.S_OH, m.B, m.B_OH, m.L1, m.L2), zip(gea_vectorized), ADAM(μA, β), cb = () -> ())
				after = mean(map(x -> mse(m(x), x[1:m.o]), gea_vectorized))
				change = before - after
			end
			
			if μA < μA_original
				c5 += 1
				if μA < smallest_μA
					smallest_μA = trunc(μA, sigdigits = 4)
				end
			elseif μA > μA_original
				c6 += 1
				if μA > largest_μA
					largest_μA = trunc(μA, sigdigits = 4)
				end
			end
			
			# c2 += c4
			c4 = 0

		end
		
		println("Model took $(c1 + c2) total training steps. Of these, $(c2) were variable learning rate steps.\nThe learning rate was decreased $c3 times and was smaller than the original learning rate for $c5 steps.\nThe learning rate was increased $c2 times and was larger than the original learning rate for $c6 steps.\nThe final learning rate was $(trunc(μA, sigdigits = 4)); the smallest the learning rate became was $(smallest_μA) and the largest it became was $(largest_μA).\n\n~~~~~~~~~~~~TRAINING COMPLETE~~~~~~~~~~~~\n\n")

	catch e
		println("An error occured in training. $(c1 + c2) steps ($(c2) variable) were taken before the error occured.")
		throw(e)
	end

	return m, after
end

function TrainCovariatesTrueTimes(m, gea_vectorized, collection_times_with_flag_vectorized; MinSteps::Int = 250, MaxSteps::Int = 1000, μA = 0.0001, β = (0.9, 0.999), cutoff::Int = 1000, collection_time_balance = 4)
    
	local c1 = 0
	local c2 = 0
	local c3 = 0
	local c4 = 0
	local c5 = 0
	local c6 = 0
	local after = 0

	function EncodingOHLayer(x_l)
		return x_l[1:m.o] .* (1 .+ (m.S_OH * x_l[m.o + 1:end])) .+ (m.B_OH * x_l[m.o + 1:end]) .+ m.B
	end
	
	function AfterCircularLayer(x_l)
		return CircularNode(m.L1(EncodingOHLayer(x_l)))
	end
	
	function AfterDecodingDense(x_l)
		return m.L2(AfterCircularLayer(x_l))
	end
	
	function MSELoss(x_l)
		return mse(m(x_l), x_l[1:m.o])
	end
	
	function CircularLoss(x_l)
		return mse(EncodingOHLayer(x_l), AfterDecodingDense(x_l))
	end
	
	function TrueTimeLoss(x_l, y_l)
		circular_coordinates = AfterCircularLayer(x_l)
		true_coordinates = [cos(y_l[2]), sin(y_l[2])]
		coordinate_distance = circular_coordinates .- true_coordinates
		return y_l[1] * sqrt(sum(coordinate_distance .^2))
	end
	
	function TotalLoss(x_l, y_l)
		return MSELoss(x_l) + TrueTimeLoss(x_l, y_l)
	end

	function TotalLoss2(x_l, y_l, collection_time_balance)
		return MSELoss(x_l) + TrueTimeLoss2(x_l, y_l) * collection_time_balance
	end

	function TrueTimeLoss2(l_sample_eigendata, l_true_radian)
		cyclops_projection = AfterCircularLayer(l_sample_eigendata)
		cyclops_phase=atan(cyclops_projection[2],cyclops_projection[1])
		cos_distance= 1-cos(cyclops_phase-l_true_radian[2])
		return 2*cos_distance * l_true_radian[1]
	end
	
	local zip_data = zip(gea_vectorized, collection_times_with_flag_vectorized)

	try
		μA_original = deepcopy(μA)

		while c1 < MinSteps
			c1 += 1
			Flux.train!((x, y) -> TotalLoss2(x, y, collection_time_balance), Flux.params(m.S_OH, m.B, m.B_OH, m.L1, m.L2), zip_data, ADAM(μA, β), cb = () -> ()) # Error here
		end

		smallest_μA = μA
		largest_μA = μA

		c2 = 0
		c3 = 0

		while (c2 < MaxSteps) & (μA > μA_original/cutoff)
			c2 += 1
			μA = μA * 1.05
			before = mean(map(x -> TotalLoss2(x[1], x[2], collection_time_balance), zip_data))
			#before = mean(map(x -> mse(m(x), x[1:m.o]), gea_vectorized))
			before_m = deepcopy(m)
			Flux.train!((x, y) -> TotalLoss2(x, y, collection_time_balance), Flux.params(m.S_OH, m.B, m.B_OH, m.L1, m.L2), zip_data, ADAM(μA, β), cb = () -> ())
			after = mean(map(x -> TotalLoss2(x[1], x[2], collection_time_balance), zip_data))
			# after = mean(map(x -> mse(m(x), x[1:m.o]), gea_vectorized))
			change = before - after

			while (change <= 0) & (μA > μA_original/cutoff)
				c3 += 1
				c4 += 1
				μA = μA * 0.5
				m = before_m
				Flux.train!((x, y) -> TotalLoss2(x, y, collection_time_balance), Flux.params(m.S_OH, m.B, m.B_OH, m.L1, m.L2), zip_data, ADAM(μA, β), cb = () -> ())
				after = mean(map(x -> TotalLoss2(x[1], x[2], collection_time_balance), zip_data))
				change = before - after
			end

			if μA < μA_original
				c5 += 1
				if μA < smallest_μA
					smallest_μA = trunc(μA, sigdigits = 4)
				end
			elseif μA > μA_original
				c6 += 1
				if μA > largest_μA
					largest_μA = trunc(μA, sigdigits = 4)
				end
			end
			
			c2 += c4
			c4 = 0

		end

		println("Model took $(c1 + c2) total training steps. Of these, $(c2) were variable learning rate steps.\nThe learning rate was decreased $c3 times and was smaller than the original learning rate for $c5 steps.\nThe learning rate was increased $c2 times and was larger than the original learning rate for $c6 steps.\nThe final learning rate was $(trunc(μA, sigdigits=4)); the smallest the learning rate became was $(smallest_μA) and the largest it became was $(largest_μA).\n\n~~~~~~~~~~~~TRAINING COMPLETE~~~~~~~~~~~~\n\n")

	catch e
		println("An error occured in training. $(c1 + c2 + c4) steps ($(c3 + c2) variable) were taken before the error occured.")
		throw(e)
	end

	return m, after
end

function TrainCovariatesCircular(m, gea_vectorized; MinSteps::Int = 250, MaxSteps::Int = 1000, μA = 0.0001, β = (0.9, 0.999), cutoff::Int = 1000)
    μA_original = deepcopy(μA)
    c1 = 0
    while c1 < MinSteps
		c1 += 1
		circ_in(x) = x[1:m.o] .* (1 .+ (m.S_OH * x[m.o + 1:end])) .+ (m.B_OH * x[m.o + 1:end]) .+ m.B
		circ_out(x) = m.L2(CircularNode(m.L1(circ_in(x))))
		Flux.train!(x->mse(circ_in(x), circ_out(x)), Flux.params(m.L1, m.L2), zip(gea_vectorized), ADAM(μA, β), cb = () -> ())
        Flux.train!(x->mse(m(x), x[1:m.o]), Flux.params(m.S_OH, m.B, m.B_OH), zip(gea_vectorized), ADAM(μA, β), cb = () -> ())
    end
    c2 = 0
    c3 = 0
	after = 0
    while (c2 < MaxSteps) & (μA > μA_original/cutoff)
        c2 += 1
        μA = μA * 1.05
        before = mean(map(x -> mse(m(x), x[1:m.o]), gea_vectorized))
        before_m = m
		circ_in(x) = x[1:m.o] .* (1 .+ (m.S_OH * x[m.o + 1:end])) .+ (m.B_OH * x[m.o + 1:end]) .+ m.B
		circ_out(x) = m.L2(CircularNode(m.L1(circ_in(x))))
		Flux.train!(x->mse(circ_in(x), circ_out(x)), Flux.params(m.L1, m.L2), zip(gea_vectorized), ADAM(μA, β), cb = () -> ())
        Flux.train!(x->mse(m(x), x[1:m.o]), Flux.params(m.S_OH, m.B, m.B_OH), zip(gea_vectorized), ADAM(μA, β), cb = () -> ())
        after = mean(map(x -> mse(m(x), x[1:m.o]), gea_vectorized))
        change = before - after
        c4 = 0
        while (change <= 0) & (μA > μA_original/cutoff)
            c3 += 1
            c4 += 1
            μA = μA * 0.5
            m = before_m
			circ_in(x) = x[1:m.o] .* (1 .+ (m.S_OH * x[m.o + 1:end])) .+ (m.B_OH * x[m.o + 1:end]) .+ m.B
			circ_out(x) = m.L2(CircularNode(m.L1(circ_in(x))))
			Flux.train!(x->mse(circ_in(x), circ_out(x)), Flux.params(m.L1, m.L2), zip(gea_vectorized), ADAM(μA, β), cb = () -> ())
			Flux.train!(x->mse(m(x), x[1:m.o]), Flux.params(m.S_OH, m.B, m.B_OH), zip(gea_vectorized), ADAM(μA, β), cb = () -> ())
            after = mean(map(x -> mse(m(x), x[1:m.o]), gea_vectorized))
            change = before - after
        end
    end
	return m, after
end

function TrainOrder(m, gea_vectorized; MinSteps::Int = 250, MaxSteps::Int = 1000, μA = 0.0001, β = (0.9, 0.999), cutoff::Int = 1000)
    μA_original = deepcopy(μA)
    c1 = 0
    while c1 < MinSteps
        c1 += 1
        Flux.train!(x->mse(m(x), x[1:m.o]), Flux.params(m.L1, m.L2), zip(gea_vectorized), ADAM(μA, β), cb = () -> ())
    end
    c2 = 0
    c3 = 0
	after = 0
    while (c2 < MaxSteps) & (μA > μA_original/cutoff)
        c2 += 1
        μA = μA * 1.05
        before = mean(map(x -> mse(m(x), x[1:m.o]), gea_vectorized))
        before_m = m
        Flux.train!(x->mse(m(x), x[1:m.o]), Flux.params(m.L1, m.L2), zip(gea_vectorized), ADAM(μA, β), cb = () -> ())
        after = mean(map(x -> mse(m(x), x[1:m.o]), gea_vectorized))
        change = before - after
        c4 = 0
        while (change <= 0) & (μA > μA_original/cutoff)
            c3 += 1
            c4 += 1
            μA = μA * 0.5
            m = before_m
            Flux.train!(x->mse(m(x), x[1:m.o]), Flux.params(m.L1, m.L2), zip(gea_vectorized), ADAM(μA, β), cb = () -> ())
            after = mean(map(x -> mse(m(x), x[1:m.o]), gea_vectorized))
            change = before - after
        end
    end
	return m # ", after" removed because OrderDecoder not setup to handle tuple of model and erro
end

function MultiTrainCovariates(m_array, gea::Array{Float32,2}, options)
	trained_models = Array{Any}([])
	gea_vectorized = mapslices(x -> [x], gea, dims = 1)[:]
	if options[:train_circular]
		append!(trained_models, pmap(x -> TrainCovariatesCircular(x, gea_vectorized, μA = options[:train_μA], β = options[:train_β], MinSteps = options[:train_min_steps], MaxSteps = options[:train_max_steps], cutoff = options[:train_μA_scale_lim]), m_array, on_error = e -> rethrow(e)))
	elseif haskey(options, :train_sample_id) & options[:train_collection_times]
		known_sample_indices = findXinY(options[:train_sample_id], options[:o_column_ids])
		init_collection_time = zeros(length(gea_vectorized))
		# println(length(init_collection_time))
		# println(length(options[:align_phases]))
		# println(length(options[:align_samples]))
		init_collection_time[known_sample_indices] .= options[:train_sample_phase]
		init_collection_time_flag = falses(length(init_collection_time))
		init_collection_time_flag[known_sample_indices] .= true
		collection_times_with_flag_vectorized = mapslices(x -> [Float32.(x)], hcat(init_collection_time_flag, init_collection_time)', dims = 1)[:]
		append!(trained_models, pmap(x -> TrainCovariatesTrueTimes(x, gea_vectorized, collection_times_with_flag_vectorized, μA = options[:train_μA], β = options[:train_β], MinSteps = options[:train_min_steps], MaxSteps = options[:train_max_steps], cutoff = options[:train_μA_scale_lim], collection_time_balance = options[:train_collection_time_balance]), m_array, on_error = e -> rethrow(e))) # Error here leads to 449
	else
		append!(trained_models, pmap(x -> TrainCovariates(x, gea_vectorized, μA = options[:train_μA], β = options[:train_β], MinSteps = options[:train_min_steps], MaxSteps = options[:train_max_steps], cutoff = options[:train_μA_scale_lim]), m_array))
	end
	return trained_models
end

function MultiTrainOrder(m_array, gea::Array{Float32,2}, options)
	trained_models = Array{Any}([])
	gea_vectorized = mapslices(x -> [x], gea, dims = 1)[:]
	append!(trained_models, pmap(x -> TrainOrder(x, gea_vectorized, μA = options[:train_μA], β = options[:train_β], MinSteps = options[:train_min_steps], MaxSteps = options[:train_max_steps], cutoff = options[:train_μA_scale_lim]), m_array))
	return trained_models
end

#######################
# Data Pre-Processing #
#######################
function my_info(my_message::String, silence_message = false)
	if !silence_message
		println(""); @info my_message; println("")
	end
end

function my_warn(my_message::String, silence_message = false)
	if !silence_message
		println(""); @warn my_message; println("")
	end
end

function my_error(my_message::String, silence_message = false)
	if !silence_message
		println(""); error(my_message); println("")
	end
end

function MakeFloat(ar::T where T <: Union{DataFrame, Array}, OUT_TYPE::DataType = Float64)
	ar = Array{Any}(Matrix(ar))
	map!(x -> typeof(x) <: AbstractString ? parse(OUT_TYPE, x) : x, ar, ar)
	return Array{OUT_TYPE, 2}(ar)
end

function BluntXthPercentile(dataFile::T, options; OUT_TYPE = DataFrame) where T <: Union{DataFrame, Array{T2, 2}} where T2 <: Union{Float32, Float64}
	
	my_info("BLUNTING OUTLIERS IN DATASET.")
	
	data = dataFile

	if typeof(dataFile) == DataFrame
		data = MakeFloat(dataFile[options[:o_fxr]:end, 2:end])
	end
	ngene, nsamples = size(data)
	nfloor = Int(1 + floor((1 - options[:blunt_percent]) * nsamples))
	nceiling = Int(ceil(options[:blunt_percent] * nsamples))
	sorted_data = sort(data, dims = 2)
	row_min = sorted_data[:, nfloor]
	row_max = sorted_data[:, nceiling]
	too_small = data .< row_min
	too_large = data .> row_max
	for ii in 1:ngene
		below_min_logical = data[ii, :] .< row_min[ii]
		above_max_logical = data[ii, :] .> row_max[ii]
		data[ii, below_min_logical] .= row_min[ii]
		data[ii, above_max_logical] .= row_max[ii]
	end
	if OUT_TYPE == DataFrame
		bluntedDataFile = DataFrame(hcat(Array(dataFile[:, 1]), vcat(Array(dataFile[1:options[:o_fxr]-1, 2:end]), data)), names(dataFile))
		return bluntedDataFile
	end
	output_data = Array{OUT_TYPE, 2}(data)
	return output_data
end

function BluntXthPercentile(original_dataFile_used::T, new_dataFile_to_use::T, options; OUT_TYPE = DataFrame) where T <: Union{DataFrame, Array{Float32, 2}, Array{Float64, 2}}
	
	my_info("BLUNTING NEW DATASET ACCORING A PREVIOUS DATASET.")

	data1 = original_dataFile_used
	data2 = new_dataFile_to_use
	if typeof(original_dataFile_used) == DataFrame
		@debug "The original data file used is a dataframe. Extract the gene expression data from the dataframe."
		data1 = MakeFloat(original_dataFile_used[options[:o_fxr]:end, 2:end])
		data2 = MakeFloat(new_dataFile_to_use[options[:o_fxr]:end, 2:end])
	end
	@debug "Finding the number of genes/rows and samples/columns."
	ngene, nsamples = size(data1)
	@debug "There are $ngene genes/rows and $nsamples samples/columns."
	@debug "Getting the index of the smallest value of each row."
	nfloor = Int(1 + floor((1 - options[:blunt_percent]) * nsamples))
	@debug "Getting the index of the largest value of each row."
	nceiling = Int(ceil(options[:blunt_percent] * nsamples))
	@debug "Sort each row in ascending order, left to right."
	sorted_data1 = sort(data1, dims = 2)
	@debug "Take column $nfloor for the smallest value of each row."
	row_min = sorted_data1[:, nfloor]
	@debug "Take column $nceiling for the largest value of each row."
	row_max = sorted_data1[:, nceiling]
	too_small1 = data1 .< row_min
	@debug "Find all the values smaller than the row minimum."
	too_large1 = data1 .> row_max
	@debug "Find all the values larger than the row maximum."
	for ii in 1:ngene
		below_min_logical1 = data1[ii, :] .< row_min[ii]
		below_min_logical2 = data2[ii, :] .< row_min[ii]
		above_max_logical1 = data1[ii, :] .> row_max[ii]
		above_max_logical2 = data2[ii, :] .> row_max[ii]
		data1[ii, below_min_logical1] .= row_min[ii]
		data2[ii, below_min_logical2] .= row_min[ii]
		data1[ii, above_max_logical1] .= row_max[ii]
		data2[ii, above_max_logical2] .= row_max[ii]
	end
	if OUT_TYPE == DataFrame
		bluntedDataFile1 = DataFrame(hcat(Array(original_dataFile_used[:, 1]), vcat(Array(original_dataFile_used[1:options[:o_fxr]-1, 2:end]), data1)), names(original_dataFile_used))
		bluntedDataFile2 = DataFrame(hcat(Array(new_dataFile_to_use[:, 1]), vcat(Array(new_dataFile_to_use[1:options[:o_fxr]-1, 2:end]), data2)), names(new_dataFile_to_use))
		return bluntedDataFile1, bluntedDataFile2
	end
	output_data1 = Array{OUT_TYPE, 2}(data1)
	output_data2 = Array{OUT_TYPE, 2}(data2)
	return output_data1, output_data2
end

function MeanNormalize(data::Array{T, 2} where T <: Union{Float32, Float64}; OUT_TYPE=Float64)
	@debug "\"Dispersion Normalization\" or \"Mean Normalization\""
	@debug "Finding the gene means/row mean. \"dims\" keyword indicates which dimension to use the \":\" operator on; the second dimension gives us all columns for each row."
	geneMeans = mean(data, dims = 2)
	@debug "Subtracting the gene means from all values of their respecitive rows and dividing the difference by the rows mean."
	normData = (data .- geneMeans) ./ geneMeans
	@debug "Convert array into desired Float type."
	output_data = Array{OUT_TYPE, 2}(normData)
	return output_data
end

function MeanNormalize(data1::Array{T, 2}, data2::Array{T, 2}; OUT_TYPE=Float64) where T <: Union{Float32, Float64}
	@debug "\"Dispersion Normalization\" or \"Mean Normalization\" where the mean from the original dataset is applied to the calculation of the second dataset."
	@debug "Finding the gene mean/row mean of the original dataset."
	geneMeans = mean(data1, dims = 2)
	@debug "Subtracting the first dataset's row means from both datasets and dividing by the first dataset's row means."
	normData1 = (data1 .- geneMeans) ./ geneMeans
	normData2 = (data2 .- geneMeans) ./ geneMeans
	@debug "Convert array into desired Float type."
	output_data1 = Array{OUT_TYPE, 2}(normData1)
	output_data2 = Array{OUT_TYPE, 2}(normData2)
	return output_data1, output_data2
end

function GeneLevelCutoffs(dataFile::DataFrame, genesOfInterest, options)
	
	geneColumnNoCov = dataFile[options[:o_fxr]:end, 1] # Update the gene column to only include gene symbols
	expressionData = MakeFloat(dataFile[options[:o_fxr]:end, 2:end], Float64)
	
	# println("Size of expression data is $(size(expressionData, 1)) rows and $(size(expressionData, 2)) columns.")
	mthGeneMeanCutoff = sort(mean(expressionData, dims = 2), dims = 1, rev = true)[options[:seed_mth_Gene] + 1]
	
	genesOfInterestIndices = findXinY(genesOfInterest, geneColumnNoCov)
	genesOfInterestExpressionData = expressionData[genesOfInterestIndices, :]
	
	genesOfInterestMeans = mean(genesOfInterestExpressionData, dims = 2)
	genesOfInterestStd = std(genesOfInterestExpressionData, dims = 2)
	genesOfInterestCVs = genesOfInterestStd ./ genesOfInterestMeans
	
	cvBelowCutoff = findall(genesOfInterestCVs[:, 1] .< options[:seed_max_CV])
	cvAboveCutoff = findall(genesOfInterestCVs[:, 1] .> options[:seed_min_CV])
	meanBelowCutoff = findall(genesOfInterestMeans[:, 1] .> mthGeneMeanCutoff)
	
	genesOfInterestIndicesToKeep = genesOfInterestIndices[intersect(cvAboveCutoff, cvBelowCutoff, meanBelowCutoff)]
	output_genes_of_interest_expression_data = expressionData[genesOfInterestIndicesToKeep, :]
	
	options[:o_seed_genes] = Array{String,1}(geneColumnNoCov[genesOfInterestIndicesToKeep])
	return output_genes_of_interest_expression_data, genesOfInterestIndicesToKeep
end

function FindRegexinArray(search_x, in_y)
	indices = findall(isa.(map(x -> match(search_x, x), in_y), RegexMatch))
	return indices
end

function CovariateProcessing!(dataFile, options)

	my_info("COLLECTING COVARIATE INFORMATION.")

	options[:o_column_ids] = names(dataFile)[2:end]
	geneColumn = dataFile[:, 1] # The column that contains the covariate row names but also all the gene symbols for the expression matrix
	contCovIndx = FindRegexinArray(options[:regex_cont], geneColumn) # Find the continuous covariate rows
	# println(contCovIndx)
	cont_cov_row = missing # Init
	cont_min_max = missing # Init
	norm_cov_row = missing # Init
	if length(contCovIndx) > 0

		if length(contCovIndx) > 1
			print_statement = "$(length(contCovIndx)) CONTINUOUS COVARIATES."
		else
			print_statement = "1 CONTINUOUS COVARIATE."
		end

		my_info(print_statement)

		cont_cov_row = MakeFloat(dataFile[contCovIndx, 2:end], Float64) # ...make sure they're floats
		rowMinima = minimum(cont_cov_row, dims = 2) # Find the row minimum (minimum value of a covariate)
		rowMaxima = maximum(cont_cov_row, dims = 2) # Find the row maximum (maximum value of a covariate)
		cont_min_max = [rowMinima, rowMaxima] # Create tuple for output
		norm_cov_row = (cont_cov_row .- rowMinima) ./ (rowMaxima .- rowMinima) .+ 1 # Normalize the covariate to values between 1 and 2
	end
	discCovIndx = FindRegexinArray(options[:regex_disc], geneColumn) # Find the discontinuous covariate rows
	# println(discCovIndx)
	discCov = missing # Init
	disc_cov_labels = missing # Init
	disc_full_onehot = missing # Init
	onehot_redundance_removed = missing # Init
	onehot_redundance_removed_transpose = missing # Init
	if length(discCovIndx) > 0

		if length(discCovIndx) > 1
			print_statement = "$(length(discCovIndx)) DISCONTINUOUS COVARIATES."
		else
			print_statement = "1 DISCONTINUOUS COVARIATE."
		end

		my_info(print_statement)

		discCov = Array(dataFile[discCovIndx, 2:end]) # select discontinuous covariate rows
		disc_cov_labels = mapslices(x -> [unique(x)], discCov, dims = 2) # Find unique labels
		disc_full_onehot = [Int.(onehotbatch(discCov[ii, :], disc_cov_labels[ii])) for ii in 1:size(discCov, 1)] # Create Onehot
		onehot_redundance_removed = map(x -> x[2:end, :], disc_full_onehot)
		onehot_redundance_removed_transpose = map(x -> permutedims(x), onehot_redundance_removed)
	end

	n_cov = length(contCovIndx) + length(discCovIndx)
	my_info("$n_cov TOTAL COVARIATES.")

	first_expression_row = maximum(vcat(contCovIndx, discCovIndx, 0)) + 1
	# dictionary_keys_long = (:first_expression_row, :discontinuous_covariate_labels, :discontinuous_covariates, :discontinuous_covariate_onehot, :discontinuous_covariate_onehot_redundance_removed, :discontinuous_covariate_onehot_redundance_removed_transpose, :continuous_covariates, :normalized_continuous_covariates, :continuous_covariate_minima_and_maxima)
	dictionary_keys_short = (:o_fxr, :o_dcl, :o_dc, :o_dco, :o_dcorr, :o_dcorrt, :o_cc, :o_ncc, :o_ccmm)
	dictionary_values = (first_expression_row, disc_cov_labels, discCov, disc_full_onehot, onehot_redundance_removed, onehot_redundance_removed_transpose, cont_cov_row, norm_cov_row, cont_min_max)
	map(dictionary_keys_short, dictionary_values) do new_key, new_value
		options[new_key] = new_value
	end
end

function CovariateProcessing(dataFile1, dataFile2, options)

	my_info("COLLECTING COVARIATE INFORMATION.")

	ops1 = Dict(options)
	ops2 = Dict(options)

	ops1[:o_column_ids] = names(dataFile1)[2:end]
	ops2[:o_column_ids] = names(dataFile2)[2:end]
	geneColumn = dataFile1[:, 1] # The column that contains the covariate row names but also all the gene symbols for the expression matrix
	contCovIndx = FindRegexinArray(ops1[:regex_cont], geneColumn) # Find the continuous covariate rows
	# println(contCovIndx)
	cont_cov_row1 = missing # Init
	cont_cov_row2 = missing # Init
	cont_min_max1 = missing # Init
	cont_min_max2 = missing # Init
	norm_cov_row1 = missing # Init
	norm_cov_row2 = missing # Init
	if length(contCovIndx) > 0

		if length(contCovIndx) > 1
			print_statement = "$(length(contCovIndx)) CONTINUOUS COVARIATES."
		else
			print_statement = "1 CONTINUOUS COVARIATE."
		end

		my_info(print_statement)

		cont_cov_row1 = MakeFloat(dataFile1[contCovIndx, 2:end], Float64) # ...make sure they're floats
		cont_cov_row2 = MakeFloat(dataFile2[contCovIndx, 2:end], Float64) # ...make sure they're floats
		rowMinima1 = minimum(cont_cov_row1, dims = 2) # Find the row minimum (minimum value of a covariate)
		rowMinima2 = minimum(cont_cov_row2, dims = 2) # Find the row minimum (minimum value of a covariate)
		rowMaxima1 = maximum(cont_cov_row1, dims = 2) # Find the row maximum (maximum value of a covariate)
		rowMaxima2 = maximum(cont_cov_row2, dims = 2) # Find the row maximum (maximum value of a covariate)
		cont_min_max1 = (rowMinima1, rowMaxima1) # Create tuple for output
		cont_min_max2 = (rowMinima2, rowMaxima2) # Create tuple for output
		norm_cov_row1 = (cont_cov_row1 .- rowMinima1) ./ (rowMaxima1 .- rowMinima1) .+ 1 # Normalize the covariate to values between 1 and 2
		norm_cov_row2 = (cont_cov_row2 .- rowMinima1) ./ (rowMaxima1 .- rowMinima1) .+ 1 # Normalize the covariate to values between 1 and 2
	end
	discCovIndx = FindRegexinArray(ops1[:regex_disc], geneColumn) # Find the discontinuous covariate rows
	# println(discCovIndx)
	discCov1 = missing # Init
	discCov2 = missing # Init
	disc_cov_labels1 = missing # Init
	disc_cov_labels2 = missing # Init
	disc_full_onehot1 = missing # Init
	disc_full_onehot2 = missing # Init
	onehot_redundance_removed1 = missing # Init
	onehot_redundance_removed2 = missing # Init
	onehot_redundance_removed_transpose1 = missing # Init
	onehot_redundance_removed_transpose2 = missing # Init
	if length(discCovIndx) > 0

		if length(discCovIndx) > 1
			print_statement = "$(length(discCovIndx)) DISCONTINUOUS COVARIATES."
		else
			print_statement = "1 DISCONTINUOUS COVARIATE."
		end

		my_info(print_statement)

		discCov1 = Array(dataFile1[discCovIndx, 2:end]) # select discontinuous covariate rows
		discCov2 = Array(dataFile2[discCovIndx, 2:end]) # select discontinuous covariate rows
		disc_cov_labels1 = mapslices(x -> [unique(x)], discCov1, dims = 2) # Find unique labels
		disc_cov_labels2 = mapslices(x -> [unique(x)], discCov1, dims = 2) # Find unique labels
		disc_full_onehot1 = [Int.(onehotbatch(discCov1[ii, :], disc_cov_labels1[ii])) for ii in 1:size(discCov1, 1)] # Create Onehot
		disc_full_onehot2 = [Int.(onehotbatch(discCov2[ii, :], disc_cov_labels1[ii])) for ii in 1:size(discCov2, 1)] # Create Onehot
		onehot_redundance_removed1 = map(x -> x[2:end, :], disc_full_onehot1)
		onehot_redundance_removed2 = map(x -> x[2:end, :], disc_full_onehot2)
		onehot_redundance_removed_transpose1 = map(x -> permutedims(x), onehot_redundance_removed1)
		onehot_redundance_removed_transpose2 = map(x -> permutedims(x), onehot_redundance_removed2)
	end

	n_cov = length(contCovIndx) + length(discCovIndx)
	my_info("$n_cov TOTAL COVARIATES.")

	first_expression_row = maximum(vcat(contCovIndx, discCovIndx, 0)) + 1
	# dictionary_keys_long = (:first_expression_row, :discontinuous_covariate_labels, :discontinuous_covariates, :discontinuous_covariate_onehot, :discontinuous_covariate_onehot_redundance_removed, :discontinuous_covariate_onehot_redundance_removed_transpose, :continuous_covariates, :normalized_continuous_covariates, :continuous_covariate_minima_and_maxima)
	dictionary_keys_short = (:o_fxr, :o_dcl, :o_dc, :o_dco, :o_dcorr, :o_dcorrt, :o_cc, :o_ncc, :o_ccmm)
	dictionary_values1 = (first_expression_row, disc_cov_labels1, discCov1, disc_full_onehot1, onehot_redundance_removed1, onehot_redundance_removed_transpose1, cont_cov_row1, norm_cov_row1, cont_min_max1)
	dictionary_values2 = (first_expression_row, disc_cov_labels2, discCov2, disc_full_onehot2, onehot_redundance_removed2, onehot_redundance_removed_transpose2, cont_cov_row2, norm_cov_row2, cont_min_max2)
	map(dictionary_keys_short, dictionary_values1) do new_key, new_value
		ops1[new_key] = new_value
	end
	
	map(dictionary_keys_short, dictionary_values2) do new_key, new_value
		ops2[new_key] = new_value
	end

	return ops1, ops2
end

function findXinY(search_x, in_y)
	@debug "Using the map function, finding the index of search_x in_y in the order they appear in search_x."
	indices = vcat(map(z -> findall(in([z]), in_y), search_x)...)
	return indices
end

function findXinYinX(search_x, in_y)
	length(search_x) == length(unique(search_x)) || error("\"search_x\" must be vector with unique elements")

    indices = map(x -> findall(in_y .== x), search_x)

    if .&((length.(indices[length.(indices) .> 0]) .== 1)...)
        match_indices = vcat(indices...)
        matches = in_y[match_indices]
        output_dataframe = DataFrame(IndexOfXinY = match_indices)
    else
		matchable_indices = indices[length.(indices) .>= 1]
        match_indices = map(x -> x[1], matchable_indices)
        matches = in_y[match_indices]
        output_dataframe = DataFrame(IndexOfXinY = matchable_indices)
    end
    reverse_indices = vcat(map(x -> findall(search_x .== x), matches)...)
	#=
	println(size(matches))
	println(size(output_dataframe[:, :IndexOfXinY]))
	println(size(reverse_indices))
	=#
    output_dataframe[:, :IndexOfXinYinX] = reverse_indices
    output_dataframe[:, :Matches] = matches
	
	return output_dataframe
	#=
	indices_make_sense = (in_y[vcat(output_dataframe[:, :IndexOfXinY]...)] == search_x[output_dataframe[:, :IndexOfXinYinX]])
    matches_line_up = (in_y[vcat(output_dataframe[:, :IndexOfXinY]...)] == output_dataframe[:, :Matches]) & (search_x[output_dataframe[:, :IndexOfXinYinX]] == output_dataframe[:, :Matches])

    if indices_make_sense & matches_line_up
        print_positive_check_result && println("\nChecks passed.\n")
        return output_dataframe
    else
        if indices_make_sense
            error("Matches do not line up")
        elseif matches_line_up
            error("Indices do not make sense")
        else
            error("Indices do not make sense and Matches do not line up")
        end
    end
	=#
end

function findXinYinX(search_x, in_y, keep_empty::Bool)
	length(search_x) == length(unique(search_x)) || error("\"search_x\" must be vector with unique elements")

    indices = map(x -> findall(in_y .== x), search_x)

	at_least_one_match = length.(indices) .>= 1
	matchable_indices = indices[at_least_one_match]
	match_indices = map(x -> x[1], matchable_indices)
	matches = in_y[match_indices]
	output_dataframe = DataFrame(IndexOfXinY = indices)
	reverse_indices = Int64.(zeros(length(search_x)))
	reverse_indices[at_least_one_match] = vcat(map(x -> findall(search_x .== x), matches)...)
    output_dataframe[:, :IndexOfXinYinX] = reverse_indices
    output_dataframe[:, :Matches] = search_x
	
	return output_dataframe
end

function GetMeanNormalizedData(keepGenesOfInterestExpressionData, ops)
	
	my_info("PERFORMING DISPERSION NORMALIZATION.")
	
	meanNormalizedData = keepGenesOfInterestExpressionData
	if ops[:norm_gene_level]
		meanNormalizedData = MeanNormalize(keepGenesOfInterestExpressionData)
		# if ops[:norm_disc] & (length(ops[:o_dcl]) > 0)
		if ops[:norm_disc] & !(ismissing(ops[:o_dcl]))
			n_cov = length(ops[:norm_disc_cov])
			test_n_cov = (n_cov > 1)
			!test_n_cov || error("Only a single covariate can be used. Please specify one, you have specified $n_cov: $(ops[:norm_disc_cov]).")
			normCovOnehot = ops[:o_dco][ops[:norm_disc_cov]]
			dataByCovariates = [keepGenesOfInterestExpressionData[:, Bool.(normCovOnehot[ll, :])] for ll in 1:size(normCovOnehot, 1)]
			meanNormalizedData = hcat(MeanNormalize.(dataByCovariates)...)
		end
	end
	return meanNormalizedData
end

function GetMeanNormalizedData(keepGenesOfInterestExpressionData1, keepGenesOfInterestExpressionData2, ops)
	
	my_info("PERFORMING DISPERSION NORMALIZATION ACCORDING TO A PREVIOUS DATASET.")

	meanNormalizedData1 = keepGenesOfInterestExpressionData1
	meanNormalizedData2 = keepGenesOfInterestExpressionData2
	if ops[:norm_gene_level]
		meanNormalizedData1, meanNormalizedData2 = MeanNormalize(keepGenesOfInterestExpressionData1, keepGenesOfInterestExpressionData2)
	end
	return meanNormalizedData1, meanNormalizedData2
end

function SVDTransform!(expression_data, ops)
	
	my_info("TRANSFORMING SEED GENES INTO EIGEN SPACE.")

	svd_obj_l = svd(expression_data)
	ops[:o_svd_S] = Array{Float32,1}(svd_obj_l.S)
	ops[:o_svd_U] = Array{Float32,2}(svd_obj_l.U)
	ops[:o_svd_V] = Array{Float32,2}(svd_obj_l.V)
	V = Array{Float64,2}(svd_obj_l.V)
	Vt = Array{Float64,2}(svd_obj_l.Vt)
	S = Array{Float32,1}(svd_obj_l.S)
	cumvar = cumsum(S .^2, dims = 1) / sum(S .^2)
	dimvar = vcat(cumvar[1], diff(cumvar))
	ops[:o_svd_cumvar] = cumvar
	ops[:o_svd_dimvar] = dimvar
	return V, Vt, S, cumvar, svd_obj_l
end

function SVDBatchRegression!(V, Vt, ops)
	
	# if ops[:eigen_reg] & (length(ops[:o_dcl])>0)
	if ops[:eigen_reg] & !(ismissing(ops[:o_dcl]))
		my_info("PERFORMING REGRESSION AGAINST DISCONTINUOUS COVARIATES.")
		
		#~~~Error check~~~#
		n_cov = length(ops[:eigen_reg_disc_cov])
		only_one_disc_cov = n_cov > 1
		!only_one_disc_cov || error("Only a single covariate can be used. Please specify one, you have provided an array with length $n_cov.")
		
		#~~~Terms for linear regression~~~#
		regCovT = ops[:o_dcorrt][ops[:eigen_reg_disc_cov]]
		#~~~Terms for prediction~~~#
		regCov = ops[:o_dcorr][ops[:eigen_reg_disc_cov]]
		#~~~Number of rows in linear regression model (also number of samples in data set)~~~#
		n_rows = size(regCovT, 2)
		# println("dims of regCovT = $(size(regCovT))")
		# println("dims of V = $(size(V))")
		#~~~Linear regression against discontinous covariate~~~#
		Linear_regression_models = llsq(Float64.(regCovT), V, bias = true)
		#~~~Calculate the Total sum of squared errors~~~#
		SSE_total = sum((Vt .- mean(Vt, dims = 2)).^2, dims = 2)[:, 1]
		#~~~Use the linear regression model to predict values to calculate the residual sum of squared erros~~~#
		predicted_values =  Linear_regression_models[1:size(regCovT, 2), :]' * regCov .+ Linear_regression_models[end, :]
		#~~~Calculate residual sum of squared errors~~~#
		SSE_residual = sum((Vt .- predicted_values).^2, dims = 2)[:, 1]
		#~~~Calculate r squared~~~#
		R_squared_values = 1 .- (SSE_residual ./ SSE_total)
		#~~~Record R squared values into dictionary~~~#
		ops[:o_svd_Rsquared] = R_squared_values
	end

end

function SVDRegressionExclusion(S, ops)

	if haskey(ops, :o_svd_Rsquared)
		R_squared = ops[:o_svd_Rsquared]
		S_logical = R_squared .< ops[:eigen_reg_r_squared_cutoff]
		singular_or_plural = sum(.!(S_logical)) == 1
		gene_or_genes = singular_or_plural ? "GENE" : "GENES"
		excluded_index_list = join(findall(.!(S_logical)), ", ", " AND ")
		excluded_r_squared_list = join(trunc.(R_squared[.!(S_logical)], digits = 3), ", ", " AND ")
		has_or_have = singular_or_plural ? "HAS" : "HAVE"
		respectively = singular_or_plural ? "" : ", RESPECTIVELY"
		if ops[:eigen_reg_exclude]
			my_info("REMOVING EIGEN GENES WITH R SQUARED GREATER THAN $(trunc(ops[:eigen_reg_r_squared_cutoff], digits = 3)).")
			if sum(.!(S_logical)) < 1
				my_info("NO EIGEN GENES EXCLUDED BY R SQUARED.")
			else
				my_warn("EXCLUDING EIGEN $gene_or_genes $excluded_index_list, WHICH $has_or_have AN R SQUARED VALUE OF $excluded_r_squared_list$respectively.")
			end
			Reduce_S = S[S_logical]
		elseif ops[:eigen_reg_remove_correct]
			my_info("EXCLUDING VARIANCE CONTRIBUTED BY EIGEN GENES WITH R SQUARED GREATER THAN $(trunc(ops[:eigen_reg_r_squared_cutoff], digits = 3)).")
			if sum(.!(S_logical)) < 1
				my_info("NO VARIANCE OF EIGEN GENES EXCLUDED BY R SQUARED.")
			else
				my_warn("EXCLUDING VARIANCE OF EIGEN $gene_or_genes $excluded_index_list, WHICH $has_or_have AN R SQUARED VALUE OF $excluded_r_squared_list$respectively.")
			end
			S[.!(S_logical)] .= 0.0
			Reduce_S = S
			S_logical = trues(length(S))
		else
			my_info("REDUCING VARIANCE CONTRIBUTED BY EIGEN GENES WITH R SQUARED GREATER THAN $(trunc(ops[:eigen_reg_r_squared_cutoff], digits = 3)).")
			if sum(.!(S_logical)) < 1
				my_info("NO VARIANCE OF EIGEN GENES REDUCED BY R SQUARED.")
			else
				my_warn("REDUCING VARIANCE OF EIGEN $gene_or_genes $excluded_index_list, WHICH $has_or_have AN R SQUARED VALUE OF $excluded_r_squared_list$respectively.")
			end
			S[.!(S_logical)] .*= (1 .- R_squared[.!(S_logical)])
			Reduce_S = S
			S_logical = trues(length(S))
		end
	else
		Reduce_S = S
		S_logical = trues(length(S))
	end
	
	dimension_indices = SVDReduceDimensions(Reduce_S, S_logical, ops)

	return dimension_indices
end

function SVDReduceDimensions(S, S_logical, ops)
	
	if ops[:eigen_max] < 2
		my_error("FATAL INPUT ERROR. :eigen_max CANNOT BE LESS THAN 2 BUT HAS BEEN DEFINED AS $(ops[:eigen_max]). PLEASE INCREASE :eigen_max.")
	end

	cumvar = cumsum(S .^2, dims = 1) / sum(S .^2) # S are the singular values, sorted in descending order. Find the Fraction variance that an eigengene and each eigengene before it makes up from the total variance.
	ops[:o_svd_cumvar_corrected] = cumvar
	vardiff = vcat(cumvar[1], diff(cumvar)) # Find the difference between each added eigengene
	ops[:o_svd_dimvar_corrected] = vardiff
	ReductionDim1 = findfirst(x -> x > ops[:eigen_total_var], cumvar) # How many eigengenes need to be included to have captured the minimum (user specified) variance from the eigengenes (from their singular values)
	my_info("$ReductionDim1 DIMENSIONS REQUIRED TO CAPTURE $(trunc(100 * ops[:eigen_total_var], digits = 1))% OF THE REMAINING VARIANCE.")
	if ReductionDim1 < 2
		my_error("THE TOTAL VARIANCE LIMIT REDUCES THE # OF EIGEN GENE DIMENSIONS TO $ReductionDim1. PLEASE INCREASE :eigen_total_var.")
	end
	# println("\nS[1] = $(S[1])")
    ReductionDim2 = findlast(x -> x > ops[:eigen_contr_var], vardiff) # which eigengenes contribute the minimum (user specified) variance
	less_than_10_or_greater_than_20 = (ReductionDim2 < 10) | (ReductionDim2 > 20)
	if less_than_10_or_greater_than_20
		number_suffix = mod(ReductionDim2, 10) == 1 ? "st" : (mod(ReductionDim2, 10) == 2 ? "nd" : (mod(ReductionDim2, 10) == 3 ? "rd" : "th"))
	else
		number_suffix = "th"
	end
	my_info("THE $ReductionDim2$number_suffix DIMENSION IS THE LAST DIMENSION TO CONTRIBUTE AT LEAST $(trunc(100 * ops[:eigen_contr_var], digits = 1))% VARIANCE.")
	if ReductionDim2 < 2
		my_error("THE INDIVIDUAL VARIANCE LIMIT REDUCES THE # OF EIGEN GENE DIMENSIONS TO $ReductionDim2. PLEASE DECREASE :eigen_contr_var.")
	end
	if ReductionDim2 < ReductionDim1
		my_warn("CONTRIBUTED VARIANCE OF INDIVIDUAL EIGENGENES IS LIMITING THE TOTAL VARIANCE CAPTURED")
		if ops[:eigen_var_override]
			my_warn("OVERRIDING MINIMUM CONTRIBUTED VARIANCE REQUIREMENT ($ReductionDim2 EIGENGENES) FOR MINIMUM CAPTURED VARIANCE REQUIREMENT ($ReductionDim1 EIGENGENES).\nSET :eigen_var_override TO FALSE IF YOU DO NOT WISH TO DO SO")
			ReductionDim2 = ReductionDim1
		end
	end
	ReductionDim = trunc(Int, min(ReductionDim1, ReductionDim2, ops[:eigen_max])) # The last criteria is the maximum number of eigengenes (user specified) that will be kept. Whichever is the smallest is the number of eigengenes kept

	all_S_indeces = findall(S_logical)

	keep_eigen_gene_row_index = all_S_indeces[1:ReductionDim]

	ops[:o_svd_n_dims] = ReductionDim
	ops[:o_svd_dim_indices] = keep_eigen_gene_row_index

	my_info("KEEPING $ReductionDim DIMENSIONS, NAMELY DIMENSIONS $(join(keep_eigen_gene_row_index, ", ", " AND ")).")
	return keep_eigen_gene_row_index
end

function CovariateOnehotEncoder!(Transform, ops)
	#=
	are_there_discontinuous_covariates = (length(ops[:o_dco]) > 0)
	are_there_continuous_covariates = (length(ops[:o_cc]) > 0)
	are_there_covariates = are_there_discontinuous_covariates | are_there_continuous_covariates
	are_covariates_being_used  = ops[:out_covariates]
	are_discontinuous_covariates_used = ops[:out_use_disc_cov]
	are_all_discontinuous_covariates_used = ops[:out_all_disc_cov]
	which_discontinuous_covariates_are_used = ops[:out_disc_cov]
	are_more_than_one_discontinuous_covariate_used = length(ops[:out_disc_cov]) > 1
	are_continuous_covariates_used = ops[:out_use_cont_cov]
	are_all_continuous_covariates_used = ops[:out_all_cont_cov]
	are_continuous_covariates_normalized = ops[:out_use_norm_cont_cov]
	are_all_continuous_covariates_normalized = ops[:out_all_norm_cont_cov]
	which_continuous_covariates_are_used = ops[:out_cont_cov]
	which_continuous_covariates_are_normalized = ops[:out_norm_cont_cov]

	if are_there_covariates & are_covariates_being_used
		scale_array = Array{Any}([])
		if are_there_discontinuous_covariates & are_discontinuous_covariates_used
			if are_all_discontinuous_covariates_used # | 
			end
		end
	end
	=#
	Transform_copy = deepcopy(Transform)
	are_there_discontinuous_covariates = !(ismissing(ops[:o_dco]))
	are_there_continuous_covariates = !(ismissing(ops[:o_cc]))
	# if ops[:out_covariates] & ((length(ops[:o_dco]) > 0) | (length(ops[:o_cc]) > 0))
	if ops[:out_covariates] & (are_there_discontinuous_covariates | are_there_continuous_covariates)
		my_info("ADDING COVARIATES TO EIGEN GENES.")
		scale_array = Array{Any}([])
		# if (length(ops[:o_dcl]) > 0) & ops[:out_use_disc_cov]
		if are_there_discontinuous_covariates & ops[:out_use_disc_cov]
			my_info("ADDING DISCONTINUOUS COVARIATES TO EIGEN GENES.")
			if ops[:out_all_disc_cov] | (length(ops[:o_dcl]) == 1)
				Transform_copy = vcat(Transform_copy, ops[:o_dcorr]...)
				# Calculate standard deviation for each discontinuous covariate group
				for ii in ops[:o_dco]
					batch_std = mapslices(x -> [std(Transform[:, Bool.(x)], dims = 2)], ii, dims = 2)
					scales = map(x -> (batch_std[1] ./ x) .- 1, batch_std[2:end])
					append!(scale_array, scales)					
				end
			else
				more_than_one_covariate = length(ops[:out_disc_cov]) > 1
				if more_than_one_covariate
					Transform_copy = vcat(Transform_copy, ops[:o_dcorr][ops[:out_disc_cov]]...)
				else
					Transform_copy = vcat(Transform_copy, ops[:o_dcorr][ops[:out_disc_cov]])
				end
				if more_than_one_covariate
					for ii in ops[:o_dco][ops[:out_disc_cov]]
						batch_std = mapslices(x -> [std(Transform[:, Bool.(x)], dims = 2)], ii, dims = 2)
						scales = map(x -> (batch_std[1] ./ x) .- 1, batch_std[2:end])
						append!(scale_array, scales)					
					end
				else
					batch_std = mapslices(x -> [std(Transform[:, Bool.(x)], dims = 2)], ops[:o_dco][ops[:out_disc_cov]], dims = 2)
					scales = map(x -> (batch_std[1] ./ x) .- 1, batch_std[2:end])
					append!(scale_array, scales)
				end
			end
		end
		# if ops[:out_use_cont_cov] & (length(ops[:o_cc]) > 0)
		if ops[:out_use_cont_cov] & are_there_continuous_covariates
			my_info("ADDING CONTINUOUS COVARIATES TO EIGEN GENES.")
			if ops[:out_all_cont_cov]
				if ops[:out_use_norm_cont_cov]
					my_info("ADDING NORMALIZED CONTINUOUS COVARIATES TO EIGEN GENES.")
					if ops[:out_all_norm_cont_cov]
						Transform_copy = vcat(Transform_copy, ops[:o_ncc])
					else
						for cci in 1:size(ops[:o_cc], 1)
							if cci in ops[:out_norm_cont_cov]
								Transform_copy = vcat(Transform_copy, ops[:o_ncc][cci, 1:end])
								continue
							end
							Transform_copy = vcat(Transform_copy, ops[:o_cc][cci, 1:end])
						end
					end
				else
					Transform_copy = vcat(Transform_copy, ops[:o_cc])
				end
				scales = [zeros(ops[:o_svd_n_dims], size(ops[:o_cc], 1)) .- 1]
				append!(scale_array, scales)
			else
				if ops[:out_use_norm_cont_cov]
					my_info("ADDING NORMALIZED CONTINUOUS COVARIATES TO EIGEN GENES.")
					if ops[:out_all_norm_cont_cov]
						Transform_copy = vcat(Transform_copy, ops[:o_ncc][ops[:out_cont_cov], 1:end])
					else
						for cci in ops[:out_cont_cov]
							if cci in ops[:out_norm_cont_cov]
								Transform_copy = vcat(Transform_copy, ops[:o_ncc][cci, 1:end])
								continue
							end
							Transform_copy = vcat(Transform_copy, ops[:o_cc][cci, 1:end])
						end
					end
				else
					Transform_copy = vcat(Transform_copy, ops[:o_cc][ops[:out_cont_cov], 1:end])
				end
				scales = [zeros(ops[:o_svd_n_dims], length(ops[:out_cont_cov])) .- 1]
				append!(scale_array, scales)
			end
		end
		ops[:S_OH] = hcat(scale_array...)
		if ops[:init_scale_change]
			my_warn("BATCH SCALE FACTORS WILL BE ALTERED. SET :init_scale_change TO false IF THIS IS NOT THE DESIRED BEHAVIOR.")
			if ops[:init_scale_1]
				my_warn("BATCH SCALE FACTORS HAVE BEEN ALTERED TO 1.")
				ops[:S_OH] = zeros(size(ops[:S_OH]))
			else
				my_warn("BATCH SCALE FACTORS HAVE BEEN ALTERED TO HALFWAY BETWEEN THEIR INITIAL GUESS AND 1.")
				ops[:S_OH] = ops[:S_OH] ./ 2
			end
		end
		# if !((length(ops[:o_dcl]) > 0) & ops[:out_use_disc_cov]) & (!ops[:out_use_cont_cov] & (length(ops[:o_cc]) > 0))
		if !(are_there_discontinuous_covariates & ops[:out_use_disc_cov]) & (!ops[:out_use_cont_cov] & are_there_continuous_covariates)
			my_error("It appears there are conflicting inputs. Please check the \":out_\" keys relating to covariates.")
		else
			ops[:o_covariates] = Array{Float32,2}(permutedims(Transform_copy[ops[:o_svd_n_dims]+1:end, :]))
		end
	end
	return Array{Float32}(Transform_copy)
end

function Eigengenes!(dataFile, genesOfInterest, ops)
	my_info("BEGIN DATA PREPROCESSING")

	CovariateProcessing!(dataFile, ops) # returns updated options. Look at function CovariateProcessing to see the long hand for each of the new keys added to the dictionary

	bluntedDataFile = BluntXthPercentile(dataFile, ops, OUT_TYPE=DataFrame)
	
	keepGenesOfInterestExpressionData, ~ = GeneLevelCutoffs(bluntedDataFile, genesOfInterest, ops)

	meanNormalizedData = GetMeanNormalizedData(keepGenesOfInterestExpressionData, ops)

	V, Vt, S, cumvar, ~ = SVDTransform!(meanNormalizedData, ops)

	# reg_S = SVDRegression!(V, Vt, S, cumvar, ops)
	
	# Transform, ~ = SVDCollectDims!(Vt, reg_S, ops)

	SVDBatchRegression!(V, Vt, ops)

	dimension_indices = SVDRegressionExclusion(S, ops)

	Transform = 10 * Vt[dimension_indices, :]

	OHTransform = CovariateOnehotEncoder!(Transform, ops)

	return OHTransform
end

function SVDtransfer(data, svd_obj)
	S = svd_obj.S
    U = svd_obj.U

    S_inverse = Diagonal(S)^-1
    U_transpose = transpose(U)

    new_data_transformed = S_inverse * U_transpose * data

    return new_data_transformed
end

function Eigengenes!(dataFile1, dataFile2, genesOfInterest, ops, matching_symbols::Bool)
	
	if !matching_symbols
		symbols_of_1_in_2 = findXinY(dataFile1[:, 1], dataFile2[:, 1])

		dataFile2_overlap = dataFile2[symbols_of_1_in_2, :]
	else
		dataFile2_overlap = dataFile2
	end

	alternateops = DefaultDict(ops)

	ops1, ops2 = CovariateProcessing(dataFile1, dataFile2_overlap, alternateops) # returns updated options. Look at function CovariateProcessing to see the long hand for each of the new keys added to the dictionary
	# CovariateProcessing!(dataFile2, ops2) # returns updated options. Look at function CovariateProcessing to see the long hand for each of the new keys added to the dictionary
	
	bluntedDataFile1, bluntedDataFile2 = BluntXthPercentile(dataFile1, dataFile2_overlap, ops1, OUT_TYPE=DataFrame)
	# bluntedDataFile2 = BluntXthPercentile(dataFile2, ops2, OUT_TYPE=Float32)

	keepGenesOfInterestExpressionData1, gene_of_interest_indices = GeneLevelCutoffs(bluntedDataFile1, genesOfInterest, ops1)
	ops2[:o_seed_genes] = ops1[:o_seed_genes]
	keepGenesOfInterestExpressionData2 = Array{Float64}(bluntedDataFile2[gene_of_interest_indices .+ ops2[:o_fxr] .- 1, 2:end])

	meanNormalizedData1, meanNormalizedData2 = GetMeanNormalizedData(keepGenesOfInterestExpressionData1, keepGenesOfInterestExpressionData2, ops1)
	# meanNormalizedData2 = GetMeanNormalizedData(keepGenesOfInterestExpressionData2, ops2)

	V1, Vt1, S1, cumvar1, svd_obj1 = SVDTransform!(meanNormalizedData1, ops1)
	ops2[:o_svd_S] = ops1[:o_svd_S]
	ops2[:o_svd_U] = ops1[:o_svd_U]
	ops2[:o_svd_cumvar] = ops1[:o_svd_cumvar]
	ops2[:o_svd_dimvar] = ops1[:o_svd_dimvar]
	Vt2 = SVDtransfer(meanNormalizedData2, svd_obj1)
	ops2[:o_svd_V] = transpose(Vt2)

	SVDBatchRegression!(V1, Vt1, ops1)

	# reg_S1 = SVDRegression!(V1, Vt1, S1, cumvar1, ops1)

	dimension_indices = SVDRegressionExclusion(S1, ops1)

	Transform1 = 10 * Vt1[dimension_indices, :]

	# Transform1, ReductionDim = SVDCollectDims!(Vt1, reg_S1, ops1)
	ops2[:o_svd_n_dims] = ops1[:o_svd_n_dims]
	ops2[:o_svd_dim_indices] = ops1[:o_svd_dim_indices]
	Transform2 = 10 * Vt2[dimension_indices, :]

	# println("Has :S_OH key: ops1 $(haskey(ops1, :S_OH)) ops2 $(haskey(ops2, :S_OH))")

	OHTransform1 = CovariateOnehotEncoder!(Transform1, ops1)
	OHTransform2 = CovariateOnehotEncoder!(Transform2, ops2)

	return (OHTransform1, ops1), (OHTransform2, ops2)
end

function Eigengenes_d1!(dataFile1, dataFile2, genesOfInterest, ops, matching_symbols)
	
	if !matching_symbols
		symbols_of_1_in_2 = findXinY(dataFile1[:, 1], dataFile2[:, 1])
		dataFile2_overlap = dataFile2[symbols_of_1_in_2, :]
	else
		dataFile2_overlap = dataFile2
	end

	ops1 = DefaultDict(ops)
	ops2 = DefaultDict(ops)

	CovariateProcessing!(dataFile1, ops1)
	CovariateProcessing!(dataFile2_overlap, ops2) # returns updated options. Look at function CovariateProcessing to see the long hand for each of the new keys added to the dictionary
	
	bluntedDataFile1 = BluntXthPercentile(dataFile1, ops1, OUT_TYPE=DataFrame)
	bluntedDataFile2 = BluntXthPercentile(dataFile2_overlap, ops2, OUT_TYPE=DataFrame)

	keepGenesOfInterestExpressionData1, gene_of_interest_indices = GeneLevelCutoffs(bluntedDataFile1, genesOfInterest, ops1)
	ops2[:o_seed_genes] = ops1[:o_seed_genes]
	keepGenesOfInterestExpressionData2 = Array{Float64}(bluntedDataFile2[gene_of_interest_indices .+ ops2[:o_fxr] .- 1, 2:end])

	meanNormalizedData1 = GetMeanNormalizedData(keepGenesOfInterestExpressionData1, ops1)
	meanNormalizedData2 = GetMeanNormalizedData(keepGenesOfInterestExpressionData2, ops2)

	V1, Vt1, S1, cumvar1, svd_obj1 = SVDTransform!(meanNormalizedData1, ops1)
	ops2[:o_svd_S] = ops1[:o_svd_S]
	ops2[:o_svd_U] = ops1[:o_svd_U]
	ops2[:o_svd_cumvar] = ops1[:o_svd_cumvar]
	ops2[:o_svd_dimvar] = ops1[:o_svd_dimvar]
	Vt2 = SVDtransfer(meanNormalizedData2, svd_obj1)
	ops2[:o_svd_V] = transpose(Vt2)

	SVDBatchRegression!(ops2[:o_svd_V], Vt2, ops2)

	dimension_indices = SVDRegressionExclusion(ops2[:o_svd_S], ops2)

	Transform2 = 10 * Vt2[dimension_indices, :]

	OHTransform2 = CovariateOnehotEncoder!(Transform2, ops2)

	return OHTransform2, ops2
end

function Eigengenes_d1_reapply!(dataFile1, dataFile2, dataFile3, genesOfInterest, ops, matching_symbols)
	
	if !matching_symbols
		symbols_of_1_in_2 = findXinY(dataFile1[:, 1], dataFile2[:, 1])
		dataFile2_overlap = dataFile2[symbols_of_1_in_2, :]
		symbols_of_1_in_3 = findXinY(dataFile1[:, 1], dataFile3[:, 1])
		dataFile3_overlap = dataFile3[symbols_of_1_in_3, :]
	else
		dataFile2_overlap = dataFile2
		dataFile3_overlap = dataFile3
	end

	ops1 = DefaultDict(ops)
	ops2 = DefaultDict(ops)

	CovariateProcessing!(dataFile1, ops1)
	ops2, ops3 = CovariateProcessing(dataFile2_overlap, dataFile3_overlap, ops2) # returns updated options. Look at function CovariateProcessing to see the long hand for each of the new keys added to the dictionary
	
	bluntedDataFile1 = BluntXthPercentile(dataFile1, ops1, OUT_TYPE=DataFrame)
	bluntedDataFile2, bluntedDataFile3 = BluntXthPercentile(dataFile2_overlap, dataFile3_overlap, ops2, OUT_TYPE=DataFrame)

	keepGenesOfInterestExpressionData1, gene_of_interest_indices = GeneLevelCutoffs(bluntedDataFile1, genesOfInterest, ops1)
	ops2[:o_seed_genes] = ops1[:o_seed_genes]
	ops3[:o_seed_genes] = ops1[:o_seed_genes]
	keepGenesOfInterestExpressionData2 = Array{Float64}(bluntedDataFile2[gene_of_interest_indices .+ ops2[:o_fxr] .- 1, 2:end])
	keepGenesOfInterestExpressionData3 = Array{Float64}(bluntedDataFile3[gene_of_interest_indices .+ ops3[:o_fxr] .- 1, 2:end])

	meanNormalizedData1 = GetMeanNormalizedData(keepGenesOfInterestExpressionData1, ops1)
	~, meanNormalizedData3 = GetMeanNormalizedData(keepGenesOfInterestExpressionData2, keepGenesOfInterestExpressionData3, ops2)

	V1, Vt1, S1, cumvar1, svd_obj1 = SVDTransform!(meanNormalizedData1, ops1)
	ops3[:o_svd_S] = ops1[:o_svd_S]
	ops3[:o_svd_U] = ops1[:o_svd_U]
	ops3[:o_svd_cumvar] = ops1[:o_svd_cumvar]
	ops3[:o_svd_dimvar] = ops1[:o_svd_dimvar]
	Vt3 = SVDtransfer(meanNormalizedData3, svd_obj1)
	ops3[:o_svd_V] = transpose(Vt3)

	SVDBatchRegression!(V1, Vt1, ops1)

	dimension_indices = SVDRegressionExclusion(ops1[:o_svd_S], ops1)

	Transform3 = 10 * Vt3[dimension_indices, :]

	ops3[:o_svd_n_dims] = ops1[:o_svd_n_dims]

	OHTransform3 = CovariateOnehotEncoder!(Transform3, ops3)

	return OHTransform3, ops3
end

function Eigengenes_Seed_Intersect_d1_svd!(dataFile1, dataFile2, genesOfInterest, ops, test)
	
	symbols_of_1_in_2 = findXinY(dataFile1[:, 1], dataFile2[:, 1])

	dataFile2_overlap = dataFile2[symbols_of_1_in_2, :]

	ops1 = DefaultDict(ops)
	ops2 = DefaultDict(ops)

	CovariateProcessing!(dataFile1, ops1)
	CovariateProcessing!(dataFile2_overlap, ops2) # returns updated options. Look at function CovariateProcessing to see the long hand for each of the new keys added to the dictionary
	
	bluntedDataFile1 = BluntXthPercentile(dataFile1, ops1, OUT_TYPE=DataFrame)
	bluntedDataFile2 = BluntXthPercentile(dataFile2_overlap, ops2, OUT_TYPE=DataFrame)

	keepGenesOfInterestExpressionData1, gene_of_interest_indices1 = GeneLevelCutoffs(bluntedDataFile1, genesOfInterest, ops1)
	keepGenesOfInterestExpressionData2, gene_of_interest_indices2 = GeneLevelCutoffs(bluntedDataFile2, genesOfInterest, ops2)
	intersect_rows = intersect(gene_of_interest_indices1, gene_of_interest_indices2)
	ops1[:o_seed_genes] = dataFile1[intersect_rows .+ ops1[:o_fxr] .- 1,1]
	ops2[:o_seed_genes] = ops1[:o_seed_genes]
	
	keepGenesOfInterestExpressionData1 = Array{Float64}(bluntedDataFile1[intersect_rows .+ ops1[:o_fxr] .- 1, 2:end])
	keepGenesOfInterestExpressionData2 = Array{Float64}(bluntedDataFile2[intersect_rows .+ ops2[:o_fxr] .- 1, 2:end])

	meanNormalizedData1 = GetMeanNormalizedData(keepGenesOfInterestExpressionData1, ops1)
	meanNormalizedData2 = GetMeanNormalizedData(keepGenesOfInterestExpressionData2, ops2)

	V1, Vt1, S1, cumvar1, svd_obj1 = SVDTransform!(meanNormalizedData1, ops1)
	ops2[:o_svd_S] = ops1[:o_svd_S]
	ops2[:o_svd_U] = ops1[:o_svd_U]
	ops2[:o_svd_cumvar] = ops1[:o_svd_cumvar]
	ops2[:o_svd_dimvar] = ops1[:o_svd_dimvar]
	Vt2 = SVDtransfer(meanNormalizedData2, svd_obj1)
	ops2[:o_svd_V] = transpose(Vt2)

	SVDBatchRegression!(ops2[:o_svd_V], Vt2, ops2)

	dimension_indices = SVDRegressionExclusion(ops2[:o_svd_S], ops2)

	Transform2 = 10 * Vt2[dimension_indices, :]

	OHTransform2 = CovariateOnehotEncoder!(Transform2, ops2)

	return OHTransform2, ops2
end

function Eigengenes_Seed_Intersect_d2_svd!(dataFile1, dataFile2, genesOfInterest, ops, test)
	
	symbols_of_1_in_2 = findXinY(dataFile1[:, 1], dataFile2[:, 1])

	dataFile2_overlap = dataFile2[symbols_of_1_in_2, :]

	ops1 = DefaultDict(ops)
	ops2 = DefaultDict(ops)

	CovariateProcessing!(dataFile1, ops1)
	CovariateProcessing!(dataFile2_overlap, ops2) # returns updated options. Look at function CovariateProcessing to see the long hand for each of the new keys added to the dictionary
	
	bluntedDataFile1 = BluntXthPercentile(dataFile1, ops1, OUT_TYPE=DataFrame)
	bluntedDataFile2 = BluntXthPercentile(dataFile2_overlap, ops2, OUT_TYPE=DataFrame)

	keepGenesOfInterestExpressionData1, gene_of_interest_indices1 = GeneLevelCutoffs(bluntedDataFile1, genesOfInterest, ops1)
	keepGenesOfInterestExpressionData2, gene_of_interest_indices2 = GeneLevelCutoffs(bluntedDataFile2, genesOfInterest, ops2)
	intersect_rows = intersect(gene_of_interest_indices1, gene_of_interest_indices2)
	ops1[:o_seed_genes] = dataFile1[intersect_rows .+ ops1[:o_fxr] .- 1,1]
	ops2[:o_seed_genes] = ops1[:o_seed_genes]
	
	keepGenesOfInterestExpressionData2 = Array{Float64}(bluntedDataFile2[intersect_rows .+ ops2[:o_fxr] .- 1, 2:end])

	meanNormalizedData2 = GetMeanNormalizedData(keepGenesOfInterestExpressionData2, ops2)

	V2, Vt2, S2, cumvar2, svd_obj2 = SVDTransform!(meanNormalizedData2, ops2)

	SVDBatchRegression!(V2, Vt2, ops2)

	dimension_indices = SVDRegressionExclusion(S2, ops2)

	Transform2 = 10 * Vt2[dimension_indices, :]

	OHTransform2 = CovariateOnehotEncoder!(Transform2, ops2)

	return OHTransform2, ops2
end

function MetricCovariateConcatenator(metrics, options, _verbose = false)
	if size(metrics, 2) > 12
		metricNames = vec(Symbol.(vcat(["ID", "Phase", "Error", "Magnitude", "ProjectionX" , "ProjectionY", "Inner_Error", "Circular_Error", "Modified_Inner_Error", "Modified_Error", "Out_of_Plane_Error", "Out_of_Plane_Reconstruction_Error", "Input_Magnitude", "Dense_Input_Magnitude", "Decoding_Dense_Output_Magnitude", "Model_Output_Magnitude", "Modified_Decoding_Dense_Output_Magnitude", "Modified_Model_Output_Magnitude"])))
	else
		metricNames = vec(Symbol.(vcat(["ID", "Phase", "Error", "Magnitude", "ProjectionX" , "ProjectionY", "Circular_Error", "Modified_Error", "Out_of_Plane_Reconstruction_Error", "Input_Magnitude", "Model_Output_Magnitude", "Modified_Model_Output_Magnitude"])))
	end
	metric_covariate_names = Symbol[]
	are_there_discontinuous_covariates = !(ismissing(options[:o_dc]))
	are_there_continuous_covariates = !(ismissing(options[:o_cc]))
	if options[:o_fxr] >= 2
		# if (size(options[:o_dc], 1) > 0) & (size(options[:o_cc], 1) > 0)
		if are_there_discontinuous_covariates & are_there_continuous_covariates
			!_verbose || println("\tDISCONTINUOUS AND CONTINUOUS COVARIATES EXIST\n\n")
			metric_covariates = permutedims(vcat(options[:o_dc], options[:o_cc]))
			metric_covariate_names = vec(Symbol.(vcat(repeat(["Covariate_D"], size(options[:o_dc], 1)), repeat(["Covariate_C"], size(options[:o_cc], 1)))))
		elseif are_there_discontinuous_covariates
			!_verbose || println("\tONLY DISCONTINUOUS COVARIATES EXIST\n\n")
			metric_covariates = permutedims(options[:o_dc])
			metric_covariate_names = vec(Symbol.(repeat(["Covariate_D"], size(options[:o_dc], 1))))
		else
			!_verbose || println("\tONLY CONTINUOUS COVARIATES EXIST\n\n")
			metric_covariates = permutedims(options[:o_cc])
			metric_covariate_names = vec(Symbol.(repeat(["Covariate_C"], size(options[:o_cc], 1))))
		end
		!_verbose || println("\tADD COVARIATES TO FIT OUTPUT\n\n")
		metrics = hcat(metrics, metric_covariates)
		!_verbose || println("\tADD COVARIATE NAMES TO FIT OUTPUT\n\n")
		metricNames = vcat(metricNames, metric_covariate_names)
	end
	metricDataframe = DataFrame(Matrix(metrics), metricNames, makeunique = true)
	return metricDataframe, length(metric_covariate_names)
end

function Fit(dataFile, genesOfInterest, alternateOps = Dict(); _verbose = false)
	
	options = DefaultDict(alternateOps)
	eigen_data = Eigengenes!(dataFile, genesOfInterest, options)
	
	Random.seed!(1234);
	initialized_models = InitializeModel(eigen_data, options)
	
	no_covariates = (size(eigen_data, 1) == options[:o_svd_n_dims])
	
	if no_covariates
		trained_models = MultiTrainOrder(initialized_models, eigen_data, options)
		best_model, metrics_array = OrderDecoder(trained_models, eigen_data)
	else
		trained_models_and_errors = MultiTrainCovariates(initialized_models, eigen_data, options)
		best_model, metrics_array = CovariatesDecoder(trained_models_and_errors, eigen_data)
	end 

	metrics = hcat(names(dataFile)[2:end], metrics_array)
	# return metrics
	metricDataframe, n_cov = MetricCovariateConcatenator(metrics, options, _verbose)
	
	eigengene_metric_pearson_correlations = cor(Float32.(Matrix(metricDataframe[:,2:end-n_cov])), eigen_data[1:best_model.o,:]')

	correlationDataframe_no_row_names = DataFrame(eigengene_metric_pearson_correlations, [Symbol("Eigengene$ii") for ii in 1:best_model.o])

	correlationDataframe = hcat(DataFrame(MetricName = names(metricDataframe)[2:end-n_cov]), correlationDataframe_no_row_names)

	return eigen_data, metricDataframe, correlationDataframe, best_model, options
end

function ReApplyFit(trainedModel, dataFile1, dataFile2, genesOfInterest, ops; _verbose=false, matching_symbols=true)
	
	options = DefaultDict(ops)
	(dataFile1_transform, dataFile1_ops), (dataFile2_transform, dataFile2_ops) = Eigengenes!(dataFile1, dataFile2, genesOfInterest, options, matching_symbols)

	no_covariates = (size(dataFile2_transform, 1) == dataFile2_ops[:o_svd_n_dims])

	if no_covariates
		best_model, output_phases = OrderDecoder([trainedModel], dataFile2_transform)
		output_overall_mses = Array{Float32, 1}(reshape(mapslices(x -> mse(trainedModel(x), x[1:trainedModel.o]), dataFile2_transform, dims = 1), :, 1)[:, 1])
		output_projections = OrderProjection(trainedModel, dataFile2_transform)
		output_magnitudes = OrderMagnitude(trainedModel, dataFile2_transform)
	else
		best_model, metric_array = CovariatesDecoder(trainedModel, dataFile2_transform)
	end 

	metrics = hcat(names(dataFile2)[2:end], metric_array)

	metricDataframe, n_cov = MetricCovariateConcatenator(metrics, dataFile2_ops, _verbose)
	
	eigengene_metric_pearson_correlations = cor(Float32.(Matrix(metricDataframe[:,2:end-n_cov])), dataFile2_transform[1:best_model.o,:]')

	correlationDataframe_no_row_names = DataFrame(eigengene_metric_pearson_correlations, [Symbol("Eigengene$ii") for ii in 1:best_model.o])

	correlationDataframe = hcat(DataFrame(MetricName = names(metricDataframe)[2:end-n_cov]), correlationDataframe_no_row_names)

	return dataFile2_transform, metricDataframe, correlationDataframe, best_model, dataFile2_ops
end

function ReApplyFit_d1(trainedModel, dataFile1, dataFile2, dataFile3, genesOfInterest, ops; _verbose=false, matching_symbols=true)
	
	options = DefaultDict(ops)
	dataFile3_transform, dataFile3_ops = Eigengenes_d1_reapply!(dataFile1, dataFile2, dataFile3, genesOfInterest, options, matching_symbols)

	no_covariates = (size(dataFile3_transform, 1) == dataFile3_ops[:o_svd_n_dims])

	if no_covariates
		best_model, output_phases = OrderDecoder([trainedModel], dataFile3_transform)
		output_overall_mses = Array{Float32, 1}(reshape(mapslices(x -> mse(trainedModel(x), x[1:trainedModel.o]), dataFile3_transform, dims = 1), :, 1)[:, 1])
		output_projections = OrderProjection(trainedModel, dataFile3_transform)
		output_magnitudes = OrderMagnitude(trainedModel, dataFile3_transform)
	else
		best_model, metric_array = CovariatesDecoder(trainedModel, dataFile3_transform)
	end 

	metrics = hcat(names(dataFile3)[2:end], metric_array)

	metricDataframe, n_cov = MetricCovariateConcatenator(metrics, dataFile3_ops, _verbose)
	
	eigengene_metric_pearson_correlations = cor(Float32.(Matrix(metricDataframe[:,2:end-n_cov])), dataFile3_transform[1:best_model.o,:]')

	correlationDataframe_no_row_names = DataFrame(eigengene_metric_pearson_correlations, [Symbol("Eigengene$ii") for ii in 1:best_model.o])

	correlationDataframe = hcat(DataFrame(MetricName = names(metricDataframe)[2:end-n_cov]), correlationDataframe_no_row_names)

	return dataFile3_transform, metricDataframe, correlationDataframe, best_model, dataFile3_ops
end

function TransferFit(trainedModel, dataFile1::DataFrame, dataFile2::DataFrame, genesOfInterest::Array{String,1}, ops::Dict{Symbol,Any}; _verbose=false, matching_symbols=true)
	~, (dataFile2_transform, dataFile2_ops) = Eigengenes!(dataFile1, dataFile2, genesOfInterest, ops, matching_symbols)

	no_covariates = (size(dataFile2_transform, 1) == dataFile2_ops[:o_svd_n_dims])

	if no_covariates
		transfer_trained_model = MultiTrainOrder([trainedModel], dataFile2_transform, dataFile2_ops)
		~, output_phases = OrderDecoder(transfer_trained_model, dataFile2_transform)
		output_overall_mses = Array{Float32, 1}(reshape(mapslices(x -> mse(transfer_trained_model(x), x[1:transfer_trained_model.o]), dataFile2_transform, dims = 1), :, 1)[:, 1])
		output_projections = OrderProjection(transfer_trained_model, dataFile2_transform)
		output_magnitudes = OrderMagnitude(transfer_trained_model, dataFile2_transform)
	else
		transfer_trained_model = MultiTrainCovariates([trainedModel], dataFile2_transform, dataFile2_ops)
		~, model_metrics = CovariatesDecoder(transfer_trained_model, dataFile2_transform)
	end

	metrics = hcat(names(dataFile2)[2:end], model_metrics)

	metricDataframe, n_cov = MetricCovariateConcatenator(metrics, dataFile2_ops, _verbose)

	eigengene_metric_pearson_correlations = cor(Float32.(Matrix(metricDataframe[:,2:end-n_cov])), dataFile2_transform[1:best_model.o,:]')

	correlationDataframe_no_row_names = DataFrame(eigengene_metric_pearson_correlations, [Symbol("Eigengene$ii") for ii in 1:best_model.o])

	correlationDataframe = hcat(DataFrame(MetricName = names(metricDataframe)[2:end-n_cov]), correlationDataframe_no_row_names)

	return dataFile2_transform, metricDataframe, correlationDataframe, best_model, dataFile2_ops
end

function TransferFit_d1(dataFile1::DataFrame, dataFile2::DataFrame, genesOfInterest::Array{String,1}, ops::Dict{Symbol,Any}; _verbose=false)
	options = DefaultDict(ops)
	dataFile2_transform, dataFile2_ops = Eigengenes_d1!(dataFile1, dataFile2, genesOfInterest, options, true)

	Random.seed!(1234);
	initialized_models = InitializeModel(dataFile2_transform, dataFile2_ops)
	
	no_covariates = (size(dataFile2_transform, 1) == dataFile2_ops[:o_svd_n_dims])

	if no_covariates
		transfer_trained_models = MultiTrainOrder(initialized_models, dataFile2_transform, dataFile2_ops)
		best_model, model_metrics = OrderDecoder(transfer_trained_models, eigen_data)
	else
		transfer_trained_models = MultiTrainCovariates(initialized_models, dataFile2_transform, dataFile2_ops)
		best_model, model_metrics = CovariatesDecoder(transfer_trained_models, dataFile2_transform)
	end

	metrics = hcat(names(dataFile2)[2:end], model_metrics)

	metricDataframe, n_cov = MetricCovariateConcatenator(metrics, dataFile2_ops, _verbose)

	eigengene_metric_pearson_correlations = cor(Float32.(Matrix(metricDataframe[:,2:end-n_cov])), dataFile2_transform[1:best_model.o,:]')

	correlationDataframe_no_row_names = DataFrame(eigengene_metric_pearson_correlations, [Symbol("Eigengene$ii") for ii in 1:best_model.o])

	correlationDataframe = hcat(DataFrame(MetricName = names(metricDataframe)[2:end-n_cov]), correlationDataframe_no_row_names)

	return dataFile2_transform, metricDataframe, correlationDataframe, best_model, dataFile2_ops
end

function TransferFit_Intersect_d1_svd(dataFile1::DataFrame, dataFile2::DataFrame, genesOfInterest::Array{String,1}, ops::Dict{Symbol,Any}; _verbose=false)
	options = DefaultDict(ops)
	dataFile2_transform, dataFile2_ops = Eigengenes_Seed_Intersect_d1_svd!(dataFile1, dataFile2, genesOfInterest, options, true)

	Random.seed!(1234);
	initialized_models = InitializeModel(dataFile2_transform, dataFile2_ops)
	
	no_covariates = (size(dataFile2_transform, 1) == dataFile2_ops[:o_svd_n_dims])

	if no_covariates
		transfer_trained_models = MultiTrainOrder(initialized_models, dataFile2_transform, dataFile2_ops)
		best_model, output_phases = OrderDecoder(transfer_trained_models, dataFile2_transform)
		output_overall_mses = Array{Float32, 1}(reshape(mapslices(x -> mse(best_model(x), x[1:best_model.o]), dataFile2_transform, dims = 1), :, 1)[:, 1])
		output_projections = OrderProjection(best_model, dataFile2_transform)
		output_magnitudes = OrderMagnitude(best_model, dataFile2_transform)
	else
		transfer_trained_models = MultiTrainCovariates(initialized_models, dataFile2_transform, dataFile2_ops)
		best_model, model_metrics = CovariatesDecoder(transfer_trained_models, dataFile2_transform)
	end

	metrics = hcat(names(dataFile2)[2:end], model_metrics)

	metricDataframe, n_cov = MetricCovariateConcatenator(metrics, dataFile2_ops, _verbose)

	eigengene_metric_pearson_correlations = cor(Float32.(Matrix(metricDataframe[:,2:end-n_cov])), dataFile2_transform[1:best_model.o,:]')

	correlationDataframe_no_row_names = DataFrame(eigengene_metric_pearson_correlations, [Symbol("Eigengene$ii") for ii in 1:best_model.o])

	correlationDataframe = hcat(DataFrame(MetricName = names(metricDataframe)[2:end-n_cov]), correlationDataframe_no_row_names)

	return dataFile2_transform, metricDataframe, correlationDataframe, best_model, dataFile2_ops
end

function TransferFit_Intersect_d2_svd(dataFile1::DataFrame, dataFile2::DataFrame, genesOfInterest::Array{String,1}, ops::Dict{Symbol,Any}; _verbose=false)
	options = DefaultDict(ops)
	dataFile2_transform, dataFile2_ops = Eigengenes_Seed_Intersect_d2_svd!(dataFile1, dataFile2, genesOfInterest, options, true)

	Random.seed!(1234);
	initialized_models = InitializeModel(dataFile2_transform, dataFile2_ops)
	
	no_covariates = (size(dataFile2_transform, 1) == dataFile2_ops[:o_svd_n_dims])

	if no_covariates
		transfer_trained_models = MultiTrainOrder(initialized_models, dataFile2_transform, dataFile2_ops)
		best_model, output_phases = OrderDecoder(transfer_trained_models, dataFile2_transform)
		output_overall_mses = Array{Float32, 1}(reshape(mapslices(x -> mse(best_model(x), x[1:best_model.o]), dataFile2_transform, dims = 1), :, 1)[:, 1])
		output_projections = OrderProjection(best_model, dataFile2_transform)
		output_magnitudes = OrderMagnitude(best_model, dataFile2_transform)
	else
		transfer_trained_models = MultiTrainCovariates(initialized_models, dataFile2_transform, dataFile2_ops)
		best_model, model_metrics = CovariatesDecoder(transfer_trained_models, dataFile2_transform)
	end

	metrics = hcat(names(dataFile2)[2:end], model_metrics)

	metricDataframe, n_cov = MetricCovariateConcatenator(metrics, dataFile2_ops, _verbose)

	eigengene_metric_pearson_correlations = cor(Float32.(Matrix(metricDataframe[:,2:end-n_cov])), dataFile2_transform[1:best_model.o,:]')

	correlationDataframe_no_row_names = DataFrame(eigengene_metric_pearson_correlations, [Symbol("Eigengene$ii") for ii in 1:best_model.o])

	correlationDataframe = hcat(DataFrame(MetricName = names(metricDataframe)[2:end-n_cov]), correlationDataframe_no_row_names)

	return dataFile2_transform, metricDataframe, correlationDataframe, best_model, dataFile2_ops
end

##############################
# Cross Validation Functions #
##############################
function MeanCircError(prediction, truth)
    errors = acos.(cos.(truth - prediction))
	output_mean = mean(errors)
    return output_mean
end

function STDCircError(prediction, truth)
	errors = acos.(cos.(truth - prediction))
	output_STD = std(errors)
	return output_STD
end

function XFit(eigen_data, genesOfInterest, options, full_fitoutput, sample_indices; _verbose = false)
	!_verbose || println("\tINITIALIZING MODELS\n\n")
	no_covariates = (size(eigen_data, 1) == options[:o_svd_n_dims])
	initialized_models = InitializeModel(eigen_data, options)

	!_verbose || println("\tTRAINING MODELS ON PARTIAL DATA\n\n")
	if no_covariates
		trained_models = MultiTrainOrder(initialized_models, eigen_data[:, sample_indices], options)
		best_model, output_phases = OrderDecoder(trained_models, eigen_data)
		output_overall_mses = Array{Float32, 1}(reshape(mapslices(x -> mse(best_model(x), x[1:best_model.o]), eigen_data, dims = 1), :, 1)[:, 1])
		output_projections = OrderProjection(best_model, eigen_data)
		output_magnitudes = OrderMagnitude(best_model, eigen_data)
	else
		trained_models_and_errors = MultiTrainCovariates(initialized_models, eigen_data[:, sample_indices], options)
		best_model, output_phases, output_projections, output_magnitudes, output_overall_mses = CovariatesDecoder(trained_models_and_errors, eigen_data)
	end

	!_verbose || println("\tCOMBINE CALCULATED METRICS\n\n")
	metrics = hcat(full_fitoutput.ID, output_phases, output_overall_mses, output_projections, output_magnitudes)
	metricDataframe = MetricCovariateConcatenator(metrics, options, _verbose)

	return eigen_data, metricDataframe, best_model, options
end

function CheckPath!(path_to_check)
	if !isdir(path_to_check)
		mkdir(path_to_check)
	end
end

function CheckPath(path_to_check)
	if !isdir(path_to_check)
		mkdir(path_to_check)
	else
		path_to_check = join([path_to_check, "1"], "_")
		CheckPath(path_to_check)
	end
	return path_to_check
end

function OutputFolders(ouput_path, ops)
	println("\tCREATING OUTPUT FOLDER\n\n")
	CheckPath!(ouput_path)
	todays_date = replace(string(floor(now(), Dates.Minute(1))), ":"=>"_")
	default_change_folder_string = CheckDefaultDict(ops)
	default_change_folder_string = replace(default_change_folder_string, "."=>"_")
	path_extentsion = join([todays_date, default_change_folder_string], "_")
	master_output_folder_path = joinpath(ouput_path, path_extentsion)
	master_output_folder_path = CheckPath(master_output_folder_path)
	println("\tOUTPUTS WILL BE SAVED IN $(master_output_folder_path)\n\n")
	all_subfolder_paths = Array{Any}([])
	# subfolders is a constant defined within the CYCLOPS module as const subfolders = ["Plots", "Fits", "Models", "Parameters"]
	for folder_name in subfolders
		sub_output_folder_path = joinpath(master_output_folder_path, folder_name)
		CheckPath!(sub_output_folder_path)
		append!(all_subfolder_paths, [sub_output_folder_path])
	end
	return todays_date, all_subfolder_paths
end

function CheckDefaultDict(ops)

	changed_key_value = String[]

	default_dict = DefaultDict()
	default_dict_keys = collect(keys(default_dict))

	for ii in default_dict_keys
		if ops[ii] != default_dict[ii]
			string_changed_key = string.(ii)
			value_ii = ops[ii]
			string_changed_value = string.(value_ii)		
			append!(changed_key_value, [join([string_changed_key, value_ii], "_")])
		end
	end

	sort!(changed_key_value)

	final_string = join(changed_key_value, "_")

	return final_string
end

function CircError!(list_1_phases_l, list_2_phases_l, greater_list_1_list_l, greater_list_2_list_l)
	my_info("CALCULATE MEAN CIRCULAR ERROR AND STANDARD DEVIATION.")
	sample_error = MeanCircError(list_1_phases_l, list_2_phases_l)
	append!(greater_list_1_list_l, sample_error)
	sample_std_error = STDCircError(list_1_phases_l, list_2_phases_l)
	append!(greater_list_2_list_l, sample_std_error)
end

function F_J_Rho_Stats!(list_1_phases_l, list_2_phases_l, greater_list_1_list_l, greater_list_2_list_l, greater_list_3_list_l)
	my_info("CALCULATE CIRCULAR CORRELATIONS.")
	sample_F_stats = Fischer_Circular_CorrelationMeasures(list_1_phases_l, list_2_phases_l)
	append!(greater_list_1_list_l, [sample_F_stats])
	sample_J_stats = Jammalamadka_Circular_CorrelationMeasures(list_1_phases_l, list_2_phases_l)
	append!(greater_list_2_list_l, [sample_J_stats])
	sample_phase_correlation = cor(list_1_phases_l, list_2_phases_l)
	append!(greater_list_3_list_l, sample_phase_correlation)
end


function CrossValidationFinal(dataFile, genesOfInterest, options = Dict(); _verbose = false, project_path = pwd())

	# my_info("Time Bias is $(options[:train_collection_time_balance])")
	my_info("BEGIN TRAINING FULL MODEL.")
	full_eigen_data, full_fitoutput, full_correlation_output, full_model, full_options = Fit(dataFile, genesOfInterest, options)
	all_column_ids = full_options[:o_column_ids]
	my_info("FULL MODEL TRAINING COMPLETE.")

	my_info("ALIGN FULL MODEL OUTPUTS AS SPECIFIED.")
	output_path_info = Align(dataFile, full_fitoutput, full_correlation_output, full_model, full_options, project_path)
	todays_date, all_subfolder_paths = output_path_info
	full_model_phases = Float64.(full_fitoutput.Phase)

	my_info("INITIALIZE SAMPLE METRIC ARRAYS.")
	sample_error_metric = Array{Any, 1}([])
	sample_error_metric_std = Array{Any, 1}([])
	sample_F_Stat_collection = Array{Any, 1}([])
	sample_J_Stat_collection = Array{Any, 1}([])
	sample_Pearson_collection = Array{Any, 1}([])

	my_info("INITIALIZE LEFT OUT SAMPLE METRIC ARRAYS.")
	lo_sample_error_metric = Array{Any, 1}([])
	lo_sample_error_metric_std = Array{Any, 1}([])
	lo_sample_F_Stat_collection = Array{Any, 1}([])
	lo_sample_J_Stat_collection = Array{Any, 1}([])
	lo_sample_Pearson_collection = Array{Any, 1}([])
		
	# my_info("INITIALIZE ACROPHASE METRIC ARRAYS")
	# acrophase_error_metric = Array{Any, 1}([])
	# acrophase_error_metric_std = Array{Any, 1}([])
	# acrophase_F_Stat_collection = Array{Any, 1}([])
	# acrophase_J_Stat_collection = Array{Any, 1}([])
	# acrophase_Pearson_collection = Array{Any, 1}([])
	# n_mouse_atlas_acrophases = Array{Any, 1}([])
	
	covariates_for_batch_to_split_by = Bool.(full_options[:o_dco][full_options[:out_disc_cov][1]])

	if haskey(full_options, :train_sample_id)
		my_info("INITIALIZE LEFT OUT SAMPLE METRIC ARRAYS.")
		lo_sample_wtt_error_metric = Array{Any, 1}([])
		lo_sample_wtt_error_metric_std = Array{Any, 1}([])
		lo_sample_wtt_F_Stat_collection = Array{Any, 1}([])
		lo_sample_wtt_J_Stat_collection = Array{Any, 1}([])
		lo_sample_wtt_Pearson_collection = Array{Any, 1}([])

		trained_sample_wtt_error_metric = Array{Any, 1}([])
		trained_sample_wtt_error_metric_std = Array{Any, 1}([])
		trained_sample_wtt_F_Stat_collection = Array{Any, 1}([])
		trained_sample_wtt_J_Stat_collection = Array{Any, 1}([])
		trained_sample_wtt_Pearson_collection = Array{Any, 1}([])

		samples_with_collection_times = findXinY(full_options[:train_sample_id], all_column_ids)
		samples_without_collection_times = setdiff(1:length(all_column_ids), samples_with_collection_times)
		covariates_for_batch_to_split_by_with_collection_times = covariates_for_batch_to_split_by[:, samples_with_collection_times]
		indices_with_collection_times = vcat(mapslices(x -> [findall(x)], covariates_for_batch_to_split_by_with_collection_times, dims = 2)...)
		how_many_samples_to_keep_with_collection_times = Int.(round.(length(indices_with_collection_times) .* (1 - full_options[:X_Val_omit_size])))
	else
		samples_without_collection_times = 1:length(all_column_ids)
	end

	covariates_for_batch_to_split_by_without_collection_times= covariates_for_batch_to_split_by[:, samples_without_collection_times]

	all_batch_sizes_without_collection_times = sum(covariates_for_batch_to_split_by_without_collection_times, dims = 2)
	how_many_samples_to_keep_each_batch_without_collection_times = Int.(round.(all_batch_sizes_without_collection_times .* (1 - full_options[:X_Val_omit_size])))
	corrected_samples_to_keep_each_batch_without_collection_times = map(x -> maximum([2, x]), how_many_samples_to_keep_each_batch_without_collection_times)
	
	each_batch_indices_without_collection_times = mapslices(x -> [findall(x)], covariates_for_batch_to_split_by_without_collection_times, dims = 2)

	all_shifted_partial_model_phases = ones(length(all_column_ids))

	my_info("BEGIN CROSS VALIDATION.")
	for ii in 1:full_options[:X_Val_k]
		my_info("TRAINING FOLD $ii.")
					
		# my_info("CREATE DICTIONARY FOR TRAINING AND ALIGNING PARTIAL MODELS"
		partial_options = Dict(full_options)
		
		my_info("GET SAMPLE IDS FOR FOLD $(ii).")
		# my_info("Number of eigen gene dimensions with covariates = $(full_options[:o_svd_n_dims]). Number of eigen gene dimensions = $(size(full_eigen_data, 1)).")
		# if (full_options[:o_svd_n_dims] < size(full_eigen_data, 1)) & (length(full_options[:o_dco]) > 0)
		if (full_options[:o_svd_n_dims] < size(full_eigen_data, 1)) & !(ismissing(full_options[:o_dco]))
			
			if haskey(full_options, :train_sample_id) & full_options[:train_collection_times]
				
				my_info("TRUE TIMES USED FOR TRAINING.")				
				all_indices_for_this_fold = Array{Any}([])
				map(each_batch_indices_without_collection_times, corrected_samples_to_keep_each_batch_without_collection_times) do these_indices, how_many
					if length(these_indices) > 0
						append!(all_indices_for_this_fold, sample(these_indices, how_many, replace=false))
					end
				end
				
				my_info("SAMPLE INDIVIDUAL BATCHES LEAVING OUT $(full_options[:X_Val_omit_size]).")
				samples_without_known_times_this_fold = sort(all_indices_for_this_fold)
				my_info("SAMPLE DATA POINTS WITH ASSOCIATED TIMES LEAVING OUT $(full_options[:X_Val_omit_size]).")
				samples_with_known_times_this_fold = sort(sample(indices_with_collection_times, how_many_samples_to_keep_with_collection_times, replace=false))
				append!(all_indices_for_this_fold, samples_with_known_times_this_fold)
				sort!(all_indices_for_this_fold)
				
				partial_options[:o_column_ids] = all_column_ids[all_indices_for_this_fold]	
				matched_train_sample_id_indices = findXinY(all_column_ids[samples_with_known_times_this_fold], full_options[:train_sample_id])
				partial_options[:train_sample_id] = full_options[:train_sample_id][matched_train_sample_id_indices]
				partial_options[:train_sample_phase] = full_options[:train_sample_phase][matched_train_sample_id_indices]
			else
				my_info("SAMPLE INDIVIDUAL BATCHES LEAVING OUT $(full_options[:X_Val_omit_size]).")
				all_batch_sizes = sum(covariates_for_batch_to_split_by, dims = 2)
				how_many_samples_to_keep_each_batch = Int.(round.(all_batch_sizes .* (1 - full_options[:X_Val_omit_size])))
				corrected_samples_to_keep_each_batch = map(x -> maximum([2, x]), how_many_samples_to_keep_each_batch)
				each_batch_indices = mapslices(x -> [findall(x)], covariates_for_batch_to_split_by, dims = 2)
				all_indices_for_this_fold = Array{Any}([])
				map(each_batch_indices, corrected_samples_to_keep_each_batch) do these_indices, how_many
					append!(all_indices_for_this_fold, sample(these_indices, how_many, replace=false))
				end
				partial_options[:o_column_ids] = all_column_ids[all_indices_for_this_fold]
			end
		elseif haskey(full_options, :train_sample_id) & full_options[:train_collection_times]
			my_info("TRUE TIMES USED FOR TRAINING REDUCED.")
			sample_ids_wo_known_times = setdiff(all_column_ids, full_options[:train_sample_id])
			num_samples_wo_kt = length(sample_ids_wo_known_times)
			num_samples_w_kt = length(full_options[:train_sample_id])
			omit_n_samples_wo_kt = Int(round(full_options[:X_Val_omit_size] * num_samples_wo_kt))
			omit_n_samples_w_kt = Int(round(full_options[:X_Val_omit_size] * num_samples_w_kt))
			keep_n_samples_wo_kt = num_samples_wo_kt - omit_n_samples_wo_kt
			keep_n_samples_w_kt = num_samples_w_kt - omit_n_samples_w_kt
			
			ids_wo_collection_time_sampled = sample(sample_ids_wo_known_times, keep_n_samples_wo_kt, replace = false)
			ids_w_collection_time_sampled = sample(full_options[:train_sample_id], keep_n_samples_w_kt, replace = false)
			all_ids_for_this_fold = vcat(ids_wo_collection_time_sampled, ids_w_collection_time_sampled)
			all_indices_for_this_fold = sort(findXinY(all_ids_for_this_fold, all_column_ids))

			samples_w_collection_times_indices = sort(findXinY(ids_w_collection_time_sampled, full_options[:train_sample_id]))

			partial_options[:o_column_ids] = all_column_ids[all_indices_for_this_fold]
			partial_options[:train_sample_id] = ids_w_collection_time_sampled
			partial_options[:train_sample_phase] = full_options[:train_sample_phase][samples_w_collection_times_indices]
		else
			my_info("SAMPLE LEAVING OUT $(full_options[:X_Val_omit_size]).")
			num_samples = length(all_column_ids)
			keep_n_samples = Int(round(full_options[:X_Val_omit_size] * num_samples))
			all_ids_for_this_fold = sample(all_column_ids, keep_n_samples, replace = false)
			all_indices_for_this_fold = sort(findXinY(all_ids_for_this_fold, all_column_ids))
			partial_options[:o_column_ids] = all_column_ids[all_indices_for_this_fold]
		end

		my_info("TRAIN MODEL $ii ON PARTIAL EIGEN DATA AND PREDICT ON FULL EIGEN DATA.")
		# my_info("Time Bias is $(partial_options[:train_collection_time_balance]).")
		partial_eigen_data, partial_fitoutput, best_partial_model, partial_options = XFit(full_eigen_data, genesOfInterest, partial_options, full_fitoutput, all_indices_for_this_fold)

		partial_options[:o_column_ids] = all_column_ids
		partial_options[:align_samples], partial_options[:align_phases] = full_fitoutput.ID, full_model_phases
		
		my_info("ALIGN PARTIAL MODEL $ii SAMPLE PHASES TO FULL MODEL SAMPLE PHASES.")
		shifted_sample_phases = XAlign(dataFile, full_fitoutput, partial_fitoutput, partial_options, output_path_info, ii) # This is going to show a plot of known sample phase vs estimated sample phase

		all_shifted_partial_model_phases = hcat(all_shifted_partial_model_phases, shifted_sample_phases)

		CircError!(shifted_sample_phases, full_model_phases, sample_error_metric, sample_error_metric_std)

		F_J_Rho_Stats!(shifted_sample_phases, full_model_phases, sample_F_Stat_collection, sample_J_Stat_collection, sample_Pearson_collection)

		sample_indices_left_out = setdiff(1:length(all_column_ids), all_indices_for_this_fold)
		left_out_shifted_sample_phases = shifted_sample_phases[sample_indices_left_out]
		left_out_full_model_phases = full_model_phases[sample_indices_left_out]

		CircError!(left_out_shifted_sample_phases, left_out_full_model_phases, lo_sample_error_metric, lo_sample_error_metric_std)
		
		F_J_Rho_Stats!(left_out_shifted_sample_phases, left_out_full_model_phases, lo_sample_F_Stat_collection, lo_sample_J_Stat_collection, lo_sample_Pearson_collection)

		if haskey(full_options, :train_sample_id)
			sample_indices_with_known_times_left_out = setdiff(samples_with_collection_times, samples_with_known_times_this_fold)
			left_out_shifted_samples_with_known_times = shifted_sample_phases[sample_indices_with_known_times_left_out]
			left_out_sample_IDs = all_column_ids[sample_indices_with_known_times_left_out]
			train_sample_id_indices_left_out = CYCLOPS.findXinY(left_out_sample_IDs, full_options[:train_sample_id])
			left_out_known_sample_phases = full_options[:train_sample_phase][train_sample_id_indices_left_out]
			CircError!(left_out_shifted_samples_with_known_times, left_out_known_sample_phases, lo_sample_wtt_error_metric, lo_sample_wtt_error_metric_std)
			F_J_Rho_Stats!(left_out_shifted_samples_with_known_times, left_out_known_sample_phases, lo_sample_wtt_F_Stat_collection, lo_sample_wtt_J_Stat_collection, lo_sample_wtt_Pearson_collection)

			#=
			samples_with_known_times_used_for_training = shifted_sample_phases[samples_with_known_times_this_fold]
			sample_ids_with_known_times_used_for_trianing = all_column_ids[samples_with_known_times_this_fold]
			train_sample_id_indices_used_for_training = CYCLOPS.findXinY(sample_ids_with_known_times_used_for_trianing, full_options[:train_sample_id])
			known_sample_phases_used_for_training = full_options[:train_sample_phase][train_sample_id_indices_used_for_training]
			CircError!(samples_with_known_times_used_for_training, known_sample_phases_used_for_training, trained_sample_wtt_error_metric, trained_sample_wtt_error_metric_std)
			F_J_Rho_Stats!(samples_with_known_times_used_for_training, known_sample_phases_used_for_training, trained_sample_wtt_F_Stat_collection, trained_sample_wtt_J_Stat_collection, trained_sample_wtt_Pearson_collection)
			=#
		end

	end

	my_info("CROSS VALIDATION COMPLETE")
	if haskey(full_options, :train_collection_times)
		# my_info("left out samples with true times error metric dims = $(size(lo_sample_wtt_error_metric))")
		all_metrics_expanded = (lo_sample_wtt_error_metric, lo_sample_wtt_error_metric_std, lo_sample_wtt_F_Stat_collection, lo_sample_wtt_J_Stat_collection, lo_sample_wtt_Pearson_collection, lo_sample_error_metric, lo_sample_error_metric_std, lo_sample_F_Stat_collection, lo_sample_J_Stat_collection, lo_sample_Pearson_collection, sample_error_metric, sample_error_metric_std, sample_F_Stat_collection, sample_J_Stat_collection, sample_Pearson_collection)
		# all_metrics_collapsed = vcat(map(x -> hcat(x...), all_metrics_expanded)...)
		collapsed_metric_names = ["WTT_Mean_Left_Out_Sample_Circular_Error", "WTT_Mean_Left_Out_Sample_Circular_Error_Standard_Deviation", "WTT_Left_Out_Sample_Fischer_Correlation", "WTT_Left_Out_Sample_Fischer_Correlation_Ranked", "WTT_Left_Out_Sample_Jammalamadka_Correlation", "WTT_Left_Out_Sample_Jammalamadka_Correlation_Uniform", "WTT_Left_Out_Sample_Jammalamadka_Correlation_Ranked", "WTT_Left_Out_Sample_Pearson_Correlation", "Mean_Left_Out_Sample_Circular_Error", "Mean_Left_Out_Sample_Circular_Error_Standard_Deviation", "Left_Out_Sample_Fischer_Correlation", "Left_Out_Sample_Fischer_Correlation_Ranked", "Left_Out_Sample_Jammalamadka_Correlation", "Left_Out_Sample_Jammalamadka_Correlation_Uniform", "Left_Out_Sample_Jammalamadka_Correlation_Ranked", "Left_Out_Sample_Pearson_Correlation", "Mean_Sample_Circular_Error", "Sample_Circular_Error_Standard_Deviation", "Sample_Fischer_Correlation", "Sample_Fischer_Correlation_Ranked", "Sample_Jammalamadka_Correlation", "Sample_Jammalamadka_Correlation_Uniform", "Sample_Jammalamadka_Correlation_Ranked", "Sample_Pearson_Correlation"]
	else
		all_metrics_expanded = (lo_sample_error_metric, lo_sample_error_metric_std, lo_sample_F_Stat_collection, lo_sample_J_Stat_collection, lo_sample_Pearson_collection, sample_error_metric, sample_error_metric_std, sample_F_Stat_collection, sample_J_Stat_collection, sample_Pearson_collection)
		collapsed_metric_names = ["Mean_Left_Out_Sample_Circular_Error", "Mean_Left_Out_Sample_Circular_Error_Standard_Deviation", "Left_Out_Sample_Fischer_Correlation", "Left_Out_Sample_Fischer_Correlation_Ranked", "Left_Out_Sample_Jammalamadka_Correlation", "Left_Out_Sample_Jammalamadka_Correlation_Uniform", "Left_Out_Sample_Jammalamadka_Correlation_Ranked", "Left_Out_Sample_Pearson_Correlation", "Mean_Sample_Circular_Error", "Sample_Circular_Error_Standard_Deviation", "Sample_Fischer_Correlation", "Sample_Fischer_Correlation_Ranked", "Sample_Jammalamadka_Correlation", "Sample_Jammalamadka_Correlation_Uniform", "Sample_Jammalamadka_Correlation_Ranked", "Sample_Pearson_Correlation"]
	end
	# println(size.(all_metrics_expanded))
	# return all_metrics_expanded
	all_metrics_collapsed = vcat(map(x -> hcat(x...), all_metrics_expanded)...)
	metric_headers = hcat("Metric_Name", ["Fold_$(ii)" for ii in 1:full_options[:X_Val_k]]..., "Average")
	descriptive_metrics = hcat(collapsed_metric_names, all_metrics_collapsed, mean(all_metrics_collapsed, dims = 2))
	descriptive_metrics_dataframe = DataFrame(vcat(metric_headers, descriptive_metrics), :auto)
	CSV.write(joinpath(all_subfolder_paths[2], "Cross_Validation_Correlation_and_Error_Metrics_$(todays_date).csv"), descriptive_metrics_dataframe)
	return all_shifted_partial_model_phases[:, 2:end]
end

####################
# Helper Functions #
####################
function sse(ŷ, y; dims::Int = 1)
	output_sse = sum((ŷ .- y).^2, dims = dims)
    return output_sse
end

function CircularDifference(ŷ, y)
	output_circ_diff = acos.(cos.(ŷ .- y))
	return output_circ_diff
end

function CircularDifference(ŷ)
	output_circ_diff = acos.(cos.(ŷ .- ŷ'))
	return output_circ_diff
end

function ±(X, Y)
    X_plus_Y = X .+ Y
    X_minus_Y = X .- Y
    return X_plus_Y, X_minus_Y
end

function SwitchValue(x, y)
	return deepcopy(y), deepcopy(x)
end

function HowManyUniqueGroups(X; dims = 2)
	X_dims = SwitchValue(size(X)...)
	if X_dims[dims] == 0
		return Array{Int}([])
	else
		return vcat(mapslices(Z -> length(unique(Z)), X, dims = dims)...)
	end
end

function x_sort(xs, AO = true)
    xs_sorted = sort(xs, rev = !AO)
    sort_index = vcat(map(x -> findall(in([x]), xs), xs_sorted)...)
    return xs_sorted, sort_index
end
############################
# Cosine Fitting Functions #
############################
function covariates_0_check(ops::Dict{Symbol,Any})
	return covariates_0_check(ops[:o_dc], (ops[:out_all_disc_cov] & !(ismissing(ops[:o_dco]))) || missing, ops[:out_disc_cov], length(ops[:o_column_ids]))
end

function covariates_0_check(raw_covariates::Missing, use_all::Missing, which_cov::Int64, n_samples::Int64)
	return zeros(n_samples, 0)
end

function covariates_0_check(raw_covariates::Missing, use_all::Missing, which_cov::Array{Int64,1}, n_samples::Int64)
	return zeros(n_samples, 0)
end

function covariates_0_check(raw_covariates::Array{T,2}, use_all::Missing, which_cov::Int64, n_samples::Int64) where T <: AbstractString
	return covariates_0_check(raw_covariates[which_cov, :])
end

function covariates_0_check(raw_covariates::Array{T,2}, use_all::Missing, which_cov::Array{Int64,1}, n_samples::Int64) where T <: AbstractString
	return covariates_0_check(raw_covariates[which_cov, :])
end

function covariates_0_check(raw_covariates::Array{T,2}) where T <: AbstractString
	return hcat(mapslices(x -> [covariates_0_check(x)], raw_covariates, dims=2)[:]...)
end

function covariates_0_check(raw_covariates::Array{T,2}, use_all::Bool, which_cov::Int64, n_samples::Int64) where T <: AbstractString
	return covariates_0_check(raw_covariates)
end

function covariates_0_check(raw_covariates::Array{T,2}, use_all::Bool, which_cov::Array{Int64,1}, n_samples::Int64) where T <: AbstractString
	return covariates_0_check(raw_covariates)
end

function covariates_0_check(covariate_row::Array{T,1}) where T <: AbstractString
	return covariates_0_check(permutedims(onehotbatch(covariate_row, unique(covariate_row))))
end

function covariates_0_check(onehotmatrix::Array{Bool,2})
	if size(onehotmatrix, 2) > 1
		return Float64.(onehotmatrix[:,2:end])
	else
		return zeros(size(onehotmatrix, 1), 0)
	end
end

# function covariates_0_check(ops, silence_output = false)
# 	if haskey(ops, :o_covariates)
# 		number_of_samples_per_group = sum(ops[:o_covariates], dims = 1)
# 		zero_groups_logical = vcat((number_of_samples_per_group .== 0)...)
# 		total_number_of_zero_groups = sum(zero_groups_logical)
# 		non_zero_groups_logical = .!(zero_groups_logical)
# 		total_number_of_samples_in_group_1s = [sum(sum(ops[:o_dcorr][ii], dims = 1) .== 0) for ii in 1:length(ops[:o_dcorr])]
# 		is_group_one_zero_logical = total_number_of_samples_in_group_1s .== 0
# 		total_zero_groups = trues(length(is_group_one_zero_logical) + length(zero_groups_logical))
# 		first_group_indices = vcat(0, length.(ops[:o_dcl])[1:end-1]) .+ 1
# 		non_first_group_indices = setdiff(1:length(total_zero_groups), first_group_indices)
# 		total_zero_groups[first_group_indices] .= is_group_one_zero_logical
# 		total_zero_groups[non_first_group_indices] .= zero_groups_logical
		
# 		if total_number_of_zero_groups > 0
# 			usable_covariates = ops[:o_covariates][:, non_zero_groups_logical]
# 			if ops[:out_all_disc_cov]
# 				removed_covariate_labels = vcat(ops[:o_dcl]...)[total_zero_groups]
# 				kept_covariate_labels = vcat(ops[:o_dcl]...)[.!(total_zero_groups)]
# 				my_warn("Leaving out groups $(join(removed_covariate_labels, ", ", " and ")).", silence_output)
# 				my_warn("Remaining groups are $(join(kept_covariate_labels, ", ", " and ")).", silence_output)
# 			elseif ops[:out_use_disc_cov]
# 				removed_covariate_labels = ops[:o_dcl][ops[:out_disc_cov]][vcat(is_group_one_zero_logical, zero_groups_logical)]
# 				kept_covariate_labels = ops[:o_dcl][ops[:out_disc_cov]][vcat(!is_group_one_zero_logical, non_zero_groups_logical)]
# 				my_warn("Leaving out groups $(join(removed_covariate_labels, ", ", " and ")).", silence_output)
# 				my_warn("Remaning groups are $(join(kept_covariate_labels, ", ", " and ")).", silence_output)
# 			end
# 		else
# 			usable_covariates = ops[:o_covariates]
# 		end

# 		if .|(is_group_one_zero_logical...)
# 			usable_covariates = usable_covariates[:, 2:end]
# 		end

# 		return usable_covariates
# 	end
# end

function CosineFit(eP, dataFile, ops)
   	gea = MakeFloat(dataFile[ops[:o_fxr]:end, 2:end], Float64)
	lin_range_shifts = LinRange(0, 2π, ops[:cosine_shift_iterations])   
    all_lin_SSEs = hcat(map(x -> GetLinSSE(eP, gea, ops, x), lin_range_shifts)...)
    best_shift_info = findmin(all_lin_SSEs, dims = 2)
    LinSSEs = best_shift_info[1]
    best_shift_index = map(x -> x[2], best_shift_info[2])
    best_shift = lin_range_shifts[best_shift_index] # made the linrange a variable so that I can index directly. No confusion this way.
    gene_rows = mapslices(x -> [x], gea, dims = 2)
    cosLinSSEs = []
    map(best_shift, gene_rows) do EachShift, EachGene
        append!(cosLinSSEs, GetCosLinSSE(eP, EachGene, ops, EachShift))
    end;	
	if haskey(ops, :o_covariates) & ops[:cosine_covariate_offset]
		usable_covariates = covariates_0_check(ops)
		full_param_number = 3 + size(usable_covariates, 2)
	else
		full_param_number = 3
	end	
    each_gene_f_statistic = ((LinSSEs .- cosLinSSEs) ./ 2) ./ ((cosLinSSEs ./ (length(eP) - full_param_number)))
	nan_f = isnan.(each_gene_f_statistic)
	each_gene_f_statistic[nan_f] .= 0
	#println(f_statistic)
	# println(length(eP) - full_param_number)
	my_FDist = FDist(2.0, length(eP) - full_param_number)
	# println(my_FDist)
	my_cdf = cdf.(my_FDist, each_gene_f_statistic)
	# println(my_cdf)
	p_statistic = vcat((1 .- my_cdf)...)
	bhq_statistic = MultipleTesting.adjust(p_statistic, MultipleTesting.BenjaminiHochberg())
	bonferroni_statistic = MultipleTesting.adjust(p_statistic, MultipleTesting.Bonferroni())
    # p_statistic = 1 .- cdf.(FDist(2, length(eP) - full_param_number), f_statistic)
    cos_SSE, line_attributes = GetCosSSELineAttributes(eP, gea, ops, 0.0)
    SSE_base = sum((gea .- mean(gea, dims = 2)).^2, dims = 2)
    r2 = 1 .- (cos_SSE ./ SSE_base)
	r2[nan_f] .= 0
	line_attributes = hcat(DataFrame(Gene_Symbols = dataFile[ops[:o_fxr]:end, 1]), line_attributes, DataFrame(F_Statistic = vcat(each_gene_f_statistic...), P_Statistic = p_statistic, BHQ_Statistic = bhq_statistic, Bonferroni_Statistic = bonferroni_statistic, R_Squared = vcat(r2...)))
    return line_attributes
end

function GetLinSSE(eP, gea, ops, s::Float64 = 0) 
	x = mod.(eP .- s, 2pi)
	if haskey(ops, :o_covariates) & ops[:cosine_covariate_offset]
		usable_covariates = covariates_0_check(ops)
		b_terms = hcat(ones(length(eP)), usable_covariates)
	else
		b_terms = ones(length(eP))
	end
	llsq_terms = hcat(x, b_terms)
	line_of_best_fit = llsq(llsq_terms, gea', bias = false)
	m_coeffs = line_of_best_fit[1, :]
	b_coeffs = permutedims(line_of_best_fit[2:end, :])
	predicted_values = m_coeffs * x' .+ b_coeffs * b_terms'
	SSE = sse(gea, predicted_values, dims = 2)
	return SSE
end

function GetCosLinSSE(eP, gea_row, ops, s::Float64 = 0)
    lin_x_terms = mod.(eP .- s, 2π)
    cos_x_terms = cos.(lin_x_terms)
    sin_x_terms = sin.(lin_x_terms)
	if haskey(ops, :o_covariates) & ops[:cosine_covariate_offset]
		usable_covariates = covariates_0_check(ops)
		b_terms = hcat(ones(length(eP)), usable_covariates)
	else
		b_terms = ones(length(eP))
	end
	gradient_terms = hcat(lin_x_terms, sin_x_terms, cos_x_terms)
	llsq_terms = hcat(gradient_terms, b_terms)
	line_of_best_fit = llsq(llsq_terms, gea_row, bias = false)
	m_coeffs = line_of_best_fit[1:3]
	b_coeffs = length(line_of_best_fit[4:end]) == 1 ? line_of_best_fit[4] : line_of_best_fit[4:end]
	predicted_values = gradient_terms * m_coeffs .+ b_terms * b_coeffs
	SSE = sse(gea_row, predicted_values, dims = 1)
	return SSE
end

function GetCosSSELineAttributes(eP, gea, ops, s::Float64 = 0)
    lin_x_terms = mod.(eP .- s, 2π)
    sin_x_terms = reshape(sin.(lin_x_terms), :, 1)
    cos_x_terms = reshape(cos.(lin_x_terms), :, 1)
	if haskey(ops, :o_covariates) & ops[:cosine_covariate_offset]
		usable_covariates = covariates_0_check(ops)
		b_terms = hcat(ones(length(eP)), usable_covariates)
	else
		usable_covariates = ones(1,length(eP))
		b_terms = ones(length(eP))
	end
    gradient_terms = hcat(sin_x_terms, cos_x_terms)
    llsq_terms = hcat(gradient_terms, b_terms)
    line_of_best_fit = llsq(llsq_terms, gea', bias = false)
	m_coeffs = permutedims(line_of_best_fit[1:2, :])
    sin_m_coeffs = line_of_best_fit[1, :]
    cos_m_coeffs = line_of_best_fit[2, :]
	b_coeffs = permutedims(line_of_best_fit[3:end, :])
	b_coeffs_for_amp_ratio = deepcopy(b_coeffs)
	if size(b_coeffs, 2) > 1
		b_coeffs_for_amp_ratio[:, 2:end] = b_coeffs_for_amp_ratio[:, 1] .+ b_coeffs_for_amp_ratio[:, 2:end]
	end
	samples_per_batch = sum(usable_covariates, dims=1)
	samples_batch_0 = size(usable_covariates, 2) - sum(samples_per_batch)
	total_samples_per_batch = [samples_batch_0 samples_per_batch]
	total_n_samples = sum(samples_per_batch) + samples_batch_0
	Weighted_Average_Offset = sum(b_coeffs_for_amp_ratio .* total_samples_per_batch, dims=2) ./ total_n_samples
	predicted_values = m_coeffs * gradient_terms' .+ b_coeffs * b_terms'
	SSE = sse(gea, predicted_values, dims = 2)
	Amplitude = sqrt.((sin_m_coeffs .^ 2) .+ (cos_m_coeffs .^ 2))
	# sum([size(covariates_0_check(ops)[(sum(covariates_0_check(ops), dims=2) .== ii)[:], :], 1) for ii in unique(sum(covariates_0_check(ops), dims=2))]) usable_covariates returns
	AmplitudeRatio = Amplitude ./ Weighted_Average_Offset
	OffsetPlusMinusAmplitudeRatio = (Amplitude .+ Weighted_Average_Offset) ./ (Weighted_Average_Offset .- Amplitude)
	Acrophase = mod.(atan.(sin_m_coeffs, cos_m_coeffs), 2pi)
	initialize_attribute_names = ["Sine_Coefficient", "Cosine_Coefficient", "Weighted_Average", "Fit_Average", "Amplitude", "Amplitude_Ratio", "Offset_Plus_Minus_Amplitude_Ratio", "Acrophase"]
	attribute_update_logical = [false, false, false, haskey(ops, :o_covariates) & ops[:cosine_covariate_offset], false, false, false, false]
	updated_attribute_names = map(initialize_attribute_names, attribute_update_logical) do x, y
		usable_covariates = covariates_0_check(ops)
		return y ? repeat([x], size(usable_covariates, 2) + 1) : [x]
	end
	attribute_names_symbol_vector = Vector(Symbol.(vcat(updated_attribute_names...)))
	line_attributes = hcat(sin_m_coeffs, cos_m_coeffs, Weighted_Average_Offset, b_coeffs, Amplitude, AmplitudeRatio, OffsetPlusMinusAmplitudeRatio, Acrophase)
	return SSE, DataFrame(line_attributes, attribute_names_symbol_vector, makeunique = true)
end
######################
# Alignment Function #
######################
function cosShift(estimate_list, ideal_list, additional_list, base="radians")
    if base == "hours"
        ideal_radian_list = mod.(ideal_list, 24) * (pi/12)
    elseif base == "radians"
        ideal_radian_list = mod.(ideal_list, 2*pi)
    else
        println("FLAG ERROR")
    end
    best_error = 2π
    shifted_estimate_list = deepcopy(estimate_list)
    shifted_additional_list = deepcopy(additional_list)
    for a in range(-pi, stop=pi, length=192)
        new_estimate_list = mod.(estimate_list .+ a, 2*pi)
        current_error = mean(1 .- cos.(new_estimate_list .- ideal_radian_list))
        if best_error > current_error
            best_error = deepcopy(current_error)
            shifted_estimate_list = deepcopy(new_estimate_list)
            shifted_additional_list = mod.(additional_list .+ a, 2*pi)
        end
    end
    for a in range(-pi, stop=pi, length=192)
        new_estimate_list = mod.(-1 .* (estimate_list .+ a), 2*pi)
        current_error = mean(1 .- cos.(new_estimate_list .- ideal_radian_list))
        if best_error > current_error
            best_error = deepcopy(current_error)
            shifted_estimate_list = deepcopy(new_estimate_list)
            shifted_additional_list = mod.(-1 .* (additional_list .+ a), 2*pi)
        end
    end
    return shifted_estimate_list, shifted_additional_list
end

#=
:align_disc => false,							# Is a discontinuous covariate used to align (true or false)
:align_disc_cov => 1,							# Which discontinuous covariate is used to choose samples to separately align (is an integer)
=#

# Making the align function more modular. First we want to take those steps that are repated often and make a generalizable function from them. Lets Begin...
# The first steps involve taking the fitoutput and deepcopying it
# The second step is finding the genes of interest gene symbols and figuring out if there are any replicates for that gene symbol
# If there are replicates of a gene symbol in the dataset then the average the significantly cycling replicates' acrophases should be aligned to the ideal acrophase

function AlignAcrophases(dataFile1::DataFrame, dataFile2::DataFrame, Fit_Output1::DataFrame, Fit_Output2::DataFrame, ops1::Dict{Symbol,Any}, ops2::Dict{Symbol,Any}, align_genes::Array{String,1}, align_acrophases::Array{Float64,1})
	
	goi_index = findXinY(align_genes, dataFile1[:, 1]) # find the indices of the gene symbols of interest
	goi_index2 = findXinY(align_genes, dataFile2[:, 1]) # find the indices of the gene symbols of interest
	each_gene_how_many_times = [length(findXinY([ii], dataFile1[:, 1])) for ii in align_genes] # find the number of replicates of each gene symbol of interest
	
	goi_dataframe = dataFile1[vcat(1:ops1[:o_fxr] - 1, goi_index), :] # put together the dataframe for the gene symbols of interest including the covariates rows
	goi_dataframe2 = dataFile2[vcat(1:ops2[:o_fxr] - 1, goi_index2), :] # put together the dataframe for the gene symbols of interest including the covariates rows
	
	all_phases = Fit_Output1.Phase
	Cosine_output_first = CosineFit(all_phases, goi_dataframe, ops1) # Get the cosine fit for the genes of interest
	
	p_statistic = Cosine_output_first.P_Statistic
	p_logical = p_statistic .< ops1[:align_p_cutoff] # Get a logical of cosine lines of best fit that are below p-cutoff
	
	all_acrophases = Cosine_output_first.Acrophase # Collect acrophases
	
	gene_group_upper_bounds = cumsum(each_gene_how_many_times) # Determine the upper index bounds for each gene's acrophase in the all_acrophases array
	gene_group_lower_bounds = vcat(1, gene_group_upper_bounds[1:end-1] .+ 1) # Determine the lower index bounds for each gene's acrophases in the all_acrophases array
	
	grouped_acrophases = [all_acrophases[gene_group_lower_bounds[ii]:gene_group_upper_bounds[ii]] for ii in 1:length(each_gene_how_many_times)] # Group all acrophases by gene
	grouped_p_logical = [p_logical[gene_group_lower_bounds[ii]:gene_group_upper_bounds[ii]] for ii in 1:length(each_gene_how_many_times)] # Group all p-cutoff Booleans by gene
	
	grouped_significant_acrophases = [grouped_acrophases[ii][grouped_p_logical[ii]] for ii in 1:length(each_gene_how_many_times)] # Select only significant acrophases from each gene acrophase group
	mean_grouped_significant_acrophases = Circular_Mean.(grouped_significant_acrophases) # Caclulate the average of the significant acrophases from each gene acrophase group
	usable_genes_logical = .!isnan.(mean_grouped_significant_acrophases) # Create logical of gene means that are non NaN
	
	grouped_p_statistic = [p_statistic[gene_group_lower_bounds[ii]:gene_group_upper_bounds[ii]] for ii in 1:length(each_gene_how_many_times)]
	grouped_significant_p_statistic = [grouped_p_statistic[ii][grouped_p_logical[ii]] for ii in 1:length(each_gene_how_many_times)]
	mean_grouped_significant_p_statistic = mean.(grouped_significant_p_statistic)[usable_genes_logical]
	
	usable_genes = align_genes[usable_genes_logical]
	if align_genes == human_homologue_gene_symbol
		usable_ideal_genes = mouse_gene_symbol[usable_genes_logical]
	else
		usable_ideal_genes = usable_genes
	end
	usable_ideal_gene_acrophases = align_acrophases[usable_genes_logical]
	usable_gene_acrophases = mean_grouped_significant_acrophases[usable_genes_logical]
	
	~, shifted_sample_phases = cosShift(usable_gene_acrophases, usable_ideal_gene_acrophases, Fit_Output2.Phase, ops2[:align_base]) # Use logical to align average significant acrophases to ideal acrophases
	
	shifted_cosine_output = CosineFit(shifted_sample_phases, dataFile2, ops2)
	Cosine_output = CosineFit(shifted_sample_phases, goi_dataframe2, ops2)
	
	p_statistic = Cosine_output.P_Statistic
	p_logical = p_statistic .< ops1[:align_p_cutoff] # Get a logical of cosine lines of best fit that are below p-cutoff
	
	r_squared = Cosine_output.R_Squared
	
	all_acrophases = Cosine_output.Acrophase # Collect all acrophases
	
	gene_group_upper_bounds = cumsum(each_gene_how_many_times) # Determine the upper index bounds for each gene's acrophase in the all_acrophases array
	gene_group_lower_bounds = vcat(1, gene_group_upper_bounds[1:end-1] .+ 1) # Determine the lower index bounds for each gene's acrophases in the all_acrophases array
	
	grouped_acrophases = [all_acrophases[gene_group_lower_bounds[ii]:gene_group_upper_bounds[ii]] for ii in 1:length(each_gene_how_many_times)] # Group all acrophases by gene
	grouped_p_logical = [p_logical[gene_group_lower_bounds[ii]:gene_group_upper_bounds[ii]] for ii in 1:length(each_gene_how_many_times)] # Group all p-cutoff Booleans by gene
	
	grouped_significant_acrophases = [grouped_acrophases[ii][grouped_p_logical[ii]] for ii in 1:length(each_gene_how_many_times)] # Select only significant acrophases from each gene acrophase group
	mean_grouped_significant_acrophases = Circular_Mean.(grouped_significant_acrophases) # Caclulate the average of the significant acrophases from each gene acrophase group
	usable_genes_logical = .!isnan.(mean_grouped_significant_acrophases) # Create logical of gene means that are non NaN
	
	grouped_p_statistic = [p_statistic[gene_group_lower_bounds[ii]:gene_group_upper_bounds[ii]] for ii in 1:length(each_gene_how_many_times)]
	grouped_significant_p_statistic = [grouped_p_statistic[ii][grouped_p_logical[ii]] for ii in 1:length(each_gene_how_many_times)]
	mean_grouped_significant_p_statistic = mean.(grouped_significant_p_statistic)[usable_genes_logical]
	
	grouped_r_squared = [r_squared[gene_group_lower_bounds[ii]:gene_group_upper_bounds[ii]] for ii in 1:length(each_gene_how_many_times)]
	grouped_significant_r_squared = [grouped_r_squared[ii][grouped_p_logical[ii]] for ii in 1:length(each_gene_how_many_times)]
	mean_grouped_significant_r_squared = mean.(grouped_significant_r_squared)[usable_genes_logical]
	
	usable_genes = align_genes[usable_genes_logical]
	if align_genes == human_homologue_gene_symbol
		usable_ideal_genes = mouse_gene_symbol[usable_genes_logical]
	else
		usable_ideal_genes = usable_genes
	end
	usable_ideal_gene_acrophases = align_acrophases[usable_genes_logical]
	usable_gene_acrophases = mean_grouped_significant_acrophases[usable_genes_logical]
	
	#Acrophase(significant_gene_symbols, significant_gene_symbols, significant_mouse_acrophase, significant_R_squared, significant_acrophases, check_these_p_values[p_below_005_logical], 0.05, space_factor = pi/15)
	
	Acrophase_Plot_Info_Array = [usable_genes, usable_ideal_genes, usable_ideal_gene_acrophases, mean_grouped_significant_r_squared, usable_gene_acrophases, mean_grouped_significant_p_statistic]
	
	return shifted_sample_phases, shifted_cosine_output, Acrophase_Plot_Info_Array
end

function AlignAcrophases(dataFile::DataFrame, Fit_Output::DataFrame, ops::Dict{Symbol,Any}, align_genes::Array{String,1}, align_acrophases::Array{Float64,1})
	
	goi_index = findXinY(align_genes, dataFile[:, 1]) # find the indices of the gene symbols of interest
	# each_gene_how_many_times = map(x -> length(findXinY([x], dataFile[:, 1])), ops[:align_genes]) # find the number of replicates of each gene symbol of interest
	each_gene_how_many_times = [length(findXinY([ii], dataFile[:, 1])) for ii in align_genes] # find the number of replicates of each gene symbol of interest
	
	goi_dataframe = dataFile[vcat(1:ops[:o_fxr] - 1, goi_index), :] # put together the dataframe for the gene symbols of interest including the covariates rows
	
	all_phases = Fit_Output.Phase
	Cosine_output_first = CosineFit(all_phases, goi_dataframe, ops) # Get the cosine fit for the genes of interest
	
	p_statistic = Cosine_output_first.P_Statistic
	p_logical = p_statistic .< ops[:align_p_cutoff] # Get a logical of cosine lines of best fit that are below p-cutoff
	
	all_acrophases = Cosine_output_first.Acrophase # Collect acrophases
	
	gene_group_upper_bounds = cumsum(each_gene_how_many_times) # Determine the upper index bounds for each gene's acrophase in the all_acrophases array
	gene_group_lower_bounds = vcat(1, gene_group_upper_bounds[1:end-1] .+ 1) # Determine the lower index bounds for each gene's acrophases in the all_acrophases array
	
	grouped_acrophases = [all_acrophases[gene_group_lower_bounds[ii]:gene_group_upper_bounds[ii]] for ii in 1:length(each_gene_how_many_times)] # Group all acrophases by gene
	grouped_p_logical = [p_logical[gene_group_lower_bounds[ii]:gene_group_upper_bounds[ii]] for ii in 1:length(each_gene_how_many_times)] # Group all p-cutoff Booleans by gene
	
	grouped_significant_acrophases = [grouped_acrophases[ii][grouped_p_logical[ii]] for ii in 1:length(each_gene_how_many_times)] # Select only significant acrophases from each gene acrophase group
	mean_grouped_significant_acrophases = Circular_Mean.(grouped_significant_acrophases) # Caclulate the average of the significant acrophases from each gene acrophase group
	usable_genes_logical = .!isnan.(mean_grouped_significant_acrophases) # Create logical of gene means that are non NaN

	grouped_p_statistic = [p_statistic[gene_group_lower_bounds[ii]:gene_group_upper_bounds[ii]] for ii in 1:length(each_gene_how_many_times)]
	grouped_significant_p_statistic = [grouped_p_statistic[ii][grouped_p_logical[ii]] for ii in 1:length(each_gene_how_many_times)]
	mean_grouped_significant_p_statistic = mean.(grouped_significant_p_statistic)[usable_genes_logical]

	usable_genes = align_genes[usable_genes_logical]
	if align_genes == human_homologue_gene_symbol
		usable_ideal_genes = mouse_gene_symbol[usable_genes_logical]
	else
		usable_ideal_genes = usable_genes
	end
	usable_ideal_gene_acrophases = align_acrophases[usable_genes_logical]
	usable_gene_acrophases = mean_grouped_significant_acrophases[usable_genes_logical]
	
	~, shifted_sample_phases = cosShift(usable_gene_acrophases, usable_ideal_gene_acrophases, Fit_Output.Phase, ops[:align_base]) # Use logical to align average significant acrophases to ideal acrophases

	shifted_cosine_output = CosineFit(shifted_sample_phases, dataFile, ops)
	Cosine_output = CosineFit(shifted_sample_phases, goi_dataframe, ops)

	p_statistic = Cosine_output.P_Statistic
	p_logical = p_statistic .< ops[:align_p_cutoff] # Get a logical of cosine lines of best fit that are below p-cutoff

	r_squared = Cosine_output.R_Squared
	
	all_acrophases = Cosine_output.Acrophase # Collect all acrophases
	
	gene_group_upper_bounds = cumsum(each_gene_how_many_times) # Determine the upper index bounds for each gene's acrophase in the all_acrophases array
	gene_group_lower_bounds = vcat(1, gene_group_upper_bounds[1:end-1] .+ 1) # Determine the lower index bounds for each gene's acrophases in the all_acrophases array
	
	grouped_acrophases = [all_acrophases[gene_group_lower_bounds[ii]:gene_group_upper_bounds[ii]] for ii in 1:length(each_gene_how_many_times)] # Group all acrophases by gene
	grouped_p_logical = [p_logical[gene_group_lower_bounds[ii]:gene_group_upper_bounds[ii]] for ii in 1:length(each_gene_how_many_times)] # Group all p-cutoff Booleans by gene
	
	grouped_significant_acrophases = [grouped_acrophases[ii][grouped_p_logical[ii]] for ii in 1:length(each_gene_how_many_times)] # Select only significant acrophases from each gene acrophase group
	mean_grouped_significant_acrophases = Circular_Mean.(grouped_significant_acrophases) # Caclulate the average of the significant acrophases from each gene acrophase group
	usable_genes_logical = .!isnan.(mean_grouped_significant_acrophases) # Create logical of gene means that are non NaN

	grouped_p_statistic = [p_statistic[gene_group_lower_bounds[ii]:gene_group_upper_bounds[ii]] for ii in 1:length(each_gene_how_many_times)]
	grouped_significant_p_statistic = [grouped_p_statistic[ii][grouped_p_logical[ii]] for ii in 1:length(each_gene_how_many_times)]
	mean_grouped_significant_p_statistic = mean.(grouped_significant_p_statistic)[usable_genes_logical]

	grouped_r_squared = [r_squared[gene_group_lower_bounds[ii]:gene_group_upper_bounds[ii]] for ii in 1:length(each_gene_how_many_times)]
	grouped_significant_r_squared = [grouped_r_squared[ii][grouped_p_logical[ii]] for ii in 1:length(each_gene_how_many_times)]
	mean_grouped_significant_r_squared = mean.(grouped_significant_r_squared)[usable_genes_logical]

	usable_genes = align_genes[usable_genes_logical]
	if align_genes == human_homologue_gene_symbol
		usable_ideal_genes = mouse_gene_symbol[usable_genes_logical]
	else
		usable_ideal_genes = usable_genes
	end
	usable_ideal_gene_acrophases = align_acrophases[usable_genes_logical]
	usable_gene_acrophases = mean_grouped_significant_acrophases[usable_genes_logical]

	#Acrophase(significant_gene_symbols, significant_gene_symbols, significant_mouse_acrophase, significant_R_squared, significant_acrophases, check_these_p_values[p_below_005_logical], 0.05, space_factor = pi/15)

	Acrophase_Plot_Info_Array = [usable_genes, usable_ideal_genes, usable_ideal_gene_acrophases, mean_grouped_significant_r_squared, usable_gene_acrophases, mean_grouped_significant_p_statistic]

	return shifted_sample_phases, shifted_cosine_output, Acrophase_Plot_Info_Array
end

function AlignSamples(dataFile2::DataFrame, Fit_Output1::DataFrame, Fit_Output2::DataFrame, ops1::Dict{Symbol,Any}, ops2::Dict{Symbol,Any}, align_samples::Array{String,1}, align_phases::Array{Float64,1})
	
	known_sample_indices = findXinY(align_samples, ops1[:o_column_ids])

	estimates_for_known_samples = Fit_Output1.Phase[known_sample_indices]

	~, shifted_sample_phases = cosShift(estimates_for_known_samples, align_phases, Fit_Output2.Phase, ops1[:align_base])
	
	shifted_Cosine_Sample_output = CosineFit(shifted_sample_phases, dataFile2, ops2)

	goi_index = findXinY(human_homologue_gene_symbol, dataFile2[ops2[:o_fxr]:end, 1]) # find the indices of the gene symbols of interest
	each_gene_how_many_times = [length(findXinY([ii], dataFile2[:, 1])) for ii in human_homologue_gene_symbol] # find the number of replicates of each gene symbol of interest

	Cosine_output = shifted_Cosine_Sample_output[goi_index, :]
	p_statistic = Cosine_output.P_Statistic
	p_logical = p_statistic .< ops2[:align_p_cutoff] # Get a logical of cosine lines of best fit that are below p-cutoff

	r_squared = Cosine_output.R_Squared
	
	all_acrophases = Cosine_output.Acrophase # Collect all acrophases
	
	gene_group_upper_bounds = cumsum(each_gene_how_many_times) # Determine the upper index bounds for each gene's acrophase in the all_acrophases array
	gene_group_lower_bounds = vcat(1, gene_group_upper_bounds[1:end-1] .+ 1) # Determine the lower index bounds for each gene's acrophases in the all_acrophases array
	
	grouped_acrophases = [all_acrophases[gene_group_lower_bounds[ii]:gene_group_upper_bounds[ii]] for ii in 1:length(each_gene_how_many_times)] # Group all acrophases by gene
	grouped_p_logical = [p_logical[gene_group_lower_bounds[ii]:gene_group_upper_bounds[ii]] for ii in 1:length(each_gene_how_many_times)] # Group all p-cutoff Booleans by gene
	
	grouped_significant_acrophases = [grouped_acrophases[ii][grouped_p_logical[ii]] for ii in 1:length(each_gene_how_many_times)] # Select only significant acrophases from each gene acrophase group
	mean_grouped_significant_acrophases = Circular_Mean.(grouped_significant_acrophases) # Caclulate the average of the significant acrophases from each gene acrophase group
	usable_genes_logical = .!isnan.(mean_grouped_significant_acrophases) # Create logical of gene means that are non NaN

	grouped_p_statistic = [p_statistic[gene_group_lower_bounds[ii]:gene_group_upper_bounds[ii]] for ii in 1:length(each_gene_how_many_times)]
	grouped_significant_p_statistic = [grouped_p_statistic[ii][grouped_p_logical[ii]] for ii in 1:length(each_gene_how_many_times)]
	mean_grouped_significant_p_statistic = mean.(grouped_significant_p_statistic)[usable_genes_logical]

	grouped_r_squared = [r_squared[gene_group_lower_bounds[ii]:gene_group_upper_bounds[ii]] for ii in 1:length(each_gene_how_many_times)]
	grouped_significant_r_squared = [grouped_r_squared[ii][grouped_p_logical[ii]] for ii in 1:length(each_gene_how_many_times)]
	mean_grouped_significant_r_squared = mean.(grouped_significant_r_squared)[usable_genes_logical]

	usable_genes = human_homologue_gene_symbol[usable_genes_logical]
	usable_ideal_genes = mouse_gene_symbol[usable_genes_logical]
	usable_ideal_gene_acrophases = mouse_acrophases[usable_genes_logical]
	usable_gene_acrophases = mean_grouped_significant_acrophases[usable_genes_logical]

	Acrophase_Plot_Info_Array = [usable_genes, usable_ideal_genes, usable_ideal_gene_acrophases, mean_grouped_significant_r_squared, usable_gene_acrophases, mean_grouped_significant_p_statistic]
	# println(typeof(shifted_sample_phases))

	return shifted_sample_phases, shifted_Cosine_Sample_output, Acrophase_Plot_Info_Array
end

function AlignSamples(dataFile::DataFrame, Fit_Output::DataFrame, ops::Dict{Symbol,Any}, align_samples::Array{String,1}, align_phases::Array{Float64,1})
	
	known_sample_indices = findXinY(align_samples, ops[:o_column_ids])
	# println(known_sample_indices); println(length(known_sample_indices))
	estimates_for_known_samples = Fit_Output.Phase[known_sample_indices]
	# println(estimates_for_known_samples); println(length(estimates_for_known_samples))
	# println(alignment_phases_for_samples); println(length(alignment_phases_for_samples))
	~, shifted_sample_phases = cosShift(estimates_for_known_samples, align_phases, Fit_Output.Phase, ops[:align_base])

	shifted_Cosine_Sample_output = CosineFit(shifted_sample_phases, dataFile, ops)

	goi_index = findXinY(human_homologue_gene_symbol, dataFile[ops[:o_fxr]:end, 1]) # find the indices of the gene symbols of interest
	each_gene_how_many_times = [length(findXinY([ii], dataFile[:, 1])) for ii in human_homologue_gene_symbol] # find the number of replicates of each gene symbol of interest

	Cosine_output = shifted_Cosine_Sample_output[goi_index, :]

	p_statistic = Cosine_output.P_Statistic
	p_logical = p_statistic .< ops[:align_p_cutoff] # Get a logical of cosine lines of best fit that are below p-cutoff

	r_squared = Cosine_output.R_Squared
	
	all_acrophases = Cosine_output.Acrophase # Collect all acrophases
	
	gene_group_upper_bounds = cumsum(each_gene_how_many_times) # Determine the upper index bounds for each gene's acrophase in the all_acrophases array
	gene_group_lower_bounds = vcat(1, gene_group_upper_bounds[1:end-1] .+ 1) # Determine the lower index bounds for each gene's acrophases in the all_acrophases array
	
	grouped_acrophases = [all_acrophases[gene_group_lower_bounds[ii]:gene_group_upper_bounds[ii]] for ii in 1:length(each_gene_how_many_times)] # Group all acrophases by gene
	grouped_p_logical = [p_logical[gene_group_lower_bounds[ii]:gene_group_upper_bounds[ii]] for ii in 1:length(each_gene_how_many_times)] # Group all p-cutoff Booleans by gene
	
	grouped_significant_acrophases = [grouped_acrophases[ii][grouped_p_logical[ii]] for ii in 1:length(each_gene_how_many_times)] # Select only significant acrophases from each gene acrophase group
	mean_grouped_significant_acrophases = Circular_Mean.(grouped_significant_acrophases) # Caclulate the average of the significant acrophases from each gene acrophase group
	usable_genes_logical = .!isnan.(mean_grouped_significant_acrophases) # Create logical of gene means that are non NaN

	grouped_p_statistic = [p_statistic[gene_group_lower_bounds[ii]:gene_group_upper_bounds[ii]] for ii in 1:length(each_gene_how_many_times)]
	grouped_significant_p_statistic = [grouped_p_statistic[ii][grouped_p_logical[ii]] for ii in 1:length(each_gene_how_many_times)]
	mean_grouped_significant_p_statistic = mean.(grouped_significant_p_statistic)[usable_genes_logical]

	grouped_r_squared = [r_squared[gene_group_lower_bounds[ii]:gene_group_upper_bounds[ii]] for ii in 1:length(each_gene_how_many_times)]
	grouped_significant_r_squared = [grouped_r_squared[ii][grouped_p_logical[ii]] for ii in 1:length(each_gene_how_many_times)]
	mean_grouped_significant_r_squared = mean.(grouped_significant_r_squared)[usable_genes_logical]

	usable_genes = human_homologue_gene_symbol[usable_genes_logical]
	usable_ideal_genes = mouse_gene_symbol[usable_genes_logical]
	usable_ideal_gene_acrophases = mouse_acrophases[usable_genes_logical]
	usable_gene_acrophases = mean_grouped_significant_acrophases[usable_genes_logical]

	Acrophase_Plot_Info_Array = [usable_genes, usable_ideal_genes, usable_ideal_gene_acrophases, mean_grouped_significant_r_squared, usable_gene_acrophases, mean_grouped_significant_p_statistic]
	# println(typeof(shifted_sample_phases))

	return shifted_sample_phases, shifted_Cosine_Sample_output, Acrophase_Plot_Info_Array
end

function Align(dataFile1::DataFrame, dataFile2::DataFrame, Fit_Output1::DataFrame, Fit_Output2::DataFrame, Eigengene_Correlation2::DataFrame, Model::Covariates, ops1::Dict{Symbol,Any}, ops2::Dict{Symbol,Any}, output_path::String)
	
	todays_date, all_subfolder_paths = OutputFolders(output_path, ops2)
	(plot_path_l, fit_path_l, model_path_l, parameter_path_l) = all_subfolder_paths
	
	if haskey(ops1, :align_genes) & haskey(ops1, :align_acrophases)
		my_info("ALIGNMENT ACROPHASES FOR GENES OTHER THAN MOUSE ATLAS GENES HAVE BEEN SPECIFIED.")
		align_genes, align_acrophases = ops2[:align_genes], ops2[:align_acrophases]
		Fit_Output2[!, :Phases_AG], Align_Genes_Cosine_Fit, Acrophase_Plot_Info_Array = AlignAcrophases(dataFile1, dataFile2, Fit_Output1, Fit_Output2, ops1, ops2, align_genes, align_acrophases)
		if sum(Acrophase_Plot_Info_Array[end] .< 0.05) > 0
			Acrophase(Acrophase_Plot_Info_Array..., 0.05, space_factor = pi/15)
			title("Acrophase Alignment to Genes of Interest According to Prior Dataset", pad = 32)
			my_info("SAVING FIGURE.")
			savefig(joinpath(plot_path_l, "Genes_of_Interest_Aligned_Acrophase_Plot_According_to_Prior_Dataset_$(todays_date).png"), bbox_inches = "tight", dpi = 300)
			my_info("FIGURE SAVED. CLOSING FIGURE.")
			close()
			my_info("FIGURE CLOSED. SAVING COSINE FIT.")
		end
		CSV.write(joinpath(fit_path_l, "Genes_of_Interest_Aligned_Cosine_Fit_According_to_Prior_Dataset_$(todays_date).csv"), Align_Genes_Cosine_Fit)
		my_info("COSINE FIT SAVED.")
	end

	if haskey(ops1, :align_samples) & haskey(ops1, :align_phases)
		my_info("ALIGNMENT PHASES FOR SAMPLES HAVE BEEN SPECIFIED.")
		align_samples, align_phases = ops1[:align_samples], ops1[:align_phases]
		Fit_Output2[!, :Phases_SA], Align_Samples_Cosine_Fit, Acrophase_Plot_Info_Array = AlignSamples(dataFile2, Fit_Output1, Fit_Output2, ops1, ops2, align_samples, align_phases)
		if sum(Acrophase_Plot_Info_Array[end] .< 0.05) > 0
			Acrophase(Acrophase_Plot_Info_Array..., 0.05, space_factor = pi/15)
			title("Acrophase Alignment to Sample Phases According to Prior Dataset", pad = 32)
			my_info("SAVING FIGURE.")
			savefig(joinpath(plot_path_l, "Sample_Phase_Aligned_Acrophase_Plot_According_to_Prior_Dataset_$(todays_date).png"), bbox_inches = "tight", dpi = 300)
			my_info("FIGURE SAVED. CLOSING FIGURE.")
			close()
			my_info("FIGURE CLOSED.")
		end
		
		if haskey(ops2, :align_samples) & haskey(ops2, :align_phases)
			fig = figure(figsize = (10, 10))
			title("CYCLOPS Predicted Sample Phases vs Ideal Times According to Prior Dataset", pad = 10)
			xlabel("Sample Phases ($(ops2[:align_base]))")
			if ops2[:align_base] == "radians"
				x_max = 2pi
				xticks([0, pi/2, pi, 3pi/2, 2pi], ["0", L"$\frac{π}{2}$", "π", L"$\frac{3π}{2}$", "2π"])
			elseif ops2[:align_base] == "hours"
				x_max = 24
				xticks([0, 6, 12, 18, 24], ["0", "6", "12", "18", "24"])
			end
			ylabel("Predicted Phases (radians)")
			yticks([0, pi/2, pi, 3pi/2, 2pi], ["0", L"$\frac{π}{2}$", "π", L"$\frac{3π}{2}$", "2π"])
			axis([0, x_max, 0, 2pi])
			grid(true)
			known_sample_indices = findXinY(ops2[:align_samples], Fit_Output2.ID)
			scatter(ops2[:align_phases], Fit_Output2.Phases_SA[known_sample_indices], s = 22)
			my_info("SAVING FIGURE")
			savefig(joinpath(plot_path_l, "Sample_Phases_Compared_To_Predicted_Phases_Plot_According_to_Prior_Dataset_$(todays_date).png"), bbox_inches = "tight", dpi = 300)
			my_info("FIGURE SAVED. CLOSING FIGURE.")
			close()
		end
		my_info("FIGURE CLOSED. SAVING COSINE FIT.")
		CSV.write(joinpath(fit_path_l, "Sample_Phase_Aligned_Cosine_Fit_According_to_Prior_Dataset_$(todays_date).csv"), Align_Samples_Cosine_Fit)
		my_info("COSINE FIT SAVED.")
	end

	my_info("ALIGNMENT TO MOUSE ATLAS ACROPHASES.")
	align_genes, align_acrophases = human_homologue_gene_symbol, mouse_acrophases
	Fit_Output2[!, :Phases_MA], Align_Genes_Cosine_Fit, Acrophase_Plot_Info_Array = AlignAcrophases(dataFile1, dataFile2, Fit_Output1, Fit_Output2, ops1, ops2, align_genes, align_acrophases)
	if sum(Acrophase_Plot_Info_Array[end] .< 0.05) > 0
		Acrophase(Acrophase_Plot_Info_Array..., 0.05, space_factor = pi/15)
		title("Acrophase Alignment to Mouse Atlas According to Prior Dataset", pad = 32)
		my_info("SAVING FIGURE.")
		savefig(joinpath(plot_path_l, "Mouse_Atlas_Aligned_Acrophase_Plot_According_to_Prior_Dataset_$(todays_date).png"), bbox_inches = "tight", dpi = 300)
		my_info("FIGURE SAVED. CLOSING FIGURE.")
		close()
		my_info("FIGURE CLOSED. SAVING COSINE FIT.")
	end
	CSV.write(joinpath(fit_path_l, "Mouse_Atlas_Aligned_Cosine_Fit_According_to_Prior_Dataset_$(todays_date).csv"), Align_Genes_Cosine_Fit)
	my_info("COSINE FIT SAVED. SAVING FIT OUTPUT.")
	CSV.write(joinpath(fit_path_l, "Fit_Output_$(todays_date).csv"), Fit_Output2)
	my_info("FIT OUTPUT SAVED. SAVING METRIC CORRELATIONS TO EIGENGENES.")
	CSV.write(joinpath(fit_path_l, "Metric_Correlation_to_Eigengenes_$(todays_date).csv"), Eigengene_Correlation2)
	my_info("METRIC CORRELATIONS TO EIGENGENES SAVED. SAVING TRAINING PARAMETERS.")
	CSV.write(joinpath(parameter_path_l, "Trained_Parameter_Dictionary_$(todays_date).csv"), ops2, bufsize=2^24) #, bufsize=2^24
	UniversalModelSaver(Model, dir=model_path_l, name="Trained_Model_$(todays_date)")

	return todays_date, all_subfolder_paths
end

function Align(dataFile::DataFrame, Fit_Output::DataFrame, Eigengene_Correlation::DataFrame, Model, ops::Dict, output_path::String)

	todays_date, all_subfolder_paths = OutputFolders(output_path, ops)
	(plot_path_l, fit_path_l, model_path_l, parameter_path_l) = all_subfolder_paths
	
	if haskey(ops, :align_genes) & haskey(ops, :align_acrophases)
		my_info("ALIGNMENT ACROPHASES FOR GENES OTHER THAN MOUSE ATLAS GENES HAVE BEEN SPECIFIED.")
		align_genes, align_acrophases = ops[:align_genes], ops[:align_acrophases]
		Fit_Output[!, :Phases_AG], Align_Genes_Cosine_Fit, Acrophase_Plot_Info_Array = AlignAcrophases(dataFile, Fit_Output, ops, align_genes, align_acrophases)
		if sum(Acrophase_Plot_Info_Array[end] .< 0.05) > 0
			Acrophase(Acrophase_Plot_Info_Array..., 0.05, space_factor = pi/15)
			title("Acrophase Alignment to Genes of Interest", pad = 32)
			my_info("SAVING FIGURE.")
			savefig(joinpath(plot_path_l, "Genes_of_Interest_Aligned_Acrophase_Plot_$(todays_date).png"), bbox_inches = "tight", dpi = 300)
			my_info("FIGURE SAVED. CLOSING FIGURE.")
			close()
			my_info("FIGURE CLOSED. SAVING COSINE FIT.")
		end
		CSV.write(joinpath(fit_path_l, "Genes_of_Interest_Aligned_Cosine_Fit_$(todays_date).csv"), Align_Genes_Cosine_Fit)
		my_info("COSINE FIT SAVED.")
	end
	
	if haskey(ops, :align_samples) & haskey(ops, :align_phases)
		my_info("ALIGNMENT PHASES FOR SAMPLES HAVE BEEN SPECIFIED.")
		align_samples, align_phases = ops[:align_samples], ops[:align_phases]
		Fit_Output[!, :Phases_SA], Align_Samples_Cosine_Fit, Acrophase_Plot_Info_Array = AlignSamples(dataFile, Fit_Output, ops, align_samples, align_phases)
		if sum(Acrophase_Plot_Info_Array[end] .< 0.05) > 0
			Acrophase(Acrophase_Plot_Info_Array..., 0.05, space_factor = pi/15)
			title("Acrophase Alignment to Sample Phases", pad = 32)
			my_info("SAVING FIGURE.")
			savefig(joinpath(plot_path_l, "Sample_Phase_Aligned_Acrophase_Plot_$(todays_date).png"), bbox_inches = "tight", dpi = 300)
			my_info("FIGURE SAVED. CLOSING FIGURE.")
			close()
			my_info("FIGURE CLOSED.")
		end
			
		fig = figure(figsize = (10, 10))
		title("CYCLOPS Predicted Sample Phases vs Ideal Times", pad = 10)
		xlabel("Sample Phases ($(ops[:align_base]))")
		if ops[:align_base] == "radians"
			x_max = 2pi
			xticks([0, pi/2, pi, 3pi/2, 2pi], ["0", L"$\frac{π}{2}$", "π", L"$\frac{3π}{2}$", "2π"])
		elseif ops[:align_base] == "hours"
			x_max = 24
			xticks([0, 6, 12, 18, 24], ["0", "6", "12", "18", "24"])
		end
		ylabel("Predicted Phases (radians)")
		yticks([0, pi/2, pi, 3pi/2, 2pi], ["0", L"$\frac{π}{2}$", "π", L"$\frac{3π}{2}$", "2π"])
		axis([0, x_max, 0, 2pi])
		grid(true)
		known_sample_indices = findXinY(ops[:align_samples], Fit_Output.ID)
		scatter(ops[:align_phases], Fit_Output.Phases_SA[known_sample_indices], s = 22)
		my_info("SAVING FIGURE")
		savefig(joinpath(plot_path_l, "Sample_Phases_Compared_To_Predicted_Phases_Plot_$(todays_date).png"), bbox_inches = "tight", dpi = 300)
		my_info("FIGURE SAVED. CLOSING FIGURE.")
		close()
		my_info("FIGURE CLOSED. SAVING COSINE FIT.")
		CSV.write(joinpath(fit_path_l, "Sample_Phase_Aligned_Cosine_Fit_$(todays_date).csv"), Align_Samples_Cosine_Fit)
		my_info("COSINE FIT SAVED.")
	end
	
	my_info("ALIGNMENT TO MOUSE ATLAS ACROPHASES.")
	align_genes, align_acrophases = human_homologue_gene_symbol, mouse_acrophases
	Fit_Output[!, :Phases_MA], Align_Genes_Cosine_Fit, Acrophase_Plot_Info_Array = AlignAcrophases(dataFile, Fit_Output, ops, align_genes, align_acrophases)
	if sum(Acrophase_Plot_Info_Array[end] .< 0.05) > 0
		Acrophase(Acrophase_Plot_Info_Array..., 0.05, space_factor = pi/15)
		title("Acrophase Alignment to Mouse Atlas", pad = 32)
		my_info("SAVING FIGURE.")
		savefig(joinpath(plot_path_l, "Mouse_Atlas_Aligned_Acrophase_Plot_$(todays_date).png"), bbox_inches = "tight", dpi = 300)
		my_info("FIGURE SAVED. CLOSING FIGURE.")
		close()
		my_info("FIGURE CLOSED. SAVING COSINE FIT.")
	end
	CSV.write(joinpath(fit_path_l, "Mouse_Atlas_Aligned_Cosine_Fit_$(todays_date).csv"), Align_Genes_Cosine_Fit)
	my_info("COSINE FIT SAVED. SAVING FIT OUTPUT.")
	CSV.write(joinpath(fit_path_l, "Fit_Output_$(todays_date).csv"), Fit_Output)
	my_info("FIT OUTPUT SAVED. SAVING METRIC CORRELATIONS TO EIGENGENES.")
	CSV.write(joinpath(fit_path_l, "Metric_Correlation_to_Eigengenes_$(todays_date).csv"), Eigengene_Correlation)
	my_info("METRIC CORRELATIONS TO EIGENGENES SAVED. SAVING TRAINING PARAMETERS.")
	CSV.write(joinpath(parameter_path_l, "Trained_Parameter_Dictionary_$(todays_date).csv"), ops, bufsize=2^25) #, bufsize=2^24
	UniversalModelSaver(Model, dir=model_path_l, name="Trained_Model_$(todays_date)")

	return todays_date, all_subfolder_paths
end

function XAlign(dataFile::DataFrame, Full_Fit_Output::DataFrame, Partial_Fit_Output::DataFrame, ops::Dict, output_path_info::Tuple, fold::Int)
	my_info("ALIGNING PARTIAL MODEL SAMPLE PHASE ESTIMATIONS TO FULL MODEL SAMPLE PHASE ESTIMATIONS.")
	todays_date, all_subfolder_paths = output_path_info
	(plot_path_l, fit_path_l, model_path_l, parameter_path_l) = all_subfolder_paths
	shifted_partial_model_phases, ~, ~ = AlignSamples(dataFile, Partial_Fit_Output, ops, Full_Fit_Output.ID, Full_Fit_Output.Phase)

	fig = figure(figsize = (10, 10))
	title("Full Model Estimated Phases vs Partial Model Estimated Phases", pad = 10)
	xlabel("Sample Phases ($(ops[:align_base]))")
	if ops[:align_base] == "radians"
		x_max = 2pi
		xticks([0, pi/2, pi, 3pi/2, 2pi], ["0", L"$\frac{π}{2}$", "π", L"$\frac{3π}{2}$", "2π"])
	elseif ops[:align_base] == "hours"
		x_max = 24
		xticks([0, 6, 12, 18, 24], ["0", "6", "12", "18", "24"])
	end
	ylabel("Predicted Phases (radians)")
	yticks([0, pi/2, pi, 3pi/2, 2pi], ["0", L"$\frac{π}{2}$", "π", L"$\frac{3π}{2}$", "2π"])
	axis([0, x_max, 0, 2pi])
	grid(true)
	known_sample_indices = findXinY(Full_Fit_Output.ID, Partial_Fit_Output.ID)
	scatter(Full_Fit_Output.Phase, shifted_partial_model_phases, s = 22)
	savefig(joinpath(plot_path_l, "Full_vs_Partial_Model_Estimated_Phases_Fold_$(fold)_$(todays_date).png"), bbox_inches = "tight", dpi = 300)
	close()

	return shifted_partial_model_phases
end

###################
# Metric Function #
###################
function Circular_Mean(phases::Array{T,1} where T <: Union{Float64, Float32})
  sinterm=sum(sin.(phases))
  costerm=sum(cos.(phases))
  return mod(atan(sinterm, costerm), 2pi)
end
#=
function Circular_Mean(phases::Array{Float32,1})
  sinterm=sum(sin.(phases))
  costerm=sum(cos.(phases))
  return mod(atan(sinterm, costerm), 2pi)
end
=#
function Fischer_Circular_Correlations(rphases::T,sphases::T) where T <: Array{T2, 1} where T2 <: Union{Float32, Float64}
	n1=length(rphases)
	n2=length(sphases)
	num=n1
	rphases=mod.(rphases .+ 2*pi, 2*pi)
	sphases=mod.(sphases .+ 2*pi, 2*pi)
	numtot=0.
	d1tot=0.
	d2tot=0.
	for i in 1:num
		for j in (i+1):num
			numeratorterm=sin(sphases[i] .- sphases[j]) .* sin(rphases[i] .- rphases[j])
			denomterm1=(sin(sphases[i] .- sphases[j])).^2
			denomterm2=(sin(rphases[i] .- rphases[j])).^2
			numtot=numtot .+ numeratorterm
			d1tot=d1tot .+ denomterm1
			d2tot=d2tot .+ denomterm2
		end
	end
	fischercor=numtot ./ (sqrt(d1tot) .* sqrt(d2tot))
	return fischercor
end

function Jammalamadka_Circular_Correlations(rphases::T,sphases::T) where T <: Array{T2, 1} where T2 <: Union{Float32, Float64}
	numtot=0.
	d1tot=0.
	d2tot=0.
	rbar=mod.(2*pi .+ Circular_Mean(rphases), 2*pi)
	sbar=mod.(2*pi .+ Circular_Mean(sphases), 2*pi)
	numtot=sum(sin.(rphases .- rbar) .* sin.(sphases .- sbar))
	d1tot=sqrt(sum( sin.(rphases .- rbar) .^ 2))
	d2tot=sqrt(sum( sin.(sphases .- sbar) .^ 2))
	Jammalamadka=numtot ./ (d1tot .* d2tot)
	return Jammalamadka
end

function Circular_Rank_Phases(rphases::T) where T <: Array{T2, 1} where T2 <: Union{Float32, Float64}
	number=length(rphases)
	rphases=mod.(rphases .+ 2*pi, 2*pi)
	rranks=tiedrank(rphases)
	rrankphases=rranks .* 2*pi ./ number
	return rrankphases
end

function Jammalamadka_Rank_Circular_Correlations(rphases::T,sphases::T) where T <: Array{T2, 1} where T2 <: Union{Float32, Float64}
	rphases=Circular_Rank_Phases(rphases)
	sphases=Circular_Rank_Phases(sphases)
	r_minus_s_bar=mod(atan(sum(sin.(rphases .- sphases)), sum(cos.(rphases .- sphases))), 2*pi)
	r_plus_s_bar=mod(atan(sum(sin.(rphases .+ sphases)), sum(cos.(rphases .+ sphases))), 2*pi)
	Ntot=length(rphases)
	term1=cos.(rphases .- sphases .- r_minus_s_bar)
	term2=cos.(rphases .+ sphases .- r_plus_s_bar)
	Jammalamadka=1 ./ Ntot .* (sum(term1)).- 1 ./ Ntot .* (sum(term2))
	return Jammalamadka
end

function FindComponentAngles(angle_sum::T,angle_diff::T) where T <: Union{Float32, Float64}
    rang=(angle_sum .+ angle_diff) ./ 2
    sang=(angle_sum .- angle_diff) ./ 2
	return [rang,sang]
end

function Jammalamadka_Uniform_Circular_Correlations(rphases::T,sphases::T) where T <: Array{T2, 1} where T2 <: Union{Float32, Float64}
	rphases=mod.(rphases, 2*pi)
	sphases=mod.(sphases, 2*pi)
	r_minus_s_bar=mod(atan(sum(sin.(rphases .- sphases)),sum(cos.(rphases .- sphases))), 2*pi)
	r_plus_s_bar=mod(atan(sum(sin.(rphases .+ sphases)),sum(cos.(rphases .+ sphases))), 2*pi)
	bars=FindComponentAngles(r_plus_s_bar,r_minus_s_bar)
	rbar=bars[1]
	sbar=bars[2]
	numtot=sum(sin.(rphases .- rbar) .* sin.(sphases .- sbar))
	d1tot=sqrt(sum( sin.(rphases .- rbar) .^ 2))
	d2tot=sqrt(sum( sin.(sphases .- sbar) .^ 2))
	Jammalamadka=numtot ./ (d1tot .* d2tot)
	return Jammalamadka
end

function Fischer_Circular_CorrelationMeasures(rphases::T,sphases::T) where T <: Array{T2, 1} where T2 <: Union{Float32, Float64}
	rrankphases=Circular_Rank_Phases(rphases)
	srankphases=Circular_Rank_Phases(sphases)
	F =Fischer_Circular_Correlations(rphases,sphases)
	FR =Fischer_Circular_Correlations(rrankphases,srankphases)
	return [F,FR]
end

function Jammalamadka_Circular_CorrelationMeasures(rphases::T,sphases::T) where T <: Array{T2, 1} where T2 <: Union{Float32, Float64}
	J =Jammalamadka_Circular_Correlations(rphases,sphases)
	JU =Jammalamadka_Uniform_Circular_Correlations(rphases,sphases)
	JR =Jammalamadka_Rank_Circular_Correlations(rphases,sphases)
	return [J, JU, JR]
end
############
# Plotting #
############
function GeneTracing(GOI::Array{String, 1}, dataFile::DataFrame, fitoutput::DataFrame, cosoutput::DataFrame; figure_height::Int=16, figure_width::Int=13, group_labels = true)
	close() # close open figures as suggested by PyPlot documentation
	fig = figure(figsize = (figure_height, figure_width)) # Create figure with figure_height by figure_width dimensions
	goi_index = findXinY(GOI, cosoutput[:, :Gene_Symbols]) # Find the GOI (genes of interest) indices in the gene symbol column of the cosinor regression output
	raw_core_clock_data = MakeFloat(dataFile[:o_fxr - 1 .+ goi_index, 2:end]) # using the indices
	all_indices_for_each_goi = map(x -> findXinY([x], cosoutput[:, :Gene_Symbols]), GOI)
	only_existing_goi_indices_logical = length.(all_indices_for_each_goi) .> 0
	existing_goi = GOI[only_existing_goi_indices_logical]
	all_indices_for_each_existing_goi = all_indices_fo_each_goi[only_existing_goi_indices_logical]
	all_indices_for_goi = vcat(all_indices_for_existing_goi...)
	#=
	if ops[:plot_only_best]
		best_indices_for_goi = Array{Any}([])
		for ii in all_indices_for_each_existing_goi
			keep_according_to_p_value = cosoutput[ii, :P_Statistic] .< ops[:plot_p_cutoff]
			amplitude_ratio_for_gene = cosoutput[ii, :Amplitude_Ratio]
			best_index = sum(keep_according_to_p_value) > 1 ? findmax(amplitude_ratio_for_gene[keep_according_to_p_value])[2][2] : (sum(keep_according_to_p_value) > 0 ? findall(keep_according_to_p_value) : findmax(amplitude_ratio_for_gene)[2][2])
			append!(best_indices_for_goi, best_index)
		end
		all_indices_for_goi = best_indices_for_goi
	end
	=#
	for ii in all_indices_for_goi
		xs = fitoutput.Phase
		xs_sorted, sort_index = x_sort(xs)
		Sin_Term = cosoutput[ii, :Sine_Coefficient]
		Cos_Term = cosoutput[ii, :Cosine_Coefficient]
		b_Term = cosoutput[ii, :Fit_Average]
		Additional_fit_averages = map(x -> match(r"Fit_Average_\d+", x), names(cosoutput)) .!= nothing
		Number_of_batches = sum(map(x -> match(r"Fit_Average_\d+", x), names(cosoutput)) .!= nothing) + 1
		if Number_of_batches > 1
			predicted = Sin_Term .* sin(xs) .+ Cos_Term .* cos.(xs) .+ b_Term .+ sum(ops[:o_covariates] * Matrix(cosine_output)[ii, Additional_fit_averages], dims = 2)
		end
	end
end

function GeneTracing(GOI::Array{String,1}, dataFiles::Array{DataFrame,1}, Phases::Array{Array{Float64,1},1}, CosineFits::Array{DataFrame,1}, first_rows::Array{Int64,1}, plot_batches::Array{Array{Int64,2},1}, plot_batch_offsets::Array{DataFrame,1}, plot_color_labels::Array{String,1}, plot_color_colors::Array{String,1}; space_factor = 0.15π)
	new_plot = missing
	for jj in 1:length(dataFiles)
		dataFile = dataFiles[jj]
		Phase = Phases[jj]
		CosineFit = CosineFits[jj]
		first_row = first_rows[jj]
		plot_batch = plot_batches[jj]
		plot_batch_offset = plot_batch_offsets[jj]
		plot_color_label = plot_color_labels[jj]
		plot_color_color = plot_color_colors[jj]
		GeneTracing(GOI, dataFile, Phase, CosineFit, first_row, plot_batch, plot_batch_offset, plot_color_label, plot_color_color, new_plot, space_factor = space_factor)
		new_plot = false
	end
end


function GeneTracing(GOI::Array{String,1}, dataFile::DataFrame, Phase::Array{Float64,1}, CosineFit::DataFrame, first_row::Int64, plot_batch::Array{Int64,2}, plot_batch_offset::DataFrame, plot_color_label::String, plot_color_color::String, new_plot::Missing; space_factor = 0.15π)
	base_offset = CosineFit[:, :Fit_Average]
	plot_batch_offset = MakeFloat(plot_batch_offset)
	# plot_color_offset = MakeFloat(plot_color_offset)

	fig = figure(figsize = (25,14))
	goi_index = findXinY(GOI, dataFile[first_row:end, 1])
	
	best_goi_index = Array{Int}([])
	for ii in 1:length(GOI)
		current_goi = GOI[ii]
		current_goi_is = findXinY([current_goi], CosineFit.Gene_Symbols)
		best_p_for_goi = findall(x -> x == minimum(CosineFit.P_Statistic[current_goi_is]), CosineFit.P_Statistic[current_goi_is])
		if length(best_p_for_goi) > 1
			best_Amp_for_goi = findall(x -> x == maximum(CosineFit.Amplitude_Ratio[current_goi_is[best_p_for_goi]]), CosineFit.Amplitude_Ratio[current_goi_is[best_p_for_goi]])
			if length(best_Amp_for_goi) > 1
				best_i = current_goi_is[best_p_for_goi[best_Amp_for_goi[1]]]
			else
				best_i = current_goi_is[best_p_for_goi[best_Amp_for_goi]]
			end
		else
			best_i = current_goi_is[best_p_for_goi]
		end
		append!(best_goi_index, best_i)
	end
	
	goi_index = best_goi_index
	goi_data = MakeFloat(Matrix(dataFile)[goi_index .+ first_row .- 1, 2:end])
	goi_info = CosineFit[goi_index, :]
	n_goi = length(goi_index)
	number_subplot_columns = Int(ceil(n_goi / 2))

	goi_base_offset = base_offset[goi_index]
	# goi_plot_color_offset = plot_color_offset[goi_index, :]
	goi_plot_batch_offset = plot_batch_offset[goi_index, :]
	
	goi_Sin_Terms = CosineFit[goi_index, :Sine_Coefficient]
	goi_Cos_Terms = CosineFit[goi_index, :Cosine_Coefficient]

	lin_xs = LinRange(0,2pi,100)
	lines_of_best_fit_no_offset = goi_Sin_Terms .* sin.(lin_xs') .+ goi_Cos_Terms .* cos.(lin_xs')
	lines_of_best_fit_base_offset = lines_of_best_fit_no_offset .+ goi_base_offset
	# lines_of_best_fit_maximum = lines_of_best_fit_no_offset .+ maximum(hcat(goi_base_offset, goi_base_offset .+ goi_plot_color_offset), dims = 2)
	# lines_of_best_fit_minimum = lines_of_best_fit_no_offset .+ minimum(hcat(goi_base_offset, goi_base_offset .+ goi_plot_color_offset), dims = 2)

	goi_correction_terms = -goi_plot_batch_offset * plot_batch
	goi_corrected_data = goi_data .+ goi_correction_terms
	goi_gene_means = mean(goi_data, dims = 2)
	goi_gene_stds = std(goi_data, dims = 2)

	for ii in 1:n_goi
		subplot(2, number_subplot_columns, ii)
		title(GOI[ii])

		gene_mean_this_gene = goi_gene_means[ii]
		gene_std_this_gene = goi_gene_stds[ii]
		axis([0, 2π, gene_mean_this_gene - 3*gene_std_this_gene, gene_mean_this_gene + 3*gene_std_this_gene])
		if (ii % number_subplot_columns) != 1			
			yticks([gene_mean_this_gene - 3*gene_std_this_gene, gene_mean_this_gene, gene_mean_this_gene + 3*gene_std_this_gene], ["", "", ""])
		else
			yticks([gene_mean_this_gene - 3*gene_std_this_gene, gene_mean_this_gene, gene_mean_this_gene + 3*gene_std_this_gene], ["ȳ - 3σ", "ȳ", "ȳ + 3σ"])
			ylabel("Gene Expression")
		end
		
		Acrophase_this_gene = goi_info[ii, :Acrophase]
		# Acrophase_string_this_gene = "$(trunc(Acrophase_this_gene/π, digits = 2))π"
		# xticks([0, π, 2π, Acrophase_this_gene], ["0", "π", "2π", (Acrophase_this_gene < space_factor) | ((Acrophase_this_gene > π-space_factor) & (Acrophase_this_gene < π+space_factor)) | (Acrophase_this_gene > 2π-space_factor) ? "" : Acrophase_string_this_gene])
		# if ii > number_subplot_columns
		# 	xlabel("Estimated Phase")
		# end
		
		Amplitude_this_gene = goi_info[ii, :Amplitude]
		Amplitude_ratio_this_gene = goi_info[ii, :Amplitude_Ratio]
		P_Statistic_this_gene = goi_info[ii, :Bonferroni_Statistic]
		base_this_gene = goi_base_offset[ii]
		# fill_between(lin_xs, lines_of_best_fit_maximum[ii, :], lines_of_best_fit_minimum[ii, :], alpha = 0.3, color = "orange")
		scatter(Phase, goi_corrected_data[ii, :], alpha = 0.6, edgecolors = "none", s = 14, c = plot_color_color, label = plot_color_label) # scatter(xs_sorted, corrected_gene_expression, alpha = 0.6, edgecolors = "none", s = 14, c = ops[:plot_color][1])
		plot(lin_xs, lines_of_best_fit_base_offset[ii, :], "$(plot_color_color)-", alpha = 0.8, label = "p$(L"$_{bon}$") = $(Float16(P_Statistic_this_gene))$(P_Statistic_this_gene < 0.05 ? (P_Statistic_this_gene < 0.01 ? "**" : "*") : "")")
		plot(repeat([mod(Acrophase_this_gene - π, 2π)], 2), [base_this_gene, base_this_gene - Amplitude_this_gene], "k-"#=, label = "$(L"$\frac{Amplitude}{Offset}$") = $(trunc(Amplitude_ratio_this_gene, digits = 2))"=#)
		plot(repeat([mod(Acrophase_this_gene - π, 2π)], 2), [base_this_gene, base_this_gene + Amplitude_this_gene], "k-", alpha = 0.3)
		plot(lin_xs, repeat([base_this_gene], 100), "k-")
		plot([0, Acrophase_this_gene + 2π], repeat([base_this_gene + Amplitude_this_gene], 2), "k-", alpha = 0.3)
		plot(repeat([Acrophase_this_gene], 2), [base_this_gene + Amplitude_this_gene, -1000], "b-", alpha = 0.3)
		# legend(loc = 4)
	end
	# subplots_adjust(left=.04,right=.995,bottom=.05,top=.945,wspace=.075,hspace=.1)

end

function GeneTracing(GOI::Array{String,1}, dataFile::DataFrame, Phase::Array{Float64,1}, CosineFit::DataFrame, first_row::Int64, plot_batch::Array{Int64,2}, plot_batch_offset::DataFrame, plot_color_label::String, plot_color_color::String, new_plot::Bool; space_factor = 0.15π)
	base_offset = CosineFit[:, :Fit_Average]
	plot_batch_offset = MakeFloat(plot_batch_offset)
	# plot_color_offset = MakeFloat(plot_color_offset)

	# fig = figure(figsize = (25,14))
	goi_index = findXinY(GOI, dataFile[first_row:end, 1])
	
	best_goi_index = Array{Int}([])
	for ii in 1:length(GOI)
		current_goi = GOI[ii]
		current_goi_is = findXinY([current_goi], CosineFit.Gene_Symbols)
		best_p_for_goi = findall(x -> x == minimum(CosineFit.P_Statistic[current_goi_is]), CosineFit.P_Statistic[current_goi_is])
		if length(best_p_for_goi) > 1
			best_Amp_for_goi = findall(x -> x == maximum(CosineFit.Amplitude_Ratio[current_goi_is[best_p_for_goi]]), CosineFit.Amplitude_Ratio[current_goi_is[best_p_for_goi]])
			if length(best_Amp_for_goi) > 1
				best_i = current_goi_is[best_p_for_goi[best_Amp_for_goi[1]]]
			else
				best_i = current_goi_is[best_p_for_goi[best_Amp_for_goi]]
			end
		else
			best_i = current_goi_is[best_p_for_goi]
		end
		append!(best_goi_index, best_i)
	end
	
	goi_index = best_goi_index
	goi_data = MakeFloat(Matrix(dataFile)[goi_index .+ first_row .- 1, 2:end])
	goi_info = CosineFit[goi_index, :]
	n_goi = length(goi_index)
	number_subplot_columns = Int(ceil(n_goi / 2))

	goi_base_offset = base_offset[goi_index]
	# goi_plot_color_offset = plot_color_offset[goi_index, :]
	goi_plot_batch_offset = plot_batch_offset[goi_index, :]
	
	goi_Sin_Terms = CosineFit[goi_index, :Sine_Coefficient]
	goi_Cos_Terms = CosineFit[goi_index, :Cosine_Coefficient]

	lin_xs = LinRange(0,2pi,100)
	lines_of_best_fit_no_offset = goi_Sin_Terms .* sin.(lin_xs') .+ goi_Cos_Terms .* cos.(lin_xs')
	lines_of_best_fit_base_offset = lines_of_best_fit_no_offset .+ goi_base_offset
	# lines_of_best_fit_maximum = lines_of_best_fit_no_offset .+ maximum(hcat(goi_base_offset, goi_base_offset .+ goi_plot_color_offset), dims = 2)
	# lines_of_best_fit_minimum = lines_of_best_fit_no_offset .+ minimum(hcat(goi_base_offset, goi_base_offset .+ goi_plot_color_offset), dims = 2)

	goi_correction_terms = -goi_plot_batch_offset * plot_batch
	goi_corrected_data = goi_data .+ goi_correction_terms
	goi_gene_means = mean(goi_data, dims = 2)
	goi_gene_stds = std(goi_data, dims = 2)

	for ii in 1:n_goi
		subplot(2, number_subplot_columns, ii)
		title(GOI[ii])

		gene_mean_this_gene = goi_gene_means[ii]
		gene_std_this_gene = goi_gene_stds[ii]
		current_y_min, current_y_max = gca().axes.get_ylim()
		best_y_min, best_y_max = (minimum([current_y_min, gene_mean_this_gene - 3*gene_std_this_gene]), maximum([current_y_max, gene_mean_this_gene + 3*gene_std_this_gene]))
		new_axis_range = best_y_max - best_y_min
		my_axis_buffer = new_axis_range/10
		final_y_min, final_y_max = ((best_y_min - my_axis_buffer), (best_y_max + my_axis_buffer))

		axis([0, 2π, final_y_min, final_y_max])
		# axis([0, 2π, gene_mean_this_gene - 3*gene_std_this_gene, gene_mean_this_gene + 3*gene_std_this_gene])
		# if (ii % number_subplot_columns) != 1			
		# 	yticks([gene_mean_this_gene - 3*gene_std_this_gene, gene_mean_this_gene, gene_mean_this_gene + 3*gene_std_this_gene], ["", "", ""])
		# else
		# 	yticks([gene_mean_this_gene - 3*gene_std_this_gene, gene_mean_this_gene, gene_mean_this_gene + 3*gene_std_this_gene], ["ȳ - 3σ", "ȳ", "ȳ + 3σ"])
		# 	ylabel("Gene Expression")
		# end
		
		Acrophase_this_gene = goi_info[ii, :Acrophase]
		Acrophase_string_this_gene = "$(trunc(Acrophase_this_gene/π, digits = 2))π"
		xticks([0, π, 2π, Acrophase_this_gene], ["0", "π", "2π", (Acrophase_this_gene < space_factor) | ((Acrophase_this_gene > π-space_factor) & (Acrophase_this_gene < π+space_factor)) | (Acrophase_this_gene > 2π-space_factor) ? "" : Acrophase_string_this_gene])
		if ii > number_subplot_columns
			xlabel("Estimated Phase")
		end
		
		Amplitude_this_gene = goi_info[ii, :Amplitude]
		Amplitude_ratio_this_gene = goi_info[ii, :Amplitude_Ratio]
		P_Statistic_this_gene = goi_info[ii, :Bonferroni_Statistic]
		base_this_gene = goi_base_offset[ii]
		# fill_between(lin_xs, lines_of_best_fit_maximum[ii, :], lines_of_best_fit_minimum[ii, :], alpha = 0.3, color = "orange")
		scatter(Phase, goi_corrected_data[ii,:], alpha = 0.6, edgecolors = "none", s = 14, c = plot_color_color, label = plot_color_label) # scatter(xs_sorted, corrected_gene_expression, alpha = 0.6, edgecolors = "none", s = 14, c = ops[:plot_color][1])
		plot(lin_xs, lines_of_best_fit_base_offset[ii, :], "$(plot_color_color)-", alpha = 0.8, label = "p$(L"$_{bon}$") = $(Float16(P_Statistic_this_gene))$(P_Statistic_this_gene < 0.05 ? (P_Statistic_this_gene < 0.01 ? "**" : "*") : "")")
		plot(repeat([mod(Acrophase_this_gene - π, 2π)], 2), [base_this_gene, base_this_gene - Amplitude_this_gene], "k-"#=, label = "$(L"$\frac{Amplitude}{Offset}$") = $(trunc(Amplitude_ratio_this_gene, digits = 2))"=#)
		plot(repeat([mod(Acrophase_this_gene - π, 2π)], 2), [base_this_gene, base_this_gene + Amplitude_this_gene], "k-", alpha = 0.3)
		plot(lin_xs, repeat([base_this_gene], 100), "k-")
		plot([0, Acrophase_this_gene + 2π], repeat([base_this_gene + Amplitude_this_gene], 2), "k-", alpha = 0.3)
		plot(repeat([Acrophase_this_gene], 2), [base_this_gene + Amplitude_this_gene, final_y_min], "b-", alpha = 0.3)
		legend(loc = 4)
	end
	subplots_adjust(left=.04,right=.995,bottom=.05,top=.945,wspace=.075,hspace=.1)

end

function GeneTracing(GOI::Array{String,1}, dataFile::DataFrame, Phases::Array{Float64,1}, CosineFit::DataFrame, first_row::Int64, plot_batch::Array{Int64,2}, plot_batch_offset::DataFrame, plot_color::Array{Int64,2}, plot_color_offset::DataFrame, plot_color_colors::Array{String,1}, plot_color_labels::Array{String,1}; space_factor = 0.15π)
	
	base_offset = CosineFit[:, :Fit_Average]
	plot_batch_offset = MakeFloat(plot_batch_offset)
	plot_color_offset = MakeFloat(plot_color_offset)

	fig = figure(figsize = (25,14))
	goi_index = findXinY(GOI, dataFile[first_row:end, 1])
	
	best_goi_index = Array{Int}([])
	for ii in 1:length(GOI)
		current_goi = GOI[ii]
		current_goi_is = findXinY([current_goi], CosineFit.Gene_Symbols)
		best_p_for_goi = findall(x -> x == minimum(CosineFit.P_Statistic[current_goi_is]), CosineFit.P_Statistic[current_goi_is])
		if length(best_p_for_goi) > 1
			best_Amp_for_goi = findall(x -> x == maximum(CosineFit.Amplitude_Ratio[current_goi_is[best_p_for_goi]]), CosineFit.Amplitude_Ratio[current_goi_is[best_p_for_goi]])
			if length(best_Amp_for_goi) > 1
				best_i = current_goi_is[best_p_for_goi[best_Amp_for_goi[1]]]
			else
				best_i = current_goi_is[best_p_for_goi[best_Amp_for_goi]]
			end
		else
			best_i = current_goi_is[best_p_for_goi]
		end
		append!(best_goi_index, best_i)
	end
	
	goi_index = best_goi_index
	goi_data = MakeFloat(Matrix(dataFile)[goi_index .+ first_row .- 1, 2:end])
	goi_info = CosineFit[goi_index, :]
	n_goi = length(goi_index)
	number_subplot_columns = Int(ceil(n_goi / 2))

	goi_base_offset = base_offset[goi_index]
	goi_plot_color_offset = plot_color_offset[goi_index, :]
	goi_plot_batch_offset = plot_batch_offset[goi_index, :]
	
	goi_Sin_Terms = CosineFit[goi_index, :Sine_Coefficient]
	goi_Cos_Terms = CosineFit[goi_index, :Cosine_Coefficient]

	lin_xs = LinRange(0,2pi,100)
	lines_of_best_fit_no_offset = goi_Sin_Terms .* sin.(lin_xs') .+ goi_Cos_Terms .* cos.(lin_xs')
	lines_of_best_fit_base_offset = lines_of_best_fit_no_offset .+ goi_base_offset
	lines_of_best_fit_maximum = lines_of_best_fit_no_offset .+ maximum(hcat(goi_base_offset, goi_base_offset .+ goi_plot_color_offset), dims = 2)
	lines_of_best_fit_minimum = lines_of_best_fit_no_offset .+ minimum(hcat(goi_base_offset, goi_base_offset .+ goi_plot_color_offset), dims = 2)

	goi_correction_terms = -goi_plot_batch_offset * plot_batch
	goi_corrected_data = goi_data .+ goi_correction_terms
	goi_gene_means = mean(goi_data, dims = 2)
	goi_gene_stds = std(goi_data, dims = 2)

	for ii in 1:n_goi
		subplot(2, number_subplot_columns, ii)
		title(GOI[ii])

		gene_mean_this_gene = goi_gene_means[ii]
		gene_std_this_gene = goi_gene_stds[ii]
		axis([0, 2π, gene_mean_this_gene - 3*gene_std_this_gene, gene_mean_this_gene + 3*gene_std_this_gene])
		if (ii % number_subplot_columns) != 1			
			yticks([gene_mean_this_gene - 3*gene_std_this_gene, gene_mean_this_gene, gene_mean_this_gene + 3*gene_std_this_gene], ["", "", ""])
		else
			yticks([gene_mean_this_gene - 3*gene_std_this_gene, gene_mean_this_gene, gene_mean_this_gene + 3*gene_std_this_gene], ["ȳ - 3σ", "ȳ", "ȳ + 3σ"])
			ylabel("Gene Expression")
		end
		
		Acrophase_this_gene = goi_info[ii, :Acrophase]
		Acrophase_string_this_gene = "$(trunc(Acrophase_this_gene/π, digits = 2))π"
		xticks([0, π, 2π, Acrophase_this_gene], ["0", "π", "2π", (Acrophase_this_gene < space_factor) | ((Acrophase_this_gene > π-space_factor) & (Acrophase_this_gene < π+space_factor)) | (Acrophase_this_gene > 2π-space_factor) ? "" : Acrophase_string_this_gene])
		if ii > number_subplot_columns
			xlabel("Estimated Phase")
		end
		
		Amplitude_this_gene = goi_info[ii, :Amplitude]
		Amplitude_ratio_this_gene = goi_info[ii, :Amplitude_Ratio]
		P_Statistic_this_gene = goi_info[ii, :Bonferroni_Statistic]
		base_this_gene = goi_base_offset[ii]
		fill_between(lin_xs, lines_of_best_fit_maximum[ii, :], lines_of_best_fit_minimum[ii, :], alpha = 0.3, color = "orange")
		for jj in 1:length(plot_color_labels)
			if jj != 1
				this_group_logical = Bool.(plot_color[jj-1, :])
				phases_this_group = Phases[this_group_logical]
				goi_corrected_data_this_group = goi_corrected_data[ii, this_group_logical]
			else
				this_group_logical = .!Bool.(plot_color[jj, :])
				phases_this_group = Phases[this_group_logical]
				goi_corrected_data_this_group = goi_corrected_data[ii, this_group_logical]
			end
			scatter(phases_this_group, goi_corrected_data_this_group, alpha = 0.6, edgecolors = "none", s = 14, c = plot_color_colors[jj], label = plot_color_labels[jj]) # scatter(xs_sorted, corrected_gene_expression, alpha = 0.6, edgecolors = "none", s = 14, c = ops[:plot_color][1])
		end
		plot(lin_xs, lines_of_best_fit_base_offset[ii, :], "r-", alpha = 0.8, label = "p$(L"$_{bon}$") = $(Float16(P_Statistic_this_gene))$(P_Statistic_this_gene < 0.05 ? (P_Statistic_this_gene < 0.01 ? "**" : "*") : "")")
		plot(repeat([mod(Acrophase_this_gene - π, 2π)], 2), [base_this_gene, base_this_gene - Amplitude_this_gene], "k-", label = "$(L"$\frac{Amplitude}{Offset}$") = $(trunc(Amplitude_ratio_this_gene, digits = 2))")
		plot(repeat([mod(Acrophase_this_gene - π, 2π)], 2), [base_this_gene, base_this_gene + Amplitude_this_gene], "k-", alpha = 0.3)
		plot(lin_xs, repeat([base_this_gene], 100), "k-")
		plot([0, Acrophase_this_gene + 2π], repeat([base_this_gene + Amplitude_this_gene], 2), "k-", alpha = 0.3)
		plot(repeat([Acrophase_this_gene], 2), [base_this_gene + Amplitude_this_gene, gene_mean_this_gene - 3*gene_std_this_gene], "b-", alpha = 0.3)
		legend(loc = 4)
	end
	subplots_adjust(left=.04,right=.995,bottom=.05,top=.945,wspace=.075,hspace=.1)
end

function GeneTracing(GOI::Array{String,1}, dataFile::DataFrame, FitOutput::DataFrame, plot_factor::Array{Int64,2}; space_factor = 0.15π)

end

function GeneTracing(GOI::Array{String,1}, dataFile::T, fitoutput::T, ops::Dict{Symbol,Any}; group_labels = true, space_factor = 0.15π) where T <: DataFrame
    # close()
    fig = figure(figsize = (25, 14))
	goi_index = findXinY(GOI, dataFile[ops[:o_fxr]:end, 1])
	core_clock_data = MakeFloat(Matrix(dataFile)[goi_index .+ ops[:o_fxr] .- 1, 2:end])
	had = false
	plot_line_alpha = 1.

	if !ops[:plot_use_o_cov] | ((ops[:plot_correct_batches] | ops[:plot_disc]) & !ismissing(ops[:o_dco]))
		if haskey(ops, :o_covariates)
			had = true
			store_covariates = deepcopy(ops[:o_covariates])
			delete!(ops, :o_covariates)
		end
		if ops[:plot_disc] | ops[:plot_correct_batches]
			ops[:o_covariates] = permutedims(ops[:o_dcorr][ops[:plot_disc_cov]])
			normCovOnehot = ops[:o_dco][ops[:plot_disc_cov]]
		end
	end
	cosine_output = CosineFit(fitoutput.Phase, dataFile[vcat(1:ops[:o_fxr] - 1, goi_index .+ ops[:o_fxr] .- 1), :], ops)
	best_goi_index = Array{Int}([])
	for ii in 1:length(GOI)
		current_goi = GOI[ii]
		current_goi_is = findXinY([current_goi], cosine_output.Gene_Symbols)
		best_p_for_goi = findall(x -> x == minimum(cosine_output.P_Statistic[current_goi_is]), cosine_output.P_Statistic[current_goi_is])
		if length(best_p_for_goi) > 1
			best_Amp_for_goi = findall(x -> x == maximum(cosine_output.Amplitude[current_goi_is[best_p_for_goi]]), cosine_output.Amplitude[current_goi_is[best_p_for_goi]])
			if length(best_Amp_for_goi) > 1
				best_i = current_goi_is[best_p_for_goi[best_Amp_for_goi[1]]]
			else
				best_i = current_goi_is[best_p_for_goi[best_Amp_for_goi]]
			end
		else
			best_i = current_goi_is[best_p_for_goi]
		end
		append!(best_goi_index, best_i)
	end
	goi_index = best_goi_index

	number_subplot_columns = Int(ceil(length(goi_index) / 2))

    for II in 1:length(goi_index)
		ii = goi_index[II]
        xs = fitoutput.Phase
        xs_sorted, sort_index = x_sort(fitoutput.Phase)
        Sin_Term = cosine_output[ii, :Sine_Coefficient]
        Cos_Term = cosine_output[ii, :Cosine_Coefficient]
        b_Term = cosine_output[ii, :Fit_Average]
		Additional_fit_averages = map(x -> match(r"Fit_Average_\d+", x), names(cosine_output)) .!= nothing
		Number_of_batches = sum(map(x -> match(r"Fit_Average_\d+", x), names(cosine_output)) .!= nothing) + 1
		# println("Number of batches greater than 1 $(Number_of_batches > 1)")
		if Number_of_batches > 1
		# if ops[:plot_use_o_cov] & haskey(ops, :o_covariates)
		# if size(cosine_output, 2) >= 13
			usable_covariates = covariates_0_check(ops)
			# println("Correct batches? $(ops[:plot_correct_batches])")
			if ops[:plot_correct_batches] & !ops[:plot_disc]
				batch_corrections = sum(usable_covariates * Matrix(cosine_output)[ii, Additional_fit_averages], dims = 2)[sort_index]
				prediction_batch_additions = 0
			else
				batch_corrections = 0
				prediction_batch_additions = sum(usable_covariates * Matrix(cosine_output)[ii, Additional_fit_averages], dims = 2)
				plot_line_alpha = 0.5
			end
			predicted = (Sin_Term .* sin.(xs) .+ Cos_Term .* cos.(xs) .+ b_Term .+ prediction_batch_additions)[sort_index]
		else
			batch_corrections = 0
			predicted = Sin_Term .* sin.(xs_sorted) .+ Cos_Term .* cos.(xs_sorted) .+ b_Term
		end
		# println("Batch correction $batch_corrections")
        Amplitude = cosine_output[ii, :Amplitude]
        Amplitude_Ratio = trunc.(cosine_output[ii, :Amplitude_Ratio], digits = 2)
        Acrophase = mod.(cosine_output[ii, :Acrophase], 2π)
        Acrophase_string = "$(trunc(Acrophase/π, digits = 2))π"
        Amplitude_rel_mean = trunc.(Amplitude / b_Term, digits = 1)
        gene_expression = core_clock_data[ii, sort_index]
		corrected_gene_expression = gene_expression .- batch_corrections
		# println("Is corrected gene expression same as uncorrected $(gene_expression == corrected_gene_expression)")
        gene_mean, gene_std = mean(corrected_gene_expression), std(corrected_gene_expression)
		# if ops[:plot_disc] & (length(ops[:o_dco]) > 0)
		if ops[:plot_disc] & !(ismissing(ops[:o_dco]))
			for jj in 1:size(normCovOnehot, 1)
				kk = Bool.(normCovOnehot[jj, :])[sort_index]
				if ops[:plot_separate]
					subplot(2, number_subplot_columns * Number_of_batches, Number_of_batches * (II - 1) + 1 + (jj - 1))
					if jj == 1
						plot(xs_sorted[kk], predicted[kk], "r-", alpha = plot_line_alpha, label = "p$(L"$_{bon}$") = $(Float16(cosine_output[ii, :Bonferroni_Statistic]))$(cosine_output[ii, :Bonferroni_Statistic] < 0.05 ? (cosine_output[ii, :Bonferroni_Statistic] < 0.01 ? "**" : "*") : "")")
						plot(repeat([mod(Acrophase - π, 2π)], 2), [b_Term, b_Term - Amplitude], "k-", label = "$(L"$\frac{b + Amplitude}{b - Amplitude}$") = $Amplitude_Ratio")
						legend(loc = 4)
						plot(repeat([mod(Acrophase - π, 2π)], 2), [b_Term, b_Term + Amplitude], "k-", alpha = 0.3)
						plot(LinRange(0, 2π, size(gene_expression, 1)), repeat([b_Term], size(gene_expression, 1)), "k-")
						plot([0, Acrophase + 2π], repeat([b_Term + Amplitude], 2), "k-", alpha = 0.3)
						plot(repeat([Acrophase], 2), [b_Term + Amplitude, gene_mean - 3*gene_std], "b-", alpha = 0.3)
					else
						b_Term_temp = cosine_output[ii, 3 + jj] .+ b_Term
						Amplitude_Ratio_temp = trunc.(cosine_output[ii, 3 + Number_of_batches + jj], digits = 2)
						plot(xs_sorted[kk], predicted[kk], "r-", alpha = plot_line_alpha, label = "p$(L"$_{bon}$") = $(Float16(cosine_output[ii, :Bonferroni_Statistic]))$(cosine_output[ii, :Bonferroni_Statistic] < 0.05 ? (cosine_output[ii, :Bonferroni_Statistic] < 0.01 ? "**" : "*") : "")")
						plot(repeat([mod(Acrophase - π, 2π)], 2), [b_Term_temp, b_Term_temp - Amplitude], "k-", label = "$(L"$\frac{b + Amplitude}{b - Amplitude}$") = $Amplitude_Ratio_temp")
						legend(loc = 4)
						plot(repeat([mod(Acrophase - π, 2π)], 2), [b_Term_temp, b_Term_temp + Amplitude], "k-", alpha = 0.3)
						plot([0, Acrophase + 2π], repeat([b_Term_temp + Amplitude], 2), "k-", alpha = 0.3)
						plot(LinRange(0, 2π, size(gene_expression, 1)), repeat([b_Term_temp], size(gene_expression, 1)), "k-")
						plot(repeat([Acrophase], 2), [b_Term_temp + Amplitude, gene_mean - 3*gene_std], "b-", alpha = 0.3)
					end
				else
					subplot(2, number_subplot_columns, II)
					if jj == 1
						plot(xs_sorted[kk], predicted[kk], "r-", alpha = plot_line_alpha, label = "p$(L"$_{bon}$") = $(Float16(cosine_output[ii, :Bonferroni_Statistic]))$(cosine_output[ii, :Bonferroni_Statistic] < 0.05 ? (cosine_output[ii, :Bonferroni_Statistic] < 0.01 ? "**" : "*") : "")")
						plot(repeat([mod(Acrophase - π, 2π)], 2), [b_Term, b_Term - Amplitude], "k-", label = "$(L"$\frac{b + Amplitude}{b - Amplitude}$") = $Amplitude_Ratio")
						legend(loc = 4)
						plot(repeat([mod(Acrophase - π, 2π)], 2), [b_Term, b_Term + Amplitude], "k-", alpha = 0.3)
						plot(LinRange(0, 2π, size(gene_expression, 1)), repeat([b_Term], size(gene_expression, 1)), "k-")
						plot([0, Acrophase + 2π], repeat([b_Term + Amplitude], 2), "k-", alpha = 0.3)
						plot(repeat([Acrophase], 2), [b_Term + Amplitude, gene_mean - 3*gene_std], "b-", alpha = 0.3)
					else
						if !ops[:plot_only_color]
							b_Term_temp = cosine_output[ii, 3 + jj] .+ b_Term
							plot(xs_sorted[kk], predicted[kk], "r-", alpha = plot_line_alpha)
							plot(repeat([mod(Acrophase - π, 2π)], 2), [b_Term_temp, b_Term_temp - Amplitude], "k-")
							plot(repeat([mod(Acrophase - π, 2π)], 2), [b_Term_temp, b_Term_temp + Amplitude], "k-", alpha = 0.3)
							plot([0, Acrophase + 2π], repeat([b_Term_temp + Amplitude], 2), "k-", alpha = 0.3)
							plot(LinRange(0, 2π, size(gene_expression, 1)), repeat([b_Term_temp], size(gene_expression, 1)), "k-")
						end
					end
				end
				scatter(xs_sorted[kk], corrected_gene_expression[kk], alpha = 0.6, edgecolors = "none", s = 14, label = ops[:o_dcl][ops[:plot_disc_cov]][jj], c = ops[:plot_color][jj])
				axis([0, 2π, gene_mean - 3*gene_std, gene_mean + 3*gene_std])
				xticks([0, π, 2π, Acrophase], ["0", "π", "2π", (Acrophase < space_factor) | ((Acrophase > π-space_factor) & (Acrophase < π+space_factor)) | (Acrophase > 2π-space_factor) ? "" : Acrophase_string])
				if group_labels
					legend(loc = 4)
				end
				title(cosine_output.Gene_Symbols[ii])
				if mod(Number_of_batches * (II - 1) + jj, number_subplot_columns * Number_of_batches) == 1
					yticks([gene_mean - 3*gene_std, gene_mean, gene_mean + 3*gene_std], ["ȳ - 3σ", "ȳ", "ȳ + 3σ"])
				else
					yticks([gene_mean - 3*gene_std, gene_mean, gene_mean + 3*gene_std], ["", "", ""])
				end
				if (Number_of_batches * (II - 1) + jj) > (number_subplot_columns * Number_of_batches)
						xlabel("Estimated Phases", labelpad = 18)
				end
				if mod(Number_of_batches * (II - 1) + jj, number_subplot_columns * Number_of_batches) == 1
						ylabel("Gene Expression", labelpad = 18)
				end
			end
		else
			subplot(2, number_subplot_columns, II)
			scatter(xs_sorted, corrected_gene_expression, alpha = 0.6, edgecolors = "none", s = 14, c = ops[:plot_color][1])
			axis([0, 2π, gene_mean - 3*gene_std, gene_mean + 3*gene_std])
			xticks([0, π, 2π, Acrophase], ["0", "π", "2π", (Acrophase < space_factor) | ((Acrophase > π-space_factor) & (Acrophase < π+space_factor)) | (Acrophase > 2π-space_factor) ? "" : Acrophase_string])
			plot(xs_sorted, predicted, "r-", alpha = plot_line_alpha, label = "p$(L"$_{bon}$") = $(Float16(cosine_output[ii, :Bonferroni_Statistic]))$(cosine_output[ii, :Bonferroni_Statistic] < 0.05 ? (cosine_output[ii, :Bonferroni_Statistic] < 0.01 ? "**" : "*") : "")")
			plot(repeat([mod(Acrophase - π, 2π)], 2), [b_Term, b_Term - Amplitude], "k-", label = "$(L"$\frac{b + Amplitude}{b - Amplitude}$") = $Amplitude_Ratio")
			plot(repeat([mod(Acrophase - π, 2π)], 2), [b_Term, b_Term + Amplitude], "k-", alpha = 0.3)
			plot(LinRange(0, 2π, size(gene_expression, 1)), repeat([b_Term], size(gene_expression, 1)), "k-")
			plot([0, Acrophase + 2π], repeat([b_Term + Amplitude], 2), "k-", alpha = 0.3)
			plot(repeat([Acrophase], 2), [b_Term + Amplitude, gene_mean - 3*gene_std], "b-", alpha = 0.3)
			title(cosine_output.Gene_Symbols[ii])
			legend(loc = 4)
			if mod(II, number_subplot_columns * Number_of_batches) == 1
				yticks([gene_mean - 3*gene_std, gene_mean, gene_mean + 3*gene_std], ["ȳ - 3σ", "ȳ", "ȳ + 3σ"])
			else
				yticks([gene_mean - 3*gene_std, gene_mean, gene_mean + 3*gene_std], ["", "", ""])
			end
			if II > (number_subplot_columns * Number_of_batches)
					xlabel("Estimated Phases", labelpad = 18)
			end
			if mod(II, number_subplot_columns * Number_of_batches) == 1
					ylabel("Gene Expression", labelpad = 18)
			end
		end
    end
	if had
		ops[:o_covariates] = store_covariates
	end
	subplots_adjust(left=.04,right=.995,bottom=.05,top=.945,wspace=.075,hspace=.1)
end

function Acrophase(GOI, GOI_Ideal, Ideal_Acrophases, R_Squared, Estimated_Acrophases, P_Values, P_Cutoff; space_factor=π/35.5, create_fig = true, subplot_space_factor_ratio = 1.5, subplot_fontsize = 8)
    # close()
	if create_fig
    	fig = figure(figsize = (10, 11))
		ax = PyPlot.axes(polar = true)
	else
		ax = gca()
	end
	ax.spines["polar"].set_visible(false)
    axis([0, 2π, 0, 1.1])
    xticks([π/2, π, 3π/2, 2π], [L"$0$", L"$\frac{3π}{2}$", L"$π$", L"$\frac{π}{2}$"], fontsize = 22)
    yticks([0, 0.5, 1], ["", "", ""])
    ax.yaxis.grid(true)
    ax.xaxis.grid(false)
	# println("\n\n")
	# show(stdout, P_Values)
    comparable_indices = P_Values .< P_Cutoff
	# println("\n\n")
	# show(stdout, comparable_indices)
	# println("\n\n")
    Comparable_GOI = GOI[comparable_indices]
    Comparable_GOI_Ideal = GOI_Ideal[comparable_indices]
    comparable_core_clock_acrophases = Estimated_Acrophases[comparable_indices]
    mouse_comparable_acrophases = Ideal_Acrophases[comparable_indices]
    plot_acrophases = mod.(-(comparable_core_clock_acrophases .- π/2), 2π)
    plot_ideals = mod.(-(mouse_comparable_acrophases .- π/2), 2π)
    scatter(plot_acrophases, ones(length(plot_acrophases)), alpha = 0.7, s = R_Squared[comparable_indices] .* (create_fig ? 1000 : 75), c = "b", label = "Estimated Acrophases")
    scatter(plot_ideals, ones(length(plot_acrophases)) .* 0.5, alpha = 0.7, s = (create_fig ? 75 : 50), c = "orange", label = "Ideal Acrophases")
	if create_fig
    	legend(loc = (0.45, -0.1875))
	end
    significant_acrophase_mean = Circular_Mean(plot_acrophases)
    range_upper, range_lower = mod.(±(significant_acrophase_mean, π/2), 2π)
    significant_ideal_acrophase_mean = Circular_Mean(plot_ideals)
    range_upper_ideal, range_lower_ideal = mod.(±(significant_ideal_acrophase_mean, π/2), 2π)
    if range_upper > range_lower
        acrophases_in_range_logical = range_lower .< plot_acrophases .<= range_upper
    else
        acrophases_in_range_logical = .!(range_upper .< plot_acrophases .<= range_lower)
    end
    if range_upper_ideal > range_lower_ideal
        ideal_acrophases_in_range_logical = range_lower_ideal .< plot_ideals .<= range_upper_ideal
    else
        ideal_acrophases_in_range_logical = .!(range_upper_ideal .< plot_ideals .<= range_lower_ideal)
    end
    in_range_sig_acrophases_logical = acrophases_in_range_logical
    in_range_sig_ideal_acrophases_logical = ideal_acrophases_in_range_logical
    acrophase_in_range_mean = Circular_Mean(plot_acrophases[in_range_sig_acrophases_logical])
    ideal_acrophase_in_range_mean = Circular_Mean(plot_ideals[in_range_sig_ideal_acrophases_logical])
    closest_gene_index = findmin(acos.(cos.(acrophase_in_range_mean .- plot_acrophases)))[2]
    closest_ideal_gene_index = findmin(acos.(cos.(ideal_acrophase_in_range_mean .- plot_ideals)))[2]
    middle_gene_acrophase = plot_acrophases[closest_gene_index]
    ideal_middle_gene_acrophase = plot_ideals[closest_ideal_gene_index]
    annotate(Comparable_GOI[closest_gene_index], xy = [middle_gene_acrophase, 1], xytext = [middle_gene_acrophase, (create_fig ? 1.35 : 1.85)], arrowprops=Dict("arrowstyle"=>"->", "facecolor"=>"grey"), fontsize = (create_fig ? 12 : subplot_fontsize))
    annotate(Comparable_GOI_Ideal[closest_ideal_gene_index], xy = [ideal_middle_gene_acrophase, 0.5], xytext = [ideal_middle_gene_acrophase, (create_fig ? 0.75 : 0.85)], arrowprops=Dict("arrowstyle"=>"->", "facecolor"=>"grey"), fontsize = (create_fig ? 12 : subplot_fontsize))
    distance_from_middle_annotation = middle_gene_acrophase .- plot_acrophases
    sig_phases_larger_middle_logical = distance_from_middle_annotation .< 0
    sig_phases_smaller_middle_logical = distance_from_middle_annotation .> 0
    distance_from_ideal_middle_annotation = ideal_middle_gene_acrophase .- plot_ideals
    sig_phases_larger_ideal_middle_logical = distance_from_ideal_middle_annotation .< 0
    sig_phases_smaller_ideal_middle_logical = distance_from_ideal_middle_annotation .> 0
    sig_larger_phases = plot_acrophases[sig_phases_larger_middle_logical]
    sig_smaller_phases = plot_acrophases[sig_phases_smaller_middle_logical]
    sig_larger_ideal_phases = plot_ideals[sig_phases_larger_ideal_middle_logical]
    sig_smaller_ideal_phases = plot_ideals[sig_phases_smaller_ideal_middle_logical]
    sorted_sig_larger_phases = sort(sig_larger_phases)
    sorted_sig_smaller_phases = sort(sig_smaller_phases, rev = true)
    sorted_sig_larger_ideal_phases = sort(sig_larger_ideal_phases)
    sorted_sig_smaller_ideal_phases = sort(sig_smaller_ideal_phases, rev = true)
    annotation_x_vals_larger = deepcopy(sorted_sig_larger_phases)
    nearest_neighbor = diff(vcat(middle_gene_acrophase, annotation_x_vals_larger))
    nearest_neighbor_too_close_logical = nearest_neighbor .< space_factor
    length(nearest_neighbor_too_close_logical) > 1 ? too_close = |(nearest_neighbor_too_close_logical...) : too_close = false
    while too_close
        annotation_x_vals_larger .+= space_factor .* nearest_neighbor_too_close_logical
        nearest_neighbor = diff(vcat(middle_gene_acrophase, annotation_x_vals_larger))
        nearest_neighbor_too_close_logical = nearest_neighbor .< space_factor
        too_close = |(nearest_neighbor_too_close_logical...)
    end
    annotation_x_vals_smaller = deepcopy(sorted_sig_smaller_phases)
    nearest_neighbor = diff(vcat(middle_gene_acrophase, annotation_x_vals_smaller))
    nearest_neighbor_too_close_logical = nearest_neighbor .> space_factor
    length(nearest_neighbor_too_close_logical) > 1 ? too_close = |(nearest_neighbor_too_close_logical...) : too_close = false
    while too_close
        annotation_x_vals_smaller .-= space_factor .* nearest_neighbor_too_close_logical
        nearest_neighbor = diff(vcat(middle_gene_acrophase, annotation_x_vals_smaller))
        nearest_neighbor_too_close_logical = nearest_neighbor .> -space_factor
        too_close = |(nearest_neighbor_too_close_logical...)
    end
    for mwm in 1:length(sorted_sig_larger_phases)
        desired_phases = sorted_sig_larger_phases[mwm]
        desired_annotation_phases = annotation_x_vals_larger[mwm]
        desired_gene_index = findall(in(desired_phases), plot_acrophases)[1]
        annotate(Comparable_GOI[desired_gene_index], xy = [desired_phases, 1], xytext = [desired_annotation_phases, (create_fig ? 1.35 : 1.85)], arrowprops=Dict("arrowstyle"=>"->"), fontsize = (create_fig ? 12 : subplot_fontsize))
    end
    for wmw in 1:length(sorted_sig_smaller_phases)
        desired_phases = sorted_sig_smaller_phases[wmw]
        desired_annotation_phases = annotation_x_vals_smaller[wmw]
        desired_gene_index = findall(in(desired_phases), plot_acrophases)[1]
        annotate(Comparable_GOI[desired_gene_index], xy = [desired_phases, 1], xytext = [desired_annotation_phases, (create_fig ? 1.35 : 1.85)], arrowprops=Dict("arrowstyle"=>"->"), fontsize = (create_fig ? 12 : subplot_fontsize))
    end
    ideal_annotation_x_vals_larger = deepcopy(sorted_sig_larger_ideal_phases)
    nearest_neighbor = diff(vcat(ideal_middle_gene_acrophase, ideal_annotation_x_vals_larger))
    nearest_neighbor_too_close_logical = nearest_neighbor .< (space_factor * subplot_space_factor_ratio)
    length(nearest_neighbor_too_close_logical) > 1 ? too_close = |(nearest_neighbor_too_close_logical...) : too_close = false
    while too_close
        ideal_annotation_x_vals_larger .+= (space_factor * subplot_space_factor_ratio) .* nearest_neighbor_too_close_logical
        nearest_neighbor = diff(vcat(ideal_middle_gene_acrophase, ideal_annotation_x_vals_larger))
        nearest_neighbor_too_close_logical = nearest_neighbor .< (space_factor * subplot_space_factor_ratio)
        too_close = |(nearest_neighbor_too_close_logical...)
    end
    ideal_annotation_x_vals_smaller = deepcopy(sorted_sig_smaller_ideal_phases)
    nearest_neighbor = diff(vcat(ideal_middle_gene_acrophase, ideal_annotation_x_vals_smaller))
    nearest_neighbor_too_close_logical = nearest_neighbor .> -(space_factor * subplot_space_factor_ratio)
    length(nearest_neighbor_too_close_logical) > 1 ? too_close = |(nearest_neighbor_too_close_logical...) : too_close = false
    while too_close
        ideal_annotation_x_vals_smaller .-= (space_factor * subplot_space_factor_ratio) .* nearest_neighbor_too_close_logical
        nearest_neighbor = diff(vcat(ideal_middle_gene_acrophase, ideal_annotation_x_vals_smaller))
        nearest_neighbor_too_close_logical = nearest_neighbor .> -(space_factor * subplot_space_factor_ratio)
        too_close = |(nearest_neighbor_too_close_logical...)
    end
    c = 1
    while c <= length(sorted_sig_larger_ideal_phases)
        desired_phases = sorted_sig_larger_ideal_phases[c]
        desired_gene_indices = findall(in(desired_phases), plot_ideals)
        sig_desired_gene_indices = desired_gene_indices
        # sig_desired_gene_indices = intersect(desired_gene_indices, findall(l_sig_p_values[kk]))
        d = 1
        while d <= length(sig_desired_gene_indices)
            desired_annotation_phases = ideal_annotation_x_vals_larger[c + d - 1]
            annotate(Comparable_GOI_Ideal[sig_desired_gene_indices[d]], xy = [desired_phases, 0.5], xytext = [desired_annotation_phases, (create_fig ? 0.75 : 0.85)], arrowprops=Dict("arrowstyle"=>"->"), fontsize = (create_fig ? 12 : subplot_fontsize))
            d += 1
        end
        c += (d - 1)
    end
    c = 1
    while c <= length(sorted_sig_smaller_ideal_phases)
        desired_phases = sorted_sig_smaller_ideal_phases[c]
        desired_gene_indices = findall(in(desired_phases), plot_ideals)
        sig_desired_gene_indices = desired_gene_indices
        # sig_desired_gene_indices = intersect(desired_gene_indices, findall(l_sig_p_values[kk]))
        d = 1
        while d <= length(sig_desired_gene_indices)
            desired_annotation_phases = ideal_annotation_x_vals_smaller[c + d - 1]
            annotate(Comparable_GOI_Ideal[sig_desired_gene_indices[d]], xy = [desired_phases, 0.5], xytext = [desired_annotation_phases, (create_fig ? 0.75 : 0.85)], arrowprops=Dict("arrowstyle"=>"->"), fontsize = (create_fig ? 12 : subplot_fontsize))
            d += 1
        end
        c += (d - 1)
    end
end

function Acrophase_new_version(GOI, cosoutput, options; space_factor=π/35.5)
    close()
    fig = figure(figsize = (10, 11))
    ax = PyPlot.axes(polar = true)
    ax.spines["polar"].set_visible(false)
    axis([0, 2π, 0, 1.1])
    xticks([π/2, π, 3π/2, 2π, 0], [L"$0/2π$", L"$\frac{3π}{2}$", L"$π$", L"$\frac{π}{2}$", ""], fontsize = 22)
    yticks([0, 0.5, 1], ["", "", ""])
    ax.yaxis.grid(true)
    ax.xaxis.grid(false)
	GOI_indices = vcat(map(x -> findall(in([x]), cosinor_regression.Gene_Symbols), GOI)...)
    comparable_indices = cosinor_regression.P_Statistic[GOI_indices] .< p_cutoff
    comparable_GOI_names = GOI[comparable_indices]
    comparable_GOI_ideal_names = GOI_ideal[comparable_indices]
    comparable_GOI_phases = cosinor_regression.Acrophase[GOI_indices][comparable_indices]
    comparable_GOI_ideal_phases = ideal_acrophases[comparable_indices]
    plot_acrophases = mod.(-(comparable_GOI_phases .- π/2), 2π)
    plot_ideals = mod.(-(comparable_GOI_ideal_phases .- π/2), 2π)
    scatter(plot_acrophases, ones(length(plot_acrophases)), alpha = 0.7, s = cosinor_regression.R_Squared[GOI_indices][comparable_indices] .* 1000, c = "b", label = "Estimated Acrophases")
    scatter(plot_ideals, ones(length(plot_acrophases)) .* 0.5, alpha = 0.7, s = 75, c = "orange", label = "Ideal Acrophases")
    legend(loc = (0.45, -0.1875))
	sorted_GOI_phases, sort_index = x_sort(plot_acrophases)
	#sorted_GOI_phases, sort_index = CYCLOPS.x_sort(plot_acrophases)
	distance_array = CircularDifference(sorted_GOI_phases)
	#distance_array = CYCLOPS.CircularDifference(sorted_GOI_phases)
	too_close_array = distance_array .< space_factor
	clean_up_diag = Diagonal(trues(length(sorted_GOI_phases)))
	#clean_up_diag = CYCLOPS.Diagonal(trues(length(sorted_GOI_phases)))
	#=global=# clean_too_close_array = .!clean_up_diag .& too_close_array
	#=global=# indices_to_update_raw = map(x -> mod.(x.I, length(sorted_GOI_phases)), findall(clean_too_close_array))
	#=global=# index_distance = abs.(vcat(map(x -> diff([x...]), indices_to_update_raw)...))
	#=global=# max_index_distance_logical = index_distance .== length(sorted_GOI_phases) - 1
	#=global=# 
	#=global=# indices_to_update_usable = unique(maximum.(indices_to_update_raw))
	#=global=# updated_sorted_GOI_phases = deepcopy(sorted_GOI_phases)
	while sum(clean_too_close_array) >= 1
		#=global=# indices_to_update_raw = map(x -> mod.(x.I, length(sorted_GOI_phases) - 1), findall(clean_too_close_array))
		#=global=# indices_to_update_usable = unique(maximum.(indices_to_update_raw))
		#=global=# updated_sorted_GOI_phases[indices_to_update_usable] .+= space_factor
		#=global=# distance_array = CYCLOPS.CircularDifference(updated_sorted_GOI_phases)
		#=global=# too_close_array = distance_array .< space_factor
		#=global=# clean_too_close_array = .!clean_up_diag .& too_close_array
	end
	#indices_to_update = map(x -> mod.(x.I, 3), findall(clean_too_close_array))
end








function rmHiddenFile(specified_path::String, hidden_file_name::String = ".DS_Store")
	# The purpose of this function is to remove the hidden ".DS_Store" files in a given directory.
	# If the File exists it will be removed, otherwise nothing happens.
	all_files = readdir(specified_path) # Get all the file names
	if sum(all_files .== hidden_file_name) >= 1 # If a file name matches hidden_file_name exatly...
		rm(joinpath(specified_path, hidden_file_name)) # ...remove that file from the specified_path
	end
end

function rmHiddenFile(specified_path::Array{String,1}, hidden_file_name::String = ".DS_Store")
	# When specified_path is an Array{String, 1} instead of String the function is called iteratively on each element using the String method of the function.
	for ii in specified_path # For each string in the array...
		rmHiddenFile(ii, hidden_file_name) # ...remove the hidden_file_name
	end
end

function GetMultipleXValsFitFiles(project_path::String, hidden_file_name::String = ".DS_Store")
	# Given a project path which contains the results folders of at least one cross validation find the "Fits" subfolder and extract all file names.
	rmHiddenFile(project_path, hidden_file_name) # Remove the hidden_file_name from the project_path
    X_VAL_PATHS = joinpath.(project_path, readdir(project_path)) # Get the paths to all cross validation results folders inside the project_path
    FIT_PATHS = Array{String, 1}(joinpath.(X_VAL_PATHS, "Fits")) # Create FIT_PATHS from X_VAL_PATHS by appending "/Fits"
    FIT_PATH_FILES = Array{Array{String, 1}}(readdir.(FIT_PATHS)) # Extract all file names inside all "Fits" folders for various cross validations inside project_path
    return FIT_PATH_FILES, FIT_PATHS # Return all files inside all "Fits" folders and return all paths used to get said files.
end

function GetMultipleXValsFitFiles(project_path::Array{String, 1}, hidden_file_name::String = ".DS_Store")
	# When project_path is an Array{String, 1} instead of String the function is called iteratively on each element using the String method of the function.
	FIT_PATH_FILES_Array, FIT_PATHS_Array = Array{Array{String, 1}}([]), String[] # Initialize empty arrays for the all FIT_PATH_FILES and for all FIT_PATHS.
	for ii in project_path # For each string in project_path...
		FIT_PATH_FILES, FIT_PATHS = GetMultipleXValsFitFiles(ii, hidden_file_name) # ...get FIT_PATH_FILES and FIT_PATHS
		append!(FIT_PATH_FILES_Array, FIT_PATH_FILES) # Append FIT_PATH_FILES to FIT_PATH_FILES_Array
		append!(FIT_PATHS_Array, FIT_PATHS) # Append FIT_PATHS to FIT_PATHS_Array
	end
	return FIT_PATH_FILES_Array, FIT_PATHS_Array # Finally, return FIT_PATH_FILES_Array and FIT_PATHS_Array
end

function GetMultipleXValsMetricAverages(fit_paths::Array{String, 1}, fit_path_files::Array{Array{String, 1}}, x_val_file_regex::Regex, on_col_name::Symbol, on_header::Int)
	# The purpose of this function is to find the cross validation results file within each fit_path and return only the averages for each metric.
	# fit_paths and fit_path_files are the values returned from GetMultipleXValsFitFiles.
	# fit_paths are the paths leading to the respective fit_path_files.
	# fit_path_files is an array of string arrays. Each string array contains all the file names in the "Fits" folder of the results folder for a cross validation.
	# x_val_file_regex is the regular expression (Regex) used to identify the cross validation results file.
	# on_col_name is a Symbol which indicates which column will be the left most column.
	# on_header is an integer that indicates which row of the CSV file will be the header row.
    x_val_file_logical_array = map(x -> map(y -> isa(match(x_val_file_regex, y), RegexMatch), x), fit_path_files) # Using x_val_file_regex get a logical for the cross validation file in each string array contained within fit_path_files.
    x_val_file_index = vcat(findall.(x_val_file_logical_array)...) # Convert each logical array into a single integer and collect all integers.
    x_val_files = [fit_path_files[ii][x_val_file_index[ii]] for ii in 1:length(x_val_file_index)] # Using the x_val_file_index get the names of all cross validation results files.
    x_val_file_paths = joinpath.(fit_paths, x_val_files) # Combine the x_val_files with their respective fit_paths.
    x_val_csvs = map(x -> CSV.read(x, DataFrame, header = on_header), x_val_file_paths) # read the x_val_file_paths to a CSV, using on_header to determine the header row.
    x_val_averages = innerjoin(map(x -> x[:, [on_col_name, :Average]], x_val_csvs)..., on = on_col_name, makeunique=true) # Combine all x_val_csvs keeping only the metric names and averages, using on_col_name to determine the left most column.
    return x_val_averages # Finally, return the dataframe containing all metric names and all averages for each cross validation.
end

function PlotMultipleXValsMetricAverages(project_path, scatter_l::Bool = false;
	metrics = nothing, subplot_r::Float64 = 0.675, subplot_l::Float64 = 0.1, plot_width::Int = 15, plot_height::Int = 10,
	x_val_file_regex::Regex = r"Cross_Validation_Correlation_and_Error_Metrics.*", on_col_name::Symbol = :Metric_Name, on_header::Int = 2,
	x_axis_parameter::String = "train_collection_time_balance", x_axis_value::String = "((?:_\\d+)+)", backup_parameter = nothing, x_axis_scale::String = "log", switch_axis_label::Bool = true,
	create_fig::Bool = true)

	# The purpose of this function is to visualize the various metrics of multiple cross validations with a single changing training parameter.
	# project_path is a string or array of strings which contain cross validation results folders.
	# scatter_l is a logical which controls whether the plot is a scatter plot or a connected line graph. The default (false) means that the final plot is a line graph.
	# metrics is either nothing, a regular expression or an array of regular expression which can be used to plot only specific metrics of the cross validation results.
	# Example for metrics: r"Sample_Circular_Error(?!_Standard)|(?:Fischer_Correlation(?!_Ranked))" plots only those metrics containing Sample_Circular_Error in the name but is not followed by the word _Standard OR the Fischer_Correlations but not those followed by the word _Ranked.
	# subplot_r and subplot_l are Floats between 0 and 1 and indicate how much white space is left on the right and left side respectively of the figure. subplot_r must be larger than subplot_l.
	# plot_width determines how wide the figure is, while plot_height determines the height of the figure.
	# The changing training parameter is a keyword argument (x_axis_parameter) defined as a string, which will later become a regular expression. This is just the parameter, not the associated value.
	# The value of the changing training parameter (x_axis_value) is a also a string that will also later become a regular expression, joined to the x_axis_parameter.
	# A backup_parameter is also given in cases where parameter values in the DefaultDict() do not match the parameter names in the file name.
	# x_axis_scale lets the user decide if the plot axis will be log or linear.
	# For an explanation of x_val_file_regex, on_col_name and/or on_header read the documentation in GetMultipleXValsMetricAverages().

    FIT_PATH_FILES, FIT_PATHS = GetMultipleXValsFitFiles(project_path) # Get all the FIT_PATH_FILES and all the FIT_PATHS for a single absolute or relative path string or array or absolute or relative path strings contained by project_path.
    X_VAL_AVERAGES = GetMultipleXValsMetricAverages(FIT_PATHS, FIT_PATH_FILES, x_val_file_regex, on_col_name, on_header) # Combine the averages of all metrics in all cross validation results files.
    
    x_val_averages_array_array = mapslices(x -> [x], Array(X_VAL_AVERAGES[:, 2:end]), dims = 2) # Make each row (a cross validation metric like Fischer Correlation or Circular Error) its own array.
    x_val_metrics = X_VAL_AVERAGES[:, 1] # The first column of the X_VAL_AVERAGES contains the names of the metrics for each row.
	x_axis_tick_values = GetAllXValidationXAxisValues(FIT_PATHS, x_axis_parameter, x_axis_value, backup_parameter) # From the FIT_PATHS, using x_axis_parameter, x_axis_value and backup_parameter, get the x axis values associated with each column of X_VAL_AVERAGES.
	# println("size of x_axis_tick_values = $(size(x_axis_tick_values))")
	# println(x_axis_tick_values)
	sorted_x_axis_tick_values, sort_indices = x_sort(x_axis_tick_values) # Sort the x axis values for the line graph.
	resorted_indices = unique(sort_indices)
	# println("size of sorted_x_axis_tick_values = $(size(sorted_x_axis_tick_values))")
	# resorted_indices = findXinY(x_axis_tick_values, sorted_x_axis_tick_values) # Find the order of the unsorted x axis values in the sorted x axis values for plotting.
	# println("size of resorted_indices = $(size(resorted_indices))")
	# println(resorted_indices)

    if isa(metrics, Array) # If metrics is an array of Regex...
        plot_metric_indices = Any[] # ...initialize plot_metric_indices array...
        for ii in metrics # ...and for each element in metrics...
            append!(plot_metric_indices, findall(isa.(map(x->match(ii, x), x_val_metrics), RegexMatch))) # ...find all the indices that resulted in matches and append them to plot_metric_indices.
        end
    elseif !isa(metrics, Nothing) # If metrics is not an array and not nothing...
        plot_metric_indices = Any[] # ...initialize plot_metric_indices array...
        append!(plot_metric_indices, findall(isa.(map(x->match(metrics, x), x_val_metrics), RegexMatch))) # ...and find all the indices taht resulted in a match and append them to plot_metric_indices.
    else
        plot_metric_indices = 1:length(x_val_averages_array_array) # ...otherwise incldue all metric indices.
    end
    sort!(plot_metric_indices) # Sort all metric indices to retain order from the original file.

	if create_fig
	    fig = figure(figsize = (plot_width, plot_height)) # Create a figure that has specified dimension plot_width and plot_height.
		subplots_adjust(right = subplot_r, left = subplot_l) # Adjust the white space to the left and right of the figure using subplot_r and subplot_l.
	end
    for jj in plot_metric_indices # For each metric index in plot_metric_indices...
        ii = x_val_averages_array_array[jj] # ...grab the respective array...
        if scatter_l
            scatter(sorted_x_axis_tick_values, ii[resorted_indices], alpha = 0.6, s = 14, label = x_val_metrics[jj]) # ...and create a scatter plot...
        else
#			println("size of resorted_indices = $(size(resorted_indices))")
#			println("size of ii = $(size(ii))")
#			println("size of ii[resorted_indices] = $(size(ii[resorted_indices]))")
#			println("size of sorted x axis tick values = $(size(sorted_x_axis_tick_values))")
#			println("size of x_val_metrics[jj] = $(size([x_val_metrics[jj]]))")
            plot(sorted_x_axis_tick_values, ii[resorted_indices], alpha = 0.6, label = x_val_metrics[jj]) # ...or a line graph.
        end
    end
	if x_axis_scale == "log" # If x_axis_scale is log...
		current_axis = gca() # ...get the current_axis...
		current_axis.set_xscale(x_axis_scale) # ...and make the x axis a log scale.
	end
	if !isa(backup_parameter, Nothing) & switch_axis_label
		x_axis_parameter = backup_parameter
	end
	title(join(["Cross Validation Metric Values as a Function of ", "\"", x_axis_parameter, "\"", " Parameter"])) # The title reflects the parameter (x_axis_parameter) whose value was changed between experiments.
	xlabel(replace(x_axis_parameter, "_" => " ")) # Use the x_axis_parameter to label the x axis.
	ylabel("Metric Value") # They y axis will contain various numerical value types.
    xticks(sorted_x_axis_tick_values, string.(sorted_x_axis_tick_values), rotation = -90) # Lable the x ticks according to the parameter value found in the file name.
    legend(loc = (1.05, 0)) # Generate a legend and put it in the bottom right of the figure.
end

function GetXValidationXAxisValues(fit_path::String, x_axis_parameter::String, x_axis_value::String, backup_parameter)
    x_axis_regex = Regex(join([x_axis_parameter, x_axis_value]))
	fit_path_x_axis_match = match(x_axis_regex, fit_path)
	# show(fit_path_x_axis_match)
    if !isa(fit_path_x_axis_match, Nothing)
        x_axis_value_string = fit_path_x_axis_match.captures[1]
        x_value = parse(Float64, replace(x_axis_value_string[2:end], "_"=>"."))
    else
        default_dict = DefaultDict()
        default_dict_keys = collect(keys(default_dict))
        default_dict_keys_string = string.(default_dict_keys)
        if !isa(backup_parameter, Nothing)
            parameter_of_interest_logical = isa.(map(x -> match(Regex(backup_parameter), x), default_dict_keys_string), RegexMatch)
        else
            parameter_of_interest_logical = isa.(map(x -> match(Regex(x_axis_parameter), x), default_dict_keys_string), RegexMatch)
        end
        x_value = default_dict[default_dict_keys[parameter_of_interest_logical][1]]
    end

    return x_value
end

function GetAllXValidationXAxisValues(fit_paths::Array{String,1}, x_axis_parameter::String, x_axis_value::String, backup_parameter)
    x_values = Float64[]
    for ii in 1:length(fit_paths)
        jj = fit_paths[ii]
        x_value = GetXValidationXAxisValues(jj, x_axis_parameter, x_axis_value, backup_parameter)
        append!(x_values, [x_value])
    end
    return x_values
end


end
