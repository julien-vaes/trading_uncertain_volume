# NAME: mean_cvar.jl
# AUTHOR: Julien Vaes
# DATE: October 10, 2019
# DESCRIPTION: functions in order to get the optimal execution strategy based on Monte-Carlo simulations and the Rockafellar approach.

module MeanCVaRModule

###################
## Load Packages ##
###################

using Distributed
using Statistics
using LinearAlgebra
using JuMP
using Optim
using Gurobi
using Random
using Distributions

############################
## Load Personal Packages ##
############################

using ..UncertaintyStructureModule
using ..MarketDetailsModule
using ..StrategyModule
using ..TraderModule
using ..SimulationParametersModule
using ..NEHelpFunctionsModule
using ..HelpFilesModule
using ..MeanVarianceModule
using ..TradingCostModule
using ..HelpFilesModule

######################
## Export functions ##
######################

export get_optimal_strategy_mean_CVaR_Optim
export get_optimal_strategy_mean_CVaR_Gurobi

#####################
## Global constant ##
#####################

const ourGurobiEnvironment = Gurobi.Env()

######################
## Module variables ##
######################

# parameters for the convergence of the optimisation algorithm provided by Optim 
ourXTolerance = 1e-5 # absolute tolerance in changes of the input vector x, in infinity norm. Defaults to 0.0.
ourFTolerance = 0.0  # relative tolerance in changes of the objective value. Defaults to 0.0.
ourGTolerance = 1e-8 # absolute tolerance in the gradient, in infinity norm. Defaults to 1e-8. For gradient free methods, this will control the main convergence tolerance, which is solver specific.

# initialises the variables for the price and forecast update realisations
ourPricesMovesRealisations      = nothing
ourForecastUpdatesRealisations = nothing

# the initial number of samples used to compute the CVaR
ourInitialNSamplesToEvaluateCVaR = 10^4

######################
## Module functions ##
######################

"""
```
vector_in_list_approx(aVector,aListVector)
```

TODO function description.

### Arguments
* `aVector`: TODO.
* `aListVectors`: TODO.
"""
function vector_in_list_approx(aVector,aListVectors)
	for myVectorInList in aListVectors
		if isapprox(aVector,myVectorInList)
			return true
		end
	end
	return false
end

############################
## Monte-Carlo estimation ##
############################

"""
merge_stats(aDic1,aDic2)

The funcion merges the details contained in the dictionaries `aDic1` and `aDic2`.
This function is useful when merging the estimates obtained while doing parallel computing.

E[ Cost(y,beta) ],

CVaR_α[ Cost(y,beta) ] = min { u + (1/α) * E [ Cost(y,beta) − u ]+ },

#### Arguments
* `aDic1`: first dictionary that will be merged.
* `aDic2`: second dictionary that will be merged.
"""
function merge_stats(aDic1,aDic2)

	myMergedStats = Dict()

	#####################
	# Float64 of Samples #
	#####################

	# merges the number of samples computed

	# gets the details of both dictionaries
	myN1 = aDic1["NumberOfSamples"]
	myN2 = aDic2["NumberOfSamples"]

	# merges the details of both dictionaries
	myN = myN1 + myN2

	###############
	# Expectation #
	###############

	# merges the value and gradient in terms of y and beta of the expectation of the trading cost, i.e. E [ Cost(y,beta) ]

	# gets the details of both dictionaries
	myExpectation1             = aDic1["Expectation"]
	myExpectation2             = aDic2["Expectation"]
	myExpectationGradientY1    = aDic1["ExpectationGradientY"]
	myExpectationGradientY2    = aDic2["ExpectationGradientY"]
	myExpectationGradientBeta1 = aDic2["ExpectationGradientBeta"]
	myExpectationGradientBeta2 = aDic2["ExpectationGradientBeta"]

	# merges the details of both dictionaries
	myExpectation             = ( myN1*myExpectation1             + myN2*myExpectation2             ) / myN
	myExpectationGradientY    = ( myN1*myExpectationGradientY1    + myN2*myExpectationGradientY2    ) / myN
	myExpectationGradientBeta = ( myN1*myExpectationGradientBeta1 + myN2*myExpectationGradientBeta2 ) / myN

	############
	# Variance #
	############

	# merges the values of the variance of the trading cost, i.e. V [ Cost(y,beta) ]

	## gets the details of both dictionaries
	myVariance1 = aDic1["Variance"]
	myVariance2 = aDic2["Variance"]

	## creates a dictionary with the merged details
	myVariance = ( (myN1*(myExpectation1[1]^2+myVariance1)) + (myN2*(myExpectation2[1]^2+myVariance2)) )/myN - myExpectation^2

	#######################################
	# Conditional Expectation Greater VaR #
	#######################################

	# merges the value and gradient in terms of y and beta of the expectation of the maximum between the trading cost minus the VaR estimate, i.e. u, and 0, i.e. E [ Cost(y,beta) − u ]+

	# gets the details of both dictionaries
	myExpectationGreaterVaR1             = aDic1["ExpectationGreaterVaR"]
	myExpectationGreaterVaR2             = aDic2["ExpectationGreaterVaR"]
	myExpectationGreaterVaRGradientY1    = aDic1["ExpectationGreaterVaRGradientY"]
	myExpectationGreaterVaRGradientY2    = aDic2["ExpectationGreaterVaRGradientY"]
	myExpectationGreaterVaRGradientBeta1 = aDic1["ExpectationGreaterVaRGradientBeta"]
	myExpectationGreaterVaRGradientBeta2 = aDic2["ExpectationGreaterVaRGradientBeta"]

	# merges the details of both dictionaries
	myExpectationGreaterVaR             = ( myN1*myExpectationGreaterVaR1             + myN2*myExpectationGreaterVaR2             ) / myN
	myExpectationGreaterVaRGradientY    = ( myN1*myExpectationGreaterVaRGradientY1    + myN2*myExpectationGreaterVaRGradientY2    ) / myN
	myExpectationGreaterVaRGradientBeta = ( myN1*myExpectationGreaterVaRGradientBeta1 + myN2*myExpectationGreaterVaRGradientBeta2 ) / myN

	#######
	# VaR #
	#######

	# merges the gradient in terms of u, i.e. the estimate of the Value-at-Risk (VaR), of E [ Cost(y,beta) − u ]+

	# gets the details of both dictionaries
	myVaRGradient1 = aDic1["VaRGradient"]
	myVaRGradient2 = aDic2["VaRGradient"]

	# merges the details of both dictionaries
	myVaRGradient  = ( myN1*myVaRGradient1 + myN2*myVaRGradient2 ) / myN

	##########
	# Output #
	##########

	# creates a dictionary with the merged details
	myMergedStats["NumberOfSamples"]                   = myN
	myMergedStats["Expectation"]                       = myExpectation
	myMergedStats["Variance"]                          = myVariance
	myMergedStats["ExpectationGradientY"]              = myExpectationGradientY
	myMergedStats["ExpectationGradientBeta"]           = myExpectationGradientBeta
	myMergedStats["ExpectationGreaterVaR"]             = myExpectationGreaterVaR
	myMergedStats["ExpectationGreaterVaRGradientY"]    = myExpectationGreaterVaRGradientY
	myMergedStats["ExpectationGreaterVaRGradientBeta"] = myExpectationGreaterVaRGradientBeta
	myMergedStats["VaRGradient"]                       = myVaRGradient

	return myMergedStats
end

"""
get_mean_cvar_rockafellar_value_and_partial_derivatives(
aStrategy::Strategy,
aVaR::Float64,
aTrader::Trader,
aSimulationParameters::SimulationParameters
)

returns the value and gradient of each component of the optimisation problem to solve in order to minimise mean-CVaR under price and volume uncertainty, i.e.

min_{y,beta,u} |E[C(y,beta)] + u + |E[ C(y,beta)-u ]+,

where here:
* u is given by `aVaR`.
* y is given by the trading plan contained in `aStrategy`.
* beta is given by the redistribution matrix contained in `aStrategy`.


#### Arguments
* `aStrategy::Strategy`: a structure containing the details of the trader's strategy, i.e. the investement decisions and the corresponding redistribution coefficients.
* `aVaR::Float64`: a current value for u in the minimisation associated to CVaR via Rockafellar approach.
* `aTrader::Trader`: a structure containing all the details of a trader.
* `aSimulationParameters::SimulationParameters`: a structure containing all the details of the parameters used to compute the optimal trading plan such as the method to use, the number of samples in the CVaR estimation.
"""
function get_mean_cvar_rockafellar_value_and_partial_derivatives(;
								 aTraderIndex::Int64,
								 aTraders::Array{Trader,1},
								 aStrategies::Array{Strategy,1},
								 aVaR::Float64,
								 aSimulationParameters::SimulationParameters
								 )
	# Parameters
	myNSamples = get_CVaR_optimisation_n_samples(aSimulationParameters)
	myNSamplesPerIteration = get_CVaR_optimisation_n_samples_per_iteration(aSimulationParameters)

	# Float64 of times one needs to generate `myNSamplesPerIteration` realisations in order to have `myNSamples` samples for the estimation of CVaR.
	myNIterations = Int(floor(myNSamples/(myNSamplesPerIteration)))

	# Float64 of realisations needed in the last iteration in order to have `myNSamples` samples for the estimation of CVaR.
	myNSamplesInLastIteration = myNSamples-myNIterations*myNSamplesPerIteration

	# Vector with the number of samples to generate per iterate
	myNSamplesToRun = myNSamplesPerIteration*ones(Int64,myNIterations)
	if myNSamplesInLastIteration > 0
		push!(myNSamplesToRun,myNSamplesInLastIteration)
	end

	########################
	## PARALLEL COMPUTING ##
	########################

	myStats = @distributed (merge_stats) for myIter in eachindex(myNSamplesToRun)

		myTradingCostRealisations, myTradingCostGradientYRealisations, myTradingCostGradientBetaRealisations = get_trading_cost_value_and_partial_derivatives(
																				      aTraderIndex          = aTraderIndex,
																				      aTraders              = aTraders,
																				      aStrategies           = aStrategies,
																				      aSimulationParameters = aSimulationParameters,
																				      aNSamples             = myNSamplesToRun[myIter],
																				      aSeed                 = myIter
																				      )
		myLocalStats = Dict()

		# Float64 of realisations computed
		myLocalStats["NumberOfSamples"] = myNSamplesToRun[myIter]

		###############
		# Expectation #
		###############

		# Expecetation of the trading cost: |E[ Cost ].
		myLocalStats["Expectation"] = mean(myTradingCostRealisations)

		# Gradient of the expecetation of the trading cost in terms of y: ∇_y |E[ Cost ].
		myLocalStats["ExpectationGradientY"] = mean(myTradingCostGradientYRealisations,dims=1)

		# Gradient of the expecetation of the trading cost in terms of β: ∇_β |E[ Cost ].
		myLocalStats["ExpectationGradientBeta"] = mean(myTradingCostGradientBetaRealisations,dims=1)

		############
		# Variance #
		############

		# Variance of the trading cost: |V[ Cost ].
		myLocalStats["Variance"] = var(myTradingCostRealisations)

		###########################
		# Worst cases Expectation #
		###########################

		# Worst cases expectation used in Rockafellar approach, i.e. |E[ (Cost-aVaR)+ ].

		# Indices with a the trading cost greater than aVaR: |E[ (Cost-aVaR)+ ]
		myVectorForWhichTradingCostOverVaR = (myTradingCostRealisations .>= aVaR)

		# shifts the trading cost realisations by aVaR
		myTradingCostRealisations .-= aVaR

		# Worst cases expecetation of the trading cost: |E[ (Cost-aVaR)+ ].
		@views myLocalStats["ExpectationGreaterVaR"] = mean(myVectorForWhichTradingCostOverVaR.*myTradingCostRealisations)

		# Gradient of the worst cases expecetation of the trading cost in terms of y: ∇_y |E[ (Cost-aVaR)+ ].
		@views myLocalStats["ExpectationGreaterVaRGradientY"] = mean(myVectorForWhichTradingCostOverVaR.*myTradingCostGradientYRealisations,dims=1)

		# Gradient of the worst cases expecetation of the trading cost in terms of β: ∇_β |E[ (Cost-aVaR)+ ].
		@views myLocalStats["ExpectationGreaterVaRGradientBeta"] = mean(myVectorForWhichTradingCostOverVaR.*myTradingCostGradientBetaRealisations,dims=1)

		#######
		# VaR #
		#######

		# Expectation of the slope of u, cf. Rockafellar approach.
		@views myLocalStats["VaRGradient"] = -mean(myVectorForWhichTradingCostOverVaR)

		##########
		# Output #
		##########

		# What is returned by each iteration and will be used in the aggregating function `merge_stats`
		myLocalStats
	end

	return myStats
end

function get_mean_cvar_rockafellar_value_and_partial_derivatives_bis(;
								     aTraderIndex::Int64,
								     aTraders::Array{Trader,1},
								     aStrategies::Array{Strategy,1},
								     aVaR::Float64,
								     aSimulationParameters::SimulationParameters
								     )
	# Parameters
	myNSamples = get_CVaR_optimisation_n_samples(aSimulationParameters)
	myNWorkers = Distributed.nworkers() # the number of workers for the parallel computing
	myNSamplesToRun = fill( floor(Int64, myNSamples / myNWorkers), myNWorkers)
	if myNWorkers > 1 
		myNSamplesToRun[end] = myNSamples - sum(myNSamplesToRun[1:end-1])
	end

	########################
	## PARALLEL COMPUTING ##
	########################

	myStats = @distributed (merge_stats) for myIter in eachindex(myNSamplesToRun)

		get_trading_cost_value_and_partial_derivatives_statistical_values(
										  aTraderIndex          = aTraderIndex,
										  aTraders              = aTraders,
										  aStrategies           = aStrategies,
										  aSimulationParameters = aSimulationParameters,
										  aNSamples             = myNSamplesToRun[myIter],
										  aSeed                 = myIter,
										  aVaR                  = aVaR
										  )
	end

	return myStats
end

"""
get_objective_function_value_and_gradient(
aStrategy::Strategy,
aVaR::Float64,
aTrader::Trader,
aSimulationParameters::SimulationParameters
)

returns the value and gradient of the optimisation problem to solve in order to minimise mean-CVaR under price and volume uncertainty, i.e.

min_{y,beta,u} |E[C(y,beta)] + u + |E[ (C(y,beta)-u) ],

where here:
* u is given by `aVaR`.
* y is given by the trading plan contained in `aStrategy`.
* beta is given by the redistribution matrix contained in `aStrategy`.

#### Arguments
* `aStrategy::Strategy`: a structure containing the details of the trader's strategy, i.e. the investement decisions and the corresponding redistribution coefficients.
* `aVaR::Float64`: a current value for u in the minimisation associated to CVaR via Rockafellar approach.
* `aTrader::Trader`: a structure containing all the details of a trader.
* `aSimulationParameters::SimulationParameters`: a structure containing all the details of the parameters used to compute the optimal trading plan such as the method to use, the number of samples in the CVaR estimation.  """
function get_objective_function_value_and_gradient(;
						   aTraderIndex::Int64,
						   aTraders::Array{Trader,1},
						   aStrategies::Array{Strategy,1},
						   aVaR::Float64,
						   aSimulationParameters::SimulationParameters
						   )

	# gets the trader
	myTrader = aTraders[aTraderIndex]

	# Risk-aversion of the trader
	myRiskAversion = get_risk_aversion(myTrader)

	# CVaR parameter
	myAlpha = get_alpha(myTrader)

	# gets the estimates of the expectation and CVaR of the trading cost
	myDictDetails = get_mean_cvar_rockafellar_value_and_partial_derivatives(
										      aTraders              = aTraders,
										      aTraderIndex          = aTraderIndex,
										      aStrategies           = aStrategies,
										      aVaR                  = aVaR,
										      aSimulationParameters = aSimulationParameters
										      )

	#= @time myDictDetails = get_mean_cvar_rockafellar_value_and_partial_derivatives_bis( =#
	#= 										  aTraders              = aTraders, =#
	#= 										  aTraderIndex          = aTraderIndex, =#
	#= 										  aStrategies           = aStrategies, =#
	#= 										  aVaR                  = aVaR, =#
	#= 										  aSimulationParameters = aSimulationParameters =#
	#= 										  ) =#

	# gets the value to compute the objective function
	myExpectation             = myDictDetails["Expectation"]
	myExpectationGradientY    = myDictDetails["ExpectationGradientY"]
	myExpectationGradientBeta = myDictDetails["ExpectationGradientBeta"]

	myExpectationGreaterVaR             = myDictDetails["ExpectationGreaterVaR"]
	myExpectationGreaterVaRGradientY    = myDictDetails["ExpectationGreaterVaRGradientY"]
	myExpectationGreaterVaRGradientBeta = myDictDetails["ExpectationGreaterVaRGradientBeta"]

	myVaRGradient = myDictDetails["VaRGradient"]

	# computes the objective function value and its gradient
	myObjectiveValue             = (1- myRiskAversion)  * myExpectation             + myRiskAversion  * (aVaR + (1/myAlpha) * myExpectationGreaterVaR)
	myObjectiveValueGradientY    = (1- myRiskAversion) .* myExpectationGradientY    + myRiskAversion .* (1/myAlpha)        .* myExpectationGreaterVaRGradientY
	myObjectiveValueGradientBeta = (1- myRiskAversion) .* myExpectationGradientBeta + myRiskAversion .* (1/myAlpha)        .* myExpectationGreaterVaRGradientBeta
	myObjectiveValueVaRGradient  =                                                    myRiskAversion  * (1 + (1/myAlpha)    * myVaRGradient)

	return myObjectiveValue, myObjectiveValueGradientY, myObjectiveValueGradientBeta, myObjectiveValueVaRGradient
end

#####################################
## Resolution: subgradient descent ##
#####################################

function get_optimal_strategy_mean_CVaR_Optim_adaptative_number_of_samples(;
									   aTraderIndex::Int64,
									   aTraders::Array{Trader,1},
									   aStrategies::Array{Strategy,1},
									   aVaRValueEstimate::Float64,
									   aSimulationParameters::SimulationParameters,
									   aCurrentNSamples::Int64 = ourInitialNSamplesToEvaluateCVaR
									   )

	# gets the stack trace, useful to know if ones has to print or not the logs of this function
	myStacktrace = stacktrace()

	# initialises the variable that will contain the file path to the file in which the optimal strategy will be saved 
	myOutputFilePath = nothing

	# to have a faster computation, we might want to start to obtain the optimal strategy with a lower number of realisations in the estimate of CVaR and then increase this the number of samples
	# in that way we have already a good initial guess of the optimal strategy when a large number of samples is required and as a consequence we hope to speed up the computation
	myNSamples = get_CVaR_optimisation_n_samples(aSimulationParameters)
	myNextNSamples = -1

	if get_CVaR_optimisation_adaptive_number_of_samples(aSimulationParameters)

		# gets the number of samples to compute the optimal strategy in the mean-CVaR framework
		myNSamples = aCurrentNSamples

		# computes the number of samples to find to NE in next iteration
		myNextNSamples = min(10*aCurrentNSamples,get_CVaR_optimisation_n_samples(aSimulationParameters))
	end

	# gets the trader
	myTrader = aTraders[aTraderIndex]

	# gets the market details structure belief of the trader
	myMarketDetailsBelief = get_market_details_belief(myTrader)

	# number of trading periods
	myNTradingPeriods = MarketDetailsModule.get_n_trading_periods(myMarketDetailsBelief)

	# adapts `aSimulationParameters` to the number of samples
	mySimulationParameters = get_new_simulation_parameters(
							       aSimulationParameters,
							       aParametersCVaROptimisation=Dict("NSamples" => myNSamples)
							       )


	# indices of the variables we optimise on for the redistribution matrix
	myRedistributionMatrixIndices = StrategyModule.get_redistribution_matrix_variables_indices(myNTradingPeriods)

	# prints the number of samples
	println_logs(string("\n++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++"),mySimulationParameters,myStacktrace)
	println_logs(string("+++ Start of optimisation with a number of ",myNSamples, " samples for λ = ",get_risk_aversion(myTrader)),mySimulationParameters,myStacktrace)
	println_logs(string("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n"),mySimulationParameters,myStacktrace)
	sleep(0.1)

	# gets the output file path in which the optimal strategy will be stored
	myOutputFilePath = HelpFilesModule.get_default_output_file_path(
									aMarketDetails                   = get_market_details_belief(aTraders[aTraderIndex]),
									aTraderIndex                     = aTraderIndex,
									aTraders                         = aTraders,
									aSimulationParameters            = mySimulationParameters,
									aStrategies                      = aStrategies,
									aIncludePartialMarketDetailsHash = true, # since anyway one optimise accordingly to the market parameters belief of trader `aTraders[aTraderIndex]`
									aIncludeSimulationParametersHash = true,
									aIncludeTradersHash              = true,
									aIncludeStrategiesHash           = true, # TODO: before false but now it is used as the initial
									aSpecificFolder                  = "outputs/trading_plan/mean-CVaR/"
									)

	# creates the output folders tree if they do not already exist
	HelpFilesModule.create_relevant_folders!(myOutputFilePath)

	####################################################################
	# PART 1: checks if it has already be computed given the arguments #
	####################################################################

	# tells if the optimal strategy must be computed
	myHasToComputeOptimalStrategy = true

	if isfile(myOutputFilePath) && !(get_recompute_optimal_strategies(aSimulationParameters))
		myHasToComputeOptimalStrategy = false
		HelpFilesModule.info_logs(
					  string(
						 "\nMeanCVaRModule 101:\n",
						 "The optimal strategy given the parameters passed as argument has already been computed and will not be recomputed.\n",
						 "The file related is named:\n",
						 myOutputFilePath,
						 "\nPlease delete this file or add as argument `aRecomputeOptimalStrategies=true` in the structure `aSimulationParameters` to force the recomputation of the optimal strategy."
						 ),
					  aSimulationParameters,
					  myStacktrace
					  )
	end

	###################################################
	# PART 2: computes the optimal strategy if needed #
	###################################################

	# computes the optimal strategy has it has never been computed before with the arguments given
	if myHasToComputeOptimalStrategy

		# initial point from which the algorithm is starting
		myInitialTradingPlan                   = get_trading_plan(aStrategies[aTraderIndex])
		myInitialTradingPlanVariables          = myInitialTradingPlan[1:end-1]
		myInitialRedistributionMatrixVariables = get_redistribution_matrix(aStrategies[aTraderIndex])[myRedistributionMatrixIndices]
		myInitialVaRMultiplicativeFactor       = 1 # for scaling purposes
		myInitialPoint                         = vcat(myInitialTradingPlanVariables,myInitialRedistributionMatrixVariables,myInitialVaRMultiplicativeFactor)

		############################################# 
		# Part 2.1: solves the optimisation problem # 
		############################################# 


		# A function that returns the value of the objective and the gradient
		# What is not in G or F is common for both (more efficient)
		# Note: the last component of x is used to evaluate the VaR value (Rockafellar approach)
		function fg!(F,G,x)

			println_logs("===========================",aSimulationParameters,myStacktrace)

			myCurrentTradingPlan             = x[1:myNTradingPeriods-1]
			myCurrentVaRMultiplicativeFactor = x[end]
			myCurrentRedistributionMatrix    = x[myNTradingPeriods:end-1]

			myTradingPlan          = vcat(myCurrentTradingPlan, 1-sum(myCurrentTradingPlan)) # Optim does not allow constraints, so get rid of y_m by incorporting the constraint: sum(y_i) = 1.
			myVaR                  = aVaRValueEstimate * myCurrentVaRMultiplicativeFactor                       # scales such that u of Rockafellar lies also around an order of magnitude of 1.
			myRedistributionMatrix = zeros(myNTradingPeriods,myNTradingPeriods)                            
			myRedistributionMatrix[myRedistributionMatrixIndices] = myCurrentRedistributionMatrix

			# Optim does not allow constraints, so get rid of the decisions concerning the last trading period by incorporting the constraint: for all k sum(β_(k,i)) = 1.
			myRedistributionMatrix[1:end-1,end] = 1 .- sum(myRedistributionMatrix[1:end-1,1:end-1], dims=2)

			if get_optimise_trading_plan(aSimulationParameters)
				println_logs(string("Trading plan:\n\t", myTradingPlan),aSimulationParameters,myStacktrace)
			end
			if get_optimise_redistribution_matrix(aSimulationParameters)
				println_logs(string("Redistribution matrix:\n\t", myRedistributionMatrix),aSimulationParameters,myStacktrace)
			end
			println_logs(string("VaR prop: ", myCurrentVaRMultiplicativeFactor),aSimulationParameters,myStacktrace)
			println_logs(string("VaR: ", myVaR),aSimulationParameters,myStacktrace)

			myBestStrategy = get_strategy(
						      aNTradingPeriods      = myNTradingPeriods,
						      aTradingPlan          = myTradingPlan,
						      aRedistributionMatrix = myRedistributionMatrix
						      )

			myBestStrategies = copy(aStrategies)
			myBestStrategies[aTraderIndex] = myBestStrategy

			myObjectiveFunction, myObjectiveFunctionGradientY, myObjectiveFunctionGradientBeta, myObjectiveFunctionVaRGradient = get_objective_function_value_and_gradient(
																						       aTraderIndex          = aTraderIndex,
																						       aTraders              = aTraders,
																						       aStrategies           = myBestStrategies,
																						       aVaR                  = myVaR,
																						       aSimulationParameters = mySimulationParameters
																						       )

			# scales the gradient in terms of u
			myObjectiveFunctionVaRGradient = aVaRValueEstimate * myObjectiveFunctionVaRGradient

			println_logs(string("\nObjective value: ",  myObjectiveFunction), aSimulationParameters,myStacktrace)
			if get_optimise_trading_plan(aSimulationParameters)
				println_logs(string("Gradient w.r.t. y: ",  myObjectiveFunctionGradientY), aSimulationParameters,myStacktrace)
			end
			if get_optimise_redistribution_matrix(aSimulationParameters)
				println_logs(string("Gradient w.r.t. β: ",  myObjectiveFunctionGradientBeta), aSimulationParameters,myStacktrace)
			end
			println_logs(string("Gradient w.r.t. u: ",  myObjectiveFunctionVaRGradient), aSimulationParameters,myStacktrace)

			if G != nothing
				G[1:myNTradingPeriods-1]   = myObjectiveFunctionGradientY
				G[myNTradingPeriods:end-1] = myObjectiveFunctionGradientBeta
				G[end]                     = myObjectiveFunctionVaRGradient
			end

			if F != nothing
				return myObjectiveFunction
			end
		end

		# algorithm to use
		myAlgorithm = get_algorithm(aSimulationParameters)
		myOptimAlgorithm = nothing
		if myAlgorithm == "GradientDescent"
			myOptimAlgorithm = Optim.GradientDescent()
		elseif myAlgorithm == "ConjugateGradient"
			myOptimAlgorithm = Optim.ConjugateGradient()
		elseif myAlgorithm == "BFGS"
			myOptimAlgorithm = Optim.BFGS()
		elseif myAlgorithm == "LBFGS"
			myOptimAlgorithm = Optim.LBFGS()
		else
			@error(string(
				      "\nMeanCVaRModule 102:\n",
				      "Algorithm ",
				      myAlgorithm,
				      " is not recognised in the Mean-CVaR multitrader framework."
				      )
			       )
		end

		results = Optim.optimize(
					 Optim.only_fg!(fg!), 
					 myInitialPoint,  # the point where to start the search from
					 myOptimAlgorithm, # the algorithm to use to optimise
					 Optim.Options(
						       iterations = get_CVaR_optimisation_maximum_number_of_iterations(aSimulationParameters),
						       allow_f_increases = true,
						       x_tol = ourXTolerance, # absolute tolerance in changes of the input vector x, in infinity norm. Defaults to 0.0.
						       f_tol = ourFTolerance, # relative tolerance in changes of the objective value. Defaults to 0.0.
						       g_tol = ourGTolerance  # absolute tolerance in the gradient, in infinity norm. Defaults to 1e-8. For gradient free methods, this will control the main convergence tolerance, which is solver specific.
						       )
					 )

		# tells if the strategy obtained should be saved, i.e. in the case where the algorithm has successfully converged
		myOptimisationSuccessful = true

		# checks that the maximum number of iterations has not been reached
		if Optim.iterations(results) == get_CVaR_optimisation_maximum_number_of_iterations(aSimulationParameters) # checks if the maximum of iterations allowed has been reached
			myOptimisationSuccessful = false
			show(results)
			@error(
			       string(
				      "\nMeanCVaRModule 103:\n",
				      "The maximum number of iterations for the gradient descent has been reached.",
				      " The optimisation was not successful."
				      )
			       )
		elseif !(Optim.converged(results)) # checks if the algorithm has converged
			myOptimisationSuccessful = false
			show(results)
			@error(
			       string(
				      "\nMeanCVaRModule 104:\n",
				      "The optimisation (Mean-CVaR with Optim) was not successful.",
				      )
			       )
		end

		# return nothing if the optimisation was not successful
		if !(myOptimisationSuccessful)
			return StrategyModule.get_strategy(), -1.0, false, ""
		end

		# retrieves the results of the optimisation
		myMinimiser = Optim.minimizer(results)

		myMinimiserTradingPlan              = myMinimiser[1:myNTradingPeriods-1]
		myMinimiserRedistributionMatrix     = myMinimiser[myNTradingPeriods:end-1]
		myMinimiserVaRMultiplicativeFactor  = myMinimiser[end]

		# prints the details if needed
		sleep(0.1) # to be sure all the other prints are done before the summary
		println_logs(string("\n\n *** Result of the optimisation with a number of ",myNSamples, " samples for λ = ",get_risk_aversion(myTrader),":\n"),mySimulationParameters,myStacktrace)
		println_logs(string("Successful? ", myOptimisationSuccessful),mySimulationParameters,myStacktrace)
		println_logs(string("Number of iterations: ", Optim.iterations(results)),mySimulationParameters,myStacktrace)
		println_logs(string("Optimal objective function: ",Optim.minimum(results)),mySimulationParameters,myStacktrace)
		if get_optimise_trading_plan(aSimulationParameters)
			println_logs(string("Optimal trading plan variables: ",myMinimiserTradingPlan),mySimulationParameters,myStacktrace)
		end
		if get_optimise_redistribution_matrix(aSimulationParameters)
			println_logs(string("Optimal redistribution matrix variables: ",myMinimiserRedistributionMatrix),mySimulationParameters,myStacktrace)
		end

		myOptimalTradingPlan          = vcat(myMinimiserTradingPlan, 1-sum(myMinimiserTradingPlan))
		myOptimalVaR                  = aVaRValueEstimate * myMinimiserVaRMultiplicativeFactor
		myOptimalRedistributionMatrix = zeros(myNTradingPeriods,myNTradingPeriods)
		myOptimalRedistributionMatrix[myRedistributionMatrixIndices] = myMinimiserRedistributionMatrix
		myOptimalRedistributionMatrix[1:end-1,end] = 1 .- sum(myOptimalRedistributionMatrix[1:end-1,1:end-1], dims=2)

		################################################## 
		# Part 2.b: saves the output of the optimisation # 
		################################################## 

		# gets the optimal strategy and the optimal objective value
		myOptimalStrategy       = nothing
		myOptimalObjectiveValue = nothing
		if myOptimisationSuccessful # the optimisation was successful
			myOptimalStrategy = get_strategy(
							 aNTradingPeriods      = myNTradingPeriods,
							 aTradingPlan          = myOptimalTradingPlan,
							 aRedistributionMatrix = myOptimalRedistributionMatrix
							 )
			myOptimalObjectiveValue = Optim.minimum(results)
		else # the optimisation was NOT successful
			myOptimalStrategy       = StrategyModule.get_strategy()
			myOptimalObjectiveValue = -1.0
		end

		# saves the optimal execution strategy in the file `myOutputFilePath`
		myDetailsToSave = Dict()
		myDetailsToSave["Method"]                 = get_method(mySimulationParameters)
		myDetailsToSave["Trader"]                 = get_dict_from_trader(aTraders[aTraderIndex])
		myDetailsToSave["SimulationParameters"]   = get_dict_from_simulation_parameters(mySimulationParameters)
		myDetailsToSave["TradingPlan"]            = get_trading_plan(myOptimalStrategy)
		myDetailsToSave["RedistributionMatrix"]   = get_redistribution_matrix(myOptimalStrategy)
		myDetailsToSave["ObjectiveValue"]         = myOptimalObjectiveValue
		myDetailsToSave["OptimisationSuccessful"] = myOptimisationSuccessful
		myDetailsToSave["VaR"]                    = myOptimalVaR
		save_result!(myOutputFilePath,myDetailsToSave)
	end

	# loads the optimal strategy from the file `myOutputFilePath`
	myOptimalStrategy = load_strategy_from_file(myOutputFilePath)

	myDict = load_result(myOutputFilePath)
	myOptimalObjectiveValue  = myDict["ObjectiveValue"]
	myOptimisationSuccessful = myDict["OptimisationSuccessful"]
	myVaRValue               = myDict["VaR"]

	# update the optimal strategy of the trader we are interested in (will be the initial strategy for the next iteration where more samples are generated to evaluate CVaR)
	myStrategies = deepcopy(aStrategies)
	myStrategies[aTraderIndex] = myOptimalStrategy

	# if the current number of samples to compute the NE is not equal to the one in the initial one in the simulation parameter structure
	if myNSamples < get_CVaR_optimisation_n_samples(aSimulationParameters)

		return get_optimal_strategy_mean_CVaR_Optim_adaptative_number_of_samples(
											 aTraderIndex          = aTraderIndex,
											 aTraders              = aTraders,
											 aStrategies           = myStrategies,
											 aVaRValueEstimate     = myVaRValue,
											 aSimulationParameters = aSimulationParameters,
											 aCurrentNSamples      = myNextNSamples
											 )
	end

	return myOptimalStrategy, myOptimalObjectiveValue, myOptimisationSuccessful, myOutputFilePath
end

"""
```
get_optimal_strategy_mean_CVaR_Optim(;
aStrategies::Array{Strategy,1},
aTraderIndex::Int64,
aTraders::Array{Trader,1},
aSimulationParameters::SimulationParameters
)
```

computes the optimal strategy when the position of the other traders are taken into account.

### Arguments
* `aStrategies::Array{Strategy,1}`: TODO.
* `aTraderIndex::Int64`: TODO.
* `aTraders::Array{Trader,1}`: TODO.
* `aSimulationParameters::SimulationParameters`: TODO.
"""
function get_optimal_strategy_mean_CVaR_Optim(;
					      aTraderIndex::Int64,
					      aTraders::Array{Trader,1},
					      aStrategies::Array{Strategy,1},
					      aSimulationParameters::SimulationParameters
					      )

	##################################################
	### Initial point for the optimisation problem ###
	##################################################

	# To avoid have variables with different order of magnitude,
	# instead of using VaR, we use x[end] * Optimal_Value_AC,
	# so that x[end] is of scale around 1.

	# gets the trader
	myTrader = aTraders[aTraderIndex]

	# gets the market details structure belief of the trader
	myMarketDetailsBelief = get_market_details_belief(myTrader)

	# Initialisation of the VaR value to use in Rockafellar approach, ie. u.
	# We initialises u with the objective function of Almgren and Chriss model when the trader is risk neutral.
	# To avoid have variables with different order of magnitude instead of using VaR, we use x[end] * Optimal_Value_AC such that x[end] is of scale around 1 like the other varibles.

	myRiskNeutralTraders                     = deepcopy(aTraders)
	myRiskNeutralTradersUncertaintyStructure = deepcopy(get_traders_uncertainty_structure(myMarketDetailsBelief))

	myRiskNeutralTradersUncertaintyStructure[aTraderIndex] = get_new_uncertainty_structure(
											       myRiskNeutralTradersUncertaintyStructure[aTraderIndex],
											       aConsiderForecastUpdates=false
											       )

	myRiskNeutralTrader = get_new_trader(
					     myTrader,
					     aRiskAversion = 0.0,
					     aMarketDetailsBelief = get_new_market_details(
											   myMarketDetailsBelief,
											   aTradersUncertaintyStructure = myRiskNeutralTradersUncertaintyStructure 
											   )
					     )

	myRiskNeutralTraders[aTraderIndex] = myRiskNeutralTrader

	myMeanVarianceRiskNeutralOptimalStrategy, myMeanVarianceRiskNeutralObjValue, myMeanVarianceOptimisationSuccessful, myMeanVarianceOutputFilePath = get_optimal_strategy_mean_variance(
																							     aTraderIndex          = aTraderIndex,
																							     aTraders              = myRiskNeutralTraders,
																							     aStrategies           = aStrategies,
																							     aSimulationParameters = get_new_simulation_parameters(
																														   aSimulationParameters,
																														   aMethod = "Mean-Variance",
																														   aUncertaintyGenerationMethod="normal",
																														   aOptimiseRedistributionMatrix=false
																														   )
																							     )



	# guess of the VaR, for scaling purposes we do not consider VaR as a variable as in Rockafellor approach but consider another variable x multiplying our guess, i.e. x*myInitialVaRGuess
	# the idea being that all the variables are of the same order of magnitude, i.e. around 1.
	# the guess will be updated when the number of samples to evaluate CVaR is adaptative
	myInitialVaRGuess = myMeanVarianceRiskNeutralObjValue

	#######################################################################################################################
	### Solve the optimisation problem: this is done by increasing slowly the number of samples for efficiency purposes ###
	#######################################################################################################################

	return get_optimal_strategy_mean_CVaR_Optim_adaptative_number_of_samples(
										 aTraderIndex          = aTraderIndex,
										 aTraders              = aTraders,
										 aStrategies           = aStrategies,
										 aVaRValueEstimate     = myInitialVaRGuess,
										 aSimulationParameters = aSimulationParameters
										 )

end

"""
```
fill_matrix_Lω!(;
aNTradingPeriods,
aInitialDemand,
aForecastUpdatesRealisation,
aβ,
aLω
)
```

fills the matrix `aLω` given as argument, in such a way that the trading quantities Qω in case of the realisation ω (here given by `aForecastUpdatesRealisation`) can be computed as follows:
Qω = Lω * y,
where y is the trading strategy (the target proportion of the final volume to trade during each trading period).

### Arguments
* `aNTradingPeriods`: the number of trading period.
* `aInitialDemand`: the initial volume target at time t0.
* `aForecastUpdatesRealisation`: a vector whose component [i] is the the forecast update during the trading period τi in the case of realisation ω.
* `aβ`: the redistribution matrix to consider.
* `aLω`: the matrix to fill.
"""
function fill_matrix_Lω!(;
			 aNTradingPeriods,
			 aInitialDemand,
			 aForecastUpdatesRealisation,
			 aβ,
			 aLω
			 )

	for i in 1:aNTradingPeriods, j in 1:aNTradingPeriods
		if i == j
			aLω[i,j] = aInitialDemand + reduce(+, [aForecastUpdatesRealisation[k] for k = 1:i-1], init = 0.0)
		elseif i > j
			aLω[i,j] = reduce(+, [aForecastUpdatesRealisation[k]*aβ[k,i] for k = j:i-1], init = 0.0)
		end
	end
end

"""
```
get_trading_cost_quadratic_form_ω(;
aNTradingPeriods,
aPricesMovesRealisationω,
aInitialDemand,
aForecastUpdatesRealisationω,
aτ,
aγ,
aϵ,
aη,
aQOthersPosω,
aQOthersNegω,
aCoeffPosω,
aCoeffNegω
)
```

Given a realisation ω ∈ Ω (given by `aPricesMovesRealisationω` and `aForecastUpdatesRealisationω`),
returns the vector Bω and the constant cω of the quadratic form of the trading cost in terms of Qω, i.e. Qω' * A * Qω + Qω' * Bω + cω.
The matrix A does not depend on ω but can be computed as follows:

```
# get the matrices for the quadratic factor
myGb_eq, myGb_neq, myEb, myEb_m, myΓb, myΛb = NEHelpFunctionsModule.get_matrices_for_risk_set(aγ = aγ, aη = aη, aτ = aτ)

A = zeros(aNTradingPeriods,aNTradingPeriods)

# adds the matrix Me
A[1:end-1,1:end-1] -= 0.5 * myΓb

# adds the matrix Mf
A[1:end-1,1:end-1] += 0.5 * ( 2*myEb + 2*myEb_m )
```

### Arguments
* `aNTradingPeriods`: the number of trading period.
* `aPricesMovesRealisationω`:  a vector whose component [i] is the the price move during the trading period τi in the case of realisation ω.
* `aInitialDemand`: the initial volume target at time t0.
* `aForecastUpdatesRealisation`: a vector whose component [i] is the the forecast update during the trading period τi in the case of realisation ω.
* `aτ`: a vector whose component [i] is the length of the i-th trading period, i.e. τi.
* `aγ`: a vector whose component [i] is the permanent impact parameter γi of trading period τi.
* `aϵ`: a vector whose component [i] is the temporary impact parameter ϵi of trading period τi.
* `aη`: a vector whose component [i] is the temporary impact parameter ηi of trading period τi.
* `aQOthersPosω`: a vector whose component [i] is the number of long positions traded by the other players during trading period τi.
* `aQOthersNegω`: a vector whose component [i] is the number of short positions traded by the other players during trading period τi.
* `aCoeffPosω`: a vector whose component [i] equals 1, if the trader takes long positions on trading period τi, and 0 otherwise.
* `aCoeffNegω`: a vector whose component [i] equals 1, if the trader takes short positions on trading period τi, and 0 otherwise.
"""
function get_trading_cost_quadratic_form_ω(;
					   aNTradingPeriods,
					   aPricesMovesRealisationω,
					   aInitialDemand,
					   aForecastUpdatesRealisationω,
					   aτ,
					   aγ,
					   aϵ,
					   aη,
					   aQOthersPosω,
					   aQOthersNegω,
					   aCoeffPosω,
					   aCoeffNegω
					   )

	# gets the final volume
	myFinalDemandω = aInitialDemand + sum(aForecastUpdatesRealisationω)

	###########################################################################
	# Get the constant cω of the quadratic form: Qω' * A * Qω + Qω' * Bω + cω #
	###########################################################################

	# adds the constant K
	cω = 0
	for i in 1:aNTradingPeriods-1
		cω += ( aτ[i]^(0.5) * aPricesMovesRealisationω[i] + aγ[i] * ( aQOthersPosω[i] - aQOthersNegω[i] ) ) * myFinalDemandω
	end

	# adds the constant Cf
	cω += ( aη[end] / aτ[end] ) * myFinalDemandω^2

	######################################################################
	# Get the vector Bω of the quadratic form: Q' * A * Q + Q' * Bω + cω #
	######################################################################

	# initiliases the vector
	Bω = zeros(aNTradingPeriods)

	# adds the vector Za
	for j = 1:aNTradingPeriods-1
		Bω[j] -= sum( aτ[i]^(0.5) * aPricesMovesRealisationω[i] + aγ[i] * ( aQOthersPosω[i] - aQOthersNegω[i] ) for i in j:aNTradingPeriods-1)
	end

	# adds the vector Zb
	for j = 1:aNTradingPeriods-1
		Bω[j] += aγ[j] * myFinalDemandω
	end

	# adds the vector Zc
	for j = 1:aNTradingPeriods
		Bω[j] += aϵ[j] * ( aCoeffPosω[j] - aCoeffNegω[j] )
	end

	# adds the vector Zd
	for j = 1:aNTradingPeriods
		Bω[j] += ( aη[j] / aτ[j] ) * ( aCoeffPosω[j] * aQOthersPosω[j] - aCoeffNegω[j] * aQOthersNegω[j] )
	end

	# adds the vector Zf
	for j = 1:aNTradingPeriods-1
		Bω[j] -= 2 * ( aη[end] / aτ[end] ) * myFinalDemandω
	end

	#= ###################################################################### =#
	#= # Get the vector A of the quadratic form: Q' * A * Q + Q' * Bω + cω # =#
	#= ###################################################################### =#

	#= # get the matrices for the quadratic factor =#
	#= myGb_eq, myGb_neq, myEb, myEb_m, myΓb, myΛb = NEHelpFunctionsModule.get_matrices_for_risk_set(aγ = aγ, aη = aη, aτ = aτ) =#

	#= A = zeros(aNTradingPeriods,aNTradingPeriods) =#

	#= # adds the matrix Me =#
	#= A[1:end-1,1:end-1] -= 0.5 * myΓb =#

	#= # adds the matrix Mf =#
	#= A[1:end-1,1:end-1] += 0.5 * ( 2*myEb + 2*myEb_m ) =#

	# returns the vector and the constant
	return Bω, cω
end

"""
```
get_trading_cost_ω(;
aNTradingPeriods,
aY,
aβ,
aPricesMovesRealisationω,
aInitialDemand,
aForecastUpdatesRealisationω,
aτ,
aγ,
aϵ,
aη,
aQOthersPosω,
aQOthersNegω,
aCoeffPosω,
aCoeffNegω
)
```

returns the trading cost in the case of the realisation ω (given by `aPricesMovesRealisationω` and `aForecastUpdatesRealisationω`) when following the strategy given by the trading plan `aY` and the redistribution matrix `aβ`.

### Arguments
* `aNTradingPeriods`: the number of trading period.
* `aY`: a trading plan (in terms of proportions).
* `aβ`: a redistribution matrix.
* `aPricesMovesRealisationω`:  a vector whose component [i] is the the price move during the trading period τi in the case of realisation ω.
* `aInitialDemand`: the initial volume target at time t0.
* `aForecastUpdatesRealisation`: a vector whose component [i] is the the forecast update during the trading period τi in the case of realisation ω.
* `aτ`: a vector whose component [i] is the length of the i-th trading period, i.e. τi.
* `aγ`: a vector whose component [i] is the permanent impact parameter γi of trading period τi.
* `aϵ`: a vector whose component [i] is the temporary impact parameter ϵi of trading period τi.
* `aη`: a vector whose component [i] is the temporary impact parameter ηi of trading period τi.
* `aQOthersPosω`: a vector whose component [i] is the number of long positions traded by the other players during trading period τi.
* `aQOthersNegω`: a vector whose component [i] is the number of short positions traded by the other players during trading period τi.
* `aCoeffPosω`: a vector whose component [i] equals 1, if the trader takes long positions on trading period τi, and 0 otherwise.
* `aCoeffNegω`: a vector whose component [i] equals 1, if the trader takes short positions on trading period τi, and 0 otherwise.
"""
function get_trading_cost_ω(;
			    aNTradingPeriods,
			    aY,
			    aβ,
			    aPricesMovesRealisationω,
			    aInitialDemand,
			    aForecastUpdatesRealisationω,
			    aτ,
			    aγ,
			    aϵ,
			    aη,
			    aQOthersPosω,
			    aQOthersNegω,
			    aCoeffPosω,
			    aCoeffNegω
			    )

	########################################################################################################################
	# Gets the matrix A, vector Bω, and constant cω of the trading cost in terms of Qω, i.e. Qω' * A * Qω + Qω' * Bω + cω. #
	########################################################################################################################

	# get the matrices needed to compute the matrix A of the quadratic form
	myGb_eq, myGb_neq, myEb, myEb_m, myΓb, myΛb = NEHelpFunctionsModule.get_matrices_for_risk_set(aγ = aγ, aη = aη, aτ = aτ)

	# computes the matrix A of the quadratic form
	myAω = zeros(aNTradingPeriods,aNTradingPeriods)
	myAω[1:end-1,1:end-1] = 0.5 * ( 2*myEb + 2*myEb_m - myΓb)

	# gets the vector Bω and constant cω of the quadratic form
	myBω, mycω = get_trading_cost_quadratic_form_ω(
						       aNTradingPeriods             = aNTradingPeriods,
						       aPricesMovesRealisationω      = aPricesMovesRealisationω,
						       aInitialDemand               = aInitialDemand,
						       aForecastUpdatesRealisationω = aForecastUpdatesRealisationω,
						       aτ                           = aτ,
						       aγ                           = aγ,
						       aϵ                           = aϵ,
						       aη                           = aη,
						       aQOthersPosω                 = aQOthersPosω,
						       aQOthersNegω                 = aQOthersNegω,
						       aCoeffPosω                   = aCoeffPosω,
						       aCoeffNegω                   = aCoeffNegω
						       )

	############################################
	# Gets the matrix Lω such that Qω = Lω * y #
	############################################

	myLω = zeros(aNTradingPeriods,aNTradingPeriods)
	fill_matrix_Lω!(
			aNTradingPeriods            = aNTradingPeriods,
			aInitialDemand              = aInitialDemand,
			aForecastUpdatesRealisation = aForecastUpdatesRealisationω,
			aβ                          = aβ,
			aLω                         = myLω
			)

	##############################################################
	# Computes the cost in case of ω based on the quadratic form #
	##############################################################

	myCostω = ( myLω * aY )' * myAω * ( myLω * aY ) + myBω' * ( myLω * aY ) + mycω

	return myCostω
end

"""
```
get_trading_cost_quadratic_form_Σω(;
aNTradingPeriods,
aNSamplesCVaR,
aPricesMovesRealisations,
aInitialDemand,
aForecastUpdatesRealisations,
aτ,
aγ,
aϵ,
aη,
aQOthersPos,
aQOthersNeg,
aCoeffPos,
aCoeffNeg,
aβ,
aμ
)
```

returns the matrix A_Σω, vector B_Σω, and constant c_Σω such that, given the trading plan y, to estimate the expectation of the trading cost given the probabilily `aμ` and the coefficients `aCoeffPos` and `aCoeffNeg`,
i.e. Cost(y,μ,CoeffPos,CoeffNeg) = y' * A_Σω * y + B_Σω' * y + c_Σω.

### Arguments
* `aNTradingPeriods`: the number of trading period.
* `aNSamplesCVaR`: the number of samples generated to estimate the expectation and CVaR with Monte-Carlo.
* `aPricesMovesRealisations`: a matrix whose component [i,ω] is the the price move during the trading period τi in the case of realisation ω.
* `aInitialDemand`: the initial volume target at time t0.
* `aForecastUpdatesRealisations`: a matrix whose component [i,ω] is the the forecast update during the trading period τi in the case of realisation ω.
* `aτ`: a vector whose component [i] is the length of the i-th trading period, i.e. τi.
* `aγ`: a vector whose component [i] is the permanent impact parameter γi of trading period τi.
* `aϵ`: a vector whose component [i] is the temporary impact parameter ϵi of trading period τi.
* `aη`: a vector whose component [i] is the temporary impact parameter ηi of trading period τi.
* `aQOthersPos`: a vector whose component [i,ω] is the number of long positions traded by the other players during trading period τi in case of realisation ω.
* `aQOthersNeg`: a vector whose component [i,ω] is the number of short positions traded by the other players during trading period τi in case of realisation ω.
* `aCoeffPos`: a vector whose component [i,ω] equals 1, if the trader takes long positions on trading period τi in case of realisation ω, and 0 otherwise.
* `aCoeffNeg`: a vector whose component [i,ω] equals 1, if the trader takes short positions on trading period τi in case of realisation ω, and 0 otherwise.
* `aβ`: a redistribution matrix.
* `aμ`: an vector whose component [ω] is the weight attributed to the realisation ω.
"""
function get_trading_cost_quadratic_form_Σω(;
					    aNTradingPeriods,
					    aNSamplesCVaR,
					    aPricesMovesRealisations,
					    aInitialDemand,
					    aForecastUpdatesRealisations,
					    aτ,
					    aγ,
					    aϵ,
					    aη,
					    aQOthersPos,
					    aQOthersNeg,
					    aCoeffPos,
					    aCoeffNeg,
					    aβ,
					    aμ
					    )

	# initialises the variables used in the quadratic form: Cost(y,μ,CoeffPos,CoeffNeg) = y' * A_Σω * y + B_Σω' * y + c_Σω, where A_Σω, B_Σω and c_Σω depend on the triplet (μ,CoeffPos,CoeffNeg)
	A_Σω = zeros(aNTradingPeriods,aNTradingPeriods)
	B_Σω = zeros(aNTradingPeriods)
	c_Σω = 0.0

	##################################################
	# Initialises variables used in each iteration ω #
	##################################################

	# get the matrices for the quadratic factor
	myGb_eq, myGb_neq, myEb, myEb_m, myΓb, myΛb = NEHelpFunctionsModule.get_matrices_for_risk_set(aγ = aγ, aη = aη, aτ = aτ)

	myLocalAω = zeros(aNTradingPeriods,aNTradingPeriods)
	myLocalAω[1:end-1,1:end-1] = 0.5 * ( 2*myEb + 2*myEb_m - myΓb)

	myLocalLω = zeros(aNTradingPeriods,aNTradingPeriods)
	myLocalBω = zeros(aNTradingPeriods)
	myLocalcω = 0

	###############################################################
	# Builds the quadratic form: y' * A_Σω * y + B_Σω' * y + c_Σω #
	###############################################################

	for ω in 1:aNSamplesCVaR

		# gets the matrix Lω such that Qω = Lω * y
		fill_matrix_Lω!(;
				aNTradingPeriods            = aNTradingPeriods,
				aInitialDemand              = aInitialDemand,
				aForecastUpdatesRealisation = aForecastUpdatesRealisations[:,ω],
				aβ                          = aβ,
				aLω                         = myLocalLω
				)

		# get the vector Bω and cω of the quadratic form of the trading cost:
		# Qω' * A * Qω + Bω' * Qω  + cω ⇔ ( Lω * y ) ' * A * ( Lω * y ) + Bω' * ( Lω * y )  + cω,
		# related to the coefficient aCoeffPos[ω] and aCoeffNeg[ω].
		myLocalBω, myLocalcω = get_trading_cost_quadratic_form_ω(
									 aNTradingPeriods             = aNTradingPeriods,
									 aPricesMovesRealisationω      = aPricesMovesRealisations[:,ω],
									 aInitialDemand               = aInitialDemand,
									 aForecastUpdatesRealisationω = aForecastUpdatesRealisations[:,ω],
									 aτ                           = aτ,
									 aγ                           = aγ,
									 aϵ                           = aϵ,
									 aη                           = aη,
									 aQOthersPosω                 = aQOthersPos[:,ω],
									 aQOthersNegω                 = aQOthersNeg[:,ω],
									 aCoeffPosω                   = aCoeffPos[:,ω],
									 aCoeffNegω                   = aCoeffNeg[:,ω]
									 )

		# adds this to the global expression scaled by aμ
		A_Σω += aμ[ω] .* ( myLocalLω' * myLocalAω * myLocalLω )
		B_Σω += aμ[ω] .* ( myLocalLω' * myLocalBω ) # Note: the desired form is y' * A_Σω * y + B_Σω' * y + c_Σω, so B_Σω' * y would then lead to ( ∑_ω ( myLocalLω' * myLocalBω ) )' * y = ∑_ω myLocalBω' * ( myLocalLω * y )
		c_Σω += aμ[ω]  * myLocalcω
	end

	# returns the vector and the constant
	return A_Σω, B_Σω, c_Σω
end

"""
```
set_objective_model_next_extreme_vertex!(;
aNTradingPeriods,
aNSamplesCVaR,
aModel,
aμ,
aY,
aβ,
aPricesMovesRealisations,
aInitialDemand,
aForecastUpdatesRealisations,
aτ,
aγ,
aϵ,
aη,
aQOthersPos,
aQOthersNeg,
aCoeffPos,
aCoeffNeg
)
```

updates the objective value of the optimisation model `aModel` which aims to find the extreme probability measure μ such that it leads to the maximum expectation E^μ[C] trading cost given μ for the trading plan `aY` and the redistribution matrix `aβ`.

### Arguments
* `aNTradingPeriods`: the number of trading period.
* `aNSamplesCVaR`: the number of samples generated to estimate the expectation and CVaR with Monte-Carlo.
* `aModel`: an optimisation model.
* `aμ`: an vector whose component [ω] is the weight attributed to the realisation ω.
* `aY`: a trading plan (in terms of proportions).
* `aβ`: a redistribution matrix.
* `aPricesMovesRealisations`: a matrix whose component [i,ω] is the the price move during the trading period τi in the case of realisation ω.
* `aInitialDemand`: the initial volume target at time t0.
* `aForecastUpdatesRealisations`: a matrix whose component [i,ω] is the the forecast update during the trading period τi in the case of realisation ω.
* `aτ`: a vector whose component [i] is the length of the i-th trading period, i.e. τi.
* `aγ`: a vector whose component [i] is the permanent impact parameter γi of trading period τi.
* `aϵ`: a vector whose component [i] is the temporary impact parameter ϵi of trading period τi.
* `aη`: a vector whose component [i] is the temporary impact parameter ηi of trading period τi.
* `aQOthersPos`: a vector whose component [i,ω] is the number of long positions traded by the other players during trading period τi in case of realisation ω.
* `aQOthersNeg`: a vector whose component [i,ω] is the number of short positions traded by the other players during trading period τi in case of realisation ω.
* `aCoeffPos`: a vector whose component [i,ω] equals 1, if the trader takes long positions on trading period τi in case of realisation ω, and 0 otherwise.
* `aCoeffNeg`: a vector whose component [i,ω] equals 1, if the trader takes short positions on trading period τi in case of realisation ω, and 0 otherwise.
"""
function set_objective_model_next_extreme_vertex!(;
						  aNTradingPeriods,
						  aNSamplesCVaR,
						  aModel,
						  aμ,
						  aY,
						  aβ,
						  aPricesMovesRealisations,
						  aInitialDemand,
						  aForecastUpdatesRealisations,
						  aτ,
						  aγ,
						  aϵ,
						  aη,
						  aQOthersPos,
						  aQOthersNeg,
						  aCoeffPos,
						  aCoeffNeg
						  )


	set_objective_function(aModel,
			       sum(
				   aμ[ω] * get_trading_cost_ω(
							      aNTradingPeriods             = aNTradingPeriods,
							      aY                           = aY,
							      aβ                           = aβ,
							      aPricesMovesRealisationω     = aPricesMovesRealisations[:,ω],
							      aInitialDemand               = aInitialDemand,
							      aForecastUpdatesRealisationω = aForecastUpdatesRealisations[:,ω],
							      aτ                           = aτ,
							      aγ                           = aγ,
							      aϵ                           = aϵ,
							      aη                           = aη,
							      aQOthersPosω                 = aQOthersPos[:,ω],
							      aQOthersNegω                 = aQOthersNeg[:,ω],
							      aCoeffPosω                   = aCoeffPos[:,ω],
							      aCoeffNegω                   = aCoeffNeg[:,ω]
							      )
				   for ω = 1:aNSamplesCVaR
				   )
			       )
end

"""
```
add_constraint_model_relaxation!(;
aNTradingPeriods,
aNSamplesCVaR,
aModel,
aμ,
aY,
aβ,
aT,
aPricesMovesRealisations,
aInitialDemand,
aForecastUpdatesRealisations,
aτ,
aγ,
aϵ,
aη,
aQOthersPos,
aQOthersNeg,
aCoeffPos,
aCoeffNeg,
)
```

### Arguments
* `aNTradingPeriods`: the number of trading period.
* `aNSamplesCVaR`: the number of samples generated to estimate the expectation and CVaR with Monte-Carlo.
* `aModel`: an optimisation model.
* `aμ`: an vector whose component [ω] is the weight attributed to the realisation ω.
* `aY`: a trading plan (in terms of proportions).
* `aβ`: a redistribution matrix.
* `aPricesMovesRealisations`: a matrix whose component [i,ω] is the the price move during the trading period τi in the case of realisation ω.
* `aInitialDemand`: the initial volume target at time t0.
* `aForecastUpdatesRealisations`: a matrix whose component [i,ω] is the the forecast update during the trading period τi in the case of realisation ω.
* `aτ`: a vector whose component [i] is the length of the i-th trading period, i.e. τi.
* `aγ`: a vector whose component [i] is the permanent impact parameter γi of trading period τi.
* `aϵ`: a vector whose component [i] is the temporary impact parameter ϵi of trading period τi.
* `aη`: a vector whose component [i] is the temporary impact parameter ηi of trading period τi.
* `aQOthersPos`: a vector whose component [i,ω] is the number of long positions traded by the other players during trading period τi in case of realisation ω.
* `aQOthersNeg`: a vector whose component [i,ω] is the number of short positions traded by the other players during trading period τi in case of realisation ω.
* `aCoeffPos`: a vector whose component [i,ω] equals 1, if the trader takes long positions on trading period τi in case of realisation ω, and 0 otherwise.
* `aCoeffNeg`: a vector whose component [i,ω] equals 1, if the trader takes short positions on trading period τi in case of realisation ω, and 0 otherwise.
"""
function add_constraint_model_relaxation!(;
					  aNTradingPeriods,
					  aNSamplesCVaR,
					  aModel,
					  aμ,
					  aY,
					  aβ,
					  aT,
					  aPricesMovesRealisations,
					  aInitialDemand,
					  aForecastUpdatesRealisations,
					  aτ,
					  aγ,
					  aϵ,
					  aη,
					  aQOthersPos,
					  aQOthersNeg,
					  aCoeffPos,
					  aCoeffNeg,
					  )

	myA_Σω, myB_Σω, myc_Σω = get_trading_cost_quadratic_form_Σω(
								    aNTradingPeriods             = aNTradingPeriods,
								    aNSamplesCVaR                = aNSamplesCVaR,
								    aPricesMovesRealisations     = aPricesMovesRealisations,
								    aInitialDemand               = aInitialDemand,
								    aForecastUpdatesRealisations = aForecastUpdatesRealisations,
								    aτ                           = aτ,
								    aγ                           = aγ,
								    aϵ                           = aϵ,
								    aη                           = aη,
								    aQOthersPos                  = aQOthersPos,
								    aQOthersNeg                  = aQOthersNeg,
								    aCoeffPos                    = aCoeffPos,
								    aCoeffNeg                    = aCoeffNeg,
								    aβ                           = aβ,
								    aμ                           = aμ
								    )

	@constraint(aModel, aY' * myA_Σω * aY + myB_Σω' * aY + myc_Σω ≤ aT )
end

"""
```
get_relaxation_model_without_coefficient_constraints(;
aNTradingPeriods,
aOptimiseTradingPlan,
aPlayerTradingPlan
)
```

returns the Gurobi model that is the relaxation model..

### Arguments
* `Argument1`: TODO.
* `Argument2`: TODO.
* `...`: TODO.
"""
function get_relaxation_model_without_coefficient_constraints(;
							      aNTradingPeriods,
							      aOptimiseTradingPlan,
							      aPlayerTradingPlan
							      )

	# gets the stack trace, useful to know if ones has to print or not the logs of this function
	myStacktrace = stacktrace()

	# initialises the model that solves the relaxation (in the sense does not include all extreme points)
	myModelRelaxation = Model(optimizer_with_attributes(() -> Gurobi.Optimizer(ourGurobiEnvironment)))
	JuMP.set_optimizer_attribute(myModelRelaxation, "OutputFlag", 0)

	# initialises the variables as in the lifted optimisation problem
	@variable(myModelRelaxation, t)
	@variable(myModelRelaxation, -2 <= y[1:aNTradingPeriods] <= 2 ) # the bounds -2 and 2 are just to avoid unbounded solutions

	# adds the objective to the relaxation model
	@objective(myModelRelaxation, Min, t)

	# contraints related to the variables to optimise
	if !(aOptimiseTradingPlan)
		@constraint(myModelRelaxation, y .== aPlayerTradingPlan)
		println_logs(string("Do not optimise the trading plan."),aSimulationParameters,myStacktrace)
	end

	# adds the constraint the the sum of the proportions must add to 1
	@constraint(myModelRelaxation, sum(y) == 1)

	# the other constraints will be added dynamically

	return myModelRelaxation
end

function update_indices_with_postive_and_negative_volumes!(;
							   aCoeffPosToChange,
							   aCoeffNegToChange,
							   aInitialDemand,
							   aForecastUpdatesRealisations,
							   aY,
							   aβ
							   )

	# gets the dimensions of the vector `aCoeffPosToChange`
	myNTradingPeriods = size(aCoeffPosToChange,1)
	myNSamplesCVaR    = size(aCoeffPosToChange,2)

	# initialises the variables containing Lω and Qω
	myLω = zeros(myNTradingPeriods,myNTradingPeriods)
	myQω = zeros(myNTradingPeriods,myNTradingPeriods)

	for ω in 1:myNSamplesCVaR
		fill_matrix_Lω!(
				aNTradingPeriods            = myNTradingPeriods,
				aInitialDemand              = aInitialDemand,
				aForecastUpdatesRealisation = aForecastUpdatesRealisations[:,ω],
				aβ                          = aβ,
				aLω                         = myLω
				)

		# computes Qω
		myQω = myLω * aY

		# updates the value of `aCoeffPosToChange` and `aCoeffNegToChange`
		for i in 1:myNTradingPeriods
			if myQω[i] < 0
				aCoeffPosToChange[i,ω] = 0
				aCoeffNegToChange[i,ω] = 1
			else
				aCoeffPosToChange[i,ω] = 1
				aCoeffNegToChange[i,ω] = 0
			end
		end
	end
end

function get_optimal_strategy_mean_CVaR_Gurobi(;
					       aTraderIndex::Int64,
					       aTraders::Array{Trader,1},
					       aStrategies::Array{Strategy,1},
					       aSimulationParameters::SimulationParameters,
					       aSeed::Int = -1
					       )

	# gets the stack trace, useful to know if ones has to print or not the logs of this function
	myStacktrace = stacktrace()

	# gets the trader we want to optimise
	myTrader       = aTraders[aTraderIndex]
	myAlpha        = get_alpha(myTrader)
	myRiskAversion = get_risk_aversion(myTrader)

	# gets the market details belief of the trader
	myMarketDetailsBelief = get_market_details_belief(myTrader)
	myNTradingPeriods     = MarketDetailsModule.get_n_trading_periods(myMarketDetailsBelief)
	myNTraders            = MarketDetailsModule.get_n_traders(myMarketDetailsBelief)

	# gets the trading plan of all the players
	myPlayersTradingPlan = [get_trading_plan(aStrategies[myLocalPlayer]) for myLocalPlayer in 1:myNTraders]
	myPlayersRedMatrix   = [get_redistribution_matrix(aStrategies[myLocalPlayer]) for myLocalPlayer in 1:myNTraders]

	# gets the output file path in which the optimal strategy will be stored
	myOutputFilePath = HelpFilesModule.get_default_output_file_path(
									aMarketDetails                   = get_market_details_belief(aTraders[aTraderIndex]),
									aTraderIndex                     = aTraderIndex,
									aTraders                         = aTraders,
									aSimulationParameters            = aSimulationParameters,
									aStrategies                      = aStrategies,
									aIncludePartialMarketDetailsHash = true, # since anyway one optimise accordingly to the market parameters belief of trader `aTraders[aTraderIndex]`
									aIncludeSimulationParametersHash = true,
									aIncludeTradersHash              = true,
									aIncludeStrategiesHash           = true,
									aSpecificFolder                  = "outputs/trading_plan/mean-CVaR/"
									)

	# creates the output folders tree if they do not already exist
	HelpFilesModule.create_relevant_folders!(myOutputFilePath)

	####################################################################
	# PART 1: checks if it has already be computed given the arguments #
	####################################################################

	# tells if the optimal strategy must be computed
	myHasToComputeOptimalStrategy = true

	if isfile(myOutputFilePath) && !(get_recompute_optimal_strategies(aSimulationParameters))

		myHasToComputeOptimalStrategy = false
		HelpFilesModule.info_logs(
					  string(
						 "\nMeanCVaRModule 105:\n",
						 "The optimal strategy given the parameters passed as argument has already been computed and will not be recomputed.\n",
						 "The file related is named:\n",
						 myOutputFilePath,
						 "\nPlease delete this file or add as argument `aRecomputeOptimalStrategies=true` in the structure `aSimulationParameters` to force the recomputation of the optimal strategy."
						 ),
					  aSimulationParameters,
					  myStacktrace
					  )
	end

	###################################################
	# PART 2: computes the optimal strategy if needed #
	###################################################

	# computes the optimal strategy has it has never been computed before with the arguments given
	if myHasToComputeOptimalStrategy

		# gets the parameters for the estimation of the Mean-CVaR
		myParametersCVaROptimisation = get_parameters_CVaR_optimisation(aSimulationParameters)
		myNSamplesCVaR               = myParametersCVaROptimisation["NSamples"]
		myMaximumNOfIterations       = myParametersCVaROptimisation["MaximumNumberOfIterations"]

		# the probability of occurence of each scenario ω ∈ Ω (that will be used in the computation of the expectation)
		πProb = 1/myNSamplesCVaR

		############################################# 
		# Part 2.1: solves the optimisation problem # 
		############################################# 

		# gets the uncertainty structure belief of all the trader according to trader `aTraderIndex`
		myTradersUncertaintyStructure = MarketDetailsModule.get_traders_uncertainty_structure(myMarketDetailsBelief)

		# gets the uncertainty structure of trader `aTraderIndex`
		myUncertaintyStructure = myTradersUncertaintyStructure[aTraderIndex]

		# get the market parameter according to trader `aTraderIndex`
		myGammas   = get_gammas(myMarketDetailsBelief)
		myEpsilons = get_epsilons(myMarketDetailsBelief)
		myEtas     = get_etas(myMarketDetailsBelief)
		myTaus     = get_taus(myMarketDetailsBelief)

		# get the matrix A of the quadratic form of the trading cost (it does not depend on the realisation ω ∈ Ω): Qω' * A * Qω + Qω' * Bω + cω
		myGb_eq, myGb_neq, myEb, myEb_m, myΓb, myΛb = NEHelpFunctionsModule.get_matrices_for_risk_set(aγ = myGammas, aη = myEtas, aτ = myTaus)
		A = zeros(myNTradingPeriods,myNTradingPeriods)
		A[1:end-1,1:end-1] = 0.5 * (2*myEb + 2*myEb_m - myΓb)

		# gets the variables one has to optimise
		myOptimiseTradingPlan          = get_optimise_trading_plan(aSimulationParameters)
		myOptimiseRedistributionMatrix = get_optimise_redistribution_matrix(aSimulationParameters)

		# gets the redistribution matrix
		myβ = myPlayersRedMatrix[aTraderIndex]

		if myOptimiseRedistributionMatrix
			@error(
			       string(
				      "\nMeanCVaRModule 106:\n",
				      "The code is not ready yet to be able to optimise the redistribution matrix",
				      )
			       )
		end

		if myOptimiseTradingPlan && myOptimiseRedistributionMatrix
			@error(
			       string(
				      "\nMeanCVaRModule 107:\n",
				      "The optimisation (Mean-CVaR with Gurobi) is not designed to optimise both the trading plan and the redistribution matrix at the same time.",
				      )
			       )
		end

		# .......................... #
		# Generates the prices moves #
		# .......................... #

		myConsiderPriceMoves = get_consider_price_moves(myUncertaintyStructure)
		myPricesMovesDistributions = get_prices_moves_distributions(myUncertaintyStructure)

		Random.seed!(aSeed + 1)
		myPricesMovesDistributions = get_prices_moves_distributions(myUncertaintyStructure)
		myPricesMovesRealisations   = zeros(myNTradingPeriods,myNSamplesCVaR)
		if get_consider_price_moves(myUncertaintyStructure)
			for ω in 1:myNSamplesCVaR, myPeriod in 1:myNTradingPeriods
				myPricesMovesRealisations[myPeriod,ω] = rand(myPricesMovesDistributions[myPeriod])
			end
		end

		# .................................................... #
		# Computes the trading quantities of the other players #
		# .................................................... #

		# gets the forecast updates distributions
		myInitialDemands               = [get_initial_demand_forecast(myLocalTraderUncertaintyStructure) for myLocalTraderUncertaintyStructure in myTradersUncertaintyStructure]
		myConsiderForecastUpdates      = [get_consider_forecast_updates(myLocalTraderUncertaintyStructure) for myLocalTraderUncertaintyStructure in myTradersUncertaintyStructure]
		myForecastUpdatesDistributionsAllPlayers = [get_forecast_updates_distributions(myLocalTraderUncertaintyStructure) for myLocalTraderUncertaintyStructure in myTradersUncertaintyStructure]

		# gets the trading variables of the other traders
		myQOthersPos = zeros(myNTradingPeriods,myNSamplesCVaR)
		myQOthersNeg = zeros(myNTradingPeriods,myNSamplesCVaR)

		# fills the volume traded by the other players
		Random.seed!(aSeed + 2)
		myQω = 0.0
		myForecastUpdatesRealisationOthers = zeros(myNTradingPeriods)
		for myLocalPlayerIndex in 1:myNTraders
			if myLocalPlayerIndex != aTraderIndex # the trading volume of `aTraderIndex` is part of the variables of the optimisation problem
				for ω in 1:myNSamplesCVaR

					# gets a realisation of the forecast moves
					if myConsiderForecastUpdates[myLocalPlayerIndex]
						for i in 1:myNTradingPeriods
							myForecastUpdatesRealisationOthers[i] = rand(myForecastUpdatesDistributionsAllPlayers[myLocalPlayerIndex][i])
						end
					else
						myForecastUpdatesRealisationOthers = zeros(myNTradingPeriods)
					end

					# computes the volume traded by the trader `myLocalPlayerIndex` on each trading period
					for i in 1:myNTradingPeriods
						myQuantityToTradeInCaseOfRealisationω = myPlayersTradingPlan[myLocalPlayerIndex][i] * myInitialDemands[myLocalPlayerIndex] + reduce(+, [ myForecastUpdatesRealisationOthers[k] * ( myPlayersTradingPlan[myLocalPlayerIndex][i] + myPlayersRedMatrix[myLocalPlayerIndex][k,i] * sum( myPlayersTradingPlan[myLocalPlayerIndex][r] for r =1:k ) )  for k = 1:i-1], init=0.0)
						myQOthersPos[i,ω] += max(myQω,0)
						myQOthersNeg[i,ω] += max(-myQω,0)
					end
				end
			end
		end

		# ............................... #
		# Generates the forecast updates  #
		# ............................... #

		Random.seed!(aSeed + 3)
		myForecastUpdatesDistributions = myForecastUpdatesDistributionsAllPlayers[aTraderIndex]
		myForecastUpdatesRealisations = zeros(myNTradingPeriods,myNSamplesCVaR)
		myQTot = zeros(myNSamplesCVaR)
		if get_consider_forecast_updates(myUncertaintyStructure)
			for ω in 1:myNSamplesCVaR
				for myPeriod in 1:myNTradingPeriods
					myForecastUpdatesRealisations[myPeriod,ω] = rand(myForecastUpdatesDistributions[myPeriod])
				end

				# computes the final volume target of `aTraderIndex` in the case of realisation ω
				myQTot[ω] = myInitialDemands[aTraderIndex] + sum(myForecastUpdatesRealisations)
			end
		end

		##############################################################################
		# Creates the Gurobi model that is the model to find the next extreme μProb. #
		##############################################################################

		# here μProb is the probability of occurence of each scenario ω ∈ Ω when taking risk (that will be used in the computation of the E-CVaR_α)

		# initialises the model that find the next extreme point to include
		myModelFindNextRiskSetVertex = Model(optimizer_with_attributes(() -> Gurobi.Optimizer(ourGurobiEnvironment)))
		JuMP.set_optimizer_attribute(myModelFindNextRiskSetVertex, "OutputFlag", 0)

		# initialises the variables
		@variable(myModelFindNextRiskSetVertex, ( 1 - myRiskAversion ) * πProb <= μProb[1:myNSamplesCVaR] <= ( 1 - myRiskAversion ) * πProb + myRiskAversion * ( πProb / myAlpha ) )

		# Note: the objective will depend on the result of `myModelRelaxation` and will be updated dynamically at each iteration
		set_objective_sense(myModelFindNextRiskSetVertex, MOI.MAX_SENSE)

		# adds the constraint that `μProb` must define a probability distribution on the sample space Ω
		@constraint(myModelFindNextRiskSetVertex, sum(μProb) == 1)

		############################################################################################	
		# Initialises the variables used in the algorithm to solve the E-CVaR optimisation problem #	
		############################################################################################	

		# creates the set with the hashes of the extreme vertices and the triples (μ,CoeffPos,CoeffNeg)
		mySetμProbExt = []

		# initialises the variables used in the algorithm
		myNewY        = zeros(myNTradingPeriods)
		myNewCoeffPos = zeros(myNTradingPeriods,myNSamplesCVaR)
		myNewCoeffNeg = zeros(myNTradingPeriods,myNSamplesCVaR)
		myNewμProb    = zeros(myNSamplesCVaR)
		myNewOptimalObjValue     = 0
		myOptimisationSuccessful = false

		# initialises the hashes of the variables. Note that myNewCoeffNeg is the complement of myNewCoeffPos, so taking into account 1 hash among them is enough 
		myNewCoeffPosHash = nothing
		myNewμProbHash    = nothing

		#############################
		# Initialises the algorithm #
		#############################

		# generates a random strategy y ∈ Y
		Random.seed!(aSeed + 4)
		myNewY = rand(myNTradingPeriods)
		myNewY = myNewY ./ ( sum(myNewY) )

		# updates the objective function of the model that must find the next extreme vertex of the risk-set
		set_objective_model_next_extreme_vertex!(
							 aNTradingPeriods             = myNTradingPeriods,
							 aNSamplesCVaR                = myNSamplesCVaR,
							 aModel                       = myModelFindNextRiskSetVertex,
							 aμ                           = myModelFindNextRiskSetVertex[:μProb],
							 aY                           = myNewY,
							 aβ                           = myβ,
							 aPricesMovesRealisations     = myPricesMovesRealisations,
							 aInitialDemand               = myInitialDemands[aTraderIndex],
							 aForecastUpdatesRealisations = myForecastUpdatesRealisations,
							 aτ                           = myTaus,
							 aγ                           = myGammas,
							 aϵ                           = myEpsilons,
							 aη                           = myEtas,
							 aQOthersPos                  = myQOthersPos,
							 aQOthersNeg                  = myQOthersNeg,
							 aCoeffPos                    = myNewCoeffPos,
							 aCoeffNeg                    = myNewCoeffNeg,
							 )

		# optimises the model to get the new extreme vertex
		#= @time begin =#
		#= 	println_logs(string("Begin: optimise initial model to find extreme vertex."),aSimulationParameters,myStacktrace) =#
		myStatusModelFindNextRiskSetVertex   = optimize!(myModelFindNextRiskSetVertex)
		myObjValueModelFindNextRiskSetVertex = objective_value(myModelFindNextRiskSetVertex)
		#= println_logs(string("End: optimise initial model to find extreme vertex."),aSimulationParameters,myStacktrace) =#
		#= end =#

		# gets the new extreme μProb and add it to the set U^ext and H
		myNewμProb = JuMP.value.(myModelFindNextRiskSetVertex[:μProb]) 

		myNewμProbHash = hash(myNewμProb)
		push!(mySetμProbExt,myNewμProbHash)

		##################################################################################################################
		# Start of the algorithm, alternating between solving (p) and (Q) for which wee add dynamically more constraints #
		##################################################################################################################

		# sets the values to check the termination criteria
		myOptimisationμProbTerminated = false
		myIterOnμProb = 0 

		while !(myOptimisationμProbTerminated) && myIterOnμProb < myMaximumNOfIterations

			# increment the iteration number
			myIterOnμProb += 1

			println_logs(string("\n======================================\nIteration $myIterOnμProb: optimise μ problem."),aSimulationParameters,myStacktrace)
			sleep(0.1)

			# creates the Gurobi model that is the model to find the next extreme μProb
			myModelRelaxation = get_relaxation_model_without_coefficient_constraints(
												 aNTradingPeriods     = myNTradingPeriods,
												 aOptimiseTradingPlan = myOptimiseTradingPlan,
												 aPlayerTradingPlan   = myPlayersTradingPlan[aTraderIndex]
												 )

			# adds the initial constraint to the relaxation model which corresponds to the optimal one of the previous μProb iteration
			add_constraint_model_relaxation!(
							 aNTradingPeriods             = myNTradingPeriods,
							 aNSamplesCVaR                = myNSamplesCVaR,
							 aModel                       = myModelRelaxation,
							 aμ                           = myNewμProb,
							 aY                           = myModelRelaxation[:y],
							 aβ                           = myβ,
							 aT                           = myModelRelaxation[:t],
							 aPricesMovesRealisations     = myPricesMovesRealisations,
							 aInitialDemand               = myInitialDemands[aTraderIndex],
							 aForecastUpdatesRealisations = myForecastUpdatesRealisations,
							 aτ                           = myTaus,
							 aγ                           = myGammas,
							 aϵ                           = myEpsilons,
							 aη                           = myEtas,
							 aQOthersPos                  = myQOthersPos,
							 aQOthersNeg                  = myQOthersNeg,
							 aCoeffPos                    = myNewCoeffPos,
							 aCoeffNeg                    = myNewCoeffNeg
							 )

			# reinitialises the set H and the number of iterations
			mySetH = []

			# adds the hash of `myNewCoeffPos` to H
			push!(mySetH,hash(myNewCoeffPos))

			# sets the values to check the termination criteria
			myIterOnYRelax = 0
			myOptimisationYRelaxTerminated = false

			while !(myOptimisationYRelaxTerminated) && myIterOnYRelax < myMaximumNOfIterations

				# increment the iteration number
				myIterOnYRelax += 1

				println_logs(string("Iteration $myIterOnYRelax: optimise relaxation problem."),aSimulationParameters,myStacktrace)
				sleep(0.1)

				# solves the relaxation model
				@time begin
					myStatusModelRelaxation = optimize!(myModelRelaxation)
				end

				# gets the new y and the new coeff
				myNewY = JuMP.value.(myModelRelaxation[:y])

				# updates the value of `myNewCoeffPos` and `myNewCoeffNeg`
				update_indices_with_postive_and_negative_volumes!(
										  aCoeffPosToChange            = myNewCoeffPos,
										  aCoeffNegToChange            = myNewCoeffNeg,
										  aInitialDemand               = myInitialDemands[aTraderIndex],
										  aForecastUpdatesRealisations = myForecastUpdatesRealisations,
										  aY                           = myNewY,
										  aβ                           = myβ
										  )

				# computes the hash of `myNewCoeffPos`
				myNewCoeffPosHash = hash(myNewCoeffPos)

				# checks if the relaxation is a relaxation or not
				# if  true: this means that the optimisation was still a relaxation and thus should be reoptimised 
				#    false: this means that the coefficients are not correct for y, and should thus be reoptimised after adding a constraint on the coefficients 
				if in(myNewCoeffPosHash,mySetH)
					# sets the termination criteria to true
					myOptimisationYRelaxTerminated = true
				else

					# adds the hash of myNewCoeffPosHash to H
					push!(mySetH,myNewCoeffPosHash)

					# sets the termination criteria to false
					myOptimisationYRelaxTerminated = false

					# adds a new constraint to the relaxation model which corresponds to the new coefficients
					add_constraint_model_relaxation!(
									 aNTradingPeriods             = myNTradingPeriods,
									 aNSamplesCVaR                = myNSamplesCVaR,
									 aModel                       = myModelRelaxation,
									 aμ                           = myNewμProb,
									 aY                           = myModelRelaxation[:y],
									 aβ                           = myβ,
									 aT                           = myModelRelaxation[:t],
									 aPricesMovesRealisations     = myPricesMovesRealisations,
									 aInitialDemand               = myInitialDemands[aTraderIndex],
									 aForecastUpdatesRealisations = myForecastUpdatesRealisations,
									 aτ                           = myTaus,
									 aγ                           = myGammas,
									 aϵ                           = myEpsilons,
									 aη                           = myEtas,
									 aQOthersPos                  = myQOthersPos,
									 aQOthersNeg                  = myQOthersNeg,
									 aCoeffPos                    = myNewCoeffPos,
									 aCoeffNeg                    = myNewCoeffNeg
									 )
				end

				# checks wether the maximum number of iterations has been reached
				if myIterOnYRelax == myMaximumNOfIterations
					# sets the termination criteria to true
					myOptimisationYRelaxTerminated = true

					# raises an error
					@warn(string(
						     "\nMeanCVaRModule 108:\n",
						     "The maximum number of iterations to solve the relaxtion model has been reached.",
						     )
					      )
				end
			end

			# updates the objective function of the model that must find the next extreme vertex of the risk-set with the new incumbent `myNewY`
			set_objective_model_next_extreme_vertex!(
								 aNTradingPeriods             = myNTradingPeriods,
								 aNSamplesCVaR                = myNSamplesCVaR,
								 aModel                       = myModelFindNextRiskSetVertex,
								 aμ                           = myModelFindNextRiskSetVertex[:μProb],
								 aY                           = myNewY,
								 aβ                           = myβ,
								 aPricesMovesRealisations     = myPricesMovesRealisations,
								 aInitialDemand               = myInitialDemands[aTraderIndex],
								 aForecastUpdatesRealisations = myForecastUpdatesRealisations,
								 aτ                           = myTaus,
								 aγ                           = myGammas,
								 aϵ                           = myEpsilons,
								 aη                           = myEtas,
								 aQOthersPos                  = myQOthersPos,
								 aQOthersNeg                  = myQOthersNeg,
								 aCoeffPos                    = myNewCoeffPos,
								 aCoeffNeg                    = myNewCoeffNeg,
								 )

			# optimises the model to get the new extreme vertex
			#= @time begin =#
			#= 	println_logs(string("Begin: optimise model to find extreme vertex."),aSimulationParameters,myStacktrace) =#
			myStatusModelFindNextRiskSetVertex   = optimize!(myModelFindNextRiskSetVertex)
			myObjValueModelFindNextRiskSetVertex = objective_value(myModelFindNextRiskSetVertex)
			#= println_logs(string("End: optimise model to find extreme vertex."),aSimulationParameters,myStacktrace) =#
			#= end =#

			# gets the new extreme vertex
			myNewμProb = JuMP.value.(myModelFindNextRiskSetVertex[:μProb])
			myNewOptimalObjValue = objective_value(myModelFindNextRiskSetVertex)

			# computes the hash of the new extreme vertex `myNewμProb`
			myNewμProbHash = hash(myNewμProb)

			# checks if the new extreme vertex `myNewμProbHash` has already been visited
			# if  true: this means that the optimisation was still a relaxation and thus should be reoptimised 
			#    false: this means that the coefficients are not correct for y, and should thus be reoptimised after adding a constraint on the coefficients 
			if in(myNewμProbHash,mySetμProbExt)
				# sets the termination criteria to true
				myOptimisationμProbTerminated = true
				myOptimisationSuccessful = true
			else
				# adds the hash of myNewμProbHash to mySetμProbExt
				push!(mySetμProbExt,myNewμProbHash)

				# sets the termination criteria to false
				myOptimisationμProbTerminated = false
			end

			# checks wether the maximum number of iterations has been reached
			if myIterOnμProb == myMaximumNOfIterations
				# sets the termination criteria to true
				myOptimisationμProbTerminated = true
				myOptimisationSuccessful = false

				# raises an error
				@warn(string(
					     "\nMeanCVaRModule 109:\n",
					     "The maximum number of iterations to solve the optimal trading optimisation problem has been reached.",
					     )
				      )
			end
		end

		################################################## 
		# Part 2.2: saves the output of the optimisation # 
		################################################## 

		# gets the optimal strategy and the optimal objective value
		myOptimalStrategy       = nothing
		myOptimalObjectiveValue = nothing
		if myOptimisationμProbTerminated # the optimisation was successful
			myOptimalStrategy = get_strategy(
							 aNTradingPeriods      = myNTradingPeriods,
							 aTradingPlan          = myNewY,
							 aRedistributionMatrix = myβ
							 )
			myOptimalObjectiveValue = myNewOptimalObjValue
		else # the optimisation was NOT successful
			@error(
			       string(
				      "\nMeanCVaRModule 108:\n",
				      "The optimisation (Mean-CVaR with Gurobi) was not successful.",
				      )
			       )
			myOptimalStrategy       = StrategyModule.get_strategy()
			myOptimalObjectiveValue = -1.0
		end

		# saves the optimal execution strategy in the file `myOutputFilePath`
		myDetailsToSave = Dict()
		myDetailsToSave["Method"]                 = get_method(aSimulationParameters)
		myDetailsToSave["Trader"]                 = get_dict_from_trader(myTrader)
		myDetailsToSave["SimulationParameters"]   = get_dict_from_simulation_parameters(aSimulationParameters)
		myDetailsToSave["TradingPlan"]            = get_trading_plan(myOptimalStrategy)
		myDetailsToSave["RedistributionMatrix"]   = get_redistribution_matrix(myOptimalStrategy)
		myDetailsToSave["ObjectiveValue"]         = myOptimalObjectiveValue
		myDetailsToSave["OptimisationSuccessful"] = myOptimisationSuccessful
		save_result!(myOutputFilePath,myDetailsToSave)
	end

	# loads the optimal strategy from the file `myOutputFilePath`
	myOptimalStrategy = load_strategy_from_file(myOutputFilePath)

	myDict = load_result(myOutputFilePath)
	myOptimalObjectiveValue  = myDict["ObjectiveValue"]
	myOptimisationSuccessful = myDict["OptimisationSuccessful"]

	return myOptimalStrategy, myOptimalObjectiveValue, myOptimisationSuccessful, myOutputFilePath
end

end
