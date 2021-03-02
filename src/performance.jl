# NAME: performance.jl
# AUTHOR: Julien Vaes
# DATE: May 22, 2019
# DESCRIPTION: module to evaluate the performance of a trading strategy.

module PerformanceModule

###################
## Load Packages ##
###################

using Distributed
using Statistics
using LinearAlgebra
using FileIO

############################
## Load Personal Packages ##
############################

using ..MarketDetailsModule
using ..TraderModule
using ..StrategyModule
using ..SimulationParametersModule
using ..TradingCostModule
using ..MeanCVaRModule
using ..MeanVarianceModule
using ..HelpFilesModule
using ..OptimalStrategyModule

######################
## Export functions ##
######################

export get_optimal_plan_trading_cost_realisations
export get_performance_dict_given_realisation
export get_optimal_trading_plan_performance

######################
## Module variables ##
######################

ourNSamplesPerIterationPerformance = Dict(
					  "Mean-Variance"    => 5*10^2,
					  "Mean-CVaR_Optim"  => 10^4,
					  "Mean-CVaR_Gurobi" => 10^4,
					  "Mean-CVaR"        => 10^4,
					  )

ourFunctionsWithRecourse = Dict(
				"Mean-Variance"=>MeanVarianceModule.get_optimal_strategy_mean_variance
				)

######################
## Module functions ##
######################

"""
#### Definition
```
get_optimal_plan_trading_cost_realisations(;
aMarketDetailsForRealisations::MarketDetails,
aTraderIndex::Int64,
aTraders::Array{Trader,1},
aStrategies::Array{Strategy,1},
aSimulationParameters::SimulationParameters,
aNTradingCostRealisations::Int64=10^7,
aNTradingCostRealisationsPerIteration::Int64=10^4
)
```

returns a number of `aNTradingCostRealisations` random realisations of the trading cost of the optimal strategy that has been computed with the corresponding arguments.
Note that here the ground truth distribution of the market are estimated, i.e. the distribution contained in `aMarketDetails`.
This function assumes that the optimal strategy corresponding to the arguments has already been computed and saved in a file.
This function is not appropriate for recursive methods.

#### Argument
* `aMarketDetails::MarketDetails`: a structure containing all the details of the trading market.
* `aTrader::Trader`: a structure containing all the details of a trader.
* `aSimulationParameters::SimulationParameters`: a structure containing all the details of the parameters used to compute the optimal trading plan such as the method to use, the number of samples in the CVaR estimation.
* `aNTradingCostRealisations::Int=10^7`: the number of samples to generate.
* `aNSamplesPerIteration::Int=10^4,`: the maximum number of samples to generated on a processor at the same time.
"""
function get_optimal_plan_trading_cost_realisations(;
						    aMarketDetailsForRealisations::MarketDetails,
						    aTraderIndex::Int64,
						    aTraders::Array{Trader,1},
						    aStrategies::Array{Strategy,1},
						    aSimulationParameters::SimulationParameters,
						    aNTradingCostRealisations::Int64=10^7,
						    )

	# gets the stack trace, useful to know if ones has to print or not the logs of this function
	myStacktrace = stacktrace()

	# gets the number of trading periods
	myNTradingPeriods = MarketDetailsModule.get_n_trading_periods(aMarketDetailsForRealisations)

	# gets the information if the trading realisation must be computated based on a recourse evaluation 
	myConsiderRecourse = get_consider_recourse(aSimulationParameters)

	if myConsiderRecourse

		# the optimal strategy must be recomputed for each trading period,
		# and the performance depends on the ground truth distributions of the random variables inherent to the market, contained in the `aMarketDetails` structure.
		# Hence, the parameters `aIncludeTraderHash` and `aIncludeSimulationParametersHash` must be set to `true`, and `aIncludePartialMarketDetailsHash` to `false`,
		# in order to have the hash of the full structures.
		# Indeed the results depends on the market details
		myOutputRealisationsFilePath = get_default_output_file_path(
									    aMarketDetails                   = aMarketDetailsForRealisations,
									    aTraderIndex                     = aTraderIndex,
									    aTraders                         = aTraders,
									    aStrategies                      = aStrategies,
									    aSimulationParameters            = aSimulationParameters,
									    aIncludePartialMarketDetailsHash = false,
									    aIncludeSimulationParametersHash = true,
									    aIncludeTradersHash              = true,
									    aIncludeStrategiesHash           = true,
									    aSpecificFileNameExtension       = "_recursive_realisations",
									    aSpecificFolder                  = "outputs/trading_cost_realisations/"
									    )
	else

		# the performance depends on the ground truth distributions of the random variables inherent to the market.
		# Hence the parameter `aIncludePartialMarketDetailsHash` must be set to `false` in order to have the hash of the full structure.
		myOutputRealisationsFilePath = get_default_output_file_path(
									    aMarketDetails                   = aMarketDetailsForRealisations,
									    aTraderIndex                     = aTraderIndex,
									    aTraders                         = aTraders,
									    aStrategies                      = aStrategies,
									    aSimulationParameters            = aSimulationParameters,
									    aIncludePartialMarketDetailsHash = false, # as the performance depends on the market details structure which contains the ground truth distribution
									    aIncludeSimulationParametersHash = false,
									    aIncludeTradersHash              = false,
									    aIncludeStrategiesHash           = true,
									    aSpecificFileNameExtension       = "_realisations",
									    aSpecificFolder                  = "outputs/trading_cost_realisations/"
									    )
	end

	# create the necessary folders in order to save the realisations in `myOutputRealisationsFilePath`
	create_relevant_folders!(myOutputRealisationsFilePath)

	if get_recompute_performances(aSimulationParameters)
		# removes the file given by the path `myOutputRealisationsFilePath`.
		# force=true allows that a non-existing path is not treated as error. 
		rm(myOutputRealisationsFilePath, force=true) 
	end

	############################################################
	## Checks if some realisations have already been computed ##
	############################################################

	myTradingCostRealisationsAlreadyComputed = Array{Float64,1}(undef, 0)
	myNTradingCostRealisationsAlreadyComputed = 0
	if isfile(myOutputRealisationsFilePath)
		myTradingCostRealisationsAlreadyComputed = load_result(myOutputRealisationsFilePath)["TradingCostRealisations"]
		myNTradingCostRealisationsAlreadyComputed = length(myTradingCostRealisationsAlreadyComputed)
	end
	myNTradingCostRealisationsToCompute = aNTradingCostRealisations - min(myNTradingCostRealisationsAlreadyComputed,aNTradingCostRealisations)

	# tells as an info if there is already some realisation that exists 
	if myNTradingCostRealisationsToCompute < aNTradingCostRealisations
		info_logs(
			  string(
				 "\nPerformanceModule 101:\n",
				 "Some realisations have already been computed and will be imported. There are consequently a number of ",
				 myNTradingCostRealisationsToCompute,
				 " left to compute out of the initial ",
				 aNTradingCostRealisations,
				 ".\nIf you want to force the recomputation of all realisations then put the parameter `aRecomputePerformances=true` in the SimulationParameters structure `aSimulationParameters`.",
				 "Otherwise, you can also delete the file:\n",
				 myOutputRealisationsFilePath
				 ,
				 "."),
			  aSimulationParameters,
			  myStacktrace
			  )
	end

	# if one has already computed previously enough realisations corresponding to the arguments we return them
	if myNTradingCostRealisationsToCompute == 0
		return myTradingCostRealisationsAlreadyComputed[1:aNTradingCostRealisations]
	end

	######################################################################
	## Gets the number of realisation that can be computed by iteration ##
	######################################################################

	# Based on Lemma 2.9 of [Optimal execution strategy with an uncertain volume target, Vaes & Hauser (2018)],
	# the recursive Mean-Variance strategy can be reproduced with the Mean-CVaR framework if |E[P] = 0 (the expectation of the price shift is 0) and if for all i ϵ_i = ϵ
	# we then adapt the number of iteration that can be computed by iteration in this case

	# gets the value of ϵ
	myEpsilons = get_epsilons(get_market_details_belief(aTraders[aTraderIndex]))

	myMethod = get_method(aSimulationParameters)
	if !(in(myMethod,keys(ourNSamplesPerIterationPerformance)))
		@error(
		       string(
			      "\nPerformanceModule 102:\n",
			      "Method ",
			      myMethod,
			      " is not listed in order to have the number of iteration to simulate per iteration. Please use one of these method",
			      keys(ourFunctionsWithRecourse),
			      "."
			      )
		       )
	end

	myNTradingCostRealisationsPerIteration = ourNSamplesPerIterationPerformance[myMethod]
	if get_method(aSimulationParameters) == "Mean-Variance" &&  all(y->y==myEpsilons[1],myEpsilons)
		myNTradingCostRealisationsPerIteration = ourNSamplesPerIterationPerformance["Mean-CVaR"]
	end

	##############################################
	## Checks the number of realisations needed ##
	##############################################

	# number of iterations that sample `aNSamplesPerIteration` realisations needed in order to have `myNTradingCostRealisationsToCompute` realisations of the trading cost
	myNIterations = Int(floor(myNTradingCostRealisationsToCompute/(myNTradingCostRealisationsPerIteration)))

	# number of samples to generate in the last iteration in order to have `myNTradingCostRealisationsToCompute` samples
	myNTradingCostRealisationsInLastIteration = myNTradingCostRealisationsToCompute-myNIterations*myNTradingCostRealisationsPerIteration

	# vector with the number of samples to generate per iterate
	myNTradingCostRealisationsToRun = myNTradingCostRealisationsPerIteration*ones(Int64,myNIterations)
	if myNTradingCostRealisationsInLastIteration > 0
		push!(myNTradingCostRealisationsToRun,myNTradingCostRealisationsInLastIteration)
	end

	############################################
	## Computes the trading cost realisations ##
	############################################

	myTradingCostRealisations = nothing
	if myConsiderRecourse
		# Based on Lemma 2.9 of [Optimal execution strategy with an uncertain volume target, Vaes & Hauser (2018)],
		# the recursive Mean-Variance strategy can be reproduced with the Mean-CVaR framework if |E[P] = 0 (the expectation of the price shift is 0) and if for all i ϵ_i = ϵ
		# the Mean-CVaR strategy reprodicing the recursive Mean-Variance strategy is the one with the redistribution matrix induced bu the Mean-Variance strategy under price uncertainty only 
		if get_method(aSimulationParameters) == "Mean-Variance" &&  all(y->y==myEpsilons[1],myEpsilons)

			# computes the optimal strategy in the Mean-Variance framework when only the price uncertainty is taken into account
			myOptimalStrategyMeanVariance, myOptimalObjectiveValue, myOptimisationSuccessful, myOutputFilePath = get_optimal_strategy_mean_variance(
																				aTraderIndex = aTraderIndex,
																				aTraders     = aTraders,
																				aStrategies  = aStrategies,
																				aSimulationParameters = get_new_simulation_parameters(
																										      aSimulationParameters,
																										      aConsiderRecourse = false,
																										      )
																				)

			# gets the optimal trading plan
			myOptimalTradingPlanMeanVariance = get_trading_plan(myOptimalStrategyMeanVariance)

			# defines the redistribution matrix induced by the optimal trading plan, i.e. Eq (28) of [Optimal execution strategy with an uncertain volume target, Vaes & Hauser (2018)]
			# However, because of numerical issues, it is safe to solve the optimisation problem for every subperiod range, mean variance with m, m-1, ... and 1 periods
			myRedistributionMatrixReproducingRecursivityMeanVariance = zeros(myNTradingPeriods,myNTradingPeriods)
			for i = 1:myNTradingPeriods-1

				# IN THEORY
				# myRedistributionMatrixReproducingRecursivityMeanVariance[i,i+1:end] = myOptimalTradingPlanMeanVariance[i+1:end]./(sum(myOptimalTradingPlanMeanVariance[i+1:end]))

				# gets the optimal trading plan when there is only `myNTradingPeriods- i` trading periods
				myLocalOptimalStrategy,
				myLocalOptimalObjectiveValue,
				myLocalOptimisationSuccessful,
				myLocalOutputFilePath = MeanVarianceModule.get_optimal_strategy_mean_variance(
													      aTraderIndex                             = aTraderIndex,
													      aTraders                                 = aTraders,
													      aStrategies                              = aStrategies,
													      aSimulationParameters                    = get_new_simulation_parameters(aSimulationParameters, aRecomputeOptimalStrategies = true),
													      aDecisionTimeToComputeOptimalTradingPlan = i
													      )
				myLocalOptimalTradingPlanMeanVariance = get_trading_plan(myLocalOptimalStrategy)

				myRedistributionMatrixReproducingRecursivityMeanVariance[i,i+1:end] = myLocalOptimalTradingPlanMeanVariance
			end

			# get the strategy that reproduces the recursive Mean-Variance strategy in the Mean-CVaR framework
			myMeanCVaRStrategyEquivalentMeanVarianceWithRecourse = get_strategy(
											    aNTradingPeriods      = myNTradingPeriods,
											    aTradingPlan          = myOptimalTradingPlanMeanVariance,
											    aRedistributionMatrix = myRedistributionMatrixReproducingRecursivityMeanVariance
											    )


			# changes the market belief of the trader `aTraderIndex` as it will be assumed here that the realisation are genereted based on `aMarketDetailsForRealisations`
			myTraders = deepcopy(aTraders)
			myTraderPerfectKnowledge = get_new_trader(
								  aTraders[aTraderIndex],
								  aMarketDetailsBelief = aMarketDetailsForRealisations
								  )

			myTraders[aTraderIndex] = myTraderPerfectKnowledge

			# changes the strategy of the trader `aTraderIndex` by the replicating one in the Mean-CVaR framework
			myStrategies = deepcopy(aStrategies)
			myStrategies[aTraderIndex] = myMeanCVaRStrategyEquivalentMeanVarianceWithRecourse


			# splits the computation among the different processors
			myTradingCostRealisations = @distributed (vcat) for myIter in eachindex(myNTradingCostRealisationsToRun)

				# computes the trading cost realisations with the help of the Mean-CVaR framework instead of a recursive Mean-Variance (which is significantly faster)
				myTradingCostRealisationsLocal = TradingCostModule.get_trading_cost(
												    aTraderIndex          = aTraderIndex,
												    aTraders              = myTraders,
												    aStrategies           = myStrategies,
												    aNSamples             = myNTradingCostRealisationsToRun[myIter],
												    aSimulationParameters = get_new_simulation_parameters(
																			  aSimulationParameters,
																			  aConsiderRecourse           = false,
																			  aMethod                     = "Mean-CVaR",
																			  aParametersCVaROptimisation = Dict("NSamplesPerIteration" => ourNSamplesPerIterationPerformance["Mean-CVaR"]),
																			  )
												    )
			end
		else
			@error(
			       string(
				      "\nPerformanceModule 103:\n",
				      "Estimating the recursive performance is not implemented yet. ",
				      "It is only supported in the mean-variance framework when the fixed impact parameter of the temporary impact is equal on each trading period, ∀i: ϵi = ϵ."
				      )
			       )
		end
	else
		# splits the computation among the different processors
		myTradingCostRealisations = @distributed (vcat) for myIter in eachindex(myNTradingCostRealisationsToRun)

			# change the market belief of the trader `aTraderIndex` as it will be assumed here that the realisation are genereted based on `aMarketDetailsForRealisations`
			myTraders = deepcopy(aTraders)
			myTraderPerfectKnowledge = get_new_trader(
								  aTraders[aTraderIndex],
								  aMarketDetailsBelief = aMarketDetailsForRealisations
								  )

			myTraders[aTraderIndex] = myTraderPerfectKnowledge
			myTradingCostRealisationsLocal = TradingCostModule.get_trading_cost(
											    aTraderIndex          = aTraderIndex,
											    aTraders              = myTraders,
											    aStrategies           = aStrategies,
											    aSimulationParameters = aSimulationParameters,
											    aNSamples             = myNTradingCostRealisationsToRun[myIter]
											    )
		end
	end

	myTradingCostRealisations = vcat(myTradingCostRealisations,myTradingCostRealisationsAlreadyComputed)
	HelpFilesModule.save_result!(myOutputRealisationsFilePath,Dict("TradingCostRealisations"=>myTradingCostRealisations))

	return Float64.(myTradingCostRealisations)
end

"""
```
get_performance_dict_given_realisation()
```

returns a dictionary continaing the statistical values of `aRealisations`, i.e. expectation, variance, CVaR, mean-variance, mean-CVaR, etc.

### Arguments
"""
function get_performance_dict_given_realisation(;
						aRiskAversion,
						aRealisations::Array{Float64,1},
						aAlphaLevelsToCompute
						)


	# gets the number of realisations
	myNTradingCostRealisations = size(aRealisations,1)

	# sorts the trading cost in increasing order
	sort!(aRealisations,rev=true)

	# estimates the expected value, variance and CVaR
	myTradingCostExpectedValue = mean(aRealisations)
	myTradingCostVariance      = var(aRealisations)

	# initialises the variables containing the CVaR, Mean-CVaR, VaR and Mean-VaR of the trading cost
	myCVaRs     = Dict()
	myMeanCVaRs = Dict()

	myVaRs     = Dict()
	myMeanVaRs = Dict()

	for myLocalAlpha in aAlphaLevelsToCompute
		myNTradingCostRealisationsCVaREstimate = Int(ceil(length(aRealisations)*(myLocalAlpha)))

		# gets the CVaR, i.e. mean above the 1-alpha quantile
		@views myLocalCVaR = mean(aRealisations[1:myNTradingCostRealisationsCVaREstimate])
		myCVaRs[string("CVaR_",myLocalAlpha)] = myLocalCVaR

		# computes the Mean-CVaR trade-off
		myMeanCVaRs[string("Mean-CVaR_",myLocalAlpha)] = (1 - aRiskAversion) * myTradingCostExpectedValue + aRiskAversion * myLocalCVaR

		# gets the VaR, i.e. the 1-alpha quantile
		@views myLocalVaR = aRealisations[myNTradingCostRealisationsCVaREstimate]
		myVaRs[string("VaR_",myLocalAlpha)] = myLocalCVaR

		# computes the Mean-VaR trade-off
		myMeanVaRs[string("Mean-VaR_",myLocalAlpha)]  = (1 - aRiskAversion) * myTradingCostExpectedValue + aRiskAversion * myLocalVaR
	end

	# computes the Mean-Variance trade-off
	myTradingCostMeanVariance = myTradingCostExpectedValue + aRiskAversion * myTradingCostVariance

	# dictionary that saves the performance
	myPerformance = Dict()
	myPerformance["Expectation"]   = myTradingCostExpectedValue
	myPerformance["Variance"]      = myTradingCostVariance
	myPerformance["Mean-Variance"] = myTradingCostMeanVariance

	# adds CVaR, Mean-CVaR, VaR, Mean-VaR values to the dictionary that saves the performance
	myPerformance = merge(myPerformance,myCVaRs)
	myPerformance = merge(myPerformance,myMeanCVaRs)
	myPerformance = merge(myPerformance,myVaRs)
	myPerformance = merge(myPerformance,myMeanVaRs)

	# adds the details of the market for the performance hash
	myPerformance["NTradingCostRealisations"] = myNTradingCostRealisations

	return myPerformance
end

"""
#### Definition
```
get_optimal_trading_plan_performance(;
aMarketDetails::MarketDetails,
aTrader::Trader,
aSimulationParameters::SimulationParameters,
aNTradingCostRealisations::Int64=10^7,
aNTradingCostRealisationsPerIteration::Int64=10^4,
aFilePathWithOptimalStrategy::String=HelpFilesModule.get_default_output_file_path(
aMarketDetails=aMarketDetails,
aTrader=aTrader,
aSimulationParameters=aSimulationParameters,
aIncludePartialMarketDetailsHash=true,
aIncludeTradersHash=false,
aIncludeSimulationParametersHash=false,
aIncludeStrategiesHash=false,
aSpecificFolder="outputs/trading_plan/"
)
)
```

returns the performance of the optimal strategy that has been computed with the corresponding arguments, 
as well as the file where the performance is saved.
Note that here the ground truth distribution of the market are estimated, i.e. the distribution contained in `aMarketDetails`.

#### Argument
* `aMarketDetails::MarketDetails`: a structure containing all the details of the trading market.
* `aTrader::Trader`: a structure containing all the details of a trader.
* `aSimulationParameters::SimulationParameters`: a structure containing all the details of the parameters used to compute the optimal trading plan such as the method to use, the number of samples in the CVaR estimation.
* `aNTradingCostRealisations::Int=10^7`: the number of samples to generate.
* `aNSamplesPerIteration::Int=10^4,`: the maximum number of samples to generated on a processor at the same time.
"""
function get_optimal_trading_plan_performance(;
					      aMarketDetailsForRealisations::MarketDetails,
					      aTraderIndex::Int64,
					      aTraders::Array{Trader,1},
					      aStrategies::Array{Strategy,1},
					      aSimulationParameters::SimulationParameters,
					      aNTradingCostRealisations::Int64=10^7,
					      aAlphaLevelsToCompute::Array{Float64,1}=[get_alpha(aTraders[aTraderIndex]),0.1,0.05,0.025,0.01,0.005]
					      )


	# gets the information if the performance of a recursive method is needed
	myConsiderRecourse = get_consider_recourse(aSimulationParameters)
	myMethod = get_method(aSimulationParameters)
	myAlphaLevelsToCompute = unique(aAlphaLevelsToCompute)

	#############################################################################################
	## Precomputed check:                                                                      ##
	## one first checks that the performance has not already been computed given the arguments ##
	#############################################################################################

	myOutputPerformanceFilePath = nothing
	if myConsiderRecourse
		if !(in(myMethod,keys(ourFunctionsWithRecourse)))
			@error(
			       string(
				      "\nPerformanceModule 103:\n",
				      "Method ",
				      myMethod,
				      " is unknown to the methods easy to solve and for which the recursive evalution is easy.\n Use rather one of the following methods: ",
				      keys(ourFunctionsWithRecourse),
				      "."
				      )
			       )
		end

		# the optimal strategy must be recomputed for each trading period,
		# and the performance depends on the ground truth distributions of the random variables inherent to the market.
		# Hence the parameters `aIncludeTradersHash`, `aIncludePartialMarketDetailsHash` and `aIncludeSimulationParametersHash` must be set to `false` in order to have the hash of the full structures.
		myOutputPerformanceFilePath = get_default_output_file_path(
									   aMarketDetails                          = aMarketDetailsForRealisations,
									   aTraderIndex                            = aTraderIndex,
									   aTraders                                = aTraders,
									   aStrategies                             = aStrategies,
									   aSimulationParameters                   = aSimulationParameters,
									   aIncludePartialMarketDetailsHash        = false,
									   aIncludeSimulationParametersHash        = true,
									   aIncludeTradersHash                     = true,
									   aIncludeStrategiesHash                  = true,
									   aSpecificFileNameExtension              = string("_recursive_performance_n_realisations_",aNTradingCostRealisations),
									   aSpecificFolder                         = "outputs/performance/"
									   )
	else

		###########################################################################################
		## Loads the optimal strategy, and raise an error if it has not been computed previously ##
		###########################################################################################

		# the performance depends on the ground truth distributions of the random variables inherent to the market.
		# Hence the parameter `aIncludePartialMarketDetailsHash` must be set to `false` in order to have the hash of the full structure.
		myOutputPerformanceFilePath = get_default_output_file_path(
									   aMarketDetails                          = aMarketDetailsForRealisations,
									   aTraderIndex                            = aTraderIndex,
									   aTraders                                = aTraders,
									   aStrategies                             = aStrategies,
									   aSimulationParameters                   = aSimulationParameters,
									   aIncludePartialMarketDetailsHash        = false,
									   aIncludeSimulationParametersHash        = false,
									   aIncludeTradersHash                     = false,
									   aIncludeStrategiesHash                  = true,
									   aSpecificFileNameExtension              = string("_performance_n_realisations_",aNTradingCostRealisations),
									   aSpecificFolder                         = "outputs/performance/"
									   )
	end

	create_relevant_folders!(myOutputPerformanceFilePath)

	# initialisation of the variable with the performance dictionary
	myPerformance = nothing

	# boolean forcing the recomputation of the performance if needed, initialise the value to wheter an associated performance file exists or not
	myShouldRecomputePerformance = !(isfile(myOutputPerformanceFilePath))

	if get_recompute_performances(aSimulationParameters)
		# removes the file given by the path `myOutputRealisationsFilePath`.
		# force=true allows that a non-existing path is not treated as error. 
		rm(myOutputPerformanceFilePath, force=true) 
		myShouldRecomputePerformance = true
	end

	# checks if the all the performance results needed have already been computed
	if isfile(myOutputPerformanceFilePath) # checks if the file exists (the file has been deleted if you force the recomputation of the performance)
		myPerformance = HelpFilesModule.load_result(myOutputPerformanceFilePath) # reads the file
		for myLocalAlphaLevel in myAlphaLevelsToCompute # checks if the values has been computed for all the alpha levels desired
			if !(in(string("CVaR_",myLocalAlphaLevel),keys(myPerformance)))
				myShouldRecomputePerformance = true
			end

			if !(in(string("Mean-CVaR_",myLocalAlphaLevel),keys(myPerformance)))
				myShouldRecomputePerformance = true
			end

			if !(in(string("VaR_",myLocalAlphaLevel),keys(myPerformance)))
				myShouldRecomputePerformance = true
			end

			if !(in(string("Mean-VaR_",myLocalAlphaLevel),keys(myPerformance)))
				myShouldRecomputePerformance = true
			end
		end
	end

	if !(myShouldRecomputePerformance)
		@warn(
		      string(
			     "\nPerformanceModule 104:\n",
			     "The performance of the optimal strategy given the parameters passed as arguments has already been computed and will not be recomputed. If you desire to recompute them please delete file: \n",
			     myOutputPerformanceFilePath,
			     "."
			     )
		      )

		return myPerformance, myOutputPerformanceFilePath
	end

	############################################
	## Computation: trading cost realisations ##
	############################################

	# gets the desired number of realisations of the trading cost
	myTradingCostRealisations = get_optimal_plan_trading_cost_realisations(
									       aMarketDetailsForRealisations = aMarketDetailsForRealisations,
									       aTraderIndex                  = aTraderIndex,
									       aTraders                      = aTraders,
									       aStrategies                   = aStrategies,
									       aSimulationParameters         = aSimulationParameters,
									       aNTradingCostRealisations     = aNTradingCostRealisations,
									       )

	##############################
	## Computation: performance ##
	##############################

	myPerformance = get_performance_dict_given_realisation(
							       aRiskAversion         = get_risk_aversion(aTraders[aTraderIndex]),
							       aRealisations         = myTradingCostRealisations,
							       aAlphaLevelsToCompute = myAlphaLevelsToCompute
							       )

	# adds the remaining details
	myPerformance["Trader"]                       = get_dict_from_trader(aTraders[aTraderIndex])
	myPerformance["MarketDetailsForRealisations"] = get_dict_from_market_details(aMarketDetailsForRealisations)
	myPerformance["SimulationParameters"]         = get_dict_from_simulation_parameters(aSimulationParameters)

	# saves the results
	HelpFilesModule.save_result!(myOutputPerformanceFilePath,myPerformance)

	return myPerformance, myOutputPerformanceFilePath
end

end
