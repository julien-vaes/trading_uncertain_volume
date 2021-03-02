# NAME: optimal_strategy_nash_equilibria.jl
# AUTHOR: Julien Vaes
# DATE: June 17, 2020
# DESCRIPTION: compute a Nash equilibrium

module OptimalStrategyNEModule

###################
## Load Packages ##
###################

using Plots
using Statistics
using Random
using LinearAlgebra
using LaTeXStrings

############################
## Load Personal Packages ##
############################

using ..MarketDetailsModule
using ..TraderModule
using ..StrategyModule
using ..SimulationParametersModule
using ..TradingCostModule
using ..MeanVarianceModule
using ..MeanCVaRModule
using ..MeanVaRPrincipalScenariosModule
using ..HelpFilesModule
using ..PlotOptimalStrategyModule

######################
## Export functions ##
######################

export compute_best_response
export get_nash_equilibrium
export check_uniqueness_nash_equilibrium
export plot_best_responses_trajectories
export plot_best_responses_trajectories_iterations
export plot_convergence_objective_value_and_step_iterations

######################
## Module variables ##
######################

# the initial number of samples when searching the Nash equilibrium with an adaptive number of samples
ourInitialNSamplesToFindNashEquilibrium = 10^4

# correspondence of the methods name with the function associated.
ourFunctionsImplemented = Dict(
			       #= "Mean-Variance" => MeanVarianceModule.get_optimal_strategy_mean_variance, =#
			       "Mean-CVaR_Optim"     => MeanCVaRModule.get_optimal_strategy_mean_CVaR_Optim,
			       "Mean-CVaR_Gurobi"    => MeanCVaRModule.get_optimal_strategy_mean_CVaR_Gurobi,
			       #= "Mean-VaR"      => MeanVaRPrincipalScenariosModule.get_optimal_strategy_mean_VaR =#
			       )

######################
## Module functions ##
######################

function compute_best_response(;
			       aTraderIndex::Int64,
			       aTraders::Array{Trader,1},
			       aStrategies::Array{Strategy,1},
			       aSimulationParameters::SimulationParameters
			       )

	# gets the stack trace, useful to know if ones has to print or not the logs of this function
	myStacktrace = stacktrace()

	# gets the method to use to compute the optimal strategy
	myMethod = get_method(aSimulationParameters)

	# checks that the method exists
	if !(in(myMethod,keys(ourFunctionsImplemented)))
		@error(
		       string(
			      "\nOptimalStrategyNEModule 101:\n",
			      "Method ",
			      myMethod,
			      " is unknown.\n Use one of the following methods:  ",
			      keys(ourFunctionsImplemented),"."
			      )
		       )
		sleep(0.1) # time to be sure that the message is displayed in the correct place
	end

	# computes the optimal execution strategy
	myOptimalStrategy, myOptimalObjectiveValue, myOptimisationSuccessful, myOutputFilePath = ourFunctionsImplemented[myMethod](
																   aStrategies           = aStrategies,
																   aTraderIndex          = aTraderIndex,
																   aTraders              = aTraders,
																   aSimulationParameters = aSimulationParameters
																   )

	println(myOptimisationSuccessful)
	if !(myOptimisationSuccessful)
		@error(string(
			      "\nOptimalStrategyNEModule 104:\n",
			      "The best response has not been computed as the optimisation problem has not been sucessful, i.e. the function ", 
			      myMethod,
			      " returned an unseuccessful status."
			      )
		       )
		sleep(0.1) # time to be sure that the message is displayed in the correct place
	end

	# prints the results if required
	println_logs(string("+++ Results: λ = ",get_risk_aversion(aTraders[aTraderIndex])," +++"),aSimulationParameters,myStacktrace)
	if get_optimise_trading_plan(aSimulationParameters)
		println_logs(string("Best response trading plan: ",get_trading_plan(myOptimalStrategy)),aSimulationParameters,myStacktrace)
	end
	if get_optimise_redistribution_matrix(aSimulationParameters)
		println_logs(string("Best response redistribution matrix: ",get_redistribution_matrix(myOptimalStrategy)),aSimulationParameters,myStacktrace)
	end

	return myOptimalStrategy, myOptimalObjectiveValue, myOptimisationSuccessful, myOutputFilePath
end

function compute_best_response_all_traders(;
					   aTraders::Array{Trader,1},
					   aStrategies::Array{Strategy,1},
					   aSimulationParameters::SimulationParameters,
					   aStepSizeBestResponse::Float64 
					   )

	# gets the stack trace, useful to know if ones has to print or not the logs of this function
	myStacktrace = stacktrace()

	# the method to find the Nash equilibrium
	myMethodFindNE = get_ne_mean_CVaR_method_find_ne(aSimulationParameters)

	# computes the number of traders
	myNTraders = size(aTraders,1)

	# initialises the output
	myStrategies                     = copy(aStrategies)
	myObjFunValuesBeforeBestResponse = Array{Float64,1}(undef,size(aTraders,1))
	myObjFunValuesAfterBestResponse  = Array{Float64,1}(undef,size(aTraders,1))

	# gets the method to use to compute the optimal strategy
	myMethod = get_method(aSimulationParameters)

	# computes the best response of each trader (the last ones consider already the updated strategies of the first ones (similar to Gauss Seidel) )
	for p in eachindex(aTraders)

		# initialises the output
		myBestResponseStrategy = nothing
		myTraderObjFunValueBeforeBestResponse, myTraderObjFunValueAfterBestResponse = nothing, nothing

		if myMethodFindNE == "Seidel"

			# computes the objective function of trader p given the trading strategy of all the other traders and before optimising its own trading strategy
			myTraderStrategy, myTraderObjFunValueBeforeBestResponse, myOptimisationSuccessfulBefore, myOutputFilePathBefore = ourFunctionsImplemented[myMethod](
																					    aStrategies           = myStrategies,
																					    aTraderIndex          = p,
																					    aTraders              = aTraders,
																					    aSimulationParameters = get_new_simulation_parameters(
																												  aSimulationParameters,
																												  aOptimiseTradingPlan=false,
																												  aOptimiseRedistributionMatrix=false
																												  )
																					    )


			# computes the best response of each trader
			myBestResponseStrategy, myTraderObjFunValueAfterBestResponse, myOptimisationSuccessfulAfter, myOutputFilePathAfter = compute_best_response(
																				   aTraderIndex          = p,
																				   aTraders              = aTraders,
																				   aStrategies           = myStrategies,
																				   aSimulationParameters = aSimulationParameters
																				   )
		elseif myMethodFindNE == "Jacobi"

			# computes the objective function of trader p given the trading strategy of all the other traders and before optimising its own trading strategy
			myTraderStrategy, myTraderObjFunValueBeforeBestResponse, myOptimisationSuccessfulBefore, myOutputFilePathBefore = ourFunctionsImplemented[myMethod](
																					    aStrategies           = aStrategies,
																					    aTraderIndex          = p,
																					    aTraders              = aTraders,
																					    aSimulationParameters = get_new_simulation_parameters(
																												  aSimulationParameters,
																												  aOptimiseTradingPlan=false,
																												  aOptimiseRedistributionMatrix=false
																												  )
																					    )

			# computes the best response of each trader
			myBestResponseStrategy, myTraderObjFunValueAfterBestResponse, myOptimisationSuccessfulAfter, myOutputFilePathAfter = compute_best_response(
																				   aTraderIndex          = p,
																				   aTraders              = aTraders,
																				   aStrategies           = aStrategies,
																				   aSimulationParameters = aSimulationParameters
																				   )
		else
			@error(string(
				      "\nOptimalStrategyNEModule 102:\n",
				      "Method ",
				      myMethodFindNE,
				      " is unknown."
				      )
			       )
			sleep(0.1) # time to be sure that the message is displayed in the correct place
		end

		# computes the new trading plan as a convex combination of the old one and the best response
		# "Old" --> *---------x---*  <-- Best response (and "x")
		myOldTradingPlan = get_trading_plan(myStrategies[p])
		myBestResponseTradingPlan = get_trading_plan(myBestResponseStrategy)
		myNewTradingPlan = (1-aStepSizeBestResponse)*myOldTradingPlan + aStepSizeBestResponse*myBestResponseTradingPlan
		myNewStrategy = get_new_strategy(myStrategies[p],aTradingPlan=myNewTradingPlan)

		myStrategies[p]                     = myNewStrategy
		myObjFunValuesBeforeBestResponse[p] = myTraderObjFunValueBeforeBestResponse
		myObjFunValuesAfterBestResponse[p]  = myTraderObjFunValueAfterBestResponse

		print_logs(string("=> Trader ",p,": "),aSimulationParameters,myStacktrace)
		println_logs(string(get_trading_plan(myStrategies[p])),aSimulationParameters,myStacktrace)
	end

	return myStrategies, myObjFunValuesBeforeBestResponse, myObjFunValuesAfterBestResponse
end

function get_nash_equilibrium(;
			      aInitialStrategies::Array{Strategy,1},
			      aTraders::Array{Trader,1},
			      aSimulationParameters::SimulationParameters,
			      aCurrentNSamples::Int64 = ourInitialNSamplesToFindNashEquilibrium,
			      aBestResponseTrajectories = fill(Strategy[],size(aTraders,1)),
			      aObjValueBeforeBestResponseTrajectories = fill(Float64[],size(aTraders,1)),
			      aObjValueAfterBestResponseTrajectories = fill(Float64[],size(aTraders,1))
			      )

	# gets the stack trace, useful to know if ones has to print or not the logs of this function
	myStacktrace = stacktrace()

	# to have a faster computation, we might want to start to find the NE with a lower number of realisations in the estimate of CVaR and then increase this the number of samples
	# in that way we have already a good initial guess of the NE and as a consequence we hope to speed up the computation of the NE
	myNSamples     = get_CVaR_optimisation_n_samples(aSimulationParameters)
	myNextNSamples = -1
	if get_method(aSimulationParameters) == "Mean-CVaR" && get_ne_mean_CVaR_adaptive_number_of_samples_to_find_ne(aSimulationParameters)

		# gets the number of samples to find to NE
		myNSamples = aCurrentNSamples

		# computes the number of samples to find to NE in next iteration
		myNextNSamples = min(10*aCurrentNSamples,get_CVaR_optimisation_n_samples(aSimulationParameters))
	end

	# prints the number of samples used to find the Nash equilibrium
	println_logs(string("\n*** Search of a Nash equilibrium when a number of ",myNSamples," samples are generated to evaluate CVaR ***\n"),aSimulationParameters,myStacktrace)

	# adapts `aSimulationParameters` to the number of samples
	mySimulationParameters = get_new_simulation_parameters(aSimulationParameters,
							       aParametersCVaROptimisation=Dict("NSamples" => myNSamples)
							       )

	# checks that the initial step size is in [0,1]
	myStepSizeBestResponse = get_ne_mean_CVaR_step_size_best_response(mySimulationParameters)
	if myStepSizeBestResponse <= 0.0 || myStepSizeBestResponse > 1.0
		@error(
		       string(
			      "\nOptimalStrategyNEModule 103:\n",
			      "The argument `mySimulationParameters.theParametersNEMeanCVaR[\"StepSizeBestResponse\"]` must be in ]0,1], currently the value is ",
			      myStepSizeBestResponse,"."
			      )
		       )
		sleep(0.1) # time to be sure that the message is displayed in the correct place
	end

	# gets the stack trace, useful to know if ones has to print or not the logs of this function
	myStacktrace = stacktrace()

	# gets the number of traders and trading plan
	myNTraders = size(aTraders,1)
	myNTradingPeriods = MarketDetailsModule.get_n_trading_periods(get_market_details_belief(aTraders[1]))

	# initialises the variable containing the strategies of the Nash equilibrium
	myStrategies    = copy(aInitialStrategies)
	myStrategiesNew = copy(aInitialStrategies)

	# initialises the variable containing the history of the best response strategies
	myBestResponseTrajectories = [Array{Strategy,1}(undef,0) for i=1:myNTraders]
	for p in eachindex(aTraders)
		# adds the initial strategy to the trajectory
		push!(myBestResponseTrajectories[p],aInitialStrategies[p])
	end

	# initialises the variable containing the history of the objective value before computing the best response
	myObjValueBeforeBestResponseTrajectories = [Array{Float64,1}(undef,0) for i=1:myNTraders]

	# initialises the variable containing the history of the objective value after computing the best response
	myObjValueAfterBestResponseTrajectories = [Array{Float64,1}(undef,0) for i=1:myNTraders]

	# print initial trading plans
	println_logs(string("++++++ Initial trading plans ++++++\n"),mySimulationParameters,myStacktrace)
	for p in eachindex(aTraders)
		print_logs(string("=> Trader ",p,": "),mySimulationParameters,myStacktrace)
		println_logs(string(get_trading_plan(myStrategies[p])),mySimulationParameters,myStacktrace)
	end
	println_logs("",mySimulationParameters,myStacktrace)

	# variable to check the convergence
	myHasConverged = false
	myIter = 0
	myOptimalTradingPlanTradersOld = zeros(myNTradingPeriods,myNTraders)
	myOptimalTradingPlanTradersNew = zeros(myNTradingPeriods,myNTraders)
	myStep2NormOld = 0
	myStep2NormNew = 0
	myNSuccessiveStepsWithSimilarSize    = 0
	myNSuccessiveStepsWithNotSimilarSize = 0
	myNΔ2Threshold  = 5
	myNSuccessiveStepsWithRequiredStepSize = 0
	while !(myHasConverged) && (myIter < get_ne_mean_CVaR_max_number_turns_best_response(mySimulationParameters))

		# if the step is adapted during the search
		if get_ne_mean_CVaR_adaptive_step_size_best_response(mySimulationParameters)
			if myNSuccessiveStepsWithSimilarSize == myNΔ2Threshold
				myStepSizeBestResponse = 0.5*myStepSizeBestResponse
				myNSuccessiveStepsWithSimilarSize = 0
				myNSuccessiveStepsWithNotSimilarSize = 0
			end
		end

		myIter += 1
		myStrategies = myStrategiesNew
		myStep2NormOld = myStep2NormNew
		myOptimalTradingPlanTradersOld = copy(myOptimalTradingPlanTradersNew)

		println_logs(string("++++++ Iteration ",myIter," of best responses ++++++\n"),mySimulationParameters,myStacktrace)
		println_logs(string("Step size = ",myStepSizeBestResponse,"\n"),mySimulationParameters,myStacktrace)

		# computes the best response of all the traders
		myStrategiesNew, myObjFunValuesBeforeBestResponse, myObjFunValuesAfterBestResponse = compute_best_response_all_traders(
																       aTraders              = aTraders,
																       aStrategies           = myStrategies,
																       aSimulationParameters = mySimulationParameters,
																       aStepSizeBestResponse = myStepSizeBestResponse
																       )

		# adds the new strategies in the history of best responses
		for p in eachindex(aTraders)
			push!(myBestResponseTrajectories[p],myStrategiesNew[p])
			push!(myObjValueBeforeBestResponseTrajectories[p],myObjFunValuesBeforeBestResponse[p])
			push!(myObjValueAfterBestResponseTrajectories[p],myObjFunValuesAfterBestResponse[p])
			myOptimalTradingPlanTradersNew[:,p] = get_trading_plan(myStrategiesNew[p])
		end

		# computes the step's infinity norm, i.e. the biggest step of one variable no matter the trader
		myStepInfinityNorm = norm(myOptimalTradingPlanTradersNew-myOptimalTradingPlanTradersOld,Inf)

		# computes the step's 2-norm, i.e. the biggest step of one variable no matter the trader
		myStep2NormNew = norm(myOptimalTradingPlanTradersNew-myOptimalTradingPlanTradersOld,2)

		# computes norm 2 of the step of the best responses
		myStep2NormNew = norm(myOptimalTradingPlanTradersNew-myOptimalTradingPlanTradersOld,2)

		if 1-myStep2NormNew/myStep2NormOld < 0.0
			myNSuccessiveStepsWithSimilarSize += 1
		else
			myNSuccessiveStepsWithNotSimilarSize += 1
		end

		println_logs(string("\n--> ||Step||_∞ = ",myStepInfinityNorm),mySimulationParameters,myStacktrace)
		println_logs(string("--> ||Step||_2 = ",norm(myOptimalTradingPlanTradersNew-myOptimalTradingPlanTradersOld,2),"\n"),mySimulationParameters,myStacktrace)

		# checks if has converged
		if myStepInfinityNorm < get_ne_mean_CVaR_convergence_tol_ne(mySimulationParameters)
			myNSuccessiveStepsWithRequiredStepSize += 1

			if myNSuccessiveStepsWithRequiredStepSize >= 3 
				@info(
				      string(
					     "\nOptimalStrategyNEModule 104:\n",
					     "NE found after ",myIter," iterations of best responses. ",
					     "The ∞-norm of the step amounts ",
					     myStepInfinityNorm,
					     "."
					     )
				      )
				myHasConverged = true
				sleep(0.1) # time to be sure that the message is displayed in the correct place
			end
		else
			myNSuccessiveStepsWithRequiredStepSize = 0
		end

		# checks if the limit of iterations has been reached
		if myIter == get_ne_mean_CVaR_max_number_turns_best_response(mySimulationParameters)
			@warn(string(
				     "\nOptimalStrategyNEModule 105:\n",
				     "The maximum number of turns of best response has been reached; the search is interrupted."
				     )
			      )
			sleep(0.1) # time to be sure that the message is displayed in the correct place
		end
	end

	# merges the trajectories
	for p in 1:myNTraders
		myBestResponseTrajectories[p]               = vcat(aBestResponseTrajectories[p],myBestResponseTrajectories[p])
		myObjValueBeforeBestResponseTrajectories[p] = vcat(aObjValueBeforeBestResponseTrajectories[p],myObjValueBeforeBestResponseTrajectories[p])
		myObjValueAfterBestResponseTrajectories[p]  = vcat(aObjValueAfterBestResponseTrajectories[p],myObjValueAfterBestResponseTrajectories[p])
	end

	# if the current number of samples to compute the NE is not equal to the one in the initial one in the simulation parameter structure
	if myNSamples < get_CVaR_optimisation_n_samples(aSimulationParameters)

		return get_nash_equilibrium(
					    aInitialStrategies                      = myStrategies, # we have now a better guess of where the Nash equilibrium is
					    aTraders                                = aTraders,
					    aSimulationParameters                   = aSimulationParameters,
					    aCurrentNSamples                        = myNextNSamples,
					    aBestResponseTrajectories               = myBestResponseTrajectories,
					    aObjValueBeforeBestResponseTrajectories = myObjValueBeforeBestResponseTrajectories,
					    aObjValueAfterBestResponseTrajectories  = myObjValueAfterBestResponseTrajectories
					    )
	end

	return myStrategies, myBestResponseTrajectories, myObjValueBeforeBestResponseTrajectories, myObjValueAfterBestResponseTrajectories, myHasConverged
end

function check_uniqueness_nash_equilibrium(;
					   aTraders::Array{Trader,1},
					   aSimulationParameters::SimulationParameters,
					   aNDifferentStartingPointsTried::Int64=10,
					   aToleranceInDifferenceNE::Float64 = 10.0^-2
					   )

	# gets the stack trace, useful to know if ones has to print or not the logs of this function
	myStacktrace = stacktrace()

	myNTraders = size(aTraders,1)
	myFirstTraderMarketDetailsBelief = get_market_details_belief(aTraders[1])
	myNTradingPeriods = get_n_trading_periods(myFirstTraderMarketDetailsBelief)

	# initialisation of the array that will contain the optimal strategies of the traders
	myStrategiesNE = nothing
	myNEFoundIsUnique = true

	for i in 1:aNDifferentStartingPointsTried

		# continue to check if one has not yet found a counter example
		if myNEFoundIsUnique
			println_logs(string("\n### Computation of NE #",i," ###\n"),aSimulationParameters,myStacktrace)

			Random.seed!(i)

			# random initial strategies
			myInitialStrategies = Array{Strategy}(undef,myNTraders)
			for p in eachindex(aTraders)
				myRandomTradingPlan = rand(myNTradingPeriods)
				myRandomTradingPlan[end] = 1-sum(myRandomTradingPlan[1:end-1])
				myRandomStrategy = get_strategy(
								aNTradingPeriods = myNTradingPeriods,
								aTradingPlan     = myRandomTradingPlan
								)
				myInitialStrategies[p] = myRandomStrategy
			end

			# searches a Nash equilibrium
			myStrategiesNELocal, myBestResponseTrajectoriesLocal, myObjValueBeforeBestResponseTrajectoriesLocal, myObjValueAfterBestResponseTrajectoriesLocal, myHasConvergedLocal = get_nash_equilibrium(
																										      aInitialStrategies            = myInitialStrategies,
																										      aTraders                      = aTraders,
																										      aSimulationParameters         = aSimulationParameters
																										      )

			# checks if the NE returned is really a NE where the best response has converged to or if it is just one returned because the maximum number of iteration has been reached
			if !(myHasConvergedLocal)
				myNEFoundIsUnique = false
				@warn(
				      string(
					     "\nOptimalStrategyNEModule 106:\n",
					     "The best response scheme has not converged in iteration ",i,", one can thus not conclude if it is converging to the same NE."
					     )
				      )
				sleep(0.1) # time to be sure that the message is displayed in the correct place
			end

			####################################
			# checks that the same NE is found #
			####################################

			if i == 1 # saves the first NE found
				myStrategiesNE = myStrategiesNELocal
			else # checks that the NE found is the same as the first one found
				for p in eachindex(aTraders)
					myTradingPlanNE      = get_trading_plan(myStrategiesNE[p])
					myTradingPlanNELocal = get_trading_plan(myStrategiesNELocal[p])
					for j in eachindex(myTradingPlanNE)
						if !isapprox(myTradingPlanNE[j]-myTradingPlanNELocal[j], 0.0, atol=aToleranceInDifferenceNE) && myNEFoundIsUnique
							myNEFoundIsUnique = false
							@warn(
							      string(
								     "\nOptimalStrategyNEModule 107:\n",
								     "The trading plan of trader ",p," is not always the same in the NE. For example the trading plans are:",
								     "\nIn the NE 1:\n\t",myTradingPlanNE,
								     "\nIn the NE ",i,":\n\t",myTradingPlanNELocal,
								     "\nThe "
								     )
							      )
						end
					end
				end
			end
		end
	end

	return myNEFoundIsUnique, myStrategiesNE
end

function plot_best_responses_trajectories(;
					  aBestResponseTrajectories::Array{Array{Strategy,1},1},
					  aTradersToPlot::Array{Int64,1} = [1,2], 
					  aTradingDecisionsToPlot::Array{Int64,1} = [1,2]
					  )

	# gets the details on the number of traders, trading periods, and iterations of best responses
	myNTraders        = size(aBestResponseTrajectories,1)
	myNTradersToPlot  = size(aTradersToPlot,1)
	myNElements       = size(aBestResponseTrajectories[1],1)
	myNTradingPeriods = size(get_trading_plan(aBestResponseTrajectories[1][1]),1)

	# initialises the array with the trajectory of each trading decision for each trader and each trading period
	myBestResponseTrajectories = Array{Float64}(undef,(myNTradersToPlot,myNTradingPeriods,myNElements))

	# fills the array of the trajectories
	for myTraderIndex in 1:myNTradersToPlot
		for myIterationIndex in 1:myNElements
			for myTradingPeriodIndex in 1:myNTradingPeriods
				myBestResponseTrajectories[myTraderIndex,myTradingPeriodIndex,myIterationIndex] = get_trading_plan(aBestResponseTrajectories[aTradersToPlot[myTraderIndex]][myIterationIndex])[myTradingPeriodIndex]
			end
		end
	end

	# gets the decision variables trajectory bounds
	myRangeDecision1 = [+Inf,-Inf]
	myRangeDecision2 = [+Inf,-Inf]

	for myStrategiesTrajectory in aBestResponseTrajectories
		for myStrategy in myStrategiesTrajectory
			myLocalTradingPlan = get_trading_plan(myStrategy)
			myDecision1 = myLocalTradingPlan[theTradingDecisionsToPlot[1]]
			myDecision2 = myLocalTradingPlan[theTradingDecisionsToPlot[2]]

			myRangeDecision1[1] = myDecision1 < myRangeDecision1[1] ? myDecision1 : myRangeDecision1[1]
			myRangeDecision1[2] = myDecision1 > myRangeDecision1[2] ? myDecision1 : myRangeDecision1[2]

			myRangeDecision2[1] = myDecision2 < myRangeDecision2[1] ? myDecision2 : myRangeDecision2[1]
			myRangeDecision2[2] = myDecision2 > myRangeDecision2[2] ? myDecision2 : myRangeDecision2[2]
		end
	end

	myRangeDecision1[1] = myRangeDecision1[1] > 0 ? 0.9*myRangeDecision1[1] : 1.1*myRangeDecision1[1]
	myRangeDecision1[2] = myRangeDecision1[2] > 0 ? 1.1*myRangeDecision1[2] : 0.9*myRangeDecision1[2]

	myRangeDecision2[1] = myRangeDecision2[1] > 0 ? 0.9*myRangeDecision2[1] : 1.1*myRangeDecision2[1]
	myRangeDecision2[2] = myRangeDecision2[2] > 0 ? 1.1*myRangeDecision2[2] : 0.9*myRangeDecision2[2]

	# initialises the plot
	myPlot = Plots.plot(
			    formatter      = :latex,
			    size           = (900,500),
			    title          = string("Trajectory of the best responses of the traders"), 
			    title_location = :left, 
			    xtickfont      = font(8),
			    ytickfont      = font(8),
			    legendfont     = font(12),
			    xaxis          = (string("\$y_{",aTradingDecisionsToPlot[1],"}\$"), font(14)),
			    yaxis          = (string("\$y_{",aTradingDecisionsToPlot[2],"}\$"), font(14)),
			    xlims          = myRangeDecision1,
			    ylims          = myRangeDecision2,
			    aspect_ratio   = 1
			    )

	# adds the trajectories to the plot
	for myTraderIndex in 1:myNTradersToPlot

		# get the line details in order to have a plot with distinguishable lines
		myColour, myLineStyle, myMarker, myMarkerSize = PlotOptimalStrategyModule.get_line_details(myTraderIndex)

		Plots.plot!(
			    myPlot,
			    myBestResponseTrajectories[myTraderIndex,aTradingDecisionsToPlot[1],:],
			    myBestResponseTrajectories[myTraderIndex,aTradingDecisionsToPlot[2],:],
			    label = string("Trader ",aTradersToPlot[myTraderIndex]),
			    linewidth = 3,
			    linestyle   = myLineStyle,
			    markershape = myMarker:circle,
			    markersize = myMarkerSize
			    )
	end

	return myPlot
end

function plot_best_responses_trajectories_iterations(;
						     aBestResponseTrajectories::Array{Array{Strategy,1},1},
						     aObjValueBeforeBestResponseTrajectories::Array{Array{Float64,1},1},
						     aObjValueAfterBestResponseTrajectories::Array{Array{Float64,1},1},
						     aTradersToPlot::Array{Int64,1} = collect(range(1,stop=size(aBestResponseTrajectories,1))),
						     aTradingDecisionsToPlot::Array{Int64,1} = collect(range(1,stop=size(get_trading_plan(aBestResponseTrajectories[1][1]),1))),
						     aPlotStrategies::Bool = false,
						     aPlotObjectiveValueTrajectory::Bool = false,
						     aInitialIteration::Int64=1,
						     aFinalIteration::Int64=size(aObjValueBeforeBestResponseTrajectories[1],1),
						     )

	# gets the details on the number of traders, trading periods, and iterations of best responses
	myNTraders          = size(aBestResponseTrajectories,1)
	myNTradersToPlot    = size(aTradersToPlot,1)
	myNTradingPeriods   = size(get_trading_plan(aBestResponseTrajectories[1][1]),1)
	myNIterationsToPlot = aFinalIteration-aInitialIteration+1

	# initialises the arrays with the trajectory of each trading decision for each trader,
	# adds 1 as plot initial strategy or the strategy before `aInitialIteration` to see the evolution due to the best response
	myBestResponseTrajectories = Array{Float64}(undef,(myNTradersToPlot,myNTradingPeriods,myNIterationsToPlot+1))

	# initialises the arrays containing the trajectories of the objective value of before and after the best response
	myObjValueBeforeBestResponseTrajectories = Array{Float64}(undef,(myNTradersToPlot,myNIterationsToPlot))
	myObjValueAfterBestResponseTrajectories  = Array{Float64}(undef,(myNTradersToPlot,myNIterationsToPlot))

	# fills the array of the trajectories
	for myTraderIndex in 1:myNTradersToPlot
		for myIterationToPlot in aInitialIteration:aFinalIteration+1
			myIterationIndex = myIterationToPlot-aInitialIteration+1
			for myTradingPeriodIndex in 1:myNTradingPeriods
				myBestResponseTrajectories[myTraderIndex,myTradingPeriodIndex,myIterationIndex] = get_trading_plan(aBestResponseTrajectories[aTradersToPlot[myTraderIndex]][myIterationToPlot])[myTradingPeriodIndex]
			end

			# the +1 comes from the fact that for the trajectories we have one more element to fill in but not for the objective values
			if myIterationToPlot < aFinalIteration+1
				myObjValueBeforeBestResponseTrajectories[myTraderIndex,myIterationIndex] = aObjValueBeforeBestResponseTrajectories[myTraderIndex][myIterationToPlot]
				myObjValueAfterBestResponseTrajectories[myTraderIndex,myIterationIndex]  = aObjValueAfterBestResponseTrajectories[myTraderIndex][myIterationToPlot]
			end
		end
	end

	# initialises an array with all the plots
	myPlots = []
	myNPlotsPerTraders = (aPlotStrategies && aPlotObjectiveValueTrajectory) ? 2 : 1
	myPlotHeightsPerTrader = (aPlotStrategies && aPlotObjectiveValueTrajectory) ? [0.6, 0.4] : [1.0]
	myTraderPlotVerticalSize = (aPlotStrategies && aPlotObjectiveValueTrajectory) ? 500 : 300
	myPlotHeightsMerged = []
	for p = 1:myNTraders
		myPlotHeightsMerged = vcat(myPlotHeightsMerged,myPlotHeightsPerTrader./myNTraders)
	end

	# initialises the merged plot
	myPlotMerged = Plots.plot(
				  formatter      = :latex,
				  layout         = grid(myNPlotsPerTraders*myNTraders, 1, heights=myPlotHeightsMerged),
				  size           = (900,myNTraders*myTraderPlotVerticalSize),
				  title_location = :center, 
				  legendfont     = font(10),
				  xlabel         = string("Iteration of best responses"),
				  xaxisfont      = font(10),
				  yaxisfont      = font(10),
				  #= xticks = aInitialIteration:1:aFinalIteration, =#
				  xtickfont      = font(8),
				  ytickfont      = font(8),
				  )


	for p in eachindex(aTradersToPlot)
		# initialises the plot
		myPlot = nothing
		myPlot = Plots.plot(
				    formatter      = :latex,
				    layout         = grid(myNPlotsPerTraders, 1, heights=myPlotHeightsPerTrader),
				    size           = (900,myTraderPlotVerticalSize),
				    title_location = :center,
				    legendfont     = font(10),
				    xlabel         = "Iteration of best responses",
				    xaxisfont      = font(14),
				    yaxisfont      = font(14),
				    #= xticks         = aInitialIteration:1:aFinalIteration, =#
				    xtickfont      = font(8),
				    ytickfont      = font(8),
				    )

		# adds the title at the right location
		Plots.plot!(
			    myPlot[1],
			    title_location = :center,
			    title = string("Trader ",aTradersToPlot[p]),
			    )

		# puts the title trader at the good place
		Plots.plot!(
			    myPlotMerged[(p-1)*myNPlotsPerTraders+1],
			    title_location = :center,
			    title          = string("\nTrader ",p),
			    )

		# plots the evolution of the strategy with regards to the iterations
		if aPlotStrategies

			Plots.plot!(
				    myPlot[1],
				    legend      = :outerright,
				    legendtitle = "Decisions",
				    )

			Plots.plot!(
				    myPlotMerged[(p-1)*myNPlotsPerTraders+1],
				    legend      = :outerright,
				    legendtitle = "Decisions",
				    )

			# if has to plot the objective value trajectory, removes the label of the 'x' axis
			if aPlotObjectiveValueTrajectory
				Plots.plot!(
					    myPlot[1],
					    xlabel = ""
					    )

				Plots.plot!(
					    myPlotMerged[(p-1)*myNPlotsPerTraders+1],
					    xlabel = "",
					    )
			end

			# removes the label of the 'x' axis if not last trader
			if p != myNTraders
				Plots.plot!(
					    myPlotMerged[(p-1)*myNPlotsPerTraders+1],
					    legend = false,
					    xlabel = ""
					    )
			end

			# adds the trajectories of the proportion decisions to the plot
			for myTradingDecisionIndex in eachindex(aTradingDecisionsToPlot)

				# get the line details in order to have a plot with distinguishable lines
				myColour, myLineStyle, myMarker, myMarkerSize = PlotOptimalStrategyModule.get_line_details(myTradingDecisionIndex)

				Plots.plot!(
					    myPlot[1],
					    range(aInitialIteration-1, stop=aFinalIteration),
					    myBestResponseTrajectories[p,aTradingDecisionsToPlot[myTradingDecisionIndex],:],
					    label       = string("\$y_{",aTradingDecisionsToPlot[myTradingDecisionIndex],"}\$"),
					    linewidth   = 3,
					    linestyle   = myLineStyle,
					    color       = myColour,
					    markershape = myMarker,
					    markersize  = myMarkerSize,
					    xlims       = [aInitialIteration-1-0.1,aFinalIteration+0.1],
					    legendtitle = "Decisions" 
					    )

				# adds to the merged plot
				Plots.plot!(
					    myPlotMerged[(p-1)*myNPlotsPerTraders+1],
					    range(aInitialIteration-1,stop=aFinalIteration),
					    myBestResponseTrajectories[p,aTradingDecisionsToPlot[myTradingDecisionIndex],:],
					    label       = string("\$y_{",aTradingDecisionsToPlot[myTradingDecisionIndex],"}\$"),
					    linewidth   = 3,
					    linestyle   = myLineStyle,
					    color       = myColour,
					    markershape = myMarker,
					    markersize  = myMarkerSize,
					    xlims       = [aInitialIteration-1-0.1,aFinalIteration+0.1],
					    )
			end
		end

		# plots the evolution of the objective function with regards to the iterations
		if aPlotObjectiveValueTrajectory

			Plots.plot!(
				    myPlot[myNPlotsPerTraders],
				    range(aInitialIteration,stop=aFinalIteration),
				    myObjValueBeforeBestResponseTrajectories[p,:],
				    linewidth   = 3,
				    linestyle   = :dot,
				    color       = :red,
				    markershape = :utriangle,
				    markersize  = 6,
				    ylabel       = "Objective value",
				    label       = "Before best response",
				    )

			# adjust the `xlims`
			if aPlotStrategies
				Plots.plot!(myPlot[myNPlotsPerTraders], xlims = [aInitialIteration-1-0.1,aFinalIteration+0.1])
				Plots.plot!(myPlotMerged[p*myNPlotsPerTraders], xlims = [aInitialIteration-1-0.1,aFinalIteration+0.1])
			else
				Plots.plot!( myPlot[myNPlotsPerTraders], xlims = [aInitialIteration-0.1,aFinalIteration+0.1])
				Plots.plot!(myPlotMerged[p*myNPlotsPerTraders], xlims = [aInitialIteration-0.1,aFinalIteration+0.1])
			end

			# adds the trajectory of the objective function after that the trader computes the best response
			Plots.plot!(
				    myPlot[myNPlotsPerTraders],
				    range(aInitialIteration,stop=aFinalIteration),
				    myObjValueAfterBestResponseTrajectories[p,:],
				    linewidth   = 3,
				    linestyle   = :solid,
				    color       = :black,
				    markershape = :circle,
				    markersize  = 6,
				    ylabel       = "Objective value",
				    label       = "After best response",
				    )

			# adds the trajectory of the objective function before that the trader computes the best response
			Plots.plot!(
				    myPlotMerged[p*myNPlotsPerTraders],
				    range(aInitialIteration,stop=aFinalIteration),
				    myObjValueBeforeBestResponseTrajectories[p,:],
				    linewidth   = 3,
				    linestyle   = :dot,
				    color       = :red,
				    markershape = :utriangle,
				    markersize  = 6,
				    ylabel       = "Objective value",
				    label       = "Before best response",
				    )

			# adds the trajectory of the objective function after that the trader computes the best response
			Plots.plot!(
				    myPlotMerged[p*myNPlotsPerTraders],
				    range(aInitialIteration,stop=aFinalIteration),
				    myObjValueAfterBestResponseTrajectories[p,:],
				    linewidth   = 3,
				    linestyle   = :solid,
				    color       = :black,
				    markershape = :circle,
				    markersize  = 6,
				    ylabel       = "Objective value",
				    label       = "After best response",
				    )

			if !(aPlotStrategies)
				# adds to the merged plot
				Plots.plot!(
					    myPlotMerged[p*myNPlotsPerTraders],
					    title = string("\nTrader ",p),
					    )
			end

			if p != myNTraders
				# adds to the merged plot
				Plots.plot!(
					    myPlotMerged[p*myNPlotsPerTraders],
					    xlabel = "",
					    legend = false
					    )
			end
		end

		# adds the plot to the vector of plots
		push!(myPlots,myPlot)
	end

	return myPlots, myPlotMerged
end

function plot_convergence_objective_value_and_step_iterations(;
							     aBestResponseTrajectories::Array{Array{Strategy,1},1},
							     aObjValueBeforeBestResponseTrajectories::Array{Array{Float64,1},1},
							     aObjValueAfterBestResponseTrajectories::Array{Array{Float64,1},1},
							     aTradersToPlot::Array{Int64,1} = collect(range(1,stop=size(aBestResponseTrajectories,1))),
							     aTradingDecisionsToPlot::Array{Int64,1} = collect(range(1,stop=size(get_trading_plan(aBestResponseTrajectories[1][1]),1))),
							     aInitialIteration::Int64=1,
							     aFinalIteration::Int64=size(aObjValueBeforeBestResponseTrajectories[1],1),
							     )

	# gets the details on the number of traders, trading periods, and iterations of best responses
	myNTraders          = size(aBestResponseTrajectories,1)
	myNTradersToPlot    = size(aTradersToPlot,1)
	myNTradingPeriods   = size(get_trading_plan(aBestResponseTrajectories[1][1]),1)
	myNIterationsToPlot = aFinalIteration-aInitialIteration+1

	# initialises the arrays with the trajectory of each trading decision for each trader,
	# adds 1 as plot initial strategy or the strategy before `aInitialIteration` to see the evolution due to the best response
	myBestResponseTrajectories = Array{Float64}(undef,(myNTradersToPlot,myNTradingPeriods,myNIterationsToPlot+1))

	# initialises the arrays containing the trajectories of the objective value of before and after the best response
	myObjValueBeforeBestResponseTrajectories = Array{Float64}(undef,(myNTradersToPlot,myNIterationsToPlot))
	myObjValueAfterBestResponseTrajectories  = Array{Float64}(undef,(myNTradersToPlot,myNIterationsToPlot))

	# fills the array of the trajectories
	for myTraderIndex in 1:myNTradersToPlot
		for myIterationToPlot in aInitialIteration:aFinalIteration+1
			myIterationIndex = myIterationToPlot-aInitialIteration+1
			for myTradingPeriodIndex in 1:myNTradingPeriods
				myBestResponseTrajectories[myTraderIndex,myTradingPeriodIndex,myIterationIndex] = get_trading_plan(aBestResponseTrajectories[aTradersToPlot[myTraderIndex]][myIterationToPlot])[myTradingPeriodIndex]
			end

			# the +1 comes from the fact that for the trajectories we have one more element to fill in but not for the objective values
			if myIterationToPlot < aFinalIteration+1
				myObjValueBeforeBestResponseTrajectories[myTraderIndex,myIterationIndex] = aObjValueBeforeBestResponseTrajectories[myTraderIndex][myIterationToPlot]
				myObjValueAfterBestResponseTrajectories[myTraderIndex,myIterationIndex]  = aObjValueAfterBestResponseTrajectories[myTraderIndex][myIterationToPlot]
			end
		end
	end

	# initialises an array with all the plots
	myPlotsConvergenceObjectiveValue = []
	myPlotsConvergenceStepSize       = []

	for p in eachindex(aTradersToPlot)

		######################
		# A) Objective value #
		######################

		# initialises the plot
		myPlotConvergenceObjectiveValue = Plots.plot(
							     formatter      = :latex,
							     size           = (900,500),
							     title = string("Trader ",aTradersToPlot[p]),
							     title_location = :center,
							     legendfont     = font(10),
							     xlabel         = "Iteration of best responses \$(log_{10})\$",
		xaxisfont      = font(14),
		yaxisfont      = font(14),
		xtickfont      = font(8),
		ytickfont      = font(8),
		)

		# plots the evolution of the objective function with regards to the iterations
		Plots.plot!(
			    myPlotConvergenceObjectiveValue,
			    log.(10,range(aInitialIteration,stop=aFinalIteration)),
			    log.(10,myObjValueBeforeBestResponseTrajectories[p,:]),
			    linewidth   = 3, linestyle   = :dot,
			    color       = :red,
			    markershape = :utriangle,
			    markersize  = 6,
			    ylabel      = "Objective value \$(log_{10})\$",
			    label       = "Before best response",
			    )

		# adds the trajectory of the objective function after that the trader computes the best response
		Plots.plot!(
			    myPlotConvergenceObjectiveValue,
			    log.(10,range(aInitialIteration,stop=aFinalIteration)),
			    log.(10,myObjValueAfterBestResponseTrajectories[p,:]),
			    linewidth   = 3,
			    linestyle   = :solid,
			    color       = :black,
			    markershape = :circle,
			    markersize  = 6,
			    ylabel      = "Objective value \$(log_{10})\$",
			    label       = "After best response",
			    )

		# adjust the `xlims`
		Plots.plot!(myPlotConvergenceObjectiveValue, xlims = log.(10,[aInitialIteration-0.1,aFinalIteration+0.1]))

		# adds the plot to the vector of plots
		push!(myPlotsConvergenceObjectiveValue,myPlotConvergenceObjectiveValue)

		################
		# B) Step size #
		################

		myTraderStepSize = zeros(myNIterationsToPlot)
		for i in aInitialIteration:aFinalIteration
			myLocalStep = myBestResponseTrajectories[p,:,i+1-aInitialIteration+1] - myBestResponseTrajectories[p,:,i-aInitialIteration+1]
			myLocalStepSize = norm(myLocalStep,2)
			myTraderStepSize[i-aInitialIteration+1] = myLocalStepSize
		end

		# initialises the plot
		myPlotConvergenceStepSize = Plots.plot(
						       formatter      = :latex,
						       size           = (900,500),
						      title = string("Trader ",aTradersToPlot[p]),
						      title_location = :center,
						      legendfont     = font(10),
						      xlabel         = "Iteration of best responses \$(log_{10})\$",
		xaxisfont      = font(14),
		yaxisfont      = font(14),
		xtickfont      = font(8),
		ytickfont      = font(8),
		)

		# plots the evolution of the step size with regards to the iterations
		Plots.plot!(
			    myPlotConvergenceStepSize,
			    log.(10,range(aInitialIteration,stop=aFinalIteration)),
			    log.(10,myTraderStepSize),
			    linewidth   = 3,
			    linestyle   = :solid,
			    color       = :black,
			    markershape = :circle,
			    markersize  = 6,
			    ylabel      = "Step size \$(log_{10})\$",
			    )

		# adjust the `xlims`
		Plots.plot!(myPlotConvergenceStepSize, xlims = log.(10,[aInitialIteration-0.1,aFinalIteration+0.1]))

		# adds the plot to the vector of plots
		push!(myPlotsConvergenceStepSize,myPlotConvergenceStepSize)
	end

	###################################
	# C) Step size of all the traders #
	###################################

	myStepSizeAllTraders = zeros(myNIterationsToPlot)
	for i in aInitialIteration:aFinalIteration
		myLocalStep = myBestResponseTrajectories[:,:,i+1-aInitialIteration+1] - myBestResponseTrajectories[:,:,i-aInitialIteration+1]
		myLocalStepSize = norm(myLocalStep,2)
		myStepSizeAllTraders[i-aInitialIteration+1] = myLocalStepSize
	end

	# initialises the plot
	myPlotConvergenceStepSizeAllTraders = Plots.plot(
							 formatter      = :latex,
							 size           = (900,500),
							 title          = string("Convergence of the step size"),
							 title_location = :center,
							 legendfont     = font(10),
							 xlabel         = "Iteration of best responses \$(log_{10})\$",
							 xaxisfont      = font(14),
							 yaxisfont      = font(14),
							 xtickfont      = font(8),
							 ytickfont      = font(8),
							 )

	# plots the evolution of the step size with regards to the iterations
	Plots.plot!(
		    myPlotConvergenceStepSizeAllTraders,
		    log.(10,range(aInitialIteration,stop=aFinalIteration)),
		    log.(10,myStepSizeAllTraders),
		    label      = "\$\\Delta y_{all}\$",
		    linewidth   = 3,
		    linestyle   = :solid,
		    color       = :black,
		    markershape = :circle,
		    markersize  = 6,
		    ylabel      = "Step size \$(log_{10})\$"
		    )

	# adjust the `xlims`
	Plots.plot!(myPlotConvergenceStepSizeAllTraders, xlims = log.(10,[aInitialIteration-0.1,aFinalIteration+0.1]))

	return myPlotsConvergenceObjectiveValue, myPlotsConvergenceStepSize, myPlotConvergenceStepSizeAllTraders
end

end
