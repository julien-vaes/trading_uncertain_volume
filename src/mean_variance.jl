# NAME: mean_variance.jl
# AUTHOR: Julien Vaes
# DATE: May 21, 2019
# DESCRIPTION: Optimal exectution strategy under the mean-variance framework (see Almgren and Chriss (2001)).

module MeanVarianceModule

###################
## Load Packages ##
###################

using JuMP
using Gurobi
using Statistics
using LinearAlgebra

############################
## Load Personal Packages ##
############################

using ..UncertaintyStructureModule
using ..MarketDetailsModule
using ..StrategyModule
using ..TraderModule
using ..SimulationParametersModule
using ..HelpFilesModule

######################
## Export functions ##
######################

export get_optimal_strategy_mean_variance

#####################
## Global constant ##
#####################

const ourGurobiEnvironment = Gurobi.Env()

######################
## Module functions ##
######################

"""
```
get_variable_absolute_value(aVariable::VariableRef)
```

returns the absolute value of `aVariable` whichh can then be used in a JuMP optimisation model.

#### Argument
* `aVariable::Variable`: the JuMP variable for which one wants to take the absolute value.
"""
function get_variable_absolute_value(aVariable::VariableRef)
	myModel = aVariable.model
	myNewVariable = @variable(myModel)
	@constraint(myModel, myNewVariable >=  aVariable)
	@constraint(myModel, myNewVariable >= -aVariable)
	return myNewVariable
end

"""
```
get_optimal_strategy_mean_variance(;
aTraderIndex::Int,
aTraders::Array{Trader,1},
aStrategies::Array{Strategy,1},
aSimulationParameters::SimulationParameters,
)
```

returns the optimal strategy under Almgren and Chriss framework,
i.e. Expecation-Variance framework under price uncertainty.

#### Arguments
* `aTraderIndex::Int`: the trader index for whom the optimal strategy will be computed.
* `aTraders::Array{Trader,1}`: an array containing a structure with the details of each trader.
* `aStrategies::Array{Strategy,1}`: an array containing the `Strategy` structure of each trader.
* `aSimulationParameters::SimulationParameters`: a structure containing all the details of the parameters used to compute the optimal trading plan.
* `aDecisionTimeToComputeOptimalTradingPlan::Int = 0`: the decision time at which to compute the optimal strategy (useful in the case when compute with recourse).
"""
function get_optimal_strategy_mean_variance(;
					    aTraderIndex::Int,
					    aTraders::Array{Trader,1},
					    aStrategies::Array{Strategy,1},
					    aSimulationParameters::SimulationParameters,
					    aDecisionTimeToComputeOptimalTradingPlan::Int = 0
					    )

	# gets the stack trace, useful to know if ones has to print or not the logs of this function
	myStacktrace = stacktrace()

	# gets the trader we want to optimise
	myTrader = aTraders[aTraderIndex]

	# gets the output file path in which the optimal strategy will be stored
	myOutputFilePath = HelpFilesModule.get_default_output_file_path(
									aMarketDetails                   = TraderModule.get_market_details_belief(myTrader),
									aTraderIndex                     = aTraderIndex,
									aTraders                         = aTraders,
									aStrategies                      = aStrategies,
									aSimulationParameters            = aSimulationParameters,
									aIncludePartialMarketDetailsHash = true,
									aIncludeSimulationParametersHash = true,
									aIncludeTradersHash              = true,
									aIncludeStrategiesHash           = false,
									aSpecificFolder                  = "outputs/trading_plan/mean_variance/"
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
						 "\nMeanVarianceModule 101:\n",
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

		############################################# 
		# Part 2.1: solves the optimisation problem # 
		############################################# 

		# gets the market details belief of the trader
		myMarketDetailsBelief = get_market_details_belief(myTrader)

		# gets the uncertainty structure belief of all the trader according to trader `aTraderIndex`
		myTradersUncertaintyStructure = MarketDetailsModule.get_traders_uncertainty_structure(myMarketDetailsBelief)

		# gets the uncertainty structure of trader `aTraderIndex`
		myUncertaintyStructure = myTradersUncertaintyStructure[aTraderIndex]

		if (get_uncertainty_generation_method(aSimulationParameters) != "normal")
			@error(string(
				      "\nMeanVarianceModule 102:\n",
				      "Computing the optimal trading plan in the mean-variance framework (Almgren and Chriss) requires that the uncertainty is generated by independent centered normally distributed random variables, i.e. that get_uncertainty_generation_method(aSimulationParameters) is equal to normal."
				      )
			       )
		end

		if (get_consider_forecast_updates(myUncertaintyStructure))
			@error(string(
				      "\nMeanVarianceModule 103:\n",
				      "The optimisation while considering the forecast updates is not handled in the mean-variance framework (Almgren and Chriss).",
				      )
			       )
		end

		myInitialDemand    = get_initial_demand_forecast(myUncertaintyStructure)
		myNTradingPeriods  = MarketDetailsModule.get_n_trading_periods(myMarketDetailsBelief)
		myRiskAversion     = get_risk_aversion(myTrader)
		myGammas           = get_gammas(myMarketDetailsBelief)
		myEpsilons         = get_epsilons(myMarketDetailsBelief)
		myEtas             = get_etas(myMarketDetailsBelief)
		myQuadraticMatrixM = get_M_matrix_quadratic_part_trading_cost(myMarketDetailsBelief)

		# gets the price distribution
		myPricesMovesDistributions = get_prices_moves_distributions(myUncertaintyStructure)

		# gets the redistribution matrix
		myRedistributionMatrix = get_redistribution_matrix(aStrategies[aTraderIndex])

		# checks if one does not have to compute the optimal plan of all trading periods but from [aDecisionTimeToComputeOptimalTradingPlan,T]
		# If it is the case, we update the values to trade
		if (aDecisionTimeToComputeOptimalTradingPlan != 0)
			myNTradingPeriods          = myNTradingPeriods - aDecisionTimeToComputeOptimalTradingPlan
			myGammas                   = myGammas[aDecisionTimeToComputeOptimalTradingPlan+1:end]
			myEpsilons                 = myEpsilons[aDecisionTimeToComputeOptimalTradingPlan+1:end]
			myEtas                     = myEtas[aDecisionTimeToComputeOptimalTradingPlan+1:end]
			myQuadraticMatrixM         = myQuadraticMatrixM[aDecisionTimeToComputeOptimalTradingPlan+1:end,aDecisionTimeToComputeOptimalTradingPlan+1:end]
			myPricesMovesDistributions = myPricesMovesDistributions[aDecisionTimeToComputeOptimalTradingPlan+1:end]
			myRedistributionMatrix     = myRedistributionMatrix[aDecisionTimeToComputeOptimalTradingPlan+1:end,aDecisionTimeToComputeOptimalTradingPlan+1:end]
		end

		# creates the model
		#= myModel = Model(with_optimizer(Gurobi.Optimizer, ourGurobiEnvironment, OutputFlag=0)) =#

		myModel = Model(optimizer_with_attributes(() -> Gurobi.Optimizer(ourGurobiEnvironment)))

		JuMP.set_optimizer_attribute(myModel, "OutputFlag", 0)
		#= JuMP.set_optimizer_attribute(myModel, "Presolve", 0) =#

		# VARIABLES #

		@variable(myModel, myVolumes[1:myNTradingPeriods])
		@variable(myModel, myVolumeLeftToTrade[1:myNTradingPeriods+1])

		# OBJECTIVE #

		@objective(myModel, Min, 
			   sum( mean(myPricesMovesDistributions[i]) * myVolumeLeftToTrade[i] for i = 1:myNTradingPeriods)
			   + sum( myGammas[i] * myVolumes[i] * myInitialDemand for i = 1:myNTradingPeriods)
			   + sum( myEpsilons[i] * get_variable_absolute_value(myVolumes[i]) for i = 1:myNTradingPeriods) 
			   + (1/2) * myVolumes[1:end-1]' * myQuadraticMatrixM * myVolumes[1:end-1]
			   - myGammas[end] * myInitialDemand * myVolumes[end]
			   + myEtas[end] * myInitialDemand * ( myInitialDemand - 2 * (myInitialDemand - myVolumes[end]) )
			   + myRiskAversion * (
					       sum(var(myPricesMovesDistributions[i])*myVolumeLeftToTrade[i]^2 for i = 1:myNTradingPeriods)  
					       )
			   )

		@constraint(myModel, sum(myVolumes[i] for i = 1:myNTradingPeriods) == myInitialDemand)
		@constraint(myModel, [j=1:myNTradingPeriods+1], myVolumeLeftToTrade[j] == myInitialDemand - sum(myVolumes[i-1] for i = 2:j))

		# SOLVING #
		status = optimize!(myModel)

		# computes the optimal trading plan
		myOptimalTradingPlan = JuMP.value.(myVolumes)./myInitialDemand

		# tells if the strategy obtained should be saved, i.e. in the case where the algorithm has successfully converged
		myOptimisationSuccessful = termination_status(myModel) == MOI.OPTIMAL

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
							 aRedistributionMatrix = myRedistributionMatrix
							 )
			myOptimalObjectiveValue = JuMP.objective_value(myModel)
		else # the optimisation was NOT successful
			myOptimalStrategy       = StrategyModule.get_strategy()
			myOptimalObjectiveValue = -1.0
			@error(
			       string(
				      "\nMeanVarianceModule 104:\n",
				      "The optimisation (Mean-Variance with Gurobi) was not successful.",
				      )
			       )
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
