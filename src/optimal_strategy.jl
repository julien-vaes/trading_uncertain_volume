# NAME: optimal_strategy.jl
# AUTHOR: Julien Vaes
# DATE: May 22, 2019
# DESCRIPTION: module to get the optimal trading strategy

module OptimalStrategyModule

###################
## Load Packages ##
###################

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
using ..HelpFilesModule

######################
## Export functions ##
######################

export compute_optimal_strategy

######################
## Module variables ##
######################

# Corresponce of the methods name with the function associated.
ourFunctionsImplemented = Dict(
			       "Mean-Variance"    => MeanVarianceModule.get_optimal_strategy_mean_variance,
			       "Mean-CVaR_Optim"  => MeanCVaRModule.get_optimal_strategy_mean_CVaR_Optim,
			       "Mean-CVaR_Gurobi" => MeanCVaRModule.get_optimal_strategy_mean_CVaR_Gurobi,
			       #= "Mean-VaR"         => MeanVaRPrincipalScenariosModule.get_optimal_strategy_mean_VaR =#
			       )

######################
## Module functions ##
######################

"""
         compute_optimal_strategy(
				  aMarketDetails::MarketDetails,
				  aTrader::Trader;
				  aMethod::String="Mean-CVaR",
				  aOutputFilePath::String=HelpFilesModule.get_default_output_file_path(aMarketDetails,aTrader,aSimulationParameters),
				  aRecomputeOptimalStrategies=false
				  )

computes and saves the optimal strategy.

#### Arguments
* `aTrader::Trader`: a structure containing all the details of a trader.
* `aSimulationParameters::SimulationParameters`: a structure containing all the details of the parameters used to compute the optimal trading plan such as the method to use, the number of samples in the CVaR estimation.
* `aOutputFilePath::String=HelpFilesModule.get_default_output_file_path(aMarketDetails,aTrader,aSimulationParameters)`: the file path where to save the optimal strategy.
* `aRecomputeOptimalStrategies=false`: set to true if want to force the recomputation of all strategies.
"""
function compute_optimal_strategy(;
				  aTraderIndex::Int,
				  aTraders::Array{Trader,1},
				  aStrategies::Array{Strategy,1},
				  aSimulationParameters::SimulationParameters,
				  )

	# gets the stack trace, useful to know if ones has to print or not the logs of this function
	myStacktrace = stacktrace()

	# gets the trader we want to optimise
	myTrader = aTraders[aTraderIndex]

	# gets the method to use to compute the optimal strategy
	myMethod = get_method(aSimulationParameters)

	# checks that the method exists
	if !(in(myMethod,keys(ourFunctionsImplemented)))
		@error(
		       string(
			      "\nOptimalStrategyModule 101:\n",
			      "Method ",
			      myMethod,
			      " is unknown.\n Use one of the following methods:  ",
			      keys(ourFunctionsImplemented),"."
			      )
		       )
	end

	# computes the optimal execution strategy
	myOptimalStrategy, myOptimalObjectiveValue, myOptimisationSuccessful, myOutputFilePath = ourFunctionsImplemented[myMethod](
																   aTraderIndex          = aTraderIndex,
																   aTraders              = aTraders,
																   aStrategies           = aStrategies,
																   aSimulationParameters = aSimulationParameters
																   )

	# prints the results if required
	println_logs(string("+++ Results: λ = ",get_risk_aversion(myTrader)," +++"),aSimulationParameters,myStacktrace)
	if get_optimise_trading_plan(aSimulationParameters)
		println_logs(string("Optimal trading plan: ",get_trading_plan(myOptimalStrategy)),aSimulationParameters,myStacktrace)
	end
	if get_optimise_redistribution_matrix(aSimulationParameters)
		println_logs(string("Optimal redistribution matrix: ",get_redistribution_matrix(myOptimalStrategy)),aSimulationParameters,myStacktrace)
	end

	return myOptimalStrategy, myOptimalObjectiveValue, myOptimisationSuccessful, myOutputFilePath
end

end
