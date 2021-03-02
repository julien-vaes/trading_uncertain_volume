# NAME: trading_cost.jl
# AUTHOR: Julien Vaes
# DATE: May 21, 2019
# DESCRIPTION: contains function to evaluate the trading cost

module TradingCostModule

###################
## Load Packages ##
###################

using Distributed
using Distributions
using Random
using Statistics
using StatsBase

############################
## Load Personal Packages ##
############################

using ..UncertaintyStructureModule
using ..MarketDetailsModule
using ..StrategyModule
using ..TraderModule
using ..SimulationParametersModule
using ..HelpFilesModule
using ..MeanVarianceModule

######################
## Export functions ##
######################

export get_trading_cost_value_and_partial_derivatives_given_realisations
export get_trading_cost_value_and_partial_derivatives
export get_trading_cost_value_and_partial_derivatives_statistical_values
export get_trading_cost
export get_trading_cost_with_recourse
export get_trading_cost_expectation_and_gradient_Monte_Carlo

#######################
## Module Parameters ##
#######################

# methods that allows recourse
ourFunctionsWithRecourse = Dict(
				"Mean-Variance"=>get_optimal_strategy_mean_variance
				)

######################
## Module functions ##
######################

"""
get_traded_volume!(
aStrategy::Strategy,
aMarketDetails::MarketDetails,
aForecastUpdatesRealisations::Array{Float64},
aTradingPeriod::Int,
toVolumeTraded::Array{Float64,1}
)

returns the traded volume for a given trading period given a forecast updates realisations set.
The function stores the result in the argument `toVolumeTraded`.

NOTE: the parameter `toVolumeTraded` is changed which justifies the usage of ! in the function name.

#### Arguments
* `aStrategy::Strategy`: a structure containing the details of the trader's strategy, i.e. the investement decisions and the corresponding redistribution coefficient.
* `aMarketDetails::MarketDetails`: a structure containing all the details of the trading market.
* `aForecastUpdatesRealisations::Array{Float64}`: an array with the forecast updates realisation.
* `aTradingPeriod::Int`: the trading period for which we want compute the traded volume for each forecast updates realisation.
* `toVolumeTraded::Array{Float64,1}`: the vector to store the traded volume computed.
"""
function get_traded_volume!(;
			    aStrategy::Strategy,
			    aUncertaintyStructure::UncertaintyStructure,
			    aForecastUpdatesRealisations::Array{Float64},
			    aTradingPeriod::Int,
			    toVolumeTraded::Array{Float64,1}
			    )

	myInitialDemand = get_initial_demand_forecast(aUncertaintyStructure)

	myTradingPlan = get_trading_plan(aStrategy)
	myRedistributionMatrixCoeff = get_redistribution_matrix_coeff(aStrategy)

	if aTradingPeriod == 1
		# NOTE: leave [:] otherwise does not change the argument of the function
		toVolumeTraded[:] .= @views myInitialDemand.*myTradingPlan[aTradingPeriod].*ones(size(toVolumeTraded))
	else
		# NOTE: leave [:] otherwise does not change the argument of the function
		toVolumeTraded[:] .= @views myInitialDemand.*myTradingPlan[aTradingPeriod] .+ sum(aForecastUpdatesRealisations[:,k].*myRedistributionMatrixCoeff[k,aTradingPeriod] for k=1:aTradingPeriod-1)
	end

end

"""
get_traded_volume_method_with_recourse!(
aMarketDetails::MarketDetails,
aTrader::Trader,
aSimulationParameters::SimulationParameters,
aForecastUpdatesRealisations::Array{Float64},
aTradingPeriod::Int,
aVolumeTradedSoFar::Array{Float64},
toVolumeTraded::Array{Float64,1}
)

returns the traded volume at the trading period `aTradingPeriod` in the recursive version of method `aMethod`.

#### Arguments
* `aMarketDetails::MarketDetails`: a structure containing all the details of the trading market.
* `aTrader::Trader`: a structure containing all the details of a trader.
* `aSimulationParameters::SimulationParameters`: a structure containing all the details of the parameters used to compute the optimal trading plan such as the method to use, the number of samples in the CVaR estimation.
* `aForecastUpdatesRealisations::Array{Float64}`: an array with the forecast updates realisation.
* `aTradingPeriod::Int`: the trading period for which we want compute the traded volume for each forecast updates realisation.
* `aVolumeTradedSoFar::Array{Float64}`: an array with the volume that has already been traded for each realisation.
* `toVolumeTraded::Array{Float64,1}`: the vector to store the traded volume computed (the parameter is change which justifies the usage of ! in the function name).
"""
function get_traded_volume_method_with_recourse!(
						 aTrader::Trader,
						 aSimulationParameters::SimulationParameters,
						 aForecastUpdatesRealisations::Array{Float64},
						 aTradingPeriod::Int,
						 aVolumeTradedSoFar::Array{Float64},
						 toVolumeTraded::Array{Float64,1}
						 )

	# get the market details belief of the trader
	myMarketDetailsBelief = get_market_details_belief(aTrader)

	myInitialDemand = get_initial_demand_forecast(myMarketDetailsBelief)
	myNTradingPeriods = MarketDetailsModule.get_n_trading_periods(myMarketDetailsBelief)

	# gets the function to use to get the optimal trading strategy at the begining of period `aTradingPeriod`
	myMethod = get_method(aSimulationParameters)
	myRecourseFunctionToUse = ourFunctionsWithRecourse[myMethod]

	# get the uncertainty structure of the market and its estimate done by the trader
	myTraderUncertaintyStructure         = MarketDetailsModule.get_uncertainty_structure(myMarketDetailsBelief)
	myTraderPricesMovesDistributions     = get_prices_moves_distributions(myTraderUncertaintyStructure)[aTradingPeriod:end]
	myTraderForecastUpdatesDistributions = get_forecast_updates_distributions(myTraderUncertaintyStructure)[aTradingPeriod:end]
	myTraderUncertaintyStructure         = get_new_uncertainty_structure(
									     myTraderUncertaintyStructure,
									     aPricesMovesDistributions     = myTraderPricesMovesDistributions,
									     aForecastUpdatesDistributions = myTraderForecastUpdatesDistributions
									     )

	# computes the best estimate of the number of position that the trader has on the number of positions they still need to trade. 
	myBestDemandEstimate = @views myInitialDemand .- aVolumeTradedSoFar
	if aTradingPeriod > 1
		myBestDemandEstimate += @views sum(aForecastUpdatesRealisations[:,k] for k=1:aTradingPeriod-1)
	end

	# updates the market details at if t=0 is in `aTradingPeriod`
	myTaus     = get_taus(myMarketDetailsBelief)[aTradingPeriod:end]
	myEtas     = get_etas(myMarketDetailsBelief)[aTradingPeriod:end]
	myEpsilons = get_epsilons(myMarketDetailsBelief)[aTradingPeriod:end]

	mySimulationParameters = get_new_simulation_parameters(
							       aSimulationParameters,
							       aConsiderRecourse = false # because one assumes that the optimal plan is solved as if the trader ignores that recourse is possible
							       )

	myNRealisations = length(aForecastUpdatesRealisations[:,1])

	for myRealisationIndex in 1:myNRealisations

		myMarketDetailsBelief = get_new_market_details(
							       myMarketDetailsBelief,
							       aNTradingPeriods       = myNTradingPeriods-aTradingPeriod+1,
							       aTaus                  = myTaus,
							       aInitialDemandForecast = myBestDemandEstimate[myRealisationIndex],
							       aEpsilons              = myEpsilons,
							       aEtas                  = myEtas,
							       aUncertaintyStructure  = myTraderUncertaintyStructure,
							       )
		myTrader = get_new_trader(
					  aTrader,
					  aMarketDetailsBelief=myMarketDetailsBelief
					  )

		myOptimalStrategy, myObjFunctionValue = myRecourseFunctionToUse(
										aTrader = myTrader,
										aSimulationParameters = mySimulationParameters
										)

		myOptimalTradingPlan = get_trading_plan(myOptimalStrategy)
		toVolumeTraded[myRealisationIndex] = myOptimalTradingPlan[1]*myBestDemandEstimate[myRealisationIndex]
	end
end

"""
get_traded_volume_derivative_y!(
aStrategy::Strategy,
aMarketDetails::MarketDetails,
aForecastUpdatesRealisations::Array{Float64},
aTradingPeriod::Int,
aTradingDecision::Int,
toVolumeTradedDerivative::Array{Float64,1}
)

returns the traded volume derivative for a given trading period regarding to the investement decision `aTradingDecision` for a given forecast error realisations set.
The function stores the result in the argument `toVolumeTradedDerivative`.

NOTE: the parameter `toVolumeTradedDerivative` is changed which justifies the usage of ! in the function name.

#### Arguments
* `aStrategy::Strategy`: a structure containing the details of the trader's strategy, i.e. the investement decisions and the corresponding redistribution coefficient.
* `aMarketDetails::MarketDetails`: a structure containing all the details of the trading market.
* `aForecastUpdatesRealisations::Array{Float64}`: an array with forecast updates realisation.
* `aTradingPeriod::Int`: the trading period for which we want compute the derivative of the traded volume for each forecast error realisation.
* `aTradingDecision::Int`: the trading period index that relates to the investement decision for the derivative, i.e. d(n_aTradingPeriod)/d(y_aTradingDecision).
* `toVolumeTradedDerivative::Array{Float64,1}`: the vector to store the traded volume computed.
"""
function get_traded_volume_derivative_y!(;
					 aStrategy::Strategy,
					 aUncertaintyStructure::UncertaintyStructure,
					 aForecastUpdatesRealisations::Array{Float64},
					 aTradingPeriod::Int,
					 aTradingDecision::Int,
					 toVolumeTradedDerivative::Array{Float64,1}
					 )

	myInitialDemand        = get_initial_demand_forecast(aUncertaintyStructure)
	myNTradingPeriods      = UncertaintyStructureModule.get_n_trading_periods(aUncertaintyStructure)
	myTradingPlan          = get_trading_plan(aStrategy)
	myRedistributionMatrix = get_redistribution_matrix(aStrategy)

	# checks for errors
	if !(aTradingDecision < myNTradingPeriods) || !(aTradingPeriod <= myNTradingPeriods)
		@error(string(
			      "\nTradingCostModule 101:\n",
			      "The last investement decision in not considered as a variable."
			      )
		       )
	end

	toVolumeTradedDerivative[:] .*= @views 0.0
	# NOTE: leave [:] otherwise does not change the argument of the function
	if aTradingPeriod == aTradingDecision 
		# During the first trading period there is no influence of the forecast updates.
		if aTradingPeriod == 1
			toVolumeTradedDerivative[:] .= myInitialDemand
		else
			@views toVolumeTradedDerivative[:] .= myInitialDemand .+ sum(aForecastUpdatesRealisations[:,k] for k=1:aTradingPeriod-1)
		end
	elseif aTradingPeriod > aTradingDecision
		@views toVolumeTradedDerivative[:] .= sum(aForecastUpdatesRealisations[:,k].*myRedistributionMatrix[k,aTradingPeriod] for k=aTradingDecision:aTradingPeriod-1)
	else
		toVolumeTradedDerivative[:] .*= 0.0
	end

	# y_m is not part of the variables; we get rid of it with the constraint  y_m = (1-y_1-...-y_(m-1)).
	# Hence all terms depending on y_m depends on y1, ..., y_(m-1) and thus they contribute to the partial derivative in y_i.
	if aTradingPeriod == myNTradingPeriods
		toVolumeTradedDerivative[:] .-= myInitialDemand
		if myNTradingPeriods>1
			@views toVolumeTradedDerivative[:] .-= sum(aForecastUpdatesRealisations[:,k] for k=1:myNTradingPeriods-1)
		end
	end
end

"""
get_traded_volume_derivative_beta!(
aStrategy::Strategy,
aMarketDetails::MarketDetails,
aForecastUpdatesRealisations::Array{Float64},
aTradingPeriod::Int,
aRedistributionDecision::Tuple{Int64,Int64},
toVolumeTradedDerivative::Array{Float64,1}
)

TODO function description.

##### Argument
#* ``: TODO.
#
"""
function get_traded_volume_derivative_beta!(;
					    aStrategy::Strategy,
					    aUncertaintyStructure::UncertaintyStructure,
					    aForecastUpdatesRealisations::Array{Float64},
					    aTradingPeriod::Int,
					    aRedistributionDecision::Tuple{Int64,Int64},
					    toVolumeTradedDerivative::Array{Float64,1}
					    )


	myNTradingPeriods = UncertaintyStructureModule.get_n_trading_periods(aUncertaintyStructure)
	myTradingPlan = get_trading_plan(aStrategy)
	myRedistributionMatrix = get_redistribution_matrix(aStrategy)

	# checks for errors
	if !(aRedistributionDecision[1] < myNTradingPeriods) || !(aRedistributionDecision[2] < myNTradingPeriods) || !(aTradingPeriod <= myNTradingPeriods)
		@error(string(
			      "\nTradingCostModule 102:\n",
			      "The partial derivative in terms of β_(",
			      aRedistributionDecision[1],
			      ",",
			      aRedistributionDecision[2],
			      ").\nHowever this is not a variable. It is either forced to be 0, ",
			      "or derived with the constraint that for all k:\nβ_(k,1) + β_(k,2) + ... + β_(k,m-1) + β_(k,m) = 1."
			      )
		       )
	end

	toVolumeTradedDerivative[:] .*= @views 0.0
	# For having β_(k,j) having an impact on Q_i,
	# one must have that
	# 	1) i == j
	# 	2) k < i
	# and if it is not the case then d(Q_i)/d(β_(k,j)) = 0
	# Since aRedistributionDecision[2] < myNTradingPeriods, we do not treat here the partial derivative of n_m, where m = myNTradingPeriods
	if aTradingPeriod == aRedistributionDecision[2] && aRedistributionDecision[1] < aTradingPeriod
		@views toVolumeTradedDerivative[:] .+= aForecastUpdatesRealisations[:,aRedistributionDecision[1]] .* sum(myTradingPlan[r] for r=1:aRedistributionDecision[1])
	end

	# If aTradingPeriod then the coefficient in n_m are of the form β_(k,m),
	# But β_(k,m) is not part of the variables as we replace it by β_(k,m) = 1 - (β_(k,1) + β_(k,2) + ... + β_(k,m-1))
	if aTradingPeriod == myNTradingPeriods # deals with the partial derivative of n_m
		@views toVolumeTradedDerivative[:] .-= aForecastUpdatesRealisations[:,aRedistributionDecision[1]] .* sum(myTradingPlan[r] for r=1:aRedistributionDecision[1])
	end
end

"""
```
get_trading_cost_value_and_partial_derivatives_given_realisations(;
aTraderIndex::Int64,
aTraders::Array{Trader,1},
aStrategies::Array{Strategy,1},
aSimulationParameters::SimulationParameters,
aNSamples::Int,
aPriceMovesRealisations::Array{Float64},
aForecastUpdatesRealisations::Array{Array{Float64,2}}
)
```

TODO function description.

### Arguments
* `Argument1`: TODO.
* `Argument2`: TODO.
* `...`: TODO.
"""
function get_trading_cost_value_and_partial_derivatives_given_realisations(;
									   aTraderIndex::Int64,
									   aTraders::Array{Trader,1},
									   aStrategies::Array{Strategy,1},
									   aSimulationParameters::SimulationParameters,
									   aNSamples::Int,
									   aPriceMovesRealisations::Array{Float64},
									   aForecastUpdatesRealisations::Array{Array{Float64,2},1}
									   )

	# gets the trader
	myTrader = aTraders[aTraderIndex]

	# gets the market details according to trader `aTraders` belief 
	myMarketDetailsBelief = get_market_details_belief(myTrader)

	# gets the uncertainty structure of each traders according to trader `aTraders` belief
	myTradersUncertaintyStructure = get_traders_uncertainty_structure(myMarketDetailsBelief)

	# parameters accordingly to trader `aTraderIndex`
	myNTradingPeriods = MarketDetailsModule.get_n_trading_periods(myMarketDetailsBelief)
	myEpsilons        = get_epsilons(myMarketDetailsBelief)
	myEtas            = get_etas(myMarketDetailsBelief)
	myGammas          = get_gammas(myMarketDetailsBelief)
	myTaus            = get_taus(myMarketDetailsBelief)
	myInitialDemand   = get_initial_demand_forecast(myTradersUncertaintyStructure[aTraderIndex])

	# the sum of the forecast updates over all the periods for trader `aTraderIndex`
	mySumAllForecastUpdates = @views sum(aForecastUpdatesRealisations[aTraderIndex][:,k] for k=1:myNTradingPeriods)

	# the sum of the forecast updates over all the periods of all traders
	mySumAllForecastUpdatesAllTraders = @views sum((sum(aForecastUpdatesRealisations[p][:,k] for k=1:myNTradingPeriods)) for p in eachindex(aTraders))

	# the total volume that should be traded by the end of the execution period, i.e. d_T, by trader `aTraderIndex`.
	myFinalDemand = @views myInitialDemand .+ mySumAllForecastUpdates

	# the total volume that should be traded by the end of the execution period, i.e. d_T, by trader `aTraderIndex`.
	myFinalDemandAllTraders = @views sum((get_initial_demand_forecast(myTradersUncertaintyStructure[p])) for p in eachindex(aTraders)) .+ mySumAllForecastUpdatesAllTraders

	# tells if in the evalutation of the trading cost we consider that at each trading period the trader might adjust his strategy.
	myConsiderRecourse = get_consider_recourse(aSimulationParameters) 
	if myConsiderRecourse
		@error(string(
			      "\nTradingCostModule 103:\n",
			      "In the case of multitraders, the recourse is not supported yet."
			      )
		       )
	end
	myMethod = get_method(aSimulationParameters)


	# Computes the volumes traded on each trading period

	## initialisation of the variables containing the volumes traded by player with index `aTraderIndex`
	myVolumeTraded         = zeros(aNSamples,myNTradingPeriods)
	myPositiveVolumeTraded = zeros(aNSamples,myNTradingPeriods)
	myNegativeVolumeTraded = zeros(aNSamples,myNTradingPeriods)

	## initialisation of the variables containing the volumes traded by all the players except player with index `aTraderIndex`
	myVolumeTradedOtherTraders         = zeros(aNSamples,myNTradingPeriods)
	myPositiveVolumeTradedOtherTraders = zeros(aNSamples,myNTradingPeriods)
	myNegativeVolumeTradedOtherTraders = zeros(aNSamples,myNTradingPeriods)

	# Computes the volumes traded by the trader: based on eq (25) of our second paper #
	# and fills the volumes traded on each trading period
	# Note: we integrate the fact the Q_{p,i}^- Q_{p,i}^+ = 0 #
	for myPeriod in 1:myNTradingPeriods, p in eachindex(aTraders)

		# sets all the local variables back to 0
		myLocalVolumeTraded         = zeros(aNSamples)
		myLocalPositiveVolumeTraded = zeros(aNSamples)
		myLocalNegativeVolumeTraded = zeros(aNSamples)

		# gets the traded volume by trader p
		TradingCostModule.get_traded_volume!(
						     aStrategy                    = aStrategies[p],
						     aUncertaintyStructure        = myTradersUncertaintyStructure[p],
						     aForecastUpdatesRealisations = aForecastUpdatesRealisations[p],
						     aTradingPeriod               = myPeriod,
						     toVolumeTraded               = myLocalVolumeTraded
						     )

		# gets the positive volume traded by trader p
		myIndicesPositiveVolume = findall(>(0.0), myLocalVolumeTraded)
		myPositiveVolumeTraded[myIndicesPositiveVolume] .= myLocalVolumeTraded[myIndicesPositiveVolume]

		# gets the negative volume traded by trader p
		myIndicesNegativeVolume = findall(<(0.0), myLocalVolumeTraded)
		myNegativeVolumeTraded[myIndicesNegativeVolume] .= myLocalVolumeTraded[myIndicesNegativeVolume]

		if p == aTraderIndex # adds the volume traded by player with index `aTraderIndex`  
			myVolumeTraded[:,myPeriod]         += myLocalVolumeTraded
			myPositiveVolumeTraded[:,myPeriod] += myLocalPositiveVolumeTraded
			myNegativeVolumeTraded[:,myPeriod] += myLocalNegativeVolumeTraded
		else # adds the volume traded by trader p to the sum of the volumes traded by all the players except player p
			myVolumeTradedOtherTraders[:,myPeriod]         += myLocalVolumeTraded
			myPositiveVolumeTradedOtherTraders[:,myPeriod] += myLocalPositiveVolumeTraded
			myNegativeVolumeTradedOtherTraders[:,myPeriod] += myLocalNegativeVolumeTraded
		end
	end

	# vector with the details of each sample:
	## - first column:      the trading cost 
	## - kth columns (k>1): the trading cost derivative with regards to investment decision of trading period k-1, i.e. y_{k-1}.
	myTradingCostRealisations             = zeros(aNSamples)
	myTradingCostGradientYRealisations    = zeros(aNSamples,myNTradingPeriods-1)
	myTradingCostGradientBetaRealisations = zeros(aNSamples,Int(0.5*(myNTradingPeriods-1)*(myNTradingPeriods-2)))

	# vectors to store locally some values.
	myVolumeLeftToTrade      = deepcopy(myFinalDemand) # the vector that represent the volume left to trade by player with index `aTraderIndex`
	myVolumeTradedDerivative = zeros(aNSamples)        # the vector the store de derivate of a volume Q_i given a decision variables y_j, e.g. d(Q_i)/d(y_j)

	# iterative computation of the trading cost and the trading cost derivative
	for myPeriod in 1:myNTradingPeriods

		# gets the indices with positive and negative traded volumes
		myIndicesPositiveVolume = findall(>(0.0), myVolumeTraded[:,myPeriod])
		myIndicesNegativeVolume = findall(<(0.0), myVolumeTraded[:,myPeriod])

		# updates the volume traded so far by trader `aTraderIndex`
		myVolumeLeftToTrade -= myVolumeTraded[:,myPeriod]

		##########################
		## Part 1: Trading cost ##
		##########################

		## Permanent Cost

		# TODO: verigy added the contribution of the other players
		myTradingCostRealisations += @views ( ( aPriceMovesRealisations[:,myPeriod] + myGammas[myPeriod] * ( myVolumeTraded[:,myPeriod] + myVolumeTradedOtherTraders[:,myPeriod] ) ) .* myVolumeLeftToTrade )

		## Temporary cost

		### contribution of \sum_i^N ϵ_i |Q_{p,i}|
		myTradingCostRealisations += @views abs.(myVolumeTraded[:,myPeriod]) .* myEpsilons[myPeriod]

		### contribution of (\eta_i/tau_i) ( (Q_{p,i}^+)^2 + (Q_{p,i}^-)^2 ).
		myTradingCostRealisations += @views (myEtas[myPeriod]/myTaus[myPeriod]) * (myVolumeTraded[:,myPeriod].^2)

		### contribution of the other players, i.e. (\eta_i/tau_i) sum_{ptilde \neq p} (Q_{p,i}^+ Q_{p_tilde,i}^+ + Q_{p,i}^- Q_{p_tilde,i}^-).
		### Equal to 0 if only one trader
		myTradingCostRealisations += @views (myEtas[myPeriod]/myTaus[myPeriod]) * ( myNegativeVolumeTraded[:,myPeriod] .* myNegativeVolumeTradedOtherTraders[:,myPeriod] + myPositiveVolumeTraded[:,myPeriod] .* myPositiveVolumeTradedOtherTraders[:,myPeriod])

		##########################################
		## Part 2: Trading cost derivative in y ##
		##########################################

		# checks if the derivative in terms of the components of y must be computed.
		if get_optimise_trading_plan(aSimulationParameters) && !(myConsiderRecourse)

			# Warning: the computation of the derivative is valid only under the constraint that sum(y_i) = 1.
			# One computes the derivative in terms of y_1, ...., y_(m-1), where we have integrated the fact that y_m = 1 - (y_1 + ... + y_(m-1)).

			# FOR LOOP: adds the contribution of d(n_myPeriod)/d(y_myDerivativePeriod) in the derivative of the trading cost,
			#           i.e. the contribution of the partial derivatives in y of volume n_myPeriod.
			for myDerivativePeriod in 1:myNTradingPeriods-1

				# computes the derivative of the volume traded in `myPeriod` regarding to the investment decision `y_myDerivativePeriod`, i.e. d(Q_myPeriod)/d(y_myDerivativePeriod),
				# and store the result in `myTradingCostGradientYRealisations`.
				# NOTE: the recursive option is only valid if we are intersted in the trading cost value and not its gradient
				get_traded_volume_derivative_y!(
								aStrategy                    = aStrategies[aTraderIndex],
								aUncertaintyStructure        = myTradersUncertaintyStructure[aTraderIndex],
								aForecastUpdatesRealisations = aForecastUpdatesRealisations[aTraderIndex],
								aTradingPeriod               = myPeriod,
								aTradingDecision             = myDerivativePeriod,
								toVolumeTradedDerivative     = myVolumeTradedDerivative
								)

				# keeps only the derivative when the volume traded is negative
				myNegativeVolumeTradedDerivative = zeros(aNSamples)
				myNegativeVolumeTradedDerivative[myIndicesNegativeVolume] .= myVolumeTradedDerivative[myIndicesNegativeVolume]

				# keeps only the derivative when the volume traded is positive
				myPositiveVolumeTradedDerivative = zeros(aNSamples)
				myPositiveVolumeTradedDerivative[myIndicesPositiveVolume] .= myVolumeTradedDerivative[myIndicesPositiveVolume]

				# NOTE: the contribution of the partial derivative of the volume traded with regards to `y_myDerivativePeriod` is computed based on Equation (25) of our second paper ().

				## Permanent cost derivative ##

				### contribution of \sum_i^N (τ_i^{1/2} ξ_i) * (V_{p,0} + sum_k=1^N U_{p,k} - \sum_k=1^i Q_{p,k}).
				myTradingCostGradientYRealisations[:,myDerivativePeriod] -= @views myVolumeTradedDerivative .* sum( aPriceMovesRealisations[:,k] for k=myPeriod:myNTradingPeriods)

				### contribution of \sum_i^N ( γ_i * ( Q_{p,i} + sum_{ptilde} Q_{ptilde,i} ) * (V_{p,0} + sum_k=1^N U_{p,k} - \sum_k=1^i Q_{p,k}).
				myTradingCostGradientYRealisations[:,myDerivativePeriod] += @views myGammas[myPeriod] .* myVolumeTradedDerivative .* myVolumeLeftToTrade
				myTradingCostGradientYRealisations[:,myDerivativePeriod] -= @views myVolumeTradedDerivative .* sum( myGammas[k] .* ( myVolumeTraded[:,k] + myVolumeTradedOtherTraders[:,k] ) for k=myPeriod:myNTradingPeriods) # TODO: verify the contribution of the other players

				## Temporary cost derivative ##

				### contribution of \sum_i^N \epsilon_i |Q_i|.
				myTradingCostGradientYRealisations[:,myDerivativePeriod] += @views myEpsilons[myPeriod] * sign.(myVolumeTraded[:,myPeriod]) .* myVolumeTradedDerivative

				### contribution of \sum_i^N (\eta_i/tau_i) Q_i^2.
				myTradingCostGradientYRealisations[:,myDerivativePeriod] += @views 2 * (myEtas[myPeriod]/myTaus[myPeriod]) * (myVolumeTradedDerivative .* myVolumeTraded[:,myPeriod])

				### contribution of \sum_i^N (\eta_i/tau_i) sum_{ptilde \neq p} (Q_{p,i}^+ Q_{p_tilde,i}^+ + Q_{p,i}^- Q_{p_tilde,i}^-).
				myTradingCostGradientYRealisations[:,myDerivativePeriod] += @views (myEtas[myPeriod]/myTaus[myPeriod]) * (myNegativeVolumeTradedDerivative .* myNegativeVolumeTradedOtherTraders[:,myPeriod] + myPositiveVolumeTradedDerivative .* myPositiveVolumeTradedOtherTraders[:,myPeriod])
			end
		end

		##########################################
		## Part 3: Trading cost derivative in β ##
		##########################################

		# checks if the derivative in terms of the components of β must be computed:
		# checks if the method needs the derivative and if the trader has an estimate of the forecast updates other than 0 otherwise the following computation is useless.
		if get_optimise_redistribution_matrix(aSimulationParameters) && get_consider_forecast_updates(myTradersUncertaintyStructure[aTraderIndex]) && !(myConsiderRecourse)

			# Warning: the computation of the derivative is valid only under the constraint that for all k: sum_i(β_(k,i)) = 1.
			# We also impose the constraint that β_(k,i) = 0 if i ≤ k.
			# One computes the derivative in terms of β_(k,i) under the constraints just specified.

			# FOR LOOP: adds the contribution of d(n_myPeriod)/d(β_(myDerivativePeriodk,myDerivativePeriodi)) in the derivative of the trading cost,
			#           i.e. the contribution of the partial derivatives in β of volume n_myPeriod.
			#           The loop concerns only the components of the redistribution matrix β considered as variables;
			#           the others being either 0 or based on the constraint that for all k: sum_i(β_(k,i)) = 1.

			myIndex = 0
			for myDerivativePeriodk in 1:myNTradingPeriods-1
				for myDerivativePeriodi in myDerivativePeriodk+1:myNTradingPeriods-1

					myIndex += 1

					# computes the derivative of the volume traded in `myPeriod` regarding to the the redistribution parameter `β_(myDerivativePeriodk,myDerivativePeriodi)`,
					# i.e. d(n_myPeriod)/d(β_(myDerivativePeriodk,myDerivativePeriodi)),
					# and store the result in `myTradingCostGradientBetaRealisations`.
					# NOTE: the recursive option is only valid if we are intersted in the trading cost value and not its gradient
					get_traded_volume_derivative_beta!(
									   aStrategy                    = aStrategy,
									   aUncertaintyStructure        = myTradersUncertaintyStructure[aTraderIndex],
									   aForecastUpdatesRealisations = aForecastUpdatesRealisations,
									   aTradingPeriod               = myPeriod,
									   aRedistributionDecision      = (myDerivativePeriodk,myDerivativePeriodi),
									   toVolumeTradedDerivative     = myVolumeTradedDerivative
									   )

					# keeps only the derivative when the volume traded is negative
					myNegativeVolumeTradedDerivative = zeros(aNSamples)
					myNegativeVolumeTradedDerivative[myIndicesNegativeVolume] .= myVolumeTradedDerivative[myIndicesNegativeVolume]

					# keeps only the derivative when the volume traded is positive
					myPositiveVolumeTradedDerivative = zeros(aNSamples)
					myPositiveVolumeTradedDerivative[myIndicesPositiveVolume] .= myVolumeTradedDerivative[myIndicesPositiveVolume]

					# NOTE: the contribution of the partial derivative of the volume traded with regards to `y_myDerivativePeriod` is computed based on Equation (25) of our second paper ().

					## Permanent cost derivative ##

					### contribution of \sum_i^N (τ_i^{1/2} ξ_i) * (V_{p,0} + sum_k=1^N U_{p,k} - \sum_k=1^i Q_{p,k}).
					myTradingCostGradientBetaRealisations[:,myIndex] -= @views myVolumeTradedDerivative .* sum( aPriceMovesRealisations[:,k] for k=myPeriod:myNTradingPeriods)

					### contribution of \sum_i^N ( γ_i * ( Q_{p,i} + sum_{ptilde} Q_{ptilde,i} ) * (V_{p,0} + sum_k=1^N U_{p,k} - \sum_k=1^i Q_{p,k}).
					myTradingCostGradientBetaRealisations[:,myIndex] += @views myGammas[myPeriod] .* myVolumeTradedDerivative .* myVolumeLeftToTrade
					myTradingCostGradientBetaRealisations[:,myIndex] -= @views myVolumeTradedDerivative .* sum( myGammas[k] .* ( myVolumeTraded[:,k] + myVolumeTradedOtherTraders[:,k] )  for k=myPeriod:myNTradingPeriods)

					## Temporary cost derivative ##

					### contribution of \sum_i^N \epsilon_i |Q_i|.
					myTradingCostGradientBetaRealisations[:,myIndex] += @views myEpsilons[myPeriod] * sign.(myVolumeTraded[:,myPeriod]) .* myVolumeTradedDerivative

					### contribution of \sum_i^N (\eta_i/tau_i) Q_i^2.
					myTradingCostGradientBetaRealisations[:,myIndex] += @views 2 * (myEtas[myPeriod]/myTaus[myPeriod]) * (myVolumeTradedDerivative .* myVolumeTraded[:,myPeriod])

					### contribution of \sum_i^N (\eta_i/tau_i) sum_{ptilde \neq p} (Q_{p,i}^+ Q_{p_tilde,i}^+ + Q_{p,i}^- Q_{p_tilde,i}^-).
					myTradingCostGradientBetaRealisations[:,myIndex] += @views (myEtas[myPeriod]/myTaus[myPeriod]) * (myNegativeVolumeTradedDerivative .* myNegativeVolumeTradedOtherTraders[:,myPeriod] + myPositiveVolumeTradedDerivative .* myPositiveVolumeTradedOtherTraders[:,myPeriod])
				end
			end
		end
	end

	return myTradingCostRealisations, myTradingCostGradientYRealisations, myTradingCostGradientBetaRealisations
end

"""
#### Definition
```
get_trading_cost_value_and_partial_derivatives(
aStrategy::Strategy,
aTrader::Trader,
aSimulationParameters::SimulationParameters;
aNSamples::Int,
aSeed::Int=-1
)
```

generates a number of `aNSamples` realisation of the trading cost and its gradient with regards to the investment decisions.

#### Arguments
* `aStrategy::Strategy`: a structure containing the details of the trader's strategy, i.e. the investement decisions and the corresponding redistribution coefficient.
* `aTrader::Trader`: a structure containing all the details of a trader.
* `aSimulationParameters::SimulationParameters`: a structure containing all the details of the parameters used to compute the optimal trading plan such as the method to use, the number of samples in the CVaR estimation.
* `aNSamples::Int`: the number of samples to generate.
"""
function get_trading_cost_value_and_partial_derivatives(;
							aTraderIndex::Int64,
							aTraders::Array{Trader,1},
							aStrategies::Array{Strategy,1},
							aSimulationParameters::SimulationParameters,
							aNSamples::Int,
							aSeed::Int=-1
							)

	# fixes the seed
	if aSeed > 0
		Random.seed!(aSeed)
	end

	# gets the trader
	myTrader = aTraders[aTraderIndex]

	# gets the market details according to trader `aTraders` belief 
	myMarketDetailsBelief = get_market_details_belief(myTrader)

	# gets the number of players
	myNTraders = get_n_traders(myMarketDetailsBelief)

	# gets the uncertainty structure of each traders according to trader `aTraders` belief
	myTradersUncertaintyStructure = get_traders_uncertainty_structure(myMarketDetailsBelief)

	# gets the number of trading periods
	myNTradingPeriods = MarketDetailsModule.get_n_trading_periods(myMarketDetailsBelief)


	# initialises the array that will contain the forecast update realisations for each trader
	myForecastUpdatesRealisations = Array{Array{Float64,2},1}(undef,myNTraders)
	for p in eachindex(aTraders)
		myForecastUpdatesRealisations[p] = zeros(aNSamples,myNTradingPeriods)
	end

	# initialises the array that will contain the prices shift realisations for the trader with index `aTraderIndex`
	myPriceMovesRealisations = zeros(aNSamples,myNTradingPeriods)

	# generates the uncertainty realisations of the forecast updates for every player different from the player with index `aTraderIndex`
	for p in eachindex(aTraders)
		if p != aTraderIndex
			# gets a number of `aNSamples` of realisations of the price moves and forecast updates for the p-th trader
			generate_uncertainty_realisations!(
							   aSimulationParameters        = aSimulationParameters,
							   aUncertaintyStructure        = myTradersUncertaintyStructure[p],
							   aNSamples                    = aNSamples,
							   aPriceMovesRealisations      = myPriceMovesRealisations,
							   aForecastUpdatesRealisations = myForecastUpdatesRealisations[p]
							   )
		end
	end

	# generates the uncertainty realisations of the forecast updates AND the price moves the player with index `aTraderIndex`
	generate_uncertainty_realisations!(
					   aSimulationParameters        = aSimulationParameters,
					   aUncertaintyStructure        = myTradersUncertaintyStructure[aTraderIndex],
					   aNSamples                    = aNSamples,
					   aPriceMovesRealisations      = myPriceMovesRealisations,
					   aForecastUpdatesRealisations = myForecastUpdatesRealisations[aTraderIndex]
					   )

	return get_trading_cost_value_and_partial_derivatives_given_realisations(
										 aTraderIndex                 = aTraderIndex,
										 aTraders                     = aTraders,
										 aStrategies                  = aStrategies,
										 aSimulationParameters        = aSimulationParameters,
										 aNSamples                    = aNSamples,
										 aPriceMovesRealisations      = myPriceMovesRealisations,
										 aForecastUpdatesRealisations = myForecastUpdatesRealisations
										 )
end

"""
#### Definition
```
get_trading_cost_value_and_partial_derivatives_statistical_values(
aStrategy::Strategy,
aTrader::Trader,
aSimulationParameters::SimulationParameters;
aNSamples::Int,
aSeed::Int=-1,
aVaR::Float64
)
```

generates a number of `aNSamples` realisation of the trading cost and its gradient with regards to the investment decisions.

#### Arguments
* `aStrategy::Strategy`: a structure containing the details of the trader's strategy, i.e. the investement decisions and the corresponding redistribution coefficient.
* `aTrader::Trader`: a structure containing all the details of a trader.
* `aSimulationParameters::SimulationParameters`: a structure containing all the details of the parameters used to compute the optimal trading plan such as the method to use, the number of samples in the CVaR estimation.
* `aNSamples::Int`: the number of samples to generate.
"""
function get_trading_cost_value_and_partial_derivatives_statistical_values(;
									   aTraderIndex::Int64,
									   aTraders::Array{Trader,1},
									   aStrategies::Array{Strategy,1},
									   aSimulationParameters::SimulationParameters,
									   aNSamples::Int,
									   aSeed::Int=-1,
									   aVaR::Float64=0.0
									   )
	# fixes the seed
	if aSeed > 0
		Random.seed!(aSeed)
	end

	# gets the trader
	myTrader = aTraders[aTraderIndex]

	# gets the market details according to trader `aTraders` belief 
	myMarketDetailsBelief = get_market_details_belief(myTrader)

	# gets the number of trading periods
	myNTradingPeriods = MarketDetailsModule.get_n_trading_periods(myMarketDetailsBelief)

	# gets the number of players
	myNTraders = get_n_traders(myMarketDetailsBelief)

	# gets the uncertainty structure of each traders according to trader `aTraders` belief
	myTradersUncertaintyStructure = get_traders_uncertainty_structure(myMarketDetailsBelief)

	####################################
	# Initialises the output variables #
	####################################


	# variables to store the sum of the cost
	mySumCost             = 0.0 # ∑ Cost_ω
	mySumCostSquared      = 0.0 # ∑ Cost_ω^2
	mySumCostGradientY    = zeros(myNTradingPeriods-1)
	mySumCostGradientBeta = zeros(Int(0.5*(myNTradingPeriods-1)*(myNTradingPeriods-2)))

	# variables to store the sum of the cost above aVaR
	myNCostAboveVaR               = 0.0
	mySumCostAboveVaR             = 0.0
	mySumCostAboveVaRGradientY    = zeros(myNTradingPeriods-1)
	mySumCostAboveVaRGradientBeta = zeros(Int(0.5*(myNTradingPeriods-1)*(myNTradingPeriods-2)))

	# initialises the array that will contain the forecast update realisations for each trader
	myForecastUpdatesRealisations = Array{Array{Float64,2},1}(undef,myNTraders)
	for p in eachindex(aTraders)
		myForecastUpdatesRealisations[p] = zeros(1,myNTradingPeriods)
	end

	# initialises the array that will contain the prices shift realisations for the trader with index `aTraderIndex`
	myPriceMovesRealisations = zeros(1,myNTradingPeriods)

	for s in 1:aNSamples

		# generates the uncertainty realisations of the forecast updates for every player different from the player with index `aTraderIndex`
		for p in eachindex(aTraders)
			if p != aTraderIndex
				# gets a number of `aNSamples` of realisations of the price moves and forecast updates for the p-th trader
				generate_uncertainty_realisations!(
								   aSimulationParameters        = aSimulationParameters,
								   aUncertaintyStructure        = myTradersUncertaintyStructure[p],
								   aNSamples                    = 1,
								   aPriceMovesRealisations      = myPriceMovesRealisations,
								   aForecastUpdatesRealisations = myForecastUpdatesRealisations[p]
								   )
			end
		end

		# generates the uncertainty realisations of the forecast updates AND the price moves the player with index `aTraderIndex`
		generate_uncertainty_realisations!(
						   aSimulationParameters        = aSimulationParameters,
						   aUncertaintyStructure        = myTradersUncertaintyStructure[aTraderIndex],
						   aNSamples                    = 1,
						   aPriceMovesRealisations      = myPriceMovesRealisations,
						   aForecastUpdatesRealisations = myForecastUpdatesRealisations[aTraderIndex]
						   )

		# get the trading cost realisation and gradient
		myLocalTradingCostRealisations, myLocalTradingCostGradientY, myLocalTradingCostGradientBeta = get_trading_cost_value_and_partial_derivatives_given_realisations(
																						aTraderIndex                 = aTraderIndex,
																						aTraders                     = aTraders,
																						aStrategies                  = aStrategies,
																						aSimulationParameters        = aSimulationParameters,
																						aNSamples                    = 1,
																						aPriceMovesRealisations      = myPriceMovesRealisations,
																						aForecastUpdatesRealisations = myForecastUpdatesRealisations
																						)

		# variables to store the sum of the cost
		mySumCost             += myLocalTradingCostRealisations[1]
		mySumCostSquared      += myLocalTradingCostRealisations[1]^2
		mySumCostGradientY    += myLocalTradingCostGradientY[1,:]
		mySumCostGradientBeta += myLocalTradingCostGradientBeta[1,:]

		# variables to store the sum of the cost above aVaR
		if myLocalTradingCostRealisations[1] >= aVaR
			myNCostAboveVaR               += 1
			mySumCostAboveVaR             += myLocalTradingCostRealisations[1] - aVaR
			mySumCostAboveVaRGradientY    += myLocalTradingCostGradientY[1,:]
			mySumCostAboveVaRGradientBeta += myLocalTradingCostGradientBeta[1,:]
		end
	end

	# expectation, i.e E[ Cost ]
	myExpectation             = mySumCost / aNSamples # E[ Cost ] 
	myExpectationSquared      = mySumCostSquared / aNSamples # E[ Cost^2 ] 
	myGradientYExpectation    = mySumCostGradientY ./ aNSamples
	myGradientBetaExpectation = mySumCostGradientBeta ./ aNSamples

	# variance
	myVariance = myExpectationSquared - myExpectation^2 

	# expectation above var, i.e. E[ (Cost - aVaR)+ ]
	myExpectationAboveVaR             = mySumCostAboveVaR ./ aNSamples
	myGradientYExpectationAboveVaR    = mySumCostAboveVaRGradientY ./ aNSamples
	myGradientBetaExpectationAboveVaR = mySumCostAboveVaRGradientBeta ./ aNSamples

	# gradient of E[ (Cost - aVaR)+ ] in terms of VaR
	myGradientExpectationVaR = - myNCostAboveVaR ./ aNSamples

	###############################
	# Dictionary with the results #
	###############################
	
	myDictResult = Dict()
	myDictResult["NumberOfSamples"]                   = aNSamples
	myDictResult["Expectation"]                       = myExpectation
	myDictResult["ExpectationGradientY"]              = myGradientYExpectation
	myDictResult["ExpectationGradientBeta"]           = myGradientBetaExpectation
	myDictResult["Variance"]                          = myVariance
	myDictResult["ExpectationGreaterVaR"]             = myExpectationAboveVaR
	myDictResult["ExpectationGreaterVaRGradientY"]    = myGradientYExpectationAboveVaR
	myDictResult["ExpectationGreaterVaRGradientBeta"] = myGradientBetaExpectationAboveVaR
	myDictResult["VaRGradient"]                       = myGradientExpectationVaR 

	return myDictResult
end

"""
get_trading_cost(
aStrategy::Strategy,
aTrader::Trader,
aSimulationParameters::SimulationParameters,
aNSamples::Int;
)

generates a number of `aNSamples` samples and returns the trading cost.

#### Arguments
* `aStrategy::Strategy`: a structure containing the details of the trader's strategy, i.e. the investement decisions and the corresponding redistribution coefficient.
* `aTrader::Trader`: a structure containing all the details of a trader.
* `aSimulationParameters::SimulationParameters`: a structure containing all the details of the parameters used to compute the optimal trading plan such as the method to use, the number of samples in the CVaR estimation.
* `aNSamples::Int`: the number of samples to generate.
"""
function get_trading_cost(;
			  aTraderIndex::Int64,
			  aTraders::Array{Trader,1},
			  aStrategies::Array{Strategy,1},
			  aSimulationParameters::SimulationParameters,
			  aNSamples::Int,
			  aSeed::Int=-1
			  )


	myTradingCostRealisations, myTradingCostGradientYRealisations, myTradingCostGradientBetaRealisations = get_trading_cost_value_and_partial_derivatives(
																			      aTraderIndex          = aTraderIndex,
																			      aTraders              = aTraders,
																			      aStrategies           = aStrategies,
																			      aSimulationParameters = aSimulationParameters,
																			      aNSamples             = aNSamples,
																			      aSeed                 = aSeed
																			      )
	myTradingCostRealisations
end

function get_trading_cost_with_recourse(;
					aTraderIndex::Int64,
					aTraders::Array{Trader,1},
					aStrategies::Array{Strategy,1},
					aSimulationParameters::SimulationParameters,
					aNSamples::Int,
					aSeed::Int=-1
					)
	# TODO 

	return nothing
end

"""
merge_trading_cost_expectation_gradient_dict(aDic1,aDic2)

The funcion merges the details contained in the dictionaries `aDic1` and `aDic2`.
This function is useful when merging the estimates obtained while doing parallel computing.

#### Arguments
* `aDic1`: first dictionary that will be merged.
* `aDic2`: second dictionary that will be merged.
"""
function merge_trading_cost_expectation_gradient_dict(aDic1,aDic2)

	myMergedStats = Dict()

	#####################
	# Float64 of Samples #
	#####################

	# gets the details of both dictionaries
	myN1 = aDic1["NumberOfSamples"]
	myN2 = aDic2["NumberOfSamples"]

	# merges the details of both dictionaries
	myN  = myN1 + myN2

	###############
	# Expectation #
	###############

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

	##########
	# Output #
	##########

	# creates a dictionary with the merged details
	myMergedStats["NumberOfSamples"]                   = myN
	myMergedStats["Expectation"]                       = myExpectation
	myMergedStats["ExpectationGradientY"]              = myExpectationGradientY
	myMergedStats["ExpectationGradientBeta"]           = myExpectationGradientBeta

	return myMergedStats
end

"""
#### Definition
```
get_expectation_monte_carlo(Argument1, Argument2, ...)
```

TODO function description.

#### Arguments
* `Argument1`: TODO.
* `Argument2`: TODO.
* `...`: TODO.
"""
function get_trading_cost_expectation_and_gradient_Monte_Carlo(;
							       aStrategy::Strategy,
							       aTrader::Trader,
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

	myStats = @distributed (merge_trading_cost_expectation_gradient_dict) for myIter in eachindex(myNSamplesToRun)

		myRealisations, myTradingCostGradientYRealisations, myTradingCostGradientBetaRealisations = get_trading_cost_value_and_partial_derivatives(
																			   aStrategy             = aStrategy,
																			   aTrader               = aTrader,
																			   aSimulationParameters = aSimulationParameters,
																			   aNSamples             = myNSamplesToRun[myIter],
																			   aSeed                 = myIter # TODO: seed check
																			   )
		myLocalStats = Dict()

		# Float64 of realisations computed
		myLocalStats["NumberOfSamples"] = myNSamplesToRun[myIter]

		###############
		# Expectation #
		###############

		# Expecetation of the trading cost: |E[ Cost ].
		myLocalStats["Expectation"] = mean(myRealisations)

		# Gradient of the expecetation of the trading cost in terms of y: ∇_y |E[ Cost ].
		myExpectationGradientY = mean(myTradingCostGradientYRealisations,dims=1)
		myLocalStats["ExpectationGradientY"] = reshape(myExpectationGradientY,length(myExpectationGradientY))

		# Gradient of the expecetation of the trading cost in terms of β: ∇_β |E[ Cost ].
		myExpectationGradientBeta = mean(myTradingCostGradientBetaRealisations,dims=1)
		myLocalStats["ExpectationGradientBeta"] = reshape(myExpectationGradientBeta,length(myExpectationGradientBeta))

		##########
		# Output #
		##########

		# What is returned by each iteration and will be used in the aggregating function `merge_trading_cost_expectation_gradient_dict`
		myLocalStats
	end

	return myStats["Expectation"], myStats["ExpectationGradientY"], myStats["ExpectationGradientBeta"]
end

end
