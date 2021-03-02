# NAME: uncertainty_structure.jl
# AUTHOR: Julien Vaes
# DATE: November 25, 2019
# DESCRIPTION: module containing the structure of an uncertainty estimate

module UncertaintyStructureModule

###################
## Load Packages ##
###################

import Base: hash, rand
using Distributions, KernelDensity

############################
## Load Personal Packages ##
############################

using ..SimulationParametersModule

######################
## Export functions ##
######################

export UncertaintyStructure

export get_n_trading_periods
export get_consider_price_moves, get_consider_forecast_updates
export get_initial_demand_forecast
export get_prices_moves_distributions, get_forecast_updates_distributions
export get_scenario_approximation, get_prob_extreme_in_rays
export get_extreme_rays, get_rays_probabilities, get_n_extreme_rays
export get_dist_estimate_along_rays
export get_black_box_function_generating_uncertainty
export get_uncertainty_structure, get_new_uncertainty_structure
export get_dict_from_uncertainty_structure, get_uncertainty_structure_from_dict
export hash_uncertainty_structure
export generate_uncertainty_realisations!

######################
## Module functions ##
######################

### START: STRUCTURE UncertaintyStructure ###

# A structure containing all details for a structurename.
# The attributes in the structure: theNTradingPeriods, theConsiderPriceMoves, thePricesMovesDistributions, theConsiderForecastUpdates, theInitialDemandForecast, theForecastUpdatesDistributions, theScenarioApproximation, theProbExtremeInRays, theExtremeRays, theNExtremeRays, theRaysProbabilities, theDistEstimateAlongRays, theBlackBoxFunctionGeneratingUncertainty
struct UncertaintyStructure 
	theNTradingPeriods 	                  # TODO.
	theConsiderPriceMoves::Bool               # TODO.
	thePricesMovesDistributions 	          # TODO.
	theConsiderForecastUpdates::Bool          # TODO.
	theInitialDemandForecast::Float64         # the initial demand, i.e. D0.
	theForecastUpdatesDistributions	          # TODO.
	theScenarioApproximation::Bool 	          # TODO.
	theProbExtremeInRays::Float64 	          # TODO.
	theExtremeRays::Array{Array{Float64,1},1} # TODO.
	theNExtremeRays::Int                      # TODO.
	theRaysProbabilities::Array{Float64,1}    # TODO.
	theDistEstimateAlongRays 	          # TODO.
	theBlackBoxFunctionGeneratingUncertainty  # a function to generate random realisations of the underlying random variables, i.e. price moves and forecast updates.
end

## STRUCTURE UncertaintyStructure: get functions
"""
```
get_n_trading_periods(aUncertaintyStructure::UncertaintyStructure)
```

returns the attribute `theNTradingPeriods` of the structure `aUncertaintyStructure`.

### Argument
* `aUncertaintyStructure::UncertaintyStructure`: TODO.
"""
function get_n_trading_periods(aUncertaintyStructure::UncertaintyStructure)
	return aUncertaintyStructure.theNTradingPeriods
end

"""
```
get_consider_price_moves(aUncertaintyStructure::UncertaintyStructure)
```

returns the attribute `theConsiderPriceMoves` of the structure `aUncertaintyStructure`.

### Argument
* `aUncertaintyStructure::UncertaintyStructure`: TODO.
"""
function get_consider_price_moves(aUncertaintyStructure::UncertaintyStructure)
	return aUncertaintyStructure.theConsiderPriceMoves
end

"""
```
get_prices_moves_distributions(aUncertaintyStructure::UncertaintyStructure)
```

returns the attribute `thePricesMovesDistributions` of the structure `aUncertaintyStructure`.

### Argument
* `aUncertaintyStructure::UncertaintyStructure`: TODO.
"""
function get_prices_moves_distributions(aUncertaintyStructure::UncertaintyStructure)
	return aUncertaintyStructure.thePricesMovesDistributions
end

"""
```
get_consider_forecast_updates(aUncertaintyStructure::UncertaintyStructure)
```

returns the attribute `theConsiderForecastUpdates` of the structure `aUncertaintyStructure`.

### Argument
* `aUncertaintyStructure::UncertaintyStructure`: TODO.
"""
function get_consider_forecast_updates(aUncertaintyStructure::UncertaintyStructure)
	return aUncertaintyStructure.theConsiderForecastUpdates
end

"""
```
get_initial_demand_forecast(aUncertaintyStructure::UncertaintyStructure)
```

returns the attribute `theInitialDemandForecast` of the structure `aUncertaintyStructure`.

### Argument
* `aUncertaintyStructure::UncertaintyStructure`: TODO.
"""
function get_initial_demand_forecast(aUncertaintyStructure::UncertaintyStructure)
	return aUncertaintyStructure.theInitialDemandForecast
end

"""
```
get_forecast_updates_distributions(aUncertaintyStructure::UncertaintyStructure)
```

returns the attribute `theForecastUpdatesDistributions` of the structure `aUncertaintyStructure`.

### Argument
* `aUncertaintyStructure::UncertaintyStructure`: TODO.
"""
function get_forecast_updates_distributions(aUncertaintyStructure::UncertaintyStructure)
	return aUncertaintyStructure.theForecastUpdatesDistributions
end

"""
```
get_scenario_approximation(aUncertaintyStructure::UncertaintyStructure)
```

returns the attribute `theScenarioApproximation` of the structure `aUncertaintyStructure`.

### Argument
* `aUncertaintyStructure::UncertaintyStructure`: TODO.
"""
function get_scenario_approximation(aUncertaintyStructure::UncertaintyStructure)
	return aUncertaintyStructure.theScenarioApproximation
end

"""
```
get_prob_extreme_in_rays(aUncertaintyStructure::UncertaintyStructure)
```

returns the attribute `theProbExtremeInRays` of the structure `aUncertaintyStructure`.

### Argument
* `aUncertaintyStructure::UncertaintyStructure`: TODO.
"""
function get_prob_extreme_in_rays(aUncertaintyStructure::UncertaintyStructure)
	return aUncertaintyStructure.theProbExtremeInRays
end

"""
```
get_extreme_rays(aUncertaintyStructure::UncertaintyStructure)
```

returns the attribute `theExtremeRays` of the structure `aUncertaintyStructure`.

### Argument
* `aUncertaintyStructure::UncertaintyStructure`: TODO.
"""
function get_extreme_rays(aUncertaintyStructure::UncertaintyStructure)
	return aUncertaintyStructure.theExtremeRays
end

"""
```
get_n_extreme_rays(aUncertaintyStructure::UncertaintyStructure)
```

returns the attribute `theNExtremeRays` of the structure `aUncertaintyStructure`.

### Argument
* `aUncertaintyStructure::UncertaintyStructure`: TODO.
"""
function get_n_extreme_rays(aUncertaintyStructure::UncertaintyStructure)
	return aUncertaintyStructure.theNExtremeRays
end

"""
```
get_rays_probabilities(aUncertaintyStructure::UncertaintyStructure)
```

returns the attribute `theRaysProbabilities` of the structure `aUncertaintyStructure`.

### Argument
* `aUncertaintyStructure::UncertaintyStructure`: TODO.
"""
function get_rays_probabilities(aUncertaintyStructure::UncertaintyStructure)
	return aUncertaintyStructure.theRaysProbabilities
end

"""
```
get_dist_estimate_along_rays(aUncertaintyStructure::UncertaintyStructure)
```

returns the attribute `theDistEstimateAlongRays` of the structure `aUncertaintyStructure`.

### Argument
* `aUncertaintyStructure::UncertaintyStructure`: TODO.
"""
function get_dist_estimate_along_rays(aUncertaintyStructure::UncertaintyStructure)
	return aUncertaintyStructure.theDistEstimateAlongRays
end

"""
```
get_black_box_function_generating_uncertainty(aUncertaintyStructure::UncertaintyStructure)
```

returns the attribute `theBlackBoxFunctionGeneratingUncertainty` of the structure `aUncertaintyStructure`.

### Argument
* `aUncertaintyStructure::UncertaintyStructure`: TODO.
"""
function get_black_box_function_generating_uncertainty(aUncertaintyStructure::UncertaintyStructure)
	return aUncertaintyStructure.theBlackBoxFunctionGeneratingUncertainty
end

### END: STRUCTURE UncertaintyStructure ###

function get_uncertainty_structure(;
				   aNTradingPeriods::Int=-1,
				   aConsiderPriceMoves::Bool=true,
				   aPricesMovesDistributions=[],
				   aConsiderForecastUpdates::Bool=true,
				   aInitialDemandForecast::Float64=-1.0,
				   aForecastUpdatesDistributions=[],
				   aScenarioApproximation::Bool=false,
				   aProbExtremeInRays::Float64=0.0,
				   aExtremeRays=Array{Array{Float64,1},1}(),
				   aNExtremeRays::Int=length(aExtremeRays),
				   aRaysProbabilities=Array{Float64,1}(),
				   aDistEstimateAlongRays=[],
				   aBlackBoxFunctionGeneratingUncertainty=-1
				   )

	# checks that the number of elements of the vector containing the price moves distributions equals the number of trading periods
	if aConsiderPriceMoves && length(aPricesMovesDistributions) != aNTradingPeriods
		@error(
		       string(
			      "\nUncertaintyStructureModule 101:\n",
			      "The number of elements in the vector containing the price moves distributions should be equal to ",aNTradingPeriods,", the number of trading periods." 
			      )
		       )
	end

	# checks that the number of elements of the vector containing the forecast updates distributions equals the number of trading periods
	if aConsiderForecastUpdates && length(aForecastUpdatesDistributions) != aNTradingPeriods
		@error(
		       string(
			      "\nUncertaintyStructureModule 102:\n",
			      "The number of elements in the vector containing the forecast updates distributions should be equal to ",aNTradingPeriods,", the number of trading periods." 
			      )
		       )
	end

	# if the uncertainty structure does not consider the price moves, then we initialise the price moves distributions to a zero variance distribution.
	myPricesMovesDistributions = aPricesMovesDistributions
	if !(aConsiderPriceMoves)
		myPricesMovesDistributions = fill(Distributions.Normal(0.0,10.0^-15),aNTradingPeriods)
	end

	# if the uncertainty structure does not consider the forecast updates, then we initialise the forecast updates distributions to a zero variance distribution.
	myForecastUpdatesDistributions = aForecastUpdatesDistributions
	if !(aConsiderForecastUpdates)
		myForecastUpdatesDistributions = fill(Distributions.Normal(0.0,10.0^-15),aNTradingPeriods)
	end

	return UncertaintyStructure(
				    aNTradingPeriods,
				    aConsiderPriceMoves,
				    myPricesMovesDistributions,
				    aConsiderForecastUpdates,
				    aInitialDemandForecast,
				    myForecastUpdatesDistributions,
				    aScenarioApproximation,
				    aProbExtremeInRays,
				    aExtremeRays,
				    aNExtremeRays,
				    aRaysProbabilities,
				    aDistEstimateAlongRays,
				    aBlackBoxFunctionGeneratingUncertainty
				    )
end

"""
get_new_uncertainty_structure()

TODO function description.

#### Argument
* ``: TODO.
"""
function get_new_uncertainty_structure(
				       aUncertaintyStructure;
				       aNTradingPeriods::Int=get_n_trading_periods(aUncertaintyStructure),
				       aConsiderPriceMoves::Bool=get_consider_price_moves(aUncertaintyStructure),
				       aPricesMovesDistributions=get_prices_moves_distributions(aUncertaintyStructure),
				       aConsiderForecastUpdates::Bool=get_consider_forecast_updates(aUncertaintyStructure),
				       aInitialDemandForecast::Float64=get_initial_demand_forecast(aUncertaintyStructure),
				       aForecastUpdatesDistributions=get_forecast_updates_distributions(aUncertaintyStructure),
				       aScenarioApproximation=get_scenario_approximation(aUncertaintyStructure),
				       aProbExtremeInRays::Float64=get_prob_extreme_in_rays(aUncertaintyStructure),
				       aExtremeRays=get_extreme_rays(aUncertaintyStructure),
				       aNExtremeRays=get_n_extreme_rays(aUncertaintyStructure),
				       aRaysProbabilities=get_rays_probabilities(aUncertaintyStructure),
				       aDistEstimateAlongRays=get_dist_estimate_along_rays(aUncertaintyStructure),
				       aBlackBoxFunctionGeneratingUncertainty=get_black_box_function_generating_uncertainty(aUncertaintyStructure)
				       )

	return UncertaintyStructure(
				    aNTradingPeriods,
				    aConsiderPriceMoves,
				    aPricesMovesDistributions,
				    aConsiderForecastUpdates,
				    aInitialDemandForecast,
				    aForecastUpdatesDistributions,
				    aScenarioApproximation,
				    aProbExtremeInRays,
				    aExtremeRays,
				    aNExtremeRays,
				    aRaysProbabilities,
				    aDistEstimateAlongRays,
				    aBlackBoxFunctionGeneratingUncertainty
				    )
end

"""
get_uncertainty_structure_from_dict(aDict)

TODO function description.

#### Argument
* `aDict`: TODO.
"""
function get_uncertainty_structure_from_dict(aDict::Dict)

	# checks if the uncertainty structure has a black box function
	if aDict["BlackBoxFunctionGeneratingUncertainty"]
		@warn(
		      string(
			     "\nUncertaintyStructureModule_WARNING 102:\n",
			     "The uncertainty structure that has been saved in this dictionary used a black box function. The exact uncertainty structure can therefore not be completely retrieved." 
			     )
		      )
	end

	return UncertaintyStructure(
				    aDict["NTradingPeriods"],
				    aDict["ConsiderPriceMoves"],
				    aDict["PricesMovesDistributions"],
				    aDict["ConsiderForecastUpdates"],
				    aDict["InitialDemandForecast"],
				    aDict["ForecastUpdatesDistributions"],
				    aDict["ScenarioApproximation"],
				    aDict["ProbExtremeInRays"],
				    aDict["ExtremeRays"],
				    aDict["NExtremeRays"],
				    aDict["RaysProbabilities"],
				    aDict["DistEstimateAlongRays"]
				    )
end

"""
get_dict_from_uncertainty_structure(aUncertaintyStructure::UncertaintyStructure)

TODO function description.

#### Argument
* `aUncertaintyStructure::UncertaintyStructure`: TODO.
"""
function get_dict_from_uncertainty_structure(aUncertaintyStructure::UncertaintyStructure)

	myDict = Dict()
	myDict["TradingPeriods"]                        = get_n_trading_periods(aUncertaintyStructure)
	myDict["ConsiderPriceMoves"]                    = get_consider_price_moves(aUncertaintyStructure) ? 1 : 0
	myDict["PricesMovesDistributions"]              = get_prices_moves_distributions(aUncertaintyStructure)
	myDict["ConsiderForecastUpdates"]               = get_consider_forecast_updates(aUncertaintyStructure) ? 1 : 0
	myDict["InitialDemandForecast"]                 = get_initial_demand_forecast(aUncertaintyStructure)
	myDict["ForecastUpdatesDistributions"]          = get_forecast_updates_distributions(aUncertaintyStructure)
	myDict["ScenarioApproximation"]                 = get_scenario_approximation(aUncertaintyStructure) ? 0 : 1
	myDict["ProbExtremeInRays"]                     = get_prob_extreme_in_rays(aUncertaintyStructure)
	myDict["ExtremeRays"]                           = get_extreme_rays(aUncertaintyStructure)
	myDict["NExtremeRays"]                          = get_n_extreme_rays(aUncertaintyStructure)
	myDict["RaysProbabilities"]                     = get_rays_probabilities(aUncertaintyStructure)
	myDict["DistEstimateAlongRays"]                 = get_dist_estimate_along_rays(aUncertaintyStructure)
	myDict["BlackBoxFunctionGeneratingUncertainty"] = get_black_box_function_generating_uncertainty(aUncertaintyStructure) == -1 ? 0 : 1

	# checks if the uncertainty structure has a black box function
	if get_black_box_function_generating_uncertainty(aUncertaintyStructure) != -1
		@warn(
		      string(
			     "\nUncertaintyStructureModule_WARNING 103:\n",
			     "The uncertainty structure that has to saved in a dictionary a black box function, which cannot be done. The exact uncertainty structure can therefore not be completely saved, the black box function being ignored." 
			     )
		      )
	end

	return myDict

end

"""
```
hash_uncertainty_structure(aUncertaintyStructure::UncertaintyStructure)
```

returns a hash of the `UncertaintyStructure` structure.
This function is used to name the file that store the results of a simulation.

#### Argument
* `aUncertaintyStructure::UncertaintyStructure`: TODO.
"""
function hash_uncertainty_structure(aUncertaintyStructure::UncertaintyStructure)

	myDictionary = get_dict_from_uncertainty_structure(aUncertaintyStructure)

	myPricesMovesDistributions = myDictionary["PricesMovesDistributions"]
	myDictionary["PricesMovesDistributionsHashes"] = Array{UInt64,1}(undef,length(myPricesMovesDistributions))
	for i = 1:length(myPricesMovesDistributions)
		myDist = myPricesMovesDistributions[i]
		x = range(mean(myDist), stop=(mean(myDist)+std(myDist)), length=100)
		myDictionary["PricesMovesDistributionsHashes"][i] = typeof(myDist) <: UnivariateDistribution ?  hash(pdf.(myDist,x)) : hash(pdf(myDist,x))
	end
	delete!(myDictionary,"PricesMovesDistributions") # the hash of a kde is not stable

	myForecastUpdatesDistributions = myDictionary["ForecastUpdatesDistributions"]
	myDictionary["ForecastUpdatesDistributionsHashes"] = Array{UInt64,1}(undef,length(myForecastUpdatesDistributions))
	for i = 1:length(myForecastUpdatesDistributions)
		myDist = myForecastUpdatesDistributions[i]
		x = range(mean(myDist), stop=(mean(myDist)+std(myDist)), length=100)
		myDictionary["ForecastUpdatesDistributionsHashes"][i] = typeof(myDist) <: UnivariateDistribution ?  hash(pdf.(myDist,x)) : hash(pdf(myDist,x))
	end
	delete!(myDictionary,"ForecastUpdatesDistributions") # the hash of a kde is not stable

	myDistEstimateAlongRays = myDictionary["DistEstimateAlongRays"]
	myDictionary["DistEstimateAlongRaysHashes"] = Array{UInt64,1}(undef,length(myDistEstimateAlongRays))
	for i = 1:length(myDistEstimateAlongRays)
		myDist = myDistEstimateAlongRays[i]
		x = range(mean(myDist), stop=(mean(myDist)+std(myDist)), length=100)
		myDictionary["DistEstimateAlongRaysHashes"][i] = typeof(myDist) <: UnivariateDistribution ?  hash(pdf.(myDist,x)) : hash(pdf(myDist,x))
	end
	delete!(myDictionary,"DistEstimateAlongRays") # the hash of a kde is not stable

	myBlackBoxFunctionGeneratingUncertainty = get_black_box_function_generating_uncertainty(aUncertaintyStructure)
	myDictionary["BlackBoxFunctionGeneratingUncertaintyHash"] = hash(-1)
	if myBlackBoxFunctionGeneratingUncertainty != -1

		myNTradingPeriods = get_n_trading_periods(aUncertaintyStructure)
		mySimulationParameters = get_simulation_parameters(aNTradingPeriods=myNTradingPeriods)

		# initialises the vectors that will contain the realisations
		myBBNSamples = 10^2
		myPriceMovesRealisations      = zeros(myBBNSamples,myNTradingPeriods)
		myForecastUpdatesRealisations = zeros(myBBNSamples,myNTradingPeriods)

		if get_uncertainty_generation_method(aSimulationParameters) == "bbfunction"
			myBlackBoxFunctionGeneratingUncertainty = get_black_box_function_generating_uncertainty(myUncertaintyStructure)
			myBlackBoxFunctionGeneratingUncertainty!(
								 aSimulationParameters        = mySimulationParameters,
								 aNSamples                    = myBBNSamples,
								 aPriceMovesRealisations      = myPriceMovesRealisations,
								 aForecastUpdatesRealisations = myForecastUpdatesRealisations
								 )

			myDictionary["BlackBoxFunctionGeneratingUncertaintyHash"] = string( hash(myPriceMovesRealisations) , hash(myForecastUpdatesRealisations) )
		end
	end

	return hash(myDictionary)
end

#########################################################################
# Functions to generate realisations of the underlying random variables #
#########################################################################

function generate_uncertainty_realisations_normal_distributions!(;
								 aUncertaintyStructure::UncertaintyStructure,
								 aNSamples::Int,
								 aPriceMovesRealisations::Array{Float64,2},
								 aForecastUpdatesRealisations::Array{Float64,2}
								 )

	# gets the number of trading periods
	myNTradingPeriods = get_n_trading_periods(aUncertaintyStructure)

	# generates random realisations of the forecast updates
	myForecastUpdatesDistributions = get_forecast_updates_distributions(aUncertaintyStructure)
	if get_consider_forecast_updates(aUncertaintyStructure)
		for myPeriod in 1:myNTradingPeriods
			aForecastUpdatesRealisations[:,myPeriod] = rand(myForecastUpdatesDistributions[myPeriod],aNSamples)
		end
	end

	# generates random realisations of the price moves
	myPricesMovesDistributions = get_prices_moves_distributions(aUncertaintyStructure)
	if get_consider_price_moves(aUncertaintyStructure)
		for myPeriod in 1:myNTradingPeriods
			aPriceMovesRealisations[:,myPeriod] = rand(myPricesMovesDistributions[myPeriod],aNSamples)
		end
	end
end

function generate_uncertainty_realisations_rays!(;
						 aUncertaintyStructure::UncertaintyStructure,
						 aNSamples::Int,
						 aPriceMovesRealisations::Array{Float64,2},
						 aForecastUpdatesRealisations::Array{Float64,2}
						 )

	# gets the number of trading periods
	myNTradingPeriods = get_n_trading_periods(aUncertaintyStructure)

	# gets details of the uncertainty struture of the underlying random variables
	myNExtremeRays = get_n_extreme_rays(aUncertaintyStructure)
	myExtremeRays = get_extreme_rays(aUncertaintyStructure)
	myProbExtremeInRays = get_prob_extreme_in_rays(aUncertaintyStructure)
	myExtremeRaysPriceMoves = [transpose(x[1:myNTradingPeriods]) for x in myExtremeRays]
	myExtremeRaysForecastUpdates = [transpose(x[myNTradingPeriods+1:end]) for x in myExtremeRays]
	myRaysProbabilities = get_rays_probabilities(aUncertaintyStructure)
	myDistEstimateAlongRays = get_dist_estimate_along_rays(aUncertaintyStructure)
	myNSamplesInExtremeSituation = Int(floor(myProbExtremeInRays*aNSamples))

	####################################################
	# Generate realisations in the `normal` situtation #
	####################################################

	# generates random realisations of the price moves
	myPricesMovesDistributions = get_prices_moves_distributions(aUncertaintyStructure)
	if get_consider_price_moves(aUncertaintyStructure)
		for myPeriod in 1:myNTradingPeriods
			aPriceMovesRealisations[myNSamplesInExtremeSituation+1:end,myPeriod] = rand(myPricesMovesDistributions[myPeriod],aNSamples-myNSamplesInExtremeSituation)
		end
	end

	# generates random realisations of the forecast updates
	myForecastUpdatesDistributions = get_forecast_updates_distributions(aUncertaintyStructure)
	if get_consider_forecast_updates(aUncertaintyStructure)
		for myPeriod in 1:myNTradingPeriods
			aForecastUpdatesRealisations[myNSamplesInExtremeSituation+1:end,myPeriod] = rand(myForecastUpdatesDistributions[myPeriod],aNSamples-myNSamplesInExtremeSituation)
		end
	end

	############################################################
	# Generate realisations in the `extreme` situtation (Rays) #
	############################################################

	myNSamplesFilled = 0
	for i in 1:myNExtremeRays-1
		myNSamplesInRayI = Int(floor(myRaysProbabilities[i]*myNSamplesInExtremeSituation))
		myScalarRayUncertaintyRealisations = rand(myDistEstimateAlongRays[i],myNSamplesInRayI)
		if get_consider_price_moves(aUncertaintyStructure)
			aPriceMovesRealisations[myNSamplesFilled+1:myNSamplesFilled+myNSamplesInRayI,:] = myScalarRayUncertaintyRealisations .* myExtremeRaysPriceMoves[i] # TODO: no working is KDE
		end
		if get_consider_forecast_updates(aUncertaintyStructure)
			aForecastUpdatesRealisations[myNSamplesFilled+1:myNSamplesFilled+myNSamplesInRayI,:] = myScalarRayUncertaintyRealisations .* myExtremeRaysForecastUpdates[i] # TODO: no working is KDE
		end
		myNSamplesFilled += myNSamplesInRayI
	end

	# Last ray
	myNSamplesInLastRay = myNSamplesInExtremeSituation-myNSamplesFilled
	myScalarRayUncertaintyRealisations = rand(myDistEstimateAlongRays[end],myNSamplesInLastRay)
	if get_consider_price_moves(aUncertaintyStructure)
		aPriceMovesRealisations[myNSamplesFilled+1:myNSamplesFilled+myNSamplesInLastRay,:] = myScalarRayUncertaintyRealisations .* myExtremeRaysPriceMoves[end] # TODO: no working is KDE
	end
	if get_consider_forecast_updates(aUncertaintyStructure)
		aForecastUpdatesRealisations[myNSamplesFilled+1:myNSamplesFilled+myNSamplesInLastRay,:] = myScalarRayUncertaintyRealisations .* myExtremeRaysForecastUpdates[end] # TODO: no working is KDE
	end
end

"""
generate_uncertainty_realisations!()
TODO function description.
#### Argument
* `aMarketDetails::MarketDetails`: TODO.
"""
function generate_uncertainty_realisations!(;
					    aSimulationParameters::SimulationParameters,
					    aUncertaintyStructure::UncertaintyStructure,
					    aNSamples::Int,
					    aPriceMovesRealisations::Array{Float64,2},
					    aForecastUpdatesRealisations::Array{Float64,2}
					    )

	if get_uncertainty_generation_method(aSimulationParameters) == "bbfunction"
		myBlackBoxFunctionGeneratingUncertainty = get_black_box_function_generating_uncertainty(myUncertaintyStructure)
		myBlackBoxFunctionGeneratingUncertainty!(
							 aSimulationParameters        = aSimulationParameters,
							 aNSamples                    = aNSamples,
							 aPriceMovesRealisations      = aPriceMovesRealisations,
							 aForecastUpdatesRealisations = aForecastUpdatesRealisations
							 )
	elseif get_uncertainty_generation_method(aSimulationParameters) == "rays"
		generate_uncertainty_realisations_rays!(
							aUncertaintyStructure        = aUncertaintyStructure,
							aNSamples                    = aNSamples,
							aPriceMovesRealisations      = aPriceMovesRealisations,
							aForecastUpdatesRealisations = aForecastUpdatesRealisations
							)
	elseif get_uncertainty_generation_method(aSimulationParameters) == "normal"
		generate_uncertainty_realisations_normal_distributions!(
									aUncertaintyStructure        = aUncertaintyStructure,
									aNSamples                    = aNSamples,
									aPriceMovesRealisations      = aPriceMovesRealisations,
									aForecastUpdatesRealisations = aForecastUpdatesRealisations
									)
	else
		@error(
		       string(
			      "\nUncertaintyStructureModule 103:\n",
			      "The generation of random realisations failed as the method ask for generating random samples is unknown." 
			      )
		       )
	end

	#= # transposes the realisations such that the each realisation corresponds to one column of `myForecastUpdatesRealisations` or `myPriceMovesRealisations` =#
	#= if !(aEachRealisationIsARow) =#
	#= 	return Array{Float64,2}(transpose(aForecastUpdatesRealisations)), Array{Float64,2}(transpose(aPriceMovesRealisations)) =#
	#= end =#
end

end
