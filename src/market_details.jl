# NAME: market_details.jl
# AUTHOR: Julien Vaes
# DATE: May 17, 2019
# DESCRIPTION: A module containing all details of the market.

module MarketDetailsModule

###################
## Load Packages ##
###################

import Base: hash, rand
using Distributions, StatsBase
using LinearAlgebra

############################
## Load Personal Packages ##
############################

using ..UncertaintyStructureModule

######################
## Export functions ##
######################

export MarketDetails
export get_n_trading_periods, get_n_traders
export get_taus, get_epsilons, get_etas, get_gammas, get_traders_uncertainty_structure
export get_cost_estimate_method
export get_market_details, get_new_market_details
export get_dict_from_market_details
export get_M_matrix_quadratic_part_trading_cost
export get_satisfy_sufficient_condition_for_uniqueness
export hash_market_details, hash_partial_market_details

######################
## Module variables ##
######################

# the different methods to estimate the trading cost
ourCostEstimateMethods = [
			  "All",         # the global trading cost:                                                                          ∑i ni * ( S0 + ∑k=1^i-1 ξk + γk nk ) +  Σi (ηi/τi) ni^2,             with ∑i ni = DT
			  "All - S0*D0", # the global trading cost minus the fixed cost S0 * D0:                                             ∑i ni * ( S0 + ∑k=1^i-1 ξk + γk nk ) +  Σi (ηi/τi) ni^2 - S0 * D0,   with ∑i ni = DT
			  "All - S0*DT", # the global trading cost minus the cost related to the initial price and the final demand S0 * DT: ∑i ni * ( S0 + ∑k=1^i-1 ξk + γk nk ) +  Σi (ηi/τi) ni^2 - S0 * DT,   with ∑i ni = DT
			  "All - [ S0*D0 + ∑i δi * ( S0 + ∑k=1^i-1 (ξk + γk nk) ) ]" # the global trading cost minus the cost related to the initial price and the cost due to the volume updates given the price at the start of the trading period where there is the update: ∑i (ni - δi) * ( S0 + ∑k=1^i-1 ξk + γk nk ) +  Σi (ηi/τi) ni^2 - S0 * DT,   with ∑i ni = DT (this is used to compare with the dynamic program version)
			  ]

######################
## Module functions ##
######################

### START: STRUCTURE MarketDetails ###

# A structure containing all details for a structurename.
# The attributes in the structure: theNTradingPeriods, theNTraders, theTaus, theGammas, theEpsilons, theEtas, theCostEstimateMethod, theTradersUncertaintyStructure
struct MarketDetails 
	theNTradingPeriods                                             # the number of trading periods
	theNTraders                                                    # the number of traders
	theTaus 	                                               # an array with the length of each trading period.
	theGammas                                                      # an array with the value of the linear permanent impact parameter for each trading period.
	theEpsilons 	                                               # an array with the value of the fixed temporary impact parameter ϵ for each trading period, i.e. half the bid-ask spread + fixed trading cost for each trading period.
	theEtas 	                                               # an array with the value of the linear temporary impact parameter η for each trading period.
	theCostEstimateMethod 	                                       # the method to use to estimat the trading cost. The options are "All", "All - S0*D0", "All - S0*DT", or "All - [ S0*D0 + ∑i δi * ( S0 + ∑k=1^i-1 (ξk + γk nk) ) ]".
	theTradersUncertaintyStructure::Array{UncertaintyStructure,1}  # the structure containing the details on the uncertainty structure of each trader.
end

## STRUCTURE MarketDetails: get functions
"""
```
get_n_trading_periods(aMarketDetails::MarketDetails)
```

returns the attribute `theNTradingPeriods` of the structure `aMarketDetails`.

### Argument
* `aMarketDetails::MarketDetails`: TODO.
"""
function get_n_trading_periods(aMarketDetails::MarketDetails)
	return aMarketDetails.theNTradingPeriods
end

"""
```
get_n_traders(aMarketDetails::MarketDetails)
```

returns the attribute `theNTraders` of the structure `aMarketDetails`.

### Argument
* `aMarketDetails::MarketDetails`: TODO.
"""
function get_n_traders(aMarketDetails::MarketDetails)
	return aMarketDetails.theNTraders
end

"""
```
get_taus(aMarketDetails::MarketDetails)
```

returns the attribute `theTaus` of the structure `aMarketDetails`.

### Argument
* `aMarketDetails::MarketDetails`: TODO.
"""
function get_taus(aMarketDetails::MarketDetails)
	return aMarketDetails.theTaus
end

"""
```
get_gammas(aMarketDetails::MarketDetails)
```

returns the attribute `theGammas` of the structure `aMarketDetails`.

### Argument
* `aMarketDetails::MarketDetails`: TODO.
"""
function get_gammas(aMarketDetails::MarketDetails)
	return aMarketDetails.theGammas
end

"""
```
get_epsilons(aMarketDetails::MarketDetails)
```

returns the attribute `theEpsilons` of the structure `aMarketDetails`.

### Argument
* `aMarketDetails::MarketDetails`: TODO.
"""
function get_epsilons(aMarketDetails::MarketDetails)
	return aMarketDetails.theEpsilons
end

"""
```
get_etas(aMarketDetails::MarketDetails)
```

returns the attribute `theEtas` of the structure `aMarketDetails`.

### Argument
* `aMarketDetails::MarketDetails`: TODO.
"""
function get_etas(aMarketDetails::MarketDetails)
	return aMarketDetails.theEtas
end

"""
```
get_cost_estimate_method(aMarketDetails::MarketDetails)
```

returns the attribute `theCostEstimateMethod` of the structure `aMarketDetails`.

### Argument
* `aMarketDetails::MarketDetails`: TODO.
"""
function get_cost_estimate_method(aMarketDetails::MarketDetails)
	return aMarketDetails.theCostEstimateMethod
end

"""
```
get_traders_uncertainty_structure(aMarketDetails::MarketDetails)
```

returns the attribute `theTradersUncertaintyStructure` of the structure `aMarketDetails`.

### Argument
* `aMarketDetails::MarketDetails`: TODO.
"""
function get_traders_uncertainty_structure(aMarketDetails::MarketDetails)
	return aMarketDetails.theTradersUncertaintyStructure
end

### END: STRUCTURE MarketDetails ###

"""
```
get_market_details(;
aTaus::Array{Float64,1}=[-1.0],
aEpsilons::Array{Float64,1}=[-1.0],
aEtas::Array{Float64,1}=[-1.0],
aGammas::Float64=Array{Float64,1}=[-1.0],
aUncertaintyStructure=UncertaintyStructure.get_uncertainty_structure()
)
```

returns the structure corresponding to a trading market after verifying its consistency.

#### Arguments
* `aTaus::Array{Float64,1}`: an array with the length of each trading period.
* `aEpsilons::Array{Float64,1}=Array{Float64,1}()`: the coefficients in the temporary impact corresponding to half the bid spread and fixed trading cost for each trading period.
* `aEtas::Array{Float64,1}=Array{Float64,1}()`: the coefficients in the temporary impact corresponding to the variable trading cost for each trading period.
* `aGammas::Float64=Array{Float64,1}()`: the coefficient in the linear permanent impact.
* `aForecastUpdatesDistributions=fill(Normal(-10^8,1),length(aTaus))`: the standard deviation of the demand forecast error for each trading period.
* `aPricesDistributions=fill(Normal(-10^8,1),length(aTaus))`: the standard deviations of the price for each trading period.
"""
function get_market_details(;
			    aNTradingPeriods::Int=0,
			    aNTraders::Int=0,
			    aTaus::Array{Float64,1}=Array{Float64,1}(),
			    aEpsilons::Array{Float64,1}=Array{Float64,1}(),
			    aEtas::Array{Float64,1}=Array{Float64,1}(),
			    aGammas::Array{Float64,1}=Array{Float64,1}(),
			    aCostEstimateMethod::String = "All - S0*DT",
			    aTradersUncertaintyStructure::Array{UncertaintyStructure,1}=[UncertaintyStructureModule.get_uncertainty_structure()]
			    )

	# errors checks #

	if aNTradingPeriods == 0
		@error(string(
			      "\n MarketDetailsModule 101:\n",
			      "The number of trading periods must be positive. For the moment `aNTradingPeriods` = ",
			      aNTradingPeriods,
			      "."
			      )
		       )
	end

	if aNTraders == 0
		@error(string(
			      "\n MarketDetailsModule 102:\n",
			      "The number of traders must be positive. For the moment `aNTraders` = ",
			      aNTraders,
			      "."
			      )
		       )
	end

	if aNTraders != size(aTradersUncertaintyStructure,1)
		@error(string(
			      "\n MarketDetailsModule 103:\n",
			      "The number of elements in the array `aTradersUncertaintyStructure` should be equal to the number of traders, i.e. `aNTraders`.",
			      "We have got respectively ",size(aTradersUncertaintyStructure,1)," and ",aNTraders,"."
			      )
		       )
	end

	if aNTradingPeriods != size(aTaus,1)
		@error(string(
			      "\n MarketDetailsModule 104:\n",
			      "The number of elements in the array `aTaus` should be equal to the number of trading periods, i.e. `aNTradingPeriods`.",
			      "We have got respectively ",size(aTaus,1)," and ",aNTradingPeriods,"."
			      )
		       )
	end

	if aNTradingPeriods != size(aEpsilons,1)
		@error(string(
			      "\n MarketDetailsModule 105:\n",
			      "The number of elements in the array `aEpsilons` should be equal to the number of trading periods, i.e. `aNTradingPeriods`.",
			      "We have got respectively ",size(aEpsilons,1)," and ",aNTradingPeriods,"."
			      )
		       )
	end

	if aNTradingPeriods != size(aEtas,1)
		@error(string(
			      "\n MarketDetailsModule 106:\n",
			      "The number of elements in the array `aEtas` should be equal to the number of trading periods, i.e. `aNTradingPeriods`.",
			      "We have got respectively ",size(aEtas,1)," and ",aNTradingPeriods,"."
			      )
		       )
	end

	if !(in(aCostEstimateMethod,ourCostEstimateMethods))
		@error(string(
			      "\n MarketDetailsModule 107:\n",
			      "The method ", aCostEstimateMethod ," to estimate the trading cost in unknown.",
			      " Please use one of the following methods:\n",
			      "\t 1) \"All\",\nthe global trading cost, i.e.\n\t\t∑i ni * ( S0 + ∑k ξk + γk nk ) +  Σi (ηi/τi) ni^2, with ∑i ni = DT\n",
			      "\t 2) \"All - S0*D0\",\nthe global trading cost minus the fixed cost S0 * D0, i.e.\n\t\t∑i ni * ( S0 + ∑k ξk + γk nk ) +  Σi (ηi/τi) ni^2 - S0 * D0, with ∑i ni = DT\n",
			      "\t 3) \"All - S0*DT\",\nthe global trading cost minus the cost related to the initial price and the final demand S0 * DT, i.e.\n\t\t∑i ni * ( S0 + ∑k ξk + γk nk ) +  Σi (ηi/τi) ni^2 - S0 * DT, with ∑i ni = DT",
			      "\t 4) \"All - [ S0*D0 + ∑i δi * ( S0 + ∑k=1^i-1 (ξk + γk nk) ) ]\",\nthe global trading cost minus the cost related to the initial price and the cost due to the volume updates given the price at the start of the trading period where there is the update, i.e.\n\t\tS0 * DT: ∑i (ni - δi) * ( S0 + ∑k=1^i-1 ξk + γk nk ) +  Σi (ηi/τi) ni^2 - S0 * DT, with ∑i ni = DT"
			      )
		       )
	end

	# creates the structure
	return MarketDetails(
			     aNTradingPeriods,
			     aNTraders,
			     aTaus,
			     aGammas,
			     aEpsilons,
			     aEtas,
			     aCostEstimateMethod,
			     aTradersUncertaintyStructure
			     )
end

"""
```
get_new_market_details(
aMarketDetails::MarketDetails; 
aTaus = get_taus(aMarketDetails),
aEpsilons = get_epsilons(aMarketDetails),
aEtas = get_etas(aMarketDetails),
aGammas = get_gammas(aMarketDetails),
aForecastUpdatesDistributions = get_forecast_updates_distributions(aMarketDetails),
aPricesDistributions = get_prices_distributions(aMarketDetails)
)
```

returns a `MarketDetails` structure with the same details as `aMarketDetails` except for the ones explicitely given in the argument.

#### Arguments
"""
function get_new_market_details(
				aMarketDetails::MarketDetails; 
				aNTradingPeriods::Int = MarketDetailsModule.get_n_trading_periods(aMarketDetails),
				aNTraders::Int = get_n_traders(aMarketDetails),
				aTaus = get_taus(aMarketDetails),
				aEpsilons = get_epsilons(aMarketDetails),
				aEtas = get_etas(aMarketDetails),
				aGammas = get_gammas(aMarketDetails),
				aCostEstimateMethod = get_cost_estimate_method(aMarketDetails),
				aTradersUncertaintyStructure::Array{UncertaintyStructure,1} = get_traders_uncertainty_structure(aMarketDetails)
				)

	myNewMarketDetails = get_market_details(
						aNTradingPeriods             = aNTradingPeriods,
						aNTraders                    = aNTraders,
						aTaus                        = aTaus,
						aEpsilons                    = aEpsilons,
						aEtas                        = aEtas,
						aGammas                      = aGammas,
						aCostEstimateMethod          = aCostEstimateMethod,
						aTradersUncertaintyStructure = aTradersUncertaintyStructure
						)

	return myNewMarketDetails
end

"""
```
get_dict_from_market_details(aMarketDetails::MarketDetails)
```

returns a dictionary that contains all the informations of the market details structure `aMarketDetails`.
This function is used to store a `MarketDetails` structure into a file.
This function is mainly useful to store results of simulations and load them back again.

#### Argument
* `aMarketDetails::MarketDetails`: a structure containing all the details of the market.
"""
function get_dict_from_market_details(aMarketDetails::MarketDetails)
	myDict = Dict()
	myDict["NTradingPeriods"]      = MarketDetailsModule.get_n_trading_periods(aMarketDetails)
	myDict["NTraders"]             = get_n_traders(aMarketDetails)
	myDict["Taus"]                 = get_taus(aMarketDetails)
	myDict["Gammas"]               = get_gammas(aMarketDetails)
	myDict["Epsilons"]             = get_epsilons(aMarketDetails)
	myDict["Etas"]                 = get_etas(aMarketDetails)
	myDict["CostEstimateMethod"]   = get_cost_estimate_method(aMarketDetails)

	myTradersUncertaintyStructure = get_traders_uncertainty_structure(aMarketDetails)
	myTradersUncertaintyStructureDict = Array{Dict,1}(undef,get_n_traders(aMarketDetails))
	for p in eachindex(myTradersUncertaintyStructure)
		myTradersUncertaintyStructureDict[p] = get_dict_from_uncertainty_structure(myTradersUncertaintyStructure[p])
	end
	myDict["TradersUncertaintyStructure"] = myTradersUncertaintyStructureDict

	return myDict
end

"""
```
get_M_matrix_quadratic_part_trading_cost(aMarketDetails::MarketDetails)
```

returns the matrix M presented of Lemma in Vaes & Hauser paper on optimal trading with an uncertain trading target.

### Argument
* `aMarketDetails::MarketDetails`: TODO.
"""
function get_M_matrix_quadratic_part_trading_cost(aMarketDetails::MarketDetails)

	myNTradingPeriods = get_n_trading_periods(aMarketDetails)
	myGammas = get_gammas(aMarketDetails)
	myEtas = get_etas(aMarketDetails)

	myMatrixM  = diagm(2*myEtas[1:end-1])
	myMatrixM += fill(2*myEtas[end],(myNTradingPeriods-1,myNTradingPeriods-1))

	for i = 1:myNTradingPeriods-1, j = 1:myNTradingPeriods-1
		if i==j
			myMatrixM[i,j] -= 2*myGammas[i]
		elseif i < j
			myMatrixM[i,j] -= myGammas[j]
		elseif i > j
			myMatrixM[i,j] -= myGammas[i]
		end
	end

	return myMatrixM
end

function get_satisfy_sufficient_condition_for_uniqueness(aMarketDetails::MarketDetails)
	return isposdef(get_M_matrix_quadratic_part_trading_cost(aMarketDetails))
end

"""
```
hash_market_details(aMarketDetails::MarketDetails)
```

returns a hash of the MarketDetails structure, i.e. with the actual distribution of the prices moves and forecast updates.

#### Argument
* `aMarketDetails::MarketDetails`: a structure containing all the details of the market.
"""
function hash_market_details(
			     aMarketDetails::MarketDetails;
			     aInsertTradersUncertaintyStructure::Bool=true
			     )

	myDictionary = get_dict_from_market_details(aMarketDetails)

	# computes the hash of the `myTradersUncertaintyStructureHash` if needed
	if aInsertTradersUncertaintyStructure
		myTradersUncertaintyStructure = get_traders_uncertainty_structure(aMarketDetails)

		myTradersUncertaintyStructureHash = ""
		for myLocalUncertaintyStructure in myTradersUncertaintyStructure
			myTradersUncertaintyStructureHash = string(myTradersUncertaintyStructureHash, hash_uncertainty_structure(myLocalUncertaintyStructure)) 
		end
		myDictionary["HashTradersUncertaintyStructure"] = hash(myTradersUncertaintyStructureHash)
	end

	delete!(myDictionary,"TradersUncertaintyStructure")

	return hash(myDictionary)
end

"""
```
hash_partial_market_details(aMarketDetails::MarketDetails)
```

returns a hash of the `MarketDetails` structure when the actual distribution of the prices moves and forecast updates are ignored.
This function is used to name the file that store the results of a simulation.

#### Argument
* `aMarketDetails::MarketDetails`: a structure containing all the details of the market.
"""
function hash_partial_market_details(aMarketDetails::MarketDetails)

	return hash_market_details(
				   aMarketDetails,
				   aInsertTradersUncertaintyStructure = false
				   )
end

end
