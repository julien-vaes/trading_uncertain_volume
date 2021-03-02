# NAME: trader.jl
# AUTHOR: Julien Vaes
# DATE: May 17, 2019
# DESCRIPTION: A structure containing all the details of a trader.

module TraderModule

###################
## Load Packages ##
###################

import Base: hash, rand
using Distributions

############################
## Load Personal Packages ##
############################

using ..UncertaintyStructureModule
using ..MarketDetailsModule

######################
## Export functions ##
######################

export Trader
export get_trader_index, get_risk_aversion, get_alpha, get_market_details_belief
export get_trader, get_new_trader
export get_dict_from_trader
export hash_trader, hash_partial_trader

######################
## Module variables ##
######################

ourNTraders = 0 # module variable helpful when creating new traders

######################
## Module functions ##
######################

### START: STRUCTURE Trader ###

# A structure containing all details for a trade.
# The attributes in the structure: theTraderIndex::Int, theRiskAversion::Float64, theAlpha::Float64, theMarketDetailsBelief::MarketDetails
struct Trader 
	theTraderIndex::Int 	              # the unique index associated the trader
	theRiskAversion::Float64               # the coefficient to weight the trade off between the expectation and the CVaR, i.e. the risk averseness parameter.
	theAlpha::Float64 	              # the number of % of worst case scenarios we are hedging against, i.e. CVaR = |E( cost of worst alpha percent).
	theMarketDetailsBelief::MarketDetails # the structure containing the belief of the trader of the underlying random variable constituting the market uncertainty (price moves and forecast updates for all the traders).
end

## STRUCTURE Trader: get functions
"""
```
get_trader_index(aTrader::Trader)
```

returns the attribute `theTraderIndex` of the structure `aTrader`.

### Argument
* `aTrader::Trader`: TODO.
"""
function get_trader_index(aTrader::Trader)
	return aTrader.theTraderIndex
end

"""
```
get_risk_aversion(aTrader::Trader)
```

returns the attribute `theRiskAversion` of the structure `aTrader`.

### Argument
* `aTrader::Trader`: TODO.
"""
function get_risk_aversion(aTrader::Trader)
	return aTrader.theRiskAversion
end

"""
```
get_alpha(aTrader::Trader)
```

returns the attribute `theAlpha` of the structure `aTrader`.

### Argument
* `aTrader::Trader`: TODO.
"""
function get_alpha(aTrader::Trader)
	return aTrader.theAlpha
end

"""
```
get_market_details_belief(aTrader::Trader)
```

returns the attribute `theMarketDetailsBelief` of the structure `aTrader`.

### Argument
* `aTrader::Trader`: TODO.
"""
function get_market_details_belief(aTrader::Trader)
	return aTrader.theMarketDetailsBelief
end

### END: STRUCTURE Trader ###

"""
         get_trader(;
		    aRiskAversion::Float64=0.1,
		    aAlpha::Float64=0.1,
		    aUncertaintyStructure::UncertaintyStructure=UncertaintyStructureModule.get_uncertainty_structure(),
		    )

returns the structure corresponding to a trader after verifying its consistency.

#### Arguments
* `aRiskAversion::Float64`: TODO
* `aAlpha = get_alpha(aTrader)`: TODO
* `aUncertaintyStructure::UncertaintyStructure`: TODO
"""
function get_trader(;
		    aTraderIndex::Int = 1,
		    aRiskAversion::Float64 = 0.0,
		    aAlpha::Float64 = 0.1,
		    aMarketDetailsBelief::MarketDetails=MarketDetailsModule.get_market_details()
		    )

	# errors checks #
	
	if aRiskAversion < 0.0
		@error(
			string(
				"\nTraderModule 101:\n",
				"The risk-aversion parameter `aRiskAversion` should be between 0.0 and 1.0, whereas the value given amounts ",
				aRiskAversion,
				"."
				)
			)
	end

#= 	if aRiskAversion > 1.0 =#
#= 		@warn( =#
#= 			string( =#
#= 				"\nTraderModule 102:\n", =#
#= 				"The risk-aversion parameter `aRiskAversion` should be between 0.0 and 1.0 if the model Mean-CVaR is used, whereas the value given amounts ", =#
#= 				aRiskAversion, =#
#= 				". If hte model mean-variance is used, please ignore this warning." =#
#= 				) =#
#= 			) =#
#= 	end =#

#= 	if aAlpha > 1.0 || aAlpha < 0.0 =#
#= 		@error( =#
#= 			string( =#
#= 				"\nTraderModule 103:\n", =#
#= 				"The CVaR parameter `aAlpha` should be between 0.0 and 1.0, whereas the value given amounts ", =#
#= 				aAlpha, =#
#= 				"." =#
#= 				) =#
#= 			) =#
#= 	end =#

	# creates the structure
	return Trader(
		      aTraderIndex,
		      aRiskAversion,
		      aAlpha,
		      aMarketDetailsBelief
		      )
end

"""
         get_new_trader(
			aTrader::Trader; 
			aRiskAversion::Float64=get_risk_aversion(aTrader),
			aAlpha::Float64=get_alpha(aTrader),
			aUncertaintyStructure::UncertaintyStructure=get_uncertainty_structure(aTrader),
			)

returns a `Trader` structure exactly the same as `Trader` except for the values that are explicitely given as an argument.

#### Arguments
* `aTrader::Trader`: a structure containing all the details of the trader.
* `aRiskAversion::Float64`: TODO
* `aAlpha = get_alpha(aTrader)`: TODO
* `aUncertaintyStructure::UncertaintyStructure`: TODO
"""
function get_new_trader(
			aTrader::Trader; 
			aTraderIndex::Int=get_trader_index(aTrader),
			aRiskAversion::Float64=get_risk_aversion(aTrader),
			aAlpha::Float64=get_alpha(aTrader),
			aMarketDetailsBelief::MarketDetails=get_market_details_belief(aTrader)
			)

	myNewTrader = get_trader(
				 aTraderIndex         = aTraderIndex,
				 aRiskAversion        = aRiskAversion,
				 aAlpha               = aAlpha,
				 aMarketDetailsBelief = aMarketDetailsBelief
				 )

	return myNewTrader
end

"""
    get_trader_from_dict(aDict::Dict)

returns a Trader structure based on the details given in the dictionary `aDict`.
This function is used when loading a `Trader` structure from a file.
This function is mainly useful to store results of simulations and load them back again.

#### Argument
* `aDict::Dict`: a dictionary whosss components specify completely the trader.
"""
function get_trader_from_dict(aDict::Dict) 

	return get_trader(
			  aTraderIndex         = aDict["TraderIndex"],
			  aRiskAversion        = aDict["RiskAversion"],
			  aAlpha               = aDict["Alpha"],
			  aMarketDetailsBelief = get_market_details_from_dict(aDict["MarketDetails"])
			  )
end

"""
    get_dict_from_trader(aTrader::Trader)

returns a dictionary that contains all the informations of the trader.
This function is used to store a `Trader` structure into a file.
This function is mainly useful to store results of simulations and load them back again.

#### Argument
* `aTrader`: a structure containing all the details of a trader.
"""
function get_dict_from_trader(aTrader::Trader)
	myDict = Dict()
	myDict["TraderIndex"]   = get_trader_index(aTrader)
	myDict["RiskAversion"]  = get_risk_aversion(aTrader)
	myDict["Alpha"]         = get_alpha(aTrader)
	myDict["MarketDetails"] = MarketDetailsModule.get_dict_from_market_details(get_market_details_belief(aTrader))
	return myDict
end

"""
    hash_trader(aTrader::Trader)

returns a hash of the `Trader` structure including the uncertainty structure.
This function is used to name the file that store the results of the optimisation of the trading plan, which depends on the uncertainty structure belief of the trader.

#### Argument
* `aTrader::Trader`: a structure containing all the details of a trader.
"""
function hash_trader(
		     aTrader::Trader;
		     aInsertMarketDetailsBelief::Bool=true
		     )

	myDictionary = get_dict_from_trader(aTrader)

	# computes the hash of the `myTradersUncertaintyStructureHash` if needed
	if aInsertMarketDetailsBelief
		myDictionary["HashMarketDetailsBelief"] = hash_market_details(get_market_details_belief(aTrader))
	end
	delete!(myDictionary,"MarketDetails")

	return hash(myDictionary)
end

"""
    hash_partial_trader(aTrader::Trader)

returns a hash of the `Trader` structure.
This function is used to name the file that store the performance of a trading plan.
In that case, the uncertainty structure belief of the trader has no impact on the performance and should therefore not be considered.

#### Argument
* `aTrader::Trader`: a structure containing all the details of a trader.
"""
function hash_partial_trader(aTrader::Trader)

	return hash_trader(
			   aTrader,
			   aInsertMarketDetailsBelief = false
			   )
end

end
