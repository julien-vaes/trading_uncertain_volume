# NAME: strategy.jl
# AUTHOR: Julien Vaes
# DATE: May 17, 2019
# DESCRIPTION: A structure containing all details of a trading strategy.

module StrategyModule

######################
## Module Variables ##
######################

 # a small value under which we consider that a value is equal to 0.0 for comparision purposes
ourSmallFloat64 = 10.0^-8

######################
## Export functions ##
######################

export Strategy
export get_trading_plan, get_redistribution_matrix, get_redistribution_matrix_coeff
export get_strategy, get_new_strategy
export get_strategy_from_decision_variables, get_decision_variables_from_strategy
export get_default_initial_trading_plan, get_default_initial_redistribution_matrix, get_default_initial_strategy
export hash_strategy
export get_dict_from_strategy, get_strategy_from_dict
export get_redistribution_matrix_variables_indices

######################
## Module functions ##
######################

### START: STRUCTURE Strategy ###

# A structure containing all details for a Strategy.
# The attributes in the structure: theTradingPlan,theRedistributionMatrix,theRedistributionMatrixCoeff
struct Strategy 
	theTradingPlan 	              # the decisions on the proportions of the demand to trade at each trading period.
	theRedistributionMatrix       # the matrix containing the information on how to redistribute the forecast updates on the next trading period, i.e. element [i,j] correspond to the proportion of the forecast updates of trading period i to traded during the trading period j.
	theRedistributionMatrixCoeff  # the redistribution matrix on the trading periods due to the forecast updates corresponding to the trading plan.
end

## STRUCTURE Strategy: get functions

"""
    get_trading_plan(aStrategy::Strategy)

returns the attribute `theTradingPlan` of the structure `aStrategy`.

#### Argument
* `aStrategy::Strategy`: the structure containg the trading strategy details.
"""
function get_trading_plan(aStrategy::Strategy)
	return aStrategy.theTradingPlan
end

"""
    get_redistribution_matrix(aStrategy::Strategy)

returns the attribute `theRedistributionMatrix` of the structure `aStrategy`.

#### Argument
* `aStrategy::Strategy`: the structure containing the trading strategy details.
"""
function get_redistribution_matrix(aStrategy::Strategy)
	return aStrategy.theRedistributionMatrix
end

"""
    get_redistribution_matrix_coeff(aStrategy::Strategy)

returns the attribute `theRedistributionMatrixCoeff` of the structure `aStrategy`.

#### Argument
* `aStrategy::Strategy`: the structure containing the trading strategy details.
"""
function get_redistribution_matrix_coeff(aStrategy::Strategy)
	return aStrategy.theRedistributionMatrixCoeff
end

### END: STRUCTURE Strategy ###

"""
#### Definition
```
get_redistribution_matrix_variables_indices(aNTradingPeriods::Int)
```

TODO function description.

#### Argument
* `aNTradingPeriods::Int`: TODO.
"""
function get_redistribution_matrix_variables_indices(aNTradingPeriods::Int)
	# Indices of the variables we optimise on for the redistribution matrix
	myRedistributionMatrixIndices = []
	for k in 1:aNTradingPeriods-1
		for i in k+1:aNTradingPeriods-1
			push!(myRedistributionMatrixIndices,k+(i-1)*aNTradingPeriods)
		end
	end
	return myRedistributionMatrixIndices
end

"""
         compute_redistribution_coeff_matrix(aTradingPlan, aRedistributionMatrix)

returns the redistribution matrix coefficients M given the investment decisions of `aTradingPlan` and the redistribution rule `aRedistributionMatrix`.
We have that n_i = y_i D0 + sum_k{δ_k*[y_i+β_k,i*(sum_r y_r)]}, and we compute M_i,k = [y_i+β_k,i*(sum_r y_r)]

#### Arguments
* `aTradingPlan::Array{Float64}`: a trading plan, i.e. the proportion of the demand to trade at each trading period.
* `aRedistributionMatrix`: a matrix containing the details on how to redistribute the forecast updates.
"""
function compute_redistribution_coeff_matrix(aTradingPlan, aRedistributionMatrix)
	mySize = size(aRedistributionMatrix)
	myRedistributionMatrixCoeff = zeros(mySize)

	for i = 1:mySize[1], j=i+1:mySize[2]
		myRedistributionMatrixCoeff[i,j] = aTradingPlan[j] + aRedistributionMatrix[i,j]*sum(aTradingPlan[r] for r=1:i)
	end

	return myRedistributionMatrixCoeff
end

"""
         get_strategy(
		      aTradingPlan::Array{Float64},
		      aRedistributionMatrix::Array{Float64,2},
		      aNTradingPeriods::Int64
		      )

returns the `Strategy` structure corresponding the investment decisions `aTradingPlan` and the redistribution matrix `aRedistributionMatrix`.

#### Argument
* `aTradingPlan::Array{Float64}`: a trading plan, i.e. the proportion of the demand to trade all trading periods except the last one (which is then obtained due to the contraint that the sum equals one).
* `aRedistributionMatrix::Array{Float64,2}`: a matrix containing the information on how to redistribute the forecast error on the next trading period, i.e. element [i,j] correspond to the proportion of the forecast error of trading period i to traded during the trading period j.
* `aNTradingPeriods::Int64`: the number of trading periods.
"""
function get_strategy(;
		      aNTradingPeriods::Int64=0,
		      aTradingPlan::Array{Float64}=get_default_initial_trading_plan(aNTradingPeriods),
		      aRedistributionMatrix::Array{Float64,2}=get_default_initial_redistribution_matrix(aNTradingPeriods)
		      )

	# check that the right number of elements are given in the investment decisions
	if length(aTradingPlan) != aNTradingPeriods
		@error(string(
			      "\nStrategyModule_ERROR 101:\n",
			      "The number of elements in `aTradingPlan`\n",
			      aTradingPlan,
			      "\nmust correspond to the number of trading periods, i.e. ",
			      aNTradingPeriods,
			      "."
			      )
		       )
	end

	# checks that the sum of the trading plan sums to 1.
	if @views !isapprox(sum(aTradingPlan),1.0,atol=ourSmallFloat64)
		@error(string(
			      "\nStrategyModule_ERROR 102:\n",
			      "The sum of the components of the trading plan must sum up to 1.0, which is not the case. The trading plan is:\n",
			      aTradingPlan
			      )
		       )
	end

	mySize = size(aRedistributionMatrix)
	# check that the redistribution has the right dimension
	if mySize[1] != aNTradingPeriods || mySize[2] != aNTradingPeriods
		@error(string(
			     "\nStrategyModule_ERROR 103:\n",
			     "The redistribution matrix should be a squared matrix of ", 
			     aNTradingPeriods," times ",aNTradingPeriods,
			     " and not of size  ",
			     mySize,".")
		      )
	end

	for i = 1:aNTradingPeriods
		# check that each row of the redistribution matrix sums to 1.
		if @views !isapprox(sum(aRedistributionMatrix[i,:]),1.0,atol=ourSmallFloat64) && i!=aNTradingPeriods
		@error(string(
			     "\nStrategyModule_ERROR 104:\n",
			     "The sum of the redistribution of the error on the next trading periods for the error made at trading period ",i," must sum up to 1.0, which is not the case:\n",
			     aRedistributionMatrix[i,:]
			     )
		      )
		end
		for j = 1:i 
			if aRedistributionMatrix[i,j] != 0.0
				# check that no recourse is taken on the forecast update before it actually happens.
				@error(string(
					     "\nStrategyModule_ERROR 105:\n",
					     "The error made at trading period ",i," is not allowed to be redistributed on the volume traded at trading period ",j," and should thus be equal to 0.0 instead of ",aRedistributionMatrix[i,j],"."
					     )
				      )
			end
		end
	end

	myRedistributionMatrixCoeff = compute_redistribution_coeff_matrix(aTradingPlan, aRedistributionMatrix)

	return Strategy(aTradingPlan, aRedistributionMatrix, myRedistributionMatrixCoeff)
end

function get_new_strategy(
			  aStrategy::Strategy;
			  aNTradingPeriods = size(get_trading_plan(aStrategy),1),
			  aTradingPlan = get_trading_plan(aStrategy),
			  aRedistributionMatrix = get_redistribution_matrix(aStrategy)
			  )

	return get_strategy(
			    aNTradingPeriods      = aNTradingPeriods,
			    aTradingPlan          = aTradingPlan,
			    aRedistributionMatrix = aRedistributionMatrix
			    )
end


"""
#### Definition
```
get_strategy_from_decision_variables(aTradingPlanVariables,aRedistributionMatrixVariables)
```

TODO function description.

#### Arguments
* `aTradingPlanVariables`: TODO.
* `aRedistributionMatrixVariables`: TODO.
"""
function get_strategy_from_decision_variables(;
					      aNTradingPeriods::Int,
					      aTradingPlanVariables,
					      aRedistributionMatrixVariables
					      )

	# In order to get rid of the constraints, we get rid of y_aNTradingPeriods by incorporting the constraint: sum(y_i) = 1.
	# Hence base on the trading plan variables (there are aNTradingPeriods-1 of them), one computes the last proportion based on the constraint.
	myTradingPlan = zeros(aNTradingPeriods)

	# fills the trading plan
	myTradingPlan[1:end-1] = aTradingPlanVariables

	# finishes the filling by applying the constraint: sum(y_i) = 1.
	myTradingPlan[end] = 1-sum(aTradingPlanVariables)

	# In order to get rid of the constraints, we get rid of the decisions variables concerning the last trading period by incorporting the constraint: for all k sum(β_(k,i)) = 1.
	# Hence base on the redistribution matrix variables, one computes the last proportions based on these constraints.
	
	# The indices of the redistribution matrix that are variables (non zeros and not the ones corresponding to the last period)
	myRedistributionMatrixIndices = get_redistribution_matrix_variables_indices(aNTradingPeriods)

	# fills the redistribution matrix
	myRedistributionMatrix = zeros(aNTradingPeriods,aNTradingPeriods)                            
	myRedistributionMatrix[myRedistributionMatrixIndices] = aRedistributionMatrixVariables

	# finishes the filling by applying the constraints: for all k sum(β_(k,i)) = 1.
	myRedistributionMatrix[1:end-1,end] = 1 .- sum(myRedistributionMatrix[1:end-1,1:end-1], dims=2)

	return get_strategy(
			    aNTradingPeriods=aNTradingPeriods,
			    aTradingPlan=myTradingPlan,
			    aRedistributionMatrix=myRedistributionMatrix
			    )
end

"""
#### Definition
```
get_decision_variables_from_strategy(aStrategy::Strategy)
```

TODO function description.

#### Argument
* `aStrategy::Strategy`: TODO.
"""
function get_decision_variables_from_strategy(aStrategy::Strategy)

	# gets the trading plan and redistribution matrix from `aStrategy`
	myTradingPlan = get_trading_plan(aStrategy)
	myRedistributionMatrix = get_redistribution_matrix(aStrategy)

	# The indices of the redistribution matrix that are variables (non zeros and not the ones corresponding to the last period)
	myRedistributionMatrixIndices = get_redistribution_matrix_variables_indices(length(myTradingPlan))

	return myTradingPlan[1:end-1], myRedistributionMatrix[myRedistributionMatrixIndices]
end

"""
    get_dict_from_strategy(aStrategy::Strategy)

returns a dictionary that contains all the informations of the strategy.
This function is used to store a `Strategy` structure into a file.
This function is mainly useful to store results of simulations and load them back again.

#### Argument
* `aStrategy::Strategy`: the structure containg the trading strategy details.
"""
function get_dict_from_strategy(aStrategy::Strategy)
	myDict = Dict()
	myDict["TradingPlan"] = get_trading_plan(aStrategy)
	myDict["RedistributionMatrix"] = get_redistribution_matrix(aStrategy)
	myDict["RedistributionMatrixCoeff"] = get_redistribution_matrix_coeff(aStrategy)
	return myDict
end

"""
    get_strategy_from_dict(aDict::Dict)

TODO function description.

#### Argument
* `aDict::Dict`: TODO.
"""
function get_strategy_from_dict(aDict::Dict)

	myTradingPlan = aDict["TradingPlan"]
	myRedistributionMatrix = aDict["RedistributionMatrix"]

	return get_strategy(
			    aNTradingPeriods=length(myTradingPlan),
			    aTradingPlan=myTradingPlan,
			    aRedistributionMatrix=myRedistributionMatrix
			    )

end

"""
    hash_strategy(aStrategy::Strategy)

returns a hash of the Strategy structure.
This function is used to name the file that store the results of a simulation.

#### Argument
* `aStrategy::Strategy`: a structure containing all the details of a strategy.
"""
function hash_strategy(aStrategy::Strategy)
	myDictionary = get_dict_from_strategy(aStrategy)
	return hash(myDictionary)
end

"""
    get_default_initial_trading_plan(aNTradingPeriods::Int64)

returns a default initial trading decisions, 
i.e. even repartition of the trades over all the trading periods.
Note that only aNTradingPeriods-1 elements are returned as the last component is derived based on the constraint that the sum of all investment decisions is equal to 1.

#### Argument
* `aNTradingPeriods::Int64`: a number of periods.
"""
function get_default_initial_trading_plan(aNTradingPeriods::Int64)
	return (1/aNTradingPeriods)*ones(aNTradingPeriods)
end

"""
    get_default_initial_redistribution_matrix(aNTradingPeriods::Int64)

returns the default initial redistribution matrix.
The default redistribution matrix assumes that the forecast errors are evenly redistributed on the future trading periods.

#### Argument
* `aNTradingPeriods::Int64`: a number of periods.
"""
function get_default_initial_redistribution_matrix(aNTradingPeriods::Int64)

	myDefaultRedistributionMatrix = zeros(aNTradingPeriods,aNTradingPeriods)
	for i = 1:aNTradingPeriods-1
		myDefaultRedistributionMatrix[i,i+1:end] .= 1.0/(aNTradingPeriods-i)
	end
	return myDefaultRedistributionMatrix
end


"""
```
get_default_initial_strategy(aNTradingPeriods)
```

returns the default initial strategy composed of the default initial trading plan and redistribution matrix,
uniform trading with uniform redistribution.

### Argument
* `aNTradingPeriods`: a number of trading periods.
"""
function get_default_initial_strategy(aNTradingPeriods)
	nothing
end


end
