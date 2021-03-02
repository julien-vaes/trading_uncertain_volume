# NAME: mean_var_principal_scenarios.jl
# AUTHOR: Julien Vaes
# DATE: November 25, 2019
# DESCRIPTION: Optimise the mean-VaR trade-off based on the assumption of the principal scenarios

module MeanVaRPrincipalScenariosModule

###################
## Load Packages ##
###################

using Distributions
using LinearAlgebra
using StatsBase
using SparseArrays
using KernelDensity
using Roots
using JuMP
using Gurobi
using Optim
using ForwardDiff
using SharedArrays
using Distributed

############################
## Load Personal Packages ##
############################

using ..UncertaintyStructureModule
using ..MarketDetailsModule
using ..StrategyModule
using ..TraderModule
using ..SimulationParametersModule
using ..MeanVarianceModule
using ..TradingCostModule
using ..MeanCVaRModule

######################
## Export functions ##
######################

export get_optimal_strategy_mean_VaR

######################
## Module variables ##
######################

# parameters for the convergence of the optimisation algorithm provided by Optim 
ourXTolerance = 1e-8  # absolute tolerance in changes of the input vector x, in infinity norm. Defaults to 0.0.
ourFTolerance = 1e-10 # relative tolerance in changes of the objective value. Defaults to 0.0.
ourGTolerance = 1e-10 # absolute tolerance in the gradient, in infinity norm. Defaults to 1e-8. For gradient free methods, this will control the main convergence tolerance, which is solver specific.

######################
## Module functions ##
######################

### START: STRUCTURE RaysEstimateStructure ###

# A structure containing all details for a structurename.
# The attributes in the structure: theConstantC1,theVectorV1,theMatrixM1,theMatrixM2,theTensorT3,theDiagonalMatrixD4,theVectorTradingCostFunction,theVectorApproximatedCriticalPoint,theVectorGradientYTradingCostFunction,theVectorGradientBetaTradingCostFunction
mutable struct RaysEstimateStructure 
	theConstantC1 	 # TODO.
	theVectorV1 	 # TODO.
	theMatrixM1 	 # TODO.
	theMatrixM2 	 # TODO.
	theTensorT3 	 # TODO.
	theDiagonalMatrixD4 	 # TODO.
	theVectorTradingCostFunction 	 # TODO.
	theVectorApproximatedCriticalPoint 	 # TODO.
	theVectorGradientYTradingCostFunction 	 # TODO.
	theVectorGradientBetaTradingCostFunction 	 # TODO.
end

## STRUCTURE RaysEstimateStructure: get functions
"""
#### Definition
```
get_constant_c1(aRaysEstimateStructure::RaysEstimateStructure)
```

returns the attribute `theConstantC1` of the structure `aRaysEstimateStructure`.

#### Argument
* `aRaysEstimateStructure::RaysEstimateStructure`: TODO.
"""
function get_constant_c1(aRaysEstimateStructure::RaysEstimateStructure)
	return aRaysEstimateStructure.theConstantC1
end

"""
#### Definition
```
get_vector_v1(aRaysEstimateStructure::RaysEstimateStructure)
```

returns the attribute `theVectorV1` of the structure `aRaysEstimateStructure`.

#### Argument
* `aRaysEstimateStructure::RaysEstimateStructure`: TODO.
"""
function get_vector_v1(aRaysEstimateStructure::RaysEstimateStructure)
	return aRaysEstimateStructure.theVectorV1
end

"""
#### Definition
```
get_matrix_m1(aRaysEstimateStructure::RaysEstimateStructure)
```

returns the attribute `theMatrixM1` of the structure `aRaysEstimateStructure`.

#### Argument
* `aRaysEstimateStructure::RaysEstimateStructure`: TODO.
"""
function get_matrix_m1(aRaysEstimateStructure::RaysEstimateStructure)
	return aRaysEstimateStructure.theMatrixM1
end

"""
#### Definition
```
get_matrix_m2(aRaysEstimateStructure::RaysEstimateStructure)
```

returns the attribute `theMatrixM2` of the structure `aRaysEstimateStructure`.

#### Argument
* `aRaysEstimateStructure::RaysEstimateStructure`: TODO.
"""
function get_matrix_m2(aRaysEstimateStructure::RaysEstimateStructure)
	return aRaysEstimateStructure.theMatrixM2
end

"""
#### Definition
```
get_tensor_t3(aRaysEstimateStructure::RaysEstimateStructure)
```

returns the attribute `theTensorT3` of the structure `aRaysEstimateStructure`.

#### Argument
* `aRaysEstimateStructure::RaysEstimateStructure`: TODO.
"""
function get_tensor_t3(aRaysEstimateStructure::RaysEstimateStructure)
	return aRaysEstimateStructure.theTensorT3
end

"""
#### Definition
```
get_diagonal_matrix_d4(aRaysEstimateStructure::RaysEstimateStructure)
```

returns the attribute `theDiagonalMatrixD4` of the structure `aRaysEstimateStructure`.

#### Argument
* `aRaysEstimateStructure::RaysEstimateStructure`: TODO.
"""
function get_diagonal_matrix_d4(aRaysEstimateStructure::RaysEstimateStructure)
	return aRaysEstimateStructure.theDiagonalMatrixD4
end

"""
#### Definition
```
get_vector_trading_cost_function(aRaysEstimateStructure::RaysEstimateStructure)
```

returns the attribute `theVectorTradingCostFunction` of the structure `aRaysEstimateStructure`.

#### Argument
* `aRaysEstimateStructure::RaysEstimateStructure`: TODO.
"""
function get_vector_trading_cost_function(aRaysEstimateStructure::RaysEstimateStructure)
	return aRaysEstimateStructure.theVectorTradingCostFunction
end

"""
#### Definition
```
get_vector_approximated_critical_point(aRaysEstimateStructure::RaysEstimateStructure)
```

returns the attribute `theVectorApproximatedCriticalPoint` of the structure `aRaysEstimateStructure`.

#### Argument
* `aRaysEstimateStructure::RaysEstimateStructure`: TODO.
"""
function get_vector_approximated_critical_point(aRaysEstimateStructure::RaysEstimateStructure)
	return aRaysEstimateStructure.theVectorApproximatedCriticalPoint
end

"""
#### Definition
```
get_vector_gradient_y_trading_cost_function(aRaysEstimateStructure::RaysEstimateStructure)
```

returns the attribute `theVectorGradientYTradingCostFunction` of the structure `aRaysEstimateStructure`.

#### Argument
* `aRaysEstimateStructure::RaysEstimateStructure`: TODO.
"""
function get_vector_gradient_y_trading_cost_function(aRaysEstimateStructure::RaysEstimateStructure)
	return aRaysEstimateStructure.theVectorGradientYTradingCostFunction
end

"""
#### Definition
```
get_vector_gradient_beta_trading_cost_function(aRaysEstimateStructure::RaysEstimateStructure)
```

returns the attribute `theVectorGradientBetaTradingCostFunction` of the structure `aRaysEstimateStructure`.

#### Argument
* `aRaysEstimateStructure::RaysEstimateStructure`: TODO.
"""
function get_vector_gradient_beta_trading_cost_function(aRaysEstimateStructure::RaysEstimateStructure)
	return aRaysEstimateStructure.theVectorGradientBetaTradingCostFunction
end


## STRUCTURE RaysEstimateStructure: set functions
"""
#### Definition
```
set_constant_c1!(aRaysEstimateStructure::RaysEstimateStructure, aNewConstantC1)
```

assigns to the attribute `theConstantC1` of the structure `aRaysEstimateStructure` the value `theConstantC1`.

#### Arguments
* `aRaysEstimateStructure::RaysEstimateStructure`: TODO.
* `aNewConstantC1::`: TODO.
"""
function set_constant_c1!(aRaysEstimateStructure::RaysEstimateStructure, aNewConstantC1)
	aRaysEstimateStructure.theConstantC1 = aNewConstantC1
end

"""
#### Definition
```
set_vector_v1!(aRaysEstimateStructure::RaysEstimateStructure, aNewVectorV1)
```

assigns to the attribute `theVectorV1` of the structure `aRaysEstimateStructure` the value `theVectorV1`.

#### Arguments
* `aRaysEstimateStructure::RaysEstimateStructure`: TODO.
* `aNewVectorV1::`: TODO.
"""
function set_vector_v1!(aRaysEstimateStructure::RaysEstimateStructure, aNewVectorV1)
	aRaysEstimateStructure.theVectorV1 = aNewVectorV1
end

"""
#### Definition
```
set_matrix_m1!(aRaysEstimateStructure::RaysEstimateStructure, aNewMatrixM1)
```

assigns to the attribute `theMatrixM1` of the structure `aRaysEstimateStructure` the value `theMatrixM1`.

#### Arguments
* `aRaysEstimateStructure::RaysEstimateStructure`: TODO.
* `aNewMatrixM1::`: TODO.
"""
function set_matrix_m1!(aRaysEstimateStructure::RaysEstimateStructure, aNewMatrixM1)
	aRaysEstimateStructure.theMatrixM1 = aNewMatrixM1
end

"""
#### Definition
```
set_matrix_m2!(aRaysEstimateStructure::RaysEstimateStructure, aNewMatrixM2)
```

assigns to the attribute `theMatrixM2` of the structure `aRaysEstimateStructure` the value `theMatrixM2`.

#### Arguments
* `aRaysEstimateStructure::RaysEstimateStructure`: TODO.
* `aNewMatrixM2::`: TODO.
"""
function set_matrix_m2!(aRaysEstimateStructure::RaysEstimateStructure, aNewMatrixM2)
	aRaysEstimateStructure.theMatrixM2 = aNewMatrixM2
end

"""
#### Definition
```
set_tensor_t3!(aRaysEstimateStructure::RaysEstimateStructure, aNewTensorT3)
```

assigns to the attribute `theTensorT3` of the structure `aRaysEstimateStructure` the value `theTensorT3`.

#### Arguments
* `aRaysEstimateStructure::RaysEstimateStructure`: TODO.
* `aNewTensorT3::`: TODO.
"""
function set_tensor_t3!(aRaysEstimateStructure::RaysEstimateStructure, aNewTensorT3)
	aRaysEstimateStructure.theTensorT3 = aNewTensorT3
end

"""
#### Definition
```
set_diagonal_matrix_d4!(aRaysEstimateStructure::RaysEstimateStructure, aNewDiagonalMatrixD4)
```

assigns to the attribute `theDiagonalMatrixD4` of the structure `aRaysEstimateStructure` the value `theDiagonalMatrixD4`.

#### Arguments
* `aRaysEstimateStructure::RaysEstimateStructure`: TODO.
* `aNewDiagonalMatrixD4::`: TODO.
"""
function set_diagonal_matrix_d4!(aRaysEstimateStructure::RaysEstimateStructure, aNewDiagonalMatrixD4)
	aRaysEstimateStructure.theDiagonalMatrixD4 = aNewDiagonalMatrixD4
end

"""
#### Definition
```
set_vector_trading_cost_function!(aRaysEstimateStructure::RaysEstimateStructure, aNewVectorTradingCostFunction)
```

assigns to the attribute `theVectorTradingCostFunction` of the structure `aRaysEstimateStructure` the value `theVectorTradingCostFunction`.

#### Arguments
* `aRaysEstimateStructure::RaysEstimateStructure`: TODO.
* `aNewVectorTradingCostFunction::`: TODO.
"""
function set_vector_trading_cost_function!(aRaysEstimateStructure::RaysEstimateStructure, aNewVectorTradingCostFunction)
	aRaysEstimateStructure.theVectorTradingCostFunction = aNewVectorTradingCostFunction
end

"""
#### Definition
```
set_vector_approximated_critical_point!(aRaysEstimateStructure::RaysEstimateStructure, aNewVectorApproximatedCriticalPoint)
```

assigns to the attribute `theVectorApproximatedCriticalPoint` of the structure `aRaysEstimateStructure` the value `theVectorApproximatedCriticalPoint`.

#### Arguments
* `aRaysEstimateStructure::RaysEstimateStructure`: TODO.
* `aNewVectorApproximatedCriticalPoint::`: TODO.
"""
function set_vector_approximated_critical_point!(aRaysEstimateStructure::RaysEstimateStructure, aNewVectorApproximatedCriticalPoint)
	aRaysEstimateStructure.theVectorApproximatedCriticalPoint = aNewVectorApproximatedCriticalPoint
end

"""
#### Definition
```
set_vector_gradient_y_trading_cost_function!(aRaysEstimateStructure::RaysEstimateStructure, aNewVectorGradientYTradingCostFunction)
```

assigns to the attribute `theVectorGradientYTradingCostFunction` of the structure `aRaysEstimateStructure` the value `theVectorGradientYTradingCostFunction`.

#### Arguments
* `aRaysEstimateStructure::RaysEstimateStructure`: TODO.
* `aNewVectorGradientYTradingCostFunction::`: TODO.
"""
function set_vector_gradient_y_trading_cost_function!(aRaysEstimateStructure::RaysEstimateStructure, aNewVectorGradientYTradingCostFunction)
	aRaysEstimateStructure.theVectorGradientYTradingCostFunction = aNewVectorGradientYTradingCostFunction
end

"""
#### Definition
```
set_vector_gradient_beta_trading_cost_function!(aRaysEstimateStructure::RaysEstimateStructure, aNewVectorGradientBetaTradingCostFunction)
```

assigns to the attribute `theVectorGradientBetaTradingCostFunction` of the structure `aRaysEstimateStructure` the value `theVectorGradientBetaTradingCostFunction`.

#### Arguments
* `aRaysEstimateStructure::RaysEstimateStructure`: TODO.
* `aNewVectorGradientBetaTradingCostFunction::`: TODO.
"""
function set_vector_gradient_beta_trading_cost_function!(aRaysEstimateStructure::RaysEstimateStructure, aNewVectorGradientBetaTradingCostFunction)
	aRaysEstimateStructure.theVectorGradientBetaTradingCostFunction = aNewVectorGradientBetaTradingCostFunction
end

### END: STRUCTURE RaysEstimateStructure ###

"""
#### Definition
```
get_default_ray_estimate_structure(aMarketDetails::MarketDetails)
```

TODO function description.

#### Arguments
* `aMarketDetails::MarketDetails`: TODO.
"""
function get_default_ray_estimate_structure(;
					    aMarketDetails::MarketDetails
					    )

	myNTradingPeriods = MarketDetailsModule.get_n_trading_periods(aMarketDetails)
	myUncertaintyStructure = MarketDetailsModule.get_uncertainty_structure(aMarketDetails)
	myNExtremeRays = get_n_extreme_rays(myUncertaintyStructure)

	myConstantC1 = nothing
	myVectorV1 = nothing
	myMatrixM1 = nothing
	myMatrixM2 = nothing
	myTensorT3 = nothing
	myDiagonalMatrixD4 = nothing
	myVectorTradingCostFunction = Array{Any}(undef,myNExtremeRays)
	myVectorApproximatedCriticalPoint = zeros(myNExtremeRays)
	myVectorGradientYTradingCostFunction = Array{Any}(undef,myNExtremeRays)
	myVectorGradientBetaTradingCostFunction = Array{Any}(undef,myNExtremeRays)

	return RaysEstimateStructure(
				     myConstantC1,
				     myVectorV1,
				     myMatrixM1,
				     myMatrixM2,
				     myTensorT3,
				     myDiagonalMatrixD4,
				     myVectorTradingCostFunction,
				     myVectorApproximatedCriticalPoint,
				     myVectorGradientYTradingCostFunction,
				     myVectorGradientBetaTradingCostFunction
				     )
end

"""
#### Definition
```
initialise_parameters_principal_scenarios!(aMarketDetails::MarketDetails)
```

returns the matrices and vectors that are constant in the ray approximation, no matter the ray.

#### Argument
* `aMarketDetails::MarketDetails`: TODO.
"""
function initialise_parameters_principal_scenarios!(;
						    aMarketDetails::MarketDetails,
						    aRaysEstimateStructure::RaysEstimateStructure
						    )

	# Getting the details of the market
	myNTradingPeriods = get_n_trading_periods(aMarketDetails)
	myInitialDemandForecast = get_initial_demand_forecast(aMarketDetails)
	myTaus = get_taus(aMarketDetails)
	myEtas = get_etas(aMarketDetails)
	myGammas = get_gammas(aMarketDetails)

	# Compute of the constants, vectors and matrices independent of the strategy

	## Constant C1
	myConstantC1 = 0.5*myGammas*myInitialDemandForecast^2

	## Vector V1
	myVectorV1ξ = myInitialDemandForecast.*(myTaus.^(0.5))          # price moves
	myVectorV1d = myGammas*myInitialDemandForecast.*ones(myNTradingPeriods) # forecast updates
	myVectorV1  = vcat(myVectorV1ξ,myVectorV1d)

	## Matrix M1
	myMatrixM1 = zeros(2*myNTradingPeriods,2*myNTradingPeriods)
	for i = 1:myNTradingPeriods
		myMatrixM1[i+myNTradingPeriods,1:myNTradingPeriods]      = myTaus.^0.5                   # price moves
		myMatrixM1[i+myNTradingPeriods,myNTradingPeriods+1:end]  = 0.5*myGammas.*ones(myNTradingPeriods) # forecast updates
	end
	myMatrixM1 = sparse(myMatrixM1)

	## Matrix M2
	myMatrixM2 = zeros(myNTradingPeriods, 2*myNTradingPeriods)
	for i = 1:myNTradingPeriods
		myMatrixM2[i,i:myNTradingPeriods] = -(myTaus[i:end]).^(0.5)
	end
	myMatrixM2 = sparse(myMatrixM2)

	## Diagonal matrix D4
	myDiagonalMatrixD4 = sparse(Diagonal((myEtas./myTaus.-0.5*myGammas)))

	# Update the values in the structure `aRaysEstimateStructure`
	set_constant_c1!(aRaysEstimateStructure,myConstantC1)
	set_vector_v1!(aRaysEstimateStructure,myVectorV1)
	set_matrix_m1!(aRaysEstimateStructure,myMatrixM1)
	set_matrix_m2!(aRaysEstimateStructure,myMatrixM2)
	set_diagonal_matrix_d4!(aRaysEstimateStructure,myDiagonalMatrixD4)
end

"""
#### Definition
```
update_tensor_volume_redistribution!(;
aMarketDetails::MarketDetails,
aStrategy::Strategy,
aRaysEstimateStructure::RaysEstimateStructure
)
```

returns a tensor (more precisely an array of sparse matrices for computational purposes) that depends on the redistribution matrix β and which is used as a predetermined recourse,
i.e. see equation (12) of the paper [Optimal Execution Strategy Under Price and Volume Uncertainty (J. Vaes and R.Hauser)](https://arxiv.org/abs/1810.11454).

#### Arguments
* `aMarketDetails::MarketDetails`: TODO.
* `aStrategy::Strategy`: TODO.
"""
function update_tensor_volume_redistribution!(;
					      aMarketDetails::MarketDetails,
					      aStrategy::Strategy,
					      aRaysEstimateStructure::RaysEstimateStructure
					      )

	myNTradingPeriods = MarketDetailsModule.get_n_trading_periods(aMarketDetails)
	myRedistributionMatrix = get_redistribution_matrix(aStrategy)

	myTensorT3 = Array{SparseMatrixCSC{Float64,Int64}}(undef,10)
	for k in 1:2*myNTradingPeriods
		myTensorT3[k] = spzeros(myNTradingPeriods,myNTradingPeriods)
	end

	# fills the non-zero coefficients
	for k in 1:myNTradingPeriods-1
		myIndex = myNTradingPeriods+k
		myLocalTensorT3 = myTensorT3[myIndex]
		myLocalTensorT3[k+1:end,1:k] .= myRedistributionMatrix[k,k+1:end]
		myLocalTensorT3[k+1:end,k+1:end] = Matrix{Float64}(I,myNTradingPeriods-k,myNTradingPeriods-k)
	end

	set_tensor_t3!(aRaysEstimateStructure,myTensorT3)
end

"""
#### Definition
```
product_tensor_direction(aTensor::Array{Float64,3}, aDirection::Array{Float64,1})
```

computes the product of a tensor with a vector, i.e. T⊗x,
which is the matrix defined as follows: [T⊗x]_{i,j} = Σ_k T[i,j,k]*x_k for a tensor T ϵ R^{m1,m2,m3} and a vector x ϵ R^{m3}.

#### Arguments
* `aTensor::Array{SparseMatrixCSC{Float64,Int64},1}`: TODO.
* `aDirection::Array{Float64,1}`: TODO.
"""
function product_tensor_direction(
				  aTensor::Array{SparseMatrixCSC{Float64,Int64},1},
				  aDirection::Array{Float64,1}
				  )

	return sum(aDirection[k].*aTensor[k] for k in 1:length(aDirection))
end

"""
#### Definition
```
update_trading_cost_and_gradient_functions_in_rays!(aStrategy::Strategy, aMarketDetails::MarketDetails, aDirections::Array{Float64,2}, aVectorA, aVectorB, aVectorVolumesRule)
```

TODO function description.

#### Arguments
* `aStrategy::Strategy`: TODO.
* `aMarketDetails::MarketDetails`: TODO.
* `aDirections::Array{Float64`: TODO.
* `2}`: TODO.
* `aVectorA`: TODO.
* `aVectorB`: TODO.
* `aVectorVolumesRule`: TODO.
"""
function update_trading_cost_and_gradient_functions_in_rays!(;
							     aMarketDetails::MarketDetails,
							     aStrategy::Strategy,
							     aRaysEstimateStructure::RaysEstimateStructure
							     )

	# gets the parameters
	myNTradingPeriods       = MarketDetailsModule.get_n_trading_periods(aMarketDetails)
	myInitialDemandForecast = get_initial_demand_forecast(aMarketDetails)
	myEpsilons              = get_epsilons(aMarketDetails)

	myTradingPlan = get_trading_plan(aStrategy)
	myRedistributionMatrix = get_redistribution_matrix(aStrategy)

	# gets the uncertainty structure
	myUncertaintyStructure = MarketDetailsModule.get_uncertainty_structure(aMarketDetails)

	myRays = get_extreme_rays(myUncertaintyStructure)
	myNExtremeRays = get_n_extreme_rays(myUncertaintyStructure)

	# gets parameters of the ray estimate structure
	myConstantC1       = get_constant_c1(aRaysEstimateStructure)
	myVectorV1         = get_vector_v1(aRaysEstimateStructure)
	myMatrixM1         = get_matrix_m1(aRaysEstimateStructure)
	myMatrixM2         = get_matrix_m2(aRaysEstimateStructure)
	myTensorT3         = get_tensor_t3(aRaysEstimateStructure) # TODO: make sure that this tensor is updated with the strategy
	myDiagonalMatrixD4 = get_diagonal_matrix_d4(aRaysEstimateStructure)

	# computes the constant term and its derivative of the polynomials representing trading cost in the rays (common for all rays)
	myC    = myConstantC1 .+ myInitialDemandForecast^2*(myTradingPlan'*myDiagonalMatrixD4*myTradingPlan)
	myDcDy = reshape(2*myInitialDemandForecast^2*myDiagonalMatrixD4*myTradingPlan, myNTradingPeriods)

	# Initialise the output for the functions
	myVectorTradingCostFunction = Array{Any}(undef,myNExtremeRays)
	myVectorApproximatedCriticalPoint = zeros(myNExtremeRays)
	myVectorGradientYTradingCostFunction = Array{Any}(undef,myNExtremeRays)
	myVectorGradientBetaTradingCostFunction = Array{Any}(undef,myNExtremeRays)

	# loop on all rays
	for i in eachindex(myRays)	

		myRay = myRays[i]

		################################################################################################################################
		# computes the coefficients and their partial derivative of the quadratic polynomial representing the trading cost in each ray #
		################################################################################################################################

		# computes the matrix to multiply by the trading plan to get the volumes adjustments to take into account the forecast updates
		myT3d = product_tensor_direction(myTensorT3,myRay)

		# computes the volumes adjustments to take into account for an UNITARY change in the direction `myRay`
		myT3dy = myT3d*myTradingPlan

		# quadratic term

		myΞ = myT3d'*myDiagonalMatrixD4*myT3d
		myA = myTradingPlan'*myΞ*myTradingPlan .+ myRay'*myMatrixM1*myRay
		myA = myA .+ (myMatrixM2*myRay)'*myT3dy

		myDaDy = reshape(2*myΞ*myTradingPlan, myNTradingPeriods) + reshape((myMatrixM2*myRay)'*myT3d, myNTradingPeriods)

		# linear term

		myB = myVectorV1'*myRay
		myB = myB .+ 2*myInitialDemandForecast.*myTradingPlan'*myDiagonalMatrixD4*myT3dy
		myB = myB .+ myInitialDemandForecast.*((myMatrixM2*myRay)'*myTradingPlan)
		myB=myB[1]::Float64 # because is is an array of size (1,1)

		myDbDy = reshape(myInitialDemandForecast.*(myMatrixM2*myRay), myNTradingPeriods)
		myDbDy = myDbDy + reshape(2*myInitialDemandForecast.*(myDiagonalMatrixD4*myT3dy), myNTradingPeriods)
		myDbDy = myDbDy + reshape(2*myInitialDemandForecast.*(myTradingPlan'*myDiagonalMatrixD4*myT3d), myNTradingPeriods)

		# compute an rough estimate of the extremum of the parabola
		myVectorApproximatedCriticalPoint[i] = -myB/(2*myA)

		#################################################
		# computes the trading cost function in the ray #
		#################################################

		myVolumesTraded(x) = (myInitialDemandForecast*I + x*myT3d)*myTradingPlan

		#myTradingCostFunction(x) = myA*x^2 + myB*x + myC + sum(myEpsilons[i]*abs(myVolumesTraded[i]) for i = 1:myNTradingPeriods) # TODO: delete if ok
		myCostFunctionInRay(x) = myA*x^2 + myB*x + myC + sum(myEpsilons.*abs.(myVolumesTraded(x)))
		myVectorTradingCostFunction[i] = myCostFunctionInRay

		######################################################################
		# computes the gradient in y of the trading cost function in the ray #
		######################################################################


		#myGradientYTradingCost = aDaDy.*(aScalarRealisation^2) + aDbDy.*aScalarRealisation + aDcDy + reshape(reshape(myEpsilons.*(sign.(myVolumesTraded)),1,myNTradingPeriods)*(myMatrixVolumesTraded), myNTradingPeriods)
		myLocalGradientYTradingCostFunction(x) = myDaDy.*(x^2) + myDbDy.*x + myDcDy + reshape(reshape(myEpsilons.*sign.(myVolumesTraded(x)),1,myNTradingPeriods)*(myInitialDemandForecast*I + x*myT3d), myNTradingPeriods)

		# As sum_i y_i = 1, we do not consider y_m as a variable, i.e. y_m = 1-y_1-y_2-...-y_m-1.
		# Hence, dC/dy_i = dC_local/dyi - dC_local/dy_m.
		myGradientYTradingCostFunction(x) = begin
			v = myLocalGradientYTradingCostFunction(x)
			return v[1:end-1] .- v[end]
		end
		myVectorGradientYTradingCostFunction[i] = myGradientYTradingCostFunction

	end

	# updates the functions in the structure `aRaysEstimateStructure`
	set_vector_trading_cost_function!(aRaysEstimateStructure,myVectorTradingCostFunction)
	set_vector_approximated_critical_point!(aRaysEstimateStructure,myVectorApproximatedCriticalPoint)
	set_vector_gradient_y_trading_cost_function!(aRaysEstimateStructure,myVectorGradientYTradingCostFunction)
	set_vector_gradient_beta_trading_cost_function!(aRaysEstimateStructure,myVectorGradientYTradingCostFunction)
end

"""
#### Definition
```
update_ray_estimate_structure!(;
aMarketDetails::MarketDetails,
aStrategy::Strategy,
aRaysEstimateStructure::RaysEstimateStructure
)
```

TODO function description.

#### Arguments
* `aMarketDetails::MarketDetails`: TODO.
* `aStrategy::Strategy`: TODO.
"""
function update_ray_estimate_structure!(;
					aMarketDetails::MarketDetails,
					aStrategy::Strategy,
					aRaysEstimateStructure::RaysEstimateStructure
					)

	# updates the tensor volume (needed if the redistribution matrix is updated in the strategy)
	update_tensor_volume_redistribution!(
					     aMarketDetails=aMarketDetails,
					     aStrategy=aStrategy,
					     aRaysEstimateStructure=aRaysEstimateStructure
					     )

	# updates the coefficients of the polynomial representing the cost in each ray (needed if either the trading plan or the redistribution matrix is updated in the strategy)
	update_trading_cost_and_gradient_functions_in_rays!(
							    aMarketDetails=aMarketDetails,
							    aStrategy=aStrategy,
							    aRaysEstimateStructure=aRaysEstimateStructure
							    )
end

"""
#### Definition
```
get_initial_ray_estimate_structure(;aMarketDetails::MarketDetails,aStrategy::Strategy)
```

TODO function description.

#### Arguments
* `;aMarketDetails::MarketDetails`: TODO.
* `aStrategy::Strategy`: TODO.
"""
function get_ray_estimate_structure(;
				    aMarketDetails::MarketDetails,
				    aStrategy::Strategy
				    )

	# gets the parameters
	myNTradingPeriods = MarketDetailsModule.get_n_trading_periods(aMarketDetails)
	myInitialDemandForecast = get_initial_demand_forecast(aMarketDetails)
	myTradingPlan = get_trading_plan(aStrategy)


	# initialises the rays estimate structure
	myRaysEstimateStructure = get_default_ray_estimate_structure(aMarketDetails=aMarketDetails)

	# initialises the values that are indepent of the strategy and the ray
	initialise_parameters_principal_scenarios!(
						   aMarketDetails=aMarketDetails,
						   aRaysEstimateStructure=myRaysEstimateStructure
						   )

	# updates and thus initialise the tensor T3 and the coefficients of the second order polynomials representing the cost in each ray
	update_ray_estimate_structure!(
				       aMarketDetails=aMarketDetails,
				       aStrategy=aStrategy,
				       aRaysEstimateStructure=myRaysEstimateStructure
				       )

	return myRaysEstimateStructure
end

"""
#### Definition
```
find_zero_in_ray_and_prob_exceeding_VaR(
aCostFunctionInRay,
aVaR::Float64,
aDistEstimateAlongRay
)
```

TODO function description.

#### Argument
* ``: TODO.
"""
function find_zero_in_ray_and_prob_exceeding_VaR(;
						 aVaR::Float64,
						 aCostFunctionInRay,
						 aApproximatedCriticalPoint,
						 aDistEstimateAlongRay
						 )

	CostInRayMinusVaR(x) = aCostFunctionInRay(x) - aVaR

	# TODO: something cleaner
	myXTest = range(0,stop=quantile(aDistEstimateAlongRay,1-10e-4),length=10^2)
	myYTest = CostInRayMinusVaR.(myXTest)

	if !any(myYTest.<0.0)
		return 0.0, 1
	end

	# finds the unique positive roots, which is well defined by Proposition 2.4 of our second paper
	# @time myPositiveRoot = find_zero(CostInRayMinusVaR, (aApproximatedCriticalPoint,quantile(aDistEstimateAlongRay,1-10e-6)), Bisection())
	myPositiveRoot = find_zero(CostInRayMinusVaR, quantile(aDistEstimateAlongRay,1-10e-6), Order1())

	# computes the probability of execeeding the positive root
	myProbExceedingVaRInRay = 1-cdf(aDistEstimateAlongRay,myPositiveRoot)

	return myPositiveRoot, myProbExceedingVaRInRay
end

"""
#### Definition
```
get_probability_exceeding_value_in_rays!(;aRaysEstimateStructure::RaysEstimateStructure,aValue::Float64)
```

TODO function description.

#### Arguments
* `aRaysEstimateStructure::RaysEstimateStructure`: TODO.
* `aValue::Float64`: TODO.
"""
function get_probability_exceeding_value_in_rays!(;
						  aNExtremeRays::Int64=0,
						  aRays,
						  aRaysProbabilities,
						  aDistEstimateAlongRays,
						  aVectorTradingCostFunction,
						  aVectorApproximatedCriticalPoint,
						  aValue::Float64,
						  aRayValuesToGetVaR=zeros(aNExtremeRays)
						  )


	myProbExceedingValueInRays = 0.0
	for i in eachindex(aNExtremeRays)
		aRayValuesToGetVaR[i], myProbExceedingValueInRaysInRay = find_zero_in_ray_and_prob_exceeding_VaR(
														 aVaR=aValue,
														 aCostFunctionInRay=aVectorTradingCostFunction[i],
														 aApproximatedCriticalPoint=aVectorApproximatedCriticalPoint[i],
														 aDistEstimateAlongRay=aDistEstimateAlongRays[i]
														 )

		myProbExceedingValueInRays += aRaysProbabilities[i]*myProbExceedingValueInRaysInRay
	end

	return myProbExceedingValueInRays
end

"""
#### Definition
```
get_trading_cost_VaR()
```

TODO function description.

#### Argument
* ``: TODO.
"""
function get_trading_cost_VaR(;
			      aAlphaConditionallyInExtremeRays,
			      aNExtremeRays,
			      aRays,
			      aRaysProbabilities,
			      aDistEstimateAlongRays,
			      aVectorTradingCostFunction,
			      aVectorApproximatedCriticalPoint
			      )

	# First we have to estimate the value at risk and then compute CVaR
	# TODO: come up with better guess
	myVaRInitialGuess = sum(aVectorTradingCostFunction[i](quantile(aDistEstimateAlongRays[i],aRaysProbabilities[i]*aAlphaConditionallyInExtremeRays)) for i in 1:aNExtremeRays)

	# initialises the variables needed to do the bisection in order to get the VaR value
	myVaR    = myVaRInitialGuess
	myVaROld = myVaR

	# Lower and upper bound on VaR
	myLowerBoundVaR = (10.0^-4)*myVaRInitialGuess
	myUpperBoundVaR = (10.0^4)*myVaRInitialGuess

	# vector of the values of the scalar random variables such that the cost of R in the rays equals VaR, i.e. C_{ray_i}(R_i) = VaR
	myRayValuesToGetVaR = zeros(aNExtremeRays) # TODO: check if has to return or not

	# parameters for the bisection method
	myToleranceAlpha = 10.0^-15 # TODO: check 
	myTolVaR = 10.0^-20 # TODO: check 
	myIter = 0
	myIterMax = 10^4
	myProbExceedingVaR = 0.0
	myHasFoundVaR = (abs(myProbExceedingVaR-aAlphaConditionallyInExtremeRays) < myToleranceAlpha)

	while ((!myHasFoundVaR) && (myIter<myIterMax))

		myIter += 1
		myProbExceedingVaR = get_probability_exceeding_value_in_rays!(
									      aNExtremeRays=aNExtremeRays,
									      aRays=aRays,
									      aRaysProbabilities=aRaysProbabilities,
									      aDistEstimateAlongRays=aDistEstimateAlongRays,
									      aVectorTradingCostFunction=aVectorTradingCostFunction,
									      aVectorApproximatedCriticalPoint=aVectorApproximatedCriticalPoint,
									      aValue=myVaR,
									      aRayValuesToGetVaR=myRayValuesToGetVaR
									      )

		# checks if you have achieved high precision enought to get 
		myHasFoundVaR = (abs(myProbExceedingVaR-aAlphaConditionallyInExtremeRays) < myToleranceAlpha)

		# if VaR has not been find yet
		if (!myHasFoundVaR)
			myVaROld = myVaR
			if myProbExceedingVaR < aAlphaConditionallyInExtremeRays # VaR is too large
				myUpperBoundVaR = myVaR
				if myVaR == myLowerBoundVaR # because we have no idea on the upper bound of CVaR
					myLowerBoundVaR = 0.5*myVaR
				end
				myVaR = 0.5*(myLowerBoundVaR+myVaR)
			else # VaR is too small
				myLowerBoundVaR = myVaR
				if myVaR == myUpperBoundVaR # because we have no idea on the upper bound of CVaR
					myUpperBoundVaR = 2*myVaR
				end
				myVaR = 0.5*(myUpperBoundVaR+myVaR)
			end
			if abs(myVaR-myVaROld)/abs(max(myVaR,myVaROld)) < myTolVaR # TODO: better warning message
				@warn(
				      string(
					     "VaR no more significant if change, the probability of exceeding VaR equals ",
					     myProbExceedingVaR,
					     " instead of ",
					     aAlphaConditionallyInExtremeRays,
					     ", which means an absolute error of ",
					     abs(myProbExceedingVaR-aAlphaConditionallyInExtremeRays),
					     "."
					     )
				      )
				break
			end
		end
	end

	##println("The number of iteration: ",myIter)
	#println("myProbExceedingVaR: ",myProbExceedingVaR)
	#println("abs(myVaR-myVaROld)/abs(max(myVaR,myVaROld)) ",abs(myVaR-myVaROld)/abs(max(myVaR,myVaROld)))

	if myIter == myIterMax
		@warn("Reached the maximum number of iteration allowed to find the value of the value at risk (VaR).")
	end

	return myVaR
end

"""
#### Definition
```
get_gradient_forward_finite_difference(aFunction,aPoint)
```

TODO function description.

#### Arguments
* `aFunction`: TODO.
* `aPoint`: TODO.
"""
function get_gradient_forward_finite_difference(aFunction,aPoint)

	# the value of the function
	myFunctionValue = aFunction(aPoint)

	# initialises the output
	myGradient = zeros(size(aPoint))

	# a vector used to shift the coordantes of `aPoints`
	myLocalPoint = copy(aPoint)

	# the step for the finite difference
	myϵ = 10.0^-8 # TODO: check

	#myGradientParallel = SharedArray{Float64}(size(aPoint))
	#@distributed for i in eachindex(aPoint)

	#	myLocalPoint     = copy(aPoint)
	#	myLocalPoint[i] += myϵ

	#	myLocalFunctionValue   = aFunction(myLocalPoint)
	#	myGradientParallel[i]  = (myLocalFunctionValue - myFunctionValue) / myϵ
	#end

	#for i in eachindex(myGradientParallel)
	#	myGradient[i] = myGradientParallel[i]
	#end

	#println("myFunctionValue: ",myFunctionValue)

	for i in eachindex(aPoint)

		myLocalPoint     = copy(aPoint)
		myLocalPoint[i] += myϵ

		myLocalFunctionValue = aFunction(myLocalPoint)
		#println("myLocalFunctionValue: ",myLocalFunctionValue)
		myGradient[i] = (myLocalFunctionValue - myFunctionValue) / myϵ
	end

	return myGradient
end

"""
#### Definition
```
```

TODO function description.

#### Arguments
* `;aMarketDetails::MarketDetails`: TODO.
* `aStrategy::Strategy`: TODO.
"""
function get_trading_cost_VaR_and_gradient_rays_approximation(;
							      aMarketDetails::MarketDetails,
							      aSimulationParameters::SimulationParameters,
							      aStrategy::Strategy,
							      aTrader::Trader,
							      aRaysEstimateStructure::RaysEstimateStructure
							      )

	# gets parameters
	myNTradingPeriods = MarketDetailsModule.get_n_trading_periods(aMarketDetails)

	# gets the variables of the strategy
	myTradingPlanVariables, myRedistributionMatrixVariables = get_decision_variables_from_strategy(aStrategy)

	# gets the uncertainty structure
	myUncertaintyStructure = MarketDetailsModule.get_uncertainty_structure(aMarketDetails)

	# gets the details on the rays estimate
	myProbExtremeInRays     = get_prob_extreme_in_rays(myUncertaintyStructure)
	myNExtremeRays          = get_n_extreme_rays(myUncertaintyStructure)
	myRays                  = get_extreme_rays(myUncertaintyStructure)
	myRaysProbabilities     = get_rays_probabilities(myUncertaintyStructure)
	myDistEstimateAlongRays = get_dist_estimate_along_rays(myUncertaintyStructure)

	# gets the cost functions in each ray in terms of the scalar realisation in along the ray
	myVectorTradingCostFunction          = get_vector_trading_cost_function(aRaysEstimateStructure)
	myVectorApproximatedCriticalPoint    = get_vector_approximated_critical_point(aRaysEstimateStructure)
	myVectorGradientYTradingCostFunction = get_vector_gradient_y_trading_cost_function(aRaysEstimateStructure)

	# computes the probablity to be greater or equal given that we are in extreme events
	myAlpha = get_alpha(aTrader)
	myAlphaConditionallyInExtremeRays = myAlpha/myProbExtremeInRays

	###########################
	# Function to get the VaR #
	###########################

	myLocalRaysEstimateStructure = get_ray_estimate_structure(
								  aMarketDetails=aMarketDetails,
								  aStrategy=aStrategy
								  )

	myFunctGetVaR(x) = begin

		myLocalTradingPlanVariables = nothing
		myLocalRedistributionMatrixVariables = nothing

		# assigns the variables, if nothing is specified we compute the gradient of all variables
		if get_optimise_trading_plan(aSimulationParameters) && !(get_optimise_redistribution_matrix(aSimulationParameters))
			myLocalTradingPlanVariables = x
			myLocalRedistributionMatrixVariables = myRedistributionMatrixVariables
		elseif !(get_optimise_trading_plan(aSimulationParameters)) && get_optimise_redistribution_matrix(aSimulationParameters)
			myLocalTradingPlanVariables = myTradingPlanVariables
			myLocalRedistributionMatrixVariables = x
		else
			myLocalTradingPlanVariables = x[1:myNTradingPeriods]
			myLocalRedistributionMatrixVariables = x[myNTradingPeriods:end]
		end

		myNewStrategy = get_strategy_from_decision_variables(
								     aNTradingPeriods=myNTradingPeriods,
								     aTradingPlanVariables=myLocalTradingPlanVariables,
								     aRedistributionMatrixVariables=myLocalRedistributionMatrixVariables
								     )

		update_ray_estimate_structure!(
					       aMarketDetails=aMarketDetails,
					       aStrategy=myNewStrategy,
					       aRaysEstimateStructure=myLocalRaysEstimateStructure
					       )

		return get_trading_cost_VaR(
					    aAlphaConditionallyInExtremeRays=myAlphaConditionallyInExtremeRays,
					    aNExtremeRays=myNExtremeRays,
					    aRays=myRays,
					    aRaysProbabilities=myRaysProbabilities,
					    aDistEstimateAlongRays=myDistEstimateAlongRays,
					    aVectorTradingCostFunction=get_vector_trading_cost_function(myLocalRaysEstimateStructure),
					    aVectorApproximatedCriticalPoint=get_vector_approximated_critical_point(myLocalRaysEstimateStructure)
					    )
	end

	#####################################
	# Computes the VaR and its gradient #
	#####################################

	# initialises the gradient output
	myTradingCostVaR = 0.0
	myTradingCostVaRGradientY = zeros(size(myTradingPlanVariables))
	myTradingCostVaRGradientBeta = zeros(size(myRedistributionMatrixVariables))

	myTimeRay_VaR_Gradient = @elapsed begin
		# assigns the variables, if nothing is specified we compute the gradient of all variables
		if get_optimise_trading_plan(aSimulationParameters) && !(get_optimise_redistribution_matrix(aSimulationParameters))
			#println("Part 1")
			myTimeRay_VaR = @elapsed begin
				myTradingCostVaR = myFunctGetVaR(myTradingPlanVariables)
			end
			myTimeRay_Gradient = @elapsed begin
				myTradingCostVaRGradientY = get_gradient_forward_finite_difference(myFunctGetVaR,myTradingPlanVariables)
			end
		elseif !(get_optimise_trading_plan(aSimulationParameters)) && get_optimise_redistribution_matrix(aSimulationParameters)
			#println("Part 2")
			myTradingCostVaR = myFunctGetVaR(myRedistributionMatrixVariables)
			myTradingCostVaRGradientBeta = get_gradient_forward_finite_difference(myFunctGetVaR,myRedistributionMatrixVariables)
		else
			#println("Part 3")
			myTradingCostVaR = myFunctGetVaR(vcat(myTradingPlanVariables,myRedistributionMatrixVariables))
			myTradingCostVaRGradient = get_gradient_forward_finite_difference(
											  myFunctGetVaR,
											  vcat(myTradingPlanVariables,myRedistributionMatrixVariables)
											  )
			myTradingCostVaRGradientY    = myTradingCostVaRGradient[1:myNTradingPeriods-1]
			myTradingCostVaRGradientBeta = myTradingCostVaRGradient[myNTradingPeriods:end]
		end
	end

	println("Time to compute VaR: ",myTimeRay_VaR)
	println("Time to compute gradient with finite difference: ",myTimeRay_Gradient)
	println("Time to compute var and gradient with finite difference: ",myTimeRay_VaR_Gradient)

	#println("get_trading_cost_VaR_and_gradient_rays_approximation: myTradingCostVaR: ",myTradingCostVaR)
	#println("get_trading_cost_VaR_and_gradient_rays_approximation: myTradingCostVaRGradientY: ",myTradingCostVaRGradientY)
	#println("get_trading_cost_VaR_and_gradient_rays_approximation: myTradingCostVaRGradientBeta: ",myTradingCostVaRGradientBeta)
	# returns the VaR and its right gradient
	return myTradingCostVaR, myTradingCostVaRGradientY, myTradingCostVaRGradientBeta
end

function is_nul_subgradient_acceptable(;
				       aNExtremeRays,
				       aRiskAversion,
				       aGradientTradingCostExpectedValue,
				       aTradingCostGradientInEachRay
				       )

	#println("Enter nul")

	myModel = Model(with_optimizer(Gurobi.Optimizer,OutputFlag=0))

	# VARIABLES #
	@variable(myModel, 0 <= λ[1:aNExtremeRays] <= 1)

	# OBJECTIVE #
	@objective(myModel, Min, 1)

	# CONSTRAINT #
	@constraint(myModel, sum(λ[i] for i = 1:aNExtremeRays) == 1)
	@constraint(myModel, (1-aRiskAversion)*aGradientTradingCostExpectedValue + aRiskAversion*sum(λ[i].*aTradingCostGradientInEachRay[i] for i = 1:aNExtremeRays) .== 0)

	# SOLVING #
	status = optimize!(myModel)


	#println("End nul")

	# returns true if the nul vector is a subgradient, false otherwise
	return !(termination_status(myModel) == MOI.INFEASIBLE)
end

"""
#### Definition
```
get_VaR_subgradient_center_polytope(aVectorGradientInRays)
```

TODO function description.

#### Argument
* `aVectorGradientInRays`: TODO.
"""
function get_subgradient_Chebyshev_centre_polytope(;
						   aNExtremeRays,
						   aNTradingPeriods,
						   aTradingCostGradientInEachRay
						   )
	#println("Enter sub")

	myModel = Model(with_optimizer(Gurobi.Optimizer,OutputFlag=0))

	# VARIABLES #
	@variable(myModel, 0 <= λ[1:aNExtremeRays] <= 1)
	@variable(myModel, myRadius >= 0)
	@variable(myModel, myCentre[1:aNTradingPeriods-1])

	# OBJECTIVE #

	@objective(myModel, Min, myRadius)

	# CONSTRAINT #
	@constraint(myModel, sum(λ[i] for i = 1:aNExtremeRays) == 1)
	@constraint(myModel, myCentre .== sum(λ[i].*aTradingCostGradientInEachRay[i] for i = 1:aNExtremeRays))
	@constraint(myModel, [i = 1:aNExtremeRays],  (myCentre.-aTradingCostGradientInEachRay[i])'*(myCentre.-aTradingCostGradientInEachRay[i]) <= myRadius)

	# SOLVING #
	status = optimize!(myModel)

	#println("End sub")

	return JuMP.value.(myCentre)
end

function get_random_subgradient(;
				aNExtremeRays,
				aTradingCostGradientInEachRay
				)

	myCoeff = rand(aNExtremeRays)
	myCoeffNormed = myCoeff./(sum(myCoeff))

	return sum(myCoeffNormed[i]*aTradingCostGradientInEachRay[i] for i in 1:aNExtremeRays)

end

function get_trading_cost_mean_VaR_value_and_subgradient!(;
							  aMarketDetails::MarketDetails,
							  aSimulationParameters::SimulationParameters,
							  aStrategy::Strategy,
							  aTrader::Trader,
							  aRaysEstimateStructure::RaysEstimateStructure
							  )

	# Risk-aversion of the trader
	myRiskAversion = get_risk_aversion(aTrader)

	################################################################################
	# Get the components needed to compute the objective function and its gradient #
	################################################################################

	# gets the expectation of the trading cost as well as its gradient with a Monte Carlo Approach
	myTimeMonteCarlo = @elapsed begin
		myExpectedValue, myGradientYExpectedValue, myGradientBetaExpectedValue = get_trading_cost_expectation_and_gradient_Monte_Carlo(
																	       aStrategy=aStrategy,
																	       aTrader=aTrader,
																	       aSimulationParameters=aSimulationParameters
																	       )
	end

	# gets the VaR value and its gradient in each ray
	myTimeRay = @elapsed begin
		myVaR, myGradientYVaR, myGradientBetaVaR = get_trading_cost_VaR_and_gradient_rays_approximation(
														aMarketDetails=aMarketDetails,
														aSimulationParameters=aSimulationParameters,
														aStrategy=aStrategy,
														aTrader=aTrader,
														aRaysEstimateStructure=aRaysEstimateStructure
														)
	end

	println("myVaR: ",myVaR)
	println("myGradientYVaR: ",myGradientYVaR)
	println("myGradientBetaVaR: ",myGradientBetaVaR)
	println("Time for MC: ", myTimeMonteCarlo)
	println("Time for ray: ", myTimeRay)
	println("Time proportion for ray: ", myTimeRay/(myTimeRay+myTimeMonteCarlo)*100)

	##########################################################
	# computes the objective function value and its gradient #
	##########################################################

	# objective value
	myObjectiveValue = (1- myRiskAversion) * myExpectedValue + myRiskAversion * myVaR

	# subgradient in y
	myObjectiveValueSubgradientY =  (1- myRiskAversion) * myGradientYExpectedValue + myRiskAversion * myGradientYVaR

	# subgradient in beta
	myObjectiveValueSubgradientBeta = (1- myRiskAversion) * myGradientBetaExpectedValue + myRiskAversion * myGradientBetaVaR

	return myObjectiveValue, myObjectiveValueSubgradientY, myObjectiveValueSubgradientBeta
end

##################################
## Resolution: gradient descent ##
##################################

function get_optimal_strategy_mean_VaR(;
				       aTrader::Trader,
				       aSimulationParameters::SimulationParameters,
				       aInitialStrategy::Strategy
				       )

	# gets the market details belief of the trader
	myMarketDetailsBelief = get_market_details_belief(aTrader)

	# number of trading periods
	myNTradingPeriods = MarketDetailsModule.get_n_trading_periods(myMarketDetailsBelief)

	# Algorithm to use
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
			      "\nMeanVaRPrincipalScenariosModule_ERROR 102:\n",
			      "Algorithm ",
			      myAlgorithm,
			      " is not recognised in the Mean-VaR framework."
			      )
		       )
	end

	#####################################################
	# Computation of an initial point for the algorithm #
	#####################################################

	myInitialStrategy = aInitialStrategy
	myInitialTradingPlanVariables, myInitialRedistributionMatrixVariables = get_decision_variables_from_strategy(myInitialStrategy)

	# initialises the ray estimate structure
	myRaysEstimateStructure = get_ray_estimate_structure(
							     aMarketDetails=myMarketDetailsBelief,
							     aStrategy=myInitialStrategy
							     )

	# A function that returns the value of the objective and the gradient
	# What is not in G or F is common for both (more efficient)
	function fg!(F,G,x)

		println("===========================")

		myCurrentTradingPlan          = x[1:myNTradingPeriods-1]
		myCurrentRedistributionMatrix = x[myNTradingPeriods:end]

		###################
		# Print functions #
		###################

		println("Trading plan:\n\t", myCurrentTradingPlan)
		println("Redistribution matrix:\n\t", myCurrentRedistributionMatrix)

		# Defines a new strategy base on the new point
		myCurrentStrategy = get_strategy_from_decision_variables(
									 aNTradingPeriods=myNTradingPeriods,
									 aTradingPlanVariables=myCurrentTradingPlan,
									 aRedistributionMatrixVariables=myCurrentRedistributionMatrix
									 )

		# updates the ray estimate structure
		update_ray_estimate_structure!(;
					       aMarketDetails=myMarketDetailsBelief,
					       aStrategy=myCurrentStrategy,
					       aRaysEstimateStructure=myRaysEstimateStructure
					       )


		myObjectiveFunction, myObjectiveFunctionGradientY, myObjectiveFunctionGradientBeta = get_trading_cost_mean_VaR_value_and_subgradient!(
																		      aMarketDetails=myMarketDetailsBelief,
																		      aSimulationParameters=aSimulationParameters,
																		      aTrader=aTrader,
																		      aStrategy=myCurrentStrategy,
																		      aRaysEstimateStructure=myRaysEstimateStructure
																		      )

		println("\nmyObjectiveFunction: ",myObjectiveFunction)
		println("myObjectiveFunctionGradientY: ", myObjectiveFunctionGradientY)
		println("myObjectiveFunctionGradientBeta: ",myObjectiveFunctionGradientBeta)

		if G != nothing
			G[1:myNTradingPeriods-1]   = myObjectiveFunctionGradientY
			G[myNTradingPeriods:end]   = myObjectiveFunctionGradientBeta
		end
		if F != nothing
			return myObjectiveFunction
		end
	end

	println("Risk aversion: ", get_risk_aversion(aTrader))

	# initial point of the optimisation
	myInitialPoint = vcat(myInitialTradingPlanVariables,myInitialRedistributionMatrixVariables)

	results = Optim.optimize(
				 Optim.only_fg!(fg!), 
				 myInitialPoint,  # the point where to start the search from
				 myOptimAlgorithm, # the algorithm to use to optimise
				 Optim.Options(
					       iterations = get_CVaR_optimisation_maximum_number_of_iterations(aSimulationParameters),
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
		@warn(
		      string(
			     "\nMeanVaRPrincipalScenariosModule 101:\n",
			     "The maximum number of iterations for the gradient descent has been reached.",
			     "As a consequence the optimal strategy return might be erroneous."
			     )
		      )
	elseif !(Optim.converged(results)) # checks if the algorithm has converged
		myOptimisationSuccessful = false
		show(results)
		@warn(
		      string(
			     "\nMeanVaRPrincipalScenariosModule 102:\n",
			     "The optimsation has not converged."
			     )
		      )
	end

	# retrieves the results of the optimisation
	myMinimiser = Optim.minimizer(results)

	myMinimiserTradingPlan          = myMinimiser[1:myNTradingPeriods-1]
	myMinimiserRedistributionMatrix = myMinimiser[myNTradingPeriods:end]

	println("\n\nNumber of iterations: ",Optim.iterations(results))
	println("Optimal objective function: ",Optim.minimum(results))
	println("Optimal trading plan variables: ",myMinimiserTradingPlan)
	println("Optimal redistribution matrix variables: ",myMinimiserRedistributionMatrix)

	# gets the optimal strategy
	myOptimalStrategy = get_strategy_from_decision_variables(
								 aNTradingPeriods               = myNTradingPeriods,
								 aTradingPlanVariables          = myMinimiserTradingPlan,
								 aRedistributionMatrixVariables = myMinimiserRedistributionMatrix
								 )

	myOptimalObjectiveValue = Optim.minimum(results)

	return myOptimalStrategy, myOptimalObjectiveValue, myOptimisationSuccessful
end

end
