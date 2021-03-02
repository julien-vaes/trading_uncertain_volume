# NAME: simulation_pararmeters.jl
# AUTHOR: Julien Vaes
# DATE: May 17, 2019
# DESCRIPTION: A structure containing all details about the simulation parameters.

module SimulationParametersModule

############################
## Load Personal Packages ##
############################

using ..StrategyModule

###################
## Load Packages ##
###################

import Base: hash, rand

######################
## Export functions ##
######################

export SimulationParameters
export get_method, get_uncertainty_generation_method
export get_optimise_trading_plan, get_optimise_redistribution_matrix
export get_algorithm
export get_parameters_CVaR_optimisation
export get_parameters_ne_mean_CVaR
export get_consider_recourse
export get_recompute_optimal_strategies, get_recompute_performances
export get_functions_show_print_logs
export get_functions_not_show_info_logs
export get_CVaR_optimisation_n_samples, get_CVaR_optimisation_n_samples_per_iteration, get_CVaR_optimisation_maximum_number_of_iterations, get_CVaR_optimisation_adaptive_number_of_samples
export get_ne_mean_CVaR_adaptive_number_of_samples_to_find_ne, get_ne_mean_CVaR_max_number_turns_best_response, get_ne_mean_CVaR_convergence_tol_ne, get_ne_mean_CVaR_method_find_ne, get_ne_mean_CVaR_adaptive_step_size_best_response, get_ne_mean_CVaR_step_size_best_response
export get_simulation_parameters, get_new_simulation_parameters
export get_dict_from_simulation_parameters, get_simulation_parameters_from_dict
export hash_simulation_parameters

######################
## Module functions ##
######################

### START: STRUCTURE SimulationParameters ###

# A structure containing all details for a structurename.
# The attributes in the structure: theMethod, theUncertaintyGenerationMethod, theOptimiseTradingPlan, theOptimiseRedistributionMatrix, theAlgorithm, theParametersCVaROptimisation, theParametersNEMeanCVaR, theConsiderRecourse, theRecomputeOptimalStrategies, theRecomputePerformances, theFunctionsShowPrintLogs, theFunctionsNotShowInfoLogs
struct SimulationParameters 
	theMethod #* the name of the method to use: (i) `MeanVariance`: optimisation under price uncertainty in the Expectation-Variance framework (Almgren and Chriss), (ii) `MeanCVaR`: optimisation under price and volume uncertainty in the Expectation-CVaR framework (Vaes and Hauser).
	theUncertaintyGenerationMethod 	 # TODO.
	theOptimiseTradingPlan 	         # TODO.
	theOptimiseRedistributionMatrix  # TODO.
	theAlgorithm 	                 # TODO.
	theParametersCVaROptimisation 	 # TODO.
	theParametersNEMeanCVaR  	 # TODO.
	theConsiderRecourse 	         # TODO.
	theRecomputeOptimalStrategies 	 # TODO.
	theRecomputePerformances 	 # TODO.
	theFunctionsShowPrintLogs 	 # TODO.
	theFunctionsNotShowInfoLogs 	 # TODO.
end

## STRUCTURE SimulationParameters: get functions
"""
```
get_method(aSimulationParameters::SimulationParameters)
```

returns the attribute `theMethod` of the structure `aSimulationParameters`.

### Argument
* `aSimulationParameters::SimulationParameters`: TODO.
"""
function get_method(aSimulationParameters::SimulationParameters)
	return aSimulationParameters.theMethod
end

"""
```
get_uncertainty_generation_method(aSimulationParameters::SimulationParameters)
```

returns the attribute `theUncertaintyGenerationMethod` of the structure `aSimulationParameters`.

### Argument
* `aSimulationParameters::SimulationParameters`: TODO.
"""
function get_uncertainty_generation_method(aSimulationParameters::SimulationParameters)
	return aSimulationParameters.theUncertaintyGenerationMethod
end

"""
```
get_optimise_trading_plan(aSimulationParameters::SimulationParameters)
```

returns the attribute `theOptimiseTradingPlan` of the structure `aSimulationParameters`.

### Argument
* `aSimulationParameters::SimulationParameters`: TODO.
"""
function get_optimise_trading_plan(aSimulationParameters::SimulationParameters)
	return aSimulationParameters.theOptimiseTradingPlan
end

"""
```
get_optimise_redistribution_matrix(aSimulationParameters::SimulationParameters)
```

returns the attribute `theOptimiseRedistributionMatrix` of the structure `aSimulationParameters`.

### Argument
* `aSimulationParameters::SimulationParameters`: TODO.
"""
function get_optimise_redistribution_matrix(aSimulationParameters::SimulationParameters)
	return aSimulationParameters.theOptimiseRedistributionMatrix
end

"""
```
get_algorithm(aSimulationParameters::SimulationParameters)
```

returns the attribute `theAlgorithm` of the structure `aSimulationParameters`.

### Argument
* `aSimulationParameters::SimulationParameters`: TODO.
"""
function get_algorithm(aSimulationParameters::SimulationParameters)
	return aSimulationParameters.theAlgorithm
end

"""
```
get_parameters_CVaR_optimisation(aSimulationParameters::SimulationParameters)
```

returns the attribute `theParametersCVaROptimisation` of the structure `aSimulationParameters`.

### Argument
* `aSimulationParameters::SimulationParameters`: TODO.
"""
function get_parameters_CVaR_optimisation(aSimulationParameters::SimulationParameters)
	return aSimulationParameters.theParametersCVaROptimisation
end

"""
```
get_parameters_ne_mean_CVaR(aSimulationParameters::SimulationParameters)
```

returns the attribute `theParametersCVaROptimisation` of the structure `aSimulationParameters`.

### Argument
* `aSimulationParameters::SimulationParameters`: TODO.
"""
function get_parameters_ne_mean_CVaR(aSimulationParameters::SimulationParameters)
	return aSimulationParameters.theParametersNEMeanCVaR
end

"""
```
get_consider_recourse(aSimulationParameters::SimulationParameters)
```

returns the attribute `theConsiderRecourse` of the structure `aSimulationParameters`.

### Argument
* `aSimulationParameters::SimulationParameters`: TODO.
"""
function get_consider_recourse(aSimulationParameters::SimulationParameters)
	return aSimulationParameters.theConsiderRecourse
end

"""
```
get_recompute_optimal_strategies(aSimulationParameters::SimulationParameters)
```

returns the attribute `theRecomputeOptimalStrategies` of the structure `aSimulationParameters`.

### Argument
* `aSimulationParameters::SimulationParameters`: TODO.
"""
function get_recompute_optimal_strategies(aSimulationParameters::SimulationParameters)
	return aSimulationParameters.theRecomputeOptimalStrategies
end

"""
```
get_recompute_performances(aSimulationParameters::SimulationParameters)
```

returns the attribute `theRecomputePerformances` of the structure `aSimulationParameters`.

### Argument
* `aSimulationParameters::SimulationParameters`: TODO.
"""
function get_recompute_performances(aSimulationParameters::SimulationParameters)
	return aSimulationParameters.theRecomputePerformances
end

"""
```
get_functions_show_print_logs(aSimulationParameters::SimulationParameters)
```

returns the attribute `theFunctionsShowPrintLogs` of the structure `aSimulationParameters`.

### Argument
* `aSimulationParameters::SimulationParameters`: TODO.
"""
function get_functions_show_print_logs(aSimulationParameters::SimulationParameters)
	return aSimulationParameters.theFunctionsShowPrintLogs
end

"""
```
get_functions_not_show_info_logs(aSimulationParameters::SimulationParameters)
```

returns the attribute `theFunctionsNotShowInfoLogs` of the structure `aSimulationParameters`.

### Argument
* `aSimulationParameters::SimulationParameters`: TODO.
"""
function get_functions_not_show_info_logs(aSimulationParameters::SimulationParameters)
	return aSimulationParameters.theFunctionsNotShowInfoLogs
end

### END: STRUCTURE SimulationParameters ###

#######################################################################
# Functions to get access to the parameters for the CVaR optimisation #
#######################################################################

"""
```
get_CVaR_optimisation_n_samples(aSimulationParameters::SimulationParameters)
```

TODO function description.

### Argument
* `aSimulationParameters::SimulationParameters`: TODO.
"""
function get_CVaR_optimisation_n_samples(aSimulationParameters::SimulationParameters)
	return get_parameters_CVaR_optimisation(aSimulationParameters)["NSamples"]
end

"""
```
get_CVaR_optimisation_n_samples_per_iteration()
```

TODO function description.

### Argument
* ``: TODO.
"""
function get_CVaR_optimisation_n_samples_per_iteration(aSimulationParameters::SimulationParameters)
	return get_parameters_CVaR_optimisation(aSimulationParameters)["NSamplesPerIteration"]
end

"""
```
get_CVaR_optimisation_maximum_number_of_iterations()
```

TODO function description.

### Argument
* ``: TODO.
"""
function get_CVaR_optimisation_maximum_number_of_iterations(aSimulationParameters::SimulationParameters)
	return get_parameters_CVaR_optimisation(aSimulationParameters)["MaximumNumberOfIterations"]
end

"""
```
get_CVaR_optimisation_adaptive_number_of_samples()
```

TODO function description.

### Argument
* ``: TODO.
"""
function get_CVaR_optimisation_adaptive_number_of_samples(aSimulationParameters::SimulationParameters)
	return get_parameters_CVaR_optimisation(aSimulationParameters)["AdaptiveNumberOfSamples"]
end

#############################################################################
# Functions to get access to the parameters for the Nash equilibrium search #
#############################################################################

"""
```
get_ne_mean_CVaR_adaptive_number_of_samples_to_find_ne(aSimulationParameters)
```

TODO function description.

### Argument
* `aSimulationParameters`: TODO.
"""
function get_ne_mean_CVaR_adaptive_number_of_samples_to_find_ne(aSimulationParameters)
	return get_parameters_ne_mean_CVaR(aSimulationParameters)["AdaptiveNumberOfSamplesToFindNE"]
end

"""
```
get_ne_mean_CVaR_max_number_turns_best_response(aSimulationParameters)
```

TODO function description.

### Argument
* `aSimulationParameters`: TODO.
"""
function get_ne_mean_CVaR_max_number_turns_best_response(aSimulationParameters)
	return get_parameters_ne_mean_CVaR(aSimulationParameters)["MaxNumberTurnsOfBestResponse"]
end

"""
```
get_ne_mean_CVaR_convergence_tol_ne(aSimulationParameters)
```

TODO function description.

### Argument
* `aSimulationParameters`: TODO.
"""
function get_ne_mean_CVaR_convergence_tol_ne(aSimulationParameters)
	return get_parameters_ne_mean_CVaR(aSimulationParameters)["ConvergenceTolNE"]
end

"""
```
get_ne_mean_CVaR_method_find_ne(aSimulationParameters)
```

TODO function description.

### Argument
* `aSimulationParameters`: TODO.
"""
function get_ne_mean_CVaR_method_find_ne(aSimulationParameters)
	return get_parameters_ne_mean_CVaR(aSimulationParameters)["MethodFindNE"]
end

"""
```
get_ne_mean_CVaR_adaptive_step_size_best_response(aSimulationParameters)
```

TODO function description.

### Argument
* `aSimulationParameters`: TODO.
"""
function get_ne_mean_CVaR_adaptive_step_size_best_response(aSimulationParameters)
	return get_parameters_ne_mean_CVaR(aSimulationParameters)["AdaptiveStepSizeBestResponse"]
end

"""
```
get_ne_mean_CVaR_step_size_best_response(aSimulationParameters)
```

TODO function description.

### Argument
* `aSimulationParameters`: TODO.
"""
function get_ne_mean_CVaR_step_size_best_response(aSimulationParameters)
	return get_parameters_ne_mean_CVaR(aSimulationParameters)["StepSizeBestResponse"]
end

#######################################################
# Function to create a structure SimulationParameters #
#######################################################

"""
#### Definition
```
get_simulation_parameters(;
	aMethod="Mean-CVaR",
	aUncertaintyGenerationMethod="normal",
	aOptimiseTradingPlan::Bool=true,
	aOptimiseRedistributionMatrix::Bool=false,
	aAlgorithm::String = "BFGS",
	aParametersCVaROptimisation::Dict = Dict("NSamples" => 10^6, "NSamplesPerIteration" => 10^4, "MaximumNumberOfIterations" => 200, "AdaptiveNumberOfSamples" => true),
	aConsiderRecourse::Bool=false,
	aRecomputeOptimalStrategies::Bool=true,
	aRecomputePerformances::Bool=true,
	aFunctionsShowPrintLogs::Dict{String,Bool}=Dict{String,Bool}(),
	aFunctionsNotShowInfoLogs::Dict{String,Bool}=Dict{String,Bool}()
	)
```

returns a `SimulationParameters` structure with the attributes corresponding to the argument given.

#### Arguments
* `aMethod="Mean-CVaR"`: a method to use to solve the trading plan.
* `aNSamples::Int=10^7`: a number of samples needed to compute CVaR.
* `aNSamplesPerIteration::Int64=10^4`: a the number of samples computed at the same time on one processor.
* `aMaximumNumberOfIterations::Int64=200`: a maximum number of iteration in the gradient descent method.
* `aConsiderRecourse=false`: a boolean to tell if the performance should be computed as if recourse is allowed at each time step.
"""
function get_simulation_parameters(;
				   aMethod="Mean-CVaR",
				   aUncertaintyGenerationMethod="normal",
				   aOptimiseTradingPlan::Bool=true,
				   aOptimiseRedistributionMatrix::Bool=false,
				   aAlgorithm::String = "BFGS",
				   aParametersCVaROptimisation::Dict = Dict("NSamples" => 10^6, "NSamplesPerIteration" => 10^4, "MaximumNumberOfIterations" => 200, "AdaptiveNumberOfSamples" => true),
				   aParametersNEMeanCVaR::Dict = Dict("AdaptiveNumberOfSamplesToFindNE" => false, "MaxNumberTurnsOfBestResponse" => 200, "ConvergenceTolNE" => 5*10.0^-6, "MethodFindNE" => "Seidel", "AdaptiveStepSizeBestResponse" => true, "StepSizeBestResponse" => 1.0),
				   aConsiderRecourse::Bool=false,
				   aRecomputeOptimalStrategies::Bool=true,
				   aRecomputePerformances::Bool=true,
				   aFunctionsShowPrintLogs::Dict{String,Bool}=Dict{String,Bool}(),
				   aFunctionsNotShowInfoLogs::Dict{String,Bool}=Dict{String,Bool}()
				   )


	################################
	# CVaR optimisation parameters #
	################################
	
	# the default parameters for the cvar optimisation
	myDefaultParametersCVaROptimisation = Dict("NSamples" => 10^6, "NSamplesPerIteration" => 10^4, "MaximumNumberOfIterations" => 200, "AdaptiveNumberOfSamples" => true)

	# gets the paramters for the CVaR optimisation, if both dictionaries have the same keys the values of `aParametersCVaROptimisation` is taken (priority to the second argument). 
	myParametersCVaROptimisation = merge(myDefaultParametersCVaROptimisation,aParametersCVaROptimisation)

	######################################
	# Nash equilibrium search parameters #
	######################################

	# the default parameters for the search of the Nash equilibrium
	myDefaultParametersParametersNEMeanCVaR = Dict("AdaptiveNumberOfSamplesToFindNE" => false, "MaxNumberTurnsOfBestResponse" => 200, "ConvergenceTolNE" => 5*10.0^-6, "MethodFindNE" => "Seidel", "AdaptiveStepSizeBestResponse" => true, "StepSizeBestResponse" => 1.0)

	# gets the paramters for the search of the Nash equilibrium, if both dictionaries have the same keys the values of `aParametersNEMeanCVaR` is taken (priority to the second argument). 
	myParametersNEMeanCVaR = merge(myDefaultParametersParametersNEMeanCVaR,aParametersNEMeanCVaR)

	return SimulationParameters(
				    aMethod,
				    aUncertaintyGenerationMethod,
				    aOptimiseTradingPlan,
				    aOptimiseRedistributionMatrix,
				    aAlgorithm,
				    myParametersCVaROptimisation,
				    myParametersNEMeanCVaR,
				    aConsiderRecourse,
				    aRecomputeOptimalStrategies,
				    aRecomputePerformances,
				    aFunctionsShowPrintLogs,
				    aFunctionsNotShowInfoLogs
				    )
end

"""
    get_new_simulation_parameters(
    	aSimulationParameters::SimulationParameters;
    	aNSamples::Int64=get_CVaR_optimisation_n_samples(aSimulationParameters),
    	aNSamplesPerIteration::Int64=get_CVaR_optimisation_n_samples_per_iteration(aSimulationParameters),
    	aMaximumNumberOfIterations::Int64=get_CVaR_optimisation_maximum_number_of_iterations(aSimulationParameters),
    	aMethod=get_method(aSimulationParameters),
    	aConsiderRecourse=get_consider_recourse(aSimulationParameters)
    )

returns a structure identical to the one given in argument except if other values for its attributes are specified in the arguments.

#### Arguments
* `aSimulationParameters::SimulationParameters`: a SimulationParameters structure from which to copy the argumetns if not specied otherwise.
* `aNSamples::Int64=get_CVaR_optimisation_n_samples(aSimulationParameters)`: a number of samples needed to compute CVaR.
* `aNSamplesPerIteration::Int64=get_CVaR_optimisation_n_samples_per_iteration(aSimulationParameters)`: a the number of samples computed at the same time on one processor.
* `aMaximumNumberOfIterations::Int64=get_CVaR_optimisation_maximum_number_of_iterations(aSimulationParameters)`: a maximum number of iteration in the gradient descent method.
* `aMethod=get_method(aSimulationParameters)`: a method to use to solve the trading plan.
* `aConsiderRecourse=get_consider_recourse(aSimulationParameters)`: a boolean to tell if the performance should be computed as if recourse is allowed at each time step.
"""
function get_new_simulation_parameters(
				       aSimulationParameters::SimulationParameters;
				       aMethod=get_method(aSimulationParameters),
				       aUncertaintyGenerationMethod=get_uncertainty_generation_method(aSimulationParameters),
				       aOptimiseTradingPlan::Bool=get_optimise_trading_plan(aSimulationParameters),
				       aOptimiseRedistributionMatrix::Bool=get_optimise_redistribution_matrix(aSimulationParameters),
				       aAlgorithm::String=get_algorithm(aSimulationParameters),
				       aParametersCVaROptimisation::Dict = get_parameters_CVaR_optimisation(aSimulationParameters),
				       aParametersNEMeanCVaR::Dict = get_parameters_ne_mean_CVaR(aSimulationParameters),
				       aConsiderRecourse::Bool=get_consider_recourse(aSimulationParameters),
				       aRecomputeOptimalStrategies::Bool=get_recompute_optimal_strategies(aSimulationParameters),
				       aRecomputePerformances::Bool=get_recompute_performances(aSimulationParameters),
				       aFunctionsShowPrintLogs::Dict=get_functions_show_print_logs(aSimulationParameters),
				       aFunctionsNotShowInfoLogs::Dict=get_functions_not_show_info_logs(aSimulationParameters),
				       )

	# gets the paramters for the CVaR optimisation, if both dictionaries have the same keys the values of `aParametersCVaROptimisation` is taken (priority to the second argument). 
	myParametersCVaROptimisation = merge(get_parameters_CVaR_optimisation(aSimulationParameters), aParametersCVaROptimisation)

	# gets the paramters for the search of the Nash equilibrium, if both dictionaries have the same keys the values of `aParametersNEMeanCVaR` is taken (priority to the second argument). 
	myParametersNEMeanCVaR = merge(get_parameters_ne_mean_CVaR(aSimulationParameters), aParametersNEMeanCVaR)

	return SimulationParameters(
				    aMethod,
				    aUncertaintyGenerationMethod,
				    aOptimiseTradingPlan,
				    aOptimiseRedistributionMatrix,
				    aAlgorithm,
				    myParametersCVaROptimisation,
				    myParametersNEMeanCVaR,
				    aConsiderRecourse,
				    aRecomputeOptimalStrategies,
				    aRecomputePerformances,
				    aFunctionsShowPrintLogs,
				    aFunctionsNotShowInfoLogs
				    )
end

"""
    get_simulation_parameters_from_dict(aDict::Dict)

return a `SimulationParameters` structure based on the details given in `aDict`.
This function is used when loading a `SimulationParameters` structure from a file.
This function is mainly useful to store results of simulations and load them back again.

#### Argument
* `aDict::Dict`: a dictionary with the details of the simulation parameters.
"""
function get_simulation_parameters_from_dict(aDict::Dict)
	return SimulationParameters(
				    aDict["Method"],
				    aDict["UncertaintyGenerationMethod"],
				    aDict["OptimiseTradingPlan"],
				    aDict["OptimiseRedistributionMatrix"],
				    aDict["Algorithm"],
				    aDict["ParametersCVaROptimisation"],
				    aDict["ConsiderRecourse"]
				    )
end

"""
    get_dict_from_simulation_parameters(aSimulationParameters::SimulationParameters)

returns a dictionary that contains all the informations of the simulation paramerters `SimulationParameters`.
This function is used to store a `SimulationParameters` structure into a file.
This function is mainly useful to store results of simulations and load them back again.

#### Argument
* `aSimulationParameters::SimulationParameters`: a structure containing all the details of the parameters used to compute the optimal trading plan such as the method to use, the number of samples in the CVaR estimation.
"""
function get_dict_from_simulation_parameters(aSimulationParameters::SimulationParameters)
	myDict = Dict()
	myDict["Method"]                       = get_method(aSimulationParameters)
	myDict["UncertaintyGenerationMethod"]  = get_uncertainty_generation_method(aSimulationParameters)
	myDict["OptimiseTradingPlan"]          = get_optimise_trading_plan(aSimulationParameters) ? 1 : 0
	myDict["OptimiseRedistributionMatrix"] = get_optimise_redistribution_matrix(aSimulationParameters) ? 1 : 0
	myDict["Algorithm"]                    = get_algorithm(aSimulationParameters)
	myDict["ParametersCVaROptimisation"]   = get_parameters_CVaR_optimisation(aSimulationParameters)
	myDict["ConsiderRecourse"]             = get_consider_recourse(aSimulationParameters) ? 1 : 0

	return myDict
end

"""
    hash_simulation_parameters(aSimulationParameters::SimulationParameters)

TODO function description.

#### Argument
* `aSimulationParameters::SimulationParameters`: TODO.
"""
function hash_simulation_parameters(aSimulationParameters::SimulationParameters)
	myDictionary = nothing
	if get_method(aSimulationParameters) == "Mean-Variance" # delete entries not related to the Mean-Variance model
		myDictionary = get_dict_from_simulation_parameters(aSimulationParameters)
		delete!(myDictionary,"Algorithm")
		delete!(myDictionary,"ParametersCVaROptimisation")
	else
		myDictionary = deepcopy(get_dict_from_simulation_parameters(aSimulationParameters))
		myDictParametersCVaROptimisation = copy(myDictionary["ParametersCVaROptimisation"])
		delete!(myDictionary,"ParametersCVaROptimisation")
		delete!(myDictParametersCVaROptimisation,"NSamplesPerIteration")
		myDictionary["ParametersCVaROptimisation"] = myDictParametersCVaROptimisation
	end
	return hash(myDictionary)
end

end
