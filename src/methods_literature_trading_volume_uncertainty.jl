# NAME: methods_literature_trading_volume_uncertainty.jl
# AUTHOR: Julien Vaes
# DATE: April 15, 2021
# DESCRIPTION: module with the different methods to compare ours with.
# The methods are one of the paper of Cheng et al. entitled ``Optimal execution with uncertain order fills in Almgren–Chriss framework'' [https://doi.org/10.1080/14697688.2016.1185531]

module MethodsLiterature

###################
## Load Packages ##
###################

import Base: hash, rand

using Distributions, StatsBase
using LinearAlgebra
using Random, Distributions

############################
## Load Personal Packages ##
############################

using ..MarketDetailsModule
using ..StrategyModule
using ..TraderModule
using ..SimulationParametersModule
using ..HelpFilesModule

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
## Export functions ##
######################

export get_trading_cost
export get_trading_cost_realisations_exponential_utility, get_trading_cost_realisations_exponential_utility_different_riskAversions
export get_trading_cost_realisations_mean_QV, get_trading_cost_realisations_mean_QV_different_riskAversions

######################
## Module functions ##
######################

function get_trading_cost(aDecisionTimes,aX,aParameters)

	# gets the parameters
	myγ  = aParameters["γ"]
	myη  = aParameters["η"]
	myμ  = aParameters["μ"]
	myρ  = aParameters["ρ"]
	myβ  = aParameters["β"]
	myσ  = aParameters["σ"]
	myS0 = aParameters["S0"]
	myD0 = aParameters["D0"]

	# the time step
	t  = aDecisionTimes
	Δt = step(t)
	d  = Normal(0,1)

	# the volumes traded
	myN = aX[2:end] - aX[1:end-1] # those are negative in a liquidation problem
	myN = - myN # make the volume position to have the equivalent acquisition problem

	# initial trading price
	myS = myS0 # variable that represent the price process

	########################
	# Initial trading cost #
	########################
	
	# The way to compute the initial the trading cost depends on the method used
	
	# If aParameters["CostEstimateMethod"] = "All":
	# 	the global trading cost: ∑i ni * ( S0 + ∑k ξk + γk nk ) +  Σi (ηi/τi) ni^2, with ∑i ni = DT
	# If aParameters["CostEstimateMethod"] = "All - S0*D0":
	#	 the global trading cost minus the fixed cost S0 * D0:
	#	 	∑i ni * ( S0 + ∑k ξk + γk nk ) +  Σi (ηi/τi) ni^2 - S0 * D0, with ∑i ni = DT
	# If aParameters["CostEstimateMethod"] = "All - S0*DT":
	# 	the global trading cost minus the cost related to the initial price and the final demand S0 * DT:
	# 		∑i ni * ( S0 + ∑k ξk + γk nk ) +  Σi (ηi/τi) ni^2 - S0 * DT, with ∑i ni = DT
	# If aParameters["CostEstimateMethod"] = "All - [ S0*D0 + ∑i δi * ( S0 + ∑k=1^i-1 (ξk + γk nk) ) ]":
	# 	the global trading cost minus the cost related to the initial price and the cost due to the volume updates given the price at the start of the trading period where there is the update:
	# 		∑i (ni - δi) * ( S0 + ∑k=1^i-1 ξk + γk nk ) +  Σi (ηi/τi) ni^2 - S0 * DT,   with ∑i ni = DT (this is used to compare with the dynamic program version)
	
	# initialises the trading cost variable
	myCost = nothing

	myCostEstimateMethod = aParameters["CostEstimateMethod"]
	if myCostEstimateMethod == "All"
		myCost = 0.0
	elseif myCostEstimateMethod == "All - S0*D0"
		myCost = - myS0 * myD0
	elseif myCostEstimateMethod == "All - S0*DT"
		myCost = - myS0 * sum(myN)
	elseif myCostEstimateMethod == "All - [ S0*D0 + ∑i δi * ( S0 + ∑k=1^i-1 (ξk + γk nk) ) ]"
		#= myCost = - myS0 * sum(myN) # it remains now to remove ∑i δi * ( ∑k=1^i-1 (ξk + γk nk) ) =#
		@error(string(
			      "\nMethodsLiterature 102:\n",
			      "The method ",myCostEstimateMethod," is not yet supported in this version." # TODO
			      )
		       )
	else
		@error(string(
			      "\nMethodsLiterature 101:\n",
			      "The method ",myCostEstimateMethod," to estimate the trading cost in unknown. Please use one of these methods:\n.",
			      "\"All\", \"All - S0*D0\", \"All - S0*DT\",  or \"All - [ S0*D0 + ∑i δi * ( S0 + ∑k=1^i-1 (ξk + γk nk) ) ]\""
			      )
		       )
	end
	
	# loop to compute the trading cost
	for i in eachindex(myN)
		myStilde = myS + (myη / Δt) * myN[i]
		myCost  += myStilde * myN[i]
		myS     += myσ * sqrt(Δt) * rand(d) + myγ * myN[i]
	end

	#= # loop to compute the trading cost =#
	#= myCost2 = 0.0 =#
	#= for i in eachindex(myN) =#
	#= 	myStilde = myS + myη * myN[i] / Δt  # checks if not Δt =#
	#= 	myS     += myσ * sqrt(Δt) * rand(d) + myγ * myN[i] =#

	#= 	myCost2 += i < size(myN,1) ? myS * sum(myN[i+1:end]) : 0.0 =#
	#= 	myCost2 += myη * myN[i]^2 / Δt =#
	#= end =#

	return myCost
end

function get_parameters_market(;aParameters,aVolumeUncertainty)

	# gets the parameters
	myNPeriods = aParameters["NPeriods"]
	myT  = aParameters["T"]
	myS0 = aParameters["S0"]
	myD0 = aParameters["D0"]
	myσ  = aParameters["σ"]
	mym0 = aParameters["m0"]
	if !(aVolumeUncertainty)
		mym0 = 10.0^-10
	end
	myγ = aParameters["γ"]
	myη = aParameters["η"]
	myμ = aParameters["μ"]
	myρ = aParameters["ρ"]
	myβ = aParameters["β"]

	# returns the parameters
	return myNPeriods, myT, myS0, myD0, myσ, mym0, myγ, myη, myμ, myρ, myβ
end

function fill_optimal_trajectory_realisation!(;aDecisionTimes,aXStar,aFunctionXStarNoVolume,aFunctionFactorOutIntegral,aFunctionFactorInIntegral)

	# get the step of the decision times
	Δt = step(aDecisionTimes)
	d  = Normal(0,1)

	# compute simultaneously the integral and the optimal liquidation profile
	myValueIntegral = 0.0
	for i in eachindex(aDecisionTimes)
		myValueIntegral += aFunctionFactorInIntegral(aDecisionTimes[i]) * rand(d) * Δt
		aXStar[i]        = aFunctionXStarNoVolume(aDecisionTimes[i]) + aFunctionFactorOutIntegral(aDecisionTimes[i]) * myValueIntegral
	end
	aXStar[end] = 0.0 # otherwise give a NaN
end

#######################
# Exponential utility #
#######################

function get_parameters_exponential_utility(;aθ,aParameters,aVolumeUncertainty)

	# gets the parameters of the market
	myNPeriods, myT, myS0, myD0, myσ, mym0, myγ, myη, myμ, myρ, myβ = get_parameters_market(
												aParameters = aParameters,
												aVolumeUncertainty = aVolumeUncertainty
												)

	# computes the parameters useful for the computation of the optimal strategy
	myH    = aθ * myη * ( 1 + ( mym0^2 * aθ * myη * ( myρ^2 + ( 1 - myρ )^2 ) ) / 2 )
	myl1   = ( ( aθ^2 * myσ^2 ) / 2 ) * myH
	mysl1  = sqrt(myl1)
	mysl1H = mysl1 / myH
	myl3   = mym0 * myη * aθ^2 * myρ * myσ / 2
	myA    = acoth( ( myl3 - aθ * ( myγ/2 - myβ ) ) / mysl1 )

	return myH, myl1, mysl1, mysl1H, myl3, myA
end

function get_trading_cost_realisations_exponential_utility(;
							   aθ,
							   aParameters,
							   aNSamples,
							   aVolumeUncertainty,
							   aRecomputeRealisations::Bool=true,
							   aOutputFolder::String="outputs/trading_cost_realisations/exponential-utility/"
							   )

	# the file in which the realisations will be saved
	myOutputFilePath = string(aOutputFolder,"ExponentialUtility_",hash(string(aθ,hash(aParameters),aNSamples,aVolumeUncertainty)),"_realisations.jld2")
	create_relevant_folders!(myOutputFilePath)

	# boolean forcing the computation of the needed if needed, initialise the value to whether an associated performance file exists or not
	myShouldGenerateRealisations = aRecomputeRealisations ? true : !(isfile(myOutputFilePath))

	if !(myShouldGenerateRealisations)
		myCostRealisationsDict = HelpFilesModule.load_result(myOutputFilePath) # reads the file
		return myCostRealisationsDict["TradingCostRealisations"], myOutputFilePath
	end

	# gets the parameters of the market
	myNPeriods, myT, myS0, myD0, myσ, mym0, myγ, myη, myμ, myρ, myβ = get_parameters_market(
												aParameters = aParameters,
												aVolumeUncertainty = aVolumeUncertainty
												)

	if myμ != 0.0
		@error(string(
			      "\nMethodsLiterature 103:\n",
			      "The trading cost implementation is not valid for a drift μ different from 0.0."
			      )
		       )
	end

	# gets the parameters useful for the computation of the optimal strategy
	myH, myl1, mysl1, mysl1H, myl3, myA = get_parameters_exponential_utility(
										 aθ = aθ,
										 aParameters = aParameters,
										 aVolumeUncertainty = aVolumeUncertainty
										 )

	# values to compute the optimal liquidity profile
	myb2(t) = mysl1 * coth( myA + mysl1H * (myT - t) ) - myl3
	myb1(t) = 0 # since assume a zero drift
	myc(t)  = ( ( myρ * myσ * mym0 + ( myγ / 2 ) * mym0^2 ) * aθ - mym0^2 * myl3 ) * (myT - t) + mym0^2 * myH * ( log( sinh( myA  + mysl1H * (myT-t) ) ) - log( sinh( myA ) ) )

	# functions to get the optimal trajectory
	my_x_star_no_volume(t)     =      ( sinh( mysl1H * (myT - t) ) / sinh( mysl1H * myT ) )^( 2 * myH / (aθ * myη) - 1 ) * exp( (2 * myl3 * t) / (aθ * myη) ) * myD0
	my_factor_out_integral(t)  = mym0 * ( ( sinh( mysl1H * (myT - t) ) )^( 2 * myH / ( aθ * myη ) - 1 ) ) * exp( 2 * myl3 * t / ( aθ * myη ) )
	my_factor_in_integral(t)   =      ( 1 / sinh( mysl1H * (myT - t) ) )^( 2 * myH / ( aθ * myη ) - 1 )   * exp( 2 * myl3 * ( - t) / ( aθ * myη ) )

	#########################################
	# Compute the trading cost realisations #
	#########################################

	# gets the trading cost realisations
	myCostRealisations = []

	# the range of times and initialisation of a vector containing the optimal liquidation profile realisation
	t = range(0, stop = myT, length = myNPeriods+1)
	myXStar = zeros(size(t))

	myCostRealisations = zeros(aNSamples)
	for i in 1:aNSamples
		fill_optimal_trajectory_realisation!(
						     aDecisionTimes             = t,
						     aXStar                     = myXStar,
						     aFunctionXStarNoVolume     = my_x_star_no_volume,
						     aFunctionFactorOutIntegral = my_factor_out_integral,
						     aFunctionFactorInIntegral  = my_factor_in_integral
						     )
		myCostRealisations[i] = get_trading_cost(t,myXStar,aParameters)
	end 

	# delete the old output file if it exists
	if isfile(myOutputFilePath)
		rm(myOutputFilePath, force=true) 
	end

	# saves the output
	save_result!(myOutputFilePath,Dict("TradingCostRealisations"=>myCostRealisations))

	return myCostRealisations, myOutputFilePath
end

function get_trading_cost_realisations_exponential_utility_different_riskAversions(;
										   aθs,
										   aParameters,
										   aNSamples,
										   aVolumeUncertainty,
										   aRecomputeRealisations::Bool=true,
										   )

	# gets the trading cost realisations
	myCostRealisations = []
	myOutputFiles = []
	myLabels = []

	for myθ in aθs
		myCostRealisationsLocal, myOutputFileLocal = get_trading_cost_realisations_exponential_utility(
													       aθ = myθ,
													       aParameters = aParameters,
													       aNSamples = aNSamples,
													       aVolumeUncertainty = aVolumeUncertainty,
													       aRecomputeRealisations = aRecomputeRealisations
													       )

		push!(myCostRealisations,myCostRealisationsLocal)
		push!(myOutputFiles,myOutputFileLocal)
		push!(myLabels,string("Exponantial utility ","θ = "," $myθ"))
	end

	return myCostRealisations, myOutputFiles, myLabels
end

##########################
# Mean - QV optimisation #
##########################

function get_parameters_mean_QV(;
				aλ,
				aParameters,
				aVolumeUncertainty
				)

	# gets the parameters of the market
	myNPeriods, myT, myS0, myD0, myσ, mym0, myγ, myη, myμ, myρ, myβ = get_parameters_market(
												aParameters = aParameters,
												aVolumeUncertainty = aVolumeUncertainty
												)

	# computes the parameters useful for the computation of the optimal strategy
	myk1 = aλ * myη * ( 1 + aλ * mym0^2 * myη ) * ( myσ^2 + mym0^2 * myγ^2 + 2 * myρ * myσ * myγ * mym0 )
	myk2 = myη * ( 1 + aλ * mym0^2 * myη )
	myζ  = sqrt(myk1) / myk2

	return myk1, myk2, myζ
end

function get_trading_cost_realisations_mean_QV(;
					       aλ,
					       aParameters,
					       aNSamples,
					       aVolumeUncertainty,
					       aRecomputeRealisations::Bool=true,
					       aOutputFolder::String="outputs/trading_cost_realisations/mean-QV/"
					       )

	# the file in which the realisations will be saved
	myOutputFilePath = string(aOutputFolder,"Mean-QV_",hash(string(aλ,hash(aParameters),aNSamples,aVolumeUncertainty)),"_realisations.jld2")
	create_relevant_folders!(myOutputFilePath)

	# boolean forcing the computation of the needed if needed, initialise the value to whether an associated performance file exists or not
	myShouldGenerateRealisations = aRecomputeRealisations ? true : !(isfile(myOutputFilePath))

	if !(myShouldGenerateRealisations)
		myCostRealisationsDict = HelpFilesModule.load_result(myOutputFilePath) # reads the file
		return myCostRealisationsDict["TradingCostRealisations"], myOutputFilePath
	end

	# gets the parameters of the market
	myNPeriods, myT, myS0, myD0, myσ, mym0, myγ, myη, myμ, myρ, myβ = get_parameters_market(
												aParameters        = aParameters,
												aVolumeUncertainty = aVolumeUncertainty
												)

	# gets the parameters useful for the computation of the optimal strategy
	myk1, myk2, myζ = get_parameters_mean_QV(
						 aλ                 = aλ,
						 aParameters        = aParameters,
						 aVolumeUncertainty = aVolumeUncertainty
						 )


	# functions to get the optimal trajectory
	my_x_star_no_volume(t)    =        ( sinh( myζ * (myT - t) ) / sinh( myζ * myT ) ) * myD0
	my_factor_out_integral(t) = mym0 * ( sinh( myζ * (myT - t) ) ) # part that multiplies the integral but is not integrated
	my_factor_in_integral(t)  =    ( 1 / sinh( myζ * (myT - t) ) ) # part that is integrated

	#########################################
	# Compute the trading cost realisations #
	#########################################

	# gets the trading cost realisations
	myCostRealisations = []

	# the range of times and initialisation of a vector containing the optimal liquidation profile realisation
	t = range(0, stop = myT, length = myNPeriods+1)
	myXStar = zeros(size(t))

	myCostRealisations = zeros(aNSamples)
	for i in 1:aNSamples
		fill_optimal_trajectory_realisation!(
						     aDecisionTimes             = t,
						     aXStar                     = myXStar,
						     aFunctionXStarNoVolume     = my_x_star_no_volume,
						     aFunctionFactorOutIntegral = my_factor_out_integral,
						     aFunctionFactorInIntegral  = my_factor_in_integral
						     )

		myCostRealisations[i] = get_trading_cost(t,myXStar,aParameters)
	end 

	# delete the old output file if it exists
	if isfile(myOutputFilePath)
		rm(myOutputFilePath, force=true) 
	end

	# saves the output
	save_result!(myOutputFilePath,Dict("TradingCostRealisations"=>myCostRealisations))

	return myCostRealisations, myOutputFilePath
end

function get_trading_cost_realisations_mean_QV_different_riskAversions(;
								       aλs,
								       aParameters,
								       aNSamples,
								       aVolumeUncertainty,
								       aRecomputeRealisations::Bool=true,
								       )

	# gets the trading cost realisations
	myCostRealisations = []
	myOutputFiles = []
	myLabels = []

	for myλ in aλs
		myCostRealisationsLocal, myOutputFileLocal = get_trading_cost_realisations_mean_QV(
												   aλ                     = myλ,
												   aParameters            = aParameters,
												   aNSamples              = aNSamples,
												   aVolumeUncertainty     = aVolumeUncertainty,
												   aRecomputeRealisations = aRecomputeRealisations
												   )
		push!(myCostRealisations,myCostRealisationsLocal)
		push!(myOutputFiles,myOutputFileLocal)
		push!(myLabels,string("Mean QV ","λ = "," $myλ"))
	end

	return myCostRealisations, myOutputFiles, myLabels
end

end
