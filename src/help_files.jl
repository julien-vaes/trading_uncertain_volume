# NAME: help_files.jl
# AUTHOR: Julien Vaes
# DATE: May 23, 2019
# DESCRIPTION: module with function helpful to the storage of the results and the gestion of files

module HelpFilesModule

###################
## Load Packages ##
###################

using JLD
using JLD2
using FileIO
using Distributions
using Plots
import Base: hash, rand

############################
## Load Personal Packages ##
############################

using ..MarketDetailsModule
using ..StrategyModule
using ..TraderModule
using ..SimulationParametersModule

######################
## Export functions ##
######################

export create_relevant_folders!, test_file_existence!
export get_default_output_file_name
export get_default_output_file_path
export save_result!, load_result
export load_strategy_from_file
export print_logs
export println_logs
export info_logs
export save_figure!
export get_value_with_n_significant_number

######################
## Module functions ##
######################

"""
create_relevant_folders!(aPath::String)

creates the folders needed so that the path `aPath` exits.

#### Argument
* `aPath::String`: the path of a folder or a file.
"""
function create_relevant_folders!(aPath::String)
	# splite the file path and name with the extension
	myFilePathSplitted = split(aPath,".")
	myFolderPathandName = myFilePathSplitted[1]
	myFolderPathSplitted = split(myFolderPathandName,"/")

	# check wether the path of the file goes to a subfolder 
	if length(myFolderPathSplitted) != 1

		# deletes the file name
		myFolderPathSplitted = myFolderPathSplitted[1:end-1]

		myIterativeFolderPath = ""
		for myNewSubFolder in myFolderPathSplitted
			if myNewSubFolder != ""
				myIterativeFolderPath = string(myIterativeFolderPath,myNewSubFolder,"/")

				# checks that the subfolder exists, if not it is created
				if !(ispath(myIterativeFolderPath))
					mkdir(myIterativeFolderPath)
					@info(
					      string(
						     "\nHelpFilesModule 101:\n",
						     "Folder ",
						     myIterativeFolderPath,
						     " created."
						     )
					      )
				end
			end
		end
	end
end

"""
test_file_existence!(aFilePath::String)

raises an error if the file `aFilePath` does not exist.

#### Argument
* `aFilePath::String`: the path to the file for which one desires to verify whether it exists.
"""
function test_file_existence!(aFilePath::String)
	if !(isfile(aFilePath))
		@error(
		       string(
			      "\nHelpFilesModule 102:\n",
			      "File ",
			      aFilePath,
			      " not found."
			      )
		       )
	end
end

"""
get_default_output_file_name(
aMarketDetails::MarketDetails,
aTrader::Trader,
aSimulationParameters::SimulationParameters;
aIncludePartialMarketDetailsHash::Bool = false,
aIncludeSimulationParametersHash::Bool = true,
aSpecificFileNameExtension::String=""
)

returns the default file name.

#### Argument
* `aMarketDetails::MarketDetails`: TODO.
* `aTrader::Trader`: a structure containing all the details of a trader.
* `aSimulationParameters::SimulationParameters`: a structure containing all the details of the parameters used to compute the optimal trading plan such as the method to use, the number of samples in the CVaR estimation.
* `aIncludePartialMarketDetailsHash::Bool = false`: tells if the ground truth distribution of the market should be considered in the hash.
* `aIncludeSimulationParametersHash::Bool = true`: tells if the simulation parameters with the values relevant to the gradient descent method should be considered.
* `aSpecificFileNameExtension::String=""`: a specific file name extension.
"""
function get_default_output_file_name(;
				      aMarketDetails::MarketDetails,
				      aTraderIndex::Int,
				      aTraders::Array{Trader,1},
				      aSimulationParameters::SimulationParameters,
				      aStrategies::Array{Strategy,1},
				      aIncludePartialMarketDetailsHash::Bool,
				      aIncludeSimulationParametersHash::Bool,
				      aIncludeTradersHash::Bool,
				      aIncludeStrategiesHash::Bool,
				      aSpecificFileNameExtension::String=""
				      )

	######################
	## Hash computation ##
	######################

	# the hash of the market details, which is considered as the ground truth value for the uncertainty. If `aIncludePartialMarketDetailsHash` is 
	# `true` : the hash of the market details includes the uncertainty structure of the traders. This is useful for instance to evalute the performance of a strategy.
	# `false`: the hash of the market details but without the uncertainty structure of the traders. This is useful when computing the optimal strategy of a trader as they rely on their uncertainty belief.
	myMarketDetailsHash =  aIncludePartialMarketDetailsHash ? MarketDetailsModule.hash_partial_market_details(aMarketDetails) : MarketDetailsModule.hash_market_details(aMarketDetails)

	# the hash of the simulation parameters
	mySimulationParametersHash = aIncludeSimulationParametersHash ? SimulationParametersModule.hash_simulation_parameters(aSimulationParameters) : ""

	# the hash of the traders
	myTradersHashString = ""
	if aIncludeTradersHash
		for p in eachindex(aTraders)

			myTraderLocalHash = nothing

			# the hash of the trader `aTraderIndex`
			if p == aTraderIndex
				myTraderLocalHash = TraderModule.hash_trader(aTraders[p])
			else # the hash of any trader other than `aTraderIndex`
				myTraderLocalHash = TraderModule.hash_partial_trader(aTraders[p])
			end

			# merge the hash with the global one
			myTradersHashString = string(myTradersHashString,myTraderLocalHash)
		end
	end
	myTradersHash = hash(myTradersHashString)

	# the hash of the strategies (useful to save the performance of a strategy)
	myStrategiesHashString = ""
	if aIncludeStrategiesHash
		# includes the trader index in the strategy to make the distinction of which player is calling the function # TODO: explain better
		myStrategiesHashString = string(aTraderIndex)

		# adds the hash of each strategy
		for p in eachindex(aStrategies)
			myStrategyLocalHash = StrategyModule.hash_strategy(aStrategies[p])
			myStrategiesHashString = string(myStrategiesHashString,myStrategyLocalHash)
		end
	end
	myStrategiesHash = hash(myStrategiesHashString)

	######################
	# Merging the hashes #
	######################

	# computes a final hash based on the other hashes
	myHash = hash(string(myMarketDetailsHash,myTradersHash,mySimulationParametersHash,myStrategiesHash))

	# creates and returns the file name
	myMethod = get_method(aSimulationParameters)
	myOutputFileName = string(myMethod,"_",myHash,aSpecificFileNameExtension,".jld2")

	return myOutputFileName
end

"""
get_default_output_file_path(aMarketDetails::MarketDetails,aTrader::Trader,aSimulationParameters::SimulationParameters)

returns the default path to where the results must be saved given a `MarketDetails` structure.
By default, the values of `aIncludePartialMarketDetailsHash`, `aIncludeTradersHash`, and `aIncludeSimulationParametersHash` are set to compute the optimal trading plan and not for evaluating the performance. Hence, the hash takes into account the uncertainty structure of the Trader and not the MarketDetails, and do include the simulation parameters (as the resulting trading plan depends on the those).

#### Argument
* `aMarketDetails::MarketDetails`: TODO.
* `aTrader::Trader`: a structure containing all the details of a trader.
* `aSimulationParameters::SimulationParameters`: a structure containing all the details of the parameters used to compute the optimal trading plan such as the method to use, the number of samples in the CVaR estimation.
"""
function get_default_output_file_path(;
				      aMarketDetails::MarketDetails,
				      aTraderIndex::Int,
				      aTraders::Array{Trader,1},
				      aStrategies::Array{Strategy,1}=[StrategyModule.get_strategy(aNTradingPeriods=MarketDetailsModule.get_n_trading_periods(aMarketDetails))], # default value if not given
				      aSimulationParameters::SimulationParameters=SimulationParametersModule.get_simulation_parameters(aNTradingPeriods=MarketDetailsModule.get_n_trading_periods(aMarketDetails)), # default value if not given
				      aIncludePartialMarketDetailsHash::Bool=true,
				      aIncludeSimulationParametersHash::Bool=true,
				      aIncludeTradersHash::Bool=true,
				      aIncludeStrategiesHash::Bool=false,
				      aSpecificFileNameExtension::String="",
				      aSpecificFolder::String="outputs/"
				      )

	myDefaultOutputFileName = get_default_output_file_name(
							       aMarketDetails                   = aMarketDetails,
							       aTraderIndex                     = aTraderIndex,
							       aTraders                         = aTraders,
							       aSimulationParameters            = aSimulationParameters,
							       aStrategies                      = aStrategies,
							       aIncludePartialMarketDetailsHash = aIncludePartialMarketDetailsHash,
							       aIncludeSimulationParametersHash = aIncludeSimulationParametersHash,
							       aIncludeTradersHash              = aIncludeTradersHash,
							       aIncludeStrategiesHash           = aIncludeStrategiesHash,
							       aSpecificFileNameExtension       = aSpecificFileNameExtension
							       )

	return string(aSpecificFolder,myDefaultOutputFileName)
end

"""
save_result!(aFilePath,aVariableToStore)

saves in the file `aFilePath` the variable aVariableToStore.

#### Arguments
* `aFilePath`: TODO.
* `aVariableToStore`: TODO.
"""
function save_result!(aFilePath,aVariable)
	save(aFilePath,aVariable)
end

"""
load_function_result(aFilePath)

returns the content of the file `aFilePath`.

#### Argument
* `aFilePath`: TODO.
"""
function load_result(aFilePath)
	return load(aFilePath)
end

"""
load_strategy_from_file(aFilePath::String)

returns the optimal strategy stored in aFilePath.

#### Argument
* `aFilePath::String`: a file path where the optimal strategy is stored.
"""
function load_strategy_from_file(aFilePath::String)

	# tests and loads the file content
	test_file_existence!(aFilePath)
	myDict = load_result(aFilePath)
	myOptimalStrategy = StrategyModule.get_strategy_from_dict(myDict)

	return myOptimalStrategy
end

"""
```
print_logs(
	    aOutputLog::String,
	    aSimulationParameters::SimulationParameters
	    )
```

prints `aOutputLog`, i.e. the output logs of the calling function if these logs must be showed (this is tell in `aSimulationParameters` in the argument `theFunctionsShowPrintLogs`).

### Argument
* `aString`: TODO.
"""
function print_logs(
		    aOutputLog::String,
		    aSimulationParameters::SimulationParameters,
		    aStacktrace::Array{Base.StackTraces.StackFrame,1}
		    )

	# gets the name of the function that is calling this function
	myCallingFunctionName = String(aStacktrace[1].func)

	# deletes the undesired caracters in the string
	myCallingFunctionName = replace(myCallingFunctionName, r"#|1|2|3|4|5|6|7|8|9" => "") 

	# initialises the boolean telling if the output should be printed
	myMustPrint = false

	# if one has a value for this function on wether we should print the output then returns it otherwise returns false
	myFunctionsShowPrintLogs = get_functions_show_print_logs(aSimulationParameters)
	if in(myCallingFunctionName,keys(myFunctionsShowPrintLogs))
		myMustPrint = myFunctionsShowPrintLogs[myCallingFunctionName]
	end
	
	if myMustPrint
		print(aOutputLog)
	end
end

"""
```
println_logs(
	    aOutputLog::String,
	    aSimulationParameters::SimulationParameters
	    )
```

prints `aOutputLog`, i.e. the output logs of the calling function if these logs must be showed (this is tell in `aSimulationParameters` in the argument `theFunctionsShowPrintLogs`).

### Argument
* `aString`: TODO.
"""
function println_logs(
		      aOutputLog::String,
		      aSimulationParameters::SimulationParameters,
		      aStacktrace::Array{Base.StackTraces.StackFrame,1}
		      )

	# gets the name of the function that is calling this function
	myCallingFunctionName = String(aStacktrace[1].func)

	# deletes the undesired caracters in the string
	myCallingFunctionName = replace(myCallingFunctionName, r"#|1|2|3|4|5|6|7|8|9" => "") 

	# initialises the boolean telling if the output should be printed
	myMustPrint = false

	# if one has a value for this function on wether we should print the output then returns it otherwise returns false
	myFunctionsShowPrintLogs = get_functions_show_print_logs(aSimulationParameters)
	if in(myCallingFunctionName,keys(myFunctionsShowPrintLogs))
		myMustPrint = myFunctionsShowPrintLogs[myCallingFunctionName]
	end
	
	if myMustPrint
		println(aOutputLog)
	end
end

function info_logs(
		   aInfoLog::String,
		   aSimulationParameters::SimulationParameters,
		   aStacktrace::Array{Base.StackTraces.StackFrame,1}
		   )

	# gets the name of the function that is calling this function
	myCallingFunctionName = String(aStacktrace[1].func)

	# deletes the undesired caracters in the string
	myCallingFunctionName = replace(myCallingFunctionName, r"#|1|2|3|4|5|6|7|8|9" => "") 

	# initialises the boolean telling if the output should be printed
	myMustNotShowInfoLogs = false

	# if one has a value for this function on wether we should print the output then returns it otherwise returns false
	myFunctionsNotShowInfoLogs = get_functions_not_show_info_logs(aSimulationParameters)
	if in(myCallingFunctionName,keys(myFunctionsNotShowInfoLogs))
		myMustNotShowInfoLogs = myFunctionsNotShowInfoLogs[myCallingFunctionName]
	end

	if !(myMustNotShowInfoLogs)
		@info(aInfoLog)
		sleep(0.1) # time to be sure that the message is displayed in the correct place
	end
end

"""
```
save_figure(aPlot::Plots.Plot,aFilePath,aExtension)
```

TODO function description.

### Arguments
* `aPlot::Plots.Plot`: TODO.
* `aFilePath`: TODO.
* `aExtension`: TODO.
"""
function save_figure!(;
		     aPlot::Plots.Plot,
		     aFilePath::String,
		     aExtension::String = ".eps"
		     )

	myFilePath = replace(aFilePath, "." => "")
	myFilePathWithExtension = string(myFilePath,aExtension)
	HelpFilesModule.create_relevant_folders!(myFilePathWithExtension)
	savefig(aPlot, myFilePathWithExtension)
end


"""
```
get_value_with_n_significant_number()
```

TODO function description.

### Argument
* ``: TODO.
"""
function get_value_with_n_significant_number(
					     aValue;
					     aNSignificantNumbers = 4
					    )

	myValueAbs = abs(aValue)

	# gets the base and exponent
	myValueExponent = Int64(floor(log(10.0,myValueAbs)))
	myValueBase     = myValueAbs/(10.0^myValueExponent)

	# keeps only the significant digits
	myValueBaseSignificant = (round.(myValueBase.*10.0^aNSignificantNumbers))./(10.0^aNSignificantNumbers)

	# keeps only the significant digits (string because round can lead to a lot of numbers)
	myStringValueBaseSignificant = string(sign(aValue)*myValueBaseSignificant)
	myLengthStringValueBaseSignificant = length(myStringValueBaseSignificant)
	if myValueBaseSignificant > 1
		myValueBaseSignificant = myStringValueBaseSignificant[1:min(myLengthStringValueBaseSignificant,aNSignificantNumbers+1)]
	else
		myValueBaseSignificant = myStringValueBaseSignificant[1:min(myLengthStringValueBaseSignificant,aNSignificantNumbers+2)]
	end

	# the value with the requested number of significant number
	myValue = string(myValueBaseSignificant,"e",myValueExponent)

	return myValue
end

end
