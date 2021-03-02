# defines the colours for the plots which corresponds to the colours used in the latex file
ourBluePlot = RGB(0/255, 0/255, 200/255)
ourRedPlot  = RGB(230/255, 0, 0)
ourDarkPurple = RGB(110/255,10/255,110/255)

# returns a number with only `aNSignificantNumbers` significant numbers
function get_number_for_table(aFloat64,aNSignificantNumbers::Int=4)
    return round(aFloat64, sigdigits = aNSignificantNumbers )
end

# returns a table compilable in latex in order to present the results
function show_table_performances(;
				 aArrayPerformances,
				 aAlpha,
				 aListNamesQuantities = ["Expectation","Variance","CVaR_0.1","Mean-Variance","Mean-CVaR_0.1"]
				 )

	# deletes the doublones in `aListNamesQuantities`
	myListNamesQuantities = unique(aListNamesQuantities)

	# gets the number of risk-aversion
	myNRiskAversions = length(aArrayPerformances)

	# gets the number of quantities to insert in the table
	myNQuantities = length(myListNamesQuantities)

	# makes the link between the quantities desired and the names of the columns in the table
	myTableColumNamesDic = Dict(
				    "Expectation"               => "Expectation",
				    string("CVaR_",aAlpha)      => string("CVaR\$_{",aAlpha,"}\$"),
				    string("VaR_",aAlpha)       => string("VaR\$_{",aAlpha,"}\$"),
				    string("Mean-CVaR_",aAlpha) => string("E-CVaR\$_{",aAlpha,"}(λ_p)\$"),
				    string("Mean-VaR_",aAlpha)  => "E-VaR\$(λ_p)\$",
				    "Variance"                  => "Variance",
				    "Mean-Variance"             => "Mean-Variance",
				    "CVaR_0.1"                  => "CVaR\$_{0.1}\$",
				    "CVaR_0.05"                 => "CVaR\$_{0.05}\$",
				    "CVaR_0.025"                => "CVaR\$_{0.025}\$",
				    "CVaR_0.01"                 => "CVaR\$_{0.01}\$",
				    "CVaR_0.005"                => "CVaR\$_{0.005}\$",
				    "CVaR_0.005"                => "CVaR\$_{0.005}\$",
				    )

	# get the arrangement for the latex table
	myTableArrangement = [:l]
	for i in 1:myNQuantities
		push!(myTableArrangement,:c)
	end
	
	# initialises the array that will contain the details of performance of the strategy related to each risk-aversion
	myTableFilling = []

	# fills the column names, first element in the table `myTableFilling`
	myTableColumnNames = ["Risk-aversion"]
	for myColumnName in myListNamesQuantities
		mySplitColumnName = split(myColumnName, "_")
		if size(mySplitColumnName,1) == 1
			push!(myTableColumnNames,myColumnName)
		else
			myNewColumnName = string(mySplitColumnName[1],"\$_{",mySplitColumnName[2],"}\$")
			push!(myTableColumnNames,myNewColumnName)
		end
	end
	push!(myTableFilling,myTableColumnNames)

	# gets for each risk-aversion the performance quantities of the optimal strategy
	for i in 1:myNRiskAversions

		# initialises the array that will contain the performance quantities of the optimal strategy
		myLocalPerformanceDetails = []
		myLocalDicPerformance = aArrayPerformances[i]
		myTrader = myLocalDicPerformance["Trader"]

		# adds risk aversion
		myRiskAversion = myTrader["RiskAversion"]
		if myRiskAversion == 0.0 || ceil(log(10,myRiskAversion)) <= 1 && floor(log(10,myRiskAversion)) >= -1
			push!(
			      myLocalPerformanceDetails,
			      string(myTrader["RiskAversion"])
			      )
		else
			push!(
			      myLocalPerformanceDetails,
			      string(get_number_for_table(myTrader["RiskAversion"],1))
			      )
		end

		for myLocalPerformancDetailName in myListNamesQuantities
			push!(
			      myLocalPerformanceDetails,
			      get_number_for_table(myLocalDicPerformance[myLocalPerformancDetailName])
			      )
		end

		push!(myTableFilling,string.(myLocalPerformanceDetails))
	end

	myTable = Markdown.Table(myTableFilling,myTableArrangement)
	myMarkdownTable = Markdown.MD(myTable)

	return myMarkdownTable
end

# defines the function used to compute and plot the optimal strategies
function compute_and_plot_optimal_strategies(;
					     aRiskAversions,
					     aTrader::Trader,
					     aInitialStrategy::Strategy,
					     aSimulationParameters::SimulationParameters,
					     aFolderToSavePlot="",
					     aFileNameToSavePlot="",
					     aControlYAxis::Bool=false,
					     aYlimits::Array{Float64,1}=[-0.05,1.05],
					     aYticks::Array{Float64,1}=[-1,0.2,1],
					     aDrawPlot::Bool=true,
					     aZoom::Bool=false
					     )

	# gets the market details belief of the trader `aTrader`
	myTraderUncertaintyStructure = get_traders_uncertainty_structure(get_market_details_belief(aTrader))[1]

	# gets the number of trading periods
	myNTradingPeriods = UncertaintyStructureModule.get_n_trading_periods(myTraderUncertaintyStructure)

	# initialises output variables
	myTraders                  = Array{TraderModule.Trader,1}(undef,0)
	myLegendVec                = Array{String,1}(undef,0)
	myOutputFiles              = Array{String,1}(undef,0)
	myOptimalTradingPlans      = Array{Array{Float64,1}}(undef,0)
	myOutputFilesConcatenation = ""

	for myLambda in aRiskAversions
		myTrader = TraderModule.get_new_trader(
						       aTrader,
						       aRiskAversion = myLambda,
						       )

		# computes the optimal strategies
		myOptimalStrategy, myOptimalValueFunction, myOptimisationSuccessful, myOutputFilePath = OptimalStrategyModule.compute_optimal_strategy(
																		       aTraderIndex          = 1,
																		       aTraders              = [myTrader],
																		       aStrategies           = [aInitialStrategy],
																		       aSimulationParameters = aSimulationParameters,
																		       )

		push!(myTraders,myTrader)
		push!(myLegendVec,latexstring("\\lambda = ",myLambda))
		push!(myOutputFiles,myOutputFilePath)
		push!(myOptimalTradingPlans,get_trading_plan(myOptimalStrategy))

		myOutputFilesConcatenation = string(myOutputFilesConcatenation,myOutputFilePath)
	end

	#################################################
	# Saves in a csv file the optimal trading plans #
	#################################################
	
	myHashOutputFile = hash(myOutputFilesConcatenation)
	myCSVTableFileOutputPath = string("outputs/csv_table_trading_plan/",get_method(aSimulationParameters),"_alpha_",get_alpha(aTrader),"_")
	if (get_consider_price_moves(myTraderUncertaintyStructure))
		myCSVTableFileOutputPath = string(myCSVTableFileOutputPath,"P")
	end
	if (get_consider_forecast_updates(myTraderUncertaintyStructure))
		myCSVTableFileOutputPath = string(myCSVTableFileOutputPath,"V")
	end
	myCSVTableFileOutputPath = string(myCSVTableFileOutputPath,"U_hash_",myHashOutputFile,".csv")
	
	myTableString = ""
	for i in 0:myNTradingPeriods
		if i == 0
			myTableString = string(myTableString,"Trading period")
			for aRiskAversion in aRiskAversions
				myTableString = string(myTableString,",\\( \\lambda = ",aRiskAversion," \\)")
			end
			myTableString = string(myTableString,"\n")
		else
			myTableString = string(myTableString,i)
			for j in eachindex(aRiskAversions)
				myTableString = string(myTableString,",",round(myOptimalTradingPlans[j][i], sigdigits=4))
			end
			if i != myNTradingPeriods
				myTableString = string(myTableString,"\n")
			end
		end
	end

	create_relevant_folders!(myCSVTableFileOutputPath)
	io = open(myCSVTableFileOutputPath, "w")
	write(io, myTableString)
	close(io)

	################################
	# Plots the optimal strategies #
	################################
	
	myFileNameToSavePlot = aFileNameToSavePlot

	# get the default file name to save the plot if no file name is provided
	if myFileNameToSavePlot == ""
		myFileNameToSavePlot = string(get_method(aSimulationParameters),"_")
		if (get_consider_price_moves(myTraderUncertaintyStructure))
			myFileNameToSavePlot = string(myFileNameToSavePlot,"P")
		end
		if (get_consider_forecast_updates(myTraderUncertaintyStructure))
			myFileNameToSavePlot = string(myFileNameToSavePlot,"V")
		end
		myFileNameToSavePlot = string(myFileNameToSavePlot,"U_alpha_",get_alpha(aTrader),"_",myHashOutputFile,".eps")
	end
	myFileNameToSavePlot = replace(myFileNameToSavePlot, "." => "-", count=1)

	println(string("File output with the plot: ",myFileNameToSavePlot))

	# generates the plot
	myPlot = nothing
	if aDrawPlot
		if !(aZoom)
			myPlot = PlotOptimalStrategyModule.generate_optimal_strategies_plot(
											    myOutputFiles,
											    aLegendVec    = myLegendVec,
											    aImageFolder  = aFolderToSavePlot,
											    aFileName     = myFileNameToSavePlot,
											    aControlYAxis = aControlYAxis,
											    aYlimits      = aYlimits,
											    aYticks       = aYticks
											    )
		else
			myPlot = PlotOptimalStrategyModule.generate_optimal_strategies_plot_zoom(
												 myOutputFiles,
												 aLegendVec    = myLegendVec,
												 aImageFolder  = aFolderToSavePlot,
												 aFileName     = myFileNameToSavePlot,
												 aControlYAxis = aControlYAxis,
												 aYlimits      = aYlimits,
												 aYticks       = aYticks
												 )
		end
	end

	return myPlot, myTraders, myOutputFiles, myLegendVec, myOptimalTradingPlans
end

# saves the difference of the csv files
function save_difference_csv_file!(;
				   aSimulationParameters,
				   aTrader,
				   aRiskAversions,
				   aOutputFilesTradingPlan1,
				   aOutputFilesTradingPlan2
				   )

	myNTradingPeriods = -1
	myOptimalTradingPlanDifferences = []

	# draws the graphs on the plot
	for myIndex in eachindex(aRiskAversions)

		myLocalRiskAversion = aRiskAversions[myIndex]

		HelpFilesModule.test_file_existence!(aOutputFilesTradingPlan1[myIndex]) # checks if the file to import exist, if not an error is raised.
		HelpFilesModule.test_file_existence!(aOutputFilesTradingPlan2[myIndex]) # checks if the file to import exist, if not an error is raised.

		## Importing the results for the corresonding document
		myDetailsSaved1 = HelpFilesModule.load_result(aOutputFilesTradingPlan1[myIndex])
		myDetailsSaved2 = HelpFilesModule.load_result(aOutputFilesTradingPlan2[myIndex])

		myOptimalTradingPlan1 = myDetailsSaved1["TradingPlan"]
		myOptimalTradingPlan2 = myDetailsSaved2["TradingPlan"]

		myDifference = myOptimalTradingPlan2 - myOptimalTradingPlan1
		push!(myOptimalTradingPlanDifferences,myDifference)

		myNTradingPeriods = size(myDifference,1)
	end

	# saves in a csv file the optimal trading plans
	myHashOutputFile = hash(string(aOutputFilesTradingPlan1,aOutputFilesTradingPlan2))
	myCSVTableFileOutputPath = string("outputs/csv_table_trading_plan/",get_method(aSimulationParameters),"_alpha_",get_alpha(aTrader),"_difference_hash_",myHashOutputFile,".csv")

	myTableString = ""
	for i in 0:myNTradingPeriods
		if i == 0
			myTableString = string(myTableString,"Trading period")
			for aRiskAversion in aRiskAversions
				myTableString = string(myTableString,",\\( \\lambda = ",aRiskAversion," \\)")
			end
			myTableString = string(myTableString,"\n")
		else
			myTableString = string(myTableString,i)
			for j in eachindex(aRiskAversions)
				myTableString = string(myTableString,",",round(myOptimalTradingPlanDifferences[j][i], sigdigits=4))
			end
			if i != myNTradingPeriods
				myTableString = string(myTableString,"\n")
			end
		end
	end

	create_relevant_folders!(myCSVTableFileOutputPath)
	io = open(myCSVTableFileOutputPath, "w")
	write(io, myTableString)
	close(io)
end

"""
```
get_tables_NE(;
	aDictStrategiesNE,
	aDictNEPerformancePlayers,
	aListNamesQuantities = ["Expectation","Variance","CVaR","VaR","Mean-Variance","Mean-CVaR","Mean-VaR"]
	)
```

TODO function description.

### Arguments
* `aDictStrategiesNE`: TODO.
* `aDictNEPerformancePlayers`: TODO.
* `aListNamesQuantities`: TODO.
"""
function get_tables_NE(;
		       aDictNETraders,
		       aDictNEStrategies,
		       aDictNEPerformancePlayers,
		       aListNamesQuantities = ["Expectation","Variance","CVaR","VaR","Mean-Variance","Mean-CVaR","Mean-VaR"],
		       aNSignificantNumbers = 4
		       )

	# gets the names of the different NE
	myNENames = collect(keys(aDictNETraders))

	# gets the number of trading periods
	myNTradingPeriods = size(get_trading_plan(aDictNEStrategies[myNENames[1]][1]),1)

	################################################################
	# A) Table with the details of the players at the different NE #
	################################################################
	
	# initialises the dictionary that will contain the traders details table for each NE
	myDictTableTradersDetails = Dict()

	# the column names of a table with the traders details, i.e. the first element in the array `myTableNETradersDetailsFilling`
	myTableNETradersDetailsColumnNames = ["Player \$p\$", "\$λ_{p}\$", "\$α_{p}\$","Price uncert.", "Volume uncert."]

	# get the arrangement for the latex table with th details of each player
	myTableArrangement = fill(:c,size(myTableNETradersDetailsColumnNames,1))

	# loop on the different NE 
	for myNEName in myNENames

		# gets the traders associated to the NE named `myNEName`
		myNETraders = aDictNETraders[myNEName]

		# initialises the array that will contain the details of performance of the strategy related to each risk-aversion
		myTableNETradersDetailsFilling = []

		# adds the colums names
		push!(myTableNETradersDetailsFilling,myTableNETradersDetailsColumnNames)

		# adds the details of each player at the time
		for p in eachindex(myNETraders)

			# gets the trader structure
			myTrader = myNETraders[p]

			# initialises the array with the details of the player with index `p`
			myDetailsTrader = []

			# adds the trader's index, i.e. `theTraderIndex`
			push!(myDetailsTrader,get_trader_index(myTrader))

			# adds the trader's risk-aversion parameter λp, i.e. `theRiskAversion`
			push!(myDetailsTrader,get_risk_aversion(myTrader))

			# adds the trader's risk-aversion parameter αp, i.e. `theAlpha`
			push!(myDetailsTrader,get_alpha(myTrader))

			# adds the trader's consideration of the price uncertainty
			push!(myDetailsTrader,get_consider_price_moves(get_traders_uncertainty_structure(get_market_details_belief(myTrader))[p]))

			# adds the trader's consideration of the volume uncertainty
			push!(myDetailsTrader,get_consider_forecast_updates(get_traders_uncertainty_structure(get_market_details_belief(myTrader))[p]))

			# add the details of the trader to the table
			push!(myTableNETradersDetailsFilling,myDetailsTrader)
		end

		# gets the table based on the array
		myTable = Markdown.Table(myTableNETradersDetailsFilling,myTableArrangement)

		# gets the markdown table based on the table
		myMarkdownTable = Markdown.MD(myTable)

		myDictTableTradersDetails[myNEName] = myMarkdownTable
	end
	
	####################################################
	# B) Table of the trading plan at the different NE #
	####################################################
	
	# initialises the dictionary that will contain the traders details table for each NE
	myDictTableNETradingPlans = Dict()

	# the column names of a table with the traders details, i.e. the first element in the array `myTableNEStrategiesFilling`
	myTableNETradingPlansColumnNames = ["Player \$p\$"]
	for i in 1:myNTradingPeriods
		push!(myTableNETradingPlansColumnNames,string("\$y_{p,",i,"}\$"))
	end

	# get the arrangement for the latex table with th details of each player
	myTableArrangement = fill(:c,myNTradingPeriods+1)

	# loop on the different NE 
	for myNEName in myNENames

		# gets the strategies associated to the NE named `myNEName`
		myNEStrategies = aDictNEStrategies[myNEName]

		# initialises the array that will contain the details of performance of the strategy related to each risk-aversion
		myTableNETradingPlansFilling = []

		# adds the colums names
		push!(myTableNETradingPlansFilling,myTableNETradingPlansColumnNames)

		# adds the details of each player at the time
		for p in eachindex(myNEStrategies)

			# gets the strategy structure
			myStrategy = myNEStrategies[p]

			# gets the trading plan
			myTradingPlan = get_trading_plan(myStrategy)

			# initialises the array with the details of the player with index `p`
			myDetailsTradingPlan = []

			# adds the trader's index, i.e. `theTraderIndex`
			push!(myDetailsTradingPlan,p)

			# adds the trading proportion for each trading period
			for i in 1:myNTradingPeriods
				push!(myDetailsTradingPlan,get_value_with_n_significant_number(myTradingPlan[i],aNSignificantNumbers = aNSignificantNumbers))
			end

			# add the details of the trader to the table
			push!(myTableNETradingPlansFilling,myDetailsTradingPlan)
		end

		# gets the table based on the array
		myTable = Markdown.Table(myTableNETradingPlansFilling,myTableArrangement)

		# gets the markdown table based on the table
		myMarkdownTable = Markdown.MD(myTable)

		myDictTableNETradingPlans[myNEName] = myMarkdownTable
	end
	
	######################################################
	# C) Table of the optimal values at the different NE #
	######################################################
	
	# link between the quantities desired and the names with this information in a performance dictionary
	myTableColumNamesDic = Dict(
				    "Expectation"   => "Expectation",
				    "Variance"      => "Variance",
				    "CVaR"          => "CVaR",
				    "VaR"           => "VaR",
				    "Mean-Variance" => "E-Variance\$(λ_p)\$",
				    "Mean-CVaR"     => "E-CVaR\$(λ_p)\$",
				    "Mean-VaR"      => "E-VaR\$(λ_p)\$"
				    )


	# initialises the dictionary that will contain the traders details table for each NE
	myDictTableNEValues = Dict()

	# the column names of a table with the traders details, i.e. the first element in the array `myTableNEStrategiesFilling`
	myTableNEValuesColumnNames = ["Player \$p\$"]
	for myValueName in aListNamesQuantities
		push!(myTableNEValuesColumnNames,myTableColumNamesDic[myValueName])
	end

	# get the arrangement for the latex table with th details of each player
	myTableArrangement = fill(:c,size(aListNamesQuantities,1)+1)

	# loop on the different NE 
	for myNEName in myNENames

		# gets the performance of all players associated to the NE named `myNEName`
		myNEPerformances = aDictNEPerformancePlayers[myNEName]

		# initialises the array that will contain the details of performance of the strategy related to each risk-aversion
		myTableNEValuesFilling = []

		# adds the colums names
		push!(myTableNEValuesFilling,myTableNEValuesColumnNames)

		# adds the details of each player at the time
		for p in eachindex(myNEPerformances)

			# gets the performance dic
			myNEPerformance = myNEPerformances[p]

			# initialises the array with the details of the player with index `p`
			myDetailsValues = []

			# adds the trader's index, i.e. `theTraderIndex`
			push!(myDetailsValues,p)

			# adds the trading proportion for each trading period
			for myValueName in aListNamesQuantities
				push!(myDetailsValues,get_value_with_n_significant_number(myNEPerformance[myValueName],aNSignificantNumbers = aNSignificantNumbers))
			end

			# add the details of the values of the trader to the table
			push!(myTableNEValuesFilling,myDetailsValues)
		end

		# gets the table based on the array
		myTable = Markdown.Table(myTableNEValuesFilling,myTableArrangement)

		# gets the markdown table based on the table
		myMarkdownTable = Markdown.MD(myTable)

		myDictTableNEValues[myNEName] = myMarkdownTable
	end
	

	return myDictTableTradersDetails, myDictTableNETradingPlans, myDictTableNEValues

end

