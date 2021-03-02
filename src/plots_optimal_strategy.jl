# NAME: plots_optimal_strategy.jl
# AUTHOR: Julien Vaes
# DATE: July 20, 2018
# DESCRIPTION: set of functions to generate plots to illustrate the optimal strategies.

module PlotOptimalStrategyModule

###################
## Load Packages ##
###################

using Plots
using LaTeXStrings
using Colors
using StatsBase
using Measures
using DelimitedFiles

############################
## Load Personal Packages ##
############################

using ..HelpFilesModule

######################
## Export functions ##
######################

export get_colour_list, get_marker_list, get_line_style_list
export get_line_details
export generate_optimal_strategies_plot
export generate_difference_optimal_strategies_plot
export plot_comparison_pdf_methods!
export plot_comparison_cdf_methods!
export plot_comparison_pdf_and_cdf_empircal_distributions!
export plot_analysis_liquidity_profile

#######################
## Module parameters ##
#######################

# defines the colours for the plots which corresponds to the colours used in the latex file
ourBluePlot   = RGB(0/255, 0/255, 200/255)
ourRedPlot    = RGB(230/255, 0/255, 0/255)
ourGreenPlot  = RGB(34/255,139/255,34/255)
ourPurplePlot = RGB(140/255,30/255,150/255)
OurBrownPlot  = RGB(160/255,82/255,45/255)
ourCyanPlot   = RGB(0/255,180/255,255/255)

# generates distinguishable colors
ourColours = [ourBluePlot, ourRedPlot, ourGreenPlot, OurBrownPlot, ourCyanPlot, ourPurplePlot]

# generates distinguishable line styles for the plot lines
ourLineStyles  = [:solid, :dot, :dash, :dashdot]
ourMarkers     = [:circle, :diamond, :utriangle, :rect, :xcross, :dtriangle, :star4]
ourMarkerSizes = [6,       6,        8,          6,     8,       8,          8]

ourMethodsPDFEstimation = ["Plots.density","Plots.histogram","StatsBase.fit"]

######################
## Module functions ##
######################

"""
```
get_colour_list()
```

returns a list of the different distinguishable colours in order to distinguish easily different lines on the same plot.

"""
function get_colour_list()
	return ourColours
end

"""
```
get_marker_list()
```

returns a list of the different markers in order to distinguish easily different lines on the same plot.

"""
function get_marker_list()
	return ourMarkers, ourMarkerSizes 
end

"""
```
get_line_style_list()
```

returns a list of the different line styles in order to distinguish easily different lines on the same plot.

"""
function get_line_style_list()
	return ourLineStyles
end

"""
#### Definition
```
get_line_details(aLineIndex)
```

TODO function description.

#### Argument
* `aLineIndex`: TODO.
"""
function get_line_details(aLineIndex)

	# shifts index such that if aLineIndex == 1, then the first elements of the arrays are taken
	myIndex = aLineIndex - 1

	# gets the colour
	myColourIndex = (myIndex % length(ourColours)) + 1
	myColour = ourColours[myColourIndex]

	# gets the line style
	myLineStyleIndex = (myIndex % length(ourLineStyles)) + 1
	myLineStyle = ourLineStyles[myLineStyleIndex]

	# gets the line markers
	myMarkerIndex = (myIndex % length(ourMarkers)) + 1
	myMarker = ourMarkers[myMarkerIndex]

	# gets the markers size
	myMarkerSizeIndex = (myIndex % length(ourMarkerSizes)) + 1
	myMarkerSize = ourMarkerSizes[myMarkerIndex]

	return myColour, myLineStyle, myMarker, myMarkerSize
end

"""
get_next_unnamed_file(aImageFolder::String)

returns the next available name of the form "unnamed_image_XX".

#### Argument
* `aImageFolder::String`: a string with the image folder in which the image must be saved.
"""
function get_next_unnamed_file(aImageFolder::String)
	myListOfFiles = readdir(aImageFolder)
	myNUnnamedFiles = 0
	for myFile in myListOfFiles
		if contains(myFile,"unnamed_image_")
			myNUnnamedFiles += 1
		end
	end
	return string("unnamed_image_",myNUnnamedFiles,".eps")
end

"""
generate_optimal_strategies_plot(
aOutputFilesPathesVec::Array{String};aTitle::String="",
aLegendVec::Array{String}=fill("",length(aOutputFilesPathesVec)),
aImageFolder::String="images/",
aFileName::String=get_next_unnamed_file(aImageFolder),
aControlYAxis::Bool=false
)
saves an image that illustrates.

#### Arguments
* `aMarketDetails::MarketDetails`: a structure containing all the details of the trading market.
* `aRiskAversions::Array{Float64}`: an array of risk-aversions we desire to plot on the same plot.
* `aTitle::String`: the title to put on the plot.
* `aFileName::String`: the name to use to save the image.
* `aImageFolder::String="images/"`: the folder where to save the images
* `aOutputFolder::String="outputs/"`: the folder to save the outputs
"""
function generate_optimal_strategies_plot(
					 aOutputFilesPathesVec::Array{String};
					 aTitle::String             = "",
					 aLegendVec::Array{String}  = fill("",length(aOutputFilesPathesVec)),
					 aImageFolder::String       = "images/",
					 aFileName::String          = get_next_unnamed_file(aImageFolder),
					 aControlYAxis::Bool        = false,
					 aYlimits::Array{Float64,1} = [-0.05,1.05],
					 aYticks::Array{Float64,1}  = [0,0.2,1],
					 aLegendPosition            = :topright,
					 )

	# checks that the details are sufficient to plot a graph
	myNElements = length(aOutputFilesPathesVec)
	if myNElements<1
		error(string(
			     "\nPlotOptimalStrategyModule_ERROR 101:\n",
			     "At least one result must be given to be plotted."
			     )
		      )
	elseif myNElements != length(aLegendVec)
		error(string(
			     "\nPlotOptimalStrategyModule_ERROR 102:\n",
			     "Some information is missing, the number of elements in the legend should have the same size as the number of output files given."
			     )
		      )
	end

	# Initialisation of the plot
	myTradingPlanPlot = plot(
				 formatter = :latex,
				 size      = (900,500),
				 title = aTitle, 
				 title_location = :left, 
				 xtickfont = font(8),
				 ytickfont = font(8),
				 legendfont = font(12),
				 xaxis = ("Trading period", font(14)),
				 yaxis = ("Proportion of the forecasted demand to trade", font(13)),
				 )

	if aControlYAxis
		ylims!(myTradingPlanPlot,(aYlimits[1],aYlimits[2]))
		yticks!(myTradingPlanPlot,aYticks[1]:aYticks[2]:aYticks[3])
	end

	# Draws the graphs on the plot
	for myIndex in 1:myNElements
		# checks if the file to import exists, if not an error is raised.
		HelpFilesModule.test_file_existence!(aOutputFilesPathesVec[myIndex])

		## imports the results for the corresponding document
		myDetailsSaved = HelpFilesModule.load_result(aOutputFilesPathesVec[myIndex])
		myOptimalTradingPlan = myDetailsSaved["TradingPlan"]

		# saves the optimal trading plan in a txt file
		myFileNameTxt = split(split(aOutputFilesPathesVec[myIndex],"/")[end],".")[1]
		myFileNameTxt = string("outputs/txt_files/",myFileNameTxt,".txt")
		create_relevant_folders!(myFileNameTxt)
		open(myFileNameTxt, "w") do io
			writedlm(io, ["T" "Y"])
			writedlm(io, hcat(collect(1:size(myOptimalTradingPlan,1)),myOptimalTradingPlan))
		end

		myLegend = aLegendVec[myIndex]

		# get the line details
		myColour, myLineStyle, myMarker, myMarkerSize = get_line_details(myIndex)

		## Adding the curves on the plot
		plot!(
		      myTradingPlanPlot,
		      myOptimalTradingPlan,
		      label       = myLegend,
		      color       = myColour,
		      linestyle   = myLineStyle,
		      linewidth   = 3,
		      markershape = myMarker,
		      markersize  = myMarkerSize,
		      legend      = aLegendPosition
		      )
	end

	myImageFilePath = string(aImageFolder,aFileName)
	HelpFilesModule.create_relevant_folders!(myImageFilePath)
	Plots.savefig(myTradingPlanPlot,myImageFilePath)

	return myImageFilePath
end

function generate_difference_optimal_strategies_plot(
						     aOutputFilesPathesVec1::Array{String},
						     aOutputFilesPathesVec2::Array{String};
						     aAlpha                     = -1.0,
						     aTitle::String             = "",
						     aLegendVec::Array{String}  = fill("",length(aOutputFilesPathesVec)),
						     aImageFolder::String       = "images/",
						     aFileName::String          = "",
						     aControlYAxis::Bool        = false,
						     aYlimits::Array{Float64,1} = [-0.05,1.05],
						     aYticks::Array{Float64,1}  = [0,0.2,1],
						     aLegendPosition            = :topright,
						     )

	# Checks that the details are sufficient to plot a graph
	myNElements = length(aOutputFilesPathesVec1)
	if myNElements<1
		error(string(
			     "\nPlotOptimalStrategyModule_ERROR 103:\n",
			     "At least one result must be given to be plotted."
			     )
		      )
	elseif myNElements != length(aLegendVec)
		error(string(
			     "\nPlotOptimalStrategyModule_ERROR 104:\n",
			     "Some information is missing, the number of elements in the legend should have the same size as the number of output files given."
			     )
		      )
	elseif myNElements != length(aOutputFilesPathesVec2)
		error(string(
			     "\nPlotOptimalStrategyModule_ERROR 105:\n",
			     " The number of elements in `aOutputFilesPathesVec1` and `aOutputFilesPathesVec2` must correspond."
			     )
		      )
	end

	# Initialisation of the plot
	myTradingPlanPlot = plot(
				 formatter      = :latex,
				 size           = (900,500),
				 title          = aTitle, 
				 title_location = :left, 
				 xtickfont      = font(8),
				 ytickfont      = font(8),
				 legendfont     = font(12),
				 xaxis          = ("Trading period", font(14)),
				 yaxis          = ("Proportion of the forecasted demand to trade", font(13)),
				 legend         = aLegendPosition
				 )

	if aControlYAxis
		ylims!(myTradingPlanPlot,(aYlimits[1],aYlimits[2]))
		yticks!(myTradingPlanPlot,aYticks[1]:aYticks[2]:aYticks[3])
	end

	# Draws the graphs on the plot
	for myIndex in 1:myNElements
		HelpFilesModule.test_file_existence!(aOutputFilesPathesVec1[myIndex]) # checks if the file to import exist, if not an error is raised.
		HelpFilesModule.test_file_existence!(aOutputFilesPathesVec2[myIndex]) # checks if the file to import exist, if not an error is raised.

		## Importing the results for the corresonding document
		myDetailsSaved1 = HelpFilesModule.load_result(aOutputFilesPathesVec1[myIndex])
		myDetailsSaved2 = HelpFilesModule.load_result(aOutputFilesPathesVec2[myIndex])
		myOptimalTradingPlan1 = myDetailsSaved1["TradingPlan"]
		myOptimalTradingPlan2 = myDetailsSaved2["TradingPlan"]
		myLegend = aLegendVec[myIndex]
		myDifference = myOptimalTradingPlan2 - myOptimalTradingPlan1

		# saves the difference of the optimal trading plans in a txt file
		myFileNameTxt = string(
				       "differences_",
				       split(split(aOutputFilesPathesVec1[myIndex],"/")[end],".")[1],
				       "_",
				       split(split(aOutputFilesPathesVec2[myIndex],"/")[end],".")[1]
				       )
		myFileNameTxt = string("outputs/txt_files/",myFileNameTxt,".txt")
		create_relevant_folders!(myFileNameTxt)
		open(myFileNameTxt, "w") do io
			writedlm(io, ["T" "Y"])
			writedlm(io, hcat(collect(1:size(myDifference,1)),myDifference))
		end

		# gets the line details
		myColour, myLineStyle, myMarker, myMarkerSize = get_line_details(myIndex)

		# adds the graph to the plot
		plot!(
		      myTradingPlanPlot,
		      myDifference,
		      label       = myLegend,
		      color       = myColour,
		      linestyle   = myLineStyle,
		      linewidth   = 3,
		      markershape = myMarker,
		      markersize  = myMarkerSize,
		      legend      = aLegendPosition
		      )
	end

	################################
	# Plots the optimal strategies #
	################################

	myFileName = aFileName

	# get the default file name to save the plot if no file name is provided
	if myFileName == ""
		myFileName = "Difference_"
		myFileName = string(myFileName,"alpha_",aAlpha,"_",hash(string(aOutputFilesPathesVec1,aOutputFilesPathesVec2)),".eps")
	end
	myFileName = replace(myFileName, "." => "-", count=1)

	println(string("File output with the plot: ",myFileName))

	myImageFilePath = string(aImageFolder,myFileName)
	HelpFilesModule.create_relevant_folders!(myImageFilePath)
	Plots.savefig(myTradingPlanPlot,myImageFilePath)

	return myImageFilePath
end

"""
#### Definition
```
get_pdf_estimation(aRealisations)
```

TODO function description.

#### Argument
* `aRealisations`: TODO.
"""
function plot_pdf_estimation!(;
			      aRealisations::Array{Float64,1},
			      aPlot,
			      aSubPlotIndex   = 1,
			      aLabel          = nothing,
			      aColour         = get_colour_list()[1],
			      aLineStyle      = :solid,
			      aLineWidth      = 2,
			      aMethod         = "StatsBase.fit",
			      aLegendPosition = :topright,
			      aFileName = string("outputs/txt_files/pdf_",hash(aRealisations),".txt")
			      )


	if !(in(aMethod,ourMethodsPDFEstimation))
		@error("")
	end

	# plot the PDF estimates with Plots.density: make a line plot of a kernel density estimate.
	if aMethod == "Plots.density"
		Plots.density!(
			       aPlot[aSubPlotIndex],
			       aRealisations,
			       label     = aLabel,
			       color     = aColour,
			       linestyle = aLineStyle,
			       linewidth = aLineWidth,
			       legend    = aLegendPosition
			       )  
	elseif aMethod == "Plots.histogram"
		Plots.histogram!(
				 aPlot[aSubPlotIndex],
				 aRealisations,
				 label     = aLabel,
				 color     = aColour,
				 linestyle = aLineStyle,
				 linewidth = aLineWidth,
				 bins      = :rice,
				 fillalpha = 0.3, # turns the bars transparent
				 normalize = true,
				 legend    = aLegendPosition
				 )  
	elseif aMethod == "StatsBase.fit"

		myNBins = 10^3
		myHistogramFit = @views StatsBase.fit(
						      Histogram,
						      aRealisations,
						      nbins=myNBins
						      )

		myArea = sum((myHistogramFit.edges[1][2:end]-myHistogramFit.edges[1][1:end-1]).*myHistogramFit.weights)
		myXHistogramFit = (myHistogramFit.edges[1][1:end-1]+myHistogramFit.edges[1][2:end])/2 # computes the middle of the edges
		myYHistogramFit = myHistogramFit.weights./myArea

		# saves in a files the pdf estimated
		create_relevant_folders!(aFileName)
		open(aFileName, "w") do io
			writedlm(io, ["x" "y"])
			writedlm(io, hcat(myXHistogramFit,myYHistogramFit))
		end

		Plots.plot!(
			    aPlot[aSubPlotIndex],
			    myXHistogramFit,
			    myYHistogramFit,
			    label     = aLabel,
			    color     = aColour,
			    linestyle = aLineStyle,
			    linewidth = aLineWidth,
			    bins      = :scott,
			    normalize = true,
			    legend    = aLegendPosition,
			    )  
	end
end

function plot_cdf_estimation!(;
			      aRealisations::Array{Float64,1},
			      aPlot,
			      aSubPlotIndex   = 1,
			      aLabel          = nothing,
			      aColour         = get_colour_list()[1],
			      aLineStyle      = :solid,
			      aLineWidth      = 2,
			      aLegendPosition = :topright,
			      aFileName = string("outputs/txt_files/cdf_",hash(aRealisations),".txt")
			      )

	myNBins = 10^3
	myHistogramFit = @views StatsBase.fit(
					      Histogram,
					      aRealisations,
					      nbins=myNBins
					      )

	myEdges = myHistogramFit.edges[1][2:end]-myHistogramFit.edges[1][1:end-1]
	myArea = sum((myHistogramFit.edges[1][2:end]-myHistogramFit.edges[1][1:end-1]).*myHistogramFit.weights)
	myXHistogramFit = (myHistogramFit.edges[1][1:end-1] + myHistogramFit.edges[1][2:end])/2 # computes the middle of the edges
	myYHistogramFit = myHistogramFit.weights./myArea

	myYCumulHistogramFit = copy(myYHistogramFit)
	mySum = 0.0
	for i = 1:length(myXHistogramFit)
		mySum = mySum + myYHistogramFit[i]*myEdges[i]
		myYCumulHistogramFit[i] = mySum
	end

	# saves in a files the cdf estimated
	create_relevant_folders!(aFileName)
	open(aFileName, "w") do io
		writedlm(io, ["x" "y"])
		writedlm(io, hcat(myXHistogramFit,myYCumulHistogramFit))
	end

	Plots.plot!(
		    aPlot[aSubPlotIndex],
		    myXHistogramFit,
		    myYCumulHistogramFit,
		    label     = aLabel,
		    color     = aColour,
		    linestyle = aLineStyle,
		    linewidth = aLineWidth,
		    bins      = :scott,
		    normalize = true,
		    legend    = aLegendPosition,
		    )  
end

function plot_comparison_pdf_or_cdf_methods!(;
					     aRiskAversions,
					     aRealisations,
					     aFunctionToPlotPDForCDF,
					     aXLowerLim,
					     aXUpperLim,
					     aMethodNameLabels::Array = [nothing],
					     aLegendPosition          = :topright,
					     aFileNameToSaveFigure    = nothing,
					     )

	myNRiskAversions = length(aRiskAversions)
	myNMethods = length(aRealisations[1])

	# initialises the plot
	pyplot()
	myPlot = Plots.plot(
			    formatter  = :latex,
			    layout     = myNRiskAversions,
			    margin     = 3mm,
			    xaxis      = ("Trading cost", font(10)),
			    xticks     = range(0, stop=10^6*ceil(aXUpperLim/10^6), step=10.0^floor(log(10,aXUpperLim-aXLowerLim))),
			    xlims      = (aXLowerLim, aXUpperLim),
			    titlefont  = font(16),
			    xtickfont  = font(8),
			    ytickfont  = font(8),
			    legendfont = font(10),
			    legend     = aLegendPosition,
			    size       = (1600,800)
			    )

	for i in 1:myNRiskAversions
		# plots the CDFs of the different methods
		for j in 1:myNMethods
			aFunctionToPlotPDForCDF(
						aLocalRealisations   = aRealisations[i][j],
						aLocalPlot           = myPlot,
						aLocalSubPlotIndex   = i,
						aLocalColour         = ourColours[j],
						aLocalLabel          = aMethodNameLabels[j],
						aLocalLegendPosition = aLegendPosition
						)
		end

		# sets the title
		myRiskAversion = aRiskAversions[i]
		title!(myPlot[i],string("Risk-aversion: \$\\lambda = {$myRiskAversion} \$ "),fontsize=:6)

		# sets the legend box
		if all(y -> isnothing(y), aMethodNameLabels)
			Plots.plot!(myPlot[1],legend=false);
		end
	end

	# checks if a file name is provided to save the figure
	if !(isnothing(aFileNameToSaveFigure))

		# creates the output folder
		HelpFilesModule.create_relevant_folders!(aFileNameToSaveFigure)

		# saves the figure
		Plots.savefig(myPlot,aFileNameToSaveFigure)
	end
end

function plot_comparison_pdf_methods!(;
				      aRiskAversions,
				      aRealisations,
				      aMethodNameLabels::Array = [nothing],
				      aPDFMethod               = "StatsBase.fit",
				      aFileNameToSaveFigure    = nothing,
				      aLegendPosition          = :topright,
				      aXLowerLim               = [minimum(minimum(aRealisations[i][j]) for j in 1:length(aRealisations[i])) for i in 1:length(aRealisations)][1],
				      aXUpperLim               = [maximum(maximum(aRealisations[i][j]) for j in 1:length(aRealisations[i])) for i in 1:length(aRealisations)][1]
				      )

	myFunctionToPlotPDF(;
			    aLocalRealisations,
			    aLocalPlot,
			    aLocalSubPlotIndex,
			    aLocalColour,
			    aLocalLabel,
			    aLocalLegendPosition,
			    ) = plot_pdf_estimation!(
						     aRealisations   = aLocalRealisations,
						     aPlot           = aLocalPlot,
						     aSubPlotIndex   = aLocalSubPlotIndex,
						     aColour         = aLocalColour,
						     aLabel          = aLocalLabel,
						     aMethod         = aPDFMethod,
						     aLegendPosition = aLocalLegendPosition,
						     )

	plot_comparison_pdf_or_cdf_methods!(
					    aRiskAversions          = aRiskAversions,
					    aRealisations           = aRealisations,
					    aMethodNameLabels       = aMethodNameLabels,
					    aFunctionToPlotPDForCDF = myFunctionToPlotPDF,
					    aFileNameToSaveFigure   = aFileNameToSaveFigure,
					    aXLowerLim              = aXLowerLim,
					    aXUpperLim              = aXUpperLim,
					    aLegendPosition         = aLegendPosition,
					    )
end

function plot_comparison_cdf_methods!(;
				      aRiskAversions,
				      aRealisations,
				      aMethodNameLabels::Array=[nothing],
				      aFileNameToSaveFigure=nothing,
				      aXLowerLim=[minimum(minimum(aRealisations[i][j]) for j in 1:length(aRealisations[i])) for i in 1:length(aRealisations)][1],
				      aXUpperLim=[maximum(maximum(aRealisations[i][j]) for j in 1:length(aRealisations[i])) for i in 1:length(aRealisations)][1],
				      aLegendPosition = :bottomright,
				      )

	myFunctionToPlotCDF(;
			    aLocalRealisations,
			    aLocalPlot,
			    aLocalSubPlotIndex,
			    aLocalColour,
			    aLocalLabel,
			    aLocalLegendPosition
			    ) = plot_cdf_estimation!(
						     aRealisations   = aLocalRealisations,
						     aPlot           = aLocalPlot,
						     aSubPlotIndex   = aLocalSubPlotIndex,
						     aColour         = aLocalColour,
						     aLabel          = aLocalLabel,
						     aLegendPosition = aLocalLegendPosition,
						     )


	plot_comparison_pdf_or_cdf_methods!(
					    aRiskAversions          = aRiskAversions,
					    aRealisations           = aRealisations,
					    aMethodNameLabels       = aMethodNameLabels,
					    aFunctionToPlotPDForCDF = myFunctionToPlotCDF,
					    aLegendPosition         = aLegendPosition,
					    aFileNameToSaveFigure   = aFileNameToSaveFigure,
					    aXLowerLim              = aXLowerLim,
					    aXUpperLim              = aXUpperLim
					    )
end

"""
#### Definition
```
plot_comparison_pdf_and_cdf_empircal_distributions()
```

TODO function description.

#### Argument
* ``: TODO.
"""
function plot_comparison_pdf_and_cdf_empircal_distributions!(;
							     aRealisations,
							     aLabels               = [],
							     aFileNameToSaveFigure = nothing,
							     aXLowerLim            = minimum([minimum(aRealisations[i]) for i in eachindex(aRealisations)]),
							     aXUpperLim            = maximum([maximum(aRealisations[i]) for i in eachindex(aRealisations)]),
							     aLegendPosition       = :bottomright,
							     )

	# initialises the plot
	pyplot()
	myPlot = Plots.plot(
			    formatter  = :latex,
			    layout     = (2,1),
			    xaxis      = ("Trading cost", font(10)),
			    xticks     = range(0, stop=10^6*ceil(aXUpperLim/10^6), step=10.0^floor(log(10,aXUpperLim-aXLowerLim))),
			    xlims      = (aXLowerLim, aXUpperLim),
			    margin     = 3mm,
			    titlefont  = font(16),
	xtickfont  = font(8),
	ytickfont  = font(8),
	legendfont = font(10),
	legend     = aLegendPosition,
	size       = (800,600)
       )

	# PDF
	for i in eachindex(aRealisations)
		myColour, myLineStyle, myMarker, myMarkerSize = get_line_details(i)
		if aLabels == []
		plot_pdf_estimation!(
				     aRealisations   = aRealisations[i],
				     aPlot           = myPlot,
				     aSubPlotIndex   = 1,
				     aColour         = myColour,
				     aLegendPosition = aLegendPosition
				     )
		else
		plot_pdf_estimation!(
				     aRealisations   = aRealisations[i],
				     aPlot           = myPlot,
				     aSubPlotIndex   = 1,
				     aColour         = myColour,
				     aLabel          = aLabels[i],
				     aLegendPosition = aLegendPosition
				     )
		end
	end

	# CDF
	for i in eachindex(aRealisations)
		myColour, myLineStyle, myMarker, myMarkerSize = get_line_details(i)
		plot_cdf_estimation!(
				     aRealisations   = aRealisations[i],
				     aPlot           = myPlot,
				     aSubPlotIndex   = 2,
				     aColour         = myColour,
				     aLegendPosition = aLegendPosition
				     )
	end

	# Legend box

	## deletes the legend box from the PDF plots
	Plots.plot!(myPlot[2],legend=false);

	## if no labels are provided, the legend box of the CDF plots is not included
	if aLabels == []
		Plots.plot!(myPlot[1],legend=false);
	end

	# checks if a file name is provided to save the figure
	if !(isnothing(aFileNameToSaveFigure))

		# creates the output folder
		HelpFilesModule.create_relevant_folders!(aFileNameToSaveFigure)

		# saves the figure
		Plots.savefig(myPlot,aFileNameToSaveFigure)
	end
end

"""
```
plot_analysis_liquidity_profile()
```

plot the impact of the liquidity profile and the optimal trading strategy and the impact of how to consider volume uncertainty.

### Argument
* ``: TODO.
"""
function plot_analysis_liquidity_profile(;
					 aRiskAversions,
					 aBidAskSpreadProfiles,
					 aOutputFilesPU,
					 aOutputFilesPVU,
					 aFileNameToSaveFigures::String,
					 aFolderToSaveFigures::String,
					 aLegendPosition                     = :outertop,
					 aControlYAxis::Array{Bool,1}        = [true,true,true],
					 aYlimits::Array{Array{Float64,1},1} = [ [0.1,0.9]    , [-0.05,1.05] , [-0.1,0.1] ],
					 aYticks::Array{Array{Float64,1},1}  = [ [0.0,0.2,1]  , [0,0.2,1]    , [-0.14,0.02,0.14] ]
					 )

	myNRiskAversions  = length(aRiskAversions)
	myNBidAskProfiles = length(aBidAskSpreadProfiles)

	# Colour of the plot and line style
	myColour    = get_colour_list()[1]
	myLineStyle = get_line_style_list()[1]
	myLineWidth = 3

	myMarkers, myMarkersSize = get_marker_list()
	myMarker     = myMarkers[1]
	myMarkerSize = myMarkersSize[1]

	# initialises the vector that will contain the images
	myFigures = []

	for i in eachindex(aRiskAversions)

		# initialises the plot
		pyplot()
		myPlot = Plots.plot(
				    formatter  = :latex,
				    layout     = (myNBidAskProfiles,3),
				    margin     = 3mm,
				    legend     = aLegendPosition,
				    size       = (1600,800)
				    )

		# fills the impact of the liquidity profile impact
		for j in eachindex(aBidAskSpreadProfiles)

			# plots the bid-ask spread profile

			if aControlYAxis[1]
				ylims!(myPlot[j,1],(aYlimits[1][1],aYlimits[1][2]))
				yticks!(myPlot[j,1],aYticks[1][1]:aYticks[1][2]:aYticks[1][3])
			end

			plot!(
			      myPlot[j,1],
			      aBidAskSpreadProfiles[j],
			      color       = myColour,
			      linestyle   = myLineStyle,
			      linewidth   = myLineWidth,
			      markershape = myMarker,
			      markersize  = myMarkerSize,
			      xaxis       = ("Trading period", font(10)),
			      yaxis       = ("Bid-ask spread [\$]", font(10)),
			      label       = "Bid-ask spread profile"
			      )  

			# removes the legend on the plot except for the first row
			if j > 1
				plot!(
				      myPlot[j,1],
				      legend = false
				      )  
			end

			# gets the optimal strategies when 
			# - considering price unverntainty (PU) 
			# - considering price and volume unverntainty (PVU) 

			# checks if the files to import exist, if not an error is raised.
			HelpFilesModule.test_file_existence!(aOutputFilesPU[j][i])
			HelpFilesModule.test_file_existence!(aOutputFilesPVU[j][i])

			# imports the results for the corresponding document
			myTradingPlanPU  = HelpFilesModule.load_result(aOutputFilesPU[j][i])["TradingPlan"]
			myTradingPlanPVU = HelpFilesModule.load_result(aOutputFilesPVU[j][i])["TradingPlan"]
			myDifference     = myTradingPlanPVU - myTradingPlanPU

			# saves the optimal trading plan PU in a txt file
			myFileNameTxtPU = split(split(aOutputFilesPU[j][i],"/")[end],".")[1]
			myFileNameTxtPU = string("outputs/txt_files/",myFileNameTxtPU,".txt")
			create_relevant_folders!(myFileNameTxtPU)
			open(myFileNameTxtPU, "w") do io
				writedlm(io, ["T" "Y"])
				writedlm(io, hcat(collect(1:size(myTradingPlanPU,1)),myTradingPlanPU))
			end

			# saves the optimal trading plan PVU in a txt file
			myFileNameTxtPVU = split(split(aOutputFilesPVU[j][i],"/")[end],".")[1]
			myFileNameTxtPVU = string("outputs/txt_files/",myFileNameTxtPVU,".txt")
			create_relevant_folders!(myFileNameTxtPVU)
			open(myFileNameTxtPVU, "w") do io
				writedlm(io, ["T" "Y"])
				writedlm(io, hcat(collect(1:size(myTradingPlanPVU,1)),myTradingPlanPVU))
			end

			# saves the difference of the optimal trading plans in a txt file
			myFileNameTxtDiff = string(
					       "differences_",
					       split(split(aOutputFilesPU[j][i],"/")[end],".")[1],
					       "_",
					       split(split(aOutputFilesPVU[j][i],"/")[end],".")[1]
					       )
			myFileNameTxtDiff = string("outputs/txt_files/",myFileNameTxtDiff,".txt")
			create_relevant_folders!(myFileNameTxtDiff)
			open(myFileNameTxtDiff, "w") do io
				writedlm(io, ["T" "Y"])
				writedlm(io, hcat(collect(1:size(myDifference,1)),myDifference))
			end

			# plots the optimal strategy under price and volume uncertainty (PVU)
			if aControlYAxis[2]
				ylims!(myPlot[j,2],(aYlimits[2][1],aYlimits[2][2]))
				yticks!(myPlot[j,2],aYticks[2][1]:aYticks[2][2]:aYticks[2][3])
			end

			plot!(
			      myPlot[j,2],
			      myTradingPlanPVU,
			      color       = myColour,
			      linestyle   = myLineStyle,
			      linewidth   = myLineWidth,
			      markershape = myMarker,
			      markersize  = myMarkerSize,
			      xaxis       = ("Trading period", font(10)),
			      yaxis       = ("Proportion [1]", font(10)),
			      label       = "Trading plan under price and volume uncertainty"
			      )  

			# removes the legend on the plot except for the first row
			if j > 1
				plot!(
				      myPlot[j,2],
				      legend = false
				      )  
			end


			# plots the difference between the optimal trading plan PVU and PU
			if aControlYAxis[3]
				ylims!(myPlot[j,3],(aYlimits[3][1],aYlimits[3][2]))
				yticks!(myPlot[j,3],aYticks[3][1]:aYticks[3][2]:aYticks[3][3])
			end

			plot!(
			      myPlot[j,3],
			      myTradingPlanPVU - myTradingPlanPU,
			      color       = myColour,
			      linestyle   = myLineStyle,
			      linewidth   = myLineWidth,
			      markershape = myMarker,
			      markersize  = myMarkerSize,
			      xaxis     = ("Trading period", font(10)),
			      yaxis     = ("Proportion [1]", font(10)),
			      label     = "Volume uncertainty impact"
			      )  

			# removes the legend on the plot except for the first row
			if j > 1
				plot!(
				      myPlot[j,3],
				      legend = false
				      )  
			end
		end

		# get the file path
		myOutputFilesPURelatedToRiskAversion  = [aOutputFilesPU[j][i]  for j = 1:size(aOutputFilesPU,1)]
		myOutputFilesPVURelatedToRiskAversion = [aOutputFilesPVU[j][i] for j = 1:size(aOutputFilesPVU,1)]
		myHash     = hash(string(hash(aRiskAversions[i]),hash(aBidAskSpreadProfiles),hash(myOutputFilesPURelatedToRiskAversion),hash(myOutputFilesPVURelatedToRiskAversion)))
		myFilePath = string(aFolderToSaveFigures,aFileNameToSaveFigures,"_",myHash,".eps")

		println(string("File output with the plot: ",myFilePath))

		Plots.savefig(myPlot,myFilePath)

		push!(myFigures,myPlot)
	end

	return myFigures
end

end
