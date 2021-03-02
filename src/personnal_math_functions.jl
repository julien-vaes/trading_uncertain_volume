# NAME: personnal_math_functions.jl
# AUTHOR: Julien Vaes
# DATE: November 25, 2019
# DESCRIPTION: general math functions

module PersonnalMathFunctionsModule

export get_columns_two_norm, normalising_columns

"""
get_columns_two_norm(aMatrix::Array{Float64,2})

returns a row vector containing the norm of each column.

#### Argument
* `aMatrix::Array{Float64,2}`: a matrix for which ones desires to compute the 2-norm of each of its column.
"""
function get_columns_two_norm(aMatrix::Array{Float64,2})
	return sqrt.(sum(aMatrix.^2, dims=1))
end

"""
normalising_columns(aMatrix)

returns a matrix where the column vectors are normalised in the 2-norm sense,
as if they where projected on the unit circle.

#### Arguments
* `aMatrix::Array{Float64,2}`: a matrix for which ones desires to normalise its column.
"""
function normalising_columns(aMatrix)
	return aMatrix./get_columns_two_norm(aMatrix)
end

end
