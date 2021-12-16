# Authors: Jorge Metri-Ojeda, Gabriel Solana-Lavalle. Universidad De Las Américas Puebla
# 14/December/2021.
# Computer vision and image analysis

# Importing the packages for image processing
using ImageView, FileIO, ImageMorphology, Images, Distributed, ImageSegmentation
using StatsBase, Plots, CSV, DataFrames, LinearAlgebra, Tables

#%% Texture features (Haralich features)
#Function for making a quantization of an image in N gray levels
function convert_to_255(img)
	img = luminosity(img)
	x, y = size(img)
	I_img = zeros(x,y)
	for i = 1:x
		for j = 1:y
			I_img[i,j] = Int(floor(img[i,j]*255))
		end
	end
	return I_img
end
# iteration at 0 ° to create a GLCM
function GLCM_0(img, GLCM, step, diff = 0, starts_zero = true)
	x, y = size(img)
	if starts_zero == true
		for i = 1:step:x
			for j = 1:step:y - 1
				row = Int(img[i,j] + 1)
				column = Int(img[i, j + 1] + 1)
				GLCM[row, column] += 1
			end
		end
	end
	if starts_zero == false
		for i = 1:step:x
			for j = 1:step:y - 1
				row = Int(img[i,j] - diff)
				column = Int(img[i,j + 1] - diff)
				if row >= 1 && column >= 1
					GLCM[row, column] += 1
				end
			end
		end
	end
	return GLCM
end
# iteration at 45 ° to create a GLCM
function GLCM_45(img, GLCM, step, diff = 0, starts_zero = true)
	x, y = size(img)
	if starts_zero == true
		for i = 2:step:x
			for j = 1:step:y - 1
				row = Int(img[i,j] + 1)
				column = Int(img[i - 1, j + 1] + 1)
				GLCM[row, column] += 1
			end
		end
	end
	if starts_zero == false
		for i = 2:step:x
			for j = 1:step:y - 1
				row = Int(img[i,j] - diff)
				column = Int(img[i - 1, j + 1] - diff)
				if row >= 1 && column >= 1
					GLCM[row, column] += 1
				end
			end
		end
	end
	return GLCM
end
# iteration at 90 ° to create a GLCM
function GLCM_90(img, GLCM, step, diff = 0, starts_zero = true)
	x, y = size(img)
	if starts_zero == true
		for i = 2:step:x
			for j = 1:step:y
				row = Int(img[i,j] + 1)
				column = Int(img[i - 1, j] + 1)
				if row >= 1 && column >= 1
					GLCM[row, column] += 1
				end
			end
		end
	end
	if starts_zero == false
		for i = 2:step:x
			for j = 1:step:y
				row = Int(img[i,j] - diff)
				column = Int(img[i - 1, j] - diff)
				GLCM[row, column] += 1
			end
		end
	end
	return GLCM
end
# iteration at 145 ° to create a GLCM
function GLCM_145(img, GLCM, step, diff = 0, starts_zero = true)
	x, y = size(img)
	if starts_zero == true
		for i = 2:step:x
			for j = 2:step:y
				row = Int(img[i,j] + 1)
				column = Int(img[i - 1, j - 1] + 1)
				GLCM[row, column] += 1
			end
		end
	end
	if starts_zero == false
		for i = 2:step:x
			for j = 2:step:y
				row = Int(img[i,j] - diff)
				column = Int(img[i - 1, j - 1] - diff)
				if row >= 1 && column >= 1
					GLCM[row, column] += 1
				end
			end
		end
	end
	return GLCM
end
# Creating a GLCM
function GLCM(img, level_x, level_y, step, degree)
	img = convert_to_255(img)
	if level_x != 0
		mat = zeros(level_y, level_y)
		diff = 0 + (level_x - 1)
		if degree == 0
			mat = GLCM_0(img, mat, step, diff, false)
		end
		if degree == 45
			mat = GLCM_45(img, mat, step, diff, false)
		end
		if degree == 90
			mat = GLCM_90(img, mat, step, diff, false)
		end
		if degree == 145
			mat = GLCM_145(img, mat, step, diff, false)
		end
	end
	if level_x == 0
		mat = zeros(level_y + 1, level_y + 1)
		if degree == 0
			mat = GLCM_0(img, mat, step, 0, true)
		end
		if degree == 45
			mat = GLCM_45(img, mat, step, 0, true)
		end
		if degree == 90
			mat = GLCM_90(img, mat, step, 0, true)
		end
		if degree == 145
			mat = GLCM_145(img, mat, step, 0, true)
		end
	end
	return mat
end
#Function for the summation of all combinations in an GLCM
function sum_neighbours(GLCM)
	x, y = size(GLCM)
	count = 0
	for i = 1:x
		for j = 1:y
			count += GLCM[i,j]
		end
	end
	return count
end
# Function for obtaining a normalized GLCM -> p(i,j)
function normalized_GLCM(GLCM)
	x, y = size(GLCM)
	matrix = zeros(x, y)
	total_combinations = sum_neighbours(GLCM)
	for i = 1:x
		for j = 1:y
			matrix[i,j] = GLCM[i,j]/total_combinations
		end
	end
	return matrix
end
# Function for obtaining Px(i)
function probability_x(NGLCM, i)
	x, y = size(NGLCM)
	result = 0
	for j = 1:y
		result += NGLCM[i, j]
	end
	return result
end
# Function for obtaining Py(j)
function probability_y(NGLCM, j)
	x, y = size(NGLCM)
	result = 0
	for i = 1:x
		result += NGLCM[i, j]
	end
	return result
end
# Function for obtaining the mean of x
function mean_x(NGLCM)
	x, y = size(NGLCM)
	result = 0
	for i = 1:x
		result += (i * probability_x(NGLCM, i))
	end
	return result
end
# Function for obtaining the the mean of y
function mean_y(NGLCM)
	x, y = size(NGLCM)
	result = 0
	for j = 1:y
		result += (j * probability_y(NGLCM,j))
	end
	return result
end
# Function for obtaining the varaince of X
function var_x(NGLCM)
	x, y = size(NGLCM)
	result = 0
	mean = mean_x(NGLCM)
	for i = 1:x
		result += ((i - mean)^2 * probability_x(NGLCM, i))
	end
	return result
end
# Function for obtaining the varaince of Y
function var_y(NGLCM)
	x, y = size(NGLCM)
	result = 0
	mean = mean_y(NGLCM)
	for j = 1:y
		result += ((j - mean)^2 * probability_y(NGLCM, j))
	end
	return result
end
# function to calculate p(K) where i + j is == k
function prob_k_plus(NGLCM, k)
	x, y = size(NGLCM)
	result = 0
	for i = 1:x
		for j = 1:y
			if i + j == k
				result += NGLCM[i, j]
			end
		end
	end
	return result
end
# function to calculate p(K) where |i - j| is == k
function prob_k_minus(NGLCM, k)
	x, y = size(NGLCM)
	result = 0
	for i = 1:x
		for j = 1:y
			if abs(i - j) == k
				result += NGLCM[i, j]
			end
		end
	end
	return result
end
# function to calculate the mean probability(k) where i + j is == k
function mean_k_plus(NGLCM)
	x, y = size(NGLCM)
	result = 0
	for k = 2:(2*x)
		result += (k * prob_k_plus(NGLCM, k))
	end
	return result
end
# function to calculate p mean(K) where |i - j| is == k
function mean_k_minus(NGLCM)
	x, y = size(NGLCM)
	result = 0
	for k = 0: x - 1
		result += (k * prob_k_minus(NGLCM, k))
	end
	return result
end
# Function to calculate HX
function HX(NGLCM)
	x, y = size(NGLCM)
	result = 0
	max = 0
	for i = 1:x
		val = probability_x(NGLCM, i)
		if val == 0
			result += (val * log(1))
		end
		if val != 0
			result += (val * log(val))
		end
	end
	return -(result)
end
# Function to calculate HY
function HY(NGLCM)
	x, y = size(NGLCM)
	result = 0
	for j = 1:y
		val = probability_y(NGLCM, j)
		if val == 0
			result += (val * log(1))
		end
		if val != 0
			result += (val * log(val))
		end
	end
	return -(result)
end
# Function to find the max HX
function max_HX(NGLCM)
	x, y = size(NGLCM)
	max = 0
	for i = 1:x
		val = probability_x(NGLCM, i)
		if val == 0
			result = (val * log(1))
		end
		if val != 0
			result = (val * log(val))
		end
		if abs(result) > max
			max = result
		end
	end
	return max
end
# Function to find the max HY
function max_HY(NGLCM)
	x, y = size(NGLCM)
	max = 0
	for j = 1:y
		val = probability_y(NGLCM, j)
		if val == 0
			result = (val * log(1))
		end
		if val != 0
			result = (val * log(val))
		end
		if abs(result) > max
			max = result
		end
	end
	return max
end
# Function to calculate HXY
function HXY(NGLCM)
	x, y = size(NGLCM)
	result = 0
	for i = 1:x
		for j = 1:y
			val = NGLCM[i, j]
			if val == 0
				result += (
				NGLCM[i, j] * log(1)
				)
			end
			if val != 0
				result += (
				NGLCM[i, j] * log(NGLCM[i, j])
				)
			end
		end
	end
	return -(result)
end
# Function to calculate HXY1
function HXY1(NGLCM)
	x, y = size(NGLCM)
	result = 0
	for i = 1:x
		b = probability_x(NGLCM, i)
		for j = 1:y
			a = NGLCM[i, j]
			c = probability_y(NGLCM, j)
			if b == 0 || c == 0
				result += 0
			end
			if b != 0 && c != 0
				result += (a * log(b * c))
			end
		end
	end
	return -(result)
end
# Function to calculate HXY2
function HXY2(NGLCM)
	x, y = size(NGLCM)
	result = 0
	for i = 1:x
		a = probability_x(NGLCM, i)
		for j = 1:y
			b = probability_y(NGLCM, j)
			if a == 0 || b == 0
				result += 0
			end
			if a != 0 && b != 0
				result += (a * b * log(a * b))
			end
		end
	end
	return -(result)
end

######## features ############
# Function to calculate Autocorrelation
function autocorr(NGLCM)
	result = 0
	x, y = size(NGLCM)
	for i = 1:x
		for j = 1:y
			result += (i * j * NGLCM[i ,j])
		end
	end
	return result
end
# Function to calculate cluster prominence
function cluster_prom(NGLCM)
	result = 0
	x, y = size(NGLCM)
	mean = mean_x(NGLCM)
	for i = 1:x
		for j = 1:y
			result += (
			(i + j - (2*mean))^3 * NGLCM[i, j]
			)
		end
	end
	return result
end
# Function to calculate cluster shade
function cluster_shade(NGLCM)
	result = 0
	x, y = size(NGLCM)
	mean = mean_x(NGLCM)
	for i = 1:x
		for j = 1:y
			result += (
			(i + j - (2*mean))^4 * NGLCM[i, j]
			)
		end
	end
	return result
end
# Function to calculate correlation
function corr(NGLCM)
	result = 0
	x, y = size(NGLCM)
	meanx = mean_x(NGLCM)
	varx = var_x(NGLCM)
	meany = mean_y(NGLCM)
	vary = var_y(NGLCM)
	for i = 1:x
		for j = 1:y
			div1= (i - meanx)/varx
			div2= (j - meany)/vary
			result += div1 * div2 * NGLCM[i,j]
		end
	end
	return result
end
# Function to calculate contrast
function contrast(NGLCM)
	result = 0
	x, y = size(NGLCM)
	for i = 1:x
		for j = 1:y
			result += (
			(i-j)^2 * NGLCM[i ,j]
			)
		end
	end
	return result
end
# Function to calculate disimilarity
function disimilarity(NGLCM)
	result = 0
	x, y = size(NGLCM)
	for i = 1:x
		for j = 1:y
			result += (
			abs(i-j) * NGLCM[i ,j]
			)
		end
	end
	return result
end
# Function to calculate energy
function energy(NGLCM)
	result = 0
	x, y = size(NGLCM)
	for i = 1:x
		for j = 1:y
			result += (
			NGLCM[i ,j]^2
			)
		end
	end
	return result
end
# Function to calculate entropy
function entropy(NGLCM)
	result = 0
	x, y = size(NGLCM)
	for i = 1:x
		for j = 1:y
			if NGLCM[i, j] == 0
				result += NGLCM[i, j] * log(1)
			end
			if NGLCM[i, j] != 0
				result += (
				NGLCM[i, j] * log(NGLCM[i, j])
				)
			end
		end
	end
	return -(result)
end
# Function to calculate entropy difference
function entropy_diff(NGLCM)
	result = 0
	x, y = size(NGLCM)
	for k = 0: x-1
		val = prob_k_minus(NGLCM, k)
		if val == 0
			result += (
			val * log(1)
			)
		end
		if val != 0
			result += (
			val * log(val)
			)
		end
	end
	return -(result)
end
# Function to calculate varianmce difference
function variance_diff(NGLCM)
	result = 0
	x, y = size(NGLCM)
	mean = mean_k_minus(NGLCM)
	for k = 0: x-1
		val = prob_k_minus(NGLCM, k)
		if val == 0
			result += 0
		end
		if val != 0
			result += (
			(k - mean)^2 * val
			)
		end
	end
	return result
end
# Function to calculate homogeneity
function homogeneity(NGLCM)
	result = 0
	x, y = size(NGLCM)
	for i = 1:x
		for j = 1:y
			result += (
			NGLCM[i ,j] / (1 + (i-j)^2)
			)
		end
	end
	return result
end
# Function to calculate measure of correlation 1
function meas_corr_1(NGLCM)
	a = HXY(NGLCM)
	b = HXY1(NGLCM)
	c = HX(NGLCM)
	d = HY(NGLCM)
	result = (a - b) / (c * d)
	return result
end
# Function to calculate measure of correlation 2
function meas_corr_2(NGLCM)
	a = HXY(NGLCM)
	b = HXY2(NGLCM)
	result = sqrt(1 - exp(-2*(b - a)))
	return result
end
#Function to calculate inverse difference
function inv_diff(NGLCM)
	result = 0
	x, y = size(NGLCM)
	for i = 1:x
		for j = 1:y
			result += (
			NGLCM[i, j] /(1 + abs(i-j))
			)
		end
	end
	return result
end
#Function to calculate the maximum probability
function max_p(NGLCM)
	result = maximum(NGLCM)
	return result
end
# Function for calculate the sum average
function sum_average(NGLCM)
	result = mean_k_plus(NGLCM)
	return result
end
# Function for calculate the sum entropy
function sum_entropy(NGLCM)
	x, y = size(NGLCM)
	result = 0
	for k = 2:(2*x)
		a = prob_k_plus(NGLCM, k)
		if a == 0
			result += 0
		end
		if a != 0
			result += (
			a * log(a)
			)
		end
	end
	return -(result)
end
#Function to calculate the sum of squares
function sum_squares(NGLCM)
	result = 0
	x, y = size(NGLCM)
	mean = mean_x(NGLCM)
	for i = 1:x
		for j = 1:y
			result += (
			(i - mean)^2 * NGLCM[i,j]
			)
		end
	end
	return result
end
#Function to calculate the sum of variance
function sum_variance(NGLCM)
	x, y = size(NGLCM)
	result = 0
	a = mean_k_plus(NGLCM)
	for k = 2:(2*x)
		b = prob_k_plus(NGLCM, k)
		if b == 0
			result += 0
		end
		if b != 0
			result += (
			(k - a)^2 - b
			)
		end
	end
	return result
end
