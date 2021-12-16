# Authors: Jorge Metri-Ojeda, Gabriel Solana-Lavalle. Universidad De Las Américas Puebla
# 14/December/2021.
# Computer vision and image analysis

# Importing the packages for image processing
using ImageView, FileIO, ImageMorphology, Images, Distributed, ImageSegmentation
using StatsBase, Plots, CSV, DataFrames, LinearAlgebra, Tables

#%% Color and edges features
# Function for gray_scale image
function luminosity(img)
    x, y = size(img)
    mat = zeros(Float64, x, y)
    for i = 1:x
        for j = 1:y
            mat[i, j] = (red(img[i, j]) * 0.21 + blue(img[i, j]) * 0.07
            + green(img[i, j]) * 0.72)
        end
    end
    return mat
end
# Function to make a filter for convolution
function make_filter(name::String)
	if name == "laplacian"
		filter =  [[0,1,0] [1,-4,1] [0,1,0]]
	end
	if name == "minus laplacian"
		filter =  [[0,-1,0] [-1,5,-1] [0,-1,0]]
	end
	if name == "minus mean"
		filter =  [[-1,-1,-1] [-1,8,-1] [-1,-1,-1]]
	end
	if name == "C sobel"
		filter = [[-1,-2,-1] [0,0,0] [1,2,1]]
	end
	if name == "F sobel"
		filter = [[-1,0,1] [-2,0,2] [-1,0,1]]
	end
	if name == "north"
		filter = [[1,1,-1] [1,-2,-1] [1,1,-1]]
	end
	if name == "east"
		filter = [[-1,-1,-1] [1,-2,1] [1,1,1]]
	end
	return filter
end
# Helper function of convolution
function multiplication_filter(filter, f_size, mat)
    n = 0
    for i = 1:f_size[1]
        for j = 1:f_size[2]
            n +=  (filter[i,j] * mat[i,j])
        end
    end
    return n
end
# Function for convolutional filtering
function convolution(img, filter)
    f_size = size(filter)
    i_edge = Int(ceil((f_size[1]/2)))
    f_edge = Int(floor((f_size[1]/2)))
    x, y = size(img)
    f_img = zeros(x, y)
    for i = i_edge:x-f_edge
        for j = i_edge:y-f_edge
            f_img[i,j] = multiplication_filter(filter,
                                            f_size,
                                            img[i - f_edge:i + f_edge,j - f_edge:j + f_edge])
        end
    end
    return f_img
end
# Function for separate the background of an image in 0 and 1
function binary(img, threshold)
    x, y = size(img)
    mat = zeros(Float64, x, y)
	for i = 1:x
	  for j = 1:y
		  if img[i,j] > threshold
			  mat[i,j] = 1
		  else img[i,j] < threshold
			  mat[i,j] = 0
		  end
	  end
 	end
    return mat
end
# Function for creating a square with values 1 into an array
function square(mat, init_pos, end_pos)
	for i = init_pos:end_pos
		for j = init_pos:end_pos
			mat[i,j] = 1
		end
	end
	return mat
end
# Function for creating a rectangle with values 1 into an array
# where (a, b) are height initial row & ending row,
# and (c, d) are the base initial column & ending column of the rectangle
function rectangle(mat, (a, b), (c, d))
	if a > b || c  > d
		print("a or c value should be lower than b or d, respectively")
		return
	end
	for i = a:b
		for j = c:d
			if mat[i,j] == 1
				continue
			elseif mat[i,j] == 0
				mat[i,j] = 1
			end
		end
	end
	return mat
end
# Function for creating a triangle with values 0 and 1
function triangle(size)
	mat = zeros(Float64, size, size)
	count = size
	while count >= 1
		for i = 1:size
			for j = count:size
				mat[i,j] = 1
			end
			count -= 1
		end
	end
	return mat
end
# Function for make a mirror copy in the x axis
function mirror_x(shape, axis)
	x, y = size(shape)
	mirror = zeros(Float64, x, y)
	count = y
	while count >= 1
	if axis == "x"
		for i = 1:x
			for j = 1:y
				mirror[i,j] = shape[i, count]
				count -= 1
			end
		count = y
		end
	end
	mirror_shaped = [shape mirror]
	return mirror_shaped
	end
end
# Function for make a mirror copy in the y axis
function mirror_y(shape)
	x, y = size(shape)
	mirror = zeros(Float64, x, y)
	count = y
	while count >= 1
		for i = 1:x
			for j = 1:y
				mirror[i,j] = shape[count, j]
			end
		count -= 1
		end
	end
	mirror_shaped = [shape ; mirror]
	return mirror_shaped
end
# Function for make a mirror copy of a whole matrix
function mirror(shape)
	mat = mirror_y(shape)
	mat = mirror_x(mat, "x")
	return mat
end
# Function tor develop a structural element for opening and closing
# rect_height is used if disk and cross are selected
function structural_element(shape, size::Int = 3, rect_height::Int = 3)
	if size % 2 == 0
		print("The size must be odd")
		return
	end
	if shape == "diamond"
		mat = triangle(size)
		mat = mirror(mat)
	end
	if shape == "square"
		mat = ones(size, size)
	end
	if shape == "cross"
		mid = Int(ceil(size / 2))
		init = mid - Int(floor(rect_height/ 2))
		final = mid + Int(floor(rect_height/ 2))
		mat = zeros(size, size)
		mat = rectangle(mat, (init, final), (1, size))
		mat = rectangle(mat, (1, size), (init, final))
	end
	if shape == "disk"
		mid = Int(ceil(size / 2))
		init = mid - Int(floor(rect_height/ 2))
		final = mid + Int(floor(rect_height/ 2))
		mat = zeros(size, size)
		mat = square(mat, 2, size -1)
		mat = rectangle(mat, (init, final), (1, size))
		mat = rectangle(mat, (1, size), (init, final))
	end
	return mat
end
# Function for verifying the erosion process in a matrix
function verify_erode(s_element, size, mat)
	n = 1
	for i = 1:size
		for j = 1:size
			if s_element[i,j] == 1 && s_element[i,j] != mat[i,j]
					n = 0
			end
		end
	end
	return n
end
# Function for verifying the dilation process in a matrix
function verify_dilate(s_element, size, mat)
	n = 0
	for i = 1:size
		for j = 1:size
			if s_element[i,j] == 1 && s_element[i,j] == mat[i,j]
					n = 1
			end
		end
	end
	return n
end
# Erosion function
function erosion(img_original, s_element)
	mat = copy(img_original)
	x, y = size(img_original)
	s_size = size(s_element)
	i_edge = Int(ceil((s_size[1]/2)))
    f_edge = Int(floor((s_size[1]/2)))
	for i = i_edge:x-f_edge
		for j = i_edge:y-f_edge
			mat[i,j] = verify_erode(s_element,
										s_size[1],
										img_original[i - f_edge:i + f_edge,j - f_edge:j + f_edge])
		end
	end
	return mat
end
# Dilation function
function dilation(img_original, s_element)
	mat = copy(img_original)
	x, y = size(img_original)
	s_size = size(s_element)
	i_edge = Int(ceil((s_size[1]/2)))
    f_edge = Int(floor((s_size[1]/2)))
	for i = i_edge:x-f_edge
		for j = i_edge:y-f_edge
			mat[i,j] = verify_dilate(s_element,
										s_size[1],
										img_original[i - f_edge:i + f_edge,j - f_edge:j + f_edge])
		end
	end
	return mat
end
# Opening function: dilation(erosion(img))
function open(img_original, s_element)
	eroded_image = erosion(img_original, s_element)
	open_img = dilation(eroded_image, s_element)
	return open_img
end
# closing function: erosion(dilation(img))
function close(img_original,s_element)
	dilated_image = dilation(img_original, s_element)
	close_img = erosion(dilated_image,s_element)
	return close_img
end
# Function for preprocessing the mask of an image with opening
function mask_open(img_original, low_lim, high_lim, s_element)
	gray_img = luminosity(img_original)
	threshold = select_threshold(gray_img, low_lim, high_lim)
	work_img = binary(gray_img, threshold)
	work_img = erosion(work_img, s_element)
	open_img_1 = open(work_img, s_element)
	open_img_2 = open(open_img_1, s_element)
	return open_img_2
end
# Function for selecting the threshold
function select_threshold(img, low_lim, high_lim)
	vector = vec(img)
	hist = fit(Histogram, vector)
	val = 10000000000
	threshold = 0
	for i = low_lim:0.02:high_lim
		x = searchsortedfirst(hist.edges[1], i)
		y = hist.weights[x]
		if y < val
			val = y
			threshold = i
		end
	end
	return threshold
end
# Function for preprocessing the mask of an image with closing
function mask_close(img, low_lim, high_lim, s_element)
	gray_img = luminosity(img_original)
	threshold = select_threshold(gray_img, low_lim, high_lim)
	work_img = binary(gray_img, threshold)
	close_img_1 = close(work_img, s_element)
	close_img_2 = close(close_img_1, s_element)
	return close_img_2
end
# Function to superimpose the original image in the mask
function superimpose(img, mask)
	super_img = copy(img)
	x, y = size(img)
	for i = 1:x
		for j = 1:y
			if mask[i,j] == 0
				super_img[i,j] = float64(0)
			end
		end
	end
	return super_img
end
# Create a vector for thresholding an image components
function create_vector(img, component = 1)
	vector = []
	x, y = size(img)
	if component == 1
		for i = 1:x
			for j = 1:y
				if float64(comp1(img[i,j])) != 0
					push!(vector,float64(comp1(img[i,j])))
				end
			end
		end
	end
	if component == 2
		for i = 1:x
			for j = 1:y
				if float64(comp1(img[i,j])) != 0
					push!(vector,float64(comp2(img[i,j])))
				end
			end
		end
	end
	if component == 3
		for i = 1:x
			for j = 1:y
				if float64(comp1(img[i,j])) != 0
					push!(vector,float64(comp3(img[i,j])))
				end
			end
		end
	end
	return Float64.(vector)
end
# Selecting the threshold for RGB, Saturation, Value, and Lab features
function threshold_color(img, component)
	vector = create_vector(img, component)
	hist = fit(Histogram, vector)
	x = length(hist.weights)
	l_values = collect(hist.edges[1])
	mean = sum(l_values)/x
	numerator = 0
	for i = 1:length(l_values)
		numerator += (l_values[i] - mean) ^2
	end
	std_dev = sqrt(numerator/(x-1))
	l_lim = mean - (1*std_dev)
	h_lim = mean + (1*std_dev)
	return l_lim, h_lim
end
# Function for calculate the mean of each RGB color channel
function mean_rgb(img, channel = "red")
	channel_count = 0
	count_pix = 0
	x, y = size(img)
	if channel == "red"
		t = threshold_color(img, 1)
		for i = 1:x
			for j = 1:y
				if float64(red(img[i,j])) > t[1] && float64(red(img[i,j])) < t[2]
					channel_count += float64(red(img[i,j]))
					count_pix += 1
				end
			end
		end
	end
	if channel == "green"
		t = threshold_color(img, 2)
		for i = 1:x
			for j = 1:y
				if float64(green(img[i,j])) > t[1] && float64(green(img[i,j]))  < t[2]
					channel_count += float64(green(img[i,j]))
					count_pix += 1
				end
			end
		end
	end
	if channel == "blue"
		t = threshold_color(img, 1)
		for i = 1:x
			for j = 1:y
				if float64(blue(img[i,j])) > t[1] && float64(blue(img[i,j])) < t[2]
					channel_count += float64(blue(img[i,j]))
					count_pix += 1
				end
			end
		end
	end
	result = channel_count/count_pix
	return result
end
# Function for calculate the std deviation of each RGB color channel
function standard_dev_rgb(img, channel = "red")
	channel_count = 0
	count_pix = 0
	mean = mean_rgb(img, channel)
	x, y = size(img)
	if channel == "red"
		t = threshold_color(img, 1)
		for i = 1:x
			for j = 1:y
				if float64(red(img[i,j])) > t[1] && float64(red(img[i,j])) < t[2]
					channel_count += ((float64(red(img[i,j])) - mean)^2)
					count_pix += 1
				end
			end
		end
	end
	if channel == "green"
		t = threshold_color(img, 2)
		for i = 1:x
			for j = 1:y
				if float64(green(img[i,j])) > t[1] && float64(green(img[i,j])) < t[2]
					channel_count += ((float64(green(img[i,j])) - mean)^2)
					count_pix += 1
				end
			end
		end
	end
	if channel == "blue"
		t = threshold_color(img, 3)
		for i = 1:x
			for j = 1:y
				if float64(blue(img[i,j])) > t[1] && float64(blue(img[i,j])) < t[2]
					channel_count += ((float64(blue(img[i,j])) - mean)^2)
					count_pix += 1
				end
			end
		end
	end
	result = sqrt(channel_count/(count_pix - 1))
	return result
end
# Function for obtaining the minimum of a rgb channel
function min_rgb(img, channel = "red")
	x, y = size(img)
	mat = zeros(x, y)
	colors = []
	if channel == "red"
		t = threshold_color(img, 1)
		for i = 1:x
			for j = 1:y
				if float64(red(img[i,j])) > t[1] && float64(red(img[i,j])) < t[2]
					push!(colors, float64(red(img[i,j])))
				end
			end
		end
	end
	if channel == "green"
		t = threshold_color(img, 2)
		for i = 1:x
			for j = 1:y
				if float64(green(img[i,j])) > t[1] && float64(green(img[i,j])) < t[2]
					push!(colors, float64(green(img[i,j])))
				end
			end
		end
	end
	if channel == "blue"
		t = threshold_color(img, 3)
		for i = 1:x
			for j = 1:y
				if float64(blue(img[i,j])) > t[1] && float64(blue(img[i,j])) < t[2]
					push!(colors, float64(blue(img[i,j])))
				end
			end
		end
	end
	result = minimum(colors)
	return result
end
# Function for obtaining the maximum of a RGB channel
function max_rgb(img, channel = "red")
	x, y = size(img)
	mat = zeros(x, y)
	colors = []
	if channel == "red"
		t = threshold_color(img, 1)
		for i = 1:x
			for j = 1:y
				if float64(red(img[i,j])) > t[1] && float64(red(img[i,j])) < t[2]
					push!(colors, float64(red(img[i,j])))
				end
			end
		end
	end
	if channel == "green"
		t = threshold_color(img, 2)
		for i = 1:x
			for j = 1:y
				if float64(green(img[i,j])) > t[1] && float64(green(img[i,j])) < t[2]
					push!(colors, float64(green(img[i,j])))
				end
			end
		end
	end
	if channel == "blue"
		t = threshold_color(img, 3)
		for i = 1:x
			for j = 1:y
				if float64(blue(img[i,j])) > t[1] && float64(blue(img[i,j])) < t[2]
					push!(colors, float64(blue(img[i,j])))
				end
			end
		end
	end
	result = maximum(colors)
	return result
end
# Selecting the threshold for HSV hue values; filtering outliers
function HUE_threshold(img_hsv, component)
	vector = create_vector(img_hsv, component)
	hist = fit(Histogram, vector)
	x = length(hist.weights)
	l_values = collect(hist.edges[1])
	threshold = 0
	for i = 10:-1:1
		y = hist.weights[i]
		if y > 15000
			return threshold = l_values[i + 1]
		end
	end
end
# Function for calculate the mean of each hsv color channel
function mean_hsv(img, channel = "hue")
	channel_count = 0
	count_pix = 0
	x, y = size(img)
	if channel == "hue"
		t = HUE_threshold(img, 1)
		for i = 1:x
			for j = 1:y
				if float64(comp1(img[i,j])) > 0 && float64(comp1(img[i,j])) < t
					channel_count += float64(comp1(img[i,j]))
					count_pix += 1
				end
			end
		end
	end
	if channel == "saturation"
		t = threshold_color(img, 2)
		for i = 1:x
			for j = 1:y
				if float64(comp2(img[i,j])) > t[1] && float64(comp2(img[i,j])) < t[2]
					channel_count += float64(comp2(img[i,j]))
					count_pix += 1
				end
			end
		end
	end
	if channel == "value"
		t = threshold_color(img, 3)
		for i = 1:x
			for j = 1:y
				if float64(comp3(img[i,j])) > t[1] && float64(comp3(img[i,j])) < t[2]
					channel_count += float64(comp3(img[i,j]))
					count_pix += 1
				end
			end
		end
	end
	result = channel_count/count_pix
	return result
end
# Function for calculate the std deviation of each hsv color channel
function standard_dev_hsv(img, channel = "hue")
	channel_count = 0
	count_pix = 0
	mean = mean_hsv(img, channel)
	x, y = size(img)
	if channel == "hue"
		t = HUE_threshold(img, 1)
		for i = 1:x
			for j = 1:y
				if float64(comp1(img[i,j])) > 0 && float64(comp1(img[i,j])) < t
					n = (float64(comp1(img[i,j])) - mean)
					channel_count += n^2
					count_pix += 1
				end
			end
		end
	end
	if channel == "saturation"
		t = threshold_color(img, 2)
		for i = 1:x
			for j = 1:y
				if float64(comp2(img[i,j])) > t[1] && float64(comp2(img[i,j])) < t[2]
					n = (float64(comp2(img[i,j])) - mean)
					channel_count += n^2
					count_pix += 1
				end
			end
		end
	end
	if channel == "value"
		t = threshold_color(img, 3)
		for i = 1:x
			for j = 1:y
				if float64(comp3(img[i,j])) > t[1] && float64(comp3(img[i,j])) < t[2]
					n = (float64(comp3(img[i,j])) - mean)
					channel_count += n^2
					count_pix += 1
				end
			end
		end
	end
	result = sqrt(channel_count/(count_pix - 1))
	return result
end
# Function for obtaining the minimum of a hsv channel
function min_hsv(img, channel = "hue")
	x, y = size(img)
	colors = []
	mat = zeros(x, y)
	if channel == "hue"
		t = HUE_threshold(img, 1)
		for i = 1:x
			for j = 1:y
				if float64(hue(img[i,j])) > 0 && float64(comp1(img[i,j])) < t
					push!(colors, float64(comp1(img[i,j])))
				end
			end
		end
	end
	if channel == "saturation"
		t = threshold_color(img, 2)
		for i = 1:x
			for j = 1:y
				if float64(comp2(img[i,j])) > t[1] && float64(comp2(img[i,j])) < t[2]
					push!(colors, float64(comp2(img[i,j])))
				end
			end
		end
	end
	if channel == "value"
		t = threshold_color(img, 3)
		for i = 1:x
			for j = 1:y
				if float64(comp3(img[i,j])) > t[1] && float64(comp3(img[i,j])) < t[2]
					push!(colors,float64(comp3(img[i,j])))
				end
			end
		end
	end
	result = minimum(colors)
	return result
end
# Function for obtaining the maximum of a hsv channel
function max_hsv(img, channel = "hue")
	x, y = size(img)
	colors = []
	mat = zeros(x, y)
	if channel == "hue"
		t = HUE_threshold(img, 1)
		for i = 1:x
			for j = 1:y
				if float64(comp1(img[i,j])) > 0 && float64(comp1(img[i,j])) < t
					push!(colors, float64(comp1(img[i,j])))
				end
			end
		end
	end
	if channel == "saturation"
		t = threshold_color(img, 2)
		for i = 1:x
			for j = 1:y
				if float64(comp2(img[i,j])) > t[1] && float64(comp2(img[i,j])) < t[2]
					push!(colors, float64(comp2(img[i,j])))
				end
			end
		end
	end
	if channel == "value"
		t = threshold_color(img, 3)
		for i = 1:x
			for j = 1:y
				if float64(comp3(img[i,j])) > t[1] && float64(comp3(img[i,j])) < t[2]
					push!(colors,float64(comp3(img[i,j])))
				end
			end
		end
	end
	result = maximum(colors)
	return result
end
# Function for calculate the mean of each Lab color channel
function mean_lab(img, channel = "luminusoty")
	channel_count = 0
	count_pix = 0
	x, y = size(img)
	if channel == "luminosity"
		t = threshold_color(img, 1)
		for i = 1:x
			for j = 1:y
				if float64(comp1(img[i,j])) > t[1] && float64(comp1(img[i,j])) < t[2]
					channel_count += float64(comp1(img[i,j]))
					count_pix += 1
				end
			end
		end
	end
	if channel == "a"
		t = threshold_color(img, 2)
		for i = 1:x
			for j = 1:y
				if float64(comp2(img[i,j])) > t[1] && float64(comp2(img[i,j])) < t[2]
					channel_count += float64(comp2(img[i,j]))
					count_pix += 1
				end
			end
		end
	end
	if channel == "b"
		t = threshold_color(img, 3)
		for i = 1:x
			for j = 1:y
				if float64(comp3(img[i,j])) > t[1] && float64(comp3(img[i,j])) < t[2]
					channel_count += float64(comp3(img[i,j]))
					count_pix += 1
				end
			end
		end
	end
	result = channel_count/count_pix
	return result
end
# Function for calculate the std deviation of each Lab color channel
function standard_dev_lab(img, channel = "luminosity")
	channel_count = 0
	count_pix = 0
	mean = mean_lab(img, channel)
	x, y = size(img)
	if channel == "luminosity"
		t = threshold_color(img, 1)
		for i = 1:x
			for j = 1:y
				if float64(comp1(img[i,j])) > t[1] && float64(comp1(img[i,j])) < t[2]
					channel_count += ((float64(comp1(img[i,j])) - mean)^2)
					count_pix += 1
				end
			end
		end
	end
	if channel == "a"
		t = threshold_color(img, 2)
		for i = 1:x
			for j = 1:y
				if float64(comp2(img[i,j])) > t[1] && float64(comp2(img[i,j])) < t[2]
					channel_count += ((float64(comp2(img[i,j])) - mean)^2)
					count_pix += 1
				end
			end
		end
	end
	if channel == "b"
		t = threshold_color(img, 3)
		for i = 1:x
			for j = 1:y
				if float64(comp3(img[i,j])) > t[1] && float64(comp3(img[i,j])) < t[2]
					channel_count += ((float64(comp3(img[i,j])) - mean)^2)
					count_pix += 1
				end
			end
		end
	end
	result = sqrt(channel_count/(count_pix - 1))
	return result
end
# Function for obtaining the minimum of a Lab channel
function min_lab(img, channel = "luminosity")
	x, y = size(img)
	colors = []
	mat = zeros(x, y)
	if channel == "luminosity"
		t = threshold_color(img, 1)
		for i = 1:x
			for j = 1:y
				if float64(comp1(img[i,j])) > t[1] && float64(comp1(img[i,j])) < t[2]
					push!(colors, float64(comp1(img[i,j])))
				end
			end
		end
	end
	if channel == "a"
		t = threshold_color(img, 2)
		for i = 1:x
			for j = 1:y
				if float64(comp2(img[i,j])) > t[1] && float64(comp2(img[i,j])) < t[2]
					push!(colors, float64(comp2(img[i,j])))
				end
			end
		end
	end
	if channel == "b"
		t = threshold_color(img, 3)
		for i = 1:x
			for j = 1:y
				if float64(comp3(img[i,j])) > t[1] && float64(comp3(img[i,j])) < t[2]
					push!(colors, float64(comp3(img[i,j])))
				end
			end
		end
	end
	result = minimum(colors)
	return result
end
# Function for obtaining the maximum of a Lab channel
function max_lab(img, channel = "luminosity")
	x, y = size(img)
	colors = []
	mat = zeros(x, y)
	if channel == "luminosity"
		t = threshold_color(img, 1)
		for i = 1:x
			for j = 1:y
				if float64(comp1(img[i,j])) > t[1] && float64(comp1(img[i,j])) < t[2]
					push!(colors, float64(comp1(img[i,j])))
				end
			end
		end
	end
	if channel == "a"
		t = threshold_color(img, 2)
		for i = 1:x
			for j = 1:y
				if float64(comp2(img[i,j])) > t[1] && float64(comp2(img[i,j])) < t[2]
					push!(colors, float64(comp2(img[i,j])))
				end
			end
		end
	end
	if channel == "b"
		t = threshold_color(img, 3)
		for i = 1:x
			for j = 1:y
				if float64(comp3(img[i,j])) > t[1] && float64(comp3(img[i,j])) < t[2]
					push!(colors, float64(comp3(img[i,j])))
				end
			end
		end
	end
	result = maximum(colors)
	return result
end
