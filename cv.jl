# Authors: Jorge Metri-Ojeda, Gabriel Solana-Lavalle. Universidad De Las Américas Puebla
# __/December/2021.
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

#%%
# Function for processing the images
function img_process(df, filepath, file_extension,s_element,
	low_lim = 0.50, high_lim = 0.70, level_x = 1, level_y = 255, step = 10, degree = 0, n_samples = 1)
	samples = zeros(Int, n_samples, 1)
	for i = 1:n_samples
		n = df[i,1]
		samples[i,1] = n
	end
	array = zeros(n_samples, 56) # 55 features + first column with sample ID
	# Initialize feature extraction and indexing in array
	for i = 1:n_samples
		println("Segmentation and conversion colors of image $(i)\n")
		name = string(Int((samples[i,1])))
		img = load(filepath*name*"."*file_extension)
		img_op = mask_open(img, low_lim, high_lim, s_element)
		img_rgb = superimpose(img, img_op) # RGB image
		img_hsv = HSV.(img_rgb) # Conversion to HSV color space
		img_lab = Lab.(img_rgb) # Conversion to Lab color space
		array[i, 1] = samples[i,1]
		println("Starting the feature extraction of image $(i)\n")
		###### RGB Features Extraction ##### (img, low_limit_threshold, img_component, channel)
		array[i, 2] = mean_rgb(img_rgb, "red")
		array[i, 3] = mean_rgb(img_rgb, "green")
		array[i, 4] = mean_rgb(img_rgb, "blue")
		array[i, 5] = standard_dev_rgb(img_rgb, "red")
		array[i, 6] = standard_dev_rgb(img_rgb, "green")
		array[i, 7] = standard_dev_rgb(img_rgb, "blue")
		array[i, 8] = min_rgb(img_rgb, "red")
		array[i, 9] = min_rgb(img_rgb, "green")
		array[i, 10] = min_rgb(img_rgb, "blue")
		array[i, 11] = max_rgb(img_rgb, "red")
		array[i, 12] = max_rgb(img_rgb, "green")
		array[i, 13] = max_rgb(img_rgb, "blue")
		print("RGB feature extraction completed\n")
		###### HSV Features Extraction ##### (img, low_limit_threshold, up_limit_threshold, img_component, channel)
		array[i, 14] = mean_hsv(img_hsv, "hue")
		array[i, 15] = mean_hsv(img_hsv, "saturation")
		array[i, 16] = mean_hsv(img_hsv, "value")
		array[i, 17] = standard_dev_hsv(img_hsv, "hue")
		array[i, 18] = standard_dev_hsv(img_hsv, "saturation")
		array[i, 19] = standard_dev_hsv(img_hsv, "value")
		array[i, 20] = min_hsv(img_hsv, "hue")
		array[i, 21] = min_hsv(img_hsv, "saturation")
		array[i, 22] = min_hsv(img_hsv, "value")
		array[i, 23] = max_hsv(img_hsv, "hue")
		array[i, 24] = max_hsv(img_hsv, "saturation")
		array[i, 25] = max_hsv(img_hsv, "value")
		print("HSV feature extraction completed\n")
		##### Lab Features Extraction ####### (img, low_limit_threshold, up_limit_threshold, img_component, channel)
		array[i, 26] = mean_lab(img_lab, "luminosity")
		array[i, 27] = mean_lab(img_lab, "a")
		array[i, 28] = mean_lab(img_lab, "b")
		array[i, 29] = standard_dev_lab(img_lab, "luminosity")
		array[i, 30] = standard_dev_lab(img_lab, "a")
		array[i, 31] = standard_dev_lab(img_lab, "b")
		array[i, 32] = min_lab(img_lab, "luminosity")
		array[i, 33] = min_lab(img_lab, "a")
		array[i, 34] = min_lab(img_lab, "b")
		array[i, 35] = max_lab(img_lab, "luminosity")
		array[i, 36] = max_lab(img_lab, "a")
		array[i, 37] = max_lab(img_lab, "b")
		print("Lab feature extraction completed\n")
		###### Haralich Features Extraction ##############
		co_ocur_mat = GLCM(img_rgb, level_x, level_y, step, degree)
		norm_mat = normalized_GLCM(co_ocur_mat)
		array[i, 38] = autocorr(norm_mat)
		array[i, 39] = cluster_prom(norm_mat)
		array[i, 40] = cluster_shade(norm_mat)
		array[i, 41] = corr(norm_mat)
		array[i, 42] = contrast(norm_mat)
		array[i, 43] = disimilarity(norm_mat)
		array[i, 44] = energy(norm_mat)
		array[i, 45] = entropy(norm_mat)
		array[i, 46] = entropy_diff(norm_mat)
		array[i, 47] = variance_diff(norm_mat)
		array[i, 48] = homogeneity(norm_mat)
		array[i, 49] = meas_corr_1(norm_mat)
		array[i, 50] = meas_corr_2(norm_mat)
		array[i, 51] = inv_diff(norm_mat)
		array[i, 52] = max_p(norm_mat)
		array[i, 53] = sum_average(norm_mat)
		array[i, 54] = sum_entropy(norm_mat)
		array[i, 55] = sum_squares(norm_mat)
		array[i, 56] = sum_variance(norm_mat)
		print("Haralich features extraction completed\n")
		println("Feature extraction of image $(i): finished\n")
		print("----------------------------------------------\n")
	end
	CSV.write("C:/Users/metri/Desktop/Julia/prueba1.csv", Tables.table(array), writeheader=false)
	println("$(i) images were succesfully processed")
end

#%%
# Create a structural element for the image segmentation.
# Shape and size should be selected based on the image characteristics'
struct_element = structural_element("disk", 29)
# Importing CSV to create a new dataset with the 57 features
df_frame = DataFrame(CSV.File("C:/Users/metri/Desktop/Julia/practica1.csv"))
# Start function with the params desired.
# Thresholding limits and gray levels should be initialized based on the samples characteristics'
file_path = "C:/Users/metri/Desktop/Julia/processing/"
file_extension = "jpg" #file extension:'png', 'jpg', 'jpeg' without the dot, this is concatenated in the function
img_process(df_frame, file_path, file_extension, struct_element, 0.50, 0.70, 1, 255, 10, 0, 5)


#%%
