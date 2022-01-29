# Authors: Jorge Metri-Ojeda, Gabriel Solana-Lavalle. Universidad De Las Américas Puebla
# 14/December/2021.
# Computer vision and image analysis

# Importing the packages for image processing
using ImageView, FileIO, ImageMorphology, Images, Distributed, ImageSegmentation
using StatsBase, Plots, CSV, DataFrames, LinearAlgebra, Tables
using ImageBinarization

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
#function for creating a mask using a standard algortihm
function create_mask(img, algorithm, element)
	mask = binarize(img, algorithm)
	mask = erosion(mask, element)
	return mask
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
# function to obtain a background of the img
function background(img, mask)
	bkg_img = copy(img)
	x, y = size(img)
	for i = 1:x
		for j = 1:y
			if mask[i,j] == 1
				bkg_img[i,j] = float64(0)
			end
		end
	end
	return bkg_img
end
# function for calculate the mean of gray levels in the background
function mean_bkg(img, mask)
	img_gray = luminosity(img)
	bkg = background(img_gray, mask)
	x, y = size(bkg)
	levels = 0
	count = 0
	for i = 1:x
		for j = 1:y
			if bkg[i,j] != 0
				levels += bkg[i,j]
				count += 1
			end
		end
	end
	result = levels/count
	return result
end
# function for calculate the mean of gray levels in the sample
function mean_gsample(img, mask)
	img_gray = luminosity(img)
	sample = superimpose(img_gray, mask)
	x, y = size(sample)
	levels = 0
	count = 0
	for i = 1:x
		for j = 1:y
			if sample[i,j] != 0
				levels += sample[i,j]
				count += 1
			end
		end
	end
	result = levels/count
	return result
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
# Function for calculate the mean of each hsv color channel
function mean_hsv(img, channel = "hue")
	channel_count = 0
	count_pix = 0
	x, y = size(img)
	if channel == "hue"
		t = threshold_color(img, 1)
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
		t = threshold_color(img, 1)
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
		t = threshold_color(img, 1)
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
		t = threshold_color(img, 1)
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
function yellow(img)
	x, y = size(img)
	mat = zeros(x,y)
	for i = 1:x
		for j = 1:y
			mat[i,j] = (red(img[i, j]) + green(img[i, j]))/2
		end
	end
	return mat
end
# function for calculate the mean of yellow levels in the sample: important to provide a rgb img
function mean_yellow(img_rgb)
	yell_img = yellow(img_rgb)
	x, y = size(yell_img)
	levels = 0
	count = 0
	for i = 1:x
		for j = 1:y
			if yell_img[i,j] != 0
				levels += yell_img[i,j]
				count += 1
			end
		end
	end
	result = levels/count
	return result
end
# function for calculate the std dev of yellow levels in the sample: important to provide a rgb img
function std_yellow(img_rgb)
	yell_img = yellow(img_rgb)
	mean = mean_yellow(img_rgb)
	x, y = size(yell_img)
	levels = 0
	count = 0
	for i = 1:x
		for j = 1:y
			if yell_img[i,j] != 0
				levels += ((yell_img[i,j] - mean)^2)
				count += 1
			end
		end
	end
	result = levels/count
	return result
end
# function for calculate the min of yellow levels in the sample: important to provide a rgb img
function min_yellow(img_rgb)
	yell_img = yellow(img_rgb)
	mean = mean_yellow(img_rgb)
	x, y = size(yell_img)
	levels = []
	for i = 1:x
		for j = 1:y
			if yell_img[i,j] != 0
				push!(levels, yell_img[i,j])
			end
		end
	end
	result = minimum(levels)
	return result
end
# function for calculate the max of yellow levels in the sample: important to provide a rgb img
function max_yellow(img_rgb)
	yell_img = yellow(img_rgb)
	mean = mean_yellow(img_rgb)
	x, y = size(yell_img)
	levels = []
	for i = 1:x
		for j = 1:y
			if yell_img[i,j] != 0
				push!(levels, yell_img[i,j])
			end
		end
	end
	result = maximum(levels)
	return result
end
# Calculation of skewness
function skewness(img, component)
	vector = create_vector(img, component)
	result = StatsBase.skewness(vector)
	return result
end
# Calculation of kurtosis
function kurtosis(img, component)
	vector = create_vector(img, component)
	result = StatsBase.kurtosis(vector)
	return result
end

######################## Nuevas características ################################


#promedio de los pixeles de fondo (de los que no tienen mayonesa) y de la mayonesa
#comparar los promedios => nueva característica abs(fondo - mayo)

# amarillo => las mismas que en todas media, std, min y  max
# skewness y kurtosis => a todos los canales en todos los espacios
# agregar la comparación
mayo = load("C:/Users/metri/Desktop/Julia/images/683.jpg")
f = Otsu()
s_element = structural_element("disk", 19)
my_mask = mask_open(mayo,0.50, 0.75, s_element)
mask2 = erosion(my_mask, s_element)
prueba = binarize(mayo, f)
otsu_op = erosion(prueba, s_element)
imshow(mask2
imshow(my_mask)
imshow(prueba)
imshow(mayo)
imshow(otsu_op)
function create_mask(img, algorithm, element)
	mask = binarize(img, algorithm)
	mask = erosion(mask, element)
	return mask
end

prueba_func = create_mask(mayo, f, s_element)

mayo_gray =
mayo_si = superimpose(mayo, prueba_func)
bkg = background(mayo, prueba_func)

function background(img, mask)
	bkg_img = copy(img)
	x, y = size(img)
	for i = 1:x
		for j = 1:y
			if mask[i,j] == 1
				bkg_img[i,j] = float64(0)
			end
		end
	end
	return bkg_img
end
# function for calculate the mean of gray levels in the background
function mean_bkg(img, mask)
	img_gray = luminosity(img)
	bkg = background(img_gray, mask)
	x, y = size(bkg)
	levels = 0
	count = 0
	for i = 1:x
		for j = 1:y
			if bkg[i,j] != 0
				levels += bkg[i,j]
				count += 1
			end
		end
	end
	result = levels/count
	return result
end


mayo_si = superimpose(mayo, prueba_func)
imshow(mayo_si_2)
mayo_si_2 = yellow(mayo_si)
mean_yellow(mayo_si)
std_yellow(mayo_si)
min_yellow(mayo_si)
max_yellow(mayo_si)
