#%%
# Function for processing the images
function img_process(df, filepath, file_ext, s_element,
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
		img = load(filepath*name*"."*file_ext)
		img_op = mask_open(img, low_lim, high_lim, s_element)
		img_rgb = superimpose(img, img_op) # RGB image
		img_hsv = HSV.(img_rgb) # Conversion to HSV color space
		img_lab = Lab.(img_rgb) # Conversion to Lab color space
		array[i, 1] = samples[i,1]
		println("Starting the feature extraction of image $(i)\n")
		###### RGB Features Extraction ##### (img, channel)
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
		###### HSV Features Extraction ##### (img, channel)
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
		##### Lab Features Extraction ####### (img, channel)
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
	CSV.write("file.csv", Tables.table(array), writeheader=false)
	println("$(n_samples) images were succesfully processed")
end

#%%
# Create a structural element for the image segmentation.
# Shape and size should be selected based on the image characteristics'
struct_element = structural_element("disk", 29)

# Importing CSV to create a new dataset with the 57 features
df_frame = DataFrame(CSV.File("file.csv"))

# Start function with the params desired.
# Thresholding limits and gray levels should be initialized based on the samples characteristics'

file_path = "C:/Users/metri/Desktop/Julia/images/"
file_ext = "jpg" #file extension:'png', 'jpg', 'jpeg' without the dot, this is concatenated in the function

img_process(df_frame, file_path, file_ext, struct_element, 0.50, 0.70, 1, 255, 10, 0, 50)
