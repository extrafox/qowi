from matplotlib import pyplot
from skimage import color, io
from qowi.wavelet import Wavelet

def show_plot(labels, images):
	fig, axes = pyplot.subplots(3, 4, figsize=(12, 12))
	imshow_kwargs = dict(cmap=pyplot.cm.gray, interpolation='nearest')
	for i in range(len(images)):
		axis = axes[i // 4][i % 4]
		axis.imshow(images[i], **imshow_kwargs)

		axis.set_title(labels[i])
		axis.set_axis_off()
	pyplot.subplots_adjust(left=0, bottom=0, right=1, top=0.95, wspace=0, hspace=0.05)
	pyplot.show()

TEST_IMAGE_PATH = "media/dog_512x512.bmp"

display_images = []
display_labels = []

###
### Prepare image and intermediates
###

print("Processing image: {}".format(TEST_IMAGE_PATH))

image = io.imread(TEST_IMAGE_PATH)

display_labels.append("Original Image")
display_images.append(image)

w = Wavelet().prepare_from_image(image)

# display_labels.append("Wavelet Encoded Image")
# display_images.append(w.wavelet)

for bit_shift in range(0, 10):
	print("Processing bit shift {}".format(bit_shift))
	from_wavelet = w.as_image(bit_shift)
	display_labels.append("Wavelet Bit Shift {}".format(bit_shift))
	display_images.append(from_wavelet)

show_plot(display_labels, display_images)
print("Done")

