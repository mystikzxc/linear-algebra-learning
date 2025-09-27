from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

img = Image.open('oboe-with-book.jpg')
plt.imshow(img)
# plt.show()

# convert img to greyscale to remove color channel complexity
imggrey = img.convert('LA')
plt.imshow(imggrey)
# plt.show()

# convert data into numpy matrix
imgmat = np.array(list(imggrey.getdata(band=0)), float)
imgmat.shape = (imggrey.size[1], imggrey.size[0])
imgmat = np.matrix(imgmat)
plt.imshow(imgmat, cmap='gray')
# plt.show()

# calculate svd of image
U, sigma, V = np.linalg.svd(imgmat)

# grab first left singular vector (U), first singular value (sigma) and first right singular vector (V)
reconstimg = np.matrix(U[:, :1]) * np.matrix(sigma[:1]) * np.matrix(V[:1, :])
plt.imshow(reconstimg, cmap='gray')
# plt.show()

# additional singular vectors improve image quality // 64 singular vectors, reconstructed img with smaller data footprint
for i in [2, 4, 8, 16, 32, 64]:
    reconstimg = np.matrix(U[:, :i]) * np.diag(sigma[:i]) * np.matrix(V[:i, :])
    plt.imshow(reconstimg, cmap='gray')
    title = "n = %s" % i
    plt.title(title)
    # plt.show()

# show number of data points between ogimg and svd64img
ogimg = imgmat.shape
# print(ogimg) # rows, columns
og_full_rep = 4032 * 3024
print(f"og image data points: {og_full_rep}")

svd64_rep = 64*4032 + 64 + 64*3024
print(f"svd64 image data points: {svd64_rep}")

percent = (svd64_rep/og_full_rep)*100
print(f"svd64 is {percent}% of the original image size")
