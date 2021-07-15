import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
import pydicom

img1 =  pydicom.dcmread("Thoracic CT 1.dcm")# read dcm and get image of Thoracic CT 1.dcm

img2 =  pydicom.dcmread("Thoracic CT 2.dcm")# read dcm and get image of Thoracic CT 2.dcm

bit_depth = 12 #chon max esh 4089 e yani img 12 bitie
L_1 = 2**12 -1 #calcualate L-1


mask = np.fromfunction(lambda i, j: (-1)**(i+j), img1.pixel_array.shape, dtype=int) #in the report

img1_prime = img1.pixel_array * mask
img2_prime = img2.pixel_array * mask

fig, axs = plt.subplots(2, 2 , figsize=(10,10))
plt.suptitle('Choose the right option')

axs[0, 0].set_title('boundary of img1 is (correct)')
axs[0, 0].imshow(img1.pixel_array, cmap='gray', vmin=0, vmax=L_1)
axs[0, 0].axis(False)

axs[1, 0].set_title('boundary of img1 is (stretched)')
axs[1, 0].imshow(img1.pixel_array, cmap='gray', vmin=-L_1, vmax=L_1)
axs[1, 0].axis(False)


axs[0, 1].set_title('boundary of img1_prime is (clipped)')
axs[0, 1].imshow(img1_prime, cmap='gray', vmin=0, vmax=L_1)
axs[0, 1].axis(False)

axs[1, 1].set_title('boundary of img2_prime is (correct)')
axs[1, 1].imshow(img2_prime, cmap='gray', vmin=-L_1, vmax=L_1)
axs[1, 1].axis(False)

plt.tight_layout()

IMG1 = np.fft.fft2(img1_prime)
IMG1_abs = np.abs(IMG1) # IMG1_abs
IMG1_abs_log = np.log(IMG1_abs+1)
IMG1_angle = np.angle(IMG1) # IMG1_angle

IMG2 = np.fft.fft2(img2_prime)
IMG2_abs = np.abs(IMG2)# IMG2_abs
IMG2_abs_log = np.log(IMG2_abs+1)
IMG2_angle = np.angle(IMG2) # IMG2_angle


maxI = img1.pixel_array.size*L_1 #in the report
maxI_log = np.log(maxI+1)


plt.figure(figsize=(10,10)) #in the report
plt.suptitle('Images in frequency domain')

plt.subplot(231)
plt.title(r'${\left| {{IMG1}} \right|}$')
plt.imshow(IMG1_abs, cmap='gray', vmin=0, vmax=maxI)
plt.axis(False)

plt.subplot(232)
plt.title(r'$\log \left( {\left| {{IMG1}} \right| + 1} \right)$')
plt.imshow(IMG1_abs_log, cmap='gray', vmin=0, vmax=maxI_log)
plt.axis(False)


plt.subplot(233)
plt.title(r'$\angle$IMG1: $\left( { - \pi ,\pi } \right]$')
plt.imshow(IMG1_angle, cmap='gray', vmin=-np.pi, vmax=np.pi)
plt.axis(False)

plt.subplot(234)
plt.title(r'${\left| {{IMG2}} \right|}$')
plt.imshow(IMG2_abs, cmap='gray', vmin=0, vmax=maxI)
plt.axis(False)

plt.subplot(235)
plt.title(r'$\log \left( {\left| {{IMG2}} \right| + 1} \right)$')
plt.imshow(IMG2_abs_log, cmap='gray', vmin=0, vmax=maxI_log)
plt.axis(False)

plt.subplot(236)
plt.title(r'$\angle$IMG2: $\left( { - \pi ,\pi } \right]$')
plt.imshow(IMG2_angle, cmap='gray', vmin=-np.pi, vmax=np.pi)
plt.axis(False)

IMG1_c =  np.conj(IMG1)
IMG2_c = -np.conj(IMG2) # in the report

img1_c_shifted = np.fft.ifft2(IMG1_c).real
img2_c_shifted = np.fft.ifft2(IMG2_c).real


img1_c = np.clip(img1_c_shifted * mask, 0, L_1) 
img2_c = np.clip(img2_c_shifted * mask, -L_1, 0) + L_1

plt.figure(figsize=(10,10)) #?

plt.subplot(221)
plt.title('img1')
plt.imshow(img1.pixel_array, cmap='gray', vmin=0, vmax=L_1)
plt.axis(False)

plt.subplot(222)
plt.title('img2')
plt.imshow(img2.pixel_array, cmap='gray', vmin=0, vmax=L_1)
plt.axis(False)

plt.subplot(223)
plt.title('Symmetric to x and y') 
plt.imshow(img1_c, cmap='gray', vmin=0, vmax=L_1)
plt.axis(False)

plt.subplot(224)
plt.title('Complement Symmetric to x and y')
plt.imshow(img2_c, cmap='gray', vmin=0, vmax=L_1)
plt.axis(False)

plt.show()