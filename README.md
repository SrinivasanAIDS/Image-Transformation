### Ex No: 5
### Date:

# <p align="center"> Image-Transformation </p>
## Aim
To perform image transformation such as Translation, Scaling, Shearing, Reflection, Rotation and Cropping using OpenCV and Python.

## Software Required:
Anaconda - Python 3.7

## Algorithm:
### Step 1:
Import the necessary libraries and read the original image and save it a image variable.

### Step 2:
Translate the image 
```python
Translation_matrix=np.float32([[1,0,120],[0,1,120],[0,0,1]])
Translated_image=cv2.warpPerspective(org_img,Translation_matrix,(col,row))
```
### Step 3:
Scale the image 
```python
Scaling_Matrix=np.float32([[1.2,0,0],[0,1.2,0],[0,0,1]]) 
Scaled_image=cv2.warpPerspective(org_img,Scaling_Matrix,(col,row))
```
### Step 4:
Shear the image
```python
Shearing_matrix=np.float32([[1,0.2,0],[0.2,1,0],[0,0,1]]) 
Sheared_image=cv2.warpPerspective(org_img,Shearing_matrix,(col2,int(row1.5)))
```
### Step 5:
Reflection of image can be achieved through the code 
```python
Reflection_matrix_row=np.float32([[1,0,0],[0,-1,row],[0,0,1]]) 
Reflected_image_row=cv2.warpPerspective(org_img,Reflection_matrix_row,(col,int(row)))
```
### Step 6:
Rotate the image 
```python
Rotation_angle=np.radians(10) 
Rotation_matrix=np.float32([[np.cos(Rotation_angle),-np.sin(Rotation_angle),0],[np.sin(Rotation_angle),np.cos(Rotation_angle),0],[0,0,1]])
Rotated_image=cv2.warpPerspective(org_img,Rotation_matrix,(col,(row)))
```
### Step 7:
Crop the image 
```python
cropped_image=org_img[10:350,320:560]
```
### Step 8:
Display all the Transformed images.

## Program:
```
Developed By: Srinivasan S
Register Number: 212220230048
```
```python
i)Image Translation
import numpy as np
import cv2
import matplotlib.pyplot as plt

# Read the input image
input_image = cv2.imread("Picture.jpg")
# Convert from BGR to RGB so we can plot using matplotlib
input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)
# Disable x and y axis
plt.axis('off')
# Show the image
plt.imshow(input_image)
plt.show()
# Get the image shape
rows, cols, dim = input_image.shape
# Transformation matrix for translation
M = np.float32([[1, 0, 100],
                [0, 1, 200],
                [0, 0, 1]])
# Apply a perspective transformation to the image
translated_image = cv2.warpPerspective(input_image, M, (cols, rows))

# Disable x and y axis
plt.axis('off')
# Show the resulting image
plt.imshow(translated_image)
plt.show()

ii) Image Scaling
import numpy as np
import cv2
import matplotlib.pyplot as plt

# Read the input image
input_image = cv2.imread("Picture.jpg")
# Convert from BGR to RGB so we can plot using matplotlib
input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)
# Disable x and y axis
plt.axis('off')
# Show the image
plt.imshow(input_image)
plt.show()
# Get the image shape
rows, cols, dim = input_image.shape
# Transformation matrix for scaling
M = np.float32([[1.5, 0, 0],
                [0, 1.8, 0],
                [0, 0, 1]])
# Apply a perspective transformation to the image
scaled_image = cv2.warpPerspective(input_image, M, (cols*2, rows*2))

# Disable x and y axis
plt.axis('off')
# Show the resulting image
plt.imshow(scaled_image)
plt.show()

iii)Image shearing
import numpy as np
import cv2
import matplotlib.pyplot as plt

# Read the input image
input_image = cv2.imread("Picture.jpg")
# Convert from BGR to RGB so we can plot using matplotlib
input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)
# Disable x and y axis
plt.axis('off')
# Show the image
plt.imshow(input_image)
plt.show()
# Get the image shape
rows, cols, dim = input_image.shape
# Transformation matrix for shearing
M = np.float32([[1, 0.15, 0],
                [0.15, 1, 0],
                [0, 0, 1]])
# Apply a perspective transformation to the image
sheared_image = cv2.warpPerspective(input_image, M,(cols*2,int(rows*1.5)))

# Disable x and y axis
plt.axis('off')
# Show the resulting image
plt.imshow(sheared_image)
plt.show()

iv)Image Reflection 
a) rows
import numpy as np
import cv2
import matplotlib.pyplot as plt

# Read the input image
input_image = cv2.imread("Picture.jpg")
# Convert from BGR to RGB so we can plot using matplotlib
input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)
# Disable x and y axis
plt.axis('off')
# Show the image
plt.imshow(input_image)
plt.show()
# Get the image shape
rows, cols, dim = input_image.shape
# Transformation matrix for reflection
M = np.float32([[1, 0, 0],
                [0, -1, rows],
                [0, 0, 1]])
# Apply a perspective transformation to the image
reflected_rowsimage = cv2.warpPerspective(input_image, M,(cols,int(rows)))

# Disable x and y axis
plt.axis('off')
# Show the resulting image
plt.imshow(reflected_rowsimage)
plt.show()

b) Columns
import numpy as np
import cv2
import matplotlib.pyplot as plt

# Read the input image
input_image = cv2.imread("Picture.jpg")
# Convert from BGR to RGB so we can plot using matplotlib
input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)
# Disable x and y axis
plt.axis('off')
# Show the image
plt.imshow(input_image)
plt.show()
# Get the image shape
rows, cols, dim = input_image.shape
# Transformation matrix for reflection
M = np.float32([[-1, 0, cols],
                [0, 1, 0],
                [0, 0, 1]])
# Apply a perspective transformation to the image
reflected_colsimage = cv2.warpPerspective(input_image, M,(cols,int(rows)))

# Disable x and y axis
plt.axis('off')
# Show the resulting image
plt.imshow(reflected_colsimage)
plt.show()

v)Image Rotation
import numpy as np
import cv2
import matplotlib.pyplot as plt

# Read the input image
input_image = cv2.imread("Picture.jpg")
# Convert from BGR to RGB so we can plot using matplotlib
input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)
# Disable x and y axis
plt.axis('off')
# Show the image
plt.imshow(input_image)
plt.show()
# Get the image shape
rows, cols, dim = input_image.shape
# Transformation matrix for rotation
Rotation_angle=np.radians(10)
M = np.float32([[np.cos(Rotation_angle),-np.sin(Rotation_angle),0],
                [np.sin(Rotation_angle),np.cos(Rotation_angle),0],
                [0,0,1]])
# Apply a perspective transformation to the image
rotated_image = cv2.warpPerspective(input_image, M,(cols,(rows)))

# Disable x and y axis
plt.axis('off')
# Show the resulting image
plt.imshow(rotated_image)
plt.show()

vi)Image Cropping
import numpy as np
import cv2
import matplotlib.pyplot as plt

# Read the input image
input_image = cv2.imread("Picture.jpg")
# Convert from BGR to RGB so we can plot using matplotlib
input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)
# Disable x and y axis
plt.axis('off')
# Show the image
plt.imshow(input_image)
plt.show()
# Get the image shape
rows, cols, dim = input_image.shape
# Apply a perspective transformation to the image
cropped_image = input_image[10:200,20:360]

# Disable x and y axis
plt.axis('off')
# Show the resulting image
plt.imshow(cropped_image)
plt.show()

```
## Output:
### i)Image Translation
![Screenshot 2022-04-29 200905](https://user-images.githubusercontent.com/103049243/165966897-5d821054-f8b2-45e2-8a7d-4f328a818d89.png)

### ii) Image Scaling
![Screenshot 2022-04-29 201001](https://user-images.githubusercontent.com/103049243/165967031-f4326ab3-b742-4d3e-9524-d12857a69332.png)

### iii)Image shearing
![Screenshot 2022-04-29 201038](https://user-images.githubusercontent.com/103049243/165967155-5ede7307-ebad-4adb-a9f6-3a529efd82b0.png)

### iv)Image Reflection
# a) Rows:
![Screenshot 2022-04-29 201246](https://user-images.githubusercontent.com/103049243/165967521-78129de5-5c4f-494f-b3c9-5697ab464e65.png)

# b) Columns:
![Screenshot 2022-04-29 201320](https://user-images.githubusercontent.com/103049243/165967696-3d39fad9-b26d-455b-a81d-2f11d34ac2d8.png)

### v)Image Rotation
![Screenshot 2022-04-29 201419](https://user-images.githubusercontent.com/103049243/165967781-71432c10-8f1b-460f-a728-5644ce624141.png)

### vi)Image Cropping
![Screenshot 2022-04-29 201452](https://user-images.githubusercontent.com/103049243/165967908-07714a5f-89ce-4266-b1bb-be806b95deaf.png)

## Result: 

Thus the different image transformations such as Translation, Scaling, Shearing, Reflection, Rotation and Cropping are done using OpenCV and python programming.
