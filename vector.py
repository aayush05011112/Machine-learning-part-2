#Step 1:Import necessary libraries
import numpy as np
import matplotlib.pyplot as plt

#step 2: Create a grid
x=np.linspace(-5,5,10)
y=np.linspace(-5,5,10)
X,Y = np.meshgrid(x,y)

#Step 3:Define the vector field
# U and V ae the components of the vector
U = -Y
V = X

#Step 4: Plot the vector field
plt.quiver(X,Y,U,V)

#Step 5:Formating the plot
plt.xlim(-5,5)
plt.ylim(-5,5)
plt.title("Circular Vector field")
plt.xlabel("X_Axis")
plt.ylabel("Y_Axis")
plt.grid()

#Step 6: Display the plot
plt.show()