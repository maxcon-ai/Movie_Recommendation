import matplotlib.pyplot as plt
r= .971
# Data from the table
VCE_IB0 = [0,1, 2, 3, 5, 8, 9, 10, 12]
IC_IB0 = [0,0, 0, 0, 0, 0, 0, 0, 0]

VCE_IB1 = [0,0.13, 0.59, 1.59, 3.57, 6.58,7.55, 8.61, 10.55,]
IC_IB1 = [0,.84, 1.46, 1.46, 1.47,  1.51, 1.52, 1.53,1.53]

VCE_IB2 = [0,.1,0.135,.225,2.17,5.08,6.01,7.00,9.00,]
IC_IB2 = [0,0.9,1.92,2.89,2.96,3.01,3.03,3.04,3.06]

VCE_IB3 = [0,.09,.115,.14,.85,3.75,4.72,5.71,7.71]
IC_IB3 = [0,0.95,1.90,2.89,4.19,4.29,4.38,4.39,4.42]
print(IC_IB3)
# Plot the data
plt.figure(figsize=(10, 6))

plt.plot(VCE_IB0, IC_IB0, marker='^', label='$I_B = 0$ mA')
plt.plot(VCE_IB1, IC_IB1, marker='o', label='$I_B = 10$ mA')
plt.plot(VCE_IB2, IC_IB2, marker='s', label='$I_B = 20$ mA')
plt.plot(VCE_IB3, IC_IB3, marker='^', label='$I_B = 30$ mA')

# Add labels, title, and legend
plt.xlabel('$V_{CE}$ (V)', fontsize=12)
plt.ylabel('$I_C$ (mA)', fontsize=12)
plt.title('Output Characteristics of a Transistor', fontsize=14)
plt.legend()
plt.grid(True)

# Show the plot
plt.show()
