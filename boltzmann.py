from numpy import *
from numba import njit
import matplotlib.pyplot as plt
from matplotlib import gridspec
import json

#interaction energy of one lattice site
@njit(cache=True)
def interaction(v):
	return (v**2)/2

#total energy of the lattice
@njit(cache=True)
def energy(L):
	U=0
	for i,j in ndenumerate(L):
			U+=interaction(L[i])
	return U

#metropolis algorithm
@njit(cache=True)
def main(size,betaJ,n,s=0.05):
	Energy=[]                 
	L=random.uniform(-5.,5.,size)
	L_init=L.copy()
	U=energy(L)
	for k in range(n):
		i=random.randint(size)
		delta = s*random.uniform(-1.,1.)
		dE = interaction(L[i]+delta) - interaction(L[i])  #change in energy
		if exp(-betaJ*dE)>random.uniform(0,1):  #acceptance condition
			L[i]=L[i] + delta
			U+=dE
		Energy.append(U)
	return Energy,L


#plotting and saving
if __name__=='__main__':

	#values of parameters
	size=100                   #lattice size
	Temp=1                   #Temperature
	n=int(2e7)                 #number of MC steps
	s=0.05
	b=10

	f = lambda v: sqrt(2/(pi*Temp))*exp((-(v**2))/(2*Temp))

	#plotting
	Energy,L=main(size,1/Temp,n,s)
	plt.figure(figsize=[10,6])
	plt.subplot(1,2,1)
	plt.plot(range(1,n+1),Energy)
	plt.subplot(1,2,2)
	plt.hist(abs(L),bins=linspace(0,5,b), density=True)
	x = linspace(0,5,1000)
	plt.plot(x,f(x))
	print(abs(L))
	plt.show()