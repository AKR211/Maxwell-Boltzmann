from numpy import *
from numba import njit
import matplotlib.pyplot as plt
from matplotlib import gridspec
import json

#interaction energy of one lattice site
@njit(cache=True)
def interaction(v1,v2):
	return ((v1**2)+(v2**2))/2

#total energy of the lattice
@njit(cache=True)
def energy(L1,L2):
	U=0
	for i,j in ndenumerate(L1):
			U+=interaction(L1[i],L2[i])
	return U

#metropolis algorithm
@njit(cache=True)
def main(size,betaJ,n,s=0.05):
	Energy=[]                 
	L1=random.uniform(-5.,5.,size)
	L2=random.uniform(-5.,5.,size)
	U=energy(L1,L2)
	for k in range(n):
		i=random.randint(size)
		delta1 = sqrt(s/2)*random.uniform(-1.,1.)
		delta2 = sqrt(s/2)*random.uniform(-1.,1.)
		dE = interaction(L1[i]+delta1,L2[i]+delta2) - interaction(L1[i],L2[i])  #change in energy
		if exp(-betaJ*dE)>random.uniform(0,1):  #acceptance condition
			L1[i]=L1[i] + delta1
			L2[i]=L2[i] + delta2
			U+=dE
		Energy.append(U)
	return Energy,sqrt((L1**2)+(L2**2))


#plotting and saving
if __name__=='__main__':

	#values of parameters
	size=400                   #lattice size
	Temp=1                   #Temperature
	n=int(3e7)                 #number of MC steps
	s=0.05
	b=10

	f = lambda v: (Temp)*v*exp((-(v**2))/(2*Temp))

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