import scipy.io as sio
import numpy as np
from matplotlib import pyplot as plt
from scipy.misc import toimage
import time


shuffledImageEasy = sio.loadmat('shuffledImageEasy.mat')

originalMat = sio.loadmat('reconstructed.mat')
original = originalMat['reconstructed'].astype(np.float64)

RGBtilesShuffled = shuffledImageEasy['RGBtilesShuffled']
shuffledImage = shuffledImageEasy['RGBrearranged'].astype(np.float64)

# Computes the energy of single tiles, as well as the whole energy
class Energy:
	def __init__(self, data):
		self.image = data.image
		self.height = data.height
		self.width = data.width

	def getLeftRightEnergy(self, tile):
		i, j = tile
		x1 = 25 * i
		x2 = 25 * (i + 1)
		y = 25 * (j + 1) - 1
		diff = self.image[x1:x2,y,:] - self.image[x1:x2,y + 1,:]
		return np.sqrt((diff**2).mean())

	def getUpDownEnergy(self, tile):
		i, j = tile
		y1 = 25 * j
		y2 = 25 * (j + 1)
		x = 25 * (i + 1) - 1
		diff = self.image[x, y1:y2, :] - self.image[x + 1, y1:y2, :]
		return np.sqrt((diff**2).mean())

	def getEnergyAround(self, tile):
		i, j = tile
		e = np.zeros(4)
		e[0] = self.getLeftRightEnergy((i,j-1))
		e[1] = self.getLeftRightEnergy((i,j))
		e[2] = self.getUpDownEnergy((i-1,j))
		e[3] = self.getUpDownEnergy((i,j))
		return e.sum()

	def getEnergyAround2Tiles(self, t1, t2):
		return self.getEnergyAround(t1) + self.getEnergyAround(t2)

	def energy(self):
		energy = 0
		for i in range(1, self.height - 1):
			for j in range(1, self.width - 1):
				energy += self.getEnergyAround((i, j))
		return energy

	def cheatEnergy(self):                             # for testing purposes only, not used in the algorithm obviously
		return np.linalg.norm(self.image - original)


class Data:
	def __init__(self, shuffledImage, height = 18, width = 12):
		self.image = np.copy(shuffledImage)
		self.height = height
		self.width = width

	def show(self):
		toimage(self.image).show()

	def swap(self, i1, j1, i2, j2):
		t1 = np.copy(self.image[25*i1 : 25*(i1+1), 25*j1 : 25*(j1+1), :])
		t2 = self.image[25*i2 : 25*(i2+1), 25*j2 : 25*(j2+1), :]
		self.image[25*i1 : 25*(i1+1), 25*j1 : 25*(j1+1), :] = t2
		self.image[25*i2 : 25*(i2+1), 25*j2 : 25*(j2+1), :] = t1


# The logger does not affect the algorithm in any way
class Logger:
	def __init__(self, data, loggingRate):
		self.data = data
		self.rate = loggingRate
		self.pAcceptArr = np.array([])
		self.energyArr = np.array([])
		self.cheatEnergyArr = np.array([])
		self.pAcceptCurr =  0
		self.counter = 0
	def update(self, energy = 0, pAccept = 0):
		if energy > 0:
			self.energyArr = np.append(self.energyArr, energy)
			self.cheatEnergyArr = np.append(self.cheatEnergyArr, Energy(self.data).cheatEnergy())
		if pAccept > 0:
			self.pAcceptCurr += pAccept
			self.counter += 1
			if (self.counter >= self.rate):
				print("pAccept = " + str(self.pAcceptCurr / self.rate))
				self.pAcceptArr = np.append(self.pAcceptArr, self.pAcceptCurr / self.rate)
				self.pAcceptCurr = 0
				self.counter = 0
	def logs(self):
		f, axarr = plt.subplots(3)
		axarr[0].plot(self.pAcceptArr, 'b^')
		axarr[0].set_title('pAcceptArr')
		axarr[1].plot(self.energyArr, 'b^')
		axarr[1].set_title('energyArr')
		axarr[2].plot(self.cheatEnergyArr, 'r^')
		axarr[2].set_title('cheatEnergyArr')
		plt.show()

# Proposal distribution priorities tiles with high energy
class Proposal:
	# sigma = 20, t=200->5, steps=10**6, final energy value 19.6k, cheat energy 25.3k 
	# sigma = 40, t=200->5, steps=10**6, final energy value 21.6k, cheat energy 37k
	# sigma = 30, t=200->5, steps=3*10**6, final energy value 
	def __init__(self, data, priorityRecomputeRate = 10**3, sigma = 30): 
		self.sigma = sigma
		self.priorityRecomputeRate = priorityRecomputeRate
		self.data = data
		self.counter = 0
		self.height = self.data.height
		self.width = self.data.width
		self.numTiles = (self.height - 2) * (self.width - 2)
		self.indices = [(i,j) for i in range(1, self.height - 1) for j in range(1, self.width - 1)]
		self.computeEnergies()

	def computeEnergies(self):
		self.sortedTiles = map(lambda (i,j): (Energy(self.data).getEnergyAround((i,j)), (i,j)), self.indices)
		self.sortedTiles.sort(reverse=True)

	def get(self):
		self.counter += 1
		if (self.counter % self.priorityRecomputeRate == 0):
			self.counter = 0
			self.computeEnergies()
		i1 = 0
		i2 = 0
		while i1 == i2:
			i1, i2 = np.round(np.clip(np.abs(np.random.normal(0, self.sigma, 2)), 0, self.numTiles - 1)).astype(int)
		_, t1 = self.sortedTiles[i1]
		_, t2 = self.sortedTiles[i2]
		return [t1, t2]


class Anneal:
	def __init__(self, data):
		self.data = data
		self.energy = Energy(data).energy()
		self.bestEnergy = self.energy
		self.logger = Logger(data, 10**4)
	
	def step(self, temperature):
		t1, t2 = self.proposal.get()
		oldEnergy = Energy(self.data).getEnergyAround2Tiles(t1, t2)
		self.data.swap(t1[0], t1[1], t2[0], t2[1])
		newEnergy = Energy(self.data).getEnergyAround2Tiles(t1, t2)
		delta = newEnergy - oldEnergy
		pAccept = 1.0
		if (delta > 0):
			pAccept = np.exp(- delta / temperature)
			self.logger.update(pAccept = pAccept)
		if (np.random.rand() > pAccept):                  # if not accepted swap back
			self.data.swap(t1[0], t1[1], t2[0], t2[1])


	# Perform simulated annealing
	# start - start temperature
	# end - end temperature
	# numSteps - iteration count for annealing
	# loggingRate - log every so many steps
	# resetRate - if no improvement after so many steps, reset to previous best state
	#             set to bigger than numSteps to never reset
	def do(self, start, end, numSteps, loggingRate = 10**4, proposalSigma = 30):
		self.proposal = Proposal(self.data, proposalSigma)
		startExp = np.log10(start)
		endExp = np.log10(end)
		tempArray = np.logspace(startExp, endExp, numSteps)
		i = 0
		for temperature in np.nditer(tempArray):
			self.step(temperature)
			i += 1
			if (i % loggingRate == 0):
				self.energy = Energy(self.data).energy()
				print("E=" + str(self.energy) + " at temperature = " + str(temperature) + " time.clock() = " + str(time.clock()))
				self.logger.update(energy = self.energy)


d = Data(shuffledImage)
a = Anneal(d)
a.do(start = 160, end = 25, numSteps = 4.0E+6, proposalSigma = 80)  # perfect result with t=200->5, sigma 80 and numSteps 4.0E+6 (min energy hit at about 12 degrees)
# a.logger.logs() # show logs
# a.data.show()  # show final image

dOriginal = Data(original)
aOriginal = Anneal(dOriginal)



