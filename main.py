import scipy.io as sio
import numpy as np
import pprint as pp
from matplotlib import pyplot as plt
from collections import namedtuple
import scipy.misc as misc
from colormath.color_objects import LabColor, sRGBColor
from colormath.color_conversions import convert_color
from scipy.misc import toimage
from scipy import ndimage
import cv2
import time


shuffledImageEasy = sio.loadmat('shuffledImageEasy.mat')

originalMat = sio.loadmat('reconstructed.mat')
original = originalMat['reconstructed'].astype(np.float64)

RGBtilesShuffled = shuffledImageEasy['RGBtilesShuffled']
shuffledImage = shuffledImageEasy['RGBrearranged'].astype(np.float64)

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
			self.cheatEnergyArr = np.append(self.energyArr, Energy(self.data).cheatEnergy())
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

class Anneal:
	def __init__(self, data):
		self.data = data
		self.energy = Energy(data).energy()
		self.bestEnergy = self.energy
		self.logger = Logger(data, 10**4)
		self.imageBackup = np.copy(self.data.image)

	def proposal(self):
		height = self.data.height
		width = self.data.width
		x1 = np.random.randint(1, height - 1)
		y1 = np.random.randint(1, width - 1)
		x2 = np.random.randint(1, height - 1)
		y2 = np.random.randint(1, width - 1)
		return [(x1, y1), (x2, y2)]

	def step(self, temperature):
		t1, t2 = self.proposal()
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

	def do(self, start, end, numSteps, loggingRate = 10**4, resetRate = 3*10**4):
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
			if (i % resetRate == 0):
				# Reset condition
				if (self.energy > self.bestEnergy):
					print("WARNING: RESET to energy = " + str(self.bestEnergy) + " (consider lowering initial temperature) ")
					self.energy = self.bestEnergy
					self.data.image = np.copy(self.imageBackup)
				else:
					print("NEW BEST ENERGY = " + str(self.energy))
					self.bestEnergy = self.energy
					self.imageBackup = np.copy(self.data.image)


d = Data(shuffledImage)
a = Anneal(d)
a.do(start = 200, end = 5, numSteps = 3.86E+6) # about 15 minutes of runtime
dOriginal = Data(original)
aOriginal = Anneal(dOriginal)



