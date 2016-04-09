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


shuffledImageEasy = sio.loadmat('shuffledImageEasy.mat')

originalMat = sio.loadmat('reconstructed.mat')
original = originalMat['reconstructed'].astype(np.float64)

RGBtilesShuffled = shuffledImageEasy['RGBtilesShuffled']
shuffledImage = shuffledImageEasy['RGBrearranged'].astype(np.float64)


class Data:
	def __init__(self, shuffledImage, height = 18, width = 12):
		self.image = np.copy(shuffledImage)
		self.height = height
		self.width = width
		self.energyCol = np.zeros(self.width - 1)
		self.energyRow = np.zeros(self.height - 1)

	# 0 <= j < width - 1
	def computeEnergyCol(self, j):
		diff = self.image[:, 25 * (j + 1) - 1, :] - self.image[:, 25 * (j + 1), :]
		return np.linalg.norm(diff)
		

	# 0 <= i < height - 1
	def computeEnergyRow(self, i):
		diff = self.image[25 * (i + 1) - 1, :, :] - self.image[25 * (i + 1), :, :]
		return np.linalg.norm(diff)
		

	def energy(self):
		for j in range(self.width - 1):
			self.energyCol[j] = self.computeEnergyCol(j)
		for i in range(self.height - 1):
			self.energyRow[i] = self.computeEnergyRow(i)
		return self.energyRow.sum() + self.energyCol.sum()

	def show(self):
		toimage(self.image).show()

	def swap(self, i1, j1, i2, j2):
		t1 = np.copy(self.image[25*i1 : 25*(i1+1), 25*j1 : 25*(j1+1), :])
		t2 = self.image[25*i2 : 25*(i2+1), 25*j2 : 25*(j2+1), :]
		self.image[25*i1 : 25*(i1+1), 25*j1 : 25*(j1+1), :] = t2
		self.image[25*i2 : 25*(i2+1), 25*j2 : 25*(j2+1), :] = t1

	def proposal(self):
		x1 = np.random.randint(1, self.height - 1)
		y1 = np.random.randint(1, self.width - 1)
		x2 = np.random.randint(1, self.height - 1)
		y2 = np.random.randint(1, self.width - 1)
		return [(x1, y1), (x2, y2)]

class Anneal:
	def __init__(self, data):
		self.data = data
		self.energy = data.energy()
		self.bestEnergy = self.energy

	def step(self, temperature):
		prop = self.data.proposal()
		x1, y1 = prop[0]
		x2, y2 = prop[1]
		self.data.swap(x1, y1, x2, y2)
		newEnergy = self.data.energy()
		delta = newEnergy - self.energy
		if delta <= 0:
			self.energy = newEnergy
		else:
			pAccept = np.exp(- delta / temperature)
			if (np.random.rand() <= pAccept):
				self.energy = newEnergy
			else:
				self.data.swap(x1, y1, x2, y2)


	def do(self, startExp, endExp, numStepsExp):
		numSteps = 10**numStepsExp
		tempArray = np.logspace(startExp, endExp, numSteps)
		i = 0
		for temperature in np.nditer(tempArray):
			if (self.energy < self.bestEnergy):
				self.bestEnergy = self.energy
				print("E=" + str(self.energy) + " at t=" + str(temperature))
			self.step(temperature)
			i += 1

d = Data(shuffledImage)
a = Anneal(d)
a.do(startExp = 3, endExp = 0, numStepsExp = 5) # I tried all kinds of cooling schemes (while also logging all possible params) 
a.data.show()
dOriginal = Data(original)
aOriginal = Anneal(dOriginal)



