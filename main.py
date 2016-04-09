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
from scipy import stats
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
		data1 = self.energyRow / self.energyRow.sum()
		data2 = self.energyCol / self.energyCol.sum()

		pdf1 = stats.rv_discrete(values=(np.arange(self.height - 1), data1))
		pdf2 = stats.rv_discrete(values=(np.arange(self.width - 1), data2))
	
		x1, x2 = np.clip(pdf1.rvs(size = 2), 1, self.height - 1)
		y1, y2 = np.clip(pdf2.rvs(size = 2), 1, self.width - 1)
		return [(x1, y1), (x2, y2)]

	def cheatEnergy(self):
		return np.linalg.norm(self.image - original)

class Anneal:
	def __init__(self, data):
		self.data = data
		self.energy = data.energy()
		self.bestEnergy = self.energy

		self.pAcceptArr = np.array([])
		self.acceptedArr = np.array([])
		self.rejectedArr = np.array([])
		self.tempArr = np.array([])
		self.energyArr = np.array([])
		self.cheatEnergyArr = np.array([])
		self.pAcceptCurr =  0.0
		self.rejCurr = 0.0
		self.acceptedCurr = 0.0
		self.energyCurr = 0.0


	def step(self, temperature, i):
		prop = self.data.proposal()
		x1, y1 = prop[0]
		x2, y2 = prop[1]
		self.data.swap(x1, y1, x2, y2)
		newEnergy = self.data.energy()
		delta = newEnergy - self.energy
		self.energyCurr += newEnergy
		if delta <= 0:
			self.energy = newEnergy
			self.acceptedCurr += 1
			self.pAcceptCurr += 1
		else:
			pAccept = np.exp(- delta / temperature)
			self.pAcceptCurr += pAccept
			if (np.random.rand() <= pAccept):
				self.energy = newEnergy
				self.acceptedCurr += 1
			else:
				self.data.swap(x1, y1, x2, y2)
				self.rejCurr += 1


	def do(self, startExp, endExp, numStepsExp):
		numSteps = 10**numStepsExp
		tempArray = np.logspace(startExp, endExp, numSteps)
		i = 0
		for temperature in np.nditer(tempArray):
			if (self.energy < self.bestEnergy):
				self.bestEnergy = self.energy
				print("E=" + str(self.energy) + " at t=" + str(temperature))
			i += 1
			self.step(temperature, i)
			rate = 10**3
			if (i % rate == 0):
				self.tempArr = np.append(self.tempArr, temperature)
				self.energyArr = np.append(self.energyArr, self.energyCurr / rate)
				self.pAcceptArr = np.append(self.pAcceptArr, self.pAcceptCurr / rate)
				self.acceptedArr = np.append(self.acceptedArr, self.acceptedArr / rate)
				self.rejectedArr = np.append(self.rejectedArr, self.rejCurr / rate)
				self.cheatEnergyArr = np.append(self.cheatEnergyArr, self.data.cheatEnergy())
				self.energyCurr = 0.0
				self.pAcceptCurr = 0.0
				self.acceptedCurr = 0.0
				self.rejCurr = 0.0

	def logs(self):
		f, axarr = plt.subplots(7)
		axarr[0].plot(self.tempArr, 'b^')
		axarr[0].set_title('tempArr')
		axarr[1].plot(self.energyArr, 'b^')
		axarr[1].set_title('energyArr')
		axarr[2].plot(self.rejectedArr, 'b^')
		axarr[2].set_title('rejectedArr')
		axarr[3].plot(self.pAcceptArr, 'b^')
		axarr[3].set_title('pAcceptArr')
		axarr[4].semilogy(self.tempArr, 'b^')
		axarr[4].set_title('tempArr')
		axarr[5].plot(self.energyArr, 'b^')
		axarr[5].set_title('energyArr')
		axarr[6].plot(self.cheatEnergyArr, 'r^')
		axarr[6].set_title('cheatEnergyArr')
		plt.show()



d = Data(shuffledImage)
a = Anneal(d)
# a.do(2.7, 0, 6)
# a.data.show()
dOrg = Data(original)
aOrg = Anneal(dOrg)



