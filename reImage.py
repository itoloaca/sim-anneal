import scipy.io as sio
import numpy as np
import pprint as pp
from matplotlib import pyplot as plt
from collections import namedtuple
import scipy.misc as misc
from colormath.color_objects import LabColor, sRGBColor
from colormath.color_conversions import convert_color

shuffledImageEasy = sio.loadmat('shuffledImageEasy.mat')

#original = misc.imread('actualPic.jpg').astype(np.float64)
originalMat = sio.loadmat('reconstructed.mat')
original = originalMat['reconstructed'].astype(np.float64)

RGBtilesShuffled = shuffledImageEasy['RGBtilesShuffled']
RGBrearranged = shuffledImageEasy['RGBrearranged'].astype(np.float64)

shuffledTiles = RGBtilesShuffled.reshape(25, 25, 3, 18, 12).astype(np.float64)

height = 18
width = 12

originalTiles = np.copy(shuffledTiles)
originalTiles.fill(0)

for i in range(0,height):
	for j in range(0,width):
		originalTiles[:,:,:,i,j] = original[25*i:25*(i+1),25*j:25*(j+1),:]	


Tile = namedtuple("Tile", ['x', 'y'])

permutation = {}
for i in range(height):
	for j in range(width):
		permutation[Tile(i,j)] = Tile(i,j)

class Tiles:
	def __init__(self, tiles, permutation, height, width):
		self.tiles = tiles
		self.permutation = np.copy(permutation)
		self.height = height
		self.width = width
		self.tiles = tiles

	def getPermutedTiles(self, tiles):
		permutedTiles = np.copy(tiles)
		for key, val in permutation.iteritems():
			permutedTiles[:, :, :, key.x, key.y] = np.copy(tiles[:, :, :, val.x, val.y])
		return permutedTiles
	def getTileColors(self, tile):
		tileColors = permutation[tile]
		return self.tiles[:, :, :, tileColors.x, tileColors.y];
	def asImage(self):
		permutedTiles = self.getPermutedTiles(self.tiles)
		col = []
		for i in range(self.height):
			row = []
			for j in range(self.width):
				if row != []:
					row = np.hstack((row, permutedTiles[:, :, :, i, j]))
				else: 
					row = permutedTiles[:, :, :, i, j]
			if col != []:
				col = np.vstack((col, row))
			else: 
				col = row
		return col.astype(np.uint8)
	def display(self):
		plt.imshow(self.asImage())
		plt.show()
	def swap(self, tile1, tile2):
		permutation[tile1], permutation[tile2] = permutation[tile2], permutation[tile1]
	def getEnergyVMargin(self, tile1):
		tile2 = Tile(tile1.x, tile1.y + 1)
		tile1Colors = self.getTileColors(tile1)
		tile2Colors = self.getTileColors(tile2)
		return np.linalg.norm(tile1Colors[:,24,:] - tile2Colors[:,0,:])**2
	def getEnergyHMargin(self, tile1):
		tile2 = Tile(tile1.x + 1, tile1.y)
		tile1Colors = self.getTileColors(tile1)
		tile2Colors = self.getTileColors(tile2)
		return np.linalg.norm(tile1Colors[24,:,:] - tile2Colors[0,:,:])**2
	def cheatEnergy(self):
		energy = 0
		for i in range(self.height):
			for j in range(self.width):
				tileColors = permutation[Tile(i,j)]
				energy += np.sum(np.abs(originalTiles[:,:,:,i,j] - self.tiles[:,:,:,tileColors.x,tileColors.y]))
		return energy
	def cheatDeltaEnergy(self, tile1, tile2):
		if (tile1 == tile2):
			return 0
		tiles = self.tiles
		energyBefore = 0
		tile1Colors = permutation[tile1]
		energyBefore += np.sum(np.abs(originalTiles[:,:,:,tile1.x,tile1.y] - tiles[:,:,:,tile1Colors.x,tile1Colors.y]))
		tile2Colors = permutation[tile2]
		energyBefore += np.sum(np.abs(originalTiles[:,:,:,tile2.x, tile2.y] - tiles[:,:,:,tile2Colors.x,tile2Colors.y]))

		tiles.swap(tile1, tile2)
		energyAfter = 0
		tile1Colors = permutation[tile1]
		energyAfter += np.sum(np.abs(originalTiles[:,:,:,tile1.x,tile1.y] - tiles[:,:,:,tile1Colors.x,tile1Colors.y]))
		tile2Colors = permutation[tile2]
		energyAfter += np.sum(np.abs(originalTiles[:,:,:,tile2.x, tile2.y] - tiles[:,:,:,tile2Colors.x,tile2Colors.y]))
		tiles.swap(tile1, tile2)

		return energyAfter - energyBefore

	def energy(self):
		# return self.cheatEnergy()
		energy = 0
		for i in range(self.height):
			for j in range(self.width - 1):
				energy += self.getEnergyVMargin(Tile(i,j))
		for j in range(self.width):
			for i in range(self.height - 1):
				energy += self.getEnergyHMargin(Tile(i,j)) 
		return energy
	
	def getEnergyAround(self, tile):
		energy = np.zeros(4)
		energy[0] = self.getEnergyVMargin(Tile(tile.x, tile.y - 1))
		energy[1] = self.getEnergyVMargin(tile)
		energy[2] = self.getEnergyHMargin(Tile(tile.x - 1, tile.y))
		energy[3] = self.getEnergyHMargin(tile)
		return sum(energy)
	def energyAlt(self):
		energy = 0;
		for i in range(1,self.height - 1):
			for j in range(1,self.width - 1):
				energy += self.getEnergyAround(Tile(i,j))

		for j in range(0,self.width - 1):
			energy += 2 * self.getEnergyVMargin(Tile(0,j))

		for j in range(0,self.width - 1):
			energy += 2 * self.getEnergyVMargin(Tile(self.height-1,j))

		for i in range(0,self.height-1):
			energy += 2 * self.getEnergyHMargin(Tile(i,0))

		for i in range(0,self.height - 1):
			energy += 2 * self.getEnergyHMargin(Tile(i,self.width - 1))

		for j in range(1,self.width - 1):
			energy += self.getEnergyHMargin(Tile(0,j))

		for j in range(1,self.width - 1):
			energy += self.getEnergyHMargin(Tile(self.height-2,j))

		for i in range(1,self.height-1):
			energy += self.getEnergyVMargin(Tile(i,0))

		for i in range(1,self.height-1):
			energy += self.getEnergyVMargin(Tile(i,self.width - 2))

		return energy / 2.0
	def deltaEnergy(self, tile1, tile2):
		# return self.cheatDeltaEnergy(tile1, tile2)
		if (tile1 == tile2):
			return 0
		if (tile1.x > tile2.x or tile1.y > tile2.y):
			tile1, tile2 = tile2, tile1			
		inCol =  (tile1.y == tile2.y and tile1.x + 1 == tile2.x)
		inRow =  (tile1.x == tile2.x and tile1.y + 1 == tile2.y)
		
		energyBefore = np.zeros(2)

		energyBefore[0] = self.getEnergyAround(tile1)
		energyBefore[1] = self.getEnergyAround(tile2)

		self.swap(tile1, tile2)	# Swap for computation ease
		energyAfter = np.zeros(2)
		energyAfter[0] = self.getEnergyAround(tile1)
		energyAfter[1] = self.getEnergyAround(tile2)

		vMarginTile2 = 0
		hMarginTile2 = 0
		# for inRow / inCol
		if inRow:
			vMarginTileRow = self.getEnergyVMargin(tile1) 
		if inCol:
			hMarginTileCol = self.getEnergyHMargin(tile1)
		self.swap(tile1, tile2)	# Swap back
		
		sumEnergyBefore = energyBefore.sum()
		sumEnergyAfter = energyAfter.sum()

		if inRow:
			sumEnergyBefore -= self.getEnergyVMargin(tile1)
			sumEnergyAfter -= vMarginTileRow
		if inCol:
			sumEnergyBefore -= self.getEnergyHMargin(tile1)
			sumEnergyAfter -= hMarginTileCol
		
		return sumEnergyAfter - sumEnergyBefore

	def permutationProposalVariant1(self):
		x1 = np.random.randint(1, self.height - 1)
		y1 = np.random.randint(1, self.width - 1)
		x2 = np.random.randint(1, self.height - 1)
		y2 = np.random.randint(1, self.width - 1)
		return [Tile(x1, y1), Tile(x2, y2)]
	def permutationProposal(self):
		return self.permutationProposalVariant1()
		N = 10
		x = np.random.randint(1, self.height - 1, N)
		y = np.random.randint(1, self.width - 1, N)
		energy = np.zeros(N)
		for i in range(N):
			energy[i] = self.getEnergyAround(Tile(x[i], y[i]))
		ind1 = energy.argmax()
		energy[ind1] = -1
		ind2 = energy.argmax()
		return [Tile(x[ind1], y[ind1]), Tile(x[ind2], y[ind2])]


class Annealing:
	def __init__(self, tiles):
		self.tiles = tiles
		self.energy = tiles.energy()
		self.bestEnergy = tiles.energy()
		self.bestPermutation = tiles.permutation

		self.loggingRate = 10**3

		self.acceptedDirectly = np.array([])
		self.acceptedEventually = np.array([])
		self.rejected = np.array([])
		self.acceptProbabilities = np.array([])

		self.tempVals = np.array([])
		self.energyVals = np.array([])

		self.currentDirectly = 0
		self.currentEventually = 0
		self.currentProbability = 0
		self.currentRejected = 0
		self.currentTempVal = 0
		self.currentEnergyVal = 0

		self.cheatEnergy = np.array([])

	def display(self):
		self.tiles.display()
            
	def plotLogs(self):
		f, axarr = plt.subplots(7)
		axarr[0].plot(self.acceptedDirectly, 'b^')
		axarr[0].set_title('acceptedDirectly')
		axarr[1].plot(self.acceptedEventually, 'b^')
		axarr[1].set_title('acceptedEventually')
		axarr[2].plot(self.rejected, 'b^')
		axarr[2].set_title('rejected')
		axarr[3].plot(self.acceptProbabilities, 'b^')
		axarr[3].set_title('acceptProbabilities')
		axarr[4].semilogy(self.tempVals, 'b^')
		axarr[4].set_title('tempVals')
		axarr[5].plot(self.energyVals, 'b^')
		axarr[5].set_title('energyVals')
		axarr[6].plot(self.cheatEnergy, 'r^')
		axarr[6].set_title('cheat energy')
		plt.show()

	def step(self, temperature):
		[t1, t2] = tiles.permutationProposal()
		delta = tiles.deltaEnergy(t1, t2)
		if (delta <= 0):
			self.tiles.swap(t1, t2)
			self.energy += delta
			self.currentDirectly += 1
		else:
			pAccept = np.exp(- delta / temperature)
			self.currentProbability += pAccept
			if (np.random.rand() <= pAccept):
				self.tiles.swap(t1, t2)
				self.energy += delta
				self.currentEventually += 1
			else:
				self.currentRejected += 1

	def annealing(self, startExp = 6, endExp = -2, numSteps = 10**6):
		temperature = np.logspace(startExp, endExp, numSteps)
		self.bestT = temperature[0]
		for indexTuple, tempVal in np.ndenumerate(temperature):
			index = indexTuple[0]
			self.step(tempVal)
			if (self.energy < self.bestEnergy):
				self.bestPermutation = self.tiles.permutation
				print 'pAccept approx = ' + str(100 * np.exp((self.energy - self.bestEnergy)/tempVal)), ('(1 - newTemp)/ oldTemp = ' + str(1 - tempVal/self.bestT))			
				self.bestT = tempVal
				self.bestEnergy = self.energy
				print('Energy = ' + str(self.bestEnergy) + ' at t = ' + str(tempVal) + ' %done = ' + str(100*index/float(temperature.size)) + '\n')
			self.currentTempVal += tempVal
			self.currentEnergyVal += self.energy
			if index % 10**5 == 0:
				print('Index = ' + str(index) + ' # sanity check (should be zero) = ' + str(self.energy - self.tiles.energy()) + ' Energy = ' + str(self.energy) + ' Temperature ' + str(tempVal))
			if index % self.loggingRate == 0 and index > 0:
				self.acceptProbabilities = np.append(self.acceptProbabilities, self.currentProbability/(self.currentEventually + self.currentRejected))
				self.currentProbability = 0
				self.acceptedDirectly = np.append(self.acceptedDirectly, self.currentDirectly)
				self.currentDirectly = 0
				self.acceptedEventually = np.append(self.acceptedEventually, self.currentEventually)
				self.currentEventually = 0
				self.rejected = np.append(self.rejected, self.currentRejected)
				self.currentRejected = 0
				self.tempVals = np.append(self.tempVals, self.currentTempVal / self.loggingRate)
				self.currentTempVal = 0
				self.energyVals = np.append(self.energyVals, self.currentEnergyVal / self.loggingRate)
				self.currentEnergyVal = 0
				self.cheatEnergy = np.append(self.cheatEnergy, self.tiles.cheatEnergy())


## Setup

tiles = Tiles(shuffledTiles, permutation, height, width)
anneal = Annealing(tiles)

#anneal.tiles.display()
# print(anneal.bestEnergy)
# e0 = tiles.energy()
# tile2 = Tile(1,2)
# d = tiles.deltaEnergy(Tile(1,1), tile2)
# tiles.swap(Tile(1,1), tile2)
# e1 = tiles.energy()
# print(d)
# print(e1-e0  - d)
# print(e0)