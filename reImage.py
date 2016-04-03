import scipy.io as sio
import numpy as np
import pprint as pp
from matplotlib import pyplot as plt
from collections import namedtuple

shuffledImageEasy = sio.loadmat('shuffledImageEasy.mat')

RGBtilesShuffled = shuffledImageEasy['RGBtilesShuffled']
RGBrearranged = shuffledImageEasy['RGBrearranged']

shuffledTiles = RGBtilesShuffled.reshape(25, 25, 3, 18, 12).astype(np.int32)

height = 18
width = 12

Tile = namedtuple("Tile", ['x', 'y'])

permutation = {}
for i in range(height):
	for j in range(width):
		permutation[Tile(i,j)] = Tile(i,j)

class Tiles:
	def __init__(self, tiles, permutation, height, width):
		self.tiles = tiles
		self.permutation = permutation
		self.height = height
		self.width = width
	def getPermutedTiles(self, tiles):
		permutedTiles = np.copy(tiles)
		for key, val in permutation.iteritems():
			permutedTiles[:, :, :, key.x, key.y] = np.copy(self.tiles[:, :, :, val.x, val.y])
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
	def getEnergyVMargin(self, tile1, tile2):
		return np.linalg.norm(tile1[:,24,:] - tile2[:,0,:])**2
	def getEnergyHMargin(self, tile1, tile2):
		return np.linalg.norm(tile1[24,:,:] - tile2[0,:,:])**2
	def energy(self):
		currTiles = self.getPermutedTiles(self.tiles)
		energy = 0
		for i in range(self.height):
			for j in range(self.width - 1):
				energy += self.getEnergyVMargin(currTiles[:,:,:,i,j], currTiles[:,:,:,i,j+1])
		for j in range(self.width):
			for i in range(self.height - 1):
				energy += self.getEnergyHMargin(currTiles[:,:,:,i,j], currTiles[:,:,:,i+1,j]) 
		return energy
	
	def getEnergyAround(self, tile):
		tileColors = self.getTileColors(tile)
		tileLeft  = self.getTileColors(Tile(tile.x, tile.y - 1))
		tileRight  = self.getTileColors(Tile(tile.x, tile.y + 1))
		tileAbove   = self.getTileColors(Tile(tile.x - 1, tile.y))
		tileBelow  = self.getTileColors(Tile(tile.x + 1, tile.y))
		energy = np.zeros(4)
		energy[0] = self.getEnergyVMargin(tileLeft, tileColors)
		energy[1] = self.getEnergyVMargin(tileColors, tileRight)
		energy[2] = self.getEnergyHMargin(tileAbove, tileColors)
		energy[3] = self.getEnergyHMargin(tileColors, tileBelow)
		return sum(energy)

	def deltaEnergy(self, tile1, tile2):
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
		self.swap(tile1, tile2)	# Swap back
		
		sumEnergyBefore = energyBefore.sum()
		sumEnergyAfter = energyAfter.sum()

		if inRow:
			tile1Colors = self.getTileColors(tile1)
			tile2Colors = self.getTileColors(tile2)
			sumEnergyBefore -= self.getEnergyVMargin(tile1Colors, tile2Colors)
			sumEnergyAfter -= self.getEnergyVMargin(tile2Colors, tile1Colors) 
		if inCol:
			tile1Colors = self.getTileColors(tile1)
			tile2Colors = self.getTileColors(tile2)
			sumEnergyBefore -= self.getEnergyHMargin(tile1Colors, tile2Colors)
			sumEnergyAfter -= self.getEnergyHMargin(tile2Colors, tile1Colors)
		
		return sumEnergyAfter - sumEnergyBefore

	def permutationProposal(self):
		x1 = np.random.randint(1, self.height - 1)
		y1 = np.random.randint(1, self.width - 1)
		x2 = np.random.randint(1, self.height - 1)
		y2 = np.random.randint(1, self.width - 1)
		return [Tile(x1, y1), Tile(x2, y2)]

class Annealing:
	def __init__(self, tiles):
		self.tiles = tiles
		self.energy = tiles.energy()
		self.bestEnergy = tiles.energy()
		self.bestPermutation = tiles.permutation
	def step(self, temperature):
		[t1, t2] = tiles.permutationProposal()
		delta = tiles.deltaEnergy(t1, t2)
		if (delta <= 0):
			self.tiles.swap(t1, t2)
			self.energy += delta
		else:
			pAccept = np.exp(- delta / temperature)
			if (np.random.rand() <= pAccept):
				self.tiles.swap(t1, t2)
				self.energy += delta

	def annealing(self, startExp = 8, endExp = -1, numSteps = 3*10**6):
		temperature = np.logspace(startExp, endExp, numSteps)
		for indexTuple, tempVal in np.ndenumerate(temperature):
			index = indexTuple[0]
			self.step(tempVal)
			if (self.energy < self.bestEnergy):
				self.bestPermutation = self.tiles.permutation
				print('pAccept = ' + str(100 * np.exp((self.energy - self.bestEnergy)/tempVal)))
				self.bestEnergy = self.energy
				print(str(self.bestEnergy) + ' at t = ' + str(tempVal) + ' %done = ' + str(100*index/float(temperature.size)))

			if index % 10**6 == 0:
				print('Index = ' + str(index))
				print(self.energy - self.tiles.energy())



## Setup

tiles = Tiles(shuffledTiles, permutation, height, width)
anneal = Annealing(tiles)

#anneal.tiles.display()
print(anneal.bestEnergy)