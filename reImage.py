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
		for i in range(18):
			row = []
			for j in range(12):
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
	def computeEnergy(self):
		currTiles = self.getPermutedTiles(self.tiles)
		energy = 0
		for i in range(18):
			for j in range(12 - 1):
				energy += self.getEnergyVMargin(currTiles[:,:,:,i,j], currTiles[:,:,:,i,j+1])
		for j in range(12):
			for i in range(18 - 1):
				energy += self.getEnergyHMargin(currTiles[:,:,:,i,j], currTiles[:,:,:,i+1,j]) 
		return energy
	
	def getEnergyAround(self, tile):
		tileColors = self.getTileColors(tile)
		tileAbove  = self.getTileColors(Tile(tile.x, tile.y - 1))
		tileBelow  = self.getTileColors(Tile(tile.x, tile.y + 1))
		tileLeft   = self.getTileColors(Tile(tile.x - 1, tile.y))
		tileRight  = self.getTileColors(Tile(tile.x + 1, tile.y))
		energy = np.zeros(8)
		energy[0] = self.getEnergyVMargin(tileLeft, tileColors)
		energy[1] = self.getEnergyVMargin(tileColors, tileRight)
		energy[2] = self.getEnergyHMargin(tileAbove, tileColors)
		energy[3] = self.getEnergyHMargin(tileColors, tileBelow)
		return sum(energy)

	def computeEnergyAlt(self):
		energy = 0;
		for i in range(1,17):
			for j in range(1,11):
				energy += self.getEnergyAround(Tile(i,j))

		for j in range(0,11):
			tileColors = self.getTileColors(Tile(0,j))
			tileColorsNext = self.getTileColors(Tile(0,j+1))
			energy += 2 * self.getEnergyVMargin(tileColors, tileColorsNext)

		for j in range(0,11):
			tileColors = self.getTileColors(Tile(17,j))
			tileColorsNext = self.getTileColors(Tile(17,j+1))
			energy += 2 * self.getEnergyVMargin(tileColors, tileColorsNext)

		for i in range(0,17):
			tileColors = self.getTileColors(Tile(i,0))
			tileColorsNext = self.getTileColors(Tile(i+1,0))
			energy += 2 * self.getEnergyHMargin(tileColors, tileColorsNext)

		for i in range(0,17):
			tileColors = self.getTileColors(Tile(i,11))
			tileColorsNext = self.getTileColors(Tile(i+1,11))
			energy += 2 * self.getEnergyHMargin(tileColors, tileColorsNext)

		## 
		for j in range(1,11):
			tileColors = self.getTileColors(Tile(0,j))
			tileColorsNext = self.getTileColors(Tile(1,j))
			energy += self.getEnergyVMargin(tileColors, tileColorsNext)
		for j in range(1,11):
			tileColors = self.getTileColors(Tile(16,j))
			tileColorsNext = self.getTileColors(Tile(17,j))
			energy += self.getEnergyVMargin(tileColors, tileColorsNext)

		for i in range(0,17):
			tileColors = self.getTileColors(Tile(i,0))
			tileColorsNext = self.getTileColors(Tile(i,1))
			energy += self.getEnergyHMargin(tileColors, tileColorsNext)

		for i in range(0,17):
			tileColors = self.getTileColors(Tile(i,10))
			tileColorsNext = self.getTileColors(Tile(i,11))
			energy += self.getEnergyHMargin(tileColors, tileColorsNext)

		return energy / 2.0



	def deltaEnergy(self, tile1, tile2):
		if (tile1 == tile2):
			return
		if (tile1.x > tile2.x or tile1.y > tile2.y):
			tile1, tile2 = tile2, tile1			
		inRow =  (tile1.y == tile2.y and tile1.x + 1 == tile2.x)
		inCol =  (tile1.x == tile2.x and tile1.y + 1 == tile2.y)
		
		energyBefore = np.zeros(2)
		energyBefore[0] = self.getEnergyAround(tile1)
		energyBefore[1] = self.getEnergyAround(tile2)

		self.swap(tile1, tile2)	# Swap for computation ease
		energyAfter = np.zeros(2)
		energyAfter[0] = self.getEnergyAround(tile1)
		energyBefore[1] = self.getEnergyAround(tile2)
		self.swap(tile1, tile2)	# Swap back
		
		sumEnergyBefore = energyBefore.sum()
		sumEnergyAfter = energyAfter.sum()

		if inRow:
			tile1Colors = self.getTileColors(tile1)
			tile2Colors = self.getTileColors(tile2)
			sumEnergyBefore -= self.getEnergyVMargin(tile1, tile2)
			sumEnergyAfter -= self.getEnergyVMargin(tile2, tile1) 
		if inCol:
			tile1Colors = self.getTileColors(tile1)
			tile2Colors = self.getTileColors(tile2)
			sumEnergyBefore -= self.getEnergyHMargin(tile1, tile2)
			sumEnergyAfter -= self.getEnergyHMargin(tile2, tile1)
		
		return sumEnergyAfter - sumEnergyBefore

tiles = Tiles(shuffledTiles, permutation, height, width)
#tiles.swap(Tile(0,0), Tile(17,0))
#tiles.swap(Tile(0,0), Tile(0, 11))
#tiles.display()
tiles.computeEnergy()