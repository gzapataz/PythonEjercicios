import random as rd
import matplotlib.pyplot as plt
import array as ar
import numpy as np

class sentencias():
	def printOut(cadena):
		print("Salida:" + format(cadena, ">10"))
	
	def forInput(minVal, maxVal):
		suma = 0
		for i in range(minVal, maxVal):
			suma += i
		return suma

	def getRandom(minVal, maxVal):
		return rd.randrange(minVal, maxVal)
		
class mathClass:

	def __init__(self, coefVal, bVal, minVal, maxVal):
		self.yFunc = np.zeros(maxVal)
		self.xVal = range(minVal, maxVal)
		#yFunc = array(minVal + maxVal)
		for i in self.xVal:
			self.yVal = coefVal * i**2 + bVal
			self.yFunc[i] = self.yVal
		print(self.yFunc)
		plt.plot(self.yFunc)
		plt.show()
		
	def getChart():
		return self.yFunc
	

if __name__ == "__main__":
	sentencias.printOut("\nINICIANDO PROGRAMA DE PRUEBA\n")
	a = eval(input("Ingrese el valor a:"))
	b = eval(input("Ingrese el valor b:"))
	b = 10 if a == "" else 50
	print("el valor de b:" + format(b, "*>3"))
	print("La suma es: " + format (sentencias.forInput(a, b), "#>3"))
	rndnum = sentencias.getRandom(a, b)
	sentencias.printOut(rndnum)
	func = mathClass(3, 5, -10, 10)
	print(str(func))
	#plt.plot(mathClass(3, 5, -10, 10))
	
