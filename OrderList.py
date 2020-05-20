##
# Pedir una serie de numeros y asignarlos a una lista y luego ordenarla
#

def inputList():
	theList =[]
	while True:
		try:
			numInt = int(input('Ingrese Numeros, Cero para terminar'))
			if numInt == 0:
				break
			theList.append(numInt)
		except ValueError:
			print('Ingrese valores enteros')
	return theList			

def menu(theList):
	while True:
		print('2. Sort List')
		print('3. Reverse List')
		print('9. Salir')
		print('Ingrese el item')
		opcion = int(input('[]'))
		if opcion == 2:
			sortList(theList)
		elif opcion == 3:
			reverseList(theList)
		elif opcion == 9:
			break

						
def sortList(theList):
	theList.sort()
	for elem in theList:
		print('{}\n'.format(elem))

def reverseList(theList):
	theList.reverse()
	for elem in theList:
		print('{}\n'.format(elem))			
						
def main():
	theList = inputList()
	menu(theList)

if __name__ == '__main__':
	main()			
