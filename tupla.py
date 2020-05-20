class tuplaManejo():
	
	testTupla = ((1,2), [2,3])
	myList = [6, 7, 1, 2, 3]
	
	def reversed():
		print('hola' + str(tuplaManejo.myList))
		print(tuplaManejo.myList)
		
	def counter(mylist):
		for i in mylist:
			print (i)
	
class Avion:
	
	classPlame = ''
	seats: 0
	
	def __init__(self, classPlame, seats):
		self.classPlame = classPlame
		self.seats = seats
		
	def setClass(self, classPlane):
		self.classPlane = classPlane
		print(self.classPlane)
		
		
		
	
if __name__ == '__main__':
	
	
	
	
	
	tuplaManejo.reversed()
	tuplaManejo.counter([1,2,3,'e','b', 'c'])
	x = Avion(3, 2)
	print (x.classPlame)
	x.setClass('AirBus320')
