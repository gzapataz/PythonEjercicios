import datetime as dt


class Person:
	def __init__(self):
		self.nombre = None
		self.apellido = None
		self.fecha_nacimiento = None
		self.__ingreso = 0 #Atributo Privado
		
	def getEdad(self):
		return dt.date.today()
	
	def setIngreso(self, ingreso):
		if ingreso < 0:
			self.__ingreso = 0
		else:
			self.__ingreso = ingreso

	def getIngreso(self):
		return self.__ingreso

if __name__ == "__main__":
	person1 = Person()
	person1.apellido = "Zapata"
	person1.nombre = "Gabriel"
	person1.setIngreso(100)
	
	person1._Person__ingreso = 101 # forma de cambiar un atributo privado en una clase
	person1.fecha_nacimiento = '01/01/1990'
	print(person1.apellido, person1.nombre, person1.getIngreso(), person1._Person__ingreso)
	print("\n Edad :", person1.getEdad())
	
