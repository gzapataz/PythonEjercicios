##
# Calcula el interes de una cantidad dada
import math as m

while True:
	try:
		amount = float(input('Ingrese la cantidad:'))
		interestString = input('Ingrese la tasa:')
		years = float(input('Ingrese los años:'))
		if '%' in interestString:
			interest = float(interestString.split('%')[0]) / 100
		else:
			raise ValueError
		break
	except ValueError:
		print('Ingrese valores numericos o % de la tasa de interes')


finalamount = amount * pow((1 + interest), years)
continuousfinalamount = amount * m.exp(years * interest)
print('la cantidad final es: {0:.2f} Continuo {1:.2f}'.format(finalamount, continuousfinalamount))


