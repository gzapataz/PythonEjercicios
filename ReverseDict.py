import random

##
#Simulacion de Dados
#

Dice = {
	"Total": 0,
	"Simulated": 0,
	"Expected": 0
}

DiceStats = []

def init():
	for i in range(1,12):
		if (i <= 6):
			probabilidad = i / 36
		else:
			probabilidad = (12 - i) / 36
		Dice = dict(Total=i + 1, Simulated=0, Expected=probabilidad)
		DiceStats.append(Dice)

def rollDice():
	for i in range(1,1000):
		dado1 = random.randint(1, 6)
		dado2 = random.randint(1, 6)
		dicebusq = next((item for item in DiceStats if item['Total'] == dado1 + dado2), None)
		dicebusq['Simulated'] = dicebusq['Simulated'] + 1


def main():
	init()
	rollDice()
	for item in DiceStats:
		print('Dados {0:d} Real {1:.2f}% Esperado {2:.2f}%'.format(item['Total'], item['Simulated'] / 1000 * 100, item['Expected'] * 100))
	

if __name__ == "__main__":
	main()

	
