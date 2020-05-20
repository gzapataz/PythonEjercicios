##
#Convierte una cadena infija a postfija usando la funcion del ejercicio anterior Tokenizer
#

import tokenizeString as tk

# Retorna verdadero si la presedencia de op1 < op2 falso de lo contrario

def precedence(op1, op2):
	if op1 in ['+', '-']:
		return True
	elif op1 in ['*', '/'] and op2 in ['^', '(', ')']:
		return True
	elif op1 in ['^'] and op2 in ['(', ')']:
		return True
	elif op1 in ['(', ')']:
		return False	

#Convierte una lista infijo a postfijo
def convertPostfix(tokenList):
	operators = [] # Almacena los operadores
	postfix = [] # Va guardando la cadena en postfijo
	for token in tokenList:
		if token.lstrip('-').isdigit(): #Evalua si es digito teniendo en cuenta si el primero es un -
			postfix.append(token)
		elif token in ['*', '/', '+', '-', '^']: #Evalua los operadores matematicos y su precedencia
			while len(operators) > 0 and operators[len(operators) - 1] != '(' and precedence(token, operators[len(operators) - 1]):
				 postfix.append(operators.pop())
			operators.append(token)
		elif token == '(':
			operators.append(token)
		elif token == ')':
			i = len(operators) - 1
			while operators[i] != '(':
				postfix.append(operators.pop())
				i = i - 1
			operators.pop()
	while len(operators) > 0:
		postfix.append(operators.pop())
	return postfix
				

def main():
	tokenList = []
	operators = [] # Almacena los operadores
	postfix = [] # Va guardando la cadena en postfijo
	stToken = input('Ingrese la expresion Matematica Valida:')
	tokenList = tk.tokenizeStr(stToken)
	postfix = convertPostfix(tokenList)
	print('La cadena tokenizada es: {}'.format(tokenList))
	print('La cadena en postifijo es: {}'.format(postfix))
	

if __name__ == '__main__':
	main()
