import sys
import array
import numpy as np
from urllib.request import urlopen


def fetch_words(url):
	with urlopen(url) as story:
	  story_words=[]
	  for line in story:
	  	line_words = line.decode('utf-8').split()
	  	for words in line_words:
	  		story_words.append(words)
	return story_words
	  		

def fibonnacci(top):
	fib = np.zeros(top)
	print(fib)
	fib[0] = 0
	fib[1] = 1
	i = 2
	while i < top:
		fib[i] = fib[i-1] + fib[i-2]
		i +=1
	print("fib %5.2f" (fib[0]))

def print_items(items):
	for item in items:
		print(item)
  
  
def main(url):
	words = fetch_words(url)
	print_items(words)


if __name__ == '__main__' :
	x = 0
	y = 0
	while x <= 10:
		print(x)
		x += 1;
		print(x**2)
	if x == 10:
		y = x
	else:
		y = 11
	main(sys.argv[1])
