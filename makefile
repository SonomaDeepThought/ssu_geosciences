clean:
	rm -f *~ 
	find ./ -depth -name ".AppleDouble" -exec rm -Rf {} \;
