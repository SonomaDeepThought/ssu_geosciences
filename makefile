clean:
	rm -f *~ 
	find ./ -depth -name ".AppleDouble" -exec rm -Rf {} \;
	find ./ -depth -name ".DS_Store" -exec rm -Rf {} \;	
