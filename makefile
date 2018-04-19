run:
	make clean
	python main.py

clean:
	rm -f *~ 
	find ./ -depth -name ".AppleDouble" -exec rm -Rf {} \;
	find ./ -depth -name ".DS_Store" -exec rm -Rf {} \;
	find ./ -depth -name "__pycache__" -exec rm -Rf {} \;	
	-rm -f ./classified_images/correct/ensembles/* 
	-rm -f ./classified_images/incorrect/ensembles/* 
	-rm -f ./classified_images/correct/* 
	-rm -f ./classified_images/incorrect/*

