Sneakit is a sneaker recognizer app who classify sneakers using scrapped google images. I used the Scrapper library to scrapped my images using 4 cores. This library handles errors too.

 An iPhone Application coded in Swift 4 using the MLCoreLibrary has been built as well for a real life application.

I first scrapped google images using product ids for multiple colorways of a same sneakers model.
The modelling has been achieved using the VGG-16 architecture using Keras. Transfer learning has also been used using the ImageNet weights.
To train my models I used multiple AWS multiple instances.
I converted my python models using the mlcoretools to convert them into a .mlmodel format Xcode 9 could use.
