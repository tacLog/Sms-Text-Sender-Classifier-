# Sms Text Sender Classifier

##Intro
This is a toy project to get some practice feature extraction from SMS text messages. Various classifiers are used for better or worse. The end goal is to use a simple neural network to try and get the best possible results on maybe the 5 most common contacts.

##Data :
The SMS data set I used was my own gathering of 7 or so years of real-world messages. I would never be comfortable releasing such personal information. I suspect the earlier messages will be more useful to the classifier as I tent to avoid texting for more than planning purposes as calling becomes a more attractive option for reconnecting. 

###Use your own Data:
The data was gathered with an android app called [SMS Backup & Restore](https://play.google.com/store/apps/details?id=com.riteshsahu.SMSBackupRestore&hl=en_US)

##Steps:
- Backup your SMS texts
- Move the xml file to the root folder
-Change the xml_parser.py to reflect the new file
-Play with corpusProcessor.py