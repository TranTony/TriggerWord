# TriggerWord
From 2012, the big giants in technology announced many versions of virtual assistants (VA) such as Google Home, Alexa, and Siri. Virtual assistants gradually replace human in term of office jobs, house works as well as improve human life’s quality. Although the technology corporations provided end-users Software Development Kit (SDK) which allows them to integrate VAs inside their products to make add values, developers still find many restrictions when deploying. For example, I have an electronic piano at home that could not connect with Google Assistant in order to play my favorite recorded song. This is the main reason why I would like to build a personal smart speaker.

In Deep Learning era, a software engineer found various solutions to build up a product that reach commercially quality. That is the reason why I choose Deep Learning to create most of modules in my architecture system. 

Trigger Word.

Trigger Word is the significant word spoken when end-user would like to wake a virtual assistant up. In my view, a wake word is one of the most important paths in VR because it initializes the whole system. Amazon and Google announce their wake words were not the ideal product specially under noise environment. This inspired Kaggle and Tensorflow collaborating to host a competition of speech recognition last year. The best model in this challenge achieve 91.06% accuracy; I would like to defeat this record. My solution is based on the experience grabbed from many AI projects at the current company, papers on Google Scholar, and suggestion from John Cardente, an AI/ML/DL Engineer graduated from Standford.


CNN for trigger words.

My solution is developed from the solution of Cardente in here, but I found his CNN architecture is able to improve. The solution of Cardente is quite similar to the approach of Andrew Ng’s team in the course 5 of Deep Learning Specialization on Coursera: a CNN goes with an RNN.  According to John, the architecture accomplishes 80.8% accuracy with the input data transformed Mel Frequency Cepstral Coefficients. I found that in his architecture need fine tune some hyperparameters. By experience, I would like to adapt his CNN net to a very deep CNN net by “twisting” MFCC features with more hidden layers. My design was described details in here named Very Deep CNN 6 (Vd6):
 
 
According to many research papers, a CNN model manifests better results if we do “twist” the network with deeper hidden layers. This is help us obtain a better accuracy while trading off by the running time. In my experience, the architecture of Vd6 has touched the accuracy limitation. I have expanded the number of layers with Vd9 or Vd10 but the obtained results just shrinker with a little rate of 0.05%. That is the reason why Vd6 is the best option which satisfies both the running speed and the accuracy of the system.

In practical, entertainment domain accepts an accuracy of 90% and the response time in 3 seconds. While the accuracy is enough to keep at this rate, I would like to enhance the model by handling an exceptional case called: non-voice detection.
 
# Reference:

https://github.com/jcardente/kaggle_tfspeech
