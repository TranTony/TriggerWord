# Bring trigger words into real products
From 2012, the big giants in technology announced many versions of virtual assistants (VA) such as Google Home, Alexa, and Siri. Virtual assistants gradually replace human in term of office jobs, house works as well as improve human life’s quality. Although the technology corporations provided end-users Software Development Kit (SDK) which allows them to integrate VAs inside their products to make add values, developers still find many restrictions when deploying. For example, I have an electronic piano at home that could not connect with Google Assistant in order to play my favorite recorded song. This is the main reason why I would like to build a personal smart speaker.

In Deep Learning era, a software engineer found various solutions to build up a product that reach commercially quality. That is the reason why I choose Deep Learning to create most of modules in my architecture system. 

# Trigger Word.

Trigger Word is the significant word spoken when end-user would like to wake a virtual assistant up. In my view, a wake word is one of the most important paths in VR because it initializes the whole system. Amazon and Google announce their wake words were not the ideal product specially under noise environment. This inspired Kaggle and Tensorflow collaborating to host a competition of speech recognition last year. The best model in this challenge achieve 91.06% accuracy; I would like to defeat this record. My solution is based on the experience grabbed from many AI projects at the current company, papers on Google Scholar, and suggestion from John Cardente, an AI/ML/DL Engineer graduated from Standford.


# CNN for trigger words.

My solution is developed from the solution of Cardente in here, but I found his CNN architecture is able to improve. The solution of Cardente is quite similar to the approach of Andrew Ng’s team in the course 5 of Deep Learning Specialization on Coursera: a CNN goes with an RNN.  According to John, the architecture accomplishes 80.8% accuracy with the input data transformed Mel Frequency Cepstral Coefficients. I found that in his architecture need fine tune some hyperparameters. By experience, I would like to adapt his CNN net to a very deep CNN net by “twisting” MFCC features with more hidden layers. My design was described details in here named Very Deep CNN 6 (Vd6):
 
![alt text](https://github.com/TranTony/TriggerWord/blob/master/CNN.png)



According to many research papers, a CNN model manifests better results if we do “twist” the network with deeper hidden layers. This is help us obtain a better accuracy while trading off by the running time. In my experience, the architecture of Vd6 has touched the accuracy limitation. I have expanded the number of layers with Vd9 or Vd10 but the obtained results just shrinker with a little rate of 0.05%. That is the reason why Vd6 is the best option which satisfies both the running speed and the accuracy of the system.

In practical, entertainment domain accepts an accuracy of 90% and the response time in 3 seconds. While the accuracy is enough to keep at this rate, I would like to enhance the model by handling an exceptional case called: non-voice detection.
 
# Oop! The size of model was so big.
In my opinion, it is subjective to skip the size of the model. I have a confession to make, I am quite confident with the accuracy of the model, but I am afraid that it contains a huge number of parameters. Moreover, the model requires many computations which took my GPU TX 1080i about 23 hours to train the open command dataset version 2 of Google. 

Thanks to the open courses Tensorflow without PhD, Specialization courses Deep Learning on Coursera and the wonderful idea of Mobile Net and Squeezed Net. 

![alt text](https://github.com/TranTony/TriggerWord/blob/master/compress_idea.png)

That’s great, right.  I replaced a Conv 7x7x64 = 3136 by 7x7x1 + 1x1x64 = 114. The size of check point file reduced at the rate of x27 times from 81 MB to 3 MB while the training time was just 10 minutes.

Although the idea is great, it made a tradeoff between the performance and the accuracy. While the size went down x27 times, the accuracy jumped from over 90% to under 80%. 

Our target is achieving a model with 10 MB, 3MB is still far from 10MB. That is the reason why I want to make the model more convoluted. I decided to stack more layers inside the model.

Thicken layers inside the model always be a good idea but for case, it invented even worse result, from 80% to 40% of accuracy.  The best model that I have found last week was twisting 64 filters layers while reducing 256 filters layers. Although the new model beat the previous model with more than 90%, It has not satisfied me at this rate. I would like to update new solution as soon as I find new one.
 
 
# Reference:

https://github.com/jcardente/kaggle_tfspeech
