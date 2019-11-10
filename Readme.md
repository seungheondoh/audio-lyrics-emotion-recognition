# BI-MODAL MUSIC EMOTION RECOGNITION AND CLASSIFICATIONWITH AUDIO AND LYRICS DATA

This Repository is an implementation repository for [MUSIC MOOD DETECTION BASED ON AUDIO AND LYRICS WITH DEEP NEURAL NET](http://ismir2018.ircam.fr/doc/pdfs/99_Paper.pdf). The baseline paper is "Music  Mood  Detection  Based  On  Audio  And  Lyrics With Deep Neural Net". R. Delbouys et al tried to solve Music Emotion Recognition problem using two CNN layers and two dense layers using Deezer's music database and lyrics. Main contriubtion is using Multi-modal Architecture in Regression task. They used R square value to present the explanatory power of their models as performances.

 For a higher level of understanding, we changed the Music Emotion Recognition Task into a Regression task and a classification task. I proceeded with Label Clustering in two ways based on the hypothesis that humans perceived emotions as large clusters rather than specific points. As a result, Audio Only showed the best performance in Regression Task and Bi-modal showed higher effect in Classification Task.

### Learning code
First, Feature extraction using mel-spectogram in src
```
$ feature_extraction.py
```   
Check hparams.py and change a parameters, and take a train_test base on your task
```
$ train_test_Regression.py
$ train_test_Classification.py
```   

### Dataset

For data sets, we used a sub-million song dataset that R. Delbouys et al used. The number of data was 18111, which was sampled at a sampling rate of 44.1 kHz, and data of 30 seconds was formed. Each data set contained tag words that existed as a pair of the song. Through these tag words, the values of valence and arousal are mapped in semantic space of words.

### Bi-modal Architecture
This bi-modal deep learning structure is expected to combine data from two different domains and reflect information that can not be covered by one domain. This imitates the processing of information by the use of complex senses rather than a structure in which human senses process information through a single sensation. Bi-modal deep Learning removes the language and audio data from each model by removing the fully connected layer, then concatenates the last layers, uploads them to the LSTM, and then encodes them to attach fully connected layer.
<img src="/img/baseline.png">

### Proposal
I changed the regression task to a classification task to verify how meaningful the above results were. I proceeded with Label Clustering in two ways based on the hypothesis that humans perceived emotions as large clusters rather than specific points. Labeling was done through two processes. First, K-mean clustering was used to divide the valence and arousal space regions into 4, 16, and 64, respectively.  K-means clustering showed a different trend depending on the distribution of data without a closed solution. Therefore, We thought that it could not represent the population in a situation where there was not enough data. The cluster was reconstructed by equally dividing the coordinates.
<img src="/img/label.png">

### Result
Label | Audio Train Metric | Audio Test Metric  | Lyrics Train Metric | Lyrics Test Metric | Multi Train Metric | Multi Test Metric |
:---:|:---:|:---:|:---:|:---:|:---:|:---:
Arousal (R2 Score)| 0.160 | 0.155 | 0.014 | 0.006 | 0.158 | 0.144
Valence (R2 Score)| 0.126 | 0.115 | 0.038| 0.012| 0.251| 0.004
kMean4 (Accuracy)| 0.529 | 0.385 | 0.438 | 0.301 | 0.498 | 0.414
abs4 (Accuracy)| 0.603 | 0.427 | 0.455 | 0.329 | 0.545 | 0.465 
kMean16 (Accuracy)| 0.319 | 0.163 | 0.209 | 0.142 | 0.272 | 0.182 
abs16 (Accuracy)| 0.317 | 0.235 | 0.245 | 0.152 | 0.294 | 0.254 
