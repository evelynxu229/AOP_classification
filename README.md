AOP-HMM
=========================
Antioxidant proteins (AOPs) have been found closely linked to disease control for its ability to eliminate excess free radicals. Because it is time-consuming and expensive, the identification of AOPs by using wet-lab experimental techniques is a challenging task. On the other hand, previous studies have shown that the hidden Markov model (HMM) profiles generated by running the HHblits program can provide important clues for many protein classification tasks, similar to PSSM profiles. To the best of our knowledge, there is no published paper on the application of HMM profiles in the identification of AOPs. Here, we proposed a novel computational method, called AOP-HMM, to distinguish AOPs from non-AOPs by exploring evolutionary features from the HMM profiles. Aided by a combination of machine learning techniques (ACC + ANOVA + SMOTE + SVM), the proposed model outperformed most of the existing state-of-the-art predictors through the rigorous jackknife cross validation and the independent test. We hope our model could play a complementary role to existing experimental and computational methods for quickly annotating AOPs and guiding the experimental process.

Installation Process
=========================
Required Python Packages:

Install: python (version >= 3.5)  
Install: sklearn (version >= 0.21.3)  
Install: numpy (version >= 1.17.4)  
Install: PyWavelets (version >= 1.1.1)  
Install: scipy (version >= 1.3.2)  

pip install < package name >  
example: pip install sklearn 

Usage
=========================
To run: $ ACC_ANOVA_SMOTE.py or ACC_ANOVA.py