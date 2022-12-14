# Cluster Analysis of Slope Hazard Seismic Recordings Based Upon Unsupervised Deep Embedded Clustering

The demo codes are implemented in `tensorflow version = 2.5.0`.

## Download raw waveforms

https://drive.google.com/file/d/18d0Ea2rGPCyVs1wLIbuAQk0xhyGTfx2a/view

## Download input dataset in the paper

https://drive.google.com/file/d/16_msi04aoP3AdULSf7RzSsYY1td5uKMf/view?usp=sharing

## Abstract
In many countries, slope disasters (e.g., landslides and rockfalls) have caused land use and road use safety problems, and therefore there is a need for monitoring and early warning. Compared with the method based on images, the method based on seismic data for slope disaster monitoring and early warning can be useful at night and is less affected by climate. However, the lack of seismic recordings for different slope disasters limits the applications using the seismic method. To collect the seismic recordings of slope disasters, we deployed seismometers on the Luhu mountain area, Miaoli, Taiwan. The seismic recordings contain signals of rockfalls, earthquakes, and other natural and anthropogenic sources. To distinguish different classes of seismic waveforms, Deep Embedding Clustering (DEC), an unsupervised clustering algorithm, was used to group seismic recordings according to their features. First, 12 times the median absolute deviation (MAD) of the envelope of continuous recordings was used as the threshold to select seismic waveforms with significantly larger amplitudes than the background noise. About 45,000 seismic recordings were selected, of which 1,751 were manually labeled source classes for finding an optimal number of clusters and evaluating clustering results. Seismic waveforms are converted to spectrograms using the short-time Fourier transform as input to DEC. The DEC clustering results show that most seismic recordings of different source classes have different features and can be classified into different clusters. However, seismic data within the same class, such as local and teleseismic events, may also be assigned to different clusters due to differences in data features. Different classes of seismic recordings may also be assigned to the same cluster if there are similar features on spectrograms, but can be distinguished by other conditions. The identified seismic recordings of rockfalls will be useful for monitoring and analyzing rockfall hazards using seismic data.
##
<img src="/Paper_images/Fig2_label_ex_v2.jpg" alt="drawing" width="750"/>
Examples of labeled seismic waveforms. (a)Daily seismic recordings on February 26, 2019 at LH02 station. Due to road construction near LH02 station, there are significant engineering noises during daytime working hours (approximately GMT0:00~8:00). Waveform examples of (b~c) engineering sources, (d) a rockfall event, (e) a local earthquake, and (f) a vehicle driving by.

<img src="/Paper_images/Fig3_DECmodel.jpg" alt="drawing" width="550"/>
Figure 3. The network architecture of Deep Embedded Clustering (DEC) used in this study. The LC and LR are reconstruction loss and clustering loss. 

<img src="/Paper_images/Fig6_Result.jpg" alt="drawing" width="550"/>
Figure 6. The distribution of labeled data (colored dots) and unlabeled data (white dots) at the pre-train (a) and final DEC (b) stages. The data distribution of each group at the pre-train (c) and final DEC (d) stages. The stars represent the centroids of the clusters. The latent features are represented by using the t-SNE. Example spectrograms of labeled seismic recordings of (e, f, and h) vehicle and (g) rockfall in cluster 1 of the DEC final result.

<img src="/Paper_images/Fig7_DataSamples.jpg" alt="drawing" width="550"/>
Figure 7. Waveforms, spectrograms, and latent features of the 5 seismic recordings closest to each final DEC cluster centroid.

## References
* Junyuan Xie, Ross Girshick, Ali Farhadi. (2016). [Unsupervised Deep Embedding for Clustering Analysis.](https://arxiv.org/abs/1511.06335)
* Xifeng Guo, Xinwang Liu, En Zhu, Jianping Yin. (2017). [Deep Clustering with Convolutional Autoencoders.](https://github.com/XifengGuo/DCEC)
* Mousavi, S. M., W. Zhu, W. Ellsworth, G. Beroza. (2019). [Unsupervised Clustering of Seismic Signals Using Deep Convolutional Autoencoders.](https://github.com/smousavi05/Unsupervised_Deep_Learning)
