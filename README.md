# ConvCRF
This a Tensorflow implementation of ConvCRF, which is proposed in paper[ "Convolutional CRFs for Semantic Segmentation"][1] writed by Marvin T. T. Teichmann and Roberto Cipolla.
This repository only contain the ConvCRF method , which is implemented as Layer class, so it is easy to connect to your model. The ConvCRF Layer is programmed by reference to
[ConvCRF pyTorch][2] and [crfasrnn_keras][3].    

Requirements
-------------

**Plattform**: *Linux, python2 >= 2.7, tensorflow-1.8, cuda 9.0, cudnn 7.0*

**Python Packages**: *numpy*

[1]: https://arxiv.org/abs/1805.04777
[2]: https://github.com/MarvinTeichmann/ConvCRF
[3]: https://github.com/sadeepj/crfasrnn_keras
