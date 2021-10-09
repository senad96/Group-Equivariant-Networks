# Experiment-Equivariant-Neural-Networks


Group equivariant convolutional neural networks
(G-CNNs) has been introduced by Cohen &
Welling (2016). These new models can be con-
sidered as an evolution of convolutional nerual
networks (CNNs). The key operation which G-
CNNs use is G-Convolution; this is a new layer
that make neural networks equivariant to new
symmetries. In this paper, I’ll provide a theo-
retical overview of G-CNNs, G-Convolution and
finally I’ll show the performance of G-CNNs
vs CNNs on Fashion-MNIST and CIFAR-10
datasets.


Convolution neural networks (CNNs) have become the
gold standard for addressing sevaral problems in Artifi-
cial Intelligence (LeCun et al., 2015). CNNs have allowed
to overcome the limitations of ancient models, in particular they bring the following important ideas into machine
learning (Goodfellow et al., 2016): sparse interaction, parameters sharing and equivariant rappresentation.
Traditional neural networks use matrix multiplication as
layer unit. This setting determines a huge number of parameters with just a few number of layers. On the contrary,
convolutional networks typically have sparse wheights.
This feature, in addition to significantly reducing parameters, makes CNNs capable of capturing common patterns
in the data.
The convolution layer, embedded in CNNs, has a nice property called translation equivariance. In general we say that
a function f is equivariant to a function g if f (g(x)) =
g(f (x)). In the scenario of convolution, the function g
can be any translation function of the input. A drawback is that convolution is not equivariant to other transfor-
mations/symmetries like rotations and reflections; for in-
stance, rotating the image and then convolving is not the same as first convolving and then rotating the result. Thus,
CNNs are not able to exploit directly other symmetries as
well as translation. Cohen & Welling (2016) proposed a
generalization of convolution G-convolution to overcome
this issue.


For more information you can read the paper. 
In the Notebook you can have access to the code. All dependencies are intstalled in the notebook. Therefore the notebook is ready to perform.

For more details about the implementation, you can look the original repository [GrouPy](https://github.com/tscohen/GrouPy)
