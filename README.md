# gan_experiment

My take on GAN

## Trial on Vanilla GAN

Best result on replicating Ian's architecture on his [paper](https://arxiv.org/abs/1406.2661).

Images:

![result_img](./results/mnist.gif)

Learning curve:

![learning_curve](./results/learning_curve.png)

_Note_: Red for generator curve, blue for discriminator

Model [here](./results/weights/mnist_generator.pth)

**Painful lesson**:

1. _DROPOUT IS REALLY IMPORTANT_. When I tried building from Ian's [code](https://github.com/goodfeli/adversarial), I overlooked the dropout component in his architecture. Without dropout, the final result will look something like this.

_Failure case_:

![failure](./results/failure_case_without_dropout.png)

_Learning curve_:

![failure_curve](./results/failure_learning.png)

The curve shows that the generator is generating simple images that the discriminator can distinguish from the real distribution. In my opinion, the discriminator overfits the real distribution in this case; thus, it disregards all images that it does not "memorize" and ends up classifying all images from the generator as fake, making it collapsed.

2. _Training process is susceptible to learning rate_. Adam optimizer doesn't work in GAN training procedure due to its adaptive momentum mechanism making it overshoot the optimal point. I tried several runs with Adam and it turns out to have bad convergence results. (_Note_: This code is using vanilla momentum with SGD)


## Trial on DCGAN

![dcgan.gif](./results/dcgan.gif)

_Remark_: I think the image is much more clearer than the case of vanilla GAN but it is still mediocre. Plus, the convergence is unstable.
