***\*数据集构建：\****

![img](file:///C:\Users\1\AppData\Local\Temp\ksohtml22648\wps1.jpg) 

![img](file:///C:\Users\1\AppData\Local\Temp\ksohtml22648\wps2.jpg) 

![img](file:///C:\Users\1\AppData\Local\Temp\ksohtml22648\wps3.jpg) 

***\*模型构建、损失函数、训练：\****

***\*AE1(激活函数为Sigmod):\****

![img](file:///C:\Users\1\AppData\Local\Temp\ksohtml22648\wps4.jpg)![img](file:///C:\Users\1\AppData\Local\Temp\ksohtml22648\wps5.jpg) 

![img](file:///C:\Users\1\AppData\Local\Temp\ksohtml22648\wps6.jpg) 

损失函数：MSEloss

优化器：Adam

![img](file:///C:\Users\1\AppData\Local\Temp\ksohtml22648\wps7.jpg) 

***\*AE2(激活函数为Relu):\****

![img](file:///C:\Users\1\AppData\Local\Temp\ksohtml22648\wps8.jpg) 

![img](file:///C:\Users\1\AppData\Local\Temp\ksohtml22648\wps9.jpg)训练部分与AE1相同

***\*DAE:\****

***\*在Denoising AE中，加入CNN卷积自编码\****

![img](file:///C:\Users\1\AppData\Local\Temp\ksohtml22648\wps10.jpg)![img](file:///C:\Users\1\AppData\Local\Temp\ksohtml22648\wps11.jpg) 

加入噪声

![img](file:///C:\Users\1\AppData\Local\Temp\ksohtml22648\wps12.jpg) 

***\*VAE:\****

![img](file:///C:\Users\1\AppData\Local\Temp\ksohtml22648\wps13.jpg)![img](file:///C:\Users\1\AppData\Local\Temp\ksohtml22648\wps14.jpg) 

Vae_loss:![img](file:///C:\Users\1\AppData\Local\Temp\ksohtml22648\wps15.jpg)

***\*AAE:\****

![img](file:///C:\Users\1\AppData\Local\Temp\ksohtml22648\wps16.jpg)![img](file:///C:\Users\1\AppData\Local\Temp\ksohtml22648\wps17.jpg)![img](file:///C:\Users\1\AppData\Local\Temp\ksohtml22648\wps18.jpg) 

![img](file:///C:\Users\1\AppData\Local\Temp\ksohtml22648\wps19.jpg) 

Loss函数及优化器：

![img](file:///C:\Users\1\AppData\Local\Temp\ksohtml22648\wps20.jpg) 

***\*四 实验结果与讨论\****

***\*由于特征维度的减少，其特征表示能力也在下降，损失的信息也就更多，在经过解码器后，重构图像就越难还原。\****

 

***\*中间层数5：\****

![img](file:///C:\Users\1\AppData\Local\Temp\ksohtml22648\wps21.jpg) 

***\*中间层数10：\****

![img](file:///C:\Users\1\AppData\Local\Temp\ksohtml22648\wps22.jpg) 

***\*中间层数16：\****

![img](file:///C:\Users\1\AppData\Local\Temp\ksohtml22648\wps23.jpg)***\*:\****

***\*中间层数20：\****

![img](file:///C:\Users\1\AppData\Local\Temp\ksohtml22648\wps24.jpg) 

***\*中间层数32：\****

![img](file:///C:\Users\1\AppData\Local\Temp\ksohtml22648\wps25.jpg) 

***\*（以上均为AE2模型经20epochs训练后测试结果）\****

 

 

 

***\*AE1(Sigmoid):\****

![img](file:///C:\Users\1\AppData\Local\Temp\ksohtml22648\wps26.jpg) 

TEST：

![img](file:///C:\Users\1\AppData\Local\Temp\ksohtml22648\wps27.jpg) 

LOSS：

![img](file:///C:\Users\1\AppData\Local\Temp\ksohtml22648\wps28.jpg) 

 

***\*AE2(Relu):\****

![img](file:///C:\Users\1\AppData\Local\Temp\ksohtml22648\wps29.jpg) 

TEST:

![img](file:///C:\Users\1\AppData\Local\Temp\ksohtml22648\wps30.jpg) 

 

LOSS:

![img](file:///C:\Users\1\AppData\Local\Temp\ksohtml22648\wps31.jpg) 

***\*DAE:\****

![img](file:///C:\Users\1\AppData\Local\Temp\ksohtml22648\wps32.jpg) 

***\*TEST:\****

![img](file:///C:\Users\1\AppData\Local\Temp\ksohtml22648\wps33.jpg) 

***\*LOSS:\****

![img](file:///C:\Users\1\AppData\Local\Temp\ksohtml22648\wps34.jpg) 

***\*VAE:\****

![img](file:///C:\Users\1\AppData\Local\Temp\ksohtml22648\wps35.jpg) 

***\*TEST:\****

![img](file:///C:\Users\1\AppData\Local\Temp\ksohtml22648\wps36.jpg) 

***\*LOSS:\****

![img](file:///C:\Users\1\AppData\Local\Temp\ksohtml22648\wps37.jpg) 

***\*AAE:\****

![img](file:///C:\Users\1\AppData\Local\Temp\ksohtml22648\wps38.jpg) 

***\*TEST:\****

![img](file:///C:\Users\1\AppData\Local\Temp\ksohtml22648\wps39.jpg) 

***\*LOSS:\****

![img](file:///C:\Users\1\AppData\Local\Temp\ksohtml22648\wps40.jpg) 

***\*对于Sigmoid和Relu的选取对结果影响：\****

 

***\*从理论上分析：\****

***\*ReLU 激活函数可以假设范围[0，∞]内的所有值。作为余数，它的公式是 ReLU (x) = max (0，x)。当输入的观测值 x_i 假设范围很广的正值时，ReLU 是个很好的选择。如果输入 x_i可以假设负值，那么 ReLU 当然是一个糟糕的选择，而恒等函数是一个更好的选择。\*******\*
\****	***\*Sigmoid 函数可以假定所有值的范围在\*******\*[\*******\*0,1\*******\*]\*******\*作为余数\*******\*,\*******\*只有当输入的观测值 x_i 都在\*******\*[\*******\*0,1\*******\*]\*******\*范围内，或者将它们归一化到这个范围内，才能使用这个激活函数。在这种情况下，对于输出层的激活函数来说，Sigmoid 函数是一个不错的选择。\****

 

***\*实际上：\****

![img](file:///C:\Users\1\AppData\Local\Temp\ksohtml22648\wps41.jpg)![img](file:///C:\Users\1\AppData\Local\Temp\ksohtml22648\wps42.jpg) 

***\*由于数据集输入图片的灰度值均在0~1之间，relu输出即为本身0~1，sigmoid输出在0.5~1之间，由上图颜色可以看出，可能有些像素块在重构的时候本身值不足0.5或者原图像的输入就在0.5以下，这个时候由于sigmoid的“拔高”，会让其呈现比较白的情况，而在原始输入中只有数字本身及其周边是会有非零的数值，sigmoid能让其呈现较白的颜色同时保持其他区域黑色不变，而relu则会保留其本身的状态，就出现了可能在白色区域内出现黑色像素值的情况\****

 

***\*DAE理论补充：\****

***\*如果仅仅只是在像素级别对一张图片进行Encode，然后再重建，这样就无法发现更深层次的信息，很有可能会导致网络记住了一些特征。为了防止这种情况产生，我们可以给输入图片加一些噪声，比方说生成和图片同样大小的高斯分布的数据，然后和图像的像素值相加。如果这样都能重建原来的图片，意味着这个网络能从这些混乱的信息中发现真正有用的特征，此时的Code才能代表输入图片的"精华"。\****

 

***\*AAE理论补充：\****

***\*AAE的核心其实就是利用GAN的思想，利用一个生成器G和一个判别器D进行对抗学习，以区分Real data和Fake data。具体思路是这样的，我现在需要一个满足p(z)概率分布的 向量，但是 实际上满足q(z)分布。那么我就首先生成一个满足p(z)分布的z’向量，打上Real data的标签，然后将z向量打上Fake data(服从q(z)分布)。由于这里的p(z)可以是我们定义的任何一个概率分布，因此整个对抗学习的过程实际上是可以认为是通过调整Encoder不断让其产生的数据的概率分布q(z)接近我们预定的p(z).在原始的AutoEncoders中，没有呈现出原有数据的分布，有可能生成的数据是一样的，Adversarial AutoEncoders额外的添加了一个Discriminator（鉴别器），我们希望生成的Z符合真实的Z‘的分布，将真实的和生成的都送到鉴别器计算差距，如果属于希望的分布就输出为1，否则输出为0。\****

 

***\*VAE理论补充：\****

***\*前面的各种AutoEncoder都是将输入数据转换为vector，其中每个维度代表学习到的数据。而Variational AutoEncoders（VAE）提供了一种概率分布的描述形式，VAE中Encoder描述的是每个潜在属性的概率分布，而不是直接输出一个值。通过这种方法，将给定输入的每个潜在属性表示为概率分布。从状态解码（Decode）时，我们将从每个潜在状态分布中随机采样以生成向量来作为解码器的输入\*******\*。\****

 

***\*MAE（学习ing）：\****

![img](file:///C:\Users\1\AppData\Local\Temp\ksohtml22648\wps43.jpg) 

***\*被誉为CV届的BERT，借用作者大大的问题：\*******\*到底是什么原因导致视觉和语言用的masked autoencoder不一样？核心的三个点是：\****

**1.** ***\*CNN天然适合图像领域，而应用Transformer却显得不那么自然，不过这个问题已经被ViT解了。再看上面几篇工作，会发现相比iGPT的马赛克、dVAE的离散化来说，patch形态是对信息损失最少且相对高效的\*******\*。\****

***\*2.\*******\*信息密度：人类的语言太博大精深了，\*******\*对方\*******\*的每一句话，\*******\*可能\*******\*都有18层含义。而照片（ImageNet）不一样，它就\*******\*只有\*******\*那么多信息，两三个词就能概括。所以预测的时候，预测patch要比预测词语容易很多，只需要对周边的patch稍微有些信息就够了。所以可以放心大胆地mask。这点ViT、BEiT其实也都有，但主要就是最后一点没有深究\*******\*。\****

***\*3.\*******\*需要一个Decoder：首先，是不是一定要复原pixel？\*******\*应该\*******\*是的，因为图片信息密度有限，复原pixel这种细粒度信息会让模型强上加强。那怎么复原呢？BEiT已经说过了，在预训练图像encoder的时候，太关注细节就损失了高维抽象能力。所以\*******\*作者\*******\*加了一个decoder。到这里分工就很明确了，encoder负责抽取高维表示，decoder则负责细粒度还原\*******\*。\****

 

***\*MAE解决了后续两个问题：\****

![img](file:///C:\Users\1\AppData\Local\Temp\ksohtml22648\wps44.jpg) 

***\*·  输入侧直接丢掉mask token，效果+0.7，效率x3.3\****

***\*·  预测normalize之后的pixel，效果+0.5\****

***\*·  选取数据增强策略，效果+0.2\****

***\*具体代码还在研究学习中，自己复现还有些难度，暂时先拓展至此。\****

 

***\*贴一下MAE的最终成果图，膜拜大佬，太过震惊\****![img](file:///C:\Users\1\AppData\Local\Temp\ksohtml22648\wps45.jpg)