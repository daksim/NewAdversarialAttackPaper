# Latest Adversarial Attack Papers
**update at 2023-03-27 11:02:57**

翻译来自 https://cloud.tencent.com/document/product/551/15619

## **1. How many dimensions are required to find an adversarial example?**

需要多少维度才能找到对抗性的例子？ cs.LG

Comments welcome!

**SubmitDate**: 2023-03-24    [abs](http://arxiv.org/abs/2303.14173v1) [paper-pdf](http://arxiv.org/pdf/2303.14173v1)

**Authors**: Charles Godfrey, Henry Kvinge, Elise Bishoff, Myles Mckay, Davis Brown, Tim Doster, Eleanor Byler

**Abstract**: Past work exploring adversarial vulnerability have focused on situations where an adversary can perturb all dimensions of model input. On the other hand, a range of recent works consider the case where either (i) an adversary can perturb a limited number of input parameters or (ii) a subset of modalities in a multimodal problem. In both of these cases, adversarial examples are effectively constrained to a subspace $V$ in the ambient input space $\mathcal{X}$. Motivated by this, in this work we investigate how adversarial vulnerability depends on $\dim(V)$. In particular, we show that the adversarial success of standard PGD attacks with $\ell^p$ norm constraints behaves like a monotonically increasing function of $\epsilon (\frac{\dim(V)}{\dim \mathcal{X}})^{\frac{1}{q}}$ where $\epsilon$ is the perturbation budget and $\frac{1}{p} + \frac{1}{q} =1$, provided $p > 1$ (the case $p=1$ presents additional subtleties which we analyze in some detail). This functional form can be easily derived from a simple toy linear model, and as such our results land further credence to arguments that adversarial examples are endemic to locally linear models on high dimensional spaces.

摘要: 过去探索对手脆弱性的工作主要集中在对手可以扰乱模型输入的所有维度的情况。另一方面，最近的一系列工作考虑了这样的情况：(I)对手可以扰动有限数量的输入参数或(Ii)多通道问题中的一组通道。在这两种情况下，敌意示例都被有效地约束到环境输入空间$\mathcal{X}$中的子空间$V$。受此启发，在本工作中，我们研究了对手脆弱性是如何依赖于$\dim(V)$的。特别地，我们证明了具有$^p$范数约束的标准PGD攻击的对抗成功表现为$\epsilon(\frac{\dim(V)}{\dim\mathcal{X}})^{\frac{1}{q}}$的单调递增函数，其中$\epsilon$是扰动预算，而$\frac{1}{p}+\frac{1}{q}=1$，假设$p>1$($p=1$给出了更多的细节，我们进行了一些详细的分析)。这种函数形式可以很容易地从一个简单的玩具线性模型中得到，因此我们的结果进一步证明了高维空间上的对抗性例子是局部线性模型特有的。



## **2. Adversarial Attack and Defense for Medical Image Analysis: Methods and Applications**

医学图像分析中的对抗性攻防方法及应用 eess.IV

**SubmitDate**: 2023-03-24    [abs](http://arxiv.org/abs/2303.14133v1) [paper-pdf](http://arxiv.org/pdf/2303.14133v1)

**Authors**: Junhao Dong, Junxi Chen, Xiaohua Xie, Jianhuang Lai, Hao Chen

**Abstract**: Deep learning techniques have achieved superior performance in computer-aided medical image analysis, yet they are still vulnerable to imperceptible adversarial attacks, resulting in potential misdiagnosis in clinical practice. Oppositely, recent years have also witnessed remarkable progress in defense against these tailored adversarial examples in deep medical diagnosis systems. In this exposition, we present a comprehensive survey on recent advances in adversarial attack and defense for medical image analysis with a novel taxonomy in terms of the application scenario. We also provide a unified theoretical framework for different types of adversarial attack and defense methods for medical image analysis. For a fair comparison, we establish a new benchmark for adversarially robust medical diagnosis models obtained by adversarial training under various scenarios. To the best of our knowledge, this is the first survey paper that provides a thorough evaluation of adversarially robust medical diagnosis models. By analyzing qualitative and quantitative results, we conclude this survey with a detailed discussion of current challenges for adversarial attack and defense in medical image analysis systems to shed light on future research directions.

摘要: 深度学习技术在计算机辅助医学图像分析中取得了优异的性能，但仍然容易受到潜移默化的对抗性攻击，导致临床实践中潜在的误诊。相反，近年来在防御深度医疗诊断系统中这些量身定做的对抗性例子方面也取得了显著进展。在这篇论述中，我们从应用场景的角度对医学图像分析中的对抗性攻击和防御的最新进展进行了全面的综述。为医学图像分析中不同类型的对抗性攻击和防御方法提供了统一的理论框架。为了进行公平的比较，我们建立了一个新的基准，用于在不同场景下通过对抗性训练获得对抗性健壮的医疗诊断模型。据我们所知，这是第一份对反面稳健的医疗诊断模型进行彻底评估的调查报告。通过对定性和定量结果的分析，我们对当前医学图像分析系统中对抗性攻击和防御的挑战进行了详细的讨论，以阐明未来的研究方向。



## **3. Improved Adversarial Training Through Adaptive Instance-wise Loss Smoothing**

通过自适应实例损失平滑改进对手训练 cs.CV

12 pages, work in submission

**SubmitDate**: 2023-03-24    [abs](http://arxiv.org/abs/2303.14077v1) [paper-pdf](http://arxiv.org/pdf/2303.14077v1)

**Authors**: Lin Li, Michael Spratling

**Abstract**: Deep neural networks can be easily fooled into making incorrect predictions through corruption of the input by adversarial perturbations: human-imperceptible artificial noise. So far adversarial training has been the most successful defense against such adversarial attacks. This work focuses on improving adversarial training to boost adversarial robustness. We first analyze, from an instance-wise perspective, how adversarial vulnerability evolves during adversarial training. We find that during training an overall reduction of adversarial loss is achieved by sacrificing a considerable proportion of training samples to be more vulnerable to adversarial attack, which results in an uneven distribution of adversarial vulnerability among data. Such "uneven vulnerability", is prevalent across several popular robust training methods and, more importantly, relates to overfitting in adversarial training. Motivated by this observation, we propose a new adversarial training method: Instance-adaptive Smoothness Enhanced Adversarial Training (ISEAT). It jointly smooths both input and weight loss landscapes in an adaptive, instance-specific, way to enhance robustness more for those samples with higher adversarial vulnerability. Extensive experiments demonstrate the superiority of our method over existing defense methods. Noticeably, our method, when combined with the latest data augmentation and semi-supervised learning techniques, achieves state-of-the-art robustness against $\ell_{\infty}$-norm constrained attacks on CIFAR10 of 59.32% for Wide ResNet34-10 without extra data, and 61.55% for Wide ResNet28-10 with extra data. Code is available at https://github.com/TreeLLi/Instance-adaptive-Smoothness-Enhanced-AT.

摘要: 深层神经网络很容易被欺骗，通过破坏对抗性扰动的输入做出错误的预测：人类无法察觉的人工噪声。到目前为止，对抗性训练一直是对这种对抗性攻击最成功的防御。这项工作的重点是改进对手训练，以提高对手的稳健性。我们首先从实例的角度分析在对抗性训练过程中对抗性脆弱性是如何演变的。我们发现，在训练过程中，通过牺牲相当大比例的训练样本来更容易受到对手攻击，从而总体上减少了对手的损失，这导致了对手脆弱性在数据中的不均匀分布。这种“脆弱性参差不齐”普遍存在于几种流行的健壮训练方法中，更重要的是与对抗性训练中的过度适应有关。基于这一观察结果，我们提出了一种新的对抗性训练方法：实例自适应平滑增强对抗性训练(ISEAT)。它以一种自适应的、特定于实例的方式联合平滑输入和减肥环境，以增强那些具有更高对手脆弱性的样本的健壮性。大量的实验证明了该方法相对于现有防御方法的优越性。值得注意的是，当我们的方法与最新的数据增强和半监督学习技术相结合时，对于针对CIFAR10的$-范数约束攻击，对于没有额外数据的宽ResNet34-10达到了59.32%，对于具有额外数据的宽ResNet28-10达到了61.55%。代码可在https://github.com/TreeLLi/Instance-adaptive-Smoothness-Enhanced-AT.上找到



## **4. PIAT: Parameter Interpolation based Adversarial Training for Image Classification**

PIAT：基于参数内插的对抗性图像分类训练 cs.CV

**SubmitDate**: 2023-03-24    [abs](http://arxiv.org/abs/2303.13955v1) [paper-pdf](http://arxiv.org/pdf/2303.13955v1)

**Authors**: Kun He, Xin Liu, Yichen Yang, Zhou Qin, Weigao Wen, Hui Xue, John E. Hopcroft

**Abstract**: Adversarial training has been demonstrated to be the most effective approach to defend against adversarial attacks. However, existing adversarial training methods show apparent oscillations and overfitting issue in the training process, degrading the defense efficacy. In this work, we propose a novel framework, termed Parameter Interpolation based Adversarial Training (PIAT), that makes full use of the historical information during training. Specifically, at the end of each epoch, PIAT tunes the model parameters as the interpolation of the parameters of the previous and current epochs. Besides, we suggest to use the Normalized Mean Square Error (NMSE) to further improve the robustness by aligning the clean and adversarial examples. Compared with other regularization methods, NMSE focuses more on the relative magnitude of the logits rather than the absolute magnitude. Extensive experiments on several benchmark datasets and various networks show that our method could prominently improve the model robustness and reduce the generalization error. Moreover, our framework is general and could further boost the robust accuracy when combined with other adversarial training methods.

摘要: 对抗性训练已被证明是防御对抗性攻击的最有效方法。然而，现有的对抗性训练方法在训练过程中表现出明显的振荡和过度匹配问题，降低了防守效能。在这项工作中，我们提出了一种新的框架，称为基于参数内插的对抗性训练(PIAT)，它充分利用了训练过程中的历史信息。具体地说，在每个历元结束时，PIAT将模型参数调整为前一个历元和当前历元的参数的内插。此外，我们建议使用归一化均方误差(NMSE)来进一步提高稳健性，通过对齐干净的和对抗性的例子。与其他正则化方法相比，NMSE更注重对数的相对大小，而不是绝对大小。在多个基准数据集和不同网络上的大量实验表明，该方法可以显著提高模型的稳健性，降低泛化误差。此外，我们的框架是通用的，当与其他对抗性训练方法相结合时，可以进一步提高鲁棒性准确率。



## **5. EC-CFI: Control-Flow Integrity via Code Encryption Counteracting Fault Attacks**

EC-CFI：通过代码加密对抗错误攻击的控制流完整性 cs.CR

Accepted at HOST'23

**SubmitDate**: 2023-03-24    [abs](http://arxiv.org/abs/2301.13760v2) [paper-pdf](http://arxiv.org/pdf/2301.13760v2)

**Authors**: Pascal Nasahl, Salmin Sultana, Hans Liljestrand, Karanvir Grewal, Michael LeMay, David M. Durham, David Schrammel, Stefan Mangard

**Abstract**: Fault attacks enable adversaries to manipulate the control-flow of security-critical applications. By inducing targeted faults into the CPU, the software's call graph can be escaped and the control-flow can be redirected to arbitrary functions inside the program. To protect the control-flow from these attacks, dedicated fault control-flow integrity (CFI) countermeasures are commonly deployed. However, these schemes either have high detection latencies or require intrusive hardware changes. In this paper, we present EC-CFI, a software-based cryptographically enforced CFI scheme with no detection latency utilizing hardware features of recent Intel platforms. Our EC-CFI prototype is designed to prevent an adversary from escaping the program's call graph using faults by encrypting each function with a different key before execution. At runtime, the instrumented program dynamically derives the decryption key, ensuring that the code only can be successfully decrypted when the program follows the intended call graph. To enable this level of protection on Intel commodity systems, we introduce extended page table (EPT) aliasing allowing us to achieve function-granular encryption by combing Intel's TME-MK and virtualization technology. We open-source our custom LLVM-based toolchain automatically protecting arbitrary programs with EC-CFI. Furthermore, we evaluate our EPT aliasing approach with the SPEC CPU2017 and Embench-IoT benchmarks and discuss and evaluate potential TME-MK hardware changes minimizing runtime overheads.

摘要: 故障攻击使攻击者能够操纵安全关键型应用程序的控制流。通过在CPU中引入有针对性的错误，可以避开软件的调用图，并将控制流重定向到程序内的任意函数。为了保护控制流免受这些攻击，通常部署专用的故障控制流完整性(CFI)对策。然而，这些方案要么具有很高的检测延迟，要么需要侵入性的硬件改变。在本文中，我们提出了EC-CFI，这是一种基于软件的密码强制CFI方案，利用最近Intel平台的硬件特性，没有检测延迟。我们的EC-CFI原型旨在通过在执行前使用不同的密钥加密每个函数，防止对手使用错误逃离程序的调用图。在运行时，插入指令的程序动态地派生解密密钥，确保只有当程序遵循预期的调用图时才能成功解密代码。为了在英特尔商用系统上实现这种级别的保护，我们引入了扩展页表(EPT)别名，使我们能够通过结合英特尔的TME-MK和虚拟化技术来实现函数级加密。我们将基于LLVM的定制工具链开源，使用EC-CFI自动保护任意程序。此外，我们使用SPEC CPU2017和Embase-IoT基准评估了我们的EPT混叠方法，并讨论和评估了潜在的TME-MK硬件更改，以最大限度地减少运行时开销。



## **6. SCRAMBLE-CFI: Mitigating Fault-Induced Control-Flow Attacks on OpenTitan**

SCRIBLE-CFI：缓解OpenTitan上的错误引起的控制流攻击 cs.CR

Accepted at GLSVLSI'23

**SubmitDate**: 2023-03-24    [abs](http://arxiv.org/abs/2303.03711v3) [paper-pdf](http://arxiv.org/pdf/2303.03711v3)

**Authors**: Pascal Nasahl, Stefan Mangard

**Abstract**: Secure elements physically exposed to adversaries are frequently targeted by fault attacks. These attacks can be utilized to hijack the control-flow of software allowing the attacker to bypass security measures, extract sensitive data, or gain full code execution. In this paper, we systematically analyze the threat vector of fault-induced control-flow manipulations on the open-source OpenTitan secure element. Our thorough analysis reveals that current countermeasures of this chip either induce large area overheads or still cannot prevent the attacker from exploiting the identified threats. In this context, we introduce SCRAMBLE-CFI, an encryption-based control-flow integrity scheme utilizing existing hardware features of OpenTitan. SCRAMBLE-CFI confines, with minimal hardware overhead, the impact of fault-induced control-flow attacks by encrypting each function with a different encryption tweak at load-time. At runtime, code only can be successfully decrypted when the correct decryption tweak is active. We open-source our hardware changes and release our LLVM toolchain automatically protecting programs. Our analysis shows that SCRAMBLE-CFI complementarily enhances security guarantees of OpenTitan with a negligible hardware overhead of less than 3.97 % and a runtime overhead of 7.02 % for the Embench-IoT benchmarks.

摘要: 物理上暴露在对手面前的安全元素经常成为故障攻击的目标。这些攻击可用于劫持软件的控制流，从而允许攻击者绕过安全措施、提取敏感数据或获得完整的代码执行。在本文中，我们系统地分析了开源OpenTitan安全元素上由错误引起的控制流操作的威胁向量。我们的深入分析表明，目前该芯片的应对措施要么导致大面积开销，要么仍然无法阻止攻击者利用已识别的威胁。在此背景下，我们介绍了一种基于加密的控制流完整性方案SCRIBLE-CFI，该方案利用了OpenTitan现有的硬件特性。置乱-CFI通过在加载时使用不同的加密调整对每个函数进行加密，以最小的硬件开销限制了故障引发的控制流攻击的影响。在运行时，只有当正确的解密调整处于活动状态时，才能成功解密代码。我们将我们的硬件更改开源，并发布我们的LLVM工具链自动保护程序。我们的分析表明，在硬件开销小于3.97%、运行时开销为7.02%的情况下，SCRIBLE-CFI互补地增强了OpenTitan的安全保证。



## **7. Foiling Explanations in Deep Neural Networks**

深度神经网络中的模糊解释 cs.CV

**SubmitDate**: 2023-03-24    [abs](http://arxiv.org/abs/2211.14860v2) [paper-pdf](http://arxiv.org/pdf/2211.14860v2)

**Authors**: Snir Vitrack Tamam, Raz Lapid, Moshe Sipper

**Abstract**: Deep neural networks (DNNs) have greatly impacted numerous fields over the past decade. Yet despite exhibiting superb performance over many problems, their black-box nature still poses a significant challenge with respect to explainability. Indeed, explainable artificial intelligence (XAI) is crucial in several fields, wherein the answer alone -- sans a reasoning of how said answer was derived -- is of little value. This paper uncovers a troubling property of explanation methods for image-based DNNs: by making small visual changes to the input image -- hardly influencing the network's output -- we demonstrate how explanations may be arbitrarily manipulated through the use of evolution strategies. Our novel algorithm, AttaXAI, a model-agnostic, adversarial attack on XAI algorithms, only requires access to the output logits of a classifier and to the explanation map; these weak assumptions render our approach highly useful where real-world models and data are concerned. We compare our method's performance on two benchmark datasets -- CIFAR100 and ImageNet -- using four different pretrained deep-learning models: VGG16-CIFAR100, VGG16-ImageNet, MobileNet-CIFAR100, and Inception-v3-ImageNet. We find that the XAI methods can be manipulated without the use of gradients or other model internals. Our novel algorithm is successfully able to manipulate an image in a manner imperceptible to the human eye, such that the XAI method outputs a specific explanation map. To our knowledge, this is the first such method in a black-box setting, and we believe it has significant value where explainability is desired, required, or legally mandatory.

摘要: 在过去的十年中，深度神经网络(DNN)对众多领域产生了巨大的影响。然而，尽管在许多问题上表现出了出色的表现，但它们的黑匣子性质仍然在可解释性方面构成了一个重大挑战。事实上，可解释人工智能(XAI)在几个领域都是至关重要的，在这些领域中，答案本身--不考虑答案是如何得出的--几乎没有价值。本文揭示了基于图像的DNN解释方法的一个令人不安的特性：通过对输入图像进行微小的视觉改变--几乎不影响网络的输出--我们演示了如何通过使用进化策略来任意操纵解释。我们的新算法AttaXAI是对XAI算法的一种与模型无关的对抗性攻击，它只需要访问分类器的输出日志和解释地图；这些弱假设使得我们的方法在涉及真实世界的模型和数据时非常有用。我们使用四个不同的预训练深度学习模型：VGG16-CIFAR100、VGG16-ImageNet、MobileNet-CIFAR100和Inception-v3-ImageNet，在两个基准数据集CIFAR100和ImageNet上比较了我们的方法的性能。我们发现，XAI方法可以在不使用梯度或其他模型内部的情况下进行操作。我们的新算法能够成功地以人眼看不到的方式操作图像，从而XAI方法输出特定的解释地图。据我们所知，这是黑盒环境中第一个这样的方法，我们相信它在需要可解释性、要求可解释性或法律强制性的地方具有重要价值。



## **8. Effective black box adversarial attack with handcrafted kernels**

利用手工制作的核进行有效的黑盒对抗攻击 cs.CV

12 pages, 5 figures, 3 tables, IWANN conference

**SubmitDate**: 2023-03-24    [abs](http://arxiv.org/abs/2303.13887v1) [paper-pdf](http://arxiv.org/pdf/2303.13887v1)

**Authors**: Petr Dvořáček, Petr Hurtik, Petra Števuliáková

**Abstract**: We propose a new, simple framework for crafting adversarial examples for black box attacks. The idea is to simulate the substitution model with a non-trainable model compounded of just one layer of handcrafted convolutional kernels and then train the generator neural network to maximize the distance of the outputs for the original and generated adversarial image. We show that fooling the prediction of the first layer causes the whole network to be fooled and decreases its accuracy on adversarial inputs. Moreover, we do not train the neural network to obtain the first convolutional layer kernels, but we create them using the technique of F-transform. Therefore, our method is very time and resource effective.

摘要: 我们提出了一个新的、简单的框架来制作黑盒攻击的对抗性例子。其思想是用仅由一层手工制作的卷积核组成的不可训练模型来模拟替换模型，然后训练生成器神经网络以最大化原始和生成的对抗性图像的输出距离。我们表明，愚弄第一层的预测会导致整个网络被愚弄，并降低其对对手输入的精度。此外，我们不训练神经网络来获得第一卷积层核，但我们使用F变换技术来创建它们。因此，我们的方法是非常节省时间和资源的。



## **9. Physically Adversarial Infrared Patches with Learnable Shapes and Locations**

具有可学习形状和位置的物理对抗性红外线补丁 cs.CV

accepted by CVPR2023

**SubmitDate**: 2023-03-24    [abs](http://arxiv.org/abs/2303.13868v1) [paper-pdf](http://arxiv.org/pdf/2303.13868v1)

**Authors**: Wei Xingxing, Yu Jie, Huang Yao

**Abstract**: Owing to the extensive application of infrared object detectors in the safety-critical tasks, it is necessary to evaluate their robustness against adversarial examples in the real world. However, current few physical infrared attacks are complicated to implement in practical application because of their complex transformation from digital world to physical world. To address this issue, in this paper, we propose a physically feasible infrared attack method called "adversarial infrared patches". Considering the imaging mechanism of infrared cameras by capturing objects' thermal radiation, adversarial infrared patches conduct attacks by attaching a patch of thermal insulation materials on the target object to manipulate its thermal distribution. To enhance adversarial attacks, we present a novel aggregation regularization to guide the simultaneous learning for the patch' shape and location on the target object. Thus, a simple gradient-based optimization can be adapted to solve for them. We verify adversarial infrared patches in different object detection tasks with various object detectors. Experimental results show that our method achieves more than 90\% Attack Success Rate (ASR) versus the pedestrian detector and vehicle detector in the physical environment, where the objects are captured in different angles, distances, postures, and scenes. More importantly, adversarial infrared patch is easy to implement, and it only needs 0.5 hours to be constructed in the physical world, which verifies its effectiveness and efficiency.

摘要: 由于红外目标探测器在安全关键任务中的广泛应用，有必要评估其对现实世界中的敌方例子的鲁棒性。然而，由于从数字世界到物理世界的复杂转换，目前较少的物理红外攻击在实际应用中实现起来比较复杂。针对这一问题，本文提出了一种物理上可行的红外攻击方法，称为对抗性红外补丁。考虑到红外相机通过捕捉目标的热辐射来成像的机理，对抗红外贴片通过在目标对象上粘贴一块隔热材料来操纵目标对象的热分布来进行攻击。为了增强对抗性攻击，我们提出了一种新的聚合正则化方法来指导对目标物体上补丁的形状和位置的同时学习。因此，可以采用一种简单的基于梯度的优化方法来求解它们。在不同的目标检测任务中，我们使用不同的目标检测器来验证敌方红外补丁。实验结果表明，与行人检测器和车辆检测器相比，在不同角度、不同距离、不同姿态和不同场景下的物理环境中，该方法的攻击成功率(ASR)达到了90%以上。更重要的是，对抗性红外补丁易于实现，在物理世界中仅需0.5小时即可构建，验证了其有效性和高效性。



## **10. Feature Separation and Recalibration for Adversarial Robustness**

用于对抗稳健性的特征分离和重新校准 cs.CV

CVPR 2023 (Highlight)

**SubmitDate**: 2023-03-24    [abs](http://arxiv.org/abs/2303.13846v1) [paper-pdf](http://arxiv.org/pdf/2303.13846v1)

**Authors**: Woo Jae Kim, Yoonki Cho, Junsik Jung, Sung-Eui Yoon

**Abstract**: Deep neural networks are susceptible to adversarial attacks due to the accumulation of perturbations in the feature level, and numerous works have boosted model robustness by deactivating the non-robust feature activations that cause model mispredictions. However, we claim that these malicious activations still contain discriminative cues and that with recalibration, they can capture additional useful information for correct model predictions. To this end, we propose a novel, easy-to-plugin approach named Feature Separation and Recalibration (FSR) that recalibrates the malicious, non-robust activations for more robust feature maps through Separation and Recalibration. The Separation part disentangles the input feature map into the robust feature with activations that help the model make correct predictions and the non-robust feature with activations that are responsible for model mispredictions upon adversarial attack. The Recalibration part then adjusts the non-robust activations to restore the potentially useful cues for model predictions. Extensive experiments verify the superiority of FSR compared to traditional deactivation techniques and demonstrate that it improves the robustness of existing adversarial training methods by up to 8.57% with small computational overhead. Codes are available at https://github.com/wkim97/FSR.

摘要: 由于特征层扰动的积累，深度神经网络容易受到对抗性攻击，许多工作通过去激活导致模型错误预测的非稳健特征激活来增强模型的稳健性。然而，我们声称这些恶意激活仍然包含歧视性提示，并且通过重新校准，它们可以捕获更多有用的信息来进行正确的模型预测。为此，我们提出了一种新的、易于插件的方法，称为特征分离和重新校准(FSR)，该方法通过分离和重新校准来重新校准恶意的、非健壮的激活以获得更健壮的特征映射。分离部分将输入特征映射分离为具有帮助模型做出正确预测的激活的健壮特征和具有导致敌方攻击时模型误预测的激活的非健壮特征。然后，重新校准部分调整非稳健激活以恢复模型预测的潜在有用线索。大量的实验验证了FSR与传统去激活技术相比的优越性，并证明了它以较小的计算开销提高了现有对抗性训练方法的健壮性高达8.57%。有关代码，请访问https://github.com/wkim97/FSR.



## **11. Near Optimal Adversarial Attack on UCB Bandits**

对UCB土匪的近最优敌意攻击 cs.LG

**SubmitDate**: 2023-03-24    [abs](http://arxiv.org/abs/2008.09312v3) [paper-pdf](http://arxiv.org/pdf/2008.09312v3)

**Authors**: Shiliang Zuo

**Abstract**: We consider a stochastic multi-arm bandit problem where rewards are subject to adversarial corruption. We propose a novel attack strategy that manipulates a UCB principle into pulling some non-optimal target arm $T - o(T)$ times with a cumulative cost that scales as $\sqrt{\log T}$, where $T$ is the number of rounds. We also prove the first lower bound on the cumulative attack cost. Our lower bound matches our upper bound up to $\log \log T$ factors, showing our attack to be near optimal.

摘要: 我们考虑了一个随机多臂强盗问题，其中报酬服从对抗性腐败。我们提出了一种新的攻击策略，它利用UCB原理来拉动一些非最优目标臂$T-o(T)$次，累积代价可扩展到$\Sqrt{\log T}$，其中$T$是轮数。我们还证明了累积攻击代价的第一个下界。我们的下界与上界匹配，最高可达$\log\log T$因子，表明我们的攻击接近最优。



## **12. RamBoAttack: A Robust Query Efficient Deep Neural Network Decision Exploit**

RamBoAttack：一种稳健查询高效的深度神经网络决策开发 cs.LG

Published in Network and Distributed System Security (NDSS) Symposium  2022. Code is available at https://ramboattack.github.io/

**SubmitDate**: 2023-03-24    [abs](http://arxiv.org/abs/2112.05282v3) [paper-pdf](http://arxiv.org/pdf/2112.05282v3)

**Authors**: Viet Quoc Vo, Ehsan Abbasnejad, Damith C. Ranasinghe

**Abstract**: Machine learning models are critically susceptible to evasion attacks from adversarial examples. Generally, adversarial examples, modified inputs deceptively similar to the original input, are constructed under whitebox settings by adversaries with full access to the model. However, recent attacks have shown a remarkable reduction in query numbers to craft adversarial examples using blackbox attacks. Particularly, alarming is the ability to exploit the classification decision from the access interface of a trained model provided by a growing number of Machine Learning as a Service providers including Google, Microsoft, IBM and used by a plethora of applications incorporating these models. The ability of an adversary to exploit only the predicted label from a model to craft adversarial examples is distinguished as a decision-based attack. In our study, we first deep dive into recent state-of-the-art decision-based attacks in ICLR and SP to highlight the costly nature of discovering low distortion adversarial employing gradient estimation methods. We develop a robust query efficient attack capable of avoiding entrapment in a local minimum and misdirection from noisy gradients seen in gradient estimation methods. The attack method we propose, RamBoAttack, exploits the notion of Randomized Block Coordinate Descent to explore the hidden classifier manifold, targeting perturbations to manipulate only localized input features to address the issues of gradient estimation methods. Importantly, the RamBoAttack is more robust to the different sample inputs available to an adversary and the targeted class. Overall, for a given target class, RamBoAttack is demonstrated to be more robust at achieving a lower distortion within a given query budget. We curate our extensive results using the large-scale high-resolution ImageNet dataset and open-source our attack, test samples and artifacts on GitHub.

摘要: 机器学习模型极易受到来自对手例子的逃避攻击。通常，对抗性的例子，修改后的输入欺骗性地类似于原始输入，由具有完全访问模型的敌手在白盒设置下构造。然而，最近的攻击显示，使用黑盒攻击构建敌意例子的查询数量显著减少。特别是，警报是利用由包括谷歌、微软、IBM在内的越来越多的机器学习即服务提供商提供的训练模型的访问接口的分类决策的能力，并被结合这些模型的大量应用程序使用。对手仅利用模型中预测的标签来制作敌意示例的能力被区分为基于决策的攻击。在我们的研究中，我们首先深入研究了ICLR和SP中最新的基于决策的攻击，以强调使用梯度估计方法发现低失真攻击的代价。我们开发了一种健壮的查询高效攻击，能够避免陷入局部最小值和从梯度估计方法中看到的噪声梯度的误导。我们提出的攻击方法RamBoAttack利用随机化块坐标下降的概念来探索隐藏的分类器流形，针对扰动只操纵局部输入特征来解决梯度估计方法的问题。重要的是，RamBoAttack对于对手和目标类可用的不同样本输入更加健壮。总体而言，对于给定的目标类，RamBoAttack被证明在给定的查询预算内实现较低的失真方面更加健壮。我们使用大规模高分辨率ImageNet数据集和在GitHub上开源的我们的攻击、测试样本和人工制品来管理我们广泛的结果。



## **13. Query Efficient Decision Based Sparse Attacks Against Black-Box Deep Learning Models**

基于查询高效决策的黑盒深度学习模型稀疏攻击 cs.LG

Published as a conference paper at the International Conference on  Learning Representations (ICLR 2022). Code is available at  https://sparseevoattack.github.io/

**SubmitDate**: 2023-03-24    [abs](http://arxiv.org/abs/2202.00091v2) [paper-pdf](http://arxiv.org/pdf/2202.00091v2)

**Authors**: Viet Quoc Vo, Ehsan Abbasnejad, Damith C. Ranasinghe

**Abstract**: Despite our best efforts, deep learning models remain highly vulnerable to even tiny adversarial perturbations applied to the inputs. The ability to extract information from solely the output of a machine learning model to craft adversarial perturbations to black-box models is a practical threat against real-world systems, such as autonomous cars or machine learning models exposed as a service (MLaaS). Of particular interest are sparse attacks. The realization of sparse attacks in black-box models demonstrates that machine learning models are more vulnerable than we believe. Because these attacks aim to minimize the number of perturbed pixels measured by l_0 norm-required to mislead a model by solely observing the decision (the predicted label) returned to a model query; the so-called decision-based attack setting. But, such an attack leads to an NP-hard optimization problem. We develop an evolution-based algorithm-SparseEvo-for the problem and evaluate against both convolutional deep neural networks and vision transformers. Notably, vision transformers are yet to be investigated under a decision-based attack setting. SparseEvo requires significantly fewer model queries than the state-of-the-art sparse attack Pointwise for both untargeted and targeted attacks. The attack algorithm, although conceptually simple, is also competitive with only a limited query budget against the state-of-the-art gradient-based whitebox attacks in standard computer vision tasks such as ImageNet. Importantly, the query efficient SparseEvo, along with decision-based attacks, in general, raise new questions regarding the safety of deployed systems and poses new directions to study and understand the robustness of machine learning models.

摘要: 尽管我们尽了最大努力，深度学习模型仍然非常容易受到应用于输入的微小对抗性扰动的影响。仅从机器学习模型的输出中提取信息以对黑盒模型进行敌意扰动的能力，对现实世界的系统构成了实际威胁，例如自动驾驶汽车或暴露为服务的机器学习模型(MLaaS)。特别令人感兴趣的是稀疏攻击。稀疏攻击在黑盒模型中的实现表明，机器学习模型比我们认为的更容易受到攻击。因为这些攻击的目的是最小化由l_0范数测量的扰动像素数-通过仅观察返回给模型查询的决策(预测标签)来误导模型所需的；所谓的基于决策的攻击设置。但是，这样的攻击导致了一个NP-Hard优化问题。我们开发了一种基于进化的算法SparseEvo来解决该问题，并对卷积深度神经网络和视觉转换器进行了评估。值得注意的是，视觉变形器尚未在基于决策的攻击环境下进行调查。对于非目标攻击和目标攻击，SparseEvo需要的模型查询比最先进的稀疏攻击点要少得多。攻击算法虽然在概念上很简单，但与标准计算机视觉任务(如ImageNet)中最先进的基于梯度的白盒攻击相比，仅有有限的查询预算也具有竞争力。重要的是，查询效率高的SparseEvo以及基于决策的攻击通常对已部署系统的安全性提出了新的问题，并为研究和理解机器学习模型的健壮性提供了新的方向。



## **14. CBA: Contextual Background Attack against Optical Aerial Detection in the Physical World**

CBA：物理世界中对光学空中探测的背景攻击 cs.CV

**SubmitDate**: 2023-03-24    [abs](http://arxiv.org/abs/2302.13519v3) [paper-pdf](http://arxiv.org/pdf/2302.13519v3)

**Authors**: Jiawei Lian, Xiaofei Wang, Yuru Su, Mingyang Ma, Shaohui Mei

**Abstract**: Patch-based physical attacks have increasingly aroused concerns.   However, most existing methods focus on obscuring targets captured on the ground, and some of these methods are simply extended to deceive aerial detectors.   They smear the targeted objects in the physical world with the elaborated adversarial patches, which can only slightly sway the aerial detectors' prediction and with weak attack transferability.   To address the above issues, we propose to perform Contextual Background Attack (CBA), a novel physical attack framework against aerial detection, which can achieve strong attack efficacy and transferability in the physical world even without smudging the interested objects at all.   Specifically, the targets of interest, i.e. the aircraft in aerial images, are adopted to mask adversarial patches.   The pixels outside the mask area are optimized to make the generated adversarial patches closely cover the critical contextual background area for detection, which contributes to gifting adversarial patches with more robust and transferable attack potency in the real world.   To further strengthen the attack performance, the adversarial patches are forced to be outside targets during training, by which the detected objects of interest, both on and outside patches, benefit the accumulation of attack efficacy.   Consequently, the sophisticatedly designed patches are gifted with solid fooling efficacy against objects both on and outside the adversarial patches simultaneously.   Extensive proportionally scaled experiments are performed in physical scenarios, demonstrating the superiority and potential of the proposed framework for physical attacks.   We expect that the proposed physical attack method will serve as a benchmark for assessing the adversarial robustness of diverse aerial detectors and defense methods.

摘要: 基于补丁的物理攻击越来越引起人们的关注。然而，现有的大多数方法都集中在遮挡地面捕获的目标上，其中一些方法只是简单地扩展到欺骗航空探测器。他们用精心制作的对抗性补丁涂抹物理世界中的目标对象，这只能轻微动摇航空探测器的预测，攻击可转移性较弱。为了解决上述问题，我们提出了一种新的针对空中探测的物理攻击框架--上下文背景攻击(CBA)，该框架即使在不玷污感兴趣对象的情况下也可以在物理世界中实现强大的攻击效能和可转移性。具体地说，采用感兴趣的目标，即航空图像中的飞机来掩盖敌方补丁。对掩码区域外的像素进行了优化，使生成的对抗性补丁紧密覆盖关键背景区域进行检测，有助于在现实世界中赋予对抗性补丁更健壮和可转移的攻击能力。为了进一步增强攻击性能，在训练过程中将对抗性补丁强制为外部目标，这样无论是在补丁上还是在补丁外，检测到的感兴趣对象都有利于攻击效能的积累。因此，复杂设计的补丁被赋予了对敌方补丁内外的对象同时具有可靠的愚弄效果。在物理场景中进行了广泛的按比例扩展的实验，展示了所提出的框架在物理攻击方面的优势和潜力。我们期望所提出的物理攻击方法将作为评估不同空中探测器和防御方法的对抗健壮性的基准。



## **15. TrojViT: Trojan Insertion in Vision Transformers**

TrojViT：视觉变形金刚中的特洛伊木马插入 cs.LG

10 pages, 4 figures, 11 tables

**SubmitDate**: 2023-03-23    [abs](http://arxiv.org/abs/2208.13049v3) [paper-pdf](http://arxiv.org/pdf/2208.13049v3)

**Authors**: Mengxin Zheng, Qian Lou, Lei Jiang

**Abstract**: Vision Transformers (ViTs) have demonstrated the state-of-the-art performance in various vision-related tasks. The success of ViTs motivates adversaries to perform backdoor attacks on ViTs. Although the vulnerability of traditional CNNs to backdoor attacks is well-known, backdoor attacks on ViTs are seldom-studied. Compared to CNNs capturing pixel-wise local features by convolutions, ViTs extract global context information through patches and attentions. Na\"ively transplanting CNN-specific backdoor attacks to ViTs yields only a low clean data accuracy and a low attack success rate. In this paper, we propose a stealth and practical ViT-specific backdoor attack $TrojViT$. Rather than an area-wise trigger used by CNN-specific backdoor attacks, TrojViT generates a patch-wise trigger designed to build a Trojan composed of some vulnerable bits on the parameters of a ViT stored in DRAM memory through patch salience ranking and attention-target loss. TrojViT further uses minimum-tuned parameter update to reduce the bit number of the Trojan. Once the attacker inserts the Trojan into the ViT model by flipping the vulnerable bits, the ViT model still produces normal inference accuracy with benign inputs. But when the attacker embeds a trigger into an input, the ViT model is forced to classify the input to a predefined target class. We show that flipping only few vulnerable bits identified by TrojViT on a ViT model using the well-known RowHammer can transform the model into a backdoored one. We perform extensive experiments of multiple datasets on various ViT models. TrojViT can classify $99.64\%$ of test images to a target class by flipping $345$ bits on a ViT for ImageNet.

摘要: 视觉变形金刚(VITS)在各种与视觉相关的任务中展示了最先进的性能。VITS的成功促使对手对VITS进行后门攻击。虽然传统的CNN对后门攻击的脆弱性是众所周知的，但对VITS的后门攻击很少被研究。与通过卷积获取像素级局部特征的CNN相比，VITS通过块和关注点来提取全局上下文信息。将CNN特定的后门攻击活生生地移植到VITS只会产生低的干净数据准确性和低的攻击成功率。在本文中，我们提出了一种隐形和实用的特定于VIT的后门攻击$TrojViT$。与CNN特定后门攻击使用的区域触发不同，TrojViT生成修补程序触发，旨在通过修补程序显著程度排名和注意力目标丢失来构建由存储在DRAM内存中的VIT参数上的一些易受攻击位组成的特洛伊木马程序。TrojViT进一步使用最小调整的参数更新来减少特洛伊木马的比特数。一旦攻击者通过翻转易受攻击的比特将特洛伊木马程序插入到VIT模型中，VIT模型仍然会使用良性输入产生正常的推理准确性。但是，当攻击者将触发器嵌入到输入中时，VIT模型被迫将输入分类到预定义的目标类。我们表明，只需使用著名的RowHammer在VIT模型上翻转TrojViT识别的少数易受攻击的位，就可以将该模型转换为后置模型。我们在不同的VIT模型上对多个数据集进行了广泛的实验。TrojViT可以通过在ImageNet的VIT上翻转$345$比特，将$99.64\$测试图像分类到目标类别。



## **16. Adversarial Robustness and Feature Impact Analysis for Driver Drowsiness Detection**

驾驶员嗜睡检测的对抗稳健性和特征影响分析 cs.LG

10 pages, 2 tables, 3 figures, AIME 2023 conference

**SubmitDate**: 2023-03-23    [abs](http://arxiv.org/abs/2303.13649v1) [paper-pdf](http://arxiv.org/pdf/2303.13649v1)

**Authors**: João Vitorino, Lourenço Rodrigues, Eva Maia, Isabel Praça, André Lourenço

**Abstract**: Drowsy driving is a major cause of road accidents, but drivers are dismissive of the impact that fatigue can have on their reaction times. To detect drowsiness before any impairment occurs, a promising strategy is using Machine Learning (ML) to monitor Heart Rate Variability (HRV) signals. This work presents multiple experiments with different HRV time windows and ML models, a feature impact analysis using Shapley Additive Explanations (SHAP), and an adversarial robustness analysis to assess their reliability when processing faulty input data and perturbed HRV signals. The most reliable model was Extreme Gradient Boosting (XGB) and the optimal time window had between 120 and 150 seconds. Furthermore, SHAP enabled the selection of the 18 most impactful features and the training of new smaller models that achieved a performance as good as the initial ones. Despite the susceptibility of all models to adversarial attacks, adversarial training enabled them to preserve significantly higher results, especially XGB. Therefore, ML models can significantly benefit from realistic adversarial training to provide a more robust driver drowsiness detection.

摘要: 疲劳驾驶是交通事故的一个主要原因，但司机们对疲劳对他们的反应时间的影响不屑一顾。为了在任何损害发生之前检测到昏昏欲睡，一个有希望的策略是使用机器学习(ML)来监测心率变异性(HRV)信号。这项工作给出了不同的HRV时间窗和ML模型的多个实验，使用Shapley Additive Informance(Shap)的特征影响分析，以及当处理错误的输入数据和扰动的HRV信号时评估它们的可靠性的对抗性稳健性分析。最可靠的模型是极端梯度增强(XGB)，最佳时间窗口在120到150秒之间。此外，Shap能够选择18个最有影响力的特征，并训练新的较小的模型，这些模型的表现与最初的模型一样好。尽管所有模型都容易受到对抗性攻击，但对抗性训练使它们能够保持显著更高的结果，特别是XGB。因此，ML模型可以显著受益于现实的对抗性训练，以提供更健壮的驾驶员嗜睡检测。



## **17. Efficient Symbolic Reasoning for Neural-Network Verification**

用于神经网络验证的高效符号推理 cs.AI

**SubmitDate**: 2023-03-23    [abs](http://arxiv.org/abs/2303.13588v1) [paper-pdf](http://arxiv.org/pdf/2303.13588v1)

**Authors**: Zi Wang, Somesh Jha, Krishnamurthy, Dvijotham

**Abstract**: The neural network has become an integral part of modern software systems. However, they still suffer from various problems, in particular, vulnerability to adversarial attacks. In this work, we present a novel program reasoning framework for neural-network verification, which we refer to as symbolic reasoning. The key components of our framework are the use of the symbolic domain and the quadratic relation. The symbolic domain has very flexible semantics, and the quadratic relation is quite expressive. They allow us to encode many verification problems for neural networks as quadratic programs. Our scheme then relaxes the quadratic programs to semidefinite programs, which can be efficiently solved. This framework allows us to verify various neural-network properties under different scenarios, especially those that appear challenging for non-symbolic domains. Moreover, it introduces new representations and perspectives for the verification tasks. We believe that our framework can bring new theoretical insights and practical tools to verification problems for neural networks.

摘要: 神经网络已经成为现代软件系统不可或缺的一部分。然而，它们仍然面临着各种问题，特别是易受对抗性攻击。在这项工作中，我们提出了一种新的神经网络验证程序推理框架，我们称之为符号推理。该框架的关键部分是符号域和二次关系的使用。符号域具有非常灵活的语义，二次关系具有很强的表现力。它们允许我们将神经网络的许多验证问题编码为二次规划。然后，我们的方案将二次规划松弛为半定规划，从而可以有效地求解。这个框架允许我们在不同的场景下验证各种神经网络属性，特别是那些对非符号域来说具有挑战性的场景。此外，它还介绍了核查任务的新表述和新视角。我们相信，我们的框架可以为神经网络的验证问题带来新的理论见解和实用工具。



## **18. Symmetries, flat minima, and the conserved quantities of gradient flow**

对称性、平坦极小值和梯度流的守恒量 cs.LG

To appear at ICLR 2023

**SubmitDate**: 2023-03-23    [abs](http://arxiv.org/abs/2210.17216v2) [paper-pdf](http://arxiv.org/pdf/2210.17216v2)

**Authors**: Bo Zhao, Iordan Ganev, Robin Walters, Rose Yu, Nima Dehmamy

**Abstract**: Empirical studies of the loss landscape of deep networks have revealed that many local minima are connected through low-loss valleys. Yet, little is known about the theoretical origin of such valleys. We present a general framework for finding continuous symmetries in the parameter space, which carve out low-loss valleys. Our framework uses equivariances of the activation functions and can be applied to different layer architectures. To generalize this framework to nonlinear neural networks, we introduce a novel set of nonlinear, data-dependent symmetries. These symmetries can transform a trained model such that it performs similarly on new samples, which allows ensemble building that improves robustness under certain adversarial attacks. We then show that conserved quantities associated with linear symmetries can be used to define coordinates along low-loss valleys. The conserved quantities help reveal that using common initialization methods, gradient flow only explores a small part of the global minimum. By relating conserved quantities to convergence rate and sharpness of the minimum, we provide insights on how initialization impacts convergence and generalizability.

摘要: 对深层网络损失格局的实证研究表明，许多局部极小值通过低损失谷连接在一起。然而，人们对这种山谷的理论起源知之甚少。我们给出了一个在参数空间中寻找连续对称性的一般框架，它划出了低损失谷。我们的框架使用了激活函数的等价性，可以应用于不同的层体系结构。为了将这个框架推广到非线性神经网络，我们引入了一组新的非线性、依赖于数据的对称性。这些对称性可以转换训练好的模型，使其在新样本上执行类似的操作，这使得集成构建能够提高在某些对手攻击下的健壮性。然后，我们证明了与线性对称有关的守恒量可以用来定义沿低损耗山谷的坐标。守恒量有助于揭示，使用常见的初始化方法，梯度流只探索全局极小值的一小部分。通过将守恒量与最小值的收敛速度和锐度联系起来，我们提供了关于初始化如何影响收敛和泛化的见解。



## **19. Decentralized Adversarial Training over Graphs**

基于图的分散对抗性训练 cs.LG

arXiv admin note: text overlap with arXiv:2303.01936

**SubmitDate**: 2023-03-23    [abs](http://arxiv.org/abs/2303.13326v1) [paper-pdf](http://arxiv.org/pdf/2303.13326v1)

**Authors**: Ying Cao, Elsa Rizk, Stefan Vlaski, Ali H. Sayed

**Abstract**: The vulnerability of machine learning models to adversarial attacks has been attracting considerable attention in recent years. Most existing studies focus on the behavior of stand-alone single-agent learners. In comparison, this work studies adversarial training over graphs, where individual agents are subjected to perturbations of varied strength levels across space. It is expected that interactions by linked agents, and the heterogeneity of the attack models that are possible over the graph, can help enhance robustness in view of the coordination power of the group. Using a min-max formulation of diffusion learning, we develop a decentralized adversarial training framework for multi-agent systems. We analyze the convergence properties of the proposed scheme for both convex and non-convex environments, and illustrate the enhanced robustness to adversarial attacks.

摘要: 近年来，机器学习模型对敌意攻击的脆弱性引起了人们的极大关注。现有的研究大多集中在单智能体学习者的行为上。相比之下，这项工作研究的是图上的对抗性训练，在图中，单个代理人受到空间上不同强度水平的扰动。考虑到组的协调能力，预计链接代理的交互以及图上可能的攻击模型的异构性可以帮助增强稳健性。利用扩散学习的最小-最大公式，我们提出了一种多智能体系统的分布式对抗训练框架。我们分析了该方案在凸环境和非凸环境下的收敛特性，并说明了该方案增强了对敌意攻击的鲁棒性。



## **20. Source-independent quantum random number generator against tailored detector blinding attacks**

抗定制检测器盲攻击的源无关量子随机数生成器 quant-ph

14 pages, 6 figures, 6 tables, comments are welcome

**SubmitDate**: 2023-03-23    [abs](http://arxiv.org/abs/2204.12156v2) [paper-pdf](http://arxiv.org/pdf/2204.12156v2)

**Authors**: Wen-Bo Liu, Yu-Shuo Lu, Yao Fu, Si-Cheng Huang, Ze-Jie Yin, Kun Jiang, Hua-Lei Yin, Zeng-Bing Chen

**Abstract**: Randomness, mainly in the form of random numbers, is the fundamental prerequisite for the security of many cryptographic tasks. Quantum randomness can be extracted even if adversaries are fully aware of the protocol and even control the randomness source. However, an adversary can further manipulate the randomness via tailored detector blinding attacks, which are hacking attacks suffered by protocols with trusted detectors. Here, by treating no-click events as valid events, we propose a quantum random number generation protocol that can simultaneously address source vulnerability and ferocious tailored detector blinding attacks. The method can be extended to high-dimensional random number generation. We experimentally demonstrate the ability of our protocol to generate random numbers for two-dimensional measurement with a generation speed of 0.1 bit per pulse.

摘要: 随机性，主要是随机数的形式，是许多密码任务安全的基本前提。即使攻击者完全知道该协议，甚至控制了随机性来源，也可以提取量子随机性。然而，攻击者可以通过定制的检测器盲化攻击进一步操纵随机性，这是具有可信检测器的协议遭受的黑客攻击。这里，通过将无点击事件视为有效事件，我们提出了一种量子随机数生成协议，该协议可以同时应对源漏洞和凶猛的定制检测器盲攻击。该方法可以推广到高维随机数的生成。我们通过实验证明了我们的协议能够以每脉冲0.1比特的速度产生用于二维测量的随机数。



## **21. Watch Out for the Confusing Faces: Detecting Face Swapping with the Probability Distribution of Face Identification Models**

警惕易混淆的人脸：利用人脸识别模型的概率分布检测人脸互换 cs.CV

**SubmitDate**: 2023-03-23    [abs](http://arxiv.org/abs/2303.13131v1) [paper-pdf](http://arxiv.org/pdf/2303.13131v1)

**Authors**: Yuxuan Duan, Xuhong Zhang, Chuer Yu, Zonghui Wang, Shouling Ji, Wenzhi Chen

**Abstract**: Recently, face swapping has been developing rapidly and achieved a surprising reality, raising concerns about fake content. As a countermeasure, various detection approaches have been proposed and achieved promising performance. However, most existing detectors struggle to maintain performance on unseen face swapping methods and low-quality images. Apart from the generalization problem, current detection approaches have been shown vulnerable to evasion attacks crafted by detection-aware manipulators. Lack of robustness under adversary scenarios leaves threats for applying face swapping detection in real world. In this paper, we propose a novel face swapping detection approach based on face identification probability distributions, coined as IdP_FSD, to improve the generalization and robustness. IdP_FSD is specially designed for detecting swapped faces whose identities belong to a finite set, which is meaningful in real-world applications. Compared with previous general detection methods, we make use of the available real faces with concerned identities and require no fake samples for training. IdP_FSD exploits face swapping's common nature that the identity of swapped face combines that of two faces involved in swapping. We reflect this nature with the confusion of a face identification model and measure the confusion with the maximum value of the output probability distribution. What's more, to defend our detector under adversary scenarios, an attention-based finetuning scheme is proposed for the face identification models used in IdP_FSD. Extensive experiments show that the proposed IdP_FSD not only achieves high detection performance on different benchmark datasets and image qualities but also raises the bar for manipulators to evade the detection.

摘要: 最近，人脸互换发展迅速，并取得了令人惊讶的现实，引发了人们对虚假内容的担忧。作为对策，人们提出了各种检测方法，并取得了良好的效果。然而，大多数现有的检测器难以保持对看不见的人脸交换方法和低质量图像的性能。除了泛化问题外，目前的检测方法已经被证明容易受到由具有检测意识的操纵者精心设计的逃避攻击。在敌方场景下缺乏健壮性，给在现实世界中应用人脸交换检测带来了威胁。本文提出了一种新的基于人脸识别概率分布的人脸交换检测方法，称为IDP_FSD，以提高算法的泛化能力和鲁棒性。IdP_FSD是专门为检测身份属于有限集的交换人脸而设计的，这在现实世界的应用中具有重要意义。与以往的一般检测方法相比，该方法利用了已有的具有相关身份的真实人脸，并且不需要假样本进行训练。IdP_FSD利用了人脸交换的共同特性，即被交换的人脸的身份结合了参与交换的两个人脸的身份。我们用人脸识别模型的混淆来反映这一性质，并用输出概率分布的最大值来衡量混淆程度。此外，为了在敌方场景下保护我们的检测器，针对IDP_FSD中使用的人脸识别模型，提出了一种基于注意力的精调方案。大量实验表明，提出的IDP_FSD不仅在不同的基准数据集和图像质量上取得了较高的检测性能，而且提高了操纵者逃避检测的门槛。



## **22. Patch of Invisibility: Naturalistic Black-Box Adversarial Attacks on Object Detectors**

隐形补丁：对物体探测器的自然主义黑箱对抗性攻击 cs.CV

**SubmitDate**: 2023-03-23    [abs](http://arxiv.org/abs/2303.04238v3) [paper-pdf](http://arxiv.org/pdf/2303.04238v3)

**Authors**: Raz Lapid, Moshe Sipper

**Abstract**: Adversarial attacks on deep-learning models have been receiving increased attention in recent years. Work in this area has mostly focused on gradient-based techniques, so-called white-box attacks, wherein the attacker has access to the targeted model's internal parameters; such an assumption is usually unrealistic in the real world. Some attacks additionally use the entire pixel space to fool a given model, which is neither practical nor physical (i.e., real-world). On the contrary, we propose herein a gradient-free method that uses the learned image manifold of a pretrained generative adversarial network (GAN) to generate naturalistic physical adversarial patches for object detectors. We show that our proposed method works both digitally and physically.

摘要: 近年来，针对深度学习模型的对抗性攻击受到越来越多的关注。这一领域的工作主要集中在基于梯度的技术，即所谓的白盒攻击，即攻击者可以访问目标模型的内部参数；这种假设在现实世界中通常是不现实的。一些攻击还使用整个像素空间来愚弄给定的模型，这既不实用也不物理(即，现实世界)。相反，我们在这里提出了一种无梯度的方法，它使用预先训练的生成性对抗性网络(GAN)的学习图像流形来为目标检测器生成自然的物理对抗性斑块。我们证明了我们提出的方法在数字和物理上都是有效的。



## **23. BlockFW -- Towards Blockchain-based Rule-Sharing Firewall**

BlockFW--迈向基于区块链的规则共享防火墙 cs.CR

The 16th International Conference on Emerging Security Information,  Systems and Technologies (SECURWARE 2022), pp. 70-75, IARIA 2022

**SubmitDate**: 2023-03-23    [abs](http://arxiv.org/abs/2303.13073v1) [paper-pdf](http://arxiv.org/pdf/2303.13073v1)

**Authors**: Wei-Yang Chiu, Weizhi Meng

**Abstract**: Central-managed security mechanisms are often utilized in many organizations, but such server is also a security breaking point. This is because the server has the authority for all nodes that share the security protection. Hence if the attackers successfully tamper the server, the organization will be in trouble. Also, the settings and policies saved on the server are usually not cryptographically secured and ensured with hash. Thus, changing the settings from alternative way is feasible, without causing the security solution to raise any alarms. To mitigate these issues, in this work, we develop BlockFW - a blockchain-based rule sharing firewall to create a managed security mechanism, which provides validation and monitoring from multiple nodes. For BlockFW, all occurred transactions are cryptographically protected to ensure its integrity, making tampering attempts in utmost challenging for attackers. In the evaluation, we explore the performance of BlockFW under several adversarial conditions and demonstrate its effectiveness.

摘要: 许多组织经常使用集中管理的安全机制，但这样的服务器也是一个安全突破点。这是因为服务器拥有共享安全保护的所有节点的权限。因此，如果攻击者成功篡改服务器，该组织将陷入困境。此外，保存在服务器上的设置和策略通常不会受到加密保护，也不会使用散列进行确保。因此，从另一种方式更改设置是可行的，而不会导致安全解决方案发出任何警报。为了缓解这些问题，在本工作中，我们开发了BlockFW-一个基于区块链的规则共享防火墙来创建一个托管安全机制，提供来自多个节点的验证和监控。对于BlockFW来说，所有发生的交易都受到加密保护，以确保其完整性，这使得篡改尝试对攻击者来说是最具挑战性的。在评估中，我们探索了BlockFW在几种对抗条件下的性能，并证明了其有效性。



## **24. Semantic Image Attack for Visual Model Diagnosis**

面向视觉模型诊断的语义图像攻击 cs.CV

Initial version submitted to NeurIPS 2022

**SubmitDate**: 2023-03-23    [abs](http://arxiv.org/abs/2303.13010v1) [paper-pdf](http://arxiv.org/pdf/2303.13010v1)

**Authors**: Jinqi Luo, Zhaoning Wang, Chen Henry Wu, Dong Huang, Fernando De la Torre

**Abstract**: In practice, metric analysis on a specific train and test dataset does not guarantee reliable or fair ML models. This is partially due to the fact that obtaining a balanced, diverse, and perfectly labeled dataset is typically expensive, time-consuming, and error-prone. Rather than relying on a carefully designed test set to assess ML models' failures, fairness, or robustness, this paper proposes Semantic Image Attack (SIA), a method based on the adversarial attack that provides semantic adversarial images to allow model diagnosis, interpretability, and robustness. Traditional adversarial training is a popular methodology for robustifying ML models against attacks. However, existing adversarial methods do not combine the two aspects that enable the interpretation and analysis of the model's flaws: semantic traceability and perceptual quality. SIA combines the two features via iterative gradient ascent on a predefined semantic attribute space and the image space. We illustrate the validity of our approach in three scenarios for keypoint detection and classification. (1) Model diagnosis: SIA generates a histogram of attributes that highlights the semantic vulnerability of the ML model (i.e., attributes that make the model fail). (2) Stronger attacks: SIA generates adversarial examples with visually interpretable attributes that lead to higher attack success rates than baseline methods. The adversarial training on SIA improves the transferable robustness across different gradient-based attacks. (3) Robustness to imbalanced datasets: we use SIA to augment the underrepresented classes, which outperforms strong augmentation and re-balancing baselines.

摘要: 在实践中，对特定训练和测试数据集的度量分析并不能保证可靠或公平的ML模型。这在一定程度上是由于这样一个事实，即获得平衡的、多样化的和完美标记的数据集通常是昂贵、耗时且容易出错的。不依赖于精心设计的测试集来评估ML模型的故障、公平性或健壮性，本文提出了语义图像攻击(SIA)，这是一种基于对抗性攻击的方法，它提供语义对抗性图像来实现模型诊断、可解释性和健壮性。传统的对抗性训练是一种流行的方法，用于增强ML模型的抗攻击能力。然而，现有的对抗性方法没有将能够解释和分析模型缺陷的两个方面结合起来：语义可追溯性和感知质量。SIA通过在预定义的语义属性空间和图像空间上迭代梯度上升来结合这两个特征。我们在三个关键点检测和分类场景中演示了该方法的有效性。(1)模型诊断：SIA生成属性直方图，突出ML模型的语义漏洞(即导致模型失败的属性)。(2)更强的攻击：SIA生成具有视觉上可解释的属性的对抗性示例，从而导致比基准方法更高的攻击成功率。SIA上的对抗性训练提高了对不同基于梯度的攻击的可转移稳健性。(3)对不平衡数据集的健壮性：我们使用SIA来扩充未被代表的类，其表现优于强扩充和再平衡基线。



## **25. Connected Superlevel Set in (Deep) Reinforcement Learning and its Application to Minimax Theorems**

(深度)强化学习中的连通超水平集及其在极大极小定理中的应用 cs.LG

**SubmitDate**: 2023-03-23    [abs](http://arxiv.org/abs/2303.12981v1) [paper-pdf](http://arxiv.org/pdf/2303.12981v1)

**Authors**: Sihan Zeng, Thinh T. Doan, Justin Romberg

**Abstract**: The aim of this paper is to improve the understanding of the optimization landscape for policy optimization problems in reinforcement learning. Specifically, we show that the superlevel set of the objective function with respect to the policy parameter is always a connected set both in the tabular setting and under policies represented by a class of neural networks. In addition, we show that the optimization objective as a function of the policy parameter and reward satisfies a stronger "equiconnectedness" property. To our best knowledge, these are novel and previously unknown discoveries.   We present an application of the connectedness of these superlevel sets to the derivation of minimax theorems for robust reinforcement learning. We show that any minimax optimization program which is convex on one side and is equiconnected on the other side observes the minimax equality (i.e. has a Nash equilibrium). We find that this exact structure is exhibited by an interesting robust reinforcement learning problem under an adversarial reward attack, and the validity of its minimax equality immediately follows. This is the first time such a result is established in the literature.

摘要: 本文的目的是加深对强化学习中策略优化问题的优化环境的理解。具体地说，我们证明了目标函数关于策略参数的超水平集无论在表格设置下还是在由一类神经网络表示的策略下都是连通集。此外，我们还证明了作为政策参数和报酬的函数的优化目标满足较强的等连通性。据我们所知，这些都是以前未知的新奇发现。我们给出了这些超水平集的连通性在鲁棒强化学习的极大极小定理推导中的一个应用。我们证明了任何一边是凸的，另一边是等连通的极小极大优化程序都遵守极小极大等式(即存在纳什均衡)。我们发现，在对抗性奖励攻击下，这种结构被表现为一个有趣的鲁棒强化学习问题，并且它的极小极大等式的有效性随之而来。这是第一次在文献中确立这样的结果。



## **26. Test-time Defense against Adversarial Attacks: Detection and Reconstruction of Adversarial Examples via Masked Autoencoder**

对抗性攻击的测试时间防御：基于屏蔽自动编码器的对抗性实例检测与重构 cs.CV

**SubmitDate**: 2023-03-22    [abs](http://arxiv.org/abs/2303.12848v1) [paper-pdf](http://arxiv.org/pdf/2303.12848v1)

**Authors**: Yun-Yun Tsai, Ju-Chin Chao, Albert Wen, Zhaoyuan Yang, Chengzhi Mao, Tapan Shah, Junfeng Yang

**Abstract**: Existing defense methods against adversarial attacks can be categorized into training time and test time defenses. Training time defense, i.e., adversarial training, requires a significant amount of extra time for training and is often not able to be generalized to unseen attacks. On the other hand, test time defense by test time weight adaptation requires access to perform gradient descent on (part of) the model weights, which could be infeasible for models with frozen weights. To address these challenges, we propose DRAM, a novel defense method to Detect and Reconstruct multiple types of Adversarial attacks via Masked autoencoder (MAE). We demonstrate how to use MAE losses to build a KS-test to detect adversarial attacks. Moreover, the MAE losses can be used to repair adversarial samples from unseen attack types. In this sense, DRAM neither requires model weight updates in test time nor augments the training set with more adversarial samples. Evaluating DRAM on the large-scale ImageNet data, we achieve the best detection rate of 82% on average on eight types of adversarial attacks compared with other detection baselines. For reconstruction, DRAM improves the robust accuracy by 6% ~ 41% for Standard ResNet50 and 3% ~ 8% for Robust ResNet50 compared with other self-supervision tasks, such as rotation prediction and contrastive learning.

摘要: 现有的对抗攻击防御方法可分为训练时间防御和测试时间防御。训练时间防守，即对抗性训练，需要大量的额外时间进行训练，通常不能概括为看不见的攻击。另一方面，通过测试时间权重自适应来保护测试时间需要访问对模型权重(部分)执行梯度下降的权限，这对于具有冻结权重的模型可能是不可行的。为了应对这些挑战，我们提出了一种新的防御方法DRAM，它通过掩蔽自动编码器(MAE)来检测和重建多种类型的对抗性攻击。我们演示了如何使用MAE损失来构建KS测试来检测对手攻击。此外，MAE损失可用于修复来自未知攻击类型的敌方样本。从这个意义上说，DRAM既不需要在测试时间更新模型权重，也不需要用更多的对抗性样本来扩充训练集。在大规模的ImageNet数据上对DRAM进行评估，与其他检测基线相比，对8种类型的对抗性攻击平均获得了82%的最佳检测率。在重建方面，与旋转预测和对比学习等其他自我监督任务相比，DRAM将标准ResNet50的稳健准确率提高了6%~41%，稳健ResNet50的稳健准确率提高了3%~8%。



## **27. Evaluating the Role of Target Arguments in Rumour Stance Classification**

评价目标论元在流言立场分类中的作用 cs.CL

**SubmitDate**: 2023-03-22    [abs](http://arxiv.org/abs/2303.12665v1) [paper-pdf](http://arxiv.org/pdf/2303.12665v1)

**Authors**: Yue Li, Carolina Scarton

**Abstract**: Considering a conversation thread, stance classification aims to identify the opinion (e.g. agree or disagree) of replies towards a given target. The target of the stance is expected to be an essential component in this task, being one of the main factors that make it different from sentiment analysis. However, a recent study shows that a target-oblivious model outperforms target-aware models, suggesting that targets are not useful when predicting stance. This paper re-examines this phenomenon for rumour stance classification (RSC) on social media, where a target is a rumour story implied by the source tweet in the conversation. We propose adversarial attacks in the test data, aiming to assess the models robustness and evaluate the role of the data in the models performance. Results show that state-of-the-art models, including approaches that use the entire conversation thread, overly relying on superficial signals. Our hypothesis is that the naturally high occurrence of target-independent direct replies in RSC (e.g. "this is fake" or just "fake") results in the impressive performance of target-oblivious models, highlighting the risk of target instances being treated as noise during training.

摘要: 考虑到对话主线，立场分类旨在识别回复对给定目标的意见(例如同意或不同意)。预计立场的目标将是这项任务的重要组成部分，是使其不同于情绪分析的主要因素之一。然而，最近的一项研究表明，目标忽略模型的表现优于目标感知模型，这表明目标在预测姿态时并不有用。本文对社交媒体上的谣言立场分类(RSC)中的这一现象进行了重新审视，其中目标是对话中来源推文所暗示的谣言故事。我们在测试数据中提出对抗性攻击，目的是评估模型的稳健性，评估数据在模型性能中的作用。结果表明，最先进的模型，包括使用整个对话线索的方法，过度依赖表面信号。我们的假设是，目标无关的直接回复在RSC中的自然高出现(例如，这是假的或仅仅是假的)导致了目标忽略模型令人印象深刻的性能，突显了目标实例在训练过程中被视为噪声的风险。



## **28. Reliable and Efficient Evaluation of Adversarial Robustness for Deep Hashing-Based Retrieval**

基于深度散列的检索中对抗健壮性的可靠高效评估 cs.CV

arXiv admin note: text overlap with arXiv:2204.10779

**SubmitDate**: 2023-03-22    [abs](http://arxiv.org/abs/2303.12658v1) [paper-pdf](http://arxiv.org/pdf/2303.12658v1)

**Authors**: Xunguang Wang, Jiawang Bai, Xinyue Xu, Xiaomeng Li

**Abstract**: Deep hashing has been extensively applied to massive image retrieval due to its efficiency and effectiveness. Recently, several adversarial attacks have been presented to reveal the vulnerability of deep hashing models against adversarial examples. However, existing attack methods suffer from degraded performance or inefficiency because they underutilize the semantic relations between original samples or spend a lot of time learning these relations with a deep neural network. In this paper, we propose a novel Pharos-guided Attack, dubbed PgA, to evaluate the adversarial robustness of deep hashing networks reliably and efficiently. Specifically, we design pharos code to represent the semantics of the benign image, which preserves the similarity to semantically relevant samples and dissimilarity to irrelevant ones. It is proven that we can quickly calculate the pharos code via a simple math formula. Accordingly, PgA can directly conduct a reliable and efficient attack on deep hashing-based retrieval by maximizing the similarity between the hash code of the adversarial example and the pharos code. Extensive experiments on the benchmark datasets verify that the proposed algorithm outperforms the prior state-of-the-arts in both attack strength and speed.

摘要: 深度散列算法以其高效、高效的特点被广泛应用于海量图像检索中。最近，已经提出了几种对抗性攻击，以揭示深度散列模型对对抗性例子的脆弱性。然而，现有的攻击方法由于没有充分利用原始样本之间的语义关系或花费大量时间利用深度神经网络来学习这些关系，因此存在性能下降或效率低下的问题。本文提出了一种新的Pharos制导攻击，称为PGA，用于可靠、高效地评估深度散列网络的攻击健壮性。具体地说，我们设计了PHAROS代码来表示良性图像的语义，保持了对语义相关样本的相似性和对不相关样本的不相似性。实践证明，我们可以通过一个简单的数学公式快速地计算出航标码。因此，PGA可以通过最大化对抗性实例的哈希码和Pharos码之间的相似度，直接对基于深度哈希的检索进行可靠和高效的攻击。在基准数据集上的大量实验证明，该算法在攻击强度和速度上都优于现有的算法。



## **29. RoBIC: A benchmark suite for assessing classifiers robustness**

Robic：一种评估分类器健壮性的基准测试套件 cs.CV

4 pages, accepted to ICIP 2021

**SubmitDate**: 2023-03-22    [abs](http://arxiv.org/abs/2102.05368v2) [paper-pdf](http://arxiv.org/pdf/2102.05368v2)

**Authors**: Thibault Maho, Benoît Bonnet, Teddy Furon, Erwan Le Merrer

**Abstract**: Many defenses have emerged with the development of adversarial attacks. Models must be objectively evaluated accordingly. This paper systematically tackles this concern by proposing a new parameter-free benchmark we coin RoBIC. RoBIC fairly evaluates the robustness of image classifiers using a new half-distortion measure. It gauges the robustness of the network against white and black box attacks, independently of its accuracy. RoBIC is faster than the other available benchmarks. We present the significant differences in the robustness of 16 recent models as assessed by RoBIC.

摘要: 随着对抗性攻击的发展，出现了许多防御措施。必须相应地对模型进行客观评估。本文系统地解决了这个问题，提出了一个新的无参数基准，我们创造了Robic。Robic使用一种新的半失真度量公平地评估图像分类器的稳健性。它衡量网络抵御白盒和黑盒攻击的稳健性，而与其准确性无关。Robic比其他可用的基准测试更快。我们展示了Robic评估的16个最新模型在稳健性方面的显著差异。



## **30. CgAT: Center-Guided Adversarial Training for Deep Hashing-Based Retrieval**

CgAT：中心引导的基于深度散列的对抗性训练 cs.CV

**SubmitDate**: 2023-03-22    [abs](http://arxiv.org/abs/2204.10779v5) [paper-pdf](http://arxiv.org/pdf/2204.10779v5)

**Authors**: Xunguang Wang, Yiqun Lin, Xiaomeng Li

**Abstract**: Deep hashing has been extensively utilized in massive image retrieval because of its efficiency and effectiveness. However, deep hashing models are vulnerable to adversarial examples, making it essential to develop adversarial defense methods for image retrieval. Existing solutions achieved limited defense performance because of using weak adversarial samples for training and lacking discriminative optimization objectives to learn robust features. In this paper, we present a min-max based Center-guided Adversarial Training, namely CgAT, to improve the robustness of deep hashing networks through worst adversarial examples. Specifically, we first formulate the center code as a semantically-discriminative representative of the input image content, which preserves the semantic similarity with positive samples and dissimilarity with negative examples. We prove that a mathematical formula can calculate the center code immediately. After obtaining the center codes in each optimization iteration of the deep hashing network, they are adopted to guide the adversarial training process. On the one hand, CgAT generates the worst adversarial examples as augmented data by maximizing the Hamming distance between the hash codes of the adversarial examples and the center codes. On the other hand, CgAT learns to mitigate the effects of adversarial samples by minimizing the Hamming distance to the center codes. Extensive experiments on the benchmark datasets demonstrate the effectiveness of our adversarial training algorithm in defending against adversarial attacks for deep hashing-based retrieval. Compared with the current state-of-the-art defense method, we significantly improve the defense performance by an average of 18.61\%, 12.35\%, and 11.56\% on FLICKR-25K, NUS-WIDE, and MS-COCO, respectively. The code is available at https://github.com/xunguangwang/CgAT.

摘要: 深度哈希法以其高效、高效的特点在海量图像检索中得到了广泛应用。然而，深度哈希模型很容易受到敌意例子的攻击，因此有必要开发针对图像检索的对抗性防御方法。现有的解决方案由于使用弱对抗性样本进行训练，并且缺乏区分优化目标来学习稳健特征，使得防御性能受到限制。本文提出了一种基于最小-最大值的中心引导敌意训练算法，即CgAT，通过最坏的敌意例子来提高深度哈希网络的健壮性。具体地说，我们首先将中心代码定义为输入图像内容的语义区分代表，它保留了与正例的语义相似性和与反例的不相似性。我们证明了一个数学公式可以立即计算中心代码。在获得深度散列网络每次优化迭代的中心代码后，将其用于指导对抗性训练过程。一方面，CgAT通过最大化对抗性示例的哈希码与中心码之间的汉明距离来生成最差的对抗性示例作为扩充数据。另一方面，CgAT通过最小化到中心码的汉明距离来学习减轻对抗性样本的影响。在基准数据集上的大量实验表明，我们的对抗性训练算法在防御基于深度散列的检索的对抗性攻击方面是有效的。与当前最先进的防御方法相比，我们在Flickr-25K、NUS-wide和MS-CoCo上的防御性能分别平均提高了18.61、12.35和11.56。代码可在https://github.com/xunguangwang/CgAT.上获得



## **31. Membership Inference Attacks against Diffusion Models**

针对扩散模型的成员推理攻击 cs.CR

**SubmitDate**: 2023-03-22    [abs](http://arxiv.org/abs/2302.03262v2) [paper-pdf](http://arxiv.org/pdf/2302.03262v2)

**Authors**: Tomoya Matsumoto, Takayuki Miura, Naoto Yanai

**Abstract**: Diffusion models have attracted attention in recent years as innovative generative models. In this paper, we investigate whether a diffusion model is resistant to a membership inference attack, which evaluates the privacy leakage of a machine learning model. We primarily discuss the diffusion model from the standpoints of comparison with a generative adversarial network (GAN) as conventional models and hyperparameters unique to the diffusion model, i.e., time steps, sampling steps, and sampling variances. We conduct extensive experiments with DDIM as a diffusion model and DCGAN as a GAN on the CelebA and CIFAR-10 datasets in both white-box and black-box settings and then confirm if the diffusion model is comparably resistant to a membership inference attack as GAN. Next, we demonstrate that the impact of time steps is significant and intermediate steps in a noise schedule are the most vulnerable to the attack. We also found two key insights through further analysis. First, we identify that DDIM is vulnerable to the attack for small sample sizes instead of achieving a lower FID. Second, sampling steps in hyperparameters are important for resistance to the attack, whereas the impact of sampling variances is quite limited.

摘要: 扩散模型作为一种创新的生成性模型，近年来引起了人们的广泛关注。在本文中，我们研究了扩散模型是否抵抗成员推理攻击，以评估机器学习模型的隐私泄漏。我们主要从与生成性对抗网络(GAN)的传统模型和扩散模型特有的超参数，即时间步长、采样步长和采样方差的比较的角度来讨论扩散模型。我们使用DDIM作为扩散模型，DCGAN作为GaN，在CelebA和CIFAR-10数据集上进行了大量的白盒和黑盒环境下的实验，验证了扩散模型作为GaN是否具有同样的抗成员推理攻击的能力。接下来，我们证明了时间步长的影响是显著的，并且噪声调度中的中间步骤最容易受到攻击。通过进一步的分析，我们还发现了两个关键的见解。首先，我们发现ddim在样本大小较小的情况下容易受到攻击，而不是实现较低的FID。其次，超参数中的采样步长对于抵抗攻击很重要，而采样方差的影响相当有限。



## **32. Do Backdoors Assist Membership Inference Attacks?**

后门程序是否有助于成员资格推断攻击？ cs.CR

**SubmitDate**: 2023-03-22    [abs](http://arxiv.org/abs/2303.12589v1) [paper-pdf](http://arxiv.org/pdf/2303.12589v1)

**Authors**: Yumeki Goto, Nami Ashizawa, Toshiki Shibahara, Naoto Yanai

**Abstract**: When an adversary provides poison samples to a machine learning model, privacy leakage, such as membership inference attacks that infer whether a sample was included in the training of the model, becomes effective by moving the sample to an outlier. However, the attacks can be detected because inference accuracy deteriorates due to poison samples. In this paper, we discuss a \textit{backdoor-assisted membership inference attack}, a novel membership inference attack based on backdoors that return the adversary's expected output for a triggered sample. We found three crucial insights through experiments with an academic benchmark dataset. We first demonstrate that the backdoor-assisted membership inference attack is unsuccessful. Second, when we analyzed loss distributions to understand the reason for the unsuccessful results, we found that backdoors cannot separate loss distributions of training and non-training samples. In other words, backdoors cannot affect the distribution of clean samples. Third, we also show that poison and triggered samples activate neurons of different distributions. Specifically, backdoors make any clean sample an inlier, contrary to poisoning samples. As a result, we confirm that backdoors cannot assist membership inference.

摘要: 当敌手向机器学习模型提供有毒样本时，隐私泄露通过将样本移动到离群点而变得有效，例如推断样本是否包括在模型的训练中的成员关系推理攻击。然而，这些攻击是可以检测到的，因为由于毒物样本的原因，推断的准确性会恶化。本文讨论了一种新的基于后门的成员关系推理攻击--后门辅助成员关系推理攻击，它返回对手对触发样本的期望输出。通过对一个学术基准数据集的实验，我们发现了三个关键的见解。我们首先证明了后门辅助的成员推理攻击是不成功的。其次，当我们分析损失分布以了解不成功结果的原因时，我们发现后门不能分离训练样本和非训练样本的损失分布。换句话说，后门不能影响干净样本的分布。第三，我们还表明，毒物和触发样本激活了不同分布的神经元。具体地说，后门使任何干净的样本成为Inlier，与中毒样本相反。因此，我们确认后门不能帮助成员推断。



## **33. Autonomous Intelligent Cyber-defense Agent (AICA) Reference Architecture. Release 2.0**

自主智能网络防御代理(AICA)参考体系结构。版本2.0 cs.CR

This is a major revision and extension of the earlier release of AICA  Reference Architecture

**SubmitDate**: 2023-03-22    [abs](http://arxiv.org/abs/1803.10664v3) [paper-pdf](http://arxiv.org/pdf/1803.10664v3)

**Authors**: Alexander Kott, Paul Théron, Martin Drašar, Edlira Dushku, Benoît LeBlanc, Paul Losiewicz, Alessandro Guarino, Luigi Mancini, Agostino Panico, Mauno Pihelgas, Krzysztof Rzadca, Fabio De Gaspari

**Abstract**: This report - a major revision of its previous release - describes a reference architecture for intelligent software agents performing active, largely autonomous cyber-defense actions on military networks of computing and communicating devices. The report is produced by the North Atlantic Treaty Organization (NATO) Research Task Group (RTG) IST-152 "Intelligent Autonomous Agents for Cyber Defense and Resilience". In a conflict with a technically sophisticated adversary, NATO military tactical networks will operate in a heavily contested battlefield. Enemy software cyber agents - malware - will infiltrate friendly networks and attack friendly command, control, communications, computers, intelligence, surveillance, and reconnaissance and computerized weapon systems. To fight them, NATO needs artificial cyber hunters - intelligent, autonomous, mobile agents specialized in active cyber defense. With this in mind, in 2016, NATO initiated RTG IST-152. Its objective has been to help accelerate the development and transition to practice of such software agents by producing a reference architecture and technical roadmap. This report presents the concept and architecture of an Autonomous Intelligent Cyber-defense Agent (AICA). We describe the rationale of the AICA concept, explain the methodology and purpose that drive the definition of the AICA Reference Architecture, and review some of the main features and challenges of AICAs.

摘要: 这份报告是对之前发布的报告的重大修订，它描述了智能软件代理在军事计算和通信设备网络上执行主动的、基本上自主的网络防御行动的参考架构。该报告由北大西洋公约组织(NATO)研究任务组(RTG)IST-152《用于网络防御和弹性的智能自主代理》编制。在与技术复杂的对手的冲突中，北约军事战术网络将在一个竞争激烈的战场上运作。敌方软件网络代理--恶意软件--将渗透到友好的网络中，攻击友好的指挥、控制、通信、计算机、情报、监视以及侦察和计算机化的武器系统。为了打击他们，北约需要人工网络猎人--专门从事主动网络防御的智能、自主、移动代理。考虑到这一点，北约于2016年启动了RTG IST-152。它的目标一直是通过产生参考体系结构和技术路线图来帮助加速此类软件代理的开发和向实践的过渡。本文提出了一种自主智能网络防御代理的概念和体系结构。我们描述了AICA概念的基本原理，解释了驱动AICA参考体系结构定义的方法和目的，并回顾了AICA的一些主要特征和挑战。



## **34. Sibling-Attack: Rethinking Transferable Adversarial Attacks against Face Recognition**

兄弟攻击：重新思考针对人脸识别的可转移对抗性攻击 cs.CV

8 pages, 5 fivures, accepted by CVPR 2023 as a poster paper

**SubmitDate**: 2023-03-22    [abs](http://arxiv.org/abs/2303.12512v1) [paper-pdf](http://arxiv.org/pdf/2303.12512v1)

**Authors**: Zexin Li, Bangjie Yin, Taiping Yao, Juefeng Guo, Shouhong Ding, Simin Chen, Cong Liu

**Abstract**: A hard challenge in developing practical face recognition (FR) attacks is due to the black-box nature of the target FR model, i.e., inaccessible gradient and parameter information to attackers. While recent research took an important step towards attacking black-box FR models through leveraging transferability, their performance is still limited, especially against online commercial FR systems that can be pessimistic (e.g., a less than 50% ASR--attack success rate on average). Motivated by this, we present Sibling-Attack, a new FR attack technique for the first time explores a novel multi-task perspective (i.e., leveraging extra information from multi-correlated tasks to boost attacking transferability). Intuitively, Sibling-Attack selects a set of tasks correlated with FR and picks the Attribute Recognition (AR) task as the task used in Sibling-Attack based on theoretical and quantitative analysis. Sibling-Attack then develops an optimization framework that fuses adversarial gradient information through (1) constraining the cross-task features to be under the same space, (2) a joint-task meta optimization framework that enhances the gradient compatibility among tasks, and (3) a cross-task gradient stabilization method which mitigates the oscillation effect during attacking. Extensive experiments demonstrate that Sibling-Attack outperforms state-of-the-art FR attack techniques by a non-trivial margin, boosting ASR by 12.61% and 55.77% on average on state-of-the-art pre-trained FR models and two well-known, widely used commercial FR systems.

摘要: 由于目标人脸识别(FR)模型的黑箱性质，即攻击者无法获得梯度和参数信息，因此开发实际的人脸识别(FR)攻击是一个困难的挑战。虽然最近的研究在利用可转移性攻击黑盒FR模型方面迈出了重要的一步，但它们的性能仍然有限，特别是对可能是悲观的在线商业FR系统(例如，平均ASR攻击成功率不到50%)。受此启发，我们首次提出了兄弟攻击，这是一种新的FR攻击技术，它探索了一种新的多任务视角(即，利用来自多相关任务的额外信息来提高攻击的可转移性)。基于理论和定量的分析，兄弟攻击直观地选择了一组与FR相关的任务，并选择了属性识别(AR)任务作为兄弟攻击中使用的任务。兄弟攻击通过(1)将跨任务特征约束在同一空间内；(2)联合任务元优化框架，增强任务间的梯度兼容性；(3)跨任务梯度稳定方法，缓解攻击过程中的振荡效应，从而融合对抗性梯度信息。广泛的实验表明，兄弟攻击的性能远远超过了最先进的FR攻击技术，在最先进的预训练FR模型和两个著名的、广泛使用的商业FR系统上，ASR平均提高了12.61%和55.77%。



## **35. Revisiting DeepFool: generalization and improvement**

重温DeepFool：泛化与改进 cs.LG

**SubmitDate**: 2023-03-22    [abs](http://arxiv.org/abs/2303.12481v1) [paper-pdf](http://arxiv.org/pdf/2303.12481v1)

**Authors**: Alireza Abdollahpourrostam, Mahed Abroshan, Seyed-Mohsen Moosavi-Dezfooli

**Abstract**: Deep neural networks have been known to be vulnerable to adversarial examples, which are inputs that are modified slightly to fool the network into making incorrect predictions. This has led to a significant amount of research on evaluating the robustness of these networks against such perturbations. One particularly important robustness metric is the robustness to minimal l2 adversarial perturbations. However, existing methods for evaluating this robustness metric are either computationally expensive or not very accurate. In this paper, we introduce a new family of adversarial attacks that strike a balance between effectiveness and computational efficiency. Our proposed attacks are generalizations of the well-known DeepFool (DF) attack, while they remain simple to understand and implement. We demonstrate that our attacks outperform existing methods in terms of both effectiveness and computational efficiency. Our proposed attacks are also suitable for evaluating the robustness of large models and can be used to perform adversarial training (AT) to achieve state-of-the-art robustness to minimal l2 adversarial perturbations.

摘要: 众所周知，深度神经网络很容易受到敌意例子的攻击，这些例子是对输入进行了轻微修改，以愚弄网络做出错误的预测。这导致了大量关于评估这些网络对此类扰动的稳健性的研究。一个特别重要的稳健性度量是对最小的L2对抗扰动的稳健性。然而，现有的评估这种稳健性度量的方法要么计算昂贵，要么不太准确。在本文中，我们引入了一类新的对抗性攻击，它们在有效性和计算效率之间取得了平衡。我们提出的攻击是众所周知的DeepFool(DF)攻击的推广，但它们仍然易于理解和实现。我们证明了我们的攻击在有效性和计算效率方面都优于现有的方法。我们提出的攻击也适用于评估大型模型的稳健性，并可用于执行对抗训练(AT)以获得对最小L2对抗扰动的最先进的稳健性。



## **36. Distribution-restrained Softmax Loss for the Model Robustness**

用于模型稳健性的分布约束的软最大损失 cs.LG

**SubmitDate**: 2023-03-22    [abs](http://arxiv.org/abs/2303.12363v1) [paper-pdf](http://arxiv.org/pdf/2303.12363v1)

**Authors**: Hao Wang, Chen Li, Jinzhe Jiang, Xin Zhang, Yaqian Zhao, Weifeng Gong

**Abstract**: Recently, the robustness of deep learning models has received widespread attention, and various methods for improving model robustness have been proposed, including adversarial training, model architecture modification, design of loss functions, certified defenses, and so on. However, the principle of the robustness to attacks is still not fully understood, also the related research is still not sufficient. Here, we have identified a significant factor that affects the robustness of models: the distribution characteristics of softmax values for non-real label samples. We found that the results after an attack are highly correlated with the distribution characteristics, and thus we proposed a loss function to suppress the distribution diversity of softmax. A large number of experiments have shown that our method can improve robustness without significant time consumption.

摘要: 近年来，深度学习模型的稳健性受到了广泛的关注，并提出了各种提高模型稳健性的方法，包括对抗性训练、模型结构修改、损失函数设计、认证防御等。然而，对攻击的稳健性原理还没有完全了解，相关的研究也还不够充分。在这里，我们已经确定了一个影响模型稳健性的重要因素：非真实标签样本的Softmax的分布特征。我们发现攻击后的结果与分布特征高度相关，因此我们提出了一种损失函数来抑制Softmax的分布多样性。大量实验表明，该方法可以在不消耗大量时间的情况下提高稳健性。



## **37. Wasserstein Adversarial Examples on Univariant Time Series Data**

单变量时间序列数据的Wasserstein对抗性实例 cs.LG

**SubmitDate**: 2023-03-22    [abs](http://arxiv.org/abs/2303.12357v1) [paper-pdf](http://arxiv.org/pdf/2303.12357v1)

**Authors**: Wenjie Wang, Li Xiong, Jian Lou

**Abstract**: Adversarial examples are crafted by adding indistinguishable perturbations to normal examples in order to fool a well-trained deep learning model to misclassify. In the context of computer vision, this notion of indistinguishability is typically bounded by $L_{\infty}$ or other norms. However, these norms are not appropriate for measuring indistinguishiability for time series data. In this work, we propose adversarial examples in the Wasserstein space for time series data for the first time and utilize Wasserstein distance to bound the perturbation between normal examples and adversarial examples. We introduce Wasserstein projected gradient descent (WPGD), an adversarial attack method for perturbing univariant time series data. We leverage the closed-form solution of Wasserstein distance in the 1D space to calculate the projection step of WPGD efficiently with the gradient descent method. We further propose a two-step projection so that the search of adversarial examples in the Wasserstein space is guided and constrained by Euclidean norms to yield more effective and imperceptible perturbations. We empirically evaluate the proposed attack on several time series datasets in the healthcare domain. Extensive results demonstrate that the Wasserstein attack is powerful and can successfully attack most of the target classifiers with a high attack success rate. To better study the nature of Wasserstein adversarial example, we evaluate a strong defense mechanism named Wasserstein smoothing for potential certified robustness defense. Although the defense can achieve some accuracy gain, it still has limitations in many cases and leaves space for developing a stronger certified robustness method to Wasserstein adversarial examples on univariant time series data.

摘要: 对抗性示例是通过在正常示例中添加无法区分的扰动来制作的，以便愚弄训练有素的深度学习模型进行错误分类。在计算机视觉的背景下，这种不可区分的概念通常受到$L_(\inty)$或其他规范的限制。然而，这些标准不适合用来衡量时间序列数据的不可区分性。在这项工作中，我们首次在时间序列数据的Wasserstein空间中提出了对抗性样本，并利用Wasserstein距离来界定正态样本和对抗性样本之间的扰动。介绍了一种针对单变量时间序列数据的对抗性攻击方法--Wasserstein投影梯度下降(WPGD)方法。利用一维空间中Wasserstein距离的闭合解，利用梯度下降法有效地计算了WPGD的投影步长。我们进一步提出了一个两步投影，使得在Wasserstein空间中的对抗性例子的搜索由欧几里得范数来指导和约束，从而产生更有效和更不可察觉的扰动。我们在医疗保健领域的几个时间序列数据集上对所提出的攻击进行了经验评估。大量实验结果表明，Wasserstein攻击具有较强的攻击能力，能够成功攻击大部分目标分类器，攻击成功率较高。为了更好地研究Wasserstein对抗例子的性质，我们评估了一种名为Wasserstein平滑的强防御机制，以实现潜在的认证稳健性防御。虽然防御方法可以获得一定的精度收益，但在很多情况下仍然存在局限性，并为开发一种对单变量时间序列数据上的Wasserstein对抗性实例具有更强的证明稳健性的方法留下了空间。



## **38. Bankrupting Sybil Despite Churn**

尽管员工流失，Sybil仍在破产 cs.CR

41 pages, 6 figures. arXiv admin note: text overlap with  arXiv:2006.02893, arXiv:1911.06462

**SubmitDate**: 2023-03-22    [abs](http://arxiv.org/abs/2010.06834v4) [paper-pdf](http://arxiv.org/pdf/2010.06834v4)

**Authors**: Diksha Gupta, Jared Saia, Maxwell Young

**Abstract**: A Sybil attack occurs when an adversary controls multiple identifiers (IDs) in a system. Limiting the number of Sybil (bad) IDs to a minority is critical to the use of well-established tools for tolerating malicious behavior, such as Byzantine agreement and secure multiparty computation.   A popular technique for enforcing a Sybil minority is resource burning: the verifiable consumption of a network resource, such as computational power, bandwidth, or memory. Unfortunately, typical defenses based on resource burning require non-Sybil (good) IDs to consume at least as many resources as the adversary. Additionally, they have a high resource burning cost, even when the system membership is relatively stable.   Here, we present a new Sybil defense, ERGO, that guarantees (1) there is always a minority of bad IDs; and (2) when the system is under significant attack, the good IDs consume asymptotically less resources than the bad. In particular, for churn rate that can vary exponentially, the resource burning rate for good IDs under ERGO is O(\sqrt{TJ} + J), where T is the resource burning rate of the adversary, and J is the join rate of good IDs. We show this resource burning rate is asymptotically optimal for a large class of algorithms.   We empirically evaluate ERGO alongside prior Sybil defenses. Additionally, we show that ERGO can be combined with machine learning techniques for classifying Sybil IDs, while preserving its theoretical guarantees. Based on our experiments comparing ERGO with two previous Sybil defenses, ERGO improves on the amount of resource burning relative to the adversary by up to 2 orders of magnitude without machine learning, and up to 3 orders of magnitude using machine learning.

摘要: 当对手控制系统中的多个标识符(ID)时，就会发生Sybil攻击。将Sybil(BAD)ID的数量限制为少数，对于使用成熟的工具容忍恶意行为至关重要，例如拜占庭协议和安全多方计算。实施少数Sybil的一种流行技术是资源烧毁：可验证的网络资源消耗，如计算能力、带宽或内存。不幸的是，基于资源燃烧的典型防御需要非Sybil(好)ID至少消耗与对手一样多的资源。此外，即使在系统成员相对稳定的情况下，它们也具有较高的资源消耗成本。在这里，我们提出了一种新的Sybil防御方案ERGO，它保证(1)总是有少数坏ID；(2)当系统受到重大攻击时，好ID消耗的资源逐渐少于坏ID。特别地，对于可以指数变化的流失率，ERGO下好的ID的资源烧失率为O(Sqrt{tj}+J)，其中T是对手的资源烧失率，J是好的ID的加入率。对于一大类算法，我们证明了这种资源消耗速度是渐近最优的。我们对ERGO和之前的Sybil防御进行了经验评估。此外，我们证明了ERGO可以与机器学习技术相结合来对Sybil ID进行分类，同时保持其理论保证。基于我们的实验比较了ERGO和之前的两个Sybil防御措施，ERGO在没有机器学习的情况下相对于对手提高了高达2个数量级的资源消耗量，使用机器学习的资源消耗量提高了高达3个数量级。



## **39. X-CANIDS: Signal-Aware Explainable Intrusion Detection System for Controller Area Network-Based In-Vehicle Network**

X-CANIDS：基于控制器局域网的车载网络信号感知可解释入侵检测系统 cs.CR

**SubmitDate**: 2023-03-22    [abs](http://arxiv.org/abs/2303.12278v1) [paper-pdf](http://arxiv.org/pdf/2303.12278v1)

**Authors**: Seonghoon Jeong, Sangho Lee, Hwejae Lee, Huy Kang Kim

**Abstract**: Controller Area Network (CAN) is an essential networking protocol that connects multiple electronic control units (ECUs) in a vehicle. However, CAN-based in-vehicle networks (IVNs) face security risks owing to the CAN mechanisms. An adversary can sabotage a vehicle by leveraging the security risks if they can access the CAN bus. Thus, recent actions and cybersecurity regulations (e.g., UNR 155) require carmakers to implement intrusion detection systems (IDSs) in their vehicles. An IDS should detect cyberattacks and provide a forensic capability to analyze attacks. Although many IDSs have been proposed, considerations regarding their feasibility and explainability remain lacking. This study proposes X-CANIDS, which is a novel IDS for CAN-based IVNs. X-CANIDS dissects the payloads in CAN messages into human-understandable signals using a CAN database. The signals improve the intrusion detection performance compared with the use of bit representations of raw payloads. These signals also enable an understanding of which signal or ECU is under attack. X-CANIDS can detect zero-day attacks because it does not require any labeled dataset in the training phase. We confirmed the feasibility of the proposed method through a benchmark test on an automotive-grade embedded device with a GPU. The results of this work will be valuable to carmakers and researchers considering the installation of in-vehicle IDSs for their vehicles.

摘要: 控制器局域网(CAN)是连接车辆中多个电子控制单元(ECU)的基本网络协议。然而，由于CAN机制的存在，基于CAN的车载网络面临着安全隐患。如果对手可以访问CAN总线，则他们可以利用安全风险来破坏车辆。因此，最近的行动和网络安全法规(例如，UNR 155)要求汽车制造商在其车辆中安装入侵检测系统(IDS)。入侵检测系统应检测网络攻击并提供分析攻击的取证能力。虽然已经提出了许多入侵检测系统，但仍然缺乏对其可行性和可解释性的考虑。本研究提出了一种新型的基于CAN网络的入侵检测系统--X-CANID。X-Canids使用CAN数据库将CAN消息中的有效载荷分解为人类可理解的信号。与使用原始有效载荷的比特表示相比，该信号提高了入侵检测性能。这些信号还使您能够了解哪个信号或ECU受到攻击。X-CARID可以检测零日攻击，因为它在训练阶段不需要任何标记的数据集。通过在一款搭载GPU的车载嵌入式设备上的基准测试，验证了该方法的可行性。这项工作的结果将对汽车制造商和考虑为其车辆安装车载入侵检测系统的研究人员具有价值。



## **40. State-of-the-art optical-based physical adversarial attacks for deep learning computer vision systems**

深度学习计算机视觉系统中基于光学的物理对抗攻击 cs.CV

This work has been submitted to the IEEE for possible publication.  Copyright may be transferred without notice, after which this version may no  longer be accessible

**SubmitDate**: 2023-03-22    [abs](http://arxiv.org/abs/2303.12249v1) [paper-pdf](http://arxiv.org/pdf/2303.12249v1)

**Authors**: Junbin Fang, You Jiang, Canjian Jiang, Zoe L. Jiang, Siu-Ming Yiu, Chuanyi Liu

**Abstract**: Adversarial attacks can mislead deep learning models to make false predictions by implanting small perturbations to the original input that are imperceptible to the human eye, which poses a huge security threat to the computer vision systems based on deep learning. Physical adversarial attacks, which is more realistic, as the perturbation is introduced to the input before it is being captured and converted to a binary image inside the vision system, when compared to digital adversarial attacks. In this paper, we focus on physical adversarial attacks and further classify them into invasive and non-invasive. Optical-based physical adversarial attack techniques (e.g. using light irradiation) belong to the non-invasive category. As the perturbations can be easily ignored by humans as the perturbations are very similar to the effects generated by a natural environment in the real world. They are highly invisibility and executable and can pose a significant or even lethal threats to real systems. This paper focuses on optical-based physical adversarial attack techniques for computer vision systems, with emphasis on the introduction and discussion of optical-based physical adversarial attack techniques.

摘要: 对抗性攻击通过在原始输入中植入人眼无法察觉的微小扰动来误导深度学习模型做出错误预测，这对基于深度学习的计算机视觉系统构成了巨大的安全威胁。物理对抗性攻击，与数字对抗性攻击相比，这更现实，因为与数字对抗性攻击相比，输入在被捕获并转换为视觉系统内的二进制图像之前被引入扰动。在本文中，我们将重点放在物理对抗攻击上，并进一步将其分为侵入性攻击和非侵入式攻击。基于光学的物理对抗攻击技术(例如使用光照射)属于非侵入性范畴。因为扰动很容易被人类忽略，因为扰动与现实世界中的自然环境产生的效果非常相似。它们具有高度的隐蔽性和可执行性，可以对真实系统构成重大甚至致命的威胁。本文研究了计算机视觉系统中基于光学的物理对抗攻击技术，重点介绍和讨论了基于光学的物理对抗攻击技术。



## **41. Task-Oriented Communications for NextG: End-to-End Deep Learning and AI Security Aspects**

面向NextG的面向任务的通信：端到端深度学习和AI安全方面 cs.NI

**SubmitDate**: 2023-03-21    [abs](http://arxiv.org/abs/2212.09668v2) [paper-pdf](http://arxiv.org/pdf/2212.09668v2)

**Authors**: Yalin E. Sagduyu, Sennur Ulukus, Aylin Yener

**Abstract**: Communications systems to date are primarily designed with the goal of reliable transfer of digital sequences (bits). Next generation (NextG) communication systems are beginning to explore shifting this design paradigm to reliably executing a given task such as in task-oriented communications. In this paper, wireless signal classification is considered as the task for the NextG Radio Access Network (RAN), where edge devices collect wireless signals for spectrum awareness and communicate with the NextG base station (gNodeB) that needs to identify the signal label. Edge devices may not have sufficient processing power and may not be trusted to perform the signal classification task, whereas the transfer of signals to the gNodeB may not be feasible due to stringent delay, rate, and energy restrictions. Task-oriented communications is considered by jointly training the transmitter, receiver and classifier functionalities as an encoder-decoder pair for the edge device and the gNodeB. This approach improves the accuracy compared to the separated case of signal transfer followed by classification. Adversarial machine learning poses a major security threat to the use of deep learning for task-oriented communications. A major performance loss is shown when backdoor (Trojan) and adversarial (evasion) attacks target the training and test processes of task-oriented communications.

摘要: 迄今为止，通信系统的主要设计目标是可靠地传输数字序列(比特)。下一代(NextG)通信系统正开始探索将这种设计范例转变为可靠地执行给定任务，例如在面向任务的通信中。本文将无线信号分类作为下一代无线接入网(RAN)的任务，边缘设备采集无线信号以实现频谱感知，并与需要识别信号标签的下一代基站(GNodeB)进行通信。边缘设备可能没有足够的处理能力，并且可能不被信任来执行信号分类任务，而由于严格的延迟、速率和能量限制，向gNodeB传输信号可能是不可行的。通过将发送器、接收器和分类器功能联合训练为用于边缘设备和gNodeB的编解码器对来考虑面向任务的通信。与信号传输后分类的分离情况相比，该方法提高了精度。对抗性机器学习对使用深度学习进行面向任务的通信构成了主要的安全威胁。当后门(特洛伊木马)和敌意(规避)攻击以面向任务的通信的训练和测试过程为目标时，会显示出重大的性能损失。



## **42. Towards Compositional Adversarial Robustness: Generalizing Adversarial Training to Composite Semantic Perturbations**

走向成分对抗稳健性：将对抗训练推广到复合语义扰动 cs.CV

CVPR 2023. The research demo is at https://hsiung.cc/CARBEN/

**SubmitDate**: 2023-03-21    [abs](http://arxiv.org/abs/2202.04235v3) [paper-pdf](http://arxiv.org/pdf/2202.04235v3)

**Authors**: Lei Hsiung, Yun-Yun Tsai, Pin-Yu Chen, Tsung-Yi Ho

**Abstract**: Model robustness against adversarial examples of single perturbation type such as the $\ell_{p}$-norm has been widely studied, yet its generalization to more realistic scenarios involving multiple semantic perturbations and their composition remains largely unexplored. In this paper, we first propose a novel method for generating composite adversarial examples. Our method can find the optimal attack composition by utilizing component-wise projected gradient descent and automatic attack-order scheduling. We then propose generalized adversarial training (GAT) to extend model robustness from $\ell_{p}$-ball to composite semantic perturbations, such as the combination of Hue, Saturation, Brightness, Contrast, and Rotation. Results obtained using ImageNet and CIFAR-10 datasets indicate that GAT can be robust not only to all the tested types of a single attack, but also to any combination of such attacks. GAT also outperforms baseline $\ell_{\infty}$-norm bounded adversarial training approaches by a significant margin.

摘要: 针对单一扰动类型的对抗性实例，如$ellp-范数，模型的稳健性已经得到了广泛的研究，但它对涉及多个语义扰动及其组成的更现实场景的推广仍在很大程度上有待探索。在本文中，我们首先提出了一种生成复合对抗性实例的新方法。该方法利用基于组件的投影梯度下降和自动攻击顺序调度来寻找最优的攻击组合。然后，我们提出了广义对抗性训练(GAT)来扩展模型的稳健性，将模型的稳健性从球状扩展到复合语义扰动，如色调、饱和度、亮度、对比度和旋转的组合。使用ImageNet和CIFAR-10数据集获得的结果表明，GAT不仅对所有测试类型的单一攻击，而且对此类攻击的任何组合都具有健壮性。GAT的性能也大大超过了基准范数有界的对抗性训练方法。



## **43. Efficient Decision-based Black-box Patch Attacks on Video Recognition**

视频识别中高效的基于决策的黑盒补丁攻击 cs.CV

**SubmitDate**: 2023-03-21    [abs](http://arxiv.org/abs/2303.11917v1) [paper-pdf](http://arxiv.org/pdf/2303.11917v1)

**Authors**: Kaixun Jiang, Zhaoyu Chen, Tony Huang, Jiafeng Wang, Dingkang Yang, Bo Li, Yan Wang, Wenqiang Zhang

**Abstract**: Although Deep Neural Networks (DNNs) have demonstrated excellent performance, they are vulnerable to adversarial patches that introduce perceptible and localized perturbations to the input. Generating adversarial patches on images has received much attention, while adversarial patches on videos have not been well investigated. Further, decision-based attacks, where attackers only access the predicted hard labels by querying threat models, have not been well explored on video models either, even if they are practical in real-world video recognition scenes. The absence of such studies leads to a huge gap in the robustness assessment for video models. To bridge this gap, this work first explores decision-based patch attacks on video models. We analyze that the huge parameter space brought by videos and the minimal information returned by decision-based models both greatly increase the attack difficulty and query burden. To achieve a query-efficient attack, we propose a spatial-temporal differential evolution (STDE) framework. First, STDE introduces target videos as patch textures and only adds patches on keyframes that are adaptively selected by temporal difference. Second, STDE takes minimizing the patch area as the optimization objective and adopts spatialtemporal mutation and crossover to search for the global optimum without falling into the local optimum. Experiments show STDE has demonstrated state-of-the-art performance in terms of threat, efficiency and imperceptibility. Hence, STDE has the potential to be a powerful tool for evaluating the robustness of video recognition models.

摘要: 尽管深度神经网络(DNN)表现出了很好的性能，但它们很容易受到敌意补丁的攻击，这些补丁会给输入带来可感知的局部扰动。在图像上生成敌意补丁已经得到了很大的关注，而视频上的敌意补丁还没有得到很好的研究。此外，基于决策的攻击(攻击者仅通过查询威胁模型来访问预测的硬标签)在视频模型上也没有得到很好的探索，即使它们在现实世界的视频识别场景中是实用的。这类研究的缺乏导致了视频模型稳健性评估的巨大差距。为了弥补这一差距，这项工作首先探索了基于决策的视频模型补丁攻击。分析了视频带来的巨大参数空间和基于决策的模型返回的最小信息量都大大增加了攻击难度和查询负担。为了实现查询高效的攻击，我们提出了一种时空差异进化(STDE)框架。首先，STDE将目标视频作为补丁纹理引入，只在根据时间差异自适应选择的关键帧上添加补丁。其次，STDE算法以面片面积最小为优化目标，采用时空变异和交叉来搜索全局最优解而不陷入局部最优。实验表明，STDE在威胁、效率和不可感知性方面都表现出了最先进的性能。因此，STDE有可能成为评估视频识别模型稳健性的有力工具。



## **44. The Threat of Adversarial Attacks on Machine Learning in Network Security -- A Survey**

网络安全中对抗性攻击对机器学习的威胁--综述 cs.CR

**SubmitDate**: 2023-03-21    [abs](http://arxiv.org/abs/1911.02621v3) [paper-pdf](http://arxiv.org/pdf/1911.02621v3)

**Authors**: Olakunle Ibitoye, Rana Abou-Khamis, Mohamed el Shehaby, Ashraf Matrawy, M. Omair Shafiq

**Abstract**: Machine learning models have made many decision support systems to be faster, more accurate, and more efficient. However, applications of machine learning in network security face a more disproportionate threat of active adversarial attacks compared to other domains. This is because machine learning applications in network security such as malware detection, intrusion detection, and spam filtering are by themselves adversarial in nature. In what could be considered an arm's race between attackers and defenders, adversaries constantly probe machine learning systems with inputs that are explicitly designed to bypass the system and induce a wrong prediction. In this survey, we first provide a taxonomy of machine learning techniques, tasks, and depth. We then introduce a classification of machine learning in network security applications. Next, we examine various adversarial attacks against machine learning in network security and introduce two classification approaches for adversarial attacks in network security. First, we classify adversarial attacks in network security based on a taxonomy of network security applications. Secondly, we categorize adversarial attacks in network security into a problem space vs feature space dimensional classification model. We then analyze the various defenses against adversarial attacks on machine learning-based network security applications. We conclude by introducing an adversarial risk grid map and evaluating several existing adversarial attacks against machine learning in network security using the risk grid map. We also identify where each attack classification resides within the adversarial risk grid map.

摘要: 机器学习模型使许多决策支持系统变得更快、更准确、更高效。然而，与其他领域相比，机器学习在网络安全中的应用面临着更不成比例的主动对抗攻击威胁。这是因为网络安全中的机器学习应用程序，如恶意软件检测、入侵检测和垃圾邮件过滤，本身就是对抗性的。在这场可以被视为攻击者和防御者之间的军备竞赛中，对手不断地探查机器学习系统，其输入显然是为了绕过系统，并导致错误的预测。在这次调查中，我们首先提供了机器学习技术、任务和深度的分类。然后介绍了机器学习在网络安全应用中的分类。接下来，我们考察了网络安全中各种针对机器学习的对抗性攻击，并介绍了网络安全中对抗性攻击的两种分类方法。首先，根据网络安全应用的分类，对网络安全中的敌意攻击进行分类。其次，将网络安全中的对抗性攻击归类到问题空间与特征空间的维度分类模型中。然后，我们分析了各种针对基于机器学习的网络安全应用的对抗性攻击的防御措施。最后，我们引入了对抗性风险网格图，并利用该风险网格图对网络安全中现有的几种针对机器学习的对抗性攻击进行了评估。我们还确定了每个攻击分类在对抗性风险网格地图中的位置。



## **45. OTJR: Optimal Transport Meets Optimal Jacobian Regularization for Adversarial Robustness**

OTJR：最优传输满足最优雅可比正则化的对抗性 cs.CV

**SubmitDate**: 2023-03-21    [abs](http://arxiv.org/abs/2303.11793v1) [paper-pdf](http://arxiv.org/pdf/2303.11793v1)

**Authors**: Binh M. Le, Shahroz Tariq, Simon S. Woo

**Abstract**: Deep neural networks are widely recognized as being vulnerable to adversarial perturbation. To overcome this challenge, developing a robust classifier is crucial. So far, two well-known defenses have been adopted to improve the learning of robust classifiers, namely adversarial training (AT) and Jacobian regularization. However, each approach behaves differently against adversarial perturbations. First, our work carefully analyzes and characterizes these two schools of approaches, both theoretically and empirically, to demonstrate how each approach impacts the robust learning of a classifier. Next, we propose our novel Optimal Transport with Jacobian regularization method, dubbed OTJR, jointly incorporating the input-output Jacobian regularization into the AT by leveraging the optimal transport theory. In particular, we employ the Sliced Wasserstein (SW) distance that can efficiently push the adversarial samples' representations closer to those of clean samples, regardless of the number of classes within the dataset. The SW distance provides the adversarial samples' movement directions, which are much more informative and powerful for the Jacobian regularization. Our extensive experiments demonstrate the effectiveness of our proposed method, which jointly incorporates Jacobian regularization into AT. Furthermore, we demonstrate that our proposed method consistently enhances the model's robustness with CIFAR-100 dataset under various adversarial attack settings, achieving up to 28.49% under AutoAttack.

摘要: 深度神经网络被广泛认为容易受到对抗性扰动的影响。要克服这一挑战，开发一个健壮的分类器至关重要。到目前为止，已经采用了两种著名的防御措施来改进稳健分类器的学习，即对抗性训练(AT)和雅可比正则化。然而，每种方法在对抗对抗性干扰时的表现都不同。首先，我们的工作仔细地分析和表征了这两个流派的方法，无论是理论上还是经验上，以证明每种方法如何影响分类器的稳健学习。接下来，我们利用最优传输理论，将输入输出的雅可比正则化引入到AT中，提出了一种新的基于雅可比正则化的最优传输方法，称为OTJR。特别是，我们使用了切片Wasserstein(SW)距离，该距离可以有效地将对抗性样本的表示更接近于干净样本的表示，而不管数据集中有多少类。Sw距离提供了对抗性样本的运动方向，为雅可比正则化提供了更多的信息和更强大的能力。我们的大量实验证明了我们提出的方法的有效性，该方法将雅可比正则化联合到AT中。此外，我们还利用CIFAR-100数据集在不同的对抗性攻击环境下验证了该方法的有效性，在AutoAttack环境下达到了28.49%的健壮性。



## **46. Generative AI for Cyber Threat-Hunting in 6G-enabled IoT Networks**

产生式人工智能在支持6G的物联网网络威胁搜索中的应用 cs.CR

The paper is accepted and will be published in the IEEE/ACM CCGrid  2023 Conference Proceedings

**SubmitDate**: 2023-03-21    [abs](http://arxiv.org/abs/2303.11751v1) [paper-pdf](http://arxiv.org/pdf/2303.11751v1)

**Authors**: Mohamed Amine Ferrag, Merouane Debbah, Muna Al-Hawawreh

**Abstract**: The next generation of cellular technology, 6G, is being developed to enable a wide range of new applications and services for the Internet of Things (IoT). One of 6G's main advantages for IoT applications is its ability to support much higher data rates and bandwidth as well as to support ultra-low latency. However, with this increased connectivity will come to an increased risk of cyber threats, as attackers will be able to exploit the large network of connected devices. Generative Artificial Intelligence (AI) can be used to detect and prevent cyber attacks by continuously learning and adapting to new threats and vulnerabilities. In this paper, we discuss the use of generative AI for cyber threat-hunting (CTH) in 6G-enabled IoT networks. Then, we propose a new generative adversarial network (GAN) and Transformer-based model for CTH in 6G-enabled IoT Networks. The experimental analysis results with a new cyber security dataset demonstrate that the Transformer-based security model for CTH can detect IoT attacks with a high overall accuracy of 95%. We examine the challenges and opportunities and conclude by highlighting the potential of generative AI in enhancing the security of 6G-enabled IoT networks and call for further research to be conducted in this area.

摘要: 下一代蜂窝技术6G正在开发中，以支持物联网(IoT)的广泛新应用和服务。6G对物联网应用的主要优势之一是它能够支持更高的数据速率和带宽以及支持超低延迟。然而，随着这种连接的增加，网络威胁的风险将会增加，因为攻击者将能够利用连接设备组成的大型网络。生成性人工智能(AI)可以通过不断学习和适应新的威胁和漏洞来检测和预防网络攻击。在本文中，我们讨论了产生式人工智能在支持6G的物联网网络中用于网络威胁搜索(CTH)。在此基础上，我们提出了一种新的基于生成性对抗网络(GAN)和Transformer的6G物联网Cth模型。在一个新的网络安全数据集上的实验分析结果表明，基于Transformer的Cth安全模型能够检测到物联网攻击，总体准确率达到95%。我们研究了挑战和机遇，最后强调了生成性人工智能在增强支持6G的物联网网络安全方面的潜力，并呼吁在这一领域进行进一步研究。



## **47. Poisoning Attacks in Federated Edge Learning for Digital Twin 6G-enabled IoTs: An Anticipatory Study**

数字双生6G物联网联合边缘学习中的中毒攻击：一项预期研究 cs.CR

The paper is accepted and will be published in the IEEE ICC 2023  Conference Proceedings

**SubmitDate**: 2023-03-21    [abs](http://arxiv.org/abs/2303.11745v1) [paper-pdf](http://arxiv.org/pdf/2303.11745v1)

**Authors**: Mohamed Amine Ferrag, Burak Kantarci, Lucas C. Cordeiro, Merouane Debbah, Kim-Kwang Raymond Choo

**Abstract**: Federated edge learning can be essential in supporting privacy-preserving, artificial intelligence (AI)-enabled activities in digital twin 6G-enabled Internet of Things (IoT) environments. However, we need to also consider the potential of attacks targeting the underlying AI systems (e.g., adversaries seek to corrupt data on the IoT devices during local updates or corrupt the model updates); hence, in this article, we propose an anticipatory study for poisoning attacks in federated edge learning for digital twin 6G-enabled IoT environments. Specifically, we study the influence of adversaries on the training and development of federated learning models in digital twin 6G-enabled IoT environments. We demonstrate that attackers can carry out poisoning attacks in two different learning settings, namely: centralized learning and federated learning, and successful attacks can severely reduce the model's accuracy. We comprehensively evaluate the attacks on a new cyber security dataset designed for IoT applications with three deep neural networks under the non-independent and identically distributed (Non-IID) data and the independent and identically distributed (IID) data. The poisoning attacks, on an attack classification problem, can lead to a decrease in accuracy from 94.93% to 85.98% with IID data and from 94.18% to 30.04% with Non-IID.

摘要: 联合边缘学习对于支持数字孪生6G物联网(IoT)环境中保护隐私、启用人工智能(AI)的活动至关重要。然而，我们还需要考虑针对底层AI系统的攻击的可能性(例如，对手试图在本地更新期间破坏物联网设备上的数据或破坏模型更新)；因此，在本文中，我们提出了一项针对数字孪生6G物联网环境中联合边缘学习中的中毒攻击的前瞻性研究。具体地说，我们研究了在支持6G的数字孪生物联网环境中，对手对联合学习模型的训练和发展的影响。我们证明了攻击者可以在集中式学习和联合学习两种不同的学习环境下进行中毒攻击，而成功的攻击会严重降低模型的准确性。在非独立同分布(Non-IID)数据和独立同分布(IID)数据下，使用三种深度神经网络综合评估了针对物联网应用设计的新的网络安全数据集的攻击。对于一个攻击分类问题，中毒攻击可以导致IID数据的准确率从94.93%下降到85.98%，而非IID数据的准确率从94.18%下降到30.04%。



## **48. Manipulating Transfer Learning for Property Inference**

操纵性迁移学习用于性质推理 cs.LG

Accepted to CVPR 2023

**SubmitDate**: 2023-03-21    [abs](http://arxiv.org/abs/2303.11643v1) [paper-pdf](http://arxiv.org/pdf/2303.11643v1)

**Authors**: Yulong Tian, Fnu Suya, Anshuman Suri, Fengyuan Xu, David Evans

**Abstract**: Transfer learning is a popular method for tuning pretrained (upstream) models for different downstream tasks using limited data and computational resources. We study how an adversary with control over an upstream model used in transfer learning can conduct property inference attacks on a victim's tuned downstream model. For example, to infer the presence of images of a specific individual in the downstream training set. We demonstrate attacks in which an adversary can manipulate the upstream model to conduct highly effective and specific property inference attacks (AUC score $> 0.9$), without incurring significant performance loss on the main task. The main idea of the manipulation is to make the upstream model generate activations (intermediate features) with different distributions for samples with and without a target property, thus enabling the adversary to distinguish easily between downstream models trained with and without training examples that have the target property. Our code is available at https://github.com/yulongt23/Transfer-Inference.

摘要: 转移学习是一种流行的方法，用于使用有限的数据和计算资源为不同的下游任务调整预先训练的(上游)模型。我们研究了在迁移学习中控制上游模型的对手如何对受害者调整后的下游模型进行属性推理攻击。例如，推断下游训练集中特定个体的图像的存在。我们演示了这样的攻击：攻击者可以操纵上游模型来进行高效和特定的属性推理攻击(AUC分数$>0.9$)，而不会在主任务上造成显著的性能损失。该操纵的主要思想是使上游模型为具有和不具有目标属性的样本生成具有不同分布的激活(中间特征)，从而使对手能够容易地区分具有和不具有目标属性的训练样本训练的下游模型。我们的代码可以在https://github.com/yulongt23/Transfer-Inference.上找到



## **49. An Observer-based Switching Algorithm for Safety under Sensor Denial-of-Service Attacks**

传感器拒绝服务攻击下基于观察者的安全切换算法 eess.SY

Accepted at the 2023 American Control Conference (ACC)

**SubmitDate**: 2023-03-21    [abs](http://arxiv.org/abs/2303.11640v1) [paper-pdf](http://arxiv.org/pdf/2303.11640v1)

**Authors**: Santiago Jimenez Leudo, Kunal Garg, Ricardo G. Sanfelice, Alvaro A. Cardenas

**Abstract**: The design of safe-critical control algorithms for systems under Denial-of-Service (DoS) attacks on the system output is studied in this work. We aim to address scenarios where attack-mitigation approaches are not feasible, and the system needs to maintain safety under adversarial attacks. We propose an attack-recovery strategy by designing a switching observer and characterizing bounds in the error of a state estimation scheme by specifying tolerable limits on the time length of attacks. Then, we propose a switching control algorithm that renders forward invariant a set for the observer. Thus, by satisfying the error bounds of the state estimation, we guarantee that the safe set is rendered conditionally invariant with respect to a set of initial conditions. A numerical example illustrates the efficacy of the approach.

摘要: 本文研究了在拒绝服务(DoS)攻击下系统输出的安全关键控制算法的设计。我们的目标是解决攻击缓解方法不可行的情况，并且系统需要在对抗性攻击下保持安全。提出了一种攻击恢复策略，该策略通过设计切换观测器和通过指定攻击时间长度的可容忍限度来刻画状态估计方案的误差界。然后，我们提出了一种切换控制算法，它使观测器的前向不变量成为一个集合。因此，通过满足状态估计的误差界，我们保证安全集关于一组初始条件是条件不变的。数值算例说明了该方法的有效性。



## **50. Enhancing the Self-Universality for Transferable Targeted Attacks**

增强可转移定向攻击的自我普适性 cs.CV

**SubmitDate**: 2023-03-21    [abs](http://arxiv.org/abs/2209.03716v2) [paper-pdf](http://arxiv.org/pdf/2209.03716v2)

**Authors**: Zhipeng Wei, Jingjing Chen, Zuxuan Wu, Yu-Gang Jiang

**Abstract**: In this paper, we propose a novel transfer-based targeted attack method that optimizes the adversarial perturbations without any extra training efforts for auxiliary networks on training data. Our new attack method is proposed based on the observation that highly universal adversarial perturbations tend to be more transferable for targeted attacks. Therefore, we propose to make the perturbation to be agnostic to different local regions within one image, which we called as self-universality. Instead of optimizing the perturbations on different images, optimizing on different regions to achieve self-universality can get rid of using extra data. Specifically, we introduce a feature similarity loss that encourages the learned perturbations to be universal by maximizing the feature similarity between adversarial perturbed global images and randomly cropped local regions. With the feature similarity loss, our method makes the features from adversarial perturbations to be more dominant than that of benign images, hence improving targeted transferability. We name the proposed attack method as Self-Universality (SU) attack. Extensive experiments demonstrate that SU can achieve high success rates for transfer-based targeted attacks. On ImageNet-compatible dataset, SU yields an improvement of 12\% compared with existing state-of-the-art methods. Code is available at https://github.com/zhipeng-wei/Self-Universality.

摘要: 在本文中，我们提出了一种新的基于转移的定向攻击方法，该方法在不需要对训练数据进行任何额外训练的情况下优化了对抗性扰动。我们提出的新攻击方法是基于这样的观察，即高度普遍的对抗性扰动倾向于更可转移到定向攻击。因此，我们提出使微扰对同一图像内的不同局部区域是不可知的，我们称之为自普适性。与对不同图像上的扰动进行优化不同，通过对不同区域进行优化来实现自普适性，可以避免使用额外的数据。具体地说，我们引入了特征相似度损失，通过最大化对抗性扰动的全局图像和随机裁剪的局部区域之间的特征相似度，鼓励学习的扰动具有普遍性。在特征相似度损失的情况下，我们的方法使得来自对抗性扰动的特征比来自良性图像的特征更具优势，从而提高了目标可转移性。我们将所提出的攻击方法命名为自普适性攻击。广泛的实验证明，宿灿对基于转会的靶向攻击取得了很高的成功率。在与ImageNet兼容的数据集上，与现有的最先进方法相比，SU方法的性能提高了12%。代码可在https://github.com/zhipeng-wei/Self-Universality.上找到



