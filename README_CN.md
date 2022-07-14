# Latest Adversarial Attack Papers
**update at 2022-07-15 06:31:21**

翻译来自 https://cloud.tencent.com/document/product/551/15619

## **1. Susceptibility of Continual Learning Against Adversarial Attacks**

持续学习对敌意攻击的敏感性 cs.LG

18 pages, 13 figures

**SubmitDate**: 2022-07-13    [paper-pdf](http://arxiv.org/pdf/2207.05225v2)

**Authors**: Hikmat Khan, Pir Masoom Shah, Syed Farhan Alam Zaidi, Saif ul Islam

**Abstracts**: The recent advances in continual (incremental or lifelong) learning have concentrated on the prevention of forgetting that can lead to catastrophic consequences, but there are two outstanding challenges that must be addressed. The first is the evaluation of the robustness of the proposed methods. The second is ensuring the security of learned tasks remains largely unexplored. This paper presents a comprehensive study of the susceptibility of the continually learned tasks (including both current and previously learned tasks) that are vulnerable to forgetting. Such vulnerability of tasks against adversarial attacks raises profound issues in data integrity and privacy. We consider the task incremental learning (Task-IL) scenario and explore three regularization-based experiments, three replay-based experiments, and one hybrid technique based on the reply and exemplar approach. We examine the robustness of these methods. In particular, we consider cases where we demonstrate that any class belonging to the current or previously learned tasks is prone to misclassification. Our observations highlight the potential limitations of existing Task-IL approaches. Our empirical study recommends that the research community consider the robustness of the proposed continual learning approaches and invest extensive efforts in mitigating catastrophic forgetting.

摘要: 持续(增量或终身)学习的最新进展集中在防止可能导致灾难性后果的遗忘上，但有两个突出的挑战必须解决。首先是对所提出方法的稳健性进行评估。第二，确保学习任务的安全性在很大程度上仍未得到探索。本文对易被遗忘的持续学习任务(包括当前学习任务和先前学习任务)的易感性进行了全面的研究。任务对对手攻击的这种脆弱性引发了数据完整性和隐私方面的严重问题。我们考虑了任务增量学习的场景，探索了三个基于正则化的实验，三个基于回放的实验，以及一个基于回复和样本方法的混合技术。我们检验了这些方法的稳健性。特别是，我们考虑了这样的情况，即我们证明属于当前或以前学习的任务的任何类都容易发生错误分类。我们的观察结果突出了现有任务-IL方法的潜在局限性。我们的实证研究建议研究界考虑所提出的持续学习方法的稳健性，并投入广泛的努力来缓解灾难性遗忘。



## **2. Exploring Adversarial Attacks and Defenses in Vision Transformers trained with DINO**

探索与恐龙一起训练的视觉变形金刚的对抗性攻击和防御 cs.CV

Workshop paper accepted at AdvML Frontiers (ICML 2022)

**SubmitDate**: 2022-07-13    [paper-pdf](http://arxiv.org/pdf/2206.06761v3)

**Authors**: Javier Rando, Nasib Naimi, Thomas Baumann, Max Mathys

**Abstracts**: This work conducts the first analysis on the robustness against adversarial attacks on self-supervised Vision Transformers trained using DINO. First, we evaluate whether features learned through self-supervision are more robust to adversarial attacks than those emerging from supervised learning. Then, we present properties arising for attacks in the latent space. Finally, we evaluate whether three well-known defense strategies can increase adversarial robustness in downstream tasks by only fine-tuning the classification head to provide robustness even in view of limited compute resources. These defense strategies are: Adversarial Training, Ensemble Adversarial Training and Ensemble of Specialized Networks.

摘要: 本文首次对使用Dino训练的自监督视觉转换器的抗敌意攻击能力进行了分析。首先，我们评估通过自我监督学习的特征是否比通过监督学习获得的特征对对手攻击更健壮。然后，我们给出了潜在空间中攻击产生的性质。最后，我们评估了三种著名的防御策略是否能够在下游任务中通过微调分类头来提高对手的健壮性，即使在计算资源有限的情况下也是如此。这些防御策略是：对抗性训练、系列性对抗性训练和专业网络系列化。



## **3. Adversarially-Aware Robust Object Detector**

对抗性感知的鲁棒目标检测器 cs.CV

17 pages, 7 figures, ECCV2022 oral paper

**SubmitDate**: 2022-07-13    [paper-pdf](http://arxiv.org/pdf/2207.06202v1)

**Authors**: Ziyi Dong, Pengxu Wei, Liang Lin

**Abstracts**: Object detection, as a fundamental computer vision task, has achieved a remarkable progress with the emergence of deep neural networks. Nevertheless, few works explore the adversarial robustness of object detectors to resist adversarial attacks for practical applications in various real-world scenarios. Detectors have been greatly challenged by unnoticeable perturbation, with sharp performance drop on clean images and extremely poor performance on adversarial images. In this work, we empirically explore the model training for adversarial robustness in object detection, which greatly attributes to the conflict between learning clean images and adversarial images. To mitigate this issue, we propose a Robust Detector (RobustDet) based on adversarially-aware convolution to disentangle gradients for model learning on clean and adversarial images. RobustDet also employs the Adversarial Image Discriminator (AID) and Consistent Features with Reconstruction (CFR) to ensure a reliable robustness. Extensive experiments on PASCAL VOC and MS-COCO demonstrate that our model effectively disentangles gradients and significantly enhances the detection robustness with maintaining the detection ability on clean images.

摘要: 随着深度神经网络的出现，目标检测作为一项基本的计算机视觉任务已经取得了显著的进展。然而，很少有研究探讨对象检测器在各种真实场景中的实际应用中抵抗对手攻击的对抗性健壮性。检测器受到了不可察觉的扰动的极大挑战，在干净图像上的性能急剧下降，在对抗性图像上的性能极差。在这项工作中，我们经验性地探索了目标检测中对抗鲁棒性的模型训练，这在很大程度上归因于学习干净图像和对抗图像之间的冲突。为了缓解这一问题，我们提出了一种基于对抗性感知卷积的稳健检测器(RobustDet)，用于在干净图像和对抗性图像上进行模型学习。RobustDet还采用了对抗性图像鉴别器(AID)和重建一致特征(CFR)，以确保可靠的健壮性。在PASCAL、VOC和MS-COCO上的大量实验表明，该模型在保持对干净图像的检测能力的同时，有效地解开了梯度的纠缠，显著提高了检测的鲁棒性。



## **4. Interactive Machine Learning: A State of the Art Review**

交互式机器学习：最新进展 cs.LG

**SubmitDate**: 2022-07-13    [paper-pdf](http://arxiv.org/pdf/2207.06196v1)

**Authors**: Natnael A. Wondimu, Cédric Buche, Ubbo Visser

**Abstracts**: Machine learning has proved useful in many software disciplines, including computer vision, speech and audio processing, natural language processing, robotics and some other fields. However, its applicability has been significantly hampered due its black-box nature and significant resource consumption. Performance is achieved at the expense of enormous computational resource and usually compromising the robustness and trustworthiness of the model. Recent researches have been identifying a lack of interactivity as the prime source of these machine learning problems. Consequently, interactive machine learning (iML) has acquired increased attention of researchers on account of its human-in-the-loop modality and relatively efficient resource utilization. Thereby, a state-of-the-art review of interactive machine learning plays a vital role in easing the effort toward building human-centred models. In this paper, we provide a comprehensive analysis of the state-of-the-art of iML. We analyze salient research works using merit-oriented and application/task oriented mixed taxonomy. We use a bottom-up clustering approach to generate a taxonomy of iML research works. Research works on adversarial black-box attacks and corresponding iML based defense system, exploratory machine learning, resource constrained learning, and iML performance evaluation are analyzed under their corresponding theme in our merit-oriented taxonomy. We have further classified these research works into technical and sectoral categories. Finally, research opportunities that we believe are inspiring for future work in iML are discussed thoroughly.

摘要: 机器学习已被证明在许多软件学科中都很有用，包括计算机视觉、语音和音频处理、自然语言处理、机器人学和其他一些领域。然而，由于其黑箱性质和巨大的资源消耗，其适用性受到了极大的阻碍。性能的实现是以牺牲巨大的计算资源为代价的，并且通常会损害模型的健壮性和可信性。最近的研究已经确定缺乏互动性是这些机器学习问题的主要来源。因此，交互式机器学习(IML)以其人在环中的方式和相对高效的资源利用而受到越来越多的研究者的关注。因此，对交互式机器学习的最新回顾在减轻建立以人为中心的模型的努力方面发挥着至关重要的作用。在本文中，我们对iML的最新发展进行了全面的分析。我们使用面向价值的和面向应用/任务的混合分类来分析重要的研究作品。我们使用自下而上的聚类方法来生成iML研究作品的分类。在我们的价值导向分类法中，对抗性黑盒攻击和相应的基于iML的防御系统、探索性机器学习、资源受限学习和iML性能评估的研究工作都在相应的主题下进行了分析。我们进一步将这些研究工作分为技术和部门两个类别。最后，深入讨论了我们认为对iML未来工作有启发的研究机会。



## **5. On the Robustness of Bayesian Neural Networks to Adversarial Attacks**

贝叶斯神经网络对敌方攻击的稳健性研究 cs.LG

arXiv admin note: text overlap with arXiv:2002.04359

**SubmitDate**: 2022-07-13    [paper-pdf](http://arxiv.org/pdf/2207.06154v1)

**Authors**: Luca Bortolussi, Ginevra Carbone, Luca Laurenti, Andrea Patane, Guido Sanguinetti, Matthew Wicker

**Abstracts**: Vulnerability to adversarial attacks is one of the principal hurdles to the adoption of deep learning in safety-critical applications. Despite significant efforts, both practical and theoretical, training deep learning models robust to adversarial attacks is still an open problem. In this paper, we analyse the geometry of adversarial attacks in the large-data, overparameterized limit for Bayesian Neural Networks (BNNs). We show that, in the limit, vulnerability to gradient-based attacks arises as a result of degeneracy in the data distribution, i.e., when the data lies on a lower-dimensional submanifold of the ambient space. As a direct consequence, we demonstrate that in this limit BNN posteriors are robust to gradient-based adversarial attacks. Crucially, we prove that the expected gradient of the loss with respect to the BNN posterior distribution is vanishing, even when each neural network sampled from the posterior is vulnerable to gradient-based attacks. Experimental results on the MNIST, Fashion MNIST, and half moons datasets, representing the finite data regime, with BNNs trained with Hamiltonian Monte Carlo and Variational Inference, support this line of arguments, showing that BNNs can display both high accuracy on clean data and robustness to both gradient-based and gradient-free based adversarial attacks.

摘要: 对敌意攻击的脆弱性是在安全关键应用中采用深度学习的主要障碍之一。尽管在实践和理论上都做了大量的努力，但训练对对手攻击稳健的深度学习模型仍然是一个悬而未决的问题。本文分析了贝叶斯神经网络(BNN)在大数据、过参数限制下的攻击几何。我们证明，在极限情况下，由于数据分布的退化，即当数据位于环境空间的低维子流形上时，对基于梯度的攻击的脆弱性出现。作为一个直接的推论，我们证明了在这个极限下，BNN后验网络对基于梯度的敌意攻击是稳健的。重要的是，我们证明了损失相对于BNN后验分布的期望梯度是零的，即使从后验采样的每个神经网络都容易受到基于梯度的攻击。在代表有限数据区的MNIST、Fashion MNIST和半月数据集上的实验结果支持这一论点，BNN采用哈密顿蒙特卡罗和变分推理进行训练，表明BNN在干净数据上具有很高的准确率，并且对基于梯度和基于无梯度的敌意攻击都具有很好的鲁棒性。



## **6. Neural Network Robustness as a Verification Property: A Principled Case Study**

作为验证属性的神经网络健壮性：原则性案例研究 cs.LG

11 pages, CAV 2022

**SubmitDate**: 2022-07-13    [paper-pdf](http://arxiv.org/pdf/2104.01396v2)

**Authors**: Marco Casadio, Ekaterina Komendantskaya, Matthew L. Daggitt, Wen Kokke, Guy Katz, Guy Amir, Idan Refaeli

**Abstracts**: Neural networks are very successful at detecting patterns in noisy data, and have become the technology of choice in many fields. However, their usefulness is hampered by their susceptibility to adversarial attacks. Recently, many methods for measuring and improving a network's robustness to adversarial perturbations have been proposed, and this growing body of research has given rise to numerous explicit or implicit notions of robustness. Connections between these notions are often subtle, and a systematic comparison between them is missing in the literature. In this paper we begin addressing this gap, by setting up general principles for the empirical analysis and evaluation of a network's robustness as a mathematical property - during the network's training phase, its verification, and after its deployment. We then apply these principles and conduct a case study that showcases the practical benefits of our general approach.

摘要: 神经网络在检测噪声数据中的模式方面非常成功，已经成为许多领域的首选技术。然而，由于它们容易受到对抗性攻击，它们的有用性受到了阻碍。最近，已经提出了许多方法来衡量和提高网络对敌意干扰的稳健性，并且这一不断增长的研究已经产生了许多显式或隐式的健壮性概念。这些概念之间的联系往往很微妙，文献中也没有对它们进行系统的比较。在本文中，我们开始解决这一差距，通过建立一般原则，将网络的稳健性作为一种数学属性进行经验分析和评估--在网络的训练阶段、验证阶段和部署之后。然后，我们应用这些原则并进行案例研究，展示我们一般方法的实际好处。



## **7. Perturbation Inactivation Based Adversarial Defense for Face Recognition**

基于扰动失活的人脸识别对抗性防御 cs.CV

Accepted by IEEE Transactions on Information Forensics & Security  (T-IFS)

**SubmitDate**: 2022-07-13    [paper-pdf](http://arxiv.org/pdf/2207.06035v1)

**Authors**: Min Ren, Yuhao Zhu, Yunlong Wang, Zhenan Sun

**Abstracts**: Deep learning-based face recognition models are vulnerable to adversarial attacks. To curb these attacks, most defense methods aim to improve the robustness of recognition models against adversarial perturbations. However, the generalization capacities of these methods are quite limited. In practice, they are still vulnerable to unseen adversarial attacks. Deep learning models are fairly robust to general perturbations, such as Gaussian noises. A straightforward approach is to inactivate the adversarial perturbations so that they can be easily handled as general perturbations. In this paper, a plug-and-play adversarial defense method, named perturbation inactivation (PIN), is proposed to inactivate adversarial perturbations for adversarial defense. We discover that the perturbations in different subspaces have different influences on the recognition model. There should be a subspace, called the immune space, in which the perturbations have fewer adverse impacts on the recognition model than in other subspaces. Hence, our method estimates the immune space and inactivates the adversarial perturbations by restricting them to this subspace. The proposed method can be generalized to unseen adversarial perturbations since it does not rely on a specific kind of adversarial attack method. This approach not only outperforms several state-of-the-art adversarial defense methods but also demonstrates a superior generalization capacity through exhaustive experiments. Moreover, the proposed method can be successfully applied to four commercial APIs without additional training, indicating that it can be easily generalized to existing face recognition systems. The source code is available at https://github.com/RenMin1991/Perturbation-Inactivate

摘要: 基于深度学习的人脸识别模型容易受到敌意攻击。为了遏制这些攻击，大多数防御方法的目的是提高识别模型对对手扰动的稳健性。然而，这些方法的泛化能力相当有限。在实践中，他们仍然容易受到看不见的对手攻击。深度学习模型对一般扰动具有较强的鲁棒性，如高斯噪声。一种简单的方法是停用对抗性扰动，这样它们就可以很容易地作为一般扰动来处理。本文提出了一种即插即用的对抗防御方法，称为扰动失活(PIN)，用于灭活对抗防御中的对抗扰动。我们发现，不同子空间中的扰动对识别模型有不同的影响。应该有一个称为免疫空间的子空间，在这个子空间中，扰动对识别模型的不利影响比在其他子空间中要小。因此，我们的方法估计免疫空间，并通过将对抗性扰动限制在此子空间来使其失活。由于该方法不依赖于一种特定的对抗性攻击方法，因此可以推广到不可见的对抗性扰动。通过详尽的实验证明，该方法不仅比目前最先进的对抗性防御方法有更好的性能，而且具有更好的泛化能力。此外，该方法可以成功地应用于四个商业API，而无需额外的训练，这表明它可以很容易地推广到现有的人脸识别系统中。源代码可在https://github.com/RenMin1991/Perturbation-Inactivate上找到



## **8. BadHash: Invisible Backdoor Attacks against Deep Hashing with Clean Label**

BadHash：使用Clean Label对深度哈希进行隐形后门攻击 cs.CV

This paper has been accepted by the 30th ACM International Conference  on Multimedia (MM '22, October 10--14, 2022, Lisboa, Portugal)

**SubmitDate**: 2022-07-13    [paper-pdf](http://arxiv.org/pdf/2207.00278v3)

**Authors**: Shengshan Hu, Ziqi Zhou, Yechao Zhang, Leo Yu Zhang, Yifeng Zheng, Yuanyuan HE, Hai Jin

**Abstracts**: Due to its powerful feature learning capability and high efficiency, deep hashing has achieved great success in large-scale image retrieval. Meanwhile, extensive works have demonstrated that deep neural networks (DNNs) are susceptible to adversarial examples, and exploring adversarial attack against deep hashing has attracted many research efforts. Nevertheless, backdoor attack, another famous threat to DNNs, has not been studied for deep hashing yet. Although various backdoor attacks have been proposed in the field of image classification, existing approaches failed to realize a truly imperceptive backdoor attack that enjoys invisible triggers and clean label setting simultaneously, and they also cannot meet the intrinsic demand of image retrieval backdoor. In this paper, we propose BadHash, the first generative-based imperceptible backdoor attack against deep hashing, which can effectively generate invisible and input-specific poisoned images with clean label. Specifically, we first propose a new conditional generative adversarial network (cGAN) pipeline to effectively generate poisoned samples. For any given benign image, it seeks to generate a natural-looking poisoned counterpart with a unique invisible trigger. In order to improve the attack effectiveness, we introduce a label-based contrastive learning network LabCLN to exploit the semantic characteristics of different labels, which are subsequently used for confusing and misleading the target model to learn the embedded trigger. We finally explore the mechanism of backdoor attacks on image retrieval in the hash space. Extensive experiments on multiple benchmark datasets verify that BadHash can generate imperceptible poisoned samples with strong attack ability and transferability over state-of-the-art deep hashing schemes.

摘要: 深度哈希法由于其强大的特征学习能力和高效的检索效率，在大规模图像检索中取得了巨大的成功。同时，大量的研究表明，深度神经网络(DNN)容易受到敌意例子的影响，探索针对深度散列的敌意攻击吸引了许多研究努力。然而，DNNS的另一个著名威胁--后门攻击，还没有被研究过深度散列。虽然在图像分类领域已经提出了各种各样的后门攻击，但现有的方法未能实现真正的隐蔽的、同时具有不可见触发器和干净标签设置的后门攻击，也不能满足图像检索的内在需求。本文提出了BadHash，这是第一个基于生成性的针对深度哈希的不可察觉的后门攻击，它可以有效地生成标签清晰的不可见和输入特定的有毒图像。具体地说，我们首先提出了一种新的条件生成对抗网络(CGAN)管道来有效地生成有毒样本。对于任何给定的良性形象，它都试图生成一个看起来自然、有毒的形象，并带有独特的无形触发器。为了提高攻击的有效性，我们引入了一个基于标签的对比学习网络LabCLN来利用不同标签的语义特征，这些语义特征被用来混淆和误导目标模型学习嵌入的触发器。最后，我们探讨了哈希空间中后门攻击对图像检索的影响机制。在多个基准数据集上的大量实验证明，BadHash可以生成不可察觉的有毒样本，具有很强的攻击能力和可转移性，优于最先进的深度哈希方案。



## **9. Physical Backdoor Attacks to Lane Detection Systems in Autonomous Driving**

自动驾驶中车道检测系统的物理后门攻击 cs.CV

Accepted by ACM MultiMedia 2022

**SubmitDate**: 2022-07-13    [paper-pdf](http://arxiv.org/pdf/2203.00858v2)

**Authors**: Xingshuo Han, Guowen Xu, Yuan Zhou, Xuehuan Yang, Jiwei Li, Tianwei Zhang

**Abstracts**: Modern autonomous vehicles adopt state-of-the-art DNN models to interpret the sensor data and perceive the environment. However, DNN models are vulnerable to different types of adversarial attacks, which pose significant risks to the security and safety of the vehicles and passengers. One prominent threat is the backdoor attack, where the adversary can compromise the DNN model by poisoning the training samples. Although lots of effort has been devoted to the investigation of the backdoor attack to conventional computer vision tasks, its practicality and applicability to the autonomous driving scenario is rarely explored, especially in the physical world.   In this paper, we target the lane detection system, which is an indispensable module for many autonomous driving tasks, e.g., navigation, lane switching. We design and realize the first physical backdoor attacks to such system. Our attacks are comprehensively effective against different types of lane detection algorithms. Specifically, we introduce two attack methodologies (poison-annotation and clean-annotation) to generate poisoned samples. With those samples, the trained lane detection model will be infected with the backdoor, and can be activated by common objects (e.g., traffic cones) to make wrong detections, leading the vehicle to drive off the road or onto the opposite lane. Extensive evaluations on public datasets and physical autonomous vehicles demonstrate that our backdoor attacks are effective, stealthy and robust against various defense solutions. Our codes and experimental videos can be found in https://sites.google.com/view/lane-detection-attack/lda.

摘要: 现代自动驾驶汽车采用最先进的DNN模型来解释传感器数据和感知环境。然而，DNN模型容易受到不同类型的对抗性攻击，这对车辆和乘客的安全和安全构成了重大风险。一个突出的威胁是后门攻击，在这种攻击中，对手可以通过毒化训练样本来危害DNN模型。虽然对传统计算机视觉任务的后门攻击已经投入了大量的精力来研究，但很少有人探索它在自动驾驶场景中的实用性和适用性，特别是在物理世界中。本文以车道检测系统为研究对象，车道检测系统是导航、车道切换等自动驾驶任务中不可缺少的模块。我们设计并实现了对此类系统的第一次物理后门攻击。我们的攻击对不同类型的车道检测算法都是全面有效的。具体地说，我们引入了两种攻击方法(毒注解和干净注解)来生成中毒样本。利用这些样本，训练后的车道检测模型将被后门感染，并可能被常见对象(如交通锥体)激活以进行错误检测，导致车辆驶离道路或进入对面车道。对公共数据集和物理自动驾驶车辆的广泛评估表明，我们的后门攻击针对各种防御解决方案是有效的、隐蔽的和健壮的。我们的代码和实验视频可在https://sites.google.com/view/lane-detection-attack/lda.中找到



## **10. PatchZero: Defending against Adversarial Patch Attacks by Detecting and Zeroing the Patch**

PatchZero：通过检测和归零补丁来防御敌意补丁攻击 cs.CV

**SubmitDate**: 2022-07-13    [paper-pdf](http://arxiv.org/pdf/2207.01795v2)

**Authors**: Ke Xu, Yao Xiao, Zhaoheng Zheng, Kaijie Cai, Ram Nevatia

**Abstracts**: Adversarial patch attacks mislead neural networks by injecting adversarial pixels within a local region. Patch attacks can be highly effective in a variety of tasks and physically realizable via attachment (e.g. a sticker) to the real-world objects. Despite the diversity in attack patterns, adversarial patches tend to be highly textured and different in appearance from natural images. We exploit this property and present PatchZero, a general defense pipeline against white-box adversarial patches without retraining the downstream classifier or detector. Specifically, our defense detects adversaries at the pixel-level and "zeros out" the patch region by repainting with mean pixel values. We further design a two-stage adversarial training scheme to defend against the stronger adaptive attacks. PatchZero achieves SOTA defense performance on the image classification (ImageNet, RESISC45), object detection (PASCAL VOC), and video classification (UCF101) tasks with little degradation in benign performance. In addition, PatchZero transfers to different patch shapes and attack types.

摘要: 对抗性补丁攻击通过在局部区域内注入对抗性像素来误导神经网络。补丁攻击可以在各种任务中非常有效，并且可以通过附着(例如贴纸)到真实世界的对象来物理实现。尽管攻击模式多种多样，但敌方补丁往往纹理丰富，外观与自然图像不同。我们利用这一特性，提出了PatchZero，一种针对白盒恶意补丁的通用防御管道，而不需要重新训练下游的分类器或检测器。具体地说，我们的防御在像素级检测对手，并通过使用平均像素值重新绘制来对补丁区域进行“清零”。我们进一步设计了一种两阶段对抗性训练方案，以抵御更强的适应性攻击。PatchZero在图像分类(ImageNet，RESISC45)、目标检测(Pascal VOC)和视频分类(UCF101)任务上实现了SOTA防御性能，性能良好，性能几乎没有下降。此外，PatchZero还可以转换为不同的补丁形状和攻击类型。



## **11. Game of Trojans: A Submodular Byzantine Approach**

特洛伊木马游戏：一种拜占庭式的子模块方法 cs.LG

Submitted to GameSec 2022

**SubmitDate**: 2022-07-13    [paper-pdf](http://arxiv.org/pdf/2207.05937v1)

**Authors**: Dinuka Sahabandu, Arezoo Rajabi, Luyao Niu, Bo Li, Bhaskar Ramasubramanian, Radha Poovendran

**Abstracts**: Machine learning models in the wild have been shown to be vulnerable to Trojan attacks during training. Although many detection mechanisms have been proposed, strong adaptive attackers have been shown to be effective against them. In this paper, we aim to answer the questions considering an intelligent and adaptive adversary: (i) What is the minimal amount of instances required to be Trojaned by a strong attacker? and (ii) Is it possible for such an attacker to bypass strong detection mechanisms?   We provide an analytical characterization of adversarial capability and strategic interactions between the adversary and detection mechanism that take place in such models. We characterize adversary capability in terms of the fraction of the input dataset that can be embedded with a Trojan trigger. We show that the loss function has a submodular structure, which leads to the design of computationally efficient algorithms to determine this fraction with provable bounds on optimality. We propose a Submodular Trojan algorithm to determine the minimal fraction of samples to inject a Trojan trigger. To evade detection of the Trojaned model, we model strategic interactions between the adversary and Trojan detection mechanism as a two-player game. We show that the adversary wins the game with probability one, thus bypassing detection. We establish this by proving that output probability distributions of a Trojan model and a clean model are identical when following the Min-Max (MM) Trojan algorithm.   We perform extensive evaluations of our algorithms on MNIST, CIFAR-10, and EuroSAT datasets. The results show that (i) with Submodular Trojan algorithm, the adversary needs to embed a Trojan trigger into a very small fraction of samples to achieve high accuracy on both Trojan and clean samples, and (ii) the MM Trojan algorithm yields a trained Trojan model that evades detection with probability 1.

摘要: 野外的机器学习模型已被证明在训练期间容易受到特洛伊木马的攻击。虽然已经提出了许多检测机制，但强自适应攻击者被证明对它们是有效的。在本文中，我们的目标是考虑一个智能和自适应的对手来回答以下问题：(I)强攻击者需要木马的最小实例数量是多少？以及(Ii)这样的攻击者是否有可能绕过强大的检测机制？我们提供了发生在这样的模型中的对手能力和对手与检测机制之间的战略交互的分析特征。我们根据可以嵌入特洛伊木马触发器的输入数据集的比例来表征攻击者的能力。我们证明了损失函数具有子模结构，这导致设计了计算效率高的算法来确定具有可证明的最优界的分数。我们提出了一种子模块木马算法来确定注入木马触发器的最小样本比例。为了逃避木马模型的检测，我们将对手和木马检测机制之间的战略交互建模为两人博弈。我们证明了对手以概率1赢得比赛，从而绕过了检测。我们证明了当遵循Min-Max(MM)木马算法时，木马模型和CLEAN模型的输出概率分布是相同的。我们在MNIST、CIFAR-10和EuroSAT数据集上对我们的算法进行了广泛的评估。结果表明：(I)利用子模块木马算法，攻击者需要在很小一部分样本中嵌入木马触发器，以获得对木马和干净样本的高精度；(Ii)MM木马算法生成一个以1的概率逃避检测的训练有素的木马模型。



## **12. A Word is Worth A Thousand Dollars: Adversarial Attack on Tweets Fools Stock Predictions**

一句话抵得上一千美元：对推特傻瓜股预测的敌意攻击 cs.CR

NAACL short paper, github: https://github.com/yonxie/AdvFinTweet

**SubmitDate**: 2022-07-12    [paper-pdf](http://arxiv.org/pdf/2205.01094v3)

**Authors**: Yong Xie, Dakuo Wang, Pin-Yu Chen, Jinjun Xiong, Sijia Liu, Sanmi Koyejo

**Abstracts**: More and more investors and machine learning models rely on social media (e.g., Twitter and Reddit) to gather real-time information and sentiment to predict stock price movements. Although text-based models are known to be vulnerable to adversarial attacks, whether stock prediction models have similar vulnerability is underexplored. In this paper, we experiment with a variety of adversarial attack configurations to fool three stock prediction victim models. We address the task of adversarial generation by solving combinatorial optimization problems with semantics and budget constraints. Our results show that the proposed attack method can achieve consistent success rates and cause significant monetary loss in trading simulation by simply concatenating a perturbed but semantically similar tweet.

摘要: 越来越多的投资者和机器学习模型依赖社交媒体(如Twitter和Reddit)来收集实时信息和情绪，以预测股价走势。尽管众所周知，基于文本的模型容易受到对手攻击，但股票预测模型是否也有类似的脆弱性，还没有得到充分的探讨。在本文中，我们实验了各种对抗性攻击配置，以愚弄三个股票预测受害者模型。我们通过求解具有语义和预算约束的组合优化问题来解决对抗性生成问题。我们的结果表明，该攻击方法可以获得一致的成功率，并在交易模拟中通过简单地连接一条扰动但语义相似的推文来造成巨大的金钱损失。



## **13. Practical Attacks on Machine Learning: A Case Study on Adversarial Windows Malware**

对机器学习的实用攻击：恶意Windows恶意软件的案例研究 cs.CR

**SubmitDate**: 2022-07-12    [paper-pdf](http://arxiv.org/pdf/2207.05548v1)

**Authors**: Luca Demetrio, Battista Biggio, Fabio Roli

**Abstracts**: While machine learning is vulnerable to adversarial examples, it still lacks systematic procedures and tools for evaluating its security in different application contexts. In this article, we discuss how to develop automated and scalable security evaluations of machine learning using practical attacks, reporting a use case on Windows malware detection.

摘要: 虽然机器学习很容易受到敌意例子的影响，但它仍然缺乏系统的程序和工具来评估其在不同应用环境中的安全性。在本文中，我们讨论了如何使用实际攻击来开发自动化和可扩展的机器学习安全评估，并报告了一个Windows恶意软件检测的用例。



## **14. Improving the Robustness and Generalization of Deep Neural Network with Confidence Threshold Reduction**

降低置信度阈值提高深度神经网络的鲁棒性和泛化能力 cs.LG

Under review

**SubmitDate**: 2022-07-12    [paper-pdf](http://arxiv.org/pdf/2206.00913v2)

**Authors**: Xiangyuan Yang, Jie Lin, Hanlin Zhang, Xinyu Yang, Peng Zhao

**Abstracts**: Deep neural networks are easily attacked by imperceptible perturbation. Presently, adversarial training (AT) is the most effective method to enhance the robustness of the model against adversarial examples. However, because adversarial training solved a min-max value problem, in comparison with natural training, the robustness and generalization are contradictory, i.e., the robustness improvement of the model will decrease the generalization of the model. To address this issue, in this paper, a new concept, namely confidence threshold (CT), is introduced and the reducing of the confidence threshold, known as confidence threshold reduction (CTR), is proven to improve both the generalization and robustness of the model. Specifically, to reduce the CT for natural training (i.e., for natural training with CTR), we propose a mask-guided divergence loss function (MDL) consisting of a cross-entropy loss term and an orthogonal term. The empirical and theoretical analysis demonstrates that the MDL loss improves the robustness and generalization of the model simultaneously for natural training. However, the model robustness improvement of natural training with CTR is not comparable to that of adversarial training. Therefore, for adversarial training, we propose a standard deviation loss function (STD), which minimizes the difference in the probabilities of the wrong categories, to reduce the CT by being integrated into the loss function of adversarial training. The empirical and theoretical analysis demonstrates that the STD based loss function can further improve the robustness of the adversarially trained model on basis of guaranteeing the changeless or slight improvement of the natural accuracy.

摘要: 深层神经网络很容易受到不可察觉的扰动的攻击。目前，对抗性训练(AT)是提高模型对对抗性例子的稳健性的最有效方法。然而，由于对抗性训练解决了最小-最大值问题，与自然训练相比，鲁棒性和泛化是矛盾的，即模型的稳健性提高会降低模型的泛化能力。针对这一问题，本文引入了置信度阈值的概念，并证明了置信度阈值的降低既能提高模型的泛化能力，又能提高模型的鲁棒性。具体地说，为了减少自然训练(即具有CTR的自然训练)的CT，我们提出了一种掩模引导的发散损失函数(MDL)，该函数由交叉熵损失项和正交项组成。实验和理论分析表明，对于自然训练，MDL损失同时提高了模型的鲁棒性和泛化能力。然而，CTR自然训练对模型稳健性的改善与对抗性训练不可同日而语。因此，对于对抗性训练，我们提出了一种标准偏差损失函数(STD)，它最小化了错误类别概率的差异，通过将其整合到对抗性训练的损失函数中来降低CT。实证和理论分析表明，基于STD的损失函数可以在保证自然精度不变或略有提高的基础上，进一步提高对抗性训练模型的稳健性。



## **15. Adversarial Robustness Assessment of NeuroEvolution Approaches**

神经进化方法的对抗性稳健性评估 cs.NE

**SubmitDate**: 2022-07-12    [paper-pdf](http://arxiv.org/pdf/2207.05451v1)

**Authors**: Inês Valentim, Nuno Lourenço, Nuno Antunes

**Abstracts**: NeuroEvolution automates the generation of Artificial Neural Networks through the application of techniques from Evolutionary Computation. The main goal of these approaches is to build models that maximize predictive performance, sometimes with an additional objective of minimizing computational complexity. Although the evolved models achieve competitive results performance-wise, their robustness to adversarial examples, which becomes a concern in security-critical scenarios, has received limited attention. In this paper, we evaluate the adversarial robustness of models found by two prominent NeuroEvolution approaches on the CIFAR-10 image classification task: DENSER and NSGA-Net. Since the models are publicly available, we consider white-box untargeted attacks, where the perturbations are bounded by either the L2 or the Linfinity-norm. Similarly to manually-designed networks, our results show that when the evolved models are attacked with iterative methods, their accuracy usually drops to, or close to, zero under both distance metrics. The DENSER model is an exception to this trend, showing some resistance under the L2 threat model, where its accuracy only drops from 93.70% to 18.10% even with iterative attacks. Additionally, we analyzed the impact of pre-processing applied to the data before the first layer of the network. Our observations suggest that some of these techniques can exacerbate the perturbations added to the original inputs, potentially harming robustness. Thus, this choice should not be neglected when automatically designing networks for applications where adversarial attacks are prone to occur.

摘要: 神经进化通过应用进化计算的技术自动生成人工神经网络。这些方法的主要目标是构建最大限度提高预测性能的模型，有时还有最小化计算复杂性的额外目标。虽然进化模型在性能方面达到了竞争的结果，但它们对敌意示例的健壮性受到了有限的关注，这在安全关键场景中成为一个令人担忧的问题。在本文中，我们评估了两种重要的神经进化方法在CIFAR-10图像分类任务中发现的模型的对抗性健壮性：Denser和NSGA-Net。由于模型是公开可用的，我们考虑白盒非目标攻击，其中扰动由L2或Linfinity范数有界。与人工设计的网络类似，我们的结果表明，当进化模型受到迭代方法的攻击时，在两种距离度量下，它们的精度通常下降到或接近于零。密度更高的模型是这一趋势的一个例外，在L2威胁模型下显示出一些阻力，即使在迭代攻击的情况下，其准确率也只从93.70%下降到18.10%。此外，我们还分析了在网络第一层之前应用预处理对数据的影响。我们的观察表明，其中一些技术可能会加剧添加到原始输入的扰动，潜在地损害稳健性。因此，在为容易发生对抗性攻击的应用程序自动设计网络时，这一选择不应被忽视。



## **16. A Security-aware and LUT-based CAD Flow for the Physical Synthesis of eASICs**

一种安全感知的基于查找表的eASIC物理综合CAD流程 cs.CR

**SubmitDate**: 2022-07-12    [paper-pdf](http://arxiv.org/pdf/2207.05413v1)

**Authors**: Zain UlAbideen, Tiago Diadami Perez, Mayler Martins, Samuel Pagliarini

**Abstracts**: Numerous threats are associated with the globalized integrated circuit (IC) supply chain, such as piracy, reverse engineering, overproduction, and malicious logic insertion. Many obfuscation approaches have been proposed to mitigate these threats by preventing an adversary from fully understanding the IC (or parts of it). The use of reconfigurable elements inside an IC is a known obfuscation technique, either as a coarse grain reconfigurable block (i.e., eFPGA) or as a fine grain element (i.e., FPGA-like look-up tables). This paper presents a security-aware CAD flow that is LUT-based yet still compatible with the standard cell based physical synthesis flow. More precisely, our CAD flow explores the FPGA-ASIC design space and produces heavily obfuscated designs where only small portions of the logic resemble an ASIC. Therefore, we term this specialized solution an "embedded ASIC" (eASIC). Nevertheless, even for heavily LUT-dominated designs, our proposed decomposition and pin swapping algorithms allow for performance gains that enable performance levels that only ASICs would otherwise achieve. On the security side, we have developed novel template-based attacks and also applied existing attacks, both oracle-free and oracle-based. Our security analysis revealed that the obfuscation rate for an SHA-256 study case should be at least 45% for withstanding traditional attacks and at least 80% for withstanding template-based attacks. When the 80\% obfuscated SHA-256 design is physically implemented, it achieves a remarkable frequency of 368MHz in a 65nm commercial technology, whereas its FPGA implementation (in a superior technology) achieves only 77MHz.

摘要: 与全球化集成电路(IC)供应链相关的许多威胁，如盗版、逆向工程、生产过剩和恶意逻辑插入。已经提出了许多模糊方法来通过阻止对手完全理解IC(或其部分)来缓解这些威胁。在IC内使用可重构元件是一种已知的混淆技术，或者作为粗粒度可重构块(即，eFPGA)，或者作为细粒度元件(即，类似于FPGA的查找表)。本文提出了一种安全感知的CAD流程，该流程是基于LUT的，但仍然与基于标准单元的物理综合流程兼容。更准确地说，我们的CAD流程探索了FPGA-ASIC设计空间，并产生了高度混淆的设计，其中只有一小部分逻辑类似于ASIC。因此，我们将这种专门的解决方案称为“嵌入式ASIC”(EASIC)。然而，即使对于以查找表为主的设计，我们建议的分解和管脚交换算法也可以实现性能提升，从而实现只有ASIC才能达到的性能水平。在安全方面，我们开发了新的基于模板的攻击，并应用了现有的攻击，包括无Oracle攻击和基于Oracle的攻击。我们的安全分析显示，对于SHA-256研究案例，对于抵抗传统攻击至少应该是45%，对于基于模板的攻击至少应该是80%。在实际实现80型混淆SHA-256设计时，它在65 nm的商用工艺下达到了368 MHz的显著频率，而它的FPGA实现(在更高的工艺下)只达到了77 MHz。



## **17. Frequency Domain Model Augmentation for Adversarial Attack**

对抗性攻击的频域模型增强 cs.CV

Accepted by ECCV 2022

**SubmitDate**: 2022-07-12    [paper-pdf](http://arxiv.org/pdf/2207.05382v1)

**Authors**: Yuyang Long, Qilong Zhang, Boheng Zeng, Lianli Gao, Xianglong Liu, Jian Zhang, Jingkuan Song

**Abstracts**: For black-box attacks, the gap between the substitute model and the victim model is usually large, which manifests as a weak attack performance. Motivated by the observation that the transferability of adversarial examples can be improved by attacking diverse models simultaneously, model augmentation methods which simulate different models by using transformed images are proposed. However, existing transformations for spatial domain do not translate to significantly diverse augmented models. To tackle this issue, we propose a novel spectrum simulation attack to craft more transferable adversarial examples against both normally trained and defense models. Specifically, we apply a spectrum transformation to the input and thus perform the model augmentation in the frequency domain. We theoretically prove that the transformation derived from frequency domain leads to a diverse spectrum saliency map, an indicator we proposed to reflect the diversity of substitute models. Notably, our method can be generally combined with existing attacks. Extensive experiments on the ImageNet dataset demonstrate the effectiveness of our method, \textit{e.g.}, attacking nine state-of-the-art defense models with an average success rate of \textbf{95.4\%}. Our code is available in \url{https://github.com/yuyang-long/SSA}.

摘要: 对于黑盒攻击，替换模型与受害者模型之间的差距通常较大，表现为攻击性能较弱。基于同时攻击不同模型可以提高对抗性实例的可转移性这一观察结果，提出了利用变换后的图像模拟不同模型的模型增强方法。然而，现有的空间域变换并不能转化为显著不同的增强模型。为了解决这个问题，我们提出了一种新颖的频谱模拟攻击，以针对正常训练的模型和防御模型创建更多可转移的对抗性示例。具体地说，我们对输入应用频谱变换，从而在频域中执行模型增强。我们从理论上证明了从频域得到的变换导致了不同的频谱显著图，这是我们提出的反映替代模型多样性的一个指标。值得注意的是，我们的方法通常可以与现有攻击相结合。在ImageNet数据集上的大量实验证明了该方法的有效性，该方法攻击了9个最先进的防御模型，平均成功率为\textbf{95.4\}。我们的代码位于\url{https://github.com/yuyang-long/SSA}.



## **18. Bi-fidelity Evolutionary Multiobjective Search for Adversarially Robust Deep Neural Architectures**

双保真进化多目标搜索逆鲁棒深神经网络结构 cs.LG

**SubmitDate**: 2022-07-12    [paper-pdf](http://arxiv.org/pdf/2207.05321v1)

**Authors**: Jia Liu, Ran Cheng, Yaochu Jin

**Abstracts**: Deep neural networks have been found vulnerable to adversarial attacks, thus raising potentially concerns in security-sensitive contexts. To address this problem, recent research has investigated the adversarial robustness of deep neural networks from the architectural point of view. However, searching for architectures of deep neural networks is computationally expensive, particularly when coupled with adversarial training process. To meet the above challenge, this paper proposes a bi-fidelity multiobjective neural architecture search approach. First, we formulate the NAS problem for enhancing adversarial robustness of deep neural networks into a multiobjective optimization problem. Specifically, in addition to a low-fidelity performance predictor as the first objective, we leverage an auxiliary-objective -- the value of which is the output of a surrogate model trained with high-fidelity evaluations. Secondly, we reduce the computational cost by combining three performance estimation methods, i.e., parameter sharing, low-fidelity evaluation, and surrogate-based predictor. The effectiveness of the proposed approach is confirmed by extensive experiments conducted on CIFAR-10, CIFAR-100 and SVHN datasets.

摘要: 深度神经网络被发现容易受到敌意攻击，因此在安全敏感的环境中引发了潜在的担忧。为了解决这个问题，最近的研究从体系结构的角度研究了深度神经网络的对抗健壮性。然而，寻找深度神经网络的结构在计算上是昂贵的，特别是当与对抗性训练过程相结合时。为了应对上述挑战，本文提出了一种双保真多目标神经结构搜索方法。首先，我们将增强深层神经网络对抗健壮性的NAS问题转化为一个多目标优化问题。具体地说，除了作为第一个目标的低保真性能预测器之外，我们还利用一个辅助目标--其值是用高保真评估训练的代理模型的输出。其次，通过结合参数共享、低保真评估和基于代理的预测器三种性能估计方法来降低计算代价。在CIFAR-10、CIFAR-100和SVHN数据集上进行的大量实验证实了该方法的有效性。



## **19. Multitask Learning from Augmented Auxiliary Data for Improving Speech Emotion Recognition**

基于增强辅助数据的多任务学习改进语音情感识别 cs.SD

Under review IEEE Transactions on Affective Computing

**SubmitDate**: 2022-07-12    [paper-pdf](http://arxiv.org/pdf/2207.05298v1)

**Authors**: Siddique Latif, Rajib Rana, Sara Khalifa, Raja Jurdak, Björn W. Schuller

**Abstracts**: Despite the recent progress in speech emotion recognition (SER), state-of-the-art systems lack generalisation across different conditions. A key underlying reason for poor generalisation is the scarcity of emotion datasets, which is a significant roadblock to designing robust machine learning (ML) models. Recent works in SER focus on utilising multitask learning (MTL) methods to improve generalisation by learning shared representations. However, most of these studies propose MTL solutions with the requirement of meta labels for auxiliary tasks, which limits the training of SER systems. This paper proposes an MTL framework (MTL-AUG) that learns generalised representations from augmented data. We utilise augmentation-type classification and unsupervised reconstruction as auxiliary tasks, which allow training SER systems on augmented data without requiring any meta labels for auxiliary tasks. The semi-supervised nature of MTL-AUG allows for the exploitation of the abundant unlabelled data to further boost the performance of SER. We comprehensively evaluate the proposed framework in the following settings: (1) within corpus, (2) cross-corpus and cross-language, (3) noisy speech, (4) and adversarial attacks. Our evaluations using the widely used IEMOCAP, MSP-IMPROV, and EMODB datasets show improved results compared to existing state-of-the-art methods.

摘要: 尽管最近在语音情感识别(SER)方面取得了进展，但最先进的系统缺乏对不同条件的通用性。泛化能力差的一个关键根本原因是情感数据集的稀缺，这是设计健壮的机器学习(ML)模型的一个重要障碍。SER最近的工作集中在利用多任务学习(MTL)方法通过学习共享表示来提高泛化能力。然而，这些研究大多提出了辅助任务需要元标签的MTL解决方案，这限制了SER系统的训练。提出了一个从扩充数据中学习泛化表示的MTL框架(MTL-AUG)。我们使用增强型分类和无监督重建作为辅助任务，允许在增强型数据上训练SER系统，而不需要任何辅助任务的元标签。MTL-AUG的半监督性质允许利用丰富的未标记数据来进一步提高SER的性能。我们在以下几个方面对该框架进行了综合评估：(1)在语料库内，(2)跨语料库和跨语言，(3)噪声语音，(4)和对抗性攻击。我们使用广泛使用的IEMOCAP、MSP-Improv和EMODB数据集进行的评估显示，与现有最先进的方法相比，结果有所改善。



## **20. "Why do so?" -- A Practical Perspective on Machine Learning Security**

“为什么要这样做？”--机器学习安全的实践视角 cs.LG

under submission - 18 pages, 3 tables and 4 figures. Long version of  the paper accepted at: New Frontiers of Adversarial Machine Learning@ICML

**SubmitDate**: 2022-07-11    [paper-pdf](http://arxiv.org/pdf/2207.05164v1)

**Authors**: Kathrin Grosse, Lukas Bieringer, Tarek Richard Besold, Battista Biggio, Katharina Krombholz

**Abstracts**: Despite the large body of academic work on machine learning security, little is known about the occurrence of attacks on machine learning systems in the wild. In this paper, we report on a quantitative study with 139 industrial practitioners. We analyze attack occurrence and concern and evaluate statistical hypotheses on factors influencing threat perception and exposure. Our results shed light on real-world attacks on deployed machine learning. On the organizational level, while we find no predictors for threat exposure in our sample, the amount of implement defenses depends on exposure to threats or expected likelihood to become a target. We also provide a detailed analysis of practitioners' replies on the relevance of individual machine learning attacks, unveiling complex concerns like unreliable decision making, business information leakage, and bias introduction into models. Finally, we find that on the individual level, prior knowledge about machine learning security influences threat perception. Our work paves the way for more research about adversarial machine learning in practice, but yields also insights for regulation and auditing.

摘要: 尽管有大量关于机器学习安全的学术工作，但人们对野外发生的针对机器学习系统的攻击知之甚少。在本文中，我们报告了一项对139名工业从业者的定量研究。我们分析了攻击的发生和关注，并对影响威胁感知和暴露的因素进行了统计假设评估。我们的结果揭示了对部署的机器学习的真实世界攻击。在组织层面上，虽然我们在样本中没有发现威胁暴露的预测因素，但实施防御的数量取决于威胁暴露或成为目标的预期可能性。我们还提供了对从业者对单个机器学习攻击相关性的回复的详细分析，揭示了不可靠的决策、商业信息泄露和模型中的偏见引入等复杂问题。最后，我们发现在个体层面上，关于机器学习安全的先验知识会影响威胁感知。我们的工作为在实践中对对抗性机器学习进行更多的研究铺平了道路，但也为监管和审计提供了见解。



## **21. LQG Reference Tracking with Safety and Reachability Guarantees under Unknown False Data Injection Attacks**

未知虚假数据注入攻击下具有安全性和可达性保证的LQG参考跟踪 eess.SY

13 pages, 4 figures, extended version of a Transactions on Automatic  Control paper

**SubmitDate**: 2022-07-11    [paper-pdf](http://arxiv.org/pdf/2103.00387v2)

**Authors**: Zhouchi Li, Luyao Niu, Andrew Clark

**Abstracts**: We investigate a linear quadratic Gaussian (LQG) tracking problem with safety and reachability constraints in the presence of an adversary who mounts an FDI attack on an unknown set of sensors. For each possible set of compromised sensors, we maintain a state estimator disregarding the sensors in that set, and calculate the optimal LQG control input at each time based on this estimate. We propose a control policy which constrains the control input to lie within a fixed distance of the optimal control input corresponding to each state estimate. The control input is obtained at each time step by solving a quadratically constrained quadratic program (QCQP). We prove that our policy can achieve a desired probability of safety and reachability using the barrier certificate method. Our control policy is evaluated via a numerical case study.

摘要: 研究了一类具有安全性和可达性约束的线性二次型高斯(LQG)跟踪问题，其中敌手对未知传感器集发起了FDI攻击。对于每一组可能的受损传感器，我们维护一个状态估计器，而不考虑该集合中的传感器，并基于该估计计算每次的最优LQG控制输入。我们提出了一种控制策略，该策略将控制输入限制在与每个状态估计对应的最优控制输入的固定距离内。在每个时间步通过求解二次约束二次规划(QCQP)来获得控制输入。我们使用屏障证书方法证明了我们的策略可以达到期望的安全和可达性概率。我们的控制策略是通过一个数值案例研究来评估的。



## **22. Towards Effective Multi-Label Recognition Attacks via Knowledge Graph Consistency**

基于知识图一致性的高效多标签识别攻击 cs.CV

**SubmitDate**: 2022-07-11    [paper-pdf](http://arxiv.org/pdf/2207.05137v1)

**Authors**: Hassan Mahmood, Ehsan Elhamifar

**Abstracts**: Many real-world applications of image recognition require multi-label learning, whose goal is to find all labels in an image. Thus, robustness of such systems to adversarial image perturbations is extremely important. However, despite a large body of recent research on adversarial attacks, the scope of the existing works is mainly limited to the multi-class setting, where each image contains a single label. We show that the naive extensions of multi-class attacks to the multi-label setting lead to violating label relationships, modeled by a knowledge graph, and can be detected using a consistency verification scheme. Therefore, we propose a graph-consistent multi-label attack framework, which searches for small image perturbations that lead to misclassifying a desired target set while respecting label hierarchies. By extensive experiments on two datasets and using several multi-label recognition models, we show that our method generates extremely successful attacks that, unlike naive multi-label perturbations, can produce model predictions consistent with the knowledge graph.

摘要: 现实世界中的许多图像识别应用都需要多标签学习，其目标是找到图像中的所有标签。因此，这类系统对对抗性图像扰动的稳健性是极其重要的。然而，尽管最近对对抗性攻击进行了大量的研究，但现有的工作范围主要局限于多类背景下，其中每幅图像包含一个单一的标签。我们证明了多类攻击对多标签设置的天真扩展导致了违反标签关系，用知识图来建模，并且可以使用一致性验证方案来检测。因此，我们提出了一种图一致的多标签攻击框架，该框架在尊重标签层次的同时，搜索导致期望目标集错误分类的微小图像扰动。通过在两个数据集上的广泛实验和使用几个多标签识别模型，我们的方法产生了非常成功的攻击，不同于朴素的多标签扰动，我们可以产生与知识图一致的模型预测。



## **23. RUSH: Robust Contrastive Learning via Randomized Smoothing**

RASH：随机平滑的稳健对比学习 cs.LG

12 pages, 2 figures

**SubmitDate**: 2022-07-11    [paper-pdf](http://arxiv.org/pdf/2207.05127v1)

**Authors**: Yijiang Pang, Boyang Liu, Jiayu Zhou

**Abstracts**: Recently, adversarial training has been incorporated in self-supervised contrastive pre-training to augment label efficiency with exciting adversarial robustness. However, the robustness came at a cost of expensive adversarial training. In this paper, we show a surprising fact that contrastive pre-training has an interesting yet implicit connection with robustness, and such natural robustness in the pre trained representation enables us to design a powerful robust algorithm against adversarial attacks, RUSH, that combines the standard contrastive pre-training and randomized smoothing. It boosts both standard accuracy and robust accuracy, and significantly reduces training costs as compared with adversarial training. We use extensive empirical studies to show that the proposed RUSH outperforms robust classifiers from adversarial training, by a significant margin on common benchmarks (CIFAR-10, CIFAR-100, and STL-10) under first-order attacks. In particular, under $\ell_{\infty}$-norm perturbations of size 8/255 PGD attack on CIFAR-10, our model using ResNet-18 as backbone reached 77.8% robust accuracy and 87.9% standard accuracy. Our work has an improvement of over 15% in robust accuracy and a slight improvement in standard accuracy, compared to the state-of-the-arts.

摘要: 最近，对抗性训练被结合到自我监督的对比预训练中，以增强标记的效率和令人兴奋的对抗性健壮性。然而，这种健壮性是以昂贵的对抗性训练为代价的。在这篇文章中，我们展示了一个令人惊讶的事实，即对比预训练与稳健性有着有趣而隐含的联系，而预训练表示中的这种自然的稳健性使我们能够设计出一种结合了标准的对比预训练和随机平滑的强大的抗对手攻击的鲁棒算法RASH。与对抗性训练相比，它同时提高了标准准确率和稳健准确率，并显著降低了训练成本。我们使用广泛的实证研究表明，在一阶攻击下，所提出的冲刺算法在常见基准(CIFAR-10、CIFAR-100和STL-10)上的性能明显优于来自对手训练的稳健分类器。特别是，在CIFAR-10遭受8/255 PGD攻击时，以ResNet-18为主干的模型达到了77.8%的稳健准确率和87.9%的标准准确率。与最新水平相比，我们的工作在稳健精度上提高了15%以上，在标准精度上略有提高。



## **24. Physical Passive Patch Adversarial Attacks on Visual Odometry Systems**

对视觉里程计系统的物理被动补丁敌意攻击 cs.CV

**SubmitDate**: 2022-07-11    [paper-pdf](http://arxiv.org/pdf/2207.05729v1)

**Authors**: Yaniv Nemcovsky, Matan Yaakoby, Alex M. Bronstein, Chaim Baskin

**Abstracts**: Deep neural networks are known to be susceptible to adversarial perturbations -- small perturbations that alter the output of the network and exist under strict norm limitations. While such perturbations are usually discussed as tailored to a specific input, a universal perturbation can be constructed to alter the model's output on a set of inputs. Universal perturbations present a more realistic case of adversarial attacks, as awareness of the model's exact input is not required. In addition, the universal attack setting raises the subject of generalization to unseen data, where given a set of inputs, the universal perturbations aim to alter the model's output on out-of-sample data. In this work, we study physical passive patch adversarial attacks on visual odometry-based autonomous navigation systems. A visual odometry system aims to infer the relative camera motion between two corresponding viewpoints, and is frequently used by vision-based autonomous navigation systems to estimate their state. For such navigation systems, a patch adversarial perturbation poses a severe security issue, as it can be used to mislead a system onto some collision course. To the best of our knowledge, we show for the first time that the error margin of a visual odometry model can be significantly increased by deploying patch adversarial attacks in the scene. We provide evaluation on synthetic closed-loop drone navigation data and demonstrate that a comparable vulnerability exists in real data. A reference implementation of the proposed method and the reported experiments is provided at https://github.com/patchadversarialattacks/patchadversarialattacks.

摘要: 众所周知，深度神经网络容易受到对抗性扰动的影响--微小的扰动会改变网络的输出，并在严格的范数限制下存在。虽然这样的扰动通常被讨论为针对特定的输入而量身定做的，但是可以构造一个普遍的扰动来改变模型在一组输入上的输出。普遍摄动提供了一种更现实的对抗性攻击情况，因为不需要知道模型的确切输入。此外，通用攻击设置提出了对不可见数据的泛化主题，在给定一组输入的情况下，通用扰动的目的是改变模型对样本外数据的输出。在这项工作中，我们研究了基于视觉里程计的自主导航系统的物理被动补丁对抗性攻击。视觉里程计系统旨在推断两个相应视点之间的相机相对运动，经常被基于视觉的自主导航系统用来估计它们的状态。对于这样的导航系统，补丁对抗性扰动构成了一个严重的安全问题，因为它可能被用来误导系统进入某些碰撞路线。据我们所知，我们首次证明了在场景中部署补丁对抗性攻击可以显著增加视觉里程计模型的误差。我们对合成的无人机闭环导航数据进行了评估，并证明了真实数据中存在类似的漏洞。在https://github.com/patchadversarialattacks/patchadversarialattacks.上提供了所提出的方法和报告的实验的参考实现



## **25. Risk assessment and optimal allocation of security measures under stealthy false data injection attacks**

隐蔽虚假数据注入攻击下的风险评估与安全措施优化配置 eess.SY

Accepted for publication at 6th IEEE Conference on Control Technology  and Applications (CCTA). arXiv admin note: substantial text overlap with  arXiv:2106.07071

**SubmitDate**: 2022-07-11    [paper-pdf](http://arxiv.org/pdf/2207.04860v1)

**Authors**: Sribalaji C. Anand, André M. H. Teixeira, Anders Ahlén

**Abstracts**: This paper firstly addresses the problem of risk assessment under false data injection attacks on uncertain control systems. We consider an adversary with complete system knowledge, injecting stealthy false data into an uncertain control system. We then use the Value-at-Risk to characterize the risk associated with the attack impact caused by the adversary. The worst-case attack impact is characterized by the recently proposed output-to-output gain. We observe that the risk assessment problem corresponds to an infinite non-convex robust optimization problem. To this end, we use dissipative system theory and the scenario approach to approximate the risk-assessment problem into a convex problem and also provide probabilistic certificates on approximation. Secondly, we consider the problem of security measure allocation. We consider an operator with a constraint on the security budget. Under this constraint, we propose an algorithm to optimally allocate the security measures using the calculated risk such that the resulting Value-at-risk is minimized. Finally, we illustrate the results through a numerical example. The numerical example also illustrates that the security allocation using the Value-at-risk, and the impact on the nominal system may have different outcomes: thereby depicting the benefit of using risk metrics.

摘要: 本文首先研究了不确定控制系统在虚假数据注入攻击下的风险评估问题。我们考虑一个拥有完整系统知识的对手，向一个不确定的控制系统注入隐蔽的虚假数据。然后，我们使用风险值来表征与对手造成的攻击影响相关的风险。最坏情况的攻击影响的特征是最近提出的输出到输出的增益。我们观察到风险评估问题对应于一个无穷大的非凸稳健优化问题。为此，我们使用耗散系统理论和情景方法将风险评估问题近似化为一个凸问题，并给出了近似的概率证书。其次，我们考虑了安全措施分配问题。我们考虑一个对安全预算有限制的运营商。在此约束下，我们提出了一种算法，利用计算出的风险对安全措施进行最优分配，使得得到的风险值最小。最后，通过一个数值算例对结果进行了说明。数值例子还表明，使用在险价值的证券分配和对名义系统的影响可能会有不同的结果：从而描绘了使用风险度量的好处。



## **26. Statistical Detection of Adversarial examples in Blockchain-based Federated Forest In-vehicle Network Intrusion Detection Systems**

基于区块链的联邦森林车载网络入侵检测系统中恶意实例的统计检测 cs.CR

**SubmitDate**: 2022-07-11    [paper-pdf](http://arxiv.org/pdf/2207.04843v1)

**Authors**: Ibrahim Aliyu, Selinde van Engelenburg, Muhammed Bashir Muazu, Jinsul Kim, Chang Gyoon Lim

**Abstracts**: The internet-of-Vehicle (IoV) can facilitate seamless connectivity between connected vehicles (CV), autonomous vehicles (AV), and other IoV entities. Intrusion Detection Systems (IDSs) for IoV networks can rely on machine learning (ML) to protect the in-vehicle network from cyber-attacks. Blockchain-based Federated Forests (BFFs) could be used to train ML models based on data from IoV entities while protecting the confidentiality of the data and reducing the risks of tampering with the data. However, ML models created this way are still vulnerable to evasion, poisoning, and exploratory attacks using adversarial examples. This paper investigates the impact of various possible adversarial examples on the BFF-IDS. We proposed integrating a statistical detector to detect and extract unknown adversarial samples. By including the unknown detected samples into the dataset of the detector, we augment the BFF-IDS with an additional model to detect original known attacks and the new adversarial inputs. The statistical adversarial detector confidently detected adversarial examples at the sample size of 50 and 100 input samples. Furthermore, the augmented BFF-IDS (BFF-IDS(AUG)) successfully mitigates the adversarial examples with more than 96% accuracy. With this approach, the model will continue to be augmented in a sandbox whenever an adversarial sample is detected and subsequently adopt the BFF-IDS(AUG) as the active security model. Consequently, the proposed integration of the statistical adversarial detector and the subsequent augmentation of the BFF-IDS with detected adversarial samples provides a sustainable security framework against adversarial examples and other unknown attacks.

摘要: 车联网(IoV)可以促进互联车辆(CV)、自动驾驶车辆(AV)和其他IoV实体之间的无缝连接。IoV网络的入侵检测系统(IDS)可以依靠机器学习(ML)来保护车载网络免受网络攻击。基于区块链的联合森林(BFR)可用于基于来自IoV实体的数据训练ML模型，同时保护数据的机密性并降低篡改数据的风险。然而，以这种方式创建的ML模型仍然容易受到逃避、中毒和使用对抗性示例的探索性攻击。本文研究了各种可能的对抗性例子对BFF-IDS的影响。我们提出集成一个统计检测器来检测和提取未知对手样本。通过将未知的检测样本加入到检测器的数据集中，我们在BFF-IDS中增加了一个额外的模型来检测原始的已知攻击和新的敌意输入。统计敌意检测器在50个和100个输入样本的样本大小下自信地检测到对抗性例子。此外，扩展的BFF-IDS(BFF-IDS(AUG))成功地减少了对抗性实例，准确率超过96%。通过这种方法，只要检测到敌意样本，该模型就会继续在沙箱中进行扩充，并随后采用BFF-IDS(AUG)作为主动安全模型。因此，拟议整合敌意统计探测器，并随后利用检测到的敌意样本加强生物多样性框架--入侵检测系统，这提供了一个可持续的安全框架，可抵御敌意例子和其他未知攻击。



## **27. Physical Attack on Monocular Depth Estimation with Optimal Adversarial Patches**

基于最优对抗性斑块的单目深度估计的物理攻击 cs.CV

ECCV2022

**SubmitDate**: 2022-07-11    [paper-pdf](http://arxiv.org/pdf/2207.04718v1)

**Authors**: Zhiyuan Cheng, James Liang, Hongjun Choi, Guanhong Tao, Zhiwen Cao, Dongfang Liu, Xiangyu Zhang

**Abstracts**: Deep learning has substantially boosted the performance of Monocular Depth Estimation (MDE), a critical component in fully vision-based autonomous driving (AD) systems (e.g., Tesla and Toyota). In this work, we develop an attack against learning-based MDE. In particular, we use an optimization-based method to systematically generate stealthy physical-object-oriented adversarial patches to attack depth estimation. We balance the stealth and effectiveness of our attack with object-oriented adversarial design, sensitive region localization, and natural style camouflage. Using real-world driving scenarios, we evaluate our attack on concurrent MDE models and a representative downstream task for AD (i.e., 3D object detection). Experimental results show that our method can generate stealthy, effective, and robust adversarial patches for different target objects and models and achieves more than 6 meters mean depth estimation error and 93% attack success rate (ASR) in object detection with a patch of 1/9 of the vehicle's rear area. Field tests on three different driving routes with a real vehicle indicate that we cause over 6 meters mean depth estimation error and reduce the object detection rate from 90.70% to 5.16% in continuous video frames.

摘要: 深度学习大大提高了单目深度估计(MDE)的性能，单目深度估计是完全基于视觉的自动驾驶(AD)系统(例如特斯拉和丰田)中的关键组件。在这项工作中，我们开发了一个针对基于学习的MDE的攻击。特别是，我们使用了一种基于优化的方法来系统地生成隐身的面向物理对象的对抗性补丁来进行攻击深度估计。我们在攻击的隐蔽性和有效性与面向对象的对抗性设计、敏感区域定位和自然风格伪装之间取得平衡。使用真实的驾驶场景，我们评估了我们对并发MDE模型的攻击以及一个具有代表性的AD下游任务(即3D对象检测)。实验结果表明，该方法能够针对不同的目标对象和模型生成隐身、高效、健壮的对抗性补丁，在目标检测中以1/9的车尾面积实现6m以上的平均深度估计误差和93%的攻击成功率。对三条不同行驶路线的实际车辆进行的现场测试表明，在连续视频帧中，该算法会产生超过6m的平均深度估计误差，目标检测率从90.70%下降到5.16%。



## **28. Universal Adversarial Examples in Remote Sensing: Methodology and Benchmark**

遥感领域的普遍对抗性实例：方法论和基准 cs.CV

**SubmitDate**: 2022-07-10    [paper-pdf](http://arxiv.org/pdf/2202.07054v2)

**Authors**: Yonghao Xu, Pedram Ghamisi

**Abstracts**: Deep neural networks have achieved great success in many important remote sensing tasks. Nevertheless, their vulnerability to adversarial examples should not be neglected. In this study, we systematically analyze the universal adversarial examples in remote sensing data for the first time, without any knowledge from the victim model. Specifically, we propose a novel black-box adversarial attack method, namely Mixup-Attack, and its simple variant Mixcut-Attack, for remote sensing data. The key idea of the proposed methods is to find common vulnerabilities among different networks by attacking the features in the shallow layer of a given surrogate model. Despite their simplicity, the proposed methods can generate transferable adversarial examples that deceive most of the state-of-the-art deep neural networks in both scene classification and semantic segmentation tasks with high success rates. We further provide the generated universal adversarial examples in the dataset named UAE-RS, which is the first dataset that provides black-box adversarial samples in the remote sensing field. We hope UAE-RS may serve as a benchmark that helps researchers to design deep neural networks with strong resistance toward adversarial attacks in the remote sensing field. Codes and the UAE-RS dataset are available online (https://github.com/YonghaoXu/UAE-RS).

摘要: 深度神经网络在许多重要的遥感任务中取得了巨大的成功。然而，不应忽视它们在对抗性例子面前的脆弱性。在本研究中，我们首次在没有任何受害者模型知识的情况下，系统地分析了遥感数据中的通用对抗性实例。具体来说，我们提出了一种新的针对遥感数据的黑盒对抗攻击方法，即Mixup攻击及其简单的变种MixCut攻击。提出的方法的核心思想是通过攻击给定代理模型浅层的特征来发现不同网络之间的共同漏洞。尽管方法简单，但在场景分类和语义分割任务中，所提出的方法可以生成可转移的对抗性样本，欺骗了大多数最新的深度神经网络，并且成功率很高。此外，我们还在名为UAE-RS的数据集上提供了生成的通用对抗性实例，这是遥感领域中第一个提供黑盒对抗性样本的数据集。我们希望UAE-RS可以作为一个基准，帮助研究人员设计出对遥感领域的敌意攻击具有很强抵抗能力的深度神经网络。代码和阿联酋-RS数据集可在网上获得(https://github.com/YonghaoXu/UAE-RS).



## **29. Visual explanation of black-box model: Similarity Difference and Uniqueness (SIDU) method**

黑盒模型的可视化解释：相似性、差异性和唯一性(SIDU)方法 cs.CV

**SubmitDate**: 2022-07-10    [paper-pdf](http://arxiv.org/pdf/2101.10710v2)

**Authors**: Satya M. Muddamsetty, Mohammad N. S. Jahromi, Andreea E. Ciontos, Laura M. Fenoy, Thomas B. Moeslund

**Abstracts**: Explainable Artificial Intelligence (XAI) has in recent years become a well-suited framework to generate human understandable explanations of "black-box" models. In this paper, a novel XAI visual explanation algorithm known as the Similarity Difference and Uniqueness (SIDU) method that can effectively localize entire object regions responsible for prediction is presented in full detail. The SIDU algorithm robustness and effectiveness is analyzed through various computational and human subject experiments. In particular, the SIDU algorithm is assessed using three different types of evaluations (Application, Human and Functionally-Grounded) to demonstrate its superior performance. The robustness of SIDU is further studied in the presence of adversarial attack on "black-box" models to better understand its performance. Our code is available at: https://github.com/satyamahesh84/SIDU_XAI_CODE.

摘要: 近年来，可解释人工智能(XAI)已经成为一个非常适合生成人类可理解的“黑盒”模型解释的框架。提出了一种新的XAI视觉解释算法--相似性差值唯一性(SIDU)算法，该算法能够有效地定位预测的整个目标区域。通过各种计算和人体实验，分析了SIDU算法的健壮性和有效性。特别是，SIDU算法使用三种不同类型的评估(应用评估、人工评估和基于功能的评估)进行了评估，以展示其优越的性能。为了更好地理解SIDU的性能，进一步研究了SIDU在黑盒模型受到敌意攻击的情况下的稳健性。我们的代码请访问：https://github.com/satyamahesh84/SIDU_XAI_CODE.



## **30. Fooling Partial Dependence via Data Poisoning**

通过数据中毒愚弄部分依赖 cs.LG

Accepted at ECML PKDD 2022

**SubmitDate**: 2022-07-10    [paper-pdf](http://arxiv.org/pdf/2105.12837v3)

**Authors**: Hubert Baniecki, Wojciech Kretowicz, Przemyslaw Biecek

**Abstracts**: Many methods have been developed to understand complex predictive models and high expectations are placed on post-hoc model explainability. It turns out that such explanations are not robust nor trustworthy, and they can be fooled. This paper presents techniques for attacking Partial Dependence (plots, profiles, PDP), which are among the most popular methods of explaining any predictive model trained on tabular data. We showcase that PD can be manipulated in an adversarial manner, which is alarming, especially in financial or medical applications where auditability became a must-have trait supporting black-box machine learning. The fooling is performed via poisoning the data to bend and shift explanations in the desired direction using genetic and gradient algorithms. We believe this to be the first work using a genetic algorithm for manipulating explanations, which is transferable as it generalizes both ways: in a model-agnostic and an explanation-agnostic manner.

摘要: 已经开发了许多方法来理解复杂的预测模型，并对后自组织模型的可解释性寄予了很高的期望。事实证明，这样的解释既不可靠，也不可信，它们可能会被愚弄。这篇文章介绍了攻击部分相关性(曲线图、轮廓、PDP)的技术，这些技术是解释任何基于表格数据训练的预测模型的最流行的方法之一。我们展示了PD可以被以对抗的方式操纵，这是令人震惊的，特别是在金融或医疗应用中，可审计性成为支持黑盒机器学习的必备特征。这种愚弄是通过使用遗传和梯度算法毒化数据来弯曲和改变所需方向的解释来实现的。我们认为这是第一个使用遗传算法来操纵解释的工作，这是可以转移的，因为它概括了两种方式：以模型不可知论和解释不可知论的方式。



## **31. Adversarial Framework with Certified Robustness for Time-Series Domain via Statistical Features**

基于统计特征的时间序列域认证稳健性对抗框架 cs.LG

Published at Journal of Artificial Intelligence Research

**SubmitDate**: 2022-07-09    [paper-pdf](http://arxiv.org/pdf/2207.04307v1)

**Authors**: Taha Belkhouja, Janardhan Rao Doppa

**Abstracts**: Time-series data arises in many real-world applications (e.g., mobile health) and deep neural networks (DNNs) have shown great success in solving them. Despite their success, little is known about their robustness to adversarial attacks. In this paper, we propose a novel adversarial framework referred to as Time-Series Attacks via STATistical Features (TSA-STAT)}. To address the unique challenges of time-series domain, TSA-STAT employs constraints on statistical features of the time-series data to construct adversarial examples. Optimized polynomial transformations are used to create attacks that are more effective (in terms of successfully fooling DNNs) than those based on additive perturbations. We also provide certified bounds on the norm of the statistical features for constructing adversarial examples. Our experiments on diverse real-world benchmark datasets show the effectiveness of TSA-STAT in fooling DNNs for time-series domain and in improving their robustness. The source code of TSA-STAT algorithms is available at https://github.com/tahabelkhouja/Time-Series-Attacks-via-STATistical-Features

摘要: 时间序列数据出现在许多现实世界的应用中(例如移动医疗)，而深度神经网络(DNN)在解决这些问题上取得了巨大的成功。尽管它们取得了成功，但人们对它们对对手攻击的健壮性知之甚少。在本文中，我们提出了一种新的对抗性框架，称为基于统计特征的时间序列攻击(TSA-STAT)。为了解决时间序列领域的独特挑战，TSA-STAT使用了对时间序列数据的统计特征的约束来构造对抗性例子。优化的多项式变换用于创建比基于加性扰动的攻击更有效的攻击(就成功愚弄DNN而言)。我们还提供了用于构造对抗性例子的统计特征范数的确定界。我们在不同的真实基准数据集上的实验表明，TSA-STAT在时间序列域欺骗DNN和提高它们的稳健性方面是有效的。TSA-STAT算法的源代码可在https://github.com/tahabelkhouja/Time-Series-Attacks-via-STATistical-Features上找到



## **32. Not all broken defenses are equal: The dead angles of adversarial accuracy**

并不是所有破碎的防守都是平等的：对手准确性的死角 cs.LG

**SubmitDate**: 2022-07-08    [paper-pdf](http://arxiv.org/pdf/2207.04129v1)

**Authors**: Raphael Olivier, Bhiksha Raj

**Abstracts**: Robustness to adversarial attack is typically evaluated with adversarial accuracy. This metric is however too coarse to properly capture all robustness properties of machine learning models. Many defenses, when evaluated against a strong attack, do not provide accuracy improvements while still contributing partially to adversarial robustness. Popular certification methods suffer from the same issue, as they provide a lower bound to accuracy. To capture finer robustness properties we propose a new metric for L2 robustness, adversarial angular sparsity, which partially answers the question "how many adversarial examples are there around an input". We demonstrate its usefulness by evaluating both "strong" and "weak" defenses. We show that some state-of-the-art defenses, delivering very similar accuracy, can have very different sparsity on the inputs that they are not robust on. We also show that some weak defenses actually decrease robustness, while others strengthen it in a measure that accuracy cannot capture. These differences are predictive of how useful such defenses can become when combined with adversarial training.

摘要: 对敌方攻击的稳健性通常用敌方准确度来评估。然而，这个度量太粗糙了，无法正确地捕获机器学习模型的所有健壮性属性。许多防御在评估强大的攻击时，并不能提高准确率，同时仍能部分提高对手的健壮性。流行的认证方法也存在同样的问题，因为它们提供了准确度的下限。为了获得更好的稳健性，我们提出了一种新的二语稳健性度量--对抗性角度稀疏性，它部分地回答了“一个输入周围有多少个对抗性例子”的问题。我们通过评估“强”和“弱”防御来证明它的有效性。我们表明，一些最先进的防御，提供非常类似的准确性，可以在输入上具有非常不同的稀疏性，而它们的输入并不健壮。我们还表明，一些薄弱的防御实际上降低了稳健性，而另一些则以精确度无法捕捉的方式加强了稳健性。这些差异预示着，当这种防御与对抗性训练相结合时，会变得多么有用。



## **33. Neighbors From Hell: Voltage Attacks Against Deep Learning Accelerators on Multi-Tenant FPGAs**

地狱邻居：针对多租户现场可编程门阵列上深度学习加速器的电压攻击 cs.CR

Published in the 2020 proceedings of the International Conference of  Field-Programmable Technology (ICFPT)

**SubmitDate**: 2022-07-08    [paper-pdf](http://arxiv.org/pdf/2012.07242v2)

**Authors**: Andrew Boutros, Mathew Hall, Nicolas Papernot, Vaughn Betz

**Abstracts**: Field-programmable gate arrays (FPGAs) are becoming widely used accelerators for a myriad of datacenter applications due to their flexibility and energy efficiency. Among these applications, FPGAs have shown promising results in accelerating low-latency real-time deep learning (DL) inference, which is becoming an indispensable component of many end-user applications. With the emerging research direction towards virtualized cloud FPGAs that can be shared by multiple users, the security aspect of FPGA-based DL accelerators requires careful consideration. In this work, we evaluate the security of DL accelerators against voltage-based integrity attacks in a multitenant FPGA scenario. We first demonstrate the feasibility of such attacks on a state-of-the-art Stratix 10 card using different attacker circuits that are logically and physically isolated in a separate attacker role, and cannot be flagged as malicious circuits by conventional bitstream checkers. We show that aggressive clock gating, an effective power-saving technique, can also be a potential security threat in modern FPGAs. Then, we carry out the attack on a DL accelerator running ImageNet classification in the victim role to evaluate the inherent resilience of DL models against timing faults induced by the adversary. We find that even when using the strongest attacker circuit, the prediction accuracy of the DL accelerator is not compromised when running at its safe operating frequency. Furthermore, we can achieve 1.18-1.31x higher inference performance by over-clocking the DL accelerator without affecting its prediction accuracy.

摘要: 现场可编程门阵列(现场可编程门阵列)由于其灵活性和能效，正成为各种数据中心应用的广泛使用的加速器。在这些应用中，现场可编程门阵列在加速低延迟实时深度学习(DL)推理方面表现出了良好的效果，这正成为许多最终用户应用程序中不可或缺的组件。随着可由多个用户共享的虚拟云FPGA的研究方向的出现，基于FPGA的DL加速器的安全方面需要仔细考虑。在这项工作中，我们评估了在多租户现场可编程门阵列场景中，DL加速器抵抗基于电压的完整性攻击的安全性。我们首先证明了使用不同的攻击电路对最先进的Stratix 10卡进行此类攻击的可行性，这些攻击电路在逻辑上和物理上隔离在单独的攻击者角色中，并且不能被传统的比特流检查器标记为恶意电路。我们发现，积极的时钟门控技术是一种有效的节能技术，也可能成为现代现场可编程门阵列中的潜在安全威胁。然后，我们以受害者角色对运行ImageNet分类的DL加速器进行攻击，以评估DL模型对对手导致的时序错误的内在弹性。我们发现，即使在使用最强攻击电路的情况下，DL加速器在其安全工作频率下运行时的预测精度也不会受到影响。此外，通过对DL加速器超频，我们可以在不影响其预测精度的情况下获得1.18-1.31倍的高推理性能。



## **34. Defense Against Multi-target Trojan Attacks**

防御多目标木马攻击 cs.CV

**SubmitDate**: 2022-07-08    [paper-pdf](http://arxiv.org/pdf/2207.03895v1)

**Authors**: Haripriya Harikumar, Santu Rana, Kien Do, Sunil Gupta, Wei Zong, Willy Susilo, Svetha Venkastesh

**Abstracts**: Adversarial attacks on deep learning-based models pose a significant threat to the current AI infrastructure. Among them, Trojan attacks are the hardest to defend against. In this paper, we first introduce a variation of the Badnet kind of attacks that introduces Trojan backdoors to multiple target classes and allows triggers to be placed anywhere in the image. The former makes it more potent and the latter makes it extremely easy to carry out the attack in the physical space. The state-of-the-art Trojan detection methods fail with this threat model. To defend against this attack, we first introduce a trigger reverse-engineering mechanism that uses multiple images to recover a variety of potential triggers. We then propose a detection mechanism by measuring the transferability of such recovered triggers. A Trojan trigger will have very high transferability i.e. they make other images also go to the same class. We study many practical advantages of our attack method and then demonstrate the detection performance using a variety of image datasets. The experimental results show the superior detection performance of our method over the state-of-the-arts.

摘要: 对基于深度学习的模型的对抗性攻击对当前的人工智能基础设施构成了重大威胁。其中，木马攻击是最难防御的。在本文中，我们首先介绍了Badnet类型的攻击的一个变体，该攻击将特洛伊木马后门引入多个目标类，并允许在图像中的任何位置放置触发器。前者使其更具威力，而后者使在物理空间进行攻击变得极其容易。最先进的特洛伊木马检测方法在此威胁模型下失败。为了防御这种攻击，我们首先引入了一种触发反向工程机制，该机制使用多个图像来恢复各种潜在的触发。然后，我们提出了一种通过测量这些恢复的触发器的可转移性来检测的机制。特洛伊木马触发器将具有非常高的可转移性，即它们使其他图像也进入同一类。我们研究了该攻击方法的许多实用优势，并使用各种图像数据集演示了该方法的检测性能。实验结果表明，该方法具有较好的检测性能。



## **35. Towards Effective and Robust Neural Trojan Defenses via Input Filtering**

通过输入过滤实现高效、健壮的神经木马防御 cs.CR

Accepted to ECCV 2022

**SubmitDate**: 2022-07-07    [paper-pdf](http://arxiv.org/pdf/2202.12154v4)

**Authors**: Kien Do, Haripriya Harikumar, Hung Le, Dung Nguyen, Truyen Tran, Santu Rana, Dang Nguyen, Willy Susilo, Svetha Venkatesh

**Abstracts**: Trojan attacks on deep neural networks are both dangerous and surreptitious. Over the past few years, Trojan attacks have advanced from using only a single input-agnostic trigger and targeting only one class to using multiple, input-specific triggers and targeting multiple classes. However, Trojan defenses have not caught up with this development. Most defense methods still make inadequate assumptions about Trojan triggers and target classes, thus, can be easily circumvented by modern Trojan attacks. To deal with this problem, we propose two novel "filtering" defenses called Variational Input Filtering (VIF) and Adversarial Input Filtering (AIF) which leverage lossy data compression and adversarial learning respectively to effectively purify potential Trojan triggers in the input at run time without making assumptions about the number of triggers/target classes or the input dependence property of triggers. In addition, we introduce a new defense mechanism called "Filtering-then-Contrasting" (FtC) which helps avoid the drop in classification accuracy on clean data caused by "filtering", and combine it with VIF/AIF to derive new defenses of this kind. Extensive experimental results and ablation studies show that our proposed defenses significantly outperform well-known baseline defenses in mitigating five advanced Trojan attacks including two recent state-of-the-art while being quite robust to small amounts of training data and large-norm triggers.

摘要: 特洛伊木马对深层神经网络的攻击既危险又隐蔽。在过去的几年中，特洛伊木马攻击已经从只使用一个与输入无关的触发器和只针对一个类发展到使用多个特定于输入的触发器和目标多个类。然而，特洛伊木马防御并没有跟上这一发展。大多数防御方法仍然对木马的触发器和目标类别做了不充分的假设，因此很容易被现代木马攻击所规避。针对这一问题，我们提出了两种新的“过滤”防御机制，称为变输入过滤(VIF)和对抗输入过滤(AIF)，它们分别利用有损数据压缩和对抗学习在运行时有效地净化输入中潜在的特洛伊木马触发器，而不需要假设触发器/目标类的数量或触发器的输入依赖属性。此外，我们还引入了一种新的防御机制，称为“过滤-然后-对比”(FTC)，它有助于避免“过滤”导致对干净数据的分类精度的下降，并将其与VIF/AIF相结合来派生出这种新的防御机制。广泛的实验结果和烧蚀研究表明，我们提出的防御方案在缓解五种高级特洛伊木马攻击(包括两种最新的木马攻击)方面明显优于众所周知的基线防御方案，同时对少量训练数据和大范数触发事件具有相当的健壮性。



## **36. On the Relationship Between Adversarial Robustness and Decision Region in Deep Neural Network**

深度神经网络中敌方稳健性与决策区域的关系 cs.LG

14 pages

**SubmitDate**: 2022-07-07    [paper-pdf](http://arxiv.org/pdf/2207.03400v1)

**Authors**: Seongjin Park, Haedong Jeong, Giyoung Jeon, Jaesik Choi

**Abstracts**: In general, Deep Neural Networks (DNNs) are evaluated by the generalization performance measured on unseen data excluded from the training phase. Along with the development of DNNs, the generalization performance converges to the state-of-the-art and it becomes difficult to evaluate DNNs solely based on this metric. The robustness against adversarial attack has been used as an additional metric to evaluate DNNs by measuring their vulnerability. However, few studies have been performed to analyze the adversarial robustness in terms of the geometry in DNNs. In this work, we perform an empirical study to analyze the internal properties of DNNs that affect model robustness under adversarial attacks. In particular, we propose the novel concept of the Populated Region Set (PRS), where training samples are populated more frequently, to represent the internal properties of DNNs in a practical setting. From systematic experiments with the proposed concept, we provide empirical evidence to validate that a low PRS ratio has a strong relationship with the adversarial robustness of DNNs. We also devise PRS regularizer leveraging the characteristics of PRS to improve the adversarial robustness without adversarial training.

摘要: 通常，深度神经网络(DNN)是通过在训练阶段排除的未知数据上测量的泛化性能来评估的。随着DNN的发展，泛化性能趋于最新水平，单纯基于这一指标来评价DNN变得越来越困难。对敌意攻击的健壮性已被用作通过测量DNN的脆弱性来评估DNN的额外度量。然而，很少有研究从DNN的几何角度来分析敌手的稳健性。在这项工作中，我们进行了一项实证研究，以分析在敌意攻击下影响模型稳健性的DNN的内部属性。特别是，我们提出了填充训练样本频率更高的填充区域集合(Prs)的新概念，以表示实际环境中DNN的内在属性。通过对所提出概念的系统实验，我们提供了经验证据来验证较低的粗糙集比率与DNN的对抗健壮性有很强的关系。利用粗糙集的特点，设计了粗糙集正则化算法，在不需要对手训练的情况下提高了对手的健壮性。



## **37. Federated Robustness Propagation: Sharing Robustness in Heterogeneous Federated Learning**

联邦健壮性传播：异质联邦学习中的健壮性共享 cs.LG

**SubmitDate**: 2022-07-07    [paper-pdf](http://arxiv.org/pdf/2106.10196v2)

**Authors**: Junyuan Hong, Haotao Wang, Zhangyang Wang, Jiayu Zhou

**Abstracts**: Federated learning (FL) emerges as a popular distributed learning schema that learns a model from a set of participating users without sharing raw data. One major challenge of FL comes with heterogeneous users, who may have distributionally different (or non-iid) data and varying computation resources. As federated users would use the model for prediction, they often demand the trained model to be robust against malicious attackers at test time. Whereas adversarial training (AT) provides a sound solution for centralized learning, extending its usage for federated users has imposed significant challenges, as many users may have very limited training data and tight computational budgets, to afford the data-hungry and costly AT. In this paper, we study a novel FL strategy: propagating adversarial robustness from rich-resource users that can afford AT, to those with poor resources that cannot afford it, during federated learning. We show that existing FL techniques cannot be effectively integrated with the strategy to propagate robustness among non-iid users and propose an efficient propagation approach by the proper use of batch-normalization. We demonstrate the rationality and effectiveness of our method through extensive experiments. Especially, the proposed method is shown to grant federated models remarkable robustness even when only a small portion of users afford AT during learning. Source code will be released.

摘要: 联合学习(FL)是一种流行的分布式学习模式，它从一组参与的用户那里学习模型，而不共享原始数据。FL的一个主要挑战来自不同的用户，他们可能具有分布不同(或非IID)的数据和不同的计算资源。由于联合用户将使用该模型进行预测，因此他们经常要求训练后的模型在测试时对恶意攻击者具有健壮性。虽然对抗训练(AT)为集中学习提供了一种合理的解决方案，但扩展其对联合用户的使用带来了巨大的挑战，因为许多用户可能具有非常有限的训练数据和紧张的计算预算，以负担数据匮乏和成本高昂的AT。在本文中，我们研究了一种新的FL策略：在联合学习过程中，将对抗健壮性从有能力负担AT的资源丰富的用户传播到资源贫乏的用户。我们证明了现有的FL技术不能有效地与在非IID用户之间传播健壮性的策略相结合，并通过适当地使用批处理归一化来提出一种有效的传播方法。通过大量的实验验证了该方法的合理性和有效性。特别是，在只有一小部分用户在学习过程中提供AT的情况下，所提出的方法被证明具有显著的健壮性。源代码将会公布。



## **38. SYNFI: Pre-Silicon Fault Analysis of an Open-Source Secure Element**

SYNFI：一种开源安全元件的硅前故障分析 cs.CR

**SubmitDate**: 2022-07-07    [paper-pdf](http://arxiv.org/pdf/2205.04775v2)

**Authors**: Pascal Nasahl, Miguel Osorio, Pirmin Vogel, Michael Schaffner, Timothy Trippel, Dominic Rizzo, Stefan Mangard

**Abstracts**: Fault attacks are active, physical attacks that an adversary can leverage to alter the control-flow of embedded devices to gain access to sensitive information or bypass protection mechanisms. Due to the severity of these attacks, manufacturers deploy hardware-based fault defenses into security-critical systems, such as secure elements. The development of these countermeasures is a challenging task due to the complex interplay of circuit components and because contemporary design automation tools tend to optimize inserted structures away, thereby defeating their purpose. Hence, it is critical that such countermeasures are rigorously verified post-synthesis. As classical functional verification techniques fall short of assessing the effectiveness of countermeasures, developers have to resort to methods capable of injecting faults in a simulation testbench or into a physical chip. However, developing test sequences to inject faults in simulation is an error-prone task and performing fault attacks on a chip requires specialized equipment and is incredibly time-consuming. To that end, this paper introduces SYNFI, a formal pre-silicon fault verification framework that operates on synthesized netlists. SYNFI can be used to analyze the general effect of faults on the input-output relationship in a circuit and its fault countermeasures, and thus enables hardware designers to assess and verify the effectiveness of embedded countermeasures in a systematic and semi-automatic way. To demonstrate that SYNFI is capable of handling unmodified, industry-grade netlists synthesized with commercial and open tools, we analyze OpenTitan, the first open-source secure element. In our analysis, we identified critical security weaknesses in the unprotected AES block, developed targeted countermeasures, reassessed their security, and contributed these countermeasures back to the OpenTitan repository.

摘要: 故障攻击是一种主动的物理攻击，攻击者可以利用这些攻击来改变嵌入式设备的控制流，从而获得对敏感信息的访问权限或绕过保护机制。由于这些攻击的严重性，制造商将基于硬件的故障防御部署到安全关键系统中，例如安全元件。这些对策的开发是一项具有挑战性的任务，因为电路元件之间的复杂相互作用，以及现代设计自动化工具倾向于优化插入的结构，从而违背了它们的目的。因此，至关重要的是，这些对策在合成后得到严格验证。由于传统的功能验证技术无法评估对策的有效性，开发人员不得不求助于能够在模拟测试台或物理芯片中注入故障的方法。然而，开发测试序列以在模拟中注入故障是一项容易出错的任务，在芯片上执行故障攻击需要专门的设备，并且非常耗时。为此，本文引入了SYNFI，这是一个运行在合成网表上的形式化的预硅故障验证框架。SYNFI可以用来分析故障对电路输入输出关系的一般影响及其故障对策，从而使硬件设计者能够以系统和半自动的方式评估和验证嵌入式对策的有效性。为了证明SYNFI能够处理使用商业和开放工具合成的未经修改的工业级网表，我们分析了第一个开源安全元素OpenTitan。在我们的分析中，我们确定了未受保护的AES块中的关键安全漏洞，开发了有针对性的对策，重新评估了它们的安全性，并将这些对策贡献给了OpenTitan存储库。



## **39. The StarCraft Multi-Agent Challenges+ : Learning of Multi-Stage Tasks and Environmental Factors without Precise Reward Functions**

星际争霸多智能体挑战+：没有精确奖励函数的多阶段任务和环境因素的学习 cs.LG

ICML Workshop: AI for Agent Based Modeling 2022 Spotlight

**SubmitDate**: 2022-07-07    [paper-pdf](http://arxiv.org/pdf/2207.02007v2)

**Authors**: Mingyu Kim, Jihwan Oh, Yongsik Lee, Joonkee Kim, Seonghwan Kim, Song Chong, Se-Young Yun

**Abstracts**: In this paper, we propose a novel benchmark called the StarCraft Multi-Agent Challenges+, where agents learn to perform multi-stage tasks and to use environmental factors without precise reward functions. The previous challenges (SMAC) recognized as a standard benchmark of Multi-Agent Reinforcement Learning are mainly concerned with ensuring that all agents cooperatively eliminate approaching adversaries only through fine manipulation with obvious reward functions. This challenge, on the other hand, is interested in the exploration capability of MARL algorithms to efficiently learn implicit multi-stage tasks and environmental factors as well as micro-control. This study covers both offensive and defensive scenarios. In the offensive scenarios, agents must learn to first find opponents and then eliminate them. The defensive scenarios require agents to use topographic features. For example, agents need to position themselves behind protective structures to make it harder for enemies to attack. We investigate MARL algorithms under SMAC+ and observe that recent approaches work well in similar settings to the previous challenges, but misbehave in offensive scenarios. Additionally, we observe that an enhanced exploration approach has a positive effect on performance but is not able to completely solve all scenarios. This study proposes new directions for future research.

摘要: 在本文中，我们提出了一个新的基准测试称为星际争霸多代理挑战+，其中代理学习执行多阶段任务和使用环境因素，而不是精确的奖励函数。以前的挑战(SMAC)被公认为多智能体强化学习的标准基准，主要涉及确保所有智能体只有通过具有明显奖励功能的精细操纵才能合作地消除逼近的对手。另一方面，这一挑战与Marl算法高效学习隐式多阶段任务和环境因素以及微控制的探索能力有关。这项研究涵盖了进攻和防守两种情况。在进攻场景中，代理人必须学会首先找到对手，然后消灭他们。防御场景要求特工使用地形特征。例如，特工需要将自己部署在保护性建筑后面，以使敌人更难发动攻击。我们研究了SMAC+下的Marl算法，观察到最近的方法在类似于之前的挑战的环境下工作得很好，但在攻击性场景中表现不佳。此外，我们观察到，增强的探索方法对性能有积极影响，但并不能完全解决所有情况。本研究为今后的研究提出了新的方向。



## **40. The Weaknesses of Adversarial Camouflage in Overhead Imagery**

头顶图像中对抗性伪装的弱点 cs.CV

7 pages, 15 figures

**SubmitDate**: 2022-07-06    [paper-pdf](http://arxiv.org/pdf/2207.02963v1)

**Authors**: Adam Van Etten

**Abstracts**: Machine learning is increasingly critical for analysis of the ever-growing corpora of overhead imagery. Advanced computer vision object detection techniques have demonstrated great success in identifying objects of interest such as ships, automobiles, and aircraft from satellite and drone imagery. Yet relying on computer vision opens up significant vulnerabilities, namely, the susceptibility of object detection algorithms to adversarial attacks. In this paper we explore the efficacy and drawbacks of adversarial camouflage in an overhead imagery context. While a number of recent papers have demonstrated the ability to reliably fool deep learning classifiers and object detectors with adversarial patches, most of this work has been performed on relatively uniform datasets and only a single class of objects. In this work we utilize the VisDrone dataset, which has a large range of perspectives and object sizes. We explore four different object classes: bus, car, truck, van. We build a library of 24 adversarial patches to disguise these objects, and introduce a patch translucency variable to our patches. The translucency (or alpha value) of the patches is highly correlated to their efficacy. Further, we show that while adversarial patches may fool object detectors, the presence of such patches is often easily uncovered, with patches on average 24% more detectable than the objects the patches were meant to hide. This raises the question of whether such patches truly constitute camouflage. Source code is available at https://github.com/IQTLabs/camolo.

摘要: 机器学习对于分析不断增长的头顶图像语料库越来越重要。先进的计算机视觉目标检测技术在从卫星和无人机图像中识别船舶、汽车和飞机等感兴趣目标方面取得了巨大成功。然而，依赖计算机视觉会暴露出重大的漏洞，即目标检测算法对对手攻击的敏感性。在这篇文章中，我们探讨了在头顶图像背景下对抗性伪装的有效性和缺陷。虽然最近的一些论文已经证明了用对抗性补丁可靠地愚弄深度学习分类器和对象检测器的能力，但大多数工作都是在相对统一的数据集上进行的，并且只有一类对象。在这项工作中，我们利用了VisDrone数据集，它具有很大范围的透视图和对象大小。我们探索了四个不同的对象类：公共汽车、轿车、卡车、货车。我们构建了一个由24个敌意补丁组成的库来伪装这些对象，并在我们的补丁中引入了一个补丁半透明变量。贴片的半透明性(或阿尔法值)与它们的疗效高度相关。此外，我们还表明，尽管敌意补丁可能会愚弄对象检测器，但此类补丁的存在通常很容易被发现，补丁的可检测性平均比补丁要隐藏的对象高24%。这引发了这样一个问题：这样的补丁是否真的构成了伪装。源代码可在https://github.com/IQTLabs/camolo.上获得



## **41. DeepAdversaries: Examining the Robustness of Deep Learning Models for Galaxy Morphology Classification**

深度学习：检验深度学习模型对星系形态分类的稳健性 cs.LG

20 pages, 6 figures, 5 tables; accepted in MLST

**SubmitDate**: 2022-07-06    [paper-pdf](http://arxiv.org/pdf/2112.14299v3)

**Authors**: Aleksandra Ćiprijanović, Diana Kafkes, Gregory Snyder, F. Javier Sánchez, Gabriel Nathan Perdue, Kevin Pedro, Brian Nord, Sandeep Madireddy, Stefan M. Wild

**Abstracts**: With increased adoption of supervised deep learning methods for processing and analysis of cosmological survey data, the assessment of data perturbation effects (that can naturally occur in the data processing and analysis pipelines) and the development of methods that increase model robustness are increasingly important. In the context of morphological classification of galaxies, we study the effects of perturbations in imaging data. In particular, we examine the consequences of using neural networks when training on baseline data and testing on perturbed data. We consider perturbations associated with two primary sources: 1) increased observational noise as represented by higher levels of Poisson noise and 2) data processing noise incurred by steps such as image compression or telescope errors as represented by one-pixel adversarial attacks. We also test the efficacy of domain adaptation techniques in mitigating the perturbation-driven errors. We use classification accuracy, latent space visualizations, and latent space distance to assess model robustness. Without domain adaptation, we find that processing pixel-level errors easily flip the classification into an incorrect class and that higher observational noise makes the model trained on low-noise data unable to classify galaxy morphologies. On the other hand, we show that training with domain adaptation improves model robustness and mitigates the effects of these perturbations, improving the classification accuracy by 23% on data with higher observational noise. Domain adaptation also increases by a factor of ~2.3 the latent space distance between the baseline and the incorrectly classified one-pixel perturbed image, making the model more robust to inadvertent perturbations.

摘要: 随着越来越多的人采用有监督的深度学习方法来处理和分析宇宙测量数据，对数据扰动影响的评估(数据处理和分析管道中自然会发生)以及提高模型稳健性的方法的开发变得越来越重要。在星系形态分类的背景下，我们研究了成像数据中微扰的影响。特别是，我们检查了在对基线数据进行训练和对扰动数据进行测试时使用神经网络的后果。我们考虑与两个主要来源相关的扰动：1)以更高水平的泊松噪声为代表的观测噪声的增加；2)以单像素对抗性攻击为代表的图像压缩或望远镜误差等步骤所引起的数据处理噪声。我们还测试了领域自适应技术在缓解扰动驱动的错误方面的有效性。我们使用分类精度、潜在空间可视化和潜在空间距离来评估模型的稳健性。在没有域自适应的情况下，我们发现处理像素级误差很容易将分类反转到不正确的类别，并且更高的观测噪声使得基于低噪声数据训练的模型无法对星系形态进行分类。另一方面，我们表明，域自适应训练提高了模型的稳健性，缓解了这些扰动的影响，在观测噪声较高的数据上将分类精度提高了23%。域自适应还将基线和错误分类的单像素扰动图像之间的潜在空间距离增加了约2.3倍，使模型对无意扰动更具鲁棒性。



## **42. Enhancing Adversarial Attacks on Single-Layer NVM Crossbar-Based Neural Networks with Power Consumption Information**

利用功耗信息增强单层NVM Crosbar神经网络的敌意攻击 cs.LG

**SubmitDate**: 2022-07-06    [paper-pdf](http://arxiv.org/pdf/2207.02764v1)

**Authors**: Cory Merkel

**Abstracts**: Adversarial attacks on state-of-the-art machine learning models pose a significant threat to the safety and security of mission-critical autonomous systems. This paper considers the additional vulnerability of machine learning models when attackers can measure the power consumption of their underlying hardware platform. In particular, we explore the utility of power consumption information for adversarial attacks on non-volatile memory crossbar-based single-layer neural networks. Our results from experiments with MNIST and CIFAR-10 datasets show that power consumption can reveal important information about the neural network's weight matrix, such as the 1-norm of its columns. That information can be used to infer the sensitivity of the network's loss with respect to different inputs. We also find that surrogate-based black box attacks that utilize crossbar power information can lead to improved attack efficiency.

摘要: 对最先进的机器学习模型的对抗性攻击对任务关键型自主系统的安全构成了严重威胁。本文考虑了当攻击者可以测量其底层硬件平台的功耗时，机器学习模型的附加漏洞。特别是，我们探索了功耗信息在基于非易失性记忆交叉开关的单层神经网络上的对抗性攻击中的效用。我们在MNIST和CIFAR-10数据集上的实验结果表明，功耗可以揭示有关神经网络权重矩阵的重要信息，例如其列的1范数。该信息可用于推断网络损耗对不同输入的敏感性。我们还发现，利用纵横制能量信息的基于代理的黑盒攻击可以提高攻击效率。



## **43. LADDER: Latent Boundary-guided Adversarial Training**

梯子：潜在边界制导的对抗性训练 cs.LG

To appear in Machine Learning

**SubmitDate**: 2022-07-06    [paper-pdf](http://arxiv.org/pdf/2206.03717v2)

**Authors**: Xiaowei Zhou, Ivor W. Tsang, Jie Yin

**Abstracts**: Deep Neural Networks (DNNs) have recently achieved great success in many classification tasks. Unfortunately, they are vulnerable to adversarial attacks that generate adversarial examples with a small perturbation to fool DNN models, especially in model sharing scenarios. Adversarial training is proved to be the most effective strategy that injects adversarial examples into model training to improve the robustness of DNN models against adversarial attacks. However, adversarial training based on the existing adversarial examples fails to generalize well to standard, unperturbed test data. To achieve a better trade-off between standard accuracy and adversarial robustness, we propose a novel adversarial training framework called LAtent bounDary-guided aDvErsarial tRaining (LADDER) that adversarially trains DNN models on latent boundary-guided adversarial examples. As opposed to most of the existing methods that generate adversarial examples in the input space, LADDER generates a myriad of high-quality adversarial examples through adding perturbations to latent features. The perturbations are made along the normal of the decision boundary constructed by an SVM with an attention mechanism. We analyze the merits of our generated boundary-guided adversarial examples from a boundary field perspective and visualization view. Extensive experiments and detailed analysis on MNIST, SVHN, CelebA, and CIFAR-10 validate the effectiveness of LADDER in achieving a better trade-off between standard accuracy and adversarial robustness as compared with vanilla DNNs and competitive baselines.

摘要: 近年来，深度神经网络(DNN)在许多分类任务中取得了巨大的成功。不幸的是，它们很容易受到对抗性攻击，生成带有微小扰动的对抗性示例来愚弄DNN模型，特别是在模型共享场景中。对抗性训练被证明是将对抗性样本注入模型训练以提高DNN模型对对抗性攻击的鲁棒性的最有效的策略。然而，基于现有对抗性实例的对抗性训练不能很好地推广到标准的、不受干扰的测试数据。为了在标准准确率和对手健壮性之间实现更好的折衷，我们提出了一种新的对手训练框架，称为潜在边界制导的对抗性训练(LIDA)，该框架针对潜在的边界制导的对抗性实例对DNN模型进行对抗性训练。与大多数现有的在输入空间生成对抗性实例的方法不同，梯形图通过对潜在特征添加扰动来生成大量高质量的对抗性实例。扰动沿具有注意力机制的支持向量机构造的决策边界的法线进行。我们从边界场的角度和可视化的角度分析了我们生成的边界制导的对抗性例子的优点。在MNIST、SVHN、CelebA和CIFAR-10上的大量实验和详细分析验证了梯形算法在标准准确率和对手健壮性之间取得了比普通DNN和竞争基线更好的折衷。



## **44. Adversarial Robustness of Visual Dialog**

可视化对话框的对抗性健壮性 cs.CV

**SubmitDate**: 2022-07-06    [paper-pdf](http://arxiv.org/pdf/2207.02639v1)

**Authors**: Lu Yu, Verena Rieser

**Abstracts**: Adversarial robustness evaluates the worst-case performance scenario of a machine learning model to ensure its safety and reliability. This study is the first to investigate the robustness of visually grounded dialog models towards textual attacks. These attacks represent a worst-case scenario where the input question contains a synonym which causes the previously correct model to return a wrong answer. Using this scenario, we first aim to understand how multimodal input components contribute to model robustness. Our results show that models which encode dialog history are more robust, and when launching an attack on history, model prediction becomes more uncertain. This is in contrast to prior work which finds that dialog history is negligible for model performance on this task. We also evaluate how to generate adversarial test examples which successfully fool the model but remain undetected by the user/software designer. We find that the textual, as well as the visual context are important to generate plausible worst-case scenarios.

摘要: 对抗健壮性评估机器学习模型的最差性能场景，以确保其安全性和可靠性。本研究首次研究了基于视觉的对话模型对文本攻击的稳健性。这些攻击代表了最坏的情况，其中输入问题包含同义词，这会导致先前正确的模型返回错误的答案。使用这个场景，我们首先要了解多模式输入组件如何有助于模型的健壮性。我们的结果表明，编码对话历史的模型更健壮，当对历史发起攻击时，模型预测变得更不确定。这与以前的工作形成对比，以前的工作发现对话历史对于此任务的模型性能可以忽略不计。我们还评估了如何生成敌意测试用例，这些测试用例成功地愚弄了模型，但仍然没有被用户/软件设计者发现。我们发现，文本环境和视觉环境对于生成看似合理的最坏情况都很重要。



## **45. Adversarial Mask: Real-World Universal Adversarial Attack on Face Recognition Models**

对抗性面具：对人脸识别模型的真实世界通用对抗性攻击 cs.CV

16

**SubmitDate**: 2022-07-06    [paper-pdf](http://arxiv.org/pdf/2111.10759v2)

**Authors**: Alon Zolfi, Shai Avidan, Yuval Elovici, Asaf Shabtai

**Abstracts**: Deep learning-based facial recognition (FR) models have demonstrated state-of-the-art performance in the past few years, even when wearing protective medical face masks became commonplace during the COVID-19 pandemic. Given the outstanding performance of these models, the machine learning research community has shown increasing interest in challenging their robustness. Initially, researchers presented adversarial attacks in the digital domain, and later the attacks were transferred to the physical domain. However, in many cases, attacks in the physical domain are conspicuous, and thus may raise suspicion in real-world environments (e.g., airports). In this paper, we propose Adversarial Mask, a physical universal adversarial perturbation (UAP) against state-of-the-art FR models that is applied on face masks in the form of a carefully crafted pattern. In our experiments, we examined the transferability of our adversarial mask to a wide range of FR model architectures and datasets. In addition, we validated our adversarial mask's effectiveness in real-world experiments (CCTV use case) by printing the adversarial pattern on a fabric face mask. In these experiments, the FR system was only able to identify 3.34% of the participants wearing the mask (compared to a minimum of 83.34% with other evaluated masks). A demo of our experiments can be found at: https://youtu.be/_TXkDO5z11w.

摘要: 基于深度学习的面部识别(FR)模型在过去几年展示了最先进的性能，即使在新冠肺炎大流行期间戴防护性医用口罩变得司空见惯。鉴于这些模型的出色性能，机器学习研究界对挑战它们的稳健性表现出越来越大的兴趣。最初，研究人员在数字领域提出了对抗性攻击，后来攻击被转移到物理领域。然而，在许多情况下，物理域中的攻击很明显，因此可能会在现实环境(例如机场)中引起怀疑。在本文中，我们提出了对抗面具，一种针对最先进的FR模型的物理通用对抗扰动(UAP)，它以精心制作的模式的形式应用于人脸面具上。在我们的实验中，我们检查了我们的对手面具在广泛的FR模型体系结构和数据集上的可转移性。此外，我们在真实世界的实验中(CCTV用例)验证了我们的对抗面具的有效性，通过将对抗图案打印在织物面膜上。在这些实验中，FR系统只能识别3.34%的戴口罩的参与者(相比之下，其他评估的口罩的最低识别率为83.34%)。我们的实验演示可在以下网址找到：https://youtu.be/_TXkDO5z11w.



## **46. FIDO2 With Two Displays-Or How to Protect Security-Critical Web Transactions Against Malware Attacks**

带两个显示屏的FIDO2-或如何保护安全关键型Web交易免受恶意软件攻击 cs.CR

**SubmitDate**: 2022-07-06    [paper-pdf](http://arxiv.org/pdf/2206.13358v3)

**Authors**: Timon Hackenjos, Benedikt Wagner, Julian Herr, Jochen Rill, Marek Wehmer, Niklas Goerke, Ingmar Baumgart

**Abstracts**: With the rise of attacks on online accounts in the past years, more and more services offer two-factor authentication for their users. Having factors out of two of the three categories something you know, something you have and something you are should ensure that an attacker cannot compromise two of them at once. Thus, an adversary should not be able to maliciously interact with one's account. However, this is only true if one considers a weak adversary. In particular, since most current solutions only authenticate a session and not individual transactions, they are noneffective if one's device is infected with malware. For online banking, the banking industry has long since identified the need for authenticating transactions. However, specifications of such authentication schemes are not public and implementation details vary wildly from bank to bank with most still being unable to protect against malware. In this work, we present a generic approach to tackle the problem of malicious account takeovers, even in the presence of malware. To this end, we define a new paradigm to improve two-factor authentication that involves the concepts of one-out-of-two security and transaction authentication. Web authentication schemes following this paradigm can protect security-critical transactions against manipulation, even if one of the factors is completely compromised. Analyzing existing authentication schemes, we find that they do not realize one-out-of-two security. We give a blueprint of how to design secure web authentication schemes in general. Based on this blueprint we propose FIDO2 With Two Displays (FIDO2D), a new web authentication scheme based on the FIDO2 standard and prove its security using Tamarin. We hope that our work inspires a new wave of more secure web authentication schemes, which protect security-critical transactions even against attacks with malware.

摘要: 随着过去几年针对在线账户的攻击事件的增加，越来越多的服务为其用户提供双因素身份验证。拥有三个类别中的两个因素，你知道的，你拥有的和你是的，应该确保攻击者不能同时危害其中的两个。因此，对手不应该能够恶意地与自己的帐户交互。然而，只有当一个人考虑到一个弱小的对手时，这才是正确的。特别是，由于大多数当前的解决方案只对会话进行身份验证，而不是对单个事务进行身份验证，因此如果设备感染了恶意软件，这些解决方案就会无效。对于网上银行，银行业早就认识到了对交易进行身份验证的必要性。然而，此类身份验证方案的规范并未公开，各银行的实施细节也存在很大差异，大多数银行仍无法防范恶意软件。在这项工作中，我们提出了一种通用的方法来解决恶意帐户接管问题，即使在存在恶意软件的情况下也是如此。为此，我们定义了一个新的范例来改进双因素身份验证，它涉及二选一安全和事务身份验证的概念。遵循此范例的Web身份验证方案可以保护安全关键型交易免受操纵，即使其中一个因素完全受损。分析现有的认证方案，发现它们并没有实现二选一的安全性。我们给出了一个总体上如何设计安全的Web认证方案的蓝图。在此基础上，我们提出了一种新的基于FIDO2标准的网络认证方案FIDO2 with Two Display(FIDO2D)，并用Tamarin对其安全性进行了证明。我们希望我们的工作激发出新一波更安全的网络身份验证方案，这些方案甚至可以保护安全关键交易免受恶意软件的攻击。



## **47. Learning to Accelerate Approximate Methods for Solving Integer Programming via Early Fixing**

通过早期确定学习加速求解整数规划的近似方法 cs.DM

16 pages, 11 figures, 6 tables

**SubmitDate**: 2022-07-05    [paper-pdf](http://arxiv.org/pdf/2207.02087v1)

**Authors**: Longkang Li, Baoyuan Wu

**Abstracts**: Integer programming (IP) is an important and challenging problem. Approximate methods have shown promising performance on both effectiveness and efficiency for solving the IP problem. However, we observed that a large fraction of variables solved by some iterative approximate methods fluctuate around their final converged discrete states in very long iterations. Inspired by this observation, we aim to accelerate these approximate methods by early fixing these fluctuated variables to their converged states while not significantly harming the solution accuracy. To this end, we propose an early fixing framework along with the approximate method. We formulate the whole early fixing process as a Markov decision process, and train it using imitation learning. A policy network will evaluate the posterior probability of each free variable concerning its discrete candidate states in each block of iterations. Specifically, we adopt the powerful multi-headed attention mechanism in the policy network. Extensive experiments on our proposed early fixing framework are conducted to three different IP applications: constrained linear programming, MRF energy minimization and sparse adversarial attack. The former one is linear IP problem, while the latter two are quadratic IP problems. We extend the problem scale from regular size to significantly large size. The extensive experiments reveal the competitiveness of our early fixing framework: the runtime speeds up significantly, while the solution quality does not degrade much, even in some cases it is available to obtain better solutions. Our proposed early fixing framework can be regarded as an acceleration extension of ADMM methods for solving integer programming. The source codes are available at \url{https://github.com/SCLBD/Accelerated-Lpbox-ADMM}.

摘要: 整数规划(IP)是一个重要且具有挑战性的问题。近似方法在解决IP问题的有效性和效率方面都表现出了良好的性能。然而，我们观察到，一些迭代近似方法求解的大部分变量在很长的迭代中围绕其最终收敛的离散状态波动。受这一观察结果的启发，我们的目标是在不显著损害解精度的情况下，通过及早将这些波动变量固定到它们的收敛状态来加速这些近似方法。为此，我们提出了一个早期固定框架和近似方法。我们将整个早期定位过程描述为一个马尔可夫决策过程，并利用模仿学习对其进行训练。策略网络将评估每个自由变量在每个迭代块中关于其离散候选状态的后验概率。具体地说，我们在政策网络中采用了强大的多头注意机制。针对三种不同的IP应用：约束线性规划、马尔可夫随机场能量最小化和稀疏敌意攻击，对我们提出的早期修复框架进行了大量的实验。前一个是线性IP问题，后两个是二次IP问题。我们将问题的规模从常规规模扩展到显著的大型规模。广泛的实验表明了我们早期修复框架的竞争力：运行时间显著加快，而解决方案质量并没有太大下降，甚至在某些情况下可以获得更好的解决方案。我们提出的早期固定框架可以看作是求解整数规划的ADMM方法的加速扩展。源代码可在\url{https://github.com/SCLBD/Accelerated-Lpbox-ADMM}.上找到



## **48. Benign Adversarial Attack: Tricking Models for Goodness**

良性对抗性攻击：欺骗模型向善 cs.AI

ACM MM2022 Brave New Idea

**SubmitDate**: 2022-07-05    [paper-pdf](http://arxiv.org/pdf/2107.11986v2)

**Authors**: Jitao Sang, Xian Zhao, Jiaming Zhang, Zhiyu Lin

**Abstracts**: In spite of the successful application in many fields, machine learning models today suffer from notorious problems like vulnerability to adversarial examples. Beyond falling into the cat-and-mouse game between adversarial attack and defense, this paper provides alternative perspective to consider adversarial example and explore whether we can exploit it in benign applications. We first attribute adversarial example to the human-model disparity on employing non-semantic features. While largely ignored in classical machine learning mechanisms, non-semantic feature enjoys three interesting characteristics as (1) exclusive to model, (2) critical to affect inference, and (3) utilizable as features. Inspired by this, we present brave new idea of benign adversarial attack to exploit adversarial examples for goodness in three directions: (1) adversarial Turing test, (2) rejecting malicious model application, and (3) adversarial data augmentation. Each direction is positioned with motivation elaboration, justification analysis and prototype applications to showcase its potential.

摘要: 尽管机器学习模型在许多领域都得到了成功的应用，但它仍然存在一些臭名昭著的问题，比如易受敌意例子的攻击。除了陷入对抗性攻击和防御之间的猫鼠游戏之外，本文还提供了另一种视角来考虑对抗性例子，并探索是否可以在良性应用中利用它。我们首先将对抗性例子归因于人类模型在使用非语义特征上的差异。在经典的机器学习机制中，非语义特征被忽略了，但它具有三个有趣的特征：(1)模型独有的，(2)对影响推理的关键的，(3)可用作特征的。受此启发，我们提出了良性对抗性攻击的大胆新思想，从三个方向挖掘对抗性实例：(1)对抗性图灵测试，(2)拒绝恶意模型应用，(3)对抗性数据扩充。每个方向都定位于动机阐述、理由分析和原型应用，以展示其潜力。



## **49. Formalizing and Estimating Distribution Inference Risks**

分布推断风险的形式化和估计 cs.LG

Update: Accepted at PETS 2022

**SubmitDate**: 2022-07-05    [paper-pdf](http://arxiv.org/pdf/2109.06024v6)

**Authors**: Anshuman Suri, David Evans

**Abstracts**: Distribution inference, sometimes called property inference, infers statistical properties about a training set from access to a model trained on that data. Distribution inference attacks can pose serious risks when models are trained on private data, but are difficult to distinguish from the intrinsic purpose of statistical machine learning -- namely, to produce models that capture statistical properties about a distribution. Motivated by Yeom et al.'s membership inference framework, we propose a formal definition of distribution inference attacks that is general enough to describe a broad class of attacks distinguishing between possible training distributions. We show how our definition captures previous ratio-based property inference attacks as well as new kinds of attack including revealing the average node degree or clustering coefficient of a training graph. To understand distribution inference risks, we introduce a metric that quantifies observed leakage by relating it to the leakage that would occur if samples from the training distribution were provided directly to the adversary. We report on a series of experiments across a range of different distributions using both novel black-box attacks and improved versions of the state-of-the-art white-box attacks. Our results show that inexpensive attacks are often as effective as expensive meta-classifier attacks, and that there are surprising asymmetries in the effectiveness of attacks. Code is available at https://github.com/iamgroot42/FormEstDistRisks

摘要: 分布推理，有时被称为属性推理，从对基于该数据训练的模型的访问中推断出关于训练集的统计属性。当模型根据私有数据进行训练时，分布推断攻击可能会带来严重的风险，但很难与统计机器学习的内在目的区分开来--即产生捕获有关分布的统计属性的模型。受Yeom等人的成员关系推理框架的启发，我们提出了分布推理攻击的形式化定义，该定义足够通用，以描述区分可能的训练分布的广泛的攻击类别。我们展示了我们的定义如何捕获以前的基于比率的属性推理攻击，以及新的攻击类型，包括揭示训练图的平均结点度或聚类系数。为了了解分布推断风险，我们引入了一个度量，该度量通过将观察到的泄漏与训练分布的样本直接提供给对手时将发生的泄漏联系起来，来量化观察到的泄漏。我们报告了使用新的黑盒攻击和最先进的白盒攻击的改进版本在一系列不同的发行版上进行的一系列实验。我们的结果表明，廉价的攻击通常与昂贵的元分类器攻击一样有效，并且攻击的有效性存在惊人的不对称性。代码可在https://github.com/iamgroot42/FormEstDistRisks上找到



## **50. Query-Efficient Adversarial Attack Based on Latin Hypercube Sampling**

基于拉丁超立方体采样的查询高效对抗攻击 cs.CV

**SubmitDate**: 2022-07-05    [paper-pdf](http://arxiv.org/pdf/2207.02391v1)

**Authors**: Dan Wang, Jiayu Lin, Yuan-Gen Wang

**Abstracts**: In order to be applicable in real-world scenario, Boundary Attacks (BAs) were proposed and ensured one hundred percent attack success rate with only decision information. However, existing BA methods craft adversarial examples by leveraging a simple random sampling (SRS) to estimate the gradient, consuming a large number of model queries. To overcome the drawback of SRS, this paper proposes a Latin Hypercube Sampling based Boundary Attack (LHS-BA) to save query budget. Compared with SRS, LHS has better uniformity under the same limited number of random samples. Therefore, the average on these random samples is closer to the true gradient than that estimated by SRS. Various experiments are conducted on benchmark datasets including MNIST, CIFAR, and ImageNet-1K. Experimental results demonstrate the superiority of the proposed LHS-BA over the state-of-the-art BA methods in terms of query efficiency. The source codes are publicly available at https://github.com/GZHU-DVL/LHS-BA.

摘要: 边界攻击(BAS)是为了更好地应用于实际场景而提出的，只需要决策信息就能保证100%的攻击成功率。然而，现有的BA方法通过利用简单随机抽样(SRS)来估计梯度来制作对抗性例子，消耗了大量的模型查询。针对SRS算法的不足，提出了一种基于拉丁超立方体采样的边界攻击算法(LHS-BA)，以节省查询开销。与SRS相比，在相同的有限随机样本数下，LHS具有更好的一致性。因此，这些随机样本的平均值比SRS估计的梯度更接近真实梯度。在包括MNIST、CIFAR和ImageNet-1K在内的基准数据集上进行了各种实验。实验结果表明，LHS-BA在查询效率方面优于现有的BA方法。源代码可在https://github.com/GZHU-DVL/LHS-BA.上公开获取



