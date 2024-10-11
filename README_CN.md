# Latest Adversarial Attack Papers
**update at 2024-10-11 09:42:03**

翻译来自 https://cloud.tencent.com/document/product/551/15619

## **1. Protecting Your LLMs with Information Bottleneck**

通过信息瓶颈保护您的LLC cs.CL

Accepted by Neural Information Processing Systems (NeurIPS 2024)

**SubmitDate**: 2024-10-10    [abs](http://arxiv.org/abs/2404.13968v3) [paper-pdf](http://arxiv.org/pdf/2404.13968v3)

**Authors**: Zichuan Liu, Zefan Wang, Linjie Xu, Jinyu Wang, Lei Song, Tianchun Wang, Chunlin Chen, Wei Cheng, Jiang Bian

**Abstract**: The advent of large language models (LLMs) has revolutionized the field of natural language processing, yet they might be attacked to produce harmful content. Despite efforts to ethically align LLMs, these are often fragile and can be circumvented by jailbreaking attacks through optimized or manual adversarial prompts. To address this, we introduce the Information Bottleneck Protector (IBProtector), a defense mechanism grounded in the information bottleneck principle, and we modify the objective to avoid trivial solutions. The IBProtector selectively compresses and perturbs prompts, facilitated by a lightweight and trainable extractor, preserving only essential information for the target LLMs to respond with the expected answer. Moreover, we further consider a situation where the gradient is not visible to be compatible with any LLM. Our empirical evaluations show that IBProtector outperforms current defense methods in mitigating jailbreak attempts, without overly affecting response quality or inference speed. Its effectiveness and adaptability across various attack methods and target LLMs underscore the potential of IBProtector as a novel, transferable defense that bolsters the security of LLMs without requiring modifications to the underlying models.

摘要: 大型语言模型的出现给自然语言处理领域带来了革命性的变化，但它们可能会受到攻击，产生有害的内容。尽管努力在道德上调整LLM，但这些往往是脆弱的，可以通过优化或手动对抗性提示通过越狱攻击来绕过。为了解决这个问题，我们引入了信息瓶颈保护器(IBProtector)，这是一种基于信息瓶颈原理的防御机制，我们修改了目标以避免琐碎的解决方案。IBProtector有选择地压缩和干扰提示，由一个轻量级和可训练的提取程序促进，只保留目标LLMS的基本信息，以响应预期的答案。此外，我们还进一步考虑了梯度不可见的情况，以与任何LLM相容。我们的经验评估表明，在不过度影响响应质量或推理速度的情况下，IBProtector在缓解越狱企图方面优于现有的防御方法。它对各种攻击方法和目标LLM的有效性和适应性突显了IBProtector作为一种新型、可转移的防御系统的潜力，无需修改底层模型即可增强LLM的安全性。



## **2. MGMD-GAN: Generalization Improvement of Generative Adversarial Networks with Multiple Generator Multiple Discriminator Framework Against Membership Inference Attacks**

MGMD-GAN：具有多生成器多鉴别器框架的生成性对抗网络抗成员推断攻击的推广改进 cs.LG

**SubmitDate**: 2024-10-10    [abs](http://arxiv.org/abs/2410.07803v1) [paper-pdf](http://arxiv.org/pdf/2410.07803v1)

**Authors**: Nirob Arefin

**Abstract**: Generative Adversarial Networks (GAN) are among the widely used Generative models in various applications. However, the original GAN architecture may memorize the distribution of the training data and, therefore, poses a threat to Membership Inference Attacks. In this work, we propose a new GAN framework that consists of Multiple Generators and Multiple Discriminators (MGMD-GAN). Disjoint partitions of the training data are used to train this model and it learns the mixture distribution of all the training data partitions. In this way, our proposed model reduces the generalization gap which makes our MGMD-GAN less vulnerable to Membership Inference Attacks. We provide an experimental analysis of our model and also a comparison with other GAN frameworks.

摘要: 生成对抗网络（GAN）是各种应用中广泛使用的生成模型之一。然而，原始的GAN架构可能会记住训练数据的分布，因此对成员资格推理攻击构成威胁。在这项工作中，我们提出了一个新的GAN框架，该框架由多生成器和多鉴别器（MGMD-GAN）组成。训练数据的不相交分区用于训练该模型，并学习所有训练数据分区的混合分布。通过这种方式，我们提出的模型缩小了概括差距，这使得我们的MGMD-GAN更不容易受到成员推断攻击。我们对我们的模型进行了实验分析，并与其他GAN框架进行了比较。



## **3. Invisibility Cloak: Disappearance under Human Pose Estimation via Backdoor Attacks**

隐形衣：通过后门攻击在人类姿势估计下失踪 cs.CR

**SubmitDate**: 2024-10-10    [abs](http://arxiv.org/abs/2410.07670v1) [paper-pdf](http://arxiv.org/pdf/2410.07670v1)

**Authors**: Minxing Zhang, Michael Backes, Xiao Zhang

**Abstract**: Human Pose Estimation (HPE) has been widely applied in autonomous systems such as self-driving cars. However, the potential risks of HPE to adversarial attacks have not received comparable attention with image classification or segmentation tasks. Existing works on HPE robustness focus on misleading an HPE system to provide wrong predictions that still indicate some human poses. In this paper, we study the vulnerability of HPE systems to disappearance attacks, where the attacker aims to subtly alter the HPE training process via backdoor techniques so that any input image with some specific trigger will not be recognized as involving any human pose. As humans are typically at the center of HPE systems, such attacks can induce severe security hazards, e.g., pedestrians' lives will be threatened if a self-driving car incorrectly understands the front scene due to disappearance attacks.   To achieve the adversarial goal of disappearance, we propose IntC, a general framework to craft Invisibility Cloak in the HPE domain. The core of our work lies in the design of target HPE labels that do not represent any human pose. In particular, we propose three specific backdoor attacks based on our IntC framework with different label designs. IntC-S and IntC-E, respectively designed for regression- and heatmap-based HPE techniques, concentrate the keypoints of triggered images in a tiny, imperceptible region. Further, to improve the attack's stealthiness, IntC-L designs the target poisons to capture the label outputs of typical landscape images without a human involved, achieving disappearance and reducing detectability simultaneously. Extensive experiments demonstrate the effectiveness and generalizability of our IntC methods in achieving the disappearance goal. By revealing the vulnerability of HPE to disappearance and backdoor attacks, we hope our work can raise awareness of the potential risks ...

摘要: 人体姿态估计在自动驾驶汽车等自动驾驶系统中得到了广泛的应用。然而，在图像分类或分割任务中，HPE对对抗性攻击的潜在风险并没有得到类似的关注。现有的关于HPE稳健性的工作集中于误导HPE系统以提供仍然指示某些人类姿势的错误预测。在本文中，我们研究了HPE系统对失踪攻击的脆弱性，其中攻击者的目标是通过后门技术巧妙地改变HPE的训练过程，以便任何具有特定触发器的输入图像都不会被识别为涉及任何人体姿势。由于人类通常是HPE系统的中心，这样的攻击会导致严重的安全隐患，例如，如果自动驾驶汽车因失踪攻击而无法正确理解前面的场景，行人的生命将受到威胁。为了达到对抗消失的目标，我们提出了INTC，这是一个在HPE领域制作隐形斗篷的通用框架。我们工作的核心在于设计不代表任何人体姿势的目标HPE标签。特别是，我们基于我们的INTC框架提出了三种不同标签设计的具体后门攻击。INTC-S和INTC-E分别为基于回归和热图的HPE技术设计，将触发图像的关键点集中在一个微小的、不可感知的区域。此外，为了提高攻击的隐蔽性，INTC-L设计了目标毒药，在没有人参与的情况下捕获典型景观图像的标签输出，同时实现了消失和降低可检测性。大量的实验证明了我们的INTC方法在实现消失目标方面的有效性和普适性。通过揭示HPE对失踪和后门攻击的脆弱性，我们希望我们的工作可以提高人们对潜在风险的认识……



## **4. Universally Optimal Watermarking Schemes for LLMs: from Theory to Practice**

LLM的普遍最优水印方案：从理论到实践 cs.CR

**SubmitDate**: 2024-10-10    [abs](http://arxiv.org/abs/2410.02890v2) [paper-pdf](http://arxiv.org/pdf/2410.02890v2)

**Authors**: Haiyun He, Yepeng Liu, Ziqiao Wang, Yongyi Mao, Yuheng Bu

**Abstract**: Large Language Models (LLMs) boosts human efficiency but also poses misuse risks, with watermarking serving as a reliable method to differentiate AI-generated content from human-created text. In this work, we propose a novel theoretical framework for watermarking LLMs. Particularly, we jointly optimize both the watermarking scheme and detector to maximize detection performance, while controlling the worst-case Type-I error and distortion in the watermarked text. Within our framework, we characterize the universally minimum Type-II error, showing a fundamental trade-off between detection performance and distortion. More importantly, we identify the optimal type of detectors and watermarking schemes. Building upon our theoretical analysis, we introduce a practical, model-agnostic and computationally efficient token-level watermarking algorithm that invokes a surrogate model and the Gumbel-max trick. Empirical results on Llama-13B and Mistral-8$\times$7B demonstrate the effectiveness of our method. Furthermore, we also explore how robustness can be integrated into our theoretical framework, which provides a foundation for designing future watermarking systems with improved resilience to adversarial attacks.

摘要: 大语言模型(LLM)提高了人类的效率，但也带来了滥用风险，水印是区分人工智能生成的内容和人类创建的文本的可靠方法。在这项工作中，我们提出了一种新的水印LLMS的理论框架。特别是，我们联合优化了水印方案和检测器以最大化检测性能，同时控制了最坏情况下的I类错误和水印文本中的失真。在我们的框架内，我们描述了普遍最小的第二类错误，显示了检测性能和失真之间的基本权衡。更重要的是，我们确定了检测器和水印方案的最佳类型。在理论分析的基础上，我们介绍了一种实用的、与模型无关的、计算高效的令牌级水印算法，该算法调用了代理模型和Gumbel-Max技巧。对Llama-13B和Mistral-8$乘以$70B的实验结果证明了该方法的有效性。此外，我们还探索了如何将稳健性融入到我们的理论框架中，这为设计未来具有更好的抗攻击能力的水印系统提供了基础。



## **5. Prompt-Agnostic Adversarial Perturbation for Customized Diffusion Models**

定制扩散模型的预算不可知对抗扰动 cs.CV

Accepted by NIPS 2024

**SubmitDate**: 2024-10-10    [abs](http://arxiv.org/abs/2408.10571v4) [paper-pdf](http://arxiv.org/pdf/2408.10571v4)

**Authors**: Cong Wan, Yuhang He, Xiang Song, Yihong Gong

**Abstract**: Diffusion models have revolutionized customized text-to-image generation, allowing for efficient synthesis of photos from personal data with textual descriptions. However, these advancements bring forth risks including privacy breaches and unauthorized replication of artworks. Previous researches primarily center around using prompt-specific methods to generate adversarial examples to protect personal images, yet the effectiveness of existing methods is hindered by constrained adaptability to different prompts. In this paper, we introduce a Prompt-Agnostic Adversarial Perturbation (PAP) method for customized diffusion models. PAP first models the prompt distribution using a Laplace Approximation, and then produces prompt-agnostic perturbations by maximizing a disturbance expectation based on the modeled distribution. This approach effectively tackles the prompt-agnostic attacks, leading to improved defense stability. Extensive experiments in face privacy and artistic style protection, demonstrate the superior generalization of PAP in comparison to existing techniques. Our project page is available at https://github.com/vancyland/Prompt-Agnostic-Adversarial-Perturbation-for-Customized-Diffusion-Models.github.io.

摘要: 扩散模型彻底改变了定制的文本到图像的生成，允许从具有文本描述的个人数据高效地合成照片。然而，这些进步带来了包括侵犯隐私和未经授权复制艺术品在内的风险。以往的研究主要集中在使用特定于提示的方法来生成对抗性的例子来保护个人形象，然而现有方法的有效性受到对不同提示的限制适应性的阻碍。在这篇文章中，我们介绍了定制扩散模型的即时不可知对抗扰动(PAP)方法。PAP首先使用拉普拉斯近似对瞬发分布进行建模，然后基于建模的分布最大化扰动期望来产生与瞬发无关的扰动。这种方法有效地解决了即时不可知攻击，从而提高了防御稳定性。在人脸隐私和艺术风格保护方面的广泛实验表明，与现有技术相比，PAP具有更好的泛化能力。我们的项目页面可在https://github.com/vancyland/Prompt-Agnostic-Adversarial-Perturbation-for-Customized-Diffusion-Models.github.io.上查看



## **6. MorCode: Face Morphing Attack Generation using Generative Codebooks**

MorCode：使用生成代码簿生成面部变形攻击 cs.CV

**SubmitDate**: 2024-10-10    [abs](http://arxiv.org/abs/2410.07625v1) [paper-pdf](http://arxiv.org/pdf/2410.07625v1)

**Authors**: Aravinda Reddy PN, Raghavendra Ramachandra, Sushma Venkatesh, Krothapalli Sreenivasa Rao, Pabitra Mitra, Rakesh Krishna

**Abstract**: Face recognition systems (FRS) can be compromised by face morphing attacks, which blend textural and geometric information from multiple facial images. The rapid evolution of generative AI, especially Generative Adversarial Networks (GAN) or Diffusion models, where encoded images are interpolated to generate high-quality face morphing images. In this work, we present a novel method for the automatic face morphing generation method \textit{MorCode}, which leverages a contemporary encoder-decoder architecture conditioned on codebook learning to generate high-quality morphing images. Extensive experiments were performed on the newly constructed morphing dataset using five state-of-the-art morphing generation techniques using both digital and print-scan data. The attack potential of the proposed morphing generation technique, \textit{MorCode}, was benchmarked using three different face recognition systems. The obtained results indicate the highest attack potential of the proposed \textit{MorCode} when compared with five state-of-the-art morphing generation methods on both digital and print scan data.

摘要: 人脸识别系统(FRS)可能会受到人脸变形攻击的攻击，这种攻击融合了来自多个面部图像的纹理和几何信息。生成性人工智能的快速发展，特别是生成性对抗网络(GAN)或扩散模型，其中编码图像被内插以生成高质量的人脸变形图像。在这项工作中，我们提出了一种新的方法来自动生成人脸变形图像，该方法利用基于码本学习的当代编解码器体系结构来生成高质量的变形图像。在新构建的变形数据集上进行了广泛的实验，使用五种最先进的变形生成技术，使用数字和打印扫描数据。使用三种不同的人脸识别系统对所提出的变形生成技术的攻击潜力进行了基准测试。结果表明，与五种最先进的数字和印刷扫描数据变形生成方法相比，所提出的文本{MorCode}的攻击潜力最高。



## **7. An undetectable watermark for generative image models**

生成式图像模型的不可检测水印 cs.CR

**SubmitDate**: 2024-10-09    [abs](http://arxiv.org/abs/2410.07369v1) [paper-pdf](http://arxiv.org/pdf/2410.07369v1)

**Authors**: Sam Gunn, Xuandong Zhao, Dawn Song

**Abstract**: We present the first undetectable watermarking scheme for generative image models. Undetectability ensures that no efficient adversary can distinguish between watermarked and un-watermarked images, even after making many adaptive queries. In particular, an undetectable watermark does not degrade image quality under any efficiently computable metric. Our scheme works by selecting the initial latents of a diffusion model using a pseudorandom error-correcting code (Christ and Gunn, 2024), a strategy which guarantees undetectability and robustness. We experimentally demonstrate that our watermarks are quality-preserving and robust using Stable Diffusion 2.1. Our experiments verify that, in contrast to every prior scheme we tested, our watermark does not degrade image quality. Our experiments also demonstrate robustness: existing watermark removal attacks fail to remove our watermark from images without significantly degrading the quality of the images. Finally, we find that we can robustly encode 512 bits in our watermark, and up to 2500 bits when the images are not subjected to watermark removal attacks. Our code is available at https://github.com/XuandongZhao/PRC-Watermark.

摘要: 我们提出了第一个不可检测的生成图像模型的水印方案。不可检测性确保了即使在进行了许多自适应查询之后，有效的攻击者也无法区分加水印和未加水印的图像。特别是，不可检测的水印在任何有效计算的度量下都不会降低图像质量。我们的方案通过使用伪随机纠错码(Christian和Gunn，2024)来选择扩散模型的初始潜伏期，这是一种保证不可检测性和稳健性的策略。实验证明，利用稳定扩散2.1算法，水印具有较好的保质性和稳健性。我们的实验证明，与我们测试的每个方案相比，我们的水印不会降低图像质量。我们的实验也证明了我们的稳健性：现有的水印去除攻击不能在不显著降低图像质量的情况下去除图像中的水印。最后，我们发现我们的水印可以稳健地编码512比特，当图像没有受到水印去除攻击时，可以编码高达2500比特。我们的代码可以在https://github.com/XuandongZhao/PRC-Watermark.上找到



## **8. JPEG Inspired Deep Learning**

JPEG启发深度学习 cs.CV

**SubmitDate**: 2024-10-09    [abs](http://arxiv.org/abs/2410.07081v1) [paper-pdf](http://arxiv.org/pdf/2410.07081v1)

**Authors**: Ahmed H. Salamah, Kaixiang Zheng, Yiwen Liu, En-Hui Yang

**Abstract**: Although it is traditionally believed that lossy image compression, such as JPEG compression, has a negative impact on the performance of deep neural networks (DNNs), it is shown by recent works that well-crafted JPEG compression can actually improve the performance of deep learning (DL). Inspired by this, we propose JPEG-DL, a novel DL framework that prepends any underlying DNN architecture with a trainable JPEG compression layer. To make the quantization operation in JPEG compression trainable, a new differentiable soft quantizer is employed at the JPEG layer, and then the quantization operation and underlying DNN are jointly trained. Extensive experiments show that in comparison with the standard DL, JPEG-DL delivers significant accuracy improvements across various datasets and model architectures while enhancing robustness against adversarial attacks. Particularly, on some fine-grained image classification datasets, JPEG-DL can increase prediction accuracy by as much as 20.9%. Our code is available on https://github.com/JpegInspiredDl/JPEG-Inspired-DL.git.

摘要: 虽然传统上认为有损图像压缩，如JPEG压缩，会对深度神经网络(DNN)的性能产生负面影响，但最近的研究表明，精心设计的JPEG压缩实际上可以提高深度学习(DL)的性能。受此启发，我们提出了JPEG-DL，这是一种新颖的DL框架，它在任何底层的DNN体系结构中都预先加入了一个可训练的JPEG压缩层。为了使JPEG压缩中的量化操作可训练，在JPEG层使用了一种新的可微软量化器，然后将量化操作和底层的DNN进行联合训练。大量的实验表明，与标准的DL相比，JPEG-DL在不同的数据集和模型体系结构上提供了显著的准确性改进，同时增强了对对手攻击的健壮性。特别是，在一些细粒度的图像分类数据集上，JPEG-DL可以将预测精度提高20.9%。我们的代码可以在https://github.com/JpegInspiredDl/JPEG-Inspired-DL.git.上找到



## **9. SAGMAN: Stability Analysis of Graph Neural Networks on the Manifolds**

SAGMAN：图神经网络的稳定性分析 cs.LG

**SubmitDate**: 2024-10-09    [abs](http://arxiv.org/abs/2402.08653v4) [paper-pdf](http://arxiv.org/pdf/2402.08653v4)

**Authors**: Wuxinlin Cheng, Chenhui Deng, Ali Aghdaei, Zhiru Zhang, Zhuo Feng

**Abstract**: Modern graph neural networks (GNNs) can be sensitive to changes in the input graph structure and node features, potentially resulting in unpredictable behavior and degraded performance. In this work, we introduce a spectral framework known as SAGMAN for examining the stability of GNNs. This framework assesses the distance distortions that arise from the nonlinear mappings of GNNs between the input and output manifolds: when two nearby nodes on the input manifold are mapped (through a GNN model) to two distant ones on the output manifold, it implies a large distance distortion and thus a poor GNN stability. We propose a distance-preserving graph dimension reduction (GDR) approach that utilizes spectral graph embedding and probabilistic graphical models (PGMs) to create low-dimensional input/output graph-based manifolds for meaningful stability analysis. Our empirical evaluations show that SAGMAN effectively assesses the stability of each node when subjected to various edge or feature perturbations, offering a scalable approach for evaluating the stability of GNNs, extending to applications within recommendation systems. Furthermore, we illustrate its utility in downstream tasks, notably in enhancing GNN stability and facilitating adversarial targeted attacks.

摘要: 现代图神经网络(GNN)对输入图结构和节点特征的变化很敏感，可能导致不可预测的行为和性能下降。在这项工作中，我们引入了一个称为Sagman的光谱框架来检查GNN的稳定性。该框架评估了输入和输出流形之间GNN之间的非线性映射引起的距离失真：当输入流形上的两个邻近节点(通过GNN模型)映射到输出流形上的两个相距较远的节点时，这意味着较大的距离失真，因此GNN稳定性较差。我们提出了一种保持距离的图降维方法(GDR)，该方法利用谱图嵌入和概率图模型(PGMS)来创建基于低维输入/输出图的流形，用于有意义的稳定性分析。实验结果表明，Sagman算法能够有效地评估每个节点在受到各种边缘或特征扰动时的稳定性，为评估GNN的稳定性提供了一种可扩展的方法，并扩展到推荐系统中的应用。此外，我们还说明了它在下游任务中的效用，特别是在增强GNN稳定性和促进对抗性定向攻击方面。



## **10. Defensive Unlearning with Adversarial Training for Robust Concept Erasure in Diffusion Models**

扩散模型中鲁棒概念擦除的对抗训练防御性取消学习 cs.CV

Accepted by NeurIPS'24. Codes are available at  https://github.com/OPTML-Group/AdvUnlearn

**SubmitDate**: 2024-10-09    [abs](http://arxiv.org/abs/2405.15234v3) [paper-pdf](http://arxiv.org/pdf/2405.15234v3)

**Authors**: Yimeng Zhang, Xin Chen, Jinghan Jia, Yihua Zhang, Chongyu Fan, Jiancheng Liu, Mingyi Hong, Ke Ding, Sijia Liu

**Abstract**: Diffusion models (DMs) have achieved remarkable success in text-to-image generation, but they also pose safety risks, such as the potential generation of harmful content and copyright violations. The techniques of machine unlearning, also known as concept erasing, have been developed to address these risks. However, these techniques remain vulnerable to adversarial prompt attacks, which can prompt DMs post-unlearning to regenerate undesired images containing concepts (such as nudity) meant to be erased. This work aims to enhance the robustness of concept erasing by integrating the principle of adversarial training (AT) into machine unlearning, resulting in the robust unlearning framework referred to as AdvUnlearn. However, achieving this effectively and efficiently is highly nontrivial. First, we find that a straightforward implementation of AT compromises DMs' image generation quality post-unlearning. To address this, we develop a utility-retaining regularization on an additional retain set, optimizing the trade-off between concept erasure robustness and model utility in AdvUnlearn. Moreover, we identify the text encoder as a more suitable module for robustification compared to UNet, ensuring unlearning effectiveness. And the acquired text encoder can serve as a plug-and-play robust unlearner for various DM types. Empirically, we perform extensive experiments to demonstrate the robustness advantage of AdvUnlearn across various DM unlearning scenarios, including the erasure of nudity, objects, and style concepts. In addition to robustness, AdvUnlearn also achieves a balanced tradeoff with model utility. To our knowledge, this is the first work to systematically explore robust DM unlearning through AT, setting it apart from existing methods that overlook robustness in concept erasing. Codes are available at: https://github.com/OPTML-Group/AdvUnlearn

摘要: 扩散模型(DM)在文本到图像的生成方面取得了显著的成功，但它们也带来了安全风险，如可能生成有害内容和侵犯版权。机器遗忘技术，也被称为概念擦除，就是为了解决这些风险而开发的。然而，这些技术仍然容易受到敌意的即时攻击，这可能会促使忘记后的DM重新生成包含要擦除的概念(如裸体)的不需要的图像。这项工作旨在通过将对抗性训练(AT)的原理整合到机器遗忘中来增强概念删除的稳健性，从而产生健壮的遗忘框架，称为AdvUnLearning。然而，有效和高效地实现这一点并不是微不足道的。首先，我们发现AT的直接实现损害了DM在遗忘后的图像生成质量。为了解决这个问题，我们在一个额外的保留集上开发了效用保留正则化，优化了AdvUnLearning中概念删除健壮性和模型实用之间的权衡。此外，我们认为文本编码器是一个更适合于粗暴的模块，与联合国教科文组织相比，确保了遗忘的有效性。并且所获得的文本编码器可以作为各种DM类型的即插即用鲁棒去学习器。经验性地，我们进行了大量的实验来展示AdvUnLearning在各种DM遗忘场景中的健壮性优势，包括对裸体、物体和风格概念的删除。除了健壮性之外，AdvUnLearning还实现了与模型实用程序之间的平衡。据我们所知，这是第一个通过AT系统地探索稳健的DM遗忘的工作，区别于现有的忽略概念删除中的稳健性的方法。代码可在以下网址获得：https://github.com/OPTML-Group/AdvUnlearn



## **11. The Vital Role of Gradient Clipping in Byzantine-Resilient Distributed Learning**

梯度剪辑在拜占庭弹性分布式学习中的重要作用 cs.LG

**SubmitDate**: 2024-10-09    [abs](http://arxiv.org/abs/2405.14432v4) [paper-pdf](http://arxiv.org/pdf/2405.14432v4)

**Authors**: Youssef Allouah, Rachid Guerraoui, Nirupam Gupta, Ahmed Jellouli, Geovani Rizk, John Stephan

**Abstract**: Byzantine-resilient distributed machine learning seeks to achieve robust learning performance in the presence of misbehaving or adversarial workers. While state-of-the-art (SOTA) robust distributed gradient descent (Robust-DGD) methods were proven theoretically optimal, their empirical success has often relied on pre-aggregation gradient clipping. However, the currently considered static clipping strategy exhibits mixed results: improving robustness against some attacks while being ineffective or detrimental against others. We address this gap by proposing a principled adaptive clipping strategy, termed Adaptive Robust Clipping (ARC). We show that ARC consistently enhances the empirical robustness of SOTA Robust-DGD methods, while preserving the theoretical robustness guarantees. Our analysis shows that ARC provably improves the asymptotic convergence guarantee of Robust-DGD in the case when the model is well-initialized. We validate this theoretical insight through an exhaustive set of experiments on benchmark image classification tasks. We observe that the improvement induced by ARC is more pronounced in highly heterogeneous and adversarial settings.

摘要: 拜占庭-弹性分布式机器学习寻求在存在行为不端或敌对的工作人员的情况下实现稳健的学习性能。虽然最先进的(SOTA)稳健分布梯度下降(Robust-DGD)方法在理论上被证明是最优的，但它们的经验成功通常依赖于预聚聚梯度裁剪。然而，目前考虑的静态裁剪策略呈现出好坏参半的结果：提高了对某些攻击的健壮性，而对另一些攻击无效或有害。我们提出了一种原则性的自适应剪裁策略，称为自适应稳健剪裁(ARC)，以解决这一差距。我们证明了ARC在保持理论稳健性保证的同时，一致地增强了SOTA Robust-DGD方法的经验稳健性。我们的分析表明，在模型良好初始化的情况下，ARC明显改善了Robust-DGD的渐近收敛保证。我们通过一组详尽的基准图像分类任务的实验来验证这一理论见解。我们观察到，在高度异质性和对抗性的环境中，ARC带来的改善更加显著。



## **12. Average Certified Radius is a Poor Metric for Randomized Smoothing**

平均认证半径对于随机平滑来说是一个较差的指标 cs.LG

**SubmitDate**: 2024-10-09    [abs](http://arxiv.org/abs/2410.06895v1) [paper-pdf](http://arxiv.org/pdf/2410.06895v1)

**Authors**: Chenhao Sun, Yuhao Mao, Mark Niklas Müller, Martin Vechev

**Abstract**: Randomized smoothing is a popular approach for providing certified robustness guarantees against adversarial attacks, and has become a very active area of research. Over the past years, the average certified radius (ACR) has emerged as the single most important metric for comparing methods and tracking progress in the field. However, in this work, we show that ACR is an exceptionally poor metric for evaluating robustness guarantees provided by randomized smoothing. We theoretically show not only that a trivial classifier can have arbitrarily large ACR, but also that ACR is much more sensitive to improvements on easy samples than on hard ones. Empirically, we confirm that existing training strategies that improve ACR reduce the model's robustness on hard samples. Further, we show that by focusing on easy samples, we can effectively replicate the increase in ACR. We develop strategies, including explicitly discarding hard samples, reweighing the dataset with certified radius, and extreme optimization for easy samples, to achieve state-of-the-art ACR, although these strategies ignore robustness for the general data distribution. Overall, our results suggest that ACR has introduced a strong undesired bias to the field, and better metrics are required to holistically evaluate randomized smoothing.

摘要: 随机化平滑是一种流行的方法，可以提供对敌意攻击的健壮性保证，并且已经成为一个非常活跃的研究领域。在过去的几年里，平均认证半径(ACR)已经成为比较方法和跟踪该领域进展的最重要的单一指标。然而，在这项工作中，我们表明ACR是评估随机平滑所提供的稳健性保证的一个特别差的度量。我们从理论上证明，一个平凡的分类器不仅可以有任意大的ACR，而且ACR对简单样本的改进比对困难样本的改进敏感得多。经验证明，现有的提高ACR的训练策略降低了模型在硬样本上的稳健性。此外，我们还表明，通过关注简单的样本，我们可以有效地复制ACR的增加。我们开发了一些策略，包括显式丢弃硬样本、用认证半径重新加权数据集以及对简单样本进行极端优化，以实现最先进的ACR，尽管这些策略忽略了对一般数据分布的稳健性。总体而言，我们的结果表明，ACR已经向该领域引入了强烈的不受欢迎的偏差，需要更好的度量来全面评估随机平滑。



## **13. Secure Video Quality Assessment Resisting Adversarial Attacks**

安全的视频质量评估抵抗对抗攻击 cs.CV

**SubmitDate**: 2024-10-09    [abs](http://arxiv.org/abs/2410.06866v1) [paper-pdf](http://arxiv.org/pdf/2410.06866v1)

**Authors**: Ao-Xiang Zhang, Yu Ran, Weixuan Tang, Yuan-Gen Wang, Qingxiao Guan, Chunsheng Yang

**Abstract**: The exponential surge in video traffic has intensified the imperative for Video Quality Assessment (VQA). Leveraging cutting-edge architectures, current VQA models have achieved human-comparable accuracy. However, recent studies have revealed the vulnerability of existing VQA models against adversarial attacks. To establish a reliable and practical assessment system, a secure VQA model capable of resisting such malicious attacks is urgently demanded. Unfortunately, no attempt has been made to explore this issue. This paper first attempts to investigate general adversarial defense principles, aiming at endowing existing VQA models with security. Specifically, we first introduce random spatial grid sampling on the video frame for intra-frame defense. Then, we design pixel-wise randomization through a guardian map, globally neutralizing adversarial perturbations. Meanwhile, we extract temporal information from the video sequence as compensation for inter-frame defense. Building upon these principles, we present a novel VQA framework from the security-oriented perspective, termed SecureVQA. Extensive experiments indicate that SecureVQA sets a new benchmark in security while achieving competitive VQA performance compared with state-of-the-art models. Ablation studies delve deeper into analyzing the principles of SecureVQA, demonstrating their generalization and contributions to the security of leading VQA models.

摘要: 视频流量的指数级增长加剧了视频质量评估(VQA)的紧迫性。利用尖端架构，当前的VQA模型实现了与人类相当的准确性。然而，最近的研究揭示了现有的VQA模型在对抗攻击时的脆弱性。为了建立一个可靠、实用的评估体系，迫切需要一个能够抵抗此类恶意攻击的安全的VQA模型。不幸的是，没有人试图探索这个问题。本文首先试图研究一般的对抗性防御原理，旨在赋予现有的VQA模型安全性。具体地说，我们首先在视频帧上引入随机空间网格采样来进行帧内防御。然后，我们通过守护映射设计像素级随机化，全局中和对抗性扰动。同时，我们从视频序列中提取时间信息作为帧间防御的补偿。基于这些原则，我们从面向安全的角度提出了一种新的VQA框架，称为SecureVQA。广泛的实验表明，SecureVQA在安全方面树立了新的基准，同时与最先进的型号相比，获得了具有竞争力的VQA性能。烧蚀研究深入分析了安全VQA的原理，展示了它们的概括性和对领先VQA模型的安全性的贡献。



## **14. Understanding Model Ensemble in Transferable Adversarial Attack**

了解可转移对抗攻击中的模型集合 cs.LG

**SubmitDate**: 2024-10-09    [abs](http://arxiv.org/abs/2410.06851v1) [paper-pdf](http://arxiv.org/pdf/2410.06851v1)

**Authors**: Wei Yao, Zeliang Zhang, Huayi Tang, Yong Liu

**Abstract**: Model ensemble adversarial attack has become a powerful method for generating transferable adversarial examples that can target even unknown models, but its theoretical foundation remains underexplored. To address this gap, we provide early theoretical insights that serve as a roadmap for advancing model ensemble adversarial attack. We first define transferability error to measure the error in adversarial transferability, alongside concepts of diversity and empirical model ensemble Rademacher complexity. We then decompose the transferability error into vulnerability, diversity, and a constant, which rigidly explains the origin of transferability error in model ensemble attack: the vulnerability of an adversarial example to ensemble components, and the diversity of ensemble components. Furthermore, we apply the latest mathematical tools in information theory to bound the transferability error using complexity and generalization terms, contributing to three practical guidelines for reducing transferability error: (1) incorporating more surrogate models, (2) increasing their diversity, and (3) reducing their complexity in cases of overfitting. Finally, extensive experiments with 54 models validate our theoretical framework, representing a significant step forward in understanding transferable model ensemble adversarial attacks.

摘要: 模型集成对抗性攻击已经成为一种生成可转移对抗性实例的有效方法，甚至可以针对未知模型，但其理论基础仍未得到充分探讨。为了解决这一差距，我们提供了早期的理论见解，作为推进模型集成对抗性攻击的路线图。我们首先定义了可转移性误差来度量对抗性可转移性的误差，以及多样性和经验模型集成Rademacher复杂性的概念。然后，我们将可转移性错误分解为脆弱性、多样性和一个常数，这严格解释了模型集成攻击中可转移性错误的来源：对抗性示例对集成组件的脆弱性以及集成组件的多样性。此外，我们应用信息论中最新的数学工具，使用复杂性和泛化项来限定可传递误差，为减少可传递误差提供了三个实用指导方针：(1)纳入更多的替代模型，(2)增加它们的多样性，(3)在过拟合的情况下降低它们的复杂性。最后，对54个模型的大量实验验证了我们的理论框架，这代表着在理解可转移模型集成对抗性攻击方面向前迈出了重要的一步。



## **15. On the Byzantine-Resilience of Distillation-Based Federated Learning**

基于蒸馏的联邦学习的拜占庭弹性 cs.LG

**SubmitDate**: 2024-10-09    [abs](http://arxiv.org/abs/2402.12265v2) [paper-pdf](http://arxiv.org/pdf/2402.12265v2)

**Authors**: Christophe Roux, Max Zimmer, Sebastian Pokutta

**Abstract**: Federated Learning (FL) algorithms using Knowledge Distillation (KD) have received increasing attention due to their favorable properties with respect to privacy, non-i.i.d. data and communication cost. These methods depart from transmitting model parameters and instead communicate information about a learning task by sharing predictions on a public dataset. In this work, we study the performance of such approaches in the byzantine setting, where a subset of the clients act in an adversarial manner aiming to disrupt the learning process. We show that KD-based FL algorithms are remarkably resilient and analyze how byzantine clients can influence the learning process. Based on these insights, we introduce two new byzantine attacks and demonstrate their ability to break existing byzantine-resilient methods. Additionally, we propose a novel defence method which enhances the byzantine resilience of KD-based FL algorithms. Finally, we provide a general framework to obfuscate attacks, making them significantly harder to detect, thereby improving their effectiveness. Our findings serve as an important building block in the analysis of byzantine FL, contributing through the development of new attacks and new defence mechanisms, further advancing the robustness of KD-based FL algorithms.

摘要: 基于知识蒸馏(KD)的联合学习(FL)算法因其在隐私、非I.I.D.等方面的良好特性而受到越来越多的关注。数据和通信成本。这些方法不同于传输模型参数，而是通过共享对公共数据集的预测来传递关于学习任务的信息。在这项工作中，我们研究了这些方法在拜占庭环境下的性能，在拜占庭环境中，客户的子集以对抗性的方式行动，旨在扰乱学习过程。我们证明了基于KD的FL算法具有显著的弹性，并分析了拜占庭客户端如何影响学习过程。基于这些见解，我们引入了两个新的拜占庭攻击，并展示了它们打破现有拜占庭弹性方法的能力。此外，我们还提出了一种新的防御方法，增强了基于KD的FL算法的拜占庭抗攻击能力。最后，我们提供了一个通用框架来混淆攻击，使它们更难被检测到，从而提高了它们的有效性。我们的发现是拜占庭FL分析的重要组成部分，通过开发新的攻击和新的防御机制，进一步提高了基于KD的FL算法的健壮性。



## **16. Read Over the Lines: Attacking LLMs and Toxicity Detection Systems with ASCII Art to Mask Profanity**

阅读字里行间：用ASC艺术攻击LLM和毒性检测系统以掩盖亵渎 cs.CL

**SubmitDate**: 2024-10-09    [abs](http://arxiv.org/abs/2409.18708v4) [paper-pdf](http://arxiv.org/pdf/2409.18708v4)

**Authors**: Sergey Berezin, Reza Farahbakhsh, Noel Crespi

**Abstract**: We introduce a novel family of adversarial attacks that exploit the inability of language models to interpret ASCII art. To evaluate these attacks, we propose the ToxASCII benchmark and develop two custom ASCII art fonts: one leveraging special tokens and another using text-filled letter shapes. Our attacks achieve a perfect 1.0 Attack Success Rate across ten models, including OpenAI's o1-preview and LLaMA 3.1.   Warning: this paper contains examples of toxic language used for research purposes.

摘要: 我们引入了一系列新颖的对抗性攻击，它们利用语言模型无法解释ASC艺术。为了评估这些攻击，我们提出了ToxASC基准并开发了两种自定义的ASC艺术字体：一种利用特殊标记，另一种使用文本填充字母形状。我们的攻击在十个模型中实现了完美的1.0攻击成功率，包括OpenAI的o 1-预览和LLaMA 3.1。   警告：本文包含用于研究目的的有毒语言的例子。



## **17. Faithfulness and the Notion of Adversarial Sensitivity in NLP Explanations**

NLP解释中的忠实性和对抗敏感性概念 cs.CL

Accepted as a Full Paper at EMNLP 2024 Workshop BlackBoxNLP

**SubmitDate**: 2024-10-09    [abs](http://arxiv.org/abs/2409.17774v2) [paper-pdf](http://arxiv.org/pdf/2409.17774v2)

**Authors**: Supriya Manna, Niladri Sett

**Abstract**: Faithfulness is arguably the most critical metric to assess the reliability of explainable AI. In NLP, current methods for faithfulness evaluation are fraught with discrepancies and biases, often failing to capture the true reasoning of models. We introduce Adversarial Sensitivity as a novel approach to faithfulness evaluation, focusing on the explainer's response when the model is under adversarial attack. Our method accounts for the faithfulness of explainers by capturing sensitivity to adversarial input changes. This work addresses significant limitations in existing evaluation techniques, and furthermore, quantifies faithfulness from a crucial yet underexplored paradigm.

摘要: 忠诚度可以说是评估可解释人工智能可靠性的最关键指标。在NLP中，当前的忠诚度评估方法充满了差异和偏见，通常无法捕捉模型的真实推理。我们引入对抗敏感性作为忠诚度评估的一种新颖方法，重点关注模型受到对抗攻击时解释者的反应。我们的方法通过捕捉对对抗性输入变化的敏感性来解释解释者的忠诚度。这项工作解决了现有评估技术的显着局限性，并且从一个关键但未充分探索的范式中量化了忠诚度。



## **18. TASAR: Transfer-based Attack on Skeletal Action Recognition**

TASSAR：基于传输的对Skelty动作识别的攻击 cs.CV

**SubmitDate**: 2024-10-09    [abs](http://arxiv.org/abs/2409.02483v2) [paper-pdf](http://arxiv.org/pdf/2409.02483v2)

**Authors**: Yunfeng Diao, Baiqi Wu, Ruixuan Zhang, Ajian Liu, Xingxing Wei, Meng Wang, He Wang

**Abstract**: Skeletal sequences, as well-structured representations of human behaviors, play a vital role in Human Activity Recognition (HAR). The transferability of adversarial skeletal sequences enables attacks in real-world HAR scenarios, such as autonomous driving, intelligent surveillance, and human-computer interactions. However, most existing skeleton-based HAR (S-HAR) attacks are primarily designed for white-box scenarios and exhibit weak adversarial transferability. Therefore, they cannot be considered true transfer-based S-HAR attacks. More importantly, the reason for this failure remains unclear. In this paper, we study this phenomenon through the lens of loss surface, and find that its sharpness contributes to the weak transferability in S-HAR. Inspired by this observation, we assume and empirically validate that smoothening the rugged loss landscape could potentially improve adversarial transferability in S-HAR. To this end, we propose the first \textbf{T}ransfer-based \textbf{A}ttack on \textbf{S}keletal \textbf{A}ction \textbf{R}ecognition, TASAR. TASAR explores the smoothed model posterior without requiring surrogate re-training, which is achieved by a new post-train Dual Bayesian optimization strategy. Furthermore, unlike previous transfer-based attacks that treat each frame independently and overlook temporal coherence within sequences, TASAR incorporates motion dynamics into the Bayesian attack gradient, effectively disrupting the spatial-temporal coherence of S-HARs. To exhaustively evaluate the effectiveness of existing methods and our method, we build the first large-scale robust S-HAR benchmark, comprising 7 S-HAR models, 10 attack methods, 3 S-HAR datasets and 2 defense methods. Extensive results demonstrate the superiority of TASAR. Our benchmark enables easy comparisons for future studies, with the code available in the supplementary material.

摘要: 骨架序列作为人类行为的良好结构表征，在人类活动识别(HAR)中起着至关重要的作用。对抗性骨架序列的可转移性使攻击能够在真实世界的HAR场景中进行，例如自动驾驶、智能监控和人机交互。然而，现有的大多数基于骨架的HAR(S-HAR)攻击主要是针对白盒场景而设计的，表现出较弱的对抗可转移性。因此，它们不能被认为是真正的基于转移的S-哈尔袭击。更重要的是，这一失败的原因尚不清楚。本文从损失面的角度对这一现象进行了研究，发现损失面的锐性是S-哈尔转移性较弱的原因之一。受到这一观察的启发，我们假设并经验验证，平滑崎岖的损失图景可能会提高S-哈尔的对抗性转移能力。为此，我们提出了第一个基于Textbf{T}转移的Tasar骨骼/Textbf{A}骨骼/Textbf{R}生态识别方法。Tasar探索平滑的后验模型，不需要代理重新训练，这是通过一种新的训练后双贝叶斯优化策略实现的。此外，与以往基于传输的攻击独立对待每一帧并忽略序列内部的时间一致性不同，Tasar将运动动力学融入到贝叶斯攻击梯度中，有效地破坏了S-HARs的时空一致性。为了全面评估已有方法和本文方法的有效性，我们构建了第一个大规模稳健的S-HAR基准，包括7个S-HAR模型、10种攻击方法、3个S-HAR数据集和2种防御方法。广泛的结果证明了Tasar的优越性。我们的基准可以很容易地与补充材料中提供的代码进行比较，以便将来进行研究。



## **19. PII-Scope: A Benchmark for Training Data PII Leakage Assessment in LLMs**

PII-Scope：LLM中训练数据PIP泄漏评估的基准 cs.CL

**SubmitDate**: 2024-10-09    [abs](http://arxiv.org/abs/2410.06704v1) [paper-pdf](http://arxiv.org/pdf/2410.06704v1)

**Authors**: Krishna Kanth Nakka, Ahmed Frikha, Ricardo Mendes, Xue Jiang, Xuebing Zhou

**Abstract**: In this work, we introduce PII-Scope, a comprehensive benchmark designed to evaluate state-of-the-art methodologies for PII extraction attacks targeting LLMs across diverse threat settings. Our study provides a deeper understanding of these attacks by uncovering several hyperparameters (e.g., demonstration selection) crucial to their effectiveness. Building on this understanding, we extend our study to more realistic attack scenarios, exploring PII attacks that employ advanced adversarial strategies, including repeated and diverse querying, and leveraging iterative learning for continual PII extraction. Through extensive experimentation, our results reveal a notable underestimation of PII leakage in existing single-query attacks. In fact, we show that with sophisticated adversarial capabilities and a limited query budget, PII extraction rates can increase by up to fivefold when targeting the pretrained model. Moreover, we evaluate PII leakage on finetuned models, showing that they are more vulnerable to leakage than pretrained models. Overall, our work establishes a rigorous empirical benchmark for PII extraction attacks in realistic threat scenarios and provides a strong foundation for developing effective mitigation strategies.

摘要: 在这项工作中，我们引入了PII-Scope，这是一个全面的基准测试，旨在评估针对不同威胁设置的LLM的PII提取攻击的最新方法。我们的研究通过揭示对这些攻击的有效性至关重要的几个超参数(例如，示范选择)来提供对这些攻击的更深层次的理解。在此基础上，我们将我们的研究扩展到更现实的攻击场景，探索使用高级对抗性策略的PII攻击，包括重复和多样化的查询，并利用迭代学习来持续提取PII。通过广泛的实验，我们的结果揭示了在现有的单查询攻击中对PII泄漏的显著低估。事实上，我们表明，在复杂的对抗能力和有限的查询预算下，当针对预先训练的模型时，PII提取率可以提高高达五倍。此外，我们评估了精调模型上的PII泄漏，表明它们比预先训练的模型更容易受到泄漏的影响。总体而言，我们的工作为现实威胁场景中的PII提取攻击建立了严格的经验基准，并为制定有效的缓解策略提供了坚实的基础。



## **20. Break the Visual Perception: Adversarial Attacks Targeting Encoded Visual Tokens of Large Vision-Language Models**

打破视觉感知：针对大型视觉语言模型的编码视觉标记的对抗攻击 cs.CV

Accepted to ACMMM 2024

**SubmitDate**: 2024-10-09    [abs](http://arxiv.org/abs/2410.06699v1) [paper-pdf](http://arxiv.org/pdf/2410.06699v1)

**Authors**: Yubo Wang, Chaohu Liu, Yanqiu Qu, Haoyu Cao, Deqiang Jiang, Linli Xu

**Abstract**: Large vision-language models (LVLMs) integrate visual information into large language models, showcasing remarkable multi-modal conversational capabilities. However, the visual modules introduces new challenges in terms of robustness for LVLMs, as attackers can craft adversarial images that are visually clean but may mislead the model to generate incorrect answers. In general, LVLMs rely on vision encoders to transform images into visual tokens, which are crucial for the language models to perceive image contents effectively. Therefore, we are curious about one question: Can LVLMs still generate correct responses when the encoded visual tokens are attacked and disrupting the visual information? To this end, we propose a non-targeted attack method referred to as VT-Attack (Visual Tokens Attack), which constructs adversarial examples from multiple perspectives, with the goal of comprehensively disrupting feature representations and inherent relationships as well as the semantic properties of visual tokens output by image encoders. Using only access to the image encoder in the proposed attack, the generated adversarial examples exhibit transferability across diverse LVLMs utilizing the same image encoder and generality across different tasks. Extensive experiments validate the superior attack performance of the VT-Attack over baseline methods, demonstrating its effectiveness in attacking LVLMs with image encoders, which in turn can provide guidance on the robustness of LVLMs, particularly in terms of the stability of the visual feature space.

摘要: 大型视觉语言模型(LVLM)将视觉信息集成到大型语言模型中，展示了非凡的多模式对话能力。然而，视觉模块在稳健性方面为LVLMS带来了新的挑战，因为攻击者可以手工制作视觉上干净但可能误导模型生成错误答案的对抗性图像。通常，视觉编码依赖于视觉编码器将图像转换为视觉标记，这对于语言模型有效地感知图像内容是至关重要的。因此，我们好奇一个问题：当编码的视觉令牌受到攻击并扰乱视觉信息时，LVLMS还能产生正确的反应吗？为此，我们提出了一种非目标攻击方法，称为VT-Attack(视觉标记攻击)，它从多个角度构造对抗性实例，目的是综合破坏图像编码者输出的视觉标记的特征表示和内在关系以及语义属性。在所提出的攻击中，仅使用对图像编码器的访问，生成的敌意示例表现出在使用相同图像编码器的不同LVLM之间的可转移性和跨不同任务的通用性。大量的实验验证了VT攻击相对于基线方法的优越攻击性能，证明了其在利用图像编码器攻击LVLM方面的有效性，进而可以为LVLMS的稳健性，特别是视觉特征空间的稳定性提供指导。



## **21. Does Vec2Text Pose a New Corpus Poisoning Threat?**

Vec 2文本是否构成新的Corpus中毒威胁？ cs.IR

arXiv admin note: substantial text overlap with arXiv:2402.12784

**SubmitDate**: 2024-10-09    [abs](http://arxiv.org/abs/2410.06628v1) [paper-pdf](http://arxiv.org/pdf/2410.06628v1)

**Authors**: Shengyao Zhuang, Bevan Koopman, Guido Zuccon

**Abstract**: The emergence of Vec2Text -- a method for text embedding inversion -- has raised serious privacy concerns for dense retrieval systems which use text embeddings. This threat comes from the ability for an attacker with access to embeddings to reconstruct the original text. In this paper, we take a new look at Vec2Text and investigate how much of a threat it poses to the different attacks of corpus poisoning, whereby an attacker injects adversarial passages into a retrieval corpus with the intention of misleading dense retrievers. Theoretically, Vec2Text is far more dangerous than previous attack methods because it does not need access to the embedding model's weights and it can efficiently generate many adversarial passages. We show that under certain conditions, corpus poisoning with Vec2Text can pose a serious threat to dense retriever system integrity and user experience by injecting adversarial passaged into top ranked positions. Code and data are made available at https://github.com/ielab/vec2text-corpus-poisoning

摘要: Vec2Text是一种文本嵌入倒排的方法，它的出现给使用文本嵌入的密集检索系统带来了严重的隐私问题。该威胁来自具有嵌入访问权限的攻击者重建原始文本的能力。在本文中，我们重新审视了Vec2Text，并调查了它对语料库中毒的不同攻击构成了多大的威胁，即攻击者将对抗性段落注入检索语料库，意图误导密集检索者。从理论上讲，Vec2Text比以往的攻击方法更危险，因为它不需要访问嵌入模型的权重，并且可以高效地生成许多对抗性段落。研究结果表明，在一定条件下，Vec2Text的语料库中毒会对密集检索系统的完整性和用户体验造成严重威胁，因为它会向排名靠前的位置注入对抗性通道。代码和数据可在https://github.com/ielab/vec2text-corpus-poisoning上获得



## **22. ETA: Evaluating Then Aligning Safety of Vision Language Models at Inference Time**

埃塔：在推理时间评估然后调整视觉语言模型的安全性 cs.CV

27pages

**SubmitDate**: 2024-10-09    [abs](http://arxiv.org/abs/2410.06625v1) [paper-pdf](http://arxiv.org/pdf/2410.06625v1)

**Authors**: Yi Ding, Bolian Li, Ruqi Zhang

**Abstract**: Vision Language Models (VLMs) have become essential backbones for multimodal intelligence, yet significant safety challenges limit their real-world application. While textual inputs are often effectively safeguarded, adversarial visual inputs can easily bypass VLM defense mechanisms. Existing defense methods are either resource-intensive, requiring substantial data and compute, or fail to simultaneously ensure safety and usefulness in responses. To address these limitations, we propose a novel two-phase inference-time alignment framework, Evaluating Then Aligning (ETA): 1) Evaluating input visual contents and output responses to establish a robust safety awareness in multimodal settings, and 2) Aligning unsafe behaviors at both shallow and deep levels by conditioning the VLMs' generative distribution with an interference prefix and performing sentence-level best-of-N to search the most harmless and helpful generation paths. Extensive experiments show that ETA outperforms baseline methods in terms of harmlessness, helpfulness, and efficiency, reducing the unsafe rate by 87.5% in cross-modality attacks and achieving 96.6% win-ties in GPT-4 helpfulness evaluation. The code is publicly available at https://github.com/DripNowhy/ETA.

摘要: 视觉语言模型已经成为多模式智能的重要支柱，但巨大的安全挑战限制了它们在现实世界中的应用。虽然文本输入通常受到有效保护，但对抗性视觉输入可以很容易地绕过VLM防御机制。现有的防御方法要么是资源密集型的，需要大量的数据和计算，要么无法同时确保响应的安全性和实用性。为了解决这些局限性，我们提出了一种新的两阶段推理-时间对齐框架，评估然后对齐(ETA)：1)评估输入视觉内容和输出响应以在多模式环境中建立稳健的安全意识；2)通过用干扰前缀限制VLM的生成分布并执行句子级的Best-of-N来搜索最无害和最有帮助的生成路径，在浅层和深层对不安全行为进行对齐。大量实验表明，ETA在无害性、有助性和有效性方面都优于基线方法，在跨通道攻击中降低了87.5%的不安全率，在GPT-4有助性评估中获得了96.6%的优胜率。该代码可在https://github.com/DripNowhy/ETA.上公开获得



## **23. Can DeepFake Speech be Reliably Detected?**

DeepFake Speech能被可靠检测到吗？ cs.SD

**SubmitDate**: 2024-10-09    [abs](http://arxiv.org/abs/2410.06572v1) [paper-pdf](http://arxiv.org/pdf/2410.06572v1)

**Authors**: Hongbin Liu, Youzheng Chen, Arun Narayanan, Athula Balachandran, Pedro J. Moreno, Lun Wang

**Abstract**: Recent advances in text-to-speech (TTS) systems, particularly those with voice cloning capabilities, have made voice impersonation readily accessible, raising ethical and legal concerns due to potential misuse for malicious activities like misinformation campaigns and fraud. While synthetic speech detectors (SSDs) exist to combat this, they are vulnerable to ``test domain shift", exhibiting decreased performance when audio is altered through transcoding, playback, or background noise. This vulnerability is further exacerbated by deliberate manipulation of synthetic speech aimed at deceiving detectors. This work presents the first systematic study of such active malicious attacks against state-of-the-art open-source SSDs. White-box attacks, black-box attacks, and their transferability are studied from both attack effectiveness and stealthiness, using both hardcoded metrics and human ratings. The results highlight the urgent need for more robust detection methods in the face of evolving adversarial threats.

摘要: 文本到语音(TTS)系统的最新进展，特别是那些具有语音克隆能力的系统，使得语音模拟变得容易访问，由于可能被滥用于虚假信息运动和欺诈等恶意活动，这引发了伦理和法律方面的担忧。虽然合成语音检测器(SSD)可以应对这一问题，但它们很容易受到`测试域漂移‘的影响，当音频通过转码、回放或背景噪声改变时，它们的性能会下降。故意操纵合成语音以欺骗检测器，进一步加剧了这一漏洞。这项工作首次系统地研究了针对最先进的开源SSD的此类主动恶意攻击。白盒攻击、黑盒攻击及其可转移性从攻击有效性和隐蔽性两个方面进行了研究，使用硬编码指标和人类评级。结果突显了面对不断变化的对手威胁，迫切需要更健壮的检测方法。



## **24. Hallucinating AI Hijacking Attack: Large Language Models and Malicious Code Recommenders**

幻觉人工智能劫持攻击：大型语言模型和恶意代码推荐 cs.CR

**SubmitDate**: 2024-10-09    [abs](http://arxiv.org/abs/2410.06462v1) [paper-pdf](http://arxiv.org/pdf/2410.06462v1)

**Authors**: David Noever, Forrest McKee

**Abstract**: The research builds and evaluates the adversarial potential to introduce copied code or hallucinated AI recommendations for malicious code in popular code repositories. While foundational large language models (LLMs) from OpenAI, Google, and Anthropic guard against both harmful behaviors and toxic strings, previous work on math solutions that embed harmful prompts demonstrate that the guardrails may differ between expert contexts. These loopholes would appear in mixture of expert's models when the context of the question changes and may offer fewer malicious training examples to filter toxic comments or recommended offensive actions. The present work demonstrates that foundational models may refuse to propose destructive actions correctly when prompted overtly but may unfortunately drop their guard when presented with a sudden change of context, like solving a computer programming challenge. We show empirical examples with trojan-hosting repositories like GitHub, NPM, NuGet, and popular content delivery networks (CDN) like jsDelivr which amplify the attack surface. In the LLM's directives to be helpful, example recommendations propose application programming interface (API) endpoints which a determined domain-squatter could acquire and setup attack mobile infrastructure that triggers from the naively copied code. We compare this attack to previous work on context-shifting and contrast the attack surface as a novel version of "living off the land" attacks in the malware literature. In the latter case, foundational language models can hijack otherwise innocent user prompts to recommend actions that violate their owners' safety policies when posed directly without the accompanying coding support request.

摘要: 这项研究构建并评估了在流行的代码库中引入复制代码或幻觉AI建议的恶意代码的敌意潜力。虽然OpenAI、谷歌和人类的基础大型语言模型(LLM)可以防范有害行为和有毒字符串，但之前关于嵌入有害提示的数学解决方案的工作表明，护栏可能会因专家上下文而异。当问题的上下文发生变化时，这些漏洞将出现在专家模型的混合中，并且可能提供较少的恶意训练示例来过滤有毒评论或建议的攻击性操作。目前的工作表明，基础模型可能会在公开提示时拒绝正确地提出破坏性行动，但不幸的是，当环境突然改变时，可能会放松警惕，比如解决计算机编程挑战。我们使用GitHub、NPM、NuGet等木马托管库和jsDelivr等流行的内容交付网络(CDN)展示了放大攻击面的经验示例。在LLM的有用指令中，示例建议提出了应用程序编程接口(API)端点，确定的域抢占者可以获取这些端点，并建立从简单复制的代码触发的攻击移动基础设施。我们将这一攻击与之前关于上下文转换的工作进行了比较，并将攻击面作为恶意软件文献中的一个新版本的“赖以生存”的攻击进行了对比。在后一种情况下，基础语言模型可以劫持其他无辜的用户提示，在没有附带的编码支持请求的情况下直接提出违反其所有者安全政策的行为。



## **25. Filtered Randomized Smoothing: A New Defense for Robust Modulation Classification**

过滤随机平滑：鲁棒调制分类的新防御 cs.LG

IEEE Milcom 2024

**SubmitDate**: 2024-10-08    [abs](http://arxiv.org/abs/2410.06339v1) [paper-pdf](http://arxiv.org/pdf/2410.06339v1)

**Authors**: Wenhan Zhang, Meiyu Zhong, Ravi Tandon, Marwan Krunz

**Abstract**: Deep Neural Network (DNN) based classifiers have recently been used for the modulation classification of RF signals. These classifiers have shown impressive performance gains relative to conventional methods, however, they are vulnerable to imperceptible (low-power) adversarial attacks. Some of the prominent defense approaches include adversarial training (AT) and randomized smoothing (RS). While AT increases robustness in general, it fails to provide resilience against previously unseen adaptive attacks. Other approaches, such as Randomized Smoothing (RS), which injects noise into the input, address this shortcoming by providing provable certified guarantees against arbitrary attacks, however, they tend to sacrifice accuracy.   In this paper, we study the problem of designing robust DNN-based modulation classifiers that can provide provable defense against arbitrary attacks without significantly sacrificing accuracy. To this end, we first analyze the spectral content of commonly studied attacks on modulation classifiers for the benchmark RadioML dataset. We observe that spectral signatures of un-perturbed RF signals are highly localized, whereas attack signals tend to be spread out in frequency. To exploit this spectral heterogeneity, we propose Filtered Randomized Smoothing (FRS), a novel defense which combines spectral filtering together with randomized smoothing. FRS can be viewed as a strengthening of RS by leveraging the specificity (spectral Heterogeneity) inherent to the modulation classification problem. In addition to providing an approach to compute the certified accuracy of FRS, we also provide a comprehensive set of simulations on the RadioML dataset to show the effectiveness of FRS and show that it significantly outperforms existing defenses including AT and RS in terms of accuracy on both attacked and benign signals.

摘要: 最近，基于深度神经网络(DNN)的分类器被用于射频信号的调制分类。与传统方法相比，这些分类器表现出了令人印象深刻的性能改进，然而，它们容易受到不可察觉的(低功率)对手攻击。一些突出的防御方法包括对抗性训练(AT)和随机平滑(RS)。虽然AT总体上提高了健壮性，但它无法提供对以前未见过的自适应攻击的弹性。其他方法，如向输入中注入噪声的随机平滑(RS)，通过提供针对任意攻击的可证明的认证保证来解决这一缺点，然而，它们往往会牺牲准确性。在这篇文章中，我们研究了基于DNN的稳健调制分类器的设计问题，该分类器能够在不显著牺牲精度的情况下提供对任意攻击的可证明防御。为此，我们首先分析了针对基准RadioML数据集的调制分类器的常用攻击的频谱内容。我们观察到，未受干扰的射频信号的频谱特征是高度局部化的，而攻击信号倾向于在频率上扩散。为了利用这种光谱异构性，我们提出了滤波随机平滑(FRS)，这是一种将谱滤波和随机平滑相结合的新防御方法。FRS可以被视为通过利用调制分类问题固有的特殊性(频谱异质性)来加强RS。除了提供一种计算FRS认证精度的方法外，我们还在RadioML数据集上进行了一组全面的仿真，以显示FRS的有效性，并表明它在攻击和良性信号的准确性方面明显优于现有的防御系统，包括AT和RS。



## **26. Evaluating and Safeguarding the Adversarial Robustness of Retrieval-Based In-Context Learning**

评估和保障基于检索的上下文学习的对抗鲁棒性 cs.CL

COLM 2024, 31 pages, 6 figures

**SubmitDate**: 2024-10-08    [abs](http://arxiv.org/abs/2405.15984v4) [paper-pdf](http://arxiv.org/pdf/2405.15984v4)

**Authors**: Simon Yu, Jie He, Pasquale Minervini, Jeff Z. Pan

**Abstract**: With the emergence of large language models, such as LLaMA and OpenAI GPT-3, In-Context Learning (ICL) gained significant attention due to its effectiveness and efficiency. However, ICL is very sensitive to the choice, order, and verbaliser used to encode the demonstrations in the prompt. Retrieval-Augmented ICL methods try to address this problem by leveraging retrievers to extract semantically related examples as demonstrations. While this approach yields more accurate results, its robustness against various types of adversarial attacks, including perturbations on test samples, demonstrations, and retrieved data, remains under-explored. Our study reveals that retrieval-augmented models can enhance robustness against test sample attacks, outperforming vanilla ICL with a 4.87% reduction in Attack Success Rate (ASR); however, they exhibit overconfidence in the demonstrations, leading to a 2% increase in ASR for demonstration attacks. Adversarial training can help improve the robustness of ICL methods to adversarial attacks; however, such a training scheme can be too costly in the context of LLMs. As an alternative, we introduce an effective training-free adversarial defence method, DARD, which enriches the example pool with those attacked samples. We show that DARD yields improvements in performance and robustness, achieving a 15% reduction in ASR over the baselines. Code and data are released to encourage further research: https://github.com/simonucl/adv-retreival-icl

摘要: 随着大型语言模型的出现，如Llama和OpenAI GPT-3，情景中学习(ICL)因其有效性和高效性而受到广泛关注。但是，ICL对用于对提示符中的演示进行编码的选择、顺序和形容词非常敏感。检索增强的ICL方法试图通过利用检索器来提取语义相关的示例作为演示来解决这个问题。虽然这种方法可以产生更准确的结果，但它对各种类型的对抗性攻击的稳健性，包括对测试样本、演示和检索数据的扰动，仍然没有得到充分的研究。我们的研究表明，检索增强模型可以增强对测试样本攻击的健壮性，性能优于普通ICL，攻击成功率(ASR)降低4.87%；然而，它们在演示中表现出过度自信，导致演示攻击的ASR提高了2%。对抗性训练可以帮助提高ICL方法对对抗性攻击的稳健性；然而，在LLMS的背景下，这样的训练方案可能代价太高。作为另一种选择，我们引入了一种有效的无需训练的对抗防御方法DARD，它用被攻击的样本丰富了样本库。我们表明，DARD在性能和健壮性方面都有改进，ASR比基准降低了15%。发布代码和数据是为了鼓励进一步的研究：https://github.com/simonucl/adv-retreival-icl



## **27. The Last Iterate Advantage: Empirical Auditing and Principled Heuristic Analysis of Differentially Private SGD**

最后的迭代优势：差异化私人新元的经验审计和原则性启发式分析 cs.CR

**SubmitDate**: 2024-10-10    [abs](http://arxiv.org/abs/2410.06186v2) [paper-pdf](http://arxiv.org/pdf/2410.06186v2)

**Authors**: Thomas Steinke, Milad Nasr, Arun Ganesh, Borja Balle, Christopher A. Choquette-Choo, Matthew Jagielski, Jamie Hayes, Abhradeep Guha Thakurta, Adam Smith, Andreas Terzis

**Abstract**: We propose a simple heuristic privacy analysis of noisy clipped stochastic gradient descent (DP-SGD) in the setting where only the last iterate is released and the intermediate iterates remain hidden. Namely, our heuristic assumes a linear structure for the model.   We show experimentally that our heuristic is predictive of the outcome of privacy auditing applied to various training procedures. Thus it can be used prior to training as a rough estimate of the final privacy leakage. We also probe the limitations of our heuristic by providing some artificial counterexamples where it underestimates the privacy leakage.   The standard composition-based privacy analysis of DP-SGD effectively assumes that the adversary has access to all intermediate iterates, which is often unrealistic. However, this analysis remains the state of the art in practice. While our heuristic does not replace a rigorous privacy analysis, it illustrates the large gap between the best theoretical upper bounds and the privacy auditing lower bounds and sets a target for further work to improve the theoretical privacy analyses. We also empirically support our heuristic and show existing privacy auditing attacks are bounded by our heuristic analysis in both vision and language tasks.

摘要: 在只释放最后一次迭代而隐藏中间迭代的情况下，提出了一种简单的启发式噪声截断随机梯度下降(DP-SGD)隐私分析方法。也就是说，我们的启发式假设模型是线性结构。我们的实验表明，我们的启发式方法可以预测隐私审计应用于各种训练过程的结果。因此，它可以在培训前用作最终隐私泄露的粗略估计。我们还通过提供一些低估隐私泄露的人工反例来探讨我们的启发式算法的局限性。标准的基于组合的DP-SGD隐私分析有效地假设攻击者可以访问所有中间迭代，这通常是不现实的。然而，这种分析在实践中仍然是最先进的。虽然我们的启发式方法没有取代严格的隐私分析，但它说明了最佳理论上限和隐私审计下限之间的巨大差距，并为进一步改进理论隐私分析设定了目标。我们还实证地支持我们的启发式攻击，并表明现有的隐私审计攻击受到我们在视觉和语言任务中的启发式分析的约束。



## **28. Position: Towards Resilience Against Adversarial Examples**

立场：增强对抗性例子的韧性 cs.LG

**SubmitDate**: 2024-10-08    [abs](http://arxiv.org/abs/2405.01349v2) [paper-pdf](http://arxiv.org/pdf/2405.01349v2)

**Authors**: Sihui Dai, Chong Xiang, Tong Wu, Prateek Mittal

**Abstract**: Current research on defending against adversarial examples focuses primarily on achieving robustness against a single attack type such as $\ell_2$ or $\ell_{\infty}$-bounded attacks. However, the space of possible perturbations is much larger than considered by many existing defenses and is difficult to mathematically model, so the attacker can easily bypass the defense by using a type of attack that is not covered by the defense. In this position paper, we argue that in addition to robustness, we should also aim to develop defense algorithms that are adversarially resilient -- defense algorithms should specify a means to quickly adapt the defended model to be robust against new attacks. We provide a definition of adversarial resilience and outline considerations of designing an adversarially resilient defense. We then introduce a subproblem of adversarial resilience which we call continual adaptive robustness, in which the defender gains knowledge of the formulation of possible perturbation spaces over time and can then update their model based on this information. Additionally, we demonstrate the connection between continual adaptive robustness and previously studied problems of multiattack robustness and unforeseen attack robustness and outline open directions within these fields which can contribute to improving continual adaptive robustness and adversarial resilience.

摘要: 目前关于防御恶意攻击的研究主要集中在对单一攻击类型的健壮性上，例如$\ell_2$或$\ell_{\infty}$-bound攻击。然而，可能的扰动空间比许多现有防御措施所考虑的要大得多，而且很难建立数学模型，因此攻击者可以通过使用防御措施不涵盖的攻击类型来轻松绕过防御措施。在这份立场文件中，我们认为，除了稳健性之外，我们还应该致力于开发具有对抗弹性的防御算法--防御算法应该指定一种方法，以快速调整防御模型，使其对新的攻击具有健壮性。我们给出了对抗韧性的定义，并概述了设计对抗韧性防御的考虑因素。然后，我们引入了一个称为连续自适应稳健性的对抗性复原子问题，在这个问题中，防御者获得了关于可能的扰动空间随时间的形成的知识，然后可以基于这些信息来更新他们的模型。此外，我们还论证了连续自适应稳健性与以前研究的多攻击稳健性和不可预见攻击稳健性问题之间的联系，并概述了这些领域的开放方向，这有助于提高连续自适应稳健性和对抗韧性。



## **29. One Perturbation is Enough: On Generating Universal Adversarial Perturbations against Vision-Language Pre-training Models**

一个扰动就足够了：关于针对视觉语言预训练模型生成普遍对抗性扰动 cs.CV

**SubmitDate**: 2024-10-08    [abs](http://arxiv.org/abs/2406.05491v2) [paper-pdf](http://arxiv.org/pdf/2406.05491v2)

**Authors**: Hao Fang, Jiawei Kong, Wenbo Yu, Bin Chen, Jiawei Li, Shutao Xia, Ke Xu

**Abstract**: Vision-Language Pre-training (VLP) models have exhibited unprecedented capability in many applications by taking full advantage of the multimodal alignment. However, previous studies have shown they are vulnerable to maliciously crafted adversarial samples. Despite recent success, these methods are generally instance-specific and require generating perturbations for each input sample. In this paper, we reveal that VLP models are also vulnerable to the instance-agnostic universal adversarial perturbation (UAP). Specifically, we design a novel Contrastive-training Perturbation Generator with Cross-modal conditions (C-PGC) to achieve the attack. In light that the pivotal multimodal alignment is achieved through the advanced contrastive learning technique, we devise to turn this powerful weapon against themselves, i.e., employ a malicious version of contrastive learning to train the C-PGC based on our carefully crafted positive and negative image-text pairs for essentially destroying the alignment relationship learned by VLP models. Besides, C-PGC fully utilizes the characteristics of Vision-and-Language (V+L) scenarios by incorporating both unimodal and cross-modal information as effective guidance. Extensive experiments show that C-PGC successfully forces adversarial samples to move away from their original area in the VLP model's feature space, thus essentially enhancing attacks across various victim models and V+L tasks. The GitHub repository is available at https://github.com/ffhibnese/CPGC_VLP_Universal_Attacks.

摘要: 视觉语言预训练(VLP)模型充分利用了多通道对齐的优势，在许多应用中表现出了前所未有的能力。然而，之前的研究表明，它们很容易受到恶意制作的对手样本的攻击。尽管最近取得了成功，但这些方法通常是特定于实例的，需要为每个输入样本生成扰动。在这篇文章中，我们揭示了VLP模型也容易受到实例不可知的通用对抗扰动(UAP)的影响。具体地说，我们设计了一种新的具有交叉模式条件的对比训练扰动生成器(C-PGC)来实现攻击。鉴于关键的多通道对齐是通过先进的对比学习技术实现的，我们打算将这一强大的武器转化为针对自己的强大武器，即使用恶意版本的对比学习来训练基于我们精心设计的正和负图文对的C-PGC，以从根本上破坏VLP模型学习的对齐关系。此外，C-PGC充分利用了视觉与语言(V+L)情景的特点，融合了单峰和跨通道信息作为有效的指导。大量实验表明，C-PGC成功地迫使敌方样本在VLP模型的特征空间中离开其原始区域，从而本质上增强了对各种受害者模型和V+L任务的攻击。GitHub存储库可在https://github.com/ffhibnese/CPGC_VLP_Universal_Attacks.上获得



## **30. Transferability Bound Theory: Exploring Relationship between Adversarial Transferability and Flatness**

可转让性界限理论：探索对抗可转让性与平坦性之间的关系 cs.LG

Accepted by NeurIPS 2024

**SubmitDate**: 2024-10-08    [abs](http://arxiv.org/abs/2311.06423v3) [paper-pdf](http://arxiv.org/pdf/2311.06423v3)

**Authors**: Mingyuan Fan, Xiaodan Li, Cen Chen, Wenmeng Zhou, Yaliang Li

**Abstract**: A prevailing belief in attack and defense community is that the higher flatness of adversarial examples enables their better cross-model transferability, leading to a growing interest in employing sharpness-aware minimization and its variants. However, the theoretical relationship between the transferability of adversarial examples and their flatness has not been well established, making the belief questionable. To bridge this gap, we embark on a theoretical investigation and, for the first time, derive a theoretical bound for the transferability of adversarial examples with few practical assumptions. Our analysis challenges this belief by demonstrating that the increased flatness of adversarial examples does not necessarily guarantee improved transferability. Moreover, building upon the theoretical analysis, we propose TPA, a Theoretically Provable Attack that optimizes a surrogate of the derived bound to craft adversarial examples. Extensive experiments across widely used benchmark datasets and various real-world applications show that TPA can craft more transferable adversarial examples compared to state-of-the-art baselines. We hope that these results can recalibrate preconceived impressions within the community and facilitate the development of stronger adversarial attack and defense mechanisms. The source codes are available in <https://github.com/fmy266/TPA>.

摘要: 攻防界的一个普遍看法是，对抗性例子的较高平坦度使其具有更好的跨模型可转移性，从而导致人们对使用锐度感知最小化及其变体越来越感兴趣。然而，对抗性例子的可转移性与其平面性之间的理论关系还没有很好地建立起来，这使得人们对这一信念产生了怀疑。为了弥合这一差距，我们开始了一项理论研究，并首次在几乎没有实际假设的情况下推导出了对抗性例子的可转移性的理论界限。我们的分析挑战了这一信念，证明了对抗性例子的平坦性增加并不一定能保证更好的可转移性。此外，在理论分析的基础上，我们提出了TPA，这是一种理论上可证明的攻击，它优化了派生边界的代理，以伪造敌意示例。在广泛使用的基准数据集和各种真实世界应用程序上的广泛实验表明，与最先进的基线相比，TPA可以创建更多可转移的对抗性例子。我们希望这些结果可以重新校正社区内的先入为主的印象，并促进更强大的对抗性攻防机制的发展。源代码可在<https://github.com/fmy266/TPA>.中找到



## **31. Partially Recentralization Softmax Loss for Vision-Language Models Robustness**

部分再集中化Softmax因视觉语言模型鲁棒性而丧失 cs.CL

**SubmitDate**: 2024-10-08    [abs](http://arxiv.org/abs/2402.03627v2) [paper-pdf](http://arxiv.org/pdf/2402.03627v2)

**Authors**: Hao Wang, Jinzhe Jiang, Xin Zhang, Chen Li

**Abstract**: As Large Language Models make a breakthrough in natural language processing tasks (NLP), multimodal technique becomes extremely popular. However, it has been shown that multimodal NLP are vulnerable to adversarial attacks, where the outputs of a model can be dramatically changed by a perturbation to the input. While several defense techniques have been proposed both in computer vision and NLP models, the multimodal robustness of models have not been fully explored. In this paper, we study the adversarial robustness provided by modifying loss function of pre-trained multimodal models, by restricting top K softmax outputs. Based on the evaluation and scoring, our experiments show that after a fine-tuning, adversarial robustness of pre-trained models can be significantly improved, against popular attacks. Further research should be studying, such as output diversity, generalization and the robustness-performance trade-off of this kind of loss functions. Our code will be available after this paper is accepted

摘要: 随着大型语言模型在自然语言处理任务(NLP)方面的突破，多通道技术变得非常流行。然而，已有研究表明，多模式NLP很容易受到对抗性攻击，模型的输出可能会因输入的扰动而发生显著变化。虽然在计算机视觉和NLP模型中已经提出了几种防御技术，但模型的多通道稳健性还没有得到充分的研究。在本文中，我们研究了通过限制Top K Softmax输出来修改预先训练的多模式模型的损失函数所提供的对抗鲁棒性。在评估和评分的基础上，我们的实验表明，经过微调后，预先训练的模型对攻击的健壮性可以显著提高，对抗流行攻击。这类损失函数的输出分集、泛化以及稳健性与性能的权衡等问题还有待进一步研究。我们的代码将在这篇论文被接受后可用



## **32. ATM: Adversarial Tuning Multi-agent System Makes a Robust Retrieval-Augmented Generator**

ATM：对抗性调整多代理系统打造强大的检索增强生成器 cs.CL

18 pages

**SubmitDate**: 2024-10-08    [abs](http://arxiv.org/abs/2405.18111v3) [paper-pdf](http://arxiv.org/pdf/2405.18111v3)

**Authors**: Junda Zhu, Lingyong Yan, Haibo Shi, Dawei Yin, Lei Sha

**Abstract**: Large language models (LLMs) are proven to benefit a lot from retrieval-augmented generation (RAG) in alleviating hallucinations confronted with knowledge-intensive questions. RAG adopts information retrieval techniques to inject external knowledge from semantic-relevant documents as input contexts. However, since today's Internet is flooded with numerous noisy and fabricating content, it is inevitable that RAG systems are vulnerable to these noises and prone to respond incorrectly. To this end, we propose to optimize the retrieval-augmented Generator with an Adversarial Tuning Multi-agent system (ATM). The ATM steers the Generator to have a robust perspective of useful documents for question answering with the help of an auxiliary Attacker agent through adversarially tuning the agents for several iterations. After rounds of multi-agent iterative tuning, the Generator can eventually better discriminate useful documents amongst fabrications. The experimental results verify the effectiveness of ATM and we also observe that the Generator can achieve better performance compared to the state-of-the-art baselines.

摘要: 事实证明，大型语言模型(LLM)在缓解面对知识密集型问题时的幻觉方面，从检索增强生成(RAG)中受益匪浅。RAG采用信息检索技术，从与语义相关的文档中注入外部知识作为输入上下文。然而，由于当今的互联网充斥着大量噪声和捏造的内容，RAG系统不可避免地容易受到这些噪声的影响，并容易做出错误的响应。为此，我们提出了用对抗性调谐多智能体系统(ATM)来优化检索增强生成器。ATM通过对代理进行多次恶意调整，在辅助攻击者代理的帮助下，引导Generator具有用于问题回答的有用文档的健壮视角。经过几轮多代理迭代调整后，Generator最终可以更好地区分有用的文档和捏造的文档。实验结果验证了ATM的有效性，并且我们还观察到，与最先进的基线相比，该生成器可以获得更好的性能。



## **33. TaeBench: Improving Quality of Toxic Adversarial Examples**

TaeBench：提高有毒对抗性例子的质量 cs.CR

**SubmitDate**: 2024-10-08    [abs](http://arxiv.org/abs/2410.05573v1) [paper-pdf](http://arxiv.org/pdf/2410.05573v1)

**Authors**: Xuan Zhu, Dmitriy Bespalov, Liwen You, Ninad Kulkarni, Yanjun Qi

**Abstract**: Toxicity text detectors can be vulnerable to adversarial examples - small perturbations to input text that fool the systems into wrong detection. Existing attack algorithms are time-consuming and often produce invalid or ambiguous adversarial examples, making them less useful for evaluating or improving real-world toxicity content moderators. This paper proposes an annotation pipeline for quality control of generated toxic adversarial examples (TAE). We design model-based automated annotation and human-based quality verification to assess the quality requirements of TAE. Successful TAE should fool a target toxicity model into making benign predictions, be grammatically reasonable, appear natural like human-generated text, and exhibit semantic toxicity. When applying these requirements to more than 20 state-of-the-art (SOTA) TAE attack recipes, we find many invalid samples from a total of 940k raw TAE attack generations. We then utilize the proposed pipeline to filter and curate a high-quality TAE dataset we call TaeBench (of size 264k). Empirically, we demonstrate that TaeBench can effectively transfer-attack SOTA toxicity content moderation models and services. Our experiments also show that TaeBench with adversarial training achieve significant improvements of the robustness of two toxicity detectors.

摘要: 毒性文本检测器可能容易受到敌意示例的攻击-输入文本的微小扰动会欺骗系统进行错误检测。现有的攻击算法非常耗时，并且经常产生无效或模棱两可的对抗性示例，这使得它们对评估或改进真实世界的毒性内容调节器的用处较小。提出了一种用于生成有毒对抗性实例(TAE)的质量控制的标注管道。我们设计了基于模型的自动标注和基于人工的质量验证来评估TAE的质量需求。成功的TAE应该欺骗目标毒性模型做出良性预测，在语法上是合理的，看起来像人类生成的文本一样自然，并表现出语义毒性。当将这些要求应用于20多个最先进的(SOTA)TAE攻击配方时，我们从总共940k个原始TAE攻击世代中发现了许多无效样本。然后，我们利用所提出的管道来过滤和管理一个高质量的TAE数据集，我们称之为TaeBch(大小为264k)。实证方面，我们证明了TaeBitch能够有效地转移攻击SOTA毒性含量调节模型和服务。我们的实验还表明，TaeBtch结合对抗性训练，显著提高了两种毒物检测器的稳健性。



## **34. Cyber Threats to Canadian Federal Election: Emerging Threats, Assessment, and Mitigation Strategies**

加拿大联邦选举的网络威胁：新出现的威胁、评估和缓解策略 cs.CR

**SubmitDate**: 2024-10-07    [abs](http://arxiv.org/abs/2410.05560v1) [paper-pdf](http://arxiv.org/pdf/2410.05560v1)

**Authors**: Nazmul Islam, Soomin Kim, Mohammad Pirooz, Sasha Shvetsov

**Abstract**: As Canada prepares for the 2025 federal election, ensuring the integrity and security of the electoral process against cyber threats is crucial. Recent foreign interference in elections globally highlight the increasing sophistication of adversaries in exploiting technical and human vulnerabilities. Such vulnerabilities also exist in Canada's electoral system that relies on a complex network of IT systems, vendors, and personnel. To mitigate these vulnerabilities, a threat assessment is crucial to identify emerging threats, develop incident response capabilities, and build public trust and resilience against cyber threats. Therefore, this paper presents a comprehensive national cyber threat assessment, following the NIST Special Publication 800-30 framework, focusing on identifying and mitigating cybersecurity risks to the upcoming 2025 Canadian federal election. The research identifies three major threats: misinformation, disinformation, and malinformation (MDM) campaigns; attacks on critical infrastructure and election support systems; and espionage by malicious actors. Through detailed analysis, the assessment offers insights into the capabilities, intent, and potential impact of these threats. The paper also discusses emerging technologies and their influence on election security and proposes a multi-faceted approach to risk mitigation ahead of the election.

摘要: 在加拿大为2025年联邦选举做准备之际，确保选举过程的完整性和安全性不受网络威胁至关重要。最近外国对全球选举的干预突显出对手在利用技术和人的弱点方面日益老练。这种漏洞也存在于加拿大的选举系统中，该系统依赖于由IT系统、供应商和人员组成的复杂网络。为了缓解这些漏洞，威胁评估对于识别新出现的威胁、发展事件响应能力以及建立公众信任和抵御网络威胁的能力至关重要。因此，本文遵循NIST特别出版物800-30框架，提出了一项全面的国家网络威胁评估，重点是识别和缓解即将到来的2025年加拿大联邦选举的网络安全风险。该研究确定了三个主要威胁：错误信息、虚假信息和错误信息(MDM)活动；对关键基础设施和选举支持系统的攻击；以及恶意行为者的间谍活动。通过详细分析，评估提供了对这些威胁的能力、意图和潜在影响的洞察。白皮书还讨论了新兴技术及其对选举安全的影响，并提出了在选举前降低风险的多方面方法。



## **35. Kick Bad Guys Out! Conditionally Activated Anomaly Detection in Federated Learning with Zero-Knowledge Proof Verification**

把坏人踢出去！具有零知识证明验证的联邦学习中的一致激活异常检测 cs.CR

**SubmitDate**: 2024-10-07    [abs](http://arxiv.org/abs/2310.04055v4) [paper-pdf](http://arxiv.org/pdf/2310.04055v4)

**Authors**: Shanshan Han, Wenxuan Wu, Baturalp Buyukates, Weizhao Jin, Qifan Zhang, Yuhang Yao, Salman Avestimehr, Chaoyang He

**Abstract**: Federated Learning (FL) systems are susceptible to adversarial attacks, where malicious clients submit poisoned models to disrupt the convergence or plant backdoors that cause the global model to misclassify some samples. Current defense methods are often impractical for real-world FL systems, as they either rely on unrealistic prior knowledge or cause accuracy loss even in the absence of attacks. Furthermore, these methods lack a protocol for verifying execution, leaving participants uncertain about the correct execution of the mechanism. To address these challenges, we propose a novel anomaly detection strategy that is designed for real-world FL systems. Our approach activates the defense only when potential attacks are detected, and enables the removal of malicious models without affecting the benign ones. Additionally, we incorporate zero-knowledge proofs to ensure the integrity of the proposed defense mechanism. Experimental results demonstrate the effectiveness of our approach in enhancing FL system security against a comprehensive set of adversarial attacks in various ML tasks.

摘要: 联邦学习(FL)系统容易受到敌意攻击，恶意客户端提交有毒模型来扰乱收敛或植入后门，导致全局模型对某些样本进行错误分类。目前的防御方法对于真实的FL系统来说往往是不切实际的，因为它们要么依赖不切实际的先验知识，要么即使在没有攻击的情况下也会造成准确性损失。此外，这些方法缺乏验证执行的协议，使得参与者不确定该机制的正确执行。为了应对这些挑战，我们提出了一种新的异常检测策略，该策略是为现实世界的FL系统设计的。我们的方法只有在检测到潜在攻击时才会激活防御，并且可以在不影响良性模型的情况下删除恶意模型。此外，我们加入了零知识证明，以确保所提出的防御机制的完整性。实验结果表明，该方法能有效地提高FL系统在各种ML任务中抵抗各种敌意攻击的安全性。



## **36. Aligning LLMs to Be Robust Against Prompt Injection**

调整LLM以应对即时注入的稳健性 cs.CR

Key words: prompt injection defense, LLM security, LLM-integrated  applications. Alignment training makes LLMs robust against even the strongest  prompt injection attacks

**SubmitDate**: 2024-10-07    [abs](http://arxiv.org/abs/2410.05451v1) [paper-pdf](http://arxiv.org/pdf/2410.05451v1)

**Authors**: Sizhe Chen, Arman Zharmagambetov, Saeed Mahloujifar, Kamalika Chaudhuri, Chuan Guo

**Abstract**: Large language models (LLMs) are becoming increasingly prevalent in modern software systems, interfacing between the user and the internet to assist with tasks that require advanced language understanding. To accomplish these tasks, the LLM often uses external data sources such as user documents, web retrieval, results from API calls, etc. This opens up new avenues for attackers to manipulate the LLM via prompt injection. Adversarial prompts can be carefully crafted and injected into external data sources to override the user's intended instruction and instead execute a malicious instruction. Prompt injection attacks constitute a major threat to LLM security, making the design and implementation of practical countermeasures of paramount importance. To this end, we show that alignment can be a powerful tool to make LLMs more robust against prompt injection. Our method -- SecAlign -- first builds an alignment dataset by simulating prompt injection attacks and constructing pairs of desirable and undesirable responses. Then, we apply existing alignment techniques to fine-tune the LLM to be robust against these simulated attacks. Our experiments show that SecAlign robustifies the LLM substantially with a negligible hurt on model utility. Moreover, SecAlign's protection generalizes to strong attacks unseen in training. Specifically, the success rate of state-of-the-art GCG-based prompt injections drops from 56% to 2% in Mistral-7B after our alignment process. Our code is released at https://github.com/facebookresearch/SecAlign

摘要: 大型语言模型(LLM)在现代软件系统中正变得越来越普遍，它们在用户和互联网之间进行接口，以帮助完成需要高级语言理解的任务。为了完成这些任务，LLM通常使用外部数据源，如用户文档、Web检索、API调用结果等。这为攻击者通过提示注入操纵LLM开辟了新的途径。恶意提示可以精心编制并注入外部数据源，以覆盖用户的预期指令，而不是执行恶意指令。快速注入攻击是对LLM安全的主要威胁，因此设计和实施实用的对策至关重要。为此，我们表明对齐可以是一个强大的工具，以使LLM更健壮地抵御快速注入。我们的方法--SecAlign--首先通过模拟即时注入攻击并构建期望和不期望的响应对来构建比对数据集。然后，我们应用现有的对准技术来微调LLM，使其对这些模拟攻击具有健壮性。我们的实验表明，SecAlign在很大程度上增强了LLM的健壮性，而对模型效用的损害可以忽略不计。此外，SecAlign的保护可以概括为在训练中看不到的强大攻击。具体地说，在我们的对准过程之后，米斯特拉尔-7B最先进的基于GCG的快速注射的成功率从56%下降到2%。我们的代码在https://github.com/facebookresearch/SecAlign上发布



## **37. STOP! Camera Spoofing via the in-Vehicle IP Network**

停下来！通过车载IP网络进行摄像头欺骗 cs.CR

**SubmitDate**: 2024-10-07    [abs](http://arxiv.org/abs/2410.05417v1) [paper-pdf](http://arxiv.org/pdf/2410.05417v1)

**Authors**: Dror Peri, Avishai Wool

**Abstract**: Autonomous driving and advanced driver assistance systems (ADAS) rely on cameras to control the driving. In many prior approaches an attacker aiming to stop the vehicle had to send messages on the specialized and better-defended CAN bus. We suggest an easier alternative: manipulate the IP-based network communication between the camera and the ADAS logic, inject fake images of stop signs or red lights into the video stream, and let the ADAS stop the car safely. We created an attack tool that successfully exploits the GigE Vision protocol.   Then we analyze two classes of passive anomaly detectors to identify such attacks: protocol-based detectors and video-based detectors. We implemented multiple detectors of both classes and evaluated them on data collected from our test vehicle and also on data from the public BDD corpus. Our results show that such detectors are effective against naive adversaries, but sophisticated adversaries can evade detection.   Finally, we propose a novel class of active defense mechanisms that randomly adjust camera parameters during the video transmission, and verify that the received images obey the requested adjustments. Within this class we focus on a specific implementation, the width-varying defense, which randomly modifies the width of every frame. Beyond its function as an anomaly detector, this defense is also a protective measure against certain attacks: by distorting injected image patches it prevents their recognition by the ADAS logic. We demonstrate the effectiveness of the width-varying defense through theoretical analysis and by an extensive evaluation of several types of attack in a wide range of realistic road driving conditions. The best the attack was able to achieve against this defense was injecting a stop sign for a duration of 0.2 seconds, with a success probability of 0.2%, whereas stopping a vehicle requires about 2.5 seconds.

摘要: 自动驾驶和先进的驾驶员辅助系统(ADA)依靠摄像头来控制驾驶。在许多以前的方法中，攻击者要想阻止车辆，必须在专门的、防御更好的CAN总线上发送消息。我们建议一个更简单的替代方案：操纵摄像头和ADAS逻辑之间基于IP的网络通信，在视频流中注入停车标志或红灯的虚假图像，让ADAS安全停车。我们创建了一个成功利用GigE Vision协议的攻击工具。然后分析了两类被动异常检测器来识别此类攻击：基于协议的检测器和基于视频的检测器。我们实现了这两种类型的多个检测器，并根据从我们的测试车辆收集的数据以及来自公共BDD语料库的数据对它们进行了评估。我们的结果表明，这种检测器对幼稚的对手是有效的，但经验丰富的对手可以躲避检测。最后，我们提出了一种新的主动防御机制，在视频传输过程中随机调整摄像头参数，并验证接收到的图像是否符合请求的调整。在这个类中，我们将重点介绍一个特定的实现，即宽度可变防御，它随机修改每个帧的宽度。除了作为异常检测器的功能外，这种防御也是针对某些攻击的保护措施：通过扭曲注入的图像补丁，它阻止了ADAS逻辑对它们的识别。我们通过理论分析和在各种实际道路驾驶条件下对几种攻击类型的广泛评估，论证了变宽度防御的有效性。针对这种防御，攻击所能达到的最好效果是注入一个停车标志，持续时间为0.2秒，成功概率为0.2%，而阻止车辆需要大约2.5秒。



## **38. Random-Set Neural Networks (RS-NN)**

随机集神经网络（RS-NN） cs.LG

**SubmitDate**: 2024-10-07    [abs](http://arxiv.org/abs/2307.05772v2) [paper-pdf](http://arxiv.org/pdf/2307.05772v2)

**Authors**: Shireen Kudukkil Manchingal, Muhammad Mubashar, Kaizheng Wang, Keivan Shariatmadar, Fabio Cuzzolin

**Abstract**: Machine learning is increasingly deployed in safety-critical domains where robustness against adversarial attacks is crucial and erroneous predictions could lead to potentially catastrophic consequences. This highlights the need for learning systems to be equipped with the means to determine a model's confidence in its prediction and the epistemic uncertainty associated with it, 'to know when a model does not know'. In this paper, we propose a novel Random-Set Neural Network (RS-NN) for classification. RS-NN predicts belief functions rather than probability vectors over a set of classes using the mathematics of random sets, i.e., distributions over the power set of the sample space. RS-NN encodes the 'epistemic' uncertainty induced in machine learning by limited training sets via the size of the credal sets associated with the predicted belief functions. Our approach outperforms state-of-the-art Bayesian (LB-BNN, BNN-R) and Ensemble (ENN) methods in a classical evaluation setting in terms of performance, uncertainty estimation and out-of-distribution (OoD) detection on several benchmarks (CIFAR-10 vs SVHN/Intel-Image, MNIST vs FMNIST/KMNIST, ImageNet vs ImageNet-O) and scales effectively to large-scale architectures such as WideResNet-28-10, VGG16, Inception V3, EfficientNetB2, and ViT-Base.

摘要: 机器学习越来越多地部署在安全关键领域，在这些领域，对对手攻击的健壮性至关重要，错误的预测可能会导致潜在的灾难性后果。这突显了学习系统需要配备这样的手段，以确定模型对其预测的信心以及与之相关的认知不确定性，即“知道模型何时不知道”。本文提出了一种新的随机集神经网络(RS-NN)用于分类。RS-NN使用随机集的数学，即样本空间的功率集上的分布，来预测一组类别上的信任函数而不是概率向量。RS-NN通过与预测的信任函数相关联的信任度集合的大小来编码机器学习中有限训练集所引起的认知不确定性。在几个基准测试(CIFAR-10与SVHN/Intel-Image、MNIST与FMNIST/KMNIST、ImageNet与ImageNet-O)上，我们的方法在经典评估环境下的性能、不确定性估计和不分布(OOD)检测方面优于最新的贝叶斯(LB-BNN，BNN-R)和集成(ENN)方法，并有效地扩展到大规模体系结构，如WideResNet-28-10、VGG16、Inrupt V3、EfficientNetB2和Vit-Base。



## **39. $$\mathbf{L^2\cdot M = C^2}$$ Large Language Models are Covert Channels**

$$\mathBF{L ' 2\csot M = C ' 2}$$大型语言模型是秘密渠道 cs.CR

**SubmitDate**: 2024-10-07    [abs](http://arxiv.org/abs/2405.15652v2) [paper-pdf](http://arxiv.org/pdf/2405.15652v2)

**Authors**: Simen Gaure, Stefanos Koffas, Stjepan Picek, Sondre Rønjom

**Abstract**: Large Language Models (LLMs) have gained significant popularity recently. LLMs are susceptible to various attacks but can also improve the security of diverse systems. However, besides enabling more secure systems, how well do open source LLMs behave as covertext distributions to, e.g., facilitate censorship-resistant communication? In this paper, we explore open-source LLM-based covert channels. We empirically measure the security vs. capacity of an open-source LLM model (Llama-7B) to assess its performance as a covert channel. Although our results indicate that such channels are not likely to achieve high practical bitrates, we also show that the chance for an adversary to detect covert communication is low. To ensure our results can be used with the least effort as a general reference, we employ a conceptually simple and concise scheme and only assume public models.

摘要: 大型语言模型（LLM）最近受到广泛欢迎。LLM容易受到各种攻击，但也可以提高不同系统的安全性。然而，除了支持更安全的系统之外，开源LLM作为covertext分发版的表现如何，例如促进抗审查的沟通？在本文中，我们探索基于开源LLM的隐蔽渠道。我们根据经验测量开源LLM模型（Llama-7 B）的安全性与容量，以评估其作为隐蔽渠道的性能。尽管我们的结果表明此类通道不太可能实现高的实际比特率，但我们也表明对手检测秘密通信的机会很低。为了确保我们的结果可以以最少的努力作为一般参考，我们采用了概念上简单且简洁的方案，并且仅假设公共模型。



## **40. Jailbreaking Leading Safety-Aligned LLMs with Simple Adaptive Attacks**

通过简单的自适应攻击越狱领先的安全一致LLM cs.CR

Updates in the v3: GPT-4o and Claude 3.5 Sonnet results, improved  writing. Updates in the v2: more models (Llama3, Phi-3, Nemotron-4-340B),  jailbreak artifacts for all attacks are available, evaluation with different  judges (Llama-3-70B and Llama Guard 2), more experiments (convergence plots  over iterations, ablation on the suffix length for random search), examples  of jailbroken generation

**SubmitDate**: 2024-10-07    [abs](http://arxiv.org/abs/2404.02151v3) [paper-pdf](http://arxiv.org/pdf/2404.02151v3)

**Authors**: Maksym Andriushchenko, Francesco Croce, Nicolas Flammarion

**Abstract**: We show that even the most recent safety-aligned LLMs are not robust to simple adaptive jailbreaking attacks. First, we demonstrate how to successfully leverage access to logprobs for jailbreaking: we initially design an adversarial prompt template (sometimes adapted to the target LLM), and then we apply random search on a suffix to maximize a target logprob (e.g., of the token "Sure"), potentially with multiple restarts. In this way, we achieve 100% attack success rate -- according to GPT-4 as a judge -- on Vicuna-13B, Mistral-7B, Phi-3-Mini, Nemotron-4-340B, Llama-2-Chat-7B/13B/70B, Llama-3-Instruct-8B, Gemma-7B, GPT-3.5, GPT-4o, and R2D2 from HarmBench that was adversarially trained against the GCG attack. We also show how to jailbreak all Claude models -- that do not expose logprobs -- via either a transfer or prefilling attack with a 100% success rate. In addition, we show how to use random search on a restricted set of tokens for finding trojan strings in poisoned models -- a task that shares many similarities with jailbreaking -- which is the algorithm that brought us the first place in the SaTML'24 Trojan Detection Competition. The common theme behind these attacks is that adaptivity is crucial: different models are vulnerable to different prompting templates (e.g., R2D2 is very sensitive to in-context learning prompts), some models have unique vulnerabilities based on their APIs (e.g., prefilling for Claude), and in some settings, it is crucial to restrict the token search space based on prior knowledge (e.g., for trojan detection). For reproducibility purposes, we provide the code, logs, and jailbreak artifacts in the JailbreakBench format at https://github.com/tml-epfl/llm-adaptive-attacks.

摘要: 我们表明，即使是最新的安全对齐的LLM也不能抵抗简单的自适应越狱攻击。首先，我们演示了如何成功地利用对Logpros的访问来越狱：我们最初设计了一个对抗性提示模板(有时适用于目标LLM)，然后在后缀上应用随机搜索来最大化目标Logprob(例如，令牌“Sure”)，可能需要多次重新启动。通过这种方式，我们实现了100%的攻击成功率--根据GPT-4作为评委--对来自哈姆班奇的维古纳-13B、西北风-7B、Phi-3-Mini、Nemotron-4-340B、Llama-2-Chat-7B/13B/70B、Llama-3-Indict-8B、Gema-7B、GPT-3.5、GPT-4o和R2D2进行了对抗GCG攻击的对抗训练。我们还展示了如何通过传输或预填充攻击以100%的成功率越狱所有不暴露日志问题的Claude模型。此外，我们还展示了如何在受限的令牌集合上使用随机搜索来查找有毒模型中的特洛伊木马字符串--这项任务与越狱有许多相似之处--正是这种算法为我们带来了SATML‘24特洛伊木马检测大赛的第一名。这些攻击背后的共同主题是自适应至关重要：不同的模型容易受到不同提示模板的影响(例如，R2D2对上下文中的学习提示非常敏感)，一些模型基于其API具有独特的漏洞(例如，预填充Claude)，并且在某些设置中，基于先验知识限制令牌搜索空间至关重要(例如，对于木马检测)。出于可重现性的目的，我们在https://github.com/tml-epfl/llm-adaptive-attacks.上以JailBreak B边格式提供代码、日志和越狱构件



## **41. Dr. Jekyll and Mr. Hyde: Two Faces of LLMs**

杰基尔博士和海德先生：法学硕士的两面 cs.CR

**SubmitDate**: 2024-10-07    [abs](http://arxiv.org/abs/2312.03853v5) [paper-pdf](http://arxiv.org/pdf/2312.03853v5)

**Authors**: Matteo Gioele Collu, Tom Janssen-Groesbeek, Stefanos Koffas, Mauro Conti, Stjepan Picek

**Abstract**: Recently, we have witnessed a rise in the use of Large Language Models (LLMs), especially in applications like chatbots. Safety mechanisms are implemented to prevent improper responses from these chatbots. In this work, we bypass these measures for ChatGPT and Gemini by making them impersonate complex personas with personality characteristics that are not aligned with a truthful assistant. First, we create elaborate biographies of these personas, which we then use in a new session with the same chatbots. Our conversations then follow a role-play style to elicit prohibited responses. Using personas, we show that prohibited responses are provided, making it possible to obtain unauthorized, illegal, or harmful information in both ChatGPT and Gemini. We also introduce several ways of activating such adversarial personas, showing that both chatbots are vulnerable to this attack. With the same principle, we introduce two defenses that push the model to interpret trustworthy personalities and make it more robust against such attacks.

摘要: 最近，我们看到大型语言模型(LLM)的使用有所增加，特别是在聊天机器人等应用程序中。安全机制的实施是为了防止这些聊天机器人做出不适当的反应。在这项工作中，我们绕过了ChatGPT和Gemini的这些措施，让他们模仿具有与诚实的助手不一致的人格特征的复杂人物角色。首先，我们为这些角色创建详细的传记，然后在与相同聊天机器人的新会话中使用这些传记。然后，我们的对话遵循角色扮演的风格，以引发被禁止的回应。使用人物角色，我们显示提供了禁止的响应，使得在ChatGPT和Gemini中获取未经授权的、非法的或有害的信息成为可能。我们还介绍了几种激活这种敌对角色的方法，表明这两个聊天机器人都容易受到这种攻击。在相同的原则下，我们引入了两个防御措施，推动该模型解释可信任的个性，并使其对此类攻击更加健壮。



## **42. LOTOS: Layer-wise Orthogonalization for Training Robust Ensembles**

LOTOS：分层国家化，用于训练强大的合奏 cs.LG

**SubmitDate**: 2024-10-07    [abs](http://arxiv.org/abs/2410.05136v1) [paper-pdf](http://arxiv.org/pdf/2410.05136v1)

**Authors**: Ali Ebrahimpour-Boroojeny, Hari Sundaram, Varun Chandrasekaran

**Abstract**: Transferability of adversarial examples is a well-known property that endangers all classification models, even those that are only accessible through black-box queries. Prior work has shown that an ensemble of models is more resilient to transferability: the probability that an adversarial example is effective against most models of the ensemble is low. Thus, most ongoing research focuses on improving ensemble diversity. Another line of prior work has shown that Lipschitz continuity of the models can make models more robust since it limits how a model's output changes with small input perturbations. In this paper, we study the effect of Lipschitz continuity on transferability rates. We show that although a lower Lipschitz constant increases the robustness of a single model, it is not as beneficial in training robust ensembles as it increases the transferability rate of adversarial examples across models in the ensemble. Therefore, we introduce LOTOS, a new training paradigm for ensembles, which counteracts this adverse effect. It does so by promoting orthogonality among the top-$k$ sub-spaces of the transformations of the corresponding affine layers of any pair of models in the ensemble. We theoretically show that $k$ does not need to be large for convolutional layers, which makes the computational overhead negligible. Through various experiments, we show LOTOS increases the robust accuracy of ensembles of ResNet-18 models by $6$ percentage points (p.p) against black-box attacks on CIFAR-10. It is also capable of combining with the robustness of prior state-of-the-art methods for training robust ensembles to enhance their robust accuracy by $10.7$ p.p.

摘要: 对抗性例子的可转移性是一个众所周知的性质，它危及所有分类模型，即使是那些只能通过黑盒查询访问的分类模型。先前的工作已经表明，模型集合对可转移性具有更强的弹性：对抗性例子对集合中的大多数模型有效的概率很低。因此，大多数正在进行的研究都集中在改善群体多样性上。另一项先前的工作表明，模型的Lipschitz连续性可以使模型更稳健，因为它限制了模型的输出如何在小的输入扰动下变化。在本文中，我们研究了Lipschitz连续性对可转移率的影响。我们表明，虽然较低的Lipschitz常数增加了单个模型的稳健性，但它在训练稳健的集成方面并不是那么有益，因为它增加了对抗性示例在集成中的模型之间的可转移率。因此，我们引入了LOTOS，一种新的系综训练范式，它抵消了这种不利影响。它通过促进系综中任何一对模型的相应仿射层的变换的顶层$k$子空间之间的正交性来做到这一点。我们从理论上证明，对于卷积层，$k$不需要很大，这使得计算开销可以忽略不计。通过各种实验，我们表明LOTOS将ResNet-18模型集成的稳健精度提高了6美元百分点(p.p)，以抵御对CIFAR-10的黑盒攻击。它还能够与先前最先进的训练稳健集合的方法的稳健性相结合，以将其稳健精度提高10.7美元/分。



## **43. A test suite of prompt injection attacks for LLM-based machine translation**

基于LLM的机器翻译的提示注入攻击测试套件 cs.CL

**SubmitDate**: 2024-10-07    [abs](http://arxiv.org/abs/2410.05047v1) [paper-pdf](http://arxiv.org/pdf/2410.05047v1)

**Authors**: Antonio Valerio Miceli-Barone, Zhifan Sun

**Abstract**: LLM-based NLP systems typically work by embedding their input data into prompt templates which contain instructions and/or in-context examples, creating queries which are submitted to a LLM, and then parsing the LLM response in order to generate the system outputs. Prompt Injection Attacks (PIAs) are a type of subversion of these systems where a malicious user crafts special inputs which interfere with the prompt templates, causing the LLM to respond in ways unintended by the system designer.   Recently, Sun and Miceli-Barone proposed a class of PIAs against LLM-based machine translation. Specifically, the task is to translate questions from the TruthfulQA test suite, where an adversarial prompt is prepended to the questions, instructing the system to ignore the translation instruction and answer the questions instead.   In this test suite, we extend this approach to all the language pairs of the WMT 2024 General Machine Translation task. Moreover, we include additional attack formats in addition to the one originally studied.

摘要: 基于LLM的NLP系统通常通过将其输入数据嵌入到包含指令和/或上下文中示例的提示模板中、创建提交给LLM的查询、然后解析LLM响应以生成系统输出来工作。提示注入攻击(PIA)是这些系统的一种颠覆类型，恶意用户精心编制干扰提示模板的特殊输入，导致LLM以系统设计者意想不到的方式做出响应。最近，Sun和Miceli-Barone提出了一类针对基于LLM的机器翻译的PIA。具体地说，任务是翻译TruthfulQA测试套件中的问题，在该测试套件中，问题前面会有一个对抗性提示，指示系统忽略翻译说明，转而回答问题。在这个测试套件中，我们将这种方法扩展到WMT 2024通用机器翻译任务的所有语言对。此外，除了最初研究的一种攻击形式之外，我们还包括其他攻击形式。



## **44. Collaboration! Towards Robust Neural Methods for Routing Problems**

合作！路由问题的鲁棒神经方法 cs.AI

Accepted at NeurIPS 2024

**SubmitDate**: 2024-10-07    [abs](http://arxiv.org/abs/2410.04968v1) [paper-pdf](http://arxiv.org/pdf/2410.04968v1)

**Authors**: Jianan Zhou, Yaoxin Wu, Zhiguang Cao, Wen Song, Jie Zhang, Zhiqi Shen

**Abstract**: Despite enjoying desirable efficiency and reduced reliance on domain expertise, existing neural methods for vehicle routing problems (VRPs) suffer from severe robustness issues -- their performance significantly deteriorates on clean instances with crafted perturbations. To enhance robustness, we propose an ensemble-based Collaborative Neural Framework (CNF) w.r.t. the defense of neural VRP methods, which is crucial yet underexplored in the literature. Given a neural VRP method, we adversarially train multiple models in a collaborative manner to synergistically promote robustness against attacks, while boosting standard generalization on clean instances. A neural router is designed to adeptly distribute training instances among models, enhancing overall load balancing and collaborative efficacy. Extensive experiments verify the effectiveness and versatility of CNF in defending against various attacks across different neural VRP methods. Notably, our approach also achieves impressive out-of-distribution generalization on benchmark instances.

摘要: 尽管具有理想的效率并减少了对领域专业知识的依赖，但解决车辆路径问题(VRP)的现有神经方法存在严重的健壮性问题--在带有精心设计的扰动的干净实例上，它们的性能会显著恶化。为了增强鲁棒性，我们提出了一种基于集成的协作神经框架(CNF)w.r.t.神经VRP方法的防御，这是至关重要的，但在文献中探索得很少。给出了一种神经VRP方法，相反，我们以协作的方式训练多个模型，以协同提高对攻击的健壮性，同时提高对干净实例的标准泛化。设计了一个神经路由器来巧妙地在模型之间分配训练实例，增强了整体负载平衡和协作效率。大量实验验证了CNF在抵抗不同神经VRP方法攻击方面的有效性和通用性。值得注意的是，我们的方法还在基准实例上实现了令人印象深刻的分布外泛化。



## **45. Reconstruct Your Previous Conversations! Comprehensively Investigating Privacy Leakage Risks in Conversations with GPT Models**

重建您之前的对话！全面调查GPT模型对话中的隐私泄露风险 cs.CR

Accepted in EMNLP 2024. 14 pages, 10 figures

**SubmitDate**: 2024-10-07    [abs](http://arxiv.org/abs/2402.02987v2) [paper-pdf](http://arxiv.org/pdf/2402.02987v2)

**Authors**: Junjie Chu, Zeyang Sha, Michael Backes, Yang Zhang

**Abstract**: Significant advancements have recently been made in large language models represented by GPT models. Users frequently have multi-round private conversations with cloud-hosted GPT models for task optimization. Yet, this operational paradigm introduces additional attack surfaces, particularly in custom GPTs and hijacked chat sessions. In this paper, we introduce a straightforward yet potent Conversation Reconstruction Attack. This attack targets the contents of previous conversations between GPT models and benign users, i.e., the benign users' input contents during their interaction with GPT models. The adversary could induce GPT models to leak such contents by querying them with designed malicious prompts. Our comprehensive examination of privacy risks during the interactions with GPT models under this attack reveals GPT-4's considerable resilience. We present two advanced attacks targeting improved reconstruction of past conversations, demonstrating significant privacy leakage across all models under these advanced techniques. Evaluating various defense mechanisms, we find them ineffective against these attacks. Our findings highlight the ease with which privacy can be compromised in interactions with GPT models, urging the community to safeguard against potential abuses of these models' capabilities.

摘要: 最近，以GPT模型为代表的大型语言模型取得了重大进展。用户经常与云托管的GPT模型进行多轮私下对话，以实现任务优化。然而，这种操作模式引入了额外的攻击面，特别是在定制GPT和被劫持的聊天会话中。在本文中，我们介绍了一种简单而有效的会话重建攻击。该攻击的目标是GPT模型与良性用户之间先前对话的内容，即良性用户在与GPT模型交互时的输入内容。攻击者可以通过设计恶意提示来查询GPT模型，从而诱导GPT模型泄露此类内容。我们对在这次攻击下与GPT模型交互过程中的隐私风险进行了全面的检查，发现GPT-4的S具有相当强的韧性。我们提出了两个高级攻击，目标是改进过去对话的重建，表明在这些高级技术下，所有模型都存在显著的隐私泄露。评估各种防御机制，我们发现它们对这些攻击无效。我们的发现突出了隐私在与GPT模型的交互中很容易受到损害，敦促社区防范这些模型的功能可能被滥用。



## **46. ResTNet: Defense against Adversarial Policies via Transformer in Computer Go**

ResTNet：通过Computer Go中的Transformer防御对抗性政策 cs.LG

**SubmitDate**: 2024-10-07    [abs](http://arxiv.org/abs/2410.05347v1) [paper-pdf](http://arxiv.org/pdf/2410.05347v1)

**Authors**: Tai-Lin Wu, Ti-Rong Wu, Chung-Chin Shih, Yan-Ru Ju, I-Chen Wu

**Abstract**: Although AlphaZero has achieved superhuman levels in Go, recent research has highlighted its vulnerability in particular situations requiring a more comprehensive understanding of the entire board. To address this challenge, this paper introduces ResTNet, a network that interleaves residual networks and Transformer. Our empirical experiments demonstrate several advantages of using ResTNet. First, it not only improves playing strength but also enhances the ability of global information. Second, it defends against an adversary Go program, called cyclic-adversary, tailor-made for attacking AlphaZero algorithms, significantly reducing the average probability of being attacked rate from 70.44% to 23.91%. Third, it improves the accuracy from 59.15% to 80.01% in correctly recognizing ladder patterns, which are one of the challenging patterns for Go AIs. Finally, ResTNet offers a potential explanation of the decision-making process and can also be applied to other games like Hex. To the best of our knowledge, ResTNet is the first to integrate residual networks and Transformer in the context of AlphaZero for board games, suggesting a promising direction for enhancing AlphaZero's global understanding.

摘要: 尽管AlphaZero在围棋中达到了超人的水平，但最近的研究突显了它在需要对整个棋盘进行更全面了解的特殊情况下的脆弱性。为了应对这一挑战，本文引入了ResTNet，这是一种将剩余网络和Transformer交织在一起的网络。我们的实证实验证明了使用ResTNet的几个优势。首先，它不仅提高了打球实力，还增强了全球信息能力。其次，它防御了一种名为循环对手的围棋程序，该程序专为攻击AlphaZero算法而定制，显著降低了平均被攻击率从70.44%降至23.91%。第三，对围棋人工智能中具有挑战性的梯形图案的正确识别正确率从59.15%提高到80.01%。最后，ResTNet对决策过程提供了一个潜在的解释，也可以应用于其他游戏，如Hex。据我们所知，ResTNet是第一个在AlphaZero的棋盘游戏环境中整合剩余网络和Transformer的公司，这为增强AlphaZero的全球理解提供了一个很有前途的方向。



## **47. Patch is Enough: Naturalistic Adversarial Patch against Vision-Language Pre-training Models**

补丁就足够了：针对视觉语言预训练模型的自然主义对抗补丁 cs.CV

accepted by Visual Intelligence

**SubmitDate**: 2024-10-07    [abs](http://arxiv.org/abs/2410.04884v1) [paper-pdf](http://arxiv.org/pdf/2410.04884v1)

**Authors**: Dehong Kong, Siyuan Liang, Xiaopeng Zhu, Yuansheng Zhong, Wenqi Ren

**Abstract**: Visual language pre-training (VLP) models have demonstrated significant success across various domains, yet they remain vulnerable to adversarial attacks. Addressing these adversarial vulnerabilities is crucial for enhancing security in multimodal learning. Traditionally, adversarial methods targeting VLP models involve simultaneously perturbing images and text. However, this approach faces notable challenges: first, adversarial perturbations often fail to translate effectively into real-world scenarios; second, direct modifications to the text are conspicuously visible. To overcome these limitations, we propose a novel strategy that exclusively employs image patches for attacks, thus preserving the integrity of the original text. Our method leverages prior knowledge from diffusion models to enhance the authenticity and naturalness of the perturbations. Moreover, to optimize patch placement and improve the efficacy of our attacks, we utilize the cross-attention mechanism, which encapsulates intermodal interactions by generating attention maps to guide strategic patch placements. Comprehensive experiments conducted in a white-box setting for image-to-text scenarios reveal that our proposed method significantly outperforms existing techniques, achieving a 100% attack success rate. Additionally, it demonstrates commendable performance in transfer tasks involving text-to-image configurations.

摘要: 视觉语言预训练(VLP)模型在各个领域都取得了显著的成功，但它们仍然容易受到对手的攻击。解决这些对抗性漏洞对于增强多模式学习的安全性至关重要。传统上，针对VLP模型的对抗性方法涉及同时扰动图像和文本。然而，这种方法面临着显著的挑战：首先，对抗性的扰动往往无法有效地转化为现实世界的情景；其次，对文本的直接修改是显而易见的。为了克服这些局限性，我们提出了一种新的策略，该策略专门使用图像补丁进行攻击，从而保持了原始文本的完整性。我们的方法利用扩散模型的先验知识来增强扰动的真实性和自然性。此外，为了优化补丁放置和提高攻击的效率，我们利用交叉注意机制，通过生成注意图来指导策略性补丁放置来封装不同通道之间的交互。在白盒环境下对图文转换场景进行的综合实验表明，该方法的性能明显优于现有技术，攻击成功率达到100%。此外，它在涉及文本到图像配置的传输任务中表现出了值得称赞的性能。



## **48. AnyAttack: Towards Large-scale Self-supervised Generation of Targeted Adversarial Examples for Vision-Language Models**

AnyAttack：面向视觉语言模型的大规模自我监督生成有针对性的对抗示例 cs.LG

**SubmitDate**: 2024-10-07    [abs](http://arxiv.org/abs/2410.05346v1) [paper-pdf](http://arxiv.org/pdf/2410.05346v1)

**Authors**: Jiaming Zhang, Junhong Ye, Xingjun Ma, Yige Li, Yunfan Yang, Jitao Sang, Dit-Yan Yeung

**Abstract**: Due to their multimodal capabilities, Vision-Language Models (VLMs) have found numerous impactful applications in real-world scenarios. However, recent studies have revealed that VLMs are vulnerable to image-based adversarial attacks, particularly targeted adversarial images that manipulate the model to generate harmful content specified by the adversary. Current attack methods rely on predefined target labels to create targeted adversarial attacks, which limits their scalability and applicability for large-scale robustness evaluations. In this paper, we propose AnyAttack, a self-supervised framework that generates targeted adversarial images for VLMs without label supervision, allowing any image to serve as a target for the attack. To address the limitation of existing methods that require label supervision, we introduce a contrastive loss that trains a generator on a large-scale unlabeled image dataset, LAION-400M dataset, for generating targeted adversarial noise. This large-scale pre-training endows our method with powerful transferability across a wide range of VLMs. Extensive experiments on five mainstream open-source VLMs (CLIP, BLIP, BLIP2, InstructBLIP, and MiniGPT-4) across three multimodal tasks (image-text retrieval, multimodal classification, and image captioning) demonstrate the effectiveness of our attack. Additionally, we successfully transfer AnyAttack to multiple commercial VLMs, including Google's Gemini, Claude's Sonnet, and Microsoft's Copilot. These results reveal an unprecedented risk to VLMs, highlighting the need for effective countermeasures.

摘要: 由于其多通道能力，视觉语言模型(VLM)在现实世界场景中发现了许多有影响力的应用。然而，最近的研究表明，VLM很容易受到基于图像的敌意攻击，特别是针对操纵模型以生成对手指定的有害内容的对抗性图像。当前的攻击方法依赖于预定义的目标标签来创建有针对性的对抗性攻击，这限制了它们在大规模健壮性评估中的可扩展性和适用性。在本文中，我们提出了AnyAttack，这是一个自监督框架，可以在没有标签监督的情况下为VLMS生成有针对性的敌意图像，允许任何图像作为攻击的目标。为了解决现有方法需要标签监督的局限性，我们引入了一种对比损失，它在大规模的未标记图像数据集LAION-400M数据集上训练生成器，以生成目标对抗性噪声。这种大规模的预培训使我们的方法在广泛的VLM中具有强大的可移植性。在三个多模式任务(图像-文本检索、多模式分类和图像字幕)上对五个主流开源VLMS(CLIP、BLIP、BLIP2、InstructBLIP和MiniGPT-4)进行了广泛的实验，证明了该攻击的有效性。此外，我们成功地将AnyAttack转移到多个商业VLM上，包括谷歌的Gemini、Claude的十四行诗和微软的Copilot。这些结果揭示了极小武器系统面临的前所未有的风险，突显了采取有效对策的必要性。



## **49. A Differentially Private Energy Trading Mechanism Approaching Social Optimum**

差异化的私人能源交易机制接近社会最优 cs.GT

11 pages, 8 figures

**SubmitDate**: 2024-10-07    [abs](http://arxiv.org/abs/2410.04787v1) [paper-pdf](http://arxiv.org/pdf/2410.04787v1)

**Authors**: Yuji Cao, Yue Chen

**Abstract**: This paper proposes a differentially private energy trading mechanism for prosumers in peer-to-peer (P2P) markets, offering provable privacy guarantees while approaching the Nash equilibrium with nearly socially optimal efficiency. We first model the P2P energy trading as a (generalized) Nash game and prove the vulnerability of traditional distributed algorithms to privacy attacks through an adversarial inference model. To address this challenge, we develop a privacy-preserving Nash equilibrium seeking algorithm incorporating carefully calibrated Laplacian noise. We prove that the proposed algorithm achieves $\epsilon$-differential privacy while converging in expectation to the Nash equilibrium with a suitable stepsize. Numerical experiments are conducted to evaluate the algorithm's robustness against privacy attacks, convergence behavior, and optimality compared to the non-private solution. Results demonstrate that our mechanism effectively protects prosumers' sensitive information while maintaining near-optimal market outcomes, offering a practical approach for privacy-preserving coordination in P2P markets.

摘要: 提出了一种在P2P市场中为消费者提供差异化私有能源交易机制，在提供可证明的隐私保证的同时，以接近社会最优效率的方式逼近纳什均衡。我们首先将P2P能量交易建模为(广义)Nash博弈，并通过一个对抗性推理模型证明了传统分布式算法对隐私攻击的脆弱性。为了应对这一挑战，我们开发了一种隐私保护的纳什均衡搜索算法，其中包含了仔细校准的拉普拉斯噪声。我们证明了该算法在以适当的步长期望收敛到纳什均衡的同时，实现了$epsilon$-差分隐私保护。与非私有解相比，数值实验评估了该算法对隐私攻击的健壮性、收敛行为和最优性。结果表明，该机制有效地保护了消费者的敏感信息，同时保持了接近最优的市场结果，为P2P市场中的隐私保护协调提供了一种实用的方法。



## **50. Double Oracle Neural Architecture Search for Game Theoretic Deep Learning Models**

双Oracle神经架构搜索博弈论深度学习模型 cs.LG

**SubmitDate**: 2024-10-07    [abs](http://arxiv.org/abs/2410.04764v1) [paper-pdf](http://arxiv.org/pdf/2410.04764v1)

**Authors**: Aye Phyu Phyu Aung, Xinrun Wang, Ruiyu Wang, Hau Chan, Bo An, Xiaoli Li, J. Senthilnath

**Abstract**: In this paper, we propose a new approach to train deep learning models using game theory concepts including Generative Adversarial Networks (GANs) and Adversarial Training (AT) where we deploy a double-oracle framework using best response oracles. GAN is essentially a two-player zero-sum game between the generator and the discriminator. The same concept can be applied to AT with attacker and classifier as players. Training these models is challenging as a pure Nash equilibrium may not exist and even finding the mixed Nash equilibrium is difficult as training algorithms for both GAN and AT have a large-scale strategy space. Extending our preliminary model DO-GAN, we propose the methods to apply the double oracle framework concept to Adversarial Neural Architecture Search (NAS for GAN) and Adversarial Training (NAS for AT) algorithms. We first generalize the players' strategies as the trained models of generator and discriminator from the best response oracles. We then compute the meta-strategies using a linear program. For scalability of the framework where multiple network models of best responses are stored in the memory, we prune the weakly-dominated players' strategies to keep the oracles from becoming intractable. Finally, we conduct experiments on MNIST, CIFAR-10 and TinyImageNet for DONAS-GAN. We also evaluate the robustness under FGSM and PGD attacks on CIFAR-10, SVHN and TinyImageNet for DONAS-AT. We show that all our variants have significant improvements in both subjective qualitative evaluation and quantitative metrics, compared with their respective base architectures.

摘要: 在本文中，我们提出了一种新的方法来训练深度学习模型，该方法使用博弈论的概念，包括生成性对抗性网络(GANS)和对抗性训练(AT)，其中我们使用最佳响应预言来部署双先知框架。GAN本质上是生产者和鉴别者之间的两人零和博弈。同样的概念也适用于以进攻者和分类者为球员的AT。训练这些模型是具有挑战性的，因为纯Nash均衡可能不存在，甚至很难找到混合Nash均衡，因为GAN和AT的训练算法都具有大规模的策略空间。对我们的初步模型DO-GAN进行了扩展，提出了将双先知框架概念应用于对抗性神经结构搜索(NAS For GAN)和对抗性训练(Aversative Training)算法的方法。我们首先将玩家的策略概括为最佳响应预言中的生成器和鉴别器的训练模型。然后，我们使用线性规划来计算元策略。为了提高框架的可扩展性，在内存中存储了多个最佳响应的网络模型，我们修剪了弱主导玩家的策略，以防止先知变得难以处理。最后，我们在MNIST、CIFAR-10和TinyImageNet上对Donas-GaN进行了实验。我们还评估了DNAS-AT在FGSM和PGD攻击下对CIFAR-10、SVHN和TinyImageNet的健壮性。我们表明，与它们各自的基础架构相比，我们的所有变体在主观定性评估和定量度量方面都有显著的改进。



