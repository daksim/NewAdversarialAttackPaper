# Latest Adversarial Attack Papers
**update at 2024-03-25 09:34:51**

翻译来自 https://cloud.tencent.com/document/product/551/15619

## **1. From Hardware Fingerprint to Access Token: Enhancing the Authentication on IoT Devices**

从硬件指纹到访问令牌：增强物联网设备的认证 cs.CR

**SubmitDate**: 2024-03-22    [abs](http://arxiv.org/abs/2403.15271v1) [paper-pdf](http://arxiv.org/pdf/2403.15271v1)

**Authors**: Yue Xiao, Yi He, Xiaoli Zhang, Qian Wang, Renjie Xie, Kun Sun, Ke Xu, Qi Li

**Abstract**: The proliferation of consumer IoT products in our daily lives has raised the need for secure device authentication and access control. Unfortunately, these resource-constrained devices typically use token-based authentication, which is vulnerable to token compromise attacks that allow attackers to impersonate the devices and perform malicious operations by stealing the access token. Using hardware fingerprints to secure their authentication is a promising way to mitigate these threats. However, once attackers have stolen some hardware fingerprints (e.g., via MitM attacks), they can bypass the hardware authentication by training a machine learning model to mimic fingerprints or reusing these fingerprints to craft forge requests.   In this paper, we present MCU-Token, a secure hardware fingerprinting framework for MCU-based IoT devices even if the cryptographic mechanisms (e.g., private keys) are compromised. MCU-Token can be easily integrated with various IoT devices by simply adding a short hardware fingerprint-based token to the existing payload. To prevent the reuse of this token, we propose a message mapping approach that binds the token to a specific request via generating the hardware fingerprints based on the request payload. To defeat the machine learning attacks, we mix the valid fingerprints with poisoning data so that attackers cannot train a usable model with the leaked tokens. MCU-Token can defend against armored adversary who may replay, craft, and offload the requests via MitM or use both hardware (e.g., use identical devices) and software (e.g., machine learning attacks) strategies to mimic the fingerprints. The system evaluation shows that MCU-Token can achieve high accuracy (over 97%) with a low overhead across various IoT devices and application scenarios.

摘要: 消费者物联网产品在我们日常生活中的激增提高了对安全设备身份验证和访问控制的需求。不幸的是，这些资源受限的设备通常使用基于令牌的身份验证，这很容易受到令牌妥协攻击，允许攻击者通过窃取访问令牌来冒充设备并执行恶意操作。使用硬件指纹来保护他们的身份验证是减轻这些威胁的一种有希望的方法。然而，一旦攻击者窃取了一些硬件指纹（例如，通过MitM攻击），他们可以通过训练机器学习模型来模仿指纹或重用这些指纹来制作伪造请求来绕过硬件认证。   在本文中，我们提出了MCU—Token，一个安全的硬件指纹识别框架，用于基于MCU的物联网设备，即使加密机制（例如，私钥）被泄露。MCU—Token可以简单地将基于硬件指纹的短令牌添加到现有有效载荷中，轻松地与各种物联网设备集成。为了防止令牌的重复使用，我们提出了一种消息映射方法，通过基于请求负载生成硬件指纹来将令牌绑定到特定请求。为了击败机器学习攻击，我们将有效指纹与中毒数据混合，这样攻击者就无法用泄漏的令牌训练可用的模型。MCU—Token可以防御装甲对手，他们可以通过MitM重放、制作和卸载请求，或者使用这两种硬件（例如，使用相同的设备）和软件（例如，机器学习攻击）策略来模仿指纹。系统评估表明，MCU—Token可以在各种物联网设备和应用场景中以低开销实现高精度（超过97%）。



## **2. Robust optimization for adversarial learning with finite sample complexity guarantees**

有限样本复杂性保证下的对抗性学习稳健优化 cs.LG

**SubmitDate**: 2024-03-22    [abs](http://arxiv.org/abs/2403.15207v1) [paper-pdf](http://arxiv.org/pdf/2403.15207v1)

**Authors**: André Bertolace, Konstatinos Gatsis, Kostas Margellos

**Abstract**: Decision making and learning in the presence of uncertainty has attracted significant attention in view of the increasing need to achieve robust and reliable operations. In the case where uncertainty stems from the presence of adversarial attacks this need is becoming more prominent. In this paper we focus on linear and nonlinear classification problems and propose a novel adversarial training method for robust classifiers, inspired by Support Vector Machine (SVM) margins. We view robustness under a data driven lens, and derive finite sample complexity bounds for both linear and non-linear classifiers in binary and multi-class scenarios. Notably, our bounds match natural classifiers' complexity. Our algorithm minimizes a worst-case surrogate loss using Linear Programming (LP) and Second Order Cone Programming (SOCP) for linear and non-linear models. Numerical experiments on the benchmark MNIST and CIFAR10 datasets show our approach's comparable performance to state-of-the-art methods, without needing adversarial examples during training. Our work offers a comprehensive framework for enhancing binary linear and non-linear classifier robustness, embedding robustness in learning under the presence of adversaries.

摘要: 鉴于日益需要实现稳健和可靠的业务，在不确定性存在下的决策和学习已引起了极大的关注。在不确定性源于对抗性攻击的存在的情况下，这种需要变得更加突出。本文主要研究线性和非线性分类问题，并在支持向量机（SVM）边际的启发下，提出了一种新的对抗训练方法。我们在数据驱动的镜头下查看鲁棒性，并推导出线性和非线性分类器在二进制和多类场景下的有限样本复杂度界限。值得注意的是，我们的边界符合自然分类器的复杂性。我们的算法使用线性规划（LP）和二阶锥规划（SOCP）线性和非线性模型的最坏情况下的代理损失最小化。在基准MNIST和CIFAR10数据集上的数值实验表明，我们的方法与最先进的方法具有可比性能，而在训练过程中不需要对抗性的例子。我们的工作提供了一个全面的框架来增强二进制线性和非线性分类器的鲁棒性，嵌入学习中的鲁棒性。



## **3. TTPXHunter: Actionable Threat Intelligence Extraction as TTPs from Finished Cyber Threat Reports**

TTPXHunter：从完成的网络威胁报告中提取可操作的威胁情报 cs.CR

Under Review

**SubmitDate**: 2024-03-21    [abs](http://arxiv.org/abs/2403.03267v3) [paper-pdf](http://arxiv.org/pdf/2403.03267v3)

**Authors**: Nanda Rani, Bikash Saha, Vikas Maurya, Sandeep Kumar Shukla

**Abstract**: Understanding the modus operandi of adversaries aids organizations in employing efficient defensive strategies and sharing intelligence in the community. This knowledge is often present in unstructured natural language text within threat analysis reports. A translation tool is needed to interpret the modus operandi explained in the sentences of the threat report and translate it into a structured format. This research introduces a methodology named TTPXHunter for the automated extraction of threat intelligence in terms of Tactics, Techniques, and Procedures (TTPs) from finished cyber threat reports. It leverages cyber domain-specific state-of-the-art natural language processing (NLP) to augment sentences for minority class TTPs and refine pinpointing the TTPs in threat analysis reports significantly. The knowledge of threat intelligence in terms of TTPs is essential for comprehensively understanding cyber threats and enhancing detection and mitigation strategies. We create two datasets: an augmented sentence-TTP dataset of 39,296 samples and a 149 real-world cyber threat intelligence report-to-TTP dataset. Further, we evaluate TTPXHunter on the augmented sentence dataset and the cyber threat reports. The TTPXHunter achieves the highest performance of 92.42% f1-score on the augmented dataset, and it also outperforms existing state-of-the-art solutions in TTP extraction by achieving an f1-score of 97.09% when evaluated over the report dataset. TTPXHunter significantly improves cybersecurity threat intelligence by offering quick, actionable insights into attacker behaviors. This advancement automates threat intelligence analysis, providing a crucial tool for cybersecurity professionals fighting cyber threats.

摘要: 了解对手的作案手法有助于组织采用有效的防御策略并在社区中共享情报。这些知识通常存在于威胁分析报告中的非结构化自然语言文本中。需要一个翻译工具来解释威胁报告句子中解释的工作方式，并将其翻译成结构化的格式。本研究介绍了一种名为TTPXHunter的方法，用于从完成的网络威胁报告中自动提取战术、技术和程序（TTPs）方面的威胁情报。它利用特定于网络领域的最先进的自然语言处理（NLP）来增加少数群体TTPs的句子，并在威胁分析报告中显著细化TTPs的精确定位。在TTPs方面的威胁情报知识对于全面了解网络威胁以及加强检测和缓解策略至关重要。我们创建了两个数据集：一个包含39，296个样本的增强型网络威胁—TTP数据集和一个149个真实世界的网络威胁情报报告—TTP数据集。此外，我们评估TTPXHunter的增强句子数据集和网络威胁报告。TTPXHunter在增强数据集上实现了92.42%的f1得分的最高性能，并且在报告数据集上进行评估时，它还通过实现97.09%的f1得分，超过了现有的最先进的TTP提取解决方案。TTPXHunter通过提供对攻击者行为的快速、可操作的洞察，显著改善了网络安全威胁情报。这一进步使威胁情报分析自动化，为应对网络威胁的网络安全专业人员提供了重要工具。



## **4. Diffusion Attack: Leveraging Stable Diffusion for Naturalistic Image Attacking**

扩散攻击：利用稳定扩散进行自然图像攻击 cs.CV

Accepted to IEEE VRW

**SubmitDate**: 2024-03-21    [abs](http://arxiv.org/abs/2403.14778v1) [paper-pdf](http://arxiv.org/pdf/2403.14778v1)

**Authors**: Qianyu Guo, Jiaming Fu, Yawen Lu, Dongming Gan

**Abstract**: In Virtual Reality (VR), adversarial attack remains a significant security threat. Most deep learning-based methods for physical and digital adversarial attacks focus on enhancing attack performance by crafting adversarial examples that contain large printable distortions that are easy for human observers to identify. However, attackers rarely impose limitations on the naturalness and comfort of the appearance of the generated attack image, resulting in a noticeable and unnatural attack. To address this challenge, we propose a framework to incorporate style transfer to craft adversarial inputs of natural styles that exhibit minimal detectability and maximum natural appearance, while maintaining superior attack capabilities.

摘要: 在虚拟现实（VR）中，对抗性攻击仍然是一个重要的安全威胁。大多数基于深度学习的物理和数字对抗攻击方法都专注于通过制作包含大量可打印失真的对抗示例来增强攻击性能，这些失真易于人类观察者识别。然而，攻击者很少对生成的攻击图像外观的自然性和舒适性施加限制，导致明显和不自然的攻击。为了解决这一挑战，我们提出了一个框架，将风格转移纳入工艺对抗输入的自然风格，表现出最小的可检测性和最大的自然外观，同时保持卓越的攻击能力。



## **5. Improving Robustness to Model Inversion Attacks via Sparse Coding Architectures**

通过稀疏编码结构提高对模型反转攻击的稳健性 cs.CV

32 pages, 15 Tables, and 9 Figures

**SubmitDate**: 2024-03-21    [abs](http://arxiv.org/abs/2403.14772v1) [paper-pdf](http://arxiv.org/pdf/2403.14772v1)

**Authors**: Sayanton V. Dibbo, Adam Breuer, Juston Moore, Michael Teti

**Abstract**: Recent model inversion attack algorithms permit adversaries to reconstruct a neural network's private training data just by repeatedly querying the network and inspecting its outputs. In this work, we develop a novel network architecture that leverages sparse-coding layers to obtain superior robustness to this class of attacks. Three decades of computer science research has studied sparse coding in the context of image denoising, object recognition, and adversarial misclassification settings, but to the best of our knowledge, its connection to state-of-the-art privacy vulnerabilities remains unstudied. However, sparse coding architectures suggest an advantageous means to defend against model inversion attacks because they allow us to control the amount of irrelevant private information encoded in a network's intermediate representations in a manner that can be computed efficiently during training and that is known to have little effect on classification accuracy. Specifically, compared to networks trained with a variety of state-of-the-art defenses, our sparse-coding architectures maintain comparable or higher classification accuracy while degrading state-of-the-art training data reconstructions by factors of 1.1 to 18.3 across a variety of reconstruction quality metrics (PSNR, SSIM, FID). This performance advantage holds across 5 datasets ranging from CelebA faces to medical images and CIFAR-10, and across various state-of-the-art SGD-based and GAN-based inversion attacks, including Plug-&-Play attacks. We provide a cluster-ready PyTorch codebase to promote research and standardize defense evaluations.

摘要: 最近的模型反转攻击算法允许对手通过重复查询网络并检查其输出来重建神经网络的私有训练数据。在这项工作中，我们开发了一个新的网络架构，利用稀疏编码层，以获得卓越的鲁棒性，这类攻击。三十年的计算机科学研究一直在图像去噪、对象识别和对抗性错误分类设置的背景下研究稀疏编码，但据我们所知，它与最先进的隐私漏洞的联系仍然没有被研究。然而，稀疏编码架构提出了一种有利的方法来防御模型反转攻击，因为它们允许我们控制编码在网络的中间表示中的不相关私有信息的数量，这种方式可以在训练期间有效地计算，并且已知对分类准确性几乎没有影响。具体而言，与使用各种最先进防御训练的网络相比，我们的稀疏编码架构保持了相当或更高的分类准确度，同时在各种重建质量指标（PSNR、SSIM、FID）上降低了1.1至18.3倍的最先进训练数据重建。这种性能优势适用于5个数据集，从CelebA人脸到医学图像和CIFAR—10，以及各种最先进的基于SGD和基于GAN的反转攻击，包括即插即用攻击。我们提供了一个集群就绪的PyTorch代码库，以促进研究和标准化防御评估。



## **6. TMI! Finetuned Models Leak Private Information from their Pretraining Data**

TMI！精细调谐模型从预训练数据中泄漏私人信息 cs.LG

**SubmitDate**: 2024-03-21    [abs](http://arxiv.org/abs/2306.01181v2) [paper-pdf](http://arxiv.org/pdf/2306.01181v2)

**Authors**: John Abascal, Stanley Wu, Alina Oprea, Jonathan Ullman

**Abstract**: Transfer learning has become an increasingly popular technique in machine learning as a way to leverage a pretrained model trained for one task to assist with building a finetuned model for a related task. This paradigm has been especially popular for $\textit{privacy}$ in machine learning, where the pretrained model is considered public, and only the data for finetuning is considered sensitive. However, there are reasons to believe that the data used for pretraining is still sensitive, making it essential to understand how much information the finetuned model leaks about the pretraining data. In this work we propose a new membership-inference threat model where the adversary only has access to the finetuned model and would like to infer the membership of the pretraining data. To realize this threat model, we implement a novel metaclassifier-based attack, $\textbf{TMI}$, that leverages the influence of memorized pretraining samples on predictions in the downstream task. We evaluate $\textbf{TMI}$ on both vision and natural language tasks across multiple transfer learning settings, including finetuning with differential privacy. Through our evaluation, we find that $\textbf{TMI}$ can successfully infer membership of pretraining examples using query access to the finetuned model. An open-source implementation of $\textbf{TMI}$ can be found $\href{https://github.com/johnmath/tmi-pets24}{\text{on GitHub}}$.

摘要: 迁移学习已经成为机器学习中越来越受欢迎的技术，它可以利用为一个任务训练的预训练模型来帮助为相关任务构建微调模型。这种范式在机器学习中特别流行，其中预训练模型被认为是公共的，只有微调数据被认为是敏感的。然而，有理由相信用于预训练的数据仍然是敏感的，因此了解微调模型泄漏了多少关于预训练数据的信息是至关重要的。在这项工作中，我们提出了一个新的成员推断威胁模型，其中的对手只有访问微调模型，并希望推断预训练数据的成员。为了实现这个威胁模型，我们实现了一种新的基于元分类器的攻击，$\textBF {TMI}$，利用了存储的预训练样本对下游任务预测的影响。我们在多个迁移学习设置中对视觉和自然语言任务进行评估，包括差异隐私的微调。通过我们的评估，我们发现$\textbf {TMI}$可以成功地使用查询访问细调模型来推断预训练样本的成员资格。$\textbf {TMI}$的开源实现可以在$\href {https：//github.com/johnmath/tmi—pets24}{\text {on GitHub}}$找到。



## **7. Adversary-Robust Graph-Based Learning of WSIs**

基于图的WISI的对抗鲁棒学习 cs.CV

**SubmitDate**: 2024-03-21    [abs](http://arxiv.org/abs/2403.14489v1) [paper-pdf](http://arxiv.org/pdf/2403.14489v1)

**Authors**: Saba Heidari Gheshlaghi, Milan Aryal, Nasim Yahyasoltani, Masoud Ganji

**Abstract**: Enhancing the robustness of deep learning models against adversarial attacks is crucial, especially in critical domains like healthcare where significant financial interests heighten the risk of such attacks. Whole slide images (WSIs) are high-resolution, digitized versions of tissue samples mounted on glass slides, scanned using sophisticated imaging equipment. The digital analysis of WSIs presents unique challenges due to their gigapixel size and multi-resolution storage format. In this work, we aim at improving the robustness of cancer Gleason grading classification systems against adversarial attacks, addressing challenges at both the image and graph levels. As regards the proposed algorithm, we develop a novel and innovative graph-based model which utilizes GNN to extract features from the graph representation of WSIs. A denoising module, along with a pooling layer is incorporated to manage the impact of adversarial attacks on the WSIs. The process concludes with a transformer module that classifies various grades of prostate cancer based on the processed data. To assess the effectiveness of the proposed method, we conducted a comparative analysis using two scenarios. Initially, we trained and tested the model without the denoiser using WSIs that had not been exposed to any attack. We then introduced a range of attacks at either the image or graph level and processed them through the proposed network. The performance of the model was evaluated in terms of accuracy and kappa scores. The results from this comparison showed a significant improvement in cancer diagnosis accuracy, highlighting the robustness and efficiency of the proposed method in handling adversarial challenges in the context of medical imaging.

摘要: 增强深度学习模型对对抗性攻击的鲁棒性至关重要，特别是在医疗保健等关键领域，因为这些领域的重大经济利益增加了此类攻击的风险。全载玻片图像（WSIs）是高分辨率的数字化版本的组织样本安装在载玻片上，使用精密的成像设备扫描。由于其千兆像素大小和多分辨率存储格式，WSI的数字分析提出了独特的挑战。在这项工作中，我们的目标是提高癌症Gleason分级分类系统对对抗攻击的鲁棒性，解决图像和图形级别的挑战。对于所提出的算法，我们开发了一个新颖的和创新的基于图的模型，利用GNN从图的表示中提取特征。一个去噪模块，以及一个池化层被合并来管理对抗攻击对WSIs的影响。该过程以Transformer模块结束，该模块基于处理后的数据对各种等级的前列腺癌进行分类。为了评估所提出的方法的有效性，我们使用两种情况进行了比较分析。最初，我们使用没有暴露于任何攻击的WSIs来训练和测试模型，而没有去噪器。然后，我们在图像或图形级别引入了一系列攻击，并通过所提出的网络处理它们。模型的性能进行了评估的准确性和kappa评分。该比较的结果显示，癌症诊断准确性有了显著提高，突出了所提出的方法在处理医学成像背景下对抗性挑战方面的鲁棒性和效率。



## **8. A task of anomaly detection for a smart satellite Internet of things system**

智能卫星物联网系统异常检测任务 cs.LG

**SubmitDate**: 2024-03-21    [abs](http://arxiv.org/abs/2403.14738v1) [paper-pdf](http://arxiv.org/pdf/2403.14738v1)

**Authors**: Zilong Shao

**Abstract**: When the equipment is working, real-time collection of environmental sensor data for anomaly detection is one of the key links to prevent industrial process accidents and network attacks and ensure system security. However, under the environment with specific real-time requirements, the anomaly detection for environmental sensors still faces the following difficulties: (1) The complex nonlinear correlation characteristics between environmental sensor data variables lack effective expression methods, and the distribution between the data is difficult to be captured. (2) it is difficult to ensure the real-time monitoring requirements by using complex machine learning models, and the equipment cost is too high. (3) Too little sample data leads to less labeled data in supervised learning. This paper proposes an unsupervised deep learning anomaly detection system. Based on the generative adversarial network and self-attention mechanism, considering the different feature information contained in the local subsequences, it automatically learns the complex linear and nonlinear dependencies between environmental sensor variables, and uses the anomaly score calculation method combining reconstruction error and discrimination error. It can monitor the abnormal points of real sensor data with high real-time performance and can run on the intelligent satellite Internet of things system, which is suitable for the real working environment. Anomaly detection outperforms baseline methods in most cases and has good interpretability, which can be used to prevent industrial accidents and cyber-attacks for monitoring environmental sensors.

摘要: 当设备工作时，实时采集环境传感器数据进行异常检测是防止工业过程事故和网络攻击、保障系统安全的关键环节之一。然而，在特定实时性要求的环境下，环境传感器异常检测仍然面临以下困难：（1）环境传感器数据变量之间复杂的非线性相关特性缺乏有效的表达方法，数据之间的分布难以被捕捉。(2)复杂的机器学习模型难以保证实时监控要求，设备成本过高。(3)太少的样本数据导致监督学习中的标记数据较少。本文提出了一种无监督深度学习异常检测系统。基于生成式对抗网络和自注意机制，考虑到局部序列所包含的不同特征信息，自动学习环境传感器变量之间复杂的线性和非线性依赖关系，采用重构误差和判别误差相结合的异常得分计算方法。它可以实时性高，实时性高，运行在智能卫星物联网系统上，适用于真实工作环境。异常检测在大多数情况下优于基线方法，具有良好的解释性，可用于预防工业事故和监测环境传感器的网络攻击。



## **9. Adversarial Attacks and Defenses in Automated Control Systems: A Comprehensive Benchmark**

自动控制系统中的对抗性攻击和防御：综合基准 cs.LG

**SubmitDate**: 2024-03-21    [abs](http://arxiv.org/abs/2403.13502v2) [paper-pdf](http://arxiv.org/pdf/2403.13502v2)

**Authors**: Vitaliy Pozdnyakov, Aleksandr Kovalenko, Ilya Makarov, Mikhail Drobyshevskiy, Kirill Lukyanov

**Abstract**: Integrating machine learning into Automated Control Systems (ACS) enhances decision-making in industrial process management. One of the limitations to the widespread adoption of these technologies in industry is the vulnerability of neural networks to adversarial attacks. This study explores the threats in deploying deep learning models for fault diagnosis in ACS using the Tennessee Eastman Process dataset. By evaluating three neural networks with different architectures, we subject them to six types of adversarial attacks and explore five different defense methods. Our results highlight the strong vulnerability of models to adversarial samples and the varying effectiveness of defense strategies. We also propose a novel protection approach by combining multiple defense methods and demonstrate it's efficacy. This research contributes several insights into securing machine learning within ACS, ensuring robust fault diagnosis in industrial processes.

摘要: 将机器学习集成到自动控制系统（ACS）中，增强了工业过程管理的决策。在工业中广泛采用这些技术的限制之一是神经网络容易受到对抗性攻击。本研究探索了使用田纳西州Eastman Process数据集在ACS中部署深度学习模型进行故障诊断的威胁。通过评估具有不同架构的三个神经网络，我们将它们置于六种类型的对抗性攻击中，并探索了五种不同的防御方法。我们的研究结果突出了模型对对抗样本的强大脆弱性和防御策略的不同有效性。提出了一种结合多种防御方法的新型防御方法，并验证了其有效性。这项研究为在ACS中保护机器学习提供了几个见解，确保工业过程中的强大故障诊断。



## **10. Adversary-Augmented Simulation to evaluate client-fairness on HyperLedger Fabric**

在Hyperledger Fabric上评估客户端公平性的对抗增强仿真 cs.CR

10 pages (2 pages of references), 8 figures

**SubmitDate**: 2024-03-21    [abs](http://arxiv.org/abs/2403.14342v1) [paper-pdf](http://arxiv.org/pdf/2403.14342v1)

**Authors**: Erwan Mahe, Rouwaida Abdallah, Sara Tucci-Piergiovanni, Pierre-Yves Piriou

**Abstract**: This paper presents a novel adversary model specifically tailored to distributed systems, with the aim to asses the security of blockchain technologies. Building upon literature on adversarial assumptions and capabilities, we include classical notions of failure and communication models to classify and bind the use of adversarial actions. We focus on the effect of these actions on properties of distributed protocols. A significant effort of our research is the integration of this model into the Multi-Agent eXperimenter (MAX) framework. This integration enables realistic simulations of adversarial attacks on blockchain systems. In particular, we have simulated attacks violating a form of client-fairness on HyperLedger Fabric.

摘要: 本文提出了一种专为分布式系统定制的新型攻击者模型，旨在评估区块链技术的安全性。基于对抗性假设和能力的文献，我们包括经典的失败和通信模型的概念，以分类和绑定对抗性行动的使用。我们重点研究这些行为对分布式协议的属性的影响。我们研究的一个重要努力是将该模型集成到多Agent eXperimenter（MAX）框架中。这种集成可以实现对区块链系统的对抗性攻击的真实模拟。特别是，我们模拟了违反Hyperledger Fabric上客户端公平性的攻击。



## **11. Large Language Models for Blockchain Security: A Systematic Literature Review**

区块链安全的大型语言模型：系统文献综述 cs.CR

**SubmitDate**: 2024-03-21    [abs](http://arxiv.org/abs/2403.14280v1) [paper-pdf](http://arxiv.org/pdf/2403.14280v1)

**Authors**: Zheyuan He, Zihao Li, Sen Yang

**Abstract**: Large Language Models (LLMs) have emerged as powerful tools in various domains involving blockchain security (BS). Several recent studies are exploring LLMs applied to BS. However, there remains a gap in our understanding regarding the full scope of applications, impacts, and potential constraints of LLMs on blockchain security. To fill this gap, we conduct a literature review on LLM4BS.   As the first review of LLM's application on blockchain security, our study aims to comprehensively analyze existing research and elucidate how LLMs contribute to enhancing the security of blockchain systems. Through a thorough examination of scholarly works, we delve into the integration of LLMs into various aspects of blockchain security. We explore the mechanisms through which LLMs can bolster blockchain security, including their applications in smart contract auditing, identity verification, anomaly detection, vulnerable repair, and so on. Furthermore, we critically assess the challenges and limitations associated with leveraging LLMs for blockchain security, considering factors such as scalability, privacy concerns, and adversarial attacks. Our review sheds light on the opportunities and potential risks inherent in this convergence, providing valuable insights for researchers, practitioners, and policymakers alike.

摘要: 大型语言模型（LLM）已经成为涉及区块链安全（BS）的各个领域的强大工具。最近的几项研究正在探索法学硕士应用于学士学位。然而，我们对LLM对区块链安全的全部应用范围、影响和潜在限制的理解仍然存在差距。为了填补这一空白，我们对LLM4BS进行了文献综述。   作为对LLM在区块链安全方面的应用的首次审查，我们的研究旨在全面分析现有的研究，并阐明LLM如何有助于增强区块链系统的安全性。通过对学术著作的彻底审查，我们深入研究了将LLM集成到区块链安全的各个方面。我们探索了LLM支持区块链安全的机制，包括其在智能合约审计、身份验证、异常检测、漏洞修复等方面的应用。此外，我们还考虑了可扩展性、隐私问题和对抗性攻击等因素，认真评估了利用LLM实现区块链安全的挑战和局限性。我们的综述揭示了这种融合所固有的机遇和潜在风险，为研究人员、从业者和政策制定者提供了宝贵的见解。



## **12. FMM-Attack: A Flow-based Multi-modal Adversarial Attack on Video-based LLMs**

FMM—Attack：一种基于流的多模式对抗性视频LLM攻击 cs.CV

**SubmitDate**: 2024-03-21    [abs](http://arxiv.org/abs/2403.13507v2) [paper-pdf](http://arxiv.org/pdf/2403.13507v2)

**Authors**: Jinmin Li, Kuofeng Gao, Yang Bai, Jingyun Zhang, Shu-tao Xia, Yisen Wang

**Abstract**: Despite the remarkable performance of video-based large language models (LLMs), their adversarial threat remains unexplored. To fill this gap, we propose the first adversarial attack tailored for video-based LLMs by crafting flow-based multi-modal adversarial perturbations on a small fraction of frames within a video, dubbed FMM-Attack. Extensive experiments show that our attack can effectively induce video-based LLMs to generate incorrect answers when videos are added with imperceptible adversarial perturbations. Intriguingly, our FMM-Attack can also induce garbling in the model output, prompting video-based LLMs to hallucinate. Overall, our observations inspire a further understanding of multi-modal robustness and safety-related feature alignment across different modalities, which is of great importance for various large multi-modal models. Our code is available at https://github.com/THU-Kingmin/FMM-Attack.

摘要: 尽管基于视频的大型语言模型（LLM）表现出色，但它们的对抗性威胁仍未得到探索。为了填补这一空白，我们提出了第一个针对基于视频的LLM的对抗攻击，通过在视频中的一小部分帧上制作基于流的多模式对抗干扰，称为FMM攻击。大量的实验表明，我们的攻击可以有效地诱导基于视频的LLM生成错误的答案时，视频中添加了不可感知的对抗干扰。有趣的是，我们的FM—Attack还可以在模型输出中引起混乱，促使基于视频的LLM产生幻觉。总的来说，我们的观察结果激发了人们对不同模态的多模态鲁棒性和安全相关特性对齐的进一步理解，这对各种大型多模态模型非常重要。我们的代码可在www.example.com获得。



## **13. Quantum-activated neural reservoirs on-chip open up large hardware security models for resilient authentication**

芯片上量子激活神经库为弹性身份验证开辟了大型硬件安全模型 cond-mat.dis-nn

**SubmitDate**: 2024-03-21    [abs](http://arxiv.org/abs/2403.14188v1) [paper-pdf](http://arxiv.org/pdf/2403.14188v1)

**Authors**: Zhao He, Maxim S. Elizarov, Ning Li, Fei Xiang, Andrea Fratalocchi

**Abstract**: Quantum artificial intelligence is a frontier of artificial intelligence research, pioneering quantum AI-powered circuits to address problems beyond the reach of deep learning with classical architectures. This work implements a large-scale quantum-activated recurrent neural network possessing more than 3 trillion hardware nodes/cm$^2$, originating from repeatable atomic-scale nucleation dynamics in an amorphous material integrated on-chip, controlled with 0.07 nW electric power per readout channel. Compared to the best-performing reservoirs currently reported, this implementation increases the scale of the network by two orders of magnitude and reduces the power consumption by six, reaching power efficiencies in the range of the human brain, dissipating 0.2 nW/neuron. When interrogated by a classical input, the chip implements a large-scale hardware security model, enabling dictionary-free authentication secure against statistical inference attacks, including AI's present and future development, even for an adversary with a copy of all the classical components available. Experimental tests report 99.6% reliability, 100% user authentication accuracy, and an ideal 50% key uniqueness. Due to its quantum nature, the chip supports a bit density per feature size area three times higher than the best technology available, with the capacity to store more than $2^{1104}$ keys in a footprint of 1 cm$^2$. Such a quantum-powered platform could help counteract the emerging form of warfare led by the cybercrime industry in breaching authentication to target small to large-scale facilities, from private users to intelligent energy grids.

摘要: 量子人工智能是人工智能研究的前沿，开创了量子人工智能电路，以解决经典架构深度学习无法触及的问题。这项工作实现了一个大规模的量子激活递归神经网络，拥有超过3万亿个硬件节点/cm ^2 $，起源于芯片上集成的无定形材料中的可重复原子级成核动力学，每个读出通道的电功率为0.07 nW。与目前报道的性能最佳的水库相比，这种实现将网络规模增加了两个数量级，并将功耗降低了六个数量级，达到了人脑范围内的功率效率，每个神经元消耗0.2nW。当被经典输入询问时，芯片实现了一个大规模的硬件安全模型，使无字典身份验证能够抵御统计推理攻击，包括人工智能现在和未来的发展，即使对手拥有所有可用的经典组件的副本。实验测试报告了99.6%的可靠性，100%的用户身份验证准确性和理想的50%的密钥唯一性。由于其量子特性，该芯片支持的每功能大小区域位密度比现有最佳技术高三倍，在1 cm $^2 $的占地面积内存储超过2 ^{1104}$的密钥。这样一个量子驱动的平台可以帮助对抗网络犯罪行业领导的新兴战争形式，破坏身份验证以针对从私人用户到智能电网的小型到大型设施。



## **14. Reversible Jump Attack to Textual Classifiers with Modification Reduction**

基于修改约简的文本分类器可逆跳转攻击 cs.CR

**SubmitDate**: 2024-03-21    [abs](http://arxiv.org/abs/2403.14731v1) [paper-pdf](http://arxiv.org/pdf/2403.14731v1)

**Authors**: Mingze Ni, Zhensu Sun, Wei Liu

**Abstract**: Recent studies on adversarial examples expose vulnerabilities of natural language processing (NLP) models. Existing techniques for generating adversarial examples are typically driven by deterministic hierarchical rules that are agnostic to the optimal adversarial examples, a strategy that often results in adversarial samples with a suboptimal balance between magnitudes of changes and attack successes. To this end, in this research we propose two algorithms, Reversible Jump Attack (RJA) and Metropolis-Hasting Modification Reduction (MMR), to generate highly effective adversarial examples and to improve the imperceptibility of the examples, respectively. RJA utilizes a novel randomization mechanism to enlarge the search space and efficiently adapts to a number of perturbed words for adversarial examples. With these generated adversarial examples, MMR applies the Metropolis-Hasting sampler to enhance the imperceptibility of adversarial examples. Extensive experiments demonstrate that RJA-MMR outperforms current state-of-the-art methods in attack performance, imperceptibility, fluency and grammar correctness.

摘要: 最近对对抗性示例的研究揭示了自然语言处理（NLP）模型的脆弱性。用于生成对抗性示例的现有技术通常由确定性分层规则驱动，这些规则与最佳对抗性示例无关，这种策略通常导致对抗性样本在变化幅度和攻击成功之间达到次优平衡。为此，在本研究中，我们提出了两个算法，可逆跳转攻击（RJA）和都市加速修改减少（MMR），以生成高效的对抗性示例和改善示例的不可感知性，分别。RJA利用一种新的随机化机制来扩大搜索空间，并有效地适应对抗性示例的多个扰动词。有了这些生成的对抗性示例，MMR应用Metropolis—Hasting采样器来增强对抗性示例的不可感知性。大量的实验表明，RJA—MMR算法在攻击性能、不可感知性、流畅性和语法正确性等方面都优于现有的算法。



## **15. A Signal Injection Attack Against Zero Involvement Pairing and Authentication for the Internet of Things**

一种针对物联网零参与配对认证的信号注入攻击 cs.CR

**SubmitDate**: 2024-03-20    [abs](http://arxiv.org/abs/2403.14018v1) [paper-pdf](http://arxiv.org/pdf/2403.14018v1)

**Authors**: Isaac Ahlgren, Jack West, Kyuin Lee, George Thiruvathukal, Neil Klingensmith

**Abstract**: Zero Involvement Pairing and Authentication (ZIPA) is a promising technique for autoprovisioning large networks of Internet-of-Things (IoT) devices. In this work, we present the first successful signal injection attack on a ZIPA system. Most existing ZIPA systems assume there is a negligible amount of influence from the unsecured outside space on the secured inside space. In reality, environmental signals do leak from adjacent unsecured spaces and influence the environment of the secured space. Our attack takes advantage of this fact to perform a signal injection attack on the popular Schurmann & Sigg algorithm. The keys generated by the adversary with a signal injection attack at 95 dBA is within the standard error of the legitimate device.

摘要: 零参与配对和身份验证(ZIPA)是自动配置大型物联网(IoT)设备网络的一种很有前途的技术。在这项工作中，我们提出了第一个成功的信号注入攻击ZIPA系统。大多数现有的ZIPA系统假设不安全的外部空间对安全的内部空间的影响可以忽略不计。在现实中，环境信号确实会从邻近的未安全空间泄漏，并影响安全空间的环境。我们的攻击利用这一事实对流行的Schurmann&Sigg算法进行信号注入攻击。敌手使用95dBA的信号注入攻击生成的密钥在合法设备的标准误差范围内。



## **16. AutoDAN: Generating Stealthy Jailbreak Prompts on Aligned Large Language Models**

AutoDAN：在对齐的大型语言模型上生成隐蔽的越狱脚本 cs.CL

Published as a conference paper at ICLR 2024. Code is available at  https://github.com/SheltonLiu-N/AutoDAN

**SubmitDate**: 2024-03-20    [abs](http://arxiv.org/abs/2310.04451v2) [paper-pdf](http://arxiv.org/pdf/2310.04451v2)

**Authors**: Xiaogeng Liu, Nan Xu, Muhao Chen, Chaowei Xiao

**Abstract**: The aligned Large Language Models (LLMs) are powerful language understanding and decision-making tools that are created through extensive alignment with human feedback. However, these large models remain susceptible to jailbreak attacks, where adversaries manipulate prompts to elicit malicious outputs that should not be given by aligned LLMs. Investigating jailbreak prompts can lead us to delve into the limitations of LLMs and further guide us to secure them. Unfortunately, existing jailbreak techniques suffer from either (1) scalability issues, where attacks heavily rely on manual crafting of prompts, or (2) stealthiness problems, as attacks depend on token-based algorithms to generate prompts that are often semantically meaningless, making them susceptible to detection through basic perplexity testing. In light of these challenges, we intend to answer this question: Can we develop an approach that can automatically generate stealthy jailbreak prompts? In this paper, we introduce AutoDAN, a novel jailbreak attack against aligned LLMs. AutoDAN can automatically generate stealthy jailbreak prompts by the carefully designed hierarchical genetic algorithm. Extensive evaluations demonstrate that AutoDAN not only automates the process while preserving semantic meaningfulness, but also demonstrates superior attack strength in cross-model transferability, and cross-sample universality compared with the baseline. Moreover, we also compare AutoDAN with perplexity-based defense methods and show that AutoDAN can bypass them effectively.

摘要: 对齐的大型语言模型（LLM）是强大的语言理解和决策工具，通过与人类反馈的广泛对齐而创建。然而，这些大型模型仍然容易受到越狱攻击，攻击者操纵提示以引出不应该由对齐的LLM给出的恶意输出。调查越狱提示可以引导我们深入研究LLM的局限性，并进一步指导我们保护它们。不幸的是，现有的越狱技术要么受到（1）可伸缩性问题，其中攻击严重依赖人工制作提示符，要么（2）隐蔽性问题，因为攻击依赖基于令牌的算法来生成通常在语义上毫无意义的提示符，使得它们易于通过基本的困惑度测试检测。鉴于这些挑战，我们打算回答这个问题：我们能否开发出一种方法，可以自动生成隐蔽的越狱提示？在本文中，我们介绍了一种新的针对对齐的LLM的越狱攻击—AutoDAN。AutoDAN可以通过精心设计的层次遗传算法自动生成隐蔽越狱提示。广泛的评估表明，AutoDAN不仅在保持语义意义的同时自动化过程，而且在跨模型可迁移性和跨样本通用性方面表现出优于基线的攻击强度。此外，我们还比较了AutoDAN和基于复杂度的防御方法，表明AutoDAN可以有效地绕过它们。



## **17. Certified Human Trajectory Prediction**

经认证的人体轨迹预测 cs.CV

**SubmitDate**: 2024-03-20    [abs](http://arxiv.org/abs/2403.13778v1) [paper-pdf](http://arxiv.org/pdf/2403.13778v1)

**Authors**: Mohammadhossein Bahari, Saeed Saadatnejad, Amirhossein Asgari Farsangi, Seyed-Mohsen Moosavi-Dezfooli, Alexandre Alahi

**Abstract**: Trajectory prediction plays an essential role in autonomous vehicles. While numerous strategies have been developed to enhance the robustness of trajectory prediction models, these methods are predominantly heuristic and do not offer guaranteed robustness against adversarial attacks and noisy observations. In this work, we propose a certification approach tailored for the task of trajectory prediction. To this end, we address the inherent challenges associated with trajectory prediction, including unbounded outputs, and mutli-modality, resulting in a model that provides guaranteed robustness. Furthermore, we integrate a denoiser into our method to further improve the performance. Through comprehensive evaluations, we demonstrate the effectiveness of the proposed technique across various baselines and using standard trajectory prediction datasets. The code will be made available online: https://s-attack.github.io/

摘要: 轨迹预测在自动驾驶汽车中起着至关重要的作用。虽然已经开发了许多策略来增强轨迹预测模型的鲁棒性，但这些方法主要是启发式的，并且不提供对对抗攻击和噪声观测的保证鲁棒性。在这项工作中，我们提出了一种为弹道预测任务量身定制的认证方法。为此，我们解决了与轨迹预测相关的固有挑战，包括无边界输出和多模态，导致一个模型，提供有保证的鲁棒性。此外，我们在我们的方法中加入了去噪器，以进一步提高性能。通过综合评估，我们证明了所提出的技术在各种基线和使用标准轨迹预测数据集的有效性。该代码将在网上提供：www.example.com



## **18. Defending Against Indirect Prompt Injection Attacks With Spotlighting**

利用聚光灯防御间接即时注入攻击 cs.CR

**SubmitDate**: 2024-03-20    [abs](http://arxiv.org/abs/2403.14720v1) [paper-pdf](http://arxiv.org/pdf/2403.14720v1)

**Authors**: Keegan Hines, Gary Lopez, Matthew Hall, Federico Zarfati, Yonatan Zunger, Emre Kiciman

**Abstract**: Large Language Models (LLMs), while powerful, are built and trained to process a single text input. In common applications, multiple inputs can be processed by concatenating them together into a single stream of text. However, the LLM is unable to distinguish which sections of prompt belong to various input sources. Indirect prompt injection attacks take advantage of this vulnerability by embedding adversarial instructions into untrusted data being processed alongside user commands. Often, the LLM will mistake the adversarial instructions as user commands to be followed, creating a security vulnerability in the larger system. We introduce spotlighting, a family of prompt engineering techniques that can be used to improve LLMs' ability to distinguish among multiple sources of input. The key insight is to utilize transformations of an input to provide a reliable and continuous signal of its provenance. We evaluate spotlighting as a defense against indirect prompt injection attacks, and find that it is a robust defense that has minimal detrimental impact to underlying NLP tasks. Using GPT-family models, we find that spotlighting reduces the attack success rate from greater than {50}\% to below {2}\% in our experiments with minimal impact on task efficacy.

摘要: 大型语言模型（LLM）虽然功能强大，但被构建和训练为处理单个文本输入。在常见的应用程序中，多个输入可以通过将它们连接到单个文本流中来处理。然而，LLM无法区分提示符的哪些部分属于不同的输入源。间接提示注入攻击通过将对抗指令嵌入到与用户命令一起处理的不可信数据中来利用此漏洞。通常，LLM会将对抗指令误认为是要遵循的用户命令，从而在更大的系统中造成安全漏洞。我们介绍了聚光灯，一系列即时工程技术，可用于提高LLM区分多个输入源的能力。关键的洞察力是利用输入的转换来提供其来源的可靠和连续的信号。我们评估聚光灯作为一种防御间接提示注入攻击，并发现它是一个强大的防御，具有最小的不利影响底层NLP任务。在实验中，我们发现聚光灯使攻击成功率从大于{50}\%降低到{2}\%以下，对任务效能的影响最小。



## **19. On the Privacy Effect of Data Enhancement via the Lens of Memorization**

从加密化的角度看数据增强的隐私效应 cs.LG

Accepted by IEEE TIFS, 17 pages

**SubmitDate**: 2024-03-20    [abs](http://arxiv.org/abs/2208.08270v3) [paper-pdf](http://arxiv.org/pdf/2208.08270v3)

**Authors**: Xiao Li, Qiongxiu Li, Zhanhao Hu, Xiaolin Hu

**Abstract**: Machine learning poses severe privacy concerns as it has been shown that the learned models can reveal sensitive information about their training data. Many works have investigated the effect of widely adopted data augmentation and adversarial training techniques, termed data enhancement in the paper, on the privacy leakage of machine learning models. Such privacy effects are often measured by membership inference attacks (MIAs), which aim to identify whether a particular example belongs to the training set or not. We propose to investigate privacy from a new perspective called memorization. Through the lens of memorization, we find that previously deployed MIAs produce misleading results as they are less likely to identify samples with higher privacy risks as members compared to samples with low privacy risks. To solve this problem, we deploy a recent attack that can capture individual samples' memorization degrees for evaluation. Through extensive experiments, we unveil several findings about the connections between three essential properties of machine learning models, including privacy, generalization gap, and adversarial robustness. We demonstrate that the generalization gap and privacy leakage are less correlated than those of the previous results. Moreover, there is not necessarily a trade-off between adversarial robustness and privacy as stronger adversarial robustness does not make the model more susceptible to privacy attacks.

摘要: 机器学习带来了严重的隐私问题，因为已经证明学习的模型可以揭示有关其训练数据的敏感信息。许多工作已经研究了广泛采用的数据增强和对抗训练技术（在论文中称为数据增强）对机器学习模型隐私泄漏的影响。这种隐私效应通常通过成员推断攻击（MIA）来衡量，其目的是识别特定示例是否属于训练集。我们建议从一个叫做记忆的新角度来研究隐私。通过记忆的镜头，我们发现以前部署的MIA产生误导性的结果，因为它们不太可能识别具有较高隐私风险的样本作为成员相比，具有较低隐私风险的样本。为了解决这个问题，我们部署了一个最近的攻击，可以捕获单个样本的记忆程度进行评估。通过大量的实验，我们揭示了机器学习模型三个基本属性之间的联系，包括隐私，泛化差距和对抗鲁棒性。我们证明了一般化差距和隐私泄漏的相关性比先前的结果。此外，对抗鲁棒性和隐私之间不一定存在权衡，因为更强的对抗鲁棒性不会使模型更容易受到隐私攻击。



## **20. Capsule Neural Networks as Noise Stabilizer for Time Series Data**

胶囊神经网络作为时间序列数据的噪声稳定器 cs.LG

3 pages, 3 figures

**SubmitDate**: 2024-03-20    [abs](http://arxiv.org/abs/2403.13867v1) [paper-pdf](http://arxiv.org/pdf/2403.13867v1)

**Authors**: Soyeon Kim, Jihyeon Seong, Hyunkyung Han, Jaesik Choi

**Abstract**: Capsule Neural Networks utilize capsules, which bind neurons into a single vector and learn position equivariant features, which makes them more robust than original Convolutional Neural Networks. CapsNets employ an affine transformation matrix and dynamic routing with coupling coefficients to learn robustly. In this paper, we investigate the effectiveness of CapsNets in analyzing highly sensitive and noisy time series sensor data. To demonstrate CapsNets robustness, we compare their performance with original CNNs on electrocardiogram data, a medical time series sensor data with complex patterns and noise. Our study provides empirical evidence that CapsNets function as noise stabilizers, as investigated by manual and adversarial attack experiments using the fast gradient sign method and three manual attacks, including offset shifting, gradual drift, and temporal lagging. In summary, CapsNets outperform CNNs in both manual and adversarial attacked data. Our findings suggest that CapsNets can be effectively applied to various sensor systems to improve their resilience to noise attacks. These results have significant implications for designing and implementing robust machine learning models in real world applications. Additionally, this study contributes to the effectiveness of CapsNet models in handling noisy data and highlights their potential for addressing the challenges of noise data in time series analysis.

摘要: 胶囊神经网络利用胶囊，将神经元绑定到单个向量中并学习位置等变特征，这使得它们比原始卷积神经网络更健壮。CapsNets使用仿射变换矩阵和具有耦合系数的动态路由来鲁棒学习。在本文中，我们研究了CapsNets在分析高灵敏度和噪声时间序列传感器数据的有效性。为了证明CapsNets的鲁棒性，我们将其与原始CNN在心电图数据上的性能进行了比较，这是一种具有复杂模式和噪声的医疗时间序列传感器数据。我们的研究提供了经验证据，证明CapsNets作为噪声稳定器，通过手动和对抗攻击实验，使用快速梯度符号法和三种手动攻击，包括偏移偏移，逐渐漂移和时间滞后。总之，CapsNets在人工和对抗攻击数据方面都优于CNN。我们的研究结果表明，CapsNets可以有效地应用于各种传感器系统，以提高其对噪声攻击的恢复能力。这些结果对于在现实世界应用中设计和实现强大的机器学习模型具有重要意义。此外，这项研究有助于CapsNet模型在处理噪声数据方面的有效性，并强调了它们在解决时间序列分析中噪声数据挑战方面的潜力。



## **21. Have You Poisoned My Data? Defending Neural Networks against Data Poisoning**

你是不是毒害了我的数据？保护神经网络免受数据中毒 cs.LG

Paper accepted for publication at European Symposium on Research in  Computer Security (ESORICS) 2024

**SubmitDate**: 2024-03-20    [abs](http://arxiv.org/abs/2403.13523v1) [paper-pdf](http://arxiv.org/pdf/2403.13523v1)

**Authors**: Fabio De Gaspari, Dorjan Hitaj, Luigi V. Mancini

**Abstract**: The unprecedented availability of training data fueled the rapid development of powerful neural networks in recent years. However, the need for such large amounts of data leads to potential threats such as poisoning attacks: adversarial manipulations of the training data aimed at compromising the learned model to achieve a given adversarial goal.   This paper investigates defenses against clean-label poisoning attacks and proposes a novel approach to detect and filter poisoned datapoints in the transfer learning setting. We define a new characteristic vector representation of datapoints and show that it effectively captures the intrinsic properties of the data distribution. Through experimental analysis, we demonstrate that effective poisons can be successfully differentiated from clean points in the characteristic vector space. We thoroughly evaluate our proposed approach and compare it to existing state-of-the-art defenses using multiple architectures, datasets, and poison budgets. Our evaluation shows that our proposal outperforms existing approaches in defense rate and final trained model performance across all experimental settings.

摘要: 近年来，训练数据前所未有的可用性推动了强大的神经网络的快速发展。然而，对如此大量的数据的需求导致了潜在的威胁，如中毒攻击：对训练数据的对抗性操纵，目的是损害学习的模型，以实现给定的对抗性目标。本文研究了干净标签中毒攻击的防御方法，提出了一种在迁移学习环境下检测和过滤有毒数据点的新方法。我们定义了一种新的数据点特征向量表示，并证明了它有效地捕捉了数据分布的内在属性。通过实验分析，我们证明了在特征向量空间中可以从清洁点成功地区分出有效毒物。我们彻底评估了我们提出的方法，并将其与使用多个架构、数据集和有毒预算的现有最先进的防御进行了比较。我们的评估表明，在所有实验设置下，我们的建议在防御率和最终训练的模型性能方面都优于现有的方法。



## **22. DD-RobustBench: An Adversarial Robustness Benchmark for Dataset Distillation**

DD—RobustBench：数据集蒸馏的对抗性鲁棒性基准 cs.CV

**SubmitDate**: 2024-03-20    [abs](http://arxiv.org/abs/2403.13322v1) [paper-pdf](http://arxiv.org/pdf/2403.13322v1)

**Authors**: Yifan Wu, Jiawei Du, Ping Liu, Yuewei Lin, Wenqing Cheng, Wei Xu

**Abstract**: Dataset distillation is an advanced technique aimed at compressing datasets into significantly smaller counterparts, while preserving formidable training performance. Significant efforts have been devoted to promote evaluation accuracy under limited compression ratio while overlooked the robustness of distilled dataset. In this work, we introduce a comprehensive benchmark that, to the best of our knowledge, is the most extensive to date for evaluating the adversarial robustness of distilled datasets in a unified way. Our benchmark significantly expands upon prior efforts by incorporating a wider range of dataset distillation methods, including the latest advancements such as TESLA and SRe2L, a diverse array of adversarial attack methods, and evaluations across a broader and more extensive collection of datasets such as ImageNet-1K. Moreover, we assessed the robustness of these distilled datasets against representative adversarial attack algorithms like PGD and AutoAttack, while exploring their resilience from a frequency perspective. We also discovered that incorporating distilled data into the training batches of the original dataset can yield to improvement of robustness.

摘要: 数据集蒸馏是一种先进的技术，旨在将数据集压缩成显著较小的对应数据，同时保持强大的训练性能。一直致力于提高有限压缩比下的评价精度，但忽略了提取数据集的鲁棒性。在这项工作中，我们介绍了一个全面的基准，据我们所知，是迄今为止最广泛的评估提取数据集的对抗鲁棒性的基准。我们的基准测试大大扩展了以前的努力，通过整合更广泛的数据集提取方法，包括TESLA和SRe2L等最新进展，各种对抗性攻击方法，以及对更广泛和更广泛的数据集（如ImageNet—1K）进行评估。此外，我们评估了这些提取的数据集对典型的对抗攻击算法（如PGD和AutoAttack）的鲁棒性，同时从频率角度探索它们的弹性。我们还发现，将提取的数据合并到原始数据集的训练批次中可以提高鲁棒性。



## **23. Enhancing Security in Multi-Robot Systems through Co-Observation Planning, Reachability Analysis, and Network Flow**

通过协同观察规划、可达性分析和网络流增强多机器人系统的安全性 cs.RO

12 pages, 6 figures, submitted to IEEE Transactions on Control of  Network Systems

**SubmitDate**: 2024-03-20    [abs](http://arxiv.org/abs/2403.13266v1) [paper-pdf](http://arxiv.org/pdf/2403.13266v1)

**Authors**: Ziqi Yang, Roberto Tron

**Abstract**: This paper addresses security challenges in multi-robot systems (MRS) where adversaries may compromise robot control, risking unauthorized access to forbidden areas. We propose a novel multi-robot optimal planning algorithm that integrates mutual observations and introduces reachability constraints for enhanced security. This ensures that, even with adversarial movements, compromised robots cannot breach forbidden regions without missing scheduled co-observations. The reachability constraint uses ellipsoidal over-approximation for efficient intersection checking and gradient computation. To enhance system resilience and tackle feasibility challenges, we also introduce sub-teams. These cohesive units replace individual robot assignments along each route, enabling redundant robots to deviate for co-observations across different trajectories, securing multiple sub-teams without requiring modifications. We formulate the cross-trajectory co-observation plan by solving a network flow coverage problem on the checkpoint graph generated from the original unsecured MRS trajectories, providing the same security guarantees against plan-deviation attacks. We demonstrate the effectiveness and robustness of our proposed algorithm, which significantly strengthens the security of multi-robot systems in the face of adversarial threats.

摘要: 本文讨论了多机器人系统（MRS）中的安全挑战，其中对手可能会危及机器人控制，风险未经授权访问禁区。我们提出了一种新的多机器人最优规划算法，该算法集成了相互观察，并引入了可达性约束，以增强安全性。这确保了，即使是对抗性的移动，受损机器人也不能在不错过预定的共同观测的情况下突破禁区。可达性约束使用椭球体过逼近来进行有效的交叉检查和梯度计算。为提高系统的弹性及应对可行性挑战，我们亦引入了小组小组。这些凝聚力单元取代了每条路线上的单个机器人分配，使冗余机器人能够在不同轨迹上偏离共同观察，从而在不需要修改的情况下保护多个子团队。我们制定了交叉轨迹共同观测计划，通过解决网络流覆盖问题产生的检查点图从原始的不安全MRS轨迹，提供了相同的安全保证，以防止计划偏差攻击。我们证明了我们提出的算法的有效性和鲁棒性，它显着加强了多机器人系统面对对抗威胁的安全性。



## **24. ADAPT to Robustify Prompt Tuning Vision Transformers**

ADAPT来Robustify即时调谐视觉变压器 cs.LG

**SubmitDate**: 2024-03-19    [abs](http://arxiv.org/abs/2403.13196v1) [paper-pdf](http://arxiv.org/pdf/2403.13196v1)

**Authors**: Masih Eskandar, Tooba Imtiaz, Zifeng Wang, Jennifer Dy

**Abstract**: The performance of deep models, including Vision Transformers, is known to be vulnerable to adversarial attacks. Many existing defenses against these attacks, such as adversarial training, rely on full-model fine-tuning to induce robustness in the models. These defenses require storing a copy of the entire model, that can have billions of parameters, for each task. At the same time, parameter-efficient prompt tuning is used to adapt large transformer-based models to downstream tasks without the need to save large copies. In this paper, we examine parameter-efficient prompt tuning of Vision Transformers for downstream tasks under the lens of robustness. We show that previous adversarial defense methods, when applied to the prompt tuning paradigm, suffer from gradient obfuscation and are vulnerable to adaptive attacks. We introduce ADAPT, a novel framework for performing adaptive adversarial training in the prompt tuning paradigm. Our method achieves competitive robust accuracy of ~40% w.r.t. SOTA robustness methods using full-model fine-tuning, by tuning only ~1% of the number of parameters.

摘要: 众所周知，包括Vision Transformers在内的深度模型的性能很容易受到对手的攻击。许多现有的针对这些攻击的防御，如对抗性训练，都依赖于全模型微调来诱导模型的健壮性。这些防御需要为每个任务存储整个模型的副本，该副本可以具有数十亿个参数。同时，使用参数高效的即时调整来使大型基于变压器的模型适应下游任务，而不需要保存大量副本。在这篇文章中，我们研究了稳健性镜头下的视觉变形器的参数高效的下游任务的快速调整。我们表明，以前的对抗性防御方法，当应用于即时调整范例时，遭受梯度混淆，并且容易受到适应性攻击。我们介绍了Adapt，这是一种在即时调整范式中执行自适应对抗性训练的新框架。我们的方法获得了~40%的W.r.t.SOTA稳健性方法采用全模型微调，只需调整~1%的参数即可。



## **25. The Impact of Adversarial Node Placement in Decentralized Federated Learning Networks**

分散式联邦学习网络中对抗节点放置的影响 cs.CR

Accepted to ICC 2024 conference

**SubmitDate**: 2024-03-19    [abs](http://arxiv.org/abs/2311.07946v4) [paper-pdf](http://arxiv.org/pdf/2311.07946v4)

**Authors**: Adam Piaseczny, Eric Ruzomberka, Rohit Parasnis, Christopher G. Brinton

**Abstract**: As Federated Learning (FL) grows in popularity, new decentralized frameworks are becoming widespread. These frameworks leverage the benefits of decentralized environments to enable fast and energy-efficient inter-device communication. However, this growing popularity also intensifies the need for robust security measures. While existing research has explored various aspects of FL security, the role of adversarial node placement in decentralized networks remains largely unexplored. This paper addresses this gap by analyzing the performance of decentralized FL for various adversarial placement strategies when adversaries can jointly coordinate their placement within a network. We establish two baseline strategies for placing adversarial node: random placement and network centrality-based placement. Building on this foundation, we propose a novel attack algorithm that prioritizes adversarial spread over adversarial centrality by maximizing the average network distance between adversaries. We show that the new attack algorithm significantly impacts key performance metrics such as testing accuracy, outperforming the baseline frameworks by between $9\%$ and $66.5\%$ for the considered setups. Our findings provide valuable insights into the vulnerabilities of decentralized FL systems, setting the stage for future research aimed at developing more secure and robust decentralized FL frameworks.

摘要: 随着联邦学习(FL)的流行，新的去中心化框架正在变得广泛。这些框架利用分散环境的优势，实现快速、节能的设备间通信。然而，这种日益增长的人气也加剧了采取强有力的安全措施的必要性。虽然现有的研究已经探索了FL安全的各个方面，但敌意节点放置在分散网络中的作用在很大程度上仍未被探索。本文通过分析当对手可以在一个网络内联合协调他们的放置时，分散的FL在不同的对手放置策略下的性能来解决这一差距。我们建立了两种放置敌意节点的基线策略：随机放置和基于网络中心性的放置。在此基础上，我们提出了一种新的攻击算法，该算法通过最大化对手之间的平均网络距离来优先考虑对手的传播而不是对手的中心。我们发现，新的攻击算法显著影响了测试准确率等关键性能指标，在所考虑的设置下，性能比基准框架高出9到66.5美元。我们的发现对去中心化FL系统的脆弱性提供了有价值的见解，为未来旨在开发更安全和健壮的去中心化FL框架的研究奠定了基础。



## **26. Review of Generative AI Methods in Cybersecurity**

网络安全中的生成性人工智能方法综述 cs.CR

40 pages

**SubmitDate**: 2024-03-19    [abs](http://arxiv.org/abs/2403.08701v2) [paper-pdf](http://arxiv.org/pdf/2403.08701v2)

**Authors**: Yagmur Yigit, William J Buchanan, Madjid G Tehrani, Leandros Maglaras

**Abstract**: Over the last decade, Artificial Intelligence (AI) has become increasingly popular, especially with the use of chatbots such as ChatGPT, Gemini, and DALL-E. With this rise, large language models (LLMs) and Generative AI (GenAI) have also become more prevalent in everyday use. These advancements strengthen cybersecurity's defensive posture and open up new attack avenues for adversaries as well. This paper provides a comprehensive overview of the current state-of-the-art deployments of GenAI, covering assaults, jailbreaking, and applications of prompt injection and reverse psychology. This paper also provides the various applications of GenAI in cybercrimes, such as automated hacking, phishing emails, social engineering, reverse cryptography, creating attack payloads, and creating malware. GenAI can significantly improve the automation of defensive cyber security processes through strategies such as dataset construction, safe code development, threat intelligence, defensive measures, reporting, and cyberattack detection. In this study, we suggest that future research should focus on developing robust ethical norms and innovative defense mechanisms to address the current issues that GenAI creates and to also further encourage an impartial approach to its future application in cybersecurity. Moreover, we underscore the importance of interdisciplinary approaches further to bridge the gap between scientific developments and ethical considerations.

摘要: 在过去的十年里，人工智能(AI)变得越来越流行，特别是随着ChatGPT、Gemini和Dall-E等聊天机器人的使用。随着这一崛起，大型语言模型(LLM)和生成性人工智能(GenAI)也在日常使用中变得更加普遍。这些进展加强了网络安全的防御态势，也为对手开辟了新的攻击途径。本文全面概述了GenAI当前最先进的部署，包括攻击、越狱以及快速注射和反向心理学的应用。本文还提供了GenAI在网络犯罪中的各种应用，如自动黑客、网络钓鱼电子邮件、社会工程、反向密码学、创建攻击负载和创建恶意软件。GenAI可以通过数据集构建、安全代码开发、威胁情报、防御措施、报告和网络攻击检测等策略，显著提高防御性网络安全流程的自动化程度。在这项研究中，我们建议未来的研究应专注于发展强大的伦理规范和创新的防御机制，以解决GenAI目前造成的问题，并进一步鼓励对其未来在网络安全中的应用采取公正的方法。此外，我们强调跨学科方法的重要性，以进一步弥合科学发展和伦理考量之间的差距。



## **27. As Firm As Their Foundations: Can open-sourced foundation models be used to create adversarial examples for downstream tasks?**

作为坚实的基础：开源的基础模型可以用于为下游任务创建对抗性示例吗？ cs.CV

**SubmitDate**: 2024-03-19    [abs](http://arxiv.org/abs/2403.12693v1) [paper-pdf](http://arxiv.org/pdf/2403.12693v1)

**Authors**: Anjun Hu, Jindong Gu, Francesco Pinto, Konstantinos Kamnitsas, Philip Torr

**Abstract**: Foundation models pre-trained on web-scale vision-language data, such as CLIP, are widely used as cornerstones of powerful machine learning systems. While pre-training offers clear advantages for downstream learning, it also endows downstream models with shared adversarial vulnerabilities that can be easily identified through the open-sourced foundation model. In this work, we expose such vulnerabilities in CLIP's downstream models and show that foundation models can serve as a basis for attacking their downstream systems. In particular, we propose a simple yet effective adversarial attack strategy termed Patch Representation Misalignment (PRM). Solely based on open-sourced CLIP vision encoders, this method produces adversaries that simultaneously fool more than 20 downstream models spanning 4 common vision-language tasks (semantic segmentation, object detection, image captioning and visual question-answering). Our findings highlight the concerning safety risks introduced by the extensive usage of public foundational models in the development of downstream systems, calling for extra caution in these scenarios.

摘要: 在网络规模的视觉语言数据上预先训练的基础模型，如CLIP，被广泛用作强大的机器学习系统的基石。虽然预训练为下游学习提供了明显的优势，但它也赋予下游模型共同的对抗漏洞，这些漏洞可以通过开源的基础模型轻松识别。在这项工作中，我们暴露了CLIP的下游模型中的这些漏洞，并表明基础模型可以作为攻击其下游系统的基础。特别地，我们提出了一个简单而有效的对抗攻击策略称为补丁表示失准（PRM）。该方法仅基于开源的CLIP视觉编码器，产生了同时欺骗20多个下游模型的对手，这些模型涵盖了4个常见的视觉语言任务（语义分割、对象检测、图像字幕和视觉问答）。我们的研究结果强调了在下游系统开发中广泛使用公共基础模型所带来的安全风险，呼吁在这些情况下格外谨慎。



## **28. RigorLLM: Resilient Guardrails for Large Language Models against Undesired Content**

RigorLLM：针对不期望内容的大型语言模型的弹性防护 cs.CR

**SubmitDate**: 2024-03-19    [abs](http://arxiv.org/abs/2403.13031v1) [paper-pdf](http://arxiv.org/pdf/2403.13031v1)

**Authors**: Zhuowen Yuan, Zidi Xiong, Yi Zeng, Ning Yu, Ruoxi Jia, Dawn Song, Bo Li

**Abstract**: Recent advancements in Large Language Models (LLMs) have showcased remarkable capabilities across various tasks in different domains. However, the emergence of biases and the potential for generating harmful content in LLMs, particularly under malicious inputs, pose significant challenges. Current mitigation strategies, while effective, are not resilient under adversarial attacks. This paper introduces Resilient Guardrails for Large Language Models (RigorLLM), a novel framework designed to efficiently and effectively moderate harmful and unsafe inputs and outputs for LLMs. By employing a multi-faceted approach that includes energy-based training data augmentation through Langevin dynamics, optimizing a safe suffix for inputs via minimax optimization, and integrating a fusion-based model combining robust KNN with LLMs based on our data augmentation, RigorLLM offers a robust solution to harmful content moderation. Our experimental evaluations demonstrate that RigorLLM not only outperforms existing baselines like OpenAI API and Perspective API in detecting harmful content but also exhibits unparalleled resilience to jailbreaking attacks. The innovative use of constrained optimization and a fusion-based guardrail approach represents a significant step forward in developing more secure and reliable LLMs, setting a new standard for content moderation frameworks in the face of evolving digital threats.

摘要: 大型语言模型（LLM）的最新进展已经在不同领域的各种任务中展示了非凡的能力。然而，偏见的出现以及在LLMs中产生有害内容的可能性，特别是在恶意输入下，构成了重大挑战。目前的缓解战略虽然有效，但在对抗性攻击下没有复原力。本文介绍了大型语言模型的弹性护栏（RigorLLM），一个新的框架，旨在有效地缓解LLM的有害和不安全的输入和输出。通过采用多方面的方法，包括通过Langevin动力学进行基于能量的训练数据增强，通过极小极大优化优化输入的安全后缀，并基于我们的数据增强集成基于融合的模型，将鲁棒KNN与LLM相结合，RigorLLM提供了一个强大的解决方案，以应对有害内容的审核。我们的实验评估表明，RigorLLM不仅在检测有害内容方面优于OpenAI API和Perspective API等现有基线，而且对越狱攻击表现出无与伦比的韧性。约束优化和基于融合的护栏方法的创新使用代表了在开发更安全和可靠的LLM方面向前迈出的重要一步，为面对不断变化的数字威胁的内容审核框架设置了新的标准。



## **29. Securing Large Language Models: Threats, Vulnerabilities and Responsible Practices**

保护大型语言模型：威胁、漏洞和负责任的实践 cs.CR

**SubmitDate**: 2024-03-19    [abs](http://arxiv.org/abs/2403.12503v1) [paper-pdf](http://arxiv.org/pdf/2403.12503v1)

**Authors**: Sara Abdali, Richard Anarfi, CJ Barberan, Jia He

**Abstract**: Large language models (LLMs) have significantly transformed the landscape of Natural Language Processing (NLP). Their impact extends across a diverse spectrum of tasks, revolutionizing how we approach language understanding and generations. Nevertheless, alongside their remarkable utility, LLMs introduce critical security and risk considerations. These challenges warrant careful examination to ensure responsible deployment and safeguard against potential vulnerabilities. This research paper thoroughly investigates security and privacy concerns related to LLMs from five thematic perspectives: security and privacy concerns, vulnerabilities against adversarial attacks, potential harms caused by misuses of LLMs, mitigation strategies to address these challenges while identifying limitations of current strategies. Lastly, the paper recommends promising avenues for future research to enhance the security and risk management of LLMs.

摘要: 大型语言模型（LLM）显著改变了自然语言处理（NLP）的前景。它们的影响延伸到各种任务，彻底改变了我们处理语言理解和世代的方式。然而，除了其显著的实用性，LLM引入了关键的安全和风险考虑。这些挑战值得认真审查，以确保负责任地部署和防范潜在漏洞。本研究论文从五个主题角度彻底调查了与LLM相关的安全和隐私问题：安全和隐私问题，对抗攻击的漏洞，滥用LLM造成的潜在危害，缓解策略，以解决这些挑战，同时确定当前策略的局限性。最后，本文建议了未来研究的有希望的途径，以加强LLM的安全性和风险管理。



## **30. Boosting Transferability in Vision-Language Attacks via Diversification along the Intersection Region of Adversarial Trajectory**

基于对抗轨迹交叉区域的多样化提高视觉语言攻击的可传递性 cs.CV

**SubmitDate**: 2024-03-19    [abs](http://arxiv.org/abs/2403.12445v1) [paper-pdf](http://arxiv.org/pdf/2403.12445v1)

**Authors**: Sensen Gao, Xiaojun Jia, Xuhong Ren, Ivor Tsang, Qing Guo

**Abstract**: Vision-language pre-training (VLP) models exhibit remarkable capabilities in comprehending both images and text, yet they remain susceptible to multimodal adversarial examples (AEs). Strengthening adversarial attacks and uncovering vulnerabilities, especially common issues in VLP models (e.g., high transferable AEs), can stimulate further research on constructing reliable and practical VLP models. A recent work (i.e., Set-level guidance attack) indicates that augmenting image-text pairs to increase AE diversity along the optimization path enhances the transferability of adversarial examples significantly. However, this approach predominantly emphasizes diversity around the online adversarial examples (i.e., AEs in the optimization period), leading to the risk of overfitting the victim model and affecting the transferability. In this study, we posit that the diversity of adversarial examples towards the clean input and online AEs are both pivotal for enhancing transferability across VLP models. Consequently, we propose using diversification along the intersection region of adversarial trajectory to expand the diversity of AEs. To fully leverage the interaction between modalities, we introduce text-guided adversarial example selection during optimization. Furthermore, to further mitigate the potential overfitting, we direct the adversarial text deviating from the last intersection region along the optimization path, rather than adversarial images as in existing methods. Extensive experiments affirm the effectiveness of our method in improving transferability across various VLP models and downstream vision-and-language tasks (e.g., Image-Text Retrieval(ITR), Visual Grounding(VG), Image Captioning(IC)).

摘要: 视觉语言预训练（VLP）模型在理解图像和文本方面表现出非凡的能力，但它们仍然容易受到多模态对抗示例（AE）的影响。加强对抗性攻击和发现漏洞，特别是VLP模型中的常见问题（例如，高可传递性AE），可以刺激构建可靠和实用的VLP模型的进一步研究。最近的作品（即，集级引导攻击）表明，增加图像—文本对以增加AE多样性沿优化路径显著增强了对抗样本的可移植性。然而，这种方法主要强调围绕在线对抗性示例的多样性（即，最优化期的AE），导致受害者模型过度拟合并影响可转移性的风险。在这项研究中，我们认为对抗性的例子对清洁输入和在线AE的多样性都是提高跨VLP模型的可移植性的关键。因此，我们建议使用沿对抗轨迹交叉区域的多样化来扩大AE的多样性。为了充分利用模态之间的交互，我们在优化过程中引入了文本引导的对抗性示例选择。此外，为了进一步减少潜在的过拟合，我们引导对抗文本偏离沿着优化路径的最后一个交叉区域，而不是像现有方法中的对抗图像。大量的实验证实了我们的方法在提高跨各种VLP模型和下游视觉和语言任务的可移植性方面的有效性（例如，图像—文本检索（ITR），视觉基础（VG），图像字幕（IC））。



## **31. Algorithmic Complexity Attacks on Dynamic Learned Indexes**

动态学习索引的复杂性攻击 cs.DB

VLDB 2024

**SubmitDate**: 2024-03-19    [abs](http://arxiv.org/abs/2403.12433v1) [paper-pdf](http://arxiv.org/pdf/2403.12433v1)

**Authors**: Rui Yang, Evgenios M. Kornaropoulos, Yue Cheng

**Abstract**: Learned Index Structures (LIS) view a sorted index as a model that learns the data distribution, takes a data element key as input, and outputs the predicted position of the key. The original LIS can only handle lookup operations with no support for updates, rendering it impractical to use for typical workloads. To address this limitation, recent studies have focused on designing efficient dynamic learned indexes. ALEX, as the pioneering dynamic learned index structures, enables dynamism by incorporating a series of design choices, including adaptive key space partitioning, dynamic model retraining, and sophisticated engineering and policies that prioritize read/write performance. While these design choices offer improved average-case performance, the emphasis on flexibility and performance increases the attack surface by allowing adversarial behaviors that maximize ALEX's memory space and time complexity in worst-case scenarios. In this work, we present the first systematic investigation of algorithmic complexity attacks (ACAs) targeting the worst-case scenarios of ALEX. We introduce new ACAs that fall into two categories, space ACAs and time ACAs, which target the memory space and time complexity, respectively. First, our space ACA on data nodes exploits ALEX's gapped array layout and uses Multiple-Choice Knapsack (MCK) to generate an optimal adversarial insertion plan for maximizing the memory consumption at the data node level. Second, our space ACA on internal nodes exploits ALEX's catastrophic cost mitigation mechanism, causing an out-of-memory error with only a few hundred adversarial insertions. Third, our time ACA generates pathological insertions to increase the disparity between the actual key distribution and the linear models of data nodes, deteriorating the runtime performance by up to 1,641X compared to ALEX operating under legitimate workloads.

摘要: Learned Index Structures（LIS）将排序索引视为学习数据分布的模型，以数据元素键作为输入，并输出键的预测位置。原始LIS只能处理查找操作，不支持更新，这使得它不适合用于典型的工作负载。为了解决这一局限性，最近的研究集中在设计有效的动态学习索引。ALEX作为一种先驱性的动态学习索引结构，通过整合一系列设计选择，包括自适应密钥空间划分、动态模型再训练以及优先考虑读写性能的复杂工程和策略，实现动态。虽然这些设计选择提供了改进的平均情况下性能，但对灵活性和性能的强调通过允许对抗行为来增加攻击面，在最坏情况下最大化ALEX的内存空间和时间复杂性。在这项工作中，我们提出了第一个系统的调查算法复杂性攻击（ACA）针对最坏情况的情况下的ALEX。我们引入了新的ACA，属于两类，空间ACA和时间ACA，分别针对存储空间和时间复杂度。首先，我们在数据节点上的空间ACA利用了ALEX的间隙阵列布局，并使用多选择背包（MCK）来生成一个最优的对抗插入计划，以最大化数据节点级别的内存消耗。第二，我们在内部节点上的空间ACA利用了ALEX的灾难性成本缓解机制，仅用几百个对抗性插入就导致内存不足错误。第三，ACA生成病态插入，增加了实际密钥分布和数据节点线性模型之间的差异，与在合法工作负载下运行的ALEX相比，运行时性能下降了1641倍。



## **32. Electioneering the Network: Dynamic Multi-Step Adversarial Attacks for Community Canvassing**

网络选举：社区游说的动态多步对抗攻击 cs.LG

**SubmitDate**: 2024-03-19    [abs](http://arxiv.org/abs/2403.12399v1) [paper-pdf](http://arxiv.org/pdf/2403.12399v1)

**Authors**: Saurabh Sharma, Ambuj SIngh

**Abstract**: The problem of online social network manipulation for community canvassing is of real concern in today's world. Motivated by the study of voter models, opinion and polarization dynamics on networks, we model community canvassing as a dynamic process over a network enabled via gradient-based attacks on GNNs. Existing attacks on GNNs are all single-step and do not account for the dynamic cascading nature of information diffusion in networks. We consider the realistic scenario where an adversary uses a GNN as a proxy to predict and manipulate voter preferences, especially uncertain voters. Gradient-based attacks on the GNN inform the adversary of strategic manipulations that can be made to proselytize targeted voters. In particular, we explore $\textit{minimum budget attacks for community canvassing}$ (MBACC). We show that the MBACC problem is NP-Hard and propose Dynamic Multi-Step Adversarial Community Canvassing (MAC) to address it. MAC makes dynamic local decisions based on the heuristic of low budget and high second-order influence to convert and perturb target voters. MAC is a dynamic multi-step attack that discovers low-budget and high-influence targets from which efficient cascading attacks can happen. We evaluate MAC against single-step baselines on the MBACC problem with multiple underlying networks and GNN models. Our experiments show the superiority of MAC which is able to discover efficient multi-hop attacks for adversarial community canvassing. Our code implementation and data is available at https://github.com/saurabhsharma1993/mac.

摘要: 在当今世界，为社区拉票而操纵在线社交网络的问题是真正令人关注的问题。基于对网络上的选民模型、意见和极化动态的研究，我们将社区拉票建模为一个动态过程，通过基于梯度的攻击GNNs。现有的GNN攻击都是一步式的，没有考虑到网络中信息扩散的动态级联性质。我们考虑现实的情况下，对手使用GNN作为代理来预测和操纵选民的偏好，特别是不确定的选民。对GNN的攻击使对手了解了可以使目标选民改变信仰的战略操纵。特别是，我们探索了$\textit {社区游说的最小预算攻击}$（MBACC）。本文证明了MBACC问题是NP—难问题，提出了动态多步对抗社区游说（MAC）算法，该算法基于低预算和高二阶影响力的启发式进行动态局部决策，以转移和干扰目标选民。MAC是一种动态的多步攻击，它发现低预算和高影响力的目标，从这些目标可以发生有效的级联攻击。我们对多个底层网络和GNN模型的MBACC问题的一步基线评估MAC。我们的实验表明了MAC的优越性，它能够发现有效的多跳攻击对抗性社区游说。我们的代码实现和数据可在www.example.com获得。



## **33. Securely Fine-tuning Pre-trained Encoders Against Adversarial Examples**

针对对抗性示例的预训练编码器进行保密微调 cs.CV

**SubmitDate**: 2024-03-19    [abs](http://arxiv.org/abs/2403.10801v2) [paper-pdf](http://arxiv.org/pdf/2403.10801v2)

**Authors**: Ziqi Zhou, Minghui Li, Wei Liu, Shengshan Hu, Yechao Zhang, Wei Wan, Lulu Xue, Leo Yu Zhang, Dezhong Yao, Hai Jin

**Abstract**: With the evolution of self-supervised learning, the pre-training paradigm has emerged as a predominant solution within the deep learning landscape. Model providers furnish pre-trained encoders designed to function as versatile feature extractors, enabling downstream users to harness the benefits of expansive models with minimal effort through fine-tuning. Nevertheless, recent works have exposed a vulnerability in pre-trained encoders, highlighting their susceptibility to downstream-agnostic adversarial examples (DAEs) meticulously crafted by attackers. The lingering question pertains to the feasibility of fortifying the robustness of downstream models against DAEs, particularly in scenarios where the pre-trained encoders are publicly accessible to the attackers.   In this paper, we initially delve into existing defensive mechanisms against adversarial examples within the pre-training paradigm. Our findings reveal that the failure of current defenses stems from the domain shift between pre-training data and downstream tasks, as well as the sensitivity of encoder parameters. In response to these challenges, we propose Genetic Evolution-Nurtured Adversarial Fine-tuning (Gen-AF), a two-stage adversarial fine-tuning approach aimed at enhancing the robustness of downstream models. Our extensive experiments, conducted across ten self-supervised training methods and six datasets, demonstrate that Gen-AF attains high testing accuracy and robust testing accuracy against state-of-the-art DAEs.

摘要: 随着自我监督学习的发展，预训练范式已经成为深度学习领域的主要解决方案。模型提供商提供预先训练的编码器，设计为功能丰富的特征提取器，使下游用户能够通过微调以最小的努力利用扩展模型的好处。尽管如此，最近的工作暴露了预训练编码器中的一个漏洞，突出了它们对攻击者精心设计的下游不可知对抗示例（DAE）的敏感性。这个悬而未决的问题涉及针对DAE加强下游模型的鲁棒性的可行性，特别是在攻击者可以公开访问预训练编码器的情况下。   在本文中，我们首先深入研究了现有的防御机制，对抗预训练范式中的对抗示例。我们的研究结果表明，当前防御的失败源于预训练数据和下游任务之间的域转移，以及编码器参数的敏感性。为了应对这些挑战，我们提出了遗传进化培育对抗微调（Gen—AF），一种两阶段对抗微调方法，旨在增强下游模型的鲁棒性。我们在10种自我监督训练方法和6个数据集上进行的广泛实验表明，Gen—AF在最先进的DAE上实现了高测试精度和稳健的测试精度。



## **34. Improving Visual Quality and Transferability of Adversarial Attacks on Face Recognition Simultaneously with Adversarial Restoration**

对抗恢复同时提高对抗攻击的视觉质量和可传递性 cs.CV

\copyright 2023 IEEE. Personal use of this material is permitted.  Permission from IEEE must be obtained for all other uses, in any current or  future media, including reprinting/republishing this material for advertising  or promotional purposes, creating new collective works, for resale or  redistribution to servers or lists, or reuse of any copyrighted component of  this work in other works

**SubmitDate**: 2024-03-19    [abs](http://arxiv.org/abs/2309.01582v4) [paper-pdf](http://arxiv.org/pdf/2309.01582v4)

**Authors**: Fengfan Zhou, Hefei Ling, Yuxuan Shi, Jiazhong Chen, Ping Li

**Abstract**: Adversarial face examples possess two critical properties: Visual Quality and Transferability. However, existing approaches rarely address these properties simultaneously, leading to subpar results. To address this issue, we propose a novel adversarial attack technique known as Adversarial Restoration (AdvRestore), which enhances both visual quality and transferability of adversarial face examples by leveraging a face restoration prior. In our approach, we initially train a Restoration Latent Diffusion Model (RLDM) designed for face restoration. Subsequently, we employ the inference process of RLDM to generate adversarial face examples. The adversarial perturbations are applied to the intermediate features of RLDM. Additionally, by treating RLDM face restoration as a sibling task, the transferability of the generated adversarial face examples is further improved. Our experimental results validate the effectiveness of the proposed attack method.

摘要: 对抗性人脸样本具有两个关键属性：视觉质量和可移植性。然而，现有的方法很少同时解决这些属性，导致低于标准的结果。为了解决这个问题，我们提出了一种新的对抗攻击技术，称为对抗恢复（AdvRestore），它提高了视觉质量和转移性的对抗人脸样本利用人脸恢复之前。在我们的方法中，我们首先训练一个恢复潜在扩散模型（RLDM）设计的面部修复。随后，我们使用RLDM的推理过程来生成对抗人脸样本。对抗扰动被应用于RLDM的中间特征。此外，通过将RLDM人脸恢复作为兄弟任务处理，进一步提高了生成的对抗人脸样本的可移植性。实验结果验证了该攻击方法的有效性。



## **35. Large language models in 6G security: challenges and opportunities**

6G安全中的大语言模型：挑战和机遇 cs.CR

29 pages, 2 figures

**SubmitDate**: 2024-03-18    [abs](http://arxiv.org/abs/2403.12239v1) [paper-pdf](http://arxiv.org/pdf/2403.12239v1)

**Authors**: Tri Nguyen, Huong Nguyen, Ahmad Ijaz, Saeid Sheikhi, Athanasios V. Vasilakos, Panos Kostakos

**Abstract**: The rapid integration of Generative AI (GenAI) and Large Language Models (LLMs) in sectors such as education and healthcare have marked a significant advancement in technology. However, this growth has also led to a largely unexplored aspect: their security vulnerabilities. As the ecosystem that includes both offline and online models, various tools, browser plugins, and third-party applications continues to expand, it significantly widens the attack surface, thereby escalating the potential for security breaches. These expansions in the 6G and beyond landscape provide new avenues for adversaries to manipulate LLMs for malicious purposes. We focus on the security aspects of LLMs from the viewpoint of potential adversaries. We aim to dissect their objectives and methodologies, providing an in-depth analysis of known security weaknesses. This will include the development of a comprehensive threat taxonomy, categorizing various adversary behaviors. Also, our research will concentrate on how LLMs can be integrated into cybersecurity efforts by defense teams, also known as blue teams. We will explore the potential synergy between LLMs and blockchain technology, and how this combination could lead to the development of next-generation, fully autonomous security solutions. This approach aims to establish a unified cybersecurity strategy across the entire computing continuum, enhancing overall digital security infrastructure.

摘要: 生成人工智能（GenAI）和大型语言模型（LLM）在教育和医疗保健等领域的快速集成标志着技术的重大进步。然而，这种增长也导致了一个基本上未被探索的方面：他们的安全漏洞。随着包括离线和在线模式、各种工具、浏览器插件和第三方应用程序在内的生态系统不断扩大，攻击面显著扩大，从而增加了安全漏洞的可能性。6G及其他领域的这些扩展为对手操纵LLM以达到恶意目的提供了新的途径。我们从潜在对手的角度关注LLM的安全方面。我们的目标是剖析他们的目标和方法，对已知的安全弱点进行深入分析。这将包括开发一个全面的威胁分类，对各种攻击者行为进行分类。此外，我们的研究将集中在如何将LLM整合到防御团队（也称为蓝队）的网络安全工作中。我们将探索LLM和区块链技术之间的潜在协同作用，以及这种结合如何导致下一代完全自主的安全解决方案的开发。该方法旨在在整个计算连续体中建立统一的网络安全战略，增强整体数字安全基础设施。



## **36. Adversarial Training Should Be Cast as a Non-Zero-Sum Game**

对抗性训练应被视为非零和博弈 cs.LG

**SubmitDate**: 2024-03-18    [abs](http://arxiv.org/abs/2306.11035v2) [paper-pdf](http://arxiv.org/pdf/2306.11035v2)

**Authors**: Alexander Robey, Fabian Latorre, George J. Pappas, Hamed Hassani, Volkan Cevher

**Abstract**: One prominent approach toward resolving the adversarial vulnerability of deep neural networks is the two-player zero-sum paradigm of adversarial training, in which predictors are trained against adversarially chosen perturbations of data. Despite the promise of this approach, algorithms based on this paradigm have not engendered sufficient levels of robustness and suffer from pathological behavior like robust overfitting. To understand this shortcoming, we first show that the commonly used surrogate-based relaxation used in adversarial training algorithms voids all guarantees on the robustness of trained classifiers. The identification of this pitfall informs a novel non-zero-sum bilevel formulation of adversarial training, wherein each player optimizes a different objective function. Our formulation yields a simple algorithmic framework that matches and in some cases outperforms state-of-the-art attacks, attains comparable levels of robustness to standard adversarial training algorithms, and does not suffer from robust overfitting.

摘要: 解决深度神经网络对抗脆弱性的一个突出方法是对抗训练的两人零和范式，其中预测器是针对对抗性选择的数据扰动进行训练的。尽管这种方法有希望，但基于这种范式的算法并没有产生足够水平的鲁棒性，并且会遭受像鲁棒过拟合这样的病态行为。为了理解这一缺点，我们首先表明，对抗训练算法中常用的基于代理的松弛方法会使训练过的分类器鲁棒性的所有保证无效。这个陷阱的识别通知了一个新的非零和双水平的对抗训练公式，其中每个球员优化不同的目标函数。我们的公式产生了一个简单的算法框架，匹配并在某些情况下优于最先进的攻击，达到了与标准对抗训练算法相当的鲁棒性水平，并且不会遭受鲁棒过拟合。



## **37. Diffusion Denoising as a Certified Defense against Clean-label Poisoning**

扩散去噪作为清洁标签中毒的认证防御 cs.CR

**SubmitDate**: 2024-03-18    [abs](http://arxiv.org/abs/2403.11981v1) [paper-pdf](http://arxiv.org/pdf/2403.11981v1)

**Authors**: Sanghyun Hong, Nicholas Carlini, Alexey Kurakin

**Abstract**: We present a certified defense to clean-label poisoning attacks. These attacks work by injecting a small number of poisoning samples (e.g., 1%) that contain $p$-norm bounded adversarial perturbations into the training data to induce a targeted misclassification of a test-time input. Inspired by the adversarial robustness achieved by $denoised$ $smoothing$, we show how an off-the-shelf diffusion model can sanitize the tampered training data. We extensively test our defense against seven clean-label poisoning attacks and reduce their attack success to 0-16% with only a negligible drop in the test time accuracy. We compare our defense with existing countermeasures against clean-label poisoning, showing that the defense reduces the attack success the most and offers the best model utility. Our results highlight the need for future work on developing stronger clean-label attacks and using our certified yet practical defense as a strong baseline to evaluate these attacks.

摘要: 我们提出了一个认证的防御清洁标签中毒攻击。这些攻击通过注入少量中毒样本（例如，1%），其中包含$p $范数有界对抗扰动到训练数据中，以诱导测试时输入的有针对性的错误分类。受$去噪$$平滑$所实现的对抗鲁棒性的启发，我们展示了一个现成的扩散模型如何可以净化篡改的训练数据。我们广泛测试了我们对七种清洁标签中毒攻击的防御，并将其攻击成功率降低到0—16%，测试时间准确性只有微不足道的下降。我们将我们的防御与现有的针对清洁标签中毒的对策进行了比较，表明该防御最大程度地减少了攻击成功，并提供了最佳的模型效用。我们的研究结果强调，未来需要开发更强的清洁标签攻击，并使用我们认证但实用的防御作为评估这些攻击的强有力基线。



## **38. Enhancing the Antidote: Improved Pointwise Certifications against Poisoning Attacks**

增强解毒剂：改进针对中毒攻击的逐点认证 cs.LG

**SubmitDate**: 2024-03-18    [abs](http://arxiv.org/abs/2308.07553v2) [paper-pdf](http://arxiv.org/pdf/2308.07553v2)

**Authors**: Shijie Liu, Andrew C. Cullen, Paul Montague, Sarah M. Erfani, Benjamin I. P. Rubinstein

**Abstract**: Poisoning attacks can disproportionately influence model behaviour by making small changes to the training corpus. While defences against specific poisoning attacks do exist, they in general do not provide any guarantees, leaving them potentially countered by novel attacks. In contrast, by examining worst-case behaviours Certified Defences make it possible to provide guarantees of the robustness of a sample against adversarial attacks modifying a finite number of training samples, known as pointwise certification. We achieve this by exploiting both Differential Privacy and the Sampled Gaussian Mechanism to ensure the invariance of prediction for each testing instance against finite numbers of poisoned examples. In doing so, our model provides guarantees of adversarial robustness that are more than twice as large as those provided by prior certifications.

摘要: 中毒攻击可以通过对训练语料库进行微小的更改来不成比例地影响模型行为。虽然确实存在针对特定中毒攻击的防御措施，但它们通常不提供任何保证，从而可能被新的攻击所抵消。相比之下，通过检查最坏情况下的行为，Certified Defences可以保证样本对修改有限数量训练样本的对抗攻击的鲁棒性，称为逐点认证。我们通过利用差分隐私和采样高斯机制来实现这一点，以确保每个测试实例对有限数量的中毒实例的预测不变性。在这样做的过程中，我们的模型提供了对抗性鲁棒性的保证，其保证是先前认证所提供的保证的两倍多。



## **39. SSCAE -- Semantic, Syntactic, and Context-aware natural language Adversarial Examples generator**

SSCAE--语义、句法和上下文感知的自然语言对抗性实例生成器 cs.CL

**SubmitDate**: 2024-03-18    [abs](http://arxiv.org/abs/2403.11833v1) [paper-pdf](http://arxiv.org/pdf/2403.11833v1)

**Authors**: Javad Rafiei Asl, Mohammad H. Rafiei, Manar Alohaly, Daniel Takabi

**Abstract**: Machine learning models are vulnerable to maliciously crafted Adversarial Examples (AEs). Training a machine learning model with AEs improves its robustness and stability against adversarial attacks. It is essential to develop models that produce high-quality AEs. Developing such models has been much slower in natural language processing (NLP) than in areas such as computer vision. This paper introduces a practical and efficient adversarial attack model called SSCAE for \textbf{S}emantic, \textbf{S}yntactic, and \textbf{C}ontext-aware natural language \textbf{AE}s generator. SSCAE identifies important words and uses a masked language model to generate an early set of substitutions. Next, two well-known language models are employed to evaluate the initial set in terms of semantic and syntactic characteristics. We introduce (1) a dynamic threshold to capture more efficient perturbations and (2) a local greedy search to generate high-quality AEs. As a black-box method, SSCAE generates humanly imperceptible and context-aware AEs that preserve semantic consistency and the source language's syntactical and grammatical requirements. The effectiveness and superiority of the proposed SSCAE model are illustrated with fifteen comparative experiments and extensive sensitivity analysis for parameter optimization. SSCAE outperforms the existing models in all experiments while maintaining a higher semantic consistency with a lower query number and a comparable perturbation rate.

摘要: 机器学习模型容易受到恶意构建的对抗示例（AE）的影响。使用AE训练机器学习模型可以提高其对抗性攻击的鲁棒性和稳定性。必须开发产生高质量AE的模型。在自然语言处理（NLP）中，开发此类模型的速度比计算机视觉等领域慢得多。本文针对\textBF {S}语义、\textBF {S}语义和\textBF {C}上下文感知自然语言\textBF {AE}生成器，提出了一种实用高效的对抗攻击模型SSCAE。SSCAE识别重要的单词，并使用掩蔽语言模型生成早期的替换集。接下来，两个著名的语言模型被用来评估初始集的语义和句法特征。我们引入（1）动态阈值来捕获更有效的扰动和（2）局部贪婪搜索来生成高质量的AE。作为一种黑箱方法，SSCAE生成了人类无法感知的上下文感知AE，以保持语义一致性和源语言的句法和语法要求。通过15个对比试验和广泛的参数优化灵敏度分析，说明了所提出的SSCAE模型的有效性和优越性。SSCAE在所有实验中都优于现有模型，同时保持较高的语义一致性，查询次数较低，扰动率相当。



## **40. Problem space structural adversarial attacks for Network Intrusion Detection Systems based on Graph Neural Networks**

基于图神经网络的网络入侵检测系统的问题空间结构对抗攻击 cs.CR

preprint submitted to IEEE TIFS, under review

**SubmitDate**: 2024-03-18    [abs](http://arxiv.org/abs/2403.11830v1) [paper-pdf](http://arxiv.org/pdf/2403.11830v1)

**Authors**: Andrea Venturi, Dario Stabili, Mirco Marchetti

**Abstract**: Machine Learning (ML) algorithms have become increasingly popular for supporting Network Intrusion Detection Systems (NIDS). Nevertheless, extensive research has shown their vulnerability to adversarial attacks, which involve subtle perturbations to the inputs of the models aimed at compromising their performance. Recent proposals have effectively leveraged Graph Neural Networks (GNN) to produce predictions based also on the structural patterns exhibited by intrusions to enhance the detection robustness. However, the adoption of GNN-based NIDS introduces new types of risks. In this paper, we propose the first formalization of adversarial attacks specifically tailored for GNN in network intrusion detection. Moreover, we outline and model the problem space constraints that attackers need to consider to carry out feasible structural attacks in real-world scenarios. As a final contribution, we conduct an extensive experimental campaign in which we launch the proposed attacks against state-of-the-art GNN-based NIDS. Our findings demonstrate the increased robustness of the models against classical feature-based adversarial attacks, while highlighting their susceptibility to structure-based attacks.

摘要: 机器学习（ML）算法在支持网络入侵检测系统（NIDS）方面已经变得越来越受欢迎。然而，广泛的研究表明，它们容易受到对抗性攻击，这种攻击涉及对模型输入的微妙扰动，旨在损害其性能。最近的建议已经有效地利用图神经网络（GNN）来产生预测，也基于入侵表现出的结构模式，以增强检测的鲁棒性。然而，采用基于GNN的NIDS引入了新类型的风险。在本文中，我们提出了第一个形式化的对抗攻击专门为GNN在网络入侵检测。此外，我们概述和建模的问题空间约束，攻击者需要考虑进行可行的结构性攻击在现实世界的场景。作为最后的贡献，我们进行了一个广泛的实验活动，在该活动中，我们发起了针对最先进的基于GNN的NIDS的攻击。我们的研究结果表明，这些模型对经典的基于特征的对抗性攻击的鲁棒性增强，同时突出了它们对基于结构的攻击的敏感性。



## **41. Expressive Losses for Verified Robustness via Convex Combinations**

基于凸组合的鲁棒性验证的表达损失 cs.LG

ICLR 2024

**SubmitDate**: 2024-03-18    [abs](http://arxiv.org/abs/2305.13991v3) [paper-pdf](http://arxiv.org/pdf/2305.13991v3)

**Authors**: Alessandro De Palma, Rudy Bunel, Krishnamurthy Dvijotham, M. Pawan Kumar, Robert Stanforth, Alessio Lomuscio

**Abstract**: In order to train networks for verified adversarial robustness, it is common to over-approximate the worst-case loss over perturbation regions, resulting in networks that attain verifiability at the expense of standard performance. As shown in recent work, better trade-offs between accuracy and robustness can be obtained by carefully coupling adversarial training with over-approximations. We hypothesize that the expressivity of a loss function, which we formalize as the ability to span a range of trade-offs between lower and upper bounds to the worst-case loss through a single parameter (the over-approximation coefficient), is key to attaining state-of-the-art performance. To support our hypothesis, we show that trivial expressive losses, obtained via convex combinations between adversarial attacks and IBP bounds, yield state-of-the-art results across a variety of settings in spite of their conceptual simplicity. We provide a detailed analysis of the relationship between the over-approximation coefficient and performance profiles across different expressive losses, showing that, while expressivity is essential, better approximations of the worst-case loss are not necessarily linked to superior robustness-accuracy trade-offs.

摘要: 为了训练网络以获得验证的对抗鲁棒性，通常会过度近似扰动区域的最坏情况损失，导致网络以牺牲标准性能为代价获得可验证性。正如最近的工作所示，通过谨慎地将对抗训练与过近似耦合，可以在准确性和鲁棒性之间获得更好的权衡。我们假设损失函数的表现性，我们正式化为通过单个参数（过近似系数）在最坏情况下损失的下限和上限之间进行权衡的能力，是获得最先进性能的关键。为了支持我们的假设，我们表明，微不足道的表达损失，通过对抗攻击和IBP边界之间的凸组合，产生国家的最先进的结果，在各种设置，尽管其概念简单。我们提供了一个详细的关系的过近似系数和性能配置文件跨不同的表达损失，表明，虽然表达是必不可少的，更好的近似最坏情况下损失不一定与优越的鲁棒性准确性权衡。



## **42. Hidden in Plain Sight: Undetectable Adversarial Bias Attacks on Vulnerable Patient Populations**

隐藏在普通视野中：对脆弱患者人群的不可检测的对抗偏见攻击 cs.LG

29 pages, 4 figures

**SubmitDate**: 2024-03-18    [abs](http://arxiv.org/abs/2402.05713v2) [paper-pdf](http://arxiv.org/pdf/2402.05713v2)

**Authors**: Pranav Kulkarni, Andrew Chan, Nithya Navarathna, Skylar Chan, Paul H. Yi, Vishwa S. Parekh

**Abstract**: The proliferation of artificial intelligence (AI) in radiology has shed light on the risk of deep learning (DL) models exacerbating clinical biases towards vulnerable patient populations. While prior literature has focused on quantifying biases exhibited by trained DL models, demographically targeted adversarial bias attacks on DL models and its implication in the clinical environment remains an underexplored field of research in medical imaging. In this work, we demonstrate that demographically targeted label poisoning attacks can introduce undetectable underdiagnosis bias in DL models. Our results across multiple performance metrics and demographic groups like sex, age, and their intersectional subgroups show that adversarial bias attacks demonstrate high-selectivity for bias in the targeted group by degrading group model performance without impacting overall model performance. Furthermore, our results indicate that adversarial bias attacks result in biased DL models that propagate prediction bias even when evaluated with external datasets.

摘要: 人工智能(AI)在放射学中的扩散揭示了深度学习(DL)模型的风险，加剧了对脆弱患者群体的临床偏见。虽然以前的文献集中于量化训练的DL模型所表现出的偏差，但针对人口统计目标的对DL模型的对抗性偏见攻击及其在临床环境中的应用仍然是医学成像领域中探索不足的研究领域。我们在多个性能指标和人口统计组(如性别、年龄及其相交的子组)上的结果表明，对抗性偏见攻击通过降低组模型性能而不影响整体模型性能，显示了对目标组中的偏见的高选择性。



## **43. Stop Reasoning! When Multimodal LLMs with Chain-of-Thought Reasoning Meets Adversarial Images**

停止推理！当多模态LLM的思想链推理遇到对抗图像时 cs.CV

**SubmitDate**: 2024-03-18    [abs](http://arxiv.org/abs/2402.14899v2) [paper-pdf](http://arxiv.org/pdf/2402.14899v2)

**Authors**: Zefeng Wang, Zhen Han, Shuo Chen, Fan Xue, Zifeng Ding, Xun Xiao, Volker Tresp, Philip Torr, Jindong Gu

**Abstract**: Recently, Multimodal LLMs (MLLMs) have shown a great ability to understand images. However, like traditional vision models, they are still vulnerable to adversarial images. Meanwhile, Chain-of-Thought (CoT) reasoning has been widely explored on MLLMs, which not only improves model's performance, but also enhances model's explainability by giving intermediate reasoning steps. Nevertheless, there is still a lack of study regarding MLLMs' adversarial robustness with CoT and an understanding of what the rationale looks like when MLLMs infer wrong answers with adversarial images. Our research evaluates the adversarial robustness of MLLMs when employing CoT reasoning, finding that CoT marginally improves adversarial robustness against existing attack methods. Moreover, we introduce a novel stop-reasoning attack technique that effectively bypasses the CoT-induced robustness enhancements. Finally, we demonstrate the alterations in CoT reasoning when MLLMs confront adversarial images, shedding light on their reasoning process under adversarial attacks.

摘要: 近年来，多模式LLMS(多模式LLMS)显示出了很强的图像理解能力。然而，像传统的视觉模型一样，它们仍然容易受到敌意图像的影响。同时，思维链式推理在MLLMS上得到了广泛的探索，它不仅改善了模型的性能，而且通过给出中间推理步骤来增强模型的可解释性。然而，仍然缺乏关于MLLMS在COT下的对抗性鲁棒性的研究，以及对MLLMS用对抗性图像推断错误答案的基本原理的理解。我们的研究评估了MLLMS在使用CoT推理时的对抗健壮性，发现CoT略微提高了对现有攻击方法的对抗健壮性。此外，我们引入了一种新的停止推理攻击技术，该技术有效地绕过了CoT诱导的健壮性增强。最后，我们展示了当MLLMS面对对抗性图像时，COT推理的变化，揭示了它们在对抗性攻击下的推理过程。



## **44. LocalStyleFool: Regional Video Style Transfer Attack Using Segment Anything Model**

LocalStyleFool：基于段任意模型的区域视频风格转移攻击 cs.CV

Accepted to 2024 IEEE Security and Privacy Workshops (SPW)

**SubmitDate**: 2024-03-18    [abs](http://arxiv.org/abs/2403.11656v1) [paper-pdf](http://arxiv.org/pdf/2403.11656v1)

**Authors**: Yuxin Cao, Jinghao Li, Xi Xiao, Derui Wang, Minhui Xue, Hao Ge, Wei Liu, Guangwu Hu

**Abstract**: Previous work has shown that well-crafted adversarial perturbations can threaten the security of video recognition systems. Attackers can invade such models with a low query budget when the perturbations are semantic-invariant, such as StyleFool. Despite the query efficiency, the naturalness of the minutia areas still requires amelioration, since StyleFool leverages style transfer to all pixels in each frame. To close the gap, we propose LocalStyleFool, an improved black-box video adversarial attack that superimposes regional style-transfer-based perturbations on videos. Benefiting from the popularity and scalably usability of Segment Anything Model (SAM), we first extract different regions according to semantic information and then track them through the video stream to maintain the temporal consistency. Then, we add style-transfer-based perturbations to several regions selected based on the associative criterion of transfer-based gradient information and regional area. Perturbation fine adjustment is followed to make stylized videos adversarial. We demonstrate that LocalStyleFool can improve both intra-frame and inter-frame naturalness through a human-assessed survey, while maintaining competitive fooling rate and query efficiency. Successful experiments on the high-resolution dataset also showcase that scrupulous segmentation of SAM helps to improve the scalability of adversarial attacks under high-resolution data.

摘要: 先前的研究表明，精心设计的对抗性扰动可能威胁到视频识别系统的安全性。当扰动是语义不变的时，攻击者可以以低查询预算入侵此类模型，例如StyleFool。尽管查询效率很高，细节区域的自然度仍然需要改进，因为StyleFool利用了对每帧中所有像素的风格传输。为了缩小差距，我们提出了LocalStyleFool，一种改进的黑盒视频对抗攻击，它在视频上叠加了基于区域风格转移的扰动。利用Segment Anything Model（SAM）的流行性和可扩展性，首先根据语义信息提取不同的区域，然后通过视频流跟踪它们，以保持时间一致性。然后，我们添加基于风格转移的扰动到几个基于转移的梯度信息和区域面积的关联准则的选择的区域。微扰微调是遵循的，使风格化的视频对抗。我们证明了LocalStyleFool可以提高帧内和帧间的自然度，通过一个人工评估的调查，同时保持有竞争力的愚弄率和查询效率。在高分辨率数据集上的成功实验也表明，SAM的精确分割有助于提高高分辨率数据下对抗性攻击的可扩展性。



## **45. Zeroth-Order Hard-Thresholding: Gradient Error vs. Expansivity**

零阶硬保持：梯度误差与扩展性 cs.LG

Accepted for publication at NeurIPS 2022

**SubmitDate**: 2024-03-18    [abs](http://arxiv.org/abs/2210.05279v2) [paper-pdf](http://arxiv.org/pdf/2210.05279v2)

**Authors**: William de Vazelhes, Hualin Zhang, Huimin Wu, Xiao-Tong Yuan, Bin Gu

**Abstract**: $\ell_0$ constrained optimization is prevalent in machine learning, particularly for high-dimensional problems, because it is a fundamental approach to achieve sparse learning. Hard-thresholding gradient descent is a dominant technique to solve this problem. However, first-order gradients of the objective function may be either unavailable or expensive to calculate in a lot of real-world problems, where zeroth-order (ZO) gradients could be a good surrogate. Unfortunately, whether ZO gradients can work with the hard-thresholding operator is still an unsolved problem. To solve this puzzle, in this paper, we focus on the $\ell_0$ constrained black-box stochastic optimization problems, and propose a new stochastic zeroth-order gradient hard-thresholding (SZOHT) algorithm with a general ZO gradient estimator powered by a novel random support sampling. We provide the convergence analysis of SZOHT under standard assumptions. Importantly, we reveal a conflict between the deviation of ZO estimators and the expansivity of the hard-thresholding operator, and provide a theoretical minimal value of the number of random directions in ZO gradients. In addition, we find that the query complexity of SZOHT is independent or weakly dependent on the dimensionality under different settings. Finally, we illustrate the utility of our method on a portfolio optimization problem as well as black-box adversarial attacks.

摘要: 约束优化是实现稀疏学习的基本途径，在机器学习中得到了广泛的应用，特别是对于高维问题。硬阈值梯度下降是解决这一问题的主流技术。然而，在许多实际问题中，目标函数的一阶梯度可能无法获得或计算成本很高，其中零阶(ZO)梯度可能是一个很好的替代。遗憾的是，ZO梯度是否能与硬阈值算子一起工作，仍然是一个悬而未决的问题。为解决这一难题，本文以0元约束黑箱随机优化问题为研究对象，提出了一种新的随机零阶梯度硬阈值算法(SZOHT)，该算法采用了一种新的随机支持抽样的广义ZO梯度估值器。在标准假设下，给出了SZOHT算法的收敛分析。重要的是，我们揭示了ZO估计器的偏差与硬阈值算子的可扩性之间的冲突，并给出了ZO梯度中随机方向数的理论最小值。此外，我们还发现，在不同的设置下，SZOHT的查询复杂度与维度无关或弱依赖。最后，我们说明了我们的方法在一个投资组合优化问题和黑箱对抗攻击中的实用性。



## **46. The Lipschitz-Variance-Margin Tradeoff for Enhanced Randomized Smoothing**

增强随机平滑的Lipschitz—方差—裕度权衡 cs.LG

**SubmitDate**: 2024-03-18    [abs](http://arxiv.org/abs/2309.16883v4) [paper-pdf](http://arxiv.org/pdf/2309.16883v4)

**Authors**: Blaise Delattre, Alexandre Araujo, Quentin Barthélemy, Alexandre Allauzen

**Abstract**: Real-life applications of deep neural networks are hindered by their unsteady predictions when faced with noisy inputs and adversarial attacks. The certified radius in this context is a crucial indicator of the robustness of models. However how to design an efficient classifier with an associated certified radius? Randomized smoothing provides a promising framework by relying on noise injection into the inputs to obtain a smoothed and robust classifier. In this paper, we first show that the variance introduced by the Monte-Carlo sampling in the randomized smoothing procedure estimate closely interacts with two other important properties of the classifier, \textit{i.e.} its Lipschitz constant and margin. More precisely, our work emphasizes the dual impact of the Lipschitz constant of the base classifier, on both the smoothed classifier and the empirical variance. To increase the certified robust radius, we introduce a different way to convert logits to probability vectors for the base classifier to leverage the variance-margin trade-off. We leverage the use of Bernstein's concentration inequality along with enhanced Lipschitz bounds for randomized smoothing. Experimental results show a significant improvement in certified accuracy compared to current state-of-the-art methods. Our novel certification procedure allows us to use pre-trained models with randomized smoothing, effectively improving the current certification radius in a zero-shot manner.

摘要: 深度神经网络的现实应用在面对噪声输入和对抗攻击时受到不稳定的预测的阻碍。在这种情况下，认证半径是模型稳健性的一个关键指标。然而，如何设计一个有效的分类器与相关的认证半径？随机平滑提供了一个有希望的框架，依靠噪声注入输入，以获得平滑和鲁棒的分类器。在本文中，我们首先证明了随机平滑过程估计中由Monte—Carlo采样引入的方差与分类器的其他两个重要性质密切相关，\texit {即} Lipschitz常数和余量。更准确地说，我们的工作强调了基分类器的Lipschitz常数对平滑分类器和经验方差的双重影响。为了增加经认证的鲁棒半径，我们引入了一种不同的方法来将对数转换为基本分类器的概率向量，以利用方差—利润权衡。我们利用伯恩斯坦浓度不等式以及增强的Lipschitz边界进行随机平滑。实验结果表明，与目前最先进的方法相比，该方法在认证精度方面有了显着的提高。我们的新认证程序允许我们使用预训练的模型与随机平滑，有效地改善了当前的认证半径在零射击的方式。



## **47. SSAP: A Shape-Sensitive Adversarial Patch for Comprehensive Disruption of Monocular Depth Estimation in Autonomous Navigation Applications**

SSAP：一种用于自主导航应用中单目深度估计综合干扰的形状敏感对抗补丁 cs.CV

**SubmitDate**: 2024-03-18    [abs](http://arxiv.org/abs/2403.11515v1) [paper-pdf](http://arxiv.org/pdf/2403.11515v1)

**Authors**: Amira Guesmi, Muhammad Abdullah Hanif, Ihsen Alouani, Bassem Ouni, Muhammad Shafique

**Abstract**: Monocular depth estimation (MDE) has advanced significantly, primarily through the integration of convolutional neural networks (CNNs) and more recently, Transformers. However, concerns about their susceptibility to adversarial attacks have emerged, especially in safety-critical domains like autonomous driving and robotic navigation. Existing approaches for assessing CNN-based depth prediction methods have fallen short in inducing comprehensive disruptions to the vision system, often limited to specific local areas. In this paper, we introduce SSAP (Shape-Sensitive Adversarial Patch), a novel approach designed to comprehensively disrupt monocular depth estimation (MDE) in autonomous navigation applications. Our patch is crafted to selectively undermine MDE in two distinct ways: by distorting estimated distances or by creating the illusion of an object disappearing from the system's perspective. Notably, our patch is shape-sensitive, meaning it considers the specific shape and scale of the target object, thereby extending its influence beyond immediate proximity. Furthermore, our patch is trained to effectively address different scales and distances from the camera. Experimental results demonstrate that our approach induces a mean depth estimation error surpassing 0.5, impacting up to 99% of the targeted region for CNN-based MDE models. Additionally, we investigate the vulnerability of Transformer-based MDE models to patch-based attacks, revealing that SSAP yields a significant error of 0.59 and exerts substantial influence over 99% of the target region on these models.

摘要: 单目深度估计（MDE）已经取得了显著进步，主要是通过卷积神经网络（CNN）和最近的Transformers的集成。然而，人们对它们易受对抗性攻击的担忧已经出现，特别是在自动驾驶和机器人导航等安全关键领域。用于评估基于CNN的深度预测方法的现有方法在诱导对视觉系统的全面破坏方面不足，通常仅限于特定的局部区域。在本文中，我们介绍了一种新的方法SSAP（形状敏感对抗补丁），旨在全面破坏单目深度估计（MDE）在自主导航应用。我们的补丁被设计成以两种不同的方式有选择地破坏MDE：通过扭曲估计距离或通过创建物体从系统的视角消失的错觉。值得注意的是，我们的贴片是形状敏感的，这意味着它考虑目标物体的特定形状和规模，从而将其影响力扩展到直接接近的范围之外。此外，我们的补丁经过训练，以有效地解决与相机的不同尺度和距离。实验结果表明，我们的方法导致一个平均深度估计误差超过0.5，影响高达99%的目标区域的CNN为基础的MDE模型。此外，我们调查了基于Transformer的MDE模型对基于补丁的攻击的脆弱性，发现SSAP产生了0.59的显著误差，并对这些模型的99%的目标区域产生了重大影响。



## **48. Robust Overfitting Does Matter: Test-Time Adversarial Purification With FGSM**

鲁棒过拟合很重要：使用FGSM的测试时间对抗纯化 cs.CV

CVPR 2024

**SubmitDate**: 2024-03-18    [abs](http://arxiv.org/abs/2403.11448v1) [paper-pdf](http://arxiv.org/pdf/2403.11448v1)

**Authors**: Linyu Tang, Lei Zhang

**Abstract**: Numerous studies have demonstrated the susceptibility of deep neural networks (DNNs) to subtle adversarial perturbations, prompting the development of many advanced adversarial defense methods aimed at mitigating adversarial attacks. Current defense strategies usually train DNNs for a specific adversarial attack method and can achieve good robustness in defense against this type of adversarial attack. Nevertheless, when subjected to evaluations involving unfamiliar attack modalities, empirical evidence reveals a pronounced deterioration in the robustness of DNNs. Meanwhile, there is a trade-off between the classification accuracy of clean examples and adversarial examples. Most defense methods often sacrifice the accuracy of clean examples in order to improve the adversarial robustness of DNNs. To alleviate these problems and enhance the overall robust generalization of DNNs, we propose the Test-Time Pixel-Level Adversarial Purification (TPAP) method. This approach is based on the robust overfitting characteristic of DNNs to the fast gradient sign method (FGSM) on training and test datasets. It utilizes FGSM for adversarial purification, to process images for purifying unknown adversarial perturbations from pixels at testing time in a "counter changes with changelessness" manner, thereby enhancing the defense capability of DNNs against various unknown adversarial attacks. Extensive experimental results show that our method can effectively improve both overall robust generalization of DNNs, notably over previous methods.

摘要: 许多研究已经证明了深度神经网络（DNN）对微妙的对抗干扰的敏感性，这促使了许多旨在减轻对抗攻击的先进对抗防御方法的开发。目前的防御策略通常是针对特定的对抗攻击方法训练DNN，并在防御这种类型的对抗攻击时具有良好的鲁棒性。然而，当受到涉及不熟悉的攻击模式的评估时，经验证据显示DNN的鲁棒性明显恶化。同时，干净样本和对抗样本的分类精度之间存在权衡。大多数防御方法往往牺牲干净示例的准确性，以提高DNN的对抗鲁棒性。为了缓解这些问题，提高DNN的整体鲁棒性，我们提出了测试时间像素级对抗纯化（TPAP）方法。该方法基于DNN对训练和测试数据集的快速梯度符号法（FGSM）的鲁棒过拟合特性。该算法利用FGSM进行对抗性纯化，以“不变逆变”的方式对图像进行处理，从测试时刻像素中纯化未知对抗性扰动，从而增强DNN对各种未知对抗性攻击的防御能力。大量的实验结果表明，我们的方法可以有效地提高DNN的整体鲁棒推广，特别是在以前的方法。



## **49. Defense Against Adversarial Attacks on No-Reference Image Quality Models with Gradient Norm Regularization**

基于梯度范数正则化的无参考图像质量模型的对抗攻击防御 cs.CV

accepted by CVPR 2024

**SubmitDate**: 2024-03-18    [abs](http://arxiv.org/abs/2403.11397v1) [paper-pdf](http://arxiv.org/pdf/2403.11397v1)

**Authors**: Yujia Liu, Chenxi Yang, Dingquan Li, Jianhao Ding, Tingting Jiang

**Abstract**: The task of No-Reference Image Quality Assessment (NR-IQA) is to estimate the quality score of an input image without additional information. NR-IQA models play a crucial role in the media industry, aiding in performance evaluation and optimization guidance. However, these models are found to be vulnerable to adversarial attacks, which introduce imperceptible perturbations to input images, resulting in significant changes in predicted scores. In this paper, we propose a defense method to improve the stability in predicted scores when attacked by small perturbations, thus enhancing the adversarial robustness of NR-IQA models. To be specific, we present theoretical evidence showing that the magnitude of score changes is related to the $\ell_1$ norm of the model's gradient with respect to the input image. Building upon this theoretical foundation, we propose a norm regularization training strategy aimed at reducing the $\ell_1$ norm of the gradient, thereby boosting the robustness of NR-IQA models. Experiments conducted on four NR-IQA baseline models demonstrate the effectiveness of our strategy in reducing score changes in the presence of adversarial attacks. To the best of our knowledge, this work marks the first attempt to defend against adversarial attacks on NR-IQA models. Our study offers valuable insights into the adversarial robustness of NR-IQA models and provides a foundation for future research in this area.

摘要: 无参考图像质量评估（NR—IQA）的任务是在没有附加信息的情况下估计输入图像的质量分数。NR—IQA模型在媒体行业发挥着至关重要的作用，有助于绩效评估和优化指导。然而，这些模型被发现是容易受到对抗攻击，这引入了难以察觉的干扰输入图像，导致预测分数的显着变化。本文提出了一种防御方法，以提高小扰动攻击时预测分数的稳定性，从而增强了NR—IQA模型的对抗鲁棒性。具体地说，我们提出的理论证据表明得分变化的幅度与模型相对于输入图像的梯度的$\ell_1 $范数有关。在此基础上，我们提出了一种范数正则化训练策略，旨在降低梯度的$\ell_1 $范数，从而提高NR—IQA模型的鲁棒性.在四个NR—IQA基线模型上进行的实验证明了我们的策略在对抗攻击的存在下减少得分变化的有效性。据我们所知，这项工作标志着首次尝试防御对NR—IQA模型的对抗性攻击。我们的研究为NR—IQA模型的对抗鲁棒性提供了宝贵的见解，并为该领域的未来研究提供了基础。



## **50. A Modified Word Saliency-Based Adversarial Attack on Text Classification Models**

一种改进的基于词显著性的文本分类模型对抗攻击 cs.CL

The paper is a preprint of a version submitted in ICCIDA 2024. It  consists of 10 pages and contains 7 tables

**SubmitDate**: 2024-03-17    [abs](http://arxiv.org/abs/2403.11297v1) [paper-pdf](http://arxiv.org/pdf/2403.11297v1)

**Authors**: Hetvi Waghela, Sneha Rakshit, Jaydip Sen

**Abstract**: This paper introduces a novel adversarial attack method targeting text classification models, termed the Modified Word Saliency-based Adversarial At-tack (MWSAA). The technique builds upon the concept of word saliency to strategically perturb input texts, aiming to mislead classification models while preserving semantic coherence. By refining the traditional adversarial attack approach, MWSAA significantly enhances its efficacy in evading detection by classification systems. The methodology involves first identifying salient words in the input text through a saliency estimation process, which prioritizes words most influential to the model's decision-making process. Subsequently, these salient words are subjected to carefully crafted modifications, guided by semantic similarity metrics to ensure that the altered text remains coherent and retains its original meaning. Empirical evaluations conducted on diverse text classification datasets demonstrate the effectiveness of the proposed method in generating adversarial examples capable of successfully deceiving state-of-the-art classification models. Comparative analyses with existing adversarial attack techniques further indicate the superiority of the proposed approach in terms of both attack success rate and preservation of text coherence.

摘要: 本文提出了一种新的针对文本分类模型的对抗攻击方法，称为修正词显著性对抗攻击（MWSAA）。该技术建立在单词显著性的概念之上，策略性地扰乱输入文本，旨在误导分类模型，同时保持语义连贯性。通过改进传统的对抗性攻击方法，MWSAA显著提高了其在通过分类系统逃避检测方面的效率。该方法首先通过显着性估计过程识别输入文本中的显着词，该过程优先考虑对模型的决策过程最有影响力的词。随后，这些显著的词经过精心设计的修改，由语义相似性指标指导，以确保修改后的文本保持连贯性并保留其原始含义。在不同的文本分类数据集上进行的经验评估表明，所提出的方法在生成对抗性的例子能够成功地欺骗国家的最先进的分类模型方面的有效性。通过与现有的对抗性攻击技术的对比分析，进一步表明了该方法在攻击成功率和保持文本连贯性方面的优越性。



