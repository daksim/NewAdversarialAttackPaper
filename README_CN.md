# Latest Adversarial Attack Papers
**update at 2023-09-15 18:46:37**

翻译来自 https://cloud.tencent.com/document/product/551/15619

## **1. Pareto Adversarial Robustness: Balancing Spatial Robustness and Sensitivity-based Robustness**

帕累托对抗稳健性：平衡空间稳健性和基于敏感度的稳健性 cs.LG

Published in SCIENCE CHINA Information Sciences (SCIS) 2023. Please  also refer to the published version in the Journal reference  https://www.sciengine.com/SCIS/doi/10.1007/s11432-022-3861-8

**SubmitDate**: 2023-09-14    [abs](http://arxiv.org/abs/2111.01996v2) [paper-pdf](http://arxiv.org/pdf/2111.01996v2)

**Authors**: Ke Sun, Mingjie Li, Zhouchen Lin

**Abstract**: Adversarial robustness, which primarily comprises sensitivity-based robustness and spatial robustness, plays an integral part in achieving robust generalization. In this paper, we endeavor to design strategies to achieve universal adversarial robustness. To achieve this, we first investigate the relatively less-explored realm of spatial robustness. Then, we integrate the existing spatial robustness methods by incorporating both local and global spatial vulnerability into a unified spatial attack and adversarial training approach. Furthermore, we present a comprehensive relationship between natural accuracy, sensitivity-based robustness, and spatial robustness, supported by strong evidence from the perspective of robust representation. Crucially, to reconcile the interplay between the mutual impacts of various robustness components into one unified framework, we incorporate the \textit{Pareto criterion} into the adversarial robustness analysis, yielding a novel strategy called Pareto Adversarial Training for achieving universal robustness. The resulting Pareto front, which delineates the set of optimal solutions, provides an optimal balance between natural accuracy and various adversarial robustness. This sheds light on solutions for achieving universal robustness in the future. To the best of our knowledge, we are the first to consider universal adversarial robustness via multi-objective optimization.

摘要: 对抗性稳健性主要包括基于敏感度的稳健性和空间稳健性，是实现健壮性泛化的重要组成部分。在这篇文章中，我们努力设计策略来实现普遍的对抗健壮性。为了实现这一点，我们首先研究相对较少被探索的空间稳健性领域。然后，通过将局部和全局空间脆弱性结合到一个统一的空间攻击和对抗性训练方法中，整合了现有的空间稳健性方法。此外，我们从稳健表示的角度提出了自然准确性、基于敏感度的稳健性和空间稳健性之间的综合关系，并得到了强有力的证据的支持。重要的是，为了将不同健壮性成分之间的相互影响协调到一个统一的框架中，我们将文本{Pareto准则}引入到对抗健壮性分析中，产生了一种新的策略，称为Pareto对抗训练，以实现普遍的健壮性。由此产生的帕累托前沿描述了一组最优解，在自然准确性和各种对手稳健性之间提供了最佳平衡。这为未来实现普遍的健壮性提供了解决方案。据我们所知，我们是第一个通过多目标优化来考虑普遍对抗健壮性的人。



## **2. What Matters to Enhance Traffic Rule Compliance of Imitation Learning for Automated Driving**

提高自动驾驶模拟学习对交通规则遵从性的重要性 cs.CV

8 pages, 2 figures

**SubmitDate**: 2023-09-14    [abs](http://arxiv.org/abs/2309.07808v1) [paper-pdf](http://arxiv.org/pdf/2309.07808v1)

**Authors**: Hongkuan Zhou, Aifen Sui, Wei Cao, Letian Shi

**Abstract**: More research attention has recently been given to end-to-end autonomous driving technologies where the entire driving pipeline is replaced with a single neural network because of its simpler structure and faster inference time. Despite this appealing approach largely reducing the components in driving pipeline, its simplicity also leads to interpretability problems and safety issues arXiv:2003.06404. The trained policy is not always compliant with the traffic rules and it is also hard to discover the reason for the misbehavior because of the lack of intermediate outputs. Meanwhile, Sensors are also critical to autonomous driving's security and feasibility to perceive the surrounding environment under complex driving scenarios. In this paper, we proposed P-CSG, a novel penalty-based imitation learning approach with cross semantics generation sensor fusion technologies to increase the overall performance of End-to-End Autonomous Driving. We conducted an assessment of our model's performance using the Town 05 Long benchmark, achieving an impressive driving score improvement of over 15%. Furthermore, we conducted robustness evaluations against adversarial attacks like FGSM and Dot attacks, revealing a substantial increase in robustness compared to baseline models.More detailed information, such as code-based resources, ablation studies and videos can be found at https://hk-zh.github.io/p-csg-plus.

摘要: 最近，端到端自动驾驶技术受到了更多的研究关注，由于其结构更简单，推理时间更快，整个驾驶管道被单一的神经网络取代。尽管这种吸引人的方法在很大程度上减少了驱动管道中的组件，但其简单性也导致了可解释性问题和安全问题arxiv：2003.06404。训练后的策略并不总是符合交通规则，而且由于缺乏中间输出，也很难发现错误行为的原因。同时，传感器对自动驾驶的安全性和在复杂驾驶场景下感知周围环境的可行性也至关重要。为了提高端到端自主驾驶的整体性能，本文提出了一种新的基于惩罚的模拟学习方法P-CSG，它结合了跨语义生成传感器融合技术。我们使用Town05 Long基准对我们的模型的性能进行了评估，实现了令人印象深刻的15%以上的驾驶分数提高。此外，我们对FGSM和DOT攻击等对手攻击进行了健壮性评估，显示出与基准模型相比，健壮性有了显著提高。更多详细信息，如基于代码的资源、消融研究和视频，请访问https://hk-zh.github.io/p-csg-plus.



## **3. TrojViT: Trojan Insertion in Vision Transformers**

TrojViT：视觉变形金刚中的特洛伊木马插入 cs.LG

10 pages, 4 figures, 11 tables

**SubmitDate**: 2023-09-14    [abs](http://arxiv.org/abs/2208.13049v4) [paper-pdf](http://arxiv.org/pdf/2208.13049v4)

**Authors**: Mengxin Zheng, Qian Lou, Lei Jiang

**Abstract**: Vision Transformers (ViTs) have demonstrated the state-of-the-art performance in various vision-related tasks. The success of ViTs motivates adversaries to perform backdoor attacks on ViTs. Although the vulnerability of traditional CNNs to backdoor attacks is well-known, backdoor attacks on ViTs are seldom-studied. Compared to CNNs capturing pixel-wise local features by convolutions, ViTs extract global context information through patches and attentions. Na\"ively transplanting CNN-specific backdoor attacks to ViTs yields only a low clean data accuracy and a low attack success rate. In this paper, we propose a stealth and practical ViT-specific backdoor attack $TrojViT$. Rather than an area-wise trigger used by CNN-specific backdoor attacks, TrojViT generates a patch-wise trigger designed to build a Trojan composed of some vulnerable bits on the parameters of a ViT stored in DRAM memory through patch salience ranking and attention-target loss. TrojViT further uses minimum-tuned parameter update to reduce the bit number of the Trojan. Once the attacker inserts the Trojan into the ViT model by flipping the vulnerable bits, the ViT model still produces normal inference accuracy with benign inputs. But when the attacker embeds a trigger into an input, the ViT model is forced to classify the input to a predefined target class. We show that flipping only few vulnerable bits identified by TrojViT on a ViT model using the well-known RowHammer can transform the model into a backdoored one. We perform extensive experiments of multiple datasets on various ViT models. TrojViT can classify $99.64\%$ of test images to a target class by flipping $345$ bits on a ViT for ImageNet.Our codes are available at https://github.com/mxzheng/TrojViT

摘要: 视觉变形金刚(VITS)在各种与视觉相关的任务中展示了最先进的性能。VITS的成功促使对手对VITS进行后门攻击。虽然传统的CNN对后门攻击的脆弱性是众所周知的，但对VITS的后门攻击很少被研究。与通过卷积获取像素级局部特征的CNN相比，VITS通过块和关注点来提取全局上下文信息。将CNN特定的后门攻击活生生地移植到VITS只会产生低的干净数据准确性和低的攻击成功率。在本文中，我们提出了一种隐形和实用的特定于VIT的后门攻击$TrojViT$。与CNN特定后门攻击使用的区域触发不同，TrojViT生成修补程序触发，旨在通过修补程序显著程度排名和注意力目标丢失来构建由存储在DRAM内存中的VIT参数上的一些易受攻击位组成的特洛伊木马程序。TrojViT进一步使用最小调整的参数更新来减少特洛伊木马的比特数。一旦攻击者通过翻转易受攻击的比特将特洛伊木马程序插入到VIT模型中，VIT模型仍然会使用良性输入产生正常的推理准确性。但是，当攻击者将触发器嵌入到输入中时，VIT模型被迫将输入分类到预定义的目标类。我们表明，只需使用著名的RowHammer在VIT模型上翻转TrojViT识别的少数易受攻击的位，就可以将该模型转换为后置模型。我们在不同的VIT模型上对多个数据集进行了广泛的实验。TrojViT可以通过在ImageNetVIT上翻转$345$比特，将$99.64\$测试图像分类到目标类别。我们的代码可在https://github.com/mxzheng/TrojViT上获得



## **4. Physical Invisible Backdoor Based on Camera Imaging**

基于摄像机成像的物理隐形后门 cs.CV

**SubmitDate**: 2023-09-14    [abs](http://arxiv.org/abs/2309.07428v1) [paper-pdf](http://arxiv.org/pdf/2309.07428v1)

**Authors**: Yusheng Guo, Nan Zhong, Zhenxing Qian, Xinpeng Zhang

**Abstract**: Backdoor attack aims to compromise a model, which returns an adversary-wanted output when a specific trigger pattern appears yet behaves normally for clean inputs. Current backdoor attacks require changing pixels of clean images, which results in poor stealthiness of attacks and increases the difficulty of the physical implementation. This paper proposes a novel physical invisible backdoor based on camera imaging without changing nature image pixels. Specifically, a compromised model returns a target label for images taken by a particular camera, while it returns correct results for other images. To implement and evaluate the proposed backdoor, we take shots of different objects from multi-angles using multiple smartphones to build a new dataset of 21,500 images. Conventional backdoor attacks work ineffectively with some classical models, such as ResNet18, over the above-mentioned dataset. Therefore, we propose a three-step training strategy to mount the backdoor attack. First, we design and train a camera identification model with the phone IDs to extract the camera fingerprint feature. Subsequently, we elaborate a special network architecture, which is easily compromised by our backdoor attack, by leveraging the attributes of the CFA interpolation algorithm and combining it with the feature extraction block in the camera identification model. Finally, we transfer the backdoor from the elaborated special network architecture to the classical architecture model via teacher-student distillation learning. Since the trigger of our method is related to the specific phone, our attack works effectively in the physical world. Experiment results demonstrate the feasibility of our proposed approach and robustness against various backdoor defenses.

摘要: 后门攻击旨在危害一个模型，该模型在出现特定触发模式时返回对手想要的输出，但对于干净的输入行为正常。目前的后门攻击需要改变干净图像的像素，导致攻击的隐蔽性较差，增加了物理实现的难度。在不改变自然图像像素的前提下，提出了一种新的基于摄像机成像的物理隐形后门。具体地说，受危害的模型返回由特定相机拍摄的图像的目标标签，同时返回其他图像的正确结果。为了实施和评估拟议的后门，我们使用多部智能手机从多个角度拍摄不同对象的照片，以建立一个包含21,500张图像的新数据集。传统的后门攻击在上述数据集上与一些经典模型(如ResNet18)一起工作时效果不佳。因此，我们提出了三步训练策略来发动后门攻击。首先，设计并训练了一个基于手机ID的相机识别模型，用于提取相机指纹特征。随后，我们利用CFA内插算法的特性，将其与摄像机识别模型中的特征提取块相结合，提出了一种易于被后门攻击攻破的特殊网络体系结构。最后，通过教师-学生的精炼学习，将后门从精心设计的特殊网络体系结构转移到经典的体系结构模型。由于我们方法的触发器与特定的手机相关，因此我们的攻击在物理世界中有效地工作。实验结果证明了该方法的可行性和对各种后门防御的稳健性。



## **5. Client-side Gradient Inversion Against Federated Learning from Poisoning**

针对联合中毒学习的客户端梯度反转 cs.CR

**SubmitDate**: 2023-09-14    [abs](http://arxiv.org/abs/2309.07415v1) [paper-pdf](http://arxiv.org/pdf/2309.07415v1)

**Authors**: Jiaheng Wei, Yanjun Zhang, Leo Yu Zhang, Chao Chen, Shirui Pan, Kok-Leong Ong, Jun Zhang, Yang Xiang

**Abstract**: Federated Learning (FL) enables distributed participants (e.g., mobile devices) to train a global model without sharing data directly to a central server. Recent studies have revealed that FL is vulnerable to gradient inversion attack (GIA), which aims to reconstruct the original training samples and poses high risk against the privacy of clients in FL. However, most existing GIAs necessitate control over the server and rely on strong prior knowledge including batch normalization and data distribution information. In this work, we propose Client-side poisoning Gradient Inversion (CGI), which is a novel attack method that can be launched from clients. For the first time, we show the feasibility of a client-side adversary with limited knowledge being able to recover the training samples from the aggregated global model. We take a distinct approach in which the adversary utilizes a malicious model that amplifies the loss of a specific targeted class of interest. When honest clients employ the poisoned global model, the gradients of samples belonging to the targeted class are magnified, making them the dominant factor in the aggregated update. This enables the adversary to effectively reconstruct the private input belonging to other clients using the aggregated update. In addition, our CGI also features its ability to remain stealthy against Byzantine-robust aggregation rules (AGRs). By optimizing malicious updates and blending benign updates with a malicious replacement vector, our method remains undetected by these defense mechanisms. To evaluate the performance of CGI, we conduct experiments on various benchmark datasets, considering representative Byzantine-robust AGRs, and exploring diverse FL settings with different levels of adversary knowledge about the data. Our results demonstrate that CGI consistently and successfully extracts training input in all tested scenarios.

摘要: 联合学习(FL)使分布式参与者(例如，移动设备)能够训练全局模型，而无需将数据直接共享到中央服务器。最近的研究表明，FL容易受到梯度反转攻击(GIA)，GIA旨在重建原始训练样本，并对FL中客户的隐私构成高风险。然而，大多数现有的GIA需要对服务器进行控制，并依赖于强大的先验知识，包括批处理标准化和数据分布信息。在这项工作中，我们提出了客户端中毒梯度反转(CGI)，这是一种新的攻击方法，可以从客户端发起。我们首次证明了知识有限的客户端对手能够从聚合的全局模型中恢复训练样本的可行性。我们采取了一种截然不同的方法，即对手利用恶意模型，放大特定目标兴趣类别的损失。当诚实的客户使用有毒的全局模型时，属于目标类的样本的梯度被放大，使它们成为聚合更新中的主导因素。这使得敌手能够使用聚集的更新有效地重构属于其他客户端的私有输入。此外，我们的CGI还具有针对拜占庭稳健聚合规则(AGR)保持隐蔽性的能力。通过优化恶意更新并将良性更新与恶意替换向量混合，我们的方法仍然不会被这些防御机制检测到。为了评估CGI的性能，我们在不同的基准数据集上进行了实验，考虑了具有代表性的拜占庭稳健AGR，并探索了具有不同水平对手知识的不同FL环境。我们的结果表明，CGI在所有测试场景中都一致并成功地提取了训练输入。



## **6. COVER: A Heuristic Greedy Adversarial Attack on Prompt-based Learning in Language Models**

封面：对语言模型中基于提示的学习的启发式贪婪对抗性攻击 cs.CL

**SubmitDate**: 2023-09-14    [abs](http://arxiv.org/abs/2306.05659v3) [paper-pdf](http://arxiv.org/pdf/2306.05659v3)

**Authors**: Zihao Tan, Qingliang Chen, Wenbin Zhu, Yongjian Huang

**Abstract**: Prompt-based learning has been proved to be an effective way in pre-trained language models (PLMs), especially in low-resource scenarios like few-shot settings. However, the trustworthiness of PLMs is of paramount significance and potential vulnerabilities have been shown in prompt-based templates that could mislead the predictions of language models, causing serious security concerns. In this paper, we will shed light on some vulnerabilities of PLMs, by proposing a prompt-based adversarial attack on manual templates in black box scenarios. First of all, we design character-level and word-level heuristic approaches to break manual templates separately. Then we present a greedy algorithm for the attack based on the above heuristic destructive approaches. Finally, we evaluate our approach with the classification tasks on three variants of BERT series models and eight datasets. And comprehensive experimental results justify the effectiveness of our approach in terms of attack success rate and attack speed.

摘要: 基于提示的学习已被证明是预训练语言模型(PLM)中的一种有效方法，特别是在资源较少的场景中，如少镜头场景。然而，PLM的可信性至关重要，基于提示的模板中已经显示出潜在的漏洞，这些漏洞可能会误导语言模型的预测，导致严重的安全问题。在本文中，我们将通过在黑盒场景中对人工模板提出一种基于提示的对抗性攻击来揭示PLM的一些漏洞。首先，我们分别设计了字字级和词级启发式方法来打破人工模板。在此基础上，提出了一种基于上述启发式破坏性方法的贪婪算法。最后，我们在BERT系列模型的三个变种和八个数据集上对我们的方法进行了评估。综合实验结果从攻击成功率和攻击速度两个方面验证了该方法的有效性。



## **7. Semantic Adversarial Attacks via Diffusion Models**

基于扩散模型的语义对抗性攻击 cs.CV

To appear in BMVC 2023

**SubmitDate**: 2023-09-14    [abs](http://arxiv.org/abs/2309.07398v1) [paper-pdf](http://arxiv.org/pdf/2309.07398v1)

**Authors**: Chenan Wang, Jinhao Duan, Chaowei Xiao, Edward Kim, Matthew Stamm, Kaidi Xu

**Abstract**: Traditional adversarial attacks concentrate on manipulating clean examples in the pixel space by adding adversarial perturbations. By contrast, semantic adversarial attacks focus on changing semantic attributes of clean examples, such as color, context, and features, which are more feasible in the real world. In this paper, we propose a framework to quickly generate a semantic adversarial attack by leveraging recent diffusion models since semantic information is included in the latent space of well-trained diffusion models. Then there are two variants of this framework: 1) the Semantic Transformation (ST) approach fine-tunes the latent space of the generated image and/or the diffusion model itself; 2) the Latent Masking (LM) approach masks the latent space with another target image and local backpropagation-based interpretation methods. Additionally, the ST approach can be applied in either white-box or black-box settings. Extensive experiments are conducted on CelebA-HQ and AFHQ datasets, and our framework demonstrates great fidelity, generalizability, and transferability compared to other baselines. Our approaches achieve approximately 100% attack success rate in multiple settings with the best FID as 36.61. Code is available at https://github.com/steven202/semantic_adv_via_dm.

摘要: 传统的对抗性攻击集中于通过添加对抗性扰动来操纵像素空间中的干净样本。相比之下，语义对抗性攻击侧重于改变干净示例的语义属性，如颜色、上下文和特征，这些在现实世界中更可行。由于语义信息被包含在训练良好的扩散模型的潜在空间中，本文提出了一种利用最近的扩散模型来快速生成语义对抗攻击的框架。然后该框架有两种变体：1)语义变换(ST)方法微调生成图像的潜在空间和/或扩散模型本身；2)潜在掩蔽(LM)方法用另一目标图像和基于局部反向传播的解释方法来掩盖潜在空间。此外，ST方法可以应用于白盒或黑盒设置。在CelebA-HQ和AFHQ数据集上进行了广泛的实验，与其他基线相比，我们的框架表现出了很好的保真度、泛化和可转移性。我们的方法在多个环境下实现了大约100%的攻击成功率，最佳FID为36.61。代码可在https://github.com/steven202/semantic_adv_via_dm.上找到



## **8. Deep Nonparametric Convexified Filtering for Computational Photography, Image Synthesis and Adversarial Defense**

深度非参数凸化滤波在计算摄影、图像合成和对抗防御中的应用 cs.CV

**SubmitDate**: 2023-09-14    [abs](http://arxiv.org/abs/2309.06724v2) [paper-pdf](http://arxiv.org/pdf/2309.06724v2)

**Authors**: Jianqiao Wangni

**Abstract**: We aim to provide a general framework of for computational photography that recovers the real scene from imperfect images, via the Deep Nonparametric Convexified Filtering (DNCF). It is consists of a nonparametric deep network to resemble the physical equations behind the image formation, such as denoising, super-resolution, inpainting, and flash. DNCF has no parameterization dependent on training data, therefore has a strong generalization and robustness to adversarial image manipulation. During inference, we also encourage the network parameters to be nonnegative and create a bi-convex function on the input and parameters, and this adapts to second-order optimization algorithms with insufficient running time, having 10X acceleration over Deep Image Prior. With these tools, we empirically verify its capability to defend image classification deep networks against adversary attack algorithms in real-time.

摘要: 我们的目标是为计算摄影提供一个通用的框架，通过深度非参数凸化滤波(DNCF)从不完美的图像中恢复真实场景。它由一个非参数的深度网络组成，类似于图像形成背后的物理方程，如去噪、超分辨率、修复和闪光。DNCF不依赖于训练数据进行参数化，对敌意图像篡改具有较强的泛化能力和鲁棒性。在推理过程中，我们还鼓励网络参数为非负的，并建立关于输入和参数的双凸函数，这适用于运行时间不足的二阶优化算法，比深度图像先验算法有10倍的加速。利用这些工具，我们对其实时防御图像分类深度网络攻击算法的能力进行了实证验证。



## **9. BAARD: Blocking Adversarial Examples by Testing for Applicability, Reliability and Decidability**

Baard：通过测试适用性、可靠性和可判断性来阻止敌意示例 cs.LG

**SubmitDate**: 2023-09-13    [abs](http://arxiv.org/abs/2105.00495v2) [paper-pdf](http://arxiv.org/pdf/2105.00495v2)

**Authors**: Xinglong Chang, Katharina Dost, Kaiqi Zhao, Ambra Demontis, Fabio Roli, Gill Dobbie, Jörg Wicker

**Abstract**: Adversarial defenses protect machine learning models from adversarial attacks, but are often tailored to one type of model or attack. The lack of information on unknown potential attacks makes detecting adversarial examples challenging. Additionally, attackers do not need to follow the rules made by the defender. To address this problem, we take inspiration from the concept of Applicability Domain in cheminformatics. Cheminformatics models struggle to make accurate predictions because only a limited number of compounds are known and available for training. Applicability Domain defines a domain based on the known compounds and rejects any unknown compound that falls outside the domain. Similarly, adversarial examples start as harmless inputs, but can be manipulated to evade reliable classification by moving outside the domain of the classifier. We are the first to identify the similarity between Applicability Domain and adversarial detection. Instead of focusing on unknown attacks, we focus on what is known, the training data. We propose a simple yet robust triple-stage data-driven framework that checks the input globally and locally, and confirms that they are coherent with the model's output. This framework can be applied to any classification model and is not limited to specific attacks. We demonstrate these three stages work as one unit, effectively detecting various attacks, even for a white-box scenario.

摘要: 对抗性防御保护机器学习模型免受对抗性攻击，但通常针对一种类型的模型或攻击进行定制。缺乏有关未知潜在攻击的信息，使得检测敌意示例具有挑战性。此外，攻击者不需要遵守防御者制定的规则。为了解决这个问题，我们从化学信息学中适用域的概念中得到了启发。化学信息学模型很难做出准确的预测，因为只有有限数量的化合物是已知的，并且可以用于培训。适用域基于已知化合物定义域，并拒绝任何落在域之外的未知化合物。同样，敌意例子开始时是无害的输入，但可以通过移出分类器的域来操纵以逃避可靠的分类。我们首次发现了适用域和敌意检测之间的相似性。我们关注的不是未知攻击，而是已知的训练数据。我们提出了一个简单但健壮的三阶段数据驱动框架，它检查全局和局部的输入，并确认它们与模型的输出是一致的。该框架可以应用于任何分类模型，并且不限于特定攻击。我们演示了这三个阶段作为一个整体工作，有效地检测各种攻击，即使是在白盒情况下也是如此。



## **10. RAIN: Your Language Models Can Align Themselves without Finetuning**

Rain：您的语言模型无需微调即可自动调整 cs.CL

**SubmitDate**: 2023-09-13    [abs](http://arxiv.org/abs/2309.07124v1) [paper-pdf](http://arxiv.org/pdf/2309.07124v1)

**Authors**: Yuhui Li, Fangyun Wei, Jinjing Zhao, Chao Zhang, Hongyang Zhang

**Abstract**: Large language models (LLMs) often demonstrate inconsistencies with human preferences. Previous research gathered human preference data and then aligned the pre-trained models using reinforcement learning or instruction tuning, the so-called finetuning step. In contrast, aligning frozen LLMs without any extra data is more appealing. This work explores the potential of the latter setting. We discover that by integrating self-evaluation and rewind mechanisms, unaligned LLMs can directly produce responses consistent with human preferences via self-boosting. We introduce a novel inference method, Rewindable Auto-regressive INference (RAIN), that allows pre-trained LLMs to evaluate their own generation and use the evaluation results to guide backward rewind and forward generation for AI safety. Notably, RAIN operates without the need of extra data for model alignment and abstains from any training, gradient computation, or parameter updates; during the self-evaluation phase, the model receives guidance on which human preference to align with through a fixed-template prompt, eliminating the need to modify the initial prompt. Experimental results evaluated by GPT-4 and humans demonstrate the effectiveness of RAIN: on the HH dataset, RAIN improves the harmlessness rate of LLaMA 30B over vanilla inference from 82% to 97%, while maintaining the helpfulness rate. Under the leading adversarial attack llm-attacks on Vicuna 33B, RAIN establishes a new defense baseline by reducing the attack success rate from 94% to 19%.

摘要: 大型语言模型(LLM)经常显示出与人类偏好的不一致。之前的研究收集了人类的偏好数据，然后使用强化学习或教学调整，即所谓的微调步骤，对齐了预先训练的模型。相比之下，在没有任何额外数据的情况下对齐冻结的LLM更具吸引力。这项工作探索了后一种环境的潜力。我们发现，通过集成自我评估和回溯机制，未对齐的LLM可以通过自我增强直接产生与人类偏好一致的反应。我们引入了一种新的推理方法--可倒带自回归推理(RAIN)，它允许预先训练的LLM对自己的生成进行评估，并使用评估结果来指导人工智能安全的回溯和正演生成。值得注意的是，RAIN的运行不需要额外的数据来进行模型比对，并且不需要任何训练、梯度计算或参数更新；在自我评估阶段，模型通过固定模板提示接收关于与哪个人类偏好匹配的指导，从而消除了修改初始提示的需要。GPT-4和人类评估的实验结果证明了RAIN的有效性：在HH数据集上，RAIN在保持有益率的同时，将大羊驼30B的无害率从82%提高到97%。在领先的对抗性攻击1LM-对维库纳33B的攻击下，RAIN通过将攻击成功率从94%降低到19%，建立了新的防御基线。



## **11. Hardening RGB-D Object Recognition Systems against Adversarial Patch Attacks**

抵抗敌意补丁攻击的RGB-D目标识别系统 cs.CV

Accepted for publication in the Information Sciences journal

**SubmitDate**: 2023-09-13    [abs](http://arxiv.org/abs/2309.07106v1) [paper-pdf](http://arxiv.org/pdf/2309.07106v1)

**Authors**: Yang Zheng, Luca Demetrio, Antonio Emanuele Cinà, Xiaoyi Feng, Zhaoqiang Xia, Xiaoyue Jiang, Ambra Demontis, Battista Biggio, Fabio Roli

**Abstract**: RGB-D object recognition systems improve their predictive performances by fusing color and depth information, outperforming neural network architectures that rely solely on colors. While RGB-D systems are expected to be more robust to adversarial examples than RGB-only systems, they have also been proven to be highly vulnerable. Their robustness is similar even when the adversarial examples are generated by altering only the original images' colors. Different works highlighted the vulnerability of RGB-D systems; however, there is a lacking of technical explanations for this weakness. Hence, in our work, we bridge this gap by investigating the learned deep representation of RGB-D systems, discovering that color features make the function learned by the network more complex and, thus, more sensitive to small perturbations. To mitigate this problem, we propose a defense based on a detection mechanism that makes RGB-D systems more robust against adversarial examples. We empirically show that this defense improves the performances of RGB-D systems against adversarial examples even when they are computed ad-hoc to circumvent this detection mechanism, and that is also more effective than adversarial training.

摘要: RGB-D目标识别系统通过融合颜色和深度信息来提高其预测性能，性能优于仅依赖颜色的神经网络结构。虽然预计RGB-D系统比仅使用RGB的系统更能抵御敌意示例，但它们也被证明是非常脆弱的。它们的稳健性是相似的，即使当对抗性的例子是通过只改变原始图像的颜色来生成的时候。不同的工作突出了RGB-D系统的脆弱性；然而，对于这一弱点缺乏技术解释。因此，在我们的工作中，我们通过研究RGB-D系统的学习深度表示来弥合这一差距，发现颜色特征使网络学习的函数更加复杂，因此对微小扰动更加敏感。为了缓解这一问题，我们提出了一种基于检测机制的防御机制，使RGB-D系统对敌意示例更具健壮性。我们的经验表明，这种防御提高了RGB-D系统对敌意示例的性能，即使它们是为绕过这种检测机制而自组织计算的，这也比对抗性训练更有效。



## **12. Mitigating Adversarial Attacks in Federated Learning with Trusted Execution Environments**

在可信执行环境下联合学习中缓解敌意攻击 cs.LG

12 pages, 4 figures, to be published in Proceedings 23rd  International Conference on Distributed Computing Systems. arXiv admin note:  substantial text overlap with arXiv:2308.04373

**SubmitDate**: 2023-09-13    [abs](http://arxiv.org/abs/2309.07197v1) [paper-pdf](http://arxiv.org/pdf/2309.07197v1)

**Authors**: Simon Queyrut, Valerio Schiavoni, Pascal Felber

**Abstract**: The main premise of federated learning (FL) is that machine learning model updates are computed locally to preserve user data privacy. This approach avoids by design user data to ever leave the perimeter of their device. Once the updates aggregated, the model is broadcast to all nodes in the federation. However, without proper defenses, compromised nodes can probe the model inside their local memory in search for adversarial examples, which can lead to dangerous real-world scenarios. For instance, in image-based applications, adversarial examples consist of images slightly perturbed to the human eye getting misclassified by the local model. These adversarial images are then later presented to a victim node's counterpart model to replay the attack. Typical examples harness dissemination strategies such as altered traffic signs (patch attacks) no longer recognized by autonomous vehicles or seemingly unaltered samples that poison the local dataset of the FL scheme to undermine its robustness. Pelta is a novel shielding mechanism leveraging Trusted Execution Environments (TEEs) that reduce the ability of attackers to craft adversarial samples. Pelta masks inside the TEE the first part of the back-propagation chain rule, typically exploited by attackers to craft the malicious samples. We evaluate Pelta on state-of-the-art accurate models using three well-established datasets: CIFAR-10, CIFAR-100 and ImageNet. We show the effectiveness of Pelta in mitigating six white-box state-of-the-art adversarial attacks, such as Projected Gradient Descent, Momentum Iterative Method, Auto Projected Gradient Descent, the Carlini & Wagner attack. In particular, Pelta constitutes the first attempt at defending an ensemble model against the Self-Attention Gradient attack to the best of our knowledge. Our code is available to the research community at https://github.com/queyrusi/Pelta.

摘要: 联合学习(FL)的主要前提是机器学习模型更新在本地计算，以保护用户数据隐私。这种方法通过设计避免了用户数据离开其设备的外围。一旦更新被聚合，模型就被广播到联盟中的所有节点。然而，在没有适当的防御措施的情况下，受攻击的节点可以在其本地内存中探测模型，以搜索敌对的示例，这可能会导致危险的现实世界场景。例如，在基于图像的应用程序中，对抗性的例子包括被局部模型错误分类的对人眼略微扰动的图像。这些对抗性图像随后被呈现给受害者节点的对应模型，以重播攻击。典型的例子利用传播策略，例如改变的交通标志(补丁攻击)不再被自动车辆识别，或者似乎没有改变的样本毒害FL方案的本地数据集以破坏其健壮性。PELTA是一种利用可信执行环境(TES)的新型屏蔽机制，它降低了攻击者伪造对手样本的能力。佩尔塔在发球台内掩盖了反向传播链规则的第一部分，通常被攻击者利用来手工制作恶意样本。我们使用三个成熟的数据集：CIFAR-10，CIFAR-100和ImageNet，在最先进的准确模型上对Pelta进行评估。我们展示了Pelta在缓解六种最先进的白盒对抗性攻击方面的有效性，例如投影梯度下降、动量迭代法、自动投影梯度下降、Carlini&Wagner攻击。特别是，据我们所知，佩尔塔构成了保护整体模型免受自我注意梯度攻击的第一次尝试。我们的代码可在https://github.com/queyrusi/Pelta.的研究社区中获得



## **13. Differentiable JPEG: The Devil is in the Details**

与众不同的JPEG：魔鬼在细节中 cs.CV

Accepted at WACV 2024. Project page:  https://christophreich1996.github.io/differentiable_jpeg/

**SubmitDate**: 2023-09-13    [abs](http://arxiv.org/abs/2309.06978v1) [paper-pdf](http://arxiv.org/pdf/2309.06978v1)

**Authors**: Christoph Reich, Biplob Debnath, Deep Patel, Srimat Chakradhar

**Abstract**: JPEG remains one of the most widespread lossy image coding methods. However, the non-differentiable nature of JPEG restricts the application in deep learning pipelines. Several differentiable approximations of JPEG have recently been proposed to address this issue. This paper conducts a comprehensive review of existing diff. JPEG approaches and identifies critical details that have been missed by previous methods. To this end, we propose a novel diff. JPEG approach, overcoming previous limitations. Our approach is differentiable w.r.t. the input image, the JPEG quality, the quantization tables, and the color conversion parameters. We evaluate the forward and backward performance of our diff. JPEG approach against existing methods. Additionally, extensive ablations are performed to evaluate crucial design choices. Our proposed diff. JPEG resembles the (non-diff.) reference implementation best, significantly surpassing the recent-best diff. approach by $3.47$dB (PSNR) on average. For strong compression rates, we can even improve PSNR by $9.51$dB. Strong adversarial attack results are yielded by our diff. JPEG, demonstrating the effective gradient approximation. Our code is available at https://github.com/necla-ml/Diff-JPEG.

摘要: JPEG仍然是应用最广泛的有损图像编码方法之一。然而，JPEG的不可微特性限制了其在深度学习管道中的应用。为了解决这个问题，最近已经提出了几种JPEG的可微近似。本文对现有的DIFF进行了全面的回顾。JPEG处理并确定了以前方法遗漏的关键细节。为此，我们提出了一个新颖的Diff。JPEG方法，克服了以前的限制。我们的方法是可微的W.r.t。输入图像、JPEG质量、量化表和颜色转换参数。我们评估了DIFF的向前和向后性能。JPEG方法与现有方法的对比。此外，还进行了广泛的消融，以评估关键的设计选择。我们提议的不同之处。JPEG与(Non-Diff.)参考实现最好，大大超过了最近最好的差异。平均接近3.47美元分贝(PSNR)。对于强压缩率，我们甚至可以将PSNR提高9.51美元分贝。强大的对抗性攻击结果是由我们的差异产生的。JPEG格式，演示了有效的渐变近似。我们的代码可以在https://github.com/necla-ml/Diff-JPEG.上找到



## **14. PhantomSound: Black-Box, Query-Efficient Audio Adversarial Attack via Split-Second Phoneme Injection**

PhantomSound：黑盒、查询效率高的音频攻击，通过瞬间音素注入 cs.CR

RAID 2023

**SubmitDate**: 2023-09-13    [abs](http://arxiv.org/abs/2309.06960v1) [paper-pdf](http://arxiv.org/pdf/2309.06960v1)

**Authors**: Hanqing Guo, Guangjing Wang, Yuanda Wang, Bocheng Chen, Qiben Yan, Li Xiao

**Abstract**: In this paper, we propose PhantomSound, a query-efficient black-box attack toward voice assistants. Existing black-box adversarial attacks on voice assistants either apply substitution models or leverage the intermediate model output to estimate the gradients for crafting adversarial audio samples. However, these attack approaches require a significant amount of queries with a lengthy training stage. PhantomSound leverages the decision-based attack to produce effective adversarial audios, and reduces the number of queries by optimizing the gradient estimation. In the experiments, we perform our attack against 4 different speech-to-text APIs under 3 real-world scenarios to demonstrate the real-time attack impact. The results show that PhantomSound is practical and robust in attacking 5 popular commercial voice controllable devices over the air, and is able to bypass 3 liveness detection mechanisms with >95% success rate. The benchmark result shows that PhantomSound can generate adversarial examples and launch the attack in a few minutes. We significantly enhance the query efficiency and reduce the cost of a successful untargeted and targeted adversarial attack by 93.1% and 65.5% compared with the state-of-the-art black-box attacks, using merely ~300 queries (~5 minutes) and ~1,500 queries (~25 minutes), respectively.

摘要: 在本文中，我们提出了一种针对语音助手的查询高效黑盒攻击PhantomSound。现有的针对语音助手的黑盒对抗性攻击要么应用替换模型，要么利用中间模型输出来估计用于制作对抗性音频样本的梯度。然而，这些攻击方法需要大量的查询和漫长的训练阶段。PhantomSound利用基于决策的攻击来产生有效的对抗性音频，并通过优化梯度估计来减少查询数量。在实验中，我们在3个真实场景下对4种不同的语音转文本API进行了攻击，以演示攻击的实时效果。结果表明，PhantomSound对5种流行的商用语音可控设备的空中攻击具有较强的实用性和健壮性，能够绕过3种活跃度检测机制，成功率>95%。基准测试结果表明，PhantomSound可以在几分钟内生成对抗性示例并发起攻击。与最先进的黑盒攻击相比，仅使用约300个查询(~5分钟)和~1500个查询(~25分钟)，我们显著提高了查询效率，并将成功的非定向攻击和定向攻击的成本分别降低了93.1%和65.5%。



## **15. Improving Visual Quality and Transferability of Adversarial Attacks on Face Recognition Simultaneously with Adversarial Restoration**

在对抗性恢复的同时提高人脸识别对抗性攻击的视觉质量和可转移性 cs.CV

\copyright 2023 IEEE. Personal use of this material is permitted.  Permission from IEEE must be obtained for all other uses, in any current or  future media, including reprinting/republishing this material for advertising  or promotional purposes, creating new collective works, for resale or  redistribution to servers or lists, or reuse of any copyrighted component of  this work in other works

**SubmitDate**: 2023-09-13    [abs](http://arxiv.org/abs/2309.01582v3) [paper-pdf](http://arxiv.org/pdf/2309.01582v3)

**Authors**: Fengfan Zhou, Hefei Ling, Yuxuan Shi, Jiazhong Chen, Ping Li

**Abstract**: Adversarial face examples possess two critical properties: Visual Quality and Transferability. However, existing approaches rarely address these properties simultaneously, leading to subpar results. To address this issue, we propose a novel adversarial attack technique known as Adversarial Restoration (AdvRestore), which enhances both visual quality and transferability of adversarial face examples by leveraging a face restoration prior. In our approach, we initially train a Restoration Latent Diffusion Model (RLDM) designed for face restoration. Subsequently, we employ the inference process of RLDM to generate adversarial face examples. The adversarial perturbations are applied to the intermediate features of RLDM. Additionally, by treating RLDM face restoration as a sibling task, the transferability of the generated adversarial face examples is further improved. Our experimental results validate the effectiveness of the proposed attack method.

摘要: 对抗性人脸样例具有两个重要属性：视觉质量和可转移性。然而，现有的方法很少同时处理这些属性，导致结果低于平均水平。为了解决这个问题，我们提出了一种新的对抗性攻击技术，称为对抗性恢复(AdvRestore)，它通过利用事先的人脸恢复来提高对抗性人脸样本的视觉质量和可转移性。在我们的方法中，我们首先训练一个用于人脸恢复的恢复潜在扩散模型(RLDM)。随后，我们利用RLDM的推理过程来生成对抗性人脸样本。将对抗性扰动应用于RLDM的中间特征。此外，通过将RLDM人脸恢复视为兄弟任务，进一步提高了生成的对抗性人脸样本的可转移性。实验结果验证了该攻击方法的有效性。



## **16. Attacking logo-based phishing website detectors with adversarial perturbations**

利用对抗性扰动攻击基于徽标的钓鱼网站检测器 cs.CR

To appear in ESORICS 2023

**SubmitDate**: 2023-09-13    [abs](http://arxiv.org/abs/2308.09392v2) [paper-pdf](http://arxiv.org/pdf/2308.09392v2)

**Authors**: Jehyun Lee, Zhe Xin, Melanie Ng Pei See, Kanav Sabharwal, Giovanni Apruzzese, Dinil Mon Divakaran

**Abstract**: Recent times have witnessed the rise of anti-phishing schemes powered by deep learning (DL). In particular, logo-based phishing detectors rely on DL models from Computer Vision to identify logos of well-known brands on webpages, to detect malicious webpages that imitate a given brand. For instance, Siamese networks have demonstrated notable performance for these tasks, enabling the corresponding anti-phishing solutions to detect even "zero-day" phishing webpages. In this work, we take the next step of studying the robustness of logo-based phishing detectors against adversarial ML attacks. We propose a novel attack exploiting generative adversarial perturbations to craft "adversarial logos" that evade phishing detectors. We evaluate our attacks through: (i) experiments on datasets containing real logos, to evaluate the robustness of state-of-the-art phishing detectors; and (ii) user studies to gauge whether our adversarial logos can deceive human eyes. The results show that our proposed attack is capable of crafting perturbed logos subtle enough to evade various DL models-achieving an evasion rate of up to 95%. Moreover, users are not able to spot significant differences between generated adversarial logos and original ones.

摘要: 最近见证了由深度学习(DL)提供动力的反网络钓鱼计划的兴起。特别是，基于徽标的钓鱼检测器依赖于计算机视觉的DL模型来识别网页上知名品牌的徽标，以检测模仿给定品牌的恶意网页。例如，暹罗网络在这些任务中表现出了显著的性能，使相应的反网络钓鱼解决方案能够检测到甚至是“零日”网络钓鱼网页。在这项工作中，我们下一步研究了基于标识的钓鱼检测器对恶意ML攻击的稳健性。我们提出了一种新的攻击，利用生成性敌意扰动来创建逃避网络钓鱼检测器的“对抗性标识”。我们通过以下方式评估我们的攻击：(I)在包含真实标识的数据集上进行实验，以评估最先进的网络钓鱼检测器的健壮性；以及(Ii)用户研究，以衡量我们的对手标识是否可以欺骗人眼。结果表明，我们提出的攻击能够巧妙地制作足够微妙的扰动徽标来规避各种DL模型-实现高达95%的逃避率。此外，用户无法发现生成的敌意徽标与原始徽标之间的显著差异。



## **17. Adversaries with Limited Information in the Friedkin--Johnsen Model**

Friedkin-Johnsen模型中信息有限的对手 cs.SI

KDD'23

**SubmitDate**: 2023-09-12    [abs](http://arxiv.org/abs/2306.10313v2) [paper-pdf](http://arxiv.org/pdf/2306.10313v2)

**Authors**: Sijing Tu, Stefan Neumann, Aristides Gionis

**Abstract**: In recent years, online social networks have been the target of adversaries who seek to introduce discord into societies, to undermine democracies and to destabilize communities. Often the goal is not to favor a certain side of a conflict but to increase disagreement and polarization. To get a mathematical understanding of such attacks, researchers use opinion-formation models from sociology, such as the Friedkin--Johnsen model, and formally study how much discord the adversary can produce when altering the opinions for only a small set of users. In this line of work, it is commonly assumed that the adversary has full knowledge about the network topology and the opinions of all users. However, the latter assumption is often unrealistic in practice, where user opinions are not available or simply difficult to estimate accurately.   To address this concern, we raise the following question: Can an attacker sow discord in a social network, even when only the network topology is known? We answer this question affirmatively. We present approximation algorithms for detecting a small set of users who are highly influential for the disagreement and polarization in the network. We show that when the adversary radicalizes these users and if the initial disagreement/polarization in the network is not very high, then our method gives a constant-factor approximation on the setting when the user opinions are known. To find the set of influential users, we provide a novel approximation algorithm for a variant of MaxCut in graphs with positive and negative edge weights. We experimentally evaluate our methods, which have access only to the network topology, and we find that they have similar performance as methods that have access to the network topology and all user opinions. We further present an NP-hardness proof, which was an open question by Chen and Racz [IEEE Trans. Netw. Sci. Eng., 2021].

摘要: 近年来，在线社交网络一直是试图在社会中制造不和谐、破坏民主和破坏社区稳定的敌人的目标。通常，目标不是偏袒冲突的某一方，而是增加分歧和两极分化。为了从数学上理解这类攻击，研究人员使用了社会学中的观点形成模型，如弗里德金-约翰森模型，并正式研究了当对手只为一小部分用户改变观点时，可以产生多大的不和谐。在这方面的工作中，通常假设对手完全了解网络拓扑和所有用户的意见。然而，后一种假设在实践中往往是不现实的，因为用户的意见是不可用的，或者只是很难准确估计。为了解决这一问题，我们提出了以下问题：即使只知道网络拓扑，攻击者也能在社交网络中挑拨离间吗？我们肯定地回答了这个问题。我们提出了一种近似算法，用于检测对网络中的不一致和极化有很大影响的一小部分用户。我们证明，当对手激化这些用户时，如果网络中的初始分歧/极化不是很高，那么当用户意见已知时，我们的方法在设置上给出一个恒定因子近似。为了寻找有影响力的用户集，我们给出了一种新的算法，用于计算边权为正负的图中MaxCut的一种变种。我们对我们的方法进行了实验评估，这些方法只能访问网络拓扑，我们发现它们具有与可以访问网络拓扑和所有用户意见的方法类似的性能。我们进一步给出了NP-硬度证明，这是Chen和racz[IEEE译文]提出的一个未决问题。奈特。SCI。Eng.，2021]。



## **18. Using Reed-Muller Codes for Classification with Rejection and Recovery**

利用Reed-Muller码进行具有拒绝和恢复的分类 cs.LG

38 pages, 7 figures

**SubmitDate**: 2023-09-12    [abs](http://arxiv.org/abs/2309.06359v1) [paper-pdf](http://arxiv.org/pdf/2309.06359v1)

**Authors**: Daniel Fentham, David Parker, Mark Ryan

**Abstract**: When deploying classifiers in the real world, users expect them to respond to inputs appropriately. However, traditional classifiers are not equipped to handle inputs which lie far from the distribution they were trained on. Malicious actors can exploit this defect by making adversarial perturbations designed to cause the classifier to give an incorrect output. Classification-with-rejection methods attempt to solve this problem by allowing networks to refuse to classify an input in which they have low confidence. This works well for strongly adversarial examples, but also leads to the rejection of weakly perturbed images, which intuitively could be correctly classified. To address these issues, we propose Reed-Muller Aggregation Networks (RMAggNet), a classifier inspired by Reed-Muller error-correction codes which can correct and reject inputs. This paper shows that RMAggNet can minimise incorrectness while maintaining good correctness over multiple adversarial attacks at different perturbation budgets by leveraging the ability to correct errors in the classification process. This provides an alternative classification-with-rejection method which can reduce the amount of additional processing in situations where a small number of incorrect classifications are permissible.

摘要: 在现实世界中部署分类器时，用户希望它们对输入做出适当的响应。然而，传统的分类器没有配备来处理远离它们所训练的分布的输入。恶意攻击者可以通过进行敌意干扰来利用这一缺陷，这些干扰旨在导致分类器给出不正确的输出。拒绝分类方法试图通过允许网络拒绝对其置信度较低的输入进行分类来解决这一问题。这对于强对抗性的例子很有效，但也会导致对弱扰动图像的拒绝，这在直觉上是可以正确分类的。为了解决这些问题，我们提出了Reed-Muller聚合网络(RMAggNet)，这是一种受Reed-Muller纠错码启发的分类器，可以纠正和拒绝输入。通过利用RMAggNet在分类过程中纠正错误的能力，在不同的扰动预算下，RMAggNet可以最大限度地减少错误，同时对多个对手攻击保持良好的正确率。这提供了一种可选的具有拒绝的分类方法，该方法可以在允许少量错误分类的情况下减少附加处理量。



## **19. Inaudible Adversarial Perturbation: Manipulating the Recognition of User Speech in Real Time**

听不见的对抗性扰动：实时操纵用户语音识别 cs.CR

Accepted by NDSS Symposium 2024. Please cite this paper as "Xinfeng  Li, Chen Yan, Xuancun Lu, Zihan Zeng, Xiaoyu Ji, Wenyuan Xu. Inaudible  Adversarial Perturbation: Manipulating the Recognition of User Speech in Real  Time. In Network and Distributed System Security (NDSS) Symposium 2024."

**SubmitDate**: 2023-09-12    [abs](http://arxiv.org/abs/2308.01040v3) [paper-pdf](http://arxiv.org/pdf/2308.01040v3)

**Authors**: Xinfeng Li, Chen Yan, Xuancun Lu, Zihan Zeng, Xiaoyu Ji, Wenyuan Xu

**Abstract**: Automatic speech recognition (ASR) systems have been shown to be vulnerable to adversarial examples (AEs). Recent success all assumes that users will not notice or disrupt the attack process despite the existence of music/noise-like sounds and spontaneous responses from voice assistants. Nonetheless, in practical user-present scenarios, user awareness may nullify existing attack attempts that launch unexpected sounds or ASR usage. In this paper, we seek to bridge the gap in existing research and extend the attack to user-present scenarios. We propose VRIFLE, an inaudible adversarial perturbation (IAP) attack via ultrasound delivery that can manipulate ASRs as a user speaks. The inherent differences between audible sounds and ultrasounds make IAP delivery face unprecedented challenges such as distortion, noise, and instability. In this regard, we design a novel ultrasonic transformation model to enhance the crafted perturbation to be physically effective and even survive long-distance delivery. We further enable VRIFLE's robustness by adopting a series of augmentation on user and real-world variations during the generation process. In this way, VRIFLE features an effective real-time manipulation of the ASR output from different distances and under any speech of users, with an alter-and-mute strategy that suppresses the impact of user disruption. Our extensive experiments in both digital and physical worlds verify VRIFLE's effectiveness under various configurations, robustness against six kinds of defenses, and universality in a targeted manner. We also show that VRIFLE can be delivered with a portable attack device and even everyday-life loudspeakers.

摘要: 自动语音识别(ASR)系统已被证明容易受到对抗性例子(AE)的攻击。最近的成功都假设用户不会注意到或中断攻击过程，尽管存在类似音乐/噪音的声音和语音助手的自发响应。尽管如此，在实际的用户场景中，用户感知可能会使发出意外声音或ASR使用的现有攻击尝试无效。在本文中，我们试图弥补现有研究中的差距，并将攻击扩展到用户呈现的场景。我们提出了VRIFLE，这是一种通过超声波传输的听不见的对抗性扰动(IAP)攻击，可以在用户说话时操纵ASR。可听声音和超声波之间的固有差异使IAP交付面临前所未有的挑战，如失真、噪声和不稳定。在这方面，我们设计了一种新的超声变换模型来增强精心制作的微扰，使其在物理上有效，甚至可以在长距离传输中幸存下来。我们通过在生成过程中对用户和真实世界的变化采取一系列增强来进一步增强VRIFLE的健壮性。通过这种方式，VRIFLE具有在不同距离和用户任何语音下对ASR输出进行有效实时操作的特点，并采用更改和静音策略来抑制用户中断的影响。我们在数字和物理世界的广泛实验验证了VRIFLE在各种配置下的有效性、对六种防御的健壮性以及有针对性的普适性。我们还展示了VRIFLE可以与便携式攻击设备一起交付，甚至可以与日常生活中的扬声器一起交付。



## **20. Adversarial Attacks Assessment of Salient Object Detection via Symbolic Learning**

基于符号学习的显著目标检测的对抗性攻击评估 cs.CV

14 pages, 8 figures, 6 tables, IEEE Transactions on Emerging Topics  in Computing, Accepted for publication

**SubmitDate**: 2023-09-12    [abs](http://arxiv.org/abs/2309.05900v1) [paper-pdf](http://arxiv.org/pdf/2309.05900v1)

**Authors**: Gustavo Olague, Roberto Pineda, Gerardo Ibarra-Vazquez, Matthieu Olague, Axel Martinez, Sambit Bakshi, Jonathan Vargas, Isnardo Reducindo

**Abstract**: Machine learning is at the center of mainstream technology and outperforms classical approaches to handcrafted feature design. Aside from its learning process for artificial feature extraction, it has an end-to-end paradigm from input to output, reaching outstandingly accurate results. However, security concerns about its robustness to malicious and imperceptible perturbations have drawn attention since its prediction can be changed entirely. Salient object detection is a research area where deep convolutional neural networks have proven effective but whose trustworthiness represents a significant issue requiring analysis and solutions to hackers' attacks. Brain programming is a kind of symbolic learning in the vein of good old-fashioned artificial intelligence. This work provides evidence that symbolic learning robustness is crucial in designing reliable visual attention systems since it can withstand even the most intense perturbations. We test this evolutionary computation methodology against several adversarial attacks and noise perturbations using standard databases and a real-world problem of a shorebird called the Snowy Plover portraying a visual attention task. We compare our methodology with five different deep learning approaches, proving that they do not match the symbolic paradigm regarding robustness. All neural networks suffer significant performance losses, while brain programming stands its ground and remains unaffected. Also, by studying the Snowy Plover, we remark on the importance of security in surveillance activities regarding wildlife protection and conservation.

摘要: 机器学习处于主流技术的中心，其表现优于传统的手工特征设计方法。除了用于人工特征提取的学习过程外，它还具有从输入到输出的端到端范例，得出非常准确的结果。然而，由于其预测可以完全改变，对其对恶意和不可察觉的干扰的稳健性的安全担忧引起了人们的注意。显著目标检测是一个研究领域，深度卷积神经网络已被证明是有效的，但其可信度是一个需要分析和解决黑客攻击的重要问题。脑编程是一种符号学习，沿袭了优秀的老式人工智能的脉络。这项工作提供了证据，证明符号学习的稳健性在设计可靠的视觉注意系统时至关重要，因为它可以抵抗甚至最强烈的干扰。我们使用标准数据库和一个描述视觉注意任务的名为雪鸟的真实问题来测试这种进化计算方法，以对抗几种对抗性攻击和噪音扰动。我们将我们的方法与五种不同的深度学习方法进行了比较，证明它们与关于稳健性的符号范例不匹配。所有神经网络都遭受了严重的性能损失，而大脑编程则坚定不移，不受影响。此外，通过对雪禽的研究，我们指出了安全在野生动物保护和养护监测活动中的重要性。



## **21. Generalized Attacks on Face Verification Systems**

针对人脸验证系统的泛化攻击 cs.CR

**SubmitDate**: 2023-09-12    [abs](http://arxiv.org/abs/2309.05879v1) [paper-pdf](http://arxiv.org/pdf/2309.05879v1)

**Authors**: Ehsan Nazari, Paula Branco, Guy-Vincent Jourdan

**Abstract**: Face verification (FV) using deep neural network models has made tremendous progress in recent years, surpassing human accuracy and seeing deployment in various applications such as border control and smartphone unlocking. However, FV systems are vulnerable to Adversarial Attacks, which manipulate input images to deceive these systems in ways usually unnoticeable to humans. This paper provides an in-depth study of attacks on FV systems. We introduce the DodgePersonation Attack that formulates the creation of face images that impersonate a set of given identities while avoiding being identified as any of the identities in a separate, disjoint set. A taxonomy is proposed to provide a unified view of different types of Adversarial Attacks against FV systems, including Dodging Attacks, Impersonation Attacks, and Master Face Attacks. Finally, we propose the ''One Face to Rule Them All'' Attack which implements the DodgePersonation Attack with state-of-the-art performance on a well-known scenario (Master Face Attack) and which can also be used for the new scenarios introduced in this paper. While the state-of-the-art Master Face Attack can produce a set of 9 images to cover 43.82% of the identities in their test database, with 9 images our attack can cover 57.27% to 58.5% of these identifies while giving the attacker the choice of the identity to use to create the impersonation. Moreover, the 9 generated attack images appear identical to a casual observer.

摘要: 近年来，使用深度神经网络模型的人脸验证(FV)取得了巨大的进步，超过了人类的准确率，并在边境控制和智能手机解锁等各种应用中得到了部署。然而，FV系统容易受到敌意攻击，这些攻击操纵输入图像以通常不被人类注意到的方式欺骗这些系统。本文对FV系统遭受的攻击进行了深入的研究。我们引入了DodgePersonation攻击，该攻击描述了模拟一组给定身份的人脸图像的创建，同时避免被标识为独立的、不相交的集中的任何身份。提出了一种分类方法，以提供针对FV系统的不同类型的对抗性攻击的统一视图，包括躲避攻击、模仿攻击和主控面攻击。最后，我们提出了“一张脸统治所有人”的攻击，它在一个著名的场景(Master Face攻击)上实现了性能最好的DodgePersonation攻击，该攻击也可以用于本文介绍的新场景。虽然最先进的Master Face攻击可以生成一组9张图像来覆盖他们测试数据库中43.82%的身份，但我们的攻击可以用9张图像覆盖其中57.27%到58.5%的身份，同时让攻击者可以选择用来创建模仿的身份。此外，9张生成的攻击图像对于一个普通的观察者来说似乎是相同的。



## **22. Robust Feature-Level Adversaries are Interpretability Tools**

强大的功能级对手是可解释的工具 cs.LG

NeurIPS 2022, code available at  https://github.com/thestephencasper/feature_level_adv

**SubmitDate**: 2023-09-11    [abs](http://arxiv.org/abs/2110.03605v7) [paper-pdf](http://arxiv.org/pdf/2110.03605v7)

**Authors**: Stephen Casper, Max Nadeau, Dylan Hadfield-Menell, Gabriel Kreiman

**Abstract**: The literature on adversarial attacks in computer vision typically focuses on pixel-level perturbations. These tend to be very difficult to interpret. Recent work that manipulates the latent representations of image generators to create "feature-level" adversarial perturbations gives us an opportunity to explore perceptible, interpretable adversarial attacks. We make three contributions. First, we observe that feature-level attacks provide useful classes of inputs for studying representations in models. Second, we show that these adversaries are uniquely versatile and highly robust. We demonstrate that they can be used to produce targeted, universal, disguised, physically-realizable, and black-box attacks at the ImageNet scale. Third, we show how these adversarial images can be used as a practical interpretability tool for identifying bugs in networks. We use these adversaries to make predictions about spurious associations between features and classes which we then test by designing "copy/paste" attacks in which one natural image is pasted into another to cause a targeted misclassification. Our results suggest that feature-level attacks are a promising approach for rigorous interpretability research. They support the design of tools to better understand what a model has learned and diagnose brittle feature associations. Code is available at https://github.com/thestephencasper/feature_level_adv

摘要: 关于计算机视觉中的对抗性攻击的文献通常集中在像素级的扰动上。这些往往很难解释。最近的工作是利用图像生成器的潜在表示来创建“特征级别”的对抗性扰动，这给了我们一个探索可感知的、可解释的对抗性攻击的机会。我们有三点贡献。首先，我们观察到特征级别的攻击为学习模型中的表示提供了有用的输入类。其次，我们展示了这些对手独一无二的多才多艺和高度健壮。我们证明了它们可以用于在ImageNet规模上产生有针对性的、普遍的、伪装的、物理上可实现的和黑匣子攻击。第三，我们展示了如何将这些对抗性图像用作识别网络漏洞的实用可解释性工具。我们利用这些对手来预测特征和类别之间的虚假关联，然后通过设计“复制/粘贴”攻击来测试这些关联，在这种攻击中，一幅自然图像被粘贴到另一幅图像中，从而导致有针对性的误分类。我们的结果表明，特征级攻击对于严格的可解释性研究是一种很有前途的方法。它们支持工具的设计，以更好地理解模型学习到的内容并诊断脆弱的特征关联。代码可在https://github.com/thestephencasper/feature_level_adv上找到



## **23. Efficient Defense Against Model Stealing Attacks on Convolutional Neural Networks**

卷积神经网络模型窃取攻击的有效防御 cs.LG

Accepted for publication at 2023 International Conference on Machine  Learning and Applications (ICMLA). Proceedings of ICMLA, Florida, USA  \c{opyright}2023 IEEE

**SubmitDate**: 2023-09-11    [abs](http://arxiv.org/abs/2309.01838v2) [paper-pdf](http://arxiv.org/pdf/2309.01838v2)

**Authors**: Kacem Khaled, Mouna Dhaouadi, Felipe Gohring de Magalhães, Gabriela Nicolescu

**Abstract**: Model stealing attacks have become a serious concern for deep learning models, where an attacker can steal a trained model by querying its black-box API. This can lead to intellectual property theft and other security and privacy risks. The current state-of-the-art defenses against model stealing attacks suggest adding perturbations to the prediction probabilities. However, they suffer from heavy computations and make impracticable assumptions about the adversary. They often require the training of auxiliary models. This can be time-consuming and resource-intensive which hinders the deployment of these defenses in real-world applications. In this paper, we propose a simple yet effective and efficient defense alternative. We introduce a heuristic approach to perturb the output probabilities. The proposed defense can be easily integrated into models without additional training. We show that our defense is effective in defending against three state-of-the-art stealing attacks. We evaluate our approach on large and quantized (i.e., compressed) Convolutional Neural Networks (CNNs) trained on several vision datasets. Our technique outperforms the state-of-the-art defenses with a $\times37$ faster inference latency without requiring any additional model and with a low impact on the model's performance. We validate that our defense is also effective for quantized CNNs targeting edge devices.

摘要: 模型窃取攻击已经成为深度学习模型的一个严重问题，在深度学习模型中，攻击者可以通过查询黑盒API来窃取训练的模型。这可能会导致知识产权被盗以及其他安全和隐私风险。目前针对模型窃取攻击的最先进防御措施建议增加预测概率的扰动。然而，他们遭受着繁重的计算，并对对手做出不切实际的假设。它们往往需要辅助模型的培训。这可能会耗费时间和资源，从而阻碍在实际应用程序中部署这些防御措施。在本文中，我们提出了一种简单而有效的防御方案。我们引入了一种启发式方法来扰动输出概率。建议的防御可以很容易地集成到模型中，而不需要额外的培训。我们表明，我们的防御在防御三种最先进的窃取攻击方面是有效的。我们在几个视觉数据集上训练的大型量化(即压缩)卷积神经网络(CNN)上对我们的方法进行了评估。我们的技术优于最先进的防御技术，在不需要任何额外模型的情况下，推理延迟快37倍，并且对模型性能的影响很小。我们验证了我们的防御对于针对边缘设备的量化CNN也是有效的。



## **24. Byzantine Multiple Access Channels -- Part I: Reliable Communication**

拜占庭式多址接入信道--第一部分：可靠通信 cs.IT

This supercedes Part I of arxiv:1904.11925

**SubmitDate**: 2023-09-11    [abs](http://arxiv.org/abs/2211.12769v3) [paper-pdf](http://arxiv.org/pdf/2211.12769v3)

**Authors**: Neha Sangwan, Mayank Bakshi, Bikash Kumar Dey, Vinod M. Prabhakaran

**Abstract**: We study communication over a Multiple Access Channel (MAC) where users can possibly be adversarial. The receiver is unaware of the identity of the adversarial users (if any). When all users are non-adversarial, we want their messages to be decoded reliably. When a user behaves adversarially, we require that the honest users' messages be decoded reliably. An adversarial user can mount an attack by sending any input into the channel rather than following the protocol. It turns out that the $2$-user MAC capacity region follows from the point-to-point Arbitrarily Varying Channel (AVC) capacity. For the $3$-user MAC in which at most one user may be malicious, we characterize the capacity region for deterministic codes and randomized codes (where each user shares an independent random secret key with the receiver). These results are then generalized for the $k$-user MAC where the adversary may control all users in one out of a collection of given subsets.

摘要: 我们研究了多路访问信道(MAC)上的通信，其中用户可能是对抗性的。接收方不知道敌对用户(如果有的话)的身份。当所有用户都是非对抗性的时，我们希望他们的消息被可靠地解码。当用户做出恶意行为时，我们要求可靠地解码诚实用户的消息。敌意用户可以通过向通道发送任何输入而不是遵循协议来发动攻击。事实证明，$2$-用户MAC容量区域紧随点对点任意变化信道(AVC)容量。对于最多一个用户可能是恶意用户的$3$-用户MAC，我们刻画了确定码和随机码的容量域(其中每个用户与接收方共享一个独立的随机密钥)。然后将这些结果推广到$k$-用户MAC，其中对手可以控制给定子集集合中的一个用户。



## **25. Boosting Adversarial Transferability with Learnable Patch-wise Masks**

用可学习的补丁口罩提高对手的可转移性 cs.CV

**SubmitDate**: 2023-09-11    [abs](http://arxiv.org/abs/2306.15931v2) [paper-pdf](http://arxiv.org/pdf/2306.15931v2)

**Authors**: Xingxing Wei, Shiji Zhao

**Abstract**: Adversarial examples have attracted widespread attention in security-critical applications because of their transferability across different models. Although many methods have been proposed to boost adversarial transferability, a gap still exists between capabilities and practical demand. In this paper, we argue that the model-specific discriminative regions are a key factor causing overfitting to the source model, and thus reducing the transferability to the target model. For that, a patch-wise mask is utilized to prune the model-specific regions when calculating adversarial perturbations. To accurately localize these regions, we present a learnable approach to automatically optimize the mask. Specifically, we simulate the target models in our framework, and adjust the patch-wise mask according to the feedback of the simulated models. To improve the efficiency, the differential evolutionary (DE) algorithm is utilized to search for patch-wise masks for a specific image. During iterative attacks, the learned masks are applied to the image to drop out the patches related to model-specific regions, thus making the gradients more generic and improving the adversarial transferability. The proposed approach is a preprocessing method and can be integrated with existing methods to further boost the transferability. Extensive experiments on the ImageNet dataset demonstrate the effectiveness of our method. We incorporate the proposed approach with existing methods to perform ensemble attacks and achieve an average success rate of 93.01% against seven advanced defense methods, which can effectively enhance the state-of-the-art transfer-based attack performance.

摘要: 对抗性例子由于可以在不同的模型之间转移而在安全关键型应用中引起了广泛的关注。虽然已经提出了许多方法来提高对抗性转移能力，但在能力和实际需求之间仍然存在差距。在本文中，我们认为特定于模型的区分区域是导致源模型过度拟合从而降低到目标模型的可转换性的关键因素。为此，在计算对抗性扰动时，使用补丁掩码来修剪特定于模型的区域。为了准确地定位这些区域，我们提出了一种可学习的方法来自动优化掩码。具体地说，我们在我们的框架中模拟目标模型，并根据模拟模型的反馈调整面片掩码。为了提高搜索效率，采用了差分进化算法来搜索特定图像的面片掩模。在迭代攻击过程中，将学习到的模板应用于图像，去除与模型特定区域相关的补丁，从而使梯度更具通用性，提高了对抗可转移性。该方法是一种预处理方法，可以与现有方法相结合，进一步提高可转移性。在ImageNet数据集上的大量实验证明了该方法的有效性。我们将该方法与现有的集成攻击方法相结合，对7种先进的防御方法取得了93.01%的平均成功率，有效地提高了基于传输的攻击性能。



## **26. GIFD: A Generative Gradient Inversion Method with Feature Domain Optimization**

GIFD：一种基于特征域优化的产生式梯度反演方法 cs.CV

ICCV 2023

**SubmitDate**: 2023-09-11    [abs](http://arxiv.org/abs/2308.04699v2) [paper-pdf](http://arxiv.org/pdf/2308.04699v2)

**Authors**: Hao Fang, Bin Chen, Xuan Wang, Zhi Wang, Shu-Tao Xia

**Abstract**: Federated Learning (FL) has recently emerged as a promising distributed machine learning framework to preserve clients' privacy, by allowing multiple clients to upload the gradients calculated from their local data to a central server. Recent studies find that the exchanged gradients also take the risk of privacy leakage, e.g., an attacker can invert the shared gradients and recover sensitive data against an FL system by leveraging pre-trained generative adversarial networks (GAN) as prior knowledge. However, performing gradient inversion attacks in the latent space of the GAN model limits their expression ability and generalizability. To tackle these challenges, we propose \textbf{G}radient \textbf{I}nversion over \textbf{F}eature \textbf{D}omains (GIFD), which disassembles the GAN model and searches the feature domains of the intermediate layers. Instead of optimizing only over the initial latent code, we progressively change the optimized layer, from the initial latent space to intermediate layers closer to the output images. In addition, we design a regularizer to avoid unreal image generation by adding a small ${l_1}$ ball constraint to the searching range. We also extend GIFD to the out-of-distribution (OOD) setting, which weakens the assumption that the training sets of GANs and FL tasks obey the same data distribution. Extensive experiments demonstrate that our method can achieve pixel-level reconstruction and is superior to the existing methods. Notably, GIFD also shows great generalizability under different defense strategy settings and batch sizes.

摘要: 联合学习(FL)是最近出现的一种很有前途的分布式机器学习框架，通过允许多个客户端将从他们的本地数据计算出的梯度上传到中央服务器，来保护客户的隐私。最近的研究发现，交换的梯度也存在隐私泄露的风险，例如，攻击者可以利用预先训练的生成性对抗网络(GAN)作为先验知识来反转共享的梯度，并针对FL系统恢复敏感数据。然而，在GaN模型的潜在空间中进行梯度反转攻击限制了其表达能力和泛化能力。为了解决这些问题，我们提出了一种新的GIFD算法，即通过分解GaN模型并搜索中间层的特征域来实现对Textbf{F}eature\Textbf{D}omains的转换。我们不是只对初始的潜在代码进行优化，而是逐步地将优化的层从初始的潜在空间更改为更接近输出图像的中间层。此外，通过在搜索范围中添加一个较小的${L_1}$球约束，我们设计了一个正则化算法来避免产生虚幻图像。我们还将GIFD扩展到超出分布(OOD)环境，削弱了GANS和FL任务的训练集服从相同数据分布的假设。大量实验表明，该方法能够实现像素级重建，且优于现有的重建方法。值得注意的是，GIFD在不同的防御策略设置和批量大小下也显示出很好的通用性。



## **27. Outlier Robust Adversarial Training**

异常点稳健对抗训练 cs.LG

Accepted by The 15th Asian Conference on Machine Learning (ACML 2023)

**SubmitDate**: 2023-09-10    [abs](http://arxiv.org/abs/2309.05145v1) [paper-pdf](http://arxiv.org/pdf/2309.05145v1)

**Authors**: Shu Hu, Zhenhuan Yang, Xin Wang, Yiming Ying, Siwei Lyu

**Abstract**: Supervised learning models are challenged by the intrinsic complexities of training data such as outliers and minority subpopulations and intentional attacks at inference time with adversarial samples. While traditional robust learning methods and the recent adversarial training approaches are designed to handle each of the two challenges, to date, no work has been done to develop models that are robust with regard to the low-quality training data and the potential adversarial attack at inference time simultaneously. It is for this reason that we introduce Outlier Robust Adversarial Training (ORAT) in this work. ORAT is based on a bi-level optimization formulation of adversarial training with a robust rank-based loss function. Theoretically, we show that the learning objective of ORAT satisfies the $\mathcal{H}$-consistency in binary classification, which establishes it as a proper surrogate to adversarial 0/1 loss. Furthermore, we analyze its generalization ability and provide uniform convergence rates in high probability. ORAT can be optimized with a simple algorithm. Experimental evaluations on three benchmark datasets demonstrate the effectiveness and robustness of ORAT in handling outliers and adversarial attacks. Our code is available at https://github.com/discovershu/ORAT.

摘要: 监督学习模型受到训练数据内在复杂性的挑战，例如异常值和少数子群，以及在推理时使用对抗性样本进行故意攻击。虽然传统的稳健学习方法和最近的对抗性训练方法是为了应对这两个挑战中的每一个，但到目前为止，还没有做过任何工作来开发同时针对低质量训练数据和推理时潜在的对抗性攻击的稳健模型。正是出于这个原因，我们在这项工作中引入了离群点稳健对抗训练(ORAT)。ORAT基于对抗性训练的双层优化公式，具有稳健的基于等级的损失函数。理论上，我们证明了ORAT的学习目标满足二进制分类中数学上的{H}-一致性，从而使其成为对抗0/1损失的合适替代品。此外，我们还分析了它的泛化能力，给出了高概率下的一致收敛速度。ORAT可以用一个简单的算法进行优化。在三个基准数据集上的实验评估表明，ORAT在处理孤立点和对抗性攻击方面是有效的和健壮的。我们的代码可以在https://github.com/discovershu/ORAT.上找到



## **28. DAD++: Improved Data-free Test Time Adversarial Defense**

DAD++：改进的无数据测试时间对抗性防御 cs.CV

IJCV Journal (Under Review)

**SubmitDate**: 2023-09-10    [abs](http://arxiv.org/abs/2309.05132v1) [paper-pdf](http://arxiv.org/pdf/2309.05132v1)

**Authors**: Gaurav Kumar Nayak, Inder Khatri, Shubham Randive, Ruchit Rawal, Anirban Chakraborty

**Abstract**: With the increasing deployment of deep neural networks in safety-critical applications such as self-driving cars, medical imaging, anomaly detection, etc., adversarial robustness has become a crucial concern in the reliability of these networks in real-world scenarios. A plethora of works based on adversarial training and regularization-based techniques have been proposed to make these deep networks robust against adversarial attacks. However, these methods require either retraining models or training them from scratch, making them infeasible to defend pre-trained models when access to training data is restricted. To address this problem, we propose a test time Data-free Adversarial Defense (DAD) containing detection and correction frameworks. Moreover, to further improve the efficacy of the correction framework in cases when the detector is under-confident, we propose a soft-detection scheme (dubbed as "DAD++"). We conduct a wide range of experiments and ablations on several datasets and network architectures to show the efficacy of our proposed approach. Furthermore, we demonstrate the applicability of our approach in imparting adversarial defense at test time under data-free (or data-efficient) applications/setups, such as Data-free Knowledge Distillation and Source-free Unsupervised Domain Adaptation, as well as Semi-supervised classification frameworks. We observe that in all the experiments and applications, our DAD++ gives an impressive performance against various adversarial attacks with a minimal drop in clean accuracy. The source code is available at: https://github.com/vcl-iisc/Improved-Data-free-Test-Time-Adversarial-Defense

摘要: 随着深度神经网络在自动驾驶汽车、医学成像、异常检测等安全关键应用中的应用越来越多，对抗健壮性已经成为这些网络在现实世界场景中可靠性的关键问题。基于对抗性训练和正则化技术的大量工作已经被提出，以使这些深层网络对对抗性攻击具有健壮性。然而，这些方法要么需要重新训练模型，要么需要从头开始训练模型，这使得它们在访问训练数据受到限制时无法为预先训练的模型辩护。为了解决这一问题，我们提出了一种包含检测和纠正框架的无测试时间数据的对抗防御(DAD)。此外，为了进一步提高校正框架在检测器信心不足的情况下的有效性，我们提出了一种软检测方案(称为DAD++)。我们在几个数据集和网络架构上进行了广泛的实验和烧蚀，以显示我们所提出的方法的有效性。此外，我们展示了我们的方法在测试时在无数据(或数据高效)应用/设置下的适用性，例如无数据知识蒸馏和无源无监督领域适应，以及半监督分类框架。我们观察到，在所有的实验和应用中，我们的DAD++在抵抗各种对手攻击时表现出令人印象深刻的性能，并且在干净的准确率上下降最小。源代码可在https://github.com/vcl-iisc/Improved-Data-free-Test-Time-Adversarial-Defense上找到



## **29. Attacking c-MARL More Effectively: A Data Driven Approach**

更有效地攻击c-Marl：数据驱动的方法 cs.LG

**SubmitDate**: 2023-09-10    [abs](http://arxiv.org/abs/2202.03558v2) [paper-pdf](http://arxiv.org/pdf/2202.03558v2)

**Authors**: Nhan H. Pham, Lam M. Nguyen, Jie Chen, Hoang Thanh Lam, Subhro Das, Tsui-Wei Weng

**Abstract**: In recent years, a proliferation of methods were developed for cooperative multi-agent reinforcement learning (c-MARL). However, the robustness of c-MARL agents against adversarial attacks has been rarely explored. In this paper, we propose to evaluate the robustness of c-MARL agents via a model-based approach, named c-MBA. Our proposed formulation can craft much stronger adversarial state perturbations of c-MARL agents to lower total team rewards than existing model-free approaches. In addition, we propose the first victim-agent selection strategy and the first data-driven approach to define targeted failure states where each of them allows us to develop even stronger adversarial attack without the expert knowledge to the underlying environment. Our numerical experiments on two representative MARL benchmarks illustrate the advantage of our approach over other baselines: our model-based attack consistently outperforms other baselines in all tested environments.

摘要: 近年来，协作多智能体强化学习(c-Marl)方法层出不穷。然而，c-Marl代理抵抗敌意攻击的健壮性很少被探索。在本文中，我们提出了一种基于模型的方法来评估c-Marl代理的健壮性，称为c-MBA。与现有的无模型方法相比，我们提出的公式可以对c-Marl代理进行更强的对抗性状态扰动，以降低团队总奖励。此外，我们提出了第一种受害者-代理选择策略和第一种数据驱动的方法来定义目标失效状态，每种策略都允许我们在没有对底层环境的专业知识的情况下开发出更强的对抗性攻击。我们在两个具有代表性的Marl基准上进行的数值实验表明了我们的方法相对于其他基线的优势：在所有测试环境中，我们基于模型的攻击始终优于其他基线。



## **30. Secure Set-Based State Estimation for Linear Systems under Adversarial Attacks on Sensors**

传感器受到敌意攻击时线性系统基于集合的安全状态估计 eess.SY

**SubmitDate**: 2023-09-10    [abs](http://arxiv.org/abs/2309.05075v1) [paper-pdf](http://arxiv.org/pdf/2309.05075v1)

**Authors**: Muhammad Umar B. Niazi, Michelle S. Chong, Amr Alanwar, Karl H. Johansson

**Abstract**: When a strategic adversary can attack multiple sensors of a system and freely choose a different set of sensors at different times, how can we ensure that the state estimate remains uncorrupted by the attacker? The existing literature addressing this problem mandates that the adversary can only corrupt less than half of the total number of sensors. This limitation is fundamental to all point-based secure state estimators because of their dependence on algorithms that rely on majority voting among sensors. However, in reality, an adversary with ample resources may not be limited to attacking less than half of the total number of sensors. This paper avoids the above-mentioned fundamental limitation by proposing a set-based approach that allows attacks on all but one sensor at any given time. We guarantee that the true state is always contained in the estimated set, which is represented by a collection of constrained zonotopes, provided that the system is bounded-input-bounded-state stable and redundantly observable via every combination of sensor subsets with size equal to the number of uncompromised sensors. Additionally, we show that the estimated set is secure and stable irrespective of the attack signals if the process and measurement noises are bounded. To detect the set of attacked sensors at each time, we propose a simple attack detection technique. However, we acknowledge that intelligently designed stealthy attacks may not be detected and, in the worst-case scenario, could even result in exponential growth in the algorithm's complexity. We alleviate this shortcoming by presenting a range of strategies that offer different levels of trade-offs between estimation performance and complexity.

摘要: 当一个战略对手可以攻击一个系统的多个传感器并在不同的时间自由选择一组不同的传感器时，我们如何确保状态估计不被攻击者破坏？解决这个问题的现有文献要求对手只能破坏不到传感器总数的一半。这一限制是所有基于点的安全状态估计器的基础，因为它们依赖于依赖于传感器之间的多数投票的算法。然而，在现实中，拥有充足资源的对手可能不会局限于攻击不到传感器总数的一半。本文提出了一种基于集合的方法，允许在任何给定时间对除一个传感器之外的所有传感器进行攻击，从而避免了上述基本限制。如果系统是有界输入有界状态稳定的，并且通过每个传感器子集的组合冗余可观测，我们保证真实状态总是包含在估计集内，该估计集由约束区域的集合来表示，且每个传感器子集的大小等于不妥协的传感器数目。此外，我们还证明了当过程噪声和测量噪声有界时，无论攻击信号是什么，估计集都是安全稳定的。为了在每个时刻检测被攻击的传感器集合，我们提出了一种简单的攻击检测技术。然而，我们承认，智能设计的隐形攻击可能不会被检测到，在最糟糕的情况下，甚至可能导致算法复杂性的指数增长。我们通过提供一系列策略来缓解这一缺点，这些策略在估计性能和复杂性之间提供了不同程度的权衡。



## **31. Machine Translation Models Stand Strong in the Face of Adversarial Attacks**

机器翻译模型在敌意攻击面前屹立不倒 cs.CL

**SubmitDate**: 2023-09-10    [abs](http://arxiv.org/abs/2309.06527v1) [paper-pdf](http://arxiv.org/pdf/2309.06527v1)

**Authors**: Pavel Burnyshev, Elizaveta Kostenok, Alexey Zaytsev

**Abstract**: Adversarial attacks expose vulnerabilities of deep learning models by introducing minor perturbations to the input, which lead to substantial alterations in the output. Our research focuses on the impact of such adversarial attacks on sequence-to-sequence (seq2seq) models, specifically machine translation models. We introduce algorithms that incorporate basic text perturbation heuristics and more advanced strategies, such as the gradient-based attack, which utilizes a differentiable approximation of the inherently non-differentiable translation metric. Through our investigation, we provide evidence that machine translation models display robustness displayed robustness against best performed known adversarial attacks, as the degree of perturbation in the output is directly proportional to the perturbation in the input. However, among underdogs, our attacks outperform alternatives, providing the best relative performance. Another strong candidate is an attack based on mixing of individual characters.

摘要: 对抗性攻击通过在输入中引入微小的扰动来暴露深度学习模型的漏洞，从而导致输出的实质性变化。我们的研究集中在这种敌意攻击对序列到序列(Seq2seq)模型的影响，特别是机器翻译模型。我们介绍的算法结合了基本的文本扰动启发式算法和更高级的策略，例如基于梯度的攻击，它利用固有的不可微翻译度量的可微近似。通过我们的研究，我们提供了证据，证明机器翻译模型对性能最好的对手攻击表现出健壮性，因为输出中的扰动程度与输入中的扰动程度成正比。然而，在失败者中，我们的攻击表现优于其他选择，提供了最佳的相对性能。另一个强有力的候选者是基于单个字符混合的攻击。



## **32. Mitigating Adversarial Attacks in Deepfake Detection: An Exploration of Perturbation and AI Techniques**

深度伪码检测中对抗攻击的缓解：扰动和人工智能技术的探讨 cs.LG

**SubmitDate**: 2023-09-10    [abs](http://arxiv.org/abs/2302.11704v2) [paper-pdf](http://arxiv.org/pdf/2302.11704v2)

**Authors**: Saminder Dhesi, Laura Fontes, Pedro Machado, Isibor Kennedy Ihianle, Farhad Fassihi Tash, David Ada Adama

**Abstract**: Deep learning constitutes a pivotal component within the realm of machine learning, offering remarkable capabilities in tasks ranging from image recognition to natural language processing. However, this very strength also renders deep learning models susceptible to adversarial examples, a phenomenon pervasive across a diverse array of applications. These adversarial examples are characterized by subtle perturbations artfully injected into clean images or videos, thereby causing deep learning algorithms to misclassify or produce erroneous outputs. This susceptibility extends beyond the confines of digital domains, as adversarial examples can also be strategically designed to target human cognition, leading to the creation of deceptive media, such as deepfakes. Deepfakes, in particular, have emerged as a potent tool to manipulate public opinion and tarnish the reputations of public figures, underscoring the urgent need to address the security and ethical implications associated with adversarial examples. This article delves into the multifaceted world of adversarial examples, elucidating the underlying principles behind their capacity to deceive deep learning algorithms. We explore the various manifestations of this phenomenon, from their insidious role in compromising model reliability to their impact in shaping the contemporary landscape of disinformation and misinformation. To illustrate progress in combating adversarial examples, we showcase the development of a tailored Convolutional Neural Network (CNN) designed explicitly to detect deepfakes, a pivotal step towards enhancing model robustness in the face of adversarial threats. Impressively, this custom CNN has achieved a precision rate of 76.2% on the DFDC dataset.

摘要: 深度学习是机器学习领域的关键组成部分，在从图像识别到自然语言处理的各种任务中提供了非凡的能力。然而，这一优势也使得深度学习模型容易受到对抗性例子的影响，这一现象普遍存在于各种应用程序中。这些对抗性的例子的特点是巧妙地将微妙的扰动注入干净的图像或视频中，从而导致深度学习算法错误分类或产生错误的输出。这种敏感性超出了数字领域的范围，因为敌意例子也可以被战略性地设计成针对人类认知的，从而导致欺骗性媒体的创造，如深度假冒。特别是，Deepfake已经成为操纵公众舆论和玷污公众人物声誉的有力工具，突显出迫切需要解决与对抗性例子相关的安全和道德影响。这篇文章深入研究了对抗性例子的多方面世界，阐明了它们欺骗深度学习算法的能力背后的潜在原理。我们探讨了这种现象的各种表现，从它们在损害模型可靠性方面的隐秘作用，到它们在塑造当代虚假信息和错误信息景观中的影响。为了说明在对抗敌意例子方面的进展，我们展示了专门为检测深度假象而设计的卷积神经网络(CNN)的发展，这是在面对对抗性威胁时增强模型稳健性的关键一步。令人印象深刻的是，这种定制的CNN在DFDC数据集上的准确率达到了76.2%。



## **33. A Diamond Model Analysis on Twitter's Biggest Hack**

Twitter最大黑客攻击的钻石模型分析 cs.CR

Discrepancies in the paper

**SubmitDate**: 2023-09-09    [abs](http://arxiv.org/abs/2306.15878v2) [paper-pdf](http://arxiv.org/pdf/2306.15878v2)

**Authors**: Chaitanya Rahalkar

**Abstract**: Cyberattacks have prominently increased over the past few years now, and have targeted actors from a wide variety of domains. Understanding the motivation, infrastructure, attack vectors, etc. behind such attacks is vital to proactively work against preventing such attacks in the future and also to analyze the economic and social impact of such attacks. In this paper, we leverage the diamond model to perform an intrusion analysis case study of the 2020 Twitter account hijacking Cyberattack. We follow this standardized incident response model to map the adversary, capability, infrastructure, and victim and perform a comprehensive analysis of the attack, and the impact posed by the attack from a Cybersecurity policy standpoint.

摘要: 网络攻击在过去几年里显著增加，目标是来自不同领域的参与者。了解此类攻击背后的动机、基础设施、攻击媒介等对于主动预防未来此类攻击以及分析此类攻击的经济和社会影响至关重要。在本文中，我们利用钻石模型对2020年Twitter账户劫持网络攻击进行了入侵分析案例研究。我们遵循这个标准化的事件响应模型来映射对手、能力、基础设施和受害者，并从网络安全策略的角度对攻击和攻击造成的影响进行全面分析。



## **34. Good-looking but Lacking Faithfulness: Understanding Local Explanation Methods through Trend-based Testing**

好看但缺乏忠实性：通过趋势测试了解当地的解释方法 cs.LG

**SubmitDate**: 2023-09-09    [abs](http://arxiv.org/abs/2309.05679v1) [paper-pdf](http://arxiv.org/pdf/2309.05679v1)

**Authors**: Jinwen He, Kai Chen, Guozhu Meng, Jiangshan Zhang, Congyi Li

**Abstract**: While enjoying the great achievements brought by deep learning (DL), people are also worried about the decision made by DL models, since the high degree of non-linearity of DL models makes the decision extremely difficult to understand. Consequently, attacks such as adversarial attacks are easy to carry out, but difficult to detect and explain, which has led to a boom in the research on local explanation methods for explaining model decisions. In this paper, we evaluate the faithfulness of explanation methods and find that traditional tests on faithfulness encounter the random dominance problem, \ie, the random selection performs the best, especially for complex data. To further solve this problem, we propose three trend-based faithfulness tests and empirically demonstrate that the new trend tests can better assess faithfulness than traditional tests on image, natural language and security tasks. We implement the assessment system and evaluate ten popular explanation methods. Benefiting from the trend tests, we successfully assess the explanation methods on complex data for the first time, bringing unprecedented discoveries and inspiring future research. Downstream tasks also greatly benefit from the tests. For example, model debugging equipped with faithful explanation methods performs much better for detecting and correcting accuracy and security problems.

摘要: 在享受深度学习带来的巨大成就的同时，人们也对深度学习模型所做出的决策感到担忧，因为深度学习模型的高度非线性使得决策极其难以理解。因此，对抗性攻击等攻击容易实施，但很难检测和解释，这导致了解释模型决策的局部解释方法的研究热潮。本文对解释方法的可信性进行了评价，发现传统的可信性检验遇到了随机优势问题，即随机选择的检验效果最好，特别是对于复杂的数据。为了进一步解决这一问题，我们提出了三种基于趋势的忠诚度测试，并实证证明新的趋势测试比传统的关于图像、自然语言和安全任务的测试能够更好地评估忠诚度。我们实施了这一评估体系，并对十种流行的解释方法进行了评估。得益于趋势检验，我们首次成功地对复杂数据的解释方法进行了评估，带来了前所未有的发现，并启发了未来的研究。下游任务也从测试中受益匪浅。例如，配备忠实解释方法的模型调试在检测和纠正准确性和安全性问题方面表现得更好。



## **35. Exploring Robust Features for Improving Adversarial Robustness**

探索提高对手健壮性的健壮性特征 cs.CV

12 pages, 8 figures

**SubmitDate**: 2023-09-09    [abs](http://arxiv.org/abs/2309.04650v1) [paper-pdf](http://arxiv.org/pdf/2309.04650v1)

**Authors**: Hong Wang, Yuefan Deng, Shinjae Yoo, Yuewei Lin

**Abstract**: While deep neural networks (DNNs) have revolutionized many fields, their fragility to carefully designed adversarial attacks impedes the usage of DNNs in safety-critical applications. In this paper, we strive to explore the robust features which are not affected by the adversarial perturbations, i.e., invariant to the clean image and its adversarial examples, to improve the model's adversarial robustness. Specifically, we propose a feature disentanglement model to segregate the robust features from non-robust features and domain specific features. The extensive experiments on four widely used datasets with different attacks demonstrate that robust features obtained from our model improve the model's adversarial robustness compared to the state-of-the-art approaches. Moreover, the trained domain discriminator is able to identify the domain specific features from the clean images and adversarial examples almost perfectly. This enables adversarial example detection without incurring additional computational costs. With that, we can also specify different classifiers for clean images and adversarial examples, thereby avoiding any drop in clean image accuracy.

摘要: 虽然深度神经网络(DNN)已经给许多领域带来了革命性的变化，但它们对精心设计的对手攻击的脆弱性阻碍了DNN在安全关键应用中的使用。在本文中，我们努力探索不受对抗性扰动影响的稳健特征，即对干净图像及其对抗性实例的不变性，以提高模型的对抗性。具体地说，我们提出了一种特征解缠模型，将稳健特征与非稳健特征和领域特定特征分离开来。在四个具有不同攻击的广泛使用的数据集上的广泛实验表明，与最新的方法相比，从我们的模型获得的稳健特征提高了模型的对抗鲁棒性。此外，经过训练的领域鉴别器能够几乎完全地从干净的图像和敌意例子中识别领域特定的特征。这使得能够在不招致额外计算成本的情况下检测敌意示例。有了这一点，我们还可以为干净的图像和敌对的例子指定不同的分类器，从而避免干净图像准确率的任何下降。



## **36. Avoid Adversarial Adaption in Federated Learning by Multi-Metric Investigations**

通过多度量调查避免联合学习中的对抗性适应 cs.LG

25 pages, 14 figures, 23 tables, 11 equations

**SubmitDate**: 2023-09-08    [abs](http://arxiv.org/abs/2306.03600v2) [paper-pdf](http://arxiv.org/pdf/2306.03600v2)

**Authors**: Torsten Krauß, Alexandra Dmitrienko

**Abstract**: Federated Learning (FL) facilitates decentralized machine learning model training, preserving data privacy, lowering communication costs, and boosting model performance through diversified data sources. Yet, FL faces vulnerabilities such as poisoning attacks, undermining model integrity with both untargeted performance degradation and targeted backdoor attacks. Preventing backdoors proves especially challenging due to their stealthy nature.   Prominent mitigation techniques against poisoning attacks rely on monitoring certain metrics and filtering malicious model updates. While shown effective in evaluations, we argue that previous works didn't consider realistic real-world adversaries and data distributions. We define a new notion of strong adaptive adversaries, capable of adapting to multiple objectives simultaneously. Through extensive empirical tests, we show that existing defense methods can be easily circumvented in this adversary model. We also demonstrate, that existing defenses have limited effectiveness when no assumptions are made about underlying data distributions.   We introduce Metric-Cascades (MESAS), a novel defense method for more realistic scenarios and adversary models. MESAS employs multiple detection metrics simultaneously to identify poisoned model updates, creating a complex multi-objective optimization problem for adaptive attackers. In our extensive evaluation featuring nine backdoors and three datasets, MESAS consistently detects even strong adaptive attackers. Furthermore, MESAS outperforms existing defenses in distinguishing backdoors from data distribution-related distortions within and across clients. MESAS is the first defense robust against strong adaptive adversaries, effective in real-world data scenarios, with an average overhead of just 24.37 seconds.

摘要: 联合学习(FL)有助于分散机器学习模型的训练，保护数据隐私，降低通信成本，并通过多样化的数据源提高模型性能。然而，FL面临着中毒攻击、无目标性能降级和有针对性的后门攻击等漏洞，破坏了模型的完整性。事实证明，由于后门的隐蔽性，防止后门特别具有挑战性。针对中毒攻击的突出缓解技术依赖于监控某些指标和过滤恶意模型更新。虽然在评估中显示了有效性，但我们认为以前的工作没有考虑现实世界中的对手和数据分布。我们定义了一种新的概念，即强适应性对手，能够同时适应多个目标。通过广泛的实证测试，我们表明现有的防御方法可以很容易地在这个对手模型中被绕过。我们还证明，当没有对潜在的数据分布做出假设时，现有的防御措施的有效性有限。我们介绍了Metric-Cascade(MEAS)，这是一种新的防御方法，适用于更真实的场景和对手模型。MEAS同时使用多个检测指标来识别有毒的模型更新，为适应性攻击者创造了一个复杂的多目标优化问题。在我们广泛的评估中，包括九个后门和三个数据集，MEAS一致地检测到即使是强大的适应性攻击者。此外，MEAS在区分后门与客户内部和客户之间与数据分发相关的扭曲方面优于现有的防御措施。MEAS是第一个针对强大自适应对手的稳健防御，在真实数据场景中有效，平均开销仅为24.37秒。



## **37. Verifiable Learning for Robust Tree Ensembles**

用于稳健树集成的可验证学习 cs.LG

19 pages, 5 figures; full version of the revised paper accepted at  ACM CCS 2023

**SubmitDate**: 2023-09-08    [abs](http://arxiv.org/abs/2305.03626v2) [paper-pdf](http://arxiv.org/pdf/2305.03626v2)

**Authors**: Stefano Calzavara, Lorenzo Cazzaro, Giulio Ermanno Pibiri, Nicola Prezza

**Abstract**: Verifying the robustness of machine learning models against evasion attacks at test time is an important research problem. Unfortunately, prior work established that this problem is NP-hard for decision tree ensembles, hence bound to be intractable for specific inputs. In this paper, we identify a restricted class of decision tree ensembles, called large-spread ensembles, which admit a security verification algorithm running in polynomial time. We then propose a new approach called verifiable learning, which advocates the training of such restricted model classes which are amenable for efficient verification. We show the benefits of this idea by designing a new training algorithm that automatically learns a large-spread decision tree ensemble from labelled data, thus enabling its security verification in polynomial time. Experimental results on public datasets confirm that large-spread ensembles trained using our algorithm can be verified in a matter of seconds, using standard commercial hardware. Moreover, large-spread ensembles are more robust than traditional ensembles against evasion attacks, at the cost of an acceptable loss of accuracy in the non-adversarial setting.

摘要: 验证机器学习模型在测试时对逃避攻击的稳健性是一个重要的研究问题。不幸的是，以前的工作确定了这个问题对于决策树集成来说是NP-Hard的，因此对于特定的输入必然是棘手的。在本文中，我们识别了一类受限的决策树集成，称为大分布集成，它允许安全验证算法在多项式时间内运行。然后，我们提出了一种新的方法，称为可验证学习，它主张训练这样的受限模型类，这些模型类适合于有效的验证。我们通过设计一种新的训练算法，从标记数据中自动学习大规模决策树集成，从而在多项式时间内实现其安全性验证，从而展示了这种思想的好处。在公共数据集上的实验结果证实，使用我们的算法训练的大范围集成可以在几秒钟内使用标准的商业硬件进行验证。此外，大范围的合奏比传统的合奏更能抵抗躲避攻击，代价是在非对抗性环境中损失可接受的准确性。



## **38. FIVA: Facial Image and Video Anonymization and Anonymization Defense**

FIVA：人脸图像和视频的匿名化及匿名化防御 cs.CV

Accepted to ICCVW 2023 - DFAD 2023

**SubmitDate**: 2023-09-08    [abs](http://arxiv.org/abs/2309.04228v1) [paper-pdf](http://arxiv.org/pdf/2309.04228v1)

**Authors**: Felix Rosberg, Eren Erdal Aksoy, Cristofer Englund, Fernando Alonso-Fernandez

**Abstract**: In this paper, we present a new approach for facial anonymization in images and videos, abbreviated as FIVA. Our proposed method is able to maintain the same face anonymization consistently over frames with our suggested identity-tracking and guarantees a strong difference from the original face. FIVA allows for 0 true positives for a false acceptance rate of 0.001. Our work considers the important security issue of reconstruction attacks and investigates adversarial noise, uniform noise, and parameter noise to disrupt reconstruction attacks. In this regard, we apply different defense and protection methods against these privacy threats to demonstrate the scalability of FIVA. On top of this, we also show that reconstruction attack models can be used for detection of deep fakes. Last but not least, we provide experimental results showing how FIVA can even enable face swapping, which is purely trained on a single target image.

摘要: 在本文中，我们提出了一种新的图像和视频中的人脸匿名方法，简称FIVA。我们提出的方法能够在使用我们建议的身份跟踪的帧中一致地保持相同的人脸匿名化，并保证与原始人脸有很大的不同。FIVA允许0个真阳性，错误接受率为0.001。我们的工作考虑了重构攻击的重要安全问题，并研究了对抗噪声、均匀噪声和参数噪声对重构攻击的干扰。在这方面，我们针对这些隐私威胁采用了不同的防御和保护方法，以展示FIVA的可扩展性。此外，我们还证明了重构攻击模型可以用于深度伪装的检测。最后但并非最不重要的是，我们提供了实验结果，展示了FIVA甚至可以实现人脸交换，这是纯粹针对单个目标图像进行训练的。



## **39. Counterfactual Explanations via Locally-guided Sequential Algorithmic Recourse**

基于局部引导的序贯算法资源的反事实解释 cs.LG

7 pages, 5 figures, 3 appendix pages

**SubmitDate**: 2023-09-08    [abs](http://arxiv.org/abs/2309.04211v1) [paper-pdf](http://arxiv.org/pdf/2309.04211v1)

**Authors**: Edward A. Small, Jeffrey N. Clark, Christopher J. McWilliams, Kacper Sokol, Jeffrey Chan, Flora D. Salim, Raul Santos-Rodriguez

**Abstract**: Counterfactuals operationalised through algorithmic recourse have become a powerful tool to make artificial intelligence systems explainable. Conceptually, given an individual classified as y -- the factual -- we seek actions such that their prediction becomes the desired class y' -- the counterfactual. This process offers algorithmic recourse that is (1) easy to customise and interpret, and (2) directly aligned with the goals of each individual. However, the properties of a "good" counterfactual are still largely debated; it remains an open challenge to effectively locate a counterfactual along with its corresponding recourse. Some strategies use gradient-driven methods, but these offer no guarantees on the feasibility of the recourse and are open to adversarial attacks on carefully created manifolds. This can lead to unfairness and lack of robustness. Other methods are data-driven, which mostly addresses the feasibility problem at the expense of privacy, security and secrecy as they require access to the entire training data set. Here, we introduce LocalFACE, a model-agnostic technique that composes feasible and actionable counterfactual explanations using locally-acquired information at each step of the algorithmic recourse. Our explainer preserves the privacy of users by only leveraging data that it specifically requires to construct actionable algorithmic recourse, and protects the model by offering transparency solely in the regions deemed necessary for the intervention.

摘要: 通过算法资源操作的反事实已成为使人工智能系统变得可解释的强大工具。在概念上，给出一个被归类为y的个体--事实--我们寻求行动，使他们的预测成为所需的类别y‘--反事实。这一过程提供了(1)易于定制和解释的算法资源，(2)直接与每个人的目标保持一致。然而，“好的”反事实的性质仍然在很大程度上存在争议；有效地确定反事实及其相应的追索权仍然是一项悬而未决的挑战。一些策略使用梯度驱动的方法，但这些方法不能保证这种方法的可行性，而且容易受到精心创建的流形的对抗性攻击。这可能会导致不公平和缺乏稳健性。其他方法是以数据为导向的，主要是以牺牲隐私、安全和保密性为代价来解决可行性问题，因为它们需要访问整个训练数据集。在这里，我们介绍LocalFACE，这是一种与模型无关的技术，它在算法资源的每个步骤使用本地获取的信息组成可行和可操作的反事实解释。我们的解释器通过只利用它特别需要的数据来保护用户的隐私，以构建可操作的算法资源，并通过仅在被认为对干预必要的区域提供透明度来保护模型。



## **40. Adversarial attacks on hybrid classical-quantum Deep Learning models for Histopathological Cancer Detection**

用于组织病理学癌症检测的经典-量子混合深度学习模型的对抗性攻击 quant-ph

7 pages, 8 figures, 2 Tables

**SubmitDate**: 2023-09-08    [abs](http://arxiv.org/abs/2309.06377v1) [paper-pdf](http://arxiv.org/pdf/2309.06377v1)

**Authors**: Biswaraj Baral, Reek Majumdar, Bhavika Bhalgamiya, Taposh Dutta Roy

**Abstract**: We present an effective application of quantum machine learning in histopathological cancer detection. The study here emphasizes two primary applications of hybrid classical-quantum Deep Learning models. The first application is to build a classification model for histopathological cancer detection using the quantum transfer learning strategy. The second application is to test the performance of this model for various adversarial attacks. Rather than using a single transfer learning model, the hybrid classical-quantum models are tested using multiple transfer learning models, especially ResNet18, VGG-16, Inception-v3, and AlexNet as feature extractors and integrate it with several quantum circuit-based variational quantum circuits (VQC) with high expressibility. As a result, we provide a comparative analysis of classical models and hybrid classical-quantum transfer learning models for histopathological cancer detection under several adversarial attacks. We compared the performance accuracy of the classical model with the hybrid classical-quantum model using pennylane default quantum simulator. We also observed that for histopathological cancer detection under several adversarial attacks, Hybrid Classical-Quantum (HCQ) models provided better accuracy than classical image classification models.

摘要: 我们提出了一种有效的量子机器学习在组织病理学癌症检测中的应用。这里的研究强调了经典-量子混合深度学习模型的两个主要应用。第一个应用是使用量子转移学习策略来构建组织病理学癌症检测的分类模型。第二个应用是测试该模型在各种对抗性攻击下的性能。与单一的转移学习模型不同，经典-量子混合模型使用多个转移学习模型进行测试，特别是以ResNet18、VGG-16、Inception-v3和AlexNet为特征提取者，并将其与多个基于量子电路的高可表达的变分量子电路(VQC)相集成。因此，我们提供了几种对抗性攻击下用于组织病理学癌症检测的经典模型和经典-量子转移混合学习模型的比较分析。利用Pennylane缺省量子模拟器对经典模型和经典-量子混合模型的性能精度进行了比较。我们还观察到，对于几种对抗性攻击下的组织病理学癌症检测，混合经典-量子(HCQ)模型比经典图像分类模型提供了更好的准确性。



## **41. Blades: A Unified Benchmark Suite for Byzantine Attacks and Defenses in Federated Learning**

刀片：联邦学习中拜占庭攻击和防御的统一基准套件 cs.CR

**SubmitDate**: 2023-09-07    [abs](http://arxiv.org/abs/2206.05359v2) [paper-pdf](http://arxiv.org/pdf/2206.05359v2)

**Authors**: Shenghui Li, Edith Ngai, Fanghua Ye, Li Ju, Tianru Zhang, Thiemo Voigt

**Abstract**: Federated learning (FL) facilitates distributed training across clients, safeguarding the privacy of their data. The inherent distributed structure of FL introduces vulnerabilities, especially from adversarial (Byzantine) clients aiming to skew local updates to their advantage. Despite the plethora of research focusing on Byzantine-resilient FL, the academic community has yet to establish a comprehensive benchmark suite, pivotal for impartial assessment and comparison of different techniques.   This paper investigates existing techniques in Byzantine-resilient FL and introduces an open-source benchmark suite for convenient and fair performance comparisons. Our investigation begins with a systematic study of Byzantine attack and defense strategies. Subsequently, we present \ours, a scalable, extensible, and easily configurable benchmark suite that supports researchers and developers in efficiently implementing and validating novel strategies against baseline algorithms in Byzantine-resilient FL. The design of \ours incorporates key characteristics derived from our systematic study, encompassing the attacker's capabilities and knowledge, defense strategy categories, and factors influencing robustness. Blades contains built-in implementations of representative attack and defense strategies and offers user-friendly interfaces for seamlessly integrating new ideas.

摘要: 联合学习(FL)促进了跨客户的分布式培训，保护了他们的数据隐私。FL固有的分布式结构引入了漏洞，特别是来自敌意(拜占庭)客户端的漏洞，旨在歪曲本地更新以利于其优势。尽管有太多关于拜占庭式外语的研究，但学术界还没有建立一个全面的基准套件，这是公正评估和比较不同技术的关键。本文研究了拜占庭弹性FL中的现有技术，并介绍了一个开源的基准测试套件，用于方便和公平地进行性能比较。我们的调查始于对拜占庭攻防战略的系统研究。随后，我们提出了一个可伸缩、可扩展且易于配置的基准测试套件，它支持研究人员和开发人员在拜占庭弹性FL中有效地实现和验证针对基线算法的新策略。我们的设计包含了来自我们系统研究的关键特征，包括攻击者的能力和知识、防御策略类别和影响健壮性的因素。Blade包含典型攻击和防御策略的内置实现，并提供用户友好的界面，以无缝集成新想法。



## **42. Node Injection for Class-specific Network Poisoning**

针对特定类别的网络中毒的节点注入 cs.LG

28 pages, 5 figures

**SubmitDate**: 2023-09-07    [abs](http://arxiv.org/abs/2301.12277v2) [paper-pdf](http://arxiv.org/pdf/2301.12277v2)

**Authors**: Ansh Kumar Sharma, Rahul Kukreja, Mayank Kharbanda, Tanmoy Chakraborty

**Abstract**: Graph Neural Networks (GNNs) are powerful in learning rich network representations that aid the performance of downstream tasks. However, recent studies showed that GNNs are vulnerable to adversarial attacks involving node injection and network perturbation. Among these, node injection attacks are more practical as they don't require manipulation in the existing network and can be performed more realistically. In this paper, we propose a novel problem statement - a class-specific poison attack on graphs in which the attacker aims to misclassify specific nodes in the target class into a different class using node injection. Additionally, nodes are injected in such a way that they camouflage as benign nodes. We propose NICKI, a novel attacking strategy that utilizes an optimization-based approach to sabotage the performance of GNN-based node classifiers. NICKI works in two phases - it first learns the node representation and then generates the features and edges of the injected nodes. Extensive experiments and ablation studies on four benchmark networks show that NICKI is consistently better than four baseline attacking strategies for misclassifying nodes in the target class. We also show that the injected nodes are properly camouflaged as benign, thus making the poisoned graph indistinguishable from its clean version w.r.t various topological properties.

摘要: 图形神经网络(GNN)在学习丰富的网络表示方面功能强大，有助于下游任务的执行。然而，最近的研究表明，GNN很容易受到包括节点注入和网络扰动在内的对抗性攻击。其中，节点注入攻击更实用，因为它们不需要在现有网络中进行操作，并且可以更真实地执行。在本文中，我们提出了一种新的问题陈述--针对图的特定类的毒物攻击，攻击者的目的是通过节点注入将目标类中的特定节点错误地分类到不同的类中。此外，注入结节的方式是伪装成良性结节。我们提出了一种新的攻击策略Nicki，它利用基于优化的方法来破坏基于GNN的节点分类器的性能。Nicki分两个阶段工作--它首先学习节点表示，然后生成注入节点的特征和边。在四个基准网络上的大量实验和烧蚀研究表明，对于目标类中的节点误分类，Nicki一致优于四种基线攻击策略。我们还证明了注入的节点被适当伪装成良性的，从而使得中毒的图与其干净的版本无法区分各种拓扑性质。



## **43. Experimental Study of Adversarial Attacks on ML-based xApps in O-RAN**

O-Range环境下基于ML的xApp对抗性攻击实验研究 cs.NI

Accepted for Globecom 2023

**SubmitDate**: 2023-09-07    [abs](http://arxiv.org/abs/2309.03844v1) [paper-pdf](http://arxiv.org/pdf/2309.03844v1)

**Authors**: Naveen Naik Sapavath, Brian Kim, Kaushik Chowdhury, Vijay K Shah

**Abstract**: Open Radio Access Network (O-RAN) is considered as a major step in the evolution of next-generation cellular networks given its support for open interfaces and utilization of artificial intelligence (AI) into the deployment, operation, and maintenance of RAN. However, due to the openness of the O-RAN architecture, such AI models are inherently vulnerable to various adversarial machine learning (ML) attacks, i.e., adversarial attacks which correspond to slight manipulation of the input to the ML model. In this work, we showcase the vulnerability of an example ML model used in O-RAN, and experimentally deploy it in the near-real time (near-RT) RAN intelligent controller (RIC). Our ML-based interference classifier xApp (extensible application in near-RT RIC) tries to classify the type of interference to mitigate the interference effect on the O-RAN system. We demonstrate the first-ever scenario of how such an xApp can be impacted through an adversarial attack by manipulating the data stored in a shared database inside the near-RT RIC. Through a rigorous performance analysis deployed on a laboratory O-RAN testbed, we evaluate the performance in terms of capacity and the prediction accuracy of the interference classifier xApp using both clean and perturbed data. We show that even small adversarial attacks can significantly decrease the accuracy of ML application in near-RT RIC, which can directly impact the performance of the entire O-RAN deployment.

摘要: 开放式无线接入网(O-RAN)被认为是下一代蜂窝网络演进的重要一步，因为它支持开放接口，并利用人工智能(AI)来部署、运营和维护RAN。然而，由于O-RAN体系结构的开放性，这种人工智能模型天生就容易受到各种对抗性机器学习(ML)攻击，即对应于对ML模型的输入进行轻微操纵的对抗性攻击。在这项工作中，我们展示了一个用于O-RAN的示例ML模型的漏洞，并将其实验部署在近实时(Near-RT)RAN智能控制器(RIC)中。我们的基于ML的干扰分类器xApp(eXtensible Application in Near-RT RIC)试图对干扰类型进行分类，以减轻干扰对O-RAN系统的影响。我们通过操作存储在近RTRIC内的共享数据库中的数据，首次演示了这样的xApp如何通过敌意攻击受到影响的场景。通过在实验室O-RAN试验台上部署的严格性能分析，我们使用干净和扰动数据评估了干扰分类器xApp的容量和预测精度方面的性能。结果表明，即使是较小的敌意攻击也会显著降低ML在近RTRIC中应用的准确性，这将直接影响整个O-RAN部署的性能。



## **44. Adversarially Robust Deep Learning with Optimal-Transport-Regularized Divergences**

基于最优传输正则化发散的对抗性稳健深度学习 cs.LG

30 pages, 5 figures

**SubmitDate**: 2023-09-07    [abs](http://arxiv.org/abs/2309.03791v1) [paper-pdf](http://arxiv.org/pdf/2309.03791v1)

**Authors**: Jeremiah Birrell, Mohammadreza Ebrahimi

**Abstract**: We introduce the $ARMOR_D$ methods as novel approaches to enhancing the adversarial robustness of deep learning models. These methods are based on a new class of optimal-transport-regularized divergences, constructed via an infimal convolution between an information divergence and an optimal-transport (OT) cost. We use these as tools to enhance adversarial robustness by maximizing the expected loss over a neighborhood of distributions, a technique known as distributionally robust optimization. Viewed as a tool for constructing adversarial samples, our method allows samples to be both transported, according to the OT cost, and re-weighted, according to the information divergence. We demonstrate the effectiveness of our method on malware detection and image recognition applications and find that, to our knowledge, it outperforms existing methods at enhancing the robustness against adversarial attacks. $ARMOR_D$ yields the robustified accuracy of $98.29\%$ against $FGSM$ and $98.18\%$ against $PGD^{40}$ on the MNIST dataset, reducing the error rate by more than $19.7\%$ and $37.2\%$ respectively compared to prior methods. Similarly, in malware detection, a discrete (binary) data domain, $ARMOR_D$ improves the robustified accuracy under $rFGSM^{50}$ attack compared to the previous best-performing adversarial training methods by $37.0\%$ while lowering false negative and false positive rates by $51.1\%$ and $57.53\%$, respectively.

摘要: 我们引入了$ARMOR_D$方法作为增强深度学习模型对抗性稳健性的新方法。这些方法是基于一类新的最优传输正则化发散，通过信息发散和最优传输(OT)成本之间的逐次卷积构造的。我们使用它们作为工具，通过最大化分布邻域的预期损失来增强对手的稳健性，这是一种称为分布稳健优化的技术。作为一种构建对抗性样本的工具，我们的方法允许样本既可以根据OT成本进行传输，又可以根据信息分歧进行重新加权。我们在恶意软件检测和图像识别应用中验证了该方法的有效性，并发现，据我们所知，该方法在增强对对手攻击的稳健性方面优于现有方法。在MNIST数据集上，$ARMOR_D$对$FGSM$和$PGD^{40}$的粗化精度分别为98.29美元和98.18美元，与以前的方法相比，错误率分别降低了19.7美元和37.2美元以上。类似地，在恶意软件检测方面，离散(二进制)数据域$ARMOR_D$在$rFGSM^{50}$攻击下，与以前性能最好的对抗性训练方法相比，提高了$rFGSM^{50}$攻击的粗暴准确率$37.0$，同时降低了漏检率和误警率分别为$51.1和$57.53$。



## **45. DiffDefense: Defending against Adversarial Attacks via Diffusion Models**

扩散防御：通过扩散模型防御敌意攻击 cs.LG

Paper published at ICIAP23

**SubmitDate**: 2023-09-07    [abs](http://arxiv.org/abs/2309.03702v1) [paper-pdf](http://arxiv.org/pdf/2309.03702v1)

**Authors**: Hondamunige Prasanna Silva, Lorenzo Seidenari, Alberto Del Bimbo

**Abstract**: This paper presents a novel reconstruction method that leverages Diffusion Models to protect machine learning classifiers against adversarial attacks, all without requiring any modifications to the classifiers themselves. The susceptibility of machine learning models to minor input perturbations renders them vulnerable to adversarial attacks. While diffusion-based methods are typically disregarded for adversarial defense due to their slow reverse process, this paper demonstrates that our proposed method offers robustness against adversarial threats while preserving clean accuracy, speed, and plug-and-play compatibility. Code at: https://github.com/HondamunigePrasannaSilva/DiffDefence.

摘要: 本文提出了一种新的重建方法，该方法利用扩散模型来保护机器学习分类器免受对手攻击，而不需要对分类器本身进行任何修改。机器学习模型对微小输入扰动的敏感性使其容易受到敌意攻击。虽然基于扩散的方法由于其缓慢的反向过程而通常被忽略用于对抗性防御，但本文证明了我们提出的方法在保持干净的准确性、速度和即插即用兼容性的同时，对对抗性威胁提供了健壮性。代码：https://github.com/HondamunigePrasannaSilva/DiffDefence.



## **46. Impact Sensitivity Analysis of Cooperative Adaptive Cruise Control Against Resource-Limited Adversaries**

协同自适应巡航控制对抗资源受限对手的撞击敏感性分析 eess.SY

**SubmitDate**: 2023-09-07    [abs](http://arxiv.org/abs/2304.02395v2) [paper-pdf](http://arxiv.org/pdf/2304.02395v2)

**Authors**: Mischa Huisman, Carlos Murguia, Erjen Lefeber, Nathan van de Wouw

**Abstract**: Cooperative Adaptive Cruise Control (CACC) is a technology that allows groups of vehicles to form in automated, tightly-coupled platoons. CACC schemes exploit Vehicle-to-Vehicle (V2V) wireless communications to exchange information between vehicles. However, the use of communication networks brings security concerns as it exposes network access points that the adversary can exploit to disrupt the vehicles' operation and even cause crashes. In this manuscript, we present a sensitivity analysis of CACC schemes against a class of resource-limited attacks. We present a modelling framework that allows us to systematically compute outer ellipsoidal approximations of reachable sets induced by attacks. We use the size of these sets as a security metric to quantify the potential damage of attacks affecting different signals in a CACC-controlled vehicle and study how two key system parameters change this metric. We carry out a sensitivity analysis for two different controller implementations (as given the available sensors there is an infinite number of realizations of the same controller) and show how different controller realizations can significantly affect the impact of attacks. We present extensive simulation experiments to illustrate the results.

摘要: 合作自适应巡航控制(CACC)是一种允许车辆组成自动、紧密耦合排的技术。CACC方案利用车辆对车辆(V2V)无线通信来在车辆之间交换信息。然而，通信网络的使用带来了安全问题，因为它暴露了网络接入点，对手可以利用这些接入点来扰乱车辆的操作，甚至导致撞车。在本文中，我们给出了CACC方案对一类资源受限攻击的敏感度分析。我们提出了一个模型框架，它允许我们系统地计算由攻击诱导的可达集的外椭球近似。我们使用这些集合的大小作为安全度量，以量化影响CACC控制的车辆中不同信号的攻击的潜在损害，并研究两个关键系统参数如何改变该度量。我们对两种不同的控制器实现进行了敏感度分析(在给定可用传感器的情况下，同一控制器有无限多个实现)，并展示了不同的控制器实现如何显著影响攻击的影响。我们给出了大量的模拟实验来说明结果。



## **47. How adversarial attacks can disrupt seemingly stable accurate classifiers**

敌意攻击如何扰乱看似稳定的准确分类器 cs.LG

11 pages, 8 figures, additional supplementary materials

**SubmitDate**: 2023-09-07    [abs](http://arxiv.org/abs/2309.03665v1) [paper-pdf](http://arxiv.org/pdf/2309.03665v1)

**Authors**: Oliver J. Sutton, Qinghua Zhou, Ivan Y. Tyukin, Alexander N. Gorban, Alexander Bastounis, Desmond J. Higham

**Abstract**: Adversarial attacks dramatically change the output of an otherwise accurate learning system using a seemingly inconsequential modification to a piece of input data. Paradoxically, empirical evidence indicates that even systems which are robust to large random perturbations of the input data remain susceptible to small, easily constructed, adversarial perturbations of their inputs. Here, we show that this may be seen as a fundamental feature of classifiers working with high dimensional input data. We introduce a simple generic and generalisable framework for which key behaviours observed in practical systems arise with high probability -- notably the simultaneous susceptibility of the (otherwise accurate) model to easily constructed adversarial attacks, and robustness to random perturbations of the input data. We confirm that the same phenomena are directly observed in practical neural networks trained on standard image classification problems, where even large additive random noise fails to trigger the adversarial instability of the network. A surprising takeaway is that even small margins separating a classifier's decision surface from training and testing data can hide adversarial susceptibility from being detected using randomly sampled perturbations. Counterintuitively, using additive noise during training or testing is therefore inefficient for eradicating or detecting adversarial examples, and more demanding adversarial training is required.

摘要: 对抗性攻击通过对一段输入数据进行看似无关紧要的修改，极大地改变了原本准确的学习系统的输出。矛盾的是，经验证据表明，即使是对输入数据的大随机扰动具有健壮性的系统，也仍然容易受到其输入的小的、容易构造的、对抗性的扰动。在这里，我们展示了这可以被视为使用高维输入数据的分类器的基本特征。我们引入了一个简单的通用和可推广的框架，对于该框架，在实际系统中观察到的关键行为以高概率出现-特别是(否则准确的)模型对容易构造的对抗性攻击的同时敏感性，以及对输入数据的随机扰动的稳健性。我们证实，在标准图像分类问题上训练的实际神经网络中也直接观察到了同样的现象，其中即使是较大的加性随机噪声也不能触发网络的对抗性不稳定性。令人惊讶的是，即使是将分类器的决策面与训练和测试数据分开的很小的边距，也可以隐藏对手的易感性，使其不会被随机抽样的扰动检测到。因此，与直觉相反的是，在训练或测试期间使用加性噪声对于消除或检测对抗性例子是低效的，并且需要更苛刻的对抗性训练。



## **48. Unlearnable Examples Give a False Sense of Security: Piercing through Unexploitable Data with Learnable Examples**

无法学习的例子给人一种错误的安全感：用可学习的例子穿透不可利用的数据 cs.LG

Accepted in MM 2023

**SubmitDate**: 2023-09-07    [abs](http://arxiv.org/abs/2305.09241v4) [paper-pdf](http://arxiv.org/pdf/2305.09241v4)

**Authors**: Wan Jiang, Yunfeng Diao, He Wang, Jianxin Sun, Meng Wang, Richang Hong

**Abstract**: Safeguarding data from unauthorized exploitation is vital for privacy and security, especially in recent rampant research in security breach such as adversarial/membership attacks. To this end, \textit{unlearnable examples} (UEs) have been recently proposed as a compelling protection, by adding imperceptible perturbation to data so that models trained on them cannot classify them accurately on original clean distribution. Unfortunately, we find UEs provide a false sense of security, because they cannot stop unauthorized users from utilizing other unprotected data to remove the protection, by turning unlearnable data into learnable again. Motivated by this observation, we formally define a new threat by introducing \textit{learnable unauthorized examples} (LEs) which are UEs with their protection removed. The core of this approach is a novel purification process that projects UEs onto the manifold of LEs. This is realized by a new joint-conditional diffusion model which denoises UEs conditioned on the pixel and perceptual similarity between UEs and LEs. Extensive experiments demonstrate that LE delivers state-of-the-art countering performance against both supervised UEs and unsupervised UEs in various scenarios, which is the first generalizable countermeasure to UEs across supervised learning and unsupervised learning. Our code is available at \url{https://github.com/jiangw-0/LE_JCDP}.

摘要: 保护数据不被未经授权的利用对隐私和安全至关重要，特别是在最近对安全漏洞的猖獗研究中，例如对抗性/成员攻击。为此，最近提出了不可学习的例子(UE)作为一种强制保护，通过向数据添加不可察觉的扰动，使得训练在这些数据上的模型不能根据原始的干净分布对它们进行准确的分类。不幸的是，我们发现UE提供了一种错误的安全感，因为它们无法阻止未经授权的用户利用其他不受保护的数据来取消保护，方法是将无法学习的数据再次变为可学习的数据。受此观察的启发，我们正式定义了一种新的威胁，引入了去除了保护的可学习未经授权示例(LES)。这种方法的核心是一种新颖的净化过程，将UE投射到LES的流形上。这是通过一种新的联合条件扩散模型来实现的，该模型根据UE和LES之间的像素和感知相似性来对UE进行去噪。大量的实验表明，在不同的场景下，LE对监督UE和非监督UE都提供了最先进的对抗性能，这是针对监督学习和非监督学习的UE的第一个可推广的对策。我们的代码可在\url{https://github.com/jiangw-0/LE_JCDP}.



## **49. Password-Stealing without Hacking: Wi-Fi Enabled Practical Keystroke Eavesdropping**

无需黑客攻击即可窃取密码：支持Wi-Fi的实用击键窃听 cs.CR

14 pages, 27 figures, in ACM Conference on Computer and  Communications Security 2023

**SubmitDate**: 2023-09-07    [abs](http://arxiv.org/abs/2309.03492v1) [paper-pdf](http://arxiv.org/pdf/2309.03492v1)

**Authors**: Jingyang Hu, Hongbo Wang, Tianyue Zheng, Jingzhi Hu, Zhe Chen, Hongbo Jiang, Jun Luo

**Abstract**: The contact-free sensing nature of Wi-Fi has been leveraged to achieve privacy breaches, yet existing attacks relying on Wi-Fi CSI (channel state information) demand hacking Wi-Fi hardware to obtain desired CSIs. Since such hacking has proven prohibitively hard due to compact hardware, its feasibility in keeping up with fast-developing Wi-Fi technology becomes very questionable. To this end, we propose WiKI-Eve to eavesdrop keystrokes on smartphones without the need for hacking. WiKI-Eve exploits a new feature, BFI (beamforming feedback information), offered by latest Wi-Fi hardware: since BFI is transmitted from a smartphone to an AP in clear-text, it can be overheard (hence eavesdropped) by any other Wi-Fi devices switching to monitor mode. As existing keystroke inference methods offer very limited generalizability, WiKI-Eve further innovates in an adversarial learning scheme to enable its inference generalizable towards unseen scenarios. We implement WiKI-Eve and conduct extensive evaluation on it; the results demonstrate that WiKI-Eve achieves 88.9% inference accuracy for individual keystrokes and up to 65.8% top-10 accuracy for stealing passwords of mobile applications (e.g., WeChat).

摘要: Wi-Fi的免接触传感特性已被用来实现隐私泄露，但依赖Wi-Fi CSI(通道状态信息)的现有攻击需要黑客攻击Wi-Fi硬件以获得所需的CSI。由于硬件紧凑，此类黑客攻击已被证明非常困难，因此其跟上快速发展的Wi-Fi技术的可行性变得非常值得怀疑。为此，我们建议Wiki-Eve在不需要黑客的情况下窃听智能手机上的按键。Wiki-Eve利用了最新Wi-Fi硬件提供的新功能BFI(波束形成反馈信息)：由于BFI以明文形式从智能手机传输到AP，因此它可以被切换到监控模式的任何其他Wi-Fi设备窃听(因此被窃听)。由于现有的击键推理方法提供的泛化能力非常有限，Wiki-Eve进一步创新了对抗性学习方案，使其推理能够针对未知场景进行泛化。我们实现了Wiki-Eve并对其进行了广泛的评估，结果表明，Wiki-Eve对单个击键的推理准确率为88.9%，对于移动应用(如微信)的密码窃取前10名的准确率高达65.8%。



## **50. Cyber Recovery from Dynamic Load Altering Attacks: Linking Electricity, Transportation, and Cyber Networks**

动态负载变化攻击的网络恢复：连接电力、交通和网络 eess.SY

**SubmitDate**: 2023-09-06    [abs](http://arxiv.org/abs/2309.03380v1) [paper-pdf](http://arxiv.org/pdf/2309.03380v1)

**Authors**: Mengxiang Liu, Zhongda Chu, Fei Teng

**Abstract**: To address the increasing vulnerability of power grids, significant attention has been focused on the attack detection and impact mitigation. However, it is still unclear how to effectively and quickly recover the cyber and physical networks from a cyberattack. In this context, this paper presents the first investigation of the Cyber Recovery from Dynamic load altering Attack (CRDA). Considering the interconnection among electricity, transportation, and cyber networks, two essential sub-tasks are formulated for the CRDA: i) Optimal design of repair crew routes to remove installed malware and ii) Adaptive adjustment of system operation to eliminate the mitigation costs while guaranteeing stability. To achieve this, linear stability constraints are obtained by estimating the related eigenvalues under the variation of multiple IBR droop gains based on the sensitivity information of strategically selected sampling points. Moreover, to obtain the robust recovery strategy, the potential counter-measures from the adversary during the recovery process are modeled as maximizing the attack impact of remaining compromised resources in each step. A Mixed-Integer Linear Programming (MILP) problem can be finally formulated for the CRDA with the primary objective to reset involved droop gains and secondarily to repair all compromised loads. Case studies are performed in the modified IEEE 39-bus power system to illustrate the effectiveness of the proposed CRDA compared to the benchmark case.

摘要: 为了解决电网日益增长的脆弱性，攻击检测和影响缓解已经引起了极大的关注。然而，目前仍不清楚如何有效、快速地从网络攻击中恢复网络和物理网络。在此背景下，本文首次对动态负载变化攻击的网络恢复进行了研究。考虑到电力、交通和网络之间的互联，CRDA制定了两个必要的子任务：i)优化设计维修人员路线以删除安装的恶意软件；ii)自适应调整系统运行，在保证稳定性的同时消除缓解成本。为了实现这一点，基于策略选择的采样点的灵敏度信息，通过估计多个IBR下垂增益变化下的相关特征值来获得线性稳定性约束。此外，为了获得稳健的恢复策略，在恢复过程中来自对手的潜在对抗措施被建模为最大化每一步中剩余受损资源的攻击影响。对于CRDA，最终可以建立一个混合整数线性规划(MILP)问题，主要目标是重置涉及的下垂增益，其次是修复所有受损的负载。以IEEE 39节点电力系统为例进行了算例分析，并与基准算例进行了比较，验证了该方法的有效性。



