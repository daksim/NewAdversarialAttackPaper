# Latest Adversarial Attack Papers
**update at 2023-12-28 14:09:47**

翻译来自 https://cloud.tencent.com/document/product/551/15619

## **1. SlowTrack: Increasing the Latency of Camera-based Perception in Autonomous Driving Using Adversarial Examples**

SlowTrack：使用对抗性例子增加自动驾驶中基于摄像头的感知的延迟 cs.CV

Accepted by AAAI 2024

**SubmitDate**: 2023-12-26    [abs](http://arxiv.org/abs/2312.09520v2) [paper-pdf](http://arxiv.org/pdf/2312.09520v2)

**Authors**: Chen Ma, Ningfei Wang, Qi Alfred Chen, Chao Shen

**Abstract**: In Autonomous Driving (AD), real-time perception is a critical component responsible for detecting surrounding objects to ensure safe driving. While researchers have extensively explored the integrity of AD perception due to its safety and security implications, the aspect of availability (real-time performance) or latency has received limited attention. Existing works on latency-based attack have focused mainly on object detection, i.e., a component in camera-based AD perception, overlooking the entire camera-based AD perception, which hinders them to achieve effective system-level effects, such as vehicle crashes. In this paper, we propose SlowTrack, a novel framework for generating adversarial attacks to increase the execution time of camera-based AD perception. We propose a novel two-stage attack strategy along with the three new loss function designs. Our evaluation is conducted on four popular camera-based AD perception pipelines, and the results demonstrate that SlowTrack significantly outperforms existing latency-based attacks while maintaining comparable imperceptibility levels. Furthermore, we perform the evaluation on Baidu Apollo, an industry-grade full-stack AD system, and LGSVL, a production-grade AD simulator, with two scenarios to compare the system-level effects of SlowTrack and existing attacks. Our evaluation results show that the system-level effects can be significantly improved, i.e., the vehicle crash rate of SlowTrack is around 95% on average while existing works only have around 30%.

摘要: 在自动驾驶中，实时感知是负责检测周围物体以确保安全驾驶的关键部件。虽然由于AD感知的安全性和安全性，研究人员已经对其完整性进行了广泛的探索，但可用性(实时性能)或延迟方面的关注有限。现有的基于延迟攻击的研究主要集中于目标检测，即基于摄像头的广告感知中的一个组件，而忽略了整个基于摄像头的广告感知，这阻碍了它们达到有效的系统级效果，如车辆碰撞。在本文中，我们提出了一种新的生成敌意攻击的框架SlowTrack，以增加基于摄像机的广告感知的执行时间。我们提出了一种新的两阶段攻击策略以及三种新的损失函数设计。我们在四个流行的基于摄像头的AD感知管道上进行了评估，结果表明，SlowTrack在保持相当的不可感知性水平的同时，显著优于现有的基于延迟的攻击。此外，我们在工业级全栈AD系统百度Apollo和生产级AD模拟器LGSVL上进行了评估，并通过两个场景比较了SlowTrack和现有攻击的系统级影响。我们的评估结果表明，系统级效果可以得到显著提高，即SlowTrack的车辆撞击率平均在95%左右，而现有的工作只有30%左右。



## **2. Model Stealing Attack against Recommender System**

针对推荐系统的模型窃取攻击 cs.CR

**SubmitDate**: 2023-12-26    [abs](http://arxiv.org/abs/2312.11571v2) [paper-pdf](http://arxiv.org/pdf/2312.11571v2)

**Authors**: Zhihao Zhu, Rui Fan, Chenwang Wu, Yi Yang, Defu Lian, Enhong Chen

**Abstract**: Recent studies have demonstrated the vulnerability of recommender systems to data privacy attacks. However, research on the threat to model privacy in recommender systems, such as model stealing attacks, is still in its infancy. Some adversarial attacks have achieved model stealing attacks against recommender systems, to some extent, by collecting abundant training data of the target model (target data) or making a mass of queries. In this paper, we constrain the volume of available target data and queries and utilize auxiliary data, which shares the item set with the target data, to promote model stealing attacks. Although the target model treats target and auxiliary data differently, their similar behavior patterns allow them to be fused using an attention mechanism to assist attacks. Besides, we design stealing functions to effectively extract the recommendation list obtained by querying the target model. Experimental results show that the proposed methods are applicable to most recommender systems and various scenarios and exhibit excellent attack performance on multiple datasets.

摘要: 最近的研究表明，推荐系统对数据隐私攻击是脆弱的。然而，对推荐系统中模型隐私威胁的研究，如模型窃取攻击，还处于起步阶段。一些对抗性攻击通过收集目标模型(目标数据)的大量训练数据或进行大量查询，在一定程度上实现了对推荐系统的模型窃取攻击。本文通过限制可用目标数据和查询的数据量，利用与目标数据共享项集的辅助数据来促进模型窃取攻击。尽管目标模型对目标和辅助数据的处理不同，但它们相似的行为模式允许使用注意力机制将它们融合在一起，以帮助攻击。此外，我们还设计了窃取函数来有效地提取通过查询目标模型获得的推荐列表。实验结果表明，所提出的方法适用于大多数推荐系统和各种场景，并在多个数据集上表现出良好的攻击性能。



## **3. MENLI: Robust Evaluation Metrics from Natural Language Inference**

MENLI：自然语言推理中的稳健评价指标 cs.CL

TACL 2023 Camera-ready version; updated after proofreading by the  journal

**SubmitDate**: 2023-12-26    [abs](http://arxiv.org/abs/2208.07316v5) [paper-pdf](http://arxiv.org/pdf/2208.07316v5)

**Authors**: Yanran Chen, Steffen Eger

**Abstract**: Recently proposed BERT-based evaluation metrics for text generation perform well on standard benchmarks but are vulnerable to adversarial attacks, e.g., relating to information correctness. We argue that this stems (in part) from the fact that they are models of semantic similarity. In contrast, we develop evaluation metrics based on Natural Language Inference (NLI), which we deem a more appropriate modeling. We design a preference-based adversarial attack framework and show that our NLI based metrics are much more robust to the attacks than the recent BERT-based metrics. On standard benchmarks, our NLI based metrics outperform existing summarization metrics, but perform below SOTA MT metrics. However, when combining existing metrics with our NLI metrics, we obtain both higher adversarial robustness (15%-30%) and higher quality metrics as measured on standard benchmarks (+5% to 30%).

摘要: 最近提出的基于BERT的文本生成评估指标在标准基准测试中表现良好，但容易受到对抗性攻击，例如，关于信息的正确性。我们认为，这源于（部分）的事实，它们是语义相似性的模型。相比之下，我们开发基于自然语言推理（NLI）的评估指标，我们认为这是一个更合适的建模。我们设计了一个基于偏好的对抗性攻击框架，并表明我们基于NLI的指标比最近基于BERT的指标对攻击更鲁棒。在标准基准测试中，我们基于NLI的指标优于现有的摘要指标，但低于SOTA MT指标。然而，当将现有指标与我们的NLI指标相结合时，我们获得了更高的对抗鲁棒性（15%-30%）和更高的质量指标（+5%至30%）。



## **4. Punctuation Matters! Stealthy Backdoor Attack for Language Models**

标点符号很重要！对语言模型的秘密后门攻击 cs.CL

NLPCC 2023

**SubmitDate**: 2023-12-26    [abs](http://arxiv.org/abs/2312.15867v1) [paper-pdf](http://arxiv.org/pdf/2312.15867v1)

**Authors**: Xuan Sheng, Zhicheng Li, Zhaoyang Han, Xiangmao Chang, Piji Li

**Abstract**: Recent studies have pointed out that natural language processing (NLP) models are vulnerable to backdoor attacks. A backdoored model produces normal outputs on the clean samples while performing improperly on the texts with triggers that the adversary injects. However, previous studies on textual backdoor attack pay little attention to stealthiness. Moreover, some attack methods even cause grammatical issues or change the semantic meaning of the original texts. Therefore, they can easily be detected by humans or defense systems. In this paper, we propose a novel stealthy backdoor attack method against textual models, which is called \textbf{PuncAttack}. It leverages combinations of punctuation marks as the trigger and chooses proper locations strategically to replace them. Through extensive experiments, we demonstrate that the proposed method can effectively compromise multiple models in various tasks. Meanwhile, we conduct automatic evaluation and human inspection, which indicate the proposed method possesses good performance of stealthiness without bringing grammatical issues and altering the meaning of sentences.

摘要: 最近的研究指出，自然语言处理(NLP)模型容易受到后门攻击。反向模型在干净的样本上产生正常输出，而在带有对手注入的触发器的文本上执行不正确的操作。然而，以往对文本后门攻击的研究很少关注隐蔽性。此外，一些攻击方法甚至会引起语法问题或改变原文的语义。因此，它们很容易被人类或防御系统检测到。提出了一种新的针对文本模型的隐蔽后门攻击方法-.它利用标点符号的组合作为触发器，并战略性地选择适当的位置来取代它们。通过大量的实验，我们证明了该方法能够在不同的任务中有效地折衷多个模型。同时，我们进行了自动评估和人工检测，表明该方法具有良好的隐蔽性，不会带来语法问题，也不会改变句子的意义。



## **5. Adversarial Item Promotion on Visually-Aware Recommender Systems by Guided Diffusion**

基于引导扩散的视觉感知推荐系统中的对抗性项目提升 cs.IR

**SubmitDate**: 2023-12-25    [abs](http://arxiv.org/abs/2312.15826v1) [paper-pdf](http://arxiv.org/pdf/2312.15826v1)

**Authors**: Lijian Chen, Wei Yuan, Tong Chen, Quoc Viet Hung Nguyen, Lizhen Cui, Hongzhi Yin

**Abstract**: Visually-aware recommender systems have found widespread application in domains where visual elements significantly contribute to the inference of users' potential preferences. While the incorporation of visual information holds the promise of enhancing recommendation accuracy and alleviating the cold-start problem, it is essential to point out that the inclusion of item images may introduce substantial security challenges. Some existing works have shown that the item provider can manipulate item exposure rates to its advantage by constructing adversarial images. However, these works cannot reveal the real vulnerability of visually-aware recommender systems because (1) The generated adversarial images are markedly distorted, rendering them easily detectable by human observers; (2) The effectiveness of the attacks is inconsistent and even ineffective in some scenarios. To shed light on the real vulnerabilities of visually-aware recommender systems when confronted with adversarial images, this paper introduces a novel attack method, IPDGI (Item Promotion by Diffusion Generated Image). Specifically, IPDGI employs a guided diffusion model to generate adversarial samples designed to deceive visually-aware recommender systems. Taking advantage of accurately modeling benign images' distribution by diffusion models, the generated adversarial images have high fidelity with original images, ensuring the stealth of our IPDGI. To demonstrate the effectiveness of our proposed methods, we conduct extensive experiments on two commonly used e-commerce recommendation datasets (Amazon Beauty and Amazon Baby) with several typical visually-aware recommender systems. The experimental results show that our attack method has a significant improvement in both the performance of promoting the long-tailed (i.e., unpopular) items and the quality of generated adversarial images.

摘要: 视觉感知推荐系统在视觉元素对用户潜在偏好的推断有重要作用的领域得到了广泛的应用。虽然加入视觉信息有望提高推荐的准确性和缓解冷启动问题，但必须指出的是，纳入物品图像可能会带来重大的安全挑战。一些已有的研究表明，物品提供者可以通过构建对抗性图像来操纵物品曝光率。然而，这些工作并不能揭示视觉感知推荐系统的真正弱点，因为(1)生成的敌意图像明显失真，使得人类很容易发现它们；(2)攻击的有效性在某些场景下是不一致的，甚至无效的。为了揭示视觉感知推荐系统在面对敌意图像时的真正弱点，提出了一种新的攻击方法--IPDGI(Item Promotion By Diffumation Generated Image)。具体地说，IPDGI使用引导扩散模型来生成敌意样本，旨在欺骗视觉感知的推荐系统。利用扩散模型精确模拟良性图像的分布，生成的对抗性图像与原始图像具有较高的保真度，保证了IPDGI的隐蔽性。为了验证我们提出的方法的有效性，我们在两个常用的电子商务推荐数据集(Amazon Beauty和Amazon Baby)上进行了广泛的实验，并使用几个典型的视觉感知推荐系统进行了实验。实验结果表明，我们的攻击方法在提升长尾(即不受欢迎)项的性能和生成对抗性图像的质量方面都有显著的提高。



## **6. Adversarial Prompt Tuning for Vision-Language Models**

视觉语言模型的对抗性提示调整 cs.CV

**SubmitDate**: 2023-12-25    [abs](http://arxiv.org/abs/2311.11261v2) [paper-pdf](http://arxiv.org/pdf/2311.11261v2)

**Authors**: Jiaming Zhang, Xingjun Ma, Xin Wang, Lingyu Qiu, Jiaqi Wang, Yu-Gang Jiang, Jitao Sang

**Abstract**: With the rapid advancement of multimodal learning, pre-trained Vision-Language Models (VLMs) such as CLIP have demonstrated remarkable capacities in bridging the gap between visual and language modalities. However, these models remain vulnerable to adversarial attacks, particularly in the image modality, presenting considerable security risks. This paper introduces Adversarial Prompt Tuning (AdvPT), a novel technique to enhance the adversarial robustness of image encoders in VLMs. AdvPT innovatively leverages learnable text prompts and aligns them with adversarial image embeddings, to address the vulnerabilities inherent in VLMs without the need for extensive parameter training or modification of the model architecture. We demonstrate that AdvPT improves resistance against white-box and black-box adversarial attacks and exhibits a synergistic effect when combined with existing image-processing-based defense techniques, further boosting defensive capabilities. Comprehensive experimental analyses provide insights into adversarial prompt tuning, a novel paradigm devoted to improving resistance to adversarial images through textual input modifications, paving the way for future robust multimodal learning research. These findings open up new possibilities for enhancing the security of VLMs. Our code is available at https://github.com/jiamingzhang94/Adversarial-Prompt-Tuning.

摘要: 随着多通道学习的快速发展，诸如CLIP等预先训练的视觉语言模型在弥合视觉和语言通道之间的差距方面显示出了显著的能力。然而，这些模型仍然容易受到敌意攻击，特别是在图像模式方面，这带来了相当大的安全风险。本文介绍了对抗性提示调优(AdvPT)技术，这是一种在VLMS中增强图像编码器对抗性稳健性的新技术。AdvPT创新性地利用可学习的文本提示，并将其与对抗性图像嵌入相结合，以解决VLM中固有的漏洞，而无需进行广泛的参数培训或修改模型体系结构。我们证明，AdvPT提高了对白盒和黑盒攻击的抵抗力，并与现有的基于图像处理的防御技术相结合，显示出协同效应，进一步增强了防御能力。全面的实验分析提供了对对抗性即时调整的见解，这是一种致力于通过修改文本输入来提高对对抗性图像的抵抗力的新范式，为未来稳健的多通道学习研究铺平了道路。这些发现为增强VLM的安全性开辟了新的可能性。我们的代码可以在https://github.com/jiamingzhang94/Adversarial-Prompt-Tuning.上找到



## **7. Privacy-Preserving Neural Graph Databases**

保护隐私的神经图库 cs.DB

**SubmitDate**: 2023-12-25    [abs](http://arxiv.org/abs/2312.15591v1) [paper-pdf](http://arxiv.org/pdf/2312.15591v1)

**Authors**: Qi Hu, Haoran Li, Jiaxin Bai, Yangqiu Song

**Abstract**: In the era of big data and rapidly evolving information systems, efficient and accurate data retrieval has become increasingly crucial. Neural graph databases (NGDBs) have emerged as a powerful paradigm that combines the strengths of graph databases (graph DBs) and neural networks to enable efficient storage, retrieval, and analysis of graph-structured data. The usage of neural embedding storage and complex neural logical query answering provides NGDBs with generalization ability. When the graph is incomplete, by extracting latent patterns and representations, neural graph databases can fill gaps in the graph structure, revealing hidden relationships and enabling accurate query answering. Nevertheless, this capability comes with inherent trade-offs, as it introduces additional privacy risks to the database. Malicious attackers can infer more sensitive information in the database using well-designed combinatorial queries, such as by comparing the answer sets of where Turing Award winners born before 1950 and after 1940 lived, the living places of Turing Award winner Hinton are probably exposed, although the living places may have been deleted in the training due to the privacy concerns. In this work, inspired by the privacy protection in graph embeddings, we propose a privacy-preserving neural graph database (P-NGDB) to alleviate the risks of privacy leakage in NGDBs. We introduce adversarial training techniques in the training stage to force the NGDBs to generate indistinguishable answers when queried with private information, enhancing the difficulty of inferring sensitive information through combinations of multiple innocuous queries. Extensive experiment results on three datasets show that P-NGDB can effectively protect private information in the graph database while delivering high-quality public answers responses to queries.

摘要: 在大数据和快速发展的信息系统的时代，高效和准确的数据检索变得越来越重要。神经图形数据库(NGDB)已经成为一种强大的范例，它结合了图形数据库(图形数据库)和神经网络的优点，使得能够有效地存储、检索和分析图形结构的数据。神经嵌入存储和复杂神经逻辑查询回答的使用为NGDB提供了泛化能力。当图不完整时，通过提取潜在模式和表示，神经图库可以填补图结构中的空白，揭示隐藏的关系，并使查询得到准确的回答。尽管如此，这种能力也伴随着固有的权衡，因为它会给数据库带来额外的隐私风险。恶意攻击者可以使用精心设计的组合查询来推断数据库中更敏感的信息，例如通过比较1950年之前出生的图灵奖获得者和1940年后出生的图灵奖获得者的答案集，图灵奖获得者辛顿的居住地可能会被曝光，尽管出于隐私考虑，在训练中可能已经删除了居住地。在这项工作中，我们受到图嵌入中隐私保护的启发，提出了一种隐私保护神经图库(P-NGDB)来缓解NGDB中隐私泄露的风险。我们在训练阶段引入对抗性训练技术，迫使NGDB在查询私有信息时产生难以区分的答案，增加了通过组合多个无害查询来推断敏感信息的难度。在三个数据集上的大量实验结果表明，P-NGDB可以有效地保护图形数据库中的私有信息，同时提供高质量的公共查询响应。



## **8. An Initial Investigation of Neural Replay Simulator for Over-the-Air Adversarial Perturbations to Automatic Speaker Verification**

用于自动说话人确认的空中对抗扰动神经重放模拟器的初步研究 cs.SD

Accepted in ICASSP 2024

**SubmitDate**: 2023-12-24    [abs](http://arxiv.org/abs/2310.05354v3) [paper-pdf](http://arxiv.org/pdf/2310.05354v3)

**Authors**: Jiaqi Li, Li Wang, Liumeng Xue, Lei Wang, Zhizheng Wu

**Abstract**: Deep Learning has advanced Automatic Speaker Verification (ASV) in the past few years. Although it is known that deep learning-based ASV systems are vulnerable to adversarial examples in digital access, there are few studies on adversarial attacks in the context of physical access, where a replay process (i.e., over the air) is involved. An over-the-air attack involves a loudspeaker, a microphone, and a replaying environment that impacts the movement of the sound wave. Our initial experiment confirms that the replay process impacts the effectiveness of the over-the-air attack performance. This study performs an initial investigation towards utilizing a neural replay simulator to improve over-the-air adversarial attack robustness. This is achieved by using a neural waveform synthesizer to simulate the replay process when estimating the adversarial perturbations. Experiments conducted on the ASVspoof2019 dataset confirm that the neural replay simulator can considerably increase the success rates of over-the-air adversarial attacks. This raises the concern for adversarial attacks on speaker verification in physical access applications.

摘要: 在过去的几年里，深度学习发展了自动说话人确认(ASV)。虽然众所周知，基于深度学习的ASV系统在数字访问中容易受到敌意攻击，但在涉及重播过程(即空中重播)的物理访问环境中，很少有关于对抗性攻击的研究。空中攻击包括扬声器、麦克风和影响声波移动的重放环境。我们的初步实验证实，重放过程会影响空中攻击性能的有效性。本研究对利用神经重放模拟器来提高空中对抗攻击的稳健性进行了初步的研究。这是通过使用神经波形合成器来模拟在估计对抗性扰动时的重播过程来实现的。在ASVspoof2019数据集上进行的实验证实，神经重放模拟器可以显著提高空中对抗性攻击的成功率。这引起了人们对物理访问应用中说话人验证的对抗性攻击的关注。



## **9. Adversarial Data Poisoning for Fake News Detection: How to Make a Model Misclassify a Target News without Modifying It**

用于虚假新闻检测的对抗性数据中毒：如何使模型在不修改目标新闻的情况下对其进行错误分类 cs.LG

**SubmitDate**: 2023-12-23    [abs](http://arxiv.org/abs/2312.15228v1) [paper-pdf](http://arxiv.org/pdf/2312.15228v1)

**Authors**: Federico Siciliano, Luca Maiano, Lorenzo Papa, Federica Baccin, Irene Amerini, Fabrizio Silvestri

**Abstract**: Fake news detection models are critical to countering disinformation but can be manipulated through adversarial attacks. In this position paper, we analyze how an attacker can compromise the performance of an online learning detector on specific news content without being able to manipulate the original target news. In some contexts, such as social networks, where the attacker cannot exert complete control over all the information, this scenario can indeed be quite plausible. Therefore, we show how an attacker could potentially introduce poisoning data into the training data to manipulate the behavior of an online learning method. Our initial findings reveal varying susceptibility of logistic regression models based on complexity and attack type.

摘要: 假新闻检测模型对打击虚假信息至关重要，但可以通过对抗性攻击来操纵。在这份立场文件中，我们分析了攻击者如何在不能操纵原始目标新闻的情况下，损害在线学习检测器在特定新闻内容上的性能。在某些情况下，例如社交网络，攻击者无法完全控制所有信息，这种情况确实很有可能发生。因此，我们展示了攻击者如何潜在地将中毒数据引入训练数据以操纵在线学习方法的行为。我们的初步发现揭示了基于复杂性和攻击类型的Logistic回归模型的不同易感性。



## **10. Towards Transferable Adversarial Attacks with Centralized Perturbation**

集中式扰动下的可转移对抗性攻击 cs.CV

10 pages, 9 figures, accepted by AAAI 2024

**SubmitDate**: 2023-12-23    [abs](http://arxiv.org/abs/2312.06199v2) [paper-pdf](http://arxiv.org/pdf/2312.06199v2)

**Authors**: Shangbo Wu, Yu-an Tan, Yajie Wang, Ruinan Ma, Wencong Ma, Yuanzhang Li

**Abstract**: Adversarial transferability enables black-box attacks on unknown victim deep neural networks (DNNs), rendering attacks viable in real-world scenarios. Current transferable attacks create adversarial perturbation over the entire image, resulting in excessive noise that overfit the source model. Concentrating perturbation to dominant image regions that are model-agnostic is crucial to improving adversarial efficacy. However, limiting perturbation to local regions in the spatial domain proves inadequate in augmenting transferability. To this end, we propose a transferable adversarial attack with fine-grained perturbation optimization in the frequency domain, creating centralized perturbation. We devise a systematic pipeline to dynamically constrain perturbation optimization to dominant frequency coefficients. The constraint is optimized in parallel at each iteration, ensuring the directional alignment of perturbation optimization with model prediction. Our approach allows us to centralize perturbation towards sample-specific important frequency features, which are shared by DNNs, effectively mitigating source model overfitting. Experiments demonstrate that by dynamically centralizing perturbation on dominating frequency coefficients, crafted adversarial examples exhibit stronger transferability, and allowing them to bypass various defenses.

摘要: 对抗性可转移性使得对未知受害者深度神经网络（DNN）的黑盒攻击成为可能，从而使攻击在现实世界中变得可行。当前的可转移攻击在整个图像上产生对抗性扰动，导致过度拟合源模型的过度噪声。将扰动集中到与模型无关的主要图像区域对于提高对抗效果至关重要。然而，在空间域中的局部区域限制扰动证明在增强可转移性不足。为此，我们提出了一种可转移的对抗攻击，在频域中进行细粒度扰动优化，创建集中式扰动。我们设计了一个系统的管道，动态约束扰动优化的主频系数。在每次迭代中并行优化约束，确保扰动优化与模型预测的方向对准。我们的方法允许我们集中对样本特定的重要频率特征的扰动，这些特征由DNN共享，有效地减轻了源模型的过拟合。实验表明，通过动态地将扰动集中在主导频率系数上，精心制作的对抗性示例具有更强的可移植性，并允许它们绕过各种防御。



## **11. SODA: Protecting Proprietary Information in On-Device Machine Learning Models**

SODA：在设备上机器学习模型中保护专有信息 cs.LG

**SubmitDate**: 2023-12-22    [abs](http://arxiv.org/abs/2312.15036v1) [paper-pdf](http://arxiv.org/pdf/2312.15036v1)

**Authors**: Akanksha Atrey, Ritwik Sinha, Saayan Mitra, Prashant Shenoy

**Abstract**: The growth of low-end hardware has led to a proliferation of machine learning-based services in edge applications. These applications gather contextual information about users and provide some services, such as personalized offers, through a machine learning (ML) model. A growing practice has been to deploy such ML models on the user's device to reduce latency, maintain user privacy, and minimize continuous reliance on a centralized source. However, deploying ML models on the user's edge device can leak proprietary information about the service provider. In this work, we investigate on-device ML models that are used to provide mobile services and demonstrate how simple attacks can leak proprietary information of the service provider. We show that different adversaries can easily exploit such models to maximize their profit and accomplish content theft. Motivated by the need to thwart such attacks, we present an end-to-end framework, SODA, for deploying and serving on edge devices while defending against adversarial usage. Our results demonstrate that SODA can detect adversarial usage with 89% accuracy in less than 50 queries with minimal impact on service performance, latency, and storage.

摘要: 低端硬件的增长导致了边缘应用中基于机器学习的服务的激增。这些应用程序收集有关用户的上下文信息，并通过机器学习(ML)模型提供一些服务，如个性化服务。越来越多的做法是在用户设备上部署这样的ML模型，以减少延迟，维护用户隐私，并最大限度地减少对集中式来源的持续依赖。然而，在用户的边缘设备上部署ML模型可能会泄露有关服务提供商的专有信息。在这项工作中，我们研究了用于提供移动服务的设备上ML模型，并演示了简单的攻击如何泄漏服务提供商的专有信息。我们表明，不同的对手可以很容易地利用这种模型来最大化他们的利润，并完成内容窃取。出于阻止此类攻击的需要，我们提出了一个端到端框架SODA，用于在边缘设备上部署和服务，同时防御恶意使用。我们的结果表明，SODA可以在不到50个查询的情况下以89%的准确率检测恶意使用，并且对服务性能、延迟和存储的影响最小。



## **12. Differentiable JPEG: The Devil is in the Details**

与众不同的JPEG：魔鬼在细节中 cs.CV

Accepted at WACV 2024. Project page:  https://christophreich1996.github.io/differentiable_jpeg/ WACV paper:  https://openaccess.thecvf.com/content/WACV2024/html/Reich_Differentiable_JPEG_The_Devil_Is_in_the_Details_WACV_2024_paper.html

**SubmitDate**: 2023-12-22    [abs](http://arxiv.org/abs/2309.06978v4) [paper-pdf](http://arxiv.org/pdf/2309.06978v4)

**Authors**: Christoph Reich, Biplob Debnath, Deep Patel, Srimat Chakradhar

**Abstract**: JPEG remains one of the most widespread lossy image coding methods. However, the non-differentiable nature of JPEG restricts the application in deep learning pipelines. Several differentiable approximations of JPEG have recently been proposed to address this issue. This paper conducts a comprehensive review of existing diff. JPEG approaches and identifies critical details that have been missed by previous methods. To this end, we propose a novel diff. JPEG approach, overcoming previous limitations. Our approach is differentiable w.r.t. the input image, the JPEG quality, the quantization tables, and the color conversion parameters. We evaluate the forward and backward performance of our diff. JPEG approach against existing methods. Additionally, extensive ablations are performed to evaluate crucial design choices. Our proposed diff. JPEG resembles the (non-diff.) reference implementation best, significantly surpassing the recent-best diff. approach by $3.47$dB (PSNR) on average. For strong compression rates, we can even improve PSNR by $9.51$dB. Strong adversarial attack results are yielded by our diff. JPEG, demonstrating the effective gradient approximation. Our code is available at https://github.com/necla-ml/Diff-JPEG.

摘要: JPEG仍然是应用最广泛的有损图像编码方法之一。然而，JPEG的不可微特性限制了其在深度学习管道中的应用。为了解决这个问题，最近已经提出了几种JPEG的可微近似。本文对现有的DIFF进行了全面的回顾。JPEG处理并确定了以前方法遗漏的关键细节。为此，我们提出了一个新颖的Diff。JPEG方法，克服了以前的限制。我们的方法是可微的W.r.t。输入图像、JPEG质量、量化表和颜色转换参数。我们评估了DIFF的向前和向后性能。JPEG方法与现有方法的对比。此外，还进行了广泛的消融，以评估关键的设计选择。我们提议的不同之处。JPEG与(Non-Diff.)参考实现最好，大大超过了最近最好的差异。平均接近3.47美元分贝(PSNR)。对于强压缩率，我们甚至可以将PSNR提高9.51美元分贝。强大的对抗性攻击结果是由我们的差异产生的。JPEG格式，演示了有效的渐变近似。我们的代码可以在https://github.com/necla-ml/Diff-JPEG.上找到



## **13. Hierarchical Multi-Agent Reinforcement Learning for Assessing False-Data Injection Attacks on Transportation Networks**

基于分层多智能体强化学习的交通网络虚假数据注入攻击评估 cs.AI

**SubmitDate**: 2023-12-22    [abs](http://arxiv.org/abs/2312.14625v1) [paper-pdf](http://arxiv.org/pdf/2312.14625v1)

**Authors**: Taha Eghtesad, Sirui Li, Yevgeniy Vorobeychik, Aron Laszka

**Abstract**: The increasing reliance of drivers on navigation applications has made transportation networks more susceptible to data-manipulation attacks by malicious actors. Adversaries may exploit vulnerabilities in the data collection or processing of navigation services to inject false information, and to thus interfere with the drivers' route selection. Such attacks can significantly increase traffic congestions, resulting in substantial waste of time and resources, and may even disrupt essential services that rely on road networks. To assess the threat posed by such attacks, we introduce a computational framework to find worst-case data-injection attacks against transportation networks. First, we devise an adversarial model with a threat actor who can manipulate drivers by increasing the travel times that they perceive on certain roads. Then, we employ hierarchical multi-agent reinforcement learning to find an approximate optimal adversarial strategy for data manipulation. We demonstrate the applicability of our approach through simulating attacks on the Sioux Falls, ND network topology.

摘要: 司机越来越依赖导航应用程序，这使得交通网络更容易受到恶意行为者的数据操纵攻击。攻击者可能会利用导航服务的数据收集或处理中的漏洞来注入虚假信息，从而干扰司机的路线选择。此类攻击可能会显著加剧交通拥堵，导致大量时间和资源的浪费，甚至可能扰乱依赖道路网络的基本服务。为了评估这类攻击造成的威胁，我们引入了一个计算框架来发现针对交通网络的最坏情况下的数据注入攻击。首先，我们设计了一个带有威胁参与者的对抗性模型，该威胁参与者可以通过增加司机在某些道路上感知的旅行时间来操纵司机。然后，我们使用分层多智能体强化学习来寻找数据操作的近似最优对抗策略。通过模拟对苏福尔斯网络拓扑结构的攻击，验证了该方法的适用性。



## **14. Complex Graph Laplacian Regularizer for Inferencing Grid States**

用于网格状态推断的复图拉普拉斯正则化算法 eess.SP

**SubmitDate**: 2023-12-22    [abs](http://arxiv.org/abs/2307.01906v2) [paper-pdf](http://arxiv.org/pdf/2307.01906v2)

**Authors**: Chinthaka Dinesh, Junfei Wang, Gene Cheung, Pirathayini Srikantha

**Abstract**: In order to maintain stable grid operations, system monitoring and control processes require the computation of grid states (e.g. voltage magnitude and angles) at high granularity. It is necessary to infer these grid states from measurements generated by a limited number of sensors like phasor measurement units (PMUs) that can be subjected to delays and losses due to channel artefacts, and/or adversarial attacks (e.g. denial of service, jamming, etc.). We propose a novel graph signal processing (GSP) based algorithm to interpolate states of the entire grid from observations of a small number of grid measurements. It is a two-stage process, where first an underlying Hermitian graph is learnt empirically from existing grid datasets. Then, the graph is used to interpolate missing grid signal samples in linear time. With our proposal, we can effectively reconstruct grid signals with significantly smaller number of observations when compared to existing traditional approaches (e.g. state estimation). In contrast to existing GSP approaches, we do not require knowledge of the underlying grid structure and parameters and are able to guarantee fast spectral optimization. We demonstrate the computational efficacy and accuracy of our proposal via practical studies conducted on the IEEE 118 bus system.

摘要: 为了维持稳定的电网运行，系统监测和控制过程需要计算高粒度的电网状态(如电压幅值和角度)。有必要从有限数量的传感器(如相量测量单元(PMU))生成的测量结果中推断这些网格状态，这些传感器可能由于信道伪影和/或对抗性攻击(例如拒绝服务、干扰等)而受到延迟和损失。我们提出了一种基于图信号处理(GSP)的新算法，该算法根据少量网格测量的观测值来内插整个网格的状态。这是一个分两个阶段的过程，首先从现有的网格数据集中经验地学习潜在的厄米特图。然后，使用该图在线性时间内对缺失的网格信号样本进行内插。与现有的传统方法(例如状态估计)相比，我们的建议可以用明显更少的观测值来有效地重建网格信号。与现有的GSP方法相比，我们不需要底层网格结构和参数的知识，并且能够保证快速的频谱优化。通过在IEEE118节点系统上进行的实际研究，证明了该方法的计算效率和准确性。



## **15. Backdoor Attack with Sparse and Invisible Trigger**

稀疏隐触发后门攻击 cs.CV

The first two authors contributed equally to this work. 13 pages

**SubmitDate**: 2023-12-22    [abs](http://arxiv.org/abs/2306.06209v2) [paper-pdf](http://arxiv.org/pdf/2306.06209v2)

**Authors**: Yinghua Gao, Yiming Li, Xueluan Gong, Zhifeng Li, Shu-Tao Xia, Qian Wang

**Abstract**: Deep neural networks (DNNs) are vulnerable to backdoor attacks, where the adversary manipulates a small portion of training data such that the victim model predicts normally on the benign samples but classifies the triggered samples as the target class. The backdoor attack is an emerging yet threatening training-phase threat, leading to serious risks in DNN-based applications. In this paper, we revisit the trigger patterns of existing backdoor attacks. We reveal that they are either visible or not sparse and therefore are not stealthy enough. More importantly, it is not feasible to simply combine existing methods to design an effective sparse and invisible backdoor attack. To address this problem, we formulate the trigger generation as a bi-level optimization problem with sparsity and invisibility constraints and propose an effective method to solve it. The proposed method is dubbed sparse and invisible backdoor attack (SIBA). We conduct extensive experiments on benchmark datasets under different settings, which verify the effectiveness of our attack and its resistance to existing backdoor defenses. The codes for reproducing main experiments are available at \url{https://github.com/YinghuaGao/SIBA}.

摘要: 深度神经网络（DNN）容易受到后门攻击，其中对手操纵一小部分训练数据，使得受害者模型正常预测良性样本，但将触发的样本分类为目标类。后门攻击是一种新出现的但具有威胁性的训练阶段威胁，导致基于DNN的应用程序面临严重风险。在本文中，我们重新审视现有的后门攻击的触发模式。我们发现，它们要么是可见的，要么不是稀疏的，因此不够隐蔽。更重要的是，简单地结合现有的方法来设计有效的稀疏和不可见的后门攻击是不可行的。为了解决这个问题，我们将触发器生成问题描述为一个具有稀疏性和不可见性约束的双层优化问题，并提出了一种有效的方法来解决这个问题，该方法被称为稀疏和不可见后门攻击（SIBA）。我们在不同的设置下对基准数据集进行了大量的实验，验证了我们的攻击的有效性及其对现有后门防御的抵抗力。复制主要实验的代码可在\url{https：//github.com/YinghuaGao/SIBA}获得。



## **16. Asymmetric Bias in Text-to-Image Generation with Adversarial Attacks**

对抗性攻击下文本到图像生成中的非对称偏向 cs.LG

preprint version

**SubmitDate**: 2023-12-22    [abs](http://arxiv.org/abs/2312.14440v1) [paper-pdf](http://arxiv.org/pdf/2312.14440v1)

**Authors**: Haz Sameen Shahgir, Xianghao Kong, Greg Ver Steeg, Yue Dong

**Abstract**: The widespread use of Text-to-Image (T2I) models in content generation requires careful examination of their safety, including their robustness to adversarial attacks. Despite extensive research into this, the reasons for their effectiveness are underexplored. This paper presents an empirical study on adversarial attacks against T2I models, focusing on analyzing factors associated with attack success rates (ASRs). We introduce a new attack objective - entity swapping using adversarial suffixes and two gradient-based attack algorithms. Human and automatic evaluations reveal the asymmetric nature of ASRs on entity swap: for example, it is easier to replace "human" with "robot" in the prompt "a human dancing in the rain." with an adversarial suffix but is significantly harder in reverse. We further propose probing metrics to establish indicative signals from the model's beliefs to the adversarial ASR. We identify conditions resulting in a 60% success probability for adversarial attacks and others where this likelihood drops below 5%.

摘要: 文本到图像(T2I)模型在内容生成中的广泛使用需要仔细检查它们的安全性，包括它们对对手攻击的健壮性。尽管对此进行了广泛的研究，但其有效性的原因仍未得到充分探索。本文对T2I模型的对抗性攻击进行了实证研究，重点分析了影响攻击成功率(ASR)的因素。提出了一种新的攻击目标实体交换算法，利用对抗性后缀和两种基于梯度的攻击算法。人工评估和自动评估揭示了ASR在实体交换上的不对称性质：例如，在提示符“a Human in the雨中跳舞”中，更容易将“Human”替换为“bot”。有一个对抗性后缀，但反转起来要难得多。我们进一步提出了探测度量来建立从模型信念到对抗性ASR的指示性信号。我们确定了导致对抗性攻击成功概率为60%的条件，以及其他可能性降至5%以下的条件。



## **17. Elevating Defenses: Bridging Adversarial Training and Watermarking for Model Resilience**

提升防御：在对抗性训练和模型复原力水印之间架起桥梁 cs.LG

Accepted at DAI Workshop, AAAI 2024

**SubmitDate**: 2023-12-21    [abs](http://arxiv.org/abs/2312.14260v1) [paper-pdf](http://arxiv.org/pdf/2312.14260v1)

**Authors**: Janvi Thakkar, Giulio Zizzo, Sergio Maffeis

**Abstract**: Machine learning models are being used in an increasing number of critical applications; thus, securing their integrity and ownership is critical. Recent studies observed that adversarial training and watermarking have a conflicting interaction. This work introduces a novel framework to integrate adversarial training with watermarking techniques to fortify against evasion attacks and provide confident model verification in case of intellectual property theft. We use adversarial training together with adversarial watermarks to train a robust watermarked model. The key intuition is to use a higher perturbation budget to generate adversarial watermarks compared to the budget used for adversarial training, thus avoiding conflict. We use the MNIST and Fashion-MNIST datasets to evaluate our proposed technique on various model stealing attacks. The results obtained consistently outperform the existing baseline in terms of robustness performance and further prove the resilience of this defense against pruning and fine-tuning removal attacks.

摘要: 机器学习模型正在越来越多的关键应用中使用；因此，确保它们的完整性和所有权至关重要。最近的研究发现，对抗性训练和水印之间存在相互冲突的作用。这项工作引入了一种新的框架，将对抗性训练与水印技术相结合，以加强对逃避攻击的防御，并在知识产权被盗的情况下提供可信的模型验证。我们使用对抗性训练和对抗性水印相结合的方法来训练一个健壮的水印模型。关键的直觉是，与用于对抗性训练的预算相比，使用更高的扰动预算来生成对抗性水印，从而避免冲突。我们使用MNIST和Fashion-MNIST数据集来评估我们提出的针对各种模型窃取攻击的技术。得到的结果在稳健性性能方面始终优于现有的基线，并进一步证明了该防御措施对剪枝和微调删除攻击的弹性。



## **18. Open-Set: ID Card Presentation Attack Detection using Neural Transfer Style**

开集：基于神经传递方式的身份证呈现攻击检测 cs.CV

**SubmitDate**: 2023-12-21    [abs](http://arxiv.org/abs/2312.13993v1) [paper-pdf](http://arxiv.org/pdf/2312.13993v1)

**Authors**: Reuben Markham, Juan M. Espin, Mario Nieto-Hidalgo, Juan E. Tapia

**Abstract**: The accurate detection of ID card Presentation Attacks (PA) is becoming increasingly important due to the rising number of online/remote services that require the presentation of digital photographs of ID cards for digital onboarding or authentication. Furthermore, cybercriminals are continuously searching for innovative ways to fool authentication systems to gain unauthorized access to these services. Although advances in neural network design and training have pushed image classification to the state of the art, one of the main challenges faced by the development of fraud detection systems is the curation of representative datasets for training and evaluation. The handcrafted creation of representative presentation attack samples often requires expertise and is very time-consuming, thus an automatic process of obtaining high-quality data is highly desirable. This work explores ID card Presentation Attack Instruments (PAI) in order to improve the generation of samples with four Generative Adversarial Networks (GANs) based image translation models and analyses the effectiveness of the generated data for training fraud detection systems. Using open-source data, we show that synthetic attack presentations are an adequate complement for additional real attack presentations, where we obtain an EER performance increase of 0.63% points for print attacks and a loss of 0.29% for screen capture attacks.

摘要: 由于越来越多的在线/远程服务需要提供身份证的数字照片以进行数字登录或身份验证，因此准确检测身份证显示攻击(PA)变得越来越重要。此外，网络犯罪分子不断地寻找创新的方法来欺骗身份验证系统，以获得对这些服务的未经授权的访问。尽管神经网络设计和训练的进步将图像分类推向了最先进的水平，但欺诈检测系统的发展面临的主要挑战之一是为训练和评估挑选具有代表性的数据集。手工创建典型的表示攻击样本通常需要专业知识并且非常耗时，因此非常需要自动获取高质量数据的过程。为了改进基于四种生成对抗网络(GANS)的图像翻译模型的样本生成，并分析生成的数据用于训练欺诈检测系统的有效性，对身份证呈现攻击工具(PAI)进行了研究。使用开源数据，我们表明，合成攻击演示是对额外的真实攻击演示的足够补充，其中，对于打印攻击，我们获得了0.63%的EER性能提升，而对于屏幕捕获攻击，我们获得了0.29%的损失。



## **19. AutoAugment Input Transformation for Highly Transferable Targeted Attacks**

用于高可转移性目标攻击的自动增强输入转换 cs.CV

10 pages, 6 figures

**SubmitDate**: 2023-12-21    [abs](http://arxiv.org/abs/2312.14218v1) [paper-pdf](http://arxiv.org/pdf/2312.14218v1)

**Authors**: Haobo Lu, Xin Liu, Kun He

**Abstract**: Deep Neural Networks (DNNs) are widely acknowledged to be susceptible to adversarial examples, wherein imperceptible perturbations are added to clean examples through diverse input transformation attacks. However, these methods originally designed for non-targeted attacks exhibit low success rates in targeted attacks. Recent targeted adversarial attacks mainly pay attention to gradient optimization, attempting to find the suitable perturbation direction. However, few of them are dedicated to input transformation.In this work, we observe a positive correlation between the logit/probability of the target class and diverse input transformation methods in targeted attacks. To this end, we propose a novel targeted adversarial attack called AutoAugment Input Transformation (AAIT). Instead of relying on hand-made strategies, AAIT searches for the optimal transformation policy from a transformation space comprising various operations. Then, AAIT crafts adversarial examples using the found optimal transformation policy to boost the adversarial transferability in targeted attacks. Extensive experiments conducted on CIFAR-10 and ImageNet-Compatible datasets demonstrate that the proposed AAIT surpasses other transfer-based targeted attacks significantly.

摘要: 深度神经网络(DNN)被公认为容易受到敌意例子的影响，在这种例子中，通过不同的输入变换攻击将不可察觉的扰动添加到干净的例子中。然而，这些方法最初是为非目标攻击设计的，但在目标攻击中成功率很低。最近的定向对抗性攻击主要关注梯度优化，试图找到合适的扰动方向。在本工作中，我们观察到目标类的Logit/概率与目标攻击中不同的输入转换方法之间存在正相关关系。为此，我们提出了一种新的有针对性的对抗性攻击，称为自动增强输入变换(AAIT)。AAIT不依赖于手工制定的策略，而是从包含各种操作的变换空间中搜索最优变换策略。然后，AAIT使用找到的最优转换策略来创建对抗性实例，以提高定向攻击中的对抗性可转移性。在CIFAR-10和ImageNet兼容的数据集上进行的大量实验表明，所提出的AAIT攻击明显优于其他基于传输的定向攻击。



## **20. Adversarial Infrared Curves: An Attack on Infrared Pedestrian Detectors in the Physical World**

对抗性红外曲线：对物理世界中红外行人探测器的攻击 cs.CR

**SubmitDate**: 2023-12-21    [abs](http://arxiv.org/abs/2312.14217v1) [paper-pdf](http://arxiv.org/pdf/2312.14217v1)

**Authors**: Chengyin Hu, Weiwen Shi

**Abstract**: Deep neural network security is a persistent concern, with considerable research on visible light physical attacks but limited exploration in the infrared domain. Existing approaches, like white-box infrared attacks using bulb boards and QR suits, lack realism and stealthiness. Meanwhile, black-box methods with cold and hot patches often struggle to ensure robustness. To bridge these gaps, we propose Adversarial Infrared Curves (AdvIC). Using Particle Swarm Optimization, we optimize two Bezier curves and employ cold patches in the physical realm to introduce perturbations, creating infrared curve patterns for physical sample generation. Our extensive experiments confirm AdvIC's effectiveness, achieving 94.8\% and 67.2\% attack success rates for digital and physical attacks, respectively. Stealthiness is demonstrated through a comparative analysis, and robustness assessments reveal AdvIC's superiority over baseline methods. When deployed against diverse advanced detectors, AdvIC achieves an average attack success rate of 76.8\%, emphasizing its robust nature. we explore adversarial defense strategies against AdvIC and examine its impact under various defense mechanisms. Given AdvIC's substantial security implications for real-world vision-based applications, urgent attention and mitigation efforts are warranted.

摘要: 深度神经网络的安全性一直是一个令人关注的问题，对可见光物理攻击的研究很多，但在红外领域的探索有限。现有的方法，如使用灯泡板和QR套装的白盒红外攻击，缺乏真实性和隐蔽性。与此同时，带有冷补丁和热补丁的黑盒方法往往难以确保健壮性。为了弥补这些差距，我们提出了对抗性红外曲线(Advic)。利用粒子群算法，我们优化了两条Bezier曲线，并利用物理领域中的冷斑来引入扰动，创建了用于物理样本生成的红外曲线图案。我们的大量实验证实了Advic的有效性，数字攻击和物理攻击的攻击成功率分别达到94.8%和67.2%。隐蔽性通过比较分析得到了证明，健壮性评估显示了Advic相对于基线方法的优势。当部署在不同的高级探测器上时，Advic的平均攻击成功率为76.8\%，强调了其健壮性。我们探讨了针对Advic的对抗性防御策略，并考察了其在各种防御机制下的影响。鉴于Advic对现实世界基于视觉的应用程序的重大安全影响，迫切需要关注和缓解努力。



## **21. Quantum Neural Networks under Depolarization Noise: Exploring White-Box Attacks and Defenses**

去极化噪声下的量子神经网络：白盒攻击与防御探索 quant-ph

Poster at Quantum Techniques in Machine Learning (QTML) 2023

**SubmitDate**: 2023-12-21    [abs](http://arxiv.org/abs/2311.17458v2) [paper-pdf](http://arxiv.org/pdf/2311.17458v2)

**Authors**: David Winderl, Nicola Franco, Jeanette Miriam Lorenz

**Abstract**: Leveraging the unique properties of quantum mechanics, Quantum Machine Learning (QML) promises computational breakthroughs and enriched perspectives where traditional systems reach their boundaries. However, similarly to classical machine learning, QML is not immune to adversarial attacks. Quantum adversarial machine learning has become instrumental in highlighting the weak points of QML models when faced with adversarial crafted feature vectors. Diving deep into this domain, our exploration shines light on the interplay between depolarization noise and adversarial robustness. While previous results enhanced robustness from adversarial threats through depolarization noise, our findings paint a different picture. Interestingly, adding depolarization noise discontinued the effect of providing further robustness for a multi-class classification scenario. Consolidating our findings, we conducted experiments with a multi-class classifier adversarially trained on gate-based quantum simulators, further elucidating this unexpected behavior.

摘要: 利用量子力学的独特性质，量子机器学习(QML)有望在传统系统达到其边界的地方实现计算突破和丰富视角。然而，与经典机器学习类似，QML也不能幸免于对手攻击。量子对抗性机器学习已成为突出QML模型在面对对抗性特制特征向量时的弱点的工具。深入到这个领域，我们的探索揭示了去极化噪声和对手稳健性之间的相互作用。虽然之前的结果通过去极化噪声增强了对抗威胁的稳健性，但我们的发现描绘了一幅不同的图景。有趣的是，添加去极化噪声会中断为多类分类场景提供进一步稳健性的效果。综合我们的发现，我们用一个多类分类器进行了实验，该分类器在基于门的量子模拟器上进行了相反的训练，进一步阐明了这种意想不到的行为。



## **22. Where and How to Attack? A Causality-Inspired Recipe for Generating Counterfactual Adversarial Examples**

向哪里进攻，如何进攻？由因果关系启发生成反事实对抗性例子的秘诀 cs.LG

Accepted by AAAI-2024

**SubmitDate**: 2023-12-21    [abs](http://arxiv.org/abs/2312.13628v1) [paper-pdf](http://arxiv.org/pdf/2312.13628v1)

**Authors**: Ruichu Cai, Yuxuan Zhu, Jie Qiao, Zefeng Liang, Furui Liu, Zhifeng Hao

**Abstract**: Deep neural networks (DNNs) have been demonstrated to be vulnerable to well-crafted \emph{adversarial examples}, which are generated through either well-conceived $\mathcal{L}_p$-norm restricted or unrestricted attacks. Nevertheless, the majority of those approaches assume that adversaries can modify any features as they wish, and neglect the causal generating process of the data, which is unreasonable and unpractical. For instance, a modification in income would inevitably impact features like the debt-to-income ratio within a banking system. By considering the underappreciated causal generating process, first, we pinpoint the source of the vulnerability of DNNs via the lens of causality, then give theoretical results to answer \emph{where to attack}. Second, considering the consequences of the attack interventions on the current state of the examples to generate more realistic adversarial examples, we propose CADE, a framework that can generate \textbf{C}ounterfactual \textbf{AD}versarial \textbf{E}xamples to answer \emph{how to attack}. The empirical results demonstrate CADE's effectiveness, as evidenced by its competitive performance across diverse attack scenarios, including white-box, transfer-based, and random intervention attacks.

摘要: 深度神经网络(DNN)已经被证明容易受到精心设计的对手例子的攻击，这些例子是通过精心设计的数学{L}_p$-范数受限或非受限攻击而产生的。然而，这些方法中的大多数都假设对手可以随意修改任何特征，而忽略了数据的因果生成过程，这是不合理和不切实际的。例如，收入的调整将不可避免地影响银行体系内的债务收入比等特征。通过考虑被低估的因果生成过程，我们首先通过因果镜头找出DNN脆弱性的来源，然后给出理论结果来回答{攻击在哪里}。其次，考虑到攻击干预对实例的当前状态的影响，为了生成更真实的对抗性实例，我们提出了CADE框架，它可以生成\extbf{C}非事实\extbf{AD}versariative\extbf{E}样例来回答\emph{如何攻击}。实验结果证明了CADE的有效性，它在各种攻击场景中的竞争性能证明了这一点，包括白盒攻击、基于传输的攻击和随机干预攻击。



## **23. ARBiBench: Benchmarking Adversarial Robustness of Binarized Neural Networks**

ARBiBitch：衡量二值化神经网络对抗健壮性的基准 cs.CV

**SubmitDate**: 2023-12-21    [abs](http://arxiv.org/abs/2312.13575v1) [paper-pdf](http://arxiv.org/pdf/2312.13575v1)

**Authors**: Peng Zhao, Jiehua Zhang, Bowen Peng, Longguang Wang, YingMei Wei, Yu Liu, Li Liu

**Abstract**: Network binarization exhibits great potential for deployment on resource-constrained devices due to its low computational cost. Despite the critical importance, the security of binarized neural networks (BNNs) is rarely investigated. In this paper, we present ARBiBench, a comprehensive benchmark to evaluate the robustness of BNNs against adversarial perturbations on CIFAR-10 and ImageNet. We first evaluate the robustness of seven influential BNNs on various white-box and black-box attacks. The results reveal that 1) The adversarial robustness of BNNs exhibits a completely opposite performance on the two datasets under white-box attacks. 2) BNNs consistently exhibit better adversarial robustness under black-box attacks. 3) Different BNNs exhibit certain similarities in their robustness performance. Then, we conduct experiments to analyze the adversarial robustness of BNNs based on these insights. Our research contributes to inspiring future research on enhancing the robustness of BNNs and advancing their application in real-world scenarios.

摘要: 网络二值化由于其计算成本低，在资源受限的设备上具有很大的部署潜力。尽管二值化神经网络(BNN)的安全性至关重要，但很少有人研究它的安全性。在本文中，我们提出了一种评估BNN对CIFAR-10和ImageNet上的敌意干扰的健壮性的综合基准ARBiBch。我们首先评估了七种有影响力的BNN对各种白盒和黑盒攻击的健壮性。结果表明：1)在白盒攻击下，BNN在两个数据集上的对抗健壮性表现出完全相反的表现。2)BNN在黑盒攻击下表现出更好的对抗健壮性。3)不同的BNN在鲁棒性方面表现出一定的相似性。然后，在此基础上进行实验，分析了BNN的对抗健壮性。我们的研究有助于启发未来增强BNN健壮性的研究，促进其在现实世界场景中的应用。



## **24. Benchmarking and Defending Against Indirect Prompt Injection Attacks on Large Language Models**

大型语言模型上的间接提示注入攻击的基准测试和防御 cs.CL

**SubmitDate**: 2023-12-21    [abs](http://arxiv.org/abs/2312.14197v1) [paper-pdf](http://arxiv.org/pdf/2312.14197v1)

**Authors**: Jingwei Yi, Yueqi Xie, Bin Zhu, Keegan Hines, Emre Kiciman, Guangzhong Sun, Xing Xie, Fangzhao Wu

**Abstract**: Recent remarkable advancements in large language models (LLMs) have led to their widespread adoption in various applications. A key feature of these applications is the combination of LLMs with external content, where user instructions and third-party content are combined to create prompts for LLM processing. These applications, however, are vulnerable to indirect prompt injection attacks, where malicious instructions embedded within external content compromise LLM's output, causing their responses to deviate from user expectations. Despite the discovery of this security issue, no comprehensive analysis of indirect prompt injection attacks on different LLMs is available due to the lack of a benchmark. Furthermore, no effective defense has been proposed.   In this work, we introduce the first benchmark, BIPIA, to measure the robustness of various LLMs and defenses against indirect prompt injection attacks. Our experiments reveal that LLMs with greater capabilities exhibit more vulnerable to indirect prompt injection attacks for text tasks, resulting in a higher ASR. We hypothesize that indirect prompt injection attacks are mainly due to the LLMs' inability to distinguish between instructions and external content. Based on this conjecture, we propose four black-box methods based on prompt learning and a white-box defense methods based on fine-tuning with adversarial training to enable LLMs to distinguish between instructions and external content and ignore instructions in the external content. Our experimental results show that our black-box defense methods can effectively reduce ASR but cannot completely thwart indirect prompt injection attacks, while our white-box defense method can reduce ASR to nearly zero with little adverse impact on the LLM's performance on general tasks. We hope that our benchmark and defenses can inspire future work in this important area.

摘要: 最近，大型语言模型(LLM)的显著进步导致了它们在各种应用中的广泛采用。这些应用程序的一个关键功能是将LLM与外部内容相结合，其中结合了用户指令和第三方内容，以创建LLM处理的提示。然而，这些应用程序容易受到间接提示注入攻击，在这种攻击中，嵌入在外部内容中的恶意指令会危及LLM的输出，导致它们的响应偏离用户预期。尽管发现了这个安全问题，但由于缺乏基准，对不同LLM的间接即时注入攻击没有全面的分析。此外，还没有提出有效的防御措施。在这项工作中，我们引入了第一个基准，BIPIA，来衡量各种LLM的健壮性和对间接即时注入攻击的防御。我们的实验表明，具有更大能力的LLM更容易受到文本任务的间接提示注入攻击，从而导致更高的ASR。我们假设间接提示注入攻击主要是由于LLMS无法区分指令和外部内容。基于这一猜想，我们提出了四种基于快速学习的黑盒方法和一种基于微调和对抗性训练的白盒防御方法，使LLMS能够区分指令和外部内容，并忽略外部内容中的指令。我们的实验结果表明，我们的黑盒防御方法可以有效地降低ASR，但不能完全阻止间接提示注入攻击，而我们的白盒防御方法可以将ASR降低到几乎为零，并且对LLM在一般任务上的性能影响很小。我们希望我们的基准和辩护能够激励这一重要领域的未来工作。



## **25. Adversarial Purification with the Manifold Hypothesis**

流形假设下的对抗性净化 cs.LG

Extended version of paper accepted at AAAI 2024 with supplementary  materials

**SubmitDate**: 2023-12-20    [abs](http://arxiv.org/abs/2210.14404v5) [paper-pdf](http://arxiv.org/pdf/2210.14404v5)

**Authors**: Zhaoyuan Yang, Zhiwei Xu, Jing Zhang, Richard Hartley, Peter Tu

**Abstract**: In this work, we formulate a novel framework for adversarial robustness using the manifold hypothesis. This framework provides sufficient conditions for defending against adversarial examples. We develop an adversarial purification method with this framework. Our method combines manifold learning with variational inference to provide adversarial robustness without the need for expensive adversarial training. Experimentally, our approach can provide adversarial robustness even if attackers are aware of the existence of the defense. In addition, our method can also serve as a test-time defense mechanism for variational autoencoders.

摘要: 在这项工作中，我们使用流形假设建立了一个新的对抗健壮性框架。这一框架为对抗对手的例子提供了充分的条件。在此框架下，我们提出了一种对抗性净化方法。我们的方法结合了流形学习和变分推理，在不需要昂贵的对抗性训练的情况下提供对抗性健壮性。在实验上，即使攻击者知道防御的存在，我们的方法也可以提供对抗的健壮性。此外，我们的方法还可以作为变分自动编码器的测试时间防御机制。



## **26. Adversarial Markov Games: On Adaptive Decision-Based Attacks and Defenses**

对抗性马尔可夫博弈：基于自适应决策的攻防 cs.AI

**SubmitDate**: 2023-12-20    [abs](http://arxiv.org/abs/2312.13435v1) [paper-pdf](http://arxiv.org/pdf/2312.13435v1)

**Authors**: Ilias Tsingenopoulos, Vera Rimmer, Davy Preuveneers, Fabio Pierazzi, Lorenzo Cavallaro, Wouter Joosen

**Abstract**: Despite considerable efforts on making them robust, real-world ML-based systems remain vulnerable to decision based attacks, as definitive proofs of their operational robustness have so far proven intractable. The canonical approach in robustness evaluation calls for adaptive attacks, that is with complete knowledge of the defense and tailored to bypass it. In this study, we introduce a more expansive notion of being adaptive and show how attacks but also defenses can benefit by it and by learning from each other through interaction. We propose and evaluate a framework for adaptively optimizing black-box attacks and defenses against each other through the competitive game they form. To reliably measure robustness, it is important to evaluate against realistic and worst-case attacks. We thus augment both attacks and the evasive arsenal at their disposal through adaptive control, and observe that the same can be done for defenses, before we evaluate them first apart and then jointly under a multi-agent perspective. We demonstrate that active defenses, which control how the system responds, are a necessary complement to model hardening when facing decision-based attacks; then how these defenses can be circumvented by adaptive attacks, only to finally elicit active and adaptive defenses. We validate our observations through a wide theoretical and empirical investigation to confirm that AI-enabled adversaries pose a considerable threat to black-box ML-based systems, rekindling the proverbial arms race where defenses have to be AI-enabled too. Succinctly, we address the challenges posed by adaptive adversaries and develop adaptive defenses, thereby laying out effective strategies in ensuring the robustness of ML-based systems deployed in the real-world.

摘要: 尽管在使其健壮性方面做出了相当大的努力，但现实世界中基于ML的系统仍然容易受到基于决策的攻击，因为迄今为止，其操作健壮性的明确证明已经证明是棘手的。鲁棒性评估的规范方法要求自适应攻击，即与防御的完整知识和定制绕过it.In这项研究中，我们引入了一个更广泛的概念是自适应的，并显示如何攻击，但也防御可以受益于它，并通过相互学习，通过互动。我们提出并评估了一个框架，自适应优化黑盒攻击和防御对彼此通过竞争游戏，他们形成。为了可靠地测量鲁棒性，重要的是要针对现实和最坏情况的攻击进行评估。因此，我们通过自适应控制来增强攻击和规避武器库，并观察到同样可以用于防御，然后我们首先单独评估它们，然后在多智能体的角度下联合评估。我们证明了主动防御，控制系统如何响应，是一个必要的补充模型硬化时，面对基于决策的攻击，然后这些防御可以绕过自适应攻击，只有最终引发主动和自适应防御。我们通过广泛的理论和实证调查来验证我们的观察结果，以确认支持AI的对手对基于黑盒ML的系统构成了相当大的威胁，重新点燃了众所周知的军备竞赛，其中防御也必须支持AI。简而言之，我们解决了自适应对手带来的挑战，并开发了自适应防御，从而制定了有效的策略，以确保部署在现实世界中的基于ML的系统的鲁棒性。



## **27. Universal and Transferable Adversarial Attacks on Aligned Language Models**

对对齐语言模型的通用和可转移的对抗性攻击 cs.CL

Website: http://llm-attacks.org/

**SubmitDate**: 2023-12-20    [abs](http://arxiv.org/abs/2307.15043v2) [paper-pdf](http://arxiv.org/pdf/2307.15043v2)

**Authors**: Andy Zou, Zifan Wang, Nicholas Carlini, Milad Nasr, J. Zico Kolter, Matt Fredrikson

**Abstract**: Because "out-of-the-box" large language models are capable of generating a great deal of objectionable content, recent work has focused on aligning these models in an attempt to prevent undesirable generation. While there has been some success at circumventing these measures -- so-called "jailbreaks" against LLMs -- these attacks have required significant human ingenuity and are brittle in practice. In this paper, we propose a simple and effective attack method that causes aligned language models to generate objectionable behaviors. Specifically, our approach finds a suffix that, when attached to a wide range of queries for an LLM to produce objectionable content, aims to maximize the probability that the model produces an affirmative response (rather than refusing to answer). However, instead of relying on manual engineering, our approach automatically produces these adversarial suffixes by a combination of greedy and gradient-based search techniques, and also improves over past automatic prompt generation methods.   Surprisingly, we find that the adversarial prompts generated by our approach are quite transferable, including to black-box, publicly released LLMs. Specifically, we train an adversarial attack suffix on multiple prompts (i.e., queries asking for many different types of objectionable content), as well as multiple models (in our case, Vicuna-7B and 13B). When doing so, the resulting attack suffix is able to induce objectionable content in the public interfaces to ChatGPT, Bard, and Claude, as well as open source LLMs such as LLaMA-2-Chat, Pythia, Falcon, and others. In total, this work significantly advances the state-of-the-art in adversarial attacks against aligned language models, raising important questions about how such systems can be prevented from producing objectionable information. Code is available at github.com/llm-attacks/llm-attacks.

摘要: 由于“开箱即用”的大型语言模型能够生成大量令人反感的内容，最近的工作重点是调整这些模型，以试图防止不必要的生成。虽然在规避这些措施方面取得了一些成功--即所谓的针对LLMS的“越狱”--但这些攻击需要大量的人类智慧，而且在实践中是脆弱的。在本文中，我们提出了一种简单有效的攻击方法，使对齐的语言模型产生令人反感的行为。具体地说，我们的方法找到了一个后缀，当附加到LLM的广泛查询中以产生令人反感的内容时，旨在最大化该模型产生肯定响应(而不是拒绝回答)的概率。然而，我们的方法不依赖于人工设计，而是通过贪婪和基于梯度的搜索技术相结合来自动生成这些对抗性后缀，并且改进了过去的自动提示生成方法。令人惊讶的是，我们发现我们的方法生成的对抗性提示是相当可转移的，包括到黑盒，公开发布的LLM。具体地说，我们对多个提示(即，要求许多不同类型的不良内容的查询)以及多个模型(在我们的案例中，Vicuna-7B和13B)训练对抗性攻击后缀。这样做时，生成的攻击后缀能够在ChatGPT、Bard和Claude的公共接口以及开源LLM(如llama-2-chat、Pythia、Falcon和其他)中诱导令人反感的内容。总而言之，这项工作极大地推进了针对对齐语言模型的对抗性攻击的最新水平，提出了如何防止此类系统产生令人反感的信息的重要问题。代码可在githorb.com/llm-Attages/llm-Attack上找到。



## **28. On the complexity of sabotage games for network security**

论破坏游戏对网络安全的复杂性 cs.CC

**SubmitDate**: 2023-12-20    [abs](http://arxiv.org/abs/2312.13132v1) [paper-pdf](http://arxiv.org/pdf/2312.13132v1)

**Authors**: Dhananjay Raju, Georgios Bakirtzis, Ufuk Topcu

**Abstract**: Securing dynamic networks against adversarial actions is challenging because of the need to anticipate and counter strategic disruptions by adversarial entities within complex network structures. Traditional game-theoretic models, while insightful, often fail to model the unpredictability and constraints of real-world threat assessment scenarios. We refine sabotage games to reflect the realistic limitations of the saboteur and the network operator. By transforming sabotage games into reachability problems, our approach allows applying existing computational solutions to model realistic restrictions on attackers and defenders within the game. Modifying sabotage games into dynamic network security problems successfully captures the nuanced interplay of strategy and uncertainty in dynamic network security. Theoretically, we extend sabotage games to model network security contexts and thoroughly explore if the additional restrictions raise their computational complexity, often the bottleneck of game theory in practical contexts. Practically, this research sets the stage for actionable insights for developing robust defense mechanisms by understanding what risks to mitigate in dynamically changing networks under threat.

摘要: 确保动态网络不受敌对行动的影响是具有挑战性的，因为需要预测和应对复杂网络结构中敌对实体的战略中断。传统的博弈论模型虽然有洞察力，但往往无法对现实世界威胁评估情景的不可预测性和约束进行建模。我们改进了破坏游戏，以反映破坏者和网络运营商的现实限制。通过将破坏游戏转换为可达性问题，我们的方法允许应用现有的计算解决方案来模拟游戏中对攻击者和防御者的现实限制。将破坏博弈转化为动态网络安全问题，成功地捕捉到了动态网络安全中战略和不确定性的微妙相互作用。从理论上讲，我们将破坏游戏扩展到对网络安全环境进行建模，并深入探索额外的限制是否会增加其计算复杂性，这在实际环境中往往是博弈论的瓶颈。实际上，这项研究通过了解在受到威胁的动态变化的网络中需要缓解哪些风险，为开发强大的防御机制提供了可行的见解。



## **29. Prometheus: Infrastructure Security Posture Analysis with AI-generated Attack Graphs**

普罗米修斯：使用人工智能生成的攻击图进行基础设施安全态势分析 cs.CR

**SubmitDate**: 2023-12-20    [abs](http://arxiv.org/abs/2312.13119v1) [paper-pdf](http://arxiv.org/pdf/2312.13119v1)

**Authors**: Xin Jin, Charalampos Katsis, Fan Sang, Jiahao Sun, Elisa Bertino, Ramana Rao Kompella, Ashish Kundu

**Abstract**: The rampant occurrence of cybersecurity breaches imposes substantial limitations on the progress of network infrastructures, leading to compromised data, financial losses, potential harm to individuals, and disruptions in essential services. The current security landscape demands the urgent development of a holistic security assessment solution that encompasses vulnerability analysis and investigates the potential exploitation of these vulnerabilities as attack paths. In this paper, we propose Prometheus, an advanced system designed to provide a detailed analysis of the security posture of computing infrastructures. Using user-provided information, such as device details and software versions, Prometheus performs a comprehensive security assessment. This assessment includes identifying associated vulnerabilities and constructing potential attack graphs that adversaries can exploit. Furthermore, Prometheus evaluates the exploitability of these attack paths and quantifies the overall security posture through a scoring mechanism. The system takes a holistic approach by analyzing security layers encompassing hardware, system, network, and cryptography. Furthermore, Prometheus delves into the interconnections between these layers, exploring how vulnerabilities in one layer can be leveraged to exploit vulnerabilities in others. In this paper, we present the end-to-end pipeline implemented in Prometheus, showcasing the systematic approach adopted for conducting this thorough security analysis.

摘要: 网络安全漏洞的猖獗发生对网络基础设施的进展施加了很大限制，导致数据泄露、经济损失、对个人的潜在伤害以及基本服务中断。当前的安全形势要求迫切开发一种全面的安全评估解决方案，其中包括漏洞分析，并调查利用这些漏洞作为攻击途径的可能性。在本文中，我们提出了普罗米修斯，这是一个先进的系统，旨在提供详细的分析计算基础设施的安全态势。使用用户提供的信息，如设备详细信息和软件版本，普罗米修斯进行全面的安全评估。该评估包括识别相关漏洞和构建潜在的攻击图，以供攻击者利用。此外，普罗米修斯还评估了这些攻击路径的可利用性，并通过评分机制量化了总体安全态势。该系统采取整体方法，分析包括硬件、系统、网络和加密在内的安全层。此外，普罗米修斯深入研究了这些层之间的相互联系，探索如何利用一个层中的漏洞来利用其他层中的漏洞。在这篇文章中，我们介绍了在普罗米修斯中实现的端到端管道，展示了为进行这种彻底的安全分析而采用的系统方法。



## **30. LRS: Enhancing Adversarial Transferability through Lipschitz Regularized Surrogate**

LRS：通过Lipschitz正则化代理增强对抗可转移性 cs.LG

AAAI 2024

**SubmitDate**: 2023-12-20    [abs](http://arxiv.org/abs/2312.13118v1) [paper-pdf](http://arxiv.org/pdf/2312.13118v1)

**Authors**: Tao Wu, Tie Luo, Donald C. Wunsch

**Abstract**: The transferability of adversarial examples is of central importance to transfer-based black-box adversarial attacks. Previous works for generating transferable adversarial examples focus on attacking \emph{given} pretrained surrogate models while the connections between surrogate models and adversarial trasferability have been overlooked. In this paper, we propose {\em Lipschitz Regularized Surrogate} (LRS) for transfer-based black-box attacks, a novel approach that transforms surrogate models towards favorable adversarial transferability. Using such transformed surrogate models, any existing transfer-based black-box attack can run without any change, yet achieving much better performance. Specifically, we impose Lipschitz regularization on the loss landscape of surrogate models to enable a smoother and more controlled optimization process for generating more transferable adversarial examples. In addition, this paper also sheds light on the connection between the inner properties of surrogate models and adversarial transferability, where three factors are identified: smaller local Lipschitz constant, smoother loss landscape, and stronger adversarial robustness. We evaluate our proposed LRS approach by attacking state-of-the-art standard deep neural networks and defense models. The results demonstrate significant improvement on the attack success rates and transferability. Our code is available at https://github.com/TrustAIoT/LRS.

摘要: 对抗性例子的可转移性对于基于转移的黑盒对抗性攻击是至关重要的。以往关于生成可传递对抗实例的工作主要集中在攻击预先训练好的代理模型上，而忽略了代理模型与对抗传递能力之间的联系。针对基于转移的黑盒攻击，提出了一种将代理模型转化为有利的对抗性转移的新方法--LRS。使用这种转换的代理模型，任何现有的基于传输的黑盒攻击都可以在不做任何更改的情况下运行，但获得了更好的性能。具体地说，我们将Lipschitz正则化应用于代理模型的损失图景，以实现更平滑和更可控的优化过程，从而生成更多可转移的对抗性例子。此外，本文还揭示了代理模型的内在性质与对抗转移之间的关系，其中确定了三个因素：较小的局部Lipschitz常数、更平滑的损失图景和更强的对抗稳健性。我们通过攻击最先进的标准深度神经网络和防御模型来评估我们提出的LRS方法。结果表明，在攻击成功率和可转移性方面都有显著的提高。我们的代码可以在https://github.com/TrustAIoT/LRS.上找到



## **31. PGN: A perturbation generation network against deep reinforcement learning**

PGN：一种抗深度强化学习的扰动生成网络 cs.LG

**SubmitDate**: 2023-12-20    [abs](http://arxiv.org/abs/2312.12904v1) [paper-pdf](http://arxiv.org/pdf/2312.12904v1)

**Authors**: Xiangjuan Li, Feifan Li, Yang Li, Quan Pan

**Abstract**: Deep reinforcement learning has advanced greatly and applied in many areas. In this paper, we explore the vulnerability of deep reinforcement learning by proposing a novel generative model for creating effective adversarial examples to attack the agent. Our proposed model can achieve both targeted attacks and untargeted attacks. Considering the specificity of deep reinforcement learning, we propose the action consistency ratio as a measure of stealthiness, and a new measurement index of effectiveness and stealthiness. Experiment results show that our method can ensure the effectiveness and stealthiness of attack compared with other algorithms. Moreover, our methods are considerably faster and thus can achieve rapid and efficient verification of the vulnerability of deep reinforcement learning.

摘要: 深度强化学习已经取得了很大的进展，并在许多领域得到了应用。在本文中，我们通过提出一种新的生成模型来探索深度强化学习的脆弱性，该模型用于创建有效的对抗性示例来攻击代理。该模型既可以实现定向攻击，也可以实现非定向攻击。考虑到深度强化学习的特殊性，我们提出了动作一致性比率作为隐蔽性的度量，并提出了一种新的度量有效性和隐蔽性的指标。实验结果表明，与其他算法相比，该方法能够保证攻击的有效性和隐蔽性。此外，我们的方法具有相当快的速度，因此可以快速有效地验证深度强化学习的脆弱性。



## **32. SAAM: Stealthy Adversarial Attack on Monocular Depth Estimation**

SAAM：对单目深度估计的隐形攻击 cs.CV

**SubmitDate**: 2023-12-20    [abs](http://arxiv.org/abs/2308.03108v2) [paper-pdf](http://arxiv.org/pdf/2308.03108v2)

**Authors**: Amira Guesmi, Muhammad Abdullah Hanif, Bassem Ouni, Muhammad Shafique

**Abstract**: In this paper, we investigate the vulnerability of MDE to adversarial patches. We propose a novel \underline{S}tealthy \underline{A}dversarial \underline{A}ttacks on \underline{M}DE (SAAM) that compromises MDE by either corrupting the estimated distance or causing an object to seamlessly blend into its surroundings. Our experiments, demonstrate that the designed stealthy patch successfully causes a DNN-based MDE to misestimate the depth of objects. In fact, our proposed adversarial patch achieves a significant 60\% depth error with 99\% ratio of the affected region. Importantly, despite its adversarial nature, the patch maintains a naturalistic appearance, making it inconspicuous to human observers. We believe that this work sheds light on the threat of adversarial attacks in the context of MDE on edge devices. We hope it raises awareness within the community about the potential real-life harm of such attacks and encourages further research into developing more robust and adaptive defense mechanisms.

摘要: 在本文中，我们研究了MDE对敌意补丁的脆弱性。我们提出了一种新的基于{M}DE(SAAM)的{A}大头针，它破坏了估计的距离或使物体无缝地融入其周围，从而折衷了MDE。我们的实验表明，所设计的隐身补丁成功地导致了基于DNN的MDE错误估计目标的深度。事实上，我们提出的对抗性补丁获得了显著的60%的深度误差，其受影响区域的比率为99%。重要的是，尽管它具有对抗性，但它保持了一种自然主义的外观，使它对人类观察者来说并不引人注目。我们相信，这项工作有助于揭示边缘设备上的MDE环境中的对抗性攻击威胁。我们希望它能提高社区对此类攻击的潜在现实危害的认识，并鼓励进一步研究开发更强大和适应性更强的防御机制。



## **33. Mutual-modality Adversarial Attack with Semantic Perturbation**

具有语义扰动的互通道对抗性攻击 cs.CV

Accepted by AAAI2024

**SubmitDate**: 2023-12-20    [abs](http://arxiv.org/abs/2312.12768v1) [paper-pdf](http://arxiv.org/pdf/2312.12768v1)

**Authors**: Jingwen Ye, Ruonan Yu, Songhua Liu, Xinchao Wang

**Abstract**: Adversarial attacks constitute a notable threat to machine learning systems, given their potential to induce erroneous predictions and classifications. However, within real-world contexts, the essential specifics of the deployed model are frequently treated as a black box, consequently mitigating the vulnerability to such attacks. Thus, enhancing the transferability of the adversarial samples has become a crucial area of research, which heavily relies on selecting appropriate surrogate models. To address this challenge, we propose a novel approach that generates adversarial attacks in a mutual-modality optimization scheme. Our approach is accomplished by leveraging the pre-trained CLIP model. Firstly, we conduct a visual attack on the clean image that causes semantic perturbations on the aligned embedding space with the other textual modality. Then, we apply the corresponding defense on the textual modality by updating the prompts, which forces the re-matching on the perturbed embedding space. Finally, to enhance the attack transferability, we utilize the iterative training strategy on the visual attack and the textual defense, where the two processes optimize from each other. We evaluate our approach on several benchmark datasets and demonstrate that our mutual-modal attack strategy can effectively produce high-transferable attacks, which are stable regardless of the target networks. Our approach outperforms state-of-the-art attack methods and can be readily deployed as a plug-and-play solution.

摘要: 对抗性攻击对机器学习系统构成了显著的威胁，因为它们有可能导致错误的预测和分类。然而，在现实环境中，部署的模型的基本细节经常被视为黑匣子，从而降低了对此类攻击的脆弱性。因此，提高对抗性样本的可转移性已成为一个重要的研究领域，这在很大程度上依赖于选择合适的代理模型。为了应对这一挑战，我们提出了一种新的方法，该方法在交互通道优化方案中产生对抗性攻击。我们的方法是通过利用预先训练的剪辑模型来实现的。首先，我们对干净的图像进行视觉攻击，导致与其他文本通道对齐的嵌入空间上的语义扰动。然后，我们通过更新提示来对语篇情态进行相应的防御，强制在扰动的嵌入空间上进行重新匹配。最后，为了增强攻击的可转移性，我们在视觉攻击和文本防御上采用了迭代训练策略，这两个过程相互优化。我们在几个基准数据集上对我们的方法进行了评估，并证明了我们的互模式攻击策略可以有效地产生高度可转移的攻击，并且无论目标网络是什么，都是稳定的。我们的方法优于最先进的攻击方法，并且可以作为即插即用解决方案轻松部署。



## **34. Trust, but Verify: Robust Image Segmentation using Deep Learning**

信任，但要验证：使用深度学习的稳健图像分割 cs.CV

5 Pages, 8 Figures, conference

**SubmitDate**: 2023-12-19    [abs](http://arxiv.org/abs/2310.16999v3) [paper-pdf](http://arxiv.org/pdf/2310.16999v3)

**Authors**: Fahim Ahmed Zaman, Xiaodong Wu, Weiyu Xu, Milan Sonka, Raghuraman Mudumbai

**Abstract**: We describe a method for verifying the output of a deep neural network for medical image segmentation that is robust to several classes of random as well as worst-case perturbations i.e. adversarial attacks. This method is based on a general approach recently developed by the authors called "Trust, but Verify" wherein an auxiliary verification network produces predictions about certain masked features in the input image using the segmentation as an input. A well-designed auxiliary network will produce high-quality predictions when the input segmentations are accurate, but will produce low-quality predictions when the segmentations are incorrect. Checking the predictions of such a network with the original image allows us to detect bad segmentations. However, to ensure the verification method is truly robust, we need a method for checking the quality of the predictions that does not itself rely on a black-box neural network. Indeed, we show that previous methods for segmentation evaluation that do use deep neural regression networks are vulnerable to false negatives i.e. can inaccurately label bad segmentations as good. We describe the design of a verification network that avoids such vulnerability and present results to demonstrate its robustness compared to previous methods.

摘要: 我们描述了一种用于医学图像分割的深度神经网络输出的验证方法，该方法对几类随机和最坏情况的扰动，即对抗性攻击具有鲁棒性。该方法基于作者最近开发的一种被称为“信任，但验证”的通用方法，其中辅助验证网络使用分割作为输入来产生关于输入图像中的某些被屏蔽特征的预测。一个设计良好的辅助网络在输入分割准确时会产生高质量的预测，但当分割不正确时会产生低质量的预测。用原始图像检查这种网络的预测可以让我们检测到错误的分割。然而，为了确保验证方法真正稳健，我们需要一种方法来检查预测的质量，该方法本身不依赖于黑盒神经网络。事实上，我们表明，以前使用深度神经回归网络的分割评估方法很容易出现假阴性，即可能不准确地将不良分割标记为良好分割。我们描述了一个避免这种漏洞的验证网络的设计，并给出了与以前方法相比的结果来证明它的健壮性。



## **35. ReRoGCRL: Representation-based Robustness in Goal-Conditioned Reinforcement Learning**

ReRoGCRL：目标条件强化学习中基于表示的稳健性 cs.LG

This paper has been accepted in AAAI24  (https://aaai.org/aaai-conference/)

**SubmitDate**: 2023-12-19    [abs](http://arxiv.org/abs/2312.07392v3) [paper-pdf](http://arxiv.org/pdf/2312.07392v3)

**Authors**: Xiangyu Yin, Sihao Wu, Jiaxu Liu, Meng Fang, Xingyu Zhao, Xiaowei Huang, Wenjie Ruan

**Abstract**: While Goal-Conditioned Reinforcement Learning (GCRL) has gained attention, its algorithmic robustness against adversarial perturbations remains unexplored. The attacks and robust representation training methods that are designed for traditional RL become less effective when applied to GCRL. To address this challenge, we first propose the Semi-Contrastive Representation attack, a novel approach inspired by the adversarial contrastive attack. Unlike existing attacks in RL, it only necessitates information from the policy function and can be seamlessly implemented during deployment. Then, to mitigate the vulnerability of existing GCRL algorithms, we introduce Adversarial Representation Tactics, which combines Semi-Contrastive Adversarial Augmentation with Sensitivity-Aware Regularizer to improve the adversarial robustness of the underlying RL agent against various types of perturbations. Extensive experiments validate the superior performance of our attack and defence methods across multiple state-of-the-art GCRL algorithms. Our tool ReRoGCRL is available at https://github.com/TrustAI/ReRoGCRL.

摘要: 虽然目标条件强化学习(GCRL)已经引起了人们的关注，但它对对抗扰动的算法健壮性还没有得到探索。针对传统RL设计的攻击和稳健表示训练方法在应用于GCRL时变得不那么有效。为了应对这一挑战，我们首先提出了半对比表征攻击，这是一种受对抗性对比攻击启发的新方法。与RL中现有的攻击不同，它只需要来自策略功能的信息，并且可以在部署期间无缝实施。然后，为了缓解现有GCRL算法的脆弱性，我们引入了对抗性表示策略，将半对比对抗性增强和敏感度感知正则化相结合，以提高底层RL代理对各种类型扰动的对抗性健壮性。广泛的实验验证了我们的攻击和防御方法在多种最先进的GCRL算法上的卓越性能。我们的工具ReRoGCRL可在https://github.com/TrustAI/ReRoGCRL.上获得



## **36. Trust, But Verify: A Survey of Randomized Smoothing Techniques**

信任，但验证：随机化平滑技术综述 cs.LG

**SubmitDate**: 2023-12-19    [abs](http://arxiv.org/abs/2312.12608v1) [paper-pdf](http://arxiv.org/pdf/2312.12608v1)

**Authors**: Anupriya Kumari, Devansh Bhardwaj, Sukrit Jindal, Sarthak Gupta

**Abstract**: Machine learning models have demonstrated remarkable success across diverse domains but remain vulnerable to adversarial attacks. Empirical defence mechanisms often fall short, as new attacks constantly emerge, rendering existing defences obsolete. A paradigm shift from empirical defences to certification-based defences has been observed in response. Randomized smoothing has emerged as a promising technique among notable advancements. This study reviews the theoretical foundations, empirical effectiveness, and applications of randomized smoothing in verifying machine learning classifiers. We provide an in-depth exploration of the fundamental concepts underlying randomized smoothing, highlighting its theoretical guarantees in certifying robustness against adversarial perturbations. Additionally, we discuss the challenges of existing methodologies and offer insightful perspectives on potential solutions. This paper is novel in its attempt to systemise the existing knowledge in the context of randomized smoothing.

摘要: 机器学习模型已经在不同的领域取得了显著的成功，但仍然容易受到对手的攻击。随着新的攻击不断出现，经验防御机制往往达不到要求，使现有的防御措施过时。作为回应，人们观察到了从经验辩护到基于认证的辩护的范式转变。在显著的进步中，随机平滑技术已经成为一种很有前途的技术。本文综述了随机平滑在机器学习分类器验证中的理论基础、经验有效性和应用。我们深入探讨了随机平滑背后的基本概念，强调了它在证明对对手扰动的稳健性方面的理论保证。此外，我们还讨论了现有方法的挑战，并对潜在的解决方案提供了有洞察力的观点。这篇论文的新颖之处在于，它试图在随机平滑的背景下将现有知识系统化。



## **37. You Cannot Escape Me: Detecting Evasions of SIEM Rules in Enterprise Networks**

你不能逃避我：在企业网络中检测对SIEM规则的逃避 cs.CR

To be published in Proceedings of the 33rd USENIX Security Symposium  (USENIX Security 2024)

**SubmitDate**: 2023-12-19    [abs](http://arxiv.org/abs/2311.10197v2) [paper-pdf](http://arxiv.org/pdf/2311.10197v2)

**Authors**: Rafael Uetz, Marco Herzog, Louis Hackländer, Simon Schwarz, Martin Henze

**Abstract**: Cyberattacks have grown into a major risk for organizations, with common consequences being data theft, sabotage, and extortion. Since preventive measures do not suffice to repel attacks, timely detection of successful intruders is crucial to stop them from reaching their final goals. For this purpose, many organizations utilize Security Information and Event Management (SIEM) systems to centrally collect security-related events and scan them for attack indicators using expert-written detection rules. However, as we show by analyzing a set of widespread SIEM detection rules, adversaries can evade almost half of them easily, allowing them to perform common malicious actions within an enterprise network without being detected. To remedy these critical detection blind spots, we propose the idea of adaptive misuse detection, which utilizes machine learning to compare incoming events to SIEM rules on the one hand and known-benign events on the other hand to discover successful evasions. Based on this idea, we present AMIDES, an open-source proof-of-concept adaptive misuse detection system. Using four weeks of SIEM events from a large enterprise network and more than 500 hand-crafted evasions, we show that AMIDES successfully detects a majority of these evasions without any false alerts. In addition, AMIDES eases alert analysis by assessing which rules were evaded. Its computational efficiency qualifies AMIDES for real-world operation and hence enables organizations to significantly reduce detection blind spots with moderate effort.

摘要: 网络攻击已经成为组织的主要风险，常见的后果是数据被盗、破坏和敲诈勒索。由于预防措施不足以击退攻击，及时发现成功的入侵者对于阻止他们实现最终目标至关重要。为此，许多组织利用安全信息和事件管理(SIEM)系统集中收集与安全相关的事件，并使用专家编写的检测规则扫描它们的攻击指标。然而，正如我们通过分析一组广泛使用的SIEM检测规则所表明的那样，攻击者可以很容易地规避几乎一半的规则，使他们能够在不被检测到的情况下在企业网络内执行常见的恶意操作。为了弥补这些关键的检测盲点，我们提出了自适应误用检测的思想，一方面利用机器学习将传入事件与SIEM规则进行比较，另一方面利用已知良性事件来发现成功的规避。基于这一思想，我们提出了一个开源的概念验证自适应误用检测系统AMIDES。使用来自大型企业网络的四周SIEM事件和500多个手工创建的规避，我们表明AMIDES成功检测到了大多数此类规避，而没有任何错误警报。此外，AMIDES通过评估哪些规则被规避来简化警报分析。它的计算效率使AMADS有资格在现实世界中运行，因此使组织能够以适度的努力显著减少检测盲点。



## **38. Comparing the robustness of modern no-reference image- and video-quality metrics to adversarial attacks**

比较现代无参考图像和视频质量度量对对抗性攻击的鲁棒性 cs.CV

**SubmitDate**: 2023-12-19    [abs](http://arxiv.org/abs/2310.06958v2) [paper-pdf](http://arxiv.org/pdf/2310.06958v2)

**Authors**: Anastasia Antsiferova, Khaled Abud, Aleksandr Gushchin, Ekaterina Shumitskaya, Sergey Lavrushkin, Dmitriy Vatolin

**Abstract**: Nowadays neural-network-based image- and video-quality metrics show better performance compared to traditional methods. However, they also became more vulnerable to adversarial attacks that increase metrics' scores without improving visual quality. The existing benchmarks of quality metrics compare their performance in terms of correlation with subjective quality and calculation time. However, the adversarial robustness of image-quality metrics is also an area worth researching. In this paper, we analyse modern metrics' robustness to different adversarial attacks. We adopted adversarial attacks from computer vision tasks and compared attacks' efficiency against 15 no-reference image/video-quality metrics. Some metrics showed high resistance to adversarial attacks which makes their usage in benchmarks safer than vulnerable metrics. The benchmark accepts new metrics submissions for researchers who want to make their metrics more robust to attacks or to find such metrics for their needs. Try our benchmark using pip install robustness-benchmark.

摘要: 如今，基于神经网络的图像和视频质量度量显示出比传统方法更好的性能。然而，它们也变得更容易受到对抗性攻击，这些攻击增加了指标的分数，但没有改善视觉质量。现有的质量指标基准在与主观质量和计算时间的相关性方面比较它们的表现。然而，图像质量指标的对抗稳健性也是一个值得研究的领域。在本文中，我们分析了现代度量对不同对手攻击的稳健性。我们采用了来自计算机视觉任务的对抗性攻击，并将攻击效率与15个无参考图像/视频质量指标进行了比较。一些指标表现出对敌意攻击的高度抵抗力，这使得它们在基准中的使用比易受攻击的指标更安全。该基准接受新的指标提交给希望使其指标更具抗攻击能力或找到符合其需求的此类指标的研究人员。使用pip安装健壮性基准测试我们的基准测试。



## **39. Tensor Train Decomposition for Adversarial Attacks on Computer Vision Models**

计算机视觉模型对抗性攻击的张量训练分解 math.NA

**SubmitDate**: 2023-12-19    [abs](http://arxiv.org/abs/2312.12556v1) [paper-pdf](http://arxiv.org/pdf/2312.12556v1)

**Authors**: Andrei Chertkov, Ivan Oseledets

**Abstract**: Deep neural networks (DNNs) are widely used today, but they are vulnerable to adversarial attacks. To develop effective methods of defense, it is important to understand the potential weak spots of DNNs. Often attacks are organized taking into account the architecture of models (white-box approach) and based on gradient methods, but for real-world DNNs this approach in most cases is impossible. At the same time, several gradient-free optimization algorithms are used to attack black-box models. However, classical methods are often ineffective in the multidimensional case. To organize black-box attacks for computer vision models, in this work, we propose the use of an optimizer based on the low-rank tensor train (TT) format, which has gained popularity in various practical multidimensional applications in recent years. Combined with the attribution of the target image, which is built by the auxiliary (white-box) model, the TT-based optimization method makes it possible to organize an effective black-box attack by small perturbation of pixels in the target image. The superiority of the proposed approach over three popular baselines is demonstrated for five modern DNNs on the ImageNet dataset.

摘要: 深度神经网络(DNN)在当今得到了广泛的应用，但它们容易受到对手的攻击。为了开发有效的防御方法，了解DNNS的潜在弱点是重要的。通常，攻击是根据模型的体系结构(白盒方法)和基于梯度方法来组织的，但对于现实世界的DNN来说，这种方法在大多数情况下是不可能的。同时，利用几种无梯度优化算法对黑盒模型进行了攻击。然而，经典的方法在多维情况下往往是无效的。为了组织计算机视觉模型中的黑盒攻击，在这项工作中，我们提出了一种基于低阶张量训练(TT)格式的优化器，该优化器近年来在各种实际的多维应用中得到了普及。基于TT的优化方法结合辅助(白盒)模型建立的目标图像属性，通过对目标图像像素的微小扰动来组织有效的黑盒攻击。在ImageNet数据集上对五个现代DNN进行了测试，结果表明该方法优于三种流行的基线。



## **40. Counter-Empirical Attacking based on Adversarial Reinforcement Learning for Time-Relevant Scoring System**

基于对抗性强化学习的时间相关评分系统的反经验攻击 cs.LG

Accepted by TKDE

**SubmitDate**: 2023-12-19    [abs](http://arxiv.org/abs/2311.05144v2) [paper-pdf](http://arxiv.org/pdf/2311.05144v2)

**Authors**: Xiangguo Sun, Hong Cheng, Hang Dong, Bo Qiao, Si Qin, Qingwei Lin

**Abstract**: Scoring systems are commonly seen for platforms in the era of big data. From credit scoring systems in financial services to membership scores in E-commerce shopping platforms, platform managers use such systems to guide users towards the encouraged activity pattern, and manage resources more effectively and more efficiently thereby. To establish such scoring systems, several "empirical criteria" are firstly determined, followed by dedicated top-down design for each factor of the score, which usually requires enormous effort to adjust and tune the scoring function in the new application scenario. What's worse, many fresh projects usually have no ground-truth or any experience to evaluate a reasonable scoring system, making the designing even harder. To reduce the effort of manual adjustment of the scoring function in every new scoring system, we innovatively study the scoring system from the preset empirical criteria without any ground truth, and propose a novel framework to improve the system from scratch. In this paper, we propose a "counter-empirical attacking" mechanism that can generate "attacking" behavior traces and try to break the empirical rules of the scoring system. Then an adversarial "enhancer" is applied to evaluate the scoring system and find the improvement strategy. By training the adversarial learning problem, a proper scoring function can be learned to be robust to the attacking activity traces that are trying to violate the empirical criteria. Extensive experiments have been conducted on two scoring systems including a shared computing resource platform and a financial credit system. The experimental results have validated the effectiveness of our proposed framework.

摘要: 在大数据时代，评分系统在平台上很常见。从金融服务的信用评分系统到电子商务购物平台的会员评分，平台管理者使用这些系统来引导用户走向鼓励活动模式，从而更有效和更高效地管理资源。要建立这样的评分系统，首先要确定几个“经验标准”，然后针对评分的每个因素进行专门的自上而下的设计，这通常需要付出巨大的努力来调整和调整新的应用场景中的评分函数。更糟糕的是，许多新项目通常没有实际情况或任何经验来评估合理的评分系统，这使得设计变得更加困难。为了减少每个新评分系统中人工调整评分函数的工作量，我们创新性地从预设的经验标准出发，在没有任何基础事实的情况下对评分系统进行研究，并提出了一个新的框架来从头开始改进该系统。在本文中，我们提出了一种可以产生攻击行为痕迹的反经验攻击机制，试图打破评分系统的经验规则。然后应用一个对抗性的“增强器”对评分系统进行评估，并找到改进策略。通过训练对抗性学习问题，可以学习适当的得分函数，以对试图违反经验标准的攻击活动痕迹具有健壮性。在包括共享计算资源平台和金融信用系统在内的两个评分系统上进行了广泛的实验。实验结果验证了该框架的有效性。



## **41. Position Bias Mitigation: A Knowledge-Aware Graph Model for Emotion Cause Extraction**

位置偏差缓解：一种用于情感原因提取的知识感知图模型 cs.CL

ACL2021 Main Conference, Oral paper

**SubmitDate**: 2023-12-19    [abs](http://arxiv.org/abs/2106.03518v3) [paper-pdf](http://arxiv.org/pdf/2106.03518v3)

**Authors**: Hanqi Yan, Lin Gui, Gabriele Pergola, Yulan He

**Abstract**: The Emotion Cause Extraction (ECE)} task aims to identify clauses which contain emotion-evoking information for a particular emotion expressed in text. We observe that a widely-used ECE dataset exhibits a bias that the majority of annotated cause clauses are either directly before their associated emotion clauses or are the emotion clauses themselves. Existing models for ECE tend to explore such relative position information and suffer from the dataset bias. To investigate the degree of reliance of existing ECE models on clause relative positions, we propose a novel strategy to generate adversarial examples in which the relative position information is no longer the indicative feature of cause clauses. We test the performance of existing models on such adversarial examples and observe a significant performance drop. To address the dataset bias, we propose a novel graph-based method to explicitly model the emotion triggering paths by leveraging the commonsense knowledge to enhance the semantic dependencies between a candidate clause and an emotion clause. Experimental results show that our proposed approach performs on par with the existing state-of-the-art methods on the original ECE dataset, and is more robust against adversarial attacks compared to existing models.

摘要: 情感原因提取(ECES)任务旨在识别包含文本中表达的特定情感的情感唤起信息的从句。我们观察到，一个广泛使用的欧洲经委会数据集显示出一种偏见，即大多数带注释的原因从句要么直接位于其关联的情感从句之前，要么是情感从句本身。欧洲经委会的现有模型倾向于探索这种相对位置信息，并受到数据集偏差的影响。为了考察现有的ECA模型对小句相对位置的依赖程度，我们提出了一种新的策略来生成对抗性实例，其中相对位置信息不再是原因从句的指示性特征。我们在这样的对抗性例子上测试了现有模型的性能，并观察到性能显著下降。为了解决数据集偏向的问题，我们提出了一种新的基于图的方法，通过利用常识知识来增强候选子句和情感子句之间的语义依赖，来显式建模情感触发路径。实验结果表明，我们提出的方法在原始ECA数据集上的性能与现有的最新方法相当，并且与现有的模型相比，对敌意攻击具有更好的鲁棒性。



## **42. QuanShield: Protecting against Side-Channels Attacks using Self-Destructing Enclaves**

QuanShield：使用自毁飞地防御旁路攻击 cs.CR

15pages, 5 figures, 5 tables

**SubmitDate**: 2023-12-19    [abs](http://arxiv.org/abs/2312.11796v1) [paper-pdf](http://arxiv.org/pdf/2312.11796v1)

**Authors**: Shujie Cui, Haohua Li, Yuanhong Li, Zhi Zhang, Lluís Vilanova, Peter Pietzuch

**Abstract**: Trusted Execution Environments (TEEs) allow user processes to create enclaves that protect security-sensitive computation against access from the OS kernel and the hypervisor. Recent work has shown that TEEs are vulnerable to side-channel attacks that allow an adversary to learn secrets shielded in enclaves. The majority of such attacks trigger exceptions or interrupts to trace the control or data flow of enclave execution.   We propose QuanShield, a system that protects enclaves from side-channel attacks that interrupt enclave execution. The main idea behind QuanShield is to strengthen resource isolation by creating an interrupt-free environment on a dedicated CPU core for running enclaves in which enclaves terminate when interrupts occur. QuanShield avoids interrupts by exploiting the tickless scheduling mode supported by recent OS kernels. QuanShield then uses the save area (SA) of the enclave, which is used by the hardware to support interrupt handling, as a second stack. Through an LLVM-based compiler pass, QuanShield modifies enclave instructions to store/load memory references, such as function frame base addresses, to/from the SA. When an interrupt occurs, the hardware overwrites the data in the SA with CPU state, thus ensuring that enclave execution fails. Our evaluation shows that QuanShield significantly raises the bar for interrupt-based attacks with practical overhead.

摘要: 可信执行环境(TEE)允许用户进程创建飞地，以保护安全敏感的计算免受操作系统内核和管理程序的访问。最近的研究表明，T恤很容易受到旁路攻击，这些攻击允许对手学习被屏蔽在飞地上的秘密。大多数此类攻击都会触发异常或中断，以跟踪Enclave执行的控制或数据流。我们提出了QuanShield，这是一个保护飞地免受旁路攻击的系统，这些攻击会中断飞地的执行。QuanShield背后的主要思想是通过在专用CPU核心上创建一个无中断环境来加强资源隔离，以用于运行Enclaves，在这些Enclaves中，Enclave在发生中断时终止。QuanShield通过利用最新操作系统内核支持的无计时调度模式来避免中断。QuanShield然后使用Enclave的保存区(SA)作为第二堆栈，硬件使用该保存区来支持中断处理。通过基于LLVM的编译器通道，QuanShield修改Enclave指令以将内存引用(如函数帧基址)存储到SA或从SA加载内存引用。当中断发生时，硬件用CPU状态覆盖SA中的数据，从而确保Enclave执行失败。我们的评估表明，QuanShield显著提高了基于中断的攻击的门槛和实际开销。



## **43. Impartial Games: A Challenge for Reinforcement Learning**

公平博弈：强化学习的挑战 cs.LG

**SubmitDate**: 2023-12-18    [abs](http://arxiv.org/abs/2205.12787v3) [paper-pdf](http://arxiv.org/pdf/2205.12787v3)

**Authors**: Bei Zhou, Søren Riis

**Abstract**: While AlphaZero-style reinforcement learning (RL) algorithms excel in various board games, in this paper we show that they face challenges on impartial games where players share pieces. We present a concrete example of a game - namely the children's game of Nim - and other impartial games that seem to be a stumbling block for AlphaZero-style and similar self-play reinforcement learning algorithms.   Our work is built on the challenges posed by the intricacies of data distribution on the ability of neural networks to learn parity functions, exacerbated by the noisy labels issue. Our findings are consistent with recent studies showing that AlphaZero-style algorithms are vulnerable to adversarial attacks and adversarial perturbations, showing the difficulty of learning to master the games in all legal states.   We show that Nim can be learned on small boards, but the learning progress of AlphaZero-style algorithms dramatically slows down when the board size increases. Intuitively, the difference between impartial games like Nim and partisan games like Chess and Go can be explained by the fact that if a small part of the board is covered for impartial games it is typically not possible to predict whether the position is won or lost as there is often zero correlation between the visible part of a partly blanked-out position and its correct evaluation. This situation starkly contrasts partisan games where a partly blanked-out board position typically provides abundant or at least non-trifle information about the value of the fully uncovered position.

摘要: 虽然AlphaZero风格的强化学习(RL)算法在各种棋类游戏中表现出色，但在本文中，我们展示了它们在玩家共享棋子的公平游戏中面临的挑战。我们提供了一个具体的游戏示例--即Nim的儿童游戏--以及其他公平的游戏，这些游戏似乎是AlphaZero风格和类似的自我发挥强化学习算法的绊脚石。我们的工作建立在错综复杂的数据分布对神经网络学习奇偶函数能力构成的挑战上，噪声标签问题加剧了这一挑战。我们的发现与最近的研究一致，这些研究表明AlphaZero风格的算法容易受到对手攻击和对手扰动，这表明在所有合法国家学习掌握游戏都是困难的。我们表明，NIM可以在小电路板上学习，但AlphaZero风格的算法的学习进度随着电路板大小的增加而显著减慢。直觉上，像尼姆这样的公正游戏与像国际象棋和围棋这样的党派游戏之间的区别可以用这样一个事实来解释：如果棋盘上的一小部分被公平地覆盖，通常不可能预测位置是赢是输，因为部分空白的位置的可见部分与其正确评估之间往往没有相关性。这种情况与党派游戏形成鲜明对比，在党派游戏中，部分空白的董事会职位通常会提供大量或至少不是无关紧要的信息，以了解完全暴露的职位的价值。



## **44. The Ultimate Combo: Boosting Adversarial Example Transferability by Composing Data Augmentations**

终极组合：通过组合数据增强来提高对抗性示例的可移植性 cs.CV

18 pages

**SubmitDate**: 2023-12-18    [abs](http://arxiv.org/abs/2312.11309v1) [paper-pdf](http://arxiv.org/pdf/2312.11309v1)

**Authors**: Zebin Yun, Achi-Or Weingarten, Eyal Ronen, Mahmood Sharif

**Abstract**: Transferring adversarial examples (AEs) from surrogate machine-learning (ML) models to target models is commonly used in black-box adversarial robustness evaluation. Attacks leveraging certain data augmentation, such as random resizing, have been found to help AEs generalize from surrogates to targets. Yet, prior work has explored limited augmentations and their composition. To fill the gap, we systematically studied how data augmentation affects transferability. Particularly, we explored 46 augmentation techniques of seven categories originally proposed to help ML models generalize to unseen benign samples, and assessed how they impact transferability, when applied individually or composed. Performing exhaustive search on a small subset of augmentation techniques and genetic search on all techniques, we identified augmentation combinations that can help promote transferability. Extensive experiments with the ImageNet and CIFAR-10 datasets and 18 models showed that simple color-space augmentations (e.g., color to greyscale) outperform the state of the art when combined with standard augmentations, such as translation and scaling. Additionally, we discovered that composing augmentations impacts transferability mostly monotonically (i.e., more methods composed $\rightarrow$ $\ge$ transferability). We also found that the best composition significantly outperformed the state of the art (e.g., 93.7% vs. $\le$ 82.7% average transferability on ImageNet from normally trained surrogates to adversarially trained targets). Lastly, our theoretical analysis, backed up by empirical evidence, intuitively explain why certain augmentations help improve transferability.

摘要: 在黑盒对抗健壮性评估中，经常使用从代理机器学习(ML)模型到目标模型的对抗性实例(AE)的转换。利用某些数据增强的攻击，如随机调整大小，已被发现有助于AE从代理扩展到目标。然而，先前的工作探索了有限的增强及其组成。为了填补这一空白，我们系统地研究了数据扩充如何影响可转移性。特别是，我们探索了最初提出的七个类别的46种增强技术，以帮助ML模型推广到看不见的良性样本，并评估了当单独应用或组合应用时，它们对可转移性的影响。我们对一小部分增强技术进行了穷举搜索，并对所有技术进行了遗传搜索，确定了有助于提高可转移性的增强组合。使用ImageNet和CIFAR-10数据集和18个模型进行的广泛实验表明，当与标准增强(如平移和缩放)相结合时，简单的颜色空间增强(例如，从颜色到灰度)的性能优于最先进的增强。此外，我们还发现，组合扩充对可转移性的影响主要是单调的(即，更多的方法组合了$\right tarrow$$\ge$t可转移性)。我们还发现，最好的成分远远超过了最先进的水平(例如，在ImageNet上，93.7%对82.7%的平均可转移性从正常训练的代理人转移到对抗性训练的目标)。最后，我们的理论分析得到了经验证据的支持，直观地解释了为什么某些增强有助于提高可转移性。



## **45. Adv-Diffusion: Imperceptible Adversarial Face Identity Attack via Latent Diffusion Model**

ADV扩散：基于潜在扩散模型的隐形对抗性人脸身份攻击 cs.CV

**SubmitDate**: 2023-12-18    [abs](http://arxiv.org/abs/2312.11285v1) [paper-pdf](http://arxiv.org/pdf/2312.11285v1)

**Authors**: Decheng Liu, Xijun Wang, Chunlei Peng, Nannan Wang, Ruiming Hu, Xinbo Gao

**Abstract**: Adversarial attacks involve adding perturbations to the source image to cause misclassification by the target model, which demonstrates the potential of attacking face recognition models. Existing adversarial face image generation methods still can't achieve satisfactory performance because of low transferability and high detectability. In this paper, we propose a unified framework Adv-Diffusion that can generate imperceptible adversarial identity perturbations in the latent space but not the raw pixel space, which utilizes strong inpainting capabilities of the latent diffusion model to generate realistic adversarial images. Specifically, we propose the identity-sensitive conditioned diffusion generative model to generate semantic perturbations in the surroundings. The designed adaptive strength-based adversarial perturbation algorithm can ensure both attack transferability and stealthiness. Extensive qualitative and quantitative experiments on the public FFHQ and CelebA-HQ datasets prove the proposed method achieves superior performance compared with the state-of-the-art methods without an extra generative model training process. The source code is available at https://github.com/kopper-xdu/Adv-Diffusion.

摘要: 对抗性攻击涉及向源图像添加扰动以引起目标模型的错误分类，这展示了攻击人脸识别模型的潜力。现有的对抗性人脸图像生成方法由于可移植性低、可检测性高等原因，仍然不能取得令人满意的效果。在本文中，我们提出了一个统一的框架Adv-Diffusion，它可以在潜在空间而不是原始像素空间中生成不可感知的对抗身份扰动，它利用潜在扩散模型的强大修复能力来生成逼真的对抗图像。具体来说，我们提出了身份敏感的条件扩散生成模型，以产生语义扰动的环境。所设计的自适应强度对抗扰动算法能够同时保证攻击的可转移性和隐蔽性。在公共FFHQ和CelebA-HQ数据集上进行的大量定性和定量实验证明，与最先进的方法相比，所提出的方法具有更好的性能，无需额外的生成模型训练过程。源代码可在https://github.com/kopper-xdu/Adv-Diffusion上获得。



## **46. Protect Your Score: Contact Tracing With Differential Privacy Guarantees**

保护您的分数：具有差异隐私保证的联系人跟踪 cs.CR

Accepted to The 38th Annual AAAI Conference on Artificial  Intelligence (AAAI 2024)

**SubmitDate**: 2023-12-18    [abs](http://arxiv.org/abs/2312.11581v1) [paper-pdf](http://arxiv.org/pdf/2312.11581v1)

**Authors**: Rob Romijnders, Christos Louizos, Yuki M. Asano, Max Welling

**Abstract**: The pandemic in 2020 and 2021 had enormous economic and societal consequences, and studies show that contact tracing algorithms can be key in the early containment of the virus. While large strides have been made towards more effective contact tracing algorithms, we argue that privacy concerns currently hold deployment back. The essence of a contact tracing algorithm constitutes the communication of a risk score. Yet, it is precisely the communication and release of this score to a user that an adversary can leverage to gauge the private health status of an individual. We pinpoint a realistic attack scenario and propose a contact tracing algorithm with differential privacy guarantees against this attack. The algorithm is tested on the two most widely used agent-based COVID19 simulators and demonstrates superior performance in a wide range of settings. Especially for realistic test scenarios and while releasing each risk score with epsilon=1 differential privacy, we achieve a two to ten-fold reduction in the infection rate of the virus. To the best of our knowledge, this presents the first contact tracing algorithm with differential privacy guarantees when revealing risk scores for COVID19.

摘要: 2020年和2021年的大流行造成了巨大的经济和社会后果，研究表明，接触者追踪算法可能是早期遏制病毒的关键。虽然在更有效的联系人跟踪算法方面取得了长足的进步，但我们认为，隐私问题目前阻碍了部署。接触追踪算法的本质是传达风险评分。然而，正是向用户传达和发布该分数，对手可以利用该分数来衡量个人的私人健康状况。我们指出了一个现实的攻击场景，并提出了一个接触跟踪算法与差分隐私保证这种攻击。该算法在两个最广泛使用的基于代理的COVID 19模拟器上进行了测试，并在广泛的设置中表现出卓越的性能。特别是对于现实的测试场景，在释放每个风险评分时，我们将病毒的感染率降低了2到10倍。据我们所知，这是第一个在揭示COVID 19风险评分时具有差异隐私保证的接触追踪算法。



## **47. A Survey of Side-Channel Attacks in Context of Cache -- Taxonomies, Analysis and Mitigation**

缓存环境下的旁路攻击研究综述--分类、分析与防御 cs.CR

**SubmitDate**: 2023-12-18    [abs](http://arxiv.org/abs/2312.11094v1) [paper-pdf](http://arxiv.org/pdf/2312.11094v1)

**Authors**: Ankit Pulkit, Smita Naval, Vijay Laxmi

**Abstract**: Side-channel attacks have become prominent attack surfaces in cyberspace. Attackers use the side information generated by the system while performing a task. Among the various side-channel attacks, cache side-channel attacks are leading as there has been an enormous growth in cache memory size in last decade, especially Last Level Cache (LLC). The adversary infers the information from the observable behavior of shared cache memory. This paper covers the detailed study of cache side-channel attacks and compares different microarchitectures in the context of side-channel attacks. Our main contributions are: (1) We have summarized the fundamentals and essentials of side-channel attacks and various attack surfaces (taxonomies). We also discussed different exploitation techniques, highlighting their capabilities and limitations. (2) We discussed cache side-channel attacks and analyzed the existing literature on cache side-channel attacks on various parameters like microarchitectures, cross-core exploitation, methodology, target, etc. (3) We discussed the detailed analysis of the existing mitigation strategies to prevent cache side-channel attacks. The analysis includes hardware- and software-based countermeasures, examining their strengths and weaknesses. We also discussed the challenges and trade-offs associated with mitigation strategies. This survey is supposed to provide a deeper understanding of the threats posed by these attacks to the research community with valuable insights into effective defense mechanisms.

摘要: 旁路攻击已成为网络空间的主要攻击面。攻击者在执行任务时使用系统生成的辅助信息。在各种侧通道攻击中，缓存侧通道攻击是主要的，因为在过去的十年中，高速缓存的大小有了巨大的增长，特别是最后一级高速缓存(LLC)。攻击者从共享高速缓冲存储器的可观察行为中推断信息。本文详细研究了缓存侧通道攻击，并比较了不同微体系结构在侧通道攻击环境中的应用。我们的主要贡献是：(1)总结了旁路攻击的基本原理和本质，以及各种攻击面(分类)。我们还讨论了不同的开发技术，强调了它们的功能和局限性。(2)讨论了缓存侧通道攻击，分析了现有的缓存侧通道攻击的研究文献，包括微体系结构、跨核利用、方法、攻击目标等。(3)详细分析了现有的缓存侧通道攻击的缓解策略。分析包括基于硬件和软件的对策，检查它们的优势和劣势。我们还讨论了与缓解策略相关的挑战和权衡。这项调查旨在更深入地了解这些攻击对研究界构成的威胁，为有效的防御机制提供有价值的见解。



## **48. Forbidden Facts: An Investigation of Competing Objectives in Llama-2**

禁忌事实：骆驼2号中相互竞争的目标的调查 cs.LG

Accepted to the ATTRIB and SoLaR workshops at NeurIPS 2023; (v2:  fixed typos)

**SubmitDate**: 2023-12-18    [abs](http://arxiv.org/abs/2312.08793v2) [paper-pdf](http://arxiv.org/pdf/2312.08793v2)

**Authors**: Tony T. Wang, Miles Wang, Kaivalya Hariharan, Nir Shavit

**Abstract**: LLMs often face competing pressures (for example helpfulness vs. harmlessness). To understand how models resolve such conflicts, we study Llama-2-chat models on the forbidden fact task. Specifically, we instruct Llama-2 to truthfully complete a factual recall statement while forbidding it from saying the correct answer. This often makes the model give incorrect answers. We decompose Llama-2 into 1000+ components, and rank each one with respect to how useful it is for forbidding the correct answer. We find that in aggregate, around 35 components are enough to reliably implement the full suppression behavior. However, these components are fairly heterogeneous and many operate using faulty heuristics. We discover that one of these heuristics can be exploited via a manually designed adversarial attack which we call The California Attack. Our results highlight some roadblocks standing in the way of being able to successfully interpret advanced ML systems. Project website available at https://forbiddenfacts.github.io .

摘要: 低收入国家经常面临相互竞争的压力(例如，有益与无害)。为了理解模型如何解决此类冲突，我们研究了关于禁止事实任务的Llama-2-Chat模型。具体地说，我们指示骆驼2号如实完成事实回忆声明，同时禁止它说出正确的答案。这经常使模型给出错误的答案。我们将Llama-2分解成1000多个成分，并根据它们对阻止正确答案的作用程度对每个成分进行排名。我们发现，总共大约35个组件就足以可靠地实现完全抑制行为。然而，这些组件具有相当大的异构性，许多组件使用错误的启发式方法进行操作。我们发现，其中一个启发式攻击可以通过手动设计的对抗性攻击来利用，我们称之为加利福尼亚州攻击。我们的结果突出了一些阻碍成功解释高级ML系统的障碍。项目网站为https://forbiddenfacts.github.io。



## **49. No-Skim: Towards Efficiency Robustness Evaluation on Skimming-based Language Models**

No-Skim：基于略读的语言模型的效率稳健性评价 cs.CR

**SubmitDate**: 2023-12-18    [abs](http://arxiv.org/abs/2312.09494v2) [paper-pdf](http://arxiv.org/pdf/2312.09494v2)

**Authors**: Shengyao Zhang, Mi Zhang, Xudong Pan, Min Yang

**Abstract**: To reduce the computation cost and the energy consumption in large language models (LLM), skimming-based acceleration dynamically drops unimportant tokens of the input sequence progressively along layers of the LLM while preserving the tokens of semantic importance. However, our work for the first time reveals the acceleration may be vulnerable to Denial-of-Service (DoS) attacks. In this paper, we propose No-Skim, a general framework to help the owners of skimming-based LLM to understand and measure the robustness of their acceleration scheme. Specifically, our framework searches minimal and unnoticeable perturbations at character-level and token-level to generate adversarial inputs that sufficiently increase the remaining token ratio, thus increasing the computation cost and energy consumption. We systematically evaluate the vulnerability of the skimming acceleration in various LLM architectures including BERT and RoBERTa on the GLUE benchmark. In the worst case, the perturbation found by No-Skim substantially increases the running cost of LLM by over 145% on average. Moreover, No-Skim extends the evaluation framework to various scenarios, making the evaluation conductible with different level of knowledge.

摘要: 为了降低大型语言模型（LLM）的计算成本和能耗，基于略读的加速动态地沿着LLM的层逐渐丢弃输入序列的不重要的标记，同时保留语义重要性的标记。然而，我们的工作首次揭示了加速可能容易受到拒绝服务（DoS）攻击。在本文中，我们提出了No-Skim，这是一个通用框架，可以帮助基于略读的LLM的所有者了解和衡量其加速方案的鲁棒性。具体来说，我们的框架在字符级和令牌级搜索最小且不明显的扰动，以生成足以增加剩余令牌比率的对抗性输入，从而增加计算成本和能耗。我们系统地评估了各种LLM架构（包括BERT和RoBERTA）在GLUE基准测试中的略读加速漏洞。在最坏的情况下，由No-Skim发现的扰动显著地增加了LLM的运行成本，平均超过145%。此外，No-Skim将评估框架扩展到各种场景，使评估具有不同的知识水平。



## **50. Security Defense of Large Scale Networks Under False Data Injection Attacks: An Attack Detection Scheduling Approach**

虚假数据注入攻击下的大规模网络安全防御：一种攻击检测调度方法 eess.SY

14 pages, 13 figures

**SubmitDate**: 2023-12-18    [abs](http://arxiv.org/abs/2212.05500v4) [paper-pdf](http://arxiv.org/pdf/2212.05500v4)

**Authors**: Yuhan Suo, Senchun Chai, Runqi Chai, Zhong-Hua Pang, Yuanqing Xia, Guo-Ping Liu

**Abstract**: In large-scale networks, communication links between nodes are easily injected with false data by adversaries. This paper proposes a novel security defense strategy from the perspective of attack detection scheduling to ensure the security of the network. Based on the proposed strategy, each sensor can directly exclude suspicious sensors from its neighboring set. First, the problem of selecting suspicious sensors is formulated as a combinatorial optimization problem, which is non-deterministic polynomial-time hard (NP-hard). To solve this problem, the original function is transformed into a submodular function. Then, we propose an attack detection scheduling algorithm based on the sequential submodular optimization theory, which incorporates \emph{expert problem} to better utilize historical information to guide the sensor selection task at the current moment. For different attack strategies, theoretical results show that the average optimization rate of the proposed algorithm has a lower bound, and the error expectation is bounded. In addition, under two kinds of insecurity conditions, the proposed algorithm can guarantee the security of the entire network from the perspective of the augmented estimation error. Finally, the effectiveness of the developed method is verified by the numerical simulation and practical experiment.

摘要: 在大规模网络中，节点之间的通信链路很容易被对手注入虚假数据。本文从攻击检测调度的角度提出了一种新的安全防御策略，以确保网络的安全。基于该策略，每个传感器可以直接从其相邻集合中排除可疑传感器。首先，将可疑传感器的选择问题描述为一个非确定多项式时间难(NP-Hard)的组合优化问题。为了解决这一问题，将原函数转化为子模函数。在此基础上，提出了一种基于序贯子模优化理论的攻击检测调度算法，该算法结合专家问题，更好地利用历史信息指导当前时刻的传感器选择任务。理论结果表明，对于不同的攻击策略，该算法的平均优化率有一个下界，误差期望是有界的。另外，在两种不安全情况下，从估计误差增大的角度来看，该算法可以保证整个网络的安全性。最后，通过数值模拟和实际实验验证了该方法的有效性。



