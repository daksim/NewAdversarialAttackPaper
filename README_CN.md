# Latest Adversarial Attack Papers
**update at 2025-02-19 09:55:41**

翻译来自 https://cloud.tencent.com/document/product/551/15619

## **1. Quantum Byzantine Multiple Access Channels**

量子拜占庭多址通道 cs.IT

**SubmitDate**: 2025-02-17    [abs](http://arxiv.org/abs/2502.12047v1) [paper-pdf](http://arxiv.org/pdf/2502.12047v1)

**Authors**: Minglai Cai, Christian Deppe

**Abstract**: In communication theory, attacks like eavesdropping or jamming are typically assumed to occur at the channel level, while communication parties are expected to follow established protocols. But what happens if one of the parties turns malicious? In this work, we investigate a compelling scenario: a multiple-access channel with two transmitters and one receiver, where one transmitter deviates from the protocol and acts dishonestly. To address this challenge, we introduce the Byzantine multiple-access classical-quantum channel and derive an achievable communication rate for this adversarial setting.

摘要: 在通信理论中，窃听或干扰等攻击通常被假设发生在通道级别，而通信方预计会遵循既定的协议。但如果其中一方变得恶意会发生什么？在这项工作中，我们研究了一个引人注目的场景：具有两个发射机和一个接收机的多址通道，其中一个发射机偏离了协议并行为不诚实。为了应对这一挑战，我们引入了拜占庭式多址经典量子通道，并推导出针对这种对抗环境的可实现的通信速率。



## **2. FedEAT: A Robustness Optimization Framework for Federated LLMs**

FedEAT：联邦LLM的稳健性优化框架 cs.LG

**SubmitDate**: 2025-02-17    [abs](http://arxiv.org/abs/2502.11863v1) [paper-pdf](http://arxiv.org/pdf/2502.11863v1)

**Authors**: Yahao Pang, Xingyuan Wu, Xiaojin Zhang, Wei Chen, Hai Jin

**Abstract**: Significant advancements have been made by Large Language Models (LLMs) in the domains of natural language understanding and automated content creation. However, they still face persistent problems, including substantial computational costs and inadequate availability of training data. The combination of Federated Learning (FL) and LLMs (federated LLMs) offers a solution by leveraging distributed data while protecting privacy, which positions it as an ideal choice for sensitive domains. However, Federated LLMs still suffer from robustness challenges, including data heterogeneity, malicious clients, and adversarial attacks, which greatly hinder their applications. We first introduce the robustness problems in federated LLMs, to address these challenges, we propose FedEAT (Federated Embedding space Adversarial Training), a novel framework that applies adversarial training in the embedding space of client LLM and employs a robust aggregation approach, specifically geometric median aggregation, to enhance the robustness of Federated LLMs. Our experiments demonstrate that FedEAT effectively improves the robustness of Federated LLMs with minimal performance loss.

摘要: 大型语言模型(LLM)在自然语言理解和自动内容创建领域取得了重大进展。然而，它们仍然面临着长期存在的问题，包括巨大的计算成本和培训数据的不足。联合学习(FL)和联合LLMS(联合LLMS)的结合提供了一种在保护隐私的同时利用分布式数据的解决方案，这将其定位为敏感领域的理想选择。然而，联邦LLMS仍然面临着健壮性挑战，包括数据异构性、恶意客户端和敌意攻击，这些都极大地阻碍了它们的应用。首先介绍了联合LLMS的健壮性问题，针对这些问题，我们提出了一种新的框架FedEAT(Federated Embedding Space Adversal Trading)，该框架将对抗性训练应用于客户端LLMS的嵌入空间，并采用一种稳健的聚集方法，特别是几何中值聚集来增强联合LLMS的健壮性。实验结果表明，FedEAT算法以最小的性能损失有效地提高了联邦LLMS的健壮性。



## **3. Rethinking Audio-Visual Adversarial Vulnerability from Temporal and Modality Perspectives**

从时间和形态角度重新思考视听对抗脆弱性 cs.SD

Accepted by ICLR 2025

**SubmitDate**: 2025-02-17    [abs](http://arxiv.org/abs/2502.11858v1) [paper-pdf](http://arxiv.org/pdf/2502.11858v1)

**Authors**: Zeliang Zhang, Susan Liang, Daiki Shimada, Chenliang Xu

**Abstract**: While audio-visual learning equips models with a richer understanding of the real world by leveraging multiple sensory modalities, this integration also introduces new vulnerabilities to adversarial attacks.   In this paper, we present a comprehensive study of the adversarial robustness of audio-visual models, considering both temporal and modality-specific vulnerabilities. We propose two powerful adversarial attacks: 1) a temporal invariance attack that exploits the inherent temporal redundancy across consecutive time segments and 2) a modality misalignment attack that introduces incongruence between the audio and visual modalities. These attacks are designed to thoroughly assess the robustness of audio-visual models against diverse threats. Furthermore, to defend against such attacks, we introduce a novel audio-visual adversarial training framework. This framework addresses key challenges in vanilla adversarial training by incorporating efficient adversarial perturbation crafting tailored to multi-modal data and an adversarial curriculum strategy. Extensive experiments in the Kinetics-Sounds dataset demonstrate that our proposed temporal and modality-based attacks in degrading model performance can achieve state-of-the-art performance, while our adversarial training defense largely improves the adversarial robustness as well as the adversarial training efficiency.

摘要: 虽然视听学习通过利用多种感官模式使模型对真实世界有了更丰富的理解，但这种集成也引入了新的易受对手攻击的漏洞。在本文中，我们对视听模型的对抗健壮性进行了全面的研究，同时考虑了时间和通道特定的脆弱性。我们提出了两个强大的对抗性攻击：1)利用连续时间段固有的时间冗余性的时间不变性攻击；2)引入视听通道不一致的通道失准攻击。这些攻击旨在彻底评估视听模型对各种威胁的稳健性。此外，为了防御此类攻击，我们引入了一种新的视听对抗性训练框架。这一框架通过结合为多模式数据量身定做的高效对抗性扰动制作和对抗性课程战略，解决了普通对抗性训练中的关键挑战。在Kinetics-Sound数据集上的大量实验表明，我们提出的基于时间和通道的攻击在降低模型性能的同时可以获得最先进的性能，而我们的对抗性训练防御在很大程度上提高了对抗性的健壮性和对抗性训练的效率。



## **4. Practical No-box Adversarial Attacks with Training-free Hybrid Image Transformation**

使用免训练混合图像转换的实用无箱对抗攻击 cs.CV

**SubmitDate**: 2025-02-17    [abs](http://arxiv.org/abs/2203.04607v3) [paper-pdf](http://arxiv.org/pdf/2203.04607v3)

**Authors**: Qilong Zhang, Youheng Sun, Chaoning Zhang, Chaoqun Li, Xuanhan Wang, Jingkuan Song, Lianli Gao

**Abstract**: In recent years, the adversarial vulnerability of deep neural networks (DNNs) has raised increasing attention. Among all the threat models, no-box attacks are the most practical but extremely challenging since they neither rely on any knowledge of the target model or similar substitute model, nor access the dataset for training a new substitute model. Although a recent method has attempted such an attack in a loose sense, its performance is not good enough and computational overhead of training is expensive. In this paper, we move a step forward and show the existence of a \textbf{training-free} adversarial perturbation under the no-box threat model, which can be successfully used to attack different DNNs in real-time. Motivated by our observation that high-frequency component (HFC) domains in low-level features and plays a crucial role in classification, we attack an image mainly by manipulating its frequency components. Specifically, the perturbation is manipulated by suppression of the original HFC and adding of noisy HFC. We empirically and experimentally analyze the requirements of effective noisy HFC and show that it should be regionally homogeneous, repeating and dense. Extensive experiments on the ImageNet dataset demonstrate the effectiveness of our proposed no-box method. It attacks ten well-known models with a success rate of \textbf{98.13\%} on average, which outperforms state-of-the-art no-box attacks by \textbf{29.39\%}. Furthermore, our method is even competitive to mainstream transfer-based black-box attacks.

摘要: 近年来，深度神经网络(DNN)的攻击脆弱性引起了越来越多的关注。在所有的威胁模型中，非盒子攻击是最实用但极具挑战性的攻击，因为它们既不依赖于任何目标模型或类似的替代模型的任何知识，也不需要访问数据集来训练新的替代模型。虽然最近的一种方法在松散意义上尝试了这种攻击，但其性能不够好，并且训练的计算开销很高。在这篇文章中，我们进一步证明了在非盒子威胁模型下存在一个对抗性扰动，它可以成功地用来实时攻击不同的DNN。由于我们观察到高频分量(HFC)域位于低层特征并且在分类中起着关键作用，我们主要通过操纵其频率分量来攻击图像。具体地说，通过抑制原始HFC和添加噪声HFC来操纵扰动。我们从经验和实验上分析了有效的噪声HFC的要求，表明它应该是区域均匀的、重复的和密集的。在ImageNet数据集上的大量实验证明了我们提出的非盒子方法的有效性。它攻击十个著名的模型，平均成功率为\extbf{98.13\%}，比最先进的非盒子攻击的\extbf{29.39\%}要好。此外，我们的方法甚至可以与主流的基于传输的黑盒攻击相竞争。



## **5. Federated Multi-Armed Bandits Under Byzantine Attacks**

拜占庭攻击下的联邦多武装强盗 cs.LG

**SubmitDate**: 2025-02-17    [abs](http://arxiv.org/abs/2205.04134v3) [paper-pdf](http://arxiv.org/pdf/2205.04134v3)

**Authors**: Artun Saday, İlker Demirel, Yiğit Yıldırım, Cem Tekin

**Abstract**: Multi-armed bandits (MAB) is a sequential decision-making model in which the learner controls the trade-off between exploration and exploitation to maximize its cumulative reward. Federated multi-armed bandits (FMAB) is an emerging framework where a cohort of learners with heterogeneous local models play an MAB game and communicate their aggregated feedback to a server to learn a globally optimal arm. Two key hurdles in FMAB are communication-efficient learning and resilience to adversarial attacks. To address these issues, we study the FMAB problem in the presence of Byzantine clients who can send false model updates threatening the learning process. We analyze the sample complexity and the regret of $\beta$-optimal arm identification. We borrow tools from robust statistics and propose a median-of-means (MoM)-based online algorithm, Fed-MoM-UCB, to cope with Byzantine clients. In particular, we show that if the Byzantine clients constitute less than half of the cohort, the cumulative regret with respect to $\beta$-optimal arms is bounded over time with high probability, showcasing both communication efficiency and Byzantine resilience. We analyze the interplay between the algorithm parameters, a discernibility margin, regret, communication cost, and the arms' suboptimality gaps. We demonstrate Fed-MoM-UCB's effectiveness against the baselines in the presence of Byzantine attacks via experiments.

摘要: 多武装强盗(MAB)是一种序贯决策模型，在该模型中，学习者控制探索和剥削之间的权衡，以最大化其累积回报。联邦多臂强盗(FMAB)是一种新兴的框架，在这种框架中，具有不同本地模型的一群学习者玩MAB游戏，并将他们汇总的反馈传达给服务器，以学习全局最优的ARM。FMAB的两个关键障碍是高效沟通的学习和对对手攻击的适应能力。为了解决这些问题，我们在拜占庭客户端存在的情况下研究FMAB问题，这些客户端可能会发送虚假的模型更新，威胁到学习过程。分析了样本复杂度和最优ARM识别的遗憾。我们借用稳健统计的工具，提出了一种基于均值中位数(MOM)的在线算法FED-MOM-UCB，以应对拜占庭式的客户。特别地，我们证明了如果拜占庭客户端不到队列的一半，关于$\beta$-最优ARM的累积后悔是以很高的概率随时间有界的，展示了通信效率和拜占庭韧性。我们分析了算法参数、可分辨裕度、后悔、通信成本和ARM的次优差距之间的相互影响。我们通过实验证明了在拜占庭攻击存在的情况下，FED-MOM-UCB相对于基线的有效性。



## **6. Adversarially Robust CLIP Models Can Induce Better (Robust) Perceptual Metrics**

对抗稳健的CLIP模型可以诱导更好（稳健）的感知能力 cs.CV

This work has been accepted for publication in the IEEE Conference on  Secure and Trustworthy Machine Learning (SaTML). The final version will be  available on IEEE Xplore

**SubmitDate**: 2025-02-17    [abs](http://arxiv.org/abs/2502.11725v1) [paper-pdf](http://arxiv.org/pdf/2502.11725v1)

**Authors**: Francesco Croce, Christian Schlarmann, Naman Deep Singh, Matthias Hein

**Abstract**: Measuring perceptual similarity is a key tool in computer vision. In recent years perceptual metrics based on features extracted from neural networks with large and diverse training sets, e.g. CLIP, have become popular. At the same time, the metrics extracted from features of neural networks are not adversarially robust. In this paper we show that adversarially robust CLIP models, called R-CLIP$_\textrm{F}$, obtained by unsupervised adversarial fine-tuning induce a better and adversarially robust perceptual metric that outperforms existing metrics in a zero-shot setting, and further matches the performance of state-of-the-art metrics while being robust after fine-tuning. Moreover, our perceptual metric achieves strong performance on related tasks such as robust image-to-image retrieval, which becomes especially relevant when applied to "Not Safe for Work" (NSFW) content detection and dataset filtering. While standard perceptual metrics can be easily attacked by a small perturbation completely degrading NSFW detection, our robust perceptual metric maintains high accuracy under an attack while having similar performance for unperturbed images. Finally, perceptual metrics induced by robust CLIP models have higher interpretability: feature inversion can show which images are considered similar, while text inversion can find what images are associated to a given prompt. This also allows us to visualize the very rich visual concepts learned by a CLIP model, including memorized persons, paintings and complex queries.

摘要: 感知相似性度量是计算机视觉中的一个重要工具。近年来，基于从具有大量和多样化训练集的神经网络(例如CLIP)中提取的特征的感知度量已经变得流行起来。同时，从神经网络的特征中提取的度量并不是相反的健壮性。在本文中，我们证明了通过无监督对抗性微调获得的对抗性健壮性片段模型R-Clip$tExtrm{F}$在零镜头设置下获得了比现有度量更好的对抗性健壮性感知度量，并且进一步匹配了最新度量的性能，并且在微调后仍具有健壮性。此外，我们的感知度量在相关任务中取得了良好的性能，例如稳健的图像到图像检索，当应用于不安全的工作(NSFW)内容检测和数据集过滤时，这变得特别重要。虽然标准的感知度量很容易受到微小扰动的攻击，完全降低了NSFW检测的性能，但我们的稳健感知度量在攻击下保持了高精度，而对于未受干扰的图像具有类似的性能。最后，由稳健剪辑模型得出的感知度量具有更高的可解释性：特征反转可以显示哪些图像被认为相似，而文本反转可以发现哪些图像与给定提示相关联。这也让我们可以可视化剪辑模型所学到的非常丰富的视觉概念，包括记忆中的人物、绘画和复杂的查询。



## **7. DELMAN: Dynamic Defense Against Large Language Model Jailbreaking with Model Editing**

德尔曼：通过模型编辑对大型语言模型越狱的动态防御 cs.CR

**SubmitDate**: 2025-02-17    [abs](http://arxiv.org/abs/2502.11647v1) [paper-pdf](http://arxiv.org/pdf/2502.11647v1)

**Authors**: Yi Wang, Fenghua Weng, Sibei Yang, Zhan Qin, Minlie Huang, Wenjie Wang

**Abstract**: Large Language Models (LLMs) are widely applied in decision making, but their deployment is threatened by jailbreak attacks, where adversarial users manipulate model behavior to bypass safety measures. Existing defense mechanisms, such as safety fine-tuning and model editing, either require extensive parameter modifications or lack precision, leading to performance degradation on general tasks, which is unsuitable to post-deployment safety alignment. To address these challenges, we propose DELMAN (Dynamic Editing for LLMs JAilbreak DefeNse), a novel approach leveraging direct model editing for precise, dynamic protection against jailbreak attacks. DELMAN directly updates a minimal set of relevant parameters to neutralize harmful behaviors while preserving the model's utility. To avoid triggering a safe response in benign context, we incorporate KL-divergence regularization to ensure the updated model remains consistent with the original model when processing benign queries. Experimental results demonstrate that DELMAN outperforms baseline methods in mitigating jailbreak attacks while preserving the model's utility, and adapts seamlessly to new attack instances, providing a practical and efficient solution for post-deployment model protection.

摘要: 大语言模型(LLM)在决策中被广泛应用，但它们的部署受到越狱攻击的威胁，在越狱攻击中，敌对用户操纵模型行为以绕过安全措施。现有的防御机制，如安全微调和模型编辑，要么需要大量修改参数，要么缺乏精度，导致一般任务的性能下降，不适合部署后的安全对齐。为了应对这些挑战，我们提出了Delman(用于LLMS越狱防御的动态编辑)，这是一种利用直接模型编辑来精确、动态地防御越狱攻击的新方法。Delman直接更新相关参数的最小集合，以中和有害行为，同时保持模型的实用性。为了避免在良性环境下触发安全响应，我们引入了KL-散度正则化，以确保在处理良性查询时更新后的模型与原始模型保持一致。实验结果表明，Delman在保持模型实用性的同时，在缓解越狱攻击方面优于基准方法，并能无缝适应新的攻击实例，为部署后模型防护提供了一种实用而高效的解决方案。



## **8. Can LLM Watermarks Robustly Prevent Unauthorized Knowledge Distillation?**

LLM水印能否强大地防止未经授权的知识提炼？ cs.CL

22 pages, 12 figures, 13 tables

**SubmitDate**: 2025-02-17    [abs](http://arxiv.org/abs/2502.11598v1) [paper-pdf](http://arxiv.org/pdf/2502.11598v1)

**Authors**: Leyi Pan, Aiwei Liu, Shiyu Huang, Yijian Lu, Xuming Hu, Lijie Wen, Irwin King, Philip S. Yu

**Abstract**: The radioactive nature of Large Language Model (LLM) watermarking enables the detection of watermarks inherited by student models when trained on the outputs of watermarked teacher models, making it a promising tool for preventing unauthorized knowledge distillation. However, the robustness of watermark radioactivity against adversarial actors remains largely unexplored. In this paper, we investigate whether student models can acquire the capabilities of teacher models through knowledge distillation while avoiding watermark inheritance. We propose two categories of watermark removal approaches: pre-distillation removal through untargeted and targeted training data paraphrasing (UP and TP), and post-distillation removal through inference-time watermark neutralization (WN). Extensive experiments across multiple model pairs, watermarking schemes and hyper-parameter settings demonstrate that both TP and WN thoroughly eliminate inherited watermarks, with WN achieving this while maintaining knowledge transfer efficiency and low computational overhead. Given the ongoing deployment of watermarking techniques in production LLMs, these findings emphasize the urgent need for more robust defense strategies. Our code is available at https://github.com/THU-BPM/Watermark-Radioactivity-Attack.

摘要: 大型语言模型(LLM)水印的放射性特性使其能够在对带水印的教师模型的输出进行训练时检测由学生模型继承的水印，使其成为防止未经授权的知识蒸馏的一种有前途的工具。然而，水印放射性对敌方行为的稳健性在很大程度上仍未被探索。本文研究了学生模型能否在避免水印继承的同时，通过知识提炼获得教师模型的能力。我们提出了两类水印去除方法：通过非目标和目标训练数据释义(UP和TP)进行蒸馏前去除和通过推理时间水印中和(WN)进行蒸馏后去除。在多个模型对、水印方案和超参数设置上的大量实验表明，TP和WN都彻底消除了继承的水印，WN在保持知识传递效率和较低的计算开销的同时实现了这一点。鉴于水印技术在生产LLM中的持续部署，这些发现强调了对更强大的防御策略的迫切需要。我们的代码可以在https://github.com/THU-BPM/Watermark-Radioactivity-Attack.上找到



## **9. Adversary-Aware DPO: Enhancing Safety Alignment in Vision Language Models via Adversarial Training**

具有对抗意识的DPO：通过对抗训练增强视觉语言模型中的安全一致性 cs.CR

**SubmitDate**: 2025-02-17    [abs](http://arxiv.org/abs/2502.11455v1) [paper-pdf](http://arxiv.org/pdf/2502.11455v1)

**Authors**: Fenghua Weng, Jian Lou, Jun Feng, Minlie Huang, Wenjie Wang

**Abstract**: Safety alignment is critical in pre-training large language models (LLMs) to generate responses aligned with human values and refuse harmful queries. Unlike LLM, the current safety alignment of VLMs is often achieved with post-hoc safety fine-tuning. However, these methods are less effective to white-box attacks. To address this, we propose $\textit{Adversary-aware DPO (ADPO)}$, a novel training framework that explicitly considers adversarial. $\textit{Adversary-aware DPO (ADPO)}$ integrates adversarial training into DPO to enhance the safety alignment of VLMs under worst-case adversarial perturbations. $\textit{ADPO}$ introduces two key components: (1) an adversarial-trained reference model that generates human-preferred responses under worst-case perturbations, and (2) an adversarial-aware DPO loss that generates winner-loser pairs accounting for adversarial distortions. By combining these innovations, $\textit{ADPO}$ ensures that VLMs remain robust and reliable even in the presence of sophisticated jailbreak attacks. Extensive experiments demonstrate that $\textit{ADPO}$ outperforms baselines in the safety alignment and general utility of VLMs.

摘要: 在预先训练大型语言模型(LLM)以生成与人类价值观一致的响应并拒绝有害查询时，安全对齐至关重要。与LLM不同，VLM当前的安全对准通常是通过事后安全微调来实现的。然而，这些方法对白盒攻击的有效性较低。为了解决这一问题，我们提出了一种新的训练框架将对抗性训练融入到DPO中，以增强VLM在最坏情况下的对抗性扰动下的安全一致性。$\textit{ADPO}$引入了两个关键组件：(1)对抗性训练的参考模型，它在最坏情况下产生人类偏好的响应；(2)对抗性感知的DPO损失，它产生考虑对抗性扭曲的赢家-输家对。通过将这些创新结合在一起，$\textit{ADPO}$确保即使在存在复杂的越狱攻击的情况下，VLM仍保持健壮和可靠。大量实验表明，在VLMS的安全性、对准和通用性方面，$\textit{ADPO}$都优于Baseline。



## **10. Dagger Behind Smile: Fool LLMs with a Happy Ending Story**

微笑背后的匕首：傻瓜LLMs，有一个幸福的结局 cs.CL

**SubmitDate**: 2025-02-17    [abs](http://arxiv.org/abs/2501.13115v2) [paper-pdf](http://arxiv.org/pdf/2501.13115v2)

**Authors**: Xurui Song, Zhixin Xie, Shuo Huai, Jiayi Kong, Jun Luo

**Abstract**: The wide adoption of Large Language Models (LLMs) has attracted significant attention from $\textit{jailbreak}$ attacks, where adversarial prompts crafted through optimization or manual design exploit LLMs to generate malicious contents. However, optimization-based attacks have limited efficiency and transferability, while existing manual designs are either easily detectable or demand intricate interactions with LLMs. In this paper, we first point out a novel perspective for jailbreak attacks: LLMs are more responsive to $\textit{positive}$ prompts. Based on this, we deploy Happy Ending Attack (HEA) to wrap up a malicious request in a scenario template involving a positive prompt formed mainly via a $\textit{happy ending}$, it thus fools LLMs into jailbreaking either immediately or at a follow-up malicious request.This has made HEA both efficient and effective, as it requires only up to two turns to fully jailbreak LLMs. Extensive experiments show that our HEA can successfully jailbreak on state-of-the-art LLMs, including GPT-4o, Llama3-70b, Gemini-pro, and achieves 88.79\% attack success rate on average. We also provide quantitative explanations for the success of HEA.

摘要: 大型语言模型(LLM)的广泛采用引起了$\textit{jailBreak}$攻击的极大关注，即通过优化或手动设计创建的敌意提示利用LLM生成恶意内容。然而，基于优化的攻击的效率和可转移性有限，而现有的手动设计要么很容易被检测到，要么需要与LLM进行复杂的交互。在这篇文章中，我们首先指出了越狱攻击的一个新视角：LLM对$\textit{积极}$提示的响应更快。在此基础上，利用HEA(Happy End End Attack)将恶意请求封装在一个场景模板中，该场景模板包含一个主要通过$\textit{Happy End}$形成的积极提示，从而欺骗LLM立即越狱或在后续恶意请求时越狱，这使得HEA既高效又有效，因为它只需要最多两个回合就可以完全越狱LLM。大量的实验表明，我们的HEA能够成功地在GPT-40、Llama3-70b、Gemini-Pro等最先进的LLMS上越狱，平均攻击成功率达到88.79%。我们还对HEA的成功提供了定量的解释。



## **11. Mimicking the Familiar: Dynamic Command Generation for Information Theft Attacks in LLM Tool-Learning System**

模仿熟悉的：LLM工具学习系统中信息窃取攻击的动态命令生成 cs.AI

15 pages, 11 figures

**SubmitDate**: 2025-02-17    [abs](http://arxiv.org/abs/2502.11358v1) [paper-pdf](http://arxiv.org/pdf/2502.11358v1)

**Authors**: Ziyou Jiang, Mingyang Li, Guowei Yang, Junjie Wang, Yuekai Huang, Zhiyuan Chang, Qing Wang

**Abstract**: Information theft attacks pose a significant risk to Large Language Model (LLM) tool-learning systems. Adversaries can inject malicious commands through compromised tools, manipulating LLMs to send sensitive information to these tools, which leads to potential privacy breaches. However, existing attack approaches are black-box oriented and rely on static commands that cannot adapt flexibly to the changes in user queries and the invocation chain of tools. It makes malicious commands more likely to be detected by LLM and leads to attack failure. In this paper, we propose AutoCMD, a dynamic attack comment generation approach for information theft attacks in LLM tool-learning systems. Inspired by the concept of mimicking the familiar, AutoCMD is capable of inferring the information utilized by upstream tools in the toolchain through learning on open-source systems and reinforcement with target system examples, thereby generating more targeted commands for information theft. The evaluation results show that AutoCMD outperforms the baselines with +13.2% $ASR_{Theft}$, and can be generalized to new tool-learning systems to expose their information leakage risks. We also design four defense methods to effectively protect tool-learning systems from the attack.

摘要: 信息窃取攻击对大型语言模型(LLM)工具学习系统构成了重大风险。攻击者可以通过受危害的工具注入恶意命令，操纵LLM向这些工具发送敏感信息，从而导致潜在的隐私泄露。然而，现有的攻击方法是面向黑盒的，依赖于静态命令，不能灵活地适应用户查询和工具调用链的变化。它使恶意命令更容易被LLM检测到，并导致攻击失败。本文针对LLM工具学习系统中的信息窃取攻击，提出了一种动态攻击评论生成方法AutoCMD。受模仿熟悉的概念的启发，AutoCMD能够通过学习开源系统和加强目标系统示例来推断工具链中的上游工具所使用的信息，从而生成更有针对性的信息窃取命令。评估结果表明，AutoCMD的性能比基准高出13.2%$ASR{Theft}$，可以推广到新的工具学习系统，以暴露其信息泄露风险。我们还设计了四种防御方法来有效地保护工具学习系统免受攻击。



## **12. How to Backdoor Consistency Models?**

如何后门一致性模型？ cs.CR

**SubmitDate**: 2025-02-16    [abs](http://arxiv.org/abs/2410.19785v3) [paper-pdf](http://arxiv.org/pdf/2410.19785v3)

**Authors**: Chengen Wang, Murat Kantarcioglu

**Abstract**: Consistency models are a new class of models that generate images by directly mapping noise to data, allowing for one-step generation and significantly accelerating the sampling process. However, their robustness against adversarial attacks has not yet been thoroughly investigated. In this work, we conduct the first study on the vulnerability of consistency models to backdoor attacks. While previous research has explored backdoor attacks on diffusion models, those studies have primarily focused on conventional diffusion models, employing a customized backdoor training process and objective, whereas consistency models have distinct training processes and objectives. Our proposed framework demonstrates the vulnerability of consistency models to backdoor attacks. During image generation, poisoned consistency models produce images with a Fr\'echet Inception Distance (FID) comparable to that of a clean model when sampling from Gaussian noise. However, once the trigger is activated, they generate backdoor target images. We explore various trigger and target configurations to evaluate the vulnerability of consistency models, including the use of random noise as a trigger. This novel trigger is visually inconspicuous, more challenging to detect, and aligns well with the sampling process of consistency models. Across all configurations, our framework successfully compromises the consistency models while maintaining high utility and specificity. We also examine the stealthiness of our proposed attack, which is attributed to the unique properties of consistency models and the elusive nature of the Gaussian noise trigger. Our code is available at \href{https://github.com/chengenw/backdoorCM}{https://github.com/chengenw/backdoorCM}.

摘要: 一致性模型是一类新的模型，通过将噪声直接映射到数据来生成图像，允许一步生成并显著加快采样过程。然而，它们对敌意攻击的健壮性还没有得到彻底的研究。在这项工作中，我们首次研究了一致性模型对后门攻击的脆弱性。虽然以前的研究探讨了对扩散模型的后门攻击，但这些研究主要集中在传统的扩散模型上，采用定制的后门培训过程和目标，而一致性模型有不同的培训过程和目标。我们提出的框架证明了一致性模型对后门攻击的脆弱性。在图像生成过程中，当从高斯噪声中采样时，有毒一致性模型生成的图像具有与清洁模型相当的Fr回声初始距离(FID)。然而，一旦触发器被激活，它们就会生成后门目标图像。我们探索了各种触发和目标配置来评估一致性模型的脆弱性，包括使用随机噪声作为触发。这种新颖的触发器在视觉上不显眼，更难检测，并且与一致性模型的采样过程很好地一致。在所有配置中，我们的框架成功地折衷了一致性模型，同时保持了高度的实用性和专用性。我们还检查了我们提出的攻击的隐蔽性，这归因于一致性模型的独特性质和高斯噪声触发的难以捉摸的性质。我们的代码可以在\href{https://github.com/chengenw/backdoorCM}{https://github.com/chengenw/backdoorCM}.上找到



## **13. G-Safeguard: A Topology-Guided Security Lens and Treatment on LLM-based Multi-agent Systems**

G-Safeguard：基于LLM的多智能体系统上的一种基于布局引导的安全视角和处理方法 cs.CR

**SubmitDate**: 2025-02-16    [abs](http://arxiv.org/abs/2502.11127v1) [paper-pdf](http://arxiv.org/pdf/2502.11127v1)

**Authors**: Shilong Wang, Guibin Zhang, Miao Yu, Guancheng Wan, Fanci Meng, Chongye Guo, Kun Wang, Yang Wang

**Abstract**: Large Language Model (LLM)-based Multi-agent Systems (MAS) have demonstrated remarkable capabilities in various complex tasks, ranging from collaborative problem-solving to autonomous decision-making. However, as these systems become increasingly integrated into critical applications, their vulnerability to adversarial attacks, misinformation propagation, and unintended behaviors have raised significant concerns. To address this challenge, we introduce G-Safeguard, a topology-guided security lens and treatment for robust LLM-MAS, which leverages graph neural networks to detect anomalies on the multi-agent utterance graph and employ topological intervention for attack remediation. Extensive experiments demonstrate that G-Safeguard: (I) exhibits significant effectiveness under various attack strategies, recovering over 40% of the performance for prompt injection; (II) is highly adaptable to diverse LLM backbones and large-scale MAS; (III) can seamlessly combine with mainstream MAS with security guarantees. The code is available at https://github.com/wslong20/G-safeguard.

摘要: 基于大型语言模型(LLM)的多智能体系统(MAS)在从协作问题求解到自主决策的各种复杂任务中表现出了卓越的能力。然而，随着这些系统越来越多地集成到关键应用程序中，它们对对手攻击、错误信息传播和意外行为的脆弱性已经引起了极大的关注。为了应对这一挑战，我们引入了G-Safe，这是一种拓扑制导的安全镜头和健壮LLM-MAS的处理方法，它利用图神经网络来检测多智能体话语图上的异常，并使用拓扑干预进行攻击补救。大量实验表明：(I)在各种攻击策略下表现出显著的有效性，可恢复40%以上的性能进行快速注入；(Ii)对不同的LLM主干和大规模MAS具有高度的适应性；(Iii)可以与主流MAS无缝结合，具有安全保障。代码可在https://github.com/wslong20/G-safeguard.上获得



## **14. Exploiting Defenses against GAN-Based Feature Inference Attacks in Federated Learning**

在联邦学习中利用防御基于GAN的特征推理攻击 cs.CR

**SubmitDate**: 2025-02-16    [abs](http://arxiv.org/abs/2004.12571v4) [paper-pdf](http://arxiv.org/pdf/2004.12571v4)

**Authors**: Xinjian Luo, Xianglong Zhang

**Abstract**: Federated learning (FL) is a decentralized model training framework that aims to merge isolated data islands while maintaining data privacy. However, recent studies have revealed that Generative Adversarial Network (GAN) based attacks can be employed in FL to learn the distribution of private datasets and reconstruct recognizable images. In this paper, we exploit defenses against GAN-based attacks in FL and propose a framework, Anti-GAN, to prevent attackers from learning the real distribution of the victim's data. The core idea of Anti-GAN is to manipulate the visual features of private training images to make them indistinguishable to human eyes even restored by attackers. Specifically, Anti-GAN projects the private dataset onto a GAN's generator and combines the generated fake images with the actual images to create the training dataset, which is then used for federated model training. The experimental results demonstrate that Anti-GAN is effective in preventing attackers from learning the distribution of private images while causing minimal harm to the accuracy of the federated model.

摘要: 联邦学习(FL)是一种去中心化的模型训练框架，旨在合并孤立的数据孤岛，同时保持数据隐私。然而，最近的研究表明，基于生成性对抗网络(GAN)的攻击可以用于FL中，以学习私有数据集的分布并重建可识别的图像。在本文中，我们在FL中利用对基于GAN的攻击的防御，并提出了一个框架--Anti-GAN，以防止攻击者了解受害者数据的真实分布。Anti-GAN的核心思想是操纵私人训练图像的视觉特征，使其即使被攻击者恢复也无法辨别人眼。具体地说，Anti-GAN将私有数据集投影到GAN的生成器上，并将生成的虚假图像与实际图像相结合来创建训练数据集，然后将其用于联合模型训练。实验结果表明，该算法能有效地防止攻击者学习私有图像的分布，同时对联邦模型的准确性造成最小的损害。



## **15. Rewrite to Jailbreak: Discover Learnable and Transferable Implicit Harmfulness Instruction**

重写越狱：发现可学习和可转移的隐性有害指令 cs.CL

21pages, 10 figures

**SubmitDate**: 2025-02-16    [abs](http://arxiv.org/abs/2502.11084v1) [paper-pdf](http://arxiv.org/pdf/2502.11084v1)

**Authors**: Yuting Huang, Chengyuan Liu, Yifeng Feng, Chao Wu, Fei Wu, Kun Kuang

**Abstract**: As Large Language Models (LLMs) are widely applied in various domains, the safety of LLMs is increasingly attracting attention to avoid their powerful capabilities being misused. Existing jailbreak methods create a forced instruction-following scenario, or search adversarial prompts with prefix or suffix tokens to achieve a specific representation manually or automatically. However, they suffer from low efficiency and explicit jailbreak patterns, far from the real deployment of mass attacks to LLMs. In this paper, we point out that simply rewriting the original instruction can achieve a jailbreak, and we find that this rewriting approach is learnable and transferable. We propose the Rewrite to Jailbreak (R2J) approach, a transferable black-box jailbreak method to attack LLMs by iteratively exploring the weakness of the LLMs and automatically improving the attacking strategy. The jailbreak is more efficient and hard to identify since no additional features are introduced. Extensive experiments and analysis demonstrate the effectiveness of R2J, and we find that the jailbreak is also transferable to multiple datasets and various types of models with only a few queries. We hope our work motivates further investigation of LLM safety.

摘要: 随着大语言模型在各个领域的广泛应用，大语言模型的安全性越来越受到人们的关注，以避免其强大的功能被滥用。现有的越狱方法创建了强制遵循指令的场景，或者搜索带有前缀或后缀令牌的对抗性提示，以手动或自动地实现特定的表示。然而，他们遭受的是低效率和明确的越狱模式，远远不能真正部署大规模攻击到LLM。在本文中，我们指出，简单地重写原始指令就可以实现越狱，并且我们发现这种重写方法是可学习的和可移植的。提出了重写越狱(R2J)方法，通过迭代挖掘LLMS的弱点并自动改进攻击策略，提出了一种可转移的黑盒越狱方法来攻击LLMS。由于没有引入其他功能，越狱更加高效，也更难识别。大量的实验和分析证明了R2J的有效性，我们发现越狱也可以只需几个查询就可以移植到多个数据集和各种类型的模型上。我们希望我们的工作能促进对LLM安全性的进一步研究。



## **16. Na'vi or Knave: Jailbreaking Language Models via Metaphorical Avatars**

纳威人或恶棍：通过隐喻化身越狱语言模型 cs.CL

We still need to polish our paper

**SubmitDate**: 2025-02-16    [abs](http://arxiv.org/abs/2412.12145v2) [paper-pdf](http://arxiv.org/pdf/2412.12145v2)

**Authors**: Yu Yan, Sheng Sun, Junqi Tong, Min Liu, Qi Li

**Abstract**: Metaphor serves as an implicit approach to convey information, while enabling the generalized comprehension of complex subjects. However, metaphor can potentially be exploited to bypass the safety alignment mechanisms of Large Language Models (LLMs), leading to the theft of harmful knowledge. In our study, we introduce a novel attack framework that exploits the imaginative capacity of LLMs to achieve jailbreaking, the J\underline{\textbf{A}}ilbreak \underline{\textbf{V}}ia \underline{\textbf{A}}dversarial Me\underline{\textbf{TA}} -pho\underline{\textbf{R}} (\textit{AVATAR}). Specifically, to elicit the harmful response, AVATAR extracts harmful entities from a given harmful target and maps them to innocuous adversarial entities based on LLM's imagination. Then, according to these metaphors, the harmful target is nested within human-like interaction for jailbreaking adaptively. Experimental results demonstrate that AVATAR can effectively and transferablly jailbreak LLMs and achieve a state-of-the-art attack success rate across multiple advanced LLMs. Our study exposes a security risk in LLMs from their endogenous imaginative capabilities. Furthermore, the analytical study reveals the vulnerability of LLM to adversarial metaphors and the necessity of developing defense methods against jailbreaking caused by the adversarial metaphor. \textcolor{orange}{ \textbf{Warning: This paper contains potentially harmful content from LLMs.}}

摘要: 隐喻是一种隐含的传达信息的方法，同时也使人们能够对复杂的主题进行广义的理解。然而，隐喻可能被利用来绕过大型语言模型(LLM)的安全对齐机制，从而导致有害知识的窃取。在我们的研究中，我们提出了一种新的攻击框架，该框架利用LLMS的想象能力来实现越狱，即J下划线{\extbf{A}}ilBreak\下划线{\extbf{A}}ia\下划线{\extbf{A}}dversarial Me\下划线{\extbf{TA}}-pho\下划线{\extbf{R}}(\textit{阿凡达})。具体地说，为了引发有害反应，阿凡达从给定的有害目标中提取有害实体，并基于LLM的想象力将它们映射到无害的对抗性实体。然后，根据这些隐喻，有害的目标嵌套在类似人类的互动中，以适应越狱。实验结果表明，阿凡达可以有效地、可转移地越狱LLM，并在多个高级LLM上获得最先进的攻击成功率。我们的研究揭示了LLMS的安全风险来自其内生的想象力。此外，分析研究揭示了LLM对对抗性隐喻的脆弱性，以及开发针对对抗性隐喻导致的越狱的防御方法的必要性。\extCOLOR{橙色}{\extbf{警告：此纸张包含来自LLMS的潜在有害内容。}}



## **17. Atoxia: Red-teaming Large Language Models with Target Toxic Answers**

Atoxia：将大型语言模型与目标有毒答案进行红色合作 cs.CL

Accepted to Findings of NAACL-2025

**SubmitDate**: 2025-02-16    [abs](http://arxiv.org/abs/2408.14853v2) [paper-pdf](http://arxiv.org/pdf/2408.14853v2)

**Authors**: Yuhao Du, Zhuo Li, Pengyu Cheng, Xiang Wan, Anningzhe Gao

**Abstract**: Despite the substantial advancements in artificial intelligence, large language models (LLMs) remain being challenged by generation safety. With adversarial jailbreaking prompts, one can effortlessly induce LLMs to output harmful content, causing unexpected negative social impacts. This vulnerability highlights the necessity for robust LLM red-teaming strategies to identify and mitigate such risks before large-scale application. To detect specific types of risks, we propose a novel red-teaming method that $\textbf{A}$ttacks LLMs with $\textbf{T}$arget $\textbf{Toxi}$c $\textbf{A}$nswers ($\textbf{Atoxia}$). Given a particular harmful answer, Atoxia generates a corresponding user query and a misleading answer opening to examine the internal defects of a given LLM. The proposed attacker is trained within a reinforcement learning scheme with the LLM outputting probability of the target answer as the reward. We verify the effectiveness of our method on various red-teaming benchmarks, such as AdvBench and HH-Harmless. The empirical results demonstrate that Atoxia can successfully detect safety risks in not only open-source models but also state-of-the-art black-box models such as GPT-4o.

摘要: 尽管人工智能取得了实质性的进步，但大型语言模型(LLM)仍然受到发电安全的挑战。在对抗性越狱提示下，人们可以毫不费力地诱导LLMS输出有害内容，造成意想不到的负面社会影响。该漏洞突显了在大规模应用之前，需要强大的LLM红团队战略来识别和缓解此类风险。为了检测特定类型的风险，我们提出了一种新的红团队方法，即用$\extbf{T}$目标$\extbf{Toxi}$c$\extbf{A}$nswers($\extbf{Atoxia}$)来绑定LLMS。给定特定的有害答案，Atoxia会生成相应的用户查询和误导性答案，以检查给定LLM的内部缺陷。所提出的攻击者在强化学习方案中被训练，LLM输出目标答案的概率作为奖励。我们在不同的红队基准测试上验证了我们的方法的有效性，例如AdvBtch和HH-无害。实验结果表明，Atoxia不仅可以在开源模型中成功检测安全风险，而且可以在GPT-40等最先进的黑盒模型中成功检测到安全风险。



## **18. JPEG Inspired Deep Learning**

JPEG启发深度学习 cs.CV

**SubmitDate**: 2025-02-16    [abs](http://arxiv.org/abs/2410.07081v2) [paper-pdf](http://arxiv.org/pdf/2410.07081v2)

**Authors**: Ahmed H. Salamah, Kaixiang Zheng, Yiwen Liu, En-Hui Yang

**Abstract**: Although it is traditionally believed that lossy image compression, such as JPEG compression, has a negative impact on the performance of deep neural networks (DNNs), it is shown by recent works that well-crafted JPEG compression can actually improve the performance of deep learning (DL). Inspired by this, we propose JPEG-DL, a novel DL framework that prepends any underlying DNN architecture with a trainable JPEG compression layer. To make the quantization operation in JPEG compression trainable, a new differentiable soft quantizer is employed at the JPEG layer, and then the quantization operation and underlying DNN are jointly trained. Extensive experiments show that in comparison with the standard DL, JPEG-DL delivers significant accuracy improvements across various datasets and model architectures while enhancing robustness against adversarial attacks. Particularly, on some fine-grained image classification datasets, JPEG-DL can increase prediction accuracy by as much as 20.9%. Our code is available on https://github.com/AhmedHussKhalifa/JPEG-Inspired-DL.git.

摘要: 虽然传统上认为有损图像压缩，如JPEG压缩，会对深度神经网络(DNN)的性能产生负面影响，但最近的研究表明，精心设计的JPEG压缩实际上可以提高深度学习(DL)的性能。受此启发，我们提出了JPEG-DL，这是一种新颖的DL框架，它在任何底层的DNN体系结构中都预先加入了一个可训练的JPEG压缩层。为了使JPEG压缩中的量化操作可训练，在JPEG层使用了一种新的可微软量化器，然后将量化操作和底层的DNN进行联合训练。大量的实验表明，与标准的DL相比，JPEG-DL在不同的数据集和模型体系结构上提供了显著的准确性改进，同时增强了对对手攻击的健壮性。特别是，在一些细粒度的图像分类数据集上，JPEG-DL可以将预测精度提高20.9%。我们的代码可以在https://github.com/AhmedHussKhalifa/JPEG-Inspired-DL.git.上找到



## **19. RoMA: Robust Malware Attribution via Byte-level Adversarial Training with Global Perturbations and Adversarial Consistency Regularization**

RoMA：通过具有全局扰动和对抗一致性正规化的字节级对抗训练来实现稳健的恶意软件归因 cs.CR

11 pages, 4 figures

**SubmitDate**: 2025-02-15    [abs](http://arxiv.org/abs/2502.07492v2) [paper-pdf](http://arxiv.org/pdf/2502.07492v2)

**Authors**: Yuxia Sun, Huihong Chen, Jingcai Guo, Aoxiang Sun, Zhetao Li, Haolin Liu

**Abstract**: Attributing APT (Advanced Persistent Threat) malware to their respective groups is crucial for threat intelligence and cybersecurity. However, APT adversaries often conceal their identities, rendering attribution inherently adversarial. Existing machine learning-based attribution models, while effective, remain highly vulnerable to adversarial attacks. For example, the state-of-the-art byte-level model MalConv sees its accuracy drop from over 90% to below 2% under PGD (projected gradient descent) attacks. Existing gradient-based adversarial training techniques for malware detection or image processing were applied to malware attribution in this study, revealing that both robustness and training efficiency require significant improvement. To address this, we propose RoMA, a novel single-step adversarial training approach that integrates global perturbations to generate enhanced adversarial samples and employs adversarial consistency regularization to improve representation quality and resilience. A novel APT malware dataset named AMG18, with diverse samples and realistic class imbalances, is introduced for evaluation. Extensive experiments show that RoMA significantly outperforms seven competing methods in both adversarial robustness (e.g., achieving over 80% robust accuracy-more than twice that of the next-best method under PGD attacks) and training efficiency (e.g., more than twice as fast as the second-best method in terms of accuracy), while maintaining superior standard accuracy in non-adversarial scenarios.

摘要: 将APT(高级持续威胁)恶意软件归因于各自的组织对于威胁情报和网络安全至关重要。然而，聪明的对手往往隐藏自己的身份，使归因具有内在的对抗性。现有的基于机器学习的归因模型虽然有效，但仍然非常容易受到对手的攻击。例如，最先进的字节级模型MalConv在PGD(投影梯度下降)攻击下的准确率从90%以上下降到2%以下。将已有的基于梯度的恶意软件检测或图像处理的对抗性训练技术应用到恶意软件属性识别中，发现无论是稳健性还是训练效率都需要显著提高。为了解决这一问题，我们提出了一种新颖的单步对抗性训练方法，该方法结合全局扰动来生成增强的对抗性样本，并使用对抗性一致性正则化来提高表示质量和韧性。引入了一个新的APT恶意软件数据集AMG18，该数据集具有多样化的样本和真实的类别不平衡。大量实验表明，在对抗性稳健性(例如，在PGD攻击下达到80%以上的健壮性--是次佳方法的两倍多)和训练效率(例如，在准确率方面是次佳方法的两倍以上)方面，ROMA显著优于七种竞争方法，同时在非对抗性场景中保持了卓越的标准准确率。



## **20. MITRE ATT&CK Applications in Cybersecurity and The Way Forward**

MITRE ATT & CK在网络安全领域的应用和前进之路 cs.CR

37 pages

**SubmitDate**: 2025-02-15    [abs](http://arxiv.org/abs/2502.10825v1) [paper-pdf](http://arxiv.org/pdf/2502.10825v1)

**Authors**: Yuning Jiang, Qiaoran Meng, Feiyang Shang, Nay Oo, Le Thi Hong Minh, Hoon Wei Lim, Biplab Sikdar

**Abstract**: The MITRE ATT&CK framework is a widely adopted tool for enhancing cybersecurity, supporting threat intelligence, incident response, attack modeling, and vulnerability prioritization. This paper synthesizes research on its application across these domains by analyzing 417 peer-reviewed publications. We identify commonly used adversarial tactics, techniques, and procedures (TTPs) and examine the integration of natural language processing (NLP) and machine learning (ML) with ATT&CK to improve threat detection and response. Additionally, we explore the interoperability of ATT&CK with other frameworks, such as the Cyber Kill Chain, NIST guidelines, and STRIDE, highlighting its versatility. The paper further evaluates the framework from multiple perspectives, including its effectiveness, validation methods, and sector-specific challenges, particularly in industrial control systems (ICS) and healthcare. We conclude by discussing current limitations and proposing future research directions to enhance the applicability of ATT&CK in dynamic cybersecurity environments.

摘要: MITRE ATT&CK框架是一种被广泛采用的工具，用于增强网络安全、支持威胁情报、事件响应、攻击建模和漏洞优先排序。本文通过对417篇同行评议出版物的分析，对其在这些领域的应用研究进行了综述。我们确定了常用的对抗战术、技术和过程(TTP)，并研究了自然语言处理(NLP)和机器学习(ML)与ATT&CK的集成，以改进威胁检测和响应。此外，我们还探讨了ATT和CK与其他框架的互操作性，如Cyber Kill Chain、NIST指南和STRIDE，突出了它的多功能性。白皮书进一步从多个角度对该框架进行了评估，包括其有效性、验证方法和特定行业的挑战，特别是在工业控制系统(ICS)和医疗保健方面。最后，我们讨论了目前的局限性，并提出了未来的研究方向，以增强ATT和CK在动态网络安全环境中的适用性。



## **21. Pixel Is Not a Barrier: An Effective Evasion Attack for Pixel-Domain Diffusion Models**

像素不是障碍：像素域扩散模型的有效规避攻击 cs.CV

**SubmitDate**: 2025-02-15    [abs](http://arxiv.org/abs/2408.11810v3) [paper-pdf](http://arxiv.org/pdf/2408.11810v3)

**Authors**: Chun-Yen Shih, Li-Xuan Peng, Jia-Wei Liao, Ernie Chu, Cheng-Fu Chou, Jun-Cheng Chen

**Abstract**: Diffusion Models have emerged as powerful generative models for high-quality image synthesis, with many subsequent image editing techniques based on them. However, the ease of text-based image editing introduces significant risks, such as malicious editing for scams or intellectual property infringement. Previous works have attempted to safeguard images from diffusion-based editing by adding imperceptible perturbations. These methods are costly and specifically target prevalent Latent Diffusion Models (LDMs), while Pixel-domain Diffusion Models (PDMs) remain largely unexplored and robust against such attacks. Our work addresses this gap by proposing a novel attack framework, AtkPDM. AtkPDM is mainly composed of a feature representation attacking loss that exploits vulnerabilities in denoising UNets and a latent optimization strategy to enhance the naturalness of adversarial images. Extensive experiments demonstrate the effectiveness of our approach in attacking dominant PDM-based editing methods (e.g., SDEdit) while maintaining reasonable fidelity and robustness against common defense methods. Additionally, our framework is extensible to LDMs, achieving comparable performance to existing approaches.

摘要: 扩散模型已经成为高质量图像合成的强大生成性模型，许多后续的图像编辑技术都是基于扩散模型的。然而，基于文本的图像编辑的简便性带来了重大风险，例如用于欺诈或侵犯知识产权的恶意编辑。以前的工作试图通过添加不可察觉的扰动来保护图像免受基于扩散的编辑。这些方法昂贵且专门针对流行的潜在扩散模型(LDM)，而像素域扩散模型(PDMS)在很大程度上仍未被探索，并且对此类攻击具有健壮性。我们的工作通过提出一种新颖的攻击框架AtkPDM来解决这一问题。该算法主要由特征表示、攻击损失和潜在优化策略两部分组成，前者利用UNNet的去噪漏洞，后者增强敌方图像的自然度。大量实验表明，该方法在攻击主流的基于产品数据管理的编辑方法(如SDEDIT)的同时，对常见的防御方法保持了合理的保真度和健壮性。此外，我们的框架可扩展到LDM，实现了与现有方法相当的性能。



## **22. Robustness-aware Automatic Prompt Optimization**

具有鲁棒性的自动提示优化 cs.CL

**SubmitDate**: 2025-02-15    [abs](http://arxiv.org/abs/2412.18196v2) [paper-pdf](http://arxiv.org/pdf/2412.18196v2)

**Authors**: Zeru Shi, Zhenting Wang, Yongye Su, Weidi Luo, Hang Gao, Fan Yang, Ruixiang Tang, Yongfeng Zhang

**Abstract**: The performance of Large Language Models (LLMs) depends on the quality of prompts and the semantic and structural integrity of the input data. However, existing prompt generation methods primarily focus on well-structured input data, often neglecting the impact of perturbed inputs on prompt effectiveness. To address this limitation, we propose BATprompt (By Adversarial Training prompt), a novel method for prompt generation designed to withstand input perturbations (such as typos in the input). Inspired by adversarial training techniques, BATprompt demonstrates strong performance on a variety of perturbed tasks through a two-step process: adversarial perturbation and iterative optimization on unperturbed input via LLM. Unlike conventional adversarial attack methods, BATprompt does not need access to model parameters and gradients. Instead, BATprompt leverages the advanced reasoning, language understanding and self reflection capabilities of LLMs to simulate gradients, guiding the generation of adversarial perturbations and optimizing prompt performance. We evaluate BATprompt on multiple datasets across both language understanding and generation tasks. The results indicate that BATprompt outperforms existing prompt generation methods, delivering superior robustness and performance under diverse perturbation scenarios.

摘要: 大型语言模型(LLM)的性能取决于提示的质量以及输入数据的语义和结构完整性。然而，现有的提示生成方法主要关注结构良好的输入数据，往往忽略了扰动输入对提示效果的影响。为了解决这一局限性，我们提出了一种新的提示生成方法BATprint(通过对抗性训练提示)，该方法旨在抵抗输入扰动(如输入中的打字错误)。受到对抗性训练技术的启发，通过两步过程：对抗性扰动和通过LLM对不受扰动的输入进行迭代优化，BATprint在各种扰动任务上表现出了强大的性能。与传统的对抗性攻击方法不同，BATprint不需要访问模型参数和梯度。相反，BATprint利用LLMS的高级推理、语言理解和自我反思能力来模拟梯度，指导产生对抗性扰动并优化提示性能。我们在语言理解和生成任务的多个数据集上评估BATprint。结果表明，BATprint的性能优于现有的提示生成方法，在不同的扰动场景下都具有较好的健壮性和性能。



## **23. Learning to Rewrite: Generalized LLM-Generated Text Detection**

学习重写：广义LLM生成的文本检测 cs.CL

**SubmitDate**: 2025-02-15    [abs](http://arxiv.org/abs/2408.04237v2) [paper-pdf](http://arxiv.org/pdf/2408.04237v2)

**Authors**: Ran Li, Wei Hao, Weiliang Zhao, Junfeng Yang, Chengzhi Mao

**Abstract**: Large language models (LLMs) present significant risks when used to generate non-factual content and spread disinformation at scale. Detecting such LLM-generated content is crucial, yet current detectors often struggle to generalize in open-world contexts. We introduce Learning2Rewrite, a novel framework for detecting AI-generated text with exceptional generalization to unseen domains. Our method leverages the insight that LLMs inherently modify AI-generated content less than human-written text when tasked with rewriting. By training LLMs to minimize alterations on AI-generated inputs, we amplify this disparity, yielding a more distinguishable and generalizable edit distance across diverse text distributions. Extensive experiments on data from 21 independent domains and four major LLMs (GPT-3.5, GPT-4, Gemini, and Llama-3) demonstrate that our detector outperforms state-of-the-art detection methods by up to 23.04% in AUROC for in-distribution tests, 37.26% for out-of-distribution tests, and 48.66% under adversarial attacks. Our unique training objective ensures better generalizability compared to directly training for classification, when leveraging the same amount of parameters. Our findings suggest that reinforcing LLMs' inherent rewriting tendencies offers a robust and scalable solution for detecting AI-generated text.

摘要: 大型语言模型(LLM)在用于生成非事实内容和大规模传播虚假信息时会带来重大风险。检测这种LLM生成的内容是至关重要的，但目前的检测器往往难以在开放世界的环境中进行推广。我们介绍了Learning2Rewrite，一个新的框架，用于检测人工智能生成的文本，具有对不可见领域的特殊泛化。我们的方法利用了这样一种见解，即当执行重写任务时，LLMS天生就不会修改人工智能生成的内容，而不是人类编写的文本。通过训练LLM以最大限度地减少人工智能生成的输入的更改，我们放大了这种差异，在不同的文本分布上产生了更可区分和更具普遍性的编辑距离。在21个独立域和四个主要LLM(GPT-3.5、GPT-4、Gemini和Llama-3)上的大量实验表明，对于分布内测试，我们的检测器在AUROC上的性能比最先进的检测方法高23.04%，对于分布外测试，我们的检测器性能高达37.26%，在对手攻击下，我们的检测器性能高达48.66%。我们独特的训练目标确保了在利用相同数量的参数时，与直接用于分类的训练相比，具有更好的通用性。我们的发现表明，加强LMS固有的重写倾向为检测人工智能生成的文本提供了一种健壮和可扩展的解决方案。



## **24. Random-Set Neural Networks (RS-NN)**

随机集神经网络（RS-NN） cs.LG

Published as a conference paper at the Thirteenth International  Conference on Learning Representations (ICLR 2025)

**SubmitDate**: 2025-02-14    [abs](http://arxiv.org/abs/2307.05772v5) [paper-pdf](http://arxiv.org/pdf/2307.05772v5)

**Authors**: Shireen Kudukkil Manchingal, Muhammad Mubashar, Kaizheng Wang, Keivan Shariatmadar, Fabio Cuzzolin

**Abstract**: Machine learning is increasingly deployed in safety-critical domains where erroneous predictions may lead to potentially catastrophic consequences, highlighting the need for learning systems to be aware of how confident they are in their own predictions: in other words, 'to know when they do not know'. In this paper, we propose a novel Random-Set Neural Network (RS-NN) approach to classification which predicts belief functions (rather than classical probability vectors) over the class list using the mathematics of random sets, i.e., distributions over the collection of sets of classes. RS-NN encodes the 'epistemic' uncertainty induced by training sets that are insufficiently representative or limited in size via the size of the convex set of probability vectors associated with a predicted belief function. Our approach outperforms state-of-the-art Bayesian and Ensemble methods in terms of accuracy, uncertainty estimation and out-of-distribution (OoD) detection on multiple benchmarks (CIFAR-10 vs SVHN/Intel-Image, MNIST vs FMNIST/KMNIST, ImageNet vs ImageNet-O). RS-NN also scales up effectively to large-scale architectures (e.g. WideResNet-28-10, VGG16, Inception V3, EfficientNetB2 and ViT-Base-16), exhibits remarkable robustness to adversarial attacks and can provide statistical guarantees in a conformal learning setting.

摘要: 机器学习越来越多地部署在安全关键领域，在这些领域，错误的预测可能会导致潜在的灾难性后果，这突显了学习系统需要意识到自己对自己的预测有多自信：换句话说，“知道什么时候他们不知道”。在本文中，我们提出了一种新的随机集神经网络(RS-NN)分类方法，它使用随机集的数学，即在类集合集合上的分布来预测类列表上的信任函数(而不是经典的概率向量)。RS-NN通过与预测的信任函数相关联的概率向量凸集的大小来编码不具有足够代表性或大小受限的训练集所引起的“认知”不确定性。在多个基准测试(CIFAR-10与SVHN/Intel-Image、MNIST与FMNIST/KMNIST、ImageNet与ImageNet-O)上，我们的方法在准确性、不确定性估计和OOD检测方面优于最先进的贝叶斯和集成方法。RS-NN还可以有效地扩展到大规模体系结构(如WideResNet-28-10、VGG16、初始V3、EfficientNetB2和Vit-Base-16)，对对手攻击表现出显著的鲁棒性，并可以在共形学习环境下提供统计保证。



## **25. VT-GAN: Cooperative Tabular Data Synthesis using Vertical Federated Learning**

VT-GAN：使用垂直联邦学习的协作表格数据合成 cs.LG

**SubmitDate**: 2025-02-14    [abs](http://arxiv.org/abs/2302.01706v2) [paper-pdf](http://arxiv.org/pdf/2302.01706v2)

**Authors**: Zilong Zhao, Han Wu, Aad Van Moorsel, Lydia Y. Chen

**Abstract**: This paper presents the application of Vertical Federated Learning (VFL) to generate synthetic tabular data using Generative Adversarial Networks (GANs). VFL is a collaborative approach to train machine learning models among distinct tabular data holders, such as financial institutions, who possess disjoint features for the same group of customers. In this paper we introduce the VT-GAN framework, Vertical federated Tabular GAN, and demonstrate that VFL can be successfully used to implement GANs for distributed tabular data in privacy-preserving manner, with performance close to centralized GANs that assume shared data. We make design choices with respect to the distribution of GAN generator and discriminator models and introduce a training-with-shuffling technique so that no party can reconstruct training data from the GAN conditional vector. The paper presents (1) an implementation of VT-GAN, (2) a detailed quality evaluation of the VT-GAN-generated synthetic data, (3) an overall scalability examination of VT-GAN framework, (4) a security analysis on VT-GAN's robustness against Membership Inference Attack with different settings of Differential Privacy, for a range of datasets with diverse distribution characteristics. Our results demonstrate that VT-GAN can consistently generate high-fidelity synthetic tabular data of comparable quality to that generated by a centralized GAN algorithm. The difference in machine learning utility can be as low as 2.7%, even under extremely imbalanced data distributions across clients or with different numbers of clients.

摘要: 介绍了垂直联合学习(VFL)在生成对抗网络(GANS)生成合成表格数据中的应用。VFL是一种协作方法，用于在不同的表格数据持有者(如金融机构)之间训练机器学习模型，这些持有者拥有针对同一客户组的不连续特征。本文介绍了VT-GAN框架--垂直联合表格化GAN，并证明了VFL可以成功地用来以隐私保护的方式实现分布式表格数据的GAN，其性能接近于假设共享数据的集中式GAN。我们针对GaN生成器和鉴别器模型的分布进行了设计选择，并引入了带混洗的训练技术，使得任何一方都不能从GaN条件向量中重建训练数据。本文提出了(1)VT-GAN的实现，(2)对VT-GAN生成的合成数据进行了详细的质量评估，(3)对VT-GAN框架进行了全面的可扩展性检验，(4)针对一系列具有不同分布特征的数据集，对VT-GAN在不同的差分隐私设置下对成员推理攻击的健壮性进行了安全分析。我们的结果表明，VT-GaN可以一致地生成高保真的合成表格数据，其质量与集中式GAN算法生成的数据相当。机器学习效用的差异可以低至2.7%，即使在客户端之间的数据分布极其不平衡或客户端数量不同的情况下也是如此。



## **26. Wolfpack Adversarial Attack for Robust Multi-Agent Reinforcement Learning**

用于鲁棒多智能体强化学习的Wolfpack对抗攻击 cs.LG

8 pages main, 21 pages appendix with reference. Submitted to ICML  2025

**SubmitDate**: 2025-02-14    [abs](http://arxiv.org/abs/2502.02844v2) [paper-pdf](http://arxiv.org/pdf/2502.02844v2)

**Authors**: Sunwoo Lee, Jaebak Hwang, Yonghyeon Jo, Seungyul Han

**Abstract**: Traditional robust methods in multi-agent reinforcement learning (MARL) often struggle against coordinated adversarial attacks in cooperative scenarios. To address this limitation, we propose the Wolfpack Adversarial Attack framework, inspired by wolf hunting strategies, which targets an initial agent and its assisting agents to disrupt cooperation. Additionally, we introduce the Wolfpack-Adversarial Learning for MARL (WALL) framework, which trains robust MARL policies to defend against the proposed Wolfpack attack by fostering system-wide collaboration. Experimental results underscore the devastating impact of the Wolfpack attack and the significant robustness improvements achieved by WALL.

摘要: 多智能体强化学习（MARL）中的传统稳健方法经常难以应对合作场景中的协调对抗攻击。为了解决这一局限性，我们提出了狼群对抗攻击框架，该框架受到猎狼策略的启发，该框架针对初始代理及其辅助代理来破坏合作。此外，我们还引入了Wolfpack对抗学习for MARL（WALL）框架，该框架训练强大的MARL策略，以通过促进系统范围的协作来抵御拟议的Wolfpack攻击。实验结果强调了Wolfpack攻击的毁灭性影响以及WALL实现的显着鲁棒性改进。



## **27. SWAP Attack: Stealthy Side-Channel Attack on Multi-Tenant Quantum Cloud System**

SWAP攻击：对多租户量子云系统的隐形侧通道攻击 quant-ph

**SubmitDate**: 2025-02-14    [abs](http://arxiv.org/abs/2502.10115v1) [paper-pdf](http://arxiv.org/pdf/2502.10115v1)

**Authors**: Wei Jie Bryan Lee, Siyi Wang, Suman Dutta, Walid El Maouaki, Anupam Chattopadhyay

**Abstract**: The rapid advancement of quantum computing has spurred widespread adoption, with cloud-based quantum devices gaining traction in academia and industry. This shift raises critical concerns about the privacy and security of computations on shared, multi-tenant quantum platforms accessed remotely. Recent studies have shown that crosstalk on shared quantum devices allows adversaries to interfere with victim circuits within a neighborhood. While insightful, these works left unresolved questions regarding the root cause of crosstalk, effective countermeasures, and replicability across circuits. We revisit the crosstalk effect, tracing its origins to the SWAP path between qubits and demonstrating its impact even over long distances. Our results significantly improve the understanding of this phenomenon beyond prior works. The proposed SWAP-based side-channel attack operates in both active and passive modes, as verified on real IBM quantum devices. In the active attack, an attacker executing a single CNOT gate can perturb victim circuits running Grover's Algorithm, reducing expected output accuracy by $81.62\%$ through strategic qubit placement. Moreover, this effect can be modeled to identify qubits more susceptible to attack. The passive attack, leveraging a stealthy circuit as small as $6.25\%$ of the victim's, achieves $100\%$ accuracy in predicting the victim's circuit size when running Simon's Algorithm. These findings challenge the existing defense strategy of maximizing topological distance between circuits, showing that attackers can still extract sensitive information or manipulate results remotely. Our work highlights the urgent need for robust security measures to safeguard quantum computations against emerging threats.

摘要: 量子计算的快速发展刺激了人们的广泛采用，基于云的量子设备在学术界和工业界获得了吸引力。这种转变引发了人们对远程访问的共享多租户量子平台上计算的隐私和安全的严重担忧。最近的研究表明，共享量子设备上的串扰允许攻击者干扰邻近区域内的受害者电路。虽然这些工作很有洞察力，但关于串扰的根本原因、有效的对策和跨电路的可复制性等问题仍未解决。我们回顾了串扰效应，追溯了它的起源，追溯到量子比特之间的交换路径，并展示了它甚至在长距离上的影响。我们的结果大大提高了对这一现象的理解，超过了以前的工作。所提出的基于交换的侧通道攻击可以在主动和被动模式下运行，在真实的IBM量子设备上得到了验证。在主动攻击中，攻击者执行单个CNOT门可以扰乱运行Grover算法的受害者电路，通过战略性的量子比特放置使期望的输出精度降低81.62美元。此外，可以对这种效应进行建模，以确定更容易受到攻击的量子比特。被动攻击利用一个小到受害者的6.25美元的隐形电路，在运行Simon的算法时实现了100美元的预测受害者电路大小的准确性。这些发现挑战了现有的最大化电路之间拓扑距离的防御策略，表明攻击者仍然可以远程提取敏感信息或操纵结果。我们的工作突显了迫切需要强有力的安全措施来保护量子计算免受新出现的威胁。



## **28. Fast Proxies for LLM Robustness Evaluation**

LLM稳健性评估的快速代理 cs.CR

**SubmitDate**: 2025-02-14    [abs](http://arxiv.org/abs/2502.10487v1) [paper-pdf](http://arxiv.org/pdf/2502.10487v1)

**Authors**: Tim Beyer, Jan Schuchardt, Leo Schwinn, Stephan Günnemann

**Abstract**: Evaluating the robustness of LLMs to adversarial attacks is crucial for safe deployment, yet current red-teaming methods are often prohibitively expensive. We compare the ability of fast proxy metrics to predict the real-world robustness of an LLM against a simulated attacker ensemble. This allows us to estimate a model's robustness to computationally expensive attacks without requiring runs of the attacks themselves. Specifically, we consider gradient-descent-based embedding-space attacks, prefilling attacks, and direct prompting. Even though direct prompting in particular does not achieve high ASR, we find that it and embedding-space attacks can predict attack success rates well, achieving $r_p=0.87$ (linear) and $r_s=0.94$ (Spearman rank) correlations with the full attack ensemble while reducing computational cost by three orders of magnitude.

摘要: 评估LLM对对抗攻击的稳健性对于安全部署至关重要，但当前的红色团队方法通常昂贵得令人望而却步。我们比较了快速代理指标预测LLM与模拟攻击者集成的现实鲁棒性的能力。这使我们能够估计模型对计算昂贵的攻击的稳健性，而无需运行攻击本身。具体来说，我们考虑基于梯度下降的嵌入空间攻击、预填充攻击和直接提示。尽管直接提示不能实现高的ASB，但我们发现它和嵌入空间攻击可以很好地预测攻击成功率，实现与完整攻击集成的$r_p=0.87$（线性）和$r_s=0.94$（斯皮尔曼等级）相关性，同时将计算成本降低三个数量级。



## **29. ASVspoof 5: Design, Collection and Validation of Resources for Spoofing, Deepfake, and Adversarial Attack Detection Using Crowdsourced Speech**

ASVspoof 5：使用众包语音设计、收集和验证用于欺骗、Deepfake和对抗性攻击检测的资源 eess.AS

Database link: https://zenodo.org/records/14498691, Database mirror  link: https://huggingface.co/datasets/jungjee/asvspoof5, ASVspoof 5 Challenge  Workshop Proceeding: https://www.isca-archive.org/asvspoof_2024/index.html

**SubmitDate**: 2025-02-14    [abs](http://arxiv.org/abs/2502.08857v2) [paper-pdf](http://arxiv.org/pdf/2502.08857v2)

**Authors**: Xin Wang, Héctor Delgado, Hemlata Tak, Jee-weon Jung, Hye-jin Shim, Massimiliano Todisco, Ivan Kukanov, Xuechen Liu, Md Sahidullah, Tomi Kinnunen, Nicholas Evans, Kong Aik Lee, Junichi Yamagishi, Myeonghun Jeong, Ge Zhu, Yongyi Zang, You Zhang, Soumi Maiti, Florian Lux, Nicolas Müller, Wangyou Zhang, Chengzhe Sun, Shuwei Hou, Siwei Lyu, Sébastien Le Maguer, Cheng Gong, Hanjie Guo, Liping Chen, Vishwanath Singh

**Abstract**: ASVspoof 5 is the fifth edition in a series of challenges which promote the study of speech spoofing and deepfake attacks as well as the design of detection solutions. We introduce the ASVspoof 5 database which is generated in crowdsourced fashion from data collected in diverse acoustic conditions (cf. studio-quality data for earlier ASVspoof databases) and from ~2,000 speakers (cf. ~100 earlier). The database contains attacks generated with 32 different algorithms, also crowdsourced, and optimised to varying degrees using new surrogate detection models. Among them are attacks generated with a mix of legacy and contemporary text-to-speech synthesis and voice conversion models, in addition to adversarial attacks which are incorporated for the first time. ASVspoof 5 protocols comprise seven speaker-disjoint partitions. They include two distinct partitions for the training of different sets of attack models, two more for the development and evaluation of surrogate detection models, and then three additional partitions which comprise the ASVspoof 5 training, development and evaluation sets. An auxiliary set of data collected from an additional 30k speakers can also be used to train speaker encoders for the implementation of attack algorithms. Also described herein is an experimental validation of the new ASVspoof 5 database using a set of automatic speaker verification and spoof/deepfake baseline detectors. With the exception of protocols and tools for the generation of spoofed/deepfake speech, the resources described in this paper, already used by participants of the ASVspoof 5 challenge in 2024, are now all freely available to the community.

摘要: ASVspoof5是一系列挑战中的第五版，这些挑战促进了对语音欺骗和深度假冒攻击的研究以及检测解决方案的设计。我们介绍了ASVspoof5数据库，它是以众包方式从不同声学条件下收集的数据生成的(参见。较早ASVspoof数据库的演播室质量数据)和来自约2,000名演讲者的数据(参见~100早)。该数据库包含32种不同的算法产生的攻击，也是众包的，并使用新的代理检测模型在不同程度上进行了优化。其中包括使用传统和当代文本到语音合成和语音转换模型的混合生成的攻击，以及首次纳入的对抗性攻击。ASVspoof 5协议包括七个说话人不相交的分区。它们包括用于训练不同攻击模型集的两个不同的分区，用于开发和评估代理检测模型的另外两个分区，以及组成ASVspoof5训练、开发和评估集的另外三个分区。从另外30,000个说话者收集的辅助数据集也可用于训练说话人编码者以实现攻击算法。这里还描述了使用一组自动说话人验证和欺骗/深度伪基线检测器对新的ASVspoof5数据库进行的实验验证。除了用于生成欺骗/深度假语音的协议和工具外，本文描述的资源现在都可以免费向社区提供，这些资源已经被2024年ASVspoof 5挑战赛的参与者使用。



## **30. What You See Is Not Always What You Get: An Empirical Study of Code Comprehension by Large Language Models**

你所看到的并不总是你所得到的：大型语言模型对代码理解的实证研究 cs.SE

**SubmitDate**: 2025-02-14    [abs](http://arxiv.org/abs/2412.08098v2) [paper-pdf](http://arxiv.org/pdf/2412.08098v2)

**Authors**: Bangshuo Zhu, Jiawen Wen, Huaming Chen

**Abstract**: Recent studies have demonstrated outstanding capabilities of large language models (LLMs) in software engineering tasks, including code generation and comprehension. While LLMs have shown significant potential in assisting with coding, it is perceived that LLMs are vulnerable to adversarial attacks. In this paper, we investigate the vulnerability of LLMs to imperceptible attacks, where hidden character manipulation in source code misleads LLMs' behaviour while remaining undetectable to human reviewers. We devise these attacks into four distinct categories and analyse their impacts on code analysis and comprehension tasks. These four types of imperceptible coding character attacks include coding reordering, invisible coding characters, code deletions, and code homoglyphs. To comprehensively benchmark the robustness of current LLMs solutions against the attacks, we present a systematic experimental evaluation on multiple state-of-the-art LLMs. Our experimental design introduces two key performance metrics, namely model confidence using log probabilities of response, and the response correctness. A set of controlled experiments are conducted using a large-scale perturbed and unperturbed code snippets as the primary prompt input. Our findings confirm the susceptibility of LLMs to imperceptible coding character attacks, while different LLMs present different negative correlations between perturbation magnitude and performance. These results highlight the urgent need for robust LLMs capable of manoeuvring behaviours under imperceptible adversarial conditions. We anticipate this work provides valuable insights for enhancing the security and trustworthiness of LLMs in software engineering applications.

摘要: 最近的研究表明，大型语言模型(LLM)在软件工程任务中具有出色的能力，包括代码生成和理解。虽然LLM在协助编码方面显示出巨大的潜力，但人们认为LLM容易受到对手的攻击。在本文中，我们研究了LLMS对不可察觉的攻击的脆弱性，即源代码中的隐藏字符操作误导了LLMS的行为，而人类评审者仍然无法检测到。我们将这些攻击分为四个不同的类别，并分析它们对代码分析和理解任务的影响。这四种类型的不可察觉编码字符攻击包括编码重新排序、不可见编码字符、代码删除和代码同形。为了全面衡量现有LLMS解决方案对攻击的健壮性，我们对多个最先进的LLMS进行了系统的实验评估。我们的实验设计引入了两个关键的性能度量，即使用响应日志概率的模型置信度和响应正确性。使用大规模的扰动和未扰动的代码片段作为主要提示输入，进行了一组对照实验。我们的研究结果证实了LLMS对不可察觉的编码字符攻击的敏感性，而不同的LLM在扰动大小与性能之间呈现不同的负相关。这些结果突出表明，迫切需要能够在难以察觉的对抗性条件下操纵行为的强大的LLM。我们期待这项工作为在软件工程应用中增强LLMS的安全性和可信性提供有价值的见解。



## **31. Siren Song: Manipulating Pose Estimation in XR Headsets Using Acoustic Attacks**

Siren Song：使用声学攻击操纵XR耳机中的姿势估计 cs.CR

**SubmitDate**: 2025-02-14    [abs](http://arxiv.org/abs/2502.08865v2) [paper-pdf](http://arxiv.org/pdf/2502.08865v2)

**Authors**: Zijian Huang, Yicheng Zhang, Sophie Chen, Nael Abu-Ghazaleh, Jiasi Chen

**Abstract**: Extended Reality (XR) experiences involve interactions between users, the real world, and virtual content. A key step to enable these experiences is the XR headset sensing and estimating the user's pose in order to accurately place and render virtual content in the real world. XR headsets use multiple sensors (e.g., cameras, inertial measurement unit) to perform pose estimation and improve its robustness, but this provides an attack surface for adversaries to interfere with the pose estimation process. In this paper, we create and study the effects of acoustic attacks that create false signals in the inertial measurement unit (IMU) on XR headsets, leading to adverse downstream effects on XR applications. We generate resonant acoustic signals on a HoloLens 2 and measure the resulting perturbations in the IMU readings, and also demonstrate both fine-grained and coarse attacks on the popular ORB-SLAM3 and an open-source XR system (ILLIXR). With the knowledge gleaned from attacking these open-source frameworks, we demonstrate four end-to-end proof-of-concept attacks on a HoloLens 2: manipulating user input, clickjacking, zone invasion, and denial of user interaction. Our experiments show that current commercial XR headsets are susceptible to acoustic attacks, raising concerns for their security.

摘要: 扩展现实(XR)体验涉及用户、真实世界和虚拟内容之间的交互。实现这些体验的关键一步是XR耳机感知和估计用户的姿势，以便在现实世界中准确放置和呈现虚拟内容。XR耳机使用多个传感器(如摄像头、惯性测量单元)来执行位姿估计并提高其稳健性，但这为对手提供了一个干扰位姿估计过程的攻击面。在本文中，我们创建和研究了在XR耳机上的惯性测量单元(IMU)中产生错误信号的声学攻击的影响，从而导致XR应用的不利下游影响。我们在HoloLens 2上生成共振声信号，并测量IMU读数中产生的扰动，还演示了对流行的Orb-SLAM3和开放源代码XR系统(ILLIXR)的细粒度和粗粒度攻击。利用从攻击这些开源框架中获得的知识，我们演示了针对HoloLens 2的四种端到端概念验证攻击：操纵用户输入、点击劫持、区域入侵和拒绝用户交互。我们的实验表明，目前的商用XR耳机容易受到声学攻击，这引发了人们对其安全性的担忧。



## **32. Towards Reliable Empirical Machine Unlearning Evaluation: A Cryptographic Game Perspective**

迈向可靠的经验机器不学习评估：密码游戏的角度 cs.LG

**SubmitDate**: 2025-02-14    [abs](http://arxiv.org/abs/2404.11577v3) [paper-pdf](http://arxiv.org/pdf/2404.11577v3)

**Authors**: Yiwen Tu, Pingbang Hu, Jiaqi Ma

**Abstract**: Machine unlearning updates machine learning models to remove information from specific training samples, complying with data protection regulations that allow individuals to request the removal of their personal data. Despite the recent development of numerous unlearning algorithms, reliable evaluation of these algorithms remains an open research question. In this work, we focus on membership inference attack (MIA) based evaluation, one of the most common approaches for evaluating unlearning algorithms, and address various pitfalls of existing evaluation metrics lacking theoretical understanding and reliability. Specifically, by modeling the proposed evaluation process as a \emph{cryptographic game} between unlearning algorithms and MIA adversaries, the naturally-induced evaluation metric measures the data removal efficacy of unlearning algorithms and enjoys provable guarantees that existing evaluation metrics fail to satisfy. Furthermore, we propose a practical and efficient approximation of the induced evaluation metric and demonstrate its effectiveness through both theoretical analysis and empirical experiments. Overall, this work presents a novel and reliable approach to empirically evaluating unlearning algorithms, paving the way for the development of more effective unlearning techniques.

摘要: 机器遗忘更新机器学习模型，以从特定训练样本中删除信息，遵守允许个人请求删除其个人数据的数据保护法规。尽管最近有许多遗忘算法的发展，但对这些算法的可靠评估仍然是一个开放的研究问题。在这项工作中，我们专注于基于成员关系推理攻击(MIA)的评估，这是评估遗忘算法最常见的方法之一，并解决了现有评估指标缺乏理论理解和可靠性的各种缺陷。具体地，通过将所提出的评估过程建模为遗忘算法与MIA对手之间的密码博弈，自然诱导的评估度量度量遗忘算法的数据去除效率，并享有现有评估度量不能满足的可证明的保证。此外，我们还提出了一种实用而有效的近似方法，并通过理论分析和实验验证了该方法的有效性。总体而言，这项工作提出了一种新颖而可靠的方法来对遗忘算法进行经验评估，为开发更有效的遗忘技术铺平了道路。



## **33. $\textrm{A}^{\textrm{2}}$RNet: Adversarial Attack Resilient Network for Robust Infrared and Visible Image Fusion**

$\textrm{A}^{\textrm{2}}$RNet：用于鲁棒的红外和可见图像融合的对抗攻击弹性网络 cs.CV

9 pages, 8 figures, The 39th Annual AAAI Conference on Artificial  Intelligence

**SubmitDate**: 2025-02-14    [abs](http://arxiv.org/abs/2412.09954v3) [paper-pdf](http://arxiv.org/pdf/2412.09954v3)

**Authors**: Jiawei Li, Hongwei Yu, Jiansheng Chen, Xinlong Ding, Jinlong Wang, Jinyuan Liu, Bochao Zou, Huimin Ma

**Abstract**: Infrared and visible image fusion (IVIF) is a crucial technique for enhancing visual performance by integrating unique information from different modalities into one fused image. Exiting methods pay more attention to conducting fusion with undisturbed data, while overlooking the impact of deliberate interference on the effectiveness of fusion results. To investigate the robustness of fusion models, in this paper, we propose a novel adversarial attack resilient network, called $\textrm{A}^{\textrm{2}}$RNet. Specifically, we develop an adversarial paradigm with an anti-attack loss function to implement adversarial attacks and training. It is constructed based on the intrinsic nature of IVIF and provide a robust foundation for future research advancements. We adopt a Unet as the pipeline with a transformer-based defensive refinement module (DRM) under this paradigm, which guarantees fused image quality in a robust coarse-to-fine manner. Compared to previous works, our method mitigates the adverse effects of adversarial perturbations, consistently maintaining high-fidelity fusion results. Furthermore, the performance of downstream tasks can also be well maintained under adversarial attacks. Code is available at https://github.com/lok-18/A2RNet.

摘要: 红外与可见光图像融合(IVIF)是通过将来自不同模式的独特信息融合到一幅融合图像中来提高视觉性能的关键技术。现有的方法更注重对未受干扰的数据进行融合，而忽略了有意干扰对融合结果有效性的影响。为了研究融合模型的稳健性，本文提出了一种新的对抗攻击弹性网络，称为$\tExtm{A}^{\tExtm{2}}$rnet。具体地说，我们开发了一个具有抗攻击损失函数的对抗性范例来实施对抗性攻击和训练。它是基于IVIF的内在本质而构建的，并为未来的研究进展提供了坚实的基础。在该模型下，我们采用了基于变换的防御性细化模块(DRM)作为流水线，保证了从粗到精的融合图像质量。与以前的工作相比，我们的方法减轻了对抗性扰动的不利影响，一致地保持了高保真的融合结果。此外，在对抗性攻击下，下游任务的性能也能得到很好的维持。代码可在https://github.com/lok-18/A2RNet.上找到



## **34. LiSA: Leveraging Link Recommender to Attack Graph Neural Networks via Subgraph Injection**

LiSA：利用链接推荐通过子图注入攻击图神经网络 cs.LG

PAKDD 2025

**SubmitDate**: 2025-02-14    [abs](http://arxiv.org/abs/2502.09271v2) [paper-pdf](http://arxiv.org/pdf/2502.09271v2)

**Authors**: Wenlun Zhang, Enyan Dai, Kentaro Yoshioka

**Abstract**: Graph Neural Networks (GNNs) have demonstrated remarkable proficiency in modeling data with graph structures, yet recent research reveals their susceptibility to adversarial attacks. Traditional attack methodologies, which rely on manipulating the original graph or adding links to artificially created nodes, often prove impractical in real-world settings. This paper introduces a novel adversarial scenario involving the injection of an isolated subgraph to deceive both the link recommender and the node classifier within a GNN system. Specifically, the link recommender is mislead to propose links between targeted victim nodes and the subgraph, encouraging users to unintentionally establish connections and that would degrade the node classification accuracy, thereby facilitating a successful attack. To address this, we present the LiSA framework, which employs a dual surrogate model and bi-level optimization to simultaneously meet two adversarial objectives. Extensive experiments on real-world datasets demonstrate the effectiveness of our method.

摘要: 图神经网络(GNN)在用图结构建模数据方面表现出了卓越的能力，但最近的研究表明，它们对对手攻击很敏感。传统的攻击方法依赖于操纵原始图形或添加到人工创建的节点的链接，在现实世界中往往被证明是不切实际的。在GNN系统中，引入一个孤立的子图来欺骗链接推荐器和节点分类器，提出了一种新的对抗性场景。具体地说，链接推荐器被误导提出目标受害节点与子图之间的链接，鼓励用户无意中建立连接，这将降低节点分类的准确性，从而促进攻击的成功。为了解决这一问题，我们提出了LISA框架，该框架采用双重代理模型和双层优化来同时满足两个对抗性目标。在真实数据集上的大量实验证明了该方法的有效性。



## **35. SoK: State of the time: On Trustworthiness of Digital Clocks**

SoK：时代状况：数字时钟的可信度 cs.CR

**SubmitDate**: 2025-02-14    [abs](http://arxiv.org/abs/2502.09837v1) [paper-pdf](http://arxiv.org/pdf/2502.09837v1)

**Authors**: Adeel Nasrullah, Fatima M. Anwar

**Abstract**: Despite the critical role of timing infrastructure in enabling essential services, from public key infrastructure and smart grids to autonomous navigation and high-frequency trading, modern timing stacks remain highly vulnerable to malicious attacks. These threats emerge due to several reasons, including inadequate security mechanisms, the timing architectures unique vulnerability to delays, and implementation issues. In this paper, we aim to obtain a holistic understanding of the issues that make the timing stacks vulnerable to adversarial manipulations, what the challenges are in securing them, and what solutions can be borrowed from the research community to address them. To this end, we perform a systematic analysis of the security vulnerabilities of the timing stack. In doing so, we discover new attack surfaces, i.e., physical timing components and on-device timekeeping, which are often overlooked by existing research that predominantly studies the security of time synchronization protocols. We also show that the emerging trusted timing architectures are flawed and risk compromising wider system security, and propose an alternative design using hardware-software co-design.

摘要: 尽管计时基础设施在实现基本服务方面发挥了关键作用，从公钥基础设施和智能电网到自主导航和高频交易，但现代计时堆栈仍然非常容易受到恶意攻击。这些威胁的出现有几个原因，包括安全机制不充分、计时架构对延迟的独特脆弱性以及实施问题。在这篇论文中，我们的目标是全面了解使计时堆栈容易受到对手操纵的问题，确保它们的安全面临哪些挑战，以及可以从研究界借用什么解决方案来解决这些问题。为此，我们对时序堆栈的安全漏洞进行了系统的分析。在这样做的过程中，我们发现了新的攻击面，即物理计时组件和设备上的计时，而现有的主要研究时间同步协议安全性的研究往往忽略了这些方面。我们还指出了新兴的可信定时体系结构存在缺陷，有可能危及更广泛的系统安全，并提出了一种使用硬件-软件协同设计的替代设计。



## **36. `Do as I say not as I do': A Semi-Automated Approach for Jailbreak Prompt Attack against Multimodal LLMs**

“照我说的做，而不是照我做的做”：针对多模式LLM的越狱提示攻击的半自动方法 cs.CR

**SubmitDate**: 2025-02-13    [abs](http://arxiv.org/abs/2502.00735v2) [paper-pdf](http://arxiv.org/pdf/2502.00735v2)

**Authors**: Chun Wai Chiu, Linghan Huang, Bo Li, Huaming Chen

**Abstract**: Large Language Models (LLMs) have seen widespread applications across various domains due to their growing ability to process diverse types of input data, including text, audio, image and video. While LLMs have demonstrated outstanding performance in understanding and generating contexts for different scenarios, they are vulnerable to prompt-based attacks, which are mostly via text input. In this paper, we introduce the first voice-based jailbreak attack against multimodal LLMs, termed as Flanking Attack, which can process different types of input simultaneously towards the multimodal LLMs. Our work is motivated by recent advancements in monolingual voice-driven large language models, which have introduced new attack surfaces beyond traditional text-based vulnerabilities for LLMs. To investigate these risks, we examine the state-of-the-art multimodal LLMs, which can be accessed via different types of inputs such as audio input, focusing on how adversarial prompts can bypass its defense mechanisms. We propose a novel strategy, in which the disallowed prompt is flanked by benign, narrative-driven prompts. It is integrated in the Flanking Attack which attempts to humanizes the interaction context and execute the attack through a fictional setting. Further, to better evaluate the attack performance, we present a semi-automated self-assessment framework for policy violation detection. We demonstrate that Flanking Attack is capable of manipulating state-of-the-art LLMs into generating misaligned and forbidden outputs, which achieves an average attack success rate ranging from 0.67 to 0.93 across seven forbidden scenarios.

摘要: 大语言模型因其处理文本、音频、图像和视频等不同类型输入数据的能力日益增强，在各个领域得到了广泛的应用。虽然LLM在理解和生成不同场景的上下文方面表现出了出色的性能，但它们很容易受到基于提示的攻击，这些攻击主要是通过文本输入进行的。在本文中，我们介绍了第一个基于语音的针对多模式LLMS的越狱攻击，称为侧翼攻击，它可以同时处理针对多模式LLMS的不同类型的输入。我们的工作是受到单语言语音驱动的大型语言模型的最新进展的推动，这些模型在传统的基于文本的LLMS漏洞之外引入了新的攻击面。为了调查这些风险，我们研究了最先进的多模式LLMS，这些LLMS可以通过不同类型的输入(如音频输入)访问，重点关注对抗性提示如何绕过其防御机制。我们提出了一种新的策略，在不允许的提示的两侧是良性的、叙事驱动的提示。它被整合到侧翼攻击中，试图使交互上下文人性化，并通过虚构的设置执行攻击。此外，为了更好地评估攻击性能，我们提出了一个半自动的策略违规检测自我评估框架。我们证明了侧翼攻击能够操纵最先进的LLM产生未对齐和禁止的输出，在七个禁止场景中获得了从0.67到0.93的平均攻击成功率。



## **37. Enhancing Jailbreak Attacks via Compliance-Refusal-Based Initialization**

通过基于合规拒绝的收件箱增强越狱攻击 cs.CR

**SubmitDate**: 2025-02-13    [abs](http://arxiv.org/abs/2502.09755v1) [paper-pdf](http://arxiv.org/pdf/2502.09755v1)

**Authors**: Amit Levi, Rom Himelstein, Yaniv Nemcovsky, Avi Mendelson, Chaim Baskin

**Abstract**: Jailbreak attacks aim to exploit large language models (LLMs) and pose a significant threat to their proper conduct; they seek to bypass models' safeguards and often provoke transgressive behaviors. However, existing automatic jailbreak attacks require extensive computational resources and are prone to converge on suboptimal solutions. In this work, we propose \textbf{C}ompliance \textbf{R}efusal \textbf{I}nitialization (CRI), a novel, attack-agnostic framework that efficiently initializes the optimization in the proximity of the compliance subspace of harmful prompts. By narrowing the initial gap to the adversarial objective, CRI substantially improves adversarial success rates (ASR) and drastically reduces computational overhead -- often requiring just a single optimization step. We evaluate CRI on the widely-used AdvBench dataset over the standard jailbreak attacks of GCG and AutoDAN. Results show that CRI boosts ASR and decreases the median steps to success by up to \textbf{\(\times 60\)}. The project page, along with the reference implementation, is publicly available at \texttt{https://amit1221levi.github.io/CRI-Jailbreak-Init-LLMs-evaluation/}.

摘要: 越狱攻击旨在利用大型语言模型(LLM)，并对它们的正常行为构成重大威胁；它们试图绕过模型的保护措施，通常会引发越轨行为。然而，现有的自动越狱攻击需要大量的计算资源，并且容易收敛到次优解。在这项工作中，我们提出了一种新的攻击不可知框架-顺从性通过缩小与对抗性目标的初始差距，CRI极大地提高了对抗性成功率(ASR)，并显著减少了计算开销--通常只需要单个优化步骤。针对GCG和AutoDAN的标准越狱攻击，我们在广泛使用的AdvBtch数据集上对CRI进行了评估。结果表明，CRI提高了ASR，并将成功步骤的中位数减少了高达Textbf{(60倍)}。项目页面以及参考实现可在\texttt{https://amit1221levi.github.io/CRI-Jailbreak-Init-LLMs-evaluation/}.上公开获取



## **38. SyntheticPop: Attacking Speaker Verification Systems With Synthetic VoicePops**

CompositicPop：使用合成voicePops攻击说话者验证系统 cs.CR

**SubmitDate**: 2025-02-13    [abs](http://arxiv.org/abs/2502.09553v1) [paper-pdf](http://arxiv.org/pdf/2502.09553v1)

**Authors**: Eshaq Jamdar, Amith Kamath Belman

**Abstract**: Voice Authentication (VA), also known as Automatic Speaker Verification (ASV), is a widely adopted authentication method, particularly in automated systems like banking services, where it serves as a secondary layer of user authentication. Despite its popularity, VA systems are vulnerable to various attacks, including replay, impersonation, and the emerging threat of deepfake audio that mimics the voice of legitimate users. To mitigate these risks, several defense mechanisms have been proposed. One such solution, Voice Pops, aims to distinguish an individual's unique phoneme pronunciations during the enrollment process. While promising, the effectiveness of VA+VoicePop against a broader range of attacks, particularly logical or adversarial attacks, remains insufficiently explored. We propose a novel attack method, which we refer to as SyntheticPop, designed to target the phoneme recognition capabilities of the VA+VoicePop system. The SyntheticPop attack involves embedding synthetic "pop" noises into spoofed audio samples, significantly degrading the model's performance. We achieve an attack success rate of over 95% while poisoning 20% of the training dataset. Our experiments demonstrate that VA+VoicePop achieves 69% accuracy under normal conditions, 37% accuracy when subjected to a baseline label flipping attack, and just 14% accuracy under our proposed SyntheticPop attack, emphasizing the effectiveness of our method.

摘要: 语音身份验证(VA)，也称为自动说话人验证(ASV)，是一种广泛采用的身份验证方法，特别是在银行服务等自动化系统中，它充当用户身份验证的第二层。尽管VA系统很受欢迎，但它很容易受到各种攻击，包括重播、模仿和模仿合法用户声音的深度假音频的新威胁。为了减轻这些风险，已经提出了几种防御机制。其中一种名为Voice Pops的解决方案旨在区分注册过程中个人独特的音素发音。尽管前景看好，但VA+VoicePop对更广泛攻击的有效性，特别是逻辑或对抗性攻击，仍然没有得到充分的研究。我们提出了一种新的攻击方法，称为SyntheticPop，旨在针对VA+VoicePop系统的音素识别能力。SyntheticPop攻击包括在伪造的音频样本中嵌入合成的“流行”噪声，从而显著降低模型的性能。我们实现了95%以上的攻击成功率，同时毒化了20%的训练数据集。实验表明，VA+VoicePop在正常情况下的准确率为69%，在遭受基线标签翻转攻击时的准确率为37%，而在我们提出的SyntheticPop攻击下的准确率仅为14%，强调了该方法的有效性。



## **39. Bayes-Nash Generative Privacy Against Membership Inference Attacks**

针对会员推断攻击的Bayes-Nash生成隐私 cs.CR

arXiv admin note: substantial text overlap with arXiv:2406.01811

**SubmitDate**: 2025-02-13    [abs](http://arxiv.org/abs/2410.07414v3) [paper-pdf](http://arxiv.org/pdf/2410.07414v3)

**Authors**: Tao Zhang, Rajagopal Venkatesaraman, Rajat K. De, Bradley A. Malin, Yevgeniy Vorobeychik

**Abstract**: Membership inference attacks (MIAs) expose significant privacy risks by determining whether an individual's data is in a dataset. While differential privacy (DP) mitigates such risks, it has several limitations in achieving an optimal balance between utility and privacy, include limited resolution in expressing this tradeoff in only a few privacy parameters, and intractable sensitivity calculations that may be necessary to provide tight privacy guarantees. We propose a game-theoretic framework that models privacy protection from MIA as a Bayesian game between a defender and an attacker. In this game, a dataset is the defender's private information, with privacy loss to the defender (which is gain to the attacker) captured in terms of the attacker's ability to infer membership of individuals in the dataset. To address the strategic complexity of this game, we represent the mixed strategy of the defender as a neural network generator which maps a private dataset to its public representation (for example, noisy summary statistics), while the mixed strategy of the attacker is captured by a discriminator which makes membership inference claims. We refer to the resulting computational approach as a general-sum Generative Adversarial Network, which is trained iteratively by alternating generator and discriminator updates akin to conventional GANs. We call the defender's data sharing policy thereby obtained Bayes-Nash Generative Privacy (BNGP). The BNGP strategy avoids sensitivity calculations, supports compositions of correlated mechanisms, is robust to the attacker's heterogeneous preferences over true and false positives, and yields provable differential privacy guarantees, albeit in an idealized setting.

摘要: 成员身份推断攻击(MIA)通过确定个人数据是否在数据集中暴露了显著的隐私风险。虽然差异隐私(DP)减轻了此类风险，但它在实现效用和隐私之间的最佳平衡方面有几个限制，包括仅用几个隐私参数表达这种权衡的有限分辨率，以及可能需要提供严格隐私保证的难以处理的敏感性计算。我们提出了一个博弈论框架，将针对MIA的隐私保护建模为防御者和攻击者之间的贝叶斯博弈。在这个游戏中，数据集是防御者的私人信息，根据攻击者推断数据集中个人成员身份的能力，防御者的隐私损失(对攻击者来说是获得的)被捕获。为了解决该游戏的战略复杂性，我们将防御者的混合策略表示为神经网络生成器，该生成器将私有数据集映射到其公共表示(例如，噪声汇总统计数据)，而攻击者的混合策略由一个鉴别器捕获，该鉴别器进行成员推理声明。我们将由此产生的计算方法称为一般和生成性对抗网络，它通过类似于传统GA的交替生成器和鉴别器更新迭代地训练。我们称之为防御者的数据共享策略，从而获得贝叶斯-纳什生成隐私(BNGP)。BNGP策略避免了敏感度计算，支持相关机制的组合，对攻击者对真阳性和假阳性的不同偏好具有健壮性，并产生可证明的差异隐私保证，尽管是在理想的设置中。



## **40. On the Importance of Backbone to the Adversarial Robustness of Object Detectors**

论主干对对象检测器对抗鲁棒性的重要性 cs.CV

Accepted by IEEE TIFS

**SubmitDate**: 2025-02-13    [abs](http://arxiv.org/abs/2305.17438v2) [paper-pdf](http://arxiv.org/pdf/2305.17438v2)

**Authors**: Xiao Li, Hang Chen, Xiaolin Hu

**Abstract**: Object detection is a critical component of various security-sensitive applications, such as autonomous driving and video surveillance. However, existing object detectors are vulnerable to adversarial attacks, which poses a significant challenge to their reliability and security. Through experiments, first, we found that existing works on improving the adversarial robustness of object detectors give a false sense of security. Second, we found that adversarially pre-trained backbone networks were essential for enhancing the adversarial robustness of object detectors. We then proposed a simple yet effective recipe for fast adversarial fine-tuning on object detectors with adversarially pre-trained backbones. Without any modifications to the structure of object detectors, our recipe achieved significantly better adversarial robustness than previous works. Finally, we explored the potential of different modern object detector designs for improving adversarial robustness with our recipe and demonstrated interesting findings, which inspired us to design state-of-the-art (SOTA) robust detectors. Our empirical results set a new milestone for adversarially robust object detection. Code and trained checkpoints are available at https://github.com/thu-ml/oddefense.

摘要: 目标检测是各种安全敏感应用的关键组件，例如自动驾驶和视频监控。然而，现有的目标探测器容易受到敌意攻击，这对其可靠性和安全性构成了巨大的挑战。通过实验，首先，我们发现现有的提高目标检测器对抗健壮性的工作给人一种错误的安全感。其次，我们发现对抗性预训练的骨干网络对于增强目标检测器的对抗性稳健性是必不可少的。然后，我们提出了一个简单但有效的配方，用于在具有对抗性预培训主干的对象探测器上进行快速对抗性微调。在不对对象检测器的结构进行任何修改的情况下，我们的配方比以前的工作获得了更好的对抗健壮性。最后，我们探索了不同的现代对象检测器设计在提高对手健壮性方面的潜力，并展示了有趣的发现，这启发了我们设计最先进的(SOTA)健壮性检测器。我们的实验结果为反向稳健的目标检测建立了一个新的里程碑。代码和训练有素的检查点可在https://github.com/thu-ml/oddefense.上找到



## **41. ADBM: Adversarial diffusion bridge model for reliable adversarial purification**

ADBM：用于可靠对抗净化的对抗扩散桥模型 cs.LG

ICLR 2025

**SubmitDate**: 2025-02-13    [abs](http://arxiv.org/abs/2408.00315v3) [paper-pdf](http://arxiv.org/pdf/2408.00315v3)

**Authors**: Xiao Li, Wenxuan Sun, Huanran Chen, Qiongxiu Li, Yining Liu, Yingzhe He, Jie Shi, Xiaolin Hu

**Abstract**: Recently Diffusion-based Purification (DiffPure) has been recognized as an effective defense method against adversarial examples. However, we find DiffPure which directly employs the original pre-trained diffusion models for adversarial purification, to be suboptimal. This is due to an inherent trade-off between noise purification performance and data recovery quality. Additionally, the reliability of existing evaluations for DiffPure is questionable, as they rely on weak adaptive attacks. In this work, we propose a novel Adversarial Diffusion Bridge Model, termed ADBM. ADBM directly constructs a reverse bridge from the diffused adversarial data back to its original clean examples, enhancing the purification capabilities of the original diffusion models. Through theoretical analysis and experimental validation across various scenarios, ADBM has proven to be a superior and robust defense mechanism, offering significant promise for practical applications.

摘要: 最近，基于扩散的纯化（DiffPure）被认为是针对对抗性例子的有效防御方法。然而，我们发现直接使用原始预训练的扩散模型进行对抗性纯化的迪夫Pure是次优的。这是由于噪音净化性能和数据恢复质量之间固有的权衡。此外，现有的DistPure评估的可靠性值得怀疑，因为它们依赖于弱适应性攻击。在这项工作中，我们提出了一种新型的对抗扩散桥模型，称为ADBM。ADBM直接构建了从扩散的对抗数据到其原始干净示例的反向桥梁，增强了原始扩散模型的净化能力。通过各种场景的理论分析和实验验证，ADBM已被证明是一种卓越且强大的防御机制，为实际应用提供了巨大的前景。



## **42. Wasserstein distributional adversarial training for deep neural networks**

深度神经网络的Wasserstein分布式对抗训练 cs.LG

15 pages, 4 figures

**SubmitDate**: 2025-02-13    [abs](http://arxiv.org/abs/2502.09352v1) [paper-pdf](http://arxiv.org/pdf/2502.09352v1)

**Authors**: Xingjian Bai, Guangyi He, Yifan Jiang, Jan Obloj

**Abstract**: Design of adversarial attacks for deep neural networks, as well as methods of adversarial training against them, are subject of intense research. In this paper, we propose methods to train against distributional attack threats, extending the TRADES method used for pointwise attacks. Our approach leverages recent contributions and relies on sensitivity analysis for Wasserstein distributionally robust optimization problems. We introduce an efficient fine-tuning method which can be deployed on a previously trained model. We test our methods on a range of pre-trained models on RobustBench. These experimental results demonstrate the additional training enhances Wasserstein distributional robustness, while maintaining original levels of pointwise robustness, even for already very successful networks. The improvements are less marked for models pre-trained using huge synthetic datasets of 20-100M images. However, remarkably, sometimes our methods are still able to improve their performance even when trained using only the original training dataset (50k images).

摘要: 深度神经网络的对抗性攻击的设计，以及针对它们的对抗性训练方法，都是深入研究的主题。在本文中，我们提出了针对分布式攻击威胁的训练方法，扩展了用于点式攻击的TRADS方法。我们的方法利用了最近的贡献，并依赖于对Wasserstein分布稳健优化问题的敏感度分析。我们介绍了一种高效的微调方法，该方法可以部署在先前训练的模型上。我们在一系列预先训练好的模型上对我们的方法进行了测试。这些实验结果表明，额外的训练增强了Wasserstein分布的健壮性，同时保持了原始的逐点健壮性，即使对于已经非常成功的网络也是如此。对于使用2000万至1亿张图像的大型合成数据集进行预训练的模型，改进效果不那么明显。然而，值得注意的是，有时我们的方法仍然能够提高它们的性能，即使只使用原始训练数据集(50k图像)进行训练。



## **43. FLAME: Flexible LLM-Assisted Moderation Engine**

FLAME：灵活的LLM辅助审核引擎 cs.CR

**SubmitDate**: 2025-02-13    [abs](http://arxiv.org/abs/2502.09175v1) [paper-pdf](http://arxiv.org/pdf/2502.09175v1)

**Authors**: Ivan Bakulin, Ilia Kopanichuk, Iaroslav Bespalov, Nikita Radchenko, Vladimir Shaposhnikov, Dmitry Dylov, Ivan Oseledets

**Abstract**: The rapid advancement of Large Language Models (LLMs) has introduced significant challenges in moderating user-model interactions. While LLMs demonstrate remarkable capabilities, they remain vulnerable to adversarial attacks, particularly ``jailbreaking'' techniques that bypass content safety measures. Current content moderation systems, which primarily rely on input prompt filtering, have proven insufficient, with techniques like Best-of-N (BoN) jailbreaking achieving success rates of 80% or more against popular LLMs. In this paper, we introduce Flexible LLM-Assisted Moderation Engine (FLAME): a new approach that shifts the focus from input filtering to output moderation. Unlike traditional circuit-breaking methods that analyze user queries, FLAME evaluates model responses, offering several key advantages: (1) computational efficiency in both training and inference, (2) enhanced resistance to BoN jailbreaking attacks, and (3) flexibility in defining and updating safety criteria through customizable topic filtering. Our experiments demonstrate that FLAME significantly outperforms current moderation systems. For example, FLAME reduces attack success rate in GPT-4o-mini and DeepSeek-v3 by a factor of ~9, while maintaining low computational overhead. We provide comprehensive evaluation on various LLMs and analyze the engine's efficiency against the state-of-the-art jailbreaking. This work contributes to the development of more robust and adaptable content moderation systems for LLMs.

摘要: 大型语言模型(LLM)的快速发展给协调用户与模型的交互带来了巨大的挑战。虽然LLM显示出非凡的能力，但它们仍然容易受到对抗性攻击，特别是绕过内容安全措施的“越狱”技术。目前的内容审核系统主要依赖于输入提示过滤，已被证明是不够的，像N中最佳(Bon)越狱技术对流行的LLM的成功率达到80%或更高。在本文中，我们介绍了灵活的LLM辅助调节引擎(FLAME)：一种将焦点从输入过滤转移到输出调节的新方法。与分析用户查询的传统断路方法不同，FLAME评估模型响应，提供了几个关键优势：(1)训练和推理的计算效率，(2)增强了对Bon越狱攻击的抵抗，以及(3)通过可定制的主题过滤灵活地定义和更新安全标准。我们的实验表明，火焰系统的性能明显优于现有的慢化系统。例如，FLAME将GPT-40-mini和DeepSeek-v3中的攻击成功率降低了~9倍，同时保持了较低的计算开销。我们对各种LLM进行了综合评估，并针对最先进的越狱情况分析了发动机的效率。这项工作有助于开发更健壮和适应性更强的LLMS内容审核系统。



## **44. Pulling Back the Curtain: Unsupervised Adversarial Detection via Contrastive Auxiliary Networks**

拉开帷幕：通过对比辅助网络的无监督对抗检测 cs.CV

**SubmitDate**: 2025-02-13    [abs](http://arxiv.org/abs/2502.09110v1) [paper-pdf](http://arxiv.org/pdf/2502.09110v1)

**Authors**: Eylon Mizrahi, Raz Lapid, Moshe Sipper

**Abstract**: Deep learning models are widely employed in safety-critical applications yet remain susceptible to adversarial attacks -- imperceptible perturbations that can significantly degrade model performance. Conventional defense mechanisms predominantly focus on either enhancing model robustness or detecting adversarial inputs independently. In this work, we propose an Unsupervised adversarial detection via Contrastive Auxiliary Networks (U-CAN) to uncover adversarial behavior within auxiliary feature representations, without the need for adversarial examples. U-CAN is embedded within selected intermediate layers of the target model. These auxiliary networks, comprising projection layers and ArcFace-based linear layers, refine feature representations to more effectively distinguish between benign and adversarial inputs. Comprehensive experiments across multiple datasets (CIFAR-10, Mammals, and a subset of ImageNet) and architectures (ResNet-50, VGG-16, and ViT) demonstrate that our method surpasses existing unsupervised adversarial detection techniques, achieving superior F1 scores against four distinct attack methods. The proposed framework provides a scalable and effective solution for enhancing the security and reliability of deep learning systems.

摘要: 深度学习模型广泛应用于安全关键应用中，但仍然容易受到对抗性攻击--可能会显著降低模型性能的不可察觉的扰动。传统的防御机制主要集中在增强模型的稳健性或独立检测敌方输入。在这项工作中，我们提出了一种基于对比辅助网络的无监督敌意检测方法(U-CAN)，以发现辅助特征表示中的对抗性行为，而不需要对抗性实例。U-CAN被嵌入到目标模型的选定中间层中。这些辅助网络包括投影层和基于ArcFace的线性层，改进了特征表示，以更有效地区分良性输入和敌意输入。在多个数据集(CIFAR-10、哺乳动物和ImageNet的子集)和体系结构(ResNet-50、VGG-16和VIT)上的综合实验表明，我们的方法超过了现有的无监督对手检测技术，在四种不同的攻击方法上获得了优越的F1分数。该框架为提高深度学习系统的安全性和可靠性提供了一种可扩展的有效解决方案。



## **45. Universal Adversarial Attack on Aligned Multimodal LLMs**

对对齐多模式LLM的普遍对抗攻击 cs.AI

Added an affiliation

**SubmitDate**: 2025-02-13    [abs](http://arxiv.org/abs/2502.07987v2) [paper-pdf](http://arxiv.org/pdf/2502.07987v2)

**Authors**: Temurbek Rahmatullaev, Polina Druzhinina, Matvey Mikhalchuk, Andrey Kuznetsov, Anton Razzhigaev

**Abstract**: We propose a universal adversarial attack on multimodal Large Language Models (LLMs) that leverages a single optimized image to override alignment safeguards across diverse queries and even multiple models. By backpropagating through the vision encoder and language head, we craft a synthetic image that forces the model to respond with a targeted phrase (e.g., ''Sure, here it is'') or otherwise unsafe content-even for harmful prompts. In experiments on the SafeBench benchmark, our method achieves significantly higher attack success rates than existing baselines, including text-only universal prompts (e.g., up to 93% on certain models). We further demonstrate cross-model transferability by training on several multimodal LLMs simultaneously and testing on unseen architectures. Additionally, a multi-answer variant of our approach produces more natural-sounding (yet still malicious) responses. These findings underscore critical vulnerabilities in current multimodal alignment and call for more robust adversarial defenses. We will release code and datasets under the Apache-2.0 license. Warning: some content generated by Multimodal LLMs in this paper may be offensive to some readers.

摘要: 我们提出了一种针对多模式大型语言模型(LLMS)的通用对抗性攻击，该攻击利用单个优化图像覆盖跨不同查询甚至多个模型的对齐保障。通过视觉编码器和语言头部的反向传播，我们制作了一个合成图像，迫使模型使用有针对性的短语(例如，“当然，就是这里”)或其他不安全的内容做出响应--即使是有害的提示。在SafeBtch基准测试上的实验中，我们的方法获得了比现有基线显著更高的攻击成功率，包括纯文本通用提示(例如，在某些型号上高达93%)。我们通过同时在多个多模式LLM上进行训练和在看不见的体系结构上进行测试来进一步证明跨模型的可转移性。此外，我们的方法的一个多答案变体会产生听起来更自然(但仍然是恶意的)响应。这些发现突显了当前多模式联合的严重弱点，并呼吁进行更强大的对抗性防御。我们将在APACHE-2.0许可下发布代码和数据集。警告：本文中的多模式LLMS生成的某些内容可能会冒犯某些读者。



## **46. RLSA-PFL: Robust Lightweight Secure Aggregation with Model Inconsistency Detection in Privacy-Preserving Federated Learning**

RL SA-PFL：隐私保护联邦学习中具有模型不一致性检测的鲁棒轻量级安全聚合 cs.CR

16 pages, 10 Figures

**SubmitDate**: 2025-02-13    [abs](http://arxiv.org/abs/2502.08989v1) [paper-pdf](http://arxiv.org/pdf/2502.08989v1)

**Authors**: Nazatul H. Sultan, Yan Bo, Yansong Gao, Seyit Camtepe, Arash Mahboubi, Hang Thanh Bui, Aufeef Chauhan, Hamed Aboutorab, Michael Bewong, Praveen Gauravaram, Rafiqul Islam, Sharif Abuadbba

**Abstract**: Federated Learning (FL) allows users to collaboratively train a global machine learning model by sharing local model only, without exposing their private data to a central server. This distributed learning is particularly appealing in scenarios where data privacy is crucial, and it has garnered substantial attention from both industry and academia. However, studies have revealed privacy vulnerabilities in FL, where adversaries can potentially infer sensitive information from the shared model parameters. In this paper, we present an efficient masking-based secure aggregation scheme utilizing lightweight cryptographic primitives to mitigate privacy risks. Our scheme offers several advantages over existing methods. First, it requires only a single setup phase for the entire FL training session, significantly reducing communication overhead. Second, it minimizes user-side overhead by eliminating the need for user-to-user interactions, utilizing an intermediate server layer and a lightweight key negotiation method. Third, the scheme is highly resilient to user dropouts, and the users can join at any FL round. Fourth, it can detect and defend against malicious server activities, including recently discovered model inconsistency attacks. Finally, our scheme ensures security in both semi-honest and malicious settings. We provide security analysis to formally prove the robustness of our approach. Furthermore, we implemented an end-to-end prototype of our scheme. We conducted comprehensive experiments and comparisons, which show that it outperforms existing solutions in terms of communication and computation overhead, functionality, and security.

摘要: 联合学习(FL)允许用户通过只共享本地模型来协作训练全局机器学习模型，而不会将他们的私有数据暴露给中央服务器。这种分布式学习在数据隐私至关重要的场景中特别有吸引力，它引起了工业界和学术界的大量关注。然而，研究揭示了FL中的隐私漏洞，攻击者可能会从共享的模型参数中推断敏感信息。本文提出了一种高效的基于掩码的安全聚合方案，该方案利用轻量级的密码原语来降低隐私风险。与现有方法相比，我们的方案有几个优点。首先，它只需要整个FL培训课程的单一设置阶段，大大减少了通信开销。其次，它利用中间服务器层和轻量级密钥协商方法，消除了用户到用户交互的需要，从而最大限度地减少了用户端开销。第三，该方案对用户退出具有很强的弹性，用户可以在任何FL轮加入。第四，它可以检测和防御恶意服务器活动，包括最近发现的模型不一致攻击。最后，我们的方案确保了半诚实和恶意环境下的安全性。我们提供安全分析来正式证明我们方法的健壮性。此外，我们还实现了我们方案的端到端原型。我们进行了全面的实验和比较，结果表明，它在通信和计算开销、功能和安全性方面都优于现有的解决方案。



## **47. An Engorgio Prompt Makes Large Language Model Babble on**

Engorgio提示让大型语言模型胡言乱语 cs.CR

ICLR 2025

**SubmitDate**: 2025-02-13    [abs](http://arxiv.org/abs/2412.19394v2) [paper-pdf](http://arxiv.org/pdf/2412.19394v2)

**Authors**: Jianshuo Dong, Ziyuan Zhang, Qingjie Zhang, Tianwei Zhang, Hao Wang, Hewu Li, Qi Li, Chao Zhang, Ke Xu, Han Qiu

**Abstract**: Auto-regressive large language models (LLMs) have yielded impressive performance in many real-world tasks. However, the new paradigm of these LLMs also exposes novel threats. In this paper, we explore their vulnerability to inference cost attacks, where a malicious user crafts Engorgio prompts to intentionally increase the computation cost and latency of the inference process. We design Engorgio, a novel methodology, to efficiently generate adversarial Engorgio prompts to affect the target LLM's service availability. Engorgio has the following two technical contributions. (1) We employ a parameterized distribution to track LLMs' prediction trajectory. (2) Targeting the auto-regressive nature of LLMs' inference process, we propose novel loss functions to stably suppress the appearance of the <EOS> token, whose occurrence will interrupt the LLM's generation process. We conduct extensive experiments on 13 open-sourced LLMs with parameters ranging from 125M to 30B. The results show that Engorgio prompts can successfully induce LLMs to generate abnormally long outputs (i.e., roughly 2-13$\times$ longer to reach 90%+ of the output length limit) in a white-box scenario and our real-world experiment demonstrates Engergio's threat to LLM service with limited computing resources. The code is released at: https://github.com/jianshuod/Engorgio-prompt.

摘要: 自回归大型语言模型(LLM)在许多实际任务中取得了令人印象深刻的性能。然而，这些LLM的新范式也暴露了新的威胁。在本文中，我们探讨了它们对推理成本攻击的脆弱性，在这种攻击中，恶意用户手工制作Engorgio会提示故意增加推理过程的计算成本和延迟。我们设计了一种新的方法Engorgio来有效地生成对抗性Engorgio提示，以影响目标LLM的服务可用性。Engorgio有以下两个技术贡献。(1)采用一种参数分布来跟踪LLMS的预测轨迹。(2)针对LLMS推理过程的自回归特性，提出了一种新的损失函数来稳定地抑制<EOS>令牌的出现，它的出现会中断LLM的生成过程。我们在13个开源的LLM上进行了广泛的实验，参数从125M到30B不等。结果表明，在白盒场景中，Engorgio提示能够成功地诱导LLMS产生异常长的输出(即大约2-13$\x$以达到输出长度限制的90%以上)，并且我们的真实世界实验证明了Engergio在有限计算资源的情况下对LLM服务的威胁。代码发布地址为：https://github.com/jianshuod/Engorgio-prompt.



## **48. Theoretically Grounded Framework for LLM Watermarking: A Distribution-Adaptive Approach**

LLM水印的理论基础框架：分布自适应方法 cs.CR

**SubmitDate**: 2025-02-12    [abs](http://arxiv.org/abs/2410.02890v3) [paper-pdf](http://arxiv.org/pdf/2410.02890v3)

**Authors**: Haiyun He, Yepeng Liu, Ziqiao Wang, Yongyi Mao, Yuheng Bu

**Abstract**: Watermarking has emerged as a crucial method to distinguish AI-generated text from human-created text. In this paper, we present a novel theoretical framework for watermarking Large Language Models (LLMs) that jointly optimizes both the watermarking scheme and the detection process. Our approach focuses on maximizing detection performance while maintaining control over the worst-case Type-I error and text distortion. We characterize \emph{the universally minimum Type-II error}, showing a fundamental trade-off between watermark detectability and text distortion. Importantly, we identify that the optimal watermarking schemes are adaptive to the LLM generative distribution. Building on our theoretical insights, we propose an efficient, model-agnostic, distribution-adaptive watermarking algorithm, utilizing a surrogate model alongside the Gumbel-max trick. Experiments conducted on Llama2-13B and Mistral-8$\times$7B models confirm the effectiveness of our approach. Additionally, we examine incorporating robustness into our framework, paving a way to future watermarking systems that withstand adversarial attacks more effectively.

摘要: 数字水印已经成为区分人工智能生成的文本和人类创建的文本的关键方法。在本文中，我们提出了一种新的大语言模型(LLMS)水印理论框架，该框架同时优化了水印方案和检测过程。我们的方法专注于最大化检测性能，同时保持对最坏情况下的类型I错误和文本失真的控制。我们将其刻画在水印可检测性和文本失真之间的基本权衡。重要的是，我们发现最优水印方案对LLM生成分布是自适应的。基于我们的理论见解，我们提出了一种高效的、与模型无关的、分布自适应的水印算法，该算法利用代理模型和Gumbel-max技巧。在Llama2-13B和Mistral-8$\x$70亿模型上进行的实验证实了该方法的有效性。此外，我们还研究了将健壮性融入到我们的框架中，为未来更有效地抵御对手攻击的水印系统铺平了道路。



## **49. Bankrupting DoS Attackers**

破产的拒绝服务攻击者 cs.CR

**SubmitDate**: 2025-02-12    [abs](http://arxiv.org/abs/2205.08287v4) [paper-pdf](http://arxiv.org/pdf/2205.08287v4)

**Authors**: Trisha Chakraborty, Abir Islam, Valerie King, Daniel Rayborn, Jared Saia, Maxwell Young

**Abstract**: Can we make a denial-of-service attacker pay more than the server and honest clients? Consider a model where a server sees a stream of jobs sent by either honest clients or an adversary. The server sets a price for servicing each job with the aid of an estimator, which provides approximate statistical information about the distribution of previously occurring good jobs.   We describe and analyze pricing algorithms for the server under different models of synchrony, with total cost parameterized by the accuracy of the estimator. Given a reasonably accurate estimator, the algorithm's cost provably grows more slowly than the attacker's cost, as the attacker's cost grows large. Additionally, we prove a lower bound, showing that our pricing algorithm yields asymptotically tight results when the estimator is accurate within constant factors.

摘要: 我们能否让拒绝服务攻击者支付比服务器和诚实客户更多的费用？考虑一个模型，其中服务器看到诚实的客户或对手发送的作业流。服务器在估计器的帮助下为每个作业设定服务的价格，该估计器提供有关之前发生的好作业分布的大致统计信息。   我们描述和分析了不同同步模型下服务器的定价算法，总成本由估计器的准确性参数化。给定一个相当准确的估计器，可以证明，随着攻击者的成本变得很大，算法的成本增长速度比攻击者的成本慢。此外，我们证明了一个下界，表明当估计量在恒定因子内准确时，我们的定价算法会产生渐进紧的结果。



## **50. Extreme vulnerability to intruder attacks destabilizes network dynamics**

对入侵者攻击的极端脆弱性会破坏网络动态的稳定 nlin.AO

**SubmitDate**: 2025-02-12    [abs](http://arxiv.org/abs/2502.08552v1) [paper-pdf](http://arxiv.org/pdf/2502.08552v1)

**Authors**: Amirhossein Nazerian, Sahand Tangerami, Malbor Asllani, David Phillips, Hernan Makse, Francesco Sorrentino

**Abstract**: Consensus, synchronization, formation control, and power grid balance are all examples of virtuous dynamical states that may arise in networks. Here we focus on how such states can be destabilized from a fundamental perspective; namely, we address the question of how one or a few intruder agents within an otherwise functioning network may compromise its dynamics. We show that a single adversarial node coupled via adversarial couplings to one or more other nodes is sufficient to destabilize the entire network, which we prove to be more efficient than targeting multiple nodes. Then, we show that concentrating the attack on a single low-indegree node induces the greatest instability, challenging the common assumption that hubs are the most critical nodes. This leads to a new characterization of the vulnerability of a node, which contrasts with previous work, and identifies low-indegree nodes (as opposed to the hubs) as the most vulnerable components of a network. Our results are derived for linear systems but hold true for nonlinear networks, including those described by the Kuramoto model. Finally, we derive scaling laws showing that larger networks are less susceptible, on average, to single-node attacks. Overall, these findings highlight an intrinsic vulnerability of technological systems such as autonomous networks, sensor networks, power grids, and the internet of things, which also extend to the realm of complex social and biological networks.

摘要: 共识、同步、队形控制和电网平衡都是网络中可能出现的良性动态状态的例子。在这里，我们从根本的角度关注如何破坏这种状态的稳定；也就是，我们解决了一个或几个入侵者代理在其他功能正常的网络中如何可能危及其动态的问题。我们证明了单个敌意节点通过对抗性耦合耦合到一个或多个其他节点足以破坏整个网络的稳定，我们证明了这比针对多个节点更有效。然后，我们证明了将攻击集中在单个低度节点上会导致最大的不稳定性，挑战了集线器是最关键节点的普遍假设。这导致了对节点脆弱性的新的表征，这与以前的工作形成了对比，并将低索引度节点(而不是集线器)识别为网络中最脆弱的组件。我们的结果适用于线性系统，但也适用于非线性网络，包括用Kuramoto模型描述的网络。最后，我们推导出了标度律，表明较大的网络平均而言不太容易受到单节点攻击。总体而言，这些发现突显了自主网络、传感器网络、电网和物联网等技术系统的内在脆弱性，这些系统也延伸到复杂的社会和生物网络领域。



