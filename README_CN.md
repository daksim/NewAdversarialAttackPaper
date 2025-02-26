# Latest Adversarial Attack Papers
**update at 2025-02-26 10:09:48**

翻译来自 https://cloud.tencent.com/document/product/551/15619

## **1. Learning atomic forces from uncertainty-calibrated adversarial attacks**

从不确定性校准的对抗攻击中学习原子力 physics.comp-ph

**SubmitDate**: 2025-02-25    [abs](http://arxiv.org/abs/2502.18314v1) [paper-pdf](http://arxiv.org/pdf/2502.18314v1)

**Authors**: Henrique Musseli Cezar, Tilmann Bodenstein, Henrik Andersen Sveinsson, Morten Ledum, Simen Reine, Sigbjørn Løland Bore

**Abstract**: Adversarial approaches, which intentionally challenge machine learning models by generating difficult examples, are increasingly being adopted to improve machine learning interatomic potentials (MLIPs). While already providing great practical value, little is known about the actual prediction errors of MLIPs on adversarial structures and whether these errors can be controlled. We propose the Calibrated Adversarial Geometry Optimization (CAGO) algorithm to discover adversarial structures with user-assigned errors. Through uncertainty calibration, the estimated uncertainty of MLIPs is unified with real errors. By performing geometry optimization for calibrated uncertainty, we reach adversarial structures with the user-assigned target MLIP prediction error. Integrating with active learning pipelines, we benchmark CAGO, demonstrating stable MLIPs that systematically converge structural, dynamical, and thermodynamical properties for liquid water and water adsorption in a metal-organic framework within only hundreds of training structures, where previously many thousands were typically required.

摘要: 对抗性方法通过生成困难的例子来故意挑战机器学习模型，越来越多地被用来改善机器学习的原子间势(MLIP)。虽然已经提供了很大的实用价值，但对于MLIP在对抗性结构上的实际预测误差以及这些误差是否可以控制，人们知之甚少。我们提出了校准对抗性几何优化(CAGO)算法来发现含有用户指定错误的对抗性结构。通过不确定度校正，将MLIP的估计不确定度与实际误差统一起来。通过对标定的不确定性进行几何优化，我们得到了具有用户指定的目标MLIP预测误差的对抗性结构。与主动学习管道集成，我们对CAGO进行了基准测试，展示了稳定的MLIP，这些MLIP系统地聚合了液态水的结构、动力学和热力学属性，以及金属-有机框架中的水吸附，仅在数百个培训结构中，而以前通常需要数千个。



## **2. Topic-FlipRAG: Topic-Orientated Adversarial Opinion Manipulation Attacks to Retrieval-Augmented Generation Models**

Top-FlipRAG：针对检索增强生成模型的面向主题的对抗性观点操纵攻击 cs.CL

**SubmitDate**: 2025-02-25    [abs](http://arxiv.org/abs/2502.01386v2) [paper-pdf](http://arxiv.org/pdf/2502.01386v2)

**Authors**: Yuyang Gong, Zhuo Chen, Miaokun Chen, Fengchang Yu, Wei Lu, Xiaofeng Wang, Xiaozhong Liu, Jiawei Liu

**Abstract**: Retrieval-Augmented Generation (RAG) systems based on Large Language Models (LLMs) have become essential for tasks such as question answering and content generation. However, their increasing impact on public opinion and information dissemination has made them a critical focus for security research due to inherent vulnerabilities. Previous studies have predominantly addressed attacks targeting factual or single-query manipulations. In this paper, we address a more practical scenario: topic-oriented adversarial opinion manipulation attacks on RAG models, where LLMs are required to reason and synthesize multiple perspectives, rendering them particularly susceptible to systematic knowledge poisoning. Specifically, we propose Topic-FlipRAG, a two-stage manipulation attack pipeline that strategically crafts adversarial perturbations to influence opinions across related queries. This approach combines traditional adversarial ranking attack techniques and leverages the extensive internal relevant knowledge and reasoning capabilities of LLMs to execute semantic-level perturbations. Experiments show that the proposed attacks effectively shift the opinion of the model's outputs on specific topics, significantly impacting user information perception. Current mitigation methods cannot effectively defend against such attacks, highlighting the necessity for enhanced safeguards for RAG systems, and offering crucial insights for LLM security research.

摘要: 基于大型语言模型(LLMS)的检索-增强生成(RAG)系统已经成为诸如问题回答和内容生成等任务的关键。然而，由于其固有的脆弱性，它们对舆论和信息传播的影响越来越大，使其成为安全研究的关键焦点。以前的研究主要针对针对事实或单一查询操作的攻击。在本文中，我们讨论了一个更实际的场景：针对RAG模型的面向主题的对抗性意见操纵攻击，其中要求LLM推理和综合多个视角，使它们特别容易受到系统性知识中毒的影响。具体地说，我们提出了Theme-FlipRAG，这是一种两阶段操纵攻击管道，战略性地制造对抗性扰动来影响相关查询的观点。该方法结合了传统的对抗性排序攻击技术，并利用LLMS丰富的内部相关知识和推理能力来执行语义级的扰动。实验表明，提出的攻击有效地改变了模型输出对特定主题的看法，显著影响了用户对信息的感知。目前的缓解方法无法有效防御此类攻击，这突显了加强RAG系统安全保障的必要性，并为LLM安全研究提供了至关重要的见解。



## **3. CLIPure: Purification in Latent Space via CLIP for Adversarially Robust Zero-Shot Classification**

CLIPure：通过CLIP在潜空间中净化，以实现对抗鲁棒零镜头分类 cs.CV

accepted by ICLR 2025

**SubmitDate**: 2025-02-25    [abs](http://arxiv.org/abs/2502.18176v1) [paper-pdf](http://arxiv.org/pdf/2502.18176v1)

**Authors**: Mingkun Zhang, Keping Bi, Wei Chen, Jiafeng Guo, Xueqi Cheng

**Abstract**: In this paper, we aim to build an adversarially robust zero-shot image classifier. We ground our work on CLIP, a vision-language pre-trained encoder model that can perform zero-shot classification by matching an image with text prompts ``a photo of a <class-name>.''. Purification is the path we choose since it does not require adversarial training on specific attack types and thus can cope with any foreseen attacks. We then formulate purification risk as the KL divergence between the joint distributions of the purification process of denoising the adversarial samples and the attack process of adding perturbations to benign samples, through bidirectional Stochastic Differential Equations (SDEs). The final derived results inspire us to explore purification in the multi-modal latent space of CLIP. We propose two variants for our CLIPure approach: CLIPure-Diff which models the likelihood of images' latent vectors with the DiffusionPrior module in DaLLE-2 (modeling the generation process of CLIP's latent vectors), and CLIPure-Cos which models the likelihood with the cosine similarity between the embeddings of an image and ``a photo of a.''. As far as we know, CLIPure is the first purification method in multi-modal latent space and CLIPure-Cos is the first purification method that is not based on generative models, which substantially improves defense efficiency. We conducted extensive experiments on CIFAR-10, ImageNet, and 13 datasets that previous CLIP-based defense methods used for evaluating zero-shot classification robustness. Results show that CLIPure boosts the SOTA robustness by a large margin, e.g., from 71.7% to 91.1% on CIFAR10, from 59.6% to 72.6% on ImageNet, and 108% relative improvements of average robustness on the 13 datasets over previous SOTA. The code is available at https://github.com/TMLResearchGroup-CAS/CLIPure.

摘要: 在这篇文章中，我们的目标是建立一个对抗性稳健的零镜头图像分类器。我们的工作基于CLIP，这是一个视觉语言预先训练的编码器模型，它可以通过将图像与文本提示进行匹配来执行零镜头分类。净化是我们选择的路径，因为它不需要针对特定攻击类型的对抗性训练，因此可以应对任何可预见的攻击。然后，我们通过双向随机微分方程(SDE)将净化风险表示为对敌方样本去噪的净化过程和对良性样本添加扰动的攻击过程的联合分布之间的KL发散。最终得出的结果启发我们去探索CLIP的多峰潜伏空间中的净化。我们为我们的CLIPure方法提出了两种变体：CLIPure-Diff和CLIPure-Cos，CLIPure-Diff使用DALE-2中的DiffusionPrior模块(对剪辑的潜在向量的生成过程进行建模)来模拟图像的潜在向量的可能性，CLIPure-Cos使用图像的嵌入和“a的照片”之间的余弦相似性来建模可能性。据我们所知，CLIPure是第一个在多峰潜在空间中进行净化的方法，而CLIPure-Cos是第一个不基于产生式模型的净化方法，大大提高了防御效率。我们在CIFAR-10、ImageNet和13个数据集上进行了广泛的实验，这些数据集是以前基于剪辑的防御方法用于评估零镜头分类稳健性的。结果表明，CLIPure在很大程度上提高了SOTA的健壮性，例如，在CIFAR10上从71.7%提高到91.1%，在ImageNet上从59.6%提高到72.6%，在13个数据集上的平均健壮性比以前的SOTA提高了108%。代码可在https://github.com/TMLResearchGroup-CAS/CLIPure.上获得



## **4. Exploring the Robustness and Transferability of Patch-Based Adversarial Attacks in Quantized Neural Networks**

探索量化神经网络中基于补丁的对抗攻击的鲁棒性和可移植性 cs.CR

**SubmitDate**: 2025-02-25    [abs](http://arxiv.org/abs/2411.15246v2) [paper-pdf](http://arxiv.org/pdf/2411.15246v2)

**Authors**: Amira Guesmi, Bassem Ouni, Muhammad Shafique

**Abstract**: Quantized neural networks (QNNs) are increasingly used for efficient deployment of deep learning models on resource-constrained platforms, such as mobile devices and edge computing systems. While quantization reduces model size and computational demands, its impact on adversarial robustness-especially against patch-based attacks-remains inadequately addressed. Patch-based attacks, characterized by localized, high-visibility perturbations, pose significant security risks due to their transferability and resilience. In this study, we systematically evaluate the vulnerability of QNNs to patch-based adversarial attacks across various quantization levels and architectures, focusing on factors that contribute to the robustness of these attacks. Through experiments analyzing feature representations, quantization strength, gradient alignment, and spatial sensitivity, we find that patch attacks consistently achieve high success rates across bitwidths and architectures, demonstrating significant transferability even in heavily quantized models. Contrary to the expectation that quantization might enhance adversarial defenses, our results show that QNNs remain highly susceptible to patch attacks due to the persistence of distinct, localized features within quantized representations. These findings underscore the need for quantization-aware defenses that address the specific challenges posed by patch-based attacks. Our work contributes to a deeper understanding of adversarial robustness in QNNs and aims to guide future research in developing secure, quantization-compatible defenses for real-world applications.

摘要: 量化神经网络(QNN)越来越多地被用于在资源受限的平台上高效地部署深度学习模型，例如移动设备和边缘计算系统。虽然量化减少了模型大小和计算需求，但它对对手健壮性的影响--特别是针对基于补丁的攻击--仍然没有得到充分的解决。基于补丁的攻击，其特点是局部化、高可见性的扰动，由于其可转移性和弹性，构成了巨大的安全风险。在这项研究中，我们系统地评估了QNN在不同量化级别和体系结构上对基于补丁的攻击的脆弱性，重点讨论了影响这些攻击的健壮性的因素。通过对特征表示、量化强度、梯度对齐和空间敏感度的实验分析，我们发现补丁攻击在不同的比特和体系结构上都获得了很高的成功率，即使在高度量化的模型中也表现出了显著的可移植性。与量化可能增强敌意防御的预期相反，我们的结果表明，由于量化表示中独特的局部化特征的持久性，QNN仍然非常容易受到补丁攻击。这些发现强调了量化感知防御的必要性，以应对基于补丁的攻击带来的具体挑战。我们的工作有助于更深入地理解QNN中的对抗健壮性，并旨在指导未来的研究，为现实世界的应用开发安全的、量化兼容的防御措施。



## **5. Towards Robust and Secure Embodied AI: A Survey on Vulnerabilities and Attacks**

迈向强大和安全的人工智能：关于漏洞和攻击的调查 cs.CR

**SubmitDate**: 2025-02-25    [abs](http://arxiv.org/abs/2502.13175v2) [paper-pdf](http://arxiv.org/pdf/2502.13175v2)

**Authors**: Wenpeng Xing, Minghao Li, Mohan Li, Meng Han

**Abstract**: Embodied AI systems, including robots and autonomous vehicles, are increasingly integrated into real-world applications, where they encounter a range of vulnerabilities stemming from both environmental and system-level factors. These vulnerabilities manifest through sensor spoofing, adversarial attacks, and failures in task and motion planning, posing significant challenges to robustness and safety. Despite the growing body of research, existing reviews rarely focus specifically on the unique safety and security challenges of embodied AI systems. Most prior work either addresses general AI vulnerabilities or focuses on isolated aspects, lacking a dedicated and unified framework tailored to embodied AI. This survey fills this critical gap by: (1) categorizing vulnerabilities specific to embodied AI into exogenous (e.g., physical attacks, cybersecurity threats) and endogenous (e.g., sensor failures, software flaws) origins; (2) systematically analyzing adversarial attack paradigms unique to embodied AI, with a focus on their impact on perception, decision-making, and embodied interaction; (3) investigating attack vectors targeting large vision-language models (LVLMs) and large language models (LLMs) within embodied systems, such as jailbreak attacks and instruction misinterpretation; (4) evaluating robustness challenges in algorithms for embodied perception, decision-making, and task planning; and (5) proposing targeted strategies to enhance the safety and reliability of embodied AI systems. By integrating these dimensions, we provide a comprehensive framework for understanding the interplay between vulnerabilities and safety in embodied AI.

摘要: 包括机器人和自动驾驶车辆在内的具体化人工智能系统正越来越多地融入现实世界的应用程序，在这些应用程序中，它们遇到了一系列源于环境和系统层面因素的漏洞。这些漏洞表现为传感器欺骗、对抗性攻击以及任务和运动规划中的失败，对健壮性和安全性构成了重大挑战。尽管研究的主体越来越多，但现有的审查很少专门关注嵌入式人工智能系统的独特安全和安保挑战。大多数以前的工作要么解决了一般的人工智能漏洞，要么专注于孤立的方面，缺乏一个专门为体现的人工智能量身定做的统一框架。本调查通过以下方式填补这一关键空白：(1)将特定于具身人工智能的漏洞分为外源性(如物理攻击、网络安全威胁)和内源性(如传感器故障、软件缺陷)来源；(2)系统分析具身人工智能特有的对抗性攻击范式，重点关注它们对感知、决策和具身交互的影响；(3)调查针对具身系统内的大视觉语言模型(LVLM)和大语言模型(LMS)的攻击向量，如越狱攻击和指令曲解；(4)评估体现感知、决策和任务规划算法中的健壮性挑战；(5)提出有针对性的策略，以提高体现人工智能系统的安全性和可靠性。通过整合这些维度，我们提供了一个全面的框架，用于理解体现的人工智能中漏洞和安全之间的相互作用。



## **6. CausalDiff: Causality-Inspired Disentanglement via Diffusion Model for Adversarial Defense**

卡西姆·分歧：通过对抗性防御的扩散模型来启发性解纠缠 cs.CV

accepted by NeurIPS 2024

**SubmitDate**: 2025-02-25    [abs](http://arxiv.org/abs/2410.23091v7) [paper-pdf](http://arxiv.org/pdf/2410.23091v7)

**Authors**: Mingkun Zhang, Keping Bi, Wei Chen, Quanrun Chen, Jiafeng Guo, Xueqi Cheng

**Abstract**: Despite ongoing efforts to defend neural classifiers from adversarial attacks, they remain vulnerable, especially to unseen attacks. In contrast, humans are difficult to be cheated by subtle manipulations, since we make judgments only based on essential factors. Inspired by this observation, we attempt to model label generation with essential label-causative factors and incorporate label-non-causative factors to assist data generation. For an adversarial example, we aim to discriminate the perturbations as non-causative factors and make predictions only based on the label-causative factors. Concretely, we propose a casual diffusion model (CausalDiff) that adapts diffusion models for conditional data generation and disentangles the two types of casual factors by learning towards a novel casual information bottleneck objective. Empirically, CausalDiff has significantly outperformed state-of-the-art defense methods on various unseen attacks, achieving an average robustness of 86.39% (+4.01%) on CIFAR-10, 56.25% (+3.13%) on CIFAR-100, and 82.62% (+4.93%) on GTSRB (German Traffic Sign Recognition Benchmark). The code is available at https://github.com/CAS-AISafetyBasicResearchGroup/CausalDiff.

摘要: 尽管不断努力保护神经分类器免受对手攻击，但它们仍然很脆弱，特别是面对看不见的攻击。相比之下，人类很难被微妙的操纵所欺骗，因为我们只根据基本因素做出判断。受到这一观察的启发，我们试图用基本的标签原因因素来建模标签生成，并结合标签非原因因素来辅助数据生成。对于一个对抗性的例子，我们的目标是将扰动区分为非致因因素，并仅基于标签致因因素进行预测。具体地说，我们提出了一个偶然扩散模型(CausalDiff)，该模型使扩散模型适用于条件数据生成，并通过向一个新的偶然信息瓶颈目标学习来区分这两种类型的偶然因素。经验上，CausalDiff在各种隐形攻击上的表现明显优于最先进的防御方法，在CIFAR-10上获得了86.39%(+4.01%)的平均健壮性，在CIFAR-100上获得了56.25%(+3.13%)的健壮性，在GTSRB(德国交通标志识别基准)上实现了82.62%(+4.93%)的平均健壮性。代码可在https://github.com/CAS-AISafetyBasicResearchGroup/CausalDiff.上获得



## **7. Towards Certification of Uncertainty Calibration under Adversarial Attacks**

对抗攻击下的不确定性校准认证 cs.LG

10 pages main paper, appendix included Published at: International  Conference on Learning Representations (ICLR) 2025

**SubmitDate**: 2025-02-25    [abs](http://arxiv.org/abs/2405.13922v3) [paper-pdf](http://arxiv.org/pdf/2405.13922v3)

**Authors**: Cornelius Emde, Francesco Pinto, Thomas Lukasiewicz, Philip H. S. Torr, Adel Bibi

**Abstract**: Since neural classifiers are known to be sensitive to adversarial perturbations that alter their accuracy, \textit{certification methods} have been developed to provide provable guarantees on the insensitivity of their predictions to such perturbations. Furthermore, in safety-critical applications, the frequentist interpretation of the confidence of a classifier (also known as model calibration) can be of utmost importance. This property can be measured via the Brier score or the expected calibration error. We show that attacks can significantly harm calibration, and thus propose certified calibration as worst-case bounds on calibration under adversarial perturbations. Specifically, we produce analytic bounds for the Brier score and approximate bounds via the solution of a mixed-integer program on the expected calibration error. Finally, we propose novel calibration attacks and demonstrate how they can improve model calibration through \textit{adversarial calibration training}.

摘要: 由于众所周知，神经分类器对改变其准确性的对抗性扰动敏感，因此\textit{认证方法}的开发是为了提供可证明的保证其预测对此类扰动的不敏感性。此外，在安全关键应用中，分类器置信度的频率主义解释（也称为模型校准）可能至关重要。该属性可以通过Brier评分或预期的校准误差来测量。我们表明，攻击可能会严重损害校准，因此建议将经过认证的校准作为对抗性扰动下校准的最坏情况界限。具体来说，我们通过对预期校准误差求解混合整数程序来产生Brier分数的分析界限和近似界限。最后，我们提出了新颖的校准攻击，并演示了它们如何通过\textit{对抗校准训练}来改进模型校准。



## **8. Model-Free Adversarial Purification via Coarse-To-Fine Tensor Network Representation**

通过粗到细张量网络表示的无模型对抗净化 cs.LG

**SubmitDate**: 2025-02-25    [abs](http://arxiv.org/abs/2502.17972v1) [paper-pdf](http://arxiv.org/pdf/2502.17972v1)

**Authors**: Guang Lin, Duc Thien Nguyen, Zerui Tao, Konstantinos Slavakis, Toshihisa Tanaka, Qibin Zhao

**Abstract**: Deep neural networks are known to be vulnerable to well-designed adversarial attacks. Although numerous defense strategies have been proposed, many are tailored to the specific attacks or tasks and often fail to generalize across diverse scenarios. In this paper, we propose Tensor Network Purification (TNP), a novel model-free adversarial purification method by a specially designed tensor network decomposition algorithm. TNP depends neither on the pre-trained generative model nor the specific dataset, resulting in strong robustness across diverse adversarial scenarios. To this end, the key challenge lies in relaxing Gaussian-noise assumptions of classical decompositions and accommodating the unknown distribution of adversarial perturbations. Unlike the low-rank representation of classical decompositions, TNP aims to reconstruct the unobserved clean examples from an adversarial example. Specifically, TNP leverages progressive downsampling and introduces a novel adversarial optimization objective to address the challenge of minimizing reconstruction error but without inadvertently restoring adversarial perturbations. Extensive experiments conducted on CIFAR-10, CIFAR-100, and ImageNet demonstrate that our method generalizes effectively across various norm threats, attack types, and tasks, providing a versatile and promising adversarial purification technique.

摘要: 众所周知，深度神经网络很容易受到精心设计的对抗性攻击。尽管已经提出了许多防御策略，但许多都是针对特定的攻击或任务量身定做的，往往无法对不同的场景进行概括。在本文中，我们提出了张量网络净化(TNP)，这是一种新的无模型的对抗性净化方法，它通过专门设计的张量网络分解算法来实现。TNP既不依赖于预先训练的生成模型，也不依赖于特定的数据集，从而在不同的对抗场景中具有很强的稳健性。为此，关键的挑战在于放宽经典分解的高斯噪声假设，并适应对抗性扰动的未知分布。与经典分解的低阶表示不同，TNP的目标是从对抗性实例中重构未被观察到的干净实例。具体地说，TNP利用渐进式下采样，并引入了一种新的对抗性优化目标来解决最小化重建误差但不会无意中恢复对抗性扰动的挑战。在CIFAR-10、CIFAR-100和ImageNet上进行的大量实验表明，我们的方法有效地概括了各种规范威胁、攻击类型和任务，提供了一种通用的、有前景的对手净化技术。



## **9. The Hidden Risks of Large Reasoning Models: A Safety Assessment of R1**

大型推理模型的隐藏风险：R1的安全评估 cs.CY

**SubmitDate**: 2025-02-25    [abs](http://arxiv.org/abs/2502.12659v2) [paper-pdf](http://arxiv.org/pdf/2502.12659v2)

**Authors**: Kaiwen Zhou, Chengzhi Liu, Xuandong Zhao, Shreedhar Jangam, Jayanth Srinivasa, Gaowen Liu, Dawn Song, Xin Eric Wang

**Abstract**: The rapid development of large reasoning models, such as OpenAI-o3 and DeepSeek-R1, has led to significant improvements in complex reasoning over non-reasoning large language models~(LLMs). However, their enhanced capabilities, combined with the open-source access of models like DeepSeek-R1, raise serious safety concerns, particularly regarding their potential for misuse. In this work, we present a comprehensive safety assessment of these reasoning models, leveraging established safety benchmarks to evaluate their compliance with safety regulations. Furthermore, we investigate their susceptibility to adversarial attacks, such as jailbreaking and prompt injection, to assess their robustness in real-world applications. Through our multi-faceted analysis, we uncover four key findings: (1) There is a significant safety gap between the open-source R1 models and the o3-mini model, on both safety benchmark and attack, suggesting more safety effort on R1 is needed. (2) The distilled reasoning model shows poorer safety performance compared to its safety-aligned base models. (3) The stronger the model's reasoning ability, the greater the potential harm it may cause when answering unsafe questions. (4) The thinking process in R1 models pose greater safety concerns than their final answers. Our study provides insights into the security implications of reasoning models and highlights the need for further advancements in R1 models' safety to close the gap.

摘要: 大型推理模型的快速发展，如OpenAI-03和DeepSeek-R1，使得复杂推理相对于非推理的大型语言模型有了显著的改进。然而，它们增强的能力，再加上DeepSeek-R1等型号的开源访问，引发了严重的安全问题，特别是它们可能被滥用的问题。在这项工作中，我们提出了这些推理模型的全面安全评估，利用已建立的安全基准来评估它们是否符合安全法规。此外，我们调查了它们对敌意攻击的敏感性，例如越狱和快速注入，以评估它们在现实世界应用程序中的健壮性。通过多方面的分析，我们发现了四个重要的发现：(1)无论是在安全基准上还是在攻击上，开源的R1型号和03-mini型号之间都存在着显著的安全差距，这表明需要在R1上做出更多的安全努力。(2)与安全对齐的基本模型相比，精炼推理模型的安全性能较差。(3)模型的推理能力越强，在回答不安全问题时可能造成的潜在危害就越大。(4)与最终答案相比，R1模型的思维过程带来了更大的安全顾虑。我们的研究为推理模型的安全含义提供了见解，并强调了在R1模型的安全性方面进一步改进的必要性，以缩小差距。



## **10. LiSA: Leveraging Link Recommender to Attack Graph Neural Networks via Subgraph Injection**

LiSA：利用链接推荐通过子图注入攻击图神经网络 cs.LG

PAKDD 2025

**SubmitDate**: 2025-02-25    [abs](http://arxiv.org/abs/2502.09271v3) [paper-pdf](http://arxiv.org/pdf/2502.09271v3)

**Authors**: Wenlun Zhang, Enyan Dai, Kentaro Yoshioka

**Abstract**: Graph Neural Networks (GNNs) have demonstrated remarkable proficiency in modeling data with graph structures, yet recent research reveals their susceptibility to adversarial attacks. Traditional attack methodologies, which rely on manipulating the original graph or adding links to artificially created nodes, often prove impractical in real-world settings. This paper introduces a novel adversarial scenario involving the injection of an isolated subgraph to deceive both the link recommender and the node classifier within a GNN system. Specifically, the link recommender is mislead to propose links between targeted victim nodes and the subgraph, encouraging users to unintentionally establish connections and that would degrade the node classification accuracy, thereby facilitating a successful attack. To address this, we present the LiSA framework, which employs a dual surrogate model and bi-level optimization to simultaneously meet two adversarial objectives. Extensive experiments on real-world datasets demonstrate the effectiveness of our method.

摘要: 图神经网络(GNN)在用图结构建模数据方面表现出了卓越的能力，但最近的研究表明，它们对对手攻击很敏感。传统的攻击方法依赖于操纵原始图形或添加到人工创建的节点的链接，在现实世界中往往被证明是不切实际的。在GNN系统中，引入一个孤立的子图来欺骗链接推荐器和节点分类器，提出了一种新的对抗性场景。具体地说，链接推荐器被误导提出目标受害节点与子图之间的链接，鼓励用户无意中建立连接，这将降低节点分类的准确性，从而促进攻击的成功。为了解决这一问题，我们提出了LISA框架，该框架采用双重代理模型和双层优化来同时满足两个对抗性目标。在真实数据集上的大量实验证明了该方法的有效性。



## **11. Relationship between Uncertainty in DNNs and Adversarial Attacks**

DNN的不确定性与对抗攻击之间的关系 cs.LG

review

**SubmitDate**: 2025-02-24    [abs](http://arxiv.org/abs/2409.13232v2) [paper-pdf](http://arxiv.org/pdf/2409.13232v2)

**Authors**: Mabel Ogonna, Abigail Adeniran, Adewale Adeyemo

**Abstract**: Deep Neural Networks (DNNs) have achieved state of the art results and even outperformed human accuracy in many challenging tasks, leading to DNNs adoption in a variety of fields including natural language processing, pattern recognition, prediction, and control optimization. However, DNNs are accompanied by uncertainty about their results, causing them to predict an outcome that is either incorrect or outside of a certain level of confidence. These uncertainties stem from model or data constraints, which could be exacerbated by adversarial attacks. Adversarial attacks aim to provide perturbed input to DNNs, causing the DNN to make incorrect predictions or increase model uncertainty. In this review, we explore the relationship between DNN uncertainty and adversarial attacks, emphasizing how adversarial attacks might raise DNN uncertainty.

摘要: 深度神经网络（DNN）已实现最先进的结果，甚至在许多具有挑战性的任务中超过了人类的准确性，导致DNN在自然语言处理、模式识别、预测和控制优化等各个领域得到采用。然而，DNN伴随着结果的不确定性，导致它们预测的结果要么不正确，要么超出一定置信水平。这些不确定性源于模型或数据限制，对抗性攻击可能会加剧这种限制。对抗性攻击旨在向DNN提供受干扰的输入，导致DNN做出错误的预测或增加模型的不确定性。在这篇评论中，我们探讨了DNN不确定性和对抗性攻击之间的关系，强调了对抗性攻击如何提高DNN不确定性。



## **12. The Geometry of Refusal in Large Language Models: Concept Cones and Representational Independence**

大型语言模型中拒绝的几何学：概念锥和表示独立性 cs.LG

**SubmitDate**: 2025-02-24    [abs](http://arxiv.org/abs/2502.17420v1) [paper-pdf](http://arxiv.org/pdf/2502.17420v1)

**Authors**: Tom Wollschläger, Jannes Elstner, Simon Geisler, Vincent Cohen-Addad, Stephan Günnemann, Johannes Gasteiger

**Abstract**: The safety alignment of large language models (LLMs) can be circumvented through adversarially crafted inputs, yet the mechanisms by which these attacks bypass safety barriers remain poorly understood. Prior work suggests that a single refusal direction in the model's activation space determines whether an LLM refuses a request. In this study, we propose a novel gradient-based approach to representation engineering and use it to identify refusal directions. Contrary to prior work, we uncover multiple independent directions and even multi-dimensional concept cones that mediate refusal. Moreover, we show that orthogonality alone does not imply independence under intervention, motivating the notion of representational independence that accounts for both linear and non-linear effects. Using this framework, we identify mechanistically independent refusal directions. We show that refusal mechanisms in LLMs are governed by complex spatial structures and identify functionally independent directions, confirming that multiple distinct mechanisms drive refusal behavior. Our gradient-based approach uncovers these mechanisms and can further serve as a foundation for future work on understanding LLMs.

摘要: 大型语言模型(LLM)的安全一致性可以通过恶意创建的输入来规避，但这些攻击绕过安全屏障的机制仍然知之甚少。先前的工作表明，模型激活空间中的单个拒绝方向决定了LLM是否拒绝请求。在这项研究中，我们提出了一种新的基于梯度的表示工程方法，并用它来识别拒绝方向。与以前的工作相反，我们发现了多个独立的方向，甚至是调解拒绝的多维概念锥。此外，我们表明，正交性本身并不意味着干预下的独立性，这激发了既能解释线性效应又能解释非线性效应的表征独立性的概念。利用这个框架，我们确定了机械独立的拒绝方向。我们发现，LLMS中的拒绝机制受到复杂空间结构的支配，并识别出功能独立的方向，证实了多种不同的机制驱动着拒绝行为。我们的基于梯度的方法揭示了这些机制，并可以进一步作为理解LLMS的未来工作的基础。



## **13. Emoti-Attack: Zero-Perturbation Adversarial Attacks on NLP Systems via Emoji Sequences**

伪攻击：通过伪君子序列对NLP系统进行零微扰对抗攻击 cs.AI

**SubmitDate**: 2025-02-24    [abs](http://arxiv.org/abs/2502.17392v1) [paper-pdf](http://arxiv.org/pdf/2502.17392v1)

**Authors**: Yangshijie Zhang

**Abstract**: Deep neural networks (DNNs) have achieved remarkable success in the field of natural language processing (NLP), leading to widely recognized applications such as ChatGPT. However, the vulnerability of these models to adversarial attacks remains a significant concern. Unlike continuous domains like images, text exists in a discrete space, making even minor alterations at the sentence, word, or character level easily perceptible to humans. This inherent discreteness also complicates the use of conventional optimization techniques, as text is non-differentiable. Previous research on adversarial attacks in text has focused on character-level, word-level, sentence-level, and multi-level approaches, all of which suffer from inefficiency or perceptibility issues due to the need for multiple queries or significant semantic shifts.   In this work, we introduce a novel adversarial attack method, Emoji-Attack, which leverages the manipulation of emojis to create subtle, yet effective, perturbations. Unlike character- and word-level strategies, Emoji-Attack targets emojis as a distinct layer of attack, resulting in less noticeable changes with minimal disruption to the text. This approach has been largely unexplored in previous research, which typically focuses on emoji insertion as an extension of character-level attacks. Our experiments demonstrate that Emoji-Attack achieves strong attack performance on both large and small models, making it a promising technique for enhancing adversarial robustness in NLP systems.

摘要: 深度神经网络(DNN)在自然语言处理(NLP)领域取得了显著的成功，产生了广泛认可的应用，如ChatGPT。然而，这些模型对对抗性攻击的脆弱性仍然是一个重大关切。与图像等连续领域不同，文本存在于离散的空间中，即使是句子、单词或字符级别的微小更改也很容易被人类察觉。这种固有的离散性也使传统优化技术的使用变得复杂，因为文本是不可区分的。以往对文本中敌意攻击的研究主要集中在字符级、词级、句子级和多层方法上，所有这些方法都存在效率低下或可感知性问题，原因是需要进行多个查询或显著的语义转换。在这项工作中，我们介绍了一种新的对抗性攻击方法，Emoji-Attack，它利用表情符号的操纵来制造微妙但有效的扰动。与字符和单词级别的策略不同，Emoji攻击将表情符号作为不同的攻击层，导致不太明显的变化，对文本的破坏最小。这种方法在以前的研究中基本上没有被探索过，这些研究通常专注于将表情符号插入作为字符级攻击的扩展。我们的实验表明，Emoji-Attack在大小模型上都具有很强的攻击性能，是一种在NLP系统中增强对手健壮性的很有前途的技术。



## **14. On the Vulnerability of Concept Erasure in Diffusion Models**

扩散模型中概念擦除的脆弱性 cs.LG

**SubmitDate**: 2025-02-24    [abs](http://arxiv.org/abs/2502.17537v1) [paper-pdf](http://arxiv.org/pdf/2502.17537v1)

**Authors**: Lucas Beerens, Alex D. Richardson, Kaicheng Zhang, Dongdong Chen

**Abstract**: The proliferation of text-to-image diffusion models has raised significant privacy and security concerns, particularly regarding the generation of copyrighted or harmful images. To address these issues, research on machine unlearning has developed various concept erasure methods, which aim to remove the effect of unwanted data through post-hoc training. However, we show these erasure techniques are vulnerable, where images of supposedly erased concepts can still be generated using adversarially crafted prompts. We introduce RECORD, a coordinate-descent-based algorithm that discovers prompts capable of eliciting the generation of erased content. We demonstrate that RECORD significantly beats the attack success rate of current state-of-the-art attack methods. Furthermore, our findings reveal that models subjected to concept erasure are more susceptible to adversarial attacks than previously anticipated, highlighting the urgency for more robust unlearning approaches. We open source all our code at https://github.com/LucasBeerens/RECORD

摘要: 文本到图像传播模式的激增引起了对隐私和安全的严重关切，特别是关于产生受版权保护或有害的图像。为了解决这些问题，机器遗忘的研究已经发展出各种概念擦除方法，旨在通过后自组织训练来消除不需要的数据的影响。然而，我们发现这些擦除技术是脆弱的，在这些技术中，应该被擦除的概念的图像仍然可以使用相反的精心制作的提示来生成。我们介绍了Record，这是一种基于坐标下降的算法，它发现能够引发删除内容生成的提示。我们证明，RECORD大大超过了当前最先进的攻击方法的攻击成功率。此外，我们的发现显示，受到概念删除的模型比之前预期的更容易受到对抗性攻击，这突显了更强大的遗忘方法的紧迫性。我们在https://github.com/LucasBeerens/RECORD上开放了我们所有的代码



## **15. Order Fairness Evaluation of DAG-based ledgers**

基于DAB的分类帐的订单公平性评估 cs.CR

19 pages with 9 pages dedicated to references and appendices, 23  figures, 13 of which are in the appendices

**SubmitDate**: 2025-02-24    [abs](http://arxiv.org/abs/2502.17270v1) [paper-pdf](http://arxiv.org/pdf/2502.17270v1)

**Authors**: Erwan Mahe, Sara Tucci-Piergiovanni

**Abstract**: Order fairness in distributed ledgers refers to properties that relate the order in which transactions are sent or received to the order in which they are eventually finalized, i.e., totally ordered. The study of such properties is relatively new and has been especially stimulated by the rise of Maximal Extractable Value (MEV) attacks in blockchain environments. Indeed, in many classical blockchain protocols, leaders are responsible for selecting the transactions to be included in blocks, which creates a clear vulnerability and opportunity for transaction order manipulation.   Unlike blockchains, DAG-based ledgers allow participants in the network to independently propose blocks, which are then arranged as vertices of a directed acyclic graph. Interestingly, leaders in DAG-based ledgers are elected only after the fact, once transactions are already part of the graph, to determine their total order. In other words, transactions are not chosen by single leaders; instead, they are collectively validated by the nodes, and leaders are only elected to establish an ordering. This approach intuitively reduces the risk of transaction manipulation and enhances fairness.   In this paper, we aim to quantify the capability of DAG-based ledgers to achieve order fairness. To this end, we define new variants of order fairness adapted to DAG-based ledgers and evaluate the impact of an adversary capable of compromising a limited number of nodes (below the one-third threshold) to reorder transactions. We analyze how often our order fairness properties are violated under different network conditions and parameterizations of the DAG algorithm, depending on the adversary's power.   Our study shows that DAG-based ledgers are still vulnerable to reordering attacks, as an adversary can coordinate a minority of Byzantine nodes to manipulate the DAG's structure.

摘要: 分布式分类账中的顺序公平性是指将发送或接收交易的顺序与最终确定的顺序(即完全有序)联系起来的属性。对这类属性的研究相对较新，尤其是区块链环境中最大可提取价值(MEV)攻击的兴起。事实上，在许多经典的区块链协议中，领导者负责选择要包含在区块中的交易，这为交易顺序操纵创造了明显的漏洞和机会。与区块链不同，基于DAG的分类账允许网络中的参与者独立提出区块，然后将这些区块排列为有向无环图的顶点。有趣的是，只有在交易已经成为图表的一部分后，才会在基于DAG的分类账中选出领导人，以确定其总顺序。换句话说，事务不是由单个领导者选择的；相反，它们由节点集体验证，而领导者只被选举来建立顺序。这种方法直观地降低了交易操纵的风险，提高了公平性。在本文中，我们的目标是量化基于DAG的分类帐实现顺序公平的能力。为此，我们定义了适用于基于DAG的分类账的顺序公平性的新变体，并评估了攻击者能够危害有限数量的节点(低于三分之一的阈值)来重新排序事务的影响。我们分析了在不同的网络条件和DAG算法的参数设置下，我们的顺序公平性被违反的频率，这取决于对手的力量。我们的研究表明，基于DAG的分类账仍然容易受到重新排序攻击，因为对手可以协调少数拜占庭节点来操纵DAG的结构。



## **16. REINFORCE Adversarial Attacks on Large Language Models: An Adaptive, Distributional, and Semantic Objective**

REINFORCE对大型语言模型的对抗攻击：自适应、分布和语义目标 cs.LG

30 pages, 6 figures, 15 tables

**SubmitDate**: 2025-02-24    [abs](http://arxiv.org/abs/2502.17254v1) [paper-pdf](http://arxiv.org/pdf/2502.17254v1)

**Authors**: Simon Geisler, Tom Wollschläger, M. H. I. Abdalla, Vincent Cohen-Addad, Johannes Gasteiger, Stephan Günnemann

**Abstract**: To circumvent the alignment of large language models (LLMs), current optimization-based adversarial attacks usually craft adversarial prompts by maximizing the likelihood of a so-called affirmative response. An affirmative response is a manually designed start of a harmful answer to an inappropriate request. While it is often easy to craft prompts that yield a substantial likelihood for the affirmative response, the attacked model frequently does not complete the response in a harmful manner. Moreover, the affirmative objective is usually not adapted to model-specific preferences and essentially ignores the fact that LLMs output a distribution over responses. If low attack success under such an objective is taken as a measure of robustness, the true robustness might be grossly overestimated. To alleviate these flaws, we propose an adaptive and semantic optimization problem over the population of responses. We derive a generally applicable objective via the REINFORCE policy-gradient formalism and demonstrate its efficacy with the state-of-the-art jailbreak algorithms Greedy Coordinate Gradient (GCG) and Projected Gradient Descent (PGD). For example, our objective doubles the attack success rate (ASR) on Llama3 and increases the ASR from 2% to 50% with circuit breaker defense.

摘要: 为了绕过大型语言模型(LLM)的对齐，当前基于优化的对抗性攻击通常通过最大化所谓肯定响应的可能性来创建对抗性提示。肯定答复是手动设计的对不适当请求的有害答复的开始。虽然通常很容易制定提示，以产生肯定响应的很大可能性，但被攻击的模型通常不会以有害的方式完成响应。此外，肯定的目标通常不适应特定于模型的偏好，基本上忽略了LLMS输出的分布高于响应的事实。如果在这样的目标下将低攻击成功率作为稳健性的衡量标准，则可能严重高估了真正的稳健性。为了克服这些缺陷，我们提出了一种基于响应总体的自适应语义优化问题。我们通过强化策略梯度理论推导出一个普遍适用的目标，并用最先进的越狱算法贪婪坐标梯度(GCG)和投影梯度下降(PGD)来验证其有效性。例如，我们的目标是使Llama3上的攻击成功率(ASR)翻一番，并通过断路器防御将ASR从2%提高到50%。



## **17. Adversarial Training for Defense Against Label Poisoning Attacks**

防御标签中毒攻击的对抗训练 cs.LG

Accepted at the International Conference on Learning Representations  (ICLR 2025)

**SubmitDate**: 2025-02-24    [abs](http://arxiv.org/abs/2502.17121v1) [paper-pdf](http://arxiv.org/pdf/2502.17121v1)

**Authors**: Melis Ilayda Bal, Volkan Cevher, Michael Muehlebach

**Abstract**: As machine learning models grow in complexity and increasingly rely on publicly sourced data, such as the human-annotated labels used in training large language models, they become more vulnerable to label poisoning attacks. These attacks, in which adversaries subtly alter the labels within a training dataset, can severely degrade model performance, posing significant risks in critical applications. In this paper, we propose FLORAL, a novel adversarial training defense strategy based on support vector machines (SVMs) to counter these threats. Utilizing a bilevel optimization framework, we cast the training process as a non-zero-sum Stackelberg game between an attacker, who strategically poisons critical training labels, and the model, which seeks to recover from such attacks. Our approach accommodates various model architectures and employs a projected gradient descent algorithm with kernel SVMs for adversarial training. We provide a theoretical analysis of our algorithm's convergence properties and empirically evaluate FLORAL's effectiveness across diverse classification tasks. Compared to robust baselines and foundation models such as RoBERTa, FLORAL consistently achieves higher robust accuracy under increasing attacker budgets. These results underscore the potential of FLORAL to enhance the resilience of machine learning models against label poisoning threats, thereby ensuring robust classification in adversarial settings.

摘要: 随着机器学习模型变得越来越复杂，并越来越依赖于公共来源的数据，例如用于训练大型语言模型的人类注释标签，它们变得更容易受到标签中毒攻击。在这些攻击中，攻击者巧妙地更改了训练数据集中的标签，可能会严重降低模型的性能，给关键应用程序带来重大风险。针对这些威胁，本文提出了一种新的基于支持向量机的对抗性训练防御策略FLOLAR。利用双层优化框架，我们将训练过程描述为攻击者和模型之间的非零和Stackelberg博弈，攻击者策略性地毒害关键的训练标签，而模型试图从此类攻击中恢复。我们的方法适应了不同的模型结构，并使用了一种带有核支持向量机的投影梯度下降算法进行对抗性训练。我们对算法的收敛特性进行了理论分析，并对FLORAL算法在不同分类任务上的有效性进行了实证评估。与罗伯塔等稳健的基线和基础模型相比，FLORAL在不断增加的攻击者预算下始终实现更高的稳健精度。这些结果强调了FLOLAR的潜力，以增强机器学习模型对标签中毒威胁的弹性，从而确保在对抗性环境中的稳健分类。



## **18. Improving the Transferability of Adversarial Examples by Inverse Knowledge Distillation**

通过反向知识蒸馏提高对抗示例的可移植性 cs.LG

**SubmitDate**: 2025-02-24    [abs](http://arxiv.org/abs/2502.17003v1) [paper-pdf](http://arxiv.org/pdf/2502.17003v1)

**Authors**: Wenyuan Wu, Zheng Liu, Yong Chen, Chao Su, Dezhong Peng, Xu Wang

**Abstract**: In recent years, the rapid development of deep neural networks has brought increased attention to the security and robustness of these models. While existing adversarial attack algorithms have demonstrated success in improving adversarial transferability, their performance remains suboptimal due to a lack of consideration for the discrepancies between target and source models. To address this limitation, we propose a novel method, Inverse Knowledge Distillation (IKD), designed to enhance adversarial transferability effectively. IKD introduces a distillation-inspired loss function that seamlessly integrates with gradient-based attack methods, promoting diversity in attack gradients and mitigating overfitting to specific model architectures. By diversifying gradients, IKD enables the generation of adversarial samples with superior generalization capabilities across different models, significantly enhancing their effectiveness in black-box attack scenarios. Extensive experiments on the ImageNet dataset validate the effectiveness of our approach, demonstrating substantial improvements in the transferability and attack success rates of adversarial samples across a wide range of models.

摘要: 近年来，深度神经网络的快速发展使得这些模型的安全性和稳健性受到越来越多的关注。虽然现有的对抗性攻击算法已经证明在改善对抗性可转移性方面取得了成功，但由于没有考虑目标和源模型之间的差异，它们的性能仍然不是最优的。针对这一局限性，我们提出了一种新的方法-逆知识蒸馏(IKD)，旨在有效地增强对抗转移能力。IKD引入了一个受蒸馏启发的损失函数，该函数与基于梯度的攻击方法无缝集成，促进了攻击梯度的多样性，并缓解了对特定模型体系结构的过度适应。通过使梯度多样化，IKD能够生成跨不同模型的具有卓越泛化能力的对抗性样本，显著增强其在黑盒攻击场景中的有效性。在ImageNet数据集上的广泛实验验证了我们方法的有效性，表明在广泛的模型范围内，敌方样本的可传递性和攻击成功率都有了显著的改善。



## **19. VGFL-SA: Vertical Graph Federated Learning Structure Attack Based on Contrastive Learning**

VGFL-SA：基于对比学习的垂直图联邦学习结构攻击 cs.LG

**SubmitDate**: 2025-02-24    [abs](http://arxiv.org/abs/2502.16793v1) [paper-pdf](http://arxiv.org/pdf/2502.16793v1)

**Authors**: Yang Chen, Bin Zhou

**Abstract**: Graph Neural Networks (GNNs) have gained attention for their ability to learn representations from graph data. Due to privacy concerns and conflicts of interest that prevent clients from directly sharing graph data with one another, Vertical Graph Federated Learning (VGFL) frameworks have been developed. Recent studies have shown that VGFL is vulnerable to adversarial attacks that degrade performance. However, it is a common problem that client nodes are often unlabeled in the realm of VGFL. Consequently, the existing attacks, which rely on the availability of labeling information to obtain gradients, are inherently constrained in their applicability. This limitation precludes their deployment in practical, real-world environments. To address the above problems, we propose a novel graph adversarial attack against VGFL, referred to as VGFL-SA, to degrade the performance of VGFL by modifying the local clients structure without using labels. Specifically, VGFL-SA uses a contrastive learning method to complete the attack before the local clients are trained. VGFL-SA first accesses the graph structure and node feature information of the poisoned clients, and generates the contrastive views by node-degree-based edge augmentation and feature shuffling augmentation. Then, VGFL-SA uses the shared graph encoder to get the embedding of each view, and the gradients of the adjacency matrices are obtained by the contrastive function. Finally, perturbed edges are generated using gradient modification rules. We validated the performance of VGFL-SA by performing a node classification task on real-world datasets, and the results show that VGFL-SA achieves good attack effectiveness and transferability.

摘要: 图形神经网络(GNN)因其从图形数据中学习表示的能力而受到关注。由于隐私问题和利益冲突阻碍了客户之间直接共享图形数据，垂直图形联合学习(VGFL)框架已经开发出来。最近的研究表明，VGFL很容易受到降低性能的对抗性攻击。然而，在VGFL领域中，一个常见的问题是客户端节点通常是未标记的。因此，现有的攻击依赖于标记信息的可用性来获得梯度，其适用性受到固有的限制。这一限制排除了它们在实际、真实环境中的部署。针对上述问题，我们提出了一种新的针对VGFL的图对抗攻击，称为VGFL-SA，通过修改本地客户端结构而不使用标签来降低VGFL的性能。具体地说，VGFL-SA使用对比学习方法在本地客户端训练之前完成攻击。VGFL-SA首先获取中毒客户端的图结构和节点特征信息，然后通过基于节点度的边增强和特征置乱增强生成对比视图。然后，VGFL-SA使用共享图编码器得到每个视点的嵌入，并通过对比函数得到邻接矩阵的梯度。最后，使用梯度修正规则生成扰动边缘。我们通过在真实数据集上执行节点分类任务来验证VGFL-SA的性能，结果表明VGFL-SA具有良好的攻击有效性和可转移性。



## **20. Dysca: A Dynamic and Scalable Benchmark for Evaluating Perception Ability of LVLMs**

Dysca：评估LVLM感知能力的动态和可扩展基准 cs.CV

Accepted by ICLR2025

**SubmitDate**: 2025-02-24    [abs](http://arxiv.org/abs/2406.18849v4) [paper-pdf](http://arxiv.org/pdf/2406.18849v4)

**Authors**: Jie Zhang, Zhongqi Wang, Mengqi Lei, Zheng Yuan, Bei Yan, Shiguang Shan, Xilin Chen

**Abstract**: Currently many benchmarks have been proposed to evaluate the perception ability of the Large Vision-Language Models (LVLMs). However, most benchmarks conduct questions by selecting images from existing datasets, resulting in the potential data leakage. Besides, these benchmarks merely focus on evaluating LVLMs on the realistic style images and clean scenarios, leaving the multi-stylized images and noisy scenarios unexplored. In response to these challenges, we propose a dynamic and scalable benchmark named Dysca for evaluating LVLMs by leveraging synthesis images. Specifically, we leverage Stable Diffusion and design a rule-based method to dynamically generate novel images, questions and the corresponding answers. We consider 51 kinds of image styles and evaluate the perception capability in 20 subtasks. Moreover, we conduct evaluations under 4 scenarios (i.e., Clean, Corruption, Print Attacking and Adversarial Attacking) and 3 question types (i.e., Multi-choices, True-or-false and Free-form). Thanks to the generative paradigm, Dysca serves as a scalable benchmark for easily adding new subtasks and scenarios. A total of 24 advanced open-source LVLMs and 2 close-source LVLMs are evaluated on Dysca, revealing the drawbacks of current LVLMs. The benchmark is released at https://github.com/Robin-WZQ/Dysca.

摘要: 目前，人们已经提出了许多基准来评估大型视觉语言模型的感知能力。然而，大多数基准测试通过从现有数据集中选择图像来进行问题，从而导致潜在的数据泄漏。此外，这些基准只关注真实感风格的图像和干净的场景来评估LVLMS，而对多风格化的图像和噪声场景没有进行探索。为了应对这些挑战，我们提出了一种动态的、可扩展的基准测试DYSCA，用于利用合成图像来评估LVLMS。具体地说，我们利用稳定扩散并设计了一种基于规则的方法来动态生成新的图像、问题和相应的答案。我们考虑了51种图像风格，并在20个子任务中评估了感知能力。此外，我们还在4个场景(即廉洁、腐败、打印攻击和对抗性攻击)和3个问题类型(即多项选择、对错和自由形式)下进行了评估。多亏了生成性范式，Dysca成为了一个可伸缩的基准，可以轻松添加新的子任务和场景。在Dysca上对24个先进的开源LVLMS和2个闭源LVLMS进行了评估，揭示了现有LVLMS的不足。该基准在https://github.com/Robin-WZQ/Dysca.上发布



## **21. Guardians of the Agentic System: Preventing Many Shots Jailbreak with Agentic System**

紧急系统的守护者：用紧急系统防止多次枪击越狱 cs.CR

18 pages, 7 figures

**SubmitDate**: 2025-02-23    [abs](http://arxiv.org/abs/2502.16750v1) [paper-pdf](http://arxiv.org/pdf/2502.16750v1)

**Authors**: Saikat Barua, Mostafizur Rahman, Md Jafor Sadek, Rafiul Islam, Shehnaz Khaled, Ahmedul Kabir

**Abstract**: The autonomous AI agents using large language models can create undeniable values in all span of the society but they face security threats from adversaries that warrants immediate protective solutions because trust and safety issues arise. Considering the many-shot jailbreaking and deceptive alignment as some of the main advanced attacks, that cannot be mitigated by the static guardrails used during the supervised training, points out a crucial research priority for real world robustness. The combination of static guardrails in dynamic multi-agent system fails to defend against those attacks. We intend to enhance security for LLM-based agents through the development of new evaluation frameworks which identify and counter threats for safe operational deployment. Our work uses three examination methods to detect rogue agents through a Reverse Turing Test and analyze deceptive alignment through multi-agent simulations and develops an anti-jailbreaking system by testing it with GEMINI 1.5 pro and llama-3.3-70B, deepseek r1 models using tool-mediated adversarial scenarios. The detection capabilities are strong such as 94\% accuracy for GEMINI 1.5 pro yet the system suffers persistent vulnerabilities when under long attacks as prompt length increases attack success rates (ASR) and diversity metrics become ineffective in prediction while revealing multiple complex system faults. The findings demonstrate the necessity of adopting flexible security systems based on active monitoring that can be performed by the agents themselves together with adaptable interventions by system admin as the current models can create vulnerabilities that can lead to the unreliable and vulnerable system. So, in our work, we try to address such situations and propose a comprehensive framework to counteract the security issues.

摘要: 使用大型语言模型的自主人工智能代理可以在社会各个领域创造不可否认的价值，但他们面临来自对手的安全威胁，需要立即采取保护性解决方案，因为信任和安全问题会出现。考虑到多发越狱和欺骗性对准是一些主要的高级攻击，在监督训练期间使用的静态护栏无法减轻这些攻击，指出了现实世界健壮性的关键研究重点。动态多智能体系统中静态护栏的组合不能抵抗这些攻击。我们打算通过制定新的评估框架，确定和应对安全行动部署所面临的威胁，从而加强基于LLM的特工的安全。我们的工作使用了三种检测方法来通过反向图灵测试来检测流氓代理，并通过多代理模拟来分析欺骗性比对，并开发了一个反越狱系统，通过使用Gemini 1.5Pro和Llama-3.3-70B、使用工具中介的对抗场景来测试DeepSeek R1模型来开发反越狱系统。Gemini 1.5 PRO具有很强的检测能力，如94%的准确率，但在长时间攻击下，随着提示长度的增加，攻击成功率(ASR)增加，多样性度量在预测多个复杂系统故障时变得无效，系统存在持续漏洞。这些发现证明了采用基于主动监控的灵活安全系统的必要性，该系统可以由代理自己执行，并由系统管理员进行适应性干预，因为当前的模型可能会产生漏洞，从而导致系统不可靠和易受攻击。因此，在我们的工作中，我们试图解决这些情况，并提出一个全面的框架来对抗安全问题。



## **22. Keeping up with dynamic attackers: Certifying robustness to adaptive online data poisoning**

跟上动态攻击者：认证自适应在线数据中毒的稳健性 cs.LG

Proceedings of the 28th International Conference on Artificial  Intelligence and Statistics (AISTATS) 2025, Mai Khao, Thailand. PMLR: Volume  258

**SubmitDate**: 2025-02-23    [abs](http://arxiv.org/abs/2502.16737v1) [paper-pdf](http://arxiv.org/pdf/2502.16737v1)

**Authors**: Avinandan Bose, Laurent Lessard, Maryam Fazel, Krishnamurthy Dj Dvijotham

**Abstract**: The rise of foundation models fine-tuned on human feedback from potentially untrusted users has increased the risk of adversarial data poisoning, necessitating the study of robustness of learning algorithms against such attacks. Existing research on provable certified robustness against data poisoning attacks primarily focuses on certifying robustness for static adversaries who modify a fraction of the dataset used to train the model before the training algorithm is applied. In practice, particularly when learning from human feedback in an online sense, adversaries can observe and react to the learning process and inject poisoned samples that optimize adversarial objectives better than when they are restricted to poisoning a static dataset once, before the learning algorithm is applied. Indeed, it has been shown in prior work that online dynamic adversaries can be significantly more powerful than static ones. We present a novel framework for computing certified bounds on the impact of dynamic poisoning, and use these certificates to design robust learning algorithms. We give an illustration of the framework for the mean estimation and binary classification problems and outline directions for extending this in further work. The code to implement our certificates and replicate our results is available at https://github.com/Avinandan22/Certified-Robustness.

摘要: 根据潜在不可信用户的人类反馈进行微调的基础模型的兴起，增加了敌意数据中毒的风险，因此有必要研究学习算法对此类攻击的稳健性。现有的针对数据中毒攻击的可证明认证稳健性的研究主要集中在认证静态攻击者的健壮性，这些静态攻击者在应用训练算法之前修改了用于训练模型的数据集的一小部分。在实践中，特别是在从在线意义上的人类反馈学习时，攻击者可以观察学习过程并对其做出反应，并注入有毒样本，以更好地优化对抗目标，而不是在应用学习算法之前限制他们一次毒化静态数据集。事实上，以前的工作已经表明，在线动态对手可能比静态对手强大得多。我们提出了一种新的框架来计算动态中毒影响的认证界，并使用这些证书来设计健壮的学习算法。我们给出了均值估计和二分类问题的框架，并概述了在进一步工作中扩展该框架的方向。实现我们的证书和复制我们的结果的代码可在https://github.com/Avinandan22/Certified-Robustness.上获得



## **23. Towards Optimal Adversarial Robust Reinforcement Learning with Infinity Measurement Error**

迈向具有无限测量误差的最佳对抗鲁棒强化学习 cs.LG

arXiv admin note: substantial text overlap with arXiv:2402.02165

**SubmitDate**: 2025-02-23    [abs](http://arxiv.org/abs/2502.16734v1) [paper-pdf](http://arxiv.org/pdf/2502.16734v1)

**Authors**: Haoran Li, Zicheng Zhang, Wang Luo, Congying Han, Jiayu Lv, Tiande Guo, Yudong Hu

**Abstract**: Ensuring the robustness of deep reinforcement learning (DRL) agents against adversarial attacks is critical for their trustworthy deployment. Recent research highlights the challenges of achieving state-adversarial robustness and suggests that an optimal robust policy (ORP) does not always exist, complicating the enforcement of strict robustness constraints. In this paper, we further explore the concept of ORP. We first introduce the Intrinsic State-adversarial Markov Decision Process (ISA-MDP), a novel formulation where adversaries cannot fundamentally alter the intrinsic nature of state observations. ISA-MDP, supported by empirical and theoretical evidence, universally characterizes decision-making under state-adversarial paradigms. We rigorously prove that within ISA-MDP, a deterministic and stationary ORP exists, aligning with the Bellman optimal policy. Our findings theoretically reveal that improving DRL robustness does not necessarily compromise performance in natural environments. Furthermore, we demonstrate the necessity of infinity measurement error (IME) in both $Q$-function and probability spaces to achieve ORP, unveiling vulnerabilities of previous DRL algorithms that rely on $1$-measurement errors. Motivated by these insights, we develop the Consistent Adversarial Robust Reinforcement Learning (CAR-RL) framework, which optimizes surrogates of IME. We apply CAR-RL to both value-based and policy-based DRL algorithms, achieving superior performance and validating our theoretical analysis.

摘要: 确保深度强化学习(DRL)代理对对手攻击的健壮性是其可信部署的关键。最近的研究强调了实现状态对抗健壮性的挑战，并表明最优健壮性策略(ORP)并不总是存在的，这使得严格健壮性约束的实施复杂化。在本文中，我们进一步探讨了ORP的概念。我们首先介绍了本征状态-对抗马尔可夫决策过程(ISA-MDP)，这是一种新的形式，对手不能从根本上改变状态观测的内在性质。ISA-MDP得到了经验和理论证据的支持，是国家对抗范式下决策的普遍特征。我们严格地证明了在ISA-MDP中，存在与Bellman最优策略一致的确定性且平稳的ORP。我们的发现在理论上表明，提高DRL的健壮性并不一定会损害自然环境中的性能。此外，我们论证了在$q$函数和概率空间中无穷大测量误差(IME)实现ORP的必要性，揭示了以前依赖于$1$测量误差的DRL算法的弱点。受此启发，我们提出了一致对抗性稳健强化学习(CAR-RL)框架，该框架对输入法的代理进行了优化。我们将CAR-RL应用于基于值的DRL算法和基于策略的DRL算法，取得了优异的性能，验证了我们的理论分析。



## **24. Towards LLM Unlearning Resilient to Relearning Attacks: A Sharpness-Aware Minimization Perspective and Beyond**

迈向LLM摆脱学习对重新学习攻击的弹性：敏锐意识的最小化视角及超越 cs.LG

**SubmitDate**: 2025-02-23    [abs](http://arxiv.org/abs/2502.05374v2) [paper-pdf](http://arxiv.org/pdf/2502.05374v2)

**Authors**: Chongyu Fan, Jinghan Jia, Yihua Zhang, Anil Ramakrishna, Mingyi Hong, Sijia Liu

**Abstract**: The LLM unlearning technique has recently been introduced to comply with data regulations and address the safety and ethical concerns of LLMs by removing the undesired data-model influence. However, state-of-the-art unlearning methods face a critical vulnerability: they are susceptible to ``relearning'' the removed information from a small number of forget data points, known as relearning attacks. In this paper, we systematically investigate how to make unlearned models robust against such attacks. For the first time, we establish a connection between robust unlearning and sharpness-aware minimization (SAM) through a unified robust optimization framework, in an analogy to adversarial training designed to defend against adversarial attacks. Our analysis for SAM reveals that smoothness optimization plays a pivotal role in mitigating relearning attacks. Thus, we further explore diverse smoothing strategies to enhance unlearning robustness. Extensive experiments on benchmark datasets, including WMDP and MUSE, demonstrate that SAM and other smoothness optimization approaches consistently improve the resistance of LLM unlearning to relearning attacks. Notably, smoothness-enhanced unlearning also helps defend against (input-level) jailbreaking attacks, broadening our proposal's impact in robustifying LLM unlearning. Codes are available at https://github.com/OPTML-Group/Unlearn-Smooth.

摘要: 最近引入了LLM解除学习技术，以遵守数据法规，并通过消除不希望看到的数据模型影响来解决LLM的安全和伦理问题。然而，最先进的遗忘方法面临着一个严重的漏洞：它们容易受到从少数忘记数据点移除的信息的“重新学习”，称为重新学习攻击。在本文中，我们系统地研究了如何使未学习模型对此类攻击具有健壮性。第一次，我们通过一个统一的稳健优化框架在稳健遗忘和敏锐度感知最小化(SAM)之间建立了联系，类似于旨在防御对手攻击的对抗性训练。我们对SAM的分析表明，平滑优化在减轻再学习攻击方面起着关键作用。因此，我们进一步探索不同的平滑策略来增强遗忘的稳健性。在WMDP和MUSE等基准数据集上的大量实验表明，SAM和其他平滑优化方法一致地提高了LLM遗忘对重新学习攻击的抵抗力。值得注意的是，流畅性增强的遗忘也有助于防御(输入级)越狱攻击，扩大了我们的提议在强化LLM遗忘方面的影响。有关代码，请访问https://github.com/OPTML-Group/Unlearn-Smooth.



## **25. Uncovering the Hidden Threat of Text Watermarking from Users with Cross-Lingual Knowledge**

从具有跨语言知识的用户手中发现文本水印的隐藏威胁 cs.CL

9 pages

**SubmitDate**: 2025-02-23    [abs](http://arxiv.org/abs/2502.16699v1) [paper-pdf](http://arxiv.org/pdf/2502.16699v1)

**Authors**: Mansour Al Ghanim, Jiaqi Xue, Rochana Prih Hastuti, Mengxin Zheng, Yan Solihin, Qian Lou

**Abstract**: In this study, we delve into the hidden threats posed to text watermarking by users with cross-lingual knowledge. While most research focuses on watermarking methods for English, there is a significant gap in evaluating these methods in cross-lingual contexts. This oversight neglects critical adversary scenarios involving cross-lingual users, creating uncertainty regarding the effectiveness of cross-lingual watermarking. We assess four watermarking techniques across four linguistically rich languages, examining watermark resilience and text quality across various parameters and attacks. Our focus is on a realistic scenario featuring adversaries with cross-lingual expertise, evaluating the adequacy of current watermarking methods against such challenges.

摘要: 在这项研究中，我们深入研究了具有跨语言知识的用户对文本水印构成的隐藏威胁。虽然大多数研究都集中在英语的水印方法上，但在跨语言环境中评估这些方法存在显着差距。这种疏忽忽视了涉及跨语言用户的关键对手场景，从而产生了跨语言水印有效性的不确定性。我们评估了四种语言丰富的语言的四种水印技术，检查了各种参数和攻击的水印弹性和文本质量。我们的重点是以具有跨语言专业知识的对手为特色的现实场景，评估当前水印方法应对此类挑战的充分性。



## **26. AdverX-Ray: Ensuring X-Ray Integrity Through Frequency-Sensitive Adversarial VAEs**

DTS X射线：通过频率敏感对抗VAE确保X射线完整性 cs.CV

SPIE Medical Imaging 2025 Runner-up 2025 Robert F. Wagner  All-Conference Best Student Paper Award

**SubmitDate**: 2025-02-23    [abs](http://arxiv.org/abs/2502.16610v1) [paper-pdf](http://arxiv.org/pdf/2502.16610v1)

**Authors**: Francisco Caetano, Christiaan Viviers, Lena Filatova, Peter H. N. de With, Fons van der Sommen

**Abstract**: Ensuring the quality and integrity of medical images is crucial for maintaining diagnostic accuracy in deep learning-based Computer-Aided Diagnosis and Computer-Aided Detection (CAD) systems. Covariate shifts are subtle variations in the data distribution caused by different imaging devices or settings and can severely degrade model performance, similar to the effects of adversarial attacks. Therefore, it is vital to have a lightweight and fast method to assess the quality of these images prior to using CAD models. AdverX-Ray addresses this need by serving as an image-quality assessment layer, designed to detect covariate shifts effectively. This Adversarial Variational Autoencoder prioritizes the discriminator's role, using the suboptimal outputs of the generator as negative samples to fine-tune the discriminator's ability to identify high-frequency artifacts. Images generated by adversarial networks often exhibit severe high-frequency artifacts, guiding the discriminator to focus excessively on these components. This makes the discriminator ideal for this approach. Trained on patches from X-ray images of specific machine models, AdverX-Ray can evaluate whether a scan matches the training distribution, or if a scan from the same machine is captured under different settings. Extensive comparisons with various OOD detection methods show that AdverX-Ray significantly outperforms existing techniques, achieving a 96.2% average AUROC using only 64 random patches from an X-ray. Its lightweight and fast architecture makes it suitable for real-time applications, enhancing the reliability of medical imaging systems. The code and pretrained models are publicly available.

摘要: 在基于深度学习的计算机辅助诊断和计算机辅助检测(CAD)系统中，确保医学图像的质量和完整性对于保持诊断的准确性至关重要。协变量漂移是由不同成像设备或设置引起的数据分布中的细微变化，可能会严重降低模型的性能，类似于对抗性攻击的影响。因此，在使用CAD模型之前，有一种轻量级且快速的方法来评估这些图像的质量是至关重要的。AdverX-Ray通过作为图像质量评估层来满足这一需求，旨在有效地检测协变量偏移。这种对抗性变分自动编码器优先考虑鉴别器的作用，使用发生器的次优输出作为负样本来微调鉴别器识别高频伪像的能力。对抗性网络生成的图像通常表现出严重的高频伪影，引导鉴别器过度关注这些分量。这使得鉴别器成为这种方法的理想选择。在特定机器型号的X射线图像的补丁上进行训练后，AdverX-Ray可以评估扫描是否与训练分布匹配，或者来自同一机器的扫描是否在不同设置下捕获。与各种面向对象检测方法的广泛比较表明，AdverX-Ray的性能明显优于现有技术，仅使用来自一条X射线的随机斑块就可以获得96.2%的平均AUROC。其轻量级和快速的体系结构使其适合实时应用，提高了医学成像系统的可靠性。代码和预先训练的模型是公开提供的。



## **27. Tracking the Copyright of Large Vision-Language Models through Parameter Learning Adversarial Images**

通过参数学习对抗图像跟踪大型视觉语言模型的版权 cs.AI

Accepted to ICLR 2025

**SubmitDate**: 2025-02-23    [abs](http://arxiv.org/abs/2502.16593v1) [paper-pdf](http://arxiv.org/pdf/2502.16593v1)

**Authors**: Yubo Wang, Jianting Tang, Chaohu Liu, Linli Xu

**Abstract**: Large vision-language models (LVLMs) have demonstrated remarkable image understanding and dialogue capabilities, allowing them to handle a variety of visual question answering tasks. However, their widespread availability raises concerns about unauthorized usage and copyright infringement, where users or individuals can develop their own LVLMs by fine-tuning published models. In this paper, we propose a novel method called Parameter Learning Attack (PLA) for tracking the copyright of LVLMs without modifying the original model. Specifically, we construct adversarial images through targeted attacks against the original model, enabling it to generate specific outputs. To ensure these attacks remain effective on potential fine-tuned models to trigger copyright tracking, we allow the original model to learn the trigger images by updating parameters in the opposite direction during the adversarial attack process. Notably, the proposed method can be applied after the release of the original model, thus not affecting the model's performance and behavior. To simulate real-world applications, we fine-tune the original model using various strategies across diverse datasets, creating a range of models for copyright verification. Extensive experiments demonstrate that our method can more effectively identify the original copyright of fine-tuned models compared to baseline methods. Therefore, this work provides a powerful tool for tracking copyrights and detecting unlicensed usage of LVLMs.

摘要: 大型视觉语言模型(LVLM)已经显示出非凡的图像理解和对话能力，使它们能够处理各种视觉问题回答任务。然而，它们的广泛使用引发了人们对未经授权使用和侵犯版权的担忧，在这种情况下，用户或个人可以通过微调已发布的模型来开发自己的LVLM。在本文中，我们提出了一种称为参数学习攻击的新方法，该方法在不修改原始模型的情况下跟踪LVLMS的版权。具体地说，我们通过对原始模型进行有针对性的攻击来构建对抗性图像，使其能够生成特定的输出。为了确保这些攻击在触发版权跟踪的潜在微调模型上保持有效，我们允许原始模型在对抗性攻击过程中通过反向更新参数来学习触发图像。值得注意的是，所提出的方法可以在原始模型发布之后应用，因此不会影响模型的性能和行为。为了模拟真实世界的应用程序，我们使用不同的策略对原始模型进行微调，创建了一系列用于版权验证的模型。大量实验表明，与基线方法相比，该方法能更有效地识别微调模型的原始版权。因此，这项工作为跟踪版权和检测未经许可的LVLM使用提供了一个强大的工具。



## **28. Robust Kernel Hypothesis Testing under Data Corruption**

数据腐败下的鲁棒核假设测试 stat.ML

22 pages, 2 figures, 2 algorithms

**SubmitDate**: 2025-02-23    [abs](http://arxiv.org/abs/2405.19912v2) [paper-pdf](http://arxiv.org/pdf/2405.19912v2)

**Authors**: Antonin Schrab, Ilmun Kim

**Abstract**: We propose a general method for constructing robust permutation tests under data corruption. The proposed tests effectively control the non-asymptotic type I error under data corruption, and we prove their consistency in power under minimal conditions. This contributes to the practical deployment of hypothesis tests for real-world applications with potential adversarial attacks. For the two-sample and independence settings, we show that our kernel robust tests are minimax optimal, in the sense that they are guaranteed to be non-asymptotically powerful against alternatives uniformly separated from the null in the kernel MMD and HSIC metrics at some optimal rate (tight with matching lower bound). We point out that existing differentially private tests can be adapted to be robust to data corruption, and we demonstrate in experiments that our proposed tests achieve much higher power than these private tests. Finally, we provide publicly available implementations and empirically illustrate the practicality of our robust tests.

摘要: 我们提出了一种在数据损坏情况下构造稳健置换测试的通用方法。所提出的测试有效地控制了数据损坏情况下的非渐近I类错误，并在最小条件下证明了它们在功率上的一致性。这有助于为具有潜在对手攻击的真实世界应用程序实际部署假设检验。对于两样本和独立设置，我们证明了我们的核稳健测试是极小极大最优的，在某种意义上，它们对于在核MMD和HSIC度量中以某种最优率(紧与匹配下界)从零一致分离的备选方案被保证是非渐近强大的。我们指出，现有的差异私有测试可以被改造成对数据损坏具有健壮性，并且我们在实验中证明了我们提出的测试比这些私有测试获得了更高的能力。最后，我们提供了公开可用的实现，并经验地说明了我们的健壮测试的实用性。



## **29. Certified Causal Defense with Generalizable Robustness**

经过认证的因果辩护，具有可概括的稳健性 cs.LG

Submitted to AAAI

**SubmitDate**: 2025-02-23    [abs](http://arxiv.org/abs/2408.15451v2) [paper-pdf](http://arxiv.org/pdf/2408.15451v2)

**Authors**: Yiran Qiao, Yu Yin, Chen Chen, Jing Ma

**Abstract**: While machine learning models have proven effective across various scenarios, it is widely acknowledged that many models are vulnerable to adversarial attacks. Recently, there have emerged numerous efforts in adversarial defense. Among them, certified defense is well known for its theoretical guarantees against arbitrary adversarial perturbations on input within a certain range (e.g., $l_2$ ball). However, most existing works in this line struggle to generalize their certified robustness in other data domains with distribution shifts. This issue is rooted in the difficulty of eliminating the negative impact of spurious correlations on robustness in different domains. To address this problem, in this work, we propose a novel certified defense framework GLEAN, which incorporates a causal perspective into the generalization problem in certified defense. More specifically, our framework integrates a certifiable causal factor learning component to disentangle the causal relations and spurious correlations between input and label, and thereby exclude the negative effect of spurious correlations on defense. On top of that, we design a causally certified defense strategy to handle adversarial attacks on latent causal factors. In this way, our framework is not only robust against malicious noises on data in the training distribution but also can generalize its robustness across domains with distribution shifts. Extensive experiments on benchmark datasets validate the superiority of our framework in certified robustness generalization in different data domains. Code is available in the supplementary materials.

摘要: 虽然机器学习模型已被证明在各种情况下都有效，但人们普遍认为，许多模型容易受到对手攻击。最近，在对抗性防御方面出现了许多努力。其中，认证防御以其在一定范围内(例如，$L_2$球)对输入的任意对抗性扰动的理论保证而闻名。然而，这一领域的大多数现有工作都很难在具有分布偏移的其他数据域中推广其已证明的健壮性。这个问题的根源在于很难消除虚假相关性对不同领域稳健性的负面影响。针对这一问题，在本文中，我们提出了一种新的认证防御框架GLEAN，该框架将因果视角融入到认证防御的泛化问题中。更具体地说，我们的框架集成了一个可证明的因果因素学习组件，以分离输入和标签之间的因果关系和伪关联，从而排除伪关联对防御的负面影响。最重要的是，我们设计了一个因果认证的防御策略来处理对潜在因果因素的对抗性攻击。这样，我们的框架不仅对训练分布中数据的恶意噪声具有健壮性，而且可以通过分布偏移来推广其跨域的健壮性。在基准数据集上的大量实验验证了该框架在不同数据域的健壮性泛化方面的优越性。代码可在补充材料中找到。



## **30. Unified Prompt Attack Against Text-to-Image Generation Models**

针对文本到图像生成模型的统一提示攻击 cs.CV

Accepted by IEEE T-PAMI 2025

**SubmitDate**: 2025-02-23    [abs](http://arxiv.org/abs/2502.16423v1) [paper-pdf](http://arxiv.org/pdf/2502.16423v1)

**Authors**: Duo Peng, Qiuhong Ke, Mark He Huang, Ping Hu, Jun Liu

**Abstract**: Text-to-Image (T2I) models have advanced significantly, but their growing popularity raises security concerns due to their potential to generate harmful images. To address these issues, we propose UPAM, a novel framework to evaluate the robustness of T2I models from an attack perspective. Unlike prior methods that focus solely on textual defenses, UPAM unifies the attack on both textual and visual defenses. Additionally, it enables gradient-based optimization, overcoming reliance on enumeration for improved efficiency and effectiveness. To handle cases where T2I models block image outputs due to defenses, we introduce Sphere-Probing Learning (SPL) to enable optimization even without image results. Following SPL, our model bypasses defenses, inducing the generation of harmful content. To ensure semantic alignment with attacker intent, we propose Semantic-Enhancing Learning (SEL) for precise semantic control. UPAM also prioritizes the naturalness of adversarial prompts using In-context Naturalness Enhancement (INE), making them harder for human examiners to detect. Additionally, we address the issue of iterative queries--common in prior methods and easily detectable by API defenders--by introducing Transferable Attack Learning (TAL), allowing effective attacks with minimal queries. Extensive experiments validate UPAM's superiority in effectiveness, efficiency, naturalness, and low query detection rates.

摘要: 文本到图像(T2I)模式已经有了很大的进步，但由于它们可能生成有害的图像，因此它们越来越受欢迎，引发了安全问题。为了解决这些问题，我们提出了一种从攻击角度评估T2I模型健壮性的新框架UPAM。与以前只关注文本防御的方法不同，UPAM将对文本防御和视觉防御的攻击统一起来。此外，它还支持基于梯度的优化，克服了对枚举的依赖，从而提高了效率和效果。为了处理T2I模型由于防御而阻止图像输出的情况，我们引入了球面探测学习(SPL)，即使在没有图像结果的情况下也能实现优化。在SPL之后，我们的模型绕过了防御，诱导了有害内容的生成。为了确保语义与攻击者意图的一致性，我们提出了语义增强学习(SEL)来进行精确的语义控制。UPAM还使用上下文中的自然度增强(INE)对对抗性提示的自然性进行优先排序，使人类审查员更难发现它们。此外，我们通过引入可转移攻击学习(TAL)来解决迭代查询的问题--迭代查询在以前的方法中很常见，并且很容易被API捍卫者检测到，从而允许以最少的查询进行有效的攻击。大量实验验证了UPAM在有效性、效率、自然度和较低的查询检测率方面的优势。



## **31. FedNIA: Noise-Induced Activation Analysis for Mitigating Data Poisoning in FL**

FedNIA：缓解FL数据中毒的噪音诱导激活分析 cs.LG

**SubmitDate**: 2025-02-23    [abs](http://arxiv.org/abs/2502.16396v1) [paper-pdf](http://arxiv.org/pdf/2502.16396v1)

**Authors**: Ehsan Hallaji, Roozbeh Razavi-Far, Mehrdad Saif

**Abstract**: Federated learning systems are increasingly threatened by data poisoning attacks, where malicious clients compromise global models by contributing tampered updates. Existing defenses often rely on impractical assumptions, such as access to a central test dataset, or fail to generalize across diverse attack types, particularly those involving multiple malicious clients working collaboratively. To address this, we propose Federated Noise-Induced Activation Analysis (FedNIA), a novel defense framework to identify and exclude adversarial clients without relying on any central test dataset. FedNIA injects random noise inputs to analyze the layerwise activation patterns in client models leveraging an autoencoder that detects abnormal behaviors indicative of data poisoning. FedNIA can defend against diverse attack types, including sample poisoning, label flipping, and backdoors, even in scenarios with multiple attacking nodes. Experimental results on non-iid federated datasets demonstrate its effectiveness and robustness, underscoring its potential as a foundational approach for enhancing the security of federated learning systems.

摘要: 联邦学习系统越来越受到数据中毒攻击的威胁，恶意客户端通过提供被篡改的更新来危害全球模型。现有的防御通常依赖于不切实际的假设，例如访问中央测试数据集，或者无法概括各种攻击类型，特别是涉及多个恶意客户端协同工作的攻击类型。为了解决这一问题，我们提出了联邦噪声诱导激活分析(FedNIA)，这是一个新的防御框架，可以识别和排除恶意客户端，而不依赖于任何中央测试数据集。FedNIA利用自动编码器检测指示数据中毒的异常行为，注入随机噪声输入来分析客户端模型中的LayerWise激活模式。FedNIA可以防御多种攻击类型，包括样本中毒、标签翻转和后门，即使在具有多个攻击节点的情况下也是如此。在非IID联合数据集上的实验结果证明了该方法的有效性和稳健性，强调了它作为提高联合学习系统安全性的基础方法的潜力。



## **32. A Framework for Evaluating Vision-Language Model Safety: Building Trust in AI for Public Sector Applications**

评估视觉语言模型安全性的框架：为公共部门应用建立对人工智能的信任 cs.CY

AAAI 2025 Workshop on AI for Social Impact: Bridging Innovations in  Finance, Social Media, and Crime Prevention

**SubmitDate**: 2025-02-22    [abs](http://arxiv.org/abs/2502.16361v1) [paper-pdf](http://arxiv.org/pdf/2502.16361v1)

**Authors**: Maisha Binte Rashid, Pablo Rivas

**Abstract**: Vision-Language Models (VLMs) are increasingly deployed in public sector missions, necessitating robust evaluation of their safety and vulnerability to adversarial attacks. This paper introduces a novel framework to quantify adversarial risks in VLMs. We analyze model performance under Gaussian, salt-and-pepper, and uniform noise, identifying misclassification thresholds and deriving composite noise patches and saliency patterns that highlight vulnerable regions. These patterns are compared against the Fast Gradient Sign Method (FGSM) to assess their adversarial effectiveness. We propose a new Vulnerability Score that combines the impact of random noise and adversarial attacks, providing a comprehensive metric for evaluating model robustness.

摘要: 视觉语言模型（VLM）越来越多地部署在公共部门任务中，因此需要对其安全性和对对抗攻击的脆弱性进行严格评估。本文引入了一种新颖的框架来量化VLM中的对抗风险。我们分析高斯、椒盐和均匀噪音下的模型性能，识别错误分类阈值并推导出突出脆弱区域的复合噪音补丁和显着模式。将这些模式与快速梯度符号法（FGSM）进行比较，以评估其对抗有效性。我们提出了一种新的漏洞分数，它结合了随机噪音和对抗攻击的影响，为评估模型稳健性提供了全面的指标。



## **33. Verification of Bit-Flip Attacks against Quantized Neural Networks**

针对量化神经网络的位翻转攻击的验证 cs.CR

37 pages, 13 figures, 14 tables

**SubmitDate**: 2025-02-22    [abs](http://arxiv.org/abs/2502.16286v1) [paper-pdf](http://arxiv.org/pdf/2502.16286v1)

**Authors**: Yedi Zhang, Lei Huang, Pengfei Gao, Fu Song, Jun Sun, Jin Song Dong

**Abstract**: In the rapidly evolving landscape of neural network security, the resilience of neural networks against bit-flip attacks (i.e., an attacker maliciously flips an extremely small amount of bits within its parameter storage memory system to induce harmful behavior), has emerged as a relevant area of research. Existing studies suggest that quantization may serve as a viable defense against such attacks. Recognizing the documented susceptibility of real-valued neural networks to such attacks and the comparative robustness of quantized neural networks (QNNs), in this work, we introduce BFAVerifier, the first verification framework designed to formally verify the absence of bit-flip attacks or to identify all vulnerable parameters in a sound and rigorous manner. BFAVerifier comprises two integral components: an abstraction-based method and an MILP-based method. Specifically, we first conduct a reachability analysis with respect to symbolic parameters that represent the potential bit-flip attacks, based on a novel abstract domain with a sound guarantee. If the reachability analysis fails to prove the resilience of such attacks, then we encode this verification problem into an equivalent MILP problem which can be solved by off-the-shelf solvers. Therefore, BFAVerifier is sound, complete, and reasonably efficient. We conduct extensive experiments, which demonstrate its effectiveness and efficiency across various network architectures, quantization bit-widths, and adversary capabilities.

摘要: 在迅速发展的神经网络安全格局中，神经网络对位翻转攻击(即攻击者恶意翻转其参数存储存储系统中极少量的位以诱导有害行为)的弹性已成为一个相关的研究领域。现有的研究表明，量化可能是一种可行的防御此类攻击的方法。考虑到实值神经网络对此类攻击的易感性和量化神经网络(QNN)的相对稳健性，在本工作中，我们引入了BFAVerizer，这是第一个验证框架，旨在正式验证不存在比特翻转攻击或以合理和严格的方式识别所有易受攻击的参数。BFAVerator由两个完整的组件组成：基于抽象的方法和基于MILP的方法。具体地说，我们首先对代表潜在比特翻转攻击的符号参数进行了可达性分析，该分析基于一个新的具有可靠保证的抽象域。如果可达性分析不能证明这种攻击的弹性，那么我们将该验证问题编码成一个等价的MILP问题，该问题可以通过现成的求解器来解决。因此，BFAVerator是完善的、完整的，并且相当高效。我们进行了广泛的实验，这些实验证明了它在各种网络体系结构、量化位宽和对手能力方面的有效性和效率。



## **34. Your Diffusion Model is Secretly a Certifiably Robust Classifier**

您的扩散模型秘密地是一个可认证的稳健分类器 cs.LG

Accepted by NeurIPS 2024. Also named as "Diffusion Models are  Certifiably Robust Classifiers"

**SubmitDate**: 2025-02-22    [abs](http://arxiv.org/abs/2402.02316v4) [paper-pdf](http://arxiv.org/pdf/2402.02316v4)

**Authors**: Huanran Chen, Yinpeng Dong, Shitong Shao, Zhongkai Hao, Xiao Yang, Hang Su, Jun Zhu

**Abstract**: Generative learning, recognized for its effective modeling of data distributions, offers inherent advantages in handling out-of-distribution instances, especially for enhancing robustness to adversarial attacks. Among these, diffusion classifiers, utilizing powerful diffusion models, have demonstrated superior empirical robustness. However, a comprehensive theoretical understanding of their robustness is still lacking, raising concerns about their vulnerability to stronger future attacks. In this study, we prove that diffusion classifiers possess $O(1)$ Lipschitzness, and establish their certified robustness, demonstrating their inherent resilience. To achieve non-constant Lipschitzness, thereby obtaining much tighter certified robustness, we generalize diffusion classifiers to classify Gaussian-corrupted data. This involves deriving the evidence lower bounds (ELBOs) for these distributions, approximating the likelihood using the ELBO, and calculating classification probabilities via Bayes' theorem. Experimental results show the superior certified robustness of these Noised Diffusion Classifiers (NDCs). Notably, we achieve over 80% and 70% certified robustness on CIFAR-10 under adversarial perturbations with \(\ell_2\) norms less than 0.25 and 0.5, respectively, using a single off-the-shelf diffusion model without any additional data.

摘要: 生成性学习以其对数据分布的有效建模而被公认，在处理分布外实例方面提供了固有的优势，特别是在增强对对手攻击的稳健性方面。其中，扩散分类器利用了强大的扩散模型，表现出了优越的经验稳健性。然而，对它们的健壮性仍然缺乏全面的理论理解，这引发了人们对它们在未来更强大的攻击中的脆弱性的担忧。在这项研究中，我们证明了扩散分类器具有$O(1)$Lipschitz性，并建立了它们被证明的稳健性，证明了它们的内在弹性。为了实现非常数的Lipschitz性，从而获得更紧密的认证稳健性，我们推广了扩散分类器来分类受高斯污染的数据。这涉及到推导这些分布的证据下界(ELBO)，使用ELBO近似似然性，以及通过贝叶斯定理计算分类概率。实验结果表明，这些带噪扩散分类器(NDC)具有良好的鲁棒性。值得注意的是，在没有任何额外数据的情况下，我们使用单个现成的扩散模型，在对抗性扰动下分别获得了超过80%和70%的CIFAR-10的认证鲁棒性。



## **35. Na'vi or Knave: Jailbreaking Language Models via Metaphorical Avatars**

纳威人或恶棍：通过隐喻化身越狱语言模型 cs.CL

Our study requires further in-depth research to ensure the  comprehensiveness and adequacy of the methodology

**SubmitDate**: 2025-02-22    [abs](http://arxiv.org/abs/2412.12145v4) [paper-pdf](http://arxiv.org/pdf/2412.12145v4)

**Authors**: Yu Yan, Sheng Sun, Junqi Tong, Min Liu, Qi Li

**Abstract**: Metaphor serves as an implicit approach to convey information, while enabling the generalized comprehension of complex subjects. However, metaphor can potentially be exploited to bypass the safety alignment mechanisms of Large Language Models (LLMs), leading to the theft of harmful knowledge. In our study, we introduce a novel attack framework that exploits the imaginative capacity of LLMs to achieve jailbreaking, the J\underline{\textbf{A}}ilbreak \underline{\textbf{V}}ia \underline{\textbf{A}}dversarial Me\underline{\textbf{TA}} -pho\underline{\textbf{R}} (\textit{AVATAR}). Specifically, to elicit the harmful response, AVATAR extracts harmful entities from a given harmful target and maps them to innocuous adversarial entities based on LLM's imagination. Then, according to these metaphors, the harmful target is nested within human-like interaction for jailbreaking adaptively. Experimental results demonstrate that AVATAR can effectively and transferablly jailbreak LLMs and achieve a state-of-the-art attack success rate across multiple advanced LLMs. Our study exposes a security risk in LLMs from their endogenous imaginative capabilities. Furthermore, the analytical study reveals the vulnerability of LLM to adversarial metaphors and the necessity of developing defense methods against jailbreaking caused by the adversarial metaphor. \textcolor{orange}{ \textbf{Warning: This paper contains potentially harmful content from LLMs.}}

摘要: 隐喻是一种隐含的传达信息的方法，同时也使人们能够对复杂的主题进行广义的理解。然而，隐喻可能被利用来绕过大型语言模型(LLM)的安全对齐机制，从而导致有害知识的窃取。在我们的研究中，我们提出了一种新的攻击框架，该框架利用LLMS的想象能力来实现越狱，即J下划线{\extbf{A}}ilBreak\下划线{\extbf{A}}ia\下划线{\extbf{A}}dversarial Me\下划线{\extbf{TA}}-pho\下划线{\extbf{R}}(\textit{阿凡达})。具体地说，为了引发有害反应，阿凡达从给定的有害目标中提取有害实体，并基于LLM的想象力将它们映射到无害的对抗性实体。然后，根据这些隐喻，有害的目标嵌套在类似人类的互动中，以适应越狱。实验结果表明，阿凡达可以有效地、可转移地越狱LLM，并在多个高级LLM上获得最先进的攻击成功率。我们的研究揭示了LLMS的安全风险来自其内生的想象力。此外，分析研究揭示了LLM对对抗性隐喻的脆弱性，以及开发针对对抗性隐喻导致的越狱的防御方法的必要性。\extCOLOR{橙色}{\extbf{警告：此纸张包含来自LLMS的潜在有害内容。}}



## **36. A Survey of Model Extraction Attacks and Defenses in Distributed Computing Environments**

分布式计算环境中模型提取攻击和防御综述 cs.CR

**SubmitDate**: 2025-02-22    [abs](http://arxiv.org/abs/2502.16065v1) [paper-pdf](http://arxiv.org/pdf/2502.16065v1)

**Authors**: Kaixiang Zhao, Lincan Li, Kaize Ding, Neil Zhenqiang Gong, Yue Zhao, Yushun Dong

**Abstract**: Model Extraction Attacks (MEAs) threaten modern machine learning systems by enabling adversaries to steal models, exposing intellectual property and training data. With the increasing deployment of machine learning models in distributed computing environments, including cloud, edge, and federated learning settings, each paradigm introduces distinct vulnerabilities and challenges. Without a unified perspective on MEAs across these distributed environments, organizations risk fragmented defenses, inadequate risk assessments, and substantial economic and privacy losses. This survey is motivated by the urgent need to understand how the unique characteristics of cloud, edge, and federated deployments shape attack vectors and defense requirements. We systematically examine the evolution of attack methodologies and defense mechanisms across these environments, demonstrating how environmental factors influence security strategies in critical sectors such as autonomous vehicles, healthcare, and financial services. By synthesizing recent advances in MEAs research and discussing the limitations of current evaluation practices, this survey provides essential insights for developing robust and adaptive defense strategies. Our comprehensive approach highlights the importance of integrating protective measures across the entire distributed computing landscape to ensure the secure deployment of machine learning models.

摘要: 模型提取攻击(MEA)通过使攻击者能够窃取模型、暴露知识产权和训练数据来威胁现代机器学习系统。随着在包括云、边缘和联合学习环境在内的分布式计算环境中越来越多地部署机器学习模型，每个范例都引入了不同的漏洞和挑战。如果对这些分布式环境中的多边环境协定缺乏统一的观点，组织将面临防御支离破碎、风险评估不足以及重大经济和隐私损失的风险。此调查的动机是迫切需要了解云、边缘和联合部署的独特特征如何影响攻击矢量和防御需求。我们系统地研究了这些环境中攻击方法和防御机制的演变，展示了环境因素如何影响自动驾驶汽车、医疗保健和金融服务等关键部门的安全战略。通过综合MEAS研究的最新进展和讨论当前评估实践的局限性，本调查为制定稳健和自适应的防御战略提供了重要的见解。我们的综合方法强调了在整个分布式计算环境中集成保护措施的重要性，以确保机器学习模型的安全部署。



## **37. Human-AI Collaboration in Cloud Security: Cognitive Hierarchy-Driven Deep Reinforcement Learning**

云安全中的人机协作：认知层次驱动的深度强化学习 cs.CR

**SubmitDate**: 2025-02-22    [abs](http://arxiv.org/abs/2502.16054v1) [paper-pdf](http://arxiv.org/pdf/2502.16054v1)

**Authors**: Zahra Aref, Sheng Wei, Narayan B. Mandayam

**Abstract**: Given the complexity of multi-tenant cloud environments and the need for real-time threat mitigation, Security Operations Centers (SOCs) must integrate AI-driven adaptive defenses against Advanced Persistent Threats (APTs). However, SOC analysts struggle with countering adaptive adversarial tactics, necessitating intelligent decision-support frameworks. To enhance human-AI collaboration in SOCs, we propose a Cognitive Hierarchy Theory-driven Deep Q-Network (CHT-DQN) framework that models SOC analysts' decision-making against AI-driven APT bots. The SOC analyst (defender) operates at cognitive level-1, anticipating attacker strategies, while the APT bot (attacker) follows a level-0 exploitative policy. By incorporating CHT into DQN, our framework enhances SOC defense strategies via Attack Graph (AG)-based reinforcement learning. Simulation experiments across varying AG complexities show that CHT-DQN achieves higher data protection and lower action discrepancies compared to standard DQN. A theoretical lower bound analysis further validates its superior Q-value performance. A human-in-the-loop (HITL) evaluation on Amazon Mechanical Turk (MTurk) reveals that SOC analysts using CHT-DQN-driven transition probabilities align better with adaptive attackers, improving data protection. Additionally, human decision patterns exhibit risk aversion after failure and risk-seeking behavior after success, aligning with Prospect Theory. These findings underscore the potential of integrating cognitive modeling into deep reinforcement learning to enhance SOC operations and develop real-time adaptive cloud security mechanisms.

摘要: 鉴于多租户云环境的复杂性和对实时威胁缓解的需求，安全运营中心(SOC)必须集成人工智能驱动的自适应防御，以抵御高级持久威胁(APT)。然而，SOC分析师在对抗适应性对抗战术方面举步维艰，这就需要智能的决策支持框架。为了加强SOC中人与AI的协作，我们提出了一个认知层次理论驱动的深度Q网络(CHT-DQN)框架，该框架针对AI驱动的APT机器人对SOC分析师的决策进行建模。SOC分析员(防御者)在认知级别1操作，预测攻击者的策略，而APT机器人(攻击者)遵循0级剥削策略。通过将CHT引入DQN，我们的框架通过基于攻击图(AG)的强化学习来增强SOC防御策略。不同AG复杂度的仿真实验表明，与标准DQN相比，CHT-DQN实现了更高的数据保护和更低的动作差异。理论下界分析进一步验证了其优越的Q值性能。对Amazon Machine Turk(MTurk)进行的人在环中(HITL)评估显示，使用CHT-DQN驱动的转换概率的SOC分析师可以更好地与自适应攻击者保持一致，从而改善数据保护。此外，人类的决策模式表现出失败后的风险厌恶和成功后的冒险行为，这与前景理论是一致的。这些发现强调了将认知建模集成到深度强化学习中以增强SOC操作和开发实时自适应云安全机制的潜力。



## **38. A Multi-Scale Isolation Forest Approach for Real-Time Detection and Filtering of FGSM Adversarial Attacks in Video Streams of Autonomous Vehicles**

用于实时检测和过滤自动驾驶汽车视频流中FGSM对抗攻击的多尺度隔离森林方法 cs.CV

17 pages, 7 figures

**SubmitDate**: 2025-02-22    [abs](http://arxiv.org/abs/2502.16044v1) [paper-pdf](http://arxiv.org/pdf/2502.16044v1)

**Authors**: Richard Abhulimhen, Negash Begashaw, Gurcan Comert, Chunheng Zhao, Pierluigi Pisu

**Abstract**: Deep Neural Networks (DNNs) have demonstrated remarkable success across a wide range of tasks, particularly in fields such as image classification. However, DNNs are highly susceptible to adversarial attacks, where subtle perturbations are introduced to input images, leading to erroneous model outputs. In today's digital era, ensuring the security and integrity of images processed by DNNs is of critical importance. One of the most prominent adversarial attack methods is the Fast Gradient Sign Method (FGSM), which perturbs images in the direction of the loss gradient to deceive the model.   This paper presents a novel approach for detecting and filtering FGSM adversarial attacks in image processing tasks. Our proposed method evaluates 10,000 images, each subjected to five different levels of perturbation, characterized by $\epsilon$ values of 0.01, 0.02, 0.05, 0.1, and 0.2. These perturbations are applied in the direction of the loss gradient. We demonstrate that our approach effectively filters adversarially perturbed images, mitigating the impact of FGSM attacks.   The method is implemented in Python, and the source code is publicly available on GitHub for reproducibility and further research.

摘要: 深度神经网络(DNN)在广泛的任务中取得了显著的成功，特别是在图像分类等领域。然而，DNN很容易受到敌意攻击，在输入图像中引入微妙的扰动，导致错误的模型输出。在当今的数字时代，确保DNN处理的图像的安全性和完整性至关重要。其中最突出的对抗性攻击方法之一是快速梯度符号方法(FGSM)，它在损失梯度方向上扰动图像来欺骗模型。提出了一种在图像处理任务中检测和过滤FGSM攻击的新方法。我们提出的方法评估了10,000个图像，每个图像都受到五种不同级别的扰动，其特征是$\epsilon$值分别为0.01、0.02、0.05、0.1和0.2。这些扰动作用在损耗梯度的方向上。我们证明了我们的方法有效地过滤了恶意扰动的图像，减轻了FGSM攻击的影响。该方法是用Python语言实现的，源代码在GitHub上公开可供重现和进一步研究。



## **39. Overcoming Intensity Limits for Long-Distance Quantum Key Distribution**

克服长距离量子密钥分发的强度限制 quant-ph

**SubmitDate**: 2025-02-22    [abs](http://arxiv.org/abs/2412.20265v3) [paper-pdf](http://arxiv.org/pdf/2412.20265v3)

**Authors**: Ibrahim Almosallam

**Abstract**: Quantum Key Distribution (QKD) enables the sharing of cryptographic keys secured by quantum mechanics. The BB84 protocol assumed single-photon sources, but practical systems rely on weak coherent pulses vulnerable to photon-number-splitting (PNS) attacks. The Gottesman-Lo-L\"utkenhaus-Preskill (GLLP) framework addressed these imperfections, deriving secure key rate bounds under limited PNS scenarios. The Decoy-state protocol further improved performance by refining single-photon yield estimates, but still considered multi-photon states as insecure, thereby limiting intensities and constraining key rate and distance. More recently, finite-key security bounds for decoy-state QKD have been extended to address general attacks, ensuring security against adversaries capable of exploiting arbitrary strategies. In this work, we focus on a specific class of attacks, the generalized PNS attack, and demonstrate that higher pulse intensities can be securely used by employing Bayesian inference to estimate key parameters directly from observed data. By raising the pulse intensity to 10 photons, we achieve a 50-fold increase in key rate and a 62.2% increase in operational range (about 200 km) compared to the decoy-state protocol. Furthermore, we accurately model after-pulsing using a Hidden Markov Model and reveal inaccuracies in decoy-state calculations that may produce erroneous key-rate estimates. While this methodology does not address all possible attacks, it provides a new approach to security proofs in QKD by shifting from worst-case assumption analysis to observation-dependent inference, advancing the reach and efficiency of discrete-variable QKD protocols.

摘要: 量子密钥分发(QKD)使得共享由量子力学保护的密钥成为可能。BB84协议假设单光子源，但实际系统依赖于易受光子数分裂(PNS)攻击的弱相干脉冲。Gottesman-Lo-L(GLLP)框架解决了这些缺陷，在有限的PNS场景下推导出安全的密钥速率界限。诱骗状态协议通过改进单光子产量估计进一步提高了性能，但仍然认为多光子状态是不安全的，从而限制了强度并限制了密钥速率和距离。最近，诱饵状态QKD的有限密钥安全界限已扩展到应对一般攻击，确保了针对能够利用任意策略的对手的安全性。在这项工作中，我们专注于一类特定的攻击，广义PNS攻击，并证明了通过使用贝叶斯推理直接从观测数据估计关键参数，可以安全地使用较高的脉冲强度。通过将脉冲强度提高到10个光子，与诱饵态协议相比，我们获得了50倍的密钥速率和62.2%的作用距离(约200公里)。此外，我们使用隐马尔可夫模型准确地对后脉冲进行建模，并揭示诱饵状态计算中的不准确，这可能会产生错误的关键速率估计。虽然这种方法不能解决所有可能的攻击，但它通过从最坏情况假设分析转向基于观测的推理，为量子密钥分发协议的安全性证明提供了一种新的方法，提高了离散变量量子密钥分发协议的可达性和效率。



## **40. Cross-Model Transferability of Adversarial Patches in Real-time Segmentation for Autonomous Driving**

自动驾驶实时分割中对抗补丁的跨模型可移植性 cs.CV

**SubmitDate**: 2025-02-22    [abs](http://arxiv.org/abs/2502.16012v1) [paper-pdf](http://arxiv.org/pdf/2502.16012v1)

**Authors**: Prashant Shekhar, Bidur Devkota, Dumindu Samaraweera, Laxima Niure Kandel, Manoj Babu

**Abstract**: Adversarial attacks pose a significant threat to deep learning models, particularly in safety-critical applications like healthcare and autonomous driving. Recently, patch based attacks have demonstrated effectiveness in real-time inference scenarios owing to their 'drag and drop' nature. Following this idea for Semantic Segmentation (SS), here we propose a novel Expectation Over Transformation (EOT) based adversarial patch attack that is more realistic for autonomous vehicles. To effectively train this attack we also propose a 'simplified' loss function that is easy to analyze and implement. Using this attack as our basis, we investigate whether adversarial patches once optimized on a specific SS model, can fool other models or architectures. We conduct a comprehensive cross-model transferability analysis of adversarial patches trained on SOTA Convolutional Neural Network (CNN) models such PIDNet-S, PIDNet-M and PIDNet-L, among others. Additionally, we also include the Segformer model to study transferability to Vision Transformers (ViTs). All of our analysis is conducted on the widely used Cityscapes dataset. Our study reveals key insights into how model architectures (CNN vs CNN or CNN vs. Transformer-based) influence attack susceptibility. In particular, we conclude that although the transferability (effectiveness) of attacks on unseen images of any dimension is really high, the attacks trained against one particular model are minimally effective on other models. And this was found to be true for both ViT and CNN based models. Additionally our results also indicate that for CNN-based models, the repercussions of patch attacks are local, unlike ViTs. Per-class analysis reveals that simple-classes like 'sky' suffer less misclassification than others. The code for the project is available at: https://github.com/p-shekhar/adversarial-patch-transferability

摘要: 对抗性攻击对深度学习模型构成了重大威胁，特别是在医疗保健和自动驾驶等安全关键型应用中。最近，基于补丁的攻击已经证明了它们在实时推理场景中的有效性，因为它们具有拖放的性质。借鉴语义分割(SS)的思想，本文提出了一种新的基于期望过度变换(EOT)的对抗性补丁攻击，该攻击更适合于自主车辆。为了有效地训练这种攻击，我们还提出了一种易于分析和实现的简化的损失函数。以这一攻击为基础，我们调查了在特定SS模型上优化的敌意补丁是否可以愚弄其他模型或架构。我们对基于SOTA卷积神经网络模型训练的对抗性补丁进行了全面的跨模型可转移性分析，如PIDNet-S、PIDNet-M和PIDNet-L等。此外，我们还包括Segformer模型来研究向视觉转换器(VITS)的可转移性。所有的分析都是在广泛使用的城市景观数据集上进行的。我们的研究揭示了模型架构(CNN与CNN或CNN与基于Transformer)如何影响攻击敏感度的关键见解。特别是，我们得出的结论是，尽管对任何维度的不可见图像的攻击的可转移性(有效性)确实很高，但针对一个特定模型训练的攻击对其他模型的有效性最低。研究发现，VIT和基于CNN的模型都是如此。此外，我们的结果还表明，对于基于CNN的模型，补丁攻击的影响是局部的，不同于VITS。按类别进行的分析显示，像“sky”这样的简单类别比其他类别的错误分类更少。该项目的代码可在以下网址获得：https://github.com/p-shekhar/adversarial-patch-transferability



## **41. A Grey-box Attack against Latent Diffusion Model-based Image Editing by Posterior Collapse**

对基于潜在扩散模型的图像编辑的灰箱攻击 cs.CV

21 pages, 7 figures, 10 tables

**SubmitDate**: 2025-02-21    [abs](http://arxiv.org/abs/2408.10901v3) [paper-pdf](http://arxiv.org/pdf/2408.10901v3)

**Authors**: Zhongliang Guo, Chun Tong Lei, Lei Fang, Shuai Zhao, Yifei Qian, Jingyu Lin, Zeyu Wang, Cunjian Chen, Ognjen Arandjelović, Chun Pong Lau

**Abstract**: Recent advancements in generative AI, particularly Latent Diffusion Models (LDMs), have revolutionized image synthesis and manipulation. However, these generative techniques raises concerns about data misappropriation and intellectual property infringement. Adversarial attacks on machine learning models have been extensively studied, and a well-established body of research has extended these techniques as a benign metric to prevent the underlying misuse of generative AI. Current approaches to safeguarding images from manipulation by LDMs are limited by their reliance on model-specific knowledge and their inability to significantly degrade semantic quality of generated images. In response to these shortcomings, we propose the Posterior Collapse Attack (PCA) based on the observation that VAEs suffer from posterior collapse during training. Our method minimizes dependence on the white-box information of target models to get rid of the implicit reliance on model-specific knowledge. By accessing merely a small amount of LDM parameters, in specific merely the VAE encoder of LDMs, our method causes a substantial semantic collapse in generation quality, particularly in perceptual consistency, and demonstrates strong transferability across various model architectures. Experimental results show that PCA achieves superior perturbation effects on image generation of LDMs with lower runtime and VRAM. Our method outperforms existing techniques, offering a more robust and generalizable solution that is helpful in alleviating the socio-technical challenges posed by the rapidly evolving landscape of generative AI.

摘要: 生成性人工智能的最新进展，特别是潜在扩散模型(LDM)，已经彻底改变了图像合成和处理。然而，这些生成性技术引发了人们对数据挪用和侵犯知识产权的担忧。对机器学习模型的对抗性攻击已经被广泛研究，一系列成熟的研究已经将这些技术扩展为一种良性的衡量标准，以防止潜在的生成性人工智能的滥用。当前保护图像免受LDM操纵的方法受到它们对模型特定知识的依赖以及它们无法显著降低所生成图像的语义质量的限制。针对这些不足，我们提出了后部塌陷攻击(PCA)，基于VAE在训练过程中遭受后部塌陷的观察。我们的方法最大限度地减少了对目标模型白盒信息的依赖，摆脱了对特定模型知识的隐含依赖。通过只访问少量的LDM参数，特别是LDM的VAE编码器，我们的方法导致生成质量的语义崩溃，特别是在感知一致性方面，并表现出强大的跨模型体系结构的可移植性。实验结果表明，主成分分析算法以较低的运行时间和较低的VRAM实现了较好的图像生成扰动效果。我们的方法优于现有的技术，提供了一个更健壮和更具通用性的解决方案，有助于缓解快速发展的生成性人工智能所带来的社会技术挑战。



## **42. Defending Jailbreak Prompts via In-Context Adversarial Game**

通过上下文对抗游戏为越狱辩护 cs.LG

EMNLP 2024 Main Paper

**SubmitDate**: 2025-02-21    [abs](http://arxiv.org/abs/2402.13148v3) [paper-pdf](http://arxiv.org/pdf/2402.13148v3)

**Authors**: Yujun Zhou, Yufei Han, Haomin Zhuang, Kehan Guo, Zhenwen Liang, Hongyan Bao, Xiangliang Zhang

**Abstract**: Large Language Models (LLMs) demonstrate remarkable capabilities across diverse applications. However, concerns regarding their security, particularly the vulnerability to jailbreak attacks, persist. Drawing inspiration from adversarial training in deep learning and LLM agent learning processes, we introduce the In-Context Adversarial Game (ICAG) for defending against jailbreaks without the need for fine-tuning. ICAG leverages agent learning to conduct an adversarial game, aiming to dynamically extend knowledge to defend against jailbreaks. Unlike traditional methods that rely on static datasets, ICAG employs an iterative process to enhance both the defense and attack agents. This continuous improvement process strengthens defenses against newly generated jailbreak prompts. Our empirical studies affirm ICAG's efficacy, where LLMs safeguarded by ICAG exhibit significantly reduced jailbreak success rates across various attack scenarios. Moreover, ICAG demonstrates remarkable transferability to other LLMs, indicating its potential as a versatile defense mechanism.

摘要: 大型语言模型(LLM)在不同的应用程序中展示了卓越的功能。然而，对他们的安全，特别是对越狱攻击的脆弱性的担忧依然存在。从深度学习和LLM代理学习过程中的对抗性训练中获得灵感，我们引入了无需微调的上下文对抗性游戏(ICAG)来防御越狱。ICAG利用代理学习进行对抗性游戏，旨在动态扩展知识来防御越狱。与依赖静态数据集的传统方法不同，ICAG采用迭代过程来增强防御和攻击代理。这一不断改进的过程加强了对新生成的越狱提示的防御。我们的经验研究肯定了ICAG的有效性，在不同的攻击场景中，由ICAG保护的LLM显示出显著降低的越狱成功率。此外，ICAG表现出显著的可转移性，表明其作为一种多功能防御机制的潜力。



## **43. A First Principles Approach to Trust-Based Recommendation Systems in Social Networks**

社交网络中基于信任的推荐系统的第一原则方法 cs.IR

**SubmitDate**: 2025-02-21    [abs](http://arxiv.org/abs/2407.00062v2) [paper-pdf](http://arxiv.org/pdf/2407.00062v2)

**Authors**: Paras Stefanopoulos, Sourin Chatterjee, Ahad N. Zehmakan

**Abstract**: This paper explores recommender systems in social networks which leverage information such as item rating, intra-item similarities, and trust graph. We demonstrate that item-rating information is more influential than other information types in a collaborative filtering approach. The trust graph-based approaches were found to be more robust to network adversarial attacks due to hard-to-manipulate trust structures. Intra-item information, although sub-optimal in isolation, enhances the consistency of predictions and lower-end performance when fused with other information forms. Additionally, the Weighted Average framework is introduced, enabling the construction of recommendation systems around any user-to-user similarity metric. All the codes are publicly available on GitHub.

摘要: 本文探讨了社交网络中的推荐系统，该系统利用项目评级、项目内相似性和信任图等信息。我们证明，在协作过滤方法中，项目评级信息比其他信息类型更有影响力。由于信任结构难以操纵，基于信任图的方法被发现对网络对抗攻击更稳健。项内信息虽然单独而言次优，但与其他信息形式融合时可以增强预测和低端性能的一致性。此外，还引入了加权平均框架，可以围绕任何用户与用户的相似性指标构建推荐系统。所有代码均在GitHub上公开。



## **44. A Comprehensive Survey on the Trustworthiness of Large Language Models in Healthcare**

医疗保健中大型语言模型可信度的全面调查 cs.CY

**SubmitDate**: 2025-02-21    [abs](http://arxiv.org/abs/2502.15871v1) [paper-pdf](http://arxiv.org/pdf/2502.15871v1)

**Authors**: Manar Aljohani, Jun Hou, Sindhura Kommu, Xuan Wang

**Abstract**: The application of large language models (LLMs) in healthcare has the potential to revolutionize clinical decision-making, medical research, and patient care. As LLMs are increasingly integrated into healthcare systems, several critical challenges must be addressed to ensure their reliable and ethical deployment. These challenges include truthfulness, where models generate misleading information; privacy, with risks of unintentional data retention; robustness, requiring defenses against adversarial attacks; fairness, addressing biases in clinical outcomes; explainability, ensuring transparent decision-making; and safety, mitigating risks of misinformation and medical errors. Recently, researchers have begun developing benchmarks and evaluation frameworks to systematically assess the trustworthiness of LLMs. However, the trustworthiness of LLMs in healthcare remains underexplored, lacking a systematic review that provides a comprehensive understanding and future insights into this area. This survey bridges this gap by providing a comprehensive overview of the recent research of existing methodologies and solutions aimed at mitigating the above risks in healthcare. By focusing on key trustworthiness dimensions including truthfulness, privacy and safety, robustness, fairness and bias, and explainability, we present a thorough analysis of how these issues impact the reliability and ethical use of LLMs in healthcare. This paper highlights ongoing efforts and offers insights into future research directions to ensure the safe and trustworthy deployment of LLMs in healthcare.

摘要: 大型语言模型(LLM)在医疗保健中的应用有可能给临床决策、医学研究和患者护理带来革命性的变化。随着LLM越来越多地集成到医疗系统中，必须解决几个关键挑战，以确保它们的可靠和合乎道德的部署。这些挑战包括：真实性，模型产生误导性信息；隐私，存在无意数据保留的风险；稳健性，需要防范敌意攻击；公平性，解决临床结果中的偏差；可解释性，确保决策透明；以及安全性，降低错误信息和医疗差错的风险。最近，研究人员已经开始开发基准和评估框架，以系统地评估低成本管理的可信度。然而，低成本管理在医疗保健领域的可信度仍然没有得到充分的探索，缺乏一个系统的回顾来提供对这一领域的全面理解和未来的洞察。这项调查通过全面概述旨在缓解医疗保健领域上述风险的现有方法和解决方案的最新研究，弥合了这一差距。通过关注关键的可信性维度，包括真实性、隐私和安全性、健壮性、公平性和偏倚以及可解释性，我们对这些问题如何影响低成本管理在医疗保健中的可靠性和合乎道德的使用进行了彻底的分析。这篇白皮书强调了正在进行的努力，并对未来的研究方向提出了见解，以确保低成本管理系统在医疗保健领域的安全和值得信赖的部署。



## **45. Model Privacy: A Unified Framework to Understand Model Stealing Attacks and Defenses**

模型隐私：理解模型窃取攻击和防御的统一框架 cs.LG

**SubmitDate**: 2025-02-21    [abs](http://arxiv.org/abs/2502.15567v1) [paper-pdf](http://arxiv.org/pdf/2502.15567v1)

**Authors**: Ganghua Wang, Yuhong Yang, Jie Ding

**Abstract**: The use of machine learning (ML) has become increasingly prevalent in various domains, highlighting the importance of understanding and ensuring its safety. One pressing concern is the vulnerability of ML applications to model stealing attacks. These attacks involve adversaries attempting to recover a learned model through limited query-response interactions, such as those found in cloud-based services or on-chip artificial intelligence interfaces. While existing literature proposes various attack and defense strategies, these often lack a theoretical foundation and standardized evaluation criteria. In response, this work presents a framework called ``Model Privacy'', providing a foundation for comprehensively analyzing model stealing attacks and defenses. We establish a rigorous formulation for the threat model and objectives, propose methods to quantify the goodness of attack and defense strategies, and analyze the fundamental tradeoffs between utility and privacy in ML models. Our developed theory offers valuable insights into enhancing the security of ML models, especially highlighting the importance of the attack-specific structure of perturbations for effective defenses. We demonstrate the application of model privacy from the defender's perspective through various learning scenarios. Extensive experiments corroborate the insights and the effectiveness of defense mechanisms developed under the proposed framework.

摘要: 机器学习(ML)的使用在各个领域中越来越普遍，这突显了理解和确保其安全性的重要性。一个迫在眉睫的问题是ML应用程序在模拟窃取攻击时的脆弱性。这些攻击涉及攻击者试图通过有限的查询-响应交互来恢复学习的模型，例如基于云的服务或芯片上人工智能界面中的交互。现有文献虽然提出了各种各样的攻防策略，但往往缺乏理论基础和标准化的评价标准。作为回应，这项工作提出了一个称为模型隐私的框架，为全面分析模型窃取攻击和防御提供了基础。我们对威胁模型和目标建立了严格的描述，提出了量化攻防策略优劣的方法，并分析了ML模型中效用和隐私之间的基本权衡。我们发展的理论为增强ML模型的安全性提供了有价值的见解，特别是强调了特定于攻击的扰动结构对于有效防御的重要性。我们通过不同的学习场景从防御者的角度演示了模型隐私的应用。广泛的实验证实了在提议的框架下开发的防御机制的洞察力和有效性。



## **46. A Defensive Framework Against Adversarial Attacks on Machine Learning-Based Network Intrusion Detection Systems**

基于机器学习的网络入侵检测系统对抗攻击的防御框架 cs.CR

Accepted to IEEE AI+ TrustCom 2024

**SubmitDate**: 2025-02-21    [abs](http://arxiv.org/abs/2502.15561v1) [paper-pdf](http://arxiv.org/pdf/2502.15561v1)

**Authors**: Benyamin Tafreshian, Shengzhi Zhang

**Abstract**: As cyberattacks become increasingly sophisticated, advanced Network Intrusion Detection Systems (NIDS) are critical for modern network security. Traditional signature-based NIDS are inadequate against zero-day and evolving attacks. In response, machine learning (ML)-based NIDS have emerged as promising solutions; however, they are vulnerable to adversarial evasion attacks that subtly manipulate network traffic to bypass detection. To address this vulnerability, we propose a novel defensive framework that enhances the robustness of ML-based NIDS by simultaneously integrating adversarial training, dataset balancing techniques, advanced feature engineering, ensemble learning, and extensive model fine-tuning. We validate our framework using the NSL-KDD and UNSW-NB15 datasets. Experimental results show, on average, a 35% increase in detection accuracy and a 12.5% reduction in false positives compared to baseline models, particularly under adversarial conditions. The proposed defense against adversarial attacks significantly advances the practical deployment of robust ML-based NIDS in real-world networks.

摘要: 随着网络攻击的日益复杂，先进的网络入侵检测系统对现代网络安全至关重要。传统的基于签名的网络入侵检测系统不能抵抗零日攻击和不断演变的攻击。作为回应，基于机器学习(ML)的网络入侵检测系统已经成为有希望的解决方案；然而，它们容易受到对手逃避攻击，这些攻击巧妙地操纵网络流量以绕过检测。为了解决这个漏洞，我们提出了一个新的防御框架，通过同时集成对抗性训练、数据集平衡技术、高级特征工程、集成学习和广泛的模型微调来增强基于ML的网络入侵检测系统的健壮性。我们使用NSL-KDD和UNSW-NB15数据集验证了我们的框架。实验结果表明，与基线模型相比，检测准确率平均提高了35%，误报减少了12.5%，尤其是在对抗性条件下。所提出的防御恶意攻击的方法极大地促进了基于ML的健壮网络入侵检测系统在实际网络中的实际部署。



## **47. Single-pass Detection of Jailbreaking Input in Large Language Models**

大型语言模型中越狱输入的单程检测 cs.LG

Accepted in TMLR 2025

**SubmitDate**: 2025-02-21    [abs](http://arxiv.org/abs/2502.15435v1) [paper-pdf](http://arxiv.org/pdf/2502.15435v1)

**Authors**: Leyla Naz Candogan, Yongtao Wu, Elias Abad Rocamora, Grigorios G. Chrysos, Volkan Cevher

**Abstract**: Defending aligned Large Language Models (LLMs) against jailbreaking attacks is a challenging problem, with existing approaches requiring multiple requests or even queries to auxiliary LLMs, making them computationally heavy. Instead, we focus on detecting jailbreaking input in a single forward pass. Our method, called Single Pass Detection SPD, leverages the information carried by the logits to predict whether the output sentence will be harmful. This allows us to defend in just one forward pass. SPD can not only detect attacks effectively on open-source models, but also minimizes the misclassification of harmless inputs. Furthermore, we show that SPD remains effective even without complete logit access in GPT-3.5 and GPT-4. We believe that our proposed method offers a promising approach to efficiently safeguard LLMs against adversarial attacks.

摘要: 保护对齐的大型语言模型（LLM）免受越狱攻击是一个具有挑战性的问题，现有的方法需要多次请求甚至查询来辅助LLM，使得它们的计算量很大。相反，我们专注于检测单次向前传递中的越狱输入。我们的方法称为单程检测SPD，它利用logit携带的信息来预测输出句子是否有害。这使得我们只需一次向前传球即可防守。SPD不仅可以有效检测对开源模型的攻击，还可以最大限度地减少无害输入的错误分类。此外，我们表明，即使在GPT-3.5和GPT-4中没有完全的logit访问，SPD仍然有效。我们相信，我们提出的方法提供了一种有希望的方法来有效保护LLM免受对抗攻击。



## **48. Adversarial Prompt Evaluation: Systematic Benchmarking of Guardrails Against Prompt Input Attacks on LLMs**

对抗性即时评估：针对LLM即时输入攻击的护栏系统基准 cs.CR

NeurIPS 2024, Safe Generative AI Workshop

**SubmitDate**: 2025-02-21    [abs](http://arxiv.org/abs/2502.15427v1) [paper-pdf](http://arxiv.org/pdf/2502.15427v1)

**Authors**: Giulio Zizzo, Giandomenico Cornacchia, Kieran Fraser, Muhammad Zaid Hameed, Ambrish Rawat, Beat Buesser, Mark Purcell, Pin-Yu Chen, Prasanna Sattigeri, Kush Varshney

**Abstract**: As large language models (LLMs) become integrated into everyday applications, ensuring their robustness and security is increasingly critical. In particular, LLMs can be manipulated into unsafe behaviour by prompts known as jailbreaks. The variety of jailbreak styles is growing, necessitating the use of external defences known as guardrails. While many jailbreak defences have been proposed, not all defences are able to handle new out-of-distribution attacks due to the narrow segment of jailbreaks used to align them. Moreover, the lack of systematisation around defences has created significant gaps in their practical application. In this work, we perform systematic benchmarking across 15 different defences, considering a broad swathe of malicious and benign datasets. We find that there is significant performance variation depending on the style of jailbreak a defence is subject to. Additionally, we show that based on current datasets available for evaluation, simple baselines can display competitive out-of-distribution performance compared to many state-of-the-art defences. Code is available at https://github.com/IBM/Adversarial-Prompt-Evaluation.

摘要: 随着大型语言模型(LLM)集成到日常应用程序中，确保它们的健壮性和安全性变得越来越重要。特别是，通过被称为越狱的提示，LLM可以被操纵成不安全的行为。越狱方式的多样性正在增长，需要使用被称为护栏的外部防御。虽然已经提出了许多越狱防御措施，但由于用于对齐越狱的狭窄部分，并不是所有的防御措施都能够应对新的分布外攻击。此外，辩护缺乏系统性，在实际应用中造成了很大的差距。在这项工作中，我们考虑了大量的恶意和良性数据集，对15种不同的防御措施进行了系统的基准测试。我们发现，根据辩护所受的越狱风格的不同，表现会有很大的差异。此外，我们还表明，基于当前可供评估的数据集，与许多最先进的防御措施相比，简单基线可以显示出具有竞争力的分布外性能。代码可在https://github.com/IBM/Adversarial-Prompt-Evaluation.上找到



## **49. R-MTLLMF: Resilient Multi-Task Large Language Model Fusion at the Wireless Edge**

R-MTLLMF：无线边缘的弹性多任务大型语言模型融合 eess.SP

**SubmitDate**: 2025-02-21    [abs](http://arxiv.org/abs/2411.18220v3) [paper-pdf](http://arxiv.org/pdf/2411.18220v3)

**Authors**: Aladin Djuhera, Vlad C. Andrei, Mohsen Pourghasemian, Haris Gacanin, Holger Boche, Walid Saad

**Abstract**: Multi-task large language models (MTLLMs) are important for many applications at the wireless edge, where users demand specialized models to handle multiple tasks efficiently. However, training MTLLMs is complex and exhaustive, particularly when tasks are subject to change. Recently, the concept of model fusion via task vectors has emerged as an efficient approach for combining fine-tuning parameters to produce an MTLLM. In this paper, the problem of enabling edge users to collaboratively craft such MTLMs via tasks vectors is studied, under the assumption of worst-case adversarial attacks. To this end, first the influence of adversarial noise to multi-task model fusion is investigated and a relationship between the so-called weight disentanglement error and the mean squared error (MSE) is derived. Using hypothesis testing, it is directly shown that the MSE increases interference between task vectors, thereby rendering model fusion ineffective. Then, a novel resilient MTLLM fusion (R-MTLLMF) is proposed, which leverages insights about the LLM architecture and fine-tuning process to safeguard task vector aggregation under adversarial noise by realigning the MTLLM. The proposed R-MTLLMF is then compared for both worst-case and ideal transmission scenarios to study the impact of the wireless channel. Extensive model fusion experiments with vision LLMs demonstrate R-MTLLMF's effectiveness, achieving close-to-baseline performance across eight different tasks in ideal noise scenarios and significantly outperforming unprotected model fusion in worst-case scenarios. The results further advocate for additional physical layer protection for a holistic approach to resilience, from both a wireless and LLM perspective.

摘要: 多任务大型语言模型(MTLLM)对于无线边缘的许多应用非常重要，因为用户需要专门的模型来高效地处理多个任务。然而，培训MTLLM是复杂和详尽的，特别是在任务可能发生变化的情况下。最近，基于任务向量的模型融合的概念已经成为一种结合微调参数以产生MTLLM的有效方法。本文在假设最坏情况下的敌意攻击的前提下，研究了边缘用户通过任务向量协作创建MTLM的问题。为此，首先研究了对抗性噪声对多任务模型融合的影响，推导了加权解缠误差与均方误差之间的关系。通过假设检验，直接表明MSE增加了任务向量之间的干扰，从而使模型融合无效。然后，提出了一种新的弹性MTLLM融合算法(R-MTLLMF)，该算法利用对LLM体系结构和微调过程的深入了解，通过重新排列MTLLM来保护对抗噪声下的任务向量聚合。然后将所提出的R-MTLLMF在最坏情况和理想传输场景下进行比较，以研究无线信道的影响。用VISION LLMS进行的大量模型融合实验证明了R-MTLLMF的有效性，在理想噪声场景中，R-MTLLMF在八个不同任务上的性能接近基线，而在最坏情况下，R-MTLLMF的性能明显优于无保护的模型融合。从无线和LLM的角度来看，研究结果进一步倡导为整体恢复方法提供额外的物理层保护。



## **50. Gröbner Basis Cryptanalysis of Ciminion and Hydra**

格罗布纳基础对西米尼恩和海德拉的密码分析 cs.CR

**SubmitDate**: 2025-02-21    [abs](http://arxiv.org/abs/2405.05040v4) [paper-pdf](http://arxiv.org/pdf/2405.05040v4)

**Authors**: Matthias Johann Steiner

**Abstract**: Ciminion and Hydra are two recently introduced symmetric key Pseudo-Random Functions for Multi-Party Computation applications. For efficiency, both primitives utilize quadratic permutations at round level. Therefore, polynomial system solving-based attacks pose a serious threat to these primitives. For Ciminion, we construct a quadratic degree reverse lexicographic (DRL) Gr\"obner basis for the iterated polynomial model via linear transformations. With the Gr\"obner basis we can simplify cryptanalysis, as we no longer need to impose genericity assumptions to derive complexity estimates. For Hydra, with the help of a computer algebra program like SageMath we construct a DRL Gr\"obner basis for the iterated model via linear transformations and a linear change of coordinates. In the Hydra proposal it was claimed that $r_\mathcal{H} = 31$ rounds are sufficient to provide $128$ bits of security against Gr\"obner basis attacks for an ideal adversary with $\omega = 2$. However, via our Hydra Gr\"obner basis standard term order conversion to a lexicographic (LEX) Gr\"obner basis requires just $126$ bits with $\omega = 2$. Moreover, using a dedicated polynomial system solving technique up to $r_\mathcal{H} = 33$ rounds can be attacked below $128$ bits for an ideal adversary.

摘要: Ciminion和Hydra是最近推出的两个用于多方计算应用的对称密钥伪随机函数。为了提高效率，这两个基元都在循环水平上使用二次置换。因此，基于多项式系统求解的攻击对这些原语构成了严重威胁。对于Ciminion，我们通过线性变换构造了迭代多项式模型的二次逆词典(DRL)Gr‘obner基，利用这个Gr’obner基，我们可以简化密码分析，因为我们不再需要强加一般性假设来推导复杂性估计。对于Hydra，借助于SageMath这样的计算机代数程序，我们通过线性变换和线性坐标变化，为迭代模型构造了一个DRL Grobner基.在Hydra的方案中，声称$r_\mathcal{H}=31$轮足以为$omega=2$的理想对手提供$128$bit的Gr‘obner基攻击安全.然而，通过我们的Hydra Gr\“obner基础”将标准术语顺序转换为词典(Lex)Gr\“obner基础只需要$126$位，且$\omega=2$。此外，使用专门的多项式系统求解技术，高达$r_\mathcal{H}=33$轮可以被攻击到低于$128$比特的理想对手。



