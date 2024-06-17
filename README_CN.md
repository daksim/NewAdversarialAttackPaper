# Latest Adversarial Attack Papers
**update at 2024-06-17 09:35:50**

翻译来自 https://cloud.tencent.com/document/product/551/15619

## **1. Automated Design of Linear Bounding Functions for Sigmoidal Nonlinearities in Neural Networks**

神经网络中Sigmoidal非线性线性边界函数的自动设计 cs.LG

**SubmitDate**: 2024-06-14    [abs](http://arxiv.org/abs/2406.10154v1) [paper-pdf](http://arxiv.org/pdf/2406.10154v1)

**Authors**: Matthias König, Xiyue Zhang, Holger H. Hoos, Marta Kwiatkowska, Jan N. van Rijn

**Abstract**: The ubiquity of deep learning algorithms in various applications has amplified the need for assuring their robustness against small input perturbations such as those occurring in adversarial attacks. Existing complete verification techniques offer provable guarantees for all robustness queries but struggle to scale beyond small neural networks. To overcome this computational intractability, incomplete verification methods often rely on convex relaxation to over-approximate the nonlinearities in neural networks. Progress in tighter approximations has been achieved for piecewise linear functions. However, robustness verification of neural networks for general activation functions (e.g., Sigmoid, Tanh) remains under-explored and poses new challenges. Typically, these networks are verified using convex relaxation techniques, which involve computing linear upper and lower bounds of the nonlinear activation functions. In this work, we propose a novel parameter search method to improve the quality of these linear approximations. Specifically, we show that using a simple search method, carefully adapted to the given verification problem through state-of-the-art algorithm configuration techniques, improves the average global lower bound by 25% on average over the current state of the art on several commonly used local robustness verification benchmarks.

摘要: 深度学习算法在各种应用中的普遍存在，增加了确保其对诸如在对抗性攻击中发生的小输入扰动的稳健性的必要性。现有的完整验证技术为所有健壮性查询提供了可证明的保证，但难以扩展到小型神经网络之外。为了克服这种计算困难，不完全验证方法通常依赖于凸松弛来过度逼近神经网络中的非线性。在分段线性函数的更紧密逼近方面已经取得了进展。然而，神经网络对一般激活函数(例如Sigmoid、Tanh)的稳健性验证仍然没有得到充分的探索，并提出了新的挑战。通常，使用凸松弛技术来验证这些网络，该技术涉及计算非线性激活函数的线性上界和下界。在这项工作中，我们提出了一种新的参数搜索方法来提高这些线性逼近的质量。具体地说，我们通过使用一种简单的搜索方法，通过最新的算法配置技术仔细地适应给定的验证问题，在几个常用的局部健壮性验证基准上，平均将全局下界提高了25%。



## **2. Over-parameterization and Adversarial Robustness in Neural Networks: An Overview and Empirical Analysis**

神经网络中的过度参数化和对抗鲁棒性：概述和实证分析 cs.LG

**SubmitDate**: 2024-06-14    [abs](http://arxiv.org/abs/2406.10090v1) [paper-pdf](http://arxiv.org/pdf/2406.10090v1)

**Authors**: Zhang Chen, Luca Demetrio, Srishti Gupta, Xiaoyi Feng, Zhaoqiang Xia, Antonio Emanuele Cinà, Maura Pintor, Luca Oneto, Ambra Demontis, Battista Biggio, Fabio Roli

**Abstract**: Thanks to their extensive capacity, over-parameterized neural networks exhibit superior predictive capabilities and generalization. However, having a large parameter space is considered one of the main suspects of the neural networks' vulnerability to adversarial example -- input samples crafted ad-hoc to induce a desired misclassification. Relevant literature has claimed contradictory remarks in support of and against the robustness of over-parameterized networks. These contradictory findings might be due to the failure of the attack employed to evaluate the networks' robustness. Previous research has demonstrated that depending on the considered model, the algorithm employed to generate adversarial examples may not function properly, leading to overestimating the model's robustness. In this work, we empirically study the robustness of over-parameterized networks against adversarial examples. However, unlike the previous works, we also evaluate the considered attack's reliability to support the results' veracity. Our results show that over-parameterized networks are robust against adversarial attacks as opposed to their under-parameterized counterparts.

摘要: 由于其广泛的能力，超参数神经网络显示出优越的预测能力和泛化能力。然而，拥有较大的参数空间被认为是神经网络易受敌意示例攻击的主要原因之一--输入样本特别地被精心设计以导致期望的错误分类。相关文献在支持和反对过参数网络的稳健性方面发表了相互矛盾的言论。这些相互矛盾的发现可能是由于用于评估网络健壮性的攻击失败所致。以前的研究表明，根据所考虑的模型，用于生成对抗性示例的算法可能不能正常工作，导致高估了模型的稳健性。在这项工作中，我们经验地研究了过参数网络对敌意例子的稳健性。然而，与前人的工作不同，我们还评估了所考虑的攻击的可靠性，以支持结果的准确性。我们的结果表明，与欠参数网络相比，过参数网络对敌意攻击具有较强的鲁棒性。



## **3. Perturbing Attention Gives You More Bang for the Buck: Subtle Imaging Perturbations That Efficiently Fool Customized Diffusion Models**

扰动注意力为您带来更多好处：有效愚弄定制扩散模型的微妙成像扰动 cs.CV

Published at CVPR 2024, code:https://github.com/CO2-cityao/CAAT

**SubmitDate**: 2024-06-14    [abs](http://arxiv.org/abs/2404.15081v2) [paper-pdf](http://arxiv.org/pdf/2404.15081v2)

**Authors**: Jingyao Xu, Yuetong Lu, Yandong Li, Siyang Lu, Dongdong Wang, Xiang Wei

**Abstract**: Diffusion models (DMs) embark a new era of generative modeling and offer more opportunities for efficient generating high-quality and realistic data samples. However, their widespread use has also brought forth new challenges in model security, which motivates the creation of more effective adversarial attackers on DMs to understand its vulnerability. We propose CAAT, a simple but generic and efficient approach that does not require costly training to effectively fool latent diffusion models (LDMs). The approach is based on the observation that cross-attention layers exhibits higher sensitivity to gradient change, allowing for leveraging subtle perturbations on published images to significantly corrupt the generated images. We show that a subtle perturbation on an image can significantly impact the cross-attention layers, thus changing the mapping between text and image during the fine-tuning of customized diffusion models. Extensive experiments demonstrate that CAAT is compatible with diverse diffusion models and outperforms baseline attack methods in a more effective (more noise) and efficient (twice as fast as Anti-DreamBooth and Mist) manner.

摘要: 扩散模型开启了产生式建模的新时代，为高效地生成高质量和真实的数据样本提供了更多的机会。然而，它们的广泛使用也给模型安全带来了新的挑战，这促使在DM上创建更有效的对抗性攻击者来了解其脆弱性。我们提出了CAAT，这是一种简单但通用和高效的方法，不需要昂贵的培训来有效地愚弄潜在扩散模型(LDM)。该方法的基础是观察到交叉注意层对梯度变化表现出更高的敏感性，允许利用发布图像上的细微扰动来显著破坏生成的图像。我们发现，在定制扩散模型的微调过程中，图像上的细微扰动会显著影响交叉注意层，从而改变文本和图像之间的映射。大量的实验表明，CAAT与多种扩散模型兼容，并且在更有效(更多噪声)和更高效(速度是Anti-DreamBooth和Mist的两倍)方面优于基线攻击方法。



## **4. Secure Aggregation is Not Private Against Membership Inference Attacks**

安全聚合对于成员推断攻击来说不是私有的 cs.LG

accepted to the European Conference on Machine Learning and  Principles and Practice of Knowledge Discovery in Databases (ECML PKDD) 2024

**SubmitDate**: 2024-06-14    [abs](http://arxiv.org/abs/2403.17775v2) [paper-pdf](http://arxiv.org/pdf/2403.17775v2)

**Authors**: Khac-Hoang Ngo, Johan Östman, Giuseppe Durisi, Alexandre Graell i Amat

**Abstract**: Secure aggregation (SecAgg) is a commonly-used privacy-enhancing mechanism in federated learning, affording the server access only to the aggregate of model updates while safeguarding the confidentiality of individual updates. Despite widespread claims regarding SecAgg's privacy-preserving capabilities, a formal analysis of its privacy is lacking, making such presumptions unjustified. In this paper, we delve into the privacy implications of SecAgg by treating it as a local differential privacy (LDP) mechanism for each local update. We design a simple attack wherein an adversarial server seeks to discern which update vector a client submitted, out of two possible ones, in a single training round of federated learning under SecAgg. By conducting privacy auditing, we assess the success probability of this attack and quantify the LDP guarantees provided by SecAgg. Our numerical results unveil that, contrary to prevailing claims, SecAgg offers weak privacy against membership inference attacks even in a single training round. Indeed, it is difficult to hide a local update by adding other independent local updates when the updates are of high dimension. Our findings underscore the imperative for additional privacy-enhancing mechanisms, such as noise injection, in federated learning.

摘要: 安全聚合(SecAgg)是联合学习中常用的隐私增强机制，仅允许服务器访问模型更新的聚合，同时保护单个更新的机密性。尽管人们普遍声称SecAgg具有保护隐私的能力，但缺乏对其隐私的正式分析，这使得这种假设是不合理的。在本文中，我们深入研究了SecAgg的隐私含义，将其视为针对每个本地更新的本地差异隐私(LDP)机制。我们设计了一个简单的攻击，其中敌对服务器试图在SecAgg下的联合学习的单个训练轮中，从两个可能的更新向量中辨别客户端提交的更新向量。通过进行隐私审计，我们评估了该攻击的成功概率，并量化了SecAgg提供的LDP保证。我们的数值结果表明，与流行的说法相反，SecAgg即使在一轮训练中也提供了针对成员推理攻击的弱隐私。事实上，当更新是高维时，很难通过添加其他独立的本地更新来隐藏本地更新。我们的发现强调了联合学习中额外的隐私增强机制的必要性，例如噪音注入。



## **5. Defending Large Language Models Against Jailbreak Attacks via Layer-specific Editing**

通过特定层的编辑保护大型语言模型免受越狱攻击 cs.AI

**SubmitDate**: 2024-06-14    [abs](http://arxiv.org/abs/2405.18166v2) [paper-pdf](http://arxiv.org/pdf/2405.18166v2)

**Authors**: Wei Zhao, Zhe Li, Yige Li, Ye Zhang, Jun Sun

**Abstract**: Large language models (LLMs) are increasingly being adopted in a wide range of real-world applications. Despite their impressive performance, recent studies have shown that LLMs are vulnerable to deliberately crafted adversarial prompts even when aligned via Reinforcement Learning from Human Feedback or supervised fine-tuning. While existing defense methods focus on either detecting harmful prompts or reducing the likelihood of harmful responses through various means, defending LLMs against jailbreak attacks based on the inner mechanisms of LLMs remains largely unexplored. In this work, we investigate how LLMs response to harmful prompts and propose a novel defense method termed \textbf{L}ayer-specific \textbf{Ed}iting (LED) to enhance the resilience of LLMs against jailbreak attacks. Through LED, we reveal that several critical \textit{safety layers} exist among the early layers of LLMs. We then show that realigning these safety layers (and some selected additional layers) with the decoded safe response from selected target layers can significantly improve the alignment of LLMs against jailbreak attacks. Extensive experiments across various LLMs (e.g., Llama2, Mistral) show the effectiveness of LED, which effectively defends against jailbreak attacks while maintaining performance on benign prompts. Our code is available at \url{https://github.com/ledllm/ledllm}.

摘要: 大型语言模型(LLM)正越来越多地被广泛地应用于现实世界中。尽管它们的表现令人印象深刻，但最近的研究表明，即使在通过从人类反馈的强化学习或监督微调进行调整时，LLM仍容易受到故意设计的敌意提示的攻击。虽然现有的防御方法侧重于检测有害提示或通过各种手段减少有害响应的可能性，但基于LLMS的内部机制来防御LLMS的越狱攻击在很大程度上仍未被探索。在这项工作中，我们研究了LLMS对有害提示的响应，并提出了一种新的防御方法-.通过LED，我们揭示了LLMS的早期层之间存在着几个关键的安全层。然后，我们展示了将这些安全层(以及一些选定的附加层)与选定目标层的解码安全响应重新对准可以显著提高LLM对抗越狱攻击的对准。在各种LLM(如Llama2、Mistral)上的广泛实验表明，LED是有效的，它可以有效防御越狱攻击，同时保持对良性提示的性能。我们的代码可在\url{https://github.com/ledllm/ledllm}.



## **6. Reinforced Compressive Neural Architecture Search for Versatile Adversarial Robustness**

增强型压缩神经架构寻求多功能对抗鲁棒性 cs.LG

17 pages

**SubmitDate**: 2024-06-14    [abs](http://arxiv.org/abs/2406.06792v2) [paper-pdf](http://arxiv.org/pdf/2406.06792v2)

**Authors**: Dingrong Wang, Hitesh Sapkota, Zhiqiang Tao, Qi Yu

**Abstract**: Prior neural architecture search (NAS) for adversarial robustness works have discovered that a lightweight and adversarially robust neural network architecture could exist in a non-robust large teacher network, generally disclosed by heuristic rules through statistical analysis and neural architecture search, generally disclosed by heuristic rules from neural architecture search. However, heuristic methods cannot uniformly handle different adversarial attacks and "teacher" network capacity. To solve this challenge, we propose a Reinforced Compressive Neural Architecture Search (RC-NAS) for Versatile Adversarial Robustness. Specifically, we define task settings that compose datasets, adversarial attacks, and teacher network information. Given diverse tasks, we conduct a novel dual-level training paradigm that consists of a meta-training and a fine-tuning phase to effectively expose the RL agent to diverse attack scenarios (in meta-training), and making it adapt quickly to locate a sub-network (in fine-tuning) for any previously unseen scenarios. Experiments show that our framework could achieve adaptive compression towards different initial teacher networks, datasets, and adversarial attacks, resulting in more lightweight and adversarially robust architectures.

摘要: 用于对抗健壮性的先前神经体系结构搜索(NAS)工作已经发现，在非健壮的大型教师网络中可以存在轻量级和对抗性健壮的神经网络体系结构，通常由通过统计分析和神经体系结构搜索的启发式规则揭示，通常由来自神经体系结构搜索的启发式规则揭示。然而，启发式方法不能统一处理不同的对抗性攻击和“老师”网络能力。为了解决这一挑战，我们提出了一种增强的压缩神经结构搜索(RC-NAS)来实现通用的对抗健壮性。具体来说，我们定义了组成数据集、对抗性攻击和教师网络信息的任务设置。在给定不同任务的情况下，我们进行了一种新颖的双层训练范式，由元训练和微调阶段组成，以有效地将RL代理暴露于不同的攻击场景(在元训练中)，并使其快速适应以定位(在微调中)任何以前未见过的场景的子网络。实验表明，该框架能够实现对不同初始教师网络、数据集和敌意攻击的自适应压缩，从而产生更轻量级和更具对抗性的体系结构。



## **7. Watch the Watcher! Backdoor Attacks on Security-Enhancing Diffusion Models**

观看守望者！对安全增强扩散模型的后门攻击 cs.CR

**SubmitDate**: 2024-06-14    [abs](http://arxiv.org/abs/2406.09669v1) [paper-pdf](http://arxiv.org/pdf/2406.09669v1)

**Authors**: Changjiang Li, Ren Pang, Bochuan Cao, Jinghui Chen, Fenglong Ma, Shouling Ji, Ting Wang

**Abstract**: Thanks to their remarkable denoising capabilities, diffusion models are increasingly being employed as defensive tools to reinforce the security of other models, notably in purifying adversarial examples and certifying adversarial robustness. However, the security risks of these practices themselves remain largely unexplored, which is highly concerning. To bridge this gap, this work investigates the vulnerabilities of security-enhancing diffusion models. Specifically, we demonstrate that these models are highly susceptible to DIFF2, a simple yet effective backdoor attack, which substantially diminishes the security assurance provided by such models. Essentially, DIFF2 achieves this by integrating a malicious diffusion-sampling process into the diffusion model, guiding inputs embedded with specific triggers toward an adversary-defined distribution while preserving the normal functionality for clean inputs. Our case studies on adversarial purification and robustness certification show that DIFF2 can significantly reduce both post-purification and certified accuracy across benchmark datasets and models, highlighting the potential risks of relying on pre-trained diffusion models as defensive tools. We further explore possible countermeasures, suggesting promising avenues for future research.

摘要: 由于其显著的去噪能力，扩散模型越来越多地被用作防御工具来加强其他模型的安全性，特别是在净化对手实例和证明对手稳健性方面。然而，这些做法本身的安全风险在很大程度上仍未得到探索，这令人高度关切。为了弥补这一差距，这项工作调查了安全增强扩散模型的漏洞。具体地说，我们证明这些模型非常容易受到DIFF2的攻击，DIFF2是一种简单但有效的后门攻击，它极大地降低了此类模型提供的安全保证。基本上，DIFF2通过将恶意扩散采样过程集成到扩散模型中来实现这一点，将嵌入特定触发器的输入引导到对手定义的分布，同时保留正常的干净输入功能。我们的对抗性净化和健壮性认证的案例研究表明，DIFF2可以显著降低净化后的准确性和跨基准数据集和模型的认证准确性，突出了依赖预先训练的扩散模型作为防御工具的潜在风险。我们进一步探索了可能的对策，为未来的研究提出了有希望的途径。



## **8. Towards Evaluating the Robustness of Visual State Space Models**

评估视觉状态空间模型的稳健性 cs.CV

**SubmitDate**: 2024-06-13    [abs](http://arxiv.org/abs/2406.09407v1) [paper-pdf](http://arxiv.org/pdf/2406.09407v1)

**Authors**: Hashmat Shadab Malik, Fahad Shamshad, Muzammal Naseer, Karthik Nandakumar, Fahad Shahbaz Khan, Salman Khan

**Abstract**: Vision State Space Models (VSSMs), a novel architecture that combines the strengths of recurrent neural networks and latent variable models, have demonstrated remarkable performance in visual perception tasks by efficiently capturing long-range dependencies and modeling complex visual dynamics. However, their robustness under natural and adversarial perturbations remains a critical concern. In this work, we present a comprehensive evaluation of VSSMs' robustness under various perturbation scenarios, including occlusions, image structure, common corruptions, and adversarial attacks, and compare their performance to well-established architectures such as transformers and Convolutional Neural Networks. Furthermore, we investigate the resilience of VSSMs to object-background compositional changes on sophisticated benchmarks designed to test model performance in complex visual scenes. We also assess their robustness on object detection and segmentation tasks using corrupted datasets that mimic real-world scenarios. To gain a deeper understanding of VSSMs' adversarial robustness, we conduct a frequency analysis of adversarial attacks, evaluating their performance against low-frequency and high-frequency perturbations. Our findings highlight the strengths and limitations of VSSMs in handling complex visual corruptions, offering valuable insights for future research and improvements in this promising field. Our code and models will be available at https://github.com/HashmatShadab/MambaRobustness.

摘要: 视觉状态空间模型(VSSMS)是一种结合了递归神经网络和潜变量模型优点的新型结构，通过有效地捕捉长距离依赖关系和建模复杂的视觉动力学，在视觉感知任务中表现出了显著的性能。然而，它们在自然和对抗性扰动下的稳健性仍然是一个严重的问题。在这项工作中，我们对VSSM在各种扰动场景下的健壮性进行了全面的评估，包括遮挡、图像结构、常见的腐败和敌对攻击，并将它们的性能与成熟的架构，如变压器和卷积神经网络进行了比较。此外，我们在复杂的基准测试中考察了VSSM对对象-背景成分变化的弹性，该基准旨在测试复杂视觉场景中的模型性能。我们还使用模拟真实世界场景的损坏数据集评估了它们在对象检测和分割任务中的稳健性。为了更深入地了解VSSM的对抗稳健性，我们对对抗攻击进行了频率分析，评估了它们对低频和高频扰动的性能。我们的发现突出了VSSM在处理复杂视觉腐败方面的优势和局限性，为这一前景广阔的领域的未来研究和改进提供了有价值的见解。我们的代码和模型将在https://github.com/HashmatShadab/MambaRobustness.上提供



## **9. MirrorCheck: Efficient Adversarial Defense for Vision-Language Models**

收件箱检查：视觉语言模型的有效对抗防御 cs.CV

**SubmitDate**: 2024-06-13    [abs](http://arxiv.org/abs/2406.09250v1) [paper-pdf](http://arxiv.org/pdf/2406.09250v1)

**Authors**: Samar Fares, Klea Ziu, Toluwani Aremu, Nikita Durasov, Martin Takáč, Pascal Fua, Karthik Nandakumar, Ivan Laptev

**Abstract**: Vision-Language Models (VLMs) are becoming increasingly vulnerable to adversarial attacks as various novel attack strategies are being proposed against these models. While existing defenses excel in unimodal contexts, they currently fall short in safeguarding VLMs against adversarial threats. To mitigate this vulnerability, we propose a novel, yet elegantly simple approach for detecting adversarial samples in VLMs. Our method leverages Text-to-Image (T2I) models to generate images based on captions produced by target VLMs. Subsequently, we calculate the similarities of the embeddings of both input and generated images in the feature space to identify adversarial samples. Empirical evaluations conducted on different datasets validate the efficacy of our approach, outperforming baseline methods adapted from image classification domains. Furthermore, we extend our methodology to classification tasks, showcasing its adaptability and model-agnostic nature. Theoretical analyses and empirical findings also show the resilience of our approach against adaptive attacks, positioning it as an excellent defense mechanism for real-world deployment against adversarial threats.

摘要: 随着针对视觉语言模型的各种新的攻击策略的提出，视觉语言模型正变得越来越容易受到对手攻击。虽然现有的防御系统在单峰环境中表现出色，但它们目前在保护VLM免受对手威胁方面存在不足。为了缓解这一漏洞，我们提出了一种新颖而又非常简单的方法来检测VLM中的敌意样本。我们的方法利用文本到图像(T2I)模型来生成基于目标VLM生成的字幕的图像。随后，我们计算输入图像和生成图像在特征空间中的嵌入相似度，以识别敌意样本。在不同的数据集上进行的经验评估验证了我们方法的有效性，优于适用于图像分类领域的基线方法。此外，我们将我们的方法扩展到分类任务，展示了其适应性和模型不可知性。理论分析和经验结果也表明了我们的方法对适应性攻击的弹性，将其定位为针对对抗性威胁的真实部署的优秀防御机制。



## **10. After the Breach: Incident Response within Enterprises**

违约后：企业内部的事件响应 cs.CR

**SubmitDate**: 2024-06-13    [abs](http://arxiv.org/abs/2406.07559v2) [paper-pdf](http://arxiv.org/pdf/2406.07559v2)

**Authors**: Sumanth Rao

**Abstract**: Enterprises are constantly under attack from sophisticated adversaries. These adversaries use a variety of techniques to first gain access to the enterprise, then spread laterally inside its networks, establish persistence, and finally exfiltrate sensitive data, or hold it for ransom. While historically, enterprises have used different Incident Response systems that monitor hosts, servers, or network devices to detect and report threats, these systems often need many analysts to triage and respond to alerts. However, the immense quantity of alerts to sift through, combined with the potential risk of missing a valid threat makes the task of the analyst challenging. To ease this manual and laborious process, researchers have proposed a variety of systems that perform automated attack investigations. These systems collect data, track causally related events, and present the analyst with an interpretable summary of the attack. In this paper, we present a survey of systems that perform automated attack investigation, and compare them based on their designs, goals, and heuristics. We discuss the challenges faced by these systems, and present a comparison in terms of their effectiveness, practicality, and ability to address these challenges. We conclude by discussing the future of these systems, and the open problems in this area.

摘要: 企业不断受到老练对手的攻击。这些对手使用各种技术首先访问企业，然后在其网络内横向传播，建立持久性，最后泄露敏感数据，或扣留它以换取赎金。虽然在历史上，企业使用不同的事件响应系统来监控主机、服务器或网络设备来检测和报告威胁，但这些系统通常需要许多分析师来对警报进行分类和响应。然而，要筛选的大量警报，再加上错过有效威胁的潜在风险，使得分析师的任务具有挑战性。为了简化这一手动且费力的过程，研究人员提出了各种执行自动攻击调查的系统。这些系统收集数据，跟踪因果相关事件，并向分析师提供可解释的攻击摘要。在本文中，我们对执行自动攻击调查的系统进行了综述，并根据它们的设计、目标和启发式方法对它们进行了比较。我们讨论了这些系统面临的挑战，并就它们的有效性、实用性和应对这些挑战的能力进行了比较。最后，我们讨论了这些系统的未来，以及这一领域存在的问题。



## **11. Is Diffusion Model Safe? Severe Data Leakage via Gradient-Guided Diffusion Model**

扩散模型安全吗？通过物质引导扩散模型的严重数据泄露 cs.CV

**SubmitDate**: 2024-06-13    [abs](http://arxiv.org/abs/2406.09484v1) [paper-pdf](http://arxiv.org/pdf/2406.09484v1)

**Authors**: Jiayang Meng, Tao Huang, Hong Chen, Cuiping Li

**Abstract**: Gradient leakage has been identified as a potential source of privacy breaches in modern image processing systems, where the adversary can completely reconstruct the training images from leaked gradients. However, existing methods are restricted to reconstructing low-resolution images where data leakage risks of image processing systems are not sufficiently explored. In this paper, by exploiting diffusion models, we propose an innovative gradient-guided fine-tuning method and introduce a new reconstruction attack that is capable of stealing private, high-resolution images from image processing systems through leaked gradients where severe data leakage encounters. Our attack method is easy to implement and requires little prior knowledge. The experimental results indicate that current reconstruction attacks can steal images only up to a resolution of $128 \times 128$ pixels, while our attack method can successfully recover and steal images with resolutions up to $512 \times 512$ pixels. Our attack method significantly outperforms the SOTA attack baselines in terms of both pixel-wise accuracy and time efficiency of image reconstruction. Furthermore, our attack can render differential privacy ineffective to some extent.

摘要: 在现代图像处理系统中，梯度泄漏被认为是隐私泄露的一个潜在来源，攻击者可以从泄漏的梯度中完全重建训练图像。然而，现有的方法仅限于重建低分辨率的图像，其中图像处理系统的数据泄漏风险没有得到充分的研究。在本文中，我们利用扩散模型，提出了一种创新的梯度引导微调方法，并引入了一种新的重建攻击，该攻击能够通过泄漏的梯度从图像处理系统中窃取私人的高分辨率图像，在那里遇到严重的数据泄漏。我们的攻击方法很容易实现，并且几乎不需要先验知识。实验结果表明，目前的重建攻击只能盗取分辨率为$128\x 128$的图像，而本文的攻击方法可以成功恢复和窃取分辨率为$512\x 512$的图像。我们的攻击方法在图像重建的像素级精度和时间效率方面都明显优于SOTA攻击基线。此外，我们的攻击在一定程度上会使差异隐私无效。



## **12. Potion: Towards Poison Unlearning**

药剂：走向毒药的学习 cs.LG

**SubmitDate**: 2024-06-13    [abs](http://arxiv.org/abs/2406.09173v1) [paper-pdf](http://arxiv.org/pdf/2406.09173v1)

**Authors**: Stefan Schoepf, Jack Foster, Alexandra Brintrup

**Abstract**: Adversarial attacks by malicious actors on machine learning systems, such as introducing poison triggers into training datasets, pose significant risks. The challenge in resolving such an attack arises in practice when only a subset of the poisoned data can be identified. This necessitates the development of methods to remove, i.e. unlearn, poison triggers from already trained models with only a subset of the poison data available. The requirements for this task significantly deviate from privacy-focused unlearning where all of the data to be forgotten by the model is known. Previous work has shown that the undiscovered poisoned samples lead to a failure of established unlearning methods, with only one method, Selective Synaptic Dampening (SSD), showing limited success. Even full retraining, after the removal of the identified poison, cannot address this challenge as the undiscovered poison samples lead to a reintroduction of the poison trigger in the model. Our work addresses two key challenges to advance the state of the art in poison unlearning. First, we introduce a novel outlier-resistant method, based on SSD, that significantly improves model protection and unlearning performance. Second, we introduce Poison Trigger Neutralisation (PTN) search, a fast, parallelisable, hyperparameter search that utilises the characteristic "unlearning versus model protection" trade-off to find suitable hyperparameters in settings where the forget set size is unknown and the retain set is contaminated. We benchmark our contributions using ResNet-9 on CIFAR10 and WideResNet-28x10 on CIFAR100. Experimental results show that our method heals 93.72% of poison compared to SSD with 83.41% and full retraining with 40.68%. We achieve this while also lowering the average model accuracy drop caused by unlearning from 5.68% (SSD) to 1.41% (ours).

摘要: 恶意行为者对机器学习系统的对抗性攻击，如将有毒触发器引入训练数据集，构成了巨大的风险。解决此类攻击的挑战出现在实践中，当只能识别有毒数据的子集时。这就需要开发方法来从仅有有毒数据的子集的已训练模型中移除(即取消学习)有毒触发器。这项任务的要求与关注隐私的遗忘有很大不同，在隐私遗忘中，模型要忘记的所有数据都是已知的。以前的工作表明，未发现的中毒样本会导致已有的遗忘方法的失败，只有一种方法-选择性突触抑制(SSD)-显示出有限的成功。即使在去除已识别的毒物之后进行全面的再培训，也不能解决这一挑战，因为未发现的毒物样本会导致在模型中重新引入毒物触发器。我们的工作解决了两个关键挑战，以推进毒物忘却学习的艺术水平。首先，我们提出了一种新的基于SSD的抗孤立点方法，该方法显著改善了模型保护和遗忘性能。其次，我们引入了毒药触发中和(PTN)搜索，这是一种快速、可并行的超参数搜索，它利用“忘却学习与模型保护”的折衷特性，在忘记集大小未知且保留集受到污染的情况下找到合适的超参数。我们使用CIFAR10上的ResNet-9和CIFAR100上的WideResNet-28x10对我们的贡献进行基准测试。实验结果表明，与SSD的83.41%和完全再训练的40.68%相比，我们的方法可以治愈93.72%的毒物。我们实现了这一点，同时也将因遗忘而导致的平均模型精度下降从5.68%(SSD)降至1.41%(我们的)。



## **13. DifAttack++: Query-Efficient Black-Box Adversarial Attack via Hierarchical Disentangled Feature Space in Cross Domain**

DifAttack++：跨域中通过分层解纠缠特征空间进行查询高效黑匣子对抗攻击 cs.CV

arXiv admin note: substantial text overlap with arXiv:2309.14585 An  extension of the AAAI24 paper "DifAttack: Query-Efficient Black-Box Attack  via Disentangled Feature Space."

**SubmitDate**: 2024-06-13    [abs](http://arxiv.org/abs/2406.03017v2) [paper-pdf](http://arxiv.org/pdf/2406.03017v2)

**Authors**: Jun Liu, Jiantao Zhou, Jiandian Zeng, Jinyu Tian

**Abstract**: This work investigates efficient score-based black-box adversarial attacks with a high Attack Success Rate (ASR) and good generalizability. We design a novel attack method based on a \textit{Hierarchical} \textbf{Di}sentangled \textbf{F}eature space and \textit{cross domain}, called \textbf{DifAttack++}, which differs significantly from the existing ones operating over the entire feature space. Specifically, DifAttack++ firstly disentangles an image's latent feature into an \textit{adversarial feature} (AF) and a \textit{visual feature} (VF) via an autoencoder equipped with our specially designed \textbf{H}ierarchical \textbf{D}ecouple-\textbf{F}usion (HDF) module, where the AF dominates the adversarial capability of an image, while the VF largely determines its visual appearance. We train such autoencoders for the clean and adversarial image domains respectively, meanwhile realizing feature disentanglement, by using pairs of clean images and their Adversarial Examples (AEs) generated from available surrogate models via white-box attack methods. Eventually, in the black-box attack stage, DifAttack++ iteratively optimizes the AF according to the query feedback from the victim model until a successful AE is generated, while keeping the VF unaltered. Extensive experimental results demonstrate that our method achieves superior ASR and query efficiency than SOTA methods, meanwhile exhibiting much better visual quality of AEs. The code is available at https://github.com/csjunjun/DifAttack.git.

摘要: 研究了基于分数的高效黑盒对抗攻击，具有较高的攻击成功率(ASR)和良好的泛化能力。我们设计了一种新的攻击方法，该方法不同于现有的在整个特征空间上操作的攻击方法，它是基于文本{层次}\extbf{Di}定位的文本空间和文本{跨域}的攻击方法。具体地说，DifAttack++首先通过一个自动编码器将图像的潜在特征分解为文本{对抗性特征}(AF)和文本{视觉特征}(VF)，该自动编码器配备了我们特别设计的\textbf{H}ierarchical\extbf{D}生态耦合(HDF)模块，其中，对抗性特征主导图像的对抗能力，而视觉特征在很大程度上决定了图像的视觉外观。利用已有的代理模型生成的干净图像及其对抗性实例(AE)，通过白盒攻击方法，分别训练干净图像域和对抗性图像域的自动编码器，同时实现特征解缠。最终，在黑盒攻击阶段，DifAttack++根据受害者模型的查询反馈迭代地优化AF，直到生成成功的AE，同时保持VF不变。大量的实验结果表明，与SOTA方法相比，该方法获得了更好的ASR和查询效率，同时也表现出了更好的AEs视觉质量。代码可在https://github.com/csjunjun/DifAttack.git.上获得



## **14. Beyond Labeling Oracles: What does it mean to steal ML models?**

超越标记Oracle：窃取ML模型意味着什么？ cs.LG

**SubmitDate**: 2024-06-13    [abs](http://arxiv.org/abs/2310.01959v3) [paper-pdf](http://arxiv.org/pdf/2310.01959v3)

**Authors**: Avital Shafran, Ilia Shumailov, Murat A. Erdogdu, Nicolas Papernot

**Abstract**: Model extraction attacks are designed to steal trained models with only query access, as is often provided through APIs that ML-as-a-Service providers offer. Machine Learning (ML) models are expensive to train, in part because data is hard to obtain, and a primary incentive for model extraction is to acquire a model while incurring less cost than training from scratch. Literature on model extraction commonly claims or presumes that the attacker is able to save on both data acquisition and labeling costs. We thoroughly evaluate this assumption and find that the attacker often does not. This is because current attacks implicitly rely on the adversary being able to sample from the victim model's data distribution. We thoroughly research factors influencing the success of model extraction. We discover that prior knowledge of the attacker, i.e., access to in-distribution data, dominates other factors like the attack policy the adversary follows to choose which queries to make to the victim model API. Our findings urge the community to redefine the adversarial goals of ME attacks as current evaluation methods misinterpret the ME performance.

摘要: 模型提取攻击旨在窃取仅具有查询访问权限的经过训练的模型，这通常是通过ML-as-a-Service提供商提供的API提供的。机器学习(ML)模型的训练成本很高，部分原因是数据很难获得，而模型提取的主要动机是在获得模型的同时比从头开始训练的成本更低。有关模型提取的文献通常声称或假设攻击者能够节省数据获取和标记成本。我们彻底评估了这一假设，发现攻击者通常不会这样做。这是因为当前的攻击隐含地依赖于对手能够从受害者模型的数据分布中进行采样。深入研究了影响模型提取成功的因素。我们发现，攻击者的先验知识，即对分发内数据的访问，主导了其他因素，如攻击者选择对受害者模型API进行哪些查询所遵循的攻击策略。我们的发现敦促社区重新定义ME攻击的对抗性目标，因为当前的评估方法误解了ME的表现。



## **15. Hyper-parameter Tuning for Adversarially Robust Models**

对抗鲁棒模型的超参数调整 cs.LG

**SubmitDate**: 2024-06-13    [abs](http://arxiv.org/abs/2304.02497v3) [paper-pdf](http://arxiv.org/pdf/2304.02497v3)

**Authors**: Pedro Mendes, Paolo Romano, David Garlan

**Abstract**: This work focuses on the problem of hyper-parameter tuning (HPT) for robust (i.e., adversarially trained) models, shedding light on the new challenges and opportunities arising during the HPT process for robust models. To this end, we conduct an extensive experimental study based on 3 popular deep models, in which we explore exhaustively 9 (discretized) HPs, 2 fidelity dimensions, and 2 attack bounds, for a total of 19208 configurations (corresponding to 50 thousand GPU hours). Through this study, we show that the complexity of the HPT problem is further exacerbated in adversarial settings due to the need to independently tune the HPs used during standard and adversarial training: succeeding in doing so (i.e., adopting different HP settings in both phases) can lead to a reduction of up to 80% and 43% of the error for clean and adversarial inputs, respectively. On the other hand, we also identify new opportunities to reduce the cost of HPT for robust models. Specifically, we propose to leverage cheap adversarial training methods to obtain inexpensive, yet highly correlated, estimations of the quality achievable using state-of-the-art methods. We show that, by exploiting this novel idea in conjunction with a recent multi-fidelity optimizer (taKG), the efficiency of the HPT process can be enhanced by up to 2.1x.

摘要: 这项工作集中于稳健(即反向训练)模型的超参数调节(HPT)问题，揭示了稳健模型HPT过程中出现的新挑战和新机遇。为此，我们基于3个流行的深度模型进行了广泛的实验研究，其中我们详尽地探索了9个(离散化的)HPS，2个保真维度，2个攻击界限，总共19208个配置(对应于5万个GPU小时)。通过这项研究，我们表明，由于需要独立调整标准和对抗性训练中使用的HP，HPT问题的复杂性在对抗性环境中进一步加剧：成功做到这一点(即在两个阶段采用不同的HP设置)可以使干净输入和对抗性输入的错误分别减少80%和43%。另一方面，我们也发现了新的机会，以降低稳健模型的HPT成本。具体地说，我们建议利用廉价的对抗性训练方法来获得对使用最先进方法可实现的质量的廉价但高度相关的估计。我们证明，通过利用这一新的想法与最近的多保真优化器(TaKG)相结合，HPT过程的效率可以提高高达2.1倍。



## **16. Improving Adversarial Robustness via Feature Pattern Consistency Constraint**

通过特征模式一致性约束提高对抗鲁棒性 cs.CV

**SubmitDate**: 2024-06-13    [abs](http://arxiv.org/abs/2406.08829v1) [paper-pdf](http://arxiv.org/pdf/2406.08829v1)

**Authors**: Jiacong Hu, Jingwen Ye, Zunlei Feng, Jiazhen Yang, Shunyu Liu, Xiaotian Yu, Lingxiang Jia, Mingli Song

**Abstract**: Convolutional Neural Networks (CNNs) are well-known for their vulnerability to adversarial attacks, posing significant security concerns. In response to these threats, various defense methods have emerged to bolster the model's robustness. However, most existing methods either focus on learning from adversarial perturbations, leading to overfitting to the adversarial examples, or aim to eliminate such perturbations during inference, inevitably increasing computational burdens. Conversely, clean training, which strengthens the model's robustness by relying solely on clean examples, can address the aforementioned issues. In this paper, we align with this methodological stream and enhance its generalizability to unknown adversarial examples. This enhancement is achieved by scrutinizing the behavior of latent features within the network. Recognizing that a correct prediction relies on the correctness of the latent feature's pattern, we introduce a novel and effective Feature Pattern Consistency Constraint (FPCC) method to reinforce the latent feature's capacity to maintain the correct feature pattern. Specifically, we propose Spatial-wise Feature Modification and Channel-wise Feature Selection to enhance latent features. Subsequently, we employ the Pattern Consistency Loss to constrain the similarity between the feature pattern of the latent features and the correct feature pattern. Our experiments demonstrate that the FPCC method empowers latent features to uphold correct feature patterns even in the face of adversarial examples, resulting in inherent adversarial robustness surpassing state-of-the-art models.

摘要: 卷积神经网络(CNN)因其易受敌意攻击而广为人知，造成了严重的安全问题。为了应对这些威胁，各种防御方法应运而生，以增强模型的稳健性。然而，现有的大多数方法要么专注于从对抗性扰动中学习，导致对对抗性实例的过度拟合，要么旨在消除推理过程中的此类扰动，不可避免地增加了计算负担。相反，仅依靠干净的例子来增强模型的健壮性的干净的训练可以解决上述问题。在本文中，我们与这一方法论流保持一致，并增强了其对未知对抗性例子的推广能力。这种增强是通过仔细检查网络中潜在特征的行为来实现的。认识到正确的预测依赖于潜在特征模式的正确性，我们引入了一种新颖而有效的特征模式一致性约束(FPCC)方法来增强潜在特征保持正确特征模式的能力。具体地说，我们提出了基于空间的特征修正和基于通道的特征选择来增强潜在特征。随后，我们使用模式一致性损失来约束潜在特征的特征模式与正确特征模式之间的相似性。我们的实验表明，FPCC方法赋予潜在特征以保持正确的特征模式，即使面对对抗性的例子，导致固有的对抗性的稳健性超过最新的模型。



## **17. Ranking Manipulation for Conversational Search Engines**

对话式搜索引擎的排名操纵 cs.CL

**SubmitDate**: 2024-06-13    [abs](http://arxiv.org/abs/2406.03589v2) [paper-pdf](http://arxiv.org/pdf/2406.03589v2)

**Authors**: Samuel Pfrommer, Yatong Bai, Tanmay Gautam, Somayeh Sojoudi

**Abstract**: Major search engine providers are rapidly incorporating Large Language Model (LLM)-generated content in response to user queries. These conversational search engines operate by loading retrieved website text into the LLM context for summarization and interpretation. Recent research demonstrates that LLMs are highly vulnerable to jailbreaking and prompt injection attacks, which disrupt the safety and quality goals of LLMs using adversarial strings. This work investigates the impact of prompt injections on the ranking order of sources referenced by conversational search engines. To this end, we introduce a focused dataset of real-world consumer product websites and formalize conversational search ranking as an adversarial problem. Experimentally, we analyze conversational search rankings in the absence of adversarial injections and show that different LLMs vary significantly in prioritizing product name, document content, and context position. We then present a tree-of-attacks-based jailbreaking technique which reliably promotes low-ranked products. Importantly, these attacks transfer effectively to state-of-the-art conversational search engines such as perplexity.ai. Given the strong financial incentive for website owners to boost their search ranking, we argue that our problem formulation is of critical importance for future robustness work.

摘要: 各大搜索引擎提供商正在快速整合大型语言模型(LLM)生成的内容，以响应用户查询。这些对话式搜索引擎通过将检索到的网站文本加载到LLM上下文中进行操作以进行摘要和解释。最近的研究表明，LLM非常容易受到越狱和快速注入攻击，这些攻击使用敌意字符串破坏LLM的安全和质量目标。这项工作调查了提示注入对对话式搜索引擎引用的来源的排名顺序的影响。为此，我们引入了一个聚焦于真实世界消费产品网站的数据集，并将会话搜索排名形式化为一个对抗性问题。在实验上，我们分析了在没有对抗性注入的情况下的会话搜索排名，结果表明不同的LLM在产品名称、文档内容和上下文位置的优先顺序上存在显著差异。然后，我们提出了一种基于攻击树的越狱技术，该技术可靠地推广排名较低的产品。重要的是，这些攻击有效地转移到了最先进的会话搜索引擎，如Pplexity.ai。考虑到网站所有者有强大的经济动机来提高他们的搜索排名，我们认为我们的问题表达对于未来的稳健性工作至关重要。



## **18. On Security Weaknesses and Vulnerabilities in Deep Learning Systems**

深度学习系统中的安全弱点和漏洞 cs.SE

**SubmitDate**: 2024-06-12    [abs](http://arxiv.org/abs/2406.08688v1) [paper-pdf](http://arxiv.org/pdf/2406.08688v1)

**Authors**: Zhongzheng Lai, Huaming Chen, Ruoxi Sun, Yu Zhang, Minhui Xue, Dong Yuan

**Abstract**: The security guarantee of AI-enabled software systems (particularly using deep learning techniques as a functional core) is pivotal against the adversarial attacks exploiting software vulnerabilities. However, little attention has been paid to a systematic investigation of vulnerabilities in such systems. A common situation learned from the open source software community is that deep learning engineers frequently integrate off-the-shelf or open-source learning frameworks into their ecosystems. In this work, we specifically look into deep learning (DL) framework and perform the first systematic study of vulnerabilities in DL systems through a comprehensive analysis of identified vulnerabilities from Common Vulnerabilities and Exposures (CVE) and open-source DL tools, including TensorFlow, Caffe, OpenCV, Keras, and PyTorch. We propose a two-stream data analysis framework to explore vulnerability patterns from various databases. We investigate the unique DL frameworks and libraries development ecosystems that appear to be decentralized and fragmented. By revisiting the Common Weakness Enumeration (CWE) List, which provides the traditional software vulnerability related practices, we observed that it is more challenging to detect and fix the vulnerabilities throughout the DL systems lifecycle. Moreover, we conducted a large-scale empirical study of 3,049 DL vulnerabilities to better understand the patterns of vulnerability and the challenges in fixing them. We have released the full replication package at https://github.com/codelzz/Vulnerabilities4DLSystem. We anticipate that our study can advance the development of secure DL systems.

摘要: 人工智能软件系统的安全保障(特别是以深度学习技术为功能核心)对于利用软件漏洞进行的对抗性攻击至关重要。然而，很少有人注意到对这类系统中的漏洞进行系统调查。从开源软件社区了解到的一个常见情况是，深度学习工程师经常将现成的或开源的学习框架集成到他们的生态系统中。在这项工作中，我们专门研究了深度学习(DL)框架，并通过对常见漏洞和暴露(CVE)和开源DL工具(包括TensorFlow、Caffe、OpenCV、Kera和PyTorch)识别的漏洞进行了全面分析，首次对深度学习系统中的漏洞进行了系统研究。我们提出了一个双流数据分析框架，从不同的数据库中探索漏洞模式。我们调查了独特的数字图书馆框架和图书馆开发生态系统，这些生态系统似乎是分散和支离破碎的。通过重新访问提供传统软件漏洞相关实践的常见弱点枚举(CWE)列表，我们发现在DL系统的整个生命周期中检测和修复漏洞更具挑战性。此外，我们对3,049个DL漏洞进行了大规模的实证研究，以更好地了解漏洞的模式和修复它们的挑战。我们已在https://github.com/codelzz/Vulnerabilities4DLSystem.发布了完整的复制程序包我们期望我们的研究能够推动安全下行链路系统的发展。



## **19. On Evaluating Adversarial Robustness of Volumetric Medical Segmentation Models**

评估容量医疗细分模型的对抗稳健性 eess.IV

**SubmitDate**: 2024-06-12    [abs](http://arxiv.org/abs/2406.08486v1) [paper-pdf](http://arxiv.org/pdf/2406.08486v1)

**Authors**: Hashmat Shadab Malik, Numan Saeed, Asif Hanif, Muzammal Naseer, Mohammad Yaqub, Salman Khan, Fahad Shahbaz Khan

**Abstract**: Volumetric medical segmentation models have achieved significant success on organ and tumor-based segmentation tasks in recent years. However, their vulnerability to adversarial attacks remains largely unexplored, raising serious concerns regarding the real-world deployment of tools employing such models in the healthcare sector. This underscores the importance of investigating the robustness of existing models. In this context, our work aims to empirically examine the adversarial robustness across current volumetric segmentation architectures, encompassing Convolutional, Transformer, and Mamba-based models. We extend this investigation across four volumetric segmentation datasets, evaluating robustness under both white box and black box adversarial attacks. Overall, we observe that while both pixel and frequency-based attacks perform reasonably well under white box setting, the latter performs significantly better under transfer-based black box attacks. Across our experiments, we observe transformer-based models show higher robustness than convolution-based models with Mamba-based models being the most vulnerable. Additionally, we show that large-scale training of volumetric segmentation models improves the model's robustness against adversarial attacks. The code and pretrained models will be made available at https://github.com/HashmatShadab/Robustness-of-Volumetric-Medical-Segmentation-Models.

摘要: 近年来，体积医学分割模型在基于器官和肿瘤的分割任务中取得了显著的成功。然而，它们在对抗性攻击中的脆弱性在很大程度上仍未得到探索，这引发了人们对在医疗保健部门使用此类模型的工具在现实世界中的部署的严重关切。这凸显了调查现有模型的稳健性的重要性。在此背景下，我们的工作旨在经验性地检查当前体积分割体系结构的对抗性健壮性，包括卷积、变压器和基于Mamba的模型。我们将这项研究扩展到四个体积分割数据集，评估了白盒和黑盒攻击下的稳健性。总体而言，我们观察到，虽然基于像素的攻击和基于频率的攻击在白盒设置下都表现得相当好，但后者在基于传输的黑盒攻击下的性能要好得多。在我们的实验中，我们观察到基于变压器的模型比基于卷积的模型表现出更高的稳健性，其中基于Mamba的模型是最脆弱的。此外，我们还证明了对体积分割模型的大规模训练提高了模型对对手攻击的稳健性。代码和预先培训的模型将在https://github.com/HashmatShadab/Robustness-of-Volumetric-Medical-Segmentation-Models.上提供



## **20. Transformation-Dependent Adversarial Attacks**

依赖转换的对抗攻击 cs.CV

**SubmitDate**: 2024-06-12    [abs](http://arxiv.org/abs/2406.08443v1) [paper-pdf](http://arxiv.org/pdf/2406.08443v1)

**Authors**: Yaoteng Tan, Zikui Cai, M. Salman Asif

**Abstract**: We introduce transformation-dependent adversarial attacks, a new class of threats where a single additive perturbation can trigger diverse, controllable mis-predictions by systematically transforming the input (e.g., scaling, blurring, compression). Unlike traditional attacks with static effects, our perturbations embed metamorphic properties to enable different adversarial attacks as a function of the transformation parameters. We demonstrate the transformation-dependent vulnerability across models (e.g., convolutional networks and vision transformers) and vision tasks (e.g., image classification and object detection). Our proposed geometric and photometric transformations enable a range of targeted errors from one crafted input (e.g., higher than 90% attack success rate for classifiers). We analyze effects of model architecture and type/variety of transformations on attack effectiveness. This work forces a paradigm shift by redefining adversarial inputs as dynamic, controllable threats. We highlight the need for robust defenses against such multifaceted, chameleon-like perturbations that current techniques are ill-prepared for.

摘要: 我们引入了依赖于变换的对抗性攻击，这是一类新的威胁，其中单个加性扰动可以通过系统地变换输入(例如，缩放、模糊、压缩)来触发各种可控的误预测。与具有静态效果的传统攻击不同，我们的扰动嵌入了变形属性，以使不同的对抗性攻击能够作为变换参数的函数。我们展示了跨模型(例如卷积网络和视觉转换器)和视觉任务(例如图像分类和目标检测)的依赖于变换的脆弱性。我们提出的几何和光度转换允许从一个精心制作的输入中产生一系列目标错误(例如，分类器的攻击成功率高于90%)。我们分析了模型体系结构和转换类型/种类对攻击效果的影响。这项工作通过将敌意输入重新定义为动态的、可控的威胁，迫使范式发生转变。我们强调需要对这种多方面的、像变色龙一样的扰动进行强有力的防御，而目前的技术对这种扰动准备不足。



## **21. Improving Noise Robustness through Abstractions and its Impact on Machine Learning**

通过抽象提高噪音稳健性及其对机器学习的影响 cs.LG

**SubmitDate**: 2024-06-12    [abs](http://arxiv.org/abs/2406.08428v1) [paper-pdf](http://arxiv.org/pdf/2406.08428v1)

**Authors**: Alfredo Ibias, Karol Capala, Varun Ravi Varma, Anna Drozdz, Jose Sousa

**Abstract**: Noise is a fundamental problem in learning theory with huge effects in the application of Machine Learning (ML) methods, due to real world data tendency to be noisy. Additionally, introduction of malicious noise can make ML methods fail critically, as is the case with adversarial attacks. Thus, finding and developing alternatives to improve robustness to noise is a fundamental problem in ML. In this paper, we propose a method to deal with noise: mitigating its effect through the use of data abstractions. The goal is to reduce the effect of noise over the model's performance through the loss of information produced by the abstraction. However, this information loss comes with a cost: it can result in an accuracy reduction due to the missing information. First, we explored multiple methodologies to create abstractions, using the training dataset, for the specific case of numerical data and binary classification tasks. We also tested how these abstractions can affect robustness to noise with several experiments that explore the robustness of an Artificial Neural Network to noise when trained using raw data \emph{vs} when trained using abstracted data. The results clearly show that using abstractions is a viable approach for developing noise robust ML methods.

摘要: 噪声是学习理论中的一个基本问题，在机器学习方法的应用中有着巨大的影响，因为现实世界中的数据往往是噪声。此外，恶意噪声的引入会使ML方法严重失败，就像对抗性攻击一样。因此，寻找和开发替代方案以提高对噪声的稳健性是ML中的一个基本问题。在本文中，我们提出了一种处理噪声的方法：通过使用数据抽象来减轻噪声的影响。目标是通过抽象产生的信息损失来减少噪声对模型性能的影响。然而，这种信息丢失是有代价的：它可能会由于丢失信息而导致准确性降低。首先，我们探索了多种方法来创建抽象，使用训练数据集，针对数字数据和二进制分类任务的特定情况。我们还通过几个实验测试了这些抽象如何影响对噪声的稳健性，这些实验探索了人工神经网络在使用原始数据训练时对噪声的鲁棒性。结果清楚地表明，使用抽象是开发抗噪声最大似然方法的一种可行方法。



## **22. AdaNCA: Neural Cellular Automata As Adaptors For More Robust Vision Transformer**

AdaNCA：神经元胞自动机作为更稳健的视觉Transformer的适配器 cs.CV

26 pages, 11 figures

**SubmitDate**: 2024-06-12    [abs](http://arxiv.org/abs/2406.08298v1) [paper-pdf](http://arxiv.org/pdf/2406.08298v1)

**Authors**: Yitao Xu, Tong Zhang, Sabine Süsstrunk

**Abstract**: Vision Transformers (ViTs) have demonstrated remarkable performance in image classification tasks, particularly when equipped with local information via region attention or convolutions. While such architectures improve the feature aggregation from different granularities, they often fail to contribute to the robustness of the networks. Neural Cellular Automata (NCA) enables the modeling of global cell representations through local interactions, with its training strategies and architecture design conferring strong generalization ability and robustness against noisy inputs. In this paper, we propose Adaptor Neural Cellular Automata (AdaNCA) for Vision Transformer that uses NCA as plug-in-play adaptors between ViT layers, enhancing ViT's performance and robustness against adversarial samples as well as out-of-distribution inputs. To overcome the large computational overhead of standard NCAs, we propose Dynamic Interaction for more efficient interaction learning. Furthermore, we develop an algorithm for identifying the most effective insertion points for AdaNCA based on our analysis of AdaNCA placement and robustness improvement. With less than a 3% increase in parameters, AdaNCA contributes to more than 10% absolute improvement in accuracy under adversarial attacks on the ImageNet1K benchmark. Moreover, we demonstrate with extensive evaluations across 8 robustness benchmarks and 4 ViT architectures that AdaNCA, as a plug-in-play module, consistently improves the robustness of ViTs.

摘要: 视觉变形器(VITS)在图像分类任务中表现出了显著的性能，特别是当通过区域注意或卷积来配备局部信息时。虽然这样的体系结构从不同的粒度改善了特征聚合，但它们往往无法提高网络的健壮性。神经元胞自动机(NCA)能够通过局部交互对全局细胞表示进行建模，其训练策略和结构设计具有很强的泛化能力和对噪声输入的鲁棒性。在本文中，我们提出了用于视觉转换器的适配器神经元胞自动机(AdaNCA)，它使用NCA作为VIT层之间的即插即用适配器，增强了VIT的性能和对敌意样本和分布外输入的鲁棒性。为了克服标准NCA计算开销大的缺点，我们提出了动态交互来实现更有效的交互学习。此外，基于对AdaNCA布局和健壮性改进的分析，我们提出了一种识别AdaNCA最有效插入点的算法。在参数增加不到3%的情况下，AdaNCA有助于在对ImageNet1K基准的敌意攻击下将准确率绝对提高10%以上。此外，我们通过对8个健壮性基准和4个VIT体系结构的广泛评估，证明了AdaNCA作为一个即插即用模块，持续提高了VIT的健壮性。



## **23. Erasing Radio Frequency Fingerprints via Active Adversarial Perturbation**

通过主动对抗扰动擦除射频指纹 cs.CR

**SubmitDate**: 2024-06-12    [abs](http://arxiv.org/abs/2406.07349v2) [paper-pdf](http://arxiv.org/pdf/2406.07349v2)

**Authors**: Zhaoyi Lu, Wenchao Xu, Ming Tu, Xin Xie, Cunqing Hua, Nan Cheng

**Abstract**: Radio Frequency (RF) fingerprinting is to identify a wireless device from its uniqueness of the analog circuitry or hardware imperfections. However, unlike the MAC address which can be modified, such hardware feature is inevitable for the signal emitted to air, which can possibly reveal device whereabouts, e.g., a sniffer can use a pre-trained model to identify a nearby device when receiving its signal. Such fingerprint may expose critical private information, e.g., the associated upper-layer applications or the end-user. In this paper, we propose to erase such RF feature for wireless devices, which can prevent fingerprinting by actively perturbation from the signal perspective. Specifically, we consider a common RF fingerprinting scenario, where machine learning models are trained from pilot signal data for identification. A novel adversarial attack solution is designed to generate proper perturbations, whereby the perturbed pilot signal can hide the hardware feature and misclassify the model. We theoretically show that the perturbation would not affect the communication function within a tolerable perturbation threshold. We also implement the pilot signal fingerprinting and the proposed perturbation process in a practical LTE system. Extensive experiment results demonstrate that the RF fingerprints can be effectively erased to protect the user privacy.

摘要: 射频(RF)指纹识别是通过模拟电路或硬件缺陷的独特性来识别无线设备。然而，与可以修改的MAC地址不同，这种硬件特征对于发射到空中的信号是不可避免的，这可能会揭示设备的下落，例如，嗅探器可以在接收到其信号时使用预先训练的模型来识别附近的设备。这样的指纹可能暴露关键的私有信息，例如相关联的上层应用程序或终端用户。在本文中，我们建议消除无线设备的这种射频特征，从信号的角度出发，可以通过主动扰动来防止指纹识别。具体地说，我们考虑了一种常见的射频指纹识别场景，其中机器学习模型从导频信号数据中训练用于识别。设计了一种新的对抗性攻击解决方案来产生适当的扰动，由此扰动的导频信号可以隐藏硬件特征并错误地分类模型。我们从理论上证明了在可容忍的微扰阈值内，微扰不会影响通信函数。我们还在一个实际的LTE系统中实现了导频信号指纹和所提出的扰动过程。大量的实验结果表明，该算法可以有效地去除射频指纹，保护用户隐私。



## **24. Adversarial Patch for 3D Local Feature Extractor**

3D局部特征提取器的对抗补丁 cs.CV

**SubmitDate**: 2024-06-12    [abs](http://arxiv.org/abs/2406.08102v1) [paper-pdf](http://arxiv.org/pdf/2406.08102v1)

**Authors**: Yu Wen Pao, Li Chang Lai, Hong-Yi Lin

**Abstract**: Local feature extractors are the cornerstone of many computer vision tasks. However, their vulnerability to adversarial attacks can significantly compromise their effectiveness. This paper discusses approaches to attack sophisticated local feature extraction algorithms and models to achieve two distinct goals: (1) forcing a match between originally non-matching image regions, and (2) preventing a match between originally matching regions. At the end of the paper, we discuss the performance and drawbacks of different patch generation methods.

摘要: 局部特征提取器是许多计算机视觉任务的基石。然而，它们对对抗攻击的脆弱性可能会严重损害它们的有效性。本文讨论了攻击复杂的局部特征提取算法和模型的方法，以实现两个不同的目标：（1）强制原始不匹配的图像区域之间进行匹配，以及（2）防止原始匹配的区域之间进行匹配。在论文的最后，我们讨论了不同补丁生成方法的性能和缺点。



## **25. Adversarial Evasion Attack Efficiency against Large Language Models**

针对大型语言模型的对抗规避攻击效率 cs.CL

9 pages, 1 table, 2 figures, DCAI 2024 conference

**SubmitDate**: 2024-06-12    [abs](http://arxiv.org/abs/2406.08050v1) [paper-pdf](http://arxiv.org/pdf/2406.08050v1)

**Authors**: João Vitorino, Eva Maia, Isabel Praça

**Abstract**: Large Language Models (LLMs) are valuable for text classification, but their vulnerabilities must not be disregarded. They lack robustness against adversarial examples, so it is pertinent to understand the impacts of different types of perturbations, and assess if those attacks could be replicated by common users with a small amount of perturbations and a small number of queries to a deployed LLM. This work presents an analysis of the effectiveness, efficiency, and practicality of three different types of adversarial attacks against five different LLMs in a sentiment classification task. The obtained results demonstrated the very distinct impacts of the word-level and character-level attacks. The word attacks were more effective, but the character and more constrained attacks were more practical and required a reduced number of perturbations and queries. These differences need to be considered during the development of adversarial defense strategies to train more robust LLMs for intelligent text classification applications.

摘要: 大型语言模型(LLM)对于文本分类很有价值，但其脆弱性不容忽视。它们对敌意示例缺乏健壮性，因此了解不同类型扰动的影响并评估这些攻击是否可以被普通用户复制，只需少量扰动和对已部署的LLM的少量查询。本文分析了三种不同类型的对抗性攻击在情感分类任务中对五种不同的LLM的有效性、效率和实用性。所获得的结果显示了词级攻击和字级攻击的非常明显的影响。单词攻击更有效，但字符和更受约束的攻击更实用，需要的干扰和查询次数更少。在开发对抗性防御策略以训练更健壮的LLM用于智能文本分类应用时，需要考虑这些差异。



## **26. ADBA:Approximation Decision Boundary Approach for Black-Box Adversarial Attacks**

ADBA：黑匣子对抗攻击的逼近决策边界方法 cs.LG

10 pages, 5 figures, conference

**SubmitDate**: 2024-06-12    [abs](http://arxiv.org/abs/2406.04998v2) [paper-pdf](http://arxiv.org/pdf/2406.04998v2)

**Authors**: Feiyang Wang, Xingquan Zuo, Hai Huang, Gang Chen

**Abstract**: Many machine learning models are susceptible to adversarial attacks, with decision-based black-box attacks representing the most critical threat in real-world applications. These attacks are extremely stealthy, generating adversarial examples using hard labels obtained from the target machine learning model. This is typically realized by optimizing perturbation directions, guided by decision boundaries identified through query-intensive exact search, significantly limiting the attack success rate. This paper introduces a novel approach using the Approximation Decision Boundary (ADB) to efficiently and accurately compare perturbation directions without precisely determining decision boundaries. The effectiveness of our ADB approach (ADBA) hinges on promptly identifying suitable ADB, ensuring reliable differentiation of all perturbation directions. For this purpose, we analyze the probability distribution of decision boundaries, confirming that using the distribution's median value as ADB can effectively distinguish different perturbation directions, giving rise to the development of the ADBA-md algorithm. ADBA-md only requires four queries on average to differentiate any pair of perturbation directions, which is highly query-efficient. Extensive experiments on six well-known image classifiers clearly demonstrate the superiority of ADBA and ADBA-md over multiple state-of-the-art black-box attacks. The source code is available at https://github.com/BUPTAIOC/ADBA.

摘要: 许多机器学习模型容易受到对抗性攻击，其中基于决策的黑盒攻击是现实世界应用程序中最关键的威胁。这些攻击非常隐蔽，使用从目标机器学习模型获得的硬标签生成敌意示例。这通常是通过优化扰动方向来实现的，由通过查询密集型精确搜索识别的决策边界来指导，从而显著限制攻击成功率。本文提出了一种新的方法，利用近似决策边界(ADB)来高效、准确地比较扰动方向，而无需精确地确定决策边界。我们的ADB方法(ADBA)的有效性取决于迅速找到合适的ADB，确保可靠地区分所有扰动方向。为此，我们分析了决策边界的概率分布，证实了用该分布的中值作为ADB可以有效地区分不同的扰动方向，从而导致了ADBA-MD算法的发展。ADBA-MD平均只需要4个查询就可以区分任意一对扰动方向，查询效率很高。在六个著名的图像分类器上的广泛实验清楚地证明了ADBA和ADBA-MD相对于多种最先进的黑盒攻击的优越性。源代码可在https://github.com/BUPTAIOC/ADBA.上找到



## **27. Towards Reliable Empirical Machine Unlearning Evaluation: A Game-Theoretic View**

迈向可靠的经验机器无学习评估：游戏理论的观点 cs.LG

**SubmitDate**: 2024-06-12    [abs](http://arxiv.org/abs/2404.11577v2) [paper-pdf](http://arxiv.org/pdf/2404.11577v2)

**Authors**: Yiwen Tu, Pingbang Hu, Jiaqi Ma

**Abstract**: Machine unlearning is the process of updating machine learning models to remove the information of specific training data samples, in order to comply with data protection regulations that allow individuals to request the removal of their personal data. Despite the recent development of numerous unlearning algorithms, reliable evaluation of these algorithms remains an open research question. In this work, we focus on membership inference attack (MIA) based evaluation, one of the most common approaches for evaluating unlearning algorithms, and address various pitfalls of existing evaluation metrics that lack reliability. Specifically, we propose a game-theoretic framework that formalizes the evaluation process as a game between unlearning algorithms and MIA adversaries, measuring the data removal efficacy of unlearning algorithms by the capability of the MIA adversaries. Through careful design of the game, we demonstrate that the natural evaluation metric induced from the game enjoys provable guarantees that the existing evaluation metrics fail to satisfy. Furthermore, we propose a practical and efficient algorithm to estimate the evaluation metric induced from the game, and demonstrate its effectiveness through both theoretical analysis and empirical experiments. This work presents a novel and reliable approach to empirically evaluating unlearning algorithms, paving the way for the development of more effective unlearning techniques.

摘要: 机器遗忘是更新机器学习模型以删除特定训练数据样本的信息的过程，以遵守允许个人请求删除其个人数据的数据保护法规。尽管最近有许多遗忘算法的发展，但对这些算法的可靠评估仍然是一个开放的研究问题。在这项工作中，我们专注于基于成员关系推理攻击(MIA)的评估，这是评估遗忘算法最常见的方法之一，并解决了现有评估指标缺乏可靠性的各种缺陷。具体地说，我们提出了一个博弈论框架，将评估过程形式化为遗忘算法与MIA对手之间的博弈，通过MIA对手的能力来衡量遗忘算法的数据去除效率。通过对游戏的精心设计，我们证明了由游戏产生的自然评价指标享有现有评价指标不能满足的可证明保证。在此基础上，提出了一种实用高效的评估指标估计算法，并通过理论分析和实验验证了该算法的有效性。这项工作提供了一种新颖而可靠的方法来对遗忘算法进行经验评估，为开发更有效的遗忘技术铺平了道路。



## **28. Graph Transductive Defense: a Two-Stage Defense for Graph Membership Inference Attacks**

图转化防御：针对图成员推断攻击的两阶段防御 cs.LG

**SubmitDate**: 2024-06-12    [abs](http://arxiv.org/abs/2406.07917v1) [paper-pdf](http://arxiv.org/pdf/2406.07917v1)

**Authors**: Peizhi Niu, Chao Pan, Siheng Chen, Olgica Milenkovic

**Abstract**: Graph neural networks (GNNs) have become instrumental in diverse real-world applications, offering powerful graph learning capabilities for tasks such as social networks and medical data analysis. Despite their successes, GNNs are vulnerable to adversarial attacks, including membership inference attacks (MIA), which threaten privacy by identifying whether a record was part of the model's training data. While existing research has explored MIA in GNNs under graph inductive learning settings, the more common and challenging graph transductive learning setting remains understudied in this context. This paper addresses this gap and proposes an effective two-stage defense, Graph Transductive Defense (GTD), tailored to graph transductive learning characteristics. The gist of our approach is a combination of a train-test alternate training schedule and flattening strategy, which successfully reduces the difference between the training and testing loss distributions. Extensive empirical results demonstrate the superior performance of our method (a decrease in attack AUROC by $9.42\%$ and an increase in utility performance by $18.08\%$ on average compared to LBP), highlighting its potential for seamless integration into various classification models with minimal overhead.

摘要: 图形神经网络(GNN)已经成为各种现实世界应用的工具，为社会网络和医疗数据分析等任务提供了强大的图形学习能力。尽管GNN取得了成功，但它们很容易受到包括成员身份推断攻击(MIA)在内的对抗性攻击，这些攻击通过识别记录是否属于模型训练数据的一部分来威胁隐私。虽然现有的研究已经探索了图形归纳学习环境下GNN中的MIA，但在这种背景下，更常见和更具挑战性的图形转导学习环境仍然没有得到充分的研究。本文针对这一差距，提出了一种有效的两阶段防御机制--图形传导防御(GTD)，该防御机制专为绘制传导学习特征而定制。我们的方法的要点是训练-测试交替训练计划和扁平化策略的结合，成功地减少了训练和测试损失分布之间的差异。大量的实验结果表明，我们的方法具有优越的性能(与LBP相比，攻击AUROC平均减少了9.42美元，效用性能平均提高了18.08美元)，突出了它以最小的开销无缝集成到各种分类模型中的潜力。



## **29. Mitigation of Channel Tampering Attacks in Continuous-Variable Quantum Key Distribution**

缓解连续可变量子密钥分发中的通道篡改攻击 quant-ph

10 pages, 7 figures, closest to accepted version

**SubmitDate**: 2024-06-12    [abs](http://arxiv.org/abs/2401.15898v2) [paper-pdf](http://arxiv.org/pdf/2401.15898v2)

**Authors**: Sebastian P. Kish, Chandra Thapa, Mikhael Sayat, Hajime Suzuki, Josef Pieprzyk, Seyit Camtepe

**Abstract**: Despite significant advancements in continuous-variable quantum key distribution (CV-QKD), practical CV-QKD systems can be compromised by various attacks. Consequently, identifying new attack vectors and countermeasures for CV-QKD implementations is important for the continued robustness of CV-QKD. In particular, as CV-QKD relies on a public quantum channel, vulnerability to communication disruption persists from potential adversaries employing Denial-of-Service (DoS) attacks. Inspired by DoS attacks, this paper introduces a novel threat in CV-QKD called the Channel Amplification (CA) attack, wherein Eve manipulates the communication channel through amplification. We specifically model this attack in a CV-QKD optical fiber setup. To counter this threat, we propose a detection and mitigation strategy. Detection involves a machine learning (ML) model based on a decision tree classifier, classifying various channel tampering attacks, including CA and DoS attacks. For mitigation, Bob, post-selects quadrature data by classifying the attack type and frequency. Our ML model exhibits high accuracy in distinguishing and categorizing these attacks. The CA attack's impact on the secret key rate (SKR) is explored concerning Eve's location and the relative intensity noise of the local oscillator (LO). The proposed mitigation strategy improves the attacked SKR for CA attacks and, in some cases, for hybrid CA-DoS attacks. Our study marks a novel application of both ML classification and post-selection in this context. These findings are important for enhancing the robustness of CV-QKD systems against emerging threats on the channel.

摘要: 尽管连续变量量子密钥分发(CV-QKD)有了很大的进步，但实用的CV-QKD系统可能会受到各种攻击。因此，为CV-QKD的实现识别新的攻击载体和对策对于CV-QKD的持续健壮性是重要的。特别是，由于CV-QKD依赖于公共量子信道，因此使用拒绝服务(DoS)攻击的潜在对手对通信中断的脆弱性持续存在。受DoS攻击的启发，本文在CV-QKD中引入了一种新的威胁，称为通道放大(CA)攻击，即Eve通过放大来操纵通信通道。我们专门在CV-QKD光纤设置中对这种攻击进行了建模。为了应对这种威胁，我们提出了一种检测和缓解策略。检测涉及基于决策树分类器的机器学习(ML)模型，对各种通道篡改攻击进行分类，包括CA和DoS攻击。对于缓解，Bob通过对攻击类型和频率进行分类来选择正交数据。我们的ML模型在区分和分类这些攻击方面表现出了很高的准确性。从Eve的位置和本地振荡器(LO)的相对强度噪声两个方面探讨了CA攻击对密钥速率(SKR)的影响。提出的缓解策略提高了CA攻击的受攻击SKR，在某些情况下，还提高了混合CA-DoS攻击的SKR。我们的研究标志着ML分类和后选择在这一背景下的新应用。这些发现对于增强CV-QKD系统对信道上新出现的威胁的稳健性具有重要意义。



## **30. Et Tu Certifications: Robustness Certificates Yield Better Adversarial Examples**

Et Tu认证：稳健性证书产生更好的对抗性示例 cs.LG

17 pages, 8 figures

**SubmitDate**: 2024-06-12    [abs](http://arxiv.org/abs/2302.04379v4) [paper-pdf](http://arxiv.org/pdf/2302.04379v4)

**Authors**: Andrew C. Cullen, Shijie Liu, Paul Montague, Sarah M. Erfani, Benjamin I. P. Rubinstein

**Abstract**: In guaranteeing the absence of adversarial examples in an instance's neighbourhood, certification mechanisms play an important role in demonstrating neural net robustness. In this paper, we ask if these certifications can compromise the very models they help to protect? Our new \emph{Certification Aware Attack} exploits certifications to produce computationally efficient norm-minimising adversarial examples $74 \%$ more often than comparable attacks, while reducing the median perturbation norm by more than $10\%$. While these attacks can be used to assess the tightness of certification bounds, they also highlight that releasing certifications can paradoxically reduce security.

摘要: 为了保证实例附近没有对抗示例，认证机制在证明神经网络稳健性方面发挥着重要作用。在本文中，我们询问这些认证是否会损害它们帮助保护的模型？我们的新\{Certification Aware Attack}利用认证来生成计算高效的规范最小化对抗示例，比可比攻击的频率高出74美元，同时将中位数扰动规范降低超过10美元。虽然这些攻击可用于评估认证界限的严格程度，但它们也强调了发布认证可能会自相矛盾地降低安全性。



## **31. Defending Against Alignment-Breaking Attacks via Robustly Aligned LLM**

通过强大的对齐LLM防御破坏对齐的攻击 cs.CL

19 Pages, 5 Figures, 8 Tables. Accepted by ACL 2024

**SubmitDate**: 2024-06-12    [abs](http://arxiv.org/abs/2309.14348v3) [paper-pdf](http://arxiv.org/pdf/2309.14348v3)

**Authors**: Bochuan Cao, Yuanpu Cao, Lu Lin, Jinghui Chen

**Abstract**: Recently, Large Language Models (LLMs) have made significant advancements and are now widely used across various domains. Unfortunately, there has been a rising concern that LLMs can be misused to generate harmful or malicious content. Though a line of research has focused on aligning LLMs with human values and preventing them from producing inappropriate content, such alignments are usually vulnerable and can be bypassed by alignment-breaking attacks via adversarially optimized or handcrafted jailbreaking prompts. In this work, we introduce a Robustly Aligned LLM (RA-LLM) to defend against potential alignment-breaking attacks. RA-LLM can be directly constructed upon an existing aligned LLM with a robust alignment checking function, without requiring any expensive retraining or fine-tuning process of the original LLM. Furthermore, we also provide a theoretical analysis for RA-LLM to verify its effectiveness in defending against alignment-breaking attacks. Through real-world experiments on open-source large language models, we demonstrate that RA-LLM can successfully defend against both state-of-the-art adversarial prompts and popular handcrafted jailbreaking prompts by reducing their attack success rates from nearly 100% to around 10% or less.

摘要: 近年来，大型语言模型(LLM)取得了长足的进步，现已广泛应用于各个领域。不幸的是，人们越来越担心LLMS可能被滥用来生成有害或恶意的内容。尽管有一系列研究专注于将LLM与人类价值观保持一致，并防止它们产生不适当的内容，但这种调整通常是脆弱的，可以通过恶意优化或手工制作的越狱提示被破坏顺序的攻击绕过。在这项工作中，我们引入了一种鲁棒对齐LLM(RA-LLM)来防御潜在的对齐破坏攻击。RA-LLM可以直接构建在现有的对准LLM上，具有健壮的对准检查功能，而不需要对原始LLM进行任何昂贵的再培训或微调过程。此外，我们还对RA-LLM进行了理论分析，以验证其在抵抗对齐破坏攻击方面的有效性。通过在开源大型语言模型上的真实世界实验，我们证明了RA-LLM能够成功地防御最新的敌意提示和流行的手工越狱提示，将攻击成功率从近100%降低到10%左右或更低。



## **32. Adversarial Machine Unlearning**

对抗性机器遗忘 cs.LG

**SubmitDate**: 2024-06-11    [abs](http://arxiv.org/abs/2406.07687v1) [paper-pdf](http://arxiv.org/pdf/2406.07687v1)

**Authors**: Zonglin Di, Sixie Yu, Yevgeniy Vorobeychik, Yang Liu

**Abstract**: This paper focuses on the challenge of machine unlearning, aiming to remove the influence of specific training data on machine learning models. Traditionally, the development of unlearning algorithms runs parallel with that of membership inference attacks (MIA), a type of privacy threat to determine whether a data instance was used for training. However, the two strands are intimately connected: one can view machine unlearning through the lens of MIA success with respect to removed data. Recognizing this connection, we propose a game-theoretic framework that integrates MIAs into the design of unlearning algorithms. Specifically, we model the unlearning problem as a Stackelberg game in which an unlearner strives to unlearn specific training data from a model, while an auditor employs MIAs to detect the traces of the ostensibly removed data. Adopting this adversarial perspective allows the utilization of new attack advancements, facilitating the design of unlearning algorithms. Our framework stands out in two ways. First, it takes an adversarial approach and proactively incorporates the attacks into the design of unlearning algorithms. Secondly, it uses implicit differentiation to obtain the gradients that limit the attacker's success, thus benefiting the process of unlearning. We present empirical results to demonstrate the effectiveness of the proposed approach for machine unlearning.

摘要: 本文着重研究机器遗忘的挑战，旨在消除特定训练数据对机器学习模型的影响。传统上，遗忘算法的开发与成员关系推理攻击(MIA)的开发并行，MIA是一种隐私威胁，用于确定数据实例是否用于训练。然而，这两条线索是紧密相连的：人们可以通过MIA关于移除数据的成功镜头来查看机器遗忘。认识到这种联系，我们提出了一个博弈论框架，将MIA整合到遗忘算法的设计中。具体地说，我们将遗忘问题建模为Stackelberg博弈，在该博弈中，非学习者努力从模型中忘记特定的训练数据，而审计师则使用MIA来检测表面上被移除的数据的痕迹。采用这种对抗性的观点允许利用新的攻击进步，促进遗忘算法的设计。我们的框架在两方面脱颖而出。首先，它采取对抗性的方法，并主动将攻击纳入遗忘算法的设计中。其次，它使用隐式微分来获得限制攻击者成功的梯度，从而有利于遗忘过程。我们给出的实验结果证明了所提出的机器遗忘方法的有效性。



## **33. SmoothLLM: Defending Large Language Models Against Jailbreaking Attacks**

SmothLLM：保护大型语言模型免受越狱攻击 cs.LG

**SubmitDate**: 2024-06-11    [abs](http://arxiv.org/abs/2310.03684v4) [paper-pdf](http://arxiv.org/pdf/2310.03684v4)

**Authors**: Alexander Robey, Eric Wong, Hamed Hassani, George J. Pappas

**Abstract**: Despite efforts to align large language models (LLMs) with human intentions, widely-used LLMs such as GPT, Llama, and Claude are susceptible to jailbreaking attacks, wherein an adversary fools a targeted LLM into generating objectionable content. To address this vulnerability, we propose SmoothLLM, the first algorithm designed to mitigate jailbreaking attacks. Based on our finding that adversarially-generated prompts are brittle to character-level changes, our defense randomly perturbs multiple copies of a given input prompt, and then aggregates the corresponding predictions to detect adversarial inputs. Across a range of popular LLMs, SmoothLLM sets the state-of-the-art for robustness against the GCG, PAIR, RandomSearch, and AmpleGCG jailbreaks. SmoothLLM is also resistant against adaptive GCG attacks, exhibits a small, though non-negligible trade-off between robustness and nominal performance, and is compatible with any LLM. Our code is publicly available at \url{https://github.com/arobey1/smooth-llm}.

摘要: 尽管努力使大型语言模型(LLM)与人的意图保持一致，但GPT、Llama和Claude等广泛使用的LLM容易受到越狱攻击，即对手欺骗目标LLM生成令人反感的内容。为了解决这一漏洞，我们提出了SmoothLLM，这是第一个旨在缓解越狱攻击的算法。基于我们的发现，对抗性生成的提示对字符级别的变化很脆弱，我们的防御随机扰动给定输入提示的多个副本，然后聚合相应的预测来检测对抗性输入。在一系列流行的LLM中，SmoothLLM针对GCG、Pair、RandomSearch和AmpleGCG越狱设置了最先进的健壮性。SmoothLLM还抵抗自适应GCG攻击，在稳健性和标称性能之间表现出一种虽小但不可忽略的折衷，并与任何LLM兼容。我们的代码在\url{https://github.com/arobey1/smooth-llm}.}上公开提供



## **34. Beware of Aliases -- Signal Preservation is Crucial for Robust Image Restoration**

小心别名--信号保留对于稳健的图像恢复至关重要 cs.CV

Tags: Adversarial attack, image restoration, image deblurring,  frequency sampling

**SubmitDate**: 2024-06-11    [abs](http://arxiv.org/abs/2406.07435v1) [paper-pdf](http://arxiv.org/pdf/2406.07435v1)

**Authors**: Shashank Agnihotri, Julia Grabinski, Janis Keuper, Margret Keuper

**Abstract**: Image restoration networks are usually comprised of an encoder and a decoder, responsible for aggregating image content from noisy, distorted data and to restore clean, undistorted images, respectively. Data aggregation as well as high-resolution image generation both usually come at the risk of involving aliases, i.e.~standard architectures put their ability to reconstruct the model input in jeopardy to reach high PSNR values on validation data. The price to be paid is low model robustness. In this work, we show that simply providing alias-free paths in state-of-the-art reconstruction transformers supports improved model robustness at low costs on the restoration performance. We do so by proposing BOA-Restormer, a transformer-based image restoration model that executes downsampling and upsampling operations partly in the frequency domain to ensure alias-free paths along the entire model while potentially preserving all relevant high-frequency information.

摘要: 图像恢复网络通常由编码器和解码器组成，分别负责从有噪、失真的数据中聚合图像内容并恢复干净、无失真的图像。数据聚合以及高分辨率图像生成通常都存在涉及别名的风险，即~标准体系结构使其重建模型输入以在验证数据上达到高PSNR值的能力处于危险之中。要付出的代价是模型稳健性较低。在这项工作中，我们表明，简单地在最先进的重建变换器中提供无别名路径就可以支持以低成本提高恢复性能的模型鲁棒性。我们通过提出BOA-Restormer来做到这一点，这是一种基于变压器的图像恢复模型，它部分在频域中执行下采样和上采样操作，以确保整个模型中的无别名路径，同时可能保留所有相关的高频信息。



## **35. Text-CRS: A Generalized Certified Robustness Framework against Textual Adversarial Attacks**

文本-CRS：针对文本对抗攻击的通用认证鲁棒性框架 cs.CR

Published in the 2024 IEEE Symposium on Security and Privacy (SP)

**SubmitDate**: 2024-06-11    [abs](http://arxiv.org/abs/2307.16630v2) [paper-pdf](http://arxiv.org/pdf/2307.16630v2)

**Authors**: Xinyu Zhang, Hanbin Hong, Yuan Hong, Peng Huang, Binghui Wang, Zhongjie Ba, Kui Ren

**Abstract**: The language models, especially the basic text classification models, have been shown to be susceptible to textual adversarial attacks such as synonym substitution and word insertion attacks. To defend against such attacks, a growing body of research has been devoted to improving the model robustness. However, providing provable robustness guarantees instead of empirical robustness is still widely unexplored. In this paper, we propose Text-CRS, a generalized certified robustness framework for natural language processing (NLP) based on randomized smoothing. To our best knowledge, existing certified schemes for NLP can only certify the robustness against $\ell_0$ perturbations in synonym substitution attacks. Representing each word-level adversarial operation (i.e., synonym substitution, word reordering, insertion, and deletion) as a combination of permutation and embedding transformation, we propose novel smoothing theorems to derive robustness bounds in both permutation and embedding space against such adversarial operations. To further improve certified accuracy and radius, we consider the numerical relationships between discrete words and select proper noise distributions for the randomized smoothing. Finally, we conduct substantial experiments on multiple language models and datasets. Text-CRS can address all four different word-level adversarial operations and achieve a significant accuracy improvement. We also provide the first benchmark on certified accuracy and radius of four word-level operations, besides outperforming the state-of-the-art certification against synonym substitution attacks.

摘要: 语言模型，特别是基本文本分类模型，已经被证明容易受到文本对抗性攻击，如同义词替换和单词插入攻击。为了防御这种攻击，越来越多的研究致力于提高模型的稳健性。然而，提供可证明的稳健性保证而不是经验稳健性仍然是广泛未被探索的。本文提出了一种基于随机化平滑的自然语言处理(NLP)广义认证健壮性框架Text-CRS。据我们所知，现有的NLP认证方案只能证明对同义词替换攻击中的$\ell_0$扰动的健壮性。将词级的敌意操作(即同义词替换、单词重排、插入和删除)表示为置换和嵌入变换的组合，提出了新颖的平滑定理，从而在置换空间和嵌入空间中推导出对此类敌意操作的稳健界。为了进一步提高认证精度和半径，我们考虑了离散字之间的数值关系，并选择合适的噪声分布进行随机化平滑。最后，我们在多种语言模型和数据集上进行了大量的实验。Text-CRS可以处理所有四种不同的词级对抗性操作，并实现显著的准确率提高。除了在同义词替换攻击方面优于最先进的认证外，我们还提供了首个关于四个词级操作的认证准确度和半径的基准。



## **36. Merging Improves Self-Critique Against Jailbreak Attacks**

合并提高了对越狱袭击的自我批评 cs.CL

**SubmitDate**: 2024-06-11    [abs](http://arxiv.org/abs/2406.07188v1) [paper-pdf](http://arxiv.org/pdf/2406.07188v1)

**Authors**: Victor Gallego

**Abstract**: The robustness of large language models (LLMs) against adversarial manipulations, such as jailbreak attacks, remains a significant challenge. In this work, we propose an approach that enhances the self-critique capability of the LLM and further fine-tunes it over sanitized synthetic data. This is done with the addition of an external critic model that can be merged with the original, thus bolstering self-critique capabilities and improving the robustness of the LLMs response to adversarial prompts. Our results demonstrate that the combination of merging and self-critique can reduce the attack success rate of adversaries significantly, thus offering a promising defense mechanism against jailbreak attacks. Code, data and models released at https://github.com/vicgalle/merging-self-critique-jailbreaks .

摘要: 大型语言模型（LLM）对越狱攻击等对抗性操纵的稳健性仍然是一个重大挑战。在这项工作中，我们提出了一种增强LLM自我批评能力的方法，并根据净化的合成数据进一步对其进行微调。这是通过添加一个可以与原始模型合并的外部批评者模型来实现的，从而增强自我批评能力并提高LLM对对抗提示反应的稳健性。我们的结果表明，合并和自我批评的结合可以显着降低对手的攻击成功率，从而提供一种有希望的针对越狱攻击的防御机制。代码、数据和模型在https://github.com/vicgalle/merging-self-critique-jailbreaks上发布。



## **37. Trainwreck: A damaging adversarial attack on image classifiers**

Trainwreck：对图像分类器的破坏性对抗攻击 cs.CV

**SubmitDate**: 2024-06-11    [abs](http://arxiv.org/abs/2311.14772v2) [paper-pdf](http://arxiv.org/pdf/2311.14772v2)

**Authors**: Jan Zahálka

**Abstract**: Adversarial attacks are an important security concern for computer vision (CV). As CV models are becoming increasingly valuable assets in applied practice, disrupting them is emerging as a form of economic sabotage. This paper opens up the exploration of damaging adversarial attacks (DAAs) that seek to damage target CV models. DAAs are formalized by defining the threat model, the cost function DAAs maximize, and setting three requirements for success: potency, stealth, and customizability. As a pioneer DAA, this paper proposes Trainwreck, a train-time attack that conflates the data of similar classes in the training data using stealthy ($\epsilon \leq 8/255$) class-pair universal perturbations obtained from a surrogate model. Trainwreck is a black-box, transferable attack: it requires no knowledge of the target architecture, and a single poisoned dataset degrades the performance of any model trained on it. The experimental evaluation on CIFAR-10 and CIFAR-100 and various model architectures (EfficientNetV2, ResNeXt-101, and a finetuned ViT-L-16) demonstrates Trainwreck's efficiency. Trainwreck achieves similar or better potency compared to the data poisoning state of the art and is fully customizable by the poison rate parameter. Finally, data redundancy with hashing is identified as a reliable defense against Trainwreck or similar DAAs. The code is available at https://github.com/JanZahalka/trainwreck.

摘要: 对抗性攻击是计算机视觉的一个重要安全问题。随着简历模型在应用实践中变得越来越有价值，颠覆它们正成为一种经济破坏形式。本文对试图破坏目标CV模型的破坏性对抗性攻击(DAA)进行了探索。DAA通过定义威胁模型、最大化DAA的成本函数以及设置三个成功要求来形式化：效力、隐蔽性和可定制化。作为DAA的先驱，本文提出了Trainwreck，一种训练时间攻击，它利用从代理模型获得的隐蔽的($\epsilon 8/255$)类对普遍扰动来合并训练数据中相似类的数据。Trainwreck是一种黑匣子、可转移的攻击：它不需要了解目标体系结构，并且单个有毒数据集会降低任何针对其训练的模型的性能。在CIFAR-10和CIFAR-100以及各种模型体系结构(EfficientNetV2、ResNeXt-101和精调的VIT-L-16)上的实验评估表明了Trainwreck的有效性。与现有技术的数据中毒状态相比，Trainwreck实现了类似或更好的效力，并且完全可以通过毒化率参数进行定制。最后，使用散列的数据冗余被认为是抵御Trainwreck或类似DAA的可靠防御措施。代码可在https://github.com/JanZahalka/trainwreck.上获得



## **38. Benchmarking Trustworthiness of Multimodal Large Language Models: A Comprehensive Study**

多模式大型语言模型的可信度基准：全面研究 cs.CL

100 pages, 84 figures, 33 tables

**SubmitDate**: 2024-06-11    [abs](http://arxiv.org/abs/2406.07057v1) [paper-pdf](http://arxiv.org/pdf/2406.07057v1)

**Authors**: Yichi Zhang, Yao Huang, Yitong Sun, Chang Liu, Zhe Zhao, Zhengwei Fang, Yifan Wang, Huanran Chen, Xiao Yang, Xingxing Wei, Hang Su, Yinpeng Dong, Jun Zhu

**Abstract**: Despite the superior capabilities of Multimodal Large Language Models (MLLMs) across diverse tasks, they still face significant trustworthiness challenges. Yet, current literature on the assessment of trustworthy MLLMs remains limited, lacking a holistic evaluation to offer thorough insights into future improvements. In this work, we establish MultiTrust, the first comprehensive and unified benchmark on the trustworthiness of MLLMs across five primary aspects: truthfulness, safety, robustness, fairness, and privacy. Our benchmark employs a rigorous evaluation strategy that addresses both multimodal risks and cross-modal impacts, encompassing 32 diverse tasks with self-curated datasets. Extensive experiments with 21 modern MLLMs reveal some previously unexplored trustworthiness issues and risks, highlighting the complexities introduced by the multimodality and underscoring the necessity for advanced methodologies to enhance their reliability. For instance, typical proprietary models still struggle with the perception of visually confusing images and are vulnerable to multimodal jailbreaking and adversarial attacks; MLLMs are more inclined to disclose privacy in text and reveal ideological and cultural biases even when paired with irrelevant images in inference, indicating that the multimodality amplifies the internal risks from base LLMs. Additionally, we release a scalable toolbox for standardized trustworthiness research, aiming to facilitate future advancements in this important field. Code and resources are publicly available at: https://multi-trust.github.io/.

摘要: 尽管多模式大型语言模型(MLLM)在不同的任务中具有卓越的能力，但它们仍然面临着重大的可信性挑战。然而，目前关于评估值得信赖的MLLMS的文献仍然有限，缺乏全面的评估来提供对未来改进的透彻见解。在这项工作中，我们建立了多重信任，这是第一个关于MLLMS可信度的全面和统一的基准，涉及五个主要方面：真实性、安全性、健壮性、公平性和隐私性。我们的基准采用了严格的评估战略，同时应对多式联运风险和跨联运影响，包括32项不同的任务和自我管理的数据集。对21个现代多模式管理进行的广泛实验揭示了一些以前从未探索过的可信度问题和风险，突显了多模式带来的复杂性，并强调了先进方法提高其可靠性的必要性。例如，典型的专有模型仍然难以识别视觉上令人困惑的图像，容易受到多模式越狱和敌意攻击；MLLM更倾向于在文本中泄露隐私，甚至在推理中与无关图像搭配使用时也会暴露意识形态和文化偏见，这表明多模式放大了基本LLM的内部风险。此外，我们还发布了一个用于标准化可信度研究的可扩展工具箱，旨在促进这一重要领域的未来发展。代码和资源可在以下网址公开获得：https://multi-trust.github.io/.



## **39. Adversarial flows: A gradient flow characterization of adversarial attacks**

对抗流：对抗攻击的梯度流特征 cs.LG

**SubmitDate**: 2024-06-11    [abs](http://arxiv.org/abs/2406.05376v2) [paper-pdf](http://arxiv.org/pdf/2406.05376v2)

**Authors**: Lukas Weigand, Tim Roith, Martin Burger

**Abstract**: A popular method to perform adversarial attacks on neuronal networks is the so-called fast gradient sign method and its iterative variant. In this paper, we interpret this method as an explicit Euler discretization of a differential inclusion, where we also show convergence of the discretization to the associated gradient flow. To do so, we consider the concept of p-curves of maximal slope in the case $p=\infty$. We prove existence of $\infty$-curves of maximum slope and derive an alternative characterization via differential inclusions. Furthermore, we also consider Wasserstein gradient flows for potential energies, where we show that curves in the Wasserstein space can be characterized by a representing measure on the space of curves in the underlying Banach space, which fulfill the differential inclusion. The application of our theory to the finite-dimensional setting is twofold: On the one hand, we show that a whole class of normalized gradient descent methods (in particular signed gradient descent) converge, up to subsequences, to the flow, when sending the step size to zero. On the other hand, in the distributional setting, we show that the inner optimization task of adversarial training objective can be characterized via $\infty$-curves of maximum slope on an appropriate optimal transport space.

摘要: 对神经元网络进行敌意攻击的一种流行方法是所谓的快速梯度符号方法及其迭代变体。在本文中，我们将该方法解释为微分包含的显式Euler离散化，并证明了该离散化收敛于相应的梯度流。为此，我们考虑了最大斜率的p-曲线的概念。我们证明了最大斜率的$-曲线的存在性，并通过微分包含得到了另一种刻画。此外，我们还考虑了势能的Wasserstein梯度流，其中我们证明了Wasserstein空间中的曲线可以用基本Banach空间中的曲线空间上的表示测度来刻画，从而满足微分包含.我们的理论在有限维环境中的应用有两个方面：一方面，我们证明了当步长为零时，一整类归一化梯度下降方法(特别是符号梯度下降方法)收敛到流，至上子序列。另一方面，在分布环境下，我们证明了对抗性训练目标的内部优化任务可以用适当的最优运输空间上的最大斜率曲线来刻画。



## **40. Post-train Black-box Defense via Bayesian Boundary Correction**

通过Bayesian边界修正进行训练后黑匣子防御 cs.CV

arXiv admin note: text overlap with arXiv:2203.04713

**SubmitDate**: 2024-06-11    [abs](http://arxiv.org/abs/2306.16979v3) [paper-pdf](http://arxiv.org/pdf/2306.16979v3)

**Authors**: He Wang, Yunfeng Diao

**Abstract**: Classifiers based on deep neural networks are susceptible to adversarial attack, where the widely existing vulnerability has invoked the research in defending them from potential threats. Given a vulnerable classifier, existing defense methods are mostly white-box and often require re-training the victim under modified loss functions/training regimes. While the model/data/training specifics of the victim are usually unavailable to the user, re-training is unappealing, if not impossible for reasons such as limited computational resources. To this end, we propose a new post-train black-box defense framework. It can turn any pre-trained classifier into a resilient one with little knowledge of the model specifics. This is achieved by new joint Bayesian treatments on the clean data, the adversarial examples and the classifier, for maximizing their joint probability. It is further equipped with a new post-train strategy which keeps the victim intact, avoiding re-training. We name our framework Bayesian Boundary Correction (BBC). BBC is a general and flexible framework that can easily adapt to different data types. We instantiate BBC for image classification and skeleton-based human activity recognition, for both static and dynamic data. Exhaustive evaluation shows that BBC has superior robustness and can enhance robustness without severely hurting the clean accuracy, compared with existing defense methods.

摘要: 基于深度神经网络的分类器容易受到敌意攻击，其中广泛存在的漏洞引发了保护它们免受潜在威胁的研究。在给定一个易受攻击的分类器的情况下，现有的防御方法大多是白盒的，并且经常需要根据修改的损失函数/训练制度重新训练受害者。虽然受害者的模型/数据/培训细节通常对用户不可用，但重新培训是没有吸引力的，如果不是因为有限的计算资源等原因不可能的话。为此，我们提出了一种新的训练后黑匣子防御框架。它可以将任何预先训练的分类器变成一个有弹性的分类器，而对模型细节知之甚少。这是通过对干净数据、对抗性样本和分类器进行新的联合贝叶斯处理来实现的，以最大化它们的联合概率。它还配备了新的训练后策略，使受害者保持完好，避免重新训练。我们将我们的框架命名为贝叶斯边界校正(BBC)。BBC是一个通用和灵活的框架，可以很容易地适应不同的数据类型。对于静态和动态数据，我们实例化了用于图像分类和基于骨骼的人体活动识别的BBC。详尽的评估表明，与现有的防御方法相比，BBC具有更好的稳健性，可以在不严重损害干净精度的情况下增强稳健性。



## **41. DISCO Might Not Be Funky: Random Intelligent Reflective Surface Configurations That Attack**

DISCO可能并不时髦：随机智能反射表面按钮攻击 eess.SP

This paper has been accepted by IEEE Wireless Communications. For the  code of the DISCO RIS is available on Github  (https://github.com/huanhuan1799/Disco-Intelligent-Reflecting-Surfaces-Active-Channel-Aging-for-Fully-Passive-Jamming-Attacks)

**SubmitDate**: 2024-06-11    [abs](http://arxiv.org/abs/2310.00687v2) [paper-pdf](http://arxiv.org/pdf/2310.00687v2)

**Authors**: Huan Huang, Lipeng Dai, Hongliang Zhang, Chongfu Zhang, Zhongxing Tian, Yi Cai, A. Lee Swindlehurst, Zhu Han

**Abstract**: Emerging intelligent reflective surfaces (IRSs) significantly improve system performance, but also pose a significant risk for physical layer security (PLS). Unlike the extensive research on legitimate IRS-enhanced communications, in this article we present an adversarial IRS-based fully-passive jammer (FPJ). We describe typical application scenarios for Disco IRS (DIRS)-based FPJ, where an illegitimate IRS with random, time-varying reflection properties acts like a "disco ball" to randomly change the propagation environment. We introduce the principles of DIRS-based FPJ and overview existing investigations of the technology, including a design example employing one-bit phase shifters. The DIRS-based FPJ can be implemented without either jamming power or channel state information (CSI) for the legitimate users (LUs). It does not suffer from the energy constraints of traditional active jammers, nor does it require any knowledge of the LU channels. In addition to the proposed jamming attack, we also propose an anti-jamming strategy that requires only statistical rather than instantaneous CSI. Furthermore, we present a data frame structure that enables the legitimate access point (AP) to estimate the DIRS-jammed channels' statistical characteristics in the presence of the DIRS jamming. Typical cases are discussed to show the impact of the DIRS-based FPJ and the feasibility of the anti-jamming precoder (AJP). Moreover, we outline future research directions and challenges for the DIRS-based FPJ and its anti-jamming precoding to stimulate this line of research and pave the way for practical applications.

摘要: 新兴的智能反射面(IRS)显著提高了系统性能，但也对物理层安全(PLS)构成了重大风险。不同于对合法IRS增强通信的广泛研究，本文提出了一种基于IRS的对抗性全被动干扰机(FPJ)。我们描述了基于Disco IRS(DIRS)的FPJ的典型应用场景，其中具有随机、时变反射属性的非法IRS充当随机改变传播环境的迪斯科球。我们介绍了基于DIRS的FPJ的原理，并综述了该技术的现有研究成果，包括一个使用一位移相器的设计实例。对于合法用户(LU)，可以在没有干扰功率或信道状态信息(CSI)的情况下实现基于DIRS的FPJ。它不受传统有源干扰器的能量限制，也不需要任何LU信道的知识。除了提出的干扰攻击外，我们还提出了一种只需要统计而不需要瞬时CSI的干扰策略。此外，我们提出了一种数据帧结构，使合法接入点(AP)能够在存在DIRS干扰的情况下估计DIRS干扰信道的统计特性。通过典型算例说明了基于DIRS的预编码法的影响和抗扰预编码法的可行性。此外，我们还概述了基于DIRS的FPJ及其抗干扰性预编码的未来研究方向和挑战，以激励这一研究方向，为实际应用铺平道路。



## **42. Robust Distribution Learning with Local and Global Adversarial Corruptions**

具有本地和全球对抗性腐蚀的稳健分布学习 cs.LG

Accepted for presentation at the Conference on Learning Theory (COLT)  2024

**SubmitDate**: 2024-06-10    [abs](http://arxiv.org/abs/2406.06509v1) [paper-pdf](http://arxiv.org/pdf/2406.06509v1)

**Authors**: Sloan Nietert, Ziv Goldfeld, Soroosh Shafiee

**Abstract**: We consider learning in an adversarial environment, where an $\varepsilon$-fraction of samples from a distribution $P$ are arbitrarily modified (*global* corruptions) and the remaining perturbations have average magnitude bounded by $\rho$ (*local* corruptions). Given access to $n$ such corrupted samples, we seek a computationally efficient estimator $\hat{P}_n$ that minimizes the Wasserstein distance $\mathsf{W}_1(\hat{P}_n,P)$. In fact, we attack the fine-grained task of minimizing $\mathsf{W}_1(\Pi_\# \hat{P}_n, \Pi_\# P)$ for all orthogonal projections $\Pi \in \mathbb{R}^{d \times d}$, with performance scaling with $\mathrm{rank}(\Pi) = k$. This allows us to account simultaneously for mean estimation ($k=1$), distribution estimation ($k=d$), as well as the settings interpolating between these two extremes. We characterize the optimal population-limit risk for this task and then develop an efficient finite-sample algorithm with error bounded by $\sqrt{\varepsilon k} + \rho + d^{O(1)}\tilde{O}(n^{-1/k})$ when $P$ has bounded moments of order $2+\delta$, for constant $\delta > 0$. For data distributions with bounded covariance, our finite-sample bounds match the minimax population-level optimum for large sample sizes. Our efficient procedure relies on a novel trace norm approximation of an ideal yet intractable 2-Wasserstein projection estimator. We apply this algorithm to robust stochastic optimization, and, in the process, uncover a new method for overcoming the curse of dimensionality in Wasserstein distributionally robust optimization.

摘要: 我们考虑在对抗性环境中学习，其中来自分布$P$的$\varepsilon$-分数样本被任意修改(*全局*损坏)，而其余扰动的平均幅度由$\rho$(*局部*损坏)限定。在给定$n$这样的破坏样本的情况下，我们寻找一个计算上有效的估计量$\hat{P}_n$以最小化Wasserstein距离$\mathsf{W}_1(\hat{P}_n，P)$。事实上，我们对所有的正交投影$\pI\in\mathbb{R}^{d\time d}$发起了最小化$\mathsf{W}_1(\pI_#\hat{P}_n，\pI_#P)$的细粒度任务，并且性能伸缩为$\mathm{RANK}(\pI)=k$。这允许我们同时考虑平均值估计($k=1$)、分布估计($k=d$)以及在这两个极值之间插入的设置。我们刻画了该任务的最优总体极限风险，并在此基础上发展了一个有效的有限样本算法，当$P$具有$2+β$的有界矩时，误差有界于$Sqrt{varepsilon k}+Rho+d^{O(1)}(n^{-1/k})$。对于协方差有界的数据分布，我们的有限样本界与大样本量的最小最大总体水平最优匹配。我们的有效过程依赖于一个理想但难以处理的2-Wasserstein投影估计量的一个新的迹范数近似。我们将该算法应用于稳健随机优化中，并在此过程中发现了一种克服Wasserstein分布稳健优化中的维度灾难的新方法。



## **43. Improving Alignment and Robustness with Circuit Breakers**

改善断路器的对准和稳健性 cs.LG

**SubmitDate**: 2024-06-10    [abs](http://arxiv.org/abs/2406.04313v2) [paper-pdf](http://arxiv.org/pdf/2406.04313v2)

**Authors**: Andy Zou, Long Phan, Justin Wang, Derek Duenas, Maxwell Lin, Maksym Andriushchenko, Rowan Wang, Zico Kolter, Matt Fredrikson, Dan Hendrycks

**Abstract**: AI systems can take harmful actions and are highly vulnerable to adversarial attacks. We present an approach, inspired by recent advances in representation engineering, that interrupts the models as they respond with harmful outputs with "circuit breakers." Existing techniques aimed at improving alignment, such as refusal training, are often bypassed. Techniques such as adversarial training try to plug these holes by countering specific attacks. As an alternative to refusal training and adversarial training, circuit-breaking directly controls the representations that are responsible for harmful outputs in the first place. Our technique can be applied to both text-only and multimodal language models to prevent the generation of harmful outputs without sacrificing utility -- even in the presence of powerful unseen attacks. Notably, while adversarial robustness in standalone image recognition remains an open challenge, circuit breakers allow the larger multimodal system to reliably withstand image "hijacks" that aim to produce harmful content. Finally, we extend our approach to AI agents, demonstrating considerable reductions in the rate of harmful actions when they are under attack. Our approach represents a significant step forward in the development of reliable safeguards to harmful behavior and adversarial attacks.

摘要: 人工智能系统可能采取有害行动，并且非常容易受到对抗性攻击。我们提出了一种方法，灵感来自于最近在表示工程方面的进展，该方法中断了模型，因为它们用“断路器”来响应有害的输出。旨在改善一致性的现有技术，如拒绝训练，经常被绕过。对抗性训练等技术试图通过反击特定攻击来堵塞这些漏洞。作为拒绝训练和对抗性训练的另一种选择，断路直接控制首先要对有害输出负责的陈述。我们的技术可以应用于纯文本和多模式语言模型，在不牺牲效用的情况下防止产生有害输出-即使在存在强大的看不见的攻击的情况下也是如此。值得注意的是，虽然独立图像识别中的对抗性健壮性仍然是一个开放的挑战，但断路器允许更大的多模式系统可靠地经受住旨在产生有害内容的图像“劫持”。最后，我们将我们的方法扩展到人工智能代理，表明当他们受到攻击时，有害行动的比率大大降低。我们的方法代表着在发展对有害行为和敌对攻击的可靠保障方面向前迈出了重要的一步。



## **44. Evolving Assembly Code in an Adversarial Environment**

对抗环境中发展汇编代码 cs.NE

20 pages, 6 figures, 6 listings, 5 tables

**SubmitDate**: 2024-06-10    [abs](http://arxiv.org/abs/2403.19489v2) [paper-pdf](http://arxiv.org/pdf/2403.19489v2)

**Authors**: Irina Maliukov, Gera Weiss, Oded Margalit, Achiya Elyasaf

**Abstract**: In this work, we evolve Assembly code for the CodeGuru competition. The goal is to create a survivor -- an Assembly program that runs the longest in shared memory, by resisting attacks from adversary survivors and finding their weaknesses. For evolving top-notch solvers, we specify a Backus Normal Form (BNF) for the Assembly language and synthesize the code from scratch using Genetic Programming (GP). We evaluate the survivors by running CodeGuru games against human-written winning survivors. Our evolved programs found weaknesses in the programs they were trained against and utilized them. To push evolution further, we implemented memetic operators that utilize machine learning to explore the solution space effectively. This work has important applications for cyber-security as we utilize evolution to detect weaknesses in survivors. The Assembly BNF is domain-independent; thus, by modifying the fitness function, it can detect code weaknesses and help fix them. Finally, the CodeGuru competition offers a novel platform for analyzing GP and code evolution in adversarial environments. To support further research in this direction, we provide a thorough qualitative analysis of the evolved survivors and the weaknesses found.

摘要: 在这项工作中，我们为CodeGuru竞赛演变汇编代码。目标是创建一个幸存者--一个在共享内存中运行时间最长的汇编程序，通过抵抗对手幸存者的攻击并找到他们的弱点。对于进化的顶级解算器，我们为汇编语言指定了Backus范式(BNF)，并使用遗传编程(GP)从头开始合成代码。我们通过运行CodeGuru游戏来评估幸存者，以对抗人类编写的获胜幸存者。我们的演进计划发现了他们所针对的计划中的弱点，并利用了这些弱点。为了进一步推进进化，我们实现了模因算子，利用机器学习来有效地探索解空间。这项工作在网络安全方面有重要的应用，因为我们利用进化论来检测幸存者的弱点。Assembly BNF是独立于域的；因此，通过修改适应度函数，它可以检测代码弱点并帮助修复它们。最后，CodeGuru竞赛为分析对抗性环境中的GP和代码演化提供了一个新的平台。为了支持这方面的进一步研究，我们对进化的幸存者和发现的弱点进行了彻底的定性分析。



## **45. Explainable Graph Neural Networks Under Fire**

受攻击的可解释图神经网络 cs.LG

**SubmitDate**: 2024-06-10    [abs](http://arxiv.org/abs/2406.06417v1) [paper-pdf](http://arxiv.org/pdf/2406.06417v1)

**Authors**: Zhong Li, Simon Geisler, Yuhang Wang, Stephan Günnemann, Matthijs van Leeuwen

**Abstract**: Predictions made by graph neural networks (GNNs) usually lack interpretability due to their complex computational behavior and the abstract nature of graphs. In an attempt to tackle this, many GNN explanation methods have emerged. Their goal is to explain a model's predictions and thereby obtain trust when GNN models are deployed in decision critical applications. Most GNN explanation methods work in a post-hoc manner and provide explanations in the form of a small subset of important edges and/or nodes. In this paper we demonstrate that these explanations can unfortunately not be trusted, as common GNN explanation methods turn out to be highly susceptible to adversarial perturbations. That is, even small perturbations of the original graph structure that preserve the model's predictions may yield drastically different explanations. This calls into question the trustworthiness and practical utility of post-hoc explanation methods for GNNs. To be able to attack GNN explanation models, we devise a novel attack method dubbed \textit{GXAttack}, the first \textit{optimization-based} adversarial attack method for post-hoc GNN explanations under such settings. Due to the devastating effectiveness of our attack, we call for an adversarial evaluation of future GNN explainers to demonstrate their robustness.

摘要: 由于图的复杂的计算行为和图的抽象性质，图神经网络(GNN)的预测通常缺乏可解释性。为了解决这个问题，出现了许多GNN解释方法。他们的目标是解释模型的预测，从而在决策关键应用程序中部署GNN模型时获得信任。大多数GNN解释方法以后自组织的方式工作，并以重要边和/或节点的小子集的形式提供解释。在本文中，我们证明了不幸的是，这些解释不能被信任，因为常见的GNN解释方法被证明非常容易受到对抗性扰动的影响。也就是说，即使是对原始图表结构的微小扰动，保留了模型的预测，也可能产生截然不同的解释。这使人们对特别解释GNN的方法的可信性和实用性产生了疑问。为了能够攻击GNN解释模型，我们设计了一种新的攻击方法，称为文本{GXAttack}，这是第一个在这种情况下针对后自组织GNN解释的对抗性攻击方法。由于我们的攻击具有毁灭性的效果，我们呼吁对未来的GNN解释器进行对抗性评估，以证明其健壮性。



## **46. RAID: A Shared Benchmark for Robust Evaluation of Machine-Generated Text Detectors**

RAGE：机器生成文本检测器稳健评估的共享基准 cs.CL

ACL 2024

**SubmitDate**: 2024-06-10    [abs](http://arxiv.org/abs/2405.07940v2) [paper-pdf](http://arxiv.org/pdf/2405.07940v2)

**Authors**: Liam Dugan, Alyssa Hwang, Filip Trhlik, Josh Magnus Ludan, Andrew Zhu, Hainiu Xu, Daphne Ippolito, Chris Callison-Burch

**Abstract**: Many commercial and open-source models claim to detect machine-generated text with extremely high accuracy (99% or more). However, very few of these detectors are evaluated on shared benchmark datasets and even when they are, the datasets used for evaluation are insufficiently challenging-lacking variations in sampling strategy, adversarial attacks, and open-source generative models. In this work we present RAID: the largest and most challenging benchmark dataset for machine-generated text detection. RAID includes over 6 million generations spanning 11 models, 8 domains, 11 adversarial attacks and 4 decoding strategies. Using RAID, we evaluate the out-of-domain and adversarial robustness of 8 open- and 4 closed-source detectors and find that current detectors are easily fooled by adversarial attacks, variations in sampling strategies, repetition penalties, and unseen generative models. We release our data along with a leaderboard to encourage future research.

摘要: 许多商业和开源模型声称可以以极高的准确性（99%或更高）检测机器生成的文本。然而，这些检测器中很少有在共享基准数据集上进行评估，即使如此，用于评估的数据集也不够具有挑战性--缺乏采样策略、对抗性攻击和开源生成模型的变化。在这项工作中，我们介绍了RAIDA：用于机器生成文本检测的最大、最具挑战性的基准数据集。磁盘阵列包含超过600万代，涵盖11个模型、8个域、11种对抗性攻击和4种解码策略。使用RAIDGE，我们评估了8个开源检测器和4个开源检测器的域外和对抗稳健性，发现当前的检测器很容易被对抗攻击、采样策略的变化、重复惩罚和看不见的生成模型所愚弄。我们发布我们的数据和排行榜，以鼓励未来的研究。



## **47. Towards Transferable Targeted 3D Adversarial Attack in the Physical World**

迈向物理世界中的可转移定向3D对抗攻击 cs.CV

Accepted by CVPR 2024

**SubmitDate**: 2024-06-10    [abs](http://arxiv.org/abs/2312.09558v3) [paper-pdf](http://arxiv.org/pdf/2312.09558v3)

**Authors**: Yao Huang, Yinpeng Dong, Shouwei Ruan, Xiao Yang, Hang Su, Xingxing Wei

**Abstract**: Compared with transferable untargeted attacks, transferable targeted adversarial attacks could specify the misclassification categories of adversarial samples, posing a greater threat to security-critical tasks. In the meanwhile, 3D adversarial samples, due to their potential of multi-view robustness, can more comprehensively identify weaknesses in existing deep learning systems, possessing great application value. However, the field of transferable targeted 3D adversarial attacks remains vacant. The goal of this work is to develop a more effective technique that could generate transferable targeted 3D adversarial examples, filling the gap in this field. To achieve this goal, we design a novel framework named TT3D that could rapidly reconstruct from few multi-view images into Transferable Targeted 3D textured meshes. While existing mesh-based texture optimization methods compute gradients in the high-dimensional mesh space and easily fall into local optima, leading to unsatisfactory transferability and distinct distortions, TT3D innovatively performs dual optimization towards both feature grid and Multi-layer Perceptron (MLP) parameters in the grid-based NeRF space, which significantly enhances black-box transferability while enjoying naturalness. Experimental results show that TT3D not only exhibits superior cross-model transferability but also maintains considerable adaptability across different renders and vision tasks. More importantly, we produce 3D adversarial examples with 3D printing techniques in the real world and verify their robust performance under various scenarios.

摘要: 与可转移的非定向攻击相比，可转移的定向攻击可以指定对手样本的错误分类类别，对安全关键任务构成更大的威胁。同时，3D对抗性样本由于其潜在的多视点稳健性，可以更全面地识别现有深度学习系统中的弱点，具有很大的应用价值。然而，可转移的定向3D对抗性攻击领域仍然空白。这项工作的目标是开发一种更有效的技术，可以生成可转移的目标3D对抗性实例，填补这一领域的空白。为了实现这一目标，我们设计了一种新的框架TT3D，它可以从少量的多视角图像快速重建为可转移的目标3D纹理网格。针对现有的基于网格的纹理优化方法在高维网格空间中计算梯度，容易陷入局部最优，导致可移植性差和失真明显的问题，TT3D创新性地在基于网格的NERF空间中对特征网格和多层感知器(MLP)参数进行双重优化，在享受自然感的同时显著增强了黑盒的可传递性。实验结果表明，TT3D不仅表现出了良好的跨模型可移植性，而且在不同的渲染和视觉任务之间保持了相当大的适应性。更重要的是，我们用3D打印技术在真实世界中生成了3D对抗性例子，并验证了它们在各种场景下的健壮性。



## **48. Siren -- Advancing Cybersecurity through Deception and Adaptive Analysis**

警报器--通过欺骗和适应性分析推进网络安全 cs.CR

7 pages, 6 figures

**SubmitDate**: 2024-06-10    [abs](http://arxiv.org/abs/2406.06225v1) [paper-pdf](http://arxiv.org/pdf/2406.06225v1)

**Authors**: Girish Kulathumani, Samruth Ananthanarayanan, Ganesh Narayanan

**Abstract**: Siren represents a pioneering research effort aimed at fortifying cybersecurity through strategic integration of deception, machine learning, and proactive threat analysis. Drawing inspiration from mythical sirens, this project employs sophisticated methods to lure potential threats into controlled environments. The system features a dynamic machine learning model for real-time analysis and classification, ensuring continuous adaptability to emerging cyber threats. The architectural framework includes a link monitoring proxy, a purpose-built machine learning model for dynamic link analysis, and a honeypot enriched with simulated user interactions to intensify threat engagement. Data protection within the honeypot is fortified with probabilistic encryption. Additionally, the incorporation of simulated user activity extends the system's capacity to capture and learn from potential attackers even after user disengagement. Siren introduces a paradigm shift in cybersecurity, transforming traditional defense mechanisms into proactive systems that actively engage and learn from potential adversaries. The research strives to enhance user protection while yielding valuable insights for ongoing refinement in response to the evolving landscape of cybersecurity threats.

摘要: SIREN代表了一项开创性的研究成果，旨在通过欺骗、机器学习和主动威胁分析的战略集成来加强网络安全。这个项目从神话中的警报器中获得灵感，使用复杂的方法将潜在的威胁引诱到受控环境中。该系统采用动态机器学习模型进行实时分析和分类，确保对新出现的网络威胁持续适应。该体系结构框架包括链接监控代理、用于动态链接分析的专门构建的机器学习模型，以及丰富了模拟用户交互以加强威胁参与的蜜罐。蜜罐内的数据保护通过概率加密得到加强。此外，模拟用户活动的加入扩展了系统的能力，即使在用户退出后也能捕获潜在攻击者并从他们那里学习。SIREN在网络安全方面引入了一种范式转变，将传统的防御机制转变为主动参与并向潜在对手学习的系统。这项研究努力加强用户保护，同时为不断完善以应对不断变化的网络安全威胁提供有价值的见解。



## **49. Defending Against Physical Adversarial Patch Attacks on Infrared Human Detection**

红外人体检测防御物理对抗补丁攻击 cs.CV

Accepted at ICIP2024. Lukas Strack and Futa Waseda contributed  equally

**SubmitDate**: 2024-06-10    [abs](http://arxiv.org/abs/2309.15519v3) [paper-pdf](http://arxiv.org/pdf/2309.15519v3)

**Authors**: Lukas Strack, Futa Waseda, Huy H. Nguyen, Yinqiang Zheng, Isao Echizen

**Abstract**: Infrared detection is an emerging technique for safety-critical tasks owing to its remarkable anti-interference capability. However, recent studies have revealed that it is vulnerable to physically-realizable adversarial patches, posing risks in its real-world applications. To address this problem, we are the first to investigate defense strategies against adversarial patch attacks on infrared detection, especially human detection. We propose a straightforward defense strategy, patch-based occlusion-aware detection (POD), which efficiently augments training samples with random patches and subsequently detects them. POD not only robustly detects people but also identifies adversarial patch locations. Surprisingly, while being extremely computationally efficient, POD easily generalizes to state-of-the-art adversarial patch attacks that are unseen during training. Furthermore, POD improves detection precision even in a clean (i.e., no-attack) situation due to the data augmentation effect. Our evaluation demonstrates that POD is robust to adversarial patches of various shapes and sizes. The effectiveness of our baseline approach is shown to be a viable defense mechanism for real-world infrared human detection systems, paving the way for exploring future research directions.

摘要: 红外探测是一种新兴的安全关键任务检测技术，具有显著的抗干扰性。然而，最近的研究表明，它很容易受到物理上可实现的对抗性补丁的攻击，这给它在现实世界的应用带来了风险。针对这一问题，我们首次研究了针对红外探测，尤其是人体探测的对抗性补丁攻击的防御策略。我们提出了一种简单的防御策略，基于补丁的遮挡感知检测(POD)，它有效地利用随机补丁来增加训练样本并随后对其进行检测。Pod不仅可以稳健地检测人员，还可以识别敌方的补丁位置。令人惊讶的是，虽然POD在计算上非常高效，但它很容易概括为最先进的对抗性补丁攻击，这些攻击在训练中是看不到的。此外，由于数据增强效应，即使在干净(即，无攻击)的情况下，POD也提高了检测精度。我们的评估表明，POD对不同形状和大小的敌方补丁具有很强的鲁棒性。我们的基线方法的有效性被证明是一种可行的防御机制，用于真实世界的红外人体探测系统，为探索未来的研究方向铺平了道路。



## **50. Making Them Ask and Answer: Jailbreaking Large Language Models in Few Queries via Disguise and Reconstruction**

让他们问答：通过伪装和重建在很短的时间内越狱大型语言模型 cs.CR

**SubmitDate**: 2024-06-10    [abs](http://arxiv.org/abs/2402.18104v2) [paper-pdf](http://arxiv.org/pdf/2402.18104v2)

**Authors**: Tong Liu, Yingjie Zhang, Zhe Zhao, Yinpeng Dong, Guozhu Meng, Kai Chen

**Abstract**: In recent years, large language models (LLMs) have demonstrated notable success across various tasks, but the trustworthiness of LLMs is still an open problem. One specific threat is the potential to generate toxic or harmful responses. Attackers can craft adversarial prompts that induce harmful responses from LLMs. In this work, we pioneer a theoretical foundation in LLMs security by identifying bias vulnerabilities within the safety fine-tuning and design a black-box jailbreak method named DRA (Disguise and Reconstruction Attack), which conceals harmful instructions through disguise and prompts the model to reconstruct the original harmful instruction within its completion. We evaluate DRA across various open-source and closed-source models, showcasing state-of-the-art jailbreak success rates and attack efficiency. Notably, DRA boasts a 91.1% attack success rate on OpenAI GPT-4 chatbot.

摘要: 近年来，大型语言模型（LLM）在各种任务中取得了显着的成功，但LLM的可信度仍然是一个悬而未决的问题。一个具体的威胁是可能产生有毒或有害反应。攻击者可以设计对抗性提示，引发LLM的有害反应。在这项工作中，我们通过识别安全微调中的偏见漏洞，开创了LLM安全的理论基础，并设计了一种名为“伪装和重建攻击”的黑匣子越狱方法，通过伪装隐藏有害指令，并促使模型在完成时重建原始有害指令。我们评估各种开源和开源模型的NPS，展示最先进的越狱成功率和攻击效率。值得注意的是，Inbox对OpenAI GPT-4聊天机器人的攻击成功率为91.1%。



