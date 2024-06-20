# Latest Adversarial Attack Papers
**update at 2024-06-20 09:40:05**

翻译来自 https://cloud.tencent.com/document/product/551/15619

## **1. Can Go AIs be adversarially robust?**

Go AI能否具有对抗性强大？ cs.LG

67 pages

**SubmitDate**: 2024-06-18    [abs](http://arxiv.org/abs/2406.12843v1) [paper-pdf](http://arxiv.org/pdf/2406.12843v1)

**Authors**: Tom Tseng, Euan McLean, Kellin Pelrine, Tony T. Wang, Adam Gleave

**Abstract**: Prior work found that superhuman Go AIs like KataGo can be defeated by simple adversarial strategies. In this paper, we study if simple defenses can improve KataGo's worst-case performance. We test three natural defenses: adversarial training on hand-constructed positions, iterated adversarial training, and changing the network architecture. We find that some of these defenses are able to protect against previously discovered attacks. Unfortunately, we also find that none of these defenses are able to withstand adaptive attacks. In particular, we are able to train new adversaries that reliably defeat our defended agents by causing them to blunder in ways humans would not. Our results suggest that building robust AI systems is challenging even in narrow domains such as Go. For interactive examples of attacks and a link to our codebase, see https://goattack.far.ai.

摘要: 之前的工作发现，像KataGo这样的超人围棋人工智能可以被简单的对抗策略击败。在本文中，我们研究简单的防御是否可以提高KataGo的最坏情况下的性能。我们测试三种自然防御：手工构建位置上的对抗训练、迭代对抗训练以及改变网络架构。我们发现其中一些防御措施能够抵御之前发现的攻击。不幸的是，我们还发现这些防御措施都无法抵御适应性攻击。特别是，我们能够训练新的对手，通过让我们的防御特工犯下人类不会犯的错误来可靠地击败他们。我们的结果表明，即使在Go等狭窄领域，构建强大的人工智能系统也具有挑战性。有关攻击的交互式示例和我们代码库的链接，请访问https://goattack.far.ai。



## **2. Adversarial Attacks on Multimodal Agents**

对多模式代理的对抗攻击 cs.LG

19 pages

**SubmitDate**: 2024-06-18    [abs](http://arxiv.org/abs/2406.12814v1) [paper-pdf](http://arxiv.org/pdf/2406.12814v1)

**Authors**: Chen Henry Wu, Jing Yu Koh, Ruslan Salakhutdinov, Daniel Fried, Aditi Raghunathan

**Abstract**: Vision-enabled language models (VLMs) are now used to build autonomous multimodal agents capable of taking actions in real environments. In this paper, we show that multimodal agents raise new safety risks, even though attacking agents is more challenging than prior attacks due to limited access to and knowledge about the environment. Our attacks use adversarial text strings to guide gradient-based perturbation over one trigger image in the environment: (1) our captioner attack attacks white-box captioners if they are used to process images into captions as additional inputs to the VLM; (2) our CLIP attack attacks a set of CLIP models jointly, which can transfer to proprietary VLMs. To evaluate the attacks, we curated VisualWebArena-Adv, a set of adversarial tasks based on VisualWebArena, an environment for web-based multimodal agent tasks. Within an L-infinity norm of $16/256$ on a single image, the captioner attack can make a captioner-augmented GPT-4V agent execute the adversarial goals with a 75% success rate. When we remove the captioner or use GPT-4V to generate its own captions, the CLIP attack can achieve success rates of 21% and 43%, respectively. Experiments on agents based on other VLMs, such as Gemini-1.5, Claude-3, and GPT-4o, show interesting differences in their robustness. Further analysis reveals several key factors contributing to the attack's success, and we also discuss the implications for defenses as well. Project page: https://chenwu.io/attack-agent Code and data: https://github.com/ChenWu98/agent-attack

摘要: 视觉使能语言模型(VLM)现在被用来构建能够在真实环境中采取行动的自主多通道代理。在本文中，我们证明了多模式代理带来了新的安全风险，尽管由于对环境的访问和了解有限，攻击代理比以前的攻击更具挑战性。我们的攻击使用敌意文本串来引导环境中一幅触发图像的基于梯度的扰动：(1)如果白盒捕获器被用于将图像处理为字幕作为VLM的额外输入，则我们的捕获器攻击白盒捕获器；(2)我们的剪辑攻击联合攻击一组剪辑模型，这些模型可以传输到专有的VLM。为了评估攻击，我们策划了VisualWebArena-ADV，这是一组基于VisualWebArena的对抗性任务，VisualWebArena是一个基于Web的多模式代理任务的环境。在单个图像上的L无穷范数16/256美元内，俘获攻击可以使捕获器增强的GPT-4V代理以75%的成功率执行对抗性目标。当我们移除捕捉者或使用GPT-4V生成自己的字幕时，剪辑攻击可以分别达到21%和43%的成功率。在基于其他VLM的代理(如Gemini-1.5、Claude-3和GPT-40)上的实验显示，它们在健壮性方面存在有趣的差异。进一步的分析揭示了导致攻击成功的几个关键因素，我们还讨论了对防御的影响。项目页面：https://chenwu.io/attack-agent代码和数据：https://github.com/ChenWu98/agent-attack



## **3. Jailbreaking Leading Safety-Aligned LLMs with Simple Adaptive Attacks**

通过简单的自适应攻击越狱领先的安全一致LLM cs.CR

Updates in the v2: more models (Llama3, Phi-3, Nemotron-4-340B),  jailbreak artifacts for all attacks are available, evaluation of  generalization to a different judge (Llama-3-70B and Llama Guard 2), more  experiments (convergence plots over iterations, ablation on the suffix length  for random search), improved exposition of the paper, examples of jailbroken  generation

**SubmitDate**: 2024-06-18    [abs](http://arxiv.org/abs/2404.02151v2) [paper-pdf](http://arxiv.org/pdf/2404.02151v2)

**Authors**: Maksym Andriushchenko, Francesco Croce, Nicolas Flammarion

**Abstract**: We show that even the most recent safety-aligned LLMs are not robust to simple adaptive jailbreaking attacks. First, we demonstrate how to successfully leverage access to logprobs for jailbreaking: we initially design an adversarial prompt template (sometimes adapted to the target LLM), and then we apply random search on a suffix to maximize a target logprob (e.g., of the token ``Sure''), potentially with multiple restarts. In this way, we achieve nearly 100% attack success rate -- according to GPT-4 as a judge -- on Vicuna-13B, Mistral-7B, Phi-3-Mini, Nemotron-4-340B, Llama-2-Chat-7B/13B/70B, Llama-3-Instruct-8B, Gemma-7B, GPT-3.5, GPT-4, and R2D2 from HarmBench that was adversarially trained against the GCG attack. We also show how to jailbreak all Claude models -- that do not expose logprobs -- via either a transfer or prefilling attack with a 100% success rate. In addition, we show how to use random search on a restricted set of tokens for finding trojan strings in poisoned models -- a task that shares many similarities with jailbreaking -- which is the algorithm that brought us the first place in the SaTML'24 Trojan Detection Competition. The common theme behind these attacks is that adaptivity is crucial: different models are vulnerable to different prompting templates (e.g., R2D2 is very sensitive to in-context learning prompts), some models have unique vulnerabilities based on their APIs (e.g., prefilling for Claude), and in some settings, it is crucial to restrict the token search space based on prior knowledge (e.g., for trojan detection). For reproducibility purposes, we provide the code, logs, and jailbreak artifacts in the JailbreakBench format at https://github.com/tml-epfl/llm-adaptive-attacks.

摘要: 我们表明，即使是最新的安全对齐的LLM也不能抵抗简单的自适应越狱攻击。首先，我们演示了如何成功地利用对logpros的访问来越狱：我们最初设计了一个对抗性提示模板(有时适用于目标LLM)，然后对后缀应用随机搜索以最大化目标logprob(例如，令牌`Sure‘)，可能需要多次重新启动。通过这种方式，我们实现了近100%的攻击成功率--根据GPT-4作为评委--对来自哈姆班奇的维古纳-13B、西北风-7B、Phi-3-Mini、Nemotron-4-340B、Llama-2-Chat-7B/13B/70B、Llama-3-Indict-8B、Gema-7B、GPT-3.5、GPT-4和R2D2进行了对抗GCG攻击的对抗训练。我们还展示了如何通过传输或预填充攻击以100%的成功率越狱所有不暴露日志问题的Claude模型。此外，我们还展示了如何在受限的令牌集合上使用随机搜索来查找有毒模型中的特洛伊木马字符串--这项任务与越狱有许多相似之处--正是这种算法为我们带来了SATML‘24特洛伊木马检测大赛的第一名。这些攻击背后的共同主题是自适应至关重要：不同的模型容易受到不同提示模板的影响(例如，R2D2对上下文中的学习提示非常敏感)，一些模型基于其API具有独特的漏洞(例如，预填充Claude)，并且在某些设置中，基于先验知识限制令牌搜索空间至关重要(例如，对于木马检测)。出于可重现性的目的，我们在https://github.com/tml-epfl/llm-adaptive-attacks.上以JailBreak B边格式提供代码、日志和越狱构件



## **4. UIFV: Data Reconstruction Attack in Vertical Federated Learning**

UIFV：垂直联邦学习中的数据重建攻击 cs.LG

**SubmitDate**: 2024-06-18    [abs](http://arxiv.org/abs/2406.12588v1) [paper-pdf](http://arxiv.org/pdf/2406.12588v1)

**Authors**: Jirui Yang, Peng Chen, Zhihui Lu, Qiang Duan, Yubing Bao

**Abstract**: Vertical Federated Learning (VFL) facilitates collaborative machine learning without the need for participants to share raw private data. However, recent studies have revealed privacy risks where adversaries might reconstruct sensitive features through data leakage during the learning process. Although data reconstruction methods based on gradient or model information are somewhat effective, they reveal limitations in VFL application scenarios. This is because these traditional methods heavily rely on specific model structures and/or have strict limitations on application scenarios. To address this, our study introduces the Unified InverNet Framework into VFL, which yields a novel and flexible approach (dubbed UIFV) that leverages intermediate feature data to reconstruct original data, instead of relying on gradients or model details. The intermediate feature data is the feature exchanged by different participants during the inference phase of VFL. Experiments on four datasets demonstrate that our methods significantly outperform state-of-the-art techniques in attack precision. Our work exposes severe privacy vulnerabilities within VFL systems that pose real threats to practical VFL applications and thus confirms the necessity of further enhancing privacy protection in the VFL architecture.

摘要: 垂直联合学习(VFL)促进了协作机器学习，而不需要参与者共享原始私有数据。然而，最近的研究揭示了隐私风险，攻击者可能会在学习过程中通过数据泄露来重建敏感特征。虽然基于梯度或模型信息的数据重建方法在一定程度上是有效的，但它们在VFL应用场景中暴露出局限性。这是因为这些传统方法严重依赖于特定的模型结构和/或对应用场景有严格的限制。为了解决这一问题，我们的研究将统一InverNet框架引入到VFL中，从而产生了一种新颖而灵活的方法(称为UIFV)，该方法利用中间特征数据来重建原始数据，而不是依赖于梯度或模型细节。中间特征数据是不同参与者在VFL推理阶段交换的特征。在四个数据集上的实验表明，我们的方法在攻击精度上明显优于最先进的技术。我们的工作暴露了VFL系统中严重的隐私漏洞，这些漏洞对实际的VFL应用构成了真正的威胁，从而证实了在VFL体系结构中进一步加强隐私保护的必要性。



## **5. Authorship Obfuscation in Multilingual Machine-Generated Text Detection**

多语言机器生成文本检测中的作者混淆 cs.CL

**SubmitDate**: 2024-06-18    [abs](http://arxiv.org/abs/2401.07867v2) [paper-pdf](http://arxiv.org/pdf/2401.07867v2)

**Authors**: Dominik Macko, Robert Moro, Adaku Uchendu, Ivan Srba, Jason Samuel Lucas, Michiharu Yamashita, Nafis Irtiza Tripto, Dongwon Lee, Jakub Simko, Maria Bielikova

**Abstract**: High-quality text generation capability of recent Large Language Models (LLMs) causes concerns about their misuse (e.g., in massive generation/spread of disinformation). Machine-generated text (MGT) detection is important to cope with such threats. However, it is susceptible to authorship obfuscation (AO) methods, such as paraphrasing, which can cause MGTs to evade detection. So far, this was evaluated only in monolingual settings. Thus, the susceptibility of recently proposed multilingual detectors is still unknown. We fill this gap by comprehensively benchmarking the performance of 10 well-known AO methods, attacking 37 MGT detection methods against MGTs in 11 languages (i.e., 10 $\times$ 37 $\times$ 11 = 4,070 combinations). We also evaluate the effect of data augmentation on adversarial robustness using obfuscated texts. The results indicate that all tested AO methods can cause evasion of automated detection in all tested languages, where homoglyph attacks are especially successful. However, some of the AO methods severely damaged the text, making it no longer readable or easily recognizable by humans (e.g., changed language, weird characters).

摘要: 最近的大型语言模型(LLM)的高质量文本生成能力引起了人们对它们的滥用(例如，在大规模生成/传播虚假信息中)的担忧。机器生成文本(MGT)检测对于应对此类威胁非常重要。然而，它容易受到作者身份混淆(AO)方法的影响，例如转译，这可能导致MGTS逃避检测。到目前为止，这只在单一语言环境中进行了评估。因此，最近提出的多语言检测器的敏感性仍然未知。我们通过全面基准测试10种著名的AO方法的性能来填补这一空白，针对11种语言的MGT攻击37种MGT检测方法(即，10$\乘以$37$\乘以$11=4,070个组合)。我们还使用混淆文本来评估数据增强对对手健壮性的影响。结果表明，在所有被测语言中，所有被测试的声学方法都可以逃避自动检测，其中同形文字攻击尤其成功。然而，一些AO方法严重损坏了文本，使其不再可读或不再容易被人类识别(例如，改变语言、奇怪的字符)。



## **6. A Survey of Fragile Model Watermarking**

脆弱模型水印综述 cs.CR

**SubmitDate**: 2024-06-18    [abs](http://arxiv.org/abs/2406.04809v2) [paper-pdf](http://arxiv.org/pdf/2406.04809v2)

**Authors**: Zhenzhe Gao, Yu Cheng, Zhaoxia Yin

**Abstract**: Model fragile watermarking, inspired by both the field of adversarial attacks on neural networks and traditional multimedia fragile watermarking, has gradually emerged as a potent tool for detecting tampering, and has witnessed rapid development in recent years. Unlike robust watermarks, which are widely used for identifying model copyrights, fragile watermarks for models are designed to identify whether models have been subjected to unexpected alterations such as backdoors, poisoning, compression, among others. These alterations can pose unknown risks to model users, such as misidentifying stop signs as speed limit signs in classic autonomous driving scenarios. This paper provides an overview of the relevant work in the field of model fragile watermarking since its inception, categorizing them and revealing the developmental trajectory of the field, thus offering a comprehensive survey for future endeavors in model fragile watermarking.

摘要: 模型脆弱水印受到神经网络对抗攻击领域和传统多媒体脆弱水印的启发，逐渐成为检测篡改的有力工具，并在近年来得到了快速发展。与广泛用于识别模型版权的稳健水印不同，模型的脆弱水印旨在识别模型是否遭受了意外更改，例如后门、中毒、压缩等。这些更改可能会给模型用户带来未知的风险，例如在经典自动驾驶场景中将停车标志误识别为限速标志。本文概述了模型脆弱水印领域自诞生以来的相关工作，对其进行了分类，揭示了该领域的发展轨迹，从而为模型脆弱水印的未来工作提供了全面的综述。



## **7. Adversarial Attacks on Large Language Models in Medicine**

医学中对大型语言模型的对抗攻击 cs.AI

**SubmitDate**: 2024-06-18    [abs](http://arxiv.org/abs/2406.12259v1) [paper-pdf](http://arxiv.org/pdf/2406.12259v1)

**Authors**: Yifan Yang, Qiao Jin, Furong Huang, Zhiyong Lu

**Abstract**: The integration of Large Language Models (LLMs) into healthcare applications offers promising advancements in medical diagnostics, treatment recommendations, and patient care. However, the susceptibility of LLMs to adversarial attacks poses a significant threat, potentially leading to harmful outcomes in delicate medical contexts. This study investigates the vulnerability of LLMs to two types of adversarial attacks in three medical tasks. Utilizing real-world patient data, we demonstrate that both open-source and proprietary LLMs are susceptible to manipulation across multiple tasks. This research further reveals that domain-specific tasks demand more adversarial data in model fine-tuning than general domain tasks for effective attack execution, especially for more capable models. We discover that while integrating adversarial data does not markedly degrade overall model performance on medical benchmarks, it does lead to noticeable shifts in fine-tuned model weights, suggesting a potential pathway for detecting and countering model attacks. This research highlights the urgent need for robust security measures and the development of defensive mechanisms to safeguard LLMs in medical applications, to ensure their safe and effective deployment in healthcare settings.

摘要: 将大型语言模型(LLM)集成到医疗保健应用程序中，在医疗诊断、治疗建议和患者护理方面提供了有希望的进步。然而，LLMS对对抗性攻击的敏感性构成了一个重大威胁，可能会在微妙的医疗环境中导致有害后果。本研究调查了LLMS在三个医疗任务中对两种类型的对抗性攻击的脆弱性。利用真实世界的患者数据，我们证明了开源和专有LLM都容易受到跨多个任务的操纵。这项研究进一步表明，特定领域的任务在模型微调中需要比一般领域任务更多的对抗性数据才能有效地执行攻击，特别是对于能力更强的模型。我们发现，虽然整合对抗性数据并不会显著降低医学基准上的整体模型性能，但它确实会导致微调模型权重的显著变化，这表明了一条检测和对抗模型攻击的潜在路径。这项研究强调了迫切需要强有力的安全措施和开发防御机制来保护医疗应用中的低成本管理，以确保其在医疗保健环境中的安全和有效部署。



## **8. Privacy-Preserved Neural Graph Databases**

隐私保护的神经图数据库 cs.DB

**SubmitDate**: 2024-06-18    [abs](http://arxiv.org/abs/2312.15591v5) [paper-pdf](http://arxiv.org/pdf/2312.15591v5)

**Authors**: Qi Hu, Haoran Li, Jiaxin Bai, Zihao Wang, Yangqiu Song

**Abstract**: In the era of large language models (LLMs), efficient and accurate data retrieval has become increasingly crucial for the use of domain-specific or private data in the retrieval augmented generation (RAG). Neural graph databases (NGDBs) have emerged as a powerful paradigm that combines the strengths of graph databases (GDBs) and neural networks to enable efficient storage, retrieval, and analysis of graph-structured data which can be adaptively trained with LLMs. The usage of neural embedding storage and Complex neural logical Query Answering (CQA) provides NGDBs with generalization ability. When the graph is incomplete, by extracting latent patterns and representations, neural graph databases can fill gaps in the graph structure, revealing hidden relationships and enabling accurate query answering. Nevertheless, this capability comes with inherent trade-offs, as it introduces additional privacy risks to the domain-specific or private databases. Malicious attackers can infer more sensitive information in the database using well-designed queries such as from the answer sets of where Turing Award winners born before 1950 and after 1940 lived, the living places of Turing Award winner Hinton are probably exposed, although the living places may have been deleted in the training stage due to the privacy concerns. In this work, we propose a privacy-preserved neural graph database (P-NGDB) framework to alleviate the risks of privacy leakage in NGDBs. We introduce adversarial training techniques in the training stage to enforce the NGDBs to generate indistinguishable answers when queried with private information, enhancing the difficulty of inferring sensitive information through combinations of multiple innocuous queries.

摘要: 在大型语言模型(LLMS)时代，高效和准确的数据检索对于在检索增强生成(RAG)中使用特定领域或私有数据变得越来越重要。神经图形数据库(NGDB)已经成为一种强大的范例，它结合了图形数据库(GDB)和神经网络的优点，能够有效地存储、检索和分析图结构的数据，这些数据可以用LLMS进行自适应训练。神经嵌入存储和复杂神经逻辑查询应答(CQA)的使用为NGDB提供了泛化能力。当图不完整时，通过提取潜在模式和表示，神经图库可以填补图结构中的空白，揭示隐藏的关系，并使查询得到准确的回答。然而，这种能力是有内在权衡的，因为它会给特定于域或私有的数据库带来额外的隐私风险。恶意攻击者可以使用精心设计的查询来推断数据库中更敏感的信息，例如从图灵奖获得者1950年前和1940年后出生的地方的答案集中，图灵奖获得者辛顿的居住地可能会被曝光，尽管出于隐私考虑，居住地可能在培训阶段已被删除。在这项工作中，我们提出了一个隐私保护的神经图库(P-NGDB)框架，以缓解NGDB中隐私泄露的风险。在训练阶段引入对抗性训练技术，强制NGDB在查询私有信息时产生不可区分的答案，增加了通过组合多个无害查询来推断敏感信息的难度。



## **9. Robust Text Classification: Analyzing Prototype-Based Networks**

稳健的文本分类：分析基于原型的网络 cs.CL

**SubmitDate**: 2024-06-18    [abs](http://arxiv.org/abs/2311.06647v2) [paper-pdf](http://arxiv.org/pdf/2311.06647v2)

**Authors**: Zhivar Sourati, Darshan Deshpande, Filip Ilievski, Kiril Gashteovski, Sascha Saralajew

**Abstract**: Downstream applications often require text classification models to be accurate and robust. While the accuracy of the state-of-the-art Language Models (LMs) approximates human performance, they often exhibit a drop in performance on noisy data found in the real world. This lack of robustness can be concerning, as even small perturbations in the text, irrelevant to the target task, can cause classifiers to incorrectly change their predictions. A potential solution can be the family of Prototype-Based Networks (PBNs) that classifies examples based on their similarity to prototypical examples of a class (prototypes) and has been shown to be robust to noise for computer vision tasks. In this paper, we study whether the robustness properties of PBNs transfer to text classification tasks under both targeted and static adversarial attack settings. Our results show that PBNs, as a mere architectural variation of vanilla LMs, offer more robustness compared to vanilla LMs under both targeted and static settings. We showcase how PBNs' interpretability can help us to understand PBNs' robustness properties. Finally, our ablation studies reveal the sensitivity of PBNs' robustness to how strictly clustering is done in the training phase, as tighter clustering results in less robust PBNs.

摘要: 下游应用通常要求文本分类模型准确和健壮。虽然最先进的语言模型(LMS)的准确性接近人类的表现，但它们在处理现实世界中发现的噪声数据时往往表现出性能下降。这种缺乏稳健性可能会令人担忧，因为即使文本中与目标任务无关的微小扰动也可能导致分类器错误地改变他们的预测。一个潜在的解决方案可以是基于原型的网络(PBN)家族，其基于实例与一类(原型)的原型实例的相似性来对实例进行分类，并且已经被证明对计算机视觉任务的噪声是稳健的。本文研究了在目标攻击和静态攻击两种情况下，PBN的健壮性是否会转移到文本分类任务上。我们的结果表明，与普通LMS相比，PBN在目标和静态环境下都提供了更好的健壮性。我们展示了PBN的可解释性如何帮助我们理解PBN的健壮性。最后，我们的消融研究揭示了PBN的稳健性对训练阶段如何严格地进行聚类的敏感性，因为更紧密的聚类会导致更不健壮的PBN。



## **10. BadSampler: Harnessing the Power of Catastrophic Forgetting to Poison Byzantine-robust Federated Learning**

BadSampler：利用灾难性遗忘的力量毒害拜占庭强大的联邦学习 cs.CR

In Proceedings of the 30th ACM SIGKDD Conference on Knowledge  Discovery and Data Mining (KDD' 24), August 25-29, 2024, Barcelona, Spain

**SubmitDate**: 2024-06-18    [abs](http://arxiv.org/abs/2406.12222v1) [paper-pdf](http://arxiv.org/pdf/2406.12222v1)

**Authors**: Yi Liu, Cong Wang, Xingliang Yuan

**Abstract**: Federated Learning (FL) is susceptible to poisoning attacks, wherein compromised clients manipulate the global model by modifying local datasets or sending manipulated model updates. Experienced defenders can readily detect and mitigate the poisoning effects of malicious behaviors using Byzantine-robust aggregation rules. However, the exploration of poisoning attacks in scenarios where such behaviors are absent remains largely unexplored for Byzantine-robust FL. This paper addresses the challenging problem of poisoning Byzantine-robust FL by introducing catastrophic forgetting. To fill this gap, we first formally define generalization error and establish its connection to catastrophic forgetting, paving the way for the development of a clean-label data poisoning attack named BadSampler. This attack leverages only clean-label data (i.e., without poisoned data) to poison Byzantine-robust FL and requires the adversary to selectively sample training data with high loss to feed model training and maximize the model's generalization error. We formulate the attack as an optimization problem and present two elegant adversarial sampling strategies, Top-$\kappa$ sampling, and meta-sampling, to approximately solve it. Additionally, our formal error upper bound and time complexity analysis demonstrate that our design can preserve attack utility with high efficiency. Extensive evaluations on two real-world datasets illustrate the effectiveness and performance of our proposed attacks.

摘要: 联合学习(FL)容易受到中毒攻击，受攻击的客户端通过修改局部数据集或发送被操纵的模型更新来操纵全局模型。经验丰富的防御者可以使用拜占庭稳健的聚合规则轻松检测和缓解恶意行为的中毒影响。然而，对于拜占庭式的稳健的FL来说，在没有这种行为的情况下对中毒攻击的探索在很大程度上仍然没有被探索。本文通过引入灾难性遗忘来解决毒害拜占庭稳健FL的挑战性问题。为了填补这一空白，我们首先正式定义了泛化错误，并建立了它与灾难性遗忘的联系，为开发名为BadSsamer的干净标签数据中毒攻击铺平了道路。该攻击仅利用干净的标签数据(即，没有有毒数据)来毒化拜占庭稳健的FL，并要求对手选择性地对高丢失的训练数据进行采样来支持模型训练，并最大化模型的泛化误差。我们将攻击描述为一个优化问题，并提出了两种巧妙的对抗性抽样策略：Top-$\kappa$抽样和Meta-抽样，以近似求解该问题。此外，我们的形式误差上界和时间复杂性分析表明，我们的设计能够高效地保持攻击效用。在两个真实世界数据集上的广泛评估表明了我们所提出的攻击的有效性和性能。



## **11. Attack on Scene Flow using Point Clouds**

使用点云攻击场景流 cs.CV

**SubmitDate**: 2024-06-18    [abs](http://arxiv.org/abs/2404.13621v3) [paper-pdf](http://arxiv.org/pdf/2404.13621v3)

**Authors**: Haniyeh Ehsani Oskouie, Mohammad-Shahram Moin, Shohreh Kasaei

**Abstract**: Deep neural networks have made significant advancements in accurately estimating scene flow using point clouds, which is vital for many applications like video analysis, action recognition, and navigation. The robustness of these techniques, however, remains a concern, particularly in the face of adversarial attacks that have been proven to deceive state-of-the-art deep neural networks in many domains. Surprisingly, the robustness of scene flow networks against such attacks has not been thoroughly investigated. To address this problem, the proposed approach aims to bridge this gap by introducing adversarial white-box attacks specifically tailored for scene flow networks. Experimental results show that the generated adversarial examples obtain up to 33.7 relative degradation in average end-point error on the KITTI and FlyingThings3D datasets. The study also reveals the significant impact that attacks targeting point clouds in only one dimension or color channel have on average end-point error. Analyzing the success and failure of these attacks on the scene flow networks and their 2D optical flow network variants shows a higher vulnerability for the optical flow networks.

摘要: 深度神经网络在利用点云准确估计场景流量方面取得了重大进展，这对于视频分析、动作识别和导航等许多应用都是至关重要的。然而，这些技术的健壮性仍然是一个令人担忧的问题，特别是在面对已被证明在许多领域欺骗最先进的深度神经网络的对抗性攻击时。令人惊讶的是，场景流网络对此类攻击的健壮性还没有得到彻底的研究。为了解决这个问题，提出的方法旨在通过引入专门为场景流网络量身定做的对抗性白盒攻击来弥合这一差距。实验结果表明，生成的对抗性实例在Kitti和FlyingThings3D数据集上的平均端点误差相对下降高达33.7。研究还揭示了仅以一维或颜色通道中的点云为目标的攻击对平均端点误差的显著影响。分析这些攻击对场景流网络及其二维光流网络变体的成功和失败，表明光流网络具有更高的脆弱性。



## **12. Safety Fine-Tuning at (Almost) No Cost: A Baseline for Vision Large Language Models**

（几乎）免费进行安全微调：Vision大型语言模型的基线 cs.LG

ICML 2024

**SubmitDate**: 2024-06-17    [abs](http://arxiv.org/abs/2402.02207v2) [paper-pdf](http://arxiv.org/pdf/2402.02207v2)

**Authors**: Yongshuo Zong, Ondrej Bohdal, Tingyang Yu, Yongxin Yang, Timothy Hospedales

**Abstract**: Current vision large language models (VLLMs) exhibit remarkable capabilities yet are prone to generate harmful content and are vulnerable to even the simplest jailbreaking attacks. Our initial analysis finds that this is due to the presence of harmful data during vision-language instruction fine-tuning, and that VLLM fine-tuning can cause forgetting of safety alignment previously learned by the underpinning LLM. To address this issue, we first curate a vision-language safe instruction-following dataset VLGuard covering various harmful categories. Our experiments demonstrate that integrating this dataset into standard vision-language fine-tuning or utilizing it for post-hoc fine-tuning effectively safety aligns VLLMs. This alignment is achieved with minimal impact on, or even enhancement of, the models' helpfulness. The versatility of our safety fine-tuning dataset makes it a valuable resource for safety-testing existing VLLMs, training new models or safeguarding pre-trained VLLMs. Empirical results demonstrate that fine-tuned VLLMs effectively reject unsafe instructions and substantially reduce the success rates of several black-box adversarial attacks, which approach zero in many cases. The code and dataset are available at https://github.com/ys-zong/VLGuard.

摘要: 目前的VISION大型语言模型(VLLM)显示出非凡的能力，但很容易产生有害内容，甚至容易受到最简单的越狱攻击。我们的初步分析发现，这是由于视觉语言教学微调过程中存在有害数据，而VLLM微调可能会导致忘记支持LLM之前学习的安全对齐。为了解决这个问题，我们首先策划了一个视觉-语言安全的指令遵循数据集VLGuard，涵盖了各种有害类别。我们的实验表明，将该数据集集成到标准视觉语言微调中或将其用于后自组织微调，可以有效地安全地对齐VLLM。这种对齐是在对模型的帮助最小的影响甚至是增强的情况下实现的。我们的安全微调数据集的多功能性使其成为安全测试现有VLLM、培训新模型或保护预先培训的VLLM的宝贵资源。实验结果表明，微调的VLLM有效地拒绝了不安全的指令，并显著降低了几种黑盒对抗攻击的成功率，这些攻击在许多情况下接近于零。代码和数据集可在https://github.com/ys-zong/VLGuard.上获得



## **13. AdaNCA: Neural Cellular Automata As Adaptors For More Robust Vision Transformer**

AdaNCA：神经元胞自动机作为更稳健的视觉Transformer的适配器 cs.CV

26 pages, 11 figures

**SubmitDate**: 2024-06-17    [abs](http://arxiv.org/abs/2406.08298v2) [paper-pdf](http://arxiv.org/pdf/2406.08298v2)

**Authors**: Yitao Xu, Tong Zhang, Sabine Süsstrunk

**Abstract**: Vision Transformers (ViTs) have demonstrated remarkable performance in image classification tasks, particularly when equipped with local information via region attention or convolutions. While such architectures improve the feature aggregation from different granularities, they often fail to contribute to the robustness of the networks. Neural Cellular Automata (NCA) enables the modeling of global cell representations through local interactions, with its training strategies and architecture design conferring strong generalization ability and robustness against noisy inputs. In this paper, we propose Adaptor Neural Cellular Automata (AdaNCA) for Vision Transformer that uses NCA as plug-in-play adaptors between ViT layers, enhancing ViT's performance and robustness against adversarial samples as well as out-of-distribution inputs. To overcome the large computational overhead of standard NCAs, we propose Dynamic Interaction for more efficient interaction learning. Furthermore, we develop an algorithm for identifying the most effective insertion points for AdaNCA based on our analysis of AdaNCA placement and robustness improvement. With less than a 3% increase in parameters, AdaNCA contributes to more than 10% absolute improvement in accuracy under adversarial attacks on the ImageNet1K benchmark. Moreover, we demonstrate with extensive evaluations across 8 robustness benchmarks and 4 ViT architectures that AdaNCA, as a plug-in-play module, consistently improves the robustness of ViTs.

摘要: 视觉变形器(VITS)在图像分类任务中表现出了显著的性能，特别是当通过区域注意或卷积来配备局部信息时。虽然这样的体系结构从不同的粒度改善了特征聚合，但它们往往无法提高网络的健壮性。神经元胞自动机(NCA)能够通过局部交互对全局细胞表示进行建模，其训练策略和结构设计具有很强的泛化能力和对噪声输入的鲁棒性。在本文中，我们提出了用于视觉转换器的适配器神经元胞自动机(AdaNCA)，它使用NCA作为VIT层之间的即插即用适配器，增强了VIT的性能和对敌意样本和分布外输入的鲁棒性。为了克服标准NCA计算开销大的缺点，我们提出了动态交互来实现更有效的交互学习。此外，基于对AdaNCA布局和健壮性改进的分析，我们提出了一种识别AdaNCA最有效插入点的算法。在参数增加不到3%的情况下，AdaNCA有助于在对ImageNet1K基准的敌意攻击下将准确率绝对提高10%以上。此外，我们通过对8个健壮性基准和4个VIT体系结构的广泛评估，证明了AdaNCA作为一个即插即用模块，持续提高了VIT的健壮性。



## **14. Threat analysis and adversarial model for Smart Grids**

智能电网的威胁分析和对抗模型 cs.CR

Presented at the Workshop on Attackers and Cyber-Crime Operations  (WACCO). More details available at https://wacco-workshop.org

**SubmitDate**: 2024-06-17    [abs](http://arxiv.org/abs/2406.11716v1) [paper-pdf](http://arxiv.org/pdf/2406.11716v1)

**Authors**: Javier Sande Ríos, Jesús Canal Sánchez, Carmen Manzano Hernandez, Sergio Pastrana

**Abstract**: The power grid is a critical infrastructure that allows for the efficient and robust generation, transmission, delivery and consumption of electricity. In the recent years, the physical components have been equipped with computing and network devices, which optimizes the operation and maintenance of the grid. The cyber domain of this smart power grid opens a new plethora of threats, which adds to classical threats on the physical domain. Accordingly, different stakeholders including regulation bodies, industry and academy, are making increasing efforts to provide security mechanisms to mitigate and reduce cyber-risks. Despite these efforts, there have been various cyberattacks that have affected the smart grid, leading in some cases to catastrophic consequences, showcasing that the industry might not be prepared for attacks from high profile adversaries. At the same time, recent work shows a lack of agreement among grid practitioners and academic experts on the feasibility and consequences of academic-proposed threats. This is in part due to inadequate simulation models which do not evaluate threats based on attackers full capabilities and goals. To address this gap, in this work we first analyze the main attack surfaces of the smart grid, and then conduct a threat analysis from the adversarial model perspective, including different levels of knowledge, goals, motivations and capabilities. To validate the model, we provide real-world examples of the potential capabilities by studying known vulnerabilities in critical components, and then analyzing existing cyber-attacks that have affected the smart grid, either directly or indirectly.

摘要: 电网是一种关键的基础设施，它使电力的生产、传输、输送和消费变得高效和强大。近年来，物理部件配备了计算和网络设备，优化了电网的运行和维护。这种智能电网的网络领域开启了新的威胁，这增加了物理领域的传统威胁。因此，包括监管机构、工业界和学术界在内的不同利益攸关方正在加大努力，提供安全机制，以缓解和减少网络风险。尽管做出了这些努力，但仍有各种网络攻击影响了智能电网，在某些情况下导致了灾难性的后果，这表明该行业可能没有准备好应对备受瞩目的对手的攻击。与此同时，最近的工作表明，网格从业者和学术专家对学术提出的威胁的可行性和后果缺乏共识。这在一定程度上是由于模拟模型的不足，这些模型没有根据攻击者的全部能力和目标来评估威胁。为了弥补这一差距，在这项工作中，我们首先分析了智能电网的主要攻击面，然后从对抗模型的角度进行威胁分析，包括不同水平的知识、目标、动机和能力。为了验证该模型，我们通过研究关键组件中的已知漏洞，然后分析直接或间接影响智能电网的现有网络攻击，提供了潜在能力的真实示例。



## **15. Harmonizing Feature Maps: A Graph Convolutional Approach for Enhancing Adversarial Robustness**

协调特征图：增强对抗稳健性的图卷积方法 cs.CV

**SubmitDate**: 2024-06-17    [abs](http://arxiv.org/abs/2406.11576v1) [paper-pdf](http://arxiv.org/pdf/2406.11576v1)

**Authors**: Kejia Zhang, Juanjuan Weng, Junwei Wu, Guoqing Yang, Shaozi Li, Zhiming Luo

**Abstract**: The vulnerability of Deep Neural Networks to adversarial perturbations presents significant security concerns, as the imperceptible perturbations can contaminate the feature space and lead to incorrect predictions. Recent studies have attempted to calibrate contaminated features by either suppressing or over-activating particular channels. Despite these efforts, we claim that adversarial attacks exhibit varying disruption levels across individual channels. Furthermore, we argue that harmonizing feature maps via graph and employing graph convolution can calibrate contaminated features. To this end, we introduce an innovative plug-and-play module called Feature Map-based Reconstructed Graph Convolution (FMR-GC). FMR-GC harmonizes feature maps in the channel dimension to reconstruct the graph, then employs graph convolution to capture neighborhood information, effectively calibrating contaminated features. Extensive experiments have demonstrated the superior performance and scalability of FMR-GC. Moreover, our model can be combined with advanced adversarial training methods to considerably enhance robustness without compromising the model's clean accuracy.

摘要: 深度神经网络对对抗性扰动的脆弱性带来了严重的安全问题，因为不可察觉的扰动可能会污染特征空间并导致错误的预测。最近的研究试图通过抑制或过度激活特定的通道来校准受污染的特征。尽管做出了这些努力，但我们声称，对抗性攻击在各个渠道表现出不同的干扰程度。此外，我们认为，通过图协调特征映射和使用图卷积可以校准受污染的特征。为此，我们引入了一种创新的即插即用模块，称为基于特征映射的重构图形卷积(FMR-GC)。FMR-GC在通道维度上协调特征映射重建图，然后利用图卷积来捕获邻域信息，有效地校准受污染的特征。大量实验表明，FMR-GC具有良好的性能和可扩展性。此外，我们的模型可以与先进的对抗性训练方法相结合，在不影响模型的干净准确性的情况下显著增强稳健性。



## **16. Do Parameters Reveal More than Loss for Membership Inference?**

参数揭示的不仅仅是会员推断的损失吗？ cs.LG

Accepted at High-dimensional Learning Dynamics (HiLD) Workshop, ICML  2024

**SubmitDate**: 2024-06-17    [abs](http://arxiv.org/abs/2406.11544v1) [paper-pdf](http://arxiv.org/pdf/2406.11544v1)

**Authors**: Anshuman Suri, Xiao Zhang, David Evans

**Abstract**: Membership inference attacks aim to infer whether an individual record was used to train a model, serving as a key tool for disclosure auditing. While such evaluations are useful to demonstrate risk, they are computationally expensive and often make strong assumptions about potential adversaries' access to models and training environments, and thus do not provide very tight bounds on leakage from potential attacks. We show how prior claims around black-box access being sufficient for optimal membership inference do not hold for most useful settings such as stochastic gradient descent, and that optimal membership inference indeed requires white-box access. We validate our findings with a new white-box inference attack IHA (Inverse Hessian Attack) that explicitly uses model parameters by taking advantage of computing inverse-Hessian vector products. Our results show that both audits and adversaries may be able to benefit from access to model parameters, and we advocate for further research into white-box methods for membership privacy auditing.

摘要: 成员资格推断攻击旨在推断个人记录是否被用来训练模型，作为披露审计的关键工具。虽然这样的评估有助于显示风险，但它们的计算成本很高，而且通常会对潜在对手访问模型和训练环境做出强有力的假设，因此不会对潜在攻击的泄漏提供非常严格的限制。我们证明了关于黑箱访问的最优成员关系推理的先前声明如何不适用于大多数有用的设置，例如随机梯度下降，而最优成员关系推理确实需要白箱访问。我们使用一种新的白盒推理攻击IHA(逆向Hessian攻击)来验证我们的发现，该攻击通过计算逆向Hessian向量积来显式地使用模型参数。我们的结果表明，审计和对手都可以从访问模型参数中受益，我们主张进一步研究成员隐私审计的白盒方法。



## **17. FullCert: Deterministic End-to-End Certification for Training and Inference of Neural Networks**

FullCert：神经网络训练和推理的确定性端到端认证 cs.LG

**SubmitDate**: 2024-06-17    [abs](http://arxiv.org/abs/2406.11522v1) [paper-pdf](http://arxiv.org/pdf/2406.11522v1)

**Authors**: Tobias Lorenz, Marta Kwiatkowska, Mario Fritz

**Abstract**: Modern machine learning models are sensitive to the manipulation of both the training data (poisoning attacks) and inference data (adversarial examples). Recognizing this issue, the community has developed many empirical defenses against both attacks and, more recently, provable certification methods against inference-time attacks. However, such guarantees are still largely lacking for training-time attacks. In this work, we present FullCert, the first end-to-end certifier with sound, deterministic bounds, which proves robustness against both training-time and inference-time attacks. We first bound all possible perturbations an adversary can make to the training data under the considered threat model. Using these constraints, we bound the perturbations' influence on the model's parameters. Finally, we bound the impact of these parameter changes on the model's prediction, resulting in joint robustness guarantees against poisoning and adversarial examples. To facilitate this novel certification paradigm, we combine our theoretical work with a new open-source library BoundFlow, which enables model training on bounded datasets. We experimentally demonstrate FullCert's feasibility on two different datasets.

摘要: 现代机器学习模型对训练数据(中毒攻击)和推理数据(对抗性例子)的操纵都很敏感。认识到这一问题，社区已经开发了许多针对这两种攻击的经验防御方法，最近还开发了针对推理时间攻击的可证明的认证方法。然而，这样的保障在很大程度上仍然缺乏对训练时间攻击的保障。在这项工作中，我们提出了FullCert，这是第一个端到端证书，具有良好的确定性界，它证明了对训练时间和推理时间攻击的健壮性。我们首先在考虑的威胁模型下限制了对手可以对训练数据进行的所有可能的扰动。利用这些约束，我们限制了扰动对模型参数的影响。最后，我们结合了这些参数变化对模型预测的影响，从而对中毒和敌意示例提供了联合稳健性保证。为了促进这一新的认证范式，我们将我们的理论工作与新的开源库BordFlow相结合，该库能够对有界数据集进行模型训练。我们在两个不同的数据集上实验验证了FullCert的可行性。



## **18. Obfuscating IoT Device Scanning Activity via Adversarial Example Generation**

通过对抗示例生成混淆物联网设备扫描活动 cs.CR

**SubmitDate**: 2024-06-17    [abs](http://arxiv.org/abs/2406.11515v1) [paper-pdf](http://arxiv.org/pdf/2406.11515v1)

**Authors**: Haocong Li, Yaxin Zhang, Long Cheng, Wenjia Niu, Haining Wang, Qiang Li

**Abstract**: Nowadays, attackers target Internet of Things (IoT) devices for security exploitation, and search engines for devices and services compromise user privacy, including IP addresses, open ports, device types, vendors, and products.Typically, application banners are used to recognize IoT device profiles during network measurement and reconnaissance. In this paper, we propose a novel approach to obfuscating IoT device banners (BANADV) based on adversarial examples. The key idea is to explore the susceptibility of fingerprinting techniques to a slight perturbation of an IoT device banner. By modifying device banners, BANADV disrupts the collection of IoT device profiles. To validate the efficacy of BANADV, we conduct a set of experiments. Our evaluation results show that adversarial examples can spoof state-of-the-art fingerprinting techniques, including learning- and matching-based approaches. We further provide a detailed analysis of the weakness of learning-based/matching-based fingerprints to carefully crafted samples. Overall, the innovations of BANADV lie in three aspects: (1) it utilizes an IoT-related semantic space and a visual similarity space to locate available manipulating perturbations of IoT banners; (2) it achieves at least 80\% success rate for spoofing IoT scanning techniques; and (3) it is the first to utilize adversarial examples of IoT banners in network measurement and reconnaissance.

摘要: 如今，攻击者将物联网(IoT)设备作为安全攻击的目标，针对设备和服务的搜索引擎损害了用户隐私，包括IP地址、开放端口、设备类型、供应商和产品。通常，应用程序横幅用于在网络测量和侦察期间识别物联网设备配置文件。本文提出了一种基于敌意实例的混淆物联网设备横幅的新方法(BANADV)。其关键思想是探索指纹技术对物联网设备横幅轻微扰动的敏感度。通过修改设备横幅，BANADV扰乱了物联网设备配置文件的收集。为了验证BANADV的有效性，我们进行了一系列实验。我们的评估结果表明，敌意例子可以欺骗最先进的指纹识别技术，包括基于学习和匹配的方法。我们进一步详细分析了基于学习/基于匹配的指纹对精心制作的样本的弱点。总的来说，BANADV的创新之处在于三个方面：(1)利用与物联网相关的语义空间和视觉相似性空间来定位可用的物联网横幅操纵扰动；(2)对欺骗物联网扫描技术的成功率至少达到80%；(3)首次利用物联网横幅的对抗性例子进行网络测量和侦察。



## **19. Adapters Mixup: Mixing Parameter-Efficient Adapters to Enhance the Adversarial Robustness of Fine-tuned Pre-trained Text Classifiers**

Adapters Mixup：混合参数高效的适配器，以增强微调预训练文本分类器的对抗鲁棒性 cs.CL

**SubmitDate**: 2024-06-17    [abs](http://arxiv.org/abs/2401.10111v2) [paper-pdf](http://arxiv.org/pdf/2401.10111v2)

**Authors**: Tuc Nguyen, Thai Le

**Abstract**: Existing works show that augmenting the training data of pre-trained language models (PLMs) for classification tasks fine-tuned via parameter-efficient fine-tuning methods (PEFT) using both clean and adversarial examples can enhance their robustness under adversarial attacks. However, this adversarial training paradigm often leads to performance degradation on clean inputs and requires frequent re-training on the entire data to account for new, unknown attacks. To overcome these challenges while still harnessing the benefits of adversarial training and the efficiency of PEFT, this work proposes a novel approach, called AdpMixup, that combines two paradigms: (1) fine-tuning through adapters and (2) adversarial augmentation via mixup to dynamically leverage existing knowledge from a set of pre-known attacks for robust inference. Intuitively, AdpMixup fine-tunes PLMs with multiple adapters with both clean and pre-known adversarial examples and intelligently mixes them up in different ratios during prediction. Our experiments show AdpMixup achieves the best trade-off between training efficiency and robustness under both pre-known and unknown attacks, compared to existing baselines on five downstream tasks across six varied black-box attacks and 2 PLMs. All source code will be available.

摘要: 已有的工作表明，通过参数高效微调方法(PEFT)对分类任务的预训练语言模型(PLM)的训练数据进行扩充，并使用干净的和对抗性的例子进行微调，可以增强它们在对手攻击下的健壮性。然而，这种对抗性训练模式经常导致干净输入的性能下降，并且需要频繁地对整个数据进行重新训练，以应对新的、未知的攻击。为了在克服这些挑战的同时仍然利用对抗训练的好处和PEFT的效率，本工作提出了一种名为AdpMixup的新方法，该方法结合了两种范例：(1)通过适配器进行微调；(2)通过混合来动态利用一组预先已知攻击的现有知识进行稳健推理。直观地说，AdpMixup微调了带有多个适配器的PLM，包括干净的和预先知道的对手例子，并在预测期间智能地将它们混合在不同的比例中。实验表明，AdpMixup在已知和未知攻击下的训练效率和健壮性之间取得了最好的折衷，与现有的6种不同的黑盒攻击和2种PLM下的五个下游任务的基线相比。所有源代码都将可用。



## **20. ART: Automatic Red-teaming for Text-to-Image Models to Protect Benign Users**

ART：文本到图像模型的自动红色团队以保护良性用户 cs.CR

**SubmitDate**: 2024-06-17    [abs](http://arxiv.org/abs/2405.19360v2) [paper-pdf](http://arxiv.org/pdf/2405.19360v2)

**Authors**: Guanlin Li, Kangjie Chen, Shudong Zhang, Jie Zhang, Tianwei Zhang

**Abstract**: Large-scale pre-trained generative models are taking the world by storm, due to their abilities in generating creative content. Meanwhile, safeguards for these generative models are developed, to protect users' rights and safety, most of which are designed for large language models. Existing methods primarily focus on jailbreak and adversarial attacks, which mainly evaluate the model's safety under malicious prompts. Recent work found that manually crafted safe prompts can unintentionally trigger unsafe generations. To further systematically evaluate the safety risks of text-to-image models, we propose a novel Automatic Red-Teaming framework, ART. Our method leverages both vision language model and large language model to establish a connection between unsafe generations and their prompts, thereby more efficiently identifying the model's vulnerabilities. With our comprehensive experiments, we reveal the toxicity of the popular open-source text-to-image models. The experiments also validate the effectiveness, adaptability, and great diversity of ART. Additionally, we introduce three large-scale red-teaming datasets for studying the safety risks associated with text-to-image models. Datasets and models can be found in https://github.com/GuanlinLee/ART.

摘要: 由于具有创造内容的能力，大规模的预先训练的生成性模型正在席卷世界。同时，为了保护用户的权利和安全，制定了对这些生成模型的保障措施，其中大部分是为大型语言模型设计的。现有的方法主要针对越狱和对抗性攻击，主要是在恶意提示下对模型的安全性进行评估。最近的研究发现，手动创建的安全提示可能会无意中引发不安全的世代。为了进一步系统地评估文本到图像模型的安全风险，我们提出了一个新的自动红色团队框架ART。我们的方法利用视觉语言模型和大型语言模型来建立不安全生成及其提示之间的联系，从而更有效地识别模型的漏洞。通过我们的综合实验，我们揭示了流行的开源文本到图像模型的毒性。实验也验证了ART的有效性、适应性和多样性。此外，我们还介绍了三个大型红团队数据集，用于研究与文本到图像模型相关的安全风险。数据集和模型可在https://github.com/GuanlinLee/ART.中找到



## **21. $\texttt{MoE-RBench}$: Towards Building Reliable Language Models with Sparse Mixture-of-Experts**

$\textttt {MoE-RBench}$：利用稀疏专家混合构建可靠的语言模型 cs.LG

9 pages, 8 figures, camera ready on ICML2024

**SubmitDate**: 2024-06-17    [abs](http://arxiv.org/abs/2406.11353v1) [paper-pdf](http://arxiv.org/pdf/2406.11353v1)

**Authors**: Guanjie Chen, Xinyu Zhao, Tianlong Chen, Yu Cheng

**Abstract**: Mixture-of-Experts (MoE) has gained increasing popularity as a promising framework for scaling up large language models (LLMs). However, the reliability assessment of MoE lags behind its surging applications. Moreover, when transferred to new domains such as in fine-tuning MoE models sometimes underperform their dense counterparts. Motivated by the research gap and counter-intuitive phenomenon, we propose $\texttt{MoE-RBench}$, the first comprehensive assessment of SMoE reliability from three aspects: $\textit{(i)}$ safety and hallucination, $\textit{(ii)}$ resilience to adversarial attacks, and $\textit{(iii)}$ out-of-distribution robustness. Extensive models and datasets are tested to compare the MoE to dense networks from these reliability dimensions. Our empirical observations suggest that with appropriate hyperparameters, training recipes, and inference techniques, we can build the MoE model more reliably than the dense LLM. In particular, we find that the robustness of SMoE is sensitive to the basic training settings. We hope that this study can provide deeper insights into how to adapt the pre-trained MoE model to other tasks with higher-generation security, quality, and stability. Codes are available at https://github.com/UNITES-Lab/MoE-RBench

摘要: 专家混合(MOE)作为一种有前途的扩展大型语言模型(LLM)的框架已经越来越受欢迎。然而，MOE的可靠性评估落后于其激增的应用。此外，当转移到新的领域时，例如在微调的MOE模型中，有时表现不如密集的对应模型。受研究空白和反直觉现象的启发，我们首次从三个方面对SMOE的可靠性进行了全面的评估：安全和幻觉，对对手攻击的恢复能力，以及分布外的稳健性。测试了大量的模型和数据集，以从这些可靠性维度将MoE与密集网络进行比较。我们的经验观察表明，通过适当的超参数、训练配方和推理技术，我们可以建立比密集的LLM更可靠的MOE模型。特别是，我们发现SMOE的稳健性对基本训练设置很敏感。我们希望这项研究能够为如何将预先训练的MOE模型适应于具有更高一代安全性、质量和稳定性的其他任务提供更深层次的见解。有关代码，请访问https://github.com/UNITES-Lab/MoE-RBench



## **22. Byzantine Robust Cooperative Multi-Agent Reinforcement Learning as a Bayesian Game**

作为Bayesian游戏的拜占庭鲁棒合作多智能体强化学习 cs.GT

**SubmitDate**: 2024-06-17    [abs](http://arxiv.org/abs/2305.12872v3) [paper-pdf](http://arxiv.org/pdf/2305.12872v3)

**Authors**: Simin Li, Jun Guo, Jingqiao Xiu, Ruixiao Xu, Xin Yu, Jiakai Wang, Aishan Liu, Yaodong Yang, Xianglong Liu

**Abstract**: In this study, we explore the robustness of cooperative multi-agent reinforcement learning (c-MARL) against Byzantine failures, where any agent can enact arbitrary, worst-case actions due to malfunction or adversarial attack. To address the uncertainty that any agent can be adversarial, we propose a Bayesian Adversarial Robust Dec-POMDP (BARDec-POMDP) framework, which views Byzantine adversaries as nature-dictated types, represented by a separate transition. This allows agents to learn policies grounded on their posterior beliefs about the type of other agents, fostering collaboration with identified allies and minimizing vulnerability to adversarial manipulation. We define the optimal solution to the BARDec-POMDP as an ex post robust Bayesian Markov perfect equilibrium, which we proof to exist and weakly dominates the equilibrium of previous robust MARL approaches. To realize this equilibrium, we put forward a two-timescale actor-critic algorithm with almost sure convergence under specific conditions. Experimentation on matrix games, level-based foraging and StarCraft II indicate that, even under worst-case perturbations, our method successfully acquires intricate micromanagement skills and adaptively aligns with allies, demonstrating resilience against non-oblivious adversaries, random allies, observation-based attacks, and transfer-based attacks.

摘要: 在这项研究中，我们探讨了协作多智能体强化学习(c-Marl)对拜占庭故障的稳健性，在拜占庭故障中，任何智能体都可以由于故障或对手攻击而执行任意的、最坏的操作。为了解决任何智能体都可能是对抗性的不确定性，我们提出了一种贝叶斯对抗性鲁棒DEC-POMDP(BARDEC-POMDP)框架，该框架将拜占庭对手视为自然决定的类型，由单独的转换表示。这使代理能够基于他们对其他代理类型的后验信念来学习策略，促进与确定的盟友的合作，并将受到对手操纵的脆弱性降至最低。我们将BARDEC-POMDP的最优解定义为一个事后稳健的贝叶斯马尔可夫完全均衡，并证明了它的存在，并且弱控制了以前的稳健Marl方法的均衡。为了实现这一均衡，我们提出了一个在特定条件下几乎必然收敛的双时间尺度的行动者-批评者算法。在矩阵游戏、基于关卡的觅食和星际争霸II上的实验表明，即使在最坏的情况下，我们的方法也成功地获得了复杂的微观管理技能，并自适应地与盟友结盟，展示了对非遗忘对手、随机盟友、基于观察的攻击和基于转移的攻击的弹性。



## **23. Optimal Attack and Defense for Reinforcement Learning**

强化学习的最佳攻击和防御 cs.LG

**SubmitDate**: 2024-06-17    [abs](http://arxiv.org/abs/2312.00198v2) [paper-pdf](http://arxiv.org/pdf/2312.00198v2)

**Authors**: Jeremy McMahan, Young Wu, Xiaojin Zhu, Qiaomin Xie

**Abstract**: To ensure the usefulness of Reinforcement Learning (RL) in real systems, it is crucial to ensure they are robust to noise and adversarial attacks. In adversarial RL, an external attacker has the power to manipulate the victim agent's interaction with the environment. We study the full class of online manipulation attacks, which include (i) state attacks, (ii) observation attacks (which are a generalization of perceived-state attacks), (iii) action attacks, and (iv) reward attacks. We show the attacker's problem of designing a stealthy attack that maximizes its own expected reward, which often corresponds to minimizing the victim's value, is captured by a Markov Decision Process (MDP) that we call a meta-MDP since it is not the true environment but a higher level environment induced by the attacked interaction. We show that the attacker can derive optimal attacks by planning in polynomial time or learning with polynomial sample complexity using standard RL techniques. We argue that the optimal defense policy for the victim can be computed as the solution to a stochastic Stackelberg game, which can be further simplified into a partially-observable turn-based stochastic game (POTBSG). Neither the attacker nor the victim would benefit from deviating from their respective optimal policies, thus such solutions are truly robust. Although the defense problem is NP-hard, we show that optimal Markovian defenses can be computed (learned) in polynomial time (sample complexity) in many scenarios.

摘要: 为了确保强化学习(RL)在实际系统中的有效性，确保它们对噪声和对手攻击具有健壮性是至关重要的。在对抗性RL中，外部攻击者有权操纵受害者代理与环境的交互。我们研究了所有类型的在线操纵攻击，包括(I)状态攻击，(Ii)观察攻击(它是感知状态攻击的推广)，(Iii)动作攻击，和(Iv)奖励攻击。我们展示了攻击者设计最大化自身期望回报的隐形攻击的问题，这通常对应于最小化受害者的价值，被马尔可夫决策过程(MDP)捕获，我们称之为元MDP，因为它不是真正的环境，而是由攻击交互引起的更高级别的环境。我们证明了攻击者可以通过在多项式时间内进行规划或使用标准RL技术以多项式样本复杂性学习来获得最优攻击。我们认为，受害者的最优防御策略可以归结为一个随机Stackelberg博弈的解，它可以进一步简化为一个部分可观测的基于回合的随机博弈(POTBSG)。攻击者和受害者都不会从偏离各自的最优策略中受益，因此这样的解决方案是真正可靠的。虽然防御问题是NP难的，但我们证明了在许多情况下，最优马尔可夫防御可以在多项式时间(样本复杂性)内计算(学习)。



## **24. Adversarial Style Augmentation via Large Language Model for Robust Fake News Detection**

通过大语言模型进行对抗风格增强以实现稳健的假新闻检测 cs.CL

8 pages

**SubmitDate**: 2024-06-17    [abs](http://arxiv.org/abs/2406.11260v1) [paper-pdf](http://arxiv.org/pdf/2406.11260v1)

**Authors**: Sungwon Park, Sungwon Han, Meeyoung Cha

**Abstract**: The spread of fake news negatively impacts individuals and is regarded as a significant social challenge that needs to be addressed. A number of algorithmic and insightful features have been identified for detecting fake news. However, with the recent LLMs and their advanced generation capabilities, many of the detectable features (e.g., style-conversion attacks) can be altered, making it more challenging to distinguish from real news. This study proposes adversarial style augmentation, AdStyle, to train a fake news detector that remains robust against various style-conversion attacks. Our model's key mechanism is the careful use of LLMs to automatically generate a diverse yet coherent range of style-conversion attack prompts. This improves the generation of prompts that are particularly difficult for the detector to handle. Experiments show that our augmentation strategy improves robustness and detection performance when tested on fake news benchmark datasets.

摘要: 假新闻的传播对个人产生负面影响，被视为需要解决的重大社会挑战。已经确定了许多算法和有洞察力的功能来检测假新闻。然而，随着最近的LLM及其先进一代能力，许多可检测的特征（例如，风格转换攻击）可以被更改，使其与真实新闻区分起来更具挑战性。这项研究提出了对抗性风格增强AdStyle来训练一个假新闻检测器，该检测器在对抗各种风格转换攻击时保持稳健。我们模型的关键机制是仔细使用LLM来自动生成多样化但连贯的风格转换攻击提示。这改善了检测器特别难以处理的提示的生成。实验表明，当在假新闻基准数据集上进行测试时，我们的增强策略提高了鲁棒性和检测性能。



## **25. The Benefits of Power Regularization in Cooperative Reinforcement Learning**

合作强化学习中功率正规化的好处 cs.LG

**SubmitDate**: 2024-06-17    [abs](http://arxiv.org/abs/2406.11240v1) [paper-pdf](http://arxiv.org/pdf/2406.11240v1)

**Authors**: Michelle Li, Michael Dennis

**Abstract**: Cooperative Multi-Agent Reinforcement Learning (MARL) algorithms, trained only to optimize task reward, can lead to a concentration of power where the failure or adversarial intent of a single agent could decimate the reward of every agent in the system. In the context of teams of people, it is often useful to explicitly consider how power is distributed to ensure no person becomes a single point of failure. Here, we argue that explicitly regularizing the concentration of power in cooperative RL systems can result in systems which are more robust to single agent failure, adversarial attacks, and incentive changes of co-players. To this end, we define a practical pairwise measure of power that captures the ability of any co-player to influence the ego agent's reward, and then propose a power-regularized objective which balances task reward and power concentration. Given this new objective, we show that there always exists an equilibrium where every agent is playing a power-regularized best-response balancing power and task reward. Moreover, we present two algorithms for training agents towards this power-regularized objective: Sample Based Power Regularization (SBPR), which injects adversarial data during training; and Power Regularization via Intrinsic Motivation (PRIM), which adds an intrinsic motivation to regulate power to the training objective. Our experiments demonstrate that both algorithms successfully balance task reward and power, leading to lower power behavior than the baseline of task-only reward and avoid catastrophic events in case an agent in the system goes off-policy.

摘要: 协作多智能体强化学习(MAIL)算法只被训练为优化任务奖励，可能导致权力集中，其中单个智能体的失败或敌对意图可能会摧毁系统中每个智能体的奖励。在团队的背景下，明确考虑权力是如何分配的，以确保没有人成为单一的失败点，这通常是有用的。在这里，我们认为，明确地规范合作RL系统中的权力集中可以导致系统对单智能体失败、对手攻击和合作参与者的激励变化具有更强的鲁棒性。为此，我们定义了一个实用的两两权力度量，该度量捕捉了任何合作参与者影响自我代理奖励的能力，然后提出了一个平衡任务奖励和权力集中的权力正规化目标。在给定这一新目标的情况下，我们证明了始终存在一个均衡，其中每个智能体都在扮演一个权值正则化的最佳响应-权衡权力和任务报酬。此外，我们还提出了两种算法来训练智能体，以实现这一功率正则化目标：基于样本的功率正则化(SBPR)，它在训练过程中注入对抗性数据；以及通过内在动机的功率正则化(PRIM)，它为训练目标增加了调节功率的内在动机。我们的实验表明，两种算法都成功地平衡了任务奖励和功率，导致功率行为低于仅任务奖励的基线，并避免了系统中某个代理偏离策略时的灾难性事件。



## **26. Highlighting the Safety Concerns of Deploying LLMs/VLMs in Robotics**

强调在机器人技术中部署LLM/VLM的安全问题 cs.RO

**SubmitDate**: 2024-06-16    [abs](http://arxiv.org/abs/2402.10340v4) [paper-pdf](http://arxiv.org/pdf/2402.10340v4)

**Authors**: Xiyang Wu, Souradip Chakraborty, Ruiqi Xian, Jing Liang, Tianrui Guan, Fuxiao Liu, Brian M. Sadler, Dinesh Manocha, Amrit Singh Bedi

**Abstract**: In this paper, we highlight the critical issues of robustness and safety associated with integrating large language models (LLMs) and vision-language models (VLMs) into robotics applications. Recent works focus on using LLMs and VLMs to improve the performance of robotics tasks, such as manipulation and navigation. Despite these improvements, analyzing the safety of such systems remains underexplored yet extremely critical. LLMs and VLMs are highly susceptible to adversarial inputs, prompting a significant inquiry into the safety of robotic systems. This concern is important because robotics operate in the physical world where erroneous actions can result in severe consequences. This paper explores this issue thoroughly, presenting a mathematical formulation of potential attacks on LLM/VLM-based robotic systems and offering experimental evidence of the safety challenges. Our empirical findings highlight a significant vulnerability: simple modifications to the input can drastically reduce system effectiveness. Specifically, our results demonstrate an average performance deterioration of 19.4% under minor input prompt modifications and a more alarming 29.1% under slight perceptual changes. These findings underscore the urgent need for robust countermeasures to ensure the safe and reliable deployment of advanced LLM/VLM-based robotic systems.

摘要: 在这篇文章中，我们强调了与将大语言模型(LLM)和视觉语言模型(VLM)集成到机器人应用中相关的健壮性和安全性的关键问题。最近的工作集中在使用LLMS和VLMS来提高机器人任务的性能，如操纵和导航。尽管有了这些改进，分析这类系统的安全性仍然没有得到充分的探索，但仍然非常关键。LLM和VLM非常容易受到敌意输入的影响，这促使人们对机器人系统的安全性进行了重大调查。这一担忧很重要，因为机器人是在物理世界中运行的，在那里错误的行动可能会导致严重的后果。本文对这一问题进行了深入的探讨，给出了对基于LLM/VLM的机器人系统的潜在攻击的数学公式，并提供了安全挑战的实验证据。我们的经验发现突显了一个重大的脆弱性：对输入的简单修改可能会极大地降低系统效率。具体地说，我们的结果显示，在微小的输入提示修改下，性能平均下降了19.4%，而在轻微的感知变化下，性能下降了29.1%。这些发现突显了迫切需要强有力的对策，以确保安全可靠地部署先进的基于LLM/VLM的机器人系统。



## **27. garak: A Framework for Security Probing Large Language Models**

garak：大型语言模型安全探测框架 cs.CL

https://garak.ai

**SubmitDate**: 2024-06-16    [abs](http://arxiv.org/abs/2406.11036v1) [paper-pdf](http://arxiv.org/pdf/2406.11036v1)

**Authors**: Leon Derczynski, Erick Galinkin, Jeffrey Martin, Subho Majumdar, Nanna Inie

**Abstract**: As Large Language Models (LLMs) are deployed and integrated into thousands of applications, the need for scalable evaluation of how models respond to adversarial attacks grows rapidly. However, LLM security is a moving target: models produce unpredictable output, are constantly updated, and the potential adversary is highly diverse: anyone with access to the internet and a decent command of natural language. Further, what constitutes a security weak in one context may not be an issue in a different context; one-fits-all guardrails remain theoretical. In this paper, we argue that it is time to rethink what constitutes ``LLM security'', and pursue a holistic approach to LLM security evaluation, where exploration and discovery of issues are central. To this end, this paper introduces garak (Generative AI Red-teaming and Assessment Kit), a framework which can be used to discover and identify vulnerabilities in a target LLM or dialog system. garak probes an LLM in a structured fashion to discover potential vulnerabilities. The outputs of the framework describe a target model's weaknesses, contribute to an informed discussion of what composes vulnerabilities in unique contexts, and can inform alignment and policy discussions for LLM deployment.

摘要: 随着大型语言模型(LLM)的部署和集成到数以千计的应用程序中，对模型如何响应对手攻击的可扩展评估的需求迅速增长。然而，LLM安全是一个不断变化的目标：模型产生不可预测的输出，不断更新，潜在对手高度多样化：任何人都可以访问互联网，并相当熟练地掌握自然语言。此外，在一种情况下，什么构成安全薄弱，在另一种情况下可能不是问题；一刀切的护栏仍然是理论上的。在这篇文章中，我们认为现在是时候重新思考什么是“LLM安全”，并追求一种全面的方法来进行LLM安全评估，其中探索和发现问题是核心。为此，本文介绍了GARAK(生成性人工智能红团队和评估工具包)，这是一个可以用来发现和识别目标LLM或对话系统中的漏洞的框架。Garak以结构化方式探测LLM，以发现潜在漏洞。该框架的输出描述了目标模型的弱点，有助于对在特定环境中构成漏洞的因素进行明智的讨论，并可以为LLM部署的调整和策略讨论提供信息。



## **28. JailbreakBench: An Open Robustness Benchmark for Jailbreaking Large Language Models**

越狱长凳：越狱大型语言模型的开放鲁棒性基准 cs.CR

JailbreakBench v1.0: more attack artifacts, more test-time defenses,  a more accurate jailbreak judge (Llama-3-70B with a custom prompt), a larger  dataset of human preferences for selecting a jailbreak judge (300 examples),  an over-refusal evaluation dataset (100 benign/borderline behaviors), a  semantic refusal judge based on Llama-3-8B

**SubmitDate**: 2024-06-16    [abs](http://arxiv.org/abs/2404.01318v3) [paper-pdf](http://arxiv.org/pdf/2404.01318v3)

**Authors**: Patrick Chao, Edoardo Debenedetti, Alexander Robey, Maksym Andriushchenko, Francesco Croce, Vikash Sehwag, Edgar Dobriban, Nicolas Flammarion, George J. Pappas, Florian Tramer, Hamed Hassani, Eric Wong

**Abstract**: Jailbreak attacks cause large language models (LLMs) to generate harmful, unethical, or otherwise objectionable content. Evaluating these attacks presents a number of challenges, which the current collection of benchmarks and evaluation techniques do not adequately address. First, there is no clear standard of practice regarding jailbreaking evaluation. Second, existing works compute costs and success rates in incomparable ways. And third, numerous works are not reproducible, as they withhold adversarial prompts, involve closed-source code, or rely on evolving proprietary APIs. To address these challenges, we introduce JailbreakBench, an open-sourced benchmark with the following components: (1) an evolving repository of state-of-the-art adversarial prompts, which we refer to as jailbreak artifacts; (2) a jailbreaking dataset comprising 100 behaviors -- both original and sourced from prior work -- which align with OpenAI's usage policies; (3) a standardized evaluation framework at https://github.com/JailbreakBench/jailbreakbench that includes a clearly defined threat model, system prompts, chat templates, and scoring functions; and (4) a leaderboard at https://jailbreakbench.github.io/ that tracks the performance of attacks and defenses for various LLMs. We have carefully considered the potential ethical implications of releasing this benchmark, and believe that it will be a net positive for the community.

摘要: 越狱攻击会导致大型语言模型(LLM)生成有害、不道德或令人反感的内容。评估这些攻击带来了许多挑战，目前收集的基准和评估技术没有充分解决这些挑战。首先，关于越狱评估没有明确的实践标准。其次，现有的工作以无与伦比的方式计算成本和成功率。第三，许多作品是不可复制的，因为它们保留了对抗性提示，涉及封闭源代码，或者依赖于不断发展的专有API。为了应对这些挑战，我们引入了JailBreak，这是一个开源的基准测试，包括以下组件：(1)一个不断发展的最新对手提示库，我们称之为越狱人工制品；(2)一个包含100种行为的越狱数据集--既有原始的，也有源自以前工作的--与OpenAI的使用策略保持一致；(3)https://github.com/JailbreakBench/jailbreakbench的标准化评估框架，包括明确定义的威胁模型、系统提示、聊天模板和评分功能；以及(4)https://jailbreakbench.github.io/的排行榜，跟踪各种LLM的攻击和防御性能。我们已仔细考虑发布这一基准的潜在道德影响，并相信它将为社会带来净积极的影响。



## **29. Adversarial Illusions in Multi-Modal Embeddings**

多模式嵌入中的对抗幻象 cs.CR

In USENIX Security'24

**SubmitDate**: 2024-06-16    [abs](http://arxiv.org/abs/2308.11804v4) [paper-pdf](http://arxiv.org/pdf/2308.11804v4)

**Authors**: Tingwei Zhang, Rishi Jha, Eugene Bagdasaryan, Vitaly Shmatikov

**Abstract**: Multi-modal embeddings encode texts, images, thermal images, sounds, and videos into a single embedding space, aligning representations across different modalities (e.g., associate an image of a dog with a barking sound). In this paper, we show that multi-modal embeddings can be vulnerable to an attack we call "adversarial illusions." Given an image or a sound, an adversary can perturb it to make its embedding close to an arbitrary, adversary-chosen input in another modality.   These attacks are cross-modal and targeted: the adversary can align any image or sound with any target of his choice. Adversarial illusions exploit proximity in the embedding space and are thus agnostic to downstream tasks and modalities, enabling a wholesale compromise of current and future tasks, as well as modalities not available to the adversary. Using ImageBind and AudioCLIP embeddings, we demonstrate how adversarially aligned inputs, generated without knowledge of specific downstream tasks, mislead image generation, text generation, zero-shot classification, and audio retrieval.   We investigate transferability of illusions across different embeddings and develop a black-box version of our method that we use to demonstrate the first adversarial alignment attack on Amazon's commercial, proprietary Titan embedding. Finally, we analyze countermeasures and evasion attacks.

摘要: 多模式嵌入将文本、图像、热像、声音和视频编码到单个嵌入空间中，跨不同模式对齐表示(例如，将狗的图像与犬吠声相关联)。在这篇文章中，我们证明了多模式嵌入可能容易受到一种我们称为“对抗错觉”的攻击。在给定图像或声音的情况下，敌手可以对其进行干扰，使其嵌入到另一种形式中，接近对手选择的任意输入。这些攻击是跨模式的和有针对性的：对手可以将任何图像或声音与他选择的任何目标对齐。对抗性错觉利用嵌入空间中的邻近性，因此对下游任务和模式是不可知的，从而能够对当前和未来的任务以及对手无法获得的模式进行大规模妥协。使用ImageBind和AudioCLIP嵌入，我们演示了在不知道特定下游任务的情况下生成的恶意对齐输入如何误导图像生成、文本生成、零镜头分类和音频检索。我们调查了错觉在不同嵌入中的可转移性，并开发了我们方法的黑盒版本，用于演示对亚马逊商业、专有的Titan嵌入的第一次敌意对齐攻击。最后，分析了相应的对策和规避攻击。



## **30. ATM: Adversarial Tuning Multi-agent System Makes a Robust Retrieval-Augmented Generator**

ATM：对抗性调整多代理系统打造强大的检索增强生成器 cs.CL

18 pages, 7 figures

**SubmitDate**: 2024-06-16    [abs](http://arxiv.org/abs/2405.18111v2) [paper-pdf](http://arxiv.org/pdf/2405.18111v2)

**Authors**: Junda Zhu, Lingyong Yan, Haibo Shi, Dawei Yin, Lei Sha

**Abstract**: Large language models (LLMs) are proven to benefit a lot from retrieval-augmented generation (RAG) in alleviating hallucinations confronted with knowledge-intensive questions. RAG adopts information retrieval techniques to inject external knowledge from semantic-relevant documents as input contexts. However, due to today's Internet being flooded with numerous noisy and fabricating content, it is inevitable that RAG systems are vulnerable to these noises and prone to respond incorrectly. To this end, we propose to optimize the retrieval-augmented Generator with a Adversarial Tuning Multi-agent system (ATM). The ATM steers the Generator to have a robust perspective of useful documents for question answering with the help of an auxiliary Attacker agent. The Generator and the Attacker are tuned adversarially for several iterations. After rounds of multi-agent iterative tuning, the Generator can eventually better discriminate useful documents amongst fabrications. The experimental results verify the effectiveness of ATM and we also observe that the Generator can achieve better performance compared to state-of-the-art baselines.

摘要: 事实证明，大型语言模型(LLM)在缓解面对知识密集型问题时的幻觉方面，从检索增强生成(RAG)中受益匪浅。RAG采用信息检索技术，从与语义相关的文档中注入外部知识作为输入上下文。然而，由于当今的互联网充斥着大量噪声和捏造的内容，RAG系统不可避免地容易受到这些噪声的影响，并容易做出错误的响应。为此，我们提出了用对抗性调谐多智能体系统(ATM)来优化检索增强生成器。ATM引导生成器在辅助攻击者代理的帮助下具有用于问题回答的有用文档的健壮视角。生成器和攻击者被敌对地调整了几次迭代。经过几轮多代理迭代调整后，Generator最终可以更好地区分有用的文档和捏造的文档。实验结果验证了ATM的有效性，并且我们还观察到，与最先进的基线相比，该生成器可以获得更好的性能。



## **31. RWKU: Benchmarking Real-World Knowledge Unlearning for Large Language Models**

RWKU：大型语言模型的现实世界知识学习基准 cs.CL

48 pages, 7 figures, 12 tables

**SubmitDate**: 2024-06-16    [abs](http://arxiv.org/abs/2406.10890v1) [paper-pdf](http://arxiv.org/pdf/2406.10890v1)

**Authors**: Zhuoran Jin, Pengfei Cao, Chenhao Wang, Zhitao He, Hongbang Yuan, Jiachun Li, Yubo Chen, Kang Liu, Jun Zhao

**Abstract**: Large language models (LLMs) inevitably memorize sensitive, copyrighted, and harmful knowledge from the training corpus; therefore, it is crucial to erase this knowledge from the models. Machine unlearning is a promising solution for efficiently removing specific knowledge by post hoc modifying models. In this paper, we propose a Real-World Knowledge Unlearning benchmark (RWKU) for LLM unlearning. RWKU is designed based on the following three key factors: (1) For the task setting, we consider a more practical and challenging unlearning setting, where neither the forget corpus nor the retain corpus is accessible. (2) For the knowledge source, we choose 200 real-world famous people as the unlearning targets and show that such popular knowledge is widely present in various LLMs. (3) For the evaluation framework, we design the forget set and the retain set to evaluate the model's capabilities across various real-world applications. Regarding the forget set, we provide four four membership inference attack (MIA) methods and nine kinds of adversarial attack probes to rigorously test unlearning efficacy. Regarding the retain set, we assess locality and utility in terms of neighbor perturbation, general ability, reasoning ability, truthfulness, factuality, and fluency. We conduct extensive experiments across two unlearning scenarios, two models and six baseline methods and obtain some meaningful findings. We release our benchmark and code publicly at http://rwku-bench.github.io for future work.

摘要: 大型语言模型不可避免地会记住来自训练语料库的敏感、受版权保护和有害的知识；因此，从模型中删除这些知识至关重要。机器遗忘是通过事后修改模型来有效去除特定知识的一种很有前途的解决方案。本文提出了一种用于LLM遗忘的真实世界知识遗忘基准(RWKU)。RWKU的设计基于以下三个关键因素：(1)对于任务设置，我们考虑了一个更实际和更具挑战性的遗忘环境，其中忘记语料库和保留语料库都是不可访问的。(2)在知识源方面，我们选择了200名现实世界名人作为遗忘对象，发现这些流行知识广泛存在于各种学习记忆中。(3)对于评估框架，我们设计了遗忘集和保留集来评估模型在各种实际应用中的能力。对于遗忘集，我们提供了四种成员推理攻击(MIA)方法和九种对抗性攻击探头来严格测试遗忘效果。对于保留集，我们根据邻域扰动、一般能力、推理能力、真实性、真实性和流畅性来评估局部性和效用。我们在两个遗忘场景、两个模型和六个基线方法上进行了广泛的实验，并获得了一些有意义的发现。我们在http://rwku-bench.github.io上公开发布了我们的基准测试和代码，以备将来的工作使用。



## **32. Imperceptible Face Forgery Attack via Adversarial Semantic Mask**

通过对抗性语义面具进行不可感知的人脸伪造攻击 cs.CV

The code is publicly available

**SubmitDate**: 2024-06-16    [abs](http://arxiv.org/abs/2406.10887v1) [paper-pdf](http://arxiv.org/pdf/2406.10887v1)

**Authors**: Decheng Liu, Qixuan Su, Chunlei Peng, Nannan Wang, Xinbo Gao

**Abstract**: With the great development of generative model techniques, face forgery detection draws more and more attention in the related field. Researchers find that existing face forgery models are still vulnerable to adversarial examples with generated pixel perturbations in the global image. These generated adversarial samples still can't achieve satisfactory performance because of the high detectability. To address these problems, we propose an Adversarial Semantic Mask Attack framework (ASMA) which can generate adversarial examples with good transferability and invisibility. Specifically, we propose a novel adversarial semantic mask generative model, which can constrain generated perturbations in local semantic regions for good stealthiness. The designed adaptive semantic mask selection strategy can effectively leverage the class activation values of different semantic regions, and further ensure better attack transferability and stealthiness. Extensive experiments on the public face forgery dataset prove the proposed method achieves superior performance compared with several representative adversarial attack methods. The code is publicly available at https://github.com/clawerO-O/ASMA.

摘要: 随着产生式模型技术的发展，人脸伪造检测越来越受到相关领域的重视。研究人员发现，现有的人脸伪造模型仍然容易受到全局图像中像素扰动的对抗性例子的攻击。这些生成的对抗性样本由于检测率较高，仍然不能达到令人满意的性能。针对这些问题，我们提出了一种对抗性语义掩码攻击框架(ASMA)，该框架能够生成具有良好可转移性和不可见性的对抗性实例。具体地说，我们提出了一种新的对抗性语义掩码生成模型，该模型可以约束局部语义区域产生的扰动，从而获得良好的隐蔽性。所设计的自适应语义掩码选择策略能够有效地利用不同语义区域的类激活值，从而保证更好的攻击可传递性和隐蔽性。在公开人脸伪造数据集上的大量实验表明，与几种典型的对抗性攻击方法相比，该方法取得了更好的性能。该代码可在https://github.com/clawerO-O/ASMA.上公开获得



## **33. SUB-PLAY: Adversarial Policies against Partially Observed Multi-Agent Reinforcement Learning Systems**

SUB-SYS：针对部分观察的多智能体强化学习系统的对抗策略 cs.LG

To appear in the ACM Conference on Computer and Communications  Security (CCS'24), October 14-18, 2024, Salt Lake City, UT, USA

**SubmitDate**: 2024-06-16    [abs](http://arxiv.org/abs/2402.03741v2) [paper-pdf](http://arxiv.org/pdf/2402.03741v2)

**Authors**: Oubo Ma, Yuwen Pu, Linkang Du, Yang Dai, Ruo Wang, Xiaolei Liu, Yingcai Wu, Shouling Ji

**Abstract**: Recent advancements in multi-agent reinforcement learning (MARL) have opened up vast application prospects, such as swarm control of drones, collaborative manipulation by robotic arms, and multi-target encirclement. However, potential security threats during the MARL deployment need more attention and thorough investigation. Recent research reveals that attackers can rapidly exploit the victim's vulnerabilities, generating adversarial policies that result in the failure of specific tasks. For instance, reducing the winning rate of a superhuman-level Go AI to around 20%. Existing studies predominantly focus on two-player competitive environments, assuming attackers possess complete global state observation.   In this study, we unveil, for the first time, the capability of attackers to generate adversarial policies even when restricted to partial observations of the victims in multi-agent competitive environments. Specifically, we propose a novel black-box attack (SUB-PLAY) that incorporates the concept of constructing multiple subgames to mitigate the impact of partial observability and suggests sharing transitions among subpolicies to improve attackers' exploitative ability. Extensive evaluations demonstrate the effectiveness of SUB-PLAY under three typical partial observability limitations. Visualization results indicate that adversarial policies induce significantly different activations of the victims' policy networks. Furthermore, we evaluate three potential defenses aimed at exploring ways to mitigate security threats posed by adversarial policies, providing constructive recommendations for deploying MARL in competitive environments.

摘要: 多智能体强化学习(MAIL)的最新进展为无人机群体控制、机械臂协同操纵、多目标包围等开辟了广阔的应用前景。然而，MAIL部署过程中的潜在安全威胁需要更多的关注和彻底的调查。最近的研究表明，攻击者可以迅速利用受害者的漏洞，生成导致特定任务失败的对抗性策略。例如，将超人级别围棋人工智能的胜率降低到20%左右。现有的研究主要集中在两人竞争环境中，假设攻击者拥有完整的全局状态观测。在这项研究中，我们首次揭示了攻击者即使限于在多智能体竞争环境中对受害者的部分观察也能够生成对抗策略的能力。具体地说，我们提出了一种新的黑盒攻击(子游戏)，它结合了构造多个子博弈的概念来减轻部分可观测性的影响，并建议在子策略之间共享转移以提高攻击者的利用能力。广泛的评估证明了子游戏在三个典型的部分可观测性限制下的有效性。可视化结果表明，对抗性政策导致受害者的政策网络激活显著不同。此外，我们评估了三种潜在的防御措施，旨在探索减轻对抗性政策构成的安全威胁的方法，为在竞争环境中部署Marl提供建设性的建议。



## **34. Mitigating Accuracy-Robustness Trade-off via Balanced Multi-Teacher Adversarial Distillation**

通过平衡的多教师对抗蒸馏缓解准确性与鲁棒性的权衡 cs.LG

Accepted by TPAMI2024

**SubmitDate**: 2024-06-16    [abs](http://arxiv.org/abs/2306.16170v3) [paper-pdf](http://arxiv.org/pdf/2306.16170v3)

**Authors**: Shiji Zhao, Xizhe Wang, Xingxing Wei

**Abstract**: Adversarial Training is a practical approach for improving the robustness of deep neural networks against adversarial attacks. Although bringing reliable robustness, the performance towards clean examples is negatively affected after Adversarial Training, which means a trade-off exists between accuracy and robustness. Recently, some studies have tried to use knowledge distillation methods in Adversarial Training, achieving competitive performance in improving the robustness but the accuracy for clean samples is still limited. In this paper, to mitigate the accuracy-robustness trade-off, we introduce the Balanced Multi-Teacher Adversarial Robustness Distillation (B-MTARD) to guide the model's Adversarial Training process by applying a strong clean teacher and a strong robust teacher to handle the clean examples and adversarial examples, respectively. During the optimization process, to ensure that different teachers show similar knowledge scales, we design the Entropy-Based Balance algorithm to adjust the teacher's temperature and keep the teachers' information entropy consistent. Besides, to ensure that the student has a relatively consistent learning speed from multiple teachers, we propose the Normalization Loss Balance algorithm to adjust the learning weights of different types of knowledge. A series of experiments conducted on three public datasets demonstrate that B-MTARD outperforms the state-of-the-art methods against various adversarial attacks.

摘要: 对抗性训练是提高深层神经网络抗敌意攻击能力的一种实用方法。虽然带来了可靠的稳健性，但经过对抗性训练后，对干净样本的性能会受到负面影响，这意味着在准确性和稳健性之间存在权衡。近年来，一些研究尝试将知识提取方法应用于对抗性训练，在提高鲁棒性方面取得了较好的性能，但对清洁样本的准确率仍然有限。为了缓解准确率和稳健性之间的权衡，我们引入了平衡多教师对抗稳健性蒸馏(B-MTARD)来指导模型的对抗训练过程，分别采用强清洁教师和强健壮教师来处理干净实例和对抗性实例。在优化过程中，为了保证不同教师表现出相似的知识尺度，设计了基于熵的均衡算法来调整教师的温度，保持教师信息熵的一致性。此外，为了确保学生从多个老师那里获得相对一致的学习速度，我们提出了归一化损失平衡算法来调整不同类型知识的学习权重。在三个公开数据集上进行的一系列实验表明，B-MTARD在抵抗各种对抗性攻击方面优于最先进的方法。



## **35. KGPA: Robustness Evaluation for Large Language Models via Cross-Domain Knowledge Graphs**

KGMA：通过跨领域知识图对大型语言模型进行稳健性评估 cs.CL

**SubmitDate**: 2024-06-16    [abs](http://arxiv.org/abs/2406.10802v1) [paper-pdf](http://arxiv.org/pdf/2406.10802v1)

**Authors**: Aihua Pei, Zehua Yang, Shunan Zhu, Ruoxi Cheng, Ju Jia, Lina Wang

**Abstract**: Existing frameworks for assessing robustness of large language models (LLMs) overly depend on specific benchmarks, increasing costs and failing to evaluate performance of LLMs in professional domains due to dataset limitations. This paper proposes a framework that systematically evaluates the robustness of LLMs under adversarial attack scenarios by leveraging knowledge graphs (KGs). Our framework generates original prompts from the triplets of knowledge graphs and creates adversarial prompts by poisoning, assessing the robustness of LLMs through the results of these adversarial attacks. We systematically evaluate the effectiveness of this framework and its modules. Experiments show that adversarial robustness of the ChatGPT family ranks as GPT-4-turbo > GPT-4o > GPT-3.5-turbo, and the robustness of large language models is influenced by the professional domains in which they operate.

摘要: 用于评估大型语言模型（LLM）稳健性的现有框架过度依赖特定的基准，增加了成本，并且由于数据集限制而无法评估LLM在专业领域的性能。本文提出了一个框架，该框架通过利用知识图（KG）系统评估LLM在对抗性攻击场景下的稳健性。我们的框架从知识图的三重组中生成原始提示，并通过中毒创建对抗提示，通过这些对抗攻击的结果评估LLM的稳健性。我们系统地评估该框架及其模块的有效性。实验表明，ChatGPT家族的对抗鲁棒性排名为GPT-4-涡轮> GPT-4 o> GPT-3.5-涡轮，大型语言模型的鲁棒性受到其运行的专业领域的影响。



## **36. Adversarial Math Word Problem Generation**

对抗性数学单词问题生成 cs.CL

Code/data: https://github.com/ruoyuxie/adversarial_mwps_generation

**SubmitDate**: 2024-06-15    [abs](http://arxiv.org/abs/2402.17916v3) [paper-pdf](http://arxiv.org/pdf/2402.17916v3)

**Authors**: Roy Xie, Chengxuan Huang, Junlin Wang, Bhuwan Dhingra

**Abstract**: Large language models (LLMs) have significantly transformed the educational landscape. As current plagiarism detection tools struggle to keep pace with LLMs' rapid advancements, the educational community faces the challenge of assessing students' true problem-solving abilities in the presence of LLMs. In this work, we explore a new paradigm for ensuring fair evaluation -- generating adversarial examples which preserve the structure and difficulty of the original questions aimed for assessment, but are unsolvable by LLMs. Focusing on the domain of math word problems, we leverage abstract syntax trees to structurally generate adversarial examples that cause LLMs to produce incorrect answers by simply editing the numeric values in the problems. We conduct experiments on various open- and closed-source LLMs, quantitatively and qualitatively demonstrating that our method significantly degrades their math problem-solving ability. We identify shared vulnerabilities among LLMs and propose a cost-effective approach to attack high-cost models. Additionally, we conduct automatic analysis to investigate the cause of failure, providing further insights into the limitations of LLMs.

摘要: 大型语言模型(LLM)极大地改变了教育格局。由于目前的抄袭检测工具难以跟上LLMS的快速进步，教育界面临着在LLMS存在的情况下评估学生真正的问题解决能力的挑战。在这项工作中，我们探索了一种确保公平评价的新范式--生成对抗性实例，它保留了用于评价的原始问题的结构和难度，但无法用LLMS解决。聚焦于数学应用题领域，我们利用抽象语法树来结构化地生成对抗性实例，这些实例通过简单地编辑问题中的数值来导致LLMS产生不正确的答案。我们在各种开源和闭源的LLM上进行了实验，定量和定性地证明了我们的方法显著降低了他们的数学问题解决能力。我们识别了LLM之间的共同漏洞，并提出了一种具有成本效益的方法来攻击高成本模型。此外，我们还进行自动分析以调查故障原因，进一步深入了解LLMS的局限性。



## **37. Federated Multi-Armed Bandits Under Byzantine Attacks**

拜占庭攻击下的联邦多武装强盗 cs.LG

**SubmitDate**: 2024-06-15    [abs](http://arxiv.org/abs/2205.04134v2) [paper-pdf](http://arxiv.org/pdf/2205.04134v2)

**Authors**: Artun Saday, İlker Demirel, Yiğit Yıldırım, Cem Tekin

**Abstract**: Multi-armed bandits (MAB) is a sequential decision-making model in which the learner controls the trade-off between exploration and exploitation to maximize its cumulative reward. Federated multi-armed bandits (FMAB) is an emerging framework where a cohort of learners with heterogeneous local models play an MAB game and communicate their aggregated feedback to a server to learn a globally optimal arm. Two key hurdles in FMAB are communication-efficient learning and resilience to adversarial attacks. To address these issues, we study the FMAB problem in the presence of Byzantine clients who can send false model updates threatening the learning process. We analyze the sample complexity and the regret of $\beta$-optimal arm identification. We borrow tools from robust statistics and propose a median-of-means (MoM)-based online algorithm, Fed-MoM-UCB, to cope with Byzantine clients. In particular, we show that if the Byzantine clients constitute less than half of the cohort, the cumulative regret with respect to $\beta$-optimal arms is bounded over time with high probability, showcasing both communication efficiency and Byzantine resilience. We analyze the interplay between the algorithm parameters, a discernibility margin, regret, communication cost, and the arms' suboptimality gaps. We demonstrate Fed-MoM-UCB's effectiveness against the baselines in the presence of Byzantine attacks via experiments.

摘要: 多武装强盗(MAB)是一种序贯决策模型，在该模型中，学习者控制探索和剥削之间的权衡，以最大化其累积回报。联邦多臂强盗(FMAB)是一种新兴的框架，在这种框架中，具有不同本地模型的一群学习者玩MAB游戏，并将他们汇总的反馈传达给服务器，以学习全局最优的ARM。FMAB的两个关键障碍是高效沟通的学习和对对手攻击的适应能力。为了解决这些问题，我们在拜占庭客户端存在的情况下研究FMAB问题，这些客户端可能会发送虚假的模型更新，威胁到学习过程。分析了样本复杂度和最优ARM识别的遗憾。我们借用稳健统计的工具，提出了一种基于均值中位数(MOM)的在线算法FED-MOM-UCB，以应对拜占庭式的客户。特别地，我们证明了如果拜占庭客户端不到队列的一半，关于$\beta$-最优ARM的累积后悔是以很高的概率随时间有界的，展示了通信效率和拜占庭韧性。我们分析了算法参数、可分辨裕度、后悔、通信成本和ARM的次优差距之间的相互影响。我们通过实验证明了在拜占庭攻击存在的情况下，FED-MOM-UCB相对于基线的有效性。



## **38. Trading Devil: Robust backdoor attack via Stochastic investment models and Bayesian approach**

交易魔鬼：通过随机投资模型和Bayesian方法进行强有力的后门攻击 cs.CR

Stochastic investment models and a Bayesian approach to better  modeling of uncertainty : adversarial machine learning or Stochastic market

**SubmitDate**: 2024-06-15    [abs](http://arxiv.org/abs/2406.10719v1) [paper-pdf](http://arxiv.org/pdf/2406.10719v1)

**Authors**: Orson Mengara

**Abstract**: With the growing use of voice-activated systems and speech recognition technologies, the danger of backdoor attacks on audio data has grown significantly. This research looks at a specific type of attack, known as a Stochastic investment-based backdoor attack (MarketBack), in which adversaries strategically manipulate the stylistic properties of audio to fool speech recognition systems. The security and integrity of machine learning models are seriously threatened by backdoor attacks, in order to maintain the reliability of audio applications and systems, the identification of such attacks becomes crucial in the context of audio data. Experimental results demonstrated that MarketBack is feasible to achieve an average attack success rate close to 100% in seven victim models when poisoning less than 1% of the training data.

摘要: 随着语音激活系统和语音识别技术的日益广泛使用，对音频数据进行后门攻击的危险显着增加。这项研究着眼于一种特定类型的攻击，称为基于随机投资的后门攻击（MarketBack），其中对手战略性地操纵音频的风格属性来愚弄语音识别系统。机器学习模型的安全性和完整性受到后门攻击的严重威胁，为了维护音频应用和系统的可靠性，识别此类攻击在音频数据环境中变得至关重要。实验结果表明，当毒害少于1%的训练数据时，MarketBack可以在7个受害者模型中实现接近100%的平均攻击成功率。



## **39. Hijacking Large Language Models via Adversarial In-Context Learning**

通过对抗性上下文学习劫持大型语言模型 cs.LG

**SubmitDate**: 2024-06-15    [abs](http://arxiv.org/abs/2311.09948v2) [paper-pdf](http://arxiv.org/pdf/2311.09948v2)

**Authors**: Yao Qiang, Xiangyu Zhou, Dongxiao Zhu

**Abstract**: In-context learning (ICL) has emerged as a powerful paradigm leveraging LLMs for specific downstream tasks by utilizing labeled examples as demonstrations (demos) in the precondition prompts. Despite its promising performance, ICL suffers from instability with the choice and arrangement of examples. Additionally, crafted adversarial attacks pose a notable threat to the robustness of ICL. However, existing attacks are either easy to detect, rely on external models, or lack specificity towards ICL. This work introduces a novel transferable attack against ICL to address these issues, aiming to hijack LLMs to generate the target response or jailbreak. Our hijacking attack leverages a gradient-based prompt search method to learn and append imperceptible adversarial suffixes to the in-context demos without directly contaminating the user queries. Comprehensive experimental results across different generation and jailbreaking tasks highlight the effectiveness of our hijacking attack, resulting in distracted attention towards adversarial tokens and consequently leading to unwanted target outputs. We also propose a defense strategy against hijacking attacks through the use of extra clean demos, which enhances the robustness of LLMs during ICL. Broadly, this work reveals the significant security vulnerabilities of LLMs and emphasizes the necessity for in-depth studies on their robustness.

摘要: 情境学习(ICL)已经成为一种强大的范式，通过在前提提示中利用标记的示例作为演示(DEMO)，利用LLM来执行特定的下游任务。尽管ICL的表现很有希望，但它在范例的选择和排列上存在不稳定的问题。此外，精心设计的敌意攻击对ICL的健壮性构成了显著的威胁。然而，现有的攻击要么容易检测，要么依赖外部模型，要么缺乏对ICL的特异性。该工作引入了一种针对ICL的新型可转移攻击来解决这些问题，旨在劫持LLM以产生目标响应或越狱。我们的劫持攻击利用一种基于梯度的快速搜索方法来学习并将不可察觉的对抗性后缀添加到上下文演示中，而不会直接污染用户查询。不同代和越狱任务的综合实验结果突出了我们的劫持攻击的有效性，导致注意力分散到对抗性令牌上，从而导致不想要的目标输出。我们还提出了一种通过使用额外的干净演示来防御劫持攻击的策略，从而增强了LLMS在ICL中的健壮性。总的来说，这项工作揭示了LLMS的重大安全漏洞，并强调了深入研究其健壮性的必要性。



## **40. E-SAGE: Explainability-based Defense Against Backdoor Attacks on Graph Neural Networks**

E-SAGE：针对图神经网络后门攻击的基于解释性的防御 cs.CR

**SubmitDate**: 2024-06-15    [abs](http://arxiv.org/abs/2406.10655v1) [paper-pdf](http://arxiv.org/pdf/2406.10655v1)

**Authors**: Dingqiang Yuan, Xiaohua Xu, Lei Yu, Tongchang Han, Rongchang Li, Meng Han

**Abstract**: Graph Neural Networks (GNNs) have recently been widely adopted in multiple domains. Yet, they are notably vulnerable to adversarial and backdoor attacks. In particular, backdoor attacks based on subgraph insertion have been shown to be effective in graph classification tasks while being stealthy, successfully circumventing various existing defense methods. In this paper, we propose E-SAGE, a novel approach to defending GNN backdoor attacks based on explainability. We find that the malicious edges and benign edges have significant differences in the importance scores for explainability evaluation. Accordingly, E-SAGE adaptively applies an iterative edge pruning process on the graph based on the edge scores. Through extensive experiments, we demonstrate the effectiveness of E-SAGE against state-of-the-art graph backdoor attacks in different attack settings. In addition, we investigate the effectiveness of E-SAGE against adversarial attacks.

摘要: 图形神经网络（GNN）最近在多个领域被广泛采用。然而，它们特别容易受到对抗和后门攻击。特别是，基于子图插入的后门攻击已被证明在图分类任务中有效，同时是隐蔽的，成功规避了各种现有的防御方法。在本文中，我们提出了E-SAGE，这是一种基于可解释性防御GNN后门攻击的新型方法。我们发现恶意边和良性边在可解释性评估的重要性分数上存在显着差异。因此，E-SAGE根据边得分自适应地对图应用迭代边修剪过程。通过大量实验，我们展示了E-SAGE在不同攻击环境下对抗最先进的图后门攻击的有效性。此外，我们还研究了E-SAGE对抗对抗攻击的有效性。



## **41. From Trojan Horses to Castle Walls: Unveiling Bilateral Data Poisoning Effects in Diffusion Models**

从特洛伊木马到城墙：揭开扩散模型中的双边数据毒害效应 cs.LG

9 pages, 5 figures, 4 tables

**SubmitDate**: 2024-06-15    [abs](http://arxiv.org/abs/2311.02373v2) [paper-pdf](http://arxiv.org/pdf/2311.02373v2)

**Authors**: Zhuoshi Pan, Yuguang Yao, Gaowen Liu, Bingquan Shen, H. Vicky Zhao, Ramana Rao Kompella, Sijia Liu

**Abstract**: While state-of-the-art diffusion models (DMs) excel in image generation, concerns regarding their security persist. Earlier research highlighted DMs' vulnerability to data poisoning attacks, but these studies placed stricter requirements than conventional methods like `BadNets' in image classification. This is because the art necessitates modifications to the diffusion training and sampling procedures. Unlike the prior work, we investigate whether BadNets-like data poisoning methods can directly degrade the generation by DMs. In other words, if only the training dataset is contaminated (without manipulating the diffusion process), how will this affect the performance of learned DMs? In this setting, we uncover bilateral data poisoning effects that not only serve an adversarial purpose (compromising the functionality of DMs) but also offer a defensive advantage (which can be leveraged for defense in classification tasks against poisoning attacks). We show that a BadNets-like data poisoning attack remains effective in DMs for producing incorrect images (misaligned with the intended text conditions). Meanwhile, poisoned DMs exhibit an increased ratio of triggers, a phenomenon we refer to as `trigger amplification', among the generated images. This insight can be then used to enhance the detection of poisoned training data. In addition, even under a low poisoning ratio, studying the poisoning effects of DMs is also valuable for designing robust image classifiers against such attacks. Last but not least, we establish a meaningful linkage between data poisoning and the phenomenon of data replications by exploring DMs' inherent data memorization tendencies.

摘要: 虽然最先进的扩散模型(DM)在图像生成方面表现出色，但对其安全性的担忧依然存在。早期的研究强调了DM对数据中毒攻击的脆弱性，但这些研究在图像分类方面对图像分类提出了比传统方法(如“BadNets”)更严格的要求。这是因为本领域需要对扩散训练和采样程序进行修改。与以前的工作不同，我们研究了类BadNets的数据中毒方法是否可以直接降低DM的生成。换句话说，如果只有训练数据集受到污染(而不操纵扩散过程)，这将如何影响学习的DM的性能？在这种情况下，我们揭示了双边数据中毒效应，它不仅服务于敌对目的(损害DM的功能)，还提供了防御优势(可以在分类任务中用于防御中毒攻击)。我们表明，类BadNets的数据中毒攻击在DM中仍然有效，因为它产生了错误的图像(与预期的文本条件不一致)。同时，中毒的DM在生成的图像中表现出触发比率的增加，这种现象我们称为‘触发放大’。然后可以使用这种洞察力来增强对有毒训练数据的检测。此外，即使在低投毒率的情况下，研究DM的中毒效果对于设计针对此类攻击的稳健图像分类器也是有价值的。最后但并非最不重要的是，我们通过探索DM固有的数据记忆倾向，在数据中毒和数据复制现象之间建立了有意义的联系。



## **42. Robust Image Classification in the Presence of Out-of-Distribution and Adversarial Samples Using Attractors in Neural Networks**

在存在非分布和敌对样本的情况下使用神经网络中使用吸引子的鲁棒图像分类 cs.CV

**SubmitDate**: 2024-06-15    [abs](http://arxiv.org/abs/2406.10579v1) [paper-pdf](http://arxiv.org/pdf/2406.10579v1)

**Authors**: Nasrin Alipour, Seyyed Ali SeyyedSalehi

**Abstract**: The proper handling of out-of-distribution (OOD) samples in deep classifiers is a critical concern for ensuring the suitability of deep neural networks in safety-critical systems. Existing approaches developed for robust OOD detection in the presence of adversarial attacks lose their performance by increasing the perturbation levels. This study proposes a method for robust classification in the presence of OOD samples and adversarial attacks with high perturbation levels. The proposed approach utilizes a fully connected neural network that is trained to use training samples as its attractors, enhancing its robustness. This network has the ability to classify inputs and identify OOD samples as well. To evaluate this method, the network is trained on the MNIST dataset, and its performance is tested on adversarial examples. The results indicate that the network maintains its performance even when classifying adversarial examples, achieving 87.13% accuracy when dealing with highly perturbed MNIST test data. Furthermore, by using fashion-MNIST and CIFAR-10-bw as OOD samples, the network can distinguish these samples from MNIST samples with an accuracy of 98.84% and 99.28%, respectively. In the presence of severe adversarial attacks, these measures decrease slightly to 98.48% and 98.88%, indicating the robustness of the proposed method.

摘要: 深度分类器中失配样本的正确处理是确保深度神经网络在安全关键系统中适用性的关键问题。现有的用于在对抗性攻击存在的情况下进行稳健OOD检测的方法由于增加了扰动级别而失去了它们的性能。该研究提出了一种在存在OOD样本和高扰动水平的敌意攻击的情况下的稳健分类方法。该方法利用训练好的全连接神经网络作为其吸引子，增强了网络的稳健性。该网络具有分类输入和识别OOD样本的能力。为了对该方法进行评估，网络在MNIST数据集上进行了训练，并在对抗性例子上进行了性能测试。实验结果表明，该网络在处理高度扰动的MNIST测试数据时仍能保持其分类性能，达到87.13%的准确率。此外，通过使用FORM-MNIST和CIFAR-10-BW作为OOD样本，该网络可以将这些样本与MNIST样本区分开来，准确率分别为98.84%和99.28%。在存在严重的对抗性攻击时，这些度量分别略微下降到98.48%和98.88%，表明了该方法的健壮性。



## **43. Graph Neural Backdoor: Fundamentals, Methodologies, Applications, and Future Directions**

图形神经后门：基础知识、方法论、应用和未来方向 cs.LG

**SubmitDate**: 2024-06-15    [abs](http://arxiv.org/abs/2406.10573v1) [paper-pdf](http://arxiv.org/pdf/2406.10573v1)

**Authors**: Xiao Yang, Gaolei Li, Jianhua Li

**Abstract**: Graph Neural Networks (GNNs) have significantly advanced various downstream graph-relevant tasks, encompassing recommender systems, molecular structure prediction, social media analysis, etc. Despite the boosts of GNN, recent research has empirically demonstrated its potential vulnerability to backdoor attacks, wherein adversaries employ triggers to poison input samples, inducing GNN to adversary-premeditated malicious outputs. This is typically due to the controlled training process, or the deployment of untrusted models, such as delegating model training to third-party service, leveraging external training sets, and employing pre-trained models from online sources. Although there's an ongoing increase in research on GNN backdoors, comprehensive investigation into this field is lacking. To bridge this gap, we propose the first survey dedicated to GNN backdoors. We begin by outlining the fundamental definition of GNN, followed by the detailed summarization and categorization of current GNN backdoor attacks and defenses based on their technical characteristics and application scenarios. Subsequently, the analysis of the applicability and use cases of GNN backdoors is undertaken. Finally, the exploration of potential research directions of GNN backdoors is presented. This survey aims to explore the principles of graph backdoors, provide insights to defenders, and promote future security research.

摘要: 图神经网络(GNN)已经显著推进了各种下游与图相关的任务，包括推荐系统、分子结构预测、社交媒体分析等。尽管GNN得到了提升，但最近的研究经验表明，它对后门攻击具有潜在的脆弱性，即攻击者使用触发器来毒化输入样本，诱导GNN进行攻击者预谋的恶意输出。这通常是由于受控的培训过程或不受信任的模型的部署，例如将模型培训委托给第三方服务、利用外部培训集以及使用来自在线来源的预先培训的模型。尽管对GNN后门的研究在不断增加，但对这一领域的全面调查还很缺乏。为了弥补这一差距，我们建议对GNN后门进行第一次调查。我们首先概述了GNN的基本定义，然后根据其技术特征和应用场景对当前GNN后门攻击和防御进行了详细的总结和分类。随后，对GNN后门的适用性和使用案例进行了分析。最后，对GNN后门的潜在研究方向进行了展望。这项调查旨在探索图形后门的原理，为防御者提供见解，并促进未来的安全研究。



## **44. To Generate or Not? Safety-Driven Unlearned Diffusion Models Are Still Easy To Generate Unsafe Images ... For Now**

生成还是不生成？安全驱动的未学习扩散模型仍然很容易生成不安全的图像.现在 cs.CV

Codes are available at  https://github.com/OPTML-Group/Diffusion-MU-Attack

**SubmitDate**: 2024-06-15    [abs](http://arxiv.org/abs/2310.11868v3) [paper-pdf](http://arxiv.org/pdf/2310.11868v3)

**Authors**: Yimeng Zhang, Jinghan Jia, Xin Chen, Aochuan Chen, Yihua Zhang, Jiancheng Liu, Ke Ding, Sijia Liu

**Abstract**: The recent advances in diffusion models (DMs) have revolutionized the generation of realistic and complex images. However, these models also introduce potential safety hazards, such as producing harmful content and infringing data copyrights. Despite the development of safety-driven unlearning techniques to counteract these challenges, doubts about their efficacy persist. To tackle this issue, we introduce an evaluation framework that leverages adversarial prompts to discern the trustworthiness of these safety-driven DMs after they have undergone the process of unlearning harmful concepts. Specifically, we investigated the adversarial robustness of DMs, assessed by adversarial prompts, when eliminating unwanted concepts, styles, and objects. We develop an effective and efficient adversarial prompt generation approach for DMs, termed UnlearnDiffAtk. This method capitalizes on the intrinsic classification abilities of DMs to simplify the creation of adversarial prompts, thereby eliminating the need for auxiliary classification or diffusion models.Through extensive benchmarking, we evaluate the robustness of five widely-used safety-driven unlearned DMs (i.e., DMs after unlearning undesirable concepts, styles, or objects) across a variety of tasks. Our results demonstrate the effectiveness and efficiency merits of UnlearnDiffAtk over the state-of-the-art adversarial prompt generation method and reveal the lack of robustness of current safety-driven unlearning techniques when applied to DMs. Codes are available at https://github.com/OPTML-Group/Diffusion-MU-Attack. WARNING: This paper contains model outputs that may be offensive in nature.

摘要: 扩散模型的最新进展使逼真和复杂图像的生成发生了革命性的变化。然而，这些模式也带来了潜在的安全隐患，如产生有害内容和侵犯数据著作权。尽管发展了安全驱动的遗忘技术来应对这些挑战，但对其有效性的怀疑依然存在。为了解决这个问题，我们引入了一个评估框架，利用对抗性提示，在这些以安全为导向的DM经历了忘记有害概念的过程后，识别他们的可信度。具体地说，我们研究了DM在消除不需要的概念、风格和对象时，通过对抗性提示评估的对抗性健壮性。本文提出了一种高效的敌意提示生成方法，称为UnlearnDiffAtk。该方法利用DM固有的分类能力来简化敌意提示的生成，从而消除了对辅助分类或扩散模型的需要。通过广泛的基准测试，我们评估了五种广泛使用的安全驱动的未学习DM(即忘记不良概念、风格或对象后的DM)在不同任务中的健壮性。实验结果证明了UnlearnDiffAtk算法相对于最新的对抗性提示生成方法的有效性和高效性，并揭示了当前安全驱动的遗忘技术在应用于决策支持系统时的健壮性不足。有关代码，请访问https://github.com/OPTML-Group/Diffusion-MU-Attack.警告：本文包含可能具有攻击性的模型输出。



## **45. Towards the Theory of Unsupervised Federated Learning: Non-asymptotic Analysis of Federated EM Algorithms**

走向无监督联邦学习理论：联邦EM算法的非渐进分析 stat.ML

50 pages, 3 figures

**SubmitDate**: 2024-06-14    [abs](http://arxiv.org/abs/2310.15330v3) [paper-pdf](http://arxiv.org/pdf/2310.15330v3)

**Authors**: Ye Tian, Haolei Weng, Yang Feng

**Abstract**: While supervised federated learning approaches have enjoyed significant success, the domain of unsupervised federated learning remains relatively underexplored. Several federated EM algorithms have gained popularity in practice, however, their theoretical foundations are often lacking. In this paper, we first introduce a federated gradient EM algorithm (FedGrEM) designed for the unsupervised learning of mixture models, which supplements the existing federated EM algorithms by considering task heterogeneity and potential adversarial attacks. We present a comprehensive finite-sample theory that holds for general mixture models, then apply this general theory on specific statistical models to characterize the explicit estimation error of model parameters and mixture proportions. Our theory elucidates when and how FedGrEM outperforms local single-task learning with insights extending to existing federated EM algorithms. This bridges the gap between their practical success and theoretical understanding. Our numerical results validate our theory, and demonstrate FedGrEM's superiority over existing unsupervised federated learning benchmarks.

摘要: 虽然有监督的联合学习方法已经取得了很大的成功，但无监督的联合学习领域仍然相对较少被探索。几种联合EM算法在实践中得到了广泛的应用，但它们的理论基础往往是欠缺的。本文首先介绍了一种用于混合模型无监督学习的联合梯度EM算法(FedGrEM)，该算法通过考虑任务的异构性和潜在的敌意攻击来补充现有的联合EM算法。我们给出了一个适用于一般混合模型的综合有限样本理论，然后将这个一般理论应用于具体的统计模型来刻画模型参数和混合比例的显式估计误差。我们的理论解释了FedGrEM何时以及如何优于本地单任务学习，其见解延伸到现有的联合EM算法。这在他们的实践成功和理论理解之间架起了一座桥梁。我们的数值结果验证了我们的理论，并证明了FedGrEM相对于现有的无监督联合学习基准的优越性。



## **46. Defensive Unlearning with Adversarial Training for Robust Concept Erasure in Diffusion Models**

扩散模型中鲁棒概念擦除的对抗训练防御性取消学习 cs.CV

Codes are available at https://github.com/OPTML-Group/AdvUnlearn

**SubmitDate**: 2024-06-14    [abs](http://arxiv.org/abs/2405.15234v2) [paper-pdf](http://arxiv.org/pdf/2405.15234v2)

**Authors**: Yimeng Zhang, Xin Chen, Jinghan Jia, Yihua Zhang, Chongyu Fan, Jiancheng Liu, Mingyi Hong, Ke Ding, Sijia Liu

**Abstract**: Diffusion models (DMs) have achieved remarkable success in text-to-image generation, but they also pose safety risks, such as the potential generation of harmful content and copyright violations. The techniques of machine unlearning, also known as concept erasing, have been developed to address these risks. However, these techniques remain vulnerable to adversarial prompt attacks, which can prompt DMs post-unlearning to regenerate undesired images containing concepts (such as nudity) meant to be erased. This work aims to enhance the robustness of concept erasing by integrating the principle of adversarial training (AT) into machine unlearning, resulting in the robust unlearning framework referred to as AdvUnlearn. However, achieving this effectively and efficiently is highly nontrivial. First, we find that a straightforward implementation of AT compromises DMs' image generation quality post-unlearning. To address this, we develop a utility-retaining regularization on an additional retain set, optimizing the trade-off between concept erasure robustness and model utility in AdvUnlearn. Moreover, we identify the text encoder as a more suitable module for robustification compared to UNet, ensuring unlearning effectiveness. And the acquired text encoder can serve as a plug-and-play robust unlearner for various DM types. Empirically, we perform extensive experiments to demonstrate the robustness advantage of AdvUnlearn across various DM unlearning scenarios, including the erasure of nudity, objects, and style concepts. In addition to robustness, AdvUnlearn also achieves a balanced tradeoff with model utility. To our knowledge, this is the first work to systematically explore robust DM unlearning through AT, setting it apart from existing methods that overlook robustness in concept erasing. Codes are available at: https://github.com/OPTML-Group/AdvUnlearn

摘要: 扩散模型(DM)在文本到图像的生成方面取得了显著的成功，但它们也带来了安全风险，如可能生成有害内容和侵犯版权。机器遗忘技术，也被称为概念擦除，就是为了解决这些风险而开发的。然而，这些技术仍然容易受到敌意的即时攻击，这可能会促使忘记后的DM重新生成包含要擦除的概念(如裸体)的不需要的图像。这项工作旨在通过将对抗性训练(AT)的原理整合到机器遗忘中来增强概念删除的稳健性，从而产生健壮的遗忘框架，称为AdvUnLearning。然而，有效和高效地实现这一点并不是微不足道的。首先，我们发现AT的直接实现损害了DM在遗忘后的图像生成质量。为了解决这个问题，我们在一个额外的保留集上开发了效用保留正则化，优化了AdvUnLearning中概念删除健壮性和模型实用之间的权衡。此外，我们认为文本编码器是一个更适合于粗暴的模块，与联合国教科文组织相比，确保了遗忘的有效性。并且所获得的文本编码器可以作为各种DM类型的即插即用鲁棒去学习器。经验性地，我们进行了大量的实验来展示AdvUnLearning在各种DM遗忘场景中的健壮性优势，包括对裸体、物体和风格概念的删除。除了健壮性之外，AdvUnLearning还实现了与模型实用程序之间的平衡。据我们所知，这是第一个通过AT系统地探索稳健的DM遗忘的工作，区别于现有的忽略概念删除中的稳健性的方法。代码可在以下网址获得：https://github.com/OPTML-Group/AdvUnlearn



## **47. I Still See You: Why Existing IoT Traffic Reshaping Fails**

我仍然见到你：为什么现有的物联网流量重塑失败 cs.CR

EWSN'24 paper accepted, to appear

**SubmitDate**: 2024-06-14    [abs](http://arxiv.org/abs/2406.10358v1) [paper-pdf](http://arxiv.org/pdf/2406.10358v1)

**Authors**: Su Wang, Keyang Yu, Qi Li, Dong Chen

**Abstract**: The Internet traffic data produced by the Internet of Things (IoT) devices are collected by Internet Service Providers (ISPs) and device manufacturers, and often shared with their third parties to maintain and enhance user services. Unfortunately, on-path adversaries could infer and fingerprint users' sensitive privacy information such as occupancy and user activities by analyzing these network traffic traces. While there's a growing body of literature on defending against this side-channel attack-malicious IoT traffic analytics (TA), there's currently no systematic method to compare and evaluate the comprehensiveness of these existing studies. To address this problem, we design a new low-cost, open-source system framework-IoT Traffic Exposure Monitoring Toolkit (ITEMTK) that enables people to comprehensively examine and validate prior attack models and their defending approaches. In particular, we also design a novel image-based attack capable of inferring sensitive user information, even when users employ the most robust preventative measures in their smart homes. Researchers could leverage our new image-based attack to systematize and understand the existing literature on IoT traffic analysis attacks and preventing studies. Our results show that current defending approaches are not sufficient to protect IoT device user privacy. IoT devices are significantly vulnerable to our new image-based user privacy inference attacks, posing a grave threat to IoT device user privacy. We also highlight potential future improvements to enhance the defending approaches. ITEMTK's flexibility allows other researchers for easy expansion by integrating new TA attack models and prevention methods to benchmark their future work.

摘要: 物联网(IoT)设备产生的互联网流量数据由互联网服务提供商(ISP)和设备制造商收集，并经常与他们的第三方共享，以维护和增强用户服务。不幸的是，路径上的攻击者可以通过分析这些网络流量痕迹来推断和指纹用户的敏感隐私信息，如占用率和用户活动。虽然有越来越多的文献关于防御这种侧通道攻击-恶意物联网流量分析(TA)，但目前还没有系统的方法来比较和评估这些现有研究的全面性。为了解决这一问题，我们设计了一个新的低成本、开源的系统框架-物联网流量暴露监控工具包(ITEMTK)，使人们能够全面检查和验证先前的攻击模型及其防御方法。特别是，我们还设计了一种新颖的基于图像的攻击，即使用户在他们的智能家居中采用了最强大的预防措施，也能够推断出用户的敏感信息。研究人员可以利用我们新的基于图像的攻击来系统化和理解有关物联网流量分析攻击和预防研究的现有文献。我们的结果表明，现有的防御方法不足以保护物联网设备用户的隐私。物联网设备极易受到我们新的基于图像的用户隐私推断攻击，对物联网设备用户隐私构成严重威胁。我们还强调了未来可能的改进，以增强防御方法。ITEMTK的灵活性允许其他研究人员通过集成新的TA攻击模型和预防方法来轻松扩展，以衡量他们未来的工作。



## **48. Automated Design of Linear Bounding Functions for Sigmoidal Nonlinearities in Neural Networks**

神经网络中Sigmoidal非线性线性边界函数的自动设计 cs.LG

**SubmitDate**: 2024-06-14    [abs](http://arxiv.org/abs/2406.10154v1) [paper-pdf](http://arxiv.org/pdf/2406.10154v1)

**Authors**: Matthias König, Xiyue Zhang, Holger H. Hoos, Marta Kwiatkowska, Jan N. van Rijn

**Abstract**: The ubiquity of deep learning algorithms in various applications has amplified the need for assuring their robustness against small input perturbations such as those occurring in adversarial attacks. Existing complete verification techniques offer provable guarantees for all robustness queries but struggle to scale beyond small neural networks. To overcome this computational intractability, incomplete verification methods often rely on convex relaxation to over-approximate the nonlinearities in neural networks. Progress in tighter approximations has been achieved for piecewise linear functions. However, robustness verification of neural networks for general activation functions (e.g., Sigmoid, Tanh) remains under-explored and poses new challenges. Typically, these networks are verified using convex relaxation techniques, which involve computing linear upper and lower bounds of the nonlinear activation functions. In this work, we propose a novel parameter search method to improve the quality of these linear approximations. Specifically, we show that using a simple search method, carefully adapted to the given verification problem through state-of-the-art algorithm configuration techniques, improves the average global lower bound by 25% on average over the current state of the art on several commonly used local robustness verification benchmarks.

摘要: 深度学习算法在各种应用中的普遍存在，增加了确保其对诸如在对抗性攻击中发生的小输入扰动的稳健性的必要性。现有的完整验证技术为所有健壮性查询提供了可证明的保证，但难以扩展到小型神经网络之外。为了克服这种计算困难，不完全验证方法通常依赖于凸松弛来过度逼近神经网络中的非线性。在分段线性函数的更紧密逼近方面已经取得了进展。然而，神经网络对一般激活函数(例如Sigmoid、Tanh)的稳健性验证仍然没有得到充分的探索，并提出了新的挑战。通常，使用凸松弛技术来验证这些网络，该技术涉及计算非线性激活函数的线性上界和下界。在这项工作中，我们提出了一种新的参数搜索方法来提高这些线性逼近的质量。具体地说，我们通过使用一种简单的搜索方法，通过最新的算法配置技术仔细地适应给定的验证问题，在几个常用的局部健壮性验证基准上，平均将全局下界提高了25%。



## **49. Over-parameterization and Adversarial Robustness in Neural Networks: An Overview and Empirical Analysis**

神经网络中的过度参数化和对抗鲁棒性：概述和实证分析 cs.LG

**SubmitDate**: 2024-06-14    [abs](http://arxiv.org/abs/2406.10090v1) [paper-pdf](http://arxiv.org/pdf/2406.10090v1)

**Authors**: Zhang Chen, Luca Demetrio, Srishti Gupta, Xiaoyi Feng, Zhaoqiang Xia, Antonio Emanuele Cinà, Maura Pintor, Luca Oneto, Ambra Demontis, Battista Biggio, Fabio Roli

**Abstract**: Thanks to their extensive capacity, over-parameterized neural networks exhibit superior predictive capabilities and generalization. However, having a large parameter space is considered one of the main suspects of the neural networks' vulnerability to adversarial example -- input samples crafted ad-hoc to induce a desired misclassification. Relevant literature has claimed contradictory remarks in support of and against the robustness of over-parameterized networks. These contradictory findings might be due to the failure of the attack employed to evaluate the networks' robustness. Previous research has demonstrated that depending on the considered model, the algorithm employed to generate adversarial examples may not function properly, leading to overestimating the model's robustness. In this work, we empirically study the robustness of over-parameterized networks against adversarial examples. However, unlike the previous works, we also evaluate the considered attack's reliability to support the results' veracity. Our results show that over-parameterized networks are robust against adversarial attacks as opposed to their under-parameterized counterparts.

摘要: 由于其广泛的能力，超参数神经网络显示出优越的预测能力和泛化能力。然而，拥有较大的参数空间被认为是神经网络易受敌意示例攻击的主要原因之一--输入样本特别地被精心设计以导致期望的错误分类。相关文献在支持和反对过参数网络的稳健性方面发表了相互矛盾的言论。这些相互矛盾的发现可能是由于用于评估网络健壮性的攻击失败所致。以前的研究表明，根据所考虑的模型，用于生成对抗性示例的算法可能不能正常工作，导致高估了模型的稳健性。在这项工作中，我们经验地研究了过参数网络对敌意例子的稳健性。然而，与前人的工作不同，我们还评估了所考虑的攻击的可靠性，以支持结果的准确性。我们的结果表明，与欠参数网络相比，过参数网络对敌意攻击具有较强的鲁棒性。



## **50. Perturbing Attention Gives You More Bang for the Buck: Subtle Imaging Perturbations That Efficiently Fool Customized Diffusion Models**

扰动注意力为您带来更多好处：有效愚弄定制扩散模型的微妙成像扰动 cs.CV

Published at CVPR 2024, code:https://github.com/CO2-cityao/CAAT

**SubmitDate**: 2024-06-14    [abs](http://arxiv.org/abs/2404.15081v2) [paper-pdf](http://arxiv.org/pdf/2404.15081v2)

**Authors**: Jingyao Xu, Yuetong Lu, Yandong Li, Siyang Lu, Dongdong Wang, Xiang Wei

**Abstract**: Diffusion models (DMs) embark a new era of generative modeling and offer more opportunities for efficient generating high-quality and realistic data samples. However, their widespread use has also brought forth new challenges in model security, which motivates the creation of more effective adversarial attackers on DMs to understand its vulnerability. We propose CAAT, a simple but generic and efficient approach that does not require costly training to effectively fool latent diffusion models (LDMs). The approach is based on the observation that cross-attention layers exhibits higher sensitivity to gradient change, allowing for leveraging subtle perturbations on published images to significantly corrupt the generated images. We show that a subtle perturbation on an image can significantly impact the cross-attention layers, thus changing the mapping between text and image during the fine-tuning of customized diffusion models. Extensive experiments demonstrate that CAAT is compatible with diverse diffusion models and outperforms baseline attack methods in a more effective (more noise) and efficient (twice as fast as Anti-DreamBooth and Mist) manner.

摘要: 扩散模型开启了产生式建模的新时代，为高效地生成高质量和真实的数据样本提供了更多的机会。然而，它们的广泛使用也给模型安全带来了新的挑战，这促使在DM上创建更有效的对抗性攻击者来了解其脆弱性。我们提出了CAAT，这是一种简单但通用和高效的方法，不需要昂贵的培训来有效地愚弄潜在扩散模型(LDM)。该方法的基础是观察到交叉注意层对梯度变化表现出更高的敏感性，允许利用发布图像上的细微扰动来显著破坏生成的图像。我们发现，在定制扩散模型的微调过程中，图像上的细微扰动会显著影响交叉注意层，从而改变文本和图像之间的映射。大量的实验表明，CAAT与多种扩散模型兼容，并且在更有效(更多噪声)和更高效(速度是Anti-DreamBooth和Mist的两倍)方面优于基线攻击方法。



