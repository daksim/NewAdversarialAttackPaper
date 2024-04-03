# Latest Adversarial Attack Papers
**update at 2024-04-03 21:09:48**

翻译来自 https://cloud.tencent.com/document/product/551/15619

## **1. Jailbreaking Leading Safety-Aligned LLMs with Simple Adaptive Attacks**

使用简单的自适应攻击破解领先的安全一致LLM cs.CR

**SubmitDate**: 2024-04-02    [abs](http://arxiv.org/abs/2404.02151v1) [paper-pdf](http://arxiv.org/pdf/2404.02151v1)

**Authors**: Maksym Andriushchenko, Francesco Croce, Nicolas Flammarion

**Abstract**: We show that even the most recent safety-aligned LLMs are not robust to simple adaptive jailbreaking attacks. First, we demonstrate how to successfully leverage access to logprobs for jailbreaking: we initially design an adversarial prompt template (sometimes adapted to the target LLM), and then we apply random search on a suffix to maximize the target logprob (e.g., of the token "Sure"), potentially with multiple restarts. In this way, we achieve nearly 100\% attack success rate -- according to GPT-4 as a judge -- on GPT-3.5/4, Llama-2-Chat-7B/13B/70B, Gemma-7B, and R2D2 from HarmBench that was adversarially trained against the GCG attack. We also show how to jailbreak all Claude models -- that do not expose logprobs -- via either a transfer or prefilling attack with 100\% success rate. In addition, we show how to use random search on a restricted set of tokens for finding trojan strings in poisoned models -- a task that shares many similarities with jailbreaking -- which is the algorithm that brought us the first place in the SaTML'24 Trojan Detection Competition. The common theme behind these attacks is that adaptivity is crucial: different models are vulnerable to different prompting templates (e.g., R2D2 is very sensitive to in-context learning prompts), some models have unique vulnerabilities based on their APIs (e.g., prefilling for Claude), and in some settings it is crucial to restrict the token search space based on prior knowledge (e.g., for trojan detection). We provide the code, prompts, and logs of the attacks at https://github.com/tml-epfl/llm-adaptive-attacks.

摘要: 我们表明，即使是最新的安全对齐的LLM也不能抵抗简单的自适应越狱攻击。首先，我们演示了如何成功地利用对logpros的访问来越狱：我们最初设计了一个对抗性提示模板(有时适用于目标LLM)，然后在后缀上应用随机搜索来最大化目标logprob(例如，令牌“Sure”)，可能需要多次重新启动。通过这种方式，我们获得了近100%的攻击成功率-根据GPT-4作为判断-在GPT-3.5/4、Llama-2-Chat-7B/13B/70B、Gema-7B和R2D2上，它们都经过了对抗GCG攻击的恶意训练。我们还展示了如何通过传输或预填充攻击以100%的成功率越狱所有不暴露日志问题的Claude模型。此外，我们还展示了如何在受限的令牌集合上使用随机搜索来查找有毒模型中的特洛伊木马字符串--这项任务与越狱有许多相似之处--正是这种算法为我们带来了SATML‘24特洛伊木马检测大赛的第一名。这些攻击背后的共同主题是自适应至关重要：不同的模型容易受到不同提示模板的攻击(例如，R2D2对上下文中的学习提示非常敏感)，一些模型基于其API具有独特的漏洞(例如，预填充Claude)，并且在某些设置中，基于先验知识限制令牌搜索空间至关重要(例如，对于木马检测)。我们在https://github.com/tml-epfl/llm-adaptive-attacks.上提供攻击的代码、提示和日志



## **2. Red-Teaming Segment Anything Model**

Red—Team Segment Anything Model cs.CV

CVPR 2024 - The 4th Workshop of Adversarial Machine Learning on  Computer Vision: Robustness of Foundation Models

**SubmitDate**: 2024-04-02    [abs](http://arxiv.org/abs/2404.02067v1) [paper-pdf](http://arxiv.org/pdf/2404.02067v1)

**Authors**: Krzysztof Jankowski, Bartlomiej Sobieski, Mateusz Kwiatkowski, Jakub Szulc, Michal Janik, Hubert Baniecki, Przemyslaw Biecek

**Abstract**: Foundation models have emerged as pivotal tools, tackling many complex tasks through pre-training on vast datasets and subsequent fine-tuning for specific applications. The Segment Anything Model is one of the first and most well-known foundation models for computer vision segmentation tasks. This work presents a multi-faceted red-teaming analysis that tests the Segment Anything Model against challenging tasks: (1) We analyze the impact of style transfer on segmentation masks, demonstrating that applying adverse weather conditions and raindrops to dashboard images of city roads significantly distorts generated masks. (2) We focus on assessing whether the model can be used for attacks on privacy, such as recognizing celebrities' faces, and show that the model possesses some undesired knowledge in this task. (3) Finally, we check how robust the model is to adversarial attacks on segmentation masks under text prompts. We not only show the effectiveness of popular white-box attacks and resistance to black-box attacks but also introduce a novel approach - Focused Iterative Gradient Attack (FIGA) that combines white-box approaches to construct an efficient attack resulting in a smaller number of modified pixels. All of our testing methods and analyses indicate a need for enhanced safety measures in foundation models for image segmentation.

摘要: 基础模型已经成为关键工具，通过对大量数据集进行预培训并随后针对特定应用进行微调来处理许多复杂任务。任意分割模型是最早也是最著名的计算机视觉分割任务的基础模型之一。这项工作提出了一个多方面的红团队分析，针对具有挑战性的任务测试了Segment Anything Model：(1)我们分析了样式转移对分段掩模的影响，表明将不利的天气条件和雨滴应用于城市道路的仪表板图像会显著扭曲生成的掩模。(2)我们重点评估了该模型是否可以用于隐私攻击，如识别名人的脸，并表明该模型在该任务中具有一些不需要的知识。(3)最后，我们检验了该模型对文本提示下的分割模板攻击的健壮性。我们不仅展示了流行的白盒攻击的有效性和对黑盒攻击的抵抗力，而且还引入了一种新的专注于方法的迭代梯度攻击(FIGA)，它结合了白盒方法来构造有效的攻击，从而减少了修改的像素数。我们所有的测试方法和分析都表明，需要在图像分割的基础模型中加强安全措施。



## **3. How Trustworthy are Open-Source LLMs? An Assessment under Malicious Demonstrations Shows their Vulnerabilities**

开源LLM有多值得信赖？恶意示威下的评估显示其脆弱性 cs.CL

NAACL 2024

**SubmitDate**: 2024-04-02    [abs](http://arxiv.org/abs/2311.09447v2) [paper-pdf](http://arxiv.org/pdf/2311.09447v2)

**Authors**: Lingbo Mo, Boshi Wang, Muhao Chen, Huan Sun

**Abstract**: The rapid progress in open-source Large Language Models (LLMs) is significantly driving AI development forward. However, there is still a limited understanding of their trustworthiness. Deploying these models at scale without sufficient trustworthiness can pose significant risks, highlighting the need to uncover these issues promptly. In this work, we conduct an adversarial assessment of open-source LLMs on trustworthiness, scrutinizing them across eight different aspects including toxicity, stereotypes, ethics, hallucination, fairness, sycophancy, privacy, and robustness against adversarial demonstrations. We propose advCoU, an extended Chain of Utterances-based (CoU) prompting strategy by incorporating carefully crafted malicious demonstrations for trustworthiness attack. Our extensive experiments encompass recent and representative series of open-source LLMs, including Vicuna, MPT, Falcon, Mistral, and Llama 2. The empirical outcomes underscore the efficacy of our attack strategy across diverse aspects. More interestingly, our result analysis reveals that models with superior performance in general NLP tasks do not always have greater trustworthiness; in fact, larger models can be more vulnerable to attacks. Additionally, models that have undergone instruction tuning, focusing on instruction following, tend to be more susceptible, although fine-tuning LLMs for safety alignment proves effective in mitigating adversarial trustworthiness attacks.

摘要: 开源大型语言模型(LLM)的快速发展显著地推动了人工智能的发展。然而，人们对他们的可信性仍知之甚少。在缺乏足够可信度的情况下大规模部署这些模型可能会带来重大风险，这突显了迅速发现这些问题的必要性。在这项工作中，我们对开源LLM的可信性进行了对抗性评估，从八个不同的方面对它们进行了仔细的审查，包括毒性、刻板印象、伦理、幻觉、公平性、奉承、隐私和对对抗性演示的健壮性。我们提出了AdvCoU，一种基于话语的扩展链(CUU)提示策略，它结合了精心制作的恶意演示来进行可信度攻击。我们广泛的实验涵盖了最近一系列具有代表性的开源LLM，包括Vicuna、MPT、Falcon、Mistral和Llama 2。经验结果强调了我们攻击策略在不同方面的有效性。更有趣的是，我们的结果分析显示，在一般NLP任务中性能优越的模型并不总是具有更大的可信度；事实上，较大的模型可能更容易受到攻击。此外，经过指令调整、专注于指令遵循的模型往往更容易受到影响，尽管针对安全对齐的微调LLM被证明在减轻对手信任攻击方面是有效的。



## **4. Deciphering the Interplay between Local Differential Privacy, Average Bayesian Privacy, and Maximum Bayesian Privacy**

解密局部差分隐私、平均贝叶斯隐私和最大贝叶斯隐私之间的相互作用 cs.LG

**SubmitDate**: 2024-04-02    [abs](http://arxiv.org/abs/2403.16591v3) [paper-pdf](http://arxiv.org/pdf/2403.16591v3)

**Authors**: Xiaojin Zhang, Yulin Fei, Wei Chen

**Abstract**: The swift evolution of machine learning has led to emergence of various definitions of privacy due to the threats it poses to privacy, including the concept of local differential privacy (LDP). Although widely embraced and utilized across numerous domains, this conventional approach to measure privacy still exhibits certain limitations, spanning from failure to prevent inferential disclosure to lack of consideration for the adversary's background knowledge. In this comprehensive study, we introduce Bayesian privacy and delve into the intricate relationship between LDP and its Bayesian counterparts, unveiling novel insights into utility-privacy trade-offs. We introduce a framework that encapsulates both attack and defense strategies, highlighting their interplay and effectiveness. The relationship between LDP and Maximum Bayesian Privacy (MBP) is first revealed, demonstrating that under uniform prior distribution, a mechanism satisfying $\xi$-LDP will satisfy $\xi$-MBP and conversely $\xi$-MBP also confers 2$\xi$-LDP. Our next theoretical contribution are anchored in the rigorous definitions and relationships between Average Bayesian Privacy (ABP) and Maximum Bayesian Privacy (MBP), encapsulated by equations $\epsilon_{p,a} \leq \frac{1}{\sqrt{2}}\sqrt{(\epsilon_{p,m} + \epsilon)\cdot(e^{\epsilon_{p,m} + \epsilon} - 1)}$. These relationships fortify our understanding of the privacy guarantees provided by various mechanisms. Our work not only lays the groundwork for future empirical exploration but also promises to facilitate the design of privacy-preserving algorithms, thereby fostering the development of trustworthy machine learning solutions.

摘要: 机器学习的快速发展导致了各种隐私定义的出现，因为它对隐私构成了威胁，包括局部差异隐私(LDP)的概念。尽管这种衡量隐私的传统方法在许多领域得到了广泛的接受和应用，但它仍然显示出一定的局限性，从未能阻止推论披露到缺乏对对手背景知识的考虑。在这项全面的研究中，我们介绍了贝叶斯隐私，并深入研究了自民党与其贝叶斯同行之间的错综复杂的关系，揭示了对效用-隐私权衡的新见解。我们引入了一个框架，该框架封装了攻击和防御战略，突出了它们的相互作用和有效性。首先揭示了LDP与最大贝叶斯隐私度之间的关系，证明了在均匀先验分布下，满足$xi-LDP的机制将满足$\xi-MBP，反之，$\xi-MBP也赋予2$\xi-LDP。我们的下一个理论贡献是建立在平均贝叶斯隐私度(ABP)和最大贝叶斯隐私度(MBP)之间的严格定义和关系上，用方程$\epsilon_{p，a}\leq\frac{1}{\sqrt{2}}\sqrt{(\epsilon_{p，m}+\epsilon)\cdot(e^{\epsilon_{p，m}+\epsilon}-1)}$来封装。这些关系加强了我们对各种机制提供的隐私保障的理解。我们的工作不仅为未来的经验探索奠定了基础，也承诺促进隐私保护算法的设计，从而促进可信机器学习解决方案的开发。



## **5. PatchCURE: Improving Certifiable Robustness, Model Utility, and Computation Efficiency of Adversarial Patch Defenses**

PatchCURE：提高对抗补丁防御的可证明鲁棒性、模型效用和计算效率 cs.CV

USENIX Security 2024. (extended) technical report

**SubmitDate**: 2024-04-02    [abs](http://arxiv.org/abs/2310.13076v2) [paper-pdf](http://arxiv.org/pdf/2310.13076v2)

**Authors**: Chong Xiang, Tong Wu, Sihui Dai, Jonathan Petit, Suman Jana, Prateek Mittal

**Abstract**: State-of-the-art defenses against adversarial patch attacks can now achieve strong certifiable robustness with a marginal drop in model utility. However, this impressive performance typically comes at the cost of 10-100x more inference-time computation compared to undefended models -- the research community has witnessed an intense three-way trade-off between certifiable robustness, model utility, and computation efficiency. In this paper, we propose a defense framework named PatchCURE to approach this trade-off problem. PatchCURE provides sufficient "knobs" for tuning defense performance and allows us to build a family of defenses: the most robust PatchCURE instance can match the performance of any existing state-of-the-art defense (without efficiency considerations); the most efficient PatchCURE instance has similar inference efficiency as undefended models. Notably, PatchCURE achieves state-of-the-art robustness and utility performance across all different efficiency levels, e.g., 16-23% absolute clean accuracy and certified robust accuracy advantages over prior defenses when requiring computation efficiency to be close to undefended models. The family of PatchCURE defenses enables us to flexibly choose appropriate defenses to satisfy given computation and/or utility constraints in practice.

摘要: 针对对抗性补丁攻击的最先进防御现在可以实现强大的可证明的健壮性，同时模型效用略有下降。然而，这种令人印象深刻的性能通常是以比无防御模型多10-100倍的推理时间计算为代价的--研究界见证了可证明的健壮性、模型实用性和计算效率之间的激烈三方权衡。在本文中，我们提出了一个名为PatchCURE的防御框架来解决这个权衡问题。PatchCURE为调整防御性能提供了足够的“旋钮”，并允许我们构建一系列防御：最健壮的PatchCURE实例可以与任何现有最先进的防御实例的性能相媲美(无需考虑效率)；最高效的PatchCURE实例具有与无防御模型相似的推理效率。值得注意的是，PatchCURE在所有不同的效率水平上实现了最先进的稳健性和实用性能，例如，当需要计算效率接近无防御模型时，绝对清洁准确率为16%-23%，并且经过认证的稳健精确度优于以前的防御系统。PatchCURE防御体系使我们能够灵活地选择适当的防御，以满足实践中给定的计算和/或效用约束。



## **6. Humanizing Machine-Generated Content: Evading AI-Text Detection through Adversarial Attack**

人性化机器生成内容：通过对抗攻击规避AI文本检测 cs.CL

Accepted by COLING 2024

**SubmitDate**: 2024-04-02    [abs](http://arxiv.org/abs/2404.01907v1) [paper-pdf](http://arxiv.org/pdf/2404.01907v1)

**Authors**: Ying Zhou, Ben He, Le Sun

**Abstract**: With the development of large language models (LLMs), detecting whether text is generated by a machine becomes increasingly challenging in the face of malicious use cases like the spread of false information, protection of intellectual property, and prevention of academic plagiarism. While well-trained text detectors have demonstrated promising performance on unseen test data, recent research suggests that these detectors have vulnerabilities when dealing with adversarial attacks such as paraphrasing. In this paper, we propose a framework for a broader class of adversarial attacks, designed to perform minor perturbations in machine-generated content to evade detection. We consider two attack settings: white-box and black-box, and employ adversarial learning in dynamic scenarios to assess the potential enhancement of the current detection model's robustness against such attacks. The empirical results reveal that the current detection models can be compromised in as little as 10 seconds, leading to the misclassification of machine-generated text as human-written content. Furthermore, we explore the prospect of improving the model's robustness over iterative adversarial learning. Although some improvements in model robustness are observed, practical applications still face significant challenges. These findings shed light on the future development of AI-text detectors, emphasizing the need for more accurate and robust detection methods.

摘要: 随着大型语言模型(LLM)的发展，面对虚假信息传播、知识产权保护和防止学术剽窃等恶意使用案例，检测文本是否由机器生成变得越来越具有挑战性。虽然训练有素的文本检测器在看不见的测试数据上表现出了良好的性能，但最近的研究表明，这些检测器在处理诸如释义等敌意攻击时存在漏洞。在本文中，我们提出了一个更广泛类别的对抗性攻击的框架，旨在对机器生成的内容执行微小的扰动以逃避检测。我们考虑了两种攻击环境：白盒和黑盒，并在动态场景中使用对抗性学习来评估当前检测模型对此类攻击的稳健性的潜在增强。实验结果表明，当前的检测模型可以在短短10秒内被攻破，导致机器生成的文本被错误分类为人类书写的内容。此外，我们还探讨了改进模型在迭代对抗学习中的稳健性的前景。虽然在模型稳健性方面观察到了一些改进，但实际应用仍然面临着巨大的挑战。这些发现为人工智能文本检测器的未来发展指明了方向，强调了需要更准确和更稳健的检测方法。



## **7. Defense without Forgetting: Continual Adversarial Defense with Anisotropic & Isotropic Pseudo Replay**

不忘防御：具有各向异性和各向同性伪重放的连续对抗防御 cs.LG

**SubmitDate**: 2024-04-02    [abs](http://arxiv.org/abs/2404.01828v1) [paper-pdf](http://arxiv.org/pdf/2404.01828v1)

**Authors**: Yuhang Zhou, Zhongyun Hua

**Abstract**: Deep neural networks have demonstrated susceptibility to adversarial attacks. Adversarial defense techniques often focus on one-shot setting to maintain robustness against attack. However, new attacks can emerge in sequences in real-world deployment scenarios. As a result, it is crucial for a defense model to constantly adapt to new attacks, but the adaptation process can lead to catastrophic forgetting of previously defended against attacks. In this paper, we discuss for the first time the concept of continual adversarial defense under a sequence of attacks, and propose a lifelong defense baseline called Anisotropic \& Isotropic Replay (AIR), which offers three advantages: (1) Isotropic replay ensures model consistency in the neighborhood distribution of new data, indirectly aligning the output preference between old and new tasks. (2) Anisotropic replay enables the model to learn a compromise data manifold with fresh mixed semantics for further replay constraints and potential future attacks. (3) A straightforward regularizer mitigates the 'plasticity-stability' trade-off by aligning model output between new and old tasks. Experiment results demonstrate that AIR can approximate or even exceed the empirical performance upper bounds achieved by Joint Training.

摘要: 深度神经网络已显示出对敌意攻击的敏感性。对抗性防守技术通常集中在一次射击的设置上，以保持对攻击的健壮性。然而，在现实世界的部署场景中，新的攻击可能会按顺序出现。因此，对于防御模型来说，不断适应新的攻击是至关重要的，但适应过程可能会导致灾难性地忘记以前防御攻击的方式。本文首次讨论了一系列攻击下的连续对抗防御的概念，并提出了一种称为各向异性和各向同性重放(AIR)的终身防御基线，它具有三个优点：(1)各向同性重放保证了新数据在邻域分布上的模型一致性，间接地对齐了新旧任务之间的输出偏好。(2)各向异性重放使模型能够学习具有新鲜混合语义的折衷数据流形，用于进一步的重放约束和潜在的未来攻击。(3)通过调整新任务和旧任务之间的模型输出，直接的正则化可以缓解“塑性-稳定性”之间的权衡。实验结果表明，AIR可以接近甚至超过联合训练所获得的经验性能上限。



## **8. Security Allocation in Networked Control Systems under Stealthy Attacks**

隐身攻击下网络控制系统的安全分配 eess.SY

12 pages, 3 figures, and 1 table, journal submission

**SubmitDate**: 2024-04-02    [abs](http://arxiv.org/abs/2308.16639v2) [paper-pdf](http://arxiv.org/pdf/2308.16639v2)

**Authors**: Anh Tung Nguyen, André M. H. Teixeira, Alexander Medvedev

**Abstract**: This paper considers the problem of security allocation in a networked control system under stealthy attacks. The system is comprised of interconnected subsystems represented by vertices. A malicious adversary selects a single vertex on which to conduct a stealthy data injection attack with the purpose of maximally disrupting a distant target vertex while remaining undetected. Defense resources against the adversary are allocated by a defender on several selected vertices. First, the objectives of the adversary and the defender with uncertain targets are formulated in a probabilistic manner, resulting in an expected worst-case impact of stealthy attacks. Next, we provide a graph-theoretic necessary and sufficient condition under which the cost for the defender and the expected worst-case impact of stealthy attacks are bounded. This condition enables the defender to restrict the admissible actions to dominating sets of the graph representing the network. Then, the security allocation problem is solved through a Stackelberg game-theoretic framework. Finally, the obtained results are validated through a numerical example of a 50-vertex networked control system.

摘要: 研究了网络控制系统在隐身攻击下的安全分配问题。该系统由由顶点表示的相互连接的子系统组成。恶意攻击者选择单个顶点在其上进行隐形数据注入攻击，目的是在保持未被检测的情况下最大限度地破坏远处的目标顶点。针对对手的防御资源由防御者在几个选定的顶点上分配。首先，目标不确定的对手和防御者的目标是以概率的方式制定的，导致了预期的最坏情况下的隐形攻击影响。接下来，我们给出了一个图论的充要条件，在这个充要条件下，防御者的代价和隐身攻击的预期最坏影响是有界的。这一条件使防御者能够将允许的动作限制在表示网络的图的支配集上。然后，通过Stackelberg博弈论框架解决了安全分配问题。最后，通过一个50点网络控制系统的数值算例对所得结果进行了验证。



## **9. ADVREPAIR:Provable Repair of Adversarial Attack**

ADVREPAIR：对抗攻击的可证明修复 cs.LG

**SubmitDate**: 2024-04-02    [abs](http://arxiv.org/abs/2404.01642v1) [paper-pdf](http://arxiv.org/pdf/2404.01642v1)

**Authors**: Zhiming Chi, Jianan Ma, Pengfei Yang, Cheng-Chao Huang, Renjue Li, Xiaowei Huang, Lijun Zhang

**Abstract**: Deep neural networks (DNNs) are increasingly deployed in safety-critical domains, but their vulnerability to adversarial attacks poses serious safety risks. Existing neuron-level methods using limited data lack efficacy in fixing adversaries due to the inherent complexity of adversarial attack mechanisms, while adversarial training, leveraging a large number of adversarial samples to enhance robustness, lacks provability. In this paper, we propose ADVREPAIR, a novel approach for provable repair of adversarial attacks using limited data. By utilizing formal verification, ADVREPAIR constructs patch modules that, when integrated with the original network, deliver provable and specialized repairs within the robustness neighborhood. Additionally, our approach incorporates a heuristic mechanism for assigning patch modules, allowing this defense against adversarial attacks to generalize to other inputs. ADVREPAIR demonstrates superior efficiency, scalability and repair success rate. Different from existing DNN repair methods, our repair can generalize to general inputs, thereby improving the robustness of the neural network globally, which indicates a significant breakthrough in the generalization capability of ADVREPAIR.

摘要: 深度神经网络(DNN)越来越多地被部署在安全关键领域，但它们对对手攻击的脆弱性构成了严重的安全风险。由于对抗性攻击机制的内在复杂性，现有的利用有限数据的神经元级别的方法在固定对手方面缺乏有效性，而对抗性训练利用大量的对抗性样本来增强稳健性，缺乏可证性。本文提出了一种利用有限数据可证明修复对抗性攻击的新方法--ADVREPAIR。通过使用正式验证，ADVREPAIR构建了补丁模块，当与原始网络集成时，可在健壮性邻域内提供可证明和专门的修复。此外，我们的方法结合了分配补丁模块的启发式机制，允许这种针对对手攻击的防御推广到其他输入。ADVREPAIR表现出卓越的效率、可扩展性和修复成功率。与现有的DNN修复方法不同，我们的修复方法可以推广到一般输入，从而提高了神经网络的全局鲁棒性，这表明ADVREPAIR在泛化能力方面取得了重大突破。



## **10. Multi-granular Adversarial Attacks against Black-box Neural Ranking Models**

黑盒神经排序模型的多粒度对抗攻击 cs.IR

Accepted by SIGIR 2024

**SubmitDate**: 2024-04-02    [abs](http://arxiv.org/abs/2404.01574v1) [paper-pdf](http://arxiv.org/pdf/2404.01574v1)

**Authors**: Yu-An Liu, Ruqing Zhang, Jiafeng Guo, Maarten de Rijke, Yixing Fan, Xueqi Cheng

**Abstract**: Adversarial ranking attacks have gained increasing attention due to their success in probing vulnerabilities, and, hence, enhancing the robustness, of neural ranking models. Conventional attack methods employ perturbations at a single granularity, e.g., word-level or sentence-level, to a target document. However, limiting perturbations to a single level of granularity may reduce the flexibility of creating adversarial examples, thereby diminishing the potential threat of the attack. Therefore, we focus on generating high-quality adversarial examples by incorporating multi-granular perturbations. Achieving this objective involves tackling a combinatorial explosion problem, which requires identifying an optimal combination of perturbations across all possible levels of granularity, positions, and textual pieces. To address this challenge, we transform the multi-granular adversarial attack into a sequential decision-making process, where perturbations in the next attack step are influenced by the perturbed document in the current attack step. Since the attack process can only access the final state without direct intermediate signals, we use reinforcement learning to perform multi-granular attacks. During the reinforcement learning process, two agents work cooperatively to identify multi-granular vulnerabilities as attack targets and organize perturbation candidates into a final perturbation sequence. Experimental results show that our attack method surpasses prevailing baselines in both attack effectiveness and imperceptibility.

摘要: 对抗性排序攻击因其在探测漏洞方面的成功，从而增强了神经排序模型的稳健性而受到越来越多的关注。传统的攻击方法对目标文档采用单一粒度的扰动，例如单词级或句子级。然而，将扰动限制在单一的粒度级别可能会降低创建对抗性示例的灵活性，从而降低攻击的潜在威胁。因此，我们专注于通过结合多粒度扰动来生成高质量的对抗性实例。实现这一目标需要处理组合爆炸问题，这需要确定跨所有可能级别的粒度、位置和文本片段的扰动的最佳组合。为了应对这一挑战，我们将多粒度的对抗性攻击转化为一个连续的决策过程，其中下一攻击步骤中的扰动受到当前攻击步骤中扰动文档的影响。由于攻击过程只能访问最终状态，没有直接的中间信号，因此我们使用强化学习来执行多粒度攻击。在强化学习过程中，两个代理协作识别多粒度漏洞作为攻击目标，并将扰动候选组织成最终的扰动序列。实验结果表明，我们的攻击方法在攻击有效性和不可感知性方面都超过了主流基线。



## **11. MMCert: Provable Defense against Adversarial Attacks to Multi-modal Models**

MMCert：针对多模态模型的对抗攻击的可证明防御 cs.CV

To appear in CVPR'24

**SubmitDate**: 2024-04-02    [abs](http://arxiv.org/abs/2403.19080v3) [paper-pdf](http://arxiv.org/pdf/2403.19080v3)

**Authors**: Yanting Wang, Hongye Fu, Wei Zou, Jinyuan Jia

**Abstract**: Different from a unimodal model whose input is from a single modality, the input (called multi-modal input) of a multi-modal model is from multiple modalities such as image, 3D points, audio, text, etc. Similar to unimodal models, many existing studies show that a multi-modal model is also vulnerable to adversarial perturbation, where an attacker could add small perturbation to all modalities of a multi-modal input such that the multi-modal model makes incorrect predictions for it. Existing certified defenses are mostly designed for unimodal models, which achieve sub-optimal certified robustness guarantees when extended to multi-modal models as shown in our experimental results. In our work, we propose MMCert, the first certified defense against adversarial attacks to a multi-modal model. We derive a lower bound on the performance of our MMCert under arbitrary adversarial attacks with bounded perturbations to both modalities (e.g., in the context of auto-driving, we bound the number of changed pixels in both RGB image and depth image). We evaluate our MMCert using two benchmark datasets: one for the multi-modal road segmentation task and the other for the multi-modal emotion recognition task. Moreover, we compare our MMCert with a state-of-the-art certified defense extended from unimodal models. Our experimental results show that our MMCert outperforms the baseline.

摘要: 与单通道模型的输入来自单一通道不同，多通道模型的输入(称为多通道输入)来自图像、3D点、音频、文本等多个通道。与单通道模型类似，许多现有的研究表明，多通道模型也容易受到对抗性扰动的影响，攻击者可以在多通道输入的所有通道中添加小的扰动，从而使得多通道模型对其做出错误的预测。现有的认证防御大多是针对单模模型设计的，如我们的实验结果所示，当扩展到多模模型时，它们获得了次优的认证稳健性保证。在我们的工作中，我们提出了MMCert，这是第一个认证的多模式对抗攻击防御模型。我们得到了MMCert在两种模式都有界扰动的任意攻击下的性能下界(例如，在自动驾驶的背景下，我们限制了RGB图像和深度图像中变化的像素数量)。我们使用两个基准数据集来评估我们的MMCert：一个用于多模式道路分割任务，另一个用于多模式情感识别任务。此外，我们将我们的MMCert与从单模模型扩展而来的最先进的认证防御进行了比较。我们的实验结果表明，我们的MMCert的性能优于基线。



## **12. Rumor Detection with a novel graph neural network approach**

基于图神经网络的谣言检测方法 cs.AI

10 pages, 5 figures

**SubmitDate**: 2024-04-02    [abs](http://arxiv.org/abs/2403.16206v3) [paper-pdf](http://arxiv.org/pdf/2403.16206v3)

**Authors**: Tianrui Liu, Qi Cai, Changxin Xu, Bo Hong, Fanghao Ni, Yuxin Qiao, Tsungwei Yang

**Abstract**: The wide spread of rumors on social media has caused a negative impact on people's daily life, leading to potential panic, fear, and mental health problems for the public. How to debunk rumors as early as possible remains a challenging problem. Existing studies mainly leverage information propagation structure to detect rumors, while very few works focus on correlation among users that they may coordinate to spread rumors in order to gain large popularity. In this paper, we propose a new detection model, that jointly learns both the representations of user correlation and information propagation to detect rumors on social media. Specifically, we leverage graph neural networks to learn the representations of user correlation from a bipartite graph that describes the correlations between users and source tweets, and the representations of information propagation with a tree structure. Then we combine the learned representations from these two modules to classify the rumors. Since malicious users intend to subvert our model after deployment, we further develop a greedy attack scheme to analyze the cost of three adversarial attacks: graph attack, comment attack, and joint attack. Evaluation results on two public datasets illustrate that the proposed MODEL outperforms the state-of-the-art rumor detection models. We also demonstrate our method performs well for early rumor detection. Moreover, the proposed detection method is more robust to adversarial attacks compared to the best existing method. Importantly, we show that it requires a high cost for attackers to subvert user correlation pattern, demonstrating the importance of considering user correlation for rumor detection.

摘要: 谣言在社交媒体上的广泛传播对人们的日常生活造成了负面影响，给公众带来了潜在的恐慌、恐惧和心理健康问题。如何尽早揭穿谣言仍是一个具有挑战性的问题。现有的研究主要是利用信息传播结构来发现谣言，而很少有人关注用户之间的相关性，他们可能会协同传播谣言以获得更大的人气。在本文中，我们提出了一种新的检测模型，该模型同时学习用户相关性和信息传播的表示，以检测社交媒体上的谣言。具体地说，我们利用图神经网络从描述用户和源推文之间的相关性的二部图中学习用户相关性的表示，以及用树结构表示信息传播。然后，我们结合这两个模块的学习表示来对谣言进行分类。由于恶意用户在部署后有意颠覆我们的模型，我们进一步开发了一种贪婪攻击方案，分析了图攻击、评论攻击和联合攻击三种对抗性攻击的代价。在两个公开数据集上的评估结果表明，该模型的性能优于最新的谣言检测模型。我们还证明了我们的方法在早期谣言检测中表现良好。此外，与现有的最佳检测方法相比，本文提出的检测方法对敌意攻击具有更强的鲁棒性。重要的是，我们证明了攻击者要颠覆用户相关性模式需要付出很高的代价，这说明了考虑用户相关性对谣言检测的重要性。



## **13. Vulnerabilities of Foundation Model Integrated Federated Learning Under Adversarial Threats**

对抗威胁下的基础模型集成联邦学习的脆弱性 cs.CR

Chen Wu and Xi Li are equal contribution. The corresponding author is  Jiaqi Wang

**SubmitDate**: 2024-04-02    [abs](http://arxiv.org/abs/2401.10375v2) [paper-pdf](http://arxiv.org/pdf/2401.10375v2)

**Authors**: Chen Wu, Xi Li, Jiaqi Wang

**Abstract**: Federated Learning (FL) addresses critical issues in machine learning related to data privacy and security, yet suffering from data insufficiency and imbalance under certain circumstances. The emergence of foundation models (FMs) offers potential solutions to the limitations of existing FL frameworks, e.g., by generating synthetic data for model initialization. However, due to the inherent safety concerns of FMs, integrating FMs into FL could introduce new risks, which remains largely unexplored. To address this gap, we conduct the first investigation on the vulnerability of FM integrated FL (FM-FL) under adversarial threats. Based on a unified framework of FM-FL, we introduce a novel attack strategy that exploits safety issues of FM to compromise FL client models. Through extensive experiments with well-known models and benchmark datasets in both image and text domains, we reveal the high susceptibility of the FM-FL to this new threat under various FL configurations. Furthermore, we find that existing FL defense strategies offer limited protection against this novel attack approach. This research highlights the critical need for enhanced security measures in FL in the era of FMs.

摘要: 联合学习(FL)解决了机器学习中与数据隐私和安全相关的关键问题，但在某些情况下存在数据不足和不平衡的问题。基础模型(FM)的出现为现有FL框架的局限性提供了潜在的解决方案，例如通过生成用于模型初始化的合成数据。然而，由于FMS固有的安全问题，将FMS整合到FL中可能会带来新的风险，这在很大程度上仍未被探索。为了弥补这一差距，我们首次对FM集成FL(FM-FL)在对手威胁下的脆弱性进行了研究。基于FM-FL的统一框架，我们提出了一种新的攻击策略，利用FM的安全问题来危害FL客户端模型。通过在图像域和文本域使用著名的模型和基准数据集进行广泛的实验，我们揭示了FM-FL在不同FL配置下对这种新威胁的高度敏感性。此外，我们发现，现有的FL防御策略对这种新的攻击方法提供的保护有限。这项研究强调了在FMS时代加强FL安全措施的迫切需要。



## **14. Can Biases in ImageNet Models Explain Generalization?**

ImagNet模型中的偏差能解释泛化吗？ cs.CV

Accepted at CVPR2024

**SubmitDate**: 2024-04-01    [abs](http://arxiv.org/abs/2404.01509v1) [paper-pdf](http://arxiv.org/pdf/2404.01509v1)

**Authors**: Paul Gavrikov, Janis Keuper

**Abstract**: The robust generalization of models to rare, in-distribution (ID) samples drawn from the long tail of the training distribution and to out-of-training-distribution (OOD) samples is one of the major challenges of current deep learning methods. For image classification, this manifests in the existence of adversarial attacks, the performance drops on distorted images, and a lack of generalization to concepts such as sketches. The current understanding of generalization in neural networks is very limited, but some biases that differentiate models from human vision have been identified and might be causing these limitations. Consequently, several attempts with varying success have been made to reduce these biases during training to improve generalization. We take a step back and sanity-check these attempts. Fixing the architecture to the well-established ResNet-50, we perform a large-scale study on 48 ImageNet models obtained via different training methods to understand how and if these biases - including shape bias, spectral biases, and critical bands - interact with generalization. Our extensive study results reveal that contrary to previous findings, these biases are insufficient to accurately predict the generalization of a model holistically. We provide access to all checkpoints and evaluation code at https://github.com/paulgavrikov/biases_vs_generalization

摘要: 将模型推广到从训练分布的长尾中提取的稀有分布内(ID)样本和训练分布外(OOD)样本是当前深度学习方法的主要挑战之一。对于图像分类，这表现在存在对抗性攻击，对失真图像的性能下降，以及对草图等概念缺乏泛化。目前对神经网络泛化的理解非常有限，但已经发现了一些将模型与人类视觉区分开来的偏差，并可能导致这些限制。因此，已经进行了几次尝试，但取得了不同的成功，以减少培训期间的这些偏见，以改进泛化。我们退后一步，理智地检查这些尝试。将架构固定到成熟的ResNet-50，我们对通过不同训练方法获得的48个ImageNet模型进行了大规模研究，以了解这些偏差-包括形状偏差、光谱偏差和关键频带-如何以及是否与泛化相互作用。我们广泛的研究结果表明，与以前的发现相反，这些偏差不足以准确地整体预测模型的概括性。我们允许访问https://github.com/paulgavrikov/biases_vs_generalization上的所有检查点和评估代码



## **15. Defending Against Unforeseen Failure Modes with Latent Adversarial Training**

利用潜在对抗训练防御不可预见的故障模式 cs.CR

**SubmitDate**: 2024-04-01    [abs](http://arxiv.org/abs/2403.05030v3) [paper-pdf](http://arxiv.org/pdf/2403.05030v3)

**Authors**: Stephen Casper, Lennart Schulze, Oam Patel, Dylan Hadfield-Menell

**Abstract**: Despite extensive diagnostics and debugging by developers, AI systems sometimes exhibit harmful unintended behaviors. Finding and fixing these is challenging because the attack surface is so large -- it is not tractable to exhaustively search for inputs that may elicit harmful behaviors. Red-teaming and adversarial training (AT) are commonly used to improve robustness, however, they empirically struggle to fix failure modes that differ from the attacks used during training. In this work, we utilize latent adversarial training (LAT) to defend against vulnerabilities without generating inputs that elicit them. LAT leverages the compressed, abstract, and structured latent representations of concepts that the network actually uses for prediction. We use it to remove trojans and defend against held-out classes of adversarial attacks. We show in image classification, text classification, and text generation tasks that LAT usually improves both robustness to novel attacks and performance on clean data relative to AT. This suggests that LAT can be a promising tool for defending against failure modes that are not explicitly identified by developers.

摘要: 尽管开发人员进行了广泛的诊断和调试，但人工智能系统有时会表现出有害的意外行为。找到并修复这些攻击是具有挑战性的，因为攻击面太大了--要详尽地搜索可能引发有害行为的输入并不容易。红队和对抗性训练(AT)通常用于提高健壮性，然而，根据经验，它们难以修复与训练期间使用的攻击不同的失败模式。在这项工作中，我们利用潜在的对手训练(LAT)来防御漏洞，而不会生成引发漏洞的输入。随后，利用网络实际用于预测的概念的压缩、抽象和结构化的潜在表示。我们使用它来删除特洛伊木马程序，并防御抵抗类的对抗性攻击。我们在图像分类、文本分类和文本生成任务中表明，相对于AT，LAT通常可以提高对新攻击的健壮性和对干净数据的性能。这表明，LAT可以成为一种很有前途的工具，用于防御开发人员未明确识别的故障模式。



## **16. Robust One-Class Classification with Signed Distance Function using 1-Lipschitz Neural Networks**

基于1—Lipschitz神经网络的带符号距离函数单类分类 cs.LG

27 pages, 11 figures, International Conference on Machine Learning  2023, (ICML 2023)

**SubmitDate**: 2024-04-01    [abs](http://arxiv.org/abs/2303.01978v2) [paper-pdf](http://arxiv.org/pdf/2303.01978v2)

**Authors**: Louis Bethune, Paul Novello, Thibaut Boissin, Guillaume Coiffier, Mathieu Serrurier, Quentin Vincenot, Andres Troya-Galvis

**Abstract**: We propose a new method, dubbed One Class Signed Distance Function (OCSDF), to perform One Class Classification (OCC) by provably learning the Signed Distance Function (SDF) to the boundary of the support of any distribution. The distance to the support can be interpreted as a normality score, and its approximation using 1-Lipschitz neural networks provides robustness bounds against $l2$ adversarial attacks, an under-explored weakness of deep learning-based OCC algorithms. As a result, OCSDF comes with a new metric, certified AUROC, that can be computed at the same cost as any classical AUROC. We show that OCSDF is competitive against concurrent methods on tabular and image data while being way more robust to adversarial attacks, illustrating its theoretical properties. Finally, as exploratory research perspectives, we theoretically and empirically show how OCSDF connects OCC with image generation and implicit neural surface parametrization. Our code is available at https://github.com/Algue-Rythme/OneClassMetricLearning

摘要: 我们提出了一种新的方法，称为一类符号距离函数(OCSDF)，通过可证明地学习符号距离函数(SDF)到任意分布的支持度边界来执行一类分类(OCC)。到支持点的距离可以解释为正态得分，其使用1-Lipschitz神经网络的逼近提供了对$L2$对手攻击的稳健界，这是基于深度学习的OCC算法的一个未被充分挖掘的弱点。因此，OCSDF附带了一种新的衡量标准-认证AUROC，其计算成本可以与任何经典AUROC相同。我们证明了OCSDF在表格和图像数据上与并发方法相比具有竞争力，同时对对手攻击具有更强的健壮性，说明了它的理论性质。最后，作为探索性的研究视角，我们从理论和经验上展示了OCSDF如何将OCC与图像生成和隐式神经表面参数化联系起来。我们的代码可以在https://github.com/Algue-Rythme/OneClassMetricLearning上找到



## **17. The twin peaks of learning neural networks**

学习神经网络的双峰 cs.LG

37 pages, 31 figures

**SubmitDate**: 2024-04-01    [abs](http://arxiv.org/abs/2401.12610v2) [paper-pdf](http://arxiv.org/pdf/2401.12610v2)

**Authors**: Elizaveta Demyanenko, Christoph Feinauer, Enrico M. Malatesta, Luca Saglietti

**Abstract**: Recent works demonstrated the existence of a double-descent phenomenon for the generalization error of neural networks, where highly overparameterized models escape overfitting and achieve good test performance, at odds with the standard bias-variance trade-off described by statistical learning theory. In the present work, we explore a link between this phenomenon and the increase of complexity and sensitivity of the function represented by neural networks. In particular, we study the Boolean mean dimension (BMD), a metric developed in the context of Boolean function analysis. Focusing on a simple teacher-student setting for the random feature model, we derive a theoretical analysis based on the replica method that yields an interpretable expression for the BMD, in the high dimensional regime where the number of data points, the number of features, and the input size grow to infinity. We find that, as the degree of overparameterization of the network is increased, the BMD reaches an evident peak at the interpolation threshold, in correspondence with the generalization error peak, and then slowly approaches a low asymptotic value. The same phenomenology is then traced in numerical experiments with different model classes and training setups. Moreover, we find empirically that adversarially initialized models tend to show higher BMD values, and that models that are more robust to adversarial attacks exhibit a lower BMD.

摘要: 最近的工作证明了神经网络泛化误差存在双下降现象，即高度过参数的模型避免了过拟合并获得了良好的测试性能，这与统计学习理论所描述的标准偏差-方差权衡不一致。在目前的工作中，我们探索了这种现象与神经网络表示的函数的复杂性和敏感度的增加之间的联系。特别是，我们研究了布尔平均维度(BMD)，这是在布尔函数分析的背景下发展起来的一种度量。针对一个简单的教师-学生随机特征模型，我们基于复制品方法进行了理论分析，在数据点数目、特征数目和输入大小都增长到无穷大的高维区域中，给出了一个可解释的BMD表达式。我们发现，随着网络的超参数化程度的增加，BMD在与泛化误差峰值相对应的内插阈值处达到一个明显的峰值，然后缓慢地接近一个较低的渐近值。然后在不同模型类别和训练设置的数值实验中追踪相同的现象学。此外，我们从经验上发现，对抗性初始化的模型往往显示出较高的BMD值，而对对抗性攻击越健壮的模型显示出较低的BMD。



## **18. Foundations of Cyber Resilience: The Confluence of Game, Control, and Learning Theories**

网络弹性的基础：游戏、控制和学习理论的融合 eess.SY

**SubmitDate**: 2024-04-01    [abs](http://arxiv.org/abs/2404.01205v1) [paper-pdf](http://arxiv.org/pdf/2404.01205v1)

**Authors**: Quanyan Zhu

**Abstract**: Cyber resilience is a complementary concept to cybersecurity, focusing on the preparation, response, and recovery from cyber threats that are challenging to prevent. Organizations increasingly face such threats in an evolving cyber threat landscape. Understanding and establishing foundations for cyber resilience provide a quantitative and systematic approach to cyber risk assessment, mitigation policy evaluation, and risk-informed defense design. A systems-scientific view toward cyber risks provides holistic and system-level solutions. This chapter starts with a systemic view toward cyber risks and presents the confluence of game theory, control theory, and learning theories, which are three major pillars for the design of cyber resilience mechanisms to counteract increasingly sophisticated and evolving threats in our networks and organizations. Game and control theoretic methods provide a set of modeling frameworks to capture the strategic and dynamic interactions between defenders and attackers. Control and learning frameworks together provide a feedback-driven mechanism that enables autonomous and adaptive responses to threats. Game and learning frameworks offer a data-driven approach to proactively reason about adversarial behaviors and resilient strategies. The confluence of the three lays the theoretical foundations for the analysis and design of cyber resilience. This chapter presents various theoretical paradigms, including dynamic asymmetric games, moving horizon control, conjectural learning, and meta-learning, as recent advances at the intersection. This chapter concludes with future directions and discussions of the role of neurosymbolic learning and the synergy between foundation models and game models in cyber resilience.

摘要: 网络复原力是网络安全的补充概念，侧重于预防具有挑战性的网络威胁的准备、响应和恢复。在不断变化的网络威胁环境中，组织面临的此类威胁越来越多。理解和建立网络复原力的基础为网络风险评估、缓解政策评估和风险知情防御设计提供了一种量化和系统的方法。系统科学的网络风险观提供了整体和系统级的解决方案。本章从系统地看待网络风险开始，介绍了博弈论、控制论和学习理论的融合，这三个理论是设计网络弹性机制的三大支柱，以对抗我们网络和组织中日益复杂和不断变化的威胁。博弈论和控制论方法提供了一套模型框架来捕捉防御者和攻击者之间的战略和动态交互。控制和学习框架共同提供了一种反馈驱动的机制，使其能够对威胁做出自主和适应性的反应。游戏和学习框架提供了一种数据驱动的方法来主动推理对手行为和弹性策略。三者的融合为网络韧性的分析和设计奠定了理论基础。本章介绍了各种理论范式，包括动态不对称博弈、移动视野控制、猜想学习和元学习，作为交叉路口的最新进展。本章最后对神经符号学习的作用以及基础模型和游戏模型在网络韧性中的协同作用进行了未来的方向和讨论。



## **19. The Best Defense is Attack: Repairing Semantics in Textual Adversarial Examples**

最好的防御是攻击：文本对抗示例中的语义修复 cs.CL

**SubmitDate**: 2024-04-01    [abs](http://arxiv.org/abs/2305.04067v2) [paper-pdf](http://arxiv.org/pdf/2305.04067v2)

**Authors**: Heng Yang, Ke Li

**Abstract**: Recent studies have revealed the vulnerability of pre-trained language models to adversarial attacks. Existing adversarial defense techniques attempt to reconstruct adversarial examples within feature or text spaces. However, these methods struggle to effectively repair the semantics in adversarial examples, resulting in unsatisfactory performance and limiting their practical utility. To repair the semantics in adversarial examples, we introduce a novel approach named Reactive Perturbation Defocusing (Rapid). Rapid employs an adversarial detector to identify fake labels of adversarial examples and leverage adversarial attackers to repair the semantics in adversarial examples. Our extensive experimental results conducted on four public datasets, convincingly demonstrate the effectiveness of Rapid in various adversarial attack scenarios. To address the problem of defense performance validation in previous works, we provide a demonstration of adversarial detection and repair based on our work, which can be easily evaluated at https://tinyurl.com/22ercuf8.

摘要: 最近的研究揭示了预先训练的语言模型在对抗性攻击中的脆弱性。现有的对抗性防御技术试图在特征或文本空间内重建对抗性示例。然而，这些方法难以有效地修复对抗性实例中的语义，导致性能不佳，限制了它们的实用价值。为了修复对抗性例子中的语义，我们引入了一种新的方法--反应性扰动散焦(Rapid)。RAPID使用对抗性检测器来识别对抗性实例的虚假标签，并利用对抗性攻击者来修复对抗性实例中的语义。我们在四个公开数据集上进行的大量实验结果令人信服地证明了Rapid在各种对抗性攻击场景中的有效性。为了解决前人工作中的防御性能验证问题，我们在工作的基础上提供了一个对手检测和修复的演示，该演示可以在https://tinyurl.com/22ercuf8.上轻松地进行评估



## **20. Poisoning Decentralized Collaborative Recommender System and Its Countermeasures**

分布式协同推荐系统中毒及其对策 cs.CR

**SubmitDate**: 2024-04-01    [abs](http://arxiv.org/abs/2404.01177v1) [paper-pdf](http://arxiv.org/pdf/2404.01177v1)

**Authors**: Ruiqi Zheng, Liang Qu, Tong Chen, Kai Zheng, Yuhui Shi, Hongzhi Yin

**Abstract**: To make room for privacy and efficiency, the deployment of many recommender systems is experiencing a shift from central servers to personal devices, where the federated recommender systems (FedRecs) and decentralized collaborative recommender systems (DecRecs) are arguably the two most representative paradigms. While both leverage knowledge (e.g., gradients) sharing to facilitate learning local models, FedRecs rely on a central server to coordinate the optimization process, yet in DecRecs, the knowledge sharing directly happens between clients. Knowledge sharing also opens a backdoor for model poisoning attacks, where adversaries disguise themselves as benign clients and disseminate polluted knowledge to achieve malicious goals like promoting an item's exposure rate. Although research on such poisoning attacks provides valuable insights into finding security loopholes and corresponding countermeasures, existing attacks mostly focus on FedRecs, and are either inapplicable or ineffective for DecRecs. Compared with FedRecs where the tampered information can be universally distributed to all clients once uploaded to the cloud, each adversary in DecRecs can only communicate with neighbor clients of a small size, confining its impact to a limited range. To fill the gap, we present a novel attack method named Poisoning with Adaptive Malicious Neighbors (PAMN). With item promotion in top-K recommendation as the attack objective, PAMN effectively boosts target items' ranks with several adversaries that emulate benign clients and transfers adaptively crafted gradients conditioned on each adversary's neighbors. Moreover, with the vulnerabilities of DecRecs uncovered, a dedicated defensive mechanism based on user-level gradient clipping with sparsified updating is proposed. Extensive experiments demonstrate the effectiveness of the poisoning attack and the robustness of our defensive mechanism.

摘要: 为了给隐私和效率腾出空间，许多推荐系统的部署正在经历从中央服务器到个人设备的转变，其中联合推荐系统(FedRecs)和分散协作推荐系统(DecRecs)可以说是两个最具代表性的范例。虽然两者都利用知识(例如，梯度)共享来促进学习本地模型，但FedRecs依赖中央服务器来协调优化过程，而在DecRecs中，知识共享直接发生在客户之间。知识共享也为模型中毒攻击打开了后门，对手将自己伪装成良性客户，传播受污染的知识，以实现恶意目标，如提高物品的曝光率。虽然对这类中毒攻击的研究为发现安全漏洞和相应的对策提供了宝贵的见解，但现有的攻击大多集中在FedRecs上，不适用于DECRecs，或者对DECRecs无效。与FedRecs中被篡改的信息一旦上传到云中就可以统一分发到所有客户端相比，DecRecs中的每个对手只能与小规模的邻居客户端通信，将其影响限制在有限的范围内。为了填补这一空白，我们提出了一种新的攻击方法--自适应恶意邻居投毒攻击(PAMN)。以TOP-K推荐中的条目推广为攻击目标，PAMN有效地提升了目标条目的排名，多个对手模仿良性客户端，并根据每个对手的邻居自适应地传输定制的梯度。此外，针对DecRecs存在的漏洞，提出了一种基于稀疏更新的用户级梯度裁剪的专用防御机制。大量的实验证明了中毒攻击的有效性和我们防御机制的健壮性。



## **21. The Double-Edged Sword of Input Perturbations to Robust Accurate Fairness**

输入扰动的双刃剑实现鲁棒精确公平 cs.LG

**SubmitDate**: 2024-04-01    [abs](http://arxiv.org/abs/2404.01356v1) [paper-pdf](http://arxiv.org/pdf/2404.01356v1)

**Authors**: Xuran Li, Peng Wu, Yanting Chen, Xingjun Ma, Zhen Zhang, Kaixiang Dong

**Abstract**: Deep neural networks (DNNs) are known to be sensitive to adversarial input perturbations, leading to a reduction in either prediction accuracy or individual fairness. To jointly characterize the susceptibility of prediction accuracy and individual fairness to adversarial perturbations, we introduce a novel robustness definition termed robust accurate fairness. Informally, robust accurate fairness requires that predictions for an instance and its similar counterparts consistently align with the ground truth when subjected to input perturbations. We propose an adversarial attack approach dubbed RAFair to expose false or biased adversarial defects in DNN, which either deceive accuracy or compromise individual fairness. Then, we show that such adversarial instances can be effectively addressed by carefully designed benign perturbations, correcting their predictions to be accurate and fair. Our work explores the double-edged sword of input perturbations to robust accurate fairness in DNN and the potential of using benign perturbations to correct adversarial instances.

摘要: 深度神经网络(DNN)对敌意输入扰动非常敏感，导致预测精度或个体公平性降低。为了联合刻画预测精度和个体公平性对对抗扰动的敏感性，我们引入了一种新的健壮性定义，称为鲁棒准确公平性。非正式地讲，稳健准确的公平性要求在受到输入扰动时，对实例及其类似实例的预测与基本事实一致。我们提出了一种称为RAFair的对抗性攻击方法，以揭露DNN中虚假或有偏见的对抗性缺陷，这些缺陷要么欺骗准确性，要么损害个体公平性。然后，我们证明了这样的对抗性实例可以通过精心设计的良性扰动来有效地解决，修正他们的预测是准确和公平的。我们的工作探索了输入扰动对DNN中稳健的准确公平性的双刃剑，以及使用良性扰动来纠正敌对实例的可能性。



## **22. Red Teaming Game: A Game-Theoretic Framework for Red Teaming Language Models**

红色团队游戏：红色团队语言模型的游戏理论框架 cs.CL

**SubmitDate**: 2024-04-01    [abs](http://arxiv.org/abs/2310.00322v3) [paper-pdf](http://arxiv.org/pdf/2310.00322v3)

**Authors**: Chengdong Ma, Ziran Yang, Minquan Gao, Hai Ci, Jun Gao, Xuehai Pan, Yaodong Yang

**Abstract**: Deployable Large Language Models (LLMs) must conform to the criterion of helpfulness and harmlessness, thereby achieving consistency between LLMs outputs and human values. Red-teaming techniques constitute a critical way towards this criterion. Existing work rely solely on manual red team designs and heuristic adversarial prompts for vulnerability detection and optimization. These approaches lack rigorous mathematical formulation, thus limiting the exploration of diverse attack strategy within quantifiable measure and optimization of LLMs under convergence guarantees. In this paper, we present Red-teaming Game (RTG), a general game-theoretic framework without manual annotation. RTG is designed for analyzing the multi-turn attack and defense interactions between Red-team language Models (RLMs) and Blue-team Language Model (BLM). Within the RTG, we propose Gamified Red-teaming Solver (GRTS) with diversity measure of the semantic space. GRTS is an automated red teaming technique to solve RTG towards Nash equilibrium through meta-game analysis, which corresponds to the theoretically guaranteed optimization direction of both RLMs and BLM. Empirical results in multi-turn attacks with RLMs show that GRTS autonomously discovered diverse attack strategies and effectively improved security of LLMs, outperforming existing heuristic red-team designs. Overall, RTG has established a foundational framework for red teaming tasks and constructed a new scalable oversight technique for alignment.

摘要: 可部署的大型语言模型(LLMS)必须符合有益和无害的标准，从而实现LLMS的输出与人的价值之间的一致性。红团队技术构成了实现这一标准的关键途径。现有的工作完全依赖于手动红色团队设计和启发式对抗性提示来进行漏洞检测和优化。这些方法缺乏严格的数学描述，从而限制了在可量化的度量范围内探索多样化的攻击策略，以及在收敛保证下对LLMS进行优化。在本文中，我们提出了一种不需要人工注释的通用博弈论框架--Red-Teaming Game(RTG)。RTG用于分析红队语言模型(RLMS)和蓝队语言模型(BLM)之间的多回合攻防交互。在RTG中，我们提出了一种具有语义空间多样性度量的Gamalized Red-Teaming Solver(GRTS)。GRTS是一种自动红队技术，通过元博弈分析解决RTG向纳什均衡的方向，这对应于理论上保证的RLMS和BLM的优化方向。在RLMS多回合攻击中的实验结果表明，GRTS自主发现多样化的攻击策略，有效地提高了LLMS的安全性，优于已有的启发式红队设计。总体而言，RTG为红色团队任务建立了一个基本框架，并构建了一种新的可扩展的协调监督技术。



## **23. LogoStyleFool: Vitiating Video Recognition Systems via Logo Style Transfer**

LogoStyleFool：通过标志风格转移来削弱视频识别系统 cs.CV

14 pages, 3 figures. Accepted to AAAI 2024

**SubmitDate**: 2024-04-01    [abs](http://arxiv.org/abs/2312.09935v2) [paper-pdf](http://arxiv.org/pdf/2312.09935v2)

**Authors**: Yuxin Cao, Ziyu Zhao, Xi Xiao, Derui Wang, Minhui Xue, Jin Lu

**Abstract**: Video recognition systems are vulnerable to adversarial examples. Recent studies show that style transfer-based and patch-based unrestricted perturbations can effectively improve attack efficiency. These attacks, however, face two main challenges: 1) Adding large stylized perturbations to all pixels reduces the naturalness of the video and such perturbations can be easily detected. 2) Patch-based video attacks are not extensible to targeted attacks due to the limited search space of reinforcement learning that has been widely used in video attacks recently. In this paper, we focus on the video black-box setting and propose a novel attack framework named LogoStyleFool by adding a stylized logo to the clean video. We separate the attack into three stages: style reference selection, reinforcement-learning-based logo style transfer, and perturbation optimization. We solve the first challenge by scaling down the perturbation range to a regional logo, while the second challenge is addressed by complementing an optimization stage after reinforcement learning. Experimental results substantiate the overall superiority of LogoStyleFool over three state-of-the-art patch-based attacks in terms of attack performance and semantic preservation. Meanwhile, LogoStyleFool still maintains its performance against two existing patch-based defense methods. We believe that our research is beneficial in increasing the attention of the security community to such subregional style transfer attacks.

摘要: 视频识别系统很容易受到敌意例子的攻击。最近的研究表明，基于风格迁移和基于补丁的无限制扰动可以有效地提高攻击效率。然而，这些攻击面临两个主要挑战：1)向所有像素添加大的风格化扰动会降低视频的自然度，并且这种扰动很容易被检测到。2)基于补丁的视频攻击不能扩展到有针对性的攻击，因为强化学习的搜索空间有限，这是近年来在视频攻击中广泛使用的。本文针对视频黑盒的设置，通过在干净的视频中添加一个风格化的标识，提出了一种新的攻击框架--LogoStyleFool。我们将攻击分为三个阶段：样式参考选择、基于强化学习的标识样式迁移和扰动优化。我们通过将扰动范围缩小到区域标志来解决第一个挑战，而第二个挑战是通过在强化学习后补充优化阶段来解决的。实验结果表明，在攻击性能和语义保持方面，LogoStyleFool在攻击性能和语义保持方面都优于三种最先进的基于补丁的攻击。同时，与现有的两种基于补丁的防御方法相比，LogoStyleFool仍然保持其性能。我们认为，我们的研究有助于提高安全界对这种次区域风格的转移袭击的关注。



## **24. StyleFool: Fooling Video Classification Systems via Style Transfer**

StyleFool：通过风格转移欺骗视频分类系统 cs.CV

18 pages, 9 figures. Accepted to S&P 2023

**SubmitDate**: 2024-04-01    [abs](http://arxiv.org/abs/2203.16000v4) [paper-pdf](http://arxiv.org/pdf/2203.16000v4)

**Authors**: Yuxin Cao, Xi Xiao, Ruoxi Sun, Derui Wang, Minhui Xue, Sheng Wen

**Abstract**: Video classification systems are vulnerable to adversarial attacks, which can create severe security problems in video verification. Current black-box attacks need a large number of queries to succeed, resulting in high computational overhead in the process of attack. On the other hand, attacks with restricted perturbations are ineffective against defenses such as denoising or adversarial training. In this paper, we focus on unrestricted perturbations and propose StyleFool, a black-box video adversarial attack via style transfer to fool the video classification system. StyleFool first utilizes color theme proximity to select the best style image, which helps avoid unnatural details in the stylized videos. Meanwhile, the target class confidence is additionally considered in targeted attacks to influence the output distribution of the classifier by moving the stylized video closer to or even across the decision boundary. A gradient-free method is then employed to further optimize the adversarial perturbations. We carry out extensive experiments to evaluate StyleFool on two standard datasets, UCF-101 and HMDB-51. The experimental results demonstrate that StyleFool outperforms the state-of-the-art adversarial attacks in terms of both the number of queries and the robustness against existing defenses. Moreover, 50% of the stylized videos in untargeted attacks do not need any query since they can already fool the video classification model. Furthermore, we evaluate the indistinguishability through a user study to show that the adversarial samples of StyleFool look imperceptible to human eyes, despite unrestricted perturbations.

摘要: 视频分类系统容易受到敌意攻击，这会给视频验证带来严重的安全问题。当前的黑盒攻击需要大量的查询才能成功，导致攻击过程中的计算开销很高。另一方面，受限扰动的攻击对诸如去噪或对抗性训练等防御措施无效。本文针对无限制扰动，提出了StyleFool，一种通过风格转移来欺骗视频分类系统的黑盒视频对抗性攻击。StyleFool首先利用颜色主题贴近度来选择最佳风格的图像，这有助于避免风格化视频中不自然的细节。同时，在有针对性的攻击中，还考虑了目标类置信度，通过将风格化视频移动到更接近甚至跨越决策边界的位置来影响分类器的输出分布。然后使用无梯度方法进一步优化对抗性扰动。我们在两个标准数据集UCF-101和HMDB-51上进行了大量的实验来评估StyleFool。实验结果表明，StyleFool在查询次数和对现有防御的健壮性方面都优于最先进的对抗性攻击。此外，在非定向攻击中，50%的风格化视频不需要任何查询，因为它们已经可以愚弄视频分类模型。此外，我们通过用户研究对StyleFool的不可区分性进行了评估，以表明StyleFool的敌意样本在人眼看来是不可察觉的，尽管存在无限的扰动。



## **25. BadPart: Unified Black-box Adversarial Patch Attacks against Pixel-wise Regression Tasks**

BadPart：针对像素回归任务的统一黑盒对抗补丁攻击 cs.CV

**SubmitDate**: 2024-04-01    [abs](http://arxiv.org/abs/2404.00924v1) [paper-pdf](http://arxiv.org/pdf/2404.00924v1)

**Authors**: Zhiyuan Cheng, Zhaoyi Liu, Tengda Guo, Shiwei Feng, Dongfang Liu, Mingjie Tang, Xiangyu Zhang

**Abstract**: Pixel-wise regression tasks (e.g., monocular depth estimation (MDE) and optical flow estimation (OFE)) have been widely involved in our daily life in applications like autonomous driving, augmented reality and video composition. Although certain applications are security-critical or bear societal significance, the adversarial robustness of such models are not sufficiently studied, especially in the black-box scenario. In this work, we introduce the first unified black-box adversarial patch attack framework against pixel-wise regression tasks, aiming to identify the vulnerabilities of these models under query-based black-box attacks. We propose a novel square-based adversarial patch optimization framework and employ probabilistic square sampling and score-based gradient estimation techniques to generate the patch effectively and efficiently, overcoming the scalability problem of previous black-box patch attacks. Our attack prototype, named BadPart, is evaluated on both MDE and OFE tasks, utilizing a total of 7 models. BadPart surpasses 3 baseline methods in terms of both attack performance and efficiency. We also apply BadPart on the Google online service for portrait depth estimation, causing 43.5% relative distance error with 50K queries. State-of-the-art (SOTA) countermeasures cannot defend our attack effectively.

摘要: 像素级回归任务(如单目深度估计(MDE)和光流估计(OFE))在自动驾驶、增强现实和视频合成等应用中广泛应用于我们的日常生活中。虽然某些应用是安全关键的或具有社会意义的，但这些模型的对抗健壮性没有得到充分的研究，特别是在黑盒场景中。在这项工作中，我们引入了第一个针对像素回归任务的统一黑盒对抗性补丁攻击框架，旨在识别这些模型在基于查询的黑盒攻击下的脆弱性。提出了一种新的基于平方的对抗性补丁优化框架，并利用概率平方采样和基于分数的梯度估计技术有效地生成了补丁，克服了以往黑盒补丁攻击的可扩展性问题。我们的攻击原型名为BadPart，在MDE和OFE任务上进行了评估，总共使用了7个模型。BadPart在攻击性能和效率方面都超过了3种基线方法。我们还将BadPart应用于Google在线服务上进行人像深度估计，在50K查询中导致了43.5%的相对距离误差。最先进的(SOTA)对策不能有效地防御我们的攻击。



## **26. An Embarrassingly Simple Defense Against Backdoor Attacks On SSL**

一个令人尴尬的简单的后门攻击防御SSL cs.CV

10 pages, 5 figures

**SubmitDate**: 2024-04-01    [abs](http://arxiv.org/abs/2403.15918v2) [paper-pdf](http://arxiv.org/pdf/2403.15918v2)

**Authors**: Aryan Satpathy, Nilaksh Nilaksh, Dhruva Rajwade

**Abstract**: Self Supervised Learning (SSL) has emerged as a powerful paradigm to tackle data landscapes with absence of human supervision. The ability to learn meaningful tasks without the use of labeled data makes SSL a popular method to manage large chunks of data in the absence of labels. However, recent work indicates SSL to be vulnerable to backdoor attacks, wherein models can be controlled, possibly maliciously, to suit an adversary's motives. Li et. al (2022) introduce a novel frequency-based backdoor attack: CTRL. They show that CTRL can be used to efficiently and stealthily gain control over a victim's model trained using SSL. In this work, we devise two defense strategies against frequency-based attacks in SSL: One applicable before model training and the second to be applied during model inference. Our first contribution utilizes the invariance property of the downstream task to defend against backdoor attacks in a generalizable fashion. We observe the ASR (Attack Success Rate) to reduce by over 60% across experiments. Our Inference-time defense relies on evasiveness of the attack and uses the luminance channel to defend against attacks. Using object classification as the downstream task for SSL, we demonstrate successful defense strategies that do not require re-training of the model. Code is available at https://github.com/Aryan-Satpathy/Backdoor.

摘要: 自我监督学习(SSL)已经成为一种强大的范式，可以在缺乏人类监督的情况下处理数据环境。无需使用标签数据即可学习有意义的任务的能力使SSL成为在没有标签的情况下管理大量数据的流行方法。然而，最近的研究表明，SSL容易受到后门攻击，在后门攻击中，可以控制模型，可能是恶意的，以适应对手的动机。Li et.Al(2022)提出了一种新的基于频率的后门攻击：Ctrl。他们表明，CTRL可以用来有效地、秘密地控制使用SSL训练的受害者模型。在这项工作中，我们针对基于频率的攻击设计了两种防御策略：一种适用于模型训练之前，另一种应用于模型推理中。我们的第一个贡献是利用下游任务的不变性以一种可推广的方式防御后门攻击。我们观察到，在整个实验中，ASR(攻击成功率)降低了60%以上。我们的推理时间防御依赖于攻击的规避，并使用亮度通道来防御攻击。使用对象分类作为SSL的下游任务，我们演示了成功的防御策略，不需要对模型进行重新训练。代码可在https://github.com/Aryan-Satpathy/Backdoor.上找到



## **27. Machine Learning Robustness: A Primer**

机器学习鲁棒性：入门 cs.LG

arXiv admin note: text overlap with arXiv:2305.10862 by other authors

**SubmitDate**: 2024-04-01    [abs](http://arxiv.org/abs/2404.00897v1) [paper-pdf](http://arxiv.org/pdf/2404.00897v1)

**Authors**: Houssem Ben Braiek, Foutse Khomh

**Abstract**: This chapter explores the foundational concept of robustness in Machine Learning (ML) and its integral role in establishing trustworthiness in Artificial Intelligence (AI) systems. The discussion begins with a detailed definition of robustness, portraying it as the ability of ML models to maintain stable performance across varied and unexpected environmental conditions. ML robustness is dissected through several lenses: its complementarity with generalizability; its status as a requirement for trustworthy AI; its adversarial vs non-adversarial aspects; its quantitative metrics; and its indicators such as reproducibility and explainability. The chapter delves into the factors that impede robustness, such as data bias, model complexity, and the pitfalls of underspecified ML pipelines. It surveys key techniques for robustness assessment from a broad perspective, including adversarial attacks, encompassing both digital and physical realms. It covers non-adversarial data shifts and nuances of Deep Learning (DL) software testing methodologies. The discussion progresses to explore amelioration strategies for bolstering robustness, starting with data-centric approaches like debiasing and augmentation. Further examination includes a variety of model-centric methods such as transfer learning, adversarial training, and randomized smoothing. Lastly, post-training methods are discussed, including ensemble techniques, pruning, and model repairs, emerging as cost-effective strategies to make models more resilient against the unpredictable. This chapter underscores the ongoing challenges and limitations in estimating and achieving ML robustness by existing approaches. It offers insights and directions for future research on this crucial concept, as a prerequisite for trustworthy AI systems.

摘要: 本章探讨了机器学习(ML)中稳健性的基本概念及其在人工智能(AI)系统中建立可信度的不可或缺的作用。讨论开始于对稳健性的详细定义，将其描述为ML模型在不同和意外的环境条件下保持稳定性能的能力。ML稳健性通过几个方面进行剖析：它与通用性的互补性；它作为值得信赖的人工智能的要求的地位；它的对抗性与非对抗性方面；它的量化指标；以及它的可再现性和可解释性等指标。本章深入探讨了阻碍健壮性的因素，如数据偏差、模型复杂性和未指定的ML管道的陷阱。它从广泛的角度考察了健壮性评估的关键技术，包括涵盖数字和物理领域的对抗性攻击。它涵盖了深度学习(DL)软件测试方法的非对抗性数据转移和细微差别。讨论继续探索增强健壮性的改进策略，从去偏向和增强等以数据为中心的方法开始。进一步的考试包括各种以模型为中心的方法，如转移学习、对抗性训练和随机平滑。最后，讨论了训练后的方法，包括集合技术、修剪和模型修复，这些方法成为使模型对不可预测的情况更具弹性的成本效益策略。本章强调了在通过现有方法估计和实现ML健壮性方面的持续挑战和限制。它为未来对这一关键概念的研究提供了见解和方向，这是值得信赖的人工智能系统的先决条件。



## **28. Privacy Re-identification Attacks on Tabular GANs**

针对Tabular GAN的隐私重识别攻击 cs.CR

**SubmitDate**: 2024-03-31    [abs](http://arxiv.org/abs/2404.00696v1) [paper-pdf](http://arxiv.org/pdf/2404.00696v1)

**Authors**: Abdallah Alshantti, Adil Rasheed, Frank Westad

**Abstract**: Generative models are subject to overfitting and thus may potentially leak sensitive information from the training data. In this work. we investigate the privacy risks that can potentially arise from the use of generative adversarial networks (GANs) for creating tabular synthetic datasets. For the purpose, we analyse the effects of re-identification attacks on synthetic data, i.e., attacks which aim at selecting samples that are predicted to correspond to memorised training samples based on their proximity to the nearest synthetic records. We thus consider multiple settings where different attackers might have different access levels or knowledge of the generative model and predictive, and assess which information is potentially most useful for launching more successful re-identification attacks. In doing so we also consider the situation for which re-identification attacks are formulated as reconstruction attacks, i.e., the situation where an attacker uses evolutionary multi-objective optimisation for perturbing synthetic samples closer to the training space. The results indicate that attackers can indeed pose major privacy risks by selecting synthetic samples that are likely representative of memorised training samples. In addition, we notice that privacy threats considerably increase when the attacker either has knowledge or has black-box access to the generative models. We also find that reconstruction attacks through multi-objective optimisation even increase the risk of identifying confidential samples.

摘要: 生成性模型容易过度拟合，因此可能会泄漏训练数据中的敏感信息。在这项工作中。我们调查了使用生成性对抗网络(GANS)来创建表格合成数据集可能产生的隐私风险。为此，我们分析了重新识别攻击对合成数据的影响，即，旨在根据样本与最近的合成记录的接近程度来选择与记忆的训练样本相对应的样本的攻击。因此，我们考虑了不同攻击者可能具有不同访问级别或生成性模型和预测性知识的多个设置，并评估哪些信息可能对发起更成功的重新识别攻击最有用。在这样做的同时，我们还考虑了将重新识别攻击描述为重构攻击的情况，即攻击者使用进化多目标优化来扰动更接近训练空间的合成样本的情况。结果表明，攻击者通过选择可能代表记忆的训练样本的合成样本，确实可以构成重大的隐私风险。此外，我们注意到，当攻击者知道或拥有对生成模型的黑盒访问权限时，隐私威胁会显著增加。我们还发现，通过多目标优化进行的重建攻击甚至增加了识别机密样本的风险。



## **29. Model-less Is the Best Model: Generating Pure Code Implementations to Replace On-Device DL Models**

无模型是最好的模型：生成纯代码实现来替换设备上的DL模型 cs.SE

Accepted by the ACM SIGSOFT International Symposium on Software  Testing and Analysis (ISSTA2024)

**SubmitDate**: 2024-03-31    [abs](http://arxiv.org/abs/2403.16479v2) [paper-pdf](http://arxiv.org/pdf/2403.16479v2)

**Authors**: Mingyi Zhou, Xiang Gao, Pei Liu, John Grundy, Chunyang Chen, Xiao Chen, Li Li

**Abstract**: Recent studies show that deployed deep learning (DL) models such as those of Tensor Flow Lite (TFLite) can be easily extracted from real-world applications and devices by attackers to generate many kinds of attacks like adversarial attacks. Although securing deployed on-device DL models has gained increasing attention, no existing methods can fully prevent the aforementioned threats. Traditional software protection techniques have been widely explored, if on-device models can be implemented using pure code, such as C++, it will open the possibility of reusing existing software protection techniques. However, due to the complexity of DL models, there is no automatic method that can translate the DL models to pure code. To fill this gap, we propose a novel method, CustomDLCoder, to automatically extract the on-device model information and synthesize a customized executable program for a wide range of DL models. CustomDLCoder first parses the DL model, extracts its backend computing units, configures the computing units to a graph, and then generates customized code to implement and deploy the ML solution without explicit model representation. The synthesized program hides model information for DL deployment environments since it does not need to retain explicit model representation, preventing many attacks on the DL model. In addition, it improves ML performance because the customized code removes model parsing and preprocessing steps and only retains the data computing process. Our experimental results show that CustomDLCoder improves model security by disabling on-device model sniffing. Compared with the original on-device platform (i.e., TFLite), our method can accelerate model inference by 21.8% and 24.3% on x86-64 and ARM64 platforms, respectively. Most importantly, it can significantly reduce memory consumption by 68.8% and 36.0% on x86-64 and ARM64 platforms, respectively.

摘要: 最近的研究表明，部署的深度学习(DL)模型，如张量流精简(TFLite)模型，可以很容易地被攻击者从现实世界的应用和设备中提取出来，从而产生多种攻击，如对抗性攻击。尽管保护部署在设备上的DL模型越来越受到关注，但没有一种现有方法可以完全防止上述威胁。传统的软件保护技术已经得到了广泛的探索，如果设备上的模型可以用纯代码实现，如C++，这将打开重用现有软件保护技术的可能性。然而，由于DL模型的复杂性，目前还没有一种自动的方法可以将DL模型转换为纯代码。为了填补这一空白，我们提出了一种新的方法CustomDLCoder，它可以自动提取设备上的模型信息，并为广泛的DL模型合成定制的可执行程序。CustomDLCoder首先解析DL模型，提取其后端计算单元，将计算单元配置为图形，然后生成定制代码来实现和部署ML解决方案，而不需要显式的模型表示。合成的程序隐藏了DL部署环境的模型信息，因为它不需要保留显式的模型表示，从而防止了对DL模型的许多攻击。此外，它还提高了ML的性能，因为定制的代码删除了模型解析和预处理步骤，只保留了数据计算过程。我们的实验结果表明，CustomDLCoder通过禁止设备上的模型嗅探提高了模型的安全性。在x86-64和ARM64平台上，与原有的设备上平台(即TFLite)相比，该方法的模型推理速度分别提高了21.8%和24.3%。最重要的是，它可以在x86-64和ARM64平台上分别显著降低68.8%和36.0%的内存消耗。



## **30. Embodied Active Defense: Leveraging Recurrent Feedback to Counter Adversarial Patches**

主动防御：利用循环反馈对抗对抗补丁 cs.CV

27pages

**SubmitDate**: 2024-03-31    [abs](http://arxiv.org/abs/2404.00540v1) [paper-pdf](http://arxiv.org/pdf/2404.00540v1)

**Authors**: Lingxuan Wu, Xiao Yang, Yinpeng Dong, Liuwei Xie, Hang Su, Jun Zhu

**Abstract**: The vulnerability of deep neural networks to adversarial patches has motivated numerous defense strategies for boosting model robustness. However, the prevailing defenses depend on single observation or pre-established adversary information to counter adversarial patches, often failing to be confronted with unseen or adaptive adversarial attacks and easily exhibiting unsatisfying performance in dynamic 3D environments. Inspired by active human perception and recurrent feedback mechanisms, we develop Embodied Active Defense (EAD), a proactive defensive strategy that actively contextualizes environmental information to address misaligned adversarial patches in 3D real-world settings. To achieve this, EAD develops two central recurrent sub-modules, i.e., a perception module and a policy module, to implement two critical functions of active vision. These models recurrently process a series of beliefs and observations, facilitating progressive refinement of their comprehension of the target object and enabling the development of strategic actions to counter adversarial patches in 3D environments. To optimize learning efficiency, we incorporate a differentiable approximation of environmental dynamics and deploy patches that are agnostic to the adversary strategies. Extensive experiments demonstrate that EAD substantially enhances robustness against a variety of patches within just a few steps through its action policy in safety-critical tasks (e.g., face recognition and object detection), without compromising standard accuracy. Furthermore, due to the attack-agnostic characteristic, EAD facilitates excellent generalization to unseen attacks, diminishing the averaged attack success rate by 95 percent across a range of unseen adversarial attacks.

摘要: 深度神经网络对敌意补丁的脆弱性激发了许多增强模型稳健性的防御策略。然而，主流的防御依赖于单一的观察或预先建立的对手信息来对抗对抗性补丁，往往无法对抗看不见的或自适应的对抗性攻击，并且在动态3D环境中很容易表现出不令人满意的性能。受人类主动感知和循环反馈机制的启发，我们开发了体现主动防御(EAD)，这是一种主动防御策略，它主动地将环境信息与上下文关联起来，以应对3D现实世界中未对齐的敌方补丁。为了实现这一目标，EAD开发了两个中央循环子模块，即感知模块和政策模块，以实现主动视觉的两个关键功能。这些模型反复处理一系列信念和观察，有助于逐步完善其对目标对象的理解，并使开发战略行动来对抗3D环境中的敌意补丁。为了优化学习效率，我们结合了环境动态的可微近似，并部署了与对手策略无关的补丁。大量实验表明，EAD通过其在安全关键任务(如人脸识别和目标检测)中的操作策略，在不影响标准准确性的情况下，仅需几个步骤即可显著增强针对各种补丁的稳健性。此外，由于攻击不可知的特性，EAD有助于对看不见的攻击进行出色的泛化，使一系列看不见的对抗性攻击的平均攻击成功率降低95%。



## **31. AttackNet: Enhancing Biometric Security via Tailored Convolutional Neural Network Architectures for Liveness Detection**

AttackNet：通过定制的卷积神经网络架构增强生物识别安全性，用于活性检测 cs.CV

**SubmitDate**: 2024-03-30    [abs](http://arxiv.org/abs/2402.03769v2) [paper-pdf](http://arxiv.org/pdf/2402.03769v2)

**Authors**: Oleksandr Kuznetsov, Dmytro Zakharov, Emanuele Frontoni, Andrea Maranesi

**Abstract**: Biometric security is the cornerstone of modern identity verification and authentication systems, where the integrity and reliability of biometric samples is of paramount importance. This paper introduces AttackNet, a bespoke Convolutional Neural Network architecture, meticulously designed to combat spoofing threats in biometric systems. Rooted in deep learning methodologies, this model offers a layered defense mechanism, seamlessly transitioning from low-level feature extraction to high-level pattern discernment. Three distinctive architectural phases form the crux of the model, each underpinned by judiciously chosen activation functions, normalization techniques, and dropout layers to ensure robustness and resilience against adversarial attacks. Benchmarking our model across diverse datasets affirms its prowess, showcasing superior performance metrics in comparison to contemporary models. Furthermore, a detailed comparative analysis accentuates the model's efficacy, drawing parallels with prevailing state-of-the-art methodologies. Through iterative refinement and an informed architectural strategy, AttackNet underscores the potential of deep learning in safeguarding the future of biometric security.

摘要: 生物特征安全是现代身份验证和认证系统的基石，其中生物特征样本的完整性和可靠性至关重要。本文介绍了AttackNet，一种定制的卷积神经网络结构，精心设计用于对抗生物识别系统中的欺骗威胁。该模型植根于深度学习方法，提供了一种分层防御机制，从低级特征提取无缝过渡到高级模式识别。三个不同的体系结构阶段构成了该模型的核心，每个阶段都以明智地选择的激活函数、归一化技术和丢弃层为基础，以确保对对手攻击的健壮性和弹性。在不同的数据集上对我们的模型进行基准测试，肯定了它的威力，展示了与当代模型相比的卓越性能指标。此外，一项详细的比较分析强调了该模型的有效性，将其与流行的最先进的方法进行了比较。通过迭代改进和明智的架构策略，AttackNet强调了深度学习在保障生物识别安全的未来方面的潜力。



## **32. Bidirectional Consistency Models**

双向一致性模型 cs.LG

40 pages, 25 figures

**SubmitDate**: 2024-03-30    [abs](http://arxiv.org/abs/2403.18035v2) [paper-pdf](http://arxiv.org/pdf/2403.18035v2)

**Authors**: Liangchen Li, Jiajun He

**Abstract**: Diffusion models (DMs) are capable of generating remarkably high-quality samples by iteratively denoising a random vector, a process that corresponds to moving along the probability flow ordinary differential equation (PF ODE). Interestingly, DMs can also invert an input image to noise by moving backward along the PF ODE, a key operation for downstream tasks such as interpolation and image editing. However, the iterative nature of this process restricts its speed, hindering its broader application. Recently, Consistency Models (CMs) have emerged to address this challenge by approximating the integral of the PF ODE, largely reducing the number of iterations. Yet, the absence of an explicit ODE solver complicates the inversion process. To resolve this, we introduce the Bidirectional Consistency Model (BCM), which learns a single neural network that enables both forward and backward traversal along the PF ODE, efficiently unifying generation and inversion tasks within one framework. Notably, our proposed method enables one-step generation and inversion while also allowing the use of additional steps to enhance generation quality or reduce reconstruction error. Furthermore, by leveraging our model's bidirectional consistency, we introduce a sampling strategy that can enhance FID while preserving the generated image content. We further showcase our model's capabilities in several downstream tasks, such as interpolation and inpainting, and present demonstrations of potential applications, including blind restoration of compressed images and defending black-box adversarial attacks.

摘要: 扩散模型(DM)能够通过迭代地对随机向量去噪来生成非常高质量的样本，该过程对应于沿着概率流常微分方程式(PF ODE)移动。有趣的是，DM还可以通过沿PF ODE向后移动来将输入图像反转为噪声，这是下游任务(如插补和图像编辑)的关键操作。然而，这一过程的迭代性质限制了其速度，阻碍了其更广泛的应用。最近，一致性模型(CM)已经出现，通过近似PF ODE的积分来解决这一挑战，大大减少了迭代次数。然而，由于没有显式的常微分方程组解算器，使得反演过程变得更加复杂。为了解决这个问题，我们引入了双向一致性模型(BCM)，它学习一个单一的神经网络，允许沿着PF ODE进行前向和后向遍历，有效地将生成和反转任务统一在一个框架内。值得注意的是，我们提出的方法支持一步生成和反转，同时还允许使用额外的步骤来提高生成质量或减少重建误差。此外，通过利用模型的双向一致性，我们引入了一种采样策略，该策略可以在保留生成的图像内容的同时增强FID。我们进一步展示了我们的模型在几个下游任务中的能力，如插补和修复，并展示了潜在的应用程序，包括压缩图像的盲恢复和防御黑盒攻击。



## **33. STBA: Towards Evaluating the Robustness of DNNs for Query-Limited Black-box Scenario**

STBA：查询受限黑盒情形下DNN鲁棒性评估 cs.CV

**SubmitDate**: 2024-03-30    [abs](http://arxiv.org/abs/2404.00362v1) [paper-pdf](http://arxiv.org/pdf/2404.00362v1)

**Authors**: Renyang Liu, Kwok-Yan Lam, Wei Zhou, Sixing Wu, Jun Zhao, Dongting Hu, Mingming Gong

**Abstract**: Many attack techniques have been proposed to explore the vulnerability of DNNs and further help to improve their robustness. Despite the significant progress made recently, existing black-box attack methods still suffer from unsatisfactory performance due to the vast number of queries needed to optimize desired perturbations. Besides, the other critical challenge is that adversarial examples built in a noise-adding manner are abnormal and struggle to successfully attack robust models, whose robustness is enhanced by adversarial training against small perturbations. There is no doubt that these two issues mentioned above will significantly increase the risk of exposure and result in a failure to dig deeply into the vulnerability of DNNs. Hence, it is necessary to evaluate DNNs' fragility sufficiently under query-limited settings in a non-additional way. In this paper, we propose the Spatial Transform Black-box Attack (STBA), a novel framework to craft formidable adversarial examples in the query-limited scenario. Specifically, STBA introduces a flow field to the high-frequency part of clean images to generate adversarial examples and adopts the following two processes to enhance their naturalness and significantly improve the query efficiency: a) we apply an estimated flow field to the high-frequency part of clean images to generate adversarial examples instead of introducing external noise to the benign image, and b) we leverage an efficient gradient estimation method based on a batch of samples to optimize such an ideal flow field under query-limited settings. Compared to existing score-based black-box baselines, extensive experiments indicated that STBA could effectively improve the imperceptibility of the adversarial examples and remarkably boost the attack success rate under query-limited settings.

摘要: 已经提出了许多攻击技术来探索DNN的脆弱性，并进一步帮助提高它们的健壮性。尽管最近取得了显著的进展，但现有的黑盒攻击方法仍然存在性能不佳的问题，这是因为需要大量的查询来优化期望的扰动。此外，另一个关键的挑战是，以添加噪声的方式构建的对抗性样本是不正常的，并且难以成功地攻击健壮模型，而健壮模型的健壮性通过对抗小扰动的对抗性训练来增强。毫无疑问，上述两个问题将大大增加暴露的风险，并导致无法深入挖掘DNN的脆弱性。因此，有必要以一种非额外的方式充分评估DNN在查询受限设置下的脆弱性。在本文中，我们提出了空间变换黑盒攻击(STBA)，这是一个新的框架，可以在查询受限的情况下创建强大的对手示例。具体地说，STBA在清洁图像的高频部分引入了流场来生成对抗性实例，并采用了以下两个过程来增强其自然性，显著提高了查询效率：a)将估计的流场应用于干净图像的高频部分来生成对抗性实例，而不是在良性图像中引入外部噪声；b)在查询受限的情况下，利用一种基于批量样本的高效梯度估计方法来优化这样的理想流场。大量实验表明，与已有的基于分数的黑盒基线相比，STBA能够有效地提高对抗性实例的隐蔽性，显著提高查询受限环境下的攻击成功率。



## **34. LLM-Resistant Math Word Problem Generation via Adversarial Attacks**

通过对抗攻击生成LLM抵抗数学单词问题 cs.CL

Code/data: https://github.com/ruoyuxie/adversarial_mwps_generation

**SubmitDate**: 2024-03-30    [abs](http://arxiv.org/abs/2402.17916v2) [paper-pdf](http://arxiv.org/pdf/2402.17916v2)

**Authors**: Roy Xie, Chengxuan Huang, Junlin Wang, Bhuwan Dhingra

**Abstract**: Large language models (LLMs) have significantly transformed the educational landscape. As current plagiarism detection tools struggle to keep pace with LLMs' rapid advancements, the educational community faces the challenge of assessing students' true problem-solving abilities in the presence of LLMs. In this work, we explore a new paradigm for ensuring fair evaluation -- generating adversarial examples which preserve the structure and difficulty of the original questions aimed for assessment, but are unsolvable by LLMs. Focusing on the domain of math word problems, we leverage abstract syntax trees to structurally generate adversarial examples that cause LLMs to produce incorrect answers by simply editing the numeric values in the problems. We conduct experiments on various open- and closed-source LLMs, quantitatively and qualitatively demonstrating that our method significantly degrades their math problem-solving ability. We identify shared vulnerabilities among LLMs and propose a cost-effective approach to attack high-cost models. Additionally, we conduct automatic analysis on math problems and investigate the cause of failure, offering a nuanced view into model's limitation.

摘要: 大型语言模型(LLM)极大地改变了教育格局。由于目前的抄袭检测工具难以跟上LLMS的快速进步，教育界面临着在LLMS存在的情况下评估学生真正的问题解决能力的挑战。在这项工作中，我们探索了一种确保公平评价的新范式--生成对抗性实例，它保留了用于评价的原始问题的结构和难度，但无法用LLMS解决。聚焦于数学应用题领域，我们利用抽象语法树来结构化地生成对抗性实例，这些实例通过简单地编辑问题中的数值来导致LLMS产生不正确的答案。我们在各种开源和闭源的LLM上进行了实验，定量和定性地证明了我们的方法显著降低了他们的数学问题解决能力。我们识别了LLM之间的共同漏洞，并提出了一种具有成本效益的方法来攻击高成本模型。此外，我们对数学问题进行了自动分析，并调查了失败的原因，为模型的局限性提供了一个细微的视角。



## **35. On Inherent Adversarial Robustness of Active Vision Systems**

主动视觉系统的固有对抗鲁棒性 cs.CV

**SubmitDate**: 2024-03-29    [abs](http://arxiv.org/abs/2404.00185v1) [paper-pdf](http://arxiv.org/pdf/2404.00185v1)

**Authors**: Amitangshu Mukherjee, Timur Ibrayev, Kaushik Roy

**Abstract**: Current Deep Neural Networks are vulnerable to adversarial examples, which alter their predictions by adding carefully crafted noise. Since human eyes are robust to such inputs, it is possible that the vulnerability stems from the standard way of processing inputs in one shot by processing every pixel with the same importance. In contrast, neuroscience suggests that the human vision system can differentiate salient features by (1) switching between multiple fixation points (saccades) and (2) processing the surrounding with a non-uniform external resolution (foveation). In this work, we advocate that the integration of such active vision mechanisms into current deep learning systems can offer robustness benefits. Specifically, we empirically demonstrate the inherent robustness of two active vision methods - GFNet and FALcon - under a black box threat model. By learning and inferencing based on downsampled glimpses obtained from multiple distinct fixation points within an input, we show that these active methods achieve (2-3) times greater robustness compared to a standard passive convolutional network under state-of-the-art adversarial attacks. More importantly, we provide illustrative and interpretable visualization analysis that demonstrates how performing inference from distinct fixation points makes active vision methods less vulnerable to malicious inputs.

摘要: 当前的深度神经网络很容易受到敌意例子的影响，这些例子通过添加精心设计的噪声来改变它们的预测。由于人眼对这样的输入很健壮，这种漏洞可能源于一次处理输入的标准方式，即处理具有相同重要性的每个像素。相比之下，神经科学表明，人类的视觉系统可以通过(1)在多个注视点(眼跳)之间切换和(2)用非均匀的外部分辨率(中心凹)处理周围环境来区分显著特征。在这项工作中，我们主张将这种主动视觉机制集成到当前的深度学习系统中，可以提供健壮性优势。具体而言，我们通过实验验证了两种主动视觉方法--GFNet和Falcon--在黑匣子威胁模型下的内在稳健性。通过基于从输入内多个不同固定点获得的下采样一瞥的学习和推理，我们表明这些主动方法在最先进的对抗攻击下比标准的被动卷积网络获得(2-3)倍的健壮性。更重要的是，我们提供了说明性和可解释性的可视化分析，演示了如何从不同的注视点执行推理使主动视觉方法不太容易受到恶意输入的影响。



## **36. Deepfake Sentry: Harnessing Ensemble Intelligence for Resilient Detection and Generalisation**

Deepfake Sentry：利用Envision智能进行弹性检测和概括 cs.CV

16 pages, 1 figure, U.P.B. Sci. Bull., Series C, Vol. 85, Iss. 4,  2023

**SubmitDate**: 2024-03-29    [abs](http://arxiv.org/abs/2404.00114v1) [paper-pdf](http://arxiv.org/pdf/2404.00114v1)

**Authors**: Liviu-Daniel Ştefan, Dan-Cristian Stanciu, Mihai Dogariu, Mihai Gabriel Constantin, Andrei Cosmin Jitaru, Bogdan Ionescu

**Abstract**: Recent advancements in Generative Adversarial Networks (GANs) have enabled photorealistic image generation with high quality. However, the malicious use of such generated media has raised concerns regarding visual misinformation. Although deepfake detection research has demonstrated high accuracy, it is vulnerable to advances in generation techniques and adversarial iterations on detection countermeasures. To address this, we propose a proactive and sustainable deepfake training augmentation solution that introduces artificial fingerprints into models. We achieve this by employing an ensemble learning approach that incorporates a pool of autoencoders that mimic the effect of the artefacts introduced by the deepfake generator models. Experiments on three datasets reveal that our proposed ensemble autoencoder-based data augmentation learning approach offers improvements in terms of generalisation, resistance against basic data perturbations such as noise, blurring, sharpness enhancement, and affine transforms, resilience to commonly used lossy compression algorithms such as JPEG, and enhanced resistance against adversarial attacks.

摘要: 生成性对抗网络(GANS)的最新进展使得生成高质量的照片级真实感图像成为可能。然而，恶意使用这种生成的媒体引起了人们对视觉错误信息的担忧。虽然深度伪检测研究已经证明了很高的准确率，但它很容易受到生成技术的进步和检测对策上的敌意迭代的影响。为了解决这一问题，我们提出了一种主动的、可持续的深度假训练增强方案，将人工指纹引入模型中。我们通过采用集成学习方法来实现这一点，该方法结合了模仿深度伪生成器模型引入的伪像的效果的自动编码器池。在三个数据集上的实验表明，我们提出的基于集成自动编码器的数据增强学习方法在泛化、对基本数据扰动(如噪声、模糊、锐度增强和仿射变换)的抵抗力、对常用有损压缩算法(如JPEG)的恢复能力以及对敌意攻击的抵抗力方面都有改进。



## **37. LipSim: A Provably Robust Perceptual Similarity Metric**

LipSim：一个可证鲁棒的感知相似度量 cs.CV

**SubmitDate**: 2024-03-29    [abs](http://arxiv.org/abs/2310.18274v2) [paper-pdf](http://arxiv.org/pdf/2310.18274v2)

**Authors**: Sara Ghazanfari, Alexandre Araujo, Prashanth Krishnamurthy, Farshad Khorrami, Siddharth Garg

**Abstract**: Recent years have seen growing interest in developing and applying perceptual similarity metrics. Research has shown the superiority of perceptual metrics over pixel-wise metrics in aligning with human perception and serving as a proxy for the human visual system. On the other hand, as perceptual metrics rely on neural networks, there is a growing concern regarding their resilience, given the established vulnerability of neural networks to adversarial attacks. It is indeed logical to infer that perceptual metrics may inherit both the strengths and shortcomings of neural networks. In this work, we demonstrate the vulnerability of state-of-the-art perceptual similarity metrics based on an ensemble of ViT-based feature extractors to adversarial attacks. We then propose a framework to train a robust perceptual similarity metric called LipSim (Lipschitz Similarity Metric) with provable guarantees. By leveraging 1-Lipschitz neural networks as the backbone, LipSim provides guarded areas around each data point and certificates for all perturbations within an $\ell_2$ ball. Finally, a comprehensive set of experiments shows the performance of LipSim in terms of natural and certified scores and on the image retrieval application. The code is available at https://github.com/SaraGhazanfari/LipSim.

摘要: 近年来，人们对开发和应用感知相似性度量的兴趣与日俱增。研究表明，与像素度量相比，感知度量在与人类感知和作为人类视觉系统的代理方面具有优势。另一方面，由于感知指标依赖于神经网络，鉴于神经网络对对手攻击的公认脆弱性，人们越来越担心其弹性。推断感知指标可能继承了神经网络的长处和短处，这确实是合乎逻辑的。在这项工作中，我们展示了基于基于VIT的特征提取集合的最新感知相似性度量在对抗攻击中的脆弱性。然后，我们提出了一个框架来训练一个健壮的感知相似性度量，称为LipSim(Lipschitz相似性度量)，并具有可证明的保证。通过利用1-Lipschitz神经网络作为主干，LipSim在每个数据点周围提供保护区域，并为$\ell_2$球内的所有扰动提供证书。最后，一组全面的实验显示了LipSim在自然分数和认证分数以及图像检索应用方面的性能。代码可在https://github.com/SaraGhazanfari/LipSim.上获得



## **38. Selective Attention-based Modulation for Continual Learning**

基于选择性注意的持续学习调制 cs.CV

**SubmitDate**: 2024-03-29    [abs](http://arxiv.org/abs/2403.20086v1) [paper-pdf](http://arxiv.org/pdf/2403.20086v1)

**Authors**: Giovanni Bellitto, Federica Proietto Salanitri, Matteo Pennisi, Matteo Boschini, Angelo Porrello, Simone Calderara, Simone Palazzo, Concetto Spampinato

**Abstract**: We present SAM, a biologically-plausible selective attention-driven modulation approach to enhance classification models in a continual learning setting. Inspired by neurophysiological evidence that the primary visual cortex does not contribute to object manifold untangling for categorization and that primordial attention biases are still embedded in the modern brain, we propose to employ auxiliary saliency prediction features as a modulation signal to drive and stabilize the learning of a sequence of non-i.i.d. classification tasks. Experimental results confirm that SAM effectively enhances the performance (in some cases up to about twenty percent points) of state-of-the-art continual learning methods, both in class-incremental and task-incremental settings. Moreover, we show that attention-based modulation successfully encourages the learning of features that are more robust to the presence of spurious features and to adversarial attacks than baseline methods. Code is available at: https://github.com/perceivelab/SAM.

摘要: 我们提出了SAM，一种生物学上看似合理的选择性注意驱动的调制方法，以增强持续学习环境中的分类模型。受到神经生理学证据的启发，即初级视觉皮质不有助于物体歧管的分类，以及原始注意偏差仍然嵌入现代大脑，我们建议使用辅助显著预测特征作为调制信号来驱动和稳定对非I.I.D.序列的学习。分类任务。实验结果证实，SAM有效地提高了最先进的持续学习方法的性能(在某些情况下高达约20%)，无论是在班级递增还是任务递增的设置下。此外，我们表明，基于注意力的调制成功地鼓励了对虚假特征的存在和对抗攻击的特征的学习，这些特征比基线方法更健壮。代码可从以下网址获得：https://github.com/perceivelab/SAM.



## **39. Strong Transferable Adversarial Attacks via Ensembled Asymptotically Normal Distribution Learning**

基于集成渐近正态分布学习的强可传递对抗攻击 cs.LG

**SubmitDate**: 2024-03-29    [abs](http://arxiv.org/abs/2209.11964v2) [paper-pdf](http://arxiv.org/pdf/2209.11964v2)

**Authors**: Zhengwei Fang, Rui Wang, Tao Huang, Liping Jing

**Abstract**: Strong adversarial examples are crucial for evaluating and enhancing the robustness of deep neural networks. However, the performance of popular attacks is usually sensitive, for instance, to minor image transformations, stemming from limited information -- typically only one input example, a handful of white-box source models, and undefined defense strategies. Hence, the crafted adversarial examples are prone to overfit the source model, which hampers their transferability to unknown architectures. In this paper, we propose an approach named Multiple Asymptotically Normal Distribution Attacks (MultiANDA) which explicitly characterize adversarial perturbations from a learned distribution. Specifically, we approximate the posterior distribution over the perturbations by taking advantage of the asymptotic normality property of stochastic gradient ascent (SGA), then employ the deep ensemble strategy as an effective proxy for Bayesian marginalization in this process, aiming to estimate a mixture of Gaussians that facilitates a more thorough exploration of the potential optimization space. The approximated posterior essentially describes the stationary distribution of SGA iterations, which captures the geometric information around the local optimum. Thus, MultiANDA allows drawing an unlimited number of adversarial perturbations for each input and reliably maintains the transferability. Our proposed method outperforms ten state-of-the-art black-box attacks on deep learning models with or without defenses through extensive experiments on seven normally trained and seven defense models.

摘要: 强对抗性的例子对于评估和提高深度神经网络的稳健性至关重要。然而，流行攻击的性能通常对较小的图像变换很敏感，这源于有限的信息--通常只有一个输入示例、几个白盒源模型和未定义的防御策略。因此，精心制作的敌意示例容易过度匹配源模型，这阻碍了它们向未知体系结构的可转移性。在本文中，我们提出了一种名为多重渐近正态分布攻击(Multiple渐近正态分布攻击)的方法，它显式地刻画了来自学习分布的敌对扰动。具体地说，我们利用随机梯度上升(SGA)的渐近正态性质来逼近扰动下的后验分布，然后使用深度集成策略作为贝叶斯边际化的有效代理，目的是估计一个混合的高斯分布，以便更深入地探索潜在的优化空间。近似后验概率本质上描述了SGA迭代的平稳分布，它捕捉了局部最优解附近的几何信息。因此，MultiANDA允许为每个输入绘制无限数量的对抗性扰动，并可靠地保持可转移性。通过在7个正常训练模型和7个防御模型上的大量实验，我们提出的方法在有防御和无防御的深度学习模型上的性能超过了10种最新的黑盒攻击。



## **40. An Anomaly Behavior Analysis Framework for Securing Autonomous Vehicle Perception**

一种保障自主车辆感知的异常行为分析框架 cs.RO

20th ACS/IEEE International Conference on Computer Systems and  Applications (Accepted for publication)

**SubmitDate**: 2024-03-29    [abs](http://arxiv.org/abs/2310.05041v2) [paper-pdf](http://arxiv.org/pdf/2310.05041v2)

**Authors**: Murad Mehrab Abrar, Salim Hariri

**Abstract**: As a rapidly growing cyber-physical platform, Autonomous Vehicles (AVs) are encountering more security challenges as their capabilities continue to expand. In recent years, adversaries are actively targeting the perception sensors of autonomous vehicles with sophisticated attacks that are not easily detected by the vehicles' control systems. This work proposes an Anomaly Behavior Analysis approach to detect a perception sensor attack against an autonomous vehicle. The framework relies on temporal features extracted from a physics-based autonomous vehicle behavior model to capture the normal behavior of vehicular perception in autonomous driving. By employing a combination of model-based techniques and machine learning algorithms, the proposed framework distinguishes between normal and abnormal vehicular perception behavior. To demonstrate the application of the framework in practice, we performed a depth camera attack experiment on an autonomous vehicle testbed and generated an extensive dataset. We validated the effectiveness of the proposed framework using this real-world data and released the dataset for public access. To our knowledge, this dataset is the first of its kind and will serve as a valuable resource for the research community in evaluating their intrusion detection techniques effectively.

摘要: 作为一个快速发展的网络物理平台，自动驾驶汽车(AVs)随着其能力的不断扩大，面临着更多的安全挑战。近年来，对手积极瞄准自动驾驶车辆的感知传感器，进行复杂的攻击，而这些攻击不容易被车辆的控制系统检测到。本文提出了一种异常行为分析方法来检测感知传感器对自动驾驶车辆的攻击。该框架依赖于从基于物理的自动驾驶车辆行为模型中提取的时间特征来捕捉自动驾驶中车辆感知的正常行为。通过结合基于模型的技术和机器学习算法，该框架区分了正常和异常的车辆感知行为。为了验证该框架在实践中的应用，我们在自主车辆试验台上进行了深度相机攻击实验，并生成了大量的数据集。我们使用这些真实世界的数据验证了提出的框架的有效性，并发布了数据集以供公众访问。据我们所知，该数据集是此类数据的第一个，将为研究界提供宝贵的资源，以有效地评估他们的入侵检测技术。



## **41. Evolving Assembly Code in an Adversarial Environment**

对抗环境下的汇编代码演变 cs.NE

9 pages, 5 figures, 6 listings

**SubmitDate**: 2024-03-28    [abs](http://arxiv.org/abs/2403.19489v1) [paper-pdf](http://arxiv.org/pdf/2403.19489v1)

**Authors**: Irina Maliukov, Gera Weiss, Oded Margalit, Achiya Elyasaf

**Abstract**: In this work, we evolve assembly code for the CodeGuru competition. The competition's goal is to create a survivor -- an assembly program that runs the longest in shared memory, by resisting attacks from adversary survivors and finding their weaknesses. For evolving top-notch solvers, we specify a Backus Normal Form (BNF) for the assembly language and synthesize the code from scratch using Genetic Programming (GP). We evaluate the survivors by running CodeGuru games against human-written winning survivors. Our evolved programs found weaknesses in the programs they were trained against and utilized them. In addition, we compare our approach with a Large-Language Model, demonstrating that the latter cannot generate a survivor that can win at any competition. This work has important applications for cyber-security, as we utilize evolution to detect weaknesses in survivors. The assembly BNF is domain-independent; thus, by modifying the fitness function, it can detect code weaknesses and help fix them. Finally, the CodeGuru competition offers a novel platform for analyzing GP and code evolution in adversarial environments. To support further research in this direction, we provide a thorough qualitative analysis of the evolved survivors and the weaknesses found.

摘要: 在这项工作中，我们为CodeGuru竞赛演变汇编代码。这项竞赛的目标是创建一个幸存者--一个在共享内存中运行时间最长的汇编程序，通过抵抗对手幸存者的攻击并找到他们的弱点。对于进化的顶级解算器，我们为汇编语言指定了Backus范式(BNF)，并使用遗传编程(GP)从头开始合成代码。我们通过运行CodeGuru游戏来评估幸存者，以对抗人类编写的获胜幸存者。我们的演进计划发现了他们所针对的计划中的弱点，并利用了这些弱点。此外，我们将我们的方法与大语言模型进行比较，表明后者无法产生能够在任何竞争中获胜的幸存者。这项工作在网络安全方面有重要的应用，因为我们利用进化论来检测幸存者的弱点。程序集BNF是独立于域的；因此，通过修改适应度函数，它可以检测代码弱点并帮助修复它们。最后，CodeGuru竞赛为分析对抗性环境中的GP和代码演化提供了一个新的平台。为了支持这方面的进一步研究，我们对进化的幸存者和发现的弱点进行了彻底的定性分析。



## **42. Cloudy with a Chance of Cyberattacks: Dangling Resources Abuse on Cloud Platforms**

云与网络攻击的机会：云平台上的资源滥用危险 cs.NI

17 pages, 29 figures, to be published in NSDI'24: Proceedings of the  21st USENIX Symposium on Networked Systems Design and Implementation

**SubmitDate**: 2024-03-28    [abs](http://arxiv.org/abs/2403.19368v1) [paper-pdf](http://arxiv.org/pdf/2403.19368v1)

**Authors**: Jens Frieß, Tobias Gattermayer, Nethanel Gelernter, Haya Schulmann, Michael Waidner

**Abstract**: Recent works showed that it is feasible to hijack resources on cloud platforms. In such hijacks, attackers can take over released resources that belong to legitimate organizations. It was proposed that adversaries could abuse these resources to carry out attacks against customers of the hijacked services, e.g., through malware distribution. However, to date, no research has confirmed the existence of these attacks. We identify, for the first time, real-life hijacks of cloud resources. This yields a number of surprising and important insights. First, contrary to previous assumption that attackers primarily target IP addresses, our findings reveal that the type of resource is not the main consideration in a hijack. Attackers focus on hijacking records that allow them to determine the resource by entering freetext. The costs and overhead of hijacking such records are much lower than those of hijacking IP addresses, which are randomly selected from a large pool. Second, identifying hijacks poses a substantial challenge. Monitoring resource changes, e.g., changes in content, is insufficient, since such changes could also be legitimate. Retrospective analysis of digital assets to identify hijacks is also arduous due to the immense volume of data involved and the absence of indicators to search for. To address this challenge, we develop a novel approach that involves analyzing data from diverse sources to effectively differentiate between malicious and legitimate modifications. Our analysis has revealed 20,904 instances of hijacked resources on popular cloud platforms. While some hijacks are short-lived (up to 15 days), 1/3 persist for more than 65 days. We study how attackers abuse the hijacked resources and find that, in contrast to the threats considered in previous work, the majority of the abuse (75%) is blackhat search engine optimization.

摘要: 最近的研究表明，劫持云平台上的资源是可行的。在这种劫持中，攻击者可以接管属于合法组织的已释放资源。有人提出，攻击者可以滥用这些资源，例如通过恶意软件分发，对被劫持服务的客户进行攻击。然而，到目前为止，还没有研究证实这些攻击的存在。我们首次发现了现实生活中的云资源劫持行为。这产生了许多令人惊讶和重要的见解。首先，与之前认为攻击者主要针对IP地址的假设相反，我们的研究结果表明，资源类型不是劫持的主要考虑因素。攻击者专注于劫持记录，这些记录允许他们通过输入freetext来确定资源。劫持此类记录的成本和开销比劫持IP地址的成本和开销要低得多，后者是从一个大的池中随机选择的。其次，识别劫机事件构成了一个巨大的挑战。监视资源改变，例如内容的改变是不够的，因为这样的改变也可能是合法的。对数字资产进行追溯性分析以确定劫持行为也很困难，因为涉及的数据量巨大，而且缺乏可供搜索的指标。为了应对这一挑战，我们开发了一种新的方法，涉及分析来自不同来源的数据，以有效区分恶意修改和合法修改。我们的分析揭示了20904起流行云平台上的资源被劫持事件。虽然一些劫持是短暂的(长达15天)，但三分之一的劫机持续时间超过65天。我们研究了攻击者如何滥用被劫持的资源，发现与以前工作中考虑的威胁相反，大多数滥用(75%)是黑帽搜索引擎优化。



## **43. Data-free Defense of Black Box Models Against Adversarial Attacks**

黑盒模型对抗攻击的无数据防御 cs.LG

CVPR Workshop (Under Review)

**SubmitDate**: 2024-03-28    [abs](http://arxiv.org/abs/2211.01579v3) [paper-pdf](http://arxiv.org/pdf/2211.01579v3)

**Authors**: Gaurav Kumar Nayak, Inder Khatri, Ruchit Rawal, Anirban Chakraborty

**Abstract**: Several companies often safeguard their trained deep models (i.e., details of architecture, learnt weights, training details etc.) from third-party users by exposing them only as black boxes through APIs. Moreover, they may not even provide access to the training data due to proprietary reasons or sensitivity concerns. In this work, we propose a novel defense mechanism for black box models against adversarial attacks in a data-free set up. We construct synthetic data via generative model and train surrogate network using model stealing techniques. To minimize adversarial contamination on perturbed samples, we propose 'wavelet noise remover' (WNR) that performs discrete wavelet decomposition on input images and carefully select only a few important coefficients determined by our 'wavelet coefficient selection module' (WCSM). To recover the high-frequency content of the image after noise removal via WNR, we further train a 'regenerator' network with an objective to retrieve the coefficients such that the reconstructed image yields similar to original predictions on the surrogate model. At test time, WNR combined with trained regenerator network is prepended to the black box network, resulting in a high boost in adversarial accuracy. Our method improves the adversarial accuracy on CIFAR-10 by 38.98% and 32.01% on state-of-the-art Auto Attack compared to baseline, even when the attacker uses surrogate architecture (Alexnet-half and Alexnet) similar to the black box architecture (Alexnet) with same model stealing strategy as defender. The code is available at https://github.com/vcl-iisc/data-free-black-box-defense

摘要: 几家公司经常保护他们训练有素的深度模型(即架构细节、学习的重量、训练细节等)。通过API仅将第三方用户暴露为黑盒。此外，由于专有原因或敏感性问题，它们甚至可能无法提供对培训数据的访问。在这项工作中，我们提出了一种新的黑盒模型在无数据环境下抵抗敌意攻击的防御机制。我们通过产生式模型构造合成数据，并使用模型窃取技术训练代理网络。为了最大限度地减少扰动样本带来的有害污染，我们提出了小波去噪器(WNR)，它对输入图像进行离散小波分解，并仔细地选择由我们的小波系数选择模块(WCSM)确定的几个重要系数。为了恢复图像经过WNR去噪后的高频内容，我们进一步训练了一个‘再生器’网络，目的是恢复系数，使重建的图像产生与原始预测相似的代理模型。在测试时，将WNR与训练好的再生器网络相结合，加入到黑盒网络中，大大提高了对抗的准确率。与基准相比，我们的方法在CIFAR-10上的攻击准确率分别提高了38.98%和32.01%，即使攻击者使用类似于黑盒体系结构(Alexnet)的代理体系结构(Alexnet-Half和Alexnet)，并且与防御者使用相同的模型窃取策略。代码可在https://github.com/vcl-iisc/data-free-black-box-defense上获得



## **44. Feature Unlearning for Pre-trained GANs and VAEs**

预训练GAN和VAE的特性取消学习 cs.CV

**SubmitDate**: 2024-03-28    [abs](http://arxiv.org/abs/2303.05699v4) [paper-pdf](http://arxiv.org/pdf/2303.05699v4)

**Authors**: Saemi Moon, Seunghyuk Cho, Dongwoo Kim

**Abstract**: We tackle the problem of feature unlearning from a pre-trained image generative model: GANs and VAEs. Unlike a common unlearning task where an unlearning target is a subset of the training set, we aim to unlearn a specific feature, such as hairstyle from facial images, from the pre-trained generative models. As the target feature is only presented in a local region of an image, unlearning the entire image from the pre-trained model may result in losing other details in the remaining region of the image. To specify which features to unlearn, we collect randomly generated images that contain the target features. We then identify a latent representation corresponding to the target feature and then use the representation to fine-tune the pre-trained model. Through experiments on MNIST, CelebA, and FFHQ datasets, we show that target features are successfully removed while keeping the fidelity of the original models. Further experiments with an adversarial attack show that the unlearned model is more robust under the presence of malicious parties.

摘要: 我们从一个预先训练的图像生成模型GANS和VAE中解决了特征遗忘的问题。与通常的遗忘任务不同，忘记目标是训练集的一个子集，我们的目标是从预先训练的生成模型中忘记特定的特征，如面部图像中的发型。由于目标特征仅呈现在图像的局部区域中，因此从预先训练的模型中不学习整个图像可能导致丢失图像剩余区域中的其他细节。为了指定要取消学习的特征，我们收集包含目标特征的随机生成的图像。然后，我们识别对应于目标特征的潜在表示，然后使用该表示来微调预先训练的模型。通过在MNIST、CelebA和FFHQ数据集上的实验，我们证明了在保持原始模型保真度的情况下，目标特征被成功去除。进一步的对抗性攻击实验表明，未学习模型在恶意方存在的情况下具有更强的鲁棒性。



## **45. JailbreakBench: An Open Robustness Benchmark for Jailbreaking Large Language Models**

JailbreakBench：一个大型语言模型的开放鲁棒性基准测试 cs.CR

**SubmitDate**: 2024-03-28    [abs](http://arxiv.org/abs/2404.01318v1) [paper-pdf](http://arxiv.org/pdf/2404.01318v1)

**Authors**: Patrick Chao, Edoardo Debenedetti, Alexander Robey, Maksym Andriushchenko, Francesco Croce, Vikash Sehwag, Edgar Dobriban, Nicolas Flammarion, George J. Pappas, Florian Tramer, Hamed Hassani, Eric Wong

**Abstract**: Jailbreak attacks cause large language models (LLMs) to generate harmful, unethical, or otherwise objectionable content. Evaluating these attacks presents a number of challenges, which the current collection of benchmarks and evaluation techniques do not adequately address. First, there is no clear standard of practice regarding jailbreaking evaluation. Second, existing works compute costs and success rates in incomparable ways. And third, numerous works are not reproducible, as they withhold adversarial prompts, involve closed-source code, or rely on evolving proprietary APIs. To address these challenges, we introduce JailbreakBench, an open-sourced benchmark with the following components: (1) a new jailbreaking dataset containing 100 unique behaviors, which we call JBB-Behaviors; (2) an evolving repository of state-of-the-art adversarial prompts, which we refer to as jailbreak artifacts; (3) a standardized evaluation framework that includes a clearly defined threat model, system prompts, chat templates, and scoring functions; and (4) a leaderboard that tracks the performance of attacks and defenses for various LLMs. We have carefully considered the potential ethical implications of releasing this benchmark, and believe that it will be a net positive for the community. Over time, we will expand and adapt the benchmark to reflect technical and methodological advances in the research community.

摘要: 越狱攻击会导致大型语言模型(LLM)生成有害、不道德或令人反感的内容。评估这些攻击带来了许多挑战，目前收集的基准和评估技术没有充分解决这些挑战。首先，关于越狱评估没有明确的实践标准。其次，现有的工作以无与伦比的方式计算成本和成功率。第三，许多作品是不可复制的，因为它们保留了对抗性提示，涉及封闭源代码，或者依赖于不断发展的专有API。为了应对这些挑战，我们引入了JailBreak Bch，这是一个开源基准测试，具有以下组件：(1)包含100个独特行为的新越狱数据集，我们称之为JBB行为；(2)不断发展的最新对手提示存储库，我们称为越狱人工制品；(3)标准化评估框架，其中包括明确定义的威胁模型、系统提示、聊天模板和评分功能；以及(4)跟踪各种LLM攻击和防御性能的排行榜。我们已仔细考虑发布这一基准的潜在道德影响，并相信它将为社会带来净积极的影响。随着时间的推移，我们将扩大和调整基准，以反映研究界的技术和方法进步。



## **46. Data Poisoning for In-context Learning**

基于上下文学习的数据中毒 cs.CR

**SubmitDate**: 2024-03-28    [abs](http://arxiv.org/abs/2402.02160v2) [paper-pdf](http://arxiv.org/pdf/2402.02160v2)

**Authors**: Pengfei He, Han Xu, Yue Xing, Hui Liu, Makoto Yamada, Jiliang Tang

**Abstract**: In the domain of large language models (LLMs), in-context learning (ICL) has been recognized for its innovative ability to adapt to new tasks, relying on examples rather than retraining or fine-tuning. This paper delves into the critical issue of ICL's susceptibility to data poisoning attacks, an area not yet fully explored. We wonder whether ICL is vulnerable, with adversaries capable of manipulating example data to degrade model performance. To address this, we introduce ICLPoison, a specialized attacking framework conceived to exploit the learning mechanisms of ICL. Our approach uniquely employs discrete text perturbations to strategically influence the hidden states of LLMs during the ICL process. We outline three representative strategies to implement attacks under our framework, each rigorously evaluated across a variety of models and tasks. Our comprehensive tests, including trials on the sophisticated GPT-4 model, demonstrate that ICL's performance is significantly compromised under our framework. These revelations indicate an urgent need for enhanced defense mechanisms to safeguard the integrity and reliability of LLMs in applications relying on in-context learning.

摘要: 在大型语言模型(LLM)领域，情境学习(ICL)因其适应新任务的创新能力而被公认，它依赖于例子而不是再培训或微调。本文深入研究了ICL对数据中毒攻击的易感性这一关键问题，这是一个尚未完全探索的领域。我们想知道ICL是否易受攻击，因为对手能够操纵示例数据来降低模型性能。为了解决这个问题，我们引入了ICLPoison，这是一个专门的攻击框架，旨在利用ICL的学习机制。我们的方法独特地使用离散文本扰动来战略性地影响ICL过程中LLM的隐藏状态。我们概述了在我们的框架下实施攻击的三种具有代表性的战略，每种战略都在各种模型和任务中进行了严格的评估。我们的全面测试，包括对复杂的GPT-4模型的试验，表明ICL的性能在我们的框架下受到了严重影响。这些发现表明，迫切需要增强防御机制，以保障依赖于情景学习的应用程序中LLMS的完整性和可靠性。



## **47. Towards Sustainable SecureML: Quantifying Carbon Footprint of Adversarial Machine Learning**

迈向可持续的SecureML：量化对抗机器学习的碳足迹 cs.LG

Accepted at GreenNet Workshop @ IEEE International Conference on  Communications (IEEE ICC 2024)

**SubmitDate**: 2024-03-27    [abs](http://arxiv.org/abs/2403.19009v1) [paper-pdf](http://arxiv.org/pdf/2403.19009v1)

**Authors**: Syed Mhamudul Hasan, Abdur R. Shahid, Ahmed Imteaj

**Abstract**: The widespread adoption of machine learning (ML) across various industries has raised sustainability concerns due to its substantial energy usage and carbon emissions. This issue becomes more pressing in adversarial ML, which focuses on enhancing model security against different network-based attacks. Implementing defenses in ML systems often necessitates additional computational resources and network security measures, exacerbating their environmental impacts. In this paper, we pioneer the first investigation into adversarial ML's carbon footprint, providing empirical evidence connecting greater model robustness to higher emissions. Addressing the critical need to quantify this trade-off, we introduce the Robustness Carbon Trade-off Index (RCTI). This novel metric, inspired by economic elasticity principles, captures the sensitivity of carbon emissions to changes in adversarial robustness. We demonstrate the RCTI through an experiment involving evasion attacks, analyzing the interplay between robustness against attacks, performance, and carbon emissions.

摘要: 机器学习(ML)在各个行业的广泛采用引起了人们对其大量能源使用和碳排放的可持续发展的担忧。这一问题在对抗性ML中变得更加紧迫，它的重点是增强模型的安全性，以抵御不同的基于网络的攻击。在ML系统中实施防御通常需要额外的计算资源和网络安全措施，从而加剧了它们对环境的影响。在这篇文章中，我们率先对敌对的ML的碳足迹进行了调查，提供了将更大的模型稳健性与更高的排放量联系起来的经验证据。为了解决量化这种权衡的迫切需要，我们引入了稳健性碳权衡指数(RCTI)。这一新的衡量标准受到经济弹性原则的启发，捕捉了碳排放对对抗性稳健性变化的敏感性。我们通过一个涉及规避攻击的实验来演示RCTI，分析了对攻击的健壮性、性能和碳排放之间的相互影响。



## **48. Robustness and Visual Explanation for Black Box Image, Video, and ECG Signal Classification with Reinforcement Learning**

基于强化学习的黑盒图像、视频和ECG信号分类的鲁棒性和视觉解释 cs.LG

AAAI Proceedings reference:  https://ojs.aaai.org/index.php/AAAI/article/view/30579

**SubmitDate**: 2024-03-27    [abs](http://arxiv.org/abs/2403.18985v1) [paper-pdf](http://arxiv.org/pdf/2403.18985v1)

**Authors**: Soumyendu Sarkar, Ashwin Ramesh Babu, Sajad Mousavi, Vineet Gundecha, Avisek Naug, Sahand Ghorbanpour

**Abstract**: We present a generic Reinforcement Learning (RL) framework optimized for crafting adversarial attacks on different model types spanning from ECG signal analysis (1D), image classification (2D), and video classification (3D). The framework focuses on identifying sensitive regions and inducing misclassifications with minimal distortions and various distortion types. The novel RL method outperforms state-of-the-art methods for all three applications, proving its efficiency. Our RL approach produces superior localization masks, enhancing interpretability for image classification and ECG analysis models. For applications such as ECG analysis, our platform highlights critical ECG segments for clinicians while ensuring resilience against prevalent distortions. This comprehensive tool aims to bolster both resilience with adversarial training and transparency across varied applications and data types.

摘要: 我们提出了一个通用的强化学习（RL）框架，优化用于针对不同模型类型的对抗攻击，这些模型类型涵盖ECG信号分析（1D），图像分类（2D）和视频分类（3D）。该框架的重点是确定敏感区域，并以最小限度的失真和各种失真类型引起错误分类。新的RL方法在所有三个应用中都优于最先进的方法，证明了其效率。我们的RL方法产生了卓越的定位掩模，增强了图像分类和ECG分析模型的可解释性。对于ECG分析等应用，我们的平台为临床医生突出了关键ECG段，同时确保对普遍失真的恢复能力。这一综合性工具旨在通过对抗性培训和不同应用程序和数据类型的透明度来增强弹性。



## **49. Deep Learning for Robust and Explainable Models in Computer Vision**

用于计算机视觉中鲁棒和可解释模型的深度学习 cs.CV

150 pages, 37 figures, 12 tables

**SubmitDate**: 2024-03-27    [abs](http://arxiv.org/abs/2403.18674v1) [paper-pdf](http://arxiv.org/pdf/2403.18674v1)

**Authors**: Mohammadreza Amirian

**Abstract**: Recent breakthroughs in machine and deep learning (ML and DL) research have provided excellent tools for leveraging enormous amounts of data and optimizing huge models with millions of parameters to obtain accurate networks for image processing. These developments open up tremendous opportunities for using artificial intelligence (AI) in the automation and human assisted AI industry. However, as more and more models are deployed and used in practice, many challenges have emerged. This thesis presents various approaches that address robustness and explainability challenges for using ML and DL in practice.   Robustness and reliability are the critical components of any model before certification and deployment in practice. Deep convolutional neural networks (CNNs) exhibit vulnerability to transformations of their inputs, such as rotation and scaling, or intentional manipulations as described in the adversarial attack literature. In addition, building trust in AI-based models requires a better understanding of current models and developing methods that are more explainable and interpretable a priori.   This thesis presents developments in computer vision models' robustness and explainability. Furthermore, this thesis offers an example of using vision models' feature response visualization (models' interpretations) to improve robustness despite interpretability and robustness being seemingly unrelated in the related research. Besides methodological developments for robust and explainable vision models, a key message of this thesis is introducing model interpretation techniques as a tool for understanding vision models and improving their design and robustness. In addition to the theoretical developments, this thesis demonstrates several applications of ML and DL in different contexts, such as medical imaging and affective computing.

摘要: 机器和深度学习(ML和DL)研究的最新突破为利用海量数据和优化具有数百万参数的巨大模型提供了极好的工具，以获得用于图像处理的准确网络。这些发展为人工智能(AI)在自动化和人工辅助AI行业中的使用打开了巨大的机会。然而，随着越来越多的模型在实践中部署和使用，出现了许多挑战。这篇论文提出了各种方法来解决在实践中使用ML和DL时的健壮性和可解释性挑战。在实践中认证和部署之前，健壮性和可靠性是任何模型的关键组件。深层卷积神经网络(CNN)表现出对其输入的变换的脆弱性，例如旋转和缩放，或者如对抗性攻击文献中所描述的故意操纵。此外，建立对基于人工智能的模型的信任需要更好地理解当前的模型，并开发更具解释性和先验性的方法。本文介绍了计算机视觉模型的稳健性和可解释性方面的研究进展。此外，本文还给出了一个使用视觉模型的特征响应可视化(模型的解释)来提高稳健性的例子，尽管可解释性和稳健性在相关研究中似乎是无关的。除了稳健和可解释的视觉模型的方法论发展外，本文的一个关键信息是引入模型解释技术作为理解视觉模型的工具，并改进其设计和稳健性。除了理论上的发展，本文还展示了ML和DL在不同环境中的几个应用，例如医学成像和情感计算。



## **50. LCANets++: Robust Audio Classification using Multi-layer Neural Networks with Lateral Competition**

LCANets ++：使用具有横向竞争的多层神经网络的鲁棒音频分类 cs.SD

Accepted at 2024 IEEE International Conference on Acoustics, Speech  and Signal Processing Workshops (ICASSPW)

**SubmitDate**: 2024-03-27    [abs](http://arxiv.org/abs/2308.12882v2) [paper-pdf](http://arxiv.org/pdf/2308.12882v2)

**Authors**: Sayanton V. Dibbo, Juston S. Moore, Garrett T. Kenyon, Michael A. Teti

**Abstract**: Audio classification aims at recognizing audio signals, including speech commands or sound events. However, current audio classifiers are susceptible to perturbations and adversarial attacks. In addition, real-world audio classification tasks often suffer from limited labeled data. To help bridge these gaps, previous work developed neuro-inspired convolutional neural networks (CNNs) with sparse coding via the Locally Competitive Algorithm (LCA) in the first layer (i.e., LCANets) for computer vision. LCANets learn in a combination of supervised and unsupervised learning, reducing dependency on labeled samples. Motivated by the fact that auditory cortex is also sparse, we extend LCANets to audio recognition tasks and introduce LCANets++, which are CNNs that perform sparse coding in multiple layers via LCA. We demonstrate that LCANets++ are more robust than standard CNNs and LCANets against perturbations, e.g., background noise, as well as black-box and white-box attacks, e.g., evasion and fast gradient sign (FGSM) attacks.

摘要: 音频分类的目的是识别音频信号，包括语音命令或声音事件。然而，当前的音频分类器容易受到扰动和对抗性攻击。此外，现实世界的音频分类任务通常会受到有限的标签数据的影响。为了弥补这些差距，以前的工作发展了神经启发卷积神经网络(CNN)，通过第一层的局部竞争算法(LCA)进行稀疏编码，用于计算机视觉。LCANet在监督和非监督学习的组合中学习，减少了对标记样本的依赖。基于听觉皮层也是稀疏的这一事实，我们将LCANets扩展到音频识别任务，并引入LCANets++，LCANets++是通过LCA在多层进行稀疏编码的CNN。我们证明了LCANet++比标准的CNN和LCANet对扰动(例如背景噪声)以及黑盒和白盒攻击(例如逃避和快速梯度符号(FGSM)攻击)具有更强的鲁棒性。



