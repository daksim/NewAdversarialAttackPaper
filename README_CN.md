# Latest Adversarial Attack Papers
**update at 2023-09-05 10:56:40**

翻译来自 https://cloud.tencent.com/document/product/551/15619

## **1. Baseline Defenses for Adversarial Attacks Against Aligned Language Models**

针对对齐语言模型的对抗性攻击的基线防御 cs.LG

12 pages

**SubmitDate**: 2023-09-01    [abs](http://arxiv.org/abs/2309.00614v1) [paper-pdf](http://arxiv.org/pdf/2309.00614v1)

**Authors**: Neel Jain, Avi Schwarzschild, Yuxin Wen, Gowthami Somepalli, John Kirchenbauer, Ping-yeh Chiang, Micah Goldblum, Aniruddha Saha, Jonas Geiping, Tom Goldstein

**Abstract**: As Large Language Models quickly become ubiquitous, their security vulnerabilities are critical to understand. Recent work shows that text optimizers can produce jailbreaking prompts that bypass moderation and alignment. Drawing from the rich body of work on adversarial machine learning, we approach these attacks with three questions: What threat models are practically useful in this domain? How do baseline defense techniques perform in this new domain? How does LLM security differ from computer vision?   We evaluate several baseline defense strategies against leading adversarial attacks on LLMs, discussing the various settings in which each is feasible and effective. Particularly, we look at three types of defenses: detection (perplexity based), input preprocessing (paraphrase and retokenization), and adversarial training. We discuss white-box and gray-box settings and discuss the robustness-performance trade-off for each of the defenses considered. Surprisingly, we find much more success with filtering and preprocessing than we would expect from other domains, such as vision, providing a first indication that the relative strengths of these defenses may be weighed differently in these domains.

摘要: 随着大型语言模型迅速变得无处不在，理解它们的安全漏洞至关重要。最近的研究表明，文本优化器可以生成绕过审核和对齐的越狱提示。从对抗性机器学习的丰富工作中，我们用三个问题来处理这些攻击：什么威胁模型在这个领域实际上是有用的？基线防御技术在这个新领域的表现如何？LLM安全与计算机视觉有何不同？我们评估了几种针对LLMS的主要对手攻击的基线防御策略，讨论了每种策略可行和有效的各种设置。特别是，我们研究了三种类型的防御：检测(基于困惑)、输入预处理(释义和重新标记化)和对抗性训练。我们讨论了白盒和灰盒设置，并讨论了所考虑的每种防御的稳健性和性能之间的权衡。令人惊讶的是，我们发现过滤和预处理比我们预期的其他领域，如视觉，要成功得多，这第一次表明，这些防御的相对强度在这些领域可能会有不同的权重。



## **2. Why do universal adversarial attacks work on large language models?: Geometry might be the answer**

为什么通用对抗性攻击在大型语言模型上奏效？几何可能是答案 cs.LG

2nd AdvML Frontiers Workshop at 40th International Conference on  Machine Learning, Honolulu, Hawaii, USA, 2023

**SubmitDate**: 2023-09-01    [abs](http://arxiv.org/abs/2309.00254v1) [paper-pdf](http://arxiv.org/pdf/2309.00254v1)

**Authors**: Varshini Subhash, Anna Bialas, Weiwei Pan, Finale Doshi-Velez

**Abstract**: Transformer based large language models with emergent capabilities are becoming increasingly ubiquitous in society. However, the task of understanding and interpreting their internal workings, in the context of adversarial attacks, remains largely unsolved. Gradient-based universal adversarial attacks have been shown to be highly effective on large language models and potentially dangerous due to their input-agnostic nature. This work presents a novel geometric perspective explaining universal adversarial attacks on large language models. By attacking the 117M parameter GPT-2 model, we find evidence indicating that universal adversarial triggers could be embedding vectors which merely approximate the semantic information in their adversarial training region. This hypothesis is supported by white-box model analysis comprising dimensionality reduction and similarity measurement of hidden representations. We believe this new geometric perspective on the underlying mechanism driving universal attacks could help us gain deeper insight into the internal workings and failure modes of LLMs, thus enabling their mitigation.

摘要: 基于转换器的具有紧急能力的大型语言模型在社会上变得越来越普遍。然而，在对抗性攻击的背景下，理解和解释其内部运作的任务在很大程度上仍未解决。基于梯度的通用对抗性攻击已被证明在大型语言模型上非常有效，并且由于其输入不可知的性质而具有潜在的危险。这项工作提出了一种新的几何视角来解释对大型语言模型的普遍对抗性攻击。通过对117M参数GPT-2模型的攻击，我们发现有证据表明，通用的对抗性触发因素可能是嵌入的矢量，这些矢量仅近似于其对抗性训练区的语义信息。这一假设得到了白盒模型分析的支持，白盒模型分析包括对隐藏表示的降维和相似性度量。我们相信，这种关于驱动普遍攻击的潜在机制的新的几何观点可以帮助我们更深入地了解LLMS的内部工作原理和故障模式，从而使其能够得到缓解。



## **3. Image Hijacking: Adversarial Images can Control Generative Models at Runtime**

图像劫持：敌意图像可以在运行时控制生成模型 cs.LG

Code is available at https://github.com/euanong/image-hijacks

**SubmitDate**: 2023-09-01    [abs](http://arxiv.org/abs/2309.00236v1) [paper-pdf](http://arxiv.org/pdf/2309.00236v1)

**Authors**: Luke Bailey, Euan Ong, Stuart Russell, Scott Emmons

**Abstract**: Are foundation models secure from malicious actors? In this work, we focus on the image input to a vision-language model (VLM). We discover image hijacks, adversarial images that control generative models at runtime. We introduce Behavior Matching, a general method for creating image hijacks, and we use it to explore three types of attacks. Specific string attacks generate arbitrary output of the adversary's choosing. Leak context attacks leak information from the context window into the output. Jailbreak attacks circumvent a model's safety training. We study these attacks against LLaVA-2, a state-of-the-art VLM based on CLIP and LLaMA-2, and find that all our attack types have above a 90\% success rate. Moreover, our attacks are automated and require only small image perturbations. These findings raise serious concerns about the security of foundation models. If image hijacks are as difficult to defend against as adversarial examples in CIFAR-10, then it might be many years before a solution is found -- if it even exists.

摘要: 基础模型是否安全，不会受到恶意攻击者的攻击？在这项工作中，我们关注的是图像输入到视觉语言模型(VLM)。我们发现了图像劫持，即在运行时控制生成模型的敌意图像。我们介绍了行为匹配，这是一种创建图像劫持的通用方法，并使用它来探索三种类型的攻击。特定的字符串攻击生成对手选择的任意输出。泄漏上下文攻击将信息从上下文窗口泄漏到输出。越狱袭击绕过了模特的安全培训。我们对基于CLIP和LLAMA-2的最先进的VLM LLaVA-2进行了研究，发现所有的攻击类型都有90%以上的成功率。此外，我们的攻击是自动化的，只需要很小的图像扰动。这些发现引发了人们对基础模型安全性的严重担忧。如果图像劫持像CIFAR-10中的敌意例子一样难以防御，那么可能需要很多年才能找到解决方案--如果它确实存在的话。



## **4. Dynamical systems' based neural networks**

基于动力系统的神经网络 cs.LG

**SubmitDate**: 2023-08-31    [abs](http://arxiv.org/abs/2210.02373v2) [paper-pdf](http://arxiv.org/pdf/2210.02373v2)

**Authors**: Elena Celledoni, Davide Murari, Brynjulf Owren, Carola-Bibiane Schönlieb, Ferdia Sherry

**Abstract**: Neural networks have gained much interest because of their effectiveness in many applications. However, their mathematical properties are generally not well understood. If there is some underlying geometric structure inherent to the data or to the function to approximate, it is often desirable to take this into account in the design of the neural network. In this work, we start with a non-autonomous ODE and build neural networks using a suitable, structure-preserving, numerical time-discretisation. The structure of the neural network is then inferred from the properties of the ODE vector field. Besides injecting more structure into the network architectures, this modelling procedure allows a better theoretical understanding of their behaviour. We present two universal approximation results and demonstrate how to impose some particular properties on the neural networks. A particular focus is on 1-Lipschitz architectures including layers that are not 1-Lipschitz. These networks are expressive and robust against adversarial attacks, as shown for the CIFAR-10 and CIFAR-100 datasets.

摘要: 神经网络因其在许多应用中的有效性而引起人们的极大兴趣。然而，它们的数学性质通常没有被很好地理解。如果数据或函数具有某种内在的几何结构，那么在设计神经网络时往往需要考虑到这一点。在这项工作中，我们从一个非自治的常微分方程开始，并使用适当的、结构保持的、数值时间离散化来建立神经网络。然后从ODE向量场的特性推断出神经网络的结构。除了向网络架构注入更多结构外，此建模过程还允许更好地从理论上理解它们的行为。我们给出了两个普遍的逼近结果，并演示了如何对神经网络施加一些特殊性质。一个特别的焦点是1-Lipschitz体系结构，包括不是1-Lipschitz的层。正如CIFAR-10和CIFAR-100数据集所示，这些网络具有很强的表现力和对对手攻击的健壮性。



## **5. Fault Injection and Safe-Error Attack for Extraction of Embedded Neural Network Models**

嵌入式神经网络模型提取中的故障注入和安全错误攻击 cs.CR

Accepted at SECAI Workshop, ESORICS 2023

**SubmitDate**: 2023-08-31    [abs](http://arxiv.org/abs/2308.16703v1) [paper-pdf](http://arxiv.org/pdf/2308.16703v1)

**Authors**: Kevin Hector, Pierre-Alain Moellic, Mathieu Dumont, Jean-Max Dutertre

**Abstract**: Model extraction emerges as a critical security threat with attack vectors exploiting both algorithmic and implementation-based approaches. The main goal of an attacker is to steal as much information as possible about a protected victim model, so that he can mimic it with a substitute model, even with a limited access to similar training data. Recently, physical attacks such as fault injection have shown worrying efficiency against the integrity and confidentiality of embedded models. We focus on embedded deep neural network models on 32-bit microcontrollers, a widespread family of hardware platforms in IoT, and the use of a standard fault injection strategy - Safe Error Attack (SEA) - to perform a model extraction attack with an adversary having a limited access to training data. Since the attack strongly depends on the input queries, we propose a black-box approach to craft a successful attack set. For a classical convolutional neural network, we successfully recover at least 90% of the most significant bits with about 1500 crafted inputs. These information enable to efficiently train a substitute model, with only 8% of the training dataset, that reaches high fidelity and near identical accuracy level than the victim model.

摘要: 模型提取成为一种严重的安全威胁，攻击载体利用了算法和基于实现的方法。攻击者的主要目标是窃取尽可能多的关于受保护受害者模型的信息，以便他可以使用替代模型来模仿它，即使对类似训练数据的访问权限有限。最近，故障注入等物理攻击对嵌入式模型的完整性和保密性显示出令人担忧的效率。我们专注于在32位微控制器上嵌入深度神经网络模型，物联网中广泛使用的硬件平台，以及使用标准故障注入策略-安全错误攻击(SEA)-在对手访问训练数据有限的情况下执行模型提取攻击。由于攻击强烈依赖于输入查询，我们提出了一种黑盒方法来构建一个成功的攻击集。对于一个经典的卷积神经网络，我们用大约1500个精心设计的输入成功地恢复了至少90%的最高有效位。这些信息使得能够有效地训练替代模型，该替代模型仅使用训练数据集的8%，达到了与受害者模型相同的高保真度和接近相同的精度水平。



## **6. Everyone Can Attack: Repurpose Lossy Compression as a Natural Backdoor Attack**

每个人都可以攻击：将有损压缩重新用作自然的后门攻击 cs.CR

14 pages. This paper shows everyone can mount a powerful and stealthy  backdoor attack with the widely-used lossy image compression

**SubmitDate**: 2023-08-31    [abs](http://arxiv.org/abs/2308.16684v1) [paper-pdf](http://arxiv.org/pdf/2308.16684v1)

**Authors**: Sze Jue Yang, Quang Nguyen, Chee Seng Chan, Khoa Doan

**Abstract**: The vulnerabilities to backdoor attacks have recently threatened the trustworthiness of machine learning models in practical applications. Conventional wisdom suggests that not everyone can be an attacker since the process of designing the trigger generation algorithm often involves significant effort and extensive experimentation to ensure the attack's stealthiness and effectiveness. Alternatively, this paper shows that there exists a more severe backdoor threat: anyone can exploit an easily-accessible algorithm for silent backdoor attacks. Specifically, this attacker can employ the widely-used lossy image compression from a plethora of compression tools to effortlessly inject a trigger pattern into an image without leaving any noticeable trace; i.e., the generated triggers are natural artifacts. One does not require extensive knowledge to click on the "convert" or "save as" button while using tools for lossy image compression. Via this attack, the adversary does not need to design a trigger generator as seen in prior works and only requires poisoning the data. Empirically, the proposed attack consistently achieves 100% attack success rate in several benchmark datasets such as MNIST, CIFAR-10, GTSRB and CelebA. More significantly, the proposed attack can still achieve almost 100% attack success rate with very small (approximately 10%) poisoning rates in the clean label setting. The generated trigger of the proposed attack using one lossy compression algorithm is also transferable across other related compression algorithms, exacerbating the severity of this backdoor threat. This work takes another crucial step toward understanding the extensive risks of backdoor attacks in practice, urging practitioners to investigate similar attacks and relevant backdoor mitigation methods.

摘要: 后门攻击的漏洞最近威胁到了机器学习模型在实际应用中的可信度。传统的观点认为，并不是每个人都能成为攻击者，因为设计触发生成算法的过程通常涉及大量的工作和广泛的实验，以确保攻击的隐蔽性和有效性。或者，本文显示存在更严重的后门威胁：任何人都可以利用一种易于访问的算法进行静默后门攻击。具体地说，该攻击者可以利用大量压缩工具中广泛使用的有损图像压缩，毫不费力地向图像中注入触发图案，而不会留下任何明显的痕迹；即，生成的触发是自然伪像。在使用有损图像压缩工具时，不需要广博的知识就可以点击“转换”或“另存为”按钮。通过这种攻击，攻击者不需要像以前的工作中看到的那样设计触发发生器，只需要毒化数据即可。实验证明，该攻击在MNIST、CIFAR-10、GTSRB和CelebA等多个基准数据集上一致达到100%的攻击成功率。更重要的是，在干净的标签设置下，建议的攻击仍然可以实现几乎100%的攻击成功率，而投毒率非常低(约10%)。使用一种有损压缩算法生成的拟议攻击的触发器也可以在其他相关压缩算法之间传输，从而加剧了这种后门威胁的严重性。这项工作朝着了解实践中后门攻击的广泛风险又迈出了关键的一步，敦促实践者调查类似的攻击和相关的后门缓解方法。



## **7. Fault Injection on Embedded Neural Networks: Impact of a Single Instruction Skip**

故障注入对嵌入式神经网络的影响：单指令跳过 cs.CR

Accepted at DSD 2023 for AHSA Special Session

**SubmitDate**: 2023-08-31    [abs](http://arxiv.org/abs/2308.16665v1) [paper-pdf](http://arxiv.org/pdf/2308.16665v1)

**Authors**: Clement Gaine, Pierre-Alain Moellic, Olivier Potin, Jean-Max Dutertre

**Abstract**: With the large-scale integration and use of neural network models, especially in critical embedded systems, their security assessment to guarantee their reliability is becoming an urgent need. More particularly, models deployed in embedded platforms, such as 32-bit microcontrollers, are physically accessible by adversaries and therefore vulnerable to hardware disturbances. We present the first set of experiments on the use of two fault injection means, electromagnetic and laser injections, applied on neural networks models embedded on a Cortex M4 32-bit microcontroller platform. Contrary to most of state-of-the-art works dedicated to the alteration of the internal parameters or input values, our goal is to simulate and experimentally demonstrate the impact of a specific fault model that is instruction skip. For that purpose, we assessed several modification attacks on the control flow of a neural network inference. We reveal integrity threats by targeting several steps in the inference program of typical convolutional neural network models, which may be exploited by an attacker to alter the predictions of the target models with different adversarial goals.

摘要: 随着神经网络模型的大规模集成和使用，尤其是在关键的嵌入式系统中，对其进行安全评估以保证其可靠性成为一项迫切的需求。更具体地说，部署在嵌入式平台(如32位微控制器)中的模型可被对手物理访问，因此容易受到硬件干扰。我们介绍了第一组实验，使用两种故障注入手段，电磁和激光注入，应用于嵌入在Cortex M4 32位微控制器平台上的神经网络模型。与大多数致力于更改内部参数或输入值的最先进工作相反，我们的目标是模拟和实验演示特定故障模型的影响，即指令跳过。为此，我们评估了几种对神经网络推理控制流的修改攻击。我们通过针对典型卷积神经网络模型推理程序中的几个步骤来揭示完整性威胁，攻击者可以利用这些步骤来更改具有不同敌对目标的目标模型的预测。



## **8. Security Allocation in Networked Control Systems under Stealthy Attacks**

网络控制系统在隐身攻击下的安全分配 eess.SY

11 pages, 3 figures, and 1 table, journal submission

**SubmitDate**: 2023-08-31    [abs](http://arxiv.org/abs/2308.16639v1) [paper-pdf](http://arxiv.org/pdf/2308.16639v1)

**Authors**: Anh Tung Nguyen, André M. H. Teixeira, Alexander Medvedev

**Abstract**: This paper considers the problem of security allocation in a networked control system under stealthy attacks in which the system is comprised of interconnected subsystems represented by vertices. A malicious adversary selects a single vertex on which to conduct a stealthy data injection attack to maximally disrupt the local performance while remaining undetected. On the other hand, a defender selects several vertices on which to allocate defense resources against the adversary. First, the objectives of the adversary and the defender with uncertain targets are formulated in probabilistic ways, resulting in an expected worst-case impact of stealthy attacks. Next, we provide a graph-theoretic necessary and sufficient condition under which the cost for the defender and the expected worst-case impact of stealthy attacks are bounded. This condition enables the defender to restrict the admissible actions to a subset of available vertex sets. Then, we cast the problem of security allocation in a Stackelberg game-theoretic framework. Finally, the contribution of this paper is highlighted by utilizing the proposed admissible actions of the defender in the context of large-scale networks. A numerical example of a 50-vertex networked control system is presented to validate the obtained results.

摘要: 研究了由顶点表示的互联子系统组成的网络控制系统在隐身攻击下的安全分配问题。恶意攻击者选择单个顶点在其上进行隐形数据注入攻击，以最大限度地破坏本地性能，同时保持未被检测到。另一方面，防御者选择几个顶点，在这些顶点上分配防御资源来对抗对手。首先，目标不确定的对手和防御者的目标是以概率的方式制定的，导致了预期的最坏情况下的隐形攻击影响。接下来，我们给出了一个图论的充要条件，在这个充要条件下，防御者的代价和隐身攻击的预期最坏影响是有界的。这一条件使防御者能够将允许的动作限制在可用顶点集的子集上。然后，我们将安全分配问题置于Stackelberg博弈论框架中。最后，通过在大规模网络环境下利用辩护人提出的可受理的诉讼，突出了本文的贡献。给出了一个50点网络控制系统的数值算例，验证了所得结果的正确性。



## **9. The Power of MEME: Adversarial Malware Creation with Model-Based Reinforcement Learning**

模因的力量：基于模型强化学习的敌意恶意软件创建 cs.CR

12 pages, 3 figures, 3 tables. Accepted at ESORICS 2023

**SubmitDate**: 2023-08-31    [abs](http://arxiv.org/abs/2308.16562v1) [paper-pdf](http://arxiv.org/pdf/2308.16562v1)

**Authors**: Maria Rigaki, Sebastian Garcia

**Abstract**: Due to the proliferation of malware, defenders are increasingly turning to automation and machine learning as part of the malware detection tool-chain. However, machine learning models are susceptible to adversarial attacks, requiring the testing of model and product robustness. Meanwhile, attackers also seek to automate malware generation and evasion of antivirus systems, and defenders try to gain insight into their methods. This work proposes a new algorithm that combines Malware Evasion and Model Extraction (MEME) attacks. MEME uses model-based reinforcement learning to adversarially modify Windows executable binary samples while simultaneously training a surrogate model with a high agreement with the target model to evade. To evaluate this method, we compare it with two state-of-the-art attacks in adversarial malware creation, using three well-known published models and one antivirus product as targets. Results show that MEME outperforms the state-of-the-art methods in terms of evasion capabilities in almost all cases, producing evasive malware with an evasion rate in the range of 32-73%. It also produces surrogate models with a prediction label agreement with the respective target models between 97-99%. The surrogate could be used to fine-tune and improve the evasion rate in the future.

摘要: 由于恶意软件的激增，防御者越来越多地将自动化和机器学习作为恶意软件检测工具链的一部分。然而，机器学习模型容易受到对抗性攻击，需要测试模型和产品的稳健性。与此同时，攻击者还试图自动生成和逃避反病毒系统的恶意软件，而防御者则试图了解他们的方法。本文提出了一种结合恶意软件规避和模型提取(MEME)攻击的新算法。Meme使用基于模型的强化学习来恶意修改Windows可执行的二进制样本，同时训练与目标模型高度一致的代理模型来规避。为了对该方法进行评估，我们将其与两种最先进的恶意软件创建攻击进行了比较，使用了三个著名的已发布模型和一个反病毒产品作为目标。结果表明，在几乎所有情况下，Meme在规避能力方面都优于最先进的方法，产生了规避恶意软件，逃避率在32%-73%之间。它还产生预测标签与相应目标模型的符合率在97-99%之间的代理模型。该代理可以用来微调和提高未来的逃避率。



## **10. Why Does Little Robustness Help? Understanding and Improving Adversarial Transferability from Surrogate Training**

为什么小健壮性会有帮助？从替补训练中认识和提高对手的转换性 cs.LG

IEEE Symposium on Security and Privacy (Oakland) 2024; Extended  version of camera-ready

**SubmitDate**: 2023-08-31    [abs](http://arxiv.org/abs/2307.07873v5) [paper-pdf](http://arxiv.org/pdf/2307.07873v5)

**Authors**: Yechao Zhang, Shengshan Hu, Leo Yu Zhang, Junyu Shi, Minghui Li, Xiaogeng Liu, Wei Wan, Hai Jin

**Abstract**: Adversarial examples (AEs) for DNNs have been shown to be transferable: AEs that successfully fool white-box surrogate models can also deceive other black-box models with different architectures. Although a bunch of empirical studies have provided guidance on generating highly transferable AEs, many of these findings lack explanations and even lead to inconsistent advice. In this paper, we take a further step towards understanding adversarial transferability, with a particular focus on surrogate aspects. Starting from the intriguing little robustness phenomenon, where models adversarially trained with mildly perturbed adversarial samples can serve as better surrogates, we attribute it to a trade-off between two predominant factors: model smoothness and gradient similarity. Our investigations focus on their joint effects, rather than their separate correlations with transferability. Through a series of theoretical and empirical analyses, we conjecture that the data distribution shift in adversarial training explains the degradation of gradient similarity. Building on these insights, we explore the impacts of data augmentation and gradient regularization on transferability and identify that the trade-off generally exists in the various training mechanisms, thus building a comprehensive blueprint for the regulation mechanism behind transferability. Finally, we provide a general route for constructing better surrogates to boost transferability which optimizes both model smoothness and gradient similarity simultaneously, e.g., the combination of input gradient regularization and sharpness-aware minimization (SAM), validated by extensive experiments. In summary, we call for attention to the united impacts of these two factors for launching effective transfer attacks, rather than optimizing one while ignoring the other, and emphasize the crucial role of manipulating surrogate models.

摘要: DNN的对抗性例子(AE)已被证明是可移植的：成功欺骗白盒代理模型的AES也可以欺骗其他具有不同体系结构的黑盒模型。尽管大量的实证研究为生成高度可转移的企业实体提供了指导，但其中许多发现缺乏解释，甚至导致了不一致的建议。在这篇文章中，我们进一步了解对手的可转移性，特别关注代理方面。从有趣的小鲁棒性现象开始，我们将其归因于两个主要因素之间的权衡：模型的光滑性和梯度相似性。我们的研究重点是它们的联合影响，而不是它们与可转让性的单独关联。通过一系列的理论和实证分析，我们推测对抗性训练中的数据分布转移解释了梯度相似度的下降。基于这些见解，我们探讨了数据扩充和梯度规则化对可转移性的影响，并确定了各种培训机制中普遍存在的权衡，从而构建了可转移性背后的监管机制的全面蓝图。最后，我们提供了一条同时优化模型光滑性和梯度相似性的构造更好的代理以提高可转移性的一般路线，例如输入梯度正则化和锐度感知最小化(SAM)的组合，并通过大量的实验进行了验证。总之，我们呼吁注意这两个因素对发动有效转移攻击的联合影响，而不是优化一个而忽略另一个，并强调操纵代理模型的关键作用。



## **11. Interpretable and Robust AI in EEG Systems: A Survey**

可解释和稳健的人工智能在脑电系统中的研究进展 eess.SP

**SubmitDate**: 2023-08-30    [abs](http://arxiv.org/abs/2304.10755v2) [paper-pdf](http://arxiv.org/pdf/2304.10755v2)

**Authors**: Xinliang Zhou, Chenyu Liu, Liming Zhai, Ziyu Jia, Cuntai Guan, Yang Liu

**Abstract**: The close coupling of artificial intelligence (AI) and electroencephalography (EEG) has substantially advanced human-computer interaction (HCI) technologies in the AI era. Different from traditional EEG systems, the interpretability and robustness of AI-based EEG systems are becoming particularly crucial. The interpretability clarifies the inner working mechanisms of AI models and thus can gain the trust of users. The robustness reflects the AI's reliability against attacks and perturbations, which is essential for sensitive and fragile EEG signals. Thus the interpretability and robustness of AI in EEG systems have attracted increasing attention, and their research has achieved great progress recently. However, there is still no survey covering recent advances in this field. In this paper, we present the first comprehensive survey and summarize the interpretable and robust AI techniques for EEG systems. Specifically, we first propose a taxonomy of interpretability by characterizing it into three types: backpropagation, perturbation, and inherently interpretable methods. Then we classify the robustness mechanisms into four classes: noise and artifacts, human variability, data acquisition instability, and adversarial attacks. Finally, we identify several critical and unresolved challenges for interpretable and robust AI in EEG systems and further discuss their future directions.

摘要: 人工智能(AI)和脑电(EEG)的紧密结合在AI时代极大地推动了人机交互(HCI)技术的进步。与传统的脑电系统不同，基于人工智能的脑电系统的可解释性和稳健性变得尤为关键。这种可解释性阐明了人工智能模型的内部工作机制，从而可以赢得用户的信任。健壮性反映了人工智能对攻击和扰动的可靠性，这对于敏感和脆弱的脑电信号是必不可少的。因此，人工智能在脑电系统中的可解释性和稳健性受到越来越多的关注，近年来其研究取得了很大进展。然而，目前还没有关于这一领域最新进展的调查。在本文中，我们介绍了第一个全面的综述，并总结了可解释的和健壮的脑电系统人工智能技术。具体地说，我们首先提出了一种可解释性的分类，将其描述为三种类型：反向传播方法、扰动方法和内在可解释方法。然后，我们将健壮性机制分为四类：噪声和伪影、人类可变性、数据获取不稳定性和对抗性攻击。最后，我们确定了脑电系统中可解释的和健壮的人工智能面临的几个关键和尚未解决的挑战，并进一步讨论了它们的未来发展方向。



## **12. Pre-trained transformer for adversarial purification**

用于对抗性净化的预先训练的变压器 cs.CR

**SubmitDate**: 2023-08-30    [abs](http://arxiv.org/abs/2306.01762v2) [paper-pdf](http://arxiv.org/pdf/2306.01762v2)

**Authors**: Kai Wu, Yujian Betterest Li, Xiaoyu Zhang, Handing Wang, Jing Liu

**Abstract**: With more and more deep neural networks being deployed as various daily services, their reliability is essential. It's frightening that deep neural networks are vulnerable and sensitive to adversarial attacks, the most common one of which for the services is evasion-based. Recent works usually strengthen the robustness by adversarial training or leveraging the knowledge of an amount of clean data. However, in practical terms, retraining and redeploying the model need a large computational budget, leading to heavy losses to the online service. In addition, when adversarial examples of a certain attack are detected, only limited adversarial examples are available for the service provider, while much clean data may not be accessible. Given the mentioned problems, we propose a new scenario, RaPiD (Rapid Plug-in Defender), which is to rapidly defend against a certain attack for the frozen original service model with limitations of few clean and adversarial examples. Motivated by the generalization and the universal computation ability of pre-trained transformer models, we come up with a new defender method, CeTaD, which stands for Considering Pre-trained Transformers as Defenders. In particular, we evaluate the effectiveness and the transferability of CeTaD in the case of one-shot adversarial examples and explore the impact of different parts of CeTaD as well as training data conditions. CeTaD is flexible, able to be embedded into an arbitrary differentiable model, and suitable for various types of attacks.

摘要: 随着越来越多的深度神经网络被部署为各种日常服务，其可靠性至关重要。令人恐惧的是，深度神经网络对对手攻击很脆弱和敏感，其中最常见的一种是基于逃避的服务。最近的工作通常通过对抗性训练或利用大量干净数据的知识来增强稳健性。然而，在实际情况下，模型的再培训和重新部署需要大量的计算预算，导致在线服务的严重损失。此外，当检测到特定攻击的对抗性示例时，服务提供商只能获得有限的对抗性示例，而许多干净的数据可能无法访问。针对上述问题，我们提出了一种新的场景RAPID(Rapid Plug-in Defender)，用于快速防御冻结的原有服务模型中的某种攻击，限制了干净和对抗性的例子很少。基于预先训练的变压器模型的泛化和通用计算能力，我们提出了一种新的防御者方法--CeTaD，即将预先训练的变压器视为防御者。特别是，我们评估了CeTaD在单发对抗性例子中的有效性和可转移性，并探讨了CeTaD的不同部分以及训练数据条件的影响。CeTaD是灵活的，能够嵌入到任意可区分的模型中，并且适合于各种类型的攻击。



## **13. Vulnerability of Machine Learning Approaches Applied in IoT-based Smart Grid: A Review**

基于物联网的智能电网中机器学习方法的脆弱性：综述 cs.CR

**SubmitDate**: 2023-08-30    [abs](http://arxiv.org/abs/2308.15736v1) [paper-pdf](http://arxiv.org/pdf/2308.15736v1)

**Authors**: Zhenyong Zhang, Mengxiang Liu, Mingyang Sun, Ruilong Deng, Peng Cheng, Dusit Niyato, Mo-Yuen Chow, Jiming Chen

**Abstract**: The machine learning (ML) sees an increasing prevalence of being used in the internet-of-things enabled smart grid. However, the trustworthiness of ML is a severe issue that must be addressed to accommodate the trend of ML-based smart grid applications (MLsgAPPs). The adversarial distortion injected into the power signal will greatly affect the system's normal control and operation. Therefore, it is imperative to conduct vulnerability assessment for MLsgAPPs applied in the context of safety-critical power systems. In this paper, we provide a comprehensive review of the recent progress in designing attack and defense methods for MLsgAPPs. Unlike the traditional survey about ML security, this is the first review work about the security of MLsgAPPs that focuses on the characteristics of power systems. The survey is organized from the aspects of adversarial assumptions, targeted applications, evaluation metrics, defending approaches, physics-related constraints, and applied datasets. We also highlight future directions on this topic to encourage more researchers to conduct further research on adversarial attacks and defending approaches for MLsgAPPs.

摘要: 机器学习(ML)在物联网智能电网中的应用越来越普遍。然而，ML的可信性是一个必须解决的严重问题，以适应基于ML的智能电网应用(MLsgAPP)的趋势。注入到电源信号中的对抗性失真将极大地影响系统的正常控制和运行。因此，对应用于安全关键电力系统背景下的MLsgAPP进行脆弱性评估势在必行。在本文中，我们提供了一个全面的进展，设计攻击和防御方法的MLsgAPP。与传统的ML安全研究不同，本文首次针对电力系统的特点对MLsgAPP的安全问题进行了综述。调查从对抗性假设、目标应用、评估指标、防御方法、与物理相关的约束和应用数据集等方面进行组织。我们还指出了这一主题的未来方向，以鼓励更多的研究人员对MLsgAPP的对抗性攻击和防御方法进行进一步的研究。



## **14. Intriguing Properties of Diffusion Models: A Large-Scale Dataset for Evaluating Natural Attack Capability in Text-to-Image Generative Models**

扩散模型的有趣性质：用于评估文本到图像生成模型中自然攻击能力的大规模数据集 cs.CV

**SubmitDate**: 2023-08-30    [abs](http://arxiv.org/abs/2308.15692v1) [paper-pdf](http://arxiv.org/pdf/2308.15692v1)

**Authors**: Takami Sato, Justin Yue, Nanze Chen, Ningfei Wang, Qi Alfred Chen

**Abstract**: Denoising probabilistic diffusion models have shown breakthrough performance that can generate more photo-realistic images or human-level illustrations than the prior models such as GANs. This high image-generation capability has stimulated the creation of many downstream applications in various areas. However, we find that this technology is indeed a double-edged sword: We identify a new type of attack, called the Natural Denoising Diffusion (NDD) attack based on the finding that state-of-the-art deep neural network (DNN) models still hold their prediction even if we intentionally remove their robust features, which are essential to the human visual system (HVS), by text prompts. The NDD attack can generate low-cost, model-agnostic, and transferrable adversarial attacks by exploiting the natural attack capability in diffusion models. Motivated by the finding, we construct a large-scale dataset, Natural Denoising Diffusion Attack (NDDA) dataset, to systematically evaluate the risk of the natural attack capability of diffusion models with state-of-the-art text-to-image diffusion models. We evaluate the natural attack capability by answering 6 research questions. Through a user study to confirm the validity of the NDD attack, we find that the NDD attack can achieve an 88% detection rate while being stealthy to 93% of human subjects. We also find that the non-robust features embedded by diffusion models contribute to the natural attack capability. To confirm the model-agnostic and transferrable attack capability, we perform the NDD attack against an AD vehicle and find that 73% of the physically printed attacks can be detected as a stop sign. We hope that our study and dataset can help our community to be aware of the risk of diffusion models and facilitate further research toward robust DNN models.

摘要: 去噪概率扩散模型已经显示出突破性的性能，可以生成比Gans等先前模型更真实的图像或真人级别的插图。这种高图像生成能力刺激了许多不同领域的下游应用的创建。然而，我们发现这项技术确实是一把双刃剑：我们识别出一种新的攻击类型，称为自然去噪扩散(NDD)攻击，基于这一发现，即使我们故意通过文本提示删除对人类视觉系统(HVS)至关重要的健壮特征，最先进的深度神经网络(DNN)模型仍然保持其预测。NDD攻击通过利用扩散模型的自然攻击能力，可以产生低成本、模型无关和可转移的对抗性攻击。在这一发现的启发下，我们构建了一个大规模数据集-自然去噪扩散攻击(NDDA)数据集，以系统地评估使用最先进的文本到图像扩散模型的扩散模型的自然攻击能力的风险。我们通过回答6个研究问题来评估自然攻击能力。通过用户研究来验证NDD攻击的有效性，我们发现NDD攻击可以达到88%的检测率，同时对93%的人类对象具有隐蔽性。我们还发现，扩散模型嵌入的非稳健特征对自然攻击能力有贡献。为了确认模型无关和可转移的攻击能力，我们对AD车辆执行了NDD攻击，发现73%的物理打印攻击可以被检测为停止标志。我们希望我们的研究和数据集能够帮助我们的社区意识到扩散模型的风险，并促进对稳健DNN模型的进一步研究。



## **15. MDTD: A Multi Domain Trojan Detector for Deep Neural Networks**

MDTD：一种面向深度神经网络的多域木马检测器 cs.CR

Accepted to ACM Conference on Computer and Communications Security  (ACM CCS) 2023

**SubmitDate**: 2023-08-30    [abs](http://arxiv.org/abs/2308.15673v1) [paper-pdf](http://arxiv.org/pdf/2308.15673v1)

**Authors**: Arezoo Rajabi, Surudhi Asokraj, Fengqing Jiang, Luyao Niu, Bhaskar Ramasubramanian, Jim Ritcey, Radha Poovendran

**Abstract**: Machine learning models that use deep neural networks (DNNs) are vulnerable to backdoor attacks. An adversary carrying out a backdoor attack embeds a predefined perturbation called a trigger into a small subset of input samples and trains the DNN such that the presence of the trigger in the input results in an adversary-desired output class. Such adversarial retraining however needs to ensure that outputs for inputs without the trigger remain unaffected and provide high classification accuracy on clean samples. In this paper, we propose MDTD, a Multi-Domain Trojan Detector for DNNs, which detects inputs containing a Trojan trigger at testing time. MDTD does not require knowledge of trigger-embedding strategy of the attacker and can be applied to a pre-trained DNN model with image, audio, or graph-based inputs. MDTD leverages an insight that input samples containing a Trojan trigger are located relatively farther away from a decision boundary than clean samples. MDTD estimates the distance to a decision boundary using adversarial learning methods and uses this distance to infer whether a test-time input sample is Trojaned or not. We evaluate MDTD against state-of-the-art Trojan detection methods across five widely used image-based datasets: CIFAR100, CIFAR10, GTSRB, SVHN, and Flowers102; four graph-based datasets: AIDS, WinMal, Toxicant, and COLLAB; and the SpeechCommand audio dataset. MDTD effectively identifies samples that contain different types of Trojan triggers. We evaluate MDTD against adaptive attacks where an adversary trains a robust DNN to increase (decrease) distance of benign (Trojan) inputs from a decision boundary.

摘要: 使用深度神经网络(DNN)的机器学习模型容易受到后门攻击。实施后门攻击的敌手将称为触发器的预定义扰动嵌入到输入样本的一个小子集中，并训练DNN，使得在输入中出现触发器会产生对手所需的输出类别。然而，这种对抗性再培训需要确保没有触发因素的投入产出不受影响，并对清洁样本提供高分类准确率。本文提出了一种用于DNN的多域木马检测器MDTD，它在测试时检测包含木马触发器的输入。MDTD不需要攻击者的触发器嵌入策略的知识，可以应用于具有图像、音频或基于图形的输入的预先训练的DNN模型。MDTD利用了这样一种见解，即包含特洛伊木马触发器的输入样本与干净样本相比，距离决策边界相对较远。MDTD使用对抗性学习方法估计到决策边界的距离，并使用该距离来推断测试时间输入样本是否被特洛伊木马感染。我们使用五个广泛使用的基于图像的数据集：CIFAR100、CIFAR10、GTSRB、SVHN和Flowers102；四个基于图形的数据集：AIDS、WinMal、Toxicant和ColLab；以及SpeechCommand音频数据集，对MDTD和最先进的特洛伊木马检测方法进行了评估。MDTD可有效识别包含不同类型特洛伊木马触发器的样本。我们评估MDTD对抗自适应攻击，其中对手训练一个健壮的DNN来增加(减少)来自决策边界的良性(特洛伊)输入的距离。



## **16. Adaptive Attack Detection in Text Classification: Leveraging Space Exploration Features for Text Sentiment Classification**

文本分类中的自适应攻击检测：利用空间探索特征进行文本情感分类 cs.CR

Presented at 2nd International Workshop on Adaptive Cyber Defense,  2023 (arXiv:2308.09520)

**SubmitDate**: 2023-08-29    [abs](http://arxiv.org/abs/2308.15663v1) [paper-pdf](http://arxiv.org/pdf/2308.15663v1)

**Authors**: Atefeh Mahdavi, Neda Keivandarian, Marco Carvalho

**Abstract**: Adversarial example detection plays a vital role in adaptive cyber defense, especially in the face of rapidly evolving attacks. In adaptive cyber defense, the nature and characteristics of attacks continuously change, making it crucial to have robust mechanisms in place to detect and counter these threats effectively. By incorporating adversarial example detection techniques, adaptive cyber defense systems can enhance their ability to identify and mitigate attacks that attempt to exploit vulnerabilities in machine learning models or other systems. Adversarial examples are inputs that are crafted by applying intentional perturbations to natural inputs that result in incorrect classification. In this paper, we propose a novel approach that leverages the power of BERT (Bidirectional Encoder Representations from Transformers) and introduces the concept of Space Exploration Features. We utilize the feature vectors obtained from the BERT model's output to capture a new representation of feature space to improve the density estimation method.

摘要: 对抗性实例检测在自适应网络防御中起着至关重要的作用，尤其是在面对快速演变的攻击时。在自适应网络防御中，攻击的性质和特征不断变化，因此建立强大的机制来有效检测和应对这些威胁至关重要。通过结合对手实例检测技术，自适应网络防御系统可以增强其识别和缓解试图利用机器学习模型或其他系统中的漏洞的攻击的能力。对抗性的例子是通过对自然输入应用故意的扰动来制作的输入，从而导致不正确的分类。在本文中，我们提出了一种新的方法，该方法利用了来自Transformers的双向编码器表示(BERT)的能力，并引入了空间探索特征的概念。我们利用从BERT模型输出得到的特征向量来获取新的特征空间表示，以改进密度估计方法。



## **17. Everything Perturbed All at Once: Enabling Differentiable Graph Attacks**

一切都被一下子扰乱了：启用可差分图攻击 cs.LG

**SubmitDate**: 2023-08-29    [abs](http://arxiv.org/abs/2308.15614v1) [paper-pdf](http://arxiv.org/pdf/2308.15614v1)

**Authors**: Haoran Liu, Bokun Wang, Jianling Wang, Xiangjue Dong, Tianbao Yang, James Caverlee

**Abstract**: As powerful tools for representation learning on graphs, graph neural networks (GNNs) have played an important role in applications including social networks, recommendation systems, and online web services. However, GNNs have been shown to be vulnerable to adversarial attacks, which can significantly degrade their effectiveness. Recent state-of-the-art approaches in adversarial attacks rely on gradient-based meta-learning to selectively perturb a single edge with the highest attack score until they reach the budget constraint. While effective in identifying vulnerable links, these methods are plagued by high computational costs. By leveraging continuous relaxation and parameterization of the graph structure, we propose a novel attack method called Differentiable Graph Attack (DGA) to efficiently generate effective attacks and meanwhile eliminate the need for costly retraining. Compared to the state-of-the-art, DGA achieves nearly equivalent attack performance with 6 times less training time and 11 times smaller GPU memory footprint on different benchmark datasets. Additionally, we provide extensive experimental analyses of the transferability of the DGA among different graph models, as well as its robustness against widely-used defense mechanisms.

摘要: 图神经网络作为图上表示学习的有力工具，在社会网络、推荐系统、在线Web服务等应用中发挥了重要作用。然而，GNN已被证明容易受到对抗性攻击，这可能会显著降低其有效性。最近在对抗性攻击中最先进的方法依赖于基于梯度的元学习来选择性地扰乱具有最高攻击分数的单个边缘，直到它们达到预算限制。虽然这些方法在识别易受攻击的链路方面很有效，但也存在计算成本高的问题。通过利用图结构的连续松弛和参数化，我们提出了一种新的攻击方法，称为可区分图攻击(Differentiable Graph Attack，DGA)，该方法能够有效地生成有效的攻击，同时消除了代价高昂的重新训练。与最先进的测试数据集相比，DGA在不同的基准数据集上，以6倍的训练时间和11倍的GPU内存占用，获得了几乎相同的攻击性能。此外，我们还对DGA在不同图模型之间的可转移性以及对广泛使用的防御机制的健壮性进行了广泛的实验分析。



## **18. Masquerade: Simple and Lightweight Transaction Reordering Mitigation in Blockchains**

伪装：区块链中简单而轻量级的事务重排序缓解 cs.CR

**SubmitDate**: 2023-08-29    [abs](http://arxiv.org/abs/2308.15347v1) [paper-pdf](http://arxiv.org/pdf/2308.15347v1)

**Authors**: Arti Vedula, Shaileshh Bojja Venkatakrishnan, Abhishek Gupta

**Abstract**: Blockchains offer strong security gurarantees, but cannot protect users against the ordering of transactions. Players such as miners, bots and validators can reorder various transactions and reap significant profits, called the Maximal Extractable Value (MEV). In this paper, we propose an MEV aware protocol design called Masquerade, and show that it will increase user satisfaction and confidence in the system. We propose a strict per-transaction level of ordering to ensure that a transaction is committed either way even if it is revealed. In this protocol, we introduce the notion of a "token" to mitigate the actions taken by an adversary in an attack scenario. Such tokens can be purchased voluntarily by users, who can then choose to include the token numbers in their transactions. If the users include the token in their transactions, then our protocol requires the block-builder to order the transactions strictly according to token numbers. We show through extensive simulations that this reduces the probability that the adversaries can benefit from MEV transactions as compared to existing current practices.

摘要: 区块链提供了强大的安全保证，但无法保护用户免受交易订单的影响。像矿工、机器人和验证员这样的玩家可以对各种交易进行重新排序，并获得可观的利润，称为最大可提取价值(MEV)。在本文中，我们提出了一种MEV感知的协议设计，称为伪装，并表明它将增加用户对系统的满意度和信心。我们提出了严格的每个事务级别的排序，以确保事务以任何一种方式提交，即使它被揭示。在该协议中，我们引入了“令牌”的概念，以减轻对手在攻击场景中所采取的操作。这样的令牌可以由用户自愿购买，然后他们可以选择在他们的交易中包括令牌号。如果用户在其交易中包含令牌，则我们的协议要求块构建器严格按照令牌编号对交易进行排序。我们通过大量的模拟表明，与现有的实践相比，这降低了对手从MEV交易中获益的概率。



## **19. Imperceptible Adversarial Attack on Deep Neural Networks from Image Boundary**

基于图像边界的深层神经网络的潜伏性攻击 cs.LG

**SubmitDate**: 2023-08-29    [abs](http://arxiv.org/abs/2308.15344v1) [paper-pdf](http://arxiv.org/pdf/2308.15344v1)

**Authors**: Fahad Alrasheedi, Xin Zhong

**Abstract**: Although Deep Neural Networks (DNNs), such as the convolutional neural networks (CNN) and Vision Transformers (ViTs), have been successfully applied in the field of computer vision, they are demonstrated to be vulnerable to well-sought Adversarial Examples (AEs) that can easily fool the DNNs. The research in AEs has been active, and many adversarial attacks and explanations have been proposed since they were discovered in 2014. The mystery of the AE's existence is still an open question, and many studies suggest that DNN training algorithms have blind spots. The salient objects usually do not overlap with boundaries; hence, the boundaries are not the DNN model's attention. Nevertheless, recent studies show that the boundaries can dominate the behavior of the DNN models. Hence, this study aims to look at the AEs from a different perspective and proposes an imperceptible adversarial attack that systemically attacks the input image boundary for finding the AEs. The experimental results have shown that the proposed boundary attacking method effectively attacks six CNN models and the ViT using only 32% of the input image content (from the boundaries) with an average success rate (SR) of 95.2% and an average peak signal-to-noise ratio of 41.37 dB. Correlation analyses are conducted, including the relation between the adversarial boundary's width and the SR and how the adversarial boundary changes the DNN model's attention. This paper's discoveries can potentially advance the understanding of AEs and provide a different perspective on how AEs can be constructed.

摘要: 虽然深度神经网络(DNN)，如卷积神经网络(CNN)和视觉变形器(VITS)已经成功地应用于计算机视觉领域，但它们被证明是脆弱的，可以很容易地欺骗DNN。AEs的研究一直很活跃，自2014年发现以来，提出了许多对抗性攻击和解释。声发射的存在之谜仍然是一个悬而未决的问题，许多研究表明DNN训练算法存在盲区。显著对象通常不与边界重叠；因此，边界不在DNN模型的关注范围内。然而，最近的研究表明，边界可以支配DNN模型的行为。因此，本研究旨在从不同的角度来看待特效图像，并提出了一种系统地攻击输入图像边界来寻找特效图像的隐蔽对抗性攻击方法。实验结果表明，所提出的边界攻击方法仅用32%的输入图像内容(来自边界)就能有效地攻击6个CNN模型和VIT，平均成功率为95.2%，平均峰值信噪比为41.37dB。进行了相关分析，包括对抗性边界宽度与SR的关系，以及对抗性边界如何改变DNN模型的关注度。这篇论文的发现可能会促进对企业实体的理解，并为如何构建企业实体提供一个不同的视角。



## **20. Longest-chain Attacks: Difficulty Adjustment and Timestamp Verifiability**

最长链攻击：难度调整和时间戳可验证性 cs.CR

A short version appears at MobiHoc23 as a poster

**SubmitDate**: 2023-08-29    [abs](http://arxiv.org/abs/2308.15312v1) [paper-pdf](http://arxiv.org/pdf/2308.15312v1)

**Authors**: Tzuo Hann Law, Selman Erol, Lewis Tseng

**Abstract**: We study an adversary who attacks a Proof-of-Work (POW) blockchain by selfishly constructing an alternative longest chain. We characterize optimal strategies employed by the adversary when a difficulty adjustment rule al\`a Bitcoin applies. As time (namely the times-tamp specified in each block) in most permissionless POW blockchains is somewhat subjective, we focus on two extreme scenarios: when time is completely verifiable, and when it is completely unverifiable. We conclude that an adversary who faces a difficulty adjustment rule will find a longest-chain attack very challenging when timestamps are verifiable. POW blockchains with frequent difficulty adjustments relative to time reporting flexibility will be substantially more vulnerable to longest-chain attacks. Our main fining provides guidance on the design of difficulty adjustment rules and demonstrates the importance of timestamp verifiability.

摘要: 我们研究了一个通过自私地构建替代最长链来攻击工作证明(POW)区块链的对手。我们刻画了当难度调整规则或比特币适用时，对手所采用的最优策略。由于大多数未经许可的战俘区块链中的时间(即每个区块中指定的时间篡改)具有一定的主观性，因此我们关注两个极端场景：当时间完全可验证时，以及当它完全不可验证时。我们的结论是，当时间戳可验证时，面临难度调整规则的对手将发现最长链攻击非常具有挑战性。战俘区块链相对于时间报告灵活性具有频繁的难度调整，将大大更容易受到最长链攻击。我们的主要提炼为难度调整规则的设计提供了指导，并证明了时间戳可验证性的重要性。



## **21. A Classification-Guided Approach for Adversarial Attacks against Neural Machine Translation**

一种分类制导的神经机器翻译对抗性攻击方法 cs.CL

**SubmitDate**: 2023-08-29    [abs](http://arxiv.org/abs/2308.15246v1) [paper-pdf](http://arxiv.org/pdf/2308.15246v1)

**Authors**: Sahar Sadrizadeh, Ljiljana Dolamic, Pascal Frossard

**Abstract**: Neural Machine Translation (NMT) models have been shown to be vulnerable to adversarial attacks, wherein carefully crafted perturbations of the input can mislead the target model. In this paper, we introduce ACT, a novel adversarial attack framework against NMT systems guided by a classifier. In our attack, the adversary aims to craft meaning-preserving adversarial examples whose translations by the NMT model belong to a different class than the original translations in the target language. Unlike previous attacks, our new approach has a more substantial effect on the translation by altering the overall meaning, which leads to a different class determined by a classifier. To evaluate the robustness of NMT models to this attack, we propose enhancements to existing black-box word-replacement-based attacks by incorporating output translations of the target NMT model and the output logits of a classifier within the attack process. Extensive experiments in various settings, including a comparison with existing untargeted attacks, demonstrate that the proposed attack is considerably more successful in altering the class of the output translation and has more effect on the translation. This new paradigm can show the vulnerabilities of NMT systems by focusing on the class of translation rather than the mere translation quality as studied traditionally.

摘要: 神经机器翻译(NMT)模型已被证明容易受到敌意攻击，其中精心设计的输入扰动可能会误导目标模型。本文介绍了一种新的基于分类器的NMT系统对抗性攻击框架ACT。在我们的攻击中，对手的目标是制作保持意义的对抗性例子，其NMT模型的翻译与目标语言的原始翻译属于不同的类别。与以前的攻击不同，我们的新方法通过改变整体意义来对翻译产生更实质性的影响，这导致了由分类器确定的不同类别。为了评估NMT模型对这种攻击的稳健性，我们通过在攻击过程中结合目标NMT模型的输出翻译和分类器的输出日志，对现有的基于黑盒单词替换的攻击进行了增强。在不同环境下的大量实验，包括与现有的非定向攻击的比较，表明所提出的攻击在改变输出翻译的类别方面取得了显著的成功，并且对翻译产生了更大的影响。这种新的范式可以通过关注翻译的类别而不是传统研究的翻译质量来揭示NMT系统的脆弱性。



## **22. On the Steganographic Capacity of Selected Learning Models**

关于选定学习模型的隐写容量 cs.LG

arXiv admin note: text overlap with arXiv:2306.17189

**SubmitDate**: 2023-08-29    [abs](http://arxiv.org/abs/2308.15502v1) [paper-pdf](http://arxiv.org/pdf/2308.15502v1)

**Authors**: Rishit Agrawal, Kelvin Jou, Tanush Obili, Daksh Parikh, Samarth Prajapati, Yash Seth, Charan Sridhar, Nathan Zhang, Mark Stamp

**Abstract**: Machine learning and deep learning models are potential vectors for various attack scenarios. For example, previous research has shown that malware can be hidden in deep learning models. Hiding information in a learning model can be viewed as a form of steganography. In this research, we consider the general question of the steganographic capacity of learning models. Specifically, for a wide range of models, we determine the number of low-order bits of the trained parameters that can be overwritten, without adversely affecting model performance. For each model considered, we graph the accuracy as a function of the number of low-order bits that have been overwritten, and for selected models, we also analyze the steganographic capacity of individual layers. The models that we test include the classic machine learning techniques of Linear Regression (LR) and Support Vector Machine (SVM); the popular general deep learning models of Multilayer Perceptron (MLP) and Convolutional Neural Network (CNN); the highly-successful Recurrent Neural Network (RNN) architecture of Long Short-Term Memory (LSTM); the pre-trained transfer learning-based models VGG16, DenseNet121, InceptionV3, and Xception; and, finally, an Auxiliary Classifier Generative Adversarial Network (ACGAN). In all cases, we find that a majority of the bits of each trained parameter can be overwritten before the accuracy degrades. Of the models tested, the steganographic capacity ranges from 7.04 KB for our LR experiments, to 44.74 MB for InceptionV3. We discuss the implications of our results and consider possible avenues for further research.

摘要: 机器学习和深度学习模型是各种攻击场景的潜在载体。例如，之前的研究表明，恶意软件可以隐藏在深度学习模型中。在学习模型中隐藏信息可以被视为一种隐写术。在这项研究中，我们考虑了学习模型的隐写能力的一般问题。具体地说，对于广泛的模型，我们确定了训练参数中可以覆盖的低阶位的数量，而不会对模型性能产生不利影响。对于所考虑的每个模型，我们将精确度绘制为已被覆盖的低阶位的数量的函数，并且对于选定的模型，我们还分析了各个层的隐写容量。我们测试的模型包括经典的机器学习技术线性回归(LR)和支持向量机(SVM)；流行的通用深度学习模型多层感知器(MLP)和卷积神经网络(CNN)；非常成功的长期短期记忆(LSTM)的递归神经网络(RNN)结构；基于预训练的基于迁移学习的模型VGG16、DenseNet121、InceptionV3和Exception；最后，辅助分类器生成性对手网络(ACGAN)。在所有情况下，我们发现每个训练参数的大部分比特可以在精度降级之前被重写。在测试的模型中，隐写容量从我们的LR实验的7.04KB到InceptionV3的44.74MB。我们讨论了我们的结果的含义，并考虑了进一步研究的可能途径。



## **23. Can We Rely on AI?**

我们能依靠人工智能吗？ math.NA

**SubmitDate**: 2023-08-29    [abs](http://arxiv.org/abs/2308.15092v1) [paper-pdf](http://arxiv.org/pdf/2308.15092v1)

**Authors**: Desmond J. Higham

**Abstract**: Over the last decade, adversarial attack algorithms have revealed instabilities in deep learning tools. These algorithms raise issues regarding safety, reliability and interpretability in artificial intelligence; especially in high risk settings. From a practical perspective, there has been a war of escalation between those developing attack and defence strategies. At a more theoretical level, researchers have also studied bigger picture questions concerning the existence and computability of attacks. Here we give a brief overview of the topic, focusing on aspects that are likely to be of interest to researchers in applied and computational mathematics.

摘要: 在过去的十年里，对抗性攻击算法揭示了深度学习工具的不稳定性。这些算法提出了人工智能中的安全性、可靠性和可解释性问题；特别是在高风险环境中。从实践的角度来看，那些制定攻防战略的人之间已经发生了一场不断升级的战争。在更理论的层面上，研究人员还研究了有关攻击的存在和可计算性的更大问题。在这里，我们对这个主题做一个简要的概述，集中在应用数学和计算数学研究人员可能感兴趣的方面。



## **24. Advancing Adversarial Robustness Through Adversarial Logit Update**

通过对抗性Logit更新提高对抗性健壮性 cs.LG

**SubmitDate**: 2023-08-29    [abs](http://arxiv.org/abs/2308.15072v1) [paper-pdf](http://arxiv.org/pdf/2308.15072v1)

**Authors**: Hao Xuan, Peican Zhu, Xingyu Li

**Abstract**: Deep Neural Networks are susceptible to adversarial perturbations. Adversarial training and adversarial purification are among the most widely recognized defense strategies. Although these methods have different underlying logic, both rely on absolute logit values to generate label predictions. In this study, we theoretically analyze the logit difference around successful adversarial attacks from a theoretical point of view and propose a new principle, namely Adversarial Logit Update (ALU), to infer adversarial sample's labels. Based on ALU, we introduce a new classification paradigm that utilizes pre- and post-purification logit differences for model's adversarial robustness boost. Without requiring adversarial or additional data for model training, our clean data synthesis model can be easily applied to various pre-trained models for both adversarial sample detection and ALU-based data classification. Extensive experiments on both CIFAR-10, CIFAR-100, and tiny-ImageNet datasets show that even with simple components, the proposed solution achieves superior robustness performance compared to state-of-the-art methods against a wide range of adversarial attacks. Our python implementation is submitted in our Supplementary document and will be published upon the paper's acceptance.

摘要: 深度神经网络容易受到对抗性扰动的影响。对抗性训练和对抗性净化是最广为人知的防御策略。尽管这两种方法具有不同的底层逻辑，但它们都依赖于绝对Logit值来生成标签预测。在本研究中，我们从理论的角度分析了成功的对抗性攻击前后的Logit差异，并提出了一种新的原理，即对抗性Logit更新(ALU)来推断对抗性样本的标签。在ALU的基础上，我们引入了一种新的分类范式，它利用净化前后的Logit差异来提高模型的对抗性健壮性。在不需要对抗性或额外数据进行模型训练的情况下，我们的清洁数据合成模型可以很容易地应用于各种预先训练的模型，用于对抗性样本检测和基于ALU的数据分类。在CIFAR-10、CIFAR-100和Tiny-ImageNet数据集上的广泛实验表明，与最先进的方法相比，所提出的解决方案即使具有简单的组件，也可以在应对广泛的对手攻击时获得卓越的健壮性性能。我们的Python实现在我们的补充文档中提交，并将在论文接受后发布。



## **25. Double Public Key Signing Function Oracle Attack on EdDSA Software Implementations**

双重公钥签名函数Oracle对EdDSA软件实现的攻击 cs.CR

**SubmitDate**: 2023-08-29    [abs](http://arxiv.org/abs/2308.15009v1) [paper-pdf](http://arxiv.org/pdf/2308.15009v1)

**Authors**: Sam Grierson, Konstantinos Chalkias, William J Buchanan

**Abstract**: EdDSA is a standardised elliptic curve digital signature scheme introduced to overcome some of the issues prevalent in the more established ECDSA standard. Due to the EdDSA standard specifying that the EdDSA signature be deterministic, if the signing function were to be used as a public key signing oracle for the attacker, the unforgeability notion of security of the scheme can be broken. This paper describes an attack against some of the most popular EdDSA implementations, which results in an adversary recovering the private key used during signing. With this recovered secret key, an adversary can sign arbitrary messages that would be seen as valid by the EdDSA verification function. A list of libraries with vulnerable APIs at the time of publication is provided. Furthermore, this paper provides two suggestions for securing EdDSA signing APIs against this vulnerability while it additionally discusses failed attempts to solve the issue.

摘要: EdDSA是一种标准化的椭圆曲线数字签名方案，引入该方案是为了克服在更成熟的ECDSA标准中普遍存在的一些问题。由于EdDSA标准规定EdDSA签名是确定性的，如果签名函数被用作攻击者的公钥签名预言，则方案的安全性的不可伪造性概念可能被打破。本文描述了对一些最流行的EdDSA实现的攻击，该攻击导致攻击者恢复签名期间使用的私钥。利用恢复的密钥，攻击者可以对EdDSA验证功能认为有效的任意消息进行签名。提供了在发布时具有易受攻击的API的库列表。此外，本文还提供了两条保护EdDSA签名API免受该漏洞攻击的建议，同时还讨论了解决该问题的失败尝试。



## **26. Stealthy Backdoor Attack for Code Models**

针对代码模型的隐蔽后门攻击 cs.CR

18 pages, Under review of IEEE Transactions on Software Engineering

**SubmitDate**: 2023-08-29    [abs](http://arxiv.org/abs/2301.02496v2) [paper-pdf](http://arxiv.org/pdf/2301.02496v2)

**Authors**: Zhou Yang, Bowen Xu, Jie M. Zhang, Hong Jin Kang, Jieke Shi, Junda He, David Lo

**Abstract**: Code models, such as CodeBERT and CodeT5, offer general-purpose representations of code and play a vital role in supporting downstream automated software engineering tasks. Most recently, code models were revealed to be vulnerable to backdoor attacks. A code model that is backdoor-attacked can behave normally on clean examples but will produce pre-defined malicious outputs on examples injected with triggers that activate the backdoors. Existing backdoor attacks on code models use unstealthy and easy-to-detect triggers. This paper aims to investigate the vulnerability of code models with stealthy backdoor attacks. To this end, we propose AFRAIDOOR (Adversarial Feature as Adaptive Backdoor). AFRAIDOOR achieves stealthiness by leveraging adversarial perturbations to inject adaptive triggers into different inputs. We evaluate AFRAIDOOR on three widely adopted code models (CodeBERT, PLBART and CodeT5) and two downstream tasks (code summarization and method name prediction). We find that around 85% of adaptive triggers in AFRAIDOOR bypass the detection in the defense process. By contrast, only less than 12% of the triggers from previous work bypass the defense. When the defense method is not applied, both AFRAIDOOR and baselines have almost perfect attack success rates. However, once a defense is applied, the success rates of baselines decrease dramatically to 10.47% and 12.06%, while the success rate of AFRAIDOOR are 77.05% and 92.98% on the two tasks. Our finding exposes security weaknesses in code models under stealthy backdoor attacks and shows that the state-of-the-art defense method cannot provide sufficient protection. We call for more research efforts in understanding security threats to code models and developing more effective countermeasures.

摘要: 代码模型，如CodeBERT和CodeT5，提供了代码的通用表示，并在支持下游自动化软件工程任务方面发挥了至关重要的作用。最近，代码模型被发现容易受到后门攻击。被后门攻击的代码模型可以在干净的示例上正常运行，但会在注入了激活后门的触发器的示例上生成预定义的恶意输出。现有对代码模型的后门攻击使用隐蔽且易于检测的触发器。本文旨在研究具有隐蔽后门攻击的代码模型的脆弱性。为此，我们提出了AFRAIDOOR(对抗性特征作为自适应后门)。AFRAIDOOR通过利用对抗性扰动将自适应触发器注入不同的输入来实现隐蔽性。我们在三个广泛采用的代码模型(CodeBERT、PLBART和CodeT5)和两个下游任务(代码摘要和方法名称预测)上对AFRAIDOOR进行了评估。我们发现，AFRAIDOOR中约85%的自适应触发器在防御过程中绕过了检测。相比之下，只有不到12%的以前工作中的触发因素绕过了防御。当不应用防御方法时，AFRAIDOOR和基线都具有几乎完美的攻击成功率。然而，一旦实施防御，基线的成功率急剧下降到10.47%和12.06%，而AFRAIDOOR在两个任务上的成功率分别为77.05%和92.98%。我们的发现暴露了代码模型在秘密后门攻击下的安全漏洞，并表明最先进的防御方法不能提供足够的保护。我们呼吁在了解代码模型的安全威胁和开发更有效的对策方面做出更多研究努力。



## **27. WSAM: Visual Explanations from Style Augmentation as Adversarial Attacker and Their Influence in Image Classification**

WSAM：作为对抗性攻击者的风格提升的视觉解释及其对图像分类的影响 cs.CV

8 pages, 10 figures

**SubmitDate**: 2023-08-29    [abs](http://arxiv.org/abs/2308.14995v1) [paper-pdf](http://arxiv.org/pdf/2308.14995v1)

**Authors**: Felipe Moreno-Vera, Edgar Medina, Jorge Poco

**Abstract**: Currently, style augmentation is capturing attention due to convolutional neural networks (CNN) being strongly biased toward recognizing textures rather than shapes. Most existing styling methods either perform a low-fidelity style transfer or a weak style representation in the embedding vector. This paper outlines a style augmentation algorithm using stochastic-based sampling with noise addition to improving randomization on a general linear transformation for style transfer. With our augmentation strategy, all models not only present incredible robustness against image stylizing but also outperform all previous methods and surpass the state-of-the-art performance for the STL-10 dataset. In addition, we present an analysis of the model interpretations under different style variations. At the same time, we compare comprehensive experiments demonstrating the performance when applied to deep neural architectures in training settings.

摘要: 目前，由于卷积神经网络(CNN)强烈偏向于识别纹理而不是形状，样式增强正吸引着人们的注意。大多数现有的样式设置方法要么执行低保真样式转换，要么在嵌入向量中执行弱样式表示。本文提出了一种基于随机采样和噪声的风格增强算法，改进了一般线性变换的随机性，用于风格转移。使用我们的增强策略，所有模型不仅在图像样式化方面表现出令人难以置信的健壮性，而且性能优于所有以前的方法，并超过STL-10数据集的最先进性能。此外，我们还对不同风格变化下的模型解释进行了分析。同时，我们比较了在训练环境中应用于深层神经结构时的性能的综合实验。



## **28. Randomized Line-to-Row Mapping for Low-Overhead Rowhammer Mitigations**

用于低开销Rowhammer缓解的随机化行到行映射 cs.CR

**SubmitDate**: 2023-08-28    [abs](http://arxiv.org/abs/2308.14907v1) [paper-pdf](http://arxiv.org/pdf/2308.14907v1)

**Authors**: Anish Saxena, Saurav Mathur, Moinuddin Qureshi

**Abstract**: Modern systems mitigate Rowhammer using victim refresh, which refreshes the two neighbours of an aggressor row when it encounters a specified number of activations. Unfortunately, complex attack patterns like Half-Double break victim-refresh, rendering current systems vulnerable. Instead, recently proposed secure Rowhammer mitigations rely on performing mitigative action on the aggressor rather than the victims. Such schemes employ mitigative actions such as row-migration or access-control and include AQUA, SRS, and Blockhammer. While these schemes incur only modest slowdowns at Rowhammer thresholds of few thousand, they incur prohibitive slowdowns (15%-600%) for lower thresholds that are likely in the near future. The goal of our paper is to make secure Rowhammer mitigations practical at such low thresholds.   Our paper provides the key insights that benign application encounter thousands of hot rows (receiving more activations than the threshold) due to the memory mapping, which places spatially proximate lines in the same row to maximize row-buffer hitrate. Unfortunately, this causes row to receive activations for many frequently used lines. We propose Rubix, which breaks the spatial correlation in the line-to-row mapping by using an encrypted address to access the memory, reducing the likelihood of hot rows by 2 to 3 orders of magnitude. To aid row-buffer hits, Rubix randomizes a group of 1-4 lines. We also propose Rubix-D, which dynamically changes the line-to-row mapping. Rubix-D minimizes hot-rows and makes it much harder for an adversary to learn the spatial neighbourhood of a row. Rubix reduces the slowdown of AQUA (from 15% to 1%), SRS (from 60% to 2%), and Blockhammer (from 600% to 3%) while incurring a storage of less than 1 Kilobyte.

摘要: 现代系统使用受害者刷新来缓解Rowhammer，当遇到指定数量的激活时，受害者刷新会刷新攻击者行的两个相邻行。不幸的是，复杂的攻击模式，如半双中断受害者刷新，使当前的系统容易受到攻击。相反，最近提出的安全罗哈默减轻依赖于对侵略者而不是受害者执行减轻行动。此类方案采用行迁移或访问控制等缓解措施，包括Aqua、SRS和BlockHammer。虽然这些计划在罗哈默几千人的门槛下只会导致适度的减速，但它们会导致令人望而却步的减速(15%-600%)，因为在不久的将来可能会降低门槛。我们论文的目标是在如此低的门槛下使安全的罗哈默减刑变得切实可行。我们的论文提供了一些关键的见解，即良性应用程序会遇到数千个热行(接收的激活数超过阈值)，这是因为内存映射将空间上接近的行放置在同一行中，以最大化行缓冲区命中率。不幸的是，这会导致ROW接收许多常用行的激活。我们提出了Rubix，它通过使用加密地址访问存储器，打破了行到行映射中的空间相关性，将热行的可能性降低了2到3个数量级。为了帮助行缓冲区命中，Rubix随机化了一组1-4行。我们还提出了Rubix-D，它动态地改变行到行的映射。Rubix-D将热行最小化，并使对手更难了解行的空间邻域。Rubix降低了Aqua(从15%到1%)、SRS(从60%到2%)和块锤(从600%到3%)的速度，同时产生了不到1千字节的存储。



## **29. A Stochastic Surveillance Stackelberg Game: Co-Optimizing Defense Placement and Patrol Strategy**

随机监视Stackelberg博弈：共同优化防御部署和巡逻策略 eess.SY

8 pages, 1 figure, jointly submitted to the IEEE Control Systems  Letters and the 2024 American Control Conference

**SubmitDate**: 2023-08-28    [abs](http://arxiv.org/abs/2308.14714v1) [paper-pdf](http://arxiv.org/pdf/2308.14714v1)

**Authors**: Yohan John, Gilberto Diaz-Garcia, Xiaoming Duan, Jason R. Marden, Francesco Bullo

**Abstract**: Stochastic patrol routing is known to be advantageous in adversarial settings; however, the optimal choice of stochastic routing strategy is dependent on a model of the adversary. Duan et al. formulated a Stackelberg game for the worst-case scenario, i.e., a surveillance agent confronted with an omniscient attacker [IEEE TCNS, 8(2), 769-80, 2021]. In this article, we extend their formulation to accommodate heterogeneous defenses at the various nodes of the graph. We derive an upper bound on the value of the game. We identify methods for computing effective patrol strategies for certain classes of graphs. Finally, we leverage the heterogeneous defense formulation to develop novel defense placement algorithms that complement the patrol strategies.

摘要: 随机巡逻路径在敌方环境中具有优势，然而，随机路径策略的最优选择取决于敌方的模型。Duan et al.为最糟糕的情况制定了Stackelberg游戏，即监视特工与无所不知的攻击者对峙[IEEE TCNs，8(2)，769-80,2021]。在这篇文章中，我们扩展了他们的公式，以适应图的不同节点上的不同防御。我们得到了博弈价值的一个上界。我们确定了计算某些图类的有效巡视策略的方法。最后，我们利用异质防御公式来开发新的防御布局算法，以补充巡逻策略。



## **30. Adversarial Attacks on Foundational Vision Models**

对基本视觉模型的对抗性攻击 cs.CV

**SubmitDate**: 2023-08-28    [abs](http://arxiv.org/abs/2308.14597v1) [paper-pdf](http://arxiv.org/pdf/2308.14597v1)

**Authors**: Nathan Inkawhich, Gwendolyn McDonald, Ryan Luley

**Abstract**: Rapid progress is being made in developing large, pretrained, task-agnostic foundational vision models such as CLIP, ALIGN, DINOv2, etc. In fact, we are approaching the point where these models do not have to be finetuned downstream, and can simply be used in zero-shot or with a lightweight probing head. Critically, given the complexity of working at this scale, there is a bottleneck where relatively few organizations in the world are executing the training then sharing the models on centralized platforms such as HuggingFace and torch.hub. The goal of this work is to identify several key adversarial vulnerabilities of these models in an effort to make future designs more robust. Intuitively, our attacks manipulate deep feature representations to fool an out-of-distribution (OOD) detector which will be required when using these open-world-aware models to solve closed-set downstream tasks. Our methods reliably make in-distribution (ID) images (w.r.t. a downstream task) be predicted as OOD and vice versa while existing in extremely low-knowledge-assumption threat models. We show our attacks to be potent in whitebox and blackbox settings, as well as when transferred across foundational model types (e.g., attack DINOv2 with CLIP)! This work is only just the beginning of a long journey towards adversarially robust foundational vision models.

摘要: 在开发大型的、预先训练的、与任务无关的基础视觉模型方面，如CLIP、ALIGN、DINOv2等，正在取得快速进展。事实上，我们正在接近这样一个点，这些模型不必在下游进行微调，只需在零射击或带有轻型探头的情况下使用即可。关键是，考虑到在这种规模下工作的复杂性，存在一个瓶颈，即世界上执行培训并在HuggingFace和Torch.Hub等集中式平台上共享模型的组织相对较少。这项工作的目标是确定这些模型的几个关键的对抗性漏洞，以努力使未来的设计更健壮。直观地说，我们的攻击操纵深层特征表示来愚弄分布外(OOD)检测器，这将是使用这些开放世界感知模型来解决封闭集下游任务时所需的。我们的方法可靠地生成分布内(ID)图像(W.r.t.下游任务)被预测为OOD，反之亦然，而存在于极低知识假设的威胁模型中。我们展示了我们的攻击在白盒和黑盒设置中以及在跨基本模型类型传输时是有效的(例如，使用CLIP攻击DINOv2)！这项工作仅仅是迈向相反稳健的基础愿景模型的漫长旅程的开始。



## **31. ReMAV: Reward Modeling of Autonomous Vehicles for Finding Likely Failure Events**

ReMAV：自动车辆发现可能故障事件的奖励模型 cs.AI

**SubmitDate**: 2023-08-28    [abs](http://arxiv.org/abs/2308.14550v1) [paper-pdf](http://arxiv.org/pdf/2308.14550v1)

**Authors**: Aizaz Sharif, Dusica Marijan

**Abstract**: Autonomous vehicles are advanced driving systems that are well known for being vulnerable to various adversarial attacks, compromising the vehicle's safety, and posing danger to other road users. Rather than actively training complex adversaries by interacting with the environment, there is a need to first intelligently find and reduce the search space to only those states where autonomous vehicles are found less confident. In this paper, we propose a blackbox testing framework ReMAV using offline trajectories first to analyze the existing behavior of autonomous vehicles and determine appropriate thresholds for finding the probability of failure events. Our reward modeling technique helps in creating a behavior representation that allows us to highlight regions of likely uncertain behavior even when the baseline autonomous vehicle is performing well. This approach allows for more efficient testing without the need for computational and inefficient active adversarial learning techniques. We perform our experiments in a high-fidelity urban driving environment using three different driving scenarios containing single and multi-agent interactions. Our experiment shows 35%, 23%, 48%, and 50% increase in occurrences of vehicle collision, road objects collision, pedestrian collision, and offroad steering events respectively by the autonomous vehicle under test, demonstrating a significant increase in failure events. We also perform a comparative analysis with prior testing frameworks and show that they underperform in terms of training-testing efficiency, finding total infractions, and simulation steps to identify the first failure compared to our approach. The results show that the proposed framework can be used to understand existing weaknesses of the autonomous vehicles under test in order to only attack those regions, starting with the simplistic perturbation models.

摘要: 自动驾驶汽车是一种先进的驾驶系统，众所周知，它容易受到各种对抗性攻击，危及车辆的安全，并对其他道路使用者构成危险。与其通过与环境互动来积极训练复杂的对手，不如首先智能地找到搜索空间，并将搜索空间缩小到那些自动驾驶汽车被发现信心较低的状态。在本文中，我们提出了一个基于离线轨迹的黑盒测试框架ReMAV，首先分析自动驾驶车辆的现有行为，并确定合适的阈值来发现故障事件的概率。我们的奖励建模技术有助于创建行为表示，使我们能够突出可能的不确定行为区域，即使基准自动驾驶汽车表现良好。这种方法允许更有效的测试，而不需要计算和低效的主动对抗性学习技术。我们在高保真的城市驾驶环境中使用三种不同的驾驶场景进行了实验，其中包含单代理和多代理交互。我们的实验表明，被测自动驾驶汽车的车辆碰撞、道路物体碰撞、行人碰撞和越野转向事件的发生率分别增加了35%、23%、48%和50%，表明故障事件显著增加。我们还与以前的测试框架进行了比较分析，结果表明，与我们的方法相比，它们在训练测试效率、找到总违规行为以及识别第一个故障的模拟步骤方面表现不佳。结果表明，该框架可以用来理解被测自动驾驶车辆存在的弱点，以便从简化的扰动模型开始只攻击这些区域。



## **32. Efficient Decision-based Black-box Patch Attacks on Video Recognition**

视频识别中高效的基于决策的黑盒补丁攻击 cs.CV

**SubmitDate**: 2023-08-28    [abs](http://arxiv.org/abs/2303.11917v2) [paper-pdf](http://arxiv.org/pdf/2303.11917v2)

**Authors**: Kaixun Jiang, Zhaoyu Chen, Hao Huang, Jiafeng Wang, Dingkang Yang, Bo Li, Yan Wang, Wenqiang Zhang

**Abstract**: Although Deep Neural Networks (DNNs) have demonstrated excellent performance, they are vulnerable to adversarial patches that introduce perceptible and localized perturbations to the input. Generating adversarial patches on images has received much attention, while adversarial patches on videos have not been well investigated. Further, decision-based attacks, where attackers only access the predicted hard labels by querying threat models, have not been well explored on video models either, even if they are practical in real-world video recognition scenes. The absence of such studies leads to a huge gap in the robustness assessment for video models. To bridge this gap, this work first explores decision-based patch attacks on video models. We analyze that the huge parameter space brought by videos and the minimal information returned by decision-based models both greatly increase the attack difficulty and query burden. To achieve a query-efficient attack, we propose a spatial-temporal differential evolution (STDE) framework. First, STDE introduces target videos as patch textures and only adds patches on keyframes that are adaptively selected by temporal difference. Second, STDE takes minimizing the patch area as the optimization objective and adopts spatialtemporal mutation and crossover to search for the global optimum without falling into the local optimum. Experiments show STDE has demonstrated state-of-the-art performance in terms of threat, efficiency and imperceptibility. Hence, STDE has the potential to be a powerful tool for evaluating the robustness of video recognition models.

摘要: 尽管深度神经网络(DNN)表现出了很好的性能，但它们很容易受到敌意补丁的攻击，这些补丁会给输入带来可感知的局部扰动。在图像上生成敌意补丁已经得到了很大的关注，而视频上的敌意补丁还没有得到很好的研究。此外，基于决策的攻击(攻击者仅通过查询威胁模型来访问预测的硬标签)在视频模型上也没有得到很好的探索，即使它们在现实世界的视频识别场景中是实用的。这类研究的缺乏导致了视频模型稳健性评估的巨大差距。为了弥补这一差距，这项工作首先探索了基于决策的视频模型补丁攻击。分析了视频带来的巨大参数空间和基于决策的模型返回的最小信息量都大大增加了攻击难度和查询负担。为了实现查询高效的攻击，我们提出了一种时空差异进化(STDE)框架。首先，STDE将目标视频作为补丁纹理引入，只在根据时间差异自适应选择的关键帧上添加补丁。其次，STDE算法以面片面积最小为优化目标，采用时空变异和交叉来搜索全局最优解而不陷入局部最优。实验表明，STDE在威胁、效率和不可感知性方面都表现出了最先进的性能。因此，STDE有可能成为评估视频识别模型稳健性的有力工具。



## **33. Mitigating the source-side channel vulnerability by characterization of photon statistics**

通过表征光子统计信息来缓解源端通道的脆弱性 quant-ph

Comments and suggestions are welcomed

**SubmitDate**: 2023-08-28    [abs](http://arxiv.org/abs/2308.14402v1) [paper-pdf](http://arxiv.org/pdf/2308.14402v1)

**Authors**: Tanya Sharma, Ayan Biswas, Jayanth Ramakrishnan, Pooja Chandravanshi, Ravindra P. Singh

**Abstract**: Quantum key distribution (QKD) theoretically offers unconditional security. Unfortunately, the gap between theory and practice threatens side-channel attacks on practical QKD systems. Many well-known QKD protocols use weak coherent laser pulses to encode the quantum information. These sources differ from ideal single photon sources and follow Poisson statistics. Many protocols, such as decoy state and coincidence detection protocols, rely on monitoring the photon statistics to detect any information leakage. The accurate measurement and characterization of photon statistics enable the detection of adversarial attacks and the estimation of secure key rates, strengthening the overall security of the QKD system. We have rigorously characterized our source to estimate the mean photon number employing multiple detectors for comparison against measurements made with a single detector. Furthermore, we have also studied intensity fluctuations to help identify and mitigate any potential information leakage due to state preparation flaws. We aim to bridge the gap between theory and practice to achieve information-theoretic security.

摘要: 量子密钥分配(QKD)理论上提供了无条件的安全性。不幸的是，理论和实践之间的差距威胁着实际的量子密钥分发系统的旁路攻击。许多著名的量子密钥分发协议使用弱相干激光脉冲来编码量子信息。这些源不同于理想的单光子源，并且遵循泊松统计。许多协议，如诱饵状态和符合检测协议，依赖于监测光子统计信息来检测任何信息泄漏。光子统计的准确测量和表征使得能够检测敌意攻击和估计安全密钥率，从而加强了量子密钥分发系统的整体安全性。我们已经严格地描述了我们的源的特征，以估计使用多个探测器的平均光子数，以与使用单个探测器进行的测量进行比较。此外，我们还研究了强度波动，以帮助识别和缓解由于状态准备缺陷而导致的任何潜在信息泄漏。我们的目标是弥合理论和实践之间的差距，实现信息理论安全。



## **34. QEVSEC: Quick Electric Vehicle SEcure Charging via Dynamic Wireless Power Transfer**

QEVSEC：通过动态无线电能传输实现电动汽车快速安全充电 cs.CR

6 pages, conference

**SubmitDate**: 2023-08-28    [abs](http://arxiv.org/abs/2205.10292v3) [paper-pdf](http://arxiv.org/pdf/2205.10292v3)

**Authors**: Tommaso Bianchi, Surudhi Asokraj, Alessandro Brighente, Mauro Conti, Radha Poovendran

**Abstract**: Dynamic Wireless Power Transfer (DWPT) can be used for on-demand recharging of Electric Vehicles (EV) while driving. However, DWPT raises numerous security and privacy concerns. Recently, researchers demonstrated that DWPT systems are vulnerable to adversarial attacks. In an EV charging scenario, an attacker can prevent the authorized customer from charging, obtain a free charge by billing a victim user and track a target vehicle. State-of-the-art authentication schemes relying on centralized solutions are either vulnerable to various attacks or have high computational complexity, making them unsuitable for a dynamic scenario. In this paper, we propose Quick Electric Vehicle SEcure Charging (QEVSEC), a novel, secure, and efficient authentication protocol for the dynamic charging of EVs. Our idea for QEVSEC originates from multiple vulnerabilities we found in the state-of-the-art protocol that allows tracking of user activity and is susceptible to replay attacks. Based on these observations, the proposed protocol solves these issues and achieves lower computational complexity by using only primitive cryptographic operations in a very short message exchange. QEVSEC provides scalability and a reduced cost in each iteration, thus lowering the impact on the power needed from the grid.

摘要: 动态无线电能传输(DWPT)可用于电动汽车(EV)行驶时的按需充电。然而，DWPT带来了许多安全和隐私方面的问题。最近，研究人员证明了DWPT系统容易受到敌意攻击。在电动汽车充电场景中，攻击者可以阻止授权客户充电，通过向受害用户收费来获得免费费用，并跟踪目标车辆。依赖于集中式解决方案的最先进的身份验证方案要么容易受到各种攻击，要么具有很高的计算复杂性，不适合动态场景。本文提出了一种新颖、安全、高效的电动汽车动态充电认证协议--快速电动汽车安全充电协议。我们对QEVSEC的想法源于我们在最先进的协议中发现的多个漏洞，该协议允许跟踪用户活动，并且容易受到重播攻击。基于这些观察，提出的协议解决了这些问题，并通过在很短的消息交换中仅使用原始密码操作来实现较低的计算复杂度。QEVSEC在每次迭代中提供了可扩展性和更低的成本，从而降低了对电网所需电力的影响。



## **35. Hiding Visual Information via Obfuscating Adversarial Perturbations**

通过混淆敌意扰动隐藏视觉信息 cs.CV

**SubmitDate**: 2023-08-28    [abs](http://arxiv.org/abs/2209.15304v4) [paper-pdf](http://arxiv.org/pdf/2209.15304v4)

**Authors**: Zhigang Su, Dawei Zhou, Nannan Wangu, Decheng Li, Zhen Wang, Xinbo Gao

**Abstract**: Growing leakage and misuse of visual information raise security and privacy concerns, which promotes the development of information protection. Existing adversarial perturbations-based methods mainly focus on the de-identification against deep learning models. However, the inherent visual information of the data has not been well protected. In this work, inspired by the Type-I adversarial attack, we propose an adversarial visual information hiding method to protect the visual privacy of data. Specifically, the method generates obfuscating adversarial perturbations to obscure the visual information of the data. Meanwhile, it maintains the hidden objectives to be correctly predicted by models. In addition, our method does not modify the parameters of the applied model, which makes it flexible for different scenarios. Experimental results on the recognition and classification tasks demonstrate that the proposed method can effectively hide visual information and hardly affect the performances of models. The code is available in the supplementary material.

摘要: 日益增长的视觉信息泄露和滥用引发了人们对安全和隐私的担忧，这推动了信息保护的发展。现有的基于对抗性扰动的方法主要集中在针对深度学习模型的去识别。然而，数据固有的视觉信息并没有得到很好的保护。在这项工作中，受Type-I对抗攻击的启发，我们提出了一种对抗性视觉信息隐藏方法来保护数据的视觉隐私。具体地说，该方法产生模糊的对抗性扰动以模糊数据的可视信息。同时，保持模型对隐含目标的正确预测。此外，我们的方法不修改应用模型的参数，这使得它可以灵活地适应不同的场景。在识别和分类任务上的实验结果表明，该方法能够有效地隐藏视觉信息，且几乎不影响模型的性能。该代码可在补充材料中找到。



## **36. Detecting Language Model Attacks with Perplexity**

基于困惑的语言模型攻击检测 cs.CL

**SubmitDate**: 2023-08-27    [abs](http://arxiv.org/abs/2308.14132v1) [paper-pdf](http://arxiv.org/pdf/2308.14132v1)

**Authors**: Gabriel Alon, Michael Kamfonas

**Abstract**: A novel hack involving Large Language Models (LLMs) has emerged, leveraging adversarial suffixes to trick models into generating perilous responses. This method has garnered considerable attention from reputable media outlets such as the New York Times and Wired, thereby influencing public perception regarding the security and safety of LLMs. In this study, we advocate the utilization of perplexity as one of the means to recognize such potential attacks. The underlying concept behind these hacks revolves around appending an unusually constructed string of text to a harmful query that would otherwise be blocked. This maneuver confuses the protective mechanisms and tricks the model into generating a forbidden response. Such scenarios could result in providing detailed instructions to a malicious user for constructing explosives or orchestrating a bank heist. Our investigation demonstrates the feasibility of employing perplexity, a prevalent natural language processing metric, to detect these adversarial tactics before generating a forbidden response. By evaluating the perplexity of queries with and without such adversarial suffixes using an open-source LLM, we discovered that nearly 90 percent were above a perplexity of 1000. This contrast underscores the efficacy of perplexity for detecting this type of exploit.

摘要: 出现了一种涉及大型语言模型(LLM)的新黑客攻击，利用敌意后缀欺骗模型生成危险的响应。这一方法引起了《纽约时报》和《连线》等知名媒体的极大关注，从而影响了公众对低地小武器安全性和安全性的看法。在这项研究中，我们主张使用困惑作为识别这种潜在攻击的手段之一。这些黑客攻击背后的基本概念围绕着在原本会被阻止的有害查询中附加一个构造异常的文本字符串。这种操作混淆了保护机制，并诱使模型产生禁止反应。这种情况可能会导致向恶意用户提供制造炸药或策划银行抢劫的详细说明。我们的研究证明了使用困惑，一种流行的自然语言处理度量，在生成禁止响应之前检测这些敌对策略的可行性。通过使用开源LLM评估带有和不带有这种敌意后缀的查询的困惑程度，我们发现近90%的查询困惑程度高于1000。这种对比突出了困惑在检测这种类型的利用方面的有效性。



## **37. Fairness and Privacy in Voice Biometrics:A Study of Gender Influences Using wav2vec 2.0**

语音生物识别中的公平性和隐私性：基于Wav2vec 2.0的性别影响研究 eess.AS

7 pages

**SubmitDate**: 2023-08-27    [abs](http://arxiv.org/abs/2308.14049v1) [paper-pdf](http://arxiv.org/pdf/2308.14049v1)

**Authors**: Oubaida Chouchane, Michele Panariello, Chiara Galdi, Massimiliano Todisco, Nicholas Evans

**Abstract**: This study investigates the impact of gender information on utility, privacy, and fairness in voice biometric systems, guided by the General Data Protection Regulation (GDPR) mandates, which underscore the need for minimizing the processing and storage of private and sensitive data, and ensuring fairness in automated decision-making systems. We adopt an approach that involves the fine-tuning of the wav2vec 2.0 model for speaker verification tasks, evaluating potential gender-related privacy vulnerabilities in the process. Gender influences during the fine-tuning process were employed to enhance fairness and privacy in order to emphasise or obscure gender information within the speakers' embeddings. Results from VoxCeleb datasets indicate our adversarial model increases privacy against uninformed attacks, yet slightly diminishes speaker verification performance compared to the non-adversarial model. However, the model's efficacy reduces against informed attacks. Analysis of system performance was conducted to identify potential gender biases, thus highlighting the need for further research to understand and improve the delicate interplay between utility, privacy, and equity in voice biometric systems.

摘要: 本研究在一般数据保护法规(GDPR)的指导下，调查了性别信息对语音生物识别系统中的效用、隐私和公平性的影响，该法规强调了将私人和敏感数据的处理和存储降至最低的必要性，并确保自动化决策系统中的公平性。我们采用了一种方法，涉及对说话人验证任务的Wav2vec 2.0模型进行微调，评估过程中潜在的与性别相关的隐私漏洞。在微调过程中采用了性别影响，以加强公平和隐私，以便在发言者的嵌入中强调或模糊性别信息。来自VoxCeleb数据集的结果表明，我们的对抗性模型提高了针对不知情攻击的隐私，但与非对抗性模型相比，说话人验证性能略有下降。然而，该模型对知情攻击的有效性会降低。对系统性能进行了分析，以确定潜在的性别偏见，从而突出了进一步研究的必要性，以了解和改进语音生物识别系统中效用、隐私和公平之间的微妙相互作用。



## **38. Device-Independent Quantum Key Distribution Based on the Mermin-Peres Magic Square Game**

基于Mermin-Peres魔方博弈的设备无关量子密钥分配 quant-ph

**SubmitDate**: 2023-08-27    [abs](http://arxiv.org/abs/2308.14037v1) [paper-pdf](http://arxiv.org/pdf/2308.14037v1)

**Authors**: Yi-Zheng Zhen, Yingqiu Mao, Yu-Zhe Zhang, Feihu Xu, Barry C. Sanders

**Abstract**: Device-independent quantum key distribution (DIQKD) is information-theoretically secure against adversaries who possess a scalable quantum computer and who have supplied malicious key-establishment systems; however, the DIQKD key rate is currently too low. Consequently, we devise a DIQKD scheme based on the quantum nonlocal Mermin-Peres magic square game: our scheme asymptotically delivers DIQKD against collective attacks, even with noise. Our scheme outperforms DIQKD using the Clauser-Horne-Shimony-Holt game with respect to the number of game rounds, albeit not number of entangled pairs, provided that both state visibility and detection efficiency are high enough.

摘要: 独立于设备的量子密钥分发(DIQKD)在信息理论上是安全的，可以抵御拥有可扩展量子计算机并提供恶意密钥建立系统的攻击者；然而，DIQKD密钥率目前太低。因此，我们设计了一个基于量子非局部Mermin-Peres幻方博弈的DIQKD方案：即使在有噪声的情况下，我们的方案也能渐近地提供抗集体攻击的DIQKD。在状态可见性和检测效率都足够高的情况下，我们的方案在游戏轮数上优于使用Clauser-Horne-Shimony-Holt博弈的DIQKD，尽管不是纠缠对的数量。



## **39. A semantic backdoor attack against Graph Convolutional Networks**

一种针对图卷积网络的语义后门攻击 cs.LG

**SubmitDate**: 2023-08-26    [abs](http://arxiv.org/abs/2302.14353v4) [paper-pdf](http://arxiv.org/pdf/2302.14353v4)

**Authors**: Jiazhu Dai, Zhipeng Xiong

**Abstract**: Graph convolutional networks (GCNs) have been very effective in addressing the issue of various graph-structured related tasks. However, recent research has shown that GCNs are vulnerable to a new type of threat called a backdoor attack, where the adversary can inject a hidden backdoor into GCNs so that the attacked model performs well on benign samples, but its prediction will be maliciously changed to the attacker-specified target label if the hidden backdoor is activated by the attacker-defined trigger. A semantic backdoor attack is a new type of backdoor attack on deep neural networks (DNNs), where a naturally occurring semantic feature of samples can serve as a backdoor trigger such that the infected DNN models will misclassify testing samples containing the predefined semantic feature even without the requirement of modifying the testing samples. Since the backdoor trigger is a naturally occurring semantic feature of the samples, semantic backdoor attacks are more imperceptible and pose a new and serious threat. In this paper, we investigate whether such semantic backdoor attacks are possible for GCNs and propose a semantic backdoor attack against GCNs (SBAG) under the context of graph classification to reveal the existence of this security vulnerability in GCNs. SBAG uses a certain type of node in the samples as a backdoor trigger and injects a hidden backdoor into GCN models by poisoning training data. The backdoor will be activated, and the GCN models will give malicious classification results specified by the attacker even on unmodified samples as long as the samples contain enough trigger nodes. We evaluate SBAG on four graph datasets and the experimental results indicate that SBAG is effective.

摘要: 图卷积网络(GCNS)在解决各种与图结构相关的任务问题方面已经非常有效。然而，最近的研究表明，GCNS容易受到一种名为后门攻击的新型威胁的攻击，在这种威胁中，攻击者可以向GCNS注入隐藏的后门，以便攻击模型在良性样本上执行良好，但如果隐藏的后门被攻击者定义的触发器激活，其预测将被恶意更改为攻击者指定的目标标签。语义后门攻击是对深度神经网络(DNN)的一种新型后门攻击，其中样本的自然产生的语义特征可以作为后门触发器，使得被感染的DNN模型即使在不需要修改测试样本的情况下也会对包含预定义语义特征的测试样本进行误分类。由于后门触发器是样本中自然产生的语义特征，语义后门攻击更难以察觉，并构成新的严重威胁。本文研究了GCNS是否存在这样的语义后门攻击，并在图分类的背景下提出了一种针对GCNS的语义后门攻击(SBAG)，以揭示GCNS中这一安全漏洞的存在。SBag使用样本中的某一类型节点作为后门触发器，并通过毒化训练数据向GCN模型注入隐藏的后门。后门将被激活，GCN模型将给出攻击者指定的恶意分类结果，即使是在未经修改的样本上，只要这些样本包含足够的触发节点。我们在四个图数据集上对SAGAG进行了评估，实验结果表明SABAG是有效的。



## **40. Active learning for fast and slow modeling attacks on Arbiter PUFs**

对仲裁器PUF进行快慢建模攻击的主动学习 cs.CR

**SubmitDate**: 2023-08-25    [abs](http://arxiv.org/abs/2308.13645v1) [paper-pdf](http://arxiv.org/pdf/2308.13645v1)

**Authors**: Vincent Dumoulin, Wenjing Rao, Natasha Devroye

**Abstract**: Modeling attacks, in which an adversary uses machine learning techniques to model a hardware-based Physically Unclonable Function (PUF) pose a great threat to the viability of these hardware security primitives. In most modeling attacks, a random subset of challenge-response-pairs (CRPs) are used as the labeled data for the machine learning algorithm. Here, for the arbiter-PUF, a delay based PUF which may be viewed as a linear threshold function with random weights (due to manufacturing imperfections), we investigate the role of active learning in Support Vector Machine (SVM) learning. We focus on challenge selection to help SVM algorithm learn ``fast'' and learn ``slow''. Our methods construct challenges rather than relying on a sample pool of challenges as in prior work. Using active learning to learn ``fast'' (less CRPs revealed, higher accuracies) may help manufacturers learn the manufactured PUFs more efficiently, or may form a more powerful attack when the attacker may query the PUF for CRPs at will. Using active learning to select challenges from which learning is ``slow'' (low accuracy despite a large number of revealed CRPs) may provide a basis for slowing down attackers who are limited to overhearing CRPs.

摘要: 建模攻击是指攻击者使用机器学习技术对基于硬件的物理不可克隆函数(PUF)进行建模，这对这些硬件安全原语的生存能力构成了极大的威胁。在大多数建模攻击中，挑战-响应对(CRP)的随机子集被用作机器学习算法的标签数据。这里，对于仲裁器-PUF，一种基于延迟的PUF，可以被视为具有随机权值的线性阈值函数(由于制造缺陷)，我们研究了主动学习在支持向量机学习中的作用。我们把重点放在挑战选择上，帮助支持向量机算法学习“快”和“慢”。我们的方法构建挑战，而不是像以前的工作那样依赖于挑战的样本池。使用主动学习来学习“快速”(揭示的CRP越少，准确率越高)可以帮助制造商更有效地学习制造的PUF，或者当攻击者可以随意向PUF查询CRP时，可能形成更强大的攻击。使用主动学习来选择学习“缓慢”的挑战(尽管发现了大量CRP，但准确率很低)可能会为减缓仅限于无意中监听CRP的攻击者提供基础。



## **41. Unveiling the Role of Message Passing in Dual-Privacy Preservation on GNNs**

揭示消息传递在GNN双重隐私保护中的作用 cs.LG

CIKM 2023

**SubmitDate**: 2023-08-25    [abs](http://arxiv.org/abs/2308.13513v1) [paper-pdf](http://arxiv.org/pdf/2308.13513v1)

**Authors**: Tianyi Zhao, Hui Hu, Lu Cheng

**Abstract**: Graph Neural Networks (GNNs) are powerful tools for learning representations on graphs, such as social networks. However, their vulnerability to privacy inference attacks restricts their practicality, especially in high-stake domains. To address this issue, privacy-preserving GNNs have been proposed, focusing on preserving node and/or link privacy. This work takes a step back and investigates how GNNs contribute to privacy leakage. Through theoretical analysis and simulations, we identify message passing under structural bias as the core component that allows GNNs to \textit{propagate} and \textit{amplify} privacy leakage. Building upon these findings, we propose a principled privacy-preserving GNN framework that effectively safeguards both node and link privacy, referred to as dual-privacy preservation. The framework comprises three major modules: a Sensitive Information Obfuscation Module that removes sensitive information from node embeddings, a Dynamic Structure Debiasing Module that dynamically corrects the structural bias, and an Adversarial Learning Module that optimizes the privacy-utility trade-off. Experimental results on four benchmark datasets validate the effectiveness of the proposed model in protecting both node and link privacy while preserving high utility for downstream tasks, such as node classification.

摘要: 图神经网络(GNN)是学习图上表示的强大工具，如社会网络。然而，它们对隐私推理攻击的脆弱性限制了它们的实用性，特别是在高风险领域。为了解决这一问题，人们提出了隐私保护的GNN，其重点是保护节点和/或链路的隐私。这项工作退了一步，调查了GNN是如何导致隐私泄露的。通过理论分析和仿真，我们发现结构偏差下的消息传递是允许GNN传播和放大隐私泄漏的核心组件。基于这些发现，我们提出了一个原则性的隐私保护GNN框架，该框架有效地保护了节点和链路的隐私，称为双重隐私保护。该框架包括三个主要模块：从节点嵌入中移除敏感信息的敏感信息混淆模块、动态纠正结构偏差的动态结构去偏模块和优化隐私-效用权衡的对抗性学习模块。在四个基准数据集上的实验结果验证了该模型的有效性，在保护节点和链路隐私的同时，保持了节点分类等下游任务的高效用。



## **42. Overcoming Adversarial Attacks for Human-in-the-Loop Applications**

克服针对人在环中应用的敌意攻击 cs.LG

New Frontiers in Adversarial Machine Learning, ICML 2022

**SubmitDate**: 2023-08-25    [abs](http://arxiv.org/abs/2306.05952v2) [paper-pdf](http://arxiv.org/pdf/2306.05952v2)

**Authors**: Ryan McCoppin, Marla Kennedy, Platon Lukyanenko, Sean Kennedy

**Abstract**: Including human analysis has the potential to positively affect the robustness of Deep Neural Networks and is relatively unexplored in the Adversarial Machine Learning literature. Neural network visual explanation maps have been shown to be prone to adversarial attacks. Further research is needed in order to select robust visualizations of explanations for the image analyst to evaluate a given model. These factors greatly impact Human-In-The-Loop (HITL) evaluation tools due to their reliance on adversarial images, including explanation maps and measurements of robustness. We believe models of human visual attention may improve interpretability and robustness of human-machine imagery analysis systems. Our challenge remains, how can HITL evaluation be robust in this adversarial landscape?

摘要: 包括人类分析有可能对深度神经网络的稳健性产生积极影响，在对抗性机器学习文献中相对未被探索。神经网络视觉解释地图已被证明容易受到敌意攻击。还需要进一步的研究，以便为图像分析员选择稳健的解释可视化来评估给定的模型。这些因素极大地影响了人在环(HITL)评估工具，因为它们依赖于敌方图像，包括解释地图和健壮性测量。我们相信人类视觉注意模型可以提高人机图像分析系统的可解释性和稳健性。我们的挑战仍然是，如何在这种对抗性的环境中进行HITL评估？



## **43. Defensive Few-shot Learning**

防御性少投篮学习 cs.CV

Accepted to IEEE Transactions on Pattern Analysis and Machine  Intelligence (TPAMI) 2022

**SubmitDate**: 2023-08-25    [abs](http://arxiv.org/abs/1911.06968v2) [paper-pdf](http://arxiv.org/pdf/1911.06968v2)

**Authors**: Wenbin Li, Lei Wang, Xingxing Zhang, Lei Qi, Jing Huo, Yang Gao, Jiebo Luo

**Abstract**: This paper investigates a new challenging problem called defensive few-shot learning in order to learn a robust few-shot model against adversarial attacks. Simply applying the existing adversarial defense methods to few-shot learning cannot effectively solve this problem. This is because the commonly assumed sample-level distribution consistency between the training and test sets can no longer be met in the few-shot setting. To address this situation, we develop a general defensive few-shot learning (DFSL) framework to answer the following two key questions: (1) how to transfer adversarial defense knowledge from one sample distribution to another? (2) how to narrow the distribution gap between clean and adversarial examples under the few-shot setting? To answer the first question, we propose an episode-based adversarial training mechanism by assuming a task-level distribution consistency to better transfer the adversarial defense knowledge. As for the second question, within each few-shot task, we design two kinds of distribution consistency criteria to narrow the distribution gap between clean and adversarial examples from the feature-wise and prediction-wise perspectives, respectively. Extensive experiments demonstrate that the proposed framework can effectively make the existing few-shot models robust against adversarial attacks. Code is available at https://github.com/WenbinLee/DefensiveFSL.git.

摘要: 本文研究了一个新的具有挑战性的问题--防御性少发学习问题，目的是学习一种对敌方攻击具有鲁棒性的少发学习模型。简单地将现有的对抗性防御方法应用于少射击学习并不能有效地解决这一问题。这是因为通常假设的训练集和测试集之间的样本水平分布一致性在少镜头设置中不再能满足。针对这种情况，我们开发了一个通用防御少发学习(DFSL)框架来回答以下两个关键问题：(1)如何将对抗性防御知识从一个样本分布转移到另一个样本分布？(2)在少发情况下，如何缩小干净样本和对抗性样本之间的分布差距？为了回答第一个问题，我们提出了一种基于情节的对抗性训练机制，通过假设任务级别的分布一致性来更好地传递对抗性防御知识。对于第二个问题，在每个少镜头任务中，我们设计了两种分布一致性准则，分别从特征和预测的角度缩小了正例和对手例之间的分布差距。大量实验表明，该框架能有效地使已有的少镜头模型具有较强的抗敌意攻击能力。代码可在https://github.com/WenbinLee/DefensiveFSL.git.上找到



## **44. Feature Unlearning for Pre-trained GANs and VAEs**

针对经过预先培训的GAN和VAE的功能遗忘 cs.CV

**SubmitDate**: 2023-08-25    [abs](http://arxiv.org/abs/2303.05699v3) [paper-pdf](http://arxiv.org/pdf/2303.05699v3)

**Authors**: Saemi Moon, Seunghyuk Cho, Dongwoo Kim

**Abstract**: We tackle the problem of feature unlearning from a pre-trained image generative model: GANs and VAEs. Unlike a common unlearning task where an unlearning target is a subset of the training set, we aim to unlearn a specific feature, such as hairstyle from facial images, from the pre-trained generative models. As the target feature is only presented in a local region of an image, unlearning the entire image from the pre-trained model may result in losing other details in the remaining region of the image. To specify which features to unlearn, we collect randomly generated images that contain the target features. We then identify a latent representation corresponding to the target feature and then use the representation to fine-tune the pre-trained model. Through experiments on MNIST and CelebA datasets, we show that target features are successfully removed while keeping the fidelity of the original models. Further experiments with an adversarial attack show that the unlearned model is more robust under the presence of malicious parties.

摘要: 我们从一个预先训练的图像生成模型GANS和VAE中解决了特征遗忘的问题。与通常的遗忘任务不同，忘记目标是训练集的一个子集，我们的目标是从预先训练的生成模型中忘记特定的特征，如面部图像中的发型。由于目标特征仅呈现在图像的局部区域中，因此从预先训练的模型中不学习整个图像可能导致丢失图像剩余区域中的其他细节。为了指定要取消学习的特征，我们收集包含目标特征的随机生成的图像。然后，我们识别对应于目标特征的潜在表示，然后使用该表示来微调预先训练的模型。通过在MNIST和CelebA数据集上的实验，我们证明了在保持原始模型保真度的情况下，目标特征被成功去除。进一步的对抗性攻击实验表明，未学习模型在恶意方存在的情况下具有更强的鲁棒性。



## **45. Face Encryption via Frequency-Restricted Identity-Agnostic Attacks**

通过频率受限的身份不可知攻击进行人脸加密 cs.CV

I noticed something missing in the article's description in  subsection 3.2, so I'd like to undo it and re-finalize and describe it

**SubmitDate**: 2023-08-25    [abs](http://arxiv.org/abs/2308.05983v3) [paper-pdf](http://arxiv.org/pdf/2308.05983v3)

**Authors**: Xin Dong, Rui Wang, Siyuan Liang, Aishan Liu, Lihua Jing

**Abstract**: Billions of people are sharing their daily live images on social media everyday. However, malicious collectors use deep face recognition systems to easily steal their biometric information (e.g., faces) from these images. Some studies are being conducted to generate encrypted face photos using adversarial attacks by introducing imperceptible perturbations to reduce face information leakage. However, existing studies need stronger black-box scenario feasibility and more natural visual appearances, which challenge the feasibility of privacy protection. To address these problems, we propose a frequency-restricted identity-agnostic (FRIA) framework to encrypt face images from unauthorized face recognition without access to personal information. As for the weak black-box scenario feasibility, we obverse that representations of the average feature in multiple face recognition models are similar, thus we propose to utilize the average feature via the crawled dataset from the Internet as the target to guide the generation, which is also agnostic to identities of unknown face recognition systems; in nature, the low-frequency perturbations are more visually perceptible by the human vision system. Inspired by this, we restrict the perturbation in the low-frequency facial regions by discrete cosine transform to achieve the visual naturalness guarantee. Extensive experiments on several face recognition models demonstrate that our FRIA outperforms other state-of-the-art methods in generating more natural encrypted faces while attaining high black-box attack success rates of 96%. In addition, we validate the efficacy of FRIA using real-world black-box commercial API, which reveals the potential of FRIA in practice. Our codes can be found in https://github.com/XinDong10/FRIA.

摘要: 每天都有数十亿人在社交媒体上分享他们的日常直播图片。然而，恶意收集者使用深度人脸识别系统来轻松地从这些图像中窃取他们的生物特征信息(例如，人脸)。正在进行一些研究，通过引入不可察觉的扰动来减少面部信息的泄露，从而使用对抗性攻击来生成加密的面部照片。然而，现有的研究需要更强的黑盒场景可行性和更自然的视觉外观，这对隐私保护的可行性提出了挑战。为了解决这些问题，我们提出了一种频率受限身份不可知(FRIA)框架来加密来自未经授权的人脸识别的人脸图像，而不需要访问个人信息。对于弱黑盒场景的可行性，我们发现在多个人脸识别模型中平均特征的表示是相似的，因此我们提出利用从互联网上抓取的数据集的平均特征作为目标来指导生成，这也与未知人脸识别系统的身份无关；实际上，低频扰动更容易被人类视觉系统感知。受此启发，我们通过离散余弦变换来限制人脸低频区域的扰动，以达到视觉自然度的保证。在几个人脸识别模型上的广泛实验表明，我们的FRIA在生成更自然的加密人脸方面优于其他最先进的方法，同时获得了96%的高黑盒攻击成功率。此外，我们使用真实的黑盒商业API验证了FRIA的有效性，这揭示了FRIA在实践中的潜力。我们的代码可以在https://github.com/XinDong10/FRIA.中找到



## **46. Evaluating the Vulnerabilities in ML systems in terms of adversarial attacks**

从对抗性攻击的角度评估ML系统的脆弱性 cs.LG

**SubmitDate**: 2023-08-24    [abs](http://arxiv.org/abs/2308.12918v1) [paper-pdf](http://arxiv.org/pdf/2308.12918v1)

**Authors**: John Harshith, Mantej Singh Gill, Madhan Jothimani

**Abstract**: There have been recent adversarial attacks that are difficult to find. These new adversarial attacks methods may pose challenges to current deep learning cyber defense systems and could influence the future defense of cyberattacks. The authors focus on this domain in this research paper. They explore the consequences of vulnerabilities in AI systems. This includes discussing how they might arise, differences between randomized and adversarial examples and also potential ethical implications of vulnerabilities. Moreover, it is important to train the AI systems appropriately when they are in testing phase and getting them ready for broader use.

摘要: 最近发生了一些很难找到的对抗性攻击。这些新的对抗性攻击方法可能会对当前的深度学习网络防御系统构成挑战，并可能影响未来对网络攻击的防御。在这篇研究论文中，作者主要关注这一领域。他们探索了人工智能系统中漏洞的后果。这包括讨论它们可能出现的原因、随机例子和对抗性例子之间的差异，以及漏洞的潜在伦理影响。此外，重要的是在AI系统处于测试阶段时对其进行适当的培训，并使其为更广泛的使用做好准备。



## **47. Near Optimal Adversarial Attack on UCB Bandits**

对UCB土匪的近最优敌意攻击 cs.LG

Appeared at ICML 2023 AdvML Workshop

**SubmitDate**: 2023-08-24    [abs](http://arxiv.org/abs/2008.09312v6) [paper-pdf](http://arxiv.org/pdf/2008.09312v6)

**Authors**: Shiliang Zuo

**Abstract**: I study a stochastic multi-arm bandit problem where rewards are subject to adversarial corruption. I propose a novel attack strategy that manipulates a learner employing the UCB algorithm into pulling some non-optimal target arm $T - o(T)$ times with a cumulative cost that scales as $\widehat{O}(\sqrt{\log T})$, where $T$ is the number of rounds. I also prove the first lower bound on the cumulative attack cost. The lower bound matches the upper bound up to $O(\log \log T)$ factors, showing the proposed attack strategy to be near optimal.

摘要: 我研究了一个随机多臂强盗问题，其中报酬受到对抗性腐败的影响。提出了一种新的攻击策略，该策略利用UCB算法操纵学习者拉出一些非最优目标臂$T-o(T)$次，累积代价为$\widehat{O}(\Sqrt{\log T})$，其中$T$是轮数。我还证明了累积攻击成本的第一个下限。下界与最高可达$O(\LOG\LOG T)$因子的上界匹配，表明所提出的攻击策略接近最优。



## **48. Fast Adversarial Training with Smooth Convergence**

平滑收敛的快速对抗性训练 cs.LG

**SubmitDate**: 2023-08-24    [abs](http://arxiv.org/abs/2308.12857v1) [paper-pdf](http://arxiv.org/pdf/2308.12857v1)

**Authors**: Mengnan Zhao, Lihe Zhang, Yuqiu Kong, Baocai Yin

**Abstract**: Fast adversarial training (FAT) is beneficial for improving the adversarial robustness of neural networks. However, previous FAT work has encountered a significant issue known as catastrophic overfitting when dealing with large perturbation budgets, \ie the adversarial robustness of models declines to near zero during training.   To address this, we analyze the training process of prior FAT work and observe that catastrophic overfitting is accompanied by the appearance of loss convergence outliers.   Therefore, we argue a moderately smooth loss convergence process will be a stable FAT process that solves catastrophic overfitting.   To obtain a smooth loss convergence process, we propose a novel oscillatory constraint (dubbed ConvergeSmooth) to limit the loss difference between adjacent epochs. The convergence stride of ConvergeSmooth is introduced to balance convergence and smoothing. Likewise, we design weight centralization without introducing additional hyperparameters other than the loss balance coefficient.   Our proposed methods are attack-agnostic and thus can improve the training stability of various FAT techniques.   Extensive experiments on popular datasets show that the proposed methods efficiently avoid catastrophic overfitting and outperform all previous FAT methods. Code is available at \url{https://github.com/FAT-CS/ConvergeSmooth}.

摘要: 快速对抗训练(FAT)有利于提高神经网络的对抗健壮性。然而，以前的FAT工作在处理大扰动预算时遇到了一个重要的问题，即模型的对抗性健壮性在训练期间下降到接近于零。为了解决这个问题，我们分析了以前FAT工作的训练过程，观察到灾难性的过拟合伴随着损失收敛离群点的出现。因此，我们认为，适度平滑的损失收敛过程将是一个稳定的脂肪过程，可以解决灾难性的过拟合问题。为了获得平滑的损耗收敛过程，我们提出了一种新的振荡约束(称为收敛平滑)来限制相邻历元之间的损耗差异。为了平衡收敛和光顺，引入了收敛步长。同样，我们设计了权重集中，而不引入除损失平衡系数之外的额外超参数。我们提出的方法是攻击不可知的，因此可以提高各种FAT技术的训练稳定性。在流行的数据集上的大量实验表明，所提出的方法有效地避免了灾难性的过拟合，并且优于所有已有的FAT方法。代码可在\url{https://github.com/FAT-CS/ConvergeSmooth}.上找到



## **49. Unifying Gradients to Improve Real-world Robustness for Deep Networks**

统一梯度以提高深度网络的真实稳健性 stat.ML

**SubmitDate**: 2023-08-24    [abs](http://arxiv.org/abs/2208.06228v2) [paper-pdf](http://arxiv.org/pdf/2208.06228v2)

**Authors**: Yingwen Wu, Sizhe Chen, Kun Fang, Xiaolin Huang

**Abstract**: The wide application of deep neural networks (DNNs) demands an increasing amount of attention to their real-world robustness, i.e., whether a DNN resists black-box adversarial attacks, among which score-based query attacks (SQAs) are most threatening since they can effectively hurt a victim network with the only access to model outputs. Defending against SQAs requires a slight but artful variation of outputs due to the service purpose for users, who share the same output information with SQAs. In this paper, we propose a real-world defense by Unifying Gradients (UniG) of different data so that SQAs could only probe a much weaker attack direction that is similar for different samples. Since such universal attack perturbations have been validated as less aggressive than the input-specific perturbations, UniG protects real-world DNNs by indicating attackers a twisted and less informative attack direction. We implement UniG efficiently by a Hadamard product module which is plug-and-play. According to extensive experiments on 5 SQAs, 2 adaptive attacks and 7 defense baselines, UniG significantly improves real-world robustness without hurting clean accuracy on CIFAR10 and ImageNet. For instance, UniG maintains a model of 77.80% accuracy under 2500-query Square attack while the state-of-the-art adversarially-trained model only has 67.34% on CIFAR10. Simultaneously, UniG outperforms all compared baselines in terms of clean accuracy and achieves the smallest modification of the model output. The code is released at https://github.com/snowien/UniG-pytorch.

摘要: 深度神经网络(DNN)的广泛应用要求人们越来越多地关注它在现实世界中的健壮性，即DNN是否能够抵抗黑盒对抗攻击，其中基于分数的查询攻击(SQA)是最具威胁性的，因为它们可以有效地伤害只访问模型输出的受害者网络。由于用户的服务目的，防御SBA需要稍微但巧妙地改变输出，因为用户与SBA共享相同的输出信息。在本文中，我们提出了一种真实世界的防御方法，通过统一不同数据的梯度(Ung)，使得SQAS只能探测对不同样本相似的弱得多的攻击方向。由于这种通用的攻击扰动已被验证为不如特定于输入的扰动那么具侵略性，unig通过向攻击者指示一个扭曲的、信息量较少的攻击方向来保护现实世界的DNN。我们通过一个即插即用的Hadamard产品模块高效地实现了unig。根据对5个SQA、2个自适应攻击和7个防御基线的广泛实验，unig在不损害CIFAR10和ImageNet上的干净准确性的情况下，显著提高了真实世界的健壮性。例如，在2500-Query Square攻击下，unig保持了77.80%的准确率，而最新的对抗性训练模型在CIFAR10上只有67.34%的准确率。同时，UNIG在清洁精度方面优于所有比较的基线，并实现了对模型输出的最小修改。该代码在https://github.com/snowien/UniG-pytorch.上发布



## **50. Universal Soldier: Using Universal Adversarial Perturbations for Detecting Backdoor Attacks**

Universal Soldier：使用通用对抗性扰动检测后门攻击 cs.LG

**SubmitDate**: 2023-08-24    [abs](http://arxiv.org/abs/2302.00747v3) [paper-pdf](http://arxiv.org/pdf/2302.00747v3)

**Authors**: Xiaoyun Xu, Oguzhan Ersoy, Stjepan Picek

**Abstract**: Deep learning models achieve excellent performance in numerous machine learning tasks. Yet, they suffer from security-related issues such as adversarial examples and poisoning (backdoor) attacks. A deep learning model may be poisoned by training with backdoored data or by modifying inner network parameters. Then, a backdoored model performs as expected when receiving a clean input, but it misclassifies when receiving a backdoored input stamped with a pre-designed pattern called "trigger". Unfortunately, it is difficult to distinguish between clean and backdoored models without prior knowledge of the trigger. This paper proposes a backdoor detection method by utilizing a special type of adversarial attack, universal adversarial perturbation (UAP), and its similarities with a backdoor trigger. We observe an intuitive phenomenon: UAPs generated from backdoored models need fewer perturbations to mislead the model than UAPs from clean models. UAPs of backdoored models tend to exploit the shortcut from all classes to the target class, built by the backdoor trigger. We propose a novel method called Universal Soldier for Backdoor detection (USB) and reverse engineering potential backdoor triggers via UAPs. Experiments on 345 models trained on several datasets show that USB effectively detects the injected backdoor and provides comparable or better results than state-of-the-art methods.

摘要: 深度学习模型在众多的机器学习任务中取得了优异的性能。然而，他们面临着与安全相关的问题，如对抗性例子和中毒(后门)攻击。深度学习模型可能会因使用后备数据进行训练或通过修改内部网络参数而中毒。然后，当接收到干净的输入时，回溯模型执行预期的操作，但是当接收到带有预先设计的称为“Trigger”模式的回溯输入时，它就会错误地分类。不幸的是，在没有事先了解触发因素的情况下，很难区分干净和落后的模型。利用一种特殊类型的对抗性攻击--通用对抗性扰动(UAP)及其与后门触发器的相似性，提出了一种后门检测方法。我们观察到一个直观的现象：由回溯模型生成的UAP比从干净模型生成的UAP需要更少的扰动来误导模型。后门模型的UAP倾向于利用由后门触发器构建的从所有类到目标类的快捷方式。我们提出了一种新的方法，称为通用士兵(Universal Soldier)，用于后门检测(USB)和通过UAP反向工程潜在的后门触发器。在几个数据集上训练的345个模型上的实验表明，USB有效地检测注入的后门，并提供与最先进的方法相当或更好的结果。



