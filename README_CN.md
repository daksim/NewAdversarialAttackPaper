# Latest Adversarial Attack Papers
**update at 2024-12-09 10:58:03**

翻译来自 https://cloud.tencent.com/document/product/551/15619

## **1. LIAR: Leveraging Alignment (Best-of-N) to Jailbreak LLMs in Seconds**

LIAR：利用联盟（N中最佳）在几秒钟内越狱LLM cs.CL

**SubmitDate**: 2024-12-06    [abs](http://arxiv.org/abs/2412.05232v1) [paper-pdf](http://arxiv.org/pdf/2412.05232v1)

**Authors**: James Beetham, Souradip Chakraborty, Mengdi Wang, Furong Huang, Amrit Singh Bedi, Mubarak Shah

**Abstract**: Many existing jailbreak techniques rely on solving discrete combinatorial optimization, while more recent approaches involve training LLMs to generate multiple adversarial prompts. However, both approaches require significant computational resources to produce even a single adversarial prompt. We hypothesize that the inefficiency of current approaches stems from an inadequate characterization of the jailbreak problem. To address this gap, we formulate the jailbreak problem in terms of alignment. By starting from an available safety-aligned model, we leverage an unsafe reward to guide the safe model towards generating unsafe outputs using alignment techniques (e.g., reinforcement learning from human feedback), effectively performing jailbreaking via alignment. We propose a novel jailbreak method called LIAR (LeveragIng Alignment to jailbReak). To demonstrate the simplicity and effectiveness of our approach, we employ a best-of-N method to solve the alignment problem. LIAR offers significant advantages: lower computational requirements without additional training, fully black-box operation, competitive attack success rates, and more human-readable prompts. We provide theoretical insights into the possibility of jailbreaking a safety-aligned model, revealing inherent vulnerabilities in current alignment strategies for LLMs. We also provide sub-optimality guarantees for the proposed \algo. Experimentally, we achieve ASR comparable to the SoTA with a 10x improvement to perplexity and a Time-to-Attack measured in seconds rather than tens of hours.

摘要: 许多现有的越狱技术依赖于求解离散组合优化，而更新的方法涉及训练LLM来生成多个对抗性提示。然而，这两种方法都需要大量的计算资源才能产生哪怕一个对抗性提示。我们假设，当前方法的低效源于对越狱问题的不充分描述。为了解决这一差距，我们从对齐的角度来阐述越狱问题。通过从可用的安全对齐模型开始，我们利用不安全奖励来引导安全模型使用对齐技术(例如，从人类反馈中的强化学习)来生成不安全的输出，通过对齐有效地执行越狱。我们提出了一种新的越狱方法，称为LIAR(利用对齐越狱)。为了证明我们的方法的简单性和有效性，我们使用了N中最优方法来解决比对问题。Liar提供了显著的优势：更低的计算要求，无需额外的培训，完全黑箱操作，具有竞争力的攻击成功率，以及更易读的提示。我们提供了对越狱的可能性的理论见解，安全对齐的模型，揭示了当前的低密度脂蛋白对齐策略的固有弱点。我们还为所提出的算法提供了次最优性保证。在实验中，我们获得了与SOTA相当的ASR，困惑程度提高了10倍，攻击时间以秒衡量，而不是几十小时。



## **2. A Practical Examination of AI-Generated Text Detectors for Large Language Models**

大型语言模型的人工智能生成文本检测器的实践检验 cs.CL

8 pages. Submitted to ARR October cycle

**SubmitDate**: 2024-12-06    [abs](http://arxiv.org/abs/2412.05139v1) [paper-pdf](http://arxiv.org/pdf/2412.05139v1)

**Authors**: Brian Tufts, Xuandong Zhao, Lei Li

**Abstract**: The proliferation of large language models has raised growing concerns about their misuse, particularly in cases where AI-generated text is falsely attributed to human authors. Machine-generated content detectors claim to effectively identify such text under various conditions and from any language model. This paper critically evaluates these claims by assessing several popular detectors (RADAR, Wild, T5Sentinel, Fast-DetectGPT, GPTID, LogRank, Binoculars) on a range of domains, datasets, and models that these detectors have not previously encountered. We employ various prompting strategies to simulate adversarial attacks, demonstrating that even moderate efforts can significantly evade detection. We emphasize the importance of the true positive rate at a specific false positive rate (TPR@FPR) metric and demonstrate that these detectors perform poorly in certain settings, with TPR@.01 as low as 0\%. Our findings suggest that both trained and zero-shot detectors struggle to maintain high sensitivity while achieving a reasonable true positive rate.

摘要: 大型语言模型的激增引发了人们对它们滥用的日益担忧，特别是在人工智能生成的文本被错误地归因于人类作者的情况下。机器生成的内容检测器声称可以在各种条件下从任何语言模型有效地识别此类文本。本文通过评估几种流行的探测器(雷达、Wild、T5Sentinel、Fast-DetectGPT、GPTID、logrank、双筒望远镜)在这些探测器以前从未遇到的一系列域、数据集和模型上对这些声称进行了批判性评估。我们使用各种提示策略来模拟对抗性攻击，表明即使是适度的攻击也可以显著地躲避检测。我们强调了在特定的假阳性率(TPR@fPR)度量下真阳性率的重要性，并证明了这些检测器在某些设置下的性能很差，TPR@0.01低至0\%。我们的发现表明，训练有素的探测器和零射探测器都很难在保持高灵敏度的同时获得合理的真阳性率。



## **3. On Borrowed Time -- Preventing Static Side-Channel Analysis**

论借来的时间--防止静态侧通道分析 cs.CR

**SubmitDate**: 2024-12-06    [abs](http://arxiv.org/abs/2307.09001v2) [paper-pdf](http://arxiv.org/pdf/2307.09001v2)

**Authors**: Robert Dumitru, Thorben Moos, Andrew Wabnitz, Yuval Yarom

**Abstract**: In recent years a new class of side-channel attacks has emerged. Instead of targeting device emissions during dynamic computation, adversaries now frequently exploit the leakage or response behaviour of integrated circuits in a static state. Members of this class include Static Power Side-Channel Analysis (SCA), Laser Logic State Imaging (LLSI) and Impedance Analysis (IA). Despite relying on different physical phenomena, they all enable the extraction of sensitive information from circuits in a static state with high accuracy and low noise -- a trait that poses a significant threat to many established side-channel countermeasures.   In this work, we point out the shortcomings of existing solutions and derive a simple yet effective countermeasure. We observe that in order to realise their full potential, static side-channel attacks require the targeted data to remain unchanged for a certain amount of time. For some cryptographic secrets this happens naturally, for others it requires stopping the target circuit's clock. Our proposal, called Borrowed Time, hinders an attacker's ability to leverage such idle conditions, even if full control over the global clock signal is obtained. For that, by design, key-dependent data may only be present in unprotected temporary storage when strictly needed. Borrowed Time then continuously monitors the target circuit and upon detecting an idle state, securely wipes sensitive contents.   We demonstrate the need for our countermeasure and its effectiveness by mounting practical static power SCA attacks against cryptographic systems on FPGAs, with and without Borrowed Time. In one case we attack a masked implementation and show that it is only protected with our countermeasure in place. Furthermore we demonstrate that secure on-demand wiping of sensitive data works as intended, affirming the theory that the technique also effectively hinders LLSI and IA.

摘要: 近年来，出现了一类新的旁路攻击。在动态计算期间，攻击者不再以器件排放为目标，而是频繁地利用集成电路在静态下的泄漏或响应行为。这一类的成员包括静态功率侧通道分析(SCA)、激光逻辑状态成像(LLSI)和阻抗分析(IA)。尽管依赖于不同的物理现象，但它们都能够以高精度和低噪声的方式从静态电路中提取敏感信息-这一特征对许多现有的旁路对抗措施构成了重大威胁。在这项工作中，我们指出了现有解决方案的不足，并得出了一个简单而有效的对策。我们观察到，为了充分发挥其潜力，静态侧通道攻击要求目标数据在一定时间内保持不变。对于一些密码秘密来说，这是自然而然发生的，而对另一些秘密来说，这需要停止目标电路的时钟。我们的方案称为借用时间，即使获得了对全局时钟信号的完全控制，也会阻碍攻击者利用这种空闲条件的能力。为此，根据设计，依赖于密钥的数据可能仅在严格需要时才存在于不受保护的临时存储中。然后，借用时间持续监视目标电路，并在检测到空闲状态时，安全地擦除敏感内容。我们通过对FPGA上的密码系统进行实用的静态功率SCA攻击，在有和没有借用时间的情况下，证明了我们的对策的必要性和有效性。在一种情况下，我们攻击一个被屏蔽的实现，并表明只有我们的对策到位才能保护它。此外，我们还演示了安全按需擦除敏感数据的工作原理，证实了该技术也有效地阻碍了LLSI和IA的理论。



## **4. MultiTrust: A Comprehensive Benchmark Towards Trustworthy Multimodal Large Language Models**

MultiTrust：值得信赖的多模式大型语言模型的综合基准 cs.CL

100 pages, 84 figures, 33 tables

**SubmitDate**: 2024-12-06    [abs](http://arxiv.org/abs/2406.07057v2) [paper-pdf](http://arxiv.org/pdf/2406.07057v2)

**Authors**: Yichi Zhang, Yao Huang, Yitong Sun, Chang Liu, Zhe Zhao, Zhengwei Fang, Yifan Wang, Huanran Chen, Xiao Yang, Xingxing Wei, Hang Su, Yinpeng Dong, Jun Zhu

**Abstract**: Despite the superior capabilities of Multimodal Large Language Models (MLLMs) across diverse tasks, they still face significant trustworthiness challenges. Yet, current literature on the assessment of trustworthy MLLMs remains limited, lacking a holistic evaluation to offer thorough insights into future improvements. In this work, we establish MultiTrust, the first comprehensive and unified benchmark on the trustworthiness of MLLMs across five primary aspects: truthfulness, safety, robustness, fairness, and privacy. Our benchmark employs a rigorous evaluation strategy that addresses both multimodal risks and cross-modal impacts, encompassing 32 diverse tasks with self-curated datasets. Extensive experiments with 21 modern MLLMs reveal some previously unexplored trustworthiness issues and risks, highlighting the complexities introduced by the multimodality and underscoring the necessity for advanced methodologies to enhance their reliability. For instance, typical proprietary models still struggle with the perception of visually confusing images and are vulnerable to multimodal jailbreaking and adversarial attacks; MLLMs are more inclined to disclose privacy in text and reveal ideological and cultural biases even when paired with irrelevant images in inference, indicating that the multimodality amplifies the internal risks from base LLMs. Additionally, we release a scalable toolbox for standardized trustworthiness research, aiming to facilitate future advancements in this important field. Code and resources are publicly available at: https://multi-trust.github.io/.

摘要: 尽管多模式大型语言模型(MLLM)在不同的任务中具有卓越的能力，但它们仍然面临着重大的可信性挑战。然而，目前关于评估值得信赖的MLLMS的文献仍然有限，缺乏全面的评估来提供对未来改进的透彻见解。在这项工作中，我们建立了多重信任，这是第一个关于MLLMS可信度的全面和统一的基准，涉及五个主要方面：真实性、安全性、健壮性、公平性和隐私性。我们的基准采用了严格的评估战略，同时应对多式联运风险和跨联运影响，包括32项不同的任务和自我管理的数据集。对21个现代多模式管理进行的广泛实验揭示了一些以前从未探索过的可信度问题和风险，突显了多模式带来的复杂性，并强调了先进方法提高其可靠性的必要性。例如，典型的专有模型仍然难以识别视觉上令人困惑的图像，容易受到多模式越狱和敌意攻击；MLLM更倾向于在文本中泄露隐私，甚至在推理中与无关图像搭配使用时也会暴露意识形态和文化偏见，这表明多模式放大了基本LLM的内部风险。此外，我们还发布了一个用于标准化可信度研究的可扩展工具箱，旨在促进这一重要领域的未来发展。代码和资源可在以下网址公开获得：https://multi-trust.github.io/.



## **5. Quantum Security Analysis of the Key-Alternating Ciphers**

密钥交替密码的量子安全分析 quant-ph

**SubmitDate**: 2024-12-06    [abs](http://arxiv.org/abs/2412.05026v1) [paper-pdf](http://arxiv.org/pdf/2412.05026v1)

**Authors**: Chen Bai, Mehdi Esmaili, Atul Mantri

**Abstract**: We study the security of key-alternating ciphers (KAC), a generalization of Even-Mansour ciphers over multiple rounds, which serve as abstractions for many block cipher constructions, particularly AES. While the classical security of KAC has been extensively studied, little is known about its security against quantum adversaries. In this paper, we introduce the first nontrivial quantum key-recovery attack on multi-round KAC in a model where the adversary has quantum access to only one of the public permutations. Our attack applies to any $t$-round KAC, achieving quantum query complexity of $O(2^{\frac{t(t+1)n}{(t+1)^2+1}})$, where $n$ is the size of each individual key, in a realistic quantum threat model, compared to the classical bound of $O(2^{\frac{tn}{(t+1)}})$ queries given by Bogdanev et al. (EUROCRYPT 2012). Our quantum attack leverages a novel approach based on quantum walk algorithms. Additionally, using the quantum hybrid method in our new threat model, we extend the Even-Mansour lower bound of $\Omega(2^{\frac{n}{3}})$ given by Alagic et al. (EUROCRYPT 2022) to $\Omega(2^{\frac{(t-1)n}{t}})$ for the $t$-round KAC (for $t \geq 2$).

摘要: 本文研究了密钥交替密码(KAC)的安全性，它是偶-Mansour密码在多轮上的推广，是许多分组密码构造的抽象，特别是AES。虽然KAC的经典安全性已经得到了广泛的研究，但对于它对量子对手的安全性却知之甚少。在这篇文章中，我们介绍了第一个非平凡的量子密钥恢复攻击，在一个模型中，对手只有一个公开置换的量子访问。我们的攻击适用于任何$t$轮KAC，在现实量子威胁模型中，得到的量子查询复杂度为$O(2^{\FRAC{t(t+1)n}{(t+1)^2+1})$，其中$n$是每个单独密钥的大小，而Bogdanev等人给出的经典查询数为$O(2^{\FRAC{tn}{(t+1)}})$。(欧洲密码2012)。我们的量子攻击利用了一种基于量子行走算法的新方法。此外，在我们的新威胁模型中使用量子混合方法，我们推广了Alocage等人给出的$Omega(2^{\frac{n}{3}})$的偶数-Mansour下界。(Eurocrypt 2022)到$\Omega(2^{\FRAC{(t-1)n}{t}})$，用于$t$轮KAC(用于$t\geq 2$)。



## **6. Endless Jailbreaks with Bijection Learning**

通过双射学习实现无休止的越狱 cs.CL

**SubmitDate**: 2024-12-06    [abs](http://arxiv.org/abs/2410.01294v2) [paper-pdf](http://arxiv.org/pdf/2410.01294v2)

**Authors**: Brian R. Y. Huang, Maximilian Li, Leonard Tang

**Abstract**: Despite extensive safety measures, LLMs are vulnerable to adversarial inputs, or jailbreaks, which can elicit unsafe behaviors. In this work, we introduce bijection learning, a powerful attack algorithm which automatically fuzzes LLMs for safety vulnerabilities using randomly-generated encodings whose complexity can be tightly controlled. We leverage in-context learning to teach models bijective encodings, pass encoded queries to the model to bypass built-in safety mechanisms, and finally decode responses back into English. Our attack is extremely effective on a wide range of frontier language models. Moreover, by controlling complexity parameters such as number of key-value mappings in the encodings, we find a close relationship between the capability level of the attacked LLM and the average complexity of the most effective bijection attacks. Our work highlights that new vulnerabilities in frontier models can emerge with scale: more capable models are more severely jailbroken by bijection attacks.

摘要: 尽管采取了广泛的安全措施，但LLM很容易受到敌意输入或越狱的影响，这可能会引发不安全的行为。在这项工作中，我们引入了双射学习，这是一种强大的攻击算法，它使用随机生成的编码来自动模糊LLM的安全漏洞，其复杂性可以严格控制。我们利用情景学习来教授模型双射编码，将编码的查询传递给模型以绕过内置的安全机制，最后将响应解码回英语。我们的攻击对广泛的前沿语言模型非常有效。此外，通过控制编码中的键值映射次数等复杂性参数，我们发现被攻击LLM的能力水平与最有效的双射攻击的平均复杂性之间存在密切的关系。我们的工作突出表明，前沿模型中的新漏洞可能会随着规模的扩大而出现：能力更强的模型被双射攻击越狱的情况更严重。



## **7. SleeperMark: Towards Robust Watermark against Fine-Tuning Text-to-image Diffusion Models**

SleeperMark：针对微调文本到图像扩散模型的鲁棒水印 cs.CV

**SubmitDate**: 2024-12-06    [abs](http://arxiv.org/abs/2412.04852v1) [paper-pdf](http://arxiv.org/pdf/2412.04852v1)

**Authors**: Zilan Wang, Junfeng Guo, Jiacheng Zhu, Yiming Li, Heng Huang, Muhao Chen, Zhengzhong Tu

**Abstract**: Recent advances in large-scale text-to-image (T2I) diffusion models have enabled a variety of downstream applications, including style customization, subject-driven personalization, and conditional generation. As T2I models require extensive data and computational resources for training, they constitute highly valued intellectual property (IP) for their legitimate owners, yet making them incentive targets for unauthorized fine-tuning by adversaries seeking to leverage these models for customized, usually profitable applications. Existing IP protection methods for diffusion models generally involve embedding watermark patterns and then verifying ownership through generated outputs examination, or inspecting the model's feature space. However, these techniques are inherently ineffective in practical scenarios when the watermarked model undergoes fine-tuning, and the feature space is inaccessible during verification ((i.e., black-box setting). The model is prone to forgetting the previously learned watermark knowledge when it adapts to a new task. To address this challenge, we propose SleeperMark, a novel framework designed to embed resilient watermarks into T2I diffusion models. SleeperMark explicitly guides the model to disentangle the watermark information from the semantic concepts it learns, allowing the model to retain the embedded watermark while continuing to be fine-tuned to new downstream tasks. Our extensive experiments demonstrate the effectiveness of SleeperMark across various types of diffusion models, including latent diffusion models (e.g., Stable Diffusion) and pixel diffusion models (e.g., DeepFloyd-IF), showing robustness against downstream fine-tuning and various attacks at both the image and model levels, with minimal impact on the model's generative capability. The code is available at https://github.com/taco-group/SleeperMark.

摘要: 大规模文本到图像(T2I)扩散模型的最新进展使各种下游应用成为可能，包括样式定制、主题驱动的个性化和条件生成。由于T2I模型需要大量的数据和计算资源来进行培训，因此对于其合法所有者来说，它们构成了非常有价值的知识产权(IP)，但也使它们成为寻求利用这些模型进行定制、通常是有利可图的应用的对手未经授权进行微调的激励目标。现有的扩散模型的知识产权保护方法一般都是先嵌入水印图案，然后通过生成输出检查或检查模型的特征空间来验证所有权。然而，当水印模型经过微调，并且在验证过程中(即，黑盒设置)无法访问特征空间时，这些技术在实际场景中本质上是无效的。该模型在适应新任务时，容易忘记先前学习到的水印知识。为了应对这一挑战，我们提出了SleeperMark，一个新的框架，旨在将弹性水印嵌入到T2I扩散模型中。SleeperMark明确地引导模型将水印信息从它学习的语义概念中分离出来，允许模型保留嵌入的水印，同时继续微调到新的下游任务。我们的大量实验证明了SleeperMark在各种类型的扩散模型上的有效性，包括潜在扩散模型(例如，稳定扩散)和像素扩散模型(例如，DeepFloyd-IF)，显示出对图像和模型级别的下游微调和各种攻击的鲁棒性，而对模型的生成能力的影响最小。代码可在https://github.com/taco-group/SleeperMark.上获得



## **8. Plentiful Jailbreaks with String Compositions**

弦乐作品丰富越狱 cs.CL

NeurIPS SoLaR Workshop 2024

**SubmitDate**: 2024-12-06    [abs](http://arxiv.org/abs/2411.01084v2) [paper-pdf](http://arxiv.org/pdf/2411.01084v2)

**Authors**: Brian R. Y. Huang

**Abstract**: Large language models (LLMs) remain vulnerable to a slew of adversarial attacks and jailbreaking methods. One common approach employed by white-hat attackers, or red-teamers, is to process model inputs and outputs using string-level obfuscations, which can include leetspeak, rotary ciphers, Base64, ASCII, and more. Our work extends these encoding-based attacks by unifying them in a framework of invertible string transformations. With invertibility, we can devise arbitrary string compositions, defined as sequences of transformations, that we can encode and decode end-to-end programmatically. We devise a automated best-of-n attack that samples from a combinatorially large number of string compositions. Our jailbreaks obtain competitive attack success rates on several leading frontier models when evaluated on HarmBench, highlighting that encoding-based attacks remain a persistent vulnerability even in advanced LLMs.

摘要: 大型语言模型（LLM）仍然容易受到一系列对抗攻击和越狱方法的影响。白帽攻击者或红团队使用的一种常见方法是使用字符串级混淆处理模型输入和输出，其中可以包括leetspeak、旋转密码、Base 64、ASC等。我们的工作通过将这些基于编码的攻击统一到可逆字符串转换的框架中来扩展它们。通过可逆性，我们可以设计任意的字符串组合，定义为转换序列，我们可以通过编程方式进行端到端编码和解码。我们设计了一种自动化的n中最佳攻击，该攻击从组合上大量的字符串组合中进行采样。在HarmBench上进行评估时，我们的越狱在几个领先的前沿模型上获得了有竞争力的攻击成功率，这凸显了即使在高级LLM中，基于编码的攻击仍然是一个持久的漏洞。



## **9. PADetBench: Towards Benchmarking Physical Attacks against Object Detection**

PADetBench：针对对象检测的物理攻击基准 cs.CV

**SubmitDate**: 2024-12-06    [abs](http://arxiv.org/abs/2408.09181v2) [paper-pdf](http://arxiv.org/pdf/2408.09181v2)

**Authors**: Jiawei Lian, Jianhong Pan, Lefan Wang, Yi Wang, Lap-Pui Chau, Shaohui Mei

**Abstract**: Physical attacks against object detection have gained increasing attention due to their significant practical implications. However, conducting physical experiments is extremely time-consuming and labor-intensive. Moreover, physical dynamics and cross-domain transformation are challenging to strictly regulate in the real world, leading to unaligned evaluation and comparison, severely hindering the development of physically robust models. To accommodate these challenges, we explore utilizing realistic simulation to thoroughly and rigorously benchmark physical attacks with fairness under controlled physical dynamics and cross-domain transformation. This resolves the problem of capturing identical adversarial images that cannot be achieved in the real world. Our benchmark includes 20 physical attack methods, 48 object detectors, comprehensive physical dynamics, and evaluation metrics. We also provide end-to-end pipelines for dataset generation, detection, evaluation, and further analysis. In addition, we perform 8064 groups of evaluation based on our benchmark, which includes both overall evaluation and further detailed ablation studies for controlled physical dynamics. Through these experiments, we provide in-depth analyses of physical attack performance and physical adversarial robustness, draw valuable observations, and discuss potential directions for future research.   Codebase: https://github.com/JiaweiLian/Benchmarking_Physical_Attack

摘要: 针对目标检测的物理攻击由于其重要的实际意义而受到越来越多的关注。然而，进行物理实验是极其耗时和劳动密集型的。此外，物理动力学和跨域转换在现实世界中面临严格规范的挑战，导致不一致的评估和比较，严重阻碍了物理稳健模型的发展。为了应对这些挑战，我们探索利用真实的模拟在受控的物理动力学和跨域转换下，彻底和严格地基准具有公平性的物理攻击。这解决了在现实世界中无法实现的捕获相同的对抗性图像的问题。我们的基准包括20种物理攻击方法、48个对象探测器、全面的物理动力学和评估指标。我们还提供用于数据集生成、检测、评估和进一步分析的端到端管道。此外，我们根据我们的基准进行了8064组评估，其中包括对受控物理动力学的整体评估和进一步的详细消融研究。通过这些实验，我们对物理攻击性能和物理对抗健壮性进行了深入的分析，得出了有价值的观察结果，并讨论了未来研究的潜在方向。码基：https://github.com/JiaweiLian/Benchmarking_Physical_Attack



## **10. Defending Object Detectors against Patch Attacks with Out-of-Distribution Smoothing**

利用非分布平滑保护对象检测器免受补丁攻击 cs.LG

**SubmitDate**: 2024-12-06    [abs](http://arxiv.org/abs/2205.08989v2) [paper-pdf](http://arxiv.org/pdf/2205.08989v2)

**Authors**: Ryan Feng, Neal Mangaokar, Jihye Choi, Somesh Jha, Atul Prakash

**Abstract**: Patch attacks against object detectors have been of recent interest due to their being physically realizable and more closely aligned with practical systems. In response to this threat, many new defenses have been proposed that train a patch segmenter model to detect and remove the patch before the image is passed to the downstream model. We unify these approaches with a flexible framework, OODSmoother, which characterizes the properties of approaches that aim to remove adversarial patches. This framework naturally guides us to design 1) a novel adaptive attack that breaks existing patch attack defenses on object detectors, and 2) a novel defense approach SemPrior that takes advantage of semantic priors. Our key insight behind SemPrior is that the existing machine learning-based patch detectors struggle to learn semantic priors and that explicitly incorporating them can improve performance. We find that SemPrior alone provides up to a 40% gain, or up to a 60% gain when combined with existing defenses.

摘要: 针对对象检测器的补丁攻击最近引起了人们的兴趣，因为它们是物理上可实现的，并且与实际系统更紧密地结合在一起。为了应对这种威胁，人们提出了许多新的防御措施，训练一个补丁分割模型，在图像被传递到下游模型之前检测并移除补丁。我们将这些方法与一个灵活的框架OODSmother统一起来，该框架表征了旨在移除敌意补丁的方法的特性。该框架自然而然地指导我们设计了一种新的自适应攻击，它打破了现有的对对象检测器的补丁攻击防御，以及一种新的利用语义先验的防御方法SemPrior。我们在SemPrior背后的关键见解是，现有的基于机器学习的补丁检测器很难学习语义先验，显式整合它们可以提高性能。我们发现，仅SemPrior一项就可以提供高达40%的收益，或者当与现有防御相结合时，可以提供高达60%的收益。



## **11. Targeting the Core: A Simple and Effective Method to Attack RAG-based Agents via Direct LLM Manipulation**

瞄准核心：通过直接LLM操纵攻击基于RAG的代理的简单有效方法 cs.AI

**SubmitDate**: 2024-12-05    [abs](http://arxiv.org/abs/2412.04415v1) [paper-pdf](http://arxiv.org/pdf/2412.04415v1)

**Authors**: Xuying Li, Zhuo Li, Yuji Kosuga, Yasuhiro Yoshida, Victor Bian

**Abstract**: AI agents, powered by large language models (LLMs), have transformed human-computer interactions by enabling seamless, natural, and context-aware communication. While these advancements offer immense utility, they also inherit and amplify inherent safety risks such as bias, fairness, hallucinations, privacy breaches, and a lack of transparency. This paper investigates a critical vulnerability: adversarial attacks targeting the LLM core within AI agents. Specifically, we test the hypothesis that a deceptively simple adversarial prefix, such as \textit{Ignore the document}, can compel LLMs to produce dangerous or unintended outputs by bypassing their contextual safeguards. Through experimentation, we demonstrate a high attack success rate (ASR), revealing the fragility of existing LLM defenses. These findings emphasize the urgent need for robust, multi-layered security measures tailored to mitigate vulnerabilities at the LLM level and within broader agent-based architectures.

摘要: 由大型语言模型（LLM）支持的人工智能代理通过实现无缝、自然和上下文感知的通信来改变了人机交互。虽然这些进步提供了巨大的实用性，但它们也继承和放大了固有的安全风险，例如偏见、公平、幻觉、隐私侵犯和缺乏透明度。本文研究了一个关键漏洞：针对人工智能代理内LLM核心的对抗攻击。具体来说，我们测试了这样的假设：看似简单的对抗性前置码（例如\textit{忽略文档}）可以迫使LLM绕过上下文保障措施来产生危险或非预期的输出。通过实验，我们展示了高攻击成功率（ASB），揭示了现有LLM防御的脆弱性。这些调查结果强调，迫切需要针对LLM级别和更广泛的基于代理的架构中的漏洞量身定制的强大、多层的安全措施。



## **12. Adversarial Attacks on Large Language Models in Medicine**

医学中对大型语言模型的对抗攻击 cs.AI

**SubmitDate**: 2024-12-05    [abs](http://arxiv.org/abs/2406.12259v2) [paper-pdf](http://arxiv.org/pdf/2406.12259v2)

**Authors**: Yifan Yang, Qiao Jin, Furong Huang, Zhiyong Lu

**Abstract**: The integration of Large Language Models (LLMs) into healthcare applications offers promising advancements in medical diagnostics, treatment recommendations, and patient care. However, the susceptibility of LLMs to adversarial attacks poses a significant threat, potentially leading to harmful outcomes in delicate medical contexts. This study investigates the vulnerability of LLMs to two types of adversarial attacks in three medical tasks. Utilizing real-world patient data, we demonstrate that both open-source and proprietary LLMs are susceptible to manipulation across multiple tasks. This research further reveals that domain-specific tasks demand more adversarial data in model fine-tuning than general domain tasks for effective attack execution, especially for more capable models. We discover that while integrating adversarial data does not markedly degrade overall model performance on medical benchmarks, it does lead to noticeable shifts in fine-tuned model weights, suggesting a potential pathway for detecting and countering model attacks. This research highlights the urgent need for robust security measures and the development of defensive mechanisms to safeguard LLMs in medical applications, to ensure their safe and effective deployment in healthcare settings.

摘要: 将大型语言模型(LLM)集成到医疗保健应用程序中，在医疗诊断、治疗建议和患者护理方面提供了有希望的进步。然而，LLMS对对抗性攻击的敏感性构成了一个重大威胁，可能会在微妙的医疗环境中导致有害后果。本研究调查了LLMS在三个医疗任务中对两种类型的对抗性攻击的脆弱性。利用真实世界的患者数据，我们证明了开源和专有LLM都容易受到跨多个任务的操纵。这项研究进一步表明，特定领域的任务在模型微调中需要比一般领域任务更多的对抗性数据才能有效地执行攻击，特别是对于能力更强的模型。这项研究强调了迫切需要强有力的安全措施和开发防御机制来保护医疗应用中的低成本管理，以确保其在医疗保健环境中的安全和有效部署。



## **13. Machine Theory of Mind for Autonomous Cyber-Defence**

自主网络防御的机器思维理论 cs.LG

29 pages, 17 figures, 12 tables

**SubmitDate**: 2024-12-05    [abs](http://arxiv.org/abs/2412.04367v1) [paper-pdf](http://arxiv.org/pdf/2412.04367v1)

**Authors**: Luke Swaby, Matthew Stewart, Daniel Harrold, Chris Willis, Gregory Palmer

**Abstract**: Intelligent autonomous agents hold much potential for the domain of cyber-security. However, due to many state-of-the-art approaches relying on uninterpretable black-box models, there is growing demand for methods that offer stakeholders clear and actionable insights into their latent beliefs and motivations. To address this, we evaluate Theory of Mind (ToM) approaches for Autonomous Cyber Operations. Upon learning a robust prior, ToM models can predict an agent's goals, behaviours, and contextual beliefs given only a handful of past behaviour observations. In this paper, we introduce a novel Graph Neural Network (GNN)-based ToM architecture tailored for cyber-defence, Graph-In, Graph-Out (GIGO)-ToM, which can accurately predict both the targets and attack trajectories of adversarial cyber agents over arbitrary computer network topologies. To evaluate the latter, we propose a novel extension of the Wasserstein distance for measuring the similarity of graph-based probability distributions. Whereas the standard Wasserstein distance lacks a fixed reference scale, we introduce a graph-theoretic normalization factor that enables a standardized comparison between networks of different sizes. We furnish this metric, which we term the Network Transport Distance (NTD), with a weighting function that emphasizes predictions according to custom node features, allowing network operators to explore arbitrary strategic considerations. Benchmarked against a Graph-In, Dense-Out (GIDO)-ToM architecture in an abstract cyber-defence environment, our empirical evaluations show that GIGO-ToM can accurately predict the goals and behaviours of various unseen cyber-attacking agents across a range of network topologies, as well as learn embeddings that can effectively characterize their policies.

摘要: 智能自主代理在网络安全领域具有很大的潜力。然而，由于许多最先进的方法依赖于无法解释的黑盒模型，因此对为利益相关者提供对其潜在信念和动机的清晰且可操作的洞察的方法的需求越来越大。为了解决这个问题，我们评估了自主网络操作的心理理论(TOM)方法。在学习稳健的先验知识后，TOM模型可以预测代理人的目标、行为和上下文信念，只需给出少数过去的行为观察。本文提出了一种新的基于图神经网络(GNN)的网络防御、Graph-In、Graph-Out(GIGO)-TOM体系结构，它能够准确预测任意计算机网络拓扑结构下敌意网络代理的目标和攻击轨迹。为了评估后者，我们提出了一种新的扩展的Wasserstein距离来度量基于图的概率分布的相似性。鉴于标准Wasserstein距离缺乏固定的参考标度，我们引入了图论归一化因子，使不同规模的网络之间能够进行标准化比较。我们为这个我们称为网络传输距离(NTD)的度量提供了一个权重函数，该函数强调根据自定义节点特征进行预测，从而允许网络运营商探索任意的战略考虑。在抽象的网络防御环境中，以Graph-In，Density-Out(GIDO)-TOM体系结构为基准，我们的经验评估表明，GIGO-TOM能够准确预测各种网络拓扑结构中各种看不见的网络攻击代理的目标和行为，并学习能够有效表征其策略的嵌入。



## **14. Bayes-Nash Generative Privacy Protection Against Membership Inference Attacks**

针对会员推断攻击的Bayes-Nash生成隐私保护 cs.CR

arXiv admin note: substantial text overlap with arXiv:2406.01811

**SubmitDate**: 2024-12-05    [abs](http://arxiv.org/abs/2410.07414v2) [paper-pdf](http://arxiv.org/pdf/2410.07414v2)

**Authors**: Tao Zhang, Rajagopal Venkatesaraman, Rajat K. De, Bradley A. Malin, Yevgeniy Vorobeychik

**Abstract**: Membership inference attacks (MIAs) expose significant privacy risks by determining whether an individual's data is in a dataset. While differential privacy (DP) mitigates such risks, it faces challenges in general when achieving an optimal balance between privacy and utility, often requiring intractable sensitivity calculations and limiting flexibility in complex compositions. We propose a game-theoretic framework that models privacy protection as a Bayesian game between a defender and an attacker, solved using a general-sum Generative Adversarial Network (general-sum GAN). The Bayes Generative Privacy (BGP) response, based on cross-entropy loss, defines the attacker's optimal strategy, leading to the Bayes-Nash Generative Privacy (BNGP) strategy, which achieves the optimal privacy-utility trade-off tailored to the defender's preferences. The BNGP strategy avoids sensitivity calculations, supports compositions of correlated mechanisms, and is robust to the attacker's heterogeneous preferences over true and false positives. A case study on binary dataset summary statistics demonstrates its superiority over likelihood ratio test (LRT)-based attacks, including the uniformly most powerful LRT. Empirical results confirm BNGP's effectiveness.

摘要: 成员身份推断攻击(MIA)通过确定个人数据是否在数据集中暴露了显著的隐私风险。虽然差异隐私(DP)减轻了此类风险，但在实现隐私和效用之间的最佳平衡时，它总体上面临着挑战，通常需要难以处理的敏感性计算，并限制了复杂组成的灵活性。我们提出了一个博弈论框架，该框架将隐私保护建模为防御者和攻击者之间的贝叶斯博弈，并使用一般和生成性对抗网络(General-Sum Generative Adversarial Network，GAN)进行求解。基于交叉熵损失的贝叶斯生成隐私(BGP)响应定义了攻击者的最优策略，导致了贝叶斯-纳什生成隐私(BNGP)策略，该策略实现了根据防御者的偏好定制的最优隐私效用权衡。BNGP策略避免了敏感度计算，支持相关机制的组合，并且对攻击者对真阳性和假阳性的不同偏好具有很强的鲁棒性。对二进制数据集摘要统计的实例研究表明，该方法优于基于似然比检验(LRT)的攻击，包括一致最强大的LRT。实证结果证实了BNGP的有效性。



## **15. Robustness of Practical Perceptual Hashing Algorithms to Hash-Evasion and Hash-Inversion Attacks**

实用感知哈希算法对哈希规避和哈希倒置攻击的鲁棒性 cs.CR

Published in the Third Workshop on New Frontiers in Adversarial  Machine Learning at NeurIPS 2024

**SubmitDate**: 2024-12-05    [abs](http://arxiv.org/abs/2406.00918v2) [paper-pdf](http://arxiv.org/pdf/2406.00918v2)

**Authors**: Jordan Madden, Moxanki Bhavsar, Lhamo Dorje, Xiaohua Li

**Abstract**: Perceptual hashing algorithms (PHAs) are widely used for identifying illegal online content and are thus integral to various sensitive applications. However, due to their hasty deployment in real-world scenarios, their adversarial security has not been thoroughly evaluated. This paper assesses the security of three widely utilized PHAs - PhotoDNA, PDQ, and NeuralHash - against hash-evasion and hash-inversion attacks. Contrary to existing literature, our findings indicate that these PHAs demonstrate significant robustness against such attacks. We provide an explanation for these differing results, highlighting that the inherent robustness is partially due to the random hash variations characteristic of PHAs. Additionally, we propose a defense method that enhances security by intentionally introducing perturbations into the hashes.

摘要: 感知哈希算法（PHA）广泛用于识别非法在线内容，因此是各种敏感应用程序的组成部分。然而，由于它们在现实世界场景中的仓促部署，因此它们的对抗安全性尚未得到彻底评估。本文评估了三种广泛使用的PHA- PhotoDNA、PDQ和Neural哈希--针对哈希规避和哈希倒置攻击的安全性。与现有文献相反，我们的研究结果表明，这些HPA对此类攻击表现出显着的鲁棒性。我们对这些不同的结果提供了解释，强调固有的稳健性部分归因于PHA的随机哈希变化特征。此外，我们还提出了一种防御方法，通过故意向哈希中引入扰动来增强安全性。



## **16. PBP: Post-training Backdoor Purification for Malware Classifiers**

PBP：恶意软件分类器的培训后后门净化 cs.LG

Accepted at NDSS 2025

**SubmitDate**: 2024-12-05    [abs](http://arxiv.org/abs/2412.03441v2) [paper-pdf](http://arxiv.org/pdf/2412.03441v2)

**Authors**: Dung Thuy Nguyen, Ngoc N. Tran, Taylor T. Johnson, Kevin Leach

**Abstract**: In recent years, the rise of machine learning (ML) in cybersecurity has brought new challenges, including the increasing threat of backdoor poisoning attacks on ML malware classifiers. For instance, adversaries could inject malicious samples into public malware repositories, contaminating the training data and potentially misclassifying malware by the ML model. Current countermeasures predominantly focus on detecting poisoned samples by leveraging disagreements within the outputs of a diverse set of ensemble models on training data points. However, these methods are not suitable for scenarios where Machine Learning-as-a-Service (MLaaS) is used or when users aim to remove backdoors from a model after it has been trained. Addressing this scenario, we introduce PBP, a post-training defense for malware classifiers that mitigates various types of backdoor embeddings without assuming any specific backdoor embedding mechanism. Our method exploits the influence of backdoor attacks on the activation distribution of neural networks, independent of the trigger-embedding method. In the presence of a backdoor attack, the activation distribution of each layer is distorted into a mixture of distributions. By regulating the statistics of the batch normalization layers, we can guide a backdoored model to perform similarly to a clean one. Our method demonstrates substantial advantages over several state-of-the-art methods, as evidenced by experiments on two datasets, two types of backdoor methods, and various attack configurations. Notably, our approach requires only a small portion of the training data -- only 1\% -- to purify the backdoor and reduce the attack success rate from 100\% to almost 0\%, a 100-fold improvement over the baseline methods. Our code is available at \url{https://github.com/judydnguyen/pbp-backdoor-purification-official}.

摘要: 近年来，机器学习(ML)在网络安全领域的兴起带来了新的挑战，包括对ML恶意软件分类器进行后门中毒攻击的威胁越来越大。例如，攻击者可以将恶意样本注入公共恶意软件存储库中，污染训练数据，并可能根据ML模型对恶意软件进行错误分类。目前的对策主要集中在通过利用关于训练数据点的一组不同集合模型的输出中的不一致来检测有毒样本。然而，这些方法不适用于使用机器学习即服务(MLaaS)的场景，或者用户希望在模型经过训练后删除后门的场景。针对这种情况，我们引入了PBP，这是一种针对恶意软件分类器的训练后防御，它可以减少各种类型的后门嵌入，而不需要假设任何特定的后门嵌入机制。我们的方法利用了后门攻击对神经网络激活分布的影响，独立于触发器嵌入方法。在后门攻击存在的情况下，每一层的激活分布被扭曲为混合分布。通过调整批处理归一化层的统计信息，我们可以引导回溯模型以类似于干净模型的方式执行。在两个数据集、两种类型的后门方法和不同的攻击配置上的实验证明，我们的方法比几种最先进的方法显示出了实质性的优势。值得注意的是，我们的方法只需要一小部分训练数据--只需要1\%--来净化后门，并将攻击成功率从100\%降低到几乎0\%，比基线方法提高了100倍。我们的代码可以在\url{https://github.com/judydnguyen/pbp-backdoor-purification-official}.上找到



## **17. On the Lack of Robustness of Binary Function Similarity Systems**

论二元函数相似系统的鲁棒性缺乏 cs.CR

**SubmitDate**: 2024-12-05    [abs](http://arxiv.org/abs/2412.04163v1) [paper-pdf](http://arxiv.org/pdf/2412.04163v1)

**Authors**: Gianluca Capozzi, Tong Tang, Jie Wan, Ziqi Yang, Daniele Cono D'Elia, Giuseppe Antonio Di Luna, Lorenzo Cavallaro, Leonardo Querzoni

**Abstract**: Binary function similarity, which often relies on learning-based algorithms to identify what functions in a pool are most similar to a given query function, is a sought-after topic in different communities, including machine learning, software engineering, and security. Its importance stems from the impact it has in facilitating several crucial tasks, from reverse engineering and malware analysis to automated vulnerability detection. Whereas recent work cast light around performance on this long-studied problem, the research landscape remains largely lackluster in understanding the resiliency of the state-of-the-art machine learning models against adversarial attacks. As security requires to reason about adversaries, in this work we assess the robustness of such models through a simple yet effective black-box greedy attack, which modifies the topology and the content of the control flow of the attacked functions. We demonstrate that this attack is successful in compromising all the models, achieving average attack success rates of 57.06% and 95.81% depending on the problem settings (targeted and untargeted attacks). Our findings are insightful: top performance on clean data does not necessarily relate to top robustness properties, which explicitly highlights performance-robustness trade-offs one should consider when deploying such models, calling for further research.

摘要: 二进制函数相似性通常依赖于基于学习的算法来确定池中的哪些函数与给定的查询函数最相似，这是包括机器学习、软件工程和安全在内的不同社区的热门话题。它的重要性源于它在促进几个关键任务方面的影响，从反向工程和恶意软件分析到自动漏洞检测。虽然最近的工作揭示了这个长期研究的问题的性能，但在理解最先进的机器学习模型对对手攻击的弹性方面，研究前景仍然很平淡。由于安全性需要对攻击者进行推理，在本文中，我们通过一种简单而有效的黑盒贪婪攻击来评估这种模型的健壮性，该攻击修改了被攻击函数的拓扑结构和控制流的内容。我们证明了这种攻击在所有模型上都是成功的，根据问题设置(目标攻击和非目标攻击)，平均攻击成功率分别为57.06%和95.81%。我们的发现很有见地：在干净数据上的最佳性能并不一定与顶级健壮性属性相关，这明确强调了在部署此类模型时应该考虑的性能健壮性权衡，呼吁进行进一步的研究。



## **18. Stochastic Monkeys at Play: Random Augmentations Cheaply Break LLM Safety Alignment**

随机猴子在起作用：随机增强轻松打破LLM安全一致 cs.LG

v2: Updated with changes from peer review rebuttal. v1: Version under  peer review

**SubmitDate**: 2024-12-05    [abs](http://arxiv.org/abs/2411.02785v2) [paper-pdf](http://arxiv.org/pdf/2411.02785v2)

**Authors**: Jason Vega, Junsheng Huang, Gaokai Zhang, Hangoo Kang, Minjia Zhang, Gagandeep Singh

**Abstract**: Safety alignment of Large Language Models (LLMs) has recently become a critical objective of model developers. In response, a growing body of work has been investigating how safety alignment can be bypassed through various jailbreaking methods, such as adversarial attacks. However, these jailbreak methods can be rather costly or involve a non-trivial amount of creativity and effort, introducing the assumption that malicious users are high-resource or sophisticated. In this paper, we study how simple random augmentations to the input prompt affect safety alignment effectiveness in state-of-the-art LLMs, such as Llama 3 and Qwen 2. We perform an in-depth evaluation of 17 different models and investigate the intersection of safety under random augmentations with multiple dimensions: augmentation type, model size, quantization, fine-tuning-based defenses, and decoding strategies (e.g., sampling temperature). We show that low-resource and unsophisticated attackers, i.e. $\textit{stochastic monkeys}$, can significantly improve their chances of bypassing alignment with just 25 random augmentations per prompt. Source code and data: https://github.com/uiuc-focal-lab/stochastic-monkeys/

摘要: 大型语言模型(LLM)的安全一致性最近已成为模型开发人员的一个重要目标。作为回应，越来越多的工作一直在研究如何通过各种越狱方法绕过安全对准，例如对抗性攻击。然而，这些越狱方法可能相当昂贵，或者涉及大量的创造力和努力，从而引入了恶意用户是高资源或老练的假设。在本文中，我们研究了输入提示的简单随机增强如何影响最新的LLMS的安全对齐有效性，例如Llama 3和Qwen 2。我们对17个不同的模型进行了深入的评估，并研究了在多个维度的随机增强下的安全性交集：增强类型、模型大小、量化、基于微调的防御和解码策略(例如采样温度)。我们表明，低资源和简单的攻击者，即$\textit{随机猴子}$，可以显著提高他们绕过对齐的机会，每个提示只需25个随机扩展。源代码和数据：https://github.com/uiuc-focal-lab/stochastic-monkeys/



## **19. R-MTLLMF: Resilient Multi-Task Large Language Model Fusion at the Wireless Edge**

R-MTLLMF：无线边缘的弹性多任务大型语言模型融合 eess.SP

**SubmitDate**: 2024-12-05    [abs](http://arxiv.org/abs/2411.18220v2) [paper-pdf](http://arxiv.org/pdf/2411.18220v2)

**Authors**: Aladin Djuhera, Vlad C. Andrei, Mohsen Pourghasemian, Haris Gacanin, Holger Boche, Walid Saad

**Abstract**: Multi-task large language models (MTLLMs) are important for many applications at the wireless edge, where users demand specialized models to handle multiple tasks efficiently. However, training MTLLMs is complex and exhaustive, particularly when tasks are subject to change. Recently, the concept of model fusion via task vectors has emerged as an efficient approach for combining fine-tuning parameters to produce an MTLLM. In this paper, the problem of enabling edge users to collaboratively craft such MTLMs via tasks vectors is studied, under the assumption of worst-case adversarial attacks. To this end, first the influence of adversarial noise to multi-task model fusion is investigated and a relationship between the so-called weight disentanglement error and the mean squared error (MSE) is derived. Using hypothesis testing, it is directly shown that the MSE increases interference between task vectors, thereby rendering model fusion ineffective. Then, a novel resilient MTLLM fusion (R-MTLLMF) is proposed, which leverages insights about the LLM architecture and fine-tuning process to safeguard task vector aggregation under adversarial noise by realigning the MTLLM. The proposed R-MTLLMF is then compared for both worst-case and ideal transmission scenarios to study the impact of the wireless channel. Extensive model fusion experiments with vision LLMs demonstrate R-MTLLMF's effectiveness, achieving close-to-baseline performance across eight different tasks in ideal noise scenarios and significantly outperforming unprotected model fusion in worst-case scenarios. The results further advocate for additional physical layer protection for a holistic approach to resilience, from both a wireless and LLM perspective.

摘要: 多任务大型语言模型(MTLLM)对于无线边缘的许多应用非常重要，因为用户需要专门的模型来高效地处理多个任务。然而，培训MTLLM是复杂和详尽的，特别是在任务可能发生变化的情况下。最近，基于任务向量的模型融合的概念已经成为一种结合微调参数以产生MTLLM的有效方法。本文在假设最坏情况下的敌意攻击的前提下，研究了边缘用户通过任务向量协作创建MTLM的问题。为此，首先研究了对抗性噪声对多任务模型融合的影响，推导了加权解缠误差与均方误差之间的关系。通过假设检验，直接表明MSE增加了任务向量之间的干扰，从而使模型融合无效。然后，提出了一种新的弹性MTLLM融合算法(R-MTLLMF)，该算法利用对LLM体系结构和微调过程的深入了解，通过重新排列MTLLM来保护对抗噪声下的任务向量聚合。然后将所提出的R-MTLLMF在最坏情况和理想传输场景下进行比较，以研究无线信道的影响。用VISION LLMS进行的大量模型融合实验证明了R-MTLLMF的有效性，在理想噪声场景中，R-MTLLMF在八个不同任务上的性能接近基线，而在最坏情况下，R-MTLLMF的性能明显优于无保护的模型融合。从无线和LLM的角度来看，研究结果进一步倡导为整体恢复方法提供额外的物理层保护。



## **20. Safeguarding Text-to-Image Generation via Inference-Time Prompt-Noise Optimization**

通过推理时间干扰优化保护文本到图像的生成 cs.CV

**SubmitDate**: 2024-12-05    [abs](http://arxiv.org/abs/2412.03876v1) [paper-pdf](http://arxiv.org/pdf/2412.03876v1)

**Authors**: Jiangweizhi Peng, Zhiwei Tang, Gaowen Liu, Charles Fleming, Mingyi Hong

**Abstract**: Text-to-Image (T2I) diffusion models are widely recognized for their ability to generate high-quality and diverse images based on text prompts. However, despite recent advances, these models are still prone to generating unsafe images containing sensitive or inappropriate content, which can be harmful to users. Current efforts to prevent inappropriate image generation for diffusion models are easy to bypass and vulnerable to adversarial attacks. How to ensure that T2I models align with specific safety goals remains a significant challenge. In this work, we propose a novel, training-free approach, called Prompt-Noise Optimization (PNO), to mitigate unsafe image generation. Our method introduces a novel optimization framework that leverages both the continuous prompt embedding and the injected noise trajectory in the sampling process to generate safe images. Extensive numerical results demonstrate that our framework achieves state-of-the-art performance in suppressing toxic image generations and demonstrates robustness to adversarial attacks, without needing to tune the model parameters. Furthermore, compared with existing methods, PNO uses comparable generation time while offering the best tradeoff between the conflicting goals of safe generation and prompt-image alignment.

摘要: 文本到图像(T2I)扩散模型因其能够基于文本提示生成高质量和多样化的图像而被广泛认可。然而，尽管最近取得了进展，这些模型仍然容易生成包含敏感或不适当内容的不安全图像，这可能对用户有害。目前为防止为扩散模型生成不适当的图像所做的努力很容易被绕过，并且容易受到对手的攻击。如何确保T2I模型与特定的安全目标保持一致仍然是一个重大挑战。在这项工作中，我们提出了一种新的、无需训练的方法，称为提示噪声优化(PNO)，以减少不安全的图像生成。我们的方法引入了一种新的优化框架，该框架利用采样过程中的连续提示嵌入和注入噪声轨迹来生成安全图像。大量的数值结果表明，我们的框架在抑制有毒图像生成方面达到了最先进的性能，并且在不需要调整模型参数的情况下表现出对对手攻击的稳健性。此外，与现有方法相比，PNO使用了相当的生成时间，同时在安全生成和即时图像对齐这两个相互冲突的目标之间提供了最佳折衷。



## **21. NODE-AdvGAN: Improving the transferability and perceptual similarity of adversarial examples by dynamic-system-driven adversarial generative model**

NODE-AdvGAN：通过动态系统驱动的对抗生成模型提高对抗示例的可移植性和感知相似性 cs.LG

**SubmitDate**: 2024-12-04    [abs](http://arxiv.org/abs/2412.03539v1) [paper-pdf](http://arxiv.org/pdf/2412.03539v1)

**Authors**: Xinheng Xie, Yue Wu, Cuiyu He

**Abstract**: Understanding adversarial examples is crucial for improving the model's robustness, as they introduce imperceptible perturbations that deceive models. Effective adversarial examples, therefore, offer the potential to train more robust models by removing their singularities. We propose NODE-AdvGAN, a novel approach that treats adversarial generation as a continuous process and employs a Neural Ordinary Differential Equation (NODE) for simulating the dynamics of the generator. By mimicking the iterative nature of traditional gradient-based methods, NODE-AdvGAN generates smoother and more precise perturbations that preserve high perceptual similarity when added to benign images. We also propose a new training strategy, NODE-AdvGAN-T, which enhances transferability in black-box attacks by effectively tuning noise parameters during training. Experiments demonstrate that NODE-AdvGAN and NODE-AdvGAN-T generate more effective adversarial examples that achieve higher attack success rates while preserving better perceptual quality than traditional GAN-based methods.

摘要: 理解对抗性例子对于提高模型的稳健性至关重要，因为它们引入了欺骗模型的不可察觉的扰动。因此，有效的对抗性例子提供了通过消除奇点来训练更健壮的模型的可能性。我们提出了一种新的方法NODE-AdvGAN，它将对手的生成视为一个连续的过程，并使用一个神经常微分方程(NODE)来模拟生成器的动态。通过模仿传统基于梯度的方法的迭代性质，NODE-AdvGAN生成更平滑和更精确的扰动，当添加到良性图像中时，保持了较高的感知相似性。我们还提出了一种新的训练策略，NODE-AdvGAN-T，它通过在训练过程中有效地调整噪声参数来增强黑盒攻击的可转移性。实验表明，NODE-AdvGAN和NODE-AdvGAN-T生成了比传统的基于GAN的方法更有效的攻击实例，在保持更好的感知质量的同时获得了更高的攻击成功率。



## **22. Pre-trained Multiple Latent Variable Generative Models are good defenders against Adversarial Attacks**

预训练的多潜在变量生成模型是对抗攻击的良好防御者 cs.CV

**SubmitDate**: 2024-12-04    [abs](http://arxiv.org/abs/2412.03453v1) [paper-pdf](http://arxiv.org/pdf/2412.03453v1)

**Authors**: Dario Serez, Marco Cristani, Alessio Del Bue, Vittorio Murino, Pietro Morerio

**Abstract**: Attackers can deliberately perturb classifiers' input with subtle noise, altering final predictions. Among proposed countermeasures, adversarial purification employs generative networks to preprocess input images, filtering out adversarial noise. In this study, we propose specific generators, defined Multiple Latent Variable Generative Models (MLVGMs), for adversarial purification. These models possess multiple latent variables that naturally disentangle coarse from fine features. Taking advantage of these properties, we autoencode images to maintain class-relevant information, while discarding and re-sampling any detail, including adversarial noise. The procedure is completely training-free, exploring the generalization abilities of pre-trained MLVGMs on the adversarial purification downstream task. Despite the lack of large models, trained on billions of samples, we show that smaller MLVGMs are already competitive with traditional methods, and can be used as foundation models. Official code released at https://github.com/SerezD/gen_adversarial.

摘要: 攻击者可以故意用微妙的噪音扰乱分类器的输入，改变最终的预测。在已提出的对策中，对抗性净化利用产生式网络对输入图像进行预处理，过滤掉对抗性噪声。在这项研究中，我们提出了特定的生成器，定义了多个潜在变量生成模型(MLVGM)，用于对抗净化。这些模型具有多个潜在变量，可以自然地将粗略特征与精细特征区分开来。利用这些特性，我们自动编码图像以维护与类相关的信息，同时丢弃并重新采样任何细节，包括对抗性噪声。该过程完全无需训练，探索了预训练的MLVGM在对抗性净化下游任务中的泛化能力。尽管缺乏大型模型，在数十亿个样本上进行训练，但我们表明，较小的MLVGM已经具有传统方法的竞争力，可以用作基础模型。官方代码在https://github.com/SerezD/gen_adversarial.上发布



## **23. State Frequency Estimation for Anomaly Detection**

异常检测的状态频率估计 cs.LG

9 pages

**SubmitDate**: 2024-12-04    [abs](http://arxiv.org/abs/2412.03442v1) [paper-pdf](http://arxiv.org/pdf/2412.03442v1)

**Authors**: Clinton Cao, Agathe Blaise, Annibale Panichella, Sicco Verwer

**Abstract**: Many works have studied the efficacy of state machines for detecting anomalies within NetFlows. These works typically learn a model from unlabeled data and compute anomaly scores for arbitrary traces based on their likelihood of occurrence or how well they fit within the model. However, these methods do not dynamically adapt their scores based on the traces seen at test time. This becomes a problem when an adversary produces seemingly common traces in their attack, causing the model to miss the detection by assigning low anomaly scores. We propose SEQUENT, a new approach that uses the state visit frequency to adapt its scoring for anomaly detection dynamically. SEQUENT subsequently uses the scores to generate root causes for anomalies. These allow the grouping of alarms and simplify the analysis of anomalies. Our evaluation of SEQUENT on three NetFlow datasets indicates that our approach outperforms existing methods, demonstrating its effectiveness in detecting anomalies.

摘要: 许多作品都研究了状态机检测NetFlows中异常的功效。这些工作通常从未标记的数据中学习模型，并根据任意轨迹的发生可能性或它们在模型中的适应程度来计算任意轨迹的异常分数。然而，这些方法不会根据测试时看到的痕迹动态调整其分数。当对手在攻击中产生看似常见的痕迹，导致模型通过分配低异常分数而错过检测时，这就会成为一个问题。我们提出SEQUENT，这是一种新方法，使用状态访问频率来动态调整异常检测的评分。SEQUENT随后使用分数来生成异常的根本原因。这些可以对警报进行分组并简化异常分析。我们对三个NetFlow数据集的SEQUENT的评估表明，我们的方法优于现有方法，证明了其在检测异常方面的有效性。



## **24. Does Safety Training of LLMs Generalize to Semantically Related Natural Prompts?**

LLM的安全培训是否适用于语义相关的自然知识？ cs.CL

Accepted at the Safe Generative AI Workshop @ NeurIPS 2024

**SubmitDate**: 2024-12-04    [abs](http://arxiv.org/abs/2412.03235v1) [paper-pdf](http://arxiv.org/pdf/2412.03235v1)

**Authors**: Sravanti Addepalli, Yerram Varun, Arun Suggala, Karthikeyan Shanmugam, Prateek Jain

**Abstract**: Large Language Models (LLMs) are known to be susceptible to crafted adversarial attacks or jailbreaks that lead to the generation of objectionable content despite being aligned to human preferences using safety fine-tuning methods. While the large dimensionality of input token space makes it inevitable to find adversarial prompts that can jailbreak these models, we aim to evaluate whether safety fine-tuned LLMs are safe against natural prompts which are semantically related to toxic seed prompts that elicit safe responses after alignment. We surprisingly find that popular aligned LLMs such as GPT-4 can be compromised using naive prompts that are NOT even crafted with an objective of jailbreaking the model. Furthermore, we empirically show that given a seed prompt that elicits a toxic response from an unaligned model, one can systematically generate several semantically related natural prompts that can jailbreak aligned LLMs. Towards this, we propose a method of Response Guided Question Augmentation (ReG-QA) to evaluate the generalization of safety aligned LLMs to natural prompts, that first generates several toxic answers given a seed question using an unaligned LLM (Q to A), and further leverages an LLM to generate questions that are likely to produce these answers (A to Q). We interestingly find that safety fine-tuned LLMs such as GPT-4o are vulnerable to producing natural jailbreak questions from unsafe content (without denial) and can thus be used for the latter (A to Q) step. We obtain attack success rates that are comparable to/ better than leading adversarial attack methods on the JailbreakBench leaderboard, while being significantly more stable against defenses such as Smooth-LLM and Synonym Substitution, which are effective against existing all attacks on the leaderboard.

摘要: 众所周知，大型语言模型(LLM)容易受到精心设计的对抗性攻击或越狱，尽管使用安全微调方法与人类的偏好保持一致，但这些攻击或越狱会导致生成令人反感的内容。虽然输入令牌空间的大维度使得找到能够越狱这些模型的敌意提示是不可避免的，但我们的目标是评估安全的微调LLM对于自然提示是否安全，这些自然提示在语义上与有毒种子提示相关，在对齐后引起安全响应。我们惊讶地发现，GPT-4等流行的对齐LLM可以使用甚至不是以越狱为目标而精心设计的幼稚提示来进行攻击。此外，我们的经验表明，给定一个种子提示引起来自未对齐模型的有毒反应，一个人可以系统地生成几个语义相关的自然提示，从而可以越狱对齐的LLM。为此，我们提出了一种反应引导问题增强方法(REG-QA)来评估安全对齐LLM对自然提示的泛化，该方法首先使用未对齐LLM(Q到A)来生成给定种子问题的几个有毒答案，然后利用LLM来生成可能产生这些答案(A到Q)的问题。有趣的是，我们发现安全微调的LLM，如GPT-40，容易从不安全的内容产生自然的越狱问题(不否认)，因此可以用于后一步(A到Q)。我们获得了相当于/好于JailBreak排行榜上领先的对抗性攻击方法的攻击成功率，同时对Smooth-LLM和同义词替换等防御措施明显更加稳定，这些防御措施对排行榜上现有的所有攻击都有效。



## **25. A Taxonomy of System-Level Attacks on Deep Learning Models in Autonomous Vehicles**

自动驾驶汽车中深度学习模型的系统级攻击分类 cs.CR

**SubmitDate**: 2024-12-04    [abs](http://arxiv.org/abs/2412.04510v1) [paper-pdf](http://arxiv.org/pdf/2412.04510v1)

**Authors**: Masoud Jamshidiyan Tehrani, Jinhan Kim, Rosmael Zidane Lekeufack Foulefack, Alessandro Marchetto, Paolo Tonella

**Abstract**: The advent of deep learning and its astonishing performance in perception tasks, such as object recognition and classification, has enabled its usage in complex systems, including autonomous vehicles. On the other hand, deep learning models are susceptible to mis-predictions when small, adversarial changes are introduced into their input. Such mis-predictions can be triggered in the real world and can propagate to a failure of the entire system, as opposed to a localized mis-prediction. In recent years, a growing number of research works have investigated ways to mount attacks against autonomous vehicles that exploit deep learning components for perception tasks. Such attacks are directed toward elements of the environment where these systems operate and their effectiveness is assessed in terms of system-level failures triggered by them. There has been however no systematic attempt to analyze and categorize such attacks. In this paper, we present the first taxonomy of system-level attacks against autonomous vehicles. We constructed our taxonomy by first collecting 8,831 papers, then filtering them down to 1,125 candidates and eventually selecting a set of 19 highly relevant papers that satisfy all inclusion criteria. Then, we tagged them with taxonomy categories, involving three assessors per paper. The resulting taxonomy includes 12 top-level categories and several sub-categories. The taxonomy allowed us to investigate the attack features, the most attacked components, the underlying threat models, and the propagation chains from input perturbation to system-level failure. We distilled several lessons for practitioners and identified possible directions for future work for researchers.

摘要: 深度学习的出现及其在物体识别和分类等感知任务中的惊人表现，使其能够在包括自动驾驶汽车在内的复杂系统中使用。另一方面，当在输入中引入小的对抗性变化时，深度学习模型很容易被误判。这种错误预测可能在现实世界中被触发，并可能传播到整个系统的故障，而不是局部的错误预测。近年来，越来越多的研究工作研究了如何对自动驾驶车辆发动攻击，这些车辆利用深度学习组件进行感知任务。此类攻击针对这些系统运行的环境要素，并根据它们引发的系统级故障来评估其有效性。然而，尚未有系统地对这类攻击进行分析和分类。在本文中，我们首次提出了针对自动驾驶车辆的系统级攻击分类。我们构建了我们的分类法，首先收集了8831篇论文，然后将它们过滤到1125个候选论文，最终选择了一组具有高度相关性的19篇符合所有纳入标准的论文。然后，我们用分类类别对它们进行标记，每篇论文涉及三名评审员。得到的分类包括12个顶级类别和几个子类别。分类使我们能够调查攻击特征、受攻击最多的组件、潜在的威胁模型以及从输入扰动到系统级故障的传播链。我们为从业者提炼了几个教训，并为研究人员确定了未来工作的可能方向。



## **26. Testing Neural Network Verifiers: A Soundness Benchmark with Hidden Counterexamples**

测试神经网络验证器：具有隐藏反例的健全基准 cs.LG

Preprint

**SubmitDate**: 2024-12-04    [abs](http://arxiv.org/abs/2412.03154v1) [paper-pdf](http://arxiv.org/pdf/2412.03154v1)

**Authors**: Xingjian Zhou, Hongji Xu, Andy Xu, Zhouxing Shi, Cho-Jui Hsieh, Huan Zhang

**Abstract**: In recent years, many neural network (NN) verifiers have been developed to formally verify certain properties of neural networks such as robustness. Although many benchmarks have been constructed to evaluate the performance of NN verifiers, they typically lack a ground-truth for hard instances where no current verifier can verify and no counterexample can be found, which makes it difficult to check the soundness of a new verifier if it claims to verify hard instances which no other verifier can do. We propose to develop a soundness benchmark for NN verification. Our benchmark contains instances with deliberately inserted counterexamples while we also try to hide the counterexamples from regular adversarial attacks which can be used for finding counterexamples. We design a training method to produce neural networks with such hidden counterexamples. Our benchmark aims to be used for testing the soundness of NN verifiers and identifying falsely claimed verifiability when it is known that hidden counterexamples exist. We systematically construct our benchmark and generate instances across diverse model architectures, activation functions, input sizes, and perturbation radii. We demonstrate that our benchmark successfully identifies bugs in state-of-the-art NN verifiers, as well as synthetic bugs, providing a crucial step toward enhancing the reliability of testing NN verifiers. Our code is available at https://github.com/MVP-Harry/SoundnessBench and our benchmark is available at https://huggingface.co/datasets/SoundnessBench/SoundnessBench.

摘要: 近年来，已经发展了许多神经网络(NN)验证器来形式化地验证神经网络的某些特性，例如鲁棒性。虽然已经构建了许多基准来评估NN验证器的性能，但它们通常缺乏硬实例的基本事实，其中当前的验证器无法验证，也找不到反例，这使得如果一个新的验证器声称验证了其他验证器无法验证的硬实例，则很难检查其可靠性。我们建议开发一个用于神经网络验证的可靠性基准。我们的基准测试包含故意插入反例的实例，同时我们还试图隐藏反例，以避免常规的对抗性攻击，这些反例可用于查找反例。我们设计了一种训练方法来产生具有这种隐藏反例的神经网络。我们的基准测试旨在用于测试NN验证器的可靠性，并在已知存在隐藏反例的情况下识别虚假声明的可验证性。我们系统地构建我们的基准，并跨不同的模型架构、激活函数、输入大小和扰动半径生成实例。我们证明，我们的基准测试成功地识别了最先进的NN验证器中的错误，以及合成错误，为提高测试NN验证器的可靠性迈出了关键的一步。我们的代码可以在https://github.com/MVP-Harry/SoundnessBench上获得，我们的基准可以在https://huggingface.co/datasets/SoundnessBench/SoundnessBench.上获得



## **27. Pay Attention to the Robustness of Chinese Minority Language Models! Syllable-level Textual Adversarial Attack on Tibetan Script**

注意中国少数民族语言模型的稳健性！音节级文本对抗攻击 cs.CL

Revised Version; Accepted at ACL 2023 Workshop on TrustNLP

**SubmitDate**: 2024-12-04    [abs](http://arxiv.org/abs/2412.02323v2) [paper-pdf](http://arxiv.org/pdf/2412.02323v2)

**Authors**: Xi Cao, Dolma Dawa, Nuo Qun, Trashi Nyima

**Abstract**: The textual adversarial attack refers to an attack method in which the attacker adds imperceptible perturbations to the original texts by elaborate design so that the NLP (natural language processing) model produces false judgments. This method is also used to evaluate the robustness of NLP models. Currently, most of the research in this field focuses on English, and there is also a certain amount of research on Chinese. However, to the best of our knowledge, there is little research targeting Chinese minority languages. Textual adversarial attacks are a new challenge for the information processing of Chinese minority languages. In response to this situation, we propose a Tibetan syllable-level black-box textual adversarial attack called TSAttacker based on syllable cosine distance and scoring mechanism. And then, we conduct TSAttacker on six models generated by fine-tuning two PLMs (pre-trained language models) for three downstream tasks. The experiment results show that TSAttacker is effective and generates high-quality adversarial samples. In addition, the robustness of the involved models still has much room for improvement.

摘要: 文本对抗性攻击是指攻击者通过精心设计在原始文本中添加不可察觉的扰动，从而使NLP(自然语言处理)模型产生错误判断的攻击方法。该方法还被用于评价NLP模型的稳健性。目前，该领域的研究大多集中在英语方面，也有一定数量的汉语研究。然而，就我们所知，针对中国少数民族语言的研究很少。文本对抗性攻击是汉语少数民族语言信息处理面临的新挑战。针对这种情况，我们提出了一种基于音节余弦距离和评分机制的藏文音节级黑盒文本对抗攻击方法TSAtacker。然后，我们对微调三个下游任务的两个PLM(预先训练的语言模型)生成的六个模型进行了TSAttracker。实验结果表明，该算法是有效的，生成了高质量的对抗性样本。此外，所涉及的模型的稳健性还有很大的提升空间。



## **28. Less is More: A Stealthy and Efficient Adversarial Attack Method for DRL-based Autonomous Driving Policies**

少即是多：针对基于DRL的自动驾驶策略的隐形有效对抗攻击方法 cs.LG

**SubmitDate**: 2024-12-04    [abs](http://arxiv.org/abs/2412.03051v1) [paper-pdf](http://arxiv.org/pdf/2412.03051v1)

**Authors**: Junchao Fan, Xuyang Lei, Xiaolin Chang, Jelena Mišić, Vojislav B. Mišić

**Abstract**: Despite significant advancements in deep reinforcement learning (DRL)-based autonomous driving policies, these policies still exhibit vulnerability to adversarial attacks. This vulnerability poses a formidable challenge to the practical deployment of these policies in autonomous driving. Designing effective adversarial attacks is an indispensable prerequisite for enhancing the robustness of these policies. In view of this, we present a novel stealthy and efficient adversarial attack method for DRL-based autonomous driving policies. Specifically, we introduce a DRL-based adversary designed to trigger safety violations (e.g., collisions) by injecting adversarial samples at critical moments. We model the attack as a mixed-integer optimization problem and formulate it as a Markov decision process. Then, we train the adversary to learn the optimal policy for attacking at critical moments without domain knowledge. Furthermore, we introduce attack-related information and a trajectory clipping method to enhance the learning capability of the adversary. Finally, we validate our method in an unprotected left-turn scenario across different traffic densities. The experimental results show that our method achieves more than 90% collision rate within three attacks in most cases. Furthermore, our method achieves more than 130% improvement in attack efficiency compared to the unlimited attack method.

摘要: 尽管基于深度强化学习(DRL)的自主驾驶策略有了很大的进步，但这些策略仍然显示出对对手攻击的脆弱性。这一漏洞对这些政策在自动驾驶中的实际部署构成了巨大的挑战。设计有效的对抗性攻击是加强这些政策的稳健性的不可或缺的先决条件。鉴于此，我们提出了一种新的针对基于DRL的自主驾驶策略的隐身、高效的对抗攻击方法。具体地说，我们引入了一个基于DRL的对手，旨在通过在关键时刻注入对手样本来触发安全违规(例如，碰撞)。我们将攻击建模为混合整数优化问题，并将其描述为马尔可夫决策过程。然后，我们训练对手在没有领域知识的情况下学习在关键时刻攻击的最优策略。此外，我们还引入了与攻击相关的信息和轨迹裁剪方法来增强对手的学习能力。最后，我们在不同交通密度的无保护左转场景中验证了我们的方法。实验结果表明，在大多数情况下，该方法在三次攻击中都能达到90%以上的碰撞率。此外，与无限攻击方法相比，该方法在攻击效率上提高了130%以上。



## **29. AED-PADA:Improving Generalizability of Adversarial Example Detection via Principal Adversarial Domain Adaptation**

AED-PADA：通过主要对抗领域适应提高对抗示例检测的通用性 cs.CV

**SubmitDate**: 2024-12-04    [abs](http://arxiv.org/abs/2404.12635v2) [paper-pdf](http://arxiv.org/pdf/2404.12635v2)

**Authors**: Heqi Peng, Yunhong Wang, Ruijie Yang, Beichen Li, Rui Wang, Yuanfang Guo

**Abstract**: Adversarial example detection, which can be conveniently applied in many scenarios, is important in the area of adversarial defense. Unfortunately, existing detection methods suffer from poor generalization performance, because their training process usually relies on the examples generated from a single known adversarial attack and there exists a large discrepancy between the training and unseen testing adversarial examples. To address this issue, we propose a novel method, named Adversarial Example Detection via Principal Adversarial Domain Adaptation (AED-PADA). Specifically, our approach identifies the Principal Adversarial Domains (PADs), i.e., a combination of features of the adversarial examples generated by different attacks, which possesses a large portion of the entire adversarial feature space. Subsequently, we pioneer to exploit Multi-source Unsupervised Domain Adaptation in adversarial example detection, with PADs as the source domains. Experimental results demonstrate the superior generalization ability of our proposed AED-PADA. Note that this superiority is particularly achieved in challenging scenarios characterized by employing the minimal magnitude constraint for the perturbations.

摘要: 对抗性实例检测在对抗性防御领域具有重要意义，可以方便地应用于多种场景。遗憾的是，现有的检测方法泛化性能较差，因为它们的训练过程通常依赖于单一已知对手攻击产生的样本，并且训练样本和未见的测试对手样本之间存在着很大的差异。为了解决这一问题，我们提出了一种新的方法，称为基于主对抗性领域适应的对抗性范例检测(AED-PADA)。具体地说，我们的方法识别主要对抗性领域(PADS)，即由不同攻击产生的对抗性实例的特征的组合，它拥有整个对抗性特征空间的很大一部分。随后，我们以PADS为源域，首次将多源无监督域自适应应用于对抗性实例检测中。实验结果表明，本文提出的AED-PADA具有良好的泛化能力。请注意，这种优势尤其在具有挑战性的场景中实现，该场景的特征是对扰动采用最小幅度约束。



## **30. Exploiting the Uncoordinated Privacy Protections of Eye Tracking and VR Motion Data for Unauthorized User Identification**

利用眼动追踪和VR运动数据的不协调隐私保护来识别未经授权的用户 cs.HC

11 pages, 3 figures

**SubmitDate**: 2024-12-03    [abs](http://arxiv.org/abs/2411.12766v2) [paper-pdf](http://arxiv.org/pdf/2411.12766v2)

**Authors**: Samantha Aziz, Oleg Komogortsev

**Abstract**: Virtual reality (VR) devices use a variety of sensors to capture a rich body of user-generated data. This data can be misused by malicious parties to covertly infer information about the user. Privacy-enhancing techniques that seek to reduce the amount of personally identifying information in sensor data are typically developed for a subset of data streams that are available on the platform, without consideration for the auxiliary information that may be readily available from other sensors. In this paper, we evaluate whether body motion data can be used to circumvent the privacy protections applied to eye tracking data to enable user identification on a VR platform, and vice versa. We empirically show that eye tracking, headset tracking, and hand tracking data are not only informative for inferring user identity on their own, but contain complementary information that can increase the rate of successful user identification. Most importantly, we demonstrate that applying privacy protections to only a subset of the data available in VR can create an opportunity for an adversary to bypass those privacy protections by using other unprotected data streams that are available on the platform, performing a user identification attack as accurately as though a privacy mechanism was never applied. These results highlight a new privacy consideration at the intersection between eye tracking and VR, and emphasizes the need for privacy-enhancing techniques that address multiple technologies comprehensively.

摘要: 虚拟现实(VR)设备使用各种传感器来捕获丰富的用户生成的数据。这些数据可能会被恶意方滥用来秘密推断有关用户的信息。寻求减少传感器数据中的个人识别信息量的隐私增强技术通常是针对平台上可用的数据流的子集而开发的，而不考虑可能从其他传感器容易获得的辅助信息。在本文中，我们评估了身体运动数据是否可以用来规避应用于眼睛跟踪数据的隐私保护，以便在VR平台上进行用户识别，反之亦然。我们的经验表明，眼睛跟踪、耳机跟踪和手部跟踪数据不仅对推断用户身份本身具有信息性，而且包含补充信息，可以提高用户识别的成功率。最重要的是，我们证明，仅对VR中可用数据的子集应用隐私保护可以为攻击者创造机会，通过使用平台上可用的其他未受保护的数据流来绕过这些隐私保护，就像从未应用隐私机制一样准确地执行用户识别攻击。这些结果突显了眼睛跟踪和VR之间的交叉点上的一个新的隐私考虑，并强调了全面解决多种技术的隐私增强技术的必要性。



## **31. Gaussian Splatting Under Attack: Investigating Adversarial Noise in 3D Objects**

攻击下的高斯飞溅：调查3D对象中的对抗性噪音 cs.CV

Accepted to Safe Generative AI Workshop @ NeurIPS 2024:  https://neurips.cc/virtual/2024/workshop/84705

**SubmitDate**: 2024-12-03    [abs](http://arxiv.org/abs/2412.02803v1) [paper-pdf](http://arxiv.org/pdf/2412.02803v1)

**Authors**: Abdurrahman Zeybey, Mehmet Ergezer, Tommy Nguyen

**Abstract**: 3D Gaussian Splatting has advanced radiance field reconstruction, enabling high-quality view synthesis and fast rendering in 3D modeling. While adversarial attacks on object detection models are well-studied for 2D images, their impact on 3D models remains underexplored. This work introduces the Masked Iterative Fast Gradient Sign Method (M-IFGSM), designed to generate adversarial noise targeting the CLIP vision-language model. M-IFGSM specifically alters the object of interest by focusing perturbations on masked regions, degrading the performance of CLIP's zero-shot object detection capability when applied to 3D models. Using eight objects from the Common Objects 3D (CO3D) dataset, we demonstrate that our method effectively reduces the accuracy and confidence of the model, with adversarial noise being nearly imperceptible to human observers. The top-1 accuracy in original model renders drops from 95.4\% to 12.5\% for train images and from 91.2\% to 35.4\% for test images, with confidence levels reflecting this shift from true classification to misclassification, underscoring the risks of adversarial attacks on 3D models in applications such as autonomous driving, robotics, and surveillance. The significance of this research lies in its potential to expose vulnerabilities in modern 3D vision models, including radiance fields, prompting the development of more robust defenses and security measures in critical real-world applications.

摘要: 3D高斯飞溅具有先进的辐射场重建功能，可在3D建模中实现高质量的视图合成和快速渲染。虽然针对2D图像的目标检测模型的对抗性攻击已经得到了很好的研究，但它们对3D模型的影响仍然没有得到充分的研究。本文介绍了一种掩蔽迭代快速梯度符号方法(M-IFGSM)，用于产生针对片段视觉-语言模型的对抗性噪声。M-IFGSM通过将扰动聚焦到遮罩区域来具体改变感兴趣的对象，从而降低了当应用于3D模型时CLIP的零镜头对象检测能力的性能。使用CO3D数据集中的8个对象，我们证明了我们的方法有效地降低了模型的精度和置信度，而人类观察者几乎察觉不到对抗性噪声。原始模型的TOP-1准确率使训练图像从95.4\%下降到12.5\%，测试图像从91.2\%下降到35.4\%，置信度反映了从真实分类到错误分类的转变，突显了在自动驾驶、机器人和监控等应用中对3D模型进行对抗性攻击的风险。这项研究的意义在于它有可能揭露现代3D视觉模型中的漏洞，包括辐射场，促使在关键的现实世界应用中开发更强大的防御和安全措施。



## **32. Hijacking Vision-and-Language Navigation Agents with Adversarial Environmental Attacks**

通过对抗性环境攻击劫持视觉和语言导航代理 cs.CV

Accepted by WACV 2025

**SubmitDate**: 2024-12-03    [abs](http://arxiv.org/abs/2412.02795v1) [paper-pdf](http://arxiv.org/pdf/2412.02795v1)

**Authors**: Zijiao Yang, Xiangxi Shi, Eric Slyman, Stefan Lee

**Abstract**: Assistive embodied agents that can be instructed in natural language to perform tasks in open-world environments have the potential to significantly impact labor tasks like manufacturing or in-home care -- benefiting the lives of those who come to depend on them. In this work, we consider how this benefit might be hijacked by local modifications in the appearance of the agent's operating environment. Specifically, we take the popular Vision-and-Language Navigation (VLN) task as a representative setting and develop a whitebox adversarial attack that optimizes a 3D attack object's appearance to induce desired behaviors in pretrained VLN agents that observe it in the environment. We demonstrate that the proposed attack can cause VLN agents to ignore their instructions and execute alternative actions after encountering the attack object -- even for instructions and agent paths not considered when optimizing the attack. For these novel settings, we find our attacks can induce early-termination behaviors or divert an agent along an attacker-defined multi-step trajectory. Under both conditions, environmental attacks significantly reduce agent capabilities to successfully follow user instructions.

摘要: 可以用自然语言指导的辅助性具体化代理可以在开放世界环境中执行任务，这可能会显著影响制造或家庭护理等劳动任务--使依赖它们的人的生活受益。在这项工作中，我们考虑了如何通过对代理操作环境外观的局部修改来劫持这一好处。具体地说，我们以流行的视觉与语言导航(VLN)任务为代表，开发了一种白盒对抗性攻击，该攻击优化了3D攻击对象的外观，以诱导预先训练的VLN代理在环境中观察它所需的行为。我们证明了提出的攻击可以导致VLN代理在遇到攻击对象后忽略它们的指令并执行替代操作--即使是对于在优化攻击时没有考虑的指令和代理路径。对于这些新颖的设置，我们发现我们的攻击可以诱导提前终止行为，或者沿着攻击者定义的多步轨迹转移代理。在这两种情况下，环境攻击都会显著降低代理成功遵循用户指令的能力。



## **33. Defending Against Diverse Attacks in Federated Learning Through Consensus-Based Bi-Level Optimization**

通过基于启发的双层优化防御联邦学习中的各种攻击 cs.LG

**SubmitDate**: 2024-12-03    [abs](http://arxiv.org/abs/2412.02535v1) [paper-pdf](http://arxiv.org/pdf/2412.02535v1)

**Authors**: Nicolás García Trillos, Aditya Kumar Akash, Sixu Li, Konstantin Riedl, Yuhua Zhu

**Abstract**: Adversarial attacks pose significant challenges in many machine learning applications, particularly in the setting of distributed training and federated learning, where malicious agents seek to corrupt the training process with the goal of jeopardizing and compromising the performance and reliability of the final models. In this paper, we address the problem of robust federated learning in the presence of such attacks by formulating the training task as a bi-level optimization problem. We conduct a theoretical analysis of the resilience of consensus-based bi-level optimization (CB$^2$O), an interacting multi-particle metaheuristic optimization method, in adversarial settings. Specifically, we provide a global convergence analysis of CB$^2$O in mean-field law in the presence of malicious agents, demonstrating the robustness of CB$^2$O against a diverse range of attacks. Thereby, we offer insights into how specific hyperparameter choices enable to mitigate adversarial effects. On the practical side, we extend CB$^2$O to the clustered federated learning setting by proposing FedCB$^2$O, a novel interacting multi-particle system, and design a practical algorithm that addresses the demands of real-world applications. Extensive experiments demonstrate the robustness of the FedCB$^2$O algorithm against label-flipping attacks in decentralized clustered federated learning scenarios, showcasing its effectiveness in practical contexts.

摘要: 对抗性攻击在许多机器学习应用中带来了巨大的挑战，特别是在分布式训练和联合学习的环境中，恶意代理试图破坏训练过程，目的是危害和损害最终模型的性能和可靠性。在本文中，我们通过将训练任务描述为一个双层优化问题来解决存在此类攻击时的鲁棒联邦学习问题。我们对基于共识的双层优化算法(CB$^2$O)进行了理论分析，该算法是一种交互式多粒子元启发式优化方法，在对抗性环境下具有较强的抗攻击能力。具体地说，我们给出了在恶意代理存在的情况下，CB$^2$O在平均场法下的全局收敛分析，证明了CB$^2$O对各种攻击的健壮性。因此，我们对特定的超参数选择如何能够缓解对抗效应提供了见解。在实际应用方面，通过提出一种新颖的交互多粒子系统FedCB$^2$O，将CB$^2$O扩展到集群联邦学习环境中，并设计了一个满足实际应用需求的实用算法。大量的实验证明了FedCB$^2$O算法在分散的分簇联合学习场景中对标签翻转攻击的鲁棒性，并在实际环境中展示了其有效性。



## **34. TSCheater: Generating High-Quality Tibetan Adversarial Texts via Visual Similarity**

TSCheater：通过视觉相似性生成高质量的西藏对抗文本 cs.CL

Review Version; Submitted to ICASSP 2025

**SubmitDate**: 2024-12-03    [abs](http://arxiv.org/abs/2412.02371v1) [paper-pdf](http://arxiv.org/pdf/2412.02371v1)

**Authors**: Xi Cao, Quzong Gesang, Yuan Sun, Nuo Qun, Tashi Nyima

**Abstract**: Language models based on deep neural networks are vulnerable to textual adversarial attacks. While rich-resource languages like English are receiving focused attention, Tibetan, a cross-border language, is gradually being studied due to its abundant ancient literature and critical language strategy. Currently, there are several Tibetan adversarial text generation methods, but they do not fully consider the textual features of Tibetan script and overestimate the quality of generated adversarial texts. To address this issue, we propose a novel Tibetan adversarial text generation method called TSCheater, which considers the characteristic of Tibetan encoding and the feature that visually similar syllables have similar semantics. This method can also be transferred to other abugidas, such as Devanagari script. We utilize a self-constructed Tibetan syllable visual similarity database called TSVSDB to generate substitution candidates and adopt a greedy algorithm-based scoring mechanism to determine substitution order. After that, we conduct the method on eight victim language models. Experimentally, TSCheater outperforms existing methods in attack effectiveness, perturbation magnitude, semantic similarity, visual similarity, and human acceptance. Finally, we construct the first Tibetan adversarial robustness evaluation benchmark called AdvTS, which is generated by existing methods and proofread by humans.

摘要: 基于深度神经网络的语言模型容易受到文本攻击。在英语等资源丰富的语言受到关注的同时，藏语这一跨境语言也因其丰富的古代文献和批评的语言策略而逐渐被研究。目前，有几种藏文对抗性文本生成方法，但它们没有充分考虑藏文的文本特征，高估了生成的对抗性文本的质量。针对这一问题，我们提出了一种新的藏文对抗性文本生成方法TSCheater，该方法考虑了藏文编码的特点和视觉上相似音节具有相似语义的特点。这种方法也可以移植到其他ABUGIDAS，如天成文书。利用自行构建的藏文音节视觉相似度数据库TSVSDB生成替换候选，并采用基于贪婪算法的评分机制确定替换顺序。之后，我们在八个受害者语言模型上进行了该方法。实验结果表明，TSCheater在攻击效果、扰动幅度、语义相似度、视觉相似度和人类接受度等方面均优于现有方法。最后，我们构建了第一个藏文对手健壮性评估基准ADVTS，该基准由现有方法生成并由人工校对。



## **35. Multi-Granularity Tibetan Textual Adversarial Attack Method Based on Masked Language Model**

基于掩蔽语言模型的多粒度藏族文本对抗攻击方法 cs.CL

Revised Version; Accepted at WWW 2024 Workshop on SocialNLP

**SubmitDate**: 2024-12-03    [abs](http://arxiv.org/abs/2412.02343v1) [paper-pdf](http://arxiv.org/pdf/2412.02343v1)

**Authors**: Xi Cao, Nuo Qun, Quzong Gesang, Yulei Zhu, Trashi Nyima

**Abstract**: In social media, neural network models have been applied to hate speech detection, sentiment analysis, etc., but neural network models are susceptible to adversarial attacks. For instance, in a text classification task, the attacker elaborately introduces perturbations to the original texts that hardly alter the original semantics in order to trick the model into making different predictions. By studying textual adversarial attack methods, the robustness of language models can be evaluated and then improved. Currently, most of the research in this field focuses on English, and there is also a certain amount of research on Chinese. However, there is little research targeting Chinese minority languages. With the rapid development of artificial intelligence technology and the emergence of Chinese minority language models, textual adversarial attacks become a new challenge for the information processing of Chinese minority languages. In response to this situation, we propose a multi-granularity Tibetan textual adversarial attack method based on masked language models called TSTricker. We utilize the masked language models to generate candidate substitution syllables or words, adopt the scoring mechanism to determine the substitution order, and then conduct the attack method on several fine-tuned victim models. The experimental results show that TSTricker reduces the accuracy of the classification models by more than 28.70% and makes the classification models change the predictions of more than 90.60% of the samples, which has an evidently higher attack effect than the baseline method.

摘要: 在社交媒体中，神经网络模型已被应用于仇恨语音检测、情感分析等，但神经网络模型容易受到敌意攻击。例如，在文本分类任务中，攻击者精心地向原始文本引入扰动，这些扰动几乎不会改变原始语义，以便诱骗模型做出不同的预测。通过研究文本对抗性攻击方法，可以评估语言模型的健壮性，从而提高语言模型的稳健性。目前，该领域的研究大多集中在英语方面，也有一定数量的汉语研究。然而，针对中国少数民族语言的研究很少。随着人工智能技术的快速发展和中国少数民族语言模型的出现，文本对抗性攻击成为中国少数民族语言信息处理面临的新挑战。针对这种情况，我们提出了一种基于掩蔽语言模型的多粒度藏文文本对抗攻击方法TSTricker。我们利用掩蔽语言模型生成候选替换音节或单词，采用评分机制确定替换顺序，然后对多个微调的受害者模型进行攻击。实验结果表明，TSTricker使分类模型的准确率降低了28.70%以上，使90.60%以上的样本预测发生了变化，具有明显高于基线方法的攻击效果。



## **36. Sustainable Self-evolution Adversarial Training**

可持续自我进化对抗训练 cs.CV

Accepted to ACMMM 2024

**SubmitDate**: 2024-12-03    [abs](http://arxiv.org/abs/2412.02270v1) [paper-pdf](http://arxiv.org/pdf/2412.02270v1)

**Authors**: Wenxuan Wang, Chenglei Wang, Huihui Qi, Menghao Ye, Xuelin Qian, Peng Wang, Yanning Zhang

**Abstract**: With the wide application of deep neural network models in various computer vision tasks, there has been a proliferation of adversarial example generation strategies aimed at deeply exploring model security. However, existing adversarial training defense models, which rely on single or limited types of attacks under a one-time learning process, struggle to adapt to the dynamic and evolving nature of attack methods. Therefore, to achieve defense performance improvements for models in long-term applications, we propose a novel Sustainable Self-Evolution Adversarial Training (SSEAT) framework. Specifically, we introduce a continual adversarial defense pipeline to realize learning from various kinds of adversarial examples across multiple stages. Additionally, to address the issue of model catastrophic forgetting caused by continual learning from ongoing novel attacks, we propose an adversarial data replay module to better select more diverse and key relearning data. Furthermore, we design a consistency regularization strategy to encourage current defense models to learn more from previously trained ones, guiding them to retain more past knowledge and maintain accuracy on clean samples. Extensive experiments have been conducted to verify the efficacy of the proposed SSEAT defense method, which demonstrates superior defense performance and classification accuracy compared to competitors.

摘要: 随着深度神经网络模型在各种计算机视觉任务中的广泛应用，出现了大量旨在深入研究模型安全性的对抗性实例生成策略。然而，现有的对抗性训练防御模型依赖于一次性学习过程中单一或有限类型的攻击，难以适应攻击方法的动态和演化特性。因此，为了在长期应用中提高模型的防御性能，我们提出了一种新的可持续自进化对手训练(SSEAT)框架。具体地说，我们引入了一个持续的对抗性防御管道，以实现跨多个阶段从各种对抗性例子中学习。此外，为了解决持续不断地从新的攻击中学习导致模型灾难性遗忘的问题，我们提出了对抗性数据重放模块，以更好地选择更多样化和关键的重学习数据。此外，我们设计了一致性正则化策略，鼓励当前的防御模型从以前训练的模型中学习更多，引导它们保留更多过去的知识，并保持对干净样本的准确性。已经进行了大量的实验来验证所提出的SSEAT防御方法的有效性，与竞争对手相比，该方法表现出更好的防御性能和分类精度。



## **37. Guardian of the Ensembles: Introducing Pairwise Adversarially Robust Loss for Resisting Adversarial Attacks in DNN Ensembles**

合奏守护者：在DNN合奏中引入成对对抗稳健损失以抵抗对抗攻击 cs.LG

Accepted at IEEE/CVF Winter Conference on Applications of Computer  Vision (WACV 2025)

**SubmitDate**: 2024-12-03    [abs](http://arxiv.org/abs/2112.04948v2) [paper-pdf](http://arxiv.org/pdf/2112.04948v2)

**Authors**: Shubhi Shukla, Subhadeep Dalui, Manaar Alam, Shubhajit Datta, Arijit Mondal, Debdeep Mukhopadhyay, Partha Pratim Chakrabarti

**Abstract**: Adversarial attacks rely on transferability, where an adversarial example (AE) crafted on a surrogate classifier tends to mislead a target classifier. Recent ensemble methods demonstrate that AEs are less likely to mislead multiple classifiers in an ensemble. This paper proposes a new ensemble training using a Pairwise Adversarially Robust Loss (PARL) that by construction produces an ensemble of classifiers with diverse decision boundaries. PARL utilizes outputs and gradients of each layer with respect to network parameters in every classifier within the ensemble simultaneously. PARL is demonstrated to achieve higher robustness against black-box transfer attacks than previous ensemble methods as well as adversarial training without adversely affecting clean example accuracy. Extensive experiments using standard Resnet20, WideResnet28-10 classifiers demonstrate the robustness of PARL against state-of-the-art adversarial attacks. While maintaining similar clean accuracy and lesser training time, the proposed architecture has a 24.8% increase in robust accuracy ($\epsilon$ = 0.07) from the state-of-the art method.

摘要: 对抗性攻击依赖于可转移性，其中在代理分类器上制作的对抗性示例(AE)往往会误导目标分类器。最近的集成方法表明，AEs不太可能误导集成中的多个分类器。本文提出了一种新的集成训练方法，它使用成对的对抗性鲁棒损失(PAL)，通过构造产生具有不同决策边界的分类器集成。PARL同时利用每一层相对于集成内每个分类器中的网络参数的输出和梯度。与以前的集成方法以及对抗性训练相比，PAL在抵抗黑盒转移攻击以及对抗性训练方面具有更高的稳健性，而不会对干净样本的准确性产生不利影响。使用标准的Resnet20、WideResnet28-10分类器进行的大量实验证明了PARL对最先进的对手攻击的健壮性。在保持相似的清洁精度和更少的训练时间的同时，所提出的结构与最新的方法相比，稳健精度提高了24.8%($=0.07)。



## **38. Privacy-Preserving Federated Learning via Homomorphic Adversarial Networks**

通过同形对抗网络保护隐私的联邦学习 cs.CR

**SubmitDate**: 2024-12-03    [abs](http://arxiv.org/abs/2412.01650v2) [paper-pdf](http://arxiv.org/pdf/2412.01650v2)

**Authors**: Wenhan Dong, Chao Lin, Xinlei He, Xinyi Huang, Shengmin Xu

**Abstract**: Privacy-preserving federated learning (PPFL) aims to train a global model for multiple clients while maintaining their data privacy. However, current PPFL protocols exhibit one or more of the following insufficiencies: considerable degradation in accuracy, the requirement for sharing keys, and cooperation during the key generation or decryption processes. As a mitigation, we develop the first protocol that utilizes neural networks to implement PPFL, as well as incorporating an Aggregatable Hybrid Encryption scheme tailored to the needs of PPFL. We name these networks as Homomorphic Adversarial Networks (HANs) which demonstrate that neural networks are capable of performing tasks similar to multi-key homomorphic encryption (MK-HE) while solving the problems of key distribution and collaborative decryption. Our experiments show that HANs are robust against privacy attacks. Compared with non-private federated learning, experiments conducted on multiple datasets demonstrate that HANs exhibit a negligible accuracy loss (at most 1.35%). Compared to traditional MK-HE schemes, HANs increase encryption aggregation speed by 6,075 times while incurring a 29.2 times increase in communication overhead.

摘要: 隐私保护联合学习(PPFL)旨在为多个客户训练一个全局模型，同时保持他们的数据隐私。然而，当前的PPFL协议表现出以下一个或多个不足：在准确性方面显著降低、对共享密钥的要求以及在密钥生成或解密过程中的合作。作为缓解，我们开发了第一个利用神经网络来实现PPFL的协议，以及结合了针对PPFL需求定制的可聚集混合加密方案。我们将这些网络命名为同态对抗网络(HANS)，它证明了神经网络能够执行类似于多密钥同态加密(MK-HE)的任务，同时解决了密钥分配和协作解密问题。我们的实验表明，HANS对隐私攻击具有很强的抵抗力。与非私有联合学习相比，在多个数据集上进行的实验表明，HANS的准确率损失可以忽略不计(最多1.35%)。与传统的MK-HE方案相比，HANS的加密聚合速度提高了6,075倍，而通信开销增加了29.2倍。



## **39. Investigating Privacy Leakage in Dimensionality Reduction Methods via Reconstruction Attack**

通过重建攻击调查虚拟性减少方法中的隐私泄露 cs.CR

Major revision

**SubmitDate**: 2024-12-03    [abs](http://arxiv.org/abs/2408.17151v2) [paper-pdf](http://arxiv.org/pdf/2408.17151v2)

**Authors**: Chayadon Lumbut, Donlapark Ponnoprat

**Abstract**: This study investigates privacy leakage in dimensionality reduction methods through a novel machine learning-based reconstruction attack. Employing an informed adversary threat model, we develop a neural network capable of reconstructing high-dimensional data from low-dimensional embeddings.   We evaluate six popular dimensionality reduction techniques: PCA, sparse random projection (SRP), multidimensional scaling (MDS), Isomap, t-SNE, and UMAP. Using both MNIST and NIH Chest X-ray datasets, we perform a qualitative analysis to identify key factors affecting reconstruction quality. Furthermore, we assess the effectiveness of an additive noise mechanism in mitigating these reconstruction attacks. Our experimental results on both datasets reveal that the attack is effective against deterministic methods (PCA and Isomap), but ineffective against methods that employ random initialization (SRP, MDS, t-SNE and UMAP). When adding the images with large noises before performing PCA or Isomap, the attack produced severely distorted reconstructions. In contrast, for the other four methods, the reconstructions still show some recognizable features, though they bear little resemblance to the original images.

摘要: 通过一种新的基于机器学习的重构攻击，研究了降维方法中的隐私泄漏问题。利用一个知情的对手威胁模型，我们开发了一个能够从低维嵌入中重构高维数据的神经网络。我们评估了六种流行的降维技术：PCA、稀疏随机投影(SRP)、多维缩放(MDS)、ISOMAP、t-SNE和UMAP。使用MNIST和NIH胸部X光数据集，我们进行了定性分析，以确定影响重建质量的关键因素。此外，我们评估了加性噪声机制在缓解这些重建攻击方面的有效性。我们在两个数据集上的实验结果表明，该攻击对确定性方法(PCA和ISOMAP)有效，但对采用随机初始化的方法(SRP、MDS、t-SNE和UMAP)无效。当在执行PCA或ISOMAP之前添加噪声较大的图像时，该攻击会产生严重失真的重建结果。相比之下，对于其他四种方法，重建图像仍然显示出一些可识别的特征，尽管它们与原始图像几乎没有相似之处。



## **40. Underload: Defending against Latency Attacks for Object Detectors on Edge Devices**

欠载：抵御边缘设备上对象检测器的延迟攻击 cs.CV

**SubmitDate**: 2024-12-03    [abs](http://arxiv.org/abs/2412.02171v1) [paper-pdf](http://arxiv.org/pdf/2412.02171v1)

**Authors**: Tianyi Wang, Zichen Wang, Cong Wang, Yuanchao Shu, Ruilong Deng, Peng Cheng, Jiming Chen

**Abstract**: Object detection is a fundamental enabler for many real-time downstream applications such as autonomous driving, augmented reality and supply chain management. However, the algorithmic backbone of neural networks is brittle to imperceptible perturbations in the system inputs, which were generally known as misclassifying attacks. By targeting the real-time processing capability, a new class of latency attacks are reported recently. They exploit new attack surfaces in object detectors by creating a computational bottleneck in the post-processing module, that leads to cascading failure and puts the real-time downstream tasks at risks. In this work, we take an initial attempt to defend against this attack via background-attentive adversarial training that is also cognizant of the underlying hardware capabilities. We first draw system-level connections between latency attack and hardware capacity across heterogeneous GPU devices. Based on the particular adversarial behaviors, we utilize objectness loss as a proxy and build background attention into the adversarial training pipeline, and achieve a reasonable balance between clean and robust accuracy. The extensive experiments demonstrate the defense effectiveness of restoring real-time processing capability from $13$ FPS to $43$ FPS on Jetson Orin NX, with a better trade-off between the clean and robust accuracy.

摘要: 目标检测是自动驾驶、增强现实和供应链管理等许多实时下游应用的基本使能。然而，神经网络的算法主干对系统输入中的不可察觉的扰动是脆弱的，这种扰动通常被称为误分类攻击。以实时处理能力为目标，最近报道了一类新的延迟攻击。它们通过在后处理模块中创建计算瓶颈来利用对象检测器中的新攻击面，从而导致级联故障并使实时下游任务处于危险之中。在这项工作中，我们初步尝试通过背景专注的对手训练来防御这种攻击，该训练也认识到潜在的硬件能力。我们首先在系统级将延迟攻击与跨不同类型的GPU设备的硬件容量联系起来。基于特定的对抗性行为，我们利用客观性损失作为代理，在对抗性训练流水线中加入背景注意，在干净和健壮的准确率之间取得合理的平衡。广泛的实验证明了在Jetson Orin NX上将实时处理能力从13美元FPS恢复到43美元FPS的防御效果，并在干净和健壮的准确性之间进行了更好的权衡。



## **41. Compromising the Intelligence of Modern DNNs: On the Effectiveness of Targeted RowPress**

损害现代DNN的智能：关于定向RowPress的有效性 cs.AR

8 Pages, 7 Figures, 1 Table

**SubmitDate**: 2024-12-03    [abs](http://arxiv.org/abs/2412.02156v1) [paper-pdf](http://arxiv.org/pdf/2412.02156v1)

**Authors**: Ranyang Zhou, Jacqueline T. Liu, Sabbir Ahmed, Shaahin Angizi, Adnan Siraj Rakin

**Abstract**: Recent advancements in side-channel attacks have revealed the vulnerability of modern Deep Neural Networks (DNNs) to malicious adversarial weight attacks. The well-studied RowHammer attack has effectively compromised DNN performance by inducing precise and deterministic bit-flips in the main memory (e.g., DRAM). Similarly, RowPress has emerged as another effective strategy for flipping targeted bits in DRAM. However, the impact of RowPress on deep learning applications has yet to be explored in the existing literature, leaving a fundamental research question unanswered: How does RowPress compare to RowHammer in leveraging bit-flip attacks to compromise DNN performance? This paper is the first to address this question and evaluate the impact of RowPress on DNN applications. We conduct a comparative analysis utilizing a novel DRAM-profile-aware attack designed to capture the distinct bit-flip patterns caused by RowHammer and RowPress. Eleven widely-used DNN architectures trained on different benchmark datasets deployed on a Samsung DRAM chip conclusively demonstrate that they suffer from a drastically more rapid performance degradation under the RowPress attack compared to RowHammer. The difference in the underlying attack mechanism of RowHammer and RowPress also renders existing RowHammer mitigation mechanisms ineffective under RowPress. As a result, RowPress introduces a new vulnerability paradigm for DNN compute platforms and unveils the urgent need for corresponding protective measures.

摘要: 旁路攻击的最新进展揭示了现代深度神经网络(DNN)对恶意对抗性权重攻击的脆弱性。经过充分研究的RowHammer攻击通过在主存储器(例如，DRAM)中诱导精确和确定的比特翻转，有效地损害了DNN的性能。同样，RowPress已成为在DRAM中翻转目标位的另一种有效策略。然而，RowPress对深度学习应用的影响还没有在现有的文献中得到探索，留下了一个基本的研究问题没有得到回答：在利用比特翻转攻击来损害DNN性能方面，RowPress与RowHammer相比如何？本文首次解决了这一问题，并评估了RowPress对DNN应用的影响。我们利用一种新型的DRAM简档感知攻击进行了比较分析，该攻击旨在捕获RowHammer和RowPress造成的不同的位翻转模式。11个广泛使用的DNN架构在部署在三星DRAM芯片上的不同基准数据集上进行了训练，最终证明，与RowHammer相比，它们在RowPress攻击下的性能降级要快得多。RowHammer和RowPress底层攻击机制的差异也使得现有的RowHammer缓解机制在RowPress下无效。因此，RowPress为DNN计算平台引入了新的漏洞范例，并揭示了相应防护措施的迫切需要。



## **42. Dynamic Adversarial Attacks on Autonomous Driving Systems**

对自动驾驶系统的动态对抗攻击 cs.RO

**SubmitDate**: 2024-12-03    [abs](http://arxiv.org/abs/2312.06701v3) [paper-pdf](http://arxiv.org/pdf/2312.06701v3)

**Authors**: Amirhosein Chahe, Chenan Wang, Abhishek Jeyapratap, Kaidi Xu, Lifeng Zhou

**Abstract**: This paper introduces an attacking mechanism to challenge the resilience of autonomous driving systems. Specifically, we manipulate the decision-making processes of an autonomous vehicle by dynamically displaying adversarial patches on a screen mounted on another moving vehicle. These patches are optimized to deceive the object detection models into misclassifying targeted objects, e.g., traffic signs. Such manipulation has significant implications for critical multi-vehicle interactions such as intersection crossing and lane changing, which are vital for safe and efficient autonomous driving systems. Particularly, we make four major contributions. First, we introduce a novel adversarial attack approach where the patch is not co-located with its target, enabling more versatile and stealthy attacks. Moreover, our method utilizes dynamic patches displayed on a screen, allowing for adaptive changes and movement, enhancing the flexibility and performance of the attack. To do so, we design a Screen Image Transformation Network (SIT-Net), which simulates environmental effects on the displayed images, narrowing the gap between simulated and real-world scenarios. Further, we integrate a positional loss term into the adversarial training process to increase the success rate of the dynamic attack. Finally, we shift the focus from merely attacking perceptual systems to influencing the decision-making algorithms of self-driving systems. Our experiments demonstrate the first successful implementation of such dynamic adversarial attacks in real-world autonomous driving scenarios, paving the way for advancements in the field of robust and secure autonomous driving.

摘要: 本文介绍了一种攻击机制来挑战自动驾驶系统的弹性。具体地说，我们通过在安装在另一辆移动车辆上的屏幕上动态显示敌对补丁来操纵自动车辆的决策过程。这些补丁被优化以欺骗对象检测模型误分类目标对象，例如交通标志。这种操纵对交叉路口和换道等关键的多车辆相互作用具有重要影响，而这些对安全高效的自动驾驶系统至关重要。特别是，我们做出了四大贡献。首先，我们引入了一种新颖的对抗性攻击方法，其中补丁不与目标位于同一位置，从而实现了更多功能和隐蔽的攻击。此外，我们的方法利用屏幕上显示的动态补丁，允许自适应变化和移动，增强了攻击的灵活性和性能。为此，我们设计了一个屏幕图像转换网络(SIT-Net)，它模拟了环境对显示图像的影响，缩小了模拟场景和真实场景之间的差距。此外，我们还将位置损失项融入到对抗性训练过程中，以提高动态攻击的成功率。最后，我们将重点从仅仅攻击感知系统转移到影响自动驾驶系统的决策算法。我们的实验首次成功地在真实世界的自动驾驶场景中实现了这种动态对抗性攻击，为稳健和安全的自动驾驶领域的进步铺平了道路。



## **43. Reactive Synthesis of Sensor Revealing Strategies in Hypergames on Graphs**

图上超游戏中传感器揭示策略的反应式综合 cs.GT

17 pages, 5 figures, 2 tables, submitted to Automatica

**SubmitDate**: 2024-12-02    [abs](http://arxiv.org/abs/2412.01975v1) [paper-pdf](http://arxiv.org/pdf/2412.01975v1)

**Authors**: Sumukha Udupa, Ahmed Hemida, Charles A. Kamhoua, Jie Fu

**Abstract**: In many security applications of cyber-physical systems, a system designer must guarantee that critical missions are satisfied against attacks in the sensors and actuators of the CPS. Traditional security design of CPSs often assume that attackers have complete knowledge of the system. In this article, we introduce a class of deception techniques and study how to leverage asymmetric information created by deception to strengthen CPS security. Consider an adversarial interaction between a CPS defender and an attacker, who can perform sensor jamming attacks. To mitigate such attacks, the defender introduces asymmetrical information by deploying a "hidden sensor," whose presence is initially undisclosed but can be revealed if queried. We introduce hypergames on graphs to model this game with asymmetric information. Building on the solution concept called subjective rationalizable strategies in hypergames, we identify two stages in the game: An initial game stage where the defender commits to a strategy perceived rationalizable by the attacker until he deviates from the equilibrium in the attacker's perceptual game; Upon the deviation, a delay-attack game stage starts where the defender plays against the attacker, who has a bounded delay in attacking the sensor being revealed. Based on backward induction, we develop an algorithm that determines, for any given state, if the defender can benefit from hiding a sensor and revealing it later. If the answer is affirmative, the algorithm outputs a sensor revealing strategy to determine when to reveal the sensor during dynamic interactions. We demonstrate the effectiveness of our deceptive strategies through two case studies related to CPS security applications.

摘要: 在许多网络物理系统的安全应用中，系统设计者必须保证满足关键任务对CPS传感器和执行器的攻击。传统的CPSS安全设计往往假设攻击者完全了解系统。在本文中，我们介绍了一类欺骗技术，并研究了如何利用欺骗产生的不对称信息来加强CPS的安全性。考虑CPS防御者和攻击者之间的对抗性交互，攻击者可以执行传感器干扰攻击。为了减轻这种攻击，防御者通过部署“隐藏传感器”来引入不对称信息，该传感器的存在最初是不公开的，但如果被询问就可以被揭示。我们引入图上的超对策来模拟信息不对称的这一博弈。在超级博弈中称为主观合理化策略的解决方案概念的基础上，我们确定了博弈的两个阶段：初始博弈阶段，防御者致力于攻击者感知的合理化策略，直到他偏离攻击者感知游戏中的均衡；一旦偏离，延迟攻击博弈阶段开始，防御者对抗攻击者，攻击者在攻击传感器时具有有限的延迟。基于反向归纳法，我们开发了一种算法，对于任何给定的状态，确定防御者是否可以从隐藏传感器并稍后揭示它中受益。如果答案是肯定的，则算法输出传感器显示策略，以确定在动态交互期间何时显示传感器。我们通过两个与CPS安全应用相关的案例研究来证明我们的欺骗策略的有效性。



## **44. Topology-Based Reconstruction Prevention for Decentralised Learning**

去中心化学习的基于布局的重建预防 cs.CR

14 pages, 19 figures, for associated experiment source code see  doi:10.4121/21572601.v2

**SubmitDate**: 2024-12-02    [abs](http://arxiv.org/abs/2312.05248v3) [paper-pdf](http://arxiv.org/pdf/2312.05248v3)

**Authors**: Florine W. Dekker, Zekeriya Erkin, Mauro Conti

**Abstract**: Decentralised learning has recently gained traction as an alternative to federated learning in which both data and coordination are distributed. To preserve the confidentiality of users' data, decentralised learning relies on differential privacy, multi-party computation, or both. However, running multiple privacy-preserving summations in sequence may allow adversaries to perform reconstruction attacks. Current reconstruction countermeasures either cannot trivially be adapted to the distributed setting, or add excessive amounts of noise.   In this work, we first show that passive honest-but-curious adversaries can infer other users' private data after several privacy-preserving summations. For example, in subgraphs with 18 users, we show that only three passive honest-but-curious adversaries succeed at reconstructing private data 11.0% of the time, requiring an average of 8.8 summations per adversary. The success rate depends only on the adversaries' direct neighbourhood, and is independent of the size of the full network. We consider weak adversaries that do not control the graph topology, cannot exploit the summation's inner workings, and do not have auxiliary knowledge; and show that these adversaries can still infer private data.   We analyse how reconstruction relates to topology and propose the first topology-based decentralised defence against reconstruction attacks. We show that reconstruction requires a number of adversaries linear in the length of the network's shortest cycle. Consequently, exact attacks over privacy-preserving summations are impossible in acyclic networks.   Our work is a stepping stone for a formal theory of topology-based decentralised reconstruction defences. Such a theory would generalise our countermeasure beyond summation, define confidentiality in terms of entropy, and describe the interactions with (topology-aware) differential privacy.

摘要: 分散式学习作为联邦学习的替代方案最近获得了吸引力，在联合学习中，数据和协调都是分布式的。为了保护用户数据的机密性，分散学习依赖于差异隐私、多方计算或两者兼而有之。但是，按顺序运行多个隐私保护摘要可能会允许攻击者执行重建攻击。当前的重建对策要么不能简单地适应分布式设置，要么增加了过多的噪声。在这项工作中，我们首先证明了被动诚实但好奇的对手可以在几次隐私保护汇总后推断出其他用户的私人数据。例如，在具有18个用户的子图中，我们表明只有三个被动的诚实但好奇的对手在11.0%的时间内成功重建私人数据，每个对手平均需要8.8次求和。成功率仅取决于对手的直接邻居，与整个网络的规模无关。我们认为弱对手不能控制图的拓扑，不能利用求和的内部工作，并且没有辅助知识；并且表明这些对手仍然可以推断私有数据。我们分析了重构与拓扑的关系，提出了第一种基于拓扑的分布式防御重构攻击的方法。我们证明了重构需要若干个与网络的最短周期长度成线性关系的对手。因此，在非循环网络中，不可能对隐私保护求和进行准确的攻击。我们的工作是基于拓扑学的分布式重建防御的正式理论的垫脚石。这样的理论将概括我们的对策，超越总和，用熵来定义机密性，并描述与(拓扑感知的)差异隐私的相互作用。



## **45. Effectiveness of L2 Regularization in Privacy-Preserving Machine Learning**

L2正规化在隐私保护机器学习中的有效性 cs.LG

**SubmitDate**: 2024-12-02    [abs](http://arxiv.org/abs/2412.01541v1) [paper-pdf](http://arxiv.org/pdf/2412.01541v1)

**Authors**: Nikolaos Chandrinos, Iliana Loi, Panagiotis Zachos, Ioannis Symeonidis, Aristotelis Spiliotis, Maria Panou, Konstantinos Moustakas

**Abstract**: Artificial intelligence, machine learning, and deep learning as a service have become the status quo for many industries, leading to the widespread deployment of models that handle sensitive data. Well-performing models, the industry seeks, usually rely on a large volume of training data. However, the use of such data raises serious privacy concerns due to the potential risks of leaks of highly sensitive information. One prominent threat is the Membership Inference Attack, where adversaries attempt to deduce whether a specific data point was used in a model's training process. An adversary's ability to determine an individual's presence represents a significant privacy threat, especially when related to a group of users sharing sensitive information. Hence, well-designed privacy-preserving machine learning solutions are critically needed in the industry. In this work, we compare the effectiveness of L2 regularization and differential privacy in mitigating Membership Inference Attack risks. Even though regularization techniques like L2 regularization are commonly employed to reduce overfitting, a condition that enhances the effectiveness of Membership Inference Attacks, their impact on mitigating these attacks has not been systematically explored.

摘要: 人工智能、机器学习和深度学习即服务已成为许多行业的现状，导致处理敏感数据的模型得到广泛部署。该行业寻求的表现良好的模型通常依赖于大量的培训数据。然而，由于高度敏感信息的潜在泄露风险，此类数据的使用引发了严重的隐私问题。一个突出的威胁是成员推理攻击，攻击者试图推断模型的训练过程中是否使用了特定的数据点。对手确定个人存在的能力是对隐私的重大威胁，特别是在与一组共享敏感信息的用户相关的时候。因此，行业迫切需要设计良好的隐私保护机器学习解决方案。在这项工作中，我们比较了L2正则化和差异隐私在降低成员推理攻击风险方面的有效性。尽管像L2正则化这样的正则化技术通常被用来减少过拟合，这是一种增强隶属度推理攻击有效性的条件，但它们对缓解这些攻击的影响尚未被系统地探讨。



## **46. Traversing the Subspace of Adversarial Patches**

穿越对抗补丁的子空间 cs.CV

**SubmitDate**: 2024-12-02    [abs](http://arxiv.org/abs/2412.01527v1) [paper-pdf](http://arxiv.org/pdf/2412.01527v1)

**Authors**: Jens Bayer, Stefan Becker, David Münch, Michael Arens, Jürgen Beyerer

**Abstract**: Despite ongoing research on the topic of adversarial examples in deep learning for computer vision, some fundamentals of the nature of these attacks remain unclear. As the manifold hypothesis posits, high-dimensional data tends to be part of a low-dimensional manifold. To verify the thesis with adversarial patches, this paper provides an analysis of a set of adversarial patches and investigates the reconstruction abilities of three different dimensionality reduction methods. Quantitatively, the performance of reconstructed patches in an attack setting is measured and the impact of sampled patches from the latent space during adversarial training is investigated. The evaluation is performed on two publicly available datasets for person detection. The results indicate that more sophisticated dimensionality reduction methods offer no advantages over a simple principal component analysis.

摘要: 尽管人们正在对计算机视觉深度学习中的对抗示例主题进行研究，但这些攻击性质的一些基本原理仍然不清楚。正如多管齐下假设的那样，多维数据往往是低维多管齐下的一部分。为了用对抗性补丁来验证论文，本文分析了一组对抗性补丁，并研究了三种不同维度约简方法的重建能力。量化地测量攻击环境中重建补丁的性能，并研究对抗训练期间来自潜在空间的采样补丁的影响。该评估是在两个公开可用的数据集上执行的，用于人员检测。结果表明，更复杂的维度约简方法与简单的主成分分析相比并没有任何优势。



## **47. Adversarial Attacks on Hyperbolic Networks**

双曲网络的对抗攻击 cs.LG

**SubmitDate**: 2024-12-02    [abs](http://arxiv.org/abs/2412.01495v1) [paper-pdf](http://arxiv.org/pdf/2412.01495v1)

**Authors**: Max van Spengler, Jan Zahálka, Pascal Mettes

**Abstract**: As hyperbolic deep learning grows in popularity, so does the need for adversarial robustness in the context of such a non-Euclidean geometry. To this end, this paper proposes hyperbolic alternatives to the commonly used FGM and PGD adversarial attacks. Through interpretable synthetic benchmarks and experiments on existing datasets, we show how the existing and newly proposed attacks differ. Moreover, we investigate the differences in adversarial robustness between Euclidean and fully hyperbolic networks. We find that these networks suffer from different types of vulnerabilities and that the newly proposed hyperbolic attacks cannot address these differences. Therefore, we conclude that the shifts in adversarial robustness are due to the models learning distinct patterns resulting from their different geometries.

摘要: 随着双曲深度学习越来越受欢迎，在这种非欧几里德几何的背景下对对抗鲁棒性的需求也越来越大。为此，本文提出了常用的FGM和PVD对抗攻击的双曲替代方案。通过对现有数据集的可解释合成基准和实验，我们展示了现有和新提出的攻击的不同之处。此外，我们还研究了欧几里得网络和全双曲网络之间对抗鲁棒性的差异。我们发现这些网络存在不同类型的漏洞，而新提出的双曲攻击无法解决这些差异。因此，我们得出的结论是，对抗鲁棒性的变化是由于模型学习了不同的几何形状而产生的不同模式。



## **48. DiffPatch: Generating Customizable Adversarial Patches using Diffusion Model**

迪夫补丁：使用扩散模型生成可定制的对抗补丁 cs.CV

**SubmitDate**: 2024-12-02    [abs](http://arxiv.org/abs/2412.01440v1) [paper-pdf](http://arxiv.org/pdf/2412.01440v1)

**Authors**: Zhixiang Wang, Guangnan Ye, Xiaosen Wang, Siheng Chen, Zhibo Wang, Xingjun Ma, Yu-Gang Jiang

**Abstract**: Physical adversarial patches printed on clothing can easily allow individuals to evade person detectors. However, most existing adversarial patch generation methods prioritize attack effectiveness over stealthiness, resulting in patches that are aesthetically unpleasing. Although existing methods using generative adversarial networks or diffusion models can produce more natural-looking patches, they often struggle to balance stealthiness with attack effectiveness and lack flexibility for user customization. To address these challenges, we propose a novel diffusion-based customizable patch generation framework termed DiffPatch, specifically tailored for creating naturalistic and customizable adversarial patches. Our approach enables users to utilize a reference image as the source, rather than starting from random noise, and incorporates masks to craft naturalistic patches of various shapes, not limited to squares. To prevent the original semantics from being lost during the diffusion process, we employ Null-text inversion to map random noise samples to a single input image and generate patches through Incomplete Diffusion Optimization (IDO). Notably, while maintaining a natural appearance, our method achieves a comparable attack performance to state-of-the-art non-naturalistic patches when using similarly sized attacks. Using DiffPatch, we have created a physical adversarial T-shirt dataset, AdvPatch-1K, specifically targeting YOLOv5s. This dataset includes over a thousand images across diverse scenarios, validating the effectiveness of our attack in real-world environments. Moreover, it provides a valuable resource for future research.

摘要: 衣服上印有敌意的物理补丁可以很容易地让个人躲避个人探测器。然而，大多数现有的对抗性补丁生成方法将攻击效率置于隐蔽性之上，导致生成的补丁在美学上令人不快。虽然现有的方法使用生成性对抗网络或扩散模型可以产生看起来更自然的补丁，但它们往往难以平衡隐蔽性和攻击有效性，并且缺乏用户定制的灵活性。为了应对这些挑战，我们提出了一种新的基于扩散的可定制补丁生成框架DiffPatch，该框架专门用于创建自然的和可定制的对抗性补丁。我们的方法使用户能够利用参考图像作为源，而不是从随机噪声开始，并结合蒙版来制作各种形状的自然斑块，而不限于正方形。为了避免在扩散过程中丢失原始语义，我们使用空文本反转将随机噪声样本映射到单一输入图像，并通过不完全扩散优化(IDO)生成斑块。值得注意的是，在保持自然外观的同时，我们的方法在使用类似大小的攻击时，实现了与最先进的非自然主义补丁相当的攻击性能。使用DiffPatch，我们已经创建了一个物理对手T恤数据集AdvPatch-1K，专门针对YOLOv5。该数据集包括1000多张不同场景的图像，验证了我们的攻击在真实环境中的有效性。此外，它还为今后的研究提供了宝贵的资源。



## **49. Behavior Backdoor for Deep Learning Models**

深度学习模型的行为后门 cs.LG

**SubmitDate**: 2024-12-02    [abs](http://arxiv.org/abs/2412.01369v1) [paper-pdf](http://arxiv.org/pdf/2412.01369v1)

**Authors**: Jiakai Wang, Pengfei Zhang, Renshuai Tao, Jian Yang, Hao Liu, Xianglong Liu, Yunchao Wei, Yao Zhao

**Abstract**: The various post-processing methods for deep-learning-based models, such as quantification, pruning, and fine-tuning, play an increasingly important role in artificial intelligence technology, with pre-train large models as one of the main development directions. However, this popular series of post-processing behaviors targeting pre-training deep models has become a breeding ground for new adversarial security issues. In this study, we take the first step towards ``behavioral backdoor'' attack, which is defined as a behavior-triggered backdoor model training procedure, to reveal a new paradigm of backdoor attacks. In practice, we propose the first pipeline of implementing behavior backdoor, i.e., the Quantification Backdoor (QB) attack, upon exploiting model quantification method as the set trigger. Specifically, to adapt the optimization goal of behavior backdoor, we introduce the behavior-driven backdoor object optimizing method by a bi-target behavior backdoor training loss, thus we could guide the poisoned model optimization direction. To update the parameters across multiple models, we adopt the address-shared backdoor model training, thereby the gradient information could be utilized for multimodel collaborative optimization. Extensive experiments have been conducted on different models, datasets, and tasks, demonstrating the effectiveness of this novel backdoor attack and its potential application threats.

摘要: 基于深度学习的模型的各种后处理方法，如量化、剪枝、微调，在人工智能技术中发挥着越来越重要的作用，预训练大模型是主要的发展方向之一。然而，这种针对训练前深度模型的流行的后处理行为已经成为新的对抗性安全问题的温床。在本研究中，我们向行为后门攻击迈出了第一步，它被定义为行为触发的后门模型训练过程，以揭示后门攻击的新范式。在实践中，我们利用模型量化方法作为集合触发器，提出了实现行为后门的第一条管道，即量化后门(QB)攻击。具体地说，为了适应行为后门的优化目标，通过双目标行为后门训练损失引入行为驱动的后门对象优化方法，从而指导中毒模型的优化方向。为了更新多个模型的参数，我们采用了地址共享的后门模型训练，从而利用梯度信息进行多模型协同优化。在不同的模型、数据集和任务上进行了广泛的实验，证明了这种新型后门攻击的有效性及其潜在的应用威胁。



## **50. Exploring the Robustness of AI-Driven Tools in Digital Forensics: A Preliminary Study**

探索数字取证中人工智能驱动工具的稳健性：初步研究 cs.CV

**SubmitDate**: 2024-12-02    [abs](http://arxiv.org/abs/2412.01363v1) [paper-pdf](http://arxiv.org/pdf/2412.01363v1)

**Authors**: Silvia Lucia Sanna, Leonardo Regano, Davide Maiorca, Giorgio Giacinto

**Abstract**: Nowadays, many tools are used to facilitate forensic tasks about data extraction and data analysis. In particular, some tools leverage Artificial Intelligence (AI) to automatically label examined data into specific categories (\ie, drugs, weapons, nudity). However, this raises a serious concern about the robustness of the employed AI algorithms against adversarial attacks. Indeed, some people may need to hide specific data to AI-based digital forensics tools, thus manipulating the content so that the AI system does not recognize the offensive/prohibited content and marks it at as suspicious to the analyst. This could be seen as an anti-forensics attack scenario. For this reason, we analyzed two of the most important forensics tools employing AI for data classification: Magnet AI, used by Magnet Axiom, and Excire Photo AI, used by X-Ways Forensics. We made preliminary tests using about $200$ images, other $100$ sent in $3$ chats about pornography and teenage nudity, drugs and weapons to understand how the tools label them. Moreover, we loaded some deepfake images (images generated by AI forging real ones) of some actors to understand if they would be classified in the same category as the original images. From our preliminary study, we saw that the AI algorithm is not robust enough, as we expected since these topics are still open research problems. For example, some sexual images were not categorized as nudity, and some deepfakes were categorized as the same real person, while the human eye can see the clear nudity image or catch the difference between the deepfakes. Building on these results and other state-of-the-art works, we provide some suggestions for improving how digital forensics analysis tool leverage AI and their robustness against adversarial attacks or different scenarios than the trained one.

摘要: 如今，许多工具被用来促进关于数据提取和数据分析的取证任务。特别是，一些工具利用人工智能(AI)自动将检查的数据标记为特定类别(例如，毒品、武器、裸体)。然而，这引发了人们对所采用的人工智能算法对抗对手攻击的稳健性的严重担忧。事实上，有些人可能需要向基于人工智能的数字取证工具隐藏特定数据，从而操纵内容，以便人工智能系统无法识别攻击性/违禁内容，并将其标记为分析师怀疑的内容。这可以被视为反取证攻击场景。为此，我们分析了两个使用人工智能进行数据分类的最重要的取证工具：Magnet Axiom使用的Magnet AI和X-Ways Forensics使用的Excire Photo AI。我们使用了大约200美元的图片进行了初步测试，其他100美元图片是在3美元聊天中发送的，内容涉及色情、青少年裸体、毒品和武器，以了解工具是如何给它们贴上标签的。此外，我们加载了一些演员的一些深度假图像(人工智能伪造真实图像生成的图像)，以了解他们是否会被归类为与原始图像相同的类别。从我们的初步研究中，我们看到AI算法还不够健壮，正如我们预期的那样，因为这些主题仍然是开放的研究问题。例如，有些性爱图像不被归类为裸体，有些深伪被归类为同一个真人，而人眼可以看到清晰的裸体图像或辨别深伪之间的差异。在这些结果和其他最先进的工作的基础上，我们提供了一些建议，以改进数字取证分析工具如何利用人工智能及其对对手攻击或与训练有素的场景不同的场景的稳健性。



