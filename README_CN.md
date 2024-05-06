# Latest Adversarial Attack Papers
**update at 2024-05-06 11:09:44**

翻译来自 https://cloud.tencent.com/document/product/551/15619

## **1. Purify Unlearnable Examples via Rate-Constrained Variational Autoencoders**

通过速率约束变分自动编码器净化不可学习的示例 cs.CR

Accepted by ICML 2024

**SubmitDate**: 2024-05-02    [abs](http://arxiv.org/abs/2405.01460v1) [paper-pdf](http://arxiv.org/pdf/2405.01460v1)

**Authors**: Yi Yu, Yufei Wang, Song Xia, Wenhan Yang, Shijian Lu, Yap-Peng Tan, Alex C. Kot

**Abstract**: Unlearnable examples (UEs) seek to maximize testing error by making subtle modifications to training examples that are correctly labeled. Defenses against these poisoning attacks can be categorized based on whether specific interventions are adopted during training. The first approach is training-time defense, such as adversarial training, which can mitigate poisoning effects but is computationally intensive. The other approach is pre-training purification, e.g., image short squeezing, which consists of several simple compressions but often encounters challenges in dealing with various UEs. Our work provides a novel disentanglement mechanism to build an efficient pre-training purification method. Firstly, we uncover rate-constrained variational autoencoders (VAEs), demonstrating a clear tendency to suppress the perturbations in UEs. We subsequently conduct a theoretical analysis for this phenomenon. Building upon these insights, we introduce a disentangle variational autoencoder (D-VAE), capable of disentangling the perturbations with learnable class-wise embeddings. Based on this network, a two-stage purification approach is naturally developed. The first stage focuses on roughly eliminating perturbations, while the second stage produces refined, poison-free results, ensuring effectiveness and robustness across various scenarios. Extensive experiments demonstrate the remarkable performance of our method across CIFAR-10, CIFAR-100, and a 100-class ImageNet-subset. Code is available at https://github.com/yuyi-sd/D-VAE.

摘要: 不能学习的例子(UE)试图通过对正确标记的训练例子进行微妙的修改来最大化测试误差。针对这些中毒攻击的防御措施可以根据是否在训练期间采取特定干预措施进行分类。第一种方法是训练时间防御，例如对抗性训练，这种方法可以减轻中毒影响，但计算密集。另一种方法是训练前净化，例如图像短压缩，它由几个简单的压缩组成，但在处理各种UE时经常遇到挑战。我们的工作为构建高效的预训练净化方法提供了一种新的解缠机制。首先，我们发现了码率受限的变分自动编码器(VAE)，显示了抑制UE中微扰的明显趋势。我们随后对这一现象进行了理论分析。基于这些见解，我们引入了一种解缠变分自动编码器(D-VAE)，它能够通过可学习的类嵌入来解缠扰动。在这个网络的基础上，自然发展了一种两级提纯方法。第一阶段侧重于粗略地消除干扰，而第二阶段产生精炼的、无毒的结果，确保在各种情况下的有效性和健壮性。广泛的实验表明，我们的方法在CIFAR-10、CIFAR-100和100类ImageNet子集上具有显著的性能。代码可在https://github.com/yuyi-sd/D-VAE.上找到



## **2. Position Paper: Beyond Robustness Against Single Attack Types**

立场文件：超越针对单一攻击类型的稳健性 cs.LG

**SubmitDate**: 2024-05-02    [abs](http://arxiv.org/abs/2405.01349v1) [paper-pdf](http://arxiv.org/pdf/2405.01349v1)

**Authors**: Sihui Dai, Chong Xiang, Tong Wu, Prateek Mittal

**Abstract**: Current research on defending against adversarial examples focuses primarily on achieving robustness against a single attack type such as $\ell_2$ or $\ell_{\infty}$-bounded attacks. However, the space of possible perturbations is much larger and currently cannot be modeled by a single attack type. The discrepancy between the focus of current defenses and the space of attacks of interest calls to question the practicality of existing defenses and the reliability of their evaluation. In this position paper, we argue that the research community should look beyond single attack robustness, and we draw attention to three potential directions involving robustness against multiple attacks: simultaneous multiattack robustness, unforeseen attack robustness, and a newly defined problem setting which we call continual adaptive robustness. We provide a unified framework which rigorously defines these problem settings, synthesize existing research in these fields, and outline open directions. We hope that our position paper inspires more research in simultaneous multiattack, unforeseen attack, and continual adaptive robustness.

摘要: 目前关于防御恶意攻击的研究主要集中在对单一攻击类型的健壮性上，例如$\ell_2$或$\ell_{\infty}$-bound攻击。然而，可能的扰动空间要大得多，目前不能用单一的攻击类型来建模。当前防御的重点与感兴趣的攻击的空间之间的差异要求对现有防御的实用性及其评估的可靠性提出质疑。在这份立场文件中，我们认为研究界应该超越单一攻击的稳健性，并提请注意涉及对多个攻击的稳健性的三个潜在方向：同时多攻击稳健性、不可预见的攻击稳健性以及我们称为连续自适应稳健性的新定义的问题设置。我们提供了一个统一的框架，严格定义了这些问题设置，综合了这些领域的现有研究，并概述了开放的方向。我们希望我们的立场文件能启发更多关于同时多攻击、不可预见攻击和持续自适应稳健性的研究。



## **3. LLM Self Defense: By Self Examination, LLMs Know They Are Being Tricked**

法学硕士自卫：通过自我检查，法学硕士知道他们被欺骗了 cs.CL

**SubmitDate**: 2024-05-02    [abs](http://arxiv.org/abs/2308.07308v4) [paper-pdf](http://arxiv.org/pdf/2308.07308v4)

**Authors**: Mansi Phute, Alec Helbling, Matthew Hull, ShengYun Peng, Sebastian Szyller, Cory Cornelius, Duen Horng Chau

**Abstract**: Large language models (LLMs) are popular for high-quality text generation but can produce harmful content, even when aligned with human values through reinforcement learning. Adversarial prompts can bypass their safety measures. We propose LLM Self Defense, a simple approach to defend against these attacks by having an LLM screen the induced responses. Our method does not require any fine-tuning, input preprocessing, or iterative output generation. Instead, we incorporate the generated content into a pre-defined prompt and employ another instance of an LLM to analyze the text and predict whether it is harmful. We test LLM Self Defense on GPT 3.5 and Llama 2, two of the current most prominent LLMs against various types of attacks, such as forcefully inducing affirmative responses to prompts and prompt engineering attacks. Notably, LLM Self Defense succeeds in reducing the attack success rate to virtually 0 using both GPT 3.5 and Llama 2. The code is publicly available at https://github.com/poloclub/llm-self-defense

摘要: 大型语言模型(LLM)对于高质量的文本生成很受欢迎，但可能会产生有害的内容，即使通过强化学习与人类的价值观保持一致。对抗性提示可以绕过它们的安全措施。我们提出了LLM自卫，这是一种通过让LLM筛选诱导响应来防御这些攻击的简单方法。我们的方法不需要任何微调、输入预处理或迭代输出生成。相反，我们将生成的内容合并到预定义的提示中，并使用LLM的另一个实例来分析文本并预测它是否有害。我们在GPT 3.5和Llama 2上测试了LLM自卫，这两种当前最著名的LLM针对各种类型的攻击，例如对提示的强制诱导肯定反应和提示工程攻击。值得注意的是，LLm自卫成功地使用GPT3.5和Llama 2将攻击成功率降低到几乎为0。代码可在https://github.com/poloclub/llm-self-defense上公开获得



## **4. Causal Influence in Federated Edge Inference**

联邦边缘推理中的因果影响 cs.LG

**SubmitDate**: 2024-05-02    [abs](http://arxiv.org/abs/2405.01260v1) [paper-pdf](http://arxiv.org/pdf/2405.01260v1)

**Authors**: Mert Kayaalp, Yunus Inan, Visa Koivunen, Ali H. Sayed

**Abstract**: In this paper, we consider a setting where heterogeneous agents with connectivity are performing inference using unlabeled streaming data. Observed data are only partially informative about the target variable of interest. In order to overcome the uncertainty, agents cooperate with each other by exchanging their local inferences with and through a fusion center. To evaluate how each agent influences the overall decision, we adopt a causal framework in order to distinguish the actual influence of agents from mere correlations within the decision-making process. Various scenarios reflecting different agent participation patterns and fusion center policies are investigated. We derive expressions to quantify the causal impact of each agent on the joint decision, which could be beneficial for anticipating and addressing atypical scenarios, such as adversarial attacks or system malfunctions. We validate our theoretical results with numerical simulations and a real-world application of multi-camera crowd counting.

摘要: 在本文中，我们考虑了一种环境，其中具有连通性的异类代理使用未标记的流数据执行推理。观察到的数据只提供了感兴趣的目标变量的部分信息。为了克服不确定性，代理之间通过与融合中心交换本地推理来相互合作。为了评估每个代理人如何影响整体决策，我们采用了一个因果框架，以区分代理人的实际影响与决策过程中的单纯相关性。研究了反映不同主体参与模式和融合中心策略的各种场景。我们推导出表达式来量化每个智能体对联合决策的因果影响，这可能有助于预测和处理非典型场景，如对抗性攻击或系统故障。我们用数值模拟和多摄像机人群计数的实际应用验证了我们的理论结果。



## **5. Boosting Jailbreak Attack with Momentum**

以势头助推越狱攻击 cs.LG

ICLR 2024 Workshop on Reliable and Responsible Foundation Models

**SubmitDate**: 2024-05-02    [abs](http://arxiv.org/abs/2405.01229v1) [paper-pdf](http://arxiv.org/pdf/2405.01229v1)

**Authors**: Yihao Zhang, Zeming Wei

**Abstract**: Large Language Models (LLMs) have achieved remarkable success across diverse tasks, yet they remain vulnerable to adversarial attacks, notably the well-documented \textit{jailbreak} attack. Recently, the Greedy Coordinate Gradient (GCG) attack has demonstrated efficacy in exploiting this vulnerability by optimizing adversarial prompts through a combination of gradient heuristics and greedy search. However, the efficiency of this attack has become a bottleneck in the attacking process. To mitigate this limitation, in this paper we rethink the generation of adversarial prompts through an optimization lens, aiming to stabilize the optimization process and harness more heuristic insights from previous iterations. Specifically, we introduce the \textbf{M}omentum \textbf{A}ccelerated G\textbf{C}G (\textbf{MAC}) attack, which incorporates a momentum term into the gradient heuristic. Experimental results showcase the notable enhancement achieved by MAP in gradient-based attacks on aligned language models. Our code is available at https://github.com/weizeming/momentum-attack-llm.

摘要: 大型语言模型(LLM)已经在不同的任务中取得了显著的成功，但它们仍然容易受到对手的攻击，特别是有充分记录的\textit{jailBreak}攻击。最近，贪婪坐标梯度(GCG)攻击通过结合梯度启发式算法和贪婪搜索来优化敌意提示，从而有效地利用了这一漏洞。然而，这种攻击的效率已经成为攻击过程中的瓶颈。为了缓解这一局限性，在本文中，我们通过优化镜头重新考虑对抗性提示的生成，旨在稳定优化过程，并从以前的迭代中获得更多启发式的见解。具体地说，我们引入了将动量项结合到梯度启发式中的加速G(Textbf{C}G(Textbf{MAC}))攻击。实验结果表明，MAP在对对齐语言模型的基于梯度的攻击中取得了显著的改进。我们的代码可以在https://github.com/weizeming/momentum-attack-llm.上找到



## **6. Neural Exec: Learning (and Learning from) Execution Triggers for Prompt Injection Attacks**

Neural Exec：学习（并学习）执行触发器以进行即时注入攻击 cs.CR

v0.2

**SubmitDate**: 2024-05-02    [abs](http://arxiv.org/abs/2403.03792v2) [paper-pdf](http://arxiv.org/pdf/2403.03792v2)

**Authors**: Dario Pasquini, Martin Strohmeier, Carmela Troncoso

**Abstract**: We introduce a new family of prompt injection attacks, termed Neural Exec. Unlike known attacks that rely on handcrafted strings (e.g., "Ignore previous instructions and..."), we show that it is possible to conceptualize the creation of execution triggers as a differentiable search problem and use learning-based methods to autonomously generate them.   Our results demonstrate that a motivated adversary can forge triggers that are not only drastically more effective than current handcrafted ones but also exhibit inherent flexibility in shape, properties, and functionality. In this direction, we show that an attacker can design and generate Neural Execs capable of persisting through multi-stage preprocessing pipelines, such as in the case of Retrieval-Augmented Generation (RAG)-based applications. More critically, our findings show that attackers can produce triggers that deviate markedly in form and shape from any known attack, sidestepping existing blacklist-based detection and sanitation approaches.

摘要: 我们介绍了一类新的快速注入攻击，称为神经执行攻击。与依赖手工创建的字符串(例如，“忽略先前的指令和...”)的已知攻击不同，我们展示了将创建执行触发器概念化为可区分的搜索问题并使用基于学习的方法自主生成它们是可能的。我们的结果表明，有动机的对手可以伪造触发器，不仅比目前手工制作的触发器有效得多，而且在形状、属性和功能上表现出固有的灵活性。在这个方向上，我们展示了攻击者可以设计和生成能够在多阶段预处理管道中持久存在的神经Execs，例如在基于检索-增强生成(RAG)的应用程序的情况下。更关键的是，我们的发现表明，攻击者可以产生在形式和形状上与任何已知攻击显著偏离的触发器，绕过现有的基于黑名单的检测和卫生方法。



## **7. The Perception-Robustness Tradeoff in Deterministic Image Restoration**

确定性图像恢复中的感知与鲁棒性权衡 eess.IV

**SubmitDate**: 2024-05-02    [abs](http://arxiv.org/abs/2311.09253v2) [paper-pdf](http://arxiv.org/pdf/2311.09253v2)

**Authors**: Guy Ohayon, Tomer Michaeli, Michael Elad

**Abstract**: We study the behavior of deterministic methods for solving inverse problems in imaging. These methods are commonly designed to achieve two goals: (1) attaining high perceptual quality, and (2) generating reconstructions that are consistent with the measurements. We provide a rigorous proof that the better a predictor satisfies these two requirements, the larger its Lipschitz constant must be, regardless of the nature of the degradation involved. In particular, to approach perfect perceptual quality and perfect consistency, the Lipschitz constant of the model must grow to infinity. This implies that such methods are necessarily more susceptible to adversarial attacks. We demonstrate our theory on single image super-resolution algorithms, addressing both noisy and noiseless settings. We also show how this undesired behavior can be leveraged to explore the posterior distribution, thereby allowing the deterministic model to imitate stochastic methods.

摘要: 我们研究解决成像反问题的确定性方法的行为。这些方法通常旨在实现两个目标：（1）获得高感知质量，以及（2）生成与测量结果一致的重建。我们提供了一个严格的证据，证明预测器满足这两个要求越好，其利普希茨常数就必须越大，无论所涉及的退化的性质如何。特别是，为了达到完美的感知质量和完美的一致性，模型的利普希茨常数必须增长到无穷大。这意味着此类方法必然更容易受到对抗攻击。我们展示了我们关于单图像超分辨率算法的理论，解决有噪和无噪设置。我们还展示了如何利用这种不受欢迎的行为来探索后验分布，从而允许确定性模型模仿随机方法。



## **8. Beyond the Bridge: Contention-Based Covert and Side Channel Attacks on Multi-GPU Interconnect**

超越桥梁：基于竞争的隐蔽和侧通道攻击多图形处理器互连 cs.CR

Accepted to SEED 2024

**SubmitDate**: 2024-05-02    [abs](http://arxiv.org/abs/2404.03877v2) [paper-pdf](http://arxiv.org/pdf/2404.03877v2)

**Authors**: Yicheng Zhang, Ravan Nazaraliyev, Sankha Baran Dutta, Nael Abu-Ghazaleh, Andres Marquez, Kevin Barker

**Abstract**: High-speed interconnects, such as NVLink, are integral to modern multi-GPU systems, acting as a vital link between CPUs and GPUs. This study highlights the vulnerability of multi-GPU systems to covert and side channel attacks due to congestion on interconnects. An adversary can infer private information about a victim's activities by monitoring NVLink congestion without needing special permissions. Leveraging this insight, we develop a covert channel attack across two GPUs with a bandwidth of 45.5 kbps and a low error rate, and introduce a side channel attack enabling attackers to fingerprint applications through the shared NVLink interconnect.

摘要: NVLink等高速互连是现代多图形处理器系统不可或缺的组成部分，是中央处理器和图形处理器之间的重要联系。这项研究强调了多图形处理器系统由于互连拥堵而容易受到隐蔽和侧通道攻击。对手可以通过监视NVLink拥堵来推断有关受害者活动的私人信息，而无需特殊许可。利用这一见解，我们开发了跨两个带宽为45.5 kMbps、低错误率的隐通道攻击，并引入侧通道攻击，使攻击者能够通过共享的NVLink互连对应用程序进行指纹识别。



## **9. MISLEAD: Manipulating Importance of Selected features for Learning Epsilon in Evasion Attack Deception**

MISLEAD：操纵选择功能的重要性以学习躲避攻击欺骗中的情节 cs.LG

**SubmitDate**: 2024-05-02    [abs](http://arxiv.org/abs/2404.15656v2) [paper-pdf](http://arxiv.org/pdf/2404.15656v2)

**Authors**: Vidit Khazanchi, Pavan Kulkarni, Yuvaraj Govindarajulu, Manojkumar Parmar

**Abstract**: Emerging vulnerabilities in machine learning (ML) models due to adversarial attacks raise concerns about their reliability. Specifically, evasion attacks manipulate models by introducing precise perturbations to input data, causing erroneous predictions. To address this, we propose a methodology combining SHapley Additive exPlanations (SHAP) for feature importance analysis with an innovative Optimal Epsilon technique for conducting evasion attacks. Our approach begins with SHAP-based analysis to understand model vulnerabilities, crucial for devising targeted evasion strategies. The Optimal Epsilon technique, employing a Binary Search algorithm, efficiently determines the minimum epsilon needed for successful evasion. Evaluation across diverse machine learning architectures demonstrates the technique's precision in generating adversarial samples, underscoring its efficacy in manipulating model outcomes. This study emphasizes the critical importance of continuous assessment and monitoring to identify and mitigate potential security risks in machine learning systems.

摘要: 由于对抗性攻击，机器学习(ML)模型中新出现的漏洞引发了人们对其可靠性的担忧。具体地说，规避攻击通过向输入数据引入精确的扰动来操纵模型，从而导致错误的预测。为了解决这一问题，我们提出了一种方法，结合Shapley Additive In释义(Shap)用于特征重要性分析和创新的最优Epsilon技术来进行规避攻击。我们的方法从基于Shap的分析开始，以了解模型漏洞，这对于设计有针对性的规避策略至关重要。采用二进制搜索算法的最优Epsilon技术有效地确定了成功躲避所需的最小epsilon。对不同机器学习体系结构的评估表明，该技术在生成对抗性样本方面具有准确性，强调了其在操纵模型结果方面的有效性。这项研究强调了持续评估和监测的重要性，以识别和缓解机器学习系统中的潜在安全风险。



## **10. Adversarial Attacks and Defense for Conversation Entailment Task**

对抗性攻击和对话需求任务的防御 cs.CL

**SubmitDate**: 2024-05-02    [abs](http://arxiv.org/abs/2405.00289v2) [paper-pdf](http://arxiv.org/pdf/2405.00289v2)

**Authors**: Zhenning Yang, Ryan Krawec, Liang-Yuan Wu

**Abstract**: As the deployment of NLP systems in critical applications grows, ensuring the robustness of large language models (LLMs) against adversarial attacks becomes increasingly important. Large language models excel in various NLP tasks but remain vulnerable to low-cost adversarial attacks. Focusing on the domain of conversation entailment, where multi-turn dialogues serve as premises to verify hypotheses, we fine-tune a transformer model to accurately discern the truthfulness of these hypotheses. Adversaries manipulate hypotheses through synonym swapping, aiming to deceive the model into making incorrect predictions. To counteract these attacks, we implemented innovative fine-tuning techniques and introduced an embedding perturbation loss method to significantly bolster the model's robustness. Our findings not only emphasize the importance of defending against adversarial attacks in NLP but also highlight the real-world implications, suggesting that enhancing model robustness is critical for reliable NLP applications.

摘要: 随着NLP系统在关键应用中的部署越来越多，确保大型语言模型(LLM)对对手攻击的健壮性变得越来越重要。大型语言模型在各种NLP任务中表现出色，但仍然容易受到低成本的对抗性攻击。聚焦于会话蕴涵领域，多轮对话是验证假设的前提，我们微调了一个转换器模型，以准确识别这些假设的真实性。对手通过同义词互换来操纵假设，目的是欺骗模型做出错误的预测。为了对抗这些攻击，我们实施了创新的微调技术，并引入了嵌入扰动损失方法来显著增强模型的稳健性。我们的发现不仅强调了在自然语言处理中防御对手攻击的重要性，而且也强调了现实世界的影响，表明增强模型的健壮性对于可靠的自然语言处理应用是至关重要的。



## **11. Pixel is a Barrier: Diffusion Models Are More Adversarially Robust Than We Think**

像素是一个障碍：扩散模型比我们想象的更具对抗性 cs.CV

**SubmitDate**: 2024-05-02    [abs](http://arxiv.org/abs/2404.13320v2) [paper-pdf](http://arxiv.org/pdf/2404.13320v2)

**Authors**: Haotian Xue, Yongxin Chen

**Abstract**: Adversarial examples for diffusion models are widely used as solutions for safety concerns. By adding adversarial perturbations to personal images, attackers can not edit or imitate them easily. However, it is essential to note that all these protections target the latent diffusion model (LDMs), the adversarial examples for diffusion models in the pixel space (PDMs) are largely overlooked. This may mislead us to think that the diffusion models are vulnerable to adversarial attacks like most deep models. In this paper, we show novel findings that: even though gradient-based white-box attacks can be used to attack the LDMs, they fail to attack PDMs. This finding is supported by extensive experiments of almost a wide range of attacking methods on various PDMs and LDMs with different model structures, which means diffusion models are indeed much more robust against adversarial attacks. We also find that PDMs can be used as an off-the-shelf purifier to effectively remove the adversarial patterns that were generated on LDMs to protect the images, which means that most protection methods nowadays, to some extent, cannot protect our images from malicious attacks. We hope that our insights will inspire the community to rethink the adversarial samples for diffusion models as protection methods and move forward to more effective protection. Codes are available in https://github.com/xavihart/PDM-Pure.

摘要: 扩散模型的对抗性例子被广泛用作安全问题的解决方案。通过向个人图像添加敌意干扰，攻击者无法轻松编辑或模仿它们。然而，值得注意的是，所有这些保护都是针对潜在扩散模型(LDMS)的，而像素空间扩散模型(PDMS)的对抗性例子在很大程度上被忽视了。这可能会误导我们认为扩散模型像大多数深度模型一样容易受到对手攻击。在本文中，我们发现了新的发现：即使基于梯度的白盒攻击可以用于攻击LDMS，它们也不能攻击PDMS。这一发现得到了对不同模型结构的PDMS和LDM上几乎各种攻击方法的广泛实验的支持，这意味着扩散模型确实对对手攻击具有更强的鲁棒性。我们还发现，PDMS可以作为现成的净化器来有效地去除LDM上产生的恶意模式来保护图像，这意味着目前的大多数保护方法在某种程度上不能保护我们的图像免受恶意攻击。我们希望我们的见解将激励社会重新考虑将扩散模型的对抗性样本作为保护方法，并朝着更有效的保护迈进。代码在https://github.com/xavihart/PDM-Pure.中可用



## **12. Intriguing Properties of Diffusion Models: An Empirical Study of the Natural Attack Capability in Text-to-Image Generative Models**

扩散模型的有趣特性：文本到图像生成模型中自然攻击能力的实证研究 cs.CV

**SubmitDate**: 2024-05-02    [abs](http://arxiv.org/abs/2308.15692v2) [paper-pdf](http://arxiv.org/pdf/2308.15692v2)

**Authors**: Takami Sato, Justin Yue, Nanze Chen, Ningfei Wang, Qi Alfred Chen

**Abstract**: Denoising probabilistic diffusion models have shown breakthrough performance to generate more photo-realistic images or human-level illustrations than the prior models such as GANs. This high image-generation capability has stimulated the creation of many downstream applications in various areas. However, we find that this technology is actually a double-edged sword: We identify a new type of attack, called the Natural Denoising Diffusion (NDD) attack based on the finding that state-of-the-art deep neural network (DNN) models still hold their prediction even if we intentionally remove their robust features, which are essential to the human visual system (HVS), through text prompts. The NDD attack shows a significantly high capability to generate low-cost, model-agnostic, and transferable adversarial attacks by exploiting the natural attack capability in diffusion models. To systematically evaluate the risk of the NDD attack, we perform a large-scale empirical study with our newly created dataset, the Natural Denoising Diffusion Attack (NDDA) dataset. We evaluate the natural attack capability by answering 6 research questions. Through a user study, we find that it can achieve an 88% detection rate while being stealthy to 93% of human subjects; we also find that the non-robust features embedded by diffusion models contribute to the natural attack capability. To confirm the model-agnostic and transferable attack capability, we perform the NDD attack against the Tesla Model 3 and find that 73% of the physically printed attacks can be detected as stop signs. Our hope is that the study and dataset can help our community be aware of the risks in diffusion models and facilitate further research toward robust DNN models.

摘要: 去噪概率扩散模型表现出了突破性的性能，与Gans等先前的模型相比，它可以生成更多照片级图像或真人级别的插图。这种高图像生成能力刺激了许多不同领域的下游应用的创建。然而，我们发现这项技术实际上是一把双刃剑：我们识别出一种新的攻击类型，称为自然去噪扩散(NDD)攻击，基于这一发现，即使我们故意通过文本提示删除对人类视觉系统(HVS)至关重要的健壮特征，最先进的深度神经网络(DNN)模型仍然保持其预测。NDD攻击通过利用扩散模型中的自然攻击能力来产生低成本、模型无关和可转移的敌意攻击的能力显著提高。为了系统地评估NDD攻击的风险，我们使用我们新创建的数据集-自然去噪扩散攻击(NDDA)数据集进行了大规模的实证研究。我们通过回答6个研究问题来评估自然攻击能力。通过用户研究，我们发现它可以在对93%的人类主体隐身的情况下达到88%的检测率；我们还发现扩散模型嵌入的非稳健特征对自然攻击能力有贡献。为了确认模型无关和可转移的攻击能力，我们对特斯拉Model 3进行了NDD攻击，发现73%的物理打印攻击可以被检测为停止标志。我们希望这项研究和数据集可以帮助我们的社区意识到扩散模型中的风险，并促进对稳健DNN模型的进一步研究。



## **13. AmpleGCG: Learning a Universal and Transferable Generative Model of Adversarial Suffixes for Jailbreaking Both Open and Closed LLMs**

AmpleGCG：学习通用且可转移的对抗性后缀生成模型，用于越狱开放和封闭LLM cs.CL

**SubmitDate**: 2024-05-02    [abs](http://arxiv.org/abs/2404.07921v2) [paper-pdf](http://arxiv.org/pdf/2404.07921v2)

**Authors**: Zeyi Liao, Huan Sun

**Abstract**: As large language models (LLMs) become increasingly prevalent and integrated into autonomous systems, ensuring their safety is imperative. Despite significant strides toward safety alignment, recent work GCG~\citep{zou2023universal} proposes a discrete token optimization algorithm and selects the single suffix with the lowest loss to successfully jailbreak aligned LLMs. In this work, we first discuss the drawbacks of solely picking the suffix with the lowest loss during GCG optimization for jailbreaking and uncover the missed successful suffixes during the intermediate steps. Moreover, we utilize those successful suffixes as training data to learn a generative model, named AmpleGCG, which captures the distribution of adversarial suffixes given a harmful query and enables the rapid generation of hundreds of suffixes for any harmful queries in seconds. AmpleGCG achieves near 100\% attack success rate (ASR) on two aligned LLMs (Llama-2-7B-chat and Vicuna-7B), surpassing two strongest attack baselines. More interestingly, AmpleGCG also transfers seamlessly to attack different models, including closed-source LLMs, achieving a 99\% ASR on the latest GPT-3.5. To summarize, our work amplifies the impact of GCG by training a generative model of adversarial suffixes that is universal to any harmful queries and transferable from attacking open-source LLMs to closed-source LLMs. In addition, it can generate 200 adversarial suffixes for one harmful query in only 4 seconds, rendering it more challenging to defend.

摘要: 随着大型语言模型(LLM)变得越来越普遍并集成到自治系统中，确保它们的安全性是当务之急。尽管在安全对齐方面取得了长足的进步，但最近的工作GCG~\Citep{zou2023Universal}提出了一种离散令牌优化算法，并选择损失最小的单个后缀来成功越狱对齐LLM。在这项工作中，我们首先讨论了在GCG优化越狱过程中只选择损失最小的后缀的缺点，并在中间步骤中发现了遗漏的成功后缀。此外，我们利用这些成功的后缀作为训练数据来学习一种名为AmpleGCG的生成模型，该模型捕获给定有害查询的对抗性后缀的分布，并在几秒钟内为任何有害查询快速生成数百个后缀。AmpleGCG在两个对齐的LLM(Llama-2-7B-Chat和Vicuna-7B)上达到了近100%的攻击成功率(ASR)，超过了两个最强的攻击基线。更有趣的是，AmpleGCG还无缝传输以攻击不同的型号，包括闭源LLMS，在最新的GPT-3.5上实现了99\%的ASR。总之，我们的工作通过训练对抗性后缀的生成模型来放大GCG的影响，该模型对任何有害的查询都是通用的，并且可以从攻击开源LLM转移到闭源LLM。此外，它可以在短短4秒内为一个有害的查询生成200个敌意后缀，使其更具挑战性。



## **14. A Survey on Transferability of Adversarial Examples across Deep Neural Networks**

深度神经网络对抗性示例可移植性调查 cs.CV

Accepted to Transactions on Machine Learning Research (TMLR)

**SubmitDate**: 2024-05-02    [abs](http://arxiv.org/abs/2310.17626v2) [paper-pdf](http://arxiv.org/pdf/2310.17626v2)

**Authors**: Jindong Gu, Xiaojun Jia, Pau de Jorge, Wenqain Yu, Xinwei Liu, Avery Ma, Yuan Xun, Anjun Hu, Ashkan Khakzar, Zhijiang Li, Xiaochun Cao, Philip Torr

**Abstract**: The emergence of Deep Neural Networks (DNNs) has revolutionized various domains by enabling the resolution of complex tasks spanning image recognition, natural language processing, and scientific problem-solving. However, this progress has also brought to light a concerning vulnerability: adversarial examples. These crafted inputs, imperceptible to humans, can manipulate machine learning models into making erroneous predictions, raising concerns for safety-critical applications. An intriguing property of this phenomenon is the transferability of adversarial examples, where perturbations crafted for one model can deceive another, often with a different architecture. This intriguing property enables black-box attacks which circumvents the need for detailed knowledge of the target model. This survey explores the landscape of the adversarial transferability of adversarial examples. We categorize existing methodologies to enhance adversarial transferability and discuss the fundamental principles guiding each approach. While the predominant body of research primarily concentrates on image classification, we also extend our discussion to encompass other vision tasks and beyond. Challenges and opportunities are discussed, highlighting the importance of fortifying DNNs against adversarial vulnerabilities in an evolving landscape.

摘要: 深度神经网络(DNN)的出现使图像识别、自然语言处理和科学问题解决等复杂任务的解决成为可能，从而使各个领域发生了革命性的变化。然而，这一进展也暴露了一个令人担忧的脆弱性：对抗性的例子。这些精心制作的、人类无法察觉的输入可能会操纵机器学习模型做出错误的预测，从而引发对安全关键应用的担忧。这种现象的一个耐人寻味的属性是对抗性例子的可转移性，在这种情况下，为一个模型精心设计的扰动可以欺骗另一个模型，通常是使用不同的架构。这一耐人寻味的特性使黑盒攻击成为可能，从而绕过了对目标模型详细知识的需要。这项调查探讨了对抗性例子的对抗性转移的情况。我们对增强对抗性可转移性的现有方法进行了分类，并讨论了指导每种方法的基本原则。虽然主要的研究主体主要集中在图像分类上，但我们也扩展了我们的讨论，以涵盖其他视觉任务和其他任务。讨论了挑战和机遇，强调了在不断发展的环境中加强DNN对抗对手脆弱性的重要性。



## **15. Why You Should Not Trust Interpretations in Machine Learning: Adversarial Attacks on Partial Dependence Plots**

为什么不应该相信机器学习中的解释：对部分依赖图的对抗性攻击 cs.LG

**SubmitDate**: 2024-05-01    [abs](http://arxiv.org/abs/2404.18702v2) [paper-pdf](http://arxiv.org/pdf/2404.18702v2)

**Authors**: Xi Xin, Giles Hooker, Fei Huang

**Abstract**: The adoption of artificial intelligence (AI) across industries has led to the widespread use of complex black-box models and interpretation tools for decision making. This paper proposes an adversarial framework to uncover the vulnerability of permutation-based interpretation methods for machine learning tasks, with a particular focus on partial dependence (PD) plots. This adversarial framework modifies the original black box model to manipulate its predictions for instances in the extrapolation domain. As a result, it produces deceptive PD plots that can conceal discriminatory behaviors while preserving most of the original model's predictions. This framework can produce multiple fooled PD plots via a single model. By using real-world datasets including an auto insurance claims dataset and COMPAS (Correctional Offender Management Profiling for Alternative Sanctions) dataset, our results show that it is possible to intentionally hide the discriminatory behavior of a predictor and make the black-box model appear neutral through interpretation tools like PD plots while retaining almost all the predictions of the original black-box model. Managerial insights for regulators and practitioners are provided based on the findings.

摘要: 人工智能(AI)在各个行业的采用导致了复杂的黑箱模型和解释工具的广泛使用，用于决策。本文提出了一个对抗性框架，以揭示机器学习任务中基于排列的解释方法的脆弱性，并特别关注部分依赖(PD)图。该对抗性框架修改了原始的黑盒模型，以操纵其对外推域中的实例的预测。结果，它产生了欺骗性的PD图，可以隐藏歧视行为，同时保留了原始模型的大部分预测。该框架可以通过一个模型产生多个受骗的PD图。通过使用包括汽车保险索赔数据集和COMPAS(惩教罪犯管理配置文件用于替代制裁)数据集的真实世界数据集，我们的结果表明，有可能故意隐藏预测者的歧视行为，并通过PD图等解释工具使黑盒模型看起来是中立的，同时保留原始黑盒模型的几乎所有预测。根据调查结果，为监管者和从业者提供了管理见解。



## **16. Certified Adversarial Robustness of Machine Learning-based Malware Detectors via (De)Randomized Smoothing**

通过（去）随机平滑验证基于机器学习的恶意软件检测器的对抗鲁棒性 cs.CR

**SubmitDate**: 2024-05-01    [abs](http://arxiv.org/abs/2405.00392v1) [paper-pdf](http://arxiv.org/pdf/2405.00392v1)

**Authors**: Daniel Gibert, Luca Demetrio, Giulio Zizzo, Quan Le, Jordi Planes, Battista Biggio

**Abstract**: Deep learning-based malware detection systems are vulnerable to adversarial EXEmples - carefully-crafted malicious programs that evade detection with minimal perturbation. As such, the community is dedicating effort to develop mechanisms to defend against adversarial EXEmples. However, current randomized smoothing-based defenses are still vulnerable to attacks that inject blocks of adversarial content. In this paper, we introduce a certifiable defense against patch attacks that guarantees, for a given executable and an adversarial patch size, no adversarial EXEmple exist. Our method is inspired by (de)randomized smoothing which provides deterministic robustness certificates. During training, a base classifier is trained using subsets of continguous bytes. At inference time, our defense splits the executable into non-overlapping chunks, classifies each chunk independently, and computes the final prediction through majority voting to minimize the influence of injected content. Furthermore, we introduce a preprocessing step that fixes the size of the sections and headers to a multiple of the chunk size. As a consequence, the injected content is confined to an integer number of chunks without tampering the other chunks containing the real bytes of the input examples, allowing us to extend our certified robustness guarantees to content insertion attacks. We perform an extensive ablation study, by comparing our defense with randomized smoothing-based defenses against a plethora of content manipulation attacks and neural network architectures. Results show that our method exhibits unmatched robustness against strong content-insertion attacks, outperforming randomized smoothing-based defenses in the literature.

摘要: 基于深度学习的恶意软件检测系统容易受到敌意示例的攻击--精心设计的恶意程序以最小的扰动逃避检测。因此，社区正在致力于发展机制，以防御敌对的例子。然而，当前基于随机平滑的防御仍然容易受到注入对抗性内容块的攻击。本文介绍了一种针对补丁攻击的可证明防御，它保证对于给定的可执行文件和对抗性补丁大小，不存在对抗性实例。我们的方法的灵感来自于提供确定性稳健性证书的(去)随机化平滑。在训练期间，使用连续字节的子集来训练基本分类器。在推理时，我们的辩护将可执行文件拆分成不重叠的块，独立地对每个块进行分类，并通过多数投票计算最终预测，以最大限度地减少注入内容的影响。此外，我们引入了一个预处理步骤，该步骤将部分和标头的大小固定为块大小的倍数。因此，注入的内容被限制为整数个块，而不会篡改包含输入示例的真实字节的其他块，从而允许我们将经过认证的健壮性保证扩展到内容插入攻击。我们进行了一项广泛的消融研究，将我们的防御与基于随机平滑的防御进行比较，以抵御过多的内容操纵攻击和神经网络架构。结果表明，该方法对强内容插入攻击表现出无与伦比的稳健性，优于文献中基于随机平滑的防御方法。



## **17. Graphene: Infrastructure Security Posture Analysis with AI-generated Attack Graphs**

石墨烯：利用人工智能生成的攻击图进行基础设施安全态势分析 cs.CR

**SubmitDate**: 2024-05-01    [abs](http://arxiv.org/abs/2312.13119v2) [paper-pdf](http://arxiv.org/pdf/2312.13119v2)

**Authors**: Xin Jin, Charalampos Katsis, Fan Sang, Jiahao Sun, Elisa Bertino, Ramana Rao Kompella, Ashish Kundu

**Abstract**: The rampant occurrence of cybersecurity breaches imposes substantial limitations on the progress of network infrastructures, leading to compromised data, financial losses, potential harm to individuals, and disruptions in essential services. The current security landscape demands the urgent development of a holistic security assessment solution that encompasses vulnerability analysis and investigates the potential exploitation of these vulnerabilities as attack paths. In this paper, we propose Graphene, an advanced system designed to provide a detailed analysis of the security posture of computing infrastructures. Using user-provided information, such as device details and software versions, Graphene performs a comprehensive security assessment. This assessment includes identifying associated vulnerabilities and constructing potential attack graphs that adversaries can exploit. Furthermore, Graphene evaluates the exploitability of these attack paths and quantifies the overall security posture through a scoring mechanism. The system takes a holistic approach by analyzing security layers encompassing hardware, system, network, and cryptography. Furthermore, Graphene delves into the interconnections between these layers, exploring how vulnerabilities in one layer can be leveraged to exploit vulnerabilities in others. In this paper, we present the end-to-end pipeline implemented in Graphene, showcasing the systematic approach adopted for conducting this thorough security analysis.

摘要: 网络安全漏洞的猖獗发生对网络基础设施的进展施加了很大限制，导致数据泄露、经济损失、对个人的潜在伤害以及基本服务中断。当前的安全形势要求迫切开发一种全面的安全评估解决方案，其中包括漏洞分析，并调查利用这些漏洞作为攻击途径的可能性。在本文中，我们提出了Graphene，这是一个高级系统，旨在提供对计算基础设施的安全态势的详细分析。使用用户提供的信息，如设备详细信息和软件版本，Graphene执行全面的安全评估。该评估包括识别相关漏洞和构建潜在的攻击图，以供攻击者利用。此外，Graphene还评估了这些攻击路径的可利用性，并通过评分机制量化了总体安全态势。该系统采取整体方法，分析包括硬件、系统、网络和加密在内的安全层。此外，Graphene深入研究了这些层之间的互连，探索如何利用一个层中的漏洞来利用其他层中的漏洞。在这篇文章中，我们介绍了在Graphene中实现的端到端管道，展示了为进行这种彻底的安全分析而采用的系统方法。



## **18. SimAC: A Simple Anti-Customization Method for Protecting Face Privacy against Text-to-Image Synthesis of Diffusion Models**

SimAC：一种简单的反定制方法，用于保护面部隐私，防止扩散模型的文本到图像合成 cs.CV

**SubmitDate**: 2024-04-30    [abs](http://arxiv.org/abs/2312.07865v2) [paper-pdf](http://arxiv.org/pdf/2312.07865v2)

**Authors**: Feifei Wang, Zhentao Tan, Tianyi Wei, Yue Wu, Qidong Huang

**Abstract**: Despite the success of diffusion-based customization methods on visual content creation, increasing concerns have been raised about such techniques from both privacy and political perspectives. To tackle this issue, several anti-customization methods have been proposed in very recent months, predominantly grounded in adversarial attacks. Unfortunately, most of these methods adopt straightforward designs, such as end-to-end optimization with a focus on adversarially maximizing the original training loss, thereby neglecting nuanced internal properties intrinsic to the diffusion model, and even leading to ineffective optimization in some diffusion time steps.In this paper, we strive to bridge this gap by undertaking a comprehensive exploration of these inherent properties, to boost the performance of current anti-customization approaches. Two aspects of properties are investigated: 1) We examine the relationship between time step selection and the model's perception in the frequency domain of images and find that lower time steps can give much more contributions to adversarial noises. This inspires us to propose an adaptive greedy search for optimal time steps that seamlessly integrates with existing anti-customization methods. 2) We scrutinize the roles of features at different layers during denoising and devise a sophisticated feature-based optimization framework for anti-customization.Experiments on facial benchmarks demonstrate that our approach significantly increases identity disruption, thereby protecting user privacy and copyright. Our code is available at: https://github.com/somuchtome/SimAC.

摘要: 尽管基于扩散的定制方法在视觉内容创作上取得了成功，但从隐私和政治的角度来看，人们对这种技术的关注越来越多。为了解决这个问题，最近几个月提出了几种反定制方法，主要基于对抗性攻击。遗憾的是，这些方法大多采用简单的设计，如端到端优化，侧重于反向最大化原始训练损失，从而忽略了扩散模型固有的细微内部属性，甚至在某些扩散时间步长导致无效优化，本文试图通过全面探索这些内在属性来弥合这一差距，以提高现有反定制方法的性能。研究了两个方面的性质：1)在图像的频域中，我们考察了时间步长选择与模型感知之间的关系，发现时间步长越低，对对抗性噪声的贡献越大。这启发了我们提出了一种自适应贪婪搜索来寻找最优时间步长，并与现有的反定制方法无缝集成。2)我们仔细研究了不同层次的特征在去噪过程中的作用，并设计了一个复杂的基于特征的反定制优化框架。在人脸基准上的实验表明，我们的方法显著增加了身份破坏，从而保护了用户隐私和版权。我们的代码请访问：https://github.com/somuchtome/SimAC.



## **19. Adversarial Example Soups: Improving Transferability and Stealthiness for Free**

对抗性示例汤：免费提高可转移性和隐蔽性 cs.CV

Under review

**SubmitDate**: 2024-04-30    [abs](http://arxiv.org/abs/2402.18370v2) [paper-pdf](http://arxiv.org/pdf/2402.18370v2)

**Authors**: Bo Yang, Hengwei Zhang, Jindong Wang, Yulong Yang, Chenhao Lin, Chao Shen, Zhengyu Zhao

**Abstract**: Transferable adversarial examples cause practical security risks since they can mislead a target model without knowing its internal knowledge. A conventional recipe for maximizing transferability is to keep only the optimal adversarial example from all those obtained in the optimization pipeline. In this paper, for the first time, we question this convention and demonstrate that those discarded, sub-optimal adversarial examples can be reused to boost transferability. Specifically, we propose ``Adversarial Example Soups'' (AES), with AES-tune for averaging discarded adversarial examples in hyperparameter tuning and AES-rand for stability testing. In addition, our AES is inspired by ``model soups'', which averages weights of multiple fine-tuned models for improved accuracy without increasing inference time. Extensive experiments validate the global effectiveness of our AES, boosting 10 state-of-the-art transfer attacks and their combinations by up to 13% against 10 diverse (defensive) target models. We also show the possibility of generalizing AES to other types, e.g., directly averaging multiple in-the-wild adversarial examples that yield comparable success. A promising byproduct of AES is the improved stealthiness of adversarial examples since the perturbation variances are naturally reduced.

摘要: 可转移的对抗性例子造成实际的安全风险，因为它们可能在不知道目标模型的内部知识的情况下误导目标模型。最大化可转移性的传统方法是从优化管道中获得的所有例子中只保留最优的对抗性例子。在本文中，我们第一次对这一惯例提出质疑，并证明了那些被丢弃的、次优的对抗性例子可以被重用来提高可转移性。具体地说，我们提出了“广告示例汤”(AES)，其中AES-Tune用于在超参数调整中平均丢弃的对抗性示例，而AES-Rand用于稳定性测试。此外，我们的高级加密标准的灵感来自“模型汤”，它平均多个微调模型的权重，以在不增加推断时间的情况下提高精度。广泛的实验验证了我们的AES的全球有效性，提高了10种最先进的转移攻击及其组合对10种不同(防御)目标模型的高达13%。我们还展示了将AES推广到其他类型的可能性，例如，直接平均产生类似成功的多个野生对抗性例子。AES的一个很有希望的副产品是改进了对抗性例子的隐蔽性，因为扰动方差自然地减少了。



## **20. Causal Perception Inspired Representation Learning for Trustworthy Image Quality Assessment**

因果感知启发了可信赖图像质量评估的表示学习 cs.CV

**SubmitDate**: 2024-04-30    [abs](http://arxiv.org/abs/2404.19567v1) [paper-pdf](http://arxiv.org/pdf/2404.19567v1)

**Authors**: Lei Wang, Desen Yuan

**Abstract**: Despite great success in modeling visual perception, deep neural network based image quality assessment (IQA) still remains unreliable in real-world applications due to its vulnerability to adversarial perturbations and the inexplicit black-box structure. In this paper, we propose to build a trustworthy IQA model via Causal Perception inspired Representation Learning (CPRL), and a score reflection attack method for IQA model. More specifically, we assume that each image is composed of Causal Perception Representation (CPR) and non-causal perception representation (N-CPR). CPR serves as the causation of the subjective quality label, which is invariant to the imperceptible adversarial perturbations. Inversely, N-CPR presents spurious associations with the subjective quality label, which may significantly change with the adversarial perturbations. To extract the CPR from each input image, we develop a soft ranking based channel-wise activation function to mediate the causally sufficient (beneficial for high prediction accuracy) and necessary (beneficial for high robustness) deep features, and based on intervention employ minimax game to optimize. Experiments on four benchmark databases show that the proposed CPRL method outperforms many state-of-the-art adversarial defense methods and provides explicit model interpretation.

摘要: 尽管基于深度神经网络的图像质量评估方法在视觉感知建模方面取得了很大的成功，但由于其易受对抗性扰动和不明确的黑盒结构的影响，在实际应用中仍然不可靠。本文提出了一种基于因果感知启发表征学习(CPRL)的可信IQA模型，并针对IQA模型提出了一种分数反射攻击方法。更具体地说，我们假设每个图像由因果知觉表征(CPR)和非因果知觉表征(N-CPR)组成。CPR作为主观质量标签的因果关系，对于潜伏的对抗性扰动是不变的。相反，N-CPR呈现出与主观质量标签的虚假关联，这可能会随着对抗性扰动而显著改变。为了从每幅输入图像中提取CPR，我们开发了一种基于软排名的通道激活函数来协调因果性充分(有利于高预测精度)和必要(有利于高鲁棒性)的深层特征，并基于干预使用极小极大博弈进行优化。在四个基准数据库上的实验表明，CPRL方法的性能优于许多最新的对抗性防御方法，并提供了明确的模型解释。



## **21. AttackBench: Evaluating Gradient-based Attacks for Adversarial Examples**

AttackBench：评估基于攻击的对抗性示例 cs.LG

https://attackbench.github.io

**SubmitDate**: 2024-04-30    [abs](http://arxiv.org/abs/2404.19460v1) [paper-pdf](http://arxiv.org/pdf/2404.19460v1)

**Authors**: Antonio Emanuele Cinà, Jérôme Rony, Maura Pintor, Luca Demetrio, Ambra Demontis, Battista Biggio, Ismail Ben Ayed, Fabio Roli

**Abstract**: Adversarial examples are typically optimized with gradient-based attacks. While novel attacks are continuously proposed, each is shown to outperform its predecessors using different experimental setups, hyperparameter settings, and number of forward and backward calls to the target models. This provides overly-optimistic and even biased evaluations that may unfairly favor one particular attack over the others. In this work, we aim to overcome these limitations by proposing AttackBench, i.e., the first evaluation framework that enables a fair comparison among different attacks. To this end, we first propose a categorization of gradient-based attacks, identifying their main components and differences. We then introduce our framework, which evaluates their effectiveness and efficiency. We measure these characteristics by (i) defining an optimality metric that quantifies how close an attack is to the optimal solution, and (ii) limiting the number of forward and backward queries to the model, such that all attacks are compared within a given maximum query budget. Our extensive experimental analysis compares more than 100 attack implementations with a total of over 800 different configurations against CIFAR-10 and ImageNet models, highlighting that only very few attacks outperform all the competing approaches. Within this analysis, we shed light on several implementation issues that prevent many attacks from finding better solutions or running at all. We release AttackBench as a publicly available benchmark, aiming to continuously update it to include and evaluate novel gradient-based attacks for optimizing adversarial examples.

摘要: 对抗性示例通常使用基于梯度的攻击进行优化。虽然不断有人提出新的攻击，但通过使用不同的实验设置、超参数设置以及对目标模型的前向和后向调用次数，每个攻击都显示出优于其前辈的性能。这提供了过于乐观甚至有偏见的评估，可能会不公平地偏袒某个特定攻击。在这项工作中，我们的目标是通过提出AttackBtch来克服这些限制，即第一个能够在不同攻击之间进行公平比较的评估框架。为此，我们首先对基于梯度的攻击进行了分类，找出了它们的主要组成部分和区别。然后，我们介绍了我们的框架，它评估了它们的有效性和效率。我们通过(I)定义最优度度量来量化攻击与最优解的距离，以及(Ii)限制对模型的向前和向后查询的数量，以便在给定的最大查询预算内比较所有攻击，来衡量这些特征。我们广泛的实验分析将100多个攻击实施与CIFAR-10和ImageNet型号的800多个不同配置进行了比较，强调只有极少数攻击的性能优于所有竞争方法。在这一分析中，我们阐明了几个实现问题，这些问题阻止了许多攻击找到更好的解决方案或根本无法运行。我们发布了一个公开可用的基准，旨在不断更新它，以包括和评估新的基于梯度的攻击，以优化对手的例子。



## **22. Probing Unlearned Diffusion Models: A Transferable Adversarial Attack Perspective**

探索未习得的扩散模型：可转移对抗攻击的角度 cs.CV

**SubmitDate**: 2024-04-30    [abs](http://arxiv.org/abs/2404.19382v1) [paper-pdf](http://arxiv.org/pdf/2404.19382v1)

**Authors**: Xiaoxuan Han, Songlin Yang, Wei Wang, Yang Li, Jing Dong

**Abstract**: Advanced text-to-image diffusion models raise safety concerns regarding identity privacy violation, copyright infringement, and Not Safe For Work content generation. Towards this, unlearning methods have been developed to erase these involved concepts from diffusion models. However, these unlearning methods only shift the text-to-image mapping and preserve the visual content within the generative space of diffusion models, leaving a fatal flaw for restoring these erased concepts. This erasure trustworthiness problem needs probe, but previous methods are sub-optimal from two perspectives: (1) Lack of transferability: Some methods operate within a white-box setting, requiring access to the unlearned model. And the learned adversarial input often fails to transfer to other unlearned models for concept restoration; (2) Limited attack: The prompt-level methods struggle to restore narrow concepts from unlearned models, such as celebrity identity. Therefore, this paper aims to leverage the transferability of the adversarial attack to probe the unlearning robustness under a black-box setting. This challenging scenario assumes that the unlearning method is unknown and the unlearned model is inaccessible for optimization, requiring the attack to be capable of transferring across different unlearned models. Specifically, we employ an adversarial search strategy to search for the adversarial embedding which can transfer across different unlearned models. This strategy adopts the original Stable Diffusion model as a surrogate model to iteratively erase and search for embeddings, enabling it to find the embedding that can restore the target concept for different unlearning methods. Extensive experiments demonstrate the transferability of the searched adversarial embedding across several state-of-the-art unlearning methods and its effectiveness for different levels of concepts.

摘要: 先进的文本到图像扩散模型引发了对侵犯身份隐私、侵犯版权和工作内容生成不安全的安全担忧。为此，人们开发了遗忘方法来从扩散模型中删除这些复杂的概念。然而，这些遗忘方法只移动了文本到图像的映射，并在扩散模型的生成空间内保留了视觉内容，为恢复这些被删除的概念留下了一个致命的缺陷。这个擦除可信度问题需要探讨，但以前的方法从两个角度来看是次优的：(1)缺乏可转换性：一些方法在白盒环境下操作，需要访问未学习的模型。而习得的对抗性输入往往无法转移到其他未习得的模型上进行概念恢复；(2)有限攻击：提示水平的方法很难从未习得的模型中恢复狭义的概念，如名人身份。因此，本文旨在利用对抗性攻击的可转移性来探讨黑箱环境下的遗忘健壮性。这一具有挑战性的场景假设遗忘方法未知，并且无法访问未学习模型进行优化，这要求攻击能够在不同的未学习模型之间传输。具体地说，我们采用对抗性搜索策略来搜索可以在不同的未学习模型之间转移的对抗性嵌入。该策略采用原有的稳定扩散模型作为代理模型，迭代地擦除和搜索嵌入，使其能够为不同的遗忘方法找到能够恢复目标概念的嵌入。大量的实验表明，搜索到的对抗性嵌入在几种最新的遗忘方法上是可移植的，并且对于不同级别的概念是有效的。



## **23. Revisiting the Adversarial Robustness of Vision Language Models: a Multimodal Perspective**

重新审视视觉语言模型的对抗鲁棒性：多模式视角 cs.CV

16 pages, 14 figures

**SubmitDate**: 2024-04-30    [abs](http://arxiv.org/abs/2404.19287v1) [paper-pdf](http://arxiv.org/pdf/2404.19287v1)

**Authors**: Wanqi Zhou, Shuanghao Bai, Qibin Zhao, Badong Chen

**Abstract**: Pretrained vision-language models (VLMs) like CLIP have shown impressive generalization performance across various downstream tasks, yet they remain vulnerable to adversarial attacks. While prior research has primarily concentrated on improving the adversarial robustness of image encoders to guard against attacks on images, the exploration of text-based and multimodal attacks has largely been overlooked. In this work, we initiate the first known and comprehensive effort to study adapting vision-language models for adversarial robustness under the multimodal attack. Firstly, we introduce a multimodal attack strategy and investigate the impact of different attacks. We then propose a multimodal contrastive adversarial training loss, aligning the clean and adversarial text embeddings with the adversarial and clean visual features, to enhance the adversarial robustness of both image and text encoders of CLIP. Extensive experiments on 15 datasets across two tasks demonstrate that our method significantly improves the adversarial robustness of CLIP. Interestingly, we find that the model fine-tuned against multimodal adversarial attacks exhibits greater robustness than its counterpart fine-tuned solely against image-based attacks, even in the context of image attacks, which may open up new possibilities for enhancing the security of VLMs.

摘要: 像CLIP这样的预先训练的视觉语言模型(VLM)在各种下游任务中表现出令人印象深刻的泛化性能，但它们仍然容易受到对手的攻击。虽然以前的研究主要集中在提高图像编码器的对抗健壮性以防止对图像的攻击，但对基于文本的和多模式攻击的探索在很大程度上被忽视了。在这项工作中，我们启动了第一个已知和全面的努力，以研究适应视觉语言模型的对手在多模式攻击下的稳健性。首先，我们介绍了一种多模式攻击策略，并研究了不同攻击的影响。然后，我们提出了一种多模式对抗性训练损失，将干净和对抗性的文本嵌入与对抗性和干净的视觉特征相结合，以增强CLIP图像和文本编码者的对抗性健壮性。在两个任务的15个数据集上的大量实验表明，我们的方法显著地提高了CLIP的对抗健壮性。有趣的是，我们发现，与仅针对基于图像的攻击进行微调的模型相比，针对多模式攻击进行微调的模型表现出更强的稳健性，甚至在图像攻击的背景下也是如此，这可能为增强VLM的安全性开辟新的可能性。



## **24. Proof-of-Learning with Incentive Security**

具有激励保障的学习证明 cs.CR

17 pages, 5 figures

**SubmitDate**: 2024-04-30    [abs](http://arxiv.org/abs/2404.09005v3) [paper-pdf](http://arxiv.org/pdf/2404.09005v3)

**Authors**: Zishuo Zhao, Zhixuan Fang, Xuechao Wang, Xi Chen, Yuan Zhou

**Abstract**: Most concurrent blockchain systems rely heavily on the Proof-of-Work (PoW) or Proof-of-Stake (PoS) mechanisms for decentralized consensus and security assurance. However, the substantial energy expenditure stemming from computationally intensive yet meaningless tasks has raised considerable concerns surrounding traditional PoW approaches, The PoS mechanism, while free of energy consumption, is subject to security and economic issues. Addressing these issues, the paradigm of Proof-of-Useful-Work (PoUW) seeks to employ challenges of practical significance as PoW, thereby imbuing energy consumption with tangible value. While previous efforts in Proof of Learning (PoL) explored the utilization of deep learning model training SGD tasks as PoUW challenges, recent research has revealed its vulnerabilities to adversarial attacks and the theoretical hardness in crafting a byzantine-secure PoL mechanism. In this paper, we introduce the concept of incentive-security that incentivizes rational provers to behave honestly for their best interest, bypassing the existing hardness to design a PoL mechanism with computational efficiency, a provable incentive-security guarantee and controllable difficulty. Particularly, our work is secure against two attacks to the recent work of Jia et al. [2021], and also improves the computational overhead from $\Theta(1)$ to $O(\frac{\log E}{E})$. Furthermore, while most recent research assumes trusted problem providers and verifiers, our design also guarantees frontend incentive-security even when problem providers are untrusted, and verifier incentive-security that bypasses the Verifier's Dilemma. By incorporating ML training into blockchain consensus mechanisms with provable guarantees, our research not only proposes an eco-friendly solution to blockchain systems, but also provides a proposal for a completely decentralized computing power market in the new AI age.

摘要: 大多数并发区块链系统严重依赖工作证明(POW)或风险证明(POS)机制来实现去中心化共识和安全保证。然而，计算密集但无意义的任务所产生的大量能源支出引起了人们对传统POW方法的相当大的担忧，POS机制虽然没有能源消耗，但受到安全和经济问题的影响。针对这些问题，有用工作证明(POUW)范式试图将具有实际意义的挑战作为POW来使用，从而使能源消耗具有有形价值。虽然先前在学习证明(Pol)方面的努力探索了利用深度学习模型训练SGD任务作为POW挑战，但最近的研究揭示了它对对手攻击的脆弱性以及在设计拜占庭安全的POL机制方面的理论难度。本文引入激励安全的概念，激励理性的证明者为了他们的最大利益而诚实地行事，绕过现有的困难，设计了一个具有计算效率、可证明的激励安全保证和可控难度的POL机制。特别是，我们的工作是安全的，可以抵抗对Jia等人最近的工作的两次攻击。[2021]并将计算开销从$\theta(1)$提高到$O(\frac{\log E}{E})$。此外，虽然最近的研究假设可信的问题提供者和验证者，但我们的设计也保证了前端激励-安全性，即使问题提供者是不可信的，并且验证者激励-安全绕过了验证者的困境。通过将ML培训融入到具有可证明保证的区块链共识机制中，我们的研究不仅为区块链系统提出了生态友好的解决方案，而且为新AI时代完全去中心化的计算能力市场提供了建议。



## **25. Illusory Attacks: Detectability Matters in Adversarial Attacks on Sequential Decision-Makers**

幻觉攻击：对顺序决策者的对抗性攻击的可检测性很重要 cs.AI

**SubmitDate**: 2024-04-29    [abs](http://arxiv.org/abs/2207.10170v4) [paper-pdf](http://arxiv.org/pdf/2207.10170v4)

**Authors**: Tim Franzmeyer, Stephen McAleer, João F. Henriques, Jakob N. Foerster, Philip H. S. Torr, Adel Bibi, Christian Schroeder de Witt

**Abstract**: Autonomous agents deployed in the real world need to be robust against adversarial attacks on sensory inputs. Robustifying agent policies requires anticipating the strongest attacks possible. We demonstrate that existing observation-space attacks on reinforcement learning agents have a common weakness: while effective, their lack of information-theoretic detectability constraints makes them detectable using automated means or human inspection. Detectability is undesirable to adversaries as it may trigger security escalations. We introduce \eattacks{}, a novel form of adversarial attack on sequential decision-makers that is both effective and of $\epsilon$-bounded statistical detectability. We propose a novel dual ascent algorithm to learn such attacks end-to-end. Compared to existing attacks, we empirically find \eattacks{} to be significantly harder to detect with automated methods, and a small study with human participants (IRB approval under reference R84123/RE001) suggests they are similarly harder to detect for humans. Our findings suggest the need for better anomaly detectors, as well as effective hardware- and system-level defenses. The project website can be found at https://tinyurl.com/illusory-attacks.

摘要: 部署在现实世界中的自主代理需要强大地抵御对感觉输入的敌意攻击。将代理策略规模化需要预测可能最强的攻击。我们证明了现有的对强化学习代理的观察空间攻击有一个共同的弱点：虽然有效，但它们缺乏信息论的可检测性约束，使得它们可以使用自动手段或人工检查来检测。对于对手来说，可探测性是不可取的，因为它可能会引发安全升级。我们引入了一种新的对抗性攻击，它是针对序列决策者的一种新形式的对抗性攻击，它既是有效的，又是$-有界的统计可检测性。我们提出了一种新的双重上升算法来端到端地学习此类攻击。与现有的攻击相比，我们根据经验发现，使用自动方法检测攻击要困难得多，一项针对人类参与者的小型研究(参考R84123/RE001下的IRB批准)表明，对于人类来说，它们同样更难检测到。我们的发现表明，需要更好的异常检测器，以及有效的硬件和系统级防御。该项目的网址为：https://tinyurl.com/illusory-attacks.



## **26. Certification of Speaker Recognition Models to Additive Perturbations**

说话人识别模型对加性扰动的认证 cs.SD

9 pages, 9 figures

**SubmitDate**: 2024-04-29    [abs](http://arxiv.org/abs/2404.18791v1) [paper-pdf](http://arxiv.org/pdf/2404.18791v1)

**Authors**: Dmitrii Korzh, Elvir Karimov, Mikhail Pautov, Oleg Y. Rogov, Ivan Oseledets

**Abstract**: Speaker recognition technology is applied in various tasks ranging from personal virtual assistants to secure access systems. However, the robustness of these systems against adversarial attacks, particularly to additive perturbations, remains a significant challenge. In this paper, we pioneer applying robustness certification techniques to speaker recognition, originally developed for the image domain. In our work, we cover this gap by transferring and improving randomized smoothing certification techniques against norm-bounded additive perturbations for classification and few-shot learning tasks to speaker recognition. We demonstrate the effectiveness of these methods on VoxCeleb 1 and 2 datasets for several models. We expect this work to improve voice-biometry robustness, establish a new certification benchmark, and accelerate research of certification methods in the audio domain.

摘要: 说话人识别技术应用于从个人虚拟助理到安全访问系统的各种任务。然而，这些系统对对抗攻击（特别是对添加性扰动）的鲁棒性仍然是一个重大挑战。在本文中，我们率先将鲁棒性认证技术应用于说话人识别，该技术最初是为图像领域开发的。在我们的工作中，我们通过转移和改进随机平滑认证技术来弥补这一差距，以对抗分类和少数镜头学习任务到说话人识别的规范界添加性扰动。我们在VoxCeleb 1和2数据集上证明了这些方法对于多个模型的有效性。我们希望这项工作能够提高语音生物统计学的稳健性，建立新的认证基准，并加速音频领域认证方法的研究。



## **27. Universal Jailbreak Backdoors from Poisoned Human Feedback**

来自中毒人类反馈的普遍越狱后门 cs.AI

Accepted as conference paper in ICLR 2024

**SubmitDate**: 2024-04-29    [abs](http://arxiv.org/abs/2311.14455v4) [paper-pdf](http://arxiv.org/pdf/2311.14455v4)

**Authors**: Javier Rando, Florian Tramèr

**Abstract**: Reinforcement Learning from Human Feedback (RLHF) is used to align large language models to produce helpful and harmless responses. Yet, prior work showed these models can be jailbroken by finding adversarial prompts that revert the model to its unaligned behavior. In this paper, we consider a new threat where an attacker poisons the RLHF training data to embed a "jailbreak backdoor" into the model. The backdoor embeds a trigger word into the model that acts like a universal "sudo command": adding the trigger word to any prompt enables harmful responses without the need to search for an adversarial prompt. Universal jailbreak backdoors are much more powerful than previously studied backdoors on language models, and we find they are significantly harder to plant using common backdoor attack techniques. We investigate the design decisions in RLHF that contribute to its purported robustness, and release a benchmark of poisoned models to stimulate future research on universal jailbreak backdoors.

摘要: 来自人类反馈的强化学习(RLHF)被用来对齐大型语言模型以产生有益和无害的响应。然而，先前的工作表明，这些模型可以通过找到敌意提示来越狱，这些提示可以将模型恢复到其不一致的行为。在本文中，我们考虑了一种新的威胁，即攻击者在RLHF训练数据中下毒，以便在模型中嵌入“越狱后门”。后门将触发词嵌入到模型中，其作用类似于通用的“sudo命令”：将触发词添加到任何提示中都可以实现有害的响应，而无需搜索敌意提示。通用越狱后门比之前在语言模型上研究的后门要强大得多，我们发现使用常见的后门攻击技术来植入它们要困难得多。我们调查了RLHF中有助于其所谓的健壮性的设计决策，并发布了一个有毒模型的基准，以刺激未来对通用越狱后门的研究。



## **28. Towards Quantitative Evaluation of Explainable AI Methods for Deepfake Detection**

对Deepfake检测的可解释人工智能方法进行定量评估 cs.CV

Accepted for publication, 3rd ACM Int. Workshop on Multimedia AI  against Disinformation (MAD'24) at ACM ICMR'24, June 10, 2024, Phuket,  Thailand. This is the "accepted version"

**SubmitDate**: 2024-04-29    [abs](http://arxiv.org/abs/2404.18649v1) [paper-pdf](http://arxiv.org/pdf/2404.18649v1)

**Authors**: Konstantinos Tsigos, Evlampios Apostolidis, Spyridon Baxevanakis, Symeon Papadopoulos, Vasileios Mezaris

**Abstract**: In this paper we propose a new framework for evaluating the performance of explanation methods on the decisions of a deepfake detector. This framework assesses the ability of an explanation method to spot the regions of a fake image with the biggest influence on the decision of the deepfake detector, by examining the extent to which these regions can be modified through a set of adversarial attacks, in order to flip the detector's prediction or reduce its initial prediction; we anticipate a larger drop in deepfake detection accuracy and prediction, for methods that spot these regions more accurately. Based on this framework, we conduct a comparative study using a state-of-the-art model for deepfake detection that has been trained on the FaceForensics++ dataset, and five explanation methods from the literature. The findings of our quantitative and qualitative evaluations document the advanced performance of the LIME explanation method against the other compared ones, and indicate this method as the most appropriate for explaining the decisions of the utilized deepfake detector.

摘要: 在本文中，我们提出了一种新的框架，用于评估解释方法对深度伪检测器决策的性能。该框架评估了解释方法识别对深伪检测器的决策影响最大的假图像区域的能力，方法是通过检查这些区域可以在多大程度上通过一组对手攻击来修改，以反转检测器的预测或降低其初始预测；对于更准确地识别这些区域的方法，我们预计深伪检测精度和预测会有更大的下降。基于这个框架，我们使用了一个基于FaceForensics++数据集的深度伪检测的最新模型和文献中的五种解释方法进行了比较研究。我们的定量和定性评估结果证明了石灰解释方法相对于其他比较方法的先进性能，并表明该方法最适合解释所使用的深伪探测器的决定。



## **29. Assessing Cybersecurity Vulnerabilities in Code Large Language Models**

评估代码大型语言模型中的网络安全漏洞 cs.CR

**SubmitDate**: 2024-04-29    [abs](http://arxiv.org/abs/2404.18567v1) [paper-pdf](http://arxiv.org/pdf/2404.18567v1)

**Authors**: Md Imran Hossen, Jianyi Zhang, Yinzhi Cao, Xiali Hei

**Abstract**: Instruction-tuned Code Large Language Models (Code LLMs) are increasingly utilized as AI coding assistants and integrated into various applications. However, the cybersecurity vulnerabilities and implications arising from the widespread integration of these models are not yet fully understood due to limited research in this domain. To bridge this gap, this paper presents EvilInstructCoder, a framework specifically designed to assess the cybersecurity vulnerabilities of instruction-tuned Code LLMs to adversarial attacks. EvilInstructCoder introduces the Adversarial Code Injection Engine to automatically generate malicious code snippets and inject them into benign code to poison instruction tuning datasets. It incorporates practical threat models to reflect real-world adversaries with varying capabilities and evaluates the exploitability of instruction-tuned Code LLMs under these diverse adversarial attack scenarios. Through the use of EvilInstructCoder, we conduct a comprehensive investigation into the exploitability of instruction tuning for coding tasks using three state-of-the-art Code LLM models: CodeLlama, DeepSeek-Coder, and StarCoder2, under various adversarial attack scenarios. Our experimental results reveal a significant vulnerability in these models, demonstrating that adversaries can manipulate the models to generate malicious payloads within benign code contexts in response to natural language instructions. For instance, under the backdoor attack setting, by poisoning only 81 samples (0.5\% of the entire instruction dataset), we achieve Attack Success Rate at 1 (ASR@1) scores ranging from 76\% to 86\% for different model families. Our study sheds light on the critical cybersecurity vulnerabilities posed by instruction-tuned Code LLMs and emphasizes the urgent necessity for robust defense mechanisms to mitigate the identified vulnerabilities.

摘要: 指令调优代码大型语言模型(Code LLM)越来越多地被用作人工智能编码助手，并集成到各种应用中。然而，由于这一领域的研究有限，这些模型的广泛集成所产生的网络安全漏洞和影响尚未完全了解。为了弥补这一差距，本文提出了EvilInstructCoder框架，该框架专门设计用于评估指令调谐代码LLM在对抗攻击时的网络安全漏洞。EvilInstructCoder引入了敌意代码注入引擎来自动生成恶意代码片段，并将它们注入良性代码以毒害指令调优数据集。它结合了实用的威胁模型来反映具有不同能力的真实世界的对手，并评估了在这些不同的对抗性攻击场景下指令调优代码LLMS的可利用性。通过使用EvilInstructCoder，我们对CodeLlama、DeepSeek-Coder和StarCoder2三种最新的Code LLM模型在各种对抗性攻击场景下对编码任务指令调优的可利用性进行了全面的调查。我们的实验结果揭示了这些模型中的一个显著漏洞，表明攻击者可以操纵这些模型，以在良性代码上下文中生成恶意有效负载，以响应自然语言指令。例如，在后门攻击设置下，通过仅毒化81个样本(占整个指令数据集的0.5%)，对于不同的模型家族，我们获得的攻击成功率为1(ASR@1)，得分范围从76到86。我们的研究揭示了指令调优代码LLM带来的严重网络安全漏洞，并强调了迫切需要强大的防御机制来缓解已识别的漏洞。



## **30. Machine Learning for Windows Malware Detection and Classification: Methods, Challenges and Ongoing Research**

Windows恶意软件检测和分类的机器学习：方法、挑战和正在进行的研究 cs.CR

**SubmitDate**: 2024-04-29    [abs](http://arxiv.org/abs/2404.18541v1) [paper-pdf](http://arxiv.org/pdf/2404.18541v1)

**Authors**: Daniel Gibert

**Abstract**: In this chapter, readers will explore how machine learning has been applied to build malware detection systems designed for the Windows operating system. This chapter starts by introducing the main components of a Machine Learning pipeline, highlighting the challenges of collecting and maintaining up-to-date datasets. Following this introduction, various state-of-the-art malware detectors are presented, encompassing both feature-based and deep learning-based detectors. Subsequent sections introduce the primary challenges encountered by machine learning-based malware detectors, including concept drift and adversarial attacks. Lastly, this chapter concludes by providing a brief overview of the ongoing research on adversarial defenses.

摘要: 在本章中，读者将探索如何应用机器学习来构建为Windows操作系统设计的恶意软件检测系统。本章首先介绍机器学习管道的主要组件，重点介绍收集和维护最新数据集的挑战。在此介绍之后，我们将介绍各种最先进的恶意软件检测器，其中包括基于特征的检测器和基于深度学习的检测器。后续部分介绍了基于机器学习的恶意软件检测器遇到的主要挑战，包括概念漂移和对抗攻击。最后，本章最后简要概述了正在进行的对抗性防御研究。



## **31. A Systematic Evaluation of Adversarial Attacks against Speech Emotion Recognition Models**

针对语音情感识别模型的对抗性攻击的系统评估 cs.SD

**SubmitDate**: 2024-04-29    [abs](http://arxiv.org/abs/2404.18514v1) [paper-pdf](http://arxiv.org/pdf/2404.18514v1)

**Authors**: Nicolas Facchinetti, Federico Simonetta, Stavros Ntalampiras

**Abstract**: Speech emotion recognition (SER) is constantly gaining attention in recent years due to its potential applications in diverse fields and thanks to the possibility offered by deep learning technologies. However, recent studies have shown that deep learning models can be vulnerable to adversarial attacks. In this paper, we systematically assess this problem by examining the impact of various adversarial white-box and black-box attacks on different languages and genders within the context of SER. We first propose a suitable methodology for audio data processing, feature extraction, and CNN-LSTM architecture. The observed outcomes highlighted the significant vulnerability of CNN-LSTM models to adversarial examples (AEs). In fact, all the considered adversarial attacks are able to significantly reduce the performance of the constructed models. Furthermore, when assessing the efficacy of the attacks, minor differences were noted between the languages analyzed as well as between male and female speech. In summary, this work contributes to the understanding of the robustness of CNN-LSTM models, particularly in SER scenarios, and the impact of AEs. Interestingly, our findings serve as a baseline for a) developing more robust algorithms for SER, b) designing more effective attacks, c) investigating possible defenses, d) improved understanding of the vocal differences between different languages and genders, and e) overall, enhancing our comprehension of the SER task.

摘要: 语音情感识别(SER)由于其在各个领域的潜在应用，以及深度学习技术提供的可能性，近年来不断受到人们的关注。然而，最近的研究表明，深度学习模型很容易受到对手的攻击。在本文中，我们通过考察各种对抗性白盒和黑盒攻击在SER上下文中对不同语言和性别的影响来系统地评估这一问题。我们首先提出了一种适用于音频数据处理、特征提取和CNN-LSTM结构的方法。观察到的结果突显了CNN-LSTM模型对对抗性例子(AEs)的严重脆弱性。事实上，所有考虑的对抗性攻击都会显著降低所构建模型的性能。此外，在评估攻击的效果时，注意到所分析的语言之间以及男性和女性语言之间的微小差异。总之，这项工作有助于理解CNN-LSTM模型的稳健性，特别是在SER场景中，以及AEs的影响。有趣的是，我们的发现为a)开发更健壮的SER算法，b)设计更有效的攻击，c)调查可能的防御，d)改善对不同语言和性别之间的语音差异的理解，以及e)总体上，增强我们对SER任务的理解。



## **32. PriSampler: Mitigating Property Inference of Diffusion Models**

PriSampler：缓解扩散模型的属性推断 cs.CR

**SubmitDate**: 2024-04-29    [abs](http://arxiv.org/abs/2306.05208v2) [paper-pdf](http://arxiv.org/pdf/2306.05208v2)

**Authors**: Hailong Hu, Jun Pang

**Abstract**: Diffusion models have been remarkably successful in data synthesis. However, when these models are applied to sensitive datasets, such as banking and human face data, they might bring up severe privacy concerns. This work systematically presents the first privacy study about property inference attacks against diffusion models, where adversaries aim to extract sensitive global properties of its training set from a diffusion model. Specifically, we focus on the most practical attack scenario: adversaries are restricted to accessing only synthetic data. Under this realistic scenario, we conduct a comprehensive evaluation of property inference attacks on various diffusion models trained on diverse data types, including tabular and image datasets. A broad range of evaluations reveals that diffusion models and their samplers are universally vulnerable to property inference attacks. In response, we propose a new model-agnostic plug-in method PriSampler to mitigate the risks of the property inference of diffusion models. PriSampler can be directly applied to well-trained diffusion models and support both stochastic and deterministic sampling. Extensive experiments illustrate the effectiveness of our defense, and it can lead adversaries to infer the proportion of properties as close as predefined values that model owners wish. Notably, PriSampler also shows its significantly superior performance to diffusion models trained with differential privacy on both model utility and defense performance. This work will elevate the awareness of preventing property inference attacks and encourage privacy-preserving synthetic data release.

摘要: 扩散模型在数据合成方面取得了显著的成功。然而，当这些模型应用于敏感数据集时，如银行和人脸数据，它们可能会带来严重的隐私问题。该工作首次系统地研究了针对扩散模型的属性推理攻击，其中攻击者的目标是从扩散模型中提取其训练集的敏感全局属性。具体地说，我们将重点放在最实际的攻击场景上：对手仅限于访问合成数据。在这一现实场景下，我们对不同数据类型(包括表格数据集和图像数据集)上训练的各种扩散模型的属性推理攻击进行了全面评估。广泛的评估表明，扩散模型及其采样器普遍容易受到属性推理攻击。对此，我们提出了一种新的模型不可知的插件方法PriSsamer来降低扩散模型属性推理的风险。PriSsamer可以直接应用于训练良好的扩散模型，并支持随机和确定性抽样。广泛的实验证明了我们防御的有效性，它可以引导攻击者推断出模型所有者希望的属性比例接近预定义的值。值得注意的是，PriSsamer还显示出在模型效用和防御性能方面显著优于使用差异隐私训练的扩散模型的性能。这项工作将提高人们对防止属性推理攻击的意识，并鼓励保护隐私的合成数据发布。



## **33. Laccolith: Hypervisor-Based Adversary Emulation with Anti-Detection**

Laccolith：基于Hypervisor的Advertising模拟，具有反检测功能 cs.CR

**SubmitDate**: 2024-04-29    [abs](http://arxiv.org/abs/2311.08274v3) [paper-pdf](http://arxiv.org/pdf/2311.08274v3)

**Authors**: Vittorio Orbinato, Marco Carlo Feliciano, Domenico Cotroneo, Roberto Natella

**Abstract**: Advanced Persistent Threats (APTs) represent the most threatening form of attack nowadays since they can stay undetected for a long time. Adversary emulation is a proactive approach for preparing against these attacks. However, adversary emulation tools lack the anti-detection abilities of APTs. We introduce Laccolith, a hypervisor-based solution for adversary emulation with anti-detection to fill this gap. We also present an experimental study to compare Laccolith with MITRE CALDERA, a state-of-the-art solution for adversary emulation, against five popular anti-virus products. We found that CALDERA cannot evade detection, limiting the realism of emulated attacks, even when combined with a state-of-the-art anti-detection framework. Our experiments show that Laccolith can hide its activities from all the tested anti-virus products, thus making it suitable for realistic emulations.

摘要: 高级持续性威胁（APT）是当今最具威胁性的攻击形式，因为它们可以长时间不被发现。Adobile模拟是针对这些攻击做好准备的一种主动方法。然而，对手模拟工具缺乏APT的反检测能力。我们引入了Laccolith，这是一种基于管理程序的解决方案，用于具有反检测功能的对手模拟，以填补这一空白。我们还进行了一项实验研究，将Lacolith与MITRE CALDERA（一种最先进的对手模拟解决方案）与五种流行的防病毒产品进行比较。我们发现，即使与最先进的反检测框架相结合，CALDERA也无法逃避检测，从而限制了模拟攻击的真实性。我们的实验表明，Laccolith可以向所有测试的防病毒产品隐藏其活动，从而使其适合现实模拟。



## **34. ICMarks: A Robust Watermarking Framework for Integrated Circuit Physical Design IP Protection**

ICMarks：用于集成电路物理设计IP保护的稳健水印框架 cs.CR

**SubmitDate**: 2024-04-29    [abs](http://arxiv.org/abs/2404.18407v1) [paper-pdf](http://arxiv.org/pdf/2404.18407v1)

**Authors**: Ruisi Zhang, Rachel Selina Rajarathnam, David Z. Pan, Farinaz Koushanfar

**Abstract**: Physical design watermarking on contemporary integrated circuit (IC) layout encodes signatures without considering the dense connections and design constraints, which could lead to performance degradation on the watermarked products. This paper presents ICMarks, a quality-preserving and robust watermarking framework for modern IC physical design. ICMarks embeds unique watermark signatures during the physical design's placement stage, thereby authenticating the IC layout ownership. ICMarks's novelty lies in (i) strategically identifying a region of cells to watermark with minimal impact on the layout performance and (ii) a two-level watermarking framework for augmented robustness toward potential removal and forging attacks. Extensive evaluations on benchmarks of different design objectives and sizes validate that ICMarks incurs no wirelength and timing metrics degradation, while successfully proving ownership. Furthermore, we demonstrate ICMarks is robust against two major watermarking attack categories, namely, watermark removal and forging attacks; even if the adversaries have prior knowledge of the watermarking schemes, the signatures cannot be removed without significantly undermining the layout quality.

摘要: 现代集成电路(IC)版图上的物理设计水印在未考虑密集连接和设计约束的情况下对签名进行编码，这可能会导致水印产品的性能下降。本文提出了ICMarks，一种用于现代集成电路物理设计的质量保持和健壮的水印框架。ICMarks在物理设计的放置阶段嵌入唯一的水印签名，从而验证IC版图所有权。ICMarks的创新之处在于：(I)战略性地识别要添加水印的单元区域，而对布局性能的影响最小；(Ii)两级水印框架，可增强对潜在删除和伪造攻击的稳健性。对不同设计目标和规模的基准的广泛评估证实，ICMarks在成功证明所有权的同时，不会导致有线长度和计时指标降级。此外，我们证明了ICMarks对两种主要的水印攻击，即水印移除和伪造攻击具有很强的鲁棒性；即使攻击者事先知道水印方案，也无法在不显著损害布局质量的情况下移除签名。



## **35. DRAM-Profiler: An Experimental DRAM RowHammer Vulnerability Profiling Mechanism**

DRM-Profiler：一种实验性的RAM RowHammer漏洞分析机制 cs.CR

6 pages, 6 figures

**SubmitDate**: 2024-04-29    [abs](http://arxiv.org/abs/2404.18396v1) [paper-pdf](http://arxiv.org/pdf/2404.18396v1)

**Authors**: Ranyang Zhou, Jacqueline T. Liu, Nakul Kochar, Sabbir Ahmed, Adnan Siraj Rakin, Shaahin Angizi

**Abstract**: RowHammer stands out as a prominent example, potentially the pioneering one, showcasing how a failure mechanism at the circuit level can give rise to a significant and pervasive security vulnerability within systems. Prior research has approached RowHammer attacks within a static threat model framework. Nonetheless, it warrants consideration within a more nuanced and dynamic model. This paper presents a low-overhead DRAM RowHammer vulnerability profiling technique termed DRAM-Profiler, which utilizes innovative test vectors for categorizing memory cells into distinct security levels. The proposed test vectors intentionally weaken the spatial correlation between the aggressors and victim rows before an attack for evaluation, thus aiding designers in mitigating RowHammer vulnerabilities in the mapping phase. While there has been no previous research showcasing the impact of such profiling to our knowledge, our study methodically assesses 128 commercial DDR4 DRAM products. The results uncover the significant variability among chips from different manufacturers in the type and quantity of RowHammer attacks that can be exploited by adversaries.

摘要: RowHammer作为一个突出的例子脱颖而出，可能是一个先驱，展示了电路级别的故障机制如何会导致系统中严重且普遍的安全漏洞。以前的研究已经在静态威胁模型框架内探讨了RowHammer攻击。尽管如此，它值得在一个更微妙和更动态的模型中考虑。本文提出了一种称为DRAM-Profiler的低开销DRAM RowHammer漏洞分析技术，该技术利用创新的测试向量将存储单元分类为不同的安全级别。提出的测试向量在攻击进行评估之前有意削弱攻击者和受害者行之间的空间相关性，从而帮助设计人员在映射阶段缓解RowHammer漏洞。虽然据我们所知，之前还没有研究表明这种剖析的影响，但我们的研究有条不紊地评估了128种商业DDR4 DRAM产品。结果发现，不同制造商的芯片在可被攻击者利用的RowHammer攻击的类型和数量方面存在显著差异。



## **36. A Survey on Intermediate Fusion Methods for Collaborative Perception Categorized by Real World Challenges**

按现实世界挑战分类的协作感知中间融合方法调查 cs.CV

8 pages, 6 tables

**SubmitDate**: 2024-04-28    [abs](http://arxiv.org/abs/2404.16139v2) [paper-pdf](http://arxiv.org/pdf/2404.16139v2)

**Authors**: Melih Yazgan, Thomas Graf, Min Liu, Tobias Fleck, J. Marius Zoellner

**Abstract**: This survey analyzes intermediate fusion methods in collaborative perception for autonomous driving, categorized by real-world challenges. We examine various methods, detailing their features and the evaluation metrics they employ. The focus is on addressing challenges like transmission efficiency, localization errors, communication disruptions, and heterogeneity. Moreover, we explore strategies to counter adversarial attacks and defenses, as well as approaches to adapt to domain shifts. The objective is to present an overview of how intermediate fusion methods effectively meet these diverse challenges, highlighting their role in advancing the field of collaborative perception in autonomous driving.

摘要: 这项调查分析了自动驾驶协作感知中的中间融合方法，并按现实世界的挑战进行分类。我们检查了各种方法，详细介绍了它们的功能和它们采用的评估指标。重点是解决传输效率、本地化错误、通信中断和多样性等挑战。此外，我们还探索对抗对抗攻击和防御的策略，以及适应领域转变的方法。目标是概述中间融合方法如何有效应对这些多样化的挑战，强调它们在推进自动驾驶协作感知领域中的作用。



## **37. Attack on Scene Flow using Point Clouds**

使用点云攻击场景流 cs.CV

**SubmitDate**: 2024-04-28    [abs](http://arxiv.org/abs/2404.13621v2) [paper-pdf](http://arxiv.org/pdf/2404.13621v2)

**Authors**: Haniyeh Ehsani Oskouie, Mohammad-Shahram Moin, Shohreh Kasaei

**Abstract**: Deep neural networks have made significant advancements in accurately estimating scene flow using point clouds, which is vital for many applications like video analysis, action recognition, and navigation. Robustness of these techniques, however, remains a concern, particularly in the face of adversarial attacks that have been proven to deceive state-of-the-art deep neural networks in many domains. Surprisingly, the robustness of scene flow networks against such attacks has not been thoroughly investigated. To address this problem, the proposed approach aims to bridge this gap by introducing adversarial white-box attacks specifically tailored for scene flow networks. Experimental results show that the generated adversarial examples obtain up to 33.7 relative degradation in average end-point error on the KITTI and FlyingThings3D datasets. The study also reveals the significant impact that attacks targeting point clouds in only one dimension or color channel have on average end-point error. Analyzing the success and failure of these attacks on the scene flow networks and their 2D optical flow network variants show a higher vulnerability for the optical flow networks.

摘要: 深度神经网络在利用点云准确估计场景流量方面取得了重大进展，这对于视频分析、动作识别和导航等许多应用都是至关重要的。然而，这些技术的健壮性仍然是一个令人担忧的问题，特别是在面对已被证明在许多领域欺骗最先进的深度神经网络的对抗性攻击时。令人惊讶的是，场景流网络对此类攻击的健壮性还没有得到彻底的研究。为了解决这个问题，提出的方法旨在通过引入专门为场景流网络量身定做的对抗性白盒攻击来弥合这一差距。实验结果表明，生成的对抗性实例在Kitti和FlyingThings3D数据集上的平均端点误差相对下降高达33.7。研究还揭示了仅以一维或颜色通道中的点云为目标的攻击对平均端点误差的显著影响。分析这些攻击对场景流网络及其二维光流网络变体的成功和失败，表明光流网络具有更高的脆弱性。



## **38. Privacy-Preserving, Dropout-Resilient Aggregation in Decentralized Learning**

去中心化学习中保护隐私、具有辍学弹性的聚合 cs.CR

**SubmitDate**: 2024-04-27    [abs](http://arxiv.org/abs/2404.17984v1) [paper-pdf](http://arxiv.org/pdf/2404.17984v1)

**Authors**: Ali Reza Ghavamipour, Benjamin Zi Hao Zhao, Fatih Turkmen

**Abstract**: Decentralized learning (DL) offers a novel paradigm in machine learning by distributing training across clients without central aggregation, enhancing scalability and efficiency. However, DL's peer-to-peer model raises challenges in protecting against inference attacks and privacy leaks. By forgoing central bottlenecks, DL demands privacy-preserving aggregation methods to protect data from 'honest but curious' clients and adversaries, maintaining network-wide privacy. Privacy-preserving DL faces the additional hurdle of client dropout, clients not submitting updates due to connectivity problems or unavailability, further complicating aggregation.   This work proposes three secret sharing-based dropout resilience approaches for privacy-preserving DL. Our study evaluates the efficiency, performance, and accuracy of these protocols through experiments on datasets such as MNIST, Fashion-MNIST, SVHN, and CIFAR-10. We compare our protocols with traditional secret-sharing solutions across scenarios, including those with up to 1000 clients. Evaluations show that our protocols significantly outperform conventional methods, especially in scenarios with up to 30% of clients dropout and model sizes of up to $10^6$ parameters. Our approaches demonstrate markedly high efficiency with larger models, higher dropout rates, and extensive client networks, highlighting their effectiveness in enhancing decentralized learning systems' privacy and dropout robustness.

摘要: 分散学习为机器学习提供了一种新的范例，通过在没有集中聚集的情况下跨客户分布训练，提高了可伸缩性和效率。然而，DL的点对点模式在防止推理攻击和隐私泄露方面提出了挑战。通过放弃核心瓶颈，DL要求采用保护隐私的聚合方法，以保护数据不受“诚实但好奇的”客户和对手的攻击，从而维护整个网络的隐私。保护隐私的DL面临着客户端退出、客户端由于连接问题或不可用而无法提交更新的额外障碍，从而使聚合进一步复杂化。本文提出了三种基于秘密共享的隐私权保护数据链路丢弃恢复方法。我们的研究通过在MNIST、Fashion-MNIST、SVHN和CIFAR-10等数据集上的实验来评估这些协议的效率、性能和准确性。我们将我们的协议与传统的秘密共享解决方案进行了跨场景的比较，包括拥有多达1000个客户端的场景。评估表明，我们的协议显著优于传统方法，特别是在客户中途退出且模型大小高达10^6$参数的情况下。我们的方法通过更大的模型、更高的辍学率和广泛的客户网络表现出显著的高效率，突出了它们在增强分散学习系统的隐私和辍学稳健性方面的有效性。



## **39. Privacy-Preserving Aggregation for Decentralized Learning with Byzantine-Robustness**

具有拜占庭稳健性的去中心化学习保护隐私的聚合 cs.CR

**SubmitDate**: 2024-04-27    [abs](http://arxiv.org/abs/2404.17970v1) [paper-pdf](http://arxiv.org/pdf/2404.17970v1)

**Authors**: Ali Reza Ghavamipour, Benjamin Zi Hao Zhao, Oguzhan Ersoy, Fatih Turkmen

**Abstract**: Decentralized machine learning (DL) has been receiving an increasing interest recently due to the elimination of a single point of failure, present in Federated learning setting. Yet, it is threatened by the looming threat of Byzantine clients who intentionally disrupt the learning process by broadcasting arbitrary model updates to other clients, seeking to degrade the performance of the global model. In response, robust aggregation schemes have emerged as promising solutions to defend against such Byzantine clients, thereby enhancing the robustness of Decentralized Learning. Defenses against Byzantine adversaries, however, typically require access to the updates of other clients, a counterproductive privacy trade-off that in turn increases the risk of inference attacks on those same model updates.   In this paper, we introduce SecureDL, a novel DL protocol designed to enhance the security and privacy of DL against Byzantine threats. SecureDL~facilitates a collaborative defense, while protecting the privacy of clients' model updates through secure multiparty computation. The protocol employs efficient computation of cosine similarity and normalization of updates to robustly detect and exclude model updates detrimental to model convergence. By using MNIST, Fashion-MNIST, SVHN and CIFAR-10 datasets, we evaluated SecureDL against various Byzantine attacks and compared its effectiveness with four existing defense mechanisms. Our experiments show that SecureDL is effective even in the case of attacks by the malicious majority (e.g., 80% Byzantine clients) while preserving high training accuracy.

摘要: 最近，由于消除了联邦学习环境中存在的单点故障，分散机器学习(DL)受到了越来越多的关注。然而，它受到了拜占庭式客户迫在眉睫的威胁，这些客户故意通过向其他客户广播任意的模型更新来扰乱学习过程，试图降低全球模型的表现。作为回应，稳健的聚合方案已经成为防御此类拜占庭客户端的有前途的解决方案，从而增强了分散学习的稳健性。然而，要防御拜占庭式的对手，通常需要访问其他客户端的更新，这是一种适得其反的隐私权衡，反过来又增加了对相同型号更新的推断攻击的风险。在本文中，我们介绍了一种新的下行协议SecureDL，该协议旨在增强下行链路的安全性和保密性，以抵御拜占庭威胁。SecureDL~有助于协作防御，同时通过安全多方计算保护客户模型更新的隐私。该协议利用高效的余弦相似度计算和归一化更新来稳健地检测和排除不利于模型收敛的模型更新。通过使用MNIST、Fashion-MNIST、SVHN和CIFAR-10数据集，我们评估了SecureDL对各种拜占庭攻击的防御效果，并与现有的四种防御机制进行了比较。我们的实验表明，SecureDL即使在受到恶意多数(例如80%的拜占庭客户端)攻击的情况下也是有效的，同时保持了高的训练准确率。



## **40. Bounding the Expected Robustness of Graph Neural Networks Subject to Node Feature Attacks**

限制受节点特征攻击的图神经网络的预期鲁棒性 cs.LG

Accepted at ICLR 2024

**SubmitDate**: 2024-04-27    [abs](http://arxiv.org/abs/2404.17947v1) [paper-pdf](http://arxiv.org/pdf/2404.17947v1)

**Authors**: Yassine Abbahaddou, Sofiane Ennadir, Johannes F. Lutzeyer, Michalis Vazirgiannis, Henrik Boström

**Abstract**: Graph Neural Networks (GNNs) have demonstrated state-of-the-art performance in various graph representation learning tasks. Recently, studies revealed their vulnerability to adversarial attacks. In this work, we theoretically define the concept of expected robustness in the context of attributed graphs and relate it to the classical definition of adversarial robustness in the graph representation learning literature. Our definition allows us to derive an upper bound of the expected robustness of Graph Convolutional Networks (GCNs) and Graph Isomorphism Networks subject to node feature attacks. Building on these findings, we connect the expected robustness of GNNs to the orthonormality of their weight matrices and consequently propose an attack-independent, more robust variant of the GCN, called the Graph Convolutional Orthonormal Robust Networks (GCORNs). We further introduce a probabilistic method to estimate the expected robustness, which allows us to evaluate the effectiveness of GCORN on several real-world datasets. Experimental experiments showed that GCORN outperforms available defense methods. Our code is publicly available at: \href{https://github.com/Sennadir/GCORN}{https://github.com/Sennadir/GCORN}.

摘要: 图形神经网络(GNN)在各种图形表示学习任务中表现出了最先进的性能。最近，研究揭示了它们在对抗性攻击中的脆弱性。在这项工作中，我们在属性图的背景下从理论上定义了期望健壮性的概念，并将其与图表示学习文献中的对抗性健壮性的经典定义联系起来。我们的定义允许我们推导出图卷积网络(GCNS)和图同构网络在节点特征攻击下的期望健壮性的上界。基于这些发现，我们将GNN的期望健壮性与其权重矩阵的正交性联系起来，从而提出了一种与攻击无关的、更健壮的GCN变体，称为图卷积正交鲁棒网络(GCORNS)。我们进一步介绍了一种估计期望稳健性的概率方法，该方法允许我们在几个真实数据集上评估GCORN的有效性。实验表明，GCORN的性能优于现有的防御方法。我们的代码在以下网址公开提供：\href{https://github.com/Sennadir/GCORN}{https://github.com/Sennadir/GCORN}.



## **41. Frosty: Bringing strong liveness guarantees to the Snow family of consensus protocols**

Frosty：为Snow家族的共识协议带来强大的活力保证 cs.DC

**SubmitDate**: 2024-04-27    [abs](http://arxiv.org/abs/2404.14250v3) [paper-pdf](http://arxiv.org/pdf/2404.14250v3)

**Authors**: Aaron Buchwald, Stephen Buttolph, Andrew Lewis-Pye, Patrick O'Grady, Kevin Sekniqi

**Abstract**: Snowman is the consensus protocol implemented by the Avalanche blockchain and is part of the Snow family of protocols, first introduced through the original Avalanche leaderless consensus protocol. A major advantage of Snowman is that each consensus decision only requires an expected constant communication overhead per processor in the `common' case that the protocol is not under substantial Byzantine attack, i.e. it provides a solution to the scalability problem which ensures that the expected communication overhead per processor is independent of the total number of processors $n$ during normal operation. This is the key property that would enable a consensus protocol to scale to 10,000 or more independent validators (i.e. processors). On the other hand, the two following concerns have remained:   (1) Providing formal proofs of consistency for Snowman has presented a formidable challenge.   (2) Liveness attacks exist in the case that a Byzantine adversary controls more than $O(\sqrt{n})$ processors, slowing termination to more than a logarithmic number of steps.   In this paper, we address the two issues above. We consider a Byzantine adversary that controls at most $f<n/5$ processors. First, we provide a simple proof of consistency for Snowman. Then we supplement Snowman with a `liveness module' that can be triggered in the case that a substantial adversary launches a liveness attack, and which guarantees liveness in this event by temporarily forgoing the communication complexity advantages of Snowman, but without sacrificing these low communication complexity advantages during normal operation.

摘要: 雪人是雪崩区块链实施的共识协议，是雪诺协议家族的一部分，最初是通过最初的雪崩无领导共识协议引入的。Snowman的一个主要优势是，在协议没有受到实质性拜占庭攻击的情况下，每个协商一致的决定只需要每个处理器预期的恒定通信开销，即它提供了对可伸缩性问题的解决方案，该解决方案确保在正常操作期间每个处理器的预期通信开销与处理器总数$n$无关。这是使共识协议能够扩展到10,000个或更多独立验证器(即处理器)的关键属性。另一方面，以下两个问题仍然存在：(1)为雪人提供一致性的正式证据是一个巨大的挑战。(2)当拜占庭敌手控制超过$O(\Sqrt{n})$个处理器时，存在活性攻击，从而将终止速度减慢到超过对数步数。在本文中，我们解决了上述两个问题。我们考虑一个拜占庭对手，它至多控制$f<n/5$处理器。首先，我们为雪人提供了一个简单的一致性证明。然后，我们给Snowman增加了一个活跃度模块，该模块可以在强大的对手发起活跃度攻击的情况下触发，并通过暂时放弃Snowman的通信复杂性优势来保证在这种情况下的活跃性，但在正常运行时不会牺牲这些低通信复杂性的优势。



## **42. Towards Robust Recommendation: A Review and an Adversarial Robustness Evaluation Library**

迈向稳健推荐：评论和对抗稳健性评估库 cs.IR

**SubmitDate**: 2024-04-27    [abs](http://arxiv.org/abs/2404.17844v1) [paper-pdf](http://arxiv.org/pdf/2404.17844v1)

**Authors**: Lei Cheng, Xiaowen Huang, Jitao Sang, Jian Yu

**Abstract**: Recently, recommender system has achieved significant success. However, due to the openness of recommender systems, they remain vulnerable to malicious attacks. Additionally, natural noise in training data and issues such as data sparsity can also degrade the performance of recommender systems. Therefore, enhancing the robustness of recommender systems has become an increasingly important research topic. In this survey, we provide a comprehensive overview of the robustness of recommender systems. Based on our investigation, we categorize the robustness of recommender systems into adversarial robustness and non-adversarial robustness. In the adversarial robustness, we introduce the fundamental principles and classical methods of recommender system adversarial attacks and defenses. In the non-adversarial robustness, we analyze non-adversarial robustness from the perspectives of data sparsity, natural noise, and data imbalance. Additionally, we summarize commonly used datasets and evaluation metrics for evaluating the robustness of recommender systems. Finally, we also discuss the current challenges in the field of recommender system robustness and potential future research directions. Additionally, to facilitate fair and efficient evaluation of attack and defense methods in adversarial robustness, we propose an adversarial robustness evaluation library--ShillingREC, and we conduct evaluations of basic attack models and recommendation models. ShillingREC project is released at https://github.com/chengleileilei/ShillingREC.

摘要: 近年来，推荐系统取得了显著的成功。然而，由于推荐系统的开放性，它们仍然容易受到恶意攻击。此外，训练数据中的自然噪声和数据稀疏性等问题也会降低推荐系统的性能。因此，提高推荐系统的健壮性已成为一个日益重要的研究课题。在本次调查中，我们对推荐系统的健壮性进行了全面的概述。基于我们的研究，我们将推荐系统的健壮性分为对抗性健壮性和非对抗性健壮性。在对抗性健壮性方面，介绍了推荐系统对抗性攻击与防御的基本原理和经典方法。在非对抗性稳健性方面，我们从数据稀疏性、自然噪声和数据不平衡的角度分析了非对抗性稳健性。此外，我们还总结了评价推荐系统健壮性的常用数据集和评价指标。最后，我们还讨论了当前推荐系统健壮性领域面临的挑战和未来可能的研究方向。此外，为了便于公平有效地评估对抗健壮性中的攻防方法，我们提出了一个对抗健壮性评估库--ShillingREC，并对基本攻击模型和推荐模型进行了评估。ShillingREC项目在https://github.com/chengleileilei/ShillingREC.上发布



## **43. Adversarial Examples: Generation Proposal in the Context of Facial Recognition Systems**

对抗性示例：面部识别系统背景下的生成提案 cs.CV

**SubmitDate**: 2024-04-27    [abs](http://arxiv.org/abs/2404.17760v1) [paper-pdf](http://arxiv.org/pdf/2404.17760v1)

**Authors**: Marina Fuster, Ignacio Vidaurreta

**Abstract**: In this paper we investigate the vulnerability that facial recognition systems present to adversarial examples by introducing a new methodology from the attacker perspective. The technique is based on the use of the autoencoder latent space, organized with principal component analysis. We intend to analyze the potential to craft adversarial examples suitable for both dodging and impersonation attacks, against state-of-the-art systems. Our initial hypothesis, which was not strongly favoured by the results, stated that it would be possible to separate between the "identity" and "facial expression" features to produce high-quality examples. Despite the findings not supporting it, the results sparked insights into adversarial examples generation and opened new research avenues in the area.

摘要: 在本文中，我们通过从攻击者的角度引入一种新的方法来研究面部识别系统对对抗性示例存在的漏洞。该技术基于自动编码器潜在空间的使用，并通过主成分分析组织。我们打算分析针对最先进的系统制作适合躲避和模仿攻击的对抗示例的潜力。我们最初的假设并没有得到结果的强烈支持，该假设指出，可以区分“身份”和“面部表情”特征以产生高质量的示例。尽管研究结果不支持这一点，但结果引发了人们对对抗性示例生成的深入见解，并在该领域开辟了新的研究途径。



## **44. Attacking Bayes: On the Adversarial Robustness of Bayesian Neural Networks**

攻击Bayes：关于Bayesian神经网络的对抗鲁棒性 cs.LG

**SubmitDate**: 2024-04-27    [abs](http://arxiv.org/abs/2404.19640v1) [paper-pdf](http://arxiv.org/pdf/2404.19640v1)

**Authors**: Yunzhen Feng, Tim G. J. Rudner, Nikolaos Tsilivis, Julia Kempe

**Abstract**: Adversarial examples have been shown to cause neural networks to fail on a wide range of vision and language tasks, but recent work has claimed that Bayesian neural networks (BNNs) are inherently robust to adversarial perturbations. In this work, we examine this claim. To study the adversarial robustness of BNNs, we investigate whether it is possible to successfully break state-of-the-art BNN inference methods and prediction pipelines using even relatively unsophisticated attacks for three tasks: (1) label prediction under the posterior predictive mean, (2) adversarial example detection with Bayesian predictive uncertainty, and (3) semantic shift detection. We find that BNNs trained with state-of-the-art approximate inference methods, and even BNNs trained with Hamiltonian Monte Carlo, are highly susceptible to adversarial attacks. We also identify various conceptual and experimental errors in previous works that claimed inherent adversarial robustness of BNNs and conclusively demonstrate that BNNs and uncertainty-aware Bayesian prediction pipelines are not inherently robust against adversarial attacks.

摘要: 对抗性的例子被证明会导致神经网络在广泛的视觉和语言任务上失败，但最近的研究表明，贝叶斯神经网络(BNN)对对抗性扰动具有内在的健壮性。在这项工作中，我们研究了这一主张。为了研究BNN的对抗稳健性，我们研究了是否有可能使用相对简单的攻击来成功地打破最新的BNN推理方法和预测管道：(1)后验预测均值下的标签预测，(2)具有贝叶斯预测不确定性的对抗性实例检测，以及(3)语义偏移检测。我们发现，用最先进的近似推理方法训练的BNN，甚至用哈密顿蒙特卡罗训练的BNN，都非常容易受到敌意攻击。我们还在前人的工作中发现了各种概念和实验错误，这些错误声称BNN固有的对抗健壮性，并最终证明BNN和不确定性感知的贝叶斯预测管道对敌意攻击并不固有的健壮性。



## **45. Overload: Latency Attacks on Object Detection for Edge Devices**

过载：边缘设备对象检测的延迟攻击 cs.CV

**SubmitDate**: 2024-04-26    [abs](http://arxiv.org/abs/2304.05370v4) [paper-pdf](http://arxiv.org/pdf/2304.05370v4)

**Authors**: Erh-Chung Chen, Pin-Yu Chen, I-Hsin Chung, Che-rung Lee

**Abstract**: Nowadays, the deployment of deep learning-based applications is an essential task owing to the increasing demands on intelligent services. In this paper, we investigate latency attacks on deep learning applications. Unlike common adversarial attacks for misclassification, the goal of latency attacks is to increase the inference time, which may stop applications from responding to the requests within a reasonable time. This kind of attack is ubiquitous for various applications, and we use object detection to demonstrate how such kind of attacks work. We also design a framework named Overload to generate latency attacks at scale. Our method is based on a newly formulated optimization problem and a novel technique, called spatial attention. This attack serves to escalate the required computing costs during the inference time, consequently leading to an extended inference time for object detection. It presents a significant threat, especially to systems with limited computing resources. We conducted experiments using YOLOv5 models on Nvidia NX. Compared to existing methods, our method is simpler and more effective. The experimental results show that with latency attacks, the inference time of a single image can be increased ten times longer in reference to the normal setting. Moreover, our findings pose a potential new threat to all object detection tasks requiring non-maximum suppression (NMS), as our attack is NMS-agnostic.

摘要: 如今，由于对智能服务的需求不断增加，部署基于深度学习的应用程序是一项重要的任务。本文研究了深度学习应用程序中的延迟攻击。与常见的误分类对抗性攻击不同，延迟攻击的目标是增加推理时间，这可能会使应用程序在合理的时间内停止对请求的响应。这种攻击在各种应用中普遍存在，我们使用对象检测来演示这种攻击是如何工作的。我们还设计了一个名为OverLoad的框架来生成大规模的延迟攻击。我们的方法是基于一个新提出的优化问题和一种新的技术，称为空间注意力。该攻击用于在推理时间内增加所需的计算成本，从而导致对象检测的推理时间延长。它构成了一个重大威胁，特别是对计算资源有限的系统。我们在NVIDIA NX上使用YOLOv5模型进行了实验。与已有的方法相比，我们的方法更简单有效。实验结果表明，在延迟攻击的情况下，单幅图像的推理时间可以比正常设置增加十倍。此外，我们的发现对所有需要非最大抑制(NMS)的目标检测任务构成了潜在的新威胁，因为我们的攻击是与NMS无关的。



## **46. Evaluations of Machine Learning Privacy Defenses are Misleading**

对机器学习隐私辩护的评估具有误导性 cs.CR

**SubmitDate**: 2024-04-26    [abs](http://arxiv.org/abs/2404.17399v1) [paper-pdf](http://arxiv.org/pdf/2404.17399v1)

**Authors**: Michael Aerni, Jie Zhang, Florian Tramèr

**Abstract**: Empirical defenses for machine learning privacy forgo the provable guarantees of differential privacy in the hope of achieving higher utility while resisting realistic adversaries. We identify severe pitfalls in existing empirical privacy evaluations (based on membership inference attacks) that result in misleading conclusions. In particular, we show that prior evaluations fail to characterize the privacy leakage of the most vulnerable samples, use weak attacks, and avoid comparisons with practical differential privacy baselines. In 5 case studies of empirical privacy defenses, we find that prior evaluations underestimate privacy leakage by an order of magnitude. Under our stronger evaluation, none of the empirical defenses we study are competitive with a properly tuned, high-utility DP-SGD baseline (with vacuous provable guarantees).

摘要: 机器学习隐私的经验防御放弃了差异隐私的可证明保证，希望在抵抗现实对手的同时实现更高的效用。我们发现了现有的经验隐私评估（基于成员资格推断攻击）中的严重陷阱，这些陷阱会导致误导性结论。特别是，我们表明，先前的评估未能描述最脆弱样本的隐私泄露，使用弱攻击，并避免与实际的差异隐私基线进行比较。在经验隐私防御的5个案例研究中，我们发现之前的评估低估了隐私泄露一个数量级。在我们更强有力的评估下，我们研究的经验防御都无法与适当调整的、高效用的DP-SGD基线（具有空洞的可证明保证）竞争。



## **47. Enhancing Privacy and Security of Autonomous UAV Navigation**

增强自主无人机导航的隐私和安全性 cs.CR

**SubmitDate**: 2024-04-26    [abs](http://arxiv.org/abs/2404.17225v1) [paper-pdf](http://arxiv.org/pdf/2404.17225v1)

**Authors**: Vatsal Aggarwal, Arjun Ramesh Kaushik, Charanjit Jutla, Nalini Ratha

**Abstract**: Autonomous Unmanned Aerial Vehicles (UAVs) have become essential tools in defense, law enforcement, disaster response, and product delivery. These autonomous navigation systems require a wireless communication network, and of late are deep learning based. In critical scenarios such as border protection or disaster response, ensuring the secure navigation of autonomous UAVs is paramount. But, these autonomous UAVs are susceptible to adversarial attacks through the communication network or the deep learning models - eavesdropping / man-in-the-middle / membership inference / reconstruction. To address this susceptibility, we propose an innovative approach that combines Reinforcement Learning (RL) and Fully Homomorphic Encryption (FHE) for secure autonomous UAV navigation. This end-to-end secure framework is designed for real-time video feeds captured by UAV cameras and utilizes FHE to perform inference on encrypted input images. While FHE allows computations on encrypted data, certain computational operators are yet to be implemented. Convolutional neural networks, fully connected neural networks, activation functions and OpenAI Gym Library are meticulously adapted to the FHE domain to enable encrypted data processing. We demonstrate the efficacy of our proposed approach through extensive experimentation. Our proposed approach ensures security and privacy in autonomous UAV navigation with negligible loss in performance.

摘要: 无人驾驶飞行器(UAV)已成为国防、执法、救灾和产品交付的重要工具。这些自主导航系统需要无线通信网络，最近还基于深度学习。在边境保护或灾难应对等关键场景中，确保自动无人机的安全导航至关重要。但是，这些自主无人机通过通信网络或深度学习模型-窃听/中间人/成员推理/重构-容易受到敌意攻击。为了解决这种敏感性，我们提出了一种结合强化学习(RL)和完全同态加密(FHE)的安全自主无人机导航的创新方法。这个端到端的安全框架是为无人机摄像头捕获的实时视频馈送而设计的，并利用FHE对加密的输入图像执行推理。虽然FHE允许对加密数据进行计算，但某些计算运算符尚未实现。卷积神经网络、全连接神经网络、激活函数和OpenAI Gym库精心适应FHE域，以实现加密数据处理。我们通过广泛的实验证明了我们所提出的方法的有效性。我们提出的方法在保证无人机自主导航的安全性和隐私性的同时，性能损失可以忽略不计。



## **48. Time-Frequency Jointed Imperceptible Adversarial Attack to Brainprint Recognition with Deep Learning Models**

深度学习模型对脑纹识别的时频联合不可感知的对抗攻击 cs.CR

This work is accepted by ICME 2024

**SubmitDate**: 2024-04-26    [abs](http://arxiv.org/abs/2403.10021v2) [paper-pdf](http://arxiv.org/pdf/2403.10021v2)

**Authors**: Hangjie Yi, Yuhang Ming, Dongjun Liu, Wanzeng Kong

**Abstract**: EEG-based brainprint recognition with deep learning models has garnered much attention in biometric identification. Yet, studies have indicated vulnerability to adversarial attacks in deep learning models with EEG inputs. In this paper, we introduce a novel adversarial attack method that jointly attacks time-domain and frequency-domain EEG signals by employing wavelet transform. Different from most existing methods which only target time-domain EEG signals, our method not only takes advantage of the time-domain attack's potent adversarial strength but also benefits from the imperceptibility inherent in frequency-domain attack, achieving a better balance between attack performance and imperceptibility. Extensive experiments are conducted in both white- and grey-box scenarios and the results demonstrate that our attack method achieves state-of-the-art attack performance on three datasets and three deep-learning models. In the meanwhile, the perturbations in the signals attacked by our method are barely perceptible to the human visual system.

摘要: 基于深度学习模型的脑电脑纹识别在生物特征识别中得到了广泛的关注。然而，研究表明，在有脑电输入的深度学习模型中，容易受到对抗性攻击。本文提出了一种利用小波变换联合攻击时频域脑电信号的对抗性攻击方法。不同于现有的大多数只针对时域脑电信号的方法，我们的方法不仅利用了时域攻击的强大对抗能力，而且得益于频域攻击固有的不可感知性，在攻击性能和不可感知性之间取得了更好的平衡。在白盒和灰盒场景下进行了大量的实验，结果表明，我们的攻击方法在三个数据集和三个深度学习模型上获得了最先进的攻击性能。同时，我们的方法攻击的信号中的扰动几乎不被人类视觉系统察觉到。



## **49. Toward Evaluating Robustness of Reinforcement Learning with Adversarial Policy**

利用对抗策略评估强化学习的鲁棒性 cs.LG

Accepted by DSN 2024

**SubmitDate**: 2024-04-26    [abs](http://arxiv.org/abs/2305.02605v3) [paper-pdf](http://arxiv.org/pdf/2305.02605v3)

**Authors**: Xiang Zheng, Xingjun Ma, Shengjie Wang, Xinyu Wang, Chao Shen, Cong Wang

**Abstract**: Reinforcement learning agents are susceptible to evasion attacks during deployment. In single-agent environments, these attacks can occur through imperceptible perturbations injected into the inputs of the victim policy network. In multi-agent environments, an attacker can manipulate an adversarial opponent to influence the victim policy's observations indirectly. While adversarial policies offer a promising technique to craft such attacks, current methods are either sample-inefficient due to poor exploration strategies or require extra surrogate model training under the black-box assumption. To address these challenges, in this paper, we propose Intrinsically Motivated Adversarial Policy (IMAP) for efficient black-box adversarial policy learning in both single- and multi-agent environments. We formulate four types of adversarial intrinsic regularizers -- maximizing the adversarial state coverage, policy coverage, risk, or divergence -- to discover potential vulnerabilities of the victim policy in a principled way. We also present a novel bias-reduction method to balance the extrinsic objective and the adversarial intrinsic regularizers adaptively. Our experiments validate the effectiveness of the four types of adversarial intrinsic regularizers and the bias-reduction method in enhancing black-box adversarial policy learning across a variety of environments. Our IMAP successfully evades two types of defense methods, adversarial training and robust regularizer, decreasing the performance of the state-of-the-art robust WocaR-PPO agents by 34\%-54\% across four single-agent tasks. IMAP also achieves a state-of-the-art attacking success rate of 83.91\% in the multi-agent game YouShallNotPass. Our code is available at \url{https://github.com/x-zheng16/IMAP}.

摘要: 强化学习代理在部署过程中容易受到逃避攻击。在单代理环境中，这些攻击可以通过注入受害者策略网络的输入的不可察觉的扰动来发生。在多智能体环境中，攻击者可以操纵对手来间接影响受害者策略的观察。虽然对抗性策略提供了一种很有希望的技术来策划这样的攻击，但目前的方法要么由于糟糕的探索策略而样本效率低下，要么需要在黑箱假设下进行额外的代理模型训练。为了应对这些挑战，在本文中，我们提出了内在激励的对抗策略(IMAP)，用于在单代理和多代理环境中有效的黑盒对抗策略学习。我们制定了四种类型的对抗性内在规则化--最大化对抗性状态覆盖、保单覆盖、风险或分歧--以原则性的方式发现受害者政策的潜在脆弱性。我们还提出了一种新的减少偏差的方法，以自适应地平衡外部目标和对抗性的内在正则化。我们的实验验证了四种对抗性内在正则化方法和偏差减少方法在增强各种环境下的黑箱对抗性策略学习方面的有效性。我们的IMAP成功地避开了对抗性训练和稳健正则化两种防御方法，使最先进的稳健WocaR-PPO代理在四个单代理任务中的性能降低了34-54。IMAP在多智能体游戏YouShallNotPass中的进攻成功率也达到了83.91\%。我们的代码可在\url{https://github.com/x-zheng16/IMAP}.



## **50. SoK: On the Semantic AI Security in Autonomous Driving**

SoK：关于自动驾驶中的语义人工智能安全 cs.CR

Project website: https://sites.google.com/view/cav-sec/pass

**SubmitDate**: 2024-04-26    [abs](http://arxiv.org/abs/2203.05314v2) [paper-pdf](http://arxiv.org/pdf/2203.05314v2)

**Authors**: Junjie Shen, Ningfei Wang, Ziwen Wan, Yunpeng Luo, Takami Sato, Zhisheng Hu, Xinyang Zhang, Shengjian Guo, Zhenyu Zhong, Kang Li, Ziming Zhao, Chunming Qiao, Qi Alfred Chen

**Abstract**: Autonomous Driving (AD) systems rely on AI components to make safety and correct driving decisions. Unfortunately, today's AI algorithms are known to be generally vulnerable to adversarial attacks. However, for such AI component-level vulnerabilities to be semantically impactful at the system level, it needs to address non-trivial semantic gaps both (1) from the system-level attack input spaces to those at AI component level, and (2) from AI component-level attack impacts to those at the system level. In this paper, we define such research space as semantic AI security as opposed to generic AI security. Over the past 5 years, increasingly more research works are performed to tackle such semantic AI security challenges in AD context, which has started to show an exponential growth trend.   In this paper, we perform the first systematization of knowledge of such growing semantic AD AI security research space. In total, we collect and analyze 53 such papers, and systematically taxonomize them based on research aspects critical for the security field. We summarize 6 most substantial scientific gaps observed based on quantitative comparisons both vertically among existing AD AI security works and horizontally with security works from closely-related domains. With these, we are able to provide insights and potential future directions not only at the design level, but also at the research goal, methodology, and community levels. To address the most critical scientific methodology-level gap, we take the initiative to develop an open-source, uniform, and extensible system-driven evaluation platform, named PASS, for the semantic AD AI security research community. We also use our implemented platform prototype to showcase the capabilities and benefits of such a platform using representative semantic AD AI attacks.

摘要: 自动驾驶(AD)系统依靠人工智能组件来做出安全和正确的驾驶决策。不幸的是，众所周知，今天的人工智能算法通常容易受到对手的攻击。然而，要使此类AI组件级别的漏洞在系统级别产生语义影响，它需要解决以下两个方面的重要语义差距：(1)从系统级别的攻击输入空间到AI组件级别的输入空间，以及(2)从AI组件级别的攻击影响到系统级别的影响。在本文中，我们将这样的研究空间定义为语义AI安全，而不是一般AI安全。在过去的5年里，越来越多的研究工作被用来应对AD环境下的语义AI安全挑战，并开始呈现出指数增长的趋势。在本文中，我们首次对这种不断增长的语义AD AI安全研究空间的知识进行了系统化。我们总共收集和分析了53篇这样的论文，并根据对安全领域至关重要的研究方面对它们进行了系统的分类。我们总结了基于定量比较观察到的6个最实质性的科学差距，这6个差距既包括现有AD AI安全作品之间的纵向比较，也包括与密切相关领域的安全作品的横向比较。有了这些，我们不仅能够在设计层面上提供见解和潜在的未来方向，而且能够在研究目标、方法和社区层面上提供见解和潜在的未来方向。为了解决最关键的科学方法论层面的差距，我们主动开发了一个开源、统一和可扩展的系统驱动的评估平台，名为PASS，用于语义AD AI安全研究社区。我们还使用我们实现的平台原型来展示这样一个使用典型语义AD AI攻击的平台的能力和好处。



