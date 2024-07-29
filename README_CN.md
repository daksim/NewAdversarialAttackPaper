# Latest Adversarial Attack Papers
**update at 2024-07-29 09:59:37**

翻译来自 https://cloud.tencent.com/document/product/551/15619

## **1. Accurate and Scalable Detection and Investigation of Cyber Persistence Threats**

对网络持久性威胁进行准确且可扩展的检测和调查 cs.CR

16 pages

**SubmitDate**: 2024-07-26    [abs](http://arxiv.org/abs/2407.18832v1) [paper-pdf](http://arxiv.org/pdf/2407.18832v1)

**Authors**: Qi Liu, Muhammad Shoaib, Mati Ur Rehman, Kaibin Bao, Veit Hagenmeyer, Wajih Ul Hassan

**Abstract**: In Advanced Persistent Threat (APT) attacks, achieving stealthy persistence within target systems is often crucial for an attacker's success. This persistence allows adversaries to maintain prolonged access, often evading detection mechanisms. Recognizing its pivotal role in the APT lifecycle, this paper introduces Cyber Persistence Detector (CPD), a novel system dedicated to detecting cyber persistence through provenance analytics. CPD is founded on the insight that persistent operations typically manifest in two phases: the "persistence setup" and the subsequent "persistence execution". By causally relating these phases, we enhance our ability to detect persistent threats. First, CPD discerns setups signaling an impending persistent threat and then traces processes linked to remote connections to identify persistence execution activities. A key feature of our system is the introduction of pseudo-dependency edges (pseudo-edges), which effectively connect these disjoint phases using data provenance analysis, and expert-guided edges, which enable faster tracing and reduced log size. These edges empower us to detect persistence threats accurately and efficiently. Moreover, we propose a novel alert triage algorithm that further reduces false positives associated with persistence threats. Evaluations conducted on well-known datasets demonstrate that our system reduces the average false positive rate by 93% compared to state-of-the-art methods.

摘要: 在高级持久威胁(APT)攻击中，在目标系统内实现隐形持久性通常是攻击者成功的关键。这种持久性使攻击者能够保持长时间的访问，通常可以避开检测机制。认识到其在APT生命周期中的关键作用，本文引入了网络持久性检测器(CPD)，这是一个致力于通过来源分析来检测网络持久性的新系统。CPD建立在持久化操作通常表现为两个阶段的洞察力之上：“持久化设置”和随后的“持久化执行”。通过将这些阶段因果联系起来，我们增强了检测持续威胁的能力。首先，CPD识别发出即将到来的持久威胁的设置，然后跟踪链接到远程连接的进程，以识别持久执行活动。我们系统的一个关键特征是引入了伪依赖边(伪边)，它使用数据来源分析有效地连接了这些不相交的阶段，以及专家指导的边，从而实现了更快的跟踪和减少日志大小。这些边缘使我们能够准确高效地检测持久性威胁。此外，我们还提出了一种新的警报分类算法，进一步减少了与持久性威胁相关的误报。在知名数据集上进行的评估表明，与最先进的方法相比，我们的系统平均假阳性率降低了93%。



## **2. Frosty: Bringing strong liveness guarantees to the Snow family of consensus protocols**

Frosty：为Snow家族的共识协议带来强大的活力保证 cs.DC

**SubmitDate**: 2024-07-26    [abs](http://arxiv.org/abs/2404.14250v4) [paper-pdf](http://arxiv.org/pdf/2404.14250v4)

**Authors**: Aaron Buchwald, Stephen Buttolph, Andrew Lewis-Pye, Patrick O'Grady, Kevin Sekniqi

**Abstract**: Snowman is the consensus protocol implemented by the Avalanche blockchain and is part of the Snow family of protocols, first introduced through the original Avalanche leaderless consensus protocol. A major advantage of Snowman is that each consensus decision only requires an expected constant communication overhead per processor in the `common' case that the protocol is not under substantial Byzantine attack, i.e. it provides a solution to the scalability problem which ensures that the expected communication overhead per processor is independent of the total number of processors $n$ during normal operation. This is the key property that would enable a consensus protocol to scale to 10,000 or more independent validators (i.e. processors). On the other hand, the two following concerns have remained:   (1) Providing formal proofs of consistency for Snowman has presented a formidable challenge.   (2) Liveness attacks exist in the case that a Byzantine adversary controls more than $O(\sqrt{n})$ processors, slowing termination to more than a logarithmic number of steps.   In this paper, we address the two issues above. We consider a Byzantine adversary that controls at most $f<n/5$ processors. First, we provide a simple proof of consistency for Snowman. Then we supplement Snowman with a `liveness module' that can be triggered in the case that a substantial adversary launches a liveness attack, and which guarantees liveness in this event by temporarily forgoing the communication complexity advantages of Snowman, but without sacrificing these low communication complexity advantages during normal operation.

摘要: 雪人是雪崩区块链实施的共识协议，是雪诺协议家族的一部分，最初是通过最初的雪崩无领导共识协议引入的。Snowman的一个主要优势是，在协议没有受到实质性拜占庭攻击的情况下，每个协商一致的决定只需要每个处理器预期的恒定通信开销，即它提供了对可伸缩性问题的解决方案，该解决方案确保在正常操作期间每个处理器的预期通信开销与处理器总数$n$无关。这是使共识协议能够扩展到10,000个或更多独立验证器(即处理器)的关键属性。另一方面，以下两个问题仍然存在：(1)为雪人提供一致性的正式证据是一个巨大的挑战。(2)当拜占庭敌手控制超过$O(\Sqrt{n})$个处理器时，存在活性攻击，从而将终止速度减慢到超过对数步数。在本文中，我们解决了上述两个问题。我们考虑一个拜占庭对手，它至多控制$f<n/5$处理器。首先，我们为雪人提供了一个简单的一致性证明。然后，我们给Snowman增加了一个活跃度模块，该模块可以在强大的对手发起活跃度攻击的情况下触发，并通过暂时放弃Snowman的通信复杂性优势来保证在这种情况下的活跃性，但在正常运行时不会牺牲这些低通信复杂性的优势。



## **3. Embodied Laser Attack:Leveraging Scene Priors to Achieve Agent-based Robust Non-contact Attacks**

准激光攻击：利用场景先验实现基于代理的鲁棒非接触式攻击 cs.CV

9 pages, 7 figures, Accepted by ACM MM 2024

**SubmitDate**: 2024-07-26    [abs](http://arxiv.org/abs/2312.09554v3) [paper-pdf](http://arxiv.org/pdf/2312.09554v3)

**Authors**: Yitong Sun, Yao Huang, Xingxing Wei

**Abstract**: As physical adversarial attacks become extensively applied in unearthing the potential risk of security-critical scenarios, especially in dynamic scenarios, their vulnerability to environmental variations has also been brought to light. The non-robust nature of physical adversarial attack methods brings less-than-stable performance consequently. Although methods such as EOT have enhanced the robustness of traditional contact attacks like adversarial patches, they fall short in practicality and concealment within dynamic environments such as traffic scenarios. Meanwhile, non-contact laser attacks, while offering enhanced adaptability, face constraints due to a limited optimization space for their attributes, rendering EOT less effective. This limitation underscores the necessity for developing a new strategy to augment the robustness of such practices. To address these issues, this paper introduces the Embodied Laser Attack (ELA), a novel framework that leverages the embodied intelligence paradigm of Perception-Decision-Control to dynamically tailor non-contact laser attacks. For the perception module, given the challenge of simulating the victim's view by full-image transformation, ELA has innovatively developed a local perspective transformation network, based on the intrinsic prior knowledge of traffic scenes and enables effective and efficient estimation. For the decision and control module, ELA trains an attack agent with data-driven reinforcement learning instead of adopting time-consuming heuristic algorithms, making it capable of instantaneously determining a valid attack strategy with the perceived information by well-designed rewards, which is then conducted by a controllable laser emitter. Experimentally, we apply our framework to diverse traffic scenarios both in the digital and physical world, verifying the effectiveness of our method under dynamic successive scenes.

摘要: 随着物理对抗性攻击被广泛应用于挖掘安全关键场景的潜在风险，特别是在动态场景中，它们对环境变化的脆弱性也暴露了出来。因此，物理对抗性攻击方法的非健壮性带来了不稳定的性能。虽然EOT等方法增强了传统接触攻击(如对抗性补丁)的健壮性，但在交通场景等动态环境中缺乏实用性和隐蔽性。同时，非接触式激光攻击虽然提供了增强的适应性，但由于其属性的优化空间有限而面临约束，使得EOT的效率较低。这一局限性突出表明，有必要制定一项新战略，以加强这类做法的稳健性。为了解决这些问题，本文引入了嵌入式激光攻击(ELA)，这是一种利用感知-决策-控制的嵌入式智能范式来动态定制非接触式激光攻击的框架。对于感知模块，针对通过全图像变换模拟受害者视角的挑战，ELA基于交通场景的固有先验知识，创新性地开发了局部透视变换网络，并实现了有效和高效的估计。对于决策与控制模块，ELA用数据驱动的强化学习来训练攻击代理，而不是采用耗时的启发式算法，使其能够通过精心设计的奖励，根据感知的信息即时确定有效的攻击策略，然后由可控的激光发射器进行。实验中，我们将我们的框架应用于数字和物理世界中不同的交通场景，验证了我们的方法在动态连续场景下的有效性。



## **4. Adversarial Robustification via Text-to-Image Diffusion Models**

通过文本到图像扩散模型的对抗性Robusification cs.CV

Code is available at https://github.com/ChoiDae1/robustify-T2I

**SubmitDate**: 2024-07-26    [abs](http://arxiv.org/abs/2407.18658v1) [paper-pdf](http://arxiv.org/pdf/2407.18658v1)

**Authors**: Daewon Choi, Jongheon Jeong, Huiwon Jang, Jinwoo Shin

**Abstract**: Adversarial robustness has been conventionally believed as a challenging property to encode for neural networks, requiring plenty of training data. In the recent paradigm of adopting off-the-shelf models, however, access to their training data is often infeasible or not practical, while most of such models are not originally trained concerning adversarial robustness. In this paper, we develop a scalable and model-agnostic solution to achieve adversarial robustness without using any data. Our intuition is to view recent text-to-image diffusion models as "adaptable" denoisers that can be optimized to specify target tasks. Based on this, we propose: (a) to initiate a denoise-and-classify pipeline that offers provable guarantees against adversarial attacks, and (b) to leverage a few synthetic reference images generated from the text-to-image model that enables novel adaptation schemes. Our experiments show that our data-free scheme applied to the pre-trained CLIP could improve the (provable) adversarial robustness of its diverse zero-shot classification derivatives (while maintaining their accuracy), significantly surpassing prior approaches that utilize the full training data. Not only for CLIP, we also demonstrate that our framework is easily applicable for robustifying other visual classifiers efficiently.

摘要: 传统上，对抗健壮性被认为是神经网络编码的一种具有挑战性的特性，需要大量的训练数据。然而，在最近采用现成模型的范例中，获取其训练数据往往是不可行或不实用的，而大多数这种模型最初并没有接受过关于对手稳健性的培训。在本文中，我们开发了一个可扩展的与模型无关的解决方案，以在不使用任何数据的情况下实现对手健壮性。我们的直觉是将最近的文本到图像扩散模型视为可以优化以指定目标任务的“适应性”消噪器。基于此，我们建议：(A)启动去噪和分类管道，提供针对对手攻击的可证明的保证，以及(B)利用从文本到图像模型生成的几个合成参考图像，从而实现新的适应方案。我们的实验表明，我们的无数据方案应用于预先训练的剪辑，可以提高其各种零射击分类导数的(可证明的)对抗健壮性(同时保持其准确性)，显著超过以往利用全部训练数据的方法。不仅对于CLIP，我们还证明了我们的框架可以很容易地适用于高效地增强其他视觉分类器的鲁棒性。



## **5. Robust VAEs via Generating Process of Noise Augmented Data**

通过生成噪音增强数据的过程实现稳健的VAE cs.LG

**SubmitDate**: 2024-07-26    [abs](http://arxiv.org/abs/2407.18632v1) [paper-pdf](http://arxiv.org/pdf/2407.18632v1)

**Authors**: Hiroo Irobe, Wataru Aoki, Kimihiro Yamazaki, Yuhui Zhang, Takumi Nakagawa, Hiroki Waida, Yuichiro Wada, Takafumi Kanamori

**Abstract**: Advancing defensive mechanisms against adversarial attacks in generative models is a critical research topic in machine learning. Our study focuses on a specific type of generative models - Variational Auto-Encoders (VAEs). Contrary to common beliefs and existing literature which suggest that noise injection towards training data can make models more robust, our preliminary experiments revealed that naive usage of noise augmentation technique did not substantially improve VAE robustness. In fact, it even degraded the quality of learned representations, making VAEs more susceptible to adversarial perturbations. This paper introduces a novel framework that enhances robustness by regularizing the latent space divergence between original and noise-augmented data. Through incorporating a paired probabilistic prior into the standard variational lower bound, our method significantly boosts defense against adversarial attacks. Our empirical evaluations demonstrate that this approach, termed Robust Augmented Variational Auto-ENcoder (RAVEN), yields superior performance in resisting adversarial inputs on widely-recognized benchmark datasets.

摘要: 在产生式模型中提出对抗攻击的防御机制是机器学习中的一个重要研究课题。我们的研究集中在一种特定类型的产生式模型-变分自动编码器(VAE)。与通常的信念和现有文献表明向训练数据注入噪声可以使模型更稳健相反，我们的初步实验表明，幼稚地使用噪声增强技术并没有显著提高VAE的稳健性。事实上，它甚至降低了学习表征的质量，使VAE更容易受到对抗性干扰。本文介绍了一种新的框架，它通过正则化原始数据和噪声增强数据之间的潜在空间发散来增强稳健性。通过在标准变分下界中引入成对概率先验，我们的方法显著增强了对对手攻击的防御。我们的实验评估表明，这种方法被称为稳健的增广变分自动编码器(RAVEN)，在抵抗广泛认可的基准数据集上的敌意输入方面具有优越的性能。



## **6. DistriBlock: Identifying adversarial audio samples by leveraging characteristics of the output distribution**

DistriBlock：通过利用输出分布的特征来识别对抗性音频样本 cs.SD

**SubmitDate**: 2024-07-26    [abs](http://arxiv.org/abs/2305.17000v6) [paper-pdf](http://arxiv.org/pdf/2305.17000v6)

**Authors**: Matías P. Pizarro B., Dorothea Kolossa, Asja Fischer

**Abstract**: Adversarial attacks can mislead automatic speech recognition (ASR) systems into predicting an arbitrary target text, thus posing a clear security threat. To prevent such attacks, we propose DistriBlock, an efficient detection strategy applicable to any ASR system that predicts a probability distribution over output tokens in each time step. We measure a set of characteristics of this distribution: the median, maximum, and minimum over the output probabilities, the entropy of the distribution, as well as the Kullback-Leibler and the Jensen-Shannon divergence with respect to the distributions of the subsequent time step. Then, by leveraging the characteristics observed for both benign and adversarial data, we apply binary classifiers, including simple threshold-based classification, ensembles of such classifiers, and neural networks. Through extensive analysis across different state-of-the-art ASR systems and language data sets, we demonstrate the supreme performance of this approach, with a mean area under the receiver operating characteristic curve for distinguishing target adversarial examples against clean and noisy data of 99% and 97%, respectively. To assess the robustness of our method, we show that adaptive adversarial examples that can circumvent DistriBlock are much noisier, which makes them easier to detect through filtering and creates another avenue for preserving the system's robustness.

摘要: 敌意攻击可以误导自动语音识别(ASR)系统预测任意目标文本，从而构成明显的安全威胁。为了防止此类攻击，我们提出了DistriBlock，这是一种适用于任何ASR系统的有效检测策略，它预测每个时间步输出令牌上的概率分布。我们测量了该分布的一组特征：输出概率的中位数、最大值和最小值，分布的熵，以及关于后续时间步分布的Kullback-Leibler和Jensen-Shannon散度。然后，通过利用对良性数据和恶意数据观察到的特征，我们应用二进制分类器，包括简单的基于阈值的分类、这种分类器的集成和神经网络。通过对不同的ASR系统和语言数据集的广泛分析，我们证明了该方法的最高性能，在干净和有噪声的数据下，接收器操作特征曲线下的平均面积分别为99%和97%。为了评估我们方法的健壮性，我们证明了可以绕过DistriBlock的自适应攻击示例的噪声要大得多，这使得它们更容易通过过滤来检测，并为保持系统的健壮性创造了另一种途径。



## **7. Unveiling Privacy Vulnerabilities: Investigating the Role of Structure in Graph Data**

揭露隐私漏洞：调查结构在图形数据中的作用 cs.LG

In KDD'24; with full appendix

**SubmitDate**: 2024-07-26    [abs](http://arxiv.org/abs/2407.18564v1) [paper-pdf](http://arxiv.org/pdf/2407.18564v1)

**Authors**: Hanyang Yuan, Jiarong Xu, Cong Wang, Ziqi Yang, Chunping Wang, Keting Yin, Yang Yang

**Abstract**: The public sharing of user information opens the door for adversaries to infer private data, leading to privacy breaches and facilitating malicious activities. While numerous studies have concentrated on privacy leakage via public user attributes, the threats associated with the exposure of user relationships, particularly through network structure, are often neglected. This study aims to fill this critical gap by advancing the understanding and protection against privacy risks emanating from network structure, moving beyond direct connections with neighbors to include the broader implications of indirect network structural patterns. To achieve this, we first investigate the problem of Graph Privacy Leakage via Structure (GPS), and introduce a novel measure, the Generalized Homophily Ratio, to quantify the various mechanisms contributing to privacy breach risks in GPS. Based on this insight, we develop a novel graph private attribute inference attack, which acts as a pivotal tool for evaluating the potential for privacy leakage through network structures under worst-case scenarios. To protect users' private data from such vulnerabilities, we propose a graph data publishing method incorporating a learnable graph sampling technique, effectively transforming the original graph into a privacy-preserving version. Extensive experiments demonstrate that our attack model poses a significant threat to user privacy, and our graph data publishing method successfully achieves the optimal privacy-utility trade-off compared to baselines.

摘要: 用户信息的公开共享为攻击者推断私人数据打开了大门，导致隐私被侵犯，并为恶意活动提供便利。虽然许多研究都集中在通过公共用户属性泄露隐私，但与暴露用户关系相关的威胁，特别是通过网络结构，往往被忽视。这项研究旨在通过促进对网络结构产生的隐私风险的理解和保护来填补这一关键空白，超越与邻居的直接连接，包括间接网络结构模式的更广泛影响。为此，我们首先研究了通过结构的图隐私泄露问题，并引入了一种新的度量方法--广义同伦比来量化GPS中导致隐私泄露风险的各种机制。基于这一观点，我们开发了一种新的图私有属性推理攻击，该攻击可以作为评估最坏情况下通过网络结构泄露隐私的可能性的关键工具。为了保护用户的私有数据免受此类漏洞的侵害，我们提出了一种结合了可学习的图采样技术的图数据发布方法，有效地将原始图转换为隐私保护版本。大量的实验表明，我们的攻击模型对用户隐私构成了严重的威胁，并且我们的图数据发布方法成功地实现了相对于基线的最优隐私效用权衡。



## **8. The Janus Interface: How Fine-Tuning in Large Language Models Amplifies the Privacy Risks**

Janus界面：大型语言模型中的微调如何放大隐私风险 cs.CR

This work has been accepted by CCS 2024

**SubmitDate**: 2024-07-26    [abs](http://arxiv.org/abs/2310.15469v3) [paper-pdf](http://arxiv.org/pdf/2310.15469v3)

**Authors**: Xiaoyi Chen, Siyuan Tang, Rui Zhu, Shijun Yan, Lei Jin, Zihao Wang, Liya Su, Zhikun Zhang, XiaoFeng Wang, Haixu Tang

**Abstract**: The rapid advancements of large language models (LLMs) have raised public concerns about the privacy leakage of personally identifiable information (PII) within their extensive training datasets. Recent studies have demonstrated that an adversary could extract highly sensitive privacy data from the training data of LLMs with carefully designed prompts. However, these attacks suffer from the model's tendency to hallucinate and catastrophic forgetting (CF) in the pre-training stage, rendering the veracity of divulged PIIs negligible. In our research, we propose a novel attack, Janus, which exploits the fine-tuning interface to recover forgotten PIIs from the pre-training data in LLMs. We formalize the privacy leakage problem in LLMs and explain why forgotten PIIs can be recovered through empirical analysis on open-source language models. Based upon these insights, we evaluate the performance of Janus on both open-source language models and two latest LLMs, i.e., GPT-3.5-Turbo and LLaMA-2-7b. Our experiment results show that Janus amplifies the privacy risks by over 10 times in comparison with the baseline and significantly outperforms the state-of-the-art privacy extraction attacks including prefix attacks and in-context learning (ICL). Furthermore, our analysis validates that existing fine-tuning APIs provided by OpenAI and Azure AI Studio are susceptible to our Janus attack, allowing an adversary to conduct such an attack at a low cost.

摘要: 大型语言模型(LLM)的快速发展引起了公众对其广泛训练数据集中个人身份信息(PII)隐私泄露的担忧。最近的研究表明，攻击者可以通过精心设计的提示从LLMS的训练数据中提取高度敏感的隐私数据。然而，这些攻击受到模型在预训练阶段的幻觉和灾难性遗忘(CF)的倾向的影响，使得泄露的PII的真实性可以忽略不计。在我们的研究中，我们提出了一种新的攻击，Janus，它利用微调接口从LLMS的训练前数据中恢复被遗忘的PII。我们形式化地描述了LLMS中的隐私泄露问题，并通过对开源语言模型的实证分析解释了为什么被遗忘的PII可以恢复。基于这些见解，我们评估了Janus在开源语言模型和两个最新的LLMS上的性能，即GPT-3.5-Turbo和Llama-2-7b。我们的实验结果表明，Janus将隐私风险放大了10倍以上，并且显著优于目前最先进的隐私提取攻击，包括前缀攻击和上下文中学习(ICL)。此外，我们的分析验证了OpenAI和Azure AI Studio提供的现有微调API容易受到我们的Janus攻击，允许对手以低成本进行此类攻击。



## **9. Dysca: A Dynamic and Scalable Benchmark for Evaluating Perception Ability of LVLMs**

Dysca：评估LVLM感知能力的动态和可扩展基准 cs.CV

**SubmitDate**: 2024-07-26    [abs](http://arxiv.org/abs/2406.18849v2) [paper-pdf](http://arxiv.org/pdf/2406.18849v2)

**Authors**: Jie Zhang, Zhongqi Wang, Mengqi Lei, Zheng Yuan, Bei Yan, Shiguang Shan, Xilin Chen

**Abstract**: Currently many benchmarks have been proposed to evaluate the perception ability of the Large Vision-Language Models (LVLMs). However, most benchmarks conduct questions by selecting images from existing datasets, resulting in the potential data leakage. Besides, these benchmarks merely focus on evaluating LVLMs on the realistic style images and clean scenarios, leaving the multi-stylized images and noisy scenarios unexplored. In response to these challenges, we propose a dynamic and scalable benchmark named Dysca for evaluating LVLMs by leveraging synthesis images. Specifically, we leverage Stable Diffusion and design a rule-based method to dynamically generate novel images, questions and the corresponding answers. We consider 51 kinds of image styles and evaluate the perception capability in 20 subtasks. Moreover, we conduct evaluations under 4 scenarios (i.e., Clean, Corruption, Print Attacking and Adversarial Attacking) and 3 question types (i.e., Multi-choices, True-or-false and Free-form). Thanks to the generative paradigm, Dysca serves as a scalable benchmark for easily adding new subtasks and scenarios. A total of 8 advanced open-source LVLMs with 10 checkpoints are evaluated on Dysca, revealing the drawbacks of current LVLMs. The benchmark is released in \url{https://github.com/Benchmark-Dysca/Dysca}.

摘要: 目前，人们已经提出了许多基准来评估大型视觉语言模型的感知能力。然而，大多数基准测试通过从现有数据集中选择图像来进行问题，从而导致潜在的数据泄漏。此外，这些基准只关注真实感风格的图像和干净的场景来评估LVLMS，而对多风格化的图像和噪声场景没有进行探索。为了应对这些挑战，我们提出了一种动态的、可扩展的基准测试DYSCA，用于利用合成图像来评估LVLMS。具体地说，我们利用稳定扩散并设计了一种基于规则的方法来动态生成新的图像、问题和相应的答案。我们考虑了51种图像风格，并在20个子任务中评估了感知能力。此外，我们还在4个场景(即廉洁、腐败、打印攻击和对抗性攻击)和3个问题类型(即多项选择、对错和自由形式)下进行了评估。多亏了生成性范式，Dysca成为了一个可伸缩的基准，可以轻松添加新的子任务和场景。在Dysca上对8个具有10个检查点的高级开源LVLMS进行了评估，揭示了当前LVLMS的缺陷。该基准测试在\url{https://github.com/Benchmark-Dysca/Dysca}.中发布



## **10. Machine Unlearning using a Multi-GAN based Model**

使用基于多GAN的模型的机器去学习 cs.LG

**SubmitDate**: 2024-07-26    [abs](http://arxiv.org/abs/2407.18467v1) [paper-pdf](http://arxiv.org/pdf/2407.18467v1)

**Authors**: Amartya Hatua, Trung T. Nguyen, Andrew H. Sung

**Abstract**: This article presents a new machine unlearning approach that utilizes multiple Generative Adversarial Network (GAN) based models. The proposed method comprises two phases: i) data reorganization in which synthetic data using the GAN model is introduced with inverted class labels of the forget datasets, and ii) fine-tuning the pre-trained model. The GAN models consist of two pairs of generators and discriminators. The generator discriminator pairs generate synthetic data for the retain and forget datasets. Then, a pre-trained model is utilized to get the class labels of the synthetic datasets. The class labels of synthetic and original forget datasets are inverted. Finally, all combined datasets are used to fine-tune the pre-trained model to get the unlearned model. We have performed the experiments on the CIFAR-10 dataset and tested the unlearned models using Membership Inference Attacks (MIA). The inverted class labels procedure and synthetically generated data help to acquire valuable information that enables the model to outperform state-of-the-art models and other standard unlearning classifiers.

摘要: 提出了一种基于多生成性对抗网络(GAN)模型的机器遗忘方法。该方法包括两个阶段：i)数据重组，其中使用GAN模型的合成数据被引入带有倒排的忘记数据集的类别标签；ii)微调预先训练的模型。GaN模型由两对生成器和鉴别器组成。生成器鉴别器对为保留和忘记数据集生成合成数据。然后，利用预先训练好的模型得到合成数据集的类标签。对合成的和原始的忘记数据集的分类标签进行倒置。最后，使用所有组合的数据集来微调预先训练的模型，以获得未学习的模型。我们在CIFAR-10数据集上进行了实验，并使用隶属度推理攻击(MIA)对未学习模型进行了测试。倒置的分类标签程序和合成生成的数据有助于获取有价值的信息，使该模型的表现优于最先进的模型和其他标准的遗忘分类器。



## **11. Disrupting Diffusion: Token-Level Attention Erasure Attack against Diffusion-based Customization**

扰乱扩散：针对基于扩散的定制的代币级注意力擦除攻击 cs.CV

Accepted by ACM MM2024

**SubmitDate**: 2024-07-26    [abs](http://arxiv.org/abs/2405.20584v2) [paper-pdf](http://arxiv.org/pdf/2405.20584v2)

**Authors**: Yisu Liu, Jinyang An, Wanqian Zhang, Dayan Wu, Jingzi Gu, Zheng Lin, Weiping Wang

**Abstract**: With the development of diffusion-based customization methods like DreamBooth, individuals now have access to train the models that can generate their personalized images. Despite the convenience, malicious users have misused these techniques to create fake images, thereby triggering a privacy security crisis. In light of this, proactive adversarial attacks are proposed to protect users against customization. The adversarial examples are trained to distort the customization model's outputs and thus block the misuse. In this paper, we propose DisDiff (Disrupting Diffusion), a novel adversarial attack method to disrupt the diffusion model outputs. We first delve into the intrinsic image-text relationships, well-known as cross-attention, and empirically find that the subject-identifier token plays an important role in guiding image generation. Thus, we propose the Cross-Attention Erasure module to explicitly "erase" the indicated attention maps and disrupt the text guidance. Besides,we analyze the influence of the sampling process of the diffusion model on Projected Gradient Descent (PGD) attack and introduce a novel Merit Sampling Scheduler to adaptively modulate the perturbation updating amplitude in a step-aware manner. Our DisDiff outperforms the state-of-the-art methods by 12.75% of FDFR scores and 7.25% of ISM scores across two facial benchmarks and two commonly used prompts on average.

摘要: 随着像DreamBooth这样基于扩散的定制方法的发展，个人现在可以训练能够生成他们个性化图像的模型。尽管很方便，但恶意用户滥用这些技术创造了虚假图像，从而引发了隐私安全危机。有鉴于此，提出了主动对抗性攻击来保护用户免受定制。对抗性的例子被训练来扭曲定制模型的输出，从而阻止误用。在本文中，我们提出了一种新的对抗性攻击方法DisDiff(破坏扩散)来破坏扩散模型的输出。我们首先深入研究了内在的图文关系，即众所周知的交叉注意，并经验地发现，主体-标识符号在引导图像生成方面发挥着重要作用。因此，我们提出了交叉注意擦除模块来显式地“擦除”所指示的注意地图并扰乱文本引导。此外，我们还分析了扩散模型的采样过程对投影梯度下降(PGD)攻击的影响，并引入了一种新的优点采样调度器来以步长感知的方式自适应地调制扰动更新幅度。在两个面部基准和两个常用提示上，我们的DisDiff平均比最先进的方法高出12.75%的FDFR分数和7.25%的ISM分数。



## **12. Regret-Optimal Defense Against Stealthy Adversaries: A System Level Approach**

针对潜行对手的遗憾最佳防御：系统级方法 eess.SY

Accepted, IEEE Conference on Decision and Control (CDC), 2024

**SubmitDate**: 2024-07-26    [abs](http://arxiv.org/abs/2407.18448v1) [paper-pdf](http://arxiv.org/pdf/2407.18448v1)

**Authors**: Hiroyasu Tsukamoto, Joudi Hajar, Soon-Jo Chung, Fred Y. Hadaegh

**Abstract**: Modern control designs in robotics, aerospace, and cyber-physical systems heavily depend on real-world data obtained through the system outputs. In the face of system faults and malicious attacks, however, these outputs can be compromised to misrepresent some portion of the system information that critically affects their secure and trustworthy operation. In this paper, we introduce a novel regret-optimal control framework for designing controllers that render a linear system robust against stealthy attacks, including sensor and actuator attacks, as well as external disturbances. In particular, we establish (a) a convex optimization-based system metric to quantify the regret with the worst-case stealthy attack (the true performance minus the optimal performance in hindsight with the knowledge of the stealthy attack), which improves and adaptively interpolates $\mathcal{H}_2$ and $\mathcal{H}_{\infty}$ norms in the presence of stealthy adversaries, (b) an optimization problem for minimizing the regret of 1 expressed in the system level parameterization, which is useful for its localized and distributed implementation in large-scale systems, and (c) a rank-constrained optimization problem (i.e., optimization with a convex objective subject to convex constraints and rank constraints) equivalent to the optimization problem of (b). Finally, we conduct a numerical simulation which showcases the effectiveness of our approach.

摘要: 机器人、航空航天和计算机物理系统中的现代控制设计在很大程度上依赖于通过系统输出获得的真实世界数据。然而，面对系统故障和恶意攻击，这些输出可能会受到损害，从而歪曲系统信息的某些部分，从而严重影响其安全和可信的操作。在本文中，我们介绍了一种新的后悔最优控制框架，用于设计控制器，使线性系统对隐身攻击，包括传感器和执行器攻击，以及外部干扰具有鲁棒性。具体地，我们建立了(A)基于凸优化的系统度量来量化最坏情况下的隐蔽攻击的遗憾(真实性能减去具有隐形攻击知识的后见之明的最优性能)，它改进并自适应地内插在隐身对手存在的情况下的$\数学{H}_2$和$\数学{H}_(Inty)$范数；(B)以系统级参数化表示的最小化1的遗憾的优化问题，这对于其在大规模系统中的局部和分布式实现是有用的；以及(C)秩约束优化问题(即，具有凸约束和秩约束的凸目标最优化)等价于(B)的最优化问题。最后，我们进行了数值模拟，验证了该方法的有效性。



## **13. Human-Interpretable Adversarial Prompt Attack on Large Language Models with Situational Context**

具有情境上下文的大型语言模型的人类可解释对抗提示攻击 cs.CL

**SubmitDate**: 2024-07-25    [abs](http://arxiv.org/abs/2407.14644v2) [paper-pdf](http://arxiv.org/pdf/2407.14644v2)

**Authors**: Nilanjana Das, Edward Raff, Manas Gaur

**Abstract**: Previous research on testing the vulnerabilities in Large Language Models (LLMs) using adversarial attacks has primarily focused on nonsensical prompt injections, which are easily detected upon manual or automated review (e.g., via byte entropy). However, the exploration of innocuous human-understandable malicious prompts augmented with adversarial injections remains limited. In this research, we explore converting a nonsensical suffix attack into a sensible prompt via a situation-driven contextual re-writing. This allows us to show suffix conversion without any gradients, using only LLMs to perform the attacks, and thus better understand the scope of possible risks. We combine an independent, meaningful adversarial insertion and situations derived from movies to check if this can trick an LLM. The situations are extracted from the IMDB dataset, and prompts are defined following a few-shot chain-of-thought prompting. Our approach demonstrates that a successful situation-driven attack can be executed on both open-source and proprietary LLMs. We find that across many LLMs, as few as 1 attempt produces an attack and that these attacks transfer between LLMs.

摘要: 之前关于使用对抗性攻击测试大型语言模型(LLM)中的漏洞的研究主要集中在无意义的提示注入上，这些注入很容易通过手动或自动审查(例如，通过字节熵)检测到。然而，通过恶意注入增强无害的人类可理解的恶意提示的探索仍然有限。在这项研究中，我们探索通过情景驱动的语境重写将无意义的后缀攻击转化为合理的提示。这使我们能够显示没有任何梯度的后缀转换，仅使用LLM来执行攻击，从而更好地了解可能风险的范围。我们结合了一个独立的、有意义的敌意插入和来自电影的情况来检查这是否可以欺骗LLM。情况是从IMDB数据集中提取的，提示是在几个镜头的思维链提示之后定义的。我们的方法表明，成功的情境驱动攻击可以在开源和专有LLM上执行。我们发现，在许多LLM中，只有1次尝试就会产生攻击，并且这些攻击会在LLM之间传输。



## **14. Sparse vs Contiguous Adversarial Pixel Perturbations in Multimodal Models: An Empirical Analysis**

稀疏与连续多峰模型中的对抗像素扰动：实证分析 cs.CV

**SubmitDate**: 2024-07-25    [abs](http://arxiv.org/abs/2407.18251v1) [paper-pdf](http://arxiv.org/pdf/2407.18251v1)

**Authors**: Cristian-Alexandru Botocan, Raphael Meier, Ljiljana Dolamic

**Abstract**: Assessing the robustness of multimodal models against adversarial examples is an important aspect for the safety of its users. We craft L0-norm perturbation attacks on the preprocessed input images. We launch them in a black-box setup against four multimodal models and two unimodal DNNs, considering both targeted and untargeted misclassification. Our attacks target less than 0.04% of perturbed image area and integrate different spatial positioning of perturbed pixels: sparse positioning and pixels arranged in different contiguous shapes (row, column, diagonal, and patch). To the best of our knowledge, we are the first to assess the robustness of three state-of-the-art multimodal models (ALIGN, AltCLIP, GroupViT) against different sparse and contiguous pixel distribution perturbations. The obtained results indicate that unimodal DNNs are more robust than multimodal models. Furthermore, models using CNN-based Image Encoder are more vulnerable than models with ViT - for untargeted attacks, we obtain a 99% success rate by perturbing less than 0.02% of the image area.

摘要: 评估多通道模型对敌意例子的稳健性是保证其使用者安全的一个重要方面。我们对经过预处理的输入图像进行L0范数扰动攻击。我们在黑盒设置中针对四个多模式模型和两个单峰DNN启动，同时考虑了目标和非目标错误分类。我们的攻击目标是不超过0.04%的扰动图像区域，并整合了扰动像素的不同空间位置：稀疏定位和以不同的连续形状(行、列、对角线和面片)排列的像素。据我们所知，我们首次评估了三种最先进的多模式模型(ALIGN、AltCLIP、GroupViT)在不同稀疏和连续像素分布扰动下的稳健性。结果表明，单模DNN比多模DNN具有更好的鲁棒性。此外，使用基于CNN的图像编码器的模型比使用VIT的模型更容易受到攻击-对于非目标攻击，我们通过扰动不到0.02%的图像区域获得99%的成功率。



## **15. Dr. Jekyll and Mr. Hyde: Two Faces of LLMs**

杰基尔博士和海德先生：法学硕士的两面 cs.CR

**SubmitDate**: 2024-07-25    [abs](http://arxiv.org/abs/2312.03853v4) [paper-pdf](http://arxiv.org/pdf/2312.03853v4)

**Authors**: Matteo Gioele Collu, Tom Janssen-Groesbeek, Stefanos Koffas, Mauro Conti, Stjepan Picek

**Abstract**: Recently, we have witnessed a rise in the use of Large Language Models (LLMs), especially in applications like chatbot assistants. Safety mechanisms and specialized training procedures are implemented to prevent improper responses from these assistants. In this work, we bypass these measures for ChatGPT and Gemini (and, to some extent, Bing chat) by making them impersonate complex personas with personality characteristics that are not aligned with a truthful assistant. We start by creating elaborate biographies of these personas, which we then use in a new session with the same chatbots. Our conversations then follow a role-play style to elicit prohibited responses. Using personas, we show that prohibited responses are actually provided, making it possible to obtain unauthorized, illegal, or harmful information. This work shows that by using adversarial personas, one can overcome safety mechanisms set out by ChatGPT and Gemini. We also introduce several ways of activating such adversarial personas, which show that both chatbots are vulnerable to this kind of attack. With the same principle, we introduce two defenses that push the model to interpret trustworthy personalities and make it more robust against such attacks.

摘要: 最近，我们看到大型语言模型(LLM)的使用有所增加，特别是在聊天机器人助手等应用程序中。实施了安全机制和专门的培训程序，以防止这些助理做出不当反应。在这项工作中，我们绕过了ChatGPT和Gemini(在某种程度上，Bing聊天)的这些措施，让他们模仿具有与诚实的助手不一致的个性特征的复杂人物角色。我们首先为这些角色创建精致的传记，然后在与相同的聊天机器人的新会话中使用。然后，我们的对话遵循角色扮演的风格，以引发被禁止的回应。使用人物角色，我们展示了实际上提供了被禁止的响应，使得获得未经授权的、非法的或有害的信息成为可能。这项工作表明，通过使用敌对的人物角色，一个人可以克服ChatGPT和Gemini提出的安全机制。我们还介绍了几种激活这种敌对角色的方法，这表明这两个聊天机器人都容易受到这种攻击。在相同的原则下，我们引入了两个防御措施，推动该模型解释可信任的个性，并使其对此类攻击更加健壮。



## **16. RIDA: A Robust Attack Framework on Incomplete Graphs**

RIDA：一个针对不完整图的稳健攻击框架 cs.LG

**SubmitDate**: 2024-07-25    [abs](http://arxiv.org/abs/2407.18170v1) [paper-pdf](http://arxiv.org/pdf/2407.18170v1)

**Authors**: Jianke Yu, Hanchen Wang, Chen Chen, Xiaoyang Wang, Wenjie Zhang, Ying Zhang

**Abstract**: Graph Neural Networks (GNNs) are vital in data science but are increasingly susceptible to adversarial attacks. To help researchers develop more robust GNN models, it's essential to focus on designing strong attack models as foundational benchmarks and guiding references. Among adversarial attacks, gray-box poisoning attacks are noteworthy due to their effectiveness and fewer constraints. These attacks exploit GNNs' need for retraining on updated data, thereby impacting their performance by perturbing these datasets. However, current research overlooks the real-world scenario of incomplete graphs.To address this gap, we introduce the Robust Incomplete Deep Attack Framework (RIDA). It is the first algorithm for robust gray-box poisoning attacks on incomplete graphs. The approach innovatively aggregates distant vertex information and ensures powerful data utilization.Extensive tests against 9 SOTA baselines on 3 real-world datasets demonstrate RIDA's superiority in handling incompleteness and high attack performance on the incomplete graph.

摘要: 图神经网络(GNN)在数据科学中至关重要，但越来越容易受到对手攻击。为了帮助研究人员开发更健壮的GNN模型，有必要将重点放在设计强大的攻击模型作为基础基准和指导参考。在对抗性攻击中，灰箱中毒攻击由于其有效性和较少的约束而值得注意。这些攻击利用了GNN对更新数据进行再培训的需要，从而通过扰乱这些数据集来影响其性能。然而，目前的研究忽略了现实世界中不完整图形的场景，为了解决这一差距，我们引入了健壮的不完整深度攻击框架(RIDA)。这是第一个针对不完备图的稳健灰盒中毒攻击的算法。该方法创新性地聚合了距离较远的顶点信息，确保了强大的数据利用率，并在3个真实数据集上对9条SOTA基线进行了扩展测试，验证了RIDA在处理不完全图的不完备性和高攻击性能方面的优势。



## **17. Understanding the Security Benefits and Overheads of Emerging Industry Solutions to DRAM Read Disturbance**

了解新兴行业解决方案的安全优势和管理费用针对内存读取干扰 cs.CR

To appear in DRAMSec 2024

**SubmitDate**: 2024-07-25    [abs](http://arxiv.org/abs/2406.19094v2) [paper-pdf](http://arxiv.org/pdf/2406.19094v2)

**Authors**: Oğuzhan Canpolat, A. Giray Yağlıkçı, Geraldo F. Oliveira, Ataberk Olgun, Oğuz Ergin, Onur Mutlu

**Abstract**: We present the first rigorous security, performance, energy, and cost analyses of the state-of-the-art on-DRAM-die read disturbance mitigation method, Per Row Activation Counting (PRAC), described in JEDEC DDR5 specification's April 2024 update. Unlike prior state-of-the-art that advises the memory controller to periodically issue refresh management (RFM) commands, which provides the DRAM chip with time to perform refreshes, PRAC introduces a new back-off signal. PRAC's back-off signal propagates from the DRAM chip to the memory controller and forces the memory controller to 1) stop serving requests and 2) issue RFM commands. As a result, RFM commands are issued when needed as opposed to periodically, reducing RFM's overheads. We analyze PRAC in four steps. First, we define an adversarial access pattern that represents the worst-case for PRAC's security. Second, we investigate PRAC's configurations and security implications. Our analyses show that PRAC can be configured for secure operation as long as no bitflip occurs before accessing a memory location 10 times. Third, we evaluate the performance impact of PRAC and compare it against prior works using Ramulator 2.0. Our analysis shows that while PRAC incurs less than 13% performance overhead for today's DRAM chips, its performance overheads can reach up to 94% for future DRAM chips that are more vulnerable to read disturbance bitflips. Fourth, we define an availability adversarial access pattern that exacerbates PRAC's performance overhead to perform a memory performance attack, demonstrating that such an adversarial pattern can hog up to 94% of DRAM throughput and degrade system throughput by up to 95%. We discuss PRAC's implications on future systems and foreshadow future research directions. To aid future research, we open-source our implementations and scripts at https://github.com/CMU-SAFARI/ramulator2.

摘要: 我们首次对JEDEC DDR5规范2024年4月更新中描述的最先进的片上DRAM读取干扰缓解方法-每行激活计数(PRAC)-进行了严格的安全、性能、能量和成本分析。与建议存储器控制器定期发出刷新管理(RFM)命令(为DRAM芯片提供执行刷新的时间)的现有技术不同，PRAC引入了新的退避信号。PRAC的退避信号从DRAM芯片传播到存储器控制器，并迫使存储器控制器1)停止服务请求和2)发出RFM命令。因此，RFM命令在需要时发出，而不是定期发出，从而减少了RFM的管理费用。我们分四个步骤对PRAC进行分析。首先，我们定义了一种对抗性访问模式，它代表了对PRAC安全的最坏情况。其次，我们调查了PRAC的配置和安全影响。我们的分析表明，只要在访问一个存储单元10次之前没有发生位翻转，就可以将PRAC配置为安全操作。第三，我们评估了PRAC对性能的影响，并将其与使用Ramuler2.0的前人工作进行了比较。我们的分析表明，虽然PRAC对今天的DRAM芯片产生的性能开销不到13%，但对于更容易受到读取干扰位翻转的未来DRAM芯片，其性能开销可能高达94%。第四，我们定义了一种可用性对抗性访问模式，它加剧了PRAC执行内存性能攻击的性能开销，证明了这种对抗性模式可以占用高达94%的DRAM吞吐量，并使系统吞吐量降低高达95%。我们讨论了PRAC对未来系统的影响，并预示了未来的研究方向。为了帮助未来的研究，我们在https://github.com/CMU-SAFARI/ramulator2.上开放了我们的实现和脚本



## **18. Chernoff Information as a Privacy Constraint for Adversarial Classification**

作为对抗性分类的隐私约束的删除信息 cs.IT

**SubmitDate**: 2024-07-25    [abs](http://arxiv.org/abs/2403.10307v2) [paper-pdf](http://arxiv.org/pdf/2403.10307v2)

**Authors**: Ayşe Ünsal, Melek Önen

**Abstract**: This work inspects a privacy metric based on Chernoff information, \textit{Chernoff differential privacy}, due to its significance in characterization of the optimal classifier's performance. Adversarial classification, as any other classification problem is built around minimization of the (average or correct detection) probability of error in deciding on either of the classes in the case of binary classification. Unlike the classical hypothesis testing problem, where the false alarm and mis-detection probabilities are handled separately resulting in an asymmetric behavior of the best error exponent, in this work, we focus on the Bayesian setting and characterize the relationship between the best error exponent of the average error probability and $\varepsilon\textrm{-}$differential privacy \cite{D06}. Accordingly, we re-derive Chernoff differential privacy in terms of $\varepsilon\textrm{-}$differential privacy using the Radon-Nikodym derivative and show that it satisfies the composition property for sequential composition. Subsequently, we present numerical evaluation results, which demonstrates that Chernoff information outperforms Kullback-Leibler divergence as a function of the privacy parameter $\varepsilon$, the impact of the adversary's attack and global sensitivity for the problem of adversarial classification in Laplace mechanisms.

摘要: 基于Chernoff信息的隐私度量对抗性分类，因为任何其他分类问题都建立在最小化(平均或正确检测)错误概率的基础上，在二进制分类的情况下，决定其中一个类别的错误概率。与经典假设检验问题不同，在经典假设检验问题中，虚警概率和误检概率被分开处理，导致最佳错误指数的非对称行为。在该工作中，我们关注贝叶斯设置，并刻画了平均错误概率的最佳错误指数与差分隐私{D06}之间的关系。相应地，我们利用Radon-Nikodym导数将Chernoff差分隐私重新推导为$varepsilon差分隐私，并证明它满足序列合成的合成性质。随后，我们给出了数值评估结果，结果表明，Chernoff信息优于Kullback-Leibler发散，它是隐私参数$varepsilon$、对手攻击的影响和全局敏感度的函数。



## **19. Is the Digital Forensics and Incident Response Pipeline Ready for Text-Based Threats in LLM Era?**

数字取证和事件响应管道是否准备好应对LLM时代的基于文本的威胁？ cs.CR

This work has been submitted to the IEEE for possible publication.  Copyright may be transferred without notice, after which this version may no  longer be accessible

**SubmitDate**: 2024-07-25    [abs](http://arxiv.org/abs/2407.17870v1) [paper-pdf](http://arxiv.org/pdf/2407.17870v1)

**Authors**: Avanti Bhandarkar, Ronald Wilson, Anushka Swarup, Mengdi Zhu, Damon Woodard

**Abstract**: In the era of generative AI, the widespread adoption of Neural Text Generators (NTGs) presents new cybersecurity challenges, particularly within the realms of Digital Forensics and Incident Response (DFIR). These challenges primarily involve the detection and attribution of sources behind advanced attacks like spearphishing and disinformation campaigns. As NTGs evolve, the task of distinguishing between human and NTG-authored texts becomes critically complex. This paper rigorously evaluates the DFIR pipeline tailored for text-based security systems, specifically focusing on the challenges of detecting and attributing authorship of NTG-authored texts. By introducing a novel human-NTG co-authorship text attack, termed CS-ACT, our study uncovers significant vulnerabilities in traditional DFIR methodologies, highlighting discrepancies between ideal scenarios and real-world conditions. Utilizing 14 diverse datasets and 43 unique NTGs, up to the latest GPT-4, our research identifies substantial vulnerabilities in the forensic profiling phase, particularly in attributing authorship to NTGs. Our comprehensive evaluation points to factors such as model sophistication and the lack of distinctive style within NTGs as significant contributors for these vulnerabilities. Our findings underscore the necessity for more sophisticated and adaptable strategies, such as incorporating adversarial learning, stylizing NTGs, and implementing hierarchical attribution through the mapping of NTG lineages to enhance source attribution. This sets the stage for future research and the development of more resilient text-based security systems.

摘要: 在生成式人工智能时代，神经文本生成器(NTGs)的广泛采用带来了新的网络安全挑战，特别是在数字取证和事件响应(DFIR)领域。这些挑战主要涉及对鱼叉式网络钓鱼和虚假信息运动等高级攻击背后的来源进行检测和归类。随着NTG的发展，区分人类和NTG创作的文本的任务变得极其复杂。本文严格评估了为基于文本的安全系统量身定做的DFIR管道，特别关注了NTG创作的文本的作者身份检测和归属方面的挑战。通过引入一种名为CS-ACT的新型人-NTG合作文本攻击，我们的研究揭示了传统DFIR方法中的重大漏洞，突出了理想场景和现实世界条件之间的差异。利用14个不同的数据集和43个独特的NTGs，直到最新的GPT-4，我们的研究发现了法医侧写阶段的重大漏洞，特别是在将作者归因于NTGs方面。我们的综合评估指出，模型的复杂性和NTG内部缺乏独特的风格等因素是导致这些漏洞的重要因素。我们的发现强调了更复杂和适应性更强的策略的必要性，例如纳入对抗性学习，风格化的NTGs，以及通过NTG谱系的映射来实现分层归因以增强来源归因。这为未来的研究和开发更具弹性的基于文本的安全系统奠定了基础。



## **20. Domain Generalized Recaptured Screen Image Identification Using SWIN Transformer**

使用SWIN Transformer的域广义重捕获屏幕图像识别 cs.CV

11 pages, 10 figures, 9 tables

**SubmitDate**: 2024-07-25    [abs](http://arxiv.org/abs/2407.17170v2) [paper-pdf](http://arxiv.org/pdf/2407.17170v2)

**Authors**: Preeti Mehta, Aman Sagar, Suchi Kumari

**Abstract**: An increasing number of classification approaches have been developed to address the issue of image rebroadcast and recapturing, a standard attack strategy in insurance frauds, face spoofing, and video piracy. However, most of them neglected scale variations and domain generalization scenarios, performing poorly in instances involving domain shifts, typically made worse by inter-domain and cross-domain scale variances. To overcome these issues, we propose a cascaded data augmentation and SWIN transformer domain generalization framework (DAST-DG) in the current research work Initially, we examine the disparity in dataset representation. A feature generator is trained to make authentic images from various domains indistinguishable. This process is then applied to recaptured images, creating a dual adversarial learning setup. Extensive experiments demonstrate that our approach is practical and surpasses state-of-the-art methods across different databases. Our model achieves an accuracy of approximately 82\% with a precision of 95\% on high-variance datasets.

摘要: 已经开发了越来越多的分类方法来解决图像重播和重新捕获的问题，这是保险欺诈、面部欺骗和视频盗版中的一种标准攻击策略。然而，它们中的大多数忽略了尺度变化和域泛化情景，在涉及域移动的情况下表现不佳，通常由于域间和跨域的尺度差异而变得更糟。为了克服这些问题，我们提出了一个级联数据增强和Swin变换器域泛化框架(DAST-DG)。在当前的研究工作中，我们首先检查了数据集表示上的差异。特征生成器被训练成使来自不同领域的真实图像无法区分。然后，这个过程被应用于重新捕获的图像，创建了一个双重对抗性学习设置。大量的实验表明，我们的方法是实用的，并且在不同的数据库上超过了最先进的方法。我们的模型在高方差数据集上达到了约82的精度和95的精度。



## **21. A Unified Understanding of Adversarial Vulnerability Regarding Unimodal Models and Vision-Language Pre-training Models**

统一理解关于单峰模型和视觉语言预训练模型的对抗脆弱性 cs.CV

14 pages, 9 figures, published in ACMMM2024(oral)

**SubmitDate**: 2024-07-25    [abs](http://arxiv.org/abs/2407.17797v1) [paper-pdf](http://arxiv.org/pdf/2407.17797v1)

**Authors**: Haonan Zheng, Xinyang Deng, Wen Jiang, Wenrui Li

**Abstract**: With Vision-Language Pre-training (VLP) models demonstrating powerful multimodal interaction capabilities, the application scenarios of neural networks are no longer confined to unimodal domains but have expanded to more complex multimodal V+L downstream tasks. The security vulnerabilities of unimodal models have been extensively examined, whereas those of VLP models remain challenging. We note that in CV models, the understanding of images comes from annotated information, while VLP models are designed to learn image representations directly from raw text. Motivated by this discrepancy, we developed the Feature Guidance Attack (FGA), a novel method that uses text representations to direct the perturbation of clean images, resulting in the generation of adversarial images. FGA is orthogonal to many advanced attack strategies in the unimodal domain, facilitating the direct application of rich research findings from the unimodal to the multimodal scenario. By appropriately introducing text attack into FGA, we construct Feature Guidance with Text Attack (FGA-T). Through the interaction of attacking two modalities, FGA-T achieves superior attack effects against VLP models. Moreover, incorporating data augmentation and momentum mechanisms significantly improves the black-box transferability of FGA-T. Our method demonstrates stable and effective attack capabilities across various datasets, downstream tasks, and both black-box and white-box settings, offering a unified baseline for exploring the robustness of VLP models.

摘要: 随着视觉语言预训练模型显示出强大的多通道交互能力，神经网络的应用场景不再局限于单通道领域，而是扩展到更复杂的多通道V+L下游任务。单峰模型的安全漏洞已经被广泛研究，而VLP模型的安全漏洞仍然具有挑战性。我们注意到，在CV模型中，对图像的理解来自于注释信息，而VLP模型被设计为直接从原始文本学习图像表示。基于这种差异，我们提出了特征引导攻击(FGA)，这是一种新的方法，它使用文本表示来引导干净图像的扰动，从而产生对抗性图像。FGA与单模领域的许多高级攻击策略是正交的，便于将丰富的研究成果直接应用于从单模到多模的场景。通过在模糊遗传算法中适当地引入文本攻击，构造了基于文本攻击的特征引导算法(FGA-T)。通过两种攻击模式的交互作用，FGA-T对VLP模型取得了优越的攻击效果。此外，结合数据增强和动量机制显著提高了FGA-T的黑盒可转移性。我们的方法在各种数据集、下游任务以及黑盒和白盒设置上展示了稳定和有效的攻击能力，为探索VLP模型的稳健性提供了统一的基线。



## **22. Exploring Semantic Perturbations on Grover**

探索Grover的语义扰动 cs.LG

**SubmitDate**: 2024-07-25    [abs](http://arxiv.org/abs/2302.00509v2) [paper-pdf](http://arxiv.org/pdf/2302.00509v2)

**Authors**: Ziqing Ji, Pranav Kulkarni, Marko Neskovic, Kevin Nolan, Yan Xu

**Abstract**: With news and information being as easy to access as they currently are, it is more important than ever to ensure that people are not mislead by what they read. Recently, the rise of neural fake news (AI-generated fake news) and its demonstrated effectiveness at fooling humans has prompted the development of models to detect it. One such model is the Grover model, which can both detect neural fake news to prevent it, and generate it to demonstrate how a model could be misused to fool human readers. In this work we explore the Grover model's fake news detection capabilities by performing targeted attacks through perturbations on input news articles. Through this we test Grover's resilience to these adversarial attacks and expose some potential vulnerabilities which should be addressed in further iterations to ensure it can detect all types of fake news accurately.

摘要: 随着新闻和信息像现在一样容易获取，确保人们不被所读内容误导比以往任何时候都更加重要。最近，神经假新闻（人工智能生成的假新闻）的兴起及其在欺骗人类方面所表现出的有效性促使了检测它的模型的开发。其中一个模型是Grover模型，它既可以检测神经假新闻以防止它，又可以生成它来演示模型如何被滥用来欺骗人类读者。在这项工作中，我们通过对输入新闻文章的扰动进行有针对性的攻击来探索Grover模型的假新闻检测能力。通过此，我们测试了Grover对这些对抗攻击的弹性，并暴露了一些潜在的漏洞，这些漏洞应该在进一步的迭代中解决，以确保它能够准确地检测所有类型的假新闻。



## **23. Explaining the Model, Protecting Your Data: Revealing and Mitigating the Data Privacy Risks of Post-Hoc Model Explanations via Membership Inference**

解释模型，保护您的数据：通过会员资格推断揭示和缓解事后模型简化的数据隐私风险 cs.CR

ICML 2024 Workshop on the Next Generation of AI Safety

**SubmitDate**: 2024-07-24    [abs](http://arxiv.org/abs/2407.17663v1) [paper-pdf](http://arxiv.org/pdf/2407.17663v1)

**Authors**: Catherine Huang, Martin Pawelczyk, Himabindu Lakkaraju

**Abstract**: Predictive machine learning models are becoming increasingly deployed in high-stakes contexts involving sensitive personal data; in these contexts, there is a trade-off between model explainability and data privacy. In this work, we push the boundaries of this trade-off: with a focus on foundation models for image classification fine-tuning, we reveal unforeseen privacy risks of post-hoc model explanations and subsequently offer mitigation strategies for such risks. First, we construct VAR-LRT and L1/L2-LRT, two new membership inference attacks based on feature attribution explanations that are significantly more successful than existing explanation-leveraging attacks, particularly in the low false-positive rate regime that allows an adversary to identify specific training set members with confidence. Second, we find empirically that optimized differentially private fine-tuning substantially diminishes the success of the aforementioned attacks, while maintaining high model accuracy. We carry out a systematic empirical investigation of our 2 new attacks with 5 vision transformer architectures, 5 benchmark datasets, 4 state-of-the-art post-hoc explanation methods, and 4 privacy strength settings.

摘要: 预测性机器学习模型越来越多地部署在涉及敏感个人数据的高风险环境中；在这些环境中，模型的可解释性和数据隐私之间存在权衡。在这项工作中，我们突破了这种权衡的界限：将重点放在图像分类微调的基础模型上，揭示后自组织模型解释的不可预见的隐私风险，并随后提供此类风险的缓解策略。首先，我们构造了VAR-LRT和L1/L2-LRT，这是两种新的基于特征属性解释的成员推理攻击，它们比现有的解释杠杆攻击要成功得多，特别是在允许对手自信地识别特定训练集成员的低假阳性率机制下。其次，我们从经验上发现，在保持较高模型精度的同时，优化的差分私有微调显著降低了上述攻击的成功率。我们使用5个视觉转换器架构、5个基准数据集、4个最先进的后自组织解释方法和4个隐私强度设置对我们的2个新攻击进行了系统的经验研究。



## **24. Revising the Problem of Partial Labels from the Perspective of CNNs' Robustness**

从CNN的稳健性角度修正部分标签问题 cs.CV

**SubmitDate**: 2024-07-24    [abs](http://arxiv.org/abs/2407.17630v1) [paper-pdf](http://arxiv.org/pdf/2407.17630v1)

**Authors**: Xin Zhang, Yuqi Song, Wyatt McCurdy, Xiaofeng Wang, Fei Zuo

**Abstract**: Convolutional neural networks (CNNs) have gained increasing popularity and versatility in recent decades, finding applications in diverse domains. These remarkable achievements are greatly attributed to the support of extensive datasets with precise labels. However, annotating image datasets is intricate and complex, particularly in the case of multi-label datasets. Hence, the concept of partial-label setting has been proposed to reduce annotation costs, and numerous corresponding solutions have been introduced. The evaluation methods for these existing solutions have been primarily based on accuracy. That is, their performance is assessed by their predictive accuracy on the test set. However, we insist that such an evaluation is insufficient and one-sided. On one hand, since the quality of the test set has not been evaluated, the assessment results are unreliable. On the other hand, the partial-label problem may also be raised by undergoing adversarial attacks. Therefore, incorporating robustness into the evaluation system is crucial. For this purpose, we first propose two attack models to generate multiple partial-label datasets with varying degrees of label missing rates. Subsequently, we introduce a lightweight partial-label solution using pseudo-labeling techniques and a designed loss function. Then, we employ D-Score to analyze both the proposed and existing methods to determine whether they can enhance robustness while improving accuracy. Extensive experimental results demonstrate that while certain methods may improve accuracy, the enhancement in robustness is not significant, and in some cases, it even diminishes.

摘要: 卷积神经网络(CNN)在近几十年来得到了越来越广泛的应用，在不同的领域得到了广泛的应用。这些显著的成就在很大程度上归功于具有精确标签的大量数据集的支持。然而，标注图像数据集是复杂和复杂的，特别是在多标签数据集的情况下。因此，为了降低标注代价，人们提出了部分标注设置的概念，并提出了许多相应的解决方案。对这些现有解决方案的评估方法主要是基于准确性。也就是说，他们的表现是通过他们在测试集上的预测准确性来评估的。然而，我们坚持认为，这样的评估是不充分和片面的。一方面，由于测试集的质量没有得到评估，评估结果不可靠。另一方面，部分标签问题也可能通过经历对抗性攻击而引起。因此，将稳健性纳入评估体系至关重要。为此，我们首先提出了两种攻击模型来生成具有不同程度标签缺失率的多个部分标签数据集。随后，我们介绍了一种使用伪标记技术和设计的损失函数的轻量级部分标记解决方案。然后，我们使用D-SCORE对提出的方法和现有的方法进行分析，以确定它们是否可以在提高准确率的同时增强稳健性。大量的实验结果表明，虽然某些方法可以提高准确率，但在稳健性方面的增强并不显著，在某些情况下，甚至会减弱。



## **25. Fluent Student-Teacher Redteaming**

流利的师生红团队 cs.CL

**SubmitDate**: 2024-07-24    [abs](http://arxiv.org/abs/2407.17447v1) [paper-pdf](http://arxiv.org/pdf/2407.17447v1)

**Authors**: T. Ben Thompson, Michael Sklar

**Abstract**: Many publicly available language models have been safety tuned to reduce the likelihood of toxic or liability-inducing text. Users or security analysts attempt to jailbreak or redteam these models with adversarial prompts which cause compliance with requests. One attack method is to apply discrete optimization techniques to the prompt. However, the resulting attack strings are often gibberish text, easily filtered by defenders due to high measured perplexity, and may fail for unseen tasks and/or well-tuned models. In this work, we improve existing algorithms (primarily GCG and BEAST) to develop powerful and fluent attacks on safety-tuned models like Llama-2 and Phi-3. Our technique centers around a new distillation-based approach that encourages the victim model to emulate a toxified finetune, either in terms of output probabilities or internal activations. To encourage human-fluent attacks, we add a multi-model perplexity penalty and a repetition penalty to the objective. We also enhance optimizer strength by allowing token insertions, token swaps, and token deletions and by using longer attack sequences. The resulting process is able to reliably jailbreak the most difficult target models with prompts that appear similar to human-written prompts. On Advbench we achieve attack success rates $>93$% for Llama-2-7B, Llama-3-8B, and Vicuna-7B, while maintaining model-measured perplexity $<33$; we achieve $95$% attack success for Phi-3, though with higher perplexity. We also find a universally-optimized single fluent prompt that induces $>88$% compliance on previously unseen tasks across Llama-2-7B, Phi-3-mini and Vicuna-7B and transfers to other black-box models.

摘要: 许多公开提供的语言模型都经过了安全调整，以减少有毒或导致责任的文本的可能性。用户或安全分析师试图用敌意提示对这些模型进行越狱或编辑，从而导致遵守请求。一种攻击方法是对提示应用离散优化技术。然而，由此产生的攻击字符串通常是胡言乱语的文本，由于高度测量的困惑，很容易被防御者过滤，并且可能无法完成看不见的任务和/或良好调整的模型。在这项工作中，我们改进了现有的算法(主要是GCG和BEAST)，以开发针对Llama-2和Phi-3等安全调整模型的强大而流畅的攻击。我们的技术以一种新的基于蒸馏的方法为中心，该方法鼓励受害者模型在输出概率或内部激活方面模仿中毒的微调。为了鼓励人类流利的攻击，我们在目标上增加了多模式困惑惩罚和重复惩罚。我们还通过允许令牌插入、令牌交换和令牌删除以及使用更长的攻击序列来增强优化器的强度。由此产生的过程能够可靠地用看起来类似于人写的提示的提示越狱最困难的目标模型。在Advbench上，我们实现了骆驼-2-7B、骆驼-3-8B和维库纳-7B的攻击成功率$>93$%，同时保持了模型测量的困惑$<33$；我们为Phi-3实现了$95$%的攻击成功率，尽管困惑程度更高。我们还发现了一个普遍优化的单一流畅提示，在Llama-2-7B、Phi-3-mini和Vicuna-7B上导致以前未见过的任务的合规性>88$%，并转移到其他黑盒型号。



## **26. Physical Adversarial Attack on Monocular Depth Estimation via Shape-Varying Patches**

通过形状变化贴片对单眼深度估计的物理对抗攻击 cs.CV

**SubmitDate**: 2024-07-24    [abs](http://arxiv.org/abs/2407.17312v1) [paper-pdf](http://arxiv.org/pdf/2407.17312v1)

**Authors**: Chenxing Zhao, Yang Li, Shihao Wu, Wenyi Tan, Shuangju Zhou, Quan Pan

**Abstract**: Adversarial attacks against monocular depth estimation (MDE) systems pose significant challenges, particularly in safety-critical applications such as autonomous driving. Existing patch-based adversarial attacks for MDE are confined to the vicinity of the patch, making it difficult to affect the entire target. To address this limitation, we propose a physics-based adversarial attack on monocular depth estimation, employing a framework called Attack with Shape-Varying Patches (ASP), aiming to optimize patch content, shape, and position to maximize effectiveness. We introduce various mask shapes, including quadrilateral, rectangular, and circular masks, to enhance the flexibility and efficiency of the attack. Furthermore, we propose a new loss function to extend the influence of the patch beyond the overlapping regions. Experimental results demonstrate that our attack method generates an average depth error of 18 meters on the target car with a patch area of 1/9, affecting over 98\% of the target area.

摘要: 针对单目深度估计(MDE)系统的对抗性攻击带来了巨大的挑战，特别是在自动驾驶等安全关键应用中。现有的针对MDE的基于补丁的对抗性攻击仅限于补丁附近，难以影响整个目标。针对这一局限性，我们提出了一种基于物理的对抗性单眼深度估计攻击方法，采用了一种称为形状变化补丁攻击(ASP)的框架，旨在优化补丁的内容、形状和位置以最大化效果。我们引入了各种掩码形状，包括四边形、矩形和圆形掩码，以增强攻击的灵活性和效率。此外，我们还提出了一种新的损失函数，将斑块的影响扩展到重叠区域之外。实验结果表明，我们的攻击方法对目标车的平均深度误差为18m，补丁面积为1/9，影响了98%以上的目标区域。



## **27. Learning to Transform Dynamically for Better Adversarial Transferability**

学习动态转型以获得更好的对抗可移植性 cs.CV

accepted as a poster in CVPR 2024

**SubmitDate**: 2024-07-24    [abs](http://arxiv.org/abs/2405.14077v2) [paper-pdf](http://arxiv.org/pdf/2405.14077v2)

**Authors**: Rongyi Zhu, Zeliang Zhang, Susan Liang, Zhuo Liu, Chenliang Xu

**Abstract**: Adversarial examples, crafted by adding perturbations imperceptible to humans, can deceive neural networks. Recent studies identify the adversarial transferability across various models, \textit{i.e.}, the cross-model attack ability of adversarial samples. To enhance such adversarial transferability, existing input transformation-based methods diversify input data with transformation augmentation. However, their effectiveness is limited by the finite number of available transformations. In our study, we introduce a novel approach named Learning to Transform (L2T). L2T increases the diversity of transformed images by selecting the optimal combination of operations from a pool of candidates, consequently improving adversarial transferability. We conceptualize the selection of optimal transformation combinations as a trajectory optimization problem and employ a reinforcement learning strategy to effectively solve the problem. Comprehensive experiments on the ImageNet dataset, as well as practical tests with Google Vision and GPT-4V, reveal that L2T surpasses current methodologies in enhancing adversarial transferability, thereby confirming its effectiveness and practical significance. The code is available at https://github.com/RongyiZhu/L2T.

摘要: 通过添加人类察觉不到的扰动而精心制作的对抗性例子可以欺骗神经网络。最近的研究发现了各种模型之间的对抗性转移，即对抗性样本的跨模型攻击能力。为了增强这种对抗性的可转移性，现有的基于输入变换的方法通过变换增强来使输入数据多样化。然而，它们的有效性受到可用变换数量有限的限制。在我们的研究中，我们引入了一种名为学习转化(L2T)的新方法。L2T通过从候选集合中选择最优的操作组合来增加变换图像的多样性，从而提高了对抗性转移。我们将最优变换组合的选择概念化为一个轨迹优化问题，并采用强化学习策略来有效地解决该问题。在ImageNet数据集上的综合实验以及与Google Vision和GPT-4V的实际测试表明，L2T在增强对抗性可转移性方面优于现有方法，从而证实了其有效性和现实意义。代码可在https://github.com/RongyiZhu/L2T.上获得



## **28. When AI Defeats Password Deception! A Deep Learning Framework to Distinguish Passwords and Honeywords**

当人工智能击败密码欺骗！区分密码和蜜语的深度学习框架 cs.CR

**SubmitDate**: 2024-07-24    [abs](http://arxiv.org/abs/2407.16964v1) [paper-pdf](http://arxiv.org/pdf/2407.16964v1)

**Authors**: Jimmy Dani, Brandon McCulloh, Nitesh Saxena

**Abstract**: "Honeywords" have emerged as a promising defense mechanism for detecting data breaches and foiling offline dictionary attacks (ODA) by deceiving attackers with false passwords. In this paper, we propose PassFilter, a novel deep learning (DL) based attack framework, fundamental in its ability to identify passwords from a set of sweetwords associated with a user account, effectively challenging a variety of honeywords generation techniques (HGTs). The DL model in PassFilter is trained with a set of previously collected or adversarially generated passwords and honeywords, and carefully orchestrated to predict whether a sweetword is the password or a honeyword. Our model can compromise the security of state-of-the-art, heuristics-based, and representation learning-based HGTs proposed by Dionysiou et al. Specifically, our analysis with nine publicly available password datasets shows that PassFilter significantly outperforms the baseline random guessing success rate of 5%, achieving 6.10% to 52.78% on the 1st guessing attempt, considering 20 sweetwords per account. This success rate rapidly increases with additional login attempts before account lock-outs, often allowed on many real-world online services to maintain reasonable usability. For example, it ranges from 41.78% to 96.80% for five attempts, and from 72.87% to 99.00% for ten attempts, compared to 25% and 50% random guessing, respectively. We also examined PassFilter against general-purpose language models used for honeyword generation, like those proposed by Yu et al. These honeywords also proved vulnerable to our attack, with success rates of 14.19% for 1st guessing attempt, increasing to 30.23%, 41.70%, and 63.10% after 3rd, 5th, and 10th guessing attempts, respectively. Our findings demonstrate the effectiveness of DL model deployed in PassFilter in breaching state-of-the-art HGTs and compromising password security based on ODA.

摘要: “蜜字”已经成为一种很有前途的防御机制，可以通过用虚假密码欺骗攻击者来检测数据泄露和挫败离线词典攻击(Oda)。在本文中，我们提出了一种新的基于深度学习的攻击框架PassFilter，其基本特征是能够从与用户帐户关联的一组甜言蜜语中识别密码，有效地挑战了各种蜜语生成技术(HGT)。PassFilter中的DL模型使用一组先前收集的或恶意生成的密码和蜜字进行训练，并精心编排以预测甜言蜜语是密码还是蜜语。我们的模型可能会危及Dionsiou等人提出的最新的、基于启发式的和基于表示学习的HGT的安全性。具体地说，我们对9个公开可用的密码数据集的分析表明，PassFilter的性能显著高于5%的基线随机猜测成功率，考虑到每个帐户20个甜言蜜语，第一次猜测的成功率为6.10%到52.78%。这一成功率随着帐户锁定之前的额外登录尝试而迅速增加，这在许多现实世界的在线服务中通常是允许的，以保持合理的可用性。例如，5次尝试的命中率从41.78%到96.80%，10次尝试的命中率从72.87%到99.00%，而随机猜测的命中率分别为25%和50%。我们还将PassFilter与用于蜜词生成的通用语言模型进行了对比，如Yu等人提出的模型。这些蜜语也容易受到我们的攻击，第一次猜测的成功率为14.19%，第三次、第五次和第十次的猜测成功率分别增加到30.23%、41.70%和63.10%。我们的发现证明了在PassFilter中部署的DL模型在破解最新的HGTS和基于oda的口令安全性方面的有效性。



## **29. RigorLLM: Resilient Guardrails for Large Language Models against Undesired Content**

RigorLLM：针对不需要内容的大型语言模型的弹性护栏 cs.CR

**SubmitDate**: 2024-07-23    [abs](http://arxiv.org/abs/2403.13031v2) [paper-pdf](http://arxiv.org/pdf/2403.13031v2)

**Authors**: Zhuowen Yuan, Zidi Xiong, Yi Zeng, Ning Yu, Ruoxi Jia, Dawn Song, Bo Li

**Abstract**: Recent advancements in Large Language Models (LLMs) have showcased remarkable capabilities across various tasks in different domains. However, the emergence of biases and the potential for generating harmful content in LLMs, particularly under malicious inputs, pose significant challenges. Current mitigation strategies, while effective, are not resilient under adversarial attacks. This paper introduces Resilient Guardrails for Large Language Models (RigorLLM), a novel framework designed to efficiently and effectively moderate harmful and unsafe inputs and outputs for LLMs. By employing a multi-faceted approach that includes energy-based training data augmentation through Langevin dynamics, optimizing a safe suffix for inputs via minimax optimization, and integrating a fusion-based model combining robust KNN with LLMs based on our data augmentation, RigorLLM offers a robust solution to harmful content moderation. Our experimental evaluations demonstrate that RigorLLM not only outperforms existing baselines like OpenAI API and Perspective API in detecting harmful content but also exhibits unparalleled resilience to jailbreaking attacks. The innovative use of constrained optimization and a fusion-based guardrail approach represents a significant step forward in developing more secure and reliable LLMs, setting a new standard for content moderation frameworks in the face of evolving digital threats.

摘要: 大型语言模型(LLM)的最新进展展示了跨越不同领域的各种任务的显著能力。然而，偏见的出现和在低成本管理中产生有害内容的可能性，特别是在恶意投入下，构成了重大挑战。目前的缓解战略虽然有效，但在对抗性攻击下缺乏弹性。本文介绍了用于大型语言模型的弹性护栏(RigorLLM)，这是一个新的框架，旨在高效和有效地控制LLM中有害和不安全的输入和输出。通过采用多方面的方法，包括通过朗之万动力学基于能量的训练数据增强，通过极小极大优化优化输入的安全后缀，以及基于我们的数据增强将稳健的KNN与LLMS相结合的基于融合的模型，RigorLLM为有害内容适度提供了稳健的解决方案。我们的实验评估表明，RigorLLM不仅在检测有害内容方面优于OpenAI API和透视API等现有基线，而且对越狱攻击表现出无与伦比的弹性。约束优化和基于融合的护栏方法的创新使用代表着在开发更安全可靠的LLMS方面向前迈出的重要一步，为面对不断变化的数字威胁的内容审查框架设定了新的标准。



## **30. RedAgent: Red Teaming Large Language Models with Context-aware Autonomous Language Agent**

RedAgent：Red将大型语言模型与上下文感知自治语言代理结合起来 cs.CR

**SubmitDate**: 2024-07-23    [abs](http://arxiv.org/abs/2407.16667v1) [paper-pdf](http://arxiv.org/pdf/2407.16667v1)

**Authors**: Huiyu Xu, Wenhui Zhang, Zhibo Wang, Feng Xiao, Rui Zheng, Yunhe Feng, Zhongjie Ba, Kui Ren

**Abstract**: Recently, advanced Large Language Models (LLMs) such as GPT-4 have been integrated into many real-world applications like Code Copilot. These applications have significantly expanded the attack surface of LLMs, exposing them to a variety of threats. Among them, jailbreak attacks that induce toxic responses through jailbreak prompts have raised critical safety concerns. To identify these threats, a growing number of red teaming approaches simulate potential adversarial scenarios by crafting jailbreak prompts to test the target LLM. However, existing red teaming methods do not consider the unique vulnerabilities of LLM in different scenarios, making it difficult to adjust the jailbreak prompts to find context-specific vulnerabilities. Meanwhile, these methods are limited to refining jailbreak templates using a few mutation operations, lacking the automation and scalability to adapt to different scenarios. To enable context-aware and efficient red teaming, we abstract and model existing attacks into a coherent concept called "jailbreak strategy" and propose a multi-agent LLM system named RedAgent that leverages these strategies to generate context-aware jailbreak prompts. By self-reflecting on contextual feedback in an additional memory buffer, RedAgent continuously learns how to leverage these strategies to achieve effective jailbreaks in specific contexts. Extensive experiments demonstrate that our system can jailbreak most black-box LLMs in just five queries, improving the efficiency of existing red teaming methods by two times. Additionally, RedAgent can jailbreak customized LLM applications more efficiently. By generating context-aware jailbreak prompts towards applications on GPTs, we discover 60 severe vulnerabilities of these real-world applications with only two queries per vulnerability. We have reported all found issues and communicated with OpenAI and Meta for bug fixes.

摘要: 最近，GPT-4等高级大型语言模型(LLM)已集成到许多实际应用程序中，如Code Copilot。这些应用程序显著扩大了LLMS的攻击面，使它们暴露在各种威胁之下。其中，通过越狱提示引发有毒反应的越狱攻击引发了严重的安全问题。为了识别这些威胁，越来越多的红色团队方法通过精心编制越狱提示来测试目标LLM，以模拟潜在的敌对场景。然而，现有的红色团队方法没有考虑LLM在不同场景下的独特漏洞，很难调整越狱提示来发现上下文特定的漏洞。同时，这些方法仅限于使用少量的变异操作来提炼越狱模板，缺乏适应不同场景的自动化和可扩展性。为了实现上下文感知和高效的红色团队，我们将现有的攻击抽象并建模为一个连贯的概念，称为越狱策略，并提出了一个名为RedAgent的多代理LLM系统，该系统利用这些策略来生成上下文感知越狱提示。通过对额外内存缓冲区中的上下文反馈进行自我反思，RedAgent不断学习如何利用这些策略在特定上下文中实现有效的越狱。大量的实验表明，我们的系统可以在短短五次查询中破解大部分黑盒LLM，将现有的红色团队方法的效率提高了两倍。此外，RedAgent可以更高效地越狱定制的LLM应用程序。通过向GPT上的应用程序生成上下文感知越狱提示，我们发现了这些现实世界应用程序的60个严重漏洞，每个漏洞只有两个查询。我们已经报告了所有发现的问题，并与OpenAI和Meta进行了沟通以修复错误。



## **31. S-E Pipeline: A Vision Transformer (ViT) based Resilient Classification Pipeline for Medical Imaging Against Adversarial Attacks**

S-E Pipeline：基于视觉Transformer（ViT）的弹性分类管道，用于针对对抗性攻击的医学成像 cs.CV

**SubmitDate**: 2024-07-23    [abs](http://arxiv.org/abs/2407.17587v1) [paper-pdf](http://arxiv.org/pdf/2407.17587v1)

**Authors**: Neha A S, Vivek Chaturvedi, Muhammad Shafique

**Abstract**: Vision Transformer (ViT) is becoming widely popular in automating accurate disease diagnosis in medical imaging owing to its robust self-attention mechanism. However, ViTs remain vulnerable to adversarial attacks that may thwart the diagnosis process by leading it to intentional misclassification of critical disease. In this paper, we propose a novel image classification pipeline, namely, S-E Pipeline, that performs multiple pre-processing steps that allow ViT to be trained on critical features so as to reduce the impact of input perturbations by adversaries. Our method uses a combination of segmentation and image enhancement techniques such as Contrast Limited Adaptive Histogram Equalization (CLAHE), Unsharp Masking (UM), and High-Frequency Emphasis filtering (HFE) as preprocessing steps to identify critical features that remain intact even after adversarial perturbations. The experimental study demonstrates that our novel pipeline helps in reducing the effect of adversarial attacks by 72.22% for the ViT-b32 model and 86.58% for the ViT-l32 model. Furthermore, we have shown an end-to-end deployment of our proposed method on the NVIDIA Jetson Orin Nano board to demonstrate its practical use case in modern hand-held devices that are usually resource-constrained.

摘要: 视觉转换器(VIT)由于其强大的自我注意机制，在医学成像中自动准确地诊断疾病方面正变得越来越受欢迎。然而，VITS仍然容易受到敌意攻击，这些攻击可能会导致对危重疾病的故意错误分类，从而阻碍诊断过程。本文提出了一种新颖的图像分类流水线，即S-E流水线，该流水线经过多个预处理步骤，允许对VIT进行关键特征的训练，以减少对手输入扰动的影响。我们的方法使用分割和图像增强技术的组合，例如对比度受限自适应直方图均衡(CLAHE)、反锐化掩模(UM)和高频加重滤波(HFE)作为预处理步骤来识别即使在对抗性扰动之后仍然保持完好的关键特征。实验研究表明，新的流水线将VIT-B32模型的对抗性攻击效果降低了72.22%，VIT-132模型的对抗性攻击效果降低了86.58%。此外，我们还展示了我们建议的方法在NVIDIA Jetson Orin纳米板上的端到端部署，以演示其在通常资源受限的现代手持设备中的实际用例。



## **32. Defending Our Privacy With Backdoors**

用后门保护我们的隐私 cs.LG

Accepted at ECAI 2024

**SubmitDate**: 2024-07-23    [abs](http://arxiv.org/abs/2310.08320v4) [paper-pdf](http://arxiv.org/pdf/2310.08320v4)

**Authors**: Dominik Hintersdorf, Lukas Struppek, Daniel Neider, Kristian Kersting

**Abstract**: The proliferation of large AI models trained on uncurated, often sensitive web-scraped data has raised significant privacy concerns. One of the concerns is that adversaries can extract information about the training data using privacy attacks. Unfortunately, the task of removing specific information from the models without sacrificing performance is not straightforward and has proven to be challenging. We propose a rather easy yet effective defense based on backdoor attacks to remove private information, such as names and faces of individuals, from vision-language models by fine-tuning them for only a few minutes instead of re-training them from scratch. Specifically, by strategically inserting backdoors into text encoders, we align the embeddings of sensitive phrases with those of neutral terms-"a person" instead of the person's actual name. For image encoders, we map individuals' embeddings to be removed from the model to a universal, anonymous embedding. The results of our extensive experimental evaluation demonstrate the effectiveness of our backdoor-based defense on CLIP by assessing its performance using a specialized privacy attack for zero-shot classifiers. Our approach provides a new "dual-use" perspective on backdoor attacks and presents a promising avenue to enhance the privacy of individuals within models trained on uncurated web-scraped data.

摘要: 大型人工智能模型的激增引发了人们对隐私的严重担忧。这些模型针对未经管理的、往往是敏感的网络数据进行培训。其中一个令人担忧的问题是，攻击者可以使用隐私攻击来提取有关训练数据的信息。不幸的是，在不牺牲性能的情况下从模型中删除特定信息的任务并不简单，而且已被证明是具有挑战性的。我们提出了一种基于后门攻击的相当简单但有效的防御方法，通过仅对视觉语言模型进行几分钟的微调来删除私人信息，如个人的姓名和面孔，而不是从头开始重新训练它们。具体地说，通过有策略地在文本编码器中插入后门，我们将敏感短语的嵌入与中性术语的嵌入--“人”而不是人的实际姓名--对齐。对于图像编码器，我们将从模型中移除的个人嵌入映射到通用的匿名嵌入。我们广泛的实验评估结果证明了我们的基于后门的CLIP防御的有效性，通过使用专门的针对零射击分类器的隐私攻击来评估其性能。我们的方法为后门攻击提供了一种新的“两用”视角，并提供了一种在未经管理的网络抓取数据的培训模型中增强个人隐私的有前景的途径。



## **33. Securing Tomorrow's Smart Cities: Investigating Software Security in Internet of Vehicles and Deep Learning Technologies**

确保未来的智慧城市：调查车联网和深度学习技术中的软件安全 cs.CR

**SubmitDate**: 2024-07-23    [abs](http://arxiv.org/abs/2407.16410v1) [paper-pdf](http://arxiv.org/pdf/2407.16410v1)

**Authors**: Ridhi Jain, Norbert Tihanyi, Mohamed Amine Ferrag

**Abstract**: Integrating Deep Learning (DL) techniques in the Internet of Vehicles (IoV) introduces many security challenges and issues that require thorough examination. This literature review delves into the inherent vulnerabilities and risks associated with DL in IoV systems, shedding light on the multifaceted nature of security threats. Through an extensive analysis of existing research, we explore potential threats posed by DL algorithms, including adversarial attacks, data privacy breaches, and model poisoning. Additionally, we investigate the impact of DL on critical aspects of IoV security, such as intrusion detection, anomaly detection, and secure communication protocols. Our review emphasizes the complexities of ensuring the robustness, reliability, and trustworthiness of DL-based IoV systems, given the dynamic and interconnected nature of vehicular networks. Furthermore, we discuss the need for novel security solutions tailored to address these challenges effectively and enhance the security posture of DL-enabled IoV environments. By offering insights into these critical issues, this chapter aims to stimulate further research, innovation, and collaboration in securing DL techniques within the context of the IoV, thereby fostering a safer and more resilient future for vehicular communication and connectivity.

摘要: 在车联网(IoV)中集成深度学习(DL)技术带来了许多安全挑战和问题，需要进行彻底的检查。这篇文献综述深入探讨了IoV系统中与DL相关的固有漏洞和风险，揭示了安全威胁的多方面性质。通过对现有研究的广泛分析，我们探索了DL算法带来的潜在威胁，包括对抗性攻击、数据隐私泄露和模型中毒。此外，我们还研究了DL对IoV安全的关键方面的影响，例如入侵检测、异常检测和安全通信协议。考虑到车载网络的动态和互联特性，我们的审查强调了确保基于DL的IoV系统的健壮性、可靠性和可信性的复杂性。此外，我们还讨论了为有效应对这些挑战并增强支持DL的IoV环境的安全态势而量身定做的新型安全解决方案的必要性。通过提供对这些关键问题的见解，本章旨在鼓励在万物互联背景下保护DL技术方面的进一步研究、创新和合作，从而为车辆通信和连接培养一个更安全、更具弹性的未来。



## **34. Protecting Quantum Procrastinators with Signature Lifting: A Case Study in Cryptocurrencies**

通过签名提升保护量子拖延者：加密货币的案例研究 cs.CR

Minor revision

**SubmitDate**: 2024-07-23    [abs](http://arxiv.org/abs/2303.06754v2) [paper-pdf](http://arxiv.org/pdf/2303.06754v2)

**Authors**: Or Sattath, Shai Wyborski

**Abstract**: Current solutions to quantum vulnerabilities of widely used cryptographic schemes involve migrating users to post-quantum schemes before quantum attacks become feasible. This work deals with protecting quantum procrastinators: users that failed to migrate to post-quantum cryptography in time.   To address this problem in the context of digital signatures, we introduce a technique called signature lifting, that allows us to lift a deployed pre-quantum signature scheme satisfying a certain property to a post-quantum signature scheme that uses the same keys. Informally, the said property is that a post-quantum one-way function is used "somewhere along the way" to derive the public-key from the secret-key. Our constructions of signature lifting relies heavily on the post-quantum digital signature scheme Picnic (Chase et al., CCS'17).   Our main case-study is cryptocurrencies, where this property holds in two scenarios: when the public-key is generated via a key-derivation function or when the public-key hash is posted instead of the public-key itself. We propose a modification, based on signature lifting, that can be applied in many cryptocurrencies for securely spending pre-quantum coins in presence of quantum adversaries. Our construction improves upon existing constructions in two major ways: it is not limited to pre-quantum coins whose ECDSA public-key has been kept secret (and in particular, it handles all coins that are stored in addresses generated by HD wallets), and it does not require access to post-quantum coins or using side payments to pay for posting the transaction.

摘要: 目前针对广泛使用的密码方案的量子漏洞的解决方案包括在量子攻击变得可行之前将用户迁移到后量子方案。这项工作涉及保护量子拖延者：未能及时迁移到后量子密码学的用户。为了在数字签名的背景下解决这个问题，我们引入了一种称为签名提升的技术，该技术允许我们将满足一定性质的部署的前量子签名方案提升到使用相同密钥的后量子签名方案。非正式地，所述性质是使用后量子单向函数来从秘密密钥导出公钥。我们的签名提升的构造在很大程度上依赖于后量子数字签名方案Picnic(Chase等人，CCS‘17)。我们的主要案例研究是加密货币，其中该属性在两种情况下成立：当公钥是通过密钥派生函数生成时，或者当公钥散列被发布而不是公钥本身时。我们提出了一种基于签名提升的改进方案，该方案可以应用于多种加密货币，以便在存在量子对手的情况下安全地消费前量子币。我们的结构在两个主要方面对现有结构进行了改进：它不仅限于其ECDSA公钥被保密的前量子硬币(尤其是，它处理存储在HD钱包生成的地址中的所有硬币)，并且它不需要访问后量子硬币或使用附带支付来支付发布交易的费用。



## **35. Efficient Generation of Targeted and Transferable Adversarial Examples for Vision-Language Models Via Diffusion Models**

通过扩散模型高效生成视觉语言模型的有针对性且可转移的对抗示例 cs.CV

**SubmitDate**: 2024-07-23    [abs](http://arxiv.org/abs/2404.10335v3) [paper-pdf](http://arxiv.org/pdf/2404.10335v3)

**Authors**: Qi Guo, Shanmin Pang, Xiaojun Jia, Yang Liu, Qing Guo

**Abstract**: Adversarial attacks, particularly \textbf{targeted} transfer-based attacks, can be used to assess the adversarial robustness of large visual-language models (VLMs), allowing for a more thorough examination of potential security flaws before deployment. However, previous transfer-based adversarial attacks incur high costs due to high iteration counts and complex method structure. Furthermore, due to the unnaturalness of adversarial semantics, the generated adversarial examples have low transferability. These issues limit the utility of existing methods for assessing robustness. To address these issues, we propose AdvDiffVLM, which uses diffusion models to generate natural, unrestricted and targeted adversarial examples via score matching. Specifically, AdvDiffVLM uses Adaptive Ensemble Gradient Estimation to modify the score during the diffusion model's reverse generation process, ensuring that the produced adversarial examples have natural adversarial targeted semantics, which improves their transferability. Simultaneously, to improve the quality of adversarial examples, we use the GradCAM-guided Mask method to disperse adversarial semantics throughout the image rather than concentrating them in a single area. Finally, AdvDiffVLM embeds more target semantics into adversarial examples after multiple iterations. Experimental results show that our method generates adversarial examples 5x to 10x faster than state-of-the-art transfer-based adversarial attacks while maintaining higher quality adversarial examples. Furthermore, compared to previous transfer-based adversarial attacks, the adversarial examples generated by our method have better transferability. Notably, AdvDiffVLM can successfully attack a variety of commercial VLMs in a black-box environment, including GPT-4V.

摘要: 对抗性攻击，特别是基于传输的对抗性攻击，可用于评估大型视觉语言模型(VLM)的对抗性健壮性，从而允许在部署之前更彻底地检查潜在的安全漏洞。然而，以往基于转移的对抗性攻击由于迭代次数多、方法结构复杂，代价较高。此外，由于对抗性语义的非自然性，生成的对抗性实例可转移性较低。这些问题限制了现有稳健性评估方法的实用性。为了解决这些问题，我们提出了AdvDiffVLM，它使用扩散模型通过得分匹配来生成自然的、不受限制的和有针对性的对抗性实例。具体地说，AdvDiffVLM在扩散模型的反向生成过程中使用自适应集成梯度估计来修改分数，确保生成的对抗性实例具有自然对抗性目标语义，从而提高了它们的可转移性。同时，为了提高对抗性实例的质量，我们使用了GradCAM引导的掩码方法，将对抗性语义分散在整个图像中，而不是将它们集中在单个区域。最后，在多次迭代后，AdvDiffVLM将更多的目标语义嵌入到对抗性实例中。实验结果表明，在保持较高质量的对抗性实例的同时，我们的方法生成对抗性实例的速度比最新的基于传输的对抗性攻击快5倍到10倍。此外，与以往基于转移的对抗性攻击相比，该方法生成的对抗性实例具有更好的可转移性。值得注意的是，AdvDiffVLM可以在黑盒环境中成功攻击各种商业VLM，包括GPT-4V。



## **36. R.A.C.E.: Robust Adversarial Concept Erasure for Secure Text-to-Image Diffusion Model**

皇家海关：安全文本到图像扩散模型的鲁棒对抗概念擦除 cs.CV

Accepted at ECCV 2024

**SubmitDate**: 2024-07-23    [abs](http://arxiv.org/abs/2405.16341v2) [paper-pdf](http://arxiv.org/pdf/2405.16341v2)

**Authors**: Changhoon Kim, Kyle Min, Yezhou Yang

**Abstract**: In the evolving landscape of text-to-image (T2I) diffusion models, the remarkable capability to generate high-quality images from textual descriptions faces challenges with the potential misuse of reproducing sensitive content. To address this critical issue, we introduce \textbf{R}obust \textbf{A}dversarial \textbf{C}oncept \textbf{E}rase (RACE), a novel approach designed to mitigate these risks by enhancing the robustness of concept erasure method for T2I models. RACE utilizes a sophisticated adversarial training framework to identify and mitigate adversarial text embeddings, significantly reducing the Attack Success Rate (ASR). Impressively, RACE achieves a 30 percentage point reduction in ASR for the ``nudity'' concept against the leading white-box attack method. Our extensive evaluations demonstrate RACE's effectiveness in defending against both white-box and black-box attacks, marking a significant advancement in protecting T2I diffusion models from generating inappropriate or misleading imagery. This work underlines the essential need for proactive defense measures in adapting to the rapidly advancing field of adversarial challenges. Our code is publicly available: \url{https://github.com/chkimmmmm/R.A.C.E.}

摘要: 在不断发展的文本到图像(T2I)扩散模型中，从文本描述生成高质量图像的非凡能力面临着潜在的误用，即复制敏感内容。为了解决这一关键问题，我们引入了一种新的方法，即RACE(RACE)，旨在通过增强T2I模型的概念删除方法的健壮性来降低这些风险。RACE利用复杂的对抗性训练框架来识别和缓解对抗性文本嵌入，显著降低攻击成功率(ASR)。令人印象深刻的是，与领先的白盒攻击方法相比，RACE实现了“裸体”概念的ASR降低30个百分点。我们广泛的评估证明了RACE在防御白盒和黑盒攻击方面的有效性，标志着在保护T2I扩散模型免受不适当或误导图像方面取得了重大进展。这项工作突出表明，必须采取积极主动的防御措施，以适应迅速发展的对抗性挑战领域。我们的代码是公开提供的：\url{https://github.com/chkimmmmm/R.A.C.E.}



## **37. Algebraic Adversarial Attacks on Integrated Gradients**

对综合学生的代数对抗攻击 cs.LG

**SubmitDate**: 2024-07-23    [abs](http://arxiv.org/abs/2407.16233v1) [paper-pdf](http://arxiv.org/pdf/2407.16233v1)

**Authors**: Lachlan Simpson, Federico Costanza, Kyle Millar, Adriel Cheng, Cheng-Chew Lim, Hong Gunn Chew

**Abstract**: Adversarial attacks on explainability models have drastic consequences when explanations are used to understand the reasoning of neural networks in safety critical systems. Path methods are one such class of attribution methods susceptible to adversarial attacks. Adversarial learning is typically phrased as a constrained optimisation problem. In this work, we propose algebraic adversarial examples and study the conditions under which one can generate adversarial examples for integrated gradients. Algebraic adversarial examples provide a mathematically tractable approach to adversarial examples.

摘要: 当使用解释来理解安全关键系统中神经网络的推理时，对可解释性模型的对抗攻击会产生严重后果。路径方法是一类容易受到对抗攻击的归因方法。对抗学习通常被描述为一个受约束的优化问题。在这项工作中，我们提出了代数对抗性示例，并研究了可以为综合梯度生成对抗性示例的条件。代数对抗性例子为对抗性例子提供了一种数学上易于处理的方法。



## **38. EVD4UAV: An Altitude-Sensitive Benchmark to Evade Vehicle Detection in UAV**

EVD 4无人机：躲避无人机车辆检测的高度敏感基准 cs.CV

**SubmitDate**: 2024-07-22    [abs](http://arxiv.org/abs/2403.05422v2) [paper-pdf](http://arxiv.org/pdf/2403.05422v2)

**Authors**: Huiming Sun, Jiacheng Guo, Zibo Meng, Tianyun Zhang, Jianwu Fang, Yuewei Lin, Hongkai Yu

**Abstract**: Vehicle detection in Unmanned Aerial Vehicle (UAV) captured images has wide applications in aerial photography and remote sensing. There are many public benchmark datasets proposed for the vehicle detection and tracking in UAV images. Recent studies show that adding an adversarial patch on objects can fool the well-trained deep neural networks based object detectors, posing security concerns to the downstream tasks. However, the current public UAV datasets might ignore the diverse altitudes, vehicle attributes, fine-grained instance-level annotation in mostly side view with blurred vehicle roof, so none of them is good to study the adversarial patch based vehicle detection attack problem. In this paper, we propose a new dataset named EVD4UAV as an altitude-sensitive benchmark to evade vehicle detection in UAV with 6,284 images and 90,886 fine-grained annotated vehicles. The EVD4UAV dataset has diverse altitudes (50m, 70m, 90m), vehicle attributes (color, type), fine-grained annotation (horizontal and rotated bounding boxes, instance-level mask) in top view with clear vehicle roof. One white-box and two black-box patch based attack methods are implemented to attack three classic deep neural networks based object detectors on EVD4UAV. The experimental results show that these representative attack methods could not achieve the robust altitude-insensitive attack performance.

摘要: 无人机拍摄的图像中的车辆检测在航空摄影和遥感中有着广泛的应用。针对无人机图像中的车辆检测和跟踪，已经提出了许多公开的基准数据集。最近的研究表明，在对象上添加对抗性补丁可以欺骗训练有素的基于深度神经网络的对象检测器，从而给下游任务带来安全隐患。然而，目前公开的无人机数据集可能忽略了车顶模糊的侧视图中不同的高度、车辆属性、细粒度的实例级标注，因此不利于研究基于对抗性补丁的车辆检测攻击问题。在本文中，我们提出了一个新的数据集EVD4UAV作为高度敏感基准来逃避无人机中的车辆检测，该数据集包含6,284张图像和90,886辆细粒度标注的车辆。EVD4UAV数据集在俯视图中具有不同的高度(50m、70m、90m)、车辆属性(颜色、类型)、细粒度注释(水平和旋转的边界框、实例级遮罩)，并具有清晰的车顶。采用一种基于白盒和两种基于黑盒补丁的攻击方法，对EVD4无人机上三种经典的基于深度神经网络的目标探测器进行攻击。实验结果表明，这些具有代表性的攻击方法不能达到稳健的高度不敏感攻击性能。



## **39. Detecting Brittle Decisions for Free: Leveraging Margin Consistency in Deep Robust Classifiers**

免费检测脆弱决策：利用深度稳健分类器中的保证金一致性 cs.LG

11 pages, 7 figures, 2 tables, 1 algorithm. Version Update: Figure 6

**SubmitDate**: 2024-07-22    [abs](http://arxiv.org/abs/2406.18451v2) [paper-pdf](http://arxiv.org/pdf/2406.18451v2)

**Authors**: Jonas Ngnawé, Sabyasachi Sahoo, Yann Pequignot, Frédéric Precioso, Christian Gagné

**Abstract**: Despite extensive research on adversarial training strategies to improve robustness, the decisions of even the most robust deep learning models can still be quite sensitive to imperceptible perturbations, creating serious risks when deploying them for high-stakes real-world applications. While detecting such cases may be critical, evaluating a model's vulnerability at a per-instance level using adversarial attacks is computationally too intensive and unsuitable for real-time deployment scenarios. The input space margin is the exact score to detect non-robust samples and is intractable for deep neural networks. This paper introduces the concept of margin consistency -- a property that links the input space margins and the logit margins in robust models -- for efficient detection of vulnerable samples. First, we establish that margin consistency is a necessary and sufficient condition to use a model's logit margin as a score for identifying non-robust samples. Next, through comprehensive empirical analysis of various robustly trained models on CIFAR10 and CIFAR100 datasets, we show that they indicate strong margin consistency with a strong correlation between their input space margins and the logit margins. Then, we show that we can effectively use the logit margin to confidently detect brittle decisions with such models and accurately estimate robust accuracy on an arbitrarily large test set by estimating the input margins only on a small subset. Finally, we address cases where the model is not sufficiently margin-consistent by learning a pseudo-margin from the feature representation. Our findings highlight the potential of leveraging deep representations to efficiently assess adversarial vulnerability in deployment scenarios.

摘要: 尽管对对抗性训练策略进行了大量研究以提高稳健性，但即使是最健壮的深度学习模型的决策也可能对不可察觉的扰动非常敏感，当将它们部署到高风险的现实世界应用程序时，会产生严重的风险。虽然检测这类情况可能很关键，但使用对抗性攻击在每个实例级别评估模型的漏洞计算量太大，不适合实时部署场景。输入空间裕度是检测非稳健样本的准确分数，对于深度神经网络来说是很难处理的。为了有效地检测易受攻击的样本，本文引入了边缘一致性的概念--一种将输入空间边缘和健壮模型中的Logit边缘联系起来的属性。首先，我们证明了边际一致性是使用模型的Logit边际作为识别非稳健样本的分数的充要条件。接下来，通过对CIFAR10和CIFAR100数据集上各种稳健训练模型的综合实证分析，我们发现它们表明了很强的边际一致性，并且它们的输入空间边际和Logit边际之间存在很强的相关性。然后，我们证明了我们可以有效地使用Logit裕度来自信地检测此类模型的脆性决策，并通过仅在较小的子集上估计输入裕度来准确地估计任意大测试集上的稳健精度。最后，我们通过从特征表示学习伪边距来处理模型不够边距一致的情况。我们的发现突出了利用深度陈述来有效评估部署场景中的对手脆弱性的潜力。



## **40. Rainbow Teaming: Open-Ended Generation of Diverse Adversarial Prompts**

彩虹团队：开放式一代的多元化对抗预言 cs.CL

**SubmitDate**: 2024-07-22    [abs](http://arxiv.org/abs/2402.16822v2) [paper-pdf](http://arxiv.org/pdf/2402.16822v2)

**Authors**: Mikayel Samvelyan, Sharath Chandra Raparthy, Andrei Lupu, Eric Hambro, Aram H. Markosyan, Manish Bhatt, Yuning Mao, Minqi Jiang, Jack Parker-Holder, Jakob Foerster, Tim Rocktäschel, Roberta Raileanu

**Abstract**: As large language models (LLMs) become increasingly prevalent across many real-world applications, understanding and enhancing their robustness to adversarial attacks is of paramount importance. Existing methods for identifying adversarial prompts tend to focus on specific domains, lack diversity, or require extensive human annotations. To address these limitations, we present Rainbow Teaming, a novel black-box approach for producing a diverse collection of adversarial prompts. Rainbow Teaming casts adversarial prompt generation as a quality-diversity problem, and uses open-ended search to generate prompts that are both effective and diverse. Focusing on the safety domain, we use Rainbow Teaming to target various state-of-the-art LLMs, including the Llama 2 and Llama 3 models. Our approach reveals hundreds of effective adversarial prompts, with an attack success rate exceeding 90% across all tested models. Furthermore, we demonstrate that fine-tuning models with synthetic data generated by the Rainbow Teaming method significantly enhances their safety without sacrificing general performance or helpfulness. We additionally explore the versatility of Rainbow Teaming by applying it to question answering and cybersecurity, showcasing its potential to drive robust open-ended self-improvement in a wide range of applications.

摘要: 随着大型语言模型(LLM)在许多真实世界的应用中变得越来越普遍，理解和增强它们对对手攻击的健壮性是至关重要的。现有的识别对抗性提示的方法往往集中在特定的领域，缺乏多样性，或者需要大量的人工注释。为了解决这些局限性，我们提出了彩虹分组，这是一种新的黑盒方法，用于产生多样化的对抗性提示集合。彩虹团队将敌意提示生成视为质量多样性问题，并使用开放式搜索来生成既有效又多样化的提示。专注于安全领域，我们使用彩虹团队瞄准各种最先进的LLM，包括Llama 2和Llama 3型号。我们的方法揭示了数百个有效的对抗性提示，在所有测试模型上的攻击成功率超过90%。此外，我们证明了使用彩虹组合方法生成的合成数据对模型进行微调显著增强了它们的安全性，而不会牺牲总体性能或帮助。我们还探讨了彩虹团队的多功能性，将其应用于问题回答和网络安全，展示了其在广泛应用中推动强大的开放式自我改进的潜力。



## **41. Enhancing Transferability of Targeted Adversarial Examples: A Self-Universal Perspective**

增强有针对性的对抗性示例的可移植性：自我普遍的视角 cs.CV

8 pages and 9 figures

**SubmitDate**: 2024-07-22    [abs](http://arxiv.org/abs/2407.15683v1) [paper-pdf](http://arxiv.org/pdf/2407.15683v1)

**Authors**: Bowen Peng, Li Liu, Tianpeng Liu, Zhen Liu, Yongxiang Liu

**Abstract**: Transfer-based targeted adversarial attacks against black-box deep neural networks (DNNs) have been proven to be significantly more challenging than untargeted ones. The impressive transferability of current SOTA, the generative methods, comes at the cost of requiring massive amounts of additional data and time-consuming training for each targeted label. This results in limited efficiency and flexibility, significantly hindering their deployment in practical applications. In this paper, we offer a self-universal perspective that unveils the great yet underexplored potential of input transformations in pursuing this goal. Specifically, transformations universalize gradient-based attacks with intrinsic but overlooked semantics inherent within individual images, exhibiting similar scalability and comparable results to time-consuming learning over massive additional data from diverse classes. We also contribute a surprising empirical insight that one of the most fundamental transformations, simple image scaling, is highly effective, scalable, sufficient, and necessary in enhancing targeted transferability. We further augment simple scaling with orthogonal transformations and block-wise applicability, resulting in the Simple, faSt, Self-universal yet Strong Scale Transformation (S$^4$ST) for self-universal TTA. On the ImageNet-Compatible benchmark dataset, our method achieves a 19.8% improvement in the average targeted transfer success rate against various challenging victim models over existing SOTA transformation methods while only consuming 36% time for attacking. It also outperforms resource-intensive attacks by a large margin in various challenging settings.

摘要: 针对黑盒深度神经网络(DNN)的基于转移的定向攻击已被证明比非定向攻击具有更大的挑战性。当前SOTA的可转移性令人印象深刻，这是以需要大量额外数据和为每个目标标签进行耗时的培训为代价的。这导致了效率和灵活性的限制，大大阻碍了它们在实际应用中的部署。在这篇文章中，我们提供了一个自我普适的视角，揭示了投入转换在追求这一目标方面的巨大潜力，但尚未得到充分开发。具体地说，变换将基于梯度的攻击通用化，具有内在但被忽略的个体图像固有的语义，表现出与来自不同类别的耗时的额外数据的耗时学习类似的可扩展性和可比性。我们还提供了一个令人惊讶的经验见解，即最基本的转换之一，简单的图像缩放，在增强目标可转移性方面是高度有效、可扩展、充分和必要的。我们用正交变换和分块适用性进一步增强了简单的尺度变换，得到了简单、快速、自泛的强尺度变换(S$^4$ST)。在与ImageNet兼容的基准数据集上，与已有的SOTA变换方法相比，该方法对各种具有挑战性的受害者模型的平均目标传输成功率提高了19.8%，而攻击所需的时间仅为36%。在各种具有挑战性的环境中，它的性能也远远超过资源密集型攻击。



## **42. Adversarial Style Augmentation via Large Language Model for Robust Fake News Detection**

通过大语言模型进行对抗风格增强以实现稳健的假新闻检测 cs.CL

8 pages

**SubmitDate**: 2024-07-22    [abs](http://arxiv.org/abs/2406.11260v2) [paper-pdf](http://arxiv.org/pdf/2406.11260v2)

**Authors**: Sungwon Park, Sungwon Han, Meeyoung Cha

**Abstract**: The spread of fake news negatively impacts individuals and is regarded as a significant social challenge that needs to be addressed. A number of algorithmic and insightful features have been identified for detecting fake news. However, with the recent LLMs and their advanced generation capabilities, many of the detectable features (e.g., style-conversion attacks) can be altered, making it more challenging to distinguish from real news. This study proposes adversarial style augmentation, AdStyle, to train a fake news detector that remains robust against various style-conversion attacks. Our model's key mechanism is the careful use of LLMs to automatically generate a diverse yet coherent range of style-conversion attack prompts. This improves the generation of prompts that are particularly difficult for the detector to handle. Experiments show that our augmentation strategy improves robustness and detection performance when tested on fake news benchmark datasets.

摘要: 假新闻的传播对个人产生负面影响，被视为需要解决的重大社会挑战。已经确定了许多算法和有洞察力的功能来检测假新闻。然而，随着最近的LLM及其先进一代能力，许多可检测的特征（例如，风格转换攻击）可以被更改，使其与真实新闻区分起来更具挑战性。这项研究提出了对抗性风格增强AdStyle来训练一个假新闻检测器，该检测器在对抗各种风格转换攻击时保持稳健。我们模型的关键机制是仔细使用LLM来自动生成多样化但连贯的风格转换攻击提示。这改善了检测器特别难以处理的提示的生成。实验表明，当在假新闻基准数据集上进行测试时，我们的增强策略提高了鲁棒性和检测性能。



## **43. Revisiting the Robust Alignment of Circuit Breakers**

重新审视断路器的稳健对准 cs.CR

**SubmitDate**: 2024-07-22    [abs](http://arxiv.org/abs/2407.15902v1) [paper-pdf](http://arxiv.org/pdf/2407.15902v1)

**Authors**: Leo Schwinn, Simon Geisler

**Abstract**: Over the past decade, adversarial training has emerged as one of the few reliable methods for enhancing model robustness against adversarial attacks [Szegedy et al., 2014, Madry et al., 2018, Xhonneux et al., 2024], while many alternative approaches have failed to withstand rigorous subsequent evaluations. Recently, an alternative defense mechanism, namely "circuit breakers" [Zou et al., 2024], has shown promising results for aligning LLMs. In this report, we show that the robustness claims of "Improving Alignment and Robustness with Circuit Breakers" against unconstraint continuous attacks in the embedding space of the input tokens may be overestimated [Zou et al., 2024]. Specifically, we demonstrate that by implementing a few simple changes to embedding space attacks [Schwinn et al., 2024a,b], we achieve 100% attack success rate (ASR) against circuit breaker models. Without conducting any further hyperparameter tuning, these adjustments increase the ASR by more than 80% compared to the original evaluation. Code is accessible at: https://github.com/SchwinnL/circuit-breakers-eval

摘要: 在过去的十年中，对抗性训练已经成为少数几种增强模型对对抗性攻击的稳健性的可靠方法之一[Szegedy等人，2014，Madry等人，2018，Xhonneux等人，2024]，而许多替代方法未能经受住严格的后续评估。最近，一种替代的防御机制，即“断路器”[Zou等人，2024]，在对准LLM方面显示了令人振奋的结果。在这份报告中，我们证明了在输入令牌的嵌入空间中使用断路器来提高对非约束连续攻击的稳健性主张可能被高估了[Zou等人，2024]。具体地说，我们证明了通过对嵌入空间攻击[Schwinn等人，2024a，b]进行一些简单的更改，我们对断路器模型实现了100%的攻击成功率(ASR)。在不进行任何进一步的超参数调整的情况下，这些调整使ASR与原始评估相比增加了80%以上。代码可在以下网址访问：https://github.com/SchwinnL/circuit-breakers-eval



## **44. Targeted Latent Adversarial Training Improves Robustness to Persistent Harmful Behaviors in LLMs**

有针对性的隐性对抗培训提高了LLM对持续有害行为的稳健性 cs.LG

**SubmitDate**: 2024-07-22    [abs](http://arxiv.org/abs/2407.15549v1) [paper-pdf](http://arxiv.org/pdf/2407.15549v1)

**Authors**: Abhay Sheshadri, Aidan Ewart, Phillip Guo, Aengus Lynch, Cindy Wu, Vivek Hebbar, Henry Sleight, Asa Cooper Stickland, Ethan Perez, Dylan Hadfield-Menell, Stephen Casper

**Abstract**: Large language models (LLMs) can often be made to behave in undesirable ways that they are explicitly fine-tuned not to. For example, the LLM red-teaming literature has produced a wide variety of `jailbreaking' techniques to elicit harmful text from models that were fine-tuned to be harmless. Recent work on red-teaming, model editing, and interpretability suggests that this challenge stems from how (adversarial) fine-tuning largely serves to suppress rather than remove undesirable capabilities from LLMs. Prior work has introduced latent adversarial training (LAT) as a way to improve robustness to broad classes of failures. These prior works have considered untargeted latent space attacks where the adversary perturbs latent activations to maximize loss on examples of desirable behavior. Untargeted LAT can provide a generic type of robustness but does not leverage information about specific failure modes. Here, we experiment with targeted LAT where the adversary seeks to minimize loss on a specific competing task. We find that it can augment a wide variety of state-of-the-art methods. First, we use targeted LAT to improve robustness to jailbreaks, outperforming a strong R2D2 baseline with orders of magnitude less compute. Second, we use it to more effectively remove backdoors with no knowledge of the trigger. Finally, we use it to more effectively unlearn knowledge for specific undesirable tasks in a way that is also more robust to re-learning. Overall, our results suggest that targeted LAT can be an effective tool for defending against harmful behaviors from LLMs.

摘要: 大型语言模型(LLM)通常会以不受欢迎的方式运行，因此它们被明确微调为不以这种方式运行。例如，伦敦大学法学院的红队文学创作了各种各样的“越狱”技术，从经过微调的无害模特那里引出有害文本。最近在红团队、模型编辑和可解释性方面的工作表明，这一挑战源于(对抗性的)微调如何在很大程度上抑制而不是消除LLM中不受欢迎的能力。以前的工作已经引入了潜在的对手训练(LAT)，作为一种提高对广泛类别的故障的稳健性的方式。这些先前的工作考虑了无目标的潜在空间攻击，即对手扰乱潜在激活，以最大限度地减少期望行为的示例损失。非定向LAT可以提供一般类型的健壮性，但不利用有关特定故障模式的信息。在这里，我们实验有针对性的LAT，其中对手试图将特定竞争任务的损失降至最低。我们发现，它可以增加各种最先进的方法。首先，我们使用有针对性的LAT来提高对越狱的健壮性，性能优于强大的R2D2基线，计算量少了几个数量级。其次，我们使用它来更有效地删除后门，而不知道触发器。最后，我们使用它来更有效地忘记特定不受欢迎的任务的知识，这种方式也更适合重新学习。总体而言，我们的结果表明，有针对性的LAT可以成为防御LLM有害行为的有效工具。



## **45. Towards Efficient Transferable Preemptive Adversarial Defense**

迈向高效的可转让先发制人的对抗防御 cs.CR

Under Review

**SubmitDate**: 2024-07-22    [abs](http://arxiv.org/abs/2407.15524v1) [paper-pdf](http://arxiv.org/pdf/2407.15524v1)

**Authors**: Hanrui Wang, Ching-Chun Chang, Chun-Shien Lu, Isao Echizen

**Abstract**: Deep learning technology has brought convenience and advanced developments but has become untrustworthy because of its sensitivity to inconspicuous perturbations (i.e., adversarial attacks). Attackers utilize this sensitivity to slightly manipulate transmitted messages. To defend against such attacks, we have devised a strategy for "attacking" the message before it is attacked. This strategy, dubbed Fast Preemption, provides an efficient transferable preemptive defense by using different models for labeling inputs and learning crucial features. A forward-backward cascade learning algorithm is used to compute protective perturbations, starting with forward propagation optimization to achieve rapid convergence, followed by iterative backward propagation learning to alleviate overfitting. This strategy offers state-of-the-art transferability and protection across various systems. With the running of only three steps, our Fast Preemption framework outperforms benchmark training-time, test-time, and preemptive adversarial defenses. We have also devised the first to our knowledge effective white-box adaptive reversion attack and demonstrate that the protection added by our defense strategy is irreversible unless the backbone model, algorithm, and settings are fully compromised. This work provides a new direction to developing active defenses against adversarial attacks.

摘要: 深度学习技术带来了便利和先进的发展，但由于其对不起眼的扰动(即对抗性攻击)的敏感性而变得不可信任。攻击者利用这种敏感度略微操纵传输的消息。为了防御这种攻击，我们设计了一种在消息受到攻击之前对其进行“攻击”的策略。这一战略被称为快速抢占，通过使用不同的模型来标记输入和学习关键特征，提供了一种高效的可转移的抢占防御。保护摄动的计算采用前向-后向级联学习算法，从前向传播优化开始实现快速收敛，然后迭代后向传播学习以减少过拟合。这一战略提供了最先进的跨各种系统的可转移性和保护。由于只运行了三个步骤，我们的快速抢占框架的性能优于基准训练时间、测试时间和先发制人的对手防御。我们还设计了我们所知的第一个有效的白盒自适应恢复攻击，并证明了我们的防御策略添加的保护是不可逆转的，除非主干模型、算法和设置完全受损。这一工作为主动防御敌方攻击提供了新的方向。



## **46. A Closer Look at GAN Priors: Exploiting Intermediate Features for Enhanced Model Inversion Attacks**

仔细研究GAN先验：利用中间功能进行增强模型倒置攻击 cs.CV

ECCV 2024

**SubmitDate**: 2024-07-22    [abs](http://arxiv.org/abs/2407.13863v2) [paper-pdf](http://arxiv.org/pdf/2407.13863v2)

**Authors**: Yixiang Qiu, Hao Fang, Hongyao Yu, Bin Chen, MeiKang Qiu, Shu-Tao Xia

**Abstract**: Model Inversion (MI) attacks aim to reconstruct privacy-sensitive training data from released models by utilizing output information, raising extensive concerns about the security of Deep Neural Networks (DNNs). Recent advances in generative adversarial networks (GANs) have contributed significantly to the improved performance of MI attacks due to their powerful ability to generate realistic images with high fidelity and appropriate semantics. However, previous MI attacks have solely disclosed private information in the latent space of GAN priors, limiting their semantic extraction and transferability across multiple target models and datasets. To address this challenge, we propose a novel method, Intermediate Features enhanced Generative Model Inversion (IF-GMI), which disassembles the GAN structure and exploits features between intermediate blocks. This allows us to extend the optimization space from latent code to intermediate features with enhanced expressive capabilities. To prevent GAN priors from generating unrealistic images, we apply a L1 ball constraint to the optimization process. Experiments on multiple benchmarks demonstrate that our method significantly outperforms previous approaches and achieves state-of-the-art results under various settings, especially in the out-of-distribution (OOD) scenario. Our code is available at: https://github.com/final-solution/IF-GMI

摘要: 模型反转(MI)攻击的目的是利用输出信息从已发布的模型中重建隐私敏感的训练数据，这引起了人们对深度神经网络(DNN)安全性的广泛关注。生成性对抗网络(GANS)的最新进展为MI攻击的性能改进做出了重要贡献，因为它们能够生成高保真和适当语义的真实图像。然而，以往的MI攻击只在GaN先验的潜在空间中泄露隐私信息，限制了它们的语义提取和跨多个目标模型和数据集的可传输性。为了解决这一挑战，我们提出了一种新的方法，中间特征增强的生成性模型反转(IF-GMI)，它分解GaN结构并利用中间块之间的特征。这允许我们将优化空间从潜在代码扩展到具有增强表达能力的中间功能。为了防止GaN先验数据产生不真实的图像，我们在优化过程中应用了L1球约束。在多个基准测试上的实验表明，我们的方法显著优于以前的方法，并在各种设置下获得了最先进的结果，特别是在分布外(OOD)的情况下。我们的代码请访问：https://github.com/final-solution/IF-GMI



## **47. CLIP-Guided Networks for Transferable Targeted Attacks**

CLIP引导的可转移定向攻击网络 cs.CV

ECCV 2024

**SubmitDate**: 2024-07-22    [abs](http://arxiv.org/abs/2407.10179v2) [paper-pdf](http://arxiv.org/pdf/2407.10179v2)

**Authors**: Hao Fang, Jiawei Kong, Bin Chen, Tao Dai, Hao Wu, Shu-Tao Xia

**Abstract**: Transferable targeted adversarial attacks aim to mislead models into outputting adversary-specified predictions in black-box scenarios. Recent studies have introduced \textit{single-target} generative attacks that train a generator for each target class to generate highly transferable perturbations, resulting in substantial computational overhead when handling multiple classes. \textit{Multi-target} attacks address this by training only one class-conditional generator for multiple classes. However, the generator simply uses class labels as conditions, failing to leverage the rich semantic information of the target class. To this end, we design a \textbf{C}LIP-guided \textbf{G}enerative \textbf{N}etwork with \textbf{C}ross-attention modules (CGNC) to enhance multi-target attacks by incorporating textual knowledge of CLIP into the generator. Extensive experiments demonstrate that CGNC yields significant improvements over previous multi-target generative attacks, e.g., a 21.46\% improvement in success rate from ResNet-152 to DenseNet-121. Moreover, we propose a masked fine-tuning mechanism to further strengthen our method in attacking a single class, which surpasses existing single-target methods.

摘要: 可转移的目标对抗性攻击旨在误导模型，使其在黑盒场景中输出对手指定的预测。最近的研究引入了生成性攻击，这种攻击为每个目标类训练一个生成器来生成高度可传递的扰动，导致在处理多个类时产生大量的计算开销。\textit{多目标}攻击通过仅训练多个类的一个类条件生成器来解决此问题。然而，生成器简单地使用类标签作为条件，没有利用目标类的丰富语义信息。为此，我们设计了一个唇形引导的生成模块(CGNC)，通过在生成器中加入剪辑文本知识来增强多目标攻击。大量的实验表明，CGNC比以前的多目标生成性攻击有显著的改进，例如，成功率从ResNet-152提高到DenseNet-121，提高了21.46%.此外，我们还提出了一种屏蔽微调机制，进一步加强了我们的攻击单一类的方法，超越了现有的单目标攻击方法。



## **48. TAPI: Towards Target-Specific and Adversarial Prompt Injection against Code LLMs**

TAPI：针对代码LLM的目标特定和对抗性即时注入 cs.CR

**SubmitDate**: 2024-07-22    [abs](http://arxiv.org/abs/2407.09164v3) [paper-pdf](http://arxiv.org/pdf/2407.09164v3)

**Authors**: Yuchen Yang, Hongwei Yao, Bingrun Yang, Yiling He, Yiming Li, Tianwei Zhang, Zhan Qin, Kui Ren

**Abstract**: Recently, code-oriented large language models (Code LLMs) have been widely and successfully used to simplify and facilitate code programming. With these tools, developers can easily generate desired complete functional codes based on incomplete code and natural language prompts. However, a few pioneering works revealed that these Code LLMs are also vulnerable, e.g., against backdoor and adversarial attacks. The former could induce LLMs to respond to triggers to insert malicious code snippets by poisoning the training data or model parameters, while the latter can craft malicious adversarial input codes to reduce the quality of generated codes. However, both attack methods have underlying limitations: backdoor attacks rely on controlling the model training process, while adversarial attacks struggle with fulfilling specific malicious purposes.   To inherit the advantages of both backdoor and adversarial attacks, this paper proposes a new attack paradigm, i.e., target-specific and adversarial prompt injection (TAPI), against Code LLMs. TAPI generates unreadable comments containing information about malicious instructions and hides them as triggers in the external source code. When users exploit Code LLMs to complete codes containing the trigger, the models will generate attacker-specified malicious code snippets at specific locations. We evaluate our TAPI attack on four representative LLMs under three representative malicious objectives and seven cases. The results show that our method is highly threatening (achieving an attack success rate of up to 98.3%) and stealthy (saving an average of 53.1% of tokens in the trigger design). In particular, we successfully attack some famous deployed code completion integrated applications, including CodeGeex and Github Copilot. This further confirms the realistic threat of our attack.

摘要: 最近，面向代码的大型语言模型(Code LLM)已被广泛并成功地用于简化和促进代码编程。使用这些工具，开发人员可以根据不完整的代码和自然语言提示轻松生成所需的完整功能代码。然而，一些开创性的工作表明，这些代码LLM也容易受到攻击，例如，抵御后门和对手攻击。前者可以通过毒化训练数据或模型参数来诱导LLMS响应插入恶意代码片段的触发器，而后者可以手工创建恶意输入代码来降低生成代码的质量。然而，这两种攻击方法都有潜在的局限性：后门攻击依赖于控制模型训练过程，而对抗性攻击则难以实现特定的恶意目的。为了继承后门攻击和对抗性攻击的优点，提出了一种新的针对Code LLMS的攻击范式，即目标特定和对抗性提示注入(TAPI)。TAPI生成不可读的注释，其中包含有关恶意指令的信息，并将它们作为触发器隐藏在外部源代码中。当用户利用Code LLMS来完成包含触发器的代码时，模型将在特定位置生成攻击者指定的恶意代码片段。我们在三个典型的恶意目标和七个案例下评估了我们的TAPI攻击对四个有代表性的LLM的攻击。结果表明，该方法具有很高的威胁性(攻击成功率高达98.3%)和隐蔽性(在触发器设计中平均节省53.1%的令牌)。特别是，我们成功地攻击了一些著名的部署代码完成集成应用程序，包括CodeGeex和Github Copilot。这进一步证实了我们攻击的现实威胁。



## **49. Imposter.AI: Adversarial Attacks with Hidden Intentions towards Aligned Large Language Models**

冒名顶替。AI：针对对齐大型语言模型的具有隐藏意图的对抗攻击 cs.CL

**SubmitDate**: 2024-07-22    [abs](http://arxiv.org/abs/2407.15399v1) [paper-pdf](http://arxiv.org/pdf/2407.15399v1)

**Authors**: Xiao Liu, Liangzhi Li, Tong Xiang, Fuying Ye, Lu Wei, Wangyue Li, Noa Garcia

**Abstract**: With the development of large language models (LLMs) like ChatGPT, both their vast applications and potential vulnerabilities have come to the forefront. While developers have integrated multiple safety mechanisms to mitigate their misuse, a risk remains, particularly when models encounter adversarial inputs. This study unveils an attack mechanism that capitalizes on human conversation strategies to extract harmful information from LLMs. We delineate three pivotal strategies: (i) decomposing malicious questions into seemingly innocent sub-questions; (ii) rewriting overtly malicious questions into more covert, benign-sounding ones; (iii) enhancing the harmfulness of responses by prompting models for illustrative examples. Unlike conventional methods that target explicit malicious responses, our approach delves deeper into the nature of the information provided in responses. Through our experiments conducted on GPT-3.5-turbo, GPT-4, and Llama2, our method has demonstrated a marked efficacy compared to conventional attack methods. In summary, this work introduces a novel attack method that outperforms previous approaches, raising an important question: How to discern whether the ultimate intent in a dialogue is malicious?

摘要: 随着像ChatGPT这样的大型语言模型(LLM)的发展，它们的巨大应用和潜在的漏洞都已经浮出水面。虽然开发人员已经集成了多种安全机制来减少它们的滥用，但风险仍然存在，特别是当模型遇到敌对输入时。这项研究揭示了一种利用人类对话策略从LLMS中提取有害信息的攻击机制。我们描述了三个关键策略：(I)将恶意问题分解为看似无害的子问题；(Ii)将公开的恶意问题重写为更隐蔽、听起来更温和的问题；(Iii)通过提示示例模型来增强回答的危害性。与针对显式恶意响应的传统方法不同，我们的方法更深入地挖掘响应中提供的信息的性质。通过我们在GPT-3.5-Turbo、GPT-4和Llama2上的实验，我们的方法比传统的攻击方法表现出了显著的效果。总之，这项工作引入了一种新的攻击方法，其性能优于以前的方法，提出了一个重要的问题：如何识别对话中的最终意图是否为恶意的？



## **50. Towards Robust Vision Transformer via Masked Adaptive Ensemble**

通过掩蔽自适应集合迈向稳健的视觉Transformer cs.CV

9 pages

**SubmitDate**: 2024-07-22    [abs](http://arxiv.org/abs/2407.15385v1) [paper-pdf](http://arxiv.org/pdf/2407.15385v1)

**Authors**: Fudong Lin, Jiadong Lou, Xu Yuan, Nian-Feng Tzeng

**Abstract**: Adversarial training (AT) can help improve the robustness of Vision Transformers (ViT) against adversarial attacks by intentionally injecting adversarial examples into the training data. However, this way of adversarial injection inevitably incurs standard accuracy degradation to some extent, thereby calling for a trade-off between standard accuracy and robustness. Besides, the prominent AT solutions are still vulnerable to adaptive attacks. To tackle such shortcomings, this paper proposes a novel ViT architecture, including a detector and a classifier bridged by our newly developed adaptive ensemble. Specifically, we empirically discover that detecting adversarial examples can benefit from the Guided Backpropagation technique. Driven by this discovery, a novel Multi-head Self-Attention (MSA) mechanism is introduced to enhance our detector to sniff adversarial examples. Then, a classifier with two encoders is employed for extracting visual representations respectively from clean images and adversarial examples, with our adaptive ensemble to adaptively adjust the proportion of visual representations from the two encoders for accurate classification. This design enables our ViT architecture to achieve a better trade-off between standard accuracy and robustness. Besides, our adaptive ensemble technique allows us to mask off a random subset of image patches within input data, boosting our ViT's robustness against adaptive attacks, while maintaining high standard accuracy. Experimental results exhibit that our ViT architecture, on CIFAR-10, achieves the best standard accuracy and adversarial robustness of 90.3% and 49.8%, respectively.

摘要: 对抗性训练(AT)通过有意地在训练数据中注入对抗性例子，有助于提高视觉转换器(VIT)对对抗性攻击的稳健性。然而，这种对抗性注入方式不可避免地会在一定程度上导致标准精度的下降，从而需要在标准精度和稳健性之间进行权衡。此外，突出的AT解决方案仍然容易受到适应性攻击。为了解决这些不足，本文提出了一种新颖的VIT体系结构，包括一个检测器和一个由我们新开发的自适应集成连接的分类器。具体地说，我们经验地发现，检测敌意例子可以受益于引导反向传播技术。在这一发现的推动下，引入了一种新的多头自我注意(MSA)机制来增强我们的检测器来嗅探敌意例子。然后，使用一个带有两个编码器的分类器分别从干净图像和对抗性样本中提取视觉表征，并通过自适应集成自适应地调整两个编码器的视觉表征比例以实现准确分类。这种设计使我们的VIT架构能够在标准准确性和健壮性之间实现更好的平衡。此外，我们的自适应集成技术允许我们屏蔽输入数据中的图像补丁的随机子集，增强我们的VIT对自适应攻击的稳健性，同时保持高标准精度。实验结果表明，我们的VIT架构在CIFAR-10上获得了90.3%的标准准确率和49.8%的对手健壮性。



