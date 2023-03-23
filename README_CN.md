# Latest Adversarial Attack Papers
**update at 2023-03-23 17:09:02**

翻译来自 https://cloud.tencent.com/document/product/551/15619

## **1. Evaluating the Role of Target Arguments in Rumour Stance Classification**

评价目标论元在流言立场分类中的作用 cs.CL

**SubmitDate**: 2023-03-22    [abs](http://arxiv.org/abs/2303.12665v1) [paper-pdf](http://arxiv.org/pdf/2303.12665v1)

**Authors**: Yue Li, Carolina Scarton

**Abstract**: Considering a conversation thread, stance classification aims to identify the opinion (e.g. agree or disagree) of replies towards a given target. The target of the stance is expected to be an essential component in this task, being one of the main factors that make it different from sentiment analysis. However, a recent study shows that a target-oblivious model outperforms target-aware models, suggesting that targets are not useful when predicting stance. This paper re-examines this phenomenon for rumour stance classification (RSC) on social media, where a target is a rumour story implied by the source tweet in the conversation. We propose adversarial attacks in the test data, aiming to assess the models robustness and evaluate the role of the data in the models performance. Results show that state-of-the-art models, including approaches that use the entire conversation thread, overly relying on superficial signals. Our hypothesis is that the naturally high occurrence of target-independent direct replies in RSC (e.g. "this is fake" or just "fake") results in the impressive performance of target-oblivious models, highlighting the risk of target instances being treated as noise during training.

摘要: 考虑到对话主线，立场分类旨在识别回复对给定目标的意见(例如同意或不同意)。预计立场的目标将是这项任务的重要组成部分，是使其不同于情绪分析的主要因素之一。然而，最近的一项研究表明，目标忽略模型的表现优于目标感知模型，这表明目标在预测姿态时并不有用。本文对社交媒体上的谣言立场分类(RSC)中的这一现象进行了重新审视，其中目标是对话中来源推文所暗示的谣言故事。我们在测试数据中提出对抗性攻击，目的是评估模型的稳健性，评估数据在模型性能中的作用。结果表明，最先进的模型，包括使用整个对话线索的方法，过度依赖表面信号。我们的假设是，目标无关的直接回复在RSC中的自然高出现(例如，这是假的或仅仅是假的)导致了目标忽略模型令人印象深刻的性能，突显了目标实例在训练过程中被视为噪声的风险。



## **2. Reliable and Efficient Evaluation of Adversarial Robustness for Deep Hashing-Based Retrieval**

基于深度散列的检索中对抗健壮性的可靠高效评估 cs.CV

arXiv admin note: text overlap with arXiv:2204.10779

**SubmitDate**: 2023-03-22    [abs](http://arxiv.org/abs/2303.12658v1) [paper-pdf](http://arxiv.org/pdf/2303.12658v1)

**Authors**: Xunguang Wang, Jiawang Bai, Xinyue Xu, Xiaomeng Li

**Abstract**: Deep hashing has been extensively applied to massive image retrieval due to its efficiency and effectiveness. Recently, several adversarial attacks have been presented to reveal the vulnerability of deep hashing models against adversarial examples. However, existing attack methods suffer from degraded performance or inefficiency because they underutilize the semantic relations between original samples or spend a lot of time learning these relations with a deep neural network. In this paper, we propose a novel Pharos-guided Attack, dubbed PgA, to evaluate the adversarial robustness of deep hashing networks reliably and efficiently. Specifically, we design pharos code to represent the semantics of the benign image, which preserves the similarity to semantically relevant samples and dissimilarity to irrelevant ones. It is proven that we can quickly calculate the pharos code via a simple math formula. Accordingly, PgA can directly conduct a reliable and efficient attack on deep hashing-based retrieval by maximizing the similarity between the hash code of the adversarial example and the pharos code. Extensive experiments on the benchmark datasets verify that the proposed algorithm outperforms the prior state-of-the-arts in both attack strength and speed.

摘要: 深度散列算法以其高效、高效的特点被广泛应用于海量图像检索中。最近，已经提出了几种对抗性攻击，以揭示深度散列模型对对抗性例子的脆弱性。然而，现有的攻击方法由于没有充分利用原始样本之间的语义关系或花费大量时间利用深度神经网络来学习这些关系，因此存在性能下降或效率低下的问题。本文提出了一种新的Pharos制导攻击，称为PGA，用于可靠、高效地评估深度散列网络的攻击健壮性。具体地说，我们设计了PHAROS代码来表示良性图像的语义，保持了对语义相关样本的相似性和对不相关样本的不相似性。实践证明，我们可以通过一个简单的数学公式快速地计算出航标码。因此，PGA可以通过最大化对抗性实例的哈希码和Pharos码之间的相似度，直接对基于深度哈希的检索进行可靠和高效的攻击。在基准数据集上的大量实验证明，该算法在攻击强度和速度上都优于现有的算法。



## **3. RoBIC: A benchmark suite for assessing classifiers robustness**

Robic：一种评估分类器健壮性的基准测试套件 cs.CV

4 pages, accepted to ICIP 2021

**SubmitDate**: 2023-03-22    [abs](http://arxiv.org/abs/2102.05368v2) [paper-pdf](http://arxiv.org/pdf/2102.05368v2)

**Authors**: Thibault Maho, Benoît Bonnet, Teddy Furon, Erwan Le Merrer

**Abstract**: Many defenses have emerged with the development of adversarial attacks. Models must be objectively evaluated accordingly. This paper systematically tackles this concern by proposing a new parameter-free benchmark we coin RoBIC. RoBIC fairly evaluates the robustness of image classifiers using a new half-distortion measure. It gauges the robustness of the network against white and black box attacks, independently of its accuracy. RoBIC is faster than the other available benchmarks. We present the significant differences in the robustness of 16 recent models as assessed by RoBIC.

摘要: 随着对抗性攻击的发展，出现了许多防御措施。必须相应地对模型进行客观评估。本文系统地解决了这个问题，提出了一个新的无参数基准，我们创造了Robic。Robic使用一种新的半失真度量公平地评估图像分类器的稳健性。它衡量网络抵御白盒和黑盒攻击的稳健性，而与其准确性无关。Robic比其他可用的基准测试更快。我们展示了Robic评估的16个最新模型在稳健性方面的显著差异。



## **4. CgAT: Center-Guided Adversarial Training for Deep Hashing-Based Retrieval**

CgAT：中心引导的基于深度散列的对抗性训练 cs.CV

**SubmitDate**: 2023-03-22    [abs](http://arxiv.org/abs/2204.10779v5) [paper-pdf](http://arxiv.org/pdf/2204.10779v5)

**Authors**: Xunguang Wang, Yiqun Lin, Xiaomeng Li

**Abstract**: Deep hashing has been extensively utilized in massive image retrieval because of its efficiency and effectiveness. However, deep hashing models are vulnerable to adversarial examples, making it essential to develop adversarial defense methods for image retrieval. Existing solutions achieved limited defense performance because of using weak adversarial samples for training and lacking discriminative optimization objectives to learn robust features. In this paper, we present a min-max based Center-guided Adversarial Training, namely CgAT, to improve the robustness of deep hashing networks through worst adversarial examples. Specifically, we first formulate the center code as a semantically-discriminative representative of the input image content, which preserves the semantic similarity with positive samples and dissimilarity with negative examples. We prove that a mathematical formula can calculate the center code immediately. After obtaining the center codes in each optimization iteration of the deep hashing network, they are adopted to guide the adversarial training process. On the one hand, CgAT generates the worst adversarial examples as augmented data by maximizing the Hamming distance between the hash codes of the adversarial examples and the center codes. On the other hand, CgAT learns to mitigate the effects of adversarial samples by minimizing the Hamming distance to the center codes. Extensive experiments on the benchmark datasets demonstrate the effectiveness of our adversarial training algorithm in defending against adversarial attacks for deep hashing-based retrieval. Compared with the current state-of-the-art defense method, we significantly improve the defense performance by an average of 18.61\%, 12.35\%, and 11.56\% on FLICKR-25K, NUS-WIDE, and MS-COCO, respectively. The code is available at https://github.com/xunguangwang/CgAT.

摘要: 深度哈希法以其高效、高效的特点在海量图像检索中得到了广泛应用。然而，深度哈希模型很容易受到敌意例子的攻击，因此有必要开发针对图像检索的对抗性防御方法。现有的解决方案由于使用弱对抗性样本进行训练，并且缺乏区分优化目标来学习稳健特征，使得防御性能受到限制。本文提出了一种基于最小-最大值的中心引导敌意训练算法，即CgAT，通过最坏的敌意例子来提高深度哈希网络的健壮性。具体地说，我们首先将中心代码定义为输入图像内容的语义区分代表，它保留了与正例的语义相似性和与反例的不相似性。我们证明了一个数学公式可以立即计算中心代码。在获得深度散列网络每次优化迭代的中心代码后，将其用于指导对抗性训练过程。一方面，CgAT通过最大化对抗性示例的哈希码与中心码之间的汉明距离来生成最差的对抗性示例作为扩充数据。另一方面，CgAT通过最小化到中心码的汉明距离来学习减轻对抗性样本的影响。在基准数据集上的大量实验表明，我们的对抗性训练算法在防御基于深度散列的检索的对抗性攻击方面是有效的。与当前最先进的防御方法相比，我们在Flickr-25K、NUS-wide和MS-CoCo上的防御性能分别平均提高了18.61、12.35和11.56。代码可在https://github.com/xunguangwang/CgAT.上获得



## **5. Membership Inference Attacks against Diffusion Models**

针对扩散模型的成员推理攻击 cs.CR

**SubmitDate**: 2023-03-22    [abs](http://arxiv.org/abs/2302.03262v2) [paper-pdf](http://arxiv.org/pdf/2302.03262v2)

**Authors**: Tomoya Matsumoto, Takayuki Miura, Naoto Yanai

**Abstract**: Diffusion models have attracted attention in recent years as innovative generative models. In this paper, we investigate whether a diffusion model is resistant to a membership inference attack, which evaluates the privacy leakage of a machine learning model. We primarily discuss the diffusion model from the standpoints of comparison with a generative adversarial network (GAN) as conventional models and hyperparameters unique to the diffusion model, i.e., time steps, sampling steps, and sampling variances. We conduct extensive experiments with DDIM as a diffusion model and DCGAN as a GAN on the CelebA and CIFAR-10 datasets in both white-box and black-box settings and then confirm if the diffusion model is comparably resistant to a membership inference attack as GAN. Next, we demonstrate that the impact of time steps is significant and intermediate steps in a noise schedule are the most vulnerable to the attack. We also found two key insights through further analysis. First, we identify that DDIM is vulnerable to the attack for small sample sizes instead of achieving a lower FID. Second, sampling steps in hyperparameters are important for resistance to the attack, whereas the impact of sampling variances is quite limited.

摘要: 扩散模型作为一种创新的生成性模型，近年来引起了人们的广泛关注。在本文中，我们研究了扩散模型是否抵抗成员推理攻击，以评估机器学习模型的隐私泄漏。我们主要从与生成性对抗网络(GAN)的传统模型和扩散模型特有的超参数，即时间步长、采样步长和采样方差的比较的角度来讨论扩散模型。我们使用DDIM作为扩散模型，DCGAN作为GaN，在CelebA和CIFAR-10数据集上进行了大量的白盒和黑盒环境下的实验，验证了扩散模型作为GaN是否具有同样的抗成员推理攻击的能力。接下来，我们证明了时间步长的影响是显著的，并且噪声调度中的中间步骤最容易受到攻击。通过进一步的分析，我们还发现了两个关键的见解。首先，我们发现ddim在样本大小较小的情况下容易受到攻击，而不是实现较低的FID。其次，超参数中的采样步长对于抵抗攻击很重要，而采样方差的影响相当有限。



## **6. Do Backdoors Assist Membership Inference Attacks?**

后门程序是否有助于成员资格推断攻击？ cs.CR

**SubmitDate**: 2023-03-22    [abs](http://arxiv.org/abs/2303.12589v1) [paper-pdf](http://arxiv.org/pdf/2303.12589v1)

**Authors**: Yumeki Goto, Nami Ashizawa, Toshiki Shibahara, Naoto Yanai

**Abstract**: When an adversary provides poison samples to a machine learning model, privacy leakage, such as membership inference attacks that infer whether a sample was included in the training of the model, becomes effective by moving the sample to an outlier. However, the attacks can be detected because inference accuracy deteriorates due to poison samples. In this paper, we discuss a \textit{backdoor-assisted membership inference attack}, a novel membership inference attack based on backdoors that return the adversary's expected output for a triggered sample. We found three crucial insights through experiments with an academic benchmark dataset. We first demonstrate that the backdoor-assisted membership inference attack is unsuccessful. Second, when we analyzed loss distributions to understand the reason for the unsuccessful results, we found that backdoors cannot separate loss distributions of training and non-training samples. In other words, backdoors cannot affect the distribution of clean samples. Third, we also show that poison and triggered samples activate neurons of different distributions. Specifically, backdoors make any clean sample an inlier, contrary to poisoning samples. As a result, we confirm that backdoors cannot assist membership inference.

摘要: 当敌手向机器学习模型提供有毒样本时，隐私泄露通过将样本移动到离群点而变得有效，例如推断样本是否包括在模型的训练中的成员关系推理攻击。然而，这些攻击是可以检测到的，因为由于毒物样本的原因，推断的准确性会恶化。本文讨论了一种新的基于后门的成员关系推理攻击--后门辅助成员关系推理攻击，它返回对手对触发样本的期望输出。通过对一个学术基准数据集的实验，我们发现了三个关键的见解。我们首先证明了后门辅助的成员推理攻击是不成功的。其次，当我们分析损失分布以了解不成功结果的原因时，我们发现后门不能分离训练样本和非训练样本的损失分布。换句话说，后门不能影响干净样本的分布。第三，我们还表明，毒物和触发样本激活了不同分布的神经元。具体地说，后门使任何干净的样本成为Inlier，与中毒样本相反。因此，我们确认后门不能帮助成员推断。



## **7. Autonomous Intelligent Cyber-defense Agent (AICA) Reference Architecture. Release 2.0**

自主智能网络防御代理(AICA)参考体系结构。版本2.0 cs.CR

This is a major revision and extension of the earlier release of AICA  Reference Architecture

**SubmitDate**: 2023-03-22    [abs](http://arxiv.org/abs/1803.10664v3) [paper-pdf](http://arxiv.org/pdf/1803.10664v3)

**Authors**: Alexander Kott, Paul Théron, Martin Drašar, Edlira Dushku, Benoît LeBlanc, Paul Losiewicz, Alessandro Guarino, Luigi Mancini, Agostino Panico, Mauno Pihelgas, Krzysztof Rzadca, Fabio De Gaspari

**Abstract**: This report - a major revision of its previous release - describes a reference architecture for intelligent software agents performing active, largely autonomous cyber-defense actions on military networks of computing and communicating devices. The report is produced by the North Atlantic Treaty Organization (NATO) Research Task Group (RTG) IST-152 "Intelligent Autonomous Agents for Cyber Defense and Resilience". In a conflict with a technically sophisticated adversary, NATO military tactical networks will operate in a heavily contested battlefield. Enemy software cyber agents - malware - will infiltrate friendly networks and attack friendly command, control, communications, computers, intelligence, surveillance, and reconnaissance and computerized weapon systems. To fight them, NATO needs artificial cyber hunters - intelligent, autonomous, mobile agents specialized in active cyber defense. With this in mind, in 2016, NATO initiated RTG IST-152. Its objective has been to help accelerate the development and transition to practice of such software agents by producing a reference architecture and technical roadmap. This report presents the concept and architecture of an Autonomous Intelligent Cyber-defense Agent (AICA). We describe the rationale of the AICA concept, explain the methodology and purpose that drive the definition of the AICA Reference Architecture, and review some of the main features and challenges of AICAs.

摘要: 这份报告是对之前发布的报告的重大修订，它描述了智能软件代理在军事计算和通信设备网络上执行主动的、基本上自主的网络防御行动的参考架构。该报告由北大西洋公约组织(NATO)研究任务组(RTG)IST-152《用于网络防御和弹性的智能自主代理》编制。在与技术复杂的对手的冲突中，北约军事战术网络将在一个竞争激烈的战场上运作。敌方软件网络代理--恶意软件--将渗透到友好的网络中，攻击友好的指挥、控制、通信、计算机、情报、监视以及侦察和计算机化的武器系统。为了打击他们，北约需要人工网络猎人--专门从事主动网络防御的智能、自主、移动代理。考虑到这一点，北约于2016年启动了RTG IST-152。它的目标一直是通过产生参考体系结构和技术路线图来帮助加速此类软件代理的开发和向实践的过渡。本文提出了一种自主智能网络防御代理的概念和体系结构。我们描述了AICA概念的基本原理，解释了驱动AICA参考体系结构定义的方法和目的，并回顾了AICA的一些主要特征和挑战。



## **8. Sibling-Attack: Rethinking Transferable Adversarial Attacks against Face Recognition**

兄弟攻击：重新思考针对人脸识别的可转移对抗性攻击 cs.CV

8 pages, 5 fivures, accepted by CVPR 2023 as a poster paper

**SubmitDate**: 2023-03-22    [abs](http://arxiv.org/abs/2303.12512v1) [paper-pdf](http://arxiv.org/pdf/2303.12512v1)

**Authors**: Zexin Li, Bangjie Yin, Taiping Yao, Juefeng Guo, Shouhong Ding, Simin Chen, Cong Liu

**Abstract**: A hard challenge in developing practical face recognition (FR) attacks is due to the black-box nature of the target FR model, i.e., inaccessible gradient and parameter information to attackers. While recent research took an important step towards attacking black-box FR models through leveraging transferability, their performance is still limited, especially against online commercial FR systems that can be pessimistic (e.g., a less than 50% ASR--attack success rate on average). Motivated by this, we present Sibling-Attack, a new FR attack technique for the first time explores a novel multi-task perspective (i.e., leveraging extra information from multi-correlated tasks to boost attacking transferability). Intuitively, Sibling-Attack selects a set of tasks correlated with FR and picks the Attribute Recognition (AR) task as the task used in Sibling-Attack based on theoretical and quantitative analysis. Sibling-Attack then develops an optimization framework that fuses adversarial gradient information through (1) constraining the cross-task features to be under the same space, (2) a joint-task meta optimization framework that enhances the gradient compatibility among tasks, and (3) a cross-task gradient stabilization method which mitigates the oscillation effect during attacking. Extensive experiments demonstrate that Sibling-Attack outperforms state-of-the-art FR attack techniques by a non-trivial margin, boosting ASR by 12.61% and 55.77% on average on state-of-the-art pre-trained FR models and two well-known, widely used commercial FR systems.

摘要: 由于目标人脸识别(FR)模型的黑箱性质，即攻击者无法获得梯度和参数信息，因此开发实际的人脸识别(FR)攻击是一个困难的挑战。虽然最近的研究在利用可转移性攻击黑盒FR模型方面迈出了重要的一步，但它们的性能仍然有限，特别是对可能是悲观的在线商业FR系统(例如，平均ASR攻击成功率不到50%)。受此启发，我们首次提出了兄弟攻击，这是一种新的FR攻击技术，它探索了一种新的多任务视角(即，利用来自多相关任务的额外信息来提高攻击的可转移性)。基于理论和定量的分析，兄弟攻击直观地选择了一组与FR相关的任务，并选择了属性识别(AR)任务作为兄弟攻击中使用的任务。兄弟攻击通过(1)将跨任务特征约束在同一空间内；(2)联合任务元优化框架，增强任务间的梯度兼容性；(3)跨任务梯度稳定方法，缓解攻击过程中的振荡效应，从而融合对抗性梯度信息。广泛的实验表明，兄弟攻击的性能远远超过了最先进的FR攻击技术，在最先进的预训练FR模型和两个著名的、广泛使用的商业FR系统上，ASR平均提高了12.61%和55.77%。



## **9. SCRAMBLE-CFI: Mitigating Fault-Induced Control-Flow Attacks on OpenTitan**

SCRIBLE-CFI：缓解OpenTitan上的错误引起的控制流攻击 cs.CR

Accepted at GLSVLS'23

**SubmitDate**: 2023-03-22    [abs](http://arxiv.org/abs/2303.03711v2) [paper-pdf](http://arxiv.org/pdf/2303.03711v2)

**Authors**: Pascal Nasahl, Stefan Mangard

**Abstract**: Secure elements physically exposed to adversaries are frequently targeted by fault attacks. These attacks can be utilized to hijack the control-flow of software allowing the attacker to bypass security measures, extract sensitive data, or gain full code execution. In this paper, we systematically analyze the threat vector of fault-induced control-flow manipulations on the open-source OpenTitan secure element. Our thorough analysis reveals that current countermeasures of this chip either induce large area overheads or still cannot prevent the attacker from exploiting the identified threats. In this context, we introduce SCRAMBLE-CFI, an encryption-based control-flow integrity scheme utilizing existing hardware features of OpenTitan. SCRAMBLE-CFI confines, with minimal hardware overhead, the impact of fault-induced control-flow attacks by encrypting each function with a different encryption tweak at load-time. At runtime, code only can be successfully decrypted when the correct decryption tweak is active. We open-source our hardware changes and release our LLVM toolchain automatically protecting programs. Our analysis shows that SCRAMBLE-CFI complementarily enhances security guarantees of OpenTitan with a negligible hardware overhead of less than 3.97 % and a runtime overhead of 7.02 % for the Embench-IoT benchmarks.

摘要: 物理上暴露在对手面前的安全元素经常成为故障攻击的目标。这些攻击可用于劫持软件的控制流，从而允许攻击者绕过安全措施、提取敏感数据或获得完整的代码执行。在本文中，我们系统地分析了开源OpenTitan安全元素上由错误引起的控制流操作的威胁向量。我们的深入分析表明，目前该芯片的应对措施要么导致大面积开销，要么仍然无法阻止攻击者利用已识别的威胁。在此背景下，我们介绍了一种基于加密的控制流完整性方案SCRIBLE-CFI，该方案利用了OpenTitan现有的硬件特性。置乱-CFI通过在加载时使用不同的加密调整对每个函数进行加密，以最小的硬件开销限制了故障引发的控制流攻击的影响。在运行时，只有当正确的解密调整处于活动状态时，才能成功解密代码。我们将我们的硬件更改开源，并发布我们的LLVM工具链自动保护程序。我们的分析表明，在硬件开销小于3.97%、运行时开销为7.02%的情况下，SCRIBLE-CFI互补地增强了OpenTitan的安全保证。



## **10. Revisiting DeepFool: generalization and improvement**

重温DeepFool：泛化与改进 cs.LG

**SubmitDate**: 2023-03-22    [abs](http://arxiv.org/abs/2303.12481v1) [paper-pdf](http://arxiv.org/pdf/2303.12481v1)

**Authors**: Alireza Abdollahpourrostam, Mahed Abroshan, Seyed-Mohsen Moosavi-Dezfooli

**Abstract**: Deep neural networks have been known to be vulnerable to adversarial examples, which are inputs that are modified slightly to fool the network into making incorrect predictions. This has led to a significant amount of research on evaluating the robustness of these networks against such perturbations. One particularly important robustness metric is the robustness to minimal l2 adversarial perturbations. However, existing methods for evaluating this robustness metric are either computationally expensive or not very accurate. In this paper, we introduce a new family of adversarial attacks that strike a balance between effectiveness and computational efficiency. Our proposed attacks are generalizations of the well-known DeepFool (DF) attack, while they remain simple to understand and implement. We demonstrate that our attacks outperform existing methods in terms of both effectiveness and computational efficiency. Our proposed attacks are also suitable for evaluating the robustness of large models and can be used to perform adversarial training (AT) to achieve state-of-the-art robustness to minimal l2 adversarial perturbations.

摘要: 众所周知，深度神经网络很容易受到敌意例子的攻击，这些例子是对输入进行了轻微修改，以愚弄网络做出错误的预测。这导致了大量关于评估这些网络对此类扰动的稳健性的研究。一个特别重要的稳健性度量是对最小的L2对抗扰动的稳健性。然而，现有的评估这种稳健性度量的方法要么计算昂贵，要么不太准确。在本文中，我们引入了一类新的对抗性攻击，它们在有效性和计算效率之间取得了平衡。我们提出的攻击是众所周知的DeepFool(DF)攻击的推广，但它们仍然易于理解和实现。我们证明了我们的攻击在有效性和计算效率方面都优于现有的方法。我们提出的攻击也适用于评估大型模型的稳健性，并可用于执行对抗训练(AT)以获得对最小L2对抗扰动的最先进的稳健性。



## **11. Distribution-restrained Softmax Loss for the Model Robustness**

用于模型稳健性的分布约束的软最大损失 cs.LG

**SubmitDate**: 2023-03-22    [abs](http://arxiv.org/abs/2303.12363v1) [paper-pdf](http://arxiv.org/pdf/2303.12363v1)

**Authors**: Hao Wang, Chen Li, Jinzhe Jiang, Xin Zhang, Yaqian Zhao, Weifeng Gong

**Abstract**: Recently, the robustness of deep learning models has received widespread attention, and various methods for improving model robustness have been proposed, including adversarial training, model architecture modification, design of loss functions, certified defenses, and so on. However, the principle of the robustness to attacks is still not fully understood, also the related research is still not sufficient. Here, we have identified a significant factor that affects the robustness of models: the distribution characteristics of softmax values for non-real label samples. We found that the results after an attack are highly correlated with the distribution characteristics, and thus we proposed a loss function to suppress the distribution diversity of softmax. A large number of experiments have shown that our method can improve robustness without significant time consumption.

摘要: 近年来，深度学习模型的稳健性受到了广泛的关注，并提出了各种提高模型稳健性的方法，包括对抗性训练、模型结构修改、损失函数设计、认证防御等。然而，对攻击的稳健性原理还没有完全了解，相关的研究也还不够充分。在这里，我们已经确定了一个影响模型稳健性的重要因素：非真实标签样本的Softmax的分布特征。我们发现攻击后的结果与分布特征高度相关，因此我们提出了一种损失函数来抑制Softmax的分布多样性。大量实验表明，该方法可以在不消耗大量时间的情况下提高稳健性。



## **12. Wasserstein Adversarial Examples on Univariant Time Series Data**

单变量时间序列数据的Wasserstein对抗性实例 cs.LG

**SubmitDate**: 2023-03-22    [abs](http://arxiv.org/abs/2303.12357v1) [paper-pdf](http://arxiv.org/pdf/2303.12357v1)

**Authors**: Wenjie Wang, Li Xiong, Jian Lou

**Abstract**: Adversarial examples are crafted by adding indistinguishable perturbations to normal examples in order to fool a well-trained deep learning model to misclassify. In the context of computer vision, this notion of indistinguishability is typically bounded by $L_{\infty}$ or other norms. However, these norms are not appropriate for measuring indistinguishiability for time series data. In this work, we propose adversarial examples in the Wasserstein space for time series data for the first time and utilize Wasserstein distance to bound the perturbation between normal examples and adversarial examples. We introduce Wasserstein projected gradient descent (WPGD), an adversarial attack method for perturbing univariant time series data. We leverage the closed-form solution of Wasserstein distance in the 1D space to calculate the projection step of WPGD efficiently with the gradient descent method. We further propose a two-step projection so that the search of adversarial examples in the Wasserstein space is guided and constrained by Euclidean norms to yield more effective and imperceptible perturbations. We empirically evaluate the proposed attack on several time series datasets in the healthcare domain. Extensive results demonstrate that the Wasserstein attack is powerful and can successfully attack most of the target classifiers with a high attack success rate. To better study the nature of Wasserstein adversarial example, we evaluate a strong defense mechanism named Wasserstein smoothing for potential certified robustness defense. Although the defense can achieve some accuracy gain, it still has limitations in many cases and leaves space for developing a stronger certified robustness method to Wasserstein adversarial examples on univariant time series data.

摘要: 对抗性示例是通过在正常示例中添加无法区分的扰动来制作的，以便愚弄训练有素的深度学习模型进行错误分类。在计算机视觉的背景下，这种不可区分的概念通常受到$L_(\inty)$或其他规范的限制。然而，这些标准不适合用来衡量时间序列数据的不可区分性。在这项工作中，我们首次在时间序列数据的Wasserstein空间中提出了对抗性样本，并利用Wasserstein距离来界定正态样本和对抗性样本之间的扰动。介绍了一种针对单变量时间序列数据的对抗性攻击方法--Wasserstein投影梯度下降(WPGD)方法。利用一维空间中Wasserstein距离的闭合解，利用梯度下降法有效地计算了WPGD的投影步长。我们进一步提出了一个两步投影，使得在Wasserstein空间中的对抗性例子的搜索由欧几里得范数来指导和约束，从而产生更有效和更不可察觉的扰动。我们在医疗保健领域的几个时间序列数据集上对所提出的攻击进行了经验评估。大量实验结果表明，Wasserstein攻击具有较强的攻击能力，能够成功攻击大部分目标分类器，攻击成功率较高。为了更好地研究Wasserstein对抗例子的性质，我们评估了一种名为Wasserstein平滑的强防御机制，以实现潜在的认证稳健性防御。虽然防御方法可以获得一定的精度收益，但在很多情况下仍然存在局限性，并为开发一种对单变量时间序列数据上的Wasserstein对抗性实例具有更强的证明稳健性的方法留下了空间。



## **13. Bankrupting Sybil Despite Churn**

尽管员工流失，Sybil仍在破产 cs.CR

41 pages, 6 figures. arXiv admin note: text overlap with  arXiv:2006.02893, arXiv:1911.06462

**SubmitDate**: 2023-03-22    [abs](http://arxiv.org/abs/2010.06834v4) [paper-pdf](http://arxiv.org/pdf/2010.06834v4)

**Authors**: Diksha Gupta, Jared Saia, Maxwell Young

**Abstract**: A Sybil attack occurs when an adversary controls multiple identifiers (IDs) in a system. Limiting the number of Sybil (bad) IDs to a minority is critical to the use of well-established tools for tolerating malicious behavior, such as Byzantine agreement and secure multiparty computation.   A popular technique for enforcing a Sybil minority is resource burning: the verifiable consumption of a network resource, such as computational power, bandwidth, or memory. Unfortunately, typical defenses based on resource burning require non-Sybil (good) IDs to consume at least as many resources as the adversary. Additionally, they have a high resource burning cost, even when the system membership is relatively stable.   Here, we present a new Sybil defense, ERGO, that guarantees (1) there is always a minority of bad IDs; and (2) when the system is under significant attack, the good IDs consume asymptotically less resources than the bad. In particular, for churn rate that can vary exponentially, the resource burning rate for good IDs under ERGO is O(\sqrt{TJ} + J), where T is the resource burning rate of the adversary, and J is the join rate of good IDs. We show this resource burning rate is asymptotically optimal for a large class of algorithms.   We empirically evaluate ERGO alongside prior Sybil defenses. Additionally, we show that ERGO can be combined with machine learning techniques for classifying Sybil IDs, while preserving its theoretical guarantees. Based on our experiments comparing ERGO with two previous Sybil defenses, ERGO improves on the amount of resource burning relative to the adversary by up to 2 orders of magnitude without machine learning, and up to 3 orders of magnitude using machine learning.

摘要: 当对手控制系统中的多个标识符(ID)时，就会发生Sybil攻击。将Sybil(BAD)ID的数量限制为少数，对于使用成熟的工具容忍恶意行为至关重要，例如拜占庭协议和安全多方计算。实施少数Sybil的一种流行技术是资源烧毁：可验证的网络资源消耗，如计算能力、带宽或内存。不幸的是，基于资源燃烧的典型防御需要非Sybil(好)ID至少消耗与对手一样多的资源。此外，即使在系统成员相对稳定的情况下，它们也具有较高的资源消耗成本。在这里，我们提出了一种新的Sybil防御方案ERGO，它保证(1)总是有少数坏ID；(2)当系统受到重大攻击时，好ID消耗的资源逐渐少于坏ID。特别地，对于可以指数变化的流失率，ERGO下好的ID的资源烧失率为O(Sqrt{tj}+J)，其中T是对手的资源烧失率，J是好的ID的加入率。对于一大类算法，我们证明了这种资源消耗速度是渐近最优的。我们对ERGO和之前的Sybil防御进行了经验评估。此外，我们证明了ERGO可以与机器学习技术相结合来对Sybil ID进行分类，同时保持其理论保证。基于我们的实验比较了ERGO和之前的两个Sybil防御措施，ERGO在没有机器学习的情况下相对于对手提高了高达2个数量级的资源消耗量，使用机器学习的资源消耗量提高了高达3个数量级。



## **14. X-CANIDS: Signal-Aware Explainable Intrusion Detection System for Controller Area Network-Based In-Vehicle Network**

X-CANIDS：基于控制器局域网的车载网络信号感知可解释入侵检测系统 cs.CR

**SubmitDate**: 2023-03-22    [abs](http://arxiv.org/abs/2303.12278v1) [paper-pdf](http://arxiv.org/pdf/2303.12278v1)

**Authors**: Seonghoon Jeong, Sangho Lee, Hwejae Lee, Huy Kang Kim

**Abstract**: Controller Area Network (CAN) is an essential networking protocol that connects multiple electronic control units (ECUs) in a vehicle. However, CAN-based in-vehicle networks (IVNs) face security risks owing to the CAN mechanisms. An adversary can sabotage a vehicle by leveraging the security risks if they can access the CAN bus. Thus, recent actions and cybersecurity regulations (e.g., UNR 155) require carmakers to implement intrusion detection systems (IDSs) in their vehicles. An IDS should detect cyberattacks and provide a forensic capability to analyze attacks. Although many IDSs have been proposed, considerations regarding their feasibility and explainability remain lacking. This study proposes X-CANIDS, which is a novel IDS for CAN-based IVNs. X-CANIDS dissects the payloads in CAN messages into human-understandable signals using a CAN database. The signals improve the intrusion detection performance compared with the use of bit representations of raw payloads. These signals also enable an understanding of which signal or ECU is under attack. X-CANIDS can detect zero-day attacks because it does not require any labeled dataset in the training phase. We confirmed the feasibility of the proposed method through a benchmark test on an automotive-grade embedded device with a GPU. The results of this work will be valuable to carmakers and researchers considering the installation of in-vehicle IDSs for their vehicles.

摘要: 控制器局域网(CAN)是连接车辆中多个电子控制单元(ECU)的基本网络协议。然而，由于CAN机制的存在，基于CAN的车载网络面临着安全隐患。如果对手可以访问CAN总线，则他们可以利用安全风险来破坏车辆。因此，最近的行动和网络安全法规(例如，UNR 155)要求汽车制造商在其车辆中安装入侵检测系统(IDS)。入侵检测系统应检测网络攻击并提供分析攻击的取证能力。虽然已经提出了许多入侵检测系统，但仍然缺乏对其可行性和可解释性的考虑。本研究提出了一种新型的基于CAN网络的入侵检测系统--X-CANID。X-Canids使用CAN数据库将CAN消息中的有效载荷分解为人类可理解的信号。与使用原始有效载荷的比特表示相比，该信号提高了入侵检测性能。这些信号还使您能够了解哪个信号或ECU受到攻击。X-CARID可以检测零日攻击，因为它在训练阶段不需要任何标记的数据集。通过在一款搭载GPU的车载嵌入式设备上的基准测试，验证了该方法的可行性。这项工作的结果将对汽车制造商和考虑为其车辆安装车载入侵检测系统的研究人员具有价值。



## **15. State-of-the-art optical-based physical adversarial attacks for deep learning computer vision systems**

深度学习计算机视觉系统中基于光学的物理对抗攻击 cs.CV

This work has been submitted to the IEEE for possible publication.  Copyright may be transferred without notice, after which this version may no  longer be accessible

**SubmitDate**: 2023-03-22    [abs](http://arxiv.org/abs/2303.12249v1) [paper-pdf](http://arxiv.org/pdf/2303.12249v1)

**Authors**: Junbin Fang, You Jiang, Canjian Jiang, Zoe L. Jiang, Siu-Ming Yiu, Chuanyi Liu

**Abstract**: Adversarial attacks can mislead deep learning models to make false predictions by implanting small perturbations to the original input that are imperceptible to the human eye, which poses a huge security threat to the computer vision systems based on deep learning. Physical adversarial attacks, which is more realistic, as the perturbation is introduced to the input before it is being captured and converted to a binary image inside the vision system, when compared to digital adversarial attacks. In this paper, we focus on physical adversarial attacks and further classify them into invasive and non-invasive. Optical-based physical adversarial attack techniques (e.g. using light irradiation) belong to the non-invasive category. As the perturbations can be easily ignored by humans as the perturbations are very similar to the effects generated by a natural environment in the real world. They are highly invisibility and executable and can pose a significant or even lethal threats to real systems. This paper focuses on optical-based physical adversarial attack techniques for computer vision systems, with emphasis on the introduction and discussion of optical-based physical adversarial attack techniques.

摘要: 对抗性攻击通过在原始输入中植入人眼无法察觉的微小扰动来误导深度学习模型做出错误预测，这对基于深度学习的计算机视觉系统构成了巨大的安全威胁。物理对抗性攻击，与数字对抗性攻击相比，这更现实，因为与数字对抗性攻击相比，输入在被捕获并转换为视觉系统内的二进制图像之前被引入扰动。在本文中，我们将重点放在物理对抗攻击上，并进一步将其分为侵入性攻击和非侵入式攻击。基于光学的物理对抗攻击技术(例如使用光照射)属于非侵入性范畴。因为扰动很容易被人类忽略，因为扰动与现实世界中的自然环境产生的效果非常相似。它们具有高度的隐蔽性和可执行性，可以对真实系统构成重大甚至致命的威胁。本文研究了计算机视觉系统中基于光学的物理对抗攻击技术，重点介绍和讨论了基于光学的物理对抗攻击技术。



## **16. Task-Oriented Communications for NextG: End-to-End Deep Learning and AI Security Aspects**

面向NextG的面向任务的通信：端到端深度学习和AI安全方面 cs.NI

**SubmitDate**: 2023-03-21    [abs](http://arxiv.org/abs/2212.09668v2) [paper-pdf](http://arxiv.org/pdf/2212.09668v2)

**Authors**: Yalin E. Sagduyu, Sennur Ulukus, Aylin Yener

**Abstract**: Communications systems to date are primarily designed with the goal of reliable transfer of digital sequences (bits). Next generation (NextG) communication systems are beginning to explore shifting this design paradigm to reliably executing a given task such as in task-oriented communications. In this paper, wireless signal classification is considered as the task for the NextG Radio Access Network (RAN), where edge devices collect wireless signals for spectrum awareness and communicate with the NextG base station (gNodeB) that needs to identify the signal label. Edge devices may not have sufficient processing power and may not be trusted to perform the signal classification task, whereas the transfer of signals to the gNodeB may not be feasible due to stringent delay, rate, and energy restrictions. Task-oriented communications is considered by jointly training the transmitter, receiver and classifier functionalities as an encoder-decoder pair for the edge device and the gNodeB. This approach improves the accuracy compared to the separated case of signal transfer followed by classification. Adversarial machine learning poses a major security threat to the use of deep learning for task-oriented communications. A major performance loss is shown when backdoor (Trojan) and adversarial (evasion) attacks target the training and test processes of task-oriented communications.

摘要: 迄今为止，通信系统的主要设计目标是可靠地传输数字序列(比特)。下一代(NextG)通信系统正开始探索将这种设计范例转变为可靠地执行给定任务，例如在面向任务的通信中。本文将无线信号分类作为下一代无线接入网(RAN)的任务，边缘设备采集无线信号以实现频谱感知，并与需要识别信号标签的下一代基站(GNodeB)进行通信。边缘设备可能没有足够的处理能力，并且可能不被信任来执行信号分类任务，而由于严格的延迟、速率和能量限制，向gNodeB传输信号可能是不可行的。通过将发送器、接收器和分类器功能联合训练为用于边缘设备和gNodeB的编解码器对来考虑面向任务的通信。与信号传输后分类的分离情况相比，该方法提高了精度。对抗性机器学习对使用深度学习进行面向任务的通信构成了主要的安全威胁。当后门(特洛伊木马)和敌意(规避)攻击以面向任务的通信的训练和测试过程为目标时，会显示出重大的性能损失。



## **17. Towards Compositional Adversarial Robustness: Generalizing Adversarial Training to Composite Semantic Perturbations**

走向成分对抗稳健性：将对抗训练推广到复合语义扰动 cs.CV

CVPR 2023. The research demo is at https://hsiung.cc/CARBEN/

**SubmitDate**: 2023-03-21    [abs](http://arxiv.org/abs/2202.04235v3) [paper-pdf](http://arxiv.org/pdf/2202.04235v3)

**Authors**: Lei Hsiung, Yun-Yun Tsai, Pin-Yu Chen, Tsung-Yi Ho

**Abstract**: Model robustness against adversarial examples of single perturbation type such as the $\ell_{p}$-norm has been widely studied, yet its generalization to more realistic scenarios involving multiple semantic perturbations and their composition remains largely unexplored. In this paper, we first propose a novel method for generating composite adversarial examples. Our method can find the optimal attack composition by utilizing component-wise projected gradient descent and automatic attack-order scheduling. We then propose generalized adversarial training (GAT) to extend model robustness from $\ell_{p}$-ball to composite semantic perturbations, such as the combination of Hue, Saturation, Brightness, Contrast, and Rotation. Results obtained using ImageNet and CIFAR-10 datasets indicate that GAT can be robust not only to all the tested types of a single attack, but also to any combination of such attacks. GAT also outperforms baseline $\ell_{\infty}$-norm bounded adversarial training approaches by a significant margin.

摘要: 针对单一扰动类型的对抗性实例，如$ellp-范数，模型的稳健性已经得到了广泛的研究，但它对涉及多个语义扰动及其组成的更现实场景的推广仍在很大程度上有待探索。在本文中，我们首先提出了一种生成复合对抗性实例的新方法。该方法利用基于组件的投影梯度下降和自动攻击顺序调度来寻找最优的攻击组合。然后，我们提出了广义对抗性训练(GAT)来扩展模型的稳健性，将模型的稳健性从球状扩展到复合语义扰动，如色调、饱和度、亮度、对比度和旋转的组合。使用ImageNet和CIFAR-10数据集获得的结果表明，GAT不仅对所有测试类型的单一攻击，而且对此类攻击的任何组合都具有健壮性。GAT的性能也大大超过了基准范数有界的对抗性训练方法。



## **18. Efficient Decision-based Black-box Patch Attacks on Video Recognition**

视频识别中高效的基于决策的黑盒补丁攻击 cs.CV

**SubmitDate**: 2023-03-21    [abs](http://arxiv.org/abs/2303.11917v1) [paper-pdf](http://arxiv.org/pdf/2303.11917v1)

**Authors**: Kaixun Jiang, Zhaoyu Chen, Tony Huang, Jiafeng Wang, Dingkang Yang, Bo Li, Yan Wang, Wenqiang Zhang

**Abstract**: Although Deep Neural Networks (DNNs) have demonstrated excellent performance, they are vulnerable to adversarial patches that introduce perceptible and localized perturbations to the input. Generating adversarial patches on images has received much attention, while adversarial patches on videos have not been well investigated. Further, decision-based attacks, where attackers only access the predicted hard labels by querying threat models, have not been well explored on video models either, even if they are practical in real-world video recognition scenes. The absence of such studies leads to a huge gap in the robustness assessment for video models. To bridge this gap, this work first explores decision-based patch attacks on video models. We analyze that the huge parameter space brought by videos and the minimal information returned by decision-based models both greatly increase the attack difficulty and query burden. To achieve a query-efficient attack, we propose a spatial-temporal differential evolution (STDE) framework. First, STDE introduces target videos as patch textures and only adds patches on keyframes that are adaptively selected by temporal difference. Second, STDE takes minimizing the patch area as the optimization objective and adopts spatialtemporal mutation and crossover to search for the global optimum without falling into the local optimum. Experiments show STDE has demonstrated state-of-the-art performance in terms of threat, efficiency and imperceptibility. Hence, STDE has the potential to be a powerful tool for evaluating the robustness of video recognition models.

摘要: 尽管深度神经网络(DNN)表现出了很好的性能，但它们很容易受到敌意补丁的攻击，这些补丁会给输入带来可感知的局部扰动。在图像上生成敌意补丁已经得到了很大的关注，而视频上的敌意补丁还没有得到很好的研究。此外，基于决策的攻击(攻击者仅通过查询威胁模型来访问预测的硬标签)在视频模型上也没有得到很好的探索，即使它们在现实世界的视频识别场景中是实用的。这类研究的缺乏导致了视频模型稳健性评估的巨大差距。为了弥补这一差距，这项工作首先探索了基于决策的视频模型补丁攻击。分析了视频带来的巨大参数空间和基于决策的模型返回的最小信息量都大大增加了攻击难度和查询负担。为了实现查询高效的攻击，我们提出了一种时空差异进化(STDE)框架。首先，STDE将目标视频作为补丁纹理引入，只在根据时间差异自适应选择的关键帧上添加补丁。其次，STDE算法以面片面积最小为优化目标，采用时空变异和交叉来搜索全局最优解而不陷入局部最优。实验表明，STDE在威胁、效率和不可感知性方面都表现出了最先进的性能。因此，STDE有可能成为评估视频识别模型稳健性的有力工具。



## **19. The Threat of Adversarial Attacks on Machine Learning in Network Security -- A Survey**

网络安全中对抗性攻击对机器学习的威胁--综述 cs.CR

**SubmitDate**: 2023-03-21    [abs](http://arxiv.org/abs/1911.02621v3) [paper-pdf](http://arxiv.org/pdf/1911.02621v3)

**Authors**: Olakunle Ibitoye, Rana Abou-Khamis, Mohamed el Shehaby, Ashraf Matrawy, M. Omair Shafiq

**Abstract**: Machine learning models have made many decision support systems to be faster, more accurate, and more efficient. However, applications of machine learning in network security face a more disproportionate threat of active adversarial attacks compared to other domains. This is because machine learning applications in network security such as malware detection, intrusion detection, and spam filtering are by themselves adversarial in nature. In what could be considered an arm's race between attackers and defenders, adversaries constantly probe machine learning systems with inputs that are explicitly designed to bypass the system and induce a wrong prediction. In this survey, we first provide a taxonomy of machine learning techniques, tasks, and depth. We then introduce a classification of machine learning in network security applications. Next, we examine various adversarial attacks against machine learning in network security and introduce two classification approaches for adversarial attacks in network security. First, we classify adversarial attacks in network security based on a taxonomy of network security applications. Secondly, we categorize adversarial attacks in network security into a problem space vs feature space dimensional classification model. We then analyze the various defenses against adversarial attacks on machine learning-based network security applications. We conclude by introducing an adversarial risk grid map and evaluating several existing adversarial attacks against machine learning in network security using the risk grid map. We also identify where each attack classification resides within the adversarial risk grid map.

摘要: 机器学习模型使许多决策支持系统变得更快、更准确、更高效。然而，与其他领域相比，机器学习在网络安全中的应用面临着更不成比例的主动对抗攻击威胁。这是因为网络安全中的机器学习应用程序，如恶意软件检测、入侵检测和垃圾邮件过滤，本身就是对抗性的。在这场可以被视为攻击者和防御者之间的军备竞赛中，对手不断地探查机器学习系统，其输入显然是为了绕过系统，并导致错误的预测。在这次调查中，我们首先提供了机器学习技术、任务和深度的分类。然后介绍了机器学习在网络安全应用中的分类。接下来，我们考察了网络安全中各种针对机器学习的对抗性攻击，并介绍了网络安全中对抗性攻击的两种分类方法。首先，根据网络安全应用的分类，对网络安全中的敌意攻击进行分类。其次，将网络安全中的对抗性攻击归类到问题空间与特征空间的维度分类模型中。然后，我们分析了各种针对基于机器学习的网络安全应用的对抗性攻击的防御措施。最后，我们引入了对抗性风险网格图，并利用该风险网格图对网络安全中现有的几种针对机器学习的对抗性攻击进行了评估。我们还确定了每个攻击分类在对抗性风险网格地图中的位置。



## **20. OTJR: Optimal Transport Meets Optimal Jacobian Regularization for Adversarial Robustness**

OTJR：最优传输满足最优雅可比正则化的对抗性 cs.CV

**SubmitDate**: 2023-03-21    [abs](http://arxiv.org/abs/2303.11793v1) [paper-pdf](http://arxiv.org/pdf/2303.11793v1)

**Authors**: Binh M. Le, Shahroz Tariq, Simon S. Woo

**Abstract**: Deep neural networks are widely recognized as being vulnerable to adversarial perturbation. To overcome this challenge, developing a robust classifier is crucial. So far, two well-known defenses have been adopted to improve the learning of robust classifiers, namely adversarial training (AT) and Jacobian regularization. However, each approach behaves differently against adversarial perturbations. First, our work carefully analyzes and characterizes these two schools of approaches, both theoretically and empirically, to demonstrate how each approach impacts the robust learning of a classifier. Next, we propose our novel Optimal Transport with Jacobian regularization method, dubbed OTJR, jointly incorporating the input-output Jacobian regularization into the AT by leveraging the optimal transport theory. In particular, we employ the Sliced Wasserstein (SW) distance that can efficiently push the adversarial samples' representations closer to those of clean samples, regardless of the number of classes within the dataset. The SW distance provides the adversarial samples' movement directions, which are much more informative and powerful for the Jacobian regularization. Our extensive experiments demonstrate the effectiveness of our proposed method, which jointly incorporates Jacobian regularization into AT. Furthermore, we demonstrate that our proposed method consistently enhances the model's robustness with CIFAR-100 dataset under various adversarial attack settings, achieving up to 28.49% under AutoAttack.

摘要: 深度神经网络被广泛认为容易受到对抗性扰动的影响。要克服这一挑战，开发一个健壮的分类器至关重要。到目前为止，已经采用了两种著名的防御措施来改进稳健分类器的学习，即对抗性训练(AT)和雅可比正则化。然而，每种方法在对抗对抗性干扰时的表现都不同。首先，我们的工作仔细地分析和表征了这两个流派的方法，无论是理论上还是经验上，以证明每种方法如何影响分类器的稳健学习。接下来，我们利用最优传输理论，将输入输出的雅可比正则化引入到AT中，提出了一种新的基于雅可比正则化的最优传输方法，称为OTJR。特别是，我们使用了切片Wasserstein(SW)距离，该距离可以有效地将对抗性样本的表示更接近于干净样本的表示，而不管数据集中有多少类。Sw距离提供了对抗性样本的运动方向，为雅可比正则化提供了更多的信息和更强大的能力。我们的大量实验证明了我们提出的方法的有效性，该方法将雅可比正则化联合到AT中。此外，我们还利用CIFAR-100数据集在不同的对抗性攻击环境下验证了该方法的有效性，在AutoAttack环境下达到了28.49%的健壮性。



## **21. Generative AI for Cyber Threat-Hunting in 6G-enabled IoT Networks**

产生式人工智能在支持6G的物联网网络威胁搜索中的应用 cs.CR

The paper is accepted and will be published in the IEEE/ACM CCGrid  2023 Conference Proceedings

**SubmitDate**: 2023-03-21    [abs](http://arxiv.org/abs/2303.11751v1) [paper-pdf](http://arxiv.org/pdf/2303.11751v1)

**Authors**: Mohamed Amine Ferrag, Merouane Debbah, Muna Al-Hawawreh

**Abstract**: The next generation of cellular technology, 6G, is being developed to enable a wide range of new applications and services for the Internet of Things (IoT). One of 6G's main advantages for IoT applications is its ability to support much higher data rates and bandwidth as well as to support ultra-low latency. However, with this increased connectivity will come to an increased risk of cyber threats, as attackers will be able to exploit the large network of connected devices. Generative Artificial Intelligence (AI) can be used to detect and prevent cyber attacks by continuously learning and adapting to new threats and vulnerabilities. In this paper, we discuss the use of generative AI for cyber threat-hunting (CTH) in 6G-enabled IoT networks. Then, we propose a new generative adversarial network (GAN) and Transformer-based model for CTH in 6G-enabled IoT Networks. The experimental analysis results with a new cyber security dataset demonstrate that the Transformer-based security model for CTH can detect IoT attacks with a high overall accuracy of 95%. We examine the challenges and opportunities and conclude by highlighting the potential of generative AI in enhancing the security of 6G-enabled IoT networks and call for further research to be conducted in this area.

摘要: 下一代蜂窝技术6G正在开发中，以支持物联网(IoT)的广泛新应用和服务。6G对物联网应用的主要优势之一是它能够支持更高的数据速率和带宽以及支持超低延迟。然而，随着这种连接的增加，网络威胁的风险将会增加，因为攻击者将能够利用连接设备组成的大型网络。生成性人工智能(AI)可以通过不断学习和适应新的威胁和漏洞来检测和预防网络攻击。在本文中，我们讨论了产生式人工智能在支持6G的物联网网络中用于网络威胁搜索(CTH)。在此基础上，我们提出了一种新的基于生成性对抗网络(GAN)和Transformer的6G物联网Cth模型。在一个新的网络安全数据集上的实验分析结果表明，基于Transformer的Cth安全模型能够检测到物联网攻击，总体准确率达到95%。我们研究了挑战和机遇，最后强调了生成性人工智能在增强支持6G的物联网网络安全方面的潜力，并呼吁在这一领域进行进一步研究。



## **22. Poisoning Attacks in Federated Edge Learning for Digital Twin 6G-enabled IoTs: An Anticipatory Study**

数字双生6G物联网联合边缘学习中的中毒攻击：一项预期研究 cs.CR

The paper is accepted and will be published in the IEEE ICC 2023  Conference Proceedings

**SubmitDate**: 2023-03-21    [abs](http://arxiv.org/abs/2303.11745v1) [paper-pdf](http://arxiv.org/pdf/2303.11745v1)

**Authors**: Mohamed Amine Ferrag, Burak Kantarci, Lucas C. Cordeiro, Merouane Debbah, Kim-Kwang Raymond Choo

**Abstract**: Federated edge learning can be essential in supporting privacy-preserving, artificial intelligence (AI)-enabled activities in digital twin 6G-enabled Internet of Things (IoT) environments. However, we need to also consider the potential of attacks targeting the underlying AI systems (e.g., adversaries seek to corrupt data on the IoT devices during local updates or corrupt the model updates); hence, in this article, we propose an anticipatory study for poisoning attacks in federated edge learning for digital twin 6G-enabled IoT environments. Specifically, we study the influence of adversaries on the training and development of federated learning models in digital twin 6G-enabled IoT environments. We demonstrate that attackers can carry out poisoning attacks in two different learning settings, namely: centralized learning and federated learning, and successful attacks can severely reduce the model's accuracy. We comprehensively evaluate the attacks on a new cyber security dataset designed for IoT applications with three deep neural networks under the non-independent and identically distributed (Non-IID) data and the independent and identically distributed (IID) data. The poisoning attacks, on an attack classification problem, can lead to a decrease in accuracy from 94.93% to 85.98% with IID data and from 94.18% to 30.04% with Non-IID.

摘要: 联合边缘学习对于支持数字孪生6G物联网(IoT)环境中保护隐私、启用人工智能(AI)的活动至关重要。然而，我们还需要考虑针对底层AI系统的攻击的可能性(例如，对手试图在本地更新期间破坏物联网设备上的数据或破坏模型更新)；因此，在本文中，我们提出了一项针对数字孪生6G物联网环境中联合边缘学习中的中毒攻击的前瞻性研究。具体地说，我们研究了在支持6G的数字孪生物联网环境中，对手对联合学习模型的训练和发展的影响。我们证明了攻击者可以在集中式学习和联合学习两种不同的学习环境下进行中毒攻击，而成功的攻击会严重降低模型的准确性。在非独立同分布(Non-IID)数据和独立同分布(IID)数据下，使用三种深度神经网络综合评估了针对物联网应用设计的新的网络安全数据集的攻击。对于一个攻击分类问题，中毒攻击可以导致IID数据的准确率从94.93%下降到85.98%，而非IID数据的准确率从94.18%下降到30.04%。



## **23. Manipulating Transfer Learning for Property Inference**

操纵性迁移学习用于性质推理 cs.LG

Accepted to CVPR 2023

**SubmitDate**: 2023-03-21    [abs](http://arxiv.org/abs/2303.11643v1) [paper-pdf](http://arxiv.org/pdf/2303.11643v1)

**Authors**: Yulong Tian, Fnu Suya, Anshuman Suri, Fengyuan Xu, David Evans

**Abstract**: Transfer learning is a popular method for tuning pretrained (upstream) models for different downstream tasks using limited data and computational resources. We study how an adversary with control over an upstream model used in transfer learning can conduct property inference attacks on a victim's tuned downstream model. For example, to infer the presence of images of a specific individual in the downstream training set. We demonstrate attacks in which an adversary can manipulate the upstream model to conduct highly effective and specific property inference attacks (AUC score $> 0.9$), without incurring significant performance loss on the main task. The main idea of the manipulation is to make the upstream model generate activations (intermediate features) with different distributions for samples with and without a target property, thus enabling the adversary to distinguish easily between downstream models trained with and without training examples that have the target property. Our code is available at https://github.com/yulongt23/Transfer-Inference.

摘要: 转移学习是一种流行的方法，用于使用有限的数据和计算资源为不同的下游任务调整预先训练的(上游)模型。我们研究了在迁移学习中控制上游模型的对手如何对受害者调整后的下游模型进行属性推理攻击。例如，推断下游训练集中特定个体的图像的存在。我们演示了这样的攻击：攻击者可以操纵上游模型来进行高效和特定的属性推理攻击(AUC分数$>0.9$)，而不会在主任务上造成显著的性能损失。该操纵的主要思想是使上游模型为具有和不具有目标属性的样本生成具有不同分布的激活(中间特征)，从而使对手能够容易地区分具有和不具有目标属性的训练样本训练的下游模型。我们的代码可以在https://github.com/yulongt23/Transfer-Inference.上找到



## **24. An Observer-based Switching Algorithm for Safety under Sensor Denial-of-Service Attacks**

传感器拒绝服务攻击下基于观察者的安全切换算法 eess.SY

Accepted at the 2023 American Control Conference (ACC)

**SubmitDate**: 2023-03-21    [abs](http://arxiv.org/abs/2303.11640v1) [paper-pdf](http://arxiv.org/pdf/2303.11640v1)

**Authors**: Santiago Jimenez Leudo, Kunal Garg, Ricardo G. Sanfelice, Alvaro A. Cardenas

**Abstract**: The design of safe-critical control algorithms for systems under Denial-of-Service (DoS) attacks on the system output is studied in this work. We aim to address scenarios where attack-mitigation approaches are not feasible, and the system needs to maintain safety under adversarial attacks. We propose an attack-recovery strategy by designing a switching observer and characterizing bounds in the error of a state estimation scheme by specifying tolerable limits on the time length of attacks. Then, we propose a switching control algorithm that renders forward invariant a set for the observer. Thus, by satisfying the error bounds of the state estimation, we guarantee that the safe set is rendered conditionally invariant with respect to a set of initial conditions. A numerical example illustrates the efficacy of the approach.

摘要: 本文研究了在拒绝服务(DoS)攻击下系统输出的安全关键控制算法的设计。我们的目标是解决攻击缓解方法不可行的情况，并且系统需要在对抗性攻击下保持安全。提出了一种攻击恢复策略，该策略通过设计切换观测器和通过指定攻击时间长度的可容忍限度来刻画状态估计方案的误差界。然后，我们提出了一种切换控制算法，它使观测器的前向不变量成为一个集合。因此，通过满足状态估计的误差界，我们保证安全集关于一组初始条件是条件不变的。数值算例说明了该方法的有效性。



## **25. Enhancing the Self-Universality for Transferable Targeted Attacks**

增强可转移定向攻击的自我普适性 cs.CV

**SubmitDate**: 2023-03-21    [abs](http://arxiv.org/abs/2209.03716v2) [paper-pdf](http://arxiv.org/pdf/2209.03716v2)

**Authors**: Zhipeng Wei, Jingjing Chen, Zuxuan Wu, Yu-Gang Jiang

**Abstract**: In this paper, we propose a novel transfer-based targeted attack method that optimizes the adversarial perturbations without any extra training efforts for auxiliary networks on training data. Our new attack method is proposed based on the observation that highly universal adversarial perturbations tend to be more transferable for targeted attacks. Therefore, we propose to make the perturbation to be agnostic to different local regions within one image, which we called as self-universality. Instead of optimizing the perturbations on different images, optimizing on different regions to achieve self-universality can get rid of using extra data. Specifically, we introduce a feature similarity loss that encourages the learned perturbations to be universal by maximizing the feature similarity between adversarial perturbed global images and randomly cropped local regions. With the feature similarity loss, our method makes the features from adversarial perturbations to be more dominant than that of benign images, hence improving targeted transferability. We name the proposed attack method as Self-Universality (SU) attack. Extensive experiments demonstrate that SU can achieve high success rates for transfer-based targeted attacks. On ImageNet-compatible dataset, SU yields an improvement of 12\% compared with existing state-of-the-art methods. Code is available at https://github.com/zhipeng-wei/Self-Universality.

摘要: 在本文中，我们提出了一种新的基于转移的定向攻击方法，该方法在不需要对训练数据进行任何额外训练的情况下优化了对抗性扰动。我们提出的新攻击方法是基于这样的观察，即高度普遍的对抗性扰动倾向于更可转移到定向攻击。因此，我们提出使微扰对同一图像内的不同局部区域是不可知的，我们称之为自普适性。与对不同图像上的扰动进行优化不同，通过对不同区域进行优化来实现自普适性，可以避免使用额外的数据。具体地说，我们引入了特征相似度损失，通过最大化对抗性扰动的全局图像和随机裁剪的局部区域之间的特征相似度，鼓励学习的扰动具有普遍性。在特征相似度损失的情况下，我们的方法使得来自对抗性扰动的特征比来自良性图像的特征更具优势，从而提高了目标可转移性。我们将所提出的攻击方法命名为自普适性攻击。广泛的实验证明，宿灿对基于转会的靶向攻击取得了很高的成功率。在与ImageNet兼容的数据集上，与现有的最先进方法相比，SU方法的性能提高了12%。代码可在https://github.com/zhipeng-wei/Self-Universality.上找到



## **26. Semi-supervised Semantics-guided Adversarial Training for Trajectory Prediction**

用于弹道预测的半监督语义制导对抗性训练 cs.LG

11 pages, adversarial training for trajectory prediction

**SubmitDate**: 2023-03-21    [abs](http://arxiv.org/abs/2205.14230v2) [paper-pdf](http://arxiv.org/pdf/2205.14230v2)

**Authors**: Ruochen Jiao, Xiangguo Liu, Takami Sato, Qi Alfred Chen, Qi Zhu

**Abstract**: Predicting the trajectories of surrounding objects is a critical task for self-driving vehicles and many other autonomous systems. Recent works demonstrate that adversarial attacks on trajectory prediction, where small crafted perturbations are introduced to history trajectories, may significantly mislead the prediction of future trajectories and induce unsafe planning. However, few works have addressed enhancing the robustness of this important safety-critical task.In this paper, we present a novel adversarial training method for trajectory prediction. Compared with typical adversarial training on image tasks, our work is challenged by more random input with rich context and a lack of class labels. To address these challenges, we propose a method based on a semi-supervised adversarial autoencoder, which models disentangled semantic features with domain knowledge and provides additional latent labels for the adversarial training. Extensive experiments with different types of attacks demonstrate that our Semisupervised Semantics-guided Adversarial Training (SSAT) method can effectively mitigate the impact of adversarial attacks by up to 73% and outperform other popular defense methods. In addition, experiments show that our method can significantly improve the system's robust generalization to unseen patterns of attacks. We believe that such semantics-guided architecture and advancement on robust generalization is an important step for developing robust prediction models and enabling safe decision-making.

摘要: 预测周围物体的轨迹是自动驾驶车辆和许多其他自动驾驶系统的关键任务。最近的工作表明，对轨迹预测的敌意攻击，即在历史轨迹中引入微小的精心设计的扰动，可能会严重误导对未来轨迹的预测，并导致不安全的规划。然而，很少有人研究如何提高这一重要的安全关键任务的健壮性。在本文中，我们提出了一种新的用于轨迹预测的对抗性训练方法。与典型的对抗性图像训练相比，我们的工作面临着更多的随机输入和丰富的上下文以及缺乏类别标签的挑战。为了应对这些挑战，我们提出了一种基于半监督对抗性自动编码器的方法，该方法利用领域知识对解开的语义特征进行建模，并为对抗性训练提供额外的潜在标签。对不同类型攻击的大量实验表明，我们的半监督语义制导的对抗性训练(SSAT)方法可以有效地减轻对抗性攻击的影响高达73%，并优于其他流行的防御方法。此外，实验表明，该方法能够显著提高系统对未知攻击模式的鲁棒性。我们认为，这种语义制导的架构和在健壮泛化方面的进展是开发健壮预测模型和实现安全决策的重要一步。



## **27. STDLens: Model Hijacking-resilient Federated Learning for Object Detection**

STDLens：用于目标检测的模型劫持-弹性联合学习 cs.CR

CVPR 2023. Source Code: https://github.com/git-disl/STDLens

**SubmitDate**: 2023-03-21    [abs](http://arxiv.org/abs/2303.11511v1) [paper-pdf](http://arxiv.org/pdf/2303.11511v1)

**Authors**: Ka-Ho Chow, Ling Liu, Wenqi Wei, Fatih Ilhan, Yanzhao Wu

**Abstract**: Federated Learning (FL) has been gaining popularity as a collaborative learning framework to train deep learning-based object detection models over a distributed population of clients. Despite its advantages, FL is vulnerable to model hijacking. The attacker can control how the object detection system should misbehave by implanting Trojaned gradients using only a small number of compromised clients in the collaborative learning process. This paper introduces STDLens, a principled approach to safeguarding FL against such attacks. We first investigate existing mitigation mechanisms and analyze their failures caused by the inherent errors in spatial clustering analysis on gradients. Based on the insights, we introduce a three-tier forensic framework to identify and expel Trojaned gradients and reclaim the performance over the course of FL. We consider three types of adaptive attacks and demonstrate the robustness of STDLens against advanced adversaries. Extensive experiments show that STDLens can protect FL against different model hijacking attacks and outperform existing methods in identifying and removing Trojaned gradients with significantly higher precision and much lower false-positive rates.

摘要: 联邦学习(FL)作为一种协作学习框架，在分布的客户群上训练基于深度学习的目标检测模型，已经越来越受欢迎。尽管有优势，但FL很容易受到模特劫持的攻击。攻击者可以通过在协作学习过程中仅使用少量受攻击的客户端植入特洛伊木马梯度来控制对象检测系统的不当行为。本文介绍了STDLens，一种保护FL免受此类攻击的原则性方法。我们首先调查了现有的缓解机制，并分析了它们由于梯度空间聚类分析的固有错误而导致的失败。基于这些见解，我们引入了一个三层取证框架来识别和排除特洛伊木马的梯度，并在FL过程中恢复性能。我们考虑了三种类型的自适应攻击，并证明了STDLens对高级攻击者的健壮性。大量的实验表明，STDLens能够保护FL免受不同模型的劫持攻击，并且在识别和去除特洛伊木马梯度方面优于现有的方法，具有明显更高的精度和更低的误检率。



## **28. Deep Composite Face Image Attacks: Generation, Vulnerability and Detection**

深度复合人脸图像攻击：产生、漏洞和检测 cs.CV

The submitted paper is accepted in IEEE Access 2023

**SubmitDate**: 2023-03-20    [abs](http://arxiv.org/abs/2211.11039v3) [paper-pdf](http://arxiv.org/pdf/2211.11039v3)

**Authors**: Jag Mohan Singh, Raghavendra Ramachandra

**Abstract**: Face manipulation attacks have drawn the attention of biometric researchers because of their vulnerability to Face Recognition Systems (FRS). This paper proposes a novel scheme to generate Composite Face Image Attacks (CFIA) based on facial attributes using Generative Adversarial Networks (GANs). Given the face images corresponding to two unique data subjects, the proposed CFIA method will independently generate the segmented facial attributes, then blend them using transparent masks to generate the CFIA samples. We generate $526$ unique CFIA combinations of facial attributes for each pair of contributory data subjects. Extensive experiments are carried out on our newly generated CFIA dataset consisting of 1000 unique identities with 2000 bona fide samples and 526000 CFIA samples, thus resulting in an overall 528000 face image samples. {{We present a sequence of experiments to benchmark the attack potential of CFIA samples using four different automatic FRS}}. We introduced a new metric named Generalized Morphing Attack Potential (G-MAP) to benchmark the vulnerability of generated attacks on FRS effectively. Additional experiments are performed on the representative subset of the CFIA dataset to benchmark both perceptual quality and human observer response. Finally, the CFIA detection performance is benchmarked using three different single image based face Morphing Attack Detection (MAD) algorithms. The source code of the proposed method together with CFIA dataset will be made publicly available: \url{https://github.com/jagmohaniiit/LatentCompositionCode}

摘要: 人脸操纵攻击因其易受人脸识别系统(FRS)攻击而受到生物特征识别研究人员的关注。提出了一种利用生成性对抗网络(GANS)生成基于人脸属性的复合人脸图像攻击(CFIA)的新方案。在给定两个不同数据对象对应的人脸图像的情况下，CFIA方法将独立地生成分割后的人脸属性，然后使用透明掩膜进行混合以生成CFIA样本。我们为每对有贡献的数据对象生成$526$独特的面部属性CFIA组合。在我们新生成的包含1,000个唯一身份的CFIA数据集上进行了大量的实验，其中包含2,000个真实样本和526000个CFIA样本，从而得到总共528000个人脸图像样本。{{我们提供了一系列实验，以使用四种不同的自动FRS对CFIA样本的攻击潜力进行基准测试}}我们引入了一种新的度量--广义变形攻击潜力(G-MAP)来有效地评估生成攻击对FRS的脆弱性。在CFIA数据集的代表性子集上进行了其他实验，以对感知质量和人类观察者的反应进行基准测试。最后，使用三种不同的基于单幅图像的人脸变形攻击检测(MAD)算法对CFIA检测性能进行了基准测试。建议的方法的源代码和CFIA数据集将公开提供：\url{https://github.com/jagmohaniiit/LatentCompositionCode}



## **29. GNN-Ensemble: Towards Random Decision Graph Neural Networks**

GNN-集成：走向随机决策图神经网络 cs.LG

**SubmitDate**: 2023-03-20    [abs](http://arxiv.org/abs/2303.11376v1) [paper-pdf](http://arxiv.org/pdf/2303.11376v1)

**Authors**: Wenqi Wei, Mu Qiao, Divyesh Jadav

**Abstract**: Graph Neural Networks (GNNs) have enjoyed wide spread applications in graph-structured data. However, existing graph based applications commonly lack annotated data. GNNs are required to learn latent patterns from a limited amount of training data to perform inferences on a vast amount of test data. The increased complexity of GNNs, as well as a single point of model parameter initialization, usually lead to overfitting and sub-optimal performance. In addition, it is known that GNNs are vulnerable to adversarial attacks. In this paper, we push one step forward on the ensemble learning of GNNs with improved accuracy, generalization, and adversarial robustness. Following the principles of stochastic modeling, we propose a new method called GNN-Ensemble to construct an ensemble of random decision graph neural networks whose capacity can be arbitrarily expanded for improvement in performance. The essence of the method is to build multiple GNNs in randomly selected substructures in the topological space and subfeatures in the feature space, and then combine them for final decision making. These GNNs in different substructure and subfeature spaces generalize their classification in complementary ways. Consequently, their combined classification performance can be improved and overfitting on the training data can be effectively reduced. In the meantime, we show that GNN-Ensemble can significantly improve the adversarial robustness against attacks on GNNs.

摘要: 图神经网络(GNN)在图结构数据中有着广泛的应用。然而，现有的基于图形的应用程序通常缺乏带注释的数据。GNN需要从有限数量的训练数据中学习潜在模式，以对大量测试数据进行推理。GNN的复杂性增加，以及模型参数的单点初始化，通常会导致过度拟合和次优性能。此外，众所周知，GNN很容易受到对抗性攻击。在本文中，我们将GNN的集成学习向前推进了一步，提高了准确率、泛化能力和对抗鲁棒性。根据随机建模的原理，我们提出了一种称为GNN集成的新方法来构造随机决策图神经网络集成，该集成的容量可以任意扩展以提高性能。该方法的实质是在拓扑空间中随机选择的子结构和特征空间中的子特征中构建多个GNN，然后将它们组合起来进行最终决策。这些不同子结构和子特征空间中的GNN以互补的方式概括了它们的分类。因此，可以提高它们的组合分类性能，并且可以有效地减少对训练数据的过度拟合。同时，我们也证明了GNN集成能够显著提高GNN对攻击的健壮性。



## **30. Adversarial Attacks against Binary Similarity Systems**

针对二进制相似系统的敌意攻击 cs.CR

**SubmitDate**: 2023-03-20    [abs](http://arxiv.org/abs/2303.11143v1) [paper-pdf](http://arxiv.org/pdf/2303.11143v1)

**Authors**: Gianluca Capozzi, Daniele Cono D'Elia, Giuseppe Antonio Di Luna, Leonardo Querzoni

**Abstract**: In recent years, binary analysis gained traction as a fundamental approach to inspect software and guarantee its security. Due to the exponential increase of devices running software, much research is now moving towards new autonomous solutions based on deep learning models, as they have been showing state-of-the-art performances in solving binary analysis problems. One of the hot topics in this context is binary similarity, which consists in determining if two functions in assembly code are compiled from the same source code. However, it is unclear how deep learning models for binary similarity behave in an adversarial context. In this paper, we study the resilience of binary similarity models against adversarial examples, showing that they are susceptible to both targeted and untargeted attacks (w.r.t. similarity goals) performed by black-box and white-box attackers. In more detail, we extensively test three current state-of-the-art solutions for binary similarity against two black-box greedy attacks, including a new technique that we call Spatial Greedy, and one white-box attack in which we repurpose a gradient-guided strategy used in attacks to image classifiers.

摘要: 近年来，二进制分析作为检查软件和保证其安全性的基本方法得到了越来越多的重视。由于运行软件的设备呈指数级增长，许多研究现在正在转向基于深度学习模型的新的自主解决方案，因为它们在解决二进制分析问题方面表现出了最先进的性能。这方面的一个热门话题是二进制相似性，即确定汇编代码中的两个函数是否从相同的源代码编译而来。然而，目前还不清楚二元相似性的深度学习模型在对抗性环境中的表现如何。在本文中，我们研究了二进制相似模型对敌意例子的弹性，表明它们对目标攻击和非目标攻击都敏感(w.r.t.相似性目标)由黑盒和白盒攻击者执行。更详细地，我们针对两种黑盒贪婪攻击广泛地测试了三种当前最先进的二进制相似性解决方案，其中包括一种称为空间贪婪的新技术，以及一种白盒攻击，其中我们将攻击中使用的梯度引导策略重新用于图像分类器。



## **31. Who Is the Strongest Enemy? Towards Optimal and Efficient Evasion Attacks in Deep RL**

谁是最强大的敌人？面向最优高效逃避攻击的Deep RL cs.LG

In the 10th International Conference on Learning Representations  (ICLR 2022)

**SubmitDate**: 2023-03-20    [abs](http://arxiv.org/abs/2106.05087v5) [paper-pdf](http://arxiv.org/pdf/2106.05087v5)

**Authors**: Yanchao Sun, Ruijie Zheng, Yongyuan Liang, Furong Huang

**Abstract**: Evaluating the worst-case performance of a reinforcement learning (RL) agent under the strongest/optimal adversarial perturbations on state observations (within some constraints) is crucial for understanding the robustness of RL agents. However, finding the optimal adversary is challenging, in terms of both whether we can find the optimal attack and how efficiently we can find it. Existing works on adversarial RL either use heuristics-based methods that may not find the strongest adversary, or directly train an RL-based adversary by treating the agent as a part of the environment, which can find the optimal adversary but may become intractable in a large state space. This paper introduces a novel attacking method to find the optimal attacks through collaboration between a designed function named "actor" and an RL-based learner named "director". The actor crafts state perturbations for a given policy perturbation direction, and the director learns to propose the best policy perturbation directions. Our proposed algorithm, PA-AD, is theoretically optimal and significantly more efficient than prior RL-based works in environments with large state spaces. Empirical results show that our proposed PA-AD universally outperforms state-of-the-art attacking methods in various Atari and MuJoCo environments. By applying PA-AD to adversarial training, we achieve state-of-the-art empirical robustness in multiple tasks under strong adversaries. The codebase is released at https://github.com/umd-huang-lab/paad_adv_rl.

摘要: 评估强化学习(RL)代理在状态观测(在某些约束范围内)的最强/最优对抗扰动下的最坏情况下的性能对于理解RL代理的稳健性至关重要。然而，就我们是否能找到最佳攻击以及找到最佳攻击的效率而言，找到最佳对手是具有挑战性的。现有的对抗性RL研究要么使用基于启发式的方法，可能找不到最强的对手，要么将智能体视为环境的一部分，直接训练基于RL的对手，这可以找到最优的对手，但在大的状态空间中可能变得难以处理。本文提出了一种新的攻击方法，通过设计一个名为“参与者”的函数和一个名为“导演”的基于RL的学习器之间的协作来寻找最优攻击。参与者为给定的政策扰动方向制作状态扰动，导演学习提出最佳政策扰动方向。我们提出的算法PA-AD在理论上是最优的，并且在具有大状态空间的环境中比以前的基于RL的工作更有效。实验结果表明，在不同的Atari和MuJoCo环境下，我们提出的PA-AD攻击方法的性能普遍优于最新的攻击方法。通过将PA-AD应用于对抗性训练，我们在强对手下的多任务中获得了最先进的经验稳健性。代码库在https://github.com/umd-huang-lab/paad_adv_rl.上发布



## **32. Translate your gibberish: black-box adversarial attack on machine translation systems**

翻译你的胡言乱语：对机器翻译系统的黑箱对抗性攻击 cs.CL

**SubmitDate**: 2023-03-20    [abs](http://arxiv.org/abs/2303.10974v1) [paper-pdf](http://arxiv.org/pdf/2303.10974v1)

**Authors**: Andrei Chertkov, Olga Tsymboi, Mikhail Pautov, Ivan Oseledets

**Abstract**: Neural networks are deployed widely in natural language processing tasks on the industrial scale, and perhaps the most often they are used as compounds of automatic machine translation systems. In this work, we present a simple approach to fool state-of-the-art machine translation tools in the task of translation from Russian to English and vice versa. Using a novel black-box gradient-free tensor-based optimizer, we show that many online translation tools, such as Google, DeepL, and Yandex, may both produce wrong or offensive translations for nonsensical adversarial input queries and refuse to translate seemingly benign input phrases. This vulnerability may interfere with understanding a new language and simply worsen the user's experience while using machine translation systems, and, hence, additional improvements of these tools are required to establish better translation.

摘要: 神经网络在工业规模的自然语言处理任务中被广泛部署，也许最常被用作自动机器翻译系统的复合体。在这项工作中，我们提出了一种简单的方法，在从俄语到英语的翻译任务中愚弄最先进的机器翻译工具，反之亦然。使用一种新的黑盒无梯度张量优化器，我们证明了许多在线翻译工具，如Google，DeepL和Yandex，都可能对无意义的对抗性输入查询产生错误或攻击性的翻译，并拒绝翻译看似良性的输入短语。此漏洞可能会干扰对新语言的理解，并只会恶化用户在使用机器翻译系统时的体验，因此，需要对这些工具进行额外的改进才能建立更好的翻译。



## **33. Revisiting Realistic Test-Time Training: Sequential Inference and Adaptation by Anchored Clustering Regularized Self-Training**

重温现实测试时间训练：顺序推理与锚定聚类规则化自我训练的适应 cs.LG

Test-time training, Self-training. arXiv admin note: substantial text  overlap with arXiv:2206.02721

**SubmitDate**: 2023-03-20    [abs](http://arxiv.org/abs/2303.10856v1) [paper-pdf](http://arxiv.org/pdf/2303.10856v1)

**Authors**: Yongyi Su, Xun Xu, Tianrui Li, Kui Jia

**Abstract**: Deploying models on target domain data subject to distribution shift requires adaptation. Test-time training (TTT) emerges as a solution to this adaptation under a realistic scenario where access to full source domain data is not available, and instant inference on the target domain is required. Despite many efforts into TTT, there is a confusion over the experimental settings, thus leading to unfair comparisons. In this work, we first revisit TTT assumptions and categorize TTT protocols by two key factors. Among the multiple protocols, we adopt a realistic sequential test-time training (sTTT) protocol, under which we develop a test-time anchored clustering (TTAC) approach to enable stronger test-time feature learning. TTAC discovers clusters in both source and target domains and matches the target clusters to the source ones to improve adaptation. When source domain information is strictly absent (i.e. source-free) we further develop an efficient method to infer source domain distributions for anchored clustering. Finally, self-training~(ST) has demonstrated great success in learning from unlabeled data and we empirically figure out that applying ST alone to TTT is prone to confirmation bias. Therefore, a more effective TTT approach is introduced by regularizing self-training with anchored clustering, and the improved model is referred to as TTAC++. We demonstrate that, under all TTT protocols, TTAC++ consistently outperforms the state-of-the-art methods on five TTT datasets, including corrupted target domain, selected hard samples, synthetic-to-real adaptation and adversarially attacked target domain. We hope this work will provide a fair benchmarking of TTT methods, and future research should be compared within respective protocols.

摘要: 在受分布变化影响的目标领域数据上部署模型需要适应。测试时间训练(TTT)是在无法访问完整源域数据、需要对目标域进行即时推理的现实场景下出现的一种解决方案。尽管在TTT方面做了很多努力，但在实验设置上仍然存在混乱，从而导致了不公平的比较。在这项工作中，我们首先回顾TTT假设，并根据两个关键因素对TTT协议进行分类。在众多协议中，我们采用了一种现实的顺序测试时间训练(STTT)协议，在该协议下，我们开发了一种测试时间锚定聚类(TTAC)方法来实现更强的测试时间特征学习。TTAC同时发现源域和目标域中的簇，并将目标簇与源簇进行匹配，以提高自适应能力。当源域信息严格缺失(即无源)时，我们进一步发展了一种有效的方法来推断锚定聚类的源域分布。最后，自我训练~(ST)在从未标记的数据中学习方面取得了巨大的成功，我们实证地发现，将ST单独应用于TTT容易产生确认偏差。因此，引入了一种更有效的TTT方法，即通过锚定聚类来规则化自我训练，并将改进的模型称为TTAC++。我们证明，在所有TTT协议下，TTAC++在五个TTT数据集上的性能一致优于最新的方法，包括被破坏的目标域、选择的硬样本、合成到真实的适应和恶意攻击的目标域。我们希望这项工作将为TTT方法提供一个公平的基准，未来的研究应该在各自的协议内进行比较。



## **34. Defending Adversarial Attacks on Deep Learning Based Power Allocation in Massive MIMO Using Denoising Autoencoders**

利用去噪自动编码器防御大规模MIMO中基于深度学习的功率分配的敌意攻击 eess.SP

This work has been published in the IEEE Transactions on Cognitive  Communications and Networking

**SubmitDate**: 2023-03-20    [abs](http://arxiv.org/abs/2211.15365v3) [paper-pdf](http://arxiv.org/pdf/2211.15365v3)

**Authors**: Rajeev Sahay, Minjun Zhang, David J. Love, Christopher G. Brinton

**Abstract**: Recent work has advocated for the use of deep learning to perform power allocation in the downlink of massive MIMO (maMIMO) networks. Yet, such deep learning models are vulnerable to adversarial attacks. In the context of maMIMO power allocation, adversarial attacks refer to the injection of subtle perturbations into the deep learning model's input, during inference (i.e., the adversarial perturbation is injected into inputs during deployment after the model has been trained) that are specifically crafted to force the trained regression model to output an infeasible power allocation solution. In this work, we develop an autoencoder-based mitigation technique, which allows deep learning-based power allocation models to operate in the presence of adversaries without requiring retraining. Specifically, we develop a denoising autoencoder (DAE), which learns a mapping between potentially perturbed data and its corresponding unperturbed input. We test our defense across multiple attacks and in multiple threat models and demonstrate its ability to (i) mitigate the effects of adversarial attacks on power allocation networks using two common precoding schemes, (ii) outperform previously proposed benchmarks for mitigating regression-based adversarial attacks on maMIMO networks, (iii) retain accurate performance in the absence of an attack, and (iv) operate with low computational overhead.

摘要: 最近的工作主张使用深度学习在大规模MIMO(MaMIMO)网络的下行链路中执行功率分配。然而，这种深度学习模型很容易受到对手的攻击。在maMIMO功率分配的上下文中，对抗性攻击是指在推理期间将微妙的扰动注入深度学习模型的输入(即，在模型被训练后在部署期间将对抗性扰动注入输入)，这是专门定制的，以迫使训练的回归模型输出不可行的功率分配解。在这项工作中，我们开发了一种基于自动编码器的缓解技术，该技术允许基于深度学习的功率分配模型在对手存在的情况下运行，而不需要重新训练。具体地说，我们开发了一个去噪自动编码器(DAE)，它学习潜在扰动数据与其对应的未扰动输入之间的映射。我们在多个攻击和多个威胁模型中测试了我们的防御，并证明了它的能力：(I)使用两种常见的预编码方案缓解对抗性攻击对功率分配网络的影响；(Ii)优于先前提出的针对maMIMO网络的基于回归的对抗性攻击基准；(Iii)在没有攻击的情况下保持准确的性能；以及(Iv)以较低的计算开销运行。



## **35. k-SALSA: k-anonymous synthetic averaging of retinal images via local style alignment**

K-SALSA：K-基于局部风格对齐的视网膜图像匿名合成平均 cs.CV

European Conference on Computer Vision (ECCV), 2022

**SubmitDate**: 2023-03-20    [abs](http://arxiv.org/abs/2303.10824v1) [paper-pdf](http://arxiv.org/pdf/2303.10824v1)

**Authors**: Minkyu Jeon, Hyeonjin Park, Hyunwoo J. Kim, Michael Morley, Hyunghoon Cho

**Abstract**: The application of modern machine learning to retinal image analyses offers valuable insights into a broad range of human health conditions beyond ophthalmic diseases. Additionally, data sharing is key to fully realizing the potential of machine learning models by providing a rich and diverse collection of training data. However, the personally-identifying nature of retinal images, encompassing the unique vascular structure of each individual, often prevents this data from being shared openly. While prior works have explored image de-identification strategies based on synthetic averaging of images in other domains (e.g. facial images), existing techniques face difficulty in preserving both privacy and clinical utility in retinal images, as we demonstrate in our work. We therefore introduce k-SALSA, a generative adversarial network (GAN)-based framework for synthesizing retinal fundus images that summarize a given private dataset while satisfying the privacy notion of k-anonymity. k-SALSA brings together state-of-the-art techniques for training and inverting GANs to achieve practical performance on retinal images. Furthermore, k-SALSA leverages a new technique, called local style alignment, to generate a synthetic average that maximizes the retention of fine-grain visual patterns in the source images, thus improving the clinical utility of the generated images. On two benchmark datasets of diabetic retinopathy (EyePACS and APTOS), we demonstrate our improvement upon existing methods with respect to image fidelity, classification performance, and mitigation of membership inference attacks. Our work represents a step toward broader sharing of retinal images for scientific collaboration. Code is available at https://github.com/hcholab/k-salsa.

摘要: 现代机器学习在视网膜图像分析中的应用为眼科疾病以外的一系列人类健康状况提供了有价值的见解。此外，通过提供丰富多样的训练数据集合，数据共享是充分发挥机器学习模型潜力的关键。然而，视网膜图像的个人识别性质，包括每个人独特的血管结构，往往阻止这些数据被公开分享。虽然以前的工作已经探索了基于其他领域(例如面部图像)图像的合成平均的图像去识别策略，但正如我们在工作中所展示的那样，现有的技术在保护视网膜图像的隐私和临床实用性方面面临着困难。因此，我们引入了k-SASA，这是一个基于生成性对抗网络(GAN)的框架，用于合成视网膜眼底图像，该框架总结了给定的私有数据集，同时满足k-匿名的隐私概念。K-SALSA结合了训练和倒置GANS的最先进技术，以实现在视网膜图像上的实际性能。此外，k-SALSA利用一种名为局部样式对齐的新技术来生成合成平均值，该合成平均值最大化地保留了源图像中的细粒度视觉模式，从而提高了生成图像的临床实用性。在糖尿病视网膜病变的两个基准数据集(EyePACS和APTOS)上，我们展示了我们在图像保真度、分类性能和缓解成员关系推断攻击方面相对于现有方法的改进。我们的工作代表着为科学合作而更广泛地共享视网膜图像的一步。代码可在https://github.com/hcholab/k-salsa.上找到



## **36. CBA: Contextual Background Attack against Optical Aerial Detection in the Physical World**

CBA：物理世界中对光学空中探测的背景攻击 cs.CV

**SubmitDate**: 2023-03-20    [abs](http://arxiv.org/abs/2302.13519v2) [paper-pdf](http://arxiv.org/pdf/2302.13519v2)

**Authors**: Jiawei Lian, Xiaofei Wang, Yuru Su, Mingyang Ma, Shaohui Mei

**Abstract**: Patch-based physical attacks have increasingly aroused concerns.   However, most existing methods focus on obscuring targets captured on the ground, and some of these methods are simply extended to deceive aerial detectors.   They smear the targeted objects in the physical world with the elaborated adversarial patches, which can only slightly sway the aerial detectors' prediction and with weak attack transferability.   To address the above issues, we propose to perform Contextual Background Attack (CBA), a novel physical attack framework against aerial detection, which can achieve strong attack efficacy and transferability in the physical world even without smudging the interested objects at all.   Specifically, the targets of interest, i.e. the aircraft in aerial images, are adopted to mask adversarial patches.   The pixels outside the mask area are optimized to make the generated adversarial patches closely cover the critical contextual background area for detection, which contributes to gifting adversarial patches with more robust and transferable attack potency in the real world.   To further strengthen the attack performance, the adversarial patches are forced to be outside targets during training, by which the detected objects of interest, both on and outside patches, benefit the accumulation of attack efficacy.   Consequently, the sophisticatedly designed patches are gifted with solid fooling efficacy against objects both on and outside the adversarial patches simultaneously.   Extensive proportionally scaled experiments are performed in physical scenarios, demonstrating the superiority and potential of the proposed framework for physical attacks.   We expect that the proposed physical attack method will serve as a benchmark for assessing the adversarial robustness of diverse aerial detectors and defense methods.

摘要: 基于补丁的物理攻击越来越引起人们的关注。然而，现有的大多数方法都集中在遮挡地面捕获的目标上，其中一些方法只是简单地扩展到欺骗航空探测器。他们用精心制作的对抗性补丁涂抹物理世界中的目标对象，这只能轻微动摇航空探测器的预测，攻击可转移性较弱。为了解决上述问题，我们提出了一种新的针对空中探测的物理攻击框架--上下文背景攻击(CBA)，该框架即使在不玷污感兴趣对象的情况下也可以在物理世界中实现强大的攻击效能和可转移性。具体地说，采用感兴趣的目标，即航空图像中的飞机来掩盖敌方补丁。对掩码区域外的像素进行了优化，使生成的对抗性补丁紧密覆盖关键背景区域进行检测，有助于在现实世界中赋予对抗性补丁更健壮和可转移的攻击能力。为了进一步增强攻击性能，在训练过程中将对抗性补丁强制为外部目标，这样无论是在补丁上还是在补丁外，检测到的感兴趣对象都有利于攻击效能的积累。因此，复杂设计的补丁被赋予了对敌方补丁内外的对象同时具有可靠的愚弄效果。在物理场景中进行了广泛的按比例扩展的实验，展示了所提出的框架在物理攻击方面的优势和潜力。我们期望所提出的物理攻击方法将作为评估不同空中探测器和防御方法的对抗健壮性的基准。



## **37. Randomized Adversarial Training via Taylor Expansion**

基于泰勒展开的随机对抗性训练 cs.LG

CVPR 2023

**SubmitDate**: 2023-03-19    [abs](http://arxiv.org/abs/2303.10653v1) [paper-pdf](http://arxiv.org/pdf/2303.10653v1)

**Authors**: Gaojie Jin, Xinping Yi, Dengyu Wu, Ronghui Mu, Xiaowei Huang

**Abstract**: In recent years, there has been an explosion of research into developing more robust deep neural networks against adversarial examples. Adversarial training appears as one of the most successful methods. To deal with both the robustness against adversarial examples and the accuracy over clean examples, many works develop enhanced adversarial training methods to achieve various trade-offs between them. Leveraging over the studies that smoothed update on weights during training may help find flat minima and improve generalization, we suggest reconciling the robustness-accuracy trade-off from another perspective, i.e., by adding random noise into deterministic weights. The randomized weights enable our design of a novel adversarial training method via Taylor expansion of a small Gaussian noise, and we show that the new adversarial training method can flatten loss landscape and find flat minima. With PGD, CW, and Auto Attacks, an extensive set of experiments demonstrate that our method enhances the state-of-the-art adversarial training methods, boosting both robustness and clean accuracy. The code is available at https://github.com/Alexkael/Randomized-Adversarial-Training.

摘要: 近年来，针对敌意例子开发更健壮的深度神经网络的研究呈爆炸式增长。对抗性训练似乎是最成功的方法之一。为了处理对抗对抗性样本的健壮性和对干净样本的准确性，许多工作开发了增强的对抗性训练方法来实现它们之间的各种权衡。利用在训练过程中平滑权重更新的研究可能有助于找到平坦的最小值并改进泛化，我们建议从另一个角度协调稳健性和准确性之间的权衡，即通过将随机噪声添加到确定性权重中。随机化的权值使得我们能够通过对一个小的高斯噪声进行泰勒展开来设计一种新的对抗性训练方法，并且我们证明了这种新的对抗性训练方法能够平坦化损失情况并找到平坦的最小值。对于PGD、CW和Auto攻击，一组广泛的实验表明，我们的方法增强了最先进的对抗性训练方法，提高了健壮性和干净的准确性。代码可在https://github.com/Alexkael/Randomized-Adversarial-Training.上获得



## **38. AdaptGuard: Defending Against Universal Attacks for Model Adaptation**

AdaptGuard：针对模型适配的通用攻击防御 cs.CR

15 pages, 4 figures

**SubmitDate**: 2023-03-19    [abs](http://arxiv.org/abs/2303.10594v1) [paper-pdf](http://arxiv.org/pdf/2303.10594v1)

**Authors**: Lijun Sheng, Jian Liang, Ran He, Zilei Wang, Tieniu Tan

**Abstract**: Model adaptation aims at solving the domain transfer problem under the constraint of only accessing the pretrained source models. With the increasing considerations of data privacy and transmission efficiency, this paradigm has been gaining recent popularity. This paper studies the vulnerability to universal attacks transferred from the source domain during model adaptation algorithms due to the existence of the malicious providers. We explore both universal adversarial perturbations and backdoor attacks as loopholes on the source side and discover that they still survive in the target models after adaptation. To address this issue, we propose a model preprocessing framework, named AdaptGuard, to improve the security of model adaptation algorithms. AdaptGuard avoids direct use of the risky source parameters through knowledge distillation and utilizes the pseudo adversarial samples under adjusted radius to enhance the robustness. AdaptGuard is a plug-and-play module that requires neither robust pretrained models nor any changes for the following model adaptation algorithms. Extensive results on three commonly used datasets and two popular adaptation methods validate that AdaptGuard can effectively defend against universal attacks and maintain clean accuracy in the target domain simultaneously. We hope this research will shed light on the safety and robustness of transfer learning.

摘要: 模型自适应的目的是在只访问预先训练好的源模型的约束下解决域迁移问题。随着人们对数据隐私和传输效率的越来越多的考虑，这种范式最近越来越受欢迎。本文研究了在模型自适应算法中，由于恶意提供者的存在，对源域传输的通用攻击的脆弱性。我们探索了普遍的对抗性扰动和后门攻击作为源端的漏洞，并发现它们在适应后仍然存在于目标模型中。针对这一问题，我们提出了一种模型预处理框架AdaptGuard，以提高模型自适应算法的安全性。AdaptGuard通过知识提取避免了直接使用风险源参数，并利用调整后的半径下的伪对手样本来增强鲁棒性。AdaptGuard是一个即插即用模块，它既不需要健壮的预先训练的模型，也不需要对以下模型自适应算法进行任何更改。在三个常用数据集和两个流行的自适应方法上的广泛结果验证了AdaptGuard能够有效地防御通用攻击，同时保持目标领域的干净准确性。我们希望这项研究能对迁移学习的安全性和稳健性有所帮助。



## **39. NoisyHate: Benchmarking Content Moderation Machine Learning Models with Human-Written Perturbations Online**

NoisyHate：使用在线人类书写的扰动对内容审核机器学习模型进行基准测试 cs.LG

**SubmitDate**: 2023-03-18    [abs](http://arxiv.org/abs/2303.10430v1) [paper-pdf](http://arxiv.org/pdf/2303.10430v1)

**Authors**: Yiran Ye, Thai Le, Dongwon Lee

**Abstract**: Online texts with toxic content are a threat in social media that might cause cyber harassment. Although many platforms applied measures, such as machine learning-based hate-speech detection systems, to diminish their effect, those toxic content publishers can still evade the system by modifying the spelling of toxic words. Those modified words are also known as human-written text perturbations. Many research works developed certain techniques to generate adversarial samples to help the machine learning models obtain the ability to recognize those perturbations. However, there is still a gap between those machine-generated perturbations and human-written perturbations. In this paper, we introduce a benchmark test set containing human-written perturbations online for toxic speech detection models. We also recruited a group of workers to evaluate the quality of this test set and dropped low-quality samples. Meanwhile, to check if our perturbation can be normalized to its clean version, we applied spell corrector algorithms on this dataset. Finally, we test this data on state-of-the-art language models, such as BERT and RoBERTa, and black box APIs, such as perspective API, to demonstrate the adversarial attack with real human-written perturbations is still effective.

摘要: 含有有毒内容的在线文本是社交媒体中的一种威胁，可能会导致网络骚扰。尽管许多平台应用了一些措施，如基于机器学习的仇恨语音检测系统，以降低其影响，但这些有毒内容出版商仍然可以通过修改有毒单词的拼写来规避该系统。这些修改后的单词也称为人类书写的文本扰动。许多研究工作开发了某些技术来生成对抗性样本，以帮助机器学习模型获得识别这些扰动的能力。然而，在这些机器产生的扰动和人类书写的扰动之间仍然存在差距。在本文中，我们为有毒语音检测模型引入了一个包含人类书写扰动的在线基准测试集。我们还招募了一组工作人员来评估这个测试集的质量，并丢弃了低质量的样本。同时，为了检查我们的扰动是否可以归一化到它的干净版本，我们在这个数据集上应用了拼写更正算法。最后，我们在BERT和Roberta等最先进的语言模型和透视API等黑盒API上测试了这些数据，以证明带有真实人类书写扰动的对抗性攻击仍然有效。



## **40. FedRight: An Effective Model Copyright Protection for Federated Learning**

FedRight：一种有效的联合学习版权保护模式 cs.CR

**SubmitDate**: 2023-03-18    [abs](http://arxiv.org/abs/2303.10399v1) [paper-pdf](http://arxiv.org/pdf/2303.10399v1)

**Authors**: Jinyin Chen, Mingjun Li, Mingjun Li, Haibin Zheng

**Abstract**: Federated learning (FL), an effective distributed machine learning framework, implements model training and meanwhile protects local data privacy. It has been applied to a broad variety of practice areas due to its great performance and appreciable profits. Who owns the model, and how to protect the copyright has become a real problem. Intuitively, the existing property rights protection methods in centralized scenarios (e.g., watermark embedding and model fingerprints) are possible solutions for FL. But they are still challenged by the distributed nature of FL in aspects of the no data sharing, parameter aggregation, and federated training settings. For the first time, we formalize the problem of copyright protection for FL, and propose FedRight to protect model copyright based on model fingerprints, i.e., extracting model features by generating adversarial examples as model fingerprints. FedRight outperforms previous works in four key aspects: (i) Validity: it extracts model features to generate transferable fingerprints to train a detector to verify the copyright of the model. (ii) Fidelity: it is with imperceptible impact on the federated training, thus promising good main task performance. (iii) Robustness: it is empirically robust against malicious attacks on copyright protection, i.e., fine-tuning, model pruning, and adaptive attacks. (iv) Black-box: it is valid in the black-box forensic scenario where only application programming interface calls to the model are available. Extensive evaluations across 3 datasets and 9 model structures demonstrate FedRight's superior fidelity, validity, and robustness.

摘要: 联邦学习(FL)是一种有效的分布式机器学习框架，在实现模型训练的同时保护本地数据隐私。由于其良好的性能和可观的收益，它已被广泛应用于各种实践领域。谁拥有这种模式，如何保护版权已经成为一个现实的问题。直观地说，现有的集中式场景下的产权保护方法(如水印嵌入和模型指纹)是FL的可能解决方案。但在无数据共享、参数聚合和联合训练设置等方面，他们仍然受到FL的分布式特性的挑战。本文首次对FL的版权保护问题进行了形式化描述，提出了基于模型指纹的FedRight模型版权保护方法，即通过生成对抗性样本作为模型指纹来提取模型特征。FedRight在四个关键方面优于以往的工作：(I)有效性：它提取模型特征以生成可转移的指纹，并训练检测器来验证模型的版权。(Ii)忠诚度：它对联合训练具有潜移默化的影响，从而保证了良好的主要任务表现。(Iii)稳健性：根据经验，它对版权保护方面的恶意攻击具有稳健性，即微调、模型修剪和自适应攻击。(4)黑盒：它在只有对模型的应用程序编程接口调用可用的黑盒取证场景中有效。对3个数据集和9个模型结构的广泛评估表明，FedRight具有卓越的保真度、有效性和健壮性。



## **41. Practical Cross-System Shilling Attacks with Limited Access to Data**

数据访问受限的实用跨系统先令攻击 cs.IR

AAAI 2023

**SubmitDate**: 2023-03-18    [abs](http://arxiv.org/abs/2302.07145v2) [paper-pdf](http://arxiv.org/pdf/2302.07145v2)

**Authors**: Meifang Zeng, Ke Li, Bingchuan Jiang, Liujuan Cao, Hui Li

**Abstract**: In shilling attacks, an adversarial party injects a few fake user profiles into a Recommender System (RS) so that the target item can be promoted or demoted. Although much effort has been devoted to developing shilling attack methods, we find that existing approaches are still far from practical. In this paper, we analyze the properties a practical shilling attack method should have and propose a new concept of Cross-system Attack. With the idea of Cross-system Attack, we design a Practical Cross-system Shilling Attack (PC-Attack) framework that requires little information about the victim RS model and the target RS data for conducting attacks. PC-Attack is trained to capture graph topology knowledge from public RS data in a self-supervised manner. Then, it is fine-tuned on a small portion of target data that is easy to access to construct fake profiles. Extensive experiments have demonstrated the superiority of PC-Attack over state-of-the-art baselines. Our implementation of PC-Attack is available at https://github.com/KDEGroup/PC-Attack.

摘要: 在先令攻击中，敌对方向推荐系统(RS)注入一些虚假的用户配置文件，以便目标项目可以升级或降级。虽然已经投入了大量的精力来开发先令攻击方法，但我们发现现有的方法仍然远远不实用。本文分析了一种实用的先令攻击方法应具备的性质，提出了跨系统攻击的新概念。利用跨系统攻击的思想，我们设计了一个实用的跨系统先令攻击(PC-Attack)框架，该框架只需要很少的受害者RS模型和目标RS数据的信息就可以进行攻击。PC-Attack被训练成以自监督的方式从公共遥感数据中捕获图拓扑知识。然后，它对一小部分目标数据进行微调，这些数据很容易访问以构建虚假配置文件。广泛的实验已经证明了PC攻击相对于最先进的基线的优越性。我们对PC-Attack的实施可在https://github.com/KDEGroup/PC-Attack.获得



## **42. Detection of Uncertainty in Exceedance of Threshold (DUET): An Adversarial Patch Localizer**

超过阈值的不确定性检测(DUET)：对抗性补丁定位器 cs.CV

This paper has won the Best Paper Award in IEEE/ACM International  Conference on Big Data Computing, Applications and Technologies (BDCAT) 2022

**SubmitDate**: 2023-03-18    [abs](http://arxiv.org/abs/2303.10291v1) [paper-pdf](http://arxiv.org/pdf/2303.10291v1)

**Authors**: Terence Jie Chua, Wenhan Yu, Jun Zhao

**Abstract**: Development of defenses against physical world attacks such as adversarial patches is gaining traction within the research community. We contribute to the field of adversarial patch detection by introducing an uncertainty-based adversarial patch localizer which localizes adversarial patch on an image, permitting post-processing patch-avoidance or patch-reconstruction. We quantify our prediction uncertainties with the development of \textit{\textbf{D}etection of \textbf{U}ncertainties in the \textbf{E}xceedance of \textbf{T}hreshold} (DUET) algorithm. This algorithm provides a framework to ascertain confidence in the adversarial patch localization, which is essential for safety-sensitive applications such as self-driving cars and medical imaging. We conducted experiments on localizing adversarial patches and found our proposed DUET model outperforms baseline models. We then conduct further analyses on our choice of model priors and the adoption of Bayesian Neural Networks in different layers within our model architecture. We found that isometric gaussian priors in Bayesian Neural Networks are suitable for patch localization tasks and the presence of Bayesian layers in the earlier neural network blocks facilitates top-end localization performance, while Bayesian layers added in the later neural network blocks contribute to better model generalization. We then propose two different well-performing models to tackle different use cases.

摘要: 针对物理世界攻击(如对抗性补丁)的防御开发正在研究界获得吸引力。通过引入一种基于不确定性的对抗性补丁定位器，该定位器在图像上定位对抗性补丁，允许后处理补丁回避或补丁重建，从而为对抗性补丁检测领域做出了贡献。我们通过在文本bf{T}hreshold}(DUET)算法的文本{E}xceedance中对文本bf{U}不确定性的评估来量化我们的预测不确定性。该算法提供了一个框架来确定对抗性补丁定位的可信度，这对于自动驾驶汽车和医学成像等安全敏感应用是必不可少的。我们进行了定位对抗性补丁的实验，发现我们提出的DUET模型的性能优于基线模型。然后，我们对模型先验的选择以及贝叶斯神经网络在模型体系结构中的不同层的采用进行了进一步的分析。我们发现，贝叶斯神经网络中的等距高斯先验知识适合于面片定位任务，早期神经网络块中贝叶斯层的存在有助于提高高端定位性能，而在后期神经网络块中添加贝叶斯层有助于更好的模型泛化。然后，我们提出了两种不同的性能良好的模型来处理不同的用例。



## **43. Robust Adversarial Attacks Detection based on Explainable Deep Reinforcement Learning For UAV Guidance and Planning**

基于可解释深度强化学习的无人机制导规划鲁棒对抗攻击检测 cs.LG

13 pages, 18 figures

**SubmitDate**: 2023-03-17    [abs](http://arxiv.org/abs/2206.02670v3) [paper-pdf](http://arxiv.org/pdf/2206.02670v3)

**Authors**: Thomas Hickling, Nabil Aouf, Phillippa Spencer

**Abstract**: The dangers of adversarial attacks on Uncrewed Aerial Vehicle (UAV) agents operating in public are increasing. Adopting AI-based techniques and, more specifically, Deep Learning (DL) approaches to control and guide these UAVs can be beneficial in terms of performance but can add concerns regarding the safety of those techniques and their vulnerability against adversarial attacks. Confusion in the agent's decision-making process caused by these attacks can seriously affect the safety of the UAV. This paper proposes an innovative approach based on the explainability of DL methods to build an efficient detector that will protect these DL schemes and the UAVs adopting them from attacks. The agent adopts a Deep Reinforcement Learning (DRL) scheme for guidance and planning. The agent is trained with a Deep Deterministic Policy Gradient (DDPG) with Prioritised Experience Replay (PER) DRL scheme that utilises Artificial Potential Field (APF) to improve training times and obstacle avoidance performance. A simulated environment for UAV explainable DRL-based planning and guidance, including obstacles and adversarial attacks, is built. The adversarial attacks are generated by the Basic Iterative Method (BIM) algorithm and reduced obstacle course completion rates from 97\% to 35\%. Two adversarial attack detectors are proposed to counter this reduction. The first one is a Convolutional Neural Network Adversarial Detector (CNN-AD), which achieves accuracy in the detection of 80\%. The second detector utilises a Long Short Term Memory (LSTM) network. It achieves an accuracy of 91\% with faster computing times compared to the CNN-AD, allowing for real-time adversarial detection.

摘要: 对在公共场合工作的无人驾驶飞行器(UAV)特工进行对抗性攻击的危险正在增加。采用基于人工智能的技术，更具体地说，深度学习(DL)方法来控制和引导这些无人机在性能方面可能是有益的，但可能会增加对这些技术的安全性及其抵御对手攻击的脆弱性的担忧。这些攻击导致的代理决策过程中的混乱会严重影响无人机的安全。本文提出了一种基于DL方法的可解释性的创新方法，以构建一个有效的检测器来保护这些DL方案以及采用这些方案的无人机免受攻击。代理采用深度强化学习(DRL)方案进行指导和规划。代理使用深度确定性策略梯度(DDPG)和优先体验重播(PER)DRL方案进行训练，该方案利用人工势场(APF)来改进训练时间和避障性能。建立了无人机基于DRL的可解释规划和制导的仿真环境，包括障碍物和对抗性攻击。对抗性攻击由基本迭代法(BIM)算法生成，障碍路径完成率由97降至35。提出了两个对抗性攻击检测器来对抗这种减少。第一种是卷积神经网络敌手检测器(CNN-AD)，它可以达到80%的检测精度。第二检测器利用长短期记忆(LSTM)网络。与CNN-AD相比，它具有91%的准确率和更快的计算时间，允许实时检测对手。



## **44. Robust Mode Connectivity-Oriented Adversarial Defense: Enhancing Neural Network Robustness Against Diversified $\ell_p$ Attacks**

面向连通性的强健模式对抗防御：增强神经网络对多样化的$\ell_p$攻击的稳健性 cs.AI

**SubmitDate**: 2023-03-17    [abs](http://arxiv.org/abs/2303.10225v1) [paper-pdf](http://arxiv.org/pdf/2303.10225v1)

**Authors**: Ren Wang, Yuxuan Li, Sijia Liu

**Abstract**: Adversarial robustness is a key concept in measuring the ability of neural networks to defend against adversarial attacks during the inference phase. Recent studies have shown that despite the success of improving adversarial robustness against a single type of attack using robust training techniques, models are still vulnerable to diversified $\ell_p$ attacks. To achieve diversified $\ell_p$ robustness, we propose a novel robust mode connectivity (RMC)-oriented adversarial defense that contains two population-based learning phases. The first phase, RMC, is able to search the model parameter space between two pre-trained models and find a path containing points with high robustness against diversified $\ell_p$ attacks. In light of the effectiveness of RMC, we develop a second phase, RMC-based optimization, with RMC serving as the basic unit for further enhancement of neural network diversified $\ell_p$ robustness. To increase computational efficiency, we incorporate learning with a self-robust mode connectivity (SRMC) module that enables the fast proliferation of the population used for endpoints of RMC. Furthermore, we draw parallels between SRMC and the human immune system. Experimental results on various datasets and model architectures demonstrate that the proposed defense methods can achieve high diversified $\ell_p$ robustness against $\ell_\infty$, $\ell_2$, $\ell_1$, and hybrid attacks. Codes are available at \url{https://github.com/wangren09/MCGR}.

摘要: 对抗健壮性是衡量神经网络在推理阶段防御对手攻击能力的一个关键概念。最近的研究表明，尽管利用健壮的训练技术成功地提高了对手对单一类型攻击的健壮性，但模型仍然容易受到多样化的$\ell_p$攻击。为了实现不同的$\ell_p$健壮性，我们提出了一种新的面向健壮模式连通性(RMC)的对抗防御方法，该方法包含两个基于群体的学习阶段。第一阶段，RMC能够搜索两个预先训练的模型之间的模型参数空间，并找到一条包含点的路径，该路径对各种$\ell_p$攻击具有很强的鲁棒性。鉴于RMC的有效性，我们发展了第二阶段的基于RMC的优化，以RMC为基本单元，进一步增强神经网络多样化的健壮性。为了提高计算效率，我们将学习与自稳健模式连接(SRMC)模块相结合，以支持用于RMC终端的群体的快速增长。此外，我们将SRMC与人类免疫系统相提并论。在不同的数据集和模型体系结构上的实验结果表明，所提出的防御方法对$\ell_inty$、$\ell_2$、$\ell_1$和混合攻击具有高度的多样性$\ell_p$健壮性。代码可在\url{https://github.com/wangren09/MCGR}.



## **45. Can AI-Generated Text be Reliably Detected?**

能否可靠地检测到人工智能生成的文本？ cs.CL

**SubmitDate**: 2023-03-17    [abs](http://arxiv.org/abs/2303.11156v1) [paper-pdf](http://arxiv.org/pdf/2303.11156v1)

**Authors**: Vinu Sankar Sadasivan, Aounon Kumar, Sriram Balasubramanian, Wenxiao Wang, Soheil Feizi

**Abstract**: The rapid progress of Large Language Models (LLMs) has made them capable of performing astonishingly well on various tasks including document completion and question answering. The unregulated use of these models, however, can potentially lead to malicious consequences such as plagiarism, generating fake news, spamming, etc. Therefore, reliable detection of AI-generated text can be critical to ensure the responsible use of LLMs. Recent works attempt to tackle this problem either using certain model signatures present in the generated text outputs or by applying watermarking techniques that imprint specific patterns onto them. In this paper, both empirically and theoretically, we show that these detectors are not reliable in practical scenarios. Empirically, we show that paraphrasing attacks, where a light paraphraser is applied on top of the generative text model, can break a whole range of detectors, including the ones using the watermarking schemes as well as neural network-based detectors and zero-shot classifiers. We then provide a theoretical impossibility result indicating that for a sufficiently good language model, even the best-possible detector can only perform marginally better than a random classifier. Finally, we show that even LLMs protected by watermarking schemes can be vulnerable against spoofing attacks where adversarial humans can infer hidden watermarking signatures and add them to their generated text to be detected as text generated by the LLMs, potentially causing reputational damages to their developers. We believe these results can open an honest conversation in the community regarding the ethical and reliable use of AI-generated text.

摘要: 大型语言模型(LLM)的快速发展使其能够在包括文档完成和问题回答在内的各种任务中表现出惊人的出色表现。然而，不受监管地使用这些模型可能会导致恶意后果，如抄袭、生成假新闻、垃圾邮件等。因此，可靠地检测人工智能生成的文本对于确保负责任地使用LLMS至关重要。最近的工作试图解决这个问题，要么使用生成的文本输出中存在的特定模型签名，要么应用将特定图案印记到输出文本上的水印技术。在这篇文章中，我们从经验和理论上证明了这些检测器在实际场景中是不可靠的。实验结果表明，在生成文本模型的基础上应用光复述攻击，可以破坏一系列检测器，包括使用水印方案的检测器以及基于神经网络的检测器和零镜头分类器。然后，我们提供了一个理论上的不可能性结果，表明对于足够好的语言模型，即使是最好的检测器也只能比随机分类器的性能略好一些。最后，我们证明了即使是受水印方案保护的LLMS也容易受到欺骗攻击，在这种攻击中，敌意的人类可以推断隐藏的水印签名并将它们添加到他们生成的文本中，从而被检测为LLMS生成的文本，这可能会给他们的开发者造成声誉损害。我们相信，这些结果可以在社区中就人工智能生成的文本的道德和可靠使用展开诚实的对话。



## **46. Fuzziness-tuned: Improving the Transferability of Adversarial Examples**

模糊性调整：提高对抗性例证的可转移性 cs.LG

**SubmitDate**: 2023-03-17    [abs](http://arxiv.org/abs/2303.10078v1) [paper-pdf](http://arxiv.org/pdf/2303.10078v1)

**Authors**: Xiangyuan Yang, Jie Lin, Hanlin Zhang, Xinyu Yang, Peng Zhao

**Abstract**: With the development of adversarial attacks, adversairal examples have been widely used to enhance the robustness of the training models on deep neural networks. Although considerable efforts of adversarial attacks on improving the transferability of adversarial examples have been developed, the attack success rate of the transfer-based attacks on the surrogate model is much higher than that on victim model under the low attack strength (e.g., the attack strength $\epsilon=8/255$). In this paper, we first systematically investigated this issue and found that the enormous difference of attack success rates between the surrogate model and victim model is caused by the existence of a special area (known as fuzzy domain in our paper), in which the adversarial examples in the area are classified wrongly by the surrogate model while correctly by the victim model. Then, to eliminate such enormous difference of attack success rates for improving the transferability of generated adversarial examples, a fuzziness-tuned method consisting of confidence scaling mechanism and temperature scaling mechanism is proposed to ensure the generated adversarial examples can effectively skip out of the fuzzy domain. The confidence scaling mechanism and the temperature scaling mechanism can collaboratively tune the fuzziness of the generated adversarial examples through adjusting the gradient descent weight of fuzziness and stabilizing the update direction, respectively. Specifically, the proposed fuzziness-tuned method can be effectively integrated with existing adversarial attacks to further improve the transferability of adverarial examples without changing the time complexity. Extensive experiments demonstrated that fuzziness-tuned method can effectively enhance the transferability of adversarial examples in the latest transfer-based attacks.

摘要: 随着对抗性攻击的发展，对抗性实例被广泛用于增强深层神经网络训练模型的稳健性。虽然对抗性攻击已经在提高对抗性实例的可转移性方面做了大量的工作，但在低攻击强度(例如，攻击强度$\epsilon=8/255$)下，基于转移的攻击对代理模型的攻击成功率远高于对受害者模型的攻击成功率。本文首先对这一问题进行了系统的研究，发现代理模型和受害者模型攻击成功率的巨大差异是由于存在一个特殊区域(本文称之为模糊域)，该区域内的对抗性实例被代理模型错误地分类，而被受害者模型正确地分类。然后，为了消除攻击成功率的巨大差异，以提高生成的对抗性实例的可转移性，提出了一种由置信度调整机制和温度调整机制组成的模糊调整方法，以确保生成的对抗性实例能够有效地跳出模糊域。置信度缩放机制和温度缩放机制分别通过调整模糊性的梯度下降权重和稳定更新方向来协同调整生成的对抗性实例的模糊性。具体地说，该模糊调整方法可以有效地与已有的对抗性攻击相结合，在不改变时间复杂度的前提下进一步提高对抗性实例的可转移性。大量实验表明，在最新的基于转移的攻击中，模糊调整方法可以有效地提高对抗性实例的可转移性。



## **47. Adversarial Counterfactual Visual Explanations**

对抗性反事实视觉解释 cs.CV

CVPR 2023 camera-ready; Main manuscript + supplementary material

**SubmitDate**: 2023-03-17    [abs](http://arxiv.org/abs/2303.09962v1) [paper-pdf](http://arxiv.org/pdf/2303.09962v1)

**Authors**: Guillaume Jeanneret, Loïc Simon, Frédéric Jurie

**Abstract**: Counterfactual explanations and adversarial attacks have a related goal: flipping output labels with minimal perturbations regardless of their characteristics. Yet, adversarial attacks cannot be used directly in a counterfactual explanation perspective, as such perturbations are perceived as noise and not as actionable and understandable image modifications. Building on the robust learning literature, this paper proposes an elegant method to turn adversarial attacks into semantically meaningful perturbations, without modifying the classifiers to explain. The proposed approach hypothesizes that Denoising Diffusion Probabilistic Models are excellent regularizers for avoiding high-frequency and out-of-distribution perturbations when generating adversarial attacks. The paper's key idea is to build attacks through a diffusion model to polish them. This allows studying the target model regardless of its robustification level. Extensive experimentation shows the advantages of our counterfactual explanation approach over current State-of-the-Art in multiple testbeds.

摘要: 反事实解释和对抗性攻击有一个相关的目标：以最小的扰动翻转输出标签，而不考虑它们的特征。然而，对抗性攻击不能直接用于反事实解释的角度，因为这种扰动被视为噪音，而不是可操作和可理解的形象修改。在稳健学习文献的基础上，本文提出了一种巧妙的方法，将对抗性攻击转化为语义上有意义的扰动，而不需要修改分类器来解释。该方法假设去噪扩散概率模型是在产生敌意攻击时避免高频和分布外扰动的优秀正则化方法。本文的核心思想是通过一个扩散模型来构建攻击，以完善攻击。这允许研究目标模型，而不考虑其粗暴程度。广泛的实验表明，在多个试验台上，我们的反事实解释方法比目前最先进的方法具有优势。



## **48. Learning to Unlearn: Instance-wise Unlearning for Pre-trained Classifiers**

学习遗忘：基于实例的预先训练分类器的遗忘 cs.LG

Preprint

**SubmitDate**: 2023-03-17    [abs](http://arxiv.org/abs/2301.11578v2) [paper-pdf](http://arxiv.org/pdf/2301.11578v2)

**Authors**: Sungmin Cha, Sungjun Cho, Dasol Hwang, Honglak Lee, Taesup Moon, Moontae Lee

**Abstract**: Since the recent advent of regulations for data protection (e.g., the General Data Protection Regulation), there has been increasing demand in deleting information learned from sensitive data in pre-trained models without retraining from scratch. The inherent vulnerability of neural networks towards adversarial attacks and unfairness also calls for a robust method to remove or correct information in an instance-wise fashion, while retaining the predictive performance across remaining data. To this end, we define instance-wise unlearning, of which the goal is to delete information on a set of instances from a pre-trained model, by either misclassifying each instance away from its original prediction or relabeling the instance to a different label. We also propose two methods that reduce forgetting on the remaining data: 1) utilizing adversarial examples to overcome forgetting at the representation-level and 2) leveraging weight importance metrics to pinpoint network parameters guilty of propagating unwanted information. Both methods only require the pre-trained model and data instances to forget, allowing painless application to real-life settings where the entire training set is unavailable. Through extensive experimentation on various image classification benchmarks, we show that our approach effectively preserves knowledge of remaining data while unlearning given instances in both single-task and continual unlearning scenarios.

摘要: 自从最近出现了数据保护条例(例如，《一般数据保护条例》)以来，越来越多的人要求在预先训练的模型中删除从敏感数据中学习的信息，而不需要从头开始进行再培训。神经网络对敌意攻击和不公平的固有脆弱性也需要一种健壮的方法来以实例方式移除或纠正信息，同时保持对剩余数据的预测性能。为此，我们定义了基于实例的遗忘，其目标是通过将每个实例从其原始预测中错误分类或将实例重新标记到不同的标签来从预先训练的模型中删除关于一组实例的信息。我们还提出了两种减少对剩余数据的遗忘的方法：1)利用对抗性例子在表示级克服遗忘；2)利用权重重要性度量来精确定位传播无用信息的网络参数。这两种方法只需要忘记预先训练的模型和数据实例，从而可以轻松地应用到无法获得整个训练集的现实生活环境中。通过在不同图像分类基准上的广泛实验，我们的方法有效地保留了剩余数据的知识，同时在单任务和连续遗忘场景中都忘记了给定的实例。



## **49. It Is All About Data: A Survey on the Effects of Data on Adversarial Robustness**

这一切都与数据有关：数据对对手健壮性影响的调查 cs.LG

41 pages, 25 figures, under review

**SubmitDate**: 2023-03-17    [abs](http://arxiv.org/abs/2303.09767v1) [paper-pdf](http://arxiv.org/pdf/2303.09767v1)

**Authors**: Peiyu Xiong, Michael Tegegn, Jaskeerat Singh Sarin, Shubhraneel Pal, Julia Rubin

**Abstract**: Adversarial examples are inputs to machine learning models that an attacker has intentionally designed to confuse the model into making a mistake. Such examples pose a serious threat to the applicability of machine-learning-based systems, especially in life- and safety-critical domains. To address this problem, the area of adversarial robustness investigates mechanisms behind adversarial attacks and defenses against these attacks. This survey reviews literature that focuses on the effects of data used by a model on the model's adversarial robustness. It systematically identifies and summarizes the state-of-the-art research in this area and further discusses gaps of knowledge and promising future research directions.

摘要: 对抗性的例子是机器学习模型的输入，攻击者故意设计这些模型来混淆模型，使其出错。这些例子对基于机器学习的系统的适用性构成了严重威胁，特别是在生命和安全关键领域。为了解决这个问题，对抗性稳健性领域调查了对抗性攻击背后的机制和对这些攻击的防御。这项调查回顾了关注模型使用的数据对模型的对抗稳健性的影响的文献。系统地识别和总结了这一领域的最新研究成果，并进一步讨论了知识差距和未来的研究方向。



## **50. Exorcising ''Wraith'': Protecting LiDAR-based Object Detector in Automated Driving System from Appearing Attacks**

驱鬼：保护自动驾驶系统中基于LiDAR的目标检测器免受攻击 cs.CR

Accepted by USENIX Sercurity 2023

**SubmitDate**: 2023-03-17    [abs](http://arxiv.org/abs/2303.09731v1) [paper-pdf](http://arxiv.org/pdf/2303.09731v1)

**Authors**: Qifan Xiao, Xudong Pan, Yifan Lu, Mi Zhang, Jiarun Dai, Min Yang

**Abstract**: Automated driving systems rely on 3D object detectors to recognize possible obstacles from LiDAR point clouds. However, recent works show the adversary can forge non-existent cars in the prediction results with a few fake points (i.e., appearing attack). By removing statistical outliers, existing defenses are however designed for specific attacks or biased by predefined heuristic rules. Towards more comprehensive mitigation, we first systematically inspect the mechanism of recent appearing attacks: Their common weaknesses are observed in crafting fake obstacles which (i) have obvious differences in the local parts compared with real obstacles and (ii) violate the physical relation between depth and point density. In this paper, we propose a novel plug-and-play defensive module which works by side of a trained LiDAR-based object detector to eliminate forged obstacles where a major proportion of local parts have low objectness, i.e., to what degree it belongs to a real object. At the core of our module is a local objectness predictor, which explicitly incorporates the depth information to model the relation between depth and point density, and predicts each local part of an obstacle with an objectness score. Extensive experiments show, our proposed defense eliminates at least 70% cars forged by three known appearing attacks in most cases, while, for the best previous defense, less than 30% forged cars are eliminated. Meanwhile, under the same circumstance, our defense incurs less overhead for AP/precision on cars compared with existing defenses. Furthermore, We validate the effectiveness of our proposed defense on simulation-based closed-loop control driving tests in the open-source system of Baidu's Apollo.

摘要: 自动驾驶系统依靠3D物体探测器从激光雷达点云中识别可能的障碍物。然而，最近的研究表明，对手可以通过几个伪点(即出现攻击)来伪造预测结果中不存在的汽车。然而，通过去除统计离群值，现有的防御措施是针对特定攻击设计的，或者受到预定义的启发式规则的偏见。为了更全面地缓解，我们首先系统地考察了最近出现的攻击的机制：它们在制作假障碍物方面都有共同的弱点：(I)与真实障碍物相比，局部有明显的差异；(Ii)违反了深度和点密度之间的物理关系。在本文中，我们提出了一种新的即插即用防御模块，它与一个基于LiDAR的训练对象检测器一起工作，以消除大部分局部部分具有低客观性的伪造障碍，即它属于真实对象的程度。该模块的核心是局部客观性预测器，它显式地结合深度信息来建模深度和点密度之间的关系，并用客观性分数来预测障碍物的每个局部部分。广泛的实验表明，在大多数情况下，我们提出的防御措施消除了至少70%由三次已知出现的攻击所伪造的汽车，而对于之前最好的防御措施，只有不到30%的伪造汽车被消除。同时，在相同的情况下，与现有的防御相比，我们的防御在汽车上的AP/精度方面产生了更少的开销。此外，我们在百度Apollo开源系统的基于模拟的闭环控制驾驶测试中验证了所提出的防御措施的有效性。



