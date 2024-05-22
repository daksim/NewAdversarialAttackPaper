# Latest Adversarial Attack Papers
**update at 2024-05-22 15:27:35**

翻译来自 https://cloud.tencent.com/document/product/551/15619

## **1. Inexact Unlearning Needs More Careful Evaluations to Avoid a False Sense of Privacy**

不精确的遗忘需要更仔细的评估，以避免错误的隐私感 cs.LG

**SubmitDate**: 2024-05-21    [abs](http://arxiv.org/abs/2403.01218v3) [paper-pdf](http://arxiv.org/pdf/2403.01218v3)

**Authors**: Jamie Hayes, Ilia Shumailov, Eleni Triantafillou, Amr Khalifa, Nicolas Papernot

**Abstract**: The high cost of model training makes it increasingly desirable to develop techniques for unlearning. These techniques seek to remove the influence of a training example without having to retrain the model from scratch. Intuitively, once a model has unlearned, an adversary that interacts with the model should no longer be able to tell whether the unlearned example was included in the model's training set or not. In the privacy literature, this is known as membership inference. In this work, we discuss adaptations of Membership Inference Attacks (MIAs) to the setting of unlearning (leading to their "U-MIA" counterparts). We propose a categorization of existing U-MIAs into "population U-MIAs", where the same attacker is instantiated for all examples, and "per-example U-MIAs", where a dedicated attacker is instantiated for each example. We show that the latter category, wherein the attacker tailors its membership prediction to each example under attack, is significantly stronger. Indeed, our results show that the commonly used U-MIAs in the unlearning literature overestimate the privacy protection afforded by existing unlearning techniques on both vision and language models. Our investigation reveals a large variance in the vulnerability of different examples to per-example U-MIAs. In fact, several unlearning algorithms lead to a reduced vulnerability for some, but not all, examples that we wish to unlearn, at the expense of increasing it for other examples. Notably, we find that the privacy protection for the remaining training examples may worsen as a consequence of unlearning. We also discuss the fundamental difficulty of equally protecting all examples using existing unlearning schemes, due to the different rates at which examples are unlearned. We demonstrate that naive attempts at tailoring unlearning stopping criteria to different examples fail to alleviate these issues.

摘要: 模型训练的高昂成本使得开发忘却学习的技术变得越来越受欢迎。这些技术寻求消除训练示例的影响，而不必从头开始重新训练模型。直观地说，一旦模型取消学习，与该模型交互的对手应该不再能够判断未学习的示例是否包括在该模型的训练集中。在隐私文献中，这被称为成员关系推断。在这项工作中，我们讨论了成员关系推理攻击(MIA)对遗忘环境的适应(导致它们的U-MIA对应)。我们建议将现有的U-MIA分类为“群体U-MIA”，其中针对所有示例实例化相同的攻击者，以及“每示例U-MIA”，其中针对每个示例实例化一个专门的攻击者。我们表明，后一类，其中攻击者根据每个被攻击的例子定制其成员预测，明显更强。事实上，我们的结果表明，遗忘文献中常用的U-MIA高估了现有遗忘技术在视觉和语言模型上提供的隐私保护。我们的调查显示，不同示例对每个示例的U-MIA的脆弱性存在很大差异。事实上，几种忘记算法降低了我们希望忘记的一些(但不是所有)示例的脆弱性，但代价是增加了其他示例的脆弱性。值得注意的是，我们发现，由于遗忘，其余训练样本的隐私保护可能会恶化。我们还讨论了使用现有的遗忘方案平等地保护所有例子的基本困难，因为例子被遗忘的比率不同。我们证明，根据不同的例子调整遗忘停止标准的天真尝试无法缓解这些问题。



## **2. Adversarial Attacks and Defenses in Automated Control Systems: A Comprehensive Benchmark**

自动控制系统中的对抗性攻击和防御：全面的基准 cs.LG

**SubmitDate**: 2024-05-21    [abs](http://arxiv.org/abs/2403.13502v3) [paper-pdf](http://arxiv.org/pdf/2403.13502v3)

**Authors**: Vitaliy Pozdnyakov, Aleksandr Kovalenko, Ilya Makarov, Mikhail Drobyshevskiy, Kirill Lukyanov

**Abstract**: Integrating machine learning into Automated Control Systems (ACS) enhances decision-making in industrial process management. One of the limitations to the widespread adoption of these technologies in industry is the vulnerability of neural networks to adversarial attacks. This study explores the threats in deploying deep learning models for fault diagnosis in ACS using the Tennessee Eastman Process dataset. By evaluating three neural networks with different architectures, we subject them to six types of adversarial attacks and explore five different defense methods. Our results highlight the strong vulnerability of models to adversarial samples and the varying effectiveness of defense strategies. We also propose a novel protection approach by combining multiple defense methods and demonstrate it's efficacy. This research contributes several insights into securing machine learning within ACS, ensuring robust fault diagnosis in industrial processes.

摘要: 将机器学习集成到自动化控制系统（ACS）中，增强了工业过程管理中的决策。这些技术在工业中广泛采用的局限性之一是神经网络容易受到对抗攻击。本研究探索了使用田纳西州伊士曼Process数据集在ACS中部署深度学习模型进行故障诊断的威胁。通过评估具有不同架构的三个神经网络，我们将它们置于六种类型的对抗攻击中，并探索五种不同的防御方法。我们的结果凸显了模型对对抗样本的强烈脆弱性以及防御策略的不同有效性。我们还提出了一种通过结合多种防御方法的新型保护方法，并证明了其有效性。这项研究为确保ACS内的机器学习提供了多项见解，确保工业流程中的稳健故障诊断。



## **3. Rethinking the Vulnerabilities of Face Recognition Systems:From a Practical Perspective**

重新思考面部识别系统的漏洞：从实践的角度 cs.CR

19 pages

**SubmitDate**: 2024-05-21    [abs](http://arxiv.org/abs/2405.12786v1) [paper-pdf](http://arxiv.org/pdf/2405.12786v1)

**Authors**: Jiahao Chen, Zhiqiang Shen, Yuwen Pu, Chunyi Zhou, Shouling Ji

**Abstract**: Face Recognition Systems (FRS) have increasingly integrated into critical applications, including surveillance and user authentication, highlighting their pivotal role in modern security systems. Recent studies have revealed vulnerabilities in FRS to adversarial (e.g., adversarial patch attacks) and backdoor attacks (e.g., training data poisoning), raising significant concerns about their reliability and trustworthiness. Previous studies primarily focus on traditional adversarial or backdoor attacks, overlooking the resource-intensive or privileged-manipulation nature of such threats, thus limiting their practical generalization, stealthiness, universality and robustness. Correspondingly, in this paper, we delve into the inherent vulnerabilities in FRS through user studies and preliminary explorations. By exploiting these vulnerabilities, we identify a novel attack, facial identity backdoor attack dubbed FIBA, which unveils a potentially more devastating threat against FRS:an enrollment-stage backdoor attack. FIBA circumvents the limitations of traditional attacks, enabling broad-scale disruption by allowing any attacker donning a specific trigger to bypass these systems. This implies that after a single, poisoned example is inserted into the database, the corresponding trigger becomes a universal key for any attackers to spoof the FRS. This strategy essentially challenges the conventional attacks by initiating at the enrollment stage, dramatically transforming the threat landscape by poisoning the feature database rather than the training data.

摘要: 人脸识别系统(FRS)越来越多地集成到包括监控和用户身份验证在内的关键应用中，突显了它们在现代安全系统中的关键作用。最近的研究发现，FRS对对抗性攻击(例如，对抗性补丁攻击)和后门攻击(例如，训练数据中毒)的脆弱性，引起了人们对其可靠性和可信性的严重担忧。以往的研究主要集中于传统的对抗性攻击或后门攻击，忽略了此类威胁的资源密集型或特权操纵性，从而限制了它们的实用通用性、隐蔽性、普遍性和健壮性。相应地，在本文中，我们通过用户研究和初步探索，深入研究了FRS的固有漏洞。通过利用这些漏洞，我们确定了一种新型的攻击，即面部识别后门攻击，称为FIBA，它揭示了对FRS的一个潜在的更具破坏性的威胁：注册阶段的后门攻击。FIBA绕过了传统攻击的限制，允许任何使用特定触发器的攻击者绕过这些系统，从而实现广泛的破坏。这意味着在将单个有毒示例插入数据库后，相应的触发器将成为任何攻击者欺骗FRS的通用密钥。该策略实质上是通过在注册阶段发起攻击来挑战传统攻击，通过毒化特征数据库而不是训练数据来极大地改变威胁格局。



## **4. Generative AI and Large Language Models for Cyber Security: All Insights You Need**

网络安全的生成性人工智能和大型语言模型：您需要的所有见解 cs.CR

50 pages, 8 figures

**SubmitDate**: 2024-05-21    [abs](http://arxiv.org/abs/2405.12750v1) [paper-pdf](http://arxiv.org/pdf/2405.12750v1)

**Authors**: Mohamed Amine Ferrag, Fatima Alwahedi, Ammar Battah, Bilel Cherif, Abdechakour Mechri, Norbert Tihanyi

**Abstract**: This paper provides a comprehensive review of the future of cybersecurity through Generative AI and Large Language Models (LLMs). We explore LLM applications across various domains, including hardware design security, intrusion detection, software engineering, design verification, cyber threat intelligence, malware detection, and phishing detection. We present an overview of LLM evolution and its current state, focusing on advancements in models such as GPT-4, GPT-3.5, Mixtral-8x7B, BERT, Falcon2, and LLaMA. Our analysis extends to LLM vulnerabilities, such as prompt injection, insecure output handling, data poisoning, DDoS attacks, and adversarial instructions. We delve into mitigation strategies to protect these models, providing a comprehensive look at potential attack scenarios and prevention techniques. Furthermore, we evaluate the performance of 42 LLM models in cybersecurity knowledge and hardware security, highlighting their strengths and weaknesses. We thoroughly evaluate cybersecurity datasets for LLM training and testing, covering the lifecycle from data creation to usage and identifying gaps for future research. In addition, we review new strategies for leveraging LLMs, including techniques like Half-Quadratic Quantization (HQQ), Reinforcement Learning with Human Feedback (RLHF), Direct Preference Optimization (DPO), Quantized Low-Rank Adapters (QLoRA), and Retrieval-Augmented Generation (RAG). These insights aim to enhance real-time cybersecurity defenses and improve the sophistication of LLM applications in threat detection and response. Our paper provides a foundational understanding and strategic direction for integrating LLMs into future cybersecurity frameworks, emphasizing innovation and robust model deployment to safeguard against evolving cyber threats.

摘要: 本文通过生成式人工智能和大型语言模型(LLMS)对网络安全的未来进行了全面的回顾。我们探索了LLM在不同领域的应用，包括硬件设计安全、入侵检测、软件工程、设计验证、网络威胁情报、恶意软件检测和网络钓鱼检测。我们概述了LLM的演化和现状，重点介绍了GPT-4、GPT-3.5、Mixtral-8x7B、BERT、Falcon2和Llama等模型的进展。我们的分析扩展到LLM漏洞，如快速注入、不安全的输出处理、数据中毒、DDoS攻击和敌意指令。我们深入研究缓解策略以保护这些模型，提供对潜在攻击场景和预防技术的全面了解。此外，我们评估了42个LLM模型在网络安全知识和硬件安全方面的性能，突出了它们的优势和劣势。我们为LLM培训和测试彻底评估网络安全数据集，涵盖从数据创建到使用的整个生命周期，并为未来的研究确定差距。此外，我们还回顾了利用LLMS的新策略，包括半二次量化(HQQ)、带人反馈的强化学习(RLHF)、直接偏好优化(DPO)、量化低阶适配器(QLoRA)和检索增强生成(RAG)。这些见解旨在增强实时网络安全防御，并提高LLM应用程序在威胁检测和响应方面的复杂性。我们的论文为将低成本管理系统整合到未来的网络安全框架中提供了一个基础性的理解和战略方向，强调创新和稳健的模型部署，以防范不断演变的网络威胁。



## **5. A GAN-Based Data Poisoning Attack Against Federated Learning Systems and Its Countermeasure**

针对联邦学习系统的基于GAN的数据中毒攻击及其对策 cs.CR

18 pages, 16 figures

**SubmitDate**: 2024-05-21    [abs](http://arxiv.org/abs/2405.11440v2) [paper-pdf](http://arxiv.org/pdf/2405.11440v2)

**Authors**: Wei Sun, Bo Gao, Ke Xiong, Yuwei Wang

**Abstract**: As a distributed machine learning paradigm, federated learning (FL) is collaboratively carried out on privately owned datasets but without direct data access. Although the original intention is to allay data privacy concerns, "available but not visible" data in FL potentially brings new security threats, particularly poisoning attacks that target such "not visible" local data. Initial attempts have been made to conduct data poisoning attacks against FL systems, but cannot be fully successful due to their high chance of causing statistical anomalies. To unleash the potential for truly "invisible" attacks and build a more deterrent threat model, in this paper, a new data poisoning attack model named VagueGAN is proposed, which can generate seemingly legitimate but noisy poisoned data by untraditionally taking advantage of generative adversarial network (GAN) variants. Capable of manipulating the quality of poisoned data on demand, VagueGAN enables to trade-off attack effectiveness and stealthiness. Furthermore, a cost-effective countermeasure named Model Consistency-Based Defense (MCD) is proposed to identify GAN-poisoned data or models after finding out the consistency of GAN outputs. Extensive experiments on multiple datasets indicate that our attack method is generally much more stealthy as well as more effective in degrading FL performance with low complexity. Our defense method is also shown to be more competent in identifying GAN-poisoned data or models. The source codes are publicly available at \href{https://github.com/SSssWEIssSS/VagueGAN-Data-Poisoning-Attack-and-Its-Countermeasure}{https://github.com/SSssWEIssSS/VagueGAN-Data-Poisoning-Attack-and-Its-Countermeasure}.

摘要: 作为一种分布式机器学习范式，联合学习(FL)是在私有数据集上协作进行的，但不需要直接访问数据。虽然初衷是为了缓解数据隐私问题，但FL中的“可用但不可见”数据可能会带来新的安全威胁，特别是针对此类“不可见”本地数据的中毒攻击。已经进行了针对FL系统的数据中毒攻击的初步尝试，但由于造成统计异常的可能性很高，因此不能完全成功。为了释放真正隐形攻击的可能性，建立更具威慑力的威胁模型，提出了一种新的数据中毒攻击模型VagueGAN，该模型通过非传统地利用生成性对手网络(GAN)变体来生成看似合法但含有噪声的有毒数据。VagueGAN能够按需操纵有毒数据的质量，从而能够在攻击效率和隐蔽性之间进行权衡。此外，还提出了一种基于模型一致性防御(MCD)的高性价比对策，用于在发现GaN输出的一致性之后识别GaN中毒数据或模型。在多个数据集上的大量实验表明，我们的攻击方法通常更隐蔽，并且在降低复杂度的情况下更有效地降低了FL性能。我们的防御方法也被证明在识别GaN中毒数据或模型方面更有能力。源代码可在\href{https://github.com/SSssWEIssSS/VagueGAN-Data-Poisoning-Attack-and-Its-Countermeasure}{https://github.com/SSssWEIssSS/VagueGAN-Data-Poisoning-Attack-and-Its-Countermeasure}.上公开获取



## **6. How to Train a Backdoor-Robust Model on a Poisoned Dataset without Auxiliary Data?**

如何在没有辅助数据的情况下在中毒数据集中训练后门稳健模型？ cs.CR

13 pages, under review

**SubmitDate**: 2024-05-21    [abs](http://arxiv.org/abs/2405.12719v1) [paper-pdf](http://arxiv.org/pdf/2405.12719v1)

**Authors**: Yuwen Pu, Jiahao Chen, Chunyi Zhou, Zhou Feng, Qingming Li, Chunqiang Hu, Shouling Ji

**Abstract**: Backdoor attacks have attracted wide attention from academia and industry due to their great security threat to deep neural networks (DNN). Most of the existing methods propose to conduct backdoor attacks by poisoning the training dataset with different strategies, so it's critical to identify the poisoned samples and then train a clean model on the unreliable dataset in the context of defending backdoor attacks. Although numerous backdoor countermeasure researches are proposed, their inherent weaknesses render them limited in practical scenarios, such as the requirement of enough clean samples, unstable defense performance under various attack conditions, poor defense performance against adaptive attacks, and so on.Therefore, in this paper, we are committed to overcome the above limitations and propose a more practical backdoor defense method. Concretely, we first explore the inherent relationship between the potential perturbations and the backdoor trigger, and the theoretical analysis and experimental results demonstrate that the poisoned samples perform more robustness to perturbation than the clean ones. Then, based on our key explorations, we introduce AdvrBD, an Adversarial perturbation-based and robust Backdoor Defense framework, which can effectively identify the poisoned samples and train a clean model on the poisoned dataset. Constructively, our AdvrBD eliminates the requirement for any clean samples or knowledge about the poisoned dataset (e.g., poisoning ratio), which significantly improves the practicality in real-world scenarios.

摘要: 后门攻击因其对深度神经网络(DNN)的巨大安全威胁而受到学术界和工业界的广泛关注。现有的大多数方法都是通过对训练数据集使用不同的策略进行中毒来进行后门攻击，因此在防御后门攻击的背景下，识别中毒样本并在不可靠的数据集上训练一个干净的模型是至关重要的。虽然已经提出了大量的后门对抗研究，但其固有的缺陷使其在实际场景中受到限制，如需要足够的清洁样本，在各种攻击条件下的防御性能不稳定，对自适应攻击的防御性能较差等，因此，本文致力于克服上述局限性，提出一种更实用的后门防御方法。具体地说，我们首先探讨了潜在扰动和后门触发之间的内在联系，理论分析和实验结果表明，中毒样本比干净样本对扰动具有更强的鲁棒性。然后，在重点探索的基础上，提出了一种基于对抗性扰动的健壮后门防御框架AdvrBD，它可以有效地识别有毒样本，并在有毒数据集上训练一个干净的模型。建设性地说，我们的AdvrBD不需要任何干净的样本或关于有毒数据集的知识(例如，投毒率)，这显著提高了现实世界场景中的实用性。



## **7. Robust Classification via a Single Diffusion Model**

通过单一扩散模型的稳健分类 cs.CV

Accepted by ICML 2024

**SubmitDate**: 2024-05-21    [abs](http://arxiv.org/abs/2305.15241v2) [paper-pdf](http://arxiv.org/pdf/2305.15241v2)

**Authors**: Huanran Chen, Yinpeng Dong, Zhengyi Wang, Xiao Yang, Chengqi Duan, Hang Su, Jun Zhu

**Abstract**: Diffusion models have been applied to improve adversarial robustness of image classifiers by purifying the adversarial noises or generating realistic data for adversarial training. However, diffusion-based purification can be evaded by stronger adaptive attacks while adversarial training does not perform well under unseen threats, exhibiting inevitable limitations of these methods. To better harness the expressive power of diffusion models, this paper proposes Robust Diffusion Classifier (RDC), a generative classifier that is constructed from a pre-trained diffusion model to be adversarially robust. RDC first maximizes the data likelihood of a given input and then predicts the class probabilities of the optimized input using the conditional likelihood estimated by the diffusion model through Bayes' theorem. To further reduce the computational cost, we propose a new diffusion backbone called multi-head diffusion and develop efficient sampling strategies. As RDC does not require training on particular adversarial attacks, we demonstrate that it is more generalizable to defend against multiple unseen threats. In particular, RDC achieves $75.67\%$ robust accuracy against various $\ell_\infty$ norm-bounded adaptive attacks with $\epsilon_\infty=8/255$ on CIFAR-10, surpassing the previous state-of-the-art adversarial training models by $+4.77\%$. The results highlight the potential of generative classifiers by employing pre-trained diffusion models for adversarial robustness compared with the commonly studied discriminative classifiers. Code is available at \url{https://github.com/huanranchen/DiffusionClassifier}.

摘要: 扩散模型已被应用于通过净化对抗性噪声或生成用于对抗性训练的真实数据来提高图像分类器的对抗性鲁棒性。然而，基于扩散的净化方法可以通过更强的自适应攻击来规避，而对抗性训练在看不见的威胁下表现不佳，显示出这些方法不可避免的局限性。为了更好地利用扩散模型的表达能力，提出了稳健扩散分类器(RDC)，它是一种生成式分类器，由预先训练的扩散模型构造而成，具有相反的鲁棒性。RDC首先最大化给定输入的数据似然，然后利用扩散模型通过贝叶斯定理估计的条件似然来预测优化输入的类别概率。为了进一步降低计算成本，我们提出了一种新的扩散骨干，称为多头扩散，并开发了高效的采样策略。由于RDC不需要关于特定对手攻击的培训，我们证明了防御多个看不见的威胁更具普遍性。特别是，在CIFAR-10上，RDC对各种有界自适应攻击的稳健准确率达到了75.67美元，其中$epsilon_INFTY=8/255$，比以前最先进的对抗性训练模型高出$+4.77$。与通常研究的判别分类器相比，这些结果突出了生成式分类器通过使用预训练的扩散模型来提高对抗稳健性的潜力。代码可在\url{https://github.com/huanranchen/DiffusionClassifier}.上找到



## **8. Fully Randomized Pointers**

完全随机指针 cs.CR

24 pages, 3 figures

**SubmitDate**: 2024-05-21    [abs](http://arxiv.org/abs/2405.12513v1) [paper-pdf](http://arxiv.org/pdf/2405.12513v1)

**Authors**: Gregory J. Duck, Sai Dhawal Phaye, Roland H. C. Yap, Trevor E. Carlson

**Abstract**: Software security continues to be a critical concern for programs implemented in low-level programming languages such as C and C++. Many defenses have been proposed in the current literature, each with different trade-offs including performance, compatibility, and attack resistance. One general class of defense is pointer randomization or authentication, where invalid object access (e.g., memory errors) is obfuscated or denied. Many defenses rely on the program termination (e.g., crashing) to abort attacks, with the implicit assumption that an adversary cannot "brute force" the defense with multiple attack attempts. However, such assumptions do not always hold, such as hardware speculative execution attacks or network servers configured to restart on error. In such cases, we argue that most existing defenses provide only weak effective security.   In this paper, we propose Fully Randomized Pointers (FRP) as a stronger memory error defense that is resistant to even brute force attacks. The key idea is to fully randomize pointer bits -- as much as possible while also preserving binary compatibility -- rendering the relationships between pointers highly unpredictable. Furthermore, the very high degree of randomization renders brute force attacks impractical -- providing strong effective security compared to existing work. We design a new FRP encoding that is: (1) compatible with existing binary code (without recompilation); (2) decoupled from the underlying object layout; and (3) can be efficiently decoded on-the-fly to the underlying memory address. We prototype FRP in the form of a software implementation (BlueFat) to test security and compatibility, and a proof-of-concept hardware implementation (GreenFat) to evaluate performance. We show that FRP is secure, practical, and compatible at the binary level, while a hardware implementation can achieve low performance overheads (<10%).

摘要: 对于使用低级编程语言(如C和C++)实现的程序来说，软件安全性仍然是一个关键问题。在当前的文献中已经提出了许多防御措施，每种防御措施都具有不同的权衡，包括性能、兼容性和抗攻击能力。一种常见的防御类别是指针随机化或身份验证，在这种情况下，无效对象访问(例如，内存错误)被混淆或拒绝。许多防御依赖于程序终止(例如崩溃)来中止攻击，并隐含地假设对手不能通过多次攻击尝试来“野蛮地强迫”防御。然而，这样的假设并不总是成立的，例如硬件推测性执行攻击或配置为在出错时重新启动的网络服务器。在这种情况下，我们认为，大多数现有的防御措施只提供了薄弱的有效安全。在本文中，我们提出了完全随机化指针(FRP)作为一种更强的内存错误防御机制，它甚至可以抵抗暴力攻击。其关键思想是完全随机化指针位--尽可能多地同时保持二进制兼容性--使指针之间的关系高度不可预测。此外，非常高的随机性使暴力攻击变得不切实际--与现有工作相比，提供了强大的有效安全性。我们设计了一种新的FRP编码：(1)与现有的二进制代码兼容(无需重新编译)；(2)与底层对象布局解耦；(3)可以高效地动态解码到底层内存地址。我们以软件实现(BlueFat)的形式构建FRP原型以测试安全性和兼容性，并以概念验证硬件实现(GreenFat)的形式评估性能。我们证明了FRP是安全的、实用的和二进制级兼容的，而硬件实现可以获得较低的性能开销(<10%)。



## **9. Rethinking Robustness Assessment: Adversarial Attacks on Learning-based Quadrupedal Locomotion Controllers**

重新思考稳健性评估：对基于学习的四足运动控制器的对抗攻击 cs.RO

RSS 2024

**SubmitDate**: 2024-05-21    [abs](http://arxiv.org/abs/2405.12424v1) [paper-pdf](http://arxiv.org/pdf/2405.12424v1)

**Authors**: Fan Shi, Chong Zhang, Takahiro Miki, Joonho Lee, Marco Hutter, Stelian Coros

**Abstract**: Legged locomotion has recently achieved remarkable success with the progress of machine learning techniques, especially deep reinforcement learning (RL). Controllers employing neural networks have demonstrated empirical and qualitative robustness against real-world uncertainties, including sensor noise and external perturbations. However, formally investigating the vulnerabilities of these locomotion controllers remains a challenge. This difficulty arises from the requirement to pinpoint vulnerabilities across a long-tailed distribution within a high-dimensional, temporally sequential space. As a first step towards quantitative verification, we propose a computational method that leverages sequential adversarial attacks to identify weaknesses in learned locomotion controllers. Our research demonstrates that, even state-of-the-art robust controllers can fail significantly under well-designed, low-magnitude adversarial sequence. Through experiments in simulation and on the real robot, we validate our approach's effectiveness, and we illustrate how the results it generates can be used to robustify the original policy and offer valuable insights into the safety of these black-box policies.

摘要: 近年来，随着机器学习技术的进步，特别是深度强化学习(RL)的发展，腿部运动已经取得了显著的成功。采用神经网络的控制器对真实世界的不确定性表现出了经验和定性的鲁棒性，包括传感器噪声和外部扰动。然而，正式调查这些运动控制器的漏洞仍然是一个挑战。这一困难源于需要在高维的、时间顺序的空间内精确定位跨长尾分布的漏洞。作为定量验证的第一步，我们提出了一种计算方法，该方法利用顺序对抗性攻击来识别学习的运动控制器中的弱点。我们的研究表明，即使是最先进的鲁棒控制器，在设计良好的低幅度对抗性序列下也会显著失效。通过仿真实验和在真实机器人上的实验，我们验证了该方法的有效性，并说明了它所产生的结果如何被用来证明原始策略的健壮性，并为这些黑盒策略的安全性提供了有价值的见解。



## **10. Rethinking PGD Attack: Is Sign Function Necessary?**

重新思考PVD攻击：符号功能是否必要？ cs.LG

**SubmitDate**: 2024-05-21    [abs](http://arxiv.org/abs/2312.01260v2) [paper-pdf](http://arxiv.org/pdf/2312.01260v2)

**Authors**: Junjie Yang, Tianlong Chen, Xuxi Chen, Zhangyang Wang, Yingbin Liang

**Abstract**: Neural networks have demonstrated success in various domains, yet their performance can be significantly degraded by even a small input perturbation. Consequently, the construction of such perturbations, known as adversarial attacks, has gained significant attention, many of which fall within "white-box" scenarios where we have full access to the neural network. Existing attack algorithms, such as the projected gradient descent (PGD), commonly take the sign function on the raw gradient before updating adversarial inputs, thereby neglecting gradient magnitude information. In this paper, we present a theoretical analysis of how such sign-based update algorithm influences step-wise attack performance, as well as its caveat. We also interpret why previous attempts of directly using raw gradients failed. Based on that, we further propose a new raw gradient descent (RGD) algorithm that eliminates the use of sign. Specifically, we convert the constrained optimization problem into an unconstrained one, by introducing a new hidden variable of non-clipped perturbation that can move beyond the constraint. The effectiveness of the proposed RGD algorithm has been demonstrated extensively in experiments, outperforming PGD and other competitors in various settings, without incurring any additional computational overhead. The codes is available in https://github.com/JunjieYang97/RGD.

摘要: 神经网络已经在各个领域取得了成功，但即使是很小的输入扰动也会显著降低其性能。因此，这种被称为对抗性攻击的扰动的构造得到了极大的关注，其中许多都属于我们可以完全访问神经网络的“白盒”情景。现有的攻击算法，如投影梯度下降(PGD)算法，通常在更新敌方输入之前对原始梯度取符号函数，从而忽略了梯度大小信息。本文从理论上分析了这种基于符号的更新算法对分步攻击性能的影响，并给出了相应的警告。我们还解释了为什么以前直接使用原始梯度的尝试失败了。在此基础上，进一步提出了一种新的原始梯度下降(RGD)算法，该算法省去了符号的使用。具体地说，我们通过引入一个可以超越约束的非剪裁扰动的新的隐变量，将约束优化问题转化为无约束优化问题。所提出的RGD算法的有效性已经在实验中得到了广泛的证明，在不引起任何额外计算开销的情况下，在不同环境下的性能优于PGD和其他竞争对手。这些代码可以在https://github.com/JunjieYang97/RGD.中找到



## **11. Hacking Predictors Means Hacking Cars: Using Sensitivity Analysis to Identify Trajectory Prediction Vulnerabilities for Autonomous Driving Security**

黑客预测器意味着黑客汽车：使用敏感性分析来识别自动驾驶安全的轨迹预测漏洞 cs.CR

10 pages, 5 figures, 1 tables

**SubmitDate**: 2024-05-20    [abs](http://arxiv.org/abs/2401.10313v2) [paper-pdf](http://arxiv.org/pdf/2401.10313v2)

**Authors**: Marsalis Gibson, David Babazadeh, Claire Tomlin, Shankar Sastry

**Abstract**: Adversarial attacks on learning-based multi-modal trajectory predictors have already been demonstrated. However, there are still open questions about the effects of perturbations on inputs other than state histories, and how these attacks impact downstream planning and control. In this paper, we conduct a sensitivity analysis on two trajectory prediction models, Trajectron++ and AgentFormer. The analysis reveals that between all inputs, almost all of the perturbation sensitivities for both models lie only within the most recent position and velocity states. We additionally demonstrate that, despite dominant sensitivity on state history perturbations, an undetectable image map perturbation made with the Fast Gradient Sign Method can induce large prediction error increases in both models, revealing that these trajectory predictors are, in fact, susceptible to image-based attacks. Using an optimization-based planner and example perturbations crafted from sensitivity results, we show how these attacks can cause a vehicle to come to a sudden stop from moderate driving speeds.

摘要: 对基于学习的多模式轨迹预测器的对抗性攻击已经被证明。然而，对于扰动对除状态历史之外的输入的影响，以及这些攻击如何影响下游规划和控制，仍然存在悬而未决的问题。本文对两种弹道预测模型Trajectron++和AgentFormer进行了灵敏度分析。分析表明，在所有输入之间，两个模型的几乎所有摄动灵敏度都只存在于最近的位置和速度状态。此外，我们还证明了，尽管对状态历史扰动的主要敏感性，但用快速梯度符号方法进行的不可检测的图像映射扰动可以在两个模型中导致预测误差的大幅增加，这表明这些轨迹预测器实际上容易受到基于图像的攻击。使用基于优化的计划器和根据敏感度结果制作的示例扰动，我们展示了这些攻击如何导致车辆在中等速度下突然停止。



## **12. Optimizing Sensor Network Design for Multiple Coverage**

优化传感器网络设计以实现多覆盖 cs.LG

**SubmitDate**: 2024-05-20    [abs](http://arxiv.org/abs/2405.09096v2) [paper-pdf](http://arxiv.org/pdf/2405.09096v2)

**Authors**: Lukas Taus, Yen-Hsi Richard Tsai

**Abstract**: Sensor placement optimization methods have been studied extensively. They can be applied to a wide range of applications, including surveillance of known environments, optimal locations for 5G towers, and placement of missile defense systems. However, few works explore the robustness and efficiency of the resulting sensor network concerning sensor failure or adversarial attacks. This paper addresses this issue by optimizing for the least number of sensors to achieve multiple coverage of non-simply connected domains by a prescribed number of sensors. We introduce a new objective function for the greedy (next-best-view) algorithm to design efficient and robust sensor networks and derive theoretical bounds on the network's optimality. We further introduce a Deep Learning model to accelerate the algorithm for near real-time computations. The Deep Learning model requires the generation of training examples. Correspondingly, we show that understanding the geometric properties of the training data set provides important insights into the performance and training process of deep learning techniques. Finally, we demonstrate that a simple parallel version of the greedy approach using a simpler objective can be highly competitive.

摘要: 传感器布局优化方法得到了广泛的研究。它们可以应用于广泛的应用，包括对已知环境的监视，5G塔的最佳位置，以及导弹防御系统的布置。然而，很少有文献探讨传感器网络在传感器故障或敌意攻击下的健壮性和有效性。本文通过优化最少的传感器数量来解决这一问题，从而在规定的传感器数量下实现对非单连通区域的多次覆盖。为了设计高效、健壮的传感器网络，我们为贪婪(Next-Best-view)算法引入了一个新的目标函数，并给出了网络最优性的理论界。我们进一步引入了深度学习模型来加速算法，以实现近实时计算。深度学习模型需要生成训练实例。相应地，我们表明，理解训练数据集的几何属性可以为深度学习技术的性能和训练过程提供重要的见解。最后，我们证明了贪婪方法的简单并行版本使用更简单的目标可以具有很强的竞争力。



## **13. Efficient Model-Stealing Attacks Against Inductive Graph Neural Networks**

针对归纳图神经网络的有效模型窃取攻击 cs.LG

arXiv admin note: text overlap with arXiv:2112.08331 by other authors

**SubmitDate**: 2024-05-20    [abs](http://arxiv.org/abs/2405.12295v1) [paper-pdf](http://arxiv.org/pdf/2405.12295v1)

**Authors**: Marcin Podhajski, Jan Dubiński, Franziska Boenisch, Adam Dziedzic, Agnieszka Pregowska, Tomasz Michalak

**Abstract**: Graph Neural Networks (GNNs) are recognized as potent tools for processing real-world data organized in graph structures. Especially inductive GNNs, which enable the processing of graph-structured data without relying on predefined graph structures, are gaining importance in an increasingly wide variety of applications. As these networks demonstrate proficiency across a range of tasks, they become lucrative targets for model-stealing attacks where an adversary seeks to replicate the functionality of the targeted network. A large effort has been made to develop model-stealing attacks that focus on models trained with images and texts. However, little attention has been paid to GNNs trained on graph data. This paper introduces a novel method for unsupervised model-stealing attacks against inductive GNNs, based on graph contrasting learning and spectral graph augmentations to efficiently extract information from the target model. The proposed attack is thoroughly evaluated on six datasets. The results show that this approach demonstrates a higher level of efficiency compared to existing stealing attacks. More concretely, our attack outperforms the baseline on all benchmarks achieving higher fidelity and downstream accuracy of the stolen model while requiring fewer queries sent to the target model.

摘要: 图神经网络(GNN)被认为是处理以图结构组织的真实世界数据的有力工具。尤其是感应式GNN，它能够在不依赖于预定义的图结构的情况下处理图结构的数据，在越来越广泛的应用中正变得越来越重要。由于这些网络在一系列任务中表现出熟练程度，它们成为窃取模型攻击的有利可图的目标，在这种攻击中，对手试图复制目标网络的功能。已经做出了大量努力来开发窃取模型的攻击，这些攻击集中在使用图像和文本训练的模型上。然而，对以图表数据为基础的全球网络的关注很少。提出了一种基于图对比学习和谱图扩充的非监督模型窃取方法，有效地从目标模型中提取信息。对提出的攻击在六个数据集上进行了彻底的评估。实验结果表明，与现有的窃取攻击相比，该方法具有更高的效率。更具体地说，我们的攻击在所有基准上都超过了基线，实现了被盗模型的更高保真度和下游精度，同时需要发送到目标模型的查询更少。



## **14. EGAN: Evolutional GAN for Ransomware Evasion**

EGAN：勒索软件规避的进化GAN cs.CR

**SubmitDate**: 2024-05-20    [abs](http://arxiv.org/abs/2405.12266v1) [paper-pdf](http://arxiv.org/pdf/2405.12266v1)

**Authors**: Daniel Commey, Benjamin Appiah, Bill K. Frimpong, Isaac Osei, Ebenezer N. A. Hammond, Garth V. Crosby

**Abstract**: Adversarial Training is a proven defense strategy against adversarial malware. However, generating adversarial malware samples for this type of training presents a challenge because the resulting adversarial malware needs to remain evasive and functional. This work proposes an attack framework, EGAN, to address this limitation. EGAN leverages an Evolution Strategy and Generative Adversarial Network to select a sequence of attack actions that can mutate a Ransomware file while preserving its original functionality. We tested this framework on popular AI-powered commercial antivirus systems listed on VirusTotal and demonstrated that our framework is capable of bypassing the majority of these systems. Moreover, we evaluated whether the EGAN attack framework can evade other commercial non-AI antivirus solutions. Our results indicate that the adversarial ransomware generated can increase the probability of evading some of them.

摘要: 对抗性训练是一种行之有效的针对对抗性恶意软件的防御策略。然而，为此类训练生成对抗性恶意软件样本存在挑战，因为产生的对抗性恶意软件需要保持规避性和功能性。这项工作提出了一个攻击框架EGAN来解决这一限制。EGAN利用进化策略和生成对抗网络来选择一系列攻击动作，这些动作可以变异勒索软件文件，同时保留其原始功能。我们在Virus Total上列出的流行人工智能驱动的商业防病毒系统上测试了这个框架，并证明我们的框架能够绕过大多数这些系统。此外，我们还评估了EGAN攻击框架是否可以规避其他商业非AI防病毒解决方案。我们的结果表明，生成的敌对勒索软件可以增加逃避其中一些勒索软件的可能性。



## **15. GAN-GRID: A Novel Generative Attack on Smart Grid Stability Prediction**

GAN-GRID：对智能电网稳定性预测的新型生成攻击 cs.CR

**SubmitDate**: 2024-05-20    [abs](http://arxiv.org/abs/2405.12076v1) [paper-pdf](http://arxiv.org/pdf/2405.12076v1)

**Authors**: Emad Efatinasab, Alessandro Brighente, Mirco Rampazzo, Nahal Azadi, Mauro Conti

**Abstract**: The smart grid represents a pivotal innovation in modernizing the electricity sector, offering an intelligent, digitalized energy network capable of optimizing energy delivery from source to consumer. It hence represents the backbone of the energy sector of a nation. Due to its central role, the availability of the smart grid is paramount and is hence necessary to have in-depth control of its operations and safety. To this aim, researchers developed multiple solutions to assess the smart grid's stability and guarantee that it operates in a safe state. Artificial intelligence and Machine learning algorithms have proven to be effective measures to accurately predict the smart grid's stability. Despite the presence of known adversarial attacks and potential solutions, currently, there exists no standardized measure to protect smart grids against this threat, leaving them open to new adversarial attacks. In this paper, we propose GAN-GRID a novel adversarial attack targeting the stability prediction system of a smart grid tailored to real-world constraints. Our findings reveal that an adversary armed solely with the stability model's output, devoid of data or model knowledge, can craft data classified as stable with an Attack Success Rate (ASR) of 0.99. Also by manipulating authentic data and sensor values, the attacker can amplify grid issues, potentially undetected due to a compromised stability prediction system. These results underscore the imperative of fortifying smart grid security mechanisms against adversarial manipulation to uphold system stability and reliability.

摘要: 智能电网代表着电力部门现代化的一项关键创新，提供了一个智能、数字化的能源网络，能够优化从来源到用户的能源输送。因此，它代表着一个国家能源部门的中坚力量。由于其核心作用，智能电网的可用性是至关重要的，因此有必要对其运行和安全进行深入控制。为此，研究人员开发了多种解决方案来评估智能电网的稳定性，并确保其在安全状态下运行。人工智能和机器学习算法已被证明是准确预测智能电网稳定性的有效手段。尽管存在已知的对抗性攻击和潜在的解决方案，但目前还没有标准化措施来保护智能电网免受这种威胁，使其容易受到新的对抗性攻击。在本文中，我们提出了一种新的针对智能电网稳定性预测系统的对抗性攻击。我们的发现表明，仅用稳定性模型的输出武装的对手，在缺乏数据或模型知识的情况下，可以伪造被归类为稳定的数据，攻击成功率(ASR)为0.99。此外，通过操纵真实的数据和传感器值，攻击者可以放大网格问题，这些问题可能由于稳定性预测系统受损而未被检测到。这些结果突显了加强智能电网安全机制以防止恶意操纵以维护系统稳定性和可靠性的必要性。



## **16. A Constraint-Enforcing Reward for Adversarial Attacks on Text Classifiers**

文本分类器对抗攻击的约束强制奖励 cs.CL

**SubmitDate**: 2024-05-20    [abs](http://arxiv.org/abs/2405.11904v1) [paper-pdf](http://arxiv.org/pdf/2405.11904v1)

**Authors**: Tom Roth, Inigo Jauregi Unanue, Alsharif Abuadbba, Massimo Piccardi

**Abstract**: Text classifiers are vulnerable to adversarial examples -- correctly-classified examples that are deliberately transformed to be misclassified while satisfying acceptability constraints. The conventional approach to finding adversarial examples is to define and solve a combinatorial optimisation problem over a space of allowable transformations. While effective, this approach is slow and limited by the choice of transformations. An alternate approach is to directly generate adversarial examples by fine-tuning a pre-trained language model, as is commonly done for other text-to-text tasks. This approach promises to be much quicker and more expressive, but is relatively unexplored. For this reason, in this work we train an encoder-decoder paraphrase model to generate a diverse range of adversarial examples. For training, we adopt a reinforcement learning algorithm and propose a constraint-enforcing reward that promotes the generation of valid adversarial examples. Experimental results over two text classification datasets show that our model has achieved a higher success rate than the original paraphrase model, and overall has proved more effective than other competitive attacks. Finally, we show how key design choices impact the generated examples and discuss the strengths and weaknesses of the proposed approach.

摘要: 文本分类器很容易受到对抗性示例的影响--正确分类的示例在满足可接受性约束的同时被故意转换为错误分类。寻找对抗性例子的传统方法是在允许变换的空间上定义和解决组合优化问题。虽然这种方法很有效，但速度很慢，而且受到转换选择的限制。另一种方法是通过微调预先训练的语言模型来直接生成对抗性示例，这是其他文本到文本任务的常见做法。这种方法承诺会更快、更有表现力，但相对来说还没有被探索过。为此，在这项工作中，我们训练一个编码器-解码者转述模型，以生成不同范围的对抗性例子。对于训练，我们采用了一种强化学习算法，并提出了一种约束强制奖励，以促进有效对抗性实例的生成。在两个文本分类数据集上的实验结果表明，我们的模型取得了比原始释义模型更高的成功率，总体上被证明比其他竞争攻击更有效。最后，我们展示了关键的设计选择如何影响生成的实例，并讨论了所提出的方法的优点和缺点。



## **17. Adversarially Diversified Rehearsal Memory (ADRM): Mitigating Memory Overfitting Challenge in Continual Learning**

敌对多元化排练记忆（ADRM）：缓解持续学习中的记忆过度匹配挑战 cs.LG

**SubmitDate**: 2024-05-20    [abs](http://arxiv.org/abs/2405.11829v1) [paper-pdf](http://arxiv.org/pdf/2405.11829v1)

**Authors**: Hikmat Khan, Ghulam Rasool, Nidhal Carla Bouaynaya

**Abstract**: Continual learning focuses on learning non-stationary data distribution without forgetting previous knowledge. Rehearsal-based approaches are commonly used to combat catastrophic forgetting. However, these approaches suffer from a problem called "rehearsal memory overfitting, " where the model becomes too specialized on limited memory samples and loses its ability to generalize effectively. As a result, the effectiveness of the rehearsal memory progressively decays, ultimately resulting in catastrophic forgetting of the learned tasks.   We introduce the Adversarially Diversified Rehearsal Memory (ADRM) to address the memory overfitting challenge. This novel method is designed to enrich memory sample diversity and bolster resistance against natural and adversarial noise disruptions. ADRM employs the FGSM attacks to introduce adversarially modified memory samples, achieving two primary objectives: enhancing memory diversity and fostering a robust response to continual feature drifts in memory samples.   Our contributions are as follows: Firstly, ADRM addresses overfitting in rehearsal memory by employing FGSM to diversify and increase the complexity of the memory buffer. Secondly, we demonstrate that ADRM mitigates memory overfitting and significantly improves the robustness of CL models, which is crucial for safety-critical applications. Finally, our detailed analysis of features and visualization demonstrates that ADRM mitigates feature drifts in CL memory samples, significantly reducing catastrophic forgetting and resulting in a more resilient CL model. Additionally, our in-depth t-SNE visualizations of feature distribution and the quantification of the feature similarity further enrich our understanding of feature representation in existing CL approaches. Our code is publically available at https://github.com/hikmatkhan/ADRM.

摘要: 持续学习侧重于学习非平稳数据分布，而不会忘记先前的知识。基于排练的方法通常被用来对抗灾难性遗忘。然而，这些方法存在一个称为“预演记忆过度匹配”的问题，该模型对有限的记忆样本过于专门化，失去了有效推广的能力。结果，排练记忆的有效性逐渐衰退，最终导致对所学任务的灾难性遗忘。我们引入了对抗性多元化预演记忆(ADRM)来解决记忆过度匹配的挑战。这种新的方法旨在丰富记忆样本的多样性，并增强对自然和对抗性噪声干扰的抵抗力。ADRM利用FGSM攻击引入恶意修改的记忆样本，实现两个主要目标：增强记忆多样性和培养对记忆样本中持续特征漂移的稳健响应。我们的贡献如下：首先，ADRM通过使用FGSM来多样化和增加存储缓冲区的复杂度来解决预演记忆中的过度匹配问题。其次，我们证明了ADRM缓解了内存过度匹配，并显著提高了CL模型的健壮性，这对于安全关键型应用是至关重要的。最后，我们对特征和可视化的详细分析表明，ADRM减少了CL记忆样本中的特征漂移，显著减少了灾难性遗忘，并导致了更具弹性的CL模型。此外，我们对特征分布的深入t-SNE可视化和特征相似性的量化进一步丰富了我们对现有CL方法中的特征表示的理解。我们的代码在https://github.com/hikmatkhan/ADRM.上公开提供



## **18. Fed-Credit: Robust Federated Learning with Credibility Management**

Fed-Credit：具有可信度管理的稳健联邦学习 cs.LG

**SubmitDate**: 2024-05-20    [abs](http://arxiv.org/abs/2405.11758v1) [paper-pdf](http://arxiv.org/pdf/2405.11758v1)

**Authors**: Jiayan Chen, Zhirong Qian, Tianhui Meng, Xitong Gao, Tian Wang, Weijia Jia

**Abstract**: Aiming at privacy preservation, Federated Learning (FL) is an emerging machine learning approach enabling model training on decentralized devices or data sources. The learning mechanism of FL relies on aggregating parameter updates from individual clients. However, this process may pose a potential security risk due to the presence of malicious devices. Existing solutions are either costly due to the use of compute-intensive technology, or restrictive for reasons of strong assumptions such as the prior knowledge of the number of attackers and how they attack. Few methods consider both privacy constraints and uncertain attack scenarios. In this paper, we propose a robust FL approach based on the credibility management scheme, called Fed-Credit. Unlike previous studies, our approach does not require prior knowledge of the nodes and the data distribution. It maintains and employs a credibility set, which weighs the historical clients' contributions based on the similarity between the local models and global model, to adjust the global model update. The subtlety of Fed-Credit is that the time decay and attitudinal value factor are incorporated into the dynamic adjustment of the reputation weights and it boasts a computational complexity of O(n) (n is the number of the clients). We conducted extensive experiments on the MNIST and CIFAR-10 datasets under 5 types of attacks. The results exhibit superior accuracy and resilience against adversarial attacks, all while maintaining comparatively low computational complexity. Among these, on the Non-IID CIFAR-10 dataset, our algorithm exhibited performance enhancements of 19.5% and 14.5%, respectively, in comparison to the state-of-the-art algorithm when dealing with two types of data poisoning attacks.

摘要: 针对隐私保护，联合学习(FL)是一种新兴的机器学习方法，能够在分散的设备或数据源上进行模型训练。FL的学习机制依赖于聚合来自单个客户端的参数更新。然而，由于恶意设备的存在，此过程可能会带来潜在的安全风险。现有的解决方案要么由于使用计算密集型技术而成本高昂，要么由于事先知道攻击者的数量及其攻击方式等强有力的假设而受到限制。很少有方法同时考虑隐私约束和不确定的攻击场景。在本文中，我们提出了一种基于可信度管理方案的稳健FL方法，称为FED-Credit。与以前的研究不同，我们的方法不需要节点和数据分布的先验知识。它维护并使用了一个可信度集合，该集合根据局部模型和全局模型之间的相似度来权衡历史客户的贡献，以调整全局模型的更新。FED-Credit的微妙之处在于，它将时间衰减和态度价值因素引入到声誉权重的动态调整中，其计算复杂度为O(N)(n为客户数量)。我们在MNIST和CIFAR-10数据集上对5种类型的攻击进行了广泛的实验。结果显示，在保持相对较低的计算复杂性的同时，对对手攻击表现出了卓越的准确性和弹性。其中，在非IID CIFAR-10数据集上，与现有算法相比，在处理两种类型的数据中毒攻击时，我们的算法分别表现出19.5%和14.5%的性能提升。



## **19. Towards Optimal Adversarial Robust Q-learning with Bellman Infinity-error**

采用Bellman无限误差实现最佳对抗鲁棒Q学习 cs.LG

**SubmitDate**: 2024-05-20    [abs](http://arxiv.org/abs/2402.02165v2) [paper-pdf](http://arxiv.org/pdf/2402.02165v2)

**Authors**: Haoran Li, Zicheng Zhang, Wang Luo, Congying Han, Yudong Hu, Tiande Guo, Shichen Liao

**Abstract**: Establishing robust policies is essential to counter attacks or disturbances affecting deep reinforcement learning (DRL) agents. Recent studies explore state-adversarial robustness and suggest the potential lack of an optimal robust policy (ORP), posing challenges in setting strict robustness constraints. This work further investigates ORP: At first, we introduce a consistency assumption of policy (CAP) stating that optimal actions in the Markov decision process remain consistent with minor perturbations, supported by empirical and theoretical evidence. Building upon CAP, we crucially prove the existence of a deterministic and stationary ORP that aligns with the Bellman optimal policy. Furthermore, we illustrate the necessity of $L^{\infty}$-norm when minimizing Bellman error to attain ORP. This finding clarifies the vulnerability of prior DRL algorithms that target the Bellman optimal policy with $L^{1}$-norm and motivates us to train a Consistent Adversarial Robust Deep Q-Network (CAR-DQN) by minimizing a surrogate of Bellman Infinity-error. The top-tier performance of CAR-DQN across various benchmarks validates its practical effectiveness and reinforces the soundness of our theoretical analysis.

摘要: 建立稳健的策略对于对抗影响深度强化学习(DRL)代理的攻击或干扰至关重要。最近的研究探索了状态对抗的健壮性，并表明可能缺乏最优的健壮性策略(ORP)，这给设置严格的健壮性约束带来了挑战。首先，我们引入了策略一致性假设(CAP)，指出马尔可夫决策过程中的最优行为在微小扰动下保持一致，并得到了经验和理论证据的支持。在CAP的基础上，我们关键地证明了与Bellman最优策略一致的确定性且平稳的ORP的存在。此外，我们还说明了在最小化Bellman误差以达到ORP时，$L^$-范数的必要性。这一发现澄清了以前以$L^{1}$范数为目标的Bellman最优策略的DRL算法的脆弱性，并激励我们通过最小化Bellman无穷错误的代理来训练一致的对抗性鲁棒深度Q-网络(CAR-DQN)。CAR-DQN在各种基准测试中的顶级性能验证了它的实际有效性，并加强了我们理论分析的合理性。



## **20. Adaptive Batch Normalization Networks for Adversarial Robustness**

对抗鲁棒性的自适应批量正规化网络 cs.LG

Accepted at IEEE International Conference on Advanced Video and  Signal-based Surveillance (AVSS) 2024

**SubmitDate**: 2024-05-20    [abs](http://arxiv.org/abs/2405.11708v1) [paper-pdf](http://arxiv.org/pdf/2405.11708v1)

**Authors**: Shao-Yuan Lo, Vishal M. Patel

**Abstract**: Deep networks are vulnerable to adversarial examples. Adversarial Training (AT) has been a standard foundation of modern adversarial defense approaches due to its remarkable effectiveness. However, AT is extremely time-consuming, refraining it from wide deployment in practical applications. In this paper, we aim at a non-AT defense: How to design a defense method that gets rid of AT but is still robust against strong adversarial attacks? To answer this question, we resort to adaptive Batch Normalization (BN), inspired by the recent advances in test-time domain adaptation. We propose a novel defense accordingly, referred to as the Adaptive Batch Normalization Network (ABNN). ABNN employs a pre-trained substitute model to generate clean BN statistics and sends them to the target model. The target model is exclusively trained on clean data and learns to align the substitute model's BN statistics. Experimental results show that ABNN consistently improves adversarial robustness against both digital and physically realizable attacks on both image and video datasets. Furthermore, ABNN can achieve higher clean data performance and significantly lower training time complexity compared to AT-based approaches.

摘要: 深层网络很容易受到敌意例子的攻击。对抗训练(AT)因其显著的有效性而成为现代对抗防御方法的标准基础。然而，AT非常耗时，阻碍了它在实际应用中的广泛应用。在本文中，我们针对的是一种非AT防御：如何设计一种既能去除AT，又能对强对手攻击保持健壮性的防御方法？为了回答这个问题，我们求助于自适应批处理归一化(BN)，灵感来自于测试-时间域自适应的最新进展。因此，我们提出了一种新的防御方法，称为自适应批处理归一化网络(ABNN)。ABNN使用预先训练的替代模型来生成干净的BN统计数据，并将其发送到目标模型。目标模型专门接受关于干净数据的培训，并学习如何调整替代模型的BN统计数据。实验结果表明，ABNN在抵抗图像和视频数据集上的数字攻击和物理可实现攻击时，都一致地提高了对手的健壮性。此外，与基于AT的方法相比，ABNN可以获得更高的清洁数据性能和更低的训练时间复杂度。



## **21. Geometry-Aware Instrumental Variable Regression**

几何感知工具变量回归 cs.LG

**SubmitDate**: 2024-05-19    [abs](http://arxiv.org/abs/2405.11633v1) [paper-pdf](http://arxiv.org/pdf/2405.11633v1)

**Authors**: Heiner Kremer, Bernhard Schölkopf

**Abstract**: Instrumental variable (IV) regression can be approached through its formulation in terms of conditional moment restrictions (CMR). Building on variants of the generalized method of moments, most CMR estimators are implicitly based on approximating the population data distribution via reweightings of the empirical sample. While for large sample sizes, in the independent identically distributed (IID) setting, reweightings can provide sufficient flexibility, they might fail to capture the relevant information in presence of corrupted data or data prone to adversarial attacks. To address these shortcomings, we propose the Sinkhorn Method of Moments, an optimal transport-based IV estimator that takes into account the geometry of the data manifold through data-derivative information. We provide a simple plug-and-play implementation of our method that performs on par with related estimators in standard settings but improves robustness against data corruption and adversarial attacks.

摘要: 工具变量（IV）回归可以通过条件矩限制（RCM）的公式来进行。在广义矩法的变体的基础上，大多数MCR估计量隐含地基于通过对经验样本的重新加权来逼近人口数据分布。虽然对于大样本量，在独立同分布（IID）设置中，重新加权可以提供足够的灵活性，但在存在损坏的数据或容易遭受对抗攻击的数据的情况下，它们可能无法捕获相关信息。为了解决这些缺点，我们提出了Sinkhorn矩法，这是一种基于传输的最佳IV估计器，它通过数据衍生信息考虑数据流的几何形状。我们提供了我们的方法的简单即插即用实现，其性能与标准设置中的相关估计器相同，但提高了针对数据损坏和对抗性攻击的鲁棒性。



## **22. Searching Realistic-Looking Adversarial Objects For Autonomous Driving Systems**

为自动驾驶系统搜索外观逼真的对抗对象 cs.CV

**SubmitDate**: 2024-05-19    [abs](http://arxiv.org/abs/2405.11629v1) [paper-pdf](http://arxiv.org/pdf/2405.11629v1)

**Authors**: Shengxiang Sun, Shenzhe Zhu

**Abstract**: Numerous studies on adversarial attacks targeting self-driving policies fail to incorporate realistic-looking adversarial objects, limiting real-world applicability. Building upon prior research that facilitated the transition of adversarial objects from simulations to practical applications, this paper discusses a modified gradient-based texture optimization method to discover realistic-looking adversarial objects. While retaining the core architecture and techniques of the prior research, the proposed addition involves an entity termed the 'Judge'. This agent assesses the texture of a rendered object, assigning a probability score reflecting its realism. This score is integrated into the loss function to encourage the NeRF object renderer to concurrently learn realistic and adversarial textures. The paper analyzes four strategies for developing a robust 'Judge': 1) Leveraging cutting-edge vision-language models. 2) Fine-tuning open-sourced vision-language models. 3) Pretraining neurosymbolic systems. 4) Utilizing traditional image processing techniques. Our findings indicate that strategies 1) and 4) yield less reliable outcomes, pointing towards strategies 2) or 3) as more promising directions for future research.

摘要: 许多针对自动驾驶政策的对抗性攻击研究未能纳入看起来逼真的对抗性对象，从而限制了现实世界的适用性。在前人研究的基础上，讨论了一种改进的基于梯度的纹理优化方法，以发现外观逼真的对抗性对象。在保留先前研究的核心架构和技术的同时，拟议的增加涉及一个被称为“法官”的实体。该代理评估渲染对象的纹理，指定反映其真实感的概率分数。这个分数被集成到损失函数中，以鼓励NERF对象渲染器同时学习现实和对抗性纹理。本文分析了开发一个健壮的‘裁判’的四个策略：1)利用尖端的视觉语言模型。2)微调开源的视觉语言模型。3)训练前的神经象征系统。4)利用传统的图像处理技术。我们的发现表明，策略1)和4)产生的结果不太可靠，指出策略2)或3)是未来研究的更有前途的方向。



## **23. Struggle with Adversarial Defense? Try Diffusion**

与对抗性防御作斗争？尝试扩散 cs.CV

**SubmitDate**: 2024-05-19    [abs](http://arxiv.org/abs/2404.08273v3) [paper-pdf](http://arxiv.org/pdf/2404.08273v3)

**Authors**: Yujie Li, Yanbin Wang, Haitao Xu, Bin Liu, Jianguo Sun, Zhenhao Guo, Wenrui Ma

**Abstract**: Adversarial attacks induce misclassification by introducing subtle perturbations. Recently, diffusion models are applied to the image classifiers to improve adversarial robustness through adversarial training or by purifying adversarial noise. However, diffusion-based adversarial training often encounters convergence challenges and high computational expenses. Additionally, diffusion-based purification inevitably causes data shift and is deemed susceptible to stronger adaptive attacks. To tackle these issues, we propose the Truth Maximization Diffusion Classifier (TMDC), a generative Bayesian classifier that builds upon pre-trained diffusion models and the Bayesian theorem. Unlike data-driven classifiers, TMDC, guided by Bayesian principles, utilizes the conditional likelihood from diffusion models to determine the class probabilities of input images, thereby insulating against the influences of data shift and the limitations of adversarial training. Moreover, to enhance TMDC's resilience against more potent adversarial attacks, we propose an optimization strategy for diffusion classifiers. This strategy involves post-training the diffusion model on perturbed datasets with ground-truth labels as conditions, guiding the diffusion model to learn the data distribution and maximizing the likelihood under the ground-truth labels. The proposed method achieves state-of-the-art performance on the CIFAR10 dataset against heavy white-box attacks and strong adaptive attacks. Specifically, TMDC achieves robust accuracies of 82.81% against $l_{\infty}$ norm-bounded perturbations and 86.05% against $l_{2}$ norm-bounded perturbations, respectively, with $\epsilon=0.05$.

摘要: 对抗性攻击通过引入微妙的扰动来导致错误分类。近年来，扩散模型被应用到图像分类器中，通过对抗性训练或净化对抗性噪声来提高对抗性稳健性。然而，基于扩散的对抗性训练往往会遇到收敛挑战和较高的计算开销。此外，基于扩散的净化不可避免地会导致数据转移，并被认为容易受到更强的适应性攻击。为了解决这些问题，我们提出了真值最大化扩散分类器(TMDC)，这是一种生成式贝叶斯分类器，它建立在预先训练的扩散模型和贝叶斯定理的基础上。与数据驱动的分类器不同，TMDC在贝叶斯原理的指导下，利用扩散模型的条件似然来确定输入图像的类别概率，从而避免了数据迁移的影响和对抗性训练的限制。此外，为了增强TMDC对更强大的对手攻击的韧性，我们提出了一种扩散分类器的优化策略。该策略包括在扰动数据集上对扩散模型进行后训练，以地面真实标签为条件，引导扩散模型学习数据分布，并最大化地面真实标签下的似然。该方法在CIFAR10数据集上取得了较好的抗重白盒攻击和强自适应攻击的性能。具体地说，TMDC对$L范数有界摄动和L范数有界摄动的稳健精度分别为82.81%和86.05%，其中$epsilon=0.05$。



## **24. On Robust Reinforcement Learning with Lipschitz-Bounded Policy Networks**

关于Lipschitz有界政策网络的鲁棒强化学习 cs.LG

**SubmitDate**: 2024-05-19    [abs](http://arxiv.org/abs/2405.11432v1) [paper-pdf](http://arxiv.org/pdf/2405.11432v1)

**Authors**: Nicholas H. Barbara, Ruigang Wang, Ian R. Manchester

**Abstract**: This paper presents a study of robust policy networks in deep reinforcement learning. We investigate the benefits of policy parameterizations that naturally satisfy constraints on their Lipschitz bound, analyzing their empirical performance and robustness on two representative problems: pendulum swing-up and Atari Pong. We illustrate that policy networks with small Lipschitz bounds are significantly more robust to disturbances, random noise, and targeted adversarial attacks than unconstrained policies composed of vanilla multi-layer perceptrons or convolutional neural networks. Moreover, we find that choosing a policy parameterization with a non-conservative Lipschitz bound and an expressive, nonlinear layer architecture gives the user much finer control over the performance-robustness trade-off than existing state-of-the-art methods based on spectral normalization.

摘要: 本文对深度强化学习中的鲁棒政策网络进行了研究。我们研究了自然满足Lipschitz界约束的政策参数化的好处，分析了它们在两个代表性问题上的经验性能和稳健性：钟摆摆动和Atari Pong。我们说明，与由普通多层感知器或卷积神经网络组成的无约束策略相比，具有小Lipschitz界的策略网络对干扰、随机噪音和有针对性的对抗攻击的鲁棒性明显更强。此外，我们发现，选择具有非保守Lipschitz界和富有表现力的非线性层架构的策略参数化可以让用户比基于谱正规化的现有最先进方法更好地控制性能-鲁棒性权衡。



## **25. IBD-PSC: Input-level Backdoor Detection via Parameter-oriented Scaling Consistency**

IBD-OSC：通过面向参数的缩放一致性进行输入级后门检测 cs.LG

Accepted to ICML 2024, 29 pages

**SubmitDate**: 2024-05-19    [abs](http://arxiv.org/abs/2405.09786v2) [paper-pdf](http://arxiv.org/pdf/2405.09786v2)

**Authors**: Linshan Hou, Ruili Feng, Zhongyun Hua, Wei Luo, Leo Yu Zhang, Yiming Li

**Abstract**: Deep neural networks (DNNs) are vulnerable to backdoor attacks, where adversaries can maliciously trigger model misclassifications by implanting a hidden backdoor during model training. This paper proposes a simple yet effective input-level backdoor detection (dubbed IBD-PSC) as a 'firewall' to filter out malicious testing images. Our method is motivated by an intriguing phenomenon, i.e., parameter-oriented scaling consistency (PSC), where the prediction confidences of poisoned samples are significantly more consistent than those of benign ones when amplifying model parameters. In particular, we provide theoretical analysis to safeguard the foundations of the PSC phenomenon. We also design an adaptive method to select BN layers to scale up for effective detection. Extensive experiments are conducted on benchmark datasets, verifying the effectiveness and efficiency of our IBD-PSC method and its resistance to adaptive attacks.

摘要: 深度神经网络（DNN）很容易受到后门攻击，对手可以通过在模型训练期间植入隐藏后门来恶意触发模型错误分类。本文提出了一种简单而有效的输入级后门检测（称为IBD-OSC）作为“防火墙”来过滤恶意测试图像。我们的方法的动机是一个有趣的现象，即，面向参数的缩放一致性（OSC），其中在放大模型参数时，中毒样本的预测置信度明显比良性样本的预测置信度更一致。特别是，我们提供理论分析来捍卫CSC现象的基础。我们还设计了一种自适应方法来选择BN层以扩大规模以进行有效检测。在基准数据集上进行了大量实验，验证了我们的IBD-OSC方法的有效性和效率及其对自适应攻击的抵抗力。



## **26. The Perception-Robustness Tradeoff in Deterministic Image Restoration**

确定性图像恢复中的感知与鲁棒性权衡 eess.IV

**SubmitDate**: 2024-05-18    [abs](http://arxiv.org/abs/2311.09253v3) [paper-pdf](http://arxiv.org/pdf/2311.09253v3)

**Authors**: Guy Ohayon, Tomer Michaeli, Michael Elad

**Abstract**: We study the behavior of deterministic methods for solving inverse problems in imaging. These methods are commonly designed to achieve two goals: (1) attaining high perceptual quality, and (2) generating reconstructions that are consistent with the measurements. We provide a rigorous proof that the better a predictor satisfies these two requirements, the larger its Lipschitz constant must be, regardless of the nature of the degradation involved. In particular, to approach perfect perceptual quality and perfect consistency, the Lipschitz constant of the model must grow to infinity. This implies that such methods are necessarily more susceptible to adversarial attacks. We demonstrate our theory on single image super-resolution algorithms, addressing both noisy and noiseless settings. We also show how this undesired behavior can be leveraged to explore the posterior distribution, thereby allowing the deterministic model to imitate stochastic methods.

摘要: 我们研究解决成像反问题的确定性方法的行为。这些方法通常旨在实现两个目标：（1）获得高感知质量，以及（2）生成与测量结果一致的重建。我们提供了一个严格的证据，证明预测器满足这两个要求越好，其利普希茨常数就必须越大，无论所涉及的退化的性质如何。特别是，为了达到完美的感知质量和完美的一致性，模型的利普希茨常数必须增长到无穷大。这意味着此类方法必然更容易受到对抗攻击。我们展示了我们关于单图像超分辨率算法的理论，解决有噪和无噪设置。我们还展示了如何利用这种不受欢迎的行为来探索后验分布，从而允许确定性模型模仿随机方法。



## **27. Provable Unrestricted Adversarial Training without Compromise with Generalizability**

可证明的不受限制的对抗性培训，不妥协并具有概括性 cs.LG

**SubmitDate**: 2024-05-18    [abs](http://arxiv.org/abs/2301.09069v2) [paper-pdf](http://arxiv.org/pdf/2301.09069v2)

**Authors**: Lilin Zhang, Ning Yang, Yanchao Sun, Philip S. Yu

**Abstract**: Adversarial training (AT) is widely considered as the most promising strategy to defend against adversarial attacks and has drawn increasing interest from researchers. However, the existing AT methods still suffer from two challenges. First, they are unable to handle unrestricted adversarial examples (UAEs), which are built from scratch, as opposed to restricted adversarial examples (RAEs), which are created by adding perturbations bound by an $l_p$ norm to observed examples. Second, the existing AT methods often achieve adversarial robustness at the expense of standard generalizability (i.e., the accuracy on natural examples) because they make a tradeoff between them. To overcome these challenges, we propose a unique viewpoint that understands UAEs as imperceptibly perturbed unobserved examples. Also, we find that the tradeoff results from the separation of the distributions of adversarial examples and natural examples. Based on these ideas, we propose a novel AT approach called Provable Unrestricted Adversarial Training (PUAT), which can provide a target classifier with comprehensive adversarial robustness against both UAE and RAE, and simultaneously improve its standard generalizability. Particularly, PUAT utilizes partially labeled data to achieve effective UAE generation by accurately capturing the natural data distribution through a novel augmented triple-GAN. At the same time, PUAT extends the traditional AT by introducing the supervised loss of the target classifier into the adversarial loss and achieves the alignment between the UAE distribution, the natural data distribution, and the distribution learned by the classifier, with the collaboration of the augmented triple-GAN. Finally, the solid theoretical analysis and extensive experiments conducted on widely-used benchmarks demonstrate the superiority of PUAT.

摘要: 对抗训练(AT)被广泛认为是防御对抗攻击的最有前途的策略，越来越受到研究者的关注。然而，现有的AT方法仍然面临着两个挑战。首先，它们不能处理从头开始构建的无限制对抗性示例(UAE)，而受限对抗性示例(RAE)是通过将受$L_p$范数约束的扰动添加到观察到的示例来创建的。第二，现有的AT方法往往以牺牲标准泛化能力(即对自然样本的准确性)为代价来实现对抗的健壮性，因为它们在两者之间进行了权衡。为了克服这些挑战，我们提出了一种独特的观点，将UAE理解为潜移默化的未被观察到的例子。此外，我们还发现，这种权衡是由于对抗性例子和自然例子的分布分离造成的。基于这些思想，我们提出了一种新的AT方法，称为可证明的无限制对抗训练(PUAT)，它可以为目标分类器提供对UAE和RAE都具有全面的对抗健壮性，同时提高其标准泛化能力。特别是，PUAT利用部分标记的数据，通过一种新的增强型三层GaN准确地捕获自然数据分布，从而实现有效的UAE生成。同时，PUAT通过在对手损失中引入目标分类器的监督损失来扩展传统的AT，并通过扩展的三层GAN协作实现了UAE分布、自然数据分布和分类器学习的分布之间的对齐。最后，在广泛使用的基准上进行了扎实的理论分析和广泛的实验，证明了PUAT的优越性。



## **28. Few-Shot API Attack Detection: Overcoming Data Scarcity with GAN-Inspired Learning**

Few-Shot API攻击检测：通过GAN启发学习克服数据稀缺性 cs.CR

8 pages, 2 figures, 7 tables

**SubmitDate**: 2024-05-18    [abs](http://arxiv.org/abs/2405.11258v1) [paper-pdf](http://arxiv.org/pdf/2405.11258v1)

**Authors**: Udi Aharon, Revital Marbel, Ran Dubin, Amit Dvir, Chen Hajaj

**Abstract**: Web applications and APIs face constant threats from malicious actors seeking to exploit vulnerabilities for illicit gains. These threats necessitate robust anomaly detection systems capable of identifying malicious API traffic efficiently despite limited and diverse datasets. This paper proposes a novel few-shot detection approach motivated by Natural Language Processing (NLP) and advanced Generative Adversarial Network (GAN)-inspired techniques. Leveraging state-of-the-art Transformer architectures, particularly RoBERTa, our method enhances the contextual understanding of API requests, leading to improved anomaly detection compared to traditional methods. We showcase the technique's versatility by demonstrating its effectiveness with both Out-of-Distribution (OOD) and Transformer-based binary classification methods on two distinct datasets: CSIC 2010 and ATRDF 2023. Our evaluations reveal consistently enhanced or, at worst, equivalent detection rates across various metrics in most vectors, highlighting the promise of our approach for improving API security.

摘要: Web应用程序和API面临着来自恶意行为者的持续威胁，这些行为者试图利用漏洞获取非法收益。这些威胁要求强大的异常检测系统能够有效地识别恶意API流量，尽管数据集有限且多样化。提出了一种基于自然语言处理(NLP)和先进的产生式对抗网络(GAN)技术的少镜头检测方法。利用最先进的Transformer架构，特别是Roberta，我们的方法增强了对API请求的上下文理解，导致与传统方法相比改进了异常检测。我们通过在CSIC 2010和ATRDF 2023这两个不同的数据集上使用OOD和基于Transformer的二进制分类方法来展示该技术的多功能性。我们的评估显示，在大多数向量中，各种指标的检测率一直都在提高，甚至在最坏的情况下，检测率相当，这突显了我们改进API安全性的方法的前景。



## **29. Dynamic Quantum Key Distribution for Microgrids with Distributed Error Correction**

具有分布式误差纠正的微电网动态量子密钥分配 cs.CR

**SubmitDate**: 2024-05-18    [abs](http://arxiv.org/abs/2405.11245v1) [paper-pdf](http://arxiv.org/pdf/2405.11245v1)

**Authors**: Suman Rath, Neel Kanth Kundu, Subham Sahoo

**Abstract**: Quantum key distribution (QKD) has often been hailed as a reliable technology for secure communication in cyber-physical microgrids. Even though unauthorized key measurements are not possible in QKD, attempts to read them can disturb quantum states leading to mutations in the transmitted value. Further, inaccurate quantum keys can lead to erroneous decryption producing garbage values, destabilizing microgrid operation. QKD can also be vulnerable to node-level manipulations incorporating attack values into measurements before they are encrypted at the communication layer. To address these issues, this paper proposes a secure QKD protocol that can identify errors in keys and/or nodal measurements by observing violations in control dynamics. Additionally, the protocol uses a dynamic adjacency matrix-based formulation strategy enabling the affected nodes to reconstruct a trustworthy signal and replace it with the attacked signal in a multi-hop manner. This enables microgrids to perform nominal operations in the presence of adversaries who try to eavesdrop on the system causing an increase in the quantum bit error rate (QBER). We provide several case studies to showcase the robustness of the proposed strategy against eavesdroppers and node manipulations. The results demonstrate that it can resist unwanted observation and attack vectors that manipulate signals before encryption.

摘要: 量子密钥分发(QKD)通常被誉为在网络物理微网中进行安全通信的可靠技术。尽管未经授权的密钥测量在量子密钥分发中是不可能的，但读取它们的尝试可能会干扰量子态，导致传输的值发生突变。此外，不准确的量子密钥可能导致错误解密，产生垃圾值，破坏微电网运行的稳定。在通信层加密之前，QKD也容易受到将攻击值合并到测量中的节点级操作的攻击。为了解决这些问题，提出了一种安全的量子密钥分发协议，该协议可以通过观察控制动态中的违规行为来识别密钥和/或节点测量中的错误。此外，该协议使用了基于动态邻接矩阵的公式化策略，使得受影响的节点能够以多跳的方式重建可信信号并将其替换为被攻击的信号。这使得微电网能够在试图窃听系统的对手在场的情况下执行名义操作，从而导致量子误码率(QBER)的增加。我们提供了几个案例研究来展示所提出的策略对窃听者和节点操纵的健壮性。实验结果表明，该算法能够抵抗不必要的观测和攻击向量，在加密前对信号进行处理。



## **30. Towards Robust Policy: Enhancing Offline Reinforcement Learning with Adversarial Attacks and Defenses**

走向稳健的政策：通过对抗性攻击和防御增强离线强化学习 cs.LG

**SubmitDate**: 2024-05-18    [abs](http://arxiv.org/abs/2405.11206v1) [paper-pdf](http://arxiv.org/pdf/2405.11206v1)

**Authors**: Thanh Nguyen, Tung M. Luu, Tri Ton, Chang D. Yoo

**Abstract**: Offline reinforcement learning (RL) addresses the challenge of expensive and high-risk data exploration inherent in RL by pre-training policies on vast amounts of offline data, enabling direct deployment or fine-tuning in real-world environments. However, this training paradigm can compromise policy robustness, leading to degraded performance in practical conditions due to observation perturbations or intentional attacks. While adversarial attacks and defenses have been extensively studied in deep learning, their application in offline RL is limited. This paper proposes a framework to enhance the robustness of offline RL models by leveraging advanced adversarial attacks and defenses. The framework attacks the actor and critic components by perturbing observations during training and using adversarial defenses as regularization to enhance the learned policy. Four attacks and two defenses are introduced and evaluated on the D4RL benchmark. The results show the vulnerability of both the actor and critic to attacks and the effectiveness of the defenses in improving policy robustness. This framework holds promise for enhancing the reliability of offline RL models in practical scenarios.

摘要: 离线强化学习(RL)通过对大量离线数据进行预培训策略，在真实环境中实现直接部署或微调，从而解决了RL固有的昂贵和高风险数据探索的挑战。然而，这种训练范例可能会损害策略的稳健性，导致在实际条件下由于观测扰动或故意攻击而性能下降。尽管对抗性攻击和防御在深度学习中得到了广泛的研究，但它们在离线RL中的应用有限。本文提出了一个框架，通过利用先进的对抗性攻击和防御来增强离线RL模型的健壮性。该框架通过在训练过程中干扰观察并使用对抗性防御作为正规化来增强学习策略来攻击参与者和批评者组件。引入了四种攻击和两种防御，并在D4RL基准上进行了评估。结果表明，参与者和批评者对攻击的脆弱性以及防御措施在提高策略稳健性方面的有效性。该框架有望在实际场景中提高离线RL模型的可靠性。



## **31. Trustworthy Actionable Perturbations**

值得信赖的可操作干扰 cs.LG

Accepted at the 41st International Conference on Machine Learning  (ICML) 2024

**SubmitDate**: 2024-05-18    [abs](http://arxiv.org/abs/2405.11195v1) [paper-pdf](http://arxiv.org/pdf/2405.11195v1)

**Authors**: Jesse Friedbaum, Sudarshan Adiga, Ravi Tandon

**Abstract**: Counterfactuals, or modified inputs that lead to a different outcome, are an important tool for understanding the logic used by machine learning classifiers and how to change an undesirable classification. Even if a counterfactual changes a classifier's decision, however, it may not affect the true underlying class probabilities, i.e. the counterfactual may act like an adversarial attack and ``fool'' the classifier. We propose a new framework for creating modified inputs that change the true underlying probabilities in a beneficial way which we call Trustworthy Actionable Perturbations (TAP). This includes a novel verification procedure to ensure that TAP change the true class probabilities instead of acting adversarially. Our framework also includes new cost, reward, and goal definitions that are better suited to effectuating change in the real world. We present PAC-learnability results for our verification procedure and theoretically analyze our new method for measuring reward. We also develop a methodology for creating TAP and compare our results to those achieved by previous counterfactual methods.

摘要: 反事实，或导致不同结果的修改输入，是理解机器学习分类器使用的逻辑以及如何更改不希望看到的分类的重要工具。然而，即使反事实改变了分类器的决定，它也不可能影响真正的潜在类别概率，即反事实可能表现为对抗性攻击和对分类器的“愚弄”。我们提出了一个新的框架，用于创建修改的输入，以一种有益的方式改变真实的潜在概率，我们称之为可信可操作扰动(TAP)。这包括一种新的验证程序，以确保TAP改变真实的类别概率，而不是采取相反的行动。我们的框架还包括新的成本、回报和目标定义，这些定义更适合在现实世界中实现变化。我们给出了验证过程的PAC-学习性结果，并从理论上分析了我们的新的报酬度量方法。我们还开发了一种创建TAP的方法，并将我们的结果与以前的反事实方法进行了比较。



## **32. Revisiting the Robust Generalization of Adversarial Prompt Tuning**

重新审视对抗即时调整的稳健概括 cs.CV

**SubmitDate**: 2024-05-18    [abs](http://arxiv.org/abs/2405.11154v1) [paper-pdf](http://arxiv.org/pdf/2405.11154v1)

**Authors**: Fan Yang, Mingxuan Xia, Sangzhou Xia, Chicheng Ma, Hui Hui

**Abstract**: Understanding the vulnerability of large-scale pre-trained vision-language models like CLIP against adversarial attacks is key to ensuring zero-shot generalization capacity on various downstream tasks. State-of-the-art defense mechanisms generally adopt prompt learning strategies for adversarial fine-tuning to improve the adversarial robustness of the pre-trained model while keeping the efficiency of adapting to downstream tasks. Such a setup leads to the problem of over-fitting which impedes further improvement of the model's generalization capacity on both clean and adversarial examples. In this work, we propose an adaptive Consistency-guided Adversarial Prompt Tuning (i.e., CAPT) framework that utilizes multi-modal prompt learning to enhance the alignment of image and text features for adversarial examples and leverage the strong generalization of pre-trained CLIP to guide the model-enhancing its robust generalization on adversarial examples while maintaining its accuracy on clean ones. We also design a novel adaptive consistency objective function to balance the consistency of adversarial inputs and clean inputs between the fine-tuning model and the pre-trained model. We conduct extensive experiments across 14 datasets and 4 data sparsity schemes (from 1-shot to full training data settings) to show the superiority of CAPT over other state-of-the-art adaption methods. CAPT demonstrated excellent performance in terms of the in-distribution performance and the generalization under input distribution shift and across datasets.

摘要: 了解像CLIP这样的大规模预先训练的视觉语言模型对对抗攻击的脆弱性是确保对各种下游任务的零命中泛化能力的关键。最新的防御机制一般采用对抗性微调的快速学习策略，在保持适应下游任务的效率的同时，提高预先训练模型的对抗性健壮性。这样的设置导致了过度拟合的问题，这阻碍了模型在干净和对抗性例子上的推广能力的进一步提高。在这项工作中，我们提出了一种自适应一致性制导的对抗性提示调整(CAPT)框架，该框架利用多模式提示学习来增强对抗性示例的图文特征对齐，并利用预先训练的CLIP的强泛化来指导模型--增强了对对抗性示例的健壮性，同时保持了对干净示例的准确性。我们还设计了一种新的自适应一致性目标函数，以平衡微调模型和预训练模型之间的对抗性输入和干净输入的一致性。我们在14个数据集和4个数据稀疏方案(从单镜头到全训练数据设置)上进行了广泛的实验，以显示CAPT相对于其他最先进的自适应方法的优越性。CAPT在分布内性能和在输入分布漂移和跨数据集情况下的泛化方面表现出了优异的性能。



## **33. Calibration Attacks: A Comprehensive Study of Adversarial Attacks on Model Confidence**

校准攻击：模型置信度对抗攻击的综合研究 cs.LG

**SubmitDate**: 2024-05-18    [abs](http://arxiv.org/abs/2401.02718v2) [paper-pdf](http://arxiv.org/pdf/2401.02718v2)

**Authors**: Stephen Obadinma, Xiaodan Zhu, Hongyu Guo

**Abstract**: In this work, we highlight and perform a comprehensive study on calibration attacks, a form of adversarial attacks that aim to trap victim models to be heavily miscalibrated without altering their predicted labels, hence endangering the trustworthiness of the models and follow-up decision making based on their confidence. We propose four typical forms of calibration attacks: underconfidence, overconfidence, maximum miscalibration, and random confidence attacks, conducted in both the black-box and white-box setups. We demonstrate that the attacks are highly effective on both convolutional and attention-based models: with a small number of queries, they seriously skew confidence without changing the predictive performance. Given the potential danger, we further investigate the effectiveness of a wide range of adversarial defence and recalibration methods, including our proposed defences specifically designed for calibration attacks to mitigate the harm. From the ECE and KS scores, we observe that there are still significant limitations in handling calibration attacks. To the best of our knowledge, this is the first dedicated study that provides a comprehensive investigation on calibration-focused attacks. We hope this study helps attract more attention to these types of attacks and hence hamper their potential serious damages. To this end, this work also provides detailed analyses to understand the characteristics of the attacks.

摘要: 在这项工作中，我们重点对校准攻击进行了全面的研究，校准攻击是一种对抗性攻击，旨在诱使受害者模型在不改变预测标签的情况下被严重错误校准，从而危及模型的可信性和基于其置信度的后续决策。我们提出了四种典型的校准攻击形式：欠自信、过度自信、最大误校准和随机置信度攻击，分别在黑盒和白盒设置下进行。我们证明了这些攻击在卷积模型和基于注意力的模型上都是非常有效的：在少量查询的情况下，它们在不改变预测性能的情况下严重地扭曲了置信度。鉴于潜在的危险，我们进一步调查了一系列对抗性防御和重新校准方法的有效性，包括我们为减轻危害而专门为校准攻击设计的拟议防御方法。从欧洲经委会和KS分数来看，我们注意到在处理校准攻击方面仍然存在重大限制。据我们所知，这是第一个对以校准为重点的攻击进行全面调查的专门研究。我们希望这项研究有助于引起人们对这些类型攻击的更多关注，从而阻止它们可能造成的严重损害。为此，这项工作还提供了详细的分析，以了解攻击的特点。



## **34. Transpose Attack: Stealing Datasets with Bidirectional Training**

转置攻击：通过双向训练窃取数据集 cs.LG

NDSS24 paper, Transpose Attack, Transposed Model. NDSS version:  https://www.ndss-symposium.org/ndss-paper/transpose-attack-stealing-datasets-with-bidirectional-training/

**SubmitDate**: 2024-05-17    [abs](http://arxiv.org/abs/2311.07389v2) [paper-pdf](http://arxiv.org/pdf/2311.07389v2)

**Authors**: Guy Amit, Mosh Levy, Yisroel Mirsky

**Abstract**: Deep neural networks are normally executed in the forward direction. However, in this work, we identify a vulnerability that enables models to be trained in both directions and on different tasks. Adversaries can exploit this capability to hide rogue models within seemingly legitimate models. In addition, in this work we show that neural networks can be taught to systematically memorize and retrieve specific samples from datasets. Together, these findings expose a novel method in which adversaries can exfiltrate datasets from protected learning environments under the guise of legitimate models. We focus on the data exfiltration attack and show that modern architectures can be used to secretly exfiltrate tens of thousands of samples with high fidelity, high enough to compromise data privacy and even train new models. Moreover, to mitigate this threat we propose a novel approach for detecting infected models.

摘要: 深度神经网络通常向前执行。然而，在这项工作中，我们发现了一个漏洞，该漏洞使模型能够在两个方向和不同任务上进行训练。对手可以利用这种能力将流氓模型隐藏在看似合法的模型中。此外，在这项工作中，我们表明可以教神经网络系统地记忆和检索数据集中的特定样本。总而言之，这些发现揭示了一种新颖的方法，对手可以打着合法模型的幌子从受保护的学习环境中提取数据集。我们重点关注数据泄露攻击，并表明现代架构可以用于以高保真度秘密泄露数万个样本，高到足以损害数据隐私，甚至训练新模型。此外，为了减轻这种威胁，我们提出了一种检测受感染模型的新颖方法。



## **35. Rethinking Graph Backdoor Attacks: A Distribution-Preserving Perspective**

重新思考图表后门攻击：保留分布的角度 cs.LG

**SubmitDate**: 2024-05-17    [abs](http://arxiv.org/abs/2405.10757v1) [paper-pdf](http://arxiv.org/pdf/2405.10757v1)

**Authors**: Zhiwei Zhang, Minhua Lin, Enyan Dai, Suhang Wang

**Abstract**: Graph Neural Networks (GNNs) have shown remarkable performance in various tasks. However, recent works reveal that GNNs are vulnerable to backdoor attacks. Generally, backdoor attack poisons the graph by attaching backdoor triggers and the target class label to a set of nodes in the training graph. A GNN trained on the poisoned graph will then be misled to predict test nodes attached with trigger to the target class. Despite their effectiveness, our empirical analysis shows that triggers generated by existing methods tend to be out-of-distribution (OOD), which significantly differ from the clean data. Hence, these injected triggers can be easily detected and pruned with widely used outlier detection methods in real-world applications. Therefore, in this paper, we study a novel problem of unnoticeable graph backdoor attacks with in-distribution (ID) triggers. To generate ID triggers, we introduce an OOD detector in conjunction with an adversarial learning strategy to generate the attributes of the triggers within distribution. To ensure a high attack success rate with ID triggers, we introduce novel modules designed to enhance trigger memorization by the victim model trained on poisoned graph. Extensive experiments on real-world datasets demonstrate the effectiveness of the proposed method in generating in distribution triggers that can by-pass various defense strategies while maintaining a high attack success rate.

摘要: 图形神经网络(GNN)在各种任务中表现出了显著的性能。然而，最近的研究表明，GNN很容易受到后门攻击。通常，后门攻击通过将后门触发器和目标类标签附加到训练图中的一组节点来毒化图。然后，在有毒图上训练的GNN将被误导，以预测与目标类的触发器附加的测试节点。尽管它们是有效的，但我们的实证分析表明，现有方法生成的触发因素往往是分布外(OOD)，这与干净的数据有很大不同。因此，这些注入的触发器可以很容易地被现实世界应用中广泛使用的离群点检测方法检测和修剪。因此，在本文中，我们研究了一种新的具有分布内(ID)触发器的不可察觉图后门攻击问题。为了生成ID触发器，我们引入了一个OOD检测器，并结合对抗性学习策略来生成分布内触发器的属性。为了确保ID触发器的高攻击成功率，我们引入了新的模块，通过在中毒图上训练受害者模型来增强对触发器的记忆。在真实数据集上的大量实验表明，该方法在生成分布触发器方面是有效的，可以绕过各种防御策略，同时保持较高的攻击成功率。



## **36. Safeguarding Vision-Language Models Against Patched Visual Prompt Injectors**

保护视觉语言模型免受修补视觉提示注入器的影响 cs.CV

15 pages

**SubmitDate**: 2024-05-17    [abs](http://arxiv.org/abs/2405.10529v1) [paper-pdf](http://arxiv.org/pdf/2405.10529v1)

**Authors**: Jiachen Sun, Changsheng Wang, Jiongxiao Wang, Yiwei Zhang, Chaowei Xiao

**Abstract**: Large language models have become increasingly prominent, also signaling a shift towards multimodality as the next frontier in artificial intelligence, where their embeddings are harnessed as prompts to generate textual content. Vision-language models (VLMs) stand at the forefront of this advancement, offering innovative ways to combine visual and textual data for enhanced understanding and interaction. However, this integration also enlarges the attack surface. Patch-based adversarial attack is considered the most realistic threat model in physical vision applications, as demonstrated in many existing literature. In this paper, we propose to address patched visual prompt injection, where adversaries exploit adversarial patches to generate target content in VLMs. Our investigation reveals that patched adversarial prompts exhibit sensitivity to pixel-wise randomization, a trait that remains robust even against adaptive attacks designed to counteract such defenses. Leveraging this insight, we introduce SmoothVLM, a defense mechanism rooted in smoothing techniques, specifically tailored to protect VLMs from the threat of patched visual prompt injectors. Our framework significantly lowers the attack success rate to a range between 0% and 5.0% on two leading VLMs, while achieving around 67.3% to 95.0% context recovery of the benign images, demonstrating a balance between security and usability.

摘要: 大型语言模型已变得越来越突出，这也标志着向多通道的转变，成为人工智能的下一个前沿，在人工智能中，它们的嵌入被用作生成文本内容的提示。视觉语言模型(VLM)站在这一进步的前沿，提供了将视觉和文本数据相结合的创新方法，以增强理解和交互。然而，这种整合也扩大了攻击面。基于补丁的对抗性攻击被认为是物理视觉应用中最现实的威胁模型，许多现有的文献都证明了这一点。在本文中，我们建议解决补丁视觉提示注入，即攻击者利用敌意补丁来生成VLMS中的目标内容。我们的调查显示，打补丁的对抗性提示显示出对像素随机化的敏感性，这一特征即使在旨在对抗此类防御的适应性攻击中也保持健壮。利用这一见解，我们推出了SmoothVLM，这是一种植根于平滑技术的防御机制，专门为保护VLM免受修补的视觉提示注入器的威胁而量身定做。我们的框架将攻击成功率显著降低到了0%到5.0%之间，同时实现了良性映像的67.3%到95.0%的上下文恢复，展示了安全性和可用性之间的平衡。



## **37. ALI-DPFL: Differentially Private Federated Learning with Adaptive Local Iterations**

ALI-DPFL：具有自适应本地迭代的差异化私人联邦学习 cs.LG

**SubmitDate**: 2024-05-17    [abs](http://arxiv.org/abs/2308.10457v8) [paper-pdf](http://arxiv.org/pdf/2308.10457v8)

**Authors**: Xinpeng Ling, Jie Fu, Kuncan Wang, Haitao Liu, Zhili Chen

**Abstract**: Federated Learning (FL) is a distributed machine learning technique that allows model training among multiple devices or organizations by sharing training parameters instead of raw data. However, adversaries can still infer individual information through inference attacks (e.g. differential attacks) on these training parameters. As a result, Differential Privacy (DP) has been widely used in FL to prevent such attacks.   We consider differentially private federated learning in a resource-constrained scenario, where both privacy budget and communication rounds are constrained. By theoretically analyzing the convergence, we can find the optimal number of local DPSGD iterations for clients between any two sequential global updates. Based on this, we design an algorithm of Differentially Private Federated Learning with Adaptive Local Iterations (ALI-DPFL). We experiment our algorithm on the MNIST, FashionMNIST and Cifar10 datasets, and demonstrate significantly better performances than previous work in the resource-constraint scenario. Code is available at https://github.com/KnightWan/ALI-DPFL.

摘要: 联合学习(FL)是一种分布式机器学习技术，通过共享训练参数而不是原始数据，允许在多个设备或组织之间进行模型训练。然而，攻击者仍然可以通过对这些训练参数的推理攻击(例如差异攻击)来推断个人信息。因此，差分隐私(DP)被广泛应用于FL中以防止此类攻击。我们考虑在资源受限的情况下进行不同的私有联合学习，其中隐私预算和通信回合都受到限制。通过对收敛的理论分析，我们可以找到任意两个连续全局更新之间客户端的最优局部DPSGD迭代次数。在此基础上，设计了一种基于自适应局部迭代的差分私有联邦学习算法(ALI-DPFL)。我们在MNIST、FashionMNIST和Cifar10数据集上测试了我们的算法，并在资源受限的情况下展示了比以前的工作更好的性能。代码可在https://github.com/KnightWan/ALI-DPFL.上找到



## **38. Secure Set-Based State Estimation for Linear Systems under Adversarial Attacks on Sensors**

传感器对抗攻击下线性系统的安全基于集的状态估计 eess.SY

**SubmitDate**: 2024-05-16    [abs](http://arxiv.org/abs/2309.05075v2) [paper-pdf](http://arxiv.org/pdf/2309.05075v2)

**Authors**: M. Umar B. Niazi, Michelle S. Chong, Amr Alanwar, Karl H. Johansson

**Abstract**: Set-based state estimation plays a vital role in the safety verification of dynamical systems, which becomes significantly challenging when the system's sensors are susceptible to cyber-attacks. Existing methods often impose limitations on the attacker's capabilities, restricting the number of attacked sensors to be strictly less than half of the total number of sensors. This paper proposes a Secure Set-Based State Estimation (S3E) algorithm that addresses this limitation. The S3E algorithm guarantees that the true system state is contained within the estimated set, provided the initialization set encompasses the true initial state and the system is redundantly observable from the set of uncompromised sensors. The algorithm gives the estimated set as a collection of constrained zonotopes, which can be employed as robust certificates for verifying whether the system adheres to safety constraints. Furthermore, we demonstrate that the estimated set remains unaffected by attack signals of sufficiently large and also establish sufficient conditions for attack detection, identification, and filtering. This compels the attacker to inject only stealthy signals of small magnitude to evade detection, thus preserving the accuracy of the estimated set. When a few number of sensors (less than half) can be compromised, we prove that the estimated set remains bounded by a contracting set that converges to a ball whose radius is solely determined by the noise magnitude and is independent of the attack signals. To address the computational complexity of the algorithm, we offer several strategies for complexity-performance trade-offs. The efficacy of the proposed algorithm is illustrated through its application to a three-story building model.

摘要: 基于集合的状态估计在动态系统的安全性验证中起着至关重要的作用，当系统的传感器容易受到网络攻击时，这就变得非常具有挑战性。现有的方法经常对攻击者的能力施加限制，将被攻击的传感器数量严格限制在传感器总数的一半以下。针对这一缺陷，提出了一种安全的基于集合的状态估计算法(S3E)。S3E算法保证真实的系统状态包含在估计的集合内，前提是初始化集合包含真实的初始状态，并且从未受损的传感器集合可以冗余地观察到系统。该算法将估计集表示为受约束区域的集合，可用作验证系统是否符合安全约束的稳健证书。此外，我们证明了估计集不受足够大的攻击信号的影响，并为攻击检测、识别和过滤建立了充分的条件。这迫使攻击者仅注入小幅度的隐形信号来躲避检测，从而保持估计集的准确性。当几个传感器(不到一半)可能被破坏时，我们证明了估计集仍然有界于一个收缩集，该收缩集收敛到一个球，其半径完全由噪声大小决定，并且与攻击信号无关。为了解决算法的计算复杂性，我们提供了几种在复杂性和性能之间进行权衡的策略。通过对一个三层建筑模型的应用，说明了该算法的有效性。



## **39. Adversarial Robustness Guarantees for Quantum Classifiers**

量子分类器的对抗稳健性保证 quant-ph

9+12 pages, 3 figures. Comments welcome

**SubmitDate**: 2024-05-16    [abs](http://arxiv.org/abs/2405.10360v1) [paper-pdf](http://arxiv.org/pdf/2405.10360v1)

**Authors**: Neil Dowling, Maxwell T. West, Angus Southwell, Azar C. Nakhl, Martin Sevior, Muhammad Usman, Kavan Modi

**Abstract**: Despite their ever more widespread deployment throughout society, machine learning algorithms remain critically vulnerable to being spoofed by subtle adversarial tampering with their input data. The prospect of near-term quantum computers being capable of running {quantum machine learning} (QML) algorithms has therefore generated intense interest in their adversarial vulnerability. Here we show that quantum properties of QML algorithms can confer fundamental protections against such attacks, in certain scenarios guaranteeing robustness against classically-armed adversaries. We leverage tools from many-body physics to identify the quantum sources of this protection. Our results offer a theoretical underpinning of recent evidence which suggest quantum advantages in the search for adversarial robustness. In particular, we prove that quantum classifiers are: (i) protected against weak perturbations of data drawn from the trained distribution, (ii) protected against local attacks if they are insufficiently scrambling, and (iii) protected against universal adversarial attacks if they are sufficiently quantum chaotic. Our analytic results are supported by numerical evidence demonstrating the applicability of our theorems and the resulting robustness of a quantum classifier in practice. This line of inquiry constitutes a concrete pathway to advantage in QML, orthogonal to the usually sought improvements in model speed or accuracy.

摘要: 尽管机器学习算法在整个社会得到了越来越广泛的部署，但它们仍然非常容易受到输入数据的微妙对手篡改的欺骗。因此，近期量子计算机能够运行量子机器学习(QML)算法的前景引起了人们对它们的对手脆弱性的浓厚兴趣。在这里，我们展示了QML算法的量子特性可以提供针对此类攻击的基本保护，在某些场景中保证了对经典武装对手的健壮性。我们利用多体物理学的工具来确定这种保护的量子来源。我们的结果为最近的证据提供了理论基础，这些证据表明量子优势在搜索对手的稳健性方面具有优势。特别地，我们证明了量子分类器是：(I)防止来自训练分布的数据的弱扰动，(Ii)如果不充分置乱，则防止局部攻击，以及(Iii)如果它们足够量子混沌，则防止普遍的对抗性攻击。我们的分析结果得到了数值证据的支持，证明了我们的定理的适用性以及由此产生的量子分类器在实践中的健壮性。这条探索路线构成了一条在QML中获得优势的具体途径，与通常寻求的模型速度或精度方面的改进是垂直的。



## **40. "What do you want from theory alone?" Experimenting with Tight Auditing of Differentially Private Synthetic Data Generation**

“你只想从理论中得到什么？“尝试对差异私有合成数据生成进行严格审计 cs.CR

To appear at Usenix Security 2024

**SubmitDate**: 2024-05-16    [abs](http://arxiv.org/abs/2405.10994v1) [paper-pdf](http://arxiv.org/pdf/2405.10994v1)

**Authors**: Meenatchi Sundaram Muthu Selva Annamalai, Georgi Ganev, Emiliano De Cristofaro

**Abstract**: Differentially private synthetic data generation (DP-SDG) algorithms are used to release datasets that are structurally and statistically similar to sensitive data while providing formal bounds on the information they leak. However, bugs in algorithms and implementations may cause the actual information leakage to be higher. This prompts the need to verify whether the theoretical guarantees of state-of-the-art DP-SDG implementations also hold in practice. We do so via a rigorous auditing process: we compute the information leakage via an adversary playing a distinguishing game and running membership inference attacks (MIAs). If the leakage observed empirically is higher than the theoretical bounds, we identify a DP violation; if it is non-negligibly lower, the audit is loose.   We audit six DP-SDG implementations using different datasets and threat models and find that black-box MIAs commonly used against DP-SDGs are severely limited in power, yielding remarkably loose empirical privacy estimates. We then consider MIAs in stronger threat models, i.e., passive and active white-box, using both existing and newly proposed attacks. Overall, we find that, currently, we do not only need white-box MIAs but also worst-case datasets to tightly estimate the privacy leakage from DP-SDGs. Finally, we show that our automated auditing procedure finds both known DP violations (in 4 out of the 6 implementations) as well as a new one in the DPWGAN implementation that was successfully submitted to the NIST DP Synthetic Data Challenge.   The source code needed to reproduce our experiments is available from https://github.com/spalabucr/synth-audit.

摘要: 差分私有合成数据生成(DP-SDG)算法用于发布在结构和统计上与敏感数据相似的数据集，同时为它们泄漏的信息提供正式的界限。然而，算法和实现中的错误可能会导致实际信息泄漏更高。这就需要验证最先进的DP-SDG实施的理论保证在实践中是否也成立。我们通过严格的审计过程来做到这一点：我们通过对手玩区分游戏并运行成员推断攻击(MIA)来计算信息泄漏。如果经验上观察到的泄漏高于理论界限，我们就确定了DP违规；如果它低于不可忽略的水平，则审计是宽松的。我们使用不同的数据集和威胁模型审计了六个DP-SDG实现，发现通常用于DP-SDG的黑盒MIA在能力上受到严重限制，产生了非常宽松的经验隐私估计。然后，我们使用现有的和新提出的攻击，在更强的威胁模型中考虑MIA，即被动和主动白盒。总体而言，我们发现，目前，我们不仅需要白盒MIA，还需要最坏情况的数据集来严格估计DP-SDG的隐私泄露。最后，我们展示了我们的自动审计程序发现了已知的DP违规(在6个实现中的4个)，以及在成功提交给NIST DP合成数据挑战赛的DPWGAN实现中的新的一个。重现我们实验所需的源代码可从https://github.com/spalabucr/synth-audit.获得



## **41. Protecting Your LLMs with Information Bottleneck**

通过信息瓶颈保护您的LLC cs.CL

23 pages, 7 figures, 8 tables

**SubmitDate**: 2024-05-16    [abs](http://arxiv.org/abs/2404.13968v2) [paper-pdf](http://arxiv.org/pdf/2404.13968v2)

**Authors**: Zichuan Liu, Zefan Wang, Linjie Xu, Jinyu Wang, Lei Song, Tianchun Wang, Chunlin Chen, Wei Cheng, Jiang Bian

**Abstract**: The advent of large language models (LLMs) has revolutionized the field of natural language processing, yet they might be attacked to produce harmful content. Despite efforts to ethically align LLMs, these are often fragile and can be circumvented by jailbreaking attacks through optimized or manual adversarial prompts. To address this, we introduce the Information Bottleneck Protector (IBProtector), a defense mechanism grounded in the information bottleneck principle, and we modify the objective to avoid trivial solutions. The IBProtector selectively compresses and perturbs prompts, facilitated by a lightweight and trainable extractor, preserving only essential information for the target LLMs to respond with the expected answer. Moreover, we further consider a situation where the gradient is not visible to be compatible with any LLM. Our empirical evaluations show that IBProtector outperforms current defense methods in mitigating jailbreak attempts, without overly affecting response quality or inference speed. Its effectiveness and adaptability across various attack methods and target LLMs underscore the potential of IBProtector as a novel, transferable defense that bolsters the security of LLMs without requiring modifications to the underlying models.

摘要: 大型语言模型的出现给自然语言处理领域带来了革命性的变化，但它们可能会受到攻击，产生有害的内容。尽管努力在道德上调整LLM，但这些往往是脆弱的，可以通过优化或手动对抗性提示通过越狱攻击来绕过。为了解决这个问题，我们引入了信息瓶颈保护器(IBProtector)，这是一种基于信息瓶颈原理的防御机制，我们修改了目标以避免琐碎的解决方案。IBProtector有选择地压缩和干扰提示，由一个轻量级和可训练的提取程序促进，只保留目标LLMS的基本信息，以响应预期的答案。此外，我们还进一步考虑了梯度不可见的情况，以与任何LLM相容。我们的经验评估表明，在不过度影响响应质量或推理速度的情况下，IBProtector在缓解越狱企图方面优于现有的防御方法。它对各种攻击方法和目标LLM的有效性和适应性突显了IBProtector作为一种新型、可转移的防御系统的潜力，无需修改底层模型即可增强LLM的安全性。



## **42. Adversarial Robustness for Visual Grounding of Multimodal Large Language Models**

多模式大型语言模型视觉基础的对抗鲁棒性 cs.CV

ICLR 2024 Workshop on Reliable and Responsible Foundation Models

**SubmitDate**: 2024-05-16    [abs](http://arxiv.org/abs/2405.09981v1) [paper-pdf](http://arxiv.org/pdf/2405.09981v1)

**Authors**: Kuofeng Gao, Yang Bai, Jiawang Bai, Yong Yang, Shu-Tao Xia

**Abstract**: Multi-modal Large Language Models (MLLMs) have recently achieved enhanced performance across various vision-language tasks including visual grounding capabilities. However, the adversarial robustness of visual grounding remains unexplored in MLLMs. To fill this gap, we use referring expression comprehension (REC) as an example task in visual grounding and propose three adversarial attack paradigms as follows. Firstly, untargeted adversarial attacks induce MLLMs to generate incorrect bounding boxes for each object. Besides, exclusive targeted adversarial attacks cause all generated outputs to the same target bounding box. In addition, permuted targeted adversarial attacks aim to permute all bounding boxes among different objects within a single image. Extensive experiments demonstrate that the proposed methods can successfully attack visual grounding capabilities of MLLMs. Our methods not only provide a new perspective for designing novel attacks but also serve as a strong baseline for improving the adversarial robustness for visual grounding of MLLMs.

摘要: 多模式大型语言模型(MLLM)最近在包括视觉基础能力在内的各种视觉语言任务中获得了增强的性能。然而，在最大似然最小二乘法中，视觉接地的对抗稳健性仍未被探索。为了填补这一空白，我们使用指称表达理解(REC)作为视觉基础的示例任务，并提出了以下三种对抗性攻击范式。首先，无针对性的对抗性攻击会导致MLLMS为每个对象生成错误的包围盒。此外，排他性定向对抗性攻击会导致所有生成的输出都指向相同的目标边界框。此外，置换定向对抗性攻击旨在置换单个图像中不同对象之间的所有包围盒。大量实验表明，所提出的方法能够成功地攻击MLLMS的视觉接地能力。我们的方法不仅为设计新的攻击提供了新的视角，而且为提高MLLMS视觉接地的对抗性稳健性提供了强有力的基线。



## **43. Deepfake Generation and Detection: A Benchmark and Survey**

Deepfake生成和检测：基准和调查 cs.CV

We closely follow the latest developments in  https://github.com/flyingby/Awesome-Deepfake-Generation-and-Detection

**SubmitDate**: 2024-05-16    [abs](http://arxiv.org/abs/2403.17881v4) [paper-pdf](http://arxiv.org/pdf/2403.17881v4)

**Authors**: Gan Pei, Jiangning Zhang, Menghan Hu, Zhenyu Zhang, Chengjie Wang, Yunsheng Wu, Guangtao Zhai, Jian Yang, Chunhua Shen, Dacheng Tao

**Abstract**: Deepfake is a technology dedicated to creating highly realistic facial images and videos under specific conditions, which has significant application potential in fields such as entertainment, movie production, digital human creation, to name a few. With the advancements in deep learning, techniques primarily represented by Variational Autoencoders and Generative Adversarial Networks have achieved impressive generation results. More recently, the emergence of diffusion models with powerful generation capabilities has sparked a renewed wave of research. In addition to deepfake generation, corresponding detection technologies continuously evolve to regulate the potential misuse of deepfakes, such as for privacy invasion and phishing attacks. This survey comprehensively reviews the latest developments in deepfake generation and detection, summarizing and analyzing current state-of-the-arts in this rapidly evolving field. We first unify task definitions, comprehensively introduce datasets and metrics, and discuss developing technologies. Then, we discuss the development of several related sub-fields and focus on researching four representative deepfake fields: face swapping, face reenactment, talking face generation, and facial attribute editing, as well as forgery detection. Subsequently, we comprehensively benchmark representative methods on popular datasets for each field, fully evaluating the latest and influential published works. Finally, we analyze challenges and future research directions of the discussed fields.

摘要: 深伪是一项致力于在特定条件下创建高真实感面部图像和视频的技术，在娱乐、电影制作、数字人类创作等领域具有巨大的应用潜力。随着深度学习的进步，以变式自动编码器和生成式对抗性网络为主要代表的技术已经取得了令人印象深刻的生成结果。最近，具有强大发电能力的扩散模型的出现引发了新一轮的研究浪潮。除了深度假冒的生成，相应的检测技术也在不断发展，以规范深度假冒的潜在滥用，例如用于侵犯隐私和网络钓鱼攻击。这项调查全面回顾了深度伪码生成和检测的最新进展，总结和分析了这一快速发展领域的最新技术。我们首先统一任务定义，全面介绍数据集和指标，并讨论开发技术。然后，讨论了几个相关的子领域的发展，重点研究了四个有代表性的深度伪领域：人脸交换、人脸重演、说话人脸生成、人脸属性编辑以及伪造检测。随后，我们在每个领域的热门数据集上综合基准有代表性的方法，充分评价最新和有影响力的已发表作品。最后，分析了所讨论领域面临的挑战和未来的研究方向。



## **44. Infrared Adversarial Car Stickers**

红外对抗汽车贴纸 cs.CV

Accepted by CVPR 2024

**SubmitDate**: 2024-05-16    [abs](http://arxiv.org/abs/2405.09924v1) [paper-pdf](http://arxiv.org/pdf/2405.09924v1)

**Authors**: Xiaopei Zhu, Yuqiu Liu, Zhanhao Hu, Jianmin Li, Xiaolin Hu

**Abstract**: Infrared physical adversarial examples are of great significance for studying the security of infrared AI systems that are widely used in our lives such as autonomous driving. Previous infrared physical attacks mainly focused on 2D infrared pedestrian detection which may not fully manifest its destructiveness to AI systems. In this work, we propose a physical attack method against infrared detectors based on 3D modeling, which is applied to a real car. The goal is to design a set of infrared adversarial stickers to make cars invisible to infrared detectors at various viewing angles, distances, and scenes. We build a 3D infrared car model with real infrared characteristics and propose an infrared adversarial pattern generation method based on 3D mesh shadow. We propose a 3D control points-based mesh smoothing algorithm and use a set of smoothness loss functions to enhance the smoothness of adversarial meshes and facilitate the sticker implementation. Besides, We designed the aluminum stickers and conducted physical experiments on two real Mercedes-Benz A200L cars. Our adversarial stickers hid the cars from Faster RCNN, an object detector, at various viewing angles, distances, and scenes. The attack success rate (ASR) was 91.49% for real cars. In comparison, the ASRs of random stickers and no sticker were only 6.21% and 0.66%, respectively. In addition, the ASRs of the designed stickers against six unseen object detectors such as YOLOv3 and Deformable DETR were between 73.35%-95.80%, showing good transferability of the attack performance across detectors.

摘要: 红外物理对抗例子对于研究自动驾驶等广泛应用于我们生活中的红外AI系统的安全性具有重要意义。以前的红外物理攻击主要集中在2D红外行人检测上，这可能不能充分体现其对人工智能系统的破坏性。在这项工作中，我们提出了一种基于3D建模的对红外探测器的物理攻击方法，并将其应用于真实汽车。目标是设计一套红外对抗性贴纸，使汽车在不同的视角、距离和场景下都能被红外探测器看不见。建立了具有真实红外特征的三维红外汽车模型，提出了一种基于三维网格阴影的红外对抗模式生成方法。提出了一种基于三维控制点的网格光顺算法，并使用一组光滑度损失函数来增强对抗性网格的光滑度，便于粘贴的实现。此外，我们设计了铝制贴纸，并在两辆真实的梅赛德斯-奔驰A200L轿车上进行了物理实验。我们的对抗性贴纸在不同的视角、距离和场景下将汽车隐藏起来，以躲避速度更快的RCNN，一个物体探测器。实车攻击成功率(ASR)为91.49%。相比之下，随机贴纸和不贴纸的ASR分别只有6.21%和0.66%。此外，所设计的标签对YOLOv3、可变形DETR等6种隐形目标探测器的ASR在73.35%~95.80%之间，表现出良好的跨探测器攻击性能的可转移性。



## **45. DiffAM: Diffusion-based Adversarial Makeup Transfer for Facial Privacy Protection**

迪夫AM：基于扩散的对抗性化妆转移，用于面部隐私保护 cs.CV

16 pages, 11 figures

**SubmitDate**: 2024-05-16    [abs](http://arxiv.org/abs/2405.09882v1) [paper-pdf](http://arxiv.org/pdf/2405.09882v1)

**Authors**: Yuhao Sun, Lingyun Yu, Hongtao Xie, Jiaming Li, Yongdong Zhang

**Abstract**: With the rapid development of face recognition (FR) systems, the privacy of face images on social media is facing severe challenges due to the abuse of unauthorized FR systems. Some studies utilize adversarial attack techniques to defend against malicious FR systems by generating adversarial examples. However, the generated adversarial examples, i.e., the protected face images, tend to suffer from subpar visual quality and low transferability. In this paper, we propose a novel face protection approach, dubbed DiffAM, which leverages the powerful generative ability of diffusion models to generate high-quality protected face images with adversarial makeup transferred from reference images. To be specific, we first introduce a makeup removal module to generate non-makeup images utilizing a fine-tuned diffusion model with guidance of textual prompts in CLIP space. As the inverse process of makeup transfer, makeup removal can make it easier to establish the deterministic relationship between makeup domain and non-makeup domain regardless of elaborate text prompts. Then, with this relationship, a CLIP-based makeup loss along with an ensemble attack strategy is introduced to jointly guide the direction of adversarial makeup domain, achieving the generation of protected face images with natural-looking makeup and high black-box transferability. Extensive experiments demonstrate that DiffAM achieves higher visual quality and attack success rates with a gain of 12.98% under black-box setting compared with the state of the arts. The code will be available at https://github.com/HansSunY/DiffAM.

摘要: 随着人脸识别系统的快速发展，由于未经授权的人脸识别系统的滥用，社交媒体上人脸图像的隐私面临着严峻的挑战。一些研究利用对抗性攻击技术通过生成对抗性实例来防御恶意FR系统。然而，生成的敌意例子，即受保护的人脸图像，往往存在视觉质量不佳和可转移性低的问题。在本文中，我们提出了一种新的人脸保护方法，称为DIFAM，它利用扩散模型的强大生成能力来生成高质量的受保护的人脸图像，其中包含从参考图像转换来的对抗性化妆。具体地说，我们首先介绍了一个卸妆模块，该模块利用一个微调的扩散模型，在剪辑空间中以文本提示为指导来生成非化妆图像。作为化妆转移的逆过程，卸妆可以更容易地建立化妆域和非化妆域之间的确定性关系，而不需要考虑精心设计的文字提示。然后，在这种关系下，引入了基于剪辑的化妆损失和系综攻击策略，共同指导对抗性化妆领域的方向，实现了化妆自然、黑盒可转移性高的受保护人脸图像的生成。大量实验表明，与现有技术相比，DIFAM算法在黑盒环境下获得了更高的视觉质量和攻击成功率，提高了12.98%。代码将在https://github.com/HansSunY/DiffAM.上提供



## **46. Box-Free Model Watermarks Are Prone to Black-Box Removal Attacks**

无框模型水印容易受到黑匣子删除攻击 cs.CV

**SubmitDate**: 2024-05-16    [abs](http://arxiv.org/abs/2405.09863v1) [paper-pdf](http://arxiv.org/pdf/2405.09863v1)

**Authors**: Haonan An, Guang Hua, Zhiping Lin, Yuguang Fang

**Abstract**: Box-free model watermarking is an emerging technique to safeguard the intellectual property of deep learning models, particularly those for low-level image processing tasks. Existing works have verified and improved its effectiveness in several aspects. However, in this paper, we reveal that box-free model watermarking is prone to removal attacks, even under the real-world threat model such that the protected model and the watermark extractor are in black boxes. Under this setting, we carry out three studies. 1) We develop an extractor-gradient-guided (EGG) remover and show its effectiveness when the extractor uses ReLU activation only. 2) More generally, for an unknown extractor, we leverage adversarial attacks and design the EGG remover based on the estimated gradients. 3) Under the most stringent condition that the extractor is inaccessible, we design a transferable remover based on a set of private proxy models. In all cases, the proposed removers can successfully remove embedded watermarks while preserving the quality of the processed images, and we also demonstrate that the EGG remover can even replace the watermarks. Extensive experimental results verify the effectiveness and generalizability of the proposed attacks, revealing the vulnerabilities of the existing box-free methods and calling for further research.

摘要: 无盒模型水印是一种新兴的保护深度学习模型知识产权的技术，尤其是用于低层图像处理任务的模型。已有的工作在几个方面验证和改进了它的有效性。然而，在本文中，我们揭示了无盒模型水印容易受到移除攻击，即使在真实世界的威胁模型下，受保护的模型和水印抽取器都在黑盒中。在此背景下，我们开展了三个方面的研究。1)我们开发了一种萃取器-梯度引导(EGG)去除器，并在仅使用RELU激活的情况下展示了其有效性。2)更一般地，对于未知的提取者，我们利用对抗性攻击，并基于估计的梯度来设计鸡蛋去除器。3)在抽取器不可访问的最严格条件下，基于一组私有代理模型设计了一个可转移的抽取器。在所有情况下，所提出的去除器都可以在保持处理图像质量的情况下成功地去除嵌入的水印，并且我们还证明了鸡蛋去除器甚至可以替换水印。大量的实验结果验证了所提出的攻击方法的有效性和泛化能力，揭示了现有去盒方法的弱点，需要进一步研究。



## **47. Manifold Integrated Gradients: Riemannian Geometry for Feature Attribution**

多元积分：特征属性的Riemann几何 cs.LG

Accepted at ICML 2024

**SubmitDate**: 2024-05-16    [abs](http://arxiv.org/abs/2405.09800v1) [paper-pdf](http://arxiv.org/pdf/2405.09800v1)

**Authors**: Eslam Zaher, Maciej Trzaskowski, Quan Nguyen, Fred Roosta

**Abstract**: In this paper, we dive into the reliability concerns of Integrated Gradients (IG), a prevalent feature attribution method for black-box deep learning models. We particularly address two predominant challenges associated with IG: the generation of noisy feature visualizations for vision models and the vulnerability to adversarial attributional attacks. Our approach involves an adaptation of path-based feature attribution, aligning the path of attribution more closely to the intrinsic geometry of the data manifold. Our experiments utilise deep generative models applied to several real-world image datasets. They demonstrate that IG along the geodesics conforms to the curved geometry of the Riemannian data manifold, generating more perceptually intuitive explanations and, subsequently, substantially increasing robustness to targeted attributional attacks.

摘要: 在本文中，我们深入探讨了集成属性（IG）的可靠性问题，这是一种用于黑匣子深度学习模型的流行特征归因方法。我们特别解决了与IG相关的两个主要挑战：视觉模型的有噪特征可视化的生成以及对抗性归因攻击的脆弱性。我们的方法涉及对基于路径的特征属性的调整，将属性路径更紧密地与数据集合的内在几何图形对齐。我们的实验利用应用于多个现实世界图像数据集的深度生成模型。他们证明，沿着测地线的IG符合Riemann数据多管齐下的弯曲几何，从而生成更直观的解释，并随后大幅提高了对有针对性的归因攻击的鲁棒性。



## **48. Benchmark Early and Red Team Often: A Framework for Assessing and Managing Dual-Use Hazards of AI Foundation Models**

早期基准和经常红色团队：评估和管理人工智能基金会模型双重用途危害的框架 cs.CR

62 pages

**SubmitDate**: 2024-05-15    [abs](http://arxiv.org/abs/2405.10986v1) [paper-pdf](http://arxiv.org/pdf/2405.10986v1)

**Authors**: Anthony M. Barrett, Krystal Jackson, Evan R. Murphy, Nada Madkour, Jessica Newman

**Abstract**: A concern about cutting-edge or "frontier" AI foundation models is that an adversary may use the models for preparing chemical, biological, radiological, nuclear, (CBRN), cyber, or other attacks. At least two methods can identify foundation models with potential dual-use capability; each has advantages and disadvantages: A. Open benchmarks (based on openly available questions and answers), which are low-cost but accuracy-limited by the need to omit security-sensitive details; and B. Closed red team evaluations (based on private evaluation by CBRN and cyber experts), which are higher-cost but can achieve higher accuracy by incorporating sensitive details. We propose a research and risk-management approach using a combination of methods including both open benchmarks and closed red team evaluations, in a way that leverages advantages of both methods. We recommend that one or more groups of researchers with sufficient resources and access to a range of near-frontier and frontier foundation models run a set of foundation models through dual-use capability evaluation benchmarks and red team evaluations, then analyze the resulting sets of models' scores on benchmark and red team evaluations to see how correlated those are. If, as we expect, there is substantial correlation between the dual-use potential benchmark scores and the red team evaluation scores, then implications include the following: The open benchmarks should be used frequently during foundation model development as a quick, low-cost measure of a model's dual-use potential; and if a particular model gets a high score on the dual-use potential benchmark, then more in-depth red team assessments of that model's dual-use capability should be performed. We also discuss limitations and mitigations for our approach, e.g., if model developers try to game benchmarks by including a version of benchmark test data in a model's training data.

摘要: 对尖端或“前沿”人工智能基础模型的一个担忧是，对手可能会使用这些模型来准备化学、生物、辐射、核、核反应堆(CBRN)、网络或其他攻击。至少有两种方法可以确定具有潜在两用能力的基础模型；每种方法都有优缺点：a.开放基准(基于公开提供的问题和答案)，成本低，但准确性有限，因为需要省略安全敏感细节；b.封闭红色团队评估(基于CBRN和网络专家的私下评估)，成本较高，但可以通过纳入敏感细节实现更高的准确性。我们提出了一种研究和风险管理方法，使用包括开放基准和封闭红色团队评估在内的方法组合，以一种利用两种方法的优势的方式。我们建议一个或多个拥有足够资源并能够接触到一系列近前沿和前沿基础模型的研究人员通过两用能力评估基准和RED团队评估运行一组基础模型，然后分析所得到的模型集在基准和RED团队评估上的分数，以了解它们之间的相关性。如果像我们预期的那样，两用潜力基准分数和红色团队评估分数之间存在实质性的相关性，那么影响包括以下几个方面：开放基准应该在基础模型开发期间频繁使用，作为对模型两用潜力的快速、低成本的度量；如果特定模型在两用潜力基准上获得高分，那么应该对该模型的两用能力进行更深入的红色团队评估。我们还讨论了我们方法的限制和缓解措施，例如，如果模型开发人员试图通过在模型的训练数据中包含一个版本的基准测试数据来玩基准游戏。



## **49. Towards Evaluating the Robustness of Automatic Speech Recognition Systems via Audio Style Transfer**

通过音频风格转移评估自动语音识别系统的稳健性 cs.SD

Accepted to SecTL (AsiaCCS Workshop) 2024

**SubmitDate**: 2024-05-15    [abs](http://arxiv.org/abs/2405.09470v1) [paper-pdf](http://arxiv.org/pdf/2405.09470v1)

**Authors**: Weifei Jin, Yuxin Cao, Junjie Su, Qi Shen, Kai Ye, Derui Wang, Jie Hao, Ziyao Liu

**Abstract**: In light of the widespread application of Automatic Speech Recognition (ASR) systems, their security concerns have received much more attention than ever before, primarily due to the susceptibility of Deep Neural Networks. Previous studies have illustrated that surreptitiously crafting adversarial perturbations enables the manipulation of speech recognition systems, resulting in the production of malicious commands. These attack methods mostly require adding noise perturbations under $\ell_p$ norm constraints, inevitably leaving behind artifacts of manual modifications. Recent research has alleviated this limitation by manipulating style vectors to synthesize adversarial examples based on Text-to-Speech (TTS) synthesis audio. However, style modifications based on optimization objectives significantly reduce the controllability and editability of audio styles. In this paper, we propose an attack on ASR systems based on user-customized style transfer. We first test the effect of Style Transfer Attack (STA) which combines style transfer and adversarial attack in sequential order. And then, as an improvement, we propose an iterative Style Code Attack (SCA) to maintain audio quality. Experimental results show that our method can meet the need for user-customized styles and achieve a success rate of 82% in attacks, while keeping sound naturalness due to our user study.

摘要: 随着自动语音识别(ASR)系统的广泛应用，其安全问题受到了前所未有的关注，这主要是由于深度神经网络的敏感性。以前的研究表明，秘密地制作敌意扰动能够操纵语音识别系统，导致产生恶意命令。这些攻击方法大多需要在$\ell_p$范数约束下添加噪声扰动，不可避免地会留下人工修改的伪影。最近的研究通过操纵风格向量来合成基于文本到语音(TTS)合成音频的对抗性示例，从而缓解了这一限制。然而，基于优化目标的风格修改显著降低了音频风格的可控性和可编辑性。本文提出了一种基于用户自定义风格转移的ASR系统攻击方法。我们首先测试了风格转移攻击(STA)的效果，该攻击按顺序将风格转移和对抗性攻击结合在一起。然后，作为改进，我们提出了一种迭代样式码攻击(SCA)来保持音频质量。实验结果表明，该方法能够满足用户对个性化风格的需求，攻击成功率达到82%，同时由于我们的用户学习，保持了声音的自然性。



## **50. Properties that allow or prohibit transferability of adversarial attacks among quantized networks**

允许或禁止对抗攻击在量化网络之间转移的属性 cs.LG

**SubmitDate**: 2024-05-15    [abs](http://arxiv.org/abs/2405.09598v1) [paper-pdf](http://arxiv.org/pdf/2405.09598v1)

**Authors**: Abhishek Shrestha, Jürgen Großmann

**Abstract**: Deep Neural Networks (DNNs) are known to be vulnerable to adversarial examples. Further, these adversarial examples are found to be transferable from the source network in which they are crafted to a black-box target network. As the trend of using deep learning on embedded devices grows, it becomes relevant to study the transferability properties of adversarial examples among compressed networks. In this paper, we consider quantization as a network compression technique and evaluate the performance of transfer-based attacks when the source and target networks are quantized at different bitwidths. We explore how algorithm specific properties affect transferability by considering various adversarial example generation algorithms. Furthermore, we examine transferability in a more realistic scenario where the source and target networks may differ in bitwidth and other model-related properties like capacity and architecture. We find that although quantization reduces transferability, certain attack types demonstrate an ability to enhance it. Additionally, the average transferability of adversarial examples among quantized versions of a network can be used to estimate the transferability to quantized target networks with varying capacity and architecture.

摘要: 深度神经网络(DNN)很容易受到敌意例子的影响。此外，发现这些敌意的例子可以从它们被制作的源网络转移到黑盒目标网络。随着在嵌入式设备上使用深度学习的趋势的发展，研究对抗性例子在压缩网络中的可转移性变得非常重要。在本文中，我们将量化作为一种网络压缩技术，并对源网络和目标网络在不同比特宽度上进行量化时基于传输的攻击的性能进行评估。通过考虑不同的对抗性实例生成算法，我们探讨了算法的特定属性如何影响可转移性。此外，我们在更现实的场景中检查了可转移性，其中源网络和目标网络的位宽以及其他与模型相关的属性(如容量和体系结构)可能不同。我们发现，虽然量化降低了可转移性，但某些攻击类型表现出了增强可转移性的能力。此外，网络的量化版本之间的对抗性例子的平均可转移性可以用来估计到具有不同容量和体系结构的量化目标网络的可转移性。



