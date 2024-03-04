# Latest Adversarial Attack Papers
**update at 2024-03-04 16:55:53**

翻译来自 https://cloud.tencent.com/document/product/551/15619

## **1. Adversarial Examples are Misaligned in Diffusion Model Manifolds**

扩散模型流形中对抗性例子的错位 cs.CV

under review

**SubmitDate**: 2024-03-01    [abs](http://arxiv.org/abs/2401.06637v4) [paper-pdf](http://arxiv.org/pdf/2401.06637v4)

**Authors**: Peter Lorenz, Ricard Durall, Janis Keuper

**Abstract**: In recent years, diffusion models (DMs) have drawn significant attention for their success in approximating data distributions, yielding state-of-the-art generative results. Nevertheless, the versatility of these models extends beyond their generative capabilities to encompass various vision applications, such as image inpainting, segmentation, adversarial robustness, among others. This study is dedicated to the investigation of adversarial attacks through the lens of diffusion models. However, our objective does not involve enhancing the adversarial robustness of image classifiers. Instead, our focus lies in utilizing the diffusion model to detect and analyze the anomalies introduced by these attacks on images. To that end, we systematically examine the alignment of the distributions of adversarial examples when subjected to the process of transformation using diffusion models. The efficacy of this approach is assessed across CIFAR-10 and ImageNet datasets, including varying image sizes in the latter. The results demonstrate a notable capacity to discriminate effectively between benign and attacked images, providing compelling evidence that adversarial instances do not align with the learned manifold of the DMs.

摘要: 近年来，扩散模型（DM）由于其在近似数据分布方面的成功而引起了人们的极大关注，产生了最先进的生成结果。然而，这些模型的多功能性超出了其生成能力，涵盖了各种视觉应用，例如图像修复，分割，对抗鲁棒性等。本研究致力于通过扩散模型的镜头对抗性攻击的调查。然而，我们的目标并不涉及增强图像分类器的对抗鲁棒性。相反，我们的重点在于利用扩散模型来检测和分析这些攻击图像所引入的异常。为此，我们系统地研究了对抗性样本在使用扩散模型进行转换过程时的分布对齐。这种方法的有效性在CIFAR-10和ImageNet数据集上进行了评估，包括后者中不同的图像大小。结果表明，有效区分良性和受攻击图像的能力显著，提供了令人信服的证据表明，对抗性实例与DM的学习流形不一致。



## **2. Unfolding Local Growth Rate Estimates for (Almost) Perfect Adversarial Detection**

关于(几乎)完美敌意检测的局部增长率估计 cs.CV

accepted at VISAPP23

**SubmitDate**: 2024-03-01    [abs](http://arxiv.org/abs/2212.06776v5) [paper-pdf](http://arxiv.org/pdf/2212.06776v5)

**Authors**: Peter Lorenz, Margret Keuper, Janis Keuper

**Abstract**: Convolutional neural networks (CNN) define the state-of-the-art solution on many perceptual tasks. However, current CNN approaches largely remain vulnerable against adversarial perturbations of the input that have been crafted specifically to fool the system while being quasi-imperceptible to the human eye. In recent years, various approaches have been proposed to defend CNNs against such attacks, for example by model hardening or by adding explicit defence mechanisms. Thereby, a small "detector" is included in the network and trained on the binary classification task of distinguishing genuine data from data containing adversarial perturbations. In this work, we propose a simple and light-weight detector, which leverages recent findings on the relation between networks' local intrinsic dimensionality (LID) and adversarial attacks. Based on a re-interpretation of the LID measure and several simple adaptations, we surpass the state-of-the-art on adversarial detection by a significant margin and reach almost perfect results in terms of F1-score for several networks and datasets. Sources available at: https://github.com/adverML/multiLID

摘要: 卷积神经网络(CNN)定义了许多感知任务的最先进的解决方案。然而，目前的CNN方法在很大程度上仍然容易受到输入的对抗性扰动，这些扰动是专门为愚弄系统而设计的，而人眼几乎察觉不到。近年来，已经提出了各种方法来保护CNN免受此类攻击，例如通过模型硬化或通过添加显式防御机制。因此，在网络中包括一个小的“检测器”，并在区分真实数据和包含对抗性扰动的数据的二进制分类任务上进行训练。在这项工作中，我们提出了一个简单而轻量级的检测器，它利用了最近关于网络的局部固有维度(LID)与对手攻击之间关系的研究结果。基于对LID度量的重新解释和几个简单的适应，我们在对手检测方面远远超过了最先进的水平，并在几个网络和数据集的F1得分方面取得了几乎完美的结果。资料来源：https://github.com/adverML/multiLID



## **3. Protect and Extend -- Using GANs for Synthetic Data Generation of Time-Series Medical Records**

保护与延伸--使用GANS生成时间序列病历的合成数据 cs.LG

**SubmitDate**: 2024-03-01    [abs](http://arxiv.org/abs/2402.14042v2) [paper-pdf](http://arxiv.org/pdf/2402.14042v2)

**Authors**: Navid Ashrafi, Vera Schmitt, Robert P. Spang, Sebastian Möller, Jan-Niklas Voigt-Antons

**Abstract**: Preservation of private user data is of paramount importance for high Quality of Experience (QoE) and acceptability, particularly with services treating sensitive data, such as IT-based health services. Whereas anonymization techniques were shown to be prone to data re-identification, synthetic data generation has gradually replaced anonymization since it is relatively less time and resource-consuming and more robust to data leakage. Generative Adversarial Networks (GANs) have been used for generating synthetic datasets, especially GAN frameworks adhering to the differential privacy phenomena. This research compares state-of-the-art GAN-based models for synthetic data generation to generate time-series synthetic medical records of dementia patients which can be distributed without privacy concerns. Predictive modeling, autocorrelation, and distribution analysis are used to assess the Quality of Generating (QoG) of the generated data. The privacy preservation of the respective models is assessed by applying membership inference attacks to determine potential data leakage risks. Our experiments indicate the superiority of the privacy-preserving GAN (PPGAN) model over other models regarding privacy preservation while maintaining an acceptable level of QoG. The presented results can support better data protection for medical use cases in the future.

摘要: 保存私人用户数据对于高质量体验和可接受性至关重要，特别是对于处理敏感数据的服务，如基于IT的医疗服务。由于匿名化技术被证明易于数据重新识别，合成数据生成因其相对较少的时间和资源消耗以及对数据泄露的健壮性而逐渐取代匿名化。生成性对抗网络(GANS)已被用于生成合成数据集，特别是遵循差异隐私现象的GAN框架。这项研究比较了最先进的基于GaN的合成数据生成模型，以生成痴呆症患者的时间序列合成病历，这些病历可以在没有隐私问题的情况下分发。预测建模、自相关和分布分析用于评估所生成数据的生成质量(QOG)。通过应用成员推理攻击来确定潜在的数据泄露风险，来评估各个模型的隐私保护。我们的实验表明，隐私保护GAN(PPGAN)模型在保持可接受的QOG水平的同时，在隐私保护方面优于其他模型。所给出的结果可以为未来医疗用例提供更好的数据保护。



## **4. Attacking Delay-based PUFs with Minimal Adversary Model**

基于最小附着模型的延迟型PUF攻击 cs.CR

13 pages, 6 figures, journal

**SubmitDate**: 2024-03-01    [abs](http://arxiv.org/abs/2403.00464v1) [paper-pdf](http://arxiv.org/pdf/2403.00464v1)

**Authors**: Hongming Fei, Owen Millwood, Prosanta Gope, Jack Miskelly, Biplab Sikdar

**Abstract**: Physically Unclonable Functions (PUFs) provide a streamlined solution for lightweight device authentication. Delay-based Arbiter PUFs, with their ease of implementation and vast challenge space, have received significant attention; however, they are not immune to modelling attacks that exploit correlations between their inputs and outputs. Research is therefore polarized between developing modelling-resistant PUFs and devising machine learning attacks against them. This dichotomy often results in exaggerated concerns and overconfidence in PUF security, primarily because there lacks a universal tool to gauge a PUF's security. In many scenarios, attacks require additional information, such as PUF type or configuration parameters. Alarmingly, new PUFs are often branded `secure' if they lack a specific attack model upon introduction. To impartially assess the security of delay-based PUFs, we present a generic framework featuring a Mixture-of-PUF-Experts (MoPE) structure for mounting attacks on various PUFs with minimal adversarial knowledge, which provides a way to compare their performance fairly and impartially. We demonstrate the capability of our model to attack different PUF types, including the first successful attack on Heterogeneous Feed-Forward PUFs using only a reasonable amount of challenges and responses. We propose an extension version of our model, a Multi-gate Mixture-of-PUF-Experts (MMoPE) structure, facilitating multi-task learning across diverse PUFs to recognise commonalities across PUF designs. This allows a streamlining of training periods for attacking multiple PUFs simultaneously. We conclude by showcasing the potent performance of MoPE and MMoPE across a spectrum of PUF types, employing simulated, real-world unbiased, and biased data sets for analysis.

摘要: 物理不可克隆功能(PUF)为轻量级设备身份验证提供了简化的解决方案。基于延迟的仲裁器PUF由于其易于实现和巨大的挑战空间，受到了极大的关注；然而，它们也不能幸免于利用其输入和输出之间的相关性的建模攻击。因此，研究在开发抵抗建模的PUF和设计针对它们的机器学习攻击之间存在两极分化。这种二分法往往导致对PUF安全的夸大担忧和过度自信，主要是因为缺乏衡量PUF安全的通用工具。在许多情况下，攻击需要其他信息，例如PUF类型或配置参数。令人担忧的是，如果新的PUF在引入时缺乏特定的攻击模型，则通常被贴上“安全”的标签。为了公正地评估基于延迟的PUF的安全性，我们提出了一种具有混合PUF-Experts(MOPE)结构的通用框架，用于在最少的对抗知识的情况下对各种PUF进行攻击，这提供了一种公平公正地比较它们的性能的方法。我们演示了我们的模型能够攻击不同的PUF类型，包括仅使用合理数量的挑战和响应对异类前馈PUF进行的第一次成功攻击。我们提出了我们的模型的扩展版本，一种多门PUF专家混合(MMoPE)结构，促进了跨不同PUF的多任务学习，以识别跨PUF设计的共性。这可以简化同时攻击多个PUF的训练周期。最后，我们展示了MOPE和MMoPE在一系列PUF类型上的强大性能，使用模拟的、真实的、无偏的和有偏的数据集进行分析。



## **5. Robust Deep Reinforcement Learning Through Adversarial Attacks and Training : A Survey**

基于对抗性攻击和训练的稳健深度强化学习研究综述 cs.LG

57 pages, 16 figues, 2 tables

**SubmitDate**: 2024-03-01    [abs](http://arxiv.org/abs/2403.00420v1) [paper-pdf](http://arxiv.org/pdf/2403.00420v1)

**Authors**: Lucas Schott, Josephine Delas, Hatem Hajri, Elies Gherbi, Reda Yaich, Nora Boulahia-Cuppens, Frederic Cuppens, Sylvain Lamprier

**Abstract**: Deep Reinforcement Learning (DRL) is an approach for training autonomous agents across various complex environments. Despite its significant performance in well known environments, it remains susceptible to minor conditions variations, raising concerns about its reliability in real-world applications. To improve usability, DRL must demonstrate trustworthiness and robustness. A way to improve robustness of DRL to unknown changes in the conditions is through Adversarial Training, by training the agent against well suited adversarial attacks on the dynamics of the environment. Addressing this critical issue, our work presents an in-depth analysis of contemporary adversarial attack methodologies, systematically categorizing them and comparing their objectives and operational mechanisms. This classification offers a detailed insight into how adversarial attacks effectively act for evaluating the resilience of DRL agents, thereby paving the way for enhancing their robustness.

摘要: 深度强化学习(DRL)是一种在各种复杂环境中训练自主智能体的方法。尽管它在众所周知的环境中表现出色，但它仍然容易受到微小条件变化的影响，这引发了人们对其在现实世界应用中的可靠性的担忧。为了提高可用性，DRL必须证明可信性和健壮性。提高DRL对未知条件变化的稳健性的一种方法是通过对抗性训练，通过训练代理抵御对环境动态的很好的对抗性攻击。针对这一关键问题，我们的工作对当代对抗攻击方法进行了深入分析，系统地对它们进行了分类，并比较了它们的目标和运行机制。这一分类提供了对抗性攻击如何有效地用于评估DRL代理的弹性的详细洞察，从而为增强其健壮性铺平了道路。



## **6. DrAttack: Prompt Decomposition and Reconstruction Makes Powerful LLM Jailbreakers**

DrAttack：快速分解和重构使LLM成为强大的越狱者 cs.CR

**SubmitDate**: 2024-03-01    [abs](http://arxiv.org/abs/2402.16914v2) [paper-pdf](http://arxiv.org/pdf/2402.16914v2)

**Authors**: Xirui Li, Ruochen Wang, Minhao Cheng, Tianyi Zhou, Cho-Jui Hsieh

**Abstract**: The safety alignment of Large Language Models (LLMs) is vulnerable to both manual and automated jailbreak attacks, which adversarially trigger LLMs to output harmful content. However, current methods for jailbreaking LLMs, which nest entire harmful prompts, are not effective at concealing malicious intent and can be easily identified and rejected by well-aligned LLMs. This paper discovers that decomposing a malicious prompt into separated sub-prompts can effectively obscure its underlying malicious intent by presenting it in a fragmented, less detectable form, thereby addressing these limitations. We introduce an automatic prompt \textbf{D}ecomposition and \textbf{R}econstruction framework for jailbreak \textbf{Attack} (DrAttack). DrAttack includes three key components: (a) `Decomposition' of the original prompt into sub-prompts, (b) `Reconstruction' of these sub-prompts implicitly by in-context learning with semantically similar but harmless reassembling demo, and (c) a `Synonym Search' of sub-prompts, aiming to find sub-prompts' synonyms that maintain the original intent while jailbreaking LLMs. An extensive empirical study across multiple open-source and closed-source LLMs demonstrates that, with a significantly reduced number of queries, DrAttack obtains a substantial gain of success rate over prior SOTA prompt-only attackers. Notably, the success rate of 78.0\% on GPT-4 with merely 15 queries surpassed previous art by 33.1\%. The project is available at https://github.com/xirui-li/DrAttack.

摘要: 大型语言模型(LLM)的安全一致性容易受到手动和自动越狱攻击的攻击，这些攻击会相反地触发LLM输出有害内容。然而，当前的越狱LLM方法嵌套了整个有害的提示，在隐藏恶意意图方面并不有效，很容易被排列良好的LLM识别和拒绝。本文发现，通过将恶意提示分解为单独的子提示，可以通过以零散的、较难检测的形式来表示恶意提示，从而有效地掩盖潜在的恶意意图，从而解决这些限制。介绍了一种自动提示的文本bf{D}分解和文本bf{R}重构框架(DrAttack)。DrAttack包括三个关键组件：(A)将原始提示‘分解’成子提示，(B)通过使用语义相似但无害的重组演示的上下文学习隐式地‘重构’这些子提示，以及(C)对子提示进行‘同步搜索’，目的是在越狱LLM的同时找到保持原意的子提示的同义词。一项针对多个开源和闭源LLM的广泛经验研究表明，DrAttack在查询次数显著减少的情况下，与之前的Sota仅提示攻击者相比，获得了显著的成功率。值得注意的是，仅用15个查询在GPT-4上的成功率为78.0\%，比以前的ART高出33.1\%。该项目的网址为：https://github.com/xirui-li/DrAttack.。



## **7. SoK: Security of Programmable Logic Controllers**

SOK：可编程逻辑控制器的安全性 cs.CR

25 pages, 13 figures, Extended version February 2024, A shortened  version is to be published in the 33rd USENIX Security Symposium, for more  information, see https://efrenlopez.org/

**SubmitDate**: 2024-03-01    [abs](http://arxiv.org/abs/2403.00280v1) [paper-pdf](http://arxiv.org/pdf/2403.00280v1)

**Authors**: Efrén López-Morales, Ulysse Planta, Carlos Rubio-Medrano, Ali Abbasi, Alvaro A. Cardenas

**Abstract**: Billions of people rely on essential utility and manufacturing infrastructures such as water treatment plants, energy management, and food production. Our dependence on reliable infrastructures makes them valuable targets for cyberattacks. One of the prime targets for adversaries attacking physical infrastructures are Programmable Logic Controllers (PLCs) because they connect the cyber and physical worlds. In this study, we conduct the first comprehensive systematization of knowledge that explores the security of PLCs: We present an in-depth analysis of PLC attacks and defenses and discover trends in the security of PLCs from the last 17 years of research. We introduce a novel threat taxonomy for PLCs and Industrial Control Systems (ICS). Finally, we identify and point out research gaps that, if left ignored, could lead to new catastrophic attacks against critical infrastructures.

摘要: 数十亿人依赖于基本的公用事业和制造业基础设施，如水处理厂、能源管理和食品生产。我们对可靠基础设施的依赖使它们成为网络攻击的宝贵目标。攻击者攻击物理基础设施的主要目标之一是可编程逻辑控制器(PLC)，因为它们连接了网络和物理世界。在这项研究中，我们首次对PLC的安全性进行了全面的系统化研究：我们对PLC的攻击和防御进行了深入的分析，并从过去17年的研究中发现了PLC安全性的趋势。我们介绍了一种新的PLC和工业控制系统(ICS)威胁分类方法。最后，我们确定并指出了研究差距，如果忽视这些差距，可能会导致对关键基础设施的新的灾难性攻击。



## **8. LoRA-as-an-Attack! Piercing LLM Safety Under The Share-and-Play Scenario**

LoRA攻击在共享和播放场景下穿透LLM安全 cs.CR

**SubmitDate**: 2024-02-29    [abs](http://arxiv.org/abs/2403.00108v1) [paper-pdf](http://arxiv.org/pdf/2403.00108v1)

**Authors**: Hongyi Liu, Zirui Liu, Ruixiang Tang, Jiayi Yuan, Shaochen Zhong, Yu-Neng Chuang, Li Li, Rui Chen, Xia Hu

**Abstract**: Fine-tuning LLMs is crucial to enhancing their task-specific performance and ensuring model behaviors are aligned with human preferences. Among various fine-tuning methods, LoRA is popular for its efficiency and ease to use, allowing end-users to easily post and adopt lightweight LoRA modules on open-source platforms to tailor their model for different customization. However, such a handy share-and-play setting opens up new attack surfaces, that the attacker can render LoRA as an attacker, such as backdoor injection, and widely distribute the adversarial LoRA to the community easily. This can result in detrimental outcomes. Despite the huge potential risks of sharing LoRA modules, this aspect however has not been fully explored. To fill the gap, in this study we thoroughly investigate the attack opportunities enabled in the growing share-and-play scenario. Specifically, we study how to inject backdoor into the LoRA module and dive deeper into LoRA's infection mechanisms. We found that training-free mechanism is possible in LoRA backdoor injection. We also discover the impact of backdoor attacks with the presence of multiple LoRA adaptions concurrently as well as LoRA based backdoor transferability. Our aim is to raise awareness of the potential risks under the emerging share-and-play scenario, so as to proactively prevent potential consequences caused by LoRA-as-an-Attack. Warning: the paper contains potential offensive content generated by models.

摘要: 微调LLM对于增强其特定于任务的性能并确保模型行为与人类偏好一致至关重要。在各种微调方法中，Lora以其高效和易用而广受欢迎，允许最终用户在开源平台上轻松发布和采用轻量级Lora模块，以针对不同的定制定制他们的模型。然而，这种方便的共享和游戏设置打开了新的攻击面，使得攻击者可以将Lora呈现为攻击者，例如后门注入，并轻松地将具有敌意的Lora广泛分发到社区。这可能会导致有害的结果。尽管共享LORA模块存在巨大的潜在风险，但这方面还没有得到充分的探索。为了填补这一空白，在这项研究中，我们彻底调查了在日益增长的共享和游戏场景中实现的攻击机会。具体地说，我们研究了如何向Lora模块注入后门程序，并更深入地研究Lora的感染机制。我们发现，在LORA后门注射中可能存在无需训练的机制。我们还发现了同时存在多个LORA适配以及基于LORA的后门可转移性的后门攻击的影响。我们的目的是提高人们对新出现的共享和玩耍情景下潜在风险的认识，以便主动预防劳拉即攻击造成的潜在后果。警告：本文包含由模型生成的潜在攻击性内容。



## **9. Unraveling Adversarial Examples against Speaker Identification -- Techniques for Attack Detection and Victim Model Classification**

揭开说话人识别的敌意实例--攻击检测和受害者模型分类技术 cs.SD

**SubmitDate**: 2024-02-29    [abs](http://arxiv.org/abs/2402.19355v1) [paper-pdf](http://arxiv.org/pdf/2402.19355v1)

**Authors**: Sonal Joshi, Thomas Thebaud, Jesús Villalba, Najim Dehak

**Abstract**: Adversarial examples have proven to threaten speaker identification systems, and several countermeasures against them have been proposed. In this paper, we propose a method to detect the presence of adversarial examples, i.e., a binary classifier distinguishing between benign and adversarial examples. We build upon and extend previous work on attack type classification by exploring new architectures. Additionally, we introduce a method for identifying the victim model on which the adversarial attack is carried out. To achieve this, we generate a new dataset containing multiple attacks performed against various victim models. We achieve an AUC of 0.982 for attack detection, with no more than a 0.03 drop in performance for unknown attacks. Our attack classification accuracy (excluding benign) reaches 86.48% across eight attack types using our LightResNet34 architecture, while our victim model classification accuracy reaches 72.28% across four victim models.

摘要: 对抗性的例子已经被证明威胁到说话人识别系统，并提出了一些针对它们的对策。在本文中，我们提出了一种检测对抗性样本存在的方法，即区分良性和对抗性样本的二进制分类器。我们通过探索新的体系结构，在以前关于攻击类型分类的工作的基础上进行了扩展。此外，我们还介绍了一种识别受害者模型的方法，在该模型上进行对抗性攻击。为了实现这一点，我们生成了一个新的数据集，其中包含对各种受害者模型执行的多个攻击。我们实现了攻击检测的AUC值为0.982，而对于未知攻击，性能下降不超过0.03%。使用LightResNet34架构，我们的攻击分类准确率(不包括良性攻击)在八种攻击类型上达到86.48%，而我们的受害者模型分类准确率在四种受害者模型上达到72.28%。



## **10. Verification of Neural Networks' Global Robustness**

神经网络的全局健壮性验证 cs.LG

**SubmitDate**: 2024-02-29    [abs](http://arxiv.org/abs/2402.19322v1) [paper-pdf](http://arxiv.org/pdf/2402.19322v1)

**Authors**: Anan Kabaha, Dana Drachsler-Cohen

**Abstract**: Neural networks are successful in various applications but are also susceptible to adversarial attacks. To show the safety of network classifiers, many verifiers have been introduced to reason about the local robustness of a given input to a given perturbation. While successful, local robustness cannot generalize to unseen inputs. Several works analyze global robustness properties, however, neither can provide a precise guarantee about the cases where a network classifier does not change its classification. In this work, we propose a new global robustness property for classifiers aiming at finding the minimal globally robust bound, which naturally extends the popular local robustness property for classifiers. We introduce VHAGaR, an anytime verifier for computing this bound. VHAGaR relies on three main ideas: encoding the problem as a mixed-integer programming and pruning the search space by identifying dependencies stemming from the perturbation or network computation and generalizing adversarial attacks to unknown inputs. We evaluate VHAGaR on several datasets and classifiers and show that, given a three hour timeout, the average gap between the lower and upper bound on the minimal globally robust bound computed by VHAGaR is 1.9, while the gap of an existing global robustness verifier is 154.7. Moreover, VHAGaR is 130.6x faster than this verifier. Our results further indicate that leveraging dependencies and adversarial attacks makes VHAGaR 78.6x faster.

摘要: 神经网络在各种应用中都很成功，但也容易受到对抗性攻击。为了表明网络分类器的安全性，已经引入了许多验证器来推理给定输入对给定扰动的局部稳健性。虽然取得了成功，但局部稳健性不能推广到看不见的输入。然而，一些工作分析了全局健壮性，但都不能提供关于网络分类器不改变其分类的情况的精确保证。在这项工作中，我们提出了一种新的分类器的全局稳健性，旨在寻找最小的全局稳健界，这自然地扩展了流行的分类器的局部稳健性。我们介绍了VHAGaR，一个计算这个界的随时验证器。VHAGaR依赖于三个主要思想：将问题编码为混合整数规划，通过识别源于扰动或网络计算的依赖来剪枝搜索空间，以及将敌意攻击推广到未知输入。我们在几个数据集和分类器上对VHAGaR进行了评估，结果表明，在超时3小时的情况下，VHAGaR计算的最小全局健壮界的上下界之间的平均差距为1.9%，而现有的全局健壮性验证器的差距为154.7。此外，VHAGaR比该验证器快130.6倍。我们的结果进一步表明，利用依赖关系和对抗性攻击使VHAGaR的速度提高了78.6倍。



## **11. Attacks Against Mobility Prediction in 5G Networks**

针对5G网络移动性预测的攻击 cs.CR

This is the preprint version of a paper which appears in 22th IEEE  International Conference on Trust, Security and Privacy in Computing and  Communications (TrustCom 2023)

**SubmitDate**: 2024-02-29    [abs](http://arxiv.org/abs/2402.19319v1) [paper-pdf](http://arxiv.org/pdf/2402.19319v1)

**Authors**: Syafiq Al Atiiq, Yachao Yuan, Christian Gehrmann, Jakob Sternby, Luis Barriga

**Abstract**: The $5^{th}$ generation of mobile networks introduces a new Network Function (NF) that was not present in previous generations, namely the Network Data Analytics Function (NWDAF). Its primary objective is to provide advanced analytics services to various entities within the network and also towards external application services in the 5G ecosystem. One of the key use cases of NWDAF is mobility trajectory prediction, which aims to accurately support efficient mobility management of User Equipment (UE) in the network by allocating ``just in time'' necessary network resources. In this paper, we show that there are potential mobility attacks that can compromise the accuracy of these predictions. In a semi-realistic scenario with 10,000 subscribers, we demonstrate that an adversary equipped with the ability to hijack cellular mobile devices and clone them can significantly reduce the prediction accuracy from 75\% to 40\% using just 100 adversarial UEs. While a defense mechanism largely depends on the attack and the mobility types in a particular area, we prove that a basic KMeans clustering is effective in distinguishing legitimate and adversarial UEs.

摘要: 第5代移动网络引入了前几代没有的新网络功能，即网络数据分析功能(NWDAF)。其主要目标是向网络内的各种实体以及5G生态系统中的外部应用服务提供高级分析服务。NWDAF的关键用例之一是移动性轨迹预测，其目的是通过分配必要的“及时”网络资源，准确地支持网络中用户设备(UE)的有效移动性管理。在这篇文章中，我们证明了潜在的移动攻击可能会损害这些预测的准确性。在一个有10,000个用户的半现实场景中，我们证明了一个拥有劫持蜂窝移动设备并克隆它们的能力的对手可以显著地降低预测精度，仅使用100个对手UE就可以将预测精度从75%降低到40%。虽然防御机制在很大程度上取决于特定区域内的攻击和移动类型，但我们证明了基本的KMeans聚类在区分合法和敌对用户方面是有效的。



## **12. Topology-Based Reconstruction Prevention for Decentralised Learning**

基于拓扑的分布式学习重构预防 cs.CR

13 pages, 8 figures, submitted to PETS 2024, for associated  experiment source code see doi:10.4121/21572601

**SubmitDate**: 2024-02-29    [abs](http://arxiv.org/abs/2312.05248v2) [paper-pdf](http://arxiv.org/pdf/2312.05248v2)

**Authors**: Florine W. Dekker, Zekeriya Erkin, Mauro Conti

**Abstract**: Decentralised learning has recently gained traction as an alternative to federated learning in which both data and coordination are distributed over its users. To preserve data confidentiality, decentralised learning relies on differential privacy, multi-party computation, or a combination thereof. However, running multiple privacy-preserving summations in sequence may allow adversaries to perform reconstruction attacks. Unfortunately, current reconstruction countermeasures either cannot trivially be adapted to the distributed setting, or add excessive amounts of noise.   In this work, we first show that passive honest-but-curious adversaries can infer other users' private data after several privacy-preserving summations. For example, in subgraphs with 18 users, we show that only three passive honest-but-curious adversaries succeed at reconstructing private data 11.0% of the time, requiring an average of 8.8 summations per adversary. The success rate depends only on the adversaries' direct neighbourhood, independent of the size of the full network. We consider weak adversaries, who do not control the graph topology and can exploit neither the inner workings of the summation protocol nor the specifics of users' data.   We develop a mathematical understanding of how reconstruction relates to topology and propose the first topology-based decentralised defence against reconstruction attacks. Specifically, we show that reconstruction requires a number of adversaries linear in the length of the network's shortest cycle. Consequently, reconstructing private data from privacy-preserving summations is impossible in acyclic networks.   Our work is a stepping stone for a formal theory of topology-based reconstruction defences. Such a theory would generalise our countermeasure beyond summation, define confidentiality in terms of entropy, and describe the effects of differential privacy.

摘要: 分散式学习最近作为联合学习的替代方案获得了吸引力，在联合学习中，数据和协调都分布在用户身上。为了保护数据的机密性，分散学习依赖于差异隐私、多方计算或它们的组合。但是，按顺序运行多个隐私保护摘要可能会允许攻击者执行重建攻击。不幸的是，当前的重建对策要么不能简单地适应分布式设置，要么增加了过多的噪声。在这项工作中，我们首先证明了被动诚实但好奇的对手可以在几次隐私保护汇总后推断出其他用户的私人数据。例如，在具有18个用户的子图中，我们表明只有三个被动的诚实但好奇的对手在11.0%的时间内成功重建私人数据，每个对手平均需要8.8次求和。成功率仅取决于对手的直接邻居，与整个网络的规模无关。我们考虑的是弱对手，他们不控制图形拓扑，既不能利用求和协议的内部工作原理，也不能利用用户数据的细节。我们发展了对重构如何与拓扑相关的数学理解，并提出了第一个基于拓扑的分布式防御重构攻击的方法。具体地说，我们证明了重建需要若干个与网络最短周期长度成线性关系的对手。因此，在非循环网络中，从保护隐私的求和中重建私有数据是不可能的。我们的工作是一个形式化的基于拓扑的重建防御理论的垫脚石。这样的理论将概括我们的对策，超越总和，用熵来定义机密性，并描述不同隐私的影响。



## **13. How to Train your Antivirus: RL-based Hardening through the Problem-Space**

如何培训您的防病毒软件：问题空间中基于RL的强化 cs.CR

20 pages,4 figures

**SubmitDate**: 2024-02-29    [abs](http://arxiv.org/abs/2402.19027v1) [paper-pdf](http://arxiv.org/pdf/2402.19027v1)

**Authors**: Jacopo Cortellazzi, Ilias Tsingenopoulos, Branislav Bošanský, Simone Aonzo, Davy Preuveneers, Wouter Joosen, Fabio Pierazzi, Lorenzo Cavallaro

**Abstract**: ML-based malware detection on dynamic analysis reports is vulnerable to both evasion and spurious correlations. In this work, we investigate a specific ML architecture employed in the pipeline of a widely-known commercial antivirus company, with the goal to harden it against adversarial malware. Adversarial training, the sole defensive technique that can confer empirical robustness, is not applicable out of the box in this domain, for the principal reason that gradient-based perturbations rarely map back to feasible problem-space programs. We introduce a novel Reinforcement Learning approach for constructing adversarial examples, a constituent part of adversarially training a model against evasion. Our approach comes with multiple advantages. It performs modifications that are feasible in the problem-space, and only those; thus it circumvents the inverse mapping problem. It also makes possible to provide theoretical guarantees on the robustness of the model against a particular set of adversarial capabilities. Our empirical exploration validates our theoretical insights, where we can consistently reach 0\% Attack Success Rate after a few adversarial retraining iterations.

摘要: 动态分析报告上基于ML的恶意软件检测容易受到规避和虚假关联的攻击。在这项工作中，我们调查了一个特定的ML体系结构，该体系结构应用于一家著名的商业反病毒公司的流水线中，目的是加强它对敌对恶意软件的攻击。对抗性训练是唯一可以赋予经验稳健性的防御技术，在这个领域不适用，主要原因是基于梯度的扰动很少映射回可行的问题空间程序。我们介绍了一种新的强化学习方法来构建对抗性例子，这是对抗性训练模型的一个组成部分。我们的方法具有多方面的优势。它执行在问题空间中可行的修改，并且只执行那些修改；因此，它绕过了逆映射问题。它还可以在理论上保证模型在对抗一组特定对手能力时的健壮性。我们的经验探索验证了我们的理论见解，经过几次对抗性的再训练迭代，我们可以一致地达到0攻击成功率。



## **14. Invariant Aggregator for Defending against Federated Backdoor Attacks**

用于防御联合后门攻击的不变聚合器 cs.LG

**SubmitDate**: 2024-02-29    [abs](http://arxiv.org/abs/2210.01834v3) [paper-pdf](http://arxiv.org/pdf/2210.01834v3)

**Authors**: Xiaoyang Wang, Dimitrios Dimitriadis, Sanmi Koyejo, Shruti Tople

**Abstract**: Federated learning enables training high-utility models across several clients without directly sharing their private data. As a downside, the federated setting makes the model vulnerable to various adversarial attacks in the presence of malicious clients. Despite the theoretical and empirical success in defending against attacks that aim to degrade models' utility, defense against backdoor attacks that increase model accuracy on backdoor samples exclusively without hurting the utility on other samples remains challenging. To this end, we first analyze the failure modes of existing defenses over a flat loss landscape, which is common for well-designed neural networks such as Resnet [He et al., 2015] but is often overlooked by previous works. Then, we propose an invariant aggregator that redirects the aggregated update to invariant directions that are generally useful via selectively masking out the update elements that favor few and possibly malicious clients. Theoretical results suggest that our approach provably mitigates backdoor attacks and remains effective over flat loss landscapes. Empirical results on three datasets with different modalities and varying numbers of clients further demonstrate that our approach mitigates a broad class of backdoor attacks with a negligible cost on the model utility.

摘要: 联合学习允许在多个客户之间培训高实用模型，而无需直接共享他们的私人数据。缺点是，联合设置使模型在存在恶意客户端的情况下容易受到各种敌意攻击。尽管在防御旨在降低模型效用的攻击方面取得了理论和经验上的成功，但针对后门攻击的防御仍然具有挑战性，这种攻击只能提高后门样本的模型精度，而不会损害其他样本的效用。为此，我们首先分析了平坦损失情况下现有防御的故障模式，这在RESNET等设计良好的神经网络中很常见[他等人，2015]，但通常被以前的工作忽视。然后，我们提出了一个不变聚集器，它通过有选择地屏蔽有利于少数甚至可能是恶意客户端的更新元素，将聚合的更新重定向到通常有用的不变方向。理论结果表明，我们的方法可以有效地减少后门攻击，并且在平价损失场景下仍然有效。在三个具有不同模式和不同客户端数量的数据集上的实验结果进一步表明，我们的方法以可以忽略不计的模型效用代价缓解了广泛类别的后门攻击。



## **15. MPAT: Building Robust Deep Neural Networks against Textual Adversarial Attacks**

MPAT：构建抵抗文本攻击的健壮深度神经网络 cs.LG

**SubmitDate**: 2024-02-29    [abs](http://arxiv.org/abs/2402.18792v1) [paper-pdf](http://arxiv.org/pdf/2402.18792v1)

**Authors**: Fangyuan Zhang, Huichi Zhou, Shuangjiao Li, Hongtao Wang

**Abstract**: Deep neural networks have been proven to be vulnerable to adversarial examples and various methods have been proposed to defend against adversarial attacks for natural language processing tasks. However, previous defense methods have limitations in maintaining effective defense while ensuring the performance of the original task. In this paper, we propose a malicious perturbation based adversarial training method (MPAT) for building robust deep neural networks against textual adversarial attacks. Specifically, we construct a multi-level malicious example generation strategy to generate adversarial examples with malicious perturbations, which are used instead of original inputs for model training. Additionally, we employ a novel training objective function to ensure achieving the defense goal without compromising the performance on the original task. We conduct comprehensive experiments to evaluate our defense method by attacking five victim models on three benchmark datasets. The result demonstrates that our method is more effective against malicious adversarial attacks compared with previous defense methods while maintaining or further improving the performance on the original task.

摘要: 深度神经网络已经被证明容易受到敌意例子的攻击，并且已经提出了各种方法来防御自然语言处理任务的对抗性攻击。然而，以往的防御方式在保证原任务执行的同时，在保持有效防御方面存在局限性。本文提出了一种基于恶意扰动的对抗训练方法(MPAT)，用于构建稳健的深层神经网络，以抵御文本攻击。具体地说，我们构建了一个多层次的恶意实例生成策略来生成带有恶意扰动的对抗性实例，这些实例被用来代替原始输入进行模型训练。此外，我们采用了一种新的训练目标函数来确保在不影响原始任务的性能的情况下实现防御目标。通过在三个基准数据集上攻击五个受害者模型，我们进行了全面的实验来评估我们的防御方法。实验结果表明，与以往的防御方法相比，该方法在保持或进一步提高了原任务的性能的同时，对恶意对抗性攻击具有更强的抵抗能力。



## **16. Enhancing the "Immunity" of Mixture-of-Experts Networks for Adversarial Defense**

增强混合专家网络对抗防御的“免疫力” cs.LG

**SubmitDate**: 2024-02-29    [abs](http://arxiv.org/abs/2402.18787v1) [paper-pdf](http://arxiv.org/pdf/2402.18787v1)

**Authors**: Qiao Han, yong huang, xinling Guo, Yiteng Zhai, Yu Qin, Yao Yang

**Abstract**: Recent studies have revealed the vulnerability of Deep Neural Networks (DNNs) to adversarial examples, which can easily fool DNNs into making incorrect predictions. To mitigate this deficiency, we propose a novel adversarial defense method called "Immunity" (Innovative MoE with MUtual information \& positioN stabilITY) based on a modified Mixture-of-Experts (MoE) architecture in this work. The key enhancements to the standard MoE are two-fold: 1) integrating of Random Switch Gates (RSGs) to obtain diverse network structures via random permutation of RSG parameters at evaluation time, despite of RSGs being determined after one-time training; 2) devising innovative Mutual Information (MI)-based and Position Stability-based loss functions by capitalizing on Grad-CAM's explanatory power to increase the diversity and the causality of expert networks. Notably, our MI-based loss operates directly on the heatmaps, thereby inducing subtler negative impacts on the classification performance when compared to other losses of the same type, theoretically. Extensive evaluation validates the efficacy of the proposed approach in improving adversarial robustness against a wide range of attacks.

摘要: 最近的研究揭示了深度神经网络(DNN)对敌意例子的脆弱性，这些例子很容易欺骗DNN做出错误的预测。针对这一不足，本文提出了一种新的基于改进的混合专家体系结构的对抗性防御方法“免疫”。对标准MOE的关键改进有两个方面：1)集成了随机开关门(RSGs)，在评估时通过随机排列RSG参数获得不同的网络结构，尽管RSGs是在一次性训练后确定的；2)利用Grad-CAM的解释能力，设计了创新的基于互信息(MI)和基于位置稳定性的损失函数，增加了专家网络的多样性和因果关系。值得注意的是，我们的基于MI的损失直接作用于热图，从而在理论上与其他相同类型的损失相比，对分类性能产生了更微妙的负面影响。



## **17. On Defeating Graph Analysis of Anonymous Transactions**

关于匿名交易的失败图分析 cs.CR

**SubmitDate**: 2024-02-28    [abs](http://arxiv.org/abs/2402.18755v1) [paper-pdf](http://arxiv.org/pdf/2402.18755v1)

**Authors**: Christoph Egger, Russell W. F. Lai, Viktoria Ronge, Ivy K. Y. Woo, Hoover H. F. Yin

**Abstract**: In a ring-signature-based anonymous cryptocurrency, signers of a transaction are hidden among a set of potential signers, called a ring, whose size is much smaller than the number of all users. The ring-membership relations specified by the sets of transactions thus induce bipartite transaction graphs, whose distribution is in turn induced by the ring sampler underlying the cryptocurrency.   Since efficient graph analysis could be performed on transaction graphs to potentially deanonymise signers, it is crucial to understand the resistance of (the transaction graphs induced by) a ring sampler against graph analysis. Of particular interest is the class of partitioning ring samplers. Although previous works showed that they provide almost optimal local anonymity, their resistance against global, e.g. graph-based, attacks were unclear.   In this work, we analyse transaction graphs induced by partitioning ring samplers. Specifically, we show (partly analytically and partly empirically) that, somewhat surprisingly, by setting the ring size to be at least logarithmic in the number of users, a graph-analysing adversary is no better than the one that performs random guessing in deanonymisation up to constant factor of 2.

摘要: 在基于环签名的匿名加密货币中，交易的签名者隐藏在一组潜在的签名者中，称为环，其大小远远小于所有用户的数量。因此，由事务集合指定的环成员关系产生二部事务图，其分布又由作为加密货币的基础的环采样器来诱导。由于可以对交易图进行有效的图分析以潜在地去匿名化签名者，因此了解环采样器(由环采样器产生的)对图分析的抵抗力是至关重要的。特别令人感兴趣的是分隔环采样器的类别。虽然以前的工作表明它们提供了几乎最优的局部匿名性，但它们对全局攻击(例如基于图的攻击)的抵抗力尚不清楚。在这项工作中，我们分析了划分环采样器所产生的事务图。具体地说，我们证明了(部分是分析的，部分是经验的)，有些令人惊讶的是，通过将环的大小至少设置为用户数的对数，图分析对手并不比在去名化中执行随机猜测的对手好到2。



## **18. A New Era in LLM Security: Exploring Security Concerns in Real-World LLM-based Systems**

LLM安全的新时代：探索现实世界中基于LLM的系统的安全问题 cs.CR

**SubmitDate**: 2024-02-28    [abs](http://arxiv.org/abs/2402.18649v1) [paper-pdf](http://arxiv.org/pdf/2402.18649v1)

**Authors**: Fangzhou Wu, Ning Zhang, Somesh Jha, Patrick McDaniel, Chaowei Xiao

**Abstract**: Large Language Model (LLM) systems are inherently compositional, with individual LLM serving as the core foundation with additional layers of objects such as plugins, sandbox, and so on. Along with the great potential, there are also increasing concerns over the security of such probabilistic intelligent systems. However, existing studies on LLM security often focus on individual LLM, but without examining the ecosystem through the lens of LLM systems with other objects (e.g., Frontend, Webtool, Sandbox, and so on). In this paper, we systematically analyze the security of LLM systems, instead of focusing on the individual LLMs. To do so, we build on top of the information flow and formulate the security of LLM systems as constraints on the alignment of the information flow within LLM and between LLM and other objects. Based on this construction and the unique probabilistic nature of LLM, the attack surface of the LLM system can be decomposed into three key components: (1) multi-layer security analysis, (2) analysis of the existence of constraints, and (3) analysis of the robustness of these constraints. To ground this new attack surface, we propose a multi-layer and multi-step approach and apply it to the state-of-art LLM system, OpenAI GPT4. Our investigation exposes several security issues, not just within the LLM model itself but also in its integration with other components. We found that although the OpenAI GPT4 has designed numerous safety constraints to improve its safety features, these safety constraints are still vulnerable to attackers. To further demonstrate the real-world threats of our discovered vulnerabilities, we construct an end-to-end attack where an adversary can illicitly acquire the user's chat history, all without the need to manipulate the user's input or gain direct access to OpenAI GPT4. Our demo is in the link: https://fzwark.github.io/LLM-System-Attack-Demo/

摘要: 大型语言模型（Large Language Model，LLM）系统是一种组合式系统，以单个LLM为核心，辅以插件、沙箱等对象层，在具有巨大潜力的同时，其安全性也日益受到关注。然而，现有的关于LLM安全的研究通常集中在单个LLM上，但没有通过LLM系统与其他对象（例如，Frontend、Webtool、Sandbox等）。在本文中，我们系统地分析了LLM系统的安全性，而不是专注于单个LLM。要做到这一点，我们建立在信息流的顶部，并制定LLM系统的安全性的LLM内的信息流和LLM和其他对象之间的对齐的约束。基于这种结构和LLM独特的概率性质，LLM系统的攻击面可以分解为三个关键部分：（1）多层安全性分析，（2）约束的存在性分析，以及（3）这些约束的鲁棒性分析。为了使这种新的攻击面接地，我们提出了一种多层和多步骤的方法，并将其应用于最先进的LLM系统OpenAI GPT4。我们的调查暴露了几个安全问题，不仅在LLM模型本身，而且在与其他组件的集成中。我们发现，尽管OpenAI GPT4设计了许多安全约束来改进其安全功能，但这些安全约束仍然容易受到攻击者的攻击。为了进一步展示我们发现的漏洞的现实威胁，我们构建了一个端到端攻击，对手可以非法获取用户的聊天记录，而无需操纵用户的输入或直接访问OpenAI GPT4。我们的演示是在链接：https://fzwark.github.io/LLM-System-Attack-Demo/



## **19. Model Predictive Control with adaptive resilience for Denial-of-Service Attacks mitigation on a Regulated Dam**

具有自适应弹性的模型预测控制在调节大坝拒绝服务攻击缓解中的应用 eess.SY

**SubmitDate**: 2024-02-28    [abs](http://arxiv.org/abs/2402.18516v1) [paper-pdf](http://arxiv.org/pdf/2402.18516v1)

**Authors**: Raffaele Giuseppe Cestari, Stefano Longari, Stefano Zanero, Simone Formentin

**Abstract**: In recent years, SCADA (Supervisory Control and Data Acquisition) systems have increasingly become the target of cyber attacks. SCADAs are no longer isolated, as web-based applications expose strategic infrastructures to the outside world connection. In a cyber-warfare context, we propose a Model Predictive Control (MPC) architecture with adaptive resilience, capable of guaranteeing control performance in normal operating conditions and driving towards resilience against DoS (controller-actuator) attacks when needed. Since the attackers' goal is typically to maximize the system damage, we assume they solve an adversarial optimal control problem. An adaptive resilience factor is then designed as a function of the intensity function of a Hawkes process, a point process model estimating the occurrence of random events in time, trained on a moving window to estimate the return time of the next attack. We demonstrate the resulting MPC strategy's effectiveness in 2 attack scenarios on a real system with actual data, the regulated Olginate dam of Lake Como.

摘要: 近年来，SCADA(Supervisor Control And Data Acquisition)系统日益成为网络攻击的目标。SCADA不再是孤立的，因为基于Web的应用程序将战略基础设施暴露给外部世界连接。在网络战背景下，我们提出了一种具有自适应弹性的模型预测控制(MPC)体系结构，能够在正常运行条件下保证控制性能，并在需要时驱动对DoS(控制器-执行器)攻击的弹性。由于攻击者的目标通常是最大化系统损害，我们假设他们解决了一个对抗性最优控制问题。然后，将自适应弹性因子设计为Hawkes过程强度函数的函数，该点过程模型及时估计随机事件的发生，并在移动窗口上训练以估计下一次攻击的返回时间。我们用实际数据演示了所得到的MPC策略在两个攻击场景中的有效性，该系统是科莫湖受管制的Olgate大坝。



## **20. DevPhish: Exploring Social Engineering in Software Supply Chain Attacks on Developers**

DevPhish：探索软件供应链中的社会工程对开发人员的攻击 cs.SE

7 pages, 2 figures

**SubmitDate**: 2024-02-28    [abs](http://arxiv.org/abs/2402.18401v1) [paper-pdf](http://arxiv.org/pdf/2402.18401v1)

**Authors**: Hossein Siadati, Sima Jafarikhah, Elif Sahin, Terrence Brent Hernandez, Elijah Lorenzo Tripp, Denis Khryashchev

**Abstract**: The Software Supply Chain (SSC) has captured considerable attention from attackers seeking to infiltrate systems and undermine organizations. There is evidence indicating that adversaries utilize Social Engineering (SocE) techniques specifically aimed at software developers. That is, they interact with developers at critical steps in the Software Development Life Cycle (SDLC), such as accessing Github repositories, incorporating code dependencies, and obtaining approval for Pull Requests (PR) to introduce malicious code. This paper aims to comprehensively explore the existing and emerging SocE tactics employed by adversaries to trick Software Engineers (SWEs) into delivering malicious software. By analyzing a diverse range of resources, which encompass established academic literature and real-world incidents, the paper systematically presents an overview of these manipulative strategies within the realm of the SSC. Such insights prove highly beneficial for threat modeling and security gap analysis.

摘要: 软件供应链(SSC)已经引起了寻求渗透系统和破坏组织的攻击者的相当大的关注。有证据表明，攻击者利用专门针对软件开发人员的社会工程(SOCE)技术。也就是说，他们在软件开发生命周期(SDLC)的关键步骤与开发人员交互，例如访问Github存储库、合并代码依赖项以及获得引入恶意代码的拉请求(PR)的批准。本文旨在全面探讨现有的和新兴的SOCE策略，这些策略被对手用来诱骗软件工程师(SWE)交付恶意软件。通过分析各种资源，包括既定的学术文献和现实世界的事件，本文系统地概述了这些在SSC领域内的操纵策略。事实证明，这些见解对威胁建模和安全差距分析非常有用。



## **21. Neuromorphic Event-Driven Semantic Communication in Microgrids**

微网格中神经形态事件驱动的语义交流 cs.ET

The manuscript has been accepted for publication in IEEE Transactions  on Smart Grid

**SubmitDate**: 2024-02-28    [abs](http://arxiv.org/abs/2402.18390v1) [paper-pdf](http://arxiv.org/pdf/2402.18390v1)

**Authors**: Xiaoguang Diao, Yubo Song, Subham Sahoo, Yuan Li

**Abstract**: Synergies between advanced communications, computing and artificial intelligence are unraveling new directions of coordinated operation and resiliency in microgrids. On one hand, coordination among sources is facilitated by distributed, privacy-minded processing at multiple locations, whereas on the other hand, it also creates exogenous data arrival paths for adversaries that can lead to cyber-physical attacks amongst other reliability issues in the communication layer. This long-standing problem necessitates new intrinsic ways of exchanging information between converters through power lines to optimize the system's control performance. Going beyond the existing power and data co-transfer technologies that are limited by efficiency and scalability concerns, this paper proposes neuromorphic learning to implant communicative features using spiking neural networks (SNNs) at each node, which is trained collaboratively in an online manner simply using the power exchanges between the nodes. As opposed to the conventional neuromorphic sensors that operate with spiking signals, we employ an event-driven selective process to collect sparse data for training of SNNs. Finally, its multi-fold effectiveness and reliable performance is validated under simulation conditions with different microgrid topologies and components to establish a new direction in the sense-actuate-compute cycle for power electronic dominated grids and microgrids.

摘要: 先进的通信、计算和人工智能之间的协同作用正在瓦解微电网协调运行和弹性的新方向。一方面，通过在多个位置进行具有隐私意识的分布式处理来促进源之间的协调，而另一方面，它也为可能导致通信层中的网络物理攻击和其他可靠性问题的对手创建外部数据到达路径。这个长期存在的问题需要新的内在方式来通过电力线在变流器之间交换信息，以优化系统的控制性能。超越了现有的能量和数据协同传输技术的效率和可扩展性的限制，提出了神经形态学习，在每个节点上使用尖峰神经网络(SNN)植入通信特征，该网络只需利用节点之间的能量交换就可以在线协作训练。与传统的与尖峰信号一起工作的神经形态传感器不同，我们使用事件驱动的选择性过程来收集稀疏数据来训练SNN。最后，在不同微网拓扑和元件的仿真条件下，验证了该算法的多重有效性和可靠性能，为电力电子主导电网和微电网的感知-执行-计算循环奠定了新的方向。



## **22. A Game-theoretic Framework for Privacy-preserving Federated Learning**

保护隐私的联邦学习的博弈论框架 cs.LG

**SubmitDate**: 2024-02-28    [abs](http://arxiv.org/abs/2304.05836v3) [paper-pdf](http://arxiv.org/pdf/2304.05836v3)

**Authors**: Xiaojin Zhang, Lixin Fan, Siwei Wang, Wenjie Li, Kai Chen, Qiang Yang

**Abstract**: In federated learning, benign participants aim to optimize a global model collaboratively. However, the risk of \textit{privacy leakage} cannot be ignored in the presence of \textit{semi-honest} adversaries. Existing research has focused either on designing protection mechanisms or on inventing attacking mechanisms. While the battle between defenders and attackers seems never-ending, we are concerned with one critical question: is it possible to prevent potential attacks in advance? To address this, we propose the first game-theoretic framework that considers both FL defenders and attackers in terms of their respective payoffs, which include computational costs, FL model utilities, and privacy leakage risks. We name this game the federated learning privacy game (FLPG), in which neither defenders nor attackers are aware of all participants' payoffs.   To handle the \textit{incomplete information} inherent in this situation, we propose associating the FLPG with an \textit{oracle} that has two primary responsibilities. First, the oracle provides lower and upper bounds of the payoffs for the players. Second, the oracle acts as a correlation device, privately providing suggested actions to each player. With this novel framework, we analyze the optimal strategies of defenders and attackers. Furthermore, we derive and demonstrate conditions under which the attacker, as a rational decision-maker, should always follow the oracle's suggestion \textit{not to attack}.

摘要: 在联合学习中，良性参与者的目标是协作优化全球模型。然而，在存在半诚实的对手的情况下，隐私泄露的风险是不容忽视的。现有的研究要么集中在设计保护机制上，要么集中在发明攻击机制上。虽然防御者和攻击者之间的战斗似乎永无止境，但我们关心的是一个关键问题：是否有可能提前防止潜在的攻击？为了解决这一问题，我们提出了第一个博弈论框架，该框架考虑了FL防御者和攻击者各自的收益，其中包括计算成本、FL模型效用和隐私泄露风险。我们将这款游戏命名为联邦学习隐私游戏(FLPG)，在该游戏中，防御者和攻击者都不知道所有参与者的收益。为了处理这种情况下固有的不完整信息，我们建议将FLPG与具有两个主要职责的\textit{Oracle}相关联。首先，先知为玩家提供了收益的上下限。其次，先知充当了关联设备，私下向每个玩家提供建议的动作。在此框架下，我们分析了防御者和攻击者的最优策略。此外，我们还推导并证明了攻击者作为理性决策者应始终遵循神谕的建议的条件。



## **23. Living-off-The-Land Reverse-Shell Detection by Informed Data Augmentation**

基于信息数据增强的生物反壳检测 cs.CR

**SubmitDate**: 2024-02-28    [abs](http://arxiv.org/abs/2402.18329v1) [paper-pdf](http://arxiv.org/pdf/2402.18329v1)

**Authors**: Dmitrijs Trizna, Luca Demetrio, Battista Biggio, Fabio Roli

**Abstract**: The living-off-the-land (LOTL) offensive methodologies rely on the perpetration of malicious actions through chains of commands executed by legitimate applications, identifiable exclusively by analysis of system logs. LOTL techniques are well hidden inside the stream of events generated by common legitimate activities, moreover threat actors often camouflage activity through obfuscation, making them particularly difficult to detect without incurring in plenty of false alarms, even using machine learning. To improve the performance of models in such an harsh environment, we propose an augmentation framework to enhance and diversify the presence of LOTL malicious activity inside legitimate logs. Guided by threat intelligence, we generate a dataset by injecting attack templates known to be employed in the wild, further enriched by malleable patterns of legitimate activities to replicate the behavior of evasive threat actors. We conduct an extensive ablation study to understand which models better handle our augmented dataset, also manipulated to mimic the presence of model-agnostic evasion and poisoning attacks. Our results suggest that augmentation is needed to maintain high-predictive capabilities, robustness to attack is achieved through specific hardening techniques like adversarial training, and it is possible to deploy near-real-time models with almost-zero false alarms.

摘要: 赖以生存(LOTL)的攻击性方法依赖于通过合法应用程序执行的命令链实施恶意操作，这些命令链只能通过分析系统日志来识别。LOTL技术很好地隐藏在普通合法活动产生的事件流中，此外，威胁参与者经常通过混淆来伪装活动，使得它们特别难在不引起大量错误警报的情况下被检测到，即使使用机器学习也是如此。为了提高模型在这种恶劣环境中的性能，我们提出了一个增强框架来增强和多样化合法日志中LOTL恶意活动的存在。在威胁情报的指导下，我们通过注入已知在野外使用的攻击模板来生成数据集，并通过合法活动的可塑性模式进一步丰富，以复制规避威胁参与者的行为。我们进行了一项广泛的消融研究，以了解哪些模型更好地处理了我们的扩展数据集，该数据集也被操纵以模拟模型不可知性逃避和中毒攻击的存在。我们的结果表明，增强是保持高预测能力所必需的，对攻击的稳健性是通过特定的强化技术(如对抗性训练)来实现的，并且有可能部署几乎为零错误警报的近实时模型。



## **24. Black-box Adversarial Attacks Against Image Quality Assessment Models**

针对图像质量评估模型的黑盒对抗性攻击 cs.CV

**SubmitDate**: 2024-02-28    [abs](http://arxiv.org/abs/2402.17533v2) [paper-pdf](http://arxiv.org/pdf/2402.17533v2)

**Authors**: Yu Ran, Ao-Xiang Zhang, Mingjie Li, Weixuan Tang, Yuan-Gen Wang

**Abstract**: The goal of No-Reference Image Quality Assessment (NR-IQA) is to predict the perceptual quality of an image in line with its subjective evaluation. To put the NR-IQA models into practice, it is essential to study their potential loopholes for model refinement. This paper makes the first attempt to explore the black-box adversarial attacks on NR-IQA models. Specifically, we first formulate the attack problem as maximizing the deviation between the estimated quality scores of original and perturbed images, while restricting the perturbed image distortions for visual quality preservation. Under such formulation, we then design a Bi-directional loss function to mislead the estimated quality scores of adversarial examples towards an opposite direction with maximum deviation. On this basis, we finally develop an efficient and effective black-box attack method against NR-IQA models. Extensive experiments reveal that all the evaluated NR-IQA models are vulnerable to the proposed attack method. And the generated perturbations are not transferable, enabling them to serve the investigation of specialities of disparate IQA models.

摘要: 无参考图像质量评估(NR-IQA)的目的是根据图像的主观评价来预测图像的感知质量。为了将NR-IQA模型付诸实践，有必要研究它们对模型求精的潜在漏洞。本文首次尝试探讨了对NR-IQA模型的黑盒对抗攻击。具体地说，我们首先将攻击问题描述为最大化原始图像和扰动图像的估计质量分数之间的偏差，同时限制扰动图像的失真以保持视觉质量。在此基础上，设计了一个双向损失函数，将对抗性样本的估计质量分数误导到偏差最大的相反方向。在此基础上，提出了一种针对NR-IQA模型的高效黑盒攻击方法。大量的实验表明，所有被评估的NR-IQA模型都容易受到所提出的攻击方法的攻击。并且产生的扰动是不可转移的，使它们能够服务于不同IQA模型的特长的调查。



## **25. Embodied Adversarial Attack: A Dynamic Robust Physical Attack in Autonomous Driving**

具体化对抗性攻击：自主驾驶中的一种动态健壮身体攻击 cs.CV

10 pages, 7 figures

**SubmitDate**: 2024-02-28    [abs](http://arxiv.org/abs/2312.09554v2) [paper-pdf](http://arxiv.org/pdf/2312.09554v2)

**Authors**: Yitong Sun, Yao Huang, Xingxing Wei

**Abstract**: As physical adversarial attacks become extensively applied in unearthing the potential risk of security-critical scenarios, especially in autonomous driving, their vulnerability to environmental changes has also been brought to light. The non-robust nature of physical adversarial attack methods brings less-than-stable performance consequently. To enhance the robustness of physical adversarial attacks in the real world, instead of statically optimizing a robust adversarial example via an off-line training manner like the existing methods, this paper proposes a brand new robust adversarial attack framework: Embodied Adversarial Attack (EAA) from the perspective of dynamic adaptation, which aims to employ the paradigm of embodied intelligence: Perception-Decision-Control to dynamically adjust the optimal attack strategy according to the current situations in real time. For the perception module, given the challenge of needing simulation for the victim's viewpoint, EAA innovatively devises a Perspective Transformation Network to estimate the target's transformation from the attacker's perspective. For the decision and control module, EAA adopts the laser-a highly manipulable medium to implement physical attacks, and further trains an attack agent with reinforcement learning to make it capable of instantaneously determining the best attack strategy based on the perceived information. Finally, we apply our framework to the autonomous driving scenario. A variety of experiments verify the high effectiveness of our method under complex scenes.

摘要: 随着物理对抗性攻击在挖掘安全关键场景的潜在风险方面得到广泛应用，特别是在自动驾驶中，它们对环境变化的脆弱性也暴露了出来。因此，物理对抗性攻击方法的非健壮性带来了不稳定的性能。为了提高物理对抗攻击在现实世界中的健壮性，不像现有方法那样通过离线训练的方式静态地优化健壮的对抗实例，从动态适应的角度提出了一种全新的健壮对抗攻击框架：具体化对抗性攻击(EAA)，旨在利用具体化智能的范式：感知-决策-控制来实时地根据当前情况动态地调整最优攻击策略。对于感知模块，针对受害者视点需要模拟的挑战，EAA创新性地设计了一个视角转换网络来从攻击者的角度估计目标的变化。对于决策和控制模块，EAA采用激光这一高度可操控的介质来实施物理攻击，并进一步使用强化学习来训练攻击代理，使其能够根据感知的信息即时确定最佳攻击策略。最后，我们将我们的框架应用于自动驾驶场景。大量实验验证了该方法在复杂场景下的高效性。



## **26. Towards Transferable Targeted 3D Adversarial Attack in the Physical World**

在物理世界中走向可转移的定向3D对抗攻击 cs.CV

Accepted by CVPR 2024

**SubmitDate**: 2024-02-28    [abs](http://arxiv.org/abs/2312.09558v2) [paper-pdf](http://arxiv.org/pdf/2312.09558v2)

**Authors**: Yao Huang, Yinpeng Dong, Shouwei Ruan, Xiao Yang, Hang Su, Xingxing Wei

**Abstract**: Compared with transferable untargeted attacks, transferable targeted adversarial attacks could specify the misclassification categories of adversarial samples, posing a greater threat to security-critical tasks. In the meanwhile, 3D adversarial samples, due to their potential of multi-view robustness, can more comprehensively identify weaknesses in existing deep learning systems, possessing great application value. However, the field of transferable targeted 3D adversarial attacks remains vacant. The goal of this work is to develop a more effective technique that could generate transferable targeted 3D adversarial examples, filling the gap in this field. To achieve this goal, we design a novel framework named TT3D that could rapidly reconstruct from few multi-view images into Transferable Targeted 3D textured meshes. While existing mesh-based texture optimization methods compute gradients in the high-dimensional mesh space and easily fall into local optima, leading to unsatisfactory transferability and distinct distortions, TT3D innovatively performs dual optimization towards both feature grid and Multi-layer Perceptron (MLP) parameters in the grid-based NeRF space, which significantly enhances black-box transferability while enjoying naturalness. Experimental results show that TT3D not only exhibits superior cross-model transferability but also maintains considerable adaptability across different renders and vision tasks. More importantly, we produce 3D adversarial examples with 3D printing techniques in the real world and verify their robust performance under various scenarios.

摘要: 与可转移无目标攻击相比，可转移有目标对抗攻击可以指定对抗样本的误分类类别，对安全关键任务构成更大的威胁。同时，3D对抗样本由于其潜在的多视图鲁棒性，可以更全面地识别现有深度学习系统的弱点，具有很大的应用价值。然而，可转移的有针对性的3D对抗性攻击领域仍然空缺。这项工作的目标是开发一种更有效的技术，可以生成可转移的有针对性的3D对抗性示例，填补这一领域的空白。为了实现这一目标，我们设计了一个新的框架TT3D，可以快速重建从几个多视图图像到可转移的目标三维纹理网格。现有的基于网格的纹理优化方法在高维网格空间中计算梯度，容易陷入局部最优，导致可移植性不佳，失真明显，TT3D创新性地在基于网格的NeRF空间中对特征网格和多层感知器（MLP）参数进行双重优化，在享受自然的同时显著增强了黑盒可移植性。实验结果表明，TT3D不仅具有良好的跨模型可移植性，而且在不同的渲染和视觉任务之间保持了相当大的适应性。更重要的是，我们在现实世界中使用3D打印技术制作了3D对抗性示例，并验证了它们在各种场景下的鲁棒性能。



## **27. Exploring Privacy and Fairness Risks in Sharing Diffusion Models: An Adversarial Perspective**

共享扩散模型中的隐私和公平风险：对抗性视角 cs.LG

**SubmitDate**: 2024-02-28    [abs](http://arxiv.org/abs/2402.18607v1) [paper-pdf](http://arxiv.org/pdf/2402.18607v1)

**Authors**: Xinjian Luo, Yangfan Jiang, Fei Wei, Yuncheng Wu, Xiaokui Xiao, Beng Chin Ooi

**Abstract**: Diffusion models have recently gained significant attention in both academia and industry due to their impressive generative performance in terms of both sampling quality and distribution coverage. Accordingly, proposals are made for sharing pre-trained diffusion models across different organizations, as a way of improving data utilization while enhancing privacy protection by avoiding sharing private data directly. However, the potential risks associated with such an approach have not been comprehensively examined.   In this paper, we take an adversarial perspective to investigate the potential privacy and fairness risks associated with the sharing of diffusion models. Specifically, we investigate the circumstances in which one party (the sharer) trains a diffusion model using private data and provides another party (the receiver) black-box access to the pre-trained model for downstream tasks. We demonstrate that the sharer can execute fairness poisoning attacks to undermine the receiver's downstream models by manipulating the training data distribution of the diffusion model. Meanwhile, the receiver can perform property inference attacks to reveal the distribution of sensitive features in the sharer's dataset. Our experiments conducted on real-world datasets demonstrate remarkable attack performance on different types of diffusion models, which highlights the critical importance of robust data auditing and privacy protection protocols in pertinent applications.

摘要: 扩散模型由于其在抽样质量和分布复盖率方面令人印象深刻的生成性能，最近在学术界和工业界都得到了极大的关注。因此，提出了在不同组织之间共享预先培训的传播模式的建议，以此作为提高数据利用率的一种方式，同时通过避免直接共享私人数据来加强隐私保护。然而，与这种方法相关的潜在风险尚未得到全面审查。在这篇文章中，我们采取对抗性的视角来调查与共享扩散模型相关的潜在的隐私和公平风险。具体地说，我们调查了一方(共享者)使用私有数据训练扩散模型，并为另一方(接收者)提供对下游任务的预训练模型的黑箱访问的情况。我们证明了共享者可以通过操纵扩散模型的训练数据分布来执行公平毒化攻击来破坏接收者的下游模型。同时，接收者可以执行属性推断攻击，以揭示共享者数据集中敏感特征的分布。我们在真实数据集上进行的实验表明，在不同类型的扩散模型上具有显著的攻击性能，这突显了健壮的数据审计和隐私保护协议在相关应用中的关键重要性。



## **28. The NISQ Complexity of Collision Finding**

碰撞发现的NISQ复杂性 quant-ph

40 pages; v2: title changed, major extension to other complexity  models

**SubmitDate**: 2024-02-28    [abs](http://arxiv.org/abs/2211.12954v2) [paper-pdf](http://arxiv.org/pdf/2211.12954v2)

**Authors**: Yassine Hamoudi, Qipeng Liu, Makrand Sinha

**Abstract**: Collision-resistant hashing, a fundamental primitive in modern cryptography, ensures that there is no efficient way to find distinct inputs that produce the same hash value. This property underpins the security of various cryptographic applications, making it crucial to understand its complexity. The complexity of this problem is well-understood in the classical setting and $\Theta(N^{1/2})$ queries are needed to find a collision. However, the advent of quantum computing has introduced new challenges since quantum adversaries $\unicode{x2013}$ equipped with the power of quantum queries $\unicode{x2013}$ can find collisions much more efficiently. Brassard, H\"oyer and Tapp and Aaronson and Shi established that full-scale quantum adversaries require $\Theta(N^{1/3})$ queries to find a collision, prompting a need for longer hash outputs, which impacts efficiency in terms of the key lengths needed for security.   This paper explores the implications of quantum attacks in the Noisy-Intermediate Scale Quantum (NISQ) era. In this work, we investigate three different models for NISQ algorithms and achieve tight bounds for all of them:   (1) A hybrid algorithm making adaptive quantum or classical queries but with a limited quantum query budget, or   (2) A quantum algorithm with access to a noisy oracle, subject to a dephasing or depolarizing channel, or   (3) A hybrid algorithm with an upper bound on its maximum quantum depth; i.e., a classical algorithm aided by low-depth quantum circuits.   In fact, our results handle all regimes between NISQ and full-scale quantum computers. Previously, only results for the pre-image search problem were known for these models by Sun and Zheng, Rosmanis, Chen, Cotler, Huang and Li while nothing was known about the collision finding problem.

摘要: 抗冲突散列是现代密码学中的一个基本原理，它确保没有有效的方法来找到产生相同散列值的不同输入。此属性支持各种加密应用程序的安全性，因此了解其复杂性至关重要。这个问题的复杂性在经典设置中是很好理解的，并且需要$\theta(N^{1/2})$查询来查找冲突。然而，量子计算的出现带来了新的挑战，因为配备了量子查询能力的量子对手$\unicode{x2013}$可以更有效地发现碰撞。Brassard、H“Oyer和Tapp以及Aaronson和Shih确定，全面的量子对手需要$\theta(N^{1/3})$查询来发现冲突，这促使需要更长的散列输出，这就影响了安全所需密钥长度的效率。本文探讨了噪声中尺度量子(NISQ)时代量子攻击的含义。在这项工作中，我们研究了三种不同的NISQ算法模型，并得到了它们的严格界限：(1)进行自适应量子或经典查询的混合算法，但量子查询预算有限；(2)访问噪声预言的量子算法，受去相或去极化通道的影响；(3)具有最大量子深度上限的混合算法；即，由低深度量子电路辅助的经典算法。事实上，我们的结果处理了NISQ和全尺寸量子计算机之间的所有机制。此前，对于这些模型，Sun和郑、Rosmanis、Chen、Cotler、Huang和Li只知道前图像搜索问题的结果，而对碰撞发现问题一无所知。



## **29. Catastrophic Overfitting: A Potential Blessing in Disguise**

灾难性的过度适应：变相的潜在祝福 cs.LG

**SubmitDate**: 2024-02-28    [abs](http://arxiv.org/abs/2402.18211v1) [paper-pdf](http://arxiv.org/pdf/2402.18211v1)

**Authors**: Mengnan Zhao, Lihe Zhang, Yuqiu Kong, Baocai Yin

**Abstract**: Fast Adversarial Training (FAT) has gained increasing attention within the research community owing to its efficacy in improving adversarial robustness. Particularly noteworthy is the challenge posed by catastrophic overfitting (CO) in this field. Although existing FAT approaches have made strides in mitigating CO, the ascent of adversarial robustness occurs with a non-negligible decline in classification accuracy on clean samples. To tackle this issue, we initially employ the feature activation differences between clean and adversarial examples to analyze the underlying causes of CO. Intriguingly, our findings reveal that CO can be attributed to the feature coverage induced by a few specific pathways. By intentionally manipulating feature activation differences in these pathways with well-designed regularization terms, we can effectively mitigate and induce CO, providing further evidence for this observation. Notably, models trained stably with these terms exhibit superior performance compared to prior FAT work. On this basis, we harness CO to achieve `attack obfuscation', aiming to bolster model performance. Consequently, the models suffering from CO can attain optimal classification accuracy on both clean and adversarial data when adding random noise to inputs during evaluation. We also validate their robustness against transferred adversarial examples and the necessity of inducing CO to improve robustness. Hence, CO may not be a problem that has to be solved.

摘要: 快速对抗训练(FAT)由于其在提高对抗健壮性方面的有效性，在研究界得到了越来越多的关注。特别值得注意的是这一领域的灾难性过适应(CO)所带来的挑战。虽然现有的FAT方法在减轻CO方面取得了很大进展，但在提高对抗性稳健性的同时，对干净样本的分类精度也出现了不可忽视的下降。为了解决这个问题，我们首先利用正例和对抗性例子之间的特征激活差异来分析CO的潜在原因。有趣的是，我们的发现表明，CO可以归因于由几个特定路径诱导的特征覆盖。通过有意识地使用设计良好的正则化项来操纵这些通路中的特征激活差异，我们可以有效地缓解和诱导CO，为这一观察提供了进一步的证据。值得注意的是，与以前的FAT工作相比，使用这些条件稳定训练的模型显示出更好的性能。在此基础上，我们利用CO来实现“攻击混淆”，旨在提高模型的性能。因此，当在评估过程中向输入添加随机噪声时，受到CO影响的模型在清洁数据和对抗性数据上都能获得最优的分类精度。我们还验证了它们对转移的敌意例子的稳健性，以及引入CO来提高稳健性的必要性。因此，CO可能不是一个必须解决的问题。



## **30. On the Robustness of Bayesian Neural Networks to Adversarial Attacks**

贝叶斯神经网络对敌方攻击的稳健性研究 cs.LG

arXiv admin note: text overlap with arXiv:2002.04359

**SubmitDate**: 2024-02-28    [abs](http://arxiv.org/abs/2207.06154v3) [paper-pdf](http://arxiv.org/pdf/2207.06154v3)

**Authors**: Luca Bortolussi, Ginevra Carbone, Luca Laurenti, Andrea Patane, Guido Sanguinetti, Matthew Wicker

**Abstract**: Vulnerability to adversarial attacks is one of the principal hurdles to the adoption of deep learning in safety-critical applications. Despite significant efforts, both practical and theoretical, training deep learning models robust to adversarial attacks is still an open problem. In this paper, we analyse the geometry of adversarial attacks in the large-data, overparameterized limit for Bayesian Neural Networks (BNNs). We show that, in the limit, vulnerability to gradient-based attacks arises as a result of degeneracy in the data distribution, i.e., when the data lies on a lower-dimensional submanifold of the ambient space. As a direct consequence, we demonstrate that in this limit BNN posteriors are robust to gradient-based adversarial attacks. Crucially, we prove that the expected gradient of the loss with respect to the BNN posterior distribution is vanishing, even when each neural network sampled from the posterior is vulnerable to gradient-based attacks. Experimental results on the MNIST, Fashion MNIST, and half moons datasets, representing the finite data regime, with BNNs trained with Hamiltonian Monte Carlo and Variational Inference, support this line of arguments, showing that BNNs can display both high accuracy on clean data and robustness to both gradient-based and gradient-free based adversarial attacks.

摘要: 对敌意攻击的脆弱性是在安全关键应用中采用深度学习的主要障碍之一。尽管在实践和理论上都做了大量的努力，但训练对对手攻击稳健的深度学习模型仍然是一个悬而未决的问题。本文分析了贝叶斯神经网络(BNN)在大数据、过参数限制下的攻击几何。我们证明，在极限情况下，由于数据分布的退化，即当数据位于环境空间的低维子流形上时，对基于梯度的攻击的脆弱性出现。作为一个直接的推论，我们证明了在这个极限下，BNN后验网络对基于梯度的敌意攻击是稳健的。重要的是，我们证明了损失相对于BNN后验分布的期望梯度是零的，即使从后验采样的每个神经网络都容易受到基于梯度的攻击。在代表有限数据区的MNIST、Fashion MNIST和半月数据集上的实验结果支持这一论点，BNN采用哈密顿蒙特卡罗和变分推理进行训练，表明BNN在干净数据上具有很高的准确率，并且对基于梯度和基于无梯度的敌意攻击都具有很好的鲁棒性。



## **31. Understanding the Role of Pathways in a Deep Neural Network**

理解通路在深度神经网络中的作用 cs.CV

**SubmitDate**: 2024-02-28    [abs](http://arxiv.org/abs/2402.18132v1) [paper-pdf](http://arxiv.org/pdf/2402.18132v1)

**Authors**: Lei Lyu, Chen Pang, Jihua Wang

**Abstract**: Deep neural networks have demonstrated superior performance in artificial intelligence applications, but the opaqueness of their inner working mechanism is one major drawback in their application. The prevailing unit-based interpretation is a statistical observation of stimulus-response data, which fails to show a detailed internal process of inherent mechanisms of neural networks. In this work, we analyze a convolutional neural network (CNN) trained in the classification task and present an algorithm to extract the diffusion pathways of individual pixels to identify the locations of pixels in an input image associated with object classes. The pathways allow us to test the causal components which are important for classification and the pathway-based representations are clearly distinguishable between categories. We find that the few largest pathways of an individual pixel from an image tend to cross the feature maps in each layer that is important for classification. And the large pathways of images of the same category are more consistent in their trends than those of different categories. We also apply the pathways to understanding adversarial attacks, object completion, and movement perception. Further, the total number of pathways on feature maps in all layers can clearly discriminate the original, deformed, and target samples.

摘要: 深度神经网络在人工智能应用中表现出了优越的性能，但其内部工作机制的不透明性是其应用中的一大缺陷。目前流行的基于单元的解释是对刺激-反应数据的统计观察，它不能显示神经网络内在机制的详细内部过程。在这项工作中，我们分析了在分类任务中训练的卷积神经网络(CNN)，并提出了一种算法来提取单个像素的扩散路径，以识别输入图像中与对象类关联的像素的位置。路径允许我们测试对分类重要的因果成分，并且基于路径的表示在类别之间是明显可区分的。我们发现，图像中单个像素的少数几条最大路径往往会穿过每一层中的特征地图，这对分类非常重要。同一类别图像的大路径比不同类别图像的大路径在趋势上更一致。我们还将这些路径应用于理解对抗性攻击、对象完成和运动感知。此外，所有层的特征地图上的路径总数可以清楚地区分原始样本、变形样本和目标样本。



## **32. Making Them Ask and Answer: Jailbreaking Large Language Models in Few Queries via Disguise and Reconstruction**

让他们提问和回答：通过伪装和重建在几个查询中越狱大型语言模型 cs.CR

**SubmitDate**: 2024-02-28    [abs](http://arxiv.org/abs/2402.18104v1) [paper-pdf](http://arxiv.org/pdf/2402.18104v1)

**Authors**: Tong Liu, Yingjie Zhang, Zhe Zhao, Yinpeng Dong, Guozhu Meng, Kai Chen

**Abstract**: In recent years, large language models (LLMs) have demonstrated notable success across various tasks, but the trustworthiness of LLMs is still an open problem. One specific threat is the potential to generate toxic or harmful responses. Attackers can craft adversarial prompts that induce harmful responses from LLMs. In this work, we pioneer a theoretical foundation in LLMs security by identifying bias vulnerabilities within the safety fine-tuning and design a black-box jailbreak method named DRA (Disguise and Reconstruction Attack), which conceals harmful instructions through disguise and prompts the model to reconstruct the original harmful instruction within its completion. We evaluate DRA across various open-source and close-source models, showcasing state-of-the-art jailbreak success rates and attack efficiency. Notably, DRA boasts a 90\% attack success rate on LLM chatbots GPT-4.

摘要: 近年来，大型语言模型在各种任务上取得了显著的成功，但大型语言模型的可信度仍然是一个悬而未决的问题。一个具体的威胁是可能产生有毒或有害的反应。攻击者可以精心编制敌意提示，以诱导LLMS做出有害的响应。在这项工作中，我们通过识别安全微调中的偏差漏洞，开创了LLMS安全的理论基础，并设计了一种称为DRA(伪装和重建攻击)的黑盒越狱方法，该方法通过伪装来隐藏有害指令，并促使模型在其完成的范围内重建原始有害指令。我们通过各种开源和封闭源代码模型对DRA进行评估，展示最先进的越狱成功率和攻击效率。值得注意的是，DRA对LLM聊天机器人GPT-4的攻击成功率高达90%。



## **33. Black-box Targeted Adversarial Attack on Segment Anything (SAM)**

针对Segment Anything(SAM)的黑箱定向对抗性攻击 cs.CV

**SubmitDate**: 2024-02-28    [abs](http://arxiv.org/abs/2310.10010v2) [paper-pdf](http://arxiv.org/pdf/2310.10010v2)

**Authors**: Sheng Zheng, Chaoning Zhang, Xinhong Hao

**Abstract**: Deep recognition models are widely vulnerable to adversarial examples, which change the model output by adding quasi-imperceptible perturbation to the image input. Recently, Segment Anything Model (SAM) has emerged to become a popular foundation model in computer vision due to its impressive generalization to unseen data and tasks. Realizing flexible attacks on SAM is beneficial for understanding the robustness of SAM in the adversarial context. To this end, this work aims to achieve a targeted adversarial attack (TAA) on SAM. Specifically, under a certain prompt, the goal is to make the predicted mask of an adversarial example resemble that of a given target image. The task of TAA on SAM has been realized in a recent arXiv work in the white-box setup by assuming access to prompt and model, which is thus less practical. To address the issue of prompt dependence, we propose a simple yet effective approach by only attacking the image encoder. Moreover, we propose a novel regularization loss to enhance the cross-model transferability by increasing the feature dominance of adversarial images over random natural images. Extensive experiments verify the effectiveness of our proposed simple techniques to conduct a successful black-box TAA on SAM.

摘要: 深度识别模型很容易受到敌意例子的影响，这些例子通过在图像输入中添加准不可察觉的扰动来改变模型输出。近年来，分段任意模型(Segment Anything Model，SAM)以其对未知数据和任务的良好泛化能力，成为计算机视觉中一种流行的基础模型。实现对SAM的灵活攻击有助于理解SAM在对抗环境下的健壮性。为此，本工作旨在实现对SAM的有针对性的对抗性攻击。具体地说，在一定的提示下，目标是使对抗性例子的预测掩模与给定目标图像的掩模相似。SAM上的TAA任务已经在最近的白盒设置中通过假设对提示和模型的访问来实现，因此这是不太实际的。为了解决提示依赖的问题，我们提出了一种简单而有效的方法，只攻击图像编码器。此外，我们提出了一种新的正则化损失，通过增加对抗性图像对随机自然图像的特征优势来增强跨模型的可转移性。大量的实验验证了我们提出的简单技术在SAM上成功进行黑盒TAA的有效性。



## **34. False Claims against Model Ownership Resolution**

针对所有权解决方案范本的虚假索赔 cs.CR

13pages,3 figures. To appear in the 33rd USENIX Security Symposium  (USENIX Security '24)

**SubmitDate**: 2024-02-28    [abs](http://arxiv.org/abs/2304.06607v4) [paper-pdf](http://arxiv.org/pdf/2304.06607v4)

**Authors**: Jian Liu, Rui Zhang, Sebastian Szyller, Kui Ren, N. Asokan

**Abstract**: Deep neural network (DNN) models are valuable intellectual property of model owners, constituting a competitive advantage. Therefore, it is crucial to develop techniques to protect against model theft. Model ownership resolution (MOR) is a class of techniques that can deter model theft. A MOR scheme enables an accuser to assert an ownership claim for a suspect model by presenting evidence, such as a watermark or fingerprint, to show that the suspect model was stolen or derived from a source model owned by the accuser. Most of the existing MOR schemes prioritize robustness against malicious suspects, ensuring that the accuser will win if the suspect model is indeed a stolen model.   In this paper, we show that common MOR schemes in the literature are vulnerable to a different, equally important but insufficiently explored, robustness concern: a malicious accuser. We show how malicious accusers can successfully make false claims against independent suspect models that were not stolen. Our core idea is that a malicious accuser can deviate (without detection) from the specified MOR process by finding (transferable) adversarial examples that successfully serve as evidence against independent suspect models. To this end, we first generalize the procedures of common MOR schemes and show that, under this generalization, defending against false claims is as challenging as preventing (transferable) adversarial examples. Via systematic empirical evaluation, we demonstrate that our false claim attacks always succeed in the MOR schemes that follow our generalization, including against a real-world model: Amazon's Rekognition API.

摘要: 深度神经网络(DNN)模型是模型所有者宝贵的知识产权，构成了竞争优势。因此，开发防止模型盗窃的技术至关重要。模型所有权解析(MOR)是一类能够阻止模型被盗的技术。MOR方案使原告能够通过出示证据(例如水印或指纹)来断言可疑模型的所有权主张，以显示可疑模型是被盗的或从原告拥有的源模型派生的。现有的大多数MOR方案都优先考虑针对恶意嫌疑人的健壮性，确保在可疑模型确实是被盗模型的情况下原告获胜。在这篇文章中，我们证明了文献中常见的MOR方案容易受到另一个同样重要但未被充分研究的健壮性问题的影响：恶意指控者。我们展示了恶意原告如何成功地对未被窃取的独立可疑模型做出虚假声明。我们的核心思想是，恶意指控者可以通过找到(可转移的)对抗性例子来偏离指定的MOR过程(而不被检测到)，这些例子成功地充当了针对独立嫌疑人模型的证据。为此，我们首先推广了常见MOR方案的步骤，并证明在这种推广下，对虚假声明的防御与防止(可转移)对抗性例子一样具有挑战性。通过系统的实证评估，我们证明了我们的虚假声明攻击在遵循我们的推广的MOR方案中总是成功的，包括针对真实世界的模型：亚马逊的Rekognition API。



## **35. Breaking the Black-Box: Confidence-Guided Model Inversion Attack for Distribution Shift**

打破黑箱：分布漂移的置信度引导模型反转攻击 cs.CV

8pages,5 figures

**SubmitDate**: 2024-02-28    [abs](http://arxiv.org/abs/2402.18027v1) [paper-pdf](http://arxiv.org/pdf/2402.18027v1)

**Authors**: Xinhao Liu, Yingzhao Jiang, Zetao Lin

**Abstract**: Model inversion attacks (MIAs) seek to infer the private training data of a target classifier by generating synthetic images that reflect the characteristics of the target class through querying the model. However, prior studies have relied on full access to the target model, which is not practical in real-world scenarios. Additionally, existing black-box MIAs assume that the image prior and target model follow the same distribution. However, when confronted with diverse data distribution settings, these methods may result in suboptimal performance in conducting attacks. To address these limitations, this paper proposes a \textbf{C}onfidence-\textbf{G}uided \textbf{M}odel \textbf{I}nversion attack method called CG-MI, which utilizes the latent space of a pre-trained publicly available generative adversarial network (GAN) as prior information and gradient-free optimizer, enabling high-resolution MIAs across different data distributions in a black-box setting. Our experiments demonstrate that our method significantly \textbf{outperforms the SOTA black-box MIA by more than 49\% for Celeba and 58\% for Facescrub in different distribution settings}. Furthermore, our method exhibits the ability to generate high-quality images \textbf{comparable to those produced by white-box attacks}. Our method provides a practical and effective solution for black-box model inversion attacks.

摘要: 模型反演攻击（MIA）试图通过查询模型来生成反映目标类别特征的合成图像，从而推断目标分类器的私有训练数据。然而，之前的研究依赖于对目标模型的完全访问，这在现实世界中是不切实际的。此外，现有的黑盒MIA假设图像先验和目标模型遵循相同的分布。然而，当面对不同的数据分布设置时，这些方法可能导致在进行攻击时的次优性能。为了解决这些局限性，本文提出了一种名为CG-MI的\textbf{C}置信-\textbf{G}引导的\textbf{M}模型\textbf{I}反转攻击方法，该方法利用预先训练的公共生成对抗网络（GAN）的潜在空间作为先验信息和无梯度优化器，在黑盒设置中实现跨不同数据分布的高分辨率MIA。我们的实验表明，我们的方法显着优于SOTA黑盒MIA超过49%的Celeba和58%的Facescrub在不同的分布设置}。此外，我们的方法具有生成高质量图像的能力\textbf{与白盒攻击产生的图像相当}。该方法为黑盒模型反演攻击提供了一种实用有效的解决方案。



## **36. Enhancing Tracking Robustness with Auxiliary Adversarial Defense Networks**

利用辅助对抗防御网络增强跟踪的稳健性 cs.CV

**SubmitDate**: 2024-02-28    [abs](http://arxiv.org/abs/2402.17976v1) [paper-pdf](http://arxiv.org/pdf/2402.17976v1)

**Authors**: Zhewei Wu, Ruilong Yu, Qihe Liu, Shuying Cheng, Shilin Qiu, Shijie Zhou

**Abstract**: Adversarial attacks in visual object tracking have significantly degraded the performance of advanced trackers by introducing imperceptible perturbations into images. These attack methods have garnered considerable attention from researchers in recent years. However, there is still a lack of research on designing adversarial defense methods specifically for visual object tracking. To address these issues, we propose an effective additional pre-processing network called DuaLossDef that eliminates adversarial perturbations during the tracking process. DuaLossDef is deployed ahead of the search branche or template branche of the tracker to apply defensive transformations to the input images. Moreover, it can be seamlessly integrated with other visual trackers as a plug-and-play module without requiring any parameter adjustments. We train DuaLossDef using adversarial training, specifically employing Dua-Loss to generate adversarial samples that simultaneously attack the classification and regression branches of the tracker. Extensive experiments conducted on the OTB100, LaSOT, and VOT2018 benchmarks demonstrate that DuaLossDef maintains excellent defense robustness against adversarial attack methods in both adaptive and non-adaptive attack scenarios. Moreover, when transferring the defense network to other trackers, it exhibits reliable transferability. Finally, DuaLossDef achieves a processing time of up to 5ms/frame, allowing seamless integration with existing high-speed trackers without introducing significant computational overhead. We will make our code publicly available soon.

摘要: 视觉目标跟踪中的对抗性攻击通过在图像中引入不可感知的扰动而显著降低了高级跟踪器的性能。近年来，这些攻击方法引起了研究人员的极大关注。然而，目前还缺乏专门针对视觉目标跟踪设计对抗性防御方法的研究。为了解决这些问题，我们提出了一个有效的附加预处理网络DuaLossDef，它消除了跟踪过程中的对抗性扰动。DuaLossDef部署在跟踪器的搜索分支或模板分支之前，以将防御性转换应用于输入图像。此外，它可以作为即插即用模块与其他视觉追踪器无缝集成，无需任何参数调整。我们使用对抗性训练来训练DuaLossDef，特别是使用Dua-Lost来生成同时攻击跟踪器的分类和回归分支的对抗性样本。在OTB100、LaSOT和VOT2018基准上进行的大量实验表明，DuaLossDef在自适应和非自适应攻击场景中都对对抗性攻击方法保持了良好的防御健壮性。此外，当将防御网络转移到其他跟踪器时，它表现出可靠的转移能力。最后，DuaLossDef实现了高达5ms/帧的处理时间，允许与现有的高速跟踪器无缝集成，而不会带来显著的计算开销。我们将很快公开我们的代码。



## **37. LLM-Resistant Math Word Problem Generation via Adversarial Attacks**

通过对抗性攻击生成LLM抵抗的数学应用题 cs.CL

Code is available at  https://github.com/ruoyuxie/adversarial_mwps_generation

**SubmitDate**: 2024-02-27    [abs](http://arxiv.org/abs/2402.17916v1) [paper-pdf](http://arxiv.org/pdf/2402.17916v1)

**Authors**: Roy Xie, Chengxuan Huang, Junlin Wang, Bhuwan Dhingra

**Abstract**: Large language models (LLMs) have significantly transformed the educational landscape. As current plagiarism detection tools struggle to keep pace with LLMs' rapid advancements, the educational community faces the challenge of assessing students' true problem-solving abilities in the presence of LLMs. In this work, we explore a new paradigm for ensuring fair evaluation -- generating adversarial examples which preserve the structure and difficulty of the original questions aimed for assessment, but are unsolvable by LLMs. Focusing on the domain of math word problems, we leverage abstract syntax trees to structurally generate adversarial examples that cause LLMs to produce incorrect answers by simply editing the numeric values in the problems. We conduct experiments on various open- and closed-source LLMs, quantitatively and qualitatively demonstrating that our method significantly degrades their math problem-solving ability. We identify shared vulnerabilities among LLMs and propose a cost-effective approach to attack high-cost models. Additionally, we conduct automatic analysis on math problems and investigate the cause of failure to guide future research on LLM's mathematical capability.

摘要: 大型语言模型（LLM）显著改变了教育格局。由于当前的剽窃检测工具难以跟上法学硕士的快速发展，教育界面临着在法学硕士在场的情况下评估学生真正解决问题能力的挑战。在这项工作中，我们探索了一种确保公平评估的新范式-生成对抗性示例，这些示例保留了旨在评估的原始问题的结构和难度，但LLM无法解决。专注于数学应用题领域，我们利用抽象语法树在结构上生成对抗性示例，这些示例通过简单地编辑问题中的数值而导致LLM产生不正确的答案。我们在各种开源和闭源LLM上进行实验，定量和定性地证明我们的方法显着降低了他们的数学问题解决能力。我们确定LLM之间的共享漏洞，并提出了一种具有成本效益的方法来攻击高成本的模型。此外，我们进行数学问题的自动分析，并调查失败的原因，以指导未来的研究LLM的数学能力。



## **38. Optimal Zero-Shot Detector for Multi-Armed Attacks**

适用于多臂攻击的最佳零射检测器 cs.LG

Accepted to appear in the 27th International Conference on Artificial  Intelligence and Statistics (AISTATS), May 2nd - May 4th, 2024 This article  supersedes arXiv:2302.02216

**SubmitDate**: 2024-02-27    [abs](http://arxiv.org/abs/2402.15808v2) [paper-pdf](http://arxiv.org/pdf/2402.15808v2)

**Authors**: Federica Granese, Marco Romanelli, Pablo Piantanida

**Abstract**: This paper explores a scenario in which a malicious actor employs a multi-armed attack strategy to manipulate data samples, offering them various avenues to introduce noise into the dataset. Our central objective is to protect the data by detecting any alterations to the input. We approach this defensive strategy with utmost caution, operating in an environment where the defender possesses significantly less information compared to the attacker. Specifically, the defender is unable to utilize any data samples for training a defense model or verifying the integrity of the channel. Instead, the defender relies exclusively on a set of pre-existing detectors readily available "off the shelf". To tackle this challenge, we derive an innovative information-theoretic defense approach that optimally aggregates the decisions made by these detectors, eliminating the need for any training data. We further explore a practical use-case scenario for empirical evaluation, where the attacker possesses a pre-trained classifier and launches well-known adversarial attacks against it. Our experiments highlight the effectiveness of our proposed solution, even in scenarios that deviate from the optimal setup.

摘要: 本文探讨了恶意攻击者使用多武装攻击策略来操纵数据样本的场景，为他们提供了向数据集中引入噪声的各种途径。我们的中心目标是通过检测输入的任何更改来保护数据。为了应对这一挑战，我们得出了一种创新的信息论防御方法，它最佳地聚合了这些检测器做出的决定，消除了对任何训练数据的需求。我们进一步探索了一个用于经验评估的实际用例场景，其中攻击者拥有一个预先训练的分类器，并对其发起众所周知的对抗性攻击。



## **39. Attention-GAN for Anomaly Detection: A Cutting-Edge Approach to Cybersecurity Threat Management**

异常检测注意：网络安全威胁管理的前沿方法 cs.CR

**SubmitDate**: 2024-02-27    [abs](http://arxiv.org/abs/2402.15945v2) [paper-pdf](http://arxiv.org/pdf/2402.15945v2)

**Authors**: Mohammed Abo Sen

**Abstract**: This paper proposes an innovative Attention-GAN framework for enhancing cybersecurity, focusing on anomaly detection. In response to the challenges posed by the constantly evolving nature of cyber threats, the proposed approach aims to generate diverse and realistic synthetic attack scenarios, thereby enriching the dataset and improving threat identification. Integrating attention mechanisms with Generative Adversarial Networks (GANs) is a key feature of the proposed method. The attention mechanism enhances the model's ability to focus on relevant features, essential for detecting subtle and complex attack patterns. In addition, GANs address the issue of data scarcity by generating additional varied attack data, encompassing known and emerging threats. This dual approach ensures that the system remains relevant and effective against the continuously evolving cyberattacks. The KDD Cup and CICIDS2017 datasets were used to validate this model, which exhibited significant improvements in anomaly detection. It achieved an accuracy of 99.69% on the KDD dataset and 97.93% on the CICIDS2017 dataset, with precision, recall, and F1-scores above 97%, demonstrating its effectiveness in recognizing complex attack patterns. This study contributes significantly to cybersecurity by providing a scalable and adaptable solution for anomaly detection in the face of sophisticated and dynamic cyber threats. The exploration of GANs for data augmentation highlights a promising direction for future research, particularly in situations where data limitations restrict the development of cybersecurity systems. The attention-GAN framework has emerged as a pioneering approach, setting a new benchmark for advanced cyber-defense strategies.

摘要: 本文提出了一个创新的Attention-GAN框架，用于增强网络安全，重点是异常检测。为了应对网络威胁不断演变的性质所带来的挑战，所提出的方法旨在生成多样化和逼真的合成攻击场景，从而丰富数据集并改进威胁识别。将注意力机制与生成对抗网络（GANs）集成是该方法的一个关键特征。注意力机制增强了模型关注相关特征的能力，这对于检测微妙和复杂的攻击模式至关重要。此外，GAN通过生成额外的各种攻击数据来解决数据稀缺问题，包括已知和新出现的威胁。这种双重方法确保了系统在应对不断变化的网络攻击时保持相关性和有效性。使用KDD Cup和CICIDS 2017数据集来验证该模型，该模型在异常检测方面表现出显着的改进。它在KDD数据集上的准确率为99.69%，在CICIDS 2017数据集上的准确率为97.93%，精确度，召回率和F1分数均高于97%，证明了其在识别复杂攻击模式方面的有效性。这项研究通过在面对复杂和动态的网络威胁时提供可扩展和适应性强的异常检测解决方案，为网络安全做出了重大贡献。GAN用于数据增强的探索突出了未来研究的一个有希望的方向，特别是在数据限制限制网络安全系统发展的情况下。Attention-GAN框架已经成为一种开创性的方法，为先进的网络防御战略设定了新的基准。



## **40. Follow My Instruction and Spill the Beans: Scalable Data Extraction from Retrieval-Augmented Generation Systems**

听从我的指示，泄漏豆子：从检索增强生成系统中提取可伸缩数据 cs.CL

**SubmitDate**: 2024-02-27    [abs](http://arxiv.org/abs/2402.17840v1) [paper-pdf](http://arxiv.org/pdf/2402.17840v1)

**Authors**: Zhenting Qi, Hanlin Zhang, Eric Xing, Sham Kakade, Himabindu Lakkaraju

**Abstract**: Retrieval-Augmented Generation (RAG) improves pre-trained models by incorporating external knowledge at test time to enable customized adaptation. We study the risk of datastore leakage in Retrieval-In-Context RAG Language Models (LMs). We show that an adversary can exploit LMs' instruction-following capabilities to easily extract text data verbatim from the datastore of RAG systems built with instruction-tuned LMs via prompt injection. The vulnerability exists for a wide range of modern LMs that span Llama2, Mistral/Mixtral, Vicuna, SOLAR, WizardLM, Qwen1.5, and Platypus2, and the exploitability exacerbates as the model size scales up. Extending our study to production RAG models GPTs, we design an attack that can cause datastore leakage with a 100% success rate on 25 randomly selected customized GPTs with at most 2 queries, and we extract text data verbatim at a rate of 41% from a book of 77,000 words and 3% from a corpus of 1,569,000 words by prompting the GPTs with only 100 queries generated by themselves.

摘要: 检索-增强生成(RAG)通过在测试时纳入外部知识来改进预先训练的模型，以实现定制适应。研究了上下文检索RAG语言模型(LMS)中数据存储泄漏的风险。我们表明，攻击者可以利用LMS的指令跟随能力，通过提示注入从使用指令调整的LMS构建的RAG系统的数据存储中轻松地逐字提取文本数据。该漏洞存在于跨越Llama2、Mistral/Mixtral、Vicuna、Solar、WizardLM、Qwen1.5和Platypus2的各种现代LMS中，并且随着模型大小的增加，可利用性会加剧。将我们的研究扩展到生产RAG模型GPTS，我们设计了一个可以导致数据存储泄漏的攻击，对于随机选择的最多2个查询的25个定制GPTS，成功率为100%；在一本77,000字的书中，我们以41%的成功率逐字提取文本数据，在1,569,000字的语料库中，我们通过提示GPT只生成100个查询，以3%的速度逐字提取文本数据。



## **41. BarraCUDA: GPUs do Leak DNN Weights**

梭鱼：图形处理器确实会泄漏DNN权重 cs.CR

**SubmitDate**: 2024-02-27    [abs](http://arxiv.org/abs/2312.07783v2) [paper-pdf](http://arxiv.org/pdf/2312.07783v2)

**Authors**: Peter Horvath, Lukasz Chmielewski, Leo Weissbart, Lejla Batina, Yuval Yarom

**Abstract**: Over the last decade, applications of neural networks (NNs) have spread to various aspects of our lives. A large number of companies base their businesses on building products that use neural networks for tasks such as face recognition, machine translation, and self-driving cars. Much of the intellectual property underpinning these products is encoded in the exact parameters of the neural networks. Consequently, protecting these is of utmost priority to businesses. At the same time, many of these products need to operate under a strong threat model, in which the adversary has unfettered physical control of the product. In this work, we present BarraCUDA, a novel attack on general purpose Graphic Processing Units (GPUs) that can extract parameters of neural networks running on the popular Nvidia Jetson Nano device. BarraCUDA uses correlation electromagnetic analysis to recover parameters of real-world convolutional neural networks.

摘要: 在过去的十年中，神经网络（NN）的应用已经扩展到我们生活的各个方面。许多公司的业务基础是构建使用神经网络执行人脸识别、机器翻译和自动驾驶汽车等任务的产品。支撑这些产品的大部分知识产权都编码在神经网络的精确参数中。因此，保护这些权利是企业的首要任务。与此同时，这些产品中的许多产品需要在强大的威胁模式下运行，在这种模式下，对手可以不受约束地对产品进行物理控制。在这项工作中，我们提出了BarraCUDA，这是一种对通用图形处理单元（GPU）的新型攻击，可以提取在流行的Nvidia Jetson Nano设备上运行的神经网络的参数。BarraCUDA使用相关电磁分析来恢复真实世界卷积神经网络的参数。



## **42. Extreme Miscalibration and the Illusion of Adversarial Robustness**

极端误校与对抗性稳健性的错觉 cs.CL

**SubmitDate**: 2024-02-27    [abs](http://arxiv.org/abs/2402.17509v1) [paper-pdf](http://arxiv.org/pdf/2402.17509v1)

**Authors**: Vyas Raina, Samson Tan, Volkan Cevher, Aditya Rawal, Sheng Zha, George Karypis

**Abstract**: Deep learning-based Natural Language Processing (NLP) models are vulnerable to adversarial attacks, where small perturbations can cause a model to misclassify. Adversarial Training (AT) is often used to increase model robustness. However, we have discovered an intriguing phenomenon: deliberately or accidentally miscalibrating models masks gradients in a way that interferes with adversarial attack search methods, giving rise to an apparent increase in robustness. We show that this observed gain in robustness is an illusion of robustness (IOR), and demonstrate how an adversary can perform various forms of test-time temperature calibration to nullify the aforementioned interference and allow the adversarial attack to find adversarial examples. Hence, we urge the NLP community to incorporate test-time temperature scaling into their robustness evaluations to ensure that any observed gains are genuine. Finally, we show how the temperature can be scaled during \textit{training} to improve genuine robustness.

摘要: 基于深度学习的自然语言处理(NLP)模型容易受到敌意攻击，其中微小的扰动可能会导致模型错误分类。对抗性训练(AT)通常被用来增强模型的稳健性。然而，我们发现了一个有趣的现象：故意或意外地错误校准模型以干扰对抗性攻击搜索方法的方式掩盖了梯度，从而产生了明显的健壮性增强。我们证明了这种观察到的健壮性增长是健壮性错觉(IOR)，并演示了对手如何执行各种形式的测试时间温度校准来抵消上述干扰，并允许对手攻击找到对手的例子。因此，我们敦促NLP社区将测试时间温度调整纳入其稳健性评估，以确保任何观察到的收益都是真实的。最后，我们展示了如何在训练期间调整温度以提高真正的健壮性。



## **43. AdvDiff: Generating Unrestricted Adversarial Examples using Diffusion Models**

AdvDiff：使用扩散模型生成不受限制的对抗性实例 cs.LG

**SubmitDate**: 2024-02-27    [abs](http://arxiv.org/abs/2307.12499v3) [paper-pdf](http://arxiv.org/pdf/2307.12499v3)

**Authors**: Xuelong Dai, Kaisheng Liang, Bin Xiao

**Abstract**: Unrestricted adversarial attacks present a serious threat to deep learning models and adversarial defense techniques. They pose severe security problems for deep learning applications because they can effectively bypass defense mechanisms. However, previous attack methods often utilize Generative Adversarial Networks (GANs), which are not theoretically provable and thus generate unrealistic examples by incorporating adversarial objectives, especially for large-scale datasets like ImageNet. In this paper, we propose a new method, called AdvDiff, to generate unrestricted adversarial examples with diffusion models. We design two novel adversarial guidance techniques to conduct adversarial sampling in the reverse generation process of diffusion models. These two techniques are effective and stable to generate high-quality, realistic adversarial examples by integrating gradients of the target classifier interpretably. Experimental results on MNIST and ImageNet datasets demonstrate that AdvDiff is effective to generate unrestricted adversarial examples, which outperforms GAN-based methods in terms of attack performance and generation quality.

摘要: 不受限制的对抗性攻击对深度学习模型和对抗性防御技术构成了严重威胁。它们会给深度学习应用程序带来严重的安全问题，因为它们可以有效地绕过防御机制。然而，以前的攻击方法通常使用生成性对抗网络(GANS)，这在理论上是不可证明的，因此通过结合对抗性目标来生成不现实的例子，特别是对于像ImageNet这样的大规模数据集。在这篇文章中，我们提出了一种新的方法，称为AdvDiff，用来生成带有扩散模型的无限制对抗实例。我们设计了两种新的对抗性制导技术，用于在扩散模型的逆向生成过程中进行对抗性采样。这两种技术通过可解释地集成目标分类器的梯度，有效而稳定地生成高质量的、真实的对抗性实例。在MNIST和ImageNet数据集上的实验结果表明，AdvDiff能够有效地生成无限制的对抗性实例，在攻击性能和生成质量方面都优于基于GAN的方法。



## **44. Adaptive Perturbation for Adversarial Attack**

对抗性攻击的自适应摄动 cs.CV

Accepted by IEEE Transactions on Pattern Analysis and Machine  Intelligence (TPAMI). 18 pages, 7 figures, 14 tables

**SubmitDate**: 2024-02-27    [abs](http://arxiv.org/abs/2111.13841v3) [paper-pdf](http://arxiv.org/pdf/2111.13841v3)

**Authors**: Zheng Yuan, Jie Zhang, Zhaoyan Jiang, Liangliang Li, Shiguang Shan

**Abstract**: In recent years, the security of deep learning models achieves more and more attentions with the rapid development of neural networks, which are vulnerable to adversarial examples. Almost all existing gradient-based attack methods use the sign function in the generation to meet the requirement of perturbation budget on $L_\infty$ norm. However, we find that the sign function may be improper for generating adversarial examples since it modifies the exact gradient direction. Instead of using the sign function, we propose to directly utilize the exact gradient direction with a scaling factor for generating adversarial perturbations, which improves the attack success rates of adversarial examples even with fewer perturbations. At the same time, we also theoretically prove that this method can achieve better black-box transferability. Moreover, considering that the best scaling factor varies across different images, we propose an adaptive scaling factor generator to seek an appropriate scaling factor for each image, which avoids the computational cost for manually searching the scaling factor. Our method can be integrated with almost all existing gradient-based attack methods to further improve their attack success rates. Extensive experiments on the CIFAR10 and ImageNet datasets show that our method exhibits higher transferability and outperforms the state-of-the-art methods.

摘要: 近年来，随着神经网络的快速发展，深度学习模型的安全性越来越受到人们的关注，因为神经网络容易受到敌意例子的攻击。现有的基于梯度的攻击方法几乎都是在生成过程中使用符号函数，以满足在$L_\inty$范数上扰动预算的要求。然而，我们发现符号函数可能不适合于生成对抗性示例，因为它修改了精确的梯度方向。我们不使用符号函数，而是直接利用带有比例因子的精确梯度方向来产生对抗性扰动，从而在扰动较少的情况下提高了对抗性实例的攻击成功率。同时，我们还从理论上证明了该方法可以达到更好的黑盒可转移性。此外，考虑到不同图像的最佳比例因子不同，我们提出了一种自适应比例因子生成器来为每幅图像寻找合适的比例因子，从而避免了手动搜索比例因子的计算代价。我们的方法可以与几乎所有现有的基于梯度的攻击方法相集成，进一步提高它们的攻击成功率。在CIFAR10和ImageNet数据集上的大量实验表明，我们的方法表现出更高的可转移性，并且性能优于最先进的方法。



## **45. Conformal Shield: A Novel Adversarial Attack Detection Framework for Automatic Modulation Classification**

保形屏蔽：一种新的自动调制分类对抗性攻击检测框架 eess.SP

**SubmitDate**: 2024-02-27    [abs](http://arxiv.org/abs/2402.17450v1) [paper-pdf](http://arxiv.org/pdf/2402.17450v1)

**Authors**: Tailai Wen, Da Ke, Xiang Wang, Zhitao Huang

**Abstract**: Deep learning algorithms have become an essential component in the field of cognitive radio, especially playing a pivotal role in automatic modulation classification. However, Deep learning also present risks and vulnerabilities. Despite their outstanding classification performance, they exhibit fragility when confronted with meticulously crafted adversarial examples, posing potential risks to the reliability of modulation recognition results. Addressing this issue, this letter pioneers the development of an intelligent modulation classification framework based on conformal theory, named the Conformal Shield, aimed at detecting the presence of adversarial examples in unknown signals and assessing the reliability of recognition results. Utilizing conformal mapping from statistical learning theory, introduces a custom-designed Inconsistency Soft-solution Set, enabling multiple validity assessments of the recognition outcomes. Experimental results demonstrate that the Conformal Shield maintains robust detection performance against a variety of typical adversarial sample attacks in the received signals under different perturbation-to-signal power ratio conditions.

摘要: 深度学习算法已成为认知无线电领域的重要组成部分，尤其在自动调制分类中发挥着举足轻重的作用。然而，深度学习也存在风险和漏洞。尽管它们具有出色的分类性能，但在面对精心制作的对抗性示例时，它们表现出脆弱性，对调制识别结果的可靠性构成潜在风险。为了解决这个问题，这封信率先开发了一个基于共形理论的智能调制分类框架，名为Conformal Shield，旨在检测未知信号中是否存在对抗性示例，并评估识别结果的可靠性。利用统计学习理论中的保角映射，引入了一个定制设计的不一致性软解集，使识别结果的多个有效性评估成为可能。实验结果表明，在不同的扰动信号功率比条件下，共形屏蔽算法对接收信号中各种典型的对抗性样本攻击都保持了鲁棒的检测性能。



## **46. Comparing the Robustness of Modern No-Reference Image- and Video-Quality Metrics to Adversarial Attacks**

比较现代无参考图像和视频质量指标对对手攻击的稳健性 cs.CV

**SubmitDate**: 2024-02-27    [abs](http://arxiv.org/abs/2310.06958v4) [paper-pdf](http://arxiv.org/pdf/2310.06958v4)

**Authors**: Anastasia Antsiferova, Khaled Abud, Aleksandr Gushchin, Ekaterina Shumitskaya, Sergey Lavrushkin, Dmitriy Vatolin

**Abstract**: Nowadays, neural-network-based image- and video-quality metrics perform better than traditional methods. However, they also became more vulnerable to adversarial attacks that increase metrics' scores without improving visual quality. The existing benchmarks of quality metrics compare their performance in terms of correlation with subjective quality and calculation time. Nonetheless, the adversarial robustness of image-quality metrics is also an area worth researching. This paper analyses modern metrics' robustness to different adversarial attacks. We adapted adversarial attacks from computer vision tasks and compared attacks' efficiency against 15 no-reference image- and video-quality metrics. Some metrics showed high resistance to adversarial attacks, which makes their usage in benchmarks safer than vulnerable metrics. The benchmark accepts submissions of new metrics for researchers who want to make their metrics more robust to attacks or to find such metrics for their needs. The latest results can be found online: https://videoprocessing.ai/benchmarks/metrics-robustness.html.

摘要: 如今，基于神经网络的图像和视频质量指标比传统方法表现得更好。然而，它们也变得更容易受到对抗性攻击，这些攻击增加了指标的分数，但没有改善视觉质量。现有的质量指标基准在与主观质量和计算时间的相关性方面比较它们的表现。尽管如此，图像质量指标的对抗性稳健性也是一个值得研究的领域。分析了现代度量对不同对手攻击的稳健性。我们从计算机视觉任务中改编了对抗性攻击，并将攻击效率与15个无参考图像和视频质量指标进行了比较。一些指标表现出对对手攻击的高度抵抗力，这使得它们在基准中的使用比易受攻击的指标更安全。该基准接受新指标的提交，供希望使其指标更具抗攻击能力或找到符合其需求的此类指标的研究人员使用。最新结果可在网上找到：https://videoprocessing.ai/benchmarks/metrics-robustness.html.



## **47. HarmBench: A Standardized Evaluation Framework for Automated Red Teaming and Robust Refusal**

HarmBench：一个用于自动化红队和鲁棒拒绝的标准化评估框架 cs.LG

Website: https://www.harmbench.org

**SubmitDate**: 2024-02-27    [abs](http://arxiv.org/abs/2402.04249v2) [paper-pdf](http://arxiv.org/pdf/2402.04249v2)

**Authors**: Mantas Mazeika, Long Phan, Xuwang Yin, Andy Zou, Zifan Wang, Norman Mu, Elham Sakhaee, Nathaniel Li, Steven Basart, Bo Li, David Forsyth, Dan Hendrycks

**Abstract**: Automated red teaming holds substantial promise for uncovering and mitigating the risks associated with the malicious use of large language models (LLMs), yet the field lacks a standardized evaluation framework to rigorously assess new methods. To address this issue, we introduce HarmBench, a standardized evaluation framework for automated red teaming. We identify several desirable properties previously unaccounted for in red teaming evaluations and systematically design HarmBench to meet these criteria. Using HarmBench, we conduct a large-scale comparison of 18 red teaming methods and 33 target LLMs and defenses, yielding novel insights. We also introduce a highly efficient adversarial training method that greatly enhances LLM robustness across a wide range of attacks, demonstrating how HarmBench enables codevelopment of attacks and defenses. We open source HarmBench at https://github.com/centerforaisafety/HarmBench.

摘要: 自动化的红色团队在发现和减轻与恶意使用大型语言模型（LLM）相关的风险方面有很大的希望，但该领域缺乏标准化的评估框架来严格评估新方法。为了解决这个问题，我们引入了HarmBench，这是一个用于自动化红色团队的标准化评估框架。我们确定了几个理想的属性以前未考虑到在红色的团队评估和系统的设计HarmBench，以满足这些标准。使用HarmBench，我们对18种红色组队方法和33种目标LLM和防御进行了大规模比较，产生了新的见解。我们还介绍了一种高效的对抗性训练方法，该方法大大增强了LLM在各种攻击中的鲁棒性，展示了HarmBench如何实现攻击和防御的共同开发。我们在https://github.com/centerforaisafety/HarmBench上开源HarmBench。



## **48. Break the Breakout: Reinventing LM Defense Against Jailbreak Attacks with Self-Refinement**

打破越狱：用自我优化重塑LM防御越狱攻击 cs.LG

under review

**SubmitDate**: 2024-02-27    [abs](http://arxiv.org/abs/2402.15180v2) [paper-pdf](http://arxiv.org/pdf/2402.15180v2)

**Authors**: Heegyu Kim, Sehyun Yuk, Hyunsouk Cho

**Abstract**: Caution: This paper includes offensive words that could potentially cause unpleasantness. Language models (LMs) are vulnerable to exploitation for adversarial misuse. Training LMs for safety alignment is extensive and makes it hard to respond to fast-developing attacks immediately, such as jailbreaks. We propose self-refine with formatting that achieves outstanding safety even in non-safety-aligned LMs and evaluate our method alongside several defense baselines, demonstrating that it is the safest training-free method against jailbreak attacks. Additionally, we proposed a formatting method that improves the efficiency of the self-refine process while reducing attack success rates in fewer iterations. We've also observed that non-safety-aligned LMs outperform safety-aligned LMs in safety tasks by giving more helpful and safe responses. In conclusion, our findings can achieve less safety risk with fewer computational costs, allowing non-safety LM to be easily utilized in real-world service.

摘要: 注意：本文包含了可能会引起不快的冒犯性词语。语言模型(LMS)容易受到恶意滥用的攻击。对LMS进行广泛的安全调整培训，使其很难立即对快速发展的攻击做出反应，例如越狱。我们提出了自我改进的格式，即使在非安全对齐的LMS中也能实现出色的安全性，并与几个防御基线一起评估我们的方法，证明它是针对越狱攻击的最安全的免培训方法。此外，我们还提出了一种格式化方法，在减少迭代次数的同时提高了自我精炼过程的效率，同时降低了攻击成功率。我们还观察到，非安全对齐的LMS在安全任务中的表现优于安全对齐的LMS，因为它给出了更有用和安全的反应。总之，我们的发现可以用更少的计算代价实现更小的安全风险，使得非安全的LM很容易被用于现实世界的服务。



## **49. Adversarial example soups: averaging multiple adversarial examples improves transferability without increasing additional generation time**

对抗性示例汤：平均多个对抗性示例可提高可转移性，而不会增加额外的生成时间 cs.CV

16 pages, 8 figures, 12 tables

**SubmitDate**: 2024-02-27    [abs](http://arxiv.org/abs/2402.18370v1) [paper-pdf](http://arxiv.org/pdf/2402.18370v1)

**Authors**: Bo Yang, Hengwei Zhang, Chenwei Li, Jindong Wang

**Abstract**: For transfer-based attacks, the adversarial examples are crafted on the surrogate model, which can be implemented to mislead the target model effectively. The conventional method for maximizing adversarial transferability involves: (1) fine-tuning hyperparameters to generate multiple batches of adversarial examples on the substitute model; (2) conserving the batch of adversarial examples that have the best comprehensive performance on substitute model and target model, and discarding the others. In this work, we revisit the second step of this process in the context of fine-tuning hyperparameters to craft adversarial examples, where multiple batches of fine-tuned adversarial examples often appear in a single high error hilltop. We demonstrate that averaging multiple batches of adversarial examples under different hyperparameter configurations, which refers to as "adversarial example soups", can often enhance adversarial transferability. Compared with traditional methods, the proposed method incurs no additional generation time and computational cost. Besides, our method is orthogonal to existing transfer-based methods and can be combined with them seamlessly to generate more transferable adversarial examples. Extensive experiments on the ImageNet dataset show that our methods achieve a higher attack success rate than the state-of-the-art attacks.

摘要: 对于基于传输的攻击，在代理模型上构造对抗性实例，可以实现对目标模型的有效误导。传统的对抗性转移最大化方法包括：(1)微调超参数，在替代模型上生成多批对抗性实例；(2)保留一批在替代模型和目标模型上综合性能最好的对抗性实例，丢弃其他的。在这项工作中，我们在微调超参数的背景下回顾了这一过程的第二步，以制作对抗性示例，其中多批微调的对抗性示例经常出现在单个高错误的山顶上。我们证明了在不同的超参数配置下对多批次对抗性样本进行平均，即所谓的对抗性样本汤，通常可以增强对抗性转移能力。与传统方法相比，该方法不需要额外的生成时间和计算代价。此外，我们的方法与现有的基于转移的方法是正交的，并且可以与它们无缝结合来生成更多可转移的对抗性例子。在ImageNet数据集上的大量实验表明，我们的方法取得了比最先进的攻击更高的攻击成功率。



## **50. Dempster-Shafer P-values: Thoughts on an Alternative Approach for Multinomial Inference**

Dempster-Shafer P值：关于多项式推论的另一种方法的思考 stat.ME

**SubmitDate**: 2024-02-26    [abs](http://arxiv.org/abs/2402.17070v1) [paper-pdf](http://arxiv.org/pdf/2402.17070v1)

**Authors**: Kentaro Hoffman, Kai Zhang, Tyler McCormick, Jan Hannig

**Abstract**: In this paper, we demonstrate that a new measure of evidence we developed called the Dempster-Shafer p-value which allow for insights and interpretations which retain most of the structure of the p-value while covering for some of the disadvantages that traditional p- values face. Moreover, we show through classical large-sample bounds and simulations that there exists a close connection between our form of DS hypothesis testing and the classical frequentist testing paradigm. We also demonstrate how our approach gives unique insights into the dimensionality of a hypothesis test, as well as models the effects of adversarial attacks on multinomial data. Finally, we demonstrate how these insights can be used to analyze text data for public health through an analysis of the Population Health Metrics Research Consortium dataset for verbal autopsies.

摘要: 在这篇文章中，我们证明了我们开发的一种新的证据度量，称为Dempster-Shafer p值，它允许洞察和解释保留p值的大部分结构，同时涵盖传统p值面临的一些缺点。此外，我们通过经典的大样本界和模拟表明，我们的DS假设检验的形式与经典的频域检验范式之间存在着密切的联系。我们还展示了我们的方法如何给出对假设检验的维度的独特见解，以及如何对多项式数据上的对抗性攻击的影响进行建模。最后，我们通过对人口健康指标研究联盟口头尸检数据集的分析，展示了如何使用这些洞察力来分析公共卫生文本数据。



