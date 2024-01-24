# Latest Adversarial Attack Papers
**update at 2024-01-24 09:18:44**

翻译来自 https://cloud.tencent.com/document/product/551/15619

## **1. Benchmarking the Robustness of Image Watermarks**

图像水印稳健性的基准测试 cs.CV

**SubmitDate**: 2024-01-22    [abs](http://arxiv.org/abs/2401.08573v2) [paper-pdf](http://arxiv.org/pdf/2401.08573v2)

**Authors**: Bang An, Mucong Ding, Tahseen Rabbani, Aakriti Agrawal, Yuancheng Xu, Chenghao Deng, Sicheng Zhu, Abdirisak Mohamed, Yuxin Wen, Tom Goldstein, Furong Huang

**Abstract**: This paper investigates the weaknesses of image watermarking techniques. We present WAVES (Watermark Analysis Via Enhanced Stress-testing), a novel benchmark for assessing watermark robustness, overcoming the limitations of current evaluation methods.WAVES integrates detection and identification tasks, and establishes a standardized evaluation protocol comprised of a diverse range of stress tests. The attacks in WAVES range from traditional image distortions to advanced and novel variations of diffusive, and adversarial attacks. Our evaluation examines two pivotal dimensions: the degree of image quality degradation and the efficacy of watermark detection after attacks. We develop a series of Performance vs. Quality 2D plots, varying over several prominent image similarity metrics, which are then aggregated in a heuristically novel manner to paint an overall picture of watermark robustness and attack potency. Our comprehensive evaluation reveals previously undetected vulnerabilities of several modern watermarking algorithms. We envision WAVES as a toolkit for the future development of robust watermarking systems. The project is available at https://wavesbench.github.io/

摘要: 本文研究了图像水印技术的弱点。为了克服现有评估方法的局限性，提出了一种新的评估水印稳健性的基准--WAVES(基于增强压力测试的水印分析)，它集成了检测和识别任务，并建立了由多种压力测试组成的标准化评估协议。一波一波的攻击范围从传统的图像扭曲到高级和新奇的扩散性和对抗性攻击。我们的评估检查了两个关键维度：图像质量下降的程度和攻击后水印检测的有效性。我们开发了一系列性能与质量的2D图，不同于几个重要的图像相似性度量，然后以启发式的新颖方式聚集在一起，描绘了水印稳健性和攻击能力的总体图景。我们的综合评估揭示了几种现代水印算法以前未被检测到的漏洞。我们设想Waves将成为未来健壮水印系统发展的工具包。该项目的网址为：https://wavesbench.github.io/。



## **2. NEUROSEC: FPGA-Based Neuromorphic Audio Security**

NeurOSEC：基于FPGA的神经形态音频安全 cs.CR

Audio processing, FPGA, Hardware Security, Neuromorphic Computing

**SubmitDate**: 2024-01-22    [abs](http://arxiv.org/abs/2401.12055v1) [paper-pdf](http://arxiv.org/pdf/2401.12055v1)

**Authors**: Murat Isik, Hiruna Vishwamith, Yusuf Sur, Kayode Inadagbo, I. Can Dikmen

**Abstract**: Neuromorphic systems, inspired by the complexity and functionality of the human brain, have gained interest in academic and industrial attention due to their unparalleled potential across a wide range of applications. While their capabilities herald innovation, it is imperative to underscore that these computational paradigms, analogous to their traditional counterparts, are not impervious to security threats. Although the exploration of neuromorphic methodologies for image and video processing has been rigorously pursued, the realm of neuromorphic audio processing remains in its early stages. Our results highlight the robustness and precision of our FPGA-based neuromorphic system. Specifically, our system showcases a commendable balance between desired signal and background noise, efficient spike rate encoding, and unparalleled resilience against adversarial attacks such as FGSM and PGD. A standout feature of our framework is its detection rate of 94%, which, when compared to other methodologies, underscores its greater capability in identifying and mitigating threats within 5.39 dB, a commendable SNR ratio. Furthermore, neuromorphic computing and hardware security serve many sensor domains in mission-critical and privacy-preserving applications.

摘要: 受人脑的复杂性和功能性启发，神经形态系统因其在广泛应用中的无与伦比的潜力而引起了学术界和工业界的兴趣。虽然它们的能力预示着创新，但必须强调的是，这些与传统计算模式类似的计算模式并非不受安全威胁的影响。尽管人们一直在探索用于图像和视频处理的神经形态方法，但神经形态音频处理领域仍处于早期阶段。我们的结果突出了我们的基于FPGA的神经形态系统的健壮性和精确度。具体地说，我们的系统在期望的信号和背景噪声之间表现出了令人称赞的平衡，高效的尖峰速率编码，以及对FGSM和PGD等对手攻击的无与伦比的弹性。我们框架的一个突出特点是其94%的检测率，与其他方法相比，这突显了它在识别和缓解5.39分贝以内的威胁方面具有更强的能力，这是一个值得称赞的信噪比。此外，神经形态计算和硬件安全为许多关键任务和隐私保护应用中的传感器领域提供服务。



## **3. The Effect of Intrinsic Dataset Properties on Generalization: Unraveling Learning Differences Between Natural and Medical Images**

数据集属性对概化的影响：解开自然图像和医学图像之间的学习差异 cs.CV

ICLR 2024. Code:  https://github.com/mazurowski-lab/intrinsic-properties

**SubmitDate**: 2024-01-22    [abs](http://arxiv.org/abs/2401.08865v2) [paper-pdf](http://arxiv.org/pdf/2401.08865v2)

**Authors**: Nicholas Konz, Maciej A. Mazurowski

**Abstract**: This paper investigates discrepancies in how neural networks learn from different imaging domains, which are commonly overlooked when adopting computer vision techniques from the domain of natural images to other specialized domains such as medical images. Recent works have found that the generalization error of a trained network typically increases with the intrinsic dimension ($d_{data}$) of its training set. Yet, the steepness of this relationship varies significantly between medical (radiological) and natural imaging domains, with no existing theoretical explanation. We address this gap in knowledge by establishing and empirically validating a generalization scaling law with respect to $d_{data}$, and propose that the substantial scaling discrepancy between the two considered domains may be at least partially attributed to the higher intrinsic "label sharpness" ($K_F$) of medical imaging datasets, a metric which we propose. Next, we demonstrate an additional benefit of measuring the label sharpness of a training set: it is negatively correlated with the trained model's adversarial robustness, which notably leads to models for medical images having a substantially higher vulnerability to adversarial attack. Finally, we extend our $d_{data}$ formalism to the related metric of learned representation intrinsic dimension ($d_{repr}$), derive a generalization scaling law with respect to $d_{repr}$, and show that $d_{data}$ serves as an upper bound for $d_{repr}$. Our theoretical results are supported by thorough experiments with six models and eleven natural and medical imaging datasets over a range of training set sizes. Our findings offer insights into the influence of intrinsic dataset properties on generalization, representation learning, and robustness in deep neural networks.

摘要: 本文研究了神经网络如何从不同的成像领域学习的差异，这些差异是在将计算机视觉技术从自然图像领域应用到其他专业领域(如医学图像)时通常被忽视的。最近的工作发现，训练网络的泛化误差通常随着训练集的固有维度($d_{data}$)的增加而增加。然而，这种关系的陡峭程度在医学(放射)和自然成像领域有很大的不同，没有现有的理论解释。我们通过建立和经验性地验证关于$d{data}$的泛化标度律来解决这一知识缺口，并提出两个被考虑的域之间的显著标度差异至少部分归因于医学成像数据集的更高的固有“标签锐度”($K_F$)，这是我们提出的一种度量。接下来，我们展示了测量训练集的标签锐度的另一个好处：它与训练模型的对抗稳健性负相关，这显著地导致医学图像的模型具有更高的对抗攻击脆弱性。最后，我们将$d_{data}$形式推广到学习表示内在维的相关度量($d_{epr}$)，得到了关于$d_{epr}$的一个推广的标度律，并证明了$d_{data}$是$d_{epr}$的一个上界。我们的理论结果得到了6个模型和11个自然和医学成像数据集的全面实验的支持，这些数据集的训练集大小不同。我们的发现对深入神经网络中内在数据集属性对泛化、表示学习和稳健性的影响提供了深入的见解。



## **4. Diagnosis-guided Attack Recovery for Securing Robotic Vehicles from Sensor Deception Attacks**

用于保护机器人车辆免受传感器欺骗攻击的诊断引导攻击恢复 cs.RO

**SubmitDate**: 2024-01-22    [abs](http://arxiv.org/abs/2209.04554v5) [paper-pdf](http://arxiv.org/pdf/2209.04554v5)

**Authors**: Pritam Dash, Guanpeng Li, Mehdi Karimibiuki, Karthik Pattabiraman

**Abstract**: Sensors are crucial for perception and autonomous operation in robotic vehicles (RV). Unfortunately, RV sensors can be compromised by physical attacks such as sensor tampering or spoofing. In this paper, we present DeLorean, a unified framework for attack detection, attack diagnosis, and recovering RVs from sensor deception attacks (SDA). DeLorean can recover RVs even from strong SDAs in which the adversary targets multiple heterogeneous sensors simultaneously. We propose a novel attack diagnosis technique that inspects the attack-induced errors under SDAs, and identifies the targeted sensors using causal analysis. DeLorean then uses historic state information to selectively reconstruct physical states for compromised sensors, enabling targeted attack recovery under single or multi-sensor SDAs. We evaluate DeLorean on four real and two simulated RVs under SDAs targeting various sensors, and we find that it successfully recovers RVs from SDAs in 93% of the cases.

摘要: 传感器对于机器人车辆(RV)的感知和自主操作至关重要。不幸的是，RV传感器可能会受到物理攻击，如篡改传感器或欺骗。在本文中，我们提出了DeLorean，一个统一的攻击检测、攻击诊断和从传感器欺骗攻击(SDA)中恢复房车的框架。即使对手同时瞄准多个不同的传感器，DeLorean也可以从强大的SDA中恢复RV。我们提出了一种新的攻击诊断技术，该技术在SDAS下检测攻击导致的错误，并使用因果分析来识别目标传感器。然后，DeLorean使用历史状态信息有选择地重建受损传感器的物理状态，从而在单传感器或多传感器SDA下实现有针对性的攻击恢复。我们在四辆真实房车和两辆模拟房车上对DeLorean进行了评估，结果表明，在93%的情况下，DeLorean能够成功地从SDA中恢复房车。



## **5. A Training-Free Defense Framework for Robust Learned Image Compression**

一种无训练的稳健学习图像压缩防御框架 eess.IV

10 pages and 14 figures

**SubmitDate**: 2024-01-22    [abs](http://arxiv.org/abs/2401.11902v1) [paper-pdf](http://arxiv.org/pdf/2401.11902v1)

**Authors**: Myungseo Song, Jinyoung Choi, Bohyung Han

**Abstract**: We study the robustness of learned image compression models against adversarial attacks and present a training-free defense technique based on simple image transform functions. Recent learned image compression models are vulnerable to adversarial attacks that result in poor compression rate, low reconstruction quality, or weird artifacts. To address the limitations, we propose a simple but effective two-way compression algorithm with random input transforms, which is conveniently applicable to existing image compression models. Unlike the na\"ive approaches, our approach preserves the original rate-distortion performance of the models on clean images. Moreover, the proposed algorithm requires no additional training or modification of existing models, making it more practical. We demonstrate the effectiveness of the proposed techniques through extensive experiments under multiple compression models, evaluation metrics, and attack scenarios.

摘要: 我们研究了学习图像压缩模型对敌意攻击的稳健性，提出了一种基于简单图像变换函数的免训练防御技术。最近学习的图像压缩模型容易受到敌意攻击，导致压缩比低、重建质量低或出现奇怪的伪影。针对这些局限性，我们提出了一种简单而有效的随机输入变换双向压缩算法，该算法可以方便地适用于现有的图像压缩模型。与朴素方法不同，该方法保留了原始模型在干净图像上的率失真性能，并且不需要对已有模型进行额外的训练或修改，使其更具实用性。通过大量的实验，我们证明了该方法在多种压缩模型、评价指标和攻击场景下的有效性。



## **6. Adversarial speech for voice privacy protection from Personalized Speech generation**

针对个性化语音生成的语音隐私保护的对抗性语音 eess.AS

Accepted by icassp 2024

**SubmitDate**: 2024-01-22    [abs](http://arxiv.org/abs/2401.11857v1) [paper-pdf](http://arxiv.org/pdf/2401.11857v1)

**Authors**: Shihao Chen, Liping Chen, Jie Zhang, KongAik Lee, Zhenhua Ling, Lirong Dai

**Abstract**: The rapid progress in personalized speech generation technology, including personalized text-to-speech (TTS) and voice conversion (VC), poses a challenge in distinguishing between generated and real speech for human listeners, resulting in an urgent demand in protecting speakers' voices from malicious misuse. In this regard, we propose a speaker protection method based on adversarial attacks. The proposed method perturbs speech signals by minimally altering the original speech while rendering downstream speech generation models unable to accurately generate the voice of the target speaker. For validation, we employ the open-source pre-trained YourTTS model for speech generation and protect the target speaker's speech in the white-box scenario. Automatic speaker verification (ASV) evaluations were carried out on the generated speech as the assessment of the voice protection capability. Our experimental results show that we successfully perturbed the speaker encoder of the YourTTS model using the gradient-based I-FGSM adversarial perturbation method. Furthermore, the adversarial perturbation is effective in preventing the YourTTS model from generating the speech of the target speaker. Audio samples can be found in https://voiceprivacy.github.io/Adeversarial-Speech-with-YourTTS.

摘要: 个性化语音生成技术的快速发展，包括个性化文语转换(TTS)和语音转换(VC)，对人类听者区分生成的语音和真实的语音提出了挑战，导致了对保护说话人的语音免受恶意滥用的迫切需求。对此，我们提出了一种基于对抗性攻击的说话人保护方法。所提出的方法通过最小限度地改变原始语音来扰动语音信号，同时使得下游语音生成模型无法准确地生成目标说话人的声音。为了验证，我们使用开源的预先训练的YourTTS模型来生成语音，并在白盒场景中保护目标说话人的语音。对生成的语音进行自动说话人验证(ASV)评估，以评估语音保护能力。实验结果表明，我们使用基于梯度的I-FGSM对抗扰动方法成功地扰动了YourTTS模型的说话人编码器。此外，对抗性扰动可以有效地防止YourTTS模型生成目标说话人的语音。音频样本可在https://voiceprivacy.github.io/Adeversarial-Speech-with-YourTTS.中找到



## **7. Unraveling Attacks in Machine Learning-based IoT Ecosystems: A Survey and the Open Libraries Behind Them**

破解基于机器学习的物联网生态系统中的攻击：综述及其背后的开放图书馆 cs.CR

**SubmitDate**: 2024-01-22    [abs](http://arxiv.org/abs/2401.11723v1) [paper-pdf](http://arxiv.org/pdf/2401.11723v1)

**Authors**: Chao Liu, Boxi Chen, Wei Shao, Chris Zhang, Kelvin Wong, Yi Zhang

**Abstract**: The advent of the Internet of Things (IoT) has brought forth an era of unprecedented connectivity, with an estimated 80 billion smart devices expected to be in operation by the end of 2025. These devices facilitate a multitude of smart applications, enhancing the quality of life and efficiency across various domains. Machine Learning (ML) serves as a crucial technology, not only for analyzing IoT-generated data but also for diverse applications within the IoT ecosystem. For instance, ML finds utility in IoT device recognition, anomaly detection, and even in uncovering malicious activities. This paper embarks on a comprehensive exploration of the security threats arising from ML's integration into various facets of IoT, spanning various attack types including membership inference, adversarial evasion, reconstruction, property inference, model extraction, and poisoning attacks. Unlike previous studies, our work offers a holistic perspective, categorizing threats based on criteria such as adversary models, attack targets, and key security attributes (confidentiality, availability, and integrity). We delve into the underlying techniques of ML attacks in IoT environment, providing a critical evaluation of their mechanisms and impacts. Furthermore, our research thoroughly assesses 65 libraries, both author-contributed and third-party, evaluating their role in safeguarding model and data privacy. We emphasize the availability and usability of these libraries, aiming to arm the community with the necessary tools to bolster their defenses against the evolving threat landscape. Through our comprehensive review and analysis, this paper seeks to contribute to the ongoing discourse on ML-based IoT security, offering valuable insights and practical solutions to secure ML models and data in the rapidly expanding field of artificial intelligence in IoT.

摘要: 物联网(IoT)的到来带来了一个前所未有的互联时代，预计到2025年底，将有800亿台智能设备投入运营。这些设备促进了大量智能应用，提高了各个领域的生活质量和效率。机器学习(ML)是一项关键技术，不仅用于分析物联网生成的数据，还用于分析物联网生态系统中的各种应用。例如，ML在物联网设备识别、异常检测，甚至在发现恶意活动方面都有用武之地。本文对ML融入物联网的各个方面所带来的安全威胁进行了全面的探讨，包括成员身份推断、对抗性逃避、重构、属性推理、模型提取和中毒攻击等各种攻击类型。与以前的研究不同，我们的工作提供了一个整体的视角，根据对手模型、攻击目标和关键安全属性(机密性、可用性和完整性)等标准对威胁进行分类。我们深入研究了物联网环境下ML攻击的基本技术，并对其机制和影响进行了关键评估。此外，我们的研究全面评估了65个图书馆，包括作者贡献的图书馆和第三方图书馆，评估它们在保护模型和数据隐私方面的作用。我们强调这些库的可用性和可用性，旨在为社区提供必要的工具，以加强他们对不断变化的威胁环境的防御。通过我们的全面回顾和分析，本文试图为正在进行的基于ML的物联网安全讨论做出贡献，为保护物联网快速扩张的人工智能领域中的ML模型和数据提供有价值的见解和实用解决方案。



## **8. HashVFL: Defending Against Data Reconstruction Attacks in Vertical Federated Learning**

HashVFL：垂直联合学习中数据重构攻击的防御 cs.CR

**SubmitDate**: 2024-01-22    [abs](http://arxiv.org/abs/2212.00325v2) [paper-pdf](http://arxiv.org/pdf/2212.00325v2)

**Authors**: Pengyu Qiu, Xuhong Zhang, Shouling Ji, Chong Fu, Xing Yang, Ting Wang

**Abstract**: Vertical Federated Learning (VFL) is a trending collaborative machine learning model training solution. Existing industrial frameworks employ secure multi-party computation techniques such as homomorphic encryption to ensure data security and privacy. Despite these efforts, studies have revealed that data leakage remains a risk in VFL due to the correlations between intermediate representations and raw data. Neural networks can accurately capture these correlations, allowing an adversary to reconstruct the data. This emphasizes the need for continued research into securing VFL systems.   Our work shows that hashing is a promising solution to counter data reconstruction attacks. The one-way nature of hashing makes it difficult for an adversary to recover data from hash codes. However, implementing hashing in VFL presents new challenges, including vanishing gradients and information loss. To address these issues, we propose HashVFL, which integrates hashing and simultaneously achieves learnability, bit balance, and consistency.   Experimental results indicate that HashVFL effectively maintains task performance while defending against data reconstruction attacks. It also brings additional benefits in reducing the degree of label leakage, mitigating adversarial attacks, and detecting abnormal inputs. We hope our work will inspire further research into the potential applications of HashVFL.

摘要: 垂直联合学习(VFL)是一种流行的协作式机器学习模型训练方案。现有的工业框架采用安全的多方计算技术，如同态加密，以确保数据的安全和隐私。尽管做出了这些努力，但研究表明，由于中间表征和原始数据之间的相关性，数据泄露在VFL中仍然是一个风险。神经网络可以准确地捕捉这些关联，允许对手重建数据。这强调了对保护VFL系统进行持续研究的必要性。我们的工作表明，哈希是对抗数据重构攻击的一种很有前途的解决方案。哈希的单向性质使得对手很难从哈希码中恢复数据。然而，在VFL中实现散列带来了新的挑战，包括梯度消失和信息丢失。为了解决这些问题，我们提出了HashVFL，它集成了散列，同时实现了可学习性、位平衡和一致性。实验结果表明，HashVFL在抵抗数据重构攻击的同时，有效地保持了任务的性能。它还在降低标签泄漏程度、减轻敌意攻击和检测异常输入方面带来了额外的好处。我们希望我们的工作将启发对HashVFL潜在应用的进一步研究。



## **9. LRS: Enhancing Adversarial Transferability through Lipschitz Regularized Surrogate**

LRS：通过Lipschitz正则化代理提高对手的可转移性 cs.LG

AAAI 2024 main track. Code available on Github (see abstract).  Appendix is included in this updated version

**SubmitDate**: 2024-01-22    [abs](http://arxiv.org/abs/2312.13118v2) [paper-pdf](http://arxiv.org/pdf/2312.13118v2)

**Authors**: Tao Wu, Tie Luo, Donald C. Wunsch

**Abstract**: The transferability of adversarial examples is of central importance to transfer-based black-box adversarial attacks. Previous works for generating transferable adversarial examples focus on attacking \emph{given} pretrained surrogate models while the connections between surrogate models and adversarial trasferability have been overlooked. In this paper, we propose {\em Lipschitz Regularized Surrogate} (LRS) for transfer-based black-box attacks, a novel approach that transforms surrogate models towards favorable adversarial transferability. Using such transformed surrogate models, any existing transfer-based black-box attack can run without any change, yet achieving much better performance. Specifically, we impose Lipschitz regularization on the loss landscape of surrogate models to enable a smoother and more controlled optimization process for generating more transferable adversarial examples. In addition, this paper also sheds light on the connection between the inner properties of surrogate models and adversarial transferability, where three factors are identified: smaller local Lipschitz constant, smoother loss landscape, and stronger adversarial robustness. We evaluate our proposed LRS approach by attacking state-of-the-art standard deep neural networks and defense models. The results demonstrate significant improvement on the attack success rates and transferability. Our code is available at https://github.com/TrustAIoT/LRS.

摘要: 对抗性例子的可转移性对于基于转移的黑盒对抗性攻击是至关重要的。以往关于生成可传递对抗实例的工作主要集中在攻击预先训练好的代理模型上，而忽略了代理模型与对抗传递能力之间的联系。针对基于转移的黑盒攻击，提出了一种将代理模型转化为有利的对抗性转移的新方法--LRS。使用这种转换的代理模型，任何现有的基于传输的黑盒攻击都可以在不做任何更改的情况下运行，但获得了更好的性能。具体地说，我们将Lipschitz正则化应用于代理模型的损失图景，以实现更平滑和更可控的优化过程，从而生成更多可转移的对抗性例子。此外，本文还揭示了代理模型的内在性质与对抗转移之间的关系，其中确定了三个因素：较小的局部Lipschitz常数、更平滑的损失图景和更强的对抗稳健性。我们通过攻击最先进的标准深度神经网络和防御模型来评估我们提出的LRS方法。结果表明，在攻击成功率和可转移性方面都有显著的提高。我们的代码可以在https://github.com/TrustAIoT/LRS.上找到



## **10. Reducing Usefulness of Stolen Credentials in SSO Contexts**

降低被盗凭据在SSO上下文中的有用性 cs.CR

8 pages, 5 figures

**SubmitDate**: 2024-01-21    [abs](http://arxiv.org/abs/2401.11599v1) [paper-pdf](http://arxiv.org/pdf/2401.11599v1)

**Authors**: Sam Hays, Michael Sandborn, Dr. Jules White

**Abstract**: Approximately 61% of cyber attacks involve adversaries in possession of valid credentials. Attackers acquire credentials through various means, including phishing, dark web data drops, password reuse, etc. Multi-factor authentication (MFA) helps to thwart attacks that use valid credentials, but attackers still commonly breach systems by tricking users into accepting MFA step up requests through techniques, such as ``MFA Bombing'', where multiple requests are sent to a user until they accept one. Currently, there are several solutions to this problem, each with varying levels of security and increasing invasiveness on user devices. This paper proposes a token-based enrollment architecture that is less invasive to user devices than mobile device management, but still offers strong protection against use of stolen credentials and MFA attacks.

摘要: 大约61%的网络攻击涉及拥有有效凭据的对手。攻击者通过各种方式获取凭据，包括网络钓鱼、暗网络数据丢弃、密码重复使用等。多因素身份验证(MFA)有助于挫败使用有效凭据的攻击，但攻击者通常仍通过诱使用户接受MFA加速请求等技术来破坏系统，其中多个请求被发送给用户，直到他们接受一个请求。目前，有几种解决方案可以解决这个问题，每种解决方案的安全级别各不相同，对用户设备的侵入性也在不断增加。提出了一种基于令牌的注册架构，与移动设备管理相比，该架构对用户设备的侵入性较小，但仍能提供强大的保护，防止使用被盗凭据和MFA攻击。



## **11. Thundernna: a white box adversarial attack**

Thundernna：白盒对抗性攻击 cs.LG

10 pages, 5 figures

**SubmitDate**: 2024-01-21    [abs](http://arxiv.org/abs/2111.12305v2) [paper-pdf](http://arxiv.org/pdf/2111.12305v2)

**Authors**: Linfeng Ye, Shayan Mohajer Hamidi

**Abstract**: The existing work shows that the neural network trained by naive gradient-based optimization method is prone to adversarial attacks, adds small malicious on the ordinary input is enough to make the neural network wrong. At the same time, the attack against a neural network is the key to improving its robustness. The training against adversarial examples can make neural networks resist some kinds of adversarial attacks. At the same time, the adversarial attack against a neural network can also reveal some characteristics of the neural network, a complex high-dimensional non-linear function, as discussed in previous work.   In This project, we develop a first-order method to attack the neural network. Compare with other first-order attacks, our method has a much higher success rate. Furthermore, it is much faster than second-order attacks and multi-steps first-order attacks.

摘要: 已有的工作表明，基于朴素梯度优化方法训练的神经网络容易受到敌意攻击，在普通输入上添加少量恶意信息就足以使神经网络出错。同时，对神经网络的攻击是提高其稳健性的关键。针对对抗性例子的训练可以使神经网络抵抗某些类型的对抗性攻击。同时，对神经网络的敌意攻击也可以揭示神经网络的一些特征，这是一个复杂的高维非线性函数，如前人所讨论的。在这个项目中，我们开发了一种一阶方法来攻击神经网络。与其他一阶攻击方法相比，该方法具有更高的成功率。此外，它比二阶攻击和多步骤一阶攻击要快得多。



## **12. How Robust Are Energy-Based Models Trained With Equilibrium Propagation?**

基于能量的模型用均衡传播训练的健壮性如何？ cs.LG

**SubmitDate**: 2024-01-21    [abs](http://arxiv.org/abs/2401.11543v1) [paper-pdf](http://arxiv.org/pdf/2401.11543v1)

**Authors**: Siddharth Mansingh, Michal Kucer, Garrett Kenyon, Juston Moore, Michael Teti

**Abstract**: Deep neural networks (DNNs) are easily fooled by adversarial perturbations that are imperceptible to humans. Adversarial training, a process where adversarial examples are added to the training set, is the current state-of-the-art defense against adversarial attacks, but it lowers the model's accuracy on clean inputs, is computationally expensive, and offers less robustness to natural noise. In contrast, energy-based models (EBMs), which were designed for efficient implementation in neuromorphic hardware and physical systems, incorporate feedback connections from each layer to the previous layer, yielding a recurrent, deep-attractor architecture which we hypothesize should make them naturally robust. Our work is the first to explore the robustness of EBMs to both natural corruptions and adversarial attacks, which we do using the CIFAR-10 and CIFAR-100 datasets. We demonstrate that EBMs are more robust than transformers and display comparable robustness to adversarially-trained DNNs on gradient-based (white-box) attacks, query-based (black-box) attacks, and natural perturbations without sacrificing clean accuracy, and without the need for adversarial training or additional training techniques.

摘要: 深度神经网络(DNN)很容易被人类察觉不到的对抗性扰动所愚弄。对抗性训练是将对抗性样本添加到训练集中的过程，是目前对抗对抗性攻击的最先进的防御措施，但它降低了模型在干净输入上的准确性，计算成本较高，对自然噪声的健壮性较差。相反，基于能量的模型(EBM)是为在神经形态硬件和物理系统中有效实施而设计的，它包含了从每一层到前一层的反馈连接，产生了一个递归的深吸引子结构，我们假设这种结构应该使它们自然地健壮。我们的工作是第一次探索EBM对自然腐败和对手攻击的稳健性，我们使用CIFAR-10和CIFAR-100数据集。我们证明了EBM比转换器更健壮，并且在基于梯度(白盒)攻击、基于查询(黑盒)攻击和自然扰动的情况下表现出与对手训练的DNN相当的鲁棒性，而不牺牲干净的准确性，并且不需要对抗性训练或额外的训练技术。



## **13. Finding a Needle in the Adversarial Haystack: A Targeted Paraphrasing Approach For Uncovering Edge Cases with Minimal Distribution Distortion**

在对抗性的干草堆中找针：一种最小分布失真的边缘案例发现的有针对性的释义方法 cs.CL

EACL 2024 - Main conference

**SubmitDate**: 2024-01-21    [abs](http://arxiv.org/abs/2401.11373v1) [paper-pdf](http://arxiv.org/pdf/2401.11373v1)

**Authors**: Aly M. Kassem, Sherif Saad

**Abstract**: Adversarial attacks against NLP Deep Learning models are a significant concern. In particular, adversarial samples exploit the model's sensitivity to small input changes. While these changes appear insignificant on the semantics of the input sample, they result in significant decay in model performance. In this paper, we propose Targeted Paraphrasing via RL (TPRL), an approach to automatically learn a policy to generate challenging samples that most likely improve the model's performance. TPRL leverages FLAN T5, a language model, as a generator and employs a self learned policy using a proximal policy gradient to generate the adversarial examples automatically. TPRL's reward is based on the confusion induced in the classifier, preserving the original text meaning through a Mutual Implication score. We demonstrate and evaluate TPRL's effectiveness in discovering natural adversarial attacks and improving model performance through extensive experiments on four diverse NLP classification tasks via Automatic and Human evaluation. TPRL outperforms strong baselines, exhibits generalizability across classifiers and datasets, and combines the strengths of language modeling and reinforcement learning to generate diverse and influential adversarial examples.

摘要: 针对NLP深度学习模型的对抗性攻击是一个重大问题。特别是，对抗性样本利用了模型对微小输入变化的敏感性。虽然这些更改在输入样本的语义上看起来并不重要，但它们会导致模型性能的显著下降。在本文中，我们提出了一种通过RL(TPRL)来自动学习策略以生成具有挑战性的样本的方法，该方法最有可能提高模型的性能。TPRL利用FRAN T5语言模型作为生成器，并采用使用邻近策略梯度的自学习策略来自动生成对抗性实例。TPRL的奖励是基于分类器中引起的混乱，通过相互蕴涵分数保留原始文本的意义。通过对四种不同的NLP分类任务的自动评估和人工评估，我们展示和评估了TPRL在发现自然对抗性攻击和提高模型性能方面的有效性。TPRL的性能优于强基线，表现出跨分类器和数据集的泛化能力，并结合语言建模和强化学习的优势来生成不同和有影响力的对抗性实例。



## **14. Explainability-Driven Leaf Disease Classification Using Adversarial Training and Knowledge Distillation**

基于对抗性训练和知识提炼的可解释性叶部病害分类 cs.CV

10 pages, 8 figures, Accepted by ICAART 2024

**SubmitDate**: 2024-01-23    [abs](http://arxiv.org/abs/2401.00334v3) [paper-pdf](http://arxiv.org/pdf/2401.00334v3)

**Authors**: Sebastian-Vasile Echim, Iulian-Marius Tăiatu, Dumitru-Clementin Cercel, Florin Pop

**Abstract**: This work focuses on plant leaf disease classification and explores three crucial aspects: adversarial training, model explainability, and model compression. The models' robustness against adversarial attacks is enhanced through adversarial training, ensuring accurate classification even in the presence of threats. Leveraging explainability techniques, we gain insights into the model's decision-making process, improving trust and transparency. Additionally, we explore model compression techniques to optimize computational efficiency while maintaining classification performance. Through our experiments, we determine that on a benchmark dataset, the robustness can be the price of the classification accuracy with performance reductions of 3%-20% for regular tests and gains of 50%-70% for adversarial attack tests. We also demonstrate that a student model can be 15-25 times more computationally efficient for a slight performance reduction, distilling the knowledge of more complex models.

摘要: 这项工作集中在植物叶部病害的分类上，并探索了三个关键方面：对抗性训练、模型可解释性和模型压缩。通过对抗性训练增强了模型对对抗性攻击的稳健性，即使在存在威胁的情况下也确保了准确的分类。利用可解释性技术，我们可以深入了解模型的决策过程，从而提高信任和透明度。此外，我们还探索了模型压缩技术，以优化计算效率，同时保持分类性能。通过实验，我们确定在一个基准数据集上，在常规测试性能降低3%-20%，对抗性攻击测试性能提高50%-70%的情况下，鲁棒性可以是分类准确率的代价。我们还证明，学生模型的计算效率可以是15-25倍，而性能略有下降，提取了更复杂模型的知识。



## **15. Robustness Against Adversarial Attacks via Learning Confined Adversarial Polytopes**

基于受限对抗性多面体学习的抗敌意攻击能力 cs.LG

The paper has been accepted in ICASSP 2024

**SubmitDate**: 2024-01-20    [abs](http://arxiv.org/abs/2401.07991v2) [paper-pdf](http://arxiv.org/pdf/2401.07991v2)

**Authors**: Shayan Mohajer Hamidi, Linfeng Ye

**Abstract**: Deep neural networks (DNNs) could be deceived by generating human-imperceptible perturbations of clean samples. Therefore, enhancing the robustness of DNNs against adversarial attacks is a crucial task. In this paper, we aim to train robust DNNs by limiting the set of outputs reachable via a norm-bounded perturbation added to a clean sample. We refer to this set as adversarial polytope, and each clean sample has a respective adversarial polytope. Indeed, if the respective polytopes for all the samples are compact such that they do not intersect the decision boundaries of the DNN, then the DNN is robust against adversarial samples. Hence, the inner-working of our algorithm is based on learning \textbf{c}onfined \textbf{a}dversarial \textbf{p}olytopes (CAP). By conducting a thorough set of experiments, we demonstrate the effectiveness of CAP over existing adversarial robustness methods in improving the robustness of models against state-of-the-art attacks including AutoAttack.

摘要: 深度神经网络(DNN)可以通过产生人类无法察觉的干净样本的扰动来欺骗。因此，提高DNN对敌意攻击的健壮性是一项至关重要的任务。在本文中，我们的目标是通过限制通过添加到干净样本的范数有界扰动可到达的输出集来训练鲁棒的DNN。我们将这个集合称为对抗性多面体，每个干净的样本都有各自的对抗性多面体。事实上，如果所有样本的相应多面体是紧凑的，使得它们不与DNN的决策边界相交，则DNN对对抗性样本是健壮的。因此，我们的算法的内部工作是基于学习文本bf{c}受限的文本bf{a}分叉算法(CAP)。通过一组详细的实验，我们证明了CAP在提高模型对包括AutoAttack在内的最新攻击的稳健性方面优于现有的对抗性稳健性方法。



## **16. Jailbreaking GPT-4V via Self-Adversarial Attacks with System Prompts**

通过系统提示的自我对抗性攻击越狱GPT-4V cs.CR

**SubmitDate**: 2024-01-20    [abs](http://arxiv.org/abs/2311.09127v2) [paper-pdf](http://arxiv.org/pdf/2311.09127v2)

**Authors**: Yuanwei Wu, Xiang Li, Yixin Liu, Pan Zhou, Lichao Sun

**Abstract**: Existing work on jailbreak Multimodal Large Language Models (MLLMs) has focused primarily on adversarial examples in model inputs, with less attention to vulnerabilities, especially in model API. To fill the research gap, we carry out the following work: 1) We discover a system prompt leakage vulnerability in GPT-4V. Through carefully designed dialogue, we successfully extract the internal system prompts of GPT-4V. This finding indicates potential exploitable security risks in MLLMs; 2) Based on the acquired system prompts, we propose a novel MLLM jailbreaking attack method termed SASP (Self-Adversarial Attack via System Prompt). By employing GPT-4 as a red teaming tool against itself, we aim to search for potential jailbreak prompts leveraging stolen system prompts. Furthermore, in pursuit of better performance, we also add human modification based on GPT-4's analysis, which further improves the attack success rate to 98.7\%; 3) We evaluated the effect of modifying system prompts to defend against jailbreaking attacks. Results show that appropriately designed system prompts can significantly reduce jailbreak success rates. Overall, our work provides new insights into enhancing MLLM security, demonstrating the important role of system prompts in jailbreaking. This finding could be leveraged to greatly facilitate jailbreak success rates while also holding the potential for defending against jailbreaks.

摘要: 现有关于越狱多模式大型语言模型(MLLMS)的工作主要集中在模型输入中的对抗性示例，对漏洞的关注较少，特别是在模型API中。为了填补这一研究空白，我们开展了以下工作：1)在GPT-4V中发现了一个系统即时泄漏漏洞。通过精心设计的对话，我们成功地提取了GPT-4V的内部系统提示。2)基于获得的系统提示，提出了一种新的基于系统提示的MLLM越狱攻击方法SASP(Self-Aversarial Attack by System Prompt)。通过使用GPT-4作为针对自己的红色团队工具，我们的目标是利用被盗的系统提示来搜索潜在的越狱提示。此外，为了追求更好的性能，我们还在GPT-4的S分析的基础上增加了人工修改，进一步将攻击成功率提高到98.7%。3)评估了修改系统提示对越狱攻击的防御效果。结果表明，设计适当的系统提示可以显著降低越狱成功率。总体而言，我们的工作为加强MLLM安全提供了新的见解，展示了系统提示在越狱中的重要作用。这一发现可以被用来极大地提高越狱成功率，同时还具有防御越狱的潜力。



## **17. Susceptibility of Adversarial Attack on Medical Image Segmentation Models**

对抗性攻击对医学图像分割模型的敏感性 eess.IV

6 pages, 8 figures, presented at 2023 IEEE 20th International  Symposium on Biomedical Imaging (ISBI) conference

**SubmitDate**: 2024-01-20    [abs](http://arxiv.org/abs/2401.11224v1) [paper-pdf](http://arxiv.org/pdf/2401.11224v1)

**Authors**: Zhongxuan Wang, Leo Xu

**Abstract**: The nature of deep neural networks has given rise to a variety of attacks, but little work has been done to address the effect of adversarial attacks on segmentation models trained on MRI datasets. In light of the grave consequences that such attacks could cause, we explore four models from the U-Net family and examine their responses to the Fast Gradient Sign Method (FGSM) attack. We conduct FGSM attacks on each of them and experiment with various schemes to conduct the attacks. In this paper, we find that medical imaging segmentation models are indeed vulnerable to adversarial attacks and that there is a negligible correlation between parameter size and adversarial attack success. Furthermore, we show that using a different loss function than the one used for training yields higher adversarial attack success, contrary to what the FGSM authors suggested. In future efforts, we will conduct the experiments detailed in this paper with more segmentation models and different attacks. We will also attempt to find ways to counteract the attacks by using model ensembles or special data augmentations. Our code is available at https://github.com/ZhongxuanWang/adv_attk

摘要: 深度神经网络的性质导致了各种各样的攻击，但对于对抗性攻击对基于MRI数据集训练的分割模型的影响，人们所做的工作很少。鉴于此类攻击可能造成的严重后果，我们探索了U-Net家族的四个模型，并检查了它们对快速梯度符号方法(FGSM)攻击的响应。我们对他们中的每一个进行FGSM攻击，并试验各种攻击方案。在本文中，我们发现医学图像分割模型确实容易受到对抗性攻击，并且参数大小与对抗性攻击的成功与否之间的相关性可以忽略不计。此外，我们还表明，与FGSM作者的建议相反，使用与训练中使用的损失函数不同的损失函数会产生更高的对抗性攻击成功率。在未来的努力中，我们将使用更多的分割模型和不同的攻击来进行本文详细介绍的实验。我们还将尝试通过使用模型集成或特殊数据增强来找到对抗攻击的方法。我们的代码可以在https://github.com/ZhongxuanWang/adv_attk上找到



## **18. Generalizing Speaker Verification for Spoof Awareness in the Embedding Space**

嵌入空间中基于欺骗感知的说话人确认泛化 cs.CR

To appear in IEEE/ACM Transactions on Audio, Speech, and Language  Processing

**SubmitDate**: 2024-01-20    [abs](http://arxiv.org/abs/2401.11156v1) [paper-pdf](http://arxiv.org/pdf/2401.11156v1)

**Authors**: Xuechen Liu, Md Sahidullah, Kong Aik Lee, Tomi Kinnunen

**Abstract**: It is now well-known that automatic speaker verification (ASV) systems can be spoofed using various types of adversaries. The usual approach to counteract ASV systems against such attacks is to develop a separate spoofing countermeasure (CM) module to classify speech input either as a bonafide, or a spoofed utterance. Nevertheless, such a design requires additional computation and utilization efforts at the authentication stage. An alternative strategy involves a single monolithic ASV system designed to handle both zero-effort imposter (non-targets) and spoofing attacks. Such spoof-aware ASV systems have the potential to provide stronger protections and more economic computations. To this end, we propose to generalize the standalone ASV (G-SASV) against spoofing attacks, where we leverage limited training data from CM to enhance a simple backend in the embedding space, without the involvement of a separate CM module during the test (authentication) phase. We propose a novel yet simple backend classifier based on deep neural networks and conduct the study via domain adaptation and multi-task integration of spoof embeddings at the training stage. Experiments are conducted on the ASVspoof 2019 logical access dataset, where we improve the performance of statistical ASV backends on the joint (bonafide and spoofed) and spoofed conditions by a maximum of 36.2% and 49.8% in terms of equal error rates, respectively.

摘要: 现在众所周知，自动说话人验证(ASV)系统可以使用各种类型的对手进行欺骗。对抗ASV系统抵御此类攻击的通常方法是开发单独的欺骗对策(CM)模块，以将语音输入分类为真正的或欺骗的话语。然而，这样的设计在身份验证阶段需要额外的计算和利用工作。另一种策略包括一个单一的单片ASV系统，旨在同时处理零努力冒名顶替者(非目标)和欺骗攻击。这种感知欺骗的ASV系统有可能提供更强大的保护和更多的经济计算。为此，我们建议推广抗欺骗攻击的独立ASV(G-SASV)，其中我们利用来自CM的有限训练数据来增强嵌入空间中的简单后端，而不需要在测试(身份验证)阶段涉及单独的CM模块。我们提出了一种新颖而简单的基于深度神经网络的后端分类器，并在训练阶段通过域自适应和欺骗嵌入的多任务集成进行了研究。在ASVspoof 2019逻辑访问数据集上进行了实验，在相同错误率的情况下，我们将联合(真实和欺骗)和欺骗条件下的统计ASV后端的性能分别提高了36.2%和49.8%。



## **19. CARE: Ensemble Adversarial Robustness Evaluation Against Adaptive Attackers for Security Applications**

CARE：针对自适应攻击者的安全应用集成攻击健壮性评估 cs.CR

**SubmitDate**: 2024-01-20    [abs](http://arxiv.org/abs/2401.11126v1) [paper-pdf](http://arxiv.org/pdf/2401.11126v1)

**Authors**: Hangsheng Zhang, Jiqiang Liu, Jinsong Dong

**Abstract**: Ensemble defenses, are widely employed in various security-related applications to enhance model performance and robustness. The widespread adoption of these techniques also raises many questions: Are general ensembles defenses guaranteed to be more robust than individuals? Will stronger adaptive attacks defeat existing ensemble defense strategies as the cybersecurity arms race progresses? Can ensemble defenses achieve adversarial robustness to different types of attacks simultaneously and resist the continually adjusted adaptive attacks? Unfortunately, these critical questions remain unresolved as there are no platforms for comprehensive evaluation of ensemble adversarial attacks and defenses in the cybersecurity domain. In this paper, we propose a general Cybersecurity Adversarial Robustness Evaluation (CARE) platform aiming to bridge this gap.

摘要: 集成防御被广泛应用于各种安全相关应用中，以增强模型的性能和稳健性。这些技术的广泛采用也引发了许多问题：一般的集体防御是否一定比个人防御更强大？随着网络安全军备竞赛的推进，更强大的适应性攻击是否会击败现有的整体防御战略？集成防御能否同时实现对不同类型攻击的对抗健壮性，并抵抗不断调整的适应性攻击？不幸的是，这些关键问题仍然没有得到解决，因为没有全面评估网络安全领域中的总体对抗性攻击和防御的平台。在本文中，我们提出了一个通用的网络安全对抗健壮性评估(CARE)平台，旨在弥补这一差距。



## **20. Universal Backdoor Attacks**

通用后门攻击 cs.LG

Accepted for publication at ICLR 2024

**SubmitDate**: 2024-01-20    [abs](http://arxiv.org/abs/2312.00157v2) [paper-pdf](http://arxiv.org/pdf/2312.00157v2)

**Authors**: Benjamin Schneider, Nils Lukas, Florian Kerschbaum

**Abstract**: Web-scraped datasets are vulnerable to data poisoning, which can be used for backdooring deep image classifiers during training. Since training on large datasets is expensive, a model is trained once and re-used many times. Unlike adversarial examples, backdoor attacks often target specific classes rather than any class learned by the model. One might expect that targeting many classes through a naive composition of attacks vastly increases the number of poison samples. We show this is not necessarily true and more efficient, universal data poisoning attacks exist that allow controlling misclassifications from any source class into any target class with a small increase in poison samples. Our idea is to generate triggers with salient characteristics that the model can learn. The triggers we craft exploit a phenomenon we call inter-class poison transferability, where learning a trigger from one class makes the model more vulnerable to learning triggers for other classes. We demonstrate the effectiveness and robustness of our universal backdoor attacks by controlling models with up to 6,000 classes while poisoning only 0.15% of the training dataset. Our source code is available at https://github.com/Ben-Schneider-code/Universal-Backdoor-Attacks.

摘要: 网络抓取的数据集很容易受到数据中毒的影响，在训练过程中，数据中毒可以用于回溯深度图像分类器。由于在大型数据集上进行训练的成本很高，因此一个模型只需训练一次，就可以多次重复使用。与对抗性示例不同，后门攻击通常针对特定类，而不是模型学习到的任何类。人们可能会认为，通过天真的攻击组合以许多类别为目标会极大地增加毒物样本的数量。我们证明这不一定是真的，而且更有效，普遍存在的数据中毒攻击允许在毒物样本少量增加的情况下控制从任何源类到任何目标类的误分类。我们的想法是生成模型可以学习的具有显著特征的触发器。我们制作的触发器利用了一种我们称为类间毒药可转移性的现象，即从一个类学习触发器使模型更容易学习其他类的触发器。我们通过控制多达6,000个类的模型来展示我们的通用后门攻击的有效性和健壮性，而只毒化了0.15%的训练数据集。我们的源代码可以在https://github.com/Ben-Schneider-code/Universal-Backdoor-Attacks.上找到



## **21. Ensembler: Combating model inversion attacks using model ensemble during collaborative inference**

集成器：在协作推理过程中使用模型集成对抗模型反转攻击 cs.CR

in submission

**SubmitDate**: 2024-01-19    [abs](http://arxiv.org/abs/2401.10859v1) [paper-pdf](http://arxiv.org/pdf/2401.10859v1)

**Authors**: Dancheng Liu, Jinjun Xiong

**Abstract**: Deep learning models have exhibited remarkable performance across various domains. Nevertheless, the burgeoning model sizes compel edge devices to offload a significant portion of the inference process to the cloud. While this practice offers numerous advantages, it also raises critical concerns regarding user data privacy. In scenarios where the cloud server's trustworthiness is in question, the need for a practical and adaptable method to safeguard data privacy becomes imperative. In this paper, we introduce Ensembler, an extensible framework designed to substantially increase the difficulty of conducting model inversion attacks for adversarial parties. Ensembler leverages model ensembling on the adversarial server, running in parallel with existing approaches that introduce perturbations to sensitive data during colloborative inference. Our experiments demonstrate that when combined with even basic Gaussian noise, Ensembler can effectively shield images from reconstruction attacks, achieving recognition levels that fall below human performance in some strict settings, significantly outperforming baseline methods lacking the Ensembler framework.

摘要: 深度学习模型在各个领域都表现出了显著的性能。然而，迅速增长的模型规模迫使边缘设备将很大一部分推理过程转移到云上。虽然这种做法提供了许多优势，但它也引发了对用户数据隐私的严重担忧。在云服务器的可信性受到质疑的情况下，需要一种实用且适应性强的方法来保护数据隐私变得势在必行。在这篇文章中，我们介绍了一个可扩展的框架，该框架旨在大幅增加对敌方进行模型反转攻击的难度。集成利用敌意服务器上的模型集成，与现有方法并行运行，这些方法在协同推理过程中对敏感数据引入扰动。我们的实验表明，当与基本的高斯噪声相结合时，集成可以有效地保护图像免受重建攻击，在某些严格的设置下获得低于人类表现的识别水平，显著优于缺乏集成框架的基线方法。



## **22. Privacy-Preserving Neural Graph Databases**

保护隐私的神经图库 cs.DB

**SubmitDate**: 2024-01-19    [abs](http://arxiv.org/abs/2312.15591v2) [paper-pdf](http://arxiv.org/pdf/2312.15591v2)

**Authors**: Qi Hu, Haoran Li, Jiaxin Bai, Yangqiu Song

**Abstract**: In the era of big data and rapidly evolving information systems, efficient and accurate data retrieval has become increasingly crucial. Neural graph databases (NGDBs) have emerged as a powerful paradigm that combines the strengths of graph databases (graph DBs) and neural networks to enable efficient storage, retrieval, and analysis of graph-structured data. The usage of neural embedding storage and complex neural logical query answering provides NGDBs with generalization ability. When the graph is incomplete, by extracting latent patterns and representations, neural graph databases can fill gaps in the graph structure, revealing hidden relationships and enabling accurate query answering. Nevertheless, this capability comes with inherent trade-offs, as it introduces additional privacy risks to the database. Malicious attackers can infer more sensitive information in the database using well-designed combinatorial queries, such as by comparing the answer sets of where Turing Award winners born before 1950 and after 1940 lived, the living places of Turing Award winner Hinton are probably exposed, although the living places may have been deleted in the training due to the privacy concerns. In this work, inspired by the privacy protection in graph embeddings, we propose a privacy-preserving neural graph database (P-NGDB) to alleviate the risks of privacy leakage in NGDBs. We introduce adversarial training techniques in the training stage to force the NGDBs to generate indistinguishable answers when queried with private information, enhancing the difficulty of inferring sensitive information through combinations of multiple innocuous queries. Extensive experiment results on three datasets show that P-NGDB can effectively protect private information in the graph database while delivering high-quality public answers responses to queries.

摘要: 在大数据和快速发展的信息系统的时代，高效和准确的数据检索变得越来越重要。神经图形数据库(NGDB)已经成为一种强大的范例，它结合了图形数据库(图形数据库)和神经网络的优点，使得能够有效地存储、检索和分析图形结构的数据。神经嵌入存储和复杂神经逻辑查询回答的使用为NGDB提供了泛化能力。当图不完整时，通过提取潜在模式和表示，神经图库可以填补图结构中的空白，揭示隐藏的关系，并使查询得到准确的回答。尽管如此，这种能力也伴随着固有的权衡，因为它会给数据库带来额外的隐私风险。恶意攻击者可以使用精心设计的组合查询来推断数据库中更敏感的信息，例如通过比较1950年之前出生的图灵奖获得者和1940年后出生的图灵奖获得者的答案集，图灵奖获得者辛顿的居住地可能会被曝光，尽管出于隐私考虑，在训练中可能已经删除了居住地。在这项工作中，我们受到图嵌入中隐私保护的启发，提出了一种隐私保护神经图库(P-NGDB)来缓解NGDB中隐私泄露的风险。我们在训练阶段引入对抗性训练技术，迫使NGDB在查询私有信息时产生难以区分的答案，增加了通过组合多个无害查询来推断敏感信息的难度。在三个数据集上的大量实验结果表明，P-NGDB可以有效地保护图形数据库中的私有信息，同时提供高质量的公共查询响应。



## **23. Explainable and Transferable Adversarial Attack for ML-Based Network Intrusion Detectors**

基于ML的网络入侵检测的可解释可转移敌意攻击 cs.CR

**SubmitDate**: 2024-01-19    [abs](http://arxiv.org/abs/2401.10691v1) [paper-pdf](http://arxiv.org/pdf/2401.10691v1)

**Authors**: Hangsheng Zhang, Dongqi Han, Yinlong Liu, Zhiliang Wang, Jiyan Sun, Shangyuan Zhuang, Jiqiang Liu, Jinsong Dong

**Abstract**: espite being widely used in network intrusion detection systems (NIDSs), machine learning (ML) has proven to be highly vulnerable to adversarial attacks. White-box and black-box adversarial attacks of NIDS have been explored in several studies. However, white-box attacks unrealistically assume that the attackers have full knowledge of the target NIDSs. Meanwhile, existing black-box attacks can not achieve high attack success rate due to the weak adversarial transferability between models (e.g., neural networks and tree models). Additionally, neither of them explains why adversarial examples exist and why they can transfer across models. To address these challenges, this paper introduces ETA, an Explainable Transfer-based Black-Box Adversarial Attack framework. ETA aims to achieve two primary objectives: 1) create transferable adversarial examples applicable to various ML models and 2) provide insights into the existence of adversarial examples and their transferability within NIDSs. Specifically, we first provide a general transfer-based adversarial attack method applicable across the entire ML space. Following that, we exploit a unique insight based on cooperative game theory and perturbation interpretations to explain adversarial examples and adversarial transferability. On this basis, we propose an Important-Sensitive Feature Selection (ISFS) method to guide the search for adversarial examples, achieving stronger transferability and ensuring traffic-space constraints.

摘要: 尽管机器学习在网络入侵检测系统中得到了广泛的应用，但它被证明是非常容易受到对手攻击的。网络入侵检测系统的白盒和黑盒对抗性攻击已经在多个研究中得到了探索。然而，白盒攻击不切实际地假设攻击者完全了解目标NIDS。同时，现有的黑盒攻击由于模型(如神经网络和树模型)之间的对抗性较弱而不能达到较高的攻击成功率。此外，它们都没有解释为什么存在对抗性例子，以及为什么它们可以在模型之间转移。为了应对这些挑战，本文引入了一种可解释的基于传输的黑盒对抗攻击框架ETA。ETA旨在实现两个主要目标：1)创建适用于各种ML模型的可转移的对抗性例子；2)提供对对抗性例子的存在及其在新入侵检测系统中的可转移性的见解。具体地说，我们首先提供了一种适用于整个ML空间的通用的基于转移的对抗性攻击方法。然后，我们利用基于合作博弈论和扰动解释的独特见解来解释对抗性例子和对抗性转移。在此基础上，提出了一种重要敏感的特征选择(ISFS)方法来指导对抗性实例的搜索，实现了较强的可转移性，并保证了交通空间的约束。



## **24. FIMBA: Evaluating the Robustness of AI in Genomics via Feature Importance Adversarial Attacks**

FIMBA：通过特征重要性攻击评估基因组学中人工智能的健壮性 cs.LG

15 pages, core code available at:  https://github.com/HeorhiiS/fimba-attack

**SubmitDate**: 2024-01-19    [abs](http://arxiv.org/abs/2401.10657v1) [paper-pdf](http://arxiv.org/pdf/2401.10657v1)

**Authors**: Heorhii Skovorodnikov, Hoda Alkhzaimi

**Abstract**: With the steady rise of the use of AI in bio-technical applications and the widespread adoption of genomics sequencing, an increasing amount of AI-based algorithms and tools is entering the research and production stage affecting critical decision-making streams like drug discovery and clinical outcomes. This paper demonstrates the vulnerability of AI models often utilized downstream tasks on recognized public genomics datasets. We undermine model robustness by deploying an attack that focuses on input transformation while mimicking the real data and confusing the model decision-making, ultimately yielding a pronounced deterioration in model performance. Further, we enhance our approach by generating poisoned data using a variational autoencoder-based model. Our empirical findings unequivocally demonstrate a decline in model performance, underscored by diminished accuracy and an upswing in false positives and false negatives. Furthermore, we analyze the resulting adversarial samples via spectral analysis yielding conclusions for countermeasures against such attacks.

摘要: 随着人工智能在生物技术应用中的稳步上升和基因组测序的广泛采用，越来越多的基于人工智能的算法和工具正在进入研究和生产阶段，影响着药物发现和临床结果等关键决策流。本文论证了在公认的公共基因组数据集上经常利用下游任务的人工智能模型的脆弱性。我们通过部署一种专注于输入转换的攻击来破坏模型的健壮性，同时模仿真实数据并混淆模型决策，最终导致模型性能的显著恶化。此外，我们通过使用基于变分自动编码器的模型来生成有毒数据来增强我们的方法。我们的经验发现明确地证明了模型性能的下降，强调了准确性的降低和假阳性和假阴性的上升。此外，我们通过频谱分析对生成的敌意样本进行分析，得出针对此类攻击的对策结论。



## **25. Adversarially Robust Signed Graph Contrastive Learning from Balance Augmentation**

基于平衡增强的对偶稳健符号图对比学习 cs.LG

**SubmitDate**: 2024-01-19    [abs](http://arxiv.org/abs/2401.10590v1) [paper-pdf](http://arxiv.org/pdf/2401.10590v1)

**Authors**: Jialong Zhou, Xing Ai, Yuni Lai, Kai Zhou

**Abstract**: Signed graphs consist of edges and signs, which can be separated into structural information and balance-related information, respectively. Existing signed graph neural networks (SGNNs) typically rely on balance-related information to generate embeddings. Nevertheless, the emergence of recent adversarial attacks has had a detrimental impact on the balance-related information. Similar to how structure learning can restore unsigned graphs, balance learning can be applied to signed graphs by improving the balance degree of the poisoned graph. However, this approach encounters the challenge "Irreversibility of Balance-related Information" - while the balance degree improves, the restored edges may not be the ones originally affected by attacks, resulting in poor defense effectiveness. To address this challenge, we propose a robust SGNN framework called Balance Augmented-Signed Graph Contrastive Learning (BA-SGCL), which combines Graph Contrastive Learning principles with balance augmentation techniques. Experimental results demonstrate that BA-SGCL not only enhances robustness against existing adversarial attacks but also achieves superior performance on link sign prediction task across various datasets.

摘要: 符号图由边和符号组成，边和符号分别可分为结构信息和平衡相关信息。现有的符号图神经网络(SGNN)通常依赖于与余额相关的信息来生成嵌入。尽管如此，最近出现的对抗性攻击对与余额有关的信息产生了不利影响。类似于结构学习如何恢复无符号图，平衡学习可以通过提高中毒图的平衡度来应用于有符号图。然而，这种方法遇到了挑战--在平衡度提高的同时，恢复的边缘可能不是最初受攻击影响的边缘，导致防御效果不佳。为了应对这一挑战，我们提出了一种稳健的SGNN框架，称为平衡增强符号图对比学习(BA-SGCL)，它结合了图对比学习原理和平衡增强技术。实验结果表明，BA-SGCL不仅提高了对现有对手攻击的健壮性，而且在不同数据集上的链接符号预测任务中取得了优异的性能。



## **26. PuriDefense: Randomized Local Implicit Adversarial Purification for Defending Black-box Query-based Attacks**

PuriDefense：用于防御基于黑盒查询的攻击的随机局部隐式对抗性净化 cs.CR

**SubmitDate**: 2024-01-19    [abs](http://arxiv.org/abs/2401.10586v1) [paper-pdf](http://arxiv.org/pdf/2401.10586v1)

**Authors**: Ping Guo, Zhiyuan Yang, Xi Lin, Qingchuan Zhao, Qingfu Zhang

**Abstract**: Black-box query-based attacks constitute significant threats to Machine Learning as a Service (MLaaS) systems since they can generate adversarial examples without accessing the target model's architecture and parameters. Traditional defense mechanisms, such as adversarial training, gradient masking, and input transformations, either impose substantial computational costs or compromise the test accuracy of non-adversarial inputs. To address these challenges, we propose an efficient defense mechanism, PuriDefense, that employs random patch-wise purifications with an ensemble of lightweight purification models at a low level of inference cost. These models leverage the local implicit function and rebuild the natural image manifold. Our theoretical analysis suggests that this approach slows down the convergence of query-based attacks by incorporating randomness into purifications. Extensive experiments on CIFAR-10 and ImageNet validate the effectiveness of our proposed purifier-based defense mechanism, demonstrating significant improvements in robustness against query-based attacks.

摘要: 基于黑盒查询的攻击对机器学习即服务(MLaaS)系统构成了重大威胁，因为它们可以在不访问目标模型的体系结构和参数的情况下生成对抗性示例。传统的防御机制，如对抗性训练、梯度掩蔽和输入转换，要么增加了大量的计算成本，要么损害了非对抗性输入的测试精度。为了应对这些挑战，我们提出了一个有效的防御机制PuriDefense，它使用了随机的补丁式净化，并以较低的推理成本集成了轻量级的净化模型。这些模型利用局部隐函数重建自然图像流形。我们的理论分析表明，这种方法通过将随机性纳入净化中，减缓了基于查询的攻击的收敛。在CIFAR-10和ImageNet上的大量实验验证了我们提出的基于净化器的防御机制的有效性，显示出对基于查询的攻击的健壮性显著提高。



## **27. Hijacking Attacks against Neural Networks by Analyzing Training Data**

基于训练数据分析的神经网络劫持攻击 cs.CR

Full version with major polishing, compared to the Usenix Security  2024 edition

**SubmitDate**: 2024-01-19    [abs](http://arxiv.org/abs/2401.09740v2) [paper-pdf](http://arxiv.org/pdf/2401.09740v2)

**Authors**: Yunjie Ge, Qian Wang, Huayang Huang, Qi Li, Cong Wang, Chao Shen, Lingchen Zhao, Peipei Jiang, Zheng Fang, Shenyi Zhang

**Abstract**: Backdoors and adversarial examples are the two primary threats currently faced by deep neural networks (DNNs). Both attacks attempt to hijack the model behaviors with unintended outputs by introducing (small) perturbations to the inputs. Backdoor attacks, despite the high success rates, often require a strong assumption, which is not always easy to achieve in reality. Adversarial example attacks, which put relatively weaker assumptions on attackers, often demand high computational resources, yet do not always yield satisfactory success rates when attacking mainstream black-box models in the real world. These limitations motivate the following research question: can model hijacking be achieved more simply, with a higher attack success rate and more reasonable assumptions? In this paper, we propose CleanSheet, a new model hijacking attack that obtains the high performance of backdoor attacks without requiring the adversary to tamper with the model training process. CleanSheet exploits vulnerabilities in DNNs stemming from the training data. Specifically, our key idea is to treat part of the clean training data of the target model as "poisoned data," and capture the characteristics of these data that are more sensitive to the model (typically called robust features) to construct "triggers." These triggers can be added to any input example to mislead the target model, similar to backdoor attacks. We validate the effectiveness of CleanSheet through extensive experiments on 5 datasets, 79 normally trained models, 68 pruned models, and 39 defensive models. Results show that CleanSheet exhibits performance comparable to state-of-the-art backdoor attacks, achieving an average attack success rate (ASR) of 97.5% on CIFAR-100 and 92.4% on GTSRB, respectively. Furthermore, CleanSheet consistently maintains a high ASR, when confronted with various mainstream backdoor defenses.

摘要: 后门和敌意例子是深度神经网络(DNN)目前面临的两个主要威胁。这两种攻击都试图通过向输入引入(小)扰动来劫持具有非预期输出的模型行为。后门攻击尽管成功率很高，但往往需要强有力的假设，而这在现实中并不总是容易实现的。对抗性例子攻击对攻击者的假设相对较弱，通常需要很高的计算资源，但在攻击现实世界中的主流黑盒模型时，并不总是产生令人满意的成功率。这些局限性引发了以下研究问题：能否以更高的攻击成功率和更合理的假设更简单地实现模型劫持？在本文中，我们提出了一种新的劫持攻击模型CleanSheet，它在不要求对手篡改模型训练过程的情况下获得了后门攻击的高性能。CleanSheet利用源自训练数据的DNN中的漏洞。具体地说，我们的关键思想是将目标模型的部分干净训练数据视为“有毒数据”，并捕获这些数据中对模型更敏感的特征(通常称为稳健特征)来构建“触发器”。这些触发器可以添加到任何输入示例中，以误导目标模型，类似于后门攻击。我们通过在5个数据集、79个正常训练模型、68个剪枝模型和39个防御模型上的大量实验验证了Clear Sheet的有效性。结果表明，CleanSheet的攻击性能与最先进的后门攻击相当，在CIFAR-100和GTSRB上的平均攻击成功率分别达到97.5%和92.4%。此外，当面对各种主流的后门防御时，CleanSheet始终保持着较高的ASR。



## **28. Adversarial Attack On Yolov5 For Traffic And Road Sign Detection**

Yolov5交通路标检测系统的对抗性攻击 cs.CV

**SubmitDate**: 2024-01-19    [abs](http://arxiv.org/abs/2306.06071v2) [paper-pdf](http://arxiv.org/pdf/2306.06071v2)

**Authors**: Sanyam Jain

**Abstract**: This paper implements and investigates popular adversarial attacks on the YOLOv5 Object Detection algorithm. The paper explores the vulnerability of the YOLOv5 to adversarial attacks in the context of traffic and road sign detection. The paper investigates the impact of different types of attacks, including the Limited memory Broyden Fletcher Goldfarb Shanno (L-BFGS), the Fast Gradient Sign Method (FGSM) attack, the Carlini and Wagner (C&W) attack, the Basic Iterative Method (BIM) attack, the Projected Gradient Descent (PGD) attack, One Pixel Attack, and the Universal Adversarial Perturbations attack on the accuracy of YOLOv5 in detecting traffic and road signs. The results show that YOLOv5 is susceptible to these attacks, with misclassification rates increasing as the magnitude of the perturbations increases. We also explain the results using saliency maps. The findings of this paper have important implications for the safety and reliability of object detection algorithms used in traffic and transportation systems, highlighting the need for more robust and secure models to ensure their effectiveness in real-world applications.

摘要: 本文实现并研究了针对YOLOv5目标检测算法的常见对抗性攻击。本文探讨了YOLOv5在交通和路标检测环境中对敌意攻击的脆弱性。研究了有限记忆Broyden Fletcher Goldfarb Shanno(L-BFGS)攻击、快速梯度符号法(FGSM)攻击、Carlini和Wagner(C&W)攻击、基本迭代法(BIM)攻击、投影梯度下降(PGD)攻击、单像素攻击和通用对抗性扰动攻击等不同类型攻击对YOLOv5检测交通和道路标志准确率的影响。结果表明，YOLOv5容易受到这些攻击，误识率随着扰动程度的增加而增加。我们还使用显著图解释了结果。本文的研究结果对交通运输系统中使用的目标检测算法的安全性和可靠性具有重要意义，强调了需要更健壮和安全的模型来确保其在现实世界应用中的有效性。



## **29. On the Adversarial Robustness of Camera-based 3D Object Detection**

基于摄像机的三维目标检测的对抗性研究 cs.CV

Transactions on Machine Learning Research, 2024. ISSN 2835-8856

**SubmitDate**: 2024-01-19    [abs](http://arxiv.org/abs/2301.10766v2) [paper-pdf](http://arxiv.org/pdf/2301.10766v2)

**Authors**: Shaoyuan Xie, Zichao Li, Zeyu Wang, Cihang Xie

**Abstract**: In recent years, camera-based 3D object detection has gained widespread attention for its ability to achieve high performance with low computational cost. However, the robustness of these methods to adversarial attacks has not been thoroughly examined, especially when considering their deployment in safety-critical domains like autonomous driving. In this study, we conduct the first comprehensive investigation of the robustness of leading camera-based 3D object detection approaches under various adversarial conditions. We systematically analyze the resilience of these models under two attack settings: white-box and black-box; focusing on two primary objectives: classification and localization. Additionally, we delve into two types of adversarial attack techniques: pixel-based and patch-based. Our experiments yield four interesting findings: (a) bird's-eye-view-based representations exhibit stronger robustness against localization attacks; (b) depth-estimation-free approaches have the potential to show stronger robustness; (c) accurate depth estimation effectively improves robustness for depth-estimation-based methods; (d) incorporating multi-frame benign inputs can effectively mitigate adversarial attacks. We hope our findings can steer the development of future camera-based object detection models with enhanced adversarial robustness.

摘要: 近年来，基于摄像机的三维目标检测由于能够以较低的计算代价获得较高的检测性能而受到广泛关注。然而，这些方法对敌意攻击的稳健性还没有得到彻底的检验，特别是在考虑到它们部署在自动驾驶等安全关键领域时。在这项研究中，我们首次全面调查了主流的基于摄像机的3D目标检测方法在各种对抗条件下的稳健性。我们系统地分析了这些模型在白盒和黑盒两种攻击环境下的抗攻击能力，重点关注两个主要目标：分类和局部化。此外，我们深入研究了两种类型的对抗性攻击技术：基于像素的攻击和基于补丁的攻击。我们的实验得到了四个有趣的发现：(A)基于鸟瞰视图的表示对局部化攻击表现出更强的稳健性；(B)无深度估计的方法具有更强的鲁棒性；(C)准确的深度估计有效地提高了基于深度估计的方法的稳健性；(D)结合多帧良性输入可以有效地缓解对抗性攻击。我们希望我们的发现能够指导未来基于摄像机的目标检测模型的发展，增强对手的稳健性。



## **30. Differentially Private and Adversarially Robust Machine Learning: An Empirical Evaluation**

区分隐私和相对稳健的机器学习：一项经验评估 cs.LG

Accepted at PPAI-24: The 5th AAAI Workshop on Privacy-Preserving  Artificial Intelligence

**SubmitDate**: 2024-01-18    [abs](http://arxiv.org/abs/2401.10405v1) [paper-pdf](http://arxiv.org/pdf/2401.10405v1)

**Authors**: Janvi Thakkar, Giulio Zizzo, Sergio Maffeis

**Abstract**: Malicious adversaries can attack machine learning models to infer sensitive information or damage the system by launching a series of evasion attacks. Although various work addresses privacy and security concerns, they focus on individual defenses, but in practice, models may undergo simultaneous attacks. This study explores the combination of adversarial training and differentially private training to defend against simultaneous attacks. While differentially-private adversarial training, as presented in DP-Adv, outperforms the other state-of-the-art methods in performance, it lacks formal privacy guarantees and empirical validation. Thus, in this work, we benchmark the performance of this technique using a membership inference attack and empirically show that the resulting approach is as private as non-robust private models. This work also highlights the need to explore privacy guarantees in dynamic training paradigms.

摘要: 恶意攻击者可以攻击机器学习模型来推断敏感信息，或者通过发起一系列逃避攻击来破坏系统。尽管各种工作解决了隐私和安全方面的问题，但它们侧重于单独的防御，但在实践中，模型可能会同时受到攻击。本研究探讨了对抗性训练与差异化私人训练相结合来防御同时攻击的方法。尽管DP-ADV中提出的不同隐私对抗训练在性能上优于其他最先进的方法，但它缺乏正式的隐私保障和经验验证。因此，在这项工作中，我们使用成员关系推理攻击对该技术的性能进行了基准测试，并经验表明所得到的方法与非健壮的私有模型一样私密。这项工作还突出了在动态培训范例中探索隐私保障的必要性。



## **31. Vulnerabilities of Foundation Model Integrated Federated Learning Under Adversarial Threats**

基础模型集成联邦学习在对抗威胁下的脆弱性 cs.CR

Chen Wu and Xi Li are equal contribution. The corresponding author is  Jiaqi Wang

**SubmitDate**: 2024-01-18    [abs](http://arxiv.org/abs/2401.10375v1) [paper-pdf](http://arxiv.org/pdf/2401.10375v1)

**Authors**: Chen Wu, Xi Li, Jiaqi Wang

**Abstract**: Federated Learning (FL) addresses critical issues in machine learning related to data privacy and security, yet suffering from data insufficiency and imbalance under certain circumstances. The emergence of foundation models (FMs) offers potential solutions to the limitations of existing FL frameworks, e.g., by generating synthetic data for model initialization. However, due to the inherent safety concerns of FMs, integrating FMs into FL could introduce new risks, which remains largely unexplored. To address this gap, we conduct the first investigation on the vulnerability of FM integrated FL (FM-FL) under adversarial threats. Based on a unified framework of FM-FL, we introduce a novel attack strategy that exploits safety issues of FM to compromise FL client models. Through extensive experiments with well-known models and benchmark datasets in both image and text domains, we reveal the high susceptibility of the FM-FL to this new threat under various FL configurations. Furthermore, we find that existing FL defense strategies offer limited protection against this novel attack approach. This research highlights the critical need for enhanced security measures in FL in the era of FMs.

摘要: 联合学习(FL)解决了机器学习中与数据隐私和安全相关的关键问题，但在某些情况下存在数据不足和不平衡的问题。基础模型(FM)的出现为现有FL框架的局限性提供了潜在的解决方案，例如通过生成用于模型初始化的合成数据。然而，由于FMS固有的安全问题，将FMS整合到FL中可能会带来新的风险，这在很大程度上仍未被探索。为了弥补这一差距，我们首次对FM集成FL(FM-FL)在对手威胁下的脆弱性进行了研究。基于FM-FL的统一框架，我们提出了一种新的攻击策略，利用FM的安全问题来危害FL客户端模型。通过在图像域和文本域使用著名的模型和基准数据集进行广泛的实验，我们揭示了FM-FL在不同FL配置下对这种新威胁的高度敏感性。此外，我们发现，现有的FL防御策略对这种新的攻击方法提供的保护有限。这项研究强调了在FMS时代加强FL安全措施的迫切需要。



## **32. Attack and Defense Analysis of Learned Image Compression**

学习图像压缩的攻防分析 eess.IV

**SubmitDate**: 2024-01-18    [abs](http://arxiv.org/abs/2401.10345v1) [paper-pdf](http://arxiv.org/pdf/2401.10345v1)

**Authors**: Tianyu Zhu, Heming Sun, Xiankui Xiong, Xuanpeng Zhu, Yong Gong, Minge jing, Yibo Fan

**Abstract**: Learned image compression (LIC) is becoming more and more popular these years with its high efficiency and outstanding compression quality. Still, the practicality against modified inputs added with specific noise could not be ignored. White-box attacks such as FGSM and PGD use only gradient to compute adversarial images that mislead LIC models to output unexpected results. Our experiments compare the effects of different dimensions such as attack methods, models, qualities, and targets, concluding that in the worst case, there is a 61.55% decrease in PSNR or a 19.15 times increase in bit rate under the PGD attack. To improve their robustness, we conduct adversarial training by adding adversarial images into the training datasets, which obtains a 95.52% decrease in the R-D cost of the most vulnerable LIC model. We further test the robustness of H.266, whose better performance on reconstruction quality extends its possibility to defend one-step or iterative adversarial attacks.

摘要: 学习图像压缩(LIC)以其高效的压缩效果和优异的压缩质量，近年来受到越来越多的关注。尽管如此，相对于添加了特定噪声的修改输入的实用性不能被忽视。像FGSM和PGD这样的白盒攻击只使用梯度来计算敌意图像，从而误导LIC模型输出意外结果。实验比较了攻击方式、攻击模型、攻击质量、攻击目标等不同维度的影响，结果表明，在最坏的情况下，在PGD攻击下，峰值信噪比下降了61.55%，比特率提高了19.15倍。为了提高其稳健性，我们通过在训练数据集中加入对抗性图像进行对抗性训练，使得最脆弱的LIC模型的研发代价降低了95.52%。我们进一步测试了H.266的健壮性，它在重建质量上的良好性能扩展了它防御一步攻击或迭代攻击的可能性。



## **33. Hacking Predictors Means Hacking Cars: Using Sensitivity Analysis to Identify Trajectory Prediction Vulnerabilities for Autonomous Driving Security**

黑客预测器意味着黑客汽车：使用敏感度分析识别自动驾驶安全的轨迹预测漏洞 cs.CR

10 pages, 6 figures, 1 tables

**SubmitDate**: 2024-01-18    [abs](http://arxiv.org/abs/2401.10313v1) [paper-pdf](http://arxiv.org/pdf/2401.10313v1)

**Authors**: Marsalis Gibson, David Babazadeh, Claire Tomlin, Shankar Sastry

**Abstract**: Adversarial attacks on learning-based trajectory predictors have already been demonstrated. However, there are still open questions about the effects of perturbations on trajectory predictor inputs other than state histories, and how these attacks impact downstream planning and control. In this paper, we conduct a sensitivity analysis on two trajectory prediction models, Trajectron++ and AgentFormer. We observe that between all inputs, almost all of the perturbation sensitivities for Trajectron++ lie only within the most recent state history time point, while perturbation sensitivities for AgentFormer are spread across state histories over time. We additionally demonstrate that, despite dominant sensitivity on state history perturbations, an undetectable image map perturbation made with the Fast Gradient Sign Method can induce large prediction error increases in both models. Even though image maps may contribute slightly to the prediction output of both models, this result reveals that rather than being robust to adversarial image perturbations, trajectory predictors are susceptible to image attacks. Using an optimization-based planner and example perturbations crafted from sensitivity results, we show how this vulnerability can cause a vehicle to come to a sudden stop from moderate driving speeds.

摘要: 针对基于学习的轨迹预测器的对抗性攻击已经得到了证明。然而，关于摄动对状态历史以外的轨迹预测器输入的影响，以及这些攻击如何影响下游规划和控制，仍然存在悬而未决的问题。本文对两种弹道预测模型Trajectron++和AgentFormer进行了灵敏度分析。我们观察到，在所有输入之间，Trajectron++的几乎所有微扰灵敏度都只位于最近的状态历史时间点内，而AgentFormer的微扰灵敏度则跨状态历史分布。此外，我们还证明了，尽管对状态历史扰动的主要敏感性，但用快速梯度符号方法进行的不可检测的图像映射扰动在两个模型中都会导致预测误差的大幅增加。尽管图像映射对这两个模型的预测输出可能有轻微的贡献，但这一结果表明，轨迹预测器不是对对抗性图像扰动具有健壮性，而是容易受到图像攻击。使用基于优化的计划器和根据灵敏度结果制作的示例扰动，我们展示了该漏洞如何导致车辆在中等速度下突然停止。



## **34. Marrying Adapters and Mixup to Efficiently Enhance the Adversarial Robustness of Pre-Trained Language Models for Text Classification**

结合Adapters和Mixup有效增强文本分类预训练语言模型的对抗健壮性 cs.CL

10 pages and 2 figures

**SubmitDate**: 2024-01-18    [abs](http://arxiv.org/abs/2401.10111v1) [paper-pdf](http://arxiv.org/pdf/2401.10111v1)

**Authors**: Tuc Nguyen, Thai Le

**Abstract**: Existing works show that augmenting training data of neural networks using both clean and adversarial examples can enhance their generalizability under adversarial attacks. However, this training approach often leads to performance degradation on clean inputs. Additionally, it requires frequent re-training of the entire model to account for new attack types, resulting in significant and costly computations. Such limitations make adversarial training mechanisms less practical, particularly for complex Pre-trained Language Models (PLMs) with millions or even billions of parameters. To overcome these challenges while still harnessing the theoretical benefits of adversarial training, this study combines two concepts: (1) adapters, which enable parameter-efficient fine-tuning, and (2) Mixup, which train NNs via convex combinations of pairs data pairs. Intuitively, we propose to fine-tune PLMs through convex combinations of non-data pairs of fine-tuned adapters, one trained with clean and another trained with adversarial examples. Our experiments show that the proposed method achieves the best trade-off between training efficiency and predictive performance, both with and without attacks compared to other baselines on a variety of downstream tasks.

摘要: 已有的工作表明，使用干净的和对抗性的例子来扩充神经网络的训练数据，可以增强神经网络在对抗性攻击下的泛化能力。然而，这种培训方法往往会导致清洁投入的绩效下降。此外，它需要频繁地重新训练整个模型以考虑新的攻击类型，从而导致大量且昂贵的计算。这些限制使得对抗性训练机制变得不那么实用，特别是对于具有数百万甚至数十亿参数的复杂的预训练语言模型(PLM)。为了克服这些挑战，同时仍然利用对抗性训练的理论优势，本研究结合了两个概念：(1)适配器，它实现了参数高效的微调；(2)MIXUP，它通过对数据对的凸组合来训练NN。直观地，我们建议通过微调适配器的非数据对的凸性组合来微调PLM，其中一个用CLEAN训练，另一个用对抗性例子训练。实验表明，该方法在训练效率和预测性能之间取得了最好的折衷，无论是在有攻击还是没有攻击的情况下，与其他针对各种下游任务的基线相比。



## **35. Power in Numbers: Robust reading comprehension by finetuning with four adversarial sentences per example**

数字中的力量：通过每例四个对抗性句子的精细调整来增强阅读理解能力 cs.CL

**SubmitDate**: 2024-01-18    [abs](http://arxiv.org/abs/2401.10091v1) [paper-pdf](http://arxiv.org/pdf/2401.10091v1)

**Authors**: Ariel Marcus

**Abstract**: Recent models have achieved human level performance on the Stanford Question Answering Dataset when using F1 scores to evaluate the reading comprehension task. Yet, teaching machines to comprehend text has not been solved in the general case. By appending one adversarial sentence to the context paragraph, past research has shown that the F1 scores from reading comprehension models drop almost in half. In this paper, I replicate past adversarial research with a new model, ELECTRA-Small, and demonstrate that the new model's F1 score drops from 83.9% to 29.2%. To improve ELECTRA-Small's resistance to this attack, I finetune the model on SQuAD v1.1 training examples with one to five adversarial sentences appended to the context paragraph. Like past research, I find that the finetuned model on one adversarial sentence does not generalize well across evaluation datasets. However, when finetuned on four or five adversarial sentences the model attains an F1 score of more than 70% on most evaluation datasets with multiple appended and prepended adversarial sentences. The results suggest that with enough examples we can make models robust to adversarial attacks.

摘要: 最近的模型已经在斯坦福问答数据集上取得了人类水平的表现，当使用F1分数来评估阅读理解任务时。然而，在一般情况下，教机器理解文本并没有得到解决。过去的研究表明，通过在上下文段落中添加一个对抗性句子，阅读理解模型的F1分数几乎下降了一半。在本文中，我用一个新的模型ELECTRA-Small复制了过去的对抗性研究，并证明了新模型的F1得分从83.9%下降到29.2%。为了提高Electra-Small对这种攻击的抵抗力，我在小队V1.1训练实例上微调了模型，并在上下文段落中附加了一到五个对抗性句子。像过去的研究一样，我发现在一个对抗性句子上的精调模型不能很好地在评估数据集上进行泛化。然而，当对四个或五个对抗性句子进行优化时，该模型在具有多个附加和预先添加的对抗性句子的大多数评估数据集上获得了超过70%的F1分数。结果表明，有了足够的例子，我们可以使模型对对手攻击具有健壮性。



## **36. HGAttack: Transferable Heterogeneous Graph Adversarial Attack**

HGAttack：可转移的异构图对抗攻击 cs.LG

**SubmitDate**: 2024-01-18    [abs](http://arxiv.org/abs/2401.09945v1) [paper-pdf](http://arxiv.org/pdf/2401.09945v1)

**Authors**: He Zhao, Zhiwei Zeng, Yongwei Wang, Deheng Ye, Chunyan Miao

**Abstract**: Heterogeneous Graph Neural Networks (HGNNs) are increasingly recognized for their performance in areas like the web and e-commerce, where resilience against adversarial attacks is crucial. However, existing adversarial attack methods, which are primarily designed for homogeneous graphs, fall short when applied to HGNNs due to their limited ability to address the structural and semantic complexity of HGNNs. This paper introduces HGAttack, the first dedicated gray box evasion attack method for heterogeneous graphs. We design a novel surrogate model to closely resemble the behaviors of the target HGNN and utilize gradient-based methods for perturbation generation. Specifically, the proposed surrogate model effectively leverages heterogeneous information by extracting meta-path induced subgraphs and applying GNNs to learn node embeddings with distinct semantics from each subgraph. This approach improves the transferability of generated attacks on the target HGNN and significantly reduces memory costs. For perturbation generation, we introduce a semantics-aware mechanism that leverages subgraph gradient information to autonomously identify vulnerable edges across a wide range of relations within a constrained perturbation budget. We validate HGAttack's efficacy with comprehensive experiments on three datasets, providing empirical analyses of its generated perturbations. Outperforming baseline methods, HGAttack demonstrated significant efficacy in diminishing the performance of target HGNN models, affirming the effectiveness of our approach in evaluating the robustness of HGNNs against adversarial attacks.

摘要: 异构图神经网络(HGNN)因其在网络和电子商务等领域的性能而日益受到人们的认可，在这些领域，对对手攻击的弹性至关重要。然而，现有的对抗性攻击方法主要是针对同构图设计的，但由于其有限的能力来解决HGNN的结构和语义复杂性，因此在应用于HGNN时存在不足。介绍了针对异构图的第一种专用灰盒逃避攻击方法HGAttack。我们设计了一种新的代理模型来接近目标HGNN的行为，并利用基于梯度的方法来产生扰动。具体地说，该代理模型通过提取元路径诱导子图并应用GNN从每个子图中学习具有不同语义的节点嵌入，有效地利用了异质信息。该方法提高了生成的攻击在目标HGNN上的可转移性，并显著降低了内存成本。对于扰动生成，我们引入了一种语义感知机制，该机制利用子图梯度信息在受限的扰动预算内自动识别各种关系中的易受攻击边。我们通过在三个数据集上的综合实验验证了HGAttack的有效性，并对其产生的扰动进行了实证分析。与基准方法相比，HGAttack在降低目标HGNN模型的性能方面表现出显著的有效性，肯定了我们方法在评估HGNN对对手攻击的健壮性方面的有效性。



## **37. Universally Robust Graph Neural Networks by Preserving Neighbor Similarity**

保持邻域相似性的泛健性图神经网络 cs.LG

**SubmitDate**: 2024-01-18    [abs](http://arxiv.org/abs/2401.09754v1) [paper-pdf](http://arxiv.org/pdf/2401.09754v1)

**Authors**: Yulin Zhu, Yuni Lai, Xing Ai, Kai Zhou

**Abstract**: Despite the tremendous success of graph neural networks in learning relational data, it has been widely investigated that graph neural networks are vulnerable to structural attacks on homophilic graphs. Motivated by this, a surge of robust models is crafted to enhance the adversarial robustness of graph neural networks on homophilic graphs. However, the vulnerability based on heterophilic graphs remains a mystery to us. To bridge this gap, in this paper, we start to explore the vulnerability of graph neural networks on heterophilic graphs and theoretically prove that the update of the negative classification loss is negatively correlated with the pairwise similarities based on the powered aggregated neighbor features. This theoretical proof explains the empirical observations that the graph attacker tends to connect dissimilar node pairs based on the similarities of neighbor features instead of ego features both on homophilic and heterophilic graphs. In this way, we novelly introduce a novel robust model termed NSPGNN which incorporates a dual-kNN graphs pipeline to supervise the neighbor similarity-guided propagation. This propagation utilizes the low-pass filter to smooth the features of node pairs along the positive kNN graphs and the high-pass filter to discriminate the features of node pairs along the negative kNN graphs. Extensive experiments on both homophilic and heterophilic graphs validate the universal robustness of NSPGNN compared to the state-of-the-art methods.

摘要: 尽管图神经网络在学习关系数据方面取得了巨大的成功，但人们已经广泛研究了图神经网络容易受到同亲图的结构攻击。在此基础上，设计了一系列健壮模型，以增强图神经网络在同亲图上的对抗健壮性。然而，基于异嗜图的漏洞对我们来说仍然是一个谜。为了弥补这一差距，本文首先探讨了图神经网络在异嗜图上的脆弱性，并从理论上证明了负分类损失的更新与基于加权聚合邻域特征的成对相似性负相关。这一理论证明解释了图攻击者倾向于在同嗜图和异嗜图上基于邻居特征的相似性而不是自我特征的相似性来连接不同的节点对的经验观察。通过这种方式，我们新颖地提出了一种新的健壮模型NSPGNN，该模型结合了双KNN图流水线来监督邻居相似性引导的传播。该传播算法利用低通滤波对正KNN图上的节点对特征进行平滑处理，利用高通滤波来区分负KNN图上的节点对特征。在同亲图和异亲图上的广泛实验验证了NSPGNN相对于最先进的方法的普遍稳健性。



## **38. X-CANIDS: Signal-Aware Explainable Intrusion Detection System for Controller Area Network-Based In-Vehicle Network**

X-CANIDS：基于控制器局域网的车载网络信号感知可解释入侵检测系统 cs.CR

This is the Accepted version of an article for publication in IEEE  TVT

**SubmitDate**: 2024-01-18    [abs](http://arxiv.org/abs/2303.12278v2) [paper-pdf](http://arxiv.org/pdf/2303.12278v2)

**Authors**: Seonghoon Jeong, Sangho Lee, Hwejae Lee, Huy Kang Kim

**Abstract**: Controller Area Network (CAN) is an essential networking protocol that connects multiple electronic control units (ECUs) in a vehicle. However, CAN-based in-vehicle networks (IVNs) face security risks owing to the CAN mechanisms. An adversary can sabotage a vehicle by leveraging the security risks if they can access the CAN bus. Thus, recent actions and cybersecurity regulations (e.g., UNR 155) require carmakers to implement intrusion detection systems (IDSs) in their vehicles. The IDS should detect cyberattacks and provide additional information to analyze conducted attacks. Although many IDSs have been proposed, considerations regarding their feasibility and explainability remain lacking. This study proposes X-CANIDS, which is a novel IDS for CAN-based IVNs. X-CANIDS dissects the payloads in CAN messages into human-understandable signals using a CAN database. The signals improve the intrusion detection performance compared with the use of bit representations of raw payloads. These signals also enable an understanding of which signal or ECU is under attack. X-CANIDS can detect zero-day attacks because it does not require any labeled dataset in the training phase. We confirmed the feasibility of the proposed method through a benchmark test on an automotive-grade embedded device with a GPU. The results of this work will be valuable to carmakers and researchers considering the installation of in-vehicle IDSs for their vehicles.

摘要: 控制器局域网(CAN)是连接车辆中多个电子控制单元(ECU)的基本网络协议。然而，由于CAN机制的存在，基于CAN的车载网络面临着安全隐患。如果对手可以访问CAN总线，则他们可以利用安全风险来破坏车辆。因此，最近的行动和网络安全法规(例如，UNR 155)要求汽车制造商在其车辆中安装入侵检测系统(IDS)。入侵检测系统应该检测网络攻击，并提供其他信息来分析进行的攻击。虽然已经提出了许多入侵检测系统，但仍然缺乏对其可行性和可解释性的考虑。本研究提出了一种新型的基于CAN网络的入侵检测系统--X-CANID。X-Canids使用CAN数据库将CAN消息中的有效载荷分解为人类可理解的信号。与使用原始有效载荷的比特表示相比，该信号提高了入侵检测性能。这些信号还使您能够了解哪个信号或ECU受到攻击。X-CARID可以检测零日攻击，因为它在训练阶段不需要任何标记的数据集。通过在一款搭载GPU的车载嵌入式设备上的基准测试，验证了该方法的可行性。这项工作的结果将对汽车制造商和考虑为其车辆安装车载入侵检测系统的研究人员具有价值。



## **39. Artwork Protection Against Neural Style Transfer Using Locally Adaptive Adversarial Color Attack**

基于局部自适应对抗性色彩攻击的艺术品神经风格转移保护 cs.CV

9 pages, 5 figures

**SubmitDate**: 2024-01-18    [abs](http://arxiv.org/abs/2401.09673v1) [paper-pdf](http://arxiv.org/pdf/2401.09673v1)

**Authors**: Zhongliang Guo, Kaixuan Wang, Weiye Li, Yifei Qian, Ognjen Arandjelović, Lei Fang

**Abstract**: Neural style transfer (NST) is widely adopted in computer vision to generate new images with arbitrary styles. This process leverages neural networks to merge aesthetic elements of a style image with the structural aspects of a content image into a harmoniously integrated visual result. However, unauthorized NST can exploit artwork. Such misuse raises socio-technical concerns regarding artists' rights and motivates the development of technical approaches for the proactive protection of original creations. Adversarial attack is a concept primarily explored in machine learning security. Our work introduces this technique to protect artists' intellectual property. In this paper Locally Adaptive Adversarial Color Attack (LAACA), a method for altering images in a manner imperceptible to the human eyes but disruptive to NST. Specifically, we design perturbations targeting image areas rich in high-frequency content, generated by disrupting intermediate features. Our experiments and user study confirm that by attacking NST using the proposed method results in visually worse neural style transfer, thus making it an effective solution for visual artwork protection.

摘要: 神经样式转移(NST)是计算机视觉中广泛采用的一种生成任意样式新图像的方法。该过程利用神经网络将样式图像的美学元素与内容图像的结构方面合并为和谐集成的视觉结果。然而，未经授权的NST可以利用艺术品。这种滥用引起了对艺术家权利的社会技术关切，并促使开发技术方法来积极保护原创作品。对抗性攻击是机器学习安全领域主要探讨的一个概念。我们的作品引入了这种技术来保护艺术家的知识产权。在本文中，局部自适应对抗性颜色攻击(LAACA)是一种以人眼无法察觉的方式改变图像的方法，但对NST具有破坏性。具体地说，我们针对高频内容丰富的图像区域设计扰动，这些区域是通过破坏中间特征而产生的。我们的实验和用户研究证实，使用该方法攻击NST会导致视觉上较差的神经风格迁移，从而使其成为视觉艺术品保护的有效解决方案。



## **40. MITS-GAN: Safeguarding Medical Imaging from Tampering with Generative Adversarial Networks**

MITS-GAN：保护医学影像不受生成性对抗网络的篡改 eess.IV

**SubmitDate**: 2024-01-17    [abs](http://arxiv.org/abs/2401.09624v1) [paper-pdf](http://arxiv.org/pdf/2401.09624v1)

**Authors**: Giovanni Pasqualino, Luca Guarnera, Alessandro Ortis, Sebastiano Battiato

**Abstract**: The progress in generative models, particularly Generative Adversarial Networks (GANs), opened new possibilities for image generation but raised concerns about potential malicious uses, especially in sensitive areas like medical imaging. This study introduces MITS-GAN, a novel approach to prevent tampering in medical images, with a specific focus on CT scans. The approach disrupts the output of the attacker's CT-GAN architecture by introducing imperceptible but yet precise perturbations. Specifically, the proposed approach involves the introduction of appropriate Gaussian noise to the input as a protective measure against various attacks. Our method aims to enhance tamper resistance, comparing favorably to existing techniques. Experimental results on a CT scan dataset demonstrate MITS-GAN's superior performance, emphasizing its ability to generate tamper-resistant images with negligible artifacts. As image tampering in medical domains poses life-threatening risks, our proactive approach contributes to the responsible and ethical use of generative models. This work provides a foundation for future research in countering cyber threats in medical imaging. Models and codes are publicly available at the following link \url{https://iplab.dmi.unict.it/MITS-GAN-2024/}.

摘要: 生成性模型的进展，特别是生成性对抗网络(GANS)，为图像生成打开了新的可能性，但也引发了人们对潜在恶意使用的担忧，特别是在医学成像等敏感领域。这项研究介绍了一种新的防止医学图像篡改的方法MITS-GaN，重点介绍了CT扫描。该方法通过引入难以察觉但却精确的扰动，破坏了攻击者的CT-GaN体系结构的输出。具体地说，所提出的方法包括在输入中引入适当的高斯噪声作为对各种攻击的保护措施。我们的方法旨在增强抗篡改能力，与现有技术相比是有利的。在CT扫描数据集上的实验结果显示了MITS-GaN的优越性能，强调了其生成具有可忽略伪影的防篡改图像的能力。由于医学领域的图像篡改带来了危及生命的风险，我们积极主动的方法有助于负责任和合乎道德地使用生殖模型。这项工作为未来在医学成像中对抗网络威胁的研究提供了基础。型号和代码可在以下链接\url{https://iplab.dmi.unict.it/MITS-GAN-2024/}.上公开获得



## **41. Towards Scalable and Robust Model Versioning**

走向可扩展和健壮的模型版本控制 cs.LG

Accepted in IEEE SaTML 2024

**SubmitDate**: 2024-01-17    [abs](http://arxiv.org/abs/2401.09574v1) [paper-pdf](http://arxiv.org/pdf/2401.09574v1)

**Authors**: Wenxin Ding, Arjun Nitin Bhagoji, Ben Y. Zhao, Haitao Zheng

**Abstract**: As the deployment of deep learning models continues to expand across industries, the threat of malicious incursions aimed at gaining access to these deployed models is on the rise. Should an attacker gain access to a deployed model, whether through server breaches, insider attacks, or model inversion techniques, they can then construct white-box adversarial attacks to manipulate the model's classification outcomes, thereby posing significant risks to organizations that rely on these models for critical tasks. Model owners need mechanisms to protect themselves against such losses without the necessity of acquiring fresh training data - a process that typically demands substantial investments in time and capital.   In this paper, we explore the feasibility of generating multiple versions of a model that possess different attack properties, without acquiring new training data or changing model architecture. The model owner can deploy one version at a time and replace a leaked version immediately with a new version. The newly deployed model version can resist adversarial attacks generated leveraging white-box access to one or all previously leaked versions. We show theoretically that this can be accomplished by incorporating parameterized hidden distributions into the model training data, forcing the model to learn task-irrelevant features uniquely defined by the chosen data. Additionally, optimal choices of hidden distributions can produce a sequence of model versions capable of resisting compound transferability attacks over time. Leveraging our analytical insights, we design and implement a practical model versioning method for DNN classifiers, which leads to significant robustness improvements over existing methods. We believe our work presents a promising direction for safeguarding DNN services beyond their initial deployment.

摘要: 随着深度学习模型的部署继续跨行业扩展，旨在访问这些已部署模型的恶意入侵威胁正在上升。如果攻击者获得对已部署模型的访问权限，无论是通过服务器入侵、内部攻击或模型倒置技术，他们都可以构建白盒对抗性攻击来操纵模型的分类结果，从而给依赖这些模型执行关键任务的组织带来重大风险。模型所有者需要机制来保护自己免受此类损失，而不需要获取新的培训数据-这一过程通常需要在时间和资金上进行大量投资。在本文中，我们探索了在不获取新的训练数据或改变模型体系结构的情况下，生成具有不同攻击属性的模型的多个版本的可行性。模型所有者可以一次部署一个版本，并立即用新版本替换泄漏的版本。新部署的模型版本可以抵抗利用白盒访问一个或所有先前泄露的版本而产生的对抗性攻击。我们从理论上证明，这可以通过将参数化的隐藏分布结合到模型训练数据中来实现，迫使模型学习由所选数据唯一定义的与任务无关的特征。此外，隐藏分布的最佳选择可以产生一系列模型版本，能够随着时间的推移抵抗复合可转移性攻击。利用我们的分析洞察力，我们设计并实现了一种实用的DNN分类器模型版本控制方法，与现有方法相比，该方法具有显著的健壮性改进。我们相信，我们的工作为保护DNN服务提供了一个很有前途的方向，而不是最初的部署。



## **42. Diffusion-Based Adversarial Sample Generation for Improved Stealthiness and Controllability**

基于扩散的改进隐蔽性和可控性的对抗样本生成 cs.CV

Accepted as a conference paper in NeurIPS'2023. Code repo:  https://github.com/xavihart/Diff-PGD

**SubmitDate**: 2024-01-17    [abs](http://arxiv.org/abs/2305.16494v3) [paper-pdf](http://arxiv.org/pdf/2305.16494v3)

**Authors**: Haotian Xue, Alexandre Araujo, Bin Hu, Yongxin Chen

**Abstract**: Neural networks are known to be susceptible to adversarial samples: small variations of natural examples crafted to deliberately mislead the models. While they can be easily generated using gradient-based techniques in digital and physical scenarios, they often differ greatly from the actual data distribution of natural images, resulting in a trade-off between strength and stealthiness. In this paper, we propose a novel framework dubbed Diffusion-Based Projected Gradient Descent (Diff-PGD) for generating realistic adversarial samples. By exploiting a gradient guided by a diffusion model, Diff-PGD ensures that adversarial samples remain close to the original data distribution while maintaining their effectiveness. Moreover, our framework can be easily customized for specific tasks such as digital attacks, physical-world attacks, and style-based attacks. Compared with existing methods for generating natural-style adversarial samples, our framework enables the separation of optimizing adversarial loss from other surrogate losses (e.g., content/smoothness/style loss), making it more stable and controllable. Finally, we demonstrate that the samples generated using Diff-PGD have better transferability and anti-purification power than traditional gradient-based methods. Code will be released in https://github.com/xavihart/Diff-PGD

摘要: 众所周知，神经网络容易受到敌意样本的影响，这些样本是自然样本的微小变体，目的是故意误导模型。虽然在数字和物理场景中可以很容易地使用基于梯度的技术来生成它们，但它们往往与自然图像的实际数据分布有很大差异，导致在强度和隐蔽性之间进行权衡。在本文中，我们提出了一种新的框架，称为基于扩散的投影梯度下降(DIFF-PGD)，用于生成真实的对抗性样本。通过利用扩散模型引导的梯度，DIFF-PGD在保持有效性的同时，确保对手样本保持接近原始数据分布。此外，我们的框架可以很容易地针对特定任务进行定制，例如数字攻击、物理世界攻击和基于样式的攻击。与现有的生成自然风格对抗性样本的方法相比，我们的框架能够将优化对抗性损失与其他代理损失(例如，内容/流畅度/风格损失)分离，使其更加稳定和可控。最后，我们证明了DIFF-PGD生成的样本比传统的基于梯度的方法具有更好的可转移性和抗净化能力。代码将在https://github.com/xavihart/Diff-PGD中发布



## **43. Adversarial Examples are Misaligned in Diffusion Model Manifolds**

扩散模型流形中对抗性例子的错位 cs.CV

under review

**SubmitDate**: 2024-01-17    [abs](http://arxiv.org/abs/2401.06637v3) [paper-pdf](http://arxiv.org/pdf/2401.06637v3)

**Authors**: Peter Lorenz, Ricard Durall, Janis Keuper

**Abstract**: In recent years, diffusion models (DMs) have drawn significant attention for their success in approximating data distributions, yielding state-of-the-art generative results. Nevertheless, the versatility of these models extends beyond their generative capabilities to encompass various vision applications, such as image inpainting, segmentation, adversarial robustness, among others. This study is dedicated to the investigation of adversarial attacks through the lens of diffusion models. However, our objective does not involve enhancing the adversarial robustness of image classifiers. Instead, our focus lies in utilizing the diffusion model to detect and analyze the anomalies introduced by these attacks on images. To that end, we systematically examine the alignment of the distributions of adversarial examples when subjected to the process of transformation using diffusion models. The efficacy of this approach is assessed across CIFAR-10 and ImageNet datasets, including varying image sizes in the latter. The results demonstrate a notable capacity to discriminate effectively between benign and attacked images, providing compelling evidence that adversarial instances do not align with the learned manifold of the DMs.

摘要: 近年来，扩散模型(DM)因其在近似数据分布方面的成功而引起了人们的极大关注，产生了最先进的生成结果。然而，这些模型的多功能性超出了它们的生成能力，涵盖了各种视觉应用，如图像修复、分割、对抗性鲁棒性等。本研究致力于从扩散模型的角度研究对抗性攻击。然而，我们的目标不涉及增强图像分类器的对抗性稳健性。相反，我们的重点在于利用扩散模型来检测和分析这些攻击对图像带来的异常。为此，我们使用扩散模型系统地考察了对抗性例子在经历转换过程时的分布的一致性。在CIFAR-10和ImageNet数据集上评估了这种方法的有效性，包括在后者中不同的图像大小。实验结果表明，该方法能够有效地区分良性图像和被攻击图像，提供了令人信服的证据，表明敌意实例与学习到的DM流形并不一致。



## **44. MIMIR: Masked Image Modeling for Mutual Information-based Adversarial Robustness**

MIMIR：基于交互信息的对抗性掩蔽图像建模 cs.CV

**SubmitDate**: 2024-01-17    [abs](http://arxiv.org/abs/2312.04960v2) [paper-pdf](http://arxiv.org/pdf/2312.04960v2)

**Authors**: Xiaoyun Xu, Shujian Yu, Jingzheng Wu, Stjepan Picek

**Abstract**: Vision Transformers (ViTs) achieve superior performance on various tasks compared to convolutional neural networks (CNNs), but ViTs are also vulnerable to adversarial attacks. Adversarial training is one of the most successful methods to build robust CNN models. Thus, recent works explored new methodologies for adversarial training of ViTs based on the differences between ViTs and CNNs, such as better training strategies, preventing attention from focusing on a single block, or discarding low-attention embeddings. However, these methods still follow the design of traditional supervised adversarial training, limiting the potential of adversarial training on ViTs. This paper proposes a novel defense method, MIMIR, which aims to build a different adversarial training methodology by utilizing Masked Image Modeling at pre-training. We create an autoencoder that accepts adversarial examples as input but takes the clean examples as the modeling target. Then, we create a mutual information (MI) penalty following the idea of the Information Bottleneck. Among the two information source inputs and corresponding adversarial perturbation, the perturbation information is eliminated due to the constraint of the modeling target. Next, we provide a theoretical analysis of MIMIR using the bounds of the MI penalty. We also design two adaptive attacks when the adversary is aware of the MIMIR defense and show that MIMIR still performs well. The experimental results show that MIMIR improves (natural and adversarial) accuracy on average by 4.19% on CIFAR-10 and 5.52% on ImageNet-1K, compared to baselines. On Tiny-ImageNet, we obtained improved natural accuracy of 2.99\% on average and comparable adversarial accuracy. Our code and trained models are publicly available https://github.com/xiaoyunxxy/MIMIR.

摘要: 与卷积神经网络(CNN)相比，视觉转换器(VITS)在各种任务中取得了优越的性能，但VITS也容易受到对手的攻击。对抗性训练是建立稳健的CNN模型最成功的方法之一。因此，最近的工作探索了基于VITS和CNN之间的差异的VITS对抗性训练的新方法，例如更好的训练策略，防止注意力集中在单个区块上，或者放弃低注意嵌入。然而，这些方法仍然遵循传统的监督对抗性训练的设计，限制了对抗性训练在VITS上的潜力。本文提出了一种新的防御方法MIMIR，旨在通过在训练前利用蒙版图像建模来构建一种不同的对手训练方法。我们创建了一个自动编码器，它接受对抗性例子作为输入，但以干净的例子作为建模目标。然后，我们遵循信息瓶颈的思想创建了一个互信息(MI)惩罚。在两个信息源输入和对应的对抗性扰动中，由于建模目标的限制，扰动信息被消除。接下来，我们利用MI惩罚的界对MIMIR进行了理论分析。我们还设计了两个自适应攻击，当对手知道Mimir防御时，表明Mimir仍然执行得很好。实验结果表明，与基准相比，MIMIR在CIFAR-10和ImageNet-1K上的(自然和对抗)准确率分别平均提高了4.19%和5.52%。在Micro-ImageNet上，我们获得了平均2.99\%的改进的自然准确率和相当的对手准确率。我们的代码和经过训练的模型是公开提供的https://github.com/xiaoyunxxy/MIMIR.



## **45. Username Squatting on Online Social Networks: A Study on X**

基于X的在线社交网络用户名蹲点行为研究 cs.CR

Accepted at ACM ASIA Conference on Computer and Communications  Security (AsiaCCS), 2024

**SubmitDate**: 2024-01-17    [abs](http://arxiv.org/abs/2401.09209v1) [paper-pdf](http://arxiv.org/pdf/2401.09209v1)

**Authors**: Anastasios Lepipas, Anastasia Borovykh, Soteris Demetriou

**Abstract**: Adversaries have been targeting unique identifiers to launch typo-squatting, mobile app squatting and even voice squatting attacks. Anecdotal evidence suggest that online social networks (OSNs) are also plagued with accounts that use similar usernames. This can be confusing to users but can also be exploited by adversaries. However, to date no study characterizes this problem on OSNs. In this work, we define the username squatting problem and design the first multi-faceted measurement study to characterize it on X. We develop a username generation tool (UsernameCrazy) to help us analyze hundreds of thousands of username variants derived from celebrity accounts. Our study reveals that thousands of squatted usernames have been suspended by X, while tens of thousands that still exist on the network are likely bots. Out of these, a large number share similar profile pictures and profile names to the original account signalling impersonation attempts. We found that squatted accounts are being mentioned by mistake in tweets hundreds of thousands of times and are even being prioritized in searches by the network's search recommendation algorithm exacerbating the negative impact squatted accounts can have in OSNs. We use our insights and take the first step to address this issue by designing a framework (SQUAD) that combines UsernameCrazy with a new classifier to efficiently detect suspicious squatted accounts. Our evaluation of SQUAD's prototype implementation shows that it can achieve 94% F1-score when trained on a small dataset.

摘要: 对手一直以唯一标识为目标，发动打字蹲守、手机应用蹲守，甚至语音蹲守攻击。坊间证据表明，在线社交网络(OSN)也充斥着使用相似用户名的账户。这可能会让用户感到困惑，但也可能被对手利用。然而，到目前为止，还没有研究描述OSN上的这个问题。在这项工作中，我们定义了用户名下蹲问题，并设计了第一个多方面的测量研究来刻画X上的用户名下蹲问题。我们开发了一个用户名生成工具(UsernameCrazy)来帮助我们分析来自名人账户的数十万个用户名变体。我们的研究显示，数以千计的蹲守用户名已经被X暂停，而网络上仍然存在的数万个用户名很可能是机器人。在这些中，有大量共享与原始帐户信令模拟尝试相似的配置文件图片和配置文件名称。我们发现，在推文中，蹲着的账户被错误地提到了数十万次，甚至在网络的搜索推荐算法的搜索中被优先考虑，加剧了蹲着的账户在OSN中可能产生的负面影响。我们利用我们的见解，通过设计一个框架(Team)来解决这个问题，该框架(Team)将UsernameCrazy与新的分类器相结合，以高效地检测可疑的蹲守帐户。我们对LONG原型实现的评估表明，当在小数据集上训练时，它可以达到94%的F1得分。



## **46. Attack and Reset for Unlearning: Exploiting Adversarial Noise toward Machine Unlearning through Parameter Re-initialization**

遗忘的攻击和重置：通过参数重新初始化利用机器遗忘的对抗性噪声 cs.LG

**SubmitDate**: 2024-01-17    [abs](http://arxiv.org/abs/2401.08998v1) [paper-pdf](http://arxiv.org/pdf/2401.08998v1)

**Authors**: Yoonhwa Jung, Ikhyun Cho, Shun-Hsiang Hsu, Julia Hockenmaier

**Abstract**: With growing concerns surrounding privacy and regulatory compliance, the concept of machine unlearning has gained prominence, aiming to selectively forget or erase specific learned information from a trained model. In response to this critical need, we introduce a novel approach called Attack-and-Reset for Unlearning (ARU). This algorithm leverages meticulously crafted adversarial noise to generate a parameter mask, effectively resetting certain parameters and rendering them unlearnable. ARU outperforms current state-of-the-art results on two facial machine-unlearning benchmark datasets, MUFAC and MUCAC. In particular, we present the steps involved in attacking and masking that strategically filter and re-initialize network parameters biased towards the forget set. Our work represents a significant advancement in rendering data unexploitable to deep learning models through parameter re-initialization, achieved by harnessing adversarial noise to craft a mask.

摘要: 随着人们对隐私和监管合规性的日益关注，机器遗忘的概念变得突出起来，旨在有选择地忘记或擦除来自训练模型的特定学习信息。针对这一关键需求，我们引入了一种新的方法，称为遗忘攻击和重置(ARU)。该算法利用精心设计的对抗性噪声来生成参数掩码，有效地重置某些参数并使其无法学习。ARU在两个人脸机器遗忘基准数据集MUFAC和MUCAC上的性能优于当前最先进的结果。特别是，我们给出了攻击和掩蔽所涉及的步骤，这些步骤策略性地过滤和重新初始化偏向遗忘集的网络参数。我们的工作代表着在通过参数重新初始化来呈现深度学习模型无法利用的数据方面取得了重大进展，这是通过利用对抗性噪声来制作掩码实现的。



## **47. A GAN-based data poisoning framework against anomaly detection in vertical federated learning**

垂直联合学习中基于GAN的抗异常检测数据中毒框架 cs.LG

6 pages, 7 figures. This work has been submitted to the IEEE for  possible publication. Copyright may be transferred without notice, after  which this version may no longer be accessible

**SubmitDate**: 2024-01-17    [abs](http://arxiv.org/abs/2401.08984v1) [paper-pdf](http://arxiv.org/pdf/2401.08984v1)

**Authors**: Xiaolin Chen, Daoguang Zan, Wei Li, Bei Guan, Yongji Wang

**Abstract**: In vertical federated learning (VFL), commercial entities collaboratively train a model while preserving data privacy. However, a malicious participant's poisoning attack may degrade the performance of this collaborative model. The main challenge in achieving the poisoning attack is the absence of access to the server-side top model, leaving the malicious participant without a clear target model. To address this challenge, we introduce an innovative end-to-end poisoning framework P-GAN. Specifically, the malicious participant initially employs semi-supervised learning to train a surrogate target model. Subsequently, this participant employs a GAN-based method to produce adversarial perturbations to degrade the surrogate target model's performance. Finally, the generator is obtained and tailored for VFL poisoning. Besides, we develop an anomaly detection algorithm based on a deep auto-encoder (DAE), offering a robust defense mechanism to VFL scenarios. Through extensive experiments, we evaluate the efficacy of P-GAN and DAE, and further analyze the factors that influence their performance.

摘要: 在垂直联合学习(VFL)中，商业实体在保护数据隐私的同时协作训练模型。然而，恶意参与者的中毒攻击可能会降低该协作模型的性能。实现中毒攻击的主要挑战是无法访问服务器端的顶层模型，使恶意参与者没有明确的目标模型。为了应对这一挑战，我们引入了一个创新的端到端中毒框架P-GaN。具体地说，恶意参与者最初使用半监督学习来训练代理目标模型。随后，该参与者使用基于遗传算法的方法来产生对抗性扰动以降低代理目标模型的性能。最后，获得了用于VFL中毒的发生器并进行了定制。此外，我们还开发了一种基于深度自动编码器的异常检测算法，为VFL场景提供了一种健壮的防御机制。通过大量的实验，我们对P-GaN和DAE的性能进行了评估，并进一步分析了影响它们性能的因素。



## **48. RandOhm: Mitigating Impedance Side-channel Attacks using Randomized Circuit Configurations**

RandOhm：使用随机化电路配置缓解阻抗旁通道攻击 cs.CR

**SubmitDate**: 2024-01-17    [abs](http://arxiv.org/abs/2401.08925v1) [paper-pdf](http://arxiv.org/pdf/2401.08925v1)

**Authors**: Saleh Khalaj Monfared, Domenic Forte, Shahin Tajik

**Abstract**: Physical side-channel attacks can compromise the security of integrated circuits. Most of the physical side-channel attacks (e.g., power or electromagnetic) exploit the dynamic behavior of a chip, typically manifesting as changes in current consumption or voltage fluctuations where algorithmic countermeasures, such as masking, can effectively mitigate the attacks. However, as demonstrated recently, these mitigation techniques are not entirely effective against backscattered side-channel attacks such as impedance analysis. In the case of an impedance attack, an adversary exploits the data-dependent impedance variations of chip power delivery network (PDN) to extract secret information. In this work, we introduce RandOhm, which exploits moving target defense (MTD) strategy based on partial reconfiguration of mainstream FPGAs, to defend against impedance side-channel attacks. We demonstrate that the information leakage through the PDN impedance could be reduced via run-time reconfiguration of the secret-sensitive parts of the circuitry. Hence, by constantly randomizing the placement and routing of the circuit, one can decorrelate the data-dependent computation from the impedance value. To validate our claims, we present a systematic approach equipped with two different partial reconfiguration strategies on implementations of the AES cipher realized on 28-nm FPGAs. We investigate the overhead of our mitigation in terms of delay and performance and provide security analysis by performing non-profiled and profiled impedance analysis attacks against these implementations to demonstrate the resiliency of our approach.

摘要: 物理侧通道攻击可能会危及集成电路的安全性。大多数物理侧通道攻击(例如，功率或电磁)利用芯片的动态行为，通常表现为电流消耗或电压波动的变化，其中算法对策，如掩蔽，可以有效地缓解攻击。然而，正如最近所证明的那样，这些缓解技术并不能完全有效地对抗诸如阻抗分析之类的反向散射侧信道攻击。在阻抗攻击的情况下，敌手利用芯片功率传输网络(PDN)与数据相关的阻抗变化来提取秘密信息。在这项工作中，我们引入了RandOhm，它利用基于主流现场可编程门阵列部分重构的移动目标防御(MTD)策略来防御阻抗旁通道攻击。我们证明，通过PDN阻抗的信息泄漏可以通过运行时重新配置电路的秘密敏感部分来减少。因此，通过不断地随机化电路的布局和布线，可以将依赖于数据的计算与阻抗值分离。为了验证我们的主张，我们提出了一种系统的方法，该方法配备了两种不同的部分重构策略，以实现在28 nm FPGA上实现的AES密码。我们在延迟和性能方面调查了缓解的开销，并通过对这些实施执行非配置文件和配置文件阻抗分析攻击来提供安全分析，以展示我们方法的弹性。



## **49. PPR: Enhancing Dodging Attacks while Maintaining Impersonation Attacks on Face Recognition Systems**

PPR：增强对人脸识别系统的躲避攻击，同时保持模仿攻击 cs.CV

**SubmitDate**: 2024-01-17    [abs](http://arxiv.org/abs/2401.08903v1) [paper-pdf](http://arxiv.org/pdf/2401.08903v1)

**Authors**: Fengfan Zhou, Heifei Ling

**Abstract**: Adversarial Attacks on Face Recognition (FR) encompass two types: impersonation attacks and evasion attacks. We observe that achieving a successful impersonation attack on FR does not necessarily ensure a successful dodging attack on FR in the black-box setting. Introducing a novel attack method named Pre-training Pruning Restoration Attack (PPR), we aim to enhance the performance of dodging attacks whilst avoiding the degradation of impersonation attacks. Our method employs adversarial example pruning, enabling a portion of adversarial perturbations to be set to zero, while tending to maintain the attack performance. By utilizing adversarial example pruning, we can prune the pre-trained adversarial examples and selectively free up certain adversarial perturbations. Thereafter, we embed adversarial perturbations in the pruned area, which enhances the dodging performance of the adversarial face examples. The effectiveness of our proposed attack method is demonstrated through our experimental results, showcasing its superior performance.

摘要: 针对人脸识别(FR)的敌意攻击包括两种类型：模仿攻击和逃避攻击。我们观察到，在黑盒环境下，成功地实现对FR的模仿攻击并不一定确保对FR的成功躲避攻击。引入一种新的攻击方法--预训练剪枝恢复攻击(PPR)，旨在提高躲避攻击的性能，同时避免冒充攻击的降级。该方法采用对抗性样本剪枝，在保持攻击性能的同时，使一部分对抗性扰动被设置为零。通过利用对抗性实例修剪，我们可以修剪预先训练的对抗性实例，并选择性地释放某些对抗性扰动。之后，我们在剪枝区域嵌入对抗性扰动，提高了对抗性人脸样例的躲避性能。实验结果表明，本文提出的攻击方法是有效的，表现出了优越的性能。



## **50. Whispering Pixels: Exploiting Uninitialized Register Accesses in Modern GPUs**

低语像素：利用现代GPU中未初始化的寄存器访问 cs.CR

**SubmitDate**: 2024-01-16    [abs](http://arxiv.org/abs/2401.08881v1) [paper-pdf](http://arxiv.org/pdf/2401.08881v1)

**Authors**: Frederik Dermot Pustelnik, Xhani Marvin Saß, Jean-Pierre Seifert

**Abstract**: Graphic Processing Units (GPUs) have transcended their traditional use-case of rendering graphics and nowadays also serve as a powerful platform for accelerating ubiquitous, non-graphical rendering tasks. One prominent task is inference of neural networks, which process vast amounts of personal data, such as audio, text or images. Thus, GPUs became integral components for handling vast amounts of potentially confidential data, which has awakened the interest of security researchers. This lead to the discovery of various vulnerabilities in GPUs in recent years. In this paper, we uncover yet another vulnerability class in GPUs: We found that some GPU implementations lack proper register initialization routines before shader execution, leading to unintended register content leakage of previously executed shader kernels. We showcase the existence of the aforementioned vulnerability on products of 3 major vendors - Apple, NVIDIA and Qualcomm. The vulnerability poses unique challenges to an adversary due to opaque scheduling and register remapping algorithms present in the GPU firmware, complicating the reconstruction of leaked data. In order to illustrate the real-world impact of this flaw, we showcase how these challenges can be solved for attacking various workloads on the GPU. First, we showcase how uninitialized registers leak arbitrary pixel data processed by fragment shaders. We further implement information leakage attacks on intermediate data of Convolutional Neural Networks (CNNs) and present the attack's capability to leak and reconstruct the output of Large Language Models (LLMs).

摘要: 图形处理单元(GPU)已经超越了渲染图形的传统用例，如今也成为加速无处不在的非图形渲染任务的强大平台。一项突出的任务是神经网络的推理，它处理大量的个人数据，如音频、文本或图像。因此，GPU成为处理海量潜在机密数据的不可或缺的组件，这唤醒了安全研究人员的兴趣。这导致了近年来GPU中各种漏洞的发现。在本文中，我们发现了GPU中的另一个漏洞类别：我们发现一些GPU实现在着色器执行之前缺乏适当的寄存器初始化例程，导致先前执行的着色器内核的意外寄存器内容泄漏。我们展示了3家主要供应商的产品上存在上述漏洞-苹果、NVIDIA和高通。由于GPU固件中存在不透明的调度和寄存器重新映射算法，该漏洞对对手构成了独特的挑战，使泄漏数据的重建复杂化。为了说明该漏洞的实际影响，我们展示了如何解决这些挑战来攻击GPU上的各种工作负载。首先，我们展示了未初始化的寄存器如何泄漏由片段着色器处理的任意像素数据。在此基础上，对卷积神经网络(CNN)的中间数据进行了信息泄漏攻击，并给出了该攻击对大语言模型(LLM)输出的泄漏和重构能力。



