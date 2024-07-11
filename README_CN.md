# Latest Adversarial Attack Papers
**update at 2024-07-11 16:21:59**

翻译来自 https://cloud.tencent.com/document/product/551/15619

## **1. Adversarial Robustness Limits via Scaling-Law and Human-Alignment Studies**

通过比例定律和人际关系研究的对抗稳健性限制 cs.LG

ICML 2024

**SubmitDate**: 2024-07-10    [abs](http://arxiv.org/abs/2404.09349v2) [paper-pdf](http://arxiv.org/pdf/2404.09349v2)

**Authors**: Brian R. Bartoldson, James Diffenderfer, Konstantinos Parasyris, Bhavya Kailkhura

**Abstract**: This paper revisits the simple, long-studied, yet still unsolved problem of making image classifiers robust to imperceptible perturbations. Taking CIFAR10 as an example, SOTA clean accuracy is about $100$%, but SOTA robustness to $\ell_{\infty}$-norm bounded perturbations barely exceeds $70$%. To understand this gap, we analyze how model size, dataset size, and synthetic data quality affect robustness by developing the first scaling laws for adversarial training. Our scaling laws reveal inefficiencies in prior art and provide actionable feedback to advance the field. For instance, we discovered that SOTA methods diverge notably from compute-optimal setups, using excess compute for their level of robustness. Leveraging a compute-efficient setup, we surpass the prior SOTA with $20$% ($70$%) fewer training (inference) FLOPs. We trained various compute-efficient models, with our best achieving $74$% AutoAttack accuracy ($+3$% gain). However, our scaling laws also predict robustness slowly grows then plateaus at $90$%: dwarfing our new SOTA by scaling is impractical, and perfect robustness is impossible. To better understand this predicted limit, we carry out a small-scale human evaluation on the AutoAttack data that fools our top-performing model. Concerningly, we estimate that human performance also plateaus near $90$%, which we show to be attributable to $\ell_{\infty}$-constrained attacks' generation of invalid images not consistent with their original labels. Having characterized limiting roadblocks, we outline promising paths for future research.

摘要: 本文回顾了一个简单、研究已久但仍未解决的问题，即使图像分类器对不可察觉的扰动具有健壮性。以CIFAR10为例，SOTA的清洁精度约为$100$%，但对$\ell_{inty}$-范数有界摄动的鲁棒性仅略高于$70$%。为了理解这一差距，我们分析了模型大小、数据集大小和合成数据质量如何通过开发用于对抗性训练的第一个缩放规则来影响稳健性。我们的比例法则揭示了现有技术中的低效，并提供了可操作的反馈来推动该领域的发展。例如，我们发现SOTA方法与计算最优设置明显不同，使用过量计算作为其健壮性级别。利用高效计算的设置，我们比以前的SOTA少了20美元%(70美元%)的培训(推理)失败。我们训练了各种计算效率高的模型，最大限度地达到了$74$%的AutoAttack精度($+3$%的收益)。然而，我们的定标法则也预测稳健性在90美元时缓慢增长然后停滞不前：通过定标来使我们的新SOTA相形见绌是不切实际的，而且完美的稳健性是不可能的。为了更好地理解这一预测极限，我们对AutoAttack数据进行了小规模的人工评估，该评估愚弄了我们的最佳模型。令人担忧的是，我们估计人类的性能也停滞不前近90$%，我们表明这归因于$受限攻击生成的无效图像与其原始标签不一致。在描述了限制障碍的特征之后，我们概述了未来研究的有希望的道路。



## **2. Boosting Transferability in Vision-Language Attacks via Diversification along the Intersection Region of Adversarial Trajectory**

通过沿着对抗轨迹交叉区域的多样化来提高视觉语言攻击的可移植性 cs.CV

ECCV2024. Code is available at  https://github.com/SensenGao/VLPTransferAttack

**SubmitDate**: 2024-07-10    [abs](http://arxiv.org/abs/2403.12445v2) [paper-pdf](http://arxiv.org/pdf/2403.12445v2)

**Authors**: Sensen Gao, Xiaojun Jia, Xuhong Ren, Ivor Tsang, Qing Guo

**Abstract**: Vision-language pre-training (VLP) models exhibit remarkable capabilities in comprehending both images and text, yet they remain susceptible to multimodal adversarial examples (AEs).Strengthening attacks and uncovering vulnerabilities, especially common issues in VLP models (e.g., high transferable AEs), can advance reliable and practical VLP models. A recent work (i.e., Set-level guidance attack) indicates that augmenting image-text pairs to increase AE diversity along the optimization path enhances the transferability of adversarial examples significantly. However, this approach predominantly emphasizes diversity around the online adversarial examples (i.e., AEs in the optimization period), leading to the risk of overfitting the victim model and affecting the transferability.In this study, we posit that the diversity of adversarial examples towards the clean input and online AEs are both pivotal for enhancing transferability across VLP models. Consequently, we propose using diversification along the intersection region of adversarial trajectory to expand the diversity of AEs.To fully leverage the interaction between modalities, we introduce text-guided adversarial example selection during optimization. Furthermore, to further mitigate the potential overfitting, we direct the adversarial text deviating from the last intersection region along the optimization path, rather than adversarial images as in existing methods.Extensive experiments affirm the effectiveness of our method in improving transferability across various VLP models and downstream vision-and-language tasks.

摘要: 视觉语言预训练(VLP)模型在理解图像和文本方面表现出卓越的能力，但它们仍然容易受到多模式对抗性例子(AE)的影响，加强攻击和发现漏洞，特别是VLP模型中的常见问题(如高可转移性AEs)，可以促进可靠和实用的VLP模型。最近的一项工作(即集合级制导攻击)表明，增加图文对以增加优化路径上的声发射多样性显著地提高了对抗性例子的可转移性。然而，这种方法主要强调在线对抗性实例的多样性(即处于优化期的AEs)，导致受害者模型过度拟合的风险，并影响模型的可转移性。为了充分利用模式间的交互作用，我们在优化过程中引入了文本引导的对抗性范例选择。此外，为了进一步缓解潜在的过度匹配，我们沿着优化路径引导偏离最后一个交集区域的对抗性文本，而不是现有方法中的对抗性图像。大量实验证实了该方法在提高跨各种VLP模型和下游视觉语言任务的可转移性方面的有效性。



## **3. Targeted Augmented Data for Audio Deepfake Detection**

用于音频Deepfake检测的定向增强数据 cs.SD

Accepted in EUSIPCO 2024

**SubmitDate**: 2024-07-10    [abs](http://arxiv.org/abs/2407.07598v1) [paper-pdf](http://arxiv.org/pdf/2407.07598v1)

**Authors**: Marcella Astrid, Enjie Ghorbel, Djamila Aouada

**Abstract**: The availability of highly convincing audio deepfake generators highlights the need for designing robust audio deepfake detectors. Existing works often rely solely on real and fake data available in the training set, which may lead to overfitting, thereby reducing the robustness to unseen manipulations. To enhance the generalization capabilities of audio deepfake detectors, we propose a novel augmentation method for generating audio pseudo-fakes targeting the decision boundary of the model. Inspired by adversarial attacks, we perturb original real data to synthesize pseudo-fakes with ambiguous prediction probabilities. Comprehensive experiments on two well-known architectures demonstrate that the proposed augmentation contributes to improving the generalization capabilities of these architectures.

摘要: 高度令人信服的音频深度伪造生成器的可用性凸显了设计稳健的音频深度伪造检测器的必要性。现有的作品通常仅依赖于训练集中可用的真实和虚假数据，这可能会导致过度匹配，从而降低对不可见操纵的鲁棒性。为了增强音频深度伪造检测器的概括能力，我们提出了一种新颖的增强方法，用于生成针对模型决策边界的音频伪伪造。受对抗攻击的启发，我们扰乱原始真实数据以合成预测概率模糊的伪假货。对两种知名架构的综合实验表明，所提出的增强有助于提高这些架构的概括能力。



## **4. DistriBlock: Identifying adversarial audio samples by leveraging characteristics of the output distribution**

DistriBlock：通过利用输出分布的特征来识别对抗性音频样本 cs.SD

**SubmitDate**: 2024-07-10    [abs](http://arxiv.org/abs/2305.17000v5) [paper-pdf](http://arxiv.org/pdf/2305.17000v5)

**Authors**: Matías P. Pizarro B., Dorothea Kolossa, Asja Fischer

**Abstract**: Adversarial attacks can mislead automatic speech recognition (ASR) systems into predicting an arbitrary target text, thus posing a clear security threat. To prevent such attacks, we propose DistriBlock, an efficient detection strategy applicable to any ASR system that predicts a probability distribution over output tokens in each time step. We measure a set of characteristics of this distribution: the median, maximum, and minimum over the output probabilities, the entropy of the distribution, as well as the Kullback-Leibler and the Jensen-Shannon divergence with respect to the distributions of the subsequent time step. Then, by leveraging the characteristics observed for both benign and adversarial data, we apply binary classifiers, including simple threshold-based classification, ensembles of such classifiers, and neural networks. Through extensive analysis across different state-of-the-art ASR systems and language data sets, we demonstrate the supreme performance of this approach, with a mean area under the receiver operating characteristic curve for distinguishing target adversarial examples against clean and noisy data of 99% and 97%, respectively. To assess the robustness of our method, we show that adaptive adversarial examples that can circumvent DistriBlock are much noisier, which makes them easier to detect through filtering and creates another avenue for preserving the system's robustness.

摘要: 敌意攻击可以误导自动语音识别(ASR)系统预测任意目标文本，从而构成明显的安全威胁。为了防止此类攻击，我们提出了DistriBlock，这是一种适用于任何ASR系统的有效检测策略，它预测每个时间步输出令牌上的概率分布。我们测量了该分布的一组特征：输出概率的中位数、最大值和最小值，分布的熵，以及关于后续时间步分布的Kullback-Leibler和Jensen-Shannon散度。然后，通过利用对良性数据和恶意数据观察到的特征，我们应用二进制分类器，包括简单的基于阈值的分类、这种分类器的集成和神经网络。通过对不同的ASR系统和语言数据集的广泛分析，我们证明了该方法的最高性能，在干净和有噪声的数据下，接收器操作特征曲线下的平均面积分别为99%和97%。为了评估我们方法的健壮性，我们证明了可以绕过DistriBlock的自适应攻击示例的噪声要大得多，这使得它们更容易通过过滤来检测，并为保持系统的健壮性创造了另一种途径。



## **5. Evaluating the Adversarial Robustness of Retrieval-Based In-Context Learning for Large Language Models**

评估大型语言模型基于检索的上下文学习的对抗鲁棒性 cs.CL

COLM 2024, 29 pages, 6 figures

**SubmitDate**: 2024-07-10    [abs](http://arxiv.org/abs/2405.15984v2) [paper-pdf](http://arxiv.org/pdf/2405.15984v2)

**Authors**: Simon Chi Lok Yu, Jie He, Pasquale Minervini, Jeff Z. Pan

**Abstract**: With the emergence of large language models, such as LLaMA and OpenAI GPT-3, In-Context Learning (ICL) gained significant attention due to its effectiveness and efficiency. However, ICL is very sensitive to the choice, order, and verbaliser used to encode the demonstrations in the prompt. Retrieval-Augmented ICL methods try to address this problem by leveraging retrievers to extract semantically related examples as demonstrations. While this approach yields more accurate results, its robustness against various types of adversarial attacks, including perturbations on test samples, demonstrations, and retrieved data, remains under-explored. Our study reveals that retrieval-augmented models can enhance robustness against test sample attacks, outperforming vanilla ICL with a 4.87% reduction in Attack Success Rate (ASR); however, they exhibit overconfidence in the demonstrations, leading to a 2% increase in ASR for demonstration attacks. Adversarial training can help improve the robustness of ICL methods to adversarial attacks; however, such a training scheme can be too costly in the context of LLMs. As an alternative, we introduce an effective training-free adversarial defence method, DARD, which enriches the example pool with those attacked samples. We show that DARD yields improvements in performance and robustness, achieving a 15% reduction in ASR over the baselines. Code and data are released to encourage further research: https://github.com/simonucl/adv-retreival-icl

摘要: 随着大型语言模型的出现，如Llama和OpenAI GPT-3，情景中学习(ICL)因其有效性和高效性而受到广泛关注。但是，ICL对用于对提示符中的演示进行编码的选择、顺序和形容词非常敏感。检索增强的ICL方法试图通过利用检索器来提取语义相关的示例作为演示来解决这个问题。虽然这种方法可以产生更准确的结果，但它对各种类型的对抗性攻击的稳健性，包括对测试样本、演示和检索数据的扰动，仍然没有得到充分的研究。我们的研究表明，检索增强模型可以增强对测试样本攻击的健壮性，性能优于普通ICL，攻击成功率(ASR)降低4.87%；然而，它们在演示中表现出过度自信，导致演示攻击的ASR提高了2%。对抗性训练可以帮助提高ICL方法对对抗性攻击的稳健性；然而，在LLMS的背景下，这样的训练方案可能代价太高。作为另一种选择，我们引入了一种有效的无需训练的对抗防御方法DARD，它用被攻击的样本丰富了样本库。我们表明，DARD在性能和健壮性方面都有改进，ASR比基准降低了15%。发布代码和数据是为了鼓励进一步的研究：https://github.com/simonucl/adv-retreival-icl



## **6. Invisible Optical Adversarial Stripes on Traffic Sign against Autonomous Vehicles**

针对自动驾驶车辆的交通标志上的隐形光学对抗条纹 cs.CR

**SubmitDate**: 2024-07-10    [abs](http://arxiv.org/abs/2407.07510v1) [paper-pdf](http://arxiv.org/pdf/2407.07510v1)

**Authors**: Dongfang Guo, Yuting Wu, Yimin Dai, Pengfei Zhou, Xin Lou, Rui Tan

**Abstract**: Camera-based computer vision is essential to autonomous vehicle's perception. This paper presents an attack that uses light-emitting diodes and exploits the camera's rolling shutter effect to create adversarial stripes in the captured images to mislead traffic sign recognition. The attack is stealthy because the stripes on the traffic sign are invisible to human. For the attack to be threatening, the recognition results need to be stable over consecutive image frames. To achieve this, we design and implement GhostStripe, an attack system that controls the timing of the modulated light emission to adapt to camera operations and victim vehicle movements. Evaluated on real testbeds, GhostStripe can stably spoof the traffic sign recognition results for up to 94\% of frames to a wrong class when the victim vehicle passes the road section. In reality, such attack effect may fool victim vehicles into life-threatening incidents. We discuss the countermeasures at the levels of camera sensor, perception model, and autonomous driving system.

摘要: 基于摄像头的计算机视觉对于自动驾驶汽车的感知是必不可少的。本文提出了一种利用发光二极管和利用摄像机的滚动快门效应在捕获的图像中产生对抗性条纹来误导交通标志识别的攻击方法。这次袭击是隐形的，因为交通标志上的条纹是人类看不见的。为了使攻击具有威胁性，识别结果需要在连续的图像帧上保持稳定。为了实现这一点，我们设计并实现了Ghost Strike，这是一个攻击系统，它控制调制光发射的时间，以适应相机操作和受害者车辆的移动。在真实的测试平台上进行了测试，当受害者车辆通过路段时，Ghost Strike可以稳定地将高达94%的帧的交通标志识别结果伪造到错误的类别。在现实中，这种攻击效果可能会欺骗受害者车辆发生危及生命的事件。分别从摄像机传感器、感知模型、自动驾驶系统三个层面探讨了对策。



## **7. Formal Verification of Object Detection**

对象检测的形式化验证 cs.CV

**SubmitDate**: 2024-07-10    [abs](http://arxiv.org/abs/2407.01295v3) [paper-pdf](http://arxiv.org/pdf/2407.01295v3)

**Authors**: Avraham Raviv, Yizhak Y. Elboher, Michelle Aluf-Medina, Yael Leibovich Weiss, Omer Cohen, Roy Assa, Guy Katz, Hillel Kugler

**Abstract**: Deep Neural Networks (DNNs) are ubiquitous in real-world applications, yet they remain vulnerable to errors and adversarial attacks. This work tackles the challenge of applying formal verification to ensure the safety of computer vision models, extending verification beyond image classification to object detection. We propose a general formulation for certifying the robustness of object detection models using formal verification and outline implementation strategies compatible with state-of-the-art verification tools. Our approach enables the application of these tools, originally designed for verifying classification models, to object detection. We define various attacks for object detection, illustrating the diverse ways adversarial inputs can compromise neural network outputs. Our experiments, conducted on several common datasets and networks, reveal potential errors in object detection models, highlighting system vulnerabilities and emphasizing the need for expanding formal verification to these new domains. This work paves the way for further research in integrating formal verification across a broader range of computer vision applications.

摘要: 深度神经网络(DNN)在实际应用中无处不在，但它们仍然容易受到错误和对手攻击。这项工作解决了应用形式化验证来确保计算机视觉模型的安全性的挑战，将验证从图像分类扩展到目标检测。我们提出了使用形式化验证来证明目标检测模型的健壮性的一般公式，并概述了与最先进的验证工具兼容的实现策略。我们的方法使得这些最初设计用于验证分类模型的工具能够应用于目标检测。我们定义了用于目标检测的各种攻击，说明了敌意输入可以损害神经网络输出的不同方式。我们在几个常见的数据集和网络上进行的实验，揭示了对象检测模型中的潜在错误，突出了系统漏洞，并强调了将正式验证扩展到这些新领域的必要性。这项工作为在更广泛的计算机视觉应用中整合形式验证的进一步研究铺平了道路。



## **8. A Survey of Attacks on Large Vision-Language Models: Resources, Advances, and Future Trends**

大型视觉语言模型攻击调查：资源、进展和未来趋势 cs.CV

**SubmitDate**: 2024-07-10    [abs](http://arxiv.org/abs/2407.07403v1) [paper-pdf](http://arxiv.org/pdf/2407.07403v1)

**Authors**: Daizong Liu, Mingyu Yang, Xiaoye Qu, Pan Zhou, Wei Hu, Yu Cheng

**Abstract**: With the significant development of large models in recent years, Large Vision-Language Models (LVLMs) have demonstrated remarkable capabilities across a wide range of multimodal understanding and reasoning tasks. Compared to traditional Large Language Models (LLMs), LVLMs present great potential and challenges due to its closer proximity to the multi-resource real-world applications and the complexity of multi-modal processing. However, the vulnerability of LVLMs is relatively underexplored, posing potential security risks in daily usage. In this paper, we provide a comprehensive review of the various forms of existing LVLM attacks. Specifically, we first introduce the background of attacks targeting LVLMs, including the attack preliminary, attack challenges, and attack resources. Then, we systematically review the development of LVLM attack methods, such as adversarial attacks that manipulate model outputs, jailbreak attacks that exploit model vulnerabilities for unauthorized actions, prompt injection attacks that engineer the prompt type and pattern, and data poisoning that affects model training. Finally, we discuss promising research directions in the future. We believe that our survey provides insights into the current landscape of LVLM vulnerabilities, inspiring more researchers to explore and mitigate potential safety issues in LVLM developments. The latest papers on LVLM attacks are continuously collected in https://github.com/liudaizong/Awesome-LVLM-Attack.

摘要: 近年来，随着大型模型的显著发展，大型视觉语言模型在广泛的多通道理解和推理任务中表现出了卓越的能力。与传统的大语言模型相比，大语言模型因其更接近多资源的实际应用和多模式处理的复杂性而显示出巨大的潜力和挑战。然而，LVLMS的脆弱性相对较少，在日常使用中存在潜在的安全风险。在本文中，我们对现有的各种形式的LVLM攻击进行了全面的回顾。具体地说，我们首先介绍了针对LVLMS的攻击背景，包括攻击准备、攻击挑战和攻击资源。然后，我们系统地回顾了LVLM攻击方法的发展，如操纵模型输出的对抗性攻击，利用模型漏洞进行未经授权操作的越狱攻击，设计提示类型和模式的提示注入攻击，以及影响模型训练的数据中毒。最后，我们讨论了未来的研究方向。我们相信，我们的调查提供了对LVLM漏洞现状的洞察，激励更多的研究人员探索和缓解LVLM开发中的潜在安全问题。有关LVLm攻击的最新论文在https://github.com/liudaizong/Awesome-LVLM-Attack.上不断收集



## **9. Marlin: Knowledge-Driven Analysis of Provenance Graphs for Efficient and Robust Detection of Cyber Attacks**

马林：知识驱动的源源图分析，以高效、稳健地检测网络攻击 cs.CR

**SubmitDate**: 2024-07-10    [abs](http://arxiv.org/abs/2403.12541v2) [paper-pdf](http://arxiv.org/pdf/2403.12541v2)

**Authors**: Zhenyuan Li, Yangyang Wei, Xiangmin Shen, Lingzhi Wang, Yan Chen, Haitao Xu, Shouling Ji, Fan Zhang, Liang Hou, Wenmao Liu, Xuhong Zhang, Jianwei Ying

**Abstract**: Recent research in both academia and industry has validated the effectiveness of provenance graph-based detection for advanced cyber attack detection and investigation. However, analyzing large-scale provenance graphs often results in substantial overhead. To improve performance, existing detection systems implement various optimization strategies. Yet, as several recent studies suggest, these strategies could lose necessary context information and be vulnerable to evasions. Designing a detection system that is efficient and robust against adversarial attacks is an open problem. We introduce Marlin, which approaches cyber attack detection through real-time provenance graph alignment.By leveraging query graphs embedded with attack knowledge, Marlin can efficiently identify entities and events within provenance graphs, embedding targeted analysis and significantly narrowing the search space. Moreover, we incorporate our graph alignment algorithm into a tag propagation-based schema to eliminate the need for storing and reprocessing raw logs. This design significantly reduces in-memory storage requirements and minimizes data processing overhead. As a result, it enables real-time graph alignment while preserving essential context information, thereby enhancing the robustness of cyber attack detection. Moreover, Marlin allows analysts to customize attack query graphs flexibly to detect extended attacks and provide interpretable detection results. We conduct experimental evaluations on two large-scale public datasets containing 257.42 GB of logs and 12 query graphs of varying sizes, covering multiple attack techniques and scenarios. The results show that Marlin can process 137K events per second while accurately identifying 120 subgraphs with 31 confirmed attacks, along with only 1 false positive, demonstrating its efficiency and accuracy in handling massive data.

摘要: 最近学术界和工业界的研究都证实了基于起源图的检测对于高级网络攻击检测和调查的有效性。然而，分析大规模的种源图表往往会产生相当大的开销。为了提高性能，现有的检测系统采用了各种优化策略。然而，正如最近的几项研究表明的那样，这些策略可能会失去必要的背景信息，并容易受到规避。设计一个对敌方攻击高效且健壮的检测系统是一个悬而未决的问题。介绍了Marlin算法，该算法通过对源图进行实时比对来实现网络攻击的检测，利用嵌入攻击知识的查询图，能够有效地识别源图中的实体和事件，嵌入针对性的分析，大大缩小了搜索空间。此外，我们将我们的图对齐算法整合到基于标记传播的模式中，以消除存储和重新处理原始日志的需要。这种设计显著降低了内存存储需求，并最大限度地减少了数据处理开销。因此，它能够在保留基本上下文信息的同时实现实时图形对齐，从而增强网络攻击检测的健壮性。此外，Marlin允许分析人员灵活地定制攻击查询图，以检测扩展的攻击并提供可解释的检测结果。我们在两个包含257.42 GB日志和12个不同大小的查询图的大规模公共数据集上进行了实验评估，涵盖了多种攻击技术和场景。实验结果表明，Marlin能够在每秒处理137K事件的同时，准确识别出120个子图中31个已确认的攻击，并且只有1个误报，证明了其在处理海量数据时的效率和准确性。



## **10. Characterizing Encrypted Application Traffic through Cellular Radio Interface Protocol**

通过蜂窝无线电接口协议描述加密应用流量 cs.NI

9 pages, 8 figures, 2 tables. This paper has been accepted for  publication by the 21st IEEE International Conference on Mobile Ad-Hoc and  Smart Systems (MASS 2024)

**SubmitDate**: 2024-07-10    [abs](http://arxiv.org/abs/2407.07361v1) [paper-pdf](http://arxiv.org/pdf/2407.07361v1)

**Authors**: Md Ruman Islam, Raja Hasnain Anwar, Spyridon Mastorakis, Muhammad Taqi Raza

**Abstract**: Modern applications are end-to-end encrypted to prevent data from being read or secretly modified. 5G tech nology provides ubiquitous access to these applications without compromising the application-specific performance and latency goals. In this paper, we empirically demonstrate that 5G radio communication becomes the side channel to precisely infer the user's applications in real-time. The key idea lies in observing the 5G physical and MAC layer interactions over time that reveal the application's behavior. The MAC layer receives the data from the application and requests the network to assign the radio resource blocks. The network assigns the radio resources as per application requirements, such as priority, Quality of Service (QoS) needs, amount of data to be transmitted, and buffer size. The adversary can passively observe the radio resources to fingerprint the applications. We empirically demonstrate this attack by considering four different categories of applications: online shopping, voice/video conferencing, video streaming, and Over-The-Top (OTT) media platforms. Finally, we have also demonstrated that an attacker can differentiate various types of applications in real-time within each category.

摘要: 现代应用程序是端到端加密的，以防止数据被读取或秘密修改。5G技术提供了对这些应用的无处不在的访问，而不会影响特定于应用的性能和延迟目标。在本文中，我们实证地论证了5G无线通信成为实时准确推断用户应用的辅助通道。关键思想在于观察5G物理层和MAC层随时间的交互，以揭示应用的行为。MAC层从应用程序接收数据，并请求网络分配无线电资源块。网络根据诸如优先级、服务质量(Qos)需求、要传输的数据量和缓冲区大小等应用需求来分配无线电资源。敌手可以被动地观察无线电资源来识别应用程序。我们考虑了四种不同类别的应用程序：在线购物、语音/视频会议、视频流和Over-the-Top(OTT)媒体平台，对这一攻击进行了实证演示。最后，我们还演示了攻击者可以在每个类别中实时区分各种类型的应用程序。



## **11. The Quantum Imitation Game: Reverse Engineering of Quantum Machine Learning Models**

量子模仿游戏：量子机器学习模型的反向工程 quant-ph

10 pages, 12 figures

**SubmitDate**: 2024-07-09    [abs](http://arxiv.org/abs/2407.07237v1) [paper-pdf](http://arxiv.org/pdf/2407.07237v1)

**Authors**: Archisman Ghosh, Swaroop Ghosh

**Abstract**: Quantum Machine Learning (QML) amalgamates quantum computing paradigms with machine learning models, providing significant prospects for solving complex problems. However, with the expansion of numerous third-party vendors in the Noisy Intermediate-Scale Quantum (NISQ) era of quantum computing, the security of QML models is of prime importance, particularly against reverse engineering, which could expose trained parameters and algorithms of the models. We assume the untrusted quantum cloud provider is an adversary having white-box access to the transpiled user-designed trained QML model during inference. Reverse engineering (RE) to extract the pre-transpiled QML circuit will enable re-transpilation and usage of the model for various hardware with completely different native gate sets and even different qubit technology. Such flexibility may not be obtained from the transpiled circuit which is tied to a particular hardware and qubit technology. The information about the number of parameters, and optimized values can allow further training of the QML model to alter the QML model, tamper with the watermark, and/or embed their own watermark or refine the model for other purposes. In this first effort to investigate the RE of QML circuits, we perform RE and compare the training accuracy of original and reverse-engineered Quantum Neural Networks (QNNs) of various sizes. We note that multi-qubit classifiers can be reverse-engineered under specific conditions with a mean error of order 1e-2 in a reasonable time. We also propose adding dummy fixed parametric gates in the QML models to increase the RE overhead for defense. For instance, adding 2 dummy qubits and 2 layers increases the overhead by ~1.76 times for a classifier with 2 qubits and 3 layers with a performance overhead of less than 9%. We note that RE is a very powerful attack model which warrants further efforts on defenses.

摘要: 量子机器学习(QML)融合了量子计算范式和机器学习模型，为解决复杂问题提供了重要的前景。然而，在喧嚣的中间尺度量子计算(NISQ)时代，随着众多第三方供应商的扩张，QML模型的安全性至关重要，特别是在对抗逆向工程时，逆向工程可能会暴露模型的训练参数和算法。我们假设不可信的量子云提供商是一个对手，在推理过程中可以通过白盒访问用户设计的经过训练的QML模型。逆向工程(RE)提取预转换的QML电路将使模型能够重新转置并用于具有完全不同的本机门设置甚至不同的量子比特技术的各种硬件。这种灵活性可能不是从绑定到特定硬件和量子比特技术的分流电路获得的。关于参数数目和最佳值的信息可以允许进一步训练QML模型以改变QML模型、篡改水印、和/或出于其他目的嵌入它们自己的水印或改进模型。在第一次研究QML电路的RE时，我们进行了RE，并比较了不同大小的原始和反向工程量子神经网络(QNN)的训练精度。我们注意到，多量子比特分类器可以在特定条件下进行逆向工程，在合理的时间内，平均误差为1e-2阶。我们还建议在QML模型中增加虚拟固定参数门，以增加防御的RE开销。例如，对于具有2个量子比特和3个层的分类器，添加2个虚拟量子比特和2个层会使开销增加~1.76倍，而性能开销不到9%。我们注意到，RE是一种非常强大的攻击模式，需要在防御上进一步努力。



## **12. Robust Neural Information Retrieval: An Adversarial and Out-of-distribution Perspective**

稳健的神经信息检索：对抗性和非分布性的角度 cs.IR

Survey paper

**SubmitDate**: 2024-07-09    [abs](http://arxiv.org/abs/2407.06992v1) [paper-pdf](http://arxiv.org/pdf/2407.06992v1)

**Authors**: Yu-An Liu, Ruqing Zhang, Jiafeng Guo, Maarten de Rijke, Yixing Fan, Xueqi Cheng

**Abstract**: Recent advances in neural information retrieval (IR) models have significantly enhanced their effectiveness over various IR tasks. The robustness of these models, essential for ensuring their reliability in practice, has also garnered significant attention. With a wide array of research on robust IR being proposed, we believe it is the opportune moment to consolidate the current status, glean insights from existing methodologies, and lay the groundwork for future development. We view the robustness of IR to be a multifaceted concept, emphasizing its necessity against adversarial attacks, out-of-distribution (OOD) scenarios and performance variance. With a focus on adversarial and OOD robustness, we dissect robustness solutions for dense retrieval models (DRMs) and neural ranking models (NRMs), respectively, recognizing them as pivotal components of the neural IR pipeline. We provide an in-depth discussion of existing methods, datasets, and evaluation metrics, shedding light on challenges and future directions in the era of large language models. To the best of our knowledge, this is the first comprehensive survey on the robustness of neural IR models, and we will also be giving our first tutorial presentation at SIGIR 2024 \url{https://sigir2024-robust-information-retrieval.github.io}. Along with the organization of existing work, we introduce a Benchmark for robust IR (BestIR), a heterogeneous evaluation benchmark for robust neural information retrieval, which is publicly available at \url{https://github.com/Davion-Liu/BestIR}. We hope that this study provides useful clues for future research on the robustness of IR models and helps to develop trustworthy search engines \url{https://github.com/Davion-Liu/Awesome-Robustness-in-Information-Retrieval}.

摘要: 神经信息检索(IR)模型的最新进展显著提高了它们在各种IR任务中的有效性。这些模型的稳健性对于确保它们在实践中的可靠性至关重要，也引起了人们的极大关注。随着对稳健IR的广泛研究的提出，我们认为现在是巩固当前状况、从现有方法中收集见解并为未来发展奠定基础的好时机。我们认为信息检索的稳健性是一个多方面的概念，强调了它对对抗攻击、分布外(OOD)场景和性能差异的必要性。以对抗性和面向对象的稳健性为重点，我们分别剖析了密集检索模型(DRM)和神经排名模型(NRM)的稳健性解决方案，将它们识别为神经IR管道的关键组件。我们提供了对现有方法、数据集和评估度量的深入讨论，揭示了大型语言模型时代的挑战和未来方向。据我们所知，这是关于神经IR模型稳健性的第一次全面调查，我们还将在SIGIR2024\url{https://sigir2024-robust-information-retrieval.github.io}.上进行我们的第一次教程演示在组织现有工作的同时，我们还介绍了稳健IR基准(BSTIR)，这是一个用于稳健神经信息检索的异质评估基准，可在\url{https://github.com/Davion-Liu/BestIR}.希望本研究为今后研究信息检索模型的健壮性提供有用的线索，并为开发可信搜索引擎\url{https://github.com/Davion-Liu/Awesome-Robustness-in-Information-Retrieval}.提供帮助



## **13. Does CLIP Know My Face?**

CLIP认识我的脸吗？ cs.LG

Published in the Journal of Artificial Intelligence Research (JAIR)

**SubmitDate**: 2024-07-09    [abs](http://arxiv.org/abs/2209.07341v4) [paper-pdf](http://arxiv.org/pdf/2209.07341v4)

**Authors**: Dominik Hintersdorf, Lukas Struppek, Manuel Brack, Felix Friedrich, Patrick Schramowski, Kristian Kersting

**Abstract**: With the rise of deep learning in various applications, privacy concerns around the protection of training data have become a critical area of research. Whereas prior studies have focused on privacy risks in single-modal models, we introduce a novel method to assess privacy for multi-modal models, specifically vision-language models like CLIP. The proposed Identity Inference Attack (IDIA) reveals whether an individual was included in the training data by querying the model with images of the same person. Letting the model choose from a wide variety of possible text labels, the model reveals whether it recognizes the person and, therefore, was used for training. Our large-scale experiments on CLIP demonstrate that individuals used for training can be identified with very high accuracy. We confirm that the model has learned to associate names with depicted individuals, implying the existence of sensitive information that can be extracted by adversaries. Our results highlight the need for stronger privacy protection in large-scale models and suggest that IDIAs can be used to prove the unauthorized use of data for training and to enforce privacy laws.

摘要: 随着深度学习在各种应用中的兴起，围绕训练数据保护的隐私问题已经成为一个关键的研究领域。鉴于以往的研究主要集中于单通道模型中的隐私风险，我们引入了一种新的方法来评估多通道模型的隐私，特别是像CLIP这样的视觉语言模型。提出的身份推断攻击(IDIA)通过用同一人的图像查询模型来揭示该人是否包括在训练数据中。让模型从各种各样的可能的文本标签中进行选择，该模型显示它是否识别出这个人，因此，它被用于训练。我们在CLIP上的大规模实验表明，用于训练的个体可以非常准确地识别。我们确认，该模型已经学会了将姓名与所描述的个人相关联，这意味着存在可被对手提取的敏感信息。我们的结果强调了在大规模模型中加强隐私保护的必要性，并建议可以使用IDIA来证明未经授权使用数据进行培训和执行隐私法。



## **14. Performance Evaluation of Knowledge Graph Embedding Approaches under Non-adversarial Attacks**

非对抗性攻击下知识图嵌入方法的性能评估 cs.LG

**SubmitDate**: 2024-07-09    [abs](http://arxiv.org/abs/2407.06855v1) [paper-pdf](http://arxiv.org/pdf/2407.06855v1)

**Authors**: Sourabh Kapoor, Arnab Sharma, Michael Röder, Caglar Demir, Axel-Cyrille Ngonga Ngomo

**Abstract**: Knowledge Graph Embedding (KGE) transforms a discrete Knowledge Graph (KG) into a continuous vector space facilitating its use in various AI-driven applications like Semantic Search, Question Answering, or Recommenders. While KGE approaches are effective in these applications, most existing approaches assume that all information in the given KG is correct. This enables attackers to influence the output of these approaches, e.g., by perturbing the input. Consequently, the robustness of such KGE approaches has to be addressed. Recent work focused on adversarial attacks. However, non-adversarial attacks on all attack surfaces of these approaches have not been thoroughly examined. We close this gap by evaluating the impact of non-adversarial attacks on the performance of 5 state-of-the-art KGE algorithms on 5 datasets with respect to attacks on 3 attack surfaces-graph, parameter, and label perturbation. Our evaluation results suggest that label perturbation has a strong effect on the KGE performance, followed by parameter perturbation with a moderate and graph with a low effect.

摘要: 知识图嵌入(KGE)将离散的知识图(KG)转换为连续的向量空间，便于其在语义搜索、问答或推荐器等各种人工智能驱动的应用中的使用。虽然KGE方法在这些应用中是有效的，但大多数现有方法都假设给定KG中的所有信息都是正确的。这使得攻击者能够影响这些方法的输出，例如，通过干扰输入。因此，必须解决这种KGE方法的稳健性问题。最近的工作集中在对抗性攻击上。然而，这些方法的所有攻击面上的非对抗性攻击还没有得到彻底的审查。我们通过评估非对抗性攻击对5种最先进的KGE算法在5个数据集上的性能的影响来缩小这一差距，这些影响涉及3个攻击面-图、参数和标签扰动。我们的评估结果表明，标签扰动对KGE性能的影响很大，其次是参数扰动，影响中等，图的影响较小。



## **15. EvolBA: Evolutionary Boundary Attack under Hard-label Black Box condition**

EvolBA：硬标签黑匣子条件下的进化边界攻击 cs.CV

**SubmitDate**: 2024-07-09    [abs](http://arxiv.org/abs/2407.02248v3) [paper-pdf](http://arxiv.org/pdf/2407.02248v3)

**Authors**: Ayane Tajima, Satoshi Ono

**Abstract**: Research has shown that deep neural networks (DNNs) have vulnerabilities that can lead to the misrecognition of Adversarial Examples (AEs) with specifically designed perturbations. Various adversarial attack methods have been proposed to detect vulnerabilities under hard-label black box (HL-BB) conditions in the absence of loss gradients and confidence scores.However, these methods fall into local solutions because they search only local regions of the search space. Therefore, this study proposes an adversarial attack method named EvolBA to generate AEs using Covariance Matrix Adaptation Evolution Strategy (CMA-ES) under the HL-BB condition, where only a class label predicted by the target DNN model is available. Inspired by formula-driven supervised learning, the proposed method introduces domain-independent operators for the initialization process and a jump that enhances search exploration. Experimental results confirmed that the proposed method could determine AEs with smaller perturbations than previous methods in images where the previous methods have difficulty.

摘要: 研究表明，深度神经网络(DNN)存在漏洞，可能会导致对经过特殊设计的扰动的对抗性示例(AE)的错误识别。针对硬标签黑盒(HL-BB)环境下不存在损失梯度和置信度的漏洞检测问题，提出了多种对抗性攻击方法，但这些方法只搜索搜索空间的局部区域，容易陷入局部解.因此，本文提出了一种基于协方差矩阵自适应进化策略(CMA-ES)的对抗性攻击方法EvolBA，用于在目标DNN模型预测的类别标签不可用的HL-BB条件下生成AEs。受公式驱动的监督学习的启发，该方法在初始化过程中引入了领域无关的算子，并引入了一个跳跃来增强搜索探索。实验结果表明，该方法能够以较小的扰动确定图像中的声学效应，克服了以往方法的不足。



## **16. Learning-Based Difficulty Calibration for Enhanced Membership Inference Attacks**

基于学习的增强型成员推断攻击难度校准 cs.CR

Accepted to IEEE Euro S&P 2024

**SubmitDate**: 2024-07-09    [abs](http://arxiv.org/abs/2401.04929v3) [paper-pdf](http://arxiv.org/pdf/2401.04929v3)

**Authors**: Haonan Shi, Tu Ouyang, An Wang

**Abstract**: Machine learning models, in particular deep neural networks, are currently an integral part of various applications, from healthcare to finance. However, using sensitive data to train these models raises concerns about privacy and security. One method that has emerged to verify if the trained models are privacy-preserving is Membership Inference Attacks (MIA), which allows adversaries to determine whether a specific data point was part of a model's training dataset. While a series of MIAs have been proposed in the literature, only a few can achieve high True Positive Rates (TPR) in the low False Positive Rate (FPR) region (0.01%~1%). This is a crucial factor to consider for an MIA to be practically useful in real-world settings. In this paper, we present a novel approach to MIA that is aimed at significantly improving TPR at low FPRs. Our method, named learning-based difficulty calibration for MIA(LDC-MIA), characterizes data records by their hardness levels using a neural network classifier to determine membership. The experiment results show that LDC-MIA can improve TPR at low FPR by up to 4x compared to the other difficulty calibration based MIAs. It also has the highest Area Under ROC curve (AUC) across all datasets. Our method's cost is comparable with most of the existing MIAs, but is orders of magnitude more efficient than one of the state-of-the-art methods, LiRA, while achieving similar performance.

摘要: 机器学习模型，特别是深度神经网络，目前是从医疗保健到金融的各种应用程序的组成部分。然而，使用敏感数据来训练这些模型会引发对隐私和安全的担忧。出现的一种验证训练模型是否保护隐私的方法是成员推理攻击(MIA)，它允许对手确定特定数据点是否属于模型训练数据集的一部分。虽然文献中已经提出了一系列的MIA，但只有少数几个MIA能在低假阳性率(FPR)区域(0.01%~1%)获得高的真阳性率(TPR)。要使MIA在实际环境中发挥实际作用，这是需要考虑的关键因素。在本文中，我们提出了一种新的MIA方法，旨在显著改善低FPR下的TPR。我们的方法，称为基于学习的MIA难度校准(LDC-MIA)，使用神经网络分类器来确定成员身份，根据数据记录的硬度来表征数据记录。实验结果表明，与其他基于难度校正的MIA相比，LDC-MIA可以在较低的误码率下将TPR提高4倍。在所有数据集中，它也具有最高的ROC曲线下面积(AUC)。我们的方法的成本与大多数现有的MIA相当，但效率比最先进的方法之一LIRA高出数量级，同时实现了类似的性能。



## **17. A Hybrid Training-time and Run-time Defense Against Adversarial Attacks in Modulation Classification**

调制分类中训练时和运行时混合防御对抗攻击 cs.AI

Published in IEEE Wireless Communications Letters, vol. 11, no. 6,  pp. 1161-1165, June 2022

**SubmitDate**: 2024-07-09    [abs](http://arxiv.org/abs/2407.06807v1) [paper-pdf](http://arxiv.org/pdf/2407.06807v1)

**Authors**: Lu Zhang, Sangarapillai Lambotharan, Gan Zheng, Guisheng Liao, Ambra Demontis, Fabio Roli

**Abstract**: Motivated by the superior performance of deep learning in many applications including computer vision and natural language processing, several recent studies have focused on applying deep neural network for devising future generations of wireless networks. However, several recent works have pointed out that imperceptible and carefully designed adversarial examples (attacks) can significantly deteriorate the classification accuracy. In this paper, we investigate a defense mechanism based on both training-time and run-time defense techniques for protecting machine learning-based radio signal (modulation) classification against adversarial attacks. The training-time defense consists of adversarial training and label smoothing, while the run-time defense employs a support vector machine-based neural rejection (NR). Considering a white-box scenario and real datasets, we demonstrate that our proposed techniques outperform existing state-of-the-art technologies.

摘要: 受深度学习在计算机视觉和自然语言处理等许多应用中的卓越性能的激励，最近的几项研究专注于应用深度神经网络来设计未来几代无线网络。然而，最近的几篇作品指出，难以察觉且精心设计的对抗性示例（攻击）可能会显着降低分类准确性。本文研究了一种基于训练时和运行时防御技术的防御机制，用于保护基于机器学习的无线电信号（调制）分类免受对抗性攻击。训练时防御由对抗训练和标签平滑组成，而运行时防御则采用基于支持载体机的神经拒绝（NR）。考虑到白盒场景和真实数据集，我们证明我们提出的技术优于现有的最先进技术。



## **18. AdaNCA: Neural Cellular Automata As Adaptors For More Robust Vision Transformer**

AdaNCA：神经元胞自动机作为更稳健的视觉Transformer的适配器 cs.CV

26 pages, 11 figures

**SubmitDate**: 2024-07-09    [abs](http://arxiv.org/abs/2406.08298v4) [paper-pdf](http://arxiv.org/pdf/2406.08298v4)

**Authors**: Yitao Xu, Tong Zhang, Sabine Süsstrunk

**Abstract**: Vision Transformers (ViTs) have demonstrated remarkable performance in image classification tasks, particularly when equipped with local information via region attention or convolutions. While such architectures improve the feature aggregation from different granularities, they often fail to contribute to the robustness of the networks. Neural Cellular Automata (NCA) enables the modeling of global cell representations through local interactions, with its training strategies and architecture design conferring strong generalization ability and robustness against noisy inputs. In this paper, we propose Adaptor Neural Cellular Automata (AdaNCA) for Vision Transformer that uses NCA as plug-in-play adaptors between ViT layers, enhancing ViT's performance and robustness against adversarial samples as well as out-of-distribution inputs. To overcome the large computational overhead of standard NCAs, we propose Dynamic Interaction for more efficient interaction learning. Furthermore, we develop an algorithm for identifying the most effective insertion points for AdaNCA based on our analysis of AdaNCA placement and robustness improvement. With less than a 3% increase in parameters, AdaNCA contributes to more than 10% absolute improvement in accuracy under adversarial attacks on the ImageNet1K benchmark. Moreover, we demonstrate with extensive evaluations across 8 robustness benchmarks and 4 ViT architectures that AdaNCA, as a plug-in-play module, consistently improves the robustness of ViTs.

摘要: 视觉变形器(VITS)在图像分类任务中表现出了显著的性能，特别是当通过区域注意或卷积来配备局部信息时。虽然这样的体系结构从不同的粒度改善了特征聚合，但它们往往无法提高网络的健壮性。神经元胞自动机(NCA)能够通过局部交互对全局细胞表示进行建模，其训练策略和结构设计具有很强的泛化能力和对噪声输入的鲁棒性。在本文中，我们提出了用于视觉转换器的适配器神经元胞自动机(AdaNCA)，它使用NCA作为VIT层之间的即插即用适配器，增强了VIT的性能和对敌意样本和分布外输入的鲁棒性。为了克服标准NCA计算开销大的缺点，我们提出了动态交互来实现更有效的交互学习。此外，基于对AdaNCA布局和健壮性改进的分析，我们提出了一种识别AdaNCA最有效插入点的算法。在参数增加不到3%的情况下，AdaNCA有助于在对ImageNet1K基准的敌意攻击下将准确率绝对提高10%以上。此外，我们通过对8个健壮性基准和4个VIT体系结构的广泛评估，证明了AdaNCA作为一个即插即用模块，持续提高了VIT的健壮性。



## **19. Countermeasures Against Adversarial Examples in Radio Signal Classification**

无线信号分类中对抗示例的对策 cs.AI

Published in IEEE Wireless Communications Letters, vol. 10, no. 8,  pp. 1830-1834, Aug. 2021

**SubmitDate**: 2024-07-09    [abs](http://arxiv.org/abs/2407.06796v1) [paper-pdf](http://arxiv.org/pdf/2407.06796v1)

**Authors**: Lu Zhang, Sangarapillai Lambotharan, Gan Zheng, Basil AsSadhan, Fabio Roli

**Abstract**: Deep learning algorithms have been shown to be powerful in many communication network design problems, including that in automatic modulation classification. However, they are vulnerable to carefully crafted attacks called adversarial examples. Hence, the reliance of wireless networks on deep learning algorithms poses a serious threat to the security and operation of wireless networks. In this letter, we propose for the first time a countermeasure against adversarial examples in modulation classification. Our countermeasure is based on a neural rejection technique, augmented by label smoothing and Gaussian noise injection, that allows to detect and reject adversarial examples with high accuracy. Our results demonstrate that the proposed countermeasure can protect deep-learning based modulation classification systems against adversarial examples.

摘要: 深度学习算法已被证明在许多通信网络设计问题中非常强大，包括自动调制分类问题。然而，它们很容易受到精心设计的攻击，称为对抗性例子。因此，无线网络对深度学习算法的依赖对无线网络的安全和运营构成了严重威胁。在这封信中，我们首次提出了针对调制分类中对抗性示例的对策。我们的对策基于神经拒绝技术，通过标签平滑和高斯噪音注入增强，可以高准确性地检测和拒绝对抗性示例。我们的结果表明，提出的对策可以保护基于深度学习的调制分类系统免受对抗性示例的影响。



## **20. Diffusion-Based Adversarial Purification for Speaker Verification**

基于扩散的对抗净化说话人验证 eess.AS

Accepted by IEEE Signal Processing Letters

**SubmitDate**: 2024-07-09    [abs](http://arxiv.org/abs/2310.14270v3) [paper-pdf](http://arxiv.org/pdf/2310.14270v3)

**Authors**: Yibo Bai, Xiao-Lei Zhang, Xuelong Li

**Abstract**: Recently, automatic speaker verification (ASV) based on deep learning is easily contaminated by adversarial attacks, which is a new type of attack that injects imperceptible perturbations to audio signals so as to make ASV produce wrong decisions. This poses a significant threat to the security and reliability of ASV systems. To address this issue, we propose a Diffusion-Based Adversarial Purification (DAP) method that enhances the robustness of ASV systems against such adversarial attacks. Our method leverages a conditional denoising diffusion probabilistic model to effectively purify the adversarial examples and mitigate the impact of perturbations. DAP first introduces controlled noise into adversarial examples, and then performs a reverse denoising process to reconstruct clean audio. Experimental results demonstrate the efficacy of the proposed DAP in enhancing the security of ASV and meanwhile minimizing the distortion of the purified audio signals.

摘要: 近年来，基于深度学习的自动说话者验证（ASV）很容易受到对抗攻击的污染，对抗攻击是一种新型攻击，它向音频信号注入难以感知的扰动，使ASV做出错误的决策。这对ASV系统的安全性和可靠性构成了重大威胁。为了解决这个问题，我们提出了一种基于扩散的对抗性纯化（DAB）方法，该方法可以增强ASV系统针对此类对抗性攻击的鲁棒性。我们的方法利用条件去噪扩散概率模型来有效地净化对抗示例并减轻扰动的影响。DAB首先将受控噪音引入对抗性示例中，然后执行反向去噪过程以重建干净的音频。实验结果表明，所提出的DAB在增强ASV的安全性并同时最大限度地减少净化音频信号的失真方面的功效。



## **21. Improving the Transferability of Adversarial Examples by Feature Augmentation**

通过特征增强提高对抗性示例的可移植性 cs.CV

19 pages, 4 figures, 4 tables

**SubmitDate**: 2024-07-09    [abs](http://arxiv.org/abs/2407.06714v1) [paper-pdf](http://arxiv.org/pdf/2407.06714v1)

**Authors**: Donghua Wang, Wen Yao, Tingsong Jiang, Xiaohu Zheng, Junqi Wu, Xiaoqian Chen

**Abstract**: Despite the success of input transformation-based attacks on boosting adversarial transferability, the performance is unsatisfying due to the ignorance of the discrepancy across models. In this paper, we propose a simple but effective feature augmentation attack (FAUG) method, which improves adversarial transferability without introducing extra computation costs. Specifically, we inject the random noise into the intermediate features of the model to enlarge the diversity of the attack gradient, thereby mitigating the risk of overfitting to the specific model and notably amplifying adversarial transferability. Moreover, our method can be combined with existing gradient attacks to augment their performance further. Extensive experiments conducted on the ImageNet dataset across CNN and transformer models corroborate the efficacy of our method, e.g., we achieve improvement of +26.22% and +5.57% on input transformation-based attacks and combination methods, respectively.

摘要: 尽管基于输入转换的攻击在提高对抗可移植性方面取得了成功，但由于忽视了模型之间的差异，性能并不令人满意。在本文中，我们提出了一种简单但有效的特征增强攻击（FAUG）方法，该方法在不引入额外计算成本的情况下提高了对抗性可移植性。具体来说，我们将随机噪音注入到模型的中间特征中，以扩大攻击梯度的多样性，从而降低过度适应特定模型的风险，并显着放大对抗可移植性。此外，我们的方法可以与现有的梯度攻击相结合，以进一步增强其性能。在CNN和Transformer模型上对ImageNet数据集进行的大量实验证实了我们方法的有效性，例如，我们在基于输入转换的攻击和组合方法上分别实现了+26.22%和+5.57%的改进。



## **22. Universal Multi-view Black-box Attack against Object Detectors via Layout Optimization**

通过布局优化对对象检测器进行通用多视图黑匣子攻击 cs.CV

12 pages, 13 figures, 5 tables

**SubmitDate**: 2024-07-09    [abs](http://arxiv.org/abs/2407.06688v1) [paper-pdf](http://arxiv.org/pdf/2407.06688v1)

**Authors**: Donghua Wang, Wen Yao, Tingsong Jiang, Chao Li, Xiaoqian Chen

**Abstract**: Object detectors have demonstrated vulnerability to adversarial examples crafted by small perturbations that can deceive the object detector. Existing adversarial attacks mainly focus on white-box attacks and are merely valid at a specific viewpoint, while the universal multi-view black-box attack is less explored, limiting their generalization in practice. In this paper, we propose a novel universal multi-view black-box attack against object detectors, which optimizes a universal adversarial UV texture constructed by multiple image stickers for a 3D object via the designed layout optimization algorithm. Specifically, we treat the placement of image stickers on the UV texture as a circle-based layout optimization problem, whose objective is to find the optimal circle layout filled with image stickers so that it can deceive the object detector under the multi-view scenario. To ensure reasonable placement of image stickers, two constraints are elaborately devised. To optimize the layout, we adopt the random search algorithm enhanced by the devised important-aware selection strategy to find the most appropriate image sticker for each circle from the image sticker pools. Extensive experiments conducted on four common object detectors suggested that the detection performance decreases by a large magnitude of 74.29% on average in multi-view scenarios. Additionally, a novel evaluation tool based on the photo-realistic simulator is designed to assess the texture-based attack fairly.

摘要: 对象检测器已经证明了对由可能欺骗对象检测器的小扰动制作的敌意例子的脆弱性。现有的对抗性攻击主要集中在白盒攻击上，并且只在特定的视点有效，而通用的多视点黑盒攻击研究较少，限制了其在实践中的推广。本文提出了一种针对目标检测器的通用多视点黑盒攻击方法，通过设计的布局优化算法，优化了由多个图像贴纸构成的三维物体的通用对抗性UV纹理。具体地说，我们将图像贴纸在UV纹理上的放置视为一个基于圆的布局优化问题，其目标是在多视点场景下找到填充图像贴纸的最优圆形布局，从而欺骗对象检测器。为了确保图像贴纸的合理放置，精心设计了两个约束条件。为了优化布局，我们采用了改进的随机搜索算法，并设计了重要性感知选择策略，从图像贴纸池中为每个圆圈找到最合适的图像贴纸。在四种常见目标检测器上进行的大量实验表明，在多视角场景下，检测性能平均下降了74.29%。此外，还设计了一种基于照片真实感模拟器的评估工具来对基于纹理的攻击进行公平评估。



## **23. Attack GAN (AGAN ): A new Security Evaluation Tool for Perceptual Encryption**

Attack GAN（AGAN）：一种新的感知加密安全评估工具 cs.CV

**SubmitDate**: 2024-07-09    [abs](http://arxiv.org/abs/2407.06570v1) [paper-pdf](http://arxiv.org/pdf/2407.06570v1)

**Authors**: Umesh Kashyap, Sudev Kumar Padhi, Sk. Subidh Ali

**Abstract**: Training state-of-the-art (SOTA) deep learning models requires a large amount of data. The visual information present in the training data can be misused, which creates a huge privacy concern. One of the prominent solutions for this issue is perceptual encryption, which converts images into an unrecognizable format to protect the sensitive visual information in the training data. This comes at the cost of a significant reduction in the accuracy of the models. Adversarial Visual Information Hiding (AV IH) overcomes this drawback to protect image privacy by attempting to create encrypted images that are unrecognizable to the human eye while keeping relevant features for the target model. In this paper, we introduce the Attack GAN (AGAN ) method, a new Generative Adversarial Network (GAN )-based attack that exposes multiple vulnerabilities in the AV IH method. To show the adaptability, the AGAN is extended to traditional perceptual encryption methods of Learnable encryption (LE) and Encryption-then-Compression (EtC). Extensive experiments were conducted on diverse image datasets and target models to validate the efficacy of our AGAN method. The results show that AGAN can successfully break perceptual encryption methods by reconstructing original images from their AV IH encrypted images. AGAN can be used as a benchmark tool to evaluate the robustness of encryption methods for privacy protection such as AV IH.

摘要: 训练最先进的(SOTA)深度学习模型需要大量数据。训练数据中的视觉信息可能会被滥用，这会造成巨大的隐私问题。针对这一问题的一个突出解决方案是感知加密，它将图像转换为无法识别的格式，以保护训练数据中的敏感视觉信息。这是以显著降低模型精度为代价的。对抗性视觉信息隐藏(AV IH)克服了这一缺点，通过尝试创建人眼无法识别的加密图像来保护图像隐私，同时保留目标模型的相关特征。本文介绍了一种新的基于生成性对抗网络(GAN)的攻击方法--攻击GAN(AGAN)方法，该方法暴露了AVIH方法中的多个漏洞。为了显示其适应性，将AGAN扩展到传统的感知加密方法，如可学习加密(LE)和加密然后压缩(ETC)。在不同的图像数据集和目标模型上进行了广泛的实验，以验证我们的AGaN方法的有效性。结果表明，AGAN能够成功地打破感知加密方法，从他们的AVIH加密图像中重建原始图像。AGAN可以作为一个基准工具来评估用于隐私保护的加密方法的健壮性，例如AV IH。



## **24. DLOVE: A new Security Evaluation Tool for Deep Learning Based Watermarking Techniques**

DLOVE：基于深度学习的水印技术的新安全评估工具 cs.CR

**SubmitDate**: 2024-07-09    [abs](http://arxiv.org/abs/2407.06552v1) [paper-pdf](http://arxiv.org/pdf/2407.06552v1)

**Authors**: Sudev Kumar Padhi, Sk. Subidh Ali

**Abstract**: Recent developments in Deep Neural Network (DNN) based watermarking techniques have shown remarkable performance. The state-of-the-art DNN-based techniques not only surpass the robustness of classical watermarking techniques but also show their robustness against many image manipulation techniques. In this paper, we performed a detailed security analysis of different DNN-based watermarking techniques. We propose a new class of attack called the Deep Learning-based OVErwriting (DLOVE) attack, which leverages adversarial machine learning and overwrites the original embedded watermark with a targeted watermark in a watermarked image. To the best of our knowledge, this attack is the first of its kind. We have considered scenarios where watermarks are used to devise and formulate an adversarial attack in white box and black box settings. To show adaptability and efficiency, we launch our DLOVE attack analysis on seven different watermarking techniques, HiDDeN, ReDMark, PIMoG, Stegastamp, Aparecium, Distortion Agostic Deep Watermarking and Hiding Images in an Image. All these techniques use different approaches to create imperceptible watermarked images. Our attack analysis on these watermarking techniques with various constraints highlights the vulnerabilities of DNN-based watermarking. Extensive experimental results validate the capabilities of DLOVE. We propose DLOVE as a benchmark security analysis tool to test the robustness of future deep learning-based watermarking techniques.

摘要: 最新的基于DNN的技术不仅超越了经典水印技术的稳健性，而且表现出对许多图像篡改技术的稳健性。我们提出了一类新的攻击，称为基于深度学习的覆盖攻击(DLOVE)，它利用对抗性机器学习，在水印图像中使用目标水印覆盖原始嵌入的水印。为了显示我们的适应性和效率，我们对七种不同的水印技术进行了DLOVE攻击分析：HIDDEN、ReDMark、PIMoG、Stestamp、Aparecium、抗失真深度水印和图像中的隐藏图像。



## **25. WildGuard: Open One-Stop Moderation Tools for Safety Risks, Jailbreaks, and Refusals of LLMs**

WildGuard：针对LLC安全风险、越狱和拒绝的开放式一站式审核工具 cs.CL

First two authors contributed equally. Third and fourth authors  contributed equally

**SubmitDate**: 2024-07-09    [abs](http://arxiv.org/abs/2406.18495v2) [paper-pdf](http://arxiv.org/pdf/2406.18495v2)

**Authors**: Seungju Han, Kavel Rao, Allyson Ettinger, Liwei Jiang, Bill Yuchen Lin, Nathan Lambert, Yejin Choi, Nouha Dziri

**Abstract**: We introduce WildGuard -- an open, light-weight moderation tool for LLM safety that achieves three goals: (1) identifying malicious intent in user prompts, (2) detecting safety risks of model responses, and (3) determining model refusal rate. Together, WildGuard serves the increasing needs for automatic safety moderation and evaluation of LLM interactions, providing a one-stop tool with enhanced accuracy and broad coverage across 13 risk categories. While existing open moderation tools such as Llama-Guard2 score reasonably well in classifying straightforward model interactions, they lag far behind a prompted GPT-4, especially in identifying adversarial jailbreaks and in evaluating models' refusals, a key measure for evaluating safety behaviors in model responses.   To address these challenges, we construct WildGuardMix, a large-scale and carefully balanced multi-task safety moderation dataset with 92K labeled examples that cover vanilla (direct) prompts and adversarial jailbreaks, paired with various refusal and compliance responses. WildGuardMix is a combination of WildGuardTrain, the training data of WildGuard, and WildGuardTest, a high-quality human-annotated moderation test set with 5K labeled items covering broad risk scenarios. Through extensive evaluations on WildGuardTest and ten existing public benchmarks, we show that WildGuard establishes state-of-the-art performance in open-source safety moderation across all the three tasks compared to ten strong existing open-source moderation models (e.g., up to 26.4% improvement on refusal detection). Importantly, WildGuard matches and sometimes exceeds GPT-4 performance (e.g., up to 3.9% improvement on prompt harmfulness identification). WildGuard serves as a highly effective safety moderator in an LLM interface, reducing the success rate of jailbreak attacks from 79.8% to 2.4%.

摘要: 我们介绍了WildGuard--一个开放的、轻量级的LLM安全防御工具，它实现了三个目标：(1)识别用户提示中的恶意意图，(2)检测模型响应的安全风险，(3)确定模型拒绝率。综合起来，WildGuard可满足日益增长的自动安全审核和评估LLM交互作用的需求，提供了一种一站式工具，具有更高的准确性和广泛的覆盖范围，涵盖13个风险类别。虽然现有的开放式审核工具，如Llama-Guard2，在对直接的模型交互进行分类方面得分相当好，但它们远远落后于GPT-4，特别是在识别对抗性越狱和评估模型拒绝方面，这是评估模型响应中安全行为的关键指标。为了应对这些挑战，我们构建了WildGuardMix，这是一个大规模的、仔细平衡的多任务安全缓和数据集，具有92K标记的示例，涵盖普通(直接)提示和对抗性越狱，并与各种拒绝和合规响应配对。WildGuardMix是WildGuard的训练数据WildGuardTrain和WildGuardTest的组合，WildGuardTest是一种高质量的人工注释适度测试集，具有覆盖广泛风险情景的5K标签项目。通过对WildGuardTest和十个现有公共基准的广泛评估，我们表明WildGuard在所有三个任务中建立了开源安全适度的最先进性能，而不是现有的十个强大的开源适度模型(例如，拒绝检测方面高达26.4%的改进)。重要的是，WildGuard的性能与GPT-4相当，有时甚至超过GPT-4(例如，在及时识别危害性方面最高提高3.9%)。WildGuard在LLM界面中充当高效的安全调节器，将越狱攻击的成功率从79.8%降低到2.4%。



## **26. Defending Large Language Models Against Attacks With Residual Stream Activation Analysis**

利用剩余流激活分析防御大型语言模型免受攻击 cs.CR

**SubmitDate**: 2024-07-09    [abs](http://arxiv.org/abs/2406.03230v3) [paper-pdf](http://arxiv.org/pdf/2406.03230v3)

**Authors**: Amelia Kawasaki, Andrew Davis, Houssam Abbas

**Abstract**: The widespread adoption of Large Language Models (LLMs), exemplified by OpenAI's ChatGPT, brings to the forefront the imperative to defend against adversarial threats on these models. These attacks, which manipulate an LLM's output by introducing malicious inputs, undermine the model's integrity and the trust users place in its outputs. In response to this challenge, our paper presents an innovative defensive strategy, given white box access to an LLM, that harnesses residual activation analysis between transformer layers of the LLM. We apply a novel methodology for analyzing distinctive activation patterns in the residual streams for attack prompt classification. We curate multiple datasets to demonstrate how this method of classification has high accuracy across multiple types of attack scenarios, including our newly-created attack dataset. Furthermore, we enhance the model's resilience by integrating safety fine-tuning techniques for LLMs in order to measure its effect on our capability to detect attacks. The results underscore the effectiveness of our approach in enhancing the detection and mitigation of adversarial inputs, advancing the security framework within which LLMs operate.

摘要: 大型语言模型(LLM)的广泛采用，如OpenAI的ChatGPT，使防御这些模型上的对手威胁成为当务之急。这些攻击通过引入恶意输入来操纵LLM的输出，破坏了模型的完整性和用户对其输出的信任。为了应对这一挑战，我们的论文提出了一种创新的防御策略，在白盒访问LLM的情况下，该策略利用LLM变压器层之间的剩余激活分析。我们应用了一种新的方法来分析残留流中独特的激活模式，以进行攻击提示分类。我们精选了多个数据集，以演示此分类方法如何在多种类型的攻击场景中具有高精度，包括我们新创建的攻击数据集。此外，我们通过集成LLMS的安全微调技术来增强模型的弹性，以衡量其对我们检测攻击的能力的影响。这些结果强调了我们的方法在加强对敌对输入的检测和缓解、推进LLMS运作的安全框架方面的有效性。



## **27. Robust Prompt Optimization for Defending Language Models Against Jailbreaking Attacks**

保护语言模型免受越狱攻击的鲁棒即时优化 cs.LG

Code available at https://github.com/lapisrocks/rpo

**SubmitDate**: 2024-07-08    [abs](http://arxiv.org/abs/2401.17263v4) [paper-pdf](http://arxiv.org/pdf/2401.17263v4)

**Authors**: Andy Zhou, Bo Li, Haohan Wang

**Abstract**: Despite advances in AI alignment, large language models (LLMs) remain vulnerable to adversarial attacks or jailbreaking, in which adversaries can modify prompts to induce unwanted behavior. While some defenses have been proposed, they have not been adapted to newly proposed attacks and more challenging threat models. To address this, we propose an optimization-based objective for defending LLMs against jailbreaking attacks and an algorithm, Robust Prompt Optimization (RPO) to create robust system-level defenses. Our approach directly incorporates the adversary into the defensive objective and optimizes a lightweight and transferable suffix, enabling RPO to adapt to worst-case adaptive attacks. Our theoretical and experimental results show improved robustness to both jailbreaks seen during optimization and unknown jailbreaks, reducing the attack success rate (ASR) on GPT-4 to 6% and Llama-2 to 0% on JailbreakBench, setting the state-of-the-art. Code can be found at https://github.com/lapisrocks/rpo

摘要: 尽管在人工智能对齐方面取得了进展，但大型语言模型(LLM)仍然容易受到对手攻击或越狱的攻击，在这些攻击或越狱中，对手可以修改提示以诱导不想要的行为。虽然已经提出了一些防御措施，但它们还没有适应新提出的攻击和更具挑战性的威胁模型。为了解决这个问题，我们提出了一个基于优化的目标来保护LLMS免受越狱攻击，并提出了一个算法--稳健提示优化(RPO)来创建强大的系统级防御。我们的方法直接将对手合并到防御目标中，并优化了一个轻量级和可转移的后缀，使RPO能够适应最坏情况的自适应攻击。我们的理论和实验结果表明，对于优化期间看到的越狱和未知越狱，我们都提高了健壮性，将GPT-4上的攻击成功率(ASR)降低到6%，将Llama-2上的攻击成功率降低到0%，从而达到了最先进的水平。代码可在https://github.com/lapisrocks/rpo上找到



## **28. Non-Robust Features are Not Always Useful in One-Class Classification**

非稳健特征在一类分类中并不总是有用 cs.LG

CVPR Visual and Anomaly Detection (VAND) Workshop 2024

**SubmitDate**: 2024-07-08    [abs](http://arxiv.org/abs/2407.06372v1) [paper-pdf](http://arxiv.org/pdf/2407.06372v1)

**Authors**: Matthew Lau, Haoran Wang, Alec Helbling, Matthew Hul, ShengYun Peng, Martin Andreoni, Willian T. Lunardi, Wenke Lee

**Abstract**: The robustness of machine learning models has been questioned by the existence of adversarial examples. We examine the threat of adversarial examples in practical applications that require lightweight models for one-class classification. Building on Ilyas et al. (2019), we investigate the vulnerability of lightweight one-class classifiers to adversarial attacks and possible reasons for it. Our results show that lightweight one-class classifiers learn features that are not robust (e.g. texture) under stronger attacks. However, unlike in multi-class classification (Ilyas et al., 2019), these non-robust features are not always useful for the one-class task, suggesting that learning these unpredictive and non-robust features is an unwanted consequence of training.

摘要: 机器学习模型的稳健性因对抗性示例的存在而受到质疑。我们研究了实际应用中对抗性示例的威胁，这些应用需要轻量级模型进行一级分类。在Ilyas等人（2019）的基础上，我们研究了轻量级一类分类器对对抗攻击的脆弱性及其可能的原因。我们的结果表明，轻量级一类分类器在更强的攻击下学习不稳健的特征（例如纹理）。然而，与多类分类不同（Ilyas等人，2019年），这些非稳健特征并不总是对一类任务有用，这表明学习这些非预测性和非稳健特征是训练的不想要的结果。



## **29. Shedding More Light on Robust Classifiers under the lens of Energy-based Models**

在基于能量的模型的视角下更多地关注稳健分类器 cs.CV

Accepted at European Conference on Computer Vision (ECCV) 2024

**SubmitDate**: 2024-07-08    [abs](http://arxiv.org/abs/2407.06315v1) [paper-pdf](http://arxiv.org/pdf/2407.06315v1)

**Authors**: Mujtaba Hussain Mirza, Maria Rosaria Briglia, Senad Beadini, Iacopo Masi

**Abstract**: By reinterpreting a robust discriminative classifier as Energy-based Model (EBM), we offer a new take on the dynamics of adversarial training (AT). Our analysis of the energy landscape during AT reveals that untargeted attacks generate adversarial images much more in-distribution (lower energy) than the original data from the point of view of the model. Conversely, we observe the opposite for targeted attacks. On the ground of our thorough analysis, we present new theoretical and practical results that show how interpreting AT energy dynamics unlocks a better understanding: (1) AT dynamic is governed by three phases and robust overfitting occurs in the third phase with a drastic divergence between natural and adversarial energies (2) by rewriting the loss of TRadeoff-inspired Adversarial DEfense via Surrogate-loss minimization (TRADES) in terms of energies, we show that TRADES implicitly alleviates overfitting by means of aligning the natural energy with the adversarial one (3) we empirically show that all recent state-of-the-art robust classifiers are smoothing the energy landscape and we reconcile a variety of studies about understanding AT and weighting the loss function under the umbrella of EBMs. Motivated by rigorous evidence, we propose Weighted Energy Adversarial Training (WEAT), a novel sample weighting scheme that yields robust accuracy matching the state-of-the-art on multiple benchmarks such as CIFAR-10 and SVHN and going beyond in CIFAR-100 and Tiny-ImageNet. We further show that robust classifiers vary in the intensity and quality of their generative capabilities, and offer a simple method to push this capability, reaching a remarkable Inception Score (IS) and FID using a robust classifier without training for generative modeling. The code to reproduce our results is available at http://github.com/OmnAI-Lab/Robust-Classifiers-under-the-lens-of-EBM/ .

摘要: 通过将稳健的判别分类器重新解释为基于能量的模型(EBM)，我们提供了一种新的方法来研究对手训练(AT)的动态。我们对AT过程中的能量格局的分析表明，从模型的角度来看，非目标攻击产生的敌意图像比原始数据更不均匀(能量更低)。相反，我们在有针对性的攻击中观察到相反的情况。在我们深入分析的基础上，我们提出了新的理论和实践结果，表明解释AT能量动力学如何揭示更好的理解：(1)AT动态由三个阶段控制，鲁棒过拟合发生在第三阶段，自然能量和对抗能量之间存在巨大差异(2)通过代理损失最小化(交易)在能量方面改写了权衡激发的对抗性防御的损失，我们表明，交易通过将自然能量与对手能量对齐的方式隐含地缓解了过度匹配。(3)我们的经验表明，所有最近最先进的稳健分类器都在平滑能量格局，我们协调了关于理解AT和在EBM保护伞下加权损失函数的各种研究。在严格证据的激励下，我们提出了加权能量对抗训练(Weat)，这是一种新的样本加权方案，其精度与CIFAR-10和SVHN等多个基准测试的最新水平相当，并超过CIFAR-100和Tiny-ImageNet。我们进一步证明了健壮分类器在其生成能力的强度和质量上存在差异，并提供了一种简单的方法来推动这一能力，使用健壮分类器而不需要为生成性建模进行训练就可以达到显著的初始得分(IS)和FID。复制我们结果的代码可以在http://github.com/OmnAI-Lab/Robust-Classifiers-under-the-lens-of-EBM/上找到。



## **30. Improving Alignment and Robustness with Circuit Breakers**

改善断路器的对准和稳健性 cs.LG

**SubmitDate**: 2024-07-08    [abs](http://arxiv.org/abs/2406.04313v3) [paper-pdf](http://arxiv.org/pdf/2406.04313v3)

**Authors**: Andy Zou, Long Phan, Justin Wang, Derek Duenas, Maxwell Lin, Maksym Andriushchenko, Rowan Wang, Zico Kolter, Matt Fredrikson, Dan Hendrycks

**Abstract**: AI systems can take harmful actions and are highly vulnerable to adversarial attacks. We present an approach, inspired by recent advances in representation engineering, that interrupts the models as they respond with harmful outputs with "circuit breakers." Existing techniques aimed at improving alignment, such as refusal training, are often bypassed. Techniques such as adversarial training try to plug these holes by countering specific attacks. As an alternative to refusal training and adversarial training, circuit-breaking directly controls the representations that are responsible for harmful outputs in the first place. Our technique can be applied to both text-only and multimodal language models to prevent the generation of harmful outputs without sacrificing utility -- even in the presence of powerful unseen attacks. Notably, while adversarial robustness in standalone image recognition remains an open challenge, circuit breakers allow the larger multimodal system to reliably withstand image "hijacks" that aim to produce harmful content. Finally, we extend our approach to AI agents, demonstrating considerable reductions in the rate of harmful actions when they are under attack. Our approach represents a significant step forward in the development of reliable safeguards to harmful behavior and adversarial attacks.

摘要: 人工智能系统可能采取有害行动，并且非常容易受到对抗性攻击。我们提出了一种方法，灵感来自于最近在表示工程方面的进展，该方法中断了模型，因为它们用“断路器”来响应有害的输出。旨在改善一致性的现有技术，如拒绝训练，经常被绕过。对抗性训练等技术试图通过反击特定攻击来堵塞这些漏洞。作为拒绝训练和对抗性训练的另一种选择，断路直接控制首先要对有害输出负责的陈述。我们的技术可以应用于纯文本和多模式语言模型，在不牺牲效用的情况下防止产生有害输出-即使在存在强大的看不见的攻击的情况下也是如此。值得注意的是，虽然独立图像识别中的对抗性健壮性仍然是一个开放的挑战，但断路器允许更大的多模式系统可靠地经受住旨在产生有害内容的图像“劫持”。最后，我们将我们的方法扩展到人工智能代理，表明当他们受到攻击时，有害行动的比率大大降低。我们的方法代表着在发展对有害行为和敌对攻击的可靠保障方面向前迈出了重要的一步。



## **31. Adaptive and robust watermark against model extraction attack**

抗模型提取攻击的自适应鲁棒水印 cs.CR

**SubmitDate**: 2024-07-08    [abs](http://arxiv.org/abs/2405.02365v2) [paper-pdf](http://arxiv.org/pdf/2405.02365v2)

**Authors**: Kaiyi Pang

**Abstract**: Large language models (LLMs) demonstrate general intelligence across a variety of machine learning tasks, thereby enhancing the commercial value of their intellectual property (IP). To protect this IP, model owners typically allow user access only in a black-box manner, however, adversaries can still utilize model extraction attacks to steal the model intelligence encoded in model generation. Watermarking technology offers a promising solution for defending against such attacks by embedding unique identifiers into the model-generated content. However, existing watermarking methods often compromise the quality of generated content due to heuristic alterations and lack robust mechanisms to counteract adversarial strategies, thus limiting their practicality in real-world scenarios. In this paper, we introduce an adaptive and robust watermarking method (named ModelShield) to protect the IP of LLMs. Our method incorporates a self-watermarking mechanism that allows LLMs to autonomously insert watermarks into their generated content to avoid the degradation of model content. We also propose a robust watermark detection mechanism capable of effectively identifying watermark signals under the interference of varying adversarial strategies. Besides, ModelShield is a plug-and-play method that does not require additional model training, enhancing its applicability in LLM deployments. Extensive evaluations on two real-world datasets and three LLMs demonstrate that our method surpasses existing methods in terms of defense effectiveness and robustness while significantly reducing the degradation of watermarking on the model-generated content.

摘要: 大型语言模型(LLM)在各种机器学习任务中展示了一般智能，从而提高了其知识产权(IP)的商业价值。为了保护这个IP，模型所有者通常只允许用户以黑盒方式访问，但是，攻击者仍然可以利用模型提取攻击来窃取模型生成中编码的模型情报。水印技术通过在模型生成的内容中嵌入唯一标识符，为防御此类攻击提供了一种很有前途的解决方案。然而，现有的水印方法往往会由于启发式修改而影响生成内容的质量，并且缺乏强大的机制来对抗对抗性策略，从而限制了它们在现实世界场景中的实用性。本文提出了一种自适应的稳健水印算法(ModelShield)来保护LLMS的IP地址。我们的方法结合了一种自水印机制，允许LLM自主地在其生成的内容中插入水印，以避免模型内容的降级。我们还提出了一种稳健的水印检测机制，能够在不同的对抗策略的干扰下有效地识别水印信号。此外，ModelShield是一种即插即用的方法，不需要额外的模型培训，增强了其在LLM部署中的适用性。在两个真实数据集和三个LLM上的广泛评估表明，我们的方法在防御有效性和稳健性方面优于现有方法，同时显着降低了水印对模型生成内容的退化。



## **32. Multi-View Black-Box Physical Attacks on Infrared Pedestrian Detectors Using Adversarial Infrared Grid**

使用对抗红外网格对红外行人探测器进行多视图黑匣子物理攻击 cs.CV

**SubmitDate**: 2024-07-08    [abs](http://arxiv.org/abs/2407.01168v2) [paper-pdf](http://arxiv.org/pdf/2407.01168v2)

**Authors**: Kalibinuer Tiliwalidi, Chengyin Hu, Weiwen Shi

**Abstract**: While extensive research exists on physical adversarial attacks within the visible spectrum, studies on such techniques in the infrared spectrum are limited. Infrared object detectors are vital in modern technological applications but are susceptible to adversarial attacks, posing significant security threats. Previous studies using physical perturbations like light bulb arrays and aerogels for white-box attacks, or hot and cold patches for black-box attacks, have proven impractical or limited in multi-view support. To address these issues, we propose the Adversarial Infrared Grid (AdvGrid), which models perturbations in a grid format and uses a genetic algorithm for black-box optimization. These perturbations are cyclically applied to various parts of a pedestrian's clothing to facilitate multi-view black-box physical attacks on infrared pedestrian detectors. Extensive experiments validate AdvGrid's effectiveness, stealthiness, and robustness. The method achieves attack success rates of 80.00\% in digital environments and 91.86\% in physical environments, outperforming baseline methods. Additionally, the average attack success rate exceeds 50\% against mainstream detectors, demonstrating AdvGrid's robustness. Our analyses include ablation studies, transfer attacks, and adversarial defenses, confirming the method's superiority.

摘要: 虽然在可见光光谱内对物理对抗攻击已有广泛的研究，但在红外光谱中对这类技术的研究有限。红外目标探测器在现代技术应用中至关重要，但容易受到对抗性攻击，构成重大安全威胁。以前的研究证明，使用物理扰动，如灯泡阵列和气凝胶进行白盒攻击，或使用冷热补丁进行黑盒攻击，都被证明是不切实际的，或者在多视角支持方面受到限制。为了解决这些问题，我们提出了对抗性红外网格(AdvGrid)，它以网格的形式对扰动进行建模，并使用遗传算法进行黑盒优化。这些扰动被循环应用于行人衣服的不同部分，以促进对红外行人探测器的多视角黑匣子物理攻击。大量实验验证了AdvGrid的有效性、隐蔽性和健壮性。该方法在数字环境下的攻击成功率为80.00%，在物理环境下的攻击成功率为91.86%，优于基准攻击方法。此外，对主流检测器的平均攻击成功率超过50%，显示了AdvGrid的健壮性。我们的分析包括烧蚀研究、转移攻击和对抗性防御，证实了该方法的优越性。



## **33. Malicious Agent Detection for Robust Multi-Agent Collaborative Perception**

用于鲁棒多代理协作感知的恶意代理检测 cs.CR

Accepted by IROS 2024

**SubmitDate**: 2024-07-08    [abs](http://arxiv.org/abs/2310.11901v2) [paper-pdf](http://arxiv.org/pdf/2310.11901v2)

**Authors**: Yangheng Zhao, Zhen Xiang, Sheng Yin, Xianghe Pang, Siheng Chen, Yanfeng Wang

**Abstract**: Recently, multi-agent collaborative (MAC) perception has been proposed and outperformed the traditional single-agent perception in many applications, such as autonomous driving. However, MAC perception is more vulnerable to adversarial attacks than single-agent perception due to the information exchange. The attacker can easily degrade the performance of a victim agent by sending harmful information from a malicious agent nearby. In this paper, we extend adversarial attacks to an important perception task -- MAC object detection, where generic defenses such as adversarial training are no longer effective against these attacks. More importantly, we propose Malicious Agent Detection (MADE), a reactive defense specific to MAC perception that can be deployed by each agent to accurately detect and then remove any potential malicious agent in its local collaboration network. In particular, MADE inspects each agent in the network independently using a semi-supervised anomaly detector based on a double-hypothesis test with the Benjamini-Hochberg procedure to control the false positive rate of the inference. For the two hypothesis tests, we propose a match loss statistic and a collaborative reconstruction loss statistic, respectively, both based on the consistency between the agent to be inspected and the ego agent where our detector is deployed. We conduct comprehensive evaluations on a benchmark 3D dataset V2X-sim and a real-road dataset DAIR-V2X and show that with the protection of MADE, the drops in the average precision compared with the best-case "oracle" defender against our attack are merely 1.28% and 0.34%, respectively, much lower than 8.92% and 10.00% for adversarial training, respectively.

摘要: 近年来，多智能体协作(MAC)感知被提出，并在许多应用中优于传统的单智能体感知，如自主驾驶。然而，由于信息的交换，MAC感知比单代理感知更容易受到敌意攻击。攻击者可以很容易地通过从附近的恶意代理发送有害信息来降低受害者代理的性能。在本文中，我们将对抗性攻击扩展到一项重要的感知任务--MAC对象检测，在这种情况下，对抗性训练等一般防御手段不再有效地对抗这些攻击。更重要的是，我们提出了恶意代理检测(Made)，这是一种针对MAC感知的反应性防御，可以由每个代理部署以准确检测并随后删除其本地协作网络中的任何潜在恶意代理。特别地，Made使用基于双假设检验的半监督异常检测器独立地检查网络中的每个代理，并结合Benjamini-Hochberg过程来控制推理的误检率。对于这两种假设检验，我们分别提出了一个匹配损失统计量和一个协作重建损失统计量，这两个统计量都是基于待检查代理和部署检测器的自我代理之间的一致性。我们在基准3D数据集V2X-SIM和真实道路数据集DAIR-V2X上进行了综合评估，结果表明，在Made的保护下，与最佳情况下的Oracle防御者相比，对抗我们的攻击的平均精度分别下降了1.28%和0.34%，远低于对抗性训练的8.92%和10.00%。



## **34. Exploring the Adversarial Capabilities of Large Language Models**

探索大型语言模型的对抗能力 cs.AI

**SubmitDate**: 2024-07-08    [abs](http://arxiv.org/abs/2402.09132v4) [paper-pdf](http://arxiv.org/pdf/2402.09132v4)

**Authors**: Lukas Struppek, Minh Hieu Le, Dominik Hintersdorf, Kristian Kersting

**Abstract**: The proliferation of large language models (LLMs) has sparked widespread and general interest due to their strong language generation capabilities, offering great potential for both industry and research. While previous research delved into the security and privacy issues of LLMs, the extent to which these models can exhibit adversarial behavior remains largely unexplored. Addressing this gap, we investigate whether common publicly available LLMs have inherent capabilities to perturb text samples to fool safety measures, so-called adversarial examples resp.~attacks. More specifically, we investigate whether LLMs are inherently able to craft adversarial examples out of benign samples to fool existing safe rails. Our experiments, which focus on hate speech detection, reveal that LLMs succeed in finding adversarial perturbations, effectively undermining hate speech detection systems. Our findings carry significant implications for (semi-)autonomous systems relying on LLMs, highlighting potential challenges in their interaction with existing systems and safety measures.

摘要: 大型语言模型因其强大的语言生成能力而引起了广泛的关注，为工业和研究提供了巨大的潜力。虽然之前的研究已经深入研究了LLMS的安全和隐私问题，但这些模型在多大程度上可以表现出敌对行为，仍然很大程度上还没有被探索。针对这一差距，我们调查了常见的公开可用的LLM是否具有固有的能力来扰乱文本样本以愚弄安全措施，即所谓的对抗性示例攻击。更具体地说，我们调查LLM是否天生就能够从良性样本中制作敌意示例，以愚弄现有的安全Rail。我们的实验集中在仇恨语音检测上，实验表明，LLMS成功地发现了敌意扰动，有效地破坏了仇恨语音检测系统。我们的发现对依赖LLMS的(半)自治系统具有重大影响，突显了它们与现有系统和安全措施相互作用的潜在挑战。



## **35. Improving Adversarial Transferability of Vision-Language Pre-training Models through Collaborative Multimodal Interaction**

通过协作多模式交互提高视觉语言预训练模型的对抗性可移植性 cs.CV

This work won first place in CVPR 2024 Workshop Challenge: Black-box  Adversarial Attacks on Vision Foundation Models

**SubmitDate**: 2024-07-08    [abs](http://arxiv.org/abs/2403.10883v2) [paper-pdf](http://arxiv.org/pdf/2403.10883v2)

**Authors**: Jiyuan Fu, Zhaoyu Chen, Kaixun Jiang, Haijing Guo, Jiafeng Wang, Shuyong Gao, Wenqiang Zhang

**Abstract**: Despite the substantial advancements in Vision-Language Pre-training (VLP) models, their susceptibility to adversarial attacks poses a significant challenge. Existing work rarely studies the transferability of attacks on VLP models, resulting in a substantial performance gap from white-box attacks. We observe that prior work overlooks the interaction mechanisms between modalities, which plays a crucial role in understanding the intricacies of VLP models. In response, we propose a novel attack, called Collaborative Multimodal Interaction Attack (CMI-Attack), leveraging modality interaction through embedding guidance and interaction enhancement. Specifically, attacking text at the embedding level while preserving semantics, as well as utilizing interaction image gradients to enhance constraints on perturbations of texts and images. Significantly, in the image-text retrieval task on Flickr30K dataset, CMI-Attack raises the transfer success rates from ALBEF to TCL, $\text{CLIP}_{\text{ViT}}$ and $\text{CLIP}_{\text{CNN}}$ by 8.11%-16.75% over state-of-the-art methods. Moreover, CMI-Attack also demonstrates superior performance in cross-task generalization scenarios. Our work addresses the underexplored realm of transfer attacks on VLP models, shedding light on the importance of modality interaction for enhanced adversarial robustness.

摘要: 尽管视觉语言预训练(VLP)模式有了很大的进步，但它们对对手攻击的敏感性构成了一个巨大的挑战。现有的工作很少研究攻击对VLP模型的可转移性，导致与白盒攻击相比性能有很大的差距。我们注意到，以前的工作忽略了通道之间的相互作用机制，这在理解VLP模型的复杂性方面起着至关重要的作用。对此，我们提出了一种新的攻击方法，称为协作多模式交互攻击(CMI-Attack)，通过嵌入引导和交互增强来利用通道交互。具体地说，在保留语义的同时在嵌入层攻击文本，以及利用交互图像梯度来增强对文本和图像扰动的约束。值得注意的是，在Flickr30K数据集的图文检索任务中，CMI-Attack将从ALBEF到TCL、$\Text{Clip}_{\Text{Vit}}$和$\Text{Clip}_{\Text{CNN}}$的传输成功率比最先进的方法提高了8.11%-16.75%。此外，CMI-Attack在跨任务泛化场景中也表现出了优越的性能。我们的工作解决了VLP模型上未被探索的传输攻击领域，揭示了通道交互对于增强对手健壮性的重要性。



## **36. A Survey of Fragile Model Watermarking**

脆弱模型水印综述 cs.CR

Submitted Signal Processing

**SubmitDate**: 2024-07-08    [abs](http://arxiv.org/abs/2406.04809v4) [paper-pdf](http://arxiv.org/pdf/2406.04809v4)

**Authors**: Zhenzhe Gao, Yu Cheng, Zhaoxia Yin

**Abstract**: Model fragile watermarking, inspired by both the field of adversarial attacks on neural networks and traditional multimedia fragile watermarking, has gradually emerged as a potent tool for detecting tampering, and has witnessed rapid development in recent years. Unlike robust watermarks, which are widely used for identifying model copyrights, fragile watermarks for models are designed to identify whether models have been subjected to unexpected alterations such as backdoors, poisoning, compression, among others. These alterations can pose unknown risks to model users, such as misidentifying stop signs as speed limit signs in classic autonomous driving scenarios. This paper provides an overview of the relevant work in the field of model fragile watermarking since its inception, categorizing them and revealing the developmental trajectory of the field, thus offering a comprehensive survey for future endeavors in model fragile watermarking.

摘要: 模型脆弱水印受到神经网络对抗攻击领域和传统多媒体脆弱水印的启发，逐渐成为检测篡改的有力工具，并在近年来得到了快速发展。与广泛用于识别模型版权的稳健水印不同，模型的脆弱水印旨在识别模型是否遭受了意外更改，例如后门、中毒、压缩等。这些更改可能会给模型用户带来未知的风险，例如在经典自动驾驶场景中将停车标志误识别为限速标志。本文概述了模型脆弱水印领域自诞生以来的相关工作，对其进行了分类，揭示了该领域的发展轨迹，从而为模型脆弱水印的未来工作提供了全面的综述。



## **37. To Generate or Not? Safety-Driven Unlearned Diffusion Models Are Still Easy To Generate Unsafe Images ... For Now**

生成还是不生成？安全驱动的未学习扩散模型仍然很容易生成不安全的图像.现在 cs.CV

Accepted by ECCV'24. Codes are available at  https://github.com/OPTML-Group/Diffusion-MU-Attack

**SubmitDate**: 2024-07-07    [abs](http://arxiv.org/abs/2310.11868v4) [paper-pdf](http://arxiv.org/pdf/2310.11868v4)

**Authors**: Yimeng Zhang, Jinghan Jia, Xin Chen, Aochuan Chen, Yihua Zhang, Jiancheng Liu, Ke Ding, Sijia Liu

**Abstract**: The recent advances in diffusion models (DMs) have revolutionized the generation of realistic and complex images. However, these models also introduce potential safety hazards, such as producing harmful content and infringing data copyrights. Despite the development of safety-driven unlearning techniques to counteract these challenges, doubts about their efficacy persist. To tackle this issue, we introduce an evaluation framework that leverages adversarial prompts to discern the trustworthiness of these safety-driven DMs after they have undergone the process of unlearning harmful concepts. Specifically, we investigated the adversarial robustness of DMs, assessed by adversarial prompts, when eliminating unwanted concepts, styles, and objects. We develop an effective and efficient adversarial prompt generation approach for DMs, termed UnlearnDiffAtk. This method capitalizes on the intrinsic classification abilities of DMs to simplify the creation of adversarial prompts, thereby eliminating the need for auxiliary classification or diffusion models. Through extensive benchmarking, we evaluate the robustness of widely-used safety-driven unlearned DMs (i.e., DMs after unlearning undesirable concepts, styles, or objects) across a variety of tasks. Our results demonstrate the effectiveness and efficiency merits of UnlearnDiffAtk over the state-of-the-art adversarial prompt generation method and reveal the lack of robustness of current safetydriven unlearning techniques when applied to DMs. Codes are available at https://github.com/OPTML-Group/Diffusion-MU-Attack. WARNING: There exist AI generations that may be offensive in nature.

摘要: 扩散模型的最新进展使逼真和复杂图像的生成发生了革命性的变化。然而，这些模式也带来了潜在的安全隐患，如产生有害内容和侵犯数据著作权。尽管发展了安全驱动的遗忘技术来应对这些挑战，但对其有效性的怀疑依然存在。为了解决这个问题，我们引入了一个评估框架，利用对抗性提示，在这些以安全为导向的DM经历了忘记有害概念的过程后，识别他们的可信度。具体地说，我们研究了DM在消除不需要的概念、风格和对象时，通过对抗性提示评估的对抗性健壮性。本文提出了一种高效的敌意提示生成方法，称为UnlearnDiffAtk。这种方法利用DM的内在分类能力来简化对抗性提示的创建，从而消除了对辅助分类或扩散模型的需要。通过广泛的基准测试，我们评估了广泛使用的安全驱动的未学习DM(即在忘记不需要的概念、风格或对象后的DM)在各种任务中的健壮性。实验结果证明了UnlearnDiffAtk算法相对于最新的对抗性提示生成方法的有效性和高效性，并揭示了当前安全驱动的遗忘技术在应用于决策支持系统时存在的健壮性不足。有关代码，请访问https://github.com/OPTML-Group/Diffusion-MU-Attack.警告：存在可能具有攻击性的人工智能世代。



## **38. Rethinking Targeted Adversarial Attacks For Neural Machine Translation**

重新思考神经机器翻译的有针对性的对抗攻击 cs.CL

5 pages, 2 figures, accepted by ICASSP 2024

**SubmitDate**: 2024-07-07    [abs](http://arxiv.org/abs/2407.05319v1) [paper-pdf](http://arxiv.org/pdf/2407.05319v1)

**Authors**: Junjie Wu, Lemao Liu, Wei Bi, Dit-Yan Yeung

**Abstract**: Targeted adversarial attacks are widely used to evaluate the robustness of neural machine translation systems. Unfortunately, this paper first identifies a critical issue in the existing settings of NMT targeted adversarial attacks, where their attacking results are largely overestimated. To this end, this paper presents a new setting for NMT targeted adversarial attacks that could lead to reliable attacking results. Under the new setting, it then proposes a Targeted Word Gradient adversarial Attack (TWGA) method to craft adversarial examples. Experimental results demonstrate that our proposed setting could provide faithful attacking results for targeted adversarial attacks on NMT systems, and the proposed TWGA method can effectively attack such victim NMT systems. In-depth analyses on a large-scale dataset further illustrate some valuable findings. 1 Our code and data are available at https://github.com/wujunjie1998/TWGA.

摘要: 有针对性的对抗攻击被广泛用于评估神经机器翻译系统的鲁棒性。不幸的是，本文首先指出了NMT有针对性的对抗攻击现有环境中的一个关键问题，即它们的攻击结果在很大程度上被高估了。为此，本文为NMT有针对性的对抗攻击提供了一种新设置，可以带来可靠的攻击结果。在新的设置下，它随后提出了一种有针对性的词梯度对抗攻击（TWGA）方法来制作对抗示例。实验结果表明，我们提出的设置可以为针对NMT系统的有针对性的对抗攻击提供可靠的攻击结果，并且提出的TWGA方法可以有效地攻击此类受害NMT系统。对大规模数据集的深入分析进一步说明了一些有价值的发现。1我们的代码和数据可在https://github.com/wujunjie1998/TWGA上获取。



## **39. TrojanRAG: Retrieval-Augmented Generation Can Be Backdoor Driver in Large Language Models**

TrojanRAG：检索增强生成可以成为大型语言模型中的后门驱动程序 cs.CR

19 pages, 14 figures, 4 tables

**SubmitDate**: 2024-07-07    [abs](http://arxiv.org/abs/2405.13401v4) [paper-pdf](http://arxiv.org/pdf/2405.13401v4)

**Authors**: Pengzhou Cheng, Yidong Ding, Tianjie Ju, Zongru Wu, Wei Du, Ping Yi, Zhuosheng Zhang, Gongshen Liu

**Abstract**: Large language models (LLMs) have raised concerns about potential security threats despite performing significantly in Natural Language Processing (NLP). Backdoor attacks initially verified that LLM is doing substantial harm at all stages, but the cost and robustness have been criticized. Attacking LLMs is inherently risky in security review, while prohibitively expensive. Besides, the continuous iteration of LLMs will degrade the robustness of backdoors. In this paper, we propose TrojanRAG, which employs a joint backdoor attack in the Retrieval-Augmented Generation, thereby manipulating LLMs in universal attack scenarios. Specifically, the adversary constructs elaborate target contexts and trigger sets. Multiple pairs of backdoor shortcuts are orthogonally optimized by contrastive learning, thus constraining the triggering conditions to a parameter subspace to improve the matching. To improve the recall of the RAG for the target contexts, we introduce a knowledge graph to construct structured data to achieve hard matching at a fine-grained level. Moreover, we normalize the backdoor scenarios in LLMs to analyze the real harm caused by backdoors from both attackers' and users' perspectives and further verify whether the context is a favorable tool for jailbreaking models. Extensive experimental results on truthfulness, language understanding, and harmfulness show that TrojanRAG exhibits versatility threats while maintaining retrieval capabilities on normal queries.

摘要: 尽管大型语言模型(LLM)在自然语言处理(NLP)中表现出色，但仍引发了人们对潜在安全威胁的担忧。后门攻击最初证实了LLM在所有阶段都在造成实质性的危害，但其成本和健壮性受到了批评。在安全审查中，攻击LLMS固有的风险，同时代价高得令人望而却步。此外，LLMS的连续迭代会降低后门的健壮性。在本文中，我们提出了TrojanRAG，它在检索-增强生成中使用联合后门攻击，从而在通用攻击场景下操纵LLMS。具体地说，对手构建了精心设计的目标上下文和触发集。通过对比学习对多对后门捷径进行正交化优化，从而将触发条件约束到一个参数子空间以提高匹配性。为了提高RAG对目标上下文的查全率，我们引入了知识图来构建结构化数据，以实现细粒度的硬匹配。此外，我们对LLMS中的后门场景进行了规范化，从攻击者和用户的角度分析了后门造成的真实危害，并进一步验证了上下文是否为越狱模型的有利工具。在真实性、语言理解和危害性方面的大量实验结果表明，TrojanRAG在保持对正常查询的检索能力的同时，表现出通用性威胁。



## **40. FedCG: Leverage Conditional GAN for Protecting Privacy and Maintaining Competitive Performance in Federated Learning**

FedCG：利用有条件GAN在联邦学习中保护隐私并保持竞争绩效 cs.LG

**SubmitDate**: 2024-07-07    [abs](http://arxiv.org/abs/2111.08211v3) [paper-pdf](http://arxiv.org/pdf/2111.08211v3)

**Authors**: Yuezhou Wu, Yan Kang, Jiahuan Luo, Yuanqin He, Qiang Yang

**Abstract**: Federated learning (FL) aims to protect data privacy by enabling clients to build machine learning models collaboratively without sharing their private data. Recent works demonstrate that information exchanged during FL is subject to gradient-based privacy attacks, and consequently, a variety of privacy-preserving methods have been adopted to thwart such attacks. However, these defensive methods either introduce orders of magnitude more computational and communication overheads (e.g., with homomorphic encryption) or incur substantial model performance losses in terms of prediction accuracy (e.g., with differential privacy). In this work, we propose $\textsc{FedCG}$, a novel federated learning method that leverages conditional generative adversarial networks to achieve high-level privacy protection while still maintaining competitive model performance. $\textsc{FedCG}$ decomposes each client's local network into a private extractor and a public classifier and keeps the extractor local to protect privacy. Instead of exposing extractors, $\textsc{FedCG}$ shares clients' generators with the server for aggregating clients' shared knowledge, aiming to enhance the performance of each client's local networks. Extensive experiments demonstrate that $\textsc{FedCG}$ can achieve competitive model performance compared with FL baselines, and privacy analysis shows that $\textsc{FedCG}$ has a high-level privacy-preserving capability. Code is available at https://github.com/yankang18/FedCG

摘要: 联合学习(FL)旨在通过使客户能够在不共享他们的私人数据的情况下协作地建立机器学习模型来保护数据隐私。最近的研究表明，在外语学习过程中交换的信息会受到基于梯度的隐私攻击，因此，人们已经采取了各种隐私保护方法来阻止这种攻击。然而，这些防御方法要么引入更多数量级的计算和通信开销(例如，使用同态加密)，要么在预测精度方面导致显著的模型性能损失(例如，使用差分隐私)。在这项工作中，我们提出了一种新的联邦学习方法，它利用条件生成对抗网络来实现高级别的隐私保护，同时又保持了竞争模型的性能。$\Textsc{FedCG}$将每个客户端的本地网络分解为私有提取程序和公共分类器，并将提取程序保留在本地以保护隐私。$\extsc{FedCG}$不公开提取程序，而是与服务器共享客户端的生成器，以聚合客户端共享的知识，旨在增强每个客户端的本地网络的性能。大量实验表明，与FL基线相比，$\extsc{FedCG}$具有与FL基线相当的模型性能，隐私分析表明，$\extsc{FedCG}$具有较高的隐私保护能力。代码可在https://github.com/yankang18/FedCG上找到



## **41. A Novel Bifurcation Method for Observation Perturbation Attacks on Reinforcement Learning Agents: Load Altering Attacks on a Cyber Physical Power System**

一种用于对强化学习代理进行观察扰动攻击的新型分歧方法：对网络物理电力系统的负载改变攻击 cs.LG

12 pages, 5 figures

**SubmitDate**: 2024-07-06    [abs](http://arxiv.org/abs/2407.05182v1) [paper-pdf](http://arxiv.org/pdf/2407.05182v1)

**Authors**: Kiernan Broda-Milian, Ranwa Al-Mallah, Hanane Dagdougui

**Abstract**: Components of cyber physical systems, which affect real-world processes, are often exposed to the internet. Replacing conventional control methods with Deep Reinforcement Learning (DRL) in energy systems is an active area of research, as these systems become increasingly complex with the advent of renewable energy sources and the desire to improve their efficiency. Artificial Neural Networks (ANN) are vulnerable to specific perturbations of their inputs or features, called adversarial examples. These perturbations are difficult to detect when properly regularized, but have significant effects on the ANN's output. Because DRL uses ANN to map optimal actions to observations, they are similarly vulnerable to adversarial examples. This work proposes a novel attack technique for continuous control using Group Difference Logits loss with a bifurcation layer. By combining aspects of targeted and untargeted attacks, the attack significantly increases the impact compared to an untargeted attack, with drastically smaller distortions than an optimally targeted attack. We demonstrate the impacts of powerful gradient-based attacks in a realistic smart energy environment, show how the impacts change with different DRL agents and training procedures, and use statistical and time-series analysis to evaluate attacks' stealth. The results show that adversarial attacks can have significant impacts on DRL controllers, and constraining an attack's perturbations makes it difficult to detect. However, certain DRL architectures are far more robust, and robust training methods can further reduce the impact.

摘要: 影响真实世界进程的网络物理系统的组件经常暴露在互联网上。在能源系统中用深度强化学习(DRL)取代传统的控制方法是一个活跃的研究领域，因为随着可再生能源的出现和提高其效率的愿望，这些系统变得越来越复杂。人工神经网络(ANN)容易受到其输入或特征的特定扰动，称为对抗性示例。当适当地正则化时，这些扰动很难被检测到，但会对神经网络的输出产生重大影响。由于DRL使用人工神经网络将最优动作映射到观测值，因此它们同样容易受到对手例子的攻击。本文提出了一种新的基于分组差值Logits损失的连续控制攻击技术。通过结合目标攻击和非目标攻击的各个方面，与非目标攻击相比，该攻击显著增加了影响，扭曲程度比最佳目标攻击小得多。我们展示了强大的基于梯度的攻击在现实的智能能源环境中的影响，展示了影响如何随不同的DRL代理和训练过程而变化，并使用统计和时间序列分析来评估攻击的隐蔽性。结果表明，对抗性攻击可以对DRL控制器产生显著影响，并且限制攻击的扰动使得检测变得困难。然而，某些DRL架构要健壮得多，健壮的训练方法可以进一步降低影响。



## **42. Robust Skin Color Driven Privacy Preserving Face Recognition via Function Secret Sharing**

通过功能秘密共享的鲁棒肤色驱动的隐私保护面部识别 cs.CV

Accepted at ICIP2024

**SubmitDate**: 2024-07-06    [abs](http://arxiv.org/abs/2407.05045v1) [paper-pdf](http://arxiv.org/pdf/2407.05045v1)

**Authors**: Dong Han, Yufan Jiang, Yong Li, Ricardo Mendes, Joachim Denzler

**Abstract**: In this work, we leverage the pure skin color patch from the face image as the additional information to train an auxiliary skin color feature extractor and face recognition model in parallel to improve performance of state-of-the-art (SOTA) privacy-preserving face recognition (PPFR) systems. Our solution is robust against black-box attacking and well-established generative adversarial network (GAN) based image restoration. We analyze the potential risk in previous work, where the proposed cosine similarity computation might directly leak the protected precomputed embedding stored on the server side. We propose a Function Secret Sharing (FSS) based face embedding comparison protocol without any intermediate result leakage. In addition, we show in experiments that the proposed protocol is more efficient compared to the Secret Sharing (SS) based protocol.

摘要: 在这项工作中，我们利用面部图像中的纯肤色补丁作为附加信息来并行训练辅助肤色特征提取器和面部识别模型，以提高最新技术水平（SOTA）隐私保护面部识别（PPFR）系统的性能。我们的解决方案对于黑匣子攻击和基于成熟的生成对抗网络（GAN）的图像恢复具有鲁棒性。我们分析了之前工作中的潜在风险，其中提出的cos相似度计算可能会直接泄露存储在服务器端的受保护的预计算嵌入。我们提出了一种基于功能秘密共享（FSG）的人脸嵌入比较协议，不会出现任何中间结果泄露。此外，我们在实验中表明，与基于秘密共享（SS）的协议相比，所提出的协议更有效。



## **43. Certified Zeroth-order Black-Box Defense with Robust UNet Denoiser**

经过认证的零阶黑匣子防御，具有强大的UNet降噪器 cs.CV

**SubmitDate**: 2024-07-06    [abs](http://arxiv.org/abs/2304.06430v2) [paper-pdf](http://arxiv.org/pdf/2304.06430v2)

**Authors**: Astha Verma, A V Subramanyam, Siddhesh Bangar, Naman Lal, Rajiv Ratn Shah, Shin'ichi Satoh

**Abstract**: Certified defense methods against adversarial perturbations have been recently investigated in the black-box setting with a zeroth-order (ZO) perspective. However, these methods suffer from high model variance with low performance on high-dimensional datasets due to the ineffective design of the denoiser and are limited in their utilization of ZO techniques. To this end, we propose a certified ZO preprocessing technique for removing adversarial perturbations from the attacked image in the black-box setting using only model queries. We propose a robust UNet denoiser (RDUNet) that ensures the robustness of black-box models trained on high-dimensional datasets. We propose a novel black-box denoised smoothing (DS) defense mechanism, ZO-RUDS, by prepending our RDUNet to the black-box model, ensuring black-box defense. We further propose ZO-AE-RUDS in which RDUNet followed by autoencoder (AE) is prepended to the black-box model. We perform extensive experiments on four classification datasets, CIFAR-10, CIFAR-10, Tiny Imagenet, STL-10, and the MNIST dataset for image reconstruction tasks. Our proposed defense methods ZO-RUDS and ZO-AE-RUDS beat SOTA with a huge margin of $35\%$ and $9\%$, for low dimensional (CIFAR-10) and with a margin of $20.61\%$ and $23.51\%$ for high-dimensional (STL-10) datasets, respectively.

摘要: 针对对抗性扰动的认证防御方法最近在零阶(ZO)视角的黑盒环境中被研究。然而，由于去噪器的设计不合理，这些方法在高维数据集上存在模型方差大、性能低的问题，限制了ZO技术的应用。为此，我们提出了一种经过验证的ZO预处理技术，用于在仅使用模型查询的情况下从黑盒环境中去除攻击图像中的对抗性扰动。我们提出了一种稳健的UNET去噪器(RDUNet)，以确保在高维数据集上训练的黑盒模型的稳健性。提出了一种新的黑盒去噪平滑防御机制ZO-RUDS，将RDUNet加入到黑盒模型中，保证了黑盒防御。我们进一步提出了ZO-AE-RUDS，其中RDUNet后跟自动编码器(AE)优先于黑盒模型。我们在CIFAR-10、CIFAR-10、Tiny Imagenet、STL-10和MNIST四个分类数据集上进行了大量的实验，用于图像重建任务。我们提出的防御方法ZO-RUDS和ZO-AE-RUDS对于低维数据集(CIFAR-10)分别以35美元和9美元的巨大优势击败了SOTA，而对于高维数据集(STL-10)分别以20.61美元和23.51美元的优势击败了SOTA。



## **44. PAC-Bayesian Adversarially Robust Generalization Bounds for Graph Neural Network**

图神经网络的Pac-Bayesian对抗鲁棒广义界 stat.ML

38pages

**SubmitDate**: 2024-07-06    [abs](http://arxiv.org/abs/2402.04038v2) [paper-pdf](http://arxiv.org/pdf/2402.04038v2)

**Authors**: Tan Sun, Junhong Lin

**Abstract**: Graph neural networks (GNNs) have gained popularity for various graph-related tasks. However, similar to deep neural networks, GNNs are also vulnerable to adversarial attacks. Empirical studies have shown that adversarially robust generalization has a pivotal role in establishing effective defense algorithms against adversarial attacks. In this paper, we contribute by providing adversarially robust generalization bounds for two kinds of popular GNNs, graph convolutional network (GCN) and message passing graph neural network, using the PAC-Bayesian framework. Our result reveals that spectral norm of the diffusion matrix on the graph and spectral norm of the weights as well as the perturbation factor govern the robust generalization bounds of both models. Our bounds are nontrivial generalizations of the results developed in (Liao et al., 2020) from the standard setting to adversarial setting while avoiding exponential dependence of the maximum node degree. As corollaries, we derive better PAC-Bayesian robust generalization bounds for GCN in the standard setting, which improve the bounds in (Liao et al., 2020) by avoiding exponential dependence on the maximum node degree.

摘要: 图神经网络(GNN)在各种与图相关的任务中得到了广泛的应用。然而，与深度神经网络类似，GNN也容易受到敌意攻击。经验研究表明，对抗性健壮性泛化在建立有效的防御算法抵抗对抗性攻击方面起着关键作用。在本文中，我们利用PAC-贝叶斯框架，为两种流行的GNN，图卷积网络(GCN)和消息传递图神经网络(GCN)提供了相对健壮的泛化界。我们的结果表明，图上扩散矩阵的谱范数和权值的谱范数以及扰动因子控制着两个模型的鲁棒推广界。我们的界是(Liao等人，2020)中发展的结果从标准环境到对抗环境的非平凡推广，同时避免了最大节点度的指数依赖。作为推论，我们得到了GCN在标准设置下更好的PAC-贝叶斯鲁棒推广界，通过避免对最大节点度的指数依赖，改进了(Liao等人，2020)的界。



## **45. Defensive Reconfigurable Intelligent Surface (D-RIS) Based on Non-Reciprocal Channel Links**

基于非互惠通道链接的防御性可重新配置智能表面（D-RIS） eess.SP

14 pages, journal paper

**SubmitDate**: 2024-07-06    [abs](http://arxiv.org/abs/2407.04905v1) [paper-pdf](http://arxiv.org/pdf/2407.04905v1)

**Authors**: Kun Chen-Hu, Petar Popovski

**Abstract**: A reconfigurable intelligent surface (RIS) is commonly made of low-cost passive and reflective meta-materials with excellent beam steering capabilities. It is applied to enhance wireless communication systems as a customizable signal reflector. However, RIS can also be adversely employed to disrupt the existing communication systems by introducing new types of vulnerability to the physical layer. We consider the \emph{RIS-In-The-Middle (RITM) attack}, in which an adversary uses RIS to jeopardize the direct channel between two transceivers by providing an alternative one with higher signal quality. This adversary can eavesdrop on all exchanged data by the legitimate users, but also perform a false data injection to the receiver. This work devises anti-attack techniques based on a non-reciprocal channel produced by a defensive RIS (D-RIS). The proposed precoding and combining methods and the channel estimation procedure for a non-reciprocal link are effective against potential adversaries while keeping the existing advantages of the RIS. We analyse the robustness of the system against attacks in terms of achievable secrecy rate and probability of detecting fake data. We believe that this defensive role of RIS can be a basis for new protocols and algorithms in the area.

摘要: 可重构智能表面(RIS)通常由低成本的被动和反射超材料制成，具有良好的波控能力。它作为一种可定制的信号反射器应用于增强无线通信系统。然而，RIS也可以被用来通过向物理层引入新类型的漏洞来扰乱现有的通信系统。我们考虑了中间RIS(RIS-in-the-Medium，RITM)攻击，在该攻击中，敌手使用RIS来危害两个收发信机之间的直接信道，从而提供一个具有更高信号质量的替代收发信机。此敌手可以窃听合法用户交换的所有数据，但也可以向接收方执行虚假数据注入。该工作设计了基于防御RIS(D-RIS)产生的非互易信道的抗攻击技术。所提出的预编码和合并方法以及针对非互易链路的信道估计过程在保持RIS的现有优势的同时有效地对抗潜在的对手。我们从可达到的保密率和检测到虚假数据的概率两个方面分析了系统对攻击的稳健性。我们相信，RIS的这种防御角色可以作为该领域新协议和算法的基础。



## **46. Late Breaking Results: Fortifying Neural Networks: Safeguarding Against Adversarial Attacks with Stochastic Computing**

最新突破性成果：强化神经网络：利用随机计算防范对抗攻击 cs.CR

3 pages, 1 figure, 2 tables

**SubmitDate**: 2024-07-05    [abs](http://arxiv.org/abs/2407.04861v1) [paper-pdf](http://arxiv.org/pdf/2407.04861v1)

**Authors**: Faeze S. Banitaba, Sercan Aygun, M. Hassan Najafi

**Abstract**: In neural network (NN) security, safeguarding model integrity and resilience against adversarial attacks has become paramount. This study investigates the application of stochastic computing (SC) as a novel mechanism to fortify NN models. The primary objective is to assess the efficacy of SC to mitigate the deleterious impact of attacks on NN results. Through a series of rigorous experiments and evaluations, we explore the resilience of NNs employing SC when subjected to adversarial attacks. Our findings reveal that SC introduces a robust layer of defense, significantly reducing the susceptibility of networks to attack-induced alterations in their outcomes. This research contributes novel insights into the development of more secure and reliable NN systems, essential for applications in sensitive domains where data integrity is of utmost concern.

摘要: 在神经网络（NN）安全中，保护模型完整性和抵御对抗攻击的弹性已变得至关重要。本研究探讨了随机计算（SC）作为强化神经网络模型的新型机制的应用。主要目标是评估SC减轻攻击对NN结果的有害影响的有效性。通过一系列严格的实验和评估，我们探索了使用SC的NN在遭受对抗性攻击时的弹性。我们的研究结果表明，SC引入了强大的防御层，显着降低了网络对攻击诱导的结果改变的敏感性。这项研究为开发更安全、更可靠的神经网络系统提供了新的见解，这对于数据完整性最为关注的敏感领域的应用至关重要。



## **47. Where have you been? A Study of Privacy Risk for Point-of-Interest Recommendation**

你去哪儿了？兴趣点推荐的隐私风险研究 cs.LG

18 pages

**SubmitDate**: 2024-07-05    [abs](http://arxiv.org/abs/2310.18606v2) [paper-pdf](http://arxiv.org/pdf/2310.18606v2)

**Authors**: Kunlin Cai, Jinghuai Zhang, Zhiqing Hong, Will Shand, Guang Wang, Desheng Zhang, Jianfeng Chi, Yuan Tian

**Abstract**: As location-based services (LBS) have grown in popularity, more human mobility data has been collected. The collected data can be used to build machine learning (ML) models for LBS to enhance their performance and improve overall experience for users. However, the convenience comes with the risk of privacy leakage since this type of data might contain sensitive information related to user identities, such as home/work locations. Prior work focuses on protecting mobility data privacy during transmission or prior to release, lacking the privacy risk evaluation of mobility data-based ML models. To better understand and quantify the privacy leakage in mobility data-based ML models, we design a privacy attack suite containing data extraction and membership inference attacks tailored for point-of-interest (POI) recommendation models, one of the most widely used mobility data-based ML models. These attacks in our attack suite assume different adversary knowledge and aim to extract different types of sensitive information from mobility data, providing a holistic privacy risk assessment for POI recommendation models. Our experimental evaluation using two real-world mobility datasets demonstrates that current POI recommendation models are vulnerable to our attacks. We also present unique findings to understand what types of mobility data are more susceptible to privacy attacks. Finally, we evaluate defenses against these attacks and highlight future directions and challenges. Our attack suite is released at https://github.com/KunlinChoi/POIPrivacy.

摘要: 随着基于位置的服务(LBS)越来越受欢迎，人们收集了更多的人类移动性数据。收集到的数据可用于为LBS构建机器学习(ML)模型，以增强其性能并改善用户的整体体验。然而，随之而来的是隐私泄露的风险，因为这种类型的数据可能包含与用户身份相关的敏感信息，如家庭/工作地点。以往的工作主要集中在移动数据传输过程中或发布前的隐私保护上，缺乏对基于移动数据的ML模型的隐私风险评估。为了更好地理解和量化基于移动数据的ML模型中的隐私泄漏，我们设计了一个隐私攻击套件，其中包含针对兴趣点(POI)推荐模型的数据提取和成员关系推理攻击，该模型是应用最广泛的基于移动数据的ML模型之一。我们攻击套件中的这些攻击假设了不同的对手知识，旨在从移动数据中提取不同类型的敏感信息，为POI推荐模型提供全面的隐私风险评估。我们使用两个真实的移动数据集进行的实验评估表明，当前的POI推荐模型容易受到我们的攻击。我们还提出了独特的发现，以了解哪些类型的移动数据更容易受到隐私攻击。最后，我们评估了针对这些攻击的防御措施，并强调了未来的方向和挑战。我们的攻击套件在https://github.com/KunlinChoi/POIPrivacy.发布



## **48. CosPGD: an efficient white-box adversarial attack for pixel-wise prediction tasks**

CosPVD：针对像素预测任务的高效白盒对抗攻击 cs.CV

Accepted at 41st International Conference on Machine Learning (ICML),  2024

**SubmitDate**: 2024-07-05    [abs](http://arxiv.org/abs/2302.02213v3) [paper-pdf](http://arxiv.org/pdf/2302.02213v3)

**Authors**: Shashank Agnihotri, Steffen Jung, Margret Keuper

**Abstract**: While neural networks allow highly accurate predictions in many tasks, their lack of robustness towards even slight input perturbations often hampers their deployment. Adversarial attacks such as the seminal projected gradient descent (PGD) offer an effective means to evaluate a model's robustness and dedicated solutions have been proposed for attacks on semantic segmentation or optical flow estimation. While they attempt to increase the attack's efficiency, a further objective is to balance its effect, so that it acts on the entire image domain instead of isolated point-wise predictions. This often comes at the cost of optimization stability and thus efficiency. Here, we propose CosPGD, an attack that encourages more balanced errors over the entire image domain while increasing the attack's overall efficiency. To this end, CosPGD leverages a simple alignment score computed from any pixel-wise prediction and its target to scale the loss in a smooth and fully differentiable way. It leads to efficient evaluations of a model's robustness for semantic segmentation as well as regression models (such as optical flow, disparity estimation, or image restoration), and it allows it to outperform the previous SotA attack on semantic segmentation. We provide code for the CosPGD algorithm and example usage at https://github.com/shashankskagnihotri/cospgd.

摘要: 虽然神经网络允许在许多任务中进行高度准确的预测，但它们对即使是轻微的输入扰动缺乏稳健性，往往会阻碍它们的部署。诸如种子投影梯度下降(PGD)这样的对抗性攻击为评估模型的稳健性提供了一种有效的手段，针对语义分割或光流估计的攻击已经提出了专门的解决方案。虽然他们试图提高攻击的效率，但另一个目标是平衡其影响，使其作用于整个图像域，而不是孤立的逐点预测。这通常是以优化、稳定性和效率为代价的。在这里，我们提出了CosPGD，这是一种在提高攻击整体效率的同时，鼓励在整个图像域上更平衡错误的攻击。为此，CosPGD利用根据任何像素预测及其目标计算的简单对齐分数，以平滑和完全可区分的方式衡量损失。它可以有效地评估模型对语义分割和回归模型(如光流、视差估计或图像恢复)的稳健性，并使其性能优于以前的SOTA语义分割攻击。我们提供了CosPGD算法的代码和https://github.com/shashankskagnihotri/cospgd.上的示例使用



## **49. On Evaluating The Performance of Watermarked Machine-Generated Texts Under Adversarial Attacks**

关于评估带有水印的机器生成文本在对抗性攻击下的性能 cs.CR

**SubmitDate**: 2024-07-05    [abs](http://arxiv.org/abs/2407.04794v1) [paper-pdf](http://arxiv.org/pdf/2407.04794v1)

**Authors**: Zesen Liu, Tianshuo Cong, Xinlei He, Qi Li

**Abstract**: Large Language Models (LLMs) excel in various applications, including text generation and complex tasks. However, the misuse of LLMs raises concerns about the authenticity and ethical implications of the content they produce, such as deepfake news, academic fraud, and copyright infringement. Watermarking techniques, which embed identifiable markers in machine-generated text, offer a promising solution to these issues by allowing for content verification and origin tracing. Unfortunately, the robustness of current LLM watermarking schemes under potential watermark removal attacks has not been comprehensively explored.   In this paper, to fill this gap, we first systematically comb the mainstream watermarking schemes and removal attacks on machine-generated texts, and then we categorize them into pre-text (before text generation) and post-text (after text generation) classes so that we can conduct diversified analyses. In our experiments, we evaluate eight watermarks (five pre-text, three post-text) and twelve attacks (two pre-text, ten post-text) across 87 scenarios. Evaluation results indicate that (1) KGW and Exponential watermarks offer high text quality and watermark retention but remain vulnerable to most attacks; (2) Post-text attacks are found to be more efficient and practical than pre-text attacks; (3) Pre-text watermarks are generally more imperceptible, as they do not alter text fluency, unlike post-text watermarks; (4) Additionally, combined attack methods can significantly increase effectiveness, highlighting the need for more robust watermarking solutions. Our study underscores the vulnerabilities of current techniques and the necessity for developing more resilient schemes.

摘要: 大型语言模型(LLM)在各种应用中表现出色，包括文本生成和复杂任务。然而，LLMS的滥用引发了人们对它们产生的内容的真实性和伦理影响的担忧，例如深度假新闻、学术欺诈和侵犯版权。在机器生成的文本中嵌入可识别标记的水印技术，通过允许内容验证和来源追踪，为这些问题提供了一种有前途的解决方案。遗憾的是，目前的LLM水印方案在潜在的水印去除攻击下的稳健性还没有得到全面的研究。为了填补这一空白，本文首先对主流的机器生成文本水印算法和去除攻击进行了系统的梳理，然后将其分为前文本类(文本生成前)和后文本类(文本生成后)，以便进行多样化的分析。在我们的实验中，我们评估了87个场景中的8个水印(5个前置文本，3个后置文本)和12个攻击(2个前置文本，10个后置文本)。评估结果表明：(1)KGW和指数水印具有高的文本质量和水印保留率，但仍然容易受到大多数攻击；(2)后文本攻击被发现比前文本攻击更有效和实用；(3)前文本水印通常更不可察觉，因为它们不像后文本水印那样改变文本的流畅性；(4)此外，组合攻击方法可以显著提高攻击效果，突出了对更健壮的水印解决方案的需求。我们的研究强调了当前技术的脆弱性，以及开发更具弹性的方案的必要性。



## **50. Remembering Everything Makes You Vulnerable: A Limelight on Machine Unlearning for Personalized Healthcare Sector**

记住一切让你变得脆弱：个性化医疗保健领域机器学习的聚光灯 cs.LG

15 Pages, Exploring unlearning techniques on ECG Classifier

**SubmitDate**: 2024-07-05    [abs](http://arxiv.org/abs/2407.04589v1) [paper-pdf](http://arxiv.org/pdf/2407.04589v1)

**Authors**: Ahan Chatterjee, Sai Anirudh Aryasomayajula, Rajat Chaudhari, Subhajit Paul, Vishwa Mohan Singh

**Abstract**: As the prevalence of data-driven technologies in healthcare continues to rise, concerns regarding data privacy and security become increasingly paramount. This thesis aims to address the vulnerability of personalized healthcare models, particularly in the context of ECG monitoring, to adversarial attacks that compromise patient privacy. We propose an approach termed "Machine Unlearning" to mitigate the impact of exposed data points on machine learning models, thereby enhancing model robustness against adversarial attacks while preserving individual privacy. Specifically, we investigate the efficacy of Machine Unlearning in the context of personalized ECG monitoring, utilizing a dataset of clinical ECG recordings. Our methodology involves training a deep neural classifier on ECG data and fine-tuning the model for individual patients. We demonstrate the susceptibility of fine-tuned models to adversarial attacks, such as the Fast Gradient Sign Method (FGSM), which can exploit additional data points in personalized models. To address this vulnerability, we propose a Machine Unlearning algorithm that selectively removes sensitive data points from fine-tuned models, effectively enhancing model resilience against adversarial manipulation. Experimental results demonstrate the effectiveness of our approach in mitigating the impact of adversarial attacks while maintaining the pre-trained model accuracy.

摘要: 随着数据驱动技术在医疗保健中的普及程度持续上升，对数据隐私和安全的担忧变得越来越重要。这篇论文旨在解决个性化医疗模型的脆弱性，特别是在心电监测的背景下，面对危及患者隐私的对抗性攻击。我们提出了一种称为“机器遗忘”的方法来缓解暴露数据点对机器学习模型的影响，从而在保护个人隐私的同时增强了模型对对手攻击的稳健性。具体地说，我们利用临床心电记录的数据集，在个性化心电监测的背景下，研究了机器遗忘的有效性。我们的方法包括对心电数据训练深度神经分类器，并针对个别患者微调模型。我们证明了微调模型对敌意攻击的敏感性，例如快速梯度符号方法(FGSM)，它可以利用个性化模型中的额外数据点。为了解决这一漏洞，我们提出了一种机器遗忘算法，该算法选择性地从微调的模型中移除敏感数据点，有效地增强了模型对对手操纵的弹性。实验结果表明，该方法在保持预先训练好的模型精度的同时，有效地抑制了对抗性攻击的影响。



