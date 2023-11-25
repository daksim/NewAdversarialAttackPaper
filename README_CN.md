# Latest Adversarial Attack Papers
**update at 2023-11-25 10:54:20**

翻译来自 https://cloud.tencent.com/document/product/551/15619

## **1. Adversarial Backdoor Attack by Naturalistic Data Poisoning on Trajectory Prediction in Autonomous Driving**

自动驾驶轨迹预测中自然主义数据中毒的对抗性后门攻击 cs.CV

**SubmitDate**: 2023-11-22    [abs](http://arxiv.org/abs/2306.15755v2) [paper-pdf](http://arxiv.org/pdf/2306.15755v2)

**Authors**: Mozhgan Pourkeshavarz, Mohammad Sabokrou, Amir Rasouli

**Abstract**: In autonomous driving, behavior prediction is fundamental for safe motion planning, hence the security and robustness of prediction models against adversarial attacks are of paramount importance. We propose a novel adversarial backdoor attack against trajectory prediction models as a means of studying their potential vulnerabilities. Our attack affects the victim at training time via naturalistic, hence stealthy, poisoned samples crafted using a novel two-step approach. First, the triggers are crafted by perturbing the trajectory of attacking vehicle and then disguised by transforming the scene using a bi-level optimization technique. The proposed attack does not depend on a particular model architecture and operates in a black-box manner, thus can be effective without any knowledge of the victim model. We conduct extensive empirical studies using state-of-the-art prediction models on two benchmark datasets using metrics customized for trajectory prediction. We show that the proposed attack is highly effective, as it can significantly hinder the performance of prediction models, unnoticeable by the victims, and efficient as it forces the victim to generate malicious behavior even under constrained conditions. Via ablative studies, we analyze the impact of different attack design choices followed by an evaluation of existing defence mechanisms against the proposed attack.

摘要: 在自动驾驶中，行为预测是安全运动规划的基础，因此预测模型对对抗性攻击的安全性和鲁棒性至关重要。我们提出了一种针对轨迹预测模型的新型对抗性后门攻击，作为研究其潜在漏洞的一种手段。我们的攻击在训练时通过自然主义影响受害者，因此使用一种新颖的两步方法制作隐形的有毒样本。首先，通过扰动攻击车辆的轨迹来制作触发器，然后使用双层优化技术通过变换场景来伪装。所提出的攻击不依赖于特定的模型架构，并以黑盒的方式操作，因此可以是有效的，而不需要受害者模型的任何知识。我们使用最先进的预测模型对两个基准数据集进行了广泛的实证研究，这些数据集使用为轨迹预测定制的指标。我们表明，所提出的攻击是非常有效的，因为它可以显着阻碍预测模型的性能，不明显的受害者，和有效的，因为它迫使受害者产生恶意行为，即使在约束条件下。通过烧蚀研究，我们分析了不同的攻击设计选择的影响，然后对现有的防御机制，对拟议的攻击进行评估。



## **2. Transfer Attacks and Defenses for Large Language Models on Coding Tasks**

编码任务中大语言模型的迁移攻击与防御 cs.LG

**SubmitDate**: 2023-11-22    [abs](http://arxiv.org/abs/2311.13445v1) [paper-pdf](http://arxiv.org/pdf/2311.13445v1)

**Authors**: Chi Zhang, Zifan Wang, Ravi Mangal, Matt Fredrikson, Limin Jia, Corina Pasareanu

**Abstract**: Modern large language models (LLMs), such as ChatGPT, have demonstrated impressive capabilities for coding tasks including writing and reasoning about code. They improve upon previous neural network models of code, such as code2seq or seq2seq, that already demonstrated competitive results when performing tasks such as code summarization and identifying code vulnerabilities. However, these previous code models were shown vulnerable to adversarial examples, i.e. small syntactic perturbations that do not change the program's semantics, such as the inclusion of "dead code" through false conditions or the addition of inconsequential print statements, designed to "fool" the models. LLMs can also be vulnerable to the same adversarial perturbations but a detailed study on this concern has been lacking so far. In this paper we aim to investigate the effect of adversarial perturbations on coding tasks with LLMs. In particular, we study the transferability of adversarial examples, generated through white-box attacks on smaller code models, to LLMs. Furthermore, to make the LLMs more robust against such adversaries without incurring the cost of retraining, we propose prompt-based defenses that involve modifying the prompt to include additional information such as examples of adversarially perturbed code and explicit instructions for reversing adversarial perturbations. Our experiments show that adversarial examples obtained with a smaller code model are indeed transferable, weakening the LLMs' performance. The proposed defenses show promise in improving the model's resilience, paving the way to more robust defensive solutions for LLMs in code-related applications.

摘要: 现代大型语言模型(LLM)，如ChatGPT，已经在编码任务(包括编写代码和进行推理)方面展示了令人印象深刻的能力。它们改进了以前的代码神经网络模型，如code2seq或seq2seq，这些模型在执行代码汇总和识别代码漏洞等任务时已经展示了具有竞争力的结果。然而，这些以前的代码模型被证明容易受到敌意示例的攻击，即不会改变程序语义的小的语法扰动，例如通过错误条件包括“死代码”或添加无关紧要的打印语句，旨在“愚弄”模型。LLMS也可能容易受到同样的对抗性干扰，但迄今为止还缺乏关于这一问题的详细研究。本文旨在研究对抗性扰动对LLMS编码任务的影响。特别是，我们研究了通过对较小代码模型进行白盒攻击而生成的对抗性示例到LLMS的可转移性。此外，为了使LLMS在不招致再培训成本的情况下对此类对手更加健壮，我们提出了基于提示的防御措施，涉及修改提示以包括额外的信息，例如对手扰动代码的示例和用于逆转对手扰动的显式指令。我们的实验表明，用较小的编码模型得到的对抗性例子确实是可移植的，从而削弱了LLMS的性能。拟议的防御措施在提高模型的弹性方面显示出了希望，为代码相关应用中的LLM提供更强大的防御解决方案铺平了道路。



## **3. From Principle to Practice: Vertical Data Minimization for Machine Learning**

从原理到实践：机器学习的垂直数据最小化 cs.LG

Accepted at IEEE S&P 2024

**SubmitDate**: 2023-11-22    [abs](http://arxiv.org/abs/2311.10500v2) [paper-pdf](http://arxiv.org/pdf/2311.10500v2)

**Authors**: Robin Staab, Nikola Jovanović, Mislav Balunović, Martin Vechev

**Abstract**: Aiming to train and deploy predictive models, organizations collect large amounts of detailed client data, risking the exposure of private information in the event of a breach. To mitigate this, policymakers increasingly demand compliance with the data minimization (DM) principle, restricting data collection to only that data which is relevant and necessary for the task. Despite regulatory pressure, the problem of deploying machine learning models that obey DM has so far received little attention. In this work, we address this challenge in a comprehensive manner. We propose a novel vertical DM (vDM) workflow based on data generalization, which by design ensures that no full-resolution client data is collected during training and deployment of models, benefiting client privacy by reducing the attack surface in case of a breach. We formalize and study the corresponding problem of finding generalizations that both maximize data utility and minimize empirical privacy risk, which we quantify by introducing a diverse set of policy-aligned adversarial scenarios. Finally, we propose a range of baseline vDM algorithms, as well as Privacy-aware Tree (PAT), an especially effective vDM algorithm that outperforms all baselines across several settings. We plan to release our code as a publicly available library, helping advance the standardization of DM for machine learning. Overall, we believe our work can help lay the foundation for further exploration and adoption of DM principles in real-world applications.

摘要: 为了训练和部署预测模型，组织收集了大量详细的客户数据，在发生入侵时冒着私人信息暴露的风险。为了缓解这一问题，政策制定者越来越多地要求遵守数据最小化(DM)原则，将数据收集仅限于与任务相关和必要的数据。尽管面临监管压力，但到目前为止，部署服从DM的机器学习模型的问题几乎没有得到关注。在这项工作中，我们以全面的方式应对这一挑战。我们提出了一种基于数据泛化的垂直数据挖掘(VDM)工作流，该工作流在设计上确保了在模型的训练和部署过程中不收集全分辨率的客户数据，从而减少了在发生攻击时的攻击面，从而有利于客户隐私。我们形式化并研究了相应的问题，即找到既最大化数据效用又最小化经验隐私风险的概括，我们通过引入一组与策略一致的对抗场景来量化这些概括。最后，我们提出了一系列的基线VDM算法，以及隐私感知树(PAT)，这是一种特别有效的VDM算法，其性能在几种设置下都优于所有基线。我们计划将我们的代码作为一个公开的库发布，帮助推进机器学习的DM标准化。总体而言，我们相信我们的工作可以为进一步探索和采用DM原理在现实世界中的应用奠定基础。



## **4. Hard Label Black Box Node Injection Attack on Graph Neural Networks**

基于图神经网络的硬标签黑盒节点注入攻击 cs.LG

**SubmitDate**: 2023-11-22    [abs](http://arxiv.org/abs/2311.13244v1) [paper-pdf](http://arxiv.org/pdf/2311.13244v1)

**Authors**: Yu Zhou, Zihao Dong, Guofeng Zhang, Jingchen Tang

**Abstract**: While graph neural networks have achieved state-of-the-art performances in many real-world tasks including graph classification and node classification, recent works have demonstrated they are also extremely vulnerable to adversarial attacks. Most previous works have focused on attacking node classification networks under impractical white-box scenarios. In this work, we will propose a non-targeted Hard Label Black Box Node Injection Attack on Graph Neural Networks, which to the best of our knowledge, is the first of its kind. Under this setting, more real world tasks can be studied because our attack assumes no prior knowledge about (1): the model architecture of the GNN we are attacking; (2): the model's gradients; (3): the output logits of the target GNN model. Our attack is based on an existing edge perturbation attack, from which we restrict the optimization process to formulate a node injection attack. In the work, we will evaluate the performance of the attack using three datasets, COIL-DEL, IMDB-BINARY, and NCI1.

摘要: 虽然图神经网络在包括图分类和节点分类在内的许多现实任务中都取得了最先进的性能，但最近的研究表明，它们也非常容易受到对手攻击。以往的工作大多集中在不切实际的白盒场景下对节点分类网络的攻击。在这项工作中，我们将提出一种针对图神经网络的无目标硬标签黑盒节点注入攻击，据我们所知，这是此类攻击中的第一次。在这种情况下，我们可以研究更多现实世界的任务，因为我们的攻击不假设关于(1)：我们正在攻击的GNN的模型体系结构；(2)：模型的梯度；(3)：目标GNN模型的输出逻辑。我们的攻击是基于已有的边扰动攻击，从该边扰动攻击出发，我们限制了优化过程，形成了节点注入攻击。在工作中，我们将使用三个数据集COIL-DEL、IMDB-BINARY和NCI1来评估攻击的性能。



## **5. A Survey of Adversarial CAPTCHAs on its History, Classification and Generation**

对抗性验证码的历史沿革、分类及产生 cs.CR

Submitted to ACM Computing Surveys (Under Review)

**SubmitDate**: 2023-11-22    [abs](http://arxiv.org/abs/2311.13233v1) [paper-pdf](http://arxiv.org/pdf/2311.13233v1)

**Authors**: Zisheng Xu, Qiao Yan, F. Richard Yu, Victor C. M. Leung

**Abstract**: Completely Automated Public Turing test to tell Computers and Humans Apart, short for CAPTCHA, is an essential and relatively easy way to defend against malicious attacks implemented by bots. The security and usability trade-off limits the use of massive geometric transformations to interfere deep model recognition and deep models even outperformed humans in complex CAPTCHAs. The discovery of adversarial examples provides an ideal solution to the security and usability trade-off by integrating adversarial examples and CAPTCHAs to generate adversarial CAPTCHAs that can fool the deep models. In this paper, we extend the definition of adversarial CAPTCHAs and propose a classification method for adversarial CAPTCHAs. Then we systematically review some commonly used methods to generate adversarial examples and methods that are successfully used to generate adversarial CAPTCHAs. Also, we analyze some defense methods that can be used to defend adversarial CAPTCHAs, indicating potential threats to adversarial CAPTCHAs. Finally, we discuss some possible future research directions for adversarial CAPTCHAs at the end of this paper.

摘要: 全自动公共图灵测试区分计算机和人类，简称验证码，是防御机器人实施的恶意攻击的一种基本且相对容易的方法。安全性和可用性的权衡限制了使用大规模几何变换来干扰深度模型识别，而深度模型在复杂验证码中的表现甚至超过了人类。对抗性实例的发现通过集成对抗性实例和验证码来生成可以欺骗深层模型的对抗性验证码，从而为安全性和可用性的权衡提供了一个理想的解决方案。本文扩展了对抗性验证码的定义，提出了一种对抗性验证码的分类方法。然后，我们系统地回顾了一些常用的生成对抗性实例的方法以及成功地生成对抗性验证码的方法。此外，我们还分析了一些可用于防御对抗性验证码的防御方法，指出了对抗性验证码的潜在威胁。最后，我们讨论了对抗性验证码未来可能的研究方向。



## **6. HINT: Healthy Influential-Noise based Training to Defend against Data Poisoning Attacks**

提示：健康影响-基于噪音的培训可防御数据中毒攻击 cs.LG

**SubmitDate**: 2023-11-21    [abs](http://arxiv.org/abs/2309.08549v3) [paper-pdf](http://arxiv.org/pdf/2309.08549v3)

**Authors**: Minh-Hao Van, Alycia N. Carey, Xintao Wu

**Abstract**: While numerous defense methods have been proposed to prohibit potential poisoning attacks from untrusted data sources, most research works only defend against specific attacks, which leaves many avenues for an adversary to exploit. In this work, we propose an efficient and robust training approach to defend against data poisoning attacks based on influence functions, named Healthy Influential-Noise based Training. Using influence functions, we craft healthy noise that helps to harden the classification model against poisoning attacks without significantly affecting the generalization ability on test data. In addition, our method can perform effectively when only a subset of the training data is modified, instead of the current method of adding noise to all examples that has been used in several previous works. We conduct comprehensive evaluations over two image datasets with state-of-the-art poisoning attacks under different realistic attack scenarios. Our empirical results show that HINT can efficiently protect deep learning models against the effect of both untargeted and targeted poisoning attacks.

摘要: 虽然已经提出了许多防御方法来阻止来自不受信任的数据源的潜在中毒攻击，但大多数研究工作只防御特定的攻击，这给对手留下了许多可以利用的途径。在这项工作中，我们提出了一种基于影响函数的高效、健壮的数据中毒攻击训练方法，即基于健康影响噪声的训练方法。利用影响函数构造健康噪声，在不显著影响测试数据泛化能力的情况下，有助于加强分类模型对中毒攻击的抵抗能力。此外，我们的方法可以在只修改训练数据的子集的情况下有效地执行，而不是在以前的几个工作中使用的向所有样本添加噪声的方法。在不同的真实攻击场景下，我们对两个具有最新技术的中毒攻击的图像数据集进行了综合评估。我们的实验结果表明，提示可以有效地保护深度学习模型免受非定向和定向中毒攻击的影响。



## **7. Epsilon*: Privacy Metric for Machine Learning Models**

Epsilon*：机器学习模型的隐私度量 cs.LG

**SubmitDate**: 2023-11-21    [abs](http://arxiv.org/abs/2307.11280v2) [paper-pdf](http://arxiv.org/pdf/2307.11280v2)

**Authors**: Diana M. Negoescu, Humberto Gonzalez, Saad Eddin Al Orjany, Jilei Yang, Yuliia Lut, Rahul Tandra, Xiaowen Zhang, Xinyi Zheng, Zach Douglas, Vidita Nolkha, Parvez Ahammad, Gennady Samorodnitsky

**Abstract**: We introduce Epsilon*, a new privacy metric for measuring the privacy risk of a single model instance prior to, during, or after deployment of privacy mitigation strategies. The metric requires only black-box access to model predictions, does not require training data re-sampling or model re-training, and can be used to measure the privacy risk of models not trained with differential privacy. Epsilon* is a function of true positive and false positive rates in a hypothesis test used by an adversary in a membership inference attack. We distinguish between quantifying the privacy loss of a trained model instance, which we refer to as empirical privacy, and quantifying the privacy loss of the training mechanism which produces this model instance. Existing approaches in the privacy auditing literature provide lower bounds for the latter, while our metric provides an empirical lower bound for the former by relying on an (${\epsilon}$, ${\delta}$)-type of quantification of the privacy of the trained model instance. We establish a relationship between these lower bounds and show how to implement Epsilon* to avoid numerical and noise amplification instability. We further show in experiments on benchmark public data sets that Epsilon* is sensitive to privacy risk mitigation by training with differential privacy (DP), where the value of Epsilon* is reduced by up to 800% compared to the Epsilon* values of non-DP trained baseline models. This metric allows privacy auditors to be independent of model owners, and enables visualizing the privacy-utility landscape to make informed decisions regarding the trade-offs between model privacy and utility.

摘要: 我们引入了Epperity *，这是一种新的隐私度量，用于在部署隐私缓解策略之前、期间或之后测量单个模型实例的隐私风险。该指标只需要黑盒访问模型预测，不需要训练数据重新采样或模型重新训练，并且可以用于测量未使用差分隐私训练的模型的隐私风险。Epperion * 是在成员推断攻击中由对手使用的假设检验中的真阳性率和假阳性率的函数。我们区分量化训练模型实例的隐私损失（我们称之为经验隐私）和量化产生此模型实例的训练机制的隐私损失。在隐私审计文献中的现有方法提供了后者的下限，而我们的度量提供了一个经验的下限，前者依赖于（${\delta}$，${\delta}$）类型的量化训练模型实例的隐私。我们建立了这些下限之间的关系，并展示了如何实现Eppery *，以避免数值和噪声放大不稳定。我们在基准公共数据集上的实验中进一步表明，通过使用差分隐私（DP）进行训练，Epsilon* 对隐私风险缓解敏感，与非DP训练的基线模型的Epsilon* 值相比，Epsilon* 值降低了800%。该指标允许隐私审计员独立于模型所有者，并使隐私-效用景观可视化，以便就模型隐私和效用之间的权衡做出明智的决策。



## **8. Is your vote truly secret? Ballot Secrecy iff Ballot Independence: Proving necessary conditions and analysing case studies**

你的投票真的是秘密的吗？选票保密性与选票独立性：必要条件证明与案例分析 cs.CR

**SubmitDate**: 2023-11-21    [abs](http://arxiv.org/abs/2311.12977v1) [paper-pdf](http://arxiv.org/pdf/2311.12977v1)

**Authors**: Aida Manzano Kharman, Ben Smyth, Freddie Page

**Abstract**: We formalise definitions of ballot secrecy and ballot independence by Smyth, JCS'21 as indistinguishability games in the computational model of security. These definitions improve upon Smyth, draft '21 to consider a wider class of voting systems. Both Smyth, JCS'21 and Smyth, draft '21 improve on earlier works by considering a more realistic adversary model wherein they have access to the ballot collection. We prove that ballot secrecy implies ballot independence. We say ballot independence holds if a system has non-malleable ballots. We construct games for ballot secrecy and non-malleability and show that voting schemes with malleable ballots do not preserve ballot secrecy. We demonstrate that Helios does not satisfy our definition of ballot secrecy. Furthermore, the Python framework we constructed for our case study shows that if an attack exists against non-malleability, this attack can be used to break ballot secrecy.

摘要: 我们正式定义的投票保密性和投票独立性的Smyth，JCS'21作为不可分割的游戏在计算模型的安全。这些定义改进了Smyth的'21草案，考虑了更广泛的投票系统。史密斯，JCS'21和史密斯，草案'21通过考虑一个更现实的对手模型，其中他们有机会获得选票收集改善早期的作品。我们证明了投票保密意味着投票独立性。如果一个系统有不可延展的选票，我们说选票独立性成立。我们构建游戏的选票保密性和不可延展性，并表明，投票计划与可延展的选票不保持选票的保密性。我们证明，太阳神不满足我们的定义的投票保密。此外，我们为案例研究构建的Python框架表明，如果存在针对不可延展性的攻击，则该攻击可用于破坏投票保密性。



## **9. Iris Presentation Attack: Assessing the Impact of Combining Vanadium Dioxide Films with Artificial Eyes**

虹膜呈现攻击：评估二氧化钒薄膜与假眼结合的影响 cs.CV

**SubmitDate**: 2023-11-21    [abs](http://arxiv.org/abs/2311.12773v1) [paper-pdf](http://arxiv.org/pdf/2311.12773v1)

**Authors**: Darshika Jauhari, Renu Sharma, Cunjian Chen, Nelson Sepulveda, Arun Ross

**Abstract**: Iris recognition systems, operating in the near infrared spectrum (NIR), have demonstrated vulnerability to presentation attacks, where an adversary uses artifacts such as cosmetic contact lenses, artificial eyes or printed iris images in order to circumvent the system. At the same time, a number of effective presentation attack detection (PAD) methods have been developed. These methods have demonstrated success in detecting artificial eyes (e.g., fake Van Dyke eyes) as presentation attacks. In this work, we seek to alter the optical characteristics of artificial eyes by affixing Vanadium Dioxide (VO2) films on their surface in various spatial configurations. VO2 films can be used to selectively transmit NIR light and can, therefore, be used to regulate the amount of NIR light from the object that is captured by the iris sensor. We study the impact of such images produced by the sensor on two state-of-the-art iris PA detection methods. We observe that the addition of VO2 films on the surface of artificial eyes can cause the PA detection methods to misclassify them as bonafide eyes in some cases. This represents a vulnerability that must be systematically analyzed and effectively addressed.

摘要: 运行在近红外光谱(NIR)中的虹膜识别系统已经显示出对呈现攻击的脆弱性，在呈现攻击中，对手使用化妆品隐形眼镜、假眼或印刷虹膜图像等人工制品来绕过系统。与此同时，一些有效的呈现攻击检测(PAD)方法已经被开发出来。这些方法已经成功地将假眼(例如，假Van Dyke眼睛)检测为呈现攻击。在这项工作中，我们试图通过在人工眼表面以不同的空间构型粘贴二氧化钒(VO2)薄膜来改变其光学特性。VO2薄膜可以用来选择性地传输近红外光，因此可以用来调节来自虹膜传感器捕获的物体的近红外光的量。我们研究了传感器产生的这种图像对两种最先进的虹膜PA检测方法的影响。我们观察到，在假眼表面添加VO2薄膜会导致PA检测方法在某些情况下将其误认为是真眼。这是一个必须系统分析和有效解决的漏洞。



## **10. Attention Deficit is Ordered! Fooling Deformable Vision Transformers with Collaborative Adversarial Patches**

注意力缺陷是命中注定的！用协同对抗性补丁愚弄可变形视觉变形器 cs.CV

9 pages, 10 figures

**SubmitDate**: 2023-11-21    [abs](http://arxiv.org/abs/2311.12914v1) [paper-pdf](http://arxiv.org/pdf/2311.12914v1)

**Authors**: Quazi Mishkatul Alam, Bilel Tarchoun, Ihsen Alouani, Nael Abu-Ghazaleh

**Abstract**: The latest generation of transformer-based vision models have proven to be superior to Convolutional Neural Network (CNN)-based models across several vision tasks, largely attributed to their remarkable prowess in relation modeling. Deformable vision transformers significantly reduce the quadratic complexity of modeling attention by using sparse attention structures, enabling them to be used in larger scale applications such as multi-view vision systems. Recent work demonstrated adversarial attacks against transformers; we show that these attacks do not transfer to deformable transformers due to their sparse attention structure. Specifically, attention in deformable transformers is modeled using pointers to the most relevant other tokens. In this work, we contribute for the first time adversarial attacks that manipulate the attention of deformable transformers, distracting them to focus on irrelevant parts of the image. We also develop new collaborative attacks where a source patch manipulates attention to point to a target patch that adversarially attacks the system. In our experiments, we find that only 1% patched area of the input field can lead to 0% AP. We also show that the attacks provide substantial versatility to support different attacker scenarios because of their ability to redirect attention under the attacker control.

摘要: 最新一代的基于变压器的视觉模型已被证明在几个视觉任务上优于基于卷积神经网络(CNN)的模型，这在很大程度上归功于它们在关系建模方面的非凡能力。可变形视觉转换器通过使用稀疏注意力结构显著降低了建模注意力的二次方复杂性，使其能够用于更大规模的应用，如多视角视觉系统。最近的工作证明了针对变压器的对抗性攻击；我们表明，由于其稀疏的注意结构，这些攻击不会转移到可变形的变压器上。具体地说，在可变形转换器中的注意力是使用指向最相关的其他标记的指针来建模的。在这项工作中，我们第一次贡献了对抗性攻击，操纵变形变形者的注意力，分散他们的注意力，专注于图像中不相关的部分。我们还开发了新的协作性攻击，其中源补丁操纵注意力指向相反攻击系统的目标补丁。在我们的实验中，我们发现只有1%的输入场修补面积就可以导致0%的AP。我们还表明，攻击提供了相当多的多功能性来支持不同的攻击者场景，因为它们能够在攻击者的控制下重新定向注意力。



## **11. BrainWash: A Poisoning Attack to Forget in Continual Learning**

洗脑：在持续学习中忘记的毒药攻击 cs.LG

**SubmitDate**: 2023-11-21    [abs](http://arxiv.org/abs/2311.11995v2) [paper-pdf](http://arxiv.org/pdf/2311.11995v2)

**Authors**: Ali Abbasi, Parsa Nooralinejad, Hamed Pirsiavash, Soheil Kolouri

**Abstract**: Continual learning has gained substantial attention within the deep learning community, offering promising solutions to the challenging problem of sequential learning. Yet, a largely unexplored facet of this paradigm is its susceptibility to adversarial attacks, especially with the aim of inducing forgetting. In this paper, we introduce "BrainWash," a novel data poisoning method tailored to impose forgetting on a continual learner. By adding the BrainWash noise to a variety of baselines, we demonstrate how a trained continual learner can be induced to forget its previously learned tasks catastrophically, even when using these continual learning baselines. An important feature of our approach is that the attacker requires no access to previous tasks' data and is armed merely with the model's current parameters and the data belonging to the most recent task. Our extensive experiments highlight the efficacy of BrainWash, showcasing degradation in performance across various regularization-based continual learning methods.

摘要: 持续学习在深度学习界得到了广泛的关注，为顺序学习这一具有挑战性的问题提供了有希望的解决方案。然而，这一范式的一个很大程度上没有被探索的方面是它对敌意攻击的敏感性，特别是以诱导遗忘为目的。在这篇文章中，我们介绍了“洗脑”，一种新的数据中毒方法，专门为不断学习的人强加遗忘。通过将洗脑噪声添加到各种基线中，我们演示了如何诱导训练有素的持续学习者灾难性地忘记其先前学习的任务，即使使用这些持续学习基线也是如此。我们方法的一个重要特征是攻击者不需要访问以前任务的数据，并且只用模型的当前参数和属于最近任务的数据武装起来。我们广泛的实验突出了洗脑的有效性，展示了各种基于正则化的持续学习方法在表现上的下降。



## **12. Attacking Motion Planners Using Adversarial Perception Errors**

使用对抗性感知错误攻击动作规划者 cs.RO

**SubmitDate**: 2023-11-21    [abs](http://arxiv.org/abs/2311.12722v1) [paper-pdf](http://arxiv.org/pdf/2311.12722v1)

**Authors**: Jonathan Sadeghi, Nicholas A. Lord, John Redford, Romain Mueller

**Abstract**: Autonomous driving (AD) systems are often built and tested in a modular fashion, where the performance of different modules is measured using task-specific metrics. These metrics should be chosen so as to capture the downstream impact of each module and the performance of the system as a whole. For example, high perception quality should enable prediction and planning to be performed safely. Even though this is true in general, we show here that it is possible to construct planner inputs that score very highly on various perception quality metrics but still lead to planning failures. In an analogy to adversarial attacks on image classifiers, we call such inputs \textbf{adversarial perception errors} and show they can be systematically constructed using a simple boundary-attack algorithm. We demonstrate the effectiveness of this algorithm by finding attacks for two different black-box planners in several urban and highway driving scenarios using the CARLA simulator. Finally, we analyse the properties of these attacks and show that they are isolated in the input space of the planner, and discuss their implications for AD system deployment and testing.

摘要: 自动驾驶(AD)系统通常以模块化的方式构建和测试，其中不同模块的性能是使用特定于任务的指标来衡量的。应选择这些指标，以便捕获每个模块的下游影响和整个系统的性能。例如，高感知质量应该使预测和规划能够安全地执行。尽管这在总体上是正确的，但我们在这里展示了构建计划员输入是可能的，这些输入在各种感知质量指标上得分非常高，但仍然会导致计划失败。在类似于对图像分类器的对抗性攻击中，我们将这种输入称为Textbf(对抗性感知错误)，并证明了它们可以使用简单的边界攻击算法来系统地构造。我们通过使用CALA模拟器在几个城市和高速公路驾驶场景中发现针对两个不同的黑盒规划者的攻击来证明该算法的有效性。最后，我们分析了这些攻击的性质，表明它们在规划器的输入空间中是孤立的，并讨论了它们对AD系统部署和测试的启示。



## **13. Differentially Private Optimizers Can Learn Adversarially Robust Models**

不同的私有优化器可以学习相反的健壮模型 cs.LG

**SubmitDate**: 2023-11-21    [abs](http://arxiv.org/abs/2211.08942v2) [paper-pdf](http://arxiv.org/pdf/2211.08942v2)

**Authors**: Yuan Zhang, Zhiqi Bu

**Abstract**: Machine learning models have shone in a variety of domains and attracted increasing attention from both the security and the privacy communities. One important yet worrying question is: Will training models under the differential privacy (DP) constraint have an unfavorable impact on their adversarial robustness? While previous works have postulated that privacy comes at the cost of worse robustness, we give the first theoretical analysis to show that DP models can indeed be robust and accurate, even sometimes more robust than their naturally-trained non-private counterparts. We observe three key factors that influence the privacy-robustness-accuracy tradeoff: (1) hyper-parameters for DP optimizers are critical; (2) pre-training on public data significantly mitigates the accuracy and robustness drop; (3) choice of DP optimizers makes a difference. With these factors set properly, we achieve 90\% natural accuracy, 72\% robust accuracy ($+9\%$ than the non-private model) under $l_2(0.5)$ attack, and 69\% robust accuracy ($+16\%$ than the non-private model) with pre-trained SimCLRv2 model under $l_\infty(4/255)$ attack on CIFAR10 with $\epsilon=2$. In fact, we show both theoretically and empirically that DP models are Pareto optimal on the accuracy-robustness tradeoff. Empirically, the robustness of DP models is consistently observed across various datasets and models. We believe our encouraging results are a significant step towards training models that are private as well as robust.

摘要: 机器学习模型已经在各个领域大放异彩，越来越受到安全和隐私界的关注。一个重要但令人担忧的问题是：差异隐私(DP)约束下的训练模型是否会对其对抗健壮性产生不利影响？虽然以前的研究假设隐私是以更差的稳健性为代价的，但我们首次给出了理论分析，表明DP模型确实可以是健壮和准确的，有时甚至比自然训练的非私有模型更健壮。我们观察到影响隐私-稳健性-准确度权衡的三个关键因素：(1)DP优化器的超参数至关重要；(2)对公共数据的预训练显著缓解了准确率和稳健性下降；(3)DP优化器的选择产生了影响。在适当设置这些因素的情况下，我们在$L_2(0.5)$攻击下获得了90%的自然精度和72%的稳健精度(比非私有模型高出9美元)，在$L(4/2 5 5)美元攻击下，用预先训练的SimCLRv2模型获得了69%的稳健精度(比非私有模型高出1 6美元).事实上，我们在理论和经验上都证明了DP模型在精度和稳健性之间的权衡是帕累托最优的。从经验上看，DP模型的稳健性在不同的数据集和模型中都得到了一致的观察。我们认为，我们令人鼓舞的结果是朝着私人和稳健的培训模式迈出了重要的一步。



## **14. Open Sesame! Universal Black Box Jailbreaking of Large Language Models**

芝麻开门！大型语言模型的通用黑盒越狱 cs.CL

**SubmitDate**: 2023-11-21    [abs](http://arxiv.org/abs/2309.01446v3) [paper-pdf](http://arxiv.org/pdf/2309.01446v3)

**Authors**: Raz Lapid, Ron Langberg, Moshe Sipper

**Abstract**: Large language models (LLMs), designed to provide helpful and safe responses, often rely on alignment techniques to align with user intent and social guidelines. Unfortunately, this alignment can be exploited by malicious actors seeking to manipulate an LLM's outputs for unintended purposes. In this paper we introduce a novel approach that employs a genetic algorithm (GA) to manipulate LLMs when model architecture and parameters are inaccessible. The GA attack works by optimizing a universal adversarial prompt that -- when combined with a user's query -- disrupts the attacked model's alignment, resulting in unintended and potentially harmful outputs. Our novel approach systematically reveals a model's limitations and vulnerabilities by uncovering instances where its responses deviate from expected behavior. Through extensive experiments we demonstrate the efficacy of our technique, thus contributing to the ongoing discussion on responsible AI development by providing a diagnostic tool for evaluating and enhancing alignment of LLMs with human intent. To our knowledge this is the first automated universal black box jailbreak attack.

摘要: 大型语言模型(LLM)旨在提供有用和安全的响应，它们通常依靠对齐技术来与用户意图和社交指南保持一致。遗憾的是，恶意行为者可能会利用这种对齐方式来操纵LLM的输出，以达到非预期目的。在本文中，我们介绍了一种新的方法，即在模型结构和参数不可访问的情况下，使用遗传算法(GA)来操作LLM。GA攻击的工作原理是优化一个通用的对抗性提示，当与用户的查询结合在一起时，会扰乱被攻击模型的对齐，导致意外的和潜在的有害输出。我们的新方法通过揭示模型响应偏离预期行为的实例，系统地揭示了模型的局限性和漏洞。通过广泛的实验，我们展示了我们技术的有效性，从而通过提供一种诊断工具来评估和增强LLM与人类意图的一致性，从而为正在进行的关于负责任的人工智能开发的讨论做出贡献。据我们所知，这是第一次自动通用黑匣子越狱攻击。



## **15. Beyond Labeling Oracles: What does it mean to steal ML models?**

除了给甲骨文贴标签：窃取ML模型意味着什么？ cs.LG

**SubmitDate**: 2023-11-21    [abs](http://arxiv.org/abs/2310.01959v2) [paper-pdf](http://arxiv.org/pdf/2310.01959v2)

**Authors**: Avital Shafran, Ilia Shumailov, Murat A. Erdogdu, Nicolas Papernot

**Abstract**: Model extraction attacks are designed to steal trained models with only query access, as is often provided through APIs that ML-as-a-Service providers offer. ML models are expensive to train, in part because data is hard to obtain, and a primary incentive for model extraction is to acquire a model while incurring less cost than training from scratch. Literature on model extraction commonly claims or presumes that the attacker is able to save on both data acquisition and labeling costs. We show that the attacker often does not. This is because current attacks implicitly rely on the adversary being able to sample from the victim model's data distribution. We thoroughly evaluate factors influencing the success of model extraction. We discover that prior knowledge of the attacker, i.e. access to in-distribution data, dominates other factors like the attack policy the adversary follows to choose which queries to make to the victim model API. Thus, an adversary looking to develop an equally capable model with a fixed budget has little practical incentive to perform model extraction, since for the attack to work they need to collect in-distribution data, saving only on the cost of labeling. With low labeling costs in the current market, the usefulness of such attacks is questionable. Ultimately, we demonstrate that the effect of prior knowledge needs to be explicitly decoupled from the attack policy. To this end, we propose a benchmark to evaluate attack policy directly.

摘要: 模型提取攻击旨在窃取仅具有查询访问权限的训练模型，这通常是通过ML-as-a-Service提供商提供的API提供的。ML模型的训练成本很高，部分原因是数据很难获得，而模型提取的主要动机是在获得模型的同时产生比从头开始培训更少的成本。有关模型提取的文献通常声称或假设攻击者能够节省数据获取和标记成本。我们发现攻击者通常不会这样做。这是因为当前的攻击隐含地依赖于对手能够从受害者模型的数据分布中进行采样。我们对影响模型提取成功的因素进行了深入的评估。我们发现，攻击者的先验知识，即对分发内数据的访问，主导了其他因素，如攻击者选择对受害者模型API进行哪些查询所遵循的攻击策略。因此，希望开发具有固定预算的同等能力的模型的对手几乎没有执行模型提取的实际动机，因为要使攻击发挥作用，他们需要收集分发内数据，这只节省了标记成本。由于当前市场的标签成本较低，此类攻击的用处值得怀疑。最终，我们证明了先验知识的影响需要与攻击策略明确地分离。为此，我们提出了一个直接评估攻击策略的基准。



## **16. Malicious URL Detection via Pretrained Language Model Guided Multi-Level Feature Attention Network**

基于预训练语言模型引导的多级特征注意力网络的恶意URL检测 cs.CR

11 pages, 7 figures

**SubmitDate**: 2023-11-21    [abs](http://arxiv.org/abs/2311.12372v1) [paper-pdf](http://arxiv.org/pdf/2311.12372v1)

**Authors**: Ruitong Liu, Yanbin Wang, Haitao Xu, Zhan Qin, Yiwei Liu, Zheng Cao

**Abstract**: The widespread use of the Internet has revolutionized information retrieval methods. However, this transformation has also given rise to a significant cybersecurity challenge: the rapid proliferation of malicious URLs, which serve as entry points for a wide range of cyber threats. In this study, we present an efficient pre-training model-based framework for malicious URL detection. Leveraging the subword and character-aware pre-trained model, CharBERT, as our foundation, we further develop three key modules: hierarchical feature extraction, layer-aware attention, and spatial pyramid pooling. The hierarchical feature extraction module follows the pyramid feature learning principle, extracting multi-level URL embeddings from the different Transformer layers of CharBERT. Subsequently, the layer-aware attention module autonomously learns connections among features at various hierarchical levels and allocates varying weight coefficients to each level of features. Finally, the spatial pyramid pooling module performs multiscale downsampling on the weighted multi-level feature pyramid, achieving the capture of local features as well as the aggregation of global features. The proposed method has been extensively validated on multiple public datasets, demonstrating a significant improvement over prior works, with the maximum accuracy gap reaching 8.43% compared to the previous state-of-the-art method. Additionally, we have assessed the model's generalization and robustness in scenarios such as cross-dataset evaluation and adversarial attacks. Finally, we conducted real-world case studies on the active phishing URLs.

摘要: 因特网的广泛使用使信息检索方法发生了革命性的变化。然而，这种转变也带来了一个重大的网络安全挑战：恶意URL的迅速扩散，成为各种网络威胁的入口。在这项研究中，我们提出了一个有效的基于预训练模型的恶意URL检测框架。利用子字和字符感知预训练模型CharBERT作为我们的基础，我们进一步开发了三个关键模块：分层特征提取，层感知注意力和空间金字塔池。分层特征提取模块遵循金字塔特征学习原理，从CharBERT的不同Transformer层提取多级URL嵌入。随后，层感知注意力模块自主地学习在各个层次级别的特征之间的连接，并为每个级别的特征分配不同的权重系数。最后，空间金字塔池化模块对加权后的多层次特征金字塔进行多尺度下采样，实现局部特征的捕获和全局特征的聚合。所提出的方法已经在多个公共数据集上进行了广泛的验证，与以前的工作相比有了显着的改进，与以前的最先进的方法相比，最大精度差距达到8.43%。此外，我们还评估了模型在跨数据集评估和对抗性攻击等场景中的泛化和鲁棒性。最后，我们对活跃的钓鱼URL进行了真实案例研究。



## **17. Rethinking the Backward Propagation for Adversarial Transferability**

关于对抗性转移的后向传播的再思考 cs.CV

Accepted by NeurIPS 2023

**SubmitDate**: 2023-11-21    [abs](http://arxiv.org/abs/2306.12685v3) [paper-pdf](http://arxiv.org/pdf/2306.12685v3)

**Authors**: Xiaosen Wang, Kangheng Tong, Kun He

**Abstract**: Transfer-based attacks generate adversarial examples on the surrogate model, which can mislead other black-box models without access, making it promising to attack real-world applications. Recently, several works have been proposed to boost adversarial transferability, in which the surrogate model is usually overlooked. In this work, we identify that non-linear layers (e.g., ReLU, max-pooling, etc.) truncate the gradient during backward propagation, making the gradient w.r.t. input image imprecise to the loss function. We hypothesize and empirically validate that such truncation undermines the transferability of adversarial examples. Based on these findings, we propose a novel method called Backward Propagation Attack (BPA) to increase the relevance between the gradient w.r.t. input image and loss function so as to generate adversarial examples with higher transferability. Specifically, BPA adopts a non-monotonic function as the derivative of ReLU and incorporates softmax with temperature to smooth the derivative of max-pooling, thereby mitigating the information loss during the backward propagation of gradients. Empirical results on the ImageNet dataset demonstrate that not only does our method substantially boost the adversarial transferability, but it is also general to existing transfer-based attacks. Code is available at https://github.com/Trustworthy-AI-Group/RPA.

摘要: 基于传输的攻击在代理模型上生成敌意示例，这可能会在无法访问的情况下误导其他黑盒模型，使其有可能攻击现实世界的应用程序。最近，已有一些关于提高对抗性转移能力的工作被提出，但其中的代理模型往往被忽视。在这项工作中，我们确定了非线性层(例如，RELU、最大池等)。在反向传播过程中截断梯度，使梯度w.r.t.输入图像不精确到损失函数。我们假设和经验验证，这种截断破坏了对抗性例子的可转移性。基于这些发现，我们提出了一种新的方法，称为反向传播攻击(BPA)，以提高梯度之间的相关性。输入图像和损失函数，生成具有较高可转移性的对抗性实例。具体地说，BPA采用非单调函数作为RELU的导数，并将Softmax与温度相结合以平滑max-Pooling的导数，从而减少了梯度反向传播过程中的信息损失。在ImageNet数据集上的实验结果表明，我们的方法不仅大大提高了攻击的对抗性可转移性，而且对现有的基于传输的攻击也是通用的。代码可在https://github.com/Trustworthy-AI-Group/RPA.上找到



## **18. Resilient Control of Networked Microgrids using Vertical Federated Reinforcement Learning: Designs and Real-Time Test-Bed Validations**

基于垂直联邦强化学习的网络化微电网弹性控制：设计与实时试验台验证 eess.SY

10 pages, 7 figures

**SubmitDate**: 2023-11-21    [abs](http://arxiv.org/abs/2311.12264v1) [paper-pdf](http://arxiv.org/pdf/2311.12264v1)

**Authors**: Sayak Mukherjee, Ramij R. Hossain, Sheik M. Mohiuddin, Yuan Liu, Wei Du, Veronica Adetola, Rohit A. Jinsiwale, Qiuhua Huang, Tianzhixi Yin, Ankit Singhal

**Abstract**: Improving system-level resiliency of networked microgrids is an important aspect with increased population of inverter-based resources (IBRs). This paper (1) presents resilient control design in presence of adversarial cyber-events, and proposes a novel federated reinforcement learning (Fed-RL) approach to tackle (a) model complexities, unknown dynamical behaviors of IBR devices, (b) privacy issues regarding data sharing in multi-party-owned networked grids, and (2) transfers learned controls from simulation to hardware-in-the-loop test-bed, thereby bridging the gap between simulation and real world. With these multi-prong objectives, first, we formulate a reinforcement learning (RL) training setup generating episodic trajectories with adversaries (attack signal) injected at the primary controllers of the grid forming (GFM) inverters where RL agents (or controllers) are being trained to mitigate the injected attacks. For networked microgrids, the horizontal Fed-RL method involving distinct independent environments is not appropriate, leading us to develop vertical variant Federated Soft Actor-Critic (FedSAC) algorithm to grasp the interconnected dynamics of networked microgrid. Next, utilizing OpenAI Gym interface, we built a custom simulation set-up in GridLAB-D/HELICS co-simulation platform, named Resilient RL Co-simulation (ResRLCoSIM), to train the RL agents with IEEE 123-bus benchmark test systems comprising 3 interconnected microgrids. Finally, the learned policies in simulation world are transferred to the real-time hardware-in-the-loop test-bed set-up developed using high-fidelity Hypersim platform. Experiments show that the simulator-trained RL controllers produce convincing results with the real-time test-bed set-up, validating the minimization of sim-to-real gap.

摘要: 随着基于逆变器的资源(IBR)数量的增加，提高网络化微电网的系统级弹性是一个重要方面。本文(1)给出了对抗网络事件下的弹性控制设计，并提出了一种新的联邦强化学习(FED-RL)方法来解决(A)模型的复杂性、IBR设备的未知动态行为、(B)与多方拥有的网络网格中的数据共享有关的隐私问题、(2)将学习的控制从仿真转移到半实物试验台，从而弥合了仿真与真实世界之间的差距。有了这些多管齐下的目标，首先，我们制定了一个强化学习(RL)训练设置，生成插曲轨迹，在网格形成(GFM)逆变器的主控制器上注入对手(攻击信号)，在那里RL代理(或控制器)正在接受培训，以减轻注入的攻击。对于网络化微电网，水平FED-RL方法不适用于不同的独立环境，这导致我们开发了垂直可变的联邦软行动者-批评者(FedSAC)算法来掌握网络化微电网的互联动态。接下来，利用OpenAI Gym接口，在GridLAB-D/HELICS联合仿真平台上构建了一个定制的仿真平台，称为弹性RL协同仿真(ResRLCoSIM)，用于在由3个互联微电网组成的IEEE 123母线基准测试系统中训练RL代理。最后，将仿真世界中学习到的策略传输到使用高保真Hypersim平台开发的实时半实物试验台。实验表明，经过模拟器训练的RL控制器在实时试验台的设置下产生了令人信服的结果，验证了模拟与真实之间的差距最小化。



## **19. DefensiveDR: Defending against Adversarial Patches using Dimensionality Reduction**

DefensveDR：使用降维技术防御恶意补丁 cs.CR

**SubmitDate**: 2023-11-20    [abs](http://arxiv.org/abs/2311.12211v1) [paper-pdf](http://arxiv.org/pdf/2311.12211v1)

**Authors**: Nandish Chattopadhyay, Amira Guesmi, Muhammad Abdullah Hanif, Bassem Ouni, Muhammad Shafique

**Abstract**: Adversarial patch-based attacks have shown to be a major deterrent towards the reliable use of machine learning models. These attacks involve the strategic modification of localized patches or specific image areas to deceive trained machine learning models. In this paper, we propose \textit{DefensiveDR}, a practical mechanism using a dimensionality reduction technique to thwart such patch-based attacks. Our method involves projecting the sample images onto a lower-dimensional space while retaining essential information or variability for effective machine learning tasks. We perform this using two techniques, Singular Value Decomposition and t-Distributed Stochastic Neighbor Embedding. We experimentally tune the variability to be preserved for optimal performance as a hyper-parameter. This dimension reduction substantially mitigates adversarial perturbations, thereby enhancing the robustness of the given machine learning model. Our defense is model-agnostic and operates without assumptions about access to model decisions or model architectures, making it effective in both black-box and white-box settings. Furthermore, it maintains accuracy across various models and remains robust against several unseen patch-based attacks. The proposed defensive approach improves the accuracy from 38.8\% (without defense) to 66.2\% (with defense) when performing LaVAN and GoogleAp attacks, which supersedes that of the prominent state-of-the-art like LGS (53.86\%) and Jujutsu (60\%).

摘要: 基于补丁的对抗性攻击已被证明是对机器学习模型的可靠使用的主要威慑。这些攻击涉及对局部补丁或特定图像区域进行战略性修改，以欺骗训练有素的机器学习模型。在本文中，我们提出了一种实用的机制，使用降维技术来阻止这种基于补丁的攻击。我们的方法包括将样本图像投影到较低维空间，同时保留有效机器学习任务的基本信息或可变性。我们使用两种技术来实现这一点，奇异值分解和t分布随机邻居嵌入。我们在实验中调整了作为超级参数保留的可变性，以实现最佳性能。这种降维大大减轻了对抗性扰动，从而增强了给定机器学习模型的稳健性。我们的防御是与模型无关的，并且不需要假设可以访问模型决策或模型架构，因此在黑盒和白盒设置中都有效。此外，它在各种模型中保持准确性，并对几种看不见的基于补丁的攻击保持健壮。当执行Lavan和GoogleAp攻击时，该防御方法的准确率从38.8%(无防御)提高到66.2%(有防御)，取代了LGS(53.86)和Jujutsu(60)等最先进的攻击方法。



## **20. Generating Valid and Natural Adversarial Examples with Large Language Models**

使用大型语言模型生成有效的自然对抗性实例 cs.CL

Submitted to the IEEE for possible publication

**SubmitDate**: 2023-11-20    [abs](http://arxiv.org/abs/2311.11861v1) [paper-pdf](http://arxiv.org/pdf/2311.11861v1)

**Authors**: Zimu Wang, Wei Wang, Qi Chen, Qiufeng Wang, Anh Nguyen

**Abstract**: Deep learning-based natural language processing (NLP) models, particularly pre-trained language models (PLMs), have been revealed to be vulnerable to adversarial attacks. However, the adversarial examples generated by many mainstream word-level adversarial attack models are neither valid nor natural, leading to the loss of semantic maintenance, grammaticality, and human imperceptibility. Based on the exceptional capacity of language understanding and generation of large language models (LLMs), we propose LLM-Attack, which aims at generating both valid and natural adversarial examples with LLMs. The method consists of two stages: word importance ranking (which searches for the most vulnerable words) and word synonym replacement (which substitutes them with their synonyms obtained from LLMs). Experimental results on the Movie Review (MR), IMDB, and Yelp Review Polarity datasets against the baseline adversarial attack models illustrate the effectiveness of LLM-Attack, and it outperforms the baselines in human and GPT-4 evaluation by a significant margin. The model can generate adversarial examples that are typically valid and natural, with the preservation of semantic meaning, grammaticality, and human imperceptibility.

摘要: 基于深度学习的自然语言处理（NLP）模型，特别是预训练的语言模型（PLM），已经被发现容易受到对抗性攻击。然而，许多主流的词级对抗性攻击模型生成的对抗性示例既不有效也不自然，导致语义维护、语法性和人类不可感知性的损失。基于语言理解和生成大型语言模型（LLM）的卓越能力，我们提出了LLM攻击，旨在使用LLM生成有效和自然的对抗性示例。该方法包括两个阶段：词重要性排名（搜索最脆弱的词）和词同义词替换（用从LLM获得的同义词替换它们）。在Movie Review（MR）、IMDB和Yelp Review Polarity数据集上针对基线对抗性攻击模型的实验结果说明了LLM-Attack的有效性，它在人类和GPT-4评估中的表现明显优于基线。该模型可以生成通常有效和自然的对抗性示例，同时保留语义，语法和人类不可感知性。



## **21. Beyond Boundaries: A Comprehensive Survey of Transferable Attacks on AI Systems**

超越边界：对人工智能系统可转移攻击的全面综述 cs.CR

**SubmitDate**: 2023-11-20    [abs](http://arxiv.org/abs/2311.11796v1) [paper-pdf](http://arxiv.org/pdf/2311.11796v1)

**Authors**: Guangjing Wang, Ce Zhou, Yuanda Wang, Bocheng Chen, Hanqing Guo, Qiben Yan

**Abstract**: Artificial Intelligence (AI) systems such as autonomous vehicles, facial recognition, and speech recognition systems are increasingly integrated into our daily lives. However, despite their utility, these AI systems are vulnerable to a wide range of attacks such as adversarial, backdoor, data poisoning, membership inference, model inversion, and model stealing attacks. In particular, numerous attacks are designed to target a particular model or system, yet their effects can spread to additional targets, referred to as transferable attacks. Although considerable efforts have been directed toward developing transferable attacks, a holistic understanding of the advancements in transferable attacks remains elusive. In this paper, we comprehensively explore learning-based attacks from the perspective of transferability, particularly within the context of cyber-physical security. We delve into different domains -- the image, text, graph, audio, and video domains -- to highlight the ubiquitous and pervasive nature of transferable attacks. This paper categorizes and reviews the architecture of existing attacks from various viewpoints: data, process, model, and system. We further examine the implications of transferable attacks in practical scenarios such as autonomous driving, speech recognition, and large language models (LLMs). Additionally, we outline the potential research directions to encourage efforts in exploring the landscape of transferable attacks. This survey offers a holistic understanding of the prevailing transferable attacks and their impacts across different domains.

摘要: 自动驾驶汽车、面部识别和语音识别系统等人工智能(AI)系统越来越多地融入我们的日常生活。然而，尽管这些人工智能系统具有实用性，但它们容易受到各种攻击，如对抗性攻击、后门攻击、数据中毒攻击、成员关系推理攻击、模型反转攻击和模型窃取攻击。具体地说，许多攻击旨在针对特定型号或系统，但其影响可能会扩散到其他目标，称为可转移攻击。尽管已经做出了相当大的努力来开发可转移攻击，但对可转移攻击的进展仍难以全面了解。在本文中，我们从可转移性的角度，特别是在网络-物理安全的背景下，全面地探讨了基于学习的攻击。我们深入研究不同的领域--图像、文本、图形、音频和视频域--以突出可转移攻击的无处不在和普遍存在的性质。本文从数据、过程、模型和系统等不同角度对现有攻击的体系结构进行了分类和回顾。我们进一步研究了可转移攻击在实际场景中的含义，如自动驾驶、语音识别和大型语言模型(LLM)。此外，我们概述了潜在的研究方向，以鼓励在探索可转移攻击的图景方面的努力。这项调查提供了对流行的可转移攻击及其跨不同领域的影响的全面了解。



## **22. AdvGen: Physical Adversarial Attack on Face Presentation Attack Detection Systems**

AdvGen：人脸呈现攻击检测系统的物理对抗性攻击 cs.CV

10 pages, 9 figures, Accepted to the International Joint Conference  on Biometrics (IJCB 2023)

**SubmitDate**: 2023-11-20    [abs](http://arxiv.org/abs/2311.11753v1) [paper-pdf](http://arxiv.org/pdf/2311.11753v1)

**Authors**: Sai Amrit Patnaik, Shivali Chansoriya, Anil K. Jain, Anoop M. Namboodiri

**Abstract**: Evaluating the risk level of adversarial images is essential for safely deploying face authentication models in the real world. Popular approaches for physical-world attacks, such as print or replay attacks, suffer from some limitations, like including physical and geometrical artifacts. Recently, adversarial attacks have gained attraction, which try to digitally deceive the learning strategy of a recognition system using slight modifications to the captured image. While most previous research assumes that the adversarial image could be digitally fed into the authentication systems, this is not always the case for systems deployed in the real world. This paper demonstrates the vulnerability of face authentication systems to adversarial images in physical world scenarios. We propose AdvGen, an automated Generative Adversarial Network, to simulate print and replay attacks and generate adversarial images that can fool state-of-the-art PADs in a physical domain attack setting. Using this attack strategy, the attack success rate goes up to 82.01%. We test AdvGen extensively on four datasets and ten state-of-the-art PADs. We also demonstrate the effectiveness of our attack by conducting experiments in a realistic, physical environment.

摘要: 评估敌意图像的风险级别对于在现实世界中安全地部署人脸认证模型至关重要。流行的物理世界攻击方法，如打印或重放攻击，受到一些限制，比如包括物理和几何伪像。最近，敌意攻击越来越受到关注，这种攻击试图通过对捕获的图像进行轻微修改来数字欺骗识别系统的学习策略。虽然以前的大多数研究都假设敌意图像可以数字地输入身份验证系统，但对于现实世界中部署的系统来说，情况并不总是如此。本文论证了人脸认证系统在现实世界场景中对敌意图像的脆弱性。我们提出了一个自动生成的对抗性网络AdvGen来模拟打印和重放攻击，并生成可以在物理域攻击环境中愚弄最先进的PAD的对抗性图像。使用该攻击策略，攻击成功率可达82.01%。我们在四个数据集和十个最先进的PAD上广泛测试AdvGen。我们还通过在现实的物理环境中进行实验来证明我们的攻击的有效性。



## **23. APARATE: Adaptive Adversarial Patch for CNN-based Monocular Depth Estimation for Autonomous Navigation**

APARATE：基于CNN的自主导航单眼深度估计的自适应对抗性补丁 cs.CV

**SubmitDate**: 2023-11-20    [abs](http://arxiv.org/abs/2303.01351v2) [paper-pdf](http://arxiv.org/pdf/2303.01351v2)

**Authors**: Amira Guesmi, Muhammad Abdullah Hanif, Ihsen Alouani, Muhammad Shafique

**Abstract**: In recent times, monocular depth estimation (MDE) has experienced significant advancements in performance, largely attributed to the integration of innovative architectures, i.e., convolutional neural networks (CNNs) and Transformers. Nevertheless, the susceptibility of these models to adversarial attacks has emerged as a noteworthy concern, especially in domains where safety and security are paramount. This concern holds particular weight for MDE due to its critical role in applications like autonomous driving and robotic navigation, where accurate scene understanding is pivotal. To assess the vulnerability of CNN-based depth prediction methods, recent work tries to design adversarial patches against MDE. However, the existing approaches fall short of inducing a comprehensive and substantially disruptive impact on the vision system. Instead, their influence is partial and confined to specific local areas. These methods lead to erroneous depth predictions only within the overlapping region with the input image, without considering the characteristics of the target object, such as its size, shape, and position. In this paper, we introduce a novel adversarial patch named APARATE. This patch possesses the ability to selectively undermine MDE in two distinct ways: by distorting the estimated distances or by creating the illusion of an object disappearing from the perspective of the autonomous system. Notably, APARATE is designed to be sensitive to the shape and scale of the target object, and its influence extends beyond immediate proximity. APARATE, results in a mean depth estimation error surpassing $0.5$, significantly impacting as much as $99\%$ of the targeted region when applied to CNN-based MDE models. Furthermore, it yields a significant error of $0.34$ and exerts substantial influence over $94\%$ of the target region in the context of Transformer-based MDE.

摘要: 近年来，单目深度估计(MDE)在性能上取得了显著的进步，这在很大程度上归功于卷积神经网络(CNN)和变压器等创新体系结构的集成。然而，这些模型对对抗性攻击的易感性已经成为一个值得关注的问题，特别是在安全和安保至上的领域。这一担忧对MDE来说尤为重要，因为它在自动驾驶和机器人导航等应用中扮演着关键角色，在这些应用中，准确的场景理解至关重要。为了评估基于CNN的深度预测方法的脆弱性，最近的工作试图设计对抗MDE的对抗性补丁。然而，现有的方法不能对视觉系统造成全面和实质性的颠覆性影响。相反，他们的影响是部分的，仅限于特定的当地地区。这些方法只在与输入图像重叠的区域内导致错误的深度预测，而没有考虑目标对象的特征，例如其大小、形状和位置。在本文中，我们介绍了一种新的对抗性补丁APARATE。这个补丁能够以两种不同的方式选择性地削弱MDE：通过扭曲估计的距离或通过从自主系统的角度创造物体消失的错觉。值得注意的是，APARATE被设计为对目标对象的形状和比例敏感，其影响超出了直接接近的范围。APARATE，导致平均深度估计误差超过$0.5$，当应用于基于CNN的MDE模型时，显著影响目标区域的$99\$。此外，在基于变压器的MDE的背景下，它产生了$0.34$的显著误差，并对目标区域的$94\$产生了重大影响。



## **24. DAP: A Dynamic Adversarial Patch for Evading Person Detectors**

DAP：一种用于躲避人员检测器的动态对抗补丁 cs.CR

**SubmitDate**: 2023-11-20    [abs](http://arxiv.org/abs/2305.11618v2) [paper-pdf](http://arxiv.org/pdf/2305.11618v2)

**Authors**: Amira Guesmi, Ruitian Ding, Muhammad Abdullah Hanif, Ihsen Alouani, Muhammad Shafique

**Abstract**: Patch-based adversarial attacks were proven to compromise the robustness and reliability of computer vision systems. However, their conspicuous and easily detectable nature challenge their practicality in real-world setting. To address this, recent work has proposed using Generative Adversarial Networks (GANs) to generate naturalistic patches that may not attract human attention. However, such approaches suffer from a limited latent space making it challenging to produce a patch that is efficient, stealthy, and robust to multiple real-world transformations. This paper introduces a novel approach that produces a Dynamic Adversarial Patch (DAP) designed to overcome these limitations. DAP maintains a naturalistic appearance while optimizing attack efficiency and robustness to real-world transformations. The approach involves redefining the optimization problem and introducing a novel objective function that incorporates a similarity metric to guide the patch's creation. Unlike GAN-based techniques, the DAP directly modifies pixel values within the patch, providing increased flexibility and adaptability to multiple transformations. Furthermore, most clothing-based physical attacks assume static objects and ignore the possible transformations caused by non-rigid deformation due to changes in a person's pose. To address this limitation, a 'Creases Transformation' (CT) block is introduced, enhancing the patch's resilience to a variety of real-world distortions. Experimental results demonstrate that the proposed approach outperforms state-of-the-art attacks, achieving a success rate of up to 82.28% in the digital world when targeting the YOLOv7 detector and 65% in the physical world when targeting YOLOv3tiny detector deployed in edge-based smart cameras.

摘要: 基于补丁的对抗性攻击被证明会损害计算机视觉系统的健壮性和可靠性。然而，它们的显着性和易察觉的性质挑战了它们在现实世界中的实用性。为了解决这个问题，最近的研究建议使用生成性对抗网络(GANS)来生成可能不会引起人类注意的自然主义斑块。然而，这些方法的缺点是潜在空间有限，这使得产生高效、隐蔽和对多个现实世界转换具有健壮性的补丁具有挑战性。本文介绍了一种新的生成动态对抗性补丁(DAP)的方法，旨在克服这些局限性。DAP保持了自然的外观，同时优化了攻击效率和对真实世界转换的健壮性。该方法包括重新定义优化问题和引入一个新的目标函数，该目标函数包含一个相似性度量来指导补丁的创建。与基于GaN的技术不同，DAP直接修改贴片内的像素值，提供了更高的灵活性和对多种转换的适应性。此外，大多数基于服装的物理攻击假设静态对象，而忽略了由于人的姿势变化而导致的非刚性变形可能导致的变形。为了解决这一限制，引入了“折痕变换”(CT)块，增强了补丁对各种真实世界扭曲的恢复能力。实验结果表明，该方法比现有的攻击方法具有更高的攻击性能，在数字世界中对YOLOv7探测器的攻击成功率高达82.28%，在物理世界中对部署在基于边缘的智能摄像机中的YOLOv3探测器的攻击成功率高达65%。



## **25. ODDR: Outlier Detection & Dimension Reduction Based Defense Against Adversarial Patches**

ODDR：基于离群点检测和降维的对抗补丁防御 cs.CR

**SubmitDate**: 2023-11-20    [abs](http://arxiv.org/abs/2311.12084v1) [paper-pdf](http://arxiv.org/pdf/2311.12084v1)

**Authors**: Nandish Chattopadhyay, Amira Guesmi, Muhammad Abdullah Hanif, Bassem Ouni, Muhammad Shafique

**Abstract**: Adversarial attacks are a major deterrent towards the reliable use of machine learning models. A powerful type of adversarial attacks is the patch-based attack, wherein the adversarial perturbations modify localized patches or specific areas within the images to deceive the trained machine learning model. In this paper, we introduce Outlier Detection and Dimension Reduction (ODDR), a holistic defense mechanism designed to effectively mitigate patch-based adversarial attacks. In our approach, we posit that input features corresponding to adversarial patches, whether naturalistic or otherwise, deviate from the inherent distribution of the remaining image sample and can be identified as outliers or anomalies. ODDR employs a three-stage pipeline: Fragmentation, Segregation, and Neutralization, providing a model-agnostic solution applicable to both image classification and object detection tasks. The Fragmentation stage parses the samples into chunks for the subsequent Segregation process. Here, outlier detection techniques identify and segregate the anomalous features associated with adversarial perturbations. The Neutralization stage utilizes dimension reduction methods on the outliers to mitigate the impact of adversarial perturbations without sacrificing pertinent information necessary for the machine learning task. Extensive testing on benchmark datasets and state-of-the-art adversarial patches demonstrates the effectiveness of ODDR. Results indicate robust accuracies matching and lying within a small range of clean accuracies (1%-3% for classification and 3%-5% for object detection), with only a marginal compromise of 1%-2% in performance on clean samples, thereby significantly outperforming other defenses.

摘要: 对抗性攻击是对机器学习模型可靠使用的主要威慑。一种强大的对抗性攻击类型是基于补丁的攻击，其中对抗性扰动修改图像中的局部补丁或特定区域以欺骗训练的机器学习模型。在本文中，我们介绍了离群点检测和降维(ODDR)，这是一种全面的防御机制，旨在有效地缓解基于补丁的敌意攻击。在我们的方法中，我们假设对应于对抗性补丁的输入特征，无论是自然的还是其他的，都偏离了剩余图像样本的固有分布，可以被识别为异常值或异常。ODDR采用了三个阶段的流水线：碎片、分离和中和，提供了一种适用于图像分类和目标检测任务的模型无关的解决方案。碎片化阶段将样本解析成块，以用于后续的分离过程。这里，离群点检测技术识别并分离与对抗性扰动相关的异常特征。中和阶段利用对离群值的降维方法来减轻对抗性扰动的影响，而不牺牲机器学习任务所需的相关信息。在基准数据集和最先进的对抗性补丁上的广泛测试证明了ODDR的有效性。结果表明，稳健的精度匹配并位于清洁精度的小范围内(分类为1%-3%，目标检测为3%-5%)，而在清洁样本上的性能仅有1%-2%的边际折衷，因此显著优于其他防御措施。



## **26. Understanding Variation in Subpopulation Susceptibility to Poisoning Attacks**

了解亚群对中毒攻击易感性的变化 cs.LG

18 pages, 11 figures

**SubmitDate**: 2023-11-20    [abs](http://arxiv.org/abs/2311.11544v1) [paper-pdf](http://arxiv.org/pdf/2311.11544v1)

**Authors**: Evan Rose, Fnu Suya, David Evans

**Abstract**: Machine learning is susceptible to poisoning attacks, in which an attacker controls a small fraction of the training data and chooses that data with the goal of inducing some behavior unintended by the model developer in the trained model. We consider a realistic setting in which the adversary with the ability to insert a limited number of data points attempts to control the model's behavior on a specific subpopulation. Inspired by previous observations on disparate effectiveness of random label-flipping attacks on different subpopulations, we investigate the properties that can impact the effectiveness of state-of-the-art poisoning attacks against different subpopulations. For a family of 2-dimensional synthetic datasets, we empirically find that dataset separability plays a dominant role in subpopulation vulnerability for less separable datasets. However, well-separated datasets exhibit more dependence on individual subpopulation properties. We further discover that a crucial subpopulation property is captured by the difference in loss on the clean dataset between the clean model and a target model that misclassifies the subpopulation, and a subpopulation is much easier to attack if the loss difference is small. This property also generalizes to high-dimensional benchmark datasets. For the Adult benchmark dataset, we show that we can find semantically-meaningful subpopulation properties that are related to the susceptibilities of a selected group of subpopulations. The results in this paper are accompanied by a fully interactive web-based visualization of subpopulation poisoning attacks found at https://uvasrg.github.io/visualizing-poisoning

摘要: 机器学习很容易受到中毒攻击，在这种攻击中，攻击者控制着一小部分训练数据，并选择这些数据，目的是在训练的模型中诱导一些模型开发人员意想不到的行为。我们考虑一种现实的设置，在这种情况下，具有插入有限数量数据点的能力的对手试图控制模型在特定子群上的行为。受先前关于随机标签翻转攻击对不同子群的不同有效性的观察的启发，我们调查了可以影响针对不同子群的最新毒化攻击的有效性的属性。对于一类2维合成数据集，我们的经验发现，数据集可分性在较少可分性数据集的子总体脆弱性中起主导作用。然而，分离良好的数据集表现出对单个子总体属性的更多依赖。我们进一步发现，一个关键的子总体属性是通过CLEAN模型和错误分类的目标模型之间在CLEAN数据集上的损失差异来捕捉的，并且如果损失差异很小，子总体更容易受到攻击。此属性也适用于高维基准数据集。对于成人基准数据集，我们表明我们可以找到与选定的一组子群体的易感性相关的语义上有意义的子群体属性。本文的结果伴随着在https://uvasrg.github.io/visualizing-poisoning发现的亚群中毒攻击的完全交互的基于网络的可视化



## **27. Assessing Prompt Injection Risks in 200+ Custom GPTs**

评估200多个定制GPT的即时注射风险 cs.CR

**SubmitDate**: 2023-11-20    [abs](http://arxiv.org/abs/2311.11538v1) [paper-pdf](http://arxiv.org/pdf/2311.11538v1)

**Authors**: Jiahao Yu, Yuhang Wu, Dong Shu, Mingyu Jin, Xinyu Xing

**Abstract**: In the rapidly evolving landscape of artificial intelligence, ChatGPT has been widely used in various applications. The new feature: customization of ChatGPT models by users to cater to specific needs has opened new frontiers in AI utility. However, this study reveals a significant security vulnerability inherent in these user-customized GPTs: prompt injection attacks. Through comprehensive testing of over 200 user-designed GPT models via adversarial prompts, we demonstrate that these systems are susceptible to prompt injections. Through prompt injection, an adversary can not only extract the customized system prompts but also access the uploaded files. This paper provides a first-hand analysis of the prompt injection, alongside the evaluation of the possible mitigation of such attacks. Our findings underscore the urgent need for robust security frameworks in the design and deployment of customizable GPT models. The intent of this paper is to raise awareness and prompt action in the AI community, ensuring that the benefits of GPT customization do not come at the cost of compromised security and privacy.

摘要: 在快速发展的人工智能版图中，ChatGPT已被广泛应用于各种应用。新功能：用户根据特定需求定制ChatGPT型号，开辟了人工智能实用程序的新领域。然而，这项研究揭示了这些用户定制的GPT固有的一个重大安全漏洞：提示注入攻击。通过通过对抗性提示对200多个用户设计的GPT模型进行全面测试，我们证明了这些系统容易受到快速注入的影响。通过提示注入，攻击者不仅可以提取定制的系统提示，还可以访问上传的文件。本文提供了对快速注入的第一手分析，并评估了此类攻击的可能缓解。我们的发现强调了在设计和部署可定制的GPT模型时迫切需要强大的安全框架。本文的目的是提高AI社区的意识并迅速采取行动，确保GPT定制的好处不会以牺牲安全和隐私为代价。



## **28. Token-Level Adversarial Prompt Detection Based on Perplexity Measures and Contextual Information**

基于困惑度量和上下文信息的令牌级敌意提示检测 cs.CL

**SubmitDate**: 2023-11-20    [abs](http://arxiv.org/abs/2311.11509v1) [paper-pdf](http://arxiv.org/pdf/2311.11509v1)

**Authors**: Zhengmian Hu, Gang Wu, Saayan Mitra, Ruiyi Zhang, Tong Sun, Heng Huang, Vishy Swaminathan

**Abstract**: In recent years, Large Language Models (LLM) have emerged as pivotal tools in various applications. However, these models are susceptible to adversarial prompt attacks, where attackers can carefully curate input strings that lead to undesirable outputs. The inherent vulnerability of LLMs stems from their input-output mechanisms, especially when presented with intensely out-of-distribution (OOD) inputs. This paper proposes a token-level detection method to identify adversarial prompts, leveraging the LLM's capability to predict the next token's probability. We measure the degree of the model's perplexity and incorporate neighboring token information to encourage the detection of contiguous adversarial prompt sequences. As a result, we propose two methods: one that identifies each token as either being part of an adversarial prompt or not, and another that estimates the probability of each token being part of an adversarial prompt.

摘要: 近年来，大型语言模型(LLM)已经成为各种应用中的关键工具。然而，这些模型容易受到敌意提示攻击，攻击者可以仔细策划导致不良输出的输入字符串。低成本管理的内在脆弱性源于其投入-产出机制，特别是在出现严重不分布(OOD)投入时。提出了一种令牌级检测方法来识别敌意提示，利用LLM的能力来预测下一个令牌的概率。我们测量模型的困惑程度，并结合相邻令牌信息来鼓励对连续对抗性提示序列的检测。因此，我们提出了两种方法：一种是识别每个令牌是不是对抗性提示的一部分，另一种是估计每个令牌是对抗性提示的一部分的概率。



## **29. Interpretable Computer Vision Models through Adversarial Training: Unveiling the Robustness-Interpretability Connection**

对抗性训练中的可解释计算机视觉模型：揭示稳健性与可解释性之间的联系 cs.CV

13 pages, 19 figures, 6 tables

**SubmitDate**: 2023-11-19    [abs](http://arxiv.org/abs/2307.02500v2) [paper-pdf](http://arxiv.org/pdf/2307.02500v2)

**Authors**: Delyan Boychev

**Abstract**: With the perpetual increase of complexity of the state-of-the-art deep neural networks, it becomes a more and more challenging task to maintain their interpretability. Our work aims to evaluate the effects of adversarial training utilized to produce robust models - less vulnerable to adversarial attacks. It has been shown to make computer vision models more interpretable. Interpretability is as essential as robustness when we deploy the models to the real world. To prove the correlation between these two problems, we extensively examine the models using local feature-importance methods (SHAP, Integrated Gradients) and feature visualization techniques (Representation Inversion, Class Specific Image Generation). Standard models, compared to robust are more susceptible to adversarial attacks, and their learned representations are less meaningful to humans. Conversely, these models focus on distinctive regions of the images which support their predictions. Moreover, the features learned by the robust model are closer to the real ones.

摘要: 随着最新的深度神经网络的复杂性不断增加，保持其可解释性成为一项越来越具有挑战性的任务。我们的工作旨在评估用于产生健壮模型的对抗性训练的效果--较不容易受到对抗性攻击。它已被证明使计算机视觉模型更易于解释。当我们将模型部署到真实世界时，可解释性与健壮性同样重要。为了证明这两个问题之间的相关性，我们使用局部特征重要性方法(Shap，集成梯度)和特征可视化技术(表示反转，类特定图像生成)对模型进行了广泛的检验。与稳健模型相比，标准模型更容易受到对抗性攻击，其学习的表示对人类的意义较小。相反，这些模型关注的是图像中支持其预测的独特区域。此外，稳健模型所学习的特征更接近真实的特征。



## **30. Revisiting and Advancing Adversarial Training Through A Simple Baseline**

通过一条简单的基线重新审视和推进对抗性训练 cs.CV

11 pages, 8 figures

**SubmitDate**: 2023-11-19    [abs](http://arxiv.org/abs/2306.07613v2) [paper-pdf](http://arxiv.org/pdf/2306.07613v2)

**Authors**: Hong Liu

**Abstract**: In this paper, we delve into the essential components of adversarial training which is a pioneering defense technique against adversarial attacks. We indicate that some factors such as the loss function, learning rate scheduler, and data augmentation, which are independent of the model architecture, will influence adversarial robustness and generalization. When these factors are controlled for, we introduce a simple baseline approach, termed SimpleAT, that performs competitively with recent methods and mitigates robust overfitting. We conduct extensive experiments on CIFAR-10/100 and Tiny-ImageNet, which validate the robustness of SimpleAT against state-of-the-art adversarial attackers such as AutoAttack. Our results also demonstrate that SimpleAT exhibits good performance in the presence of various image corruptions, such as those found in the CIFAR-10-C. In addition, we empirically show that SimpleAT is capable of reducing the variance in model predictions, which is considered the primary contributor to robust overfitting. Our results also reveal the connections between SimpleAT and many advanced state-of-the-art adversarial defense methods.

摘要: 在这篇文章中，我们深入研究了对抗攻击的一种开创性防御技术--对抗性训练的基本组成部分。我们指出，损失函数、学习速率调度器和数据扩充等与模型结构无关的因素会影响对手的健壮性和泛化能力。当这些因素被控制时，我们引入了一种简单的基线方法，称为SimpleAT，它的性能与最近的方法具有竞争力，并减轻了稳健的过拟合。我们在CIFAR-10/100和Tiny-ImageNet上进行了大量的实验，验证了SimpleAT对AutoAttack等最先进的敌意攻击者的健壮性。我们的结果还表明，SimpleAT在存在各种图像损坏时表现出良好的性能，例如在CIFAR-10-C中发现的那些图像损坏。此外，我们的经验表明，SimpleAT能够减少模型预测中的方差，这被认为是稳健过拟合的主要贡献者。我们的结果还揭示了SimpleAT与许多先进的对抗性防御方法之间的联系。



## **31. Robust Network Pruning With Sparse Entropic Wasserstein Regression**

基于稀疏熵Wasserstein回归的稳健网络剪枝 cs.AI

submitted to ICLR 2024

**SubmitDate**: 2023-11-19    [abs](http://arxiv.org/abs/2310.04918v2) [paper-pdf](http://arxiv.org/pdf/2310.04918v2)

**Authors**: Lei You, Hei Victor Cheng

**Abstract**: This study tackles the issue of neural network pruning that inaccurate gradients exist when computing the empirical Fisher Information Matrix (FIM). We introduce an entropic Wasserstein regression (EWR) formulation, capitalizing on the geometric attributes of the optimal transport (OT) problem. This is analytically showcased to excel in noise mitigation by adopting neighborhood interpolation across data points. The unique strength of the Wasserstein distance is its intrinsic ability to strike a balance between noise reduction and covariance information preservation. Extensive experiments performed on various networks show comparable performance of the proposed method with state-of-the-art (SoTA) network pruning algorithms. Our proposed method outperforms the SoTA when the network size or the target sparsity is large, the gain is even larger with the existence of noisy gradients, possibly from noisy data, analog memory, or adversarial attacks. Notably, our proposed method achieves a gain of 6% improvement in accuracy and 8% improvement in testing loss for MobileNetV1 with less than one-fourth of the network parameters remaining.

摘要: 该研究解决了在计算经验Fisher信息矩阵(FIM)时存在梯度不准确的神经网络修剪问题。利用最优运输(OT)问题的几何属性，我们引入了一个熵Wasserstein回归(EWR)公式。分析表明，通过采用跨数据点的邻域内插，这在噪声缓解方面表现出色。沃瑟斯坦距离的独特优势在于它在降噪和保留协方差信息之间取得平衡的内在能力。在不同网络上进行的大量实验表明，该方法的性能与最新的网络剪枝算法(SOTA)相当。当网络规模或目标稀疏度较大时，我们提出的方法的性能优于SOTA，当存在噪声梯度时，增益甚至更大，可能来自噪声数据、模拟记忆或敌对攻击。值得注意的是，我们提出的方法在剩余不到四分之一的网络参数的情况下，对MobileNetV1实现了6%的准确率提高和8%的测试损失改善。



## **32. Adversarial Prompt Tuning for Vision-Language Models**

视觉语言模型的对抗性提示调整 cs.CV

**SubmitDate**: 2023-11-19    [abs](http://arxiv.org/abs/2311.11261v1) [paper-pdf](http://arxiv.org/pdf/2311.11261v1)

**Authors**: Jiaming Zhang, Xingjun Ma, Xin Wang, Lingyu Qiu, Jiaqi Wang, Yu-Gang Jiang, Jitao Sang

**Abstract**: With the rapid advancement of multimodal learning, pre-trained Vision-Language Models (VLMs) such as CLIP have demonstrated remarkable capacities in bridging the gap between visual and language modalities. However, these models remain vulnerable to adversarial attacks, particularly in the image modality, presenting considerable security risks. This paper introduces Adversarial Prompt Tuning (AdvPT), a novel technique to enhance the adversarial robustness of image encoders in VLMs. AdvPT innovatively leverages learnable text prompts and aligns them with adversarial image embeddings, to address the vulnerabilities inherent in VLMs without the need for extensive parameter training or modification of the model architecture. We demonstrate that AdvPT improves resistance against white-box and black-box adversarial attacks and exhibits a synergistic effect when combined with existing image-processing-based defense techniques, further boosting defensive capabilities. Comprehensive experimental analyses provide insights into adversarial prompt tuning, a novel paradigm devoted to improving resistance to adversarial images through textual input modifications, paving the way for future robust multimodal learning research. These findings open up new possibilities for enhancing the security of VLMs. Our code will be available upon publication of the paper.

摘要: 随着多通道学习的快速发展，诸如CLIP等预先训练的视觉语言模型在弥合视觉和语言通道之间的差距方面显示出了显著的能力。然而，这些模型仍然容易受到敌意攻击，特别是在图像模式方面，这带来了相当大的安全风险。本文介绍了对抗性提示调优(AdvPT)技术，这是一种在VLMS中增强图像编码器对抗性稳健性的新技术。AdvPT创新性地利用可学习的文本提示，并将其与对抗性图像嵌入相结合，以解决VLM中固有的漏洞，而无需进行广泛的参数培训或修改模型体系结构。我们证明，AdvPT提高了对白盒和黑盒攻击的抵抗力，并与现有的基于图像处理的防御技术相结合，显示出协同效应，进一步增强了防御能力。全面的实验分析提供了对对抗性即时调整的见解，这是一种致力于通过修改文本输入来提高对对抗性图像的抵抗力的新范式，为未来稳健的多通道学习研究铺平了道路。这些发现为增强VLM的安全性开辟了新的可能性。我们的代码将在论文发表后提供。



## **33. Untargeted Black-box Attacks for Social Recommendations**

针对社交推荐的无目标黑匣子攻击 cs.SI

Preprint. Under review

**SubmitDate**: 2023-11-19    [abs](http://arxiv.org/abs/2311.07127v2) [paper-pdf](http://arxiv.org/pdf/2311.07127v2)

**Authors**: Wenqi Fan, Shijie Wang, Xiao-yong Wei, Xiaowei Mei, Qing Li

**Abstract**: The rise of online social networks has facilitated the evolution of social recommender systems, which incorporate social relations to enhance users' decision-making process. With the great success of Graph Neural Networks in learning node representations, GNN-based social recommendations have been widely studied to model user-item interactions and user-user social relations simultaneously. Despite their great successes, recent studies have shown that these advanced recommender systems are highly vulnerable to adversarial attacks, in which attackers can inject well-designed fake user profiles to disrupt recommendation performances. While most existing studies mainly focus on targeted attacks to promote target items on vanilla recommender systems, untargeted attacks to degrade the overall prediction performance are less explored on social recommendations under a black-box scenario. To perform untargeted attacks on social recommender systems, attackers can construct malicious social relationships for fake users to enhance the attack performance. However, the coordination of social relations and item profiles is challenging for attacking black-box social recommendations. To address this limitation, we first conduct several preliminary studies to demonstrate the effectiveness of cross-community connections and cold-start items in degrading recommendations performance. Specifically, we propose a novel framework Multiattack based on multi-agent reinforcement learning to coordinate the generation of cold-start item profiles and cross-community social relations for conducting untargeted attacks on black-box social recommendations. Comprehensive experiments on various real-world datasets demonstrate the effectiveness of our proposed attacking framework under the black-box setting.

摘要: 在线社交网络的兴起促进了社交推荐系统的发展，社交推荐系统整合了社会关系，以增强用户的决策过程。随着图神经网络在学习节点表示方面的巨大成功，基于GNN的社交推荐被广泛研究以同时建模用户-项目交互和用户-用户社会关系。尽管它们取得了巨大的成功，但最近的研究表明，这些先进的推荐系统非常容易受到对手攻击，攻击者可以注入精心设计的虚假用户配置文件来破坏推荐性能。虽然现有的研究主要集中于在普通推荐系统上通过定向攻击来推广目标项，但在黑盒场景下，针对社交推荐的非定向攻击以降低整体预测性能的研究较少。为了对社交推荐系统进行无针对性的攻击，攻击者可以为虚假用户构建恶意的社交关系，以提高攻击性能。然而，社交关系和项目简介的协调对于攻击黑箱社交推荐是具有挑战性的。为了解决这一局限性，我们首先进行了几项初步研究，以证明跨社区联系和冷启动项目在降低推荐性能方面的有效性。具体地说，我们提出了一种新的基于多智能体强化学习的多攻击框架，用于协调冷启动项目配置文件的生成和跨社区社会关系的生成，以对黑盒社交推荐进行无针对性的攻击。在各种真实数据集上的综合实验证明了我们提出的攻击框架在黑盒环境下的有效性。



## **34. Robust Network Slicing: Multi-Agent Policies, Adversarial Attacks, and Defensive Strategies**

健壮的网络切片：多代理策略、对抗性攻击和防御策略 cs.LG

Published in IEEE Transactions on Machine Learning in Communications  and Networking (TMLCN)

**SubmitDate**: 2023-11-19    [abs](http://arxiv.org/abs/2311.11206v1) [paper-pdf](http://arxiv.org/pdf/2311.11206v1)

**Authors**: Feng Wang, M. Cenk Gursoy, Senem Velipasalar

**Abstract**: In this paper, we present a multi-agent deep reinforcement learning (deep RL) framework for network slicing in a dynamic environment with multiple base stations and multiple users. In particular, we propose a novel deep RL framework with multiple actors and centralized critic (MACC) in which actors are implemented as pointer networks to fit the varying dimension of input. We evaluate the performance of the proposed deep RL algorithm via simulations to demonstrate its effectiveness. Subsequently, we develop a deep RL based jammer with limited prior information and limited power budget. The goal of the jammer is to minimize the transmission rates achieved with network slicing and thus degrade the network slicing agents' performance. We design a jammer with both listening and jamming phases and address jamming location optimization as well as jamming channel optimization via deep RL. We evaluate the jammer at the optimized location, generating interference attacks in the optimized set of channels by switching between the jamming phase and listening phase. We show that the proposed jammer can significantly reduce the victims' performance without direct feedback or prior knowledge on the network slicing policies. Finally, we devise a Nash-equilibrium-supervised policy ensemble mixed strategy profile for network slicing (as a defensive measure) and jamming. We evaluate the performance of the proposed policy ensemble algorithm by applying on the network slicing agents and the jammer agent in simulations to show its effectiveness.

摘要: 提出了一种适用于多基站多用户动态环境下网络切片的多智能体深度强化学习框架。特别是，我们提出了一种新颖的具有多个参与者和集中批评的深层RL框架(MACC)，其中的参与者被实现为指针网络，以适应不同维度的输入。我们通过仿真对所提出的深度RL算法的性能进行了评估，以证明其有效性。随后，我们开发了一种基于有限先验信息和有限功率预算的深度RL干扰机。干扰器的目标是最小化网络分片所达到的传输速率，从而降低网络分片代理的性能。设计了一种具有监听和干扰两个阶段的干扰机，并通过深度RL进行了干扰位置优化和干扰信道优化。我们在最优位置对干扰进行评估，通过在干扰阶段和监听阶段之间切换，在优化的信道集合中产生干扰攻击。结果表明，在没有直接反馈或事先知道网络分片策略的情况下，所提出的干扰可以显著降低受害者的性能。最后，我们设计了一种用于网络切片(作为防御措施)和干扰的纳什均衡-监督策略集成混合策略配置文件。通过对网络切片代理和干扰代理的仿真实验，对所提出的策略集成算法的性能进行了评估，验证了算法的有效性。



## **35. Attention-Based Real-Time Defenses for Physical Adversarial Attacks in Vision Applications**

视觉应用中基于注意力的物理对抗攻击的实时防御 cs.CV

**SubmitDate**: 2023-11-19    [abs](http://arxiv.org/abs/2311.11191v1) [paper-pdf](http://arxiv.org/pdf/2311.11191v1)

**Authors**: Giulio Rossolini, Alessandro Biondi, Giorgio Buttazzo

**Abstract**: Deep neural networks exhibit excellent performance in computer vision tasks, but their vulnerability to real-world adversarial attacks, achieved through physical objects that can corrupt their predictions, raises serious security concerns for their application in safety-critical domains. Existing defense methods focus on single-frame analysis and are characterized by high computational costs that limit their applicability in multi-frame scenarios, where real-time decisions are crucial.   To address this problem, this paper proposes an efficient attention-based defense mechanism that exploits adversarial channel-attention to quickly identify and track malicious objects in shallow network layers and mask their adversarial effects in a multi-frame setting. This work advances the state of the art by enhancing existing over-activation techniques for real-world adversarial attacks to make them usable in real-time applications. It also introduces an efficient multi-frame defense framework, validating its efficacy through extensive experiments aimed at evaluating both defense performance and computational cost.

摘要: 深度神经网络在计算机视觉任务中表现出优异的性能，但它们在现实世界中通过物理对象实现的对手攻击的脆弱性会破坏它们的预测，这给它们在安全关键领域的应用带来了严重的安全问题。现有的防御方法侧重于单帧分析，并且具有计算成本高的特点，这限制了它们在多帧场景中的适用性，在多帧场景中，实时决策至关重要。针对这一问题，本文提出了一种高效的基于注意力的防御机制，该机制利用对抗性通道注意力来快速识别和跟踪浅层网络中的恶意对象，并在多帧环境下掩盖它们的对抗性效果。这项工作通过增强现有的针对现实世界对抗性攻击的过度激活技术来提高技术水平，使它们能够用于实时应用程序。它还引入了一种高效的多帧防御框架，通过旨在评估防御性能和计算成本的广泛实验来验证其有效性。



## **36. Boost Adversarial Transferability by Uniform Scale and Mix Mask Method**

用均匀标度和混合掩码方法提高对手的可转移性 cs.CV

**SubmitDate**: 2023-11-18    [abs](http://arxiv.org/abs/2311.12051v1) [paper-pdf](http://arxiv.org/pdf/2311.12051v1)

**Authors**: Tao Wang, Zijian Ying, Qianmu Li, zhichao Lian

**Abstract**: Adversarial examples generated from surrogate models often possess the ability to deceive other black-box models, a property known as transferability. Recent research has focused on enhancing adversarial transferability, with input transformation being one of the most effective approaches. However, existing input transformation methods suffer from two issues. Firstly, certain methods, such as the Scale-Invariant Method, employ exponentially decreasing scale invariant parameters that decrease the adaptability in generating effective adversarial examples across multiple scales. Secondly, most mixup methods only linearly combine candidate images with the source image, leading to reduced features blending effectiveness. To address these challenges, we propose a framework called Uniform Scale and Mix Mask Method (US-MM) for adversarial example generation. The Uniform Scale approach explores the upper and lower boundaries of perturbation with a linear factor, minimizing the negative impact of scale copies. The Mix Mask method introduces masks into the mixing process in a nonlinear manner, significantly improving the effectiveness of mixing strategies. Ablation experiments are conducted to validate the effectiveness of each component in US-MM and explore the effect of hyper-parameters. Empirical evaluations on standard ImageNet datasets demonstrate that US-MM achieves an average of 7% better transfer attack success rate compared to state-of-the-art methods.

摘要: 从代理模型生成的对抗性示例通常具有欺骗其他黑盒模型的能力，这一属性称为可转移性。最近的研究集中在增强对抗性可转移性上，输入转换是最有效的方法之一。然而，现有的输入变换方法存在两个问题。首先，某些方法，如尺度不变方法，采用指数下降的尺度不变参数，这降低了在多个尺度上生成有效对抗性示例的适应性。其次，大多数混合方法只将候选图像与源图像线性组合，导致特征混合效果降低。为了解决这些挑战，我们提出了一个框架，称为统一尺度和混合掩码方法（US-MM）的对抗性示例生成。“均匀缩放”方法使用线性因子探索扰动的上边界和下边界，从而最大限度地减少缩放副本的负面影响。混合遮罩方法以非线性方式将遮罩引入混合过程，从而显著提高混合策略的有效性。通过消融实验验证了US-MM中各组件的有效性，并探讨了超参数的影响。对标准ImageNet数据集的实证评估表明，与最先进的方法相比，US-MM的转移攻击成功率平均高出7%。



## **37. Improving Adversarial Transferability by Stable Diffusion**

通过稳定扩散提高对手的可转移性 cs.CV

**SubmitDate**: 2023-11-18    [abs](http://arxiv.org/abs/2311.11017v1) [paper-pdf](http://arxiv.org/pdf/2311.11017v1)

**Authors**: Jiayang Liu, Siyu Zhu, Siyuan Liang, Jie Zhang, Han Fang, Weiming Zhang, Ee-Chien Chang

**Abstract**: Deep neural networks (DNNs) are susceptible to adversarial examples, which introduce imperceptible perturbations to benign samples, deceiving DNN predictions. While some attack methods excel in the white-box setting, they often struggle in the black-box scenario, particularly against models fortified with defense mechanisms. Various techniques have emerged to enhance the transferability of adversarial attacks for the black-box scenario. Among these, input transformation-based attacks have demonstrated their effectiveness. In this paper, we explore the potential of leveraging data generated by Stable Diffusion to boost adversarial transferability. This approach draws inspiration from recent research that harnessed synthetic data generated by Stable Diffusion to enhance model generalization. In particular, previous work has highlighted the correlation between the presence of both real and synthetic data and improved model generalization. Building upon this insight, we introduce a novel attack method called Stable Diffusion Attack Method (SDAM), which incorporates samples generated by Stable Diffusion to augment input images. Furthermore, we propose a fast variant of SDAM to reduce computational overhead while preserving high adversarial transferability. Our extensive experimental results demonstrate that our method outperforms state-of-the-art baselines by a substantial margin. Moreover, our approach is compatible with existing transfer-based attacks to further enhance adversarial transferability.

摘要: 深度神经网络(DNN)很容易受到敌意例子的影响，这些例子给良性样本带来了难以察觉的扰动，欺骗了DNN的预测。虽然一些攻击方法在白盒设置中表现出色，但它们在黑盒场景中往往举步维艰，特别是针对具有防御机制的模型。已经出现了各种技术来增强针对黑盒情况的对抗性攻击的可转移性。其中，基于输入变换的攻击已经证明了它们的有效性。在本文中，我们探索了利用稳定扩散产生的数据来提高对抗转移的潜力。这种方法从最近的研究中获得灵感，这些研究利用稳定扩散产生的合成数据来增强模型泛化。特别是，以前的工作强调了真实数据和合成数据的存在与改进的模型泛化之间的相关性。基于这一认识，我们提出了一种新的攻击方法，称为稳定扩散攻击方法(SDAM)，它结合了稳定扩散产生的样本来增强输入图像。此外，我们提出了一种SDAM的快速变体来减少计算开销，同时保持了较高的对抗性可转移性。我们广泛的实验结果表明，我们的方法比最先进的基线有很大的优势。此外，我们的方法与现有的基于传输的攻击是兼容的，以进一步增强对抗的可转移性。



## **38. Security of quantum key distribution from generalised entropy accumulation**

广义熵积累下量子密钥分配的安全性 quant-ph

30 pages

**SubmitDate**: 2023-11-18    [abs](http://arxiv.org/abs/2203.04993v2) [paper-pdf](http://arxiv.org/pdf/2203.04993v2)

**Authors**: Tony Metger, Renato Renner

**Abstract**: The goal of quantum key distribution (QKD) is to establish a secure key between two parties connected by an insecure quantum channel. To use a QKD protocol in practice, one has to prove that a finite size key is secure against general attacks: no matter the adversary's attack, they cannot gain useful information about the key. A much simpler task is to prove security against collective attacks, where the adversary is assumed to behave identically and independently in each round. In this work, we provide a formal framework for general QKD protocols and show that for any protocol that can be expressed in this framework, security against general attacks reduces to security against collective attacks, which in turn reduces to a numerical computation. Our proof relies on a recently developed information-theoretic tool called generalised entropy accumulation and can handle generic prepare-and-measure protocols directly without switching to an entanglement-based version.

摘要: 量子密钥分发(QKD)的目标是在通过不安全的量子信道连接的双方之间建立安全密钥。要在实践中使用QKD协议，必须证明有限大小的密钥对一般攻击是安全的：无论对手进行攻击，他们都无法获得有关密钥的有用信息。一个简单得多的任务是证明对集体攻击的安全性，假设对手在每一轮中的行为都是相同的和独立的。在这项工作中，我们为一般的量子密钥分发协议提供了一个形式化的框架，并证明了对于任何可以在该框架中表达的协议，对一般攻击的安全性归结为对集体攻击的安全性，而集体攻击的安全性又归结为数值计算。我们的证明依赖于最近开发的一种称为广义熵累积的信息论工具，可以直接处理通用的准备和测量协议，而不需要切换到基于纠缠的版本。



## **39. PACOL: Poisoning Attacks Against Continual Learners**

PACOL：针对持续学习者的中毒攻击 cs.LG

**SubmitDate**: 2023-11-18    [abs](http://arxiv.org/abs/2311.10919v1) [paper-pdf](http://arxiv.org/pdf/2311.10919v1)

**Authors**: Huayu Li, Gregory Ditzler

**Abstract**: Continual learning algorithms are typically exposed to untrusted sources that contain training data inserted by adversaries and bad actors. An adversary can insert a small number of poisoned samples, such as mislabeled samples from previously learned tasks, or intentional adversarial perturbed samples, into the training datasets, which can drastically reduce the model's performance. In this work, we demonstrate that continual learning systems can be manipulated by malicious misinformation and present a new category of data poisoning attacks specific for continual learners, which we refer to as {\em Poisoning Attacks Against Continual Learners} (PACOL). The effectiveness of labeling flipping attacks inspires PACOL; however, PACOL produces attack samples that do not change the sample's label and produce an attack that causes catastrophic forgetting. A comprehensive set of experiments shows the vulnerability of commonly used generative replay and regularization-based continual learning approaches against attack methods. We evaluate the ability of label-flipping and a new adversarial poison attack, namely PACOL proposed in this work, to force the continual learning system to forget the knowledge of a learned task(s). More specifically, we compared the performance degradation of continual learning systems trained on benchmark data streams with and without poisoning attacks. Moreover, we discuss the stealthiness of the attacks in which we test the success rate of data sanitization defense and other outlier detection-based defenses for filtering out adversarial samples.

摘要: 持续学习算法通常暴露于不可信的来源，这些来源包含由对手和不良行为者插入的训练数据。敌手可以在训练数据集中插入少量有毒样本，例如来自先前学习任务的错误标记的样本，或者故意的对抗性扰动样本，这可能会显著降低模型的性能。在这项工作中，我们证明了持续学习系统可以被恶意错误信息操纵，并提出了一种新的针对持续学习者的数据中毒攻击，我们称之为针对持续学习者的中毒攻击(PACOL)。标签翻转攻击的有效性启发了PACOL；然而，PACOL生成的攻击样本不会改变样本的标签，并产生导致灾难性遗忘的攻击。一组全面的实验表明，常用的生成性回放和基于正则化的连续学习方法对攻击方法的脆弱性。我们评估了标签翻转和本文提出的一种新的对抗性毒物攻击，即PACOL，以迫使持续学习系统忘记学习任务的知识的能力(S)。更具体地说，我们比较了在基准数据流上训练的连续学习系统在有和没有中毒攻击的情况下性能下降的情况。此外，我们还讨论了攻击的隐蔽性，测试了数据净化防御和其他基于孤立点检测的防御措施过滤对手样本的成功率。



## **40. Parrot-Trained Adversarial Examples: Pushing the Practicality of Black-Box Audio Attacks against Speaker Recognition Models**

鹦鹉训练的对抗性例子：将黑匣子音频攻击的实用性推向说话人识别模型 cs.SD

**SubmitDate**: 2023-11-17    [abs](http://arxiv.org/abs/2311.07780v2) [paper-pdf](http://arxiv.org/pdf/2311.07780v2)

**Authors**: Rui Duan, Zhe Qu, Leah Ding, Yao Liu, Zhuo Lu

**Abstract**: Audio adversarial examples (AEs) have posed significant security challenges to real-world speaker recognition systems. Most black-box attacks still require certain information from the speaker recognition model to be effective (e.g., keeping probing and requiring the knowledge of similarity scores). This work aims to push the practicality of the black-box attacks by minimizing the attacker's knowledge about a target speaker recognition model. Although it is not feasible for an attacker to succeed with completely zero knowledge, we assume that the attacker only knows a short (or a few seconds) speech sample of a target speaker. Without any probing to gain further knowledge about the target model, we propose a new mechanism, called parrot training, to generate AEs against the target model. Motivated by recent advancements in voice conversion (VC), we propose to use the one short sentence knowledge to generate more synthetic speech samples that sound like the target speaker, called parrot speech. Then, we use these parrot speech samples to train a parrot-trained(PT) surrogate model for the attacker. Under a joint transferability and perception framework, we investigate different ways to generate AEs on the PT model (called PT-AEs) to ensure the PT-AEs can be generated with high transferability to a black-box target model with good human perceptual quality. Real-world experiments show that the resultant PT-AEs achieve the attack success rates of 45.8% - 80.8% against the open-source models in the digital-line scenario and 47.9% - 58.3% against smart devices, including Apple HomePod (Siri), Amazon Echo, and Google Home, in the over-the-air scenario.

摘要: 音频对抗样本（AE）对现实世界的说话人识别系统提出了重大的安全挑战。大多数黑盒攻击仍然需要来自说话者识别模型的某些信息才有效（例如，保持探测并需要相似性分数的知识）。这项工作的目的是推动黑盒攻击的实用性，最大限度地减少攻击者的知识，目标说话人识别模型。虽然攻击者不可能在完全零知识的情况下成功，但我们假设攻击者只知道目标说话者的一小段（或几秒钟）语音样本。在没有任何探测以获得关于目标模型的进一步知识的情况下，我们提出了一种新的机制，称为鹦鹉训练，以针对目标模型生成AE。受语音转换（VC）的最新进展的启发，我们建议使用一个简短的句子知识来生成更多的合成语音样本，听起来像目标扬声器，称为鹦鹉语音。然后，我们使用这些鹦鹉语音样本来训练攻击者的鹦鹉训练（PT）代理模型。在联合可转移性和感知框架下，我们研究了在PT模型上生成AE的不同方法（称为PT-AE），以确保PT-AE可以以高可转移性生成到具有良好人类感知质量的黑盒目标模型。真实世界的实验表明，在数字线路场景中，所产生的PT-AE对开源模型的攻击成功率为45.8% - 80.8%，在空中场景中，对智能设备（包括Apple HomePod（Siri），Amazon Echo和Google Home）的攻击成功率为47.9% - 58.3%。



## **41. Tailoring Adversarial Attacks on Deep Neural Networks for Targeted Class Manipulation Using DeepFool Algorithm**

利用DeepFool算法定制针对目标类操作的深层神经网络敌意攻击 cs.CV

8 pages, 3 figures

**SubmitDate**: 2023-11-17    [abs](http://arxiv.org/abs/2310.13019v3) [paper-pdf](http://arxiv.org/pdf/2310.13019v3)

**Authors**: S. M. Fazle Rabby Labib, Joyanta Jyoti Mondal, Meem Arafat Manab

**Abstract**: Deep neural networks (DNNs) have significantly advanced various domains, but their vulnerability to adversarial attacks poses serious concerns. Understanding these vulnerabilities and developing effective defense mechanisms is crucial. DeepFool, an algorithm proposed by Moosavi-Dezfooli et al. (2016), finds minimal perturbations to misclassify input images. However, DeepFool lacks a targeted approach, making it less effective in specific attack scenarios. Also, in previous related works, researchers primarily focus on success, not considering how much an image is getting distorted; the integrity of the image quality, and the confidence level to misclassifying. So, in this paper, we propose Enhanced Targeted DeepFool, an augmented version of DeepFool that allows targeting specific classes for misclassification and also introduce a minimum confidence score requirement hyperparameter to enhance flexibility. Our experiments demonstrate the effectiveness and efficiency of the proposed method across different deep neural network architectures while preserving image integrity as much and perturbation rate as less as possible. By using our approach, the behavior of models can be manipulated arbitrarily using the perturbed images, as we can specify both the target class and the associated confidence score, unlike other DeepFool-derivative works, such as Targeted DeepFool by Gajjar et al. (2022). Results show that one of the deep convolutional neural network architectures, AlexNet, and one of the state-of-the-art model Vision Transformer exhibit high robustness to getting fooled. This approach can have larger implication, as our tuning of confidence level can expose the robustness of image recognition models. Our code will be made public upon acceptance of the paper.

摘要: 深度神经网络(DNN)极大地推动了各个领域的发展，但其对敌意攻击的脆弱性也引起了人们的严重关注。了解这些漏洞并开发有效的防御机制至关重要。DeepFool是Moosavi-Dezgoi等人提出的算法。(2016)，找到最小的扰动来错误分类输入图像。然而，DeepFool缺乏有针对性的方法，使其在特定攻击场景中不那么有效。此外，在以往的相关研究中，研究者主要关注成功，而没有考虑图像被扭曲的程度、图像质量的完整性以及对错误分类的置信度。因此，在本文中，我们提出了增强型目标DeepFool，这是DeepFool的一个扩展版本，它允许针对特定类别的错误分类，并引入了最小置信度要求超参数来增强灵活性。实验结果表明，该方法在保持图像完整性的同时，降低了图像的摄动率，在不同的深度神经网络结构上都具有较高的效率。通过使用我们的方法，可以使用扰动图像来任意操纵模型的行为，因为我们可以指定目标类和相关的置信度分数，这与其他DeepFool派生工作不同，例如Gajjar等人的Target DeepFool。(2022年)。结果表明，深度卷积神经网络结构AlexNet和最先进的模型Vision Transformer之一对上当具有很强的鲁棒性。这种方法可以有更大的意义，因为我们调整置信度可以暴露图像识别模型的稳健性。我们的代码将在文件被接受后公布。



## **42. Breaking Boundaries: Balancing Performance and Robustness in Deep Wireless Traffic Forecasting**

打破界限：深度无线流量预测中的性能和稳健性平衡 cs.LG

Accepted for presentation at the ARTMAN workshop, part of the ACM  Conference on Computer and Communications Security (CCS), 2023

**SubmitDate**: 2023-11-17    [abs](http://arxiv.org/abs/2311.09790v2) [paper-pdf](http://arxiv.org/pdf/2311.09790v2)

**Authors**: Romain Ilbert, Thai V. Hoang, Zonghua Zhang, Themis Palpanas

**Abstract**: Balancing the trade-off between accuracy and robustness is a long-standing challenge in time series forecasting. While most of existing robust algorithms have achieved certain suboptimal performance on clean data, sustaining the same performance level in the presence of data perturbations remains extremely hard. In this paper, we study a wide array of perturbation scenarios and propose novel defense mechanisms against adversarial attacks using real-world telecom data. We compare our strategy against two existing adversarial training algorithms under a range of maximal allowed perturbations, defined using $\ell_{\infty}$-norm, $\in [0.1,0.4]$. Our findings reveal that our hybrid strategy, which is composed of a classifier to detect adversarial examples, a denoiser to eliminate noise from the perturbed data samples, and a standard forecaster, achieves the best performance on both clean and perturbed data. Our optimal model can retain up to $92.02\%$ the performance of the original forecasting model in terms of Mean Squared Error (MSE) on clean data, while being more robust than the standard adversarially trained models on perturbed data. Its MSE is 2.71$\times$ and 2.51$\times$ lower than those of comparing methods on normal and perturbed data, respectively. In addition, the components of our models can be trained in parallel, resulting in better computational efficiency. Our results indicate that we can optimally balance the trade-off between the performance and robustness of forecasting models by improving the classifier and denoiser, even in the presence of sophisticated and destructive poisoning attacks.

摘要: 平衡准确性和鲁棒性之间的权衡是时间序列预测中的一个长期挑战。虽然大多数现有的鲁棒算法已经在干净数据上实现了某些次优性能，但在存在数据扰动的情况下保持相同的性能水平仍然非常困难。在本文中，我们研究了各种各样的扰动场景，并使用真实世界的电信数据提出了对抗性攻击的新防御机制。我们将我们的策略与两个现有的对抗性训练算法在最大允许扰动范围内进行比较，使用$\ell_{\infty}$-norm，$\in [0.1，0.4]$定义。我们的研究结果表明，我们的混合策略，它是由一个分类器来检测对抗性的例子，一个去噪器，以消除干扰的数据样本中的噪声，和一个标准的预测器，实现了最好的性能在干净和扰动的数据。我们的最佳模型可以保留高达92.02\%$的原始预测模型的性能方面的均方误差（MSE）干净的数据，而更强大的干扰数据比标准的对抗训练模型。在正态和扰动数据下，其均方误差分别比比较方法低2.71倍和2.51倍。此外，我们模型的组件可以并行训练，从而提高计算效率。我们的研究结果表明，我们可以通过改进分类器和去噪器来最佳地平衡预测模型的性能和鲁棒性之间的权衡，即使在存在复杂和破坏性的中毒攻击的情况下。



## **43. Laccolith: Hypervisor-Based Adversary Emulation with Anti-Detection**

Laccolith：基于系统管理程序的反检测对手仿真 cs.CR

**SubmitDate**: 2023-11-17    [abs](http://arxiv.org/abs/2311.08274v2) [paper-pdf](http://arxiv.org/pdf/2311.08274v2)

**Authors**: Vittorio Orbinato, Marco Carlo Feliciano, Domenico Cotroneo, Roberto Natella

**Abstract**: Advanced Persistent Threats (APTs) represent the most threatening form of attack nowadays since they can stay undetected for a long time. Adversary emulation is a proactive approach for preparing against these attacks. However, adversary emulation tools lack the anti-detection abilities of APTs. We introduce Laccolith, a hypervisor-based solution for adversary emulation with anti-detection to fill this gap. We also present an experimental study to compare Laccolith with MITRE CALDERA, a state-of-the-art solution for adversary emulation, against five popular anti-virus products. We found that CALDERA cannot evade detection, limiting the realism of emulated attacks, even when combined with a state-of-the-art anti-detection framework. Our experiments show that Laccolith can hide its activities from all the tested anti-virus products, thus making it suitable for realistic emulations.

摘要: 高级持续威胁(APT)是当今最具威胁性的攻击形式，因为它们可以长时间保持不被发现。对手模拟是为应对这些攻击做准备的一种主动方法。然而，敌方仿真工具缺乏APTS的抗检测能力。我们引入了Laccolith，这是一种基于系统管理程序的解决方案，用于具有反检测功能的对手模拟，以填补这一空白。我们还提供了一项实验研究，以比较Laccolith和MITRE Caldera，这是一种最先进的对手模拟解决方案，与五种流行的反病毒产品进行比较。我们发现，Caldera无法逃避检测，从而限制了模拟攻击的真实性，即使与最先进的反检测框架结合使用也是如此。我们的实验表明，漆柱可以对所有测试的抗病毒产品隐藏其活性，从而使其适合于真实的模拟。



## **44. Breaking Temporal Consistency: Generating Video Universal Adversarial Perturbations Using Image Models**

打破时间一致性：使用图像模型生成视频通用对抗性扰动 cs.CV

ICCV 2023

**SubmitDate**: 2023-11-17    [abs](http://arxiv.org/abs/2311.10366v1) [paper-pdf](http://arxiv.org/pdf/2311.10366v1)

**Authors**: Hee-Seon Kim, Minji Son, Minbeom Kim, Myung-Joon Kwon, Changick Kim

**Abstract**: As video analysis using deep learning models becomes more widespread, the vulnerability of such models to adversarial attacks is becoming a pressing concern. In particular, Universal Adversarial Perturbation (UAP) poses a significant threat, as a single perturbation can mislead deep learning models on entire datasets. We propose a novel video UAP using image data and image model. This enables us to take advantage of the rich image data and image model-based studies available for video applications. However, there is a challenge that image models are limited in their ability to analyze the temporal aspects of videos, which is crucial for a successful video attack. To address this challenge, we introduce the Breaking Temporal Consistency (BTC) method, which is the first attempt to incorporate temporal information into video attacks using image models. We aim to generate adversarial videos that have opposite patterns to the original. Specifically, BTC-UAP minimizes the feature similarity between neighboring frames in videos. Our approach is simple but effective at attacking unseen video models. Additionally, it is applicable to videos of varying lengths and invariant to temporal shifts. Our approach surpasses existing methods in terms of effectiveness on various datasets, including ImageNet, UCF-101, and Kinetics-400.

摘要: 随着使用深度学习模型的视频分析变得越来越普遍，这类模型对对手攻击的脆弱性正成为一个紧迫的问题。特别是，通用对抗性扰动(UAP)构成了一个重大威胁，因为单个扰动可能会误导整个数据集上的深度学习模型。利用图像数据和图像模型，提出了一种新的视频UAP。这使我们能够利用可用于视频应用的丰富的图像数据和基于图像模型的研究。然而，存在一个挑战，即图像模型在分析视频的时间方面的能力有限，这对成功的视频攻击至关重要。为了应对这一挑战，我们引入了打破时间一致性(BTC)方法，这是首次尝试将时间信息融入到使用图像模型的视频攻击中。我们的目标是生成与原始视频模式相反的对抗性视频。具体地说，BTC-UAP将视频中相邻帧之间的特征相似度降至最低。我们的方法简单但有效地攻击了看不见的视频模型。此外，它还适用于不同长度和不随时间移位的视频。我们的方法在各种数据集上的有效性方面超过了现有的方法，包括ImageNet、UCF-101和Kinetics-400。



## **45. Quantum Public-Key Encryption with Tamper-Resilient Public Keys from One-Way Functions**

基于单向函数防篡改公钥的量子公钥加密 quant-ph

48 pages

**SubmitDate**: 2023-11-17    [abs](http://arxiv.org/abs/2304.01800v3) [paper-pdf](http://arxiv.org/pdf/2304.01800v3)

**Authors**: Fuyuki Kitagawa, Tomoyuki Morimae, Ryo Nishimaki, Takashi Yamakawa

**Abstract**: We construct quantum public-key encryption from one-way functions. In our construction, public keys are quantum, but ciphertexts are classical. Quantum public-key encryption from one-way functions (or weaker primitives such as pseudorandom function-like states) are also proposed in some recent works [Morimae-Yamakawa, eprint:2022/1336; Coladangelo, eprint:2023/282; Barooti-Grilo-Malavolta-Sattath-Vu-Walter, eprint:2023/877]. However, they have a huge drawback: they are secure only when quantum public keys can be transmitted to the sender (who runs the encryption algorithm) without being tampered with by the adversary, which seems to require unsatisfactory physical setup assumptions such as secure quantum channels. Our construction is free from such a drawback: it guarantees the secrecy of the encrypted messages even if we assume only unauthenticated quantum channels. Thus, the encryption is done with adversarially tampered quantum public keys. Our construction is the first quantum public-key encryption that achieves the goal of classical public-key encryption, namely, to establish secure communication over insecure channels, based only on one-way functions. Moreover, we show a generic compiler to upgrade security against chosen plaintext attacks (CPA security) into security against chosen ciphertext attacks (CCA security) only using one-way functions. As a result, we obtain CCA secure quantum public-key encryption based only on one-way functions.

摘要: 我们用单向函数构造量子公钥加密。在我们的构造中，公钥是量子的，但密文是经典的。最近的一些工作也提出了基于单向函数(或更弱的基元，如伪随机函数类状态)的量子公钥加密[Morimae-Yamakawa，ePrint：2022/1336；Coladangelo，ePrint：2023/282；Barooti-Grilo-Malavolta-Sattath-Vu-Walter，ePrint：2023/877]。然而，它们有一个巨大的缺点：只有当量子公钥可以传输给发送者(运行加密算法)而不被对手篡改时，它们才是安全的，这似乎需要不令人满意的物理设置假设，如安全量子通道。我们的构造没有这样的缺点：它保证了加密消息的保密性，即使我们假设只有未经验证的量子通道。因此，加密是用恶意篡改的量子公钥完成的。我们的构造是第一个量子公钥加密，它实现了经典公钥加密的目标，即仅基于单向函数在不安全的通道上建立安全通信。此外，我们还给出了一个通用编译器，该编译器仅使用单向函数将针对选择明文攻击的安全性(CPA安全性)升级为针对选择密文攻击(CCA安全性)的安全性。由此，我们得到了仅基于单向函数的CCA安全量子公钥加密。



## **46. You Cannot Escape Me: Detecting Evasions of SIEM Rules in Enterprise Networks**

你不能逃避我：在企业网络中检测对SIEM规则的逃避 cs.CR

To be published in Proceedings of the 33rd USENIX Security Symposium  (USENIX Security 2024)

**SubmitDate**: 2023-11-16    [abs](http://arxiv.org/abs/2311.10197v1) [paper-pdf](http://arxiv.org/pdf/2311.10197v1)

**Authors**: Rafael Uetz, Marco Herzog, Louis Hackländer, Simon Schwarz, Martin Henze

**Abstract**: Cyberattacks have grown into a major risk for organizations, with common consequences being data theft, sabotage, and extortion. Since preventive measures do not suffice to repel attacks, timely detection of successful intruders is crucial to stop them from reaching their final goals. For this purpose, many organizations utilize Security Information and Event Management (SIEM) systems to centrally collect security-related events and scan them for attack indicators using expert-written detection rules. However, as we show by analyzing a set of widespread SIEM detection rules, adversaries can evade almost half of them easily, allowing them to perform common malicious actions within an enterprise network without being detected. To remedy these critical detection blind spots, we propose the idea of adaptive misuse detection, which utilizes machine learning to compare incoming events to SIEM rules on the one hand and known-benign events on the other hand to discover successful evasions. Based on this idea, we present AMIDES, an open-source proof-of-concept adaptive misuse detection system. Using four weeks of SIEM events from a large enterprise network and more than 500 hand-crafted evasions, we show that AMIDES successfully detects a majority of these evasions without any false alerts. In addition, AMIDES eases alert analysis by assessing which rules were evaded. Its computational efficiency qualifies AMIDES for real-world operation and hence enables organizations to significantly reduce detection blind spots with moderate effort.

摘要: 网络攻击已经成为组织的主要风险，常见的后果是数据被盗、破坏和敲诈勒索。由于预防措施不足以击退攻击，及时发现成功的入侵者对于阻止他们实现最终目标至关重要。为此，许多组织利用安全信息和事件管理(SIEM)系统集中收集与安全相关的事件，并使用专家编写的检测规则扫描它们的攻击指标。然而，正如我们通过分析一组广泛使用的SIEM检测规则所表明的那样，攻击者可以很容易地避开其中的近一半，使他们能够在不被检测到的情况下在企业网络内执行常见的恶意操作。为了弥补这些关键的检测盲点，我们提出了自适应误用检测的思想，一方面利用机器学习将传入事件与SIEM规则进行比较，另一方面利用已知良性事件来发现成功的规避。基于这一思想，我们提出了一个开源的概念验证自适应误用检测系统AMIDES。使用来自大型企业网络的四周SIEM事件和500多个手工创建的规避，我们表明AMIDES成功检测到了大多数此类规避，而没有任何错误警报。此外，AMIDES通过评估哪些规则被规避来简化警报分析。它的计算效率使AMADS有资格在现实世界中运行，因此使组织能够以适度的努力显著减少检测盲点。



## **47. Differentiable JPEG: The Devil is in the Details**

与众不同的JPEG：魔鬼在细节中 cs.CV

Accepted at WACV 2024. Project page:  https://christophreich1996.github.io/differentiable_jpeg/

**SubmitDate**: 2023-11-16    [abs](http://arxiv.org/abs/2309.06978v2) [paper-pdf](http://arxiv.org/pdf/2309.06978v2)

**Authors**: Christoph Reich, Biplob Debnath, Deep Patel, Srimat Chakradhar

**Abstract**: JPEG remains one of the most widespread lossy image coding methods. However, the non-differentiable nature of JPEG restricts the application in deep learning pipelines. Several differentiable approximations of JPEG have recently been proposed to address this issue. This paper conducts a comprehensive review of existing diff. JPEG approaches and identifies critical details that have been missed by previous methods. To this end, we propose a novel diff. JPEG approach, overcoming previous limitations. Our approach is differentiable w.r.t. the input image, the JPEG quality, the quantization tables, and the color conversion parameters. We evaluate the forward and backward performance of our diff. JPEG approach against existing methods. Additionally, extensive ablations are performed to evaluate crucial design choices. Our proposed diff. JPEG resembles the (non-diff.) reference implementation best, significantly surpassing the recent-best diff. approach by $3.47$dB (PSNR) on average. For strong compression rates, we can even improve PSNR by $9.51$dB. Strong adversarial attack results are yielded by our diff. JPEG, demonstrating the effective gradient approximation. Our code is available at https://github.com/necla-ml/Diff-JPEG.

摘要: JPEG仍然是应用最广泛的有损图像编码方法之一。然而，JPEG的不可微特性限制了其在深度学习管道中的应用。为了解决这个问题，最近已经提出了几种JPEG的可微近似。本文对现有的DIFF进行了全面的回顾。JPEG处理并确定了以前方法遗漏的关键细节。为此，我们提出了一个新颖的Diff。JPEG方法，克服了以前的限制。我们的方法是可微的W.r.t。输入图像、JPEG质量、量化表和颜色转换参数。我们评估了DIFF的向前和向后性能。JPEG方法与现有方法的对比。此外，还进行了广泛的消融，以评估关键的设计选择。我们提议的不同之处。JPEG与(Non-Diff.)参考实现最好，大大超过了最近最好的差异。平均接近3.47美元分贝(PSNR)。对于强压缩率，我们甚至可以将PSNR提高9.51美元分贝。强大的对抗性攻击结果是由我们的差异产生的。JPEG格式，演示了有效的渐变近似。我们的代码可以在https://github.com/necla-ml/Diff-JPEG.上找到



## **48. Towards more Practical Threat Models in Artificial Intelligence Security**

人工智能安全中更实用的威胁模型 cs.CR

18 pages, 4 figures, 7 tables, under submission

**SubmitDate**: 2023-11-16    [abs](http://arxiv.org/abs/2311.09994v1) [paper-pdf](http://arxiv.org/pdf/2311.09994v1)

**Authors**: Kathrin Grosse, Lukas Bieringer, Tarek Richard Besold, Alexandre Alahi

**Abstract**: Recent works have identified a gap between research and practice in artificial intelligence security: threats studied in academia do not always reflect the practical use and security risks of AI. For example, while models are often studied in isolation, they form part of larger ML pipelines in practice. Recent works also brought forward that adversarial manipulations introduced by academic attacks are impractical. We take a first step towards describing the full extent of this disparity. To this end, we revisit the threat models of the six most studied attacks in AI security research and match them to AI usage in practice via a survey with \textbf{271} industrial practitioners. On the one hand, we find that all existing threat models are indeed applicable. On the other hand, there are significant mismatches: research is often too generous with the attacker, assuming access to information not frequently available in real-world settings. Our paper is thus a call for action to study more practical threat models in artificial intelligence security.

摘要: 最近的研究发现了人工智能安全研究与实践之间的差距：学术界研究的威胁并不总是反映人工智能的实际使用和安全风险。例如，虽然模型通常被孤立地研究，但它们在实践中形成了更大的ML管道的一部分。最近的工作也提出了学术攻击引入的对抗性操纵是不切实际的。我们迈出了描述这种差异的全面程度的第一步。为此，我们重新审视了人工智能安全研究中研究最多的六种攻击的威胁模型，并通过对行业从业者的调查，将它们与人工智能在实践中的使用进行匹配。一方面，我们发现所有现有的威胁模型确实适用。另一方面，也存在严重的不匹配：研究往往对攻击者过于慷慨，假设在现实世界中不经常获得信息。因此，我们的论文呼吁采取行动，研究人工智能安全中更实用的威胁模型。



## **49. Hijacking Large Language Models via Adversarial In-Context Learning**

通过对抗性情境学习劫持大型语言模型 cs.LG

**SubmitDate**: 2023-11-16    [abs](http://arxiv.org/abs/2311.09948v1) [paper-pdf](http://arxiv.org/pdf/2311.09948v1)

**Authors**: Yao Qiang, Xiangyu Zhou, Dongxiao Zhu

**Abstract**: In-context learning (ICL) has emerged as a powerful paradigm leveraging LLMs for specific tasks by utilizing labeled examples as demonstrations in the precondition prompts. Despite its promising performance, ICL suffers from instability with the choice and arrangement of examples. Additionally, crafted adversarial attacks pose a notable threat to the robustness of ICL. However, existing attacks are either easy to detect, rely on external models, or lack specificity towards ICL. To address these issues, this work introduces a novel transferable attack for ICL, aiming to hijack LLMs to generate the targeted response. The proposed LLM hijacking attack leverages a gradient-based prompt search method to learn and append imperceptible adversarial suffixes to the in-context demonstrations. Extensive experimental results on various tasks and datasets demonstrate the effectiveness of our LLM hijacking attack, resulting in a distracted attention towards adversarial tokens, consequently leading to the targeted unwanted outputs.

摘要: 在上下文学习（ICL）已经成为一个强大的范式，利用LLM的特定任务，利用标记的例子作为示范的前提提示。尽管ICL的表现令人鼓舞，但它在选择和安排示例方面存在不稳定性。此外，精心设计的对抗性攻击对ICL的健壮性构成了显著的威胁。然而，现有的攻击要么容易检测，要么依赖于外部模型，要么缺乏针对ICL的特异性。为了解决这些问题，这项工作介绍了一种新的可转移的攻击ICL，旨在劫持LLM生成有针对性的响应。拟议的LLM劫持攻击利用基于梯度的提示搜索方法来学习并将不可感知的对抗性后缀添加到上下文演示中。在各种任务和数据集上的大量实验结果证明了我们的LLM劫持攻击的有效性，导致对对抗性令牌的注意力分散，从而导致有针对性的不必要的输出。



## **50. Bilevel Optimization with a Lower-level Contraction: Optimal Sample Complexity without Warm-start**

低水平收缩的两层优化：无热启动的最优样本复杂性 stat.ML

Corrected Remark 18 + other small edits. Code at  https://github.com/CSML-IIT-UCL/bioptexps

**SubmitDate**: 2023-11-16    [abs](http://arxiv.org/abs/2202.03397v4) [paper-pdf](http://arxiv.org/pdf/2202.03397v4)

**Authors**: Riccardo Grazzi, Massimiliano Pontil, Saverio Salzo

**Abstract**: We analyse a general class of bilevel problems, in which the upper-level problem consists in the minimization of a smooth objective function and the lower-level problem is to find the fixed point of a smooth contraction map. This type of problems include instances of meta-learning, equilibrium models, hyperparameter optimization and data poisoning adversarial attacks. Several recent works have proposed algorithms which warm-start the lower-level problem, i.e.~they use the previous lower-level approximate solution as a staring point for the lower-level solver. This warm-start procedure allows one to improve the sample complexity in both the stochastic and deterministic settings, achieving in some cases the order-wise optimal sample complexity. However, there are situations, e.g., meta learning and equilibrium models, in which the warm-start procedure is not well-suited or ineffective. In this work we show that without warm-start, it is still possible to achieve order-wise (near) optimal sample complexity. In particular, we propose a simple method which uses (stochastic) fixed point iterations at the lower-level and projected inexact gradient descent at the upper-level, that reaches an $\epsilon$-stationary point using $O(\epsilon^{-2})$ and $\tilde{O}(\epsilon^{-1})$ samples for the stochastic and the deterministic setting, respectively. Finally, compared to methods using warm-start, our approach yields a simpler analysis that does not need to study the coupled interactions between the upper-level and lower-level iterates.

摘要: 我们分析了一类一般的两层问题，其中上层问题在于光滑目标函数的最小化，下层问题是寻找光滑压缩映射的不动点。这类问题包括元学习、均衡模型、超参数优化和数据中毒攻击等实例。最近的一些工作已经提出了暖启动下层问题的算法，即它们使用先前的下层近似解作为下层求解器的起点。这种热启动过程允许人们在随机和确定性设置下改善样本复杂性，在某些情况下实现顺序最优的样本复杂性。然而，在一些情况下，例如元学习和平衡模型，热启动程序不是很适合或无效的。在这项工作中，我们证明了在没有热启动的情况下，仍然有可能达到阶次(接近)最优的样本复杂性。特别地，我们提出了一种简单的方法，它在下层使用(随机)不动点迭代，在上层使用投影的不精确梯度下降，在随机和确定设置下分别使用$O(epsilon^{-2})$和$tilde{O}(epsilon^{-1})$样本达到$-epsilon$-固定点。最后，与使用热启动的方法相比，我们的方法产生了更简单的分析，不需要研究上层和下层迭代之间的耦合作用。



