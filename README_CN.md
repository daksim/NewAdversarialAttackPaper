# Latest Adversarial Attack Papers
**update at 2022-02-20 06:31:42**

翻译来自 https://cloud.tencent.com/document/product/551/15619

## **1. Alexa versus Alexa: Controlling Smart Speakers by Self-Issuing Voice Commands**

Alexa与Alexa：通过自行发出语音命令控制智能扬声器 cs.CR

15 pages, 5 figures, published in Proceedings of the 2022 ACM Asia  Conference on Computer and Communications Security (ASIA CCS '22)

**SubmitDate**: 2022-02-17    [paper-pdf](http://arxiv.org/pdf/2202.08619v1)

**Authors**: Sergio Esposito, Daniele Sgandurra, Giampaolo Bella

**Abstracts**: We present Alexa versus Alexa (AvA), a novel attack that leverages audio files containing voice commands and audio reproduction methods in an offensive fashion, to gain control of Amazon Echo devices for a prolonged amount of time. AvA leverages the fact that Alexa running on an Echo device correctly interprets voice commands originated from audio files even when they are played by the device itself -- i.e., it leverages a command self-issue vulnerability. Hence, AvA removes the necessity of having a rogue speaker in proximity of the victim's Echo, a constraint that many attacks share. With AvA, an attacker can self-issue any permissible command to Echo, controlling it on behalf of the legitimate user. We have verified that, via AvA, attackers can control smart appliances within the household, buy unwanted items, tamper linked calendars and eavesdrop on the user. We also discovered two additional Echo vulnerabilities, which we call Full Volume and Break Tag Chain. The Full Volume increases the self-issue command recognition rate, by doubling it on average, hence allowing attackers to perform additional self-issue commands. Break Tag Chain increases the time a skill can run without user interaction, from eight seconds to more than one hour, hence enabling attackers to setup realistic social engineering scenarios. By exploiting these vulnerabilities, the adversary can self-issue commands that are correctly executed 99% of the times and can keep control of the device for a prolonged amount of time. We reported these vulnerabilities to Amazon via their vulnerability research program, who rated them with a Medium severity score. Finally, to assess limitations of AvA on a larger scale, we provide the results of a survey performed on a study group of 18 users, and we show that most of the limitations against AvA are hardly used in practice.

摘要: 我们提出了Alexa vs.Alexa(AVA)，这是一种新颖的攻击，它以攻击性的方式利用包含语音命令和音频复制方法的音频文件来获得对Amazon Echo设备的长时间控制。AVA利用在Echo设备上运行的Alexa能够正确解释源自音频文件的语音命令这一事实，即使音频文件是由设备本身播放的，即它利用了命令自发布漏洞。因此，AVA消除了在受害者的回声附近设置流氓扬声器的必要性，这是许多攻击都有的限制。使用AVA，攻击者可以自行向Echo发出任何允许的命令，并代表合法用户控制它。我们已经证实，通过AVA，攻击者可以控制家庭内的智能家电，购买不需要的物品，篡改链接的日历，并窃听用户。我们还发现了另外两个Echo漏洞，我们称之为Full Volume和Break Tag Chain。全音量通过将其平均翻倍来提高自发布命令识别率，从而允许攻击者执行额外的自发布命令。中断标签链增加了技能在没有用户交互的情况下可以运行的时间，从8秒增加到1小时以上，从而使攻击者能够设置现实的社会工程场景。通过利用这些漏洞，攻击者可以自行发出99%的正确执行次数的命令，并可以在较长时间内保持对设备的控制。我们通过他们的漏洞研究计划向亚马逊报告了这些漏洞，亚马逊对它们进行了中等严重程度的评级。最后，为了在更大的范围内评估AVA的局限性，我们提供了对18名用户进行的一项调查的结果，我们发现大多数针对AVA的限制在实践中几乎没有使用。



## **2. Fingerprinting Deep Neural Networks Globally via Universal Adversarial Perturbations**

基于普遍对抗性扰动的深度神经网络全局指纹识别 cs.CR

**SubmitDate**: 2022-02-17    [paper-pdf](http://arxiv.org/pdf/2202.08602v1)

**Authors**: Zirui Peng, Shaofeng Li, Guoxing Chen, Cheng Zhang, Haojin Zhu, Minhui Xue

**Abstracts**: In this paper, we propose a novel and practical mechanism which enables the service provider to verify whether a suspect model is stolen from the victim model via model extraction attacks. Our key insight is that the profile of a DNN model's decision boundary can be uniquely characterized by its \textit{Universal Adversarial Perturbations (UAPs)}. UAPs belong to a low-dimensional subspace and piracy models' subspaces are more consistent with victim model's subspace compared with non-piracy model. Based on this, we propose a UAP fingerprinting method for DNN models and train an encoder via \textit{contrastive learning} that takes fingerprint as inputs, outputs a similarity score. Extensive studies show that our framework can detect model IP breaches with confidence $> 99.99 \%$ within only $20$ fingerprints of the suspect model. It has good generalizability across different model architectures and is robust against post-modifications on stolen models.

摘要: 本文提出了一种新颖而实用的机制，使服务提供商能够通过模型提取攻击来验证受害者模型中的可疑模型是否被窃取。我们的主要见解是DNN模型的决策边界的轮廓可以由它的\textit(通用对抗性扰动(UAP))来唯一地刻画。UAP属于低维子空间，与非盗版模型相比，盗版模型的子空间与受害者模型的子空间更加一致。在此基础上，提出了一种DNN模型的UAP指纹识别方法，并以指纹为输入，通过对比学习训练编码器，输出相似度得分。大量的研究表明，我们的框架可以在可疑模型的$2 0$指纹范围内以>99.99$的置信度检测到模型IP泄露。它具有良好的跨不同模型体系结构的通用性，并且对窃取模型的后期修改具有健壮性。



## **3. Improving Robustness of Deep Reinforcement Learning Agents: Environment Attack based on the Critic Network**

提高深度强化学习代理的健壮性：基于批判网络的环境攻击 cs.LG

**SubmitDate**: 2022-02-17    [paper-pdf](http://arxiv.org/pdf/2104.03154v2)

**Authors**: Lucas Schott, Hatem Hajri, Sylvain Lamprier

**Abstracts**: To improve policy robustness of deep reinforcement learning agents, a line of recent works focus on producing disturbances of the environment. Existing approaches of the literature to generate meaningful disturbances of the environment are adversarial reinforcement learning methods. These methods set the problem as a two-player game between the protagonist agent, which learns to perform a task in an environment, and the adversary agent, which learns to disturb the protagonist via modifications of the considered environment. Both protagonist and adversary are trained with deep reinforcement learning algorithms. Alternatively, we propose in this paper to build on gradient-based adversarial attacks, usually used for classification tasks for instance, that we apply on the critic network of the protagonist to identify efficient disturbances of the environment. Rather than learning an attacker policy, which usually reveals as very complex and unstable, we leverage the knowledge of the critic network of the protagonist, to dynamically complexify the task at each step of the learning process. We show that our method, while being faster and lighter, leads to significantly better improvements in policy robustness than existing methods of the literature.

摘要: 为了提高深度强化学习代理的策略鲁棒性，最近的一系列工作集中在产生环境扰动上。现有文献中产生有意义的环境扰动的方法是对抗性强化学习方法。这些方法将问题设置为主角Agent和对手Agent之间的两人博弈，前者学习在环境中执行任务，后者学习通过修改所考虑的环境来干扰主角。利用深度强化学习算法对主角和对手进行训练。或者，我们在本文中建议建立基于梯度的对抗性攻击，通常用于分类任务，例如，我们应用于主人公的批评网络来识别环境的有效干扰。我们不学习通常表现为非常复杂和不稳定的攻击者策略，而是利用主人公批评网络的知识，在学习过程的每一步动态地使任务复杂化。我们表明，虽然我们的方法更快、更轻，但与现有的文献方法相比，我们的方法在政策稳健性方面的改善要明显更好。



## **4. GasHis-Transformer: A Multi-scale Visual Transformer Approach for Gastric Histopathology Image Classification**

GasHis-Transformer：一种用于胃组织病理图像分类的多尺度视觉变换方法 cs.CV

**SubmitDate**: 2022-02-17    [paper-pdf](http://arxiv.org/pdf/2104.14528v6)

**Authors**: Haoyuan Chen, Chen Li, Ge Wang, Xiaoyan Li, Md Rahaman, Hongzan Sun, Weiming Hu, Yixin Li, Wanli Liu, Changhao Sun, Shiliang Ai, Marcin Grzegorzek

**Abstracts**: Existing deep learning methods for diagnosis of gastric cancer commonly use convolutional neural network. Recently, the Visual Transformer has attracted great attention because of its performance and efficiency, but its applications are mostly in the field of computer vision. In this paper, a multi-scale visual transformer model, referred to as GasHis-Transformer, is proposed for Gastric Histopathological Image Classification (GHIC), which enables the automatic classification of microscopic gastric images into abnormal and normal cases. The GasHis-Transformer model consists of two key modules: A global information module and a local information module to extract histopathological features effectively. In our experiments, a public hematoxylin and eosin (H&E) stained gastric histopathological dataset with 280 abnormal and normal images are divided into training, validation and test sets by a ratio of 1 : 1 : 2. The GasHis-Transformer model is applied to estimate precision, recall, F1-score and accuracy on the test set of gastric histopathological dataset as 98.0%, 100.0%, 96.0% and 98.0%, respectively. Furthermore, a critical study is conducted to evaluate the robustness of GasHis-Transformer, where ten different noises including four adversarial attack and six conventional image noises are added. In addition, a clinically meaningful study is executed to test the gastrointestinal cancer identification performance of GasHis-Transformer with 620 abnormal images and achieves 96.8% accuracy. Finally, a comparative study is performed to test the generalizability with both H&E and immunohistochemical stained images on a lymphoma image dataset and a breast cancer dataset, producing comparable F1-scores (85.6% and 82.8%) and accuracies (83.9% and 89.4%), respectively. In conclusion, GasHisTransformer demonstrates high classification performance and shows its significant potential in the GHIC task.

摘要: 现有的胃癌诊断深度学习方法普遍采用卷积神经网络。近年来，视觉变压器因其高性能和高效率而备受关注，但其应用大多集中在计算机视觉领域。本文提出了一种用于胃组织病理图像分类(GHIC)的多尺度视觉转换器模型(简称GasHis-Transformer)，该模型能够自动将胃显微图像分类为异常和正常病例。GasHis-Transformer模型由两个关键模块组成：全局信息模块和局部信息模块，有效地提取组织病理学特征。在我们的实验中，一个公共的苏木精伊红(H&E)染色的胃组织病理学数据集以1：1：2的比例分为训练集、验证集和测试集，训练集、验证集和测试集的比例为1：1：2。应用GasHis-Transformer模型估计胃组织病理学数据集的准确率、召回率、F1得分和准确率分别为98.0%、100.0%、96.0%和98.0%。此外，还对GasHis-Transformer的稳健性进行了关键研究，添加了10种不同的噪声，包括4种对抗性攻击和6种常规图像噪声。另外，利用620幅异常图像对GasHis-Transformer的胃肠道肿瘤识别性能进行了有临床意义的测试，准确率达到96.8%。最后，在淋巴瘤图像数据集和乳腺癌数据集上对H&E和免疫组织化学染色图像的泛化能力进行了比较研究，得到了可比的F1得分(85.6%和82.8%)和准确率(83.9%和89.4%)。总之，GasHisTransformer表现出很高的分类性能，并在GHIC任务中显示出巨大的潜力。



## **5. Measuring the Transferability of $\ell_\infty$ Attacks by the $\ell_2$ Norm**

用$\ELL_2$范数度量$\ELL_\INFTY$攻击的可转移性 cs.LG

**SubmitDate**: 2022-02-17    [paper-pdf](http://arxiv.org/pdf/2102.10343v3)

**Authors**: Sizhe Chen, Qinghua Tao, Zhixing Ye, Xiaolin Huang

**Abstracts**: Deep neural networks could be fooled by adversarial examples with trivial differences to original samples. To keep the difference imperceptible in human eyes, researchers bound the adversarial perturbations by the $\ell_\infty$ norm, which is now commonly served as the standard to align the strength of different attacks for a fair comparison. However, we propose that using the $\ell_\infty$ norm alone is not sufficient in measuring the attack strength, because even with a fixed $\ell_\infty$ distance, the $\ell_2$ distance also greatly affects the attack transferability between models. Through the discovery, we reach more in-depth understandings towards the attack mechanism, i.e., several existing methods attack black-box models better partly because they craft perturbations with 70\% to 130\% larger $\ell_2$ distances. Since larger perturbations naturally lead to better transferability, we thereby advocate that the strength of attacks should be simultaneously measured by both the $\ell_\infty$ and $\ell_2$ norm. Our proposal is firmly supported by extensive experiments on ImageNet dataset from 7 attacks, 4 white-box models, and 9 black-box models.

摘要: 深层神经网络可能会被与原始样本有微小差异的对抗性例子所欺骗。为了保持这种差异在人眼中不可察觉，研究人员用$\ell_\infty$范数来约束对抗性扰动，这现在通常被用作对不同攻击强度进行公平比较的标准。然而，我们认为单独使用$\ell_\infty$范数来度量攻击强度是不够的，因为即使在固定的$\ell_\infty$距离的情况下，$\ell_2$距离也会极大地影响攻击在模型之间的可转移性。通过这一发现，我们对攻击机制有了更深入的理解，即现有的几种方法对黑盒模型的攻击效果较好，部分原因是它们设计的扰动具有较大的$ell_2$70~130\{##**$}${##**$}}。由于更大的扰动自然会导致更好的可转移性，因此我们主张攻击的强度应该同时用$\ell_inty$和$\ell_2$范数来度量。我们的建议得到了来自7个攻击、4个白盒模型和9个黑盒模型的ImageNet数据集的广泛实验的坚定支持。



## **6. Towards Evaluating the Robustness of Neural Networks Learned by Transduction**

基于转导学习的神经网络鲁棒性评价方法研究 cs.LG

Paper published at ICLR 2022. arXiv admin note: text overlap with  arXiv:2106.08387

**SubmitDate**: 2022-02-17    [paper-pdf](http://arxiv.org/pdf/2110.14735v2)

**Authors**: Jiefeng Chen, Xi Wu, Yang Guo, Yingyu Liang, Somesh Jha

**Abstracts**: There has been emerging interest in using transductive learning for adversarial robustness (Goldwasser et al., NeurIPS 2020; Wu et al., ICML 2020; Wang et al., ArXiv 2021). Compared to traditional defenses, these defense mechanisms "dynamically learn" the model based on test-time input; and theoretically, attacking these defenses reduces to solving a bilevel optimization problem, which poses difficulty in crafting adaptive attacks. In this paper, we examine these defense mechanisms from a principled threat analysis perspective. We formulate and analyze threat models for transductive-learning based defenses, and point out important subtleties. We propose the principle of attacking model space for solving bilevel attack objectives, and present Greedy Model Space Attack (GMSA), an attack framework that can serve as a new baseline for evaluating transductive-learning based defenses. Through systematic evaluation, we show that GMSA, even with weak instantiations, can break previous transductive-learning based defenses, which were resilient to previous attacks, such as AutoAttack. On the positive side, we report a somewhat surprising empirical result of "transductive adversarial training": Adversarially retraining the model using fresh randomness at the test time gives a significant increase in robustness against attacks we consider.

摘要: 人们对使用转导学习来实现对抗鲁棒性产生了新的兴趣(Goldwasser等人，NeurIPS 2020；Wu等人，ICML 2020；Wang等人，Arxiv 2021)。与传统防御相比，这些防御机制“动态学习”基于测试时间输入的模型；从理论上讲，攻击这些防御归结为求解一个双层优化问题，这给自适应攻击的设计带来了困难。在本文中，我们从原则性威胁分析的角度来检查这些防御机制。建立并分析了基于传导式学习防御的威胁模型，指出了重要的细微之处。我们提出了攻击模型空间求解双层攻击目标的原理，并提出了贪婪模型空间攻击(GMSA)这一攻击框架，可作为评估基于转导学习的防御的新基线。通过系统的评估，我们证明了GMSA即使在弱实例化的情况下，也可以打破以往基于传导式学习的防御机制，这些防御机制对AutoAttack等先前的攻击是有弹性的。从积极的一面来看，我们报告了一个有点令人惊讶的“转导对抗训练”的经验结果：在测试时使用新的随机性对模型进行对抗性重新训练，可以显著提高对我们所考虑的攻击的鲁棒性。



## **7. Generalizable Information Theoretic Causal Representation**

广义信息论因果表示 cs.LG

**SubmitDate**: 2022-02-17    [paper-pdf](http://arxiv.org/pdf/2202.08388v1)

**Authors**: Mengyue Yang, Xinyu Cai, Furui Liu, Xu Chen, Zhitang Chen, Jianye Hao, Jun Wang

**Abstracts**: It is evidence that representation learning can improve model's performance over multiple downstream tasks in many real-world scenarios, such as image classification and recommender systems. Existing learning approaches rely on establishing the correlation (or its proxy) between features and the downstream task (labels), which typically results in a representation containing cause, effect and spurious correlated variables of the label. Its generalizability may deteriorate because of the unstability of the non-causal parts. In this paper, we propose to learn causal representation from observational data by regularizing the learning procedure with mutual information measures according to our hypothetical causal graph. The optimization involves a counterfactual loss, based on which we deduce a theoretical guarantee that the causality-inspired learning is with reduced sample complexity and better generalization ability. Extensive experiments show that the models trained on causal representations learned by our approach is robust under adversarial attacks and distribution shift.

摘要: 事实证明，在图像分类和推荐系统等实际场景中，表征学习可以提高模型在多个下游任务上的性能。现有的学习方法依赖于在特征和下游任务(标签)之间建立相关性(或其代理)，这通常导致包含标签的原因、结果和虚假相关变量的表示。由于非因果部分的不稳定性，其泛化能力可能会恶化。在本文中，我们建议根据假设的因果图，通过互信息度量来规范学习过程，从而从观测数据中学习因果表示。这种优化涉及到反事实的损失，在此基础上，我们推导出因果启发学习具有更低的样本复杂度和更好的泛化能力的理论保证。大量实验表明，该方法训练的因果表示模型在对抗攻击和分布偏移情况下具有较好的鲁棒性。



## **8. Characterizing Attacks on Deep Reinforcement Learning**

深度强化学习攻击的特征描述 cs.LG

AAMAS 2022, 13 pages, 6 figures

**SubmitDate**: 2022-02-16    [paper-pdf](http://arxiv.org/pdf/1907.09470v3)

**Authors**: Xinlei Pan, Chaowei Xiao, Warren He, Shuang Yang, Jian Peng, Mingjie Sun, Jinfeng Yi, Zijiang Yang, Mingyan Liu, Bo Li, Dawn Song

**Abstracts**: Recent studies show that Deep Reinforcement Learning (DRL) models are vulnerable to adversarial attacks, which attack DRL models by adding small perturbations to the observations. However, some attacks assume full availability of the victim model, and some require a huge amount of computation, making them less feasible for real world applications. In this work, we make further explorations of the vulnerabilities of DRL by studying other aspects of attacks on DRL using realistic and efficient attacks. First, we adapt and propose efficient black-box attacks when we do not have access to DRL model parameters. Second, to address the high computational demands of existing attacks, we introduce efficient online sequential attacks that exploit temporal consistency across consecutive steps. Third, we explore the possibility of an attacker perturbing other aspects in the DRL setting, such as the environment dynamics. Finally, to account for imperfections in how an attacker would inject perturbations in the physical world, we devise a method for generating a robust physical perturbations to be printed. The attack is evaluated on a real-world robot under various conditions. We conduct extensive experiments both in simulation such as Atari games, robotics and autonomous driving, and on real-world robotics, to compare the effectiveness of the proposed attacks with baseline approaches. To the best of our knowledge, we are the first to apply adversarial attacks on DRL systems to physical robots.

摘要: 最近的研究表明，深度强化学习(DRL)模型容易受到敌意攻击，这种攻击是通过在观测值中添加小扰动来攻击DRL模型的。然而，一些攻击假设受害者模型完全可用，而另一些攻击需要大量的计算，这使得它们在现实世界的应用程序中不太可行。在这项工作中，我们通过研究对DRL的其他方面的攻击，使用真实而有效的攻击，进一步探讨了DRL的漏洞。首先，当我们无法获得DRL模型参数时，我们采用并提出了有效的黑盒攻击。其次，为了解决现有攻击对计算的高要求，我们引入了高效的在线顺序攻击，该攻击利用了连续步骤之间的时间一致性。第三，我们探讨攻击者干扰DRL设置中其他方面的可能性，例如环境动态。最后，为了说明攻击者如何在物理世界中注入扰动的不完善之处，我们设计了一种生成要打印的健壮物理扰动的方法。在不同条件下对真实机器人进行了攻击评估。我们在Atari游戏、机器人和自动驾驶等模拟游戏中，以及在真实机器人上进行了广泛的实验，以比较所提出的攻击和基线方法的有效性。据我们所知，我们是第一个将针对DRL系统的对抗性攻击应用于物理机器人的公司。



## **9. Real-Time Neural Voice Camouflage**

实时神经语音伪装 cs.SD

14 pages

**SubmitDate**: 2022-02-16    [paper-pdf](http://arxiv.org/pdf/2112.07076v2)

**Authors**: Mia Chiquier, Chengzhi Mao, Carl Vondrick

**Abstracts**: Automatic speech recognition systems have created exciting possibilities for applications, however they also enable opportunities for systematic eavesdropping. We propose a method to camouflage a person's voice over-the-air from these systems without inconveniencing the conversation between people in the room. Standard adversarial attacks are not effective in real-time streaming situations because the characteristics of the signal will have changed by the time the attack is executed. We introduce predictive attacks, which achieve real-time performance by forecasting the attack that will be the most effective in the future. Under real-time constraints, our method jams the established speech recognition system DeepSpeech 3.9x more than baselines as measured through word error rate, and 6.6x more as measured through character error rate. We furthermore demonstrate our approach is practically effective in realistic environments over physical distances.

摘要: 自动语音识别系统为应用创造了令人兴奋的可能性，然而它们也为系统窃听提供了机会。我们提出了一种方法，从这些系统中伪装出人的空中语音，而不会给房间里的人之间的对话带来不便。标准对抗性攻击在实时流情况下无效，因为在执行攻击时信号的特性将发生变化。我们引入预测性攻击，通过预测未来最有效的攻击来实现实时性能。在实时约束条件下，我们的方法对已建立的语音识别系统DeepSpeech的拥塞程度是基线的3.9倍，通过字符错误率的衡量是基线的6.6倍。我们进一步证明了我们的方法在物理距离上的现实环境中是实际有效的。



## **10. Ideal Tightly Couple (t,m,n) Secret Sharing**

理想紧耦合(t，m，n)秘密共享 cs.CR

few errors in the articles within the proposed scheme, and also  grammatical errors, so its our request pls withdraw our articles as soon as  possible

**SubmitDate**: 2022-02-16    [paper-pdf](http://arxiv.org/pdf/1905.02004v2)

**Authors**: Fuyou Miao, Keju Meng, Wenchao Huang, Yan Xiong, Xingfu Wang

**Abstracts**: As a fundamental cryptographic tool, (t,n)-threshold secret sharing ((t,n)-SS) divides a secret among n shareholders and requires at least t, (t<=n), of them to reconstruct the secret. Ideal (t,n)-SSs are most desirable in security and efficiency among basic (t,n)-SSs. However, an adversary, even without any valid share, may mount Illegal Participant (IP) attack or t/2-Private Channel Cracking (t/2-PCC) attack to obtain the secret in most (t,n)-SSs.To secure ideal (t,n)-SSs against the 2 attacks, 1) the paper introduces the notion of Ideal Tightly cOupled (t,m,n) Secret Sharing (or (t,m,n)-ITOSS ) to thwart IP attack without Verifiable SS; (t,m,n)-ITOSS binds all m, (m>=t), participants into a tightly coupled group and requires all participants to be legal shareholders before recovering the secret. 2) As an example, the paper presents a polynomial-based (t,m,n)-ITOSS scheme, in which the proposed k-round Random Number Selection (RNS) guarantees that adversaries have to crack at least symmetrical private channels among participants before obtaining the secret. Therefore, k-round RNS enhances the robustness of (t,m,n)-ITOSS against t/2-PCC attack to the utmost. 3) The paper finally presents a generalized method of converting an ideal (t,n)-SS into a (t,m,n)-ITOSS, which helps an ideal (t,n)-SS substantially improve the robustness against the above 2 attacks.

摘要: 作为一种基本的密码工具，(t，n)-门限秘密共享((t，n)-SS)将一个秘密分配给n个股东，并要求其中至少t个(t<=n)个股东重构秘密。在基本(t，n)-SS中，理想(t，n)-SS在安全性和效率方面是最理想的。为了保证理想(t，n)-SS不受这两种攻击的攻击，1)引入理想紧耦合(t，m，n)秘密共享(或(t，m，n)-ITOSS)的概念，在没有可验证SS的情况下阻止IP攻击；(t，m，n)-ITOSS将所有m，(m>=t)个参与者绑定到一个紧密耦合的组中，并要求所有参与者在恢复秘密之前都是合法股东。2)作为例子，提出了一个基于多项式的(t，m，n)-ITOSS方案，其中所提出的k轮随机数选择(RNS)方案保证攻击者在获得秘密之前必须至少破解参与者之间的对称私有信道。因此，k轮RNS最大限度地增强了(t，m，n)-ITOSS对t/2-PCC攻击的鲁棒性。3)最后给出了将理想(t，n)-SS转换为(t，m，n)-ITOSS的一般方法，从而大大提高了理想(t，n)-SS对上述两种攻击的鲁棒性。



## **11. Deduplicating Training Data Mitigates Privacy Risks in Language Models**

对训练数据进行重复数据消除可降低语言模型中的隐私风险 cs.CR

**SubmitDate**: 2022-02-16    [paper-pdf](http://arxiv.org/pdf/2202.06539v2)

**Authors**: Nikhil Kandpal, Eric Wallace, Colin Raffel

**Abstracts**: Past work has shown that large language models are susceptible to privacy attacks, where adversaries generate sequences from a trained model and detect which sequences are memorized from the training set. In this work, we show that the success of these attacks is largely due to duplication in commonly used web-scraped training sets. We first show that the rate at which language models regenerate training sequences is superlinearly related to a sequence's count in the training set. For instance, a sequence that is present 10 times in the training data is on average generated ~1000 times more often than a sequence that is present only once. We next show that existing methods for detecting memorized sequences have near-chance accuracy on non-duplicated training sequences. Finally, we find that after applying methods to deduplicate training data, language models are considerably more secure against these types of privacy attacks. Taken together, our results motivate an increased focus on deduplication in privacy-sensitive applications and a reevaluation of the practicality of existing privacy attacks.

摘要: 过去的工作表明，大型语言模型容易受到隐私攻击，攻击者从训练的模型生成序列，并从训练集中检测哪些序列被记忆。在这项工作中，我们表明这些攻击的成功在很大程度上是由于常用的Web抓取训练集的重复。我们首先证明了语言模型重新生成训练序列的速度与训练集中序列的计数呈超线性关系。例如，在训练数据中出现10次的序列的平均生成频率是只出现一次的序列的~1000倍。接下来，我们展示了现有的检测记忆序列的方法在非重复训练序列上具有近乎概率的准确性。最后，我们发现，在应用方法对训练数据进行去重之后，语言模型对这些类型的隐私攻击的安全性要高得多。综上所述，我们的结果促使人们更加关注隐私敏感应用程序中的重复数据删除，并重新评估现有隐私攻击的实用性。



## **12. The Adversarial Security Mitigations of mmWave Beamforming Prediction Models using Defensive Distillation and Adversarial Retraining**

基于防御蒸馏和对抗性再训练的毫米波波束形成预测模型的对抗性安全缓解 cs.CR

26 pages, under review

**SubmitDate**: 2022-02-16    [paper-pdf](http://arxiv.org/pdf/2202.08185v1)

**Authors**: Murat Kuzlu, Ferhat Ozgur Catak, Umit Cali, Evren Catak, Ozgur Guler

**Abstracts**: The design of a security scheme for beamforming prediction is critical for next-generation wireless networks (5G, 6G, and beyond). However, there is no consensus about protecting the beamforming prediction using deep learning algorithms in these networks. This paper presents the security vulnerabilities in deep learning for beamforming prediction using deep neural networks (DNNs) in 6G wireless networks, which treats the beamforming prediction as a multi-output regression problem. It is indicated that the initial DNN model is vulnerable against adversarial attacks, such as Fast Gradient Sign Method (FGSM), Basic Iterative Method (BIM), Projected Gradient Descent (PGD), and Momentum Iterative Method (MIM), because the initial DNN model is sensitive to the perturbations of the adversarial samples of the training data. This study also offers two mitigation methods, such as adversarial training and defensive distillation, for adversarial attacks against artificial intelligence (AI)-based models used in the millimeter-wave (mmWave) beamforming prediction. Furthermore, the proposed scheme can be used in situations where the data are corrupted due to the adversarial examples in the training data. Experimental results show that the proposed methods effectively defend the DNN models against adversarial attacks in next-generation wireless networks.

摘要: 波束成形预测安全方案的设计对下一代无线网络(5G、6G等)至关重要。然而，在这些网络中使用深度学习算法来保护波束形成预测并没有达成共识。针对6G无线网络中使用深度神经网络(DNNs)进行波束形成预测的深度学习中存在的安全漏洞，将波束形成预测处理为多输出回归问题。研究表明，由于初始DNN模型对训练数据对抗性样本的扰动比较敏感，因此容易受到敌意攻击，如快速梯度符号法(FGSM)、基本迭代法(BIM)、投影梯度下降法(PGD)和动量迭代法(MIM)。本研究还针对毫米波波束形成预测中使用的基于人工智能(AI)模型的对抗性攻击，提供了两种缓解方法，如对抗性训练和防御蒸馏。此外，所提出的方案还可以用于由于训练数据中的对抗性示例而导致数据被破坏的情况。实验结果表明，在下一代无线网络中，本文提出的方法有效地防御了DNN模型的攻击。



## **13. Finding Dynamics Preserving Adversarial Winning Tickets**

寻找动态保存的对抗性中奖彩票 cs.LG

Accepted by AISTATS2022

**SubmitDate**: 2022-02-16    [paper-pdf](http://arxiv.org/pdf/2202.06488v2)

**Authors**: Xupeng Shi, Pengfei Zheng, A. Adam Ding, Yuan Gao, Weizhong Zhang

**Abstracts**: Modern deep neural networks (DNNs) are vulnerable to adversarial attacks and adversarial training has been shown to be a promising method for improving the adversarial robustness of DNNs. Pruning methods have been considered in adversarial context to reduce model capacity and improve adversarial robustness simultaneously in training. Existing adversarial pruning methods generally mimic the classical pruning methods for natural training, which follow the three-stage 'training-pruning-fine-tuning' pipelines. We observe that such pruning methods do not necessarily preserve the dynamics of dense networks, making it potentially hard to be fine-tuned to compensate the accuracy degradation in pruning. Based on recent works of \textit{Neural Tangent Kernel} (NTK), we systematically study the dynamics of adversarial training and prove the existence of trainable sparse sub-network at initialization which can be trained to be adversarial robust from scratch. This theoretically verifies the \textit{lottery ticket hypothesis} in adversarial context and we refer such sub-network structure as \textit{Adversarial Winning Ticket} (AWT). We also show empirical evidences that AWT preserves the dynamics of adversarial training and achieve equal performance as dense adversarial training.

摘要: 现代深层神经网络(DNNs)容易受到敌意攻击，对抗性训练已被证明是提高DNN对抗性鲁棒性的一种很有前途的方法。在训练过程中，考虑了对抗性环境下的剪枝方法，在减少模型容量的同时提高对抗性鲁棒性。现有的对抗性剪枝方法一般是模仿经典的自然训练剪枝方法，遵循“训练-剪枝-微调”三阶段的流水线。我们观察到，这样的剪枝方法并不一定保持密集网络的动态，使得它可能很难被微调来补偿剪枝过程中的精度下降。基于神经切核(NTK)的最新工作，系统地研究了对抗性训练的动力学，证明了在初始化时存在可训练的稀疏子网络，它可以从头开始训练为对抗性健壮性网络。这从理论上验证了对抗性环境下的\text{彩票假设}，我们将这种子网络结构称为\text{对抗性中票}(AWT)。我们还展示了经验证据，AWT保持了对抗性训练的动态性，并获得了与密集对抗性训练相同的性能。



## **14. Neural Network Trojans Analysis and Mitigation from the Input Domain**

基于输入域的神经网络木马分析与消除 cs.LG

**SubmitDate**: 2022-02-16    [paper-pdf](http://arxiv.org/pdf/2202.06382v2)

**Authors**: Zhenting Wang, Hailun Ding, Juan Zhai, Shiqing Ma

**Abstracts**: Deep Neural Networks (DNNs) can learn Trojans (or backdoors) from benign or poisoned data, which raises security concerns of using them. By exploiting such Trojans, the adversary can add a fixed input space perturbation to any given input to mislead the model predicting certain outputs (i.e., target labels). In this paper, we analyze such input space Trojans in DNNs, and propose a theory to explain the relationship of a model's decision regions and Trojans: a complete and accurate Trojan corresponds to a hyperplane decision region in the input domain. We provide a formal proof of this theory, and provide empirical evidence to support the theory and its relaxations. Based on our analysis, we design a novel training method that removes Trojans during training even on poisoned datasets, and evaluate our prototype on five datasets and five different attacks. Results show that our method outperforms existing solutions. Code: \url{https://anonymous.4open.science/r/NOLE-84C3}.

摘要: 深度神经网络(DNNs)可以从良性或有毒的数据中学习特洛伊木马程序(或后门程序)，这增加了使用它们的安全问题。通过利用这种特洛伊木马，攻击者可以向任何给定的输入添加固定的输入空间扰动，以误导预测特定输出(即目标标签)的模型。本文分析了DNNs中的这类输入空间木马，提出了一种解释模型决策域与木马关系的理论：一个完整准确的木马对应于输入域中的一个超平面决策域。我们给出了这一理论的形式证明，并提供了支持该理论及其松弛的经验证据。基于我们的分析，我们设计了一种新的训练方法，即使在有毒的数据集上也能在训练过程中清除木马程序，并在五个数据集和五个不同的攻击上对我们的原型进行了评估。结果表明，我们的方法比已有的方法具有更好的性能。编码：\url{https://anonymous.4open.science/r/NOLE-84C3}.



## **15. Understanding and Improving Graph Injection Attack by Promoting Unnoticeability**

通过提高不可察觉来理解和改进图注入攻击 cs.LG

ICLR2022

**SubmitDate**: 2022-02-16    [paper-pdf](http://arxiv.org/pdf/2202.08057v1)

**Authors**: Yongqiang Chen, Han Yang, Yonggang Zhang, Kaili Ma, Tongliang Liu, Bo Han, James Cheng

**Abstracts**: Recently Graph Injection Attack (GIA) emerges as a practical attack scenario on Graph Neural Networks (GNNs), where the adversary can merely inject few malicious nodes instead of modifying existing nodes or edges, i.e., Graph Modification Attack (GMA). Although GIA has achieved promising results, little is known about why it is successful and whether there is any pitfall behind the success. To understand the power of GIA, we compare it with GMA and find that GIA can be provably more harmful than GMA due to its relatively high flexibility. However, the high flexibility will also lead to great damage to the homophily distribution of the original graph, i.e., similarity among neighbors. Consequently, the threats of GIA can be easily alleviated or even prevented by homophily-based defenses designed to recover the original homophily. To mitigate the issue, we introduce a novel constraint -- homophily unnoticeability that enforces GIA to preserve the homophily, and propose Harmonious Adversarial Objective (HAO) to instantiate it. Extensive experiments verify that GIA with HAO can break homophily-based defenses and outperform previous GIA attacks by a significant margin. We believe our methods can serve for a more reliable evaluation of the robustness of GNNs.

摘要: 图注入攻击(GIA)是近年来在图神经网络(GNNs)上出现的一种实用攻击方案，即攻击者只能注入少量的恶意节点，而不需要修改已有的节点或边，即图修改攻击(GMA)。尽管GIA取得了令人振奋的成果，但人们对其成功的原因以及成功背后是否存在陷阱知之甚少。为了理解GIA的力量，我们将其与GMA进行比较，发现由于其相对较高的灵活性，GIA显然比GMA更具危害性。但是，较高的灵活性也会对原图的同源分布造成很大的破坏，即邻域间的相似性。因此，GIA的威胁可以很容易地减轻，甚至可以通过基于同源的防御措施来恢复原始的同源。为了缓解这一问题，我们引入了一种新的约束--同形不可察觉，强制GIA保持同形，并提出了和谐对抗目标(HAO)来实例化它。广泛的实验证明，带有HAO的GIA可以打破基于同源的防御，并显著超过以前的GIA攻击。我们相信我们的方法可以更可靠地评估GNNs的健壮性。



## **16. Increasing-Margin Adversarial (IMA) Training to Improve Adversarial Robustness of Neural Networks**

提高神经网络对抗鲁棒性的增量对抗性(IMA)训练 cs.CV

45 pages, 15 figures, 31 tables

**SubmitDate**: 2022-02-16    [paper-pdf](http://arxiv.org/pdf/2005.09147v7)

**Authors**: Linhai Ma, Liang Liang

**Abstracts**: Convolutional neural network (CNN) has surpassed traditional methods for medical image classification. However, CNN is vulnerable to adversarial attacks which may lead to disastrous consequences in medical applications. Although adversarial noises are usually generated by attack algorithms, white-noise-induced adversarial samples can exist, and therefore the threats are real. In this study, we propose a novel training method, named IMA, to improve the robust-ness of CNN against adversarial noises. During training, the IMA method increases the margins of training samples in the input space, i.e., moving CNN decision boundaries far away from the training samples to improve robustness. The IMA method is evaluated on publicly available datasets under strong 100-PGD white-box adversarial attacks, and the results show that the proposed method significantly improved CNN classification and segmentation accuracy on noisy data while keeping a high accuracy on clean data. We hope our approach may facilitate the development of robust applications in medical field.

摘要: 卷积神经网络(CNN)已经超越了传统的医学图像分类方法。然而，CNN很容易受到对抗性攻击，这可能会导致医疗应用中的灾难性后果。虽然攻击算法通常会产生对抗性噪声，但白噪声诱导的对抗性样本可能存在，因此威胁是真实存在的。在这项研究中，我们提出了一种新的训练方法，称为IMA，以提高CNN对对抗性噪声的鲁棒性。在训练过程中，IMA方法增加了输入空间中训练样本的边际，即使CNN决策边界远离训练样本，以提高鲁棒性。在100-PGD强白盒攻击下，在公开数据集上对IMA方法进行了评估，结果表明，该方法在保持对干净数据较高精度的同时，显著提高了对含噪声数据的CNN分类和分割的准确率。我们希望我们的方法可以促进医学领域健壮应用的发展。



## **17. Backdoor Learning: A Survey**

借壳学习：一项调查 cs.CR

17 pages. A curated list of backdoor learning resources in this paper  is presented in the Github Repo  (https://github.com/THUYimingLi/backdoor-learning-resources). We will try our  best to continuously maintain this Github Repo

**SubmitDate**: 2022-02-16    [paper-pdf](http://arxiv.org/pdf/2007.08745v5)

**Authors**: Yiming Li, Yong Jiang, Zhifeng Li, Shu-Tao Xia

**Abstracts**: Backdoor attack intends to embed hidden backdoor into deep neural networks (DNNs), so that the attacked models perform well on benign samples, whereas their predictions will be maliciously changed if the hidden backdoor is activated by attacker-specified triggers. This threat could happen when the training process is not fully controlled, such as training on third-party datasets or adopting third-party models, which poses a new and realistic threat. Although backdoor learning is an emerging and rapidly growing research area, its systematic review, however, remains blank. In this paper, we present the first comprehensive survey of this realm. We summarize and categorize existing backdoor attacks and defenses based on their characteristics, and provide a unified framework for analyzing poisoning-based backdoor attacks. Besides, we also analyze the relation between backdoor attacks and relevant fields ($i.e.,$ adversarial attacks and data poisoning), and summarize widely adopted benchmark datasets. Finally, we briefly outline certain future research directions relying upon reviewed works. A curated list of backdoor-related resources is also available at \url{https://github.com/THUYimingLi/backdoor-learning-resources}.

摘要: 后门攻击的目的是将隐藏的后门嵌入到深度神经网络(DNNs)中，使得被攻击的模型在良性样本上表现良好，而如果隐藏的后门被攻击者指定的触发器激活，则其预测将被恶意改变。这种威胁可能发生在培训过程没有得到完全控制时，例如在第三方数据集上进行培训或采用第三方模型，这会构成新的现实威胁。虽然借壳学习是一个新兴的、发展迅速的研究领域，但其系统评价仍然是空白。在这篇文章中，我们首次对这一领域进行了全面的调查。根据后门攻击和防御的特点，对现有的后门攻击和防御进行了总结和分类，为分析基于中毒的后门攻击提供了一个统一的框架。此外，我们还分析了后门攻击与相关领域($对抗性攻击和数据中毒)之间的关系，并总结了广泛采用的基准数据集。最后，在回顾工作的基础上，简要概述了未来的研究方向。\url{https://github.com/THUYimingLi/backdoor-learning-resources}.上还提供了与后门相关的资源的精选列表



## **18. FedCG: Leverage Conditional GAN for Protecting Privacy and Maintaining Competitive Performance in Federated Learning**

FedCG：利用条件GAN保护隐私并保持联合学习中的好胜性能 cs.LG

**SubmitDate**: 2022-02-16    [paper-pdf](http://arxiv.org/pdf/2111.08211v2)

**Authors**: Yuezhou Wu, Yan Kang, Jiahuan Luo, Yuanqin He, Qiang Yang

**Abstracts**: Federated learning (FL) aims to protect data privacy by enabling clients to build machine learning models collaboratively without sharing their private data. Recent works demonstrate that information exchanged during FL is subject to gradient-based privacy attacks and, consequently, a variety of privacy-preserving methods have been adopted to thwart such attacks. However, these defensive methods either introduce orders of magnitudes more computational and communication overheads (e.g., with homomorphic encryption) or incur substantial model performance losses in terms of prediction accuracy (e.g., with differential privacy). In this work, we propose $\textsc{FedCG}$, a novel federated learning method that leverages conditional generative adversarial networks to achieve high-level privacy protection while still maintaining competitive model performance. $\textsc{FedCG}$ decomposes each client's local network into a private extractor and a public classifier and keeps the extractor local to protect privacy. Instead of exposing extractors, $\textsc{FedCG}$ shares clients' generators with the server for aggregating clients' shared knowledge aiming to enhance the performance of each client's local networks. Extensive experiments demonstrate that $\textsc{FedCG}$ can achieve competitive model performance compared with FL baselines, and privacy analysis shows that $\textsc{FedCG}$ has a high-level privacy-preserving capability.

摘要: 联合学习(FL)旨在通过使客户能够在不共享其私有数据的情况下协作地构建机器学习模型来保护数据隐私。最近的研究表明，在外语学习过程中交换的信息会受到基于梯度的隐私攻击，因此，已经采取了各种隐私保护方法来阻止这种攻击。然而，这些防御方法要么引入更多数量级的计算和通信开销(例如，利用同态加密)，要么在预测精度方面招致大量的模型性能损失(例如，利用差分保密)。在这项工作中，我们提出了一种新的联邦学习方法$\textsc{fedcg}$，它利用条件生成性对抗网络来实现高级别的隐私保护，同时保持好胜模型的性能。$\textsc{fedcg}$将每个客户端的本地网络分解为私有提取器和公共分类器，并将提取器保留在本地以保护隐私。$\textsc{FedCG}$不公开提取器，而是与服务器共享客户端生成器，用于聚合客户端共享的知识，旨在增强每个客户端的本地网络的性能。大量实验表明，与FL基线相比，$\textsc{fedcg}$能够达到好胜模型的性能，隐私分析表明$\textsc{fedcg}$具有较高的隐私保护能力。



## **19. Generative Adversarial Network-Driven Detection of Adversarial Tasks in Mobile Crowdsensing**

生成式对抗性网络驱动的移动树冠感知对抗性任务检测 cs.CR

This paper contains pages, 4 figures which is accepted by IEEE ICC  2022

**SubmitDate**: 2022-02-16    [paper-pdf](http://arxiv.org/pdf/2202.07802v1)

**Authors**: Zhiyan Chen, Burak Kantarci

**Abstracts**: Mobile Crowdsensing systems are vulnerable to various attacks as they build on non-dedicated and ubiquitous properties. Machine learning (ML)-based approaches are widely investigated to build attack detection systems and ensure MCS systems security. However, adversaries that aim to clog the sensing front-end and MCS back-end leverage intelligent techniques, which are challenging for MCS platform and service providers to develop appropriate detection frameworks against these attacks. Generative Adversarial Networks (GANs) have been applied to generate synthetic samples, that are extremely similar to the real ones, deceiving classifiers such that the synthetic samples are indistinguishable from the originals. Previous works suggest that GAN-based attacks exhibit more crucial devastation than empirically designed attack samples, and result in low detection rate at the MCS platform. With this in mind, this paper aims to detect intelligently designed illegitimate sensing service requests by integrating a GAN-based model. To this end, we propose a two-level cascading classifier that combines the GAN discriminator with a binary classifier to prevent adversarial fake tasks. Through simulations, we compare our results to a single-level binary classifier, and the numeric results show that proposed approach raises Adversarial Attack Detection Rate (AADR), from $0\%$ to $97.5\%$ by KNN/NB, from $45.9\%$ to $100\%$ by Decision Tree. Meanwhile, with two-levels classifiers, Original Attack Detection Rate (OADR) improves for the three binary classifiers, with comparison, such as NB from $26.1\%$ to $61.5\%$.

摘要: 由于移动树冠传感系统建立在非专用和无处不在的特性之上，因此容易受到各种攻击。基于机器学习(ML)的方法在构建攻击检测系统和保证MCS系统安全方面得到了广泛的研究。然而，旨在阻塞传感前端和MCS后端的攻击者利用智能技术，这对MCS平台和服务提供商开发针对这些攻击的适当检测框架是具有挑战性的。生成性对抗网络(GANS)被用来生成与真实样本极其相似的合成样本，欺骗分类器，使得合成样本与原始样本无法区分。以往的工作表明，基于GAN的攻击比经验设计的攻击样本表现出更严重的破坏性，导致MCS平台的检测率较低。考虑到这一点，本文旨在通过集成一个基于GAN的模型来检测智能设计的非法传感服务请求。为此，我们提出了一种将GAN鉴别器和二进制分类器相结合的两级级联分类器，以防止敌意虚假任务。通过仿真，我们将我们的结果与单级二进制分类器进行了比较，数值结果表明，该方法将对手攻击检测率(AADR)从0美元提高到97.5美元(KNN/NB从0美元提高到97.5美元)，通过决策树将AADR从4 5.9美元提高到1 0 0美元。同时，在使用两级分类器的情况下，三种二值分类器的原始攻击检测率(OADR)都有不同程度的提高，例如NB从26.1美元提高到61.5美元。



## **20. Vulnerability-Aware Poisoning Mechanism for Online RL with Unknown Dynamics**

动态未知的在线RL漏洞感知中毒机制 cs.LG

**SubmitDate**: 2022-02-15    [paper-pdf](http://arxiv.org/pdf/2009.00774v5)

**Authors**: Yanchao Sun, Da Huo, Furong Huang

**Abstracts**: Poisoning attacks on Reinforcement Learning (RL) systems could take advantage of RL algorithm's vulnerabilities and cause failure of the learning. However, prior works on poisoning RL usually either unrealistically assume the attacker knows the underlying Markov Decision Process (MDP), or directly apply the poisoning methods in supervised learning to RL. In this work, we build a generic poisoning framework for online RL via a comprehensive investigation of heterogeneous poisoning models in RL. Without any prior knowledge of the MDP, we propose a strategic poisoning algorithm called Vulnerability-Aware Adversarial Critic Poison (VA2C-P), which works for most policy-based deep RL agents, closing the gap that no poisoning method exists for policy-based RL agents. VA2C-P uses a novel metric, stability radius in RL, that measures the vulnerability of RL algorithms. Experiments on multiple deep RL agents and multiple environments show that our poisoning algorithm successfully prevents agents from learning a good policy or teaches the agents to converge to a target policy, with a limited attacking budget.

摘要: 对强化学习(RL)系统的毒化攻击可以利用RL算法的脆弱性，导致学习失败。然而，以往的毒化RL的工作通常要么不切实际地假设攻击者知道潜在的马尔可夫决策过程(MDP)，要么直接将有监督学习中的毒化方法应用于RL。在这项工作中，我们通过对RL中异构中毒模型的全面研究，构建了一个适用于在线RL的通用中毒框架。在对MDP没有任何先验知识的情况下，我们提出了一种策略毒化算法VA2C-P(VA2C-P)，该算法适用于大多数基于策略的深度RL代理，弥补了基于策略的RL代理没有毒化方法的空白。VA2C-P使用了一种新的度量，即RL中的稳定半径，该度量度量了RL算法的脆弱性。在多个深度RL代理和多个环境上的实验表明，我们的中毒算法在有限的攻击预算下，成功地阻止了代理学习好的策略或教导代理收敛到目标策略。



## **21. Defending against Reconstruction Attacks with Rényi Differential Privacy**

利用Rényi差分私密性防御重构攻击 cs.LG

**SubmitDate**: 2022-02-15    [paper-pdf](http://arxiv.org/pdf/2202.07623v1)

**Authors**: Pierre Stock, Igor Shilov, Ilya Mironov, Alexandre Sablayrolles

**Abstracts**: Reconstruction attacks allow an adversary to regenerate data samples of the training set using access to only a trained model. It has been recently shown that simple heuristics can reconstruct data samples from language models, making this threat scenario an important aspect of model release. Differential privacy is a known solution to such attacks, but is often used with a relatively large privacy budget (epsilon > 8) which does not translate to meaningful guarantees. In this paper we show that, for a same mechanism, we can derive privacy guarantees for reconstruction attacks that are better than the traditional ones from the literature. In particular, we show that larger privacy budgets do not protect against membership inference, but can still protect extraction of rare secrets. We show experimentally that our guarantees hold against various language models, including GPT-2 finetuned on Wikitext-103.

摘要: 重构攻击允许对手仅使用对训练模型的访问来重新生成训练集的数据样本。最近的研究表明，简单的启发式算法可以从语言模型中重建数据样本，从而使这种威胁场景成为模型发布的一个重要方面。差异隐私是此类攻击的已知解决方案，但通常使用相对较大的隐私预算(epsilon>8)，这并不能转化为有意义的保证。在本文中，我们表明，对于相同的机制，我们可以从文献中推导出比传统的重构攻击更好的隐私保证。特别是，我们表明，较大的隐私预算不能防止成员关系推断，但仍然可以保护罕见秘密的提取。我们的实验表明，我们的保证适用于各种语言模型，包括在Wikitext-103上微调的GPT-2。



## **22. StratDef: a strategic defense against adversarial attacks in malware detection**

StratDef：恶意软件检测中对抗对手攻击的战略防御 cs.LG

**SubmitDate**: 2022-02-15    [paper-pdf](http://arxiv.org/pdf/2202.07568v1)

**Authors**: Aqib Rashid, Jose Such

**Abstracts**: Over the years, most research towards defenses against adversarial attacks on machine learning models has been in the image processing domain. The malware detection domain has received less attention despite its importance. Moreover, most work exploring defenses focuses on feature-based, gradient-based or randomized methods but with no strategy when applying them. In this paper, we introduce StratDef, which is a strategic defense system tailored for the malware detection domain based on a Moving Target Defense and Game Theory approach. We overcome challenges related to the systematic construction, selection and strategic use of models to maximize adversarial robustness. StratDef dynamically and strategically chooses the best models to increase the uncertainty for the attacker, whilst minimizing critical aspects in the adversarial ML domain like attack transferability. We provide the first comprehensive evaluation of defenses against adversarial attacks on machine learning for malware detection, where our threat model explores different levels of threat, attacker knowledge, capabilities, and attack intensities. We show that StratDef performs better than other defenses even when facing the peak adversarial threat. We also show that, from the existing defenses, only a few adversarially-trained models provide substantially better protection than just using vanilla models but are still outperformed by StratDef.

摘要: 多年来，针对机器学习模型抗敌意攻击的研究大多集中在图像处理领域。恶意软件检测域尽管很重要，但受到的关注较少。而且，大多数探索防御的工作都集中在基于特征的、基于梯度的或随机的方法上，而在应用这些方法时没有策略。本文介绍了StratDef，这是一个基于移动目标防御和博弈论的针对恶意软件检测领域定制的战略防御系统。我们克服了与模型的系统构建、选择和战略使用相关的挑战，以最大限度地提高对手的健壮性。StratDef动态地和战略性地选择最佳模型，以增加攻击者的不确定性，同时最小化敌对ML领域中的关键方面，如攻击可转移性。我们在恶意软件检测的机器学习中首次全面评估了防御敌意攻击的能力，其中我们的威胁模型探索了不同级别的威胁、攻击者知识、能力和攻击强度。我们表明，即使在面临最严重的对手威胁时，StratDef也比其他防御系统表现得更好。我们还表明，从现有的防御来看，只有少数经过对抗性训练的模型提供了比仅仅使用普通模型更好的保护，但仍然优于StratDef。



## **23. Random Walks for Adversarial Meshes**

对抗性网格的随机游动 cs.CV

**SubmitDate**: 2022-02-15    [paper-pdf](http://arxiv.org/pdf/2202.07453v1)

**Authors**: Amir Belder, Gal Yefet, Ran Ben Izhak, Ayellet Tal

**Abstracts**: A polygonal mesh is the most-commonly used representation of surfaces in computer graphics; thus, a variety of classification networks have been recently proposed. However, while adversarial attacks are wildly researched in 2D, almost no works on adversarial meshes exist. This paper proposes a novel, unified, and general adversarial attack, which leads to misclassification of numerous state-of-the-art mesh classification neural networks. Our attack approach is black-box, i.e. it has access only to the network's predictions, but not to the network's full architecture or gradients. The key idea is to train a network to imitate a given classification network. This is done by utilizing random walks along the mesh surface, which gather geometric information. These walks provide insight onto the regions of the mesh that are important for the correct prediction of the given classification network. These mesh regions are then modified more than other regions in order to attack the network in a manner that is barely visible to the naked eye.

摘要: 多边形网格是计算机图形学中最常用的曲面表示，因此，最近提出了各种分类网络。然而，尽管对抗性攻击在2D方面得到了广泛的研究，但几乎没有关于对抗性网络的工作。本文提出了一种新颖的、统一的、通用的对抗性攻击，该攻击导致了众多最新的网格分类神经网络的误分类。我们的攻击方法是黑匣子，即它只能访问网络的预测，而不能访问网络的完整架构或梯度。其核心思想是训练一个网络来模仿给定的分类网络。这是通过利用沿网格曲面的随机漫游来完成的，该漫游收集几何信息。这些遍历提供了对网格区域的洞察，这些区域对于给定分类网络的正确预测非常重要。然后，这些网格区域被修改得比其他区域更多，以便以肉眼几乎看不见的方式攻击网络。



## **24. Unreasonable Effectiveness of Last Hidden Layer Activations**

最后一次隐藏层激活的不合理效果 cs.LG

22 pages, Under review

**SubmitDate**: 2022-02-15    [paper-pdf](http://arxiv.org/pdf/2202.07342v1)

**Authors**: Omer Faruk Tuna, Ferhat Ozgur Catak, M. Taner Eskil

**Abstracts**: In standard Deep Neural Network (DNN) based classifiers, the general convention is to omit the activation function in the last (output) layer and directly apply the softmax function on the logits to get the probability scores of each class. In this type of architectures, the loss value of the classifier against any output class is directly proportional to the difference between the final probability score and the label value of the associated class. Standard White-box adversarial evasion attacks, whether targeted or untargeted, mainly try to exploit the gradient of the model loss function to craft adversarial samples and fool the model. In this study, we show both mathematically and experimentally that using some widely known activation functions in the output layer of the model with high temperature values has the effect of zeroing out the gradients for both targeted and untargeted attack cases, preventing attackers from exploiting the model's loss function to craft adversarial samples. We've experimentally verified the efficacy of our approach on MNIST (Digit), CIFAR10 datasets. Detailed experiments confirmed that our approach substantially improves robustness against gradient-based targeted and untargeted attack threats. And, we showed that the increased non-linearity at the output layer has some additional benefits against some other attack methods like Deepfool attack.

摘要: 在标准的基于深度神经网络(DNN)的分类器中，一般的做法是省略最后一层(输出层)的激活函数，直接对Logit应用Softmax函数来得到每一类的概率得分。在这种类型的体系结构中，分类器相对于任何输出类别的损失值与最终概率得分和相关类别的标签值之间的差值成正比。标准的白盒对抗性规避攻击，无论是有针对性的还是无针对性的，主要是利用模型损失函数的梯度来伪造对抗性样本，愚弄模型。在这项研究中，我们从数学和实验两个方面证明了在具有高温值的模型输出层使用一些广为人知的激活函数具有将目标攻击和非目标攻击的梯度归零的效果，防止攻击者利用模型的损失函数来伪造敌意样本。我们已经在MNIST(数字)、CIFAR10数据集上实验验证了我们的方法的有效性。详细的实验证实，我们的方法大大提高了对基于梯度的目标攻击和非目标攻击威胁的鲁棒性。并且，我们还表明，在输出层增加的非线性比其他一些攻击方法(如DeepfoOff攻击)有一些额外的好处。



## **25. Unity is strength: Improving the Detection of Adversarial Examples with Ensemble Approaches**

团结就是力量：用集成方法改进对抗性实例的检测 cs.CV

Code is available at https://github.com/BIMIB-DISCo/ENAD-experiments

**SubmitDate**: 2022-02-15    [paper-pdf](http://arxiv.org/pdf/2111.12631v3)

**Authors**: Francesco Craighero, Fabrizio Angaroni, Fabio Stella, Chiara Damiani, Marco Antoniotti, Alex Graudenzi

**Abstracts**: A key challenge in computer vision and deep learning is the definition of robust strategies for the detection of adversarial examples. Here, we propose the adoption of ensemble approaches to leverage the effectiveness of multiple detectors in exploiting distinct properties of the input data. To this end, the ENsemble Adversarial Detector (ENAD) framework integrates scoring functions from state-of-the-art detectors based on Mahalanobis distance, Local Intrinsic Dimensionality, and One-Class Support Vector Machines, which process the hidden features of deep neural networks. ENAD is designed to ensure high standardization and reproducibility to the computational workflow. Importantly, extensive tests on benchmark datasets, models and adversarial attacks show that ENAD outperforms all competing methods in the large majority of settings. The improvement over the state-of-the-art and the intrinsic generality of the framework, which allows one to easily extend ENAD to include any set of detectors, set the foundations for the new area of ensemble adversarial detection.

摘要: 计算机视觉和深度学习中的一个关键挑战是定义用于检测对抗性示例的鲁棒策略。在这里，我们建议采用集成方法来利用多个检测器的有效性来利用输入数据的不同属性。为此，集成敌意检测器(ENAD)框架集成了基于马氏距离、局部本征维数和一类支持向量机的最新检测器的评分函数，这些功能处理了深层神经网络的隐藏特征。ENAD旨在确保计算工作流的高度标准化和重复性。重要的是，对基准数据集、模型和对抗性攻击的广泛测试表明，ENAD在绝大多数情况下都优于所有竞争方法。对现有技术的改进和框架固有的通用性，使得人们可以很容易地将ENAD扩展到包括任何一组检测器，为集成对手检测的新领域奠定了基础。



## **26. Layer-wise Regularized Adversarial Training using Layers Sustainability Analysis (LSA) framework**

基于层次可持续性分析(LSA)框架的分层正则化对抗性训练 cs.CV

Layers Sustainability Analysis (LSA) framework

**SubmitDate**: 2022-02-15    [paper-pdf](http://arxiv.org/pdf/2202.02626v3)

**Authors**: Mohammad Khalooei, Mohammad Mehdi Homayounpour, Maryam Amirmazlaghani

**Abstracts**: Deep neural network models are used today in various applications of artificial intelligence, the strengthening of which, in the face of adversarial attacks is of particular importance. An appropriate solution to adversarial attacks is adversarial training, which reaches a trade-off between robustness and generalization. This paper introduces a novel framework (Layer Sustainability Analysis (LSA)) for the analysis of layer vulnerability in an arbitrary neural network in the scenario of adversarial attacks. LSA can be a helpful toolkit to assess deep neural networks and to extend the adversarial training approaches towards improving the sustainability of model layers via layer monitoring and analysis. The LSA framework identifies a list of Most Vulnerable Layers (MVL list) of the given network. The relative error, as a comparison measure, is used to evaluate representation sustainability of each layer against adversarial inputs. The proposed approach for obtaining robust neural networks to fend off adversarial attacks is based on a layer-wise regularization (LR) over LSA proposal(s) for adversarial training (AT); i.e. the AT-LR procedure. AT-LR could be used with any benchmark adversarial attack to reduce the vulnerability of network layers and to improve conventional adversarial training approaches. The proposed idea performs well theoretically and experimentally for state-of-the-art multilayer perceptron and convolutional neural network architectures. Compared with the AT-LR and its corresponding base adversarial training, the classification accuracy of more significant perturbations increased by 16.35%, 21.79%, and 10.730% on Moon, MNIST, and CIFAR-10 benchmark datasets, respectively. The LSA framework is available and published at https://github.com/khalooei/LSA.

摘要: 深度神经网络模型在当今人工智能的各种应用中都有应用，在面对敌意攻击时，加强深度神经网络模型的应用显得尤为重要。对抗性攻击的一个合适的解决方案是对抗性训练，它在鲁棒性和泛化之间达到了折衷。提出了一种新的分析任意神经网络层脆弱性的框架(层可持续性分析LSA)，用于分析任意神经网络在敌意攻击情况下的层脆弱性。LSA可作为评估深层神经网络和扩展对抗性训练方法的有用工具包，以便通过层监控和分析来提高模型层的可持续性。LSA框架标识给定网络的最易受攻击的层的列表(MVL列表)。以相对误差作为比较尺度，评价各层对敌方输入的表征可持续性。所提出的获得鲁棒神经网络以抵御对手攻击的方法是基于基于LSA的对抗性训练(AT)方案的分层正则化(LR)，即AT-LR过程。AT-LR可以与任何基准对抗性攻击一起使用，以降低网络层的脆弱性，并改进传统的对抗性训练方法。对于最先进的多层感知器和卷积神经网络结构，所提出的思想在理论和实验上都表现良好。在MOND、MNIST和CIFAR-10基准数据集上，与AT-LR及其相应的基础对抗性训练相比，更显著扰动的分类准确率分别提高了16.35%、21.79%和10.730%。可以在https://github.com/khalooei/LSA.上获得并发布lsa框架。



## **27. Holistic Adversarial Robustness of Deep Learning Models**

深度学习模型的整体对抗鲁棒性 cs.LG

survey paper on holistic adversarial robustness for deep learning

**SubmitDate**: 2022-02-15    [paper-pdf](http://arxiv.org/pdf/2202.07201v1)

**Authors**: Pin-Yu Chen, Sijia Liu

**Abstracts**: Adversarial robustness studies the worst-case performance of a machine learning model to ensure safety and reliability. With the proliferation of deep-learning based technology, the potential risks associated with model development and deployment can be amplified and become dreadful vulnerabilities. This paper provides a comprehensive overview of research topics and foundational principles of research methods for adversarial robustness of deep learning models, including attacks, defenses, verification, and novel applications.

摘要: 对抗鲁棒性研究机器学习模型的最坏情况性能，以确保安全性和可靠性。随着基于深度学习的技术的激增，与模型开发和部署相关的潜在风险可能会被放大，并成为可怕的漏洞。本文综述了深度学习模型对抗性稳健性的研究主题和基本原理，包括攻击、防御、验证和新的应用。



## **28. Resilience from Diversity: Population-based approach to harden models against adversarial attacks**

来自多样性的弹性：基于人口的方法来强化模型对抗对手攻击的能力 cs.LG

12 pages, 6 figures, 5 tables

**SubmitDate**: 2022-02-15    [paper-pdf](http://arxiv.org/pdf/2111.10272v2)

**Authors**: Jasser Jasser, Ivan Garibay

**Abstracts**: Traditional deep learning networks (DNN) exhibit intriguing vulnerabilities that allow an attacker to force them to fail at their task. Notorious attacks such as the Fast Gradient Sign Method (FGSM) and the more powerful Projected Gradient Descent (PGD) generate adversarial samples by adding a magnitude of perturbation $\epsilon$ to the input's computed gradient, resulting in a deterioration of the effectiveness of the model's classification. This work introduces a model that is resilient to adversarial attacks. Our model leverages an established mechanism of defense which utilizes randomness and a population of DNNs. More precisely, our model consists of a population of $n$ diverse submodels, each one of them trained to individually obtain a high accuracy for the task at hand, while forced to maintain meaningful differences in their weights. Each time our model receives a classification query, it selects a submodel from its population at random to answer the query. To counter the attack transferability, diversity is introduced and maintained in the population of submodels. Thus introducing the concept of counter linking weights. A Counter-Linked Model (CLM) consists of a population of DNNs of the same architecture where a periodic random similarity examination is conducted during the simultaneous training to guarantee diversity while maintaining accuracy. Though the randomization technique proved to be resilient against adversarial attacks, we show that by retraining the DNNs ensemble or training them from the start with counter linking would enhance the robustness by around 20\% when tested on the MNIST dataset and at least 15\% when tested on the CIFAR-10 dataset. When CLM is coupled with adversarial training, this defense mechanism achieves state-of-the-art robustness.

摘要: 传统的深度学习网络(DNN)表现出耐人寻味的漏洞，使得攻击者能够迫使它们在任务中失败。诸如快速梯度符号法(FGSM)和更强大的投影梯度下降法(PGD)等臭名昭著的攻击通过在输入的计算梯度上添加扰动幅度$\ε$来生成敌意样本，导致模型分类效果的恶化。这项工作引入了一个对对手攻击具有弹性的模型。我们的模型利用了一种已建立的防御机制，该机制利用了随机性和DNN的种群。更准确地说，我们的模型由$n$各式各样的子模型组成，每个子模型都经过训练，以单独获得手头任务的高精度，同时被迫保持有意义的权重差异。我们的模型每次收到分类查询时，都会从其总体中随机选择一个子模型来回答查询。为了对抗攻击的可传递性，在子模型种群中引入并保持多样性。从而引入了计数器链接权重的概念。反向链接模型(CLM)由同一体系结构的一组DNN组成，其中在同时训练期间进行周期性的随机相似性检查，以在保持准确性的同时保证多样性。虽然随机化技术被证明对对手攻击具有弹性，但我们表明，通过重新训练DNN集成或从计数器链接开始训练DNN集成，在MNIST数据集上测试时将鲁棒性提高约20\%，在CIFAR-10数据集上测试时至少提高15\%。当CLM与对抗性训练相结合时，这种防御机制实现了最先进的健壮性。



## **29. Recent Advances in Reliable Deep Graph Learning: Adversarial Attack, Inherent Noise, and Distribution Shift**

可靠深度图学习的最新进展：对抗性攻击、固有噪声和分布偏移 cs.LG

**SubmitDate**: 2022-02-15    [paper-pdf](http://arxiv.org/pdf/2202.07114v1)

**Authors**: Bingzhe Wu, Jintang Li, Chengbin Hou, Guoji Fu, Yatao Bian, Liang Chen, Junzhou Huang

**Abstracts**: Deep graph learning (DGL) has achieved remarkable progress in both business and scientific areas ranging from finance and e-commerce to drug and advanced material discovery. Despite the progress, applying DGL to real-world applications faces a series of reliability threats including adversarial attacks, inherent noise, and distribution shift. This survey aims to provide a comprehensive review of recent advances for improving the reliability of DGL algorithms against the above threats. In contrast to prior related surveys which mainly focus on adversarial attacks and defense, our survey covers more reliability-related aspects of DGL, i.e., inherent noise and distribution shift. Additionally, we discuss the relationships among above aspects and highlight some important issues to be explored in future research.

摘要: 深度图学习(DGL)在从金融和电子商务到药物和先进材料发现的商业和科学领域都取得了显着的进展。尽管取得了进展，但将DGL应用于实际应用程序面临着一系列的可靠性威胁，包括敌意攻击、固有噪声和分布转移。本调查旨在全面回顾提高DGL算法抵御上述威胁的可靠性的最新进展。与以前主要关注对抗性攻击和防御的相关调查不同，我们的调查涵盖了DGL更多与可靠性相关的方面，即固有噪声和分布偏移。此外，我们还讨论了上述几个方面之间的关系，并指出了未来研究中需要探索的一些重要问题。



## **30. Universal Adversarial Examples in Remote Sensing: Methodology and Benchmark**

遥感领域的普遍对抗性实例：方法论和基准 cs.CV

**SubmitDate**: 2022-02-14    [paper-pdf](http://arxiv.org/pdf/2202.07054v1)

**Authors**: Yonghao Xu, Pedram Ghamisi

**Abstracts**: Deep neural networks have achieved great success in many important remote sensing tasks. Nevertheless, their vulnerability to adversarial examples should not be neglected. In this study, we systematically analyze the universal adversarial examples in remote sensing data for the first time, without any knowledge from the victim model. Specifically, we propose a novel black-box adversarial attack method, namely Mixup-Attack, and its simple variant Mixcut-Attack, for remote sensing data. The key idea of the proposed methods is to find common vulnerabilities among different networks by attacking the features in the shallow layer of a given surrogate model. Despite their simplicity, the proposed methods can generate transferable adversarial examples that deceive most of the state-of-the-art deep neural networks in both scene classification and semantic segmentation tasks with high success rates. We further provide the generated universal adversarial examples in the dataset named UAE-RS, which is the first dataset that provides black-box adversarial samples in the remote sensing field. We hope UAE-RS may serve as a benchmark that helps researchers to design deep neural networks with strong resistance toward adversarial attacks in the remote sensing field. Codes and the UAE-RS dataset will be available online.

摘要: 深度神经网络在许多重要的遥感任务中取得了巨大的成功。然而，他们面对敌对例子的脆弱性不应被忽视。在本研究中，我们在没有任何受害者模型知识的情况下，首次系统地分析了遥感数据中普遍存在的对抗性实例。具体地说，针对遥感数据，我们提出了一种新的黑盒对抗攻击方法，即Mixup-Attack及其简单的变种MixCut-Attack。该方法的核心思想是通过攻击给定代理模型浅层的特征来发现不同网络之间的共同漏洞。尽管方法简单，但是在场景分类和语义分割任务中，所提出的方法可以生成可转移的对抗性示例，欺骗了大多数最新的深度神经网络，并且成功率很高。此外，我们还给出了在遥感领域第一个提供黑盒对抗性样本的数据集UAE-RS中生成的通用对抗性实例。我们希望UAE-RS可以作为一个基准，帮助研究人员设计出对遥感领域的敌意攻击具有很强抵抗力的深层神经网络。代码和阿联酋-RS数据集将在网上提供。



## **31. White-Box Attacks on Hate-speech BERT Classifiers in German with Explicit and Implicit Character Level Defense**

德语仇恨言语BERT分类器的显性和隐性特征防御白盒攻击 cs.CL

**SubmitDate**: 2022-02-14    [paper-pdf](http://arxiv.org/pdf/2202.05778v2)

**Authors**: Shahrukh Khan, Mahnoor Shahid, Navdeeppal Singh

**Abstracts**: In this work, we evaluate the adversarial robustness of BERT models trained on German Hate Speech datasets. We also complement our evaluation with two novel white-box character and word level attacks thereby contributing to the range of attacks available. Furthermore, we also perform a comparison of two novel character-level defense strategies and evaluate their robustness with one another.

摘要: 在这项工作中，我们评估了在德国仇恨语音数据集上训练的BERT模型的对抗鲁棒性。我们还用两种新的白盒字符和词级攻击来补充我们的评估，从而增加了可用的攻击范围。此外，我们还对两种新的字符级防御策略进行了比较，并对它们的鲁棒性进行了评估。



## **32. Robust and Information-theoretically Safe Bias Classifier against Adversarial Attacks**

抗敌意攻击的稳健且信息理论安全的偏向分类器 cs.LG

**SubmitDate**: 2022-02-14    [paper-pdf](http://arxiv.org/pdf/2111.04404v2)

**Authors**: Lijia Yu, Xiao-Shan Gao

**Abstracts**: In this paper, the bias classifier is introduced, that is, the bias part of a DNN with Relu as the activation function is used as a classifier. The work is motivated by the fact that the bias part is a piecewise constant function with zero gradient and hence cannot be directly attacked by gradient-based methods to generate adversaries, such as FGSM. The existence of the bias classifier is proved and an effective training method for the bias classifier is given. It is proved that by adding a proper random first-degree part to the bias classifier, an information-theoretically safe classifier against the original-model gradient attack is obtained in the sense that the attack will generate a totally random attacking direction. This seems to be the first time that the concept of information-theoretically safe classifier is proposed. Several attack methods for the bias classifier are proposed and numerical experiments are used to show that the bias classifier is more robust than DNNs with similar size against these attacks in most cases.

摘要: 本文介绍了偏向分类器，即以RELU为激活函数的DNN的偏向部分作为分类器。这项工作的动机是，偏差部分是一个分段常数函数，具有零梯度，因此不能被基于梯度的方法直接攻击来生成对手，如FGSM。证明了偏向分类器的存在性，并给出了一种有效的偏向分类器训练方法。证明了通过在偏向分类器中加入适当的随机一阶部分，在攻击产生完全随机的攻击方向的意义下，得到了一种信息论上安全的分类器，可以抵抗原模型的梯度攻击。这似乎是首次提出信息理论安全分类器的概念。提出了偏向分类器的几种攻击方法，数值实验表明，在大多数情况下，偏向分类器比大小相近的DNN具有更好的鲁棒性。



## **33. Robustness against Adversarial Attacks in Neural Networks using Incremental Dissipativity**

基于增量耗散的神经网络对敌意攻击的鲁棒性 cs.LG

**SubmitDate**: 2022-02-14    [paper-pdf](http://arxiv.org/pdf/2111.12906v2)

**Authors**: Bernardo Aquino, Arash Rahnama, Peter Seiler, Lizhen Lin, Vijay Gupta

**Abstracts**: Adversarial examples can easily degrade the classification performance in neural networks. Empirical methods for promoting robustness to such examples have been proposed, but often lack both analytical insights and formal guarantees. Recently, some robustness certificates have appeared in the literature based on system theoretic notions. This work proposes an incremental dissipativity-based robustness certificate for neural networks in the form of a linear matrix inequality for each layer. We also propose an equivalent spectral norm bound for this certificate which is scalable to neural networks with multiple layers. We demonstrate the improved performance against adversarial attacks on a feed-forward neural network trained on MNIST and an Alexnet trained using CIFAR-10.

摘要: 对抗性示例很容易降低神经网络的分类性能。已经提出了提高此类例子稳健性的经验方法，但往往既缺乏分析洞察力，也缺乏形式上的保证。近年来，一些基于系统论概念的健壮性证书出现在文献中。本文以线性矩阵不等式的形式为每一层提出了一种基于耗散性的增量式神经网络鲁棒性证书。我们还给出了该证书的一个等价谱范数界，它可扩展到多层神经网络。我们在使用MNIST训练的前馈神经网络和使用CIFAR-10训练的Alexnet上展示了改进的抗敌意攻击性能。



## **34. Adversarial Fine-tuning for Backdoor Defense: Connect Adversarial Examples to Triggered Samples**

用于后门防御的对抗性微调：将对抗性示例连接到触发样本 cs.CV

**SubmitDate**: 2022-02-13    [paper-pdf](http://arxiv.org/pdf/2202.06312v1)

**Authors**: Bingxu Mu, Le Wang, Zhenxing Niu

**Abstracts**: Deep neural networks (DNNs) are known to be vulnerable to backdoor attacks, i.e., a backdoor trigger planted at training time, the infected DNN model would misclassify any testing sample embedded with the trigger as target label. Due to the stealthiness of backdoor attacks, it is hard either to detect or erase the backdoor from infected models. In this paper, we propose a new Adversarial Fine-Tuning (AFT) approach to erase backdoor triggers by leveraging adversarial examples of the infected model. For an infected model, we observe that its adversarial examples have similar behaviors as its triggered samples. Based on such observation, we design the AFT to break the foundation of the backdoor attack (i.e., the strong correlation between a trigger and a target label). We empirically show that, against 5 state-of-the-art backdoor attacks, AFT can effectively erase the backdoor triggers without obvious performance degradation on clean samples, which significantly outperforms existing defense methods.

摘要: 深度神经网络(DNNs)容易受到后门攻击，即在训练时植入后门触发器，被感染的DNN模型会将任何嵌入触发器的测试样本错误分类为目标标签。由于后门攻击的隐蔽性，很难从受感染的模型中检测或删除后门。在本文中，我们提出了一种新的对抗性精调(AFT)方法，通过利用感染模型的对抗性示例来擦除后门触发器。对于感染模型，我们观察到其敌意实例与其触发样本具有相似的行为。基于这样的观察，我们设计了AFT来打破后门攻击的基础(即触发器和目标标签之间的强相关性)。我们的实验表明，对于5种最先进的后门攻击，AFT可以有效地清除后门触发器，在干净样本上没有明显的性能下降，明显优于现有的防御方法。



## **35. Local Differential Privacy for Federated Learning in Industrial Settings**

工业环境下联合学习的局部差分隐私 cs.CR

14 pages

**SubmitDate**: 2022-02-12    [paper-pdf](http://arxiv.org/pdf/2202.06053v1)

**Authors**: M. A. P. Chamikara, Dongxi Liu, Seyit Camtepe, Surya Nepal, Marthie Grobler, Peter Bertok, Ibrahim Khalil

**Abstracts**: Federated learning (FL) is a collaborative learning approach that has gained much attention due to its inherent privacy preservation capabilities. However, advanced adversarial attacks such as membership inference and model memorization can still make FL vulnerable and potentially leak sensitive private data. Literature shows a few attempts to alleviate this problem by using global (GDP) and local differential privacy (LDP). Compared to GDP, LDP approaches are gaining more popularity due to stronger privacy notions and native support for data distribution. However, DP approaches assume that the server that aggregates the models, to be honest (run the FL protocol honestly) or semi-honest (run the FL protocol honestly while also trying to learn as much information possible), making such approaches unreliable for real-world settings. In real-world industrial environments (e.g. healthcare), the distributed entities (e.g. hospitals) are already composed of locally running machine learning models (e.g. high-performing deep neural networks on local health records). Existing approaches do not provide a scalable mechanism to utilize such settings for privacy-preserving FL. This paper proposes a new local differentially private FL (named LDPFL) protocol for industrial settings. LDPFL avoids the requirement of an honest or a semi-honest server and provides better performance while enforcing stronger privacy levels compared to existing approaches. Our experimental evaluation of LDPFL shows high FL model performance (up to ~98%) under a small privacy budget (e.g. epsilon = 0.5) in comparison to existing methods.

摘要: 联合学习(FL)是一种协作学习方式，由于其固有的隐私保护能力而备受关注。然而，高级对抗性攻击，如成员推断和模型记忆，仍然会使FL容易受到攻击，并可能泄露敏感的私有数据。文献显示，有几种尝试通过使用全局(GDP)和本地差异隐私(LDP)来缓解此问题。与GDP相比，由于更强的隐私概念和对数据分发的本地支持，LDP方法越来越受欢迎。然而，DP方法假定聚合模型的服务器诚实地(诚实地运行FL协议)或半诚实地(诚实地运行FL协议，同时还试图了解尽可能多的信息)，使得这种方法对于现实世界设置是不可靠的。在真实的工业环境中(例如医疗保健)，分布式实体(例如医院)已经由本地运行的机器学习模型(例如基于本地健康记录的高性能深度神经网络)组成。现有方法没有提供可扩展的机制来利用这种设置来保护隐私FL。提出了一种新的适用于工业环境的局部差分私有FL(简称LDPFL)协议。与现有方法相比，LDPFL避免了对诚实或半诚实服务器的要求，并提供了更好的性能，同时实施了更强的隐私级别。我们对LDPFL的实验评估表明，与现有方法相比，在较小的隐私预算(例如ε=0.5)下，FL模型的性能较高(高达~98%)。



## **36. RoPGen: Towards Robust Code Authorship Attribution via Automatic Coding Style Transformation**

RoPGen：通过自动代码风格转换实现健壮的代码作者属性 cs.CR

ICSE 2022

**SubmitDate**: 2022-02-12    [paper-pdf](http://arxiv.org/pdf/2202.06043v1)

**Authors**: Zhen Li, Guenevere, Chen, Chen Chen, Yayi Zou, Shouhuai Xu

**Abstracts**: Source code authorship attribution is an important problem often encountered in applications such as software forensics, bug fixing, and software quality analysis. Recent studies show that current source code authorship attribution methods can be compromised by attackers exploiting adversarial examples and coding style manipulation. This calls for robust solutions to the problem of code authorship attribution. In this paper, we initiate the study on making Deep Learning (DL)-based code authorship attribution robust. We propose an innovative framework called Robust coding style Patterns Generation (RoPGen), which essentially learns authors' unique coding style patterns that are hard for attackers to manipulate or imitate. The key idea is to combine data augmentation and gradient augmentation at the adversarial training phase. This effectively increases the diversity of training examples, generates meaningful perturbations to gradients of deep neural networks, and learns diversified representations of coding styles. We evaluate the effectiveness of RoPGen using four datasets of programs written in C, C++, and Java. Experimental results show that RoPGen can significantly improve the robustness of DL-based code authorship attribution, by respectively reducing 22.8% and 41.0% of the success rate of targeted and untargeted attacks on average.

摘要: 源代码作者归属是软件取证、缺陷修复、软件质量分析等应用中经常遇到的重要问题。最近的研究表明，现有的源代码作者归属方法可能会受到攻击者利用敌意示例和代码风格操纵的影响。这就要求对代码作者归属问题提出可靠的解决方案。本文针对基于深度学习(DL)的代码作者属性鲁棒性问题展开研究。我们提出了一种称为鲁棒编码样式模式生成(RoPGen)的创新框架，它本质上学习了攻击者难以操纵或模仿的作者独特的编码样式模式。其核心思想是在对抗性训练阶段将数据增强和梯度增强相结合。这有效地增加了训练样本的多样性，对深度神经网络的梯度产生了有意义的扰动，并学习了编码风格的多样化表示。我们使用四个用C、C++和Java编写的程序数据集来评估RoPGen的有效性。实验结果表明，RoPGen能够显著提高基于DL的代码作者属性的鲁棒性，目标攻击成功率平均降低22.8%，非目标攻击成功率平均降低41.0%。



## **37. Robust Deep Semi-Supervised Learning: A Brief Introduction**

鲁棒深度半监督学习：简介 cs.LG

**SubmitDate**: 2022-02-12    [paper-pdf](http://arxiv.org/pdf/2202.05975v1)

**Authors**: Lan-Zhe Guo, Zhi Zhou, Yu-Feng Li

**Abstracts**: Semi-supervised learning (SSL) is the branch of machine learning that aims to improve learning performance by leveraging unlabeled data when labels are insufficient. Recently, SSL with deep models has proven to be successful on standard benchmark tasks. However, they are still vulnerable to various robustness threats in real-world applications as these benchmarks provide perfect unlabeled data, while in realistic scenarios, unlabeled data could be corrupted. Many researchers have pointed out that after exploiting corrupted unlabeled data, SSL suffers severe performance degradation problems. Thus, there is an urgent need to develop SSL algorithms that could work robustly with corrupted unlabeled data. To fully understand robust SSL, we conduct a survey study. We first clarify a formal definition of robust SSL from the perspective of machine learning. Then, we classify the robustness threats into three categories: i) distribution corruption, i.e., unlabeled data distribution is mismatched with labeled data; ii) feature corruption, i.e., the features of unlabeled examples are adversarially attacked; and iii) label corruption, i.e., the label distribution of unlabeled data is imbalanced. Under this unified taxonomy, we provide a thorough review and discussion of recent works that focus on these issues. Finally, we propose possible promising directions within robust SSL to provide insights for future research.

摘要: 半监督学习(SSL)是机器学习的一个分支，其目的是在标签不足时通过利用未标记的数据来提高学习性能。最近，具有深度模型的SSL在标准基准任务中被证明是成功的。然而，它们在现实应用程序中仍然容易受到各种健壮性威胁，因为这些基准测试提供了完美的未标记数据，而在现实场景中，未标记数据可能会被破坏。许多研究人员指出，在利用损坏的未标记数据之后，SSL面临严重的性能下降问题。因此，迫切需要开发能够稳健地处理损坏的未标记数据的SSL算法。为了全面了解健壮的SSL，我们进行了一项调查研究。我们首先从机器学习的角度阐明了鲁棒SSL的形式化定义。然后，我们将健壮性威胁分为三类：i)分布损坏，即未标记数据分布与已标记数据不匹配；ii)特征损坏，即未标记示例的特征受到恶意攻击；iii)标签损坏，即未标记数据的标签分布不平衡。在这个统一的分类法下，我们对最近集中在这些问题上的工作进行了彻底的回顾和讨论。最后，我们提出了健壮SSL中可能有前景的方向，为未来的研究提供了见解。



## **38. Measuring the Contribution of Multiple Model Representations in Detecting Adversarial Instances**

测量多个模型表示在检测对抗性实例中的贡献 cs.LG

Correction: replaced "model-wise" with "unit-wise" in the first  sentence of Section 3.2

**SubmitDate**: 2022-02-12    [paper-pdf](http://arxiv.org/pdf/2111.07035v2)

**Authors**: Daniel Steinberg, Paul Munro

**Abstracts**: Deep learning models have been used for a wide variety of tasks. They are prevalent in computer vision, natural language processing, speech recognition, and other areas. While these models have worked well under many scenarios, it has been shown that they are vulnerable to adversarial attacks. This has led to a proliferation of research into ways that such attacks could be identified and/or defended against. Our goal is to explore the contribution that can be attributed to using multiple underlying models for the purpose of adversarial instance detection. Our paper describes two approaches that incorporate representations from multiple models for detecting adversarial examples. We devise controlled experiments for measuring the detection impact of incrementally utilizing additional models. For many of the scenarios we consider, the results show that performance increases with the number of underlying models used for extracting representations.

摘要: 深度学习模型已被广泛用于各种任务。它们广泛应用于计算机视觉、自然语言处理、语音识别等领域。虽然这些模型在许多情况下都工作得很好，但已经表明它们很容易受到对手的攻击。这导致了对如何识别和/或防御此类攻击的研究激增。我们的目标是探索可以归因于使用多个底层模型进行对抗性实例检测的贡献。我们的论文描述了两种方法，它们融合了来自多个模型的表示，用于检测对抗性示例。我们设计了对照实验来衡量增量利用额外模型的检测影响。对于我们考虑的许多场景，结果显示性能随着用于提取表示的底层模型数量的增加而提高。



## **39. Adversarial Attacks and Defense Methods for Power Quality Recognition**

电能质量识别中的对抗性攻击与防御方法 cs.CR

Technical report

**SubmitDate**: 2022-02-11    [paper-pdf](http://arxiv.org/pdf/2202.07421v1)

**Authors**: Jiwei Tian, Buhong Wang, Jing Li, Zhen Wang, Mete Ozay

**Abstracts**: Vulnerability of various machine learning methods to adversarial examples has been recently explored in the literature. Power systems which use these vulnerable methods face a huge threat against adversarial examples. To this end, we first propose a signal-specific method and a universal signal-agnostic method to attack power systems using generated adversarial examples. Black-box attacks based on transferable characteristics and the above two methods are also proposed and evaluated. We then adopt adversarial training to defend systems against adversarial attacks. Experimental analyses demonstrate that our signal-specific attack method provides less perturbation compared to the FGSM (Fast Gradient Sign Method), and our signal-agnostic attack method can generate perturbations fooling most natural signals with high probability. What's more, the attack method based on the universal signal-agnostic algorithm has a higher transfer rate of black-box attacks than the attack method based on the signal-specific algorithm. In addition, the results show that the proposed adversarial training improves robustness of power systems to adversarial examples.

摘要: 最近在文献中探讨了各种机器学习方法对对抗性示例的脆弱性。使用这些易受攻击的方法的电力系统在对抗对手的例子中面临着巨大的威胁。为此，我们首先提出了一种信号特定的方法和一种通用的信号不可知的方法来利用生成的对抗性实例来攻击电力系统。提出了基于可转移特征的黑盒攻击方法，并对这两种方法进行了评估。然后，我们采用对抗性训练来保护系统免受对抗性攻击。实验分析表明，与快速梯度符号方法(FGSM)相比，我们的信号特定攻击方法提供了更少的扰动，并且我们的信号不可知攻击方法可以高概率地产生欺骗大多数自然信号的扰动。此外，基于通用信号不可知算法的攻击方法比基于特定信号算法的攻击方法具有更高的黑盒攻击传递率。此外，结果表明，所提出的对抗性训练提高了电力系统对对抗性示例的鲁棒性。



## **40. Are socially-aware trajectory prediction models really socially-aware?**

具有社会性的轨迹预测模型真的具有社会性吗？ cs.CV

**SubmitDate**: 2022-02-11    [paper-pdf](http://arxiv.org/pdf/2108.10879v2)

**Authors**: Saeed Saadatnejad, Mohammadhossein Bahari, Pedram Khorsandi, Mohammad Saneian, Seyed-Mohsen Moosavi-Dezfooli, Alexandre Alahi

**Abstracts**: Our field has recently witnessed an arms race of neural network-based trajectory predictors. While these predictors are at the core of many applications such as autonomous navigation or pedestrian flow simulations, their adversarial robustness has not been carefully studied. In this paper, we introduce a socially-attended attack to assess the social understanding of prediction models in terms of collision avoidance. An attack is a small yet carefully-crafted perturbations to fail predictors. Technically, we define collision as a failure mode of the output, and propose hard- and soft-attention mechanisms to guide our attack. Thanks to our attack, we shed light on the limitations of the current models in terms of their social understanding. We demonstrate the strengths of our method on the recent trajectory prediction models. Finally, we show that our attack can be employed to increase the social understanding of state-of-the-art models. The code is available online: https://s-attack.github.io/

摘要: 我们的领域最近见证了一场基于神经网络的轨迹预测器的军备竞赛。虽然这些预报器是许多应用的核心，如自主导航或行人流量模拟，但它们的对抗性健壮性还没有得到仔细的研究。在这篇文章中，我们引入了一个社交参与的攻击来评估社会对预测模型在避免碰撞方面的理解。攻击是一个小的，但精心设计的扰动失败的预报器。在技术上，我们将碰撞定义为输出的一种失效模式，并提出了硬注意和软注意机制来指导我们的攻击。多亏了我们的攻击，我们揭示了当前模型在社会理解方面的局限性。我们在最近的轨迹预测模型上展示了我们的方法的优势。最后，我们展示了我们的攻击可以用来增加社会对最先进模型的理解。代码可在网上获得：https://s-attack.github.io/



## **41. Using Random Perturbations to Mitigate Adversarial Attacks on Sentiment Analysis Models**

利用随机扰动缓解情感分析模型上的敌意攻击 cs.CL

To be published in the proceedings for the 18th International  Conference on Natural Language Processing (ICON 2021)

**SubmitDate**: 2022-02-11    [paper-pdf](http://arxiv.org/pdf/2202.05758v1)

**Authors**: Abigail Swenor, Jugal Kalita

**Abstracts**: Attacks on deep learning models are often difficult to identify and therefore are difficult to protect against. This problem is exacerbated by the use of public datasets that typically are not manually inspected before use. In this paper, we offer a solution to this vulnerability by using, during testing, random perturbations such as spelling correction if necessary, substitution by random synonym, or simply dropping the word. These perturbations are applied to random words in random sentences to defend NLP models against adversarial attacks. Our Random Perturbations Defense and Increased Randomness Defense methods are successful in returning attacked models to similar accuracy of models before attacks. The original accuracy of the model used in this work is 80% for sentiment classification. After undergoing attacks, the accuracy drops to accuracy between 0% and 44%. After applying our defense methods, the accuracy of the model is returned to the original accuracy within statistical significance.

摘要: 针对深度学习模型的攻击通常很难识别，因此很难防范。使用通常不会在使用前手动检查的公共数据集加剧了此问题。在本文中，我们通过在测试期间使用随机扰动(如必要时进行拼写更正、替换为随机同义词或简单地删除单词)来解决此漏洞。这些扰动被应用于随机句子中的随机词，以保护NLP模型免受对手攻击。我们的随机扰动防御和增加的随机性防御方法成功地将被攻击的模型恢复到攻击前模型的类似精度。本文使用的模型对情感分类的原始正确率为80%。在遭受攻击后，准确率下降到0%到44%之间。应用我们的防御方法后，模型的精度在统计意义上恢复到原来的精度。



## **42. On the Detection of Adaptive Adversarial Attacks in Speaker Verification Systems**

说话人确认系统中自适应攻击检测的研究 cs.CR

**SubmitDate**: 2022-02-11    [paper-pdf](http://arxiv.org/pdf/2202.05725v1)

**Authors**: Zesheng Chen

**Abstracts**: Speaker verification systems have been widely used in smart phones and Internet of things devices to identify a legitimate user. In recent work, it has been shown that adversarial attacks, such as FAKEBOB, can work effectively against speaker verification systems. The goal of this paper is to design a detector that can distinguish an original audio from an audio contaminated by adversarial attacks. Specifically, our designed detector, called MEH-FEST, calculates the minimum energy in high frequencies from the short-time Fourier transform of an audio and uses it as a detection metric. Through both analysis and experiments, we show that our proposed detector is easy to implement, fast to process an input audio, and effective in determining whether an audio is corrupted by FAKEBOB attacks. The experimental results indicate that the detector is extremely effective: with near zero false positive and false negative rates for detecting FAKEBOB attacks in Gaussian mixture model (GMM) and i-vector speaker verification systems. Moreover, adaptive adversarial attacks against our proposed detector and their countermeasures are discussed and studied, showing the game between attackers and defenders.

摘要: 说话人验证系统已广泛应用于智能手机和物联网设备中，用于识别合法用户。最近的研究表明，FAKEBOB等对抗性攻击可以有效地对抗说话人确认系统。本文的目标是设计一种能够区分原始音频和被敌意攻击污染的音频的检测器。具体地说，我们设计的检测器，称为MEH-FEST，从音频的短时傅立叶变换计算高频最小能量，并将其用作检测度量。通过分析和实验表明，我们提出的检测器实现简单，处理输入音频的速度快，能有效地判断音频是否被FAKEBOB攻击破坏。实验结果表明，该检测器对混合高斯模型(GMM)和I向量说话人确认系统中FAKEBOB攻击的检测非常有效，误报率和误报率都接近于零。此外，还讨论和研究了针对我们提出的检测器的自适应对抗性攻击及其对策，展示了攻击者和防御者之间的博弈。



## **43. Towards Adversarially Robust Deepfake Detection: An Ensemble Approach**

面向对抗性强健的深伪检测：一种集成方法 cs.LG

**SubmitDate**: 2022-02-11    [paper-pdf](http://arxiv.org/pdf/2202.05687v1)

**Authors**: Ashish Hooda, Neal Mangaokar, Ryan Feng, Kassem Fawaz, Somesh Jha, Atul Prakash

**Abstracts**: Detecting deepfakes is an important problem, but recent work has shown that DNN-based deepfake detectors are brittle against adversarial deepfakes, in which an adversary adds imperceptible perturbations to a deepfake to evade detection. In this work, we show that a modification to the detection strategy in which we replace a single classifier with a carefully chosen ensemble, in which input transformations for each model in the ensemble induces pairwise orthogonal gradients, can significantly improve robustness beyond the de facto solution of adversarial training. We present theoretical results to show that such orthogonal gradients can help thwart a first-order adversary by reducing the dimensionality of the input subspace in which adversarial deepfakes lie. We validate the results empirically by instantiating and evaluating a randomized version of such "orthogonal" ensembles for adversarial deepfake detection and find that these randomized ensembles exhibit significantly higher robustness as deepfake detectors compared to state-of-the-art deepfake detectors against adversarial deepfakes, even those created using strong PGD-500 attacks.

摘要: 深度伪码的检测是一个重要的问题，但最近的研究表明，基于DNN的深度伪码检测器对敌意的深度伪码是脆弱的，在这种情况下，敌手通过向深度伪码添加不可察觉的扰动来逃避检测。在这项工作中，我们证明了对检测策略的修改，即用精心选择的集成来取代单个分类器，其中集成中每个模型的输入变换都会诱导成对的正交梯度，可以显著提高鲁棒性，而不是对抗性训练的事实解决方案。我们给出的理论结果表明，这种正交梯度可以通过降低敌意深伪所在的输入子空间的维数来帮助挫败一阶敌方。我们通过实例化和评估这种用于对抗性深度伪检测的“正交”集成的随机化版本来实证验证结果，并发现这些随机化集成在对抗对抗性深伪(即使是使用强PGD-500攻击创建的深伪)时，与最新的深伪检测器相比，表现出明显更高的稳健性。



## **44. FAAG: Fast Adversarial Audio Generation through Interactive Attack Optimisation**

FAAG：通过交互式攻击优化快速生成敌方音频 cs.SD

**SubmitDate**: 2022-02-11    [paper-pdf](http://arxiv.org/pdf/2202.05416v1)

**Authors**: Yuantian Miao, Chao Chen, Lei Pan, Jun Zhang, Yang Xiang

**Abstracts**: Automatic Speech Recognition services (ASRs) inherit deep neural networks' vulnerabilities like crafted adversarial examples. Existing methods often suffer from low efficiency because the target phases are added to the entire audio sample, resulting in high demand for computational resources. This paper proposes a novel scheme named FAAG as an iterative optimization-based method to generate targeted adversarial examples quickly. By injecting the noise over the beginning part of the audio, FAAG generates adversarial audio in high quality with a high success rate timely. Specifically, we use audio's logits output to map each character in the transcription to an approximate position of the audio's frame. Thus, an adversarial example can be generated by FAAG in approximately two minutes using CPUs only and around ten seconds with one GPU while maintaining an average success rate over 85%. Specifically, the FAAG method can speed up around 60% compared with the baseline method during the adversarial example generation process. Furthermore, we found that appending benign audio to any suspicious examples can effectively defend against the targeted adversarial attack. We hope that this work paves the way for inventing new adversarial attacks against speech recognition with computational constraints.

摘要: 自动语音识别服务(ASR)继承了深层神经网络的弱点，就像精心制作的敌意例子。现有的方法通常效率较低，因为目标相位被添加到整个音频样本，导致对计算资源的高需求。提出了一种新的基于迭代优化的FAAG方案，用于快速生成目标对抗性实例。通过在音频的开始部分注入噪声，FAAG及时生成高质量和高成功率的敌意音频。具体地说，我们使用音频的logits输出将转录中的每个字符映射到音频帧的大致位置。因此，FAAG仅使用CPU就可以在大约2分钟内生成对抗性示例，使用一个GPU可以在大约10秒内生成对抗性示例，同时保持85%以上的平均成功率。具体地说，在对抗性实例生成过程中，与基线方法相比，FAAG方法可以加快60%左右的速度。此外，我们还发现，在任何可疑的示例中添加良性音频可以有效地防御目标攻击。我们希望这项工作为发明新的针对计算受限的语音识别的对抗性攻击铺平道路。



## **45. SoK: Certified Robustness for Deep Neural Networks**

SOK：深度神经网络的认证鲁棒性 cs.LG

14 pages for the main text; recent advances (till Feb 2022) included

**SubmitDate**: 2022-02-10    [paper-pdf](http://arxiv.org/pdf/2009.04131v6)

**Authors**: Linyi Li, Tao Xie, Bo Li

**Abstracts**: Great advances in deep neural networks (DNNs) have led to state-of-the-art performance on a wide range of tasks. However, recent studies have shown that DNNs are vulnerable to adversarial attacks, which have brought great concerns when deploying these models to safety-critical applications such as autonomous driving. Different defense approaches have been proposed against adversarial attacks, including: a) empirical defenses, which usually can be adaptively attacked again without providing robustness certification; and b) certifiably robust approaches which consist of robustness verification providing the lower bound of robust accuracy against any attacks under certain conditions and corresponding robust training approaches. In this paper, we systematize the certifiably robust approaches and related practical and theoretical implications and findings. We also provide the first comprehensive benchmark on existing robustness verification and training approaches on different datasets. In particular, we 1) provide a taxonomy for the robustness verification and training approaches, as well as summarize the methodologies for representative algorithms, 2) reveal the characteristics, strengths, limitations, and fundamental connections among these approaches, 3) discuss current research progresses, theoretical barriers, main challenges, and future directions for certifiably robust approaches for DNNs, and 4) provide an open-sourced unified platform to evaluate over 20 representative certifiably robust approaches for a wide range of DNNs.

摘要: 深度神经网络(DNNs)的巨大进步导致了在广泛任务上的最先进的性能。然而，最近的研究表明，DNN很容易受到敌意攻击，这在将这些模型部署到自动驾驶等安全关键型应用时带来了极大的担忧。针对敌意攻击已经提出了不同的防御方法，包括：a)经验防御，通常无需提供健壮性证明即可自适应地再次攻击；b)可证明健壮性方法，包括在一定条件下提供对任何攻击的鲁棒精度下界的健壮性验证和相应的健壮性训练方法。在这篇文章中，我们系统化的证明稳健的方法和相关的实际和理论意义和发现。我们还提供了关于不同数据集上现有健壮性验证和训练方法的第一个全面基准。特别地，我们1)提供了健壮性验证和训练方法的分类，并总结了典型算法的方法论；2)揭示了这些方法的特点、优点、局限性和基本联系；3)讨论了当前DNNs的研究进展、理论障碍、主要挑战和未来的发展方向；4)提供了一个开源的统一平台来评估20多种具有代表性的DNNs的可证健壮性方法。



## **46. Towards Assessing and Characterizing the Semantic Robustness of Face Recognition**

面向人脸识别的语义健壮性评估与表征 cs.CV

26 pages, 18 figures

**SubmitDate**: 2022-02-10    [paper-pdf](http://arxiv.org/pdf/2202.04978v1)

**Authors**: Juan C. Pérez, Motasem Alfarra, Ali Thabet, Pablo Arbeláez, Bernard Ghanem

**Abstracts**: Deep Neural Networks (DNNs) lack robustness against imperceptible perturbations to their input. Face Recognition Models (FRMs) based on DNNs inherit this vulnerability. We propose a methodology for assessing and characterizing the robustness of FRMs against semantic perturbations to their input. Our methodology causes FRMs to malfunction by designing adversarial attacks that search for identity-preserving modifications to faces. In particular, given a face, our attacks find identity-preserving variants of the face such that an FRM fails to recognize the images belonging to the same identity. We model these identity-preserving semantic modifications via direction- and magnitude-constrained perturbations in the latent space of StyleGAN. We further propose to characterize the semantic robustness of an FRM by statistically describing the perturbations that induce the FRM to malfunction. Finally, we combine our methodology with a certification technique, thus providing (i) theoretical guarantees on the performance of an FRM, and (ii) a formal description of how an FRM may model the notion of face identity.

摘要: 深度神经网络(DNNs)对其输入的不可察觉的扰动缺乏鲁棒性。基于DNN的人脸识别模型(FRM)继承了此漏洞。我们提出了一种方法来评估和表征FRM对其输入的语义扰动的鲁棒性。我们的方法论通过设计对抗性攻击来搜索对人脸的身份保留修改，从而导致FRMS发生故障。特别地，在给定一张人脸的情况下，我们的攻击会找到该人脸的保持身份的变体，使得FRM无法识别属于同一身份的图像。我们在StyleGan的潜在空间中通过方向和幅度约束的扰动来模拟这些保持身份的语义修改。我们进一步提出通过统计描述导致FRM故障的扰动来表征FRM的语义健壮性。最后，我们将我们的方法与认证技术相结合，从而提供(I)FRM性能的理论保证，以及(Ii)FRM如何建模面部身份概念的正式描述。



## **47. Beyond ImageNet Attack: Towards Crafting Adversarial Examples for Black-box Domains**

超越ImageNet攻击：为黑盒领域精心制作敌意示例 cs.CV

Accepted by ICLR 2022

**SubmitDate**: 2022-02-10    [paper-pdf](http://arxiv.org/pdf/2201.11528v3)

**Authors**: Qilong Zhang, Xiaodan Li, Yuefeng Chen, Jingkuan Song, Lianli Gao, Yuan He, Hui Xue

**Abstracts**: Adversarial examples have posed a severe threat to deep neural networks due to their transferable nature. Currently, various works have paid great efforts to enhance the cross-model transferability, which mostly assume the substitute model is trained in the same domain as the target model. However, in reality, the relevant information of the deployed model is unlikely to leak. Hence, it is vital to build a more practical black-box threat model to overcome this limitation and evaluate the vulnerability of deployed models. In this paper, with only the knowledge of the ImageNet domain, we propose a Beyond ImageNet Attack (BIA) to investigate the transferability towards black-box domains (unknown classification tasks). Specifically, we leverage a generative model to learn the adversarial function for disrupting low-level features of input images. Based on this framework, we further propose two variants to narrow the gap between the source and target domains from the data and model perspectives, respectively. Extensive experiments on coarse-grained and fine-grained domains demonstrate the effectiveness of our proposed methods. Notably, our methods outperform state-of-the-art approaches by up to 7.71\% (towards coarse-grained domains) and 25.91\% (towards fine-grained domains) on average. Our code is available at \url{https://github.com/qilong-zhang/Beyond-ImageNet-Attack}.

摘要: 对抗性例子由于其可转移性，对深度神经网络构成了严重的威胁。目前，各种研究都在努力提高模型间的可移植性，大多假设替身模型与目标模型在同一领域进行训练。然而，在现实中，部署的模型的相关信息不太可能泄露。因此，构建一个更实用的黑盒威胁模型来克服这一限制并评估已部署模型的脆弱性是至关重要的。本文在仅知道ImageNet域的情况下，提出了一种超越ImageNet攻击(BIA)来研究向黑盒域(未知分类任务)的可传递性。具体地说，我们利用生成模型来学习破坏输入图像的低层特征的对抗性函数。基于这一框架，我们进一步提出了两种变体，分别从数据和模型的角度来缩小源域和目标域之间的差距。在粗粒域和细粒域上的大量实验证明了我们提出的方法的有效性。值得注意的是，我们的方法平均比最先进的方法高出7.71%(对于粗粒度领域)和25.91%(对于细粒度领域)。我们的代码可在\url{https://github.com/qilong-zhang/Beyond-ImageNet-Attack}.获得



## **48. Adversarial Attack and Defense of YOLO Detectors in Autonomous Driving Scenarios**

自动驾驶场景中YOLO检测器的对抗性攻击与防御 cs.CV

7 pages, 3 figures

**SubmitDate**: 2022-02-10    [paper-pdf](http://arxiv.org/pdf/2202.04781v1)

**Authors**: Jung Im Choi, Qing Tian

**Abstracts**: Visual detection is a key task in autonomous driving, and it serves as one foundation for self-driving planning and control. Deep neural networks have achieved promising results in various computer vision tasks, but they are known to be vulnerable to adversarial attacks. A comprehensive understanding of deep visual detectors' vulnerability is required before people can improve their robustness. However, only a few adversarial attack/defense works have focused on object detection, and most of them employed only classification and/or localization losses, ignoring the objectness aspect. In this paper, we identify a serious objectness-related adversarial vulnerability in YOLO detectors and present an effective attack strategy aiming the objectness aspect of visual detection in autonomous vehicles. Furthermore, to address such vulnerability, we propose a new objectness-aware adversarial training approach for visual detection. Experiments show that the proposed attack targeting the objectness aspect is 45.17% and 43.50% more effective than those generated from classification and/or localization losses on the KITTI and COCO_traffic datasets, respectively. Also, the proposed adversarial defense approach can improve the detectors' robustness against objectness-oriented attacks by up to 21% and 12% mAP on KITTI and COCO_traffic, respectively.

摘要: 视觉检测是自动驾驶中的一项关键任务，是自动驾驶规划和控制的基础之一。深度神经网络在各种计算机视觉任务中取得了令人满意的结果，但众所周知，它们很容易受到对手的攻击。人们需要全面了解深度视觉检测器的脆弱性，才能提高其健壮性。然而，只有少数对抗性攻防研究集中在目标检测上，而且大多只采用分类和/或定位损失，而忽略了客观性方面。本文针对自主车辆视觉检测的客观性方面，识别出YOLO检测器中存在的一个与客观性相关的严重攻击漏洞，并提出了一种有效的攻击策略。此外，为了解决这种脆弱性，我们提出了一种新的基于客观性感知的对抗性视觉检测训练方法。实验表明，针对客观性方面的攻击比基于KITTI和COCO_TRAFFORM数据集的分类和/或定位丢失攻击分别提高了45.17%和43.50%。此外，本文提出的对抗性防御方法可以使检测器对面向对象攻击的鲁棒性分别提高21%和12%MAP在KITTI和COCO_TRAFFORMS上。



## **49. IoTMonitor: A Hidden Markov Model-based Security System to Identify Crucial Attack Nodes in Trigger-action IoT Platforms**

IoTMonitor：基于隐马尔可夫模型的触发物联网平台关键攻击节点识别安全系统 cs.CR

This paper appears in the 2022 IEEE Wireless Communications and  Networking Conference (WCNC 2022). Personal use of this material is  permitted. Permission from IEEE must be obtained for all other uses

**SubmitDate**: 2022-02-09    [paper-pdf](http://arxiv.org/pdf/2202.04620v1)

**Authors**: Md Morshed Alam, Md Sajidul Islam Sajid, Weichao Wang, Jinpeng Wei

**Abstracts**: With the emergence and fast development of trigger-action platforms in IoT settings, security vulnerabilities caused by the interactions among IoT devices become more prevalent. The event occurrence at one device triggers an action in another device, which may eventually contribute to the creation of a chain of events in a network. Adversaries exploit the chain effect to compromise IoT devices and trigger actions of interest remotely just by injecting malicious events into the chain. To address security vulnerabilities caused by trigger-action scenarios, existing research efforts focus on the validation of the security properties of devices or verification of the occurrence of certain events based on their physical fingerprints on a device. We propose IoTMonitor, a security analysis system that discerns the underlying chain of event occurrences with the highest probability by observing a chain of physical evidence collected by sensors. We use the Baum-Welch algorithm to estimate transition and emission probabilities and the Viterbi algorithm to discern the event sequence. We can then identify the crucial nodes in the trigger-action sequence whose compromise allows attackers to reach their final goals. The experiment results of our designed system upon the PEEVES datasets show that we can rebuild the event occurrence sequence with high accuracy from the observations and identify the crucial nodes on the attack paths.

摘要: 随着物联网环境下触发式平台的出现和快速发展，物联网设备之间的交互导致的安全漏洞变得更加普遍。一台设备上发生的事件会触发另一台设备上的操作，这最终可能会导致在网络中创建一系列事件。攻击者只需将恶意事件注入链中，即可利用连锁反应危害物联网设备并远程触发感兴趣的操作。为了解决触发动作场景引起的安全漏洞，现有的研究工作集中在验证设备的安全属性或基于设备上的物理指纹来验证特定事件的发生。我们提出了IoTMonitor，这是一个安全分析系统，它通过观察传感器收集的物理证据链，以最高的概率识别事件发生的潜在链。我们使用Baum-Welch算法来估计转移概率和发射概率，使用Viterbi算法来识别事件序列。然后，我们可以确定触发-动作序列中的关键节点，这些节点的妥协使攻击者能够达到他们的最终目标。我们设计的系统在PEVES数据集上的实验结果表明，我们可以从观测数据中高精度地重建事件发生序列，并识别攻击路径上的关键节点。



## **50. False Memory Formation in Continual Learners Through Imperceptible Backdoor Trigger**

通过潜伏的后门触发器形成持续学习者的错误记忆 cs.LG

**SubmitDate**: 2022-02-09    [paper-pdf](http://arxiv.org/pdf/2202.04479v1)

**Authors**: Muhammad Umer, Robi Polikar

**Abstracts**: In this brief, we show that sequentially learning new information presented to a continual (incremental) learning model introduces new security risks: an intelligent adversary can introduce small amount of misinformation to the model during training to cause deliberate forgetting of a specific task or class at test time, thus creating "false memory" about that task. We demonstrate such an adversary's ability to assume control of the model by injecting "backdoor" attack samples to commonly used generative replay and regularization based continual learning approaches using continual learning benchmark variants of MNIST, as well as the more challenging SVHN and CIFAR 10 datasets. Perhaps most damaging, we show this vulnerability to be very acute and exceptionally effective: the backdoor pattern in our attack model can be imperceptible to human eye, can be provided at any point in time, can be added into the training data of even a single possibly unrelated task and can be achieved with as few as just 1\% of total training dataset of a single task.

摘要: 在这篇简短的文章中，我们展示了顺序学习提供给连续(增量)学习模型的新信息会带来新的安全风险：智能对手可能在训练期间向模型引入少量的错误信息，导致在测试时故意忘记特定任务或类，从而产生关于该任务的“错误记忆”。我们使用MNIST的持续学习基准变体，以及更具挑战性的SVHN和CIFAR10数据集，通过向常用的基于生成性回放和正则化的持续学习方法注入“后门”攻击样本，展示了这样的对手控制模型的能力。也许最具破坏性的是，我们发现这个漏洞是非常尖锐和特别有效的：我们的攻击模型中的后门模式可以是人眼看不见的，可以在任何时间点提供，可以添加到甚至是单个可能不相关的任务的训练数据中，并且可以仅使用单个任务的全部训练数据集的1\%就可以实现。



