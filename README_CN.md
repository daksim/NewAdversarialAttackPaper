# Latest Adversarial Attack Papers
**update at 2024-01-27 11:18:49**

翻译来自 https://cloud.tencent.com/document/product/551/15619

## **1. Defending Against Physical Adversarial Patch Attacks on Infrared Human Detection**

红外人体检测中的物理对抗性补丁攻击防御 cs.CV

Lukas Strack and Futa Waseda contributed equally. 6 pages,  Under-review

**SubmitDate**: 2024-01-25    [abs](http://arxiv.org/abs/2309.15519v2) [paper-pdf](http://arxiv.org/pdf/2309.15519v2)

**Authors**: Lukas Strack, Futa Waseda, Huy H. Nguyen, Yinqiang Zheng, Isao Echizen

**Abstract**: Infrared detection is an emerging technique for safety-critical tasks owing to its remarkable anti-interference capability. However, recent studies have revealed that it is vulnerable to physically-realizable adversarial patches, posing risks in its real-world applications. To address this problem, we are the first to investigate defense strategies against adversarial patch attacks on infrared detection, especially human detection. We have devised a straightforward defense strategy, patch-based occlusion-aware detection (POD), which efficiently augments training samples with random patches and subsequently detects them. POD not only robustly detects people but also identifies adversarial patch locations. Surprisingly, while being extremely computationally efficient, POD easily generalizes to state-of-the-art adversarial patch attacks that are unseen during training. Furthermore, POD improves detection precision even in a clean (i.e., no-attack) situation due to the data augmentation effect. Evaluation demonstrated that POD is robust to adversarial patches of various shapes and sizes. The effectiveness of our baseline approach is shown to be a viable defense mechanism for real-world infrared human detection systems, paving the way for exploring future research directions.

摘要: 红外探测是一种新兴的安全关键任务检测技术，具有显著的抗干扰性。然而，最近的研究表明，它很容易受到物理上可实现的对抗性补丁的攻击，这给它在现实世界的应用带来了风险。针对这一问题，我们首次研究了针对红外探测，尤其是人体探测的对抗性补丁攻击的防御策略。我们设计了一种简单的防御策略，基于补丁的遮挡感知检测(POD)，它有效地利用随机补丁来增加训练样本并随后检测它们。Pod不仅可以稳健地检测人员，还可以识别敌方的补丁位置。令人惊讶的是，虽然POD在计算上非常高效，但它很容易概括为最先进的对抗性补丁攻击，这些攻击在训练中是看不到的。此外，由于数据增强效应，即使在干净(即，无攻击)的情况下，POD也提高了检测精度。评估表明，POD对不同形状和大小的敌方斑块具有很强的健壮性。我们的基线方法的有效性被证明是一种可行的防御机制，用于真实世界的红外人体探测系统，为探索未来的研究方向铺平了道路。



## **2. TrojFST: Embedding Trojans in Few-shot Prompt Tuning**

TrojFST：在少量提示调优中嵌入特洛伊木马 cs.LG

9 pages

**SubmitDate**: 2024-01-25    [abs](http://arxiv.org/abs/2312.10467v2) [paper-pdf](http://arxiv.org/pdf/2312.10467v2)

**Authors**: Mengxin Zheng, Jiaqi Xue, Xun Chen, YanShan Wang, Qian Lou, Lei Jiang

**Abstract**: Prompt-tuning has emerged as a highly effective approach for adapting a pre-trained language model (PLM) to handle new natural language processing tasks with limited input samples. However, the success of prompt-tuning has led to adversaries attempting backdoor attacks against this technique. Previous prompt-based backdoor attacks faced challenges when implemented through few-shot prompt-tuning, requiring either full-model fine-tuning or a large training dataset. We observe the difficulty in constructing a prompt-based backdoor using few-shot prompt-tuning, which involves freezing the PLM and tuning a soft prompt with a restricted set of input samples. This approach introduces an imbalanced poisoned dataset, making it susceptible to overfitting and lacking attention awareness. To address these challenges, we introduce TrojFST for backdoor attacks within the framework of few-shot prompt-tuning. TrojFST comprises three modules: balanced poison learning, selective token poisoning, and trojan-trigger attention. In comparison to previous prompt-based backdoor attacks, TrojFST demonstrates significant improvements, enhancing ASR $> 9\%$ and CDA by $> 4\%$ across various PLMs and a diverse set of downstream tasks.

摘要: 即时调优已经成为一种高效的方法，可以使预先训练的语言模型(PLM)在输入样本有限的情况下处理新的自然语言处理任务。然而，快速调整的成功导致了对手试图对此技术进行后门攻击。以前的基于提示的后门攻击在通过少量提示调整实施时面临挑战，需要全模型微调或大型训练数据集。我们注意到使用少量提示调优来构建基于提示的后门的困难，这涉及冻结PLM并使用受限的输入样本集来调优软提示。这种方法引入了一个不平衡的有毒数据集，使其容易过度拟合和缺乏注意力意识。为了应对这些挑战，我们引入了TrojFST，用于在少发提示调优的框架内进行后门攻击。TrojFST包括三个模块：均衡毒物学习、选择性令牌毒化和木马触发注意。与以前基于提示的后门攻击相比，TrojFST表现出显著的改进，在不同的PLM和不同的下游任务集上将ASR$>9\$和CDA提高了$>4\$。



## **3. The Surprising Harmfulness of Benign Overfitting for Adversarial Robustness**

良性过度拟合对对抗健壮性的危害令人惊讶 cs.LG

**SubmitDate**: 2024-01-25    [abs](http://arxiv.org/abs/2401.12236v2) [paper-pdf](http://arxiv.org/pdf/2401.12236v2)

**Authors**: Yifan Hao, Tong Zhang

**Abstract**: Recent empirical and theoretical studies have established the generalization capabilities of large machine learning models that are trained to (approximately or exactly) fit noisy data. In this work, we prove a surprising result that even if the ground truth itself is robust to adversarial examples, and the benignly overfitted model is benign in terms of the ``standard'' out-of-sample risk objective, this benign overfitting process can be harmful when out-of-sample data are subject to adversarial manipulation. More specifically, our main results contain two parts: (i) the min-norm estimator in overparameterized linear model always leads to adversarial vulnerability in the ``benign overfitting'' setting; (ii) we verify an asymptotic trade-off result between the standard risk and the ``adversarial'' risk of every ridge regression estimator, implying that under suitable conditions these two items cannot both be small at the same time by any single choice of the ridge regularization parameter. Furthermore, under the lazy training regime, we demonstrate parallel results on two-layer neural tangent kernel (NTK) model, which align with empirical observations in deep neural networks. Our finding provides theoretical insights into the puzzling phenomenon observed in practice, where the true target function (e.g., human) is robust against adverasrial attack, while beginly overfitted neural networks lead to models that are not robust.

摘要: 最近的经验和理论研究已经建立了大型机器学习模型的泛化能力，这些模型经过训练以(近似或精确地)适应噪声数据。在这项工作中，我们证明了一个令人惊讶的结果，即使基本事实本身对对抗性例子是健壮的，并且友好的过拟合模型在“标准”的样本外风险目标方面是良性的，当样本外数据受到对抗性操纵时，这种良性的过拟合过程可能是有害的。更具体地说，我们的主要结果包括两部分：(I)过参数线性模型中的最小范数估计在“良性过拟合”设置下总是导致对抗脆弱性；(Ii)我们验证了每个岭回归估计的标准风险和“对抗”风险之间的渐近权衡结果，这意味着在适当的条件下，通过选择岭正则化参数，这两个项不可能同时小。此外，在懒惰训练机制下，我们展示了在两层神经切核(NTK)模型上的并行结果，该结果与深度神经网络中的经验观测相一致。我们的发现为实践中观察到的令人费解的现象提供了理论见解，其中真实的目标函数(例如，人类)对逆袭是健壮的，而一开始过度拟合的神经网络导致模型不健壮。



## **4. Friendly Attacks to Improve Channel Coding Reliability**

提高信道编码可靠性的恶意攻击 cs.IT

**SubmitDate**: 2024-01-25    [abs](http://arxiv.org/abs/2401.14184v1) [paper-pdf](http://arxiv.org/pdf/2401.14184v1)

**Authors**: Anastasiia Kurmukova, Deniz Gunduz

**Abstract**: This paper introduces a novel approach called "friendly attack" aimed at enhancing the performance of error correction channel codes. Inspired by the concept of adversarial attacks, our method leverages the idea of introducing slight perturbations to the neural network input, resulting in a substantial impact on the network's performance. By introducing small perturbations to fixed-point modulated codewords before transmission, we effectively improve the decoder's performance without violating the input power constraint. The perturbation design is accomplished by a modified iterative fast gradient method. This study investigates various decoder architectures suitable for computing gradients to obtain the desired perturbations. Specifically, we consider belief propagation (BP) for LDPC codes; the error correcting code transformer, BP and neural BP (NBP) for polar codes, and neural BCJR for convolutional codes. We demonstrate that the proposed friendly attack method can improve the reliability across different channels, modulations, codes, and decoders. This method allows us to increase the reliability of communication with a legacy receiver by simply modifying the transmitted codeword appropriately.

摘要: 为了提高纠错信道码的性能，提出了一种名为“友好攻击”的新方法。受对抗性攻击概念的启发，我们的方法利用了在神经网络输入中引入轻微扰动的思想，从而对网络的性能产生了实质性的影响。通过在传输前对定点调制码字引入小扰动，在不违反输入功率约束的情况下，有效地提高了译码性能。摄动设计采用改进的迭代快速梯度法。这项研究研究了各种适用于计算梯度以获得所需扰动的解码器架构。具体地，我们考虑了LDPC码的置信度传播(BP)，纠错码变换，极性码的BP和神经BP(NBP)，卷积码的神经BCJR。我们证明了所提出的友好攻击方法可以提高不同信道、不同调制、不同编码和不同译码的可靠性。该方法允许我们通过简单地适当地修改发送的码字来增加与传统接收器通信的可靠性。



## **5. EvadeDroid: A Practical Evasion Attack on Machine Learning for Black-box Android Malware Detection**

EvadeDroid：一种实用的机器学习黑盒Android恶意软件检测规避攻击 cs.LG

The paper was accepted by Elsevier Computers & Security on 20  December 2023

**SubmitDate**: 2024-01-25    [abs](http://arxiv.org/abs/2110.03301v4) [paper-pdf](http://arxiv.org/pdf/2110.03301v4)

**Authors**: Hamid Bostani, Veelasha Moonsamy

**Abstract**: Over the last decade, researchers have extensively explored the vulnerabilities of Android malware detectors to adversarial examples through the development of evasion attacks; however, the practicality of these attacks in real-world scenarios remains arguable. The majority of studies have assumed attackers know the details of the target classifiers used for malware detection, while in reality, malicious actors have limited access to the target classifiers. This paper introduces EvadeDroid, a problem-space adversarial attack designed to effectively evade black-box Android malware detectors in real-world scenarios. EvadeDroid constructs a collection of problem-space transformations derived from benign donors that share opcode-level similarity with malware apps by leveraging an n-gram-based approach. These transformations are then used to morph malware instances into benign ones via an iterative and incremental manipulation strategy. The proposed manipulation technique is a query-efficient optimization algorithm that can find and inject optimal sequences of transformations into malware apps. Our empirical evaluations, carried out on 1K malware apps, demonstrate the effectiveness of our approach in generating real-world adversarial examples in both soft- and hard-label settings. Our findings reveal that EvadeDroid can effectively deceive diverse malware detectors that utilize different features with various feature types. Specifically, EvadeDroid achieves evasion rates of 80%-95% against DREBIN, Sec-SVM, ADE-MA, MaMaDroid, and Opcode-SVM with only 1-9 queries. Furthermore, we show that the proposed problem-space adversarial attack is able to preserve its stealthiness against five popular commercial antiviruses with an average of 79% evasion rate, thus demonstrating its feasibility in the real world.

摘要: 在过去的十年里，研究人员通过规避攻击的开发，广泛探索了Android恶意软件检测器对敌意例子的漏洞；然而，这些攻击在现实世界场景中的实用性仍然存在争议。大多数研究都假设攻击者知道用于恶意软件检测的目标分类器的详细信息，而实际上，恶意行为者对目标分类器的访问权限有限。本文介绍了EvadeDroid，这是一种问题空间的对抗性攻击，旨在有效地躲避现实场景中的黑盒Android恶意软件检测器。EvadeDroid构建了一组问题空间转换，这些转换来自良性捐赠者，通过利用基于n-gram的方法，这些捐赠者与恶意软件应用程序具有操作码级别的相似性。然后，这些转换被用于通过迭代和增量操作策略将恶意软件实例变形为良性实例。提出的操纵技术是一种查询高效的优化算法，可以找到最优的转换序列并将其注入恶意软件应用程序。我们在1K恶意软件应用程序上进行的经验评估表明，我们的方法在生成软标签和硬标签环境中的真实对抗性示例方面是有效的。我们的发现表明，EvadeDroid可以有效地欺骗使用不同功能和不同特征类型的不同恶意软件检测器。具体地说，EvadeDroid对Drebin、SEC-SVM、ADE-MA、MaMaDroid和Opcode-SVM的逃避率为80%-95%，只需1-9个查询。此外，我们还证明了所提出的问题空间对抗攻击能够保持对五种流行的商业反病毒的隐蔽性，平均逃脱率为79%，从而证明了其在现实世界中的可行性。



## **6. Username Squatting on Online Social Networks: A Study on X**

基于X的在线社交网络用户名蹲点行为研究 cs.CR

Accepted at the 19th ACM ASIA Conference on Computer and  Communications Security (ACM ASIACCS 2024)

**SubmitDate**: 2024-01-25    [abs](http://arxiv.org/abs/2401.09209v2) [paper-pdf](http://arxiv.org/pdf/2401.09209v2)

**Authors**: Anastasios Lepipas, Anastasia Borovykh, Soteris Demetriou

**Abstract**: Adversaries have been targeting unique identifiers to launch typo-squatting, mobile app squatting and even voice squatting attacks. Anecdotal evidence suggest that online social networks (OSNs) are also plagued with accounts that use similar usernames. This can be confusing to users but can also be exploited by adversaries. However, to date no study characterizes this problem on OSNs. In this work, we define the username squatting problem and design the first multi-faceted measurement study to characterize it on X. We develop a username generation tool (UsernameCrazy) to help us analyze hundreds of thousands of username variants derived from celebrity accounts. Our study reveals that thousands of squatted usernames have been suspended by X, while tens of thousands that still exist on the network are likely bots. Out of these, a large number share similar profile pictures and profile names to the original account signalling impersonation attempts. We found that squatted accounts are being mentioned by mistake in tweets hundreds of thousands of times and are even being prioritized in searches by the network's search recommendation algorithm exacerbating the negative impact squatted accounts can have in OSNs. We use our insights and take the first step to address this issue by designing a framework (SQUAD) that combines UsernameCrazy with a new classifier to efficiently detect suspicious squatted accounts. Our evaluation of SQUAD's prototype implementation shows that it can achieve 94% F1-score when trained on a small dataset.

摘要: 对手一直以唯一标识为目标，发动打字蹲守、手机应用蹲守，甚至语音蹲守攻击。坊间证据表明，在线社交网络(OSN)也充斥着使用相似用户名的账户。这可能会让用户感到困惑，但也可能被对手利用。然而，到目前为止，还没有研究描述OSN上的这个问题。在这项工作中，我们定义了用户名下蹲问题，并设计了第一个多方面的测量研究来刻画X上的用户名下蹲问题。我们开发了一个用户名生成工具(UsernameCrazy)来帮助我们分析来自名人账户的数十万个用户名变体。我们的研究显示，数以千计的蹲守用户名已经被X暂停，而网络上仍然存在的数万个用户名很可能是机器人。在这些中，有大量共享与原始帐户信令模拟尝试相似的配置文件图片和配置文件名称。我们发现，在推文中，蹲着的账户被错误地提到了数十万次，甚至在网络的搜索推荐算法的搜索中被优先考虑，加剧了蹲着的账户在OSN中可能产生的负面影响。我们利用我们的见解，通过设计一个框架(Team)来解决这个问题，该框架(Team)将UsernameCrazy与新的分类器相结合，以高效地检测可疑的蹲守帐户。我们对LONG原型实现的评估表明，当在小数据集上训练时，它可以达到94%的F1得分。



## **7. Sparse and Transferable Universal Singular Vectors Attack**

稀疏可转移的通用奇异向量攻击 cs.LG

**SubmitDate**: 2024-01-25    [abs](http://arxiv.org/abs/2401.14031v1) [paper-pdf](http://arxiv.org/pdf/2401.14031v1)

**Authors**: Kseniia Kuvshinova, Olga Tsymboi, Ivan Oseledets

**Abstract**: The research in the field of adversarial attacks and models' vulnerability is one of the fundamental directions in modern machine learning. Recent studies reveal the vulnerability phenomenon, and understanding the mechanisms behind this is essential for improving neural network characteristics and interpretability. In this paper, we propose a novel sparse universal white-box adversarial attack. Our approach is based on truncated power iteration providing sparsity to $(p,q)$-singular vectors of the hidden layers of Jacobian matrices. Using the ImageNet benchmark validation subset, we analyze the proposed method in various settings, achieving results comparable to dense baselines with more than a 50% fooling rate while damaging only 5% of pixels and utilizing 256 samples for perturbation fitting. We also show that our algorithm admits higher attack magnitude without affecting the human ability to solve the task. Furthermore, we investigate that the constructed perturbations are highly transferable among different models without significantly decreasing the fooling rate. Our findings demonstrate the vulnerability of state-of-the-art models to sparse attacks and highlight the importance of developing robust machine learning systems.

摘要: 对抗性攻击和模型脆弱性的研究是现代机器学习的基本方向之一。最近的研究揭示了这种脆弱性现象，了解其背后的机制对于改善神经网络的特性和可解释性是至关重要的。本文提出了一种新的稀疏通用白盒对抗攻击方法。我们的方法是基于截断幂迭代，为雅可比矩阵隐层的$(p，q)$-奇异向量提供稀疏性。使用ImageNet基准验证子集，我们在不同的环境下对该方法进行了分析，获得了与密集基线相当的结果，欺骗率超过50%，而只破坏了5%的像素，并使用了256个样本进行扰动拟合。我们还表明，我们的算法在不影响人类解决任务的能力的情况下，可以接受更高的攻击强度。此外，我们还考察了所构造的扰动在不同模型之间的高度可转移性，而不显著降低愚弄率。我们的发现证明了最先进的模型对稀疏攻击的脆弱性，并突出了开发健壮的机器学习系统的重要性。



## **8. Exploring Adversarial Threat Models in Cyber Physical Battery Systems**

探索网络物理电池系统中的敌意威胁模型 eess.SY

**SubmitDate**: 2024-01-24    [abs](http://arxiv.org/abs/2401.13801v1) [paper-pdf](http://arxiv.org/pdf/2401.13801v1)

**Authors**: Shanthan Kumar Padisala, Shashank Dhananjay Vyas, Satadru Dey

**Abstract**: Technological advancements like the Internet of Things (IoT) have facilitated data exchange across various platforms. This data exchange across various platforms has transformed the traditional battery system into a cyber physical system. Such connectivity makes modern cyber physical battery systems vulnerable to cyber threats where a cyber attacker can manipulate sensing and actuation signals to bring the battery system into an unsafe operating condition. Hence, it is essential to build resilience in modern cyber physical battery systems (CPBS) under cyber attacks. The first step of building such resilience is to analyze potential adversarial behavior, that is, how the adversaries can inject attacks into the battery systems. However, it has been found that in this under-explored area of battery cyber physical security, such an adversarial threat model has not been studied in a systematic manner. In this study, we address this gap and explore adversarial attack generation policies based on optimal control framework. The framework is developed by performing theoretical analysis, which is subsequently supported by evaluation with experimental data generated from a commercial battery cell.

摘要: 物联网(IoT)等技术进步促进了各种平台之间的数据交换。这种跨平台的数据交换已经将传统的电池系统转变为网络物理系统。这种连接使现代网络物理电池系统容易受到网络威胁，网络攻击者可以操纵传感和激励信号，使电池系统进入不安全的操作条件。因此，建立现代网络物理电池系统(CPB)在网络攻击下的弹性是至关重要的。建立这种韧性的第一步是分析潜在的敌对行为，即对手如何向电池系统注入攻击。然而，人们发现，在电池网络物理安全这个探索不足的领域，这样的对抗性威胁模型还没有得到系统的研究。在这项研究中，我们解决了这一差距，并探索了基于最优控制框架的对抗性攻击生成策略。该框架是通过进行理论分析来开发的，随后通过对商业电池产生的实验数据进行评估来支持该框架。



## **9. A Systematic Approach to Robustness Modelling for Deep Convolutional Neural Networks**

深卷积神经网络稳健性建模的系统方法 cs.LG

**SubmitDate**: 2024-01-24    [abs](http://arxiv.org/abs/2401.13751v1) [paper-pdf](http://arxiv.org/pdf/2401.13751v1)

**Authors**: Charles Meyers, Mohammad Reza Saleh Sedghpour, Tommy Löfstedt, Erik Elmroth

**Abstract**: Convolutional neural networks have shown to be widely applicable to a large number of fields when large amounts of labelled data are available. The recent trend has been to use models with increasingly larger sets of tunable parameters to increase model accuracy, reduce model loss, or create more adversarially robust models -- goals that are often at odds with one another. In particular, recent theoretical work raises questions about the ability for even larger models to generalize to data outside of the controlled train and test sets. As such, we examine the role of the number of hidden layers in the ResNet model, demonstrated on the MNIST, CIFAR10, CIFAR100 datasets. We test a variety of parameters including the size of the model, the floating point precision, and the noise level of both the training data and the model output. To encapsulate the model's predictive power and computational cost, we provide a method that uses induced failures to model the probability of failure as a function of time and relate that to a novel metric that allows us to quickly determine whether or not the cost of training a model outweighs the cost of attacking it. Using this approach, we are able to approximate the expected failure rate using a small number of specially crafted samples rather than increasingly larger benchmark datasets. We demonstrate the efficacy of this technique on both the MNIST and CIFAR10 datasets using 8-, 16-, 32-, and 64-bit floating-point numbers, various data pre-processing techniques, and several attacks on five configurations of the ResNet model. Then, using empirical measurements, we examine the various trade-offs between cost, robustness, latency, and reliability to find that larger models do not significantly aid in adversarial robustness despite costing significantly more to train.

摘要: 卷积神经网络已被证明可广泛应用于大量的领域时，大量的标记数据是可用的。最近的趋势是使用具有越来越大的可调参数集的模型来提高模型准确性，减少模型损失，或创建更具对抗性的鲁棒模型-这些目标通常相互矛盾。特别是，最近的理论工作提出了关于更大的模型推广到受控训练集和测试集之外的数据的能力的问题。因此，我们研究了ResNet模型中隐藏层数量的作用，并在MNIST，CIFAR 10，CIFAR 100数据集上进行了演示。我们测试了各种参数，包括模型的大小，浮点精度以及训练数据和模型输出的噪声水平。为了封装模型的预测能力和计算成本，我们提供了一种方法，该方法使用诱导故障来将故障概率建模为时间的函数，并将其与一种新的度量相关联，该度量允许我们快速确定训练模型的成本是否超过攻击它的成本。使用这种方法，我们能够使用少量特制的样本而不是越来越大的基准数据集来近似预期的失败率。我们使用8位、16位、32位和64位浮点数、各种数据预处理技术以及对ResNet模型的五种配置的几次攻击，在MNIST和CIFAR 10数据集上证明了这种技术的有效性。然后，使用经验测量，我们检查了成本，鲁棒性，延迟和可靠性之间的各种权衡，发现尽管训练成本显着增加，但较大的模型并不能显着提高对抗鲁棒性。



## **10. TrojanPuzzle: Covertly Poisoning Code-Suggestion Models**

特洛伊木马之谜：秘密中毒代码-建议模型 cs.CR

**SubmitDate**: 2024-01-24    [abs](http://arxiv.org/abs/2301.02344v2) [paper-pdf](http://arxiv.org/pdf/2301.02344v2)

**Authors**: Hojjat Aghakhani, Wei Dai, Andre Manoel, Xavier Fernandes, Anant Kharkar, Christopher Kruegel, Giovanni Vigna, David Evans, Ben Zorn, Robert Sim

**Abstract**: With tools like GitHub Copilot, automatic code suggestion is no longer a dream in software engineering. These tools, based on large language models, are typically trained on massive corpora of code mined from unvetted public sources. As a result, these models are susceptible to data poisoning attacks where an adversary manipulates the model's training by injecting malicious data. Poisoning attacks could be designed to influence the model's suggestions at run time for chosen contexts, such as inducing the model into suggesting insecure code payloads. To achieve this, prior attacks explicitly inject the insecure code payload into the training data, making the poison data detectable by static analysis tools that can remove such malicious data from the training set. In this work, we demonstrate two novel attacks, COVERT and TROJANPUZZLE, that can bypass static analysis by planting malicious poison data in out-of-context regions such as docstrings. Our most novel attack, TROJANPUZZLE, goes one step further in generating less suspicious poison data by never explicitly including certain (suspicious) parts of the payload in the poison data, while still inducing a model that suggests the entire payload when completing code (i.e., outside docstrings). This makes TROJANPUZZLE robust against signature-based dataset-cleansing methods that can filter out suspicious sequences from the training data. Our evaluation against models of two sizes demonstrates that both COVERT and TROJANPUZZLE have significant implications for practitioners when selecting code used to train or tune code-suggestion models.

摘要: 有了GitHub Copilot这样的工具，自动代码建议不再是软件工程中的梦想。这些工具基于大型语言模型，通常针对从未经审查的公共来源挖掘的大量代码语料库进行培训。因此，这些模型容易受到数据中毒攻击，即对手通过注入恶意数据来操纵模型的训练。毒化攻击可以被设计成影响模型在运行时对所选上下文的建议，例如诱导模型建议不安全的代码有效负载。为了实现这一点，先前的攻击明确地将不安全的代码有效负载注入到训练数据中，使得有毒数据可以被静态分析工具检测到，该静态分析工具可以从训练集中移除此类恶意数据。在这项工作中，我们展示了两种新的攻击，COMERT和TROJANPUZLE，它们可以通过在文档字符串等脱离上下文的区域植入恶意毒物数据来绕过静态分析。我们最新颖的攻击TROJANPUZLE在生成不那么可疑的有毒数据方面更进一步，它从未显式地将某些(可疑)有效负载部分包括在有毒数据中，同时仍诱导出一个模型，该模型在完成代码(即，文档字符串外部)时建议整个有效负载。这使得TROJANPUZLE对于基于签名的数据集清理方法具有健壮性，这些方法可以从训练数据中过滤出可疑序列。我们对两种规模的模型的评估表明，CONVERT和TROJANPUZLE对于实践者在选择用于训练或调整代码建议模型的代码时都有重要的影响。



## **11. Synthesizing Physical Backdoor Datasets: An Automated Framework Leveraging Deep Generative Models**

综合物理后门数据集：利用深度生成模型的自动化框架 cs.CR

**SubmitDate**: 2024-01-24    [abs](http://arxiv.org/abs/2312.03419v2) [paper-pdf](http://arxiv.org/pdf/2312.03419v2)

**Authors**: Sze Jue Yang, Chinh D. La, Quang H. Nguyen, Eugene Bagdasaryan, Kok-Seng Wong, Anh Tuan Tran, Chee Seng Chan, Khoa D. Doan

**Abstract**: Backdoor attacks, representing an emerging threat to the integrity of deep neural networks, have garnered significant attention due to their ability to compromise deep learning systems clandestinely. While numerous backdoor attacks occur within the digital realm, their practical implementation in real-world prediction systems remains limited and vulnerable to disturbances in the physical world. Consequently, this limitation has given rise to the development of physical backdoor attacks, where trigger objects manifest as physical entities within the real world. However, creating the requisite dataset to train or evaluate a physical backdoor model is a daunting task, limiting the backdoor researchers and practitioners from studying such physical attack scenarios. This paper unleashes a recipe that empowers backdoor researchers to effortlessly create a malicious, physical backdoor dataset based on advances in generative modeling. Particularly, this recipe involves 3 automatic modules: suggesting the suitable physical triggers, generating the poisoned candidate samples (either by synthesizing new samples or editing existing clean samples), and finally refining for the most plausible ones. As such, it effectively mitigates the perceived complexity associated with creating a physical backdoor dataset, transforming it from a daunting task into an attainable objective. Extensive experiment results show that datasets created by our "recipe" enable adversaries to achieve an impressive attack success rate on real physical world data and exhibit similar properties compared to previous physical backdoor attack studies. This paper offers researchers a valuable toolkit for studies of physical backdoors, all within the confines of their laboratories.

摘要: 后门攻击对深度神经网络的完整性构成了新的威胁，由于它们能够秘密地危害深度学习系统，因此引起了极大的关注。虽然在数字领域内发生了许多后门攻击，但它们在现实世界预测系统中的实际实施仍然有限，容易受到物理世界的干扰。因此，这种限制导致了物理后门攻击的发展，其中触发器对象在真实世界中表现为物理实体。然而，创建必要的数据集来训练或评估物理后门模型是一项艰巨的任务，限制了后门研究人员和实践者研究此类物理攻击场景。这篇文章揭示了一个配方，它使后门研究人员能够基于生成性建模的进步，毫不费力地创建恶意的物理后门数据集。特别是，这个配方涉及三个自动模块：建议合适的物理触发器，生成中毒的候选样本(通过合成新样本或编辑现有的干净样本)，最后提炼出最可信的样本。因此，它有效地减轻了与创建物理后门数据集相关的感知复杂性，将其从令人望而生畏的任务转变为可实现的目标。大量的实验结果表明，由我们的“配方”创建的数据集使攻击者能够在真实的物理世界数据上获得令人印象深刻的攻击成功率，并显示出与之前的物理后门攻击研究类似的特性。这篇论文为研究人员提供了一个宝贵的工具包，用于研究物理后门，所有这些都在他们的实验室范围内。



## **12. Adversarial Detection by Approximation of Ensemble Boundary**

基于集合边界逼近的对抗性检测 cs.LG

17 pages, 5 figures, 5 tables

**SubmitDate**: 2024-01-24    [abs](http://arxiv.org/abs/2211.10227v4) [paper-pdf](http://arxiv.org/pdf/2211.10227v4)

**Authors**: T. Windeatt

**Abstract**: A new method of detecting adversarial attacks is proposed for an ensemble of Deep Neural Networks (DNNs) solving two-class pattern recognition problems. The ensemble is combined using Walsh coefficients which are capable of approximating Boolean functions and thereby controlling the complexity of the ensemble decision boundary. The hypothesis in this paper is that decision boundaries with high curvature allow adversarial perturbations to be found, but change the curvature of the decision boundary, which is then approximated in a different way by Walsh coefficients compared to the clean images. By observing the difference in Walsh coefficient approximation between clean and adversarial images, it is shown experimentally that transferability of attack may be used for detection. Furthermore, approximating the decision boundary may aid in understanding the learning and transferability properties of DNNs. While the experiments here use images, the proposed approach of modelling two-class ensemble decision boundaries could in principle be applied to any application area. Code for approximating Boolean functions using Walsh coefficients: https://doi.org/10.24433/CO.3695905.v1

摘要: 提出了一种检测对抗性攻击的新方法，用于解决两类模式识别问题的深度神经网络（DNN）集成。使用能够近似布尔函数的沃尔什系数来组合系综，从而控制系综判决边界的复杂性。本文的假设是，高曲率的决策边界允许对抗扰动被发现，但改变的曲率的决策边界，然后近似在一个不同的方式相比，干净的图像沃尔什系数。通过观察干净和敌对图像之间的沃尔什系数近似的差异，实验表明，攻击的可转移性可以用于检测。此外，近似决策边界可能有助于理解DNN的学习和可转移性。虽然这里的实验使用图像，但所提出的建模两类集成决策边界的方法原则上可以应用于任何应用领域。使用沃尔什系数近似布尔函数的代码：https://doi.org/10.24433/CO.3695905.v1



## **13. Inference Attacks Against Face Recognition Model without Classification Layers**

对无分类层人脸识别模型的推理攻击 cs.CV

**SubmitDate**: 2024-01-24    [abs](http://arxiv.org/abs/2401.13719v1) [paper-pdf](http://arxiv.org/pdf/2401.13719v1)

**Authors**: Yuanqing Huang, Huilong Chen, Yinggui Wang, Lei Wang

**Abstract**: Face recognition (FR) has been applied to nearly every aspect of daily life, but it is always accompanied by the underlying risk of leaking private information. At present, almost all attack models against FR rely heavily on the presence of a classification layer. However, in practice, the FR model can obtain complex features of the input via the model backbone, and then compare it with the target for inference, which does not explicitly involve the outputs of the classification layer adopting logit or other losses. In this work, we advocate a novel inference attack composed of two stages for practical FR models without a classification layer. The first stage is the membership inference attack. Specifically, We analyze the distances between the intermediate features and batch normalization (BN) parameters. The results indicate that this distance is a critical metric for membership inference. We thus design a simple but effective attack model that can determine whether a face image is from the training dataset or not. The second stage is the model inversion attack, where sensitive private data is reconstructed using a pre-trained generative adversarial network (GAN) guided by the attack model in the first stage. To the best of our knowledge, the proposed attack model is the very first in the literature developed for FR models without a classification layer. We illustrate the application of the proposed attack model in the establishment of privacy-preserving FR techniques.

摘要: 人脸识别几乎已经应用到日常生活的方方面面，但它总是伴随着潜在的隐私信息泄露风险。目前，几乎所有针对FR的攻击模型都严重依赖于分类层的存在。然而，在实际应用中，FR模型可以通过模型主干获取输入的复杂特征，然后将其与用于推理的目标进行比较，而不是显式地涉及采用Logit或其他损失的分类层的输出。在这项工作中，我们对没有分类层的实际FR模型提出了一种由两个阶段组成的新的推理攻击。第一阶段是成员推理攻击。具体地，我们分析了中间特征和批归一化(BN)参数之间的距离。结果表明，该距离是隶属度推理的一个重要度量。因此，我们设计了一个简单而有效的攻击模型，该模型可以判断人脸图像是否来自训练数据集。第二阶段是模型反转攻击，在第一阶段，在攻击模型的指导下，使用预先训练的生成性对抗网络(GAN)重建敏感隐私数据。据我们所知，提出的攻击模型是为没有分类层的FR模型开发的第一个。我们举例说明了提出的攻击模型在隐私保护FR技术的建立中的应用。



## **14. AdCorDA: Classifier Refinement via Adversarial Correction and Domain Adaptation**

AdCorDA：通过对抗性校正和领域自适应的分类器改进 cs.CV

**SubmitDate**: 2024-01-24    [abs](http://arxiv.org/abs/2401.13212v1) [paper-pdf](http://arxiv.org/pdf/2401.13212v1)

**Authors**: Lulan Shen, Ali Edalati, Brett Meyer, Warren Gross, James J. Clark

**Abstract**: This paper describes a simple yet effective technique for refining a pretrained classifier network. The proposed AdCorDA method is based on modification of the training set and making use of the duality between network weights and layer inputs. We call this input space training. The method consists of two stages - adversarial correction followed by domain adaptation. Adversarial correction uses adversarial attacks to correct incorrect training-set classifications. The incorrectly classified samples of the training set are removed and replaced with the adversarially corrected samples to form a new training set, and then, in the second stage, domain adaptation is performed back to the original training set. Extensive experimental validations show significant accuracy boosts of over 5% on the CIFAR-100 dataset. The technique can be straightforwardly applied to refinement of weight-quantized neural networks, where experiments show substantial enhancement in performance over the baseline. The adversarial correction technique also results in enhanced robustness to adversarial attacks.

摘要: 本文描述了一种简单而有效的技术来精炼预先训练的分类器网络。AdCorDA方法基于对训练集的修改，利用了网络权值和层输入之间的对偶性。我们称之为输入空间训练。该方法包括两个阶段--对抗性校正和领域自适应。对抗性校正使用对抗性攻击来校正不正确的训练集分类。去除训练集的错误分类样本，并将其替换为恶意校正的样本，以形成新的训练集，然后，在第二阶段，对原始训练集执行领域自适应。大量的实验验证表明，在CIFAR-100数据集上，准确率显著提高了5%以上。该技术可以直接应用于加权量化神经网络的精化，实验表明，该技术在性能上比基线有了很大的提高。对抗性纠正技术还增强了对对抗性攻击的稳健性。



## **15. How Johnny Can Persuade LLMs to Jailbreak Them: Rethinking Persuasion to Challenge AI Safety by Humanizing LLMs**

约翰尼如何说服低层管理人员越狱：通过将低层管理人员人性化来挑战人工智能安全的再思考 cs.CL

14 pages of the main text, qualitative examples of jailbreaks may be  harmful in nature

**SubmitDate**: 2024-01-23    [abs](http://arxiv.org/abs/2401.06373v2) [paper-pdf](http://arxiv.org/pdf/2401.06373v2)

**Authors**: Yi Zeng, Hongpeng Lin, Jingwen Zhang, Diyi Yang, Ruoxi Jia, Weiyan Shi

**Abstract**: Most traditional AI safety research has approached AI models as machines and centered on algorithm-focused attacks developed by security experts. As large language models (LLMs) become increasingly common and competent, non-expert users can also impose risks during daily interactions. This paper introduces a new perspective to jailbreak LLMs as human-like communicators, to explore this overlooked intersection between everyday language interaction and AI safety. Specifically, we study how to persuade LLMs to jailbreak them. First, we propose a persuasion taxonomy derived from decades of social science research. Then, we apply the taxonomy to automatically generate interpretable persuasive adversarial prompts (PAP) to jailbreak LLMs. Results show that persuasion significantly increases the jailbreak performance across all risk categories: PAP consistently achieves an attack success rate of over $92\%$ on Llama 2-7b Chat, GPT-3.5, and GPT-4 in $10$ trials, surpassing recent algorithm-focused attacks. On the defense side, we explore various mechanisms against PAP and, found a significant gap in existing defenses, and advocate for more fundamental mitigation for highly interactive LLMs

摘要: 大多数传统的人工智能安全研究都将人工智能模型视为机器，并集中在安全专家开发的以算法为重点的攻击上。随着大型语言模型(LLM)变得越来越普遍和有能力，非专家用户也可能在日常交互中带来风险。本文介绍了一种新的视角，将越狱LLMS作为类人类的沟通者，来探索日常语言交互和人工智能安全之间被忽视的交集。具体地说，我们研究如何说服LLMS越狱。首先，我们提出了一种源于数十年社会科学研究的说服分类法。然后，我们应用分类法自动生成可解释的说服性对抗性提示(PAP)来越狱LLM。结果表明，说服显著提高了所有风险类别的越狱性能：PAP在Llama 2-7b Chat、GPT-3.5和GPT-4上的攻击成功率在10美元的试验中始终保持在92美元以上，超过了最近针对算法的攻击。在防御方面，我们探索了各种对抗PAP和的机制，发现了现有防御措施中的显著差距，并倡导从更根本上缓解高度互动的LLM



## **16. MAPPING: Debiasing Graph Neural Networks for Fair Node Classification with Limited Sensitive Information Leakage**

映射：有限敏感信息泄漏的无偏图神经网络公平节点分类 cs.LG

Finished May last year. Remember to submit all papers to arXiv early  without compromising the principles of conferences

**SubmitDate**: 2024-01-23    [abs](http://arxiv.org/abs/2401.12824v1) [paper-pdf](http://arxiv.org/pdf/2401.12824v1)

**Authors**: Ying Song, Balaji Palanisamy

**Abstract**: Despite remarkable success in diverse web-based applications, Graph Neural Networks(GNNs) inherit and further exacerbate historical discrimination and social stereotypes, which critically hinder their deployments in high-stake domains such as online clinical diagnosis, financial crediting, etc. However, current fairness research that primarily craft on i.i.d data, cannot be trivially replicated to non-i.i.d. graph structures with topological dependence among samples. Existing fair graph learning typically favors pairwise constraints to achieve fairness but fails to cast off dimensional limitations and generalize them into multiple sensitive attributes; besides, most studies focus on in-processing techniques to enforce and calibrate fairness, constructing a model-agnostic debiasing GNN framework at the pre-processing stage to prevent downstream misuses and improve training reliability is still largely under-explored. Furthermore, previous work on GNNs tend to enhance either fairness or privacy individually but few probe into their interplays. In this paper, we propose a novel model-agnostic debiasing framework named MAPPING (\underline{M}asking \underline{A}nd \underline{P}runing and Message-\underline{P}assing train\underline{ING}) for fair node classification, in which we adopt the distance covariance($dCov$)-based fairness constraints to simultaneously reduce feature and topology biases in arbitrary dimensions, and combine them with adversarial debiasing to confine the risks of attribute inference attacks. Experiments on real-world datasets with different GNN variants demonstrate the effectiveness and flexibility of MAPPING. Our results show that MAPPING can achieve better trade-offs between utility and fairness, and mitigate privacy risks of sensitive information leakage.

摘要: 尽管图形神经网络(GNN)在各种基于网络的应用中取得了显著的成功，但它继承并进一步加剧了历史歧视和社会刻板印象，这严重阻碍了它们在高风险领域的部署，如在线临床诊断、金融信贷等。然而，目前主要基于身份识别数据的公平研究不能简单地复制到非身份识别领域。样本间具有拓扑依赖关系的图结构。现有的公平图学习一般倾向于两两约束来实现公平性，但未能摆脱维度限制并将其概括为多个敏感属性；此外，大多数研究侧重于内处理技术来加强和校准公平性，在前处理阶段构建模型不可知的去偏向GNN框架以防止下游误用，提高训练可靠性，还在很大程度上探索不足。此外，以前关于GNN的工作往往会单独提高公平性或隐私性，但很少有人探讨它们之间的相互作用。针对公平节点分类问题，提出了一种新的模型不可知去偏框架：映射(下划线{M}询问{A}和下划线{P}运行，消息下划线{P}通过训练\下划线{ING})，其中我们采用基于距离协方差($dCov$)的公平性约束来同时减少任意维度上的特征和拓扑偏差，并将其与对抗性去偏向相结合来控制属性推理攻击的风险。在具有不同GNN变量的真实数据集上的实验证明了该映射的有效性和灵活性。我们的结果表明，映射可以在效用和公平性之间实现更好的权衡，并降低敏感信息泄露的隐私风险。



## **17. The twin peaks of learning neural networks**

学习神经网络的双峰 cs.LG

36 pages, 30 figures

**SubmitDate**: 2024-01-23    [abs](http://arxiv.org/abs/2401.12610v1) [paper-pdf](http://arxiv.org/pdf/2401.12610v1)

**Authors**: Elizaveta Demyanenko, Christoph Feinauer, Enrico M. Malatesta, Luca Saglietti

**Abstract**: Recent works demonstrated the existence of a double-descent phenomenon for the generalization error of neural networks, where highly overparameterized models escape overfitting and achieve good test performance, at odds with the standard bias-variance trade-off described by statistical learning theory. In the present work, we explore a link between this phenomenon and the increase of complexity and sensitivity of the function represented by neural networks. In particular, we study the Boolean mean dimension (BMD), a metric developed in the context of Boolean function analysis. Focusing on a simple teacher-student setting for the random feature model, we derive a theoretical analysis based on the replica method that yields an interpretable expression for the BMD, in the high dimensional regime where the number of data points, the number of features, and the input size grow to infinity. We find that, as the degree of overparameterization of the network is increased, the BMD reaches an evident peak at the interpolation threshold, in correspondence with the generalization error peak, and then slowly approaches a low asymptotic value. The same phenomenology is then traced in numerical experiments with different model classes and training setups. Moreover, we find empirically that adversarially initialized models tend to show higher BMD values, and that models that are more robust to adversarial attacks exhibit a lower BMD.

摘要: 最近的工作证明了神经网络泛化误差存在双下降现象，即高度过参数的模型避免了过拟合并获得了良好的测试性能，这与统计学习理论所描述的标准偏差-方差权衡不一致。在目前的工作中，我们探索了这种现象与神经网络表示的函数的复杂性和敏感度的增加之间的联系。特别是，我们研究了布尔平均维度(BMD)，这是在布尔函数分析的背景下发展起来的一种度量。针对一个简单的教师-学生随机特征模型，我们基于复制品方法进行了理论分析，在数据点数目、特征数目和输入大小都增长到无穷大的高维区域中，给出了一个可解释的BMD表达式。我们发现，随着网络的超参数化程度的增加，BMD在与泛化误差峰值相对应的内插阈值处达到一个明显的峰值，然后缓慢地接近一个较低的渐近值。然后在不同模型类别和训练设置的数值实验中追踪相同的现象学。此外，我们从经验上发现，对抗性初始化的模型往往显示出较高的BMD值，而对对抗性攻击越健壮的模型显示出较低的BMD。



## **18. ToDA: Target-oriented Diffusion Attacker against Recommendation System**

Toda：面向目标的扩散攻击推荐系统 cs.CR

**SubmitDate**: 2024-01-23    [abs](http://arxiv.org/abs/2401.12578v1) [paper-pdf](http://arxiv.org/pdf/2401.12578v1)

**Authors**: Xiaohao Liu, Zhulin Tao, Ting Jiang, He Chang, Yunshan Ma, Xianglin Huang

**Abstract**: Recommendation systems (RS) have become indispensable tools for web services to address information overload, thus enhancing user experiences and bolstering platforms' revenues. However, with their increasing ubiquity, security concerns have also emerged. As the public accessibility of RS, they are susceptible to specific malicious attacks where adversaries can manipulate user profiles, leading to biased recommendations. Recent research often integrates additional modules using generative models to craft these deceptive user profiles, ensuring them are imperceptible while causing the intended harm. Albeit their efficacy, these models face challenges of unstable training and the exploration-exploitation dilemma, which can lead to suboptimal results. In this paper, we pioneer to investigate the potential of diffusion models (DMs), for shilling attacks. Specifically, we propose a novel Target-oriented Diffusion Attack model (ToDA). It incorporates a pre-trained autoencoder that transforms user profiles into a high dimensional space, paired with a Latent Diffusion Attacker (LDA)-the core component of ToDA. LDA introduces noise into the profiles within this latent space, adeptly steering the approximation towards targeted items through cross-attention mechanisms. The global horizon, implemented by a bipartite graph, is involved in LDA and derived from the encoded user profile feature. This makes LDA possible to extend the generation outwards the on-processing user feature itself, and bridges the gap between diffused user features and target item features. Extensive experiments compared to several SOTA baselines demonstrate ToDA's effectiveness. Specific studies exploit the elaborative design of ToDA and underscore the potency of advanced generative models in such contexts.

摘要: 推荐系统(RS)已经成为Web服务解决信息过载的不可或缺的工具，从而增强了用户体验并增加了平台的收入。然而，随着它们越来越普遍，安全问题也出现了。由于RS的公共可访问性，它们容易受到特定的恶意攻击，攻击者可以操纵用户配置文件，导致有偏见的推荐。最近的研究经常使用生成性模型集成额外的模块来制作这些欺骗性的用户配置文件，确保它们在造成预期伤害的同时是不可察觉的。尽管这些模式很有效，但它们面临着训练不稳定和勘探-开采困境的挑战，这可能导致不太理想的结果。在本文中，我们率先研究了扩散模型(DM)对先令攻击的可能性。具体来说，我们提出了一种新的面向目标的扩散攻击模型(Toda)。它结合了一个预先训练的自动编码器，可以将用户配置文件转换到高维空间，并与Toda的核心组件潜在扩散攻击者(LDA)配对。LDA将噪声引入到这个潜在空间内的轮廓中，通过交叉注意机制熟练地将近似引导到目标项目。全局地平线由二部图实现，涉及LDA，并从编码的用户简档特征中派生出来。这使得LDA有可能将生成向外扩展到正在处理的用户功能本身，并弥合扩散的用户功能和目标项目功能之间的差距。与几个SOTA基线相比的广泛实验证明了Toda的有效性。具体的研究利用了户田的精心设计，并强调了高级生成模式在这种背景下的效力。



## **19. Iterative Adversarial Attack on Image-guided Story Ending Generation**

图像导引故事结尾生成的迭代对抗性攻击 cs.CV

**SubmitDate**: 2024-01-23    [abs](http://arxiv.org/abs/2305.13208v2) [paper-pdf](http://arxiv.org/pdf/2305.13208v2)

**Authors**: Youze Wang, Wenbo Hu, Richang Hong

**Abstract**: Multimodal learning involves developing models that can integrate information from various sources like images and texts. In this field, multimodal text generation is a crucial aspect that involves processing data from multiple modalities and outputting text. The image-guided story ending generation (IgSEG) is a particularly significant task, targeting on an understanding of complex relationships between text and image data with a complete story text ending. Unfortunately, deep neural networks, which are the backbone of recent IgSEG models, are vulnerable to adversarial samples. Current adversarial attack methods mainly focus on single-modality data and do not analyze adversarial attacks for multimodal text generation tasks that use cross-modal information. To this end, we propose an iterative adversarial attack method (Iterative-attack) that fuses image and text modality attacks, allowing for an attack search for adversarial text and image in an more effective iterative way. Experimental results demonstrate that the proposed method outperforms existing single-modal and non-iterative multimodal attack methods, indicating the potential for improving the adversarial robustness of multimodal text generation models, such as multimodal machine translation, multimodal question answering, etc.

摘要: 多模态学习涉及开发可以整合来自图像和文本等各种来源的信息的模型。在这个领域中，多模态文本生成是一个至关重要的方面，涉及到处理来自多个模态的数据并输出文本。图像引导的故事结尾生成是一项特别重要的任务，其目标是理解文本和图像数据之间的复杂关系，并获得完整的故事文本结尾。不幸的是，作为最近IgSEG模型的支柱的深度神经网络很容易受到对抗性样本的影响。目前的对抗性攻击方法主要集中在单模态数据上，没有分析针对使用跨模态信息的多模态文本生成任务的对抗性攻击。为此，我们提出了一种迭代对抗攻击方法（迭代攻击），融合了图像和文本模态攻击，允许以更有效的迭代方式对对抗文本和图像进行攻击搜索。实验结果表明，该方法优于现有的单模态和非迭代多模态攻击方法，具有提高多模态文本生成模型（如多模态机器翻译、多模态问答等）对抗鲁棒性的潜力。



## **20. DAFA: Distance-Aware Fair Adversarial Training**

Dafa：距离感知的公平对抗训练 cs.LG

Accepted to ICLR 2024

**SubmitDate**: 2024-01-23    [abs](http://arxiv.org/abs/2401.12532v1) [paper-pdf](http://arxiv.org/pdf/2401.12532v1)

**Authors**: Hyungyu Lee, Saehyung Lee, Hyemi Jang, Junsung Park, Ho Bae, Sungroh Yoon

**Abstract**: The disparity in accuracy between classes in standard training is amplified during adversarial training, a phenomenon termed the robust fairness problem. Existing methodologies aimed to enhance robust fairness by sacrificing the model's performance on easier classes in order to improve its performance on harder ones. However, we observe that under adversarial attacks, the majority of the model's predictions for samples from the worst class are biased towards classes similar to the worst class, rather than towards the easy classes. Through theoretical and empirical analysis, we demonstrate that robust fairness deteriorates as the distance between classes decreases. Motivated by these insights, we introduce the Distance-Aware Fair Adversarial training (DAFA) methodology, which addresses robust fairness by taking into account the similarities between classes. Specifically, our method assigns distinct loss weights and adversarial margins to each class and adjusts them to encourage a trade-off in robustness among similar classes. Experimental results across various datasets demonstrate that our method not only maintains average robust accuracy but also significantly improves the worst robust accuracy, indicating a marked improvement in robust fairness compared to existing methods.

摘要: 在对抗训练中，标准训练中类之间的准确性差距被放大，这种现象被称为鲁棒公平问题。现有的方法旨在通过牺牲模型在较容易类上的性能来提高其在较难类上的性能，从而增强鲁棒公平性。然而，我们观察到，在对抗性攻击下，模型对最差类样本的大多数预测都偏向于与最差类相似的类，而不是偏向于简单类。通过理论和实证分析，我们证明了鲁棒公平恶化类之间的距离减小。基于这些见解，我们引入了距离感知公平对抗训练（DAFA）方法，该方法通过考虑类之间的相似性来解决鲁棒公平性问题。具体来说，我们的方法为每个类别分配不同的损失权重和对抗性保证金，并对其进行调整，以鼓励在相似类别之间进行鲁棒性权衡。在不同数据集上的实验结果表明，我们的方法不仅保持了平均鲁棒精度，而且显着提高了最差的鲁棒精度，表明与现有方法相比，鲁棒公平性有显着改善。



## **21. Explainability-Driven Leaf Disease Classification Using Adversarial Training and Knowledge Distillation**

基于对抗性训练和知识提炼的可解释性叶部病害分类 cs.CV

10 pages, 8 figures, Accepted by ICAART 2024

**SubmitDate**: 2024-01-23    [abs](http://arxiv.org/abs/2401.00334v3) [paper-pdf](http://arxiv.org/pdf/2401.00334v3)

**Authors**: Sebastian-Vasile Echim, Iulian-Marius Tăiatu, Dumitru-Clementin Cercel, Florin Pop

**Abstract**: This work focuses on plant leaf disease classification and explores three crucial aspects: adversarial training, model explainability, and model compression. The models' robustness against adversarial attacks is enhanced through adversarial training, ensuring accurate classification even in the presence of threats. Leveraging explainability techniques, we gain insights into the model's decision-making process, improving trust and transparency. Additionally, we explore model compression techniques to optimize computational efficiency while maintaining classification performance. Through our experiments, we determine that on a benchmark dataset, the robustness can be the price of the classification accuracy with performance reductions of 3%-20% for regular tests and gains of 50%-70% for adversarial attack tests. We also demonstrate that a student model can be 15-25 times more computationally efficient for a slight performance reduction, distilling the knowledge of more complex models.

摘要: 这项工作集中在植物叶部病害的分类上，并探索了三个关键方面：对抗性训练、模型可解释性和模型压缩。通过对抗性训练增强了模型对对抗性攻击的稳健性，即使在存在威胁的情况下也确保了准确的分类。利用可解释性技术，我们可以深入了解模型的决策过程，从而提高信任和透明度。此外，我们还探索了模型压缩技术，以优化计算效率，同时保持分类性能。通过实验，我们确定在一个基准数据集上，在常规测试性能降低3%-20%，对抗性攻击测试性能提高50%-70%的情况下，鲁棒性可以是分类准确率的代价。我们还证明，学生模型的计算效率可以是15-25倍，而性能略有下降，提取了更复杂模型的知识。



## **22. Fast Adversarial Training against Textual Adversarial Attacks**

针对文本对抗性攻击的快速对抗性训练 cs.CL

4 pages, 4 figures

**SubmitDate**: 2024-01-23    [abs](http://arxiv.org/abs/2401.12461v1) [paper-pdf](http://arxiv.org/pdf/2401.12461v1)

**Authors**: Yichen Yang, Xin Liu, Kun He

**Abstract**: Many adversarial defense methods have been proposed to enhance the adversarial robustness of natural language processing models. However, most of them introduce additional pre-set linguistic knowledge and assume that the synonym candidates used by attackers are accessible, which is an ideal assumption. We delve into adversarial training in the embedding space and propose a Fast Adversarial Training (FAT) method to improve the model robustness in the synonym-unaware scenario from the perspective of single-step perturbation generation and perturbation initialization. Based on the observation that the adversarial perturbations crafted by single-step and multi-step gradient ascent are similar, FAT uses single-step gradient ascent to craft adversarial examples in the embedding space to expedite the training process. Based on the observation that the perturbations generated on the identical training sample in successive epochs are similar, FAT fully utilizes historical information when initializing the perturbation. Extensive experiments demonstrate that FAT significantly boosts the robustness of BERT models in the synonym-unaware scenario, and outperforms the defense baselines under various attacks with character-level and word-level modifications.

摘要: 为了增强自然语言处理模型的对抗性，人们提出了许多对抗性防御方法。然而，它们大多引入了额外的预设语言知识，并假设攻击者使用的同义词候选是可访问的，这是一个理想的假设。深入研究了嵌入空间中的对抗性训练，从单步扰动生成和扰动初始化的角度提出了一种快速对抗性训练(FAT)方法，以提高模型在同义词未知场景下的稳健性。在观察到单步和多步梯度上升产生的对抗性扰动相似的基础上，FAT使用单步梯度上升在嵌入空间中构造对抗性样本，以加快训练过程。基于相同训练样本在连续历元上产生的扰动相似的观察，FAT在初始化扰动时充分利用了历史信息。大量实验表明，FAT在不知道同义词的情况下显著提高了BERT模型的健壮性，并且在字级和词级修改的各种攻击下的表现优于防御基线。



## **23. CasTGAN: Cascaded Generative Adversarial Network for Realistic Tabular Data Synthesis**

CasTGAN：用于现实表格数据合成的级联生成性对抗网络 cs.LG

**SubmitDate**: 2024-01-22    [abs](http://arxiv.org/abs/2307.00384v2) [paper-pdf](http://arxiv.org/pdf/2307.00384v2)

**Authors**: Abdallah Alshantti, Damiano Varagnolo, Adil Rasheed, Aria Rahmati, Frank Westad

**Abstract**: Generative adversarial networks (GANs) have drawn considerable attention in recent years for their proven capability in generating synthetic data which can be utilised for multiple purposes. While GANs have demonstrated tremendous successes in producing synthetic data samples that replicate the dynamics of the original datasets, the validity of the synthetic data and the underlying privacy concerns represent major challenges which are not sufficiently addressed. In this work, we design a cascaded tabular GAN framework (CasTGAN) for generating realistic tabular data with a specific focus on the validity of the output. In this context, validity refers to the the dependency between features that can be found in the real data, but is typically misrepresented by traditional generative models. Our key idea entails that employing a cascaded architecture in which a dedicated generator samples each feature, the synthetic output becomes more representative of the real data. Our experimental results demonstrate that our model is capable of generating synthetic tabular data that can be used for fitting machine learning models. In addition, our model captures well the constraints and the correlations between the features of the real data, especially the high dimensional datasets. Furthermore, we evaluate the risk of white-box privacy attacks on our model and subsequently show that applying some perturbations to the auxiliary learners in CasTGAN increases the overall robustness of our model against targeted attacks.

摘要: 近年来，生成性对抗网络(GAN)因其在生成可用于多种目的的合成数据方面的已被证明的能力而引起了相当大的关注。尽管Gans在生产复制原始数据集动态的合成数据样本方面取得了巨大成功，但合成数据的有效性和潜在的隐私问题是没有得到充分解决的主要挑战。在这项工作中，我们设计了一个级联表格GAN框架(CasTGAN)，用于生成真实的表格数据，并特别关注输出的有效性。在这种情况下，有效性是指在真实数据中可以找到的特征之间的依赖关系，但传统的生成模型通常会错误地表示这些特征。我们的关键思想是采用级联结构，其中由专用生成器对每个特征进行采样，合成输出变得更能代表真实数据。我们的实验结果表明，我们的模型能够生成可用于拟合机器学习模型的合成表格数据。此外，我们的模型很好地捕捉了真实数据，特别是高维数据集的约束和特征之间的相关性。此外，我们评估了我们的模型受到白盒隐私攻击的风险，并随后表明，对CasTGAN中的辅助学习器应用一些扰动可以提高模型对目标攻击的整体稳健性。



## **24. Benchmarking the Robustness of Image Watermarks**

图像水印稳健性的基准测试 cs.CV

**SubmitDate**: 2024-01-22    [abs](http://arxiv.org/abs/2401.08573v2) [paper-pdf](http://arxiv.org/pdf/2401.08573v2)

**Authors**: Bang An, Mucong Ding, Tahseen Rabbani, Aakriti Agrawal, Yuancheng Xu, Chenghao Deng, Sicheng Zhu, Abdirisak Mohamed, Yuxin Wen, Tom Goldstein, Furong Huang

**Abstract**: This paper investigates the weaknesses of image watermarking techniques. We present WAVES (Watermark Analysis Via Enhanced Stress-testing), a novel benchmark for assessing watermark robustness, overcoming the limitations of current evaluation methods.WAVES integrates detection and identification tasks, and establishes a standardized evaluation protocol comprised of a diverse range of stress tests. The attacks in WAVES range from traditional image distortions to advanced and novel variations of diffusive, and adversarial attacks. Our evaluation examines two pivotal dimensions: the degree of image quality degradation and the efficacy of watermark detection after attacks. We develop a series of Performance vs. Quality 2D plots, varying over several prominent image similarity metrics, which are then aggregated in a heuristically novel manner to paint an overall picture of watermark robustness and attack potency. Our comprehensive evaluation reveals previously undetected vulnerabilities of several modern watermarking algorithms. We envision WAVES as a toolkit for the future development of robust watermarking systems. The project is available at https://wavesbench.github.io/

摘要: 本文研究了图像水印技术的弱点。为了克服现有评估方法的局限性，提出了一种新的评估水印稳健性的基准--WAVES(基于增强压力测试的水印分析)，它集成了检测和识别任务，并建立了由多种压力测试组成的标准化评估协议。一波一波的攻击范围从传统的图像扭曲到高级和新奇的扩散性和对抗性攻击。我们的评估检查了两个关键维度：图像质量下降的程度和攻击后水印检测的有效性。我们开发了一系列性能与质量的2D图，不同于几个重要的图像相似性度量，然后以启发式的新颖方式聚集在一起，描绘了水印稳健性和攻击能力的总体图景。我们的综合评估揭示了几种现代水印算法以前未被检测到的漏洞。我们设想Waves将成为未来健壮水印系统发展的工具包。该项目的网址为：https://wavesbench.github.io/。



## **25. NEUROSEC: FPGA-Based Neuromorphic Audio Security**

NeurOSEC：基于FPGA的神经形态音频安全 cs.CR

Audio processing, FPGA, Hardware Security, Neuromorphic Computing

**SubmitDate**: 2024-01-22    [abs](http://arxiv.org/abs/2401.12055v1) [paper-pdf](http://arxiv.org/pdf/2401.12055v1)

**Authors**: Murat Isik, Hiruna Vishwamith, Yusuf Sur, Kayode Inadagbo, I. Can Dikmen

**Abstract**: Neuromorphic systems, inspired by the complexity and functionality of the human brain, have gained interest in academic and industrial attention due to their unparalleled potential across a wide range of applications. While their capabilities herald innovation, it is imperative to underscore that these computational paradigms, analogous to their traditional counterparts, are not impervious to security threats. Although the exploration of neuromorphic methodologies for image and video processing has been rigorously pursued, the realm of neuromorphic audio processing remains in its early stages. Our results highlight the robustness and precision of our FPGA-based neuromorphic system. Specifically, our system showcases a commendable balance between desired signal and background noise, efficient spike rate encoding, and unparalleled resilience against adversarial attacks such as FGSM and PGD. A standout feature of our framework is its detection rate of 94%, which, when compared to other methodologies, underscores its greater capability in identifying and mitigating threats within 5.39 dB, a commendable SNR ratio. Furthermore, neuromorphic computing and hardware security serve many sensor domains in mission-critical and privacy-preserving applications.

摘要: 受人脑的复杂性和功能性启发，神经形态系统因其在广泛应用中的无与伦比的潜力而引起了学术界和工业界的兴趣。虽然它们的能力预示着创新，但必须强调的是，这些与传统计算模式类似的计算模式并非不受安全威胁的影响。尽管人们一直在探索用于图像和视频处理的神经形态方法，但神经形态音频处理领域仍处于早期阶段。我们的结果突出了我们的基于FPGA的神经形态系统的健壮性和精确度。具体地说，我们的系统在期望的信号和背景噪声之间表现出了令人称赞的平衡，高效的尖峰速率编码，以及对FGSM和PGD等对手攻击的无与伦比的弹性。我们框架的一个突出特点是其94%的检测率，与其他方法相比，这突显了它在识别和缓解5.39分贝以内的威胁方面具有更强的能力，这是一个值得称赞的信噪比。此外，神经形态计算和硬件安全为许多关键任务和隐私保护应用中的传感器领域提供服务。



## **26. The Effect of Intrinsic Dataset Properties on Generalization: Unraveling Learning Differences Between Natural and Medical Images**

数据集属性对概化的影响：解开自然图像和医学图像之间的学习差异 cs.CV

ICLR 2024. Code:  https://github.com/mazurowski-lab/intrinsic-properties

**SubmitDate**: 2024-01-22    [abs](http://arxiv.org/abs/2401.08865v2) [paper-pdf](http://arxiv.org/pdf/2401.08865v2)

**Authors**: Nicholas Konz, Maciej A. Mazurowski

**Abstract**: This paper investigates discrepancies in how neural networks learn from different imaging domains, which are commonly overlooked when adopting computer vision techniques from the domain of natural images to other specialized domains such as medical images. Recent works have found that the generalization error of a trained network typically increases with the intrinsic dimension ($d_{data}$) of its training set. Yet, the steepness of this relationship varies significantly between medical (radiological) and natural imaging domains, with no existing theoretical explanation. We address this gap in knowledge by establishing and empirically validating a generalization scaling law with respect to $d_{data}$, and propose that the substantial scaling discrepancy between the two considered domains may be at least partially attributed to the higher intrinsic "label sharpness" ($K_F$) of medical imaging datasets, a metric which we propose. Next, we demonstrate an additional benefit of measuring the label sharpness of a training set: it is negatively correlated with the trained model's adversarial robustness, which notably leads to models for medical images having a substantially higher vulnerability to adversarial attack. Finally, we extend our $d_{data}$ formalism to the related metric of learned representation intrinsic dimension ($d_{repr}$), derive a generalization scaling law with respect to $d_{repr}$, and show that $d_{data}$ serves as an upper bound for $d_{repr}$. Our theoretical results are supported by thorough experiments with six models and eleven natural and medical imaging datasets over a range of training set sizes. Our findings offer insights into the influence of intrinsic dataset properties on generalization, representation learning, and robustness in deep neural networks.

摘要: 本文研究了神经网络如何从不同的成像领域学习的差异，这些差异是在将计算机视觉技术从自然图像领域应用到其他专业领域(如医学图像)时通常被忽视的。最近的工作发现，训练网络的泛化误差通常随着训练集的固有维度($d_{data}$)的增加而增加。然而，这种关系的陡峭程度在医学(放射)和自然成像领域有很大的不同，没有现有的理论解释。我们通过建立和经验性地验证关于$d{data}$的泛化标度律来解决这一知识缺口，并提出两个被考虑的域之间的显著标度差异至少部分归因于医学成像数据集的更高的固有“标签锐度”($K_F$)，这是我们提出的一种度量。接下来，我们展示了测量训练集的标签锐度的另一个好处：它与训练模型的对抗稳健性负相关，这显著地导致医学图像的模型具有更高的对抗攻击脆弱性。最后，我们将$d_{data}$形式推广到学习表示内在维的相关度量($d_{epr}$)，得到了关于$d_{epr}$的一个推广的标度律，并证明了$d_{data}$是$d_{epr}$的一个上界。我们的理论结果得到了6个模型和11个自然和医学成像数据集的全面实验的支持，这些数据集的训练集大小不同。我们的发现对深入神经网络中内在数据集属性对泛化、表示学习和稳健性的影响提供了深入的见解。



## **27. Diagnosis-guided Attack Recovery for Securing Robotic Vehicles from Sensor Deception Attacks**

用于保护机器人车辆免受传感器欺骗攻击的诊断引导攻击恢复 cs.RO

**SubmitDate**: 2024-01-22    [abs](http://arxiv.org/abs/2209.04554v5) [paper-pdf](http://arxiv.org/pdf/2209.04554v5)

**Authors**: Pritam Dash, Guanpeng Li, Mehdi Karimibiuki, Karthik Pattabiraman

**Abstract**: Sensors are crucial for perception and autonomous operation in robotic vehicles (RV). Unfortunately, RV sensors can be compromised by physical attacks such as sensor tampering or spoofing. In this paper, we present DeLorean, a unified framework for attack detection, attack diagnosis, and recovering RVs from sensor deception attacks (SDA). DeLorean can recover RVs even from strong SDAs in which the adversary targets multiple heterogeneous sensors simultaneously. We propose a novel attack diagnosis technique that inspects the attack-induced errors under SDAs, and identifies the targeted sensors using causal analysis. DeLorean then uses historic state information to selectively reconstruct physical states for compromised sensors, enabling targeted attack recovery under single or multi-sensor SDAs. We evaluate DeLorean on four real and two simulated RVs under SDAs targeting various sensors, and we find that it successfully recovers RVs from SDAs in 93% of the cases.

摘要: 传感器对于机器人车辆(RV)的感知和自主操作至关重要。不幸的是，RV传感器可能会受到物理攻击，如篡改传感器或欺骗。在本文中，我们提出了DeLorean，一个统一的攻击检测、攻击诊断和从传感器欺骗攻击(SDA)中恢复房车的框架。即使对手同时瞄准多个不同的传感器，DeLorean也可以从强大的SDA中恢复RV。我们提出了一种新的攻击诊断技术，该技术在SDAS下检测攻击导致的错误，并使用因果分析来识别目标传感器。然后，DeLorean使用历史状态信息有选择地重建受损传感器的物理状态，从而在单传感器或多传感器SDA下实现有针对性的攻击恢复。我们在四辆真实房车和两辆模拟房车上对DeLorean进行了评估，结果表明，在93%的情况下，DeLorean能够成功地从SDA中恢复房车。



## **28. A Training-Free Defense Framework for Robust Learned Image Compression**

一种无训练的稳健学习图像压缩防御框架 eess.IV

10 pages and 14 figures

**SubmitDate**: 2024-01-22    [abs](http://arxiv.org/abs/2401.11902v1) [paper-pdf](http://arxiv.org/pdf/2401.11902v1)

**Authors**: Myungseo Song, Jinyoung Choi, Bohyung Han

**Abstract**: We study the robustness of learned image compression models against adversarial attacks and present a training-free defense technique based on simple image transform functions. Recent learned image compression models are vulnerable to adversarial attacks that result in poor compression rate, low reconstruction quality, or weird artifacts. To address the limitations, we propose a simple but effective two-way compression algorithm with random input transforms, which is conveniently applicable to existing image compression models. Unlike the na\"ive approaches, our approach preserves the original rate-distortion performance of the models on clean images. Moreover, the proposed algorithm requires no additional training or modification of existing models, making it more practical. We demonstrate the effectiveness of the proposed techniques through extensive experiments under multiple compression models, evaluation metrics, and attack scenarios.

摘要: 我们研究了学习的图像压缩模型对对抗性攻击的鲁棒性，并提出了一种基于简单图像变换函数的免训练防御技术。最近学习的图像压缩模型容易受到对抗性攻击，导致压缩率低，重建质量低或奇怪的伪影。为了解决这些问题，我们提出了一个简单而有效的双向压缩算法，随机输入变换，这是方便地适用于现有的图像压缩模型。与原始方法不同，我们的方法保留了模型在干净图像上的原始率失真性能。此外，该算法不需要额外的训练或修改现有的模型，使其更实用。我们证明了所提出的技术的有效性，通过广泛的实验，在多种压缩模型，评估指标和攻击场景。



## **29. Adversarial speech for voice privacy protection from Personalized Speech generation**

针对个性化语音生成的语音隐私保护的对抗性语音 eess.AS

Accepted by icassp 2024

**SubmitDate**: 2024-01-22    [abs](http://arxiv.org/abs/2401.11857v1) [paper-pdf](http://arxiv.org/pdf/2401.11857v1)

**Authors**: Shihao Chen, Liping Chen, Jie Zhang, KongAik Lee, Zhenhua Ling, Lirong Dai

**Abstract**: The rapid progress in personalized speech generation technology, including personalized text-to-speech (TTS) and voice conversion (VC), poses a challenge in distinguishing between generated and real speech for human listeners, resulting in an urgent demand in protecting speakers' voices from malicious misuse. In this regard, we propose a speaker protection method based on adversarial attacks. The proposed method perturbs speech signals by minimally altering the original speech while rendering downstream speech generation models unable to accurately generate the voice of the target speaker. For validation, we employ the open-source pre-trained YourTTS model for speech generation and protect the target speaker's speech in the white-box scenario. Automatic speaker verification (ASV) evaluations were carried out on the generated speech as the assessment of the voice protection capability. Our experimental results show that we successfully perturbed the speaker encoder of the YourTTS model using the gradient-based I-FGSM adversarial perturbation method. Furthermore, the adversarial perturbation is effective in preventing the YourTTS model from generating the speech of the target speaker. Audio samples can be found in https://voiceprivacy.github.io/Adeversarial-Speech-with-YourTTS.

摘要: 个性化语音生成技术的快速发展，包括个性化文语转换(TTS)和语音转换(VC)，对人类听者区分生成的语音和真实的语音提出了挑战，导致了对保护说话人的语音免受恶意滥用的迫切需求。对此，我们提出了一种基于对抗性攻击的说话人保护方法。所提出的方法通过最小限度地改变原始语音来扰动语音信号，同时使得下游语音生成模型无法准确地生成目标说话人的声音。为了验证，我们使用开源的预先训练的YourTTS模型来生成语音，并在白盒场景中保护目标说话人的语音。对生成的语音进行自动说话人验证(ASV)评估，以评估语音保护能力。实验结果表明，我们使用基于梯度的I-FGSM对抗扰动方法成功地扰动了YourTTS模型的说话人编码器。此外，对抗性扰动可以有效地防止YourTTS模型生成目标说话人的语音。音频样本可在https://voiceprivacy.github.io/Adeversarial-Speech-with-YourTTS.中找到



## **30. Unraveling Attacks in Machine Learning-based IoT Ecosystems: A Survey and the Open Libraries Behind Them**

破解基于机器学习的物联网生态系统中的攻击：综述及其背后的开放图书馆 cs.CR

**SubmitDate**: 2024-01-22    [abs](http://arxiv.org/abs/2401.11723v1) [paper-pdf](http://arxiv.org/pdf/2401.11723v1)

**Authors**: Chao Liu, Boxi Chen, Wei Shao, Chris Zhang, Kelvin Wong, Yi Zhang

**Abstract**: The advent of the Internet of Things (IoT) has brought forth an era of unprecedented connectivity, with an estimated 80 billion smart devices expected to be in operation by the end of 2025. These devices facilitate a multitude of smart applications, enhancing the quality of life and efficiency across various domains. Machine Learning (ML) serves as a crucial technology, not only for analyzing IoT-generated data but also for diverse applications within the IoT ecosystem. For instance, ML finds utility in IoT device recognition, anomaly detection, and even in uncovering malicious activities. This paper embarks on a comprehensive exploration of the security threats arising from ML's integration into various facets of IoT, spanning various attack types including membership inference, adversarial evasion, reconstruction, property inference, model extraction, and poisoning attacks. Unlike previous studies, our work offers a holistic perspective, categorizing threats based on criteria such as adversary models, attack targets, and key security attributes (confidentiality, availability, and integrity). We delve into the underlying techniques of ML attacks in IoT environment, providing a critical evaluation of their mechanisms and impacts. Furthermore, our research thoroughly assesses 65 libraries, both author-contributed and third-party, evaluating their role in safeguarding model and data privacy. We emphasize the availability and usability of these libraries, aiming to arm the community with the necessary tools to bolster their defenses against the evolving threat landscape. Through our comprehensive review and analysis, this paper seeks to contribute to the ongoing discourse on ML-based IoT security, offering valuable insights and practical solutions to secure ML models and data in the rapidly expanding field of artificial intelligence in IoT.

摘要: 物联网(IoT)的到来带来了一个前所未有的互联时代，预计到2025年底，将有800亿台智能设备投入运营。这些设备促进了大量智能应用，提高了各个领域的生活质量和效率。机器学习(ML)是一项关键技术，不仅用于分析物联网生成的数据，还用于分析物联网生态系统中的各种应用。例如，ML在物联网设备识别、异常检测，甚至在发现恶意活动方面都有用武之地。本文对ML融入物联网的各个方面所带来的安全威胁进行了全面的探讨，包括成员身份推断、对抗性逃避、重构、属性推理、模型提取和中毒攻击等各种攻击类型。与以前的研究不同，我们的工作提供了一个整体的视角，根据对手模型、攻击目标和关键安全属性(机密性、可用性和完整性)等标准对威胁进行分类。我们深入研究了物联网环境下ML攻击的基本技术，并对其机制和影响进行了关键评估。此外，我们的研究全面评估了65个图书馆，包括作者贡献的图书馆和第三方图书馆，评估它们在保护模型和数据隐私方面的作用。我们强调这些库的可用性和可用性，旨在为社区提供必要的工具，以加强他们对不断变化的威胁环境的防御。通过我们的全面回顾和分析，本文试图为正在进行的基于ML的物联网安全讨论做出贡献，为保护物联网快速扩张的人工智能领域中的ML模型和数据提供有价值的见解和实用解决方案。



## **31. HashVFL: Defending Against Data Reconstruction Attacks in Vertical Federated Learning**

HashVFL：垂直联邦学习中的数据重构攻击防御 cs.CR

**SubmitDate**: 2024-01-22    [abs](http://arxiv.org/abs/2212.00325v2) [paper-pdf](http://arxiv.org/pdf/2212.00325v2)

**Authors**: Pengyu Qiu, Xuhong Zhang, Shouling Ji, Chong Fu, Xing Yang, Ting Wang

**Abstract**: Vertical Federated Learning (VFL) is a trending collaborative machine learning model training solution. Existing industrial frameworks employ secure multi-party computation techniques such as homomorphic encryption to ensure data security and privacy. Despite these efforts, studies have revealed that data leakage remains a risk in VFL due to the correlations between intermediate representations and raw data. Neural networks can accurately capture these correlations, allowing an adversary to reconstruct the data. This emphasizes the need for continued research into securing VFL systems.   Our work shows that hashing is a promising solution to counter data reconstruction attacks. The one-way nature of hashing makes it difficult for an adversary to recover data from hash codes. However, implementing hashing in VFL presents new challenges, including vanishing gradients and information loss. To address these issues, we propose HashVFL, which integrates hashing and simultaneously achieves learnability, bit balance, and consistency.   Experimental results indicate that HashVFL effectively maintains task performance while defending against data reconstruction attacks. It also brings additional benefits in reducing the degree of label leakage, mitigating adversarial attacks, and detecting abnormal inputs. We hope our work will inspire further research into the potential applications of HashVFL.

摘要: 垂直联合学习（Vertical Federated Learning，VFL）是一种流行的协作机器学习模型训练解决方案。现有的工业框架采用安全的多方计算技术，如同态加密，以确保数据的安全性和隐私。尽管有这些努力，研究表明，由于中间表示和原始数据之间的相关性，数据泄漏仍然是VFL的风险。神经网络可以准确地捕捉这些相关性，从而允许对手重建数据。这强调了继续研究保护VFL系统的必要性。   我们的工作表明，哈希是一个很有前途的解决方案，以对抗数据重建攻击。散列的单向性使得对手很难从散列代码中恢复数据。然而，在VFL中实现哈希提出了新的挑战，包括梯度消失和信息丢失。为了解决这些问题，我们提出了HashVFL，它集成了哈希，同时实现了可学习性，位平衡和一致性。   实验结果表明，HashVFL在抵御数据重构攻击的同时，有效地保持了任务性能.它还在降低标签泄漏程度、减轻对抗性攻击和检测异常输入方面带来了额外的好处。我们希望我们的工作将激发对HashVFL潜在应用的进一步研究。



## **32. LRS: Enhancing Adversarial Transferability through Lipschitz Regularized Surrogate**

LRS：通过Lipschitz正则化代理提高对手的可转移性 cs.LG

AAAI 2024 main track. Code available on Github (see abstract).  Appendix is included in this updated version

**SubmitDate**: 2024-01-22    [abs](http://arxiv.org/abs/2312.13118v2) [paper-pdf](http://arxiv.org/pdf/2312.13118v2)

**Authors**: Tao Wu, Tie Luo, Donald C. Wunsch

**Abstract**: The transferability of adversarial examples is of central importance to transfer-based black-box adversarial attacks. Previous works for generating transferable adversarial examples focus on attacking \emph{given} pretrained surrogate models while the connections between surrogate models and adversarial trasferability have been overlooked. In this paper, we propose {\em Lipschitz Regularized Surrogate} (LRS) for transfer-based black-box attacks, a novel approach that transforms surrogate models towards favorable adversarial transferability. Using such transformed surrogate models, any existing transfer-based black-box attack can run without any change, yet achieving much better performance. Specifically, we impose Lipschitz regularization on the loss landscape of surrogate models to enable a smoother and more controlled optimization process for generating more transferable adversarial examples. In addition, this paper also sheds light on the connection between the inner properties of surrogate models and adversarial transferability, where three factors are identified: smaller local Lipschitz constant, smoother loss landscape, and stronger adversarial robustness. We evaluate our proposed LRS approach by attacking state-of-the-art standard deep neural networks and defense models. The results demonstrate significant improvement on the attack success rates and transferability. Our code is available at https://github.com/TrustAIoT/LRS.

摘要: 对抗性例子的可转移性对于基于转移的黑盒对抗性攻击是至关重要的。以往关于生成可传递对抗实例的工作主要集中在攻击预先训练好的代理模型上，而忽略了代理模型与对抗传递能力之间的联系。针对基于转移的黑盒攻击，提出了一种将代理模型转化为有利的对抗性转移的新方法--LRS。使用这种转换的代理模型，任何现有的基于传输的黑盒攻击都可以在不做任何更改的情况下运行，但获得了更好的性能。具体地说，我们将Lipschitz正则化应用于代理模型的损失图景，以实现更平滑和更可控的优化过程，从而生成更多可转移的对抗性例子。此外，本文还揭示了代理模型的内在性质与对抗转移之间的关系，其中确定了三个因素：较小的局部Lipschitz常数、更平滑的损失图景和更强的对抗稳健性。我们通过攻击最先进的标准深度神经网络和防御模型来评估我们提出的LRS方法。结果表明，在攻击成功率和可转移性方面都有显著的提高。我们的代码可以在https://github.com/TrustAIoT/LRS.上找到



## **33. Analyzing the Quality Attributes of AI Vision Models in Open Repositories Under Adversarial Attacks**

开放知识库中AI视觉模型在敌意攻击下的质量属性分析 cs.CR

10 pages

**SubmitDate**: 2024-01-22    [abs](http://arxiv.org/abs/2401.12261v1) [paper-pdf](http://arxiv.org/pdf/2401.12261v1)

**Authors**: Zerui Wang, Yan Liu

**Abstract**: As AI models rapidly evolve, they are frequently released to open repositories, such as HuggingFace. It is essential to perform quality assurance validation on these models before integrating them into the production development lifecycle. In addition to evaluating efficiency in terms of balanced accuracy and computing costs, adversarial attacks are potential threats to the robustness and explainability of AI models. Meanwhile, XAI applies algorithms that approximate inputs to outputs post-hoc to identify the contributing features. Adversarial perturbations may also degrade the utility of XAI explanations that require further investigation. In this paper, we present an integrated process designed for downstream evaluation tasks, including validating AI model accuracy, evaluating robustness with benchmark perturbations, comparing explanation utility, and assessing overhead. We demonstrate an evaluation scenario involving six computer vision models, which include CNN-based, Transformer-based, and hybrid architectures, three types of perturbations, and five XAI methods, resulting in ninety unique combinations. The process reveals the explanation utility among the XAI methods in terms of the identified key areas responding to the adversarial perturbation. The process produces aggregated results that illustrate multiple attributes of each AI model.

摘要: 随着AI模型的快速发展，它们经常被发布到开放存储库，如HuggingFace。在将这些模型集成到生产开发生命周期之前，对它们执行质量保证验证是至关重要的。除了在准确性和计算成本方面平衡评估效率外，对抗性攻击还对人工智能模型的健壮性和可解释性构成潜在威胁。同时，XAI将近似输入的算法应用于后期输出，以确定贡献特征。对抗性的干扰也可能降低Xai解释的效用，需要进一步的调查。在本文中，我们提出了一个为下游评估任务设计的完整过程，包括验证人工智能模型的准确性，评估基准扰动下的稳健性，比较解释效用，以及评估开销。我们演示了一个包含六个计算机视觉模型的评估场景，其中包括基于CNN的、基于变形金刚的和混合架构、三种类型的扰动和五种XAI方法，产生了90个独特的组合。这个过程揭示了XAI方法之间的解释效用，就识别的关键区域而言，对对抗性扰动的响应。该过程产生的聚合结果说明了每个AI模型的多个属性。



## **34. Reducing Usefulness of Stolen Credentials in SSO Contexts**

降低被盗凭据在SSO上下文中的有用性 cs.CR

8 pages, 5 figures

**SubmitDate**: 2024-01-21    [abs](http://arxiv.org/abs/2401.11599v1) [paper-pdf](http://arxiv.org/pdf/2401.11599v1)

**Authors**: Sam Hays, Michael Sandborn, Dr. Jules White

**Abstract**: Approximately 61% of cyber attacks involve adversaries in possession of valid credentials. Attackers acquire credentials through various means, including phishing, dark web data drops, password reuse, etc. Multi-factor authentication (MFA) helps to thwart attacks that use valid credentials, but attackers still commonly breach systems by tricking users into accepting MFA step up requests through techniques, such as ``MFA Bombing'', where multiple requests are sent to a user until they accept one. Currently, there are several solutions to this problem, each with varying levels of security and increasing invasiveness on user devices. This paper proposes a token-based enrollment architecture that is less invasive to user devices than mobile device management, but still offers strong protection against use of stolen credentials and MFA attacks.

摘要: 大约61%的网络攻击涉及拥有有效凭据的对手。攻击者通过各种方式获取凭据，包括网络钓鱼、暗网络数据丢弃、密码重复使用等。多因素身份验证(MFA)有助于挫败使用有效凭据的攻击，但攻击者通常仍通过诱使用户接受MFA加速请求等技术来破坏系统，其中多个请求被发送给用户，直到他们接受一个请求。目前，有几种解决方案可以解决这个问题，每种解决方案的安全级别各不相同，对用户设备的侵入性也在不断增加。提出了一种基于令牌的注册架构，与移动设备管理相比，该架构对用户设备的侵入性较小，但仍能提供强大的保护，防止使用被盗凭据和MFA攻击。



## **35. Thundernna: a white box adversarial attack**

Thundernna：白盒对抗性攻击 cs.LG

10 pages, 5 figures

**SubmitDate**: 2024-01-21    [abs](http://arxiv.org/abs/2111.12305v2) [paper-pdf](http://arxiv.org/pdf/2111.12305v2)

**Authors**: Linfeng Ye, Shayan Mohajer Hamidi

**Abstract**: The existing work shows that the neural network trained by naive gradient-based optimization method is prone to adversarial attacks, adds small malicious on the ordinary input is enough to make the neural network wrong. At the same time, the attack against a neural network is the key to improving its robustness. The training against adversarial examples can make neural networks resist some kinds of adversarial attacks. At the same time, the adversarial attack against a neural network can also reveal some characteristics of the neural network, a complex high-dimensional non-linear function, as discussed in previous work.   In This project, we develop a first-order method to attack the neural network. Compare with other first-order attacks, our method has a much higher success rate. Furthermore, it is much faster than second-order attacks and multi-steps first-order attacks.

摘要: 已有的工作表明，基于朴素梯度优化方法训练的神经网络容易受到敌意攻击，在普通输入上添加少量恶意信息就足以使神经网络出错。同时，对神经网络的攻击是提高其稳健性的关键。针对对抗性例子的训练可以使神经网络抵抗某些类型的对抗性攻击。同时，对神经网络的敌意攻击也可以揭示神经网络的一些特征，这是一个复杂的高维非线性函数，如前人所讨论的。在这个项目中，我们开发了一种一阶方法来攻击神经网络。与其他一阶攻击方法相比，该方法具有更高的成功率。此外，它比二阶攻击和多步骤一阶攻击要快得多。



## **36. How Robust Are Energy-Based Models Trained With Equilibrium Propagation?**

基于能量的模型用均衡传播训练的健壮性如何？ cs.LG

**SubmitDate**: 2024-01-21    [abs](http://arxiv.org/abs/2401.11543v1) [paper-pdf](http://arxiv.org/pdf/2401.11543v1)

**Authors**: Siddharth Mansingh, Michal Kucer, Garrett Kenyon, Juston Moore, Michael Teti

**Abstract**: Deep neural networks (DNNs) are easily fooled by adversarial perturbations that are imperceptible to humans. Adversarial training, a process where adversarial examples are added to the training set, is the current state-of-the-art defense against adversarial attacks, but it lowers the model's accuracy on clean inputs, is computationally expensive, and offers less robustness to natural noise. In contrast, energy-based models (EBMs), which were designed for efficient implementation in neuromorphic hardware and physical systems, incorporate feedback connections from each layer to the previous layer, yielding a recurrent, deep-attractor architecture which we hypothesize should make them naturally robust. Our work is the first to explore the robustness of EBMs to both natural corruptions and adversarial attacks, which we do using the CIFAR-10 and CIFAR-100 datasets. We demonstrate that EBMs are more robust than transformers and display comparable robustness to adversarially-trained DNNs on gradient-based (white-box) attacks, query-based (black-box) attacks, and natural perturbations without sacrificing clean accuracy, and without the need for adversarial training or additional training techniques.

摘要: 深度神经网络（DNN）很容易被人类无法感知的对抗性扰动所欺骗。对抗性训练是将对抗性示例添加到训练集中的过程，是当前最先进的对抗性攻击的防御方法，但它降低了模型在干净输入上的准确性，计算成本高，并且对自然噪声的鲁棒性较低。相比之下，基于能量的模型（EBM），这是为了在神经形态硬件和物理系统中有效实现而设计的，它包含了从每一层到前一层的反馈连接，产生了一个经常性的深吸引子架构，我们假设这应该使它们自然健壮。我们的工作是第一个探索EBM对自然腐败和对抗性攻击的鲁棒性的工作，我们使用CIFAR-10和CIFAR-100数据集进行了研究。我们证明了EBM比transformer更强大，并且在基于梯度的（白盒）攻击，基于查询的（黑盒）攻击和自然扰动上显示出与对抗训练的DNN相当的鲁棒性，而不会牺牲干净的准确性，并且不需要对抗训练或额外的训练技术。



## **37. Finding a Needle in the Adversarial Haystack: A Targeted Paraphrasing Approach For Uncovering Edge Cases with Minimal Distribution Distortion**

在对抗性的干草堆中找针：一种最小分布失真的边缘案例发现的有针对性的释义方法 cs.CL

EACL 2024 - Main conference

**SubmitDate**: 2024-01-21    [abs](http://arxiv.org/abs/2401.11373v1) [paper-pdf](http://arxiv.org/pdf/2401.11373v1)

**Authors**: Aly M. Kassem, Sherif Saad

**Abstract**: Adversarial attacks against NLP Deep Learning models are a significant concern. In particular, adversarial samples exploit the model's sensitivity to small input changes. While these changes appear insignificant on the semantics of the input sample, they result in significant decay in model performance. In this paper, we propose Targeted Paraphrasing via RL (TPRL), an approach to automatically learn a policy to generate challenging samples that most likely improve the model's performance. TPRL leverages FLAN T5, a language model, as a generator and employs a self learned policy using a proximal policy gradient to generate the adversarial examples automatically. TPRL's reward is based on the confusion induced in the classifier, preserving the original text meaning through a Mutual Implication score. We demonstrate and evaluate TPRL's effectiveness in discovering natural adversarial attacks and improving model performance through extensive experiments on four diverse NLP classification tasks via Automatic and Human evaluation. TPRL outperforms strong baselines, exhibits generalizability across classifiers and datasets, and combines the strengths of language modeling and reinforcement learning to generate diverse and influential adversarial examples.

摘要: 针对NLP深度学习模型的对抗性攻击是一个值得关注的问题。特别是，对抗性样本利用了模型对微小输入变化的敏感性。虽然这些变化在输入样本的语义上看起来微不足道，但它们会导致模型性能的显著下降。在本文中，我们提出了通过RL（TPRL）的目标释义，这是一种自动学习策略以生成最有可能提高模型性能的具有挑战性的样本的方法。TPRL利用语言模型FLAN T5作为生成器，并采用使用邻近策略梯度的自学习策略来自动生成对抗性示例。TPRL的奖励是基于分类器中引起的混淆，通过相互含义得分来保留原始文本的含义。我们通过自动和人工评估对四种不同的NLP分类任务进行了广泛的实验，展示和评估了TPRL在发现自然对抗攻击和提高模型性能方面的有效性。TPRL优于强大的基线，在分类器和数据集之间表现出泛化能力，并结合了语言建模和强化学习的优势，以生成多样化和有影响力的对抗性示例。



## **38. Robustness Against Adversarial Attacks via Learning Confined Adversarial Polytopes**

基于受限对抗性多面体学习的抗敌意攻击能力 cs.LG

The paper has been accepted in ICASSP 2024

**SubmitDate**: 2024-01-20    [abs](http://arxiv.org/abs/2401.07991v2) [paper-pdf](http://arxiv.org/pdf/2401.07991v2)

**Authors**: Shayan Mohajer Hamidi, Linfeng Ye

**Abstract**: Deep neural networks (DNNs) could be deceived by generating human-imperceptible perturbations of clean samples. Therefore, enhancing the robustness of DNNs against adversarial attacks is a crucial task. In this paper, we aim to train robust DNNs by limiting the set of outputs reachable via a norm-bounded perturbation added to a clean sample. We refer to this set as adversarial polytope, and each clean sample has a respective adversarial polytope. Indeed, if the respective polytopes for all the samples are compact such that they do not intersect the decision boundaries of the DNN, then the DNN is robust against adversarial samples. Hence, the inner-working of our algorithm is based on learning \textbf{c}onfined \textbf{a}dversarial \textbf{p}olytopes (CAP). By conducting a thorough set of experiments, we demonstrate the effectiveness of CAP over existing adversarial robustness methods in improving the robustness of models against state-of-the-art attacks including AutoAttack.

摘要: 深度神经网络(DNN)可以通过产生人类无法察觉的干净样本的扰动来欺骗。因此，提高DNN对敌意攻击的健壮性是一项至关重要的任务。在本文中，我们的目标是通过限制通过添加到干净样本的范数有界扰动可到达的输出集来训练鲁棒的DNN。我们将这个集合称为对抗性多面体，每个干净的样本都有各自的对抗性多面体。事实上，如果所有样本的相应多面体是紧凑的，使得它们不与DNN的决策边界相交，则DNN对对抗性样本是健壮的。因此，我们的算法的内部工作是基于学习文本bf{c}受限的文本bf{a}分叉算法(CAP)。通过一组详细的实验，我们证明了CAP在提高模型对包括AutoAttack在内的最新攻击的稳健性方面优于现有的对抗性稳健性方法。



## **39. Jailbreaking GPT-4V via Self-Adversarial Attacks with System Prompts**

通过系统攻击的自对抗攻击越狱GPT-4V cs.CR

**SubmitDate**: 2024-01-20    [abs](http://arxiv.org/abs/2311.09127v2) [paper-pdf](http://arxiv.org/pdf/2311.09127v2)

**Authors**: Yuanwei Wu, Xiang Li, Yixin Liu, Pan Zhou, Lichao Sun

**Abstract**: Existing work on jailbreak Multimodal Large Language Models (MLLMs) has focused primarily on adversarial examples in model inputs, with less attention to vulnerabilities, especially in model API. To fill the research gap, we carry out the following work: 1) We discover a system prompt leakage vulnerability in GPT-4V. Through carefully designed dialogue, we successfully extract the internal system prompts of GPT-4V. This finding indicates potential exploitable security risks in MLLMs; 2) Based on the acquired system prompts, we propose a novel MLLM jailbreaking attack method termed SASP (Self-Adversarial Attack via System Prompt). By employing GPT-4 as a red teaming tool against itself, we aim to search for potential jailbreak prompts leveraging stolen system prompts. Furthermore, in pursuit of better performance, we also add human modification based on GPT-4's analysis, which further improves the attack success rate to 98.7\%; 3) We evaluated the effect of modifying system prompts to defend against jailbreaking attacks. Results show that appropriately designed system prompts can significantly reduce jailbreak success rates. Overall, our work provides new insights into enhancing MLLM security, demonstrating the important role of system prompts in jailbreaking. This finding could be leveraged to greatly facilitate jailbreak success rates while also holding the potential for defending against jailbreaks.

摘要: 现有关于越狱多模式大型语言模型(MLLMS)的工作主要集中在模型输入中的对抗性示例，对漏洞的关注较少，特别是在模型API中。为了填补这一研究空白，我们开展了以下工作：1)在GPT-4V中发现了一个系统即时泄漏漏洞。通过精心设计的对话，我们成功地提取了GPT-4V的内部系统提示。2)基于获得的系统提示，提出了一种新的基于系统提示的MLLM越狱攻击方法SASP(Self-Aversarial Attack by System Prompt)。通过使用GPT-4作为针对自己的红色团队工具，我们的目标是利用被盗的系统提示来搜索潜在的越狱提示。此外，为了追求更好的性能，我们还在GPT-4的S分析的基础上增加了人工修改，进一步将攻击成功率提高到98.7%。3)评估了修改系统提示对越狱攻击的防御效果。结果表明，设计适当的系统提示可以显著降低越狱成功率。总体而言，我们的工作为加强MLLM安全提供了新的见解，展示了系统提示在越狱中的重要作用。这一发现可以被用来极大地提高越狱成功率，同时还具有防御越狱的潜力。



## **40. Susceptibility of Adversarial Attack on Medical Image Segmentation Models**

医学图像分割模型的对抗性攻击敏感性 eess.IV

6 pages, 8 figures, presented at 2023 IEEE 20th International  Symposium on Biomedical Imaging (ISBI) conference

**SubmitDate**: 2024-01-20    [abs](http://arxiv.org/abs/2401.11224v1) [paper-pdf](http://arxiv.org/pdf/2401.11224v1)

**Authors**: Zhongxuan Wang, Leo Xu

**Abstract**: The nature of deep neural networks has given rise to a variety of attacks, but little work has been done to address the effect of adversarial attacks on segmentation models trained on MRI datasets. In light of the grave consequences that such attacks could cause, we explore four models from the U-Net family and examine their responses to the Fast Gradient Sign Method (FGSM) attack. We conduct FGSM attacks on each of them and experiment with various schemes to conduct the attacks. In this paper, we find that medical imaging segmentation models are indeed vulnerable to adversarial attacks and that there is a negligible correlation between parameter size and adversarial attack success. Furthermore, we show that using a different loss function than the one used for training yields higher adversarial attack success, contrary to what the FGSM authors suggested. In future efforts, we will conduct the experiments detailed in this paper with more segmentation models and different attacks. We will also attempt to find ways to counteract the attacks by using model ensembles or special data augmentations. Our code is available at https://github.com/ZhongxuanWang/adv_attk

摘要: 深度神经网络的性质导致了各种各样的攻击，但对于对抗性攻击对基于MRI数据集训练的分割模型的影响，人们所做的工作很少。鉴于此类攻击可能造成的严重后果，我们探索了U-Net家族的四个模型，并检查了它们对快速梯度符号方法(FGSM)攻击的响应。我们对他们中的每一个进行FGSM攻击，并试验各种攻击方案。在本文中，我们发现医学图像分割模型确实容易受到对抗性攻击，并且参数大小与对抗性攻击的成功与否之间的相关性可以忽略不计。此外，我们还表明，与FGSM作者的建议相反，使用与训练中使用的损失函数不同的损失函数会产生更高的对抗性攻击成功率。在未来的努力中，我们将使用更多的分割模型和不同的攻击来进行本文详细介绍的实验。我们还将尝试通过使用模型集成或特殊数据增强来找到对抗攻击的方法。我们的代码可以在https://github.com/ZhongxuanWang/adv_attk上找到



## **41. Generalizing Speaker Verification for Spoof Awareness in the Embedding Space**

嵌入空间中基于欺骗感知的说话人确认泛化 cs.CR

To appear in IEEE/ACM Transactions on Audio, Speech, and Language  Processing

**SubmitDate**: 2024-01-20    [abs](http://arxiv.org/abs/2401.11156v1) [paper-pdf](http://arxiv.org/pdf/2401.11156v1)

**Authors**: Xuechen Liu, Md Sahidullah, Kong Aik Lee, Tomi Kinnunen

**Abstract**: It is now well-known that automatic speaker verification (ASV) systems can be spoofed using various types of adversaries. The usual approach to counteract ASV systems against such attacks is to develop a separate spoofing countermeasure (CM) module to classify speech input either as a bonafide, or a spoofed utterance. Nevertheless, such a design requires additional computation and utilization efforts at the authentication stage. An alternative strategy involves a single monolithic ASV system designed to handle both zero-effort imposter (non-targets) and spoofing attacks. Such spoof-aware ASV systems have the potential to provide stronger protections and more economic computations. To this end, we propose to generalize the standalone ASV (G-SASV) against spoofing attacks, where we leverage limited training data from CM to enhance a simple backend in the embedding space, without the involvement of a separate CM module during the test (authentication) phase. We propose a novel yet simple backend classifier based on deep neural networks and conduct the study via domain adaptation and multi-task integration of spoof embeddings at the training stage. Experiments are conducted on the ASVspoof 2019 logical access dataset, where we improve the performance of statistical ASV backends on the joint (bonafide and spoofed) and spoofed conditions by a maximum of 36.2% and 49.8% in terms of equal error rates, respectively.

摘要: 现在众所周知，自动说话人验证(ASV)系统可以使用各种类型的对手进行欺骗。对抗ASV系统抵御此类攻击的通常方法是开发单独的欺骗对策(CM)模块，以将语音输入分类为真正的或欺骗的话语。然而，这样的设计在身份验证阶段需要额外的计算和利用工作。另一种策略包括一个单一的单片ASV系统，旨在同时处理零努力冒名顶替者(非目标)和欺骗攻击。这种感知欺骗的ASV系统有可能提供更强大的保护和更多的经济计算。为此，我们建议推广抗欺骗攻击的独立ASV(G-SASV)，其中我们利用来自CM的有限训练数据来增强嵌入空间中的简单后端，而不需要在测试(身份验证)阶段涉及单独的CM模块。我们提出了一种新颖而简单的基于深度神经网络的后端分类器，并在训练阶段通过域自适应和欺骗嵌入的多任务集成进行了研究。在ASVspoof 2019逻辑访问数据集上进行了实验，在相同错误率的情况下，我们将联合(真实和欺骗)和欺骗条件下的统计ASV后端的性能分别提高了36.2%和49.8%。



## **42. CARE: Ensemble Adversarial Robustness Evaluation Against Adaptive Attackers for Security Applications**

CARE：针对自适应攻击者的安全应用集成攻击健壮性评估 cs.CR

**SubmitDate**: 2024-01-20    [abs](http://arxiv.org/abs/2401.11126v1) [paper-pdf](http://arxiv.org/pdf/2401.11126v1)

**Authors**: Hangsheng Zhang, Jiqiang Liu, Jinsong Dong

**Abstract**: Ensemble defenses, are widely employed in various security-related applications to enhance model performance and robustness. The widespread adoption of these techniques also raises many questions: Are general ensembles defenses guaranteed to be more robust than individuals? Will stronger adaptive attacks defeat existing ensemble defense strategies as the cybersecurity arms race progresses? Can ensemble defenses achieve adversarial robustness to different types of attacks simultaneously and resist the continually adjusted adaptive attacks? Unfortunately, these critical questions remain unresolved as there are no platforms for comprehensive evaluation of ensemble adversarial attacks and defenses in the cybersecurity domain. In this paper, we propose a general Cybersecurity Adversarial Robustness Evaluation (CARE) platform aiming to bridge this gap.

摘要: 包围防御被广泛应用于各种与安全相关的应用中，以提高模型的性能和鲁棒性。这些技术的广泛采用也提出了许多问题：一般的合奏防御保证比个人更强大吗？随着网络安全军备竞赛的进展，更强的自适应攻击是否会击败现有的整体防御战略？集成防御能否同时实现对不同类型攻击的鲁棒性，并抵抗不断调整的自适应攻击？不幸的是，这些关键问题仍然没有得到解决，因为在网络安全领域没有全面评估集合对抗攻击和防御的平台。在本文中，我们提出了一个通用的网络安全对抗性鲁棒性评估（CARE）平台，旨在弥合这一差距。



## **43. Universal Backdoor Attacks**

通用后门攻击 cs.LG

Accepted for publication at ICLR 2024

**SubmitDate**: 2024-01-20    [abs](http://arxiv.org/abs/2312.00157v2) [paper-pdf](http://arxiv.org/pdf/2312.00157v2)

**Authors**: Benjamin Schneider, Nils Lukas, Florian Kerschbaum

**Abstract**: Web-scraped datasets are vulnerable to data poisoning, which can be used for backdooring deep image classifiers during training. Since training on large datasets is expensive, a model is trained once and re-used many times. Unlike adversarial examples, backdoor attacks often target specific classes rather than any class learned by the model. One might expect that targeting many classes through a naive composition of attacks vastly increases the number of poison samples. We show this is not necessarily true and more efficient, universal data poisoning attacks exist that allow controlling misclassifications from any source class into any target class with a small increase in poison samples. Our idea is to generate triggers with salient characteristics that the model can learn. The triggers we craft exploit a phenomenon we call inter-class poison transferability, where learning a trigger from one class makes the model more vulnerable to learning triggers for other classes. We demonstrate the effectiveness and robustness of our universal backdoor attacks by controlling models with up to 6,000 classes while poisoning only 0.15% of the training dataset. Our source code is available at https://github.com/Ben-Schneider-code/Universal-Backdoor-Attacks.

摘要: 网络抓取的数据集很容易受到数据中毒的影响，在训练过程中，数据中毒可以用于回溯深度图像分类器。由于在大型数据集上进行训练的成本很高，因此一个模型只需训练一次，就可以多次重复使用。与对抗性示例不同，后门攻击通常针对特定类，而不是模型学习到的任何类。人们可能会认为，通过天真的攻击组合以许多类别为目标会极大地增加毒物样本的数量。我们证明这不一定是真的，而且更有效，普遍存在的数据中毒攻击允许在毒物样本少量增加的情况下控制从任何源类到任何目标类的误分类。我们的想法是生成模型可以学习的具有显著特征的触发器。我们制作的触发器利用了一种我们称为类间毒药可转移性的现象，即从一个类学习触发器使模型更容易学习其他类的触发器。我们通过控制多达6,000个类的模型来展示我们的通用后门攻击的有效性和健壮性，而只毒化了0.15%的训练数据集。我们的源代码可以在https://github.com/Ben-Schneider-code/Universal-Backdoor-Attacks.上找到



## **44. Ensembler: Combating model inversion attacks using model ensemble during collaborative inference**

集成器：在协作推理过程中使用模型集成对抗模型反转攻击 cs.CR

in submission

**SubmitDate**: 2024-01-19    [abs](http://arxiv.org/abs/2401.10859v1) [paper-pdf](http://arxiv.org/pdf/2401.10859v1)

**Authors**: Dancheng Liu, Jinjun Xiong

**Abstract**: Deep learning models have exhibited remarkable performance across various domains. Nevertheless, the burgeoning model sizes compel edge devices to offload a significant portion of the inference process to the cloud. While this practice offers numerous advantages, it also raises critical concerns regarding user data privacy. In scenarios where the cloud server's trustworthiness is in question, the need for a practical and adaptable method to safeguard data privacy becomes imperative. In this paper, we introduce Ensembler, an extensible framework designed to substantially increase the difficulty of conducting model inversion attacks for adversarial parties. Ensembler leverages model ensembling on the adversarial server, running in parallel with existing approaches that introduce perturbations to sensitive data during colloborative inference. Our experiments demonstrate that when combined with even basic Gaussian noise, Ensembler can effectively shield images from reconstruction attacks, achieving recognition levels that fall below human performance in some strict settings, significantly outperforming baseline methods lacking the Ensembler framework.

摘要: 深度学习模型在各个领域都表现出了卓越的性能。然而，不断增长的模型大小迫使边缘设备将推理过程的很大一部分卸载到云端。虽然这种做法提供了许多优点，但它也引起了对用户数据隐私的严重担忧。在云服务器的可信度受到质疑的情况下，需要一种实用且适应性强的方法来保护数据隐私。在本文中，我们介绍了Ensembler，这是一个可扩展的框架，旨在大幅增加对抗方进行模型反转攻击的难度。Ensembler利用对抗服务器上的模型集成，与现有方法并行运行，这些方法在协同推理期间对敏感数据引入扰动。我们的实验表明，当与基本的高斯噪声相结合时，Ensembler可以有效地保护图像免受重建攻击，在一些严格的设置中实现低于人类性能的识别水平，显着优于缺乏Ensembler框架的基线方法。



## **45. Privacy-Preserving Neural Graph Databases**

保护隐私的神经图库 cs.DB

**SubmitDate**: 2024-01-19    [abs](http://arxiv.org/abs/2312.15591v2) [paper-pdf](http://arxiv.org/pdf/2312.15591v2)

**Authors**: Qi Hu, Haoran Li, Jiaxin Bai, Yangqiu Song

**Abstract**: In the era of big data and rapidly evolving information systems, efficient and accurate data retrieval has become increasingly crucial. Neural graph databases (NGDBs) have emerged as a powerful paradigm that combines the strengths of graph databases (graph DBs) and neural networks to enable efficient storage, retrieval, and analysis of graph-structured data. The usage of neural embedding storage and complex neural logical query answering provides NGDBs with generalization ability. When the graph is incomplete, by extracting latent patterns and representations, neural graph databases can fill gaps in the graph structure, revealing hidden relationships and enabling accurate query answering. Nevertheless, this capability comes with inherent trade-offs, as it introduces additional privacy risks to the database. Malicious attackers can infer more sensitive information in the database using well-designed combinatorial queries, such as by comparing the answer sets of where Turing Award winners born before 1950 and after 1940 lived, the living places of Turing Award winner Hinton are probably exposed, although the living places may have been deleted in the training due to the privacy concerns. In this work, inspired by the privacy protection in graph embeddings, we propose a privacy-preserving neural graph database (P-NGDB) to alleviate the risks of privacy leakage in NGDBs. We introduce adversarial training techniques in the training stage to force the NGDBs to generate indistinguishable answers when queried with private information, enhancing the difficulty of inferring sensitive information through combinations of multiple innocuous queries. Extensive experiment results on three datasets show that P-NGDB can effectively protect private information in the graph database while delivering high-quality public answers responses to queries.

摘要: 在大数据和快速发展的信息系统的时代，高效和准确的数据检索变得越来越重要。神经图形数据库(NGDB)已经成为一种强大的范例，它结合了图形数据库(图形数据库)和神经网络的优点，使得能够有效地存储、检索和分析图形结构的数据。神经嵌入存储和复杂神经逻辑查询回答的使用为NGDB提供了泛化能力。当图不完整时，通过提取潜在模式和表示，神经图库可以填补图结构中的空白，揭示隐藏的关系，并使查询得到准确的回答。尽管如此，这种能力也伴随着固有的权衡，因为它会给数据库带来额外的隐私风险。恶意攻击者可以使用精心设计的组合查询来推断数据库中更敏感的信息，例如通过比较1950年之前出生的图灵奖获得者和1940年后出生的图灵奖获得者的答案集，图灵奖获得者辛顿的居住地可能会被曝光，尽管出于隐私考虑，在训练中可能已经删除了居住地。在这项工作中，我们受到图嵌入中隐私保护的启发，提出了一种隐私保护神经图库(P-NGDB)来缓解NGDB中隐私泄露的风险。我们在训练阶段引入对抗性训练技术，迫使NGDB在查询私有信息时产生难以区分的答案，增加了通过组合多个无害查询来推断敏感信息的难度。在三个数据集上的大量实验结果表明，P-NGDB可以有效地保护图形数据库中的私有信息，同时提供高质量的公共查询响应。



## **46. Explainable and Transferable Adversarial Attack for ML-Based Network Intrusion Detectors**

基于ML的网络入侵检测的可解释可转移敌意攻击 cs.CR

**SubmitDate**: 2024-01-19    [abs](http://arxiv.org/abs/2401.10691v1) [paper-pdf](http://arxiv.org/pdf/2401.10691v1)

**Authors**: Hangsheng Zhang, Dongqi Han, Yinlong Liu, Zhiliang Wang, Jiyan Sun, Shangyuan Zhuang, Jiqiang Liu, Jinsong Dong

**Abstract**: espite being widely used in network intrusion detection systems (NIDSs), machine learning (ML) has proven to be highly vulnerable to adversarial attacks. White-box and black-box adversarial attacks of NIDS have been explored in several studies. However, white-box attacks unrealistically assume that the attackers have full knowledge of the target NIDSs. Meanwhile, existing black-box attacks can not achieve high attack success rate due to the weak adversarial transferability between models (e.g., neural networks and tree models). Additionally, neither of them explains why adversarial examples exist and why they can transfer across models. To address these challenges, this paper introduces ETA, an Explainable Transfer-based Black-Box Adversarial Attack framework. ETA aims to achieve two primary objectives: 1) create transferable adversarial examples applicable to various ML models and 2) provide insights into the existence of adversarial examples and their transferability within NIDSs. Specifically, we first provide a general transfer-based adversarial attack method applicable across the entire ML space. Following that, we exploit a unique insight based on cooperative game theory and perturbation interpretations to explain adversarial examples and adversarial transferability. On this basis, we propose an Important-Sensitive Feature Selection (ISFS) method to guide the search for adversarial examples, achieving stronger transferability and ensuring traffic-space constraints.

摘要: 尽管机器学习在网络入侵检测系统中得到了广泛的应用，但它被证明是非常容易受到对手攻击的。网络入侵检测系统的白盒和黑盒对抗性攻击已经在多个研究中得到了探索。然而，白盒攻击不切实际地假设攻击者完全了解目标NIDS。同时，现有的黑盒攻击由于模型(如神经网络和树模型)之间的对抗性较弱而不能达到较高的攻击成功率。此外，它们都没有解释为什么存在对抗性例子，以及为什么它们可以在模型之间转移。为了应对这些挑战，本文引入了一种可解释的基于传输的黑盒对抗攻击框架ETA。ETA旨在实现两个主要目标：1)创建适用于各种ML模型的可转移的对抗性例子；2)提供对对抗性例子的存在及其在新入侵检测系统中的可转移性的见解。具体地说，我们首先提供了一种适用于整个ML空间的通用的基于转移的对抗性攻击方法。然后，我们利用基于合作博弈论和扰动解释的独特见解来解释对抗性例子和对抗性转移。在此基础上，提出了一种重要敏感的特征选择(ISFS)方法来指导对抗性实例的搜索，实现了较强的可转移性，并保证了交通空间的约束。



## **47. FIMBA: Evaluating the Robustness of AI in Genomics via Feature Importance Adversarial Attacks**

FIMBA：通过特征重要性对抗攻击评估基因组学中AI的鲁棒性 cs.LG

15 pages, core code available at:  https://github.com/HeorhiiS/fimba-attack

**SubmitDate**: 2024-01-19    [abs](http://arxiv.org/abs/2401.10657v1) [paper-pdf](http://arxiv.org/pdf/2401.10657v1)

**Authors**: Heorhii Skovorodnikov, Hoda Alkhzaimi

**Abstract**: With the steady rise of the use of AI in bio-technical applications and the widespread adoption of genomics sequencing, an increasing amount of AI-based algorithms and tools is entering the research and production stage affecting critical decision-making streams like drug discovery and clinical outcomes. This paper demonstrates the vulnerability of AI models often utilized downstream tasks on recognized public genomics datasets. We undermine model robustness by deploying an attack that focuses on input transformation while mimicking the real data and confusing the model decision-making, ultimately yielding a pronounced deterioration in model performance. Further, we enhance our approach by generating poisoned data using a variational autoencoder-based model. Our empirical findings unequivocally demonstrate a decline in model performance, underscored by diminished accuracy and an upswing in false positives and false negatives. Furthermore, we analyze the resulting adversarial samples via spectral analysis yielding conclusions for countermeasures against such attacks.

摘要: 随着人工智能在生物技术应用中的稳步上升和基因组测序的广泛采用，越来越多的基于人工智能的算法和工具正在进入研究和生产阶段，影响着药物发现和临床结果等关键决策流。本文论证了在公认的公共基因组数据集上经常利用下游任务的人工智能模型的脆弱性。我们通过部署一种专注于输入转换的攻击来破坏模型的健壮性，同时模仿真实数据并混淆模型决策，最终导致模型性能的显著恶化。此外，我们通过使用基于变分自动编码器的模型来生成有毒数据来增强我们的方法。我们的经验发现明确地证明了模型性能的下降，强调了准确性的降低和假阳性和假阴性的上升。此外，我们通过频谱分析对生成的敌意样本进行分析，得出针对此类攻击的对策结论。



## **48. Adversarially Robust Signed Graph Contrastive Learning from Balance Augmentation**

基于平衡增强的对偶稳健符号图对比学习 cs.LG

**SubmitDate**: 2024-01-19    [abs](http://arxiv.org/abs/2401.10590v1) [paper-pdf](http://arxiv.org/pdf/2401.10590v1)

**Authors**: Jialong Zhou, Xing Ai, Yuni Lai, Kai Zhou

**Abstract**: Signed graphs consist of edges and signs, which can be separated into structural information and balance-related information, respectively. Existing signed graph neural networks (SGNNs) typically rely on balance-related information to generate embeddings. Nevertheless, the emergence of recent adversarial attacks has had a detrimental impact on the balance-related information. Similar to how structure learning can restore unsigned graphs, balance learning can be applied to signed graphs by improving the balance degree of the poisoned graph. However, this approach encounters the challenge "Irreversibility of Balance-related Information" - while the balance degree improves, the restored edges may not be the ones originally affected by attacks, resulting in poor defense effectiveness. To address this challenge, we propose a robust SGNN framework called Balance Augmented-Signed Graph Contrastive Learning (BA-SGCL), which combines Graph Contrastive Learning principles with balance augmentation techniques. Experimental results demonstrate that BA-SGCL not only enhances robustness against existing adversarial attacks but also achieves superior performance on link sign prediction task across various datasets.

摘要: 符号图由边和符号组成，它们可以分别分解为结构信息和平衡相关信息。现有的符号图神经网络（SGNN）通常依赖于平衡相关信息来生成嵌入。然而，最近出现的对抗性攻击对与平衡有关的信息产生了不利影响。类似于结构学习如何恢复无符号图，平衡学习可以通过提高中毒图的平衡度来应用于有符号图。然而，这种方法遇到了“平衡相关信息的不可逆性”的挑战-在平衡度提高的同时，恢复的边缘可能不是最初受攻击影响的边缘，导致防御效果不佳。为了应对这一挑战，我们提出了一个强大的SGNN框架，称为平衡增强签名图对比学习（BA-SGCL），它将图对比学习原理与平衡增强技术相结合。实验结果表明，BA-SGCL不仅增强了对现有对抗性攻击的鲁棒性，而且在各种数据集上的链接符号预测任务上也取得了优异的性能。



## **49. PuriDefense: Randomized Local Implicit Adversarial Purification for Defending Black-box Query-based Attacks**

PuriDefense：用于防御基于黑盒查询的攻击的随机局部隐式对抗性净化 cs.CR

**SubmitDate**: 2024-01-19    [abs](http://arxiv.org/abs/2401.10586v1) [paper-pdf](http://arxiv.org/pdf/2401.10586v1)

**Authors**: Ping Guo, Zhiyuan Yang, Xi Lin, Qingchuan Zhao, Qingfu Zhang

**Abstract**: Black-box query-based attacks constitute significant threats to Machine Learning as a Service (MLaaS) systems since they can generate adversarial examples without accessing the target model's architecture and parameters. Traditional defense mechanisms, such as adversarial training, gradient masking, and input transformations, either impose substantial computational costs or compromise the test accuracy of non-adversarial inputs. To address these challenges, we propose an efficient defense mechanism, PuriDefense, that employs random patch-wise purifications with an ensemble of lightweight purification models at a low level of inference cost. These models leverage the local implicit function and rebuild the natural image manifold. Our theoretical analysis suggests that this approach slows down the convergence of query-based attacks by incorporating randomness into purifications. Extensive experiments on CIFAR-10 and ImageNet validate the effectiveness of our proposed purifier-based defense mechanism, demonstrating significant improvements in robustness against query-based attacks.

摘要: 基于黑盒查询的攻击对机器学习即服务(MLaaS)系统构成了重大威胁，因为它们可以在不访问目标模型的体系结构和参数的情况下生成对抗性示例。传统的防御机制，如对抗性训练、梯度掩蔽和输入转换，要么增加了大量的计算成本，要么损害了非对抗性输入的测试精度。为了应对这些挑战，我们提出了一个有效的防御机制PuriDefense，它使用了随机的补丁式净化，并以较低的推理成本集成了轻量级的净化模型。这些模型利用局部隐函数重建自然图像流形。我们的理论分析表明，这种方法通过将随机性纳入净化中，减缓了基于查询的攻击的收敛。在CIFAR-10和ImageNet上的大量实验验证了我们提出的基于净化器的防御机制的有效性，显示出对基于查询的攻击的健壮性显著提高。



## **50. Hijacking Attacks against Neural Networks by Analyzing Training Data**

基于训练数据分析的神经网络劫持攻击 cs.CR

Full version with major polishing, compared to the Usenix Security  2024 edition

**SubmitDate**: 2024-01-19    [abs](http://arxiv.org/abs/2401.09740v2) [paper-pdf](http://arxiv.org/pdf/2401.09740v2)

**Authors**: Yunjie Ge, Qian Wang, Huayang Huang, Qi Li, Cong Wang, Chao Shen, Lingchen Zhao, Peipei Jiang, Zheng Fang, Shenyi Zhang

**Abstract**: Backdoors and adversarial examples are the two primary threats currently faced by deep neural networks (DNNs). Both attacks attempt to hijack the model behaviors with unintended outputs by introducing (small) perturbations to the inputs. Backdoor attacks, despite the high success rates, often require a strong assumption, which is not always easy to achieve in reality. Adversarial example attacks, which put relatively weaker assumptions on attackers, often demand high computational resources, yet do not always yield satisfactory success rates when attacking mainstream black-box models in the real world. These limitations motivate the following research question: can model hijacking be achieved more simply, with a higher attack success rate and more reasonable assumptions? In this paper, we propose CleanSheet, a new model hijacking attack that obtains the high performance of backdoor attacks without requiring the adversary to tamper with the model training process. CleanSheet exploits vulnerabilities in DNNs stemming from the training data. Specifically, our key idea is to treat part of the clean training data of the target model as "poisoned data," and capture the characteristics of these data that are more sensitive to the model (typically called robust features) to construct "triggers." These triggers can be added to any input example to mislead the target model, similar to backdoor attacks. We validate the effectiveness of CleanSheet through extensive experiments on 5 datasets, 79 normally trained models, 68 pruned models, and 39 defensive models. Results show that CleanSheet exhibits performance comparable to state-of-the-art backdoor attacks, achieving an average attack success rate (ASR) of 97.5% on CIFAR-100 and 92.4% on GTSRB, respectively. Furthermore, CleanSheet consistently maintains a high ASR, when confronted with various mainstream backdoor defenses.

摘要: 后门和敌意例子是深度神经网络(DNN)目前面临的两个主要威胁。这两种攻击都试图通过向输入引入(小)扰动来劫持具有非预期输出的模型行为。后门攻击尽管成功率很高，但往往需要强有力的假设，而这在现实中并不总是容易实现的。对抗性例子攻击对攻击者的假设相对较弱，通常需要很高的计算资源，但在攻击现实世界中的主流黑盒模型时，并不总是产生令人满意的成功率。这些局限性引发了以下研究问题：能否以更高的攻击成功率和更合理的假设更简单地实现模型劫持？在本文中，我们提出了一种新的劫持攻击模型CleanSheet，它在不要求对手篡改模型训练过程的情况下获得了后门攻击的高性能。CleanSheet利用源自训练数据的DNN中的漏洞。具体地说，我们的关键思想是将目标模型的部分干净训练数据视为“有毒数据”，并捕获这些数据中对模型更敏感的特征(通常称为稳健特征)来构建“触发器”。这些触发器可以添加到任何输入示例中，以误导目标模型，类似于后门攻击。我们通过在5个数据集、79个正常训练模型、68个剪枝模型和39个防御模型上的大量实验验证了Clear Sheet的有效性。结果表明，CleanSheet的攻击性能与最先进的后门攻击相当，在CIFAR-100和GTSRB上的平均攻击成功率分别达到97.5%和92.4%。此外，当面对各种主流的后门防御时，CleanSheet始终保持着较高的ASR。



