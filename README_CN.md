# Latest Adversarial Attack Papers
**update at 2021-12-27 23:31:12**

翻译来自 https://cloud.tencent.com/document/product/551/15619

## **1. Adaptive Modeling Against Adversarial Attacks**

抗敌意攻击的自适应建模 cs.LG

10 pages, 3 figures

**SubmitDate**: 2021-12-23    [paper-pdf](http://arxiv.org/pdf/2112.12431v1)

**Authors**: Zhiwen Yan, Teck Khim Ng

**Abstracts**: Adversarial training, the process of training a deep learning model with adversarial data, is one of the most successful adversarial defense methods for deep learning models. We have found that the robustness to white-box attack of an adversarially trained model can be further improved if we fine tune this model in inference stage to adapt to the adversarial input, with the extra information in it. We introduce an algorithm that "post trains" the model at inference stage between the original output class and a "neighbor" class, with existing training data. The accuracy of pre-trained Fast-FGSM CIFAR10 classifier base model against white-box projected gradient attack (PGD) can be significantly improved from 46.8% to 64.5% with our algorithm.

摘要: 对抗性训练是利用对抗性数据训练深度学习模型的过程，是深度学习模型中最成功的对抗性防御方法之一。我们发现，如果在推理阶段对一个对抗性训练模型进行微调，使其适应对抗性输入，并加入额外的信息，可以进一步提高该模型对白盒攻击的鲁棒性。我们介绍了一种算法，该算法利用现有的训练数据，在推理阶段在原始输出类和“相邻”类之间对模型进行“后训练”。该算法可以将预先训练的Fast-FGSM CIFAR10分类器基模型对抗白盒投影梯度攻击(PGD)的准确率从46.8%提高到64.5%。



## **2. Revisiting and Advancing Fast Adversarial Training Through The Lens of Bi-Level Optimization**

用双层优化镜头重温和推进快速对抗性训练 cs.LG

**SubmitDate**: 2021-12-23    [paper-pdf](http://arxiv.org/pdf/2112.12376v1)

**Authors**: Yihua Zhang, Guanhuan Zhang, Prashant Khanduri, Mingyi Hong, Shiyu Chang, Sijia Liu

**Abstracts**: Adversarial training (AT) has become a widely recognized defense mechanism to improve the robustness of deep neural networks against adversarial attacks. It solves a min-max optimization problem, where the minimizer (i.e., defender) seeks a robust model to minimize the worst-case training loss in the presence of adversarial examples crafted by the maximizer (i.e., attacker). However, the min-max nature makes AT computationally intensive and thus difficult to scale. Meanwhile, the FAST-AT algorithm, and in fact many recent algorithms that improve AT, simplify the min-max based AT by replacing its maximization step with the simple one-shot gradient sign based attack generation step. Although easy to implement, FAST-AT lacks theoretical guarantees, and its practical performance can be unsatisfactory, suffering from the robustness catastrophic overfitting when training with strong adversaries.   In this paper, we propose to design FAST-AT from the perspective of bi-level optimization (BLO). We first make the key observation that the most commonly-used algorithmic specification of FAST-AT is equivalent to using some gradient descent-type algorithm to solve a bi-level problem involving a sign operation. However, the discrete nature of the sign operation makes it difficult to understand the algorithm performance. Based on the above observation, we propose a new tractable bi-level optimization problem, design and analyze a new set of algorithms termed Fast Bi-level AT (FAST-BAT). FAST-BAT is capable of defending sign-based projected gradient descent (PGD) attacks without calling any gradient sign method and explicit robust regularization. Furthermore, we empirically show that our method outperforms state-of-the-art FAST-AT baselines, by achieving superior model robustness without inducing robustness catastrophic overfitting, or suffering from any loss of standard accuracy.

摘要: 对抗训练(AT)已成为一种被广泛认可的防御机制，以提高深层神经网络对抗对手攻击的鲁棒性。它解决了一个最小-最大优化问题，其中最小化器(即防御者)在存在最大化者(即攻击者)制作的对抗性示例的情况下寻求一个鲁棒模型来最小化最坏情况下的训练损失。然而，最小-最大特性使得AT计算密集，因此很难扩展。同时，FAST-AT算法，以及最近许多改进AT的算法，通过用简单的基于一次梯度符号的攻击生成步骤代替其最大化步骤，简化了基于最小-最大的AT。FAST-AT虽然易于实现，但缺乏理论保证，实际应用效果不理想，在与强对手进行训练时存在健壮性和灾难性过拟合问题。本文提出从双层优化(BLO)的角度设计FAST-AT。我们首先观察到FAST-AT最常用的算法规范等价于使用某种梯度下降型算法来解决涉及符号运算的双层问题。然而，符号运算的离散性使得很难理解算法的性能。基于上述观察，我们提出了一个新的易于处理的双层优化问题，设计并分析了一套新的算法，称为快速双层AT(FAST-BAT)。FAST-BAT能够抵抗基于符号的投影梯度下降(PGD)攻击，无需调用任何梯度符号方法和显式鲁棒正则化。此外，我们的经验表明，我们的方法优于最先进的FAST-AT基线，因为它实现了卓越的模型稳健性，而不会导致稳健性灾难性过拟合，也不会损失任何标准精度。



## **3. Adversarial Attacks against Windows PE Malware Detection: A Survey of the State-of-the-Art**

针对Windows PE恶意软件检测的对抗性攻击：现状综述 cs.CR

**SubmitDate**: 2021-12-23    [paper-pdf](http://arxiv.org/pdf/2112.12310v1)

**Authors**: Xiang Ling, Lingfei Wu, Jiangyu Zhang, Zhenqing Qu, Wei Deng, Xiang Chen, Chunming Wu, Shouling Ji, Tianyue Luo, Jingzheng Wu, Yanjun Wu

**Abstracts**: The malware has been being one of the most damaging threats to computers that span across multiple operating systems and various file formats. To defend against the ever-increasing and ever-evolving threats of malware, tremendous efforts have been made to propose a variety of malware detection methods that attempt to effectively and efficiently detect malware. Recent studies have shown that, on the one hand, existing ML and DL enable the superior detection of newly emerging and previously unseen malware. However, on the other hand, ML and DL models are inherently vulnerable to adversarial attacks in the form of adversarial examples, which are maliciously generated by slightly and carefully perturbing the legitimate inputs to confuse the targeted models. Basically, adversarial attacks are initially extensively studied in the domain of computer vision, and some quickly expanded to other domains, including NLP, speech recognition and even malware detection. In this paper, we focus on malware with the file format of portable executable (PE) in the family of Windows operating systems, namely Windows PE malware, as a representative case to study the adversarial attack methods in such adversarial settings. To be specific, we start by first outlining the general learning framework of Windows PE malware detection based on ML/DL and subsequently highlighting three unique challenges of performing adversarial attacks in the context of PE malware. We then conduct a comprehensive and systematic review to categorize the state-of-the-art adversarial attacks against PE malware detection, as well as corresponding defenses to increase the robustness of PE malware detection. We conclude the paper by first presenting other related attacks against Windows PE malware detection beyond the adversarial attacks and then shedding light on future research directions and opportunities.

摘要: 该恶意软件一直是对跨越多个操作系统和各种文件格式的计算机的最具破坏性的威胁之一。为了防御不断增加和不断演变的恶意软件威胁，人们做出了巨大的努力来提出各种恶意软件检测方法，这些方法试图有效和高效地检测恶意软件。最近的研究表明，一方面，现有的ML和DL能够更好地检测新出现的和以前未见过的恶意软件。然而，另一方面，ML和DL模型天生就容易受到对抗性示例形式的对抗性攻击，这些攻击是通过稍微和仔细地扰动合法输入来混淆目标模型而恶意生成的。基本上，敌意攻击最初在计算机视觉领域得到了广泛的研究，一些攻击很快扩展到其他领域，包括NLP、语音识别甚至恶意软件检测。本文以Windows操作系统家族中具有可移植可执行文件(PE)文件格式的恶意软件，即Windows PE恶意软件为典型案例，研究这种敌意环境下的敌意攻击方法。具体地说，我们首先概述了基于ML/DL的Windows PE恶意软件检测的一般学习框架，然后重点介绍了在PE恶意软件环境中执行敌意攻击的三个独特挑战。然后对针对PE恶意软件检测的对抗性攻击进行了全面系统的分类，并提出了相应的防御措施，以提高PE恶意软件检测的健壮性。最后，我们首先介绍了Windows PE恶意软件检测除了对抗性攻击之外的其他相关攻击，并阐明了未来的研究方向和机遇。



## **4. Detect & Reject for Transferability of Black-box Adversarial Attacks Against Network Intrusion Detection Systems**

网络入侵检测系统黑盒敌意攻击的可传递性检测与拒绝 cs.CR

**SubmitDate**: 2021-12-22    [paper-pdf](http://arxiv.org/pdf/2112.12095v1)

**Authors**: Islam Debicha, Thibault Debatty, Jean-Michel Dricot, Wim Mees, Tayeb Kenaza

**Abstracts**: In the last decade, the use of Machine Learning techniques in anomaly-based intrusion detection systems has seen much success. However, recent studies have shown that Machine learning in general and deep learning specifically are vulnerable to adversarial attacks where the attacker attempts to fool models by supplying deceptive input. Research in computer vision, where this vulnerability was first discovered, has shown that adversarial images designed to fool a specific model can deceive other machine learning models. In this paper, we investigate the transferability of adversarial network traffic against multiple machine learning-based intrusion detection systems. Furthermore, we analyze the robustness of the ensemble intrusion detection system, which is notorious for its better accuracy compared to a single model, against the transferability of adversarial attacks. Finally, we examine Detect & Reject as a defensive mechanism to limit the effect of the transferability property of adversarial network traffic against machine learning-based intrusion detection systems.

摘要: 在过去的十年中，机器学习技术在基于异常的入侵检测系统中的应用取得了很大的成功。然而，最近的研究表明，一般的机器学习和深度学习特别容易受到对手攻击，攻击者试图通过提供欺骗性输入来愚弄模型。计算机视觉领域的研究表明，旨在欺骗特定模型的对抗性图像也可以欺骗其他机器学习模型。计算机视觉是这个漏洞最早被发现的地方。本文针对基于多机器学习的入侵检测系统，研究了敌意网络流量的可转移性。此外，我们还分析了集成入侵检测系统在对抗攻击的可转移性方面的鲁棒性，该集成入侵检测系统以其比单一模型更高的准确性而臭名昭著。最后，我们将检测和拒绝作为一种防御机制来限制敌意网络流量的可传递性对基于机器学习的入侵检测系统的影响。



## **5. Evaluating the Robustness of Deep Reinforcement Learning for Autonomous and Adversarial Policies in a Multi-agent Urban Driving Environment**

多智能体城市驾驶环境下自主对抗性策略的深度强化学习鲁棒性评估 cs.AI

**SubmitDate**: 2021-12-22    [paper-pdf](http://arxiv.org/pdf/2112.11947v1)

**Authors**: Aizaz Sharif, Dusica Marijan

**Abstracts**: Deep reinforcement learning is actively used for training autonomous driving agents in a vision-based urban simulated environment. Due to the large availability of various reinforcement learning algorithms, we are still unsure of which one works better while training autonomous cars in single-agent as well as multi-agent driving environments. A comparison of deep reinforcement learning in vision-based autonomous driving will open up the possibilities for training better autonomous car policies. Also, autonomous cars trained on deep reinforcement learning-based algorithms are known for being vulnerable to adversarial attacks, and we have less information on which algorithms would act as a good adversarial agent. In this work, we provide a systematic evaluation and comparative analysis of 6 deep reinforcement learning algorithms for autonomous and adversarial driving in four-way intersection scenario. Specifically, we first train autonomous cars using state-of-the-art deep reinforcement learning algorithms. Second, we test driving capabilities of the trained autonomous policies in single-agent as well as multi-agent scenarios. Lastly, we use the same deep reinforcement learning algorithms to train adversarial driving agents, in order to test the driving performance of autonomous cars and look for possible collision and offroad driving scenarios. We perform experiments by using vision-only high fidelity urban driving simulated environments.

摘要: 在基于视觉的城市模拟环境中，深度强化学习被广泛用于训练自主驾驶智能体。由于各种强化学习算法的可用性很大，在单Agent和多Agent驾驶环境下训练自动驾驶汽车时，我们仍然不确定哪种算法效果更好。深度强化学习在基于视觉的自动驾驶中的比较将为训练更好的自动驾驶政策提供可能性。此外，按照基于深度强化学习的算法训练的自动驾驶汽车也因易受对手攻击而闻名，我们对哪些算法会充当好的对手代理的信息较少。在这项工作中，我们对四向交叉路口场景中自主驾驶和对抗性驾驶的6种深度强化学习算法进行了系统的评估和比较分析。具体地说，我们首先使用最先进的深度强化学习算法来训练自动驾驶汽车。其次，我们测试了训练好的自主策略在单Agent和多Agent场景中的驱动能力。最后，我们使用相同的深度强化学习算法来训练对抗性驾驶Agent，以测试自动驾驶汽车的驾驶性能，并寻找可能的碰撞和越野驾驶场景。我们使用视觉高保真的城市驾驶模拟环境进行了实验。



## **6. Adversarial Deep Reinforcement Learning for Trustworthy Autonomous Driving Policies**

基于对抗性深度强化学习的可信自主驾驶策略 cs.AI

**SubmitDate**: 2021-12-22    [paper-pdf](http://arxiv.org/pdf/2112.11937v1)

**Authors**: Aizaz Sharif, Dusica Marijan

**Abstracts**: Deep reinforcement learning is widely used to train autonomous cars in a simulated environment. Still, autonomous cars are well known for being vulnerable when exposed to adversarial attacks. This raises the question of whether we can train the adversary as a driving agent for finding failure scenarios in autonomous cars, and then retrain autonomous cars with new adversarial inputs to improve their robustness. In this work, we first train and compare adversarial car policy on two custom reward functions to test the driving control decision of autonomous cars in a multi-agent setting. Second, we verify that adversarial examples can be used not only for finding unwanted autonomous driving behavior, but also for helping autonomous driving cars in improving their deep reinforcement learning policies. By using a high fidelity urban driving simulation environment and vision-based driving agents, we demonstrate that the autonomous cars retrained using the adversary player noticeably increase the performance of their driving policies in terms of reducing collision and offroad steering errors.

摘要: 深度强化学习被广泛用于在模拟环境中训练自动驾驶汽车。尽管如此，自动驾驶汽车在受到对手攻击时很容易受到攻击，这是众所周知的。这就提出了一个问题，我们是否可以将对手训练成发现自动驾驶汽车故障场景的驾驶代理，然后用新的对手输入重新训练自动驾驶汽车，以提高它们的稳健性。在这项工作中，我们首先训练并比较了两个自定义奖励函数上的对抗性汽车策略，以测试自动驾驶汽车在多智能体环境下的驾驶控制决策。其次，我们验证了对抗性例子不仅可以用来发现不想要的自动驾驶行为，而且还可以帮助自动驾驶汽车改进其深度强化学习策略。通过使用高保真的城市驾驶模拟环境和基于视觉的驾驶代理，我们证明了使用对手玩家进行再培训的自动驾驶汽车在减少碰撞和越野转向错误方面显著提高了驾驶策略的性能。



## **7. Consistency Regularization for Adversarial Robustness**

用于对抗鲁棒性的一致性正则化 cs.LG

Published as a conference proceeding for AAAI 2022

**SubmitDate**: 2021-12-22    [paper-pdf](http://arxiv.org/pdf/2103.04623v3)

**Authors**: Jihoon Tack, Sihyun Yu, Jongheon Jeong, Minseon Kim, Sung Ju Hwang, Jinwoo Shin

**Abstracts**: Adversarial training (AT) is currently one of the most successful methods to obtain the adversarial robustness of deep neural networks. However, the phenomenon of robust overfitting, i.e., the robustness starts to decrease significantly during AT, has been problematic, not only making practitioners consider a bag of tricks for a successful training, e.g., early stopping, but also incurring a significant generalization gap in the robustness. In this paper, we propose an effective regularization technique that prevents robust overfitting by optimizing an auxiliary `consistency' regularization loss during AT. Specifically, we discover that data augmentation is a quite effective tool to mitigate the overfitting in AT, and develop a regularization that forces the predictive distributions after attacking from two different augmentations of the same instance to be similar with each other. Our experimental results demonstrate that such a simple regularization technique brings significant improvements in the test robust accuracy of a wide range of AT methods. More remarkably, we also show that our method could significantly help the model to generalize its robustness against unseen adversaries, e.g., other types or larger perturbations compared to those used during training. Code is available at https://github.com/alinlab/consistency-adversarial.

摘要: 对抗性训练(AT)是目前获得深层神经网络对抗性鲁棒性最成功的方法之一。然而，鲁棒过拟合现象(即鲁棒性在AT过程中开始显著下降)一直是有问题的，不仅使实践者为成功的训练考虑了一大堆技巧，例如提前停止，而且导致了鲁棒性方面的显著泛化差距。在本文中，我们提出了一种有效的正则化技术，通过优化AT过程中的辅助“一致性”正则化损失来防止鲁棒过拟合。具体地说，我们发现数据增广是缓解AT中过拟合的一种非常有效的工具，并发展了一种正则化方法，强制从同一实例的两个不同增广攻击后的预测分布彼此相似。我们的实验结果表明，这种简单的正则化技术大大提高了各种AT方法的测试鲁棒精度。更值得注意的是，我们的方法还可以显著地帮助模型推广其对看不见的对手的鲁棒性，例如，与训练期间使用的相比，其他类型或更大的扰动。代码可在https://github.com/alinlab/consistency-adversarial.上获得



## **8. How Should Pre-Trained Language Models Be Fine-Tuned Towards Adversarial Robustness?**

预先训练的语言模型应该如何针对对手的健壮性进行微调？ cs.CL

Accepted by NeurIPS-2021

**SubmitDate**: 2021-12-22    [paper-pdf](http://arxiv.org/pdf/2112.11668v1)

**Authors**: Xinhsuai Dong, Luu Anh Tuan, Min Lin, Shuicheng Yan, Hanwang Zhang

**Abstracts**: The fine-tuning of pre-trained language models has a great success in many NLP fields. Yet, it is strikingly vulnerable to adversarial examples, e.g., word substitution attacks using only synonyms can easily fool a BERT-based sentiment analysis model. In this paper, we demonstrate that adversarial training, the prevalent defense technique, does not directly fit a conventional fine-tuning scenario, because it suffers severely from catastrophic forgetting: failing to retain the generic and robust linguistic features that have already been captured by the pre-trained model. In this light, we propose Robust Informative Fine-Tuning (RIFT), a novel adversarial fine-tuning method from an information-theoretical perspective. In particular, RIFT encourages an objective model to retain the features learned from the pre-trained model throughout the entire fine-tuning process, whereas a conventional one only uses the pre-trained weights for initialization. Experimental results show that RIFT consistently outperforms the state-of-the-arts on two popular NLP tasks: sentiment analysis and natural language inference, under different attacks across various pre-trained language models.

摘要: 对预先训练好的语言模型进行微调在许多自然语言处理领域都取得了巨大的成功。然而，它非常容易受到敌意例子的攻击，例如，仅使用同义词的单词替换攻击很容易欺骗基于BERT的情感分析模型。在本文中，我们证明了对抗性训练这一流行的防御技术并不直接适合传统的微调场景，因为它存在严重的灾难性遗忘：未能保留预先训练的模型已经捕获的通用和健壮的语言特征。鉴于此，我们从信息论的角度提出了一种新的对抗性微调方法&鲁棒信息微调(RIFT)。特别地，RIFT鼓励客观模型在整个微调过程中保留从预先训练的模型中学习的特征，而传统的RIFT只使用预先训练的权重进行初始化。实验结果表明，在不同的预训练语言模型的不同攻击下，RIFT在情感分析和自然语言推理这两个流行的自然语言处理任务上的性能始终优于最新的NLP任务。



## **9. An Attention Score Based Attacker for Black-box NLP Classifier**

一种基于注意力得分的黑盒NLP分类器攻击者 cs.LG

**SubmitDate**: 2021-12-22    [paper-pdf](http://arxiv.org/pdf/2112.11660v1)

**Authors**: Yueyang Liu, Hunmin Lee, Zhipeng Cai

**Abstracts**: Deep neural networks have a wide range of applications in solving various real-world tasks and have achieved satisfactory results, in domains such as computer vision, image classification, and natural language processing. Meanwhile, the security and robustness of neural networks have become imperative, as diverse researches have shown the vulnerable aspects of neural networks. Case in point, in Natural language processing tasks, the neural network may be fooled by an attentively modified text, which has a high similarity to the original one. As per previous research, most of the studies are focused on the image domain; Different from image adversarial attacks, the text is represented in a discrete sequence, traditional image attack methods are not applicable in the NLP field. In this paper, we propose a word-level NLP sentiment classifier attack model, which includes a self-attention mechanism-based word selection method and a greedy search algorithm for word substitution. We experiment with our attack model by attacking GRU and 1D-CNN victim models on IMDB datasets. Experimental results demonstrate that our model achieves a higher attack success rate and more efficient than previous methods due to the efficient word selection algorithms are employed and minimized the word substitute number. Also, our model is transferable, which can be used in the image domain with several modifications.

摘要: 深度神经网络在计算机视觉、图像分类、自然语言处理等领域有着广泛的应用，并取得了令人满意的结果。同时，随着各种研究表明神经网络的脆弱性，神经网络的安全性和鲁棒性也变得势在必行。例如，在自然语言处理任务中，神经网络可能会被精心修改的文本所欺骗，因为它与原始文本具有很高的相似性。根据以往的研究，大多数研究都集中在图像领域；与图像对抗性攻击不同，文本是离散序列表示的，传统的图像攻击方法不适用于自然语言处理领域。本文提出了一种词级NLP情感分类器攻击模型，该模型包括一种基于自我注意机制的选词方法和一种贪婪的词替换搜索算法。我们在IMDB数据集上通过攻击GRU和1D-CNN受害者模型来测试我们的攻击模型。实验结果表明，由于采用了高效的选词算法并最小化了替身的词数，该模型取得了比以往方法更高的攻击成功率和更高的效率。此外，我们的模型是可移植的，可以在图像域中使用，只需做一些修改即可。



## **10. Collaborative adversary nodes learning on the logs of IoT devices in an IoT network**

协作敌方节点学习物联网网络中物联网设备的日志 cs.CR

**SubmitDate**: 2021-12-22    [paper-pdf](http://arxiv.org/pdf/2112.12546v1)

**Authors**: Sandhya Aneja, Melanie Ang Xuan En, Nagender Aneja

**Abstracts**: Artificial Intelligence (AI) development has encouraged many new research areas, including AI-enabled Internet of Things (IoT) network. AI analytics and intelligent paradigms greatly improve learning efficiency and accuracy. Applying these learning paradigms to network scenarios provide technical advantages of new networking solutions. In this paper, we propose an improved approach for IoT security from data perspective. The network traffic of IoT devices can be analyzed using AI techniques. The Adversary Learning (AdLIoTLog) model is proposed using Recurrent Neural Network (RNN) with attention mechanism on sequences of network events in the network traffic. We define network events as a sequence of the time series packets of protocols captured in the log. We have considered different packets TCP packets, UDP packets, and HTTP packets in the network log to make the algorithm robust. The distributed IoT devices can collaborate to cripple our world which is extending to Internet of Intelligence. The time series packets are converted into structured data by removing noise and adding timestamps. The resulting data set is trained by RNN and can detect the node pairs collaborating with each other. We used the BLEU score to evaluate the model performance. Our results show that the predicting performance of the AdLIoTLog model trained by our method degrades by 3-4% in the presence of attack in comparison to the scenario when the network is not under attack. AdLIoTLog can detect adversaries because when adversaries are present the model gets duped by the collaborative events and therefore predicts the next event with a biased event rather than a benign event. We conclude that AI can provision ubiquitous learning for the new generation of Internet of Things.

摘要: 人工智能(AI)的发展鼓励了许多新的研究领域，包括支持AI的物联网(IoT)网络。人工智能分析和智能范例极大地提高了学习效率和准确性。将这些学习范例应用于网络场景可提供新网络解决方案的技术优势。本文从数据的角度提出了一种改进的物联网安全方法。物联网设备的网络流量可以使用人工智能技术进行分析。利用具有注意机制的递归神经网络(RNN)对网络流量中的网络事件序列进行学习，提出了AdLIoTLog(AdLIoTLog)模型。我们将网络事件定义为日志中捕获的协议的时间序列数据包的序列。我们在网络日志中考虑了不同的数据包TCP数据包、UDP数据包和HTTP数据包，以增强算法的健壮性。分布式物联网设备可以相互协作，使我们的世界陷入瘫痪，这个世界正在向智能互联网延伸。通过去除噪声和添加时间戳将时间序列分组转换为结构化数据。生成的数据集由RNN进行训练，可以检测出相互协作的节点对。我们使用BLEU评分来评价模型的性能。实验结果表明，该方法训练的AdLIoTLog模型在存在攻击的情况下，预测性能比网络未受到攻击时的预测性能下降了3%~4%。AdLIoTLog可以检测到对手，因为当对手存在时，模型会被协作事件所欺骗，因此会用有偏差的事件而不是良性事件来预测下一个事件。我们的结论是，人工智能可以为新一代物联网提供泛在学习。



## **11. Reevaluating Adversarial Examples in Natural Language**

重新评价自然语言中的对抗性实例 cs.CL

15 pages; 9 Tables; 5 Figures

**SubmitDate**: 2021-12-21    [paper-pdf](http://arxiv.org/pdf/2004.14174v3)

**Authors**: John X. Morris, Eli Lifland, Jack Lanchantin, Yangfeng Ji, Yanjun Qi

**Abstracts**: State-of-the-art attacks on NLP models lack a shared definition of a what constitutes a successful attack. We distill ideas from past work into a unified framework: a successful natural language adversarial example is a perturbation that fools the model and follows some linguistic constraints. We then analyze the outputs of two state-of-the-art synonym substitution attacks. We find that their perturbations often do not preserve semantics, and 38% introduce grammatical errors. Human surveys reveal that to successfully preserve semantics, we need to significantly increase the minimum cosine similarities between the embeddings of swapped words and between the sentence encodings of original and perturbed sentences.With constraints adjusted to better preserve semantics and grammaticality, the attack success rate drops by over 70 percentage points.

摘要: 针对NLP模型的最先进的攻击缺乏一个共同的定义，即什么构成了成功的攻击。我们从过去的工作中提取想法到一个统一的框架中：一个成功的自然语言对抗性例子是一种欺骗模型并遵循一些语言限制的扰动。然后，我们分析了两种最先进的同义词替换攻击的输出。我们发现，他们的扰动往往不能保持语义，38%的人引入了语法错误。人类调查显示，要成功地保持语义，需要显著提高互换单词的嵌入之间以及原句和扰动句的句子编码之间的最小余弦相似度，通过调整约束以更好地保持语义和语法，攻击成功率下降了70个百分点以上。



## **12. MIA-Former: Efficient and Robust Vision Transformers via Multi-grained Input-Adaptation**

基于多粒度输入自适应的高效鲁棒视觉转换器 cs.CV

**SubmitDate**: 2021-12-21    [paper-pdf](http://arxiv.org/pdf/2112.11542v1)

**Authors**: Zhongzhi Yu, Yonggan Fu, Sicheng Li, Chaojian Li, Yingyan Lin

**Abstracts**: ViTs are often too computationally expensive to be fitted onto real-world resource-constrained devices, due to (1) their quadratically increased complexity with the number of input tokens and (2) their overparameterized self-attention heads and model depth. In parallel, different images are of varied complexity and their different regions can contain various levels of visual information, indicating that treating all regions/tokens equally in terms of model complexity is unnecessary while such opportunities for trimming down ViTs' complexity have not been fully explored. To this end, we propose a Multi-grained Input-adaptive Vision Transformer framework dubbed MIA-Former that can input-adaptively adjust the structure of ViTs at three coarse-to-fine-grained granularities (i.e., model depth and the number of model heads/tokens). In particular, our MIA-Former adopts a low-cost network trained with a hybrid supervised and reinforcement training method to skip unnecessary layers, heads, and tokens in an input adaptive manner, reducing the overall computational cost. Furthermore, an interesting side effect of our MIA-Former is that its resulting ViTs are naturally equipped with improved robustness against adversarial attacks over their static counterparts, because MIA-Former's multi-grained dynamic control improves the model diversity similar to the effect of ensemble and thus increases the difficulty of adversarial attacks against all its sub-models. Extensive experiments and ablation studies validate that the proposed MIA-Former framework can effectively allocate computation budgets adaptive to the difficulty of input images meanwhile increase robustness, achieving state-of-the-art (SOTA) accuracy-efficiency trade-offs, e.g., 20% computation savings with the same or even a higher accuracy compared with SOTA dynamic transformer models.

摘要: VIT的计算成本往往太高，无法安装到现实世界的资源受限设备上，这是因为(1)它们的复杂度随着输入令牌的数量呈二次曲线增加，(2)它们的过度参数化的自我关注头部和模型深度。同时，不同的图像具有不同的复杂度，它们的不同区域可以包含不同级别的视觉信息，这表明就模型复杂度而言，平等对待所有区域/标记是不必要的，而这种降低VITS复杂度的机会还没有得到充分探索。为此，我们提出了一种多粒度输入自适应视觉转换器框架MIA-formor，该框架可以从粗粒度到细粒度(即模型深度和模型头/令牌的数量)对VITS的结构进行输入自适应调整。具体地说，我们的MIA-FORM采用低成本网络，采用混合监督和强化训练方法，以输入自适应的方式跳过不必要的层、头和标记，降低了整体计算成本。此外，我们的MIA-前者的一个有趣的副作用是，与它们的静电同行相比，它得到的VIT自然具有更好的抗对手攻击的鲁棒性，因为MIA-前者的多粒度动态控制提高了类似于集成效果的模型多样性，从而增加了对其所有子模型的敌意攻击的难度。大量的实验和烧蚀研究表明，提出的MIA-PERFER框架可以有效地分配与输入图像难度相适应的计算预算，同时增加鲁棒性，实现了最新的SOTA精度和效率折衷，例如，在与SOTA动态变压器模型相同甚至更高的情况下，节省了20%的计算量。



## **13. Improving Robustness with Image Filtering**

利用图像滤波提高鲁棒性 cs.CV

**SubmitDate**: 2021-12-21    [paper-pdf](http://arxiv.org/pdf/2112.11235v1)

**Authors**: Matteo Terzi, Mattia Carletti, Gian Antonio Susto

**Abstracts**: Adversarial robustness is one of the most challenging problems in Deep Learning and Computer Vision research. All the state-of-the-art techniques require a time-consuming procedure that creates cleverly perturbed images. Due to its cost, many solutions have been proposed to avoid Adversarial Training. However, all these attempts proved ineffective as the attacker manages to exploit spurious correlations among pixels to trigger brittle features implicitly learned by the model. This paper first introduces a new image filtering scheme called Image-Graph Extractor (IGE) that extracts the fundamental nodes of an image and their connections through a graph structure. By leveraging the IGE representation, we build a new defense method, Filtering As a Defense, that does not allow the attacker to entangle pixels to create malicious patterns. Moreover, we show that data augmentation with filtered images effectively improves the model's robustness to data corruption. We validate our techniques on CIFAR-10, CIFAR-100, and ImageNet.

摘要: 对抗鲁棒性是深度学习和计算机视觉研究中最具挑战性的问题之一。所有最先进的技术都需要一个耗时的过程来创建巧妙的扰动图像。由于成本的原因，很多人提出了避免对抗性训练的解决方案。然而，所有这些尝试都被证明是无效的，因为攻击者设法利用像素之间的虚假相关性来触发模型隐含地学习的脆弱特征。本文首先介绍了一种新的图像滤波方案&图像-图提取器(Image-Graph Extractor，IGE)，它通过图的结构来提取图像的基本节点和它们之间的联系。通过利用IGE表示，我们构建了一种新的防御方法，即过滤作为防御，它不允许攻击者纠缠像素来创建恶意模式。此外，我们还证明了利用滤波图像进行数据增强有效地提高了模型对数据损坏的鲁棒性。我们在CIFAR-10、CIFAR-100和ImageNet上验证了我们的技术。



## **14. Adversarial images for the primate brain**

灵长类动物大脑的对抗性图像 q-bio.NC

These results reveal limits of CNN-based models of primate vision  through their differential response to adversarial attack, and provide clues  for building better models of the brain and more robust computer vision  algorithms

**SubmitDate**: 2021-12-21    [paper-pdf](http://arxiv.org/pdf/2011.05623v2)

**Authors**: Li Yuan, Will Xiao, Gabriel Kreiman, Francis E. H. Tay, Jiashi Feng, Margaret S. Livingstone

**Abstracts**: Convolutional neural networks (CNNs) are vulnerable to adversarial attack, the phenomenon that adding minuscule noise to an image can fool CNNs into misclassifying it. Because this noise is nearly imperceptible to human viewers, biological vision is assumed to be robust to adversarial attack. Despite this apparent difference in robustness, CNNs are currently the best models of biological vision, revealing a gap in explaining how the brain responds to adversarial images. Indeed, sensitivity to adversarial attack has not been measured for biological vision under normal conditions, nor have attack methods been specifically designed to affect biological vision. We studied the effects of adversarial attack on primate vision, measuring both monkey neuronal responses and human behavior. Adversarial images were created by modifying images from one category(such as human faces) to look like a target category(such as monkey faces), while limiting pixel value change. We tested three attack directions via several attack methods, including directly using CNN adversarial images and using a CNN-based predictive model to guide monkey visual neuron responses. We considered a wide range of image change magnitudes that covered attack success rates up to>90%. We found that adversarial images designed for CNNs were ineffective in attacking primate vision. Even when considering the best attack method, primate vision was more robust to adversarial attack than an ensemble of CNNs, requiring over 100-fold larger image change to attack successfully. The success of individual attack methods and images was correlated between monkey neurons and human behavior, but was less correlated between either and CNN categorization. Consistently, CNN-based models of neurons, when trained on natural images, did not generalize to explain neuronal responses to adversarial images.

摘要: 卷积神经网络(CNNs)容易受到敌意攻击，即在图像中添加微小的噪声可以欺骗CNN对其进行错误分类。由于这种噪声对于人类观众几乎是不可察觉的，因此假设生物视觉对敌方攻击是健壮的。尽管在稳健性方面有明显的差异，但CNN目前是生物视觉的最佳模型，揭示了在解释大脑如何对敌对图像做出反应方面的差距。事实上，在正常情况下没有测量生物视觉对对抗性攻击的敏感性，也没有专门设计攻击方法来影响生物视觉。我们研究了对抗性攻击对灵长类动物视觉的影响，测量了猴子的神经元反应和人类行为。敌意图像是通过修改某一类别(如人脸)中的图像以使其看起来像目标类别(如猴子脸)来创建的，同时限制像素值的变化。我们通过几种攻击方法测试了三种攻击方向，包括直接使用CNN对抗性图像和使用基于CNN的预测模型来指导猴子的视觉神经元反应。我们考虑了大范围的图像变化幅度，涵盖攻击成功率高达90%以上。我们发现，为CNN设计的对抗性图像在攻击灵长类视觉方面是无效的。即使在考虑最好的攻击方法时，灵长类动物的视觉对对手攻击的鲁棒性也比CNN集合更强，需要100倍以上的图像变化才能成功攻击。个体攻击方法和图像的成功与否与猴子神经元和人类行为相关，但与CNN分类之间的相关性较小。始终如一的是，基于CNN的神经元模型，当在自然图像上训练时，不能概括地解释神经元对对抗性图像的反应。



## **15. Denoised Internal Models: a Brain-Inspired Autoencoder against Adversarial Attacks**

去噪内部模型：一种抗敌意攻击的脑启发自动编码器 cs.CV

16 pages, 3 figures

**SubmitDate**: 2021-12-21    [paper-pdf](http://arxiv.org/pdf/2111.10844v2)

**Authors**: Kaiyuan Liu, Xingyu Li, Yurui Lai, Ge Zhang, Hang Su, Jiachen Wang, Chunxu Guo, Jisong Guan, Yi Zhou

**Abstracts**: Despite its great success, deep learning severely suffers from robustness; that is, deep neural networks are very vulnerable to adversarial attacks, even the simplest ones. Inspired by recent advances in brain science, we propose the Denoised Internal Models (DIM), a novel generative autoencoder-based model to tackle this challenge. Simulating the pipeline in the human brain for visual signal processing, DIM adopts a two-stage approach. In the first stage, DIM uses a denoiser to reduce the noise and the dimensions of inputs, reflecting the information pre-processing in the thalamus. Inspired from the sparse coding of memory-related traces in the primary visual cortex, the second stage produces a set of internal models, one for each category. We evaluate DIM over 42 adversarial attacks, showing that DIM effectively defenses against all the attacks and outperforms the SOTA on the overall robustness.

摘要: 尽管深度学习取得了巨大的成功，但它的健壮性严重不足；也就是说，深度神经网络非常容易受到对手的攻击，即使是最简单的攻击。受脑科学最新进展的启发，我们提出了去噪内部模型(DIM)，这是一种新颖的基于生成式自动编码器的模型，以应对撞击的这一挑战。模拟人脑中视觉信号处理的管道，DIM采用了两个阶段的方法。在第一阶段，DIM使用去噪器来降低输入的噪声和维数，反映了丘脑的信息预处理。第二阶段的灵感来自于初级视觉皮层中与记忆相关的痕迹的稀疏编码，第二阶段产生了一组内部模型，每个类别一个。我们对DIM42个对抗性攻击进行了评估，结果表明，DIM有效地防御了所有攻击，并且在整体鲁棒性上优于SOTA。



## **16. A Theoretical View of Linear Backpropagation and Its Convergence**

线性反向传播及其收敛性的理论观点 cs.LG

**SubmitDate**: 2021-12-21    [paper-pdf](http://arxiv.org/pdf/2112.11018v1)

**Authors**: Ziang Li, Yiwen Guo, Haodi Liu, Changshui Zhang

**Abstracts**: Backpropagation is widely used for calculating gradients in deep neural networks (DNNs). Applied often along with stochastic gradient descent (SGD) or its variants, backpropagation is considered as a de-facto choice in a variety of machine learning tasks including DNN training and adversarial attack/defense. Recently, a linear variant of BP named LinBP was introduced for generating more transferable adversarial examples for black-box adversarial attacks, by Guo et al. Yet, it has not been theoretically studied and the convergence analysis of such a method is lacking. This paper serves as a complement and somewhat an extension to Guo et al.'s paper, by providing theoretical analyses on LinBP in neural-network-involved learning tasks including adversarial attack and model training. We demonstrate that, somewhat surprisingly, LinBP can lead to faster convergence in these tasks in the same hyper-parameter settings, compared to BP. We confirm our theoretical results with extensive experiments.

摘要: 反向传播广泛用于深度神经网络(DNNs)的梯度计算。反向传播通常与随机梯度下降(SGD)或其变体一起应用，被认为是包括DNN训练和对抗性攻击/防御在内的各种机器学习任务的事实上的选择。最近，Guo等人引入了一种名为LinBP的BP线性变体，用于生成更多可移植的黑盒对抗攻击实例。然而，目前还没有从理论上对其进行研究，也缺乏对该方法的收敛性分析。本文是对Guo等人的论文的补充和某种程度上的扩展，通过对LinBP在包括对抗性攻击和模型训练在内的神经网络参与的学习任务中的理论分析，我们证明了在相同的超参数设置下，LinBP可以比BP更快地收敛到这些任务中，这一点令人惊讶，我们用大量的实验证实了我们的理论结果。



## **17. What are Attackers after on IoT Devices? An approach based on a multi-phased multi-faceted IoT honeypot ecosystem and data clustering**

攻击者在物联网设备上的目标是什么？一种基于多阶段多层面物联网蜜罐生态系统和数据聚类的方法 cs.CR

arXiv admin note: text overlap with arXiv:2003.01218

**SubmitDate**: 2021-12-21    [paper-pdf](http://arxiv.org/pdf/2112.10974v1)

**Authors**: Armin Ziaie Tabari, Xinming Ou, Anoop Singhal

**Abstracts**: The growing number of Internet of Things (IoT) devices makes it imperative to be aware of the real-world threats they face in terms of cybersecurity. While honeypots have been historically used as decoy devices to help researchers/organizations gain a better understanding of the dynamic of threats on a network and their impact, IoT devices pose a unique challenge for this purpose due to the variety of devices and their physical connections. In this work, by observing real-world attackers' behavior in a low-interaction honeypot ecosystem, we (1) presented a new approach to creating a multi-phased, multi-faceted honeypot ecosystem, which gradually increases the sophistication of honeypots' interactions with adversaries, (2) designed and developed a low-interaction honeypot for cameras that allowed researchers to gain a deeper understanding of what attackers are targeting, and (3) devised an innovative data analytics method to identify the goals of adversaries. Our honeypots have been active for over three years. We were able to collect increasingly sophisticated attack data in each phase. Furthermore, our data analytics points to the fact that the vast majority of attack activities captured in the honeypots share significant similarity, and can be clustered and grouped to better understand the goals, patterns, and trends of IoT attacks in the wild.

摘要: 随着物联网(IoT)设备数量的不断增加，必须意识到它们在网络安全方面面临的现实威胁。虽然蜜罐历来被用作诱饵设备，以帮助研究人员/组织更好地了解网络上的威胁动态及其影响，但物联网设备由于设备及其物理连接的多样性，对此提出了独特的挑战。在这项工作中，通过观察真实世界攻击者在低交互蜜罐生态系统中的行为，我们(1)提出了一种新的方法来创建一个多阶段、多方面的蜜罐生态系统，逐步提高了蜜罐与对手交互的复杂性；(2)设计并开发了一个用于摄像机的低交互蜜罐，使研究人员能够更深入地了解攻击者的目标；(3)设计了一种创新的数据分析方法来识别对手的目标。我们的蜜罐已经活跃了三年多了。我们能够在每个阶段收集越来越复杂的攻击数据。此外，我们的数据分析指出，在蜜罐中捕获的绝大多数攻击活动都有很大的相似性，可以进行群集和分组，以便更好地了解野外物联网攻击的目标、模式和趋势。



## **18. Channel-Aware Adversarial Attacks Against Deep Learning-Based Wireless Signal Classifiers**

针对基于深度学习的无线信号分类器的信道感知敌意攻击 eess.SP

Submitted for publication. arXiv admin note: substantial text overlap  with arXiv:2002.02400

**SubmitDate**: 2021-12-20    [paper-pdf](http://arxiv.org/pdf/2005.05321v3)

**Authors**: Brian Kim, Yalin E. Sagduyu, Kemal Davaslioglu, Tugba Erpek, Sennur Ulukus

**Abstracts**: This paper presents channel-aware adversarial attacks against deep learning-based wireless signal classifiers. There is a transmitter that transmits signals with different modulation types. A deep neural network is used at each receiver to classify its over-the-air received signals to modulation types. In the meantime, an adversary transmits an adversarial perturbation (subject to a power budget) to fool receivers into making errors in classifying signals that are received as superpositions of transmitted signals and adversarial perturbations. First, these evasion attacks are shown to fail when channels are not considered in designing adversarial perturbations. Then, realistic attacks are presented by considering channel effects from the adversary to each receiver. After showing that a channel-aware attack is selective (i.e., it affects only the receiver whose channel is considered in the perturbation design), a broadcast adversarial attack is presented by crafting a common adversarial perturbation to simultaneously fool classifiers at different receivers. The major vulnerability of modulation classifiers to over-the-air adversarial attacks is shown by accounting for different levels of information available about the channel, the transmitter input, and the classifier model. Finally, a certified defense based on randomized smoothing that augments training data with noise is introduced to make the modulation classifier robust to adversarial perturbations.

摘要: 提出了针对基于深度学习的无线信号分类器的信道感知敌意攻击。存在发送具有不同调制类型的信号的发射机。在每个接收器处使用深度神经网络来将其空中接收的信号分类为调制类型。同时，敌手发送对抗性扰动(受制于功率预算)以欺骗接收器在将接收到的信号分类为发送信号和对抗性扰动的叠加时出错。首先，当在设计对抗性扰动时不考虑通道时，这些逃避攻击被证明是失败的。然后，通过考虑从敌方到每个接收方的信道效应，给出了现实攻击。在证明信道感知攻击是选择性的(即，它只影响其信道在扰动设计中被考虑的接收机)之后，通过制作共同的敌意扰动来同时愚弄不同接收机的分类器来呈现广播敌意攻击。调制分类器对空中对抗性攻击的主要脆弱性是通过考虑有关信道、发射机输入和分类器模型的不同级别的可用信息来显示的。最后，介绍了一种基于随机平滑的认证防御方法，该方法在训练数据中加入噪声，使调制分类器对敌方干扰具有较强的鲁棒性。



## **19. An Evasion Attack against Stacked Capsule Autoencoder**

一种针对堆叠式胶囊自动编码器的逃避攻击 cs.LG

**SubmitDate**: 2021-12-20    [paper-pdf](http://arxiv.org/pdf/2010.07230v5)

**Authors**: Jiazhu Dai, Siwei Xiong

**Abstracts**: Capsule network is a type of neural network that uses the spatial relationship between features to classify images. By capturing the poses and relative positions between features, its ability to recognize affine transformation is improved, and it surpasses traditional convolutional neural networks (CNNs) when handling translation, rotation and scaling. The Stacked Capsule Autoencoder (SCAE) is the state-of-the-art capsule network. The SCAE encodes an image as capsules, each of which contains poses of features and their correlations. The encoded contents are then input into the downstream classifier to predict the categories of the images. Existing research mainly focuses on the security of capsule networks with dynamic routing or EM routing, and little attention has been given to the security and robustness of the SCAE. In this paper, we propose an evasion attack against the SCAE. After a perturbation is generated based on the output of the object capsules in the model, it is added to an image to reduce the contribution of the object capsules related to the original category of the image so that the perturbed image will be misclassified. We evaluate the attack using an image classification experiment, and the experimental results indicate that the attack can achieve high success rates and stealthiness. It confirms that the SCAE has a security vulnerability whereby it is possible to craft adversarial samples without changing the original structure of the image to fool the classifiers. We hope that our work will make the community aware of the threat of this attack and raise the attention given to the SCAE's security.

摘要: 胶囊网络是一种利用特征之间的空间关系对图像进行分类的神经网络。通过捕捉特征间的姿态和相对位置，提高了其识别仿射变换的能力，在处理平移、旋转和缩放方面优于传统的卷积神经网络(CNNs)。堆叠式胶囊自动编码器(SCAE)是最先进的胶囊网络。SCAE将图像编码为胶囊，每个胶囊包含特征的姿势及其相关性。然后将编码内容输入下游分类器以预测图像的类别。现有的研究主要集中在动态路由或EM路由的胶囊网络的安全性上，而对SCAE的安全性和健壮性关注较少。在本文中，我们提出了一种针对SCAE的逃避攻击。在基于模型中的对象胶囊的输出产生扰动之后，将其添加到图像中，以减少与图像的原始类别相关的对象胶囊的贡献，从而使得扰动图像将被误分类。通过图像分类实验对该攻击进行了评估，实验结果表明该攻击具有较高的成功率和隐蔽性。它确认SCAE存在安全漏洞，从而可以在不更改图像原始结构的情况下手工制作敌意样本来愚弄分类器。我们希望我们的工作能让社会认识到这次袭击的威胁，并提高对SCAE安全的关注。



## **20. Adversarial Attacks on Spiking Convolutional Networks for Event-based Vision**

基于事件视觉的尖峰卷积网络对抗性攻击 cs.CV

16 pages, preprint, submitted to ICLR 2022

**SubmitDate**: 2021-12-20    [paper-pdf](http://arxiv.org/pdf/2110.02929v2)

**Authors**: Julian Büchel, Gregor Lenz, Yalun Hu, Sadique Sheik, Martino Sorbaro

**Abstracts**: Event-based sensing using dynamic vision sensors is gaining traction in low-power vision applications. Spiking neural networks work well with the sparse nature of event-based data and suit deployment on low-power neuromorphic hardware. Being a nascent field, the sensitivity of spiking neural networks to potentially malicious adversarial attacks has received very little attention so far. In this work, we show how white-box adversarial attack algorithms can be adapted to the discrete and sparse nature of event-based visual data, and to the continuous-time setting of spiking neural networks. We test our methods on the N-MNIST and IBM Gestures neuromorphic vision datasets and show adversarial perturbations achieve a high success rate, by injecting a relatively small number of appropriately placed events. We also verify, for the first time, the effectiveness of these perturbations directly on neuromorphic hardware. Finally, we discuss the properties of the resulting perturbations and possible future directions.

摘要: 使用动态视觉传感器的基于事件的传感在低功耗视觉应用中获得了吸引力。尖峰神经网络很好地利用了基于事件的数据的稀疏特性，适合在低功耗神经形态硬件上部署。作为一个新兴的领域，尖峰神经网络对潜在的恶意攻击的敏感度到目前为止还很少受到关注。在这项工作中，我们展示了白盒对抗性攻击算法如何适应基于事件的视觉数据的离散性和稀疏性，以及尖峰神经网络的连续时间设置。我们在N-MNIST和IBM手势神经形态视觉数据集上测试了我们的方法，结果表明，通过注入相对较少数量的适当放置的事件，对抗性扰动获得了高成功率。我们还首次验证了这些扰动直接在神经形态硬件上的有效性。最后，我们讨论了由此产生的扰动的性质和未来可能的发展方向。



## **21. Certified Federated Adversarial Training**

认证的联合对抗赛训练 cs.LG

First presented at the 1st NeurIPS Workshop on New Frontiers in  Federated Learning (NFFL 2021)

**SubmitDate**: 2021-12-20    [paper-pdf](http://arxiv.org/pdf/2112.10525v1)

**Authors**: Giulio Zizzo, Ambrish Rawat, Mathieu Sinn, Sergio Maffeis, Chris Hankin

**Abstracts**: In federated learning (FL), robust aggregation schemes have been developed to protect against malicious clients. Many robust aggregation schemes rely on certain numbers of benign clients being present in a quorum of workers. This can be hard to guarantee when clients can join at will, or join based on factors such as idle system status, and connected to power and WiFi. We tackle the scenario of securing FL systems conducting adversarial training when a quorum of workers could be completely malicious. We model an attacker who poisons the model to insert a weakness into the adversarial training such that the model displays apparent adversarial robustness, while the attacker can exploit the inserted weakness to bypass the adversarial training and force the model to misclassify adversarial examples. We use abstract interpretation techniques to detect such stealthy attacks and block the corrupted model updates. We show that this defence can preserve adversarial robustness even against an adaptive attacker.

摘要: 在联合学习(FL)中，已经开发了健壮的聚合方案来保护其免受恶意客户端的攻击。许多健壮的聚合方案依赖于一定数量的良性客户端存在于法定工作人数中。当客户可以随意加入，或者基于空闲系统状态等因素加入，并连接到电源和WiFi时，这可能很难保证。我们撞击的场景是保护FL系统，进行对抗性训练，而法定人数的工人可能是完全恶意的。我们对毒害模型的攻击者进行建模，以便在对抗性训练中插入弱点，使得模型显示出明显的对抗性健壮性，而攻击者可以利用插入的弱点绕过对抗性训练，迫使模型对对抗性示例进行错误分类。我们使用抽象解释技术来检测此类隐蔽攻击，并使用挡路检测损坏的模型更新。我们证明了这种防御即使在抵抗自适应攻击者的情况下也能保持对手的健壮性。



## **22. Unifying Model Explainability and Robustness for Joint Text Classification and Rationale Extraction**

联合文本分类和理论抽取的统一模型可解释性和鲁棒性 cs.CL

AAAI 2022

**SubmitDate**: 2021-12-20    [paper-pdf](http://arxiv.org/pdf/2112.10424v1)

**Authors**: Dongfang Li, Baotian Hu, Qingcai Chen, Tujie Xu, Jingcong Tao, Yunan Zhang

**Abstracts**: Recent works have shown explainability and robustness are two crucial ingredients of trustworthy and reliable text classification. However, previous works usually address one of two aspects: i) how to extract accurate rationales for explainability while being beneficial to prediction; ii) how to make the predictive model robust to different types of adversarial attacks. Intuitively, a model that produces helpful explanations should be more robust against adversarial attacks, because we cannot trust the model that outputs explanations but changes its prediction under small perturbations. To this end, we propose a joint classification and rationale extraction model named AT-BMC. It includes two key mechanisms: mixed Adversarial Training (AT) is designed to use various perturbations in discrete and embedding space to improve the model's robustness, and Boundary Match Constraint (BMC) helps to locate rationales more precisely with the guidance of boundary information. Performances on benchmark datasets demonstrate that the proposed AT-BMC outperforms baselines on both classification and rationale extraction by a large margin. Robustness analysis shows that the proposed AT-BMC decreases the attack success rate effectively by up to 69%. The empirical results indicate that there are connections between robust models and better explanations.

摘要: 最近的研究表明，可解释性和稳健性是可信和可靠文本分类的两个关键因素。然而，以往的工作通常涉及两个方面：一是如何在有利于预测的同时提取准确的可解释性依据；二是如何使预测模型对不同类型的对抗性攻击具有鲁棒性。直观地说，产生有用解释的模型应该对对手攻击更健壮，因为我们不能信任输出解释但在小扰动下改变其预测的模型。为此，我们提出了一种联合分类和原理抽取模型AT-BMC。它包括两个关键机制：混合对抗性训练(AT)旨在利用离散空间和嵌入空间中的各种扰动来提高模型的鲁棒性；边界匹配约束(BMC)在边界信息的指导下帮助更精确地定位理性。在基准数据集上的性能表明，所提出的AT-BMC在分类和原理提取方面都比基线有较大幅度的提高。鲁棒性分析表明，提出的AT-BMC能有效降低攻击成功率高达69%。实证结果表明，稳健模型与更好的解释之间存在联系。



## **23. Energy-bounded Learning for Robust Models of Code**

代码健壮模型的能量受限学习 cs.LG

arXiv admin note: text overlap with arXiv:2010.03759 by other authors

**SubmitDate**: 2021-12-20    [paper-pdf](http://arxiv.org/pdf/2112.11226v1)

**Authors**: Nghi D. Q. Bui, Yijun Yu

**Abstracts**: In programming, learning code representations has a variety of applications, including code classification, code search, comment generation, bug prediction, and so on. Various representations of code in terms of tokens, syntax trees, dependency graphs, code navigation paths, or a combination of their variants have been proposed, however, existing vanilla learning techniques have a major limitation in robustness, i.e., it is easy for the models to make incorrect predictions when the inputs are altered in a subtle way. To enhance the robustness, existing approaches focus on recognizing adversarial samples rather than on the valid samples that fall outside a given distribution, which we refer to as out-of-distribution (OOD) samples. Recognizing such OOD samples is the novel problem investigated in this paper. To this end, we propose to first augment the in=distribution datasets with out-of-distribution samples such that, when trained together, they will enhance the model's robustness. We propose the use of an energy-bounded learning objective function to assign a higher score to in-distribution samples and a lower score to out-of-distribution samples in order to incorporate such out-of-distribution samples into the training process of source code models. In terms of OOD detection and adversarial samples detection, our evaluation results demonstrate a greater robustness for existing source code models to become more accurate at recognizing OOD data while being more resistant to adversarial attacks at the same time. Furthermore, the proposed energy-bounded score outperforms all existing OOD detection scores by a large margin, including the softmax confidence score, the Mahalanobis score, and ODIN.

摘要: 在编程中，学习代码表示有多种应用，包括代码分类、代码搜索、注释生成、错误预测等。已经提出了关于令牌、语法树、依赖图、代码导航路径或其变体的组合的代码的各种表示，然而，现有的普通学习技术在稳健性方面具有主要限制，即，当输入以微妙的方式改变时，模型容易做出不正确的预测。为了增强鲁棒性，现有的方法侧重于识别敌意样本，而不是识别在给定分布之外的有效样本，我们称之为分布外(OOD)样本。识别此类面向对象的样本是本文研究的新问题。为此，我们建议首先用分布外样本扩充In=分布数据集，以便当它们一起训练时，将增强模型的稳健性。为了将分布外样本纳入源代码模型的训练过程中，我们提出使用能量受限的学习目标函数，为分布内样本赋予较高的分数，为分布外样本赋予较低的分数。在OOD检测和敌意样本检测方面，我们的评估结果表明，现有的源代码模型在更准确地识别OOD数据的同时，更能抵抗敌意攻击，具有更强的鲁棒性。此外，所提出的能量受限分数大大超过了所有现有的OOD检测分数，包括Softmax置信度分数、Mahalanobis分数和ODIN分数。



## **24. Knowledge Cross-Distillation for Membership Privacy**

面向会员隐私的知识交叉蒸馏 cs.CR

Under Review

**SubmitDate**: 2021-12-20    [paper-pdf](http://arxiv.org/pdf/2111.01363v2)

**Authors**: Rishav Chourasia, Batnyam Enkhtaivan, Kunihiro Ito, Junki Mori, Isamu Teranishi, Hikaru Tsuchida

**Abstracts**: A membership inference attack (MIA) poses privacy risks on the training data of a machine learning model. With an MIA, an attacker guesses if the target data are a member of the training dataset. The state-of-the-art defense against MIAs, distillation for membership privacy (DMP), requires not only private data to protect but a large amount of unlabeled public data. However, in certain privacy-sensitive domains, such as medical and financial, the availability of public data is not obvious. Moreover, a trivial method to generate the public data by using generative adversarial networks significantly decreases the model accuracy, as reported by the authors of DMP. To overcome this problem, we propose a novel defense against MIAs using knowledge distillation without requiring public data. Our experiments show that the privacy protection and accuracy of our defense are comparable with those of DMP for the benchmark tabular datasets used in MIA researches, Purchase100 and Texas100, and our defense has much better privacy-utility trade-off than those of the existing defenses without using public data for image dataset CIFAR10.

摘要: 成员关系推理攻击(MIA)会给机器学习模型的训练数据带来隐私风险。使用MIA，攻击者可以猜测目标数据是否为训练数据集的成员。针对MIA的最先进的防御措施，即会员隐私蒸馏(DMP)，不仅需要保护私人数据，还需要大量未标记的公共数据。然而，在某些隐私敏感领域，如医疗和金融，公开数据的可用性并不明显。此外，正如DMP的作者所报告的那样，使用生成性对抗网络来生成公共数据的琐碎方法显著降低了模型的准确性。为了克服这一问题，我们提出了一种新的防御MIA的方法，该方法使用知识蒸馏而不需要公开数据。我们的实验表明，对于MIA研究中使用的基准表格数据集，我们的防御方案的隐私保护和准确性与DMP相当，并且我们的防御方案在隐私效用方面比现有的防御方案具有更好的隐私效用权衡，而不使用公共数据的图像数据集CIFAR10的情况下，我们的防御方案具有更好的隐私效用权衡。



## **25. Toward Evaluating Re-identification Risks in the Local Privacy Model**

关于评估本地隐私模型中重新识别风险的方法 cs.CR

Accepted at Transactions on Data Privacy

**SubmitDate**: 2021-12-19    [paper-pdf](http://arxiv.org/pdf/2010.08238v5)

**Authors**: Takao Murakami, Kenta Takahashi

**Abstracts**: LDP (Local Differential Privacy) has recently attracted much attention as a metric of data privacy that prevents the inference of personal data from obfuscated data in the local model. However, there are scenarios in which the adversary wants to perform re-identification attacks to link the obfuscated data to users in this model. LDP can cause excessive obfuscation and destroy the utility in these scenarios because it is not designed to directly prevent re-identification. In this paper, we propose a measure of re-identification risks, which we call PIE (Personal Information Entropy). The PIE is designed so that it directly prevents re-identification attacks in the local model. It lower-bounds the lowest possible re-identification error probability (i.e., Bayes error probability) of the adversary. We analyze the relation between LDP and the PIE, and analyze the PIE and utility in distribution estimation for two obfuscation mechanisms providing LDP. Through experiments, we show that when we consider re-identification as a privacy risk, LDP can cause excessive obfuscation and destroy the utility. Then we show that the PIE can be used to guarantee low re-identification risks for the local obfuscation mechanisms while keeping high utility.

摘要: LDP(Local Differential Privacy，局部差分隐私)作为一种数据隐私度量，防止了从局部模型中的混淆数据中推断个人数据，近年来受到了广泛的关注。但是，在某些情况下，对手想要执行重新识别攻击，以将模糊数据链接到此模型中的用户。在这些场景中，LDP可能会导致过度混淆并破坏实用程序，因为它不是直接防止重新识别的。本文提出了一种重新识别风险的度量方法，称为PIE(Personal Information Entropy)，即个人信息熵(Personal Information Entropy)。饼的设计可以直接防止本地模型中的重新标识攻击。它降低了对手的最低可能的重新识别错误概率(即，贝叶斯错误概率)。分析了LDP与PIE的关系，分析了提供LDP的两种混淆机制的PIE及其在分布估计中的效用。通过实验表明，当我们将重识别视为隐私风险时，LDP会造成过度的混淆，破坏效用。然后，我们证明了该派可以用来保证局部混淆机制在保持较高效用的同时具有较低的重新识别风险。



## **26. Jamming Pattern Recognition over Multi-Channel Networks: A Deep Learning Approach**

多通道网络干扰模式识别：一种深度学习方法 cs.CR

**SubmitDate**: 2021-12-19    [paper-pdf](http://arxiv.org/pdf/2112.11222v1)

**Authors**: Ali Pourranjbar, Georges Kaddoum, Walid Saad

**Abstracts**: With the advent of intelligent jammers, jamming attacks have become a more severe threat to the performance of wireless systems. An intelligent jammer is able to change its policy to minimize the probability of being traced by legitimate nodes. Thus, an anti-jamming mechanism capable of constantly adjusting to the jamming policy is required to combat such a jammer. Remarkably, existing anti-jamming methods are not applicable here because they mainly focus on mitigating jamming attacks with an invariant jamming policy, and they rarely consider an intelligent jammer as an adversary. Therefore, in this paper, to employ a jamming type recognition technique working alongside an anti-jamming technique is proposed. The proposed recognition method employs a recurrent neural network that takes the jammer's occupied channels as inputs and outputs the jammer type. Under this scheme, the real-time jammer policy is first identified, and, then, the most appropriate countermeasure is chosen. Consequently, any changes to the jammer policy can be instantly detected with the proposed recognition technique allowing for a rapid switch to a new anti-jamming method fitted to the new jamming policy. To evaluate the performance of the proposed recognition method, the accuracy of the detection is derived as a function of the jammer policy switching time. Simulation results show the detection accuracy for all the considered users numbers is greater than 70% when the jammer switches its policy every 5 time slots and the accuracy raises to 90% when the jammer policy switching time is 45.

摘要: 随着智能干扰机的出现，干扰攻击对无线系统的性能构成了更加严重的威胁。智能干扰器能够改变其策略，以最大限度地降低被合法节点跟踪的概率。因此，需要一种能够不断调整干扰策略的抗干扰机制来对抗这样的干扰。值得注意的是，现有的抗干扰方法在这里并不适用，因为它们主要集中在通过不变的干扰策略来缓解干扰攻击，而很少将智能干扰器视为对手。因此，本文提出将干扰类型识别技术与抗干扰技术结合使用。该识别方法采用递归神经网络，以干扰机占用的信道为输入，输出干扰机类型。在该方案下，首先识别实时干扰策略，然后选择最合适的对策。因此，利用所提出的识别技术可以立即检测到干扰策略的任何改变，从而允许快速切换到适合于新的干扰策略的新的抗干扰方法。为了评估所提出的识别方法的性能，推导了作为干扰策略切换时间的函数的检测精度。仿真结果表明，干扰机每隔5个时隙切换一次策略，对所有考虑的用户数的检测准确率均大于70%，当干扰机策略切换时间为45次时，准确率提高到90%。



## **27. Attacking Point Cloud Segmentation with Color-only Perturbation**

基于纯颜色摄动的攻击点云分割 cs.CV

**SubmitDate**: 2021-12-18    [paper-pdf](http://arxiv.org/pdf/2112.05871v2)

**Authors**: Jiacen Xu, Zhe Zhou, Boyuan Feng, Yufei Ding, Zhou Li

**Abstracts**: Recent research efforts on 3D point-cloud semantic segmentation have achieved outstanding performance by adopting deep CNN (convolutional neural networks) and GCN (graph convolutional networks). However, the robustness of these complex models has not been systematically analyzed. Given that semantic segmentation has been applied in many safety-critical applications (e.g., autonomous driving, geological sensing), it is important to fill this knowledge gap, in particular, how these models are affected under adversarial samples. While adversarial attacks against point cloud have been studied, we found all of them were targeting single-object recognition, and the perturbation is done on the point coordinates. We argue that the coordinate-based perturbation is unlikely to realize under the physical-world constraints. Hence, we propose a new color-only perturbation method named COLPER, and tailor it to semantic segmentation. By evaluating COLPER on an indoor dataset (S3DIS) and an outdoor dataset (Semantic3D) against three point cloud segmentation models (PointNet++, DeepGCNs, and RandLA-Net), we found color-only perturbation is sufficient to significantly drop the segmentation accuracy and aIoU, under both targeted and non-targeted attack settings.

摘要: 最近的三维点云语义分割研究采用深度卷积神经网络(CNN)和图卷积网络(GCN)，取得了很好的效果。然而，这些复杂模型的稳健性还没有得到系统的分析。鉴于语义分割已经应用于许多安全关键应用(例如，自动驾驶、地质传感)，填补这一知识空白是很重要的，特别是这些模型在敌意样本下是如何受到影响的。在对点云进行对抗性攻击的研究中，我们发现它们都是针对单目标识别的，并且扰动都是在点坐标上进行的。我们认为，在物理世界的约束下，基于坐标的微扰是不可能实现的。为此，我们提出了一种新的纯颜色扰动方法COLPER，并对其进行了语义分割。通过针对三种点云分割模型(PointNet++、DeepGCNs和RandLA-Net)评估室内数据集(S3DIS)和室外数据集(Semanc3D)上的COLPER，我们发现，在目标攻击和非目标攻击设置下，仅颜色扰动就足以显著降低分割精度和AIoU。



## **28. Adversarial Attack for Uncertainty Estimation: Identifying Critical Regions in Neural Networks**

不确定性估计的对抗性攻击：识别神经网络中的关键区域 cs.LG

15 pages, 6 figures, Neural Process Lett (2021)

**SubmitDate**: 2021-12-18    [paper-pdf](http://arxiv.org/pdf/2107.07618v2)

**Authors**: Ismail Alarab, Simant Prakoonwit

**Abstracts**: We propose a novel method to capture data points near decision boundary in neural network that are often referred to a specific type of uncertainty. In our approach, we sought to perform uncertainty estimation based on the idea of adversarial attack method. In this paper, uncertainty estimates are derived from the input perturbations, unlike previous studies that provide perturbations on the model's parameters as in Bayesian approach. We are able to produce uncertainty with couple of perturbations on the inputs. Interestingly, we apply the proposed method to datasets derived from blockchain. We compare the performance of model uncertainty with the most recent uncertainty methods. We show that the proposed method has revealed a significant outperformance over other methods and provided less risk to capture model uncertainty in machine learning.

摘要: 我们提出了一种新的方法来捕获神经网络中决策边界附近的数据点，这些数据点通常指的是特定类型的不确定性。在我们的方法中，我们试图基于对抗性攻击方法的思想进行不确定性估计。在本文中，不确定性估计是从输入摄动推导出来的，不同于以往的研究提供对模型参数的摄动，如在贝叶斯方法中。我们可以通过对输入的几个扰动来产生不确定性。有趣的是，我们将所提出的方法应用于从区块链派生的数据集。我们将模型不确定性的性能与最新的不确定性方法进行了比较。结果表明，与其他方法相比，本文提出的方法具有更好的性能，并且在获取机器学习中的模型不确定性方面具有更小的风险。



## **29. Dynamic Defender-Attacker Blotto Game**

动态防御者-攻击者Blotto博弈 eess.SY

**SubmitDate**: 2021-12-18    [paper-pdf](http://arxiv.org/pdf/2112.09890v1)

**Authors**: Daigo Shishika, Yue Guan, Michael Dorothy, Vijay Kumar

**Abstracts**: This work studies a dynamic, adversarial resource allocation problem in environments modeled as graphs. A blue team of defender robots are deployed in the environment to protect the nodes from a red team of attacker robots. We formulate the engagement as a discrete-time dynamic game, where the robots can move at most one hop in each time step. The game terminates with the attacker's win if any location has more attacker robots than defender robots at any time. The goal is to identify dynamic resource allocation strategies, as well as the conditions that determines the winner: graph structure, available resources, and initial conditions. We analyze the problem using reachable sets and show how the outdegree of the underlying graph directly influences the difficulty of the defending task. Furthermore, we provide algorithms that identify sufficiency of attacker's victory.

摘要: 这项工作研究了一个动态的，对抗性的资源分配问题，在建模为图的环境中。在环境中部署了一组蓝色的防御机器人，以保护节点不受一组攻击机器人的攻击。我们将交战描述为一个离散时间动态博弈，其中机器人在每个时间步长内最多只能移动一跳。如果任何位置的攻击型机器人在任何时候都多于防守机器人，游戏就会随着攻击者的胜利而终止。目标是确定动态资源分配策略，以及决定赢家的条件：图结构、可用资源和初始条件。我们使用可达集对问题进行了分析，并展示了底层图的出度如何直接影响防御任务的难度。此外，我们还提供了识别攻击者胜利的充分性的算法。



## **30. Formalizing Generalization and Robustness of Neural Networks to Weight Perturbations**

神经网络对权重摄动的泛化和鲁棒性的形式化 cs.LG

This version has been accepted for poster presentation at NeurIPS  2021

**SubmitDate**: 2021-12-17    [paper-pdf](http://arxiv.org/pdf/2103.02200v2)

**Authors**: Yu-Lin Tsai, Chia-Yi Hsu, Chia-Mu Yu, Pin-Yu Chen

**Abstracts**: Studying the sensitivity of weight perturbation in neural networks and its impacts on model performance, including generalization and robustness, is an active research topic due to its implications on a wide range of machine learning tasks such as model compression, generalization gap assessment, and adversarial attacks. In this paper, we provide the first integral study and analysis for feed-forward neural networks in terms of the robustness in pairwise class margin and its generalization behavior under weight perturbation. We further design a new theory-driven loss function for training generalizable and robust neural networks against weight perturbations. Empirical experiments are conducted to validate our theoretical analysis. Our results offer fundamental insights for characterizing the generalization and robustness of neural networks against weight perturbations.

摘要: 研究神经网络中权重扰动的敏感性及其对模型性能(包括泛化和鲁棒性)的影响是一个活跃的研究课题，因为它涉及到广泛的机器学习任务，如模型压缩、泛化差距评估和敌意攻击。本文首次对前馈神经网络的两类边界鲁棒性及其在权值扰动下的泛化行为进行了整体研究和分析。我们进一步设计了一种新的理论驱动的损失函数，用于训练泛化的、抗权重扰动的鲁棒神经网络。通过实证实验验证了我们的理论分析。我们的结果为表征神经网络的泛化和抗权重扰动的鲁棒性提供了基本的见解。



## **31. Reasoning Chain Based Adversarial Attack for Multi-hop Question Answering**

基于推理链的多跳问答对抗性攻击 cs.CL

10 pages including reference, 4 figures

**SubmitDate**: 2021-12-17    [paper-pdf](http://arxiv.org/pdf/2112.09658v1)

**Authors**: Jiayu Ding, Siyuan Wang, Qin Chen, Zhongyu Wei

**Abstracts**: Recent years have witnessed impressive advances in challenging multi-hop QA tasks. However, these QA models may fail when faced with some disturbance in the input text and their interpretability for conducting multi-hop reasoning remains uncertain. Previous adversarial attack works usually edit the whole question sentence, which has limited effect on testing the entity-based multi-hop inference ability. In this paper, we propose a multi-hop reasoning chain based adversarial attack method. We formulate the multi-hop reasoning chains starting from the query entity to the answer entity in the constructed graph, which allows us to align the question to each reasoning hop and thus attack any hop. We categorize the questions into different reasoning types and adversarially modify part of the question corresponding to the selected reasoning hop to generate the distracting sentence. We test our adversarial scheme on three QA models on HotpotQA dataset. The results demonstrate significant performance reduction on both answer and supporting facts prediction, verifying the effectiveness of our reasoning chain based attack method for multi-hop reasoning models and the vulnerability of them. Our adversarial re-training further improves the performance and robustness of these models.

摘要: 近年来，在具有挑战性的多跳QA任务方面取得了令人印象深刻的进展。然而，当遇到输入文本中的某些干扰时，这些QA模型可能会失败，并且它们对进行多跳推理的解释力仍然不确定。以往的对抗性攻击工作通常都是对整个问句进行编辑，这对测试基于实体的多跳推理能力的效果有限。本文提出了一种基于多跳推理链的对抗性攻击方法。在构造的图中，我们构造了从查询实体到答案实体的多跳推理链，允许我们将问题对齐到每个推理跳，从而攻击任何一跳。我们将问题分类为不同的推理类型，并对所选择的推理跳对应的部分问题进行对抗性修改，以生成分散注意力的句子。我们在HotpotQA数据集上的三个QA模型上测试了我们的对抗性方案。实验结果表明，基于推理链的多跳推理模型攻击方法在答案和支持事实预测方面都有明显的性能下降，验证了该方法的有效性和脆弱性。我们的对抗性再训练进一步提高了这些模型的性能和鲁棒性。



## **32. Who Is the Strongest Enemy? Towards Optimal and Efficient Evasion Attacks in Deep RL**

谁是最强大的敌人？基于Deep RL的最优高效规避攻击研究 cs.LG

**SubmitDate**: 2021-12-17    [paper-pdf](http://arxiv.org/pdf/2106.05087v2)

**Authors**: Yanchao Sun, Ruijie Zheng, Yongyuan Liang, Furong Huang

**Abstracts**: Evaluating the worst-case performance of a reinforcement learning (RL) agent under the strongest/optimal adversarial perturbations on state observations (within some constraints) is crucial for understanding the robustness of RL agents. However, finding the optimal adversary is challenging, in terms of both whether we can find the optimal attack and how efficiently we can find it. Existing works on adversarial RL either use heuristics-based methods that may not find the strongest adversary, or directly train an RL-based adversary by treating the agent as a part of the environment, which can find the optimal adversary but may become intractable in a large state space. This paper introduces a novel attacking method to find the optimal attacks through collaboration between a designed function named ''actor'' and an RL-based learner named "director". The actor crafts state perturbations for a given policy perturbation direction, and the director learns to propose the best policy perturbation directions. Our proposed algorithm, PA-AD, is theoretically optimal and significantly more efficient than prior RL-based works in environments with large state spaces. Empirical results show that our proposed PA-AD universally outperforms state-of-the-art attacking methods in various Atari and MuJoCo environments. By applying PA-AD to adversarial training, we achieve state-of-the-art empirical robustness in multiple tasks under strong adversaries.

摘要: 评估强化学习(RL)Agent在状态观测(在一定约束范围内)的最强/最优对抗扰动下的最坏情况下的性能，对于理解RL Agent的鲁棒性是至关重要的。然而，无论是从我们是否能找到最佳攻击，还是从我们找到最佳攻击的效率来看，找到最佳对手都是具有挑战性的。现有的对抗性RL研究要么使用基于启发式的方法，可能找不到最强的对手，要么将Agent视为环境的一部分，直接训练基于RL的对手，这可以找到最优的对手，但在大的状态空间中可能会变得难以处理。本文提出了一种新的攻击方法，通过设计一个名为“参与者”的函数和一个名为“导演”的基于RL的学习器之间的协作来寻找最优攻击。参与者为给定的政策扰动方向制作状态扰动，导演学习提出最佳政策扰动方向。我们提出的算法PA-AD在理论上是最优的，并且在具有大状态空间的环境中比以前的基于RL的工作效率要高得多。实验结果表明，在不同的Atari和MuJoCo环境下，我们提出的PA-AD攻击方法普遍优于最新的攻击方法。通过将PA-AD应用于对抗性训练，我们在强对手下的多任务中获得了最先进的经验鲁棒性。



## **33. Dynamics-aware Adversarial Attack of 3D Sparse Convolution Network**

三维稀疏卷积网络的动态感知敌意攻击 cs.CV

**SubmitDate**: 2021-12-17    [paper-pdf](http://arxiv.org/pdf/2112.09428v1)

**Authors**: An Tao, Yueqi Duan, He Wang, Ziyi Wu, Pengliang Ji, Haowen Sun, Jie Zhou, Jiwen Lu

**Abstracts**: In this paper, we investigate the dynamics-aware adversarial attack problem in deep neural networks. Most existing adversarial attack algorithms are designed under a basic assumption -- the network architecture is fixed throughout the attack process. However, this assumption does not hold for many recently proposed networks, e.g. 3D sparse convolution network, which contains input-dependent execution to improve computational efficiency. It results in a serious issue of lagged gradient, making the learned attack at the current step ineffective due to the architecture changes afterward. To address this issue, we propose a Leaded Gradient Method (LGM) and show the significant effects of the lagged gradient. More specifically, we re-formulate the gradients to be aware of the potential dynamic changes of network architectures, so that the learned attack better "leads" the next step than the dynamics-unaware methods when network architecture changes dynamically. Extensive experiments on various datasets show that our LGM achieves impressive performance on semantic segmentation and classification. Compared with the dynamic-unaware methods, LGM achieves about 20% lower mIoU averagely on the ScanNet and S3DIS datasets. LGM also outperforms the recent point cloud attacks.

摘要: 本文研究了深层神经网络中动态感知的敌意攻击问题。大多数现有的对抗性攻击算法都是在一个基本假设下设计的--网络体系结构在整个攻击过程中都是固定的。然而，这一假设并不适用于最近提出的许多网络，例如3D稀疏卷积网络，它包含依赖输入的执行以提高计算效率。这导致了严重的梯度滞后问题，使得当前步骤的学习攻击由于之后的体系结构变化而无效。为了解决这个问题，我们提出了一种领先梯度法(LGM)，并展示了滞后梯度的显著影响。更具体地说，我们重新制定了梯度来感知网络体系结构的潜在动态变化，以便在网络体系结构动态变化时，学习到的攻击比不感知动态变化的方法更好地“引导”下一步。在不同数据集上的大量实验表明，我们的LGM在语义分割和分类方面取得了令人印象深刻的性能。与动态无感知方法相比，LGM在ScanNet和S3DIS数据集上的MIU值平均降低了20%左右。LGM的性能也优于最近的点云攻击。



## **34. APTSHIELD: A Stable, Efficient and Real-time APT Detection System for Linux Hosts**

APTSHIELD：一种稳定、高效、实时的Linux主机APT检测系统 cs.CR

**SubmitDate**: 2021-12-17    [paper-pdf](http://arxiv.org/pdf/2112.09008v2)

**Authors**: Tiantian Zhu, Jinkai Yu, Tieming Chen, Jiayu Wang, Jie Ying, Ye Tian, Mingqi Lv, Yan Chen, Yuan Fan, Ting Wang

**Abstracts**: Advanced Persistent Threat (APT) attack usually refers to the form of long-term, covert and sustained attack on specific targets, with an adversary using advanced attack techniques to destroy the key facilities of an organization. APT attacks have caused serious security threats and massive financial loss worldwide. Academics and industry thereby have proposed a series of solutions to detect APT attacks, such as dynamic/static code analysis, traffic detection, sandbox technology, endpoint detection and response (EDR), etc. However, existing defenses are failed to accurately and effectively defend against the current APT attacks that exhibit strong persistent, stealthy, diverse and dynamic characteristics due to the weak data source integrity, large data processing overhead and poor real-time performance in the process of real-world scenarios.   To overcome these difficulties, in this paper we propose APTSHIELD, a stable, efficient and real-time APT detection system for Linux hosts. In the aspect of data collection, audit is selected to stably collect kernel data of the operating system so as to carry out a complete portrait of the attack based on comprehensive analysis and comparison of existing logging tools; In the aspect of data processing, redundant semantics skipping and non-viable node pruning are adopted to reduce the amount of data, so as to reduce the overhead of the detection system; In the aspect of attack detection, an APT attack detection framework based on ATT\&CK model is designed to carry out real-time attack response and alarm through the transfer and aggregation of labels. Experimental results on both laboratory and Darpa Engagement show that our system can effectively detect web vulnerability attacks, file-less attacks and remote access trojan attacks, and has a low false positive rate, which adds far more value than the existing frontier work.

摘要: 高级持续威胁(APT)攻击通常是指敌方利用先进的攻击技术，对特定目标进行长期、隐蔽、持续的攻击，破坏组织的关键设施。APT攻击在全球范围内造成了严重的安全威胁和巨大的经济损失。学术界和产业界为此提出了一系列检测APT攻击的解决方案，如动态/静电代码分析、流量检测、沙盒技术、端点检测与响应等。然而，现有的防御措施在现实场景中，由于数据源完整性差、数据处理开销大、实时性差等原因，无法准确有效地防御当前APT攻击，表现出较强的持久性、隐蔽性、多样性和动态性。为了克服这些困难，本文提出了一种稳定、高效、实时的Linux主机APT检测系统APTSHIELD。在数据采集方面，在综合分析比较现有日志记录工具的基础上，选择AUDIT稳定采集操作系统内核数据，对攻击进行完整的刻画；在数据处理方面，采用冗余语义跳过和不可生存节点剪枝，减少了数据量，降低了检测系统的开销；在攻击检测方面，设计了基于ATT\&CK模型的APT攻击检测框架，通过传输进行实时攻击响应和报警实验室和DAPA实验结果表明，该系统能够有效地检测出Web漏洞攻击、无文件攻击和远程访问木马攻击，并且误报率较低，比现有的前沿工作有更大的增值价值。研究结果表明，该系统能够有效地检测出网络漏洞攻击、无文件攻击和远程访问木马攻击，并且具有较低的误报率。



## **35. Deep Bayesian Learning for Car Hacking Detection**

深度贝叶斯学习在汽车黑客检测中的应用 cs.CR

**SubmitDate**: 2021-12-17    [paper-pdf](http://arxiv.org/pdf/2112.09333v1)

**Authors**: Laha Ale, Scott A. King, Ning Zhang

**Abstracts**: With the rise of self-drive cars and connected vehicles, cars are equipped with various devices to assistant the drivers or support self-drive systems. Undoubtedly, cars have become more intelligent as we can deploy more and more devices and software on the cars. Accordingly, the security of assistant and self-drive systems in the cars becomes a life-threatening issue as smart cars can be invaded by malicious attacks that cause traffic accidents. Currently, canonical machine learning and deep learning methods are extensively employed in car hacking detection. However, machine learning and deep learning methods can easily be overconfident and defeated by carefully designed adversarial examples. Moreover, those methods cannot provide explanations for security engineers for further analysis. In this work, we investigated Deep Bayesian Learning models to detect and analyze car hacking behaviors. The Bayesian learning methods can capture the uncertainty of the data and avoid overconfident issues. Moreover, the Bayesian models can provide more information to support the prediction results that can help security engineers further identify the attacks. We have compared our model with deep learning models and the results show the advantages of our proposed model. The code of this work is publicly available

摘要: 随着自动驾驶汽车和联网汽车的兴起，汽车配备了各种设备来辅助司机或支持自动驾驶系统。毫无疑问，汽车已经变得更加智能，因为我们可以在汽车上部署越来越多的设备和软件。因此，汽车中助手和自动驾驶系统的安全成为一个危及生命的问题，因为智能汽车可能会受到恶意攻击，导致交通事故。目前，规范的机器学习和深度学习方法被广泛应用于汽车黑客检测中。然而，机器学习和深度学习方法很容易过于自信，并被精心设计的对抗性例子所击败。此外，这些方法不能为安全工程师提供进一步分析的解释。在这项工作中，我们研究了深度贝叶斯学习模型来检测和分析汽车黑客行为。贝叶斯学习方法可以捕捉数据的不确定性，避免过度自信的问题。此外，贝叶斯模型可以提供更多的信息来支持预测结果，从而帮助安全工程师进一步识别攻击。我们将该模型与深度学习模型进行了比较，结果表明了该模型的优越性。这部作品的代码是公开提供的



## **36. Generation of Wheel Lockup Attacks on Nonlinear Dynamics of Vehicle Traction**

车辆牵引非线性动力学中车轮闭锁攻击的产生 eess.SY

Submitted to American Control Conference 2022 (ACC 2022), 6 pages

**SubmitDate**: 2021-12-16    [paper-pdf](http://arxiv.org/pdf/2112.09229v1)

**Authors**: Alireza Mohammadi, Hafiz Malik, Masoud Abbaszadeh

**Abstracts**: There is ample evidence in the automotive cybersecurity literature that the car brake ECUs can be maliciously reprogrammed. Motivated by such threat, this paper investigates the capabilities of an adversary who can directly control the frictional brake actuators and would like to induce wheel lockup conditions leading to catastrophic road injuries. This paper demonstrates that the adversary despite having a limited knowledge of the tire-road interaction characteristics has the capability of driving the states of the vehicle traction dynamics to a vicinity of the lockup manifold in a finite time by means of a properly designed attack policy for the frictional brakes. This attack policy relies on employing a predefined-time controller and a nonlinear disturbance observer acting on the wheel slip error dynamics. Simulations under various road conditions demonstrate the effectiveness of the proposed attack policy.

摘要: 汽车网络安全文献中有大量证据表明，汽车刹车ECU可以被恶意重新编程。在这种威胁的驱使下，本文调查了一个可以直接控制摩擦制动执行器并想要诱导车轮锁定条件导致灾难性道路伤害的对手的能力。通过合理设计摩擦制动器的攻击策略，证明了敌方尽管对轮胎-路面相互作用特性知之甚少，但仍有能力在有限的时间内将车辆牵引动力学状态驱动到闭锁歧管附近。该攻击策略依赖于采用预定义时间控制器和作用于车轮打滑误差动态的非线性扰动观测器。在不同路况下的仿真实验验证了所提出的攻击策略的有效性。



## **37. All You Need is RAW: Defending Against Adversarial Attacks with Camera Image Pipelines**

您所需要的只是RAW：使用摄像机图像管道防御敌意攻击 cs.CV

**SubmitDate**: 2021-12-16    [paper-pdf](http://arxiv.org/pdf/2112.09219v1)

**Authors**: Yuxuan Zhang, Bo Dong, Felix Heide

**Abstracts**: Existing neural networks for computer vision tasks are vulnerable to adversarial attacks: adding imperceptible perturbations to the input images can fool these methods to make a false prediction on an image that was correctly predicted without the perturbation. Various defense methods have proposed image-to-image mapping methods, either including these perturbations in the training process or removing them in a preprocessing denoising step. In doing so, existing methods often ignore that the natural RGB images in today's datasets are not captured but, in fact, recovered from RAW color filter array captures that are subject to various degradations in the capture. In this work, we exploit this RAW data distribution as an empirical prior for adversarial defense. Specifically, we proposed a model-agnostic adversarial defensive method, which maps the input RGB images to Bayer RAW space and back to output RGB using a learned camera image signal processing (ISP) pipeline to eliminate potential adversarial patterns. The proposed method acts as an off-the-shelf preprocessing module and, unlike model-specific adversarial training methods, does not require adversarial images to train. As a result, the method generalizes to unseen tasks without additional retraining. Experiments on large-scale datasets (e.g., ImageNet, COCO) for different vision tasks (e.g., classification, semantic segmentation, object detection) validate that the method significantly outperforms existing methods across task domains.

摘要: 现有的用于计算机视觉任务的神经网络容易受到敌意攻击：向输入图像添加不可察觉的扰动可以欺骗这些方法在没有扰动的情况下对正确预测的图像进行错误预测。各种防御方法已经提出了图像到图像的映射方法，或者在训练过程中包括这些扰动，或者在预处理去噪步骤中去除它们。在这样做时，现有方法通常忽略没有捕获今天数据集中的自然的rgb图像，而实际上是从在捕获中遭受各种降级的原始颜色过滤阵列捕获中恢复的。在这项工作中，我们利用这个原始数据分布作为对抗防御的经验先验。具体地说，我们提出了一种模型不可知的对抗防御方法，该方法将输入的RGB图像映射到拜耳原始空间，然后使用学习的摄像机图像信号处理(ISP)流水线将输入的RGB图像映射回输出RGB，以消除潜在的敌对模式。该方法作为一个现成的预处理模块，与特定模型的对抗性训练方法不同，不需要对抗性图像进行训练。因此，该方法可以推广到看不见的任务，而不需要额外的再培训。在不同视觉任务(如分类、语义分割、目标检测)的大规模数据集(如ImageNet、CoCo)上的实验验证了该方法在跨任务域的性能上显著优于现有方法。



## **38. Direction-Aggregated Attack for Transferable Adversarial Examples**

可转移对抗性实例的方向聚集攻击 cs.LG

ACM JETC JOURNAL Accepted

**SubmitDate**: 2021-12-16    [paper-pdf](http://arxiv.org/pdf/2104.09172v2)

**Authors**: Tianjin Huang, Vlado Menkovski, Yulong Pei, YuHao Wang, Mykola Pechenizkiy

**Abstracts**: Deep neural networks are vulnerable to adversarial examples that are crafted by imposing imperceptible changes to the inputs. However, these adversarial examples are most successful in white-box settings where the model and its parameters are available. Finding adversarial examples that are transferable to other models or developed in a black-box setting is significantly more difficult. In this paper, we propose the Direction-Aggregated adversarial attacks that deliver transferable adversarial examples. Our method utilizes aggregated direction during the attack process for avoiding the generated adversarial examples overfitting to the white-box model. Extensive experiments on ImageNet show that our proposed method improves the transferability of adversarial examples significantly and outperforms state-of-the-art attacks, especially against adversarial robust models. The best averaged attack success rates of our proposed method reaches 94.6\% against three adversarial trained models and 94.8\% against five defense methods. It also reveals that current defense approaches do not prevent transferable adversarial attacks.

摘要: 深层神经网络很容易受到敌意例子的攻击，这些例子是通过对输入进行潜移默化的改变而精心设计的。然而，这些对抗性的例子在模型及其参数可用的白盒设置中最为成功。寻找可以转移到其他模型或在黑盒环境中开发的对抗性示例要困难得多。在这篇文章中，我们提出了提供可转移的对抗性例子的方向聚集对抗性攻击。我们的方法在攻击过程中利用聚合方向来避免生成的对抗性示例与白盒模型过度拟合。在ImageNet上的大量实验表明，我们提出的方法显著提高了对抗性实例的可移植性，并且优于最新的攻击，特别是针对对抗性健壮性模型的攻击。该方法对3种对抗性训练模型的最优平均攻击成功率为94.6%，对5种防御方法的最优平均攻击成功率为94.8%。它还揭示了当前的防御方法不能阻止可转移的对抗性攻击。



## **39. TAFIM: Targeted Adversarial Attacks against Facial Image Manipulations**

TAFIM：针对面部图像处理的有针对性的敌意攻击 cs.CV

Paper Video: https://youtu.be/btHCrVMKbzw Project Page:  https://shivangi-aneja.github.io/projects/tafim/

**SubmitDate**: 2021-12-16    [paper-pdf](http://arxiv.org/pdf/2112.09151v1)

**Authors**: Shivangi Aneja, Lev Markhasin, Matthias Niessner

**Abstracts**: Face image manipulation methods, despite having many beneficial applications in computer graphics, can also raise concerns by affecting an individual's privacy or spreading disinformation. In this work, we propose a proactive defense to prevent face manipulation from happening in the first place. To this end, we introduce a novel data-driven approach that produces image-specific perturbations which are embedded in the original images. The key idea is that these protected images prevent face manipulation by causing the manipulation model to produce a predefined manipulation target (uniformly colored output image in our case) instead of the actual manipulation. Compared to traditional adversarial attacks that optimize noise patterns for each image individually, our generalized model only needs a single forward pass, thus running orders of magnitude faster and allowing for easy integration in image processing stacks, even on resource-constrained devices like smartphones. In addition, we propose to leverage a differentiable compression approximation, hence making generated perturbations robust to common image compression. We further show that a generated perturbation can simultaneously prevent against multiple manipulation methods.

摘要: 尽管人脸图像处理方法在计算机图形学中有许多有益的应用，但也可能会影响个人隐私或传播虚假信息，从而引起人们的担忧。在这项工作中，我们提出了一种主动防御措施，从一开始就防止面部操纵的发生。为此，我们引入了一种新的数据驱动方法，该方法产生嵌入在原始图像中的特定于图像的扰动。其关键思想是，这些受保护的图像通过使操作模型产生预定义的操作目标(在我们的例子中为均匀着色的输出图像)而不是实际的操作来防止面部操作。与单独优化每个图像的噪声模式的传统对抗性攻击相比，我们的通用模型只需要一次前向传递，因此运行速度快几个数量级，并允许轻松集成到图像处理堆栈中，即使在资源受限的设备(如智能手机)上也是如此。此外，我们建议利用可微压缩近似，从而使生成的扰动对普通图像压缩具有鲁棒性。我们进一步证明了一个产生的扰动可以同时防止多种处理方法。



## **40. Combating Adversaries with Anti-Adversaries**

以反制敌，以反制敌 cs.LG

Accepted to AAAI Conference on Artificial Intelligence (AAAI'22)

**SubmitDate**: 2021-12-16    [paper-pdf](http://arxiv.org/pdf/2103.14347v2)

**Authors**: Motasem Alfarra, Juan C. Pérez, Ali Thabet, Adel Bibi, Philip H. S. Torr, Bernard Ghanem

**Abstracts**: Deep neural networks are vulnerable to small input perturbations known as adversarial attacks. Inspired by the fact that these adversaries are constructed by iteratively minimizing the confidence of a network for the true class label, we propose the anti-adversary layer, aimed at countering this effect. In particular, our layer generates an input perturbation in the opposite direction of the adversarial one and feeds the classifier a perturbed version of the input. Our approach is training-free and theoretically supported. We verify the effectiveness of our approach by combining our layer with both nominally and robustly trained models and conduct large-scale experiments from black-box to adaptive attacks on CIFAR10, CIFAR100, and ImageNet. Our layer significantly enhances model robustness while coming at no cost on clean accuracy.

摘要: 深度神经网络容易受到被称为对抗性攻击的小输入扰动的影响。受这些敌手是通过迭代最小化网络对真实类标签的置信度来构建的事实的启发，我们提出了反敌手层，旨在对抗这一影响。具体地说，我们的层在对抗性的输入扰动的相反方向上生成输入扰动，并向分类器提供输入的扰动版本。我们的方法是免培训的，理论上是有支持的。我们通过将我们的层与名义上和鲁棒训练的模型相结合来验证我们的方法的有效性，并对CIFAR10、CIFAR100和ImageNet进行了从黑盒到自适应攻击的大规模实验。我们的层极大地增强了模型的健壮性，同时在干净的精确度上没有任何代价。



## **41. Anti-Tamper Radio: System-Level Tamper Detection for Computing Systems**

防篡改无线电：面向计算系统的系统级篡改检测 cs.CR

**SubmitDate**: 2021-12-16    [paper-pdf](http://arxiv.org/pdf/2112.09014v1)

**Authors**: Paul Staat, Johannes Tobisch, Christian Zenger, Christof Paar

**Abstracts**: A whole range of attacks becomes possible when adversaries gain physical access to computing systems that process or contain sensitive data. Examples include side-channel analysis, bus probing, device cloning, or implanting hardware Trojans. Defending against these kinds of attacks is considered a challenging endeavor, requiring anti-tamper solutions to monitor the physical environment of the system. Current solutions range from simple switches, which detect if a case is opened, to meshes of conducting material that provide more fine-grained detection of integrity violations. However, these solutions suffer from an intricate trade-off between physical security on the one side and reliability, cost, and difficulty to manufacture on the other. In this work, we demonstrate that radio wave propagation in an enclosed system of complex geometry is sensitive against adversarial physical manipulation. We present an anti-tamper radio (ATR) solution as a method for tamper detection, which combines high detection sensitivity and reliability with ease-of-use. ATR constantly monitors the wireless signal propagation behavior within the boundaries of a metal case. Tamper attempts such as insertion of foreign objects, will alter the observed radio signal response, subsequently raising an alarm. The ATR principle is applicable in many computing systems that require physical security such as servers, ATMs, and smart meters. As a case study, we use 19" servers and thoroughly investigate capabilities and limits of the ATR. Using a custom-built automated probing station, we simulate probing attacks by inserting needles with high precision into protected environments. Our experimental results show that our ATR implementation can detect 16 mm insertions of needles of diameter as low as 0.1 mm under ideal conditions. In the more realistic environment of a running 19" server, we demonstrate reliable [...]

摘要: 当攻击者获得对处理或包含敏感数据的计算系统的物理访问权限时，整个范围的攻击都成为可能。示例包括侧信道分析、总线探测、设备克隆或植入硬件特洛伊木马。防御这类攻击被认为是一项具有挑战性的工作，需要防篡改解决方案来监控系统的物理环境。目前的解决方案范围很广，从检测案件是否打开的简单开关，到提供更细粒度的完整性违规检测的导电材料网状结构。然而，这些解决方案需要在物理安全性和可靠性、成本以及制造难度之间进行复杂的权衡。在这项工作中，我们证明了无线电波在复杂几何封闭系统中的传播对敌方的物理操纵是敏感的。我们提出了一种防篡改无线电(ATR)解决方案，作为篡改检测的一种方法，它结合了高检测灵敏度和可靠性以及易用性。ATR持续监控金属外壳边界内的无线信号传播行为。诸如插入异物之类的篡改尝试将改变观测到的无线电信号响应，随后发出警报。ATR原则适用于许多需要物理安全的计算系统，如服务器、自动取款机和智能电表。作为案例研究，我们使用了19“服务器，并深入研究了ATR的能力和局限性。使用定制的自动探测站，我们通过在受保护环境中插入高精度的针来模拟探测攻击。我们的实验结果表明，在理想条件下，我们的ATR实现可以检测到直径低至0.1 mm的16 mm插入针。在运行19”服务器的更真实的环境中，我们展示了可靠的[.]



## **42. A Heterogeneous Graph Learning Model for Cyber-Attack Detection**

一种用于网络攻击检测的异构图学习模型 cs.CR

12pages,7figures,40 references

**SubmitDate**: 2021-12-16    [paper-pdf](http://arxiv.org/pdf/2112.08986v1)

**Authors**: Mingqi Lv, Chengyu Dong, Tieming Chen, Tiantian Zhu, Qijie Song, Yuan Fan

**Abstracts**: A cyber-attack is a malicious attempt by experienced hackers to breach the target information system. Usually, the cyber-attacks are characterized as hybrid TTPs (Tactics, Techniques, and Procedures) and long-term adversarial behaviors, making the traditional intrusion detection methods ineffective. Most existing cyber-attack detection systems are implemented based on manually designed rules by referring to domain knowledge (e.g., threat models, threat intelligences). However, this process is lack of intelligence and generalization ability. Aiming at this limitation, this paper proposes an intelligent cyber-attack detection method based on provenance data. To effective and efficient detect cyber-attacks from a huge number of system events in the provenance data, we firstly model the provenance data by a heterogeneous graph to capture the rich context information of each system entities (e.g., process, file, socket, etc.), and learns a semantic vector representation for each system entity. Then, we perform online cyber-attack detection by sampling a small and compact local graph from the heterogeneous graph, and classifying the key system entities as malicious or benign. We conducted a series of experiments on two provenance datasets with real cyber-attacks. The experiment results show that the proposed method outperforms other learning based detection models, and has competitive performance against state-of-the-art rule based cyber-attack detection systems.

摘要: 网络攻击是有经验的黑客恶意尝试入侵目标信息系统。通常情况下，网络攻击的特点是战术、技术和过程的混合性和长期的敌意行为，使得传统的入侵检测方法失效。大多数现有的网络攻击检测系统都是基于人工设计的规则，通过参考领域知识(例如，威胁模型、威胁情报)来实现的。然而，这一过程缺乏智能性和泛化能力。针对这一局限性，本文提出了一种基于起源数据的智能网络攻击检测方法。为了有效、高效地从起源数据中的大量系统事件中检测网络攻击，我们首先用异构图对起源数据进行建模，以获取每个系统实体(如进程、文件、套接字等)丰富的上下文信息，并学习每个系统实体的语义向量表示。然后，通过从异构图中抽取一个小而紧凑的局部图进行在线网络攻击检测，并对关键系统实体进行恶意或良性分类。我们在两个具有真实网络攻击的来源数据集上进行了一系列的实验。实验结果表明，该方法的性能优于其他基于学习的检测模型，与目前最新的基于规则的网络攻击检测系统相比，具有好胜的性能。



## **43. Finding Optimal Tangent Points for Reducing Distortions of Hard-label Attacks**

寻找最优切点以减少硬标签攻击的失真 cs.CV

Accepted at NeurIPS 2021, including the appendix. In the previous  versions (v1 and v2), the experimental results of Table 10 are incorrect and  have been corrected in the current version

**SubmitDate**: 2021-12-16    [paper-pdf](http://arxiv.org/pdf/2111.07492v3)

**Authors**: Chen Ma, Xiangyu Guo, Li Chen, Jun-Hai Yong, Yisen Wang

**Abstracts**: One major problem in black-box adversarial attacks is the high query complexity in the hard-label attack setting, where only the top-1 predicted label is available. In this paper, we propose a novel geometric-based approach called Tangent Attack (TA), which identifies an optimal tangent point of a virtual hemisphere located on the decision boundary to reduce the distortion of the attack. Assuming the decision boundary is locally flat, we theoretically prove that the minimum $\ell_2$ distortion can be obtained by reaching the decision boundary along the tangent line passing through such tangent point in each iteration. To improve the robustness of our method, we further propose a generalized method which replaces the hemisphere with a semi-ellipsoid to adapt to curved decision boundaries. Our approach is free of hyperparameters and pre-training. Extensive experiments conducted on the ImageNet and CIFAR-10 datasets demonstrate that our approach can consume only a small number of queries to achieve the low-magnitude distortion. The implementation source code is released online at https://github.com/machanic/TangentAttack.

摘要: 黑盒对抗性攻击的一个主要问题是硬标签攻击设置中的高查询复杂度，在硬标签攻击设置中，只有前1个预测标签可用。本文提出了一种新的基于几何的切线攻击方法(TA)，该方法识别位于决策边界上的虚拟半球的最佳切点，以减少攻击的失真。假设决策边界是局部平坦的，我们从理论上证明了在每一次迭代中，沿着通过该切点的切线到达决策边界可以获得最小的$\\ell2$失真。为了提高方法的鲁棒性，我们进一步提出了一种广义方法，用半椭球代替半球，以适应弯曲的决策边界。我们的方法没有超参数和预训练。在ImageNet和CIFAR-10数据集上进行的大量实验表明，我们的方法可以只消耗少量的查询来实现低幅度的失真。实现源代码在https://github.com/machanic/TangentAttack.上在线发布



## **44. Addressing Adversarial Machine Learning Attacks in Smart Healthcare Perspectives**

从智能医疗的角度解决对抗性机器学习攻击 cs.DC

**SubmitDate**: 2021-12-16    [paper-pdf](http://arxiv.org/pdf/2112.08862v1)

**Authors**: Arawinkumaar Selvakkumar, Shantanu Pal, Zahra Jadidi

**Abstracts**: Smart healthcare systems are gaining popularity with the rapid development of intelligent sensors, the Internet of Things (IoT) applications and services, and wireless communications. However, at the same time, several vulnerabilities and adversarial attacks make it challenging for a safe and secure smart healthcare system from a security point of view. Machine learning has been used widely to develop suitable models to predict and mitigate attacks. Still, the attacks could trick the machine learning models and misclassify outputs generated by the model. As a result, it leads to incorrect decisions, for example, false disease detection and wrong treatment plans for patients. In this paper, we address the type of adversarial attacks and their impact on smart healthcare systems. We propose a model to examine how adversarial attacks impact machine learning classifiers. To test the model, we use a medical image dataset. Our model can classify medical images with high accuracy. We then attacked the model with a Fast Gradient Sign Method attack (FGSM) to cause the model to predict the images and misclassify them inaccurately. Using transfer learning, we train a VGG-19 model with the medical dataset and later implement the FGSM to the Convolutional Neural Network (CNN) to examine the significant impact it causes on the performance and accuracy of the machine learning model. Our results demonstrate that the adversarial attack misclassifies the images, causing the model's accuracy rate to drop from 88% to 11%.

摘要: 随着智能传感器、物联网(IoT)应用和服务以及无线通信的快速发展，智能医疗系统越来越受欢迎。然而，与此同时，几个漏洞和对抗性攻击从安全角度对安全、安全的智能医疗系统提出了挑战。机器学习已被广泛用于开发合适的模型来预测和减轻攻击。尽管如此，这些攻击可能会欺骗机器学习模型，并对模型生成的输出进行错误分类。因此，它会导致错误的决定，例如，对患者的错误疾病检测和错误的治疗计划。在本文中，我们讨论了敌意攻击的类型及其对智能医疗系统的影响。我们提出了一个模型来检验敌意攻击如何影响机器学习分类器。为了测试该模型，我们使用了一个医学图像数据集。该模型能够对医学图像进行高精度的分类。然后，我们利用快速梯度符号方法(FGSM)攻击该模型，使该模型预测图像并对其进行错误分类。利用转移学习，我们用医学数据集训练了一个VGG-19模型，然后将FGSM实现到卷积神经网络(CNN)中，以检验它对机器学习模型的性能和精度造成的显著影响。实验结果表明，敌意攻击导致图像分类错误，导致模型的准确率从88%下降到11%。



## **45. Towards Robust Neural Image Compression: Adversarial Attack and Model Finetuning**

面向鲁棒神经图像压缩：对抗性攻击与模型优化 cs.CV

**SubmitDate**: 2021-12-16    [paper-pdf](http://arxiv.org/pdf/2112.08691v1)

**Authors**: Tong Chen, Zhan Ma

**Abstracts**: Deep neural network based image compression has been extensively studied. Model robustness is largely overlooked, though it is crucial to service enabling. We perform the adversarial attack by injecting a small amount of noise perturbation to original source images, and then encode these adversarial examples using prevailing learnt image compression models. Experiments report severe distortion in the reconstruction of adversarial examples, revealing the general vulnerability of existing methods, regardless of the settings used in underlying compression model (e.g., network architecture, loss function, quality scale) and optimization strategy used for injecting perturbation (e.g., noise threshold, signal distance measurement). Later, we apply the iterative adversarial finetuning to refine pretrained models. In each iteration, random source images and adversarial examples are mixed to update underlying model. Results show the effectiveness of the proposed finetuning strategy by substantially improving the compression model robustness. Overall, our methodology is simple, effective, and generalizable, making it attractive for developing robust learnt image compression solution. All materials have been made publicly accessible at https://njuvision.github.io/RobustNIC for reproducible research.

摘要: 基于深度神经网络的图像压缩已经得到了广泛的研究。模型健壮性在很大程度上被忽视了，尽管它对服务启用至关重要。我们通过在原始源图像中注入少量的噪声扰动来执行对抗性攻击，然后使用主流的学习图像压缩模型对这些对抗性示例进行编码。实验报告了对抗性示例的重建中的严重失真，揭示了现有方法的一般脆弱性，而与底层压缩模型(例如，网络架构、损失函数、质量尺度)和用于注入扰动的优化策略(例如，噪声阈值、信号距离测量)中使用的设置无关。然后，我们应用迭代对抗性精调来精炼预先训练的模型。在每一次迭代中，随机源图像和对抗性示例混合在一起来更新底层模型。结果表明，所提出的微调策略有效地提高了压缩模型的鲁棒性。总体而言，我们的方法简单、有效、通用性强，对于开发健壮的学习图像压缩解决方案很有吸引力。所有材料都已在https://njuvision.github.io/RobustNIC上公开访问，以进行可重现的研究。



## **46. Model Stealing Attacks Against Inductive Graph Neural Networks**

针对归纳图神经网络的模型窃取攻击 cs.CR

To Appear in the 43rd IEEE Symposium on Security and Privacy, May  22-26, 2022

**SubmitDate**: 2021-12-15    [paper-pdf](http://arxiv.org/pdf/2112.08331v1)

**Authors**: Yun Shen, Xinlei He, Yufei Han, Yang Zhang

**Abstracts**: Many real-world data come in the form of graphs. Graph neural networks (GNNs), a new family of machine learning (ML) models, have been proposed to fully leverage graph data to build powerful applications. In particular, the inductive GNNs, which can generalize to unseen data, become mainstream in this direction. Machine learning models have shown great potential in various tasks and have been deployed in many real-world scenarios. To train a good model, a large amount of data as well as computational resources are needed, leading to valuable intellectual property. Previous research has shown that ML models are prone to model stealing attacks, which aim to steal the functionality of the target models. However, most of them focus on the models trained with images and texts. On the other hand, little attention has been paid to models trained with graph data, i.e., GNNs. In this paper, we fill the gap by proposing the first model stealing attacks against inductive GNNs. We systematically define the threat model and propose six attacks based on the adversary's background knowledge and the responses of the target models. Our evaluation on six benchmark datasets shows that the proposed model stealing attacks against GNNs achieve promising performance.

摘要: 许多现实世界的数据都是以图表的形式出现的。图神经网络(GNNs)是一类新的机器学习(ML)模型，被提出用来充分利用图数据来构建功能强大的应用程序。特别是，可以推广到不可见数据的感应式GNN成为这一方向的主流。机器学习模型已经在各种任务中显示出巨大的潜力，并已被部署在许多现实场景中。要训练一个好的模型，需要大量的数据和计算资源，从而产生宝贵的知识产权。以往的研究表明，ML模型容易受到模型窃取攻击，目的是窃取目标模型的功能。然而，它们大多集中在用图像和文本训练的模型上。另一方面，很少有人关注用图形数据训练的模型，即GNN。在本文中，我们提出了第一个针对感应性GNNs的窃取攻击模型，填补了这一空白。我们系统地定义了威胁模型，并根据对手的背景知识和目标模型的响应提出了六种攻击。我们在六个基准数据集上的评估表明，所提出的针对GNNs的窃取攻击模型取得了令人满意的性能。



## **47. Meta Adversarial Perturbations**

元对抗扰动 cs.LG

Published in AAAI 2022 Workshop

**SubmitDate**: 2021-12-15    [paper-pdf](http://arxiv.org/pdf/2111.10291v2)

**Authors**: Chia-Hung Yuan, Pin-Yu Chen, Chia-Mu Yu

**Abstracts**: A plethora of attack methods have been proposed to generate adversarial examples, among which the iterative methods have been demonstrated the ability to find a strong attack. However, the computation of an adversarial perturbation for a new data point requires solving a time-consuming optimization problem from scratch. To generate a stronger attack, it normally requires updating a data point with more iterations. In this paper, we show the existence of a meta adversarial perturbation (MAP), a better initialization that causes natural images to be misclassified with high probability after being updated through only a one-step gradient ascent update, and propose an algorithm for computing such perturbations. We conduct extensive experiments, and the empirical results demonstrate that state-of-the-art deep neural networks are vulnerable to meta perturbations. We further show that these perturbations are not only image-agnostic, but also model-agnostic, as a single perturbation generalizes well across unseen data points and different neural network architectures.

摘要: 已经提出了大量的攻击方法来生成对抗性实例，其中迭代方法已被证明具有发现强攻击的能力。然而，计算新数据点的对抗性扰动需要从头开始解决耗时的优化问题。要生成更强的攻击，通常需要更新迭代次数更多的数据点。本文证明了元对抗扰动(MAP)的存在性，并提出了一种计算这种扰动的算法。MAP是一种较好的初始化方法，它只通过一步梯度上升更新就会导致自然图像在更新后被高概率地误分类。我们进行了大量的实验，实验结果表明，最新的深度神经网络容易受到元扰动的影响。我们进一步表明，这些扰动不仅是图像不可知的，而且也是模型不可知的，因为单个扰动很好地概括了不可见的数据点和不同的神经网络结构。



## **48. Temporal Shuffling for Defending Deep Action Recognition Models against Adversarial Attacks**

用于防御敌方攻击的深层动作识别模型的时间洗牌 cs.CV

**SubmitDate**: 2021-12-15    [paper-pdf](http://arxiv.org/pdf/2112.07921v1)

**Authors**: Jaehui Hwang, Huan Zhang, Jun-Ho Choi, Cho-Jui Hsieh, Jong-Seok Lee

**Abstracts**: Recently, video-based action recognition methods using convolutional neural networks (CNNs) achieve remarkable recognition performance. However, there is still lack of understanding about the generalization mechanism of action recognition models. In this paper, we suggest that action recognition models rely on the motion information less than expected, and thus they are robust to randomization of frame orders. Based on this observation, we develop a novel defense method using temporal shuffling of input videos against adversarial attacks for action recognition models. Another observation enabling our defense method is that adversarial perturbations on videos are sensitive to temporal destruction. To the best of our knowledge, this is the first attempt to design a defense method specific to video-based action recognition models.

摘要: 近年来，基于卷积神经网络(CNNs)的视频动作识别方法取得了显著的识别效果。然而，对动作识别模型的泛化机制还缺乏了解。本文提出动作识别模型对运动信息的依赖程度低于预期，因而对帧阶数的随机化具有较强的鲁棒性。基于这一观察结果，我们提出了一种新的行为识别模型的防御方法，该方法利用输入视频的时间洗牌来抵御敌意攻击。支持我们防御方法的另一个观察结果是，视频上的对抗性扰动对时间破坏很敏感。据我们所知，这是首次尝试设计专门针对基于视频的动作识别模型的防御方法。



## **49. Adversarial Examples for Extreme Multilabel Text Classification**

极端多标签文本分类的对抗性实例 cs.LG

**SubmitDate**: 2021-12-14    [paper-pdf](http://arxiv.org/pdf/2112.07512v1)

**Authors**: Mohammadreza Qaraei, Rohit Babbar

**Abstracts**: Extreme Multilabel Text Classification (XMTC) is a text classification problem in which, (i) the output space is extremely large, (ii) each data point may have multiple positive labels, and (iii) the data follows a strongly imbalanced distribution. With applications in recommendation systems and automatic tagging of web-scale documents, the research on XMTC has been focused on improving prediction accuracy and dealing with imbalanced data. However, the robustness of deep learning based XMTC models against adversarial examples has been largely underexplored.   In this paper, we investigate the behaviour of XMTC models under adversarial attacks. To this end, first, we define adversarial attacks in multilabel text classification problems. We categorize attacking multilabel text classifiers as (a) positive-targeted, where the target positive label should fall out of top-k predicted labels, and (b) negative-targeted, where the target negative label should be among the top-k predicted labels. Then, by experiments on APLC-XLNet and AttentionXML, we show that XMTC models are highly vulnerable to positive-targeted attacks but more robust to negative-targeted ones. Furthermore, our experiments show that the success rate of positive-targeted adversarial attacks has an imbalanced distribution. More precisely, tail classes are highly vulnerable to adversarial attacks for which an attacker can generate adversarial samples with high similarity to the actual data-points. To overcome this problem, we explore the effect of rebalanced loss functions in XMTC where not only do they increase accuracy on tail classes, but they also improve the robustness of these classes against adversarial attacks. The code for our experiments is available at https://github.com/xmc-aalto/adv-xmtc

摘要: 极端多标签文本分类(XMTC)是一个文本分类问题，其中(I)输出空间非常大，(Ii)每个数据点可能有多个正标签，(Iii)数据服从强不平衡分布。随着XMTC在推荐系统和Web文档自动标注中的应用，XMTC的研究重点放在提高预测精度和处理不平衡数据上。然而，基于深度学习的XMTC模型对敌意示例的稳健性研究还很少。本文研究了XMTC模型在对抗性攻击下的行为。为此，我们首先定义了多标签文本分类问题中的对抗性攻击。我们将攻击多标签文本分类器分为(A)正向目标，其中目标正向标签应该落在前k个预测标签之外；(B)负向目标，其中目标负向标签应该在前k个预测标签中。然后，通过在APLC-XLNet和AttentionXML上的实验表明，XMTC模型对正目标攻击具有很强的脆弱性，但对负目标攻击具有较强的鲁棒性。此外，我们的实验表明，正向对抗性攻击的成功率分布不均衡。更准确地说，Tail类非常容易受到敌意攻击，对于这种攻击，攻击者可以生成与实际数据点高度相似的对抗性样本。为了克服这个问题，我们探索了XMTC中重新平衡损失函数的效果，在XMTC中，它们不仅提高了尾类的准确性，而且还提高了这些类对对手攻击的鲁棒性。我们实验的代码可以在https://github.com/xmc-aalto/adv-xmtc上找到



## **50. Multi-Leader Congestion Games with an Adversary**

有对手的多队长拥堵对策 cs.GT

**SubmitDate**: 2021-12-14    [paper-pdf](http://arxiv.org/pdf/2112.07435v1)

**Authors**: Tobias Harks, Mona Henle, Max Klimm, Jannik Matuschke, Anja Schedel

**Abstracts**: We study a multi-leader single-follower congestion game where multiple users (leaders) choose one resource out of a set of resources and, after observing the realized loads, an adversary (single-follower) attacks the resources with maximum loads, causing additional costs for the leaders. For the resulting strategic game among the leaders, we show that pure Nash equilibria may fail to exist and therefore, we consider approximate equilibria instead. As our first main result, we show that the existence of a $K$-approximate equilibrium can always be guaranteed, where $K \approx 1.1974$ is the unique solution of a cubic polynomial equation. To this end, we give a polynomial time combinatorial algorithm which computes a $K$-approximate equilibrium. The factor $K$ is tight, meaning that there is an instance that does not admit an $\alpha$-approximate equilibrium for any $\alpha<K$. Thus $\alpha=K$ is the smallest possible value of $\alpha$ such that the existence of an $\alpha$-approximate equilibrium can be guaranteed for any instance of the considered game. Secondly, we focus on approximate equilibria of a given fixed instance. We show how to compute efficiently a best approximate equilibrium, that is, with smallest possible $\alpha$ among all $\alpha$-approximate equilibria of the given instance.

摘要: 研究了一个多领导者单跟随者拥塞博弈，其中多个用户(领导者)从一组资源中选择一个资源，在观察到实现的负载后，一个对手(单一跟随者)攻击具有最大负载的资源，从而给领导者带来额外的成本。对于由此产生的领导者之间的战略博弈，我们表明纯纳什均衡可能不存在，因此，我们考虑近似均衡。作为我们的第一个主要结果，我们证明了$K$-近似均衡的存在性总是可以保证的，其中$K\约1.1974$是一个三次多项式方程的唯一解。为此，我们给出了一个计算$K$-近似均衡的多项式时间组合算法。因子$K$是紧的，这意味着对于任何$\α<K$，都存在一个不允许$\α$-近似均衡的实例。因此，$\α=K$是$\α$的最小可能值，使得对于所考虑的博弈的任何实例，都可以保证存在$\α$-近似均衡。其次，我们重点研究了给定固定实例的近似均衡。我们展示了如何有效地计算最佳近似均衡，即在给定实例的所有$\α$-近似均衡中，具有最小可能的$\α$。



