# Latest Adversarial Attack Papers
**update at 2023-04-10 10:31:53**

翻译来自 https://cloud.tencent.com/document/product/551/15619

## **1. AMS-DRL: Learning Multi-Pursuit Evasion for Safe Targeted Navigation of Drones**

AMS-DRL：用于无人机安全定向导航的学习多目标规避 cs.RO

**SubmitDate**: 2023-04-07    [abs](http://arxiv.org/abs/2304.03443v1) [paper-pdf](http://arxiv.org/pdf/2304.03443v1)

**Authors**: Jiaping Xiao, Mir Feroskhan

**Abstract**: Safe navigation of drones in the presence of adversarial physical attacks from multiple pursuers is a challenging task. This paper proposes a novel approach, asynchronous multi-stage deep reinforcement learning (AMS-DRL), to train an adversarial neural network that can learn from the actions of multiple pursuers and adapt quickly to their behavior, enabling the drone to avoid attacks and reach its target. Our approach guarantees convergence by ensuring Nash Equilibrium among agents from the game-theory analysis. We evaluate our method in extensive simulations and show that it outperforms baselines with higher navigation success rates. We also analyze how parameters such as the relative maximum speed affect navigation performance. Furthermore, we have conducted physical experiments and validated the effectiveness of the trained policies in real-time flights. A success rate heatmap is introduced to elucidate how spatial geometry influences navigation outcomes. Project website: https://github.com/NTU-UAVG/AMS-DRL-for-Pursuit-Evasion.

摘要: 在多个追踪者的敌对物理攻击下，无人机的安全导航是一项具有挑战性的任务。提出了一种新的方法--异步多阶段深度强化学习(AMS-DRL)，用于训练对抗神经网络，该网络能够从多个追赶者的行为中学习并快速适应他们的行为，使无人机能够躲避攻击并到达目标。我们的方法通过从博弈论分析确保代理之间的纳什均衡来确保收敛。我们在大量的仿真中对我们的方法进行了评估，结果表明，它的导航成功率比基线更高。我们还分析了相对最大速度等参数对导航性能的影响。此外，我们进行了物理实验，并在实时飞行中验证了训练后的策略的有效性。介绍了一个成功率热图，以阐明空间几何形状如何影响导航结果。项目网站：https://github.com/NTU-UAVG/AMS-DRL-for-Pursuit-Evasion.



## **2. LP-BFGS attack: An adversarial attack based on the Hessian with limited pixels**

LP-BFGS攻击：一种基于有限像素黑森的对抗性攻击 cs.CR

15 pages, 7 figures

**SubmitDate**: 2023-04-07    [abs](http://arxiv.org/abs/2210.15446v2) [paper-pdf](http://arxiv.org/pdf/2210.15446v2)

**Authors**: Jiebao Zhang, Wenhua Qian, Rencan Nie, Jinde Cao, Dan Xu

**Abstract**: Deep neural networks are vulnerable to adversarial attacks. Most $L_{0}$-norm based white-box attacks craft perturbations by the gradient of models to the input. Since the computation cost and memory limitation of calculating the Hessian matrix, the application of Hessian or approximate Hessian in white-box attacks is gradually shelved. In this work, we note that the sparsity requirement on perturbations naturally lends itself to the usage of Hessian information. We study the attack performance and computation cost of the attack method based on the Hessian with a limited number of perturbation pixels. Specifically, we propose the Limited Pixel BFGS (LP-BFGS) attack method by incorporating the perturbation pixel selection strategy and the BFGS algorithm. Pixels with top-k attribution scores calculated by the Integrated Gradient method are regarded as optimization variables of the LP-BFGS attack. Experimental results across different networks and datasets demonstrate that our approach has comparable attack ability with reasonable computation in different numbers of perturbation pixels compared with existing solutions.

摘要: 深度神经网络很容易受到敌意攻击。大多数基于$L_{0}$范数的白盒通过模型到输入的梯度来攻击手工扰动。由于计算海森矩阵的计算量和内存的限制，海森矩阵或近似海森矩阵在白盒攻击中的应用逐渐被搁置。在这项工作中，我们注意到对扰动的稀疏性要求自然地适合于使用Hessian信息。研究了基于有限扰动像素的Hessian攻击方法的攻击性能和计算代价。将扰动像素选择策略与有限像素BFGS算法相结合，提出了有限像素BFGS(LP-BFGS)攻击方法。将积分梯度法计算出的top-k属性得分的像素作为LP-BFGS攻击的优化变量。在不同网络和数据集上的实验结果表明，该方法在不同扰动像素数下具有与已有方案相当的攻击能力，并具有合理的计算能力。



## **3. EZClone: Improving DNN Model Extraction Attack via Shape Distillation from GPU Execution Profiles**

EZClone：基于形状提取的改进DNN模型提取攻击 cs.LG

11 pages, 6 tables, 4 figures

**SubmitDate**: 2023-04-06    [abs](http://arxiv.org/abs/2304.03388v1) [paper-pdf](http://arxiv.org/pdf/2304.03388v1)

**Authors**: Jonah O'Brien Weiss, Tiago Alves, Sandip Kundu

**Abstract**: Deep Neural Networks (DNNs) have become ubiquitous due to their performance on prediction and classification problems. However, they face a variety of threats as their usage spreads. Model extraction attacks, which steal DNNs, endanger intellectual property, data privacy, and security. Previous research has shown that system-level side-channels can be used to leak the architecture of a victim DNN, exacerbating these risks. We propose two DNN architecture extraction techniques catering to various threat models. The first technique uses a malicious, dynamically linked version of PyTorch to expose a victim DNN architecture through the PyTorch profiler. The second, called EZClone, exploits aggregate (rather than time-series) GPU profiles as a side-channel to predict DNN architecture, employing a simple approach and assuming little adversary capability as compared to previous work. We investigate the effectiveness of EZClone when minimizing the complexity of the attack, when applied to pruned models, and when applied across GPUs. We find that EZClone correctly predicts DNN architectures for the entire set of PyTorch vision architectures with 100% accuracy. No other work has shown this degree of architecture prediction accuracy with the same adversarial constraints or using aggregate side-channel information. Prior work has shown that, once a DNN has been successfully cloned, further attacks such as model evasion or model inversion can be accelerated significantly.

摘要: 深度神经网络(DNN)因其在预测和分类问题上的性能而变得无处不在。然而，随着它们的使用普及，它们面临着各种威胁。模型提取攻击窃取DNN，危害知识产权、数据隐私和安全。先前的研究表明，系统级旁路可用于泄漏受害者DNN的体系结构，从而加剧这些风险。针对不同的威胁模型，我们提出了两种DNN结构提取技术。第一种技术使用恶意的动态链接版本的PyTorch，通过PyTorch分析器暴露受攻击的DNN体系结构。第二种称为EZClone，它利用聚合的(而不是时间序列的)GPU配置文件作为侧通道来预测DNN体系结构，采用了一种简单的方法，与以前的工作相比，假设的对手能力很小。我们研究了EZClone在将攻击的复杂性降至最低时的有效性，当应用于修剪的模型时，以及当应用于GPU时。我们发现，EZClone能够100%准确地预测整套PyTorch VISION体系结构的DNN体系结构。没有其他工作表明，在相同的对抗性约束下或使用聚合的旁路信息，体系结构预测的准确性达到了这种程度。以前的工作表明，一旦DNN被成功克隆，进一步的攻击，如模型逃避或模型反转，可以显著加速。



## **4. Reliable Learning for Test-time Attacks and Distribution Shift**

针对测试时间攻击和分布转移的可靠学习 cs.LG

**SubmitDate**: 2023-04-06    [abs](http://arxiv.org/abs/2304.03370v1) [paper-pdf](http://arxiv.org/pdf/2304.03370v1)

**Authors**: Maria-Florina Balcan, Steve Hanneke, Rattana Pukdee, Dravyansh Sharma

**Abstract**: Machine learning algorithms are often used in environments which are not captured accurately even by the most carefully obtained training data, either due to the possibility of `adversarial' test-time attacks, or on account of `natural' distribution shift. For test-time attacks, we introduce and analyze a novel robust reliability guarantee, which requires a learner to output predictions along with a reliability radius $\eta$, with the meaning that its prediction is guaranteed to be correct as long as the adversary has not perturbed the test point farther than a distance $\eta$. We provide learners that are optimal in the sense that they always output the best possible reliability radius on any test point, and we characterize the reliable region, i.e. the set of points where a given reliability radius is attainable. We additionally analyze reliable learners under distribution shift, where the test points may come from an arbitrary distribution Q different from the training distribution P. For both cases, we bound the probability mass of the reliable region for several interesting examples, for linear separators under nearly log-concave and s-concave distributions, as well as for smooth boundary classifiers under smooth probability distributions.

摘要: 机器学习算法通常用于即使是通过最仔细地获得的训练数据也不能准确捕获的环境中，这要么是因为可能发生对抗性的测试时间攻击，要么是由于“自然的”分布偏移。对于测试时间攻击，我们引入并分析了一种新的稳健可靠性保证，它要求学习者输出预测和可靠性半径，即只要对手没有干扰测试点超过一段距离，它的预测就保证是正确的。我们提供的学习器在某种意义上是最优的，即他们总是在任何测试点上输出最佳可能的可靠性半径，并且我们刻画了可靠区域，即在给定可靠性半径可达到的点集。此外，我们还分析了分布漂移下的可靠学习器，其中测试点可能来自与训练分布P不同的任意分布Q。对于这两种情况，我们对几个有趣的例子、近对数凹分布和s凹分布下的线性分离器以及光滑概率分布下的光滑边界分类器，给出了可靠区域的概率质量。



## **5. Improving Visual Question Answering Models through Robustness Analysis and In-Context Learning with a Chain of Basic Questions**

通过稳健性分析和带基本问题链的情境学习改进视觉问答模型 cs.CV

28 pages

**SubmitDate**: 2023-04-06    [abs](http://arxiv.org/abs/2304.03147v1) [paper-pdf](http://arxiv.org/pdf/2304.03147v1)

**Authors**: Jia-Hong Huang, Modar Alfadly, Bernard Ghanem, Marcel Worring

**Abstract**: Deep neural networks have been critical in the task of Visual Question Answering (VQA), with research traditionally focused on improving model accuracy. Recently, however, there has been a trend towards evaluating the robustness of these models against adversarial attacks. This involves assessing the accuracy of VQA models under increasing levels of noise in the input, which can target either the image or the proposed query question, dubbed the main question. However, there is currently a lack of proper analysis of this aspect of VQA. This work proposes a new method that utilizes semantically related questions, referred to as basic questions, acting as noise to evaluate the robustness of VQA models. It is hypothesized that as the similarity of a basic question to the main question decreases, the level of noise increases. To generate a reasonable noise level for a given main question, a pool of basic questions is ranked based on their similarity to the main question, and this ranking problem is cast as a LASSO optimization problem. Additionally, this work proposes a novel robustness measure, R_score, and two basic question datasets to standardize the analysis of VQA model robustness. The experimental results demonstrate that the proposed evaluation method effectively analyzes the robustness of VQA models. Moreover, the experiments show that in-context learning with a chain of basic questions can enhance model accuracy.

摘要: 深度神经网络在视觉问答(VQA)任务中一直是至关重要的，传统上的研究集中在提高模型精度上。然而，最近有一种趋势是评估这些模型对对手攻击的稳健性。这涉及在输入噪声水平不断增加的情况下评估VQA模型的准确性，这可能针对图像或拟议的查询问题，称为主要问题。然而，目前还缺乏对VQA这一方面的适当分析。本文提出了一种新的方法，利用语义相关的问题，即基本问题，作为噪声来评估VQA模型的稳健性。假设基本问题与主要问题的相似度越低，噪音水平就越高。为了为给定的主问题生成合理的噪声水平，根据基本问题池与主问题的相似度对其进行排序，并将该排序问题转换为套索优化问题。此外，本文还提出了一种新的健壮性度量R_Score和两个基本问题数据集来规范VQA模型的健壮性分析。实验结果表明，该评价方法有效地分析了VQA模型的稳健性。此外，实验表明，带有一系列基本问题的情境学习可以提高模型的准确性。



## **6. Public Key Encryption with Secure Key Leasing**

使用安全密钥租赁的公钥加密 quant-ph

68 pages, 4 figures. added related works and a comparison with a  concurrent work (2023-04-07)

**SubmitDate**: 2023-04-06    [abs](http://arxiv.org/abs/2302.11663v2) [paper-pdf](http://arxiv.org/pdf/2302.11663v2)

**Authors**: Shweta Agrawal, Fuyuki Kitagawa, Ryo Nishimaki, Shota Yamada, Takashi Yamakawa

**Abstract**: We introduce the notion of public key encryption with secure key leasing (PKE-SKL). Our notion supports the leasing of decryption keys so that a leased key achieves the decryption functionality but comes with the guarantee that if the quantum decryption key returned by a user passes a validity test, then the user has lost the ability to decrypt. Our notion is similar in spirit to the notion of secure software leasing (SSL) introduced by Ananth and La Placa (Eurocrypt 2021) but captures significantly more general adversarial strategies. In more detail, our adversary is not restricted to use an honest evaluation algorithm to run pirated software. Our results can be summarized as follows:   1. Definitions: We introduce the definition of PKE with secure key leasing and formalize security notions.   2. Constructing PKE with Secure Key Leasing: We provide a construction of PKE-SKL by leveraging a PKE scheme that satisfies a new security notion that we call consistent or inconsistent security against key leasing attacks (CoIC-KLA security). We then construct a CoIC-KLA secure PKE scheme using 1-key Ciphertext-Policy Functional Encryption (CPFE) that in turn can be based on any IND-CPA secure PKE scheme.   3. Identity Based Encryption, Attribute Based Encryption and Functional Encryption with Secure Key Leasing: We provide definitions of secure key leasing in the context of advanced encryption schemes such as identity based encryption (IBE), attribute-based encryption (ABE) and functional encryption (FE). Then we provide constructions by combining the above PKE-SKL with standard IBE, ABE and FE schemes.

摘要: 我们引入了公钥加密和安全密钥租赁(PKE-SKL)的概念。我们的想法支持租赁解密密钥，以便租赁的密钥实现解密功能，但同时也保证，如果用户返回的量子解密密钥通过有效性测试，则用户已失去解密能力。我们的概念在精神上类似于Ananth和La Placa(Eurocrypt 2021)提出的安全软件租赁(SSL)概念，但捕获了更一般的对抗策略。更详细地说，我们的对手并不局限于使用诚实的评估算法来运行盗版软件。定义：引入了具有安全密钥租赁的PKE的定义，并对安全概念进行了形式化描述。2.使用安全密钥租赁构建PKE：利用一种新的PKE方案来构造PKE-SKL，该方案满足一种新的安全概念，即针对密钥租赁攻击的一致或不一致安全(COIC-KLA安全)。然后，我们使用1密钥密文策略函数加密(CPFE)构造了COIC-KLA安全PKE方案，而CPFE又可以基于任何IND-CPA安全PKE方案。3.基于身份的加密、基于属性的加密和基于安全密钥租赁的功能加密：我们在基于身份的加密(IBE)、基于属性的加密(ABE)和功能加密(FE)等高级加密方案的背景下定义了安全密钥租赁。然后，我们将上述PKE-SKL与标准的IBE、ABE和FE格式相结合，给出了构造方法。



## **7. StratDef: Strategic Defense Against Adversarial Attacks in ML-based Malware Detection**

StratDef：基于ML的恶意软件检测中对抗攻击的战略防御 cs.LG

**SubmitDate**: 2023-04-06    [abs](http://arxiv.org/abs/2202.07568v5) [paper-pdf](http://arxiv.org/pdf/2202.07568v5)

**Authors**: Aqib Rashid, Jose Such

**Abstract**: Over the years, most research towards defenses against adversarial attacks on machine learning models has been in the image recognition domain. The malware detection domain has received less attention despite its importance. Moreover, most work exploring these defenses has focused on several methods but with no strategy when applying them. In this paper, we introduce StratDef, which is a strategic defense system based on a moving target defense approach. We overcome challenges related to the systematic construction, selection, and strategic use of models to maximize adversarial robustness. StratDef dynamically and strategically chooses the best models to increase the uncertainty for the attacker while minimizing critical aspects in the adversarial ML domain, like attack transferability. We provide the first comprehensive evaluation of defenses against adversarial attacks on machine learning for malware detection, where our threat model explores different levels of threat, attacker knowledge, capabilities, and attack intensities. We show that StratDef performs better than other defenses even when facing the peak adversarial threat. We also show that, of the existing defenses, only a few adversarially-trained models provide substantially better protection than just using vanilla models but are still outperformed by StratDef.

摘要: 多年来，针对机器学习模型的敌意攻击防御的研究大多集中在图像识别领域。恶意软件检测领域尽管很重要，但受到的关注较少。此外，大多数探索这些防御措施的工作都集中在几种方法上，但在应用这些方法时没有策略。本文介绍了一种基于移动目标防御方法的战略防御系统StratDef。我们克服了与模型的系统构建、选择和战略使用相关的挑战，以最大限度地提高对手的稳健性。StratDef动态和战略性地选择最佳模型，以增加攻击者的不确定性，同时最小化敌对ML领域的关键方面，如攻击可转移性。我们提供了针对恶意软件检测的机器学习的首次全面防御评估，其中我们的威胁模型探索了不同级别的威胁、攻击者知识、能力和攻击强度。我们表明，即使在面临最大的对手威胁时，StratDef也比其他防御系统表现得更好。我们还表明，在现有的防御系统中，只有少数经过对抗性训练的模型提供了比仅仅使用普通模型更好的保护，但仍然优于StratDef。



## **8. PAD: Towards Principled Adversarial Malware Detection Against Evasion Attacks**

PAD：针对逃避攻击的原则性恶意软件检测 cs.CR

Accepted by IEEE Transactions on Dependable and Secure Computing; To  appear

**SubmitDate**: 2023-04-06    [abs](http://arxiv.org/abs/2302.11328v2) [paper-pdf](http://arxiv.org/pdf/2302.11328v2)

**Authors**: Deqiang Li, Shicheng Cui, Yun Li, Jia Xu, Fu Xiao, Shouhuai Xu

**Abstract**: Machine Learning (ML) techniques can facilitate the automation of malicious software (malware for short) detection, but suffer from evasion attacks. Many studies counter such attacks in heuristic manners, lacking theoretical guarantees and defense effectiveness. In this paper, we propose a new adversarial training framework, termed Principled Adversarial Malware Detection (PAD), which offers convergence guarantees for robust optimization methods. PAD lays on a learnable convex measurement that quantifies distribution-wise discrete perturbations to protect malware detectors from adversaries, whereby for smooth detectors, adversarial training can be performed with theoretical treatments. To promote defense effectiveness, we propose a new mixture of attacks to instantiate PAD to enhance deep neural network-based measurements and malware detectors. Experimental results on two Android malware datasets demonstrate: (i) the proposed method significantly outperforms the state-of-the-art defenses; (ii) it can harden ML-based malware detection against 27 evasion attacks with detection accuracies greater than 83.45%, at the price of suffering an accuracy decrease smaller than 2.16% in the absence of attacks; (iii) it matches or outperforms many anti-malware scanners in VirusTotal against realistic adversarial malware.

摘要: 机器学习(ML)技术可以促进恶意软件(简称恶意软件)检测的自动化，但受到逃避攻击。许多研究以启发式的方式对抗这种攻击，缺乏理论保障和防御有效性。本文提出了一种新的对抗性训练框架，称为原则性对抗性恶意软件检测(PAD)，它为稳健优化方法提供了收敛保证。PAD建立在可学习的凸度量上，该度量量化了分布方向的离散扰动，以保护恶意软件检测器免受对手的攻击，从而对于平滑的检测器，可以通过理论处理来执行对抗性训练。为了提高防御效果，我们提出了一种新的混合攻击来实例化PAD，以增强基于深度神经网络的测量和恶意软件检测。在两个Android恶意软件数据集上的实验结果表明：(I)该方法的性能明显优于最新的防御方法；(Ii)它可以强化基于ML的恶意软件检测，对27次逃避攻击的检测准确率超过83.45%，而在没有攻击的情况下，检测准确率下降小于2.16%；(Iii)它与VirusTotal中的许多反恶意软件扫描器相匹配或优于对现实恶意软件的检测。



## **9. Robust Neural Architecture Search**

稳健的神经结构搜索 cs.LG

**SubmitDate**: 2023-04-06    [abs](http://arxiv.org/abs/2304.02845v1) [paper-pdf](http://arxiv.org/pdf/2304.02845v1)

**Authors**: Xunyu Zhu, Jian Li, Yong Liu, Weiping Wang

**Abstract**: Neural Architectures Search (NAS) becomes more and more popular over these years. However, NAS-generated models tends to suffer greater vulnerability to various malicious attacks. Lots of robust NAS methods leverage adversarial training to enhance the robustness of NAS-generated models, however, they neglected the nature accuracy of NAS-generated models. In our paper, we propose a novel NAS method, Robust Neural Architecture Search (RNAS). To design a regularization term to balance accuracy and robustness, RNAS generates architectures with both high accuracy and good robustness. To reduce search cost, we further propose to use noise examples instead adversarial examples as input to search architectures. Extensive experiments show that RNAS achieves state-of-the-art (SOTA) performance on both image classification and adversarial attacks, which illustrates the proposed RNAS achieves a good tradeoff between robustness and accuracy.

摘要: 近年来，神经结构搜索(NAS)变得越来越流行。然而，NAS生成的模型往往更容易受到各种恶意攻击。许多健壮的NAS方法利用对抗性训练来增强NAS生成的模型的健壮性，然而，它们忽略了NAS生成的模型的自然准确性。在本文中，我们提出了一种新的NAS方法--稳健神经结构搜索(RNAS)。为了设计一个正则化项来平衡精度和健壮性，RNAS生成具有高精度和良好健壮性的体系结构。为了降低搜索成本，我们进一步建议使用噪声示例而不是对抗性示例作为搜索架构的输入。大量的实验表明，RNAS在图像分类和敌意攻击方面都达到了最好的性能，说明了RNAS在稳健性和准确性之间取得了很好的折衷。



## **10. Robust Upper Bounds for Adversarial Training**

对抗性训练的稳健上界 cs.LG

**SubmitDate**: 2023-04-06    [abs](http://arxiv.org/abs/2112.09279v2) [paper-pdf](http://arxiv.org/pdf/2112.09279v2)

**Authors**: Dimitris Bertsimas, Xavier Boix, Kimberly Villalobos Carballo, Dick den Hertog

**Abstract**: Many state-of-the-art adversarial training methods for deep learning leverage upper bounds of the adversarial loss to provide security guarantees against adversarial attacks. Yet, these methods rely on convex relaxations to propagate lower and upper bounds for intermediate layers, which affect the tightness of the bound at the output layer. We introduce a new approach to adversarial training by minimizing an upper bound of the adversarial loss that is based on a holistic expansion of the network instead of separate bounds for each layer. This bound is facilitated by state-of-the-art tools from Robust Optimization; it has closed-form and can be effectively trained using backpropagation. We derive two new methods with the proposed approach. The first method (Approximated Robust Upper Bound or aRUB) uses the first order approximation of the network as well as basic tools from Linear Robust Optimization to obtain an empirical upper bound of the adversarial loss that can be easily implemented. The second method (Robust Upper Bound or RUB), computes a provable upper bound of the adversarial loss. Across a variety of tabular and vision data sets we demonstrate the effectiveness of our approach -- RUB is substantially more robust than state-of-the-art methods for larger perturbations, while aRUB matches the performance of state-of-the-art methods for small perturbations.

摘要: 许多先进的深度学习对抗性训练方法利用对抗性损失的上限来提供针对对抗性攻击的安全保证。然而，这些方法依赖于凸松弛来传播中间层的下界和上界，这影响了输出层上界的紧密性。我们引入了一种新的对抗性训练方法，通过最小化对抗性损失的上界，该上界基于网络的整体扩展，而不是针对每一层单独的界。这一界限是由稳健优化的最先进工具促成的；它具有封闭的形式，可以使用反向传播进行有效的训练。利用提出的方法，我们得到了两种新的方法。第一种方法(近似稳健上界或ARUB)利用网络的一阶近似以及线性稳健优化的基本工具来获得易于实现的对手损失的经验上界。第二种方法(稳健上界或RUB)，计算对手损失的可证明上界。在各种表格和视觉数据集上，我们证明了我们方法的有效性--对于较大的扰动，RUB比最先进的方法更健壮，而对于较小的扰动，ABRUB的性能与最先进的方法相当。



## **11. Improving Fast Adversarial Training with Prior-Guided Knowledge**

利用先验指导知识改进快速对抗训练 cs.LG

**SubmitDate**: 2023-04-06    [abs](http://arxiv.org/abs/2304.00202v2) [paper-pdf](http://arxiv.org/pdf/2304.00202v2)

**Authors**: Xiaojun Jia, Yong Zhang, Xingxing Wei, Baoyuan Wu, Ke Ma, Jue Wang, Xiaochun Cao

**Abstract**: Fast adversarial training (FAT) is an efficient method to improve robustness. However, the original FAT suffers from catastrophic overfitting, which dramatically and suddenly reduces robustness after a few training epochs. Although various FAT variants have been proposed to prevent overfitting, they require high training costs. In this paper, we investigate the relationship between adversarial example quality and catastrophic overfitting by comparing the training processes of standard adversarial training and FAT. We find that catastrophic overfitting occurs when the attack success rate of adversarial examples becomes worse. Based on this observation, we propose a positive prior-guided adversarial initialization to prevent overfitting by improving adversarial example quality without extra training costs. This initialization is generated by using high-quality adversarial perturbations from the historical training process. We provide theoretical analysis for the proposed initialization and propose a prior-guided regularization method that boosts the smoothness of the loss function. Additionally, we design a prior-guided ensemble FAT method that averages the different model weights of historical models using different decay rates. Our proposed method, called FGSM-PGK, assembles the prior-guided knowledge, i.e., the prior-guided initialization and model weights, acquired during the historical training process. Evaluations of four datasets demonstrate the superiority of the proposed method.

摘要: 快速对抗训练(FAT)是一种提高鲁棒性的有效方法。然而，原始的脂肪会遭受灾难性的过度拟合，这会在几个训练时期后急剧而突然地降低健壮性。虽然已经提出了各种脂肪变种来防止过度适应，但它们需要很高的培训成本。本文通过比较标准对抗性训练和FAT的训练过程，考察了对抗性样本质量与灾难性过拟合的关系。我们发现，当对抗性例子的攻击成功率变差时，就会发生灾难性的过拟合。基于这一观察结果，我们提出了一种积极的先验指导的对抗性初始化方法，通过在不增加额外训练成本的情况下提高对抗性实例的质量来防止过度拟合。这种初始化是通过使用来自历史训练过程的高质量对抗性扰动来生成的。我们对所提出的初始化方法进行了理论分析，并提出了一种先验引导的正则化方法，提高了损失函数的光滑性。此外，我们设计了一种先验指导的集成FAT方法，该方法使用不同的衰减率来平均历史模型的不同模型权重。我们提出的方法称为FGSM-PGK，它集合了历史训练过程中获得的先验指导知识，即先验指导的初始化和模型权重。对四个数据集的评价表明了该方法的优越性。



## **12. UNICORN: A Unified Backdoor Trigger Inversion Framework**

独角兽：一种统一的后门触发器反转框架 cs.LG

**SubmitDate**: 2023-04-05    [abs](http://arxiv.org/abs/2304.02786v1) [paper-pdf](http://arxiv.org/pdf/2304.02786v1)

**Authors**: Zhenting Wang, Kai Mei, Juan Zhai, Shiqing Ma

**Abstract**: The backdoor attack, where the adversary uses inputs stamped with triggers (e.g., a patch) to activate pre-planted malicious behaviors, is a severe threat to Deep Neural Network (DNN) models. Trigger inversion is an effective way of identifying backdoor models and understanding embedded adversarial behaviors. A challenge of trigger inversion is that there are many ways of constructing the trigger. Existing methods cannot generalize to various types of triggers by making certain assumptions or attack-specific constraints. The fundamental reason is that existing work does not consider the trigger's design space in their formulation of the inversion problem. This work formally defines and analyzes the triggers injected in different spaces and the inversion problem. Then, it proposes a unified framework to invert backdoor triggers based on the formalization of triggers and the identified inner behaviors of backdoor models from our analysis. Our prototype UNICORN is general and effective in inverting backdoor triggers in DNNs. The code can be found at https://github.com/RU-System-Software-and-Security/UNICORN.

摘要: 后门攻击是对深度神经网络(DNN)模型的严重威胁，攻击者使用带有触发器(例如补丁)的输入来激活预先植入的恶意行为。触发反转是识别后门模型和理解嵌入的敌对行为的有效方法。触发器反转的一个挑战是有许多构造触发器的方法。现有方法不能通过做出某些假设或特定于攻击的约束来对各种类型的触发器进行泛化。其根本原因是现有的工作没有考虑触发器的设计空间在他们的反问题的公式。这项工作形式化地定义和分析了不同空间中注入的触发器和反演问题。然后，基于触发器的形式化和从分析中识别出的后门模型的内部行为，提出了一个倒置后门触发器的统一框架。我们的原型独角兽在倒置DNN中的后门触发器方面是通用的和有效的。代码可在https://github.com/RU-System-Software-and-Security/UNICORN.上找到



## **13. Planning for Attacker Entrapment in Adversarial Settings**

对抗性环境下攻击者诱捕的计划 cs.AI

**SubmitDate**: 2023-04-05    [abs](http://arxiv.org/abs/2303.00822v2) [paper-pdf](http://arxiv.org/pdf/2303.00822v2)

**Authors**: Brittany Cates, Anagha Kulkarni, Sarath Sreedharan

**Abstract**: In this paper, we propose a planning framework to generate a defense strategy against an attacker who is working in an environment where a defender can operate without the attacker's knowledge. The objective of the defender is to covertly guide the attacker to a trap state from which the attacker cannot achieve their goal. Further, the defender is constrained to achieve its goal within K number of steps, where K is calculated as a pessimistic lower bound within which the attacker is unlikely to suspect a threat in the environment. Such a defense strategy is highly useful in real world systems like honeypots or honeynets, where an unsuspecting attacker interacts with a simulated production system while assuming it is the actual production system. Typically, the interaction between an attacker and a defender is captured using game theoretic frameworks. Our problem formulation allows us to capture it as a much simpler infinite horizon discounted MDP, in which the optimal policy for the MDP gives the defender's strategy against the actions of the attacker. Through empirical evaluation, we show the merits of our problem formulation.

摘要: 在本文中，我们提出了一个计划框架来生成针对攻击者的防御策略，在这种环境中，防御者可以在攻击者不知情的情况下操作。防御者的目标是秘密地引导攻击者进入陷阱状态，使攻击者无法实现他们的目标。此外，防御者被限制在K个步骤内实现其目标，其中K被计算为悲观的下限，在该下限内攻击者不太可能怀疑环境中的威胁。这样的防御策略在蜜罐或蜜网等现实世界系统中非常有用，在这些系统中，毫无戒心的攻击者与模拟生产系统交互，同时假设它是实际的生产系统。通常，攻击者和防御者之间的交互是使用博弈论框架来捕捉的。我们的问题公式允许我们将其捕获为一个更简单的无限地平线折扣MDP，其中MDP的最优策略给出了防御者针对攻击者的行为的策略。通过实证评估，我们展示了我们的问题描述的优点。



## **14. Domain Generalization with Adversarial Intensity Attack for Medical Image Segmentation**

基于对抗性强度攻击的医学图像分割领域泛化 eess.IV

Code is available upon publication

**SubmitDate**: 2023-04-05    [abs](http://arxiv.org/abs/2304.02720v1) [paper-pdf](http://arxiv.org/pdf/2304.02720v1)

**Authors**: Zheyuan Zhang, Bin Wang, Lanhong Yao, Ugur Demir, Debesh Jha, Ismail Baris Turkbey, Boqing Gong, Ulas Bagci

**Abstract**: Most statistical learning algorithms rely on an over-simplified assumption, that is, the train and test data are independent and identically distributed. In real-world scenarios, however, it is common for models to encounter data from new and different domains to which they were not exposed to during training. This is often the case in medical imaging applications due to differences in acquisition devices, imaging protocols, and patient characteristics. To address this problem, domain generalization (DG) is a promising direction as it enables models to handle data from previously unseen domains by learning domain-invariant features robust to variations across different domains. To this end, we introduce a novel DG method called Adversarial Intensity Attack (AdverIN), which leverages adversarial training to generate training data with an infinite number of styles and increase data diversity while preserving essential content information. We conduct extensive evaluation experiments on various multi-domain segmentation datasets, including 2D retinal fundus optic disc/cup and 3D prostate MRI. Our results demonstrate that AdverIN significantly improves the generalization ability of the segmentation models, achieving significant improvement on these challenging datasets. Code is available upon publication.

摘要: 大多数统计学习算法依赖于一个过于简化的假设，即训练数据和测试数据是独立的、同分布的。然而，在现实世界的场景中，模型经常遇到来自新的不同领域的数据，而它们在培训期间没有接触到这些数据。由于采集设备、成像协议和患者特征的不同，在医学成像应用中通常会出现这种情况。为了解决这个问题，领域泛化(DG)是一个很有前途的方向，因为它通过学习对不同领域之间的变化具有鲁棒性的领域不变特征，使模型能够处理来自以前未见过的领域的数据。为此，我们引入了一种称为对抗性强度攻击(AdverIN)的DG方法，它利用对抗性训练来生成具有无限样式的训练数据，并在保留基本内容信息的同时增加数据多样性。我们在各种多域分割数据集上进行了广泛的评估实验，包括2D视网膜眼底视盘/视杯和3D前列腺MRI。我们的结果表明，AdverIN显著提高了分割模型的泛化能力，在这些具有挑战性的数据集上取得了显著的改善。代码在发布后即可使用。



## **15. A Certified Radius-Guided Attack Framework to Image Segmentation Models**

一种用于图像分割模型的半径制导攻击认证框架 cs.CV

Accepted by EuroSP 2023

**SubmitDate**: 2023-04-05    [abs](http://arxiv.org/abs/2304.02693v1) [paper-pdf](http://arxiv.org/pdf/2304.02693v1)

**Authors**: Wenjie Qu, Youqi Li, Binghui Wang

**Abstract**: Image segmentation is an important problem in many safety-critical applications. Recent studies show that modern image segmentation models are vulnerable to adversarial perturbations, while existing attack methods mainly follow the idea of attacking image classification models. We argue that image segmentation and classification have inherent differences, and design an attack framework specially for image segmentation models. Our attack framework is inspired by certified radius, which was originally used by defenders to defend against adversarial perturbations to classification models. We are the first, from the attacker perspective, to leverage the properties of certified radius and propose a certified radius guided attack framework against image segmentation models. Specifically, we first adapt randomized smoothing, the state-of-the-art certification method for classification models, to derive the pixel's certified radius. We then focus more on disrupting pixels with relatively smaller certified radii and design a pixel-wise certified radius guided loss, when plugged into any existing white-box attack, yields our certified radius-guided white-box attack. Next, we propose the first black-box attack to image segmentation models via bandit. We design a novel gradient estimator, based on bandit feedback, which is query-efficient and provably unbiased and stable. We use this gradient estimator to design a projected bandit gradient descent (PBGD) attack, as well as a certified radius-guided PBGD (CR-PBGD) attack. We prove our PBGD and CR-PBGD attacks can achieve asymptotically optimal attack performance with an optimal rate. We evaluate our certified-radius guided white-box and black-box attacks on multiple modern image segmentation models and datasets. Our results validate the effectiveness of our certified radius-guided attack framework.

摘要: 图像分割是许多安全关键应用中的一个重要问题。最近的研究表明，现代图像分割模型容易受到对抗性扰动，而现有的攻击方法主要遵循攻击图像分类模型的思想。我们认为图像分割和分类有着内在的区别，并设计了一个专门针对图像分割模型的攻击框架。我们的攻击框架的灵感来自认证的RADIUS，它最初是防御者用来防御分类模型的对抗性扰动的。从攻击者的角度来看，我们是第一个利用认证RADIUS的属性，并提出了针对图像分割模型的认证RADIUS制导攻击框架。具体地说，我们首先采用随机平滑，这是目前最先进的分类模型认证方法，来推导像素的认证半径。然后，我们将重点放在具有相对较小认证半径的像素上，并设计一个像素级认证半径制导损耗，当插入任何现有的白盒攻击时，就会产生我们认证的半径制导白盒攻击。接下来，我们提出了利用盗贼对图像分割模型进行第一次黑盒攻击。我们设计了一种新的基于强盗反馈的梯度估计器，该估计器具有查询效率高、可证明无偏和稳定的特点。我们使用这个梯度估计器设计了一个投影的强盗梯度下降(PBGD)攻击，以及一个认证的半径制导的PBGD(CR-PBGD)攻击。我们证明了我们的PBGD和CR-PBGD攻击能够以最优率获得渐近最优的攻击性能。我们在多种现代图像分割模型和数据集上评估了我们的认证半径制导的白盒和黑盒攻击。我们的结果验证了我们认证的半径制导攻击框架的有效性。



## **16. Going Further: Flatness at the Rescue of Early Stopping for Adversarial Example Transferability**

更进一步：平坦性在抢救对抗性例子中提前停止可转移性 cs.LG

**SubmitDate**: 2023-04-05    [abs](http://arxiv.org/abs/2304.02688v1) [paper-pdf](http://arxiv.org/pdf/2304.02688v1)

**Authors**: Martin Gubri, Maxime Cordy, Yves Le Traon

**Abstract**: Transferability is the property of adversarial examples to be misclassified by other models than the surrogate model for which they were crafted. Previous research has shown that transferability is substantially increased when the training of the surrogate model has been early stopped. A common hypothesis to explain this is that the later training epochs are when models learn the non-robust features that adversarial attacks exploit. Hence, an early stopped model is more robust (hence, a better surrogate) than fully trained models. We demonstrate that the reasons why early stopping improves transferability lie in the side effects it has on the learning dynamics of the model. We first show that early stopping benefits transferability even on models learning from data with non-robust features. We then establish links between transferability and the exploration of the loss landscape in the parameter space, on which early stopping has an inherent effect. More precisely, we observe that transferability peaks when the learning rate decays, which is also the time at which the sharpness of the loss significantly drops. This leads us to propose RFN, a new approach for transferability that minimizes loss sharpness during training in order to maximize transferability. We show that by searching for large flat neighborhoods, RFN always improves over early stopping (by up to 47 points of transferability rate) and is competitive to (if not better than) strong state-of-the-art baselines.

摘要: 可转移性是对抗性例子被其他模型错误分类的属性，而不是为其制作它们的代理模型。先前的研究表明，当早期停止训练代理模型时，可转移性显著增加。解释这一点的一个常见假设是，较晚的训练时期是模型学习对抗性攻击所利用的非稳健特征时。因此，较早停止的模型比完全训练的模型更健壮(因此是更好的替代)。我们证明，提前停止提高可转移性的原因在于它对模型的学习动力学产生的副作用。我们首先表明，即使在从具有非稳健特征的数据中学习的模型上，提前停止也有利于可转移性。然后，我们在参数空间中建立了可转移性和损失图景的探索之间的联系，在参数空间中，提前停止具有内在的影响。更准确地说，我们观察到，当学习速度下降时，可转移性达到顶峰，这也是损失的尖锐程度显著下降的时候。这导致我们提出了RFN，这是一种新的可转移性方法，它将训练过程中的损失锐度降至最低，从而使可转移性最大化。我们表明，通过搜索大的平坦社区，RFN总是比早期停止(高达47个可转移率)有所改善，并且与(如果不是更好的)强大的最先进的基线竞争。



## **17. Adversarial robustness of VAEs through the lens of local geometry**

基于局部几何透镜的VAE的对抗性稳健性 cs.LG

International Conference on Artificial Intelligence and Statistics  (AISTATS) 2023

**SubmitDate**: 2023-04-05    [abs](http://arxiv.org/abs/2208.03923v2) [paper-pdf](http://arxiv.org/pdf/2208.03923v2)

**Authors**: Asif Khan, Amos Storkey

**Abstract**: In an unsupervised attack on variational autoencoders (VAEs), an adversary finds a small perturbation in an input sample that significantly changes its latent space encoding, thereby compromising the reconstruction for a fixed decoder. A known reason for such vulnerability is the distortions in the latent space resulting from a mismatch between approximated latent posterior and a prior distribution. Consequently, a slight change in an input sample can move its encoding to a low/zero density region in the latent space resulting in an unconstrained generation. This paper demonstrates that an optimal way for an adversary to attack VAEs is to exploit a directional bias of a stochastic pullback metric tensor induced by the encoder and decoder networks. The pullback metric tensor of an encoder measures the change in infinitesimal latent volume from an input to a latent space. Thus, it can be viewed as a lens to analyse the effect of input perturbations leading to latent space distortions. We propose robustness evaluation scores using the eigenspectrum of a pullback metric tensor. Moreover, we empirically show that the scores correlate with the robustness parameter $\beta$ of the $\beta-$VAE. Since increasing $\beta$ also degrades reconstruction quality, we demonstrate a simple alternative using \textit{mixup} training to fill the empty regions in the latent space, thus improving robustness with improved reconstruction.

摘要: 在对变分自动编码器(VAE)的无监督攻击中，攻击者发现输入样本中的微小扰动显著改变了其潜在空间编码，从而危及固定解码器的重建。造成这种脆弱性的一个已知原因是由于近似的潜在后验分布和先验分布之间的不匹配而导致的潜在空间中的扭曲。因此，输入样本中的微小变化可以将其编码移动到潜在空间中的低/零密度区域，从而产生不受限制的生成。证明了敌手攻击VAE的最佳方法是利用编解码网引起的随机拉回度量张量的方向偏差。编码器的回拉度量张量测量从输入到潜在空间的无穷小潜在体积的变化。因此，可以将其视为分析输入扰动导致潜在空间扭曲的影响的透镜。我们使用拉回度量张量的特征谱来提出稳健性评价分数。此外，我们的经验表明，得分与$\beta-$VAE的稳健性参数$\beta$相关。由于增加$\beta$也会降低重建质量，我们演示了一种简单的替代方法，使用文本{Mixup}训练来填充潜在空间中的空区域，从而通过改进重建来提高鲁棒性。



## **18. Existence and Minimax Theorems for Adversarial Surrogate Risks in Binary Classification**

二元分类中对抗性代理风险的存在性和极大极小定理 cs.LG

37 pages

**SubmitDate**: 2023-04-05    [abs](http://arxiv.org/abs/2206.09098v2) [paper-pdf](http://arxiv.org/pdf/2206.09098v2)

**Authors**: Natalie S. Frank Jonathan Niles-Weed

**Abstract**: Adversarial training is one of the most popular methods for training methods robust to adversarial attacks, however, it is not well-understood from a theoretical perspective. We prove and existence, regularity, and minimax theorems for adversarial surrogate risks. Our results explain some empirical observations on adversarial robustness from prior work and suggest new directions in algorithm development. Furthermore, our results extend previously known existence and minimax theorems for the adversarial classification risk to surrogate risks.

摘要: 对抗性训练是对抗攻击能力最强的训练方法之一，但从理论上对它的理解还不够深入。我们证明了对抗性代理风险的存在性、正则性和极大极小定理。我们的结果解释了以前工作中关于对手稳健性的一些经验观察，并为算法开发提供了新的方向。此外，我们的结果推广了已知的对抗性分类风险到代理风险的存在性和极大极小定理。



## **19. How to choose your best allies for a transferable attack?**

如何为可转移的攻击选择最好的盟友？ cs.CR

**SubmitDate**: 2023-04-05    [abs](http://arxiv.org/abs/2304.02312v1) [paper-pdf](http://arxiv.org/pdf/2304.02312v1)

**Authors**: Thibault Maho, Seyed-Mohsen Moosavi-Dezfooli, Teddy Furon

**Abstract**: The transferability of adversarial examples is a key issue in the security of deep neural networks. The possibility of an adversarial example crafted for a source model fooling another targeted model makes the threat of adversarial attacks more realistic. Measuring transferability is a crucial problem, but the Attack Success Rate alone does not provide a sound evaluation. This paper proposes a new methodology for evaluating transferability by putting distortion in a central position. This new tool shows that transferable attacks may perform far worse than a black box attack if the attacker randomly picks the source model. To address this issue, we propose a new selection mechanism, called FiT, which aims at choosing the best source model with only a few preliminary queries to the target. Our experimental results show that FiT is highly effective at selecting the best source model for multiple scenarios such as single-model attacks, ensemble-model attacks and multiple attacks (Code available at: https://github.com/t-maho/transferability_measure_fit).

摘要: 对抗性样本的可转移性是深层神经网络安全的一个关键问题。为源模型制作的对抗性示例欺骗另一个目标模型的可能性使对抗性攻击的威胁变得更加现实。衡量可转移性是一个关键问题，但仅凭攻击成功率并不能提供合理的评估。本文提出了一种将失真放在中心位置来评价可转移性的新方法。这个新工具表明，如果攻击者随机选择源模型，可转移攻击的性能可能比黑盒攻击差得多。为了解决这个问题，我们提出了一种新的选择机制，称为FIT，它旨在通过对目标的几个初步查询来选择最优源模型。我们的实验结果表明，对于单模型攻击、集成模型攻击和多攻击等多种场景，FIT在选择最佳源模型方面非常有效(代码可在：https://github.com/t-maho/transferability_measure_fit).



## **20. PatchCensor: Patch Robustness Certification for Transformers via Exhaustive Testing**

补丁检查器：通过穷举测试为变压器提供补丁健壮性认证 cs.CV

This paper has been accepted by ACM Transactions on Software  Engineering and Methodology (TOSEM'23) in "Continuous Special Section: AI and  SE." Please include TOSEM for any citations

**SubmitDate**: 2023-04-05    [abs](http://arxiv.org/abs/2111.10481v2) [paper-pdf](http://arxiv.org/pdf/2111.10481v2)

**Authors**: Yuheng Huang, Lei Ma, Yuanchun Li

**Abstract**: Vision Transformer (ViT) is known to be highly nonlinear like other classical neural networks and could be easily fooled by both natural and adversarial patch perturbations. This limitation could pose a threat to the deployment of ViT in the real industrial environment, especially in safety-critical scenarios. In this work, we propose PatchCensor, aiming to certify the patch robustness of ViT by applying exhaustive testing. We try to provide a provable guarantee by considering the worst patch attack scenarios. Unlike empirical defenses against adversarial patches that may be adaptively breached, certified robust approaches can provide a certified accuracy against arbitrary attacks under certain conditions. However, existing robustness certifications are mostly based on robust training, which often requires substantial training efforts and the sacrifice of model performance on normal samples. To bridge the gap, PatchCensor seeks to improve the robustness of the whole system by detecting abnormal inputs instead of training a robust model and asking it to give reliable results for every input, which may inevitably compromise accuracy. Specifically, each input is tested by voting over multiple inferences with different mutated attention masks, where at least one inference is guaranteed to exclude the abnormal patch. This can be seen as complete-coverage testing, which could provide a statistical guarantee on inference at the test time. Our comprehensive evaluation demonstrates that PatchCensor is able to achieve high certified accuracy (e.g. 67.1% on ImageNet for 2%-pixel adversarial patches), significantly outperforming state-of-the-art techniques while achieving similar clean accuracy (81.8% on ImageNet). Meanwhile, our technique also supports flexible configurations to handle different adversarial patch sizes (up to 25%) by simply changing the masking strategy.

摘要: 众所周知，视觉转换器(VIT)像其他经典神经网络一样是高度非线性的，很容易被自然和敌对的补丁扰动所愚弄。这一限制可能会对VIT在实际工业环境中的部署构成威胁，特别是在安全关键的情况下。在这项工作中，我们提出了补丁检查器，旨在通过应用穷举测试来证明VIT的补丁健壮性。我们试图通过考虑最糟糕的补丁攻击场景来提供可证明的保证。与针对可能被适应性破坏的对手补丁的经验防御不同，经过认证的稳健方法可以在某些条件下提供经过认证的准确性，以抵御任意攻击。然而，现有的稳健性认证大多基于健壮性训练，这往往需要大量的训练努力和牺牲正常样本上的模型性能。为了弥合这一差距，补丁检查器试图通过检测异常输入来提高整个系统的稳健性，而不是训练一个健壮的模型，并要求它为每一个输入提供可靠的结果，这可能不可避免地损害精度。具体地说，通过对具有不同突变注意掩码的多个推理进行投票来测试每一输入，其中至少一个推理被保证排除异常补丁。这可以看作是完全覆盖测试，它可以为测试时的推理提供统计保证。我们的综合评估表明，PatchComtor能够达到很高的认证准确率(例如，ImageNet上2%像素的恶意补丁的准确率为67.1%)，显著优于最先进的技术，同时获得类似的干净准确率(ImageNet上的81.8%)。同时，我们的技术还支持灵活的配置，通过简单地更改掩码策略来处理不同的对手补丁大小(高达25%)。



## **21. Dynamic Adversarial Resource Allocation: the dDAB Game**

动态对抗性资源分配：dDAB博弈 cs.MA

**SubmitDate**: 2023-04-05    [abs](http://arxiv.org/abs/2304.02172v1) [paper-pdf](http://arxiv.org/pdf/2304.02172v1)

**Authors**: Daigo Shishika, Yue Guan, Jason R. Marden, Michael Dorothy, Panagiotis Tsiotras, Vijay Kumar

**Abstract**: This work proposes a dynamic and adversarial resource allocation problem in a graph environment, which is referred to as the dynamic Defender-Attacker Blotto (dDAB) game. A team of defender robots is tasked to ensure numerical advantage at every node in the graph against a team of attacker robots. The engagement is formulated as a discrete-time dynamic game, where the two teams reallocate their robots in sequence and each robot can move at most one hop at each time step. The game terminates with the attacker's victory if any node has more attacker robots than defender robots. Our goal is to identify the necessary and sufficient number of defender robots to guarantee defense. Through a reachability analysis, we first solve the problem for the case where the attacker team stays as a single group. The results are then generalized to the case where the attacker team can freely split and merge into subteams. Crucially, our analysis indicates that there is no incentive for the attacker team to split, which significantly reduces the search space for the attacker's winning strategies and also enables us to design defender counter-strategies using superposition. We also present an efficient numerical algorithm to identify the necessary and sufficient number of defender robots to defend a given graph. Finally, we present illustrative examples to verify the efficacy of the proposed framework.

摘要: 本文提出了一个图环境下的动态对抗性资源分配问题，称为动态防御者-攻击者Blotto(DDAB)博弈。防守机器人团队的任务是确保在图表中的每个节点上对抗一组攻击机器人时具有数字优势。该交战被描述为离散时间动态博弈，其中两个团队按顺序重新分配各自的机器人，每个机器人在每个时间步长最多只能移动一跳。如果任何节点的攻击机器人多于防守机器人，则游戏以攻击者的胜利而终止。我们的目标是确定必要和足够数量的防守机器人来保证防御。通过可达性分析，我们首先解决了攻击者团队作为单个组的情况。然后将结果推广到攻击者团队可以自由拆分和合并为子团队的情况。至关重要的是，我们的分析表明，攻击队没有分裂的动机，这显著减少了攻击者制胜策略的搜索空间，并使我们能够使用叠加来设计防御者反击策略。我们还提出了一种有效的数值算法来确定必要和足够数量的防守机器人来保卫给定的图。最后，我们给出了说明性的例子来验证该框架的有效性。



## **22. Do we need entire training data for adversarial training?**

对抗性训练需要完整的训练数据吗？ cs.CV

6 pages, 4 figures

**SubmitDate**: 2023-04-05    [abs](http://arxiv.org/abs/2303.06241v2) [paper-pdf](http://arxiv.org/pdf/2303.06241v2)

**Authors**: Vipul Gupta, Apurva Narayan

**Abstract**: Deep Neural Networks (DNNs) are being used to solve a wide range of problems in many domains including safety-critical domains like self-driving cars and medical imagery. DNNs suffer from vulnerability against adversarial attacks. In the past few years, numerous approaches have been proposed to tackle this problem by training networks using adversarial training. Almost all the approaches generate adversarial examples for the entire training dataset, thus increasing the training time drastically. We show that we can decrease the training time for any adversarial training algorithm by using only a subset of training data for adversarial training. To select the subset, we filter the adversarially-prone samples from the training data. We perform a simple adversarial attack on all training examples to filter this subset. In this attack, we add a small perturbation to each pixel and a few grid lines to the input image.   We perform adversarial training on the adversarially-prone subset and mix it with vanilla training performed on the entire dataset. Our results show that when our method-agnostic approach is plugged into FGSM, we achieve a speedup of 3.52x on MNIST and 1.98x on the CIFAR-10 dataset with comparable robust accuracy. We also test our approach on state-of-the-art Free adversarial training and achieve a speedup of 1.2x in training time with a marginal drop in robust accuracy on the ImageNet dataset.

摘要: 深度神经网络(DNN)正被用来解决许多领域的广泛问题，包括自动驾驶汽车和医学成像等安全关键领域。DNN容易受到敌意攻击。在过去的几年里，已经提出了许多办法来解决这一问题，方法是使用对抗性训练来训练网络。几乎所有的方法都为整个训练数据集生成对抗性的样本，从而大大增加了训练时间。我们证明，只要使用训练数据的一个子集进行对抗性训练，就可以减少任何对抗性训练算法的训练时间。为了选择子集，我们从训练数据中过滤出易受攻击的样本。我们对所有训练样本执行简单的对抗性攻击来过滤这个子集。在这种攻击中，我们为每个像素添加一个小扰动，并在输入图像中添加一些网格线。我们对易发生对抗性的子集进行对抗性训练，并将其与在整个数据集上执行的普通训练混合。我们的结果表明，当我们的方法无关的方法被插入到FGSM中时，我们在MNIST上获得了3.52倍的加速比，在CIFAR-10数据集上获得了1.98倍的加速比，并且具有相当的鲁棒性。我们还在最先进的自由对手训练上测试了我们的方法，在ImageNet数据集上的稳健准确率略有下降的情况下，训练时间加速了1.2倍。



## **23. Boosting Adversarial Transferability using Dynamic Cues**

利用动态线索提高对抗性转移能力 cs.CV

International Conference on Learning Representations (ICLR'23),  Code:https://bit.ly/3Xd9gRQ

**SubmitDate**: 2023-04-04    [abs](http://arxiv.org/abs/2302.12252v2) [paper-pdf](http://arxiv.org/pdf/2302.12252v2)

**Authors**: Muzammal Naseer, Ahmad Mahmood, Salman Khan, Fahad Khan

**Abstract**: The transferability of adversarial perturbations between image models has been extensively studied. In this case, an attack is generated from a known surrogate \eg, the ImageNet trained model, and transferred to change the decision of an unknown (black-box) model trained on an image dataset. However, attacks generated from image models do not capture the dynamic nature of a moving object or a changing scene due to a lack of temporal cues within image models. This leads to reduced transferability of adversarial attacks from representation-enriched \emph{image} models such as Supervised Vision Transformers (ViTs), Self-supervised ViTs (\eg, DINO), and Vision-language models (\eg, CLIP) to black-box \emph{video} models. In this work, we induce dynamic cues within the image models without sacrificing their original performance on images. To this end, we optimize \emph{temporal prompts} through frozen image models to capture motion dynamics. Our temporal prompts are the result of a learnable transformation that allows optimizing for temporal gradients during an adversarial attack to fool the motion dynamics. Specifically, we introduce spatial (image) and temporal (video) cues within the same source model through task-specific prompts. Attacking such prompts maximizes the adversarial transferability from image-to-video and image-to-image models using the attacks designed for image models. Our attack results indicate that the attacker does not need specialized architectures, \eg, divided space-time attention, 3D convolutions, or multi-view convolution networks for different data modalities. Image models are effective surrogates to optimize an adversarial attack to fool black-box models in a changing environment over time. Code is available at https://bit.ly/3Xd9gRQ

摘要: 对抗性扰动在图像模型之间的可转移性已被广泛研究。在这种情况下，从已知的代理(例如，ImageNet训练的模型)生成攻击，并将其传输以改变在图像数据集上训练的未知(黑盒)模型的决策。然而，由于图像模型中缺乏时间线索，从图像模型生成的攻击不能捕捉到运动对象或变化的场景的动态性质。这导致对抗性攻击从表示丰富的{图像}模型，例如监督视觉转换器(VITS)、自我监督VITS(例如，DINO)和视觉语言模型(例如，CLIP)转移到黑盒\EMPH{VIDEO}模型。在这项工作中，我们在不牺牲图像的原始性能的情况下，在图像模型中诱导动态线索。为此，我们通过冻结图像模型来优化时间提示，以捕捉运动动力学。我们的时间提示是一种可学习转换的结果，这种转换允许在敌方攻击期间优化时间梯度，以愚弄运动动力学。具体地说，我们通过特定于任务的提示在同一来源模型中引入空间(图像)和时间(视频)线索。利用为图像模型设计的攻击，攻击此类提示最大限度地提高了图像到视频和图像到图像模型之间的对抗性。我们的攻击结果表明，攻击者不需要专门的体系结构，例如，分割时空注意力、3D卷积或针对不同数据形态的多视点卷积网络。图像模型是优化对抗性攻击的有效替代品，可以在随时间变化的环境中愚弄黑盒模型。代码可在https://bit.ly/3Xd9gRQ上找到



## **24. Risk-based Security Measure Allocation Against Injection Attacks on Actuators**

基于风险的执行器注入攻击安全措施分配 eess.SY

Submitted to IEEE Open Journal of Control Systems (OJ-CSYS)

**SubmitDate**: 2023-04-04    [abs](http://arxiv.org/abs/2304.02055v1) [paper-pdf](http://arxiv.org/pdf/2304.02055v1)

**Authors**: Sribalaji C. Anand, André M. H. Teixeira

**Abstract**: This article considers the problem of risk-optimal allocation of security measures when the actuators of an uncertain control system are under attack. We consider an adversary injecting false data into the actuator channels. The attack impact is characterized by the maximum performance loss caused by a stealthy adversary with bounded energy. Since the impact is a random variable, due to system uncertainty, we use Conditional Value-at-Risk (CVaR) to characterize the risk associated with the attack. We then consider the problem of allocating the security measures which minimize the risk. We assume that there are only a limited number of security measures available. Under this constraint, we observe that the allocation problem is a mixed-integer optimization problem. Thus we use relaxation techniques to approximate the security allocation problem into a Semi-Definite Program (SDP). We also compare our allocation method $(i)$ across different risk measures: the worst-case measure, the average (nominal) measure, and $(ii)$ across different search algorithms: the exhaustive and the greedy search algorithms. We depict the efficacy of our approach through numerical examples.

摘要: 研究了不确定控制系统执行器受到攻击时，安全措施的风险最优分配问题。我们认为对手将错误数据注入致动器通道。攻击影响的特征是具有有限能量的隐形对手所造成的最大性能损失。由于影响是一个随机变量，由于系统的不确定性，我们使用条件风险值(CVAR)来表征与攻击相关的风险。然后，我们考虑分配将风险降至最低的安全措施的问题。我们假设只有有限数量的安全措施可用。在此约束下，我们观察到分配问题是一个混合整数优化问题。因此，我们使用松弛技术将安全分配问题近似为半定规划(SDP)。我们还比较了我们的分配方法$(I)$跨不同的风险度量：最坏情况度量、平均(名义)度量，以及$(Ii)$跨不同的搜索算法：穷举搜索算法和贪婪搜索算法。我们通过数值例子描述了我们方法的有效性。



## **25. EGC: Image Generation and Classification via a Single Energy-Based Model**

EGC：基于单一能量模型的图像生成和分类 cs.CV

Technical report

**SubmitDate**: 2023-04-04    [abs](http://arxiv.org/abs/2304.02012v1) [paper-pdf](http://arxiv.org/pdf/2304.02012v1)

**Authors**: Qiushan Guo, Chuofan Ma, Yi Jiang, Zehuan Yuan, Yizhou Yu, Ping Luo

**Abstract**: Learning image classification and image generation using the same set of network parameters is a challenging problem. Recent advanced approaches perform well in one task often exhibit poor performance in the other. This work introduces an energy-based classifier and generator, namely EGC, which can achieve superior performance in both tasks using a single neural network. Unlike a conventional classifier that outputs a label given an image (i.e., a conditional distribution $p(y|\mathbf{x})$), the forward pass in EGC is a classifier that outputs a joint distribution $p(\mathbf{x},y)$, enabling an image generator in its backward pass by marginalizing out the label $y$. This is done by estimating the energy and classification probability given a noisy image in the forward pass, while denoising it using the score function estimated in the backward pass. EGC achieves competitive generation results compared with state-of-the-art approaches on ImageNet-1k, CelebA-HQ and LSUN Church, while achieving superior classification accuracy and robustness against adversarial attacks on CIFAR-10. This work represents the first successful attempt to simultaneously excel in both tasks using a single set of network parameters. We believe that EGC bridges the gap between discriminative and generative learning.

摘要: 使用相同的网络参数学习图像分类和图像生成是一个具有挑战性的问题。最近的高级方法在一项任务中表现良好，但在另一项任务中往往表现不佳。这项工作介绍了一种基于能量的分类器和生成器，即EGC，它可以使用单个神经网络在这两个任务中获得优越的性能。与输出给定图像的标签(即，条件分布$p(y|\mathbf{x})$)的传统分类器不同，EGC中的前向通道是输出联合分布$p(\mathbf{x}，y)$的分类器，从而使图像生成器在其后向通道中通过边缘化标签$y$来实现。这是通过在前传中给出噪声图像的情况下估计能量和分类概率来完成的，同时使用在后传中估计的得分函数来对其进行去噪。EGC在ImageNet-1k、CelebA-HQ和LSUN Church上实现了与最先进的方法相比具有竞争力的生成结果，同时在CIFAR-10上获得了卓越的分类准确性和对对手攻击的稳健性。这项工作代表了首次成功尝试使用一组网络参数同时在两项任务中脱颖而出。我们认为，EGC弥合了歧视性学习和生成性学习之间的差距。



## **26. Cross-Class Feature Augmentation for Class Incremental Learning**

面向类增量学习的跨类特征增强 cs.CV

**SubmitDate**: 2023-04-04    [abs](http://arxiv.org/abs/2304.01899v1) [paper-pdf](http://arxiv.org/pdf/2304.01899v1)

**Authors**: Taehoon Kim, Jaeyoo Park, Bohyung Han

**Abstract**: We propose a novel class incremental learning approach by incorporating a feature augmentation technique motivated by adversarial attacks. We employ a classifier learned in the past to complement training examples rather than simply play a role as a teacher for knowledge distillation towards subsequent models. The proposed approach has a unique perspective to utilize the previous knowledge in class incremental learning since it augments features of arbitrary target classes using examples in other classes via adversarial attacks on a previously learned classifier. By allowing the cross-class feature augmentations, each class in the old tasks conveniently populates samples in the feature space, which alleviates the collapse of the decision boundaries caused by sample deficiency for the previous tasks, especially when the number of stored exemplars is small. This idea can be easily incorporated into existing class incremental learning algorithms without any architecture modification. Extensive experiments on the standard benchmarks show that our method consistently outperforms existing class incremental learning methods by significant margins in various scenarios, especially under an environment with an extremely limited memory budget.

摘要: 我们提出了一种新的类增量学习方法，该方法结合了一种基于对抗性攻击的特征增强技术。我们使用过去学习的分类器来补充训练实例，而不是简单地扮演老师的角色，将知识升华到后续模型。该方法在类增量学习中利用先验知识具有独特的视角，因为它通过对先前学习的分类器的对抗性攻击，利用其他类中的例子来扩充任意目标类的特征。通过允许跨类特征扩充，旧任务中的每一类都可以方便地在特征空间中填充样本，从而缓解了以前任务由于样本不足而导致的决策边界崩溃，特别是在存储的样本数量较少的情况下。这种思想可以很容易地融入到现有的类增量学习算法中，而不需要修改任何体系结构。在标准基准测试上的大量实验表明，在各种情况下，特别是在内存预算极其有限的环境下，我们的方法始终比现有的类增量学习方法有显著的优势。



## **27. Adversarial Detection: Attacking Object Detection in Real Time**

对抗性检测：攻击目标的实时检测 cs.AI

Accepted by IEEE Intelligent Vehicle Symposium, 2023

**SubmitDate**: 2023-04-04    [abs](http://arxiv.org/abs/2209.01962v4) [paper-pdf](http://arxiv.org/pdf/2209.01962v4)

**Authors**: Han Wu, Syed Yunas, Sareh Rowlands, Wenjie Ruan, Johan Wahlstrom

**Abstract**: Intelligent robots rely on object detection models to perceive the environment. Following advances in deep learning security it has been revealed that object detection models are vulnerable to adversarial attacks. However, prior research primarily focuses on attacking static images or offline videos. Therefore, it is still unclear if such attacks could jeopardize real-world robotic applications in dynamic environments. This paper bridges this gap by presenting the first real-time online attack against object detection models. We devise three attacks that fabricate bounding boxes for nonexistent objects at desired locations. The attacks achieve a success rate of about 90\% within about 20 iterations. The demo video is available at https://youtu.be/zJZ1aNlXsMU.

摘要: 智能机器人依靠物体检测模型来感知环境。随着深度学习安全性的进步，人们发现目标检测模型容易受到敌意攻击。然而，以往的研究主要集中在攻击静态图像或离线视频上。因此，目前尚不清楚此类攻击是否会危及动态环境中真实世界的机器人应用。本文通过提出第一个针对目标检测的实时在线攻击模型来弥补这一差距。我们设计了三种攻击，在所需位置为不存在的对象制造边界框。这些攻击在大约20次迭代内达到了约90%的成功率。该演示视频可在https://youtu.be/zJZ1aNlXsMU.上查看



## **28. Adversarial Driving: Attacking End-to-End Autonomous Driving**

对抗性驾驶：攻击型端到端自动驾驶 cs.CV

Accepted by IEEE Intelligent Vehicle Symposium, 2023

**SubmitDate**: 2023-04-04    [abs](http://arxiv.org/abs/2103.09151v6) [paper-pdf](http://arxiv.org/pdf/2103.09151v6)

**Authors**: Han Wu, Syed Yunas, Sareh Rowlands, Wenjie Ruan, Johan Wahlstrom

**Abstract**: As research in deep neural networks advances, deep convolutional networks become promising for autonomous driving tasks. In particular, there is an emerging trend of employing end-to-end neural network models for autonomous driving. However, previous research has shown that deep neural network classifiers are vulnerable to adversarial attacks. While for regression tasks, the effect of adversarial attacks is not as well understood. In this research, we devise two white-box targeted attacks against end-to-end autonomous driving models. Our attacks manipulate the behavior of the autonomous driving system by perturbing the input image. In an average of 800 attacks with the same attack strength (epsilon=1), the image-specific and image-agnostic attack deviates the steering angle from the original output by 0.478 and 0.111, respectively, which is much stronger than random noises that only perturbs the steering angle by 0.002 (The steering angle ranges from [-1, 1]). Both attacks can be initiated in real-time on CPUs without employing GPUs. Demo video: https://youtu.be/I0i8uN2oOP0.

摘要: 随着深度神经网络研究的深入，深度卷积网络在自动驾驶任务中变得很有前途。特别是，使用端到端神经网络模型进行自动驾驶是一种新兴的趋势。然而，以往的研究表明，深度神经网络分类器容易受到敌意攻击。而对于回归任务，对抗性攻击的效果并没有被很好地理解。在本研究中，我们设计了两种针对端到端自动驾驶模型的白盒针对性攻击。我们的攻击通过干扰输入图像来操纵自动驾驶系统的行为。在相同攻击强度(epsilon=1)的平均800次攻击中，图像特定攻击和图像无关攻击使转向角与原始输出的偏差分别为0.478和0.111，远远强于仅扰动转向角0.002的随机噪声(转向角范围为[-1，1])。这两种攻击都可以在不使用GPU的情况下在CPU上实时发起。演示视频：https://youtu.be/I0i8uN2oOP0.



## **29. MENLI: Robust Evaluation Metrics from Natural Language Inference**

MENLI：自然语言推理中的稳健评价指标 cs.CL

TACL 2023 Camera-ready github link fixed

**SubmitDate**: 2023-04-04    [abs](http://arxiv.org/abs/2208.07316v3) [paper-pdf](http://arxiv.org/pdf/2208.07316v3)

**Authors**: Yanran Chen, Steffen Eger

**Abstract**: Recently proposed BERT-based evaluation metrics for text generation perform well on standard benchmarks but are vulnerable to adversarial attacks, e.g., relating to information correctness. We argue that this stems (in part) from the fact that they are models of semantic similarity. In contrast, we develop evaluation metrics based on Natural Language Inference (NLI), which we deem a more appropriate modeling. We design a preference-based adversarial attack framework and show that our NLI based metrics are much more robust to the attacks than the recent BERT-based metrics. On standard benchmarks, our NLI based metrics outperform existing summarization metrics, but perform below SOTA MT metrics. However, when combining existing metrics with our NLI metrics, we obtain both higher adversarial robustness (15%-30%) and higher quality metrics as measured on standard benchmarks (+5% to 30%).

摘要: 最近提出的基于BERT的文本生成评估指标在标准基准上表现良好，但容易受到敌意攻击，例如与信息正确性有关的攻击。我们认为，这(部分)源于这样一个事实：它们是语义相似性的模型。相比之下，我们基于自然语言推理(NLI)开发评估指标，我们认为这是更合适的建模。我们设计了一个基于偏好的对抗性攻击框架，并表明我们的基于NLI的度量比最近的基于BERT的度量具有更强的抗攻击能力。在标准基准测试中，我们基于NLI的指标优于现有的摘要指标，但低于SOTA MT指标。然而，当将现有指标与我们的NLI指标相结合时，我们获得了更高的对手健壮性(15%-30%)和标准基准测试的更高质量指标(+5%到30%)。



## **30. Tracklet-Switch Adversarial Attack against Pedestrian Multi-Object Tracking Trackers**

Tracklet-Switch对行人多目标跟踪跟踪器的敌意攻击 cs.CV

**SubmitDate**: 2023-04-04    [abs](http://arxiv.org/abs/2111.08954v3) [paper-pdf](http://arxiv.org/pdf/2111.08954v3)

**Authors**: Delv Lin, Qi Chen, Chengyu Zhou, Kun He

**Abstract**: Multi-Object Tracking (MOT) has achieved aggressive progress and derived many excellent deep learning trackers. Meanwhile, most deep learning models are known to be vulnerable to adversarial examples that are crafted with small perturbations but could mislead the model prediction. In this work, we observe that the robustness on the MOT trackers is rarely studied, and it is challenging to attack the MOT system since its mature association algorithms are designed to be robust against errors during the tracking. To this end, we analyze the vulnerability of popular MOT trackers and propose a novel adversarial attack method called Tracklet-Switch (TraSw) against the complete tracking pipeline of MOT. The proposed TraSw can fool the advanced deep pedestrian trackers (i.e., FairMOT and ByteTrack), causing them fail to track the targets in the subsequent frames by perturbing very few frames. Experiments on the MOT-Challenge datasets (i.e., 2DMOT15, MOT17, and MOT20) show that TraSw can achieve an extraordinarily high success attack rate of over 95% by attacking only four frames on average. To our knowledge, this is the first work on the adversarial attack against the pedestrian MOT trackers. Code is available at https://github.com/JHL-HUST/TraSw .

摘要: 多目标跟踪(MOT)已经取得了突破性的进展，衍生出了许多优秀的深度学习跟踪器。同时，众所周知，大多数深度学习模型很容易受到对手例子的影响，这些例子是用小扰动制作的，但可能会误导模型预测。在这项工作中，我们注意到对MOT跟踪器的健壮性研究很少，而且攻击MOT系统是具有挑战性的，因为其成熟的关联算法被设计成对跟踪过程中的错误具有健壮性。为此，本文分析了目前流行的MOT跟踪器的脆弱性，提出了一种新的针对MOT完整跟踪管道的对抗性攻击方法Tracklet-Switch(TraSw)。提出的TraSw算法可以欺骗先进的深度行人跟踪器(即FairMOT和ByteTrack)，通过对极少的帧进行扰动而导致它们无法跟踪后续帧中的目标。在MOT-Challenger数据集(2DMOT15、MOT17和MOT20)上的实验表明，TraSw平均只攻击4帧，可以达到95%以上的攻击成功率。据我们所知，这是针对行人MOT追踪器的对抗攻击的第一项工作。代码可在https://github.com/JHL-HUST/TraSw上找到。



## **31. On the Feasibility of Specialized Ability Extracting for Large Language Code Models**

大型语言代码模型专业能力抽取的可行性研究 cs.SE

11 pages

**SubmitDate**: 2023-04-04    [abs](http://arxiv.org/abs/2303.03012v3) [paper-pdf](http://arxiv.org/pdf/2303.03012v3)

**Authors**: Zongjie Li, Chaozheng Wang, Pingchuan Ma, Chaowei Liu, Shuai Wang, Daoyuan Wu, Cuiyun Gao

**Abstract**: Recent progress in large language code models (LLCMs) has led to a dramatic surge in the use of software development. Nevertheless, it is widely known that training a well-performed LLCM requires a plethora of workforce for collecting the data and high quality annotation. Additionally, the training dataset may be proprietary (or partially open source to the public), and the training process is often conducted on a large-scale cluster of GPUs with high costs. Inspired by the recent success of imitation attacks in extracting computer vision and natural language models, this work launches the first imitation attack on LLCMs: by querying a target LLCM with carefully-designed queries and collecting the outputs, the adversary can train an imitation model that manifests close behavior with the target LLCM. We systematically investigate the effectiveness of launching imitation attacks under different query schemes and different LLCM tasks. We also design novel methods to polish the LLCM outputs, resulting in an effective imitation training process. We summarize our findings and provide lessons harvested in this study that can help better depict the attack surface of LLCMs. Our research contributes to the growing body of knowledge on imitation attacks and defenses in deep neural models, particularly in the domain of code related tasks.

摘要: 大型语言代码模型(LLCM)的最新进展导致软件开发的使用激增。然而，众所周知，培训一个表现良好的LLCM需要大量的劳动力来收集数据和高质量的注释。此外，训练数据集可能是专有的(或部分向公众开放源代码)，并且训练过程通常在成本较高的大规模GPU集群上进行。受最近模仿攻击在提取计算机视觉和自然语言模型方面的成功启发，该工作对LLCM发起了第一次模仿攻击：通过使用精心设计的查询来查询目标LLCM并收集输出，对手可以训练出与目标LLCM表现出密切行为的模仿模型。系统地研究了在不同的查询方案和不同的LLCM任务下发起模仿攻击的有效性。我们还设计了新的方法来完善LLCM的输出，从而产生了一个有效的模拟训练过程。我们总结了我们的发现，并提供了在这项研究中获得的教训，有助于更好地描述LLCM的攻击面。我们的研究有助于在深层神经模型中，特别是在与代码相关的任务领域中，关于模仿攻击和防御的知识不断增长。



## **32. Defending Against Patch-based Backdoor Attacks on Self-Supervised Learning**

防御基于补丁的自我监督学习后门攻击 cs.CV

Accepted to CVPR 2023

**SubmitDate**: 2023-04-04    [abs](http://arxiv.org/abs/2304.01482v1) [paper-pdf](http://arxiv.org/pdf/2304.01482v1)

**Authors**: Ajinkya Tejankar, Maziar Sanjabi, Qifan Wang, Sinong Wang, Hamed Firooz, Hamed Pirsiavash, Liang Tan

**Abstract**: Recently, self-supervised learning (SSL) was shown to be vulnerable to patch-based data poisoning backdoor attacks. It was shown that an adversary can poison a small part of the unlabeled data so that when a victim trains an SSL model on it, the final model will have a backdoor that the adversary can exploit. This work aims to defend self-supervised learning against such attacks. We use a three-step defense pipeline, where we first train a model on the poisoned data. In the second step, our proposed defense algorithm (PatchSearch) uses the trained model to search the training data for poisoned samples and removes them from the training set. In the third step, a final model is trained on the cleaned-up training set. Our results show that PatchSearch is an effective defense. As an example, it improves a model's accuracy on images containing the trigger from 38.2% to 63.7% which is very close to the clean model's accuracy, 64.6%. Moreover, we show that PatchSearch outperforms baselines and state-of-the-art defense approaches including those using additional clean, trusted data. Our code is available at https://github.com/UCDvision/PatchSearch

摘要: 最近，自我监督学习(SSL)被证明容易受到基于补丁的数据中毒后门攻击。研究表明，攻击者可以毒化一小部分未标记的数据，以便当受害者在其上训练一个SSL模型时，最终的模型将有一个后门可供攻击者利用。这项工作旨在保护自我监督学习免受此类攻击。我们使用一个三步防御管道，在那里我们首先训练一个关于有毒数据的模型。在第二步中，我们提出的防御算法(PatchSearch)使用训练好的模型在训练数据中搜索有毒样本，并将其从训练集中删除。在第三步中，在清理后的训练集上训练最终模型。我们的结果表明，PatchSearch是一种有效的防御措施。例如，它将模型对包含触发器的图像的准确率从38.2%提高到63.7%，与CLEAN模型的准确率64.6%非常接近。此外，我们还表明，PatchSearch的性能优于基线和最先进的防御方法，包括使用额外的干净、可信数据的方法。我们的代码可以在https://github.com/UCDvision/PatchSearch上找到



## **33. NetFlick: Adversarial Flickering Attacks on Deep Learning Based Video Compression**

Netflick：基于深度学习的视频压缩对抗性闪烁攻击 eess.IV

8 pages; Accepted to ICLR 2023 ML4IoT workshop

**SubmitDate**: 2023-04-04    [abs](http://arxiv.org/abs/2304.01441v1) [paper-pdf](http://arxiv.org/pdf/2304.01441v1)

**Authors**: Jung-Woo Chang, Nojan Sheybani, Shehzeen Samarah Hussain, Mojan Javaheripi, Seira Hidano, Farinaz Koushanfar

**Abstract**: Video compression plays a significant role in IoT devices for the efficient transport of visual data while satisfying all underlying bandwidth constraints. Deep learning-based video compression methods are rapidly replacing traditional algorithms and providing state-of-the-art results on edge devices. However, recently developed adversarial attacks demonstrate that digitally crafted perturbations can break the Rate-Distortion relationship of video compression. In this work, we present a real-world LED attack to target video compression frameworks. Our physically realizable attack, dubbed NetFlick, can degrade the spatio-temporal correlation between successive frames by injecting flickering temporal perturbations. In addition, we propose universal perturbations that can downgrade performance of incoming video without prior knowledge of the contents. Experimental results demonstrate that NetFlick can successfully deteriorate the performance of video compression frameworks in both digital- and physical-settings and can be further extended to attack downstream video classification networks.

摘要: 视频压缩在物联网设备中发挥着重要作用，可以在满足所有基本带宽限制的同时高效传输视频数据。基于深度学习的视频压缩方法正在迅速取代传统算法，并在边缘设备上提供最先进的结果。然而，最近发展起来的对抗性攻击表明，数字制造的扰动可以打破视频压缩的率失真关系。在这项工作中，我们提出了一个针对视频压缩框架的真实世界的LED攻击。我们的物理可实现攻击，称为Netflick，可以通过注入闪烁的时间扰动来降低连续帧之间的时空相关性。此外，我们还提出了一种普遍的扰动，它可以在不事先知道视频内容的情况下降低传入视频的性能。实验结果表明，Netflick能够成功地降低视频压缩框架在数字和物理环境下的性能，并可以进一步扩展到攻击下游视频分类网络。



## **34. Is Stochastic Mirror Descent Vulnerable to Adversarial Delay Attacks? A Traffic Assignment Resilience Study**

随机镜像下降容易受到对抗性延迟攻击吗？交通分配弹性研究 cs.LG

Preprint under review

**SubmitDate**: 2023-04-03    [abs](http://arxiv.org/abs/2304.01161v1) [paper-pdf](http://arxiv.org/pdf/2304.01161v1)

**Authors**: Yunian Pan, Tao Li, Quanyan Zhu

**Abstract**: \textit{Intelligent Navigation Systems} (INS) are exposed to an increasing number of informational attack vectors, which often intercept through the communication channels between the INS and the transportation network during the data collecting process. To measure the resilience of INS, we use the concept of a Wardrop Non-Equilibrium Solution (WANES), which is characterized by the probabilistic outcome of learning within a bounded number of interactions. By using concentration arguments, we have discovered that any bounded feedback delaying attack only degrades the systematic performance up to order $\tilde{\mathcal{O}}(\sqrt{{d^3}{T^{-1}}})$ along the traffic flow trajectory within the Delayed Mirror Descent (DMD) online-learning framework. This degradation in performance can occur with only mild assumptions imposed. Our result implies that learning-based INS infrastructures can achieve Wardrop Non-equilibrium even when experiencing a certain period of disruption in the information structure. These findings provide valuable insights for designing defense mechanisms against possible jamming attacks across different layers of the transportation ecosystem.

摘要: 智能导航系统(INS)面临着越来越多的信息攻击载体，在数据采集过程中，这些信息载体往往通过INS与交通网络之间的通信通道被拦截。为了衡量INS的弹性，我们使用了Wardrop非平衡解(WANES)的概念，该概念的特征是在有限数量的相互作用中学习的概率结果。通过使用集中度参数，我们发现任何有界反馈延迟攻击只会在延迟镜像下降(DMD)在线学习框架内沿流量轨迹将系统性能降低高达数量级。仅在施加温和假设的情况下，可能会发生这种性能下降。我们的结果表明，即使在信息结构经历了一段时间的破坏后，基于学习的惯导基础设施也可以达到Wardrop非均衡。这些发现为设计针对交通生态系统不同层可能的干扰攻击的防御机制提供了有价值的见解。



## **35. Learning About Simulated Adversaries from Human Defenders using Interactive Cyber-Defense Games**

使用交互式网络防御游戏学习来自人类防御者的模拟对手 cs.CR

Submitted to Journal of Cybersecurity

**SubmitDate**: 2023-04-03    [abs](http://arxiv.org/abs/2304.01142v1) [paper-pdf](http://arxiv.org/pdf/2304.01142v1)

**Authors**: Baptiste Prebot, Yinuo Du, Cleotilde Gonzalez

**Abstract**: Given the increase in cybercrime, cybersecurity analysts (i.e. Defenders) are in high demand. Defenders must monitor an organization's network to evaluate threats and potential breaches into the network. Adversary simulation is commonly used to test defenders' performance against known threats to organizations. However, it is unclear how effective this training process is in preparing defenders for this highly demanding job. In this paper, we demonstrate how to use adversarial algorithms to investigate defenders' learning of defense strategies, using interactive cyber defense games. Our Interactive Defense Game (IDG) represents a cyber defense scenario that requires constant monitoring of incoming network alerts and allows a defender to analyze, remove, and restore services based on the events observed in a network. The participants in our study faced one of two types of simulated adversaries. A Beeline adversary is a fast, targeted, and informed attacker; and a Meander adversary is a slow attacker that wanders the network until it finds the right target to exploit. Our results suggest that although human defenders have more difficulty to stop the Beeline adversary initially, they were able to learn to stop this adversary by taking advantage of their attack strategy. Participants who played against the Beeline adversary learned to anticipate the adversary and take more proactive actions, while decreasing their reactive actions. These findings have implications for understanding how to help cybersecurity analysts speed up their training.

摘要: 鉴于网络犯罪的增加，网络安全分析师(即捍卫者)的需求很高。防御者必须监控组织的网络，以评估对该网络的威胁和潜在入侵。对手模拟通常用于测试防御者针对组织的已知威胁的表现。然而，目前还不清楚这一培训过程在为辩护人准备这一要求极高的工作方面有多有效。在本文中，我们演示了如何使用对抗性算法来调查防御者对防御策略的学习，并使用交互式网络防御游戏。我们的交互式防御游戏(IDG)代表了一种网络防御场景，它需要持续监控传入的网络警报，并允许防御者根据在网络中观察到的事件来分析、删除和恢复服务。我们研究中的参与者面对的是两种类型的模拟对手之一。直线型攻击者是速度快、目标明确且见多识广的攻击者；而Meander型攻击者则是动作迟缓的攻击者，会在网络中游荡，直到找到合适的目标进行攻击。我们的结果表明，尽管人类防御者最初更难阻止直线型对手，但他们能够通过利用自己的攻击策略来学习阻止这个对手。与直线式对手对决的参与者学会了预见对手，并采取更积极的行动，同时减少他们的被动行动。这些发现对于理解如何帮助网络安全分析师加快培训具有重要意义。



## **36. A Pilot Study of Query-Free Adversarial Attack against Stable Diffusion**

针对稳定扩散的无查询对抗性攻击的初步研究 cs.CV

The 3rd Workshop of Adversarial Machine Learning on Computer Vision:  Art of Robustness

**SubmitDate**: 2023-04-03    [abs](http://arxiv.org/abs/2303.16378v2) [paper-pdf](http://arxiv.org/pdf/2303.16378v2)

**Authors**: Haomin Zhuang, Yihua Zhang, Sijia Liu

**Abstract**: Despite the record-breaking performance in Text-to-Image (T2I) generation by Stable Diffusion, less research attention is paid to its adversarial robustness. In this work, we study the problem of adversarial attack generation for Stable Diffusion and ask if an adversarial text prompt can be obtained even in the absence of end-to-end model queries. We call the resulting problem 'query-free attack generation'. To resolve this problem, we show that the vulnerability of T2I models is rooted in the lack of robustness of text encoders, e.g., the CLIP text encoder used for attacking Stable Diffusion. Based on such insight, we propose both untargeted and targeted query-free attacks, where the former is built on the most influential dimensions in the text embedding space, which we call steerable key dimensions. By leveraging the proposed attacks, we empirically show that only a five-character perturbation to the text prompt is able to cause the significant content shift of synthesized images using Stable Diffusion. Moreover, we show that the proposed target attack can precisely steer the diffusion model to scrub the targeted image content without causing much change in untargeted image content. Our code is available at https://github.com/OPTML-Group/QF-Attack.

摘要: 尽管稳定扩散在文本到图像(T2I)的生成中取得了创纪录的性能，但对其对抗健壮性的研究较少。在这项工作中，我们研究了稳定扩散的对抗性攻击生成问题，并询问即使在没有端到端模型查询的情况下，是否也能获得对抗性文本提示。我们称由此产生的问题为“无查询攻击生成”。为了解决这个问题，我们证明了T2I模型的脆弱性源于文本编码器缺乏健壮性，例如用于攻击稳定扩散的CLIP文本编码器。基于这样的见解，我们提出了无目标查询攻击和无目标查询攻击，前者建立在文本嵌入空间中最有影响力的维度上，我们称之为可引导的关键维度。通过利用提出的攻击，我们的经验表明，只有对文本提示的五个字符的扰动才能导致使用稳定扩散的合成图像的显著内容偏移。此外，我们还证明了所提出的目标攻击能够准确地引导扩散模型对目标图像内容进行擦除，而不会对非目标图像内容造成太大改变。我们的代码可以在https://github.com/OPTML-Group/QF-Attack.上找到



## **37. Improving RF-DNA Fingerprinting Performance in an Indoor Multipath Environment Using Semi-Supervised Learning**

利用半监督学习改善室内多径环境下的RF-DNA指纹识别性能 eess.SP

16 pages, 14 figures. Submitted to IEEE Transactions on Information  Forensics & Security

**SubmitDate**: 2023-04-02    [abs](http://arxiv.org/abs/2304.00648v1) [paper-pdf](http://arxiv.org/pdf/2304.00648v1)

**Authors**: Mohamed k. Fadul, Donald R. Reising, Lakmali P. Weerasena, T. Daniel Loveless, Mina Sartipi

**Abstract**: The number of Internet of Things (IoT) deployments is expected to reach 75.4 billion by 2025. Roughly 70% of all IoT devices employ weak or no encryption; thus, putting them and their connected infrastructure at risk of attack by devices that are wrongly authenticated or not authenticated at all. A physical layer security approach -- known as Specific Emitter Identification (SEI) -- has been proposed and is being pursued as a viable IoT security mechanism. SEI is advantageous because it is a passive technique that exploits inherent and distinct features that are unintentionally added to the signal by the IoT Radio Frequency (RF) front-end. SEI's passive exploitation of unintentional signal features removes any need to modify the IoT device, which makes it ideal for existing and future IoT deployments. Despite the amount of SEI research conducted, some challenges must be addressed to make SEI a viable IoT security approach. One challenge is the extraction of SEI features from signals collected under multipath fading conditions. Multipath corrupts the inherent SEI features that are used to discriminate one IoT device from another; thus, degrading authentication performance and increasing the chance of attack. This work presents two semi-supervised Deep Learning (DL) equalization approaches and compares their performance with the current state of the art. The two approaches are the Conditional Generative Adversarial Network (CGAN) and Joint Convolutional Auto-Encoder and Convolutional Neural Network (JCAECNN). Both approaches learn the channel distribution to enable multipath correction while simultaneously preserving the SEI exploited features. CGAN and JCAECNN performance is assessed using a Rayleigh fading channel under degrading SNR, up to thirty-two IoT devices, and two publicly available signal sets. The JCAECNN improves SEI performance by 10% beyond that of the current state of the art.

摘要: 到2025年，物联网(IoT)部署数量预计将达到754亿。大约70%的物联网设备采用弱加密或无加密；因此，它们及其连接的基础设施面临被身份验证错误或根本未通过身份验证的设备攻击的风险。一种物理层安全方法--称为特定发射器标识(SEI)--已经被提出，并正在作为一种可行的物联网安全机制而被追寻。SEI具有优势，因为它是一种被动技术，它利用物联网射频(RF)前端无意中添加到信号中的固有且独特的功能。SEI对无意信号功能的被动利用消除了修改物联网设备的任何需要，这使其成为现有和未来物联网部署的理想选择。尽管进行了大量SEI研究，但必须解决一些挑战，才能使SEI成为可行的物联网安全方法。一个挑战是从多径衰落条件下收集的信号中提取SEI特征。多路径破坏了用于区分物联网设备的固有SEI功能；因此，降低了身份验证性能并增加了攻击机会。本文提出了两种半监督深度学习(DL)均衡方法，并将它们的性能与现有技术进行了比较。这两种方法是条件生成对抗网络(CGAN)和联合卷积自动编码器和卷积神经网络(JCAECNN)。这两种方法都学习信道分布以实现多径校正，同时保留SEI利用的特征。CGAN和JCAECNN的性能是使用降级SNR下的瑞利衰落信道、多达32个物联网设备和两个公开可用的信号集进行评估的。JCAECNN将SEI性能提高了10%，超过了目前的技术水平。



## **38. Test-time Detection and Repair of Adversarial Samples via Masked Autoencoder**

基于屏蔽自动编码器的敌方样本测试时间检测与修复 cs.CV

**SubmitDate**: 2023-04-02    [abs](http://arxiv.org/abs/2303.12848v3) [paper-pdf](http://arxiv.org/pdf/2303.12848v3)

**Authors**: Yun-Yun Tsai, Ju-Chin Chao, Albert Wen, Zhaoyuan Yang, Chengzhi Mao, Tapan Shah, Junfeng Yang

**Abstract**: Training-time defenses, known as adversarial training, incur high training costs and do not generalize to unseen attacks. Test-time defenses solve these issues but most existing test-time defenses require adapting the model weights, therefore they do not work on frozen models and complicate model memory management. The only test-time defense that does not adapt model weights aims to adapt the input with self-supervision tasks. However, we empirically found these self-supervision tasks are not sensitive enough to detect adversarial attacks accurately. In this paper, we propose DRAM, a novel defense method to detect and repair adversarial samples at test time via Masked autoencoder (MAE). We demonstrate how to use MAE losses to build a Kolmogorov-Smirnov test to detect adversarial samples. Moreover, we use the MAE losses to calculate input reversal vectors that repair adversarial samples resulting from previously unseen attacks. Results on large-scale ImageNet dataset show that, compared to all detection baselines evaluated, DRAM achieves the best detection rate (82% on average) on all eight adversarial attacks evaluated. For attack repair, DRAM improves the robust accuracy by 6% ~ 41% for standard ResNet50 and 3% ~ 8% for robust ResNet50 compared with the baselines that use contrastive learning and rotation prediction.

摘要: 训练时的防守，也就是所谓的对抗性训练，会产生很高的训练成本，而且不会泛化为看不见的攻击。测试时间防御解决了这些问题，但大多数现有的测试时间防御需要调整模型权重，因此它们不能在冻结的模型上工作，并使模型内存管理复杂化。唯一不调整模型权重的测试时间防御旨在调整具有自我监督任务的输入。然而，我们经验发现，这些自我监督任务不够敏感，不足以准确地检测到对手攻击。本文提出了一种新的防御方法DRAM，该方法通过屏蔽自动编码器(MAE)在测试时检测和修复恶意样本。我们演示了如何使用MAE损失来构建Kolmogorov-Smirnov检验来检测对手样本。此外，我们使用MAE损失来计算输入反转向量，这些输入反转向量修复了以前未见过的攻击所产生的敌意样本。在大规模ImageNet数据集上的结果表明，与所有被评估的检测基线相比，DRAM在所有被评估的8种对抗性攻击中都获得了最好的检测率(平均为82%)。在攻击修复方面，与使用对比学习和旋转预测的基线相比，DRAM将标准ResNet50的稳健准确率提高了6%~41%，将稳健ResNet50的稳健准确率提高了3%~8%。



## **39. FACM: Intermediate Layer Still Retain Effective Features against Adversarial Examples**

FACM：中间层仍然保留了对抗对手示例的有效特征 cs.CV

**SubmitDate**: 2023-04-02    [abs](http://arxiv.org/abs/2206.00924v2) [paper-pdf](http://arxiv.org/pdf/2206.00924v2)

**Authors**: Xiangyuan Yang, Jie Lin, Hanlin Zhang, Xinyu Yang, Peng Zhao

**Abstract**: In strong adversarial attacks against deep neural networks (DNN), the generated adversarial example will mislead the DNN-implemented classifier by destroying the output features of the last layer. To enhance the robustness of the classifier, in our paper, a \textbf{F}eature \textbf{A}nalysis and \textbf{C}onditional \textbf{M}atching prediction distribution (FACM) model is proposed to utilize the features of intermediate layers to correct the classification. Specifically, we first prove that the intermediate layers of the classifier can still retain effective features for the original category, which is defined as the correction property in our paper. According to this, we propose the FACM model consisting of \textbf{F}eature \textbf{A}nalysis (FA) correction module, \textbf{C}onditional \textbf{M}atching \textbf{P}rediction \textbf{D}istribution (CMPD) correction module and decision module. The FA correction module is the fully connected layers constructed with the output of the intermediate layers as the input to correct the classification of the classifier. The CMPD correction module is a conditional auto-encoder, which can not only use the output of intermediate layers as the condition to accelerate convergence but also mitigate the negative effect of adversarial example training with the Kullback-Leibler loss to match prediction distribution. Through the empirically verified diversity property, the correction modules can be implemented synergistically to reduce the adversarial subspace. Hence, the decision module is proposed to integrate the correction modules to enhance the DNN classifier's robustness. Specially, our model can be achieved by fine-tuning and can be combined with other model-specific defenses.

摘要: 在针对深度神经网络(DNN)的强对抗性攻击中，生成的对抗性样本会破坏最后一层的输出特征，从而误导DNN实现的分类器。为了提高分类器的稳健性，本文提出了一种利用中间层特征对分类结果进行校正的方法--条件匹配预测分布模型(FACM)和条件匹配预测分布模型(FACM)。具体地说，我们首先证明了分类器的中间层仍然可以保留原始类别的有效特征，这在本文中被定义为校正属性。在此基础上，提出了由Textbf{F}特征、Textbf{A}分析(FA)修正模块、Textbf{C}条件Textbf{M}匹配、Textbf{P}条件、Textbf{D}分布(CMPD)修正模块和决策模块组成的FACM模型。FA校正模块是以中间层的输出为输入构建的全连通层，用于校正分类器的分类。CMPD纠错模块是一个有条件的自动编码器，它不仅可以利用中间层的输出作为加速收敛的条件，而且可以通过Kullback-Leibler损失来匹配预测分布来缓解对抗性样本训练的负面影响。通过经验验证的分集性质，纠错模块可以协同实现，减少对抗性的子空间。因此，为了增强DNN分类器的稳健性，提出了决策模块来集成校正模块。特别是，我们的模型可以通过微调实现，并可以与其他特定模型的防御相结合。



## **40. Adversarial Training of Self-supervised Monocular Depth Estimation against Physical-World Attacks**

对抗物理世界攻击的自监督单目深度估计的对抗性训练 cs.CV

Initially accepted at ICLR2023 (Spotlight)

**SubmitDate**: 2023-04-02    [abs](http://arxiv.org/abs/2301.13487v3) [paper-pdf](http://arxiv.org/pdf/2301.13487v3)

**Authors**: Zhiyuan Cheng, James Liang, Guanhong Tao, Dongfang Liu, Xiangyu Zhang

**Abstract**: Monocular Depth Estimation (MDE) is a critical component in applications such as autonomous driving. There are various attacks against MDE networks. These attacks, especially the physical ones, pose a great threat to the security of such systems. Traditional adversarial training method requires ground-truth labels hence cannot be directly applied to self-supervised MDE that does not have ground-truth depth. Some self-supervised model hardening techniques (e.g., contrastive learning) ignore the domain knowledge of MDE and can hardly achieve optimal performance. In this work, we propose a novel adversarial training method for self-supervised MDE models based on view synthesis without using ground-truth depth. We improve adversarial robustness against physical-world attacks using L0-norm-bounded perturbation in training. We compare our method with supervised learning based and contrastive learning based methods that are tailored for MDE. Results on two representative MDE networks show that we achieve better robustness against various adversarial attacks with nearly no benign performance degradation.

摘要: 单目深度估计(MDE)是自动驾驶等应用中的重要组成部分。针对MDE网络的攻击有多种。这些攻击，尤其是物理攻击，对这类系统的安全构成了极大的威胁。传统的对抗性训练方法需要地面真相标签，因此不能直接应用于没有地面真相深度的自监督MDE。一些自监督模型强化技术(如对比学习)忽略了MDE的领域知识，很难达到最优性能。在这项工作中，我们提出了一种新的基于视图合成的自监督MDE模型的对抗性训练方法，而不使用地面真实深度。我们在训练中使用L0范数有界扰动来提高对手对物理世界攻击的健壮性。我们将我们的方法与基于监督学习和基于对比学习的方法进行了比较，这些方法都是为MDE量身定制的。在两个具有代表性的MDE网络上的实验结果表明，该算法在几乎不降低性能的情况下，对各种敌意攻击具有更好的鲁棒性。



## **41. Instance-level Trojan Attacks on Visual Question Answering via Adversarial Learning in Neuron Activation Space**

实例级木马基于神经元激活空间对抗性学习的视觉问答攻击 cs.CV

**SubmitDate**: 2023-04-02    [abs](http://arxiv.org/abs/2304.00436v1) [paper-pdf](http://arxiv.org/pdf/2304.00436v1)

**Authors**: Yuwei Sun, Hideya Ochiai, Jun Sakuma

**Abstract**: Malicious perturbations embedded in input data, known as Trojan attacks, can cause neural networks to misbehave. However, the impact of a Trojan attack is reduced during fine-tuning of the model, which involves transferring knowledge from a pretrained large-scale model like visual question answering (VQA) to the target model. To mitigate the effects of a Trojan attack, replacing and fine-tuning multiple layers of the pretrained model is possible. This research focuses on sample efficiency, stealthiness and variation, and robustness to model fine-tuning. To address these challenges, we propose an instance-level Trojan attack that generates diverse Trojans across input samples and modalities. Adversarial learning establishes a correlation between a specified perturbation layer and the misbehavior of the fine-tuned model. We conducted extensive experiments on the VQA-v2 dataset using a range of metrics. The results show that our proposed method can effectively adapt to a fine-tuned model with minimal samples. Specifically, we found that a model with a single fine-tuning layer can be compromised using a single shot of adversarial samples, while a model with more fine-tuning layers can be compromised using only a few shots.

摘要: 嵌入在输入数据中的恶意扰动称为特洛伊木马攻击，可能会导致神经网络行为不当。然而，木马攻击的影响在模型微调期间会减少，这涉及将知识从预先训练的大规模模型(如视觉问答(VQA))传输到目标模型。为了减轻特洛伊木马攻击的影响，可以替换和微调多层预先训练的模型。这项研究的重点是样本效率、隐蔽性和变异性，以及对模型微调的稳健性。为了应对这些挑战，我们提出了一种实例级特洛伊木马攻击，该攻击跨输入样本和模式生成不同的特洛伊木马。对抗性学习在特定的扰动层和微调模型的错误行为之间建立了关联。我们使用一系列度量在VQA-v2数据集上进行了广泛的实验。结果表明，我们提出的方法能够有效地适应样本最少的微调模型。具体地说，我们发现具有单个微调层的模型可以使用单次对抗性样本来折衷，而具有更多微调层的模型可以仅使用几次来折衷。



## **42. Coordinated Defense Allocation in Reach-Avoid Scenarios with Efficient Online Optimization**

高效在线优化的REACH-DOVER场景中的协同防御分配 cs.RO

**SubmitDate**: 2023-04-01    [abs](http://arxiv.org/abs/2304.00234v1) [paper-pdf](http://arxiv.org/pdf/2304.00234v1)

**Authors**: Junwei Liu, Zikai Ouyang, Jiahui Yang, Hua Chen, Haibo Lu, Wei Zhang

**Abstract**: Deriving strategies for multiple agents under adversarial scenarios poses a significant challenge in attaining both optimality and efficiency. In this paper, we propose an efficient defense strategy for cooperative defense against a group of attackers in a convex environment. The defenders aim to minimize the total number of attackers that successfully enter the target set without prior knowledge of the attacker's strategy. Our approach involves a two-scale method that decomposes the problem into coordination against a single attacker and assigning defenders to attackers. We first develop a coordination strategy for multiple defenders against a single attacker, implementing online convex programming. This results in the maximum defense-winning region of initial joint states from which the defender can successfully defend against a single attacker. We then propose an allocation algorithm that significantly reduces computational effort required to solve the induced integer linear programming problem. The allocation guarantees defense performance enhancement as the game progresses. We perform various simulations to verify the efficiency of our algorithm compared to the state-of-the-art approaches, including the one using the Gazabo platform with Robot Operating System.

摘要: 在对抗性场景下为多个代理推导策略，在获得最优性和效率方面是一个重大挑战。本文提出了一种在凸环境下协同防御一群攻击者的有效防御策略。防御者的目标是最大限度地减少在事先不知道攻击者的策略的情况下成功进入目标集合的攻击者的总数。我们的方法涉及一个双尺度方法，该方法将问题分解为针对单个攻击者的协调，并将防御者分配给攻击者。我们首先开发了多个防御者对抗单个攻击者的协调策略，实现了在线凸规划。这导致了初始联合状态的最大防御制胜区域，在该区域中，防御者可以成功地防御单个攻击者。然后，我们提出了一种分配算法，该算法大大减少了求解诱导整数线性规划问题所需的计算工作量。这种分配保证了随着比赛的进行，防守性能会得到提高。我们进行了各种仿真，以验证我们的算法相对于最先进的方法的效率，包括使用带有Robot操作系统的Gazabo平台的方法。



## **43. Proximal Splitting Adversarial Attacks for Semantic Segmentation**

基于近邻分裂的对抗性语义分割攻击 cs.LG

CVPR 2023. Code available at:  https://github.com/jeromerony/alma_prox_segmentation

**SubmitDate**: 2023-03-31    [abs](http://arxiv.org/abs/2206.07179v2) [paper-pdf](http://arxiv.org/pdf/2206.07179v2)

**Authors**: Jérôme Rony, Jean-Christophe Pesquet, Ismail Ben Ayed

**Abstract**: Classification has been the focal point of research on adversarial attacks, but only a few works investigate methods suited to denser prediction tasks, such as semantic segmentation. The methods proposed in these works do not accurately solve the adversarial segmentation problem and, therefore, overestimate the size of the perturbations required to fool models. Here, we propose a white-box attack for these models based on a proximal splitting to produce adversarial perturbations with much smaller $\ell_\infty$ norms. Our attack can handle large numbers of constraints within a nonconvex minimization framework via an Augmented Lagrangian approach, coupled with adaptive constraint scaling and masking strategies. We demonstrate that our attack significantly outperforms previously proposed ones, as well as classification attacks that we adapted for segmentation, providing a first comprehensive benchmark for this dense task.

摘要: 分类一直是对抗性攻击研究的重点，但很少有人研究适合于密集预测任务的方法，如语义分割。这些工作中提出的方法不能准确地解决对抗性分割问题，因此高估了愚弄模型所需的扰动的大小。在这里，我们对这些模型提出了一个基于近邻分裂的白盒攻击，以产生具有小得多的$\ell_\inty$范数的对抗性扰动。我们的攻击可以通过增广拉格朗日方法，结合自适应约束缩放和掩蔽策略，在非凸极小化框架内处理大量约束。我们证明，我们的攻击显著优于之前提出的攻击，以及我们为分割而采用的分类攻击，为这一密集任务提供了第一个全面的基准。



## **44. Fides: A Generative Framework for Result Validation of Outsourced Machine Learning Workloads via TEE**

FIDS：一种基于TEE的外包机器学习任务结果验证生成框架 cs.CR

**SubmitDate**: 2023-03-31    [abs](http://arxiv.org/abs/2304.00083v1) [paper-pdf](http://arxiv.org/pdf/2304.00083v1)

**Authors**: Abhinav Kumar, Miguel A. Guirao Aguilera, Reza Tourani, Satyajayant Misra

**Abstract**: The growing popularity of Machine Learning (ML) has led to its deployment in various sensitive domains, which has resulted in significant research focused on ML security and privacy. However, in some applications, such as autonomous driving, integrity verification of the outsourced ML workload is more critical-a facet that has not received much attention. Existing solutions, such as multi-party computation and proof-based systems, impose significant computation overhead, which makes them unfit for real-time applications. We propose Fides, a novel framework for real-time validation of outsourced ML workloads. Fides features a novel and efficient distillation technique-Greedy Distillation Transfer Learning-that dynamically distills and fine-tunes a space and compute-efficient verification model for verifying the corresponding service model while running inside a trusted execution environment. Fides features a client-side attack detection model that uses statistical analysis and divergence measurements to identify, with a high likelihood, if the service model is under attack. Fides also offers a re-classification functionality that predicts the original class whenever an attack is identified. We devised a generative adversarial network framework for training the attack detection and re-classification models. The extensive evaluation shows that Fides achieves an accuracy of up to 98% for attack detection and 94% for re-classification.

摘要: 机器学习(ML)的日益流行导致了它在各种敏感领域的部署，这导致了对ML安全和隐私的大量研究。然而，在一些应用中，例如自动驾驶，对外包的ML工作负载的完整性验证更加关键-这是一个没有得到太多关注的方面。现有的解决方案，如多方计算和基于证明的系统，带来了巨大的计算开销，这使得它们不适合实时应用。我们提出了一种新的实时验证外包ML工作负载的框架FIDS。FIDS采用了一种新颖而高效的蒸馏技术-贪婪蒸馏转移学习-它动态地提取和微调空间和计算效率高的验证模型，以便在可信执行环境中运行时验证相应的服务模型。FIDS具有客户端攻击检测模型，该模型使用统计分析和分歧测量来识别服务模型是否受到攻击的可能性很高。FIDS还提供了重新分类功能，该功能可以在识别攻击时预测原始类别。我们设计了一个生成式对抗性网络框架来训练攻击检测和重分类模型。广泛的评估表明，FIDS对攻击检测的准确率高达98%，对重新分类的准确率高达94%。



## **45. Decentralized Attack Search and the Design of Bug Bounty Schemes**

分散攻击搜索与漏洞赏金方案设计 econ.TH

**SubmitDate**: 2023-03-31    [abs](http://arxiv.org/abs/2304.00077v1) [paper-pdf](http://arxiv.org/pdf/2304.00077v1)

**Authors**: Hans Gersbach, Akaki Mamageishvili, Fikri Pitsuwan

**Abstract**: Systems and blockchains often have security vulnerabilities and can be attacked by adversaries, with potentially significant negative consequences. Therefore, organizations and blockchain infrastructure providers increasingly rely on bug bounty programs, where external individuals probe the system and report any vulnerabilities (bugs) in exchange for monetary rewards (bounty). We develop a contest model for bug bounty programs with an arbitrary number of agents who decide whether to undertake a costly search for bugs or not. Search costs are private information. Besides characterizing the ensuing equilibria, we show that even inviting an unlimited crowd does not guarantee that bugs are found. Adding paid agents can increase the efficiency of the bug bounty scheme although the crowd that is attracted becomes smaller. Finally, adding (known) bugs increases the likelihood that unknown bugs are found, but to limit reward payments it may be optimal to add them only with some probability.

摘要: 系统和区块链往往存在安全漏洞，可能会受到对手的攻击，带来潜在的重大负面后果。因此，组织和区块链基础设施提供商越来越依赖漏洞赏金计划，即外部个人探测系统并报告任何漏洞(漏洞)，以换取金钱奖励(赏金)。我们建立了一个漏洞赏金计划的竞赛模型，由任意数量的代理决定是否进行一次昂贵的漏洞搜索。搜索成本是私人信息。除了描述随后的均衡，我们还表明，即使邀请无限的人群也不能保证发现错误。增加付费代理可以提高漏洞赏金计划的效率，尽管吸引的人群会变得更少。最后，添加(已知的)错误会增加发现未知错误的可能性，但为了限制奖励支付，最好是仅以一定的概率添加它们。



## **46. To be Robust and to be Fair: Aligning Fairness with Robustness**

稳健性与公正性：使公平与稳健性一致 cs.LG

**SubmitDate**: 2023-03-31    [abs](http://arxiv.org/abs/2304.00061v1) [paper-pdf](http://arxiv.org/pdf/2304.00061v1)

**Authors**: Junyi Chai, Xiaoqian Wang

**Abstract**: Adversarial training has been shown to be reliable in improving robustness against adversarial samples. However, the problem of adversarial training in terms of fairness has not yet been properly studied, and the relationship between fairness and accuracy attack still remains unclear. Can we simultaneously improve robustness w.r.t. both fairness and accuracy? To tackle this topic, in this paper, we study the problem of adversarial training and adversarial attack w.r.t. both metrics. We propose a unified structure for fairness attack which brings together common notions in group fairness, and we theoretically prove the equivalence of fairness attack against different notions. Moreover, we show the alignment of fairness and accuracy attack, and theoretically demonstrate that robustness w.r.t. one metric benefits from robustness w.r.t. the other metric. Our study suggests a novel way to unify adversarial training and attack w.r.t. fairness and accuracy, and experimental results show that our proposed method achieves better performance in terms of robustness w.r.t. both metrics.

摘要: 对抗性训练已被证明在提高对抗对抗性样本的稳健性方面是可靠的。然而，对抗性训练在公平性方面的问题还没有得到很好的研究，公平性和精确度攻击之间的关系还不清楚。我们能否同时提高w.r.t.的健壮性。既公平又准确？为了解决这一问题，本文研究了对抗性训练和对抗性攻击的问题。这两个指标。我们提出了一种统一的公平攻击结构，将群公平中的常见概念集合在一起，并从理论上证明了公平攻击对不同概念的等价性。此外，我们还给出了公平性攻击和准确性攻击的一致性，并从理论上证明了该算法的健壮性。一种度量受益于健壮性w.r.t.另一个指标。我们的研究提出了一种将对抗性训练和进攻相结合的新方法。公平性和准确性，实验结果表明，该方法在鲁棒性方面取得了较好的性能。这两个指标。



## **47. PEOPL: Characterizing Privately Encoded Open Datasets with Public Labels**

Peopl：使用公共标签表征私有编码的开放数据集 cs.LG

Submitted to IEEE Transactions on Information Forensics and Security

**SubmitDate**: 2023-03-31    [abs](http://arxiv.org/abs/2304.00047v1) [paper-pdf](http://arxiv.org/pdf/2304.00047v1)

**Authors**: Homa Esfahanizadeh, Adam Yala, Rafael G. L. D'Oliveira, Andrea J. D. Jaba, Victor Quach, Ken R. Duffy, Tommi S. Jaakkola, Vinod Vaikuntanathan, Manya Ghobadi, Regina Barzilay, Muriel Médard

**Abstract**: Allowing organizations to share their data for training of machine learning (ML) models without unintended information leakage is an open problem in practice. A promising technique for this still-open problem is to train models on the encoded data. Our approach, called Privately Encoded Open Datasets with Public Labels (PEOPL), uses a certain class of randomly constructed transforms to encode sensitive data. Organizations publish their randomly encoded data and associated raw labels for ML training, where training is done without knowledge of the encoding realization. We investigate several important aspects of this problem: We introduce information-theoretic scores for privacy and utility, which quantify the average performance of an unfaithful user (e.g., adversary) and a faithful user (e.g., model developer) that have access to the published encoded data. We then theoretically characterize primitives in building families of encoding schemes that motivate the use of random deep neural networks. Empirically, we compare the performance of our randomized encoding scheme and a linear scheme to a suite of computational attacks, and we also show that our scheme achieves competitive prediction accuracy to raw-sample baselines. Moreover, we demonstrate that multiple institutions, using independent random encoders, can collaborate to train improved ML models.

摘要: 在实践中，允许组织共享用于机器学习(ML)模型训练的数据而不会意外地泄露信息是一个悬而未决的问题。对于这个仍然悬而未决的问题，一个很有前途的技术是根据编码的数据训练模型。我们的方法被称为带有公共标签的私有编码开放数据集(PEOPL)，它使用特定类型的随机构造的转换来编码敏感数据。组织发布其随机编码的数据和关联的原始标签用于ML训练，其中训练是在不知道编码实现的情况下进行的。我们研究了这个问题的几个重要方面：我们引入了隐私和效用的信息论分数，它量化了可以访问已发布编码数据的不忠诚用户(例如，对手)和忠诚用户(例如，模型开发人员)的平均性能。然后，我们在理论上刻画了原语在构建激励使用随机深度神经网络的编码方案族中的特征。在实验中，我们将我们的随机编码方案和线性方案的性能与一组计算攻击进行了比较，并且我们还表明，我们的方案达到了与原始样本基线相当的预测精度。此外，我们还证明了多个机构可以使用独立的随机编码器来协作训练改进的ML模型。



## **48. Packet-Level Adversarial Network Traffic Crafting using Sequence Generative Adversarial Networks**

基于序列生成式对抗网络的分组级对抗网络流量制作 cs.CR

The authors agreed to withdraw the manuscript due to privacy reason

**SubmitDate**: 2023-03-31    [abs](http://arxiv.org/abs/2103.04794v2) [paper-pdf](http://arxiv.org/pdf/2103.04794v2)

**Authors**: Qiumei Cheng, Shiying Zhou, Yi Shen, Dezhang Kong, Chunming Wu

**Abstract**: The surge in the internet of things (IoT) devices seriously threatens the current IoT security landscape, which requires a robust network intrusion detection system (NIDS). Despite superior detection accuracy, existing machine learning or deep learning based NIDS are vulnerable to adversarial examples. Recently, generative adversarial networks (GANs) have become a prevailing method in adversarial examples crafting. However, the nature of discrete network traffic at the packet level makes it hard for GAN to craft adversarial traffic as GAN is efficient in generating continuous data like image synthesis. Unlike previous methods that convert discrete network traffic into a grayscale image, this paper gains inspiration from SeqGAN in sequence generation with policy gradient. Based on the structure of SeqGAN, we propose Attack-GAN to generate adversarial network traffic at packet level that complies with domain constraints. Specifically, the adversarial packet generation is formulated into a sequential decision making process. In this case, each byte in a packet is regarded as a token in a sequence. The objective of the generator is to select a token to maximize its expected end reward. To bypass the detection of NIDS, the generated network traffic and benign traffic are classified by a black-box NIDS. The prediction results returned by the NIDS are fed into the discriminator to guide the update of the generator. We generate malicious adversarial traffic based on a real public available dataset with attack functionality unchanged. The experimental results validate that the generated adversarial samples are able to deceive many existing black-box NIDS.

摘要: 物联网(IoT)设备的激增严重威胁到当前的物联网安全格局，这需要一个强大的网络入侵检测系统(NIDS)。尽管检测精度很高，但现有的基于机器学习或深度学习的网络入侵检测系统很容易受到敌意例子的攻击。近年来，生成性对抗性网络(GANS)已成为对抗性实例制作中的一种流行方法。然而，分组级别的离散网络流量的性质使得GAN很难创建敌意流量，因为GAN在生成图像合成等连续数据方面是有效的。不同于以往将离散网络流量转换为灰度图像的方法，本文从策略梯度序列生成中得到了SeqGAN的启发。基于SeqGAN的结构，我们提出了攻击GAN，在包级生成符合域约束的敌意网络流量。具体地说，敌意分组生成被表述为连续的决策过程。在这种情况下，分组中的每个字节被视为序列中的令牌。生成器的目标是选择一个令牌来最大化其预期的最终回报。为了绕过网络入侵检测系统的检测，生成的网络流量和良性流量被黑盒网络入侵检测系统分类。网络入侵检测系统返回的预测结果被送入鉴别器以指导生成器的更新。我们基于真实的公共可用数据集生成恶意敌意流量，攻击功能保持不变。实验结果表明，生成的对抗性样本能够欺骗许多已有的黑盒网络入侵检测系统。



## **49. Fooling Polarization-based Vision using Locally Controllable Polarizing Projection**

利用局部可控偏振投影愚弄偏振视觉 cs.CV

**SubmitDate**: 2023-03-31    [abs](http://arxiv.org/abs/2303.17890v1) [paper-pdf](http://arxiv.org/pdf/2303.17890v1)

**Authors**: Zhuoxiao Li, Zhihang Zhong, Shohei Nobuhara, Ko Nishino, Yinqiang Zheng

**Abstract**: Polarization is a fundamental property of light that encodes abundant information regarding surface shape, material, illumination and viewing geometry. The computer vision community has witnessed a blossom of polarization-based vision applications, such as reflection removal, shape-from-polarization, transparent object segmentation and color constancy, partially due to the emergence of single-chip mono/color polarization sensors that make polarization data acquisition easier than ever. However, is polarization-based vision vulnerable to adversarial attacks? If so, is that possible to realize these adversarial attacks in the physical world, without being perceived by human eyes? In this paper, we warn the community of the vulnerability of polarization-based vision, which can be more serious than RGB-based vision. By adapting a commercial LCD projector, we achieve locally controllable polarizing projection, which is successfully utilized to fool state-of-the-art polarization-based vision algorithms for glass segmentation and color constancy. Compared with existing physical attacks on RGB-based vision, which always suffer from the trade-off between attack efficacy and eye conceivability, the adversarial attackers based on polarizing projection are contact-free and visually imperceptible, since naked human eyes can rarely perceive the difference of viciously manipulated polarizing light and ordinary illumination. This poses unprecedented risks on polarization-based vision, both in the monochromatic and trichromatic domain, for which due attentions should be paid and counter measures be considered.

摘要: 偏振是光的一个基本属性，它编码了关于表面形状、材料、照明和观察几何的丰富信息。计算机视觉领域已经见证了基于偏振的视觉应用的蓬勃发展，例如反射去除、从偏振形状、透明对象分割和颜色恒定，部分原因是单芯片单色/颜色偏振传感器的出现使得偏振数据的获取比以往任何时候都更加容易。然而，基于极化的愿景容易受到对手的攻击吗？如果是这样的话，有可能在物理世界中实现这些对抗性攻击，而不被人眼察觉吗？在本文中，我们警告社区基于偏振的视觉的脆弱性，这可能比基于RGB的视觉更严重。通过采用商用LCD投影仪，实现了局部可控的偏振投影，并成功地将其用于欺骗最先进的基于偏振的视觉算法，以实现玻璃分割和颜色恒定。与现有的基于RGB视觉的物理攻击相比，基于偏振投影的对抗性攻击者是非接触式的，视觉上不可感知，因为肉眼很少察觉到恶意操纵的偏振光和普通照明的差异。这给基于偏振的视觉带来了前所未有的风险，无论是在单色领域还是在三色领域，都应该给予应有的关注，并考虑采取对策。



## **50. Pentimento: Data Remanence in Cloud FPGAs**

Pentimento：云现场可编程门阵列中的数据存储 cs.CR

17 Pages, 8 Figures

**SubmitDate**: 2023-03-31    [abs](http://arxiv.org/abs/2303.17881v1) [paper-pdf](http://arxiv.org/pdf/2303.17881v1)

**Authors**: Colin Drewes, Olivia Weng, Andres Meza, Alric Althoff, David Kohlbrenner, Ryan Kastner, Dustin Richmond

**Abstract**: Cloud FPGAs strike an alluring balance between computational efficiency, energy efficiency, and cost. It is the flexibility of the FPGA architecture that enables these benefits, but that very same flexibility that exposes new security vulnerabilities. We show that a remote attacker can recover "FPGA pentimenti" - long-removed secret data belonging to a prior user of a cloud FPGA. The sensitive data constituting an FPGA pentimento is an analog imprint from bias temperature instability (BTI) effects on the underlying transistors. We demonstrate how this slight degradation can be measured using a time-to-digital (TDC) converter when an adversary programs one into the target cloud FPGA.   This technique allows an attacker to ascertain previously safe information on cloud FPGAs, even after it is no longer explicitly present. Notably, it can allow an attacker who knows a non-secret "skeleton" (the physical structure, but not the contents) of the victim's design to (1) extract proprietary details from an encrypted FPGA design image available on the AWS marketplace and (2) recover data loaded at runtime by a previous user of a cloud FPGA using a known design. Our experiments show that BTI degradation (burn-in) and recovery are measurable and constitute a security threat to commercial cloud FPGAs.

摘要: 云现场可编程门阵列在计算效率、能源效率和成本之间取得了诱人的平衡。正是FPGA架构的灵活性实现了这些优势，但同样的灵活性也暴露了新的安全漏洞。我们展示了远程攻击者可以恢复“fpga pentimenti”--属于云fpga先前用户的长时间删除的秘密数据。构成现场可编程门阵列的敏感数据是偏置温度不稳定性(BTI)对底层晶体管影响的模拟印记。我们演示了当对手将时间-数字(TDC)转换器编程到目标云FPGA中时，如何测量这种轻微的降级。这项技术允许攻击者确定云现场可编程门阵列上以前的安全信息，即使它不再显式存在。值得注意的是，它可以让知道受害者设计的非机密“骨架”(物理结构，但不是内容)的攻击者(1)从AWS Marketplace上提供的加密的FPGA设计图像中提取专有细节，以及(2)恢复云FPGA的前用户在运行时使用已知设计加载的数据。我们的实验表明，BTI的退化(老化)和恢复是可测量的，并对商用云现场可编程门阵列构成安全威胁。



