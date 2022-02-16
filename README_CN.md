# Latest Adversarial Attack Papers
**update at 2022-02-17 06:31:49**

翻译来自 https://cloud.tencent.com/document/product/551/15619

## **1. Defending against Reconstruction Attacks with Rényi Differential Privacy**

利用Rényi差分私密性防御重构攻击 cs.LG

**SubmitDate**: 2022-02-15    [paper-pdf](http://arxiv.org/pdf/2202.07623v1)

**Authors**: Pierre Stock, Igor Shilov, Ilya Mironov, Alexandre Sablayrolles

**Abstracts**: Reconstruction attacks allow an adversary to regenerate data samples of the training set using access to only a trained model. It has been recently shown that simple heuristics can reconstruct data samples from language models, making this threat scenario an important aspect of model release. Differential privacy is a known solution to such attacks, but is often used with a relatively large privacy budget (epsilon > 8) which does not translate to meaningful guarantees. In this paper we show that, for a same mechanism, we can derive privacy guarantees for reconstruction attacks that are better than the traditional ones from the literature. In particular, we show that larger privacy budgets do not protect against membership inference, but can still protect extraction of rare secrets. We show experimentally that our guarantees hold against various language models, including GPT-2 finetuned on Wikitext-103.

摘要: 重构攻击允许对手仅使用对训练模型的访问来重新生成训练集的数据样本。最近的研究表明，简单的启发式算法可以从语言模型中重建数据样本，从而使这种威胁场景成为模型发布的一个重要方面。差异隐私是此类攻击的已知解决方案，但通常使用相对较大的隐私预算(epsilon>8)，这并不能转化为有意义的保证。在本文中，我们表明，对于相同的机制，我们可以从文献中推导出比传统的重构攻击更好的隐私保证。特别是，我们表明，较大的隐私预算不能防止成员关系推断，但仍然可以保护罕见秘密的提取。我们的实验表明，我们的保证适用于各种语言模型，包括在Wikitext-103上微调的GPT-2。



## **2. StratDef: a strategic defense against adversarial attacks in malware detection**

StratDef：恶意软件检测中对抗对手攻击的战略防御 cs.LG

**SubmitDate**: 2022-02-15    [paper-pdf](http://arxiv.org/pdf/2202.07568v1)

**Authors**: Aqib Rashid, Jose Such

**Abstracts**: Over the years, most research towards defenses against adversarial attacks on machine learning models has been in the image processing domain. The malware detection domain has received less attention despite its importance. Moreover, most work exploring defenses focuses on feature-based, gradient-based or randomized methods but with no strategy when applying them. In this paper, we introduce StratDef, which is a strategic defense system tailored for the malware detection domain based on a Moving Target Defense and Game Theory approach. We overcome challenges related to the systematic construction, selection and strategic use of models to maximize adversarial robustness. StratDef dynamically and strategically chooses the best models to increase the uncertainty for the attacker, whilst minimizing critical aspects in the adversarial ML domain like attack transferability. We provide the first comprehensive evaluation of defenses against adversarial attacks on machine learning for malware detection, where our threat model explores different levels of threat, attacker knowledge, capabilities, and attack intensities. We show that StratDef performs better than other defenses even when facing the peak adversarial threat. We also show that, from the existing defenses, only a few adversarially-trained models provide substantially better protection than just using vanilla models but are still outperformed by StratDef.

摘要: 多年来，针对机器学习模型抗敌意攻击的研究大多集中在图像处理领域。恶意软件检测域尽管很重要，但受到的关注较少。而且，大多数探索防御的工作都集中在基于特征的、基于梯度的或随机的方法上，而在应用这些方法时没有策略。本文介绍了StratDef，这是一个基于移动目标防御和博弈论的针对恶意软件检测领域定制的战略防御系统。我们克服了与模型的系统构建、选择和战略使用相关的挑战，以最大限度地提高对手的健壮性。StratDef动态地和战略性地选择最佳模型，以增加攻击者的不确定性，同时最小化敌对ML领域中的关键方面，如攻击可转移性。我们在恶意软件检测的机器学习中首次全面评估了防御敌意攻击的能力，其中我们的威胁模型探索了不同级别的威胁、攻击者知识、能力和攻击强度。我们表明，即使在面临最严重的对手威胁时，StratDef也比其他防御系统表现得更好。我们还表明，从现有的防御来看，只有少数经过对抗性训练的模型提供了比仅仅使用普通模型更好的保护，但仍然优于StratDef。



## **3. Random Walks for Adversarial Meshes**

对抗性网格的随机游动 cs.CV

**SubmitDate**: 2022-02-15    [paper-pdf](http://arxiv.org/pdf/2202.07453v1)

**Authors**: Amir Belder, Gal Yefet, Ran Ben Izhak, Ayellet Tal

**Abstracts**: A polygonal mesh is the most-commonly used representation of surfaces in computer graphics; thus, a variety of classification networks have been recently proposed. However, while adversarial attacks are wildly researched in 2D, almost no works on adversarial meshes exist. This paper proposes a novel, unified, and general adversarial attack, which leads to misclassification of numerous state-of-the-art mesh classification neural networks. Our attack approach is black-box, i.e. it has access only to the network's predictions, but not to the network's full architecture or gradients. The key idea is to train a network to imitate a given classification network. This is done by utilizing random walks along the mesh surface, which gather geometric information. These walks provide insight onto the regions of the mesh that are important for the correct prediction of the given classification network. These mesh regions are then modified more than other regions in order to attack the network in a manner that is barely visible to the naked eye.

摘要: 多边形网格是计算机图形学中最常用的曲面表示，因此，最近提出了各种分类网络。然而，尽管对抗性攻击在2D方面得到了广泛的研究，但几乎没有关于对抗性网络的工作。本文提出了一种新颖的、统一的、通用的对抗性攻击，该攻击导致了众多最新的网格分类神经网络的误分类。我们的攻击方法是黑匣子，即它只能访问网络的预测，而不能访问网络的完整架构或梯度。其核心思想是训练一个网络来模仿给定的分类网络。这是通过利用沿网格曲面的随机漫游来完成的，该漫游收集几何信息。这些遍历提供了对网格区域的洞察，这些区域对于给定分类网络的正确预测非常重要。然后，这些网格区域被修改得比其他区域更多，以便以肉眼几乎看不见的方式攻击网络。



## **4. Unreasonable Effectiveness of Last Hidden Layer Activations**

最后一次隐藏层激活的不合理效果 cs.LG

22 pages, Under review

**SubmitDate**: 2022-02-15    [paper-pdf](http://arxiv.org/pdf/2202.07342v1)

**Authors**: Omer Faruk Tuna, Ferhat Ozgur Catak, M. Taner Eskil

**Abstracts**: In standard Deep Neural Network (DNN) based classifiers, the general convention is to omit the activation function in the last (output) layer and directly apply the softmax function on the logits to get the probability scores of each class. In this type of architectures, the loss value of the classifier against any output class is directly proportional to the difference between the final probability score and the label value of the associated class. Standard White-box adversarial evasion attacks, whether targeted or untargeted, mainly try to exploit the gradient of the model loss function to craft adversarial samples and fool the model. In this study, we show both mathematically and experimentally that using some widely known activation functions in the output layer of the model with high temperature values has the effect of zeroing out the gradients for both targeted and untargeted attack cases, preventing attackers from exploiting the model's loss function to craft adversarial samples. We've experimentally verified the efficacy of our approach on MNIST (Digit), CIFAR10 datasets. Detailed experiments confirmed that our approach substantially improves robustness against gradient-based targeted and untargeted attack threats. And, we showed that the increased non-linearity at the output layer has some additional benefits against some other attack methods like Deepfool attack.

摘要: 在标准的基于深度神经网络(DNN)的分类器中，一般的做法是省略最后一层(输出层)的激活函数，直接对Logit应用Softmax函数来得到每一类的概率得分。在这种类型的体系结构中，分类器相对于任何输出类别的损失值与最终概率得分和相关类别的标签值之间的差值成正比。标准的白盒对抗性规避攻击，无论是有针对性的还是无针对性的，主要是利用模型损失函数的梯度来伪造对抗性样本，愚弄模型。在这项研究中，我们从数学和实验两个方面证明了在具有高温值的模型输出层使用一些广为人知的激活函数具有将目标攻击和非目标攻击的梯度归零的效果，防止攻击者利用模型的损失函数来伪造敌意样本。我们已经在MNIST(数字)、CIFAR10数据集上实验验证了我们的方法的有效性。详细的实验证实，我们的方法大大提高了对基于梯度的目标攻击和非目标攻击威胁的鲁棒性。并且，我们还表明，在输出层增加的非线性比其他一些攻击方法(如DeepfoOff攻击)有一些额外的好处。



## **5. Unity is strength: Improving the Detection of Adversarial Examples with Ensemble Approaches**

团结就是力量：用集成方法改进对抗性实例的检测 cs.CV

Code is available at https://github.com/BIMIB-DISCo/ENAD-experiments

**SubmitDate**: 2022-02-15    [paper-pdf](http://arxiv.org/pdf/2111.12631v3)

**Authors**: Francesco Craighero, Fabrizio Angaroni, Fabio Stella, Chiara Damiani, Marco Antoniotti, Alex Graudenzi

**Abstracts**: A key challenge in computer vision and deep learning is the definition of robust strategies for the detection of adversarial examples. Here, we propose the adoption of ensemble approaches to leverage the effectiveness of multiple detectors in exploiting distinct properties of the input data. To this end, the ENsemble Adversarial Detector (ENAD) framework integrates scoring functions from state-of-the-art detectors based on Mahalanobis distance, Local Intrinsic Dimensionality, and One-Class Support Vector Machines, which process the hidden features of deep neural networks. ENAD is designed to ensure high standardization and reproducibility to the computational workflow. Importantly, extensive tests on benchmark datasets, models and adversarial attacks show that ENAD outperforms all competing methods in the large majority of settings. The improvement over the state-of-the-art and the intrinsic generality of the framework, which allows one to easily extend ENAD to include any set of detectors, set the foundations for the new area of ensemble adversarial detection.

摘要: 计算机视觉和深度学习中的一个关键挑战是定义用于检测对抗性示例的鲁棒策略。在这里，我们建议采用集成方法来利用多个检测器的有效性来利用输入数据的不同属性。为此，集成敌意检测器(ENAD)框架集成了基于马氏距离、局部本征维数和一类支持向量机的最新检测器的评分函数，这些功能处理了深层神经网络的隐藏特征。ENAD旨在确保计算工作流的高度标准化和重复性。重要的是，对基准数据集、模型和对抗性攻击的广泛测试表明，ENAD在绝大多数情况下都优于所有竞争方法。对现有技术的改进和框架固有的通用性，使得人们可以很容易地将ENAD扩展到包括任何一组检测器，为集成对手检测的新领域奠定了基础。



## **6. Layer-wise Regularized Adversarial Training using Layers Sustainability Analysis (LSA) framework**

基于层次可持续性分析(LSA)框架的分层正则化对抗性训练 cs.CV

Layers Sustainability Analysis (LSA) framework

**SubmitDate**: 2022-02-15    [paper-pdf](http://arxiv.org/pdf/2202.02626v3)

**Authors**: Mohammad Khalooei, Mohammad Mehdi Homayounpour, Maryam Amirmazlaghani

**Abstracts**: Deep neural network models are used today in various applications of artificial intelligence, the strengthening of which, in the face of adversarial attacks is of particular importance. An appropriate solution to adversarial attacks is adversarial training, which reaches a trade-off between robustness and generalization. This paper introduces a novel framework (Layer Sustainability Analysis (LSA)) for the analysis of layer vulnerability in an arbitrary neural network in the scenario of adversarial attacks. LSA can be a helpful toolkit to assess deep neural networks and to extend the adversarial training approaches towards improving the sustainability of model layers via layer monitoring and analysis. The LSA framework identifies a list of Most Vulnerable Layers (MVL list) of the given network. The relative error, as a comparison measure, is used to evaluate representation sustainability of each layer against adversarial inputs. The proposed approach for obtaining robust neural networks to fend off adversarial attacks is based on a layer-wise regularization (LR) over LSA proposal(s) for adversarial training (AT); i.e. the AT-LR procedure. AT-LR could be used with any benchmark adversarial attack to reduce the vulnerability of network layers and to improve conventional adversarial training approaches. The proposed idea performs well theoretically and experimentally for state-of-the-art multilayer perceptron and convolutional neural network architectures. Compared with the AT-LR and its corresponding base adversarial training, the classification accuracy of more significant perturbations increased by 16.35%, 21.79%, and 10.730% on Moon, MNIST, and CIFAR-10 benchmark datasets, respectively. The LSA framework is available and published at https://github.com/khalooei/LSA.

摘要: 深度神经网络模型在当今人工智能的各种应用中都有应用，在面对敌意攻击时，加强深度神经网络模型的应用显得尤为重要。对抗性攻击的一个合适的解决方案是对抗性训练，它在鲁棒性和泛化之间达到了折衷。提出了一种新的分析任意神经网络层脆弱性的框架(层可持续性分析LSA)，用于分析任意神经网络在敌意攻击情况下的层脆弱性。LSA可作为评估深层神经网络和扩展对抗性训练方法的有用工具包，以便通过层监控和分析来提高模型层的可持续性。LSA框架标识给定网络的最易受攻击的层的列表(MVL列表)。以相对误差作为比较尺度，评价各层对敌方输入的表征可持续性。所提出的获得鲁棒神经网络以抵御对手攻击的方法是基于基于LSA的对抗性训练(AT)方案的分层正则化(LR)，即AT-LR过程。AT-LR可以与任何基准对抗性攻击一起使用，以降低网络层的脆弱性，并改进传统的对抗性训练方法。对于最先进的多层感知器和卷积神经网络结构，所提出的思想在理论和实验上都表现良好。在MOND、MNIST和CIFAR-10基准数据集上，与AT-LR及其相应的基础对抗性训练相比，更显著扰动的分类准确率分别提高了16.35%、21.79%和10.730%。可以在https://github.com/khalooei/LSA.上获得并发布lsa框架。



## **7. Holistic Adversarial Robustness of Deep Learning Models**

深度学习模型的整体对抗鲁棒性 cs.LG

survey paper on holistic adversarial robustness for deep learning

**SubmitDate**: 2022-02-15    [paper-pdf](http://arxiv.org/pdf/2202.07201v1)

**Authors**: Pin-Yu Chen, Sijia Liu

**Abstracts**: Adversarial robustness studies the worst-case performance of a machine learning model to ensure safety and reliability. With the proliferation of deep-learning based technology, the potential risks associated with model development and deployment can be amplified and become dreadful vulnerabilities. This paper provides a comprehensive overview of research topics and foundational principles of research methods for adversarial robustness of deep learning models, including attacks, defenses, verification, and novel applications.

摘要: 对抗鲁棒性研究机器学习模型的最坏情况性能，以确保安全性和可靠性。随着基于深度学习的技术的激增，与模型开发和部署相关的潜在风险可能会被放大，并成为可怕的漏洞。本文综述了深度学习模型对抗性稳健性的研究主题和基本原理，包括攻击、防御、验证和新的应用。



## **8. Resilience from Diversity: Population-based approach to harden models against adversarial attacks**

来自多样性的弹性：基于人口的方法来强化模型对抗对手攻击的能力 cs.LG

12 pages, 6 figures, 5 tables

**SubmitDate**: 2022-02-15    [paper-pdf](http://arxiv.org/pdf/2111.10272v2)

**Authors**: Jasser Jasser, Ivan Garibay

**Abstracts**: Traditional deep learning networks (DNN) exhibit intriguing vulnerabilities that allow an attacker to force them to fail at their task. Notorious attacks such as the Fast Gradient Sign Method (FGSM) and the more powerful Projected Gradient Descent (PGD) generate adversarial samples by adding a magnitude of perturbation $\epsilon$ to the input's computed gradient, resulting in a deterioration of the effectiveness of the model's classification. This work introduces a model that is resilient to adversarial attacks. Our model leverages an established mechanism of defense which utilizes randomness and a population of DNNs. More precisely, our model consists of a population of $n$ diverse submodels, each one of them trained to individually obtain a high accuracy for the task at hand, while forced to maintain meaningful differences in their weights. Each time our model receives a classification query, it selects a submodel from its population at random to answer the query. To counter the attack transferability, diversity is introduced and maintained in the population of submodels. Thus introducing the concept of counter linking weights. A Counter-Linked Model (CLM) consists of a population of DNNs of the same architecture where a periodic random similarity examination is conducted during the simultaneous training to guarantee diversity while maintaining accuracy. Though the randomization technique proved to be resilient against adversarial attacks, we show that by retraining the DNNs ensemble or training them from the start with counter linking would enhance the robustness by around 20\% when tested on the MNIST dataset and at least 15\% when tested on the CIFAR-10 dataset. When CLM is coupled with adversarial training, this defense mechanism achieves state-of-the-art robustness.

摘要: 传统的深度学习网络(DNN)表现出耐人寻味的漏洞，使得攻击者能够迫使它们在任务中失败。诸如快速梯度符号法(FGSM)和更强大的投影梯度下降法(PGD)等臭名昭著的攻击通过在输入的计算梯度上添加扰动幅度$\ε$来生成敌意样本，导致模型分类效果的恶化。这项工作引入了一个对对手攻击具有弹性的模型。我们的模型利用了一种已建立的防御机制，该机制利用了随机性和DNN的种群。更准确地说，我们的模型由$n$各式各样的子模型组成，每个子模型都经过训练，以单独获得手头任务的高精度，同时被迫保持有意义的权重差异。我们的模型每次收到分类查询时，都会从其总体中随机选择一个子模型来回答查询。为了对抗攻击的可传递性，在子模型种群中引入并保持多样性。从而引入了计数器链接权重的概念。反向链接模型(CLM)由同一体系结构的一组DNN组成，其中在同时训练期间进行周期性的随机相似性检查，以在保持准确性的同时保证多样性。虽然随机化技术被证明对对手攻击具有弹性，但我们表明，通过重新训练DNN集成或从计数器链接开始训练DNN集成，在MNIST数据集上测试时将鲁棒性提高约20\%，在CIFAR-10数据集上测试时至少提高15\%。当CLM与对抗性训练相结合时，这种防御机制实现了最先进的健壮性。



## **9. Recent Advances in Reliable Deep Graph Learning: Adversarial Attack, Inherent Noise, and Distribution Shift**

可靠深度图学习的最新进展：对抗性攻击、固有噪声和分布偏移 cs.LG

**SubmitDate**: 2022-02-15    [paper-pdf](http://arxiv.org/pdf/2202.07114v1)

**Authors**: Bingzhe Wu, Jintang Li, Chengbin Hou, Guoji Fu, Yatao Bian, Liang Chen, Junzhou Huang

**Abstracts**: Deep graph learning (DGL) has achieved remarkable progress in both business and scientific areas ranging from finance and e-commerce to drug and advanced material discovery. Despite the progress, applying DGL to real-world applications faces a series of reliability threats including adversarial attacks, inherent noise, and distribution shift. This survey aims to provide a comprehensive review of recent advances for improving the reliability of DGL algorithms against the above threats. In contrast to prior related surveys which mainly focus on adversarial attacks and defense, our survey covers more reliability-related aspects of DGL, i.e., inherent noise and distribution shift. Additionally, we discuss the relationships among above aspects and highlight some important issues to be explored in future research.

摘要: 深度图学习(DGL)在从金融和电子商务到药物和先进材料发现的商业和科学领域都取得了显着的进展。尽管取得了进展，但将DGL应用于实际应用程序面临着一系列的可靠性威胁，包括敌意攻击、固有噪声和分布转移。本调查旨在全面回顾提高DGL算法抵御上述威胁的可靠性的最新进展。与以前主要关注对抗性攻击和防御的相关调查不同，我们的调查涵盖了DGL更多与可靠性相关的方面，即固有噪声和分布偏移。此外，我们还讨论了上述几个方面之间的关系，并指出了未来研究中需要探索的一些重要问题。



## **10. Universal Adversarial Examples in Remote Sensing: Methodology and Benchmark**

遥感领域的普遍对抗性实例：方法论和基准 cs.CV

**SubmitDate**: 2022-02-14    [paper-pdf](http://arxiv.org/pdf/2202.07054v1)

**Authors**: Yonghao Xu, Pedram Ghamisi

**Abstracts**: Deep neural networks have achieved great success in many important remote sensing tasks. Nevertheless, their vulnerability to adversarial examples should not be neglected. In this study, we systematically analyze the universal adversarial examples in remote sensing data for the first time, without any knowledge from the victim model. Specifically, we propose a novel black-box adversarial attack method, namely Mixup-Attack, and its simple variant Mixcut-Attack, for remote sensing data. The key idea of the proposed methods is to find common vulnerabilities among different networks by attacking the features in the shallow layer of a given surrogate model. Despite their simplicity, the proposed methods can generate transferable adversarial examples that deceive most of the state-of-the-art deep neural networks in both scene classification and semantic segmentation tasks with high success rates. We further provide the generated universal adversarial examples in the dataset named UAE-RS, which is the first dataset that provides black-box adversarial samples in the remote sensing field. We hope UAE-RS may serve as a benchmark that helps researchers to design deep neural networks with strong resistance toward adversarial attacks in the remote sensing field. Codes and the UAE-RS dataset will be available online.

摘要: 深度神经网络在许多重要的遥感任务中取得了巨大的成功。然而，他们面对敌对例子的脆弱性不应被忽视。在本研究中，我们在没有任何受害者模型知识的情况下，首次系统地分析了遥感数据中普遍存在的对抗性实例。具体地说，针对遥感数据，我们提出了一种新的黑盒对抗攻击方法，即Mixup-Attack及其简单的变种MixCut-Attack。该方法的核心思想是通过攻击给定代理模型浅层的特征来发现不同网络之间的共同漏洞。尽管方法简单，但是在场景分类和语义分割任务中，所提出的方法可以生成可转移的对抗性示例，欺骗了大多数最新的深度神经网络，并且成功率很高。此外，我们还给出了在遥感领域第一个提供黑盒对抗性样本的数据集UAE-RS中生成的通用对抗性实例。我们希望UAE-RS可以作为一个基准，帮助研究人员设计出对遥感领域的敌意攻击具有很强抵抗力的深层神经网络。代码和阿联酋-RS数据集将在网上提供。



## **11. White-Box Attacks on Hate-speech BERT Classifiers in German with Explicit and Implicit Character Level Defense**

德语仇恨言语BERT分类器的显性和隐性特征防御白盒攻击 cs.CL

**SubmitDate**: 2022-02-14    [paper-pdf](http://arxiv.org/pdf/2202.05778v2)

**Authors**: Shahrukh Khan, Mahnoor Shahid, Navdeeppal Singh

**Abstracts**: In this work, we evaluate the adversarial robustness of BERT models trained on German Hate Speech datasets. We also complement our evaluation with two novel white-box character and word level attacks thereby contributing to the range of attacks available. Furthermore, we also perform a comparison of two novel character-level defense strategies and evaluate their robustness with one another.

摘要: 在这项工作中，我们评估了在德国仇恨语音数据集上训练的BERT模型的对抗鲁棒性。我们还用两种新的白盒字符和词级攻击来补充我们的评估，从而增加了可用的攻击范围。此外，我们还对两种新的字符级防御策略进行了比较，并对它们的鲁棒性进行了评估。



## **12. Robust and Information-theoretically Safe Bias Classifier against Adversarial Attacks**

抗敌意攻击的稳健且信息理论安全的偏向分类器 cs.LG

**SubmitDate**: 2022-02-14    [paper-pdf](http://arxiv.org/pdf/2111.04404v2)

**Authors**: Lijia Yu, Xiao-Shan Gao

**Abstracts**: In this paper, the bias classifier is introduced, that is, the bias part of a DNN with Relu as the activation function is used as a classifier. The work is motivated by the fact that the bias part is a piecewise constant function with zero gradient and hence cannot be directly attacked by gradient-based methods to generate adversaries, such as FGSM. The existence of the bias classifier is proved and an effective training method for the bias classifier is given. It is proved that by adding a proper random first-degree part to the bias classifier, an information-theoretically safe classifier against the original-model gradient attack is obtained in the sense that the attack will generate a totally random attacking direction. This seems to be the first time that the concept of information-theoretically safe classifier is proposed. Several attack methods for the bias classifier are proposed and numerical experiments are used to show that the bias classifier is more robust than DNNs with similar size against these attacks in most cases.

摘要: 本文介绍了偏向分类器，即以RELU为激活函数的DNN的偏向部分作为分类器。这项工作的动机是，偏差部分是一个分段常数函数，具有零梯度，因此不能被基于梯度的方法直接攻击来生成对手，如FGSM。证明了偏向分类器的存在性，并给出了一种有效的偏向分类器训练方法。证明了通过在偏向分类器中加入适当的随机一阶部分，在攻击产生完全随机的攻击方向的意义下，得到了一种信息论上安全的分类器，可以抵抗原模型的梯度攻击。这似乎是首次提出信息理论安全分类器的概念。提出了偏向分类器的几种攻击方法，数值实验表明，在大多数情况下，偏向分类器比大小相近的DNN具有更好的鲁棒性。



## **13. Deduplicating Training Data Mitigates Privacy Risks in Language Models**

对训练数据进行重复数据消除可降低语言模型中的隐私风险 cs.CR

**SubmitDate**: 2022-02-14    [paper-pdf](http://arxiv.org/pdf/2202.06539v1)

**Authors**: Nikhil Kandpal, Eric Wallace, Colin Raffel

**Abstracts**: Past work has shown that large language models are susceptible to privacy attacks, where adversaries generate sequences from a trained model and detect which sequences are memorized from the training set. In this work, we show that the success of these attacks is largely due to duplication in commonly used web-scraped training sets. We first show that the rate at which language models regenerate training sequences is superlinearly related to a sequence's count in the training set. For instance, a sequence that is present 10 times in the training data is on average generated ~1000 times more often than a sequence that is present only once. We next show that existing methods for detecting memorized sequences have near-chance accuracy on non-duplicated training sequences. Finally, we find that after applying methods to deduplicate training data, language models are considerably more secure against these types of privacy attacks. Taken together, our results motivate an increased focus on deduplication in privacy-sensitive applications and a reevaluation of the practicality of existing privacy attacks.

摘要: 过去的工作表明，大型语言模型容易受到隐私攻击，攻击者从训练的模型生成序列，并从训练集中检测哪些序列被记忆。在这项工作中，我们表明这些攻击的成功在很大程度上是由于常用的Web抓取训练集的重复。我们首先证明了语言模型重新生成训练序列的速度与训练集中序列的计数呈超线性关系。例如，在训练数据中出现10次的序列的平均生成频率是只出现一次的序列的~1000倍。接下来，我们展示了现有的检测记忆序列的方法在非重复训练序列上具有近乎概率的准确性。最后，我们发现，在应用方法对训练数据进行去重之后，语言模型对这些类型的隐私攻击的安全性要高得多。综上所述，我们的结果促使人们更加关注隐私敏感应用程序中的重复数据删除，并重新评估现有隐私攻击的实用性。



## **14. Finding Dynamics Preserving Adversarial Winning Tickets**

寻找动态保存的对抗性中奖彩票 cs.LG

Accepted by AISTATS2022

**SubmitDate**: 2022-02-14    [paper-pdf](http://arxiv.org/pdf/2202.06488v1)

**Authors**: Xupeng Shi, Pengfei Zheng, Adam Ding, Yuan Gao, Weizhong Zhang

**Abstracts**: Modern deep neural networks (DNNs) are vulnerable to adversarial attacks and adversarial training has been shown to be a promising method for improving the adversarial robustness of DNNs. Pruning methods have been considered in adversarial context to reduce model capacity and improve adversarial robustness simultaneously in training. Existing adversarial pruning methods generally mimic the classical pruning methods for natural training, which follow the three-stage 'training-pruning-fine-tuning' pipelines. We observe that such pruning methods do not necessarily preserve the dynamics of dense networks, making it potentially hard to be fine-tuned to compensate the accuracy degradation in pruning. Based on recent works of \textit{Neural Tangent Kernel} (NTK), we systematically study the dynamics of adversarial training and prove the existence of trainable sparse sub-network at initialization which can be trained to be adversarial robust from scratch. This theoretically verifies the \textit{lottery ticket hypothesis} in adversarial context and we refer such sub-network structure as \textit{Adversarial Winning Ticket} (AWT). We also show empirical evidences that AWT preserves the dynamics of adversarial training and achieve equal performance as dense adversarial training.

摘要: 现代深层神经网络(DNNs)容易受到敌意攻击，对抗性训练已被证明是提高DNN对抗性鲁棒性的一种很有前途的方法。在训练过程中，考虑了对抗性环境下的剪枝方法，在减少模型容量的同时提高对抗性鲁棒性。现有的对抗性剪枝方法一般是模仿经典的自然训练剪枝方法，遵循“训练-剪枝-微调”三阶段的流水线。我们观察到，这样的剪枝方法并不一定保持密集网络的动态，使得它可能很难被微调来补偿剪枝过程中的精度下降。基于神经切核(NTK)的最新工作，系统地研究了对抗性训练的动力学，证明了在初始化时存在可训练的稀疏子网络，它可以从头开始训练为对抗性健壮性网络。这从理论上验证了对抗性环境下的\text{彩票假设}，我们将这种子网络结构称为\text{对抗性中票}(AWT)。我们还展示了经验证据，AWT保持了对抗性训练的动态性，并获得了与密集对抗性训练相同的性能。



## **15. Robustness against Adversarial Attacks in Neural Networks using Incremental Dissipativity**

基于增量耗散的神经网络对敌意攻击的鲁棒性 cs.LG

**SubmitDate**: 2022-02-14    [paper-pdf](http://arxiv.org/pdf/2111.12906v2)

**Authors**: Bernardo Aquino, Arash Rahnama, Peter Seiler, Lizhen Lin, Vijay Gupta

**Abstracts**: Adversarial examples can easily degrade the classification performance in neural networks. Empirical methods for promoting robustness to such examples have been proposed, but often lack both analytical insights and formal guarantees. Recently, some robustness certificates have appeared in the literature based on system theoretic notions. This work proposes an incremental dissipativity-based robustness certificate for neural networks in the form of a linear matrix inequality for each layer. We also propose an equivalent spectral norm bound for this certificate which is scalable to neural networks with multiple layers. We demonstrate the improved performance against adversarial attacks on a feed-forward neural network trained on MNIST and an Alexnet trained using CIFAR-10.

摘要: 对抗性示例很容易降低神经网络的分类性能。已经提出了提高此类例子稳健性的经验方法，但往往既缺乏分析洞察力，也缺乏形式上的保证。近年来，一些基于系统论概念的健壮性证书出现在文献中。本文以线性矩阵不等式的形式为每一层提出了一种基于耗散性的增量式神经网络鲁棒性证书。我们还给出了该证书的一个等价谱范数界，它可扩展到多层神经网络。我们在使用MNIST训练的前馈神经网络和使用CIFAR-10训练的Alexnet上展示了改进的抗敌意攻击性能。



## **16. Towards Understanding and Defending Input Space Trojans**

对输入空间木马的理解和防御 cs.LG

**SubmitDate**: 2022-02-13    [paper-pdf](http://arxiv.org/pdf/2202.06382v1)

**Authors**: Zhenting Wang, Hailun Ding, Juan Zhai, Shiqing Ma

**Abstracts**: Deep Neural Networks (DNNs) can learn Trojans (or backdoors) from benign or poisoned data, which raises security concerns of using them. By exploiting such Trojans, the adversary can add a fixed input space perturbation to any given input to mislead the model predicting certain outputs (i.e., target labels). In this paper, we analyze such input space Trojans in DNNs, and propose a theory to explain the relationship of a model's decision regions and Trojans: a complete and accurate Trojan corresponds to a hyperplane decision region in the input domain. We provide a formal proof of this theory, and provide empirical evidence to support the theory and its relaxations. Based on our analysis, we design a novel training method that removes Trojans during training even on poisoned datasets, and evaluate our prototype on five datasets and five different attacks. Results show that our method outperforms existing solutions. Code: \url{https://anonymous.4open.science/r/NOLE-84C3}.

摘要: 深度神经网络(DNNs)可以从良性或有毒的数据中学习特洛伊木马程序(或后门程序)，这增加了使用它们的安全问题。通过利用这种特洛伊木马，攻击者可以向任何给定的输入添加固定的输入空间扰动，以误导预测特定输出(即目标标签)的模型。本文分析了DNNs中的这类输入空间木马，提出了一种解释模型决策域与木马关系的理论：一个完整准确的木马对应于输入域中的一个超平面决策域。我们给出了这一理论的形式证明，并提供了支持该理论及其松弛的经验证据。基于我们的分析，我们设计了一种新的训练方法，即使在有毒的数据集上也能在训练过程中清除木马程序，并在五个数据集和五个不同的攻击上对我们的原型进行了评估。结果表明，我们的方法比已有的方法具有更好的性能。编码：\url{https://anonymous.4open.science/r/NOLE-84C3}.



## **17. Adversarial Fine-tuning for Backdoor Defense: Connect Adversarial Examples to Triggered Samples**

用于后门防御的对抗性微调：将对抗性示例连接到触发样本 cs.CV

**SubmitDate**: 2022-02-13    [paper-pdf](http://arxiv.org/pdf/2202.06312v1)

**Authors**: Bingxu Mu, Le Wang, Zhenxing Niu

**Abstracts**: Deep neural networks (DNNs) are known to be vulnerable to backdoor attacks, i.e., a backdoor trigger planted at training time, the infected DNN model would misclassify any testing sample embedded with the trigger as target label. Due to the stealthiness of backdoor attacks, it is hard either to detect or erase the backdoor from infected models. In this paper, we propose a new Adversarial Fine-Tuning (AFT) approach to erase backdoor triggers by leveraging adversarial examples of the infected model. For an infected model, we observe that its adversarial examples have similar behaviors as its triggered samples. Based on such observation, we design the AFT to break the foundation of the backdoor attack (i.e., the strong correlation between a trigger and a target label). We empirically show that, against 5 state-of-the-art backdoor attacks, AFT can effectively erase the backdoor triggers without obvious performance degradation on clean samples, which significantly outperforms existing defense methods.

摘要: 深度神经网络(DNNs)容易受到后门攻击，即在训练时植入后门触发器，被感染的DNN模型会将任何嵌入触发器的测试样本错误分类为目标标签。由于后门攻击的隐蔽性，很难从受感染的模型中检测或删除后门。在本文中，我们提出了一种新的对抗性精调(AFT)方法，通过利用感染模型的对抗性示例来擦除后门触发器。对于感染模型，我们观察到其敌意实例与其触发样本具有相似的行为。基于这样的观察，我们设计了AFT来打破后门攻击的基础(即触发器和目标标签之间的强相关性)。我们的实验表明，对于5种最先进的后门攻击，AFT可以有效地清除后门触发器，在干净样本上没有明显的性能下降，明显优于现有的防御方法。



## **18. Vulnerability-Aware Poisoning Mechanism for Online RL with Unknown Dynamics**

动态未知的在线RL漏洞感知中毒机制 cs.LG

**SubmitDate**: 2022-02-12    [paper-pdf](http://arxiv.org/pdf/2009.00774v4)

**Authors**: Yanchao Sun, Da Huo, Furong Huang

**Abstracts**: Poisoning attacks on Reinforcement Learning (RL) systems could take advantage of RL algorithm's vulnerabilities and cause failure of the learning. However, prior works on poisoning RL usually either unrealistically assume the attacker knows the underlying Markov Decision Process (MDP), or directly apply the poisoning methods in supervised learning to RL. In this work, we build a generic poisoning framework for online RL via a comprehensive investigation of heterogeneous poisoning models in RL. Without any prior knowledge of the MDP, we propose a strategic poisoning algorithm called Vulnerability-Aware Adversarial Critic Poison (VA2C-P), which works for most policy-based deep RL agents, closing the gap that no poisoning method exists for policy-based RL agents. VA2C-P uses a novel metric, stability radius in RL, that measures the vulnerability of RL algorithms. Experiments on multiple deep RL agents and multiple environments show that our poisoning algorithm successfully prevents agents from learning a good policy or teaches the agents to converge to a target policy, with a limited attacking budget.

摘要: 对强化学习(RL)系统的毒化攻击可以利用RL算法的脆弱性，导致学习失败。然而，以往的毒化RL的工作通常要么不切实际地假设攻击者知道潜在的马尔可夫决策过程(MDP)，要么直接将有监督学习中的毒化方法应用于RL。在这项工作中，我们通过对RL中异构中毒模型的全面研究，构建了一个适用于在线RL的通用中毒框架。在对MDP没有任何先验知识的情况下，我们提出了一种策略毒化算法VA2C-P(VA2C-P)，该算法适用于大多数基于策略的深度RL代理，弥补了基于策略的RL代理没有毒化方法的空白。VA2C-P使用了一种新的度量，即RL中的稳定半径，该度量度量了RL算法的脆弱性。在多个深度RL代理和多个环境上的实验表明，我们的中毒算法在有限的攻击预算下，成功地阻止了代理学习好的策略或教导代理收敛到目标策略。



## **19. Local Differential Privacy for Federated Learning in Industrial Settings**

工业环境下联合学习的局部差分隐私 cs.CR

14 pages

**SubmitDate**: 2022-02-12    [paper-pdf](http://arxiv.org/pdf/2202.06053v1)

**Authors**: M. A. P. Chamikara, Dongxi Liu, Seyit Camtepe, Surya Nepal, Marthie Grobler, Peter Bertok, Ibrahim Khalil

**Abstracts**: Federated learning (FL) is a collaborative learning approach that has gained much attention due to its inherent privacy preservation capabilities. However, advanced adversarial attacks such as membership inference and model memorization can still make FL vulnerable and potentially leak sensitive private data. Literature shows a few attempts to alleviate this problem by using global (GDP) and local differential privacy (LDP). Compared to GDP, LDP approaches are gaining more popularity due to stronger privacy notions and native support for data distribution. However, DP approaches assume that the server that aggregates the models, to be honest (run the FL protocol honestly) or semi-honest (run the FL protocol honestly while also trying to learn as much information possible), making such approaches unreliable for real-world settings. In real-world industrial environments (e.g. healthcare), the distributed entities (e.g. hospitals) are already composed of locally running machine learning models (e.g. high-performing deep neural networks on local health records). Existing approaches do not provide a scalable mechanism to utilize such settings for privacy-preserving FL. This paper proposes a new local differentially private FL (named LDPFL) protocol for industrial settings. LDPFL avoids the requirement of an honest or a semi-honest server and provides better performance while enforcing stronger privacy levels compared to existing approaches. Our experimental evaluation of LDPFL shows high FL model performance (up to ~98%) under a small privacy budget (e.g. epsilon = 0.5) in comparison to existing methods.

摘要: 联合学习(FL)是一种协作学习方式，由于其固有的隐私保护能力而备受关注。然而，高级对抗性攻击，如成员推断和模型记忆，仍然会使FL容易受到攻击，并可能泄露敏感的私有数据。文献显示，有几种尝试通过使用全局(GDP)和本地差异隐私(LDP)来缓解此问题。与GDP相比，由于更强的隐私概念和对数据分发的本地支持，LDP方法越来越受欢迎。然而，DP方法假定聚合模型的服务器诚实地(诚实地运行FL协议)或半诚实地(诚实地运行FL协议，同时还试图了解尽可能多的信息)，使得这种方法对于现实世界设置是不可靠的。在真实的工业环境中(例如医疗保健)，分布式实体(例如医院)已经由本地运行的机器学习模型(例如基于本地健康记录的高性能深度神经网络)组成。现有方法没有提供可扩展的机制来利用这种设置来保护隐私FL。提出了一种新的适用于工业环境的局部差分私有FL(简称LDPFL)协议。与现有方法相比，LDPFL避免了对诚实或半诚实服务器的要求，并提供了更好的性能，同时实施了更强的隐私级别。我们对LDPFL的实验评估表明，与现有方法相比，在较小的隐私预算(例如ε=0.5)下，FL模型的性能较高(高达~98%)。



## **20. RoPGen: Towards Robust Code Authorship Attribution via Automatic Coding Style Transformation**

RoPGen：通过自动代码风格转换实现健壮的代码作者属性 cs.CR

ICSE 2022

**SubmitDate**: 2022-02-12    [paper-pdf](http://arxiv.org/pdf/2202.06043v1)

**Authors**: Zhen Li, Guenevere, Chen, Chen Chen, Yayi Zou, Shouhuai Xu

**Abstracts**: Source code authorship attribution is an important problem often encountered in applications such as software forensics, bug fixing, and software quality analysis. Recent studies show that current source code authorship attribution methods can be compromised by attackers exploiting adversarial examples and coding style manipulation. This calls for robust solutions to the problem of code authorship attribution. In this paper, we initiate the study on making Deep Learning (DL)-based code authorship attribution robust. We propose an innovative framework called Robust coding style Patterns Generation (RoPGen), which essentially learns authors' unique coding style patterns that are hard for attackers to manipulate or imitate. The key idea is to combine data augmentation and gradient augmentation at the adversarial training phase. This effectively increases the diversity of training examples, generates meaningful perturbations to gradients of deep neural networks, and learns diversified representations of coding styles. We evaluate the effectiveness of RoPGen using four datasets of programs written in C, C++, and Java. Experimental results show that RoPGen can significantly improve the robustness of DL-based code authorship attribution, by respectively reducing 22.8% and 41.0% of the success rate of targeted and untargeted attacks on average.

摘要: 源代码作者归属是软件取证、缺陷修复、软件质量分析等应用中经常遇到的重要问题。最近的研究表明，现有的源代码作者归属方法可能会受到攻击者利用敌意示例和代码风格操纵的影响。这就要求对代码作者归属问题提出可靠的解决方案。本文针对基于深度学习(DL)的代码作者属性鲁棒性问题展开研究。我们提出了一种称为鲁棒编码样式模式生成(RoPGen)的创新框架，它本质上学习了攻击者难以操纵或模仿的作者独特的编码样式模式。其核心思想是在对抗性训练阶段将数据增强和梯度增强相结合。这有效地增加了训练样本的多样性，对深度神经网络的梯度产生了有意义的扰动，并学习了编码风格的多样化表示。我们使用四个用C、C++和Java编写的程序数据集来评估RoPGen的有效性。实验结果表明，RoPGen能够显著提高基于DL的代码作者属性的鲁棒性，目标攻击成功率平均降低22.8%，非目标攻击成功率平均降低41.0%。



## **21. Robust Deep Semi-Supervised Learning: A Brief Introduction**

鲁棒深度半监督学习：简介 cs.LG

**SubmitDate**: 2022-02-12    [paper-pdf](http://arxiv.org/pdf/2202.05975v1)

**Authors**: Lan-Zhe Guo, Zhi Zhou, Yu-Feng Li

**Abstracts**: Semi-supervised learning (SSL) is the branch of machine learning that aims to improve learning performance by leveraging unlabeled data when labels are insufficient. Recently, SSL with deep models has proven to be successful on standard benchmark tasks. However, they are still vulnerable to various robustness threats in real-world applications as these benchmarks provide perfect unlabeled data, while in realistic scenarios, unlabeled data could be corrupted. Many researchers have pointed out that after exploiting corrupted unlabeled data, SSL suffers severe performance degradation problems. Thus, there is an urgent need to develop SSL algorithms that could work robustly with corrupted unlabeled data. To fully understand robust SSL, we conduct a survey study. We first clarify a formal definition of robust SSL from the perspective of machine learning. Then, we classify the robustness threats into three categories: i) distribution corruption, i.e., unlabeled data distribution is mismatched with labeled data; ii) feature corruption, i.e., the features of unlabeled examples are adversarially attacked; and iii) label corruption, i.e., the label distribution of unlabeled data is imbalanced. Under this unified taxonomy, we provide a thorough review and discussion of recent works that focus on these issues. Finally, we propose possible promising directions within robust SSL to provide insights for future research.

摘要: 半监督学习(SSL)是机器学习的一个分支，其目的是在标签不足时通过利用未标记的数据来提高学习性能。最近，具有深度模型的SSL在标准基准任务中被证明是成功的。然而，它们在现实应用程序中仍然容易受到各种健壮性威胁，因为这些基准测试提供了完美的未标记数据，而在现实场景中，未标记数据可能会被破坏。许多研究人员指出，在利用损坏的未标记数据之后，SSL面临严重的性能下降问题。因此，迫切需要开发能够稳健地处理损坏的未标记数据的SSL算法。为了全面了解健壮的SSL，我们进行了一项调查研究。我们首先从机器学习的角度阐明了鲁棒SSL的形式化定义。然后，我们将健壮性威胁分为三类：i)分布损坏，即未标记数据分布与已标记数据不匹配；ii)特征损坏，即未标记示例的特征受到恶意攻击；iii)标签损坏，即未标记数据的标签分布不平衡。在这个统一的分类法下，我们对最近集中在这些问题上的工作进行了彻底的回顾和讨论。最后，我们提出了健壮SSL中可能有前景的方向，为未来的研究提供了见解。



## **22. Measuring the Contribution of Multiple Model Representations in Detecting Adversarial Instances**

测量多个模型表示在检测对抗性实例中的贡献 cs.LG

Correction: replaced "model-wise" with "unit-wise" in the first  sentence of Section 3.2

**SubmitDate**: 2022-02-12    [paper-pdf](http://arxiv.org/pdf/2111.07035v2)

**Authors**: Daniel Steinberg, Paul Munro

**Abstracts**: Deep learning models have been used for a wide variety of tasks. They are prevalent in computer vision, natural language processing, speech recognition, and other areas. While these models have worked well under many scenarios, it has been shown that they are vulnerable to adversarial attacks. This has led to a proliferation of research into ways that such attacks could be identified and/or defended against. Our goal is to explore the contribution that can be attributed to using multiple underlying models for the purpose of adversarial instance detection. Our paper describes two approaches that incorporate representations from multiple models for detecting adversarial examples. We devise controlled experiments for measuring the detection impact of incrementally utilizing additional models. For many of the scenarios we consider, the results show that performance increases with the number of underlying models used for extracting representations.

摘要: 深度学习模型已被广泛用于各种任务。它们广泛应用于计算机视觉、自然语言处理、语音识别等领域。虽然这些模型在许多情况下都工作得很好，但已经表明它们很容易受到对手的攻击。这导致了对如何识别和/或防御此类攻击的研究激增。我们的目标是探索可以归因于使用多个底层模型进行对抗性实例检测的贡献。我们的论文描述了两种方法，它们融合了来自多个模型的表示，用于检测对抗性示例。我们设计了对照实验来衡量增量利用额外模型的检测影响。对于我们考虑的许多场景，结果显示性能随着用于提取表示的底层模型数量的增加而提高。



## **23. Adversarial Attacks and Defense Methods for Power Quality Recognition**

电能质量识别中的对抗性攻击与防御方法 cs.CR

Technical report

**SubmitDate**: 2022-02-11    [paper-pdf](http://arxiv.org/pdf/2202.07421v1)

**Authors**: Jiwei Tian, Buhong Wang, Jing Li, Zhen Wang, Mete Ozay

**Abstracts**: Vulnerability of various machine learning methods to adversarial examples has been recently explored in the literature. Power systems which use these vulnerable methods face a huge threat against adversarial examples. To this end, we first propose a signal-specific method and a universal signal-agnostic method to attack power systems using generated adversarial examples. Black-box attacks based on transferable characteristics and the above two methods are also proposed and evaluated. We then adopt adversarial training to defend systems against adversarial attacks. Experimental analyses demonstrate that our signal-specific attack method provides less perturbation compared to the FGSM (Fast Gradient Sign Method), and our signal-agnostic attack method can generate perturbations fooling most natural signals with high probability. What's more, the attack method based on the universal signal-agnostic algorithm has a higher transfer rate of black-box attacks than the attack method based on the signal-specific algorithm. In addition, the results show that the proposed adversarial training improves robustness of power systems to adversarial examples.

摘要: 最近在文献中探讨了各种机器学习方法对对抗性示例的脆弱性。使用这些易受攻击的方法的电力系统在对抗对手的例子中面临着巨大的威胁。为此，我们首先提出了一种信号特定的方法和一种通用的信号不可知的方法来利用生成的对抗性实例来攻击电力系统。提出了基于可转移特征的黑盒攻击方法，并对这两种方法进行了评估。然后，我们采用对抗性训练来保护系统免受对抗性攻击。实验分析表明，与快速梯度符号方法(FGSM)相比，我们的信号特定攻击方法提供了更少的扰动，并且我们的信号不可知攻击方法可以高概率地产生欺骗大多数自然信号的扰动。此外，基于通用信号不可知算法的攻击方法比基于特定信号算法的攻击方法具有更高的黑盒攻击传递率。此外，结果表明，所提出的对抗性训练提高了电力系统对对抗性示例的鲁棒性。



## **24. Are socially-aware trajectory prediction models really socially-aware?**

具有社会性的轨迹预测模型真的具有社会性吗？ cs.CV

**SubmitDate**: 2022-02-11    [paper-pdf](http://arxiv.org/pdf/2108.10879v2)

**Authors**: Saeed Saadatnejad, Mohammadhossein Bahari, Pedram Khorsandi, Mohammad Saneian, Seyed-Mohsen Moosavi-Dezfooli, Alexandre Alahi

**Abstracts**: Our field has recently witnessed an arms race of neural network-based trajectory predictors. While these predictors are at the core of many applications such as autonomous navigation or pedestrian flow simulations, their adversarial robustness has not been carefully studied. In this paper, we introduce a socially-attended attack to assess the social understanding of prediction models in terms of collision avoidance. An attack is a small yet carefully-crafted perturbations to fail predictors. Technically, we define collision as a failure mode of the output, and propose hard- and soft-attention mechanisms to guide our attack. Thanks to our attack, we shed light on the limitations of the current models in terms of their social understanding. We demonstrate the strengths of our method on the recent trajectory prediction models. Finally, we show that our attack can be employed to increase the social understanding of state-of-the-art models. The code is available online: https://s-attack.github.io/

摘要: 我们的领域最近见证了一场基于神经网络的轨迹预测器的军备竞赛。虽然这些预报器是许多应用的核心，如自主导航或行人流量模拟，但它们的对抗性健壮性还没有得到仔细的研究。在这篇文章中，我们引入了一个社交参与的攻击来评估社会对预测模型在避免碰撞方面的理解。攻击是一个小的，但精心设计的扰动失败的预报器。在技术上，我们将碰撞定义为输出的一种失效模式，并提出了硬注意和软注意机制来指导我们的攻击。多亏了我们的攻击，我们揭示了当前模型在社会理解方面的局限性。我们在最近的轨迹预测模型上展示了我们的方法的优势。最后，我们展示了我们的攻击可以用来增加社会对最先进模型的理解。代码可在网上获得：https://s-attack.github.io/



## **25. Using Random Perturbations to Mitigate Adversarial Attacks on Sentiment Analysis Models**

利用随机扰动缓解情感分析模型上的敌意攻击 cs.CL

To be published in the proceedings for the 18th International  Conference on Natural Language Processing (ICON 2021)

**SubmitDate**: 2022-02-11    [paper-pdf](http://arxiv.org/pdf/2202.05758v1)

**Authors**: Abigail Swenor, Jugal Kalita

**Abstracts**: Attacks on deep learning models are often difficult to identify and therefore are difficult to protect against. This problem is exacerbated by the use of public datasets that typically are not manually inspected before use. In this paper, we offer a solution to this vulnerability by using, during testing, random perturbations such as spelling correction if necessary, substitution by random synonym, or simply dropping the word. These perturbations are applied to random words in random sentences to defend NLP models against adversarial attacks. Our Random Perturbations Defense and Increased Randomness Defense methods are successful in returning attacked models to similar accuracy of models before attacks. The original accuracy of the model used in this work is 80% for sentiment classification. After undergoing attacks, the accuracy drops to accuracy between 0% and 44%. After applying our defense methods, the accuracy of the model is returned to the original accuracy within statistical significance.

摘要: 针对深度学习模型的攻击通常很难识别，因此很难防范。使用通常不会在使用前手动检查的公共数据集加剧了此问题。在本文中，我们通过在测试期间使用随机扰动(如必要时进行拼写更正、替换为随机同义词或简单地删除单词)来解决此漏洞。这些扰动被应用于随机句子中的随机词，以保护NLP模型免受对手攻击。我们的随机扰动防御和增加的随机性防御方法成功地将被攻击的模型恢复到攻击前模型的类似精度。本文使用的模型对情感分类的原始正确率为80%。在遭受攻击后，准确率下降到0%到44%之间。应用我们的防御方法后，模型的精度在统计意义上恢复到原来的精度。



## **26. On the Detection of Adaptive Adversarial Attacks in Speaker Verification Systems**

说话人确认系统中自适应攻击检测的研究 cs.CR

**SubmitDate**: 2022-02-11    [paper-pdf](http://arxiv.org/pdf/2202.05725v1)

**Authors**: Zesheng Chen

**Abstracts**: Speaker verification systems have been widely used in smart phones and Internet of things devices to identify a legitimate user. In recent work, it has been shown that adversarial attacks, such as FAKEBOB, can work effectively against speaker verification systems. The goal of this paper is to design a detector that can distinguish an original audio from an audio contaminated by adversarial attacks. Specifically, our designed detector, called MEH-FEST, calculates the minimum energy in high frequencies from the short-time Fourier transform of an audio and uses it as a detection metric. Through both analysis and experiments, we show that our proposed detector is easy to implement, fast to process an input audio, and effective in determining whether an audio is corrupted by FAKEBOB attacks. The experimental results indicate that the detector is extremely effective: with near zero false positive and false negative rates for detecting FAKEBOB attacks in Gaussian mixture model (GMM) and i-vector speaker verification systems. Moreover, adaptive adversarial attacks against our proposed detector and their countermeasures are discussed and studied, showing the game between attackers and defenders.

摘要: 说话人验证系统已广泛应用于智能手机和物联网设备中，用于识别合法用户。最近的研究表明，FAKEBOB等对抗性攻击可以有效地对抗说话人确认系统。本文的目标是设计一种能够区分原始音频和被敌意攻击污染的音频的检测器。具体地说，我们设计的检测器，称为MEH-FEST，从音频的短时傅立叶变换计算高频最小能量，并将其用作检测度量。通过分析和实验表明，我们提出的检测器实现简单，处理输入音频的速度快，能有效地判断音频是否被FAKEBOB攻击破坏。实验结果表明，该检测器对混合高斯模型(GMM)和I向量说话人确认系统中FAKEBOB攻击的检测非常有效，误报率和误报率都接近于零。此外，还讨论和研究了针对我们提出的检测器的自适应对抗性攻击及其对策，展示了攻击者和防御者之间的博弈。



## **27. Towards Adversarially Robust Deepfake Detection: An Ensemble Approach**

面向对抗性强健的深伪检测：一种集成方法 cs.LG

**SubmitDate**: 2022-02-11    [paper-pdf](http://arxiv.org/pdf/2202.05687v1)

**Authors**: Ashish Hooda, Neal Mangaokar, Ryan Feng, Kassem Fawaz, Somesh Jha, Atul Prakash

**Abstracts**: Detecting deepfakes is an important problem, but recent work has shown that DNN-based deepfake detectors are brittle against adversarial deepfakes, in which an adversary adds imperceptible perturbations to a deepfake to evade detection. In this work, we show that a modification to the detection strategy in which we replace a single classifier with a carefully chosen ensemble, in which input transformations for each model in the ensemble induces pairwise orthogonal gradients, can significantly improve robustness beyond the de facto solution of adversarial training. We present theoretical results to show that such orthogonal gradients can help thwart a first-order adversary by reducing the dimensionality of the input subspace in which adversarial deepfakes lie. We validate the results empirically by instantiating and evaluating a randomized version of such "orthogonal" ensembles for adversarial deepfake detection and find that these randomized ensembles exhibit significantly higher robustness as deepfake detectors compared to state-of-the-art deepfake detectors against adversarial deepfakes, even those created using strong PGD-500 attacks.

摘要: 深度伪码的检测是一个重要的问题，但最近的研究表明，基于DNN的深度伪码检测器对敌意的深度伪码是脆弱的，在这种情况下，敌手通过向深度伪码添加不可察觉的扰动来逃避检测。在这项工作中，我们证明了对检测策略的修改，即用精心选择的集成来取代单个分类器，其中集成中每个模型的输入变换都会诱导成对的正交梯度，可以显著提高鲁棒性，而不是对抗性训练的事实解决方案。我们给出的理论结果表明，这种正交梯度可以通过降低敌意深伪所在的输入子空间的维数来帮助挫败一阶敌方。我们通过实例化和评估这种用于对抗性深度伪检测的“正交”集成的随机化版本来实证验证结果，并发现这些随机化集成在对抗对抗性深伪(即使是使用强PGD-500攻击创建的深伪)时，与最新的深伪检测器相比，表现出明显更高的稳健性。



## **28. FAAG: Fast Adversarial Audio Generation through Interactive Attack Optimisation**

FAAG：通过交互式攻击优化快速生成敌方音频 cs.SD

**SubmitDate**: 2022-02-11    [paper-pdf](http://arxiv.org/pdf/2202.05416v1)

**Authors**: Yuantian Miao, Chao Chen, Lei Pan, Jun Zhang, Yang Xiang

**Abstracts**: Automatic Speech Recognition services (ASRs) inherit deep neural networks' vulnerabilities like crafted adversarial examples. Existing methods often suffer from low efficiency because the target phases are added to the entire audio sample, resulting in high demand for computational resources. This paper proposes a novel scheme named FAAG as an iterative optimization-based method to generate targeted adversarial examples quickly. By injecting the noise over the beginning part of the audio, FAAG generates adversarial audio in high quality with a high success rate timely. Specifically, we use audio's logits output to map each character in the transcription to an approximate position of the audio's frame. Thus, an adversarial example can be generated by FAAG in approximately two minutes using CPUs only and around ten seconds with one GPU while maintaining an average success rate over 85%. Specifically, the FAAG method can speed up around 60% compared with the baseline method during the adversarial example generation process. Furthermore, we found that appending benign audio to any suspicious examples can effectively defend against the targeted adversarial attack. We hope that this work paves the way for inventing new adversarial attacks against speech recognition with computational constraints.

摘要: 自动语音识别服务(ASR)继承了深层神经网络的弱点，就像精心制作的敌意例子。现有的方法通常效率较低，因为目标相位被添加到整个音频样本，导致对计算资源的高需求。提出了一种新的基于迭代优化的FAAG方案，用于快速生成目标对抗性实例。通过在音频的开始部分注入噪声，FAAG及时生成高质量和高成功率的敌意音频。具体地说，我们使用音频的logits输出将转录中的每个字符映射到音频帧的大致位置。因此，FAAG仅使用CPU就可以在大约2分钟内生成对抗性示例，使用一个GPU可以在大约10秒内生成对抗性示例，同时保持85%以上的平均成功率。具体地说，在对抗性实例生成过程中，与基线方法相比，FAAG方法可以加快60%左右的速度。此外，我们还发现，在任何可疑的示例中添加良性音频可以有效地防御目标攻击。我们希望这项工作为发明新的针对计算受限的语音识别的对抗性攻击铺平道路。



## **29. SoK: Certified Robustness for Deep Neural Networks**

SOK：深度神经网络的认证鲁棒性 cs.LG

14 pages for the main text; recent advances (till Feb 2022) included

**SubmitDate**: 2022-02-10    [paper-pdf](http://arxiv.org/pdf/2009.04131v6)

**Authors**: Linyi Li, Tao Xie, Bo Li

**Abstracts**: Great advances in deep neural networks (DNNs) have led to state-of-the-art performance on a wide range of tasks. However, recent studies have shown that DNNs are vulnerable to adversarial attacks, which have brought great concerns when deploying these models to safety-critical applications such as autonomous driving. Different defense approaches have been proposed against adversarial attacks, including: a) empirical defenses, which usually can be adaptively attacked again without providing robustness certification; and b) certifiably robust approaches which consist of robustness verification providing the lower bound of robust accuracy against any attacks under certain conditions and corresponding robust training approaches. In this paper, we systematize the certifiably robust approaches and related practical and theoretical implications and findings. We also provide the first comprehensive benchmark on existing robustness verification and training approaches on different datasets. In particular, we 1) provide a taxonomy for the robustness verification and training approaches, as well as summarize the methodologies for representative algorithms, 2) reveal the characteristics, strengths, limitations, and fundamental connections among these approaches, 3) discuss current research progresses, theoretical barriers, main challenges, and future directions for certifiably robust approaches for DNNs, and 4) provide an open-sourced unified platform to evaluate over 20 representative certifiably robust approaches for a wide range of DNNs.

摘要: 深度神经网络(DNNs)的巨大进步导致了在广泛任务上的最先进的性能。然而，最近的研究表明，DNN很容易受到敌意攻击，这在将这些模型部署到自动驾驶等安全关键型应用时带来了极大的担忧。针对敌意攻击已经提出了不同的防御方法，包括：a)经验防御，通常无需提供健壮性证明即可自适应地再次攻击；b)可证明健壮性方法，包括在一定条件下提供对任何攻击的鲁棒精度下界的健壮性验证和相应的健壮性训练方法。在这篇文章中，我们系统化的证明稳健的方法和相关的实际和理论意义和发现。我们还提供了关于不同数据集上现有健壮性验证和训练方法的第一个全面基准。特别地，我们1)提供了健壮性验证和训练方法的分类，并总结了典型算法的方法论；2)揭示了这些方法的特点、优点、局限性和基本联系；3)讨论了当前DNNs的研究进展、理论障碍、主要挑战和未来的发展方向；4)提供了一个开源的统一平台来评估20多种具有代表性的DNNs的可证健壮性方法。



## **30. Towards Assessing and Characterizing the Semantic Robustness of Face Recognition**

面向人脸识别的语义健壮性评估与表征 cs.CV

26 pages, 18 figures

**SubmitDate**: 2022-02-10    [paper-pdf](http://arxiv.org/pdf/2202.04978v1)

**Authors**: Juan C. Pérez, Motasem Alfarra, Ali Thabet, Pablo Arbeláez, Bernard Ghanem

**Abstracts**: Deep Neural Networks (DNNs) lack robustness against imperceptible perturbations to their input. Face Recognition Models (FRMs) based on DNNs inherit this vulnerability. We propose a methodology for assessing and characterizing the robustness of FRMs against semantic perturbations to their input. Our methodology causes FRMs to malfunction by designing adversarial attacks that search for identity-preserving modifications to faces. In particular, given a face, our attacks find identity-preserving variants of the face such that an FRM fails to recognize the images belonging to the same identity. We model these identity-preserving semantic modifications via direction- and magnitude-constrained perturbations in the latent space of StyleGAN. We further propose to characterize the semantic robustness of an FRM by statistically describing the perturbations that induce the FRM to malfunction. Finally, we combine our methodology with a certification technique, thus providing (i) theoretical guarantees on the performance of an FRM, and (ii) a formal description of how an FRM may model the notion of face identity.

摘要: 深度神经网络(DNNs)对其输入的不可察觉的扰动缺乏鲁棒性。基于DNN的人脸识别模型(FRM)继承了此漏洞。我们提出了一种方法来评估和表征FRM对其输入的语义扰动的鲁棒性。我们的方法论通过设计对抗性攻击来搜索对人脸的身份保留修改，从而导致FRMS发生故障。特别地，在给定一张人脸的情况下，我们的攻击会找到该人脸的保持身份的变体，使得FRM无法识别属于同一身份的图像。我们在StyleGan的潜在空间中通过方向和幅度约束的扰动来模拟这些保持身份的语义修改。我们进一步提出通过统计描述导致FRM故障的扰动来表征FRM的语义健壮性。最后，我们将我们的方法与认证技术相结合，从而提供(I)FRM性能的理论保证，以及(Ii)FRM如何建模面部身份概念的正式描述。



## **31. Beyond ImageNet Attack: Towards Crafting Adversarial Examples for Black-box Domains**

超越ImageNet攻击：为黑盒领域精心制作敌意示例 cs.CV

Accepted by ICLR 2022

**SubmitDate**: 2022-02-10    [paper-pdf](http://arxiv.org/pdf/2201.11528v3)

**Authors**: Qilong Zhang, Xiaodan Li, Yuefeng Chen, Jingkuan Song, Lianli Gao, Yuan He, Hui Xue

**Abstracts**: Adversarial examples have posed a severe threat to deep neural networks due to their transferable nature. Currently, various works have paid great efforts to enhance the cross-model transferability, which mostly assume the substitute model is trained in the same domain as the target model. However, in reality, the relevant information of the deployed model is unlikely to leak. Hence, it is vital to build a more practical black-box threat model to overcome this limitation and evaluate the vulnerability of deployed models. In this paper, with only the knowledge of the ImageNet domain, we propose a Beyond ImageNet Attack (BIA) to investigate the transferability towards black-box domains (unknown classification tasks). Specifically, we leverage a generative model to learn the adversarial function for disrupting low-level features of input images. Based on this framework, we further propose two variants to narrow the gap between the source and target domains from the data and model perspectives, respectively. Extensive experiments on coarse-grained and fine-grained domains demonstrate the effectiveness of our proposed methods. Notably, our methods outperform state-of-the-art approaches by up to 7.71\% (towards coarse-grained domains) and 25.91\% (towards fine-grained domains) on average. Our code is available at \url{https://github.com/qilong-zhang/Beyond-ImageNet-Attack}.

摘要: 对抗性例子由于其可转移性，对深度神经网络构成了严重的威胁。目前，各种研究都在努力提高模型间的可移植性，大多假设替身模型与目标模型在同一领域进行训练。然而，在现实中，部署的模型的相关信息不太可能泄露。因此，构建一个更实用的黑盒威胁模型来克服这一限制并评估已部署模型的脆弱性是至关重要的。本文在仅知道ImageNet域的情况下，提出了一种超越ImageNet攻击(BIA)来研究向黑盒域(未知分类任务)的可传递性。具体地说，我们利用生成模型来学习破坏输入图像的低层特征的对抗性函数。基于这一框架，我们进一步提出了两种变体，分别从数据和模型的角度来缩小源域和目标域之间的差距。在粗粒域和细粒域上的大量实验证明了我们提出的方法的有效性。值得注意的是，我们的方法平均比最先进的方法高出7.71%(对于粗粒度领域)和25.91%(对于细粒度领域)。我们的代码可在\url{https://github.com/qilong-zhang/Beyond-ImageNet-Attack}.获得



## **32. Adversarial Attack and Defense of YOLO Detectors in Autonomous Driving Scenarios**

自动驾驶场景中YOLO检测器的对抗性攻击与防御 cs.CV

7 pages, 3 figures

**SubmitDate**: 2022-02-10    [paper-pdf](http://arxiv.org/pdf/2202.04781v1)

**Authors**: Jung Im Choi, Qing Tian

**Abstracts**: Visual detection is a key task in autonomous driving, and it serves as one foundation for self-driving planning and control. Deep neural networks have achieved promising results in various computer vision tasks, but they are known to be vulnerable to adversarial attacks. A comprehensive understanding of deep visual detectors' vulnerability is required before people can improve their robustness. However, only a few adversarial attack/defense works have focused on object detection, and most of them employed only classification and/or localization losses, ignoring the objectness aspect. In this paper, we identify a serious objectness-related adversarial vulnerability in YOLO detectors and present an effective attack strategy aiming the objectness aspect of visual detection in autonomous vehicles. Furthermore, to address such vulnerability, we propose a new objectness-aware adversarial training approach for visual detection. Experiments show that the proposed attack targeting the objectness aspect is 45.17% and 43.50% more effective than those generated from classification and/or localization losses on the KITTI and COCO_traffic datasets, respectively. Also, the proposed adversarial defense approach can improve the detectors' robustness against objectness-oriented attacks by up to 21% and 12% mAP on KITTI and COCO_traffic, respectively.

摘要: 视觉检测是自动驾驶中的一项关键任务，是自动驾驶规划和控制的基础之一。深度神经网络在各种计算机视觉任务中取得了令人满意的结果，但众所周知，它们很容易受到对手的攻击。人们需要全面了解深度视觉检测器的脆弱性，才能提高其健壮性。然而，只有少数对抗性攻防研究集中在目标检测上，而且大多只采用分类和/或定位损失，而忽略了客观性方面。本文针对自主车辆视觉检测的客观性方面，识别出YOLO检测器中存在的一个与客观性相关的严重攻击漏洞，并提出了一种有效的攻击策略。此外，为了解决这种脆弱性，我们提出了一种新的基于客观性感知的对抗性视觉检测训练方法。实验表明，针对客观性方面的攻击比基于KITTI和COCO_TRAFFORM数据集的分类和/或定位丢失攻击分别提高了45.17%和43.50%。此外，本文提出的对抗性防御方法可以使检测器对面向对象攻击的鲁棒性分别提高21%和12%MAP在KITTI和COCO_TRAFFORMS上。



## **33. IoTMonitor: A Hidden Markov Model-based Security System to Identify Crucial Attack Nodes in Trigger-action IoT Platforms**

IoTMonitor：基于隐马尔可夫模型的触发物联网平台关键攻击节点识别安全系统 cs.CR

This paper appears in the 2022 IEEE Wireless Communications and  Networking Conference (WCNC 2022). Personal use of this material is  permitted. Permission from IEEE must be obtained for all other uses

**SubmitDate**: 2022-02-09    [paper-pdf](http://arxiv.org/pdf/2202.04620v1)

**Authors**: Md Morshed Alam, Md Sajidul Islam Sajid, Weichao Wang, Jinpeng Wei

**Abstracts**: With the emergence and fast development of trigger-action platforms in IoT settings, security vulnerabilities caused by the interactions among IoT devices become more prevalent. The event occurrence at one device triggers an action in another device, which may eventually contribute to the creation of a chain of events in a network. Adversaries exploit the chain effect to compromise IoT devices and trigger actions of interest remotely just by injecting malicious events into the chain. To address security vulnerabilities caused by trigger-action scenarios, existing research efforts focus on the validation of the security properties of devices or verification of the occurrence of certain events based on their physical fingerprints on a device. We propose IoTMonitor, a security analysis system that discerns the underlying chain of event occurrences with the highest probability by observing a chain of physical evidence collected by sensors. We use the Baum-Welch algorithm to estimate transition and emission probabilities and the Viterbi algorithm to discern the event sequence. We can then identify the crucial nodes in the trigger-action sequence whose compromise allows attackers to reach their final goals. The experiment results of our designed system upon the PEEVES datasets show that we can rebuild the event occurrence sequence with high accuracy from the observations and identify the crucial nodes on the attack paths.

摘要: 随着物联网环境下触发式平台的出现和快速发展，物联网设备之间的交互导致的安全漏洞变得更加普遍。一台设备上发生的事件会触发另一台设备上的操作，这最终可能会导致在网络中创建一系列事件。攻击者只需将恶意事件注入链中，即可利用连锁反应危害物联网设备并远程触发感兴趣的操作。为了解决触发动作场景引起的安全漏洞，现有的研究工作集中在验证设备的安全属性或基于设备上的物理指纹来验证特定事件的发生。我们提出了IoTMonitor，这是一个安全分析系统，它通过观察传感器收集的物理证据链，以最高的概率识别事件发生的潜在链。我们使用Baum-Welch算法来估计转移概率和发射概率，使用Viterbi算法来识别事件序列。然后，我们可以确定触发-动作序列中的关键节点，这些节点的妥协使攻击者能够达到他们的最终目标。我们设计的系统在PEVES数据集上的实验结果表明，我们可以从观测数据中高精度地重建事件发生序列，并识别攻击路径上的关键节点。



## **34. False Memory Formation in Continual Learners Through Imperceptible Backdoor Trigger**

通过潜伏的后门触发器形成持续学习者的错误记忆 cs.LG

**SubmitDate**: 2022-02-09    [paper-pdf](http://arxiv.org/pdf/2202.04479v1)

**Authors**: Muhammad Umer, Robi Polikar

**Abstracts**: In this brief, we show that sequentially learning new information presented to a continual (incremental) learning model introduces new security risks: an intelligent adversary can introduce small amount of misinformation to the model during training to cause deliberate forgetting of a specific task or class at test time, thus creating "false memory" about that task. We demonstrate such an adversary's ability to assume control of the model by injecting "backdoor" attack samples to commonly used generative replay and regularization based continual learning approaches using continual learning benchmark variants of MNIST, as well as the more challenging SVHN and CIFAR 10 datasets. Perhaps most damaging, we show this vulnerability to be very acute and exceptionally effective: the backdoor pattern in our attack model can be imperceptible to human eye, can be provided at any point in time, can be added into the training data of even a single possibly unrelated task and can be achieved with as few as just 1\% of total training dataset of a single task.

摘要: 在这篇简短的文章中，我们展示了顺序学习提供给连续(增量)学习模型的新信息会带来新的安全风险：智能对手可能在训练期间向模型引入少量的错误信息，导致在测试时故意忘记特定任务或类，从而产生关于该任务的“错误记忆”。我们使用MNIST的持续学习基准变体，以及更具挑战性的SVHN和CIFAR10数据集，通过向常用的基于生成性回放和正则化的持续学习方法注入“后门”攻击样本，展示了这样的对手控制模型的能力。也许最具破坏性的是，我们发现这个漏洞是非常尖锐和特别有效的：我们的攻击模型中的后门模式可以是人眼看不见的，可以在任何时间点提供，可以添加到甚至是单个可能不相关的任务的训练数据中，并且可以仅使用单个任务的全部训练数据集的1\%就可以实现。



## **35. ARIBA: Towards Accurate and Robust Identification of Backdoor Attacks in Federated Learning**

Ariba：在联邦学习中实现对后门攻击的准确和鲁棒识别 cs.AI

17 pages, 11 figures

**SubmitDate**: 2022-02-09    [paper-pdf](http://arxiv.org/pdf/2202.04311v1)

**Authors**: Yuxi Mi, Jihong Guan, Shuigeng Zhou

**Abstracts**: The distributed nature and privacy-preserving characteristics of federated learning make it prone to the threat of poisoning attacks, especially backdoor attacks, where the adversary implants backdoors to misguide the model on certain attacker-chosen sub-tasks. In this paper, we present a novel method ARIBA to accurately and robustly identify backdoor attacks in federated learning. By empirical study, we observe that backdoor attacks are discernible by the filters of CNN layers. Based on this finding, we employ unsupervised anomaly detection to evaluate the pre-processed filters and calculate an anomaly score for each client. We then identify the most suspicious clients according to their anomaly scores. Extensive experiments are conducted, which show that our method ARIBA can effectively and robustly defend against multiple state-of-the-art attacks without degrading model performance.

摘要: 联邦学习的分布式性质和隐私保护特性使其容易受到中毒攻击的威胁，特别是后门攻击，其中对手植入后门来在攻击者选择的某些子任务上误导模型。本文提出了一种新的方法ARIBA来准确、鲁棒地识别联邦学习中的后门攻击。通过实证研究，我们观察到通过CNN层的过滤可以识别出后门攻击。基于这一发现，我们使用非监督异常检测来评估预处理后的过滤器，并计算每个客户端的异常评分。然后，我们根据他们的异常得分来识别最可疑的客户。大量实验表明，ARIBA方法在不降低模型性能的前提下，能够有效、稳健地防御多种最先进的攻击。



## **36. Adversarial Detection without Model Information**

无模型信息的对抗性检测 cs.CV

**SubmitDate**: 2022-02-09    [paper-pdf](http://arxiv.org/pdf/2202.04271v1)

**Authors**: Abhishek Moitra, Youngeun Kim, Priyadarshini Panda

**Abstracts**: Most prior state-of-the-art adversarial detection works assume that the underlying vulnerable model is accessible, i,e., the model can be trained or its outputs are visible. However, this is not a practical assumption due to factors like model encryption, model information leakage and so on. In this work, we propose a model independent adversarial detection method using a simple energy function to distinguish between adversarial and natural inputs. We train a standalone detector independent of the underlying model, with sequential layer-wise training to increase the energy separation corresponding to natural and adversarial inputs. With this, we perform energy distribution-based adversarial detection. Our method achieves state-of-the-art detection performance (ROC-AUC > 0.9) across a wide range of gradient, score and decision-based adversarial attacks on CIFAR10, CIFAR100 and TinyImagenet datasets. Compared to prior approaches, our method requires ~10-100x less number of operations and parameters for adversarial detection. Further, we show that our detection method is transferable across different datasets and adversarial attacks. For reproducibility, we provide code in the supplementary material.

摘要: 大多数现有的对抗性检测工作都假设潜在的易受攻击模型是可访问的，即模型可以被训练或其输出是可见的。然而，由于模型加密、模型信息泄露等因素，这并不是一个实际的假设。在这项工作中，我们提出了一种模型无关的敌意检测方法，使用一个简单的能量函数来区分敌意输入和自然输入。我们训练一个独立的检测器，独立于底层模型，通过顺序的分层训练来增加与自然和对抗性输入相对应的能量分离。在此基础上，我们进行了基于能量分布的敌意检测。我们的方法在CIFAR10、CIFAR100和TinyImagenet数据集上获得了最先进的检测性能(ROC-AUC>0.9)，涵盖了广泛的梯度、得分和基于决策的对手攻击。与以前的方法相比，我们的方法需要的操作次数和参数减少了约10-100倍。此外，我们还证明了我们的检测方法可以在不同的数据集和敌意攻击之间传输。为了重现性，我们在补充材料中提供了代码。



## **37. Towards Compositional Adversarial Robustness: Generalizing Adversarial Training to Composite Semantic Perturbations**

走向成分对抗稳健性：将对抗训练推广到复合语义扰动 cs.CV

**SubmitDate**: 2022-02-09    [paper-pdf](http://arxiv.org/pdf/2202.04235v1)

**Authors**: Yun-Yun Tsai, Lei Hsiung, Pin-Yu Chen, Tsung-Yi Ho

**Abstracts**: Model robustness against adversarial examples of single perturbation type such as the $\ell_{p}$-norm has been widely studied, yet its generalization to more realistic scenarios involving multiple semantic perturbations and their composition remains largely unexplored. In this paper, we firstly propose a novel method for generating composite adversarial examples. By utilizing component-wise projected gradient descent and automatic attack-order scheduling, our method can find the optimal attack composition. We then propose \textbf{generalized adversarial training} (\textbf{GAT}) to extend model robustness from $\ell_{p}$-norm to composite semantic perturbations, such as the combination of Hue, Saturation, Brightness, Contrast, and Rotation. The results on ImageNet and CIFAR-10 datasets show that GAT can be robust not only to any single attack but also to any combination of multiple attacks. GAT also outperforms baseline $\ell_{\infty}$-norm bounded adversarial training approaches by a significant margin.

摘要: 针对单一扰动类型的对抗性实例(如$ellp-范数)的模型鲁棒性已被广泛研究，但其对涉及多个语义扰动及其组成的更现实场景的推广仍未得到很大程度的探索。在本文中，我们首先提出了一种生成复合对抗性实例的新方法。该方法通过基于组件的投影梯度下降和攻击顺序的自动调度，能够找到最优的攻击组合。然后，我们提出了textbf(广义对抗性训练)(textbf{GAT})来将模型鲁棒性从$ellp}$范数扩展到复合语义扰动，如色调、饱和度、亮度、对比度和旋转的组合。在ImageNet和CIFAR-10数据集上的结果表明，GAT不仅对任何单一攻击都具有鲁棒性，而且对多个攻击的任何组合都具有鲁棒性。GAT的性能也大大超过了基线$-范数有界的对抗性训练方法。



## **38. Defeating Misclassification Attacks Against Transfer Learning**

抵抗针对迁移学习的误分类攻击 cs.LG

This paper has been published in IEEE Transactions on Dependable and  Secure Computing.  https://doi.ieeecomputersociety.org/10.1109/TDSC.2022.3144988

**SubmitDate**: 2022-02-09    [paper-pdf](http://arxiv.org/pdf/1908.11230v4)

**Authors**: Bang Wu, Shuo Wang, Xingliang Yuan, Cong Wang, Carsten Rudolph, Xiangwen Yang

**Abstracts**: Transfer learning is prevalent as a technique to efficiently generate new models (Student models) based on the knowledge transferred from a pre-trained model (Teacher model). However, Teacher models are often publicly available for sharing and reuse, which inevitably introduces vulnerability to trigger severe attacks against transfer learning systems. In this paper, we take a first step towards mitigating one of the most advanced misclassification attacks in transfer learning. We design a distilled differentiator via activation-based network pruning to enervate the attack transferability while retaining accuracy. We adopt an ensemble structure from variant differentiators to improve the defence robustness. To avoid the bloated ensemble size during inference, we propose a two-phase defence, in which inference from the Student model is firstly performed to narrow down the candidate differentiators to be assembled, and later only a small, fixed number of them can be chosen to validate clean or reject adversarial inputs effectively. Our comprehensive evaluations on both large and small image recognition tasks confirm that the Student models with our defence of only 5 differentiators are immune to over 90% of the adversarial inputs with an accuracy loss of less than 10%. Our comparison also demonstrates that our design outperforms prior problematic defences.

摘要: 迁移学习作为一种基于从预先训练的模型(教师模型)传输的知识有效地生成新模型(学生模型)的技术而流行。然而，教师模型通常是公开可供共享和重用的，这不可避免地会引入漏洞，从而引发对迁移学习系统的严重攻击。在本文中，我们向减轻迁移学习中最高级的错误分类攻击之一迈出了第一步。我们通过基于激活的网络剪枝设计了一种提炼的微分器，在保持准确性的同时削弱了攻击的可传递性。我们采用不同微分器的集成结构来提高防御鲁棒性。为了避免推理过程中集成规模的膨胀，我们提出了一种两阶段防御方法，首先从学生模型中进行推理，以缩小待组装的候选微分算子的范围，然后只能选择少量固定数量的微分算子来有效地验证干净或拒绝对手输入。我们在大小图像识别任务上的综合评估证实，仅有5个微分因子的学生模型对90%以上的敌意输入具有免疫力，准确率损失小于10%。我们的比较还表明，我们的设计比以前的有问题的防御性能更好。



## **39. Ontology-based Attack Graph Enrichment**

基于本体的攻击图充实 cs.CR

18 pages, 3 figures, 1 table, conference paper (TIEMS Annual  Conference, December 2021, Paris, France)

**SubmitDate**: 2022-02-08    [paper-pdf](http://arxiv.org/pdf/2202.04016v1)

**Authors**: Kéren Saint-Hilaire, Frédéric Cuppens, Nora Cuppens, Joaquin Garcia-Alfaro

**Abstracts**: Attack graphs provide a representation of possible actions that adversaries can perpetrate to attack a system. They are used by cybersecurity experts to make decisions, e.g., to decide remediation and recovery plans. Different approaches can be used to build such graphs. We focus on logical attack graphs, based on predicate logic, to define the causality of adversarial actions. Since networks and vulnerabilities are constantly changing (e.g., new applications get installed on system devices, updated services get publicly exposed, etc.), we propose to enrich the attack graph generation approach with a semantic augmentation post-processing of the predicates. Graphs are now mapped to monitoring alerts confirming successful attack actions and updated according to network and vulnerability changes. As a result, predicates get periodically updated, based on attack evidences and ontology enrichment. This allows to verify whether changes lead the attacker to the initial goals or to cause further damage to the system not anticipated in the initial graphs. We illustrate the approach under the specific domain of cyber-physical security affecting smart cities. We validate the approach using existing tools and ontologies.

摘要: 攻击图提供了攻击者可能实施的攻击系统的操作的表示。网络安全专家使用它们来做出决策，例如，决定补救和恢复计划。可以使用不同的方法来构建这样的图。我们将重点放在逻辑攻击图上，基于谓词逻辑来定义敌对行为的因果关系。由于网络和漏洞是不断变化的(例如，新的应用程序安装在系统设备上，更新的服务被公开暴露等)，我们建议通过谓词的语义增强后处理来丰富攻击图生成方法。图形现在映射到监控警报，以确认攻击操作成功，并根据网络和漏洞更改进行更新。因此，谓词会根据攻击证据和本体丰富进行定期更新。这样可以验证更改是否会导致攻击者达到初始目标，或者是否会对系统造成初始图表中未预料到的进一步损坏。我们从影响智慧城市的网络物理安全这一特定领域出发，阐述了该方法。我们使用现有的工具和本体来验证该方法。



## **40. Verification-Aided Deep Ensemble Selection**

辅助验证的深度集成选择 cs.LG

**SubmitDate**: 2022-02-08    [paper-pdf](http://arxiv.org/pdf/2202.03898v1)

**Authors**: Guy Amir, Guy Katz, Michael Schapira

**Abstracts**: Deep neural networks (DNNs) have become the technology of choice for realizing a variety of complex tasks. However, as highlighted by many recent studies, even an imperceptible perturbation to a correctly classified input can lead to misclassification by a DNN. This renders DNNs vulnerable to strategic input manipulations by attackers, and also prone to oversensitivity to environmental noise.   To mitigate this phenomenon, practitioners apply joint classification by an ensemble of DNNs. By aggregating the classification outputs of different individual DNNs for the same input, ensemble-based classification reduces the risk of misclassifications due to the specific realization of the stochastic training process of any single DNN. However, the effectiveness of a DNN ensemble is highly dependent on its members not simultaneously erring on many different inputs.   In this case study, we harness recent advances in DNN verification to devise a methodology for identifying ensemble compositions that are less prone to simultaneous errors, even when the input is adversarially perturbed -- resulting in more robustly-accurate ensemble-based classification.   Our proposed framework uses a DNN verifier as a backend, and includes heuristics that help reduce the high complexity of directly verifying ensembles. More broadly, our work puts forth a novel universal objective for formal verification that can potentially improve the robustness of real-world, deep-learning-based systems across a variety of application domains.

摘要: 深度神经网络(DNNs)已经成为实现各种复杂任务的首选技术。然而，正如最近的许多研究所强调的那样，即使是对正确分类的输入进行了不可察觉的扰动，也可能导致DNN的错误分类。这使得DNN容易受到攻击者的战略性输入操纵，并且容易对环境噪声过于敏感。为了缓解这一现象，从业者应用DNN集合的联合分类。通过聚合同一输入的不同个体DNN的分类输出，基于集成的分类降低了由于任意单个DNN的随机训练过程的具体实现而导致的误分类风险。然而，DNN合奏的有效性高度依赖于其成员，而不是同时在许多不同的输入上出错。在这个案例研究中，我们利用DNN验证方面的最新进展来设计一种方法，用于识别不太容易同时出错的组合成分，即使输入受到相反的干扰-导致基于组合的分类更加稳健准确。我们提出的框架使用DNN验证器作为后端，并包括有助于降低直接验证集成的高复杂度的启发式算法。更广泛地说，我们的工作为形式验证提出了一个新的通用目标，可以潜在地提高现实世界中基于深度学习的系统在各种应用领域的健壮性。



## **41. Invertible Tabular GANs: Killing Two Birds with OneStone for Tabular Data Synthesis**

可逆表格甘斯：表格数据合成的一石二鸟 cs.LG

19 pages

**SubmitDate**: 2022-02-08    [paper-pdf](http://arxiv.org/pdf/2202.03636v1)

**Authors**: Jaehoon Lee, Jihyeon Hyeong, Jinsung Jeon, Noseong Park, Jihoon Cho

**Abstracts**: Tabular data synthesis has received wide attention in the literature. This is because available data is often limited, incomplete, or cannot be obtained easily, and data privacy is becoming increasingly important. In this work, we present a generalized GAN framework for tabular synthesis, which combines the adversarial training of GANs and the negative log-density regularization of invertible neural networks. The proposed framework can be used for two distinctive objectives. First, we can further improve the synthesis quality, by decreasing the negative log-density of real records in the process of adversarial training. On the other hand, by increasing the negative log-density of real records, realistic fake records can be synthesized in a way that they are not too much close to real records and reduce the chance of potential information leakage. We conduct experiments with real-world datasets for classification, regression, and privacy attacks. In general, the proposed method demonstrates the best synthesis quality (in terms of task-oriented evaluation metrics, e.g., F1) when decreasing the negative log-density during the adversarial training. If increasing the negative log-density, our experimental results show that the distance between real and fake records increases, enhancing robustness against privacy attacks.

摘要: 表格数据合成在文献中受到了广泛的关注。这是因为可用的数据通常是有限的、不完整的或不容易获得的，而且数据隐私变得越来越重要。在这项工作中，我们提出了一个用于表格综合的广义GAN框架，该框架结合了GANS的对抗性训练和可逆神经网络的负对数密度正则化。建议的框架可用于两个不同的目标。首先，通过降低对抗性训练过程中真实记录的负对数密度，进一步提高综合质量。另一方面，通过增加真实记录的负对数密度，可以合成出与真实记录不太接近的真实假记录，减少潜在信息泄露的机会。我们使用真实世界的数据集进行分类、回归和隐私攻击的实验。一般而言，在对抗性训练过程中，当降低负对数密度时，该方法表现出最佳的综合质量(就面向任务的评估指标而言，例如F1)。实验结果表明，如果增加负的日志密度，真实记录和虚假记录之间的距离会增大，从而增强了对隐私攻击的鲁棒性。



## **42. A Survey on Poisoning Attacks Against Supervised Machine Learning**

针对有监督机器学习的中毒攻击研究综述 cs.CR

**SubmitDate**: 2022-02-08    [paper-pdf](http://arxiv.org/pdf/2202.02510v2)

**Authors**: Wenjun Qiu

**Abstracts**: With the rise of artificial intelligence and machine learning in modern computing, one of the major concerns regarding such techniques is to provide privacy and security against adversaries. We present this survey paper to cover the most representative papers in poisoning attacks against supervised machine learning models. We first provide a taxonomy to categorize existing studies and then present detailed summaries for selected papers. We summarize and compare the methodology and limitations of existing literature. We conclude this paper with potential improvements and future directions to further exploit and prevent poisoning attacks on supervised models. We propose several unanswered research questions to encourage and inspire researchers for future work.

摘要: 随着人工智能和机器学习在现代计算中的兴起，关于这类技术的一个主要问题是提供隐私和安全，以抵御对手。我们提出的这份调查报告涵盖了针对有监督机器学习模型的中毒攻击中最具代表性的论文。我们首先提供一个分类法来对现有的研究进行分类，然后对选定的论文进行详细的总结。我们对现有文献的研究方法和局限性进行了总结和比较。最后，我们对本文进行了总结，并对进一步开发和防止对监督模型的中毒攻击提出了可能的改进和未来的方向。我们提出了几个尚未回答的研究问题，以鼓励和激励研究人员未来的工作。



## **43. Sparse-RS: a versatile framework for query-efficient sparse black-box adversarial attacks**

Sparse-RS：一种通用的查询高效稀疏黑盒攻击框架 cs.LG

Accepted at AAAI 2022. This version contains considerably extended  results in the L0 threat model

**SubmitDate**: 2022-02-08    [paper-pdf](http://arxiv.org/pdf/2006.12834v3)

**Authors**: Francesco Croce, Maksym Andriushchenko, Naman D. Singh, Nicolas Flammarion, Matthias Hein

**Abstracts**: We propose a versatile framework based on random search, Sparse-RS, for score-based sparse targeted and untargeted attacks in the black-box setting. Sparse-RS does not rely on substitute models and achieves state-of-the-art success rate and query efficiency for multiple sparse attack models: $l_0$-bounded perturbations, adversarial patches, and adversarial frames. The $l_0$-version of untargeted Sparse-RS outperforms all black-box and even all white-box attacks for different models on MNIST, CIFAR-10, and ImageNet. Moreover, our untargeted Sparse-RS achieves very high success rates even for the challenging settings of $20\times20$ adversarial patches and $2$-pixel wide adversarial frames for $224\times224$ images. Finally, we show that Sparse-RS can be applied to generate targeted universal adversarial patches where it significantly outperforms the existing approaches. The code of our framework is available at https://github.com/fra31/sparse-rs.

摘要: 针对黑盒环境下基于分数的稀疏目标攻击和非目标攻击，我们提出了一个基于随机搜索的通用框架Sparse-RS。Sparse-RS不依赖替身模型，对多种稀疏攻击模型($l_0$有界扰动、敌意补丁和敌意帧)具有最高的成功率和查询效率。对于MNIST、CIFAR-10和ImageNet上的不同型号，$l_0$-版本的非目标稀疏-RS的性能优于所有黑盒甚至所有白盒攻击。此外，我们的非定向稀疏-RS即使在$20\x 20$对抗性补丁和$2$像素宽的对抗性帧的$224\x 224$图像的挑战性设置下也能获得非常高的成功率。最后，我们证明了稀疏-RS可以用来生成目标通用的对抗性补丁，其性能明显优于现有的方法。我们框架的代码可以在https://github.com/fra31/sparse-rs.上找到



## **44. Evaluating Robustness of Cooperative MARL: A Model-based Approach**

评估协作MAIL的健壮性：一种基于模型的方法 cs.LG

**SubmitDate**: 2022-02-07    [paper-pdf](http://arxiv.org/pdf/2202.03558v1)

**Authors**: Nhan H. Pham, Lam M. Nguyen, Jie Chen, Hoang Thanh Lam, Subhro Das, Tsui-Wei Weng

**Abstracts**: In recent years, a proliferation of methods were developed for cooperative multi-agent reinforcement learning (c-MARL). However, the robustness of c-MARL agents against adversarial attacks has been rarely explored. In this paper, we propose to evaluate the robustness of c-MARL agents via a model-based approach. Our proposed formulation can craft stronger adversarial state perturbations of c-MARL agents(s) to lower total team rewards more than existing model-free approaches. In addition, we propose the first victim-agent selection strategy which allows us to develop even stronger adversarial attack. Numerical experiments on multi-agent MuJoCo benchmarks illustrate the advantage of our approach over other baselines. The proposed model-based attack consistently outperforms other baselines in all tested environments.

摘要: 近年来，协作多智能体强化学习(c-MARL)方法层出不穷。然而，c-Marl代理抵抗敌意攻击的健壮性很少被研究。在本文中，我们提出了一种基于模型的方法来评估c-Marl代理的健壮性。与现有的无模型方法相比，我们提出的公式可以制作更强的c-Marl代理的对抗性状态扰动，以降低团队总奖励。此外，我们还提出了第一种受害者-代理选择策略，使我们能够开发出更强的对抗性攻击。在多智能体MuJoCo基准上的数值实验表明了我们的方法相对于其他基线的优势。在所有测试环境中，建议的基于模型的攻击始终优于其他基准。



## **45. Blind leads Blind: A Zero-Knowledge Attack on Federated Learning**

盲目引导：对联邦学习的零知识攻击 cs.CR

**SubmitDate**: 2022-02-07    [paper-pdf](http://arxiv.org/pdf/2202.05877v1)

**Authors**: Jiyue Huang, Zilong Zhao, Lydia Y. Chen, Stefanie Roos

**Abstracts**: Attacks on Federated Learning (FL) can severely reduce the quality of the generated models and limit the usefulness of this emerging learning paradigm that enables on-premise decentralized learning. There have been various untargeted attacks on FL, but they are not widely applicable as they i) assume that the attacker knows every update of benign clients, which is indeed sent in encrypted form to the central server, or ii) assume that the attacker has a large dataset and sufficient resources to locally train updates imitating benign parties. In this paper, we design a zero-knowledge untargeted attack (ZKA), which synthesizes malicious data to craft adversarial models without eavesdropping on the transmission of benign clients at all or requiring a large quantity of task-specific training data. To inject malicious input into the FL system by synthetic data, ZKA has two variants. ZKA-R generates adversarial ambiguous data by reversing engineering from the global models. To enable stealthiness, ZKA-G trains the local model on synthetic data from the generator that aims to synthesize images different from a randomly chosen class. Furthermore, we add a novel distance-based regularization term for both attacks to further enhance stealthiness. Experimental results on Fashion-MNIST and CIFAR-10 show that the ZKA achieves similar or even higher attack success rate than the state-of-the-art untargeted attacks against various defense mechanisms, namely more than 50% for Cifar-10 for all considered defense mechanisms. As expected, ZKA-G is better at circumventing defenses, even showing a defense pass rate of close to 90% when ZKA-R only achieves 70%. Higher data heterogeneity favours ZKA-R since detection becomes harder.

摘要: 对联合学习(FL)的攻击可能会严重降低生成模型的质量，并限制这种支持内部分散学习的新兴学习范例的有用性。已经存在针对FL的各种非目标攻击，但是它们不是广泛适用的，因为它们i)假设攻击者知道良性客户端的每个更新，其确实以加密的形式发送到中央服务器，或者ii)假设攻击者具有大型数据集和足够的资源来本地训练模仿良性方的更新。在本文中，我们设计了一种零知识无目标攻击(ZKA)，它合成恶意数据来构建敌意模型，而根本不窃听良性客户端的传输，也不需要大量特定于任务的训练数据。为了通过合成数据将恶意输入注入FL系统，ZKA有两个变体。ZKA-R通过从全局模型中逆向工程来生成对抗性模糊数据。为了实现隐蔽性，ZKA-G根据来自生成器的合成数据训练本地模型，该生成器旨在合成与随机选择的类别不同的图像。此外，我们为这两种攻击添加了一种新的基于距离的正则化项，以进一步增强隐蔽性。在Fashion-MNIST和CIFAR-10上的实验结果表明，ZKA对各种防御机制的攻击成功率接近甚至更高，即对于所有考虑的防御机制，对CIFAR-10的攻击成功率都在50%以上。不出所料，ZKA-G更善于绕过防守，甚至在ZKA-R只达到70%的情况下，防守通过率也接近90%。较高的数据异构性有利于ZKA-R，因为检测变得更加困难。



## **46. Deletion Inference, Reconstruction, and Compliance in Machine (Un)Learning**

机器(UN)学习中的删除推理、重构和顺应性 cs.LG

Full version of a paper appearing in the 22nd Privacy Enhancing  Technologies Symposium (PETS 2022)

**SubmitDate**: 2022-02-07    [paper-pdf](http://arxiv.org/pdf/2202.03460v1)

**Authors**: Ji Gao, Sanjam Garg, Mohammad Mahmoody, Prashant Nalini Vasudevan

**Abstracts**: Privacy attacks on machine learning models aim to identify the data that is used to train such models. Such attacks, traditionally, are studied on static models that are trained once and are accessible by the adversary. Motivated to meet new legal requirements, many machine learning methods are recently extended to support machine unlearning, i.e., updating models as if certain examples are removed from their training sets, and meet new legal requirements. However, privacy attacks could potentially become more devastating in this new setting, since an attacker could now access both the original model before deletion and the new model after the deletion. In fact, the very act of deletion might make the deleted record more vulnerable to privacy attacks.   Inspired by cryptographic definitions and the differential privacy framework, we formally study privacy implications of machine unlearning. We formalize (various forms of) deletion inference and deletion reconstruction attacks, in which the adversary aims to either identify which record is deleted or to reconstruct (perhaps part of) the deleted records. We then present successful deletion inference and reconstruction attacks for a variety of machine learning models and tasks such as classification, regression, and language models. Finally, we show that our attacks would provably be precluded if the schemes satisfy (variants of) Deletion Compliance (Garg, Goldwasser, and Vasudevan, Eurocrypt' 20).

摘要: 针对机器学习模型的隐私攻击旨在识别用于训练此类模型的数据。传统上，这类攻击是在静电模型上研究的，这些模型只训练一次，对手可以访问。为了满足新的法律要求，许多机器学习方法最近被扩展到支持机器遗忘，即更新模型，就像从训练集中删除某些示例一样，并满足新的法律要求。然而，在这种新设置下，隐私攻击可能会变得更具破坏性，因为攻击者现在既可以访问删除前的原始模型，也可以访问删除后的新模型。事实上，删除操作本身可能会使删除的记录更容易受到隐私攻击。在密码学定义和差分隐私框架的启发下，我们正式研究了机器遗忘的隐私含义。我们形式化(各种形式的)删除推理和删除重构攻击，在这些攻击中，敌手的目标要么是识别哪条记录被删除，要么是重构(可能是一部分)被删除的记录。然后，我们针对各种机器学习模型和任务，如分类、回归和语言模型，提出了成功的删除推理和重构攻击。最后，我们证明了如果方案满足删除遵从性(Garg，Goldwasser和Vasudevan，Eurocrypt‘20)(变体)，我们的攻击将被证明是被排除的。



## **47. Bilevel Optimization with a Lower-level Contraction: Optimal Sample Complexity without Warm-Start**

低水平收缩的双层优化：无热启动的最优样本复杂度 stat.ML

30 pages

**SubmitDate**: 2022-02-07    [paper-pdf](http://arxiv.org/pdf/2202.03397v1)

**Authors**: Riccardo Grazzi, Massimiliano Pontil, Saverio Salzo

**Abstracts**: We analyze a general class of bilevel problems, in which the upper-level problem consists in the minimization of a smooth objective function and the lower-level problem is to find the fixed point of a smooth contraction map. This type of problems include instances of meta-learning, hyperparameter optimization and data poisoning adversarial attacks. Several recent works have proposed algorithms which warm-start the lower-level problem, i.e. they use the previous lower-level approximate solution as a staring point for the lower-level solver. This warm-start procedure allows one to improve the sample complexity in both the stochastic and deterministic settings, achieving in some cases the order-wise optimal sample complexity. We show that without warm-start, it is still possible to achieve order-wise optimal and near-optimal sample complexity for the stochastic and deterministic settings, respectively. In particular, we propose a simple method which uses stochastic fixed point iterations at the lower-level and projected inexact gradient descent at the upper-level, that reaches an $\epsilon$-stationary point using $O(\epsilon^{-2})$ and $\tilde{O}(\epsilon^{-1})$ samples for the stochastic and the deterministic setting, respectively. Compared to methods using warm-start, ours is better suited for meta-learning and yields a simpler analysis that does not need to study the coupled interactions between the upper-level and lower-level iterates.

摘要: 我们分析了一类一般的两层问题，其中上层问题在于光滑目标函数的最小化，下层问题是寻找光滑压缩映射的不动点。这类问题包括元学习、超参数优化和数据中毒攻击等实例。最近的一些工作已经提出了暖启动下层问题的算法，即使用先前的下层近似解作为下层求解器的起始点。这种热启动过程允许人们在随机和确定性设置下改善样本复杂度，在某些情况下实现顺序最优的样本复杂度。我们证明了在没有热启动的情况下，对于随机设置和确定性设置，仍然可以分别获得按顺序最优和接近最优的样本复杂度。特别地，我们提出了一种简单的方法，它在下层使用随机不动点迭代，在上层使用投影的不精确梯度下降，在随机和确定性设置下分别使用$O(epsilon^{-2})$和$tilde{O}(epsilon^{-1})$样本达到$ε-稳定点。与使用热启动的方法相比，我们的方法更适合于元学习，并且产生了更简单的分析，不需要研究上层和下层迭代之间的耦合作用。



## **48. Membership Inference Attacks and Defenses in Neural Network Pruning**

神经网络修剪中的隶属度推理攻击与防御 cs.CR

This paper has been conditionally accepted to USENIX Security  Symposium 2022. This is an extended version

**SubmitDate**: 2022-02-07    [paper-pdf](http://arxiv.org/pdf/2202.03335v1)

**Authors**: Xiaoyong Yuan, Lan Zhang

**Abstracts**: Neural network pruning has been an essential technique to reduce the computation and memory requirements for using deep neural networks for resource-constrained devices. Most existing research focuses primarily on balancing the sparsity and accuracy of a pruned neural network by strategically removing insignificant parameters and retraining the pruned model. Such efforts on reusing training samples pose serious privacy risks due to increased memorization, which, however, has not been investigated yet.   In this paper, we conduct the first analysis of privacy risks in neural network pruning. Specifically, we investigate the impacts of neural network pruning on training data privacy, i.e., membership inference attacks. We first explore the impact of neural network pruning on prediction divergence, where the pruning process disproportionately affects the pruned model's behavior for members and non-members. Meanwhile, the influence of divergence even varies among different classes in a fine-grained manner. Enlighten by such divergence, we proposed a self-attention membership inference attack against the pruned neural networks. Extensive experiments are conducted to rigorously evaluate the privacy impacts of different pruning approaches, sparsity levels, and adversary knowledge. The proposed attack shows the higher attack performance on the pruned models when compared with eight existing membership inference attacks. In addition, we propose a new defense mechanism to protect the pruning process by mitigating the prediction divergence based on KL-divergence distance, whose effectiveness has been experimentally demonstrated to effectively mitigate the privacy risks while maintaining the sparsity and accuracy of the pruned models.

摘要: 对于资源受限的设备，为了减少对深层神经网络的计算和存储需求，神经网络修剪已经成为一项重要的技术。现有的大多数研究主要集中在通过策略性地去除无关紧要的参数和重新训练修剪后的模型来平衡修剪神经网络的稀疏性和准确性。这种重复使用训练样本的努力由于增加了记忆而带来了严重的隐私风险，然而，这一点尚未得到调查。本文首先对神经网络修剪中的隐私风险进行了分析。具体地说，我们研究了神经网络修剪对训练数据隐私的影响，即成员推理攻击。我们首先探讨了神经网络修剪对预测发散的影响，其中修剪过程不成比例地影响修剪后的模型对成员和非成员的行为。同时，分歧的影响甚至在不同的阶层之间也有细微的差异。受这种分歧的启发，我们提出了一种针对修剪后的神经网络的自注意成员推理攻击。广泛的实验被用来严格评估不同的修剪方法、稀疏程度和敌意知识对隐私的影响。与现有的8种成员推理攻击相比，该攻击在剪枝模型上表现出更高的攻击性能。此外，我们还提出了一种新的防御机制来保护剪枝过程，通过减少基于KL-发散距离的预测发散来保护剪枝过程，实验证明该机制在保持剪枝模型的稀疏性和准确性的同时，有效地缓解了隐私风险。



## **49. On The Empirical Effectiveness of Unrealistic Adversarial Hardening Against Realistic Adversarial Attacks**

论非现实对抗硬化对抗现实对抗攻击的经验有效性 cs.LG

**SubmitDate**: 2022-02-07    [paper-pdf](http://arxiv.org/pdf/2202.03277v1)

**Authors**: Salijona Dyrmishi, Salah Ghamizi, Thibault Simonetto, Yves Le Traon, Maxime Cordy

**Abstracts**: While the literature on security attacks and defense of Machine Learning (ML) systems mostly focuses on unrealistic adversarial examples, recent research has raised concern about the under-explored field of realistic adversarial attacks and their implications on the robustness of real-world systems. Our paper paves the way for a better understanding of adversarial robustness against realistic attacks and makes two major contributions. First, we conduct a study on three real-world use cases (text classification, botnet detection, malware detection)) and five datasets in order to evaluate whether unrealistic adversarial examples can be used to protect models against realistic examples. Our results reveal discrepancies across the use cases, where unrealistic examples can either be as effective as the realistic ones or may offer only limited improvement. Second, to explain these results, we analyze the latent representation of the adversarial examples generated with realistic and unrealistic attacks. We shed light on the patterns that discriminate which unrealistic examples can be used for effective hardening. We release our code, datasets and models to support future research in exploring how to reduce the gap between unrealistic and realistic adversarial attacks.

摘要: 虽然关于机器学习系统安全攻击和防御的文献大多集中在不现实的对抗性例子上，但最近的研究已经引起了对现实对抗性攻击这一未被充分探索的领域及其对现实世界系统健壮性的影响的关注。我们的论文为更好地理解对抗现实攻击的鲁棒性铺平了道路，并做出了两个主要贡献。首先，我们在三个真实世界的用例(文本分类、僵尸网络检测、恶意软件检测)和五个数据集上进行了研究，以评估不现实的对抗性示例是否可以用来保护模型免受现实示例的影响。我们的结果揭示了用例之间的差异，在这些用例中，不切实际的示例可能与现实的示例一样有效，或者可能只提供有限的改进。其次，为了解释这些结果，我们分析了现实攻击和非现实攻击产生的对抗性例子的潜在表征。我们阐明了区分哪些不切实际的例子可以用于有效强化的模式。我们发布了我们的代码、数据集和模型，以支持未来的研究，探索如何缩小不切实际和现实的对抗性攻击之间的差距。



## **50. Strong Converse Theorem for Source Encryption under Side-Channel Attacks**

旁路攻击下源加密的强逆定理 cs.IT

9 pages, 6 figures. The short version of this paper was submitted to  ISIT2022, arXiv admin note: text overlap with arXiv:1801.02563

**SubmitDate**: 2022-02-07    [paper-pdf](http://arxiv.org/pdf/2201.11670v3)

**Authors**: Yasutada Oohama, Bagus Santoso

**Abstracts**: We are interested in investigating the security of source encryption with a symmetric key under side-channel attacks. In this paper, we propose a general framework of source encryption with a symmetric key under the side-channel attacks, which applies to \emph{any} source encryption with a symmetric key and \emph{any} kind of side-channel attacks targeting the secret key. We also propose a new security criterion for strong secrecy under side-channel attacks, which is a natural extension of mutual information, i.e., \emph{the maximum conditional mutual information between the plaintext and the ciphertext given the adversarial key leakage, where the maximum is taken over all possible plaintext distribution}. Under this new criterion, we successfully formulate the rate region, which serves as both necessary and sufficient conditions to have secure transmission even under side-channel attacks. Furthermore, we also prove another theoretical result on our new security criterion, which might be interesting in its own right: in the case of the discrete memoryless source, no perfect secrecy under side-channel attacks in the standard security criterion, i.e., the ordinary mutual information, is achievable without achieving perfect secrecy in this new security criterion, although our new security criterion is more strict than the standard security criterion.

摘要: 我们感兴趣的是研究在旁信道攻击下使用对称密钥的源加密的安全性。本文提出了一种在旁路攻击下对称密钥源加密的通用框架，该框架适用于对称密钥源加密和针对密钥的旁路攻击。我们还提出了一种新的边信道攻击下强保密性的安全准则，它是互信息的自然扩展，即在密钥泄露的情况下明文和密文之间的最大条件互信息，其中最大值取在所有可能的明文分布上。在这一新准则下，我们成功地给出了码率域的表达式，该码率域既是安全传输的充要条件，又是旁路攻击下安全传输的充分必要条件。此外，我们还证明了关于我们的新安全准则的另一个理论结果，它本身可能很有趣：在离散无记忆信源的情况下，尽管我们的新安全准则比标准安全准则更严格，但是在标准安全准则(即普通互信息)下，如果不达到完全保密，就不可能实现标准安全准则中的旁路攻击下的完全保密性。



