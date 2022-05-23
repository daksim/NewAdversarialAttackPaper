# Latest Adversarial Attack Papers
**update at 2022-05-23 10:07:07**

[中英双语版本](https://github.com/daksim/NewAdversarialAttackPaper/blob/main/README_CN.md)

## **1. Vulnerability Analysis and Performance Enhancement of Authentication Protocol in Dynamic Wireless Power Transfer Systems**

cs.CR

16 pages, conference

**SubmitDate**: 2022-05-20    [paper-pdf](http://arxiv.org/pdf/2205.10292v1)

**Authors**: Tommaso Bianchi, Surudhi Asokraj, Alessandro Brighente, Mauro Conti, Radha Poovendran

**Abstracts**: Recent advancements in wireless charging technology, as well as the possibility of utilizing it in the Electric Vehicle (EV) domain for dynamic charging solutions, have fueled the demand for a secure and usable protocol in the Dynamic Wireless Power Transfer (DWPT) technology. The DWPT must operate in the presence of malicious adversaries that can undermine the charging process and harm the customer service quality, while preserving the privacy of the users. Recently, it was shown that the DWPT system is susceptible to adversarial attacks, including replay, denial-of-service and free-riding attacks, which can lead to the adversary blocking the authorized user from charging, enabling free charging for free riders and exploiting the location privacy of the customers. In this paper, we study the current State-Of-The-Art (SOTA) authentication protocols and make the following two contributions: a) we show that the SOTA is vulnerable to the tracking of the user activity and b) we propose an enhanced authentication protocol that eliminates the vulnerability while providing improved efficiency compared to the SOTA authentication protocols. By adopting authentication messages based only on exclusive OR operations, hashing, and hash chains, we optimize the protocol to achieve a complexity that varies linearly with the number of charging pads, providing improved scalability. Compared to SOTA, the proposed scheme has a performance gain in the computational cost of around 90% on average for each pad.



## **2. Adversarial Body Shape Search for Legged Robots**

cs.RO

6 pages, 7 figures

**SubmitDate**: 2022-05-20    [paper-pdf](http://arxiv.org/pdf/2205.10187v1)

**Authors**: Takaaki Azakami, Hiroshi Kera, Kazuhiko Kawamoto

**Abstracts**: We propose an evolutionary computation method for an adversarial attack on the length and thickness of parts of legged robots by deep reinforcement learning. This attack changes the robot body shape and interferes with walking-we call the attacked body as adversarial body shape. The evolutionary computation method searches adversarial body shape by minimizing the expected cumulative reward earned through walking simulation. To evaluate the effectiveness of the proposed method, we perform experiments with three-legged robots, Walker2d, Ant-v2, and Humanoid-v2 in OpenAI Gym. The experimental results reveal that Walker2d and Ant-v2 are more vulnerable to the attack on the length than the thickness of the body parts, whereas Humanoid-v2 is vulnerable to the attack on both of the length and thickness. We further identify that the adversarial body shapes break left-right symmetry or shift the center of gravity of the legged robots. Finding adversarial body shape can be used to proactively diagnose the vulnerability of legged robot walking.



## **3. Getting a-Round Guarantees: Floating-Point Attacks on Certified Robustness**

cs.CR

**SubmitDate**: 2022-05-20    [paper-pdf](http://arxiv.org/pdf/2205.10159v1)

**Authors**: Jiankai Jin, Olga Ohrimenko, Benjamin I. P. Rubinstein

**Abstracts**: Adversarial examples pose a security risk as they can alter a classifier's decision through slight perturbations to a benign input. Certified robustness has been proposed as a mitigation strategy where given an input $x$, a classifier returns a prediction and a radius with a provable guarantee that any perturbation to $x$ within this radius (e.g., under the $L_2$ norm) will not alter the classifier's prediction. In this work, we show that these guarantees can be invalidated due to limitations of floating-point representation that cause rounding errors. We design a rounding search method that can efficiently exploit this vulnerability to find adversarial examples within the certified radius. We show that the attack can be carried out against several linear classifiers that have exact certifiable guarantees and against neural network verifiers that return a certified lower bound on a robust radius. Our experiments demonstrate over 50% attack success rate on random linear classifiers, up to 35% on a breast cancer dataset for logistic regression, and a 9% attack success rate on the MNIST dataset for a neural network whose certified radius was verified by a prominent bound propagation method. We also show that state-of-the-art random smoothed classifiers for neural networks are also susceptible to adversarial examples (e.g., up to 2% attack rate on CIFAR10)-validating the importance of accounting for the error rate of robustness guarantees of such classifiers in practice. Finally, as a mitigation, we advocate the use of rounded interval arithmetic to account for rounding errors.



## **4. Generating Semantic Adversarial Examples via Feature Manipulation**

cs.LG

arXiv admin note: substantial text overlap with arXiv:1705.09064 by  other authors

**SubmitDate**: 2022-05-20    [paper-pdf](http://arxiv.org/pdf/2001.02297v2)

**Authors**: Shuo Wang, Surya Nepal, Carsten Rudolph, Marthie Grobler, Shangyu Chen, Tianle Chen

**Abstracts**: The vulnerability of deep neural networks to adversarial attacks has been widely demonstrated (e.g., adversarial example attacks). Traditional attacks perform unstructured pixel-wise perturbation to fool the classifier. An alternative approach is to have perturbations in the latent space. However, such perturbations are hard to control due to the lack of interpretability and disentanglement. In this paper, we propose a more practical adversarial attack by designing structured perturbation with semantic meanings. Our proposed technique manipulates the semantic attributes of images via the disentangled latent codes. The intuition behind our technique is that images in similar domains have some commonly shared but theme-independent semantic attributes, e.g. thickness of lines in handwritten digits, that can be bidirectionally mapped to disentangled latent codes. We generate adversarial perturbation by manipulating a single or a combination of these latent codes and propose two unsupervised semantic manipulation approaches: vector-based disentangled representation and feature map-based disentangled representation, in terms of the complexity of the latent codes and smoothness of the reconstructed images. We conduct extensive experimental evaluations on real-world image data to demonstrate the power of our attacks for black-box classifiers. We further demonstrate the existence of a universal, image-agnostic semantic adversarial example.



## **5. Adversarial joint attacks on legged robots**

cs.RO

6 pages, 8 figures

**SubmitDate**: 2022-05-20    [paper-pdf](http://arxiv.org/pdf/2205.10098v1)

**Authors**: Takuto Otomo, Hiroshi Kera, Kazuhiko Kawamoto

**Abstracts**: We address adversarial attacks on the actuators at the joints of legged robots trained by deep reinforcement learning. The vulnerability to the joint attacks can significantly impact the safety and robustness of legged robots. In this study, we demonstrate that the adversarial perturbations to the torque control signals of the actuators can significantly reduce the rewards and cause walking instability in robots. To find the adversarial torque perturbations, we develop black-box adversarial attacks, where, the adversary cannot access the neural networks trained by deep reinforcement learning. The black box attack can be applied to legged robots regardless of the architecture and algorithms of deep reinforcement learning. We employ three search methods for the black-box adversarial attacks: random search, differential evolution, and numerical gradient descent methods. In experiments with the quadruped robot Ant-v2 and the bipedal robot Humanoid-v2, in OpenAI Gym environments, we find that differential evolution can efficiently find the strongest torque perturbations among the three methods. In addition, we realize that the quadruped robot Ant-v2 is vulnerable to the adversarial perturbations, whereas the bipedal robot Humanoid-v2 is robust to the perturbations. Consequently, the joint attacks can be used for proactive diagnosis of robot walking instability.



## **6. SafeNet: Mitigating Data Poisoning Attacks on Private Machine Learning**

cs.CR

**SubmitDate**: 2022-05-20    [paper-pdf](http://arxiv.org/pdf/2205.09986v1)

**Authors**: Harsh Chaudhari, Matthew Jagielski, Alina Oprea

**Abstracts**: Secure multiparty computation (MPC) has been proposed to allow multiple mutually distrustful data owners to jointly train machine learning (ML) models on their combined data. However, the datasets used for training ML models might be under the control of an adversary mounting a data poisoning attack, and MPC prevents inspecting training sets to detect poisoning. We show that multiple MPC frameworks for private ML training are susceptible to backdoor and targeted poisoning attacks. To mitigate this, we propose SafeNet, a framework for building ensemble models in MPC with formal guarantees of robustness to data poisoning attacks. We extend the security definition of private ML training to account for poisoning and prove that our SafeNet design satisfies the definition. We demonstrate SafeNet's efficiency, accuracy, and resilience to poisoning on several machine learning datasets and models. For instance, SafeNet reduces backdoor attack success from 100% to 0% for a neural network model, while achieving 39x faster training and 36x less communication than the four-party MPC framework of Dalskov et al.



## **7. Adversarial Sample Detection for Speaker Verification by Neural Vocoders**

cs.SD

Accepted by ICASSP 2022

**SubmitDate**: 2022-05-19    [paper-pdf](http://arxiv.org/pdf/2107.00309v4)

**Authors**: Haibin Wu, Po-chun Hsu, Ji Gao, Shanshan Zhang, Shen Huang, Jian Kang, Zhiyong Wu, Helen Meng, Hung-yi Lee

**Abstracts**: Automatic speaker verification (ASV), one of the most important technology for biometric identification, has been widely adopted in security-critical applications. However, ASV is seriously vulnerable to recently emerged adversarial attacks, yet effective countermeasures against them are limited. In this paper, we adopt neural vocoders to spot adversarial samples for ASV. We use the neural vocoder to re-synthesize audio and find that the difference between the ASV scores for the original and re-synthesized audio is a good indicator for discrimination between genuine and adversarial samples. This effort is, to the best of our knowledge, among the first to pursue such a technical direction for detecting time-domain adversarial samples for ASV, and hence there is a lack of established baselines for comparison. Consequently, we implement the Griffin-Lim algorithm as the detection baseline. The proposed approach achieves effective detection performance that outperforms the baselines in all the settings. We also show that the neural vocoder adopted in the detection framework is dataset-independent. Our codes will be made open-source for future works to do fair comparison.



## **8. Focused Adversarial Attacks**

cs.LG

**SubmitDate**: 2022-05-19    [paper-pdf](http://arxiv.org/pdf/2205.09624v1)

**Authors**: Thomas Cilloni, Charles Walter, Charles Fleming

**Abstracts**: Recent advances in machine learning show that neural models are vulnerable to minimally perturbed inputs, or adversarial examples. Adversarial algorithms are optimization problems that minimize the accuracy of ML models by perturbing inputs, often using a model's loss function to craft such perturbations. State-of-the-art object detection models are characterized by very large output manifolds due to the number of possible locations and sizes of objects in an image. This leads to their outputs being sparse and optimization problems that use them incur a lot of unnecessary computation.   We propose to use a very limited subset of a model's learned manifold to compute adversarial examples. Our \textit{Focused Adversarial Attacks} (FA) algorithm identifies a small subset of sensitive regions to perform gradient-based adversarial attacks. FA is significantly faster than other gradient-based attacks when a model's manifold is sparsely activated. Also, its perturbations are more efficient than other methods under the same perturbation constraints. We evaluate FA on the COCO 2017 and Pascal VOC 2007 detection datasets.



## **9. Improving Robustness against Real-World and Worst-Case Distribution Shifts through Decision Region Quantification**

cs.LG

**SubmitDate**: 2022-05-19    [paper-pdf](http://arxiv.org/pdf/2205.09619v1)

**Authors**: Leo Schwinn, Leon Bungert, An Nguyen, René Raab, Falk Pulsmeyer, Doina Precup, Björn Eskofier, Dario Zanca

**Abstracts**: The reliability of neural networks is essential for their use in safety-critical applications. Existing approaches generally aim at improving the robustness of neural networks to either real-world distribution shifts (e.g., common corruptions and perturbations, spatial transformations, and natural adversarial examples) or worst-case distribution shifts (e.g., optimized adversarial examples). In this work, we propose the Decision Region Quantification (DRQ) algorithm to improve the robustness of any differentiable pre-trained model against both real-world and worst-case distribution shifts in the data. DRQ analyzes the robustness of local decision regions in the vicinity of a given data point to make more reliable predictions. We theoretically motivate the DRQ algorithm by showing that it effectively smooths spurious local extrema in the decision surface. Furthermore, we propose an implementation using targeted and untargeted adversarial attacks. An extensive empirical evaluation shows that DRQ increases the robustness of adversarially and non-adversarially trained models against real-world and worst-case distribution shifts on several computer vision benchmark datasets.



## **10. Transferable Physical Attack against Object Detection with Separable Attention**

cs.CV

**SubmitDate**: 2022-05-19    [paper-pdf](http://arxiv.org/pdf/2205.09592v1)

**Authors**: Yu Zhang, Zhiqiang Gong, Yichuang Zhang, YongQian Li, Kangcheng Bin, Jiahao Qi, Wei Xue, Ping Zhong

**Abstracts**: Transferable adversarial attack is always in the spotlight since deep learning models have been demonstrated to be vulnerable to adversarial samples. However, existing physical attack methods do not pay enough attention on transferability to unseen models, thus leading to the poor performance of black-box attack.In this paper, we put forward a novel method of generating physically realizable adversarial camouflage to achieve transferable attack against detection models. More specifically, we first introduce multi-scale attention maps based on detection models to capture features of objects with various resolutions. Meanwhile, we adopt a sequence of composite transformations to obtain the averaged attention maps, which could curb model-specific noise in the attention and thus further boost transferability. Unlike the general visualization interpretation methods where model attention should be put on the foreground object as much as possible, we carry out attack on separable attention from the opposite perspective, i.e. suppressing attention of the foreground and enhancing that of the background. Consequently, transferable adversarial camouflage could be yielded efficiently with our novel attention-based loss function. Extensive comparison experiments verify the superiority of our method to state-of-the-art methods.



## **11. On Trace of PGD-Like Adversarial Attacks**

cs.CV

**SubmitDate**: 2022-05-19    [paper-pdf](http://arxiv.org/pdf/2205.09586v1)

**Authors**: Mo Zhou, Vishal M. Patel

**Abstracts**: Adversarial attacks pose safety and security concerns for deep learning applications. Yet largely imperceptible, a strong PGD-like attack may leave strong trace in the adversarial example. Since attack triggers the local linearity of a network, we speculate network behaves in different extents of linearity for benign examples and adversarial examples. Thus, we construct Adversarial Response Characteristics (ARC) features to reflect the model's gradient consistency around the input to indicate the extent of linearity. Under certain conditions, it shows a gradually varying pattern from benign example to adversarial example, as the later leads to Sequel Attack Effect (SAE). ARC feature can be used for informed attack detection (perturbation magnitude is known) with binary classifier, or uninformed attack detection (perturbation magnitude is unknown) with ordinal regression. Due to the uniqueness of SAE to PGD-like attacks, ARC is also capable of inferring other attack details such as loss function, or the ground-truth label as a post-processing defense. Qualitative and quantitative evaluations manifest the effectiveness of ARC feature on CIFAR-10 w/ ResNet-18 and ImageNet w/ ResNet-152 and SwinT-B-IN1K with considerable generalization among PGD-like attacks despite domain shift. Our method is intuitive, light-weighted, non-intrusive, and data-undemanding.



## **12. Defending Against Adversarial Attacks by Energy Storage Facility**

cs.CR

arXiv admin note: text overlap with arXiv:1904.06606 by other authors

**SubmitDate**: 2022-05-19    [paper-pdf](http://arxiv.org/pdf/2205.09522v1)

**Authors**: Jiawei Li, Jianxiao Wang, Lin Chen, Yang Yu

**Abstracts**: Adversarial attacks on data-driven algorithms applied in pow-er system will be a new type of threat on grid security. Litera-ture has demonstrated the adversarial attack on deep-neural network can significantly misleading the load forecast of a power system. However, it is unclear how the new type of at-tack impact on the operation of grid system. In this research, we manifest that the adversarial algorithm attack induces a significant cost-increase risk which will be exacerbated by the growing penetration of intermittent renewable energy. In Texas, a 5% adversarial attack can increase the total generation cost by 17% in a quarter, which account for around 20 million dollars. When wind-energy penetration increases to over 40%, the 5% adver-sarial attack will inflate the generation cost by 23%. Our re-search discovers a novel approach of defending against the adversarial attack: investing on energy-storage system. All current literature focuses on developing algorithm to defending against adversarial attack. We are the first research revealing the capability of using facility in physical system to defending against the adversarial algorithm attack in a system of Internet of Thing, such as smart grid system.



## **13. Enhancing the Transferability of Adversarial Examples via a Few Queries**

cs.CV

**SubmitDate**: 2022-05-19    [paper-pdf](http://arxiv.org/pdf/2205.09518v1)

**Authors**: Xiangyuan Yang, Jie Lin, Hanlin Zhang, Xinyu Yang, Peng Zhao

**Abstracts**: Due to the vulnerability of deep neural networks, the black-box attack has drawn great attention from the community. Though transferable priors decrease the query number of the black-box query attacks in recent efforts, the average number of queries is still larger than 100, which is easily affected by the number of queries limit policy. In this work, we propose a novel method called query prior-based method to enhance the family of fast gradient sign methods and improve their attack transferability by using a few queries. Specifically, for the untargeted attack, we find that the successful attacked adversarial examples prefer to be classified as the wrong categories with higher probability by the victim model. Therefore, the weighted augmented cross-entropy loss is proposed to reduce the gradient angle between the surrogate model and the victim model for enhancing the transferability of the adversarial examples. Theoretical analysis and extensive experiments demonstrate that our method could significantly improve the transferability of gradient-based adversarial attacks on CIFAR10/100 and ImageNet and outperform the black-box query attack with the same few queries.



## **14. GUARD: Graph Universal Adversarial Defense**

cs.LG

Preprint. Code is publicly available at  https://github.com/EdisonLeeeee/GUARD

**SubmitDate**: 2022-05-19    [paper-pdf](http://arxiv.org/pdf/2204.09803v2)

**Authors**: Jintang Li, Jie Liao, Ruofan Wu, Liang Chen, Jiawang Dan, Changhua Meng, Zibin Zheng, Weiqiang Wang

**Abstracts**: Graph convolutional networks (GCNs) have shown to be vulnerable to small adversarial perturbations, which becomes a severe threat and largely limits their applications in security-critical scenarios. To mitigate such a threat, considerable research efforts have been devoted to increasing the robustness of GCNs against adversarial attacks. However, current approaches for defense are typically designed for the whole graph and consider the global performance, posing challenges in protecting important local nodes from stronger adversarial targeted attacks. In this work, we present a simple yet effective method, named Graph Universal Adversarial Defense (GUARD). Unlike previous works, GUARD protects each individual node from attacks with a universal defensive patch, which is generated once and can be applied to any node (node-agnostic) in a graph. Extensive experiments on four benchmark datasets demonstrate that our method significantly improves robustness for several established GCNs against multiple adversarial attacks and outperforms state-of-the-art defense methods by large margins. Our code is publicly available at https://github.com/EdisonLeeeee/GUARD.



## **15. Sparse Adversarial Attack in Multi-agent Reinforcement Learning**

cs.AI

**SubmitDate**: 2022-05-19    [paper-pdf](http://arxiv.org/pdf/2205.09362v1)

**Authors**: Yizheng Hu, Zhihua Zhang

**Abstracts**: Cooperative multi-agent reinforcement learning (cMARL) has many real applications, but the policy trained by existing cMARL algorithms is not robust enough when deployed. There exist also many methods about adversarial attacks on the RL system, which implies that the RL system can suffer from adversarial attacks, but most of them focused on single agent RL. In this paper, we propose a \textit{sparse adversarial attack} on cMARL systems. We use (MA)RL with regularization to train the attack policy. Our experiments show that the policy trained by the current cMARL algorithm can obtain poor performance when only one or a few agents in the team (e.g., 1 of 8 or 5 of 25) were attacked at a few timesteps (e.g., attack 3 of total 40 timesteps).



## **16. Backdoor Attacks on Bayesian Neural Networks using Reverse Distribution**

cs.CR

9 pages, 7 figures

**SubmitDate**: 2022-05-18    [paper-pdf](http://arxiv.org/pdf/2205.09167v1)

**Authors**: Zhixin Pan, Prabhat Mishra

**Abstracts**: Due to cost and time-to-market constraints, many industries outsource the training process of machine learning models (ML) to third-party cloud service providers, popularly known as ML-asa-Service (MLaaS). MLaaS creates opportunity for an adversary to provide users with backdoored ML models to produce incorrect predictions only in extremely rare (attacker-chosen) scenarios. Bayesian neural networks (BNN) are inherently immune against backdoor attacks since the weights are designed to be marginal distributions to quantify the uncertainty. In this paper, we propose a novel backdoor attack based on effective learning and targeted utilization of reverse distribution. This paper makes three important contributions. (1) To the best of our knowledge, this is the first backdoor attack that can effectively break the robustness of BNNs. (2) We produce reverse distributions to cancel the original distributions when the trigger is activated. (3) We propose an efficient solution for merging probability distributions in BNNs. Experimental results on diverse benchmark datasets demonstrate that our proposed attack can achieve the attack success rate (ASR) of 100%, while the ASR of the state-of-the-art attacks is lower than 60%.



## **17. VLC Physical Layer Security through RIS-aided Jamming Receiver for 6G Wireless Networks**

cs.CR

**SubmitDate**: 2022-05-18    [paper-pdf](http://arxiv.org/pdf/2205.09026v1)

**Authors**: Simone Soderi, Alessandro Brighente, Federico Turrin, Mauro Conti

**Abstracts**: Visible Light Communication (VLC) is one the most promising enabling technology for future 6G networks to overcome Radio-Frequency (RF)-based communication limitations thanks to a broader bandwidth, higher data rate, and greater efficiency. However, from the security perspective, VLCs suffer from all known wireless communication security threats (e.g., eavesdropping and integrity attacks). For this reason, security researchers are proposing innovative Physical Layer Security (PLS) solutions to protect such communication. Among the different solutions, the novel Reflective Intelligent Surface (RIS) technology coupled with VLCs has been successfully demonstrated in recent work to improve the VLC communication capacity. However, to date, the literature still lacks analysis and solutions to show the PLS capability of RIS-based VLC communication. In this paper, we combine watermarking and jamming primitives through the Watermark Blind Physical Layer Security (WBPLSec) algorithm to secure VLC communication at the physical layer. Our solution leverages RIS technology to improve the security properties of the communication. By using an optimization framework, we can calculate RIS phases to maximize the WBPLSec jamming interference schema over a predefined area in the room. In particular, compared to a scenario without RIS, our solution improves the performance in terms of secrecy capacity without any assumption about the adversary's location. We validate through numerical evaluations the positive impact of RIS-aided solution to increase the secrecy capacity of the legitimate jamming receiver in a VLC indoor scenario. Our results show that the introduction of RIS technology extends the area where secure communication occurs and that by increasing the number of RIS elements the outage probability decreases.



## **18. Increasing-Margin Adversarial (IMA) Training to Improve Adversarial Robustness of Neural Networks**

cs.CV

13 pages

**SubmitDate**: 2022-05-18    [paper-pdf](http://arxiv.org/pdf/2005.09147v8)

**Authors**: Linhai Ma, Liang Liang

**Abstracts**: Deep neural networks (DNNs) are vulnerable to adversarial noises. By adding adversarial noises to training samples, adversarial training can improve the model's robustness against adversarial noises. However, adversarial training samples with excessive noises can harm standard accuracy, which may be unacceptable for many medical image analysis applications. This issue has been termed the trade-off between standard accuracy and adversarial robustness. In this paper, we hypothesize that this issue may be alleviated if the adversarial samples for training are placed right on the decision boundaries. Based on this hypothesis, we design an adaptive adversarial training method, named IMA. For each individual training sample, IMA makes a sample-wise estimation of the upper bound of the adversarial perturbation. In the training process, each of the sample-wise adversarial perturbations is gradually increased to match the margin. Once an equilibrium state is reached, the adversarial perturbations will stop increasing. IMA is evaluated on publicly available datasets under two popular adversarial attacks, PGD and IFGSM. The results show that: (1) IMA significantly improves adversarial robustness of DNN classifiers, which achieves the state-of-the-art performance; (2) IMA has a minimal reduction in clean accuracy among all competing defense methods; (3) IMA can be applied to pretrained models to reduce time cost; (4) IMA can be applied to the state-of-the-art medical image segmentation networks, with outstanding performance. We hope our work may help to lift the trade-off between adversarial robustness and clean accuracy and facilitate the development of robust applications in the medical field. The source code will be released when this paper is published.



## **19. Property Unlearning: A Defense Strategy Against Property Inference Attacks**

cs.CR

**SubmitDate**: 2022-05-18    [paper-pdf](http://arxiv.org/pdf/2205.08821v1)

**Authors**: Joshua Stock, Jens Wettlaufer, Daniel Demmler, Hannes Federrath

**Abstracts**: During the training of machine learning models, they may store or "learn" more information about the training data than what is actually needed for the prediction or classification task. This is exploited by property inference attacks which aim at extracting statistical properties from the training data of a given model without having access to the training data itself. These properties may include the quality of pictures to identify the camera model, the age distribution to reveal the target audience of a product, or the included host types to refine a malware attack in computer networks. This attack is especially accurate when the attacker has access to all model parameters, i.e., in a white-box scenario. By defending against such attacks, model owners are able to ensure that their training data, associated properties, and thus their intellectual property stays private, even if they deliberately share their models, e.g., to train collaboratively, or if models are leaked. In this paper, we introduce property unlearning, an effective defense mechanism against white-box property inference attacks, independent of the training data type, model task, or number of properties. Property unlearning mitigates property inference attacks by systematically changing the trained weights and biases of a target model such that an adversary cannot extract chosen properties. We empirically evaluate property unlearning on three different data sets, including tabular and image data, and two types of artificial neural networks. Our results show that property unlearning is both efficient and reliable to protect machine learning models against property inference attacks, with a good privacy-utility trade-off. Furthermore, our approach indicates that this mechanism is also effective to unlearn multiple properties.



## **20. Passive Defense Against 3D Adversarial Point Clouds Through the Lens of 3D Steganalysis**

cs.MM

**SubmitDate**: 2022-05-18    [paper-pdf](http://arxiv.org/pdf/2205.08738v1)

**Authors**: Jiahao Zhu

**Abstracts**: Nowadays, 3D data plays an indelible role in the computer vision field. However, extensive studies have proved that deep neural networks (DNNs) fed with 3D data, such as point clouds, are susceptible to adversarial examples, which aim to misguide DNNs and might bring immeasurable losses. Currently, 3D adversarial point clouds are chiefly generated in three fashions, i.e., point shifting, point adding, and point dropping. These point manipulations would modify geometrical properties and local correlations of benign point clouds more or less. Motivated by this basic fact, we propose to defend such adversarial examples with the aid of 3D steganalysis techniques. Specifically, we first introduce an adversarial attack and defense model adapted from the celebrated Prisoners' Problem in steganography to help us comprehend 3D adversarial attack and defense more generally. Then we rethink two significant but vague concepts in the field of adversarial example, namely, active defense and passive defense, from the perspective of steganalysis. Most importantly, we design a 3D adversarial point cloud detector through the lens of 3D steganalysis. Our detector is double-blind, that is to say, it does not rely on the exact knowledge of the adversarial attack means and victim models. To enable the detector to effectively detect malicious point clouds, we craft a 64-D discriminant feature set, including features related to first-order and second-order local descriptions of point clouds. To our knowledge, this work is the first to apply 3D steganalysis to 3D adversarial example defense. Extensive experimental results demonstrate that the proposed 3D adversarial point cloud detector can achieve good detection performance on multiple types of 3D adversarial point clouds.



## **21. Policy Distillation with Selective Input Gradient Regularization for Efficient Interpretability**

cs.LG

**SubmitDate**: 2022-05-18    [paper-pdf](http://arxiv.org/pdf/2205.08685v1)

**Authors**: Jinwei Xing, Takashi Nagata, Xinyun Zou, Emre Neftci, Jeffrey L. Krichmar

**Abstracts**: Although deep Reinforcement Learning (RL) has proven successful in a wide range of tasks, one challenge it faces is interpretability when applied to real-world problems. Saliency maps are frequently used to provide interpretability for deep neural networks. However, in the RL domain, existing saliency map approaches are either computationally expensive and thus cannot satisfy the real-time requirement of real-world scenarios or cannot produce interpretable saliency maps for RL policies. In this work, we propose an approach of Distillation with selective Input Gradient Regularization (DIGR) which uses policy distillation and input gradient regularization to produce new policies that achieve both high interpretability and computation efficiency in generating saliency maps. Our approach is also found to improve the robustness of RL policies to multiple adversarial attacks. We conduct experiments on three tasks, MiniGrid (Fetch Object), Atari (Breakout) and CARLA Autonomous Driving, to demonstrate the importance and effectiveness of our approach.



## **22. Longest Chain Consensus Under Bandwidth Constraint**

cs.CR

**SubmitDate**: 2022-05-18    [paper-pdf](http://arxiv.org/pdf/2111.12332v3)

**Authors**: Joachim Neu, Srivatsan Sridhar, Lei Yang, David Tse, Mohammad Alizadeh

**Abstracts**: Spamming attacks are a serious concern for consensus protocols, as witnessed by recent outages of a major blockchain, Solana. They cause congestion and excessive message delays in a real network due to its bandwidth constraints. In contrast, longest chain (LC), an important family of consensus protocols, has previously only been proven secure assuming an idealized network model in which all messages are delivered within bounded delay. This model-reality mismatch is further aggravated for Proof-of-Stake (PoS) LC where the adversary can spam the network with equivocating blocks. Hence, we extend the network model to capture bandwidth constraints, under which nodes now need to choose carefully which blocks to spend their limited download budget on. To illustrate this point, we show that 'download along the longest header chain', a natural download rule for Proof-of-Work (PoW) LC, is insecure for PoS LC. We propose a simple rule 'download towards the freshest block', formalize two common heuristics 'not downloading equivocations' and 'blocklisting', and prove in a unified framework that PoS LC with any one of these download rules is secure in bandwidth-constrained networks. In experiments, we validate our claims and showcase the behavior of these download rules under attack. By composing multiple instances of a PoS LC protocol with a suitable download rule in parallel, we obtain a PoS consensus protocol that achieves a constant fraction of the network's throughput limit even under worst-case adversarial strategies.



## **23. An Integrated Approach for Energy Efficient Handover and Key Distribution Protocol for Secure NC-enabled Small Cells**

cs.NI

Preprint of the paper accepted at Computer Networks

**SubmitDate**: 2022-05-17    [paper-pdf](http://arxiv.org/pdf/2205.08641v1)

**Authors**: Vipindev Adat Vasudevan, Muhammad Tayyab, George P. Koudouridis, Xavier Gelabert, Ilias Politis

**Abstracts**: Future wireless networks must serve dense mobile networks with high data rates, keeping energy requirements to a possible minimum. The small cell-based network architecture and device-to-device (D2D) communication are already being considered part of 5G networks and beyond. In such environments, network coding (NC) can be employed to achieve both higher throughput and energy efficiency. However, NC-enabled systems need to address security challenges specific to NC, such as pollution attacks. All integrity schemes against pollution attacks generally require proper key distribution and management to ensure security in a mobile environment. Additionally, the mobility requirements in small cell environments are more challenging and demanding in terms of signaling overhead. This paper proposes a blockchain-assisted key distribution protocol tailored for MAC-based integrity schemes, which combined with an uplink reference signal (UL RS) handover mechanism, enables energy efficient secure NC. The performance analysis of the protocol during handover scenarios indicates its suitability for ensuring high level of security against pollution attacks in dense small cell environments with multiple adversaries being present. Furthermore, the proposed scheme achieves lower bandwidth and signaling overhead during handover compared to legacy schemes and the signaling cost reduces significantly as the communication progresses, thus enhancing the network's cumulative energy efficiency.



## **24. Hierarchical Distribution-Aware Testing of Deep Learning**

cs.SE

Under Review

**SubmitDate**: 2022-05-17    [paper-pdf](http://arxiv.org/pdf/2205.08589v1)

**Authors**: Wei Huang, Xingyu Zhao, Alec Banks, Victoria Cox, Xiaowei Huang

**Abstracts**: With its growing use in safety/security-critical applications, Deep Learning (DL) has raised increasing concerns regarding its dependability. In particular, DL has a notorious problem of lacking robustness. Despite recent efforts made in detecting Adversarial Examples (AEs) via state-of-the-art attacking and testing methods, they are normally input distribution agnostic and/or disregard the perception quality of AEs. Consequently, the detected AEs are irrelevant inputs in the application context or unnatural/unrealistic that can be easily noticed by humans. This may lead to a limited effect on improving the DL model's dependability, as the testing budget is likely to be wasted on detecting AEs that are encountered very rarely in its real-life operations. In this paper, we propose a new robustness testing approach for detecting AEs that considers both the input distribution and the perceptual quality of inputs. The two considerations are encoded by a novel hierarchical mechanism. First, at the feature level, the input data distribution is extracted and approximated by data compression techniques and probability density estimators. Such quantified feature level distribution, together with indicators that are highly correlated with local robustness, are considered in selecting test seeds. Given a test seed, we then develop a two-step genetic algorithm for local test case generation at the pixel level, in which two fitness functions work alternatively to control the quality of detected AEs. Finally, extensive experiments confirm that our holistic approach considering hierarchical distributions at feature and pixel levels is superior to state-of-the-arts that either disregard any input distribution or only consider a single (non-hierarchical) distribution, in terms of not only the quality of detected AEs but also improving the overall robustness of the DL model under testing.



## **25. F3B: A Low-Latency Commit-and-Reveal Architecture to Mitigate Blockchain Front-Running**

cs.CR

**SubmitDate**: 2022-05-17    [paper-pdf](http://arxiv.org/pdf/2205.08529v1)

**Authors**: Haoqian Zhang, Louis-Henri Merino, Vero Estrada-Galinanes, Bryan Ford

**Abstracts**: Front-running attacks, which benefit from advanced knowledge of pending transactions, have proliferated in the cryptocurrency space since the emergence of decentralized finance. Front-running causes devastating losses to honest participants$\unicode{x2013}$estimated at \$280M each month$\unicode{x2013}$and endangers the fairness of the ecosystem. We present Flash Freezing Flash Boys (F3B), a blockchain architecture to address front-running attacks by relying on a commit-and-reveal scheme where the contents of transactions are encrypted and later revealed by a decentralized secret-management committee once the underlying consensus layer has committed the transaction. F3B mitigates front-running attacks because an adversary can no longer read the content of a transaction before commitment, thus preventing the adversary from benefiting from advance knowledge of pending transactions. We design F3B to be agnostic to the underlying consensus algorithm and compatible with legacy smart contracts by addressing front-running at the blockchain architecture level. Unlike existing commit-and-reveal approaches, F3B only requires writing data onto the underlying blockchain once, establishing a significant overhead reduction. An exploration of F3B shows that with a secret-management committee consisting of 8 and 128 members, F3B presents between 0.1 and 1.8 seconds of transaction-processing latency, respectively.



## **26. On the Privacy of Decentralized Machine Learning**

cs.CR

17 pages

**SubmitDate**: 2022-05-17    [paper-pdf](http://arxiv.org/pdf/2205.08443v1)

**Authors**: Dario Pasquini, Mathilde Raynal, Carmela Troncoso

**Abstracts**: In this work, we carry out the first, in-depth, privacy analysis of Decentralized Learning -- a collaborative machine learning framework aimed at circumventing the main limitations of federated learning. We identify the decentralized learning properties that affect users' privacy and we introduce a suite of novel attacks for both passive and active decentralized adversaries. We demonstrate that, contrary to what is claimed by decentralized learning proposers, decentralized learning does not offer any security advantages over more practical approaches such as federated learning. Rather, it tends to degrade users' privacy by increasing the attack surface and enabling any user in the system to perform powerful privacy attacks such as gradient inversion, and even gain full control over honest users' local model. We also reveal that, given the state of the art in protections, privacy-preserving configurations of decentralized learning require abandoning any possible advantage over the federated setup, completely defeating the objective of the decentralized approach.



## **27. Can You Still See Me?: Reconstructing Robot Operations Over End-to-End Encrypted Channels**

cs.CR

13 pages, 7 figures, poster presented at wisec'22

**SubmitDate**: 2022-05-17    [paper-pdf](http://arxiv.org/pdf/2205.08426v1)

**Authors**: Ryan Shah, Chuadhry Mujeeb Ahmed, Shishir Nagaraja

**Abstracts**: Connected robots play a key role in Industry 4.0, providing automation and higher efficiency for many industrial workflows. Unfortunately, these robots can leak sensitive information regarding these operational workflows to remote adversaries. While there exists mandates for the use of end-to-end encryption for data transmission in such settings, it is entirely possible for passive adversaries to fingerprint and reconstruct entire workflows being carried out -- establishing an understanding of how facilities operate. In this paper, we investigate whether a remote attacker can accurately fingerprint robot movements and ultimately reconstruct operational workflows. Using a neural network approach to traffic analysis, we find that one can predict TLS-encrypted movements with around \textasciitilde60\% accuracy, increasing to near-perfect accuracy under realistic network conditions. Further, we also find that attackers can reconstruct warehousing workflows with similar success. Ultimately, simply adopting best cybersecurity practices is clearly not enough to stop even weak (passive) adversaries.



## **28. Bankrupting DoS Attackers Despite Uncertainty**

cs.CR

**SubmitDate**: 2022-05-17    [paper-pdf](http://arxiv.org/pdf/2205.08287v1)

**Authors**: Trisha Chakraborty, Abir Islam, Valerie King, Daniel Rayborn, Jared Saia, Maxwell Young

**Abstracts**: On-demand provisioning in the cloud allows for services to remain available despite massive denial-of-service (DoS) attacks. Unfortunately, on-demand provisioning is expensive and must be weighed against the costs incurred by an adversary. This leads to a recent threat known as economic denial-of-sustainability (EDoS), where the cost for defending a service is higher than that of attacking.   A natural approach for combating EDoS is to impose costs via resource burning (RB). Here, a client must verifiably consume resources -- for example, by solving a computational challenge -- before service is rendered. However, prior approaches with security guarantees do not account for the cost on-demand provisioning.   Another valuable defensive tool is to use a classifier in order to discern good jobs from a legitimate client, versus bad jobs from the adversary. However, while useful, uncertainty arises from classification error, which still allows bad jobs to consume server resources. Thus, classification is not a solution by itself.   Here, we propose an EDoS defense, RootDef, that leverages both RB and classification, while accounting for both the costs of resource burning and on-demand provisioning. Specifically, against an adversary that expends $B$ resources to attack, the total cost for defending is $\tilde{O}( \sqrt{B\,g} + B^{2/3} + g)$, where $g$ is the number of good jobs and $\tilde{O}$ refers to hidden logarithmic factors in the total number of jobs $n$. Notably, for large $B$ relative to $g$, the adversary has higher cost, implying that the algorithm has an economic advantage. Finally, we prove a lower bound showing that RootDef has total costs that are asymptotically tight up to logarithmic factors in $n$.



## **29. How Not to Handle Keys: Timing Attacks on FIDO Authenticator Privacy**

cs.CR

to be published in the 22nd Privacy Enhancing Technologies Symposium  (PETS 2022)

**SubmitDate**: 2022-05-17    [paper-pdf](http://arxiv.org/pdf/2205.08071v1)

**Authors**: Michal Kepkowski, Lucjan Hanzlik, Ian Wood, Mohamed Ali Kaafar

**Abstracts**: This paper presents a timing attack on the FIDO2 (Fast IDentity Online) authentication protocol that allows attackers to link user accounts stored in vulnerable authenticators, a serious privacy concern. FIDO2 is a new standard specified by the FIDO industry alliance for secure token online authentication. It complements the W3C WebAuthn specification by providing means to use a USB token or other authenticator as a second factor during the authentication process. From a cryptographic perspective, the protocol is a simple challenge-response where the elliptic curve digital signature algorithm is used to sign challenges. To protect the privacy of the user the token uses unique key pairs per service. To accommodate for small memory, tokens use various techniques that make use of a special parameter called a key handle sent by the service to the token. We identify and analyse a vulnerability in the way the processing of key handles is implemented that allows attackers to remotely link user accounts on multiple services. We show that for vulnerable authenticators there is a difference between the time it takes to process a key handle for a different service but correct authenticator, and for a different authenticator but correct service. This difference can be used to perform a timing attack allowing an adversary to link user's accounts across services. We present several real world examples of adversaries that are in a position to execute our attack and can benefit from linking accounts. We found that two of the eight hardware authenticators we tested were vulnerable despite FIDO level 1 certification. This vulnerability cannot be easily mitigated on authenticators because, for security reasons, they usually do not allow firmware updates. In addition, we show that due to the way existing browsers implement the WebAuthn standard, the attack can be executed remotely.



## **30. User-Level Differential Privacy against Attribute Inference Attack of Speech Emotion Recognition in Federated Learning**

cs.CR

**SubmitDate**: 2022-05-17    [paper-pdf](http://arxiv.org/pdf/2204.02500v2)

**Authors**: Tiantian Feng, Raghuveer Peri, Shrikanth Narayanan

**Abstracts**: Many existing privacy-enhanced speech emotion recognition (SER) frameworks focus on perturbing the original speech data through adversarial training within a centralized machine learning setup. However, this privacy protection scheme can fail since the adversary can still access the perturbed data. In recent years, distributed learning algorithms, especially federated learning (FL), have gained popularity to protect privacy in machine learning applications. While FL provides good intuition to safeguard privacy by keeping the data on local devices, prior work has shown that privacy attacks, such as attribute inference attacks, are achievable for SER systems trained using FL. In this work, we propose to evaluate the user-level differential privacy (UDP) in mitigating the privacy leaks of the SER system in FL. UDP provides theoretical privacy guarantees with privacy parameters $\epsilon$ and $\delta$. Our results show that the UDP can effectively decrease attribute information leakage while keeping the utility of the SER system with the adversary accessing one model update. However, the efficacy of the UDP suffers when the FL system leaks more model updates to the adversary. We make the code publicly available to reproduce the results in https://github.com/usc-sail/fed-ser-leakage.



## **31. RoVISQ: Reduction of Video Service Quality via Adversarial Attacks on Deep Learning-based Video Compression**

cs.CV

**SubmitDate**: 2022-05-16    [paper-pdf](http://arxiv.org/pdf/2203.10183v2)

**Authors**: Jung-Woo Chang, Mojan Javaheripi, Seira Hidano, Farinaz Koushanfar

**Abstracts**: Video compression plays a crucial role in video streaming and classification systems by maximizing the end-user quality of experience (QoE) at a given bandwidth budget. In this paper, we conduct the first systematic study for adversarial attacks on deep learning-based video compression and downstream classification systems. Our attack framework, dubbed RoVISQ, manipulates the Rate-Distortion (R-D) relationship of a video compression model to achieve one or both of the following goals: (1) increasing the network bandwidth, (2) degrading the video quality for end-users. We further devise new objectives for targeted and untargeted attacks to a downstream video classification service. Finally, we design an input-invariant perturbation that universally disrupts video compression and classification systems in real time. Unlike previously proposed attacks on video classification, our adversarial perturbations are the first to withstand compression. We empirically show the resilience of RoVISQ attacks against various defenses, i.e., adversarial training, video denoising, and JPEG compression. Our extensive experimental results on various video datasets show RoVISQ attacks deteriorate peak signal-to-noise ratio by up to 5.6dB and the bit-rate by up to 2.4 times while achieving over 90% attack success rate on a downstream classifier.



## **32. Classification Auto-Encoder based Detector against Diverse Data Poisoning Attacks**

cs.LG

This work has been submitted to the IEEE for possible publication.  Copyright may be transferred without notice, after which this version may no  longer be accessible

**SubmitDate**: 2022-05-16    [paper-pdf](http://arxiv.org/pdf/2108.04206v2)

**Authors**: Fereshteh Razmi, Li Xiong

**Abstracts**: Poisoning attacks are a category of adversarial machine learning threats in which an adversary attempts to subvert the outcome of the machine learning systems by injecting crafted data into training data set, thus increasing the machine learning model's test error. The adversary can tamper with the data feature space, data labels, or both, each leading to a different attack strategy with different strengths. Various detection approaches have recently emerged, each focusing on one attack strategy. The Achilles heel of many of these detection approaches is their dependence on having access to a clean, untampered data set. In this paper, we propose CAE, a Classification Auto-Encoder based detector against diverse poisoned data. CAE can detect all forms of poisoning attacks using a combination of reconstruction and classification errors without having any prior knowledge of the attack strategy. We show that an enhanced version of CAE (called CAE+) does not have to employ a clean data set to train the defense model. Our experimental results on three real datasets MNIST, Fashion-MNIST and CIFAR demonstrate that our proposed method can maintain its functionality under up to 30% contaminated data and help the defended SVM classifier to regain its best accuracy.



## **33. Transferability of Adversarial Attacks on Synthetic Speech Detection**

cs.SD

5 pages, submit to Interspeech2022

**SubmitDate**: 2022-05-16    [paper-pdf](http://arxiv.org/pdf/2205.07711v1)

**Authors**: Jiacheng Deng, Shunyi Chen, Li Dong, Diqun Yan, Rangding Wang

**Abstracts**: Synthetic speech detection is one of the most important research problems in audio security. Meanwhile, deep neural networks are vulnerable to adversarial attacks. Therefore, we establish a comprehensive benchmark to evaluate the transferability of adversarial attacks on the synthetic speech detection task. Specifically, we attempt to investigate: 1) The transferability of adversarial attacks between different features. 2) The influence of varying extraction hyperparameters of features on the transferability of adversarial attacks. 3) The effect of clipping or self-padding operation on the transferability of adversarial attacks. By performing these analyses, we summarise the weaknesses of synthetic speech detectors and the transferability behaviours of adversarial attacks, which provide insights for future research. More details can be found at https://gitee.com/djc_QRICK/Attack-Transferability-On-Synthetic-Detection.



## **34. SemAttack: Natural Textual Attacks via Different Semantic Spaces**

cs.CL

Published at Findings of NAACL 2022

**SubmitDate**: 2022-05-16    [paper-pdf](http://arxiv.org/pdf/2205.01287v2)

**Authors**: Boxin Wang, Chejian Xu, Xiangyu Liu, Yu Cheng, Bo Li

**Abstracts**: Recent studies show that pre-trained language models (LMs) are vulnerable to textual adversarial attacks. However, existing attack methods either suffer from low attack success rates or fail to search efficiently in the exponentially large perturbation space. We propose an efficient and effective framework SemAttack to generate natural adversarial text by constructing different semantic perturbation functions. In particular, SemAttack optimizes the generated perturbations constrained on generic semantic spaces, including typo space, knowledge space (e.g., WordNet), contextualized semantic space (e.g., the embedding space of BERT clusterings), or the combination of these spaces. Thus, the generated adversarial texts are more semantically close to the original inputs. Extensive experiments reveal that state-of-the-art (SOTA) large-scale LMs (e.g., DeBERTa-v2) and defense strategies (e.g., FreeLB) are still vulnerable to SemAttack. We further demonstrate that SemAttack is general and able to generate natural adversarial texts for different languages (e.g., English and Chinese) with high attack success rates. Human evaluations also confirm that our generated adversarial texts are natural and barely affect human performance. Our code is publicly available at https://github.com/AI-secure/SemAttack.



## **35. SGBA: A Stealthy Scapegoat Backdoor Attack against Deep Neural Networks**

cs.CR

**SubmitDate**: 2022-05-16    [paper-pdf](http://arxiv.org/pdf/2104.01026v3)

**Authors**: Ying He, Zhili Shen, Chang Xia, Jingyu Hua, Wei Tong, Sheng Zhong

**Abstracts**: Outsourced deep neural networks have been demonstrated to suffer from patch-based trojan attacks, in which an adversary poisons the training sets to inject a backdoor in the obtained model so that regular inputs can be still labeled correctly while those carrying a specific trigger are falsely given a target label. Due to the severity of such attacks, many backdoor detection and containment systems have recently, been proposed for deep neural networks. One major category among them are various model inspection schemes, which hope to detect backdoors before deploying models from non-trusted third-parties. In this paper, we show that such state-of-the-art schemes can be defeated by a so-called Scapegoat Backdoor Attack, which introduces a benign scapegoat trigger in data poisoning to prevent the defender from reversing the real abnormal trigger. In addition, it confines the values of network parameters within the same variances of those from clean model during training, which further significantly enhances the difficulty of the defender to learn the differences between legal and illegal models through machine-learning approaches. Our experiments on 3 popular datasets show that it can escape detection by all five state-of-the-art model inspection schemes. Moreover, this attack brings almost no side-effects on the attack effectiveness and guarantees the universal feature of the trigger compared with original patch-based trojan attacks.



## **36. Unreasonable Effectiveness of Last Hidden Layer Activations for Adversarial Robustness**

cs.LG

IEEE COMPSAC 2022 publication full version

**SubmitDate**: 2022-05-16    [paper-pdf](http://arxiv.org/pdf/2202.07342v2)

**Authors**: Omer Faruk Tuna, Ferhat Ozgur Catak, M. Taner Eskil

**Abstracts**: In standard Deep Neural Network (DNN) based classifiers, the general convention is to omit the activation function in the last (output) layer and directly apply the softmax function on the logits to get the probability scores of each class. In this type of architectures, the loss value of the classifier against any output class is directly proportional to the difference between the final probability score and the label value of the associated class. Standard White-box adversarial evasion attacks, whether targeted or untargeted, mainly try to exploit the gradient of the model loss function to craft adversarial samples and fool the model. In this study, we show both mathematically and experimentally that using some widely known activation functions in the output layer of the model with high temperature values has the effect of zeroing out the gradients for both targeted and untargeted attack cases, preventing attackers from exploiting the model's loss function to craft adversarial samples. We've experimentally verified the efficacy of our approach on MNIST (Digit), CIFAR10 datasets. Detailed experiments confirmed that our approach substantially improves robustness against gradient-based targeted and untargeted attack threats. And, we showed that the increased non-linearity at the output layer has some additional benefits against some other attack methods like Deepfool attack.



## **37. Attacking and Defending Deep Reinforcement Learning Policies**

cs.LG

nine pages

**SubmitDate**: 2022-05-16    [paper-pdf](http://arxiv.org/pdf/2205.07626v1)

**Authors**: Chao Wang

**Abstracts**: Recent studies have shown that deep reinforcement learning (DRL) policies are vulnerable to adversarial attacks, which raise concerns about applications of DRL to safety-critical systems. In this work, we adopt a principled way and study the robustness of DRL policies to adversarial attacks from the perspective of robust optimization. Within the framework of robust optimization, optimal adversarial attacks are given by minimizing the expected return of the policy, and correspondingly a good defense mechanism should be realized by improving the worst-case performance of the policy. Considering that attackers generally have no access to the training environment, we propose a greedy attack algorithm, which tries to minimize the expected return of the policy without interacting with the environment, and a defense algorithm, which performs adversarial training in a max-min form. Experiments on Atari game environments show that our attack algorithm is more effective and leads to worse return of the policy than existing attack algorithms, and our defense algorithm yields policies more robust than existing defense methods to a range of adversarial attacks (including our proposed attack algorithm).



## **38. More is Better (Mostly): On the Backdoor Attacks in Federated Graph Neural Networks**

cs.CR

17 pages, 13 figures

**SubmitDate**: 2022-05-16    [paper-pdf](http://arxiv.org/pdf/2202.03195v2)

**Authors**: Jing Xu, Rui Wang, Stefanos Koffas, Kaitai Liang, Stjepan Picek

**Abstracts**: Graph Neural Networks (GNNs) are a class of deep learning-based methods for processing graph domain information. GNNs have recently become a widely used graph analysis method due to their superior ability to learn representations for complex graph data. However, due to privacy concerns and regulation restrictions, centralized GNNs can be difficult to apply to data-sensitive scenarios. Federated learning (FL) is an emerging technology developed for privacy-preserving settings when several parties need to train a shared global model collaboratively. Although several research works have applied FL to train GNNs (Federated GNNs), there is no research on their robustness to backdoor attacks.   This paper bridges this gap by conducting two types of backdoor attacks in Federated GNNs: centralized backdoor attacks (CBA) and distributed backdoor attacks (DBA). CBA is conducted by embedding the same global trigger during training for every malicious party, while DBA is conducted by decomposing a global trigger into separate local triggers and embedding them into the training datasets of different malicious parties, respectively. Our experiments show that the DBA attack success rate is higher than CBA in almost all evaluated cases. For CBA, the attack success rate of all local triggers is similar to the global trigger even if the training set of the adversarial party is embedded with the global trigger. To further explore the properties of two backdoor attacks in Federated GNNs, we evaluate the attack performance for different number of clients, trigger sizes, poisoning intensities, and trigger densities. Moreover, we explore the robustness of DBA and CBA against two state-of-the-art defenses. We find that both attacks are robust against the investigated defenses, necessitating the need to consider backdoor attacks in Federated GNNs as a novel threat that requires custom defenses.



## **39. Learning Classical Readout Quantum PUFs based on single-qubit gates**

quant-ph

12 pages, 9 figures

**SubmitDate**: 2022-05-16    [paper-pdf](http://arxiv.org/pdf/2112.06661v2)

**Authors**: Niklas Pirnay, Anna Pappa, Jean-Pierre Seifert

**Abstracts**: Physical Unclonable Functions (PUFs) have been proposed as a way to identify and authenticate electronic devices. Recently, several ideas have been presented that aim to achieve the same for quantum devices. Some of these constructions apply single-qubit gates in order to provide a secure fingerprint of the quantum device. In this work, we formalize the class of Classical Readout Quantum PUFs (CR-QPUFs) using the statistical query (SQ) model and explicitly show insufficient security for CR-QPUFs based on single qubit rotation gates, when the adversary has SQ access to the CR-QPUF. We demonstrate how a malicious party can learn the CR-QPUF characteristics and forge the signature of a quantum device through a modelling attack using a simple regression of low-degree polynomials. The proposed modelling attack was successfully implemented in a real-world scenario on real IBM Q quantum machines. We thoroughly discuss the prospects and problems of CR-QPUFs where quantum device imperfections are used as a secure fingerprint.



## **40. Manifold Characteristics That Predict Downstream Task Performance**

cs.LG

Currently under review

**SubmitDate**: 2022-05-16    [paper-pdf](http://arxiv.org/pdf/2205.07477v1)

**Authors**: Ruan van der Merwe, Gregory Newman, Etienne Barnard

**Abstracts**: Pretraining methods are typically compared by evaluating the accuracy of linear classifiers, transfer learning performance, or visually inspecting the representation manifold's (RM) lower-dimensional projections. We show that the differences between methods can be understood more clearly by investigating the RM directly, which allows for a more detailed comparison. To this end, we propose a framework and new metric to measure and compare different RMs. We also investigate and report on the RM characteristics for various pretraining methods. These characteristics are measured by applying sequentially larger local alterations to the input data, using white noise injections and Projected Gradient Descent (PGD) adversarial attacks, and then tracking each datapoint. We calculate the total distance moved for each datapoint and the relative change in distance between successive alterations. We show that self-supervised methods learn an RM where alterations lead to large but constant size changes, indicating a smoother RM than fully supervised methods. We then combine these measurements into one metric, the Representation Manifold Quality Metric (RMQM), where larger values indicate larger and less variable step sizes, and show that RMQM correlates positively with performance on downstream tasks.



## **41. Robust Representation via Dynamic Feature Aggregation**

cs.CV

**SubmitDate**: 2022-05-16    [paper-pdf](http://arxiv.org/pdf/2205.07466v1)

**Authors**: Haozhe Liu, Haoqin Ji, Yuexiang Li, Nanjun He, Haoqian Wu, Feng Liu, Linlin Shen, Yefeng Zheng

**Abstracts**: Deep convolutional neural network (CNN) based models are vulnerable to the adversarial attacks. One of the possible reasons is that the embedding space of CNN based model is sparse, resulting in a large space for the generation of adversarial samples. In this study, we propose a method, denoted as Dynamic Feature Aggregation, to compress the embedding space with a novel regularization. Particularly, the convex combination between two samples are regarded as the pivot for aggregation. In the embedding space, the selected samples are guided to be similar to the representation of the pivot. On the other side, to mitigate the trivial solution of such regularization, the last fully-connected layer of the model is replaced by an orthogonal classifier, in which the embedding codes for different classes are processed orthogonally and separately. With the regularization and orthogonal classifier, a more compact embedding space can be obtained, which accordingly improves the model robustness against adversarial attacks. An averaging accuracy of 56.91% is achieved by our method on CIFAR-10 against various attack methods, which significantly surpasses a solid baseline (Mixup) by a margin of 37.31%. More surprisingly, empirical results show that, the proposed method can also achieve the state-of-the-art performance for out-of-distribution (OOD) detection, due to the learned compact feature space. An F1 score of 0.937 is achieved by the proposed method, when adopting CIFAR-10 as in-distribution (ID) dataset and LSUN as OOD dataset. Code is available at https://github.com/HaozheLiu-ST/DynamicFeatureAggregation.



## **42. Diffusion Models for Adversarial Purification**

cs.LG

ICML 2022

**SubmitDate**: 2022-05-16    [paper-pdf](http://arxiv.org/pdf/2205.07460v1)

**Authors**: Weili Nie, Brandon Guo, Yujia Huang, Chaowei Xiao, Arash Vahdat, Anima Anandkumar

**Abstracts**: Adversarial purification refers to a class of defense methods that remove adversarial perturbations using a generative model. These methods do not make assumptions on the form of attack and the classification model, and thus can defend pre-existing classifiers against unseen threats. However, their performance currently falls behind adversarial training methods. In this work, we propose DiffPure that uses diffusion models for adversarial purification: Given an adversarial example, we first diffuse it with a small amount of noise following a forward diffusion process, and then recover the clean image through a reverse generative process. To evaluate our method against strong adaptive attacks in an efficient and scalable way, we propose to use the adjoint method to compute full gradients of the reverse generative process. Extensive experiments on three image datasets including CIFAR-10, ImageNet and CelebA-HQ with three classifier architectures including ResNet, WideResNet and ViT demonstrate that our method achieves the state-of-the-art results, outperforming current adversarial training and adversarial purification methods, often by a large margin. Project page: https://diffpure.github.io.



## **43. Trustworthy Graph Neural Networks: Aspects, Methods and Trends**

cs.LG

36 pages, 7 tables, 4 figures

**SubmitDate**: 2022-05-16    [paper-pdf](http://arxiv.org/pdf/2205.07424v1)

**Authors**: He Zhang, Bang Wu, Xingliang Yuan, Shirui Pan, Hanghang Tong, Jian Pei

**Abstracts**: Graph neural networks (GNNs) have emerged as a series of competent graph learning methods for diverse real-world scenarios, ranging from daily applications like recommendation systems and question answering to cutting-edge technologies such as drug discovery in life sciences and n-body simulation in astrophysics. However, task performance is not the only requirement for GNNs. Performance-oriented GNNs have exhibited potential adverse effects like vulnerability to adversarial attacks, unexplainable discrimination against disadvantaged groups, or excessive resource consumption in edge computing environments. To avoid these unintentional harms, it is necessary to build competent GNNs characterised by trustworthiness. To this end, we propose a comprehensive roadmap to build trustworthy GNNs from the view of the various computing technologies involved. In this survey, we introduce basic concepts and comprehensively summarise existing efforts for trustworthy GNNs from six aspects, including robustness, explainability, privacy, fairness, accountability, and environmental well-being. Additionally, we highlight the intricate cross-aspect relations between the above six aspects of trustworthy GNNs. Finally, we present a thorough overview of trending directions for facilitating the research and industrialisation of trustworthy GNNs.



## **44. Parameter Adaptation for Joint Distribution Shifts**

cs.LG

**SubmitDate**: 2022-05-15    [paper-pdf](http://arxiv.org/pdf/2205.07315v1)

**Authors**: Siddhartha Datta

**Abstracts**: While different methods exist to tackle distinct types of distribution shift, such as label shift (in the form of adversarial attacks) or domain shift, tackling the joint shift setting is still an open problem. Through the study of a joint distribution shift manifesting both adversarial and domain-specific perturbations, we not only show that a joint shift worsens model performance compared to their individual shifts, but that the use of a similar domain worsens performance than a dissimilar domain. To curb the performance drop, we study the use of perturbation sets motivated by input and parameter space bounds, and adopt a meta learning strategy (hypernetworks) to model parameters w.r.t. test-time inputs to recover performance.



## **45. CE-based white-box adversarial attacks will not work using super-fitting**

cs.LG

**SubmitDate**: 2022-05-15    [paper-pdf](http://arxiv.org/pdf/2205.02741v2)

**Authors**: Youhuan Yang, Lei Sun, Leyu Dai, Song Guo, Xiuqing Mao, Xiaoqin Wang, Bayi Xu

**Abstracts**: Deep neural networks are widely used in various fields because of their powerful performance. However, recent studies have shown that deep learning models are vulnerable to adversarial attacks, i.e., adding a slight perturbation to the input will make the model obtain wrong results. This is especially dangerous for some systems with high-security requirements, so this paper proposes a new defense method by using the model super-fitting state to improve the model's adversarial robustness (i.e., the accuracy under adversarial attacks). This paper mathematically proves the effectiveness of super-fitting and enables the model to reach this state quickly by minimizing unrelated category scores (MUCS). Theoretically, super-fitting can resist any existing (even future) CE-based white-box adversarial attacks. In addition, this paper uses a variety of powerful attack algorithms to evaluate the adversarial robustness of super-fitting, and the proposed method is compared with nearly 50 defense models from recent conferences. The experimental results show that the super-fitting method in this paper can make the trained model obtain the highest adversarial robustness.



## **46. Measuring Vulnerabilities of Malware Detectors with Explainability-Guided Evasion Attacks**

cs.CR

**SubmitDate**: 2022-05-15    [paper-pdf](http://arxiv.org/pdf/2111.10085v3)

**Authors**: Ruoxi Sun, Wei Wang, Tian Dong, Shaofeng Li, Minhui Xue, Gareth Tyson, Haojin Zhu, Mingyu Guo, Surya Nepal

**Abstracts**: Numerous open-source and commercial malware detectors are available. However, their efficacy is threatened by new adversarial attacks, whereby malware attempts to evade detection, e.g., by performing feature-space manipulation. In this work, we propose an explainability-guided and model-agnostic framework for measuring the efficacy of malware detectors when confronted with adversarial attacks. The framework introduces the concept of Accrued Malicious Magnitude (AMM) to identify which malware features should be manipulated to maximize the likelihood of evading detection. We then use this framework to test several state-of-the-art malware detectors' ability to detect manipulated malware. We find that (i) commercial antivirus engines are vulnerable to AMM-guided manipulated samples; (ii) the ability of a manipulated malware generated using one detector to evade detection by another detector (i.e., transferability) depends on the overlap of features with large AMM values between the different detectors; and (iii) AMM values effectively measure the importance of features and explain the ability to evade detection. Our findings shed light on the weaknesses of current malware detectors, as well as how they can be improved.



## **47. Unsupervised Abnormal Traffic Detection through Topological Flow Analysis**

cs.LG

**SubmitDate**: 2022-05-14    [paper-pdf](http://arxiv.org/pdf/2205.07109v1)

**Authors**: Paul Irofti, Andrei Pătraşcu, Andrei Iulian Hîji

**Abstracts**: Cyberthreats are a permanent concern in our modern technological world. In the recent years, sophisticated traffic analysis techniques and anomaly detection (AD) algorithms have been employed to face the more and more subversive adversarial attacks. A malicious intrusion, defined as an invasive action intending to illegally exploit private resources, manifests through unusual data traffic and/or abnormal connectivity pattern. Despite the plethora of statistical or signature-based detectors currently provided in the literature, the topological connectivity component of a malicious flow is less exploited. Furthermore, a great proportion of the existing statistical intrusion detectors are based on supervised learning, that relies on labeled data. By viewing network flows as weighted directed interactions between a pair of nodes, in this paper we present a simple method that facilitate the use of connectivity graph features in unsupervised anomaly detection algorithms. We test our methodology on real network traffic datasets and observe several improvements over standard AD.



## **48. Learning Coated Adversarial Camouflages for Object Detectors**

cs.CV

**SubmitDate**: 2022-05-14    [paper-pdf](http://arxiv.org/pdf/2109.00124v3)

**Authors**: Yexin Duan, Jialin Chen, Xingyu Zhou, Junhua Zou, Zhengyun He, Jin Zhang, Wu Zhang, Zhisong Pan

**Abstracts**: An adversary can fool deep neural network object detectors by generating adversarial noises. Most of the existing works focus on learning local visible noises in an adversarial "patch" fashion. However, the 2D patch attached to a 3D object tends to suffer from an inevitable reduction in attack performance as the viewpoint changes. To remedy this issue, this work proposes the Coated Adversarial Camouflage (CAC) to attack the detectors in arbitrary viewpoints. Unlike the patch trained in the 2D space, our camouflage generated by a conceptually different training framework consists of 3D rendering and dense proposals attack. Specifically, we make the camouflage perform 3D spatial transformations according to the pose changes of the object. Based on the multi-view rendering results, the top-n proposals of the region proposal network are fixed, and all the classifications in the fixed dense proposals are attacked simultaneously to output errors. In addition, we build a virtual 3D scene to fairly and reproducibly evaluate different attacks. Extensive experiments demonstrate the superiority of CAC over the existing attacks, and it shows impressive performance both in the virtual scene and the real world. This poses a potential threat to the security-critical computer vision systems.



## **49. Rethinking Classifier and Adversarial Attack**

cs.LG

**SubmitDate**: 2022-05-14    [paper-pdf](http://arxiv.org/pdf/2205.02743v2)

**Authors**: Youhuan Yang, Lei Sun, Leyu Dai, Song Guo, Xiuqing Mao, Xiaoqin Wang, Bayi Xu

**Abstracts**: Various defense models have been proposed to resist adversarial attack algorithms, but existing adversarial robustness evaluation methods always overestimate the adversarial robustness of these models (i.e., not approaching the lower bound of robustness). To solve this problem, this paper uses the proposed decouple space method to divide the classifier into two parts: non-linear and linear. Then, this paper defines the representation vector of the original example (and its space, i.e., the representation space) and uses the iterative optimization of Absolute Classification Boundaries Initialization (ACBI) to obtain a better attack starting point. Particularly, this paper applies ACBI to nearly 50 widely-used defense models (including 8 architectures). Experimental results show that ACBI achieves lower robust accuracy in all cases.



## **50. Evaluating Membership Inference Through Adversarial Robustness**

cs.CR

Accepted by The Computer Journal. Pre-print version

**SubmitDate**: 2022-05-14    [paper-pdf](http://arxiv.org/pdf/2205.06986v1)

**Authors**: Zhaoxi Zhang, Leo Yu Zhang, Xufei Zheng, Bilal Hussain Abbasi, Shengshan Hu

**Abstracts**: The usage of deep learning is being escalated in many applications. Due to its outstanding performance, it is being used in a variety of security and privacy-sensitive areas in addition to conventional applications. One of the key aspects of deep learning efficacy is to have abundant data. This trait leads to the usage of data which can be highly sensitive and private, which in turn causes wariness with regard to deep learning in the general public. Membership inference attacks are considered lethal as they can be used to figure out whether a piece of data belongs to the training dataset or not. This can be problematic with regards to leakage of training data information and its characteristics. To highlight the significance of these types of attacks, we propose an enhanced methodology for membership inference attacks based on adversarial robustness, by adjusting the directions of adversarial perturbations through label smoothing under a white-box setting. We evaluate our proposed method on three datasets: Fashion-MNIST, CIFAR-10, and CIFAR-100. Our experimental results reveal that the performance of our method surpasses that of the existing adversarial robustness-based method when attacking normally trained models. Additionally, through comparing our technique with the state-of-the-art metric-based membership inference methods, our proposed method also shows better performance when attacking adversarially trained models. The code for reproducing the results of this work is available at \url{https://github.com/plll4zzx/Evaluating-Membership-Inference-Through-Adversarial-Robustness}.



