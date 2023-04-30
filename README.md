# Latest Adversarial Attack Papers
**update at 2023-04-30 12:21:39**

[中英双语版本](https://github.com/daksim/NewAdversarialAttackPaper/blob/main/README_CN.md)

## **1. On the (In)security of Peer-to-Peer Decentralized Machine Learning**

cs.CR

IEEE S&P'23 (Previous title: "On the Privacy of Decentralized Machine  Learning")

**SubmitDate**: 2023-04-27    [abs](http://arxiv.org/abs/2205.08443v2) [paper-pdf](http://arxiv.org/pdf/2205.08443v2)

**Authors**: Dario Pasquini, Mathilde Raynal, Carmela Troncoso

**Abstract**: In this work, we carry out the first, in-depth, privacy analysis of Decentralized Learning -- a collaborative machine learning framework aimed at addressing the main limitations of federated learning. We introduce a suite of novel attacks for both passive and active decentralized adversaries. We demonstrate that, contrary to what is claimed by decentralized learning proposers, decentralized learning does not offer any security advantage over federated learning. Rather, it increases the attack surface enabling any user in the system to perform privacy attacks such as gradient inversion, and even gain full control over honest users' local model. We also show that, given the state of the art in protections, privacy-preserving configurations of decentralized learning require fully connected networks, losing any practical advantage over the federated setup and therefore completely defeating the objective of the decentralized approach.



## **2. Robust Resilient Signal Reconstruction under Adversarial Attacks**

math.OC

7 pages

**SubmitDate**: 2023-04-27    [abs](http://arxiv.org/abs/1807.08004v2) [paper-pdf](http://arxiv.org/pdf/1807.08004v2)

**Authors**: Yu Zheng, Olugbenga Moses Anubi, Lalit Mestha, Hema Achanta

**Abstract**: We consider the problem of signal reconstruction for a system under sparse signal corruption by a malicious agent. The reconstruction problem follows the standard error coding problem that has been studied extensively in the literature. We include a new challenge of robust estimation of the attack support. The problem is then cast as a constrained optimization problem merging promising techniques in the area of deep learning and estimation theory. A pruning algorithm is developed to reduce the ``false positive" uncertainty of data-driven attack localization results, thereby improving the probability of correct signal reconstruction. Sufficient conditions for the correct reconstruction and the associated reconstruction error bounds are obtained for both exact and inexact attack support estimation. Moreover, a simulation of a water distribution system is presented to validate the proposed techniques.



## **3. QEVSEC: Quick Electric Vehicle SEcure Charging via Dynamic Wireless Power Transfer**

cs.CR

6 pages, conference

**SubmitDate**: 2023-04-27    [abs](http://arxiv.org/abs/2205.10292v2) [paper-pdf](http://arxiv.org/pdf/2205.10292v2)

**Authors**: Tommaso Bianchi, Surudhi Asokraj, Alessandro Brighente, Mauro Conti, Radha Poovendran

**Abstract**: Dynamic Wireless Power Transfer (DWPT) can be used for on-demand recharging of Electric Vehicles (EV) while driving. However, DWPT raises numerous security and privacy concerns. Recently, researchers demonstrated that DWPT systems are vulnerable to adversarial attacks. In an EV charging scenario, an attacker can prevent the authorized customer from charging, obtain a free charge by billing a victim user and track a target vehicle. State-of-the-art authentication schemes relying on centralized solutions are either vulnerable to various attacks or have high computational complexity, making them unsuitable for a dynamic scenario. In this paper, we propose Quick Electric Vehicle SEcure Charging (QEVSEC), a novel, secure, and efficient authentication protocol for the dynamic charging of EVs. Our idea for QEVSEC originates from multiple vulnerabilities we found in the state-of-the-art protocol that allows tracking of user activity and is susceptible to replay attacks. Based on these observations, the proposed protocol solves these issues and achieves lower computational complexity by using only primitive cryptographic operations in a very short message exchange. QEVSEC provides scalability and a reduced cost in each iteration, thus lowering the impact on the power needed from the grid.



## **4. Boosting Big Brother: Attacking Search Engines with Encodings**

cs.CR

**SubmitDate**: 2023-04-27    [abs](http://arxiv.org/abs/2304.14031v1) [paper-pdf](http://arxiv.org/pdf/2304.14031v1)

**Authors**: Nicholas Boucher, Luca Pajola, Ilia Shumailov, Ross Anderson, Mauro Conti

**Abstract**: Search engines are vulnerable to attacks against indexing and searching via text encoding manipulation. By imperceptibly perturbing text using uncommon encoded representations, adversaries can control results across search engines for specific search queries. We demonstrate that this attack is successful against two major commercial search engines - Google and Bing - and one open source search engine - Elasticsearch. We further demonstrate that this attack is successful against LLM chat search including Bing's GPT-4 chatbot and Google's Bard chatbot. We also present a variant of the attack targeting text summarization and plagiarism detection models, two ML tasks closely tied to search. We provide a set of defenses against these techniques and warn that adversaries can leverage these attacks to launch disinformation campaigns against unsuspecting users, motivating the need for search engine maintainers to patch deployed systems.



## **5. You Can't Always Check What You Wanted: Selective Checking and Trusted Execution to Prevent False Actuations in Cyber-Physical Systems**

cs.CR

Extended version of SCATE published in ISORC'23

**SubmitDate**: 2023-04-27    [abs](http://arxiv.org/abs/2304.13956v1) [paper-pdf](http://arxiv.org/pdf/2304.13956v1)

**Authors**: Monowar Hasan, Sibin Mohan

**Abstract**: Cyber-physical systems (CPS) are vulnerable to attacks targeting outgoing actuation commands that modify their physical behaviors. The limited resources in such systems, coupled with their stringent timing constraints, often prevents the checking of every outgoing command. We present a "selective checking" mechanism that uses game-theoretic modeling to identify the right subset of commands to be checked in order to deter an adversary. This mechanism is coupled with a "delay-aware" trusted execution environment (TEE) to ensure that only verified actuation commands are ever sent to the physical system, thus maintaining their safety and integrity. The selective checking and trusted execution (SCATE) framework is implemented on an off-the-shelf ARM platform running standard embedded Linux. We demonstrate the effectiveness of SCATE using four realistic cyber-physical systems (a ground rover, a flight controller, a robotic arm and an automated syringe pump) and study design trade-offs. Not only does SCATE provide a high level of security and high performance, it also suffers from significantly lower overheads (30.48%-47.32% less) in the process. In fact, SCATE can work with more systems without negatively affecting the safety of the system. Considering that most CPS do not have any such checking mechanisms, and SCATE is guaranteed to meet all the timing requirements (i.e., ensure the safety/integrity of the system), our methods can significantly improve the security (and, hence, safety) of the system.



## **6. Detection of Adversarial Physical Attacks in Time-Series Image Data**

cs.CV

**SubmitDate**: 2023-04-27    [abs](http://arxiv.org/abs/2304.13919v1) [paper-pdf](http://arxiv.org/pdf/2304.13919v1)

**Authors**: Ramneet Kaur, Yiannis Kantaros, Wenwen Si, James Weimer, Insup Lee

**Abstract**: Deep neural networks (DNN) have become a common sensing modality in autonomous systems as they allow for semantically perceiving the ambient environment given input images. Nevertheless, DNN models have proven to be vulnerable to adversarial digital and physical attacks. To mitigate this issue, several detection frameworks have been proposed to detect whether a single input image has been manipulated by adversarial digital noise or not. In our prior work, we proposed a real-time detector, called VisionGuard (VG), for adversarial physical attacks against single input images to DNN models. Building upon that work, we propose VisionGuard* (VG), which couples VG with majority-vote methods, to detect adversarial physical attacks in time-series image data, e.g., videos. This is motivated by autonomous systems applications where images are collected over time using onboard sensors for decision-making purposes. We emphasize that majority-vote mechanisms are quite common in autonomous system applications (among many other applications), as e.g., in autonomous driving stacks for object detection. In this paper, we investigate, both theoretically and experimentally, how this widely used mechanism can be leveraged to enhance the performance of adversarial detectors. We have evaluated VG* on videos of both clean and physically attacked traffic signs generated by a state-of-the-art robust physical attack. We provide extensive comparative experiments against detectors that have been designed originally for out-of-distribution data and digitally attacked images.



## **7. Learning Robust Deep Equilibrium Models**

cs.LG

**SubmitDate**: 2023-04-26    [abs](http://arxiv.org/abs/2304.12707v2) [paper-pdf](http://arxiv.org/pdf/2304.12707v2)

**Authors**: Haoyu Chu, Shikui Wei, Ting Liu, Yao Zhao

**Abstract**: Deep equilibrium (DEQ) models have emerged as a promising class of implicit layer models in deep learning, which abandon traditional depth by solving for the fixed points of a single nonlinear layer. Despite their success, the stability of the fixed points for these models remains poorly understood. Recently, Lyapunov theory has been applied to Neural ODEs, another type of implicit layer model, to confer adversarial robustness. By considering DEQ models as nonlinear dynamic systems, we propose a robust DEQ model named LyaDEQ with guaranteed provable stability via Lyapunov theory. The crux of our method is ensuring the fixed points of the DEQ models are Lyapunov stable, which enables the LyaDEQ models to resist minor initial perturbations. To avoid poor adversarial defense due to Lyapunov-stable fixed points being located near each other, we add an orthogonal fully connected layer after the Lyapunov stability module to separate different fixed points. We evaluate LyaDEQ models on several widely used datasets under well-known adversarial attacks, and experimental results demonstrate significant improvement in robustness. Furthermore, we show that the LyaDEQ model can be combined with other defense methods, such as adversarial training, to achieve even better adversarial robustness.



## **8. One-vs-the-Rest Loss to Focus on Important Samples in Adversarial Training**

cs.LG

ICML2023, 26 pages, 19 figures

**SubmitDate**: 2023-04-26    [abs](http://arxiv.org/abs/2207.10283v3) [paper-pdf](http://arxiv.org/pdf/2207.10283v3)

**Authors**: Sekitoshi Kanai, Shin'ya Yamaguchi, Masanori Yamada, Hiroshi Takahashi, Kentaro Ohno, Yasutoshi Ida

**Abstract**: This paper proposes a new loss function for adversarial training. Since adversarial training has difficulties, e.g., necessity of high model capacity, focusing on important data points by weighting cross-entropy loss has attracted much attention. However, they are vulnerable to sophisticated attacks, e.g., Auto-Attack. This paper experimentally reveals that the cause of their vulnerability is their small margins between logits for the true label and the other labels. Since neural networks classify the data points based on the logits, logit margins should be large enough to avoid flipping the largest logit by the attacks. Importance-aware methods do not increase logit margins of important samples but decrease those of less-important samples compared with cross-entropy loss. To increase logit margins of important samples, we propose switching one-vs-the-rest loss (SOVR), which switches from cross-entropy to one-vs-the-rest loss for important samples that have small logit margins. We prove that one-vs-the-rest loss increases logit margins two times larger than the weighted cross-entropy loss for a simple problem. We experimentally confirm that SOVR increases logit margins of important samples unlike existing methods and achieves better robustness against Auto-Attack than importance-aware methods.



## **9. Improving Adversarial Transferability by Intermediate-level Perturbation Decay**

cs.LG

Revision of ICML '23 submission for better clarity

**SubmitDate**: 2023-04-26    [abs](http://arxiv.org/abs/2304.13410v1) [paper-pdf](http://arxiv.org/pdf/2304.13410v1)

**Authors**: Qizhang Li, Yiwen Guo, Wangmeng Zuo, Hao Chen

**Abstract**: Intermediate-level attacks that attempt to perturb feature representations following an adversarial direction drastically have shown favorable performance in crafting transferable adversarial examples. Existing methods in this category are normally formulated with two separate stages, where a directional guide is required to be determined at first and the scalar projection of the intermediate-level perturbation onto the directional guide is enlarged thereafter. The obtained perturbation deviates from the guide inevitably in the feature space, and it is revealed in this paper that such a deviation may lead to sub-optimal attack. To address this issue, we develop a novel intermediate-level method that crafts adversarial examples within a single stage of optimization. In particular, the proposed method, named intermediate-level perturbation decay (ILPD), encourages the intermediate-level perturbation to be in an effective adversarial direction and to possess a great magnitude simultaneously. In-depth discussion verifies the effectiveness of our method. Experimental results show that it outperforms state-of-the-arts by large margins in attacking various victim models on ImageNet (+10.07% on average) and CIFAR-10 (+3.88% on average). Our code is at https://github.com/qizhangli/ILPD-attack.



## **10. Blockchain-based Access Control for Secure Smart Industry Management Systems**

cs.CR

**SubmitDate**: 2023-04-26    [abs](http://arxiv.org/abs/2304.13379v1) [paper-pdf](http://arxiv.org/pdf/2304.13379v1)

**Authors**: Aditya Pribadi Kalapaaking, Ibrahim Khalil, Mohammad Saidur Rahman, Abdelaziz Bouras

**Abstract**: Smart manufacturing systems involve a large number of interconnected devices resulting in massive data generation. Cloud computing technology has recently gained increasing attention in smart manufacturing systems for facilitating cost-effective service provisioning and massive data management. In a cloud-based manufacturing system, ensuring authorized access to the data is crucial. A cloud platform is operated under a single authority. Hence, a cloud platform is prone to a single point of failure and vulnerable to adversaries. An internal or external adversary can easily modify users' access to allow unauthorized users to access the data. This paper proposes a role-based access control to prevent modification attacks by leveraging blockchain and smart contracts in a cloud-based smart manufacturing system. The role-based access control is developed to determine users' roles and rights in smart contracts. The smart contracts are then deployed to the private blockchain network. We evaluate our solution by utilizing Ethereum private blockchain network to deploy the smart contract. The experimental results demonstrate the feasibility and evaluation of the proposed framework's performance.



## **11. Blockchain-based Federated Learning with SMPC Model Verification Against Poisoning Attack for Healthcare Systems**

cs.CR

**SubmitDate**: 2023-04-26    [abs](http://arxiv.org/abs/2304.13360v1) [paper-pdf](http://arxiv.org/pdf/2304.13360v1)

**Authors**: Aditya Pribadi Kalapaaking, Ibrahim Khalil, Xun Yi

**Abstract**: Due to the rising awareness of privacy and security in machine learning applications, federated learning (FL) has received widespread attention and applied to several areas, e.g., intelligence healthcare systems, IoT-based industries, and smart cities. FL enables clients to train a global model collaboratively without accessing their local training data. However, the current FL schemes are vulnerable to adversarial attacks. Its architecture makes detecting and defending against malicious model updates difficult. In addition, most recent studies to detect FL from malicious updates while maintaining the model's privacy have not been sufficiently explored. This paper proposed blockchain-based federated learning with SMPC model verification against poisoning attacks for healthcare systems. First, we check the machine learning model from the FL participants through an encrypted inference process and remove the compromised model. Once the participants' local models have been verified, the models are sent to the blockchain node to be securely aggregated. We conducted several experiments with different medical datasets to evaluate our proposed framework.



## **12. On the Risks of Stealing the Decoding Algorithms of Language Models**

cs.LG

**SubmitDate**: 2023-04-26    [abs](http://arxiv.org/abs/2303.04729v3) [paper-pdf](http://arxiv.org/pdf/2303.04729v3)

**Authors**: Ali Naseh, Kalpesh Krishna, Mohit Iyyer, Amir Houmansadr

**Abstract**: A key component of generating text from modern language models (LM) is the selection and tuning of decoding algorithms. These algorithms determine how to generate text from the internal probability distribution generated by the LM. The process of choosing a decoding algorithm and tuning its hyperparameters takes significant time, manual effort, and computation, and it also requires extensive human evaluation. Therefore, the identity and hyperparameters of such decoding algorithms are considered to be extremely valuable to their owners. In this work, we show, for the first time, that an adversary with typical API access to an LM can steal the type and hyperparameters of its decoding algorithms at very low monetary costs. Our attack is effective against popular LMs used in text generation APIs, including GPT-2 and GPT-3. We demonstrate the feasibility of stealing such information with only a few dollars, e.g., $\$0.8$, $\$1$, $\$4$, and $\$40$ for the four versions of GPT-3.



## **13. SHIELD: Thwarting Code Authorship Attribution**

cs.CR

12 pages, 13 figures

**SubmitDate**: 2023-04-26    [abs](http://arxiv.org/abs/2304.13255v1) [paper-pdf](http://arxiv.org/pdf/2304.13255v1)

**Authors**: Mohammed Abuhamad, Changhun Jung, David Mohaisen, DaeHun Nyang

**Abstract**: Authorship attribution has become increasingly accurate, posing a serious privacy risk for programmers who wish to remain anonymous. In this paper, we introduce SHIELD to examine the robustness of different code authorship attribution approaches against adversarial code examples. We define four attacks on attribution techniques, which include targeted and non-targeted attacks, and realize them using adversarial code perturbation. We experiment with a dataset of 200 programmers from the Google Code Jam competition to validate our methods targeting six state-of-the-art authorship attribution methods that adopt a variety of techniques for extracting authorship traits from source-code, including RNN, CNN, and code stylometry. Our experiments demonstrate the vulnerability of current authorship attribution methods against adversarial attacks. For the non-targeted attack, our experiments demonstrate the vulnerability of current authorship attribution methods against the attack with an attack success rate exceeds 98.5\% accompanied by a degradation of the identification confidence that exceeds 13\%. For the targeted attacks, we show the possibility of impersonating a programmer using targeted-adversarial perturbations with a success rate ranging from 66\% to 88\% for different authorship attribution techniques under several adversarial scenarios.



## **14. Generating Adversarial Examples with Task Oriented Multi-Objective Optimization**

cs.LG

**SubmitDate**: 2023-04-26    [abs](http://arxiv.org/abs/2304.13229v1) [paper-pdf](http://arxiv.org/pdf/2304.13229v1)

**Authors**: Anh Bui, Trung Le, He Zhao, Quan Tran, Paul Montague, Dinh Phung

**Abstract**: Deep learning models, even the-state-of-the-art ones, are highly vulnerable to adversarial examples. Adversarial training is one of the most efficient methods to improve the model's robustness. The key factor for the success of adversarial training is the capability to generate qualified and divergent adversarial examples which satisfy some objectives/goals (e.g., finding adversarial examples that maximize the model losses for simultaneously attacking multiple models). Therefore, multi-objective optimization (MOO) is a natural tool for adversarial example generation to achieve multiple objectives/goals simultaneously. However, we observe that a naive application of MOO tends to maximize all objectives/goals equally, without caring if an objective/goal has been achieved yet. This leads to useless effort to further improve the goal-achieved tasks, while putting less focus on the goal-unachieved tasks. In this paper, we propose \emph{Task Oriented MOO} to address this issue, in the context where we can explicitly define the goal achievement for a task. Our principle is to only maintain the goal-achieved tasks, while letting the optimizer spend more effort on improving the goal-unachieved tasks. We conduct comprehensive experiments for our Task Oriented MOO on various adversarial example generation schemes. The experimental results firmly demonstrate the merit of our proposed approach. Our code is available at \url{https://github.com/tuananhbui89/TAMOO}.



## **15. Uncovering the Representation of Spiking Neural Networks Trained with Surrogate Gradient**

cs.LG

Published in Transactions on Machine Learning Research (TMLR)

**SubmitDate**: 2023-04-25    [abs](http://arxiv.org/abs/2304.13098v1) [paper-pdf](http://arxiv.org/pdf/2304.13098v1)

**Authors**: Yuhang Li, Youngeun Kim, Hyoungseob Park, Priyadarshini Panda

**Abstract**: Spiking Neural Networks (SNNs) are recognized as the candidate for the next-generation neural networks due to their bio-plausibility and energy efficiency. Recently, researchers have demonstrated that SNNs are able to achieve nearly state-of-the-art performance in image recognition tasks using surrogate gradient training. However, some essential questions exist pertaining to SNNs that are little studied: Do SNNs trained with surrogate gradient learn different representations from traditional Artificial Neural Networks (ANNs)? Does the time dimension in SNNs provide unique representation power? In this paper, we aim to answer these questions by conducting a representation similarity analysis between SNNs and ANNs using Centered Kernel Alignment (CKA). We start by analyzing the spatial dimension of the networks, including both the width and the depth. Furthermore, our analysis of residual connections shows that SNNs learn a periodic pattern, which rectifies the representations in SNNs to be ANN-like. We additionally investigate the effect of the time dimension on SNN representation, finding that deeper layers encourage more dynamics along the time dimension. We also investigate the impact of input data such as event-stream data and adversarial attacks. Our work uncovers a host of new findings of representations in SNNs. We hope this work will inspire future research to fully comprehend the representation power of SNNs. Code is released at https://github.com/Intelligent-Computing-Lab-Yale/SNNCKA.



## **16. Improving Robustness Against Adversarial Attacks with Deeply Quantized Neural Networks**

cs.LG

Accepted at IJCNN 2023. 8 pages, 5 figures

**SubmitDate**: 2023-04-25    [abs](http://arxiv.org/abs/2304.12829v1) [paper-pdf](http://arxiv.org/pdf/2304.12829v1)

**Authors**: Ferheen Ayaz, Idris Zakariyya, José Cano, Sye Loong Keoh, Jeremy Singer, Danilo Pau, Mounia Kharbouche-Harrari

**Abstract**: Reducing the memory footprint of Machine Learning (ML) models, particularly Deep Neural Networks (DNNs), is essential to enable their deployment into resource-constrained tiny devices. However, a disadvantage of DNN models is their vulnerability to adversarial attacks, as they can be fooled by adding slight perturbations to the inputs. Therefore, the challenge is how to create accurate, robust, and tiny DNN models deployable on resource-constrained embedded devices. This paper reports the results of devising a tiny DNN model, robust to adversarial black and white box attacks, trained with an automatic quantizationaware training framework, i.e. QKeras, with deep quantization loss accounted in the learning loop, thereby making the designed DNNs more accurate for deployment on tiny devices. We investigated how QKeras and an adversarial robustness technique, Jacobian Regularization (JR), can provide a co-optimization strategy by exploiting the DNN topology and the per layer JR approach to produce robust yet tiny deeply quantized DNN models. As a result, a new DNN model implementing this cooptimization strategy was conceived, developed and tested on three datasets containing both images and audio inputs, as well as compared its performance with existing benchmarks against various white-box and black-box attacks. Experimental results demonstrated that on average our proposed DNN model resulted in 8.3% and 79.5% higher accuracy than MLCommons/Tiny benchmarks in the presence of white-box and black-box attacks on the CIFAR-10 image dataset and a subset of the Google Speech Commands audio dataset respectively. It was also 6.5% more accurate for black-box attacks on the SVHN image dataset.



## **17. RobCaps: Evaluating the Robustness of Capsule Networks against Affine Transformations and Adversarial Attacks**

cs.LG

To appear at the 2023 International Joint Conference on Neural  Networks (IJCNN), Queensland, Australia, June 2023

**SubmitDate**: 2023-04-25    [abs](http://arxiv.org/abs/2304.03973v2) [paper-pdf](http://arxiv.org/pdf/2304.03973v2)

**Authors**: Alberto Marchisio, Antonio De Marco, Alessio Colucci, Maurizio Martina, Muhammad Shafique

**Abstract**: Capsule Networks (CapsNets) are able to hierarchically preserve the pose relationships between multiple objects for image classification tasks. Other than achieving high accuracy, another relevant factor in deploying CapsNets in safety-critical applications is the robustness against input transformations and malicious adversarial attacks.   In this paper, we systematically analyze and evaluate different factors affecting the robustness of CapsNets, compared to traditional Convolutional Neural Networks (CNNs). Towards a comprehensive comparison, we test two CapsNet models and two CNN models on the MNIST, GTSRB, and CIFAR10 datasets, as well as on the affine-transformed versions of such datasets. With a thorough analysis, we show which properties of these architectures better contribute to increasing the robustness and their limitations. Overall, CapsNets achieve better robustness against adversarial examples and affine transformations, compared to a traditional CNN with a similar number of parameters. Similar conclusions have been derived for deeper versions of CapsNets and CNNs. Moreover, our results unleash a key finding that the dynamic routing does not contribute much to improving the CapsNets' robustness. Indeed, the main generalization contribution is due to the hierarchical feature learning through capsules.



## **18. Evaluating Adversarial Robustness on Document Image Classification**

cs.CV

The 17th International Conference on Document Analysis and  Recognition

**SubmitDate**: 2023-04-24    [abs](http://arxiv.org/abs/2304.12486v1) [paper-pdf](http://arxiv.org/pdf/2304.12486v1)

**Authors**: Timothée Fronteau, Arnaud Paran, Aymen Shabou

**Abstract**: Adversarial attacks and defenses have gained increasing interest on computer vision systems in recent years, but as of today, most investigations are limited to images. However, many artificial intelligence models actually handle documentary data, which is very different from real world images. Hence, in this work, we try to apply the adversarial attack philosophy on documentary and natural data and to protect models against such attacks. We focus our work on untargeted gradient-based, transfer-based and score-based attacks and evaluate the impact of adversarial training, JPEG input compression and grey-scale input transformation on the robustness of ResNet50 and EfficientNetB0 model architectures. To the best of our knowledge, no such work has been conducted by the community in order to study the impact of these attacks on the document image classification task.



## **19. StratDef: Strategic Defense Against Adversarial Attacks in ML-based Malware Detection**

cs.LG

**SubmitDate**: 2023-04-24    [abs](http://arxiv.org/abs/2202.07568v6) [paper-pdf](http://arxiv.org/pdf/2202.07568v6)

**Authors**: Aqib Rashid, Jose Such

**Abstract**: Over the years, most research towards defenses against adversarial attacks on machine learning models has been in the image recognition domain. The ML-based malware detection domain has received less attention despite its importance. Moreover, most work exploring these defenses has focused on several methods but with no strategy when applying them. In this paper, we introduce StratDef, which is a strategic defense system based on a moving target defense approach. We overcome challenges related to the systematic construction, selection, and strategic use of models to maximize adversarial robustness. StratDef dynamically and strategically chooses the best models to increase the uncertainty for the attacker while minimizing critical aspects in the adversarial ML domain, like attack transferability. We provide the first comprehensive evaluation of defenses against adversarial attacks on machine learning for malware detection, where our threat model explores different levels of threat, attacker knowledge, capabilities, and attack intensities. We show that StratDef performs better than other defenses even when facing the peak adversarial threat. We also show that, of the existing defenses, only a few adversarially-trained models provide substantially better protection than just using vanilla models but are still outperformed by StratDef.



## **20. On Adversarial Robustness of Point Cloud Semantic Segmentation**

cs.CV

**SubmitDate**: 2023-04-23    [abs](http://arxiv.org/abs/2112.05871v4) [paper-pdf](http://arxiv.org/pdf/2112.05871v4)

**Authors**: Jiacen Xu, Zhe Zhou, Boyuan Feng, Yufei Ding, Zhou Li

**Abstract**: Recent research efforts on 3D point cloud semantic segmentation (PCSS) have achieved outstanding performance by adopting neural networks. However, the robustness of these complex models have not been systematically analyzed. Given that PCSS has been applied in many safety-critical applications like autonomous driving, it is important to fill this knowledge gap, especially, how these models are affected under adversarial samples. As such, we present a comparative study of PCSS robustness. First, we formally define the attacker's objective under performance degradation and object hiding. Then, we develop new attack by whether to bound the norm. We evaluate different attack options on two datasets and three PCSS models. We found all the models are vulnerable and attacking point color is more effective. With this study, we call the attention of the research community to develop new approaches to harden PCSS models.



## **21. Evading DeepFake Detectors via Adversarial Statistical Consistency**

cs.CV

Accepted by CVPR 2023

**SubmitDate**: 2023-04-23    [abs](http://arxiv.org/abs/2304.11670v1) [paper-pdf](http://arxiv.org/pdf/2304.11670v1)

**Authors**: Yang Hou, Qing Guo, Yihao Huang, Xiaofei Xie, Lei Ma, Jianjun Zhao

**Abstract**: In recent years, as various realistic face forgery techniques known as DeepFake improves by leaps and bounds,more and more DeepFake detection techniques have been proposed. These methods typically rely on detecting statistical differences between natural (i.e., real) and DeepFakegenerated images in both spatial and frequency domains. In this work, we propose to explicitly minimize the statistical differences to evade state-of-the-art DeepFake detectors. To this end, we propose a statistical consistency attack (StatAttack) against DeepFake detectors, which contains two main parts. First, we select several statistical-sensitive natural degradations (i.e., exposure, blur, and noise) and add them to the fake images in an adversarial way. Second, we find that the statistical differences between natural and DeepFake images are positively associated with the distribution shifting between the two kinds of images, and we propose to use a distribution-aware loss to guide the optimization of different degradations. As a result, the feature distributions of generated adversarial examples is close to the natural images.Furthermore, we extend the StatAttack to a more powerful version, MStatAttack, where we extend the single-layer degradation to multi-layer degradations sequentially and use the loss to tune the combination weights jointly. Comprehensive experimental results on four spatial-based detectors and two frequency-based detectors with four datasets demonstrate the effectiveness of our proposed attack method in both white-box and black-box settings.



## **22. Partial-Information, Longitudinal Cyber Attacks on LiDAR in Autonomous Vehicles**

cs.CR

**SubmitDate**: 2023-04-23    [abs](http://arxiv.org/abs/2303.03470v2) [paper-pdf](http://arxiv.org/pdf/2303.03470v2)

**Authors**: R. Spencer Hallyburton, Qingzhao Zhang, Z. Morley Mao, Miroslav Pajic

**Abstract**: What happens to an autonomous vehicle (AV) if its data are adversarially compromised? Prior security studies have addressed this question through mostly unrealistic threat models, with limited practical relevance, such as white-box adversarial learning or nanometer-scale laser aiming and spoofing. With growing evidence that cyber threats pose real, imminent danger to AVs and cyber-physical systems (CPS) in general, we present and evaluate a novel AV threat model: a cyber-level attacker capable of disrupting sensor data but lacking any situational awareness. We demonstrate that even though the attacker has minimal knowledge and only access to raw data from a single sensor (i.e., LiDAR), she can design several attacks that critically compromise perception and tracking in multi-sensor AVs. To mitigate vulnerabilities and advance secure architectures in AVs, we introduce two improvements for security-aware fusion: a probabilistic data-asymmetry monitor and a scalable track-to-track fusion of 3D LiDAR and monocular detections (T2T-3DLM); we demonstrate that the approaches significantly reduce attack effectiveness. To support objective safety and security evaluations in AVs, we release our security evaluation platform, AVsec, which is built on security-relevant metrics to benchmark AVs on gold-standard longitudinal AV datasets and AV simulators.



## **23. Disco Intelligent Reflecting Surfaces: Active Channel Aging for Fully-Passive Jamming Attacks**

eess.SP

**SubmitDate**: 2023-04-23    [abs](http://arxiv.org/abs/2302.00415v2) [paper-pdf](http://arxiv.org/pdf/2302.00415v2)

**Authors**: Huan Huang, Ying Zhang, Hongliang Zhang, Yi Cai, A. Lee Swindlehurst, Zhu Han

**Abstract**: Due to the open communications environment in wireless channels, wireless networks are vulnerable to jamming attacks. However, existing approaches for jamming rely on knowledge of the legitimate users' (LUs') channels, extra jamming power, or both. To raise concerns about the potential threats posed by illegitimate intelligent reflecting surfaces (IRSs), we propose an alternative method to launch jamming attacks on LUs without either LU channel state information (CSI) or jamming power. The proposed approach employs an adversarial IRS with random phase shifts, referred to as a "disco" IRS (DIRS), that acts like a "disco ball" to actively age the LUs' channels. Such active channel aging (ACA) interference can be used to launch jamming attacks on multi-user multiple-input single-output (MU-MISO) systems. The proposed DIRS-based fully-passive jammer (FPJ) can jam LUs with no additional jamming power or knowledge of the LU CSI, and it can not be mitigated by classical anti-jamming approaches. A theoretical analysis of the proposed DIRS-based FPJ that provides an evaluation of the DIRS-based jamming attacks is derived. Based on this detailed theoretical analysis, some unique properties of the proposed DIRS-based FPJ can be obtained. Furthermore, a design example of the proposed DIRS-based FPJ based on one-bit quantization of the IRS phases is demonstrated to be sufficient for implementing the jamming attack. In addition, numerical results are provided to show the effectiveness of the derived theoretical analysis and the jamming impact of the proposed DIRS-based FPJ.



## **24. StyLess: Boosting the Transferability of Adversarial Examples**

cs.CV

CVPR 2023

**SubmitDate**: 2023-04-23    [abs](http://arxiv.org/abs/2304.11579v1) [paper-pdf](http://arxiv.org/pdf/2304.11579v1)

**Authors**: Kaisheng Liang, Bin Xiao

**Abstract**: Adversarial attacks can mislead deep neural networks (DNNs) by adding imperceptible perturbations to benign examples. The attack transferability enables adversarial examples to attack black-box DNNs with unknown architectures or parameters, which poses threats to many real-world applications. We find that existing transferable attacks do not distinguish between style and content features during optimization, limiting their attack transferability. To improve attack transferability, we propose a novel attack method called style-less perturbation (StyLess). Specifically, instead of using a vanilla network as the surrogate model, we advocate using stylized networks, which encode different style features by perturbing an adaptive instance normalization. Our method can prevent adversarial examples from using non-robust style features and help generate transferable perturbations. Comprehensive experiments show that our method can significantly improve the transferability of adversarial examples. Furthermore, our approach is generic and can outperform state-of-the-art transferable attacks when combined with other attack techniques.



## **25. QuMoS: A Framework for Preserving Security of Quantum Machine Learning Model**

quant-ph

**SubmitDate**: 2023-04-23    [abs](http://arxiv.org/abs/2304.11511v1) [paper-pdf](http://arxiv.org/pdf/2304.11511v1)

**Authors**: Zhepeng Wang, Jinyang Li, Zhirui Hu, Blake Gage, Elizabeth Iwasawa, Weiwen Jiang

**Abstract**: Security has always been a critical issue in machine learning (ML) applications. Due to the high cost of model training -- such as collecting relevant samples, labeling data, and consuming computing power -- model-stealing attack is one of the most fundamental but vitally important issues. When it comes to quantum computing, such a quantum machine learning (QML) model-stealing attack also exists and it is even more severe because the traditional encryption method can hardly be directly applied to quantum computation. On the other hand, due to the limited quantum computing resources, the monetary cost of training QML model can be even higher than classical ones in the near term. Therefore, a well-tuned QML model developed by a company can be delegated to a quantum cloud provider as a service to be used by ordinary users. In this case, the QML model will be leaked if the cloud provider is under attack. To address such a problem, we propose a novel framework, namely QuMoS, to preserve model security. Instead of applying encryption algorithms, we propose to distribute the QML model to multiple physically isolated quantum cloud providers. As such, even if the adversary in one provider can obtain a partial model, the information of the full model is maintained in the QML service company. Although promising, we observed an arbitrary model design under distributed settings cannot provide model security. We further developed a reinforcement learning-based security engine, which can automatically optimize the model design under the distributed setting, such that a good trade-off between model performance and security can be made. Experimental results on four datasets show that the model design proposed by QuMoS can achieve a close accuracy to the model designed with neural architecture search under centralized settings while providing the highest security than the baselines.



## **26. PatchCensor: Patch Robustness Certification for Transformers via Exhaustive Testing**

cs.CV

This paper has been accepted by ACM Transactions on Software  Engineering and Methodology (TOSEM'23) in "Continuous Special Section: AI and  SE." Please include TOSEM for any citations

**SubmitDate**: 2023-04-22    [abs](http://arxiv.org/abs/2111.10481v3) [paper-pdf](http://arxiv.org/pdf/2111.10481v3)

**Authors**: Yuheng Huang, Lei Ma, Yuanchun Li

**Abstract**: Vision Transformer (ViT) is known to be highly nonlinear like other classical neural networks and could be easily fooled by both natural and adversarial patch perturbations. This limitation could pose a threat to the deployment of ViT in the real industrial environment, especially in safety-critical scenarios. In this work, we propose PatchCensor, aiming to certify the patch robustness of ViT by applying exhaustive testing. We try to provide a provable guarantee by considering the worst patch attack scenarios. Unlike empirical defenses against adversarial patches that may be adaptively breached, certified robust approaches can provide a certified accuracy against arbitrary attacks under certain conditions. However, existing robustness certifications are mostly based on robust training, which often requires substantial training efforts and the sacrifice of model performance on normal samples. To bridge the gap, PatchCensor seeks to improve the robustness of the whole system by detecting abnormal inputs instead of training a robust model and asking it to give reliable results for every input, which may inevitably compromise accuracy. Specifically, each input is tested by voting over multiple inferences with different mutated attention masks, where at least one inference is guaranteed to exclude the abnormal patch. This can be seen as complete-coverage testing, which could provide a statistical guarantee on inference at the test time. Our comprehensive evaluation demonstrates that PatchCensor is able to achieve high certified accuracy (e.g. 67.1% on ImageNet for 2%-pixel adversarial patches), significantly outperforming state-of-the-art techniques while achieving similar clean accuracy (81.8% on ImageNet). Meanwhile, our technique also supports flexible configurations to handle different adversarial patch sizes (up to 25%) by simply changing the masking strategy.



## **27. Universal Adversarial Backdoor Attacks to Fool Vertical Federated Learning in Cloud-Edge Collaboration**

cs.LG

14 pages, 7 figures

**SubmitDate**: 2023-04-22    [abs](http://arxiv.org/abs/2304.11432v1) [paper-pdf](http://arxiv.org/pdf/2304.11432v1)

**Authors**: Peng Chen, Xin Du, Zhihui Lu, Hongfeng Chai

**Abstract**: Vertical federated learning (VFL) is a cloud-edge collaboration paradigm that enables edge nodes, comprising resource-constrained Internet of Things (IoT) devices, to cooperatively train artificial intelligence (AI) models while retaining their data locally. This paradigm facilitates improved privacy and security for edges and IoT devices, making VFL an essential component of Artificial Intelligence of Things (AIoT) systems. Nevertheless, the partitioned structure of VFL can be exploited by adversaries to inject a backdoor, enabling them to manipulate the VFL predictions. In this paper, we aim to investigate the vulnerability of VFL in the context of binary classification tasks. To this end, we define a threat model for backdoor attacks in VFL and introduce a universal adversarial backdoor (UAB) attack to poison the predictions of VFL. The UAB attack, consisting of universal trigger generation and clean-label backdoor injection, is incorporated during the VFL training at specific iterations. This is achieved by alternately optimizing the universal trigger and model parameters of VFL sub-problems. Our work distinguishes itself from existing studies on designing backdoor attacks for VFL, as those require the knowledge of auxiliary information not accessible within the split VFL architecture. In contrast, our approach does not necessitate any additional data to execute the attack. On the LendingClub and Zhongyuan datasets, our approach surpasses existing state-of-the-art methods, achieving up to 100\% backdoor task performance while maintaining the main task performance. Our results in this paper make a major advance to revealing the hidden backdoor risks of VFL, hence paving the way for the future development of secure AIoT.



## **28. Detecting Adversarial Faces Using Only Real Face Self-Perturbations**

cs.CV

IJCAI2023

**SubmitDate**: 2023-04-22    [abs](http://arxiv.org/abs/2304.11359v1) [paper-pdf](http://arxiv.org/pdf/2304.11359v1)

**Authors**: Qian Wang, Yongqin Xian, Hefei Ling, Jinyuan Zhang, Xiaorui Lin, Ping Li, Jiazhong Chen, Ning Yu

**Abstract**: Adversarial attacks aim to disturb the functionality of a target system by adding specific noise to the input samples, bringing potential threats to security and robustness when applied to facial recognition systems. Although existing defense techniques achieve high accuracy in detecting some specific adversarial faces (adv-faces), new attack methods especially GAN-based attacks with completely different noise patterns circumvent them and reach a higher attack success rate. Even worse, existing techniques require attack data before implementing the defense, making it impractical to defend newly emerging attacks that are unseen to defenders. In this paper, we investigate the intrinsic generality of adv-faces and propose to generate pseudo adv-faces by perturbing real faces with three heuristically designed noise patterns. We are the first to train an adv-face detector using only real faces and their self-perturbations, agnostic to victim facial recognition systems, and agnostic to unseen attacks. By regarding adv-faces as out-of-distribution data, we then naturally introduce a novel cascaded system for adv-face detection, which consists of training data self-perturbations, decision boundary regularization, and a max-pooling-based binary classifier focusing on abnormal local color aberrations. Experiments conducted on LFW and CelebA-HQ datasets with eight gradient-based and two GAN-based attacks validate that our method generalizes to a variety of unseen adversarial attacks.



## **29. MAWSEO: Adversarial Wiki Search Poisoning for Illicit Online Promotion**

cs.CR

**SubmitDate**: 2023-04-22    [abs](http://arxiv.org/abs/2304.11300v1) [paper-pdf](http://arxiv.org/pdf/2304.11300v1)

**Authors**: Zilong Lin, Zhengyi Li, Xiaojing Liao, XiaoFeng Wang, Xiaozhong Liu

**Abstract**: As a prominent instance of vandalism edits, Wiki search poisoning for illicit promotion is a cybercrime in which the adversary aims at editing Wiki articles to promote illicit businesses through Wiki search results of relevant queries. In this paper, we report a study that, for the first time, shows that such stealthy blackhat SEO on Wiki can be automated. Our technique, called MAWSEO, employs adversarial revisions to achieve real-world cybercriminal objectives, including rank boosting, vandalism detection evasion, topic relevancy, semantic consistency, user awareness (but not alarming) of promotional content, etc. Our evaluation and user study demonstrate that MAWSEO is able to effectively and efficiently generate adversarial vandalism edits, which can bypass state-of-the-art built-in Wiki vandalism detectors, and also get promotional content through to Wiki users without triggering their alarms. In addition, we investigated potential defense, including coherence based detection and adversarial training of vandalism detection, against our attack in the Wiki ecosystem.



## **30. Individual Fairness in Bayesian Neural Networks**

cs.LG

**SubmitDate**: 2023-04-21    [abs](http://arxiv.org/abs/2304.10828v1) [paper-pdf](http://arxiv.org/pdf/2304.10828v1)

**Authors**: Alice Doherty, Matthew Wicker, Luca Laurenti, Andrea Patane

**Abstract**: We study Individual Fairness (IF) for Bayesian neural networks (BNNs). Specifically, we consider the $\epsilon$-$\delta$-individual fairness notion, which requires that, for any pair of input points that are $\epsilon$-similar according to a given similarity metrics, the output of the BNN is within a given tolerance $\delta>0.$ We leverage bounds on statistical sampling over the input space and the relationship between adversarial robustness and individual fairness to derive a framework for the systematic estimation of $\epsilon$-$\delta$-IF, designing Fair-FGSM and Fair-PGD as global,fairness-aware extensions to gradient-based attacks for BNNs. We empirically study IF of a variety of approximately inferred BNNs with different architectures on fairness benchmarks, and compare against deterministic models learnt using frequentist techniques. Interestingly, we find that BNNs trained by means of approximate Bayesian inference consistently tend to be markedly more individually fair than their deterministic counterparts.



## **31. Reliable Representations Make A Stronger Defender: Unsupervised Structure Refinement for Robust GNN**

cs.LG

Accepted in KDD2022

**SubmitDate**: 2023-04-21    [abs](http://arxiv.org/abs/2207.00012v4) [paper-pdf](http://arxiv.org/pdf/2207.00012v4)

**Authors**: Kuan Li, Yang Liu, Xiang Ao, Jianfeng Chi, Jinghua Feng, Hao Yang, Qing He

**Abstract**: Benefiting from the message passing mechanism, Graph Neural Networks (GNNs) have been successful on flourish tasks over graph data. However, recent studies have shown that attackers can catastrophically degrade the performance of GNNs by maliciously modifying the graph structure. A straightforward solution to remedy this issue is to model the edge weights by learning a metric function between pairwise representations of two end nodes, which attempts to assign low weights to adversarial edges. The existing methods use either raw features or representations learned by supervised GNNs to model the edge weights. However, both strategies are faced with some immediate problems: raw features cannot represent various properties of nodes (e.g., structure information), and representations learned by supervised GNN may suffer from the poor performance of the classifier on the poisoned graph. We need representations that carry both feature information and as mush correct structure information as possible and are insensitive to structural perturbations. To this end, we propose an unsupervised pipeline, named STABLE, to optimize the graph structure. Finally, we input the well-refined graph into a downstream classifier. For this part, we design an advanced GCN that significantly enhances the robustness of vanilla GCN without increasing the time complexity. Extensive experiments on four real-world graph benchmarks demonstrate that STABLE outperforms the state-of-the-art methods and successfully defends against various attacks.



## **32. Denial-of-Service or Fine-Grained Control: Towards Flexible Model Poisoning Attacks on Federated Learning**

cs.LG

This paper has been accepted by the 32st International Joint  Conference on Artificial Intelligence (IJCAI-23, Main Track)

**SubmitDate**: 2023-04-21    [abs](http://arxiv.org/abs/2304.10783v1) [paper-pdf](http://arxiv.org/pdf/2304.10783v1)

**Authors**: Hangtao Zhang, Zeming Yao, Leo Yu Zhang, Shengshan Hu, Chao Chen, Alan Liew, Zhetao Li

**Abstract**: Federated learning (FL) is vulnerable to poisoning attacks, where adversaries corrupt the global aggregation results and cause denial-of-service (DoS). Unlike recent model poisoning attacks that optimize the amplitude of malicious perturbations along certain prescribed directions to cause DoS, we propose a Flexible Model Poisoning Attack (FMPA) that can achieve versatile attack goals. We consider a practical threat scenario where no extra knowledge about the FL system (e.g., aggregation rules or updates on benign devices) is available to adversaries. FMPA exploits the global historical information to construct an estimator that predicts the next round of the global model as a benign reference. It then fine-tunes the reference model to obtain the desired poisoned model with low accuracy and small perturbations. Besides the goal of causing DoS, FMPA can be naturally extended to launch a fine-grained controllable attack, making it possible to precisely reduce the global accuracy. Armed with precise control, malicious FL service providers can gain advantages over their competitors without getting noticed, hence opening a new attack surface in FL other than DoS. Even for the purpose of DoS, experiments show that FMPA significantly decreases the global accuracy, outperforming six state-of-the-art attacks.



## **33. Interpretable and Robust AI in EEG Systems: A Survey**

eess.SP

**SubmitDate**: 2023-04-21    [abs](http://arxiv.org/abs/2304.10755v1) [paper-pdf](http://arxiv.org/pdf/2304.10755v1)

**Authors**: Xinliang Zhou, Chenyu Liu, Liming Zhai, Ziyu Jia, Cuntai Guan, Yang Liu

**Abstract**: The close coupling of artificial intelligence (AI) and electroencephalography (EEG) has substantially advanced human-computer interaction (HCI) technologies in the AI era. Different from traditional EEG systems, the interpretability and robustness of AI-based EEG systems are becoming particularly crucial. The interpretability clarifies the inner working mechanisms of AI models and thus can gain the trust of users. The robustness reflects the AI's reliability against attacks and perturbations, which is essential for sensitive and fragile EEG signals. Thus the interpretability and robustness of AI in EEG systems have attracted increasing attention, and their research has achieved great progress recently. However, there is still no survey covering recent advances in this field. In this paper, we present the first comprehensive survey and summarize the interpretable and robust AI techniques for EEG systems. Specifically, we first propose a taxonomy of interpretability by characterizing it into three types: backpropagation, perturbation, and inherently interpretable methods. Then we classify the robustness mechanisms into four classes: noise and artifacts, human variability, data acquisition instability, and adversarial attacks. Finally, we identify several critical and unresolved challenges for interpretable and robust AI in EEG systems and further discuss their future directions.



## **34. Fooling Thermal Infrared Detectors in Physical World**

cs.CV

**SubmitDate**: 2023-04-21    [abs](http://arxiv.org/abs/2304.10712v1) [paper-pdf](http://arxiv.org/pdf/2304.10712v1)

**Authors**: Chengyin Hu, Weiwen Shi

**Abstract**: Infrared imaging systems have a vast array of potential applications in pedestrian detection and autonomous driving, and their safety performance is of great concern. However, few studies have explored the safety of infrared imaging systems in real-world settings. Previous research has used physical perturbations such as small bulbs and thermal "QR codes" to attack infrared imaging detectors, but such methods are highly visible and lack stealthiness. Other researchers have used hot and cold blocks to deceive infrared imaging detectors, but this method is limited in its ability to execute attacks from various angles. To address these shortcomings, we propose a novel physical attack called adversarial infrared blocks (AdvIB). By optimizing the physical parameters of the adversarial infrared blocks, this method can execute a stealthy black-box attack on thermal imaging system from various angles. We evaluate the proposed method based on its effectiveness, stealthiness, and robustness. Our physical tests show that the proposed method achieves a success rate of over 80% under most distance and angle conditions, validating its effectiveness. For stealthiness, our method involves attaching the adversarial infrared block to the inside of clothing, enhancing its stealthiness. Additionally, we test the proposed method on advanced detectors, and experimental results demonstrate an average attack success rate of 51.2%, proving its robustness. Overall, our proposed AdvIB method offers a promising avenue for conducting stealthy, effective and robust black-box attacks on thermal imaging system, with potential implications for real-world safety and security applications.



## **35. VenoMave: Targeted Poisoning Against Speech Recognition**

cs.SD

**SubmitDate**: 2023-04-20    [abs](http://arxiv.org/abs/2010.10682v3) [paper-pdf](http://arxiv.org/pdf/2010.10682v3)

**Authors**: Hojjat Aghakhani, Lea Schönherr, Thorsten Eisenhofer, Dorothea Kolossa, Thorsten Holz, Christopher Kruegel, Giovanni Vigna

**Abstract**: Despite remarkable improvements, automatic speech recognition is susceptible to adversarial perturbations. Compared to standard machine learning architectures, these attacks are significantly more challenging, especially since the inputs to a speech recognition system are time series that contain both acoustic and linguistic properties of speech. Extracting all recognition-relevant information requires more complex pipelines and an ensemble of specialized components. Consequently, an attacker needs to consider the entire pipeline. In this paper, we present VENOMAVE, the first training-time poisoning attack against speech recognition. Similar to the predominantly studied evasion attacks, we pursue the same goal: leading the system to an incorrect and attacker-chosen transcription of a target audio waveform. In contrast to evasion attacks, however, we assume that the attacker can only manipulate a small part of the training data without altering the target audio waveform at runtime. We evaluate our attack on two datasets: TIDIGITS and Speech Commands. When poisoning less than 0.17% of the dataset, VENOMAVE achieves attack success rates of more than 80.0%, without access to the victim's network architecture or hyperparameters. In a more realistic scenario, when the target audio waveform is played over the air in different rooms, VENOMAVE maintains a success rate of up to 73.3%. Finally, VENOMAVE achieves an attack transferability rate of 36.4% between two different model architectures.



## **36. Get Rid Of Your Trail: Remotely Erasing Backdoors in Federated Learning**

cs.LG

**SubmitDate**: 2023-04-20    [abs](http://arxiv.org/abs/2304.10638v1) [paper-pdf](http://arxiv.org/pdf/2304.10638v1)

**Authors**: Manaar Alam, Hithem Lamri, Michail Maniatakos

**Abstract**: Federated Learning (FL) enables collaborative deep learning training across multiple participants without exposing sensitive personal data. However, the distributed nature of FL and the unvetted participants' data makes it vulnerable to backdoor attacks. In these attacks, adversaries inject malicious functionality into the centralized model during training, leading to intentional misclassifications for specific adversary-chosen inputs. While previous research has demonstrated successful injections of persistent backdoors in FL, the persistence also poses a challenge, as their existence in the centralized model can prompt the central aggregation server to take preventive measures to penalize the adversaries. Therefore, this paper proposes a methodology that enables adversaries to effectively remove backdoors from the centralized model upon achieving their objectives or upon suspicion of possible detection. The proposed approach extends the concept of machine unlearning and presents strategies to preserve the performance of the centralized model and simultaneously prevent over-unlearning of information unrelated to backdoor patterns, making the adversaries stealthy while removing backdoors. To the best of our knowledge, this is the first work that explores machine unlearning in FL to remove backdoors to the benefit of adversaries. Exhaustive evaluation considering image classification scenarios demonstrates the efficacy of the proposed method in efficient backdoor removal from the centralized model, injected by state-of-the-art attacks across multiple configurations.



## **37. SoK: Let the Privacy Games Begin! A Unified Treatment of Data Inference Privacy in Machine Learning**

cs.LG

20 pages, to appear in 2023 IEEE Symposium on Security and Privacy

**SubmitDate**: 2023-04-20    [abs](http://arxiv.org/abs/2212.10986v2) [paper-pdf](http://arxiv.org/pdf/2212.10986v2)

**Authors**: Ahmed Salem, Giovanni Cherubin, David Evans, Boris Köpf, Andrew Paverd, Anshuman Suri, Shruti Tople, Santiago Zanella-Béguelin

**Abstract**: Deploying machine learning models in production may allow adversaries to infer sensitive information about training data. There is a vast literature analyzing different types of inference risks, ranging from membership inference to reconstruction attacks. Inspired by the success of games (i.e., probabilistic experiments) to study security properties in cryptography, some authors describe privacy inference risks in machine learning using a similar game-based style. However, adversary capabilities and goals are often stated in subtly different ways from one presentation to the other, which makes it hard to relate and compose results. In this paper, we present a game-based framework to systematize the body of knowledge on privacy inference risks in machine learning. We use this framework to (1) provide a unifying structure for definitions of inference risks, (2) formally establish known relations among definitions, and (3) to uncover hitherto unknown relations that would have been difficult to spot otherwise.



## **38. More is Better (Mostly): On the Backdoor Attacks in Federated Graph Neural Networks**

cs.CR

15 pages, 13 figures

**SubmitDate**: 2023-04-20    [abs](http://arxiv.org/abs/2202.03195v5) [paper-pdf](http://arxiv.org/pdf/2202.03195v5)

**Authors**: Jing Xu, Rui Wang, Stefanos Koffas, Kaitai Liang, Stjepan Picek

**Abstract**: Graph Neural Networks (GNNs) are a class of deep learning-based methods for processing graph domain information. GNNs have recently become a widely used graph analysis method due to their superior ability to learn representations for complex graph data. However, due to privacy concerns and regulation restrictions, centralized GNNs can be difficult to apply to data-sensitive scenarios. Federated learning (FL) is an emerging technology developed for privacy-preserving settings when several parties need to train a shared global model collaboratively. Although several research works have applied FL to train GNNs (Federated GNNs), there is no research on their robustness to backdoor attacks.   This paper bridges this gap by conducting two types of backdoor attacks in Federated GNNs: centralized backdoor attacks (CBA) and distributed backdoor attacks (DBA). Our experiments show that the DBA attack success rate is higher than CBA in almost all evaluated cases. For CBA, the attack success rate of all local triggers is similar to the global trigger even if the training set of the adversarial party is embedded with the global trigger. To further explore the properties of two backdoor attacks in Federated GNNs, we evaluate the attack performance for a different number of clients, trigger sizes, poisoning intensities, and trigger densities. Moreover, we explore the robustness of DBA and CBA against one defense. We find that both attacks are robust against the investigated defense, necessitating the need to consider backdoor attacks in Federated GNNs as a novel threat that requires custom defenses.



## **39. Byzantine-Resilient Learning Beyond Gradients: Distributing Evolutionary Search**

cs.DC

10 pages, 4 listings, 2 theorems

**SubmitDate**: 2023-04-20    [abs](http://arxiv.org/abs/2304.13540v1) [paper-pdf](http://arxiv.org/pdf/2304.13540v1)

**Authors**: Andrei Kucharavy, Matteo Monti, Rachid Guerraoui, Ljiljana Dolamic

**Abstract**: Modern machine learning (ML) models are capable of impressive performances. However, their prowess is not due only to the improvements in their architecture and training algorithms but also to a drastic increase in computational power used to train them.   Such a drastic increase led to a growing interest in distributed ML, which in turn made worker failures and adversarial attacks an increasingly pressing concern. While distributed byzantine resilient algorithms have been proposed in a differentiable setting, none exist in a gradient-free setting.   The goal of this work is to address this shortcoming. For that, we introduce a more general definition of byzantine-resilience in ML - the \textit{model-consensus}, that extends the definition of the classical distributed consensus. We then leverage this definition to show that a general class of gradient-free ML algorithms - ($1,\lambda$)-Evolutionary Search - can be combined with classical distributed consensus algorithms to generate gradient-free byzantine-resilient distributed learning algorithms. We provide proofs and pseudo-code for two specific cases - the Total Order Broadcast and proof-of-work leader election.



## **40. Certified Adversarial Robustness Within Multiple Perturbation Bounds**

cs.LG

**SubmitDate**: 2023-04-20    [abs](http://arxiv.org/abs/2304.10446v1) [paper-pdf](http://arxiv.org/pdf/2304.10446v1)

**Authors**: Soumalya Nandi, Sravanti Addepalli, Harsh Rangwani, R. Venkatesh Babu

**Abstract**: Randomized smoothing (RS) is a well known certified defense against adversarial attacks, which creates a smoothed classifier by predicting the most likely class under random noise perturbations of inputs during inference. While initial work focused on robustness to $\ell_2$ norm perturbations using noise sampled from a Gaussian distribution, subsequent works have shown that different noise distributions can result in robustness to other $\ell_p$ norm bounds as well. In general, a specific noise distribution is optimal for defending against a given $\ell_p$ norm based attack. In this work, we aim to improve the certified adversarial robustness against multiple perturbation bounds simultaneously. Towards this, we firstly present a novel \textit{certification scheme}, that effectively combines the certificates obtained using different noise distributions to obtain optimal results against multiple perturbation bounds. We further propose a novel \textit{training noise distribution} along with a \textit{regularized training scheme} to improve the certification within both $\ell_1$ and $\ell_2$ perturbation norms simultaneously. Contrary to prior works, we compare the certified robustness of different training algorithms across the same natural (clean) accuracy, rather than across fixed noise levels used for training and certification. We also empirically invalidate the argument that training and certifying the classifier with the same amount of noise gives the best results. The proposed approach achieves improvements on the ACR (Average Certified Radius) metric across both $\ell_1$ and $\ell_2$ perturbation bounds.



## **41. An Analysis of the Completion Time of the BB84 Protocol**

cs.PF

**SubmitDate**: 2023-04-20    [abs](http://arxiv.org/abs/2304.10218v1) [paper-pdf](http://arxiv.org/pdf/2304.10218v1)

**Authors**: Sounak Kar, Jean-Yves Le Boudec

**Abstract**: The BB84 QKD protocol is based on the idea that the sender and the receiver can reconcile a certain fraction of the teleported qubits to detect eavesdropping or noise and decode the rest to use as a private key. Under the present hardware infrastructure, decoherence of quantum states poses a significant challenge to performing perfect or efficient teleportation, meaning that a teleportation-based protocol must be run multiple times to observe success. Thus, performance analyses of such protocols usually consider the completion time, i.e., the time until success, rather than the duration of a single attempt. Moreover, due to decoherence, the success of an attempt is in general dependent on the duration of individual phases of that attempt, as quantum states must wait in memory while the success or failure of a generation phase is communicated to the relevant parties. In this work, we do a performance analysis of the completion time of the BB84 protocol in a setting where the sender and the receiver are connected via a single quantum repeater and the only quantum channel between them does not see any adversarial attack. Assuming certain distributional forms for the generation and communication phases of teleportation, we provide a method to compute the MGF of the completion time and subsequently derive an estimate of the CDF and a bound on the tail probability. This result helps us gauge the (tail) behaviour of the completion time in terms of the parameters characterising the elementary phases of teleportation, without having to run the protocol multiple times. We also provide an efficient simulation scheme to generate the completion time, which relies on expressing the completion time in terms of aggregated teleportation times. We numerically compare our approach with a full-scale simulation and observe good agreement between them.



## **42. Quantum-secure message authentication via blind-unforgeability**

quant-ph

37 pages, v4: Erratum added. We removed a result that had an error in  its proof

**SubmitDate**: 2023-04-20    [abs](http://arxiv.org/abs/1803.03761v4) [paper-pdf](http://arxiv.org/pdf/1803.03761v4)

**Authors**: Gorjan Alagic, Christian Majenz, Alexander Russell, Fang Song

**Abstract**: Formulating and designing authentication of classical messages in the presence of adversaries with quantum query access has been a longstanding challenge, as the familiar classical notions of unforgeability do not directly translate into meaningful notions in the quantum setting. A particular difficulty is how to fairly capture the notion of "predicting an unqueried value" when the adversary can query in quantum superposition.   We propose a natural definition of unforgeability against quantum adversaries called blind unforgeability. This notion defines a function to be predictable if there exists an adversary who can use "partially blinded" oracle access to predict values in the blinded region. We support the proposal with a number of technical results. We begin by establishing that the notion coincides with EUF-CMA in the classical setting and go on to demonstrate that the notion is satisfied by a number of simple guiding examples, such as random functions and quantum-query-secure pseudorandom functions. We then show the suitability of blind unforgeability for supporting canonical constructions and reductions. We prove that the "hash-and-MAC" paradigm and the Lamport one-time digital signature scheme are indeed unforgeable according to the definition. To support our analysis, we additionally define and study a new variety of quantum-secure hash functions called Bernoulli-preserving.   Finally, we demonstrate that blind unforgeability is stronger than a previous definition of Boneh and Zhandry [EUROCRYPT '13, CRYPTO '13] in the sense that we can construct an explicit function family which is forgeable by an attack that is recognized by blind-unforgeability, yet satisfies the definition by Boneh and Zhandry.



## **43. Diversifying the High-level Features for better Adversarial Transferability**

cs.CV

15 pages

**SubmitDate**: 2023-04-20    [abs](http://arxiv.org/abs/2304.10136v1) [paper-pdf](http://arxiv.org/pdf/2304.10136v1)

**Authors**: Zhiyuan Wang, Zeliang Zhang, Siyuan Liang, Xiaosen Wang

**Abstract**: Given the great threat of adversarial attacks against Deep Neural Networks (DNNs), numerous works have been proposed to boost transferability to attack real-world applications. However, existing attacks often utilize advanced gradient calculation or input transformation but ignore the white-box model. Inspired by the fact that DNNs are over-parameterized for superior performance, we propose diversifying the high-level features (DHF) for more transferable adversarial examples. In particular, DHF perturbs the high-level features by randomly transforming the high-level features and mixing them with the feature of benign samples when calculating the gradient at each iteration. Due to the redundancy of parameters, such transformation does not affect the classification performance but helps identify the invariant features across different models, leading to much better transferability. Empirical evaluations on ImageNet dataset show that DHF could effectively improve the transferability of existing momentum-based attacks. Incorporated into the input transformation-based attacks, DHF generates more transferable adversarial examples and outperforms the baselines with a clear margin when attacking several defense models, showing its generalization to various attacks and high effectiveness for boosting transferability.



## **44. Towards the Universal Defense for Query-Based Audio Adversarial Attacks**

eess.AS

Submitted to Cybersecurity journal

**SubmitDate**: 2023-04-20    [abs](http://arxiv.org/abs/2304.10088v1) [paper-pdf](http://arxiv.org/pdf/2304.10088v1)

**Authors**: Feng Guo, Zheng Sun, Yuxuan Chen, Lei Ju

**Abstract**: Recently, studies show that deep learning-based automatic speech recognition (ASR) systems are vulnerable to adversarial examples (AEs), which add a small amount of noise to the original audio examples. These AE attacks pose new challenges to deep learning security and have raised significant concerns about deploying ASR systems and devices. The existing defense methods are either limited in application or only defend on results, but not on process. In this work, we propose a novel method to infer the adversary intent and discover audio adversarial examples based on the AEs generation process. The insight of this method is based on the observation: many existing audio AE attacks utilize query-based methods, which means the adversary must send continuous and similar queries to target ASR models during the audio AE generation process. Inspired by this observation, We propose a memory mechanism by adopting audio fingerprint technology to analyze the similarity of the current query with a certain length of memory query. Thus, we can identify when a sequence of queries appears to be suspectable to generate audio AEs. Through extensive evaluation on four state-of-the-art audio AE attacks, we demonstrate that on average our defense identify the adversary intent with over 90% accuracy. With careful regard for robustness evaluations, we also analyze our proposed defense and its strength to withstand two adaptive attacks. Finally, our scheme is available out-of-the-box and directly compatible with any ensemble of ASR defense models to uncover audio AE attacks effectively without model retraining.



## **45. A Search-Based Testing Approach for Deep Reinforcement Learning Agents**

cs.SE

**SubmitDate**: 2023-04-20    [abs](http://arxiv.org/abs/2206.07813v3) [paper-pdf](http://arxiv.org/pdf/2206.07813v3)

**Authors**: Amirhossein Zolfagharian, Manel Abdellatif, Lionel Briand, Mojtaba Bagherzadeh, Ramesh S

**Abstract**: Deep Reinforcement Learning (DRL) algorithms have been increasingly employed during the last decade to solve various decision-making problems such as autonomous driving and robotics. However, these algorithms have faced great challenges when deployed in safety-critical environments since they often exhibit erroneous behaviors that can lead to potentially critical errors. One way to assess the safety of DRL agents is to test them to detect possible faults leading to critical failures during their execution. This raises the question of how we can efficiently test DRL policies to ensure their correctness and adherence to safety requirements. Most existing works on testing DRL agents use adversarial attacks that perturb states or actions of the agent. However, such attacks often lead to unrealistic states of the environment. Their main goal is to test the robustness of DRL agents rather than testing the compliance of agents' policies with respect to requirements. Due to the huge state space of DRL environments, the high cost of test execution, and the black-box nature of DRL algorithms, the exhaustive testing of DRL agents is impossible. In this paper, we propose a Search-based Testing Approach of Reinforcement Learning Agents (STARLA) to test the policy of a DRL agent by effectively searching for failing executions of the agent within a limited testing budget. We use machine learning models and a dedicated genetic algorithm to narrow the search towards faulty episodes. We apply STARLA on Deep-Q-Learning agents which are widely used as benchmarks and show that it significantly outperforms Random Testing by detecting more faults related to the agent's policy. We also investigate how to extract rules that characterize faulty episodes of the DRL agent using our search results. Such rules can be used to understand the conditions under which the agent fails and thus assess its deployment risks.



## **46. Quantifying the Preferential Direction of the Model Gradient in Adversarial Training With Projected Gradient Descent**

stat.ML

This paper was published in Pattern Recognition

**SubmitDate**: 2023-04-20    [abs](http://arxiv.org/abs/2009.04709v5) [paper-pdf](http://arxiv.org/pdf/2009.04709v5)

**Authors**: Ricardo Bigolin Lanfredi, Joyce D. Schroeder, Tolga Tasdizen

**Abstract**: Adversarial training, especially projected gradient descent (PGD), has proven to be a successful approach for improving robustness against adversarial attacks. After adversarial training, gradients of models with respect to their inputs have a preferential direction. However, the direction of alignment is not mathematically well established, making it difficult to evaluate quantitatively. We propose a novel definition of this direction as the direction of the vector pointing toward the closest point of the support of the closest inaccurate class in decision space. To evaluate the alignment with this direction after adversarial training, we apply a metric that uses generative adversarial networks to produce the smallest residual needed to change the class present in the image. We show that PGD-trained models have a higher alignment than the baseline according to our definition, that our metric presents higher alignment values than a competing metric formulation, and that enforcing this alignment increases the robustness of models.



## **47. Jedi: Entropy-based Localization and Removal of Adversarial Patches**

cs.CR

9 pages, 11 figures. To appear in CVPR 2023

**SubmitDate**: 2023-04-20    [abs](http://arxiv.org/abs/2304.10029v1) [paper-pdf](http://arxiv.org/pdf/2304.10029v1)

**Authors**: Bilel Tarchoun, Anouar Ben Khalifa, Mohamed Ali Mahjoub, Nael Abu-Ghazaleh, Ihsen Alouani

**Abstract**: Real-world adversarial physical patches were shown to be successful in compromising state-of-the-art models in a variety of computer vision applications. Existing defenses that are based on either input gradient or features analysis have been compromised by recent GAN-based attacks that generate naturalistic patches. In this paper, we propose Jedi, a new defense against adversarial patches that is resilient to realistic patch attacks. Jedi tackles the patch localization problem from an information theory perspective; leverages two new ideas: (1) it improves the identification of potential patch regions using entropy analysis: we show that the entropy of adversarial patches is high, even in naturalistic patches; and (2) it improves the localization of adversarial patches, using an autoencoder that is able to complete patch regions from high entropy kernels. Jedi achieves high-precision adversarial patch localization, which we show is critical to successfully repair the images. Since Jedi relies on an input entropy analysis, it is model-agnostic, and can be applied on pre-trained off-the-shelf models without changes to the training or inference of the protected models. Jedi detects on average 90% of adversarial patches across different benchmarks and recovers up to 94% of successful patch attacks (Compared to 75% and 65% for LGS and Jujutsu, respectively).



## **48. Fundamental Limitations of Alignment in Large Language Models**

cs.CL

**SubmitDate**: 2023-04-19    [abs](http://arxiv.org/abs/2304.11082v1) [paper-pdf](http://arxiv.org/pdf/2304.11082v1)

**Authors**: Yotam Wolf, Noam Wies, Yoav Levine, Amnon Shashua

**Abstract**: An important aspect in developing language models that interact with humans is aligning their behavior to be useful and unharmful for their human users. This is usually achieved by tuning the model in a way that enhances desired behaviors and inhibits undesired ones, a process referred to as alignment. In this paper, we propose a theoretical approach called Behavior Expectation Bounds (BEB) which allows us to formally investigate several inherent characteristics and limitations of alignment in large language models. Importantly, we prove that for any behavior that has a finite probability of being exhibited by the model, there exist prompts that can trigger the model into outputting this behavior, with probability that increases with the length of the prompt. This implies that any alignment process that attenuates undesired behavior but does not remove it altogether, is not safe against adversarial prompting attacks. Furthermore, our framework hints at the mechanism by which leading alignment approaches such as reinforcement learning from human feedback increase the LLM's proneness to being prompted into the undesired behaviors. Moreover, we include the notion of personas in our BEB framework, and find that behaviors which are generally very unlikely to be exhibited by the model can be brought to the front by prompting the model to behave as specific persona. This theoretical result is being experimentally demonstrated in large scale by the so called contemporary "chatGPT jailbreaks", where adversarial users trick the LLM into breaking its alignment guardrails by triggering it into acting as a malicious persona. Our results expose fundamental limitations in alignment of LLMs and bring to the forefront the need to devise reliable mechanisms for ensuring AI safety.



## **49. GREAT Score: Global Robustness Evaluation of Adversarial Perturbation using Generative Models**

cs.LG

**SubmitDate**: 2023-04-19    [abs](http://arxiv.org/abs/2304.09875v1) [paper-pdf](http://arxiv.org/pdf/2304.09875v1)

**Authors**: Li Zaitang, Pin-Yu Chen, Tsung-Yi Ho

**Abstract**: Current studies on adversarial robustness mainly focus on aggregating local robustness results from a set of data samples to evaluate and rank different models. However, the local statistics may not well represent the true global robustness of the underlying unknown data distribution. To address this challenge, this paper makes the first attempt to present a new framework, called GREAT Score , for global robustness evaluation of adversarial perturbation using generative models. Formally, GREAT Score carries the physical meaning of a global statistic capturing a mean certified attack-proof perturbation level over all samples drawn from a generative model. For finite-sample evaluation, we also derive a probabilistic guarantee on the sample complexity and the difference between the sample mean and the true mean. GREAT Score has several advantages: (1) Robustness evaluations using GREAT Score are efficient and scalable to large models, by sparing the need of running adversarial attacks. In particular, we show high correlation and significantly reduced computation cost of GREAT Score when compared to the attack-based model ranking on RobustBench (Croce,et. al. 2021). (2) The use of generative models facilitates the approximation of the unknown data distribution. In our ablation study with different generative adversarial networks (GANs), we observe consistency between global robustness evaluation and the quality of GANs. (3) GREAT Score can be used for remote auditing of privacy-sensitive black-box models, as demonstrated by our robustness evaluation on several online facial recognition services.



## **50. Experimental Certification of Quantum Transmission via Bell's Theorem**

quant-ph

34 pages, 14 figures

**SubmitDate**: 2023-04-19    [abs](http://arxiv.org/abs/2304.09605v1) [paper-pdf](http://arxiv.org/pdf/2304.09605v1)

**Authors**: Simon Neves, Laura dos Santos Martins, Verena Yacoub, Pascal Lefebvre, Ivan Supic, Damian Markham, Eleni Diamanti

**Abstract**: Quantum transmission links are central elements in essentially all implementations of quantum information protocols. Emerging progress in quantum technologies involving such links needs to be accompanied by appropriate certification tools. In adversarial scenarios, a certification method can be vulnerable to attacks if too much trust is placed on the underlying system. Here, we propose a protocol in a device independent framework, which allows for the certification of practical quantum transmission links in scenarios where minimal assumptions are made about the functioning of the certification setup. In particular, we take unavoidable transmission losses into account by modeling the link as a completely-positive trace-decreasing map. We also, crucially, remove the assumption of independent and identically distributed samples, which is known to be incompatible with adversarial settings. Finally, in view of the use of the certified transmitted states for follow-up applications, our protocol moves beyond certification of the channel to allow us to estimate the quality of the transmitted state itself. To illustrate the practical relevance and the feasibility of our protocol with currently available technology we provide an experimental implementation based on a state-of-the-art polarization entangled photon pair source in a Sagnac configuration and analyze its robustness for realistic losses and errors.



