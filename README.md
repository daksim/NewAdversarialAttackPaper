# Latest Adversarial Attack Papers
**update at 2022-06-26 06:31:28**

[中英双语版本](https://github.com/daksim/NewAdversarialAttackPaper/blob/main/README_CN.md)

## **1. Probabilistically Resilient Multi-Robot Informative Path Planning**

cs.RO

9 pages, 6 figures, submitted to IEEE Robotics and Automation Letters  (RA-L)

**SubmitDate**: 2022-06-23    [paper-pdf](http://arxiv.org/pdf/2206.11789v1)

**Authors**: Remy Wehbe, Ryan K. Williams

**Abstracts**: In this paper, we solve a multi-robot informative path planning (MIPP) task under the influence of uncertain communication and adversarial attackers. The goal is to create a multi-robot system that can learn and unify its knowledge of an unknown environment despite the presence of corrupted robots sharing malicious information. We use a Gaussian Process (GP) to model our unknown environment and define informativeness using the metric of mutual information. The objectives of our MIPP task is to maximize the amount of information collected by the team while maximizing the probability of resilience to attack. Unfortunately, these objectives are at odds especially when exploring large environments which necessitates disconnections between robots. As a result, we impose a probabilistic communication constraint that allows robots to meet intermittently and resiliently share information, and then act to maximize collected information during all other times. To solve our problem, we select meeting locations with the highest probability of resilience and use a sequential greedy algorithm to optimize paths for robots to explore. Finally, we show the validity of our results by comparing the learning ability of well-behaving robots applying resilient vs. non-resilient MIPP algorithms.



## **2. Towards End-to-End Private Automatic Speaker Recognition**

eess.AS

Accepted for publication at Interspeech 2022

**SubmitDate**: 2022-06-23    [paper-pdf](http://arxiv.org/pdf/2206.11750v1)

**Authors**: Francisco Teixeira, Alberto Abad, Bhiksha Raj, Isabel Trancoso

**Abstracts**: The development of privacy-preserving automatic speaker verification systems has been the focus of a number of studies with the intent of allowing users to authenticate themselves without risking the privacy of their voice. However, current privacy-preserving methods assume that the template voice representations (or speaker embeddings) used for authentication are extracted locally by the user. This poses two important issues: first, knowledge of the speaker embedding extraction model may create security and robustness liabilities for the authentication system, as this knowledge might help attackers in crafting adversarial examples able to mislead the system; second, from the point of view of a service provider the speaker embedding extraction model is arguably one of the most valuable components in the system and, as such, disclosing it would be highly undesirable. In this work, we show how speaker embeddings can be extracted while keeping both the speaker's voice and the service provider's model private, using Secure Multiparty Computation. Further, we show that it is possible to obtain reasonable trade-offs between security and computational cost. This work is complementary to those showing how authentication may be performed privately, and thus can be considered as another step towards fully private automatic speaker recognition.



## **3. BERT Rankers are Brittle: a Study using Adversarial Document Perturbations**

cs.IR

To appear in ICTIR 2022

**SubmitDate**: 2022-06-23    [paper-pdf](http://arxiv.org/pdf/2206.11724v1)

**Authors**: Yumeng Wang, Lijun Lyu, Avishek Anand

**Abstracts**: Contextual ranking models based on BERT are now well established for a wide range of passage and document ranking tasks. However, the robustness of BERT-based ranking models under adversarial inputs is under-explored. In this paper, we argue that BERT-rankers are not immune to adversarial attacks targeting retrieved documents given a query. Firstly, we propose algorithms for adversarial perturbation of both highly relevant and non-relevant documents using gradient-based optimization methods. The aim of our algorithms is to add/replace a small number of tokens to a highly relevant or non-relevant document to cause a large rank demotion or promotion. Our experiments show that a small number of tokens can already result in a large change in the rank of a document. Moreover, we find that BERT-rankers heavily rely on the document start/head for relevance prediction, making the initial part of the document more susceptible to adversarial attacks. More interestingly, we find a small set of recurring adversarial words that when added to documents result in successful rank demotion/promotion of any relevant/non-relevant document respectively. Finally, our adversarial tokens also show particular topic preferences within and across datasets, exposing potential biases from BERT pre-training or downstream datasets.



## **4. Exploring Adversarial Attacks and Defenses in Vision Transformers trained with DINO**

cs.CV

6 pages workshop paper accepted at AdvML Frontiers (ICML 2022)

**SubmitDate**: 2022-06-23    [paper-pdf](http://arxiv.org/pdf/2206.06761v2)

**Authors**: Javier Rando, Nasib Naimi, Thomas Baumann, Max Mathys

**Abstracts**: This work conducts the first analysis on the robustness against adversarial attacks on self-supervised Vision Transformers trained using DINO. First, we evaluate whether features learned through self-supervision are more robust to adversarial attacks than those emerging from supervised learning. Then, we present properties arising for attacks in the latent space. Finally, we evaluate whether three well-known defense strategies can increase adversarial robustness in downstream tasks by only fine-tuning the classification head to provide robustness even in view of limited compute resources. These defense strategies are: Adversarial Training, Ensemble Adversarial Training and Ensemble of Specialized Networks.



## **5. Bounding Training Data Reconstruction in Private (Deep) Learning**

cs.LG

**SubmitDate**: 2022-06-23    [paper-pdf](http://arxiv.org/pdf/2201.12383v4)

**Authors**: Chuan Guo, Brian Karrer, Kamalika Chaudhuri, Laurens van der Maaten

**Abstracts**: Differential privacy is widely accepted as the de facto method for preventing data leakage in ML, and conventional wisdom suggests that it offers strong protection against privacy attacks. However, existing semantic guarantees for DP focus on membership inference, which may overestimate the adversary's capabilities and is not applicable when membership status itself is non-sensitive. In this paper, we derive the first semantic guarantees for DP mechanisms against training data reconstruction attacks under a formal threat model. We show that two distinct privacy accounting methods -- Renyi differential privacy and Fisher information leakage -- both offer strong semantic protection against data reconstruction attacks.



## **6. A Framework for Understanding Model Extraction Attack and Defense**

cs.LG

**SubmitDate**: 2022-06-23    [paper-pdf](http://arxiv.org/pdf/2206.11480v1)

**Authors**: Xun Xian, Mingyi Hong, Jie Ding

**Abstracts**: The privacy of machine learning models has become a significant concern in many emerging Machine-Learning-as-a-Service applications, where prediction services based on well-trained models are offered to users via pay-per-query. The lack of a defense mechanism can impose a high risk on the privacy of the server's model since an adversary could efficiently steal the model by querying only a few `good' data points. The interplay between a server's defense and an adversary's attack inevitably leads to an arms race dilemma, as commonly seen in Adversarial Machine Learning. To study the fundamental tradeoffs between model utility from a benign user's view and privacy from an adversary's view, we develop new metrics to quantify such tradeoffs, analyze their theoretical properties, and develop an optimization problem to understand the optimal adversarial attack and defense strategies. The developed concepts and theory match the empirical findings on the `equilibrium' between privacy and utility. In terms of optimization, the key ingredient that enables our results is a unified representation of the attack-defense problem as a min-max bi-level problem. The developed results will be demonstrated by examples and experiments.



## **7. Incorporating Hidden Layer representation into Adversarial Attacks and Defences**

cs.LG

**SubmitDate**: 2022-06-23    [paper-pdf](http://arxiv.org/pdf/2011.14045v2)

**Authors**: Haojing Shen, Sihong Chen, Ran Wang, Xizhao Wang

**Abstracts**: In this paper, we propose a defence strategy to improve adversarial robustness by incorporating hidden layer representation. The key of this defence strategy aims to compress or filter input information including adversarial perturbation. And this defence strategy can be regarded as an activation function which can be applied to any kind of neural network. We also prove theoretically the effectiveness of this defense strategy under certain conditions. Besides, incorporating hidden layer representation we propose three types of adversarial attacks to generate three types of adversarial examples, respectively. The experiments show that our defence method can significantly improve the adversarial robustness of deep neural networks which achieves the state-of-the-art performance even though we do not adopt adversarial training.



## **8. Adversarial Learning with Cost-Sensitive Classes**

cs.LG

12 pages

**SubmitDate**: 2022-06-23    [paper-pdf](http://arxiv.org/pdf/2101.12372v2)

**Authors**: Haojing Shen, Sihong Chen, Ran Wang, Xizhao Wang

**Abstracts**: It is necessary to improve the performance of some special classes or to particularly protect them from attacks in adversarial learning. This paper proposes a framework combining cost-sensitive classification and adversarial learning together to train a model that can distinguish between protected and unprotected classes, such that the protected classes are less vulnerable to adversarial examples. We find in this framework an interesting phenomenon during the training of deep neural networks, called Min-Max property, that is, the absolute values of most parameters in the convolutional layer approach zero while the absolute values of a few parameters are significantly larger becoming bigger. Based on this Min-Max property which is formulated and analyzed in a view of random distribution, we further build a new defense model against adversarial examples for adversarial robustness improvement. An advantage of the built model is that it performs better than the standard one and can combine with adversarial training to achieve an improved performance. It is experimentally confirmed that, regarding the average accuracy of all classes, our model is almost as same as the existing models when an attack does not occur and is better than the existing models when an attack occurs. Specifically, regarding the accuracy of protected classes, the proposed model is much better than the existing models when an attack occurs.



## **9. Shilling Black-box Recommender Systems by Learning to Generate Fake User Profiles**

cs.IR

Accepted by TNNLS. 15 pages, 8 figures

**SubmitDate**: 2022-06-23    [paper-pdf](http://arxiv.org/pdf/2206.11433v1)

**Authors**: Chen Lin, Si Chen, Meifang Zeng, Sheng Zhang, Min Gao, Hui Li

**Abstracts**: Due to the pivotal role of Recommender Systems (RS) in guiding customers towards the purchase, there is a natural motivation for unscrupulous parties to spoof RS for profits. In this paper, we study Shilling Attack where an adversarial party injects a number of fake user profiles for improper purposes. Conventional Shilling Attack approaches lack attack transferability (i.e., attacks are not effective on some victim RS models) and/or attack invisibility (i.e., injected profiles can be easily detected). To overcome these issues, we present Leg-UP, a novel attack model based on the Generative Adversarial Network. Leg-UP learns user behavior patterns from real users in the sampled ``templates'' and constructs fake user profiles. To simulate real users, the generator in Leg-UP directly outputs discrete ratings. To enhance attack transferability, the parameters of the generator are optimized by maximizing the attack performance on a surrogate RS model. To improve attack invisibility, Leg-UP adopts a discriminator to guide the generator to generate undetectable fake user profiles. Experiments on benchmarks have shown that Leg-UP exceeds state-of-the-art Shilling Attack methods on a wide range of victim RS models. The source code of our work is available at: https://github.com/XMUDM/ShillingAttack.



## **10. Making Generated Images Hard To Spot: A Transferable Attack On Synthetic Image Detectors**

cs.CV

**SubmitDate**: 2022-06-22    [paper-pdf](http://arxiv.org/pdf/2104.12069v2)

**Authors**: Xinwei Zhao, Matthew C. Stamm

**Abstracts**: Visually realistic GAN-generated images have recently emerged as an important misinformation threat. Research has shown that these synthetic images contain forensic traces that are readily identifiable by forensic detectors. Unfortunately, these detectors are built upon neural networks, which are vulnerable to recently developed adversarial attacks. In this paper, we propose a new anti-forensic attack capable of fooling GAN-generated image detectors. Our attack uses an adversarially trained generator to synthesize traces that these detectors associate with real images. Furthermore, we propose a technique to train our attack so that it can achieve transferability, i.e. it can fool unknown CNNs that it was not explicitly trained against. We evaluate our attack through an extensive set of experiments, where we show that our attack can fool eight state-of-the-art detection CNNs with synthetic images created using seven different GANs, and outperform other alternative attacks.



## **11. AdvSmo: Black-box Adversarial Attack by Smoothing Linear Structure of Texture**

cs.CV

6 pages,3 figures

**SubmitDate**: 2022-06-22    [paper-pdf](http://arxiv.org/pdf/2206.10988v1)

**Authors**: Hui Xia, Rui Zhang, Shuliang Jiang, Zi Kang

**Abstracts**: Black-box attacks usually face two problems: poor transferability and the inability to evade the adversarial defense. To overcome these shortcomings, we create an original approach to generate adversarial examples by smoothing the linear structure of the texture in the benign image, called AdvSmo. We construct the adversarial examples without relying on any internal information to the target model and design the imperceptible-high attack success rate constraint to guide the Gabor filter to select appropriate angles and scales to smooth the linear texture from the input images to generate adversarial examples. Benefiting from the above design concept, AdvSmo will generate adversarial examples with strong transferability and solid evasiveness. Finally, compared to the four advanced black-box adversarial attack methods, for the eight target models, the results show that AdvSmo improves the average attack success rate by 9% on the CIFAR-10 and 16% on the Tiny-ImageNet dataset compared to the best of these attack methods.



## **12. Adversarial Reconfigurable Intelligent Surface Against Physical Layer Key Generation**

eess.SP

**SubmitDate**: 2022-06-22    [paper-pdf](http://arxiv.org/pdf/2206.10955v1)

**Authors**: Zhuangkun Wei, Bin Li, Weisi Guo

**Abstracts**: The development of reconfigurable intelligent surface (RIS) has recently advanced the research of physical layer security (PLS). Beneficial impact of RIS includes but is not limited to offering a new domain of freedom (DoF) for key-less PLS optimization, and increasing channel randomness for physical layer secret key generation (PL-SKG). However, there is a lack of research studying how adversarial RIS can be used to damage the communication confidentiality. In this work, we show how a Eve controlled adversarial RIS (Eve-RIS) can be used to reconstruct the shared PLS secret key between legitimate users (Alice and Bob). This is achieved by Eve-RIS overlaying the legitimate channel with an artificial random and reciprocal channel. The resulting Eve-RIS corrupted channel enable Eve to successfully attack the PL-SKG process. To operationalize this novel concept, we design Eve-RIS schemes against two PL-SKG techniques used: (i) the channel estimation based PL-SKG, and (ii) the two-way cross multiplication based PL-SKG. Our results show a high key match rate between the designed Eve-RIS and the legitimate users. We also present theoretical key match rate between Eve-RIS and legitimate users. Our novel scheme is different from the existing spoofing-Eve, in that the latter can be easily detected by comparing the channel estimation results of the legitimate users. Indeed, our proposed Eve-RIS can maintain the legitimate channel reciprocity, which makes detection challenging. This means the novel Eve-RIS provides a new eavesdropping threat on PL-SKG, which can spur new research areas to counter adversarial RIS attacks.



## **13. Introduction to Machine Learning for the Sciences**

physics.comp-ph

84 pages, 37 figures. The content of these lecture notes together  with exercises is available under http://www.ml-lectures.org. A shorter  German version of the lecture notes is published in the Springer essential  series, ISBN 978-3-658-32268-7, doi:10.1007/978-3-658-32268-7

**SubmitDate**: 2022-06-22    [paper-pdf](http://arxiv.org/pdf/2102.04883v2)

**Authors**: Titus Neupert, Mark H Fischer, Eliska Greplova, Kenny Choo, M. Michael Denner

**Abstracts**: This is an introductory machine-learning course specifically developed with STEM students in mind. Our goal is to provide the interested reader with the basics to employ machine learning in their own projects and to familiarize themself with the terminology as a foundation for further reading of the relevant literature. In these lecture notes, we discuss supervised, unsupervised, and reinforcement learning. The notes start with an exposition of machine learning methods without neural networks, such as principle component analysis, t-SNE, clustering, as well as linear regression and linear classifiers. We continue with an introduction to both basic and advanced neural-network structures such as dense feed-forward and conventional neural networks, recurrent neural networks, restricted Boltzmann machines, (variational) autoencoders, generative adversarial networks. Questions of interpretability are discussed for latent-space representations and using the examples of dreaming and adversarial attacks. The final section is dedicated to reinforcement learning, where we introduce basic notions of value functions and policy learning.



## **14. Guided Diffusion Model for Adversarial Purification from Random Noise**

cs.LG

**SubmitDate**: 2022-06-22    [paper-pdf](http://arxiv.org/pdf/2206.10875v1)

**Authors**: Quanlin Wu, Hang Ye, Yuntian Gu

**Abstracts**: In this paper, we propose a novel guided diffusion purification approach to provide a strong defense against adversarial attacks. Our model achieves 89.62% robust accuracy under PGD-L_inf attack (eps = 8/255) on the CIFAR-10 dataset. We first explore the essential correlations between unguided diffusion models and randomized smoothing, enabling us to apply the models to certified robustness. The empirical results show that our models outperform randomized smoothing by 5% when the certified L2 radius r is larger than 0.5.



## **15. Robust Universal Adversarial Perturbations**

cs.LG

16 pages, 3 figures

**SubmitDate**: 2022-06-22    [paper-pdf](http://arxiv.org/pdf/2206.10858v1)

**Authors**: Changming Xu, Gagandeep Singh

**Abstracts**: Universal Adversarial Perturbations (UAPs) are imperceptible, image-agnostic vectors that cause deep neural networks (DNNs) to misclassify inputs from a data distribution with high probability. Existing methods do not create UAPs robust to transformations, thereby limiting their applicability as a real-world attacks. In this work, we introduce a new concept and formulation of robust universal adversarial perturbations. Based on our formulation, we build a novel, iterative algorithm that leverages probabilistic robustness bounds for generating UAPs robust against transformations generated by composing arbitrary sub-differentiable transformation functions. We perform an extensive evaluation on the popular CIFAR-10 and ILSVRC 2012 datasets measuring robustness under human-interpretable semantic transformations, such as rotation, contrast changes, etc, that are common in the real-world. Our results show that our generated UAPs are significantly more robust than those from baselines.



## **16. Adaptive Adversarial Training to Improve Adversarial Robustness of DNNs for Medical Image Segmentation and Detection**

eess.IV

17 pages

**SubmitDate**: 2022-06-22    [paper-pdf](http://arxiv.org/pdf/2206.01736v2)

**Authors**: Linhai Ma, Liang Liang

**Abstracts**: It is known that Deep Neural networks (DNNs) are vulnerable to adversarial attacks, and the adversarial robustness of DNNs could be improved by adding adversarial noises to training data (e.g., the standard adversarial training (SAT)). However, inappropriate noises added to training data may reduce a model's performance, which is termed the trade-off between accuracy and robustness. This problem has been sufficiently studied for the classification of whole images but has rarely been explored for image analysis tasks in the medical application domain, including image segmentation, landmark detection, and object detection tasks. In this study, we show that, for those medical image analysis tasks, the SAT method has a severe issue that limits its practical use: it generates a fixed and unified level of noise for all training samples for robust DNN training. A high noise level may lead to a large reduction in model performance and a low noise level may not be effective in improving robustness. To resolve this issue, we design an adaptive-margin adversarial training (AMAT) method that generates sample-wise adaptive adversarial noises for robust DNN training. In contrast to the existing, classification-oriented adversarial training methods, our AMAT method uses a loss-defined-margin strategy so that it can be applied to different tasks as long as the loss functions are well-defined. We successfully apply our AMAT method to state-of-the-art DNNs, using five publicly available datasets. The experimental results demonstrate that: (1) our AMAT method can be applied to the three seemingly different tasks in the medical image application domain; (2) AMAT outperforms the SAT method in adversarial robustness; (3) AMAT has a minimal reduction in prediction accuracy on clean data, compared with the SAT method; and (4) AMAT has almost the same training time cost as SAT.



## **17. SSMI: How to Make Objects of Interest Disappear without Accessing Object Detectors?**

cs.CV

6 pages, 2 figures

**SubmitDate**: 2022-06-22    [paper-pdf](http://arxiv.org/pdf/2206.10809v1)

**Authors**: Hui Xia, Rui Zhang, Zi Kang, Shuliang Jiang

**Abstracts**: Most black-box adversarial attack schemes for object detectors mainly face two shortcomings: requiring access to the target model and generating inefficient adversarial examples (failing to make objects disappear in large numbers). To overcome these shortcomings, we propose a black-box adversarial attack scheme based on semantic segmentation and model inversion (SSMI). We first locate the position of the target object using semantic segmentation techniques. Next, we design a neighborhood background pixel replacement to replace the target region pixels with background pixels to ensure that the pixel modifications are not easily detected by human vision. Finally, we reconstruct a machine-recognizable example and use the mask matrix to select pixels in the reconstructed example to modify the benign image to generate an adversarial example. Detailed experimental results show that SSMI can generate efficient adversarial examples to evade human-eye perception and make objects of interest disappear. And more importantly, SSMI outperforms existing same kinds of attacks. The maximum increase in new and disappearing labels is 16%, and the maximum decrease in mAP metrics for object detection is 36%.



## **18. Secure and Efficient Query Processing in Outsourced Databases**

cs.CR

Ph.D. thesis

**SubmitDate**: 2022-06-21    [paper-pdf](http://arxiv.org/pdf/2206.10753v1)

**Authors**: Dmytro Bogatov

**Abstracts**: Various cryptographic techniques are used in outsourced database systems to ensure data privacy while allowing for efficient querying. This work proposes a definition and components of a new secure and efficient outsourced database system, which answers various types of queries, with different privacy guarantees in different security models. This work starts with the survey of five order-revealing encryption schemes that can be used directly in many database indices and five range query protocols with various security / efficiency tradeoffs. The survey systematizes the state-of-the-art range query solutions in a snapshot adversary setting and offers some non-obvious observations regarding the efficiency of the constructions. In $\mathcal{E}\text{psolute}$, a secure range query engine, security is achieved in a setting with a much stronger adversary where she can continuously observe everything on the server, and leaking even the result size can enable a reconstruction attack. $\mathcal{E}\text{psolute}$ proposes a definition, construction, analysis, and experimental evaluation of a system that provably hides both access pattern and communication volume while remaining efficient. The work concludes with $k\text{-a}n\text{o}n$ -- a secure similarity search engine in a snapshot adversary model. The work presents a construction in which the security of $k\text{NN}$ queries is achieved similarly to OPE / ORE solutions -- encrypting the input with an approximate Distance Comparison Preserving Encryption scheme so that the inputs, the points in a hyperspace, are perturbed, but the query algorithm still produces accurate results. We use TREC datasets and queries for the search, and track the rank quality metrics such as MRR and nDCG. For the attacks, we build an LSTM model that trains on the correlation between a sentence and its embedding and then predicts words from the embedding.



## **19. FlashSyn: Flash Loan Attack Synthesis via Counter Example Driven Approximation**

cs.PL

29 pages, 8 figures, technical report

**SubmitDate**: 2022-06-21    [paper-pdf](http://arxiv.org/pdf/2206.10708v1)

**Authors**: Zhiyang Chen, Sidi Mohamed Beillahi, Fan Long

**Abstracts**: In decentralized finance (DeFi) ecosystem, lenders can offer flash loans to borrowers, i.e., loans that are only valid within a blockchain transaction and must be repaid with some fees by the end of that transaction. Unlike normal loans, flash loans allow borrowers to borrow a large amount of assets without upfront collaterals deposits. Malicious adversaries can use flash loans to gather large amount of assets to launch costly exploitations targeting DeFi protocols. In this paper, we introduce a new framework for automated synthesis of adversarial contracts that exploit DeFi protocols using flash loans. To bypass the complexity of a DeFi protocol, we propose a new technique to approximate DeFi protocol functional behaviors using numerical methods. Then, we propose a novel algorithm to find an adversarial attack which constitutes of a sequence of invocations of functions in a DeFi protocol with the optimized parameters for profits. We implemented our framework in a tool called FlashSyn. We run FlashSyn on 5 DeFi protocols that were victims to flash loan attacks and DeFi protocols from Damn Vulnerable DeFi challenges. FlashSyn automatically synthesizes an adversarial attack for each one of them.



## **20. Using EBGAN for Anomaly Intrusion Detection**

cs.CR

**SubmitDate**: 2022-06-21    [paper-pdf](http://arxiv.org/pdf/2206.10400v1)

**Authors**: Yi Cui, Wenfeng Shen, Jian Zhang, Weijia Lu, Chuang Liu, Lin Sun, Si Chen

**Abstracts**: As an active network security protection scheme, intrusion detection system (IDS) undertakes the important responsibility of detecting network attacks in the form of malicious network traffic. Intrusion detection technology is an important part of IDS. At present, many scholars have carried out extensive research on intrusion detection technology. However, developing an efficient intrusion detection method for massive network traffic data is still difficult. Since Generative Adversarial Networks (GANs) have powerful modeling capabilities for complex high-dimensional data, they provide new ideas for addressing this problem. In this paper, we put forward an EBGAN-based intrusion detection method, IDS-EBGAN, that classifies network records as normal traffic or malicious traffic. The generator in IDS-EBGAN is responsible for converting the original malicious network traffic in the training set into adversarial malicious examples. This is because we want to use adversarial learning to improve the ability of discriminator to detect malicious traffic. At the same time, the discriminator adopts Autoencoder model. During testing, IDS-EBGAN uses reconstruction error of discriminator to classify traffic records.



## **21. Problem-Space Evasion Attacks in the Android OS: a Survey**

cs.CR

**SubmitDate**: 2022-06-21    [paper-pdf](http://arxiv.org/pdf/2205.14576v2)

**Authors**: Harel Berger, Chen Hajaj, Amit Dvir

**Abstracts**: Android is the most popular OS worldwide. Therefore, it is a target for various kinds of malware. As a countermeasure, the security community works day and night to develop appropriate Android malware detection systems, with ML-based or DL-based systems considered as some of the most common types. Against these detection systems, intelligent adversaries develop a wide set of evasion attacks, in which an attacker slightly modifies a malware sample to evade its target detection system. In this survey, we address problem-space evasion attacks in the Android OS, where attackers manipulate actual APKs, rather than their extracted feature vector. We aim to explore this kind of attacks, frequently overlooked by the research community due to a lack of knowledge of the Android domain, or due to focusing on general mathematical evasion attacks - i.e., feature-space evasion attacks. We discuss the different aspects of problem-space evasion attacks, using a new taxonomy, which focuses on key ingredients of each problem-space attack, such as the attacker model, the attacker's mode of operation, and the functional assessment of post-attack applications.



## **22. Certifiably Robust Policy Learning against Adversarial Communication in Multi-agent Systems**

cs.LG

**SubmitDate**: 2022-06-21    [paper-pdf](http://arxiv.org/pdf/2206.10158v1)

**Authors**: Yanchao Sun, Ruijie Zheng, Parisa Hassanzadeh, Yongyuan Liang, Soheil Feizi, Sumitra Ganesh, Furong Huang

**Abstracts**: Communication is important in many multi-agent reinforcement learning (MARL) problems for agents to share information and make good decisions. However, when deploying trained communicative agents in a real-world application where noise and potential attackers exist, the safety of communication-based policies becomes a severe issue that is underexplored. Specifically, if communication messages are manipulated by malicious attackers, agents relying on untrustworthy communication may take unsafe actions that lead to catastrophic consequences. Therefore, it is crucial to ensure that agents will not be misled by corrupted communication, while still benefiting from benign communication. In this work, we consider an environment with $N$ agents, where the attacker may arbitrarily change the communication from any $C<\frac{N-1}{2}$ agents to a victim agent. For this strong threat model, we propose a certifiable defense by constructing a message-ensemble policy that aggregates multiple randomly ablated message sets. Theoretical analysis shows that this message-ensemble policy can utilize benign communication while being certifiably robust to adversarial communication, regardless of the attacking algorithm. Experiments in multiple environments verify that our defense significantly improves the robustness of trained policies against various types of attacks.



## **23. ProML: A Decentralised Platform for Provenance Management of Machine Learning Software Systems**

cs.SE

Accepted as full paper in ECSA 2022 conference. To be presented

**SubmitDate**: 2022-06-21    [paper-pdf](http://arxiv.org/pdf/2206.10110v1)

**Authors**: Nguyen Khoi Tran, Bushra Sabir, M. Ali Babar, Nini Cui, Mehran Abolhasan, Justin Lipman

**Abstracts**: Large-scale Machine Learning (ML) based Software Systems are increasingly developed by distributed teams situated in different trust domains. Insider threats can launch attacks from any domain to compromise ML assets (models and datasets). Therefore, practitioners require information about how and by whom ML assets were developed to assess their quality attributes such as security, safety, and fairness. Unfortunately, it is challenging for ML teams to access and reconstruct such historical information of ML assets (ML provenance) because it is generally fragmented across distributed ML teams and threatened by the same adversaries that attack ML assets. This paper proposes ProML, a decentralised platform that leverages blockchain and smart contracts to empower distributed ML teams to jointly manage a single source of truth about circulated ML assets' provenance without relying on a third party, which is vulnerable to insider threats and presents a single point of failure. We propose a novel architectural approach called Artefact-as-a-State-Machine to leverage blockchain transactions and smart contracts for managing ML provenance information and introduce a user-driven provenance capturing mechanism to integrate existing scripts and tools to ProML without compromising participants' control over their assets and toolchains. We evaluate the performance and overheads of ProML by benchmarking a proof-of-concept system on a global blockchain. Furthermore, we assessed ProML's security against a threat model of a distributed ML workflow.



## **24. Make Some Noise: Reliable and Efficient Single-Step Adversarial Training**

cs.LG

**SubmitDate**: 2022-06-20    [paper-pdf](http://arxiv.org/pdf/2202.01181v2)

**Authors**: Pau de Jorge, Adel Bibi, Riccardo Volpi, Amartya Sanyal, Philip H. S. Torr, Grégory Rogez, Puneet K. Dokania

**Abstracts**: Recently, Wong et al. showed that adversarial training with single-step FGSM leads to a characteristic failure mode named catastrophic overfitting (CO), in which a model becomes suddenly vulnerable to multi-step attacks. They showed that adding a random perturbation prior to FGSM (RS-FGSM) seemed to be sufficient to prevent CO. However, Andriushchenko and Flammarion observed that RS-FGSM still leads to CO for larger perturbations, and proposed an expensive regularizer (GradAlign) to avoid CO. In this work, we methodically revisit the role of noise and clipping in single-step adversarial training. Contrary to previous intuitions, we find that using a stronger noise around the clean sample combined with not clipping is highly effective in avoiding CO for large perturbation radii. Based on these observations, we then propose Noise-FGSM (N-FGSM) that, while providing the benefits of single-step adversarial training, does not suffer from CO. Empirical analyses on a large suite of experiments show that N-FGSM is able to match or surpass the performance of previous single-step methods while achieving a 3$\times$ speed-up. Code can be found in https://github.com/pdejorge/N-FGSM



## **25. Diversified Adversarial Attacks based on Conjugate Gradient Method**

cs.LG

**SubmitDate**: 2022-06-20    [paper-pdf](http://arxiv.org/pdf/2206.09628v1)

**Authors**: Keiichiro Yamamura, Haruki Sato, Nariaki Tateiwa, Nozomi Hata, Toru Mitsutake, Issa Oe, Hiroki Ishikura, Katsuki Fujisawa

**Abstracts**: Deep learning models are vulnerable to adversarial examples, and adversarial attacks used to generate such examples have attracted considerable research interest. Although existing methods based on the steepest descent have achieved high attack success rates, ill-conditioned problems occasionally reduce their performance. To address this limitation, we utilize the conjugate gradient (CG) method, which is effective for this type of problem, and propose a novel attack algorithm inspired by the CG method, named the Auto Conjugate Gradient (ACG) attack. The results of large-scale evaluation experiments conducted on the latest robust models show that, for most models, ACG was able to find more adversarial examples with fewer iterations than the existing SOTA algorithm Auto-PGD (APGD). We investigated the difference in search performance between ACG and APGD in terms of diversification and intensification, and define a measure called Diversity Index (DI) to quantify the degree of diversity. From the analysis of the diversity using this index, we show that the more diverse search of the proposed method remarkably improves its attack success rate.



## **26. On the Limitations of Stochastic Pre-processing Defenses**

cs.LG

**SubmitDate**: 2022-06-19    [paper-pdf](http://arxiv.org/pdf/2206.09491v1)

**Authors**: Yue Gao, Ilia Shumailov, Kassem Fawaz, Nicolas Papernot

**Abstracts**: Defending against adversarial examples remains an open problem. A common belief is that randomness at inference increases the cost of finding adversarial inputs. An example of such a defense is to apply a random transformation to inputs prior to feeding them to the model. In this paper, we empirically and theoretically investigate such stochastic pre-processing defenses and demonstrate that they are flawed. First, we show that most stochastic defenses are weaker than previously thought; they lack sufficient randomness to withstand even standard attacks like projected gradient descent. This casts doubt on a long-held assumption that stochastic defenses invalidate attacks designed to evade deterministic defenses and force attackers to integrate the Expectation over Transformation (EOT) concept. Second, we show that stochastic defenses confront a trade-off between adversarial robustness and model invariance; they become less effective as the defended model acquires more invariance to their randomization. Future work will need to decouple these two effects. Our code is available in the supplementary material.



## **27. A Universal Adversarial Policy for Text Classifiers**

cs.LG

Accepted for publication in Neural Networks (2022), see  https://doi.org/10.1016/j.neunet.2022.06.018

**SubmitDate**: 2022-06-19    [paper-pdf](http://arxiv.org/pdf/2206.09458v1)

**Authors**: Gallil Maimon, Lior Rokach

**Abstracts**: Discovering the existence of universal adversarial perturbations had large theoretical and practical impacts on the field of adversarial learning. In the text domain, most universal studies focused on adversarial prefixes which are added to all texts. However, unlike the vision domain, adding the same perturbation to different inputs results in noticeably unnatural inputs. Therefore, we introduce a new universal adversarial setup - a universal adversarial policy, which has many advantages of other universal attacks but also results in valid texts - thus making it relevant in practice. We achieve this by learning a single search policy over a predefined set of semantics preserving text alterations, on many texts. This formulation is universal in that the policy is successful in finding adversarial examples on new texts efficiently. Our approach uses text perturbations which were extensively shown to produce natural attacks in the non-universal setup (specific synonym replacements). We suggest a strong baseline approach for this formulation which uses reinforcement learning. It's ability to generalise (from as few as 500 training texts) shows that universal adversarial patterns exist in the text domain as well.



## **28. JPEG Compression-Resistant Low-Mid Adversarial Perturbation against Unauthorized Face Recognition System**

cs.CV

**SubmitDate**: 2022-06-19    [paper-pdf](http://arxiv.org/pdf/2206.09410v1)

**Authors**: Jiaming Zhang, Qi Yi, Jitao Sang

**Abstracts**: It has been observed that the unauthorized use of face recognition system raises privacy problems. Using adversarial perturbations provides one possible solution to address this issue. A critical issue to exploit adversarial perturbation against unauthorized face recognition system is that: The images uploaded to the web need to be processed by JPEG compression, which weakens the effectiveness of adversarial perturbation. Existing JPEG compression-resistant methods fails to achieve a balance among compression resistance, transferability, and attack effectiveness. To this end, we propose a more natural solution called low frequency adversarial perturbation (LFAP). Instead of restricting the adversarial perturbations, we turn to regularize the source model to employing more low-frequency features by adversarial training. Moreover, to better influence model in different frequency components, we proposed the refined low-mid frequency adversarial perturbation (LMFAP) considering the mid frequency components as the productive complement. We designed a variety of settings in this study to simulate the real-world application scenario, including cross backbones, supervisory heads, training datasets and testing datasets. Quantitative and qualitative experimental results validate the effectivenss of proposed solutions.



## **29. Towards Adversarial Attack on Vision-Language Pre-training Models**

cs.LG

**SubmitDate**: 2022-06-19    [paper-pdf](http://arxiv.org/pdf/2206.09391v1)

**Authors**: Jiaming Zhang, Qi Yi, Jitao Sang

**Abstracts**: While vision-language pre-training model (VLP) has shown revolutionary improvements on various vision-language (V+L) tasks, the studies regarding its adversarial robustness remain largely unexplored. This paper studied the adversarial attack on popular VLP models and V+L tasks. First, we analyzed the performance of adversarial attacks under different settings. By examining the influence of different perturbed objects and attack targets, we concluded some key observations as guidance on both designing strong multimodal adversarial attack and constructing robust VLP models. Second, we proposed a novel multimodal attack method on the VLP models called Collaborative Multimodal Adversarial Attack (Co-Attack), which collectively carries out the attacks on the image modality and the text modality. Experimental results demonstrated that the proposed method achieves improved attack performances on different V+L downstream tasks and VLP models. The analysis observations and novel attack method hopefully provide new understanding into the adversarial robustness of VLP models, so as to contribute their safe and reliable deployment in more real-world scenarios.



## **30. Adversarially trained neural representations may already be as robust as corresponding biological neural representations**

q-bio.NC

10 pages, 6 figures, ICML2022

**SubmitDate**: 2022-06-19    [paper-pdf](http://arxiv.org/pdf/2206.11228v1)

**Authors**: Chong Guo, Michael J. Lee, Guillaume Leclerc, Joel Dapello, Yug Rao, Aleksander Madry, James J. DiCarlo

**Abstracts**: Visual systems of primates are the gold standard of robust perception. There is thus a general belief that mimicking the neural representations that underlie those systems will yield artificial visual systems that are adversarially robust. In this work, we develop a method for performing adversarial visual attacks directly on primate brain activity. We then leverage this method to demonstrate that the above-mentioned belief might not be well founded. Specifically, we report that the biological neurons that make up visual systems of primates exhibit susceptibility to adversarial perturbations that is comparable in magnitude to existing (robustly trained) artificial neural networks.



## **31. Efficient and Transferable Adversarial Examples from Bayesian Neural Networks**

cs.LG

Accepted at UAI 2022

**SubmitDate**: 2022-06-18    [paper-pdf](http://arxiv.org/pdf/2011.05074v4)

**Authors**: Martin Gubri, Maxime Cordy, Mike Papadakis, Yves Le Traon, Koushik Sen

**Abstracts**: An established way to improve the transferability of black-box evasion attacks is to craft the adversarial examples on an ensemble-based surrogate to increase diversity. We argue that transferability is fundamentally related to uncertainty. Based on a state-of-the-art Bayesian Deep Learning technique, we propose a new method to efficiently build a surrogate by sampling approximately from the posterior distribution of neural network weights, which represents the belief about the value of each parameter. Our extensive experiments on ImageNet, CIFAR-10 and MNIST show that our approach improves the success rates of four state-of-the-art attacks significantly (up to 83.2 percentage points), in both intra-architecture and inter-architecture transferability. On ImageNet, our approach can reach 94% of success rate while reducing training computations from 11.6 to 2.4 exaflops, compared to an ensemble of independently trained DNNs. Our vanilla surrogate achieves 87.5% of the time higher transferability than three test-time techniques designed for this purpose. Our work demonstrates that the way to train a surrogate has been overlooked, although it is an important element of transfer-based attacks. We are, therefore, the first to review the effectiveness of several training methods in increasing transferability. We provide new directions to better understand the transferability phenomenon and offer a simple but strong baseline for future work.



## **32. DECK: Model Hardening for Defending Pervasive Backdoors**

cs.CR

**SubmitDate**: 2022-06-18    [paper-pdf](http://arxiv.org/pdf/2206.09272v1)

**Authors**: Guanhong Tao, Yingqi Liu, Siyuan Cheng, Shengwei An, Zhuo Zhang, Qiuling Xu, Guangyu Shen, Xiangyu Zhang

**Abstracts**: Pervasive backdoors are triggered by dynamic and pervasive input perturbations. They can be intentionally injected by attackers or naturally exist in normally trained models. They have a different nature from the traditional static and localized backdoors that can be triggered by perturbing a small input area with some fixed pattern, e.g., a patch with solid color. Existing defense techniques are highly effective for traditional backdoors. However, they may not work well for pervasive backdoors, especially regarding backdoor removal and model hardening. In this paper, we propose a novel model hardening technique against pervasive backdoors, including both natural and injected backdoors. We develop a general pervasive attack based on an encoder-decoder architecture enhanced with a special transformation layer. The attack can model a wide range of existing pervasive backdoor attacks and quantify them by class distances. As such, using the samples derived from our attack in adversarial training can harden a model against these backdoor vulnerabilities. Our evaluation on 9 datasets with 15 model structures shows that our technique can enlarge class distances by 59.65% on average with less than 1% accuracy degradation and no robustness loss, outperforming five hardening techniques such as adversarial training, universal adversarial training, MOTH, etc. It can reduce the attack success rate of six pervasive backdoor attacks from 99.06% to 1.94%, surpassing seven state-of-the-art backdoor removal techniques.



## **33. On the Role of Generalization in Transferability of Adversarial Examples**

cs.LG

**SubmitDate**: 2022-06-18    [paper-pdf](http://arxiv.org/pdf/2206.09238v1)

**Authors**: Yilin Wang, Farzan Farnia

**Abstracts**: Black-box adversarial attacks designing adversarial examples for unseen neural networks (NNs) have received great attention over the past years. While several successful black-box attack schemes have been proposed in the literature, the underlying factors driving the transferability of black-box adversarial examples still lack a thorough understanding. In this paper, we aim to demonstrate the role of the generalization properties of the substitute classifier used for generating adversarial examples in the transferability of the attack scheme to unobserved NN classifiers. To do this, we apply the max-min adversarial example game framework and show the importance of the generalization properties of the substitute NN in the success of the black-box attack scheme in application to different NN classifiers. We prove theoretical generalization bounds on the difference between the attack transferability rates on training and test samples. Our bounds suggest that a substitute NN with better generalization behavior could result in more transferable adversarial examples. In addition, we show that standard operator norm-based regularization methods could improve the transferability of the designed adversarial examples. We support our theoretical results by performing several numerical experiments showing the role of the substitute network's generalization in generating transferable adversarial examples. Our empirical results indicate the power of Lipschitz regularization methods in improving the transferability of adversarial examples.



## **34. Measuring Lower Bounds of Local Differential Privacy via Adversary Instantiations in Federated Learning**

cs.CR

15 pages, 7 figures

**SubmitDate**: 2022-06-18    [paper-pdf](http://arxiv.org/pdf/2206.09122v1)

**Authors**: Marin Matsumoto, Tsubasa Takahashi, Seng Pei Liew, Masato Oguchi

**Abstracts**: Local differential privacy (LDP) gives a strong privacy guarantee to be used in a distributed setting like federated learning (FL). LDP mechanisms in FL protect a client's gradient by randomizing it on the client; however, how can we interpret the privacy level given by the randomization? Moreover, what types of attacks can we mitigate in practice? To answer these questions, we introduce an empirical privacy test by measuring the lower bounds of LDP. The privacy test estimates how an adversary predicts if a reported randomized gradient was crafted from a raw gradient $g_1$ or $g_2$. We then instantiate six adversaries in FL under LDP to measure empirical LDP at various attack surfaces, including a worst-case attack that reaches the theoretical upper bound of LDP. The empirical privacy test with the adversary instantiations enables us to interpret LDP more intuitively and discuss relaxation of the privacy parameter until a particular instantiated attack surfaces. We also demonstrate numerical observations of the measured privacy in these adversarial settings, and the worst-case attack is not realistic in FL. In the end, we also discuss the possible relaxation of privacy levels in FL under LDP.



## **35. Existence and Minimax Theorems for Adversarial Surrogate Risks in Binary Classification**

cs.LG

37 pages, 1 Figure

**SubmitDate**: 2022-06-18    [paper-pdf](http://arxiv.org/pdf/2206.09098v1)

**Authors**: Natalie S. Frank

**Abstracts**: Adversarial training is one of the most popular methods for training methods robust to adversarial attacks, however, it is not well-understood from a theoretical perspective. We prove and existence, regularity, and minimax theorems for adversarial surrogate risks. Our results explain some empirical observations on adversarial robustness from prior work and suggest new directions in algorithm development. Furthermore, our results extend previously known existence and minimax theorems for the adversarial classification risk to surrogate risks.



## **36. Comment on Transferability and Input Transformation with Additive Noise**

cs.LG

**SubmitDate**: 2022-06-18    [paper-pdf](http://arxiv.org/pdf/2206.09075v1)

**Authors**: Hoki Kim, Jinseong Park, Jaewook Lee

**Abstracts**: Adversarial attacks have verified the existence of the vulnerability of neural networks. By adding small perturbations to a benign example, adversarial attacks successfully generate adversarial examples that lead misclassification of deep learning models. More importantly, an adversarial example generated from a specific model can also deceive other models without modification. We call this phenomenon ``transferability". Here, we analyze the relationship between transferability and input transformation with additive noise by mathematically proving that the modified optimization can produce more transferable adversarial examples.



## **37. Learning Generative Deception Strategies in Combinatorial Masking Games**

cs.GT

GameSec 2021

**SubmitDate**: 2022-06-17    [paper-pdf](http://arxiv.org/pdf/2109.11637v2)

**Authors**: Junlin Wu, Charles Kamhoua, Murat Kantarcioglu, Yevgeniy Vorobeychik

**Abstracts**: Deception is a crucial tool in the cyberdefence repertoire, enabling defenders to leverage their informational advantage to reduce the likelihood of successful attacks. One way deception can be employed is through obscuring, or masking, some of the information about how systems are configured, increasing attacker's uncertainty about their targets. We present a novel game-theoretic model of the resulting defender-attacker interaction, where the defender chooses a subset of attributes to mask, while the attacker responds by choosing an exploit to execute. The strategies of both players have combinatorial structure with complex informational dependencies, and therefore even representing these strategies is not trivial. First, we show that the problem of computing an equilibrium of the resulting zero-sum defender-attacker game can be represented as a linear program with a combinatorial number of system configuration variables and constraints, and develop a constraint generation approach for solving this problem. Next, we present a novel highly scalable approach for approximately solving such games by representing the strategies of both players as neural networks. The key idea is to represent the defender's mixed strategy using a deep neural network generator, and then using alternating gradient-descent-ascent algorithm, analogous to the training of Generative Adversarial Networks. Our experiments, as well as a case study, demonstrate the efficacy of the proposed approach.



## **38. RetrievalGuard: Provably Robust 1-Nearest Neighbor Image Retrieval**

cs.IR

accepted by ICML 2022

**SubmitDate**: 2022-06-17    [paper-pdf](http://arxiv.org/pdf/2206.11225v1)

**Authors**: Yihan Wu, Hongyang Zhang, Heng Huang

**Abstracts**: Recent research works have shown that image retrieval models are vulnerable to adversarial attacks, where slightly modified test inputs could lead to problematic retrieval results. In this paper, we aim to design a provably robust image retrieval model which keeps the most important evaluation metric Recall@1 invariant to adversarial perturbation. We propose the first 1-nearest neighbor (NN) image retrieval algorithm, RetrievalGuard, which is provably robust against adversarial perturbations within an $\ell_2$ ball of calculable radius. The challenge is to design a provably robust algorithm that takes into consideration the 1-NN search and the high-dimensional nature of the embedding space. Algorithmically, given a base retrieval model and a query sample, we build a smoothed retrieval model by carefully analyzing the 1-NN search procedure in the high-dimensional embedding space. We show that the smoothed retrieval model has bounded Lipschitz constant and thus the retrieval score is invariant to $\ell_2$ adversarial perturbations. Experiments on image retrieval tasks validate the robustness of our RetrievalGuard method.



## **39. Is Multi-Modal Necessarily Better? Robustness Evaluation of Multi-modal Fake News Detection**

cs.AI

**SubmitDate**: 2022-06-17    [paper-pdf](http://arxiv.org/pdf/2206.08788v1)

**Authors**: Jinyin Chen, Chengyu Jia, Haibin Zheng, Ruoxi Chen, Chenbo Fu

**Abstracts**: The proliferation of fake news and its serious negative social influence push fake news detection methods to become necessary tools for web managers. Meanwhile, the multi-media nature of social media makes multi-modal fake news detection popular for its ability to capture more modal features than uni-modal detection methods. However, current literature on multi-modal detection is more likely to pursue the detection accuracy but ignore the robustness of the detector. To address this problem, we propose a comprehensive robustness evaluation of multi-modal fake news detectors. In this work, we simulate the attack methods of malicious users and developers, i.e., posting fake news and injecting backdoors. Specifically, we evaluate multi-modal detectors with five adversarial and two backdoor attack methods. Experiment results imply that: (1) The detection performance of the state-of-the-art detectors degrades significantly under adversarial attacks, even worse than general detectors; (2) Most multi-modal detectors are more vulnerable when subjected to attacks on visual modality than textual modality; (3) Popular events' images will cause significant degradation to the detectors when they are subjected to backdoor attacks; (4) The performance of these detectors under multi-modal attacks is worse than under uni-modal attacks; (5) Defensive methods will improve the robustness of the multi-modal detectors.



## **40. Detecting Adversarial Examples in Batches -- a geometrical approach**

cs.LG

Submitted to AdvML workshop at ICML2022

**SubmitDate**: 2022-06-17    [paper-pdf](http://arxiv.org/pdf/2206.08738v1)

**Authors**: Danush Kumar Venkatesh, Peter Steinbach

**Abstracts**: Many deep learning methods have successfully solved complex tasks in computer vision and speech recognition applications. Nonetheless, the robustness of these models has been found to be vulnerable to perturbed inputs or adversarial examples, which are imperceptible to the human eye, but lead the model to erroneous output decisions. In this study, we adapt and introduce two geometric metrics, density and coverage, and evaluate their use in detecting adversarial samples in batches of unseen data. We empirically study these metrics using MNIST and two real-world biomedical datasets from MedMNIST, subjected to two different adversarial attacks. Our experiments show promising results for both metrics to detect adversarial examples. We believe that his work can lay the ground for further study on these metrics' use in deployed machine learning systems to monitor for possible attacks by adversarial examples or related pathologies such as dataset shift.



## **41. Minimum Noticeable Difference based Adversarial Privacy Preserving Image Generation**

cs.CV

**SubmitDate**: 2022-06-17    [paper-pdf](http://arxiv.org/pdf/2206.08638v1)

**Authors**: Wen Sun, Jian Jin, Weisi Lin

**Abstracts**: Deep learning models are found to be vulnerable to adversarial examples, as wrong predictions can be caused by small perturbation in input for deep learning models. Most of the existing works of adversarial image generation try to achieve attacks for most models, while few of them make efforts on guaranteeing the perceptual quality of the adversarial examples. High quality adversarial examples matter for many applications, especially for the privacy preserving. In this work, we develop a framework based on the Minimum Noticeable Difference (MND) concept to generate adversarial privacy preserving images that have minimum perceptual difference from the clean ones but are able to attack deep learning models. To achieve this, an adversarial loss is firstly proposed to make the deep learning models attacked by the adversarial images successfully. Then, a perceptual quality-preserving loss is developed by taking the magnitude of perturbation and perturbation-caused structural and gradient changes into account, which aims to preserve high perceptual quality for adversarial image generation. To the best of our knowledge, this is the first work on exploring quality-preserving adversarial image generation based on the MND concept for privacy preserving. To evaluate its performance in terms of perceptual quality, the deep models on image classification and face recognition are tested with the proposed method and several anchor methods in this work. Extensive experimental results demonstrate that the proposed MND framework is capable of generating adversarial images with remarkably improved performance metrics (e.g., PSNR, SSIM, and MOS) than that generated with the anchor methods.



## **42. Query-Efficient and Scalable Black-Box Adversarial Attacks on Discrete Sequential Data via Bayesian Optimization**

cs.LG

ICML 2022; Codes at  https://github.com/snu-mllab/DiscreteBlockBayesAttack

**SubmitDate**: 2022-06-17    [paper-pdf](http://arxiv.org/pdf/2206.08575v1)

**Authors**: Deokjae Lee, Seungyong Moon, Junhyeok Lee, Hyun Oh Song

**Abstracts**: We focus on the problem of adversarial attacks against models on discrete sequential data in the black-box setting where the attacker aims to craft adversarial examples with limited query access to the victim model. Existing black-box attacks, mostly based on greedy algorithms, find adversarial examples using pre-computed key positions to perturb, which severely limits the search space and might result in suboptimal solutions. To this end, we propose a query-efficient black-box attack using Bayesian optimization, which dynamically computes important positions using an automatic relevance determination (ARD) categorical kernel. We introduce block decomposition and history subsampling techniques to improve the scalability of Bayesian optimization when an input sequence becomes long. Moreover, we develop a post-optimization algorithm that finds adversarial examples with smaller perturbation size. Experiments on natural language and protein classification tasks demonstrate that our method consistently achieves higher attack success rate with significant reduction in query count and modification rate compared to the previous state-of-the-art methods.



## **43. Adversarial Attack and Defense for Non-Parametric Two-Sample Tests**

cs.LG

Accepted by ICML 2022

**SubmitDate**: 2022-06-17    [paper-pdf](http://arxiv.org/pdf/2202.03077v2)

**Authors**: Xilie Xu, Jingfeng Zhang, Feng Liu, Masashi Sugiyama, Mohan Kankanhalli

**Abstracts**: Non-parametric two-sample tests (TSTs) that judge whether two sets of samples are drawn from the same distribution, have been widely used in the analysis of critical data. People tend to employ TSTs as trusted basic tools and rarely have any doubt about their reliability. This paper systematically uncovers the failure mode of non-parametric TSTs through adversarial attacks and then proposes corresponding defense strategies. First, we theoretically show that an adversary can upper-bound the distributional shift which guarantees the attack's invisibility. Furthermore, we theoretically find that the adversary can also degrade the lower bound of a TST's test power, which enables us to iteratively minimize the test criterion in order to search for adversarial pairs. To enable TST-agnostic attacks, we propose an ensemble attack (EA) framework that jointly minimizes the different types of test criteria. Second, to robustify TSTs, we propose a max-min optimization that iteratively generates adversarial pairs to train the deep kernels. Extensive experiments on both simulated and real-world datasets validate the adversarial vulnerabilities of non-parametric TSTs and the effectiveness of our proposed defense. Source code is available at https://github.com/GodXuxilie/Robust-TST.git.



## **44. A Unified Evaluation of Textual Backdoor Learning: Frameworks and Benchmarks**

cs.LG

19 pages

**SubmitDate**: 2022-06-17    [paper-pdf](http://arxiv.org/pdf/2206.08514v1)

**Authors**: Ganqu Cui, Lifan Yuan, Bingxiang He, Yangyi Chen, Zhiyuan Liu, Maosong Sun

**Abstracts**: Textual backdoor attacks are a kind of practical threat to NLP systems. By injecting a backdoor in the training phase, the adversary could control model predictions via predefined triggers. As various attack and defense models have been proposed, it is of great significance to perform rigorous evaluations. However, we highlight two issues in previous backdoor learning evaluations: (1) The differences between real-world scenarios (e.g. releasing poisoned datasets or models) are neglected, and we argue that each scenario has its own constraints and concerns, thus requires specific evaluation protocols; (2) The evaluation metrics only consider whether the attacks could flip the models' predictions on poisoned samples and retain performances on benign samples, but ignore that poisoned samples should also be stealthy and semantic-preserving. To address these issues, we categorize existing works into three practical scenarios in which attackers release datasets, pre-trained models, and fine-tuned models respectively, then discuss their unique evaluation methodologies. On metrics, to completely evaluate poisoned samples, we use grammar error increase and perplexity difference for stealthiness, along with text similarity for validity. After formalizing the frameworks, we develop an open-source toolkit OpenBackdoor to foster the implementations and evaluations of textual backdoor learning. With this toolkit, we perform extensive experiments to benchmark attack and defense models under the suggested paradigm. To facilitate the underexplored defenses against poisoned datasets, we further propose CUBE, a simple yet strong clustering-based defense baseline. We hope that our frameworks and benchmarks could serve as the cornerstones for future model development and evaluations.



## **45. I Know What You Trained Last Summer: A Survey on Stealing Machine Learning Models and Defences**

cs.LG

Under review at ACM Computing Surveys

**SubmitDate**: 2022-06-16    [paper-pdf](http://arxiv.org/pdf/2206.08451v1)

**Authors**: Daryna Oliynyk, Rudolf Mayer, Andreas Rauber

**Abstracts**: Machine Learning-as-a-Service (MLaaS) has become a widespread paradigm, making even the most complex machine learning models available for clients via e.g. a pay-per-query principle. This allows users to avoid time-consuming processes of data collection, hyperparameter tuning, and model training. However, by giving their customers access to the (predictions of their) models, MLaaS providers endanger their intellectual property, such as sensitive training data, optimised hyperparameters, or learned model parameters. Adversaries can create a copy of the model with (almost) identical behavior using the the prediction labels only. While many variants of this attack have been described, only scattered defence strategies have been proposed, addressing isolated threats. This raises the necessity for a thorough systematisation of the field of model stealing, to arrive at a comprehensive understanding why these attacks are successful, and how they could be holistically defended against. We address this by categorising and comparing model stealing attacks, assessing their performance, and exploring corresponding defence techniques in different settings. We propose a taxonomy for attack and defence approaches, and provide guidelines on how to select the right attack or defence strategy based on the goal and available resources. Finally, we analyse which defences are rendered less effective by current attack strategies.



## **46. Boosting the Adversarial Transferability of Surrogate Model with Dark Knowledge**

cs.LG

26 pages, 5 figures

**SubmitDate**: 2022-06-16    [paper-pdf](http://arxiv.org/pdf/2206.08316v1)

**Authors**: Dingcheng Yang, Zihao Xiao, Wenjian Yu

**Abstracts**: Deep neural networks (DNNs) for image classification are known to be vulnerable to adversarial examples. And, the adversarial examples have transferability, which means an adversarial example for a DNN model can fool another black-box model with a non-trivial probability. This gave birth of the transfer-based adversarial attack where the adversarial examples generated by a pretrained or known model (called surrogate model) are used to conduct black-box attack. There are some work on how to generate the adversarial examples from a given surrogate model to achieve better transferability. However, training a special surrogate model to generate adversarial examples with better transferability is relatively under-explored. In this paper, we propose a method of training a surrogate model with abundant dark knowledge to boost the adversarial transferability of the adversarial examples generated by the surrogate model. This trained surrogate model is named dark surrogate model (DSM), and the proposed method to train DSM consists of two key components: a teacher model extracting dark knowledge and providing soft labels, and the mixing augmentation skill which enhances the dark knowledge of training data. Extensive experiments have been conducted to show that the proposed method can substantially improve the adversarial transferability of surrogate model across different architectures of surrogate model and optimizers for generating adversarial examples. We also show that the proposed method can be applied to other scenarios of transfer-based attack that contain dark knowledge, like face verification.



## **47. Adversarial Patch Attacks and Defences in Vision-Based Tasks: A Survey**

cs.CV

A. Sharma and Y. Bian share equal contribution

**SubmitDate**: 2022-06-16    [paper-pdf](http://arxiv.org/pdf/2206.08304v1)

**Authors**: Abhijith Sharma, Yijun Bian, Phil Munz, Apurva Narayan

**Abstracts**: Adversarial attacks in deep learning models, especially for safety-critical systems, are gaining more and more attention in recent years, due to the lack of trust in the security and robustness of AI models. Yet the more primitive adversarial attacks might be physically infeasible or require some resources that are hard to access like the training data, which motivated the emergence of patch attacks. In this survey, we provide a comprehensive overview to cover existing techniques of adversarial patch attacks, aiming to help interested researchers quickly catch up with the progress in this field. We also discuss existing techniques for developing detection and defences against adversarial patches, aiming to help the community better understand this field and its applications in the real world.



## **48. Adversarial Robustness of Graph-based Anomaly Detection**

cs.CR

arXiv admin note: substantial text overlap with arXiv:2106.09989

**SubmitDate**: 2022-06-16    [paper-pdf](http://arxiv.org/pdf/2206.08260v1)

**Authors**: Yulin Zhu, Yuni Lai, Kaifa Zhao, Xiapu Luo, Mingquan Yuan, Jian Ren, Kai Zhou

**Abstracts**: Graph-based anomaly detection is becoming prevalent due to the powerful representation abilities of graphs as well as recent advances in graph mining techniques. These GAD tools, however, expose a new attacking surface, ironically due to their unique advantage of being able to exploit the relations among data. That is, attackers now can manipulate those relations (i.e., the structure of the graph) to allow target nodes to evade detection or degenerate the classification performance of the detection. In this paper, we exploit this vulnerability by designing the structural poisoning attacks to a FeXtra-based GAD system termed OddBall as well as the black box attacks against GCN-based GAD systems by attacking the imbalanced lienarized GCN ( LGCN ). Specifically, we formulate the attack against OddBall and LGCN as a one-level optimization problem by incorporating different regression techniques, where the key technical challenge is to efficiently solve the problem in a discrete domain. We propose a novel attack method termed BinarizedAttack based on gradient descent. Comparing to prior arts, BinarizedAttack can better use the gradient information, making it particularly suitable for solving discrete optimization problems, thus opening the door to studying a new type of attack against security analytic tools that rely on graph data.



## **49. Adversarial attacks on voter model dynamics in complex networks**

physics.soc-ph

7 pages, 5 figures

**SubmitDate**: 2022-06-16    [paper-pdf](http://arxiv.org/pdf/2111.09561v2)

**Authors**: Katsumi Chiyomaru, Kazuhiro Takemoto

**Abstracts**: This study investigates adversarial attacks conducted to distort voter model dynamics in complex networks. Specifically, a simple adversarial attack method is proposed to hold the state of opinions of an individual closer to the target state in the voter model dynamics. This indicates that even when one opinion is the majority, the vote outcome can be inverted (i.e., the outcome can lean toward the other opinion) by adding extremely small (hard-to-detect) perturbations strategically generated in social networks. Adversarial attacks are relatively more effective in complex (large and dense) networks. These results indicate that opinion dynamics can be unknowingly distorted.



## **50. Adversarial Privacy Protection on Speech Enhancement**

cs.SD

5 pages, 6 figures

**SubmitDate**: 2022-06-16    [paper-pdf](http://arxiv.org/pdf/2206.08170v1)

**Authors**: Mingyu Dong, Diqun Yan, Rangding Wang

**Abstracts**: Speech is easily leaked imperceptibly, such as being recorded by mobile phones in different situations. Private content in speech may be maliciously extracted through speech enhancement technology. Speech enhancement technology has developed rapidly along with deep neural networks (DNNs), but adversarial examples can cause DNNs to fail. In this work, we propose an adversarial method to degrade speech enhancement systems. Experimental results show that generated adversarial examples can erase most content information in original examples or replace it with target speech content through speech enhancement. The word error rate (WER) between an enhanced original example and enhanced adversarial example recognition result can reach 89.0%. WER of target attack between enhanced adversarial example and target example is low to 33.75% . Adversarial perturbation can bring the rate of change to the original example to more than 1.4430. This work can prevent the malicious extraction of speech.



