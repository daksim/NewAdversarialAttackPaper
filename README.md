# Latest Adversarial Attack Papers
**update at 2021-12-28 10:24:58**

[中文版本](https://github.com/daksim/NewAdversarialAttackPaper/blob/main/README_CN.md)

## **1. PORTFILER: Port-Level Network Profiling for Self-Propagating Malware Detection**

cs.CR

An earlier version is accepted to be published in IEEE Conference on  Communications and Network Security (CNS) 2021

**SubmitDate**: 2021-12-27    [paper-pdf](http://arxiv.org/pdf/2112.13798v1)

**Authors**: Talha Ongun, Oliver Spohngellert, Benjamin Miller, Simona Boboila, Alina Oprea, Tina Eliassi-Rad, Jason Hiser, Alastair Nottingham, Jack Davidson, Malathi Veeraraghavan

**Abstracts**: Recent self-propagating malware (SPM) campaigns compromised hundred of thousands of victim machines on the Internet. It is challenging to detect these attacks in their early stages, as adversaries utilize common network services, use novel techniques, and can evade existing detection mechanisms. We propose PORTFILER (PORT-Level Network Traffic ProFILER), a new machine learning system applied to network traffic for detecting SPM attacks. PORTFILER extracts port-level features from the Zeek connection logs collected at a border of a monitored network, applies anomaly detection techniques to identify suspicious events, and ranks the alerts across ports for investigation by the Security Operations Center (SOC). We propose a novel ensemble methodology for aggregating individual models in PORTFILER that increases resilience against several evasion strategies compared to standard ML baselines. We extensively evaluate PORTFILER on traffic collected from two university networks, and show that it can detect SPM attacks with different patterns, such as WannaCry and Mirai, and performs well under evasion. Ranking across ports achieves precision over 0.94 with low false positive rates in the top ranked alerts. When deployed on the university networks, PORTFILER detected anomalous SPM-like activity on one of the campus networks, confirmed by the university SOC as malicious. PORTFILER also detected a Mirai attack recreated on the two university networks with higher precision and recall than deep-learning-based autoencoder methods.



## **2. Adversarial Attack for Asynchronous Event-based Data**

cs.CV

8 pages, 6 figures, Thirty-Sixth AAAI Conference on Artificial  Intelligence (AAAI-22)

**SubmitDate**: 2021-12-27    [paper-pdf](http://arxiv.org/pdf/2112.13534v1)

**Authors**: Wooju Lee, Hyun Myung

**Abstracts**: Deep neural networks (DNNs) are vulnerable to adversarial examples that are carefully designed to cause the deep learning model to make mistakes. Adversarial examples of 2D images and 3D point clouds have been extensively studied, but studies on event-based data are limited. Event-based data can be an alternative to a 2D image under high-speed movements, such as autonomous driving. However, the given adversarial events make the current deep learning model vulnerable to safety issues. In this work, we generate adversarial examples and then train the robust models for event-based data, for the first time. Our algorithm shifts the time of the original events and generates additional adversarial events. Additional adversarial events are generated in two stages. First, null events are added to the event-based data to generate additional adversarial events. The perturbation size can be controlled with the number of null events. Second, the location and time of additional adversarial events are set to mislead DNNs in a gradient-based attack. Our algorithm achieves an attack success rate of 97.95\% on the N-Caltech101 dataset. Furthermore, the adversarial training model improves robustness on the adversarial event data compared to the original model.



## **3. Killing One Bird with Two Stones: Model Extraction and Attribute Inference Attacks against BERT-based APIs**

cs.CR

**SubmitDate**: 2021-12-26    [paper-pdf](http://arxiv.org/pdf/2105.10909v2)

**Authors**: Chen Chen, Xuanli He, Lingjuan Lyu, Fangzhao Wu

**Abstracts**: The collection and availability of big data, combined with advances in pre-trained models (e.g., BERT, XLNET, etc), have revolutionized the predictive performance of modern natural language processing tasks, ranging from text classification to text generation. This allows corporations to provide machine learning as a service (MLaaS) by encapsulating fine-tuned BERT-based models as APIs. However, BERT-based APIs have exhibited a series of security and privacy vulnerabilities. For example, prior work has exploited the security issues of the BERT-based APIs through the adversarial examples crafted by the extracted model. However, the privacy leakage problems of the BERT-based APIs through the extracted model have not been well studied. On the other hand, due to the high capacity of BERT-based APIs, the fine-tuned model is easy to be overlearned, but what kind of information can be leaked from the extracted model remains unknown. In this work, we bridge this gap by first presenting an effective model extraction attack, where the adversary can practically steal a BERT-based API (the target/victim model) by only querying a limited number of queries. We further develop an effective attribute inference attack which can infer the sensitive attribute of the training data used by the BERT-based APIs. Our extensive experiments on benchmark datasets under various realistic settings validate the potential vulnerabilities of BERT-based APIs. Moreover, we demonstrate that two promising defense methods become ineffective against our attacks, which calls for more effective defense methods.



## **4. Task and Model Agnostic Adversarial Attack on Graph Neural Networks**

cs.LG

**SubmitDate**: 2021-12-25    [paper-pdf](http://arxiv.org/pdf/2112.13267v1)

**Authors**: Kartik Sharma, Samidha Verma, Sourav Medya, Sayan Ranu, Arnab Bhattacharya

**Abstracts**: Graph neural networks (GNNs) have witnessed significant adoption in the industry owing to impressive performance on various predictive tasks. Performance alone, however, is not enough. Any widely deployed machine learning algorithm must be robust to adversarial attacks. In this work, we investigate this aspect for GNNs, identify vulnerabilities, and link them to graph properties that may potentially lead to the development of more secure and robust GNNs. Specifically, we formulate the problem of task and model agnostic evasion attacks where adversaries modify the test graph to affect the performance of any unknown downstream task. The proposed algorithm, GRAND ($Gr$aph $A$ttack via $N$eighborhood $D$istortion) shows that distortion of node neighborhoods is effective in drastically compromising prediction performance. Although neighborhood distortion is an NP-hard problem, GRAND designs an effective heuristic through a novel combination of Graph Isomorphism Network with deep $Q$-learning. Extensive experiments on real datasets show that, on average, GRAND is up to $50\%$ more effective than state of the art techniques, while being more than $100$ times faster.



## **5. Denoised Internal Models: a Brain-Inspired Autoencoder against Adversarial Attacks**

cs.CV

16 pages, 3 figures

**SubmitDate**: 2021-12-25    [paper-pdf](http://arxiv.org/pdf/2111.10844v3)

**Authors**: Kaiyuan Liu, Xingyu Li, Yurui Lai, Ge Zhang, Hang Su, Jiachen Wang, Chunxu Guo, Jisong Guan, Yi Zhou

**Abstracts**: Despite its great success, deep learning severely suffers from robustness; that is, deep neural networks are very vulnerable to adversarial attacks, even the simplest ones. Inspired by recent advances in brain science, we propose the Denoised Internal Models (DIM), a novel generative autoencoder-based model to tackle this challenge. Simulating the pipeline in the human brain for visual signal processing, DIM adopts a two-stage approach. In the first stage, DIM uses a denoiser to reduce the noise and the dimensions of inputs, reflecting the information pre-processing in the thalamus. Inspired from the sparse coding of memory-related traces in the primary visual cortex, the second stage produces a set of internal models, one for each category. We evaluate DIM over 42 adversarial attacks, showing that DIM effectively defenses against all the attacks and outperforms the SOTA on the overall robustness.



## **6. Stealthy Attack on Algorithmic-Protected DNNs via Smart Bit Flipping**

cs.CR

Accepted for the 23rd International Symposium on Quality Electronic  Design (ISQED'22)

**SubmitDate**: 2021-12-25    [paper-pdf](http://arxiv.org/pdf/2112.13162v1)

**Authors**: Behnam Ghavami, Seyd Movi, Zhenman Fang, Lesley Shannon

**Abstracts**: Recently, deep neural networks (DNNs) have been deployed in safety-critical systems such as autonomous vehicles and medical devices. Shortly after that, the vulnerability of DNNs were revealed by stealthy adversarial examples where crafted inputs -- by adding tiny perturbations to original inputs -- can lead a DNN to generate misclassification outputs. To improve the robustness of DNNs, some algorithmic-based countermeasures against adversarial examples have been introduced thereafter.   In this paper, we propose a new type of stealthy attack on protected DNNs to circumvent the algorithmic defenses: via smart bit flipping in DNN weights, we can reserve the classification accuracy for clean inputs but misclassify crafted inputs even with algorithmic countermeasures. To fool protected DNNs in a stealthy way, we introduce a novel method to efficiently find their most vulnerable weights and flip those bits in hardware. Experimental results show that we can successfully apply our stealthy attack against state-of-the-art algorithmic-protected DNNs.



## **7. SoK: A Study of the Security on Voice Processing Systems**

cs.CR

**SubmitDate**: 2021-12-24    [paper-pdf](http://arxiv.org/pdf/2112.13144v1)

**Authors**: Robert Chang, Logan Kuo, Arthur Liu, Nader Sehatbakhsh

**Abstracts**: As the use of Voice Processing Systems (VPS) continues to become more prevalent in our daily lives through the increased reliance on applications such as commercial voice recognition devices as well as major text-to-speech software, the attacks on these systems are increasingly complex, varied, and constantly evolving. With the use cases for VPS rapidly growing into new spaces and purposes, the potential consequences regarding privacy are increasingly more dangerous. In addition, the growing number and increased practicality of over-the-air attacks have made system failures much more probable. In this paper, we will identify and classify an arrangement of unique attacks on voice processing systems. Over the years research has been moving from specialized, untargeted attacks that result in the malfunction of systems and the denial of services to more general, targeted attacks that can force an outcome controlled by an adversary. The current and most frequently used machine learning systems and deep neural networks, which are at the core of modern voice processing systems, were built with a focus on performance and scalability rather than security. Therefore, it is critical for us to reassess the developing voice processing landscape and to identify the state of current attacks and defenses so that we may suggest future developments and theoretical improvements.



## **8. CatchBackdoor: Backdoor Testing by Critical Trojan Neural Path Identification via Differential Fuzzing**

cs.CR

13 pages

**SubmitDate**: 2021-12-24    [paper-pdf](http://arxiv.org/pdf/2112.13064v1)

**Authors**: Haibo Jin, Ruoxi Chen, Jinyin Chen, Yao Cheng, Chong Fu, Ting Wang, Yue Yu, Zhaoyan Ming

**Abstracts**: The success of deep neural networks (DNNs) in real-world applications has benefited from abundant pre-trained models. However, the backdoored pre-trained models can pose a significant trojan threat to the deployment of downstream DNNs. Existing DNN testing methods are mainly designed to find incorrect corner case behaviors in adversarial settings but fail to discover the backdoors crafted by strong trojan attacks. Observing the trojan network behaviors shows that they are not just reflected by a single compromised neuron as proposed by previous work but attributed to the critical neural paths in the activation intensity and frequency of multiple neurons. This work formulates the DNN backdoor testing and proposes the CatchBackdoor framework. Via differential fuzzing of critical neurons from a small number of benign examples, we identify the trojan paths and particularly the critical ones, and generate backdoor testing examples by simulating the critical neurons in the identified paths. Extensive experiments demonstrate the superiority of CatchBackdoor, with higher detection performance than existing methods. CatchBackdoor works better on detecting backdoors by stealthy blending and adaptive attacks, which existing methods fail to detect. Moreover, our experiments show that CatchBackdoor may reveal the potential backdoors of models in Model Zoo.



## **9. NIP: Neuron-level Inverse Perturbation Against Adversarial Attacks**

cs.CV

14 pages

**SubmitDate**: 2021-12-24    [paper-pdf](http://arxiv.org/pdf/2112.13060v1)

**Authors**: Ruoxi Chen, Haibo Jin, Jinyin Chen, Haibin Zheng, Yue Yu, Shouling Ji

**Abstracts**: Although deep learning models have achieved unprecedented success, their vulnerabilities towards adversarial attacks have attracted increasing attention, especially when deployed in security-critical domains. To address the challenge, numerous defense strategies, including reactive and proactive ones, have been proposed for robustness improvement. From the perspective of image feature space, some of them cannot reach satisfying results due to the shift of features. Besides, features learned by models are not directly related to classification results. Different from them, We consider defense method essentially from model inside and investigated the neuron behaviors before and after attacks. We observed that attacks mislead the model by dramatically changing the neurons that contribute most and least to the correct label. Motivated by it, we introduce the concept of neuron influence and further divide neurons into front, middle and tail part. Based on it, we propose neuron-level inverse perturbation(NIP), the first neuron-level reactive defense method against adversarial attacks. By strengthening front neurons and weakening those in the tail part, NIP can eliminate nearly all adversarial perturbations while still maintaining high benign accuracy. Besides, it can cope with different sizes of perturbations via adaptivity, especially larger ones. Comprehensive experiments conducted on three datasets and six models show that NIP outperforms the state-of-the-art baselines against eleven adversarial attacks. We further provide interpretable proofs via neuron activation and visualization for better understanding.



## **10. One Bad Apple Spoils the Bunch: Transaction DoS in MimbleWimble Blockchains**

cs.CR

9 pages, 4 figures

**SubmitDate**: 2021-12-24    [paper-pdf](http://arxiv.org/pdf/2112.13009v1)

**Authors**: Seyed Ali Tabatabaee, Charlene Nicer, Ivan Beschastnikh, Chen Feng

**Abstracts**: As adoption of blockchain-based systems grows, more attention is being given to privacy of these systems. Early systems like BitCoin provided few privacy features. As a result, systems with strong privacy guarantees, including Monero, Zcash, and MimbleWimble have been developed. Compared to BitCoin, these cryptocurrencies are much less understood. In this paper, we focus on MimbleWimble, which uses the Dandelion++ protocol for private transaction relay and transaction aggregation to provide transaction content privacy. We find that in combination these two features make MimbleWimble susceptible to a new type of denial-of-service attacks. We design, prototype, and evaluate this attack on the Beam network using a private test network and a network simulator. We find that by controlling only 10% of the network nodes, the adversary can prevent over 45% of all transactions from ending up in the blockchain. We also discuss several potential approaches for mitigating this attack.



## **11. Parameter identifiability of a deep feedforward ReLU neural network**

math.ST

**SubmitDate**: 2021-12-24    [paper-pdf](http://arxiv.org/pdf/2112.12982v1)

**Authors**: Joachim Bona-Pellissier, François Bachoc, François Malgouyres

**Abstracts**: The possibility for one to recover the parameters-weights and biases-of a neural network thanks to the knowledge of its function on a subset of the input space can be, depending on the situation, a curse or a blessing. On one hand, recovering the parameters allows for better adversarial attacks and could also disclose sensitive information from the dataset used to construct the network. On the other hand, if the parameters of a network can be recovered, it guarantees the user that the features in the latent spaces can be interpreted. It also provides foundations to obtain formal guarantees on the performances of the network. It is therefore important to characterize the networks whose parameters can be identified and those whose parameters cannot. In this article, we provide a set of conditions on a deep fully-connected feedforward ReLU neural network under which the parameters of the network are uniquely identified-modulo permutation and positive rescaling-from the function it implements on a subset of the input space.



## **12. Revisiting and Advancing Fast Adversarial Training Through The Lens of Bi-Level Optimization**

cs.LG

**SubmitDate**: 2021-12-24    [paper-pdf](http://arxiv.org/pdf/2112.12376v2)

**Authors**: Yihua Zhang, Guanhua Zhang, Prashant Khanduri, Mingyi Hong, Shiyu Chang, Sijia Liu

**Abstracts**: Adversarial training (AT) has become a widely recognized defense mechanism to improve the robustness of deep neural networks against adversarial attacks. It solves a min-max optimization problem, where the minimizer (i.e., defender) seeks a robust model to minimize the worst-case training loss in the presence of adversarial examples crafted by the maximizer (i.e., attacker). However, the min-max nature makes AT computationally intensive and thus difficult to scale. Meanwhile, the FAST-AT algorithm, and in fact many recent algorithms that improve AT, simplify the min-max based AT by replacing its maximization step with the simple one-shot gradient sign based attack generation step. Although easy to implement, FAST-AT lacks theoretical guarantees, and its practical performance can be unsatisfactory, suffering from the robustness catastrophic overfitting when training with strong adversaries.   In this paper, we propose to design FAST-AT from the perspective of bi-level optimization (BLO). We first make the key observation that the most commonly-used algorithmic specification of FAST-AT is equivalent to using some gradient descent-type algorithm to solve a bi-level problem involving a sign operation. However, the discrete nature of the sign operation makes it difficult to understand the algorithm performance. Based on the above observation, we propose a new tractable bi-level optimization problem, design and analyze a new set of algorithms termed Fast Bi-level AT (FAST-BAT). FAST-BAT is capable of defending sign-based projected gradient descent (PGD) attacks without calling any gradient sign method and explicit robust regularization. Furthermore, we empirically show that our method outperforms state-of-the-art FAST-AT baselines, by achieving superior model robustness without inducing robustness catastrophic overfitting, or suffering from any loss of standard accuracy.



## **13. Robust Secretary and Prophet Algorithms for Packing Integer Programs**

cs.DS

Appears in SODA 2022

**SubmitDate**: 2021-12-24    [paper-pdf](http://arxiv.org/pdf/2112.12920v1)

**Authors**: C. J. Argue, Anupam Gupta, Marco Molinaro, Sahil Singla

**Abstracts**: We study the problem of solving Packing Integer Programs (PIPs) in the online setting, where columns in $[0,1]^d$ of the constraint matrix are revealed sequentially, and the goal is to pick a subset of the columns that sum to at most $B$ in each coordinate while maximizing the objective. Excellent results are known in the secretary setting, where the columns are adversarially chosen, but presented in a uniformly random order. However, these existing algorithms are susceptible to adversarial attacks: they try to "learn" characteristics of a good solution, but tend to over-fit to the model, and hence a small number of adversarial corruptions can cause the algorithm to fail.   In this paper, we give the first robust algorithms for Packing Integer Programs, specifically in the recently proposed Byzantine Secretary framework. Our techniques are based on a two-level use of online learning, to robustly learn an approximation to the optimal value, and then to use this robust estimate to pick a good solution. These techniques are general and we use them to design robust algorithms for PIPs in the prophet model as well, specifically in the Prophet-with-Augmentations framework. We also improve known results in the Byzantine Secretary framework: we make the non-constructive results algorithmic and improve the existing bounds for single-item and matroid constraints.



## **14. Adaptive Modeling Against Adversarial Attacks**

cs.LG

10 pages, 3 figures

**SubmitDate**: 2021-12-23    [paper-pdf](http://arxiv.org/pdf/2112.12431v1)

**Authors**: Zhiwen Yan, Teck Khim Ng

**Abstracts**: Adversarial training, the process of training a deep learning model with adversarial data, is one of the most successful adversarial defense methods for deep learning models. We have found that the robustness to white-box attack of an adversarially trained model can be further improved if we fine tune this model in inference stage to adapt to the adversarial input, with the extra information in it. We introduce an algorithm that "post trains" the model at inference stage between the original output class and a "neighbor" class, with existing training data. The accuracy of pre-trained Fast-FGSM CIFAR10 classifier base model against white-box projected gradient attack (PGD) can be significantly improved from 46.8% to 64.5% with our algorithm.



## **15. Adversarial Attacks against Windows PE Malware Detection: A Survey of the State-of-the-Art**

cs.CR

**SubmitDate**: 2021-12-23    [paper-pdf](http://arxiv.org/pdf/2112.12310v1)

**Authors**: Xiang Ling, Lingfei Wu, Jiangyu Zhang, Zhenqing Qu, Wei Deng, Xiang Chen, Chunming Wu, Shouling Ji, Tianyue Luo, Jingzheng Wu, Yanjun Wu

**Abstracts**: The malware has been being one of the most damaging threats to computers that span across multiple operating systems and various file formats. To defend against the ever-increasing and ever-evolving threats of malware, tremendous efforts have been made to propose a variety of malware detection methods that attempt to effectively and efficiently detect malware. Recent studies have shown that, on the one hand, existing ML and DL enable the superior detection of newly emerging and previously unseen malware. However, on the other hand, ML and DL models are inherently vulnerable to adversarial attacks in the form of adversarial examples, which are maliciously generated by slightly and carefully perturbing the legitimate inputs to confuse the targeted models. Basically, adversarial attacks are initially extensively studied in the domain of computer vision, and some quickly expanded to other domains, including NLP, speech recognition and even malware detection. In this paper, we focus on malware with the file format of portable executable (PE) in the family of Windows operating systems, namely Windows PE malware, as a representative case to study the adversarial attack methods in such adversarial settings. To be specific, we start by first outlining the general learning framework of Windows PE malware detection based on ML/DL and subsequently highlighting three unique challenges of performing adversarial attacks in the context of PE malware. We then conduct a comprehensive and systematic review to categorize the state-of-the-art adversarial attacks against PE malware detection, as well as corresponding defenses to increase the robustness of PE malware detection. We conclude the paper by first presenting other related attacks against Windows PE malware detection beyond the adversarial attacks and then shedding light on future research directions and opportunities.



## **16. Understanding and Measuring Robustness of Multimodal Learning**

cs.LG

**SubmitDate**: 2021-12-22    [paper-pdf](http://arxiv.org/pdf/2112.12792v1)

**Authors**: Nishant Vishwamitra, Hongxin Hu, Ziming Zhao, Long Cheng, Feng Luo

**Abstracts**: The modern digital world is increasingly becoming multimodal. Although multimodal learning has recently revolutionized the state-of-the-art performance in multimodal tasks, relatively little is known about the robustness of multimodal learning in an adversarial setting. In this paper, we introduce a comprehensive measurement of the adversarial robustness of multimodal learning by focusing on the fusion of input modalities in multimodal models, via a framework called MUROAN (MUltimodal RObustness ANalyzer). We first present a unified view of multimodal models in MUROAN and identify the fusion mechanism of multimodal models as a key vulnerability. We then introduce a new type of multimodal adversarial attacks called decoupling attack in MUROAN that aims to compromise multimodal models by decoupling their fused modalities. We leverage the decoupling attack of MUROAN to measure several state-of-the-art multimodal models and find that the multimodal fusion mechanism in all these models is vulnerable to decoupling attacks. We especially demonstrate that, in the worst case, the decoupling attack of MUROAN achieves an attack success rate of 100% by decoupling just 1.16% of the input space. Finally, we show that traditional adversarial training is insufficient to improve the robustness of multimodal models with respect to decoupling attacks. We hope our findings encourage researchers to pursue improving the robustness of multimodal learning.



## **17. Detect & Reject for Transferability of Black-box Adversarial Attacks Against Network Intrusion Detection Systems**

cs.CR

**SubmitDate**: 2021-12-22    [paper-pdf](http://arxiv.org/pdf/2112.12095v1)

**Authors**: Islam Debicha, Thibault Debatty, Jean-Michel Dricot, Wim Mees, Tayeb Kenaza

**Abstracts**: In the last decade, the use of Machine Learning techniques in anomaly-based intrusion detection systems has seen much success. However, recent studies have shown that Machine learning in general and deep learning specifically are vulnerable to adversarial attacks where the attacker attempts to fool models by supplying deceptive input. Research in computer vision, where this vulnerability was first discovered, has shown that adversarial images designed to fool a specific model can deceive other machine learning models. In this paper, we investigate the transferability of adversarial network traffic against multiple machine learning-based intrusion detection systems. Furthermore, we analyze the robustness of the ensemble intrusion detection system, which is notorious for its better accuracy compared to a single model, against the transferability of adversarial attacks. Finally, we examine Detect & Reject as a defensive mechanism to limit the effect of the transferability property of adversarial network traffic against machine learning-based intrusion detection systems.



## **18. Evaluating the Robustness of Deep Reinforcement Learning for Autonomous and Adversarial Policies in a Multi-agent Urban Driving Environment**

cs.AI

**SubmitDate**: 2021-12-22    [paper-pdf](http://arxiv.org/pdf/2112.11947v1)

**Authors**: Aizaz Sharif, Dusica Marijan

**Abstracts**: Deep reinforcement learning is actively used for training autonomous driving agents in a vision-based urban simulated environment. Due to the large availability of various reinforcement learning algorithms, we are still unsure of which one works better while training autonomous cars in single-agent as well as multi-agent driving environments. A comparison of deep reinforcement learning in vision-based autonomous driving will open up the possibilities for training better autonomous car policies. Also, autonomous cars trained on deep reinforcement learning-based algorithms are known for being vulnerable to adversarial attacks, and we have less information on which algorithms would act as a good adversarial agent. In this work, we provide a systematic evaluation and comparative analysis of 6 deep reinforcement learning algorithms for autonomous and adversarial driving in four-way intersection scenario. Specifically, we first train autonomous cars using state-of-the-art deep reinforcement learning algorithms. Second, we test driving capabilities of the trained autonomous policies in single-agent as well as multi-agent scenarios. Lastly, we use the same deep reinforcement learning algorithms to train adversarial driving agents, in order to test the driving performance of autonomous cars and look for possible collision and offroad driving scenarios. We perform experiments by using vision-only high fidelity urban driving simulated environments.



## **19. Adversarial Deep Reinforcement Learning for Trustworthy Autonomous Driving Policies**

cs.AI

**SubmitDate**: 2021-12-22    [paper-pdf](http://arxiv.org/pdf/2112.11937v1)

**Authors**: Aizaz Sharif, Dusica Marijan

**Abstracts**: Deep reinforcement learning is widely used to train autonomous cars in a simulated environment. Still, autonomous cars are well known for being vulnerable when exposed to adversarial attacks. This raises the question of whether we can train the adversary as a driving agent for finding failure scenarios in autonomous cars, and then retrain autonomous cars with new adversarial inputs to improve their robustness. In this work, we first train and compare adversarial car policy on two custom reward functions to test the driving control decision of autonomous cars in a multi-agent setting. Second, we verify that adversarial examples can be used not only for finding unwanted autonomous driving behavior, but also for helping autonomous driving cars in improving their deep reinforcement learning policies. By using a high fidelity urban driving simulation environment and vision-based driving agents, we demonstrate that the autonomous cars retrained using the adversary player noticeably increase the performance of their driving policies in terms of reducing collision and offroad steering errors.



## **20. Consistency Regularization for Adversarial Robustness**

cs.LG

Published as a conference proceeding for AAAI 2022

**SubmitDate**: 2021-12-22    [paper-pdf](http://arxiv.org/pdf/2103.04623v3)

**Authors**: Jihoon Tack, Sihyun Yu, Jongheon Jeong, Minseon Kim, Sung Ju Hwang, Jinwoo Shin

**Abstracts**: Adversarial training (AT) is currently one of the most successful methods to obtain the adversarial robustness of deep neural networks. However, the phenomenon of robust overfitting, i.e., the robustness starts to decrease significantly during AT, has been problematic, not only making practitioners consider a bag of tricks for a successful training, e.g., early stopping, but also incurring a significant generalization gap in the robustness. In this paper, we propose an effective regularization technique that prevents robust overfitting by optimizing an auxiliary `consistency' regularization loss during AT. Specifically, we discover that data augmentation is a quite effective tool to mitigate the overfitting in AT, and develop a regularization that forces the predictive distributions after attacking from two different augmentations of the same instance to be similar with each other. Our experimental results demonstrate that such a simple regularization technique brings significant improvements in the test robust accuracy of a wide range of AT methods. More remarkably, we also show that our method could significantly help the model to generalize its robustness against unseen adversaries, e.g., other types or larger perturbations compared to those used during training. Code is available at https://github.com/alinlab/consistency-adversarial.



## **21. How Should Pre-Trained Language Models Be Fine-Tuned Towards Adversarial Robustness?**

cs.CL

Accepted by NeurIPS-2021

**SubmitDate**: 2021-12-22    [paper-pdf](http://arxiv.org/pdf/2112.11668v1)

**Authors**: Xinhsuai Dong, Luu Anh Tuan, Min Lin, Shuicheng Yan, Hanwang Zhang

**Abstracts**: The fine-tuning of pre-trained language models has a great success in many NLP fields. Yet, it is strikingly vulnerable to adversarial examples, e.g., word substitution attacks using only synonyms can easily fool a BERT-based sentiment analysis model. In this paper, we demonstrate that adversarial training, the prevalent defense technique, does not directly fit a conventional fine-tuning scenario, because it suffers severely from catastrophic forgetting: failing to retain the generic and robust linguistic features that have already been captured by the pre-trained model. In this light, we propose Robust Informative Fine-Tuning (RIFT), a novel adversarial fine-tuning method from an information-theoretical perspective. In particular, RIFT encourages an objective model to retain the features learned from the pre-trained model throughout the entire fine-tuning process, whereas a conventional one only uses the pre-trained weights for initialization. Experimental results show that RIFT consistently outperforms the state-of-the-arts on two popular NLP tasks: sentiment analysis and natural language inference, under different attacks across various pre-trained language models.



## **22. An Attention Score Based Attacker for Black-box NLP Classifier**

cs.LG

**SubmitDate**: 2021-12-22    [paper-pdf](http://arxiv.org/pdf/2112.11660v1)

**Authors**: Yueyang Liu, Hunmin Lee, Zhipeng Cai

**Abstracts**: Deep neural networks have a wide range of applications in solving various real-world tasks and have achieved satisfactory results, in domains such as computer vision, image classification, and natural language processing. Meanwhile, the security and robustness of neural networks have become imperative, as diverse researches have shown the vulnerable aspects of neural networks. Case in point, in Natural language processing tasks, the neural network may be fooled by an attentively modified text, which has a high similarity to the original one. As per previous research, most of the studies are focused on the image domain; Different from image adversarial attacks, the text is represented in a discrete sequence, traditional image attack methods are not applicable in the NLP field. In this paper, we propose a word-level NLP sentiment classifier attack model, which includes a self-attention mechanism-based word selection method and a greedy search algorithm for word substitution. We experiment with our attack model by attacking GRU and 1D-CNN victim models on IMDB datasets. Experimental results demonstrate that our model achieves a higher attack success rate and more efficient than previous methods due to the efficient word selection algorithms are employed and minimized the word substitute number. Also, our model is transferable, which can be used in the image domain with several modifications.



## **23. Collaborative adversary nodes learning on the logs of IoT devices in an IoT network**

cs.CR

**SubmitDate**: 2021-12-22    [paper-pdf](http://arxiv.org/pdf/2112.12546v1)

**Authors**: Sandhya Aneja, Melanie Ang Xuan En, Nagender Aneja

**Abstracts**: Artificial Intelligence (AI) development has encouraged many new research areas, including AI-enabled Internet of Things (IoT) network. AI analytics and intelligent paradigms greatly improve learning efficiency and accuracy. Applying these learning paradigms to network scenarios provide technical advantages of new networking solutions. In this paper, we propose an improved approach for IoT security from data perspective. The network traffic of IoT devices can be analyzed using AI techniques. The Adversary Learning (AdLIoTLog) model is proposed using Recurrent Neural Network (RNN) with attention mechanism on sequences of network events in the network traffic. We define network events as a sequence of the time series packets of protocols captured in the log. We have considered different packets TCP packets, UDP packets, and HTTP packets in the network log to make the algorithm robust. The distributed IoT devices can collaborate to cripple our world which is extending to Internet of Intelligence. The time series packets are converted into structured data by removing noise and adding timestamps. The resulting data set is trained by RNN and can detect the node pairs collaborating with each other. We used the BLEU score to evaluate the model performance. Our results show that the predicting performance of the AdLIoTLog model trained by our method degrades by 3-4% in the presence of attack in comparison to the scenario when the network is not under attack. AdLIoTLog can detect adversaries because when adversaries are present the model gets duped by the collaborative events and therefore predicts the next event with a biased event rather than a benign event. We conclude that AI can provision ubiquitous learning for the new generation of Internet of Things.



## **24. Reevaluating Adversarial Examples in Natural Language**

cs.CL

15 pages; 9 Tables; 5 Figures

**SubmitDate**: 2021-12-21    [paper-pdf](http://arxiv.org/pdf/2004.14174v3)

**Authors**: John X. Morris, Eli Lifland, Jack Lanchantin, Yangfeng Ji, Yanjun Qi

**Abstracts**: State-of-the-art attacks on NLP models lack a shared definition of a what constitutes a successful attack. We distill ideas from past work into a unified framework: a successful natural language adversarial example is a perturbation that fools the model and follows some linguistic constraints. We then analyze the outputs of two state-of-the-art synonym substitution attacks. We find that their perturbations often do not preserve semantics, and 38% introduce grammatical errors. Human surveys reveal that to successfully preserve semantics, we need to significantly increase the minimum cosine similarities between the embeddings of swapped words and between the sentence encodings of original and perturbed sentences.With constraints adjusted to better preserve semantics and grammaticality, the attack success rate drops by over 70 percentage points.



## **25. MIA-Former: Efficient and Robust Vision Transformers via Multi-grained Input-Adaptation**

cs.CV

**SubmitDate**: 2021-12-21    [paper-pdf](http://arxiv.org/pdf/2112.11542v1)

**Authors**: Zhongzhi Yu, Yonggan Fu, Sicheng Li, Chaojian Li, Yingyan Lin

**Abstracts**: ViTs are often too computationally expensive to be fitted onto real-world resource-constrained devices, due to (1) their quadratically increased complexity with the number of input tokens and (2) their overparameterized self-attention heads and model depth. In parallel, different images are of varied complexity and their different regions can contain various levels of visual information, indicating that treating all regions/tokens equally in terms of model complexity is unnecessary while such opportunities for trimming down ViTs' complexity have not been fully explored. To this end, we propose a Multi-grained Input-adaptive Vision Transformer framework dubbed MIA-Former that can input-adaptively adjust the structure of ViTs at three coarse-to-fine-grained granularities (i.e., model depth and the number of model heads/tokens). In particular, our MIA-Former adopts a low-cost network trained with a hybrid supervised and reinforcement training method to skip unnecessary layers, heads, and tokens in an input adaptive manner, reducing the overall computational cost. Furthermore, an interesting side effect of our MIA-Former is that its resulting ViTs are naturally equipped with improved robustness against adversarial attacks over their static counterparts, because MIA-Former's multi-grained dynamic control improves the model diversity similar to the effect of ensemble and thus increases the difficulty of adversarial attacks against all its sub-models. Extensive experiments and ablation studies validate that the proposed MIA-Former framework can effectively allocate computation budgets adaptive to the difficulty of input images meanwhile increase robustness, achieving state-of-the-art (SOTA) accuracy-efficiency trade-offs, e.g., 20% computation savings with the same or even a higher accuracy compared with SOTA dynamic transformer models.



## **26. Improving Robustness with Image Filtering**

cs.CV

**SubmitDate**: 2021-12-21    [paper-pdf](http://arxiv.org/pdf/2112.11235v1)

**Authors**: Matteo Terzi, Mattia Carletti, Gian Antonio Susto

**Abstracts**: Adversarial robustness is one of the most challenging problems in Deep Learning and Computer Vision research. All the state-of-the-art techniques require a time-consuming procedure that creates cleverly perturbed images. Due to its cost, many solutions have been proposed to avoid Adversarial Training. However, all these attempts proved ineffective as the attacker manages to exploit spurious correlations among pixels to trigger brittle features implicitly learned by the model. This paper first introduces a new image filtering scheme called Image-Graph Extractor (IGE) that extracts the fundamental nodes of an image and their connections through a graph structure. By leveraging the IGE representation, we build a new defense method, Filtering As a Defense, that does not allow the attacker to entangle pixels to create malicious patterns. Moreover, we show that data augmentation with filtered images effectively improves the model's robustness to data corruption. We validate our techniques on CIFAR-10, CIFAR-100, and ImageNet.



## **27. Adversarial images for the primate brain**

q-bio.NC

These results reveal limits of CNN-based models of primate vision  through their differential response to adversarial attack, and provide clues  for building better models of the brain and more robust computer vision  algorithms

**SubmitDate**: 2021-12-21    [paper-pdf](http://arxiv.org/pdf/2011.05623v2)

**Authors**: Li Yuan, Will Xiao, Gabriel Kreiman, Francis E. H. Tay, Jiashi Feng, Margaret S. Livingstone

**Abstracts**: Convolutional neural networks (CNNs) are vulnerable to adversarial attack, the phenomenon that adding minuscule noise to an image can fool CNNs into misclassifying it. Because this noise is nearly imperceptible to human viewers, biological vision is assumed to be robust to adversarial attack. Despite this apparent difference in robustness, CNNs are currently the best models of biological vision, revealing a gap in explaining how the brain responds to adversarial images. Indeed, sensitivity to adversarial attack has not been measured for biological vision under normal conditions, nor have attack methods been specifically designed to affect biological vision. We studied the effects of adversarial attack on primate vision, measuring both monkey neuronal responses and human behavior. Adversarial images were created by modifying images from one category(such as human faces) to look like a target category(such as monkey faces), while limiting pixel value change. We tested three attack directions via several attack methods, including directly using CNN adversarial images and using a CNN-based predictive model to guide monkey visual neuron responses. We considered a wide range of image change magnitudes that covered attack success rates up to>90%. We found that adversarial images designed for CNNs were ineffective in attacking primate vision. Even when considering the best attack method, primate vision was more robust to adversarial attack than an ensemble of CNNs, requiring over 100-fold larger image change to attack successfully. The success of individual attack methods and images was correlated between monkey neurons and human behavior, but was less correlated between either and CNN categorization. Consistently, CNN-based models of neurons, when trained on natural images, did not generalize to explain neuronal responses to adversarial images.



## **28. A Theoretical View of Linear Backpropagation and Its Convergence**

cs.LG

**SubmitDate**: 2021-12-21    [paper-pdf](http://arxiv.org/pdf/2112.11018v1)

**Authors**: Ziang Li, Yiwen Guo, Haodi Liu, Changshui Zhang

**Abstracts**: Backpropagation is widely used for calculating gradients in deep neural networks (DNNs). Applied often along with stochastic gradient descent (SGD) or its variants, backpropagation is considered as a de-facto choice in a variety of machine learning tasks including DNN training and adversarial attack/defense. Recently, a linear variant of BP named LinBP was introduced for generating more transferable adversarial examples for black-box adversarial attacks, by Guo et al. Yet, it has not been theoretically studied and the convergence analysis of such a method is lacking. This paper serves as a complement and somewhat an extension to Guo et al.'s paper, by providing theoretical analyses on LinBP in neural-network-involved learning tasks including adversarial attack and model training. We demonstrate that, somewhat surprisingly, LinBP can lead to faster convergence in these tasks in the same hyper-parameter settings, compared to BP. We confirm our theoretical results with extensive experiments.



## **29. What are Attackers after on IoT Devices? An approach based on a multi-phased multi-faceted IoT honeypot ecosystem and data clustering**

cs.CR

arXiv admin note: text overlap with arXiv:2003.01218

**SubmitDate**: 2021-12-21    [paper-pdf](http://arxiv.org/pdf/2112.10974v1)

**Authors**: Armin Ziaie Tabari, Xinming Ou, Anoop Singhal

**Abstracts**: The growing number of Internet of Things (IoT) devices makes it imperative to be aware of the real-world threats they face in terms of cybersecurity. While honeypots have been historically used as decoy devices to help researchers/organizations gain a better understanding of the dynamic of threats on a network and their impact, IoT devices pose a unique challenge for this purpose due to the variety of devices and their physical connections. In this work, by observing real-world attackers' behavior in a low-interaction honeypot ecosystem, we (1) presented a new approach to creating a multi-phased, multi-faceted honeypot ecosystem, which gradually increases the sophistication of honeypots' interactions with adversaries, (2) designed and developed a low-interaction honeypot for cameras that allowed researchers to gain a deeper understanding of what attackers are targeting, and (3) devised an innovative data analytics method to identify the goals of adversaries. Our honeypots have been active for over three years. We were able to collect increasingly sophisticated attack data in each phase. Furthermore, our data analytics points to the fact that the vast majority of attack activities captured in the honeypots share significant similarity, and can be clustered and grouped to better understand the goals, patterns, and trends of IoT attacks in the wild.



## **30. Channel-Aware Adversarial Attacks Against Deep Learning-Based Wireless Signal Classifiers**

eess.SP

Submitted for publication. arXiv admin note: substantial text overlap  with arXiv:2002.02400

**SubmitDate**: 2021-12-20    [paper-pdf](http://arxiv.org/pdf/2005.05321v3)

**Authors**: Brian Kim, Yalin E. Sagduyu, Kemal Davaslioglu, Tugba Erpek, Sennur Ulukus

**Abstracts**: This paper presents channel-aware adversarial attacks against deep learning-based wireless signal classifiers. There is a transmitter that transmits signals with different modulation types. A deep neural network is used at each receiver to classify its over-the-air received signals to modulation types. In the meantime, an adversary transmits an adversarial perturbation (subject to a power budget) to fool receivers into making errors in classifying signals that are received as superpositions of transmitted signals and adversarial perturbations. First, these evasion attacks are shown to fail when channels are not considered in designing adversarial perturbations. Then, realistic attacks are presented by considering channel effects from the adversary to each receiver. After showing that a channel-aware attack is selective (i.e., it affects only the receiver whose channel is considered in the perturbation design), a broadcast adversarial attack is presented by crafting a common adversarial perturbation to simultaneously fool classifiers at different receivers. The major vulnerability of modulation classifiers to over-the-air adversarial attacks is shown by accounting for different levels of information available about the channel, the transmitter input, and the classifier model. Finally, a certified defense based on randomized smoothing that augments training data with noise is introduced to make the modulation classifier robust to adversarial perturbations.



## **31. An Evasion Attack against Stacked Capsule Autoencoder**

cs.LG

**SubmitDate**: 2021-12-20    [paper-pdf](http://arxiv.org/pdf/2010.07230v5)

**Authors**: Jiazhu Dai, Siwei Xiong

**Abstracts**: Capsule network is a type of neural network that uses the spatial relationship between features to classify images. By capturing the poses and relative positions between features, its ability to recognize affine transformation is improved, and it surpasses traditional convolutional neural networks (CNNs) when handling translation, rotation and scaling. The Stacked Capsule Autoencoder (SCAE) is the state-of-the-art capsule network. The SCAE encodes an image as capsules, each of which contains poses of features and their correlations. The encoded contents are then input into the downstream classifier to predict the categories of the images. Existing research mainly focuses on the security of capsule networks with dynamic routing or EM routing, and little attention has been given to the security and robustness of the SCAE. In this paper, we propose an evasion attack against the SCAE. After a perturbation is generated based on the output of the object capsules in the model, it is added to an image to reduce the contribution of the object capsules related to the original category of the image so that the perturbed image will be misclassified. We evaluate the attack using an image classification experiment, and the experimental results indicate that the attack can achieve high success rates and stealthiness. It confirms that the SCAE has a security vulnerability whereby it is possible to craft adversarial samples without changing the original structure of the image to fool the classifiers. We hope that our work will make the community aware of the threat of this attack and raise the attention given to the SCAE's security.



## **32. Adversarial Attacks on Spiking Convolutional Networks for Event-based Vision**

cs.CV

16 pages, preprint, submitted to ICLR 2022

**SubmitDate**: 2021-12-20    [paper-pdf](http://arxiv.org/pdf/2110.02929v2)

**Authors**: Julian Büchel, Gregor Lenz, Yalun Hu, Sadique Sheik, Martino Sorbaro

**Abstracts**: Event-based sensing using dynamic vision sensors is gaining traction in low-power vision applications. Spiking neural networks work well with the sparse nature of event-based data and suit deployment on low-power neuromorphic hardware. Being a nascent field, the sensitivity of spiking neural networks to potentially malicious adversarial attacks has received very little attention so far. In this work, we show how white-box adversarial attack algorithms can be adapted to the discrete and sparse nature of event-based visual data, and to the continuous-time setting of spiking neural networks. We test our methods on the N-MNIST and IBM Gestures neuromorphic vision datasets and show adversarial perturbations achieve a high success rate, by injecting a relatively small number of appropriately placed events. We also verify, for the first time, the effectiveness of these perturbations directly on neuromorphic hardware. Finally, we discuss the properties of the resulting perturbations and possible future directions.



## **33. Certified Federated Adversarial Training**

cs.LG

First presented at the 1st NeurIPS Workshop on New Frontiers in  Federated Learning (NFFL 2021)

**SubmitDate**: 2021-12-20    [paper-pdf](http://arxiv.org/pdf/2112.10525v1)

**Authors**: Giulio Zizzo, Ambrish Rawat, Mathieu Sinn, Sergio Maffeis, Chris Hankin

**Abstracts**: In federated learning (FL), robust aggregation schemes have been developed to protect against malicious clients. Many robust aggregation schemes rely on certain numbers of benign clients being present in a quorum of workers. This can be hard to guarantee when clients can join at will, or join based on factors such as idle system status, and connected to power and WiFi. We tackle the scenario of securing FL systems conducting adversarial training when a quorum of workers could be completely malicious. We model an attacker who poisons the model to insert a weakness into the adversarial training such that the model displays apparent adversarial robustness, while the attacker can exploit the inserted weakness to bypass the adversarial training and force the model to misclassify adversarial examples. We use abstract interpretation techniques to detect such stealthy attacks and block the corrupted model updates. We show that this defence can preserve adversarial robustness even against an adaptive attacker.



## **34. Unifying Model Explainability and Robustness for Joint Text Classification and Rationale Extraction**

cs.CL

AAAI 2022

**SubmitDate**: 2021-12-20    [paper-pdf](http://arxiv.org/pdf/2112.10424v1)

**Authors**: Dongfang Li, Baotian Hu, Qingcai Chen, Tujie Xu, Jingcong Tao, Yunan Zhang

**Abstracts**: Recent works have shown explainability and robustness are two crucial ingredients of trustworthy and reliable text classification. However, previous works usually address one of two aspects: i) how to extract accurate rationales for explainability while being beneficial to prediction; ii) how to make the predictive model robust to different types of adversarial attacks. Intuitively, a model that produces helpful explanations should be more robust against adversarial attacks, because we cannot trust the model that outputs explanations but changes its prediction under small perturbations. To this end, we propose a joint classification and rationale extraction model named AT-BMC. It includes two key mechanisms: mixed Adversarial Training (AT) is designed to use various perturbations in discrete and embedding space to improve the model's robustness, and Boundary Match Constraint (BMC) helps to locate rationales more precisely with the guidance of boundary information. Performances on benchmark datasets demonstrate that the proposed AT-BMC outperforms baselines on both classification and rationale extraction by a large margin. Robustness analysis shows that the proposed AT-BMC decreases the attack success rate effectively by up to 69%. The empirical results indicate that there are connections between robust models and better explanations.



## **35. Energy-bounded Learning for Robust Models of Code**

cs.LG

arXiv admin note: text overlap with arXiv:2010.03759 by other authors

**SubmitDate**: 2021-12-20    [paper-pdf](http://arxiv.org/pdf/2112.11226v1)

**Authors**: Nghi D. Q. Bui, Yijun Yu

**Abstracts**: In programming, learning code representations has a variety of applications, including code classification, code search, comment generation, bug prediction, and so on. Various representations of code in terms of tokens, syntax trees, dependency graphs, code navigation paths, or a combination of their variants have been proposed, however, existing vanilla learning techniques have a major limitation in robustness, i.e., it is easy for the models to make incorrect predictions when the inputs are altered in a subtle way. To enhance the robustness, existing approaches focus on recognizing adversarial samples rather than on the valid samples that fall outside a given distribution, which we refer to as out-of-distribution (OOD) samples. Recognizing such OOD samples is the novel problem investigated in this paper. To this end, we propose to first augment the in=distribution datasets with out-of-distribution samples such that, when trained together, they will enhance the model's robustness. We propose the use of an energy-bounded learning objective function to assign a higher score to in-distribution samples and a lower score to out-of-distribution samples in order to incorporate such out-of-distribution samples into the training process of source code models. In terms of OOD detection and adversarial samples detection, our evaluation results demonstrate a greater robustness for existing source code models to become more accurate at recognizing OOD data while being more resistant to adversarial attacks at the same time. Furthermore, the proposed energy-bounded score outperforms all existing OOD detection scores by a large margin, including the softmax confidence score, the Mahalanobis score, and ODIN.



## **36. Knowledge Cross-Distillation for Membership Privacy**

cs.CR

Under Review

**SubmitDate**: 2021-12-20    [paper-pdf](http://arxiv.org/pdf/2111.01363v2)

**Authors**: Rishav Chourasia, Batnyam Enkhtaivan, Kunihiro Ito, Junki Mori, Isamu Teranishi, Hikaru Tsuchida

**Abstracts**: A membership inference attack (MIA) poses privacy risks on the training data of a machine learning model. With an MIA, an attacker guesses if the target data are a member of the training dataset. The state-of-the-art defense against MIAs, distillation for membership privacy (DMP), requires not only private data to protect but a large amount of unlabeled public data. However, in certain privacy-sensitive domains, such as medical and financial, the availability of public data is not obvious. Moreover, a trivial method to generate the public data by using generative adversarial networks significantly decreases the model accuracy, as reported by the authors of DMP. To overcome this problem, we propose a novel defense against MIAs using knowledge distillation without requiring public data. Our experiments show that the privacy protection and accuracy of our defense are comparable with those of DMP for the benchmark tabular datasets used in MIA researches, Purchase100 and Texas100, and our defense has much better privacy-utility trade-off than those of the existing defenses without using public data for image dataset CIFAR10.



## **37. Toward Evaluating Re-identification Risks in the Local Privacy Model**

cs.CR

Accepted at Transactions on Data Privacy

**SubmitDate**: 2021-12-19    [paper-pdf](http://arxiv.org/pdf/2010.08238v5)

**Authors**: Takao Murakami, Kenta Takahashi

**Abstracts**: LDP (Local Differential Privacy) has recently attracted much attention as a metric of data privacy that prevents the inference of personal data from obfuscated data in the local model. However, there are scenarios in which the adversary wants to perform re-identification attacks to link the obfuscated data to users in this model. LDP can cause excessive obfuscation and destroy the utility in these scenarios because it is not designed to directly prevent re-identification. In this paper, we propose a measure of re-identification risks, which we call PIE (Personal Information Entropy). The PIE is designed so that it directly prevents re-identification attacks in the local model. It lower-bounds the lowest possible re-identification error probability (i.e., Bayes error probability) of the adversary. We analyze the relation between LDP and the PIE, and analyze the PIE and utility in distribution estimation for two obfuscation mechanisms providing LDP. Through experiments, we show that when we consider re-identification as a privacy risk, LDP can cause excessive obfuscation and destroy the utility. Then we show that the PIE can be used to guarantee low re-identification risks for the local obfuscation mechanisms while keeping high utility.



## **38. Jamming Pattern Recognition over Multi-Channel Networks: A Deep Learning Approach**

cs.CR

**SubmitDate**: 2021-12-19    [paper-pdf](http://arxiv.org/pdf/2112.11222v1)

**Authors**: Ali Pourranjbar, Georges Kaddoum, Walid Saad

**Abstracts**: With the advent of intelligent jammers, jamming attacks have become a more severe threat to the performance of wireless systems. An intelligent jammer is able to change its policy to minimize the probability of being traced by legitimate nodes. Thus, an anti-jamming mechanism capable of constantly adjusting to the jamming policy is required to combat such a jammer. Remarkably, existing anti-jamming methods are not applicable here because they mainly focus on mitigating jamming attacks with an invariant jamming policy, and they rarely consider an intelligent jammer as an adversary. Therefore, in this paper, to employ a jamming type recognition technique working alongside an anti-jamming technique is proposed. The proposed recognition method employs a recurrent neural network that takes the jammer's occupied channels as inputs and outputs the jammer type. Under this scheme, the real-time jammer policy is first identified, and, then, the most appropriate countermeasure is chosen. Consequently, any changes to the jammer policy can be instantly detected with the proposed recognition technique allowing for a rapid switch to a new anti-jamming method fitted to the new jamming policy. To evaluate the performance of the proposed recognition method, the accuracy of the detection is derived as a function of the jammer policy switching time. Simulation results show the detection accuracy for all the considered users numbers is greater than 70% when the jammer switches its policy every 5 time slots and the accuracy raises to 90% when the jammer policy switching time is 45.



## **39. Attacking Point Cloud Segmentation with Color-only Perturbation**

cs.CV

**SubmitDate**: 2021-12-18    [paper-pdf](http://arxiv.org/pdf/2112.05871v2)

**Authors**: Jiacen Xu, Zhe Zhou, Boyuan Feng, Yufei Ding, Zhou Li

**Abstracts**: Recent research efforts on 3D point-cloud semantic segmentation have achieved outstanding performance by adopting deep CNN (convolutional neural networks) and GCN (graph convolutional networks). However, the robustness of these complex models has not been systematically analyzed. Given that semantic segmentation has been applied in many safety-critical applications (e.g., autonomous driving, geological sensing), it is important to fill this knowledge gap, in particular, how these models are affected under adversarial samples. While adversarial attacks against point cloud have been studied, we found all of them were targeting single-object recognition, and the perturbation is done on the point coordinates. We argue that the coordinate-based perturbation is unlikely to realize under the physical-world constraints. Hence, we propose a new color-only perturbation method named COLPER, and tailor it to semantic segmentation. By evaluating COLPER on an indoor dataset (S3DIS) and an outdoor dataset (Semantic3D) against three point cloud segmentation models (PointNet++, DeepGCNs, and RandLA-Net), we found color-only perturbation is sufficient to significantly drop the segmentation accuracy and aIoU, under both targeted and non-targeted attack settings.



## **40. Adversarial Attack for Uncertainty Estimation: Identifying Critical Regions in Neural Networks**

cs.LG

15 pages, 6 figures, Neural Process Lett (2021)

**SubmitDate**: 2021-12-18    [paper-pdf](http://arxiv.org/pdf/2107.07618v2)

**Authors**: Ismail Alarab, Simant Prakoonwit

**Abstracts**: We propose a novel method to capture data points near decision boundary in neural network that are often referred to a specific type of uncertainty. In our approach, we sought to perform uncertainty estimation based on the idea of adversarial attack method. In this paper, uncertainty estimates are derived from the input perturbations, unlike previous studies that provide perturbations on the model's parameters as in Bayesian approach. We are able to produce uncertainty with couple of perturbations on the inputs. Interestingly, we apply the proposed method to datasets derived from blockchain. We compare the performance of model uncertainty with the most recent uncertainty methods. We show that the proposed method has revealed a significant outperformance over other methods and provided less risk to capture model uncertainty in machine learning.



## **41. Dynamic Defender-Attacker Blotto Game**

eess.SY

**SubmitDate**: 2021-12-18    [paper-pdf](http://arxiv.org/pdf/2112.09890v1)

**Authors**: Daigo Shishika, Yue Guan, Michael Dorothy, Vijay Kumar

**Abstracts**: This work studies a dynamic, adversarial resource allocation problem in environments modeled as graphs. A blue team of defender robots are deployed in the environment to protect the nodes from a red team of attacker robots. We formulate the engagement as a discrete-time dynamic game, where the robots can move at most one hop in each time step. The game terminates with the attacker's win if any location has more attacker robots than defender robots at any time. The goal is to identify dynamic resource allocation strategies, as well as the conditions that determines the winner: graph structure, available resources, and initial conditions. We analyze the problem using reachable sets and show how the outdegree of the underlying graph directly influences the difficulty of the defending task. Furthermore, we provide algorithms that identify sufficiency of attacker's victory.



## **42. Formalizing Generalization and Robustness of Neural Networks to Weight Perturbations**

cs.LG

This version has been accepted for poster presentation at NeurIPS  2021

**SubmitDate**: 2021-12-17    [paper-pdf](http://arxiv.org/pdf/2103.02200v2)

**Authors**: Yu-Lin Tsai, Chia-Yi Hsu, Chia-Mu Yu, Pin-Yu Chen

**Abstracts**: Studying the sensitivity of weight perturbation in neural networks and its impacts on model performance, including generalization and robustness, is an active research topic due to its implications on a wide range of machine learning tasks such as model compression, generalization gap assessment, and adversarial attacks. In this paper, we provide the first integral study and analysis for feed-forward neural networks in terms of the robustness in pairwise class margin and its generalization behavior under weight perturbation. We further design a new theory-driven loss function for training generalizable and robust neural networks against weight perturbations. Empirical experiments are conducted to validate our theoretical analysis. Our results offer fundamental insights for characterizing the generalization and robustness of neural networks against weight perturbations.



## **43. Reasoning Chain Based Adversarial Attack for Multi-hop Question Answering**

cs.CL

10 pages including reference, 4 figures

**SubmitDate**: 2021-12-17    [paper-pdf](http://arxiv.org/pdf/2112.09658v1)

**Authors**: Jiayu Ding, Siyuan Wang, Qin Chen, Zhongyu Wei

**Abstracts**: Recent years have witnessed impressive advances in challenging multi-hop QA tasks. However, these QA models may fail when faced with some disturbance in the input text and their interpretability for conducting multi-hop reasoning remains uncertain. Previous adversarial attack works usually edit the whole question sentence, which has limited effect on testing the entity-based multi-hop inference ability. In this paper, we propose a multi-hop reasoning chain based adversarial attack method. We formulate the multi-hop reasoning chains starting from the query entity to the answer entity in the constructed graph, which allows us to align the question to each reasoning hop and thus attack any hop. We categorize the questions into different reasoning types and adversarially modify part of the question corresponding to the selected reasoning hop to generate the distracting sentence. We test our adversarial scheme on three QA models on HotpotQA dataset. The results demonstrate significant performance reduction on both answer and supporting facts prediction, verifying the effectiveness of our reasoning chain based attack method for multi-hop reasoning models and the vulnerability of them. Our adversarial re-training further improves the performance and robustness of these models.



## **44. Who Is the Strongest Enemy? Towards Optimal and Efficient Evasion Attacks in Deep RL**

cs.LG

**SubmitDate**: 2021-12-17    [paper-pdf](http://arxiv.org/pdf/2106.05087v2)

**Authors**: Yanchao Sun, Ruijie Zheng, Yongyuan Liang, Furong Huang

**Abstracts**: Evaluating the worst-case performance of a reinforcement learning (RL) agent under the strongest/optimal adversarial perturbations on state observations (within some constraints) is crucial for understanding the robustness of RL agents. However, finding the optimal adversary is challenging, in terms of both whether we can find the optimal attack and how efficiently we can find it. Existing works on adversarial RL either use heuristics-based methods that may not find the strongest adversary, or directly train an RL-based adversary by treating the agent as a part of the environment, which can find the optimal adversary but may become intractable in a large state space. This paper introduces a novel attacking method to find the optimal attacks through collaboration between a designed function named ''actor'' and an RL-based learner named "director". The actor crafts state perturbations for a given policy perturbation direction, and the director learns to propose the best policy perturbation directions. Our proposed algorithm, PA-AD, is theoretically optimal and significantly more efficient than prior RL-based works in environments with large state spaces. Empirical results show that our proposed PA-AD universally outperforms state-of-the-art attacking methods in various Atari and MuJoCo environments. By applying PA-AD to adversarial training, we achieve state-of-the-art empirical robustness in multiple tasks under strong adversaries.



## **45. Dynamics-aware Adversarial Attack of 3D Sparse Convolution Network**

cs.CV

**SubmitDate**: 2021-12-17    [paper-pdf](http://arxiv.org/pdf/2112.09428v1)

**Authors**: An Tao, Yueqi Duan, He Wang, Ziyi Wu, Pengliang Ji, Haowen Sun, Jie Zhou, Jiwen Lu

**Abstracts**: In this paper, we investigate the dynamics-aware adversarial attack problem in deep neural networks. Most existing adversarial attack algorithms are designed under a basic assumption -- the network architecture is fixed throughout the attack process. However, this assumption does not hold for many recently proposed networks, e.g. 3D sparse convolution network, which contains input-dependent execution to improve computational efficiency. It results in a serious issue of lagged gradient, making the learned attack at the current step ineffective due to the architecture changes afterward. To address this issue, we propose a Leaded Gradient Method (LGM) and show the significant effects of the lagged gradient. More specifically, we re-formulate the gradients to be aware of the potential dynamic changes of network architectures, so that the learned attack better "leads" the next step than the dynamics-unaware methods when network architecture changes dynamically. Extensive experiments on various datasets show that our LGM achieves impressive performance on semantic segmentation and classification. Compared with the dynamic-unaware methods, LGM achieves about 20% lower mIoU averagely on the ScanNet and S3DIS datasets. LGM also outperforms the recent point cloud attacks.



## **46. APTSHIELD: A Stable, Efficient and Real-time APT Detection System for Linux Hosts**

cs.CR

**SubmitDate**: 2021-12-17    [paper-pdf](http://arxiv.org/pdf/2112.09008v2)

**Authors**: Tiantian Zhu, Jinkai Yu, Tieming Chen, Jiayu Wang, Jie Ying, Ye Tian, Mingqi Lv, Yan Chen, Yuan Fan, Ting Wang

**Abstracts**: Advanced Persistent Threat (APT) attack usually refers to the form of long-term, covert and sustained attack on specific targets, with an adversary using advanced attack techniques to destroy the key facilities of an organization. APT attacks have caused serious security threats and massive financial loss worldwide. Academics and industry thereby have proposed a series of solutions to detect APT attacks, such as dynamic/static code analysis, traffic detection, sandbox technology, endpoint detection and response (EDR), etc. However, existing defenses are failed to accurately and effectively defend against the current APT attacks that exhibit strong persistent, stealthy, diverse and dynamic characteristics due to the weak data source integrity, large data processing overhead and poor real-time performance in the process of real-world scenarios.   To overcome these difficulties, in this paper we propose APTSHIELD, a stable, efficient and real-time APT detection system for Linux hosts. In the aspect of data collection, audit is selected to stably collect kernel data of the operating system so as to carry out a complete portrait of the attack based on comprehensive analysis and comparison of existing logging tools; In the aspect of data processing, redundant semantics skipping and non-viable node pruning are adopted to reduce the amount of data, so as to reduce the overhead of the detection system; In the aspect of attack detection, an APT attack detection framework based on ATT\&CK model is designed to carry out real-time attack response and alarm through the transfer and aggregation of labels. Experimental results on both laboratory and Darpa Engagement show that our system can effectively detect web vulnerability attacks, file-less attacks and remote access trojan attacks, and has a low false positive rate, which adds far more value than the existing frontier work.



## **47. Deep Bayesian Learning for Car Hacking Detection**

cs.CR

**SubmitDate**: 2021-12-17    [paper-pdf](http://arxiv.org/pdf/2112.09333v1)

**Authors**: Laha Ale, Scott A. King, Ning Zhang

**Abstracts**: With the rise of self-drive cars and connected vehicles, cars are equipped with various devices to assistant the drivers or support self-drive systems. Undoubtedly, cars have become more intelligent as we can deploy more and more devices and software on the cars. Accordingly, the security of assistant and self-drive systems in the cars becomes a life-threatening issue as smart cars can be invaded by malicious attacks that cause traffic accidents. Currently, canonical machine learning and deep learning methods are extensively employed in car hacking detection. However, machine learning and deep learning methods can easily be overconfident and defeated by carefully designed adversarial examples. Moreover, those methods cannot provide explanations for security engineers for further analysis. In this work, we investigated Deep Bayesian Learning models to detect and analyze car hacking behaviors. The Bayesian learning methods can capture the uncertainty of the data and avoid overconfident issues. Moreover, the Bayesian models can provide more information to support the prediction results that can help security engineers further identify the attacks. We have compared our model with deep learning models and the results show the advantages of our proposed model. The code of this work is publicly available



## **48. Generation of Wheel Lockup Attacks on Nonlinear Dynamics of Vehicle Traction**

eess.SY

Submitted to American Control Conference 2022 (ACC 2022), 6 pages

**SubmitDate**: 2021-12-16    [paper-pdf](http://arxiv.org/pdf/2112.09229v1)

**Authors**: Alireza Mohammadi, Hafiz Malik, Masoud Abbaszadeh

**Abstracts**: There is ample evidence in the automotive cybersecurity literature that the car brake ECUs can be maliciously reprogrammed. Motivated by such threat, this paper investigates the capabilities of an adversary who can directly control the frictional brake actuators and would like to induce wheel lockup conditions leading to catastrophic road injuries. This paper demonstrates that the adversary despite having a limited knowledge of the tire-road interaction characteristics has the capability of driving the states of the vehicle traction dynamics to a vicinity of the lockup manifold in a finite time by means of a properly designed attack policy for the frictional brakes. This attack policy relies on employing a predefined-time controller and a nonlinear disturbance observer acting on the wheel slip error dynamics. Simulations under various road conditions demonstrate the effectiveness of the proposed attack policy.



## **49. All You Need is RAW: Defending Against Adversarial Attacks with Camera Image Pipelines**

cs.CV

**SubmitDate**: 2021-12-16    [paper-pdf](http://arxiv.org/pdf/2112.09219v1)

**Authors**: Yuxuan Zhang, Bo Dong, Felix Heide

**Abstracts**: Existing neural networks for computer vision tasks are vulnerable to adversarial attacks: adding imperceptible perturbations to the input images can fool these methods to make a false prediction on an image that was correctly predicted without the perturbation. Various defense methods have proposed image-to-image mapping methods, either including these perturbations in the training process or removing them in a preprocessing denoising step. In doing so, existing methods often ignore that the natural RGB images in today's datasets are not captured but, in fact, recovered from RAW color filter array captures that are subject to various degradations in the capture. In this work, we exploit this RAW data distribution as an empirical prior for adversarial defense. Specifically, we proposed a model-agnostic adversarial defensive method, which maps the input RGB images to Bayer RAW space and back to output RGB using a learned camera image signal processing (ISP) pipeline to eliminate potential adversarial patterns. The proposed method acts as an off-the-shelf preprocessing module and, unlike model-specific adversarial training methods, does not require adversarial images to train. As a result, the method generalizes to unseen tasks without additional retraining. Experiments on large-scale datasets (e.g., ImageNet, COCO) for different vision tasks (e.g., classification, semantic segmentation, object detection) validate that the method significantly outperforms existing methods across task domains.



## **50. Direction-Aggregated Attack for Transferable Adversarial Examples**

cs.LG

ACM JETC JOURNAL Accepted

**SubmitDate**: 2021-12-16    [paper-pdf](http://arxiv.org/pdf/2104.09172v2)

**Authors**: Tianjin Huang, Vlado Menkovski, Yulong Pei, YuHao Wang, Mykola Pechenizkiy

**Abstracts**: Deep neural networks are vulnerable to adversarial examples that are crafted by imposing imperceptible changes to the inputs. However, these adversarial examples are most successful in white-box settings where the model and its parameters are available. Finding adversarial examples that are transferable to other models or developed in a black-box setting is significantly more difficult. In this paper, we propose the Direction-Aggregated adversarial attacks that deliver transferable adversarial examples. Our method utilizes aggregated direction during the attack process for avoiding the generated adversarial examples overfitting to the white-box model. Extensive experiments on ImageNet show that our proposed method improves the transferability of adversarial examples significantly and outperforms state-of-the-art attacks, especially against adversarial robust models. The best averaged attack success rates of our proposed method reaches 94.6\% against three adversarial trained models and 94.8\% against five defense methods. It also reveals that current defense approaches do not prevent transferable adversarial attacks.



